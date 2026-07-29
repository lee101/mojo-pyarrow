"""A zero-copy numeric subset of :mod:`pyarrow.compute` backed by Mojo."""

from __future__ import annotations

import math
import re
from dataclasses import replace

import numpy as np
import pyarrow as pa

from ._lib import Column, column, lib

_FLOAT = pa.float64()
_INT = pa.int64()
_BINARY_OPS = {"add": 0, "subtract": 1, "multiply": 2, "divide": 3, "power": 4}
_COMPARE_OPS = {
    "equal": 0,
    "not_equal": 1,
    "less": 2,
    "less_equal": 3,
    "greater": 4,
    "greater_equal": 5,
}
_UNARY_OPS = {
    "negate": 0,
    "abs": 1,
    "sqrt": 2,
    "exp": 3,
    "ln": 4,
    "log10": 5,
    "log2": 6,
    "sin": 7,
    "cos": 8,
    "tan": 9,
    "asin": 10,
    "acos": 11,
    "atan": 12,
}


def _is_arraylike(value) -> bool:
    return isinstance(value, (pa.Array, pa.ChunkedArray, np.ndarray, list, tuple))


def _require_numeric(col: Column) -> None:
    if col.array.type not in (_FLOAT, _INT):
        raise TypeError(
            f"mojo-pyarrow supports float64 and int64, got {col.array.type}"
        )


def _alloc(size: int, memory_pool=None) -> pa.Buffer:
    return pa.allocate_buffer(size, memory_pool=memory_pool, resizable=False)


def _values_buffer(length: int, dtype: np.dtype, memory_pool=None) -> pa.Buffer:
    return _alloc(length * dtype.itemsize, memory_pool)


def _bitmap_buffer(length: int, memory_pool=None) -> pa.Buffer:
    size = ((length + 7) // 8 + 63) & ~63 if length else 0
    buffer = _alloc(size, memory_pool)
    if size:
        np.frombuffer(buffer, dtype=np.uint8)[:] = 0
    return buffer


def _finish(
    type: pa.DataType,
    length: int,
    values: pa.Buffer,
    validity: pa.Buffer | None,
    *,
    scalar: bool = False,
):
    array = pa.Array.from_buffers(type, length, [validity, values])
    return array[0] if scalar else array


def _result_buffers(type: pa.DataType, length: int, memory_pool=None):
    dtype = np.dtype(np.float64 if type == _FLOAT else np.int64)
    return (
        _values_buffer(length, dtype, memory_pool),
        _bitmap_buffer(length, memory_pool),
    )


def _bool_buffers(length: int, memory_pool=None):
    return _bitmap_buffer(length, memory_pool), _bitmap_buffer(length, memory_pool)


def _column_from_array(array: pa.Array, step: int = 1) -> Column:
    return replace(column(array), step=step)


def _cast_to_float(col: Column, memory_pool=None) -> Column:
    if col.array.type == _FLOAT:
        return col
    _require_numeric(col)
    if col.step == 0:
        value = col.array[0]
        return column(
            None if not value.is_valid else float(value.as_py()),
            type=_FLOAT,
            scalar=True,
        )
    length = col.length
    values, validity = _result_buffers(_FLOAT, length, memory_pool)
    if length:
        lib().mpa_cast_i64_f64(
            col.values,
            col.bitmap,
            col.offset,
            length,
            values.address,
            validity.address,
        )
    array = _finish(_FLOAT, length, values, validity)
    return _column_from_array(array)


def _raw_column(value, other_type: pa.DataType | None = None) -> tuple[Column, bool]:
    scalar = not _is_arraylike(value)
    if scalar and (
        value is None or (isinstance(value, pa.Scalar) and not value.is_valid)
    ):
        if other_type is None:
            raise TypeError("cannot infer a type from an untyped null scalar")
        return column(value, type=other_type, scalar=True), True
    return column(value, scalar=scalar), scalar


def _coerce_pair(x, y, memory_pool=None, n_hint: int | None = None):
    x_scalar = not _is_arraylike(x)
    y_scalar = not _is_arraylike(y)
    x_type = None if x_scalar else column(x).array.type
    y_type = None if y_scalar else column(y).array.type
    left, x_scalar = _raw_column(x, y_type)
    right, y_scalar = _raw_column(y, left.array.type)

    _require_numeric(left)
    _require_numeric(right)
    lengths = []
    if not x_scalar:
        lengths.append(left.length)
    if not y_scalar:
        lengths.append(right.length)
    if n_hint is not None:
        lengths.append(n_hint)
    length = lengths[0] if lengths else 1
    if any(item != length for item in lengths):
        raise pa.ArrowInvalid("Array arguments must all be the same length")

    if left.array.type != right.array.type:
        if {left.array.type, right.array.type} != {_FLOAT, _INT}:
            raise TypeError(
                f"no matching kernel for {left.array.type} and {right.array.type}"
            )
        left = _cast_to_float(left, memory_pool)
        right = _cast_to_float(right, memory_pool)
        type = _FLOAT
    else:
        type = left.array.type
    return left, right, type, length, x_scalar and y_scalar


def _binary(name: str, x, y, memory_pool=None, *, force_float: bool = False):
    left, right, type, length, scalar = _coerce_pair(x, y, memory_pool)
    if force_float and type == _INT:
        left = _cast_to_float(left, memory_pool)
        right = _cast_to_float(right, memory_pool)
        type = _FLOAT
    values, validity = _result_buffers(type, length, memory_pool)
    if length:
        args = (
            left.values,
            left.bitmap,
            left.offset,
            left.step,
            right.values,
            right.bitmap,
            right.offset,
            right.step,
            length,
            values.address,
            validity.address,
            _BINARY_OPS.get(name, 5),
        )
        if type == _FLOAT:
            lib().mpa_binary_f64(*args)
        else:
            status = lib().mpa_binary_i64(*args)
            if status == 1:
                raise pa.ArrowInvalid("divide by zero")
            if status == 2:
                raise pa.ArrowInvalid("integers to negative integer powers are not allowed")
            if status != 0:
                raise RuntimeError(f"Mojo integer kernel failed with status {status}")
    return _finish(type, length, values, validity, scalar=scalar)


def add(x, y, /, *, memory_pool=None):
    return _binary("add", x, y, memory_pool)


def subtract(x, y, /, *, memory_pool=None):
    return _binary("subtract", x, y, memory_pool)


def multiply(x, y, /, *, memory_pool=None):
    return _binary("multiply", x, y, memory_pool)


def divide(dividend, divisor, /, *, memory_pool=None):
    return _binary("divide", dividend, divisor, memory_pool)


def power(base, exponent, /, *, memory_pool=None):
    return _binary("power", base, exponent, memory_pool)


def atan2(y, x, /, *, memory_pool=None):
    return _binary("atan2", y, x, memory_pool, force_float=True)


def _unary(name: str, x, memory_pool=None):
    scalar = not _is_arraylike(x)
    col = column(x, scalar=scalar)
    _require_numeric(col)
    type = col.array.type
    if name not in ("abs", "negate") and type == _INT:
        col = _cast_to_float(col, memory_pool)
        type = _FLOAT
    values, validity = _result_buffers(type, col.length, memory_pool)
    if col.length:
        args = (
            col.values,
            col.bitmap,
            col.offset,
            col.length,
            values.address,
            validity.address,
            _UNARY_OPS[name],
        )
        if type == _FLOAT:
            lib().mpa_unary_f64(*args)
        else:
            lib().mpa_unary_i64(*args)
    return _finish(type, col.length, values, validity, scalar=scalar)


def abs(x, /, *, memory_pool=None):
    return _unary("abs", x, memory_pool)


def negate(x, /, *, memory_pool=None):
    return _unary("negate", x, memory_pool)


def sqrt(x, /, *, memory_pool=None):
    return _unary("sqrt", x, memory_pool)


def exp(exponent, /, *, memory_pool=None):
    return _unary("exp", exponent, memory_pool)


def ln(x, /, *, memory_pool=None):
    return _unary("ln", x, memory_pool)


def log10(x, /, *, memory_pool=None):
    return _unary("log10", x, memory_pool)


def log2(x, /, *, memory_pool=None):
    return _unary("log2", x, memory_pool)


def sin(x, /, *, memory_pool=None):
    return _unary("sin", x, memory_pool)


def cos(x, /, *, memory_pool=None):
    return _unary("cos", x, memory_pool)


def tan(x, /, *, memory_pool=None):
    return _unary("tan", x, memory_pool)


def asin(x, /, *, memory_pool=None):
    return _unary("asin", x, memory_pool)


def acos(x, /, *, memory_pool=None):
    return _unary("acos", x, memory_pool)


def atan(x, /, *, memory_pool=None):
    return _unary("atan", x, memory_pool)


def _compare(name: str, x, y, memory_pool=None):
    left, right, type, length, scalar = _coerce_pair(x, y, memory_pool)
    values, validity = _bool_buffers(length, memory_pool)
    if length:
        args = (
            left.values,
            left.bitmap,
            left.offset,
            left.step,
            right.values,
            right.bitmap,
            right.offset,
            right.step,
            length,
            values.address,
            validity.address,
            _COMPARE_OPS[name],
        )
        if type == _FLOAT:
            lib().mpa_compare_f64(*args)
        else:
            lib().mpa_compare_i64(*args)
    return _finish(pa.bool_(), length, values, validity, scalar=scalar)


def equal(x, y, /, *, memory_pool=None):
    return _compare("equal", x, y, memory_pool)


def not_equal(x, y, /, *, memory_pool=None):
    return _compare("not_equal", x, y, memory_pool)


def less(x, y, /, *, memory_pool=None):
    return _compare("less", x, y, memory_pool)


def less_equal(x, y, /, *, memory_pool=None):
    return _compare("less_equal", x, y, memory_pool)


def greater(x, y, /, *, memory_pool=None):
    return _compare("greater", x, y, memory_pool)


def greater_equal(x, y, /, *, memory_pool=None):
    return _compare("greater_equal", x, y, memory_pool)


def _predicate(op: int, values, memory_pool=None):
    scalar = not _is_arraylike(values)
    col = column(values, scalar=scalar)
    _require_numeric(col)
    if col.array.type == _INT:
        col = _cast_to_float(col, memory_pool)
    result, validity = _bool_buffers(col.length, memory_pool)
    if col.length:
        lib().mpa_predicate_f64(
            col.values,
            col.bitmap,
            col.offset,
            col.length,
            result.address,
            validity.address,
            op,
        )
    return _finish(pa.bool_(), col.length, result, validity, scalar=scalar)


def is_nan(values, /, *, memory_pool=None):
    return _predicate(0, values, memory_pool)


def is_finite(values, /, *, memory_pool=None):
    return _predicate(1, values, memory_pool)


def is_inf(values, /, *, memory_pool=None):
    return _predicate(2, values, memory_pool)


def _option_int(text: str, name: str, default: int) -> int:
    match = re.search(rf"{name}=(-?\d+)", text)
    return int(match.group(1)) if match else default


def _option_bool(text: str, name: str, default: bool) -> bool:
    match = re.search(rf"{name}=(true|false)", text, flags=re.IGNORECASE)
    return (match.group(1).lower() == "true") if match else default


def _aggregate_args(skip_nulls, min_count, options, *, ddof=None):
    if options is None:
        return skip_nulls, min_count, ddof
    text = repr(options)
    skip_nulls = _option_bool(text, "skip_nulls", skip_nulls)
    min_count = _option_int(text, "min_count", min_count)
    if ddof is not None:
        ddof = _option_int(text, "ddof", ddof)
    return skip_nulls, min_count, ddof


def _aggregate_column(array) -> Column:
    col = column(array)
    _require_numeric(col)
    return col


def _aggregate_is_null(col: Column, skip_nulls: bool, min_count: int) -> bool:
    valid_count = col.length - col.null_count
    return (not skip_nulls and col.null_count > 0) or valid_count < min_count


def sum(
    array,
    /,
    *,
    skip_nulls=True,
    min_count=1,
    options=None,
    memory_pool=None,
):
    skip_nulls, min_count, _ = _aggregate_args(
        skip_nulls, min_count, options
    )
    col = _aggregate_column(array)
    if _aggregate_is_null(col, skip_nulls, min_count):
        return pa.scalar(None, type=col.array.type)
    if col.length == 0:
        return pa.scalar(0.0 if col.array.type == _FLOAT else 0, type=col.array.type)
    if col.array.type == _FLOAT:
        value = lib().mpa_sum_f64(col.values, col.bitmap, col.offset, col.length)
    else:
        value = lib().mpa_sum_i64(col.values, col.bitmap, col.offset, col.length)
    return pa.scalar(value, type=col.array.type)


def mean(
    array,
    /,
    *,
    skip_nulls=True,
    min_count=1,
    options=None,
    memory_pool=None,
):
    skip_nulls, min_count, _ = _aggregate_args(
        skip_nulls, min_count, options
    )
    col = _aggregate_column(array)
    if _aggregate_is_null(col, skip_nulls, min_count):
        return pa.scalar(None, type=_FLOAT)
    valid_count = col.length - col.null_count
    if valid_count == 0:
        return pa.scalar(float("nan"), type=_FLOAT)
    if col.array.type == _INT:
        value = lib().mpa_mean_i64(
            col.values, col.bitmap, col.offset, col.length
        )
    else:
        total = lib().mpa_sum_f64(
            col.values, col.bitmap, col.offset, col.length
        )
        value = total / valid_count
    return pa.scalar(value, type=_FLOAT)


def _minmax_values(col: Column):
    if col.array.type == _FLOAT:
        work = np.empty(2, dtype=np.float64)
        lib().mpa_minmax_f64(
            col.values, col.bitmap, col.offset, col.length, work.ctypes.data
        )
    else:
        work = np.empty(2, dtype=np.int64)
        lib().mpa_minmax_i64(
            col.values, col.bitmap, col.offset, col.length, work.ctypes.data
        )
    return work[0].item(), work[1].item()


def min_max(
    array,
    /,
    *,
    skip_nulls=True,
    min_count=1,
    options=None,
    memory_pool=None,
):
    skip_nulls, min_count, _ = _aggregate_args(
        skip_nulls, min_count, options
    )
    col = _aggregate_column(array)
    struct_type = pa.struct([("min", col.array.type), ("max", col.array.type)])
    if (
        _aggregate_is_null(col, skip_nulls, min_count)
        or col.length - col.null_count == 0
    ):
        return pa.scalar({"min": None, "max": None}, type=struct_type)
    lo, hi = _minmax_values(col)
    return pa.scalar({"min": lo, "max": hi}, type=struct_type)


def min(
    array,
    /,
    *,
    skip_nulls=True,
    min_count=1,
    options=None,
    memory_pool=None,
):
    pair = min_max(
        array,
        skip_nulls=skip_nulls,
        min_count=min_count,
        options=options,
        memory_pool=memory_pool,
    )
    return pair["min"]


def max(
    array,
    /,
    *,
    skip_nulls=True,
    min_count=1,
    options=None,
    memory_pool=None,
):
    pair = min_max(
        array,
        skip_nulls=skip_nulls,
        min_count=min_count,
        options=options,
        memory_pool=memory_pool,
    )
    return pair["max"]


def variance(
    array,
    /,
    *,
    ddof=0,
    skip_nulls=True,
    min_count=0,
    options=None,
    memory_pool=None,
):
    skip_nulls, min_count, ddof = _aggregate_args(
        skip_nulls, min_count, options, ddof=ddof
    )
    col = _aggregate_column(array)
    valid_count = col.length - col.null_count
    if (
        _aggregate_is_null(col, skip_nulls, min_count)
        or valid_count <= ddof
        or valid_count == 0
    ):
        return pa.scalar(None, type=_FLOAT)
    floating = _cast_to_float(col, memory_pool)
    value = lib().mpa_variance_f64(
        floating.values,
        floating.bitmap,
        floating.offset,
        floating.length,
        ddof,
    )
    return pa.scalar(value, type=_FLOAT)


def stddev(
    array,
    /,
    *,
    ddof=0,
    skip_nulls=True,
    min_count=0,
    options=None,
    memory_pool=None,
):
    result = variance(
        array,
        ddof=ddof,
        skip_nulls=skip_nulls,
        min_count=min_count,
        options=options,
        memory_pool=memory_pool,
    )
    if not result.is_valid:
        return result
    value = result.as_py()
    return pa.scalar(math.sqrt(value) if value >= 0.0 else float("nan"), type=_FLOAT)


def count(array, /, mode="only_valid", *, options=None, memory_pool=None):
    if options is not None:
        text = repr(options).lower()
        if "only_null" in text:
            mode = "only_null"
        elif "all" in text:
            mode = "all"
        else:
            mode = "only_valid"
    col = column(array)
    if mode == "only_valid":
        value = col.length - col.null_count
    elif mode == "only_null":
        value = col.null_count
    elif mode == "all":
        value = col.length
    else:
        raise ValueError(f"{mode!r} is not a valid count mode")
    return pa.scalar(value, type=_INT)


def cumulative_sum(
    values,
    /,
    start=None,
    *,
    skip_nulls=False,
    options=None,
    memory_pool=None,
):
    if options is not None:
        text = repr(options)
        skip_nulls = _option_bool(text, "skip_nulls", skip_nulls)
        if "start=nullopt" not in text:
            raise NotImplementedError(
                "CumulativeOptions.start is not introspectable; pass start= directly"
            )
    col = _aggregate_column(values)
    start_value = start.as_py() if isinstance(start, pa.Scalar) else start
    # Convert before entering ctypes: ctypes integers otherwise wrap oversized
    # Python values modulo 2**64, and reject valid integral floats itself.
    initial_scalar = pa.scalar(
        0 if start_value is None else start_value, type=col.array.type
    )
    initial = initial_scalar.as_py()
    if (
        col.array.type == _INT
        and start_value is not None
        and start_value != initial
    ):
        raise pa.ArrowInvalid(
            f"Float value {start_value!r} was truncated converting to int64"
        )
    result, validity = _result_buffers(col.array.type, col.length, memory_pool)
    if col.length:
        args = (
            col.values,
            col.bitmap,
            col.offset,
            col.length,
            initial,
            int(skip_nulls),
            result.address,
            validity.address,
        )
        if col.array.type == _FLOAT:
            lib().mpa_cumulative_f64(*args)
        else:
            lib().mpa_cumulative_i64(*args)
    return _finish(col.array.type, col.length, result, validity)


def if_else(cond, left, right, /, *, memory_pool=None):
    cond_scalar = not _is_arraylike(cond)
    condition = column(cond, scalar=cond_scalar)
    if not pa.types.is_boolean(condition.array.type):
        raise TypeError(f"condition must be boolean, got {condition.array.type}")
    n_hint = None if cond_scalar else condition.length
    lhs, rhs, type, length, sides_scalar = _coerce_pair(
        left, right, memory_pool, n_hint=n_hint
    )
    result, validity = _result_buffers(type, length, memory_pool)
    if length:
        args = (
            condition.values,
            condition.bitmap,
            condition.offset,
            condition.step,
            lhs.values,
            lhs.bitmap,
            lhs.offset,
            lhs.step,
            rhs.values,
            rhs.bitmap,
            rhs.offset,
            rhs.step,
            length,
            result.address,
            validity.address,
        )
        if type == _FLOAT:
            lib().mpa_if_else_f64(*args)
        else:
            lib().mpa_if_else_i64(*args)
    return _finish(
        type, length, result, validity, scalar=cond_scalar and sides_scalar
    )


def fill_null(values, fill_value):
    scalar = not _is_arraylike(values)
    col = column(values, scalar=scalar)
    _require_numeric(col)
    fill_scalar = not _is_arraylike(fill_value)
    fill = column(fill_value, type=col.array.type, scalar=fill_scalar)
    if not fill_scalar and fill.length != col.length:
        raise pa.ArrowInvalid("Array arguments must all be the same length")
    result, validity = _result_buffers(col.array.type, col.length)
    if col.length:
        args = (
            col.values,
            col.bitmap,
            col.offset,
            fill.values,
            fill.bitmap,
            fill.offset,
            fill.step,
            col.length,
            result.address,
            validity.address,
        )
        if col.array.type == _FLOAT:
            lib().mpa_fill_null_f64(*args)
        else:
            lib().mpa_fill_null_i64(*args)
    return _finish(col.array.type, col.length, result, validity, scalar=scalar)


def filter(
    input,
    selection_filter,
    /,
    null_selection_behavior="drop",
    *,
    options=None,
    memory_pool=None,
):
    col = _aggregate_column(input)
    selection = column(selection_filter)
    if not pa.types.is_boolean(selection.array.type):
        raise TypeError("selection_filter must be boolean")
    if selection.length != col.length:
        raise pa.ArrowInvalid("Filter inputs must all be the same length")
    behavior = null_selection_behavior
    if options is not None:
        behavior = "emit_null" if "EMIT_NULL" in repr(options) else "drop"
    if behavior not in ("drop", "emit_null"):
        raise ValueError(f"{behavior!r} is not a valid null selection behavior")
    result, validity = _result_buffers(col.array.type, col.length, memory_pool)
    if not col.length:
        return _finish(col.array.type, 0, result, validity)
    args = (
        col.values,
        col.bitmap,
        col.offset,
        col.length,
        selection.values,
        selection.bitmap,
        selection.offset,
        int(behavior == "emit_null"),
        result.address,
        validity.address,
    )
    kept = (
        lib().mpa_filter_f64(*args)
        if col.array.type == _FLOAT
        else lib().mpa_filter_i64(*args)
    )
    if kept < 0 or kept > col.length:
        raise RuntimeError(f"Mojo filter returned invalid length {kept}")
    return _finish(col.array.type, kept, result, validity)


__all__ = [
    "abs",
    "acos",
    "add",
    "asin",
    "atan",
    "atan2",
    "cos",
    "count",
    "cumulative_sum",
    "divide",
    "equal",
    "exp",
    "fill_null",
    "filter",
    "greater",
    "greater_equal",
    "if_else",
    "is_finite",
    "is_inf",
    "is_nan",
    "less",
    "less_equal",
    "ln",
    "log10",
    "log2",
    "max",
    "mean",
    "min",
    "min_max",
    "multiply",
    "negate",
    "not_equal",
    "power",
    "sin",
    "sqrt",
    "stddev",
    "subtract",
    "sum",
    "tan",
    "variance",
]
