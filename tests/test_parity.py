"""Behavioral parity with pyarrow.compute on the same Arrow buffers."""

import inspect

import numpy as np
import pyarrow as pa
import pyarrow.compute as upstream
import pytest

import mojo_pyarrow.compute as pc
from mojo_pyarrow._lib import column


def assert_result_equal(actual, expected, *, rtol=2e-12):
    assert actual.type == expected.type
    if isinstance(actual, pa.Scalar):
        assert actual.is_valid == expected.is_valid
        if not actual.is_valid:
            return
        av, ev = actual.as_py(), expected.as_py()
        if isinstance(av, float):
            assert av == pytest.approx(ev, rel=rtol, abs=1e-14, nan_ok=True)
        else:
            assert av == ev
        return
    assert actual.is_valid().to_pylist() == expected.is_valid().to_pylist()
    if pa.types.is_floating(actual.type):
        np.testing.assert_allclose(
            actual.to_numpy(zero_copy_only=False),
            expected.to_numpy(zero_copy_only=False),
            rtol=rtol,
            atol=1e-14,
            equal_nan=True,
        )
    else:
        assert actual.to_pylist() == expected.to_pylist()


@pytest.fixture(scope="module")
def floats():
    rng = np.random.default_rng(0)
    values = rng.normal(size=4096)
    values[[11, 99]] = [float("nan"), float("inf")]
    mask = rng.random(values.size) < 0.12
    mask[[11, 99]] = False
    return pa.array(values, mask=mask)


@pytest.fixture(scope="module")
def ints():
    rng = np.random.default_rng(1)
    values = rng.integers(-1000, 1000, size=4096, dtype=np.int64)
    mask = rng.random(values.size) < 0.1
    return pa.array(values, mask=mask)


@pytest.mark.parametrize("name", ["add", "subtract", "multiply", "divide"])
def test_float_binary_array_parity(name, floats):
    rhs = pa.array(np.linspace(1.0, 2.0, len(floats)))
    assert_result_equal(getattr(pc, name)(floats, rhs), getattr(upstream, name)(floats, rhs))


@pytest.mark.parametrize("name", ["add", "subtract", "multiply", "divide", "power"])
def test_integer_binary_array_and_scalar_parity(name):
    left = pa.array([-5, -2, None, 0, 2, 5], type=pa.int64())
    right = 2
    assert_result_equal(getattr(pc, name)(left, right), getattr(upstream, name)(left, right))


def test_integer_division_truncates_toward_zero():
    left = pa.array([-3, 3, -3, 3], type=pa.int64())
    right = pa.array([2, -2, -2, 2], type=pa.int64())
    assert_result_equal(pc.divide(left, right), upstream.divide(left, right))


def test_integer_division_by_zero_raises():
    values = pa.array([1, None, 3], type=pa.int64())
    with pytest.raises(pa.ArrowInvalid):
        pc.divide(values, pa.array([1, 0, 0], type=pa.int64()))
    assert_result_equal(
        pc.divide(values, pa.array([1, 0, 2], type=pa.int64())),
        upstream.divide(values, pa.array([1, 0, 2], type=pa.int64())),
    )


def test_power_negative_integer_exponent_raises():
    with pytest.raises(pa.ArrowInvalid):
        pc.power(pa.array([2, 3], type=pa.int64()), -1)


def test_scalar_results_and_broadcasting():
    for name, args in [
        ("add", (1, 2)),
        ("divide", (7.0, 2.0)),
        ("power", (3, 4)),
        ("atan2", (1.0, -1.0)),
    ]:
        assert_result_equal(getattr(pc, name)(*args), getattr(upstream, name)(*args))
    values = pa.array([1.0, None, 3.0])
    assert_result_equal(pc.add(2.0, values), upstream.add(2.0, values))


def test_mixed_int_float_promotes_to_double():
    left = pa.array([1, 2, None], type=pa.int64())
    right = pa.array([0.5, 1.5, 2.5], type=pa.float64())
    assert_result_equal(pc.multiply(left, right), upstream.multiply(left, right))


@pytest.mark.parametrize(
    "name",
    ["abs", "negate", "sqrt", "exp", "ln", "log10", "log2", "sin", "cos", "tan", "asin", "acos", "atan"],
)
def test_unary_float_parity(name):
    values = pa.array(
        [0.0, -0.0, 0.25, 1.0, -1.0, None, float("nan"), float("inf"), float("-inf")]
    )
    assert_result_equal(getattr(pc, name)(values), getattr(upstream, name)(values))


@pytest.mark.parametrize("name", ["abs", "negate"])
def test_unary_integer_parity(name, ints):
    assert_result_equal(getattr(pc, name)(ints), getattr(upstream, name)(ints))


@pytest.mark.parametrize(
    "name",
    [
        "sqrt",
        "exp",
        "ln",
        "log10",
        "log2",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
    ],
)
def test_transcendentals_accept_int64_and_return_double(name):
    values = pa.array([0, 1, 2, None], type=pa.int64())
    assert_result_equal(getattr(pc, name)(values), getattr(upstream, name)(values))


@pytest.mark.parametrize(
    "name", ["equal", "not_equal", "less", "less_equal", "greater", "greater_equal"]
)
def test_comparison_parity(name, floats):
    sliced = floats.slice(13, 1001)
    assert_result_equal(
        getattr(pc, name)(sliced, 0.25), getattr(upstream, name)(sliced, 0.25)
    )


def test_array_array_comparison_and_mixed_types():
    left = pa.array([1, 2, None, 4], type=pa.int64())
    right = pa.array([1.0, 1.5, 3.0, None], type=pa.float64())
    assert_result_equal(pc.greater_equal(left, right), upstream.greater_equal(left, right))


@pytest.mark.parametrize("name", ["is_nan", "is_finite", "is_inf"])
def test_float_predicates_preserve_nulls(name):
    values = pa.array([0.0, None, float("nan"), float("inf"), float("-inf")])
    assert_result_equal(getattr(pc, name)(values), getattr(upstream, name)(values))


@pytest.mark.parametrize("name", ["is_nan", "is_finite", "is_inf"])
def test_float_predicates_accept_int64(name):
    values = pa.array([0, 1, None, -2], type=pa.int64())
    assert_result_equal(getattr(pc, name)(values), getattr(upstream, name)(values))


def test_atan2_array_parity():
    y = pa.array([0, 1, None, -1], type=pa.int64())
    x = pa.array([1.0, 0.0, -1.0, None], type=pa.float64())
    assert_result_equal(pc.atan2(y, x), upstream.atan2(y, x))


def test_list_and_chunked_array_inputs():
    assert_result_equal(pc.add([1, 2, 3], 4), upstream.add([1, 2, 3], 4))
    values = pa.chunked_array(
        [pa.array([1.0, None]), pa.array([3.0, 4.0])]
    )
    assert_result_equal(pc.sum(values), upstream.sum(values))


@pytest.mark.parametrize("name", ["sum", "mean", "min", "max", "min_max"])
def test_basic_aggregates(name, floats, ints):
    for values in (floats, ints, floats.slice(37, 1500), ints.slice(5, 2000)):
        assert_result_equal(getattr(pc, name)(values), getattr(upstream, name)(values))


@pytest.mark.parametrize("ddof", [0, 1, 3])
def test_variance_and_stddev(ddof, floats, ints):
    for values in (floats, ints, floats.slice(23, 1234)):
        assert_result_equal(pc.variance(values, ddof=ddof), upstream.variance(values, ddof=ddof), rtol=2e-10)
        assert_result_equal(pc.stddev(values, ddof=ddof), upstream.stddev(values, ddof=ddof), rtol=2e-10)


def test_variance_is_stable_for_large_offsets():
    values = pa.array([1.0e12 + i for i in range(2000)])
    for ddof in (0, 1):
        assert_result_equal(
            pc.variance(values, ddof=ddof),
            upstream.variance(values, ddof=ddof),
            rtol=2e-10,
        )


def test_minmax_skips_nan_when_numbers_are_present():
    values = pa.array([float("nan"), 1.0, 2.0])
    for name in ("min", "max", "min_max"):
        assert_result_equal(getattr(pc, name)(values), getattr(upstream, name)(values))


def test_aggregate_null_and_min_count_semantics():
    values = pa.array([1.0, None])
    for name in ("sum", "mean", "min", "max", "min_max", "variance", "stddev"):
        for skip_nulls in (True, False):
            for min_count in (0, 1, 3):
                kwargs = {"skip_nulls": skip_nulls, "min_count": min_count}
                assert_result_equal(
                    getattr(pc, name)(values, **kwargs),
                    getattr(upstream, name)(values, **kwargs),
                )


def test_empty_and_all_null_aggregates():
    for values in (
        pa.array([], type=pa.float64()),
        pa.array([None, None], type=pa.float64()),
        pa.array([], type=pa.int64()),
    ):
        for name in ("sum", "mean", "min_max", "variance"):
            for min_count in (0, 1):
                assert_result_equal(
                    getattr(pc, name)(values, min_count=min_count),
                    getattr(upstream, name)(values, min_count=min_count),
                )


def test_aggregate_options_objects():
    values = pa.array([1.0, None, 3.0])
    scalar_options = upstream.ScalarAggregateOptions(skip_nulls=False, min_count=0)
    assert_result_equal(pc.sum(values, options=scalar_options), upstream.sum(values, options=scalar_options))
    variance_options = upstream.VarianceOptions(ddof=1, skip_nulls=True, min_count=2)
    assert_result_equal(
        pc.variance(values, options=variance_options),
        upstream.variance(values, options=variance_options),
    )


@pytest.mark.parametrize("mode", ["only_valid", "only_null", "all"])
def test_count_modes(mode, floats):
    assert_result_equal(pc.count(floats, mode=mode), upstream.count(floats, mode=mode))


@pytest.mark.parametrize("skip_nulls", [False, True])
def test_cumulative_sum(skip_nulls):
    for values in (
        pa.array([1.0, None, 2.0, 3.0]),
        pa.array([1, None, 2, 3], type=pa.int64()),
    ):
        assert_result_equal(
            pc.cumulative_sum(values, start=5, skip_nulls=skip_nulls),
            upstream.cumulative_sum(values, start=5, skip_nulls=skip_nulls),
        )


@pytest.mark.parametrize("start", [1.0, pa.scalar(None, type=pa.int64())])
def test_cumulative_sum_coerces_start_like_pyarrow(start):
    values = pa.array([1, 2], type=pa.int64())
    assert_result_equal(
        pc.cumulative_sum(values, start=start),
        upstream.cumulative_sum(values, start=start),
    )


def test_cumulative_sum_rejects_narrowing_and_out_of_range_start():
    values = pa.array([1, 2], type=pa.int64())
    for start in (1.5, 2**63, -(2**63) - 1):
        with pytest.raises((pa.ArrowInvalid, pa.ArrowTypeError, OverflowError)):
            pc.cumulative_sum(values, start=start)


def test_if_else_array_and_scalar_parity():
    cond = pa.array([True, False, None, True])
    left = pa.array([1.0, 2.0, 3.0, None])
    right = pa.array([10.0, None, 30.0, 40.0])
    assert_result_equal(pc.if_else(cond, left, right), upstream.if_else(cond, left, right))
    assert_result_equal(pc.if_else(cond, 1, 2), upstream.if_else(cond, 1, 2))
    assert_result_equal(pc.if_else(True, 1, 2), upstream.if_else(True, 1, 2))


def test_fill_null_scalar_and_array():
    values = pa.array([1.0, None, 3.0, None])
    for fill in (2.0, pa.array([10.0, 20.0, None, 40.0])):
        assert_result_equal(pc.fill_null(values, fill), upstream.fill_null(values, fill))


@pytest.mark.parametrize("behavior", ["drop", "emit_null"])
def test_filter_null_selection_behavior(behavior):
    values = pa.array([1.0, None, 3.0, 4.0, 5.0])
    mask = pa.array([True, True, None, False, True])
    assert_result_equal(
        pc.filter(values, mask, behavior),
        upstream.filter(values, mask, behavior),
    )


def test_filter_options_object():
    values = pa.array([1, 2, 3], type=pa.int64())
    mask = pa.array([True, None, False])
    options = upstream.FilterOptions(null_selection_behavior="emit_null")
    assert_result_equal(
        pc.filter(values, mask, options=options),
        upstream.filter(values, mask, options=options),
    )


def test_dense_simd_and_filter_tails():
    values = pa.array(np.linspace(-2.0, 2.0, 19))
    other = pa.array(np.linspace(2.0, -2.0, 19))
    mask = pa.array(
        [True, False, True, True, False, False, True, False, True]
        * 2
        + [True]
    )
    assert_result_equal(pc.add(values, other), upstream.add(values, other))
    assert_result_equal(pc.greater(values, other), upstream.greater(values, other))
    assert_result_equal(pc.filter(values, mask), upstream.filter(values, mask))


@pytest.mark.parametrize("length", [262_143, 262_149])
def test_serial_and_parallel_threshold_paths(length):
    values = pa.array(np.linspace(-1.0, 1.0, length))
    other = pa.array(np.linspace(1.0, -1.0, length))
    condition = upstream.greater(values, 0.0)
    assert_result_equal(pc.add(values, 0.25), upstream.add(values, 0.25))
    assert_result_equal(
        pc.variance(values), upstream.variance(values), rtol=2e-10
    )
    assert_result_equal(pc.sin(values), upstream.sin(values))
    assert_result_equal(pc.greater(values, 0.0), condition)
    assert_result_equal(
        pc.if_else(condition, values, other),
        upstream.if_else(condition, values, other),
    )


def test_sliced_buffers_are_zero_copy_and_offsets_are_respected(floats):
    sliced = floats.slice(17, 777)
    metadata = column(sliced)
    assert metadata.values == floats.buffers()[1].address
    assert metadata.offset == 17
    assert_result_equal(pc.add(sliced, 1.0), upstream.add(sliced, 1.0))
    assert_result_equal(pc.cumulative_sum(sliced, skip_nulls=True), upstream.cumulative_sum(sliced, skip_nulls=True))


def test_public_signatures_match_upstream_shape():
    for name in ("add", "sqrt", "sum", "variance", "cumulative_sum", "filter"):
        assert str(inspect.signature(getattr(pc, name))) == str(
            inspect.signature(getattr(upstream, name))
        )
