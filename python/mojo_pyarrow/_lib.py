"""Load the Mojo shared library and expose Arrow buffers without copying."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

import pyarrow as pa

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "kernels.mojo")
LIB = os.environ.get("MOJO_PYARROW_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pyarrow.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mpa_binary_f64": ([I] * 12, None),
    "mpa_binary_i64": ([I] * 12, I),
    "mpa_unary_f64": ([I] * 7, None),
    "mpa_unary_i64": ([I] * 7, None),
    "mpa_compare_f64": ([I] * 12, None),
    "mpa_compare_i64": ([I] * 12, None),
    "mpa_predicate_f64": ([I] * 7, None),
    "mpa_sum_f64": ([I] * 4, F),
    "mpa_sum_i64": ([I] * 4, I),
    "mpa_mean_i64": ([I] * 4, F),
    "mpa_minmax_f64": ([I] * 5, None),
    "mpa_minmax_i64": ([I] * 5, None),
    "mpa_variance_f64": ([I] * 5, F),
    "mpa_cumulative_f64": ([I, I, I, I, F, I, I, I], None),
    "mpa_cumulative_i64": ([I] * 8, None),
    "mpa_cast_i64_f64": ([I] * 6, None),
    "mpa_if_else_f64": ([I] * 15, None),
    "mpa_if_else_i64": ([I] * 15, None),
    "mpa_fill_null_f64": ([I] * 10, None),
    "mpa_fill_null_i64": ([I] * 10, None),
    "mpa_filter_f64": ([I] * 10, I),
    "mpa_filter_i64": ([I] * 10, I),
}


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJO_PYARROW_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    manifest = os.path.join(ROOT, "pixi.toml")
    if os.path.exists(pixi) and os.path.exists(manifest):
        return [pixi, "run", "--manifest-path", manifest, "mojo"]
    raise BuildError("mojo not found; set MOJO_PYARROW_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_PYARROW_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    command = _mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        SRC,
        "-o",
        LIB,
    ]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_loaded = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_loaded, name)
            function.argtypes = argtypes
            function.restype = restype
    return _loaded


@dataclass(frozen=True)
class Column:
    array: pa.Array
    bitmap: int
    values: int
    offset: int
    length: int
    null_count: int
    step: int = 1


def as_array(value, type: pa.DataType | None = None) -> pa.Array:
    if isinstance(value, pa.ChunkedArray):
        value = value.combine_chunks()
    if isinstance(value, pa.Array):
        if type is not None and value.type != type:
            raise TypeError(f"expected {type}, got {value.type}")
        return value
    return pa.array(value, type=type)


def column(value, type: pa.DataType | None = None, *, scalar: bool = False) -> Column:
    if scalar:
        scalar_value = value if isinstance(value, pa.Scalar) else pa.scalar(value, type=type)
        if type is not None and scalar_value.type != type:
            scalar_value = pa.scalar(scalar_value.as_py(), type=type)
        array = pa.array([scalar_value.as_py()], type=scalar_value.type)
        step = 0
    else:
        array = as_array(value, type)
        step = 1
    validity, values = array.buffers()[:2]
    bitmap = validity.address if validity is not None and array.null_count else 0
    return Column(
        array=array,
        bitmap=bitmap,
        values=values.address if values is not None else 0,
        offset=array.offset,
        length=len(array),
        null_count=array.null_count,
        step=step,
    )


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
