# mojo-pyarrow

`mojo-pyarrow` is a standalone implementation of frequently used
[`pyarrow.compute`](https://arrow.apache.org/docs/python/compute.html) numeric
kernels in Mojo. It accepts real PyArrow arrays, reads their Arrow buffers
without converting them to NumPy, and returns real `pyarrow.Array` and
`pyarrow.Scalar` objects.

The covered functions keep PyArrow's names, argument order, keyword arguments,
scalar broadcasting, null propagation, and return types:

```python
import pyarrow as pa
from mojo_pyarrow import compute as pc

values = pa.array([1.0, 2.0, None, 4.0])

print(pc.add(values, 0.5).to_pylist())
# [1.5, 2.5, None, 4.5]

print(pc.sum(values).as_py())
# 7.0

print(pc.filter(values, pc.greater(values, 1.5)).to_pylist())
# [2.0, 4.0]
```

For the implemented subset, replacing `import pyarrow.compute as pc` with
`from mojo_pyarrow import compute as pc` is the intended migration.

## Covered subset

| group | `pyarrow.compute`-compatible functions |
| --- | --- |
| arithmetic | `add`, `subtract`, `multiply`, `divide`, `power`, `negate`, `abs` |
| transcendental | `sqrt`, `exp`, `ln`, `log10`, `log2`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` |
| comparisons | `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` |
| floating predicates | `is_nan`, `is_finite`, `is_inf` |
| aggregation | `sum`, `mean`, `min`, `max`, `min_max`, `variance`, `stddev`, `count` |
| vector and selection | `cumulative_sum`, `if_else`, `fill_null`, `filter` |

The compute kernels support Arrow `float64` and `int64`. Mixed `int64` and
`float64` arithmetic follows PyArrow by promoting to `float64`.
Transcendental functions accept `int64` and return `float64`. Array-array,
array-scalar, scalar-array, scalar-scalar, null-carrying, sliced, and ordinary
Python list inputs are covered. Chunked arrays are accepted and combined
before execution.

The 86 parity tests compare results and types against PyArrow 25 on the same
inputs. They include null-selection behavior, aggregation options, integer
division semantics, NaN and infinity, empty arrays, all-null arrays, mixed
numeric types, non-byte-aligned sliced validity bitmaps, SIMD tails, and both
sides of the chunked execution threshold. They also cover stable variance for
large-offset values and reject scalar values that would narrow or wrap at the
FFI boundary.

## Not covered

This is a focused compute port, not a replacement for the whole PyArrow
package. It does not implement:

- widths other than `float64` and `int64`, or unsigned, decimal, temporal,
  string, binary, dictionary, list, and struct kernels;
- checked arithmetic variants such as `add_checked` and `sqrt_checked`;
- sorting, take, joins, hash aggregation, table kernels, dataset scanning,
  serialization, or IPC;
- chunk-preserving execution; `ChunkedArray` inputs are combined into one
  array;
- `CumulativeOptions.start`, whose value is opaque to Python; pass the
  supported `start=` argument directly.

PyArrow remains a runtime dependency for its array objects, allocators, and
types. The numerical loops listed above execute in Mojo.

## Install and run

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` compiles the single Mojo unit as a shared library:

```text
dist/libmojo-pyarrow.so
```

The Python loader also rebuilds the library on first import when it is absent
or older than `src/kernels.mojo`. An already-built library can be selected
with `MOJO_PYARROW_LIB=/path/to/libmojo-pyarrow.so`.

## Performance

Measured on an Intel Xeon E5-2697 v4 system with 72 logical CPUs, Mojo
`1.1.0.dev2026081105`, PyArrow 25.0.0, and five million elements. Each number
is the best of five warm runs produced by `pixi run bench`, whose task holds a
machine-wide file lock.

| kernel | mojo-pyarrow | pyarrow | relative |
| --- | ---: | ---: | ---: |
| `add, array + array (5M)` | 5.29 ms | 13.52 ms | 2.55x faster |
| `multiply, array * scalar (5M)` | 3.09 ms | 10.23 ms | 3.31x faster |
| `sin (5M)` | 18.88 ms | 89.24 ms | 4.73x faster |
| `sum, dense float64 (5M)` | 2.94 ms | 3.42 ms | 1.16x faster |
| `sum, 10% nulls (5M)` | 12.67 ms | 18.61 ms | 1.47x faster |
| `mean, int64 (5M)` | 4.34 ms | 5.26 ms | 1.21x faster |
| `variance, dense (5M)` | 0.70 ms | 8.66 ms | 12.34x faster |
| `greater than scalar (5M)` | 3.31 ms | 4.81 ms | 1.46x faster |
| `cumulative_sum (5M)` | 14.18 ms | 21.42 ms | 1.51x faster |
| `if_else (5M)` | 7.15 ms | 33.49 ms | 4.68x faster |
| `filter, about 50% kept (5M)` | 9.07 ms | 19.79 ms | 2.18x faster |

Dense arithmetic, transcendental, comparison, variance, and selection loops
stay serial below 262,144 rows and switch to a 65,536-row chunk loop above that
threshold. These kernels stream: an elementwise op moves 16 bytes per element
for at most one flop, comparison packs eight lanes into a byte for about one
compare, and selection moves one condition bit plus one 8-byte value per
element. None reaches the roughly two-flops-per-byte point where splitting the
work across threads pays, so the chunk loop stays on the calling thread.
Comparisons use native-width SIMD loads and comparisons, then pack their lane
results into Arrow bitmaps with a scalar remainder. Dense filtering walks set
bits in the selection bitmap instead of testing every row. Dense variance uses
an anchored, numerically stable two-pass SIMD reduction with per-chunk
partials; nullable variance retains Welford's recurrence.

A GPU path was evaluated with 14,095 MiB free and is intentionally not included
or benchmarked. The only compute-bound candidate here is `sin`, and the pinned
Mojo NVIDIA backend rejects float64 `sin`; using float32 would violate the
existing parity tolerance. The remaining kernels are bandwidth-bound and do
not have enough arithmetic intensity to justify host/device copies.

Run `pixi run bench` on the target machine before making a deployment choice;
memory bandwidth and Arrow allocator state materially affect these kernels.

## How it works

An Arrow primitive array is already the memory layout the kernels need:

```text
pyarrow.Array
  validity bitmap: one little-endian bit per row, optional
  values buffer:   contiguous float64 or int64
  offset:          element and bit offset for zero-copy slices
  length
```

The Python wrapper reads those buffer addresses directly. A ctypes call passes
the addresses, offsets, lengths, scalar strides, and operation code to one
compiled Mojo shared library. In accordance with Mojo's C ABI constraints,
buffers cross as integer addresses and are reconstructed as
`UnsafePointer[..., AnyOrigin[mut=True]]` inside the exported function.

Mojo never owns or allocates Arrow memory. Python allocates output value and
validity buffers from Arrow's memory pool, Mojo fills them, and
`pa.Array.from_buffers` wraps them without another copy. A missing validity
buffer selects the dense fast path. Sliced inputs keep their original base
address and pass their Arrow offset, including the bit offset for validity.

## License

MIT
