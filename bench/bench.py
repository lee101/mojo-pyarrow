"""Benchmark Mojo kernels against pyarrow.compute on identical Arrow arrays."""

from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as upstream

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import mojo_pyarrow.compute as pc  # noqa: E402

N = 5_000_000


def timeit(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as info:
            for line in info:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def mojo_version() -> str:
    try:
        result = subprocess.run(
            ["mojo", "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip().removeprefix("Mojo ").split(" ", 1)[0]
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def cases():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=N)
    other_raw = rng.normal(size=N)
    null_mask = rng.random(N) < 0.1
    dense = pa.array(raw)
    other = pa.array(other_raw)
    nullable = pa.array(raw, mask=null_mask)
    ints = pa.array(rng.integers(-1000, 1000, size=N, dtype=np.int64))
    condition = upstream.greater(dense, 0.0)

    yield "add, array + array (5M)", lambda: pc.add(dense, other), lambda: upstream.add(dense, other)
    yield "multiply, array * scalar (5M)", lambda: pc.multiply(dense, 1.5), lambda: upstream.multiply(dense, 1.5)
    yield "sin (5M)", lambda: pc.sin(dense), lambda: upstream.sin(dense)
    yield "sum, dense float64 (5M)", lambda: pc.sum(dense), lambda: upstream.sum(dense)
    yield "sum, 10% nulls (5M)", lambda: pc.sum(nullable), lambda: upstream.sum(nullable)
    yield "mean, int64 (5M)", lambda: pc.mean(ints), lambda: upstream.mean(ints)
    yield "variance, dense (5M)", lambda: pc.variance(dense), lambda: upstream.variance(dense)
    yield "greater than scalar (5M)", lambda: pc.greater(dense, 0.0), lambda: upstream.greater(dense, 0.0)
    yield "cumulative_sum (5M)", lambda: pc.cumulative_sum(dense), lambda: upstream.cumulative_sum(dense)
    yield "if_else (5M)", lambda: pc.if_else(condition, dense, other), lambda: upstream.if_else(condition, dense, other)
    yield "filter, about 50% kept (5M)", lambda: pc.filter(dense, condition), lambda: upstream.filter(dense, condition)


def main() -> None:
    print(f"Machine: {cpu_name()}, {os.cpu_count()} logical CPUs")
    print(
        f"Software: Mojo {mojo_version()}, pyarrow {pa.__version__}; "
        "best of 5 warm runs"
    )
    print()
    print("| kernel | mojo-pyarrow | pyarrow | relative |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, theirs in cases():
        ours()
        theirs()
        mojo_time = timeit(ours)
        arrow_time = timeit(theirs)
        ratio = arrow_time / mojo_time
        outcome = "faster" if ratio >= 1.0 else "slower"
        print(
            f"| `{name}` | {mojo_time * 1e3:.2f} ms | "
            f"{arrow_time * 1e3:.2f} ms | {ratio:.2f}x {outcome} |"
        )


if __name__ == "__main__":
    main()
