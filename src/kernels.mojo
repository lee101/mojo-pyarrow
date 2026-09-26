"""Numeric Arrow compute kernels and their C ABI.

Arrow buffers stay owned by Python. Addresses cross the ABI as `Int`; a zero
validity address means the array has no nulls.
"""

from std.math import (
    abs,
    acos,
    asin,
    atan,
    atan2,
    cos,
    exp,
    log,
    log10,
    log2,
    pow,
    sin,
    sqrt,
    tan,
)
from std.bit import count_trailing_zeros
from std.memory import stack_allocation
from std.sys.info import simd_width_of

comptime F64 = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime I64 = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime Bits = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()
comptime F64_MAX = 1.7976931348623157e308
comptime I64_MAX = Int64(9223372036854775807)
comptime I64_MIN = Int64(-9223372036854775808)
comptime PARALLEL_THRESHOLD = 262144
comptime PARALLEL_GRAIN = 65536
comptime VARIANCE_TASKS = 16


def fp(addr: Int) -> F64:
    return F64(unsafe_from_address=addr)


def ip(addr: Int) -> I64:
    return I64(unsafe_from_address=addr)


def bp(addr: Int) -> Bits:
    return Bits(unsafe_from_address=addr)


def get_bit(bitmap: Bits, i: Int) -> Bool:
    return ((bitmap[i >> 3] >> UInt8(i & 7)) & 1) != 0


def set_bit(bitmap: Bits, i: Int, value: Bool):
    var byte = i >> 3
    var mask = UInt8(1) << UInt8(i & 7)
    if value:
        bitmap[byte] |= mask
    else:
        bitmap[byte] &= ~mask


def valid(bitmap_addr: Int, offset: Int, i: Int) -> Bool:
    if bitmap_addr == 0:
        return True
    return get_bit(bp(bitmap_addr), offset + i)


def fill_valid(bitmap: Bits, n: Int):
    for b in range((n + 7) >> 3):
        bitmap[b] = 0xFF


def copy_validity(bitmap_addr: Int, offset: Int, n: Int, dst: Bits):
    if bitmap_addr == 0:
        fill_valid(dst, n)
        return
    for i in range(n):
        set_bit(dst, i, valid(bitmap_addr, offset, i))


def binary_f64_dense_range(
    a: F64,
    a_offset: Int,
    a_step: Int,
    b: F64,
    b_offset: Int,
    b_step: Int,
    start: Int,
    stop: Int,
    dst: F64,
    op: Int,
):
    var i = start
    while i + W <= stop:
        var av = (
            SIMD[DType.float64, W](a[a_offset])
            if a_step == 0
            else a.load[width=W](a_offset + i)
        )
        var bv = (
            SIMD[DType.float64, W](b[b_offset])
            if b_step == 0
            else b.load[width=W](b_offset + i)
        )
        if op == 0:
            dst.store(i, av + bv)
        elif op == 1:
            dst.store(i, av - bv)
        elif op == 2:
            dst.store(i, av * bv)
        else:
            dst.store(i, av / bv)
        i += W
    while i < stop:
        var av = a[a_offset + i * a_step]
        var bv = b[b_offset + i * b_step]
        if op == 0:
            dst[i] = av + bv
        elif op == 1:
            dst[i] = av - bv
        elif op == 2:
            dst[i] = av * bv
        else:
            dst[i] = av / bv
        i += 1


def binary_f64(
    a: F64,
    a_bitmap: Int,
    a_offset: Int,
    a_step: Int,
    b: F64,
    b_bitmap: Int,
    b_offset: Int,
    b_step: Int,
    n: Int,
    dst: F64,
    dst_bitmap: Bits,
    op: Int,
):
    var dense = a_bitmap == 0 and b_bitmap == 0
    if dense:
        fill_valid(dst_bitmap, n)
    var i = 0
    if dense and op < 4:
        if n >= PARALLEL_THRESHOLD:
            var tasks = (n + PARALLEL_GRAIN - 1) // PARALLEL_GRAIN

            for task in range(tasks):
                var start = task * PARALLEL_GRAIN
                binary_f64_dense_range(
                    a, a_offset, a_step, b, b_offset, b_step,
                    start, min(start + PARALLEL_GRAIN, n), dst, op,
                )
        else:
            binary_f64_dense_range(
                a, a_offset, a_step, b, b_offset, b_step, 0, n, dst, op,
            )
        return
    while i < n:
        var ai = i * a_step
        var bi = i * b_step
        if not dense:
            var ok = valid(a_bitmap, a_offset, ai) and valid(b_bitmap, b_offset, bi)
            set_bit(dst_bitmap, i, ok)
            if not ok:
                dst[i] = 0.0
                i += 1
                continue
        var av = a[a_offset + ai]
        var bv = b[b_offset + bi]
        if op == 0:
            dst[i] = av + bv
        elif op == 1:
            dst[i] = av - bv
        elif op == 2:
            dst[i] = av * bv
        elif op == 3:
            dst[i] = av / bv
        elif op == 4:
            dst[i] = pow(av, bv)
        else:
            dst[i] = atan2(av, bv)
        i += 1


def int_power(base: Int64, exponent: Int64) -> Int64:
    var result = Int64(1)
    var factor = base
    var power_value = exponent
    while power_value > 0:
        if (power_value & 1) != 0:
            result *= factor
        factor *= factor
        power_value >>= 1
    return result


def trunc_div(a: Int64, b: Int64) -> Int64:
    if a == I64_MIN and b == -1:
        return I64_MIN
    var quotient = a // b
    if ((a < 0) != (b < 0)) and a % b != 0:
        quotient += 1
    return quotient


def binary_i64(
    a: I64,
    a_bitmap: Int,
    a_offset: Int,
    a_step: Int,
    b: I64,
    b_bitmap: Int,
    b_offset: Int,
    b_step: Int,
    n: Int,
    dst: I64,
    dst_bitmap: Bits,
    op: Int,
) -> Int:
    var dense = a_bitmap == 0 and b_bitmap == 0
    if dense:
        fill_valid(dst_bitmap, n)
    var i = 0
    if dense and op < 3:
        comptime WI = simd_width_of[DType.int64]()
        while i + WI <= n:
            var av = (
                SIMD[DType.int64, WI](a[a_offset])
                if a_step == 0
                else a.load[width=WI](a_offset + i)
            )
            var bv = (
                SIMD[DType.int64, WI](b[b_offset])
                if b_step == 0
                else b.load[width=WI](b_offset + i)
            )
            if op == 0:
                dst.store(i, av + bv)
            elif op == 1:
                dst.store(i, av - bv)
            else:
                dst.store(i, av * bv)
            i += WI
    while i < n:
        var ai = i * a_step
        var bi = i * b_step
        if not dense:
            var ok = valid(a_bitmap, a_offset, ai) and valid(b_bitmap, b_offset, bi)
            set_bit(dst_bitmap, i, ok)
            if not ok:
                dst[i] = 0
                i += 1
                continue
        var av = a[a_offset + ai]
        var bv = b[b_offset + bi]
        if op == 0:
            dst[i] = av + bv
        elif op == 1:
            dst[i] = av - bv
        elif op == 2:
            dst[i] = av * bv
        elif op == 3:
            if bv == 0:
                return 1
            dst[i] = trunc_div(av, bv)
        else:
            if bv < 0:
                return 2
            dst[i] = int_power(av, bv)
        i += 1
    return 0


def unary_f64_dense_range(
    values: F64,
    offset: Int,
    start: Int,
    stop: Int,
    dst: F64,
    op: Int,
):
    var i = start
    while i + W <= stop:
        var v = values.load[width=W](offset + i)
        if op == 0:
            dst.store(i, -v)
        elif op == 1:
            dst.store(i, abs(v))
        elif op == 2:
            dst.store(i, sqrt(v))
        elif op == 3:
            dst.store(i, v.ne(v).select(v, exp(v)))
        elif op == 4:
            dst.store(i, v.gt(F64_MAX).select(v, log(v)))
        elif op == 5:
            dst.store(i, log10(v))
        elif op == 6:
            dst.store(i, v.gt(F64_MAX).select(v, log2(v)))
        elif op == 7:
            dst.store(i, sin(v))
        elif op == 8:
            dst.store(i, cos(v))
        elif op == 9:
            dst.store(i, tan(v))
        elif op == 10:
            dst.store(i, asin(v))
        elif op == 11:
            dst.store(i, acos(v))
        else:
            dst.store(i, atan(v))
        i += W
    while i < stop:
        var v = values[offset + i]
        if op == 0:
            dst[i] = -v
        elif op == 1:
            dst[i] = abs(v)
        elif op == 2:
            dst[i] = sqrt(v)
        elif op == 3:
            dst[i] = v if v != v else exp(v)
        elif op == 4:
            dst[i] = v if v > F64_MAX else log(v)
        elif op == 5:
            dst[i] = log10(v)
        elif op == 6:
            dst[i] = v if v > F64_MAX else log2(v)
        elif op == 7:
            dst[i] = sin(v)
        elif op == 8:
            dst[i] = cos(v)
        elif op == 9:
            dst[i] = tan(v)
        elif op == 10:
            dst[i] = asin(v)
        elif op == 11:
            dst[i] = acos(v)
        else:
            dst[i] = atan(v)
        i += 1


def unary_f64(
    values: F64,
    bitmap_addr: Int,
    offset: Int,
    n: Int,
    dst: F64,
    dst_bitmap: Bits,
    op: Int,
):
    copy_validity(bitmap_addr, offset, n, dst_bitmap)
    if bitmap_addr == 0:
        if n >= PARALLEL_THRESHOLD and op >= 3:
            var tasks = (n + PARALLEL_GRAIN - 1) // PARALLEL_GRAIN

            for task in range(tasks):
                var start = task * PARALLEL_GRAIN
                unary_f64_dense_range(
                    values, offset, start, min(start + PARALLEL_GRAIN, n), dst, op
                )
        else:
            unary_f64_dense_range(values, offset, 0, n, dst, op)
        return
    var i = 0
    while i < n:
        if valid(bitmap_addr, offset, i):
            var v = values[offset + i]
            if op == 0:
                dst[i] = -v
            elif op == 1:
                dst[i] = abs(v)
            elif op == 2:
                dst[i] = sqrt(v)
            elif op == 3:
                dst[i] = v if v != v else exp(v)
            elif op == 4:
                dst[i] = v if v > F64_MAX else log(v)
            elif op == 5:
                dst[i] = log10(v)
            elif op == 6:
                dst[i] = v if v > F64_MAX else log2(v)
            elif op == 7:
                dst[i] = sin(v)
            elif op == 8:
                dst[i] = cos(v)
            elif op == 9:
                dst[i] = tan(v)
            elif op == 10:
                dst[i] = asin(v)
            elif op == 11:
                dst[i] = acos(v)
            else:
                dst[i] = atan(v)
        else:
            dst[i] = 0.0
        i += 1


def unary_i64(
    values: I64,
    bitmap_addr: Int,
    offset: Int,
    n: Int,
    dst: I64,
    dst_bitmap: Bits,
    op: Int,
):
    copy_validity(bitmap_addr, offset, n, dst_bitmap)
    for i in range(n):
        if valid(bitmap_addr, offset, i):
            var v = values[offset + i]
            dst[i] = -v if op == 0 else abs(v)
        else:
            dst[i] = 0


def compare_dense_f64_bytes(
    a: F64,
    a_offset: Int,
    a_step: Int,
    b: F64,
    b_offset: Int,
    b_step: Int,
    start_byte: Int,
    stop_byte: Int,
    dst: Bits,
    dst_bitmap: Bits,
    op: Int,
):
    for byte in range(start_byte, stop_byte):
        var packed = UInt8(0)
        var base = byte << 3
        for chunk in range(8 // W):
            var i = base + chunk * W
            var av = (
                SIMD[DType.float64, W](a[a_offset])
                if a_step == 0
                else a.load[width=W](a_offset + i)
            )
            var bv = (
                SIMD[DType.float64, W](b[b_offset])
                if b_step == 0
                else b.load[width=W](b_offset + i)
            )
            var result = av.eq(bv)
            if op == 1:
                result = av.ne(bv)
            elif op == 2:
                result = av.lt(bv)
            elif op == 3:
                result = av.le(bv)
            elif op == 4:
                result = av.gt(bv)
            elif op == 5:
                result = av.ge(bv)
            for lane in range(W):
                if result[lane]:
                    packed |= UInt8(1) << UInt8(chunk * W + lane)
        dst[byte] = packed
        dst_bitmap[byte] = 0xFF


def compare_f64(
    a: F64,
    a_bitmap: Int,
    a_offset: Int,
    a_step: Int,
    b: F64,
    b_bitmap: Int,
    b_offset: Int,
    b_step: Int,
    n: Int,
    dst: Bits,
    dst_bitmap: Bits,
    op: Int,
):
    if a_bitmap == 0 and b_bitmap == 0:
        var full_bytes = n >> 3
        if n >= PARALLEL_THRESHOLD:
            comptime GRAIN_BYTES = PARALLEL_GRAIN // 8
            var tasks = (full_bytes + GRAIN_BYTES - 1) // GRAIN_BYTES

            for task in range(tasks):
                var start = task * GRAIN_BYTES
                compare_dense_f64_bytes(
                    a, a_offset, a_step, b, b_offset, b_step,
                    start, min(start + GRAIN_BYTES, full_bytes),
                    dst, dst_bitmap, op,
                )
        else:
            compare_dense_f64_bytes(
                a, a_offset, a_step, b, b_offset, b_step,
                0, full_bytes, dst, dst_bitmap, op,
            )
        if (n & 7) != 0:
            var packed = UInt8(0)
            var base = full_bytes << 3
            for lane in range(n - base):
                var i = base + lane
                var av = a[a_offset + i * a_step]
                var bv = b[b_offset + i * b_step]
                var result: Bool
                if op == 0:
                    result = av == bv
                elif op == 1:
                    result = av != bv
                elif op == 2:
                    result = av < bv
                elif op == 3:
                    result = av <= bv
                elif op == 4:
                    result = av > bv
                else:
                    result = av >= bv
                if result:
                    packed |= UInt8(1) << UInt8(lane)
            dst[full_bytes] = packed
            dst_bitmap[full_bytes] = 0xFF
        return
    for byte in range((n + 7) >> 3):
        var packed = UInt8(0)
        var packed_valid = UInt8(0)
        var base = byte << 3
        var stop = 8 if base + 8 <= n else n - base
        for lane in range(stop):
            var i = base + lane
            var ai = i * a_step
            var bi = i * b_step
            if not (
                valid(a_bitmap, a_offset, ai) and valid(b_bitmap, b_offset, bi)
            ):
                continue
            packed_valid |= UInt8(1) << UInt8(lane)
            var av = a[a_offset + ai]
            var bv = b[b_offset + bi]
            var result: Bool
            if op == 0:
                result = av == bv
            elif op == 1:
                result = av != bv
            elif op == 2:
                result = av < bv
            elif op == 3:
                result = av <= bv
            elif op == 4:
                result = av > bv
            else:
                result = av >= bv
            if result:
                packed |= UInt8(1) << UInt8(lane)
        dst[byte] = packed
        dst_bitmap[byte] = packed_valid


def compare_dense_i64_bytes(
    a: I64,
    a_offset: Int,
    a_step: Int,
    b: I64,
    b_offset: Int,
    b_step: Int,
    start_byte: Int,
    stop_byte: Int,
    dst: Bits,
    dst_bitmap: Bits,
    op: Int,
):
    comptime WI = simd_width_of[DType.int64]()
    for byte in range(start_byte, stop_byte):
        var packed = UInt8(0)
        var base = byte << 3
        for chunk in range(8 // WI):
            var i = base + chunk * WI
            var av = (
                SIMD[DType.int64, WI](a[a_offset])
                if a_step == 0
                else a.load[width=WI](a_offset + i)
            )
            var bv = (
                SIMD[DType.int64, WI](b[b_offset])
                if b_step == 0
                else b.load[width=WI](b_offset + i)
            )
            var result = av.eq(bv)
            if op == 1:
                result = av.ne(bv)
            elif op == 2:
                result = av.lt(bv)
            elif op == 3:
                result = av.le(bv)
            elif op == 4:
                result = av.gt(bv)
            elif op == 5:
                result = av.ge(bv)
            for lane in range(WI):
                if result[lane]:
                    packed |= UInt8(1) << UInt8(chunk * WI + lane)
        dst[byte] = packed
        dst_bitmap[byte] = 0xFF


def compare_i64(
    a: I64,
    a_bitmap: Int,
    a_offset: Int,
    a_step: Int,
    b: I64,
    b_bitmap: Int,
    b_offset: Int,
    b_step: Int,
    n: Int,
    dst: Bits,
    dst_bitmap: Bits,
    op: Int,
):
    if a_bitmap == 0 and b_bitmap == 0:
        var full_bytes = n >> 3
        if n >= PARALLEL_THRESHOLD:
            comptime GRAIN_BYTES = PARALLEL_GRAIN // 8
            var tasks = (full_bytes + GRAIN_BYTES - 1) // GRAIN_BYTES

            for task in range(tasks):
                var start = task * GRAIN_BYTES
                compare_dense_i64_bytes(
                    a, a_offset, a_step, b, b_offset, b_step,
                    start, min(start + GRAIN_BYTES, full_bytes),
                    dst, dst_bitmap, op,
                )
        else:
            compare_dense_i64_bytes(
                a, a_offset, a_step, b, b_offset, b_step,
                0, full_bytes, dst, dst_bitmap, op,
            )
        if (n & 7) != 0:
            var packed = UInt8(0)
            var base = full_bytes << 3
            for lane in range(n - base):
                var i = base + lane
                var av = a[a_offset + i * a_step]
                var bv = b[b_offset + i * b_step]
                var result: Bool
                if op == 0:
                    result = av == bv
                elif op == 1:
                    result = av != bv
                elif op == 2:
                    result = av < bv
                elif op == 3:
                    result = av <= bv
                elif op == 4:
                    result = av > bv
                else:
                    result = av >= bv
                if result:
                    packed |= UInt8(1) << UInt8(lane)
            dst[full_bytes] = packed
            dst_bitmap[full_bytes] = 0xFF
        return
    for byte in range((n + 7) >> 3):
        var packed = UInt8(0)
        var packed_valid = UInt8(0)
        var base = byte << 3
        var stop = 8 if base + 8 <= n else n - base
        for lane in range(stop):
            var i = base + lane
            var ai = i * a_step
            var bi = i * b_step
            if not (
                valid(a_bitmap, a_offset, ai) and valid(b_bitmap, b_offset, bi)
            ):
                continue
            packed_valid |= UInt8(1) << UInt8(lane)
            var av = a[a_offset + ai]
            var bv = b[b_offset + bi]
            var result: Bool
            if op == 0:
                result = av == bv
            elif op == 1:
                result = av != bv
            elif op == 2:
                result = av < bv
            elif op == 3:
                result = av <= bv
            elif op == 4:
                result = av > bv
            else:
                result = av >= bv
            if result:
                packed |= UInt8(1) << UInt8(lane)
        dst[byte] = packed
        dst_bitmap[byte] = packed_valid


def predicate_f64(
    values: F64,
    bitmap_addr: Int,
    offset: Int,
    n: Int,
    dst: Bits,
    dst_bitmap: Bits,
    op: Int,
):
    for byte in range((n + 7) >> 3):
        var packed = UInt8(0)
        var packed_valid = UInt8(0)
        var base = byte << 3
        var stop = 8 if base + 8 <= n else n - base
        for lane in range(stop):
            var i = base + lane
            if not valid(bitmap_addr, offset, i):
                continue
            packed_valid |= UInt8(1) << UInt8(lane)
            var v = values[offset + i]
            var result: Bool
            if op == 0:
                result = v != v
            elif op == 1:
                result = v == v and v <= F64_MAX and v >= -F64_MAX
            else:
                result = v > F64_MAX or v < -F64_MAX
            if result:
                packed |= UInt8(1) << UInt8(lane)
        dst[byte] = packed
        dst_bitmap[byte] = packed_valid


def sum_f64(values: F64, bitmap_addr: Int, offset: Int, n: Int) -> Float64:
    if bitmap_addr == 0:
        var acc = SIMD[DType.float64, W](0.0)
        var i = 0
        while i + W <= n:
            acc += values.load[width=W](offset + i)
            i += W
        var total = acc.reduce_add()
        while i < n:
            total += values[offset + i]
            i += 1
        return total
    var total = 0.0
    for i in range(n):
        if valid(bitmap_addr, offset, i):
            total += values[offset + i]
    return total


def sum_i64(values: I64, bitmap_addr: Int, offset: Int, n: Int) -> Int64:
    var total = Int64(0)
    for i in range(n):
        if valid(bitmap_addr, offset, i):
            total += values[offset + i]
    return total


def mean_i64(values: I64, bitmap_addr: Int, offset: Int, n: Int) -> Float64:
    comptime WI = simd_width_of[DType.int64]()
    if bitmap_addr == 0:
        var acc = SIMD[DType.float64, WI](0.0)
        var i = 0
        while i + WI <= n:
            acc += values.load[width=WI](offset + i).cast[DType.float64]()
            i += WI
        var total = acc.reduce_add()
        while i < n:
            total += Float64(values[offset + i])
            i += 1
        return total / Float64(n)
    var total = 0.0
    var count = 0.0
    for i in range(n):
        if valid(bitmap_addr, offset, i):
            total += Float64(values[offset + i])
            count += 1.0
    return total / count


def minmax_f64(
    values: F64, bitmap_addr: Int, offset: Int, n: Int, dst: F64
):
    var seen_number = False
    var saw_nan = False
    var lo = 0.0
    var hi = 0.0
    for i in range(n):
        if not valid(bitmap_addr, offset, i):
            continue
        var v = values[offset + i]
        if v != v:
            if not seen_number and not saw_nan:
                lo = v
                hi = v
            saw_nan = True
            continue
        if not seen_number:
            lo = v
            hi = v
            seen_number = True
        else:
            if v < lo:
                lo = v
            if v > hi:
                hi = v
    dst[0] = lo
    dst[1] = hi


def minmax_i64(
    values: I64, bitmap_addr: Int, offset: Int, n: Int, dst: I64
):
    var lo = I64_MAX
    var hi = I64_MIN
    for i in range(n):
        if not valid(bitmap_addr, offset, i):
            continue
        var v = values[offset + i]
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    dst[0] = lo
    dst[1] = hi


def variance_f64(
    values: F64, bitmap_addr: Int, offset: Int, n: Int, ddof: Int
) -> Float64:
    if bitmap_addr == 0:
        var anchor = values[offset]
        if n >= PARALLEL_THRESHOLD:
            var partials = stack_allocation[VARIANCE_TASKS, Float64]()

            for task in range(VARIANCE_TASKS):
                var start = task * n // VARIANCE_TASKS
                var stop = (task + 1) * n // VARIANCE_TASKS
                var sum_delta = SIMD[DType.float64, W](0.0)
                var i = start
                while i + W <= stop:
                    sum_delta += values.load[width=W](offset + i) - anchor
                    i += W
                var total_delta = sum_delta.reduce_add()
                while i < stop:
                    total_delta += values[offset + i] - anchor
                    i += 1
                partials[task] = total_delta
            var total_delta = 0.0
            for task in range(VARIANCE_TASKS):
                total_delta += partials[task]
            var mean = anchor + total_delta / Float64(n)

            for task in range(VARIANCE_TASKS):
                var start = task * n // VARIANCE_TASKS
                var stop = (task + 1) * n // VARIANCE_TASKS
                var sum_squared = SIMD[DType.float64, W](0.0)
                var i = start
                while i + W <= stop:
                    var delta = values.load[width=W](offset + i) - mean
                    sum_squared += delta * delta
                    i += W
                var m2 = sum_squared.reduce_add()
                while i < stop:
                    var delta = values[offset + i] - mean
                    m2 += delta * delta
                    i += 1
                partials[task] = m2
            var m2 = 0.0
            for task in range(VARIANCE_TASKS):
                m2 += partials[task]
            return m2 / Float64(n - ddof)

        var sum_delta = SIMD[DType.float64, W](0.0)
        var i = 0
        while i + W <= n:
            sum_delta += values.load[width=W](offset + i) - anchor
            i += W
        var total_delta = sum_delta.reduce_add()
        while i < n:
            total_delta += values[offset + i] - anchor
            i += 1
        var mean = anchor + total_delta / Float64(n)

        var sum_squared = SIMD[DType.float64, W](0.0)
        i = 0
        while i + W <= n:
            var delta = values.load[width=W](offset + i) - mean
            sum_squared += delta * delta
            i += W
        var m2 = sum_squared.reduce_add()
        while i < n:
            var delta = values[offset + i] - mean
            m2 += delta * delta
            i += 1
        return m2 / Float64(n - ddof)

    var count = 0.0
    var mean = 0.0
    var m2 = 0.0
    # Welford's recurrence avoids the catastrophic cancellation in
    # sum(x*x) - sum(x)**2 / n for data with a large common offset.
    for i in range(n):
        if not valid(bitmap_addr, offset, i):
            continue
        var v = values[offset + i]
        count += 1.0
        var delta = v - mean
        mean += delta / count
        var delta2 = v - mean
        m2 += delta * delta2
    return m2 / (count - Float64(ddof))


def cumulative_f64(
    values: F64,
    bitmap_addr: Int,
    offset: Int,
    n: Int,
    start: Float64,
    skip_nulls: Bool,
    dst: F64,
    dst_bitmap: Bits,
):
    var total = start
    var poisoned = False
    for i in range(n):
        if not valid(bitmap_addr, offset, i):
            set_bit(dst_bitmap, i, False)
            poisoned = poisoned or not skip_nulls
        elif poisoned:
            set_bit(dst_bitmap, i, False)
        else:
            total += values[offset + i]
            dst[i] = total
            set_bit(dst_bitmap, i, True)


def cumulative_i64(
    values: I64,
    bitmap_addr: Int,
    offset: Int,
    n: Int,
    start: Int64,
    skip_nulls: Bool,
    dst: I64,
    dst_bitmap: Bits,
):
    var total = start
    var poisoned = False
    for i in range(n):
        if not valid(bitmap_addr, offset, i):
            set_bit(dst_bitmap, i, False)
            poisoned = poisoned or not skip_nulls
        elif poisoned:
            set_bit(dst_bitmap, i, False)
        else:
            total += values[offset + i]
            dst[i] = total
            set_bit(dst_bitmap, i, True)


def cast_i64_f64(
    values: I64, bitmap_addr: Int, offset: Int, n: Int,
    dst: F64, dst_bitmap: Bits,
):
    copy_validity(bitmap_addr, offset, n, dst_bitmap)
    var i = 0
    if bitmap_addr == 0:
        comptime WI = simd_width_of[DType.int64]()
        while i + WI <= n:
            dst.store(
                i,
                values.load[width=WI](offset + i).cast[DType.float64](),
            )
            i += WI
    while i < n:
        if valid(bitmap_addr, offset, i):
            dst[i] = Float64(values[offset + i])
        else:
            dst[i] = 0.0
        i += 1


def if_else_dense_f64_bytes(
    cond: Bits,
    cond_offset: Int,
    cond_step: Int,
    left: F64,
    left_offset: Int,
    left_step: Int,
    right: F64,
    right_offset: Int,
    right_step: Int,
    start_byte: Int,
    stop_byte: Int,
    dst: F64,
    dst_bitmap: Bits,
):
    for byte in range(start_byte, stop_byte):
        var base = byte << 3
        for lane in range(8):
            var i = base + lane
            var choose_left = get_bit(cond, cond_offset + i * cond_step)
            dst[i] = (
                left[left_offset + i * left_step]
                if choose_left
                else right[right_offset + i * right_step]
            )
        dst_bitmap[byte] = 0xFF


def if_else_f64(
    cond: Bits,
    cond_bitmap: Int,
    cond_offset: Int,
    cond_step: Int,
    left: F64,
    left_bitmap: Int,
    left_offset: Int,
    left_step: Int,
    right: F64,
    right_bitmap: Int,
    right_offset: Int,
    right_step: Int,
    n: Int,
    dst: F64,
    dst_bitmap: Bits,
):
    if cond_bitmap == 0 and left_bitmap == 0 and right_bitmap == 0:
        var full_bytes = n >> 3
        if n >= PARALLEL_THRESHOLD:
            comptime GRAIN_BYTES = PARALLEL_GRAIN // 8
            var tasks = (full_bytes + GRAIN_BYTES - 1) // GRAIN_BYTES

            for task in range(tasks):
                var start = task * GRAIN_BYTES
                if_else_dense_f64_bytes(
                    cond, cond_offset, cond_step,
                    left, left_offset, left_step,
                    right, right_offset, right_step,
                    start, min(start + GRAIN_BYTES, full_bytes),
                    dst, dst_bitmap,
                )
        else:
            if_else_dense_f64_bytes(
                cond, cond_offset, cond_step,
                left, left_offset, left_step,
                right, right_offset, right_step,
                0, full_bytes, dst, dst_bitmap,
            )
        var start = full_bytes << 3
        for i in range(start, n):
            var choose_left = get_bit(cond, cond_offset + i * cond_step)
            dst[i] = (
                left[left_offset + i * left_step]
                if choose_left
                else right[right_offset + i * right_step]
            )
            set_bit(dst_bitmap, i, True)
        return
    for i in range(n):
        var cond_i = i * cond_step
        if not valid(cond_bitmap, cond_offset, cond_i):
            set_bit(dst_bitmap, i, False)
            continue
        var choose_left = get_bit(cond, cond_offset + cond_i)
        var step = left_step if choose_left else right_step
        var source_bitmap = left_bitmap if choose_left else right_bitmap
        var source_offset = left_offset if choose_left else right_offset
        var source_i = i * step
        if not valid(source_bitmap, source_offset, source_i):
            set_bit(dst_bitmap, i, False)
            continue
        dst[i] = (
            left[source_offset + source_i]
            if choose_left
            else right[source_offset + source_i]
        )
        set_bit(dst_bitmap, i, True)


def if_else_dense_i64_bytes(
    cond: Bits,
    cond_offset: Int,
    cond_step: Int,
    left: I64,
    left_offset: Int,
    left_step: Int,
    right: I64,
    right_offset: Int,
    right_step: Int,
    start_byte: Int,
    stop_byte: Int,
    dst: I64,
    dst_bitmap: Bits,
):
    for byte in range(start_byte, stop_byte):
        var base = byte << 3
        for lane in range(8):
            var i = base + lane
            var choose_left = get_bit(cond, cond_offset + i * cond_step)
            dst[i] = (
                left[left_offset + i * left_step]
                if choose_left
                else right[right_offset + i * right_step]
            )
        dst_bitmap[byte] = 0xFF


def if_else_i64(
    cond: Bits,
    cond_bitmap: Int,
    cond_offset: Int,
    cond_step: Int,
    left: I64,
    left_bitmap: Int,
    left_offset: Int,
    left_step: Int,
    right: I64,
    right_bitmap: Int,
    right_offset: Int,
    right_step: Int,
    n: Int,
    dst: I64,
    dst_bitmap: Bits,
):
    if cond_bitmap == 0 and left_bitmap == 0 and right_bitmap == 0:
        var full_bytes = n >> 3
        if n >= PARALLEL_THRESHOLD:
            comptime GRAIN_BYTES = PARALLEL_GRAIN // 8
            var tasks = (full_bytes + GRAIN_BYTES - 1) // GRAIN_BYTES

            for task in range(tasks):
                var start = task * GRAIN_BYTES
                if_else_dense_i64_bytes(
                    cond, cond_offset, cond_step,
                    left, left_offset, left_step,
                    right, right_offset, right_step,
                    start, min(start + GRAIN_BYTES, full_bytes),
                    dst, dst_bitmap,
                )
        else:
            if_else_dense_i64_bytes(
                cond, cond_offset, cond_step,
                left, left_offset, left_step,
                right, right_offset, right_step,
                0, full_bytes, dst, dst_bitmap,
            )
        var start = full_bytes << 3
        for i in range(start, n):
            var choose_left = get_bit(cond, cond_offset + i * cond_step)
            dst[i] = (
                left[left_offset + i * left_step]
                if choose_left
                else right[right_offset + i * right_step]
            )
            set_bit(dst_bitmap, i, True)
        return
    for i in range(n):
        var cond_i = i * cond_step
        if not valid(cond_bitmap, cond_offset, cond_i):
            set_bit(dst_bitmap, i, False)
            continue
        var choose_left = get_bit(cond, cond_offset + cond_i)
        var step = left_step if choose_left else right_step
        var source_bitmap = left_bitmap if choose_left else right_bitmap
        var source_offset = left_offset if choose_left else right_offset
        var source_i = i * step
        if not valid(source_bitmap, source_offset, source_i):
            set_bit(dst_bitmap, i, False)
            continue
        dst[i] = (
            left[source_offset + source_i]
            if choose_left
            else right[source_offset + source_i]
        )
        set_bit(dst_bitmap, i, True)


def fill_null_f64(
    values: F64,
    bitmap_addr: Int,
    offset: Int,
    fill: F64,
    fill_bitmap: Int,
    fill_offset: Int,
    fill_step: Int,
    n: Int,
    dst: F64,
    dst_bitmap: Bits,
):
    for i in range(n):
        if valid(bitmap_addr, offset, i):
            dst[i] = values[offset + i]
            set_bit(dst_bitmap, i, True)
        else:
            var fi = i * fill_step
            var ok = valid(fill_bitmap, fill_offset, fi)
            set_bit(dst_bitmap, i, ok)
            if ok:
                dst[i] = fill[fill_offset + fi]


def fill_null_i64(
    values: I64,
    bitmap_addr: Int,
    offset: Int,
    fill: I64,
    fill_bitmap: Int,
    fill_offset: Int,
    fill_step: Int,
    n: Int,
    dst: I64,
    dst_bitmap: Bits,
):
    for i in range(n):
        if valid(bitmap_addr, offset, i):
            dst[i] = values[offset + i]
            set_bit(dst_bitmap, i, True)
        else:
            var fi = i * fill_step
            var ok = valid(fill_bitmap, fill_offset, fi)
            set_bit(dst_bitmap, i, ok)
            if ok:
                dst[i] = fill[fill_offset + fi]


def filter_f64(
    values: F64,
    bitmap_addr: Int,
    offset: Int,
    n: Int,
    selection: Bits,
    selection_bitmap: Int,
    selection_offset: Int,
    emit_null: Bool,
    dst: F64,
    dst_bitmap: Bits,
) -> Int:
    var kept = 0
    if (
        bitmap_addr == 0
        and selection_bitmap == 0
        and (selection_offset & 7) == 0
    ):
        fill_valid(dst_bitmap, n)
        var byte_offset = selection_offset >> 3
        var byte_count = (n + 7) >> 3
        for byte in range(byte_count):
            var selected = selection[byte_offset + byte]
            if byte == byte_count - 1 and (n & 7) != 0:
                selected &= (UInt8(1) << UInt8(n & 7)) - 1
            while selected != 0:
                var lane = Int(count_trailing_zeros(selected))
                var i = (byte << 3) + lane
                dst[kept] = values[offset + i]
                kept += 1
                selected &= selected - 1
        return kept
    for i in range(n):
        if not valid(selection_bitmap, selection_offset, i):
            if emit_null:
                set_bit(dst_bitmap, kept, False)
                kept += 1
            continue
        if not get_bit(selection, selection_offset + i):
            continue
        var ok = valid(bitmap_addr, offset, i)
        set_bit(dst_bitmap, kept, ok)
        if ok:
            dst[kept] = values[offset + i]
        kept += 1
    return kept


def filter_i64(
    values: I64,
    bitmap_addr: Int,
    offset: Int,
    n: Int,
    selection: Bits,
    selection_bitmap: Int,
    selection_offset: Int,
    emit_null: Bool,
    dst: I64,
    dst_bitmap: Bits,
) -> Int:
    var kept = 0
    if (
        bitmap_addr == 0
        and selection_bitmap == 0
        and (selection_offset & 7) == 0
    ):
        fill_valid(dst_bitmap, n)
        var byte_offset = selection_offset >> 3
        var byte_count = (n + 7) >> 3
        for byte in range(byte_count):
            var selected = selection[byte_offset + byte]
            if byte == byte_count - 1 and (n & 7) != 0:
                selected &= (UInt8(1) << UInt8(n & 7)) - 1
            while selected != 0:
                var lane = Int(count_trailing_zeros(selected))
                var i = (byte << 3) + lane
                dst[kept] = values[offset + i]
                kept += 1
                selected &= selected - 1
        return kept
    for i in range(n):
        if not valid(selection_bitmap, selection_offset, i):
            if emit_null:
                set_bit(dst_bitmap, kept, False)
                kept += 1
            continue
        if not get_bit(selection, selection_offset + i):
            continue
        var ok = valid(bitmap_addr, offset, i)
        set_bit(dst_bitmap, kept, ok)
        if ok:
            dst[kept] = values[offset + i]
        kept += 1
    return kept


@export("mpa_binary_f64")
def mpa_binary_f64(
    a: Int, a_bitmap: Int, a_offset: Int, a_step: Int,
    b: Int, b_bitmap: Int, b_offset: Int, b_step: Int,
    n: Int, dst: Int, dst_bitmap: Int, op: Int,
) abi("C"):
    binary_f64(
        fp(a), a_bitmap, a_offset, a_step, fp(b), b_bitmap, b_offset, b_step,
        n, fp(dst), bp(dst_bitmap), op,
    )


@export("mpa_binary_i64")
def mpa_binary_i64(
    a: Int, a_bitmap: Int, a_offset: Int, a_step: Int,
    b: Int, b_bitmap: Int, b_offset: Int, b_step: Int,
    n: Int, dst: Int, dst_bitmap: Int, op: Int,
) abi("C") -> Int:
    return binary_i64(
        ip(a), a_bitmap, a_offset, a_step, ip(b), b_bitmap, b_offset, b_step,
        n, ip(dst), bp(dst_bitmap), op,
    )


@export("mpa_unary_f64")
def mpa_unary_f64(
    values: Int, bitmap: Int, offset: Int, n: Int,
    dst: Int, dst_bitmap: Int, op: Int,
) abi("C"):
    unary_f64(fp(values), bitmap, offset, n, fp(dst), bp(dst_bitmap), op)


@export("mpa_unary_i64")
def mpa_unary_i64(
    values: Int, bitmap: Int, offset: Int, n: Int,
    dst: Int, dst_bitmap: Int, op: Int,
) abi("C"):
    unary_i64(ip(values), bitmap, offset, n, ip(dst), bp(dst_bitmap), op)


@export("mpa_compare_f64")
def mpa_compare_f64(
    a: Int, a_bitmap: Int, a_offset: Int, a_step: Int,
    b: Int, b_bitmap: Int, b_offset: Int, b_step: Int,
    n: Int, dst: Int, dst_bitmap: Int, op: Int,
) abi("C"):
    compare_f64(
        fp(a), a_bitmap, a_offset, a_step, fp(b), b_bitmap, b_offset, b_step,
        n, bp(dst), bp(dst_bitmap), op,
    )


@export("mpa_compare_i64")
def mpa_compare_i64(
    a: Int, a_bitmap: Int, a_offset: Int, a_step: Int,
    b: Int, b_bitmap: Int, b_offset: Int, b_step: Int,
    n: Int, dst: Int, dst_bitmap: Int, op: Int,
) abi("C"):
    compare_i64(
        ip(a), a_bitmap, a_offset, a_step, ip(b), b_bitmap, b_offset, b_step,
        n, bp(dst), bp(dst_bitmap), op,
    )


@export("mpa_predicate_f64")
def mpa_predicate_f64(
    values: Int, bitmap: Int, offset: Int, n: Int,
    dst: Int, dst_bitmap: Int, op: Int,
) abi("C"):
    predicate_f64(fp(values), bitmap, offset, n, bp(dst), bp(dst_bitmap), op)


@export("mpa_sum_f64")
def mpa_sum_f64(values: Int, bitmap: Int, offset: Int, n: Int) abi("C") -> Float64:
    return sum_f64(fp(values), bitmap, offset, n)


@export("mpa_sum_i64")
def mpa_sum_i64(values: Int, bitmap: Int, offset: Int, n: Int) abi("C") -> Int64:
    return sum_i64(ip(values), bitmap, offset, n)


@export("mpa_mean_i64")
def mpa_mean_i64(values: Int, bitmap: Int, offset: Int, n: Int) abi("C") -> Float64:
    return mean_i64(ip(values), bitmap, offset, n)


@export("mpa_minmax_f64")
def mpa_minmax_f64(
    values: Int, bitmap: Int, offset: Int, n: Int, dst: Int
) abi("C"):
    minmax_f64(fp(values), bitmap, offset, n, fp(dst))


@export("mpa_minmax_i64")
def mpa_minmax_i64(
    values: Int, bitmap: Int, offset: Int, n: Int, dst: Int
) abi("C"):
    minmax_i64(ip(values), bitmap, offset, n, ip(dst))


@export("mpa_variance_f64")
def mpa_variance_f64(
    values: Int, bitmap: Int, offset: Int, n: Int, ddof: Int
) abi("C") -> Float64:
    return variance_f64(fp(values), bitmap, offset, n, ddof)


@export("mpa_cumulative_f64")
def mpa_cumulative_f64(
    values: Int, bitmap: Int, offset: Int, n: Int, start: Float64,
    skip_nulls: Int, dst: Int, dst_bitmap: Int,
) abi("C"):
    cumulative_f64(
        fp(values), bitmap, offset, n, start, skip_nulls != 0,
        fp(dst), bp(dst_bitmap),
    )


@export("mpa_cumulative_i64")
def mpa_cumulative_i64(
    values: Int, bitmap: Int, offset: Int, n: Int, start: Int64,
    skip_nulls: Int, dst: Int, dst_bitmap: Int,
) abi("C"):
    cumulative_i64(
        ip(values), bitmap, offset, n, start, skip_nulls != 0,
        ip(dst), bp(dst_bitmap),
    )


@export("mpa_cast_i64_f64")
def mpa_cast_i64_f64(
    values: Int, bitmap: Int, offset: Int, n: Int,
    dst: Int, dst_bitmap: Int,
) abi("C"):
    cast_i64_f64(
        ip(values), bitmap, offset, n, fp(dst), bp(dst_bitmap)
    )


@export("mpa_if_else_f64")
def mpa_if_else_f64(
    cond: Int, cond_bitmap: Int, cond_offset: Int, cond_step: Int,
    left: Int, left_bitmap: Int, left_offset: Int, left_step: Int,
    right: Int, right_bitmap: Int, right_offset: Int, right_step: Int,
    n: Int, dst: Int, dst_bitmap: Int,
) abi("C"):
    if_else_f64(
        bp(cond), cond_bitmap, cond_offset, cond_step,
        fp(left), left_bitmap, left_offset, left_step,
        fp(right), right_bitmap, right_offset, right_step,
        n, fp(dst), bp(dst_bitmap),
    )


@export("mpa_if_else_i64")
def mpa_if_else_i64(
    cond: Int, cond_bitmap: Int, cond_offset: Int, cond_step: Int,
    left: Int, left_bitmap: Int, left_offset: Int, left_step: Int,
    right: Int, right_bitmap: Int, right_offset: Int, right_step: Int,
    n: Int, dst: Int, dst_bitmap: Int,
) abi("C"):
    if_else_i64(
        bp(cond), cond_bitmap, cond_offset, cond_step,
        ip(left), left_bitmap, left_offset, left_step,
        ip(right), right_bitmap, right_offset, right_step,
        n, ip(dst), bp(dst_bitmap),
    )


@export("mpa_fill_null_f64")
def mpa_fill_null_f64(
    values: Int, bitmap: Int, offset: Int,
    fill: Int, fill_bitmap: Int, fill_offset: Int, fill_step: Int,
    n: Int, dst: Int, dst_bitmap: Int,
) abi("C"):
    fill_null_f64(
        fp(values), bitmap, offset, fp(fill), fill_bitmap, fill_offset, fill_step,
        n, fp(dst), bp(dst_bitmap),
    )


@export("mpa_fill_null_i64")
def mpa_fill_null_i64(
    values: Int, bitmap: Int, offset: Int,
    fill: Int, fill_bitmap: Int, fill_offset: Int, fill_step: Int,
    n: Int, dst: Int, dst_bitmap: Int,
) abi("C"):
    fill_null_i64(
        ip(values), bitmap, offset, ip(fill), fill_bitmap, fill_offset, fill_step,
        n, ip(dst), bp(dst_bitmap),
    )


@export("mpa_filter_f64")
def mpa_filter_f64(
    values: Int, bitmap: Int, offset: Int, n: Int,
    selection: Int, selection_bitmap: Int, selection_offset: Int,
    emit_null: Int, dst: Int, dst_bitmap: Int,
) abi("C") -> Int:
    return filter_f64(
        fp(values), bitmap, offset, n,
        bp(selection), selection_bitmap, selection_offset, emit_null != 0,
        fp(dst), bp(dst_bitmap),
    )


@export("mpa_filter_i64")
def mpa_filter_i64(
    values: Int, bitmap: Int, offset: Int, n: Int,
    selection: Int, selection_bitmap: Int, selection_offset: Int,
    emit_null: Int, dst: Int, dst_bitmap: Int,
) abi("C") -> Int:
    return filter_i64(
        ip(values), bitmap, offset, n,
        bp(selection), selection_bitmap, selection_offset, emit_null != 0,
        ip(dst), bp(dst_bitmap),
    )
