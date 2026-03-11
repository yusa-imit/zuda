const std = @import("std");
const testing = std.testing;

/// Computes the Greatest Common Divisor (GCD) of two integers using Euclid's algorithm.
/// Time: O(log(min(a, b)))
/// Space: O(1)
pub fn gcd(comptime T: type, a: T, b: T) T {
    const info = @typeInfo(T);
    if (info != .int) @compileError("gcd requires integer type");

    // Handle negative numbers by taking absolute values
    var x = if (a < 0) -a else a;
    var y = if (b < 0) -b else b;

    // Ensure x >= y for efficiency
    if (x < y) {
        const tmp = x;
        x = y;
        y = tmp;
    }

    // Euclid's algorithm
    while (y != 0) {
        const remainder = @mod(x, y);
        x = y;
        y = remainder;
    }

    return x;
}

/// Computes the Least Common Multiple (LCM) of two integers.
/// LCM(a, b) = |a * b| / GCD(a, b)
/// Time: O(log(min(a, b)))
/// Space: O(1)
pub fn lcm(comptime T: type, a: T, b: T) !T {
    const info = @typeInfo(T);
    if (info != .int) @compileError("lcm requires integer type");

    if (a == 0 or b == 0) return 0;

    const g = gcd(T, a, b);
    const abs_a = if (a < 0) -a else a;
    const abs_b = if (b < 0) -b else b;

    // Divide first to avoid overflow: LCM = (a / gcd) * b
    const a_div_g = @divExact(abs_a, g);

    // Check for potential overflow
    const result = if (@typeInfo(T).int.bits <= 64)
        a_div_g *% abs_b // Use wrapping multiplication for detection
    else
        a_div_g * abs_b;

    // For smaller integer types, check if overflow occurred
    if (@typeInfo(T).int.bits <= 64) {
        const check = @divTrunc(result, abs_b);
        if (check != a_div_g) return error.Overflow;
    }

    return result;
}

/// Computes the Extended Euclidean Algorithm, finding GCD and Bézout coefficients.
/// Returns: gcd(a, b), and coefficients x, y such that ax + by = gcd(a, b)
/// Time: O(log(min(a, b)))
/// Space: O(1)
pub fn extendedGcd(comptime T: type, a: T, b: T) struct { gcd: T, x: T, y: T } {
    const info = @typeInfo(T);
    if (info != .int) @compileError("extendedGcd requires integer type");
    if (info.int.signedness != .signed) @compileError("extendedGcd requires signed integer type");

    if (b == 0) {
        return .{
            .gcd = if (a < 0) -a else a,
            .x = if (a < 0) -1 else 1,
            .y = 0,
        };
    }

    var old_r: T = a;
    var r: T = b;
    var old_s: T = 1;
    var s: T = 0;
    var old_t: T = 0;
    var t: T = 1;

    while (r != 0) {
        const quotient = @divTrunc(old_r, r);

        const new_r = old_r - quotient * r;
        old_r = r;
        r = new_r;

        const new_s = old_s - quotient * s;
        old_s = s;
        s = new_s;

        const new_t = old_t - quotient * t;
        old_t = t;
        t = new_t;
    }

    return .{
        .gcd = if (old_r < 0) -old_r else old_r,
        .x = if (old_r < 0) -old_s else old_s,
        .y = if (old_r < 0) -old_t else old_t,
    };
}

/// Computes the binary GCD (Stein's algorithm), which uses bitwise operations.
/// More efficient on systems where division is expensive.
/// Time: O(log(min(a, b)))
/// Space: O(1)
pub fn binaryGcd(comptime T: type, a: T, b: T) T {
    const info = @typeInfo(T);
    if (info != .int) @compileError("binaryGcd requires integer type");
    if (info.int.signedness == .signed) @compileError("binaryGcd requires unsigned integer type");

    if (a == 0) return b;
    if (b == 0) return a;

    var x = a;
    var y = b;

    // Find common factors of 2
    const ShiftT = std.math.Log2Int(T);
    const shift: ShiftT = @intCast(@ctz(x | y));

    // Divide x by 2 until odd
    x >>= @as(ShiftT, @intCast(@ctz(x)));

    while (y != 0) {
        // Divide y by 2 until odd
        y >>= @as(ShiftT, @intCast(@ctz(y)));

        // Ensure x <= y
        if (x > y) {
            const tmp = x;
            x = y;
            y = tmp;
        }

        y -= x;
    }

    return x << shift;
}

// ============================================================================
// Tests
// ============================================================================

test "gcd - basic cases" {
    try testing.expectEqual(@as(i32, 1), gcd(i32, 1, 1));
    try testing.expectEqual(@as(i32, 5), gcd(i32, 10, 5));
    try testing.expectEqual(@as(i32, 6), gcd(i32, 48, 18));
    try testing.expectEqual(@as(i32, 1), gcd(i32, 17, 19));
    try testing.expectEqual(@as(i32, 21), gcd(i32, 1071, 462));
}

test "gcd - negative numbers" {
    try testing.expectEqual(@as(i32, 5), gcd(i32, -10, 5));
    try testing.expectEqual(@as(i32, 5), gcd(i32, 10, -5));
    try testing.expectEqual(@as(i32, 5), gcd(i32, -10, -5));
}

test "gcd - zero" {
    try testing.expectEqual(@as(i32, 5), gcd(i32, 0, 5));
    try testing.expectEqual(@as(i32, 5), gcd(i32, 5, 0));
    try testing.expectEqual(@as(i32, 0), gcd(i32, 0, 0));
}

test "gcd - large numbers" {
    try testing.expectEqual(@as(i64, 9), gcd(i64, 123456789, 987654321));
}

test "lcm - basic cases" {
    try testing.expectEqual(@as(i32, 1), try lcm(i32, 1, 1));
    try testing.expectEqual(@as(i32, 10), try lcm(i32, 2, 5));
    try testing.expectEqual(@as(i32, 12), try lcm(i32, 4, 6));
    try testing.expectEqual(@as(i32, 21), try lcm(i32, 3, 7));
}

test "lcm - negative numbers" {
    try testing.expectEqual(@as(i32, 10), try lcm(i32, -2, 5));
    try testing.expectEqual(@as(i32, 10), try lcm(i32, 2, -5));
    try testing.expectEqual(@as(i32, 10), try lcm(i32, -2, -5));
}

test "lcm - zero" {
    try testing.expectEqual(@as(i32, 0), try lcm(i32, 0, 5));
    try testing.expectEqual(@as(i32, 0), try lcm(i32, 5, 0));
    try testing.expectEqual(@as(i32, 0), try lcm(i32, 0, 0));
}

test "lcm - coprime numbers" {
    try testing.expectEqual(@as(i32, 323), try lcm(i32, 17, 19));
}

test "extendedGcd - basic cases" {
    {
        const result = extendedGcd(i32, 48, 18);
        try testing.expectEqual(@as(i32, 6), result.gcd);
        // Verify: 48*x + 18*y = 6
        try testing.expectEqual(@as(i32, 6), 48 * result.x + 18 * result.y);
    }
    {
        const result = extendedGcd(i32, 1071, 462);
        try testing.expectEqual(@as(i32, 21), result.gcd);
        try testing.expectEqual(@as(i32, 21), 1071 * result.x + 462 * result.y);
    }
}

test "extendedGcd - zero" {
    {
        const result = extendedGcd(i32, 5, 0);
        try testing.expectEqual(@as(i32, 5), result.gcd);
        try testing.expectEqual(@as(i32, 1), result.x);
        try testing.expectEqual(@as(i32, 0), result.y);
    }
}

test "extendedGcd - negative numbers" {
    {
        const result = extendedGcd(i32, -48, 18);
        try testing.expectEqual(@as(i32, 6), result.gcd);
        try testing.expectEqual(@as(i32, 6), -48 * result.x + 18 * result.y);
    }
}

test "binaryGcd - basic cases" {
    try testing.expectEqual(@as(u32, 1), binaryGcd(u32, 1, 1));
    try testing.expectEqual(@as(u32, 5), binaryGcd(u32, 10, 5));
    try testing.expectEqual(@as(u32, 6), binaryGcd(u32, 48, 18));
    try testing.expectEqual(@as(u32, 1), binaryGcd(u32, 17, 19));
    try testing.expectEqual(@as(u32, 21), binaryGcd(u32, 1071, 462));
}

test "binaryGcd - zero" {
    try testing.expectEqual(@as(u32, 5), binaryGcd(u32, 0, 5));
    try testing.expectEqual(@as(u32, 5), binaryGcd(u32, 5, 0));
    try testing.expectEqual(@as(u32, 0), binaryGcd(u32, 0, 0));
}

test "binaryGcd - powers of two" {
    try testing.expectEqual(@as(u32, 8), binaryGcd(u32, 16, 24));
    try testing.expectEqual(@as(u32, 32), binaryGcd(u32, 64, 96));
}

test "binaryGcd - large numbers" {
    try testing.expectEqual(@as(u64, 9), binaryGcd(u64, 123456789, 987654321));
}

test "gcd vs binaryGcd - consistency" {
    const values = [_]u32{ 1, 2, 5, 10, 17, 48, 100, 1071, 123456 };
    for (values) |a| {
        for (values) |b| {
            const g1 = gcd(i32, @intCast(a), @intCast(b));
            const g2 = binaryGcd(u32, a, b);
            try testing.expectEqual(g1, @as(i32, @intCast(g2)));
        }
    }
}
