//! Greatest Common Divisor (GCD) and Extended Euclidean Algorithm
//!
//! Fundamental number theory algorithms for computing GCD and solving
//! linear Diophantine equations.

const std = @import("std");
const testing = std.testing;

/// Result of extended Euclidean algorithm
pub fn ExtendedGCDResult(comptime T: type) type {
    return struct {
        /// GCD of a and b
        gcd: T,
        /// Coefficient x such that ax + by = gcd(a,b)
        x: T,
        /// Coefficient y such that ax + by = gcd(a,b)
        y: T,
    };
}

/// Compute the greatest common divisor of two integers using Euclidean algorithm.
///
/// Time: O(log min(a,b))
/// Space: O(1)
///
/// Example:
/// ```zig
/// const g = gcd(u64, 48, 18); // 6
/// ```
pub fn gcd(comptime T: type, a: T, b: T) T {
    var x = if (a < 0) -a else a;
    var y = if (b < 0) -b else b;

    while (y != 0) {
        const temp = y;
        y = @mod(x, y);
        x = temp;
    }

    return x;
}

/// Compute the least common multiple of two integers.
///
/// Time: O(log min(a,b))
/// Space: O(1)
///
/// Example:
/// ```zig
/// const l = lcm(u64, 12, 18); // 36
/// ```
pub fn lcm(comptime T: type, a: T, b: T) T {
    if (a == 0 or b == 0) return 0;
    const g = gcd(T, a, b);
    const abs_a = if (a < 0) -a else a;
    const abs_b = if (b < 0) -b else b;
    return @divExact(abs_a, g) * abs_b;
}

/// Extended Euclidean algorithm: find integers x, y such that ax + by = gcd(a,b).
///
/// This is useful for:
/// - Finding modular inverses
/// - Solving linear Diophantine equations
/// - Chinese Remainder Theorem
///
/// Time: O(log min(a,b))
/// Space: O(1)
///
/// Example:
/// ```zig
/// const result = extendedGcd(i64, 35, 15);
/// // result.gcd = 5
/// // 35 * result.x + 15 * result.y = 5
/// ```
pub fn extendedGcd(comptime T: type, a: T, b: T) ExtendedGCDResult(T) {
    if (b == 0) {
        return .{ .gcd = a, .x = 1, .y = 0 };
    }

    var old_r: T = a;
    var r: T = b;
    var old_s: T = 1;
    var s: T = 0;
    var old_t: T = 0;
    var t: T = 1;

    while (r != 0) {
        const quotient = @divFloor(old_r, r);

        const temp_r = r;
        r = old_r - quotient * r;
        old_r = temp_r;

        const temp_s = s;
        s = old_s - quotient * s;
        old_s = temp_s;

        const temp_t = t;
        t = old_t - quotient * t;
        old_t = temp_t;
    }

    return .{ .gcd = old_r, .x = old_s, .y = old_t };
}

/// Compute the modular multiplicative inverse of a modulo m.
/// Returns null if the inverse does not exist (when gcd(a,m) != 1).
///
/// Time: O(log m)
/// Space: O(1)
///
/// Example:
/// ```zig
/// const inv = modInverse(i64, 3, 11); // Some(4) because 3*4 ≡ 1 (mod 11)
/// const no_inv = modInverse(i64, 6, 9); // null because gcd(6,9) = 3 ≠ 1
/// ```
pub fn modInverse(comptime T: type, a: T, m: T) ?T {
    const result = extendedGcd(T, a, m);
    if (result.gcd != 1) {
        return null; // Inverse doesn't exist
    }

    // Normalize x to be in range [0, m)
    return @mod(result.x, m);
}

/// Solve the linear Diophantine equation ax + by = c.
/// Returns a solution (x, y) if it exists, null otherwise.
///
/// A solution exists if and only if gcd(a,b) divides c.
/// If a solution exists, there are infinitely many solutions of the form:
/// x' = x + k*(b/gcd), y' = y - k*(a/gcd) for any integer k.
///
/// Time: O(log min(a,b))
/// Space: O(1)
///
/// Example:
/// ```zig
/// const sol = solveDiophantine(i64, 35, 15, 5); // Some solution
/// const no_sol = solveDiophantine(i64, 35, 15, 6); // null (gcd(35,15)=5 doesn't divide 6)
/// ```
pub fn solveDiophantine(comptime T: type, a: T, b: T, c: T) ?struct { x: T, y: T } {
    const result = extendedGcd(T, a, b);

    // Check if solution exists
    if (@mod(c, result.gcd) != 0) {
        return null;
    }

    // Scale the solution
    const factor = @divExact(c, result.gcd);
    return .{
        .x = result.x * factor,
        .y = result.y * factor,
    };
}

// Tests

test "gcd - basic cases" {
    try testing.expectEqual(@as(u64, 6), gcd(u64, 48, 18));
    try testing.expectEqual(@as(u64, 1), gcd(u64, 17, 19));
    try testing.expectEqual(@as(u64, 5), gcd(u64, 35, 15));
    try testing.expectEqual(@as(u64, 12), gcd(u64, 0, 12));
    try testing.expectEqual(@as(u64, 15), gcd(u64, 15, 0));
}

test "gcd - edge cases" {
    try testing.expectEqual(@as(u64, 7), gcd(u64, 7, 7));
    try testing.expectEqual(@as(u64, 1), gcd(u64, 1, 100));
}

test "gcd - signed integers" {
    try testing.expectEqual(@as(i64, 6), gcd(i64, -48, 18));
    try testing.expectEqual(@as(i64, 6), gcd(i64, 48, -18));
    try testing.expectEqual(@as(i64, 6), gcd(i64, -48, -18));
}

test "lcm - basic cases" {
    try testing.expectEqual(@as(u64, 36), lcm(u64, 12, 18));
    try testing.expectEqual(@as(u64, 323), lcm(u64, 17, 19));
    try testing.expectEqual(@as(u64, 105), lcm(u64, 35, 15));
}

test "lcm - edge cases" {
    try testing.expectEqual(@as(u64, 0), lcm(u64, 0, 12));
    try testing.expectEqual(@as(u64, 0), lcm(u64, 12, 0));
    try testing.expectEqual(@as(u64, 7), lcm(u64, 7, 7));
}

test "extendedGcd - basic Bézout identity" {
    const result = extendedGcd(i64, 35, 15);
    try testing.expectEqual(@as(i64, 5), result.gcd);
    // Verify: 35*x + 15*y = 5
    try testing.expectEqual(@as(i64, 5), 35 * result.x + 15 * result.y);
}

test "extendedGcd - coprime numbers" {
    const result = extendedGcd(i64, 17, 19);
    try testing.expectEqual(@as(i64, 1), result.gcd);
    try testing.expectEqual(@as(i64, 1), 17 * result.x + 19 * result.y);
}

test "extendedGcd - with zero" {
    const result = extendedGcd(i64, 12, 0);
    try testing.expectEqual(@as(i64, 12), result.gcd);
    try testing.expectEqual(@as(i64, 1), result.x);
    try testing.expectEqual(@as(i64, 0), result.y);
}

test "modInverse - basic cases" {
    const inv = modInverse(i64, 3, 11).?;
    try testing.expectEqual(@as(i64, 4), inv);
    // Verify: 3*4 ≡ 1 (mod 11)
    try testing.expectEqual(@as(i64, 1), @mod(3 * inv, 11));
}

test "modInverse - no inverse when not coprime" {
    try testing.expectEqual(@as(?i64, null), modInverse(i64, 6, 9));
    try testing.expectEqual(@as(?i64, null), modInverse(i64, 4, 8));
}

test "modInverse - various cases" {
    const inv5_11 = modInverse(i64, 5, 11).?;
    try testing.expectEqual(@as(i64, 1), @mod(5 * inv5_11, 11));

    const inv7_13 = modInverse(i64, 7, 13).?;
    try testing.expectEqual(@as(i64, 1), @mod(7 * inv7_13, 13));
}

test "solveDiophantine - solvable case" {
    const sol = solveDiophantine(i64, 35, 15, 5).?;
    // Verify: 35*x + 15*y = 5
    try testing.expectEqual(@as(i64, 5), 35 * sol.x + 15 * sol.y);
}

test "solveDiophantine - unsolvable case" {
    // gcd(35,15) = 5, which doesn't divide 6
    try testing.expectEqual(@as(?struct { x: i64, y: i64 }, null), solveDiophantine(i64, 35, 15, 6));
}

test "solveDiophantine - various cases" {
    const sol1 = solveDiophantine(i64, 10, 6, 14).?;
    try testing.expectEqual(@as(i64, 14), 10 * sol1.x + 6 * sol1.y);

    const sol2 = solveDiophantine(i64, 17, 19, 1).?;
    try testing.expectEqual(@as(i64, 1), 17 * sol2.x + 19 * sol2.y);
}
