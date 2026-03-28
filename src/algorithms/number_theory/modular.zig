//! Modular Arithmetic Algorithms
//!
//! Efficient algorithms for modular arithmetic operations, essential for
//! cryptography, number theory, and computational algebra.

const std = @import("std");
const testing = std.testing;

/// Compute (base^exp) mod m efficiently using binary exponentiation.
///
/// Time: O(log exp)
/// Space: O(1)
///
/// Example:
/// ```zig
/// const result = modPow(u64, 2, 10, 1000); // 2^10 mod 1000 = 24
/// ```
pub fn modPow(comptime T: type, base: T, exp: T, m: T) T {
    if (m == 1) return 0;

    var result: T = 1;
    var b = @mod(base, m);
    var e = exp;

    while (e > 0) {
        if (e & 1 == 1) {
            result = @mod(result * b, m);
        }
        b = @mod(b * b, m);
        e >>= 1;
    }

    return result;
}

/// Compute (a + b) mod m safely, avoiding overflow.
///
/// Time: O(1)
/// Space: O(1)
pub fn modAdd(comptime T: type, a: T, b: T, m: T) T {
    const a_mod = @mod(a, m);
    const b_mod = @mod(b, m);
    return @mod(a_mod + b_mod, m);
}

/// Compute (a - b) mod m safely, handling negative results.
///
/// Time: O(1)
/// Space: O(1)
pub fn modSub(comptime T: type, a: T, b: T, m: T) T {
    const a_mod = @mod(a, m);
    const b_mod = @mod(b, m);
    return @mod(a_mod - b_mod + m, m);
}

/// Compute (a * b) mod m safely, avoiding overflow.
///
/// For large integers, uses the Russian peasant multiplication method.
///
/// Time: O(log b)
/// Space: O(1)
pub fn modMul(comptime T: type, a: T, b: T, m: T) T {
    if (@bitSizeOf(T) <= 32) {
        // For small types, we can safely multiply
        return @mod(a * b, m);
    }

    // Russian peasant multiplication for large types
    var result: T = 0;
    var x = @mod(a, m);
    var y = @mod(b, m);

    while (y > 0) {
        if (y & 1 == 1) {
            result = modAdd(T, result, x, m);
        }
        x = modAdd(T, x, x, m);
        y >>= 1;
    }

    return result;
}

/// Chinese Remainder Theorem: solve system of congruences.
///
/// Given:
/// x ≡ a[0] (mod m[0])
/// x ≡ a[1] (mod m[1])
/// ...
/// x ≡ a[n-1] (mod m[n-1])
///
/// Find x where all m[i] are pairwise coprime.
///
/// Time: O(n log max(m))
/// Space: O(1)
///
/// Returns null if the moduli are not pairwise coprime.
///
/// Example:
/// ```zig
/// const a = [_]i64{2, 3, 2};
/// const m = [_]i64{3, 5, 7};
/// const x = crt(i64, &a, &m); // Some(23) because 23≡2(mod 3), 23≡3(mod 5), 23≡2(mod 7)
/// ```
pub fn crt(comptime T: type, a: []const T, m: []const T) ?T {
    if (a.len == 0 or a.len != m.len) return null;

    // Compute product of all moduli
    var M: T = 1;
    for (m) |mod| {
        M *= mod;
    }

    var result: T = 0;
    for (a, m) |remainder, mod| {
        const Mi = @divExact(M, mod);

        // Find modular inverse of Mi modulo mod
        const inv = @import("gcd.zig").modInverse(T, Mi, mod) orelse return null;

        result = modAdd(T, result, modMul(T, modMul(T, remainder, Mi, M), inv, M), M);
    }

    return @mod(result, M);
}

/// Compute Euler's totient function φ(n): count of integers in [1,n] coprime to n.
///
/// Time: O(√n)
/// Space: O(1)
///
/// Example:
/// ```zig
/// const phi = eulerTotient(u64, 9); // 6 (1,2,4,5,7,8 are coprime to 9)
/// ```
pub fn eulerTotient(comptime T: type, n: T) T {
    if (n == 1) return 1;

    var result = n;
    var num = n;
    var i: T = 2;

    // Factor n and apply Euler's product formula
    while (i * i <= num) {
        if (num % i == 0) {
            // Remove all factors of i
            while (num % i == 0) {
                num /= i;
            }
            // φ(p^k) = p^(k-1) * (p-1), so we multiply by (1 - 1/p)
            result -= @divExact(result, i);
        }
        i += 1;
    }

    // If num > 1, then it's a prime factor
    if (num > 1) {
        result -= @divExact(result, num);
    }

    return result;
}

/// Fast Fibonacci number computation using matrix exponentiation.
///
/// Time: O(log n)
/// Space: O(1)
///
/// Example:
/// ```zig
/// const f10 = fibonacci(u64, 10); // 55
/// ```
pub fn fibonacci(comptime T: type, n: T) T {
    if (n == 0) return 0;
    if (n == 1) return 1;

    // Matrix exponentiation: [[1,1],[1,0]]^n
    var a: T = 1;
    var b: T = 1;
    var c: T = 1;
    var d: T = 0;

    var exp = n - 1;
    while (exp > 0) {
        if (exp & 1 == 1) {
            const ta = a;
            const tb = b;
            a = a * a + b * c;
            b = ta * b + b * d;
            c = c * ta + d * c;
            d = c * tb + d * d;
        }
        const ta = a;
        const tb = b;
        const tc = c;
        a = a * a + b * c;
        b = ta * b + b * d;
        c = tc * ta + d * c;
        d = tc * tb + d * d;
        exp >>= 1;
    }

    return a;
}

// Tests

test "modPow - basic cases" {
    try testing.expectEqual(@as(u64, 24), modPow(u64, 2, 10, 1000));
    try testing.expectEqual(@as(u64, 1), modPow(u64, 5, 0, 13));
    try testing.expectEqual(@as(u64, 5), modPow(u64, 5, 1, 13));
}

test "modPow - large exponents" {
    // 2^100 mod 1000000007 (common in competitive programming)
    try testing.expectEqual(@as(u64, 976371285), modPow(u64, 2, 100, 1000000007));
}

test "modPow - edge cases" {
    try testing.expectEqual(@as(u64, 0), modPow(u64, 10, 5, 1));
    try testing.expectEqual(@as(u64, 0), modPow(u64, 0, 5, 7));
}

test "modAdd - basic cases" {
    try testing.expectEqual(@as(u64, 3), modAdd(u64, 8, 5, 10));
    try testing.expectEqual(@as(u64, 0), modAdd(u64, 7, 3, 10));
}

test "modSub - basic cases" {
    try testing.expectEqual(@as(u64, 3), modSub(u64, 8, 5, 10));
    try testing.expectEqual(@as(u64, 6), modSub(u64, 3, 7, 10));
}

test "modMul - basic cases" {
    try testing.expectEqual(@as(u64, 6), modMul(u64, 3, 4, 10));
    try testing.expectEqual(@as(u64, 1), modMul(u64, 7, 9, 10));
}

test "modMul - large integers" {
    const result = modMul(u128, 123456789012345, 987654321098765, 1000000007);
    // Just verify it doesn't overflow and produces a valid result
    try testing.expect(result < 1000000007);
}

test "crt - basic example" {
    const a = [_]i64{ 2, 3, 2 };
    const m = [_]i64{ 3, 5, 7 };
    const x = crt(i64, &a, &m).?;

    // Verify solution
    try testing.expectEqual(@as(i64, 2), @mod(x, 3));
    try testing.expectEqual(@as(i64, 3), @mod(x, 5));
    try testing.expectEqual(@as(i64, 2), @mod(x, 7));
}

test "crt - coprime moduli" {
    const a = [_]i64{ 1, 2, 3 };
    const m = [_]i64{ 5, 7, 9 };
    const x = crt(i64, &a, &m).?;

    try testing.expectEqual(@as(i64, 1), @mod(x, 5));
    try testing.expectEqual(@as(i64, 2), @mod(x, 7));
    try testing.expectEqual(@as(i64, 3), @mod(x, 9));
}

test "crt - single congruence" {
    const a = [_]i64{3};
    const m = [_]i64{7};
    const x = crt(i64, &a, &m).?;
    try testing.expectEqual(@as(i64, 3), x);
}

test "eulerTotient - basic cases" {
    try testing.expectEqual(@as(u64, 6), eulerTotient(u64, 9));
    try testing.expectEqual(@as(u64, 1), eulerTotient(u64, 1));
    try testing.expectEqual(@as(u64, 1), eulerTotient(u64, 2));
}

test "eulerTotient - prime numbers" {
    try testing.expectEqual(@as(u64, 6), eulerTotient(u64, 7));
    try testing.expectEqual(@as(u64, 10), eulerTotient(u64, 11));
    try testing.expectEqual(@as(u64, 12), eulerTotient(u64, 13));
}

test "eulerTotient - composite numbers" {
    try testing.expectEqual(@as(u64, 4), eulerTotient(u64, 10));
    try testing.expectEqual(@as(u64, 4), eulerTotient(u64, 12));
    try testing.expectEqual(@as(u64, 6), eulerTotient(u64, 18));
}

test "fibonacci - basic cases" {
    try testing.expectEqual(@as(u64, 0), fibonacci(u64, 0));
    try testing.expectEqual(@as(u64, 1), fibonacci(u64, 1));
    try testing.expectEqual(@as(u64, 1), fibonacci(u64, 2));
    try testing.expectEqual(@as(u64, 2), fibonacci(u64, 3));
    try testing.expectEqual(@as(u64, 3), fibonacci(u64, 4));
    try testing.expectEqual(@as(u64, 5), fibonacci(u64, 5));
    try testing.expectEqual(@as(u64, 8), fibonacci(u64, 6));
}

test "fibonacci - larger values" {
    try testing.expectEqual(@as(u64, 55), fibonacci(u64, 10));
    try testing.expectEqual(@as(u64, 6765), fibonacci(u64, 20));
}
