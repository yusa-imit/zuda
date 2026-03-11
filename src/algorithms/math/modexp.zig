const std = @import("std");
const testing = std.testing;

/// Computes (base^exponent) mod modulus using binary exponentiation.
/// Also known as modular exponentiation or fast power algorithm.
/// Time: O(log(exponent))
/// Space: O(1)
pub fn modExp(comptime T: type, base: T, exponent: T, modulus: T) !T {
    const info = @typeInfo(T);
    if (info != .int) @compileError("modExp requires integer type");
    if (info.int.signedness == .signed) @compileError("modExp requires unsigned integer type");

    if (modulus <= 1) return error.InvalidModulus;
    if (exponent == 0) return 1;

    var result: T = 1;
    var b = @mod(base, modulus);
    var exp = exponent;

    while (exp > 0) {
        // If exponent is odd, multiply result by base
        if (exp & 1 == 1) {
            result = mulMod(T, result, b, modulus);
        }

        // Square the base and halve the exponent
        exp >>= 1;
        if (exp > 0) {
            b = mulMod(T, b, b, modulus);
        }
    }

    return result;
}

/// Helper function to compute (a * b) mod m without overflow.
/// Uses the double-and-add method for large integers.
/// Time: O(log(b))
/// Space: O(1)
fn mulMod(comptime T: type, a: T, b: T, m: T) T {
    if (@typeInfo(T).int.bits <= 32) {
        // For 32-bit or smaller, we can safely promote to 64-bit
        const result = (@as(u64, a) * @as(u64, b)) % @as(u64, m);
        return @intCast(result);
    }

    // For larger types, use double-and-add method
    var result: T = 0;
    var x = @mod(a, m);
    var y = b;

    while (y > 0) {
        if (y & 1 == 1) {
            result = addMod(T, result, x, m);
        }
        x = addMod(T, x, x, m);
        y >>= 1;
    }

    return result;
}

/// Helper function to compute (a + b) mod m without overflow.
/// Time: O(1)
/// Space: O(1)
fn addMod(comptime T: type, a: T, b: T, m: T) T {
    const a_mod = @mod(a, m);
    const b_mod = @mod(b, m);

    if (a_mod >= m - b_mod) {
        return a_mod - (m - b_mod);
    }
    return a_mod + b_mod;
}

/// Computes the modular multiplicative inverse of a modulo m.
/// Returns x such that (a * x) mod m = 1.
/// Uses the Extended Euclidean Algorithm.
/// Time: O(log(min(a, m)))
/// Space: O(1)
pub fn modInverse(comptime T: type, a: T, m: T) !T {
    const info = @typeInfo(T);
    if (info != .int) @compileError("modInverse requires integer type");

    if (m <= 1) return error.InvalidModulus;

    // Convert to signed type for extended GCD
    const SignedT = std.meta.Int(.signed, @typeInfo(T).int.bits);
    const a_signed: SignedT = @intCast(a);
    const m_signed: SignedT = @intCast(m);

    const result = extendedGcd(SignedT, a_signed, m_signed);

    if (result.gcd != 1) {
        return error.NoInverse; // Inverse doesn't exist
    }

    // Ensure result is positive
    const x = if (result.x < 0) result.x + m_signed else result.x;
    return @intCast(x);
}

fn extendedGcd(comptime T: type, a: T, b: T) struct { gcd: T, x: T, y: T } {
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

// ============================================================================
// Tests
// ============================================================================

test "modExp - basic cases" {
    try testing.expectEqual(@as(u32, 1), try modExp(u32, 2, 0, 5));
    try testing.expectEqual(@as(u32, 2), try modExp(u32, 2, 1, 5));
    try testing.expectEqual(@as(u32, 4), try modExp(u32, 2, 2, 5));
    try testing.expectEqual(@as(u32, 3), try modExp(u32, 2, 3, 5));
    try testing.expectEqual(@as(u32, 1), try modExp(u32, 2, 4, 5));
}

test "modExp - larger values" {
    // 3^10 mod 7 = 59049 mod 7 = 4
    try testing.expectEqual(@as(u32, 4), try modExp(u32, 3, 10, 7));

    // 5^117 mod 19 = 1 (Fermat's Little Theorem: a^(p-1) mod p = 1 for prime p)
    try testing.expectEqual(@as(u32, 1), try modExp(u32, 5, 18, 19));

    // 2^100 mod 13
    try testing.expectEqual(@as(u32, 3), try modExp(u32, 2, 100, 13));
}

test "modExp - RSA example (small)" {
    // Simplified RSA example
    // p = 61, q = 53, n = 3233, e = 17, d = 2753
    const n: u32 = 3233;
    const e: u32 = 17;
    const d: u32 = 2753;
    const message: u32 = 123;

    // Encrypt: c = m^e mod n
    const ciphertext = try modExp(u32, message, e, n);
    try testing.expectEqual(@as(u32, 855), ciphertext);

    // Decrypt: m = c^d mod n
    const decrypted = try modExp(u32, ciphertext, d, n);
    try testing.expectEqual(message, decrypted);
}

test "modExp - large exponents" {
    // 2^1000000 mod 1000000007 (common competitive programming modulus)
    const result = try modExp(u64, 2, 1000000, 1000000007);
    try testing.expect(result < 1000000007);
}

test "modExp - base larger than modulus" {
    try testing.expectEqual(@as(u32, 1), try modExp(u32, 10, 2, 3));
    try testing.expectEqual(@as(u32, 3), try modExp(u32, 7, 3, 5));
}

test "modExp - invalid modulus" {
    try testing.expectError(error.InvalidModulus, modExp(u32, 2, 3, 0));
    try testing.expectError(error.InvalidModulus, modExp(u32, 2, 3, 1));
}

test "mulMod - basic cases" {
    try testing.expectEqual(@as(u32, 2), mulMod(u32, 3, 4, 5));
    try testing.expectEqual(@as(u32, 6), mulMod(u32, 7, 8, 10));
    try testing.expectEqual(@as(u32, 0), mulMod(u32, 5, 6, 5));
}

test "mulMod - large values" {
    // Test with values that would overflow if multiplied directly
    const a: u64 = 1000000000000;
    const b: u64 = 999999999999;
    const m: u64 = 1000000007;
    const result = mulMod(u64, a, b, m);
    try testing.expect(result < m);
}

test "modInverse - basic cases" {
    // 3 * 2 = 6 ≡ 1 (mod 5), so inverse of 3 mod 5 is 2
    try testing.expectEqual(@as(u32, 2), try modInverse(u32, 3, 5));

    // 7 * 8 = 56 ≡ 1 (mod 11), so inverse of 7 mod 11 is 8
    try testing.expectEqual(@as(u32, 8), try modInverse(u32, 7, 11));

    // 17 * 17 = 289 ≡ 1 (mod 24), so inverse of 17 mod 24 is 17
    try testing.expectEqual(@as(u32, 17), try modInverse(u32, 17, 24));
}

test "modInverse - verify inverse property" {
    const a: u32 = 42;
    const m: u32 = 1000000007; // prime modulus
    const inv = try modInverse(u32, a, m);

    // Verify: (a * inv) mod m = 1
    const product = try modExp(u32, a, 1, m);
    const result = mulMod(u32, product, inv, m);
    try testing.expectEqual(@as(u32, 1), result);
}

test "modInverse - no inverse (non-coprime)" {
    // 6 and 9 are not coprime (gcd = 3), so no inverse exists
    try testing.expectError(error.NoInverse, modInverse(u32, 6, 9));

    // 10 and 15 are not coprime (gcd = 5)
    try testing.expectError(error.NoInverse, modInverse(u32, 10, 15));
}

test "modInverse - invalid modulus" {
    try testing.expectError(error.InvalidModulus, modInverse(u32, 3, 0));
    try testing.expectError(error.InvalidModulus, modInverse(u32, 3, 1));
}

test "modInverse - with prime modulus (Fermat's little theorem alternative)" {
    // For prime modulus, inverse of a is a^(p-2) mod p
    const a: u32 = 123;
    const p: u32 = 1000000007; // prime

    const inv1 = try modInverse(u32, a, p);
    const inv2 = try modExp(u32, a, p - 2, p);

    try testing.expectEqual(inv1, inv2);
}

test "addMod - basic cases" {
    try testing.expectEqual(@as(u32, 2), addMod(u32, 3, 4, 5));
    try testing.expectEqual(@as(u32, 1), addMod(u32, 7, 8, 7));
    try testing.expectEqual(@as(u32, 0), addMod(u32, 5, 5, 5));
}

test "addMod - large values near overflow" {
    const max = std.math.maxInt(u64);
    const m: u64 = 1000000007;

    const a = max - 100;
    const b = max - 200;
    const result = addMod(u64, a, b, m);
    try testing.expect(result < m);
}
