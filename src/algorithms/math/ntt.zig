//! Number Theoretic Transform (NTT) — Fast polynomial multiplication modulo a prime.
//!
//! NTT is the modular arithmetic analogue of FFT, used for fast polynomial multiplication
//! and convolution over finite fields. Common applications:
//!   - Fast polynomial multiplication mod p
//!   - Competitive programming (modular arithmetic problems)
//!   - Cryptography (lattice-based schemes)
//!   - Number theory (computing sequences mod p)
//!
//! Key features:
//!   - O(n log n) time for n-point NTT
//!   - Works with any prime p where p-1 is divisible by a high power of 2
//!   - Iterative bit-reversal implementation (cache-friendly)
//!   - Common moduli: 998244353 (2^23 * 119 + 1), 1000000007 (2^1 factor - limited)
//!
//! Algorithm:
//!   - Primitive root of unity: ω = g^((p-1)/n) mod p where g is a primitive root
//!   - Forward NTT: A[k] = Σ a[j] * ω^(jk) mod p
//!   - Inverse NTT: a[j] = (1/n) * Σ A[k] * ω^(-jk) mod p
//!   - Convolution theorem: (a * b)[k] = INTT(NTT(a) · NTT(b))
//!
//! Reference: "The Number Theoretic Transform" — Cooley-Tukey adaptation for Z_p

const std = @import("std");

/// Common NTT-friendly primes.
pub const Moduli = struct {
    /// 998244353 = 2^23 * 119 + 1 (supports up to 2^23 points)
    /// Primitive root: 3
    pub const MOD_998244353: u64 = 998244353;
    pub const ROOT_998244353: u64 = 3;

    /// 1000000007 (2^1 factor only - supports up to 2 points, not practical for NTT)
    /// Included for reference but NOT recommended
    pub const MOD_1000000007: u64 = 1000000007;

    /// 469762049 = 2^26 * 7 + 1 (supports up to 2^26 points)
    /// Primitive root: 3
    pub const MOD_469762049: u64 = 469762049;
    pub const ROOT_469762049: u64 = 3;
};

/// Compute a^b mod m using binary exponentiation.
/// Time: O(log b)
fn modPow(base: u64, exp: u64, mod: u64) u64 {
    var result: u64 = 1;
    var b = base % mod;
    var e = exp;
    while (e > 0) : (e >>= 1) {
        if (e & 1 == 1) {
            result = @intCast(((@as(u128, result) * b) % mod));
        }
        b = @intCast(((@as(u128, b) * b) % mod));
    }
    return result;
}

/// Compute modular inverse using Fermat's little theorem: a^(p-2) ≡ a^(-1) (mod p) for prime p.
/// Time: O(log p)
fn modInv(a: u64, mod: u64) u64 {
    return modPow(a, mod - 2, mod);
}

/// Bit-reverse permutation for index i with n = 2^k.
/// Time: O(log n)
fn bitReverse(i: usize, k: usize) usize {
    var result: usize = 0;
    var idx = i;
    for (0..k) |_| {
        result = (result << 1) | (idx & 1);
        idx >>= 1;
    }
    return result;
}

/// Compute the primitive n-th root of unity modulo p.
/// For prime p where p-1 = 2^s * d, ω = g^((p-1)/n) where g is a primitive root.
/// Time: O(log p)
fn computeRoot(n: usize, primitive_root: u64, mod: u64) u64 {
    const exp = (mod - 1) / @as(u64, @intCast(n));
    return modPow(primitive_root, exp, mod);
}

/// Number Theoretic Transform (forward).
/// Transforms polynomial coefficients a[0..n-1] in-place to NTT(a).
///
/// Parameters:
///   - a: Coefficient array (length must be power of 2)
///   - primitive_root: Primitive root modulo mod (e.g., 3 for 998244353)
///   - mod: Prime modulus (p-1 must be divisible by n)
///
/// Time: O(n log n) | Space: O(1)
pub fn ntt(a: []u64, primitive_root: u64, mod: u64) void {
    const n = a.len;
    std.debug.assert(n > 0 and (n & (n - 1)) == 0); // Power of 2

    // Bit-reversal permutation
    const k = @ctz(n);
    for (a, 0..) |*val, i| {
        const j = bitReverse(i, k);
        if (i < j) {
            const tmp = val.*;
            val.* = a[j];
            a[j] = tmp;
        }
    }

    // Cooley-Tukey iterative NTT
    var len: usize = 2;
    while (len <= n) : (len <<= 1) {
        const w = computeRoot(len, primitive_root, mod);
        var i: usize = 0;
        while (i < n) : (i += len) {
            var wn: u64 = 1;
            for (0..len / 2) |j| {
                const u = a[i + j];
                const v: u64 = @intCast(((@as(u128, a[i + j + len / 2]) * wn) % mod));
                a[i + j] = @intCast(((u + v) % mod));
                a[i + j + len / 2] = @intCast((((u + mod) - v) % mod));
                wn = @intCast(((@as(u128, wn) * w) % mod));
            }
        }
    }
}

/// Inverse Number Theoretic Transform.
/// Transforms NTT(a) back to polynomial coefficients a[0..n-1] in-place.
///
/// Time: O(n log n) | Space: O(1)
pub fn intt(a: []u64, primitive_root: u64, mod: u64) void {
    const n = a.len;
    std.debug.assert(n > 0 and (n & (n - 1)) == 0);

    // Compute n-th root and its inverse
    const w = computeRoot(n, primitive_root, mod);
    const w_inv = modInv(w, mod);

    // Bit-reversal permutation
    const k = @ctz(n);
    for (a, 0..) |*val, i| {
        const j = bitReverse(i, k);
        if (i < j) {
            const tmp = val.*;
            val.* = a[j];
            a[j] = tmp;
        }
    }

    // Cooley-Tukey iterative INTT (same as NTT but with w_inv)
    var len: usize = 2;
    while (len <= n) : (len <<= 1) {
        var i: usize = 0;
        while (i < n) : (i += len) {
            var wn: u64 = 1;
            const w_len = modPow(w_inv, @intCast(n / len), mod);
            for (0..len / 2) |j| {
                const u = a[i + j];
                const v: u64 = @intCast(((@as(u128, a[i + j + len / 2]) * wn) % mod));
                a[i + j] = @intCast(((u + v) % mod));
                a[i + j + len / 2] = @intCast((((u + mod) - v) % mod));
                wn = @intCast(((@as(u128, wn) * w_len) % mod));
            }
        }
    }

    // Scale by 1/n
    const n_inv = modInv(@intCast(n), mod);
    for (a) |*val| {
        val.* = @intCast(((@as(u128, val.*) * n_inv) % mod));
    }
}

/// Multiply two polynomials using NTT.
/// Computes (a * b) mod p where p is the modulus.
///
/// Parameters:
///   - allocator: Memory allocator
///   - a: First polynomial coefficients
///   - b: Second polynomial coefficients
///   - primitive_root: Primitive root modulo mod
///   - mod: Prime modulus
///
/// Returns product polynomial of degree deg(a) + deg(b).
///
/// Time: O(n log n) where n = next power of 2 ≥ len(a) + len(b) - 1
/// Space: O(n)
pub fn multiply(
    allocator: std.mem.Allocator,
    a: []const u64,
    b: []const u64,
    primitive_root: u64,
    mod: u64,
) ![]u64 {
    if (a.len == 0 or b.len == 0) {
        return try allocator.alloc(u64, 0);
    }

    // Check if either polynomial is zero (all coefficients are 0)
    var a_is_zero = true;
    for (a) |coeff| {
        if (coeff != 0) {
            a_is_zero = false;
            break;
        }
    }
    var b_is_zero = true;
    for (b) |coeff| {
        if (coeff != 0) {
            b_is_zero = false;
            break;
        }
    }
    if (a_is_zero or b_is_zero) {
        return try allocator.alloc(u64, 0);
    }

    // Find next power of 2 >= len(a) + len(b) - 1
    const result_len = a.len + b.len - 1;
    var n: usize = 1;
    while (n < result_len) : (n <<= 1) {}

    // Zero-pad to power of 2
    var a_ntt = try allocator.alloc(u64, n);
    defer allocator.free(a_ntt);
    var b_ntt = try allocator.alloc(u64, n);
    defer allocator.free(b_ntt);

    @memset(a_ntt, 0);
    @memset(b_ntt, 0);
    @memcpy(a_ntt[0..a.len], a);
    @memcpy(b_ntt[0..b.len], b);

    // Forward NTT
    ntt(a_ntt, primitive_root, mod);
    ntt(b_ntt, primitive_root, mod);

    // Pointwise multiplication
    for (a_ntt, b_ntt) |*a_val, b_val| {
        a_val.* = @intCast(((@as(u128, a_val.*) * b_val) % mod));
    }

    // Inverse NTT
    intt(a_ntt, primitive_root, mod);

    // Return result (truncate padding)
    const result = try allocator.alloc(u64, result_len);
    @memcpy(result, a_ntt[0..result_len]);
    return result;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "ntt: basic transform and inverse" {
    const mod = Moduli.MOD_998244353;
    const root = Moduli.ROOT_998244353;

    var a = [_]u64{ 1, 2, 3, 4 };
    const original = a;

    ntt(&a, root, mod);
    intt(&a, root, mod);

    // Should reconstruct original
    for (a, original) |got, expected| {
        try testing.expectEqual(expected, got);
    }
}

test "ntt: zero polynomial" {
    const mod = Moduli.MOD_998244353;
    const root = Moduli.ROOT_998244353;

    var a = [_]u64{ 0, 0, 0, 0 };
    ntt(&a, root, mod);

    for (a) |val| {
        try testing.expectEqual(@as(u64, 0), val);
    }
}

test "ntt: constant polynomial" {
    const mod = Moduli.MOD_998244353;
    const root = Moduli.ROOT_998244353;

    var a = [_]u64{ 5, 0, 0, 0 };
    const original = a;

    ntt(&a, root, mod);
    intt(&a, root, mod);

    for (a, original) |got, expected| {
        try testing.expectEqual(expected, got);
    }
}

test "multiply: (1 + x) * (1 + x) = 1 + 2x + x^2" {
    const mod = Moduli.MOD_998244353;
    const root = Moduli.ROOT_998244353;

    const a = [_]u64{ 1, 1 }; // 1 + x
    const b = [_]u64{ 1, 1 }; // 1 + x
    const result = try multiply(testing.allocator, &a, &b, root, mod);
    defer testing.allocator.free(result);

    try testing.expectEqual(@as(usize, 3), result.len);
    try testing.expectEqual(@as(u64, 1), result[0]); // x^0
    try testing.expectEqual(@as(u64, 2), result[1]); // x^1
    try testing.expectEqual(@as(u64, 1), result[2]); // x^2
}

test "multiply: (2 + 3x) * (4 + 5x) = 8 + 22x + 15x^2" {
    const mod = Moduli.MOD_998244353;
    const root = Moduli.ROOT_998244353;

    const a = [_]u64{ 2, 3 }; // 2 + 3x
    const b = [_]u64{ 4, 5 }; // 4 + 5x
    const result = try multiply(testing.allocator, &a, &b, root, mod);
    defer testing.allocator.free(result);

    try testing.expectEqual(@as(usize, 3), result.len);
    try testing.expectEqual(@as(u64, 8), result[0]); // 2*4
    try testing.expectEqual(@as(u64, 22), result[1]); // 2*5 + 3*4
    try testing.expectEqual(@as(u64, 15), result[2]); // 3*5
}

test "multiply: zero polynomial" {
    const mod = Moduli.MOD_998244353;
    const root = Moduli.ROOT_998244353;

    const a = [_]u64{ 1, 2, 3 };
    const b = [_]u64{0};
    const result = try multiply(testing.allocator, &a, &b, root, mod);
    defer testing.allocator.free(result);

    try testing.expectEqual(@as(usize, 0), result.len);
}

test "multiply: empty polynomial" {
    const mod = Moduli.MOD_998244353;
    const root = Moduli.ROOT_998244353;

    const a = [_]u64{};
    const b = [_]u64{ 1, 2 };
    const result = try multiply(testing.allocator, &a, &b, root, mod);
    defer testing.allocator.free(result);

    try testing.expectEqual(@as(usize, 0), result.len);
}

test "multiply: large coefficients mod p" {
    const mod = Moduli.MOD_998244353;
    const root = Moduli.ROOT_998244353;

    const a = [_]u64{ mod - 1, mod - 1 }; // (-1) + (-1)x
    const b = [_]u64{ mod - 1, mod - 1 }; // (-1) + (-1)x
    const result = try multiply(testing.allocator, &a, &b, root, mod);
    defer testing.allocator.free(result);

    // (-1 + -1x)^2 = 1 + 2x + x^2
    try testing.expectEqual(@as(usize, 3), result.len);
    try testing.expectEqual(@as(u64, 1), result[0]);
    try testing.expectEqual(@as(u64, 2), result[1]);
    try testing.expectEqual(@as(u64, 1), result[2]);
}

test "multiply: stress test with large polynomial" {
    const mod = Moduli.MOD_998244353;
    const root = Moduli.ROOT_998244353;

    const a = try testing.allocator.alloc(u64, 100);
    defer testing.allocator.free(a);
    const b = try testing.allocator.alloc(u64, 100);
    defer testing.allocator.free(b);

    for (a, 0..) |*val, i| val.* = @intCast((i + 1) % mod);
    for (b, 0..) |*val, i| val.* = @intCast((i + 2) % mod);

    const result = try multiply(testing.allocator, a, b, root, mod);
    defer testing.allocator.free(result);

    try testing.expectEqual(@as(usize, 199), result.len);
    // Result should be deterministic (not checking exact values, just properties)
    for (result) |val| {
        try testing.expect(val < mod);
    }
}

test "ntt: alternative modulus 469762049" {
    const mod = Moduli.MOD_469762049;
    const root = Moduli.ROOT_469762049;

    var a = [_]u64{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const original = a;

    ntt(&a, root, mod);
    intt(&a, root, mod);

    for (a, original) |got, expected| {
        try testing.expectEqual(expected, got);
    }
}

test "multiply: associativity (a*b)*c = a*(b*c)" {
    const mod = Moduli.MOD_998244353;
    const root = Moduli.ROOT_998244353;

    const a = [_]u64{ 1, 2 };
    const b = [_]u64{ 3, 4 };
    const c = [_]u64{ 5, 6 };

    const ab = try multiply(testing.allocator, &a, &b, root, mod);
    defer testing.allocator.free(ab);
    const ab_c = try multiply(testing.allocator, ab, &c, root, mod);
    defer testing.allocator.free(ab_c);

    const bc = try multiply(testing.allocator, &b, &c, root, mod);
    defer testing.allocator.free(bc);
    const a_bc = try multiply(testing.allocator, &a, bc, root, mod);
    defer testing.allocator.free(a_bc);

    try testing.expectEqual(ab_c.len, a_bc.len);
    for (ab_c, a_bc) |left, right| {
        try testing.expectEqual(left, right);
    }
}

test "multiply: commutativity a*b = b*a" {
    const mod = Moduli.MOD_998244353;
    const root = Moduli.ROOT_998244353;

    const a = [_]u64{ 7, 11, 13 };
    const b = [_]u64{ 2, 5 };

    const ab = try multiply(testing.allocator, &a, &b, root, mod);
    defer testing.allocator.free(ab);
    const ba = try multiply(testing.allocator, &b, &a, root, mod);
    defer testing.allocator.free(ba);

    try testing.expectEqual(ab.len, ba.len);
    for (ab, ba) |left, right| {
        try testing.expectEqual(left, right);
    }
}

test "ntt: memory leak check" {
    const mod = Moduli.MOD_998244353;
    const root = Moduli.ROOT_998244353;

    const a = [_]u64{ 1, 2, 3, 4 };
    const b = [_]u64{ 5, 6, 7, 8 };
    const result = try multiply(testing.allocator, &a, &b, root, mod);
    testing.allocator.free(result);
}
