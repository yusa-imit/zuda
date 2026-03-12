//! Chinese Remainder Theorem (CRT)
//!
//! Solves systems of simultaneous modular congruences:
//!   x ≡ a₁ (mod n₁)
//!   x ≡ a₂ (mod n₂)
//!   ...
//!   x ≡ aₖ (mod nₖ)
//!
//! where n₁, n₂, ..., nₖ are pairwise coprime (gcd(nᵢ, nⱼ) = 1 for i ≠ j).
//!
//! Applications:
//!   - Cryptography (RSA, secret sharing)
//!   - Fast modular arithmetic with large moduli
//!   - Solving Diophantine equations
//!   - Calendar computations
//!
//! Algorithm:
//!   1. Compute N = n₁ × n₂ × ... × nₖ
//!   2. For each i, compute Nᵢ = N / nᵢ
//!   3. Find Mᵢ such that Nᵢ × Mᵢ ≡ 1 (mod nᵢ) using extended GCD
//!   4. Solution: x = (Σ aᵢ × Nᵢ × Mᵢ) mod N
//!
//! Time: O(k log²(max n)) where k is number of congruences
//! Space: O(1) auxiliary

const std = @import("std");
const gcd_mod = @import("gcd.zig");

/// Result of solving a system of modular congruences.
pub fn CRTResult(comptime T: type) type {
    return struct {
        /// The solution x such that x ≡ aᵢ (mod nᵢ) for all i
        value: T,
        /// The modulus N = n₁ × n₂ × ... × nₖ
        modulus: T,
    };
}

/// Solves a system of modular congruences using the Chinese Remainder Theorem.
///
/// Given parallel arrays:
///   - remainders: [a₁, a₂, ..., aₖ] where x ≡ aᵢ (mod nᵢ)
///   - moduli: [n₁, n₂, ..., nₖ] (must be pairwise coprime)
///
/// Returns the unique solution x modulo N = n₁ × n₂ × ... × nₖ.
///
/// Time: O(k log²(max n)) | Space: O(1)
///
/// Errors:
///   - ModuliNotCoprime: if any pair of moduli share a common factor > 1
///   - Overflow: if N = ∏ nᵢ exceeds type capacity
///   - InvalidInput: if arrays have different lengths or are empty
pub fn chineseRemainderTheorem(comptime T: type, remainders: []const T, moduli: []const T) !CRTResult(T) {
    const info = @typeInfo(T);
    if (info != .int) @compileError("CRT requires integer type");

    if (remainders.len != moduli.len) return error.InvalidInput;
    if (remainders.len == 0) return error.InvalidInput;

    // Special case: single congruence
    if (remainders.len == 1) {
        return .{ .value = @mod(remainders[0], moduli[0]), .modulus = moduli[0] };
    }

    // Compute N = product of all moduli
    var N: T = 1;
    for (moduli) |n| {
        if (n <= 0) return error.InvalidInput;
        // Check for overflow
        const old_N = N;
        N = N *% n;
        if (@divTrunc(N, n) != old_N and old_N != 0) return error.Overflow;
    }

    // Verify moduli are pairwise coprime
    for (moduli, 0..) |ni, i| {
        for (moduli[i + 1 ..]) |nj| {
            if (gcd_mod.gcd(T, ni, nj) != 1) {
                return error.ModuliNotCoprime;
            }
        }
    }

    var result: T = 0;
    for (remainders, moduli) |a, n| {
        const Ni = @divExact(N, n); // N / nᵢ

        // Find Mᵢ such that Nᵢ × Mᵢ ≡ 1 (mod nᵢ)
        // This is the modular inverse of Nᵢ modulo nᵢ
        const Mi = try modularInverse(T, Ni, n);

        // Accumulate: x += aᵢ × Nᵢ × Mᵢ
        const term = a *% Ni *% Mi;
        result = @mod(result +% term, N);
    }

    // Normalize result to [0, N)
    result = @mod(result, N);

    return .{ .value = result, .modulus = N };
}

/// Computes the modular inverse of a modulo n using extended GCD.
/// Returns x such that a × x ≡ 1 (mod n).
/// Time: O(log n) | Space: O(1)
fn modularInverse(comptime T: type, a: T, n: T) !T {
    const info = @typeInfo(T);

    // For unsigned types, we need to convert to signed for extended GCD
    const SignedT = std.meta.Int(.signed, info.int.bits);
    const a_signed: SignedT = @intCast(a);
    const n_signed: SignedT = @intCast(n);

    const result = gcd_mod.extendedGcd(SignedT, a_signed, n_signed);

    if (result.gcd != 1) {
        return error.NoModularInverse;
    }

    // Normalize x to be positive
    var x = result.x;
    if (x < 0) {
        x = @mod(x, n_signed);
    }

    return @intCast(x);
}

/// Solves a system of two modular congruences (specialized version).
/// More efficient than the general case.
///
/// Given:
///   x ≡ a₁ (mod n₁)
///   x ≡ a₂ (mod n₂)
/// where gcd(n₁, n₂) = 1
///
/// Returns x such that x ≡ a₁ (mod n₁) AND x ≡ a₂ (mod n₂)
///
/// Time: O(log²(max(n₁, n₂))) | Space: O(1)
pub fn crtTwo(comptime T: type, a1: T, n1: T, a2: T, n2: T) !CRTResult(T) {
    const remainders = [_]T{ a1, a2 };
    const moduli = [_]T{ n1, n2 };
    return chineseRemainderTheorem(T, &remainders, &moduli);
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "CRT: basic two congruences" {
    // Solve: x ≡ 2 (mod 3), x ≡ 3 (mod 5)
    // Solution: x = 8 (mod 15)
    const result = try crtTwo(i64, 2, 3, 3, 5);
    try testing.expectEqual(@as(i64, 8), result.value);
    try testing.expectEqual(@as(i64, 15), result.modulus);

    // Verify the solution
    try testing.expectEqual(@as(i64, 2), @mod(result.value, 3));
    try testing.expectEqual(@as(i64, 3), @mod(result.value, 5));
}

test "CRT: classic example" {
    // Solve: x ≡ 2 (mod 3), x ≡ 3 (mod 4), x ≡ 1 (mod 5)
    // Solution: x = 11 (mod 60)
    const remainders = [_]i64{ 2, 3, 1 };
    const moduli = [_]i64{ 3, 4, 5 };
    const result = try chineseRemainderTheorem(i64, &remainders, &moduli);

    try testing.expectEqual(@as(i64, 11), result.value);
    try testing.expectEqual(@as(i64, 60), result.modulus);

    // Verify
    try testing.expectEqual(@as(i64, 2), @mod(result.value, 3));
    try testing.expectEqual(@as(i64, 3), @mod(result.value, 4));
    try testing.expectEqual(@as(i64, 1), @mod(result.value, 5));
}

test "CRT: Sun Tzu's problem" {
    // Ancient Chinese puzzle:
    // "Find a number that leaves remainder 2 when divided by 3,
    //  remainder 3 when divided by 5, and remainder 2 when divided by 7"
    // Solution: x = 23 (mod 105)
    const remainders = [_]i64{ 2, 3, 2 };
    const moduli = [_]i64{ 3, 5, 7 };
    const result = try chineseRemainderTheorem(i64, &remainders, &moduli);

    try testing.expectEqual(@as(i64, 23), result.value);
    try testing.expectEqual(@as(i64, 105), result.modulus);

    // Verify
    try testing.expectEqual(@as(i64, 2), @mod(result.value, 3));
    try testing.expectEqual(@as(i64, 3), @mod(result.value, 5));
    try testing.expectEqual(@as(i64, 2), @mod(result.value, 7));
}

test "CRT: single congruence" {
    const remainders = [_]i64{42};
    const moduli = [_]i64{100};
    const result = try chineseRemainderTheorem(i64, &remainders, &moduli);

    try testing.expectEqual(@as(i64, 42), result.value);
    try testing.expectEqual(@as(i64, 100), result.modulus);
}

test "CRT: coprime check" {
    // Moduli 4 and 6 are not coprime (gcd = 2)
    const remainders = [_]i64{ 1, 3 };
    const moduli = [_]i64{ 4, 6 };
    try testing.expectError(error.ModuliNotCoprime, chineseRemainderTheorem(i64, &remainders, &moduli));
}

test "CRT: invalid input - empty arrays" {
    const remainders = [_]i64{};
    const moduli = [_]i64{};
    try testing.expectError(error.InvalidInput, chineseRemainderTheorem(i64, &remainders, &moduli));
}

test "CRT: invalid input - length mismatch" {
    const remainders = [_]i64{ 1, 2 };
    const moduli = [_]i64{3};
    try testing.expectError(error.InvalidInput, chineseRemainderTheorem(i64, &remainders, &moduli));
}

test "CRT: invalid input - zero or negative modulus" {
    {
        const remainders = [_]i64{ 1, 2 };
        const moduli = [_]i64{ 0, 3 };
        try testing.expectError(error.InvalidInput, chineseRemainderTheorem(i64, &remainders, &moduli));
    }
    {
        const remainders = [_]i64{ 1, 2 };
        const moduli = [_]i64{ 3, -5 };
        try testing.expectError(error.InvalidInput, chineseRemainderTheorem(i64, &remainders, &moduli));
    }
}

test "CRT: large numbers" {
    // x ≡ 123456 (mod 1000003), x ≡ 789012 (mod 1000033)
    const remainders = [_]i64{ 123456, 789012 };
    const moduli = [_]i64{ 1000003, 1000033 };
    const result = try chineseRemainderTheorem(i64, &remainders, &moduli);

    // Verify the solution
    try testing.expectEqual(@as(i64, 123456), @mod(result.value, 1000003));
    try testing.expectEqual(@as(i64, 789012), @mod(result.value, 1000033));
    try testing.expectEqual(@as(i64, 1000003 * 1000033), result.modulus);
}

test "CRT: four congruences" {
    // x ≡ 1 (mod 2), x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 4 (mod 7)
    const remainders = [_]i64{ 1, 2, 3, 4 };
    const moduli = [_]i64{ 2, 3, 5, 7 };
    const result = try chineseRemainderTheorem(i64, &remainders, &moduli);

    try testing.expectEqual(@as(i64, 2 * 3 * 5 * 7), result.modulus); // N = 210

    // Verify all congruences
    try testing.expectEqual(@as(i64, 1), @mod(result.value, 2));
    try testing.expectEqual(@as(i64, 2), @mod(result.value, 3));
    try testing.expectEqual(@as(i64, 3), @mod(result.value, 5));
    try testing.expectEqual(@as(i64, 4), @mod(result.value, 7));
}

test "CRT: prime moduli" {
    // Using prime moduli guarantees coprimality
    const remainders = [_]i64{ 1, 4, 6, 10 };
    const moduli = [_]i64{ 7, 11, 13, 17 }; // All primes
    const result = try chineseRemainderTheorem(i64, &remainders, &moduli);

    try testing.expectEqual(@as(i64, 7 * 11 * 13 * 17), result.modulus); // N = 17017

    // Verify
    try testing.expectEqual(@as(i64, 1), @mod(result.value, 7));
    try testing.expectEqual(@as(i64, 4), @mod(result.value, 11));
    try testing.expectEqual(@as(i64, 6), @mod(result.value, 13));
    try testing.expectEqual(@as(i64, 10), @mod(result.value, 17));
}

test "CRT: modular inverse - basic" {
    // 3 × 2 ≡ 1 (mod 5)
    const inv = try modularInverse(i64, 3, 5);
    try testing.expectEqual(@as(i64, 2), inv);
    try testing.expectEqual(@as(i64, 1), @mod(3 * inv, 5));
}

test "CRT: modular inverse - no inverse" {
    // 4 has no inverse modulo 6 (gcd(4, 6) = 2 ≠ 1)
    try testing.expectError(error.NoModularInverse, modularInverse(i64, 4, 6));
}

test "CRT: edge case - all remainders zero" {
    const remainders = [_]i64{ 0, 0, 0 };
    const moduli = [_]i64{ 3, 5, 7 };
    const result = try chineseRemainderTheorem(i64, &remainders, &moduli);

    try testing.expectEqual(@as(i64, 0), result.value);
    try testing.expectEqual(@as(i64, 105), result.modulus);
}

test "CRT: edge case - remainders equal to moduli minus one" {
    const remainders = [_]i64{ 2, 4, 6 }; // [3-1, 5-1, 7-1]
    const moduli = [_]i64{ 3, 5, 7 };
    const result = try chineseRemainderTheorem(i64, &remainders, &moduli);

    try testing.expectEqual(@as(i64, 104), result.value); // 105 - 1
    try testing.expectEqual(@as(i64, 105), result.modulus);
}
