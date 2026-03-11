const std = @import("std");
const testing = std.testing;
const modexp = @import("modexp.zig");

/// Miller-Rabin probabilistic primality test.
/// Returns true if n is probably prime, false if n is definitely composite.
/// The error probability is at most 4^(-k) where k is the number of rounds.
/// Time: O(k * log³(n)) where k is the number of rounds
/// Space: O(1)
pub fn millerRabin(comptime T: type, n: T, rounds: u32, rng: std.Random) !bool {
    const info = @typeInfo(T);
    if (info != .int) @compileError("millerRabin requires integer type");
    if (info.int.signedness == .signed) @compileError("millerRabin requires unsigned integer type");

    // Handle small cases
    if (n < 2) return false;
    if (n == 2 or n == 3) return true;
    if (n % 2 == 0) return false;

    // Write n-1 as 2^r * d where d is odd
    var d = n - 1;
    var r: u32 = 0;
    while (d % 2 == 0) : (r += 1) {
        d /= 2;
    }

    // Witness loop
    var round: u32 = 0;
    while (round < rounds) : (round += 1) {
        // Pick a random witness in range [2, n-2]
        const a = if (n <= 4)
            2
        else
            2 + rng.intRangeLessThan(T, 0, n - 3);

        var x = try modexp.modExp(T, a, d, n);

        if (x == 1 or x == n - 1) {
            continue;
        }

        var composite = true;
        var i: u32 = 0;
        while (i < r - 1) : (i += 1) {
            x = try modexp.modExp(T, x, 2, n);
            if (x == n - 1) {
                composite = false;
                break;
            }
        }

        if (composite) {
            return false;
        }
    }

    return true;
}

/// Deterministic Miller-Rabin test for 64-bit integers.
/// Uses a fixed set of witnesses that work for all 64-bit integers.
/// Time: O(log³(n))
/// Space: O(1)
pub fn millerRabinDeterministic(n: u64) !bool {
    if (n < 2) return false;
    if (n == 2 or n == 3) return true;
    if (n % 2 == 0) return false;

    // These witnesses are sufficient for all n < 2^64
    const witnesses = [_]u64{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };

    // Write n-1 as 2^r * d where d is odd
    var d = n - 1;
    var r: u32 = 0;
    while (d % 2 == 0) : (r += 1) {
        d /= 2;
    }

    for (witnesses) |a| {
        if (a >= n) continue;

        var x = try modexp.modExp(u64, a, d, n);

        if (x == 1 or x == n - 1) {
            continue;
        }

        var composite = true;
        var i: u32 = 0;
        while (i < r - 1) : (i += 1) {
            x = try modexp.modExp(u64, x, 2, n);
            if (x == n - 1) {
                composite = false;
                break;
            }
        }

        if (composite) {
            return false;
        }
    }

    return true;
}

/// Simple trial division for small primes (up to sqrt(n)).
/// Efficient for small numbers or as a pre-filter before Miller-Rabin.
/// Time: O(sqrt(n))
/// Space: O(1)
pub fn trialDivision(comptime T: type, n: T) bool {
    const info = @typeInfo(T);
    if (info != .int) @compileError("trialDivision requires integer type");
    if (info.int.signedness == .signed) @compileError("trialDivision requires unsigned integer type");

    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    if (n == 3) return true;
    if (n % 3 == 0) return false;

    // Check divisibility by numbers of form 6k ± 1
    var i: T = 5;
    while (i * i <= n) {
        if (n % i == 0 or n % (i + 2) == 0) {
            return false;
        }
        i += 6;
    }

    return true;
}

/// Fermat primality test (less reliable than Miller-Rabin).
/// Returns true if n is probably prime.
/// Time: O(k * log³(n)) where k is the number of rounds
/// Space: O(1)
pub fn fermat(comptime T: type, n: T, rounds: u32, rng: std.Random) !bool {
    const info = @typeInfo(T);
    if (info != .int) @compileError("fermat requires integer type");
    if (info.int.signedness == .signed) @compileError("fermat requires unsigned integer type");

    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;

    var round: u32 = 0;
    while (round < rounds) : (round += 1) {
        const a = 2 + rng.intRangeLessThan(T, 0, n - 3);
        const result = try modexp.modExp(T, a, n - 1, n);

        if (result != 1) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// Tests
// ============================================================================

test "trialDivision - small primes" {
    try testing.expect(trialDivision(u32, 2));
    try testing.expect(trialDivision(u32, 3));
    try testing.expect(trialDivision(u32, 5));
    try testing.expect(trialDivision(u32, 7));
    try testing.expect(trialDivision(u32, 11));
    try testing.expect(trialDivision(u32, 13));
    try testing.expect(trialDivision(u32, 17));
    try testing.expect(trialDivision(u32, 19));
}

test "trialDivision - small composites" {
    try testing.expect(!trialDivision(u32, 0));
    try testing.expect(!trialDivision(u32, 1));
    try testing.expect(!trialDivision(u32, 4));
    try testing.expect(!trialDivision(u32, 6));
    try testing.expect(!trialDivision(u32, 8));
    try testing.expect(!trialDivision(u32, 9));
    try testing.expect(!trialDivision(u32, 10));
    try testing.expect(!trialDivision(u32, 12));
}

test "trialDivision - larger primes" {
    try testing.expect(trialDivision(u32, 97));
    try testing.expect(trialDivision(u32, 101));
    try testing.expect(trialDivision(u32, 1009));
    try testing.expect(trialDivision(u32, 10007));
}

test "trialDivision - larger composites" {
    try testing.expect(!trialDivision(u32, 100));
    try testing.expect(!trialDivision(u32, 1000));
    try testing.expect(!trialDivision(u32, 10000));
}

test "millerRabinDeterministic - small primes" {
    try testing.expect(try millerRabinDeterministic(2));
    try testing.expect(try millerRabinDeterministic(3));
    try testing.expect(try millerRabinDeterministic(5));
    try testing.expect(try millerRabinDeterministic(7));
    try testing.expect(try millerRabinDeterministic(11));
    try testing.expect(try millerRabinDeterministic(13));
}

test "millerRabinDeterministic - small composites" {
    try testing.expect(!try millerRabinDeterministic(0));
    try testing.expect(!try millerRabinDeterministic(1));
    try testing.expect(!try millerRabinDeterministic(4));
    try testing.expect(!try millerRabinDeterministic(6));
    try testing.expect(!try millerRabinDeterministic(8));
    try testing.expect(!try millerRabinDeterministic(9));
}

test "millerRabinDeterministic - larger primes" {
    try testing.expect(try millerRabinDeterministic(97));
    try testing.expect(try millerRabinDeterministic(1009));
    try testing.expect(try millerRabinDeterministic(10007));
    try testing.expect(try millerRabinDeterministic(1000000007));
}

test "millerRabinDeterministic - larger composites" {
    try testing.expect(!try millerRabinDeterministic(100));
    try testing.expect(!try millerRabinDeterministic(1001));
    try testing.expect(!try millerRabinDeterministic(10000));
    try testing.expect(!try millerRabinDeterministic(1000000000));
}

test "millerRabinDeterministic - Carmichael numbers" {
    // Carmichael numbers fool Fermat test but not Miller-Rabin
    try testing.expect(!try millerRabinDeterministic(561)); // 3 × 11 × 17
    try testing.expect(!try millerRabinDeterministic(1105)); // 5 × 13 × 17
    try testing.expect(!try millerRabinDeterministic(1729)); // 7 × 13 × 19
}

test "millerRabinDeterministic - large primes" {
    // Mersenne primes
    try testing.expect(try millerRabinDeterministic(2147483647)); // 2^31 - 1
}

test "millerRabin - probabilistic with fixed seed" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Test a few known primes
    try testing.expect(try millerRabin(u64, 97, 10, rng));
    try testing.expect(try millerRabin(u64, 1009, 10, rng));
    try testing.expect(try millerRabin(u64, 10007, 10, rng));

    // Test a few known composites
    try testing.expect(!try millerRabin(u64, 100, 10, rng));
    try testing.expect(!try millerRabin(u64, 1001, 10, rng));
    try testing.expect(!try millerRabin(u64, 10000, 10, rng));
}

test "fermat - basic cases" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Test a few known primes
    try testing.expect(try fermat(u64, 97, 10, rng));
    try testing.expect(try fermat(u64, 1009, 10, rng));

    // Test a few known composites
    try testing.expect(!try fermat(u64, 100, 10, rng));
    try testing.expect(!try fermat(u64, 1000, 10, rng));
}

test "consistency - trialDivision vs millerRabinDeterministic" {
    // For small numbers, both methods should agree
    var n: u64 = 2;
    while (n < 1000) : (n += 1) {
        const trial = trialDivision(u64, n);
        const miller = try millerRabinDeterministic(n);
        try testing.expectEqual(trial, miller);
    }
}
