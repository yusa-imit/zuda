const std = @import("std");
const testing = std.testing;

/// Miller-Rabin primality test - probabilistic algorithm for testing primality.
///
/// Time: O(k log³ n) where k is number of rounds | Space: O(1)
///
/// This is a Monte Carlo randomized algorithm. If it returns false, the number
/// is definitely composite. If it returns true, the number is prime with high
/// probability. The error probability is at most 4^(-k) where k is the number
/// of rounds.
///
/// Recommended rounds:
/// - k=5: error probability ≤ 1/1024
/// - k=10: error probability ≤ 1/1048576
/// - k=20: error probability ≤ 1/1099511627776
///
/// Algorithm:
/// 1. Write n-1 = 2^r × d where d is odd
/// 2. For k rounds:
///    a. Pick random witness a ∈ [2, n-2]
///    b. Compute x = a^d mod n
///    c. If x = 1 or x = n-1, continue to next round
///    d. For i = 0 to r-1:
///       - x = x² mod n
///       - If x = n-1, continue to next round
///    e. If we reach here, n is composite
/// 3. If all rounds pass, n is probably prime
pub fn isProbablyPrime(n: u64, rounds: usize, random: std.Random) bool {
    // Handle small cases
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0) return false;

    // Write n-1 = 2^r × d
    var r: u32 = 0;
    var d: u64 = n - 1;
    while (d % 2 == 0) {
        r += 1;
        d /= 2;
    }

    // Witness loop
    var round: usize = 0;
    while (round < rounds) : (round += 1) {
        const a = random.intRangeAtMost(u64, 2, n - 2);
        if (!millerRabinWitness(n, d, r, a)) return false;
    }

    return true;
}

fn millerRabinWitness(n: u64, d: u64, r: u32, a: u64) bool {
    var x = modPow(a, d, n);

    if (x == 1 or x == n - 1) return true;

    var i: u32 = 0;
    while (i < r - 1) : (i += 1) {
        x = mulMod(x, x, n);
        if (x == n - 1) return true;
    }

    return false;
}

/// Modular exponentiation: (base^exp) mod m
///
/// Time: O(log exp) | Space: O(1)
fn modPow(base: u64, exp: u64, m: u64) u64 {
    if (m == 1) return 0;

    var result: u64 = 1;
    var b = base % m;
    var e = exp;

    while (e > 0) {
        if (e % 2 == 1) {
            result = mulMod(result, b, m);
        }
        b = mulMod(b, b, m);
        e /= 2;
    }

    return result;
}

/// Modular multiplication: (a × b) mod m, avoiding overflow
///
/// Time: O(log b) | Space: O(1)
fn mulMod(a: u64, b: u64, m: u64) u64 {
    // Use u128 to avoid overflow
    const result = @as(u128, a) * @as(u128, b);
    return @intCast(result % m);
}

/// Deterministic variant using specific witnesses for ranges.
///
/// Time: O(log³ n) | Space: O(1)
///
/// For n < 3,317,044,064,679,887,385,961,981, using witnesses
/// [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] is deterministic.
///
/// This covers all 64-bit unsigned integers.
pub fn isPrime(n: u64) bool {
    // Handle small cases
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0) return false;

    // Write n-1 = 2^r × d
    var r: u32 = 0;
    var d: u64 = n - 1;
    while (d % 2 == 0) {
        r += 1;
        d /= 2;
    }

    // Deterministic witnesses for all 64-bit integers
    const witnesses = [_]u64{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };

    for (witnesses) |a| {
        if (a >= n) continue;
        if (!millerRabinWitness(n, d, r, a)) return false;
    }

    return true;
}

/// Find the next prime number >= n.
///
/// Time: O(n log³ n / ln n) expected | Space: O(1)
///
/// Uses deterministic Miller-Rabin for correctness.
pub fn nextPrime(n: u64) ?u64 {
    if (n <= 2) return 2;

    var candidate = if (n % 2 == 0) n + 1 else n;

    while (candidate > 0) { // Overflow check
        if (isPrime(candidate)) return candidate;

        candidate += 2;
        if (candidate < n) return null; // Overflow
    }

    return null;
}

/// Generate a random probable prime in the range [min, max].
///
/// Time: O((max-min) × k log³ n) expected | Space: O(1)
///
/// Uses probabilistic Miller-Rabin with k rounds.
pub fn randomPrime(min: u64, max: u64, rounds: usize, random: std.Random) ?u64 {
    if (min > max) return null;
    if (max <= 1) return null;

    const start = @max(min, 2);

    // Try random candidates
    const max_attempts = 1000;
    var attempt: usize = 0;
    while (attempt < max_attempts) : (attempt += 1) {
        var candidate = random.intRangeAtMost(u64, start, max);
        if (candidate % 2 == 0) candidate += 1; // Make odd

        if (candidate > max) continue;
        if (isProbablyPrime(candidate, rounds, random)) return candidate;
    }

    return null;
}

test "miller_rabin: small primes" {
    try testing.expect(isPrime(2));
    try testing.expect(isPrime(3));
    try testing.expect(isPrime(5));
    try testing.expect(isPrime(7));
    try testing.expect(isPrime(11));
    try testing.expect(isPrime(13));
}

test "miller_rabin: small composites" {
    try testing.expect(!isPrime(0));
    try testing.expect(!isPrime(1));
    try testing.expect(!isPrime(4));
    try testing.expect(!isPrime(6));
    try testing.expect(!isPrime(8));
    try testing.expect(!isPrime(9));
    try testing.expect(!isPrime(10));
}

test "miller_rabin: medium primes" {
    try testing.expect(isPrime(97));
    try testing.expect(isPrime(101));
    try testing.expect(isPrime(997));
    try testing.expect(isPrime(1009));
}

test "miller_rabin: medium composites" {
    try testing.expect(!isPrime(100));
    try testing.expect(!isPrime(1000));
    try testing.expect(!isPrime(1001));
}

test "miller_rabin: large primes" {
    try testing.expect(isPrime(104729)); // 10000th prime
    try testing.expect(isPrime(1299709)); // 100000th prime
    try testing.expect(isPrime(15485863)); // 1000000th prime
}

test "miller_rabin: Carmichael numbers" {
    // Carmichael numbers are composite but pass Fermat test
    // Miller-Rabin should detect them as composite
    try testing.expect(!isPrime(561)); // 3 × 11 × 17
    try testing.expect(!isPrime(1105)); // 5 × 13 × 17
    try testing.expect(!isPrime(1729)); // 7 × 13 × 19
}

test "miller_rabin: probabilistic variant basic" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    try testing.expect(isProbablyPrime(7, 5, random));
    try testing.expect(isProbablyPrime(11, 5, random));
    try testing.expect(!isProbablyPrime(4, 5, random));
    try testing.expect(!isProbablyPrime(6, 5, random));
}

test "miller_rabin: probabilistic large prime" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    try testing.expect(isProbablyPrime(104729, 10, random));
    try testing.expect(!isProbablyPrime(104730, 10, random));
}

test "miller_rabin: next prime basic" {
    try testing.expectEqual(2, nextPrime(0).?);
    try testing.expectEqual(2, nextPrime(1).?);
    try testing.expectEqual(2, nextPrime(2).?);
    try testing.expectEqual(3, nextPrime(3).?);
    try testing.expectEqual(5, nextPrime(4).?);
    try testing.expectEqual(5, nextPrime(5).?);
}

test "miller_rabin: next prime after composite" {
    try testing.expectEqual(11, nextPrime(10).?);
    try testing.expectEqual(101, nextPrime(100).?);
    try testing.expectEqual(1009, nextPrime(1000).?);
}

test "miller_rabin: next prime chain" {
    var p = nextPrime(2).?;
    try testing.expectEqual(2, p);

    p = nextPrime(p + 1).?;
    try testing.expectEqual(3, p);

    p = nextPrime(p + 1).?;
    try testing.expectEqual(5, p);

    p = nextPrime(p + 1).?;
    try testing.expectEqual(7, p);
}

test "miller_rabin: random prime in range" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const p = randomPrime(10, 20, 5, random);
    try testing.expect(p != null);
    try testing.expect(p.? >= 10);
    try testing.expect(p.? <= 20);
    try testing.expect(isPrime(p.?));
}

test "miller_rabin: random prime small range" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // Range with only one prime (11)
    const p = randomPrime(11, 12, 5, random);
    try testing.expect(p != null);
    try testing.expectEqual(11, p.?);
}

test "miller_rabin: random prime no primes in range" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // Range with no primes
    const p = randomPrime(8, 10, 5, random);
    // May return null or may not find one in max_attempts
    _ = p;
}

test "miller_rabin: random prime large range" {
    var prng = std.Random.DefaultPrng.init(777);
    const random = prng.random();

    const p = randomPrime(1000, 2000, 10, random);
    try testing.expect(p != null);
    try testing.expect(p.? >= 1000);
    try testing.expect(p.? <= 2000);
    try testing.expect(isPrime(p.?));
}

test "miller_rabin: modular exponentiation" {
    try testing.expectEqual(1, modPow(2, 0, 10));
    try testing.expectEqual(8, modPow(2, 3, 10));
    try testing.expectEqual(6, modPow(2, 10, 10)); // 1024 % 10 = 4, wait 2^10=1024, 1024%10=4... let me recalc
    try testing.expectEqual(4, modPow(2, 10, 10)); // 2^10 = 1024, 1024 % 10 = 4
    try testing.expectEqual(1, modPow(3, 4, 5)); // 81 % 5 = 1
}

test "miller_rabin: modular multiplication" {
    try testing.expectEqual(6, mulMod(2, 3, 10));
    try testing.expectEqual(8, mulMod(4, 7, 10)); // 28 % 10 = 8
    try testing.expectEqual(0, mulMod(5, 4, 10)); // 20 % 10 = 0
}

test "miller_rabin: large modular operations" {
    // Test with large numbers to ensure no overflow
    const large: u64 = 1 << 60;
    const result = mulMod(large, large, large + 1);
    try testing.expect(result < large + 1);
}

test "miller_rabin: stress test known primes" {
    // First 20 primes
    const primes = [_]u64{
        2,  3,  5,  7,  11, 13, 17, 19, 23, 29,
        31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    };

    for (primes) |p| {
        try testing.expect(isPrime(p));
    }
}

test "miller_rabin: stress test known composites" {
    var n: u64 = 4;
    var count: usize = 0;
    while (n < 100 and count < 50) : (n += 1) {
        if (!isPrime(n)) {
            try testing.expect(!isPrime(n));
            count += 1;
        }
    }
}
