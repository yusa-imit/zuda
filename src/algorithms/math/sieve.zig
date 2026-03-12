//! Prime number generation algorithms.
//!
//! Provides efficient algorithms for finding all prime numbers up to a given limit.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Generate all prime numbers up to n using the Sieve of Eratosthenes.
/// Time: O(n log log n) | Space: O(n)
///
/// This is one of the most efficient algorithms for finding all primes up to a limit.
/// The algorithm works by iteratively marking the multiples of each prime starting from 2.
///
/// Arguments:
/// - allocator: memory allocator
/// - n: upper limit (inclusive)
///
/// Returns: dynamically allocated slice of all primes <= n
pub fn sieveOfEratosthenes(allocator: Allocator, n: usize) ![]usize {
    if (n < 2) {
        return try allocator.alloc(usize, 0);
    }

    // Create a boolean array "is_prime[0..n+1]" and initialize all entries as true
    const is_prime = try allocator.alloc(bool, n + 1);
    defer allocator.free(is_prime);

    @memset(is_prime, true);
    is_prime[0] = false;
    is_prime[1] = false;

    // Sieve of Eratosthenes
    var p: usize = 2;
    while (p * p <= n) : (p += 1) {
        if (is_prime[p]) {
            // Mark all multiples of p as composite
            var multiple = p * p;
            while (multiple <= n) : (multiple += p) {
                is_prime[multiple] = false;
            }
        }
    }

    // Count primes
    var count: usize = 0;
    for (is_prime) |is_p| {
        if (is_p) count += 1;
    }

    // Collect primes
    const primes = try allocator.alloc(usize, count);
    var idx: usize = 0;
    for (is_prime, 0..) |is_p, i| {
        if (is_p) {
            primes[idx] = i;
            idx += 1;
        }
    }

    return primes;
}

/// Segmented sieve for finding primes in a range [low, high].
/// Time: O((high - low) log log high) | Space: O(sqrt(high) + (high - low))
///
/// This is more memory-efficient than the basic sieve for large ranges,
/// as it only keeps a boolean array for the range [low, high] in memory.
///
/// Arguments:
/// - allocator: memory allocator
/// - low: lower bound (inclusive)
/// - high: upper bound (inclusive)
///
/// Returns: dynamically allocated slice of all primes in [low, high]
pub fn segmentedSieve(allocator: Allocator, low: usize, high: usize) ![]usize {
    if (high < low or high < 2) {
        return try allocator.alloc(usize, 0);
    }

    // Find all primes up to sqrt(high) using simple sieve
    const limit = @as(usize, @intFromFloat(@sqrt(@as(f64, @floatFromInt(high))))) + 1;
    const base_primes = try sieveOfEratosthenes(allocator, limit);
    defer allocator.free(base_primes);

    // Create a boolean array for [low, high]
    const range_size = high - low + 1;
    const is_prime = try allocator.alloc(bool, range_size);
    defer allocator.free(is_prime);
    @memset(is_prime, true);

    // Use base primes to mark composites in [low, high]
    for (base_primes) |p| {
        // Find the minimum number in [low, high] that is a multiple of p
        var start = @max(p * p, ((low + p - 1) / p) * p);

        // If start equals p, it's prime in the range
        if (start == p and start >= low) {
            start += p;
        }

        // Mark all multiples of p in range as composite
        while (start <= high) : (start += p) {
            is_prime[start - low] = false;
        }
    }

    // Special case: if low == 1, mark it as composite
    if (low == 1) {
        is_prime[0] = false;
    }

    // Count primes in range
    var count: usize = 0;
    for (is_prime) |is_p| {
        if (is_p) count += 1;
    }

    // Collect primes
    const primes = try allocator.alloc(usize, count);
    var idx: usize = 0;
    for (is_prime, 0..) |is_p, i| {
        if (is_p) {
            primes[idx] = low + i;
            idx += 1;
        }
    }

    return primes;
}

/// Count the number of primes up to n without storing them.
/// Time: O(n log log n) | Space: O(n)
///
/// More memory-efficient than sieveOfEratosthenes when you only need the count.
pub fn countPrimes(allocator: Allocator, n: usize) !usize {
    if (n < 2) return 0;

    const is_prime = try allocator.alloc(bool, n + 1);
    defer allocator.free(is_prime);

    @memset(is_prime, true);
    is_prime[0] = false;
    is_prime[1] = false;

    var p: usize = 2;
    while (p * p <= n) : (p += 1) {
        if (is_prime[p]) {
            var multiple = p * p;
            while (multiple <= n) : (multiple += p) {
                is_prime[multiple] = false;
            }
        }
    }

    var count: usize = 0;
    for (is_prime) |is_p| {
        if (is_p) count += 1;
    }

    return count;
}

/// Check if a number is prime using trial division up to sqrt(n).
/// Time: O(sqrt(n)) | Space: O(1)
///
/// This is simpler but slower than the sieve for checking many numbers.
/// Prefer the sieve when checking multiple numbers, and this for single checks.
pub fn isPrime(n: usize) bool {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;

    var i: usize = 3;
    const limit = @as(usize, @intFromFloat(@sqrt(@as(f64, @floatFromInt(n))))) + 1;
    while (i <= limit) : (i += 2) {
        if (n % i == 0) return false;
    }

    return true;
}

/// Get the nth prime number (1-indexed: nthPrime(1) = 2).
/// Time: O(n log n log log n) | Space: O(n log n)
///
/// Uses sieve to generate primes up to an estimated upper bound.
pub fn nthPrime(allocator: Allocator, n: usize) !usize {
    if (n == 0) return error.InvalidIndex;

    // Use prime number theorem approximation: nth prime ≈ n * ln(n)
    // Add safety margin for small n
    const limit = if (n < 6)
        30
    else
        @as(usize, @intFromFloat(@as(f64, @floatFromInt(n)) * @log(@as(f64, @floatFromInt(n))) * 1.3));

    const primes = try sieveOfEratosthenes(allocator, limit);
    defer allocator.free(primes);

    if (n > primes.len) {
        return error.LimitTooSmall;
    }

    return primes[n - 1];
}

// ============================================================================
// Tests
// ============================================================================

test "sieve: basic sieve of eratosthenes" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const primes = try sieveOfEratosthenes(allocator, 30);
    defer allocator.free(primes);

    const expected = [_]usize{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29 };
    try testing.expectEqualSlices(usize, &expected, primes);
}

test "sieve: empty range" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const primes = try sieveOfEratosthenes(allocator, 1);
    defer allocator.free(primes);

    try testing.expectEqual(0, primes.len);
}

test "sieve: single prime (2)" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const primes = try sieveOfEratosthenes(allocator, 2);
    defer allocator.free(primes);

    try testing.expectEqual(1, primes.len);
    try testing.expectEqual(2, primes[0]);
}

test "sieve: first 100 primes count" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const primes = try sieveOfEratosthenes(allocator, 100);
    defer allocator.free(primes);

    try testing.expectEqual(25, primes.len);
}

test "sieve: large sieve (10000)" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const primes = try sieveOfEratosthenes(allocator, 10000);
    defer allocator.free(primes);

    // There are 1229 primes <= 10000
    try testing.expectEqual(1229, primes.len);

    // Verify first and last
    try testing.expectEqual(2, primes[0]);
    try testing.expectEqual(9973, primes[primes.len - 1]);
}

test "sieve: segmented sieve basic" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const primes = try segmentedSieve(allocator, 10, 30);
    defer allocator.free(primes);

    const expected = [_]usize{ 11, 13, 17, 19, 23, 29 };
    try testing.expectEqualSlices(usize, &expected, primes);
}

test "sieve: segmented sieve large range" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const primes = try segmentedSieve(allocator, 1000, 1100);
    defer allocator.free(primes);

    // Verify count
    try testing.expectEqual(16, primes.len);

    // Verify all are actually prime and in range
    for (primes) |p| {
        try testing.expect(p >= 1000 and p <= 1100);
        try testing.expect(isPrime(p));
    }
}

test "sieve: segmented sieve matches full sieve" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const full_primes = try sieveOfEratosthenes(allocator, 500);
    defer allocator.free(full_primes);

    const seg_primes = try segmentedSieve(allocator, 2, 500);
    defer allocator.free(seg_primes);

    try testing.expectEqualSlices(usize, full_primes, seg_primes);
}

test "sieve: count primes" {
    const testing = std.testing;
    const allocator = testing.allocator;

    try testing.expectEqual(0, try countPrimes(allocator, 1));
    try testing.expectEqual(1, try countPrimes(allocator, 2));
    try testing.expectEqual(4, try countPrimes(allocator, 10));
    try testing.expectEqual(25, try countPrimes(allocator, 100));
    try testing.expectEqual(168, try countPrimes(allocator, 1000));
}

test "sieve: isPrime basic" {
    const testing = std.testing;

    try testing.expect(!isPrime(0));
    try testing.expect(!isPrime(1));
    try testing.expect(isPrime(2));
    try testing.expect(isPrime(3));
    try testing.expect(!isPrime(4));
    try testing.expect(isPrime(5));
    try testing.expect(!isPrime(6));
    try testing.expect(isPrime(7));
    try testing.expect(!isPrime(8));
    try testing.expect(!isPrime(9));
    try testing.expect(!isPrime(10));
    try testing.expect(isPrime(11));
}

test "sieve: isPrime matches sieve results" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const primes = try sieveOfEratosthenes(allocator, 1000);
    defer allocator.free(primes);

    // All primes from sieve should return true
    for (primes) |p| {
        try testing.expect(isPrime(p));
    }

    // Check some composites
    try testing.expect(!isPrime(100));
    try testing.expect(!isPrime(999));
    try testing.expect(!isPrime(1000));
}

test "sieve: nth prime" {
    const testing = std.testing;
    const allocator = testing.allocator;

    try testing.expectEqual(2, try nthPrime(allocator, 1));
    try testing.expectEqual(3, try nthPrime(allocator, 2));
    try testing.expectEqual(5, try nthPrime(allocator, 3));
    try testing.expectEqual(7, try nthPrime(allocator, 4));
    try testing.expectEqual(11, try nthPrime(allocator, 5));
    try testing.expectEqual(29, try nthPrime(allocator, 10));
    try testing.expectEqual(541, try nthPrime(allocator, 100));
}

test "sieve: nth prime invalid index" {
    const testing = std.testing;
    const allocator = testing.allocator;

    try testing.expectError(error.InvalidIndex, nthPrime(allocator, 0));
}

test "sieve: segmented sieve edge cases" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Range with no primes
    const no_primes = try segmentedSieve(allocator, 24, 28);
    defer allocator.free(no_primes);
    try testing.expectEqual(0, no_primes.len);

    // Range with single prime
    const single = try segmentedSieve(allocator, 2, 2);
    defer allocator.free(single);
    try testing.expectEqual(1, single.len);
    try testing.expectEqual(2, single[0]);

    // Invalid range
    const invalid = try segmentedSieve(allocator, 100, 50);
    defer allocator.free(invalid);
    try testing.expectEqual(0, invalid.len);
}

test "sieve: stress test" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Generate many primes and verify properties
    const primes = try sieveOfEratosthenes(allocator, 5000);
    defer allocator.free(primes);

    // Verify all are actually prime
    for (primes) |p| {
        try testing.expect(isPrime(p));
    }

    // Verify strictly increasing
    for (primes[0 .. primes.len - 1], primes[1..]) |p1, p2| {
        try testing.expect(p1 < p2);
    }

    // Verify no primes are missing (check a few gaps)
    var i: usize = 2;
    var prime_idx: usize = 0;
    while (i <= 5000) : (i += 1) {
        if (isPrime(i)) {
            try testing.expectEqual(i, primes[prime_idx]);
            prime_idx += 1;
        }
    }
}
