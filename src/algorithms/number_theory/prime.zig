//! Prime Number Algorithms
//!
//! Algorithms for primality testing, prime factorization, and prime generation.
//! Essential for cryptography, number theory, and computational mathematics.

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const modPow = @import("modular.zig").modPow;

/// Simple trial division primality test.
///
/// Time: O(√n)
/// Space: O(1)
///
/// Example:
/// ```zig
/// const is_prime = isPrime(u64, 17); // true
/// ```
pub fn isPrime(comptime T: type, n: T) bool {
    if (n < 2) return false;
    if (n == 2 or n == 3) return true;
    if (n % 2 == 0 or n % 3 == 0) return false;

    var i: T = 5;
    while (i * i <= n) {
        if (n % i == 0 or n % (i + 2) == 0) return false;
        i += 6;
    }

    return true;
}

/// Sieve of Eratosthenes: generate all primes up to n.
///
/// Time: O(n log log n)
/// Space: O(n)
///
/// Example:
/// ```zig
/// var primes = try sieveOfEratosthenes(allocator, 30);
/// defer primes.deinit();
/// // primes.items = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
/// ```
pub fn sieveOfEratosthenes(allocator: Allocator, n: usize) !std.ArrayList(usize) {
    var primes = std.ArrayList(usize).init(allocator);
    errdefer primes.deinit();

    if (n < 2) return primes;

    // Create boolean array
    var is_prime_arr = try allocator.alloc(bool, n + 1);
    defer allocator.free(is_prime_arr);
    @memset(is_prime_arr, true);
    is_prime_arr[0] = false;
    is_prime_arr[1] = false;

    var i: usize = 2;
    while (i * i <= n) : (i += 1) {
        if (is_prime_arr[i]) {
            var j = i * i;
            while (j <= n) : (j += i) {
                is_prime_arr[j] = false;
            }
        }
    }

    // Collect primes
    for (is_prime_arr, 0..) |is_p, idx| {
        if (is_p) {
            try primes.append(idx);
        }
    }

    return primes;
}

/// Prime factorization using trial division.
///
/// Returns list of (prime, exponent) pairs.
///
/// Time: O(√n)
/// Space: O(log n) for the factor list
///
/// Example:
/// ```zig
/// var factors = try primeFactorization(allocator, 60);
/// defer factors.deinit();
/// // factors.items = [(2,2), (3,1), (5,1)] because 60 = 2^2 * 3^1 * 5^1
/// ```
pub fn primeFactorization(allocator: Allocator, n: u64) !std.ArrayList(struct { prime: u64, exp: u32 }) {
    var factors = std.ArrayList(struct { prime: u64, exp: u32 }).init(allocator);
    errdefer factors.deinit();

    if (n <= 1) return factors;

    var num = n;

    // Check for factor 2
    if (num % 2 == 0) {
        var exp: u32 = 0;
        while (num % 2 == 0) {
            num /= 2;
            exp += 1;
        }
        try factors.append(.{ .prime = 2, .exp = exp });
    }

    // Check for odd factors
    var i: u64 = 3;
    while (i * i <= num) : (i += 2) {
        if (num % i == 0) {
            var exp: u32 = 0;
            while (num % i == 0) {
                num /= i;
                exp += 1;
            }
            try factors.append(.{ .prime = i, .exp = exp });
        }
    }

    // If num > 1, then it's a prime factor
    if (num > 1) {
        try factors.append(.{ .prime = num, .exp = 1 });
    }

    return factors;
}

/// Count the number of divisors of n.
///
/// Time: O(√n)
/// Space: O(1) for direct counting, O(log n) via factorization
///
/// Example:
/// ```zig
/// const count = try countDivisors(allocator, 12); // 6 (1,2,3,4,6,12)
/// ```
pub fn countDivisors(allocator: Allocator, n: u64) !u64 {
    var factors = try primeFactorization(allocator, n);
    defer factors.deinit();

    var count: u64 = 1;
    for (factors.items) |factor| {
        count *= (factor.exp + 1);
    }

    return count;
}

/// Sum of all divisors of n (including 1 and n).
///
/// Time: O(√n)
/// Space: O(log n) via factorization
///
/// Example:
/// ```zig
/// const sum = try sumOfDivisors(allocator, 12); // 28 (1+2+3+4+6+12)
/// ```
pub fn sumOfDivisors(allocator: Allocator, n: u64) !u64 {
    var factors = try primeFactorization(allocator, n);
    defer factors.deinit();

    var sum: u64 = 1;
    for (factors.items) |factor| {
        // Sum of geometric series: (p^(e+1) - 1) / (p - 1)
        var geo_sum: u64 = 0;
        var p_pow: u64 = 1;
        var e: u32 = 0;
        while (e <= factor.exp) : (e += 1) {
            geo_sum += p_pow;
            p_pow *= factor.prime;
        }
        sum *= geo_sum;
    }

    return sum;
}

/// Find the next prime number greater than or equal to n.
///
/// Time: O(n log log n) worst case (dense primes)
/// Space: O(1)
///
/// Example:
/// ```zig
/// const next = nextPrime(u64, 100); // 101
/// ```
pub fn nextPrime(comptime T: type, n: T) T {
    if (n <= 2) return 2;

    var candidate = if (n % 2 == 0) n + 1 else n;
    while (!isPrime(T, candidate)) {
        candidate += 2;
    }

    return candidate;
}

/// Find the largest prime factor of n.
///
/// Time: O(√n)
/// Space: O(1)
///
/// Example:
/// ```zig
/// const largest = largestPrimeFactor(u64, 60); // 5
/// ```
pub fn largestPrimeFactor(n: u64) u64 {
    if (n <= 1) return 0;

    var largest: u64 = 0;
    var num = n;

    // Remove all factors of 2
    if (num % 2 == 0) {
        largest = 2;
        while (num % 2 == 0) {
            num /= 2;
        }
    }

    // Check odd factors
    var i: u64 = 3;
    while (i * i <= num) : (i += 2) {
        if (num % i == 0) {
            largest = i;
            while (num % i == 0) {
                num /= i;
            }
        }
    }

    // If num > 1, it's the largest prime factor
    if (num > 1) {
        largest = num;
    }

    return largest;
}

/// Check if n is a perfect power (n = a^b for some a,b > 1).
///
/// Time: O(log^2 n)
/// Space: O(1)
///
/// Example:
/// ```zig
/// const is_power = isPerfectPower(u64, 8); // true (2^3)
/// const not_power = isPerfectPower(u64, 12); // false
/// ```
pub fn isPerfectPower(n: u64) bool {
    if (n <= 1) return true;

    // Check for each possible exponent b from 2 to log2(n)
    var b: u64 = 2;
    while ((1 << @as(u6, @intCast(b))) <= n) : (b += 1) {
        // Binary search for base a
        var lo: u64 = 1;
        var hi: u64 = n;

        while (lo <= hi) {
            const mid = lo + (hi - lo) / 2;
            const power = std.math.pow(u64, mid, b);

            if (power == n) {
                return true;
            } else if (power < n) {
                lo = mid + 1;
            } else {
                if (mid == 0) break;
                hi = mid - 1;
            }
        }
    }

    return false;
}

// Tests

test "isPrime - basic cases" {
    try testing.expect(!isPrime(u64, 0));
    try testing.expect(!isPrime(u64, 1));
    try testing.expect(isPrime(u64, 2));
    try testing.expect(isPrime(u64, 3));
    try testing.expect(!isPrime(u64, 4));
    try testing.expect(isPrime(u64, 5));
}

test "isPrime - composite numbers" {
    try testing.expect(!isPrime(u64, 15));
    try testing.expect(!isPrime(u64, 100));
    try testing.expect(!isPrime(u64, 1000));
}

test "isPrime - primes" {
    try testing.expect(isPrime(u64, 17));
    try testing.expect(isPrime(u64, 97));
    try testing.expect(isPrime(u64, 101));
}

test "sieveOfEratosthenes - basic" {
    var primes = try sieveOfEratosthenes(testing.allocator, 30);
    defer primes.deinit();

    const expected = [_]usize{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29 };
    try testing.expectEqualSlices(usize, &expected, primes.items);
}

test "sieveOfEratosthenes - edge cases" {
    var primes0 = try sieveOfEratosthenes(testing.allocator, 0);
    defer primes0.deinit();
    try testing.expectEqual(@as(usize, 0), primes0.items.len);

    var primes1 = try sieveOfEratosthenes(testing.allocator, 1);
    defer primes1.deinit();
    try testing.expectEqual(@as(usize, 0), primes1.items.len);

    var primes2 = try sieveOfEratosthenes(testing.allocator, 2);
    defer primes2.deinit();
    try testing.expectEqual(@as(usize, 1), primes2.items.len);
    try testing.expectEqual(@as(usize, 2), primes2.items[0]);
}

test "primeFactorization - basic cases" {
    var factors = try primeFactorization(testing.allocator, 60);
    defer factors.deinit();

    try testing.expectEqual(@as(usize, 3), factors.items.len);
    try testing.expectEqual(@as(u64, 2), factors.items[0].prime);
    try testing.expectEqual(@as(u32, 2), factors.items[0].exp);
    try testing.expectEqual(@as(u64, 3), factors.items[1].prime);
    try testing.expectEqual(@as(u32, 1), factors.items[1].exp);
    try testing.expectEqual(@as(u64, 5), factors.items[2].prime);
    try testing.expectEqual(@as(u32, 1), factors.items[2].exp);
}

test "primeFactorization - prime number" {
    var factors = try primeFactorization(testing.allocator, 17);
    defer factors.deinit();

    try testing.expectEqual(@as(usize, 1), factors.items.len);
    try testing.expectEqual(@as(u64, 17), factors.items[0].prime);
    try testing.expectEqual(@as(u32, 1), factors.items[0].exp);
}

test "primeFactorization - power of prime" {
    var factors = try primeFactorization(testing.allocator, 32);
    defer factors.deinit();

    try testing.expectEqual(@as(usize, 1), factors.items.len);
    try testing.expectEqual(@as(u64, 2), factors.items[0].prime);
    try testing.expectEqual(@as(u32, 5), factors.items[0].exp);
}

test "countDivisors - basic cases" {
    try testing.expectEqual(@as(u64, 6), try countDivisors(testing.allocator, 12));
    try testing.expectEqual(@as(u64, 2), try countDivisors(testing.allocator, 17));
    try testing.expectEqual(@as(u64, 1), try countDivisors(testing.allocator, 1));
}

test "sumOfDivisors - basic cases" {
    try testing.expectEqual(@as(u64, 28), try sumOfDivisors(testing.allocator, 12));
    try testing.expectEqual(@as(u64, 18), try sumOfDivisors(testing.allocator, 17));
    try testing.expectEqual(@as(u64, 1), try sumOfDivisors(testing.allocator, 1));
}

test "nextPrime - basic cases" {
    try testing.expectEqual(@as(u64, 2), nextPrime(u64, 0));
    try testing.expectEqual(@as(u64, 2), nextPrime(u64, 2));
    try testing.expectEqual(@as(u64, 3), nextPrime(u64, 3));
    try testing.expectEqual(@as(u64, 5), nextPrime(u64, 4));
    try testing.expectEqual(@as(u64, 101), nextPrime(u64, 100));
}

test "largestPrimeFactor - basic cases" {
    try testing.expectEqual(@as(u64, 5), largestPrimeFactor(60));
    try testing.expectEqual(@as(u64, 17), largestPrimeFactor(17));
    try testing.expectEqual(@as(u64, 7), largestPrimeFactor(14));
}

test "isPerfectPower - basic cases" {
    try testing.expect(isPerfectPower(1));
    try testing.expect(isPerfectPower(8)); // 2^3
    try testing.expect(isPerfectPower(27)); // 3^3
    try testing.expect(isPerfectPower(16)); // 2^4
    try testing.expect(!isPerfectPower(12));
    try testing.expect(!isPerfectPower(17));
}
