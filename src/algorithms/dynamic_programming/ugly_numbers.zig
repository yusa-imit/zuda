// Ugly Numbers - Dynamic Programming
//
// Problem: Find the nth ugly number. Ugly numbers are positive integers whose prime factors
// only include 2, 3, and 5. The sequence starts: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, ...
//
// Algorithm:
// - DP state: dp[i] = ith ugly number
// - Three pointers (i2, i3, i5) track multiples of 2, 3, 5
// - Recurrence: dp[i] = min(dp[i2]*2, dp[i3]*3, dp[i5]*5)
// - Advance pointer(s) that produced the minimum to avoid duplicates
//
// Time: O(n) - single pass through n ugly numbers
// Space: O(n) - DP array storage
//
// Use cases:
// - Number theory (smooth numbers, regular numbers)
// - Algorithm interview problems
// - Hamming numbers (generalization to more primes)
// - Sequence generation with constraints

const std = @import("std");
const testing = std.testing;

/// Find the nth ugly number (1-indexed).
/// Ugly numbers are positive integers whose only prime factors are 2, 3, and 5.
/// The sequence starts: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, ...
///
/// Time: O(n)
/// Space: O(n)
pub fn nthUgly(comptime T: type, allocator: std.mem.Allocator, n: usize) !T {
    if (n == 0) return error.InvalidIndex;

    // Base case: first ugly number is 1
    if (n == 1) return 1;

    // Allocate DP array
    const dp = try allocator.alloc(T, n);
    defer allocator.free(dp);

    dp[0] = 1;

    // Three pointers for multiples of 2, 3, 5
    var idx2: usize = 0;
    var idx3: usize = 0;
    var idx5: usize = 0;

    // Generate ugly numbers
    for (1..n) |i| {
        const next2 = dp[idx2] * 2;
        const next3 = dp[idx3] * 3;
        const next5 = dp[idx5] * 5;

        // Next ugly number is minimum of the three candidates
        dp[i] = @min(@min(next2, next3), next5);

        // Advance pointers that produced the minimum
        // (multiple pointers may advance if there are duplicates)
        if (dp[i] == next2) idx2 += 1;
        if (dp[i] == next3) idx3 += 1;
        if (dp[i] == next5) idx5 += 1;
    }

    return dp[n - 1];
}

/// Generate the first n ugly numbers and return as an array.
/// Caller owns the returned slice.
///
/// Time: O(n)
/// Space: O(n)
pub fn firstNUgly(comptime T: type, allocator: std.mem.Allocator, n: usize) ![]T {
    if (n == 0) return try allocator.alloc(T, 0);

    const result = try allocator.alloc(T, n);
    errdefer allocator.free(result);

    result[0] = 1;

    if (n == 1) return result;

    var idx2: usize = 0;
    var idx3: usize = 0;
    var idx5: usize = 0;

    for (1..n) |i| {
        const next2 = result[idx2] * 2;
        const next3 = result[idx3] * 3;
        const next5 = result[idx5] * 5;

        result[i] = @min(@min(next2, next3), next5);

        if (result[i] == next2) idx2 += 1;
        if (result[i] == next3) idx3 += 1;
        if (result[i] == next5) idx5 += 1;
    }

    return result;
}

/// Check if a number is ugly (only has prime factors 2, 3, 5).
///
/// Time: O(log n)
/// Space: O(1)
pub fn isUgly(comptime T: type, num: T) bool {
    const ti = @typeInfo(T);
    switch (ti) {
        .int => |int_info| {
            if (int_info.signedness == .signed) {
                if (num <= 0) return false;
            } else if (num == 0) {
                return false;
            }
        },
        else => {},
    }
    if (num == 1) return true;

    var n = num;

    // Divide by 2 as many times as possible
    while (@rem(n, 2) == 0) {
        n = @divTrunc(n, 2);
    }

    // Divide by 3 as many times as possible
    while (@rem(n, 3) == 0) {
        n = @divTrunc(n, 3);
    }

    // Divide by 5 as many times as possible
    while (@rem(n, 5) == 0) {
        n = @divTrunc(n, 5);
    }

    // If n becomes 1, then it only had factors 2, 3, 5
    return n == 1;
}

/// Find all ugly numbers less than or equal to a given limit.
/// Returns sorted array of ugly numbers.
///
/// Time: O(limit * log log limit)
/// Space: O(count of ugly numbers <= limit)
pub fn uglyNumbersUpTo(comptime T: type, allocator: std.mem.Allocator, limit: T) ![]T {
    if (limit < 1) return try allocator.alloc(T, 0);

    var list = try std.ArrayList(T).initCapacity(allocator, 100);
    defer list.deinit(allocator);

    // Generate ugly numbers using DP until we exceed limit
    try list.append(allocator, 1);

    var idx2: usize = 0;
    var idx3: usize = 0;
    var idx5: usize = 0;

    while (true) {
        const next2 = list.items[idx2] * 2;
        const next3 = list.items[idx3] * 3;
        const next5 = list.items[idx5] * 5;

        const next = @min(@min(next2, next3), next5);
        if (next > limit) break;

        try list.append(allocator, next);

        if (next == next2) idx2 += 1;
        if (next == next3) idx3 += 1;
        if (next == next5) idx5 += 1;
    }

    return try list.toOwnedSlice(allocator);
}

/// Count how many ugly numbers are less than or equal to n.
///
/// Time: O(n)
/// Space: O(1)
pub fn countUgly(comptime T: type, allocator: std.mem.Allocator, n: T) !usize {
    if (n < 1) return 0;

    const uglies = try uglyNumbersUpTo(T, allocator, n);
    defer allocator.free(uglies);

    return uglies.len;
}

/// Generalized ugly numbers with custom prime factors.
/// Find the nth number whose only prime factors are in the given primes array.
///
/// Time: O(n * p) where p = number of primes
/// Space: O(n + p)
pub fn nthUglyWithPrimes(comptime T: type, allocator: std.mem.Allocator, n: usize, primes: []const T) !T {
    if (n == 0) return error.InvalidIndex;
    if (primes.len == 0) return error.NoPrimes;

    if (n == 1) return 1;

    const dp = try allocator.alloc(T, n);
    defer allocator.free(dp);

    const indices = try allocator.alloc(usize, primes.len);
    defer allocator.free(indices);

    dp[0] = 1;
    @memset(indices, 0);

    for (1..n) |i| {
        var min_val: T = std.math.maxInt(T);

        // Find minimum among all prime multiples
        for (primes, 0..) |prime, j| {
            const candidate = dp[indices[j]] * prime;
            min_val = @min(min_val, candidate);
        }

        dp[i] = min_val;

        // Advance all indices that produced the minimum
        for (primes, 0..) |prime, j| {
            if (dp[indices[j]] * prime == min_val) {
                indices[j] += 1;
            }
        }
    }

    return dp[n - 1];
}

// Tests
test "nthUgly - basic sequence" {
    const allocator = testing.allocator;

    // First 10 ugly numbers: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12
    try testing.expectEqual(@as(u32, 1), try nthUgly(u32, allocator, 1));
    try testing.expectEqual(@as(u32, 2), try nthUgly(u32, allocator, 2));
    try testing.expectEqual(@as(u32, 3), try nthUgly(u32, allocator, 3));
    try testing.expectEqual(@as(u32, 4), try nthUgly(u32, allocator, 4));
    try testing.expectEqual(@as(u32, 5), try nthUgly(u32, allocator, 5));
    try testing.expectEqual(@as(u32, 6), try nthUgly(u32, allocator, 6));
    try testing.expectEqual(@as(u32, 8), try nthUgly(u32, allocator, 7));
    try testing.expectEqual(@as(u32, 9), try nthUgly(u32, allocator, 8));
    try testing.expectEqual(@as(u32, 10), try nthUgly(u32, allocator, 9));
    try testing.expectEqual(@as(u32, 12), try nthUgly(u32, allocator, 10));
}

test "nthUgly - larger indices" {
    const allocator = testing.allocator;

    // 15th ugly number is 24
    try testing.expectEqual(@as(u32, 24), try nthUgly(u32, allocator, 15));

    // 150th ugly number is 5832
    try testing.expectEqual(@as(u32, 5832), try nthUgly(u32, allocator, 150));
}

test "nthUgly - edge cases" {
    const allocator = testing.allocator;

    // Zero index is invalid
    try testing.expectError(error.InvalidIndex, nthUgly(u32, allocator, 0));

    // First ugly number
    try testing.expectEqual(@as(u32, 1), try nthUgly(u32, allocator, 1));
}

test "firstNUgly - basic sequence" {
    const allocator = testing.allocator;

    const result = try firstNUgly(u32, allocator, 10);
    defer allocator.free(result);

    const expected = [_]u32{ 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 };
    try testing.expectEqualSlices(u32, &expected, result);
}

test "firstNUgly - empty" {
    const allocator = testing.allocator;

    const result = try firstNUgly(u32, allocator, 0);
    defer allocator.free(result);

    try testing.expectEqual(@as(usize, 0), result.len);
}

test "firstNUgly - single element" {
    const allocator = testing.allocator;

    const result = try firstNUgly(u32, allocator, 1);
    defer allocator.free(result);

    try testing.expectEqual(@as(usize, 1), result.len);
    try testing.expectEqual(@as(u32, 1), result[0]);
}

test "firstNUgly - large sequence" {
    const allocator = testing.allocator;

    const result = try firstNUgly(u32, allocator, 100);
    defer allocator.free(result);

    try testing.expectEqual(@as(usize, 100), result.len);

    // Verify sequence is strictly increasing
    for (0..99) |i| {
        try testing.expect(result[i] < result[i + 1]);
    }

    // Verify all are ugly numbers
    for (result) |num| {
        try testing.expect(isUgly(u32, num));
    }

    // 100th ugly number is 1536
    try testing.expectEqual(@as(u32, 1536), result[99]);
}

test "isUgly - basic checks" {
    try testing.expect(isUgly(u32, 1));
    try testing.expect(isUgly(u32, 2));
    try testing.expect(isUgly(u32, 3));
    try testing.expect(isUgly(u32, 4));
    try testing.expect(isUgly(u32, 5));
    try testing.expect(isUgly(u32, 6));
    try testing.expect(isUgly(u32, 8));
    try testing.expect(isUgly(u32, 9));
    try testing.expect(isUgly(u32, 10));
    try testing.expect(isUgly(u32, 12));
}

test "isUgly - non-ugly numbers" {
    try testing.expect(!isUgly(u32, 7)); // 7 is prime
    try testing.expect(!isUgly(u32, 11)); // 11 is prime
    try testing.expect(!isUgly(u32, 13)); // 13 is prime
    try testing.expect(!isUgly(u32, 14)); // 14 = 2 * 7
    try testing.expect(isUgly(u32, 15)); // 15 = 3 * 5, is ugly!
    try testing.expect(!isUgly(u32, 21)); // 21 = 3 * 7
    try testing.expect(!isUgly(u32, 22)); // 22 = 2 * 11
}

test "isUgly - edge cases" {
    try testing.expect(!isUgly(i32, 0));
    try testing.expect(!isUgly(i32, -1));
    try testing.expect(isUgly(u32, 1));
}

test "isUgly - powers" {
    try testing.expect(isUgly(u32, 16)); // 2^4
    try testing.expect(isUgly(u32, 27)); // 3^3
    try testing.expect(isUgly(u32, 125)); // 5^3
    try testing.expect(isUgly(u32, 243)); // 3^5
}

test "uglyNumbersUpTo - basic range" {
    const allocator = testing.allocator;

    const result = try uglyNumbersUpTo(u32, allocator, 12);
    defer allocator.free(result);

    const expected = [_]u32{ 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 };
    try testing.expectEqualSlices(u32, &expected, result);
}

test "uglyNumbersUpTo - empty" {
    const allocator = testing.allocator;

    const result = try uglyNumbersUpTo(u32, allocator, 0);
    defer allocator.free(result);

    try testing.expectEqual(@as(usize, 0), result.len);
}

test "uglyNumbersUpTo - large range" {
    const allocator = testing.allocator;

    const result = try uglyNumbersUpTo(u32, allocator, 100);
    defer allocator.free(result);

    // Count should be 34 (verified: 1,2,3,4,5,6,8,9,10,12,15,16,18,20,24,25,27,30,32,36,40,45,48,50,54,60,64,72,75,80,81,90,96,100)
    try testing.expectEqual(@as(usize, 34), result.len);

    // Verify all are <= 100
    for (result) |num| {
        try testing.expect(num <= 100);
        try testing.expect(isUgly(u32, num));
    }
}

test "countUgly - basic counts" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(usize, 0), try countUgly(u32, allocator, 0));
    try testing.expectEqual(@as(usize, 1), try countUgly(u32, allocator, 1));
    try testing.expectEqual(@as(usize, 10), try countUgly(u32, allocator, 12));
    try testing.expectEqual(@as(usize, 34), try countUgly(u32, allocator, 100));
}

test "nthUglyWithPrimes - standard ugly (2,3,5)" {
    const allocator = testing.allocator;

    const primes = [_]u32{ 2, 3, 5 };

    try testing.expectEqual(@as(u32, 1), try nthUglyWithPrimes(u32, allocator, 1, &primes));
    try testing.expectEqual(@as(u32, 2), try nthUglyWithPrimes(u32, allocator, 2, &primes));
    try testing.expectEqual(@as(u32, 3), try nthUglyWithPrimes(u32, allocator, 3, &primes));
    try testing.expectEqual(@as(u32, 12), try nthUglyWithPrimes(u32, allocator, 10, &primes));
}

test "nthUglyWithPrimes - custom primes (2,7)" {
    const allocator = testing.allocator;

    const primes = [_]u32{ 2, 7 };

    // Sequence: 1, 2, 4, 7, 8, 14, 16, 28, ...
    try testing.expectEqual(@as(u32, 1), try nthUglyWithPrimes(u32, allocator, 1, &primes));
    try testing.expectEqual(@as(u32, 2), try nthUglyWithPrimes(u32, allocator, 2, &primes));
    try testing.expectEqual(@as(u32, 4), try nthUglyWithPrimes(u32, allocator, 3, &primes));
    try testing.expectEqual(@as(u32, 7), try nthUglyWithPrimes(u32, allocator, 4, &primes));
    try testing.expectEqual(@as(u32, 8), try nthUglyWithPrimes(u32, allocator, 5, &primes));
}

test "nthUglyWithPrimes - single prime" {
    const allocator = testing.allocator;

    const primes = [_]u32{2};

    // Sequence: 1, 2, 4, 8, 16, ...
    try testing.expectEqual(@as(u32, 1), try nthUglyWithPrimes(u32, allocator, 1, &primes));
    try testing.expectEqual(@as(u32, 2), try nthUglyWithPrimes(u32, allocator, 2, &primes));
    try testing.expectEqual(@as(u32, 4), try nthUglyWithPrimes(u32, allocator, 3, &primes));
    try testing.expectEqual(@as(u32, 8), try nthUglyWithPrimes(u32, allocator, 4, &primes));
}

test "nthUglyWithPrimes - error cases" {
    const allocator = testing.allocator;

    const primes = [_]u32{ 2, 3 };

    try testing.expectError(error.InvalidIndex, nthUglyWithPrimes(u32, allocator, 0, &primes));
    try testing.expectError(error.NoPrimes, nthUglyWithPrimes(u32, allocator, 1, &[_]u32{}));
}

test "nthUgly - type support (u8, u16, u64)" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(u8, 12), try nthUgly(u8, allocator, 10));
    try testing.expectEqual(@as(u16, 12), try nthUgly(u16, allocator, 10));
    try testing.expectEqual(@as(u64, 12), try nthUgly(u64, allocator, 10));
}

test "ugly numbers - memory safety" {
    const allocator = testing.allocator;

    // Test all allocating functions
    _ = try nthUgly(u32, allocator, 100);
    const arr = try firstNUgly(u32, allocator, 50);
    allocator.free(arr);
    const up = try uglyNumbersUpTo(u32, allocator, 200);
    allocator.free(up);
    _ = try countUgly(u32, allocator, 100);

    const primes = [_]u32{ 2, 3, 5 };
    _ = try nthUglyWithPrimes(u32, allocator, 50, &primes);
}
