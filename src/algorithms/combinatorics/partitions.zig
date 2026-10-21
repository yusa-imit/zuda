//! Integer Partitions and Set Partitions
//!
//! This module provides algorithms for:
//! - Integer partitions (partitioning a number n into k parts)
//! - Partition counting (number of ways to partition n)
//! - Partition generation (all partitions of n)
//! - Set partitions (partitioning a set into k subsets)
//! - Bell numbers (total number of set partitions)
//! - Derangements (permutations with no fixed points)
//!
//! ## Performance Characteristics
//!
//! - **countPartitions(n, k)**: O(n×k) time, O(k) space - count integer partitions
//! - **generatePartitions(n)**: O(p(n)) time/space where p(n) is partition function
//! - **bellNumber(n)**: O(n²) time, O(n) space - count set partitions
//! - **derangements(n)**: O(n) time, O(1) space - count permutations with no fixed points
//!
//! ## Use Cases
//!
//! - **Number Theory**: Partition function analysis
//! - **Combinatorial Optimization**: Subset sum, bin packing variants
//! - **Cryptography**: Key space analysis
//! - **Algorithm Analysis**: Counting problem instances
//! - **Probability**: Computing distributions over partitions
//!
//! ## Example
//!
//! ```zig
//! const std = @import("std");
//! const partitions = @import("zuda").algorithms.combinatorics.partitions;
//!
//! // Count partitions of 5 into 2 parts: {4+1, 3+2} = 2 ways
//! const count = try partitions.countPartitions(u32, allocator, 5, 2);
//!
//! // Generate all partitions of 4: [4], [3,1], [2,2], [2,1,1], [1,1,1,1]
//! var gpa = std.heap.DebugAllocator(.{}){};
//! const allocator = gpa.allocator();
//! const parts = try partitions.generatePartitions(u32, allocator, 4);
//! defer {
//!     for (parts.items) |part| allocator.free(part);
//!     parts.deinit(allocator);
//! }
//!
//! // Count derangements of n=3: !3 = 2 (permutations with no fixed points)
//! const derang = try partitions.derangements(u64, 3);
//! ```

const std = @import("std");
const mem = std.mem;
const math = std.math;
const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;
const testing = std.testing;

/// Count the number of ways to partition integer n into exactly k parts
///
/// Uses conjugation: partitions of n into exactly k parts correspond one-to-one to
/// partitions of n - k into parts of size at most k. The latter is counted with the
/// classic unbounded-knapsack recurrence ways[s] += ways[s - part] over part = 1..k.
///
/// Returns the count of partitions, or error.Overflow if result exceeds type T.
///
/// Time: O(k×(n-k)) | Space: O(n-k)
pub fn countPartitions(comptime T: type, allocator: Allocator, n: T, k: T) !T {
    if (n < 0 or k < 0) return error.NegativeInput;
    if (k == 0) return if (n == 0) 1 else 0;
    if (k > n) return 0;
    if (k == 1 or k == n) return 1;

    const sum_max: usize = @intCast(n - k);
    const part_max: usize = @intCast(k);
    std.debug.assert(sum_max >= 1);
    std.debug.assert(part_max >= 2);

    const ways = try allocator.alloc(T, sum_max + 1);
    defer allocator.free(ways);

    @memset(ways, 0);
    ways[0] = 1;

    var part: usize = 1;
    while (part <= part_max) : (part += 1) {
        var s: usize = part;
        while (s <= sum_max) : (s += 1) {
            ways[s] = math.add(T, ways[s], ways[s - part]) catch return error.Overflow;
        }
    }

    return ways[sum_max];
}

/// Generate all integer partitions of n
///
/// Returns an ArrayList of partitions, where each partition is a slice of integers
/// summing to n, in non-increasing order.
///
/// Example: partitions of 4 = [[4], [3,1], [2,2], [2,1,1], [1,1,1,1]]
///
/// Caller must free each partition slice and call deinit(allocator) on the returned list.
///
/// Time: O(p(n)) where p(n) is the partition function | Space: O(p(n))
pub fn generatePartitions(comptime T: type, allocator: Allocator, n: T) !ArrayList([]T) {
    if (n < 0) return error.NegativeInput;

    var result: ArrayList([]T) = .empty;
    errdefer {
        for (result.items) |part| allocator.free(part);
        result.deinit(allocator);
    }

    if (n == 0) {
        // Empty partition
        const empty = try allocator.alloc(T, 0);
        errdefer allocator.free(empty);

        try result.append(allocator, empty);
        return result;
    }

    var current: ArrayList(T) = .empty;
    defer current.deinit(allocator);

    try generatePartitionsHelper(T, allocator, n, n, &current, &result);
    return result;
}

/// Recursive helper for partition generation
fn generatePartitionsHelper(
    comptime T: type,
    allocator: Allocator,
    remaining: T,
    max_value: T,
    current: *ArrayList(T),
    result: *ArrayList([]T),
) !void {
    if (remaining == 0) {
        // Found a complete partition
        const partition = try allocator.dupe(T, current.items);
        errdefer allocator.free(partition);

        try result.append(allocator, partition);
        return;
    }

    // Try all values from max_value down to 1
    var value = @min(remaining, max_value);
    while (value >= 1) : (value -= 1) {
        try current.append(allocator, value);
        try generatePartitionsHelper(T, allocator, remaining - value, value, current, result);
        _ = current.pop();
    }
}

/// Compute the nth Bell number (number of ways to partition a set of n elements)
///
/// Uses dynamic programming with Bell triangle.
/// B(n+1) = sum over k of C(n,k) * B(k)
///
/// Bell numbers grow very rapidly: B(10) = 115,975
///
/// Time: O(n²) | Space: O(n)
pub fn bellNumber(comptime T: type, allocator: Allocator, n: T) !T {
    if (n < 0) return error.NegativeInput;
    if (n == 0) return 1;

    // Bell triangle: row 0 is [1]; row i starts with the last element of row i-1 and
    // each next element is its left neighbour plus the element above-left.
    // B(n) is the last element of row n-1, so B(n+1) is never computed.
    const row_max: usize = @intCast(n);
    std.debug.assert(row_max >= 1);

    const rows = try allocator.alloc(T, 2 * row_max);
    defer allocator.free(rows);

    var prev = rows[0..row_max];
    var curr = rows[row_max..];
    prev[0] = 1;

    var i: usize = 1;
    while (i < row_max) : (i += 1) {
        curr[0] = prev[i - 1];
        var j: usize = 1;
        while (j <= i) : (j += 1) {
            curr[j] = math.add(T, curr[j - 1], prev[j - 1]) catch return error.Overflow;
        }
        const tmp = prev;
        prev = curr;
        curr = tmp;
    }

    return prev[row_max - 1];
}

/// Count derangements of n elements (!n)
///
/// Derangement: permutation where no element appears in its original position.
/// Formula: !n = n! * sum_{i=0}^{n} (-1)^i / i!
/// Recurrence: !n = (n-1) * (!(n-1) + !(n-2))
///
/// Example: !3 = 2 (permutations [2,3,1] and [3,1,2])
///
/// Time: O(n) | Space: O(1)
pub fn derangements(comptime T: type, n: T) !T {
    if (n < 0) return error.NegativeInput;
    if (n == 0) return 1;
    if (n == 1) return 0;
    if (n == 2) return 1;

    // Use recurrence: !n = (n-1) * (!(n-1) + !(n-2))
    var d_prev2: T = 1; // !(n-2)
    var d_prev1: T = 0; // !(n-1)

    var i: T = 2;
    while (i <= n) : (i += 1) {
        const d_curr = math.mul(T, i - 1, math.add(T, d_prev1, d_prev2) catch return error.Overflow) catch return error.Overflow;
        d_prev2 = d_prev1;
        d_prev1 = d_curr;
    }

    return d_prev1;
}

/// Compute multinomial coefficient: n! / (k1! * k2! * ... * km!)
///
/// Returns the number of ways to partition n items into m groups of sizes k1, k2, ..., km.
/// Requires that k1 + k2 + ... + km = n.
///
/// Time: O(m) | Space: O(1)
pub fn multinomial(comptime T: type, parts: []const T) !T {
    if (parts.len == 0) return 1;

    // Verify parts sum to n
    var total: T = 0;
    for (parts) |part| {
        if (part < 0) return error.NegativeInput;
        total = math.add(T, total, part) catch return error.Overflow;
    }

    // Compute n! / (k1! * k2! * ... * km!) incrementally
    var result: T = 1;
    var consumed: T = 0;

    for (parts) |part| {
        if (part == 0) continue;

        // Multiply by C(consumed + part, part)
        var numerator: T = 1;
        var denominator: T = 1;

        var i: T = 0;
        while (i < part) : (i += 1) {
            numerator = math.mul(T, numerator, consumed + part - i) catch return error.Overflow;
            denominator = math.mul(T, denominator, i + 1) catch return error.Overflow;
        }

        result = math.mul(T, result, @divTrunc(numerator, denominator)) catch return error.Overflow;
        consumed = math.add(T, consumed, part) catch return error.Overflow;
    }

    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "countPartitions: basic cases" {
    try testing.expectEqual(@as(u32, 1), try countPartitions(u32, testing.allocator, 0, 0));
    try testing.expectEqual(@as(u32, 0), try countPartitions(u32, testing.allocator, 5, 0));
    try testing.expectEqual(@as(u32, 0), try countPartitions(u32, testing.allocator, 3, 5)); // k > n
    try testing.expectEqual(@as(u32, 1), try countPartitions(u32, testing.allocator, 5, 1)); // [5]
    try testing.expectEqual(@as(u32, 1), try countPartitions(u32, testing.allocator, 5, 5)); // [1,1,1,1,1]
}

test "countPartitions: partitions of 5 into 2 parts" {
    // {4+1, 3+2} = 2 ways
    try testing.expectEqual(@as(u32, 2), try countPartitions(u32, testing.allocator, 5, 2));
}

test "countPartitions: partitions of 6 into 3 parts" {
    // {4+1+1, 3+2+1, 2+2+2} = 3 ways
    try testing.expectEqual(@as(u32, 3), try countPartitions(u32, testing.allocator, 6, 3));
}

test "countPartitions: partitions of 10 into 4 parts" {
    // Multiple ways, verify count
    const count = try countPartitions(u32, testing.allocator, 10, 4);
    try testing.expectEqual(@as(u32, 9), count);
}

test "countPartitions: matches enumeration for every k" {
    const gpa = testing.allocator;
    const n_max: u32 = 9;

    var n: u32 = 1;
    while (n <= n_max) : (n += 1) {
        var parts = try generatePartitions(u32, gpa, n);
        defer {
            for (parts.items) |part| gpa.free(part);
            parts.deinit(gpa);
        }

        var k: u32 = 1;
        while (k <= n) : (k += 1) {
            var expected: u32 = 0;
            for (parts.items) |part| {
                if (part.len == k) expected += 1;
            }
            try testing.expectEqual(expected, try countPartitions(u32, gpa, n, k));
        }
    }
}

test "generatePartitions: empty partition" {
    const allocator = testing.allocator;
    var parts = try generatePartitions(u32, allocator, 0);
    defer {
        for (parts.items) |part| allocator.free(part);
        parts.deinit(allocator);
    }

    try testing.expectEqual(@as(usize, 1), parts.items.len);
    try testing.expectEqual(@as(usize, 0), parts.items[0].len);
}

test "generatePartitions: n=1" {
    const allocator = testing.allocator;
    var parts = try generatePartitions(u32, allocator, 1);
    defer {
        for (parts.items) |part| allocator.free(part);
        parts.deinit(allocator);
    }

    try testing.expectEqual(@as(usize, 1), parts.items.len);
    try testing.expectEqualSlices(u32, &[_]u32{1}, parts.items[0]);
}

test "generatePartitions: n=4" {
    const allocator = testing.allocator;
    var parts = try generatePartitions(u32, allocator, 4);
    defer {
        for (parts.items) |part| allocator.free(part);
        parts.deinit(allocator);
    }

    // Expected: [4], [3,1], [2,2], [2,1,1], [1,1,1,1]
    try testing.expectEqual(@as(usize, 5), parts.items.len);

    // Verify each partition sums to 4
    for (parts.items) |part| {
        var sum: u32 = 0;
        for (part) |val| sum += val;
        try testing.expectEqual(@as(u32, 4), sum);
    }
}

test "generatePartitions: n=5" {
    const allocator = testing.allocator;
    var parts = try generatePartitions(u32, allocator, 5);
    defer {
        for (parts.items) |part| allocator.free(part);
        parts.deinit(allocator);
    }

    // p(5) = 7 partitions
    try testing.expectEqual(@as(usize, 7), parts.items.len);

    // Verify all sum to 5
    for (parts.items) |part| {
        var sum: u32 = 0;
        for (part) |val| sum += val;
        try testing.expectEqual(@as(u32, 5), sum);
    }
}

test "bellNumber: small values" {
    try testing.expectEqual(@as(u32, 1), try bellNumber(u32, testing.allocator, 0)); // B(0) = 1
    try testing.expectEqual(@as(u32, 1), try bellNumber(u32, testing.allocator, 1)); // B(1) = 1
    try testing.expectEqual(@as(u32, 2), try bellNumber(u32, testing.allocator, 2)); // B(2) = 2
    try testing.expectEqual(@as(u32, 5), try bellNumber(u32, testing.allocator, 3)); // B(3) = 5
    try testing.expectEqual(@as(u32, 15), try bellNumber(u32, testing.allocator, 4)); // B(4) = 15
    try testing.expectEqual(@as(u32, 52), try bellNumber(u32, testing.allocator, 5)); // B(5) = 52
}

test "bellNumber: overflow is reported, not wrapped" {
    // B(7) = 877 does not fit in u8.
    try testing.expectError(error.Overflow, bellNumber(u8, testing.allocator, 7));
    try testing.expectEqual(@as(u32, 877), try bellNumber(u32, testing.allocator, 7));
}

test "bellNumber: B(6) = 203" {
    try testing.expectEqual(@as(u32, 203), try bellNumber(u32, testing.allocator, 6));
}

test "bellNumber: B(10) = 115975" {
    try testing.expectEqual(@as(u64, 115975), try bellNumber(u64, testing.allocator, 10));
}

test "derangements: small values" {
    try testing.expectEqual(@as(u32, 1), try derangements(u32, 0)); // !0 = 1
    try testing.expectEqual(@as(u32, 0), try derangements(u32, 1)); // !1 = 0
    try testing.expectEqual(@as(u32, 1), try derangements(u32, 2)); // !2 = 1
    try testing.expectEqual(@as(u32, 2), try derangements(u32, 3)); // !3 = 2
    try testing.expectEqual(@as(u32, 9), try derangements(u32, 4)); // !4 = 9
    try testing.expectEqual(@as(u32, 44), try derangements(u32, 5)); // !5 = 44
}

test "derangements: !10 = 1334961" {
    try testing.expectEqual(@as(u64, 1334961), try derangements(u64, 10));
}

test "multinomial: basic cases" {
    // Empty partition
    const empty = [_]u32{};
    try testing.expectEqual(@as(u32, 1), try multinomial(u32, &empty));

    // Single group
    try testing.expectEqual(@as(u32, 1), try multinomial(u32, &[_]u32{5}));

    // Two equal groups: 6!/(3!*3!) = 20
    try testing.expectEqual(@as(u32, 20), try multinomial(u32, &[_]u32{ 3, 3 }));

    // Three groups: 10!/(2!*3!*5!) = 2520
    try testing.expectEqual(@as(u32, 2520), try multinomial(u32, &[_]u32{ 2, 3, 5 }));
}

test "multinomial: with zeros" {
    // Zeros should be ignored
    // n = 3, groups (0, 3, 0): 3! / (0! * 3! * 0!) = 1
    try testing.expectEqual(@as(u32, 1), try multinomial(u32, &[_]u32{ 0, 3, 0 }));
}

test "multinomial: four groups" {
    // 12!/(3!*3!*3!*3!) = 369600
    try testing.expectEqual(@as(u64, 369600), try multinomial(u64, &[_]u64{ 3, 3, 3, 3 }));
}

test "countPartitions: negative input error" {
    try testing.expectError(error.NegativeInput, countPartitions(i32, testing.allocator, -5, 2));
    try testing.expectError(error.NegativeInput, countPartitions(i32, testing.allocator, 5, -2));
}

test "generatePartitions: negative input error" {
    const allocator = testing.allocator;
    try testing.expectError(error.NegativeInput, generatePartitions(i32, allocator, -3));
}

test "bellNumber: negative input error" {
    try testing.expectError(error.NegativeInput, bellNumber(i32, testing.allocator, -1));
}

test "derangements: negative input error" {
    try testing.expectError(error.NegativeInput, derangements(i32, -5));
}

test "multinomial: negative part error" {
    try testing.expectError(error.NegativeInput, multinomial(i32, &[_]i32{ 2, -3, 5 }));
}

test "partitions: memory safety" {
    const allocator = testing.allocator;

    // Multiple generations without leaks
    var i: u32 = 1;
    while (i <= 6) : (i += 1) {
        var parts = try generatePartitions(u32, allocator, i);
        for (parts.items) |part| allocator.free(part);
        parts.deinit(allocator);
    }
}

test "partitions: type variants" {
    // u8
    try testing.expectEqual(@as(u8, 2), try countPartitions(u8, testing.allocator, 5, 2));
    try testing.expectEqual(@as(u8, 1), try bellNumber(u8, testing.allocator, 0));
    try testing.expectEqual(@as(u8, 2), try derangements(u8, 3));

    // u16
    try testing.expectEqual(@as(u16, 3), try countPartitions(u16, testing.allocator, 6, 3));
    try testing.expectEqual(@as(u16, 52), try bellNumber(u16, testing.allocator, 5));
    try testing.expectEqual(@as(u16, 44), try derangements(u16, 5));

    // u64
    try testing.expectEqual(@as(u64, 1), try bellNumber(u64, testing.allocator, 0));
    try testing.expectEqual(@as(u64, 1334961), try derangements(u64, 10));
}
