const std = @import("std");
const Allocator = std.mem.Allocator;

/// Longest Increasing Subsequence (LIS)
///
/// Provides multiple algorithms for finding the longest strictly increasing subsequence.

/// Find LIS length using dynamic programming
/// Time: O(n²) | Space: O(n)
pub fn lengthDP(comptime T: type, items: []const T) !usize {
    if (items.len == 0) return 0;
    if (items.len == 1) return 1;

    const allocator = std.heap.page_allocator;
    const dp = try allocator.alloc(usize, items.len);
    defer allocator.free(dp);

    @memset(dp, 1);

    var max_len: usize = 1;
    for (items, 0..) |item, i| {
        for (0..i) |j| {
            if (items[j] < item) {
                dp[i] = @max(dp[i], dp[j] + 1);
            }
        }
        max_len = @max(max_len, dp[i]);
    }

    return max_len;
}

/// Find LIS length using binary search optimization
/// Time: O(n log n) | Space: O(n)
pub fn lengthBinarySearch(comptime T: type, items: []const T) !usize {
    if (items.len == 0) return 0;
    if (items.len == 1) return 1;

    const allocator = std.heap.page_allocator;
    const tails = try allocator.alloc(T, items.len);
    defer allocator.free(tails);

    var len: usize = 0;

    for (items) |item| {
        // Binary search for the position to insert/replace
        var left: usize = 0;
        var right: usize = len;

        while (left < right) {
            const mid = left + (right - left) / 2;
            if (tails[mid] < item) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        tails[left] = item;
        if (left == len) {
            len += 1;
        }
    }

    return len;
}

/// Find LIS and return the actual subsequence
/// Time: O(n log n) | Space: O(n)
pub fn findSequence(comptime T: type, allocator: Allocator, items: []const T) ![]T {
    if (items.len == 0) return try allocator.alloc(T, 0);
    if (items.len == 1) {
        const result = try allocator.alloc(T, 1);
        result[0] = items[0];
        return result;
    }

    const tails = try allocator.alloc(T, items.len);
    defer allocator.free(tails);

    const indices = try allocator.alloc(usize, items.len);
    defer allocator.free(indices);

    const prev = try allocator.alloc(?usize, items.len);
    defer allocator.free(prev);

    @memset(prev, null);

    var len: usize = 0;

    for (items, 0..) |item, i| {
        var left: usize = 0;
        var right: usize = len;

        while (left < right) {
            const mid = left + (right - left) / 2;
            if (tails[mid] < item) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        tails[left] = item;
        indices[left] = i;

        if (left > 0) {
            prev[i] = indices[left - 1];
        }

        if (left == len) {
            len += 1;
        }
    }

    // Reconstruct the sequence
    const result = try allocator.alloc(T, len);
    var idx: ?usize = indices[len - 1];
    var pos: usize = len;

    while (idx) |i| {
        pos -= 1;
        result[pos] = items[i];
        idx = prev[i];
    }

    return result;
}

/// Find LIS with custom comparator
/// Time: O(n log n) | Space: O(n)
pub fn lengthWithComparator(
    comptime T: type,
    items: []const T,
    comptime lessThan: fn (T, T) bool,
) !usize {
    if (items.len == 0) return 0;
    if (items.len == 1) return 1;

    const allocator = std.heap.page_allocator;
    const tails = try allocator.alloc(T, items.len);
    defer allocator.free(tails);

    var len: usize = 0;

    for (items) |item| {
        var left: usize = 0;
        var right: usize = len;

        while (left < right) {
            const mid = left + (right - left) / 2;
            if (lessThan(tails[mid], item)) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        tails[left] = item;
        if (left == len) {
            len += 1;
        }
    }

    return len;
}

/// Find Longest Non-Decreasing Subsequence (allows equal elements)
/// Time: O(n log n) | Space: O(n)
pub fn lengthNonDecreasing(comptime T: type, items: []const T) !usize {
    if (items.len == 0) return 0;
    if (items.len == 1) return 1;

    const allocator = std.heap.page_allocator;
    const tails = try allocator.alloc(T, items.len);
    defer allocator.free(tails);

    var len: usize = 0;

    for (items) |item| {
        var left: usize = 0;
        var right: usize = len;

        // Find the leftmost position where tails[mid] > item
        while (left < right) {
            const mid = left + (right - left) / 2;
            if (tails[mid] <= item) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        tails[left] = item;
        if (left == len) {
            len += 1;
        }
    }

    return len;
}

// ============================================================================
// Tests
// ============================================================================

test "LIS: empty array" {
    const items: []const i32 = &[_]i32{};
    try std.testing.expectEqual(0, try lengthDP(i32, items));
    try std.testing.expectEqual(0, try lengthBinarySearch(i32, items));

    const seq = try findSequence(i32, std.testing.allocator, items);
    defer std.testing.allocator.free(seq);
    try std.testing.expectEqual(0, seq.len);
}

test "LIS: single element" {
    const items = [_]i32{42};
    try std.testing.expectEqual(1, try lengthDP(i32, &items));
    try std.testing.expectEqual(1, try lengthBinarySearch(i32, &items));

    const seq = try findSequence(i32, std.testing.allocator, &items);
    defer std.testing.allocator.free(seq);
    try std.testing.expectEqual(1, seq.len);
    try std.testing.expectEqual(42, seq[0]);
}

test "LIS: all increasing" {
    const items = [_]i32{ 1, 2, 3, 4, 5 };
    try std.testing.expectEqual(5, try lengthDP(i32, &items));
    try std.testing.expectEqual(5, try lengthBinarySearch(i32, &items));

    const seq = try findSequence(i32, std.testing.allocator, &items);
    defer std.testing.allocator.free(seq);
    try std.testing.expectEqual(5, seq.len);
    try std.testing.expectEqualSlices(i32, &items, seq);
}

test "LIS: all decreasing" {
    const items = [_]i32{ 5, 4, 3, 2, 1 };
    try std.testing.expectEqual(1, try lengthDP(i32, &items));
    try std.testing.expectEqual(1, try lengthBinarySearch(i32, &items));

    const seq = try findSequence(i32, std.testing.allocator, &items);
    defer std.testing.allocator.free(seq);
    try std.testing.expectEqual(1, seq.len);
}

test "LIS: mixed sequence" {
    const items = [_]i32{ 10, 9, 2, 5, 3, 7, 101, 18 };
    // LIS: 2, 3, 7, 101 or 2, 3, 7, 18 (length 4)
    try std.testing.expectEqual(4, try lengthDP(i32, &items));
    try std.testing.expectEqual(4, try lengthBinarySearch(i32, &items));

    const seq = try findSequence(i32, std.testing.allocator, &items);
    defer std.testing.allocator.free(seq);
    try std.testing.expectEqual(4, seq.len);

    // Verify it's actually increasing
    for (1..seq.len) |i| {
        try std.testing.expect(seq[i - 1] < seq[i]);
    }
}

test "LIS: duplicates (strictly increasing)" {
    const items = [_]i32{ 1, 3, 3, 4, 5 };
    // Strictly increasing, so can't use duplicate 3s
    try std.testing.expectEqual(4, try lengthDP(i32, &items));
    try std.testing.expectEqual(4, try lengthBinarySearch(i32, &items));
}

test "LIS: non-decreasing allows duplicates" {
    const items = [_]i32{ 1, 3, 3, 4, 5 };
    // Non-decreasing allows equal elements
    try std.testing.expectEqual(5, try lengthNonDecreasing(i32, &items));
}

test "LIS: custom comparator (reverse order)" {
    const items = [_]i32{ 5, 4, 3, 2, 1 };
    const greaterThan = struct {
        fn gt(a: i32, b: i32) bool {
            return a > b;
        }
    }.gt;

    // With reverse comparator, this is longest decreasing sequence
    try std.testing.expectEqual(5, try lengthWithComparator(i32, &items, greaterThan));
}

test "LIS: large sequence" {
    const allocator = std.testing.allocator;
    const n = 1000;
    const items = try allocator.alloc(i32, n);
    defer allocator.free(items);

    // Generate sequence: i % 100
    for (0..n) |i| {
        items[i] = @intCast(i % 100);
    }

    const len1 = try lengthDP(i32, items);
    const len2 = try lengthBinarySearch(i32, items);
    try std.testing.expectEqual(len1, len2);

    const seq = try findSequence(i32, allocator, items);
    defer allocator.free(seq);
    try std.testing.expectEqual(len2, seq.len);

    // Verify strictly increasing
    for (1..seq.len) |i| {
        try std.testing.expect(seq[i - 1] < seq[i]);
    }
}

test "LIS: negative numbers" {
    const items = [_]i32{ -3, -1, -4, 0, 2, -2, 5 };
    // LIS: -3, -1, 0, 2, 5 (length 5)
    try std.testing.expectEqual(5, try lengthDP(i32, &items));
    try std.testing.expectEqual(5, try lengthBinarySearch(i32, &items));

    const seq = try findSequence(i32, std.testing.allocator, &items);
    defer std.testing.allocator.free(seq);
    try std.testing.expectEqual(5, seq.len);

    for (1..seq.len) |i| {
        try std.testing.expect(seq[i - 1] < seq[i]);
    }
}

test "LIS: floating point" {
    const items = [_]f64{ 1.5, 2.3, 1.8, 3.2, 2.9, 4.1 };
    // LIS: 1.5, 1.8, 2.9, 4.1 or 1.5, 2.3, 2.9, 4.1 (length 4)
    try std.testing.expectEqual(4, try lengthBinarySearch(f64, &items));

    const seq = try findSequence(f64, std.testing.allocator, &items);
    defer std.testing.allocator.free(seq);
    try std.testing.expectEqual(4, seq.len);

    for (1..seq.len) |i| {
        try std.testing.expect(seq[i - 1] < seq[i]);
    }
}

test "LIS: performance comparison DP vs Binary Search" {
    const allocator = std.testing.allocator;
    const n = 500;
    const items = try allocator.alloc(i32, n);
    defer allocator.free(items);

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (0..n) |i| {
        items[i] = random.intRangeAtMost(i32, 0, 1000);
    }

    const len1 = try lengthDP(i32, items);
    const len2 = try lengthBinarySearch(i32, items);

    // Both should give the same result
    try std.testing.expectEqual(len1, len2);
}
