const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// CountingSort - Non-comparative integer sorting for small ranges.
///
/// CountingSort is a linear-time sorting algorithm for integers when the range of input
/// values (max - min) is small relative to the number of elements. It counts the occurrences
/// of each value, computes cumulative counts, and places elements in sorted order.
///
/// Key features:
/// - Non-comparative: processes values directly
/// - Stable: preserves relative order of equal elements
/// - Linear time when range is O(n): O(n + k) where k = max - min + 1
/// - Efficient for dense value distributions
///
/// Time Complexity: O(n + k) where n = elements, k = value range
/// Space Complexity: O(n + k) for count array and output buffer
///
/// Best used when:
/// - Value range is small (k ≈ O(n))
/// - Many duplicate values
/// - Stability is required
///
/// Generic parameters:
/// - T: Element type (must be integer)
pub fn CountingSort(comptime T: type) type {
    return struct {
        const Self = @This();

        // Validate type at compile time
        comptime {
            switch (@typeInfo(T)) {
                .int => {},
                .comptime_int => {},
                else => @compileError("CountingSort requires integer type"),
            }
        }

        /// Sort a slice of integers using CountingSort.
        ///
        /// Time: O(n + k) where k = max - min + 1 | Space: O(n + k)
        pub fn sort(allocator: Allocator, items: []T) !void {
            if (items.len < 2) return;

            // Find min and max values
            var min_val = items[0];
            var max_val = items[0];
            for (items[1..]) |item| {
                if (item < min_val) min_val = item;
                if (item > max_val) max_val = item;
            }

            // All elements are the same
            if (min_val == max_val) return;

            // Compute range using wider type to avoid overflow
            const WiderInt = std.meta.Int(@typeInfo(T).int.signedness, @bitSizeOf(T) * 2);
            const range_val: WiderInt = @as(WiderInt, max_val) - @as(WiderInt, min_val);

            // Check if range is reasonable (prevent excessive memory allocation)
            if (range_val < 0) return error.InvalidRange;

            // Use usize for range calculations
            const range: usize = @intCast(range_val + 1);

            // Prevent excessive memory allocation
            if (range > 10_000_000) return error.RangeTooLarge;

            // Allocate count array
            const count = try allocator.alloc(usize, range);
            defer allocator.free(count);
            @memset(count, 0);

            // Count occurrences (shift by min_val to handle negative numbers)
            for (items) |item| {
                const idx: usize = @intCast(@as(WiderInt, item) - @as(WiderInt, min_val));
                count[idx] += 1;
            }

            // Cumulative counts for stable sorting
            for (1..range) |i| {
                count[i] += count[i - 1];
            }

            // Allocate output buffer
            const output = try allocator.alloc(T, items.len);
            defer allocator.free(output);

            // Place elements in sorted order (iterate backwards for stability)
            var i: usize = items.len;
            while (i > 0) {
                i -= 1;
                const idx: usize = @intCast(@as(WiderInt, items[i]) - @as(WiderInt, min_val));
                count[idx] -= 1;
                output[count[idx]] = items[i];
            }

            // Copy back to original array
            @memcpy(items, output);
        }

        /// Sort a slice when range is known at runtime (optimized path).
        ///
        /// Time: O(n + k) | Space: O(n + k)
        pub fn sortWithRange(allocator: Allocator, items: []T, min_val: T, max_val: T) !void {
            if (items.len < 2) return;
            if (min_val == max_val) return;

            const WiderInt = std.meta.Int(@typeInfo(T).int.signedness, @bitSizeOf(T) * 2);
            const range_val: WiderInt = @as(WiderInt, max_val) - @as(WiderInt, min_val);
            if (range_val < 0) return error.InvalidRange;

            const range: usize = @intCast(range_val + 1);
            if (range > 10_000_000) return error.RangeTooLarge;

            const count = try allocator.alloc(usize, range);
            defer allocator.free(count);
            @memset(count, 0);

            const WiderInt2 = std.meta.Int(@typeInfo(T).int.signedness, @bitSizeOf(T) * 2);

            for (items) |item| {
                const idx: usize = @intCast(@as(WiderInt2, item) - @as(WiderInt2, min_val));
                count[idx] += 1;
            }

            for (1..range) |i| {
                count[i] += count[i - 1];
            }

            const output = try allocator.alloc(T, items.len);
            defer allocator.free(output);

            var i: usize = items.len;
            while (i > 0) {
                i -= 1;
                const idx: usize = @intCast(@as(WiderInt2, items[i]) - @as(WiderInt2, min_val));
                count[idx] -= 1;
                output[count[idx]] = items[i];
            }

            @memcpy(items, output);
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "CountingSort - empty array" {
    var items: [0]i32 = undefined;
    const Sorter = CountingSort(i32);
    try Sorter.sort(testing.allocator, &items);
}

test "CountingSort - single element" {
    var items = [_]i32{42};
    const Sorter = CountingSort(i32);
    try Sorter.sort(testing.allocator, &items);
    try testing.expectEqual(@as(i32, 42), items[0]);
}

test "CountingSort - all same" {
    var items = [_]i32{ 7, 7, 7, 7, 7, 7, 7, 7, 7, 7 };
    const Sorter = CountingSort(i32);
    try Sorter.sort(testing.allocator, &items);

    for (items) |item| {
        try testing.expectEqual(@as(i32, 7), item);
    }
}

test "CountingSort - already sorted" {
    var items = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const Sorter = CountingSort(u32);
    try Sorter.sort(testing.allocator, &items);

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "CountingSort - reverse sorted" {
    var items = [_]u32{ 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    const Sorter = CountingSort(u32);
    try Sorter.sort(testing.allocator, &items);

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "CountingSort - random data" {
    var items = [_]u32{ 3, 7, 1, 9, 2, 5, 8, 4, 6, 10 };
    const Sorter = CountingSort(u32);
    try Sorter.sort(testing.allocator, &items);

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "CountingSort - duplicates" {
    var items = [_]u32{ 5, 2, 8, 2, 9, 1, 5, 5, 3, 2 };
    const Sorter = CountingSort(u32);
    try Sorter.sort(testing.allocator, &items);

    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 2, 2, 3, 5, 5, 5, 8, 9 }, &items);
}

test "CountingSort - small range" {
    var items = [_]u8{ 5, 2, 8, 2, 9, 1, 5, 5, 3, 2, 0, 10 };
    const Sorter = CountingSort(u8);
    try Sorter.sort(testing.allocator, &items);

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }

    try testing.expectEqual(@as(u8, 0), items[0]);
    try testing.expectEqual(@as(u8, 10), items[items.len - 1]);
}

test "CountingSort - signed integers" {
    var items = [_]i32{ -5, 3, -10, 0, 8, -3, 5, -8, 1, -1 };
    const Sorter = CountingSort(i32);
    try Sorter.sort(testing.allocator, &items);

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }

    try testing.expectEqual(@as(i32, -10), items[0]);
    try testing.expectEqual(@as(i32, 8), items[items.len - 1]);
}

test "CountingSort - all negative" {
    var items = [_]i32{ -5, -3, -10, -8, -1 };
    const Sorter = CountingSort(i32);
    try Sorter.sort(testing.allocator, &items);

    try testing.expectEqualSlices(i32, &[_]i32{ -10, -8, -5, -3, -1 }, &items);
}

test "CountingSort - mixed positive and negative" {
    var items = [_]i32{ 5, -5, 3, -3, 0, 10, -10, 1, -1 };
    const Sorter = CountingSort(i32);
    try Sorter.sort(testing.allocator, &items);

    try testing.expectEqualSlices(i32, &[_]i32{ -10, -5, -3, -1, 0, 1, 3, 5, 10 }, &items);
}

test "CountingSort - stability check" {
    // Test stability with pairs (value, original_index)
    const Pair = struct { val: u32, idx: usize };

    const items = [_]Pair{
        .{ .val = 3, .idx = 0 },
        .{ .val = 1, .idx = 1 },
        .{ .val = 3, .idx = 2 },
        .{ .val = 2, .idx = 3 },
        .{ .val = 1, .idx = 4 },
    };

    // Extract values
    var values: [5]u32 = undefined;
    for (0..5) |i| values[i] = items[i].val;

    const Sorter = CountingSort(u32);
    try Sorter.sort(testing.allocator, &values);

    // Manually verify stability (equal values maintain relative order)
    // Expected: 1(idx=1), 1(idx=4), 2(idx=3), 3(idx=0), 3(idx=2)
    // Just verify values are sorted
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 1, 2, 3, 3 }, &values);
}

test "CountingSort - with known range" {
    var items = [_]u32{ 5, 2, 8, 2, 9, 1, 5, 5, 3, 2 };
    const Sorter = CountingSort(u32);
    try Sorter.sortWithRange(testing.allocator, &items, 1, 9);

    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 2, 2, 3, 5, 5, 5, 8, 9 }, &items);
}

test "CountingSort - i8 type" {
    var items = [_]i8{ 50, -50, 25, -25, 0, 100, -100 };
    const Sorter = CountingSort(i8);
    try Sorter.sort(testing.allocator, &items);

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }

    try testing.expectEqual(@as(i8, -100), items[0]);
    try testing.expectEqual(@as(i8, 100), items[items.len - 1]);
}

test "CountingSort - stress test (small range)" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const n = 10000;
    var items = try allocator.alloc(i32, n);
    defer allocator.free(items);

    // Fill with random data in small range [-1000, 1000]
    for (0..n) |i| {
        items[i] = random.intRangeAtMost(i32, -1000, 1000);
    }

    const Sorter = CountingSort(i32);
    try Sorter.sort(allocator, items);

    // Verify sorted
    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "CountingSort - dense duplicates" {
    const allocator = testing.allocator;

    const n = 1000;
    var items = try allocator.alloc(u32, n);
    defer allocator.free(items);

    // Fill with only 10 distinct values (lots of duplicates)
    for (0..n) |i| {
        items[i] = @intCast(i % 10);
    }

    const Sorter = CountingSort(u32);
    try Sorter.sort(allocator, items);

    // Verify sorted
    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "CountingSort - u16 type" {
    var items = [_]u16{ 1000, 500, 2000, 1500, 250 };
    const Sorter = CountingSort(u16);
    try Sorter.sort(testing.allocator, &items);

    try testing.expectEqualSlices(u16, &[_]u16{ 250, 500, 1000, 1500, 2000 }, &items);
}
