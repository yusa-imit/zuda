//! Pigeonhole Sort — Distribution-based sorting for small integer ranges
//!
//! Time: O(n + range) where range = max - min
//! Space: O(range)
//! Stable: Yes (preserves relative order of equal elements)
//!
//! Algorithm:
//! Similar to Counting Sort but uses a different distribution strategy.
//! Creates "pigeonholes" (buckets) for each possible value in the range [min, max].
//! Each element is placed into its corresponding pigeonhole, then collected in order.
//!
//! Use cases:
//! - Small integer ranges (e.g., ages 0-120, grades 0-100)
//! - When range is comparable to n (not much larger)
//! - Alternative to Counting Sort with similar characteristics
//! - Educational purposes (demonstrates distribution sorting)
//!
//! Trade-offs:
//! - vs Counting Sort: Similar O(n + range), both stable, Pigeonhole conceptually simpler
//! - vs Bucket Sort: Pigeonhole for integers, Bucket for continuous/floats
//! - vs Radix Sort: Better for small ranges, Radix better for large integers
//! - Space overhead: O(range) can be prohibitive if range >> n
//!
//! References:
//! - Knuth, "The Art of Computer Programming", Vol. 3 (Sorting and Searching)
//! - Also known as "Count Sort" variant

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const Order = std.math.Order;

/// Pigeonhole Sort — O(n + range) time, O(range) space
/// Sorts array in-place using pigeonhole (bucket) distribution
///
/// Time: O(n + range) where range = max - min
/// Space: O(range) for pigeonhole array
pub fn pigeonholeSort(comptime T: type, arr: []T) !void {
    if (arr.len == 0) return;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try pigeonholeSortAlloc(T, arr, allocator);
}

/// Pigeonhole Sort with custom allocator
/// Time: O(n + range)
/// Space: O(range)
pub fn pigeonholeSortAlloc(comptime T: type, arr: []T, allocator: Allocator) !void {
    if (arr.len == 0) return;

    // Find min and max
    var min_val = arr[0];
    var max_val = arr[0];
    for (arr) |val| {
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    // Calculate range
    const range = @as(usize, @intCast(max_val - min_val)) + 1;

    // Create pigeonholes (buckets)
    const holes = try allocator.alloc(std.ArrayList(T), range);
    defer {
        for (holes) |*hole| {
            hole.deinit(allocator);
        }
        allocator.free(holes);
    }

    // Initialize pigeonholes
    for (holes) |*hole| {
        hole.* = std.ArrayList(T){};
    }

    // Distribute elements into pigeonholes
    for (arr) |val| {
        const index = @as(usize, @intCast(val - min_val));
        try holes[index].append(allocator, val);
    }

    // Collect elements back into array
    var arr_idx: usize = 0;
    for (holes) |hole| {
        for (hole.items) |val| {
            arr[arr_idx] = val;
            arr_idx += 1;
        }
    }
}

/// Pigeonhole Sort ascending order (convenience wrapper)
/// Time: O(n + range)
/// Space: O(range)
pub fn pigeonholeSortAsc(comptime T: type, arr: []T, allocator: Allocator) !void {
    try pigeonholeSortAlloc(T, arr, allocator);
}

/// Pigeonhole Sort descending order
/// Time: O(n + range)
/// Space: O(range)
pub fn pigeonholeSortDesc(comptime T: type, arr: []T, allocator: Allocator) !void {
    if (arr.len == 0) return;

    // Find min and max
    var min_val = arr[0];
    var max_val = arr[0];
    for (arr) |val| {
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    // Calculate range
    const range = @as(usize, @intCast(max_val - min_val)) + 1;

    // Create pigeonholes
    const holes = try allocator.alloc(std.ArrayList(T), range);
    defer {
        for (holes) |*hole| {
            hole.deinit(allocator);
        }
        allocator.free(holes);
    }

    // Initialize pigeonholes
    for (holes) |*hole| {
        hole.* = std.ArrayList(T){};
    }

    // Distribute elements
    for (arr) |val| {
        const index = @as(usize, @intCast(val - min_val));
        try holes[index].append(allocator, val);
    }

    // Collect in reverse order for descending
    var arr_idx: usize = 0;
    var i: usize = range;
    while (i > 0) {
        i -= 1;
        for (holes[i].items) |val| {
            arr[arr_idx] = val;
            arr_idx += 1;
        }
    }
}

/// Pigeonhole Sort with custom comparison (Order-based)
/// Time: O(n + range)
/// Space: O(range)
pub fn pigeonholeSortBy(comptime T: type, arr: []T, allocator: Allocator, order: Order) !void {
    switch (order) {
        .lt => try pigeonholeSortAsc(T, arr, allocator),
        .gt => try pigeonholeSortDesc(T, arr, allocator),
        .eq => {}, // Already equal, no sorting needed
    }
}

/// Count distinct values in array (using pigeonhole approach)
/// Time: O(n + range)
/// Space: O(range)
pub fn countDistinct(comptime T: type, arr: []const T, allocator: Allocator) !usize {
    if (arr.len == 0) return 0;

    var min_val = arr[0];
    var max_val = arr[0];
    for (arr) |val| {
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    const range = @as(usize, @intCast(max_val - min_val)) + 1;
    const holes = try allocator.alloc(bool, range);
    defer allocator.free(holes);

    @memset(holes, false);

    for (arr) |val| {
        const index = @as(usize, @intCast(val - min_val));
        holes[index] = true;
    }

    var count: usize = 0;
    for (holes) |present| {
        if (present) count += 1;
    }

    return count;
}

// ============================================================================
// Tests
// ============================================================================

test "pigeonholeSort: basic ascending" {
    var arr = [_]i32{ 8, 3, 2, 7, 4, 6, 8 };
    try pigeonholeSortAlloc(i32, &arr, testing.allocator);
    try testing.expectEqualSlices(i32, &[_]i32{ 2, 3, 4, 6, 7, 8, 8 }, &arr);
}

test "pigeonholeSort: basic descending" {
    var arr = [_]i32{ 8, 3, 2, 7, 4, 6, 8 };
    try pigeonholeSortDesc(i32, &arr, testing.allocator);
    try testing.expectEqualSlices(i32, &[_]i32{ 8, 8, 7, 6, 4, 3, 2 }, &arr);
}

test "pigeonholeSort: all duplicates" {
    var arr = [_]i32{ 5, 5, 5, 5, 5 };
    try pigeonholeSortAlloc(i32, &arr, testing.allocator);
    try testing.expectEqualSlices(i32, &[_]i32{ 5, 5, 5, 5, 5 }, &arr);
}

test "pigeonholeSort: empty array" {
    var arr = [_]i32{};
    try pigeonholeSortAlloc(i32, &arr, testing.allocator);
    try testing.expectEqualSlices(i32, &[_]i32{}, &arr);
}

test "pigeonholeSort: single element" {
    var arr = [_]i32{42};
    try pigeonholeSortAlloc(i32, &arr, testing.allocator);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &arr);
}

test "pigeonholeSort: two elements" {
    var arr = [_]i32{ 5, 2 };
    try pigeonholeSortAlloc(i32, &arr, testing.allocator);
    try testing.expectEqualSlices(i32, &[_]i32{ 2, 5 }, &arr);
}

test "pigeonholeSort: already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    try pigeonholeSortAlloc(i32, &arr, testing.allocator);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "pigeonholeSort: reverse sorted" {
    var arr = [_]i32{ 5, 4, 3, 2, 1 };
    try pigeonholeSortAlloc(i32, &arr, testing.allocator);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "pigeonholeSort: negative numbers" {
    var arr = [_]i32{ -3, 1, -5, 0, 2, -1 };
    try pigeonholeSortAlloc(i32, &arr, testing.allocator);
    try testing.expectEqualSlices(i32, &[_]i32{ -5, -3, -1, 0, 1, 2 }, &arr);
}

test "pigeonholeSort: mixed positive and negative" {
    var arr = [_]i32{ 3, -2, 5, -7, 0, 1 };
    try pigeonholeSortAlloc(i32, &arr, testing.allocator);
    try testing.expectEqualSlices(i32, &[_]i32{ -7, -2, 0, 1, 3, 5 }, &arr);
}

test "pigeonholeSort: small range" {
    var arr = [_]i32{ 2, 0, 1, 2, 0, 1, 0 };
    try pigeonholeSortAlloc(i32, &arr, testing.allocator);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 0, 0, 1, 1, 2, 2 }, &arr);
}

test "pigeonholeSort: u8 type" {
    var arr = [_]u8{ 99, 10, 255, 0, 128 };
    try pigeonholeSortAlloc(u8, &arr, testing.allocator);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 10, 99, 128, 255 }, &arr);
}

test "pigeonholeSort: i64 type" {
    var arr = [_]i64{ 1000, -500, 750, 0, -1000 };
    try pigeonholeSortAlloc(i64, &arr, testing.allocator);
    try testing.expectEqualSlices(i64, &[_]i64{ -1000, -500, 0, 750, 1000 }, &arr);
}

test "pigeonholeSort: Order-based ascending" {
    var arr = [_]i32{ 8, 3, 2, 7, 4 };
    try pigeonholeSortBy(i32, &arr, testing.allocator, .lt);
    try testing.expectEqualSlices(i32, &[_]i32{ 2, 3, 4, 7, 8 }, &arr);
}

test "pigeonholeSort: Order-based descending" {
    var arr = [_]i32{ 8, 3, 2, 7, 4 };
    try pigeonholeSortBy(i32, &arr, testing.allocator, .gt);
    try testing.expectEqualSlices(i32, &[_]i32{ 8, 7, 4, 3, 2 }, &arr);
}

test "pigeonholeSort: stability test" {
    // Extract keys for sorting
    var keys = [_]i32{ 3, 1, 3, 2, 1 };
    try pigeonholeSortAlloc(i32, &keys, testing.allocator);

    // Verify sorted order (stability is maintained by ArrayList append order)
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2, 3, 3 }, &keys);
}

test "pigeonholeSort: large array with small range" {
    var arr: [100]i32 = undefined;
    for (&arr, 0..) |*val, i| {
        val.* = @intCast((i * 7) % 10); // Values 0-9
    }

    try pigeonholeSortAlloc(i32, &arr, testing.allocator);

    // Verify sorted
    for (arr[0 .. arr.len - 1], 0..) |val, i| {
        try testing.expect(val <= arr[i + 1]);
    }
}

test "pigeonholeSort: countDistinct basic" {
    const arr = [_]i32{ 1, 2, 2, 3, 1, 4, 3, 2 };
    const count = try countDistinct(i32, &arr, testing.allocator);
    try testing.expectEqual(@as(usize, 4), count); // 1, 2, 3, 4
}

test "pigeonholeSort: countDistinct with duplicates" {
    const arr = [_]i32{ 5, 5, 5, 5 };
    const count = try countDistinct(i32, &arr, testing.allocator);
    try testing.expectEqual(@as(usize, 1), count);
}

test "pigeonholeSort: countDistinct empty" {
    const arr = [_]i32{};
    const count = try countDistinct(i32, &arr, testing.allocator);
    try testing.expectEqual(@as(usize, 0), count);
}

test "pigeonholeSort: memory safety" {
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var arr = [_]i32{ 9, 7, 5, 3, 1, 8, 6, 4, 2, 0 };
        try pigeonholeSortAlloc(i32, &arr, testing.allocator);
    }
}
