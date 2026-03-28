//! Binary Search Algorithms
//!
//! This module provides variants of binary search for sorted slices.
//! All algorithms assume the input slice is sorted according to the provided comparator.

const std = @import("std");
const testing = std.testing;

/// Standard binary search - finds an element equal to the target.
/// Returns the index of any matching element, or null if not found.
///
/// Time: O(log n) | Space: O(1)
pub fn binarySearch(
    comptime T: type,
    slice: []const T,
    target: T,
    comptime compareFn: fn (T, T) std.math.Order,
) ?usize {
    if (slice.len == 0) return null;

    var left: usize = 0;
    var right: usize = slice.len;

    while (left < right) {
        const mid = left + (right - left) / 2;
        const cmp = compareFn(slice[mid], target);

        switch (cmp) {
            .eq => return mid,
            .lt => left = mid + 1,
            .gt => right = mid,
        }
    }

    return null;
}

/// Lower bound - finds the first element >= target.
/// Returns the index of the first element >= target, or slice.len if all elements < target.
///
/// Time: O(log n) | Space: O(1)
pub fn lowerBound(
    comptime T: type,
    slice: []const T,
    target: T,
    comptime compareFn: fn (T, T) std.math.Order,
) usize {
    var left: usize = 0;
    var right: usize = slice.len;

    while (left < right) {
        const mid = left + (right - left) / 2;
        const cmp = compareFn(slice[mid], target);

        switch (cmp) {
            .lt => left = mid + 1,
            .eq, .gt => right = mid,
        }
    }

    return left;
}

/// Upper bound - finds the first element > target.
/// Returns the index of the first element > target, or slice.len if all elements <= target.
///
/// Time: O(log n) | Space: O(1)
pub fn upperBound(
    comptime T: type,
    slice: []const T,
    target: T,
    comptime compareFn: fn (T, T) std.math.Order,
) usize {
    var left: usize = 0;
    var right: usize = slice.len;

    while (left < right) {
        const mid = left + (right - left) / 2;
        const cmp = compareFn(slice[mid], target);

        switch (cmp) {
            .lt, .eq => left = mid + 1,
            .gt => right = mid,
        }
    }

    return left;
}

/// Equal range - finds the range [first, last) of elements equal to target.
/// Returns a struct with `start` (lower bound) and `end` (upper bound) indices.
///
/// Time: O(log n) | Space: O(1)
pub fn equalRange(
    comptime T: type,
    slice: []const T,
    target: T,
    comptime compareFn: fn (T, T) std.math.Order,
) struct { start: usize, end: usize } {
    return .{
        .start = lowerBound(T, slice, target, compareFn),
        .end = upperBound(T, slice, target, compareFn),
    };
}

/// Exponential search - efficient for unbounded or very large arrays where target is near start.
/// Returns the index of the target, or null if not found.
///
/// Time: O(log n) | Space: O(1)
pub fn exponentialSearch(
    comptime T: type,
    slice: []const T,
    target: T,
    comptime compareFn: fn (T, T) std.math.Order,
) ?usize {
    if (slice.len == 0) return null;

    // Check first element
    if (compareFn(slice[0], target) == .eq) return 0;
    if (compareFn(slice[0], target) == .gt) return null;

    // Find range where target may exist
    var bound: usize = 1;
    while (bound < slice.len and compareFn(slice[bound], target) == .lt) {
        bound *= 2;
    }

    // Binary search in the found range
    const right = @min(bound + 1, slice.len);
    const left = bound / 2;

    var l = left;
    var r = right;

    while (l < r) {
        const mid = l + (r - l) / 2;
        const cmp = compareFn(slice[mid], target);

        switch (cmp) {
            .eq => return mid,
            .lt => l = mid + 1,
            .gt => r = mid,
        }
    }

    return null;
}

/// Default comparator for ordered types
fn defaultCompare(comptime T: type) fn (T, T) std.math.Order {
    return struct {
        fn compare(a: T, b: T) std.math.Order {
            return std.math.order(a, b);
        }
    }.compare;
}

// ============================================================================
// Tests
// ============================================================================

test "binarySearch - basic" {
    const arr = [_]i32{ 1, 3, 5, 7, 9, 11 };
    const cmp = defaultCompare(i32);

    try testing.expectEqual(@as(?usize, 0), binarySearch(i32, &arr, 1, cmp));
    try testing.expectEqual(@as(?usize, 2), binarySearch(i32, &arr, 5, cmp));
    try testing.expectEqual(@as(?usize, 5), binarySearch(i32, &arr, 11, cmp));
    try testing.expectEqual(@as(?usize, null), binarySearch(i32, &arr, 4, cmp));
    try testing.expectEqual(@as(?usize, null), binarySearch(i32, &arr, 12, cmp));
}

test "binarySearch - empty array" {
    const arr = [_]i32{};
    const cmp = defaultCompare(i32);
    try testing.expectEqual(@as(?usize, null), binarySearch(i32, &arr, 1, cmp));
}

test "binarySearch - single element" {
    const arr = [_]i32{42};
    const cmp = defaultCompare(i32);
    try testing.expectEqual(@as(?usize, 0), binarySearch(i32, &arr, 42, cmp));
    try testing.expectEqual(@as(?usize, null), binarySearch(i32, &arr, 1, cmp));
}

test "binarySearch - duplicates" {
    const arr = [_]i32{ 1, 3, 3, 3, 5 };
    const cmp = defaultCompare(i32);
    const result = binarySearch(i32, &arr, 3, cmp);
    try testing.expect(result != null);
    try testing.expectEqual(@as(i32, 3), arr[result.?]);
}

test "lowerBound - basic" {
    const arr = [_]i32{ 1, 3, 5, 7, 9 };
    const cmp = defaultCompare(i32);

    try testing.expectEqual(@as(usize, 0), lowerBound(i32, &arr, 1, cmp));
    try testing.expectEqual(@as(usize, 2), lowerBound(i32, &arr, 5, cmp));
    try testing.expectEqual(@as(usize, 2), lowerBound(i32, &arr, 4, cmp)); // First >= 4 is 5 at index 2
    try testing.expectEqual(@as(usize, 5), lowerBound(i32, &arr, 10, cmp)); // All < 10
}

test "lowerBound - duplicates" {
    const arr = [_]i32{ 1, 3, 3, 3, 5, 7 };
    const cmp = defaultCompare(i32);
    try testing.expectEqual(@as(usize, 1), lowerBound(i32, &arr, 3, cmp)); // First 3 at index 1
}

test "upperBound - basic" {
    const arr = [_]i32{ 1, 3, 5, 7, 9 };
    const cmp = defaultCompare(i32);

    try testing.expectEqual(@as(usize, 1), upperBound(i32, &arr, 1, cmp)); // First > 1 is 3 at index 1
    try testing.expectEqual(@as(usize, 3), upperBound(i32, &arr, 5, cmp)); // First > 5 is 7 at index 3
    try testing.expectEqual(@as(usize, 5), upperBound(i32, &arr, 9, cmp)); // All <= 9
}

test "upperBound - duplicates" {
    const arr = [_]i32{ 1, 3, 3, 3, 5, 7 };
    const cmp = defaultCompare(i32);
    try testing.expectEqual(@as(usize, 4), upperBound(i32, &arr, 3, cmp)); // First > 3 is 5 at index 4
}

test "equalRange - basic" {
    const arr = [_]i32{ 1, 3, 5, 7, 9 };
    const cmp = defaultCompare(i32);

    const range = equalRange(i32, &arr, 5, cmp);
    try testing.expectEqual(@as(usize, 2), range.start);
    try testing.expectEqual(@as(usize, 3), range.end);
}

test "equalRange - duplicates" {
    const arr = [_]i32{ 1, 3, 3, 3, 5, 7 };
    const cmp = defaultCompare(i32);

    const range = equalRange(i32, &arr, 3, cmp);
    try testing.expectEqual(@as(usize, 1), range.start);
    try testing.expectEqual(@as(usize, 4), range.end);
}

test "equalRange - not found" {
    const arr = [_]i32{ 1, 3, 5, 7, 9 };
    const cmp = defaultCompare(i32);

    const range = equalRange(i32, &arr, 4, cmp);
    try testing.expectEqual(range.start, range.end); // Empty range
}

test "exponentialSearch - basic" {
    const arr = [_]i32{ 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25 };
    const cmp = defaultCompare(i32);

    try testing.expectEqual(@as(?usize, 0), exponentialSearch(i32, &arr, 1, cmp));
    try testing.expectEqual(@as(?usize, 6), exponentialSearch(i32, &arr, 13, cmp));
    try testing.expectEqual(@as(?usize, null), exponentialSearch(i32, &arr, 4, cmp));
}

test "exponentialSearch - empty array" {
    const arr = [_]i32{};
    const cmp = defaultCompare(i32);
    try testing.expectEqual(@as(?usize, null), exponentialSearch(i32, &arr, 1, cmp));
}

test "exponentialSearch - near start" {
    const arr = [_]i32{ 1, 2, 3, 100, 200, 300, 400, 500 };
    const cmp = defaultCompare(i32);
    try testing.expectEqual(@as(?usize, 1), exponentialSearch(i32, &arr, 2, cmp));
}

test "binarySearch - large array stress test" {
    const allocator = testing.allocator;
    const n = 10000;

    const arr = try allocator.alloc(i32, n);
    defer allocator.free(arr);

    // Fill with even numbers: 0, 2, 4, 6, ...
    for (arr, 0..) |*val, i| {
        val.* = @as(i32, @intCast(i * 2));
    }

    const cmp = defaultCompare(i32);

    // Search for existing even numbers
    try testing.expectEqual(@as(?usize, 0), binarySearch(i32, arr, 0, cmp));
    try testing.expectEqual(@as(?usize, 5000), binarySearch(i32, arr, 10000, cmp));
    try testing.expectEqual(@as(?usize, 9999), binarySearch(i32, arr, 19998, cmp));

    // Search for non-existing odd numbers
    try testing.expectEqual(@as(?usize, null), binarySearch(i32, arr, 1, cmp));
    try testing.expectEqual(@as(?usize, null), binarySearch(i32, arr, 9999, cmp));
}

test "lowerBound/upperBound - boundary conditions" {
    const arr = [_]i32{ 1, 1, 1, 1, 1 };
    const cmp = defaultCompare(i32);

    // All elements equal to target
    try testing.expectEqual(@as(usize, 0), lowerBound(i32, &arr, 1, cmp));
    try testing.expectEqual(@as(usize, 5), upperBound(i32, &arr, 1, cmp));

    // All elements less than target
    try testing.expectEqual(@as(usize, 5), lowerBound(i32, &arr, 2, cmp));
    try testing.expectEqual(@as(usize, 5), upperBound(i32, &arr, 2, cmp));

    // All elements greater than target
    try testing.expectEqual(@as(usize, 0), lowerBound(i32, &arr, 0, cmp));
    try testing.expectEqual(@as(usize, 0), upperBound(i32, &arr, 0, cmp));
}

test "custom comparator - descending order" {
    const arr = [_]i32{ 9, 7, 5, 3, 1 };

    const descendingCmp = struct {
        fn cmp(a: i32, b: i32) std.math.Order {
            return std.math.order(b, a); // Reversed
        }
    }.cmp;

    try testing.expectEqual(@as(?usize, 0), binarySearch(i32, &arr, 9, descendingCmp));
    try testing.expectEqual(@as(?usize, 2), binarySearch(i32, &arr, 5, descendingCmp));
    try testing.expectEqual(@as(?usize, 4), binarySearch(i32, &arr, 1, descendingCmp));
    try testing.expectEqual(@as(?usize, null), binarySearch(i32, &arr, 4, descendingCmp));
}

test "binarySearch - floats" {
    const arr = [_]f64{ 1.1, 2.2, 3.3, 4.4, 5.5 };
    const cmp = defaultCompare(f64);

    try testing.expectEqual(@as(?usize, 2), binarySearch(f64, &arr, 3.3, cmp));
    try testing.expectEqual(@as(?usize, null), binarySearch(f64, &arr, 3.0, cmp));
}
