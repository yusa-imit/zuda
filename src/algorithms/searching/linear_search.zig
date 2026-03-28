//! Linear Search Algorithms
//!
//! Linear search is the simplest search algorithm that checks every element
//! sequentially until a match is found or the end is reached.
//!
//! Use cases:
//! - Unsorted data (other algorithms require sorted data)
//! - Small datasets where O(n) is acceptable
//! - When data structure doesn't support random access efficiently
//! - First/last occurrence search with custom predicates

const std = @import("std");

/// Performs linear search on a slice.
///
/// Time: O(n) | Space: O(1)
///
/// Returns the index of the first element equal to `target`, or null if not found.
///
/// Works on unsorted data. For sorted data, consider binary search for O(log n) performance.
///
/// Example:
/// ```zig
/// const arr = [_]i32{ 5, 2, 8, 1, 9 };
/// const idx = linearSearch(i32, &arr, 8); // returns 2
/// ```
pub fn linearSearch(comptime T: type, arr: []const T, target: T) ?usize {
    for (arr, 0..) |item, i| {
        if (item == target) return i;
    }
    return null;
}

/// Performs linear search with a custom comparison function.
///
/// Time: O(n) | Space: O(1)
///
/// Returns the index of the first element for which `eqlFn` returns true, or null if not found.
///
/// The comparison function should return true if the element matches the search criteria.
///
/// Example:
/// ```zig
/// const arr = [_]i32{ 5, 2, 8, 1, 9 };
/// const eqlFn = struct {
///     fn f(x: i32, target: i32) bool { return x > target; }
/// }.f;
/// const idx = linearSearchBy(i32, &arr, 6, eqlFn); // returns 0 (first element > 6)
/// ```
pub fn linearSearchBy(comptime T: type, arr: []const T, target: T, eqlFn: fn (T, T) bool) ?usize {
    for (arr, 0..) |item, i| {
        if (eqlFn(item, target)) return i;
    }
    return null;
}

/// Performs linear search with a sentinel value to eliminate boundary checks.
///
/// Time: O(n) | Space: O(1)
///
/// This optimization places the target at the end of the array to eliminate
/// the need for bounds checking in the loop. The original last element is preserved.
///
/// Note: Modifies the input array temporarily (restored before return).
/// Use only when array modification during search is acceptable.
///
/// Example:
/// ```zig
/// var arr = [_]i32{ 5, 2, 8, 1, 9 };
/// const idx = sentinelLinearSearch(i32, &arr, 8); // returns 2
/// ```
pub fn sentinelLinearSearch(comptime T: type, arr: []T, target: T) ?usize {
    if (arr.len == 0) return null;

    // Save the last element
    const last = arr[arr.len - 1];

    // If last element is the target, return immediately
    if (last == target) return arr.len - 1;

    // Place sentinel at the end
    arr[arr.len - 1] = target;

    var i: usize = 0;
    // No boundary check needed - sentinel guarantees we'll find target
    while (arr[i] != target) : (i += 1) {}

    // Restore the last element
    arr[arr.len - 1] = last;

    // If we stopped before the sentinel position, we found the real target
    if (i < arr.len - 1) return i;

    return null;
}

/// Finds the last occurrence of target in the array.
///
/// Time: O(n) | Space: O(1)
///
/// Returns the index of the last element equal to `target`, or null if not found.
///
/// Example:
/// ```zig
/// const arr = [_]i32{ 5, 2, 8, 2, 9 };
/// const idx = linearSearchLast(i32, &arr, 2); // returns 3
/// ```
pub fn linearSearchLast(comptime T: type, arr: []const T, target: T) ?usize {
    var result: ?usize = null;
    for (arr, 0..) |item, i| {
        if (item == target) result = i;
    }
    return result;
}

/// Finds all occurrences of target in the array.
///
/// Time: O(n) | Space: O(k) where k is number of matches
///
/// Returns an ArrayList of indices where target is found.
/// Caller owns the returned ArrayList and must call deinit().
///
/// Example:
/// ```zig
/// const arr = [_]i32{ 5, 2, 8, 2, 9, 2 };
/// var indices = try linearSearchAll(i32, std.testing.allocator, &arr, 2);
/// defer indices.deinit();
/// // indices contains [1, 3, 5]
/// ```
pub fn linearSearchAll(comptime T: type, allocator: std.mem.Allocator, arr: []const T, target: T) !std.ArrayList(usize) {
    var result = std.ArrayList(usize).init(allocator);
    errdefer result.deinit();

    for (arr, 0..) |item, i| {
        if (item == target) {
            try result.append(i);
        }
    }

    return result;
}

/// Counts the number of occurrences of target in the array.
///
/// Time: O(n) | Space: O(1)
///
/// Example:
/// ```zig
/// const arr = [_]i32{ 5, 2, 8, 2, 9, 2 };
/// const count = linearSearchCount(i32, &arr, 2); // returns 3
/// ```
pub fn linearSearchCount(comptime T: type, arr: []const T, target: T) usize {
    var count: usize = 0;
    for (arr) |item| {
        if (item == target) count += 1;
    }
    return count;
}

// Tests
test "linearSearch - basic functionality" {
    const arr = [_]i32{ 5, 2, 8, 1, 9, 3 };

    try std.testing.expectEqual(@as(?usize, 2), linearSearch(i32, &arr, 8));
    try std.testing.expectEqual(@as(?usize, 0), linearSearch(i32, &arr, 5));
    try std.testing.expectEqual(@as(?usize, 5), linearSearch(i32, &arr, 3));
    try std.testing.expectEqual(@as(?usize, null), linearSearch(i32, &arr, 99));
}

test "linearSearch - empty array" {
    const arr = [_]i32{};
    try std.testing.expectEqual(@as(?usize, null), linearSearch(i32, &arr, 5));
}

test "linearSearch - single element" {
    const arr = [_]i32{42};
    try std.testing.expectEqual(@as(?usize, 0), linearSearch(i32, &arr, 42));
    try std.testing.expectEqual(@as(?usize, null), linearSearch(i32, &arr, 99));
}

test "linearSearch - duplicates returns first" {
    const arr = [_]i32{ 5, 2, 8, 2, 9, 2 };
    try std.testing.expectEqual(@as(?usize, 1), linearSearch(i32, &arr, 2));
}

test "linearSearch - floating point" {
    const arr = [_]f64{ 1.5, 2.7, 3.14, 9.99 };
    try std.testing.expectEqual(@as(?usize, 2), linearSearch(f64, &arr, 3.14));
    try std.testing.expectEqual(@as(?usize, null), linearSearch(f64, &arr, 5.5));
}

test "linearSearchBy - custom comparison" {
    const arr = [_]i32{ 5, 2, 8, 1, 9, 3 };

    const greaterThan = struct {
        fn f(x: i32, target: i32) bool {
            return x > target;
        }
    }.f;

    // First element > 6
    try std.testing.expectEqual(@as(?usize, 0), linearSearchBy(i32, &arr, 6, greaterThan));

    // First element > 10 (none)
    try std.testing.expectEqual(@as(?usize, null), linearSearchBy(i32, &arr, 10, greaterThan));

    const lessThan = struct {
        fn f(x: i32, target: i32) bool {
            return x < target;
        }
    }.f;

    // First element < 6
    try std.testing.expectEqual(@as(?usize, 0), linearSearchBy(i32, &arr, 6, lessThan));
}

test "linearSearchBy - even number predicate" {
    const arr = [_]i32{ 5, 7, 8, 1, 9, 4 };

    const isEven = struct {
        fn f(x: i32, _: i32) bool {
            return @mod(x, 2) == 0;
        }
    }.f;

    try std.testing.expectEqual(@as(?usize, 2), linearSearchBy(i32, &arr, 0, isEven));
}

test "sentinelLinearSearch - basic functionality" {
    var arr = [_]i32{ 5, 2, 8, 1, 9, 3 };

    try std.testing.expectEqual(@as(?usize, 2), sentinelLinearSearch(i32, &arr, 8));
    try std.testing.expectEqual(@as(?usize, 0), sentinelLinearSearch(i32, &arr, 5));
    try std.testing.expectEqual(@as(?usize, 5), sentinelLinearSearch(i32, &arr, 3));
    try std.testing.expectEqual(@as(?usize, null), sentinelLinearSearch(i32, &arr, 99));

    // Verify array is unchanged
    try std.testing.expectEqualSlices(i32, &[_]i32{ 5, 2, 8, 1, 9, 3 }, &arr);
}

test "sentinelLinearSearch - empty array" {
    var arr = [_]i32{};
    try std.testing.expectEqual(@as(?usize, null), sentinelLinearSearch(i32, &arr, 5));
}

test "sentinelLinearSearch - single element" {
    var arr = [_]i32{42};
    try std.testing.expectEqual(@as(?usize, 0), sentinelLinearSearch(i32, &arr, 42));
    try std.testing.expectEqual(@as(?usize, null), sentinelLinearSearch(i32, &arr, 99));
}

test "sentinelLinearSearch - target is last element" {
    var arr = [_]i32{ 5, 2, 8, 1, 9 };
    try std.testing.expectEqual(@as(?usize, 4), sentinelLinearSearch(i32, &arr, 9));
}

test "sentinelLinearSearch - duplicates returns first" {
    var arr = [_]i32{ 5, 2, 8, 2, 9, 2 };
    try std.testing.expectEqual(@as(?usize, 1), sentinelLinearSearch(i32, &arr, 2));
}

test "linearSearchLast - finds last occurrence" {
    const arr = [_]i32{ 5, 2, 8, 2, 9, 2 };
    try std.testing.expectEqual(@as(?usize, 5), linearSearchLast(i32, &arr, 2));
    try std.testing.expectEqual(@as(?usize, 0), linearSearchLast(i32, &arr, 5));
    try std.testing.expectEqual(@as(?usize, null), linearSearchLast(i32, &arr, 99));
}

test "linearSearchLast - single element" {
    const arr = [_]i32{42};
    try std.testing.expectEqual(@as(?usize, 0), linearSearchLast(i32, &arr, 42));
}

test "linearSearchAll - finds all occurrences" {
    const arr = [_]i32{ 5, 2, 8, 2, 9, 2 };

    var indices = try linearSearchAll(i32, std.testing.allocator, &arr, 2);
    defer indices.deinit();

    try std.testing.expectEqual(@as(usize, 3), indices.items.len);
    try std.testing.expectEqual(@as(usize, 1), indices.items[0]);
    try std.testing.expectEqual(@as(usize, 3), indices.items[1]);
    try std.testing.expectEqual(@as(usize, 5), indices.items[2]);
}

test "linearSearchAll - no matches" {
    const arr = [_]i32{ 5, 2, 8, 1, 9 };

    var indices = try linearSearchAll(i32, std.testing.allocator, &arr, 99);
    defer indices.deinit();

    try std.testing.expectEqual(@as(usize, 0), indices.items.len);
}

test "linearSearchAll - empty array" {
    const arr = [_]i32{};

    var indices = try linearSearchAll(i32, std.testing.allocator, &arr, 5);
    defer indices.deinit();

    try std.testing.expectEqual(@as(usize, 0), indices.items.len);
}

test "linearSearchAll - all elements match" {
    const arr = [_]i32{ 7, 7, 7, 7 };

    var indices = try linearSearchAll(i32, std.testing.allocator, &arr, 7);
    defer indices.deinit();

    try std.testing.expectEqual(@as(usize, 4), indices.items.len);
    try std.testing.expectEqual(@as(usize, 0), indices.items[0]);
    try std.testing.expectEqual(@as(usize, 1), indices.items[1]);
    try std.testing.expectEqual(@as(usize, 2), indices.items[2]);
    try std.testing.expectEqual(@as(usize, 3), indices.items[3]);
}

test "linearSearchCount - counts occurrences" {
    const arr = [_]i32{ 5, 2, 8, 2, 9, 2 };
    try std.testing.expectEqual(@as(usize, 3), linearSearchCount(i32, &arr, 2));
    try std.testing.expectEqual(@as(usize, 1), linearSearchCount(i32, &arr, 5));
    try std.testing.expectEqual(@as(usize, 0), linearSearchCount(i32, &arr, 99));
}

test "linearSearchCount - empty array" {
    const arr = [_]i32{};
    try std.testing.expectEqual(@as(usize, 0), linearSearchCount(i32, &arr, 5));
}

test "linearSearchCount - all elements match" {
    const arr = [_]i32{ 7, 7, 7, 7 };
    try std.testing.expectEqual(@as(usize, 4), linearSearchCount(i32, &arr, 7));
}

test "linear search - large array performance" {
    const allocator = std.testing.allocator;

    // Create array with target at various positions
    const arr = try allocator.alloc(i32, 10000);
    defer allocator.free(arr);

    for (arr, 0..) |*item, i| {
        item.* = @intCast(i);
    }

    // Test finding element at beginning
    try std.testing.expectEqual(@as(?usize, 0), linearSearch(i32, arr, 0));

    // Test finding element in middle
    try std.testing.expectEqual(@as(?usize, 5000), linearSearch(i32, arr, 5000));

    // Test finding element at end
    try std.testing.expectEqual(@as(?usize, 9999), linearSearch(i32, arr, 9999));

    // Test not finding element
    try std.testing.expectEqual(@as(?usize, null), linearSearch(i32, arr, 10000));
}

test "linear search - string search" {
    const arr = [_][]const u8{ "apple", "banana", "cherry", "date" };

    try std.testing.expectEqual(@as(?usize, 1), linearSearch([]const u8, &arr, "banana"));
    try std.testing.expectEqual(@as(?usize, null), linearSearch([]const u8, &arr, "grape"));
}

test "linear search - memory safety" {
    const allocator = std.testing.allocator;
    const arr = [_]i32{ 1, 2, 3, 2, 5, 2 };

    var indices = try linearSearchAll(i32, allocator, &arr, 2);
    defer indices.deinit();

    // Verify no memory leaks by using testing.allocator
    try std.testing.expectEqual(@as(usize, 3), indices.items.len);
}
