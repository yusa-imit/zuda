const std = @import("std");

/// Binary Search Variants
///
/// Provides various binary search algorithms for different use cases.

/// Standard binary search - returns index if found, null otherwise
/// Time: O(log n) | Space: O(1)
pub fn search(comptime T: type, items: []const T, target: T) ?usize {
    if (items.len == 0) return null;

    var left: usize = 0;
    var right: usize = items.len;

    while (left < right) {
        const mid = left + (right - left) / 2;

        if (items[mid] == target) {
            return mid;
        } else if (items[mid] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return null;
}

/// Lower bound: first position where items[i] >= target
/// Returns items.len if all elements are < target
/// Time: O(log n) | Space: O(1)
pub fn lowerBound(comptime T: type, items: []const T, target: T) usize {
    var left: usize = 0;
    var right: usize = items.len;

    while (left < right) {
        const mid = left + (right - left) / 2;

        if (items[mid] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}

/// Upper bound: first position where items[i] > target
/// Returns items.len if all elements are <= target
/// Time: O(log n) | Space: O(1)
pub fn upperBound(comptime T: type, items: []const T, target: T) usize {
    var left: usize = 0;
    var right: usize = items.len;

    while (left < right) {
        const mid = left + (right - left) / 2;

        if (items[mid] <= target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}

/// Equal range: returns [lower_bound, upper_bound) for target
/// Time: O(log n) | Space: O(1)
pub fn equalRange(comptime T: type, items: []const T, target: T) struct { start: usize, end: usize } {
    return .{
        .start = lowerBound(T, items, target),
        .end = upperBound(T, items, target),
    };
}

/// Binary search with custom comparator
/// Time: O(log n) | Space: O(1)
pub fn searchWithComparator(
    comptime T: type,
    items: []const T,
    target: T,
    comptime lessThan: fn (T, T) bool,
    comptime equal: fn (T, T) bool,
) ?usize {
    if (items.len == 0) return null;

    var left: usize = 0;
    var right: usize = items.len;

    while (left < right) {
        const mid = left + (right - left) / 2;

        if (equal(items[mid], target)) {
            return mid;
        } else if (lessThan(items[mid], target)) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return null;
}

/// Find first occurrence of target
/// Time: O(log n) | Space: O(1)
pub fn findFirst(comptime T: type, items: []const T, target: T) ?usize {
    const lb = lowerBound(T, items, target);
    if (lb < items.len and items[lb] == target) {
        return lb;
    }
    return null;
}

/// Find last occurrence of target
/// Time: O(log n) | Space: O(1)
pub fn findLast(comptime T: type, items: []const T, target: T) ?usize {
    const ub = upperBound(T, items, target);
    if (ub > 0 and items[ub - 1] == target) {
        return ub - 1;
    }
    return null;
}

/// Count occurrences of target
/// Time: O(log n) | Space: O(1)
pub fn count(comptime T: type, items: []const T, target: T) usize {
    const range = equalRange(T, items, target);
    return range.end - range.start;
}

/// Binary search on answer (for optimization problems)
/// Finds the minimum value in [low, high] where predicate is true
/// Time: O(log(high-low) * predicate_time) | Space: O(1)
pub fn searchOnAnswer(
    comptime T: type,
    low: T,
    high: T,
    comptime predicate: fn (T) bool,
) ?T {
    if (!predicate(high)) return null;

    var l = low;
    var h = high;

    while (l < h) {
        const mid = l + (h - l) / 2;

        if (predicate(mid)) {
            h = mid;
        } else {
            l = mid + 1;
        }
    }

    return if (predicate(l)) l else null;
}

/// Floating-point binary search with epsilon precision
/// Time: O(log((high-low)/epsilon)) | Space: O(1)
pub fn searchFloat(
    low: f64,
    high: f64,
    comptime predicate: fn (f64) bool,
    epsilon: f64,
) ?f64 {
    if (!predicate(high)) return null;

    var l = low;
    var h = high;

    while (h - l > epsilon) {
        const mid = (l + h) / 2.0;

        if (predicate(mid)) {
            h = mid;
        } else {
            l = mid;
        }
    }

    return (l + h) / 2.0;
}

/// Rotated sorted array search
/// Time: O(log n) | Space: O(1)
pub fn searchRotated(comptime T: type, items: []const T, target: T) ?usize {
    if (items.len == 0) return null;

    var left: usize = 0;
    var right: usize = items.len;

    while (left < right) {
        const mid = left + (right - left) / 2;

        if (items[mid] == target) {
            return mid;
        }

        // Determine which half is sorted
        if (items[left] <= items[mid]) {
            // Left half is sorted
            if (items[left] <= target and target < items[mid]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        } else {
            // Right half is sorted
            if (items[mid] < target and target <= items[right - 1]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
    }

    return null;
}

/// Find peak element in array (element greater than neighbors)
/// Time: O(log n) | Space: O(1)
pub fn findPeak(comptime T: type, items: []const T) ?usize {
    if (items.len == 0) return null;
    if (items.len == 1) return 0;

    var left: usize = 0;
    var right: usize = items.len - 1;

    while (left < right) {
        const mid = left + (right - left) / 2;

        if (items[mid] < items[mid + 1]) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}

/// Find minimum in rotated sorted array
/// Time: O(log n) | Space: O(1)
pub fn findMinRotated(comptime T: type, items: []const T) ?T {
    if (items.len == 0) return null;
    if (items.len == 1) return items[0];

    var left: usize = 0;
    var right: usize = items.len - 1;

    while (left < right) {
        const mid = left + (right - left) / 2;

        if (items[mid] > items[right]) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return items[left];
}

// ============================================================================
// Tests
// ============================================================================

test "Binary search: empty array" {
    const items: []const i32 = &[_]i32{};
    try std.testing.expectEqual(null, search(i32, items, 5));
}

test "Binary search: single element" {
    const items = [_]i32{42};
    try std.testing.expectEqual(0, search(i32, &items, 42));
    try std.testing.expectEqual(null, search(i32, &items, 41));
}

test "Binary search: found" {
    const items = [_]i32{ 1, 3, 5, 7, 9, 11, 13 };
    try std.testing.expectEqual(0, search(i32, &items, 1));
    try std.testing.expectEqual(3, search(i32, &items, 7));
    try std.testing.expectEqual(6, search(i32, &items, 13));
}

test "Binary search: not found" {
    const items = [_]i32{ 1, 3, 5, 7, 9 };
    try std.testing.expectEqual(null, search(i32, &items, 0));
    try std.testing.expectEqual(null, search(i32, &items, 4));
    try std.testing.expectEqual(null, search(i32, &items, 10));
}

test "Lower bound: basic" {
    const items = [_]i32{ 1, 3, 5, 7, 9 };

    try std.testing.expectEqual(0, lowerBound(i32, &items, 0)); // Before first
    try std.testing.expectEqual(0, lowerBound(i32, &items, 1)); // Equal to first
    try std.testing.expectEqual(1, lowerBound(i32, &items, 2)); // Between 1 and 3
    try std.testing.expectEqual(2, lowerBound(i32, &items, 5)); // Equal to middle
    try std.testing.expectEqual(5, lowerBound(i32, &items, 10)); // After last
}

test "Upper bound: basic" {
    const items = [_]i32{ 1, 3, 5, 7, 9 };

    try std.testing.expectEqual(0, upperBound(i32, &items, 0)); // Before first
    try std.testing.expectEqual(1, upperBound(i32, &items, 1)); // Equal to first
    try std.testing.expectEqual(1, upperBound(i32, &items, 2)); // Between 1 and 3
    try std.testing.expectEqual(3, upperBound(i32, &items, 5)); // Equal to middle
    try std.testing.expectEqual(5, upperBound(i32, &items, 10)); // After last
}

test "Equal range: with duplicates" {
    const items = [_]i32{ 1, 3, 3, 3, 5, 7, 7, 9 };

    const range3 = equalRange(i32, &items, 3);
    try std.testing.expectEqual(1, range3.start);
    try std.testing.expectEqual(4, range3.end);

    const range7 = equalRange(i32, &items, 7);
    try std.testing.expectEqual(5, range7.start);
    try std.testing.expectEqual(7, range7.end);

    const range4 = equalRange(i32, &items, 4);
    try std.testing.expectEqual(range4.start, range4.end); // Not found
}

test "Find first/last: duplicates" {
    const items = [_]i32{ 1, 3, 3, 3, 5, 7, 7, 9 };

    try std.testing.expectEqual(1, findFirst(i32, &items, 3));
    try std.testing.expectEqual(3, findLast(i32, &items, 3));

    try std.testing.expectEqual(5, findFirst(i32, &items, 7));
    try std.testing.expectEqual(6, findLast(i32, &items, 7));

    try std.testing.expectEqual(null, findFirst(i32, &items, 4));
    try std.testing.expectEqual(null, findLast(i32, &items, 4));
}

test "Count occurrences" {
    const items = [_]i32{ 1, 3, 3, 3, 5, 7, 7, 9 };

    try std.testing.expectEqual(3, count(i32, &items, 3));
    try std.testing.expectEqual(2, count(i32, &items, 7));
    try std.testing.expectEqual(1, count(i32, &items, 1));
    try std.testing.expectEqual(0, count(i32, &items, 4));
}

test "Binary search on answer: square root" {
    // Find smallest integer x where x*x >= 50
    const predicate = struct {
        fn check(x: i32) bool {
            return x * x >= 50;
        }
    }.check;

    const result = searchOnAnswer(i32, 0, 100, predicate);
    try std.testing.expectEqual(8, result.?); // 8*8 = 64 >= 50, 7*7 = 49 < 50
}

test "Binary search on answer: not satisfiable" {
    const predicate = struct {
        fn check(_: i32) bool {
            return false;
        }
    }.check;

    const result = searchOnAnswer(i32, 0, 100, predicate);
    try std.testing.expectEqual(null, result);
}

test "Floating-point binary search: square root" {
    const predicate = struct {
        fn check(x: f64) bool {
            return x * x >= 2.0;
        }
    }.check;

    const result = searchFloat(0.0, 10.0, predicate, 0.0001);
    try std.testing.expect(result.? > 1.414);
    try std.testing.expect(result.? < 1.415);
}

test "Rotated array search: basic" {
    const items = [_]i32{ 4, 5, 6, 7, 0, 1, 2 };

    try std.testing.expectEqual(0, searchRotated(i32, &items, 4));
    try std.testing.expectEqual(4, searchRotated(i32, &items, 0));
    try std.testing.expectEqual(6, searchRotated(i32, &items, 2));
    try std.testing.expectEqual(null, searchRotated(i32, &items, 3));
}

test "Rotated array search: not rotated" {
    const items = [_]i32{ 1, 2, 3, 4, 5 };

    try std.testing.expectEqual(2, searchRotated(i32, &items, 3));
}

test "Find peak: basic" {
    const items = [_]i32{ 1, 3, 20, 4, 1, 0 };
    const peak = findPeak(i32, &items);
    try std.testing.expectEqual(2, peak.?); // 20 is peak
}

test "Find peak: increasing" {
    const items = [_]i32{ 1, 2, 3, 4, 5 };
    const peak = findPeak(i32, &items);
    try std.testing.expectEqual(4, peak.?); // Last element
}

test "Find peak: decreasing" {
    const items = [_]i32{ 5, 4, 3, 2, 1 };
    const peak = findPeak(i32, &items);
    try std.testing.expectEqual(0, peak.?); // First element
}

test "Find min rotated: basic" {
    const items = [_]i32{ 4, 5, 6, 7, 0, 1, 2 };
    try std.testing.expectEqual(0, findMinRotated(i32, &items));
}

test "Find min rotated: not rotated" {
    const items = [_]i32{ 1, 2, 3, 4, 5 };
    try std.testing.expectEqual(1, findMinRotated(i32, &items));
}

test "Custom comparator: descending order" {
    const items = [_]i32{ 9, 7, 5, 3, 1 };

    const greaterThan = struct {
        fn gt(a: i32, b: i32) bool {
            return a > b;
        }
    }.gt;

    const equal = struct {
        fn eq(a: i32, b: i32) bool {
            return a == b;
        }
    }.eq;

    try std.testing.expectEqual(2, searchWithComparator(i32, &items, 5, greaterThan, equal));
}

test "Binary search: large array" {
    const allocator = std.testing.allocator;
    const n = 10000;
    const items = try allocator.alloc(i32, n);
    defer allocator.free(items);

    for (0..n) |i| {
        items[i] = @intCast(i * 2);
    }

    try std.testing.expectEqual(500, search(i32, items, 1000));
    try std.testing.expectEqual(null, search(i32, items, 1001));
}

test "Lower/upper bound: all duplicates" {
    const items = [_]i32{ 5, 5, 5, 5, 5 };

    try std.testing.expectEqual(0, lowerBound(i32, &items, 5));
    try std.testing.expectEqual(5, upperBound(i32, &items, 5));
    try std.testing.expectEqual(5, count(i32, &items, 5));
}

test "Binary search on answer: minimize resource allocation" {
    // Allocate minimum resources where total >= 100
    const predicate = struct {
        fn check(x: i32) bool {
            const total = x * 10;
            return total >= 100;
        }
    }.check;

    const result = searchOnAnswer(i32, 0, 100, predicate);
    try std.testing.expectEqual(10, result.?);
}
