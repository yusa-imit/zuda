/// Quick Select algorithm for finding the kth smallest element in unsorted data.
///
/// Quick Select is a selection algorithm to find the kth smallest element in an unordered list.
/// It is related to Quick Sort and uses the same partitioning scheme, but only recurses into
/// one partition.
///
/// ## Algorithm
///
/// 1. Choose a pivot element
/// 2. Partition array around pivot (elements < pivot on left, >= pivot on right)
/// 3. If pivot is at position k, return pivot
/// 4. If k < pivot position, recurse on left partition
/// 5. If k > pivot position, recurse on right partition
///
/// ## Complexity
///
/// - **Time**: O(n) average, O(n²) worst case (rare with good pivot selection)
/// - **Space**: O(log n) average for recursion stack, O(n) worst case
///
/// ## Use Cases
///
/// - Finding median in O(n) average time
/// - Finding kth order statistics
/// - Top-K elements selection
/// - Streaming data analysis
///
const std = @import("std");
const testing = std.testing;

/// Find the kth smallest element in an array (0-indexed).
///
/// Modifies the input array during execution but restores the kth element
/// to its correct position. The array will be partially sorted after this operation.
///
/// Time: O(n) average, O(n²) worst case
/// Space: O(log n) average recursion depth
///
/// ## Example
///
/// ```zig
/// var arr = [_]i32{ 3, 2, 1, 5, 4 };
/// const third = try quickSelect(i32, &arr, 2); // 3 (0-indexed: 0=1, 1=2, 2=3)
/// ```
pub fn quickSelect(comptime T: type, arr: []T, k: usize) error{InvalidIndex}!T {
    if (k >= arr.len) return error.InvalidIndex;
    return quickSelectImpl(T, arr, 0, arr.len - 1, k, std.sort.asc(T));
}

/// Find the kth smallest element using a custom comparison function.
///
/// Time: O(n) average, O(n²) worst case
/// Space: O(log n) average recursion depth
pub fn quickSelectBy(
    comptime T: type,
    arr: []T,
    k: usize,
    comptime lessThan: fn (lhs: T, rhs: T) bool,
) error{InvalidIndex}!T {
    if (k >= arr.len) return error.InvalidIndex;
    return quickSelectImpl(T, arr, 0, arr.len - 1, k, lessThan);
}

fn quickSelectImpl(
    comptime T: type,
    arr: []T,
    left: usize,
    right: usize,
    k: usize,
    comptime lessThan: fn (lhs: T, rhs: T) bool,
) T {
    if (left == right) return arr[left];

    // Choose pivot (median-of-three for better average case)
    const pivot_idx = medianOfThree(T, arr, left, right, lessThan);
    const pivot_idx_new = partition(T, arr, left, right, pivot_idx, lessThan);

    if (k == pivot_idx_new) {
        return arr[k];
    } else if (k < pivot_idx_new) {
        return quickSelectImpl(T, arr, left, pivot_idx_new - 1, k, lessThan);
    } else {
        return quickSelectImpl(T, arr, pivot_idx_new + 1, right, k, lessThan);
    }
}

fn medianOfThree(
    comptime T: type,
    arr: []T,
    left: usize,
    right: usize,
    comptime lessThan: fn (lhs: T, rhs: T) bool,
) usize {
    const mid = left + (right - left) / 2;

    // Sort left, mid, right
    if (lessThan(arr[mid], arr[left])) {
        std.mem.swap(T, &arr[left], &arr[mid]);
    }
    if (lessThan(arr[right], arr[left])) {
        std.mem.swap(T, &arr[left], &arr[right]);
    }
    if (lessThan(arr[right], arr[mid])) {
        std.mem.swap(T, &arr[mid], &arr[right]);
    }

    return mid;
}

fn partition(
    comptime T: type,
    arr: []T,
    left: usize,
    right: usize,
    pivot_idx: usize,
    comptime lessThan: fn (lhs: T, rhs: T) bool,
) usize {
    const pivot_value = arr[pivot_idx];

    // Move pivot to end
    std.mem.swap(T, &arr[pivot_idx], &arr[right]);

    var store_idx = left;
    var i = left;
    while (i < right) : (i += 1) {
        if (lessThan(arr[i], pivot_value)) {
            std.mem.swap(T, &arr[i], &arr[store_idx]);
            store_idx += 1;
        }
    }

    // Move pivot to final position
    std.mem.swap(T, &arr[store_idx], &arr[right]);
    return store_idx;
}

/// Find the median element in an array.
///
/// For even-length arrays, returns the lower median (element at position n/2 - 1).
///
/// Time: O(n) average
/// Space: O(log n) average
pub fn median(comptime T: type, arr: []T) error{EmptyArray}!T {
    if (arr.len == 0) return error.EmptyArray;
    const k = (arr.len - 1) / 2; // Lower median for even-length arrays
    return quickSelect(T, arr, k) catch unreachable; // k is always valid
}

/// Find the top K smallest elements in an array.
///
/// Returns the kth smallest element. After this call, the first k elements
/// of arr will be the k smallest elements (not necessarily sorted).
///
/// Time: O(n) average
/// Space: O(log n) average
pub fn topK(comptime T: type, arr: []T, k: usize) error{InvalidK}!T {
    if (k == 0 or k > arr.len) return error.InvalidK;
    return quickSelect(T, arr, k - 1) catch unreachable; // k-1 is always valid
}

// ============================================================================
// Tests
// ============================================================================

test "quickSelect - basic" {
    var arr = [_]i32{ 3, 2, 1, 5, 4 };

    try testing.expectEqual(@as(i32, 1), try quickSelect(i32, &arr, 0)); // Smallest
    try testing.expectEqual(@as(i32, 5), try quickSelect(i32, &arr, 4)); // Largest

    var arr2 = [_]i32{ 3, 2, 1, 5, 4 };
    try testing.expectEqual(@as(i32, 3), try quickSelect(i32, &arr2, 2)); // Middle
}

test "quickSelect - single element" {
    var arr = [_]i32{42};
    try testing.expectEqual(@as(i32, 42), try quickSelect(i32, &arr, 0));
}

test "quickSelect - two elements" {
    var arr = [_]i32{ 2, 1 };
    try testing.expectEqual(@as(i32, 1), try quickSelect(i32, &arr, 0));
    try testing.expectEqual(@as(i32, 2), try quickSelect(i32, &arr, 1));
}

test "quickSelect - already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    try testing.expectEqual(@as(i32, 3), try quickSelect(i32, &arr, 2));
}

test "quickSelect - reverse sorted" {
    var arr = [_]i32{ 5, 4, 3, 2, 1 };
    try testing.expectEqual(@as(i32, 3), try quickSelect(i32, &arr, 2));
}

test "quickSelect - duplicates" {
    var arr = [_]i32{ 3, 1, 2, 1, 3 };
    try testing.expectEqual(@as(i32, 1), try quickSelect(i32, &arr, 0));
    try testing.expectEqual(@as(i32, 1), try quickSelect(i32, &arr, 1));
    try testing.expectEqual(@as(i32, 2), try quickSelect(i32, &arr, 2));
    try testing.expectEqual(@as(i32, 3), try quickSelect(i32, &arr, 3));
}

test "quickSelect - all same" {
    var arr = [_]i32{ 5, 5, 5, 5, 5 };
    try testing.expectEqual(@as(i32, 5), try quickSelect(i32, &arr, 2));
}

test "quickSelect - large array" {
    const allocator = testing.allocator;
    const n = 1000;
    var arr = try allocator.alloc(i32, n);
    defer allocator.free(arr);

    // Fill with values n-1 down to 0
    for (0..n) |i| {
        arr[i] = @intCast(n - 1 - i);
    }

    try testing.expectEqual(@as(i32, 499), try quickSelect(i32, arr, 499)); // Middle
}

test "quickSelect - invalid index" {
    var arr = [_]i32{ 1, 2, 3 };
    try testing.expectError(error.InvalidIndex, quickSelect(i32, &arr, 3));
    try testing.expectError(error.InvalidIndex, quickSelect(i32, &arr, 100));
}

test "quickSelectBy - custom comparator" {
    const Point = struct {
        x: i32,
        y: i32,

        fn lessThanX(a: @This(), b: @This()) bool {
            return a.x < b.x;
        }
    };

    var points = [_]Point{
        .{ .x = 3, .y = 1 },
        .{ .x = 1, .y = 2 },
        .{ .x = 5, .y = 3 },
        .{ .x = 2, .y = 4 },
    };

    const result = try quickSelectBy(Point, &points, 2, Point.lessThanX);
    try testing.expectEqual(@as(i32, 3), result.x);
}

test "median - odd length" {
    var arr = [_]i32{ 3, 1, 2 };
    try testing.expectEqual(@as(i32, 2), try median(i32, &arr));
}

test "median - even length" {
    var arr = [_]i32{ 4, 1, 3, 2 };
    const result = try median(i32, &arr);
    // Lower median: position (4-1)/2 = 1 → 2nd smallest = 2
    try testing.expectEqual(@as(i32, 2), result);
}

test "median - single element" {
    var arr = [_]i32{42};
    try testing.expectEqual(@as(i32, 42), try median(i32, &arr));
}

test "median - empty array" {
    var arr = [_]i32{};
    try testing.expectError(error.EmptyArray, median(i32, &arr));
}

test "topK - basic" {
    var arr = [_]i32{ 5, 2, 8, 1, 9, 3 };
    const k3 = try topK(i32, &arr, 3); // 3rd smallest
    try testing.expectEqual(@as(i32, 3), k3);

    // First 3 elements should be {1, 2, 3} (not necessarily in order)
    var found = [_]bool{ false, false, false };
    for (arr[0..3]) |val| {
        if (val == 1) found[0] = true;
        if (val == 2) found[1] = true;
        if (val == 3) found[2] = true;
    }
    try testing.expect(found[0] and found[1] and found[2]);
}

test "topK - k=1" {
    var arr = [_]i32{ 5, 2, 8, 1, 9 };
    try testing.expectEqual(@as(i32, 1), try topK(i32, &arr, 1));
}

test "topK - k=n" {
    var arr = [_]i32{ 3, 1, 2 };
    try testing.expectEqual(@as(i32, 3), try topK(i32, &arr, 3)); // Largest
}

test "topK - invalid k" {
    var arr = [_]i32{ 1, 2, 3 };
    try testing.expectError(error.InvalidK, topK(i32, &arr, 0));
    try testing.expectError(error.InvalidK, topK(i32, &arr, 4));
}

test "quickSelect - floats" {
    var arr = [_]f64{ 3.5, 1.2, 4.7, 2.3, 5.1 };
    try testing.expectEqual(@as(f64, 3.5), try quickSelect(f64, &arr, 2));
}

test "quickSelect - strings" {
    var arr = [_][]const u8{ "dog", "cat", "elephant", "ant", "bear" };
    const result = try quickSelectBy([]const u8, &arr, 2, struct {
        fn cmp(a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.cmp);
    try testing.expectEqualStrings("cat", result);
}
