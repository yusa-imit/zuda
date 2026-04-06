const std = @import("std");
const testing = std.testing;
const Order = std.math.Order;

/// Insertion Sort - Stable, adaptive sorting algorithm
///
/// Best for:
/// - Small arrays (n < 30-50)
/// - Nearly sorted data (O(n) in best case)
/// - When stability is required
/// - As a subroutine in hybrid algorithms
///
/// Characteristics:
/// - **Stable**: Equal elements maintain relative order
/// - **Adaptive**: Efficient for nearly sorted data
/// - **In-place**: O(1) extra space
/// - **Online**: Can sort stream as it receives data
///
/// Time complexity:
/// - Best: O(n) when array is already sorted
/// - Average: O(n²) comparisons, O(n²) swaps
/// - Worst: O(n²) when array is reverse sorted
///
/// Space complexity: O(1)
///
/// Use cases:
/// - Small datasets where simplicity matters
/// - Nearly sorted data (e.g., maintaining sorted lists)
/// - When stability is critical (e.g., multi-key sorting)
/// - Real-time data streams (online property)
/// - Subroutine in IntroSort, TimSort for small partitions

/// Sorts a slice using insertion sort with default ascending order
///
/// Time: O(n²) average, O(n) best (sorted), O(n²) worst (reverse sorted)
/// Space: O(1)
///
/// Example:
/// ```zig
/// var arr = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6 };
/// insertionSort(i32, &arr, {}, comptime asc(i32));
/// // arr is now [1, 1, 2, 3, 4, 5, 6, 9]
/// ```
pub fn insertionSort(
    comptime T: type,
    items: []T,
    context: anytype,
    comptime lessThan: fn (@TypeOf(context), T, T) bool,
) void {
    if (items.len < 2) return;

    var i: usize = 1;
    while (i < items.len) : (i += 1) {
        const key = items[i];
        var j: usize = i;

        // Shift elements greater than key to the right
        while (j > 0 and lessThan(context, key, items[j - 1])) {
            items[j] = items[j - 1];
            j -= 1;
        }

        items[j] = key;
    }
}

/// Binary insertion sort - uses binary search to find insertion position
///
/// Reduces comparisons from O(n²) to O(n log n), but shifts are still O(n²).
/// Useful when comparisons are expensive.
///
/// Time: O(n log n) comparisons, O(n²) shifts
/// Space: O(1)
///
/// Example:
/// ```zig
/// var arr = [_]i32{ 3, 1, 4, 1, 5 };
/// binaryInsertionSort(i32, &arr, {}, comptime asc(i32));
/// // arr is now [1, 1, 3, 4, 5]
/// ```
pub fn binaryInsertionSort(
    comptime T: type,
    items: []T,
    context: anytype,
    comptime lessThan: fn (@TypeOf(context), T, T) bool,
) void {
    if (items.len < 2) return;

    var i: usize = 1;
    while (i < items.len) : (i += 1) {
        const key = items[i];

        // Binary search for insertion position
        const pos = binarySearchInsertPos(T, items[0..i], key, context, lessThan);

        // Shift elements to make room
        var j: usize = i;
        while (j > pos) : (j -= 1) {
            items[j] = items[j - 1];
        }

        items[pos] = key;
    }
}

/// Helper: binary search to find insertion position in sorted range [0..n)
fn binarySearchInsertPos(
    comptime T: type,
    items: []const T,
    key: T,
    context: anytype,
    comptime lessThan: fn (@TypeOf(context), T, T) bool,
) usize {
    var left: usize = 0;
    var right: usize = items.len;

    while (left < right) {
        const mid = left + (right - left) / 2;
        if (lessThan(context, key, items[mid])) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }

    return left;
}

/// Sorts a slice in-place using insertion sort with custom comparison
///
/// Time: O(n²) average, O(n) best (sorted), O(n²) worst (reverse sorted)
/// Space: O(1)
pub fn sortBy(comptime T: type, items: []T, comptime cmp: fn (T, T) bool) void {
    insertionSort(T, items, {}, struct {
        fn lessThan(_: void, a: T, b: T) bool {
            return cmp(a, b);
        }
    }.lessThan);
}

/// Default ascending comparison for types with '<' operator
pub fn asc(comptime T: type) fn (void, T, T) bool {
    return struct {
        fn lessThan(_: void, a: T, b: T) bool {
            return a < b;
        }
    }.lessThan;
}

/// Default descending comparison for types with '>' operator
pub fn desc(comptime T: type) fn (void, T, T) bool {
    return struct {
        fn lessThan(_: void, a: T, b: T) bool {
            return a > b;
        }
    }.lessThan;
}

// ============================================================================
// Tests
// ============================================================================

test "insertionSort - basic ascending" {
    var arr = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    insertionSort(i32, &arr, {}, comptime asc(i32));
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2, 3, 4, 5, 6, 9 }, &arr);
}

test "insertionSort - already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    insertionSort(i32, &arr, {}, comptime asc(i32));
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "insertionSort - reverse sorted" {
    var arr = [_]i32{ 5, 4, 3, 2, 1 };
    insertionSort(i32, &arr, {}, comptime asc(i32));
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "insertionSort - descending order" {
    var arr = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    insertionSort(i32, &arr, {}, comptime desc(i32));
    try testing.expectEqualSlices(i32, &[_]i32{ 9, 6, 5, 4, 3, 2, 1, 1 }, &arr);
}

test "insertionSort - single element" {
    var arr = [_]i32{42};
    insertionSort(i32, &arr, {}, comptime asc(i32));
    try testing.expectEqual(42, arr[0]);
}

test "insertionSort - empty array" {
    var arr = [_]i32{};
    insertionSort(i32, &arr, {}, comptime asc(i32));
    try testing.expectEqual(0, arr.len);
}

test "insertionSort - two elements" {
    var arr = [_]i32{ 2, 1 };
    insertionSort(i32, &arr, {}, comptime asc(i32));
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2 }, &arr);
}

test "insertionSort - duplicates" {
    var arr = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 };
    insertionSort(i32, &arr, {}, comptime asc(i32));
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9 }, &arr);
}

test "insertionSort - all same" {
    var arr = [_]i32{ 7, 7, 7, 7, 7 };
    insertionSort(i32, &arr, {}, comptime asc(i32));
    try testing.expectEqualSlices(i32, &[_]i32{ 7, 7, 7, 7, 7 }, &arr);
}

test "insertionSort - negative numbers" {
    var arr = [_]i32{ -3, 1, -4, 1, -5, 9, -2, 6 };
    insertionSort(i32, &arr, {}, comptime asc(i32));
    try testing.expectEqualSlices(i32, &[_]i32{ -5, -4, -3, -2, 1, 1, 6, 9 }, &arr);
}

test "insertionSort - floating point" {
    var arr = [_]f64{ 3.14, 1.41, 2.71, 0.57, 1.61 };
    insertionSort(f64, &arr, {}, comptime asc(f64));
    try testing.expectApproxEqAbs(0.57, arr[0], 0.001);
    try testing.expectApproxEqAbs(1.41, arr[1], 0.001);
    try testing.expectApproxEqAbs(1.61, arr[2], 0.001);
    try testing.expectApproxEqAbs(2.71, arr[3], 0.001);
    try testing.expectApproxEqAbs(3.14, arr[4], 0.001);
}

test "insertionSort - nearly sorted (best case performance)" {
    // Nearly sorted data - only one element out of place
    var arr = [_]i32{ 1, 2, 3, 4, 0, 5, 6, 7, 8, 9 };
    insertionSort(i32, &arr, {}, comptime asc(i32));
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &arr);
}

test "binaryInsertionSort - basic ascending" {
    var arr = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    binaryInsertionSort(i32, &arr, {}, comptime asc(i32));
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2, 3, 4, 5, 6, 9 }, &arr);
}

test "binaryInsertionSort - already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    binaryInsertionSort(i32, &arr, {}, comptime asc(i32));
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "binaryInsertionSort - reverse sorted" {
    var arr = [_]i32{ 5, 4, 3, 2, 1 };
    binaryInsertionSort(i32, &arr, {}, comptime asc(i32));
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "binaryInsertionSort - duplicates" {
    var arr = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 };
    binaryInsertionSort(i32, &arr, {}, comptime asc(i32));
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9 }, &arr);
}

test "binaryInsertionSort - single element" {
    var arr = [_]i32{42};
    binaryInsertionSort(i32, &arr, {}, comptime asc(i32));
    try testing.expectEqual(42, arr[0]);
}

test "binaryInsertionSort - empty array" {
    var arr = [_]i32{};
    binaryInsertionSort(i32, &arr, {}, comptime asc(i32));
    try testing.expectEqual(0, arr.len);
}

test "binaryInsertionSort - floating point" {
    var arr = [_]f64{ 3.14, 1.41, 2.71, 0.57, 1.61 };
    binaryInsertionSort(f64, &arr, {}, comptime asc(f64));
    try testing.expectApproxEqAbs(0.57, arr[0], 0.001);
    try testing.expectApproxEqAbs(1.41, arr[1], 0.001);
    try testing.expectApproxEqAbs(1.61, arr[2], 0.001);
    try testing.expectApproxEqAbs(2.71, arr[3], 0.001);
    try testing.expectApproxEqAbs(3.14, arr[4], 0.001);
}

test "sortBy - custom comparison" {
    const Point = struct {
        x: i32,
        y: i32,
    };

    var points = [_]Point{
        .{ .x = 3, .y = 1 },
        .{ .x = 1, .y = 4 },
        .{ .x = 2, .y = 3 },
    };

    sortBy(Point, &points, struct {
        fn cmp(a: Point, b: Point) bool {
            return a.x < b.x;
        }
    }.cmp);

    try testing.expectEqual(1, points[0].x);
    try testing.expectEqual(2, points[1].x);
    try testing.expectEqual(3, points[2].x);
}

test "insertionSort - stability check" {
    const Item = struct {
        key: i32,
        id: u32,
    };

    var items = [_]Item{
        .{ .key = 3, .id = 0 },
        .{ .key = 1, .id = 1 },
        .{ .key = 3, .id = 2 },
        .{ .key = 1, .id = 3 },
        .{ .key = 2, .id = 4 },
    };

    insertionSort(Item, &items, {}, struct {
        fn lessThan(_: void, a: Item, b: Item) bool {
            return a.key < b.key;
        }
    }.lessThan);

    // Verify sorting by key
    try testing.expectEqual(1, items[0].key);
    try testing.expectEqual(1, items[1].key);
    try testing.expectEqual(2, items[2].key);
    try testing.expectEqual(3, items[3].key);
    try testing.expectEqual(3, items[4].key);

    // Verify stability: items with same key maintain original order
    try testing.expectEqual(1, items[0].id); // First "1"
    try testing.expectEqual(3, items[1].id); // Second "1"
    try testing.expectEqual(0, items[3].id); // First "3"
    try testing.expectEqual(2, items[4].id); // Second "3"
}

test "insertionSort - large array" {
    const n = 100;
    var arr: [n]i32 = undefined;

    // Fill with reverse sorted data
    for (0..n) |i| {
        arr[i] = @intCast(n - i);
    }

    insertionSort(i32, &arr, {}, comptime asc(i32));

    // Verify sorted
    for (0..n) |i| {
        try testing.expectEqual(@as(i32, @intCast(i + 1)), arr[i]);
    }
}

test "binaryInsertionSort - large array" {
    const n = 100;
    var arr: [n]i32 = undefined;

    // Fill with random-ish data
    for (0..n) |i| {
        arr[i] = @intCast((i * 17 + 31) % 100);
    }

    binaryInsertionSort(i32, &arr, {}, comptime asc(i32));

    // Verify sorted
    for (1..n) |i| {
        try testing.expect(arr[i - 1] <= arr[i]);
    }
}
