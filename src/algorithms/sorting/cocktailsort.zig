const std = @import("std");
const testing = std.testing;
const Order = std.math.Order;
const mem = std.mem;

/// Cocktail Sort (bidirectional bubble sort, shaker sort)
///
/// A variation of bubble sort that sorts in both directions on each pass through the list.
/// Alternates between bubbling the largest element to the end and the smallest element to the beginning.
///
/// Algorithm:
/// - Pass 1 (forward): Bubble largest element to the right end
/// - Pass 2 (backward): Bubble smallest element to the left end
/// - Repeat until no swaps occur
/// - Terminates early if already sorted
///
/// Time Complexity:
/// - Best case: O(n) when array is already sorted
/// - Average case: O(n²) - same as bubble sort
/// - Worst case: O(n²) when array is reverse sorted
///
/// Space Complexity: O(1) - in-place, no additional memory
///
/// Stability: Stable - preserves relative order of equal elements
///
/// Adaptiveness: Adaptive - terminates early if no swaps occur
///
/// Advantages over standard bubble sort:
/// - Faster on certain types of data (e.g., arrays with "turtles" - small values near end)
/// - Reduces the number of passes by sorting from both ends simultaneously
/// - Slightly better performance in practice (up to 2x on some inputs)
///
/// Use cases:
/// - Educational purposes (demonstrates bidirectional sorting)
/// - Small datasets where simplicity matters
/// - Nearly sorted data (adaptive behavior)
/// - Drop-in replacement for bubble sort with better performance
///
/// History:
/// - Also known as Shaker Sort, Ripple Sort, Shuttle Sort, Happy Hour Sort
/// - Improves upon bubble sort by addressing the "turtle problem"
///
/// References:
/// - Knuth, "The Art of Computer Programming", Vol. 3 (1998)
/// - Bidirectional variant of bubble sort
///
/// Generic cocktail sort with custom comparison function.
/// Time: O(n²) average/worst, O(n) best
/// Space: O(1) - in-place
///
/// Type constraints:
/// - T: Any type
/// - compareFn: fn (context: Context, lhs: T, rhs: T) bool
///   Returns true if lhs < rhs (for ascending order)
pub fn cocktailSort(
    comptime T: type,
    items: []T,
    context: anytype,
    comptime compareFn: fn (ctx: @TypeOf(context), lhs: T, rhs: T) bool,
) void {
    if (items.len <= 1) return;

    var start: usize = 0;
    var end: usize = items.len - 1;
    var swapped = true;

    while (swapped) {
        swapped = false;

        // Forward pass: bubble largest element to the right
        var i = start;
        while (i < end) : (i += 1) {
            if (compareFn(context, items[i + 1], items[i])) {
                mem.swap(T, &items[i], &items[i + 1]);
                swapped = true;
            }
        }

        // If no swaps occurred, array is sorted
        if (!swapped) break;

        // Reduce end boundary (largest element is now in place)
        end -= 1;
        swapped = false;

        // Backward pass: bubble smallest element to the left
        i = end;
        while (i > start) : (i -= 1) {
            if (compareFn(context, items[i], items[i - 1])) {
                mem.swap(T, &items[i - 1], &items[i]);
                swapped = true;
            }
        }

        // Increase start boundary (smallest element is now in place)
        start += 1;
    }
}

/// Cocktail sort in ascending order (convenience wrapper).
/// Time: O(n²) average/worst, O(n) best
/// Space: O(1)
pub fn cocktailSortAsc(comptime T: type, items: []T) void {
    const asc = struct {
        fn lessThan(_: void, lhs: T, rhs: T) bool {
            return lhs < rhs;
        }
    }.lessThan;
    cocktailSort(T, items, {}, asc);
}

/// Cocktail sort in descending order (convenience wrapper).
/// Time: O(n²) average/worst, O(n) best
/// Space: O(1)
pub fn cocktailSortDesc(comptime T: type, items: []T) void {
    const desc = struct {
        fn greaterThan(_: void, lhs: T, rhs: T) bool {
            return lhs > rhs;
        }
    }.greaterThan;
    cocktailSort(T, items, {}, desc);
}

/// Cocktail sort using std.math.Order-based comparison.
/// Time: O(n²) average/worst, O(n) best
/// Space: O(1)
pub fn cocktailSortBy(
    comptime T: type,
    items: []T,
    context: anytype,
    comptime orderFn: fn (ctx: @TypeOf(context), lhs: T, rhs: T) Order,
) void {
    const Wrapper = struct {
        fn lessThan(ctx: @TypeOf(context), lhs: T, rhs: T) bool {
            return orderFn(ctx, lhs, rhs) == .lt;
        }
    };
    cocktailSort(T, items, context, Wrapper.lessThan);
}

// ============================================================================
// Tests
// ============================================================================

test "cocktailSort - basic ascending order" {
    var arr = [_]i32{ 5, 2, 8, 1, 9 };
    cocktailSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 5, 8, 9 }, &arr);
}

test "cocktailSort - basic descending order" {
    var arr = [_]i32{ 5, 2, 8, 1, 9 };
    cocktailSortDesc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 9, 8, 5, 2, 1 }, &arr);
}

test "cocktailSort - empty array" {
    var arr = [_]i32{};
    cocktailSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{}, &arr);
}

test "cocktailSort - single element" {
    var arr = [_]i32{42};
    cocktailSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &arr);
}

test "cocktailSort - two elements" {
    var arr = [_]i32{ 2, 1 };
    cocktailSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2 }, &arr);
}

test "cocktailSort - already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    cocktailSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "cocktailSort - reverse sorted" {
    var arr = [_]i32{ 5, 4, 3, 2, 1 };
    cocktailSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "cocktailSort - all equal elements" {
    var arr = [_]i32{ 7, 7, 7, 7, 7 };
    cocktailSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 7, 7, 7, 7, 7 }, &arr);
}

test "cocktailSort - duplicates" {
    var arr = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6, 5 };
    cocktailSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2, 3, 4, 5, 5, 6, 9 }, &arr);
}

test "cocktailSort - negative numbers" {
    var arr = [_]i32{ -5, 3, -1, 0, 2, -3 };
    cocktailSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ -5, -3, -1, 0, 2, 3 }, &arr);
}

test "cocktailSort - floating point (f64)" {
    var arr = [_]f64{ 3.14, 1.41, 2.71, 0.57, 1.73 };
    cocktailSortAsc(f64, &arr);
    try testing.expectApproxEqAbs(0.57, arr[0], 1e-9);
    try testing.expectApproxEqAbs(1.41, arr[1], 1e-9);
    try testing.expectApproxEqAbs(1.73, arr[2], 1e-9);
    try testing.expectApproxEqAbs(2.71, arr[3], 1e-9);
    try testing.expectApproxEqAbs(3.14, arr[4], 1e-9);
}

test "cocktailSort - custom comparison (struct sorting)" {
    const Point = struct {
        x: i32,
        y: i32,
    };

    var points = [_]Point{
        .{ .x = 3, .y = 1 },
        .{ .x = 1, .y = 5 },
        .{ .x = 2, .y = 3 },
    };

    const cmp = struct {
        fn byX(_: void, a: Point, b: Point) bool {
            return a.x < b.x;
        }
    }.byX;

    cocktailSort(Point, &points, {}, cmp);

    try testing.expectEqual(1, points[0].x);
    try testing.expectEqual(2, points[1].x);
    try testing.expectEqual(3, points[2].x);
}

test "cocktailSort - Order-based comparison" {
    var arr = [_]i32{ 5, 2, 8, 1, 9 };

    const cmp = struct {
        fn order(_: void, a: i32, b: i32) Order {
            if (a < b) return .lt;
            if (a > b) return .gt;
            return .eq;
        }
    }.order;

    cocktailSortBy(i32, &arr, {}, cmp);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 5, 8, 9 }, &arr);
}

test "cocktailSort - large array with allocator" {
    const allocator = testing.allocator;

    const arr = try allocator.alloc(i32, 100);
    defer allocator.free(arr);

    // Fill with reverse sorted data
    for (arr, 0..) |*item, i| {
        item.* = @intCast(100 - i);
    }

    cocktailSortAsc(i32, arr);

    // Verify sorted
    for (arr, 0..) |item, i| {
        try testing.expectEqual(@as(i32, @intCast(i + 1)), item);
    }
}

test "cocktailSort - u8 type" {
    var arr = [_]u8{ 255, 128, 64, 32, 192 };
    cocktailSortAsc(u8, &arr);
    try testing.expectEqualSlices(u8, &[_]u8{ 32, 64, 128, 192, 255 }, &arr);
}

test "cocktailSort - stability test (preserves order of equal elements)" {
    const Item = struct {
        key: i32,
        value: u8,
    };

    var items = [_]Item{
        .{ .key = 3, .value = 1 },
        .{ .key = 1, .value = 2 },
        .{ .key = 3, .value = 3 }, // Same key as first
        .{ .key = 2, .value = 4 },
        .{ .key = 1, .value = 5 }, // Same key as second
    };

    const cmp = struct {
        fn byKey(_: void, a: Item, b: Item) bool {
            return a.key < b.key;
        }
    }.byKey;

    cocktailSort(Item, &items, {}, cmp);

    // Verify sorted by key
    try testing.expectEqual(1, items[0].key);
    try testing.expectEqual(1, items[1].key);
    try testing.expectEqual(2, items[2].key);
    try testing.expectEqual(3, items[3].key);
    try testing.expectEqual(3, items[4].key);

    // Verify stability (original order preserved for equal keys)
    try testing.expectEqual(2, items[0].value); // First 1
    try testing.expectEqual(5, items[1].value); // Second 1
    try testing.expectEqual(4, items[2].value); // Only 2
    try testing.expectEqual(1, items[3].value); // First 3
    try testing.expectEqual(3, items[4].value); // Second 3
}

test "cocktailSort - turtle problem (small value at end)" {
    // Cocktail sort should handle "turtles" (small values near end) better than bubble sort
    var arr = [_]i32{ 2, 3, 4, 5, 6, 7, 8, 1 };
    cocktailSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 }, &arr);
}

test "cocktailSort - rabbit problem (large value at start)" {
    // Cocktail sort should handle "rabbits" (large values near start) efficiently
    var arr = [_]i32{ 9, 1, 2, 3, 4, 5, 6, 7 };
    cocktailSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 9 }, &arr);
}

test "cocktailSort - stress test with pseudo-random data" {
    var arr = [_]i32{
        17, 42, 3,  89, 12, 56, 7,  91, 23, 67, 34, 78, 5,  61, 29, 83, 11,
        72, 38, 94, 2,  58, 26, 85, 14, 69, 31, 76, 8,  64, 19, 88, 46, 99,
        6,  53, 21, 74, 37, 81, 15, 70, 28, 92, 4,  59, 25, 87, 13, 68,
    };
    cocktailSortAsc(i32, &arr);

    // Verify sorted
    for (1..arr.len) |i| {
        try testing.expect(arr[i - 1] <= arr[i]);
    }
}

test "cocktailSort - memory safety (no allocation)" {
    // Verify that cocktail sort doesn't allocate memory
    const allocator = testing.allocator;

    const arr = try allocator.alloc(i32, 10);
    defer allocator.free(arr);

    for (arr, 0..) |*item, i| {
        item.* = @intCast(10 - i);
    }

    // This should work without any additional allocations
    cocktailSortAsc(i32, arr);

    // Verify sorted
    for (arr, 0..) |item, i| {
        try testing.expectEqual(@as(i32, @intCast(i + 1)), item);
    }
}
