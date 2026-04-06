//! Odd-Even Sort (Brick Sort) — Parallel sorting algorithm variant of bubble sort
//!
//! Time: O(n²) — same as bubble sort but parallelizable
//! Space: O(1) — in-place sorting
//! Stability: Stable — preserves relative order of equal elements
//! Adaptive: Yes — terminates early when sorted
//!
//! Algorithm:
//! Alternates between odd and even phases:
//! - Odd phase: Compare all (odd, odd+1) pairs: (1,2), (3,4), (5,6), ...
//! - Even phase: Compare all (even, even+1) pairs: (0,1), (2,3), (4,5), ...
//! Repeat until no swaps occur (array is sorted)
//!
//! Parallelization:
//! - All comparisons in odd phase can run in parallel
//! - All comparisons in even phase can run in parallel
//! - This implementation is sequential but demonstrates the algorithm
//!
//! Use cases:
//! - Educational purposes (demonstrates parallel sorting concept)
//! - Parallel hardware (CUDA, OpenCL, FPGA) where phases can be parallelized
//! - Small datasets where simplicity matters
//! - Networked sorting (each processor handles adjacent pairs)
//!
//! References:
//! - Habermann (1972) "Parallel Neighbor-Sort"
//! - Knuth "The Art of Computer Programming" Vol. 3

const std = @import("std");
const Order = std.math.Order;
const testing = std.testing;

/// Odd-Even Sort (Brick Sort) — parallel sorting algorithm with odd/even phases
/// Time: O(n²), Space: O(1), Stable: Yes, Adaptive: Yes
///
/// Example:
/// ```zig
/// var arr = [_]i32{ 5, 2, 8, 1, 9 };
/// oddEvenSort(i32, &arr, {}, std.sort.asc(i32));
/// // arr is now [1, 2, 5, 8, 9]
/// ```
pub fn oddEvenSort(
    comptime T: type,
    items: []T,
    context: anytype,
    comptime lessThan: fn (@TypeOf(context), T, T) bool,
) void {
    if (items.len <= 1) return;

    var sorted = false;
    while (!sorted) {
        sorted = true;

        // Odd phase: compare (1,2), (3,4), (5,6), ...
        var i: usize = 1;
        while (i < items.len - 1) : (i += 2) {
            if (lessThan(context, items[i + 1], items[i])) {
                std.mem.swap(T, &items[i], &items[i + 1]);
                sorted = false;
            }
        }

        // Even phase: compare (0,1), (2,3), (4,5), ...
        i = 0;
        while (i < items.len - 1) : (i += 2) {
            if (lessThan(context, items[i + 1], items[i])) {
                std.mem.swap(T, &items[i], &items[i + 1]);
                sorted = false;
            }
        }
    }
}

/// Odd-Even Sort with ascending order
pub fn oddEvenSortAsc(comptime T: type, items: []T) void {
    oddEvenSort(T, items, {}, struct {
        fn less(_: void, a: T, b: T) bool {
            return a < b;
        }
    }.less);
}

/// Odd-Even Sort with descending order
pub fn oddEvenSortDesc(comptime T: type, items: []T) void {
    oddEvenSort(T, items, {}, struct {
        fn less(_: void, a: T, b: T) bool {
            return a > b;
        }
    }.less);
}

/// Odd-Even Sort with Order-based comparison
pub fn oddEvenSortBy(comptime T: type, items: []T, order: Order) void {
    switch (order) {
        .lt => oddEvenSortAsc(T, items),
        .gt => oddEvenSortDesc(T, items),
        .eq => {}, // already equal, no sorting needed
    }
}

// Tests

test "oddEvenSort: basic ascending" {
    var arr = [_]i32{ 5, 2, 8, 1, 9, 3 };
    oddEvenSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 5, 8, 9 }, &arr);
}

test "oddEvenSort: basic descending" {
    var arr = [_]i32{ 5, 2, 8, 1, 9, 3 };
    oddEvenSortDesc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 9, 8, 5, 3, 2, 1 }, &arr);
}

test "oddEvenSort: empty array" {
    var arr = [_]i32{};
    oddEvenSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{}, &arr);
}

test "oddEvenSort: single element" {
    var arr = [_]i32{42};
    oddEvenSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &arr);
}

test "oddEvenSort: two elements sorted" {
    var arr = [_]i32{ 1, 2 };
    oddEvenSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2 }, &arr);
}

test "oddEvenSort: two elements unsorted" {
    var arr = [_]i32{ 2, 1 };
    oddEvenSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2 }, &arr);
}

test "oddEvenSort: already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    oddEvenSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, &arr);
}

test "oddEvenSort: reverse sorted" {
    var arr = [_]i32{ 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    oddEvenSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, &arr);
}

test "oddEvenSort: all equal" {
    var arr = [_]i32{ 5, 5, 5, 5, 5 };
    oddEvenSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 5, 5, 5, 5, 5 }, &arr);
}

test "oddEvenSort: duplicates" {
    var arr = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    oddEvenSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2, 3, 3, 4, 5, 5, 6, 9 }, &arr);
}

test "oddEvenSort: negative numbers" {
    var arr = [_]i32{ -5, 3, -2, 8, -1, 0, 4 };
    oddEvenSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ -5, -2, -1, 0, 3, 4, 8 }, &arr);
}

test "oddEvenSort: floating point f64" {
    var arr = [_]f64{ 3.14, 1.41, 2.71, 0.5, 1.73 };
    oddEvenSortAsc(f64, &arr);
    try testing.expectApproxEqAbs(0.5, arr[0], 1e-9);
    try testing.expectApproxEqAbs(1.41, arr[1], 1e-9);
    try testing.expectApproxEqAbs(1.73, arr[2], 1e-9);
    try testing.expectApproxEqAbs(2.71, arr[3], 1e-9);
    try testing.expectApproxEqAbs(3.14, arr[4], 1e-9);
}

test "oddEvenSort: custom comparison (struct)" {
    const Point = struct {
        x: i32,
        y: i32,
    };

    var points = [_]Point{
        .{ .x = 3, .y = 4 },
        .{ .x = 1, .y = 2 },
        .{ .x = 2, .y = 1 },
    };

    oddEvenSort(Point, &points, {}, struct {
        fn less(_: void, a: Point, b: Point) bool {
            return a.x < b.x;
        }
    }.less);

    try testing.expectEqual(1, points[0].x);
    try testing.expectEqual(2, points[1].x);
    try testing.expectEqual(3, points[2].x);
}

test "oddEvenSort: Order-based comparison" {
    var arr = [_]i32{ 5, 2, 8, 1, 9 };
    oddEvenSortBy(i32, &arr, .lt);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 5, 8, 9 }, &arr);

    oddEvenSortBy(i32, &arr, .gt);
    try testing.expectEqualSlices(i32, &[_]i32{ 9, 8, 5, 2, 1 }, &arr);
}

test "oddEvenSort: u8 type" {
    var arr = [_]u8{ 255, 0, 128, 64, 192 };
    oddEvenSortAsc(u8, &arr);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 64, 128, 192, 255 }, &arr);
}

test "oddEvenSort: odd length array" {
    var arr = [_]i32{ 9, 7, 5, 3, 1 };
    oddEvenSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 3, 5, 7, 9 }, &arr);
}

test "oddEvenSort: even length array" {
    var arr = [_]i32{ 8, 6, 4, 2 };
    oddEvenSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 2, 4, 6, 8 }, &arr);
}

test "oddEvenSort: stability test" {
    const Item = struct {
        key: i32,
        value: u8,
    };

    var items = [_]Item{
        .{ .key = 3, .value = 'a' },
        .{ .key = 1, .value = 'b' },
        .{ .key = 3, .value = 'c' },
        .{ .key = 1, .value = 'd' },
        .{ .key = 2, .value = 'e' },
    };

    oddEvenSort(Item, &items, {}, struct {
        fn less(_: void, a: Item, b: Item) bool {
            return a.key < b.key;
        }
    }.less);

    // Check sorted by key
    try testing.expectEqual(1, items[0].key);
    try testing.expectEqual(1, items[1].key);
    try testing.expectEqual(2, items[2].key);
    try testing.expectEqual(3, items[3].key);
    try testing.expectEqual(3, items[4].key);

    // Check stability: equal keys preserve original order
    try testing.expectEqual('b', items[0].value);
    try testing.expectEqual('d', items[1].value);
    try testing.expectEqual('a', items[3].value);
    try testing.expectEqual('c', items[4].value);
}

test "oddEvenSort: large array" {
    const allocator = testing.allocator;
    const n = 100;
    var arr = try allocator.alloc(i32, n);
    defer allocator.free(arr);

    // Fill with pseudo-random values
    var seed: u32 = 12345;
    for (arr, 0..) |*val, i| {
        seed = seed *% 1103515245 +% 12345;
        val.* = @as(i32, @intCast(i)) - @as(i32, @intCast((seed >> 16) % n));
    }

    oddEvenSortAsc(i32, arr);

    // Verify sorted
    for (arr[0 .. arr.len - 1], 0..) |val, i| {
        try testing.expect(val <= arr[i + 1]);
    }
}

test "oddEvenSort: memory safety (no allocations)" {
    var arr = [_]i32{ 5, 2, 8, 1, 9, 3, 7, 4, 6 };
    oddEvenSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &arr);
}
