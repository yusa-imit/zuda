//! Gnome Sort - Simple position-based sorting algorithm
//!
//! Also known as "Stupid Sort", Gnome Sort works like a garden gnome sorting
//! flower pots. It moves forward until it finds an out-of-order pair, swaps them,
//! and moves back to check the previous position. Similar to insertion sort but
//! simpler - no nested loop structure.
//!
//! Algorithm:
//! 1. Start at position 0
//! 2. If current element is greater than or equal to previous (or at start), move forward
//! 3. If current element is less than previous, swap and move backward
//! 4. Repeat until reaching the end
//!
//! Time Complexity:
//! - Best case: O(n) when already sorted
//! - Average case: O(n²)
//! - Worst case: O(n²) when reverse sorted
//!
//! Space Complexity: O(1) - in-place sorting
//!
//! Properties:
//! - Stable: Preserves relative order of equal elements
//! - In-place: No extra memory needed
//! - Adaptive: Runs in O(n) on already sorted data
//! - Simple: One of the simplest sorting algorithms to implement
//!
//! Use Cases:
//! - Educational purposes (demonstrates position-based sorting)
//! - Small datasets where simplicity matters
//! - Nearly sorted data (adaptive behavior)
//! - Situations where minimal code size is critical
//!
//! Comparison to similar algorithms:
//! - vs Insertion Sort: Simpler to implement (no nested loops), same complexity
//! - vs Bubble Sort: More efficient (no unnecessary passes)
//! - vs Cocktail Sort: Gnome sort moves backward when needed, not bidirectional passes
//!
//! Reference: Dick Grune (2000) - Originally called "Stupid Sort"

const std = @import("std");
const testing = std.testing;
const Order = std.math.Order;

/// Gnome Sort - Simple position-based sorting with custom comparison
/// Sorts array in-place by moving elements backward when out of order
/// Time: O(n²) average/worst, O(n) best case | Space: O(1)
pub fn gnomeSort(
    comptime T: type,
    items: []T,
    comptime lessThan: fn (lhs: T, rhs: T) bool,
) void {
    if (items.len <= 1) return;
    
    var pos: usize = 0;
    while (pos < items.len) {
        // At start or current >= previous: move forward
        if (pos == 0 or !lessThan(items[pos], items[pos - 1])) {
            pos += 1;
        } else {
            // Current < previous: swap and move backward
            std.mem.swap(T, &items[pos], &items[pos - 1]);
            pos -= 1;
        }
    }
}

/// Gnome Sort in ascending order (convenience wrapper)
/// Time: O(n²) average/worst, O(n) best case | Space: O(1)
pub fn gnomeSortAsc(comptime T: type, items: []T) void {
    const lessThan = struct {
        fn lt(lhs: T, rhs: T) bool {
            return lhs < rhs;
        }
    }.lt;
    gnomeSort(T, items, lessThan);
}

/// Gnome Sort in descending order (convenience wrapper)
/// Time: O(n²) average/worst, O(n) best case | Space: O(1)
pub fn gnomeSortDesc(comptime T: type, items: []T) void {
    const lessThan = struct {
        fn lt(lhs: T, rhs: T) bool {
            return lhs > rhs;
        }
    }.lt;
    gnomeSort(T, items, lessThan);
}

/// Gnome Sort with Order-based comparison (std.math.Order compatible)
/// Time: O(n²) average/worst, O(n) best case | Space: O(1)
pub fn gnomeSortBy(
    comptime T: type,
    items: []T,
    comptime compare: fn (lhs: T, rhs: T) Order,
) void {
    const lessThan = struct {
        fn lt(lhs: T, rhs: T) bool {
            return compare(lhs, rhs) == .lt;
        }
    }.lt;
    gnomeSort(T, items, lessThan);
}

/// Gnome Sort with optimization - remembers last sorted position
/// When moving backward, we can jump back to where we were instead of
/// incrementing by 1 each time. This is the "optimized" or "fast" Gnome Sort.
/// Time: O(n²) average/worst, O(n) best case | Space: O(1)
pub fn gnomeSortOptimized(
    comptime T: type,
    items: []T,
    comptime lessThan: fn (lhs: T, rhs: T) bool,
) void {
    if (items.len <= 1) return;
    
    var pos: usize = 0;
    while (pos < items.len) {
        if (pos == 0 or !lessThan(items[pos], items[pos - 1])) {
            pos += 1;
        } else {
            // Swap and move backward
            std.mem.swap(T, &items[pos], &items[pos - 1]);
            pos -= 1;
            
            // After swap, continue moving backward until in order
            // This is the optimization - we don't jump back to previous position
            while (pos > 0 and lessThan(items[pos], items[pos - 1])) {
                std.mem.swap(T, &items[pos], &items[pos - 1]);
                pos -= 1;
            }
            pos += 1; // Move forward again
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "gnome sort - basic ascending" {
    var arr = [_]i32{ 5, 2, 8, 1, 9, 3 };
    gnomeSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 5, 8, 9 }, &arr);
}

test "gnome sort - basic descending" {
    var arr = [_]i32{ 5, 2, 8, 1, 9, 3 };
    gnomeSortDesc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 9, 8, 5, 3, 2, 1 }, &arr);
}

test "gnome sort - empty array" {
    var arr = [_]i32{};
    gnomeSortAsc(i32, &arr);
    try testing.expectEqual(@as(usize, 0), arr.len);
}

test "gnome sort - single element" {
    var arr = [_]i32{42};
    gnomeSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &arr);
}

test "gnome sort - two elements sorted" {
    var arr = [_]i32{ 1, 2 };
    gnomeSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2 }, &arr);
}

test "gnome sort - two elements unsorted" {
    var arr = [_]i32{ 2, 1 };
    gnomeSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2 }, &arr);
}

test "gnome sort - already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    gnomeSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "gnome sort - reverse sorted" {
    var arr = [_]i32{ 5, 4, 3, 2, 1 };
    gnomeSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "gnome sort - all equal" {
    var arr = [_]i32{ 7, 7, 7, 7 };
    gnomeSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 7, 7, 7, 7 }, &arr);
}

test "gnome sort - duplicates" {
    var arr = [_]i32{ 5, 2, 5, 1, 2, 1 };
    gnomeSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2, 2, 5, 5 }, &arr);
}

test "gnome sort - negative numbers" {
    var arr = [_]i32{ -5, 2, -8, 1, -3 };
    gnomeSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ -8, -5, -3, 1, 2 }, &arr);
}

test "gnome sort - floating point" {
    var arr = [_]f64{ 3.14, 1.41, 2.71, 0.57 };
    gnomeSortAsc(f64, &arr);
    try testing.expectEqualSlices(f64, &[_]f64{ 0.57, 1.41, 2.71, 3.14 }, &arr);
}

test "gnome sort - custom comparison" {
    const Point = struct {
        x: i32,
        y: i32,
    };
    
    var points = [_]Point{
        .{ .x = 5, .y = 2 },
        .{ .x = 1, .y = 8 },
        .{ .x = 3, .y = 4 },
    };
    
    const lessThan = struct {
        fn lt(a: Point, b: Point) bool {
            return a.x < b.x;
        }
    }.lt;
    
    gnomeSort(Point, &points, lessThan);
    
    try testing.expectEqual(@as(i32, 1), points[0].x);
    try testing.expectEqual(@as(i32, 3), points[1].x);
    try testing.expectEqual(@as(i32, 5), points[2].x);
}

test "gnome sort - Order-based comparison" {
    var arr = [_]i32{ 5, 2, 8, 1, 9, 3 };

    const compare = struct {
        fn cmp(a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    }.cmp;

    gnomeSortBy(i32, &arr, compare);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 5, 8, 9 }, &arr);
}

test "gnome sort - u8 type" {
    var arr = [_]u8{ 200, 50, 150, 100, 255 };
    gnomeSortAsc(u8, &arr);
    try testing.expectEqualSlices(u8, &[_]u8{ 50, 100, 150, 200, 255 }, &arr);
}

test "gnome sort - optimized variant" {
    var arr = [_]i32{ 5, 2, 8, 1, 9, 3 };
    
    const lessThan = struct {
        fn lt(a: i32, b: i32) bool {
            return a < b;
        }
    }.lt;
    
    gnomeSortOptimized(i32, &arr, lessThan);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 5, 8, 9 }, &arr);
}

test "gnome sort - optimized vs standard consistency" {
    var arr1 = [_]i32{ 9, 3, 7, 1, 5, 2, 8, 4, 6 };
    var arr2 = arr1;
    
    const lessThan = struct {
        fn lt(a: i32, b: i32) bool {
            return a < b;
        }
    }.lt;
    
    gnomeSort(i32, &arr1, lessThan);
    gnomeSortOptimized(i32, &arr2, lessThan);
    
    try testing.expectEqualSlices(i32, &arr1, &arr2);
}

test "gnome sort - large array" {
    const allocator = testing.allocator;
    const n = 100;

    const arr = try allocator.alloc(i32, n);
    defer allocator.free(arr);

    // Fill with reverse order
    for (arr, 0..) |*item, i| {
        item.* = @intCast(n - i);
    }

    gnomeSortAsc(i32, arr);

    // Verify sorted
    for (arr, 0..) |item, i| {
        try testing.expectEqual(@as(i32, @intCast(i + 1)), item);
    }
}

test "gnome sort - stability test" {
    const Item = struct {
        key: i32,
        value: u8,
    };
    
    var items = [_]Item{
        .{ .key = 3, .value = 1 },
        .{ .key = 1, .value = 2 },
        .{ .key = 3, .value = 3 },
        .{ .key = 2, .value = 4 },
        .{ .key = 3, .value = 5 },
    };
    
    const lessThan = struct {
        fn lt(a: Item, b: Item) bool {
            return a.key < b.key;
        }
    }.lt;
    
    gnomeSort(Item, &items, lessThan);
    
    // Verify sorted by key
    try testing.expectEqual(@as(i32, 1), items[0].key);
    try testing.expectEqual(@as(i32, 2), items[1].key);
    try testing.expectEqual(@as(i32, 3), items[2].key);
    try testing.expectEqual(@as(i32, 3), items[3].key);
    try testing.expectEqual(@as(i32, 3), items[4].key);
    
    // Verify stability - items with key=3 should maintain order (1, 3, 5)
    try testing.expectEqual(@as(u8, 1), items[2].value);
    try testing.expectEqual(@as(u8, 3), items[3].value);
    try testing.expectEqual(@as(u8, 5), items[4].value);
}

test "gnome sort - stress test" {
    const allocator = testing.allocator;

    // Pseudo-random sequence
    const arr = try allocator.alloc(i32, 50);
    defer allocator.free(arr);

    var seed: u32 = 12345;
    for (arr) |*item| {
        seed = seed *% 1103515245 +% 12345;
        item.* = @as(i32, @intCast(seed % 1000));
    }

    gnomeSortAsc(i32, arr);

    // Verify sorted
    for (arr[0 .. arr.len - 1], 1..) |item, i| {
        try testing.expect(item <= arr[i]);
    }
}
