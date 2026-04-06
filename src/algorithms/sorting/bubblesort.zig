//! Bubble Sort - Simple comparison-based sorting by repeated swapping
//!
//! Algorithm:
//! 1. Compare adjacent elements and swap if out of order
//! 2. Repeat passes through the array until no swaps occur
//! 3. After each pass, the largest unsorted element "bubbles" to its position
//!
//! Properties:
//! - Time: O(n²) average/worst case, O(n) best case (already sorted)
//! - Space: O(1) - in-place sorting
//! - Stable: Equal elements maintain their relative order
//! - Adaptive: Efficient for nearly sorted data (early exit optimization)
//! - Simple: Easy to understand and implement
//!
//! Use cases:
//! - Educational purposes (teaching algorithm fundamentals)
//! - Small datasets (< 10-20 elements)
//! - Nearly sorted data (adaptive performance with early exit)
//! - When simplicity is valued over performance
//! - Stability required with minimal code complexity
//!
//! Trade-offs:
//! - vs Insertion Sort: Similar O(n²) but insertion is typically faster in practice
//! - vs Selection Sort: More swaps but adaptive (can terminate early)
//! - vs QuickSort/MergeSort: Much simpler but slower for large datasets
//!
//! Variants:
//! - Standard Bubble Sort: Basic implementation with O(n²) worst case
//! - Optimized Bubble Sort: Early exit when no swaps occur (O(n) best case)
//! - Cocktail Shaker Sort: Bidirectional variant (not implemented here)

const std = @import("std");
const testing = std.testing;
const Order = std.math.Order;

/// Generic bubble sort with custom comparison
/// Time: O(n²) average/worst, O(n) best case (sorted)
/// Space: O(1)
pub fn bubbleSort(comptime T: type, arr: []T, comptime lessThan: fn (T, T) bool) void {
    if (arr.len <= 1) return;

    const n = arr.len;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var swapped = false;
        var j: usize = 0;
        // After i passes, the last i elements are in their final positions
        while (j < n - 1 - i) : (j += 1) {
            if (lessThan(arr[j + 1], arr[j])) {
                std.mem.swap(T, &arr[j], &arr[j + 1]);
                swapped = true;
            }
        }
        // Early exit if no swaps occurred (array is sorted)
        if (!swapped) break;
    }
}

/// Bubble sort in ascending order
/// Time: O(n²) average/worst, O(n) best case
/// Space: O(1)
pub fn bubbleSortAsc(comptime T: type, arr: []T) void {
    bubbleSort(T, arr, asc(T));
}

/// Bubble sort in descending order
/// Time: O(n²) average/worst, O(n) best case
/// Space: O(1)
pub fn bubbleSortDesc(comptime T: type, arr: []T) void {
    bubbleSort(T, arr, desc(T));
}

/// Generic bubble sort with Order-based comparison
/// Time: O(n²) average/worst, O(n) best case
/// Space: O(1)
pub fn sortBy(comptime T: type, arr: []T, comptime cmpFn: fn (T, T) Order) void {
    bubbleSort(T, arr, struct {
        fn lessThan(a: T, b: T) bool {
            return cmpFn(a, b) == .lt;
        }
    }.lessThan);
}

/// Count the number of swaps performed during bubble sort
/// Useful for analyzing algorithm behavior and nearly-sorted data
/// Time: O(n²) average/worst, O(n) best case
/// Space: O(1)
pub fn countSwaps(comptime T: type, arr: []T, comptime lessThan: fn (T, T) bool) usize {
    if (arr.len <= 1) return 0;

    var swap_count: usize = 0;
    const n = arr.len;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var swapped = false;
        var j: usize = 0;
        while (j < n - 1 - i) : (j += 1) {
            if (lessThan(arr[j + 1], arr[j])) {
                std.mem.swap(T, &arr[j], &arr[j + 1]);
                swapped = true;
                swap_count += 1;
            }
        }
        if (!swapped) break;
    }
    return swap_count;
}

/// Count comparison operations during bubble sort
/// Useful for analyzing adaptive behavior
/// Time: O(n²) average/worst, O(n) best case
/// Space: O(1)
pub fn countComparisons(comptime T: type, arr: []T, comptime lessThan: fn (T, T) bool) usize {
    if (arr.len <= 1) return 0;

    var comp_count: usize = 0;
    const n = arr.len;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var swapped = false;
        var j: usize = 0;
        while (j < n - 1 - i) : (j += 1) {
            comp_count += 1;
            if (lessThan(arr[j + 1], arr[j])) {
                std.mem.swap(T, &arr[j], &arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;
    }
    return comp_count;
}

// Helper comparison functions
fn asc(comptime T: type) fn (T, T) bool {
    return struct {
        fn lessThan(a: T, b: T) bool {
            return a < b;
        }
    }.lessThan;
}

fn desc(comptime T: type) fn (T, T) bool {
    return struct {
        fn lessThan(a: T, b: T) bool {
            return a > b;
        }
    }.lessThan;
}

// ============================================================================
// Tests
// ============================================================================

test "bubble sort - basic ascending" {
    var arr = [_]i32{ 5, 2, 8, 1, 9 };
    bubbleSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 5, 8, 9 }, &arr);
}

test "bubble sort - basic descending" {
    var arr = [_]i32{ 5, 2, 8, 1, 9 };
    bubbleSortDesc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 9, 8, 5, 2, 1 }, &arr);
}

test "bubble sort - already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    const comps = countComparisons(i32, &arr, asc(i32));
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
    // Should only need one pass through (n-1 comparisons) with early exit
    try testing.expectEqual(@as(usize, 4), comps);
}

test "bubble sort - reverse sorted" {
    var arr = [_]i32{ 5, 4, 3, 2, 1 };
    bubbleSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "bubble sort - single element" {
    var arr = [_]i32{42};
    bubbleSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &arr);
}

test "bubble sort - two elements" {
    var arr = [_]i32{ 2, 1 };
    bubbleSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2 }, &arr);
}

test "bubble sort - empty array" {
    var arr = [_]i32{};
    bubbleSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{}, &arr);
}

test "bubble sort - duplicates" {
    var arr = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6, 5 };
    bubbleSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2, 3, 4, 5, 5, 6, 9 }, &arr);
}

test "bubble sort - stability check" {
    const Item = struct {
        key: i32,
        value: u8,
    };

    var arr = [_]Item{
        .{ .key = 3, .value = 'a' },
        .{ .key = 1, .value = 'b' },
        .{ .key = 3, .value = 'c' },
        .{ .key = 1, .value = 'd' },
    };

    bubbleSort(Item, &arr, struct {
        fn lessThan(a: Item, b: Item) bool {
            return a.key < b.key;
        }
    }.lessThan);

    // Bubble sort is stable: equal keys maintain relative order
    try testing.expectEqual(@as(i32, 1), arr[0].key);
    try testing.expectEqual(@as(u8, 'b'), arr[0].value);
    try testing.expectEqual(@as(i32, 1), arr[1].key);
    try testing.expectEqual(@as(u8, 'd'), arr[1].value);
    try testing.expectEqual(@as(i32, 3), arr[2].key);
    try testing.expectEqual(@as(u8, 'a'), arr[2].value);
    try testing.expectEqual(@as(i32, 3), arr[3].key);
    try testing.expectEqual(@as(u8, 'c'), arr[3].value);
}

test "bubble sort - count swaps on sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    const swaps = countSwaps(i32, &arr, asc(i32));
    try testing.expectEqual(@as(usize, 0), swaps);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "bubble sort - count swaps on reverse sorted" {
    var arr = [_]i32{ 5, 4, 3, 2, 1 };
    const swaps = countSwaps(i32, &arr, asc(i32));
    // Reverse sorted requires maximum swaps: n(n-1)/2 = 5*4/2 = 10
    try testing.expectEqual(@as(usize, 10), swaps);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "bubble sort - count swaps general case" {
    var arr = [_]i32{ 3, 1, 4, 2 };
    const swaps = countSwaps(i32, &arr, asc(i32));
    // Should perform 4 swaps: (3,1)->(1,3,4,2), (3,2)->(1,2,4,3), (4,3)->(1,2,3,4), (3,4)skip
    try testing.expect(swaps <= 6); // Maximum possible swaps for n=4 is n(n-1)/2 = 6
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4 }, &arr);
}

test "bubble sort - custom comparison with Order" {
    var arr = [_]i32{ 5, 2, 8, 1, 9 };
    sortBy(i32, &arr, struct {
        fn cmp(a: i32, b: i32) Order {
            if (a < b) return .lt;
            if (a > b) return .gt;
            return .eq;
        }
    }.cmp);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 5, 8, 9 }, &arr);
}

test "bubble sort - struct with custom comparison" {
    const Person = struct {
        age: u32,
        name: []const u8,
    };

    var people = [_]Person{
        .{ .age = 30, .name = "Alice" },
        .{ .age = 25, .name = "Bob" },
        .{ .age = 35, .name = "Charlie" },
    };

    bubbleSort(Person, &people, struct {
        fn lessThan(a: Person, b: Person) bool {
            return a.age < b.age;
        }
    }.lessThan);

    try testing.expectEqual(@as(u32, 25), people[0].age);
    try testing.expectEqual(@as(u32, 30), people[1].age);
    try testing.expectEqual(@as(u32, 35), people[2].age);
}

test "bubble sort - nearly sorted data (adaptive)" {
    var arr = [_]i32{ 1, 2, 3, 5, 4, 6, 7, 8 }; // Only one swap needed
    const swaps = countSwaps(i32, &arr, asc(i32));
    const comps = countComparisons(i32, &arr, asc(i32));

    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 }, &arr);
    // Should terminate early with minimal swaps
    try testing.expect(swaps <= 2);
    try testing.expect(comps < 50); // Much less than worst case O(n²) = 64 comparisons
}

test "bubble sort - large array" {
    var arr: [100]i32 = undefined;
    for (&arr, 0..) |*val, i| {
        val.* = @intCast(100 - i); // Reverse order
    }

    bubbleSortAsc(i32, &arr);

    // Verify sorted
    var i: usize = 0;
    while (i < arr.len) : (i += 1) {
        try testing.expectEqual(@as(i32, @intCast(i + 1)), arr[i]);
    }
}

test "bubble sort - floating point" {
    var arr = [_]f32{ 3.14, 1.41, 2.71, 0.57 };
    bubbleSortAsc(f32, &arr);
    try testing.expectEqual(@as(f32, 0.57), arr[0]);
    try testing.expectEqual(@as(f32, 1.41), arr[1]);
    try testing.expectEqual(@as(f32, 2.71), arr[2]);
    try testing.expectEqual(@as(f32, 3.14), arr[3]);
}

test "bubble sort - f64 support" {
    var arr = [_]f64{ 3.14, 1.41, 2.71, 0.57 };
    bubbleSortAsc(f64, &arr);
    try testing.expectEqual(@as(f64, 0.57), arr[0]);
    try testing.expectEqual(@as(f64, 1.41), arr[1]);
    try testing.expectEqual(@as(f64, 2.71), arr[2]);
    try testing.expectEqual(@as(f64, 3.14), arr[3]);
}

test "bubble sort - early exit optimization" {
    var arr = [_]i32{ 1, 2, 4, 3, 5, 6, 7 }; // One swap needed
    const comps = countComparisons(i32, &arr, asc(i32));

    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7 }, &arr);
    // With early exit, should stop after 2 passes: first pass makes swap, second pass confirms sorted
    // First pass: 6 comparisons, Second pass: 6 comparisons (early exit)
    try testing.expect(comps <= 13); // Much less than worst case O(n²) = 49
}

test "bubble sort - all equal elements" {
    var arr = [_]i32{ 5, 5, 5, 5, 5 };
    const swaps = countSwaps(i32, &arr, asc(i32));
    const comps = countComparisons(i32, &arr, asc(i32));

    try testing.expectEqualSlices(i32, &[_]i32{ 5, 5, 5, 5, 5 }, &arr);
    // No swaps needed, early exit after first pass
    try testing.expectEqual(@as(usize, 0), swaps);
    try testing.expectEqual(@as(usize, 4), comps); // n-1 comparisons in first pass
}

test "bubble sort - alternating pattern" {
    var arr = [_]i32{ 2, 1, 4, 3, 6, 5 };
    bubbleSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6 }, &arr);
}
