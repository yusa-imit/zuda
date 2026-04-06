const std = @import("std");
const testing = std.testing;

/// Selection sort - simple comparison-based sorting algorithm.
/// Finds minimum element in unsorted portion and swaps it to the front.
///
/// Properties:
/// - NOT stable (can swap equal elements out of original order)
/// - In-place: O(1) extra space
/// - Minimal swaps: Always O(n) swaps regardless of input
/// - Non-adaptive: O(n²) comparisons even for sorted data
/// - Good for: Teaching, when write operations are expensive (flash memory)
///
/// Time: O(n²) comparisons, O(n) swaps (best/average/worst case)
/// Space: O(1)
///
/// Algorithm:
/// 1. For each position i from 0 to n-1:
///    - Find minimum element in arr[i..n]
///    - Swap minimum with arr[i]
/// 2. After pass i, arr[0..i] is sorted and contains the i smallest elements
///
/// Example:
/// ```zig
/// var arr = [_]i32{ 64, 25, 12, 22, 11 };
/// selectionSort(i32, &arr, {}, asc);
/// // arr = [11, 12, 22, 25, 64]
/// ```
///
/// Use cases:
/// - Educational purposes (simple to understand)
/// - When write operations are expensive (minimizes swaps)
/// - Small datasets where simplicity matters
/// - Embedded systems with limited memory
pub fn selectionSort(
    comptime T: type,
    arr: []T,
    context: anytype,
    comptime lessThanFn: fn (@TypeOf(context), T, T) bool,
) void {
    if (arr.len <= 1) return;

    const n = arr.len;
    var i: usize = 0;

    // For each position, find the minimum in the remaining array
    while (i < n - 1) : (i += 1) {
        var min_idx = i;
        var j = i + 1;

        // Find minimum element in arr[i+1..n]
        while (j < n) : (j += 1) {
            if (lessThanFn(context, arr[j], arr[min_idx])) {
                min_idx = j;
            }
        }

        // Swap minimum with current position
        if (min_idx != i) {
            const temp = arr[i];
            arr[i] = arr[min_idx];
            arr[min_idx] = temp;
        }
    }
}

/// Selection sort with default ascending order.
/// Time: O(n²)
/// Space: O(1)
pub fn selectionSortAsc(comptime T: type, arr: []T) void {
    selectionSort(T, arr, {}, asc(T));
}

/// Selection sort with descending order.
/// Time: O(n²)
/// Space: O(1)
pub fn selectionSortDesc(comptime T: type, arr: []T) void {
    selectionSort(T, arr, {}, desc(T));
}

/// Selection sort wrapper for custom comparison functions.
/// Time: O(n²)
/// Space: O(1)
pub fn sortBy(
    comptime T: type,
    arr: []T,
    context: anytype,
    comptime lessThanFn: fn (@TypeOf(context), T, T) bool,
) void {
    selectionSort(T, arr, context, lessThanFn);
}

/// Creates a default ascending comparison function for type T.
fn asc(comptime T: type) fn (void, T, T) bool {
    return struct {
        fn cmp(_: void, a: T, b: T) bool {
            return a < b;
        }
    }.cmp;
}

/// Creates a default descending comparison function for type T.
fn desc(comptime T: type) fn (void, T, T) bool {
    return struct {
        fn cmp(_: void, a: T, b: T) bool {
            return a > b;
        }
    }.cmp;
}

/// Counts the number of swaps performed by selection sort (always n-1 or less).
/// Time: O(n²)
/// Space: O(1)
pub fn countSwaps(
    comptime T: type,
    arr: []T,
    context: anytype,
    comptime lessThanFn: fn (@TypeOf(context), T, T) bool,
) usize {
    if (arr.len <= 1) return 0;

    var swap_count: usize = 0;
    const n = arr.len;
    var i: usize = 0;

    while (i < n - 1) : (i += 1) {
        var min_idx = i;
        var j = i + 1;

        while (j < n) : (j += 1) {
            if (lessThanFn(context, arr[j], arr[min_idx])) {
                min_idx = j;
            }
        }

        if (min_idx != i) {
            const temp = arr[i];
            arr[i] = arr[min_idx];
            arr[min_idx] = temp;
            swap_count += 1;
        }
    }

    return swap_count;
}

// ============================================================================
// Tests
// ============================================================================

test "selectionSort - basic ascending" {
    var arr = [_]i32{ 64, 25, 12, 22, 11 };
    selectionSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 11, 12, 22, 25, 64 }, &arr);
}

test "selectionSort - basic descending" {
    var arr = [_]i32{ 64, 25, 12, 22, 11 };
    selectionSortDesc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 64, 25, 22, 12, 11 }, &arr);
}

test "selectionSort - empty array" {
    var arr = [_]i32{};
    selectionSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{}, &arr);
}

test "selectionSort - single element" {
    var arr = [_]i32{42};
    selectionSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &arr);
}

test "selectionSort - two elements" {
    var arr = [_]i32{ 2, 1 };
    selectionSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2 }, &arr);
}

test "selectionSort - already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    const original = arr;
    selectionSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &original, &arr);
}

test "selectionSort - reverse sorted" {
    var arr = [_]i32{ 5, 4, 3, 2, 1 };
    selectionSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "selectionSort - duplicates" {
    var arr = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    selectionSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2, 3, 3, 4, 5, 5, 6, 9 }, &arr);
}

test "selectionSort - all same" {
    var arr = [_]i32{ 7, 7, 7, 7, 7 };
    selectionSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 7, 7, 7, 7, 7 }, &arr);
}

test "selectionSort - negative numbers" {
    var arr = [_]i32{ -5, 3, -2, 8, -1, 0 };
    selectionSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ -5, -2, -1, 0, 3, 8 }, &arr);
}

test "selectionSort - floating point" {
    var arr = [_]f64{ 3.14, 1.41, 2.71, 1.73, 2.23 };
    selectionSortAsc(f64, &arr);
    try testing.expectEqual(@as(f64, 1.41), arr[0]);
    try testing.expectEqual(@as(f64, 1.73), arr[1]);
    try testing.expectEqual(@as(f64, 2.23), arr[2]);
    try testing.expectEqual(@as(f64, 2.71), arr[3]);
    try testing.expectEqual(@as(f64, 3.14), arr[4]);
}

test "selectionSort - NOT stable (equal elements can be reordered)" {
    const Pair = struct {
        key: i32,
        value: u8,
    };

    const lessThan = struct {
        fn cmp(_: void, a: Pair, b: Pair) bool {
            return a.key < b.key;
        }
    }.cmp;

    // Two pairs with key=2, originally in order 'a', 'b'
    var arr = [_]Pair{
        .{ .key = 3, .value = 'c' },
        .{ .key = 2, .value = 'a' },
        .{ .key = 1, .value = 'x' },
        .{ .key = 2, .value = 'b' },
    };

    selectionSort(Pair, &arr, {}, lessThan);

    // After sorting by key: [1, 2, 2, 3]
    try testing.expectEqual(@as(i32, 1), arr[0].key);
    try testing.expectEqual(@as(i32, 2), arr[1].key);
    try testing.expectEqual(@as(i32, 2), arr[2].key);
    try testing.expectEqual(@as(i32, 3), arr[3].key);

    // Selection sort is NOT stable - equal keys may be out of original order
    // We can't assert the exact order of 'a' and 'b', but we verify they exist
    const second_val = arr[1].value;
    const third_val = arr[2].value;
    const has_a = (second_val == 'a' or third_val == 'a');
    const has_b = (second_val == 'b' or third_val == 'b');
    try testing.expect(has_a);
    try testing.expect(has_b);
}

test "selectionSort - custom comparison struct" {
    const Person = struct {
        name: []const u8,
        age: u32,
    };

    const byAge = struct {
        fn cmp(_: void, a: Person, b: Person) bool {
            return a.age < b.age;
        }
    }.cmp;

    var people = [_]Person{
        .{ .name = "Alice", .age = 30 },
        .{ .name = "Bob", .age = 25 },
        .{ .name = "Charlie", .age = 35 },
        .{ .name = "Dave", .age = 20 },
    };

    sortBy(Person, &people, {}, byAge);

    try testing.expectEqual(@as(u32, 20), people[0].age);
    try testing.expectEqual(@as(u32, 25), people[1].age);
    try testing.expectEqual(@as(u32, 30), people[2].age);
    try testing.expectEqual(@as(u32, 35), people[3].age);
}

test "selectionSort - large array" {
    const allocator = testing.allocator;
    const n = 100;
    var arr = try allocator.alloc(i32, n);
    defer allocator.free(arr);

    // Fill with reverse sorted data
    for (0..n) |i| {
        arr[i] = @intCast(n - i);
    }

    selectionSortAsc(i32, arr);

    // Verify sorted
    for (0..n) |i| {
        try testing.expectEqual(@as(i32, @intCast(i + 1)), arr[i]);
    }
}

test "selectionSort - count swaps on sorted array" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    const swaps = countSwaps(i32, &arr, {}, asc(i32));
    // Already sorted - no swaps needed
    try testing.expectEqual(@as(usize, 0), swaps);
}

test "selectionSort - count swaps on reverse sorted" {
    var arr = [_]i32{ 5, 4, 3, 2, 1 };
    const swaps = countSwaps(i32, &arr, {}, asc(i32));
    // Reverse sorted - requires n/2 = 2 swaps (swap 5 with 1, swap 4 with 2)
    try testing.expectEqual(@as(usize, 2), swaps);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "selectionSort - count swaps general case" {
    var arr = [_]i32{ 64, 25, 12, 22, 11 };
    const swaps = countSwaps(i32, &arr, {}, asc(i32));
    // Each pass finds min and swaps (except when min is already at position)
    // Maximum n-1 swaps, minimum 0 swaps
    try testing.expect(swaps <= arr.len - 1);
    try testing.expectEqualSlices(i32, &[_]i32{ 11, 12, 22, 25, 64 }, &arr);
}

test "selectionSort - minimal swaps property" {
    // Selection sort always makes at most n-1 swaps
    var arr1 = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    const swaps1 = countSwaps(i32, &arr1, {}, asc(i32));
    try testing.expect(swaps1 <= arr1.len - 1);

    var arr2 = [_]i32{ 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
    const swaps2 = countSwaps(i32, &arr2, {}, asc(i32));
    try testing.expect(swaps2 <= arr2.len - 1);
}

test "selectionSort - non-adaptive (always O(n²) comparisons)" {
    // Even on already sorted data, selection sort does O(n²) comparisons
    var arr = [_]i32{ 1, 2, 3, 4, 5 };

    // We can't easily measure comparisons without instrumentation,
    // but we verify it still sorts correctly
    selectionSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);

    // Note: Unlike insertion sort which is O(n) on sorted data,
    // selection sort is always O(n²) regardless of input
}

test "selectionSort - u8 array" {
    var arr = [_]u8{ 255, 128, 64, 32, 16, 8, 4, 2, 1, 0 };
    selectionSortAsc(u8, &arr);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 1, 2, 4, 8, 16, 32, 64, 128, 255 }, &arr);
}

test "selectionSort - f32 array" {
    var arr = [_]f32{ 3.5, 1.2, 4.8, 2.1, 5.9 };
    selectionSortAsc(f32, &arr);
    try testing.expectApproxEqAbs(@as(f32, 1.2), arr[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 2.1), arr[1], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 3.5), arr[2], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 4.8), arr[3], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 5.9), arr[4], 0.001);
}
