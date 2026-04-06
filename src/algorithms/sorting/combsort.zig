const std = @import("std");
const testing = std.testing;
const Order = std.math.Order;

/// Comb Sort — Improved Bubble Sort with Gap-Based Comparisons
///
/// An improvement over Bubble Sort that eliminates "turtles" (small values near the end)
/// by comparing elements separated by a gap, similar to Shell Sort's approach for Insertion Sort.
///
/// ## Algorithm Overview
///
/// Comb Sort works by:
/// 1. Starting with a large gap (typically n / 1.3)
/// 2. Comparing and swapping elements that are gap distance apart
/// 3. Reducing the gap by dividing by the shrink factor (1.3)
/// 4. Repeating until gap becomes 1 (standard bubble sort)
/// 5. Performing a final pass with gap=1 to ensure the array is sorted
///
/// The shrink factor of 1.3 was empirically determined to provide optimal performance.
///
/// ## Properties
///
/// - **Time Complexity**:
///   * Best case: O(n log n) when shrink factor is well-chosen
///   * Average case: O(n²/2^p) where p is number of passes
///   * Worst case: O(n²) (better than Bubble Sort in practice)
/// - **Space Complexity**: O(1) — in-place sorting
/// - **Stability**: Unstable (relative order of equal elements not preserved)
/// - **Adaptive**: Slightly adaptive (can terminate early if no swaps)
///
/// ## Use Cases
///
/// - Educational: Understanding gap-based improvements over simple algorithms
/// - Small to medium datasets where simplicity matters
/// - When memory is extremely limited (no allocation needed)
/// - Replacement for Bubble Sort in legacy code (drop-in with better performance)
/// - Systems where O(n log n) sorting is unavailable
///
/// ## Comparison with Other Algorithms
///
/// - vs Bubble Sort: Much faster (eliminates turtles), same simplicity
/// - vs Shell Sort: Similar concept but for Bubble Sort instead of Insertion Sort
/// - vs QuickSort/MergeSort: Simpler but slower, no worst-case O(n log n) guarantee
/// - vs Insertion Sort: Better for random data, worse for nearly sorted data
///
/// ## References
///
/// - Włodzimierz Dobosiewicz (1980) — Original publication
/// - Stephen Lacey and Richard Box (1991) — Analysis and shrink factor 1.3

/// Comb Sort with custom comparison function
///
/// Sorts the slice in-place using Comb Sort algorithm with a custom comparison function.
///
/// ## Time Complexity
/// - Best case: O(n log n)
/// - Average case: O(n²/2^p)
/// - Worst case: O(n²)
///
/// ## Space Complexity
/// - O(1) — in-place sorting, no allocation
///
/// ## Parameters
/// - `T`: The type of elements to sort
/// - `items`: Slice to sort in-place
/// - `context`: User-defined context passed to comparison function
/// - `lessThan`: Comparison function returning true if first argument is less than second
///
/// ## Example
/// ```zig
/// const items = [_]i32{ 5, 2, 8, 1, 9 };
/// combSort(i32, &items, {}, comptime asc(i32));
/// // items is now [1, 2, 5, 8, 9]
/// ```
pub fn combSort(
    comptime T: type,
    items: []T,
    context: anytype,
    comptime lessThan: fn (@TypeOf(context), T, T) bool,
) void {
    const n = items.len;
    if (n <= 1) return;

    // Shrink factor (1.3 is empirically optimal)
    const shrink_factor = 1.3;

    var gap: usize = n;
    var swapped = true;

    while (gap > 1 or swapped) {
        // Update gap with shrink factor
        if (gap > 1) {
            gap = @intFromFloat(@as(f64, @floatFromInt(gap)) / shrink_factor);
            if (gap < 1) gap = 1;
        }

        swapped = false;
        var i: usize = 0;
        while (i + gap < n) : (i += 1) {
            if (lessThan(context, items[i + gap], items[i])) {
                std.mem.swap(T, &items[i], &items[i + gap]);
                swapped = true;
            }
        }
    }
}

/// Comb Sort in ascending order
///
/// Convenience wrapper for sorting in ascending order using default comparison.
///
/// ## Time Complexity
/// - Best case: O(n log n)
/// - Average case: O(n²/2^p)
/// - Worst case: O(n²)
///
/// ## Space Complexity
/// - O(1) — in-place sorting
///
/// ## Example
/// ```zig
/// var items = [_]i32{ 5, 2, 8, 1, 9 };
/// combSortAsc(i32, &items);
/// // items is now [1, 2, 5, 8, 9]
/// ```
pub fn combSortAsc(comptime T: type, items: []T) void {
    combSort(T, items, {}, asc(T));
}

/// Comb Sort in descending order
///
/// Convenience wrapper for sorting in descending order using default comparison.
///
/// ## Time Complexity
/// - Best case: O(n log n)
/// - Average case: O(n²/2^p)
/// - Worst case: O(n²)
///
/// ## Space Complexity
/// - O(1) — in-place sorting
///
/// ## Example
/// ```zig
/// var items = [_]i32{ 5, 2, 8, 1, 9 };
/// combSortDesc(i32, &items);
/// // items is now [9, 8, 5, 2, 1]
/// ```
pub fn combSortDesc(comptime T: type, items: []T) void {
    combSort(T, items, {}, desc(T));
}

/// Comb Sort using Order-based comparison
///
/// Sorts using a comparison function that returns std.math.Order.
///
/// ## Parameters
/// - `T`: The type of elements to sort
/// - `items`: Slice to sort in-place
/// - `context`: User-defined context
/// - `compare`: Function returning Order (.lt, .eq, .gt)
///
/// ## Example
/// ```zig
/// const S = struct { id: i32 };
/// fn compareById(_: void, a: S, b: S) Order {
///     return std.math.order(a.id, b.id);
/// }
/// var items = [_]S{ .{.id=3}, .{.id=1}, .{.id=2} };
/// combSortBy(S, &items, {}, compareById);
/// ```
pub fn combSortBy(
    comptime T: type,
    items: []T,
    context: anytype,
    comptime compare: fn (@TypeOf(context), T, T) Order,
) void {
    const Adapter = struct {
        fn lessThan(ctx: @TypeOf(context), a: T, b: T) bool {
            return compare(ctx, a, b) == .lt;
        }
    };
    combSort(T, items, context, Adapter.lessThan);
}

/// Comb Sort with custom shrink factor
///
/// Allows experimentation with different shrink factors.
/// The standard shrink factor of 1.3 is empirically optimal,
/// but this function allows testing other values.
///
/// ## Parameters
/// - `shrink_factor`: Gap reduction multiplier (standard is 1.3)
///   * Must be > 1.0 for convergence
///   * Typical range: 1.25 to 1.35
///   * Smaller values = more passes, potentially better sorted
///   * Larger values = fewer passes, potentially more work in final pass
///
/// ## Time Complexity
/// - Depends on shrink factor choice
/// - Standard 1.3: O(n log n) to O(n²/2^p)
///
/// ## Space Complexity
/// - O(1) — in-place sorting
///
/// ## Example
/// ```zig
/// var items = [_]i32{ 5, 2, 8, 1, 9 };
/// combSortCustom(i32, &items, {}, asc(i32), 1.25);
/// // Smaller shrink factor = more gradual gap reduction
/// ```
pub fn combSortCustom(
    comptime T: type,
    items: []T,
    context: anytype,
    comptime lessThan: fn (@TypeOf(context), T, T) bool,
    shrink_factor: f64,
) void {
    const n = items.len;
    if (n <= 1) return;

    var gap: usize = n;
    var swapped = true;

    while (gap > 1 or swapped) {
        if (gap > 1) {
            gap = @intFromFloat(@as(f64, @floatFromInt(gap)) / shrink_factor);
            if (gap < 1) gap = 1;
        }

        swapped = false;
        var i: usize = 0;
        while (i + gap < n) : (i += 1) {
            if (lessThan(context, items[i + gap], items[i])) {
                std.mem.swap(T, &items[i], &items[i + gap]);
                swapped = true;
            }
        }
    }
}

/// Ascending order comparison for numeric types
fn asc(comptime T: type) fn (void, T, T) bool {
    return struct {
        fn lessThan(_: void, a: T, b: T) bool {
            return a < b;
        }
    }.lessThan;
}

/// Descending order comparison for numeric types
fn desc(comptime T: type) fn (void, T, T) bool {
    return struct {
        fn lessThan(_: void, a: T, b: T) bool {
            return a > b;
        }
    }.lessThan;
}

// ============================================================================
// Tests
// ============================================================================

test "comb sort - basic ascending" {
    var items = [_]i32{ 5, 2, 8, 1, 9, 3, 7, 4, 6 };
    combSortAsc(i32, &items);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &items);
}

test "comb sort - basic descending" {
    var items = [_]i32{ 5, 2, 8, 1, 9, 3, 7, 4, 6 };
    combSortDesc(i32, &items);
    try testing.expectEqualSlices(i32, &[_]i32{ 9, 8, 7, 6, 5, 4, 3, 2, 1 }, &items);
}

test "comb sort - empty array" {
    var items: [0]i32 = .{};
    combSortAsc(i32, &items);
    try testing.expectEqual(@as(usize, 0), items.len);
}

test "comb sort - single element" {
    var items = [_]i32{42};
    combSortAsc(i32, &items);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &items);
}

test "comb sort - two elements" {
    var items1 = [_]i32{ 2, 1 };
    combSortAsc(i32, &items1);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2 }, &items1);

    var items2 = [_]i32{ 1, 2 };
    combSortAsc(i32, &items2);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2 }, &items2);
}

test "comb sort - already sorted" {
    var items = [_]i32{ 1, 2, 3, 4, 5 };
    combSortAsc(i32, &items);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &items);
}

test "comb sort - reverse sorted" {
    var items = [_]i32{ 5, 4, 3, 2, 1 };
    combSortAsc(i32, &items);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &items);
}

test "comb sort - duplicates" {
    var items = [_]i32{ 5, 2, 8, 2, 9, 5, 7, 2, 6 };
    combSortAsc(i32, &items);
    try testing.expectEqualSlices(i32, &[_]i32{ 2, 2, 2, 5, 5, 6, 7, 8, 9 }, &items);
}

test "comb sort - all same elements" {
    var items = [_]i32{ 7, 7, 7, 7, 7 };
    combSortAsc(i32, &items);
    try testing.expectEqualSlices(i32, &[_]i32{ 7, 7, 7, 7, 7 }, &items);
}

test "comb sort - negative numbers" {
    var items = [_]i32{ 5, -2, 8, -1, 0, 3, -7, 4 };
    combSortAsc(i32, &items);
    try testing.expectEqualSlices(i32, &[_]i32{ -7, -2, -1, 0, 3, 4, 5, 8 }, &items);
}

test "comb sort - floating point" {
    var items = [_]f64{ 3.14, 1.41, 2.71, 0.5, 1.73 };
    combSortAsc(f64, &items);
    try testing.expectEqualSlices(f64, &[_]f64{ 0.5, 1.41, 1.73, 2.71, 3.14 }, &items);
}

test "comb sort - custom comparison" {
    const S = struct {
        id: i32,
        value: []const u8,
    };
    var items = [_]S{
        .{ .id = 3, .value = "three" },
        .{ .id = 1, .value = "one" },
        .{ .id = 2, .value = "two" },
    };

    const Context = struct {
        fn lessThan(_: @This(), a: S, b: S) bool {
            return a.id < b.id;
        }
    };

    combSort(S, &items, Context{}, Context.lessThan);
    try testing.expectEqual(@as(i32, 1), items[0].id);
    try testing.expectEqual(@as(i32, 2), items[1].id);
    try testing.expectEqual(@as(i32, 3), items[2].id);
}

test "comb sort - Order-based comparison" {
    const S = struct {
        id: i32,
    };

    const compareById = struct {
        fn cmp(_: void, a: S, b: S) Order {
            return std.math.order(a.id, b.id);
        }
    }.cmp;

    var items = [_]S{
        .{ .id = 5 },
        .{ .id = 2 },
        .{ .id = 8 },
        .{ .id = 1 },
    };

    combSortBy(S, &items, {}, compareById);
    try testing.expectEqual(@as(i32, 1), items[0].id);
    try testing.expectEqual(@as(i32, 2), items[1].id);
    try testing.expectEqual(@as(i32, 5), items[2].id);
    try testing.expectEqual(@as(i32, 8), items[3].id);
}

test "comb sort - large array" {
    const allocator = testing.allocator;
    const n = 100;
    var items = try allocator.alloc(i32, n);
    defer allocator.free(items);

    // Fill with reverse-sorted data
    var i: usize = 0;
    while (i < n) : (i += 1) {
        items[i] = @intCast(n - i);
    }

    combSortAsc(i32, items);

    // Verify sorted
    i = 0;
    while (i < n) : (i += 1) {
        try testing.expectEqual(@as(i32, @intCast(i + 1)), items[i]);
    }
}

test "comb sort - u8 type" {
    var items = [_]u8{ 255, 0, 128, 64, 192, 32, 96, 160, 224 };
    combSortAsc(u8, &items);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 32, 64, 96, 128, 160, 192, 224, 255 }, &items);
}

test "comb sort - custom shrink factor 1.25" {
    var items = [_]i32{ 5, 2, 8, 1, 9, 3, 7, 4, 6 };
    combSortCustom(i32, &items, {}, asc(i32), 1.25);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &items);
}

test "comb sort - custom shrink factor 1.5" {
    var items = [_]i32{ 5, 2, 8, 1, 9, 3, 7, 4, 6 };
    combSortCustom(i32, &items, {}, asc(i32), 1.5);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &items);
}

test "comb sort - stress test with pseudo-random data" {
    var items: [50]i32 = undefined;
    var seed: u32 = 12345;
    for (&items) |*item| {
        // Simple LCG pseudo-random
        seed = seed *% 1103515245 +% 12345;
        item.* = @as(i32, @intCast(seed % 1000)) - 500;
    }

    combSortAsc(i32, &items);

    // Verify sorted
    var i: usize = 1;
    while (i < items.len) : (i += 1) {
        try testing.expect(items[i - 1] <= items[i]);
    }
}

test "comb sort - eliminates turtles (small values at end)" {
    // Classic problem for bubble sort: small value at the end
    var items = [_]i32{ 2, 3, 4, 5, 6, 7, 8, 9, 10, 1 };
    combSortAsc(i32, &items);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, &items);
}

test "comb sort - comparison with different shrink factors" {
    const data = [_]i32{ 9, 7, 5, 11, 12, 2, 14, 3, 10, 6 };

    // Test multiple shrink factors produce same result
    var items1 = data;
    combSortCustom(i32, &items1, {}, asc(i32), 1.25);

    var items2 = data;
    combSortCustom(i32, &items2, {}, asc(i32), 1.3);

    var items3 = data;
    combSortCustom(i32, &items3, {}, asc(i32), 1.5);

    // All should produce correctly sorted output
    const expected = [_]i32{ 2, 3, 5, 6, 7, 9, 10, 11, 12, 14 };
    try testing.expectEqualSlices(i32, &expected, &items1);
    try testing.expectEqualSlices(i32, &expected, &items2);
    try testing.expectEqualSlices(i32, &expected, &items3);
}
