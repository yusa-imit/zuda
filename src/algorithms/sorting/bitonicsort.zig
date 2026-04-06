const std = @import("std");
const testing = std.testing;
const Order = std.math.Order;

/// Bitonic Sort — Parallel Sorting Network
///
/// A comparison-based sorting algorithm designed for parallel execution.
/// Uses a bitonic sorting network with O(log² n) parallel depth.
///
/// Key Properties:
/// - Data-oblivious: comparison sequence independent of input values
/// - Parallel-friendly: O(log² n) parallel depth with O(n log² n) work
/// - Power-of-2 restriction: n must be a power of 2 (pads with sentinel values if not)
/// - In-place: O(1) auxiliary space
/// - Unstable: does not preserve relative order of equal elements
///
/// Time Complexity:
/// - Sequential: O(n log² n) comparisons
/// - Parallel: O(log² n) depth with O(n) processors
///
/// Space Complexity: O(1) auxiliary space
///
/// Use Cases:
/// - Parallel/SIMD sorting (GPU sorting, multi-core processors)
/// - Hardware sorting networks (FPGA, ASIC)
/// - Oblivious sorting (security/privacy — constant-time execution)
/// - Fixed-size sorting (when n is known to be power of 2)
///
/// Algorithm:
/// 1. Build bitonic sequence: recursively sort halves in opposite directions
/// 2. Bitonic merge: compare-and-swap elements distance apart, reducing distance
/// 3. Network structure: deterministic comparison pattern (sorting network)
///
/// Reference: K. E. Batcher (1968) "Sorting networks and their applications"
///
/// Example:
/// ```zig
/// var arr = [_]i32{ 3, 7, 4, 8, 6, 2, 1, 5 };
/// try bitonicSort(i32, &arr, {}, std.sort.asc);
/// // arr is now [1, 2, 3, 4, 5, 6, 7, 8]
/// ```

/// Bitonic Sort with custom comparison function
///
/// Time: O(n log² n) sequential, O(log² n) parallel depth
/// Space: O(1)
///
/// Note: Array length must be a power of 2. Use bitonicSortAny for arbitrary lengths.
pub fn bitonicSort(
    comptime T: type,
    items: []T,
    context: anytype,
    comptime compareFn: fn (@TypeOf(context), T, T) bool,
) void {
    const n = items.len;
    if (n < 2) return;

    // Verify power of 2
    if (!std.math.isPowerOfTwo(n)) {
        @panic("bitonicSort requires array length to be power of 2. Use bitonicSortAny for arbitrary lengths.");
    }

    bitonicSortRecursive(T, items, 0, n, true, context, compareFn);
}

/// Bitonic Sort for ascending order (convenience wrapper)
///
/// Time: O(n log² n)
/// Space: O(1)
pub fn bitonicSortAsc(comptime T: type, items: []T) void {
    bitonicSort(T, items, {}, comptime asc(T));
}

/// Bitonic Sort for descending order (convenience wrapper)
///
/// Time: O(n log² n)
/// Space: O(1)
pub fn bitonicSortDesc(comptime T: type, items: []T) void {
    bitonicSort(T, items, {}, comptime desc(T));
}

/// Bitonic Sort with std.math.Order-based comparison
///
/// Time: O(n log² n)
/// Space: O(1)
pub fn bitonicSortBy(
    comptime T: type,
    items: []T,
    context: anytype,
    comptime orderFn: fn (@TypeOf(context), T, T) Order,
) void {
    const Wrapper = struct {
        fn lessThan(ctx: @TypeOf(context), a: T, b: T) bool {
            return orderFn(ctx, a, b) == .lt;
        }
    };
    bitonicSort(T, items, context, Wrapper.lessThan);
}

/// Bitonic Sort for arbitrary array lengths (pads to next power of 2)
///
/// Time: O(n log² n) where n is rounded up to next power of 2
/// Space: O(n) for padded array
///
/// For arrays not power-of-2, pads with sentinel values and sorts the padded array.
pub fn bitonicSortAny(
    comptime T: type,
    items: []T,
    allocator: std.mem.Allocator,
    context: anytype,
    comptime compareFn: fn (@TypeOf(context), T, T) bool,
) !void {
    const n = items.len;
    if (n < 2) return;

    // If already power of 2, sort in-place
    if (std.math.isPowerOfTwo(n)) {
        bitonicSort(T, items, context, compareFn);
        return;
    }

    // Pad to next power of 2
    const next_pow2 = std.math.ceilPowerOfTwo(usize, n) catch return error.Overflow;
    var padded = try allocator.alloc(T, next_pow2);
    defer allocator.free(padded);

    // Copy original items
    @memcpy(padded[0..n], items);

    // Fill padding with sentinel value (max element from original array)
    const max_val = blk: {
        var max = items[0];
        for (items[1..]) |item| {
            if (!compareFn(context, item, max)) {
                max = item;
            }
        }
        break :blk max;
    };
    @memset(padded[n..], max_val);

    // Sort padded array
    bitonicSort(T, padded, context, compareFn);

    // Copy back (only original n elements)
    @memcpy(items, padded[0..n]);
}

// ============================================================================
// Internal Implementation
// ============================================================================

fn bitonicSortRecursive(
    comptime T: type,
    items: []T,
    low: usize,
    cnt: usize,
    ascending: bool,
    context: anytype,
    comptime compareFn: fn (@TypeOf(context), T, T) bool,
) void {
    if (cnt <= 1) return;

    const k = cnt / 2;

    // Sort first half in ascending order
    bitonicSortRecursive(T, items, low, k, true, context, compareFn);

    // Sort second half in descending order
    bitonicSortRecursive(T, items, low + k, k, false, context, compareFn);

    // Merge the whole sequence in the specified order
    bitonicMerge(T, items, low, cnt, ascending, context, compareFn);
}

fn bitonicMerge(
    comptime T: type,
    items: []T,
    low: usize,
    cnt: usize,
    ascending: bool,
    context: anytype,
    comptime compareFn: fn (@TypeOf(context), T, T) bool,
) void {
    if (cnt <= 1) return;

    const k = cnt / 2;

    // Compare and swap elements at distance k apart
    var i: usize = low;
    while (i < low + k) : (i += 1) {
        compareAndSwap(T, items, i, i + k, ascending, context, compareFn);
    }

    // Recursively merge both halves
    bitonicMerge(T, items, low, k, ascending, context, compareFn);
    bitonicMerge(T, items, low + k, k, ascending, context, compareFn);
}

fn compareAndSwap(
    comptime T: type,
    items: []T,
    i: usize,
    j: usize,
    ascending: bool,
    context: anytype,
    comptime compareFn: fn (@TypeOf(context), T, T) bool,
) void {
    const should_swap = if (ascending)
        !compareFn(context, items[i], items[j]) // i > j for ascending
    else
        compareFn(context, items[i], items[j]); // i < j for descending

    if (should_swap) {
        std.mem.swap(T, &items[i], &items[j]);
    }
}

// ============================================================================
// Comparison Functions
// ============================================================================

fn asc(comptime T: type) fn (void, T, T) bool {
    return struct {
        fn lessThan(_: void, a: T, b: T) bool {
            return a < b;
        }
    }.lessThan;
}

fn desc(comptime T: type) fn (void, T, T) bool {
    return struct {
        fn greaterThan(_: void, a: T, b: T) bool {
            return a > b;
        }
    }.greaterThan;
}

// ============================================================================
// Tests
// ============================================================================

test "bitonicSort: basic ascending (power of 2)" {
    var arr = [_]i32{ 3, 7, 4, 8, 6, 2, 1, 5 };
    bitonicSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 }, &arr);
}

test "bitonicSort: basic descending (power of 2)" {
    var arr = [_]i32{ 3, 7, 4, 8, 6, 2, 1, 5 };
    bitonicSortDesc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 8, 7, 6, 5, 4, 3, 2, 1 }, &arr);
}

test "bitonicSort: empty array" {
    var arr = [_]i32{};
    bitonicSortAsc(i32, &arr);
    try testing.expectEqual(0, arr.len);
}

test "bitonicSort: single element" {
    var arr = [_]i32{42};
    bitonicSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &arr);
}

test "bitonicSort: two elements" {
    var arr = [_]i32{ 5, 3 };
    bitonicSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 3, 5 }, &arr);
}

test "bitonicSort: already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    bitonicSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 }, &arr);
}

test "bitonicSort: reverse sorted" {
    var arr = [_]i32{ 8, 7, 6, 5, 4, 3, 2, 1 };
    bitonicSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 }, &arr);
}

test "bitonicSort: duplicates" {
    var arr = [_]i32{ 3, 1, 3, 2, 2, 1, 3, 2 };
    bitonicSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2, 2, 2, 3, 3, 3 }, &arr);
}

test "bitonicSort: all equal" {
    var arr = [_]i32{ 5, 5, 5, 5, 5, 5, 5, 5 };
    bitonicSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 5, 5, 5, 5, 5, 5, 5, 5 }, &arr);
}

test "bitonicSort: negative numbers" {
    var arr = [_]i32{ -3, 7, -4, 8, -6, 2, -1, 5 };
    bitonicSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ -6, -4, -3, -1, 2, 5, 7, 8 }, &arr);
}

test "bitonicSort: floating point (f64)" {
    var arr = [_]f64{ 3.5, 1.2, 4.8, 2.1, 5.9, 0.3, 7.6, 6.4 };
    bitonicSortAsc(f64, &arr);
    try testing.expectEqualSlices(f64, &[_]f64{ 0.3, 1.2, 2.1, 3.5, 4.8, 5.9, 6.4, 7.6 }, &arr);
}

test "bitonicSort: custom comparison (struct by age)" {
    const Person = struct {
        name: []const u8,
        age: u32,
    };

    const compareByAge = struct {
        fn lessThan(_: void, a: Person, b: Person) bool {
            return a.age < b.age;
        }
    }.lessThan;

    var people = [_]Person{
        .{ .name = "Alice", .age = 30 },
        .{ .name = "Bob", .age = 25 },
        .{ .name = "Charlie", .age = 35 },
        .{ .name = "David", .age = 20 },
        .{ .name = "Eve", .age = 28 },
        .{ .name = "Frank", .age = 32 },
        .{ .name = "Grace", .age = 22 },
        .{ .name = "Henry", .age = 27 },
    };

    bitonicSort(Person, &people, {}, compareByAge);

    try testing.expectEqual(20, people[0].age);
    try testing.expectEqual(22, people[1].age);
    try testing.expectEqual(25, people[2].age);
    try testing.expectEqual(27, people[3].age);
    try testing.expectEqual(28, people[4].age);
    try testing.expectEqual(30, people[5].age);
    try testing.expectEqual(32, people[6].age);
    try testing.expectEqual(35, people[7].age);
}

test "bitonicSort: Order-based comparison" {
    const compareOrder = struct {
        fn order(_: void, a: i32, b: i32) Order {
            if (a < b) return .lt;
            if (a > b) return .gt;
            return .eq;
        }
    }.order;

    var arr = [_]i32{ 3, 7, 4, 8, 6, 2, 1, 5 };
    bitonicSortBy(i32, &arr, {}, compareOrder);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 }, &arr);
}

test "bitonicSort: u8 type" {
    var arr = [_]u8{ 200, 50, 150, 100, 250, 25, 175, 75 };
    bitonicSortAsc(u8, &arr);
    try testing.expectEqualSlices(u8, &[_]u8{ 25, 50, 75, 100, 150, 175, 200, 250 }, &arr);
}

test "bitonicSort: larger array (16 elements)" {
    var arr = [_]i32{ 15, 3, 9, 8, 5, 2, 7, 1, 6, 11, 4, 13, 10, 14, 12, 16 };
    bitonicSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }, &arr);
}

test "bitonicSortAny: arbitrary length (5 elements)" {
    var arr = [_]i32{ 5, 2, 8, 1, 9 };
    try bitonicSortAny(i32, &arr, testing.allocator, {}, comptime asc(i32));
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 5, 8, 9 }, &arr);
}

test "bitonicSortAny: arbitrary length (7 elements)" {
    var arr = [_]i32{ 7, 3, 9, 1, 5, 2, 6 };
    try bitonicSortAny(i32, &arr, testing.allocator, {}, comptime asc(i32));
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 5, 6, 7, 9 }, &arr);
}

test "bitonicSortAny: arbitrary length (10 elements)" {
    var arr = [_]i32{ 10, 5, 8, 2, 9, 1, 7, 3, 6, 4 };
    try bitonicSortAny(i32, &arr, testing.allocator, {}, comptime asc(i32));
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, &arr);
}

test "bitonicSortAny: power of 2 (no padding)" {
    var arr = [_]i32{ 3, 7, 4, 8, 6, 2, 1, 5 };
    try bitonicSortAny(i32, &arr, testing.allocator, {}, comptime asc(i32));
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 }, &arr);
}

test "bitonicSortAny: memory safety (allocator verification)" {
    var arr = [_]i32{ 5, 2, 8, 1, 9, 3, 7, 4, 6 };
    try bitonicSortAny(i32, &arr, testing.allocator, {}, comptime asc(i32));
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &arr);
    // Allocator will detect any leaks automatically
}

test "bitonicSortAny: large arbitrary length (50 elements)" {
    var arr: [50]i32 = undefined;
    // Generate pseudo-random sequence
    for (&arr, 0..) |*item, i| {
        item.* = @intCast((i * 7 + 13) % 50);
    }

    try bitonicSortAny(i32, &arr, testing.allocator, {}, comptime asc(i32));

    // Verify sorted
    for (arr[0 .. arr.len - 1], 1..) |item, i| {
        try testing.expect(item <= arr[i]);
    }
}
