const std = @import("std");
const testing = std.testing;
const Order = std.math.Order;
const Allocator = std.mem.Allocator;

/// Pancake Sort: Sorting by prefix reversals (flips).
///
/// Algorithm invented by Jacob E. Goodman (1975), later analyzed by Bill Gates and Christos Papadimitriou.
/// Named after flipping pancakes in a stack.
///
/// Key features:
///   - Uses only flip operations (reverse prefix of array)
///   - At most 2n-3 flips worst case (optimal bound unknown, between n and 2n)
///   - Interesting theoretical problem: minimize number of flips
///   - Unstable: does not preserve relative order of equal elements
///   - In-place: O(1) space (no allocation)
///   - Type-generic: works with any comparable type
///
/// Algorithm:
///   1. Find position of maximum element in unsorted portion
///   2. Flip (reverse) prefix to bring max to position 0
///   3. Flip entire unsorted portion to bring max to its correct position
///   4. Reduce unsorted portion size by 1, repeat
///
/// Time: O(n²) comparisons and flips
/// Space: O(1) — in-place sorting, no allocation
///
/// Use cases:
///   - Routing problems (prefix reversal distances)
///   - Bioinformatics (genome rearrangements)
///   - Theoretical computer science (prefix sorting)
///   - Educational purposes (demonstrates pure flip-based sorting)
///
/// References:
///   - Gates & Papadimitriou (1979) "Bounds for sorting by prefix reversal"
///   - Discrete Mathematics 27(1): 47-57

/// Generic pancake sort with custom comparison function.
///
/// Time: O(n²) where n = arr.len
/// Space: O(1) — in-place, no allocation
///
/// Example:
/// ```zig
/// var data = [_]i32{3, 1, 4, 1, 5, 9, 2, 6};
/// pancakeSort(i32, &data, {}, comptime asc(i32));
/// // data is now [1, 1, 2, 3, 4, 5, 6, 9]
/// ```
pub fn pancakeSort(
    comptime T: type,
    arr: []T,
    context: anytype,
    comptime compareFn: fn (@TypeOf(context), T, T) Order,
) void {
    if (arr.len <= 1) return;

    var curr_size = arr.len;
    while (curr_size > 1) : (curr_size -= 1) {
        // Find index of maximum element in arr[0..curr_size]
        const max_idx = findMax(T, arr[0..curr_size], context, compareFn);

        // If max is already at the end of current portion, skip
        if (max_idx == curr_size - 1) continue;

        // Move max to position 0 (if not already there)
        if (max_idx != 0) {
            flip(T, arr, max_idx);
        }

        // Move max from position 0 to its correct position (curr_size - 1)
        flip(T, arr, curr_size - 1);
    }
}

/// Pancake sort in ascending order.
///
/// Time: O(n²)
/// Space: O(1)
pub fn pancakeSortAsc(comptime T: type, arr: []T) void {
    pancakeSort(T, arr, {}, comptime asc(T));
}

/// Pancake sort in descending order.
///
/// Time: O(n²)
/// Space: O(1)
pub fn pancakeSortDesc(comptime T: type, arr: []T) void {
    pancakeSort(T, arr, {}, comptime desc(T));
}

/// Pancake sort using Order-based comparison.
///
/// Time: O(n²)
/// Space: O(1)
pub fn pancakeSortBy(comptime T: type, arr: []T, comptime orderFn: fn (T, T) Order) void {
    pancakeSort(T, arr, {}, struct {
        fn cmp(_: void, a: T, b: T) Order {
            return orderFn(a, b);
        }
    }.cmp);
}

/// Pancake sort with flip counting.
/// Returns the number of flips performed.
///
/// Time: O(n²)
/// Space: O(1)
pub fn pancakeSortCountFlips(comptime T: type, arr: []T) usize {
    if (arr.len <= 1) return 0;

    var flips: usize = 0;
    var curr_size = arr.len;
    while (curr_size > 1) : (curr_size -= 1) {
        const max_idx = findMax(T, arr[0..curr_size], {}, comptime asc(T));
        if (max_idx == curr_size - 1) continue;

        if (max_idx != 0) {
            flip(T, arr, max_idx);
            flips += 1;
        }

        flip(T, arr, curr_size - 1);
        flips += 1;
    }
    return flips;
}

/// Returns the flip sequence needed to sort the array.
/// Each element is the index of the flip (0-indexed).
///
/// Time: O(n²)
/// Space: O(n) for flip sequence
pub fn pancakeSortSequence(comptime T: type, allocator: Allocator, arr: []T) !std.ArrayList(usize) {
    var sequence = std.ArrayList(usize).init(allocator);
    errdefer sequence.deinit();

    if (arr.len <= 1) return sequence;

    var curr_size = arr.len;
    while (curr_size > 1) : (curr_size -= 1) {
        const max_idx = findMax(T, arr[0..curr_size], {}, comptime asc(T));
        if (max_idx == curr_size - 1) continue;

        if (max_idx != 0) {
            flip(T, arr, max_idx);
            try sequence.append(max_idx);
        }

        flip(T, arr, curr_size - 1);
        try sequence.append(curr_size - 1);
    }
    return sequence;
}

/// Minimum number of flips needed to sort (theoretical bound).
/// Returns upper bound: at most 2n-3 flips needed.
///
/// Time: O(1)
/// Space: O(1)
pub fn maxFlipsBound(n: usize) usize {
    if (n <= 1) return 0;
    return 2 * n - 3;
}

// Helper: Find index of maximum element
fn findMax(
    comptime T: type,
    arr: []const T,
    context: anytype,
    comptime compareFn: fn (@TypeOf(context), T, T) Order,
) usize {
    if (arr.len == 0) return 0;

    var max_idx: usize = 0;
    var i: usize = 1;
    while (i < arr.len) : (i += 1) {
        if (compareFn(context, arr[i], arr[max_idx]) == .gt) {
            max_idx = i;
        }
    }
    return max_idx;
}

// Helper: Reverse prefix arr[0..k+1]
fn flip(comptime T: type, arr: []T, k: usize) void {
    if (k >= arr.len) return;
    var left: usize = 0;
    var right = k;
    while (left < right) {
        std.mem.swap(T, &arr[left], &arr[right]);
        left += 1;
        right -= 1;
    }
}

// Comparison helpers
fn asc(comptime T: type) fn (void, T, T) Order {
    return struct {
        fn cmp(_: void, a: T, b: T) Order {
            return std.math.order(a, b);
        }
    }.cmp;
}

fn desc(comptime T: type) fn (void, T, T) Order {
    return struct {
        fn cmp(_: void, a: T, b: T) Order {
            return std.math.order(b, a);
        }
    }.cmp;
}

// Tests
test "pancakeSort: basic ascending" {
    var data = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    pancakeSortAsc(i32, &data);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2, 3, 4, 5, 6, 9 }, &data);
}

test "pancakeSort: basic descending" {
    var data = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    pancakeSortDesc(i32, &data);
    try testing.expectEqualSlices(i32, &[_]i32{ 9, 6, 5, 4, 3, 2, 1, 1 }, &data);
}

test "pancakeSort: empty array" {
    var data = [_]i32{};
    pancakeSortAsc(i32, &data);
    try testing.expectEqualSlices(i32, &[_]i32{}, &data);
}

test "pancakeSort: single element" {
    var data = [_]i32{42};
    pancakeSortAsc(i32, &data);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &data);
}

test "pancakeSort: two elements" {
    var data = [_]i32{ 2, 1 };
    pancakeSortAsc(i32, &data);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2 }, &data);
}

test "pancakeSort: already sorted" {
    var data = [_]i32{ 1, 2, 3, 4, 5 };
    pancakeSortAsc(i32, &data);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &data);
}

test "pancakeSort: reverse sorted" {
    var data = [_]i32{ 5, 4, 3, 2, 1 };
    pancakeSortAsc(i32, &data);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &data);
}

test "pancakeSort: all equal" {
    var data = [_]i32{ 7, 7, 7, 7 };
    pancakeSortAsc(i32, &data);
    try testing.expectEqualSlices(i32, &[_]i32{ 7, 7, 7, 7 }, &data);
}

test "pancakeSort: duplicates" {
    var data = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    pancakeSortAsc(i32, &data);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2, 3, 3, 4, 5, 5, 6, 9 }, &data);
}

test "pancakeSort: negative numbers" {
    var data = [_]i32{ -5, 3, -2, 8, -9, 1 };
    pancakeSortAsc(i32, &data);
    try testing.expectEqualSlices(i32, &[_]i32{ -9, -5, -2, 1, 3, 8 }, &data);
}

test "pancakeSort: floating point (f64)" {
    var data = [_]f64{ 3.14, 2.71, 1.41, 1.73 };
    pancakeSortAsc(f64, &data);
    try testing.expectApproxEqAbs(1.41, data[0], 0.01);
    try testing.expectApproxEqAbs(1.73, data[1], 0.01);
    try testing.expectApproxEqAbs(2.71, data[2], 0.01);
    try testing.expectApproxEqAbs(3.14, data[3], 0.01);
}

test "pancakeSort: custom comparison (struct by age)" {
    const Person = struct {
        name: []const u8,
        age: u32,
    };

    var people = [_]Person{
        .{ .name = "Alice", .age = 30 },
        .{ .name = "Bob", .age = 25 },
        .{ .name = "Charlie", .age = 35 },
    };

    pancakeSort(Person, &people, {}, struct {
        fn cmp(_: void, a: Person, b: Person) Order {
            return std.math.order(a.age, b.age);
        }
    }.cmp);

    try testing.expectEqual(@as(u32, 25), people[0].age);
    try testing.expectEqual(@as(u32, 30), people[1].age);
    try testing.expectEqual(@as(u32, 35), people[2].age);
}

test "pancakeSort: Order-based comparison" {
    var data = [_]i32{ 3, 1, 4, 1, 5 };
    pancakeSortBy(i32, &data, struct {
        fn order(a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    }.order);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 3, 4, 5 }, &data);
}

test "pancakeSort: u8 type" {
    var data = [_]u8{ 200, 100, 255, 0, 128 };
    pancakeSortAsc(u8, &data);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 100, 128, 200, 255 }, &data);
}

test "pancakeSort: flip counting" {
    var data = [_]i32{ 3, 1, 4, 1, 5 };
    const flips = pancakeSortCountFlips(i32, &data);

    // Verify sorted
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 3, 4, 5 }, &data);

    // Verify flip count is within theoretical bound (2n-3 = 2*5-3 = 7)
    try testing.expect(flips <= maxFlipsBound(data.len));
    try testing.expect(flips > 0); // Should have done some flips
}

test "pancakeSort: flip sequence" {
    var data = [_]i32{ 3, 1, 4 };
    var sequence = try pancakeSortSequence(i32, testing.allocator, &data);
    defer sequence.deinit();

    // Verify sorted
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 3, 4 }, &data);

    // Verify sequence length is within bound
    try testing.expect(sequence.items.len <= maxFlipsBound(data.len));
}

test "pancakeSort: max flips bound" {
    try testing.expectEqual(@as(usize, 0), maxFlipsBound(0));
    try testing.expectEqual(@as(usize, 0), maxFlipsBound(1));
    try testing.expectEqual(@as(usize, 1), maxFlipsBound(2));
    try testing.expectEqual(@as(usize, 3), maxFlipsBound(3));
    try testing.expectEqual(@as(usize, 5), maxFlipsBound(4));
    try testing.expectEqual(@as(usize, 7), maxFlipsBound(5));
    try testing.expectEqual(@as(usize, 97), maxFlipsBound(50));
}

test "pancakeSort: large array with allocator" {
    const allocator = testing.allocator;
    const n = 100;

    const data = try allocator.alloc(i32, n);
    defer allocator.free(data);

    // Fill with descending values
    for (data, 0..) |*val, i| {
        val.* = @as(i32, @intCast(n - i));
    }

    pancakeSortAsc(i32, data);

    // Verify sorted
    for (data, 0..) |val, i| {
        try testing.expectEqual(@as(i32, @intCast(i + 1)), val);
    }
}

test "pancakeSort: worst case flip count" {
    // Reverse sorted array is typically worst case for pancake sort
    var data = [_]i32{ 5, 4, 3, 2, 1 };
    const flips = pancakeSortCountFlips(i32, &data);

    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &data);

    // Should be within bound 2n-3 = 2*5-3 = 7
    try testing.expect(flips <= 7);
}

test "pancakeSort: memory safety (no allocations)" {
    // Pancake sort should not allocate for basic operations
    var data = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    pancakeSortAsc(i32, &data);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2, 3, 4, 5, 6, 9 }, &data);
    // No allocator used, so no leaks possible
}
