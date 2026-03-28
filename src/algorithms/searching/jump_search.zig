//! Jump Search Algorithm
//!
//! Jump search is an algorithm for finding a target value in a sorted array.
//! It works by jumping ahead by fixed steps and then performing a linear search
//! in the identified block. The optimal jump size is √n, giving O(√n) complexity.
//!
//! Time Complexity:
//! - Best case: O(1) - target at first position
//! - Average case: O(√n)
//! - Worst case: O(√n)
//!
//! Space Complexity: O(1)
//!
//! Use cases:
//! - When binary search is not practical (e.g., systems where backward jumps are expensive)
//! - When jumping ahead is cheaper than binary search's random access patterns
//! - Good for sorted linked lists (unlike binary search)
//! - Cache-friendly for large arrays (sequential access)

const std = @import("std");
const math = std.math;

/// Jump search for finding a target value in a sorted array.
/// Returns the index of the target if found, null otherwise.
///
/// Time: O(√n) | Space: O(1)
///
/// Example:
/// ```zig
/// const arr = [_]i32{ 1, 3, 5, 7, 9, 11, 13, 15 };
/// const idx = jumpSearch(i32, &arr, 7, {}, std.sort.asc(i32));
/// // idx == 3
/// ```
pub fn jumpSearch(
    comptime T: type,
    arr: []const T,
    target: T,
    context: anytype,
    comptime compareFn: fn (@TypeOf(context), T, T) std.math.Order,
) ?usize {
    const n = arr.len;
    if (n == 0) return null;

    // Optimal jump size is √n
    const jump_size = @as(usize, @intFromFloat(@sqrt(@as(f64, @floatFromInt(n)))));
    if (jump_size == 0) return null;

    var prev: usize = 0;
    var curr: usize = jump_size;

    // Jump ahead until we find a block where target might be
    while (curr < n and compareFn(context, arr[curr], target) == .lt) {
        prev = curr;
        curr += jump_size;
    }

    // Clamp curr to array bounds
    if (curr > n) curr = n;

    // Linear search in the identified block [prev, curr)
    var i = prev;
    while (i < curr) : (i += 1) {
        const order = compareFn(context, arr[i], target);
        if (order == .eq) return i;
        if (order == .gt) return null; // Target not in array
    }

    return null;
}

/// Jump search with custom jump size.
/// Allows caller to specify jump size instead of using optimal √n.
///
/// Time: O(n/jump_size + jump_size) | Space: O(1)
///
/// Example:
/// ```zig
/// const arr = [_]i32{ 1, 3, 5, 7, 9, 11, 13, 15 };
/// const idx = jumpSearchCustom(i32, &arr, 7, 3, {}, std.sort.asc(i32));
/// // idx == 3
/// ```
pub fn jumpSearchCustom(
    comptime T: type,
    arr: []const T,
    target: T,
    jump_size: usize,
    context: anytype,
    comptime compareFn: fn (@TypeOf(context), T, T) std.math.Order,
) ?usize {
    const n = arr.len;
    if (n == 0 or jump_size == 0) return null;

    var prev: usize = 0;
    var curr: usize = jump_size;

    // Jump ahead until we find a block where target might be
    while (curr < n and compareFn(context, arr[curr], target) == .lt) {
        prev = curr;
        curr += jump_size;
    }

    // Clamp curr to array bounds
    if (curr > n) curr = n;

    // Linear search in the identified block [prev, curr)
    var i = prev;
    while (i < curr) : (i += 1) {
        const order = compareFn(context, arr[i], target);
        if (order == .eq) return i;
        if (order == .gt) return null; // Target not in array
    }

    return null;
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "jumpSearch: basic functionality" {
    const arr = [_]i32{ 1, 3, 5, 7, 9, 11, 13, 15, 17, 19 };

    // Found cases
    try testing.expectEqual(@as(?usize, 0), jumpSearch(i32, &arr, 1, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, 3), jumpSearch(i32, &arr, 7, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, 9), jumpSearch(i32, &arr, 19, {}, std.sort.asc(i32)));

    // Not found cases
    try testing.expectEqual(@as(?usize, null), jumpSearch(i32, &arr, 0, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, null), jumpSearch(i32, &arr, 8, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, null), jumpSearch(i32, &arr, 20, {}, std.sort.asc(i32)));
}

test "jumpSearch: edge cases" {
    // Empty array
    const empty = [_]i32{};
    try testing.expectEqual(@as(?usize, null), jumpSearch(i32, &empty, 5, {}, std.sort.asc(i32)));

    // Single element - found
    const single = [_]i32{42};
    try testing.expectEqual(@as(?usize, 0), jumpSearch(i32, &single, 42, {}, std.sort.asc(i32)));

    // Single element - not found
    try testing.expectEqual(@as(?usize, null), jumpSearch(i32, &single, 10, {}, std.sort.asc(i32)));

    // Two elements
    const two = [_]i32{ 1, 2 };
    try testing.expectEqual(@as(?usize, 0), jumpSearch(i32, &two, 1, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, 1), jumpSearch(i32, &two, 2, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, null), jumpSearch(i32, &two, 3, {}, std.sort.asc(i32)));
}

test "jumpSearch: duplicates" {
    const arr = [_]i32{ 1, 2, 2, 2, 3, 4, 5 };

    // Should find first occurrence in the block
    const idx = jumpSearch(i32, &arr, 2, {}, std.sort.asc(i32)).?;
    try testing.expect(idx >= 1 and idx <= 3);
    try testing.expectEqual(@as(i32, 2), arr[idx]);
}

test "jumpSearch: large array" {
    var arr: [10000]i32 = undefined;
    for (&arr, 0..) |*val, i| {
        val.* = @intCast(i * 2); // 0, 2, 4, 6, ..., 19998
    }

    // Test various positions
    try testing.expectEqual(@as(?usize, 0), jumpSearch(i32, &arr, 0, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, 50), jumpSearch(i32, &arr, 100, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, 5000), jumpSearch(i32, &arr, 10000, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, 9999), jumpSearch(i32, &arr, 19998, {}, std.sort.asc(i32)));

    // Not found (odd numbers)
    try testing.expectEqual(@as(?usize, null), jumpSearch(i32, &arr, 1, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, null), jumpSearch(i32, &arr, 101, {}, std.sort.asc(i32)));
}

test "jumpSearch: descending order" {
    const arr = [_]i32{ 20, 18, 15, 12, 10, 8, 5, 3, 1 };

    try testing.expectEqual(@as(?usize, 0), jumpSearch(i32, &arr, 20, {}, std.sort.desc(i32)));
    try testing.expectEqual(@as(?usize, 4), jumpSearch(i32, &arr, 10, {}, std.sort.desc(i32)));
    try testing.expectEqual(@as(?usize, 8), jumpSearch(i32, &arr, 1, {}, std.sort.desc(i32)));

    try testing.expectEqual(@as(?usize, null), jumpSearch(i32, &arr, 0, {}, std.sort.desc(i32)));
    try testing.expectEqual(@as(?usize, null), jumpSearch(i32, &arr, 21, {}, std.sort.desc(i32)));
}

test "jumpSearch: float type" {
    const arr = [_]f64{ -5.5, -2.3, 0.0, 1.1, 3.14, 7.5, 10.0 };

    try testing.expectEqual(@as(?usize, 2), jumpSearch(f64, &arr, 0.0, {}, std.sort.asc(f64)));
    try testing.expectEqual(@as(?usize, 4), jumpSearch(f64, &arr, 3.14, {}, std.sort.asc(f64)));
    try testing.expectEqual(@as(?usize, null), jumpSearch(f64, &arr, 2.0, {}, std.sort.asc(f64)));
}

test "jumpSearchCustom: various jump sizes" {
    const arr = [_]i32{ 1, 3, 5, 7, 9, 11, 13, 15, 17, 19 };

    // Jump size = 1 (equivalent to linear search)
    try testing.expectEqual(@as(?usize, 3), jumpSearchCustom(i32, &arr, 7, 1, {}, std.sort.asc(i32)));

    // Jump size = 2
    try testing.expectEqual(@as(?usize, 3), jumpSearchCustom(i32, &arr, 7, 2, {}, std.sort.asc(i32)));

    // Jump size = 5
    try testing.expectEqual(@as(?usize, 3), jumpSearchCustom(i32, &arr, 7, 5, {}, std.sort.asc(i32)));

    // Jump size = 10 (larger than array, falls back to linear)
    try testing.expectEqual(@as(?usize, 3), jumpSearchCustom(i32, &arr, 7, 10, {}, std.sort.asc(i32)));

    // Jump size = 0 (invalid)
    try testing.expectEqual(@as(?usize, null), jumpSearchCustom(i32, &arr, 7, 0, {}, std.sort.asc(i32)));
}

test "jumpSearchCustom: optimal vs suboptimal" {
    const arr = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const target = 10;

    // All should find the target
    try testing.expectEqual(@as(?usize, 9), jumpSearchCustom(i32, &arr, target, 1, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, 9), jumpSearchCustom(i32, &arr, target, 4, {}, std.sort.asc(i32))); // optimal ≈ √16
    try testing.expectEqual(@as(?usize, 9), jumpSearchCustom(i32, &arr, target, 8, {}, std.sort.asc(i32)));
}

test "jumpSearchCustom: edge cases with custom jump" {
    const arr = [_]i32{ 1, 3, 5, 7, 9 };

    // Jump size larger than array
    try testing.expectEqual(@as(?usize, 0), jumpSearchCustom(i32, &arr, 1, 100, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, null), jumpSearchCustom(i32, &arr, 10, 100, {}, std.sort.asc(i32)));

    // Empty array
    const empty = [_]i32{};
    try testing.expectEqual(@as(?usize, null), jumpSearchCustom(i32, &empty, 5, 2, {}, std.sort.asc(i32)));
}
