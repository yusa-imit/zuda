//! Lexicographic Permutation Operations
//!
//! This module provides efficient algorithms for generating permutations in lexicographic order:
//! - Next permutation: Generate the lexicographically next permutation
//! - Previous permutation: Generate the lexicographically previous permutation
//! - k-th permutation: Directly compute the k-th permutation without enumerating all previous
//! - Rank: Determine the lexicographic position of a given permutation
//!
//! ## Performance Characteristics
//!
//! - **nextPermutation()**: O(n) time, O(1) space - in-place modification
//! - **prevPermutation()**: O(n) time, O(1) space - in-place modification
//! - **kthPermutation()**: O(n²) time, O(n) space - direct computation
//! - **permutationRank()**: O(n²) time, O(n) space - compute rank
//!
//! ## Use Cases
//!
//! - **Algorithm competitions**: Enumerating permutations efficiently
//! - **Combinatorial optimization**: Exploring permutation space
//! - **Testing**: Generating test cases systematically
//! - **Cryptography**: Key generation, permutation ciphers
//! - **Random sampling**: Uniform random permutations
//!
//! ## Example
//!
//! ```zig
//! const std = @import("std");
//! const permutations = @import("zuda").algorithms.combinatorics.permutations;
//!
//! var arr = [_]u8{1, 2, 3};
//!
//! // Generate next permutation
//! const has_next = permutations.nextPermutation(u8, &arr);
//! // arr is now [1, 3, 2]
//!
//! // Generate all permutations in lexicographic order
//! var perm = [_]u8{1, 2, 3};
//! while (true) {
//!     // Process perm
//!     if (!permutations.nextPermutation(u8, &perm)) break;
//! }
//!
//! // Compute the 42nd permutation of [0..9] directly
//! const allocator = std.heap.page_allocator;
//! const kth = try permutations.kthPermutation(u32, allocator, 10, 42);
//! defer allocator.free(kth);
//! ```

const std = @import("std");
const mem = std.mem;
const math = std.math;
const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;
const testing = std.testing;

/// Generate the lexicographically next permutation in-place.
///
/// Returns true if the next permutation exists, false if the input is the last permutation.
/// When false is returned, the array is rotated to the first permutation.
///
/// Algorithm:
/// 1. Find the rightmost pair (i, i+1) where arr[i] < arr[i+1]
/// 2. Find the rightmost element arr[j] > arr[i]
/// 3. Swap arr[i] and arr[j]
/// 4. Reverse the suffix starting at i+1
///
/// Time: O(n) | Space: O(1)
pub fn nextPermutation(comptime T: type, arr: []T) bool {
    if (arr.len <= 1) return false;

    // Step 1: Find the rightmost ascending pair
    var i: isize = @intCast(arr.len - 2);
    while (i >= 0) : (i -= 1) {
        if (arr[@intCast(i)] < arr[@intCast(i + 1)]) break;
    }

    // If no such pair exists, the permutation is the last one
    if (i < 0) {
        mem.reverse(T, arr);
        return false;
    }

    // Step 2: Find the rightmost element greater than arr[i]
    var j: isize = @intCast(arr.len - 1);
    while (j > i) : (j -= 1) {
        if (arr[@intCast(j)] > arr[@intCast(i)]) break;
    }

    // Step 3: Swap arr[i] and arr[j]
    const i_idx: usize = @intCast(i);
    const j_idx: usize = @intCast(j);
    mem.swap(T, &arr[i_idx], &arr[j_idx]);

    // Step 4: Reverse the suffix starting at i+1
    const start: usize = i_idx + 1;
    mem.reverse(T, arr[start..]);

    return true;
}

/// Generate the lexicographically previous permutation in-place.
///
/// Returns true if the previous permutation exists, false if the input is the first permutation.
/// When false is returned, the array is rotated to the last permutation.
///
/// Algorithm: Mirror of nextPermutation
/// 1. Find the rightmost pair (i, i+1) where arr[i] > arr[i+1]
/// 2. Find the rightmost element arr[j] < arr[i]
/// 3. Swap arr[i] and arr[j]
/// 4. Reverse the suffix starting at i+1
///
/// Time: O(n) | Space: O(1)
pub fn prevPermutation(comptime T: type, arr: []T) bool {
    if (arr.len <= 1) return false;

    // Step 1: Find the rightmost descending pair
    var i: isize = @intCast(arr.len - 2);
    while (i >= 0) : (i -= 1) {
        if (arr[@intCast(i)] > arr[@intCast(i + 1)]) break;
    }

    // If no such pair exists, the permutation is the first one
    if (i < 0) {
        mem.reverse(T, arr);
        return false;
    }

    // Step 2: Find the rightmost element less than arr[i]
    var j: isize = @intCast(arr.len - 1);
    while (j > i) : (j -= 1) {
        if (arr[@intCast(j)] < arr[@intCast(i)]) break;
    }

    // Step 3: Swap arr[i] and arr[j]
    const i_idx: usize = @intCast(i);
    const j_idx: usize = @intCast(j);
    mem.swap(T, &arr[i_idx], &arr[j_idx]);

    // Step 4: Reverse the suffix starting at i+1
    const start: usize = i_idx + 1;
    mem.reverse(T, arr[start..]);

    return true;
}

/// Compute the k-th permutation (0-indexed) of the sequence [0, 1, ..., n-1].
///
/// Directly computes the k-th permutation without generating all previous permutations.
///
/// Algorithm:
/// 1. Build factorial table for each position
/// 2. For each position, determine which element to use based on k and factorial
/// 3. Update k by removing the contribution of the chosen element
///
/// Time: O(n²) | Space: O(n)
pub fn kthPermutation(comptime T: type, allocator: Allocator, n: usize, k: usize) ![]T {
    if (n == 0) return try allocator.alloc(T, 0);

    // Build factorial table
    var fact = try allocator.alloc(usize, n);
    defer allocator.free(fact);
    fact[0] = 1;
    for (1..n) |i| {
        fact[i] = fact[i - 1] * i;
    }

    // Total permutations is n!
    const total = fact[n - 1] * n;
    if (k >= total) return error.IndexOutOfBounds;

    // Build result array
    var result = try allocator.alloc(T, n);
    errdefer allocator.free(result);

    // Available elements [0, 1, ..., n-1]
    var available = try ArrayList(T).initCapacity(allocator, n);
    defer available.deinit(allocator);
    for (0..n) |i| {
        try available.append(allocator, @intCast(i));
    }

    var remaining_k = k;
    for (0..n) |i| {
        const pos = n - 1 - i;
        const idx = remaining_k / fact[pos];
        result[i] = available.items[idx];
        _ = available.orderedRemove(idx);
        remaining_k %= fact[pos];
    }

    return result;
}

/// Compute the lexicographic rank (0-indexed) of a permutation.
///
/// Given a permutation, determine its position in the lexicographic ordering.
///
/// Algorithm:
/// 1. For each position, count how many smaller unused elements could have been placed
/// 2. Multiply by the factorial of remaining positions and add to rank
///
/// Time: O(n²) | Space: O(n)
pub fn permutationRank(comptime T: type, allocator: Allocator, perm: []const T) !usize {
    const n = perm.len;
    if (n == 0) return 0;

    // Build factorial table
    var fact = try allocator.alloc(usize, n);
    defer allocator.free(fact);
    fact[0] = 1;
    for (1..n) |i| {
        fact[i] = fact[i - 1] * i;
    }

    // Track which elements have been used
    var used = try allocator.alloc(bool, n);
    defer allocator.free(used);
    @memset(used, false);

    var rank: usize = 0;
    for (0..n) |i| {
        // Count how many smaller unused elements exist
        var smaller_count: usize = 0;
        const val: usize = @intCast(perm[i]);
        for (0..val) |j| {
            if (!used[j]) {
                smaller_count += 1;
            }
        }

        // Add contribution: smaller_count * (n-i-1)!
        const pos = n - i - 1;
        rank += smaller_count * fact[pos];

        // Mark current element as used
        used[val] = true;
    }

    return rank;
}

/// Check if a permutation is valid (contains exactly 0..n-1 each once).
///
/// Time: O(n) | Space: O(n)
pub fn isValidPermutation(comptime T: type, allocator: Allocator, perm: []const T) !bool {
    const n = perm.len;
    var seen = try allocator.alloc(bool, n);
    defer allocator.free(seen);
    @memset(seen, false);

    for (perm) |val| {
        const idx: usize = @intCast(val);
        if (idx >= n) return false;
        if (seen[idx]) return false;
        seen[idx] = true;
    }

    return true;
}

// ============================================================================
// Tests
// ============================================================================

test "nextPermutation: basic progression" {
    var arr = [_]u8{ 1, 2, 3 };

    const has1 = nextPermutation(u8, &arr);
    try testing.expect(has1);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 3, 2 }, &arr);

    const has2 = nextPermutation(u8, &arr);
    try testing.expect(has2);
    try testing.expectEqualSlices(u8, &[_]u8{ 2, 1, 3 }, &arr);

    const has3 = nextPermutation(u8, &arr);
    try testing.expect(has3);
    try testing.expectEqualSlices(u8, &[_]u8{ 2, 3, 1 }, &arr);

    const has4 = nextPermutation(u8, &arr);
    try testing.expect(has4);
    try testing.expectEqualSlices(u8, &[_]u8{ 3, 1, 2 }, &arr);

    const has5 = nextPermutation(u8, &arr);
    try testing.expect(has5);
    try testing.expectEqualSlices(u8, &[_]u8{ 3, 2, 1 }, &arr);

    // Last permutation wraps around to first
    const has6 = nextPermutation(u8, &arr);
    try testing.expect(!has6);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 2, 3 }, &arr);
}

test "nextPermutation: empty and single element" {
    var empty: [0]u8 = .{};
    try testing.expect(!nextPermutation(u8, &empty));

    var single = [_]u8{42};
    try testing.expect(!nextPermutation(u8, &single));
    try testing.expectEqualSlices(u8, &[_]u8{42}, &single);
}

test "nextPermutation: two elements" {
    var arr = [_]u8{ 1, 2 };

    try testing.expect(nextPermutation(u8, &arr));
    try testing.expectEqualSlices(u8, &[_]u8{ 2, 1 }, &arr);

    try testing.expect(!nextPermutation(u8, &arr));
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 2 }, &arr);
}

test "nextPermutation: all permutations of 4 elements" {
    var arr = [_]u8{ 0, 1, 2, 3 };
    var count: usize = 1; // Starting permutation

    while (nextPermutation(u8, &arr)) {
        count += 1;
    }

    // 4! = 24
    try testing.expectEqual(@as(usize, 24), count);
    // Should wrap back to first
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 1, 2, 3 }, &arr);
}

test "prevPermutation: basic progression" {
    var arr = [_]u8{ 3, 2, 1 };

    const has1 = prevPermutation(u8, &arr);
    try testing.expect(has1);
    try testing.expectEqualSlices(u8, &[_]u8{ 3, 1, 2 }, &arr);

    const has2 = prevPermutation(u8, &arr);
    try testing.expect(has2);
    try testing.expectEqualSlices(u8, &[_]u8{ 2, 3, 1 }, &arr);

    const has3 = prevPermutation(u8, &arr);
    try testing.expect(has3);
    try testing.expectEqualSlices(u8, &[_]u8{ 2, 1, 3 }, &arr);

    const has4 = prevPermutation(u8, &arr);
    try testing.expect(has4);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 3, 2 }, &arr);

    const has5 = prevPermutation(u8, &arr);
    try testing.expect(has5);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 2, 3 }, &arr);

    // First permutation wraps around to last
    const has6 = prevPermutation(u8, &arr);
    try testing.expect(!has6);
    try testing.expectEqualSlices(u8, &[_]u8{ 3, 2, 1 }, &arr);
}

test "prevPermutation: empty and single element" {
    var empty: [0]u8 = .{};
    try testing.expect(!prevPermutation(u8, &empty));

    var single = [_]u8{42};
    try testing.expect(!prevPermutation(u8, &single));
    try testing.expectEqualSlices(u8, &single, &[_]u8{42});
}

test "kthPermutation: first few permutations of [0,1,2,3]" {
    const allocator = testing.allocator;

    // 0th permutation: [0,1,2,3]
    const p0 = try kthPermutation(u32, allocator, 4, 0);
    defer allocator.free(p0);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 2, 3 }, p0);

    // 1st permutation: [0,1,3,2]
    const p1 = try kthPermutation(u32, allocator, 4, 1);
    defer allocator.free(p1);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 3, 2 }, p1);

    // 4th permutation: [0,3,1,2]
    const p4 = try kthPermutation(u32, allocator, 4, 4);
    defer allocator.free(p4);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 3, 1, 2 }, p4);

    // 6th permutation: [1,0,2,3]
    const p6 = try kthPermutation(u32, allocator, 4, 6);
    defer allocator.free(p6);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 0, 2, 3 }, p6);

    // 23rd permutation (last): [3,2,1,0]
    const p23 = try kthPermutation(u32, allocator, 4, 23);
    defer allocator.free(p23);
    try testing.expectEqualSlices(u32, &[_]u32{ 3, 2, 1, 0 }, p23);
}

test "kthPermutation: edge cases" {
    const allocator = testing.allocator;

    // n = 0
    const p0 = try kthPermutation(u32, allocator, 0, 0);
    defer allocator.free(p0);
    try testing.expectEqual(@as(usize, 0), p0.len);

    // n = 1
    const p1 = try kthPermutation(u32, allocator, 1, 0);
    defer allocator.free(p1);
    try testing.expectEqualSlices(u32, &[_]u32{0}, p1);

    // Out of bounds
    try testing.expectError(error.IndexOutOfBounds, kthPermutation(u32, allocator, 3, 6)); // 3! = 6
}

test "kthPermutation: all permutations of [0,1,2]" {
    const allocator = testing.allocator;

    const expected = [_][3]u32{
        [_]u32{ 0, 1, 2 },
        [_]u32{ 0, 2, 1 },
        [_]u32{ 1, 0, 2 },
        [_]u32{ 1, 2, 0 },
        [_]u32{ 2, 0, 1 },
        [_]u32{ 2, 1, 0 },
    };

    for (0..6) |k| {
        const perm = try kthPermutation(u32, allocator, 3, k);
        defer allocator.free(perm);
        try testing.expectEqualSlices(u32, &expected[k], perm);
    }
}

test "permutationRank: basic permutations" {
    const allocator = testing.allocator;

    // [0,1,2] -> rank 0
    const r0 = try permutationRank(u32, allocator, &[_]u32{ 0, 1, 2 });
    try testing.expectEqual(@as(usize, 0), r0);

    // [0,2,1] -> rank 1
    const r1 = try permutationRank(u32, allocator, &[_]u32{ 0, 2, 1 });
    try testing.expectEqual(@as(usize, 1), r1);

    // [2,1,0] -> rank 5 (last)
    const r5 = try permutationRank(u32, allocator, &[_]u32{ 2, 1, 0 });
    try testing.expectEqual(@as(usize, 5), r5);
}

test "permutationRank: roundtrip with kthPermutation" {
    const allocator = testing.allocator;
    const n = 4;
    const total: usize = 24; // 4!

    for (0..total) |k| {
        const perm = try kthPermutation(u32, allocator, n, k);
        defer allocator.free(perm);

        const rank = try permutationRank(u32, allocator, perm);
        try testing.expectEqual(k, rank);
    }
}

test "isValidPermutation: valid permutations" {
    const allocator = testing.allocator;

    try testing.expect(try isValidPermutation(u32, allocator, &[_]u32{ 0, 1, 2 }));
    try testing.expect(try isValidPermutation(u32, allocator, &[_]u32{ 2, 0, 1 }));
    try testing.expect(try isValidPermutation(u32, allocator, &[_]u32{0}));
}

test "isValidPermutation: invalid permutations" {
    const allocator = testing.allocator;

    // Duplicate
    try testing.expect(!try isValidPermutation(u32, allocator, &[_]u32{ 0, 1, 1 }));

    // Out of range
    try testing.expect(!try isValidPermutation(u32, allocator, &[_]u32{ 0, 1, 3 }));

    // Missing element
    try testing.expect(!try isValidPermutation(u32, allocator, &[_]u32{ 1, 2, 3 }));
}

test "nextPermutation: type variants" {
    // u16
    var arr_u16 = [_]u16{ 1, 2, 3 };
    try testing.expect(nextPermutation(u16, &arr_u16));
    try testing.expectEqualSlices(u16, &[_]u16{ 1, 3, 2 }, &arr_u16);

    // u32
    var arr_u32 = [_]u32{ 5, 6, 7 };
    try testing.expect(nextPermutation(u32, &arr_u32));
    try testing.expectEqualSlices(u32, &[_]u32{ 5, 7, 6 }, &arr_u32);

    // i32
    var arr_i32 = [_]i32{ -1, 0, 1 };
    try testing.expect(nextPermutation(i32, &arr_i32));
    try testing.expectEqualSlices(i32, &[_]i32{ -1, 1, 0 }, &arr_i32);
}

test "prevPermutation: type variants" {
    // u16
    var arr_u16 = [_]u16{ 3, 2, 1 };
    try testing.expect(prevPermutation(u16, &arr_u16));
    try testing.expectEqualSlices(u16, &[_]u16{ 3, 1, 2 }, &arr_u16);

    // u64
    var arr_u64 = [_]u64{ 10, 9, 8 };
    try testing.expect(prevPermutation(u64, &arr_u64));
    try testing.expectEqualSlices(u64, &[_]u64{ 10, 8, 9 }, &arr_u64);
}

test "kthPermutation: larger n" {
    const allocator = testing.allocator;

    // 5th element: many permutations (5! = 120)
    const p0 = try kthPermutation(u32, allocator, 5, 0);
    defer allocator.free(p0);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 2, 3, 4 }, p0);

    const p119 = try kthPermutation(u32, allocator, 5, 119);
    defer allocator.free(p119);
    try testing.expectEqualSlices(u32, &[_]u32{ 4, 3, 2, 1, 0 }, p119);
}

test "nextPermutation with prevPermutation: bidirectional" {
    var arr = [_]u8{ 1, 2, 3 };

    // Forward
    try testing.expect(nextPermutation(u8, &arr));
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 3, 2 }, &arr);

    // Backward
    try testing.expect(prevPermutation(u8, &arr));
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 2, 3 }, &arr);

    // Backward from first
    try testing.expect(!prevPermutation(u8, &arr));
    try testing.expectEqualSlices(u8, &[_]u8{ 3, 2, 1 }, &arr);

    // Forward from last
    try testing.expect(!nextPermutation(u8, &arr));
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 2, 3 }, &arr);
}

test "permutationRank: all permutations of [0,1,2,3]" {
    const allocator = testing.allocator;
    var arr = [_]u32{ 0, 1, 2, 3 };
    var expected_rank: usize = 0;

    while (true) {
        const rank = try permutationRank(u32, allocator, &arr);
        try testing.expectEqual(expected_rank, rank);

        if (!nextPermutation(u32, &arr)) break;
        expected_rank += 1;
    }

    try testing.expectEqual(@as(usize, 23), expected_rank); // Last rank
}

test "memory safety: kthPermutation allocations" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const perm = try kthPermutation(u32, allocator, 5, 42);
        allocator.free(perm);
    }
}

test "memory safety: permutationRank allocations" {
    const allocator = testing.allocator;
    const perm = [_]u32{ 2, 1, 0, 3 };

    for (0..10) |_| {
        _ = try permutationRank(u32, allocator, &perm);
    }
}
