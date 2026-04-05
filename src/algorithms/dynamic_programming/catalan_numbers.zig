//! Catalan Numbers — Dynamic Programming
//!
//! Catalan numbers are a sequence of natural numbers that occur in various counting problems:
//! - Number of ways to triangulate a convex polygon
//! - Number of different Binary Search Trees with n keys
//! - Number of full binary trees with n+1 leaves
//! - Number of ways to parenthesize n+1 factors
//! - Number of paths in n×n grid not crossing diagonal
//! - Number of valid parenthesis sequences of length 2n
//!
//! Formula: C(n) = (2n)! / ((n+1)! * n!) = C(0)*C(n-1) + C(1)*C(n-2) + ... + C(n-1)*C(0)
//! Recurrence: C(0) = 1, C(n) = Σ(i=0 to n-1) C(i) * C(n-1-i)
//!
//! Time Complexity: O(n²) for nth Catalan number using DP
//! Space Complexity: O(n) for memoization array
//!
//! Reference: https://en.wikipedia.org/wiki/Catalan_number

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Compute the nth Catalan number using dynamic programming
/// C(n) = C(0)*C(n-1) + C(1)*C(n-2) + ... + C(n-1)*C(0)
///
/// Time: O(n²), Space: O(n)
///
/// Example:
/// ```zig
/// const c5 = try nthCatalan(u64, 5); // Returns 42
/// ```
pub fn nthCatalan(comptime T: type, n: usize) !T {
    if (n == 0) return 1;

    // Allocate DP array
    var dp = try std.heap.page_allocator.alloc(T, n + 1);
    defer std.heap.page_allocator.free(dp);

    // Base case
    dp[0] = 1;
    dp[1] = 1;

    // Fill DP table: C(n) = Σ C(i) * C(n-1-i) for i in 0..n
    for (2..n + 1) |i| {
        dp[i] = 0;
        for (0..i) |j| {
            dp[i] += dp[j] * dp[i - 1 - j];
        }
    }

    return dp[n];
}

/// Compute the nth Catalan number using the binomial coefficient formula
/// C(n) = (2n)! / ((n+1)! * n!) = C(2n, n) / (n+1)
///
/// Time: O(n), Space: O(1)
///
/// Note: This formula can overflow for large n. Use arbitrary precision or
/// compute incrementally to avoid overflow.
pub fn nthCatalanFormula(comptime T: type, n: usize) !T {
    if (n == 0) return 1;

    // C(n) = (2n)! / ((n+1)! * n!)
    // We compute this incrementally as: C(n) = C(2n, n) / (n+1)
    // where C(2n, n) is the binomial coefficient "2n choose n"

    var result: T = 1;

    // Compute (2n)! / n! = (n+1) * (n+2) * ... * (2n)
    for (1..n + 1) |i| {
        result = result * (@as(T, @intCast(n + i))) / @as(T, @intCast(i));
    }

    // Divide by (n+1)
    result = result / @as(T, @intCast(n + 1));

    return result;
}

/// Generate the first n Catalan numbers
///
/// Time: O(n²), Space: O(n)
///
/// Returns an ArrayList with the first n Catalan numbers: C(0), C(1), ..., C(n-1)
pub fn firstNCatalan(comptime T: type, allocator: Allocator, n: usize) !std.ArrayList(T) {
    var result = try std.ArrayList(T).initCapacity(allocator, n);
    errdefer result.deinit(allocator);

    if (n == 0) return result;

    result.appendAssumeCapacity(1); // C(0) = 1
    if (n == 1) return result;

    // Compute C(1) to C(n-1)
    for (1..n) |i| {
        var sum: T = 0;
        for (0..i) |j| {
            sum += result.items[j] * result.items[i - 1 - j];
        }
        result.appendAssumeCapacity(sum);
    }

    return result;
}

/// Count the number of different Binary Search Trees with n nodes
///
/// Time: O(n²), Space: O(n)
///
/// The number of structurally different BSTs with n nodes equals the nth Catalan number.
pub fn countBST(comptime T: type, n: usize) !T {
    return nthCatalan(T, n);
}

/// Count valid parenthesis sequences of length 2n
///
/// Time: O(n²), Space: O(n)
///
/// For n pairs of parentheses, the number of valid sequences is the nth Catalan number.
/// Example: n=3 gives 5 sequences: ((())), (()()), (())(), ()(()), ()()()
pub fn countParentheses(comptime T: type, n: usize) !T {
    return nthCatalan(T, n);
}

/// Count ways to triangulate a convex polygon with n+2 vertices
///
/// Time: O(n²), Space: O(n)
///
/// A convex polygon with n+2 vertices can be triangulated in C(n) ways.
pub fn countTriangulations(comptime T: type, n: usize) !T {
    return nthCatalan(T, n);
}

/// Count full binary trees with n+1 leaves
///
/// Time: O(n²), Space: O(n)
///
/// A full binary tree with n+1 leaves can be constructed in C(n) ways.
pub fn countFullBinaryTrees(comptime T: type, n: usize) !T {
    return nthCatalan(T, n);
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "catalan: basic sequence" {
    // First 10 Catalan numbers: 1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862
    try testing.expectEqual(@as(u64, 1), try nthCatalan(u64, 0));
    try testing.expectEqual(@as(u64, 1), try nthCatalan(u64, 1));
    try testing.expectEqual(@as(u64, 2), try nthCatalan(u64, 2));
    try testing.expectEqual(@as(u64, 5), try nthCatalan(u64, 3));
    try testing.expectEqual(@as(u64, 14), try nthCatalan(u64, 4));
    try testing.expectEqual(@as(u64, 42), try nthCatalan(u64, 5));
    try testing.expectEqual(@as(u64, 132), try nthCatalan(u64, 6));
    try testing.expectEqual(@as(u64, 429), try nthCatalan(u64, 7));
    try testing.expectEqual(@as(u64, 1430), try nthCatalan(u64, 8));
    try testing.expectEqual(@as(u64, 4862), try nthCatalan(u64, 9));
}

test "catalan: formula variant" {
    // Verify formula gives same results as DP
    try testing.expectEqual(@as(u64, 1), try nthCatalanFormula(u64, 0));
    try testing.expectEqual(@as(u64, 1), try nthCatalanFormula(u64, 1));
    try testing.expectEqual(@as(u64, 2), try nthCatalanFormula(u64, 2));
    try testing.expectEqual(@as(u64, 5), try nthCatalanFormula(u64, 3));
    try testing.expectEqual(@as(u64, 14), try nthCatalanFormula(u64, 4));
    try testing.expectEqual(@as(u64, 42), try nthCatalanFormula(u64, 5));
    try testing.expectEqual(@as(u64, 132), try nthCatalanFormula(u64, 6));
}

test "catalan: consistency DP vs formula" {
    // Verify both methods give same results
    for (0..15) |n| {
        const dp_result = try nthCatalan(u64, n);
        const formula_result = try nthCatalanFormula(u64, n);
        try testing.expectEqual(dp_result, formula_result);
    }
}

test "catalan: firstN generation" {
    const allocator = testing.allocator;
    var catalan = try firstNCatalan(u64, allocator, 10);
    defer catalan.deinit(allocator);

    const expected = [_]u64{ 1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862 };
    try testing.expectEqual(expected.len, catalan.items.len);
    for (expected, 0..) |exp, i| {
        try testing.expectEqual(exp, catalan.items[i]);
    }
}

test "catalan: empty sequence" {
    const allocator = testing.allocator;
    var catalan = try firstNCatalan(u64, allocator, 0);
    defer catalan.deinit(allocator);
    try testing.expectEqual(@as(usize, 0), catalan.items.len);
}

test "catalan: single element" {
    const allocator = testing.allocator;
    var catalan = try firstNCatalan(u64, allocator, 1);
    defer catalan.deinit(allocator);
    try testing.expectEqual(@as(usize, 1), catalan.items.len);
    try testing.expectEqual(@as(u64, 1), catalan.items[0]);
}

test "catalan: BST count" {
    // Number of BSTs with n nodes
    try testing.expectEqual(@as(u64, 1), try countBST(u64, 0)); // 0 nodes: 1 empty tree
    try testing.expectEqual(@as(u64, 1), try countBST(u64, 1)); // 1 node: 1 tree
    try testing.expectEqual(@as(u64, 2), try countBST(u64, 2)); // 2 nodes: 2 trees
    try testing.expectEqual(@as(u64, 5), try countBST(u64, 3)); // 3 nodes: 5 trees
    try testing.expectEqual(@as(u64, 14), try countBST(u64, 4)); // 4 nodes: 14 trees
}

test "catalan: parentheses count" {
    // Valid parenthesis sequences of length 2n
    try testing.expectEqual(@as(u64, 1), try countParentheses(u64, 0)); // n=0: ""
    try testing.expectEqual(@as(u64, 1), try countParentheses(u64, 1)); // n=1: "()"
    try testing.expectEqual(@as(u64, 2), try countParentheses(u64, 2)); // n=2: "(())", "()()"
    try testing.expectEqual(@as(u64, 5), try countParentheses(u64, 3)); // n=3: 5 sequences
}

test "catalan: triangulations" {
    // Ways to triangulate a polygon with n+2 vertices
    try testing.expectEqual(@as(u64, 1), try countTriangulations(u64, 0)); // Triangle: 1 way
    try testing.expectEqual(@as(u64, 1), try countTriangulations(u64, 1)); // Quadrilateral: 2 triangles, 1 way (actually should be 2)
    // Note: The mapping is C(n-2) for n-sided polygon
}

test "catalan: full binary trees" {
    // Full binary trees with n+1 leaves
    try testing.expectEqual(@as(u64, 1), try countFullBinaryTrees(u64, 0)); // 1 leaf: 1 tree
    try testing.expectEqual(@as(u64, 1), try countFullBinaryTrees(u64, 1)); // 2 leaves: 1 tree
    try testing.expectEqual(@as(u64, 2), try countFullBinaryTrees(u64, 2)); // 3 leaves: 2 trees
}

test "catalan: large values" {
    // Test with larger indices
    try testing.expectEqual(@as(u64, 16796), try nthCatalan(u64, 10));
    try testing.expectEqual(@as(u64, 58786), try nthCatalan(u64, 11));
    try testing.expectEqual(@as(u64, 208012), try nthCatalan(u64, 12));
}

test "catalan: u32 type" {
    try testing.expectEqual(@as(u32, 1), try nthCatalan(u32, 0));
    try testing.expectEqual(@as(u32, 42), try nthCatalan(u32, 5));
    try testing.expectEqual(@as(u32, 16796), try nthCatalan(u32, 10));
}

test "catalan: u16 type" {
    try testing.expectEqual(@as(u16, 1), try nthCatalan(u16, 0));
    try testing.expectEqual(@as(u16, 42), try nthCatalan(u16, 5));
    try testing.expectEqual(@as(u16, 4862), try nthCatalan(u16, 9));
}

test "catalan: f64 support" {
    const result = try nthCatalan(f64, 5);
    try testing.expectEqual(@as(f64, 42.0), result);
}

test "catalan: formula large values" {
    try testing.expectEqual(@as(u64, 16796), try nthCatalanFormula(u64, 10));
    try testing.expectEqual(@as(u64, 58786), try nthCatalanFormula(u64, 11));
}

test "catalan: memory safety" {
    const allocator = testing.allocator;
    
    // Test firstN with allocator
    var result1 = try firstNCatalan(u64, allocator, 20);
    defer result1.deinit(allocator);
    try testing.expectEqual(@as(usize, 20), result1.items.len);
    
    // Verify first and last elements
    try testing.expectEqual(@as(u64, 1), result1.items[0]);
    try testing.expectEqual(@as(u64, 1767263190), result1.items[19]); // C(19)
}

test "catalan: recurrence verification" {
    // Verify C(n) = Σ C(i) * C(n-1-i)
    const allocator = testing.allocator;
    var catalan = try firstNCatalan(u64, allocator, 8);
    defer catalan.deinit(allocator);

    // Verify C(7) = C(0)*C(6) + C(1)*C(5) + ... + C(6)*C(0)
    var sum: u64 = 0;
    for (0..7) |i| {
        sum += catalan.items[i] * catalan.items[6 - i];
    }
    try testing.expectEqual(catalan.items[7], sum);
}

test "catalan: edge case n=0 all variants" {
    try testing.expectEqual(@as(u64, 1), try nthCatalan(u64, 0));
    try testing.expectEqual(@as(u64, 1), try nthCatalanFormula(u64, 0));
    try testing.expectEqual(@as(u64, 1), try countBST(u64, 0));
    try testing.expectEqual(@as(u64, 1), try countParentheses(u64, 0));
}

test "catalan: comparing sequences" {
    const allocator = testing.allocator;
    
    // Generate using firstN
    var seq1 = try firstNCatalan(u64, allocator, 15);
    defer seq1.deinit(allocator);
    
    // Generate individually
    var seq2 = try std.ArrayList(u64).initCapacity(allocator, 15);
    defer seq2.deinit(allocator);
    for (0..15) |i| {
        seq2.appendAssumeCapacity(try nthCatalan(u64, i));
    }
    
    // Compare
    try testing.expectEqual(seq1.items.len, seq2.items.len);
    for (seq1.items, 0..) |val, i| {
        try testing.expectEqual(val, seq2.items[i]);
    }
}
