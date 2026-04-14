const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// Catalan Numbers Module
///
/// Catalan numbers C(n) are a sequence of natural numbers with numerous applications in combinatorics:
/// - C(0) = 1, C(1) = 1, C(2) = 2, C(3) = 5, C(4) = 14, C(5) = 42, ...
/// - Formula: C(n) = (2n)! / ((n+1)! * n!) = C(2n, n) / (n+1)
/// - Recurrence: C(n) = Σ(k=0 to n-1) C(k) * C(n-1-k)
/// - Direct formula: C(n) = (1/(n+1)) * C(2n, n)
///
/// Applications:
/// - Number of binary trees with n internal nodes
/// - Number of ways to parenthesize n+1 factors
/// - Number of paths from (0,0) to (n,n) not crossing y=x
/// - Number of ways to triangulate a convex (n+2)-gon
/// - Dyck words of length 2n
/// - Number of monotonic lattice paths
///
/// Reference: OEIS A000108, Stanley's Enumerative Combinatorics Vol 2

/// Compute n-th Catalan number using binomial coefficient formula
/// Formula: C(n) = C(2n, n) / (n+1)
/// Time: O(n)
/// Space: O(1)
pub fn catalan(comptime T: type, n: T) T {
    if (n <= 1) return 1;

    // C(n) = (2n)! / ((n+1)! * n!)
    // Computed using: C(n) = (2n)*(2n-1)*...*(n+1) / (n*(n-1)*...*1) / (n+1)
    // = Π(i=1 to n) (n+i) / i / (n+1)
    var result: T = 1;
    var i: T = 1;
    while (i <= n) : (i += 1) {
        result = result * (n + i) / i;
    }
    return result / (n + 1);
}

/// Compute n-th Catalan number using recurrence relation
/// Recurrence: C(n) = Σ(k=0 to n-1) C(k) * C(n-1-k)
/// Time: O(n²)
/// Space: O(n) for memoization
pub fn catalanRecurrence(comptime T: type, n: T, allocator: Allocator) !T {
    if (n == 0) return 1;

    // Use memoization to avoid recomputation
    var memo = try allocator.alloc(T, n + 1);
    defer allocator.free(memo);

    @memset(memo, 0);
    memo[0] = 1;

    var i: T = 1;
    while (i <= n) : (i += 1) {
        var sum: T = 0;
        var k: T = 0;
        while (k < i) : (k += 1) {
            sum += memo[k] * memo[i - 1 - k];
        }
        memo[i] = sum;
    }

    return memo[n];
}

/// Compute n-th Catalan number using dynamic programming
/// Formula: C(n) = C(n-1) * 2*(2n-1) / (n+1)
/// Time: O(n)
/// Space: O(1)
pub fn catalanDP(comptime T: type, n: T) T {
    if (n == 0) return 1;

    var result: T = 1;
    var i: T = 1;
    while (i <= n) : (i += 1) {
        result = result * 2 * (2 * i - 1) / (i + 1);
    }

    return result;
}

/// Generate the first n Catalan numbers
/// Time: O(n²) using recurrence or O(n) using direct formula
/// Space: O(n)
pub fn generateCatalan(comptime T: type, n: usize, allocator: Allocator) ![]T {
    var result = try allocator.alloc(T, n);
    errdefer allocator.free(result);

    if (n == 0) return result;

    result[0] = 1;

    var i: usize = 1;
    while (i < n) : (i += 1) {
        result[i] = catalan(T, @intCast(i));
    }

    return result;
}

/// Count the number of binary trees with n internal nodes
/// This is exactly the n-th Catalan number
/// Time: O(n)
/// Space: O(1)
pub fn countBinaryTrees(comptime T: type, n: T) T {
    return catalan(T, n);
}

/// Count the number of ways to parenthesize n+1 factors
/// Example: n=2 (3 factors) → 2 ways: ((ab)c) and (a(bc))
/// Time: O(n)
/// Space: O(1)
pub fn countParenthesizations(comptime T: type, n: T) T {
    return catalan(T, n);
}

/// Count the number of Dyck paths from (0,0) to (2n,0)
/// Dyck path: lattice path with up-steps (1,1) and down-steps (1,-1), never going below x-axis
/// Time: O(n)
/// Space: O(1)
pub fn countDyckPaths(comptime T: type, n: T) T {
    return catalan(T, n);
}

/// Count the number of monotonic lattice paths from (0,0) to (n,n)
/// that do not cross the diagonal y=x
/// Time: O(n)
/// Space: O(1)
pub fn countMonotonicPaths(comptime T: type, n: T) T {
    return catalan(T, n);
}

/// Count the number of ways to triangulate a convex polygon with n+2 vertices
/// Example: n=1 (triangle) → 1 way, n=2 (square) → 2 ways, n=3 (pentagon) → 5 ways
/// Time: O(n)
/// Space: O(1)
pub fn countPolygonTriangulations(comptime T: type, n: T) T {
    return catalan(T, n);
}

/// Count the number of ways to arrange n pairs of balanced parentheses
/// Example: n=2 → 2 ways: "(())" and "()()"
/// Time: O(n)
/// Space: O(1)
pub fn countBalancedParentheses(comptime T: type, n: T) T {
    return catalan(T, n);
}

/// Generate all valid parenthesis sequences of length 2n
/// Time: O(C(n) * n) where C(n) is the n-th Catalan number
/// Space: O(C(n) * n)
pub fn generateParentheses(n: usize, allocator: Allocator) ![][]const u8 {
    const count = catalan(usize, n);
    const result = try allocator.alloc([]const u8, count);
    errdefer {
        for (result) |s| allocator.free(s);
        allocator.free(result);
    }

    if (n == 0) {
        // Special case: empty string
        result[0] = try allocator.alloc(u8, 0);
        return result;
    }

    const current = try allocator.alloc(u8, 2 * n);
    defer allocator.free(current);

    var index: usize = 0;
    try generateParenthesesHelper(n, 0, 0, current, allocator, result, &index);

    return result;
}

fn generateParenthesesHelper(
    n: usize,
    open: usize,
    close: usize,
    current: []u8,
    allocator: Allocator,
    result: [][]const u8,
    index: *usize,
) !void {
    if (close == n) {
        const s = try allocator.dupe(u8, current[0..(2 * n)]);
        result[index.*] = s;
        index.* += 1;
        return;
    }

    if (open < n) {
        current[open + close] = '(';
        try generateParenthesesHelper(n, open + 1, close, current, allocator, result, index);
    }

    if (close < open) {
        current[open + close] = ')';
        try generateParenthesesHelper(n, open, close + 1, current, allocator, result, index);
    }
}

// Tests

test "catalan: basic values (OEIS A000108)" {
    // First 10 Catalan numbers: 1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862
    try testing.expectEqual(@as(u64, 1), catalan(u64, 0));
    try testing.expectEqual(@as(u64, 1), catalan(u64, 1));
    try testing.expectEqual(@as(u64, 2), catalan(u64, 2));
    try testing.expectEqual(@as(u64, 5), catalan(u64, 3));
    try testing.expectEqual(@as(u64, 14), catalan(u64, 4));
    try testing.expectEqual(@as(u64, 42), catalan(u64, 5));
    try testing.expectEqual(@as(u64, 132), catalan(u64, 6));
    try testing.expectEqual(@as(u64, 429), catalan(u64, 7));
    try testing.expectEqual(@as(u64, 1430), catalan(u64, 8));
    try testing.expectEqual(@as(u64, 4862), catalan(u64, 9));
}

test "catalan: larger values" {
    try testing.expectEqual(@as(u64, 16796), catalan(u64, 10));
    try testing.expectEqual(@as(u64, 58786), catalan(u64, 11));
    try testing.expectEqual(@as(u64, 208012), catalan(u64, 12));
}

test "catalanRecurrence: matches direct formula" {
    const allocator = testing.allocator;

    var n: u64 = 0;
    while (n <= 10) : (n += 1) {
        const direct = catalan(u64, n);
        const recur = try catalanRecurrence(u64, n, allocator);
        try testing.expectEqual(direct, recur);
    }
}

test "catalanDP: matches direct formula" {
    var n: u64 = 0;
    while (n <= 10) : (n += 1) {
        const direct = catalan(u64, n);
        const dp = catalanDP(u64, n);
        try testing.expectEqual(direct, dp);
    }
}

test "generateCatalan: sequence generation" {
    const allocator = testing.allocator;
    const catalans = try generateCatalan(u64, 10, allocator);
    defer allocator.free(catalans);

    const expected = [_]u64{ 1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862 };

    for (expected, 0..) |exp, i| {
        try testing.expectEqual(exp, catalans[i]);
    }
}

test "generateCatalan: empty sequence" {
    const allocator = testing.allocator;
    const catalans = try generateCatalan(u64, 0, allocator);
    defer allocator.free(catalans);

    try testing.expectEqual(@as(usize, 0), catalans.len);
}

test "countBinaryTrees: basic cases" {
    // C(0) = 1: empty tree
    try testing.expectEqual(@as(u32, 1), countBinaryTrees(u32, 0));
    // C(1) = 1: single node
    try testing.expectEqual(@as(u32, 1), countBinaryTrees(u32, 1));
    // C(2) = 2: two structures
    try testing.expectEqual(@as(u32, 2), countBinaryTrees(u32, 2));
    // C(3) = 5: five structures
    try testing.expectEqual(@as(u32, 5), countBinaryTrees(u32, 3));
}

test "countParenthesizations: basic cases" {
    // n=0 (1 factor): 1 way (no operation)
    try testing.expectEqual(@as(u32, 1), countParenthesizations(u32, 0));
    // n=1 (2 factors): 1 way: (ab)
    try testing.expectEqual(@as(u32, 1), countParenthesizations(u32, 1));
    // n=2 (3 factors): 2 ways: ((ab)c), (a(bc))
    try testing.expectEqual(@as(u32, 2), countParenthesizations(u32, 2));
    // n=3 (4 factors): 5 ways
    try testing.expectEqual(@as(u32, 5), countParenthesizations(u32, 3));
}

test "countDyckPaths: basic cases" {
    // n=0: 1 path (empty)
    try testing.expectEqual(@as(u32, 1), countDyckPaths(u32, 0));
    // n=1: 1 path: /\
    try testing.expectEqual(@as(u32, 1), countDyckPaths(u32, 1));
    // n=2: 2 paths: /\/\, /  \
    try testing.expectEqual(@as(u32, 2), countDyckPaths(u32, 2));
    // n=3: 5 paths
    try testing.expectEqual(@as(u32, 5), countDyckPaths(u32, 3));
}

test "countMonotonicPaths: basic cases" {
    // n=0: 1 path (origin)
    try testing.expectEqual(@as(u32, 1), countMonotonicPaths(u32, 0));
    // n=1: 1 path
    try testing.expectEqual(@as(u32, 1), countMonotonicPaths(u32, 1));
    // n=2: 2 paths
    try testing.expectEqual(@as(u32, 2), countMonotonicPaths(u32, 2));
    // n=3: 5 paths
    try testing.expectEqual(@as(u32, 5), countMonotonicPaths(u32, 3));
}

test "countPolygonTriangulations: basic cases" {
    // n=0 (2 vertices): degenerate
    try testing.expectEqual(@as(u32, 1), countPolygonTriangulations(u32, 0));
    // n=1 (triangle): 1 way (already triangulated)
    try testing.expectEqual(@as(u32, 1), countPolygonTriangulations(u32, 1));
    // n=2 (square): 2 ways (diagonal choice)
    try testing.expectEqual(@as(u32, 2), countPolygonTriangulations(u32, 2));
    // n=3 (pentagon): 5 ways
    try testing.expectEqual(@as(u32, 5), countPolygonTriangulations(u32, 3));
}

test "countBalancedParentheses: basic cases" {
    // n=0: 1 way (empty string)
    try testing.expectEqual(@as(u32, 1), countBalancedParentheses(u32, 0));
    // n=1: 1 way: ()
    try testing.expectEqual(@as(u32, 1), countBalancedParentheses(u32, 1));
    // n=2: 2 ways: (()), ()()
    try testing.expectEqual(@as(u32, 2), countBalancedParentheses(u32, 2));
    // n=3: 5 ways: ((())), (()()), (())(), ()(()), ()()()
    try testing.expectEqual(@as(u32, 5), countBalancedParentheses(u32, 3));
}

test "generateParentheses: n=0" {
    const allocator = testing.allocator;
    const result = try generateParentheses(0, allocator);
    defer {
        for (result) |s| allocator.free(s);
        allocator.free(result);
    }

    try testing.expectEqual(@as(usize, 1), result.len);
    try testing.expectEqual(@as(usize, 0), result[0].len);
}

test "generateParentheses: n=1" {
    const allocator = testing.allocator;
    const result = try generateParentheses(1, allocator);
    defer {
        for (result) |s| allocator.free(s);
        allocator.free(result);
    }

    try testing.expectEqual(@as(usize, 1), result.len);
    try testing.expectEqualStrings("()", result[0]);
}

test "generateParentheses: n=2" {
    const allocator = testing.allocator;
    const result = try generateParentheses(2, allocator);
    defer {
        for (result) |s| allocator.free(s);
        allocator.free(result);
    }

    try testing.expectEqual(@as(usize, 2), result.len);

    // Check that we have both valid sequences
    var found_nested = false;
    var found_sequential = false;

    for (result) |s| {
        if (std.mem.eql(u8, s, "(())")) found_nested = true;
        if (std.mem.eql(u8, s, "()()")) found_sequential = true;
    }

    try testing.expect(found_nested);
    try testing.expect(found_sequential);
}

test "generateParentheses: n=3" {
    const allocator = testing.allocator;
    const result = try generateParentheses(3, allocator);
    defer {
        for (result) |s| allocator.free(s);
        allocator.free(result);
    }

    try testing.expectEqual(@as(usize, 5), result.len);

    // All should be length 6
    for (result) |s| {
        try testing.expectEqual(@as(usize, 6), s.len);
    }
}

test "catalan: type variants (u16, u32, u64)" {
    try testing.expectEqual(@as(u16, 14), catalan(u16, 4));
    try testing.expectEqual(@as(u32, 42), catalan(u32, 5));
    try testing.expectEqual(@as(u64, 132), catalan(u64, 6));
}

test "catalan: consistency across implementations" {
    const allocator = testing.allocator;

    var n: u64 = 0;
    while (n <= 8) : (n += 1) {
        const direct = catalan(u64, n);
        const recur = try catalanRecurrence(u64, n, allocator);
        const dp = catalanDP(u64, n);

        try testing.expectEqual(direct, recur);
        try testing.expectEqual(direct, dp);
    }
}

test "generateParentheses: count matches Catalan number" {
    const allocator = testing.allocator;

    var n: usize = 0;
    while (n <= 4) : (n += 1) {
        const result = try generateParentheses(n, allocator);
        defer {
            for (result) |s| allocator.free(s);
            allocator.free(result);
        }

        const expected_count = catalan(usize, n);
        try testing.expectEqual(expected_count, result.len);
    }
}

test "generateParentheses: memory safety" {
    const allocator = testing.allocator;

    // Test multiple allocations/deallocations
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const result = try generateParentheses(3, allocator);
        for (result) |s| allocator.free(s);
        allocator.free(result);
    }
}
