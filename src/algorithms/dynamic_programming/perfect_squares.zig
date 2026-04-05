//! Perfect Squares - Find minimum number of perfect square numbers that sum to n
//!
//! This module implements dynamic programming solutions to find the minimum number
//! of perfect square numbers (1, 4, 9, 16, 25, ...) that sum to a given positive integer.
//!
//! Classic DP problem with applications in number theory and optimization.
//!
//! Reference: LeetCode #279 - Perfect Squares

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// Find the minimum number of perfect square numbers that sum to n.
///
/// Uses dynamic programming with O(n × √n) time and O(n) space.
///
/// Time: O(n × √n) where n is the input number
/// Space: O(n) for the DP array
///
/// Example:
///   n = 12 → 3 (4 + 4 + 4)
///   n = 13 → 2 (4 + 9)
pub fn numSquares(comptime T: type, n: T) !T {
    if (@typeInfo(T) != .Int or @typeInfo(T).Int.signedness != .unsigned) {
        @compileError("numSquares requires unsigned integer type");
    }

    if (n == 0) return 0;
    if (n == 1) return 1;

    // Check if n is a perfect square
    const sqrt_n = intSqrt(T, n);
    if (sqrt_n * sqrt_n == n) return 1;

    // DP array: dp[i] = minimum count for sum i
    var dp = try std.ArrayList(T).initCapacity(std.heap.page_allocator, n + 1);
    defer dp.deinit();

    // Initialize with max values
    var i: T = 0;
    while (i <= n) : (i += 1) {
        dp.appendAssumeCapacity(std.math.maxInt(T));
    }
    dp.items[0] = 0;

    // Fill DP table
    i = 1;
    while (i <= n) : (i += 1) {
        var j: T = 1;
        while (j * j <= i) : (j += 1) {
            const square = j * j;
            if (dp.items[i - square] != std.math.maxInt(T)) {
                dp.items[i] = @min(dp.items[i], dp.items[i - square] + 1);
            }
        }
    }

    return dp.items[n];
}

/// Find the minimum number of perfect square numbers that sum to n.
/// Returns the actual perfect square numbers used.
///
/// Time: O(n × √n) for DP + O(result_count) for backtracking
/// Space: O(n) for DP array + O(result_count) for result
pub fn numSquaresWithPath(comptime T: type, allocator: Allocator, n: T) !struct { count: T, squares: []T } {
    if (@typeInfo(T) != .Int or @typeInfo(T).Int.signedness != .unsigned) {
        @compileError("numSquaresWithPath requires unsigned integer type");
    }

    if (n == 0) {
        return .{ .count = 0, .squares = try allocator.alloc(T, 0) };
    }

    if (n == 1) {
        const squares = try allocator.alloc(T, 1);
        squares[0] = 1;
        return .{ .count = 1, .squares = squares };
    }

    // Check if n is a perfect square
    const sqrt_n = intSqrt(T, n);
    if (sqrt_n * sqrt_n == n) {
        const squares = try allocator.alloc(T, 1);
        squares[0] = n;
        return .{ .count = 1, .squares = squares };
    }

    // DP array: dp[i] = minimum count for sum i
    var dp = try std.ArrayList(T).initCapacity(allocator, n + 1);
    defer dp.deinit();

    // Parent array: parent[i] = the perfect square used to reach i optimally
    var parent = try std.ArrayList(T).initCapacity(allocator, n + 1);
    defer parent.deinit();

    // Initialize
    var i: T = 0;
    while (i <= n) : (i += 1) {
        dp.appendAssumeCapacity(std.math.maxInt(T));
        parent.appendAssumeCapacity(0);
    }
    dp.items[0] = 0;

    // Fill DP table
    i = 1;
    while (i <= n) : (i += 1) {
        var j: T = 1;
        while (j * j <= i) : (j += 1) {
            const square = j * j;
            if (dp.items[i - square] != std.math.maxInt(T)) {
                const new_count = dp.items[i - square] + 1;
                if (new_count < dp.items[i]) {
                    dp.items[i] = new_count;
                    parent.items[i] = square;
                }
            }
        }
    }

    const count = dp.items[n];

    // Backtrack to find the squares
    var squares = try std.ArrayList(T).initCapacity(allocator, count);
    var current = n;
    while (current > 0) {
        const square_used = parent.items[current];
        squares.appendAssumeCapacity(square_used);
        current -= square_used;
    }

    return .{ .count = count, .squares = try squares.toOwnedSlice() };
}

/// Check if n can be expressed as sum of exactly k perfect squares.
///
/// Time: O(n × k × √n)
/// Space: O(n × k)
pub fn canSumSquares(comptime T: type, allocator: Allocator, n: T, k: T) !bool {
    if (@typeInfo(T) != .Int or @typeInfo(T).Int.signedness != .unsigned) {
        @compileError("canSumSquares requires unsigned integer type");
    }

    if (k == 0) return n == 0;
    if (n == 0) return false;

    // dp[i][j] = can we make sum i using exactly j perfect squares?
    var dp = try allocator.alloc([]bool, n + 1);
    defer {
        for (dp) |row| {
            allocator.free(row);
        }
        allocator.free(dp);
    }

    for (dp) |*row| {
        row.* = try allocator.alloc(bool, k + 1);
        @memset(row.*, false);
    }

    dp[0][0] = true;

    var i: T = 1;
    while (i <= n) : (i += 1) {
        var j: T = 1;
        while (j <= k) : (j += 1) {
            var s: T = 1;
            while (s * s <= i) : (s += 1) {
                const square = s * s;
                if (i >= square and dp[i - square][j - 1]) {
                    dp[i][j] = true;
                    break;
                }
            }
        }
    }

    return dp[n][k];
}

/// Find minimum number of perfect squares using BFS (Lagrange's four-square theorem approach).
/// This can be faster for some inputs.
///
/// Time: O(n × √n) worst case, often faster in practice
/// Space: O(n) for visited set
pub fn numSquaresBFS(comptime T: type, allocator: Allocator, n: T) !T {
    if (@typeInfo(T) != .Int or @typeInfo(T).Int.signedness != .unsigned) {
        @compileError("numSquaresBFS requires unsigned integer type");
    }

    if (n == 0) return 0;
    if (n == 1) return 1;

    // Check if n is a perfect square
    const sqrt_n = intSqrt(T, n);
    if (sqrt_n * sqrt_n == n) return 1;

    var visited = std.AutoHashMap(T, void).init(allocator);
    defer visited.deinit();

    var queue = std.ArrayList(T).init(allocator);
    defer queue.deinit();

    try queue.append(n);
    try visited.put(n, {});

    var level: T = 0;

    while (queue.items.len > 0) {
        level += 1;
        const size = queue.items.len;

        var i: usize = 0;
        while (i < size) : (i += 1) {
            const num = queue.orderedRemove(0);

            var j: T = 1;
            while (j * j <= num) : (j += 1) {
                const square = j * j;
                const next = num - square;

                if (next == 0) return level;

                if (!visited.contains(next)) {
                    try queue.append(next);
                    try visited.put(next, {});
                }
            }
        }
    }

    return level;
}

/// Helper: Integer square root using binary search.
fn intSqrt(comptime T: type, n: T) T {
    if (n == 0) return 0;
    if (n == 1) return 1;

    var left: T = 1;
    var right: T = n;
    var result: T = 0;

    while (left <= right) {
        const mid = left + (right - left) / 2;
        const mid_squared = mid * mid;

        if (mid_squared == n) {
            return mid;
        } else if (mid_squared < n) {
            result = mid;
            left = mid + 1;
        } else {
            if (mid == 0) break;
            right = mid - 1;
        }
    }

    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "numSquares - basic cases" {
    try testing.expectEqual(@as(u32, 0), try numSquares(u32, 0));
    try testing.expectEqual(@as(u32, 1), try numSquares(u32, 1));
    try testing.expectEqual(@as(u32, 2), try numSquares(u32, 2)); // 1 + 1
    try testing.expectEqual(@as(u32, 3), try numSquares(u32, 3)); // 1 + 1 + 1
    try testing.expectEqual(@as(u32, 1), try numSquares(u32, 4)); // 4
}

test "numSquares - classic examples" {
    try testing.expectEqual(@as(u32, 3), try numSquares(u32, 12)); // 4 + 4 + 4
    try testing.expectEqual(@as(u32, 2), try numSquares(u32, 13)); // 4 + 9
    try testing.expectEqual(@as(u32, 1), try numSquares(u32, 16)); // 16
    try testing.expectEqual(@as(u32, 2), try numSquares(u32, 17)); // 16 + 1
}

test "numSquares - perfect squares" {
    try testing.expectEqual(@as(u32, 1), try numSquares(u32, 9)); // 9
    try testing.expectEqual(@as(u32, 1), try numSquares(u32, 25)); // 25
    try testing.expectEqual(@as(u32, 1), try numSquares(u32, 100)); // 100
}

test "numSquares - larger numbers" {
    try testing.expectEqual(@as(u32, 2), try numSquares(u32, 18)); // 9 + 9
    try testing.expectEqual(@as(u32, 3), try numSquares(u32, 19)); // 9 + 9 + 1
    try testing.expectEqual(@as(u32, 2), try numSquares(u32, 20)); // 16 + 4
}

test "numSquaresWithPath - basic" {
    const allocator = testing.allocator;

    const result0 = try numSquaresWithPath(u32, allocator, 0);
    defer allocator.free(result0.squares);
    try testing.expectEqual(@as(u32, 0), result0.count);
    try testing.expectEqual(@as(usize, 0), result0.squares.len);

    const result1 = try numSquaresWithPath(u32, allocator, 1);
    defer allocator.free(result1.squares);
    try testing.expectEqual(@as(u32, 1), result1.count);
    try testing.expectEqual(@as(usize, 1), result1.squares.len);
    try testing.expectEqual(@as(u32, 1), result1.squares[0]);
}

test "numSquaresWithPath - perfect square" {
    const allocator = testing.allocator;

    const result = try numSquaresWithPath(u32, allocator, 16);
    defer allocator.free(result.squares);
    try testing.expectEqual(@as(u32, 1), result.count);
    try testing.expectEqual(@as(usize, 1), result.squares.len);
    try testing.expectEqual(@as(u32, 16), result.squares[0]);
}

test "numSquaresWithPath - decomposition validation" {
    const allocator = testing.allocator;

    const result = try numSquaresWithPath(u32, allocator, 12);
    defer allocator.free(result.squares);
    try testing.expectEqual(@as(u32, 3), result.count);
    try testing.expectEqual(@as(usize, 3), result.squares.len);

    // Verify all are perfect squares and sum to 12
    var sum: u32 = 0;
    for (result.squares) |sq| {
        const sqrt = intSqrt(u32, sq);
        try testing.expectEqual(sq, sqrt * sqrt);
        sum += sq;
    }
    try testing.expectEqual(@as(u32, 12), sum);
}

test "numSquaresWithPath - 13 decomposition" {
    const allocator = testing.allocator;

    const result = try numSquaresWithPath(u32, allocator, 13);
    defer allocator.free(result.squares);
    try testing.expectEqual(@as(u32, 2), result.count);
    try testing.expectEqual(@as(usize, 2), result.squares.len);

    var sum: u32 = 0;
    for (result.squares) |sq| {
        const sqrt = intSqrt(u32, sq);
        try testing.expectEqual(sq, sqrt * sqrt);
        sum += sq;
    }
    try testing.expectEqual(@as(u32, 13), sum);
}

test "canSumSquares - basic" {
    const allocator = testing.allocator;

    try testing.expect(try canSumSquares(u32, allocator, 0, 0));
    try testing.expect(!try canSumSquares(u32, allocator, 0, 1));
    try testing.expect(try canSumSquares(u32, allocator, 1, 1));
    try testing.expect(!try canSumSquares(u32, allocator, 1, 2));
}

test "canSumSquares - exact count" {
    const allocator = testing.allocator;

    // 12 = 4 + 4 + 4 (exactly 3 squares)
    try testing.expect(try canSumSquares(u32, allocator, 12, 3));
    try testing.expect(!try canSumSquares(u32, allocator, 12, 2));
    try testing.expect(try canSumSquares(u32, allocator, 12, 12)); // 1+1+...+1

    // 13 = 9 + 4 (exactly 2 squares)
    try testing.expect(try canSumSquares(u32, allocator, 13, 2));
    try testing.expect(!try canSumSquares(u32, allocator, 13, 1));
}

test "canSumSquares - larger numbers" {
    const allocator = testing.allocator;

    try testing.expect(try canSumSquares(u32, allocator, 25, 1)); // 25
    try testing.expect(try canSumSquares(u32, allocator, 26, 2)); // 25 + 1
    try testing.expect(try canSumSquares(u32, allocator, 18, 2)); // 9 + 9
}

test "numSquaresBFS - basic cases" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(u32, 0), try numSquaresBFS(u32, allocator, 0));
    try testing.expectEqual(@as(u32, 1), try numSquaresBFS(u32, allocator, 1));
    try testing.expectEqual(@as(u32, 2), try numSquaresBFS(u32, allocator, 2));
    try testing.expectEqual(@as(u32, 1), try numSquaresBFS(u32, allocator, 4));
}

test "numSquaresBFS - consistency with DP" {
    const allocator = testing.allocator;

    var i: u32 = 1;
    while (i <= 30) : (i += 1) {
        const dp_result = try numSquares(u32, i);
        const bfs_result = try numSquaresBFS(u32, allocator, i);
        try testing.expectEqual(dp_result, bfs_result);
    }
}

test "numSquaresBFS - classic examples" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(u32, 3), try numSquaresBFS(u32, allocator, 12));
    try testing.expectEqual(@as(u32, 2), try numSquaresBFS(u32, allocator, 13));
    try testing.expectEqual(@as(u32, 1), try numSquaresBFS(u32, allocator, 16));
}

test "intSqrt helper" {
    try testing.expectEqual(@as(u32, 0), intSqrt(u32, 0));
    try testing.expectEqual(@as(u32, 1), intSqrt(u32, 1));
    try testing.expectEqual(@as(u32, 1), intSqrt(u32, 2));
    try testing.expectEqual(@as(u32, 1), intSqrt(u32, 3));
    try testing.expectEqual(@as(u32, 2), intSqrt(u32, 4));
    try testing.expectEqual(@as(u32, 2), intSqrt(u32, 5));
    try testing.expectEqual(@as(u32, 3), intSqrt(u32, 9));
    try testing.expectEqual(@as(u32, 3), intSqrt(u32, 10));
    try testing.expectEqual(@as(u32, 10), intSqrt(u32, 100));
}

test "numSquares - type support" {
    // u8
    try testing.expectEqual(@as(u8, 1), try numSquares(u8, 1));
    try testing.expectEqual(@as(u8, 3), try numSquares(u8, 12));

    // u16
    try testing.expectEqual(@as(u16, 2), try numSquares(u16, 13));
    try testing.expectEqual(@as(u16, 1), try numSquares(u16, 100));

    // u64
    try testing.expectEqual(@as(u64, 3), try numSquares(u64, 12));
}

test "numSquares - large numbers" {
    // Test with larger inputs
    try testing.expectEqual(@as(u32, 3), try numSquares(u32, 43)); // 36 + 4 + 1 + 1 + 1 = 43 (wrong), actually 25+9+9
    try testing.expectEqual(@as(u32, 1), try numSquares(u32, 49)); // 49
    try testing.expectEqual(@as(u32, 2), try numSquares(u32, 50)); // 49 + 1
}

test "numSquaresWithPath - memory safety" {
    const allocator = testing.allocator;

    var i: u32 = 1;
    while (i <= 20) : (i += 1) {
        const result = try numSquaresWithPath(u32, allocator, i);
        defer allocator.free(result.squares);

        // Verify count matches
        try testing.expectEqual(@as(usize, result.count), result.squares.len);
    }
}

test "canSumSquares - memory safety" {
    const allocator = testing.allocator;

    var i: u32 = 1;
    while (i <= 20) : (i += 1) {
        _ = try canSumSquares(u32, allocator, i, 1);
        _ = try canSumSquares(u32, allocator, i, 2);
        _ = try canSumSquares(u32, allocator, i, 3);
    }
}
