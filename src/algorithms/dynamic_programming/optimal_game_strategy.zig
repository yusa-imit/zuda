//! Optimal Strategy for Game
//!
//! Classic dynamic programming problem where two players optimally pick numbers
//! from either end of an array. Each player tries to maximize their score.
//!
//! Problem: Given an array of values, two players take turns picking from either
//! end. Player 1 moves first. Both play optimally. What's the maximum score Player 1
//! can guarantee, and can Player 1 win?
//!
//! Algorithm: 2D DP where dp[i][j] = max score advantage (your score - opponent score)
//! for subarray arr[i..j+1] when it's your turn.
//!
//! Recurrence:
//!   dp[i][j] = max(arr[i] - dp[i+1][j], arr[j] - dp[i][j-1])
//! Base case: dp[i][i] = arr[i] (single element, take it)
//!
//! Time: O(n²) | Space: O(n²)
//!
//! Use cases:
//! - Game theory (optimal play, minimax strategy)
//! - Resource allocation (competitive settings)
//! - Interview problems (two-player optimal strategy)
//! - Competitive programming (range DP)
//!
//! Reference: LeetCode #486 (Predict the Winner), classic game theory DP

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Result of optimal game strategy
pub fn GameResult(comptime T: type) type {
    return struct {
        player1_score: T,
        player2_score: T,
        player1_wins: bool,
        max_advantage: T,
    };
}

/// Compute optimal strategy for game where two players pick from array ends
/// Returns max score Player 1 can guarantee and whether Player 1 wins
/// Time: O(n²) | Space: O(n²)
pub fn optimalStrategy(comptime T: type, allocator: Allocator, arr: []const T) !GameResult(T) {
    if (arr.len == 0) return error.EmptyArray;

    const n = arr.len;
    if (n == 1) {
        return GameResult(T){
            .player1_score = arr[0],
            .player2_score = 0,
            .player1_wins = true,
            .max_advantage = arr[0],
        };
    }

    // dp[i][j] = max advantage (your score - opponent score) for arr[i..j+1]
    var dp = try allocator.alloc([]T, n);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }
    for (dp) |*row| {
        row.* = try allocator.alloc(T, n);
        @memset(row.*, 0);
    }

    // Base case: single element
    for (0..n) |i| {
        dp[i][i] = arr[i];
    }

    // Fill DP table for increasing subarray lengths
    var length: usize = 2;
    while (length <= n) : (length += 1) {
        var i: usize = 0;
        while (i + length <= n) : (i += 1) {
            const j = i + length - 1;

            // Pick from left: arr[i] - dp[i+1][j] (opponent plays optimally on i+1..j)
            const pick_left = arr[i] - (if (i + 1 <= j) dp[i + 1][j] else 0);

            // Pick from right: arr[j] - dp[i][j-1] (opponent plays optimally on i..j-1)
            const pick_right = arr[j] - (if (i <= j - 1) dp[i][j - 1] else 0);

            dp[i][j] = @max(pick_left, pick_right);
        }
    }

    const max_advantage = dp[0][n - 1];

    // Calculate actual scores
    var total: T = 0;
    for (arr) |val| {
        total += val;
    }

    // advantage = player1_score - player2_score
    // player1_score + player2_score = total
    // => player1_score = (total + advantage) / 2
    const player1_score = @divTrunc(total + max_advantage, 2);
    const player2_score = total - player1_score;

    return GameResult(T){
        .player1_score = player1_score,
        .player2_score = player2_score,
        .player1_wins = max_advantage > 0,
        .max_advantage = max_advantage,
    };
}

/// Space-optimized version using rolling array
/// Time: O(n²) | Space: O(n)
pub fn optimalStrategyOptimized(comptime T: type, allocator: Allocator, arr: []const T) !GameResult(T) {
    if (arr.len == 0) return error.EmptyArray;

    const n = arr.len;
    if (n == 1) {
        return GameResult(T){
            .player1_score = arr[0],
            .player2_score = 0,
            .player1_wins = true,
            .max_advantage = arr[0],
        };
    }

    // Use two rows: previous and current
    var prev = try allocator.alloc(T, n);
    defer allocator.free(prev);
    var curr = try allocator.alloc(T, n);
    defer allocator.free(curr);

    @memset(prev, 0);
    @memset(curr, 0);

    // Base case
    for (0..n) |i| {
        prev[i] = arr[i];
    }

    var length: usize = 2;
    while (length <= n) : (length += 1) {
        var i: usize = 0;
        while (i + length <= n) : (i += 1) {
            const j = i + length - 1;

            const pick_left = arr[i] - (if (i + 1 <= j) prev[i + 1] else 0);
            const pick_right = arr[j] - (if (i <= j - 1) curr[i] else 0);

            curr[i] = @max(pick_left, pick_right);
        }

        // Swap for next iteration
        const tmp = prev;
        prev = curr;
        curr = tmp;
    }

    const max_advantage = prev[0];

    var total: T = 0;
    for (arr) |val| {
        total += val;
    }

    const player1_score = @divTrunc(total + max_advantage, 2);
    const player2_score = total - player1_score;

    return GameResult(T){
        .player1_score = player1_score,
        .player2_score = player2_score,
        .player1_wins = max_advantage > 0,
        .max_advantage = max_advantage,
    };
}

/// Check if Player 1 can win (convenience function)
/// Time: O(n²) | Space: O(n²)
pub fn canPlayer1Win(comptime T: type, allocator: Allocator, arr: []const T) !bool {
    const result = try optimalStrategy(T, allocator, arr);
    return result.player1_wins;
}

/// Get maximum score Player 1 can guarantee
/// Time: O(n²) | Space: O(n²)
pub fn maxPlayer1Score(comptime T: type, allocator: Allocator, arr: []const T) !T {
    const result = try optimalStrategy(T, allocator, arr);
    return result.player1_score;
}

// Tests
test "optimal game strategy - basic examples" {
    const allocator = std.testing.allocator;

    // Example 1: [1, 5, 2]
    // Player 1 picks 2, Player 2 picks 5, Player 1 picks 1
    // Player 1: 2+1=3, Player 2: 5
    // But optimal: Player 1 picks 5, Player 2 picks 2, Player 1 picks 1
    // Player 1: 5+1=6, Player 2: 2
    const arr1 = [_]i32{ 1, 5, 2 };
    const result1 = try optimalStrategy(i32, allocator, &arr1);
    try std.testing.expectEqual(@as(i32, 6), result1.player1_score);
    try std.testing.expectEqual(@as(i32, 2), result1.player2_score);
    try std.testing.expect(result1.player1_wins);

    // Example 2: [1, 5, 233, 7]
    const arr2 = [_]i32{ 1, 5, 233, 7 };
    const result2 = try optimalStrategy(i32, allocator, &arr2);
    try std.testing.expect(result2.player1_wins);
}

test "optimal game strategy - single element" {
    const allocator = std.testing.allocator;

    const arr = [_]i32{42};
    const result = try optimalStrategy(i32, allocator, &arr);
    try std.testing.expectEqual(@as(i32, 42), result.player1_score);
    try std.testing.expectEqual(@as(i32, 0), result.player2_score);
    try std.testing.expect(result.player1_wins);
}

test "optimal game strategy - two elements" {
    const allocator = std.testing.allocator;

    // Player 1 picks larger
    const arr1 = [_]i32{ 3, 7 };
    const result1 = try optimalStrategy(i32, allocator, &arr1);
    try std.testing.expectEqual(@as(i32, 7), result1.player1_score);
    try std.testing.expectEqual(@as(i32, 3), result1.player2_score);
    try std.testing.expect(result1.player1_wins);

    const arr2 = [_]i32{ 10, 2 };
    const result2 = try optimalStrategy(i32, allocator, &arr2);
    try std.testing.expectEqual(@as(i32, 10), result2.player1_score);
    try std.testing.expectEqual(@as(i32, 2), result2.player2_score);
}

test "optimal game strategy - even split" {
    const allocator = std.testing.allocator;

    // [1, 2, 3, 4] - total 10
    // Optimal: Player 1 can guarantee at least 5 (tie)
    const arr = [_]i32{ 1, 2, 3, 4 };
    const result = try optimalStrategy(i32, allocator, &arr);
    try std.testing.expectEqual(@as(i32, 10), result.player1_score + result.player2_score);
    // With optimal play from [1,2,3,4]:
    // P1 picks 4 → [1,2,3]
    // P2 picks 3 → [1,2]
    // P1 picks 2 → [1]
    // P2 picks 1
    // P1: 4+2=6, P2: 3+1=4
    try std.testing.expectEqual(@as(i32, 6), result.player1_score);
    try std.testing.expectEqual(@as(i32, 4), result.player2_score);
}

test "optimal game strategy - player 1 loses" {
    const allocator = std.testing.allocator;

    // [1, 100, 1, 1]
    // P1 picks 1 → [100,1,1]
    // P2 picks 100 → [1,1]
    // P1 picks 1 → [1]
    // P2 picks 1
    // P1: 1+1=2, P2: 100+1=101
    const arr = [_]i32{ 1, 100, 1, 1 };
    const result = try optimalStrategy(i32, allocator, &arr);
    try std.testing.expect(!result.player1_wins);
    try std.testing.expect(result.player1_score < result.player2_score);
}

test "optimal game strategy - large values" {
    const allocator = std.testing.allocator;

    const arr = [_]i32{ 1000, 500, 750, 250 };
    const result = try optimalStrategy(i32, allocator, &arr);
    try std.testing.expectEqual(@as(i32, 2500), result.player1_score + result.player2_score);
    try std.testing.expect(result.player1_wins);
}

test "optimal game strategy - all equal" {
    const allocator = std.testing.allocator;

    const arr = [_]i32{ 5, 5, 5, 5 };
    const result = try optimalStrategy(i32, allocator, &arr);
    try std.testing.expectEqual(@as(i32, 10), result.player1_score);
    try std.testing.expectEqual(@as(i32, 10), result.player2_score);
    try std.testing.expect(!result.player1_wins); // Tie
}

test "optimal game strategy - optimized version consistency" {
    const allocator = std.testing.allocator;

    const test_cases = [_][]const i32{
        &[_]i32{ 1, 5, 2 },
        &[_]i32{ 1, 5, 233, 7 },
        &[_]i32{ 3, 7, 10, 2 },
        &[_]i32{ 1, 2, 3, 4, 5 },
        &[_]i32{ 10, 20, 30 },
    };

    for (test_cases) |arr| {
        const result1 = try optimalStrategy(i32, allocator, arr);
        const result2 = try optimalStrategyOptimized(i32, allocator, arr);

        try std.testing.expectEqual(result1.player1_score, result2.player1_score);
        try std.testing.expectEqual(result1.player2_score, result2.player2_score);
        try std.testing.expectEqual(result1.player1_wins, result2.player1_wins);
        try std.testing.expectEqual(result1.max_advantage, result2.max_advantage);
    }
}

test "optimal game strategy - convenience functions" {
    const allocator = std.testing.allocator;

    const arr = [_]i32{ 1, 5, 2 };
    const can_win = try canPlayer1Win(i32, allocator, &arr);
    const max_score = try maxPlayer1Score(i32, allocator, &arr);

    try std.testing.expect(can_win);
    try std.testing.expectEqual(@as(i32, 6), max_score);
}

test "optimal game strategy - negative values" {
    const allocator = std.testing.allocator;

    // Players minimize loss
    const arr = [_]i32{ -1, -5, -2 };
    const result = try optimalStrategy(i32, allocator, &arr);
    try std.testing.expectEqual(@as(i32, -8), result.player1_score + result.player2_score);
    // P1 minimizes loss: picks -1, P2 picks -2, P1 picks -5
    // P1: -1 + -5 = -6, P2: -2
    try std.testing.expectEqual(@as(i32, -3), result.player1_score);
    try std.testing.expectEqual(@as(i32, -5), result.player2_score);
    try std.testing.expect(result.player1_wins); // Less negative wins
}

test "optimal game strategy - mixed signs" {
    const allocator = std.testing.allocator;

    const arr = [_]i32{ -10, 5, -3, 8 };
    const result = try optimalStrategy(i32, allocator, &arr);
    try std.testing.expectEqual(@as(i32, 0), result.player1_score + result.player2_score);
    try std.testing.expect(result.player1_score >= result.player2_score or !result.player1_wins);
}

test "optimal game strategy - large array" {
    const allocator = std.testing.allocator;

    var arr: [20]i32 = undefined;
    for (0..20) |i| {
        arr[i] = @as(i32, @intCast(i + 1));
    }

    const result = try optimalStrategy(i32, allocator, &arr);
    var total: i32 = 0;
    for (arr) |val| total += val;
    try std.testing.expectEqual(total, result.player1_score + result.player2_score);
    try std.testing.expect(result.player1_wins);
}

test "optimal game strategy - f64 support" {
    const allocator = std.testing.allocator;

    const arr = [_]f64{ 1.5, 5.5, 2.5 };
    const result = try optimalStrategy(f64, allocator, &arr);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), result.player1_score, 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), result.player2_score, 0.01);
    try std.testing.expect(result.player1_wins);
}

test "optimal game strategy - empty array error" {
    const allocator = std.testing.allocator;

    const arr = [_]i32{};
    const result = optimalStrategy(i32, allocator, &arr);
    try std.testing.expectError(error.EmptyArray, result);
}

test "optimal game strategy - memory safety" {
    const allocator = std.testing.allocator;

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const arr = [_]i32{ 1, 2, 3, 4, 5 };
        const result = try optimalStrategy(i32, allocator, &arr);
        try std.testing.expect(result.player1_score > 0);
    }
}

test "optimal game strategy - three elements" {
    const allocator = std.testing.allocator;

    // [2, 4, 6]
    // P1 picks 6 → [2,4]
    // P2 picks 4 → [2]
    // P1 picks 2
    // P1: 6+2=8, P2: 4
    const arr = [_]i32{ 2, 4, 6 };
    const result = try optimalStrategy(i32, allocator, &arr);
    try std.testing.expectEqual(@as(i32, 8), result.player1_score);
    try std.testing.expectEqual(@as(i32, 4), result.player2_score);
}

test "optimal game strategy - advantage calculation" {
    const allocator = std.testing.allocator;

    const arr = [_]i32{ 1, 5, 2 };
    const result = try optimalStrategy(i32, allocator, &arr);
    try std.testing.expectEqual(@as(i32, 4), result.max_advantage); // 6 - 2 = 4
}
