// Maximal Square
//
// Classic 2D dynamic programming problem that finds the largest square
// containing only 1s in a binary matrix.
//
// Problem: Given an m×n binary matrix filled with 0s and 1s, find the largest
// square containing only 1s and return its area.
//
// Algorithm:
// - State: dp[i][j] = side length of largest square with bottom-right corner at (i,j)
// - Recurrence: if matrix[i][j] == 1:
//     dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
// - Base case: First row and column: dp[i][j] = matrix[i][j]
// - Answer: max(dp[i][j])² for all positions
//
// Time: O(m×n) for all variants where m = rows, n = cols
// Space: O(m×n) for standard, O(n) for space-optimized variant
//
// Reference: LeetCode #221 - Maximal Square

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Position in a 2D matrix
pub const Position = struct {
    row: usize,
    col: usize,
};

/// Result containing area and position of maximal square
pub fn SquareResult(comptime T: type) type {
    return struct {
        area: T,
        top_left: Position,
        side_length: T,
    };
}

/// Finds the largest square containing only 1s in a binary matrix.
/// Returns the area of the largest square.
///
/// Time: O(m×n) | Space: O(m×n)
///
/// Example:
/// ```
/// const matrix = [_][]const u32{
///     &[_]u32{1, 0, 1, 0, 0},
///     &[_]u32{1, 0, 1, 1, 1},
///     &[_]u32{1, 1, 1, 1, 1},
///     &[_]u32{1, 0, 0, 1, 0},
/// };
/// const area = try maximalSquare(u32, allocator, &matrix);
/// // Returns 4 (2×2 square)
/// ```
pub fn maximalSquare(comptime T: type, allocator: Allocator, matrix: []const []const T) !T {
    if (matrix.len == 0) return error.EmptyMatrix;
    const m = matrix.len;
    const n = matrix[0].len;
    if (n == 0) return error.EmptyMatrix;

    // Create DP table
    var dp = try allocator.alloc([]T, m);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }
    for (dp) |*row| {
        row.* = try allocator.alloc(T, n);
        @memset(row.*, 0);
    }

    var max_side: T = 0;

    // Fill DP table
    for (0..m) |i| {
        for (0..n) |j| {
            if (matrix[i][j] == 1) {
                if (i == 0 or j == 0) {
                    dp[i][j] = 1;
                } else {
                    const min_val = @min(@min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]);
                    dp[i][j] = min_val + 1;
                }
                max_side = @max(max_side, dp[i][j]);
            }
        }
    }

    return max_side * max_side; // Return area
}

/// Space-optimized version using rolling array.
/// Uses O(n) space instead of O(m×n).
///
/// Time: O(m×n) | Space: O(n)
pub fn maximalSquareOptimized(comptime T: type, allocator: Allocator, matrix: []const []const T) !T {
    if (matrix.len == 0) return error.EmptyMatrix;
    const m = matrix.len;
    const n = matrix[0].len;
    if (n == 0) return error.EmptyMatrix;

    // Rolling array: only need current and previous row
    var prev = try allocator.alloc(T, n);
    defer allocator.free(prev);
    var curr = try allocator.alloc(T, n);
    defer allocator.free(curr);
    @memset(prev, 0);
    @memset(curr, 0);

    var max_side: T = 0;

    for (0..m) |i| {
        for (0..n) |j| {
            if (matrix[i][j] == 1) {
                if (i == 0 or j == 0) {
                    curr[j] = 1;
                } else {
                    const min_val = @min(@min(prev[j], curr[j - 1]), prev[j - 1]);
                    curr[j] = min_val + 1;
                }
                max_side = @max(max_side, curr[j]);
            } else {
                curr[j] = 0;
            }
        }
        // Swap rows
        const tmp = prev;
        prev = curr;
        curr = tmp;
    }

    return max_side * max_side;
}

/// Returns the area, position, and side length of the largest square.
///
/// Time: O(m×n) | Space: O(m×n)
pub fn maximalSquareWithPosition(comptime T: type, allocator: Allocator, matrix: []const []const T) !SquareResult(T) {
    if (matrix.len == 0) return error.EmptyMatrix;
    const m = matrix.len;
    const n = matrix[0].len;
    if (n == 0) return error.EmptyMatrix;

    var dp = try allocator.alloc([]T, m);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }
    for (dp) |*row| {
        row.* = try allocator.alloc(T, n);
        @memset(row.*, 0);
    }

    var max_side: T = 0;
    var max_row: usize = 0;
    var max_col: usize = 0;

    for (0..m) |i| {
        for (0..n) |j| {
            if (matrix[i][j] == 1) {
                if (i == 0 or j == 0) {
                    dp[i][j] = 1;
                } else {
                    const min_val = @min(@min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]);
                    dp[i][j] = min_val + 1;
                }
                if (dp[i][j] > max_side) {
                    max_side = dp[i][j];
                    max_row = i;
                    max_col = j;
                }
            }
        }
    }

    // Calculate top-left corner from bottom-right
    const side_usize: usize = @intCast(max_side);
    const top_left = Position{
        .row = if (max_side > 0 and side_usize <= max_row + 1) max_row + 1 - side_usize else 0,
        .col = if (max_side > 0 and side_usize <= max_col + 1) max_col + 1 - side_usize else 0,
    };

    return SquareResult(T){
        .area = max_side * max_side,
        .top_left = top_left,
        .side_length = max_side,
    };
}

/// Counts the number of maximal squares (squares with the largest side length).
///
/// Time: O(m×n) | Space: O(m×n)
pub fn maximalSquareCount(comptime T: type, allocator: Allocator, matrix: []const []const T) !usize {
    if (matrix.len == 0) return error.EmptyMatrix;
    const m = matrix.len;
    const n = matrix[0].len;
    if (n == 0) return error.EmptyMatrix;

    var dp = try allocator.alloc([]T, m);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }
    for (dp) |*row| {
        row.* = try allocator.alloc(T, n);
        @memset(row.*, 0);
    }

    var max_side: T = 0;

    for (0..m) |i| {
        for (0..n) |j| {
            if (matrix[i][j] == 1) {
                if (i == 0 or j == 0) {
                    dp[i][j] = 1;
                } else {
                    const min_val = @min(@min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]);
                    dp[i][j] = min_val + 1;
                }
                max_side = @max(max_side, dp[i][j]);
            }
        }
    }

    // Count positions with max_side
    var count: usize = 0;
    for (0..m) |i| {
        for (0..n) |j| {
            if (dp[i][j] == max_side) {
                count += 1;
            }
        }
    }

    return count;
}

/// Returns just the side length of the largest square (not area).
///
/// Time: O(m×n) | Space: O(n)
pub fn largestSquareSideLength(comptime T: type, allocator: Allocator, matrix: []const []const T) !T {
    if (matrix.len == 0) return error.EmptyMatrix;
    const m = matrix.len;
    const n = matrix[0].len;
    if (n == 0) return error.EmptyMatrix;

    var prev = try allocator.alloc(T, n);
    defer allocator.free(prev);
    var curr = try allocator.alloc(T, n);
    defer allocator.free(curr);
    @memset(prev, 0);
    @memset(curr, 0);

    var max_side: T = 0;

    for (0..m) |i| {
        for (0..n) |j| {
            if (matrix[i][j] == 1) {
                if (i == 0 or j == 0) {
                    curr[j] = 1;
                } else {
                    const min_val = @min(@min(prev[j], curr[j - 1]), prev[j - 1]);
                    curr[j] = min_val + 1;
                }
                max_side = @max(max_side, curr[j]);
            } else {
                curr[j] = 0;
            }
        }
        const tmp = prev;
        prev = curr;
        curr = tmp;
    }

    return max_side;
}

// ============================================================================
// Tests
// ============================================================================

test "maximalSquare - basic example" {
    const allocator = std.testing.allocator;

    const matrix = [_][]const u32{
        &[_]u32{ 1, 0, 1, 0, 0 },
        &[_]u32{ 1, 0, 1, 1, 1 },
        &[_]u32{ 1, 1, 1, 1, 1 },
        &[_]u32{ 1, 0, 0, 1, 0 },
    };

    const area = try maximalSquare(u32, allocator, &matrix);
    try std.testing.expectEqual(@as(u32, 4), area); // 2×2 square
}

test "maximalSquare - single cell with 1" {
    const allocator = std.testing.allocator;

    const matrix = [_][]const u32{
        &[_]u32{1},
    };

    const area = try maximalSquare(u32, allocator, &matrix);
    try std.testing.expectEqual(@as(u32, 1), area);
}

test "maximalSquare - single cell with 0" {
    const allocator = std.testing.allocator;

    const matrix = [_][]const u32{
        &[_]u32{0},
    };

    const area = try maximalSquare(u32, allocator, &matrix);
    try std.testing.expectEqual(@as(u32, 0), area);
}

test "maximalSquare - all zeros" {
    const allocator = std.testing.allocator;

    const matrix = [_][]const u32{
        &[_]u32{ 0, 0, 0 },
        &[_]u32{ 0, 0, 0 },
        &[_]u32{ 0, 0, 0 },
    };

    const area = try maximalSquare(u32, allocator, &matrix);
    try std.testing.expectEqual(@as(u32, 0), area);
}

test "maximalSquare - all ones" {
    const allocator = std.testing.allocator;

    const matrix = [_][]const u32{
        &[_]u32{ 1, 1, 1 },
        &[_]u32{ 1, 1, 1 },
        &[_]u32{ 1, 1, 1 },
    };

    const area = try maximalSquare(u32, allocator, &matrix);
    try std.testing.expectEqual(@as(u32, 9), area); // 3×3 square
}

test "maximalSquare - single row" {
    const allocator = std.testing.allocator;

    const matrix = [_][]const u32{
        &[_]u32{ 1, 1, 1, 1 },
    };

    const area = try maximalSquare(u32, allocator, &matrix);
    try std.testing.expectEqual(@as(u32, 1), area); // Can't form square > 1×1
}

test "maximalSquare - single column" {
    const allocator = std.testing.allocator;

    const matrix = [_][]const u32{
        &[_]u32{1},
        &[_]u32{1},
        &[_]u32{1},
        &[_]u32{1},
    };

    const area = try maximalSquare(u32, allocator, &matrix);
    try std.testing.expectEqual(@as(u32, 1), area);
}

test "maximalSquare - large matrix" {
    const allocator = std.testing.allocator;

    var matrix_data: [20][20]u32 = undefined;
    var matrix_ptrs: [20][]const u32 = undefined;

    // Create a 20×20 matrix with a 5×5 square of 1s
    for (0..20) |i| {
        for (0..20) |j| {
            if (i >= 5 and i < 10 and j >= 5 and j < 10) {
                matrix_data[i][j] = 1;
            } else {
                matrix_data[i][j] = 0;
            }
        }
        matrix_ptrs[i] = &matrix_data[i];
    }

    const area = try maximalSquare(u32, allocator, &matrix_ptrs);
    try std.testing.expectEqual(@as(u32, 25), area); // 5×5 square
}

test "maximalSquare - empty matrix error" {
    const allocator = std.testing.allocator;

    const matrix: []const []const u32 = &[_][]const u32{};
    const result = maximalSquare(u32, allocator, matrix);
    try std.testing.expectError(error.EmptyMatrix, result);
}

test "maximalSquareOptimized - consistency with standard" {
    const allocator = std.testing.allocator;

    const matrix = [_][]const u32{
        &[_]u32{ 1, 0, 1, 0, 0 },
        &[_]u32{ 1, 0, 1, 1, 1 },
        &[_]u32{ 1, 1, 1, 1, 1 },
        &[_]u32{ 1, 0, 0, 1, 0 },
    };

    const area_standard = try maximalSquare(u32, allocator, &matrix);
    const area_optimized = try maximalSquareOptimized(u32, allocator, &matrix);
    try std.testing.expectEqual(area_standard, area_optimized);
}

test "maximalSquareOptimized - all ones" {
    const allocator = std.testing.allocator;

    const matrix = [_][]const u32{
        &[_]u32{ 1, 1, 1, 1 },
        &[_]u32{ 1, 1, 1, 1 },
        &[_]u32{ 1, 1, 1, 1 },
        &[_]u32{ 1, 1, 1, 1 },
    };

    const area = try maximalSquareOptimized(u32, allocator, &matrix);
    try std.testing.expectEqual(@as(u32, 16), area); // 4×4 square
}

test "maximalSquareWithPosition - basic" {
    const allocator = std.testing.allocator;

    const matrix = [_][]const u32{
        &[_]u32{ 1, 0, 1, 0, 0 },
        &[_]u32{ 1, 0, 1, 1, 1 },
        &[_]u32{ 1, 1, 1, 1, 1 },
        &[_]u32{ 1, 0, 0, 1, 0 },
    };

    const result = try maximalSquareWithPosition(u32, allocator, &matrix);
    try std.testing.expectEqual(@as(u32, 4), result.area);
    try std.testing.expectEqual(@as(u32, 2), result.side_length);
    // The 2×2 square is at rows 1-2, cols 2-3 (bottom-right at [2][3])
    try std.testing.expectEqual(@as(usize, 1), result.top_left.row);
    try std.testing.expectEqual(@as(usize, 2), result.top_left.col);
}

test "maximalSquareWithPosition - all ones" {
    const allocator = std.testing.allocator;

    const matrix = [_][]const u32{
        &[_]u32{ 1, 1, 1 },
        &[_]u32{ 1, 1, 1 },
        &[_]u32{ 1, 1, 1 },
    };

    const result = try maximalSquareWithPosition(u32, allocator, &matrix);
    try std.testing.expectEqual(@as(u32, 9), result.area);
    try std.testing.expectEqual(@as(u32, 3), result.side_length);
    try std.testing.expectEqual(@as(usize, 0), result.top_left.row);
    try std.testing.expectEqual(@as(usize, 0), result.top_left.col);
}

test "maximalSquareCount - single max square" {
    const allocator = std.testing.allocator;

    const matrix = [_][]const u32{
        &[_]u32{ 1, 1, 0, 0 },
        &[_]u32{ 1, 1, 0, 0 },
        &[_]u32{ 0, 0, 1, 0 },
    };

    const count = try maximalSquareCount(u32, allocator, &matrix);
    try std.testing.expectEqual(@as(usize, 1), count); // Only one 2×2 square at top-left
}

test "maximalSquareCount - multiple max squares" {
    const allocator = std.testing.allocator;

    const matrix = [_][]const u32{
        &[_]u32{ 1, 1, 0, 1, 1 },
        &[_]u32{ 1, 1, 0, 1, 1 },
    };

    const count = try maximalSquareCount(u32, allocator, &matrix);
    try std.testing.expectEqual(@as(usize, 2), count); // Two 2×2 squares
}

test "largestSquareSideLength - basic" {
    const allocator = std.testing.allocator;

    const matrix = [_][]const u32{
        &[_]u32{ 1, 0, 1, 0, 0 },
        &[_]u32{ 1, 0, 1, 1, 1 },
        &[_]u32{ 1, 1, 1, 1, 1 },
        &[_]u32{ 1, 0, 0, 1, 0 },
    };

    const side = try largestSquareSideLength(u32, allocator, &matrix);
    try std.testing.expectEqual(@as(u32, 2), side);
}

test "largestSquareSideLength - all ones" {
    const allocator = std.testing.allocator;

    const matrix = [_][]const u32{
        &[_]u32{ 1, 1, 1, 1 },
        &[_]u32{ 1, 1, 1, 1 },
        &[_]u32{ 1, 1, 1, 1 },
    };

    const side = try largestSquareSideLength(u32, allocator, &matrix);
    try std.testing.expectEqual(@as(u32, 3), side);
}

test "maximalSquare - f64 support" {
    const allocator = std.testing.allocator;

    const matrix = [_][]const f64{
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 1.0, 1.0 },
    };

    const area = try maximalSquare(f64, allocator, &matrix);
    try std.testing.expectEqual(@as(f64, 4.0), area);
}

test "maximalSquare - tall matrix" {
    const allocator = std.testing.allocator;

    const matrix = [_][]const u32{
        &[_]u32{ 1, 1 },
        &[_]u32{ 1, 1 },
        &[_]u32{ 1, 1 },
        &[_]u32{ 1, 1 },
        &[_]u32{ 1, 1 },
    };

    const area = try maximalSquare(u32, allocator, &matrix);
    try std.testing.expectEqual(@as(u32, 4), area); // 2×2 square (limited by width)
}

test "maximalSquare - wide matrix" {
    const allocator = std.testing.allocator;

    const matrix = [_][]const u32{
        &[_]u32{ 1, 1, 1, 1, 1 },
        &[_]u32{ 1, 1, 1, 1, 1 },
    };

    const area = try maximalSquare(u32, allocator, &matrix);
    try std.testing.expectEqual(@as(u32, 4), area); // 2×2 square (limited by height)
}

test "maximalSquare - diagonal pattern" {
    const allocator = std.testing.allocator;

    const matrix = [_][]const u32{
        &[_]u32{ 1, 0, 0, 0 },
        &[_]u32{ 0, 1, 0, 0 },
        &[_]u32{ 0, 0, 1, 0 },
        &[_]u32{ 0, 0, 0, 1 },
    };

    const area = try maximalSquare(u32, allocator, &matrix);
    try std.testing.expectEqual(@as(u32, 1), area); // No squares > 1×1
}

test "maximalSquare - memory safety" {
    const allocator = std.testing.allocator;

    const matrix = [_][]const u32{
        &[_]u32{ 1, 1, 1 },
        &[_]u32{ 1, 1, 1 },
        &[_]u32{ 1, 1, 1 },
    };

    _ = try maximalSquare(u32, allocator, &matrix);
    _ = try maximalSquareOptimized(u32, allocator, &matrix);
    _ = try maximalSquareWithPosition(u32, allocator, &matrix);
    _ = try maximalSquareCount(u32, allocator, &matrix);
    _ = try largestSquareSideLength(u32, allocator, &matrix);
    // No memory leaks if test passes
}
