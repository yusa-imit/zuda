//! Longest Increasing Path in Matrix
//!
//! Given an m×n matrix, finds the length of the longest strictly increasing path.
//! From each cell, you can move in 4 directions (up, down, left, right).
//!
//! Algorithm:
//! - DFS with memoization on each cell
//! - memo[i][j] = length of longest increasing path starting from (i,j)
//! - Recurrence: memo[i][j] = 1 + max(dfs(neighbor)) for valid increasing neighbors
//!
//! Time: O(m×n) where m=rows, n=cols (each cell visited once due to memoization)
//! Space: O(m×n) for memoization table
//!
//! Use cases:
//! - Terrain analysis (elevation paths)
//! - Game pathfinding (increasing difficulty levels)
//! - Matrix optimization problems
//! - Educational DP (classic 2D memoization problem)

const std = @import("std");
const testing = std.testing;

/// Longest increasing path in matrix
/// Time: O(m×n) | Space: O(m×n)
pub fn longestIncreasingPath(comptime T: type, matrix: []const []const T, allocator: std.mem.Allocator) !usize {
    if (matrix.len == 0) return error.EmptyMatrix;
    const m = matrix.len;
    const n = matrix[0].len;
    if (n == 0) return error.EmptyMatrix;

    // Validate all rows have same length
    for (matrix) |row| {
        if (row.len != n) return error.InconsistentRowLengths;
    }

    // Memoization table: memo[i][j] = longest path starting from (i,j)
    const memo = try allocator.alloc([]usize, m);
    defer {
        for (memo) |row| allocator.free(row);
        allocator.free(memo);
    }
    for (memo) |*row| {
        row.* = try allocator.alloc(usize, n);
        @memset(row.*, 0); // 0 means not computed yet
    }

    // Try starting from each cell
    var max_length: usize = 1;
    for (0..m) |i| {
        for (0..n) |j| {
            const length = try dfs(T, matrix, memo, @intCast(i), @intCast(j));
            max_length = @max(max_length, length);
        }
    }

    return max_length;
}

/// DFS with memoization to find longest path from (i,j)
fn dfs(comptime T: type, matrix: []const []const T, memo: [][]usize, i: isize, j: isize) !usize {
    const m: isize = @intCast(matrix.len);
    const n: isize = @intCast(matrix[0].len);

    const ui: usize = @intCast(i);
    const uj: usize = @intCast(j);

    // Return cached result
    if (memo[ui][uj] != 0) return memo[ui][uj];

    // Base case: at least length 1 (current cell)
    var max_length: usize = 1;

    // Try all 4 directions
    const dirs = [_][2]isize{ .{ -1, 0 }, .{ 1, 0 }, .{ 0, -1 }, .{ 0, 1 } };
    for (dirs) |dir| {
        const ni = i + dir[0];
        const nj = j + dir[1];

        // Check bounds
        if (ni < 0 or ni >= m or nj < 0 or nj >= n) continue;

        const nui: usize = @intCast(ni);
        const nuj: usize = @intCast(nj);

        // Check if strictly increasing
        if (matrix[nui][nuj] <= matrix[ui][uj]) continue;

        // Recurse
        const neighbor_length = try dfs(T, matrix, memo, ni, nj);
        max_length = @max(max_length, 1 + neighbor_length);
    }

    // Cache and return
    memo[ui][uj] = max_length;
    return max_length;
}

/// Longest increasing path with actual path reconstruction
/// Time: O(m×n) | Space: O(m×n)
pub fn longestIncreasingPathWithPath(comptime T: type, matrix: []const []const T, allocator: std.mem.Allocator) !struct { length: usize, path: []const [2]isize } {
    if (matrix.len == 0) return error.EmptyMatrix;
    const m = matrix.len;
    const n = matrix[0].len;
    if (n == 0) return error.EmptyMatrix;

    for (matrix) |row| {
        if (row.len != n) return error.InconsistentRowLengths;
    }

    // Memoization
    const memo = try allocator.alloc([]usize, m);
    defer {
        for (memo) |row| allocator.free(row);
        allocator.free(memo);
    }
    for (memo) |*row| {
        row.* = try allocator.alloc(usize, n);
        @memset(row.*, 0);
    }

    // Find starting cell with longest path
    var max_length: usize = 1;
    var start_i: usize = 0;
    var start_j: usize = 0;

    for (0..m) |i| {
        for (0..n) |j| {
            const length = try dfs(T, matrix, memo, @intCast(i), @intCast(j));
            if (length > max_length) {
                max_length = length;
                start_i = i;
                start_j = j;
            }
        }
    }

    // Reconstruct path by following greedy increasing neighbors
    var path = try allocator.alloc([2]isize, max_length);
    errdefer allocator.free(path);

    var ci: isize = @intCast(start_i);
    var cj: isize = @intCast(start_j);
    path[0] = .{ ci, cj };

    for (1..max_length) |idx| {
        const dirs = [_][2]isize{ .{ -1, 0 }, .{ 1, 0 }, .{ 0, -1 }, .{ 0, 1 } };
        var found = false;

        for (dirs) |dir| {
            const ni = ci + dir[0];
            const nj = cj + dir[1];

            if (ni < 0 or ni >= @as(isize, @intCast(m))) continue;
            if (nj < 0 or nj >= @as(isize, @intCast(n))) continue;

            const nui: usize = @intCast(ni);
            const nuj: usize = @intCast(nj);
            const cui: usize = @intCast(ci);
            const cuj: usize = @intCast(cj);

            // Check if this neighbor extends the path
            if (matrix[nui][nuj] > matrix[cui][cuj] and memo[nui][nuj] == memo[cui][cuj] - 1) {
                path[idx] = .{ ni, nj };
                ci = ni;
                cj = nj;
                found = true;
                break;
            }
        }

        if (!found) return error.PathReconstructionFailed;
    }

    return .{ .length = max_length, .path = path };
}

/// Longest decreasing path (variant)
/// Time: O(m×n) | Space: O(m×n)
pub fn longestDecreasingPath(comptime T: type, matrix: []const []const T, allocator: std.mem.Allocator) !usize {
    if (matrix.len == 0) return error.EmptyMatrix;
    const m = matrix.len;
    const n = matrix[0].len;
    if (n == 0) return error.EmptyMatrix;

    for (matrix) |row| {
        if (row.len != n) return error.InconsistentRowLengths;
    }

    const memo = try allocator.alloc([]usize, m);
    defer {
        for (memo) |row| allocator.free(row);
        allocator.free(memo);
    }
    for (memo) |*row| {
        row.* = try allocator.alloc(usize, n);
        @memset(row.*, 0);
    }

    var max_length: usize = 1;
    for (0..m) |i| {
        for (0..n) |j| {
            const length = try dfsDecreasing(T, matrix, memo, @intCast(i), @intCast(j));
            max_length = @max(max_length, length);
        }
    }

    return max_length;
}

fn dfsDecreasing(comptime T: type, matrix: []const []const T, memo: [][]usize, i: isize, j: isize) !usize {
    const m: isize = @intCast(matrix.len);
    const n: isize = @intCast(matrix[0].len);

    const ui: usize = @intCast(i);
    const uj: usize = @intCast(j);

    if (memo[ui][uj] != 0) return memo[ui][uj];

    var max_length: usize = 1;

    const dirs = [_][2]isize{ .{ -1, 0 }, .{ 1, 0 }, .{ 0, -1 }, .{ 0, 1 } };
    for (dirs) |dir| {
        const ni = i + dir[0];
        const nj = j + dir[1];

        if (ni < 0 or ni >= m or nj < 0 or nj >= n) continue;

        const nui: usize = @intCast(ni);
        const nuj: usize = @intCast(nj);

        // Check if strictly decreasing
        if (matrix[nui][nuj] >= matrix[ui][uj]) continue;

        const neighbor_length = try dfsDecreasing(T, matrix, memo, ni, nj);
        max_length = @max(max_length, 1 + neighbor_length);
    }

    memo[ui][uj] = max_length;
    return max_length;
}

// Tests
test "longest increasing path - basic 3x3" {
    const matrix = [_][]const i32{
        &[_]i32{ 9, 9, 4 },
        &[_]i32{ 6, 6, 8 },
        &[_]i32{ 2, 1, 1 },
    };
    const result = try longestIncreasingPath(i32, &matrix, testing.allocator);
    try testing.expectEqual(@as(usize, 4), result); // 1→2→6→9
}

test "longest increasing path - basic 3x3 variant" {
    const matrix = [_][]const i32{
        &[_]i32{ 3, 4, 5 },
        &[_]i32{ 3, 2, 6 },
        &[_]i32{ 2, 2, 1 },
    };
    const result = try longestIncreasingPath(i32, &matrix, testing.allocator);
    try testing.expectEqual(@as(usize, 4), result); // 1→2→5→6 or 2→3→4→5
}

test "longest increasing path - single cell" {
    const matrix = [_][]const i32{
        &[_]i32{5},
    };
    const result = try longestIncreasingPath(i32, &matrix, testing.allocator);
    try testing.expectEqual(@as(usize, 1), result);
}

test "longest increasing path - single row" {
    const matrix = [_][]const i32{
        &[_]i32{ 1, 2, 3, 4, 5 },
    };
    const result = try longestIncreasingPath(i32, &matrix, testing.allocator);
    try testing.expectEqual(@as(usize, 5), result);
}

test "longest increasing path - single column" {
    const matrix = [_][]const i32{
        &[_]i32{1},
        &[_]i32{2},
        &[_]i32{3},
        &[_]i32{4},
    };
    const result = try longestIncreasingPath(i32, &matrix, testing.allocator);
    try testing.expectEqual(@as(usize, 4), result);
}

test "longest increasing path - all equal" {
    const matrix = [_][]const i32{
        &[_]i32{ 5, 5, 5 },
        &[_]i32{ 5, 5, 5 },
        &[_]i32{ 5, 5, 5 },
    };
    const result = try longestIncreasingPath(i32, &matrix, testing.allocator);
    try testing.expectEqual(@as(usize, 1), result);
}

test "longest increasing path - strictly increasing" {
    const matrix = [_][]const i32{
        &[_]i32{ 1, 2, 3 },
        &[_]i32{ 4, 5, 6 },
        &[_]i32{ 7, 8, 9 },
    };
    const result = try longestIncreasingPath(i32, &matrix, testing.allocator);
    try testing.expectEqual(@as(usize, 5), result); // 1→2→3→6→9 or 1→4→5→6→9
}

test "longest increasing path - large matrix" {
    var matrix_data: [20][20]i32 = undefined;
    for (0..20) |i| {
        for (0..20) |j| {
            matrix_data[i][j] = @intCast(i * 20 + j);
        }
    }

    var matrix: [20][]const i32 = undefined;
    for (0..20) |i| {
        matrix[i] = &matrix_data[i];
    }

    const result = try longestIncreasingPath(i32, &matrix, testing.allocator);
    try testing.expect(result > 1);
}

test "longest increasing path - with path reconstruction" {
    const matrix = [_][]const i32{
        &[_]i32{ 9, 9, 4 },
        &[_]i32{ 6, 6, 8 },
        &[_]i32{ 2, 1, 1 },
    };
    const result = try longestIncreasingPathWithPath(i32, &matrix, testing.allocator);
    defer testing.allocator.free(result.path);

    try testing.expectEqual(@as(usize, 4), result.length);
    try testing.expectEqual(@as(usize, 4), result.path.len);

    // Validate path is strictly increasing
    for (1..result.path.len) |i| {
        const prev = result.path[i - 1];
        const curr = result.path[i];
        const prev_val = matrix[@intCast(prev[0])][@intCast(prev[1])];
        const curr_val = matrix[@intCast(curr[0])][@intCast(curr[1])];
        try testing.expect(curr_val > prev_val);
    }
}

test "longest increasing path - path validation" {
    const matrix = [_][]const i32{
        &[_]i32{ 3, 4, 5 },
        &[_]i32{ 3, 2, 6 },
        &[_]i32{ 2, 2, 1 },
    };
    const result = try longestIncreasingPathWithPath(i32, &matrix, testing.allocator);
    defer testing.allocator.free(result.path);

    try testing.expectEqual(@as(usize, 4), result.length);

    // Validate path coordinates are adjacent (4-connected)
    for (1..result.path.len) |i| {
        const prev = result.path[i - 1];
        const curr = result.path[i];
        const di = @abs(curr[0] - prev[0]);
        const dj = @abs(curr[1] - prev[1]);
        try testing.expect((di == 1 and dj == 0) or (di == 0 and dj == 1));
    }
}

test "longest decreasing path - basic" {
    const matrix = [_][]const i32{
        &[_]i32{ 9, 9, 4 },
        &[_]i32{ 6, 6, 8 },
        &[_]i32{ 2, 1, 1 },
    };
    const result = try longestDecreasingPath(i32, &matrix, testing.allocator);
    try testing.expectEqual(@as(usize, 4), result); // 9→6→2→1
}

test "longest decreasing path - strictly decreasing" {
    const matrix = [_][]const i32{
        &[_]i32{ 9, 8, 7 },
        &[_]i32{ 6, 5, 4 },
        &[_]i32{ 3, 2, 1 },
    };
    const result = try longestDecreasingPath(i32, &matrix, testing.allocator);
    try testing.expectEqual(@as(usize, 5), result); // 9→8→7→4→1 or 9→6→5→4→1
}

test "longest increasing path - empty matrix error" {
    const matrix = [_][]const i32{};
    try testing.expectError(error.EmptyMatrix, longestIncreasingPath(i32, &matrix, testing.allocator));
}

test "longest increasing path - f32 support" {
    const matrix = [_][]const f32{
        &[_]f32{ 1.5, 2.5, 3.5 },
        &[_]f32{ 4.5, 5.5, 6.5 },
    };
    const result = try longestIncreasingPath(f32, &matrix, testing.allocator);
    try testing.expectEqual(@as(usize, 4), result); // 1.5→2.5→3.5→6.5 or similar
}

test "longest increasing path - f64 support" {
    const matrix = [_][]const f64{
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 3.0, 4.0 },
    };
    const result = try longestIncreasingPath(f64, &matrix, testing.allocator);
    try testing.expectEqual(@as(usize, 3), result); // 1.0→2.0→4.0 or 1.0→3.0→4.0
}

test "longest increasing path - negative values" {
    const matrix = [_][]const i32{
        &[_]i32{ -5, -4, -3 },
        &[_]i32{ -2, -1, 0 },
    };
    const result = try longestIncreasingPath(i32, &matrix, testing.allocator);
    try testing.expectEqual(@as(usize, 4), result); // -5→-4→-3→0 or -5→-4→-1→0
}

test "longest increasing path - mixed values" {
    const matrix = [_][]const i32{
        &[_]i32{ -10, 9, 20 },
        &[_]i32{ 0, -5, 15 },
    };
    const result = try longestIncreasingPath(i32, &matrix, testing.allocator);
    try testing.expect(result >= 3);
}

test "longest increasing path - memory safety" {
    const matrix = [_][]const i32{
        &[_]i32{ 1, 2, 3 },
        &[_]i32{ 4, 5, 6 },
    };
    _ = try longestIncreasingPath(i32, &matrix, testing.allocator);
    // No memory leaks expected with testing.allocator
}

test "longest increasing path - path memory safety" {
    const matrix = [_][]const i32{
        &[_]i32{ 1, 2 },
        &[_]i32{ 3, 4 },
    };
    const result = try longestIncreasingPathWithPath(i32, &matrix, testing.allocator);
    defer testing.allocator.free(result.path);
    try testing.expectEqual(@as(usize, 3), result.length); // 1→2→4 or 1→3→4
}
