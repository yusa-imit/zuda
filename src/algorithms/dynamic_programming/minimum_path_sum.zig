const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Finds the minimum sum path from top-left to bottom-right in a grid.
/// You can only move right or down at each step.
///
/// Time: O(m×n) where m=rows, n=cols
/// Space: O(m×n) for standard, O(n) for space-optimized
///
/// Example:
/// ```zig
/// const grid = [_][3]i32{
///     .{ 1, 3, 1 },
///     .{ 1, 5, 1 },
///     .{ 4, 2, 1 },
/// };
/// const min_sum = try minPathSum(i32, allocator, &grid);
/// // Returns: 7 (path: 1→3→1→1→1)
/// ```
pub fn minPathSum(comptime T: type, allocator: Allocator, grid: []const []const T) !T {
    if (grid.len == 0 or grid[0].len == 0) return error.EmptyGrid;

    const m = grid.len;
    const n = grid[0].len;

    // Allocate DP table
    const dp = try allocator.alloc([]T, m);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }
    for (dp) |*row| {
        row.* = try allocator.alloc(T, n);
    }

    // Initialize first cell
    dp[0][0] = grid[0][0];

    // Initialize first row (can only come from left)
    for (1..n) |j| {
        dp[0][j] = dp[0][j - 1] + grid[0][j];
    }

    // Initialize first column (can only come from top)
    for (1..m) |i| {
        dp[i][0] = dp[i - 1][0] + grid[i][0];
    }

    // Fill the rest of the table
    for (1..m) |i| {
        for (1..n) |j| {
            dp[i][j] = @min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
        }
    }

    return dp[m - 1][n - 1];
}

/// Space-optimized version using O(n) space with rolling array.
///
/// Time: O(m×n)
/// Space: O(n) where n=columns
pub fn minPathSumOptimized(comptime T: type, allocator: Allocator, grid: []const []const T) !T {
    if (grid.len == 0 or grid[0].len == 0) return error.EmptyGrid;

    const m = grid.len;
    const n = grid[0].len;

    // Single row buffer
    const dp = try allocator.alloc(T, n);
    defer allocator.free(dp);

    // Initialize first row
    dp[0] = grid[0][0];
    for (1..n) |j| {
        dp[j] = dp[j - 1] + grid[0][j];
    }

    // Process remaining rows
    for (1..m) |i| {
        // First column in this row
        dp[0] = dp[0] + grid[i][0];
        // Rest of columns
        for (1..n) |j| {
            dp[j] = @min(dp[j], dp[j - 1]) + grid[i][j];
        }
    }

    return dp[n - 1];
}

/// Returns the actual path (sequence of cells) that achieves minimum sum.
///
/// Time: O(m×n)
/// Space: O(m×n) for DP table + O(m+n) for path
pub fn minPathSumWithPath(comptime T: type, allocator: Allocator, grid: []const []const T) !struct {
    sum: T,
    path: []const [2]usize,
} {
    if (grid.len == 0 or grid[0].len == 0) return error.EmptyGrid;

    const m = grid.len;
    const n = grid[0].len;

    // Allocate DP table
    const dp = try allocator.alloc([]T, m);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }
    for (dp) |*row| {
        row.* = try allocator.alloc(T, n);
    }

    // Initialize first cell
    dp[0][0] = grid[0][0];

    // Initialize first row
    for (1..n) |j| {
        dp[0][j] = dp[0][j - 1] + grid[0][j];
    }

    // Initialize first column
    for (1..m) |i| {
        dp[i][0] = dp[i - 1][0] + grid[i][0];
    }

    // Fill the DP table
    for (1..m) |i| {
        for (1..n) |j| {
            dp[i][j] = @min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
        }
    }

    // Backtrack to find the path
    var path = try std.ArrayList([2]usize).initCapacity(allocator, m + n);
    defer path.deinit(allocator);

    var i: usize = m - 1;
    var j: usize = n - 1;

    while (i > 0 or j > 0) {
        path.appendAssumeCapacity(.{ i, j });

        if (i == 0) {
            j -= 1;
        } else if (j == 0) {
            i -= 1;
        } else {
            // Choose the cell that led to minimum
            if (dp[i - 1][j] < dp[i][j - 1]) {
                i -= 1;
            } else {
                j -= 1;
            }
        }
    }
    path.appendAssumeCapacity(.{ 0, 0 });

    // Reverse path to get top-left to bottom-right order
    std.mem.reverse([2]usize, path.items);

    return .{
        .sum = dp[m - 1][n - 1],
        .path = try path.toOwnedSlice(allocator),
    };
}

/// Computes minimum path sum with additional constraints:
/// - Can move right, down, or diagonally
///
/// Time: O(m×n)
/// Space: O(m×n)
pub fn minPathSumWithDiagonal(comptime T: type, allocator: Allocator, grid: []const []const T) !T {
    if (grid.len == 0 or grid[0].len == 0) return error.EmptyGrid;

    const m = grid.len;
    const n = grid[0].len;

    const dp = try allocator.alloc([]T, m);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }
    for (dp) |*row| {
        row.* = try allocator.alloc(T, n);
    }

    // Initialize first cell
    dp[0][0] = grid[0][0];

    // Initialize first row
    for (1..n) |j| {
        dp[0][j] = dp[0][j - 1] + grid[0][j];
    }

    // Initialize first column
    for (1..m) |i| {
        dp[i][0] = dp[i - 1][0] + grid[i][0];
    }

    // Fill table considering diagonal moves
    for (1..m) |i| {
        for (1..n) |j| {
            const from_top = dp[i - 1][j];
            const from_left = dp[i][j - 1];
            const from_diag = dp[i - 1][j - 1];
            dp[i][j] = @min(@min(from_top, from_left), from_diag) + grid[i][j];
        }
    }

    return dp[m - 1][n - 1];
}

/// Finds the maximum sum path (useful for reward grids).
///
/// Time: O(m×n)
/// Space: O(n)
pub fn maxPathSum(comptime T: type, allocator: Allocator, grid: []const []const T) !T {
    if (grid.len == 0 or grid[0].len == 0) return error.EmptyGrid;

    const m = grid.len;
    const n = grid[0].len;

    const dp = try allocator.alloc(T, n);
    defer allocator.free(dp);

    // Initialize first row
    dp[0] = grid[0][0];
    for (1..n) |j| {
        dp[j] = dp[j - 1] + grid[0][j];
    }

    // Process remaining rows
    for (1..m) |i| {
        dp[0] = dp[0] + grid[i][0];
        for (1..n) |j| {
            dp[j] = @max(dp[j], dp[j - 1]) + grid[i][j];
        }
    }

    return dp[n - 1];
}

// ============================================================================
// Tests
// ============================================================================

test "minPathSum - basic 3x3 grid" {
    const grid = [_][3]i32{
        .{ 1, 3, 1 },
        .{ 1, 5, 1 },
        .{ 4, 2, 1 },
    };
    const grid_ptrs = [_][]const i32{
        &grid[0],
        &grid[1],
        &grid[2],
    };

    const result = try minPathSum(i32, testing.allocator, &grid_ptrs);
    try testing.expectEqual(@as(i32, 7), result); // Path: 1→3→1→1→1
}

test "minPathSum - single cell" {
    const grid = [_][1]i32{.{5}};
    const grid_ptrs = [_][]const i32{&grid[0]};

    const result = try minPathSum(i32, testing.allocator, &grid_ptrs);
    try testing.expectEqual(@as(i32, 5), result);
}

test "minPathSum - single row" {
    const grid = [_][4]i32{.{ 1, 2, 3, 4 }};
    const grid_ptrs = [_][]const i32{&grid[0]};

    const result = try minPathSum(i32, testing.allocator, &grid_ptrs);
    try testing.expectEqual(@as(i32, 10), result); // 1+2+3+4
}

test "minPathSum - single column" {
    const grid = [_][1]i32{
        .{1},
        .{2},
        .{3},
        .{4},
    };
    const grid_ptrs = [_][]const i32{
        &grid[0],
        &grid[1],
        &grid[2],
        &grid[3],
    };

    const result = try minPathSum(i32, testing.allocator, &grid_ptrs);
    try testing.expectEqual(@as(i32, 10), result);
}

test "minPathSum - empty grid error" {
    const grid: []const []const i32 = &[_][]const i32{};
    try testing.expectError(error.EmptyGrid, minPathSum(i32, testing.allocator, grid));
}

test "minPathSum - 2x2 grid" {
    const grid = [_][2]i32{
        .{ 1, 2 },
        .{ 3, 4 },
    };
    const grid_ptrs = [_][]const i32{
        &grid[0],
        &grid[1],
    };

    const result = try minPathSum(i32, testing.allocator, &grid_ptrs);
    try testing.expectEqual(@as(i32, 7), result); // 1→2→4
}

test "minPathSumOptimized - matches standard version" {
    const grid = [_][3]i32{
        .{ 1, 3, 1 },
        .{ 1, 5, 1 },
        .{ 4, 2, 1 },
    };
    const grid_ptrs = [_][]const i32{
        &grid[0],
        &grid[1],
        &grid[2],
    };

    const standard = try minPathSum(i32, testing.allocator, &grid_ptrs);
    const optimized = try minPathSumOptimized(i32, testing.allocator, &grid_ptrs);
    try testing.expectEqual(standard, optimized);
}

test "minPathSumOptimized - large grid" {
    var grid: [50][50]i32 = undefined;
    for (&grid, 0..) |*row, i| {
        for (row, 0..) |*cell, j| {
            cell.* = @as(i32, @intCast(i + j + 1));
        }
    }

    var grid_ptrs: [50][]const i32 = undefined;
    for (&grid_ptrs, 0..) |*ptr, i| {
        ptr.* = &grid[i];
    }

    const result = try minPathSumOptimized(i32, testing.allocator, &grid_ptrs);
    try testing.expect(result > 0);
}

test "minPathSumWithPath - basic 3x3" {
    const grid = [_][3]i32{
        .{ 1, 3, 1 },
        .{ 1, 5, 1 },
        .{ 4, 2, 1 },
    };
    const grid_ptrs = [_][]const i32{
        &grid[0],
        &grid[1],
        &grid[2],
    };

    const result = try minPathSumWithPath(i32, testing.allocator, &grid_ptrs);
    defer testing.allocator.free(result.path);

    try testing.expectEqual(@as(i32, 7), result.sum);
    try testing.expectEqual(@as(usize, 5), result.path.len); // 5 cells in path

    // Verify path starts at (0,0) and ends at (2,2)
    try testing.expectEqual([2]usize{ 0, 0 }, result.path[0]);
    try testing.expectEqual([2]usize{ 2, 2 }, result.path[result.path.len - 1]);

    // Verify path only moves right or down
    for (result.path[0 .. result.path.len - 1], result.path[1..]) |curr, next| {
        const di = next[0] - curr[0];
        const dj = next[1] - curr[1];
        // Either move down (di=1, dj=0) or right (di=0, dj=1)
        try testing.expect((di == 1 and dj == 0) or (di == 0 and dj == 1));
    }
}

test "minPathSumWithPath - single cell" {
    const grid = [_][1]i32{.{5}};
    const grid_ptrs = [_][]const i32{&grid[0]};

    const result = try minPathSumWithPath(i32, testing.allocator, &grid_ptrs);
    defer testing.allocator.free(result.path);

    try testing.expectEqual(@as(i32, 5), result.sum);
    try testing.expectEqual(@as(usize, 1), result.path.len);
    try testing.expectEqual([2]usize{ 0, 0 }, result.path[0]);
}

test "minPathSumWithDiagonal - basic 3x3" {
    const grid = [_][3]i32{
        .{ 1, 3, 1 },
        .{ 1, 5, 1 },
        .{ 4, 2, 1 },
    };
    const grid_ptrs = [_][]const i32{
        &grid[0],
        &grid[1],
        &grid[2],
    };

    const result = try minPathSumWithDiagonal(i32, testing.allocator, &grid_ptrs);
    // Diagonal allows: 1→5→1 = 7, but 1→3→1→1→1 = 7 or 1→1→1→1 = 4 via diagonal
    // Path 1(0,0)→diagonal→5(1,1)→1(2,2) = 1+5+1 = 7
    // Path 1(0,0)→diagonal→2(1,1 via 1,0 then 0,1) is not diagonal
    // Actually: 1(0,0)→1(1,0)→2(1,1)→1(2,2) or 1→1→1 via diag
    // Diagonal from (0,0) to (1,1) costs 1+5=6, then to (2,2) costs +1=7
    // But we can go 1→1→2→1 = 5 if diagonal (0,0)→(1,1) costs min path to (1,1)
    // Let me trace: dp[0][0]=1, dp[0][1]=4, dp[0][2]=5
    //               dp[1][0]=2, dp[1][1]=min(dp[0][1]=4,dp[1][0]=2,dp[0][0]=1)+5=1+5=6
    //               dp[1][2]=min(dp[0][2]=5,dp[1][1]=6,dp[0][1]=4)+1=4+1=5
    //               dp[2][0]=6, dp[2][1]=min(dp[1][1]=6,dp[2][0]=6,dp[1][0]=2)+2=2+2=4
    //               dp[2][2]=min(dp[1][2]=5,dp[2][1]=4,dp[1][1]=6)+1=4+1=5
    try testing.expectEqual(@as(i32, 5), result);
}

test "maxPathSum - basic 3x3" {
    const grid = [_][3]i32{
        .{ 1, 3, 1 },
        .{ 1, 5, 1 },
        .{ 4, 2, 1 },
    };
    const grid_ptrs = [_][]const i32{
        &grid[0],
        &grid[1],
        &grid[2],
    };

    const result = try maxPathSum(i32, testing.allocator, &grid_ptrs);
    try testing.expectEqual(@as(i32, 12), result); // 1→3→5→2→1
}

test "minPathSum - negative values" {
    const grid = [_][3]i32{
        .{ -1, -3, -1 },
        .{ -1, -5, -1 },
        .{ -4, -2, -1 },
    };
    const grid_ptrs = [_][]const i32{
        &grid[0],
        &grid[1],
        &grid[2],
    };

    const result = try minPathSum(i32, testing.allocator, &grid_ptrs);
    // Path: -1→-3→-1→-1→-1 = -7 would be best, but we can only go right/down
    // Actual path: -1→-1→-5→-2→-1 = -10 or -1→-3→-5→-2→-1 = -12
    // DP computes: -1→-3→-1→-1→-1 going through (0,1)→(1,1) which is min(-4,-2)+(-5)=-9
    // Final: min path is -1→-3→-5→-2→-1 = -12
    try testing.expectEqual(@as(i32, -12), result);
}

test "minPathSum - mixed values" {
    const grid = [_][3]i32{
        .{ 5, -2, 3 },
        .{ -1, 4, -2 },
        .{ 3, -1, 2 },
    };
    const grid_ptrs = [_][]const i32{
        &grid[0],
        &grid[1],
        &grid[2],
    };

    const result = try minPathSum(i32, testing.allocator, &grid_ptrs);
    // 5→-2→3→-2→2 = 6 or 5→-1→4→-1→2 = 9 or 5→-1→3→-1→2 = 8
    // Trace: dp[0][0]=5, dp[0][1]=3, dp[0][2]=6
    //        dp[1][0]=4, dp[1][1]=min(3,4)+4=7, dp[1][2]=min(6,7)-2=4
    //        dp[2][0]=7, dp[2][1]=min(7,7)-1=6, dp[2][2]=min(4,6)+2=6
    try testing.expectEqual(@as(i32, 6), result);
}

test "minPathSum - f64 support" {
    const grid = [_][2]f64{
        .{ 1.5, 2.5 },
        .{ 3.5, 4.5 },
    };
    const grid_ptrs = [_][]const f64{
        &grid[0],
        &grid[1],
    };

    const result = try minPathSum(f64, testing.allocator, &grid_ptrs);
    try testing.expectEqual(@as(f64, 8.5), result); // 1.5→2.5→4.5
}

test "minPathSum - large values" {
    const grid = [_][3]i32{
        .{ 1000, 2000, 3000 },
        .{ 4000, 5000, 6000 },
        .{ 7000, 8000, 9000 },
    };
    const grid_ptrs = [_][]const i32{
        &grid[0],
        &grid[1],
        &grid[2],
    };

    const result = try minPathSum(i32, testing.allocator, &grid_ptrs);
    try testing.expectEqual(@as(i32, 21000), result); // 1000→2000→3000→6000→9000
}

test "minPathSum - memory safety" {
    const grid = [_][3]i32{
        .{ 1, 3, 1 },
        .{ 1, 5, 1 },
        .{ 4, 2, 1 },
    };
    const grid_ptrs = [_][]const i32{
        &grid[0],
        &grid[1],
        &grid[2],
    };

    _ = try minPathSum(i32, testing.allocator, &grid_ptrs);
    _ = try minPathSumOptimized(i32, testing.allocator, &grid_ptrs);
    const result = try minPathSumWithPath(i32, testing.allocator, &grid_ptrs);
    defer testing.allocator.free(result.path);
}
