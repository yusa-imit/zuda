const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Count the number of unique paths in an m×n grid from top-left to bottom-right.
/// Only right and down moves allowed.
///
/// Time: O(m×n)
/// Space: O(n) — rolling array optimization
///
/// Uses DP recurrence: paths[i][j] = paths[i-1][j] + paths[i][j-1]
/// Optimized to use single row by updating in-place.
///
/// Example:
/// ```
/// const allocator = std.testing.allocator;
/// const count = try uniquePaths(allocator, 3, 3);
/// // Returns 6 unique paths in 3×3 grid
/// ```
pub fn uniquePaths(allocator: Allocator, m: usize, n: usize) !u64 {
    if (m == 0 or n == 0) return 0;
    if (m == 1 or n == 1) return 1;

    // Single row DP: paths[j] represents paths to column j in current row
    var paths = try allocator.alloc(u64, n);
    defer allocator.free(paths);

    // Initialize first row: all cells have 1 path (straight right)
    for (paths) |*cell| {
        cell.* = 1;
    }

    // Fill remaining rows
    var i: usize = 1;
    while (i < m) : (i += 1) {
        var j: usize = 1;
        while (j < n) : (j += 1) {
            // Current cell = paths from above + paths from left
            paths[j] += paths[j - 1];
        }
    }

    return paths[n - 1];
}

/// Count unique paths with full DP table for visualization.
///
/// Time: O(m×n)
/// Space: O(m×n) — full 2D table
///
/// Returns a 2D table where table[i][j] represents the number of unique paths
/// to reach cell (i, j) from (0, 0).
///
/// Example:
/// ```
/// var table = try uniquePathsTable(allocator, 3, 3);
/// defer allocator.free(table);
/// // table = [[1, 1, 1], [1, 2, 3], [1, 3, 6]]
/// ```
pub fn uniquePathsTable(allocator: Allocator, m: usize, n: usize) ![][]u64 {
    if (m == 0 or n == 0) return &[_][]u64{};

    // Allocate 2D table
    var table = try allocator.alloc([]u64, m);
    errdefer {
        for (table) |row| allocator.free(row);
        allocator.free(table);
    }

    for (table) |*row| {
        row.* = try allocator.alloc(u64, n);
    }

    // Initialize first row and column
    for (table[0]) |*cell| cell.* = 1;
    for (table) |row| row[0] = 1;

    // Fill table
    var i: usize = 1;
    while (i < m) : (i += 1) {
        var j: usize = 1;
        while (j < n) : (j += 1) {
            table[i][j] = table[i - 1][j] + table[i][j - 1];
        }
    }

    return table;
}

/// Count unique paths in a grid with obstacles.
/// Obstacles are marked as 1, free cells as 0.
///
/// Time: O(m×n)
/// Space: O(n) — rolling array
///
/// Example:
/// ```
/// const allocator = std.testing.allocator;
/// const grid = [_][3]u8{
///     [_]u8{0, 0, 0},
///     [_]u8{0, 1, 0},  // obstacle at (1, 1)
///     [_]u8{0, 0, 0},
/// };
/// const count = try uniquePathsWithObstacles(allocator, &grid);
/// // Returns 2 (paths avoiding obstacle)
/// ```
pub fn uniquePathsWithObstacles(allocator: Allocator, comptime m: usize, comptime n: usize, grid: *const [m][n]u8) !u64 {
    if (m == 0 or n == 0) return 0;
    if (grid[0][0] == 1 or grid[m - 1][n - 1] == 1) return 0; // Start or end blocked

    var paths = try allocator.alloc(u64, n);
    defer allocator.free(paths);

    // Initialize first row
    paths[0] = if (grid[0][0] == 1) 0 else 1;
    for (1..n) |j| {
        paths[j] = if (grid[0][j] == 1) 0 else paths[j - 1];
    }

    // Fill remaining rows
    for (1..m) |i| {
        // First column
        if (grid[i][0] == 1) {
            paths[0] = 0;
        }

        for (1..n) |j| {
            if (grid[i][j] == 1) {
                paths[j] = 0;
            } else {
                paths[j] = paths[j] + paths[j - 1];
            }
        }
    }

    return paths[n - 1];
}

/// Find minimum path sum in a grid (top-left to bottom-right).
/// Each cell has a cost, find path with minimum total cost.
///
/// Time: O(m×n)
/// Space: O(n) — rolling array
///
/// Example:
/// ```
/// const allocator = std.testing.allocator;
/// const grid = [_][3]u32{
///     [_]u32{1, 3, 1},
///     [_]u32{1, 5, 1},
///     [_]u32{4, 2, 1},
/// };
/// const cost = try minPathSum(u32, allocator, &grid);
/// // Returns 7 (path: 1→3→1→1→1)
/// ```
pub fn minPathSum(comptime T: type, allocator: Allocator, comptime m: usize, comptime n: usize, grid: *const [m][n]T) !T {
    if (m == 0 or n == 0) return 0;

    var costs = try allocator.alloc(T, n);
    defer allocator.free(costs);

    // Initialize first row (cumulative sum)
    costs[0] = grid[0][0];
    for (1..n) |j| {
        costs[j] = costs[j - 1] + grid[0][j];
    }

    // Fill remaining rows
    for (1..m) |i| {
        costs[0] += grid[i][0]; // First column
        for (1..n) |j| {
            costs[j] = @min(costs[j], costs[j - 1]) + grid[i][j];
        }
    }

    return costs[n - 1];
}

/// Count paths with exactly k steps.
/// 3D DP: dp[i][j][steps] = number of paths to (i,j) using exactly 'steps' moves.
///
/// Time: O(m×n×k)
/// Space: O(m×n×k)
///
/// Example:
/// ```
/// var result = try uniquePathsExact(allocator, 3, 3, 4);
/// defer result.table.deinit();
/// // Returns paths using exactly 4 moves (right+down = 4)
/// ```
/// Count paths with exactly k steps - simplified version.
///
/// Time: O(m×n×k)
/// Space: O(k) — rolling array for last row
///
/// Note: This is a simpler version that returns only the count,
/// not the full 3D table (which would be memory-intensive).
pub fn uniquePathsExact(allocator: Allocator, m: usize, n: usize, k: usize) !u64 {
    if (m == 0 or n == 0) return 0;
    // Minimum steps needed is m+n-2 (m-1 down, n-1 right)
    if (k < m + n - 2) return 0;
    // For exactly minimum steps, it's the standard uniquePaths problem
    if (k == m + n - 2) return try uniquePaths(allocator, m, n);

    // For k > minimum, we need extra back-and-forth moves
    // which is not possible with only right/down moves
    return 0;
}

// ============================================================================
// Tests
// ============================================================================

test "unique paths: basic 1×1 grid" {
    const allocator = testing.allocator;
    try testing.expectEqual(1, try uniquePaths(allocator, 1, 1));
}

test "unique paths: basic 2×2 grid" {
    const allocator = testing.allocator;
    try testing.expectEqual(2, try uniquePaths(allocator, 2, 2));
}

test "unique paths: 3×3 grid" {
    const allocator = testing.allocator;
    try testing.expectEqual(6, try uniquePaths(allocator, 3, 3));
}

test "unique paths: 4×5 grid" {
    const allocator = testing.allocator;
    try testing.expectEqual(35, try uniquePaths(allocator, 4, 5));
}

test "unique paths: 1×n grid (straight line)" {
    const allocator = testing.allocator;
    try testing.expectEqual(1, try uniquePaths(allocator, 1, 10));
}

test "unique paths: m×1 grid (straight line)" {
    const allocator = testing.allocator;
    try testing.expectEqual(1, try uniquePaths(allocator, 10, 1));
}

test "unique paths: empty grid" {
    const allocator = testing.allocator;
    try testing.expectEqual(0, try uniquePaths(allocator, 0, 5));
    try testing.expectEqual(0, try uniquePaths(allocator, 5, 0));
}

test "unique paths: large grid" {
    const allocator = testing.allocator;
    try testing.expectEqual(48620, try uniquePaths(allocator, 10, 10));
}

test "unique paths table: basic 3×3" {
    const allocator = testing.allocator;
    const table = try uniquePathsTable(allocator, 3, 3);
    defer {
        for (table) |row| allocator.free(row);
        allocator.free(table);
    }

    try testing.expectEqual(1, table[0][0]);
    try testing.expectEqual(1, table[0][1]);
    try testing.expectEqual(1, table[0][2]);
    try testing.expectEqual(1, table[1][0]);
    try testing.expectEqual(2, table[1][1]);
    try testing.expectEqual(3, table[1][2]);
    try testing.expectEqual(1, table[2][0]);
    try testing.expectEqual(3, table[2][1]);
    try testing.expectEqual(6, table[2][2]);
}

test "unique paths with obstacles: basic" {
    const allocator = testing.allocator;
    const grid = [_][3]u8{
        [_]u8{ 0, 0, 0 },
        [_]u8{ 0, 1, 0 },
        [_]u8{ 0, 0, 0 },
    };
    try testing.expectEqual(2, try uniquePathsWithObstacles(allocator, 3, 3, &grid));
}

test "unique paths with obstacles: start blocked" {
    const allocator = testing.allocator;
    const grid = [_][2]u8{
        [_]u8{ 1, 0 },
        [_]u8{ 0, 0 },
    };
    try testing.expectEqual(0, try uniquePathsWithObstacles(allocator, 2, 2, &grid));
}

test "unique paths with obstacles: end blocked" {
    const allocator = testing.allocator;
    const grid = [_][2]u8{
        [_]u8{ 0, 0 },
        [_]u8{ 0, 1 },
    };
    try testing.expectEqual(0, try uniquePathsWithObstacles(allocator, 2, 2, &grid));
}

test "unique paths with obstacles: middle path blocked" {
    const allocator = testing.allocator;
    const grid = [_][3]u8{
        [_]u8{ 0, 0, 0 },
        [_]u8{ 0, 0, 1 },
        [_]u8{ 0, 0, 0 },
    };
    try testing.expectEqual(3, try uniquePathsWithObstacles(allocator, 3, 3, &grid));
}

test "min path sum: basic 3×3" {
    const allocator = testing.allocator;
    const grid = [_][3]u32{
        [_]u32{ 1, 3, 1 },
        [_]u32{ 1, 5, 1 },
        [_]u32{ 4, 2, 1 },
    };
    try testing.expectEqual(7, try minPathSum(u32, allocator, 3, 3, &grid));
}

test "min path sum: uniform costs" {
    const allocator = testing.allocator;
    const grid = [_][2]u32{
        [_]u32{ 1, 1 },
        [_]u32{ 1, 1 },
    };
    try testing.expectEqual(3, try minPathSum(u32, allocator, 2, 2, &grid));
}

test "min path sum: increasing costs" {
    const allocator = testing.allocator;
    const grid = [_][3]u32{
        [_]u32{ 1, 2, 3 },
        [_]u32{ 2, 3, 4 },
        [_]u32{ 3, 4, 5 },
    };
    try testing.expectEqual(15, try minPathSum(u32, allocator, 3, 3, &grid));
}

test "unique paths exact: exactly 4 steps in 3×3" {
    const allocator = testing.allocator;
    // Minimum steps in 3×3 grid is 4 (2 right + 2 down)
    const count = try uniquePathsExact(allocator, 3, 3, 4);
    try testing.expectEqual(6, count); // All paths use exactly 4 steps
}

test "unique paths exact: too few steps" {
    const allocator = testing.allocator;
    // Need at least 4 steps (m+n-2) for 3×3 grid
    const count = try uniquePathsExact(allocator, 3, 3, 3);
    try testing.expectEqual(0, count);
}

test "unique paths exact: more than minimum steps" {
    const allocator = testing.allocator;
    // With only right/down moves, can't use more than minimum steps
    const count = try uniquePathsExact(allocator, 3, 3, 5);
    try testing.expectEqual(0, count);
}
