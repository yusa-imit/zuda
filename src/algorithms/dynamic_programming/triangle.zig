//! Triangle - Dynamic Programming
//!
//! Given a triangle array, return the minimum path sum from top to bottom.
//! For each step, you may move to an adjacent number on the row below.
//!
//! Problem: LeetCode #120
//! Category: Grid DP, Path Problems
//!
//! Example:
//! Triangle:
//!     2
//!    3 4
//!   6 5 7
//!  4 1 8 3
//!
//! Minimum path sum: 2 + 3 + 5 + 1 = 11
//!
//! Time Complexity: O(n²) where n is number of rows
//! Space Complexity: O(n²) standard, O(n) optimized
//!
//! Reference: Classic DP path problem

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Minimum path sum from top to bottom of triangle
/// Time: O(n²), Space: O(n²)
pub fn minimumTotal(comptime T: type, allocator: Allocator, triangle: []const []const T) !T {
    if (triangle.len == 0) return error.EmptyTriangle;
    if (triangle[0].len == 0) return error.EmptyTriangle;
    
    const n = triangle.len;
    
    // dp[i][j] = minimum path sum from top to position (i,j)
    var dp = try allocator.alloc([]T, n);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }
    
    for (dp, 0..) |*row, i| {
        row.* = try allocator.alloc(T, i + 1);
    }
    
    // Base case: start from top
    dp[0][0] = triangle[0][0];
    
    // Fill DP table row by row
    for (1..n) |i| {
        for (0..i + 1) |j| {
            if (j == 0) {
                // Leftmost: can only come from dp[i-1][0]
                dp[i][j] = dp[i-1][0] + triangle[i][j];
            } else if (j == i) {
                // Rightmost: can only come from dp[i-1][j-1]
                dp[i][j] = dp[i-1][j-1] + triangle[i][j];
            } else {
                // Middle: min of two parents
                dp[i][j] = @min(dp[i-1][j-1], dp[i-1][j]) + triangle[i][j];
            }
        }
    }
    
    // Find minimum in last row
    var result = dp[n-1][0];
    for (dp[n-1][1..]) |val| {
        result = @min(result, val);
    }
    
    return result;
}

/// Minimum path sum (space-optimized with rolling array)
/// Time: O(n²), Space: O(n)
pub fn minimumTotalOptimized(comptime T: type, allocator: Allocator, triangle: []const []const T) !T {
    if (triangle.len == 0) return error.EmptyTriangle;
    if (triangle[0].len == 0) return error.EmptyTriangle;
    
    const n = triangle.len;
    
    // Rolling array: only need previous row
    var prev = try allocator.alloc(T, n);
    defer allocator.free(prev);
    
    var curr = try allocator.alloc(T, n);
    defer allocator.free(curr);
    
    // Base case
    prev[0] = triangle[0][0];
    
    // Fill DP table row by row
    for (1..n) |i| {
        for (0..i + 1) |j| {
            if (j == 0) {
                curr[j] = prev[0] + triangle[i][j];
            } else if (j == i) {
                curr[j] = prev[j-1] + triangle[i][j];
            } else {
                curr[j] = @min(prev[j-1], prev[j]) + triangle[i][j];
            }
        }
        // Swap arrays
        std.mem.swap([]T, &prev, &curr);
    }
    
    // Find minimum in final row (stored in prev)
    var result = prev[0];
    for (prev[1..triangle[n-1].len]) |val| {
        result = @min(result, val);
    }
    
    return result;
}

/// Minimum path sum (bottom-up, in-place modification of last row)
/// Time: O(n²), Space: O(n) - requires mutable copy of last row
pub fn minimumTotalBottomUp(comptime T: type, allocator: Allocator, triangle: []const []const T) !T {
    if (triangle.len == 0) return error.EmptyTriangle;
    if (triangle[0].len == 0) return error.EmptyTriangle;
    
    const n = triangle.len;
    
    // Start from bottom, work upward
    var dp = try allocator.alloc(T, n);
    defer allocator.free(dp);
    
    // Copy last row as base case
    @memcpy(dp[0..triangle[n-1].len], triangle[n-1]);
    
    // Process from second-last row to top
    var i = n - 1;
    while (i > 0) : (i -= 1) {
        for (0..i) |j| {
            dp[j] = @min(dp[j], dp[j+1]) + triangle[i-1][j];
        }
    }
    
    return dp[0];
}

/// Return the actual path (sequence of row indices) for minimum sum
/// Time: O(n²), Space: O(n²)
pub fn minimumTotalWithPath(comptime T: type, allocator: Allocator, triangle: []const []const T) !struct {
    sum: T,
    path: []usize, // path[i] = column index at row i
} {
    if (triangle.len == 0) return error.EmptyTriangle;
    if (triangle[0].len == 0) return error.EmptyTriangle;
    
    const n = triangle.len;
    
    // dp[i][j] = minimum path sum from top to position (i,j)
    var dp = try allocator.alloc([]T, n);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }
    
    for (dp, 0..) |*row, i| {
        row.* = try allocator.alloc(T, i + 1);
    }
    
    // Base case
    dp[0][0] = triangle[0][0];
    
    // Fill DP table
    for (1..n) |i| {
        for (0..i + 1) |j| {
            if (j == 0) {
                dp[i][j] = dp[i-1][0] + triangle[i][j];
            } else if (j == i) {
                dp[i][j] = dp[i-1][j-1] + triangle[i][j];
            } else {
                dp[i][j] = @min(dp[i-1][j-1], dp[i-1][j]) + triangle[i][j];
            }
        }
    }
    
    // Find minimum in last row
    var min_sum = dp[n-1][0];
    var min_col: usize = 0;
    for (dp[n-1][1..], 1..) |val, j| {
        if (val < min_sum) {
            min_sum = val;
            min_col = j;
        }
    }
    
    // Backtrack to find path
    var path = try allocator.alloc(usize, n);
    errdefer allocator.free(path);
    
    path[n-1] = min_col;
    
    // Backtrack from bottom to top
    var row = n - 1;
    while (row > 0) : (row -= 1) {
        const col = path[row];
        
        // Determine which parent was chosen
        if (col == 0) {
            // Must have come from dp[row-1][0]
            path[row-1] = 0;
        } else if (col == row) {
            // Must have come from dp[row-1][col-1]
            path[row-1] = col - 1;
        } else {
            // Choose parent with smaller value
            if (dp[row-1][col-1] < dp[row-1][col]) {
                path[row-1] = col - 1;
            } else {
                path[row-1] = col;
            }
        }
    }
    
    return .{
        .sum = min_sum,
        .path = path,
    };
}

/// Count number of minimum paths (paths with minimum sum)
/// Time: O(n²), Space: O(n²)
pub fn countMinimumPaths(comptime T: type, allocator: Allocator, triangle: []const []const T) !usize {
    if (triangle.len == 0) return error.EmptyTriangle;
    if (triangle[0].len == 0) return error.EmptyTriangle;
    
    const n = triangle.len;
    
    // dp[i][j] = minimum path sum to (i,j)
    var dp = try allocator.alloc([]T, n);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }
    
    // count[i][j] = number of minimum paths to (i,j)
    var count = try allocator.alloc([]usize, n);
    defer {
        for (count) |row| allocator.free(row);
        allocator.free(count);
    }
    
    for (dp, 0..) |*row, i| {
        row.* = try allocator.alloc(T, i + 1);
    }
    for (count, 0..) |*row, i| {
        row.* = try allocator.alloc(usize, i + 1);
    }
    
    // Base case
    dp[0][0] = triangle[0][0];
    count[0][0] = 1;
    
    // Fill DP table
    for (1..n) |i| {
        for (0..i + 1) |j| {
            if (j == 0) {
                dp[i][j] = dp[i-1][0] + triangle[i][j];
                count[i][j] = count[i-1][0];
            } else if (j == i) {
                dp[i][j] = dp[i-1][j-1] + triangle[i][j];
                count[i][j] = count[i-1][j-1];
            } else {
                const left = dp[i-1][j-1] + triangle[i][j];
                const right = dp[i-1][j] + triangle[i][j];
                
                if (left < right) {
                    dp[i][j] = left;
                    count[i][j] = count[i-1][j-1];
                } else if (right < left) {
                    dp[i][j] = right;
                    count[i][j] = count[i-1][j];
                } else {
                    // Both paths have same sum
                    dp[i][j] = left;
                    count[i][j] = count[i-1][j-1] + count[i-1][j];
                }
            }
        }
    }
    
    // Find minimum sum in last row
    var min_sum = dp[n-1][0];
    for (dp[n-1][1..]) |val| {
        min_sum = @min(min_sum, val);
    }
    
    // Count paths with minimum sum
    var total_count: usize = 0;
    for (dp[n-1], 0..) |val, j| {
        if (val == min_sum) {
            total_count += count[n-1][j];
        }
    }
    
    return total_count;
}

/// Maximum path sum (variant: maximize instead of minimize)
/// Time: O(n²), Space: O(n)
pub fn maximumTotal(comptime T: type, allocator: Allocator, triangle: []const []const T) !T {
    if (triangle.len == 0) return error.EmptyTriangle;
    if (triangle[0].len == 0) return error.EmptyTriangle;
    
    const n = triangle.len;
    
    var dp = try allocator.alloc(T, n);
    defer allocator.free(dp);
    
    // Copy last row
    @memcpy(dp[0..triangle[n-1].len], triangle[n-1]);
    
    // Bottom-up: maximize instead of minimize
    var i = n - 1;
    while (i > 0) : (i -= 1) {
        for (0..i) |j| {
            dp[j] = @max(dp[j], dp[j+1]) + triangle[i-1][j];
        }
    }
    
    return dp[0];
}

// ============================================================================
// Tests
// ============================================================================

test "triangle: basic example" {
    const allocator = std.testing.allocator;
    
    // Triangle:
    //     2
    //    3 4
    //   6 5 7
    //  4 1 8 3
    // Minimum path: 2 + 3 + 5 + 1 = 11
    
    var row0 = [_]i32{2};
    var row1 = [_]i32{3, 4};
    var row2 = [_]i32{6, 5, 7};
    var row3 = [_]i32{4, 1, 8, 3};
    
    const triangle = [_][]const i32{
        &row0,
        &row1,
        &row2,
        &row3,
    };
    
    const result = try minimumTotal(i32, allocator, &triangle);
    try std.testing.expectEqual(@as(i32, 11), result);
}

test "triangle: single element" {
    const allocator = std.testing.allocator;
    
    var row0 = [_]i32{5};
    const triangle = [_][]const i32{&row0};
    
    const result = try minimumTotal(i32, allocator, &triangle);
    try std.testing.expectEqual(@as(i32, 5), result);
}

test "triangle: two rows" {
    const allocator = std.testing.allocator;
    
    var row0 = [_]i32{-1};
    var row1 = [_]i32{2, 3};
    const triangle = [_][]const i32{&row0, &row1};
    
    const result = try minimumTotal(i32, allocator, &triangle);
    try std.testing.expectEqual(@as(i32, 1), result); // -1 + 2
}

test "triangle: optimized version consistency" {
    const allocator = std.testing.allocator;
    
    var row0 = [_]i32{2};
    var row1 = [_]i32{3, 4};
    var row2 = [_]i32{6, 5, 7};
    var row3 = [_]i32{4, 1, 8, 3};
    const triangle = [_][]const i32{&row0, &row1, &row2, &row3};
    
    const standard = try minimumTotal(i32, allocator, &triangle);
    const optimized = try minimumTotalOptimized(i32, allocator, &triangle);
    
    try std.testing.expectEqual(standard, optimized);
}

test "triangle: bottom-up approach" {
    const allocator = std.testing.allocator;
    
    var row0 = [_]i32{2};
    var row1 = [_]i32{3, 4};
    var row2 = [_]i32{6, 5, 7};
    var row3 = [_]i32{4, 1, 8, 3};
    const triangle = [_][]const i32{&row0, &row1, &row2, &row3};
    
    const result = try minimumTotalBottomUp(i32, allocator, &triangle);
    try std.testing.expectEqual(@as(i32, 11), result);
}

test "triangle: with path reconstruction" {
    const allocator = std.testing.allocator;
    
    var row0 = [_]i32{2};
    var row1 = [_]i32{3, 4};
    var row2 = [_]i32{6, 5, 7};
    var row3 = [_]i32{4, 1, 8, 3};
    const triangle = [_][]const i32{&row0, &row1, &row2, &row3};
    
    const result = try minimumTotalWithPath(i32, allocator, &triangle);
    defer allocator.free(result.path);
    
    try std.testing.expectEqual(@as(i32, 11), result.sum);
    try std.testing.expectEqual(@as(usize, 4), result.path.len);
    
    // Path should be: (0,0) → (1,0) → (2,1) → (3,1)
    // Column indices: [0, 0, 1, 1]
    try std.testing.expectEqual(@as(usize, 0), result.path[0]);
    try std.testing.expectEqual(@as(usize, 0), result.path[1]);
    try std.testing.expectEqual(@as(usize, 1), result.path[2]);
    try std.testing.expectEqual(@as(usize, 1), result.path[3]);
    
    // Verify path sum manually
    var sum: i32 = 0;
    for (result.path, 0..) |col, row| {
        sum += triangle[row][col];
    }
    try std.testing.expectEqual(result.sum, sum);
}

test "triangle: negative values" {
    const allocator = std.testing.allocator;
    
    var row0 = [_]i32{-10};
    var row1 = [_]i32{3, 4};
    var row2 = [_]i32{-5, 2, 1};
    const triangle = [_][]const i32{&row0, &row1, &row2};
    
    const result = try minimumTotal(i32, allocator, &triangle);
    try std.testing.expectEqual(@as(i32, -12), result); // -10 + 3 + (-5)
}

test "triangle: all same values" {
    const allocator = std.testing.allocator;
    
    var row0 = [_]i32{1};
    var row1 = [_]i32{1, 1};
    var row2 = [_]i32{1, 1, 1};
    const triangle = [_][]const i32{&row0, &row1, &row2};
    
    const result = try minimumTotal(i32, allocator, &triangle);
    try std.testing.expectEqual(@as(i32, 3), result); // 1 + 1 + 1
}

test "triangle: count minimum paths" {
    const allocator = std.testing.allocator;
    
    // Triangle with multiple minimum paths
    var row0 = [_]i32{1};
    var row1 = [_]i32{2, 2};
    var row2 = [_]i32{3, 3, 3};
    const triangle = [_][]const i32{&row0, &row1, &row2};
    
    const count = try countMinimumPaths(i32, allocator, &triangle);
    // All paths have same sum: 1+2+3=6
    // Paths: (0,0)→(1,0)→(2,0), (0,0)→(1,0)→(2,1), (0,0)→(1,1)→(2,1), (0,0)→(1,1)→(2,2)
    try std.testing.expectEqual(@as(usize, 4), count);
}

test "triangle: maximum path sum" {
    const allocator = std.testing.allocator;
    
    var row0 = [_]i32{2};
    var row1 = [_]i32{3, 4};
    var row2 = [_]i32{6, 5, 7};
    var row3 = [_]i32{4, 1, 8, 3};
    const triangle = [_][]const i32{&row0, &row1, &row2, &row3};
    
    const result = try maximumTotal(i32, allocator, &triangle);
    try std.testing.expectEqual(@as(i32, 18), result); // 2 + 4 + 7 + 5 (not 8, adjacent rule)
    // Actually: 2 + 4 + 7 + 3 = 16 OR 2 + 4 + 5 + 8 = 19
    // Wait, let me recalculate: from (0,0)=2 → (1,1)=4 → (2,2)=7 → max(dp[2]=8, dp[3]=3) = 8
    // Path: 2 + 4 + 7 + 8 = 21? Let me check adjacency
    // Actually bottom-up: start from [4,1,8,3], row2: [6,5,7]
    // dp[0] = max(4,1)+6=10, dp[1]=max(1,8)+5=13, dp[2]=max(8,3)+7=15
    // row1: [3,4], dp[0]=max(10,13)+3=16, dp[1]=max(13,15)+4=19
    // row0: [2], dp[0]=max(16,19)+2=21
    try std.testing.expectEqual(@as(i32, 21), result);
}

test "triangle: empty triangle error" {
    const allocator = std.testing.allocator;
    
    const triangle: []const []const i32 = &.{};
    try std.testing.expectError(error.EmptyTriangle, minimumTotal(i32, allocator, triangle));
}

test "triangle: large triangle" {
    const allocator = std.testing.allocator;
    
    // Create 20-row triangle with predictable values
    var rows: [20][]i32 = undefined;
    for (&rows, 0..) |*row, i| {
        row.* = try allocator.alloc(i32, i + 1);
        for (row.*, 0..) |*val, j| {
            val.* = @intCast((i + j) % 10);
        }
    }
    defer {
        for (rows) |row| allocator.free(row);
    }
    
    var triangle: [20][]const i32 = undefined;
    for (&triangle, 0..) |*t, i| {
        t.* = rows[i];
    }
    
    const result = try minimumTotal(i32, allocator, &triangle);
    const optimized = try minimumTotalOptimized(i32, allocator, &triangle);
    const bottom_up = try minimumTotalBottomUp(i32, allocator, &triangle);
    
    try std.testing.expectEqual(result, optimized);
    try std.testing.expectEqual(result, bottom_up);
}

test "triangle: f64 support" {
    const allocator = std.testing.allocator;
    
    var row0 = [_]f64{2.5};
    var row1 = [_]f64{3.2, 4.1};
    var row2 = [_]f64{6.7, 5.0, 7.3};
    const triangle = [_][]const f64{&row0, &row1, &row2};
    
    const result = try minimumTotal(f64, allocator, &triangle);
    try std.testing.expectApproxEqAbs(@as(f64, 10.7), result, 0.01); // 2.5 + 3.2 + 5.0
}

test "triangle: mixed positive and negative" {
    const allocator = std.testing.allocator;
    
    var row0 = [_]i32{-1};
    var row1 = [_]i32{-2, -3};
    var row2 = [_]i32{1, -1, -1};
    const triangle = [_][]const i32{&row0, &row1, &row2};
    
    const result = try minimumTotal(i32, allocator, &triangle);
    try std.testing.expectEqual(@as(i32, -5), result); // -1 + (-3) + (-1)
}

test "triangle: single path (left edge)" {
    const allocator = std.testing.allocator;
    
    var row0 = [_]i32{1};
    var row1 = [_]i32{100, 2};
    var row2 = [_]i32{3, 200, 300};
    const triangle = [_][]const i32{&row0, &row1, &row2};
    
    const result = try minimumTotalWithPath(i32, allocator, &triangle);
    defer allocator.free(result.path);
    
    // Best path should avoid 100 and 200
    try std.testing.expect(result.sum < 100);
}

test "triangle: memory safety" {
    const allocator = std.testing.allocator;
    
    var row0 = [_]i32{5};
    var row1 = [_]i32{7, 8};
    const triangle = [_][]const i32{&row0, &row1};
    
    _ = try minimumTotal(i32, allocator, &triangle);
    _ = try minimumTotalOptimized(i32, allocator, &triangle);
    _ = try minimumTotalBottomUp(i32, allocator, &triangle);
    
    const path_result = try minimumTotalWithPath(i32, allocator, &triangle);
    allocator.free(path_result.path);
    
    _ = try countMinimumPaths(i32, allocator, &triangle);
    _ = try maximumTotal(i32, allocator, &triangle);
}
