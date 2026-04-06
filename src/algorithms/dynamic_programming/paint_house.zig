const std = @import("std");
const testing = std.testing;

/// Paint House - Minimum cost to paint n houses with k colors where adjacent houses can't have same color
///
/// Classic dynamic programming problem:
/// - Given n houses and k colors with cost[i][j] = cost to paint house i with color j
/// - No two adjacent houses can have the same color
/// - Find minimum total cost to paint all houses
///
/// Example: 3 houses, 3 colors
/// costs = [[17,2,17], [16,16,5], [14,3,19]]
/// Paint house 0 with color 1 (cost 2)
/// Paint house 1 with color 2 (cost 5)
/// Paint house 2 with color 1 (cost 3)
/// Total cost = 2 + 5 + 3 = 10
///
/// Reference: LeetCode #256 (Paint House), #265 (Paint House II)

/// Result type for paint house with path reconstruction
pub fn PaintResult(comptime T: type) type {
    return struct {
        min_cost: T,
        colors: ?std.ArrayList(usize) = null, // Color choice for each house (0-indexed)
        allocator: ?std.mem.Allocator = null,

        pub fn deinit(self: *@This()) void {
            if (self.colors) |*c| {
                if (self.allocator) |alloc| {
                    c.deinit(alloc);
                }
            }
        }
    };
}

/// Paint House - Standard 3-color problem (Red, Blue, Green)
/// Time: O(n) where n = number of houses
/// Space: O(1) - only tracks previous row
pub fn paintHouse(comptime T: type, costs: []const [3]T) !T {
    if (costs.len == 0) return error.EmptyInput;

    // dp[i][c] = minimum cost to paint house i with color c
    // We only need previous row, so we use rolling variables
    var prev: [3]T = costs[0]; // Base case: first house

    // For each subsequent house
    var i: usize = 1;
    while (i < costs.len) : (i += 1) {
        const curr = [3]T{
            costs[i][0] + @min(prev[1], prev[2]), // Red: choose min of Blue/Green from previous
            costs[i][1] + @min(prev[0], prev[2]), // Blue: choose min of Red/Green from previous
            costs[i][2] + @min(prev[0], prev[1]), // Green: choose min of Red/Blue from previous
        };
        prev = curr;
    }

    // Return minimum of last house's three options
    return @min(@min(prev[0], prev[1]), prev[2]);
}

/// Paint House - Generalized k-color problem
/// Time: O(n × k²) where n = houses, k = colors
/// Space: O(k) - rolling array
pub fn paintHouseK(comptime T: type, allocator: std.mem.Allocator, costs: []const []const T) !T {
    if (costs.len == 0) return error.EmptyInput;
    if (costs[0].len == 0) return error.EmptyColors;
    if (costs[0].len == 1 and costs.len > 1) return error.InsufficientColors; // Can't paint adjacent houses with same color

    const n = costs.len;
    const k = costs[0].len;

    // Verify all rows have same number of colors
    for (costs) |row| {
        if (row.len != k) return error.InconsistentColors;
    }

    // dp[c] = minimum cost to paint current house with color c
    var prev = try allocator.alloc(T, k);
    defer allocator.free(prev);
    var curr = try allocator.alloc(T, k);
    defer allocator.free(curr);

    // Base case: first house
    @memcpy(prev, costs[0]);

    // For each subsequent house
    var i: usize = 1;
    while (i < n) : (i += 1) {
        // For each color choice for current house
        var c: usize = 0;
        while (c < k) : (c += 1) {
            // Find minimum cost from previous house excluding color c
            var min_cost: T = std.math.maxInt(T);
            var prev_c: usize = 0;
            while (prev_c < k) : (prev_c += 1) {
                if (prev_c != c) {
                    min_cost = @min(min_cost, prev[prev_c]);
                }
            }
            curr[c] = costs[i][c] + min_cost;
        }

        // Swap arrays
        std.mem.swap([]T, &prev, &curr);
    }

    // Find minimum cost in last row
    var result: T = prev[0];
    for (prev[1..]) |cost| {
        result = @min(result, cost);
    }
    return result;
}

/// Paint House - Optimized k-color with O(n×k) time
/// Tracks minimum and second minimum from previous row
/// Time: O(n × k)
/// Space: O(k)
pub fn paintHouseKOptimized(comptime T: type, _: std.mem.Allocator, costs: []const []const T) !T {
    if (costs.len == 0) return error.EmptyInput;
    if (costs[0].len == 0) return error.EmptyColors;
    if (costs[0].len == 1 and costs.len > 1) return error.InsufficientColors;

    const n = costs.len;
    const k = costs[0].len;

    // Verify all rows have same number of colors
    for (costs) |row| {
        if (row.len != k) return error.InconsistentColors;
    }

    // Track minimum and second minimum from previous row
    var prev_min: T = 0;
    var prev_min_idx: usize = std.math.maxInt(usize);
    var prev_second_min: T = 0;

    // For each house
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var curr_min: T = std.math.maxInt(T);
        var curr_min_idx: usize = 0;
        var curr_second_min: T = std.math.maxInt(T);

        // For each color
        var c: usize = 0;
        while (c < k) : (c += 1) {
            // If this color was the min in previous row, use second min
            // Otherwise use min
            const prev_cost = if (i == 0) @as(T, 0) else if (c == prev_min_idx) prev_second_min else prev_min;
            const curr_cost = costs[i][c] + prev_cost;

            // Update current min and second min
            if (curr_cost < curr_min) {
                curr_second_min = curr_min;
                curr_min = curr_cost;
                curr_min_idx = c;
            } else if (curr_cost < curr_second_min) {
                curr_second_min = curr_cost;
            }
        }

        prev_min = curr_min;
        prev_min_idx = curr_min_idx;
        prev_second_min = curr_second_min;
    }

    return prev_min;
}

/// Paint House with path reconstruction
/// Time: O(n × k²)
/// Space: O(n × k) - full DP table needed for backtracking
pub fn paintHouseWithPath(comptime T: type, allocator: std.mem.Allocator, costs: []const []const T) !PaintResult(T) {
    if (costs.len == 0) return error.EmptyInput;
    if (costs[0].len == 0) return error.EmptyColors;
    if (costs[0].len == 1 and costs.len > 1) return error.InsufficientColors;

    const n = costs.len;
    const k = costs[0].len;

    // Verify all rows have same number of colors
    for (costs) |row| {
        if (row.len != k) return error.InconsistentColors;
    }

    // Full DP table for backtracking
    var dp = try allocator.alloc([]T, n);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }
    for (dp, 0..) |*row, i| {
        row.* = try allocator.alloc(T, k);
        if (i == 0) {
            @memcpy(row.*, costs[0]);
        }
    }

    // Parent tracking for backtracking
    var parent = try allocator.alloc([]usize, n);
    defer {
        for (parent) |row| allocator.free(row);
        allocator.free(parent);
    }
    for (parent) |*row| {
        row.* = try allocator.alloc(usize, k);
        @memset(row.*, 0);
    }

    // Fill DP table
    var i: usize = 1;
    while (i < n) : (i += 1) {
        var c: usize = 0;
        while (c < k) : (c += 1) {
            // Find minimum cost from previous house excluding color c
            var min_cost: T = std.math.maxInt(T);
            var min_prev_color: usize = 0;
            var prev_c: usize = 0;
            while (prev_c < k) : (prev_c += 1) {
                if (prev_c != c and dp[i - 1][prev_c] < min_cost) {
                    min_cost = dp[i - 1][prev_c];
                    min_prev_color = prev_c;
                }
            }
            dp[i][c] = costs[i][c] + min_cost;
            parent[i][c] = min_prev_color;
        }
    }

    // Find minimum cost and color in last row
    var min_cost: T = dp[n - 1][0];
    var last_color: usize = 0;
    for (dp[n - 1][1..], 1..) |cost, c| {
        if (cost < min_cost) {
            min_cost = cost;
            last_color = c;
        }
    }

    // Backtrack to reconstruct path
    var colors = try std.ArrayList(usize).initCapacity(allocator, n);
    errdefer colors.deinit(allocator);
    try colors.resize(allocator, n);

    colors.items[n - 1] = last_color;
    i = n - 1;
    while (i > 0) : (i -= 1) {
        colors.items[i - 1] = parent[i][colors.items[i]];
    }

    return PaintResult(T){
        .min_cost = min_cost,
        .colors = colors,
        .allocator = allocator,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "paint house - basic 3 colors" {
    const costs = [_][3]i32{
        [_]i32{ 17, 2, 17 },
        [_]i32{ 16, 16, 5 },
        [_]i32{ 14, 3, 19 },
    };
    const result = try paintHouse(i32, &costs);
    try testing.expectEqual(@as(i32, 10), result); // 2 + 5 + 3
}

test "paint house - single house" {
    const costs = [_][3]i32{
        [_]i32{ 5, 10, 15 },
    };
    const result = try paintHouse(i32, &costs);
    try testing.expectEqual(@as(i32, 5), result);
}

test "paint house - two houses" {
    const costs = [_][3]i32{
        [_]i32{ 1, 2, 3 },
        [_]i32{ 4, 5, 6 },
    };
    const result = try paintHouse(i32, &costs);
    try testing.expectEqual(@as(i32, 6), result); // 1 + 5
}

test "paint house - all same costs" {
    const costs = [_][3]i32{
        [_]i32{ 10, 10, 10 },
        [_]i32{ 10, 10, 10 },
        [_]i32{ 10, 10, 10 },
    };
    const result = try paintHouse(i32, &costs);
    try testing.expectEqual(@as(i32, 30), result);
}

test "paint house - empty input" {
    const costs = [_][3]i32{};
    try testing.expectError(error.EmptyInput, paintHouse(i32, &costs));
}

test "paint house - k colors basic" {
    const allocator = testing.allocator;
    const costs = [_][]const i32{
        &[_]i32{ 17, 2, 17 },
        &[_]i32{ 16, 16, 5 },
        &[_]i32{ 14, 3, 19 },
    };
    const result = try paintHouseK(i32, allocator, &costs);
    try testing.expectEqual(@as(i32, 10), result);
}

test "paint house - k colors 4 colors" {
    const allocator = testing.allocator;
    const costs = [_][]const i32{
        &[_]i32{ 1, 5, 3, 7 },
        &[_]i32{ 5, 8, 2, 4 },
        &[_]i32{ 3, 2, 9, 1 },
    };
    const result = try paintHouseK(i32, allocator, &costs);
    // House 0: color 0 (cost 1)
    // House 1: color 2 (cost 2, can't use 0)
    // House 2: color 3 (cost 1, can't use 2)
    try testing.expectEqual(@as(i32, 4), result); // 1 + 2 + 1
}

test "paint house - k colors single color" {
    const allocator = testing.allocator;
    const costs = [_][]const i32{
        &[_]i32{10},
        &[_]i32{20},
    };
    try testing.expectError(error.InsufficientColors, paintHouseK(i32, allocator, &costs));
}

test "paint house - k colors empty colors" {
    const allocator = testing.allocator;
    const costs = [_][]const i32{
        &[_]i32{},
    };
    try testing.expectError(error.EmptyColors, paintHouseK(i32, allocator, &costs));
}

test "paint house - k colors inconsistent" {
    const allocator = testing.allocator;
    const costs = [_][]const i32{
        &[_]i32{ 1, 2, 3 },
        &[_]i32{ 4, 5 }, // Wrong size
    };
    try testing.expectError(error.InconsistentColors, paintHouseK(i32, allocator, &costs));
}

test "paint house - optimized k colors basic" {
    const allocator = testing.allocator;
    const costs = [_][]const i32{
        &[_]i32{ 17, 2, 17 },
        &[_]i32{ 16, 16, 5 },
        &[_]i32{ 14, 3, 19 },
    };
    const result = try paintHouseKOptimized(i32, allocator, &costs);
    try testing.expectEqual(@as(i32, 10), result);
}

test "paint house - optimized vs standard consistency" {
    const allocator = testing.allocator;
    const costs = [_][]const i32{
        &[_]i32{ 1, 5, 3, 7, 2 },
        &[_]i32{ 5, 8, 2, 4, 9 },
        &[_]i32{ 3, 2, 9, 1, 6 },
        &[_]i32{ 7, 4, 1, 8, 3 },
    };
    const standard = try paintHouseK(i32, allocator, &costs);
    const optimized = try paintHouseKOptimized(i32, allocator, &costs);
    try testing.expectEqual(standard, optimized);
}

test "paint house - with path basic" {
    const allocator = testing.allocator;
    const costs = [_][]const i32{
        &[_]i32{ 17, 2, 17 },
        &[_]i32{ 16, 16, 5 },
        &[_]i32{ 14, 3, 19 },
    };
    var result = try paintHouseWithPath(i32, allocator, &costs);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 10), result.min_cost);
    try testing.expect(result.colors != null);
    const colors = result.colors.?.items;
    try testing.expectEqual(@as(usize, 3), colors.len);
    try testing.expectEqual(@as(usize, 1), colors[0]); // Color 1 (cost 2)
    try testing.expectEqual(@as(usize, 2), colors[1]); // Color 2 (cost 5)
    try testing.expectEqual(@as(usize, 1), colors[2]); // Color 1 (cost 3)

    // Verify total cost
    var total: i32 = 0;
    for (colors, 0..) |color, i| {
        total += costs[i][color];
    }
    try testing.expectEqual(result.min_cost, total);
}

test "paint house - with path 4 colors" {
    const allocator = testing.allocator;
    const costs = [_][]const i32{
        &[_]i32{ 1, 5, 3, 7 },
        &[_]i32{ 5, 8, 2, 4 },
        &[_]i32{ 3, 2, 9, 1 },
    };
    var result = try paintHouseWithPath(i32, allocator, &costs);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 4), result.min_cost);
    const colors = result.colors.?.items;

    // Verify adjacent houses have different colors
    var i: usize = 0;
    while (i < colors.len - 1) : (i += 1) {
        try testing.expect(colors[i] != colors[i + 1]);
    }

    // Verify total cost
    var total: i32 = 0;
    for (colors, 0..) |color, idx| {
        total += costs[idx][color];
    }
    try testing.expectEqual(result.min_cost, total);
}

test "paint house - large number of houses" {
    const allocator = testing.allocator;
    const costs = try allocator.alloc([]i32, 50);
    defer allocator.free(costs);

    for (costs, 0..) |*row, i| {
        row.* = try allocator.alloc(i32, 3);
        row.*[0] = @intCast(i * 2 + 1);
        row.*[1] = @intCast(i * 2 + 2);
        row.*[2] = @intCast(i * 2 + 3);
    }
    defer {
        for (costs) |row| allocator.free(row);
    }

    const result = try paintHouseK(i32, allocator, costs);
    try testing.expect(result > 0);

    // Verify optimized matches
    const optimized = try paintHouseKOptimized(i32, allocator, costs);
    try testing.expectEqual(result, optimized);
}

test "paint house - f64 support" {
    const costs = [_][3]f64{
        [_]f64{ 17.5, 2.5, 17.5 },
        [_]f64{ 16.5, 16.5, 5.5 },
        [_]f64{ 14.5, 3.5, 19.5 },
    };
    const result = try paintHouse(f64, &costs);
    try testing.expectApproxEqAbs(@as(f64, 11.5), result, 1e-6);
}

test "paint house - memory safety" {
    const allocator = testing.allocator;
    const costs = [_][]const i32{
        &[_]i32{ 1, 2, 3, 4, 5 },
        &[_]i32{ 5, 4, 3, 2, 1 },
        &[_]i32{ 2, 3, 4, 5, 6 },
    };

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        _ = try paintHouseK(i32, allocator, &costs);
        _ = try paintHouseKOptimized(i32, allocator, &costs);
        var result = try paintHouseWithPath(i32, allocator, &costs);
        result.deinit();
    }
}
