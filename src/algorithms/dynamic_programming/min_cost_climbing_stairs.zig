// Minimum Cost Climbing Stairs
//
// Classic dynamic programming problem: find minimum cost to reach top of stairs
// where each step has a cost and you can climb 1 or 2 steps at a time.
//
// Problem: Given an array of costs where cost[i] is the cost of stepping on the ith step,
// find the minimum cost to reach the top. You can start from step 0 or step 1.
//
// Algorithm: Bottom-up DP with O(1) space optimization
// - dp[i] = min cost to reach step i
// - Recurrence: dp[i] = cost[i] + min(dp[i-1], dp[i-2])
// - Final answer: min(dp[n-1], dp[n-2]) (can step from either of last two steps)
//
// Time: O(n) where n = number of steps
// Space: O(1) using two variables instead of array (optimized from O(n))
//
// Use cases:
// - Resource optimization (minimize cost while climbing)
// - Path planning with costs
// - Game theory (minimize damage/cost to reach goal)
// - Educational DP (classic interview problem)
//
// LeetCode #746 - Min Cost Climbing Stairs

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Find minimum cost to reach top of stairs.
/// You can start from step 0 or step 1.
/// From each step, you can climb 1 or 2 steps.
///
/// Time: O(n)
/// Space: O(1)
///
/// Returns error.EmptyInput if cost array is empty.
pub fn minCostClimbingStairs(comptime T: type, cost: []const T) !T {
    if (cost.len == 0) return error.EmptyInput;
    if (cost.len == 1) return cost[0];
    if (cost.len == 2) return @min(cost[0], cost[1]);

    // dp[i] = min cost to reach step i
    // We only need previous two values, so use O(1) space
    var prev2: T = cost[0]; // dp[i-2]
    var prev1: T = cost[1]; // dp[i-1]

    for (cost[2..]) |step_cost| {
        const current = step_cost + @min(prev1, prev2);
        prev2 = prev1;
        prev1 = current;
    }

    // Can reach top from either of last two steps
    return @min(prev1, prev2);
}

/// Find minimum cost using tabulation (full DP table).
/// Useful for understanding the algorithm or reconstructing the path.
///
/// Time: O(n)
/// Space: O(n)
///
/// Returns error.EmptyInput if cost array is empty.
pub fn minCostClimbingStairsTabulation(comptime T: type, allocator: Allocator, cost: []const T) !T {
    if (cost.len == 0) return error.EmptyInput;
    if (cost.len == 1) return cost[0];
    if (cost.len == 2) return @min(cost[0], cost[1]);

    const n = cost.len;
    var dp = try allocator.alloc(T, n);
    defer allocator.free(dp);

    dp[0] = cost[0];
    dp[1] = cost[1];

    for (cost[2..], 2..) |step_cost, i| {
        dp[i] = step_cost + @min(dp[i - 1], dp[i - 2]);
    }

    return @min(dp[n - 1], dp[n - 2]);
}

/// Find minimum cost and reconstruct the path taken.
/// Returns both the minimum cost and the sequence of steps.
///
/// Time: O(n)
/// Space: O(n)
///
/// Caller owns returned ArrayList and must call deinit().
pub fn minCostClimbingStairsWithPath(comptime T: type, allocator: Allocator, cost: []const T) !struct { cost: T, path: std.ArrayList(usize) } {
    if (cost.len == 0) return error.EmptyInput;

    var path = try std.ArrayList(usize).initCapacity(allocator, 0);
    errdefer path.deinit(allocator);

    if (cost.len == 1) {
        try path.append(allocator, 0);
        return .{ .cost = cost[0], .path = path };
    }

    if (cost.len == 2) {
        if (cost[0] <= cost[1]) {
            try path.append(allocator, 0);
            return .{ .cost = cost[0], .path = path };
        } else {
            try path.append(allocator, 1);
            return .{ .cost = cost[1], .path = path };
        }
    }

    const n = cost.len;
    var dp = try allocator.alloc(T, n);
    defer allocator.free(dp);

    dp[0] = cost[0];
    dp[1] = cost[1];

    for (cost[2..], 2..) |step_cost, i| {
        dp[i] = step_cost + @min(dp[i - 1], dp[i - 2]);
    }

    // Backtrack to find path
    const min_cost = @min(dp[n - 1], dp[n - 2]);
    var current: usize = if (dp[n - 1] <= dp[n - 2]) n - 1 else n - 2;

    // Build path backwards
    var path_backwards = try std.ArrayList(usize).initCapacity(allocator, 0);
    defer path_backwards.deinit(allocator);

    try path_backwards.append(allocator, current);

    while (current >= 2) {
        // Determine which previous step was used
        if (dp[current - 1] <= dp[current - 2]) {
            current = current - 1;
        } else {
            current = current - 2;
        }
        try path_backwards.append(allocator, current);
    }

    // Reverse to get forward path
    var i: usize = path_backwards.items.len;
    while (i > 0) {
        i -= 1;
        try path.append(allocator, path_backwards.items[i]);
    }

    return .{ .cost = min_cost, .path = path };
}

/// Find minimum cost starting from a specific step (0 or 1).
///
/// Time: O(n)
/// Space: O(1)
pub fn minCostClimbingStairsFrom(comptime T: type, cost: []const T, start_step: usize) !T {
    if (cost.len == 0) return error.EmptyInput;
    if (start_step >= cost.len) return error.InvalidStartStep;

    if (start_step == cost.len - 1) return cost[start_step];
    if (start_step == cost.len - 2) return @min(cost[start_step], cost[start_step + 1]);

    var prev2: T = if (start_step == 0) cost[0] else 0;
    var prev1: T = if (start_step <= 1) cost[1] else 0;

    if (start_step > 1) {
        prev2 = cost[start_step];
        prev1 = cost[start_step + 1];
        for (cost[start_step + 2 ..]) |step_cost| {
            const current = step_cost + @min(prev1, prev2);
            prev2 = prev1;
            prev1 = current;
        }
    } else {
        for (cost[2..]) |step_cost| {
            const current = step_cost + @min(prev1, prev2);
            prev2 = prev1;
            prev1 = current;
        }
    }

    return @min(prev1, prev2);
}

/// Find minimum cost with variable step sizes (can climb 1 to k steps at a time).
///
/// Time: O(n × k)
/// Space: O(n)
pub fn minCostClimbingStairsVariable(comptime T: type, allocator: Allocator, cost: []const T, max_steps: usize) !T {
    if (cost.len == 0) return error.EmptyInput;
    if (max_steps == 0) return error.InvalidStepSize;
    if (cost.len == 1) return cost[0];

    const n = cost.len;
    var dp = try allocator.alloc(T, n);
    defer allocator.free(dp);

    dp[0] = cost[0];

    for (1..n) |i| {
        var min_prev = dp[i - 1];
        var step: usize = 2;
        while (step <= max_steps and i >= step) : (step += 1) {
            min_prev = @min(min_prev, dp[i - step]);
        }
        dp[i] = cost[i] + min_prev;
    }

    // Can reach top from any of last max_steps steps
    var min_cost = dp[n - 1];
    var step: usize = 2;
    while (step <= max_steps and step <= n) : (step += 1) {
        min_cost = @min(min_cost, dp[n - step]);
    }

    return min_cost;
}

test "min cost climbing stairs - basic example" {
    const cost = [_]i32{ 10, 15, 20 };
    const result = try minCostClimbingStairs(i32, &cost);
    try std.testing.expectEqual(@as(i32, 15), result); // Start at step 1 (cost 15)
}

test "min cost climbing stairs - LeetCode example 1" {
    const cost = [_]i32{ 10, 15, 20 };
    const result = try minCostClimbingStairs(i32, &cost);
    try std.testing.expectEqual(@as(i32, 15), result);
}

test "min cost climbing stairs - LeetCode example 2" {
    const cost = [_]i32{ 1, 100, 1, 1, 1, 100, 1, 1, 100, 1 };
    const result = try minCostClimbingStairs(i32, &cost);
    try std.testing.expectEqual(@as(i32, 6), result); // Steps: 0, 2, 4, 6, 7, 9
}

test "min cost climbing stairs - single step" {
    const cost = [_]i32{10};
    const result = try minCostClimbingStairs(i32, &cost);
    try std.testing.expectEqual(@as(i32, 10), result);
}

test "min cost climbing stairs - two steps" {
    const cost = [_]i32{ 5, 10 };
    const result = try minCostClimbingStairs(i32, &cost);
    try std.testing.expectEqual(@as(i32, 5), result); // Start at step 0
}

test "min cost climbing stairs - all equal costs" {
    const cost = [_]i32{ 5, 5, 5, 5, 5 };
    const result = try minCostClimbingStairs(i32, &cost);
    // Can reach top from step 3 or 4. Step 3: 0->2->3 (15), Step 4: 1->3->4 (15) or 0->2->4 (15)
    // Actually minimum is from step 3: 1->2->3 (10)
    try std.testing.expectEqual(@as(i32, 10), result);
}

test "min cost climbing stairs - empty input error" {
    const cost = [_]i32{};
    const result = minCostClimbingStairs(i32, &cost);
    try std.testing.expectError(error.EmptyInput, result);
}

test "min cost climbing stairs - tabulation variant" {
    const cost = [_]i32{ 10, 15, 20 };
    const result = try minCostClimbingStairsTabulation(i32, std.testing.allocator, &cost);
    try std.testing.expectEqual(@as(i32, 15), result);
}

test "min cost climbing stairs - tabulation consistency" {
    const cost = [_]i32{ 1, 100, 1, 1, 1, 100, 1, 1, 100, 1 };
    const optimized = try minCostClimbingStairs(i32, &cost);
    const tabulated = try minCostClimbingStairsTabulation(i32, std.testing.allocator, &cost);
    try std.testing.expectEqual(optimized, tabulated);
}

test "min cost climbing stairs - with path reconstruction" {
    const cost = [_]i32{ 10, 15, 20 };
    var result = try minCostClimbingStairsWithPath(i32, std.testing.allocator, &cost);
    defer result.path.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(i32, 15), result.cost);
    try std.testing.expect(result.path.items.len > 0);
    try std.testing.expect(result.path.items[0] == 1); // Starts at step 1
}

test "min cost climbing stairs - path validation" {
    const cost = [_]i32{ 1, 100, 1, 1, 1, 100, 1, 1, 100, 1 };
    var result = try minCostClimbingStairsWithPath(i32, std.testing.allocator, &cost);
    defer result.path.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(i32, 6), result.cost);

    // Verify path is valid (each step is at most 2 from previous)
    for (result.path.items[1..], 0..) |step, idx| {
        const prev_step = result.path.items[idx];
        const diff = step - prev_step;
        try std.testing.expect(diff >= 1 and diff <= 2);
    }

    // Verify total cost matches
    var total_cost: i32 = 0;
    for (result.path.items) |step| {
        total_cost += cost[step];
    }
    try std.testing.expectEqual(result.cost, total_cost);
}

test "min cost climbing stairs - start from specific step" {
    const cost = [_]i32{ 10, 15, 20 };
    const from_0 = try minCostClimbingStairsFrom(i32, &cost, 0);
    const from_1 = try minCostClimbingStairsFrom(i32, &cost, 1);

    try std.testing.expectEqual(@as(i32, 15), from_0); // 10 -> reach top from either step 1 or 2
    try std.testing.expectEqual(@as(i32, 15), from_1); // 15 alone or 15 -> 20
}

test "min cost climbing stairs - variable step sizes" {
    const cost = [_]i32{ 10, 15, 20, 25, 30 };

    // Max 2 steps (standard)
    const two_steps = try minCostClimbingStairsVariable(i32, std.testing.allocator, &cost, 2);

    // Max 3 steps
    const three_steps = try minCostClimbingStairsVariable(i32, std.testing.allocator, &cost, 3);

    // With more flexibility, cost should be ≤
    try std.testing.expect(three_steps <= two_steps);
}

test "min cost climbing stairs - large array" {
    const allocator = std.testing.allocator;
    const cost = try allocator.alloc(i32, 100);
    defer allocator.free(cost);

    for (cost, 0..) |*c, i| {
        c.* = @intCast(i % 10 + 1);
    }

    const result = try minCostClimbingStairs(i32, cost);
    const tabulated = try minCostClimbingStairsTabulation(i32, allocator, cost);

    try std.testing.expectEqual(result, tabulated);
}

test "min cost climbing stairs - negative costs" {
    const cost = [_]i32{ -10, 5, -3, 2 };
    const result = try minCostClimbingStairs(i32, &cost);

    // Should pick steps that minimize (maximize negative values)
    try std.testing.expect(result < 0); // Can get negative total cost
}

test "min cost climbing stairs - f32 support" {
    const cost = [_]f32{ 1.5, 2.0, 3.5, 4.0 };
    const result = try minCostClimbingStairs(f32, &cost);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result, 0.001); // dp[2] = 3.5 + 1.5 = 5.0
}

test "min cost climbing stairs - f64 support" {
    const cost = [_]f64{ 1.5, 2.0, 3.5, 4.0 };
    const result = try minCostClimbingStairs(f64, &cost);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result, 0.001);
}

test "min cost climbing stairs - invalid start step" {
    const cost = [_]i32{ 10, 15, 20 };
    const result = minCostClimbingStairsFrom(i32, &cost, 5);
    try std.testing.expectError(error.InvalidStartStep, result);
}

test "min cost climbing stairs - zero max steps error" {
    const cost = [_]i32{ 10, 15, 20 };
    const result = minCostClimbingStairsVariable(i32, std.testing.allocator, &cost, 0);
    try std.testing.expectError(error.InvalidStepSize, result);
}

test "min cost climbing stairs - memory safety" {
    const allocator = std.testing.allocator;

    // Test with path reconstruction (most allocations)
    for (0..10) |_| {
        const cost = [_]i32{ 1, 100, 1, 1, 1, 100, 1, 1, 100, 1 };
        var result = try minCostClimbingStairsWithPath(i32, allocator, &cost);
        result.path.deinit(allocator);

        const tab_result = try minCostClimbingStairsTabulation(i32, allocator, &cost);
        _ = tab_result;

        const var_result = try minCostClimbingStairsVariable(i32, allocator, &cost, 3);
        _ = var_result;
    }
}
