const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Climbing Stairs — Count distinct ways to climb n stairs taking 1 or 2 steps at a time.
///
/// Classic dynamic programming problem with Fibonacci-like recurrence.
///
/// **Problem**: You're climbing a staircase with `n` steps. Each time you can climb
/// either 1 or 2 steps. How many distinct ways can you reach the top?
///
/// **Recurrence**: ways(n) = ways(n-1) + ways(n-2)
/// - Base: ways(0) = 1 (one way to stay at ground), ways(1) = 1 (one step)
/// - To reach step n: arrive from n-1 (1 step) OR n-2 (2 steps)
///
/// **Example**: n=3 → 3 ways: (1,1,1), (1,2), (2,1)
///
/// **Use cases**:
/// - Combinatorial counting problems
/// - Path counting in grids/graphs
/// - Educational DP introduction
/// - Interview questions (very common)

/// Count ways to climb n stairs — O(n) time, O(1) space.
///
/// Uses iterative DP with space optimization (only track last 2 states).
///
/// Time: O(n) — single pass through stairs
/// Space: O(1) — only 2 variables
///
/// **Example**:
/// ```zig
/// const ways = countWays(5); // 8 ways to climb 5 stairs
/// ```
pub fn countWays(n: usize) u64 {
    if (n == 0) return 1; // Base: staying at ground
    if (n == 1) return 1; // Base: one step

    var prev2: u64 = 1; // ways(0)
    var prev1: u64 = 1; // ways(1)
    var current: u64 = 0;

    var i: usize = 2;
    while (i <= n) : (i += 1) {
        current = prev1 + prev2;
        prev2 = prev1;
        prev1 = current;
    }

    return current;
}

/// Count ways to climb n stairs — O(n) time, O(n) space (tabulation).
///
/// Uses bottom-up DP table for clarity. Useful when path reconstruction is needed.
///
/// Time: O(n) — fill DP table
/// Space: O(n) — DP table
///
/// **Example**:
/// ```zig
/// var ways = try countWaysTabulation(allocator, 10);
/// defer allocator.free(ways.table);
/// std.debug.print("Ways: {}\n", .{ways.count});
/// ```
pub fn countWaysTabulation(allocator: Allocator, n: usize) !struct { count: u64, table: []u64 } {
    if (n == 0) {
        const table = try allocator.alloc(u64, 1);
        table[0] = 1;
        return .{ .count = 1, .table = table };
    }

    const table = try allocator.alloc(u64, n + 1);
    errdefer allocator.free(table);

    table[0] = 1; // Base: staying at ground
    table[1] = 1; // Base: one step

    var i: usize = 2;
    while (i <= n) : (i += 1) {
        table[i] = table[i - 1] + table[i - 2];
    }

    return .{ .count = table[n], .table = table };
}

/// Count ways to climb n stairs with k step options — O(n*k) time, O(n) space.
///
/// Generalized version: can take steps from `allowed_steps` array.
///
/// Time: O(n × k) where k = allowed_steps.len
/// Space: O(n) — DP table
///
/// **Example**:
/// ```zig
/// // Can take 1, 2, or 3 steps at a time
/// const ways = try countWaysGeneral(allocator, 5, &[_]usize{1, 2, 3});
/// defer allocator.free(ways);
/// ```
pub fn countWaysGeneral(allocator: Allocator, n: usize, allowed_steps: []const usize) !u64 {
    if (n == 0) return 1;
    if (allowed_steps.len == 0) return 0;

    const dp = try allocator.alloc(u64, n + 1);
    defer allocator.free(dp);

    @memset(dp, 0);
    dp[0] = 1; // Base: staying at ground

    var i: usize = 1;
    while (i <= n) : (i += 1) {
        for (allowed_steps) |step| {
            if (step <= i) {
                dp[i] += dp[i - step];
            }
        }
    }

    return dp[n];
}

/// Count ways to climb n stairs with minimum cost — O(n) time, O(1) space.
///
/// Each step has a cost. Find minimum cost to reach the top.
/// You can start from step 0 or step 1.
///
/// Time: O(n) — single pass
/// Space: O(1) — only track last 2 costs
///
/// **Example**:
/// ```zig
/// const cost = &[_]u32{10, 15, 20};
/// const min_cost = minCost(cost); // 15 (start at 1, climb to top)
/// ```
pub fn minCost(cost: []const u32) u32 {
    if (cost.len == 0) return 0;
    if (cost.len == 1) return cost[0];

    var prev2: u32 = cost[0]; // cost to reach step 0
    var prev1: u32 = cost[1]; // cost to reach step 1

    var i: usize = 2;
    while (i < cost.len) : (i += 1) {
        const current = cost[i] + @min(prev1, prev2);
        prev2 = prev1;
        prev1 = current;
    }

    // Can finish from either last or second-to-last step
    return @min(prev1, prev2);
}

/// Count ways to climb n stairs with exactly k steps — O(n*k) time, O(n*k) space.
///
/// Count ways to reach step n using exactly k moves (each move is 1 or 2 steps).
///
/// Time: O(n × k)
/// Space: O(n × k) — 2D DP table
///
/// **Example**:
/// ```zig
/// // Ways to climb 5 stairs in exactly 3 moves
/// const ways = try countWaysExactSteps(allocator, 5, 3); // (1,2,2), (2,1,2), (2,2,1)
/// defer allocator.free(ways);
/// ```
pub fn countWaysExactSteps(allocator: Allocator, n: usize, k: usize) !u64 {
    if (n == 0 and k == 0) return 1;
    if (n == 0 or k == 0) return 0;

    // dp[i][j] = ways to reach step i in exactly j moves
    const rows = n + 1;
    const cols = k + 1;
    const dp = try allocator.alloc(u64, rows * cols);
    defer allocator.free(dp);

    @memset(dp, 0);

    // Base: 0 steps in 0 moves
    dp[0 * cols + 0] = 1;

    var i: usize = 1;
    while (i <= n) : (i += 1) {
        var j: usize = 1;
        while (j <= k) : (j += 1) {
            // Arrive from i-1 (1 step) or i-2 (2 steps) in j-1 moves
            if (i >= 1 and j >= 1) {
                dp[i * cols + j] += dp[(i - 1) * cols + (j - 1)];
            }
            if (i >= 2 and j >= 1) {
                dp[i * cols + j] += dp[(i - 2) * cols + (j - 1)];
            }
        }
    }

    return dp[n * cols + k];
}

// ============================================================================
// Tests
// ============================================================================

test "climbing stairs: basic cases" {
    try testing.expectEqual(@as(u64, 1), countWays(0)); // staying at ground
    try testing.expectEqual(@as(u64, 1), countWays(1)); // 1 way: (1)
    try testing.expectEqual(@as(u64, 2), countWays(2)); // 2 ways: (1,1), (2)
    try testing.expectEqual(@as(u64, 3), countWays(3)); // 3 ways: (1,1,1), (1,2), (2,1)
    try testing.expectEqual(@as(u64, 5), countWays(4)); // Fibonacci: 5
    try testing.expectEqual(@as(u64, 8), countWays(5)); // Fibonacci: 8
}

test "climbing stairs: large n" {
    try testing.expectEqual(@as(u64, 89), countWays(10));
    try testing.expectEqual(@as(u64, 1597), countWays(16));
    try testing.expectEqual(@as(u64, 10946), countWays(20));
}

test "climbing stairs: tabulation" {
    const result = try countWaysTabulation(testing.allocator, 5);
    defer testing.allocator.free(result.table);
    try testing.expectEqual(@as(u64, 8), result.count);
    try testing.expectEqual(@as(u64, 1), result.table[0]);
    try testing.expectEqual(@as(u64, 1), result.table[1]);
    try testing.expectEqual(@as(u64, 2), result.table[2]);
    try testing.expectEqual(@as(u64, 3), result.table[3]);
}

test "climbing stairs: tabulation n=0" {
    const result = try countWaysTabulation(testing.allocator, 0);
    defer testing.allocator.free(result.table);
    try testing.expectEqual(@as(u64, 1), result.count);
    try testing.expectEqual(@as(usize, 1), result.table.len);
}

test "climbing stairs: general with 3 steps" {
    // Can take 1, 2, or 3 steps at a time
    const allowed = [_]usize{ 1, 2, 3 };
    try testing.expectEqual(@as(u64, 1), try countWaysGeneral(testing.allocator, 0, &allowed));
    try testing.expectEqual(@as(u64, 1), try countWaysGeneral(testing.allocator, 1, &allowed)); // (1)
    try testing.expectEqual(@as(u64, 2), try countWaysGeneral(testing.allocator, 2, &allowed)); // (1,1), (2)
    try testing.expectEqual(@as(u64, 4), try countWaysGeneral(testing.allocator, 3, &allowed)); // (1,1,1), (1,2), (2,1), (3)
    try testing.expectEqual(@as(u64, 7), try countWaysGeneral(testing.allocator, 4, &allowed));
}

test "climbing stairs: general with custom steps" {
    // Can take 2, 4, or 5 steps at a time
    const allowed = [_]usize{ 2, 4, 5 };
    try testing.expectEqual(@as(u64, 1), try countWaysGeneral(testing.allocator, 0, &allowed));
    try testing.expectEqual(@as(u64, 0), try countWaysGeneral(testing.allocator, 1, &allowed)); // impossible
    try testing.expectEqual(@as(u64, 1), try countWaysGeneral(testing.allocator, 2, &allowed)); // (2)
    try testing.expectEqual(@as(u64, 2), try countWaysGeneral(testing.allocator, 4, &allowed)); // (4), (2,2)
    try testing.expectEqual(@as(u64, 3), try countWaysGeneral(testing.allocator, 6, &allowed)); // (2,4), (4,2), (2,2,2)
}

test "climbing stairs: general empty steps" {
    const allowed = [_]usize{};
    try testing.expectEqual(@as(u64, 0), try countWaysGeneral(testing.allocator, 1, &allowed));
}

test "climbing stairs: min cost basic" {
    const cost1 = [_]u32{ 10, 15, 20 };
    try testing.expectEqual(@as(u32, 15), minCost(&cost1)); // start at 1, pay 15

    const cost2 = [_]u32{ 1, 100, 1, 1, 1, 100, 1, 1, 100, 1 };
    try testing.expectEqual(@as(u32, 6), minCost(&cost2)); // optimal path avoids 100s
}

test "climbing stairs: min cost edge cases" {
    const cost_empty = [_]u32{};
    try testing.expectEqual(@as(u32, 0), minCost(&cost_empty));

    const cost_single = [_]u32{10};
    try testing.expectEqual(@as(u32, 10), minCost(&cost_single));

    const cost_two = [_]u32{ 5, 3 };
    try testing.expectEqual(@as(u32, 3), minCost(&cost_two));
}

test "climbing stairs: exact steps basic" {
    // n=5, k=3: ways to reach 5 in exactly 3 moves
    const ways = try countWaysExactSteps(testing.allocator, 5, 3);
    try testing.expectEqual(@as(u64, 3), ways); // (1,2,2), (2,1,2), (2,2,1)
}

test "climbing stairs: exact steps edge cases" {
    try testing.expectEqual(@as(u64, 1), try countWaysExactSteps(testing.allocator, 0, 0)); // base
    try testing.expectEqual(@as(u64, 0), try countWaysExactSteps(testing.allocator, 5, 0)); // impossible
    try testing.expectEqual(@as(u64, 0), try countWaysExactSteps(testing.allocator, 0, 3)); // impossible
    try testing.expectEqual(@as(u64, 1), try countWaysExactSteps(testing.allocator, 2, 1)); // single 2-step
    try testing.expectEqual(@as(u64, 1), try countWaysExactSteps(testing.allocator, 2, 2)); // (1,1)
}

test "climbing stairs: exact steps n=4" {
    // n=4, k=2: (2,2) only
    try testing.expectEqual(@as(u64, 1), try countWaysExactSteps(testing.allocator, 4, 2));

    // n=4, k=3: (1,1,2), (1,2,1), (2,1,1)
    try testing.expectEqual(@as(u64, 3), try countWaysExactSteps(testing.allocator, 4, 3));

    // n=4, k=4: (1,1,1,1)
    try testing.expectEqual(@as(u64, 1), try countWaysExactSteps(testing.allocator, 4, 4));
}

test "climbing stairs: Fibonacci property" {
    // Verify Fibonacci sequence property: F(n) = F(n-1) + F(n-2)
    var i: usize = 2;
    while (i <= 15) : (i += 1) {
        const ways_n = countWays(i);
        const ways_n1 = countWays(i - 1);
        const ways_n2 = countWays(i - 2);
        try testing.expectEqual(ways_n, ways_n1 + ways_n2);
    }
}

test "climbing stairs: memory safety" {
    // Verify no leaks with allocating functions
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const result = try countWaysTabulation(testing.allocator, i);
        testing.allocator.free(result.table);

        _ = try countWaysGeneral(testing.allocator, i, &[_]usize{ 1, 2 });
        _ = try countWaysExactSteps(testing.allocator, i, i);
    }
}
