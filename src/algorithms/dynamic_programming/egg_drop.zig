/// Egg Drop Problem — Dynamic Programming Solutions
///
/// Given n eggs and k floors, find the minimum number of egg drops needed
/// in the worst case to determine the critical floor (the highest floor
/// from which an egg can be dropped without breaking).
///
/// Problem Statement:
/// - We have n identical eggs and a k-floor building
/// - Eggs may break when dropped from certain floors (critical floor f)
/// - Below floor f, eggs don't break; at/above f, eggs break
/// - We need to find floor f using minimum trials in the worst case
///
/// Key Insights:
/// - When we drop an egg from floor x:
///   * If it breaks: we have (n-1) eggs and need to check floors below x
///   * If it doesn't break: we have n eggs and need to check floors above x
/// - We take the maximum of these two cases (worst case)
/// - We try all possible floors and take the minimum over all choices
///
/// Reference: Cormen et al., "Introduction to Algorithms" (2009)

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Find the minimum number of trials needed in the worst case
/// to determine the critical floor with n eggs and k floors.
///
/// Time: O(n * k²) — for each (eggs, floors) pair, try all k floors
/// Space: O(n * k) — DP table
///
/// Algorithm:
/// - dp[i][j] = minimum trials with i eggs and j floors
/// - Base cases:
///   * dp[i][0] = 0 (no floors → 0 trials)
///   * dp[i][1] = 1 (one floor → 1 trial)
///   * dp[1][j] = j (one egg → must try linearly from floor 1)
/// - Recurrence: dp[i][j] = 1 + min over all x in [1..j] of:
///   max(dp[i-1][x-1], dp[i][j-x])
///   where:
///   * dp[i-1][x-1] = trials if egg breaks (check x-1 floors below with i-1 eggs)
///   * dp[i][j-x] = trials if egg doesn't break (check j-x floors above with i eggs)
///
/// Example:
/// ```zig
/// const min_trials = try minTrials(allocator, 2, 10);
/// defer allocator.free(min_trials);
/// // min_trials = 4 (optimal strategy: drop from floors 4, 7, 9, 10)
/// ```
pub fn minTrials(allocator: Allocator, eggs: usize, floors: usize) !usize {
    if (floors == 0) return 0;
    if (floors == 1) return 1;
    if (eggs == 1) return floors;

    // dp[i][j] = minimum trials with i eggs and j floors
    var dp = try allocator.alloc([]usize, eggs + 1);
    defer {
        for (dp) |row| {
            allocator.free(row);
        }
        allocator.free(dp);
    }

    for (dp, 0..) |*row, i| {
        row.* = try allocator.alloc(usize, floors + 1);
        @memset(row.*, 0);

        // Base case: 0 floors → 0 trials
        row.*[0] = 0;

        // Base case: 1 floor → 1 trial
        if (floors >= 1) {
            row.*[1] = 1;
        }

        // Base case: 1 egg → linear search (j trials for j floors)
        if (i == 1) {
            for (1..floors + 1) |j| {
                row.*[j] = j;
            }
        }
    }

    // Fill DP table
    for (2..eggs + 1) |i| {
        for (2..floors + 1) |j| {
            var min_result = std.math.maxInt(usize);

            // Try dropping from each floor x
            for (1..j + 1) |x| {
                // Worst case: max of two scenarios
                const breaks = dp[i - 1][x - 1]; // Egg breaks, check below
                const survives = dp[i][j - x]; // Egg survives, check above
                const worst_case = @max(breaks, survives);

                // We need 1 trial + worst_case
                const trials = 1 + worst_case;
                min_result = @min(min_result, trials);
            }

            dp[i][j] = min_result;
        }
    }

    return dp[eggs][floors];
}

/// Space-optimized version using O(k) space.
///
/// Since we only need the previous row (i-1) to compute row i,
/// we can use two 1D arrays and alternate between them.
///
/// Time: O(n * k²)
/// Space: O(k) — only store current and previous rows
pub fn minTrialsOptimized(allocator: Allocator, eggs: usize, floors: usize) !usize {
    if (floors == 0) return 0;
    if (floors == 1) return 1;
    if (eggs == 1) return floors;

    // prev[j] = minimum trials with (i-1) eggs and j floors
    // curr[j] = minimum trials with i eggs and j floors
    var prev = try allocator.alloc(usize, floors + 1);
    defer allocator.free(prev);
    var curr = try allocator.alloc(usize, floors + 1);
    defer allocator.free(curr);

    // Base case: 1 egg → linear search
    prev[0] = 0;
    for (1..floors + 1) |j| {
        prev[j] = j;
    }

    // Fill for each egg count
    for (2..eggs + 1) |_| {
        curr[0] = 0;
        curr[1] = 1;

        for (2..floors + 1) |j| {
            var min_result = std.math.maxInt(usize);

            for (1..j + 1) |x| {
                const breaks = prev[x - 1];
                const survives = curr[j - x];
                const worst_case = @max(breaks, survives);
                const trials = 1 + worst_case;
                min_result = @min(min_result, trials);
            }

            curr[j] = min_result;
        }

        // Swap for next iteration
        std.mem.swap([]usize, &prev, &curr);
    }

    return prev[floors];
}

/// Find the minimum number of trials and the optimal drop strategy.
///
/// Returns both the minimum trials and the sequence of floors to try
/// in the optimal strategy.
///
/// Time: O(n * k²)
/// Space: O(n * k) — DP table + backtracking
pub fn minTrialsWithStrategy(
    allocator: Allocator,
    eggs: usize,
    floors: usize,
) !struct { trials: usize, strategy: []usize } {
    if (floors == 0) {
        return .{ .trials = 0, .strategy = &[_]usize{} };
    }
    if (floors == 1) {
        var strategy = try allocator.alloc(usize, 1);
        strategy[0] = 1;
        return .{ .trials = 1, .strategy = strategy };
    }

    // dp[i][j] = minimum trials with i eggs and j floors
    var dp = try allocator.alloc([]usize, eggs + 1);
    defer {
        for (dp) |row| {
            allocator.free(row);
        }
        allocator.free(dp);
    }

    // choice[i][j] = optimal floor to drop from with i eggs and j floors
    var choice = try allocator.alloc([]usize, eggs + 1);
    defer {
        for (choice) |row| {
            allocator.free(row);
        }
        allocator.free(choice);
    }

    for (dp, 0..) |*dp_row, i| {
        dp_row.* = try allocator.alloc(usize, floors + 1);
        choice[i] = try allocator.alloc(usize, floors + 1);
        @memset(dp_row.*, 0);
        @memset(choice[i], 0);

        dp_row.*[0] = 0;
        if (floors >= 1) {
            dp_row.*[1] = 1;
            choice[i][1] = 1;
        }

        if (i == 1) {
            for (1..floors + 1) |j| {
                dp_row.*[j] = j;
                choice[i][j] = j; // With 1 egg, always drop from top remaining floor
            }
        }
    }

    // Fill DP table
    for (2..eggs + 1) |i| {
        for (2..floors + 1) |j| {
            var min_result = std.math.maxInt(usize);
            var best_floor: usize = 1;

            for (1..j + 1) |x| {
                const breaks = dp[i - 1][x - 1];
                const survives = dp[i][j - x];
                const worst_case = @max(breaks, survives);
                const trials = 1 + worst_case;

                if (trials < min_result) {
                    min_result = trials;
                    best_floor = x;
                }
            }

            dp[i][j] = min_result;
            choice[i][j] = best_floor;
        }
    }

    // Backtrack to get the strategy
    var strategy_list = std.ArrayList(usize).init(allocator);
    errdefer strategy_list.deinit();

    var curr_eggs = eggs;
    var curr_floors = floors;
    var floor_offset: usize = 0;

    while (curr_floors > 0 and curr_eggs > 0) {
        const drop_floor = choice[curr_eggs][curr_floors];
        try strategy_list.append(floor_offset + drop_floor);

        // Assume worst case for demonstration (egg breaks)
        if (curr_eggs > 1) {
            curr_eggs -= 1;
            curr_floors = drop_floor - 1;
        } else {
            // With 1 egg, we must try linearly
            floor_offset += drop_floor;
            curr_floors -= drop_floor;
        }

        if (curr_floors == 0) break;
    }

    return .{
        .trials = dp[eggs][floors],
        .strategy = try strategy_list.toOwnedSlice(),
    };
}

// ============================================================================
// Tests
// ============================================================================

test "egg drop: basic cases" {
    const allocator = testing.allocator;

    // 0 floors → 0 trials
    try testing.expectEqual(0, try minTrials(allocator, 2, 0));

    // 1 floor → 1 trial
    try testing.expectEqual(1, try minTrials(allocator, 2, 1));

    // 1 egg → linear search (n trials for n floors)
    try testing.expectEqual(5, try minTrials(allocator, 1, 5));
    try testing.expectEqual(10, try minTrials(allocator, 1, 10));
}

test "egg drop: 2 eggs" {
    const allocator = testing.allocator;

    // 2 eggs, 10 floors → 4 trials
    // Optimal: drop from floors 4, 7, 9, 10
    try testing.expectEqual(4, try minTrials(allocator, 2, 10));

    // 2 eggs, 6 floors → 3 trials
    try testing.expectEqual(3, try minTrials(allocator, 2, 6));

    // 2 eggs, 100 floors → 14 trials
    try testing.expectEqual(14, try minTrials(allocator, 2, 100));
}

test "egg drop: multiple eggs" {
    const allocator = testing.allocator;

    // 3 eggs, 14 floors → 4 trials
    try testing.expectEqual(4, try minTrials(allocator, 3, 14));

    // 4 eggs, 16 floors → 4 trials
    try testing.expectEqual(4, try minTrials(allocator, 4, 16));

    // Many eggs → binary search (log₂ floors)
    // 10 eggs, 100 floors → 7 trials (close to log₂ 100 ≈ 6.64)
    const result = try minTrials(allocator, 10, 100);
    try testing.expect(result >= 7 and result <= 8);
}

test "egg drop: optimized version matches standard" {
    const allocator = testing.allocator;

    const test_cases = [_]struct { eggs: usize, floors: usize }{
        .{ .eggs = 2, .floors = 10 },
        .{ .eggs = 2, .floors = 100 },
        .{ .eggs = 3, .floors = 14 },
        .{ .eggs = 4, .floors = 16 },
        .{ .eggs = 5, .floors = 50 },
    };

    for (test_cases) |case| {
        const standard = try minTrials(allocator, case.eggs, case.floors);
        const optimized = try minTrialsOptimized(allocator, case.eggs, case.floors);
        try testing.expectEqual(standard, optimized);
    }
}

test "egg drop: with strategy" {
    const allocator = testing.allocator;

    const result = try minTrialsWithStrategy(allocator, 2, 10);
    defer allocator.free(result.strategy);

    try testing.expectEqual(4, result.trials);
    try testing.expect(result.strategy.len > 0);

    // Strategy should start with a reasonable floor (not 1 or 10)
    try testing.expect(result.strategy[0] > 1);
    try testing.expect(result.strategy[0] < 10);
}

test "egg drop: edge cases with strategy" {
    const allocator = testing.allocator;

    // 0 floors
    {
        const result = try minTrialsWithStrategy(allocator, 2, 0);
        defer allocator.free(result.strategy);
        try testing.expectEqual(0, result.trials);
        try testing.expectEqual(0, result.strategy.len);
    }

    // 1 floor
    {
        const result = try minTrialsWithStrategy(allocator, 2, 1);
        defer allocator.free(result.strategy);
        try testing.expectEqual(1, result.trials);
        try testing.expectEqual(1, result.strategy.len);
        try testing.expectEqual(1, result.strategy[0]);
    }
}

test "egg drop: large scale" {
    const allocator = testing.allocator;

    // 2 eggs, 1000 floors
    const result = try minTrials(allocator, 2, 1000);
    // With 2 eggs, optimal is around √(2*floors) ≈ 45
    try testing.expect(result >= 44 and result <= 46);
}

test "egg drop: increasing floors with fixed eggs" {
    const allocator = testing.allocator;

    // With 2 eggs, trials should increase as floors increase
    var prev_trials: usize = 0;
    for (1..21) |floors| {
        const trials = try minTrials(allocator, 2, floors);
        try testing.expect(trials >= prev_trials);
        prev_trials = trials;
    }
}

test "egg drop: increasing eggs with fixed floors" {
    const allocator = testing.allocator;

    // With 100 floors, trials should decrease as eggs increase
    const floors = 100;
    var prev_trials: usize = std.math.maxInt(usize);

    for (1..11) |eggs| {
        const trials = try minTrials(allocator, eggs, floors);
        try testing.expect(trials <= prev_trials);
        prev_trials = trials;
    }
}

test "egg drop: 1 egg requires linear search" {
    const allocator = testing.allocator;

    // With 1 egg, we must try from floor 1 up to n
    for (1..21) |floors| {
        const trials = try minTrials(allocator, 1, floors);
        try testing.expectEqual(floors, trials);
    }
}

test "egg drop: many eggs approaches binary search" {
    const allocator = testing.allocator;

    const floors = 128; // 2^7
    const eggs = 20; // More than enough

    const trials = try minTrials(allocator, eggs, floors);

    // With many eggs, we can do binary search ≈ log₂(128) = 7
    // Allow small overhead for DP algorithm
    try testing.expect(trials >= 7 and trials <= 8);
}

test "egg drop: memory safety" {
    const allocator = testing.allocator;

    // Test multiple invocations to ensure no leaks
    for (0..10) |i| {
        const eggs = 2 + (i % 3);
        const floors = 10 + (i * 5);
        _ = try minTrials(allocator, eggs, floors);
        _ = try minTrialsOptimized(allocator, eggs, floors);
    }
}

test "egg drop: strategy memory safety" {
    const allocator = testing.allocator;

    for (0..5) |i| {
        const eggs = 2 + i;
        const floors = 10 + (i * 10);
        const result = try minTrialsWithStrategy(allocator, eggs, floors);
        defer allocator.free(result.strategy);

        try testing.expect(result.trials > 0);
        try testing.expect(result.strategy.len > 0);
    }
}
