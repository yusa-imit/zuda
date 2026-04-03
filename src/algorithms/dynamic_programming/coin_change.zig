const std = @import("std");
const Allocator = std.mem.Allocator;

/// Coin Change Problem Algorithms
///
/// Solves various coin change problems using dynamic programming.
/// Applications: Making change, currency exchange, resource allocation.

/// Minimum Coins: Find minimum number of coins needed to make amount
/// Time: O(n*amount) | Space: O(amount) where n = number of coin denominations
///
/// Returns minimum number of coins needed, or null if amount cannot be made.
/// Uses bottom-up DP with space-optimized 1D array.
///
/// Example:
/// ```zig
/// const coins = [_]usize{1, 5, 10, 25};
/// const min = try minCoins(&coins, 63, allocator); // 6 coins (25+25+10+1+1+1)
/// ```
pub fn minCoins(coins: []const usize, amount: usize, allocator: Allocator) !?usize {
    if (amount == 0) return 0;
    if (coins.len == 0) return null;

    // dp[i] = minimum coins needed to make amount i
    const dp = try allocator.alloc(?usize, amount + 1);
    defer allocator.free(dp);

    @memset(dp, null);
    dp[0] = 0; // 0 coins needed for amount 0

    for (1..amount + 1) |i| {
        for (coins) |coin| {
            if (coin <= i) {
                if (dp[i - coin]) |prev| {
                    if (dp[i]) |curr| {
                        dp[i] = @min(curr, prev + 1);
                    } else {
                        dp[i] = prev + 1;
                    }
                }
            }
        }
    }

    return dp[amount];
}

/// Coin Change Ways: Count number of ways to make amount
/// Time: O(n*amount) | Space: O(amount)
///
/// Returns number of distinct combinations to make amount.
/// Order does not matter (e.g., [1,2] and [2,1] count as one way).
///
/// Example:
/// ```zig
/// const coins = [_]usize{1, 2, 5};
/// const ways = try countWays(&coins, 5, allocator); // 4 ways: [5], [2,2,1], [2,1,1,1], [1,1,1,1,1]
/// ```
pub fn countWays(coins: []const usize, amount: usize, allocator: Allocator) !usize {
    if (amount == 0) return 1; // One way to make 0: use no coins
    if (coins.len == 0) return 0;

    // dp[i] = number of ways to make amount i
    const dp = try allocator.alloc(usize, amount + 1);
    defer allocator.free(dp);

    @memset(dp, 0);
    dp[0] = 1;

    // For each coin denomination
    for (coins) |coin| {
        // Update all amounts that can include this coin
        for (coin..amount + 1) |i| {
            dp[i] += dp[i - coin];
        }
    }

    return dp[amount];
}

/// Get Coins Breakdown: Return actual coins used for minimum solution
/// Time: O(n*amount) | Space: O(amount)
///
/// Returns ArrayList of coin values used to make amount with minimum coins.
/// Returns null if amount cannot be made.
///
/// Example:
/// ```zig
/// const coins = [_]usize{1, 5, 10, 25};
/// const result = try getCoinsBreakdown(&coins, 63, allocator); // [25, 25, 10, 1, 1, 1]
/// defer if (result) |list| list.deinit();
/// ```
pub fn getCoinsBreakdown(coins: []const usize, amount: usize, allocator: Allocator) !?std.ArrayList(usize) {
    if (amount == 0) {
        return try std.ArrayList(usize).initCapacity(allocator, 0);
    }
    if (coins.len == 0) return null;

    // First compute DP table with parent tracking
    const dp = try allocator.alloc(?usize, amount + 1);
    defer allocator.free(dp);
    const parent = try allocator.alloc(?usize, amount + 1); // Track which coin was used
    defer allocator.free(parent);

    @memset(dp, null);
    @memset(parent, null);
    dp[0] = 0;

    for (1..amount + 1) |i| {
        for (coins) |coin| {
            if (coin <= i) {
                if (dp[i - coin]) |prev| {
                    const new_count = prev + 1;
                    if (dp[i]) |curr| {
                        if (new_count < curr) {
                            dp[i] = new_count;
                            parent[i] = coin;
                        }
                    } else {
                        dp[i] = new_count;
                        parent[i] = coin;
                    }
                }
            }
        }
    }

    if (dp[amount] == null) return null;

    // Backtrack to find coins used
    var result = try std.ArrayList(usize).initCapacity(allocator, 0);
    errdefer result.deinit(allocator);

    var current = amount;
    while (current > 0) {
        const coin_used = parent[current] orelse return null;
        try result.append(allocator, coin_used);
        current -= coin_used;
    }

    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "coin change - basic minimum coins" {
    const allocator = std.testing.allocator;

    const coins = [_]usize{ 1, 5, 10, 25 };

    // Amount 0
    try std.testing.expectEqual(@as(?usize, 0), try minCoins(&coins, 0, allocator));

    // Amount 1
    try std.testing.expectEqual(@as(?usize, 1), try minCoins(&coins, 1, allocator));

    // Amount 6 (1 nickel + 1 penny)
    try std.testing.expectEqual(@as(?usize, 2), try minCoins(&coins, 6, allocator));

    // Amount 63 (2 quarters + 1 dime + 3 pennies)
    try std.testing.expectEqual(@as(?usize, 6), try minCoins(&coins, 63, allocator));

    // Amount 99 (3 quarters + 2 dimes + 4 pennies)
    try std.testing.expectEqual(@as(?usize, 9), try minCoins(&coins, 99, allocator));
}

test "coin change - impossible amount" {
    const allocator = std.testing.allocator;

    const coins = [_]usize{ 5, 10 }; // No 1-cent coins

    // Cannot make 3 cents
    try std.testing.expectEqual(@as(?usize, null), try minCoins(&coins, 3, allocator));

    // Can make 15 cents (1 ten + 1 five)
    try std.testing.expectEqual(@as(?usize, 2), try minCoins(&coins, 15, allocator));
}

test "coin change - count ways" {
    const allocator = std.testing.allocator;

    const coins = [_]usize{ 1, 2, 5 };

    // Amount 0: 1 way (no coins)
    try std.testing.expectEqual(@as(usize, 1), try countWays(&coins, 0, allocator));

    // Amount 5: 4 ways
    // [5], [2,2,1], [2,1,1,1], [1,1,1,1,1]
    try std.testing.expectEqual(@as(usize, 4), try countWays(&coins, 5, allocator));

    // Amount 10
    try std.testing.expectEqual(@as(usize, 10), try countWays(&coins, 10, allocator));
}

test "coin change - single coin denomination" {
    const allocator = std.testing.allocator;

    const coins = [_]usize{5};

    // Can make 10 (2 coins)
    try std.testing.expectEqual(@as(?usize, 2), try minCoins(&coins, 10, allocator));

    // Cannot make 7
    try std.testing.expectEqual(@as(?usize, null), try minCoins(&coins, 7, allocator));

    // 1 way to make 15
    try std.testing.expectEqual(@as(usize, 1), try countWays(&coins, 15, allocator));

    // 0 ways to make 7
    try std.testing.expectEqual(@as(usize, 0), try countWays(&coins, 7, allocator));
}

test "coin change - get coins breakdown" {
    const allocator = std.testing.allocator;

    const coins = [_]usize{ 1, 5, 10, 25 };

    // Amount 0
    var result = try getCoinsBreakdown(&coins, 0, allocator);
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(usize, 0), result.?.items.len);
    result.?.deinit(allocator);

    // Amount 63
    result = try getCoinsBreakdown(&coins, 63, allocator);
    try std.testing.expect(result != null);
    defer result.?.deinit(allocator);

    // Should be 6 coins total
    try std.testing.expectEqual(@as(usize, 6), result.?.items.len);

    // Verify sum equals amount
    var sum: usize = 0;
    for (result.?.items) |coin| {
        sum += coin;
    }
    try std.testing.expectEqual(@as(usize, 63), sum);
}

test "coin change - breakdown impossible" {
    const allocator = std.testing.allocator;

    const coins = [_]usize{ 5, 10 };

    const result = try getCoinsBreakdown(&coins, 3, allocator);
    try std.testing.expectEqual(@as(?std.ArrayList(usize), null), result);
}

test "coin change - large amount" {
    const allocator = std.testing.allocator;

    const coins = [_]usize{ 1, 5, 10, 25 };

    // Amount 1000 (40 quarters)
    try std.testing.expectEqual(@as(?usize, 40), try minCoins(&coins, 1000, allocator));

    // Amount 9999
    const min = try minCoins(&coins, 9999, allocator);
    try std.testing.expect(min != null);
    try std.testing.expect(min.? <= 9999); // At worst, all pennies
}

test "coin change - empty coins" {
    const allocator = std.testing.allocator;

    const coins = [_]usize{};

    try std.testing.expectEqual(@as(?usize, null), try minCoins(&coins, 1, allocator));
    try std.testing.expectEqual(@as(usize, 0), try countWays(&coins, 1, allocator));
}

test "coin change - ways with different denominations" {
    const allocator = std.testing.allocator;

    // US coins
    const us_coins = [_]usize{ 1, 5, 10, 25 };
    const ways_100 = try countWays(&us_coins, 100, allocator);
    try std.testing.expect(ways_100 > 0);

    // Euro-like coins
    const euro_coins = [_]usize{ 1, 2, 5, 10, 20, 50 };
    const ways_100_euro = try countWays(&euro_coins, 100, allocator);
    try std.testing.expect(ways_100_euro > ways_100); // More denominations = more ways
}

test "coin change - order independence" {
    const allocator = std.testing.allocator;

    // Different orderings should give same result
    const coins1 = [_]usize{ 1, 5, 10, 25 };
    const coins2 = [_]usize{ 25, 10, 5, 1 };
    const coins3 = [_]usize{ 5, 1, 25, 10 };

    const amount = 37;

    const min1 = try minCoins(&coins1, amount, allocator);
    const min2 = try minCoins(&coins2, amount, allocator);
    const min3 = try minCoins(&coins3, amount, allocator);

    try std.testing.expectEqual(min1, min2);
    try std.testing.expectEqual(min2, min3);

    const ways1 = try countWays(&coins1, amount, allocator);
    const ways2 = try countWays(&coins2, amount, allocator);
    const ways3 = try countWays(&coins3, amount, allocator);

    try std.testing.expectEqual(ways1, ways2);
    try std.testing.expectEqual(ways2, ways3);
}

test "coin change - breakdown consistency" {
    const allocator = std.testing.allocator;

    const coins = [_]usize{ 1, 5, 10, 25 };
    const amount = 41;

    const min_coins = try minCoins(&coins, amount, allocator);
    var breakdown = try getCoinsBreakdown(&coins, amount, allocator);

    try std.testing.expect(breakdown != null);
    defer breakdown.?.deinit(allocator);

    // Breakdown should have same count as minimum
    try std.testing.expectEqual(min_coins.?, breakdown.?.items.len);

    // Sum should equal amount
    var sum: usize = 0;
    for (breakdown.?.items) |coin| {
        sum += coin;
        // All coins should be valid denominations
        var valid = false;
        for (coins) |valid_coin| {
            if (coin == valid_coin) {
                valid = true;
                break;
            }
        }
        try std.testing.expect(valid);
    }
    try std.testing.expectEqual(amount, sum);
}

test "coin change - memory safety" {
    const allocator = std.testing.allocator;

    const coins = [_]usize{ 1, 2, 5 };

    // Multiple allocations should not leak
    for (0..10) |_| {
        const min = try minCoins(&coins, 11, allocator);
        try std.testing.expect(min != null);

        const ways = try countWays(&coins, 11, allocator);
        try std.testing.expect(ways > 0);
    }
}
