const std = @import("std");
const Allocator = std.mem.Allocator;

/// Best Time to Buy and Sell Stock - Dynamic Programming Solutions
///
/// This module implements various stock trading problems using dynamic programming.
/// All variants find optimal buy/sell strategies to maximize profit.
///
/// Variants:
/// - maxProfitSingle: At most 1 transaction (buy once, sell once)
/// - maxProfitUnlimited: Unlimited transactions (no cooldown)
/// - maxProfitKTransactions: At most k transactions
/// - maxProfitWithCooldown: Unlimited transactions with 1-day cooldown after sell
/// - maxProfitWithFee: Unlimited transactions with transaction fee

/// Best Time to Buy and Sell Stock I: Single Transaction
///
/// Find maximum profit from at most one transaction (buy once, sell once).
/// Must buy before you sell.
///
/// Algorithm: Track minimum price seen so far, update max profit at each step.
/// State: min_price, max_profit
/// Recurrence: max_profit = max(max_profit, price - min_price)
///
/// Time: O(n) where n = length of prices
/// Space: O(1) - only tracking min_price and max_profit
///
/// Examples:
/// ```zig
/// const prices = [_]i32{7, 1, 5, 3, 6, 4};
/// const profit = maxProfitSingle(i32, &prices); // Returns 5 (buy at 1, sell at 6)
/// ```
pub fn maxProfitSingle(comptime T: type, prices: []const T) T {
    if (prices.len < 2) return 0;

    var min_price = prices[0];
    var max_profit: T = 0;

    for (prices[1..]) |price| {
        const profit = price - min_price;
        max_profit = @max(max_profit, profit);
        min_price = @min(min_price, price);
    }

    return max_profit;
}

/// Best Time to Buy and Sell Stock II: Unlimited Transactions
///
/// Find maximum profit from unlimited transactions (no cooldown).
/// Can hold at most 1 share at a time.
///
/// Algorithm: Sum all positive price differences (greedy approach).
/// Equivalent to DP with states: hold[i], not_hold[i]
///
/// Time: O(n) where n = length of prices
/// Space: O(1) - greedy accumulation
///
/// Examples:
/// ```zig
/// const prices = [_]i32{7, 1, 5, 3, 6, 4};
/// const profit = maxProfitUnlimited(i32, &prices); // Returns 7 (buy 1→sell 5, buy 3→sell 6)
/// ```
pub fn maxProfitUnlimited(comptime T: type, prices: []const T) T {
    if (prices.len < 2) return 0;

    var total_profit: T = 0;
    for (prices[1..], 0..) |price, i| {
        const prev_price = prices[i];
        if (price > prev_price) {
            total_profit += price - prev_price;
        }
    }

    return total_profit;
}

/// Best Time to Buy and Sell Stock III: At Most 2 Transactions
///
/// Find maximum profit from at most 2 transactions.
///
/// Algorithm: Track profits for first and second transactions.
/// States:
/// - buy1: max profit after first buy
/// - sell1: max profit after first sell
/// - buy2: max profit after second buy
/// - sell2: max profit after second sell
///
/// Recurrence:
/// buy1 = max(buy1, -price)
/// sell1 = max(sell1, buy1 + price)
/// buy2 = max(buy2, sell1 - price)
/// sell2 = max(sell2, buy2 + price)
///
/// Time: O(n) where n = length of prices
/// Space: O(1) - only 4 state variables
///
/// Examples:
/// ```zig
/// const prices = [_]i32{3, 3, 5, 0, 0, 3, 1, 4};
/// const profit = maxProfitTwoTransactions(i32, &prices); // Returns 6 (buy 0→sell 3, buy 1→sell 4)
/// ```
pub fn maxProfitTwoTransactions(comptime T: type, prices: []const T) T {
    if (prices.len < 2) return 0;

    const min_val = std.math.minInt(T);

    var buy1: T = min_val;
    var sell1: T = 0;
    var buy2: T = min_val;
    var sell2: T = 0;

    for (prices) |price| {
        buy1 = @max(buy1, -price);
        sell1 = @max(sell1, buy1 + price);
        buy2 = @max(buy2, sell1 - price);
        sell2 = @max(sell2, buy2 + price);
    }

    return sell2;
}

/// Best Time to Buy and Sell Stock IV: At Most K Transactions
///
/// Find maximum profit from at most k transactions.
///
/// Algorithm: 2D DP with k transactions and n days.
/// State: dp[i][j] = max profit using at most i transactions by day j
/// Recurrence:
/// dp[i][j] = max(dp[i][j-1], max(dp[i-1][t] + prices[j] - prices[t]) for all t < j)
///
/// Optimization: If k >= n/2, reduce to unlimited transactions problem.
///
/// Time: O(n×k) where n = length of prices, k = max transactions
/// Space: O(n×k) for DP table
///
/// Examples:
/// ```zig
/// const prices = [_]i32{3, 2, 6, 5, 0, 3};
/// const profit = try maxProfitKTransactions(i32, allocator, 2, &prices); // Returns 7 (buy 2→sell 6, buy 0→sell 3)
/// defer allocator.free(profit);
/// ```
pub fn maxProfitKTransactions(comptime T: type, allocator: Allocator, k: usize, prices: []const T) !T {
    if (prices.len < 2 or k == 0) return 0;

    const n = prices.len;

    // Optimization: if k >= n/2, reduce to unlimited transactions
    if (k >= n / 2) {
        return maxProfitUnlimited(T, prices);
    }

    // dp[i][j] = max profit using at most i transactions by day j
    var dp = try allocator.alloc([]T, k + 1);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }

    for (dp) |*row| {
        row.* = try allocator.alloc(T, n);
        @memset(row.*, 0);
    }

    // Fill DP table
    for (1..k + 1) |i| {
        var max_diff: T = -prices[0];
        for (1..n) |j| {
            dp[i][j] = @max(dp[i][j - 1], prices[j] + max_diff);
            max_diff = @max(max_diff, dp[i - 1][j] - prices[j]);
        }
    }

    return dp[k][n - 1];
}

/// Best Time to Buy and Sell Stock with Cooldown
///
/// Find maximum profit with unlimited transactions and 1-day cooldown after sell.
/// After selling, must wait 1 day before buying again.
///
/// Algorithm: Track 3 states at each day.
/// States:
/// - hold[i]: max profit when holding stock on day i
/// - sold[i]: max profit when just sold stock on day i (cooldown tomorrow)
/// - rest[i]: max profit when not holding stock on day i (can buy tomorrow)
///
/// Recurrence:
/// hold[i] = max(hold[i-1], rest[i-1] - prices[i])
/// sold[i] = hold[i-1] + prices[i]
/// rest[i] = max(rest[i-1], sold[i-1])
///
/// Time: O(n) where n = length of prices
/// Space: O(1) - only 3 state variables
///
/// Examples:
/// ```zig
/// const prices = [_]i32{1, 2, 3, 0, 2};
/// const profit = maxProfitWithCooldown(i32, &prices); // Returns 3 (buy 1→sell 3, cooldown, buy 0→sell 2)
/// ```
pub fn maxProfitWithCooldown(comptime T: type, prices: []const T) T {
    if (prices.len < 2) return 0;

    const min_val = std.math.minInt(T);

    var hold: T = min_val; // Holding stock
    var sold: T = 0; // Just sold (cooldown next)
    var rest: T = 0; // Not holding, can buy

    for (prices) |price| {
        const prev_hold = hold;
        const prev_sold = sold;
        const prev_rest = rest;

        hold = @max(prev_hold, prev_rest - price);
        sold = prev_hold + price;
        rest = @max(prev_rest, prev_sold);
    }

    return @max(sold, rest);
}

/// Best Time to Buy and Sell Stock with Transaction Fee
///
/// Find maximum profit with unlimited transactions and a fee per transaction.
/// Fee is paid when selling.
///
/// Algorithm: Track 2 states at each day.
/// States:
/// - hold[i]: max profit when holding stock on day i
/// - not_hold[i]: max profit when not holding stock on day i
///
/// Recurrence:
/// hold[i] = max(hold[i-1], not_hold[i-1] - prices[i])
/// not_hold[i] = max(not_hold[i-1], hold[i-1] + prices[i] - fee)
///
/// Time: O(n) where n = length of prices
/// Space: O(1) - only 2 state variables
///
/// Examples:
/// ```zig
/// const prices = [_]i32{1, 3, 2, 8, 4, 9};
/// const profit = maxProfitWithFee(i32, &prices, 2); // Returns 8 (buy 1→sell 8-fee, buy 4→sell 9-fee)
/// ```
pub fn maxProfitWithFee(comptime T: type, prices: []const T, fee: T) T {
    if (prices.len < 2) return 0;

    const min_val = std.math.minInt(T);

    var hold: T = min_val; // Holding stock
    var not_hold: T = 0; // Not holding stock

    for (prices) |price| {
        const prev_hold = hold;
        const prev_not_hold = not_hold;

        hold = @max(prev_hold, prev_not_hold - price);
        not_hold = @max(prev_not_hold, prev_hold + price - fee);
    }

    return not_hold;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "maxProfitSingle - basic example" {
    const prices = [_]i32{ 7, 1, 5, 3, 6, 4 };
    try testing.expectEqual(@as(i32, 5), maxProfitSingle(i32, &prices)); // Buy at 1, sell at 6
}

test "maxProfitSingle - no profit possible" {
    const prices = [_]i32{ 7, 6, 4, 3, 1 };
    try testing.expectEqual(@as(i32, 0), maxProfitSingle(i32, &prices)); // Decreasing prices
}

test "maxProfitSingle - empty and single price" {
    try testing.expectEqual(@as(i32, 0), maxProfitSingle(i32, &[_]i32{}));
    try testing.expectEqual(@as(i32, 0), maxProfitSingle(i32, &[_]i32{5}));
}

test "maxProfitSingle - two prices" {
    try testing.expectEqual(@as(i32, 3), maxProfitSingle(i32, &[_]i32{ 2, 5 }));
    try testing.expectEqual(@as(i32, 0), maxProfitSingle(i32, &[_]i32{ 5, 2 }));
}

test "maxProfitSingle - f64 support" {
    const prices = [_]f64{ 7.5, 1.2, 5.8, 3.3, 6.9, 4.1 };
    try testing.expectApproxEqAbs(@as(f64, 5.7), maxProfitSingle(f64, &prices), 1e-9); // Buy at 1.2, sell at 6.9
}

test "maxProfitUnlimited - basic example" {
    const prices = [_]i32{ 7, 1, 5, 3, 6, 4 };
    try testing.expectEqual(@as(i32, 7), maxProfitUnlimited(i32, &prices)); // (5-1) + (6-3) = 7
}

test "maxProfitUnlimited - monotonic increasing" {
    const prices = [_]i32{ 1, 2, 3, 4, 5 };
    try testing.expectEqual(@as(i32, 4), maxProfitUnlimited(i32, &prices)); // Buy at 1, sell at 5
}

test "maxProfitUnlimited - monotonic decreasing" {
    const prices = [_]i32{ 5, 4, 3, 2, 1 };
    try testing.expectEqual(@as(i32, 0), maxProfitUnlimited(i32, &prices)); // No profit
}

test "maxProfitUnlimited - peak valley" {
    const prices = [_]i32{ 1, 2, 3, 0, 2, 3 };
    try testing.expectEqual(@as(i32, 5), maxProfitUnlimited(i32, &prices)); // (3-1) + (3-0) = 5
}

test "maxProfitUnlimited - f64 support" {
    const prices = [_]f64{ 1.0, 2.5, 1.5, 3.0 };
    try testing.expectApproxEqAbs(@as(f64, 3.0), maxProfitUnlimited(f64, &prices), 1e-9); // (2.5-1.0) + (3.0-1.5) = 3.0
}

test "maxProfitTwoTransactions - basic example" {
    const prices = [_]i32{ 3, 3, 5, 0, 0, 3, 1, 4 };
    try testing.expectEqual(@as(i32, 6), maxProfitTwoTransactions(i32, &prices)); // (3-0) + (4-1) = 6
}

test "maxProfitTwoTransactions - one transaction optimal" {
    const prices = [_]i32{ 1, 2, 3, 4, 5 };
    try testing.expectEqual(@as(i32, 4), maxProfitTwoTransactions(i32, &prices)); // Buy at 1, sell at 5
}

test "maxProfitTwoTransactions - two transactions optimal" {
    const prices = [_]i32{ 1, 2, 4, 2, 5, 7, 2, 4, 9, 0 };
    try testing.expectEqual(@as(i32, 13), maxProfitTwoTransactions(i32, &prices)); // (7-1) + (9-2) = 13
}

test "maxProfitTwoTransactions - no profit" {
    const prices = [_]i32{ 5, 4, 3, 2, 1 };
    try testing.expectEqual(@as(i32, 0), maxProfitTwoTransactions(i32, &prices));
}

test "maxProfitKTransactions - k=2 matches two transactions" {
    const allocator = testing.allocator;
    const prices = [_]i32{ 3, 3, 5, 0, 0, 3, 1, 4 };
    const profit = try maxProfitKTransactions(i32, allocator, 2, &prices);
    try testing.expectEqual(@as(i32, 6), profit);
}

test "maxProfitKTransactions - k=0 no transactions" {
    const allocator = testing.allocator;
    const prices = [_]i32{ 1, 2, 3, 4, 5 };
    const profit = try maxProfitKTransactions(i32, allocator, 0, &prices);
    try testing.expectEqual(@as(i32, 0), profit);
}

test "maxProfitKTransactions - k large reduces to unlimited" {
    const allocator = testing.allocator;
    const prices = [_]i32{ 1, 2, 3, 0, 2 };
    const profit = try maxProfitKTransactions(i32, allocator, 100, &prices);
    const unlimited = maxProfitUnlimited(i32, &prices);
    try testing.expectEqual(unlimited, profit);
}

test "maxProfitKTransactions - k=1 matches single transaction" {
    const allocator = testing.allocator;
    const prices = [_]i32{ 7, 1, 5, 3, 6, 4 };
    const profit = try maxProfitKTransactions(i32, allocator, 1, &prices);
    try testing.expectEqual(@as(i32, 5), profit);
}

test "maxProfitKTransactions - k=3 multiple transactions" {
    const allocator = testing.allocator;
    const prices = [_]i32{ 1, 2, 4, 2, 5, 7, 2, 4, 9, 0 };
    const profit = try maxProfitKTransactions(i32, allocator, 3, &prices);
    try testing.expectEqual(@as(i32, 15), profit); // (4-1) + (7-2) + (9-2) = 15
}

test "maxProfitKTransactions - empty prices" {
    const allocator = testing.allocator;
    const profit = try maxProfitKTransactions(i32, allocator, 2, &[_]i32{});
    try testing.expectEqual(@as(i32, 0), profit);
}

test "maxProfitKTransactions - memory safety" {
    const allocator = testing.allocator;
    const prices = [_]i32{ 1, 2, 3, 4, 5 };
    _ = try maxProfitKTransactions(i32, allocator, 2, &prices);
    // Should not leak memory
}

test "maxProfitWithCooldown - basic example" {
    const prices = [_]i32{ 1, 2, 3, 0, 2 };
    try testing.expectEqual(@as(i32, 3), maxProfitWithCooldown(i32, &prices)); // Buy 1→sell 3, cooldown, buy 0→sell 2
}

test "maxProfitWithCooldown - no cooldown needed" {
    const prices = [_]i32{ 1, 2, 4 };
    try testing.expectEqual(@as(i32, 3), maxProfitWithCooldown(i32, &prices)); // Buy 1, sell 4
}

test "maxProfitWithCooldown - multiple cooldowns" {
    const prices = [_]i32{ 1, 2, 1, 2, 1, 2 };
    try testing.expectEqual(@as(i32, 3), maxProfitWithCooldown(i32, &prices)); // (2-1) + (2-1) + (2-1) = 3
}

test "maxProfitWithCooldown - decreasing prices" {
    const prices = [_]i32{ 5, 4, 3, 2, 1 };
    try testing.expectEqual(@as(i32, 0), maxProfitWithCooldown(i32, &prices));
}

test "maxProfitWithCooldown - f64 support" {
    const prices = [_]f64{ 1.0, 2.0, 3.0, 0.0, 2.0 };
    try testing.expectApproxEqAbs(@as(f64, 3.0), maxProfitWithCooldown(f64, &prices), 1e-9);
}

test "maxProfitWithFee - basic example" {
    const prices = [_]i32{ 1, 3, 2, 8, 4, 9 };
    try testing.expectEqual(@as(i32, 8), maxProfitWithFee(i32, &prices, 2)); // (8-1-2) + (9-4-2) = 8
}

test "maxProfitWithFee - fee too high" {
    const prices = [_]i32{ 1, 2, 3, 4, 5 };
    try testing.expectEqual(@as(i32, 0), maxProfitWithFee(i32, &prices, 10)); // Fee exceeds profit
}

test "maxProfitWithFee - zero fee matches unlimited" {
    const prices = [_]i32{ 1, 2, 3, 0, 2 };
    const with_fee = maxProfitWithFee(i32, &prices, 0);
    const unlimited = maxProfitUnlimited(i32, &prices);
    try testing.expectEqual(unlimited, with_fee);
}

test "maxProfitWithFee - single transaction profitable" {
    const prices = [_]i32{ 1, 10 };
    try testing.expectEqual(@as(i32, 7), maxProfitWithFee(i32, &prices, 2)); // 10 - 1 - 2 = 7
}

test "maxProfitWithFee - f64 support" {
    const prices = [_]f64{ 1.0, 3.0, 2.0, 8.0 };
    try testing.expectApproxEqAbs(@as(f64, 5.5), maxProfitWithFee(f64, &prices, 1.5), 1e-9); // 8-1-1.5 = 5.5
}

test "maxProfitWithFee - large prices" {
    const prices = [_]i32{ 100, 200, 150, 300, 250, 400 };
    try testing.expectEqual(@as(i32, 295), maxProfitWithFee(i32, &prices, 5)); // (300-100-5) + (400-250-5) = 295
}
