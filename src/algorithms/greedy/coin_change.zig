//! Greedy coin change: repeatedly take as many of the largest remaining denomination as fit.
//!
//! Optimal only for canonical coin systems (such as US coins); for arbitrary systems use
//! dynamic programming. A zero-valued coin is ignored.
//!
//! Ownership: the temporary sorted copy of the coins is freed before every function returns. The
//! `ArrayList(CoinUsage)` returned by `greedyCoinChangeDetailed` is owned by the caller, who
//! releases it with `deinit(gpa)` using the same allocator.

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

/// Solves coin change problem using greedy approach
///
/// Algorithm: Sort coins descending, greedily use largest coin first
///
/// Time: O(n log n + k) — n = coins, k = iterations (amount/largest_coin)
/// Space: O(n) — temporary sorted array
///
/// Note: Greedy approach only works for canonical coin systems (e.g., US coins).
/// For arbitrary coin systems, use dynamic programming.
///
/// Example:
/// ```zig
/// const coins = [_]u64{ 1, 5, 10, 25 };
/// const count = try greedyCoinChange(gpa, &coins, 67);
/// // count = 6 (2×25 + 1×10 + 1×5 + 2×1)
/// ```
pub fn greedyCoinChange(
    gpa: Allocator,
    coins: []const u64,
    amount: u64,
) !u64 {
    if (coins.len == 0 or amount == 0) return 0;

    const sorted = try sortedDescending(gpa, coins);
    defer gpa.free(sorted);

    var remaining = amount;
    var coin_count: u64 = 0;

    for (sorted) |coin| {
        if (coin == 0) continue;
        if (remaining == 0) break;

        const num_coins = @divFloor(remaining, coin);
        coin_count += num_coins;
        remaining -= num_coins * coin;
    }

    assert(remaining <= amount);
    assert(coin_count <= amount);
    return coin_count;
}

/// Detailed coin change — returns breakdown of coins used
///
/// Time: O(n log n + k)
/// Space: O(n) — result storage
pub fn greedyCoinChangeDetailed(
    gpa: Allocator,
    coins: []const u64,
    amount: u64,
) !ArrayList(CoinUsage) {
    var result: ArrayList(CoinUsage) = .empty;
    errdefer result.deinit(gpa);

    if (coins.len == 0 or amount == 0) return result;

    const sorted = try sortedDescending(gpa, coins);
    defer gpa.free(sorted);

    // At most one entry per denomination, so a single allocation covers the whole result.
    try result.ensureTotalCapacityPrecise(gpa, coins.len);

    var remaining = amount;

    for (sorted) |coin| {
        if (coin == 0) continue;
        if (remaining == 0) break;

        const num_coins = @divFloor(remaining, coin);
        if (num_coins > 0) {
            result.appendAssumeCapacity(.{
                .denomination = coin,
                .count = num_coins,
            });
            remaining -= num_coins * coin;
        }
    }

    assert(remaining <= amount);
    return result;
}

/// Returns a freshly allocated copy of `coins` sorted largest first; the caller frees it.
fn sortedDescending(gpa: Allocator, coins: []const u64) ![]u64 {
    assert(coins.len > 0);

    const sorted = try gpa.alloc(u64, coins.len);
    errdefer gpa.free(sorted);

    @memcpy(sorted, coins);
    std.mem.sort(u64, sorted, {}, std.sort.desc(u64));
    assert(sorted[0] >= sorted[sorted.len - 1]);
    return sorted;
}

pub const CoinUsage = struct {
    denomination: u64,
    count: u64,
};

/// Checks if exact change is possible with greedy approach
///
/// Time: O(n log n + k)
/// Space: O(n) — temporary sorted array
pub fn canMakeChange(
    gpa: Allocator,
    coins: []const u64,
    amount: u64,
) !bool {
    if (amount == 0) return true;
    if (coins.len == 0) return false;

    const sorted = try sortedDescending(gpa, coins);
    defer gpa.free(sorted);

    var remaining = amount;

    for (sorted) |coin| {
        if (coin == 0) continue;
        if (remaining == 0) break;

        const num_coins = @divFloor(remaining, coin);
        remaining -= num_coins * coin;
    }

    assert(remaining <= amount);
    return remaining == 0;
}

/// Minimum coins for change (greedy approximation)
///
/// Note: This may not be optimal for non-canonical coin systems
///
/// Time: O(n log n + k)
/// Space: O(n) — temporary sorted arrays
pub fn minimumCoins(
    gpa: Allocator,
    coins: []const u64,
    amount: u64,
) !?u64 {
    if (amount == 0) return 0;
    if (coins.len == 0) return null;

    const count = try greedyCoinChange(gpa, coins, amount);

    // Verify if exact change was made
    if (try canMakeChange(gpa, coins, amount)) {
        return count;
    }

    return null;
}

test "greedy coin change - US coins" {
    const coins = [_]u64{ 1, 5, 10, 25 };

    // 67 cents = 2×25 + 1×10 + 1×5 + 2×1 = 6 coins
    try testing.expectEqual(@as(u64, 6), try greedyCoinChange(testing.allocator, &coins, 67));

    // 41 cents = 1×25 + 1×10 + 1×5 + 1×1 = 4 coins
    try testing.expectEqual(@as(u64, 4), try greedyCoinChange(testing.allocator, &coins, 41));

    // 99 cents = 3×25 + 2×10 + 4×1 = 9 coins
    try testing.expectEqual(@as(u64, 9), try greedyCoinChange(testing.allocator, &coins, 99));
}

test "greedy coin change - zero amount" {
    const coins = [_]u64{ 1, 5, 10 };
    try testing.expectEqual(@as(u64, 0), try greedyCoinChange(testing.allocator, &coins, 0));
}

test "greedy coin change - empty coins" {
    const coins: []const u64 = &.{};
    try testing.expectEqual(@as(u64, 0), try greedyCoinChange(testing.allocator, coins, 100));
}

test "greedy coin change - single coin type" {
    const coins = [_]u64{5};
    try testing.expectEqual(@as(u64, 4), try greedyCoinChange(testing.allocator, &coins, 20));
    try testing.expectEqual(@as(u64, 3), try greedyCoinChange(testing.allocator, &coins, 15));
}

test "greedy coin change - detailed breakdown" {
    const coins = [_]u64{ 1, 5, 10, 25 };
    var breakdown = try greedyCoinChangeDetailed(testing.allocator, &coins, 67);
    defer breakdown.deinit(testing.allocator);

    // Expected: 2×25, 1×10, 1×5, 2×1
    try testing.expectEqual(@as(usize, 4), breakdown.items.len);

    var total: u64 = 0;
    for (breakdown.items) |usage| {
        total += usage.denomination * usage.count;
    }
    try testing.expectEqual(@as(u64, 67), total);
}

test "can make change - exact" {
    const coins = [_]u64{ 1, 5, 10, 25 };
    try testing.expect(try canMakeChange(testing.allocator, &coins, 67));
    try testing.expect(try canMakeChange(testing.allocator, &coins, 0));
    try testing.expect(try canMakeChange(testing.allocator, &coins, 1));
}

test "can make change - impossible without 1 cent" {
    const coins = [_]u64{ 5, 10 };
    try testing.expect(!try canMakeChange(testing.allocator, &coins, 3)); // Cannot make 3 cents
    try testing.expect(try canMakeChange(testing.allocator, &coins, 10)); // Can make 10 cents
    try testing.expect(try canMakeChange(testing.allocator, &coins, 15)); // Can make 15 cents
}

test "minimum coins - US coins" {
    const coins = [_]u64{ 1, 5, 10, 25 };
    try testing.expectEqual(@as(?u64, 6), try minimumCoins(testing.allocator, &coins, 67));
    try testing.expectEqual(@as(?u64, 0), try minimumCoins(testing.allocator, &coins, 0));
}

test "minimum coins - impossible" {
    const coins = [_]u64{ 5, 10 };
    try testing.expectEqual(@as(?u64, null), try minimumCoins(testing.allocator, &coins, 3));
    // 10 is a single 10-coin, not two 5-coins: the largest denomination goes first.
    try testing.expectEqual(@as(?u64, 1), try minimumCoins(testing.allocator, &coins, 10));
}

test "greedy coin change - large amount" {
    const coins = [_]u64{ 1, 5, 10, 25, 100 };
    const amount = 1234;

    const count = try greedyCoinChange(testing.allocator, &coins, amount);
    // 12×100 + 1×25 + 1×5 + 4×1 = 18 coins
    try testing.expectEqual(@as(u64, 18), count);
}

test "greedy coin change - powers of 2" {
    const coins = [_]u64{ 1, 2, 4, 8, 16 };
    const amount = 31;

    const count = try greedyCoinChange(testing.allocator, &coins, amount);
    // 1×16 + 1×8 + 1×4 + 1×2 + 1×1 = 5 coins
    try testing.expectEqual(@as(u64, 5), count);
}

test "greedy coin change - non-canonical system" {
    // Example where greedy fails: coins = {1, 3, 4}, amount = 6
    // Greedy: 1×4 + 2×1 = 3 coins
    // Optimal: 2×3 = 2 coins
    const coins = [_]u64{ 1, 3, 4 };
    const amount = 6;

    const greedy_count = try greedyCoinChange(testing.allocator, &coins, amount);
    // Greedy gives 3 coins (not optimal)
    try testing.expectEqual(@as(u64, 3), greedy_count);
    // This demonstrates greedy doesn't always work
}

test "greedy coin change - zero-valued coins are ignored" {
    const coins = [_]u64{ 0, 5, 0 };
    try testing.expectEqual(@as(u64, 3), try greedyCoinChange(testing.allocator, &coins, 15));
    try testing.expect(!try canMakeChange(testing.allocator, &coins, 3));
}

test "greedy coin change - detailed breakdown lists each denomination once" {
    const coins = [_]u64{ 1, 5, 10, 25 };
    var breakdown = try greedyCoinChangeDetailed(testing.allocator, &coins, 67);
    defer breakdown.deinit(testing.allocator);

    const expected = [_]CoinUsage{
        .{ .denomination = 25, .count = 2 },
        .{ .denomination = 10, .count = 1 },
        .{ .denomination = 5, .count = 1 },
        .{ .denomination = 1, .count = 2 },
    };
    try testing.expectEqualSlices(CoinUsage, &expected, breakdown.items);
}

test "greedy coin change - no leaks when an allocation fails" {
    const Runner = struct {
        fn run(gpa: Allocator) !void {
            const coins = [_]u64{ 1, 5, 10, 25 };
            try testing.expectEqual(@as(u64, 6), try greedyCoinChange(gpa, &coins, 67));
            try testing.expect(try canMakeChange(gpa, &coins, 67));
            try testing.expectEqual(@as(?u64, 6), try minimumCoins(gpa, &coins, 67));

            var breakdown = try greedyCoinChangeDetailed(gpa, &coins, 67);
            defer breakdown.deinit(gpa);

            try testing.expectEqual(@as(usize, 4), breakdown.items.len);
        }
    };

    try testing.checkAllAllocationFailures(testing.allocator, Runner.run, .{});
}
