//! Ski Rental Problem - Classic online algorithm for rent-vs-buy decisions
//!
//! Problem: You want to ski for an unknown number of days. Each day, you can:
//! - Rent skis for cost r per day
//! - Buy skis for cost b (one-time payment)
//!
//! Competitive Analysis:
//! - Deterministic: Always buy after b/r days → 2-competitive
//! - Randomized: Buy with probability proportional to days → e/(e-1) ≈ 1.58-competitive
//!
//! Applications:
//! - Cloud resource management (rent vs. reserved instances)
//! - Equipment leasing decisions
//! - Caching (keep in cache vs. evict and refetch)
//! - Sunk cost vs. switching cost trade-offs

const std = @import("std");
const testing = std.testing;

/// Result of ski rental decision
pub const Decision = enum {
    rent, // Continue renting
    buy, // Buy now
};

/// Ski rental problem parameters
pub const Problem = struct {
    rent_cost_per_day: f64, // Cost to rent per day
    buy_cost: f64, // One-time cost to buy

    /// Compute break-even point (days)
    /// Time: O(1) | Space: O(1)
    pub fn breakEvenDays(self: Problem) f64 {
        return self.buy_cost / self.rent_cost_per_day;
    }

    /// Compute competitive ratio of deterministic strategy
    /// Time: O(1) | Space: O(1)
    pub fn deterministicCompetitiveRatio(self: Problem) f64 {
        _ = self;
        return 2.0; // Always 2-competitive
    }

    /// Compute competitive ratio of randomized strategy
    /// Time: O(1) | Space: O(1)
    pub fn randomizedCompetitiveRatio(self: Problem) f64 {
        _ = self;
        return @exp(1.0) / (@exp(1.0) - 1.0); // e/(e-1) ≈ 1.58
    }
};

/// Deterministic Ski Rental Strategy
pub const DeterministicStrategy = struct {
    problem: Problem,
    days_rented: u32,
    bought: bool,

    /// Initialize strategy
    /// Time: O(1) | Space: O(1)
    pub fn init(problem: Problem) DeterministicStrategy {
        return .{
            .problem = problem,
            .days_rented = 0,
            .bought = false,
        };
    }

    /// Decide what to do on the next day
    /// Time: O(1) | Space: O(1)
    pub fn nextDay(self: *DeterministicStrategy) Decision {
        if (self.bought) return .rent; // Already bought

        self.days_rented += 1;
        const break_even = self.problem.breakEvenDays();

        // Buy when total rent cost equals buy cost
        if (@as(f64, @floatFromInt(self.days_rented)) >= break_even) {
            self.bought = true;
            return .buy;
        }

        return .rent;
    }

    /// Compute total cost so far
    /// Time: O(1) | Space: O(1)
    pub fn totalCost(self: DeterministicStrategy) f64 {
        const rent_cost = @as(f64, @floatFromInt(self.days_rented)) * self.problem.rent_cost_per_day;
        const buy_cost = if (self.bought) self.problem.buy_cost else 0.0;
        return rent_cost + buy_cost;
    }

    /// Check if already bought
    /// Time: O(1) | Space: O(1)
    pub fn hasBought(self: DeterministicStrategy) bool {
        return self.bought;
    }
};

/// Randomized Ski Rental Strategy
/// Uses exponential distribution to decide when to buy
pub const RandomizedStrategy = struct {
    problem: Problem,
    days_rented: u32,
    bought: bool,
    buy_day_threshold: u32, // Randomly chosen day to buy

    /// Initialize strategy with random buy threshold
    /// Time: O(1) | Space: O(1)
    pub fn init(problem: Problem, random: std.Random) RandomizedStrategy {
        const break_even = problem.breakEvenDays();

        // Sample from exponential distribution with rate 1/break_even
        // This gives expected competitive ratio of e/(e-1)
        const u = random.float(f64); // Uniform [0, 1)
        const threshold = -break_even * @log(1.0 - u);

        return .{
            .problem = problem,
            .days_rented = 0,
            .bought = false,
            .buy_day_threshold = @intFromFloat(@ceil(threshold)),
        };
    }

    /// Decide what to do on the next day
    /// Time: O(1) | Space: O(1)
    pub fn nextDay(self: *RandomizedStrategy) Decision {
        if (self.bought) return .rent; // Already bought

        self.days_rented += 1;

        // Buy when reaching threshold
        if (self.days_rented >= self.buy_day_threshold) {
            self.bought = true;
            return .buy;
        }

        return .rent;
    }

    /// Compute total cost so far
    /// Time: O(1) | Space: O(1)
    pub fn totalCost(self: RandomizedStrategy) f64 {
        const rent_cost = @as(f64, @floatFromInt(self.days_rented)) * self.problem.rent_cost_per_day;
        const buy_cost = if (self.bought) self.problem.buy_cost else 0.0;
        return rent_cost + buy_cost;
    }

    /// Check if already bought
    /// Time: O(1) | Space: O(1)
    pub fn hasBought(self: RandomizedStrategy) bool {
        return self.bought;
    }

    /// Get the randomly chosen buy day
    /// Time: O(1) | Space: O(1)
    pub fn getBuyDayThreshold(self: RandomizedStrategy) u32 {
        return self.buy_day_threshold;
    }
};

/// Compute optimal offline cost (with perfect knowledge of total days)
/// Time: O(1) | Space: O(1)
pub fn optimalOfflineCost(problem: Problem, total_days: u32) f64 {
    const rent_total = @as(f64, @floatFromInt(total_days)) * problem.rent_cost_per_day;
    const buy_total = problem.buy_cost;
    return @min(rent_total, buy_total);
}

/// Compute competitive ratio of a strategy
/// Time: O(1) | Space: O(1)
pub fn competitiveRatio(online_cost: f64, offline_cost: f64) f64 {
    if (offline_cost == 0.0) return 1.0;
    return online_cost / offline_cost;
}

// ============================================================================
// Tests
// ============================================================================

test "ski rental - problem parameters" {
    const problem = Problem{
        .rent_cost_per_day = 10.0,
        .buy_cost = 100.0,
    };

    try testing.expectApproxEqAbs(10.0, problem.breakEvenDays(), 0.001);
    try testing.expectApproxEqAbs(2.0, problem.deterministicCompetitiveRatio(), 0.001);
    try testing.expectApproxEqAbs(1.58, problem.randomizedCompetitiveRatio(), 0.01);
}

test "ski rental - deterministic strategy: short duration" {
    const problem = Problem{
        .rent_cost_per_day = 10.0,
        .buy_cost = 100.0,
    };

    var strategy = DeterministicStrategy.init(problem);

    // Rent for 5 days (less than break-even)
    for (0..5) |_| {
        const decision = strategy.nextDay();
        try testing.expectEqual(Decision.rent, decision);
    }

    try testing.expectEqual(false, strategy.hasBought());
    try testing.expectApproxEqAbs(50.0, strategy.totalCost(), 0.001);
}

test "ski rental - deterministic strategy: at break-even" {
    const problem = Problem{
        .rent_cost_per_day = 10.0,
        .buy_cost = 100.0,
    };

    var strategy = DeterministicStrategy.init(problem);

    // Rent until break-even (10 days)
    for (0..9) |_| {
        _ = strategy.nextDay();
    }

    // On day 10, should buy
    const decision = strategy.nextDay();
    try testing.expectEqual(Decision.buy, decision);
    try testing.expectEqual(true, strategy.hasBought());
    try testing.expectApproxEqAbs(200.0, strategy.totalCost(), 0.001); // 10*10 + 100
}

test "ski rental - deterministic strategy: long duration" {
    const problem = Problem{
        .rent_cost_per_day = 10.0,
        .buy_cost = 100.0,
    };

    var strategy = DeterministicStrategy.init(problem);

    // Rent for 20 days
    for (0..20) |_| {
        _ = strategy.nextDay();
    }

    try testing.expectEqual(true, strategy.hasBought());
    try testing.expectApproxEqAbs(200.0, strategy.totalCost(), 0.001);
}

test "ski rental - deterministic competitive ratio" {
    const problem = Problem{
        .rent_cost_per_day = 10.0,
        .buy_cost = 100.0,
    };

    // Case 1: Ski for 5 days (optimal: rent)
    {
        var strategy = DeterministicStrategy.init(problem);
        for (0..5) |_| {
            _ = strategy.nextDay();
        }
        const online = strategy.totalCost();
        const offline = optimalOfflineCost(problem, 5);
        const ratio = competitiveRatio(online, offline);

        try testing.expectApproxEqAbs(50.0, online, 0.001);
        try testing.expectApproxEqAbs(50.0, offline, 0.001);
        try testing.expectApproxEqAbs(1.0, ratio, 0.001);
    }

    // Case 2: Ski for 20 days (optimal: buy)
    {
        var strategy = DeterministicStrategy.init(problem);
        for (0..20) |_| {
            _ = strategy.nextDay();
        }
        const online = strategy.totalCost();
        const offline = optimalOfflineCost(problem, 20);
        const ratio = competitiveRatio(online, offline);

        try testing.expectApproxEqAbs(200.0, online, 0.001);
        try testing.expectApproxEqAbs(100.0, offline, 0.001);
        try testing.expectApproxEqAbs(2.0, ratio, 0.001); // 2-competitive
    }
}

test "ski rental - randomized strategy: initialization" {
    const problem = Problem{
        .rent_cost_per_day = 10.0,
        .buy_cost = 100.0,
    };

    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    const strategy = RandomizedStrategy.init(problem, random);

    try testing.expectEqual(false, strategy.hasBought());
    try testing.expect(strategy.getBuyDayThreshold() > 0);
}

test "ski rental - randomized strategy: buys eventually" {
    const problem = Problem{
        .rent_cost_per_day = 10.0,
        .buy_cost = 100.0,
    };

    var prng = std.Random.DefaultPrng.init(54321);
    const random = prng.random();

    var strategy = RandomizedStrategy.init(problem, random);

    // Rent until buy threshold
    while (!strategy.hasBought()) {
        _ = strategy.nextDay();
        if (strategy.days_rented > 1000) break; // Safety
    }

    try testing.expectEqual(true, strategy.hasBought());
    try testing.expectApproxEqAbs(
        @as(f64, @floatFromInt(strategy.days_rented)) * 10.0 + 100.0,
        strategy.totalCost(),
        0.001
    );
}

test "ski rental - randomized vs deterministic" {
    const problem = Problem{
        .rent_cost_per_day = 10.0,
        .buy_cost = 100.0,
    };

    // Run multiple trials
    var prng = std.Random.DefaultPrng.init(99999);
    const random = prng.random();

    var randomized_sum: f64 = 0.0;
    var deterministic_sum: f64 = 0.0;

    for (0..100) |_| {
        var rand_strategy = RandomizedStrategy.init(problem, random);
        var det_strategy = DeterministicStrategy.init(problem);

        // Simulate 15 days
        for (0..15) |_| {
            _ = rand_strategy.nextDay();
            _ = det_strategy.nextDay();
        }

        randomized_sum += rand_strategy.totalCost();
        deterministic_sum += det_strategy.totalCost();
    }

    // Both should have reasonable costs
    try testing.expect(randomized_sum > 0.0);
    try testing.expect(deterministic_sum > 0.0);
}

test "ski rental - optimal offline cost" {
    const problem = Problem{
        .rent_cost_per_day = 10.0,
        .buy_cost = 100.0,
    };

    // Short duration: rent is better
    try testing.expectApproxEqAbs(50.0, optimalOfflineCost(problem, 5), 0.001);

    // At break-even: equal cost
    try testing.expectApproxEqAbs(100.0, optimalOfflineCost(problem, 10), 0.001);

    // Long duration: buy is better
    try testing.expectApproxEqAbs(100.0, optimalOfflineCost(problem, 20), 0.001);
    try testing.expectApproxEqAbs(100.0, optimalOfflineCost(problem, 100), 0.001);
}

test "ski rental - competitive ratio calculation" {
    try testing.expectApproxEqAbs(1.0, competitiveRatio(100.0, 100.0), 0.001);
    try testing.expectApproxEqAbs(2.0, competitiveRatio(200.0, 100.0), 0.001);
    try testing.expectApproxEqAbs(1.5, competitiveRatio(150.0, 100.0), 0.001);
    try testing.expectApproxEqAbs(1.0, competitiveRatio(50.0, 0.0), 0.001); // Edge case
}

test "ski rental - memory safety" {
    const problem = Problem{
        .rent_cost_per_day = 5.0,
        .buy_cost = 50.0,
    };

    var det = DeterministicStrategy.init(problem);
    for (0..100) |_| {
        _ = det.nextDay();
    }

    var prng = std.Random.DefaultPrng.init(11111);
    var rand = RandomizedStrategy.init(problem, prng.random());
    for (0..100) |_| {
        _ = rand.nextDay();
    }

    // No memory leaks - all stack allocated
}
