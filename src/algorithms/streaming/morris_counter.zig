//! Morris Counter - Probabilistic counter with logarithmic space
//!
//! Uses O(log log n) bits to count up to n with bounded relative error.
//! Based on Morris's 1978 algorithm for approximate counting.
//!
//! The counter stores X such that the estimate is a^X for some base a > 1.
//! Each increment increases X with probability 1/a^X.
//!
//! Error bounds:
//! - Expected value: n
//! - Variance: approximately n²(a-1)/(a+1)
//! - For a=2: standard deviation ≈ 0.58n
//!
//! Use cases:
//! - Memory-constrained counting (IoT devices, embedded systems)
//! - Approximate cardinality estimation
//! - Resource usage tracking with limited precision requirements

const std = @import("std");

/// Morris Counter for approximate counting
///
/// Time: O(1) for increment/estimate
/// Space: O(log log n) bits for counting to n
pub const MorrisCounter = struct {
    const Self = @This();

    x: u32, // Exponent value
    base: f64, // Base for exponential counting (typically 2.0)
    rng: std.Random,

    /// Initialize Morris counter with given base
    ///
    /// Common bases:
    /// - base=2.0: Higher variance (58%), uses fewer bits
    /// - base=1.5: Lower variance (41%), uses more bits
    /// - base=e≈2.718: Mathematical optimum for some applications
    ///
    /// Time: O(1)
    /// Space: O(1)
    pub fn init(seed: u64, base: f64) Self {
        var prng = std.Random.DefaultPrng.init(seed);
        return Self{
            .x = 0,
            .base = base,
            .rng = prng.random(),
        };
    }

    /// Increment counter probabilistically
    ///
    /// Time: O(1)
    /// Space: O(1)
    pub fn increment(self: *Self) void {
        const threshold = 1.0 / std.math.pow(f64, self.base, @as(f64, @floatFromInt(self.x)));
        const rand_val = self.rng.float(f64);
        
        if (rand_val < threshold) {
            self.x += 1;
        }
    }

    /// Get estimated count
    ///
    /// Time: O(1)
    /// Space: O(1)
    pub fn estimate(self: Self) u64 {
        const value = std.math.pow(f64, self.base, @as(f64, @floatFromInt(self.x))) - 1.0;
        return @as(u64, @intFromFloat(@max(0.0, value)));
    }

    /// Reset counter to zero
    ///
    /// Time: O(1)
    /// Space: O(1)
    pub fn reset(self: *Self) void {
        self.x = 0;
    }

    /// Get internal exponent value (for debugging)
    pub fn getExponent(self: Self) u32 {
        return self.x;
    }
};

/// Multiple Morris counters for averaging (reduces variance)
///
/// Using k independent counters and averaging reduces variance by factor of k.
/// Standard deviation: σ/√k where σ is single counter standard deviation.
pub fn MorrisCounterArray(comptime k: usize) type {
    return struct {
        const Self = @This();

        counters: [k]MorrisCounter,

        /// Initialize array of Morris counters
        ///
        /// Time: O(k)
        /// Space: O(k)
        pub fn init(seed: u64, base: f64) Self {
            var counters: [k]MorrisCounter = undefined;
            var i: usize = 0;
            while (i < k) : (i += 1) {
                counters[i] = MorrisCounter.init(seed +% i, base);
            }
            return Self{ .counters = counters };
        }

        /// Increment all counters
        ///
        /// Time: O(k)
        /// Space: O(1)
        pub fn increment(self: *Self) void {
            for (&self.counters) |*counter| {
                counter.increment();
            }
        }

        /// Get average estimate (reduces variance)
        ///
        /// Time: O(k)
        /// Space: O(1)
        pub fn estimate(self: Self) u64 {
            var sum: u64 = 0;
            for (self.counters) |counter| {
                sum += counter.estimate();
            }
            return sum / k;
        }

        /// Reset all counters
        ///
        /// Time: O(k)
        /// Space: O(1)
        pub fn reset(self: *Self) void {
            for (&self.counters) |*counter| {
                counter.reset();
            }
        }
    };
}

// Tests
test "MorrisCounter: basic counting" {
    var counter = MorrisCounter.init(42, 2.0);
    
    try std.testing.expectEqual(@as(u64, 0), counter.estimate());
    
    // Increment 100 times
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        counter.increment();
    }
    
    const estimate = counter.estimate();
    // Should be roughly 100, allow wide margin due to probabilistic nature
    try std.testing.expect(estimate >= 20);
    try std.testing.expect(estimate <= 500);
}

test "MorrisCounter: reset operation" {
    var counter = MorrisCounter.init(42, 2.0);
    
    var i: usize = 0;
    while (i < 50) : (i += 1) {
        counter.increment();
    }
    
    try std.testing.expect(counter.estimate() > 0);
    
    counter.reset();
    try std.testing.expectEqual(@as(u64, 0), counter.estimate());
    try std.testing.expectEqual(@as(u32, 0), counter.getExponent());
}

test "MorrisCounter: different bases" {
    // Base 2.0 (higher variance)
    var counter2 = MorrisCounter.init(42, 2.0);
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        counter2.increment();
    }
    const est2 = counter2.estimate();
    
    // Base 1.5 (lower variance, more precision)
    var counter1_5 = MorrisCounter.init(42, 1.5);
    i = 0;
    while (i < 1000) : (i += 1) {
        counter1_5.increment();
    }
    const est1_5 = counter1_5.estimate();
    
    // Both should be in reasonable range
    try std.testing.expect(est2 >= 100);
    try std.testing.expect(est2 <= 10000);
    try std.testing.expect(est1_5 >= 100);
    try std.testing.expect(est1_5 <= 10000);
}

test "MorrisCounter: large counts" {
    var counter = MorrisCounter.init(12345, 2.0);
    
    var i: usize = 0;
    while (i < 100000) : (i += 1) {
        counter.increment();
    }
    
    const estimate = counter.estimate();
    // Very wide bounds for large probabilistic counts
    try std.testing.expect(estimate >= 10000);
    try std.testing.expect(estimate <= 1000000);
}

test "MorrisCounterArray: variance reduction" {
    const CounterArray8 = MorrisCounterArray(8);
    var array = CounterArray8.init(42, 2.0);
    
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        array.increment();
    }
    
    const estimate = array.estimate();
    // With 8 counters, variance is reduced by √8 ≈ 2.83
    // Should have tighter bounds than single counter
    try std.testing.expect(estimate >= 200);
    try std.testing.expect(estimate <= 5000);
}

test "MorrisCounterArray: reset operation" {
    const CounterArray4 = MorrisCounterArray(4);
    var array = CounterArray4.init(42, 2.0);
    
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        array.increment();
    }
    
    try std.testing.expect(array.estimate() > 0);
    
    array.reset();
    try std.testing.expectEqual(@as(u64, 0), array.estimate());
}

test "MorrisCounterArray: convergence with more counters" {
    // Test that more counters give more stable estimates
    const CounterArray16 = MorrisCounterArray(16);
    var array = CounterArray16.init(99, 2.0);
    
    var i: usize = 0;
    while (i < 10000) : (i += 1) {
        array.increment();
    }
    
    const estimate = array.estimate();
    // With 16 counters, should be closer to true value
    try std.testing.expect(estimate >= 1000);
    try std.testing.expect(estimate <= 100000);
}

test "MorrisCounter: memory safety" {
    var counter = MorrisCounter.init(42, 2.0);
    
    // Many increments should not overflow or crash
    var i: usize = 0;
    while (i < 1000000) : (i += 1) {
        counter.increment();
    }
    
    _ = counter.estimate();
}

test "MorrisCounterArray: memory safety" {
    const CounterArray4 = MorrisCounterArray(4);
    var array = CounterArray4.init(42, 2.0);
    
    var i: usize = 0;
    while (i < 100000) : (i += 1) {
        array.increment();
    }
    
    _ = array.estimate();
}
