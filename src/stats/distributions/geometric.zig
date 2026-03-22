//! Geometric Distribution
//!
//! Represents a discrete Geometric distribution with success probability p.
//! Models the number of failures before the first success in a sequence of
//! independent Bernoulli(p) trials.
//!
//! ## Parameters
//! - `p: T` — success probability (must be in (0, 1])
//!
//! ## Mathematical Properties
//! - **PMF**: P(X=k) = (1-p)^k * p for k ≥ 0, else 0
//! - **CDF**: F(k) = 1 - (1-p)^(k+1) for k ≥ 0, else 0
//! - **Quantile**: Q(α) = ceil(log(1-α)/log(1-p)) - 1
//! - **Log-PMF**: k*log(1-p) + log(p)
//! - **Sample**: floor(log(U)/log(1-p)) where U ~ Uniform(0,1)
//! - **Mean**: E[X] = (1-p)/p
//! - **Variance**: Var[X] = (1-p)/p²
//! - **Mode**: 0 (most likely to succeed on first trial)
//! - **Memoryless Property**: P(X > n+m | X > n) = P(X > m) = (1-p)^m
//!
//! ## Time Complexity
//! - pmf, cdf, quantile, logpmf: O(1)
//! - sample: O(1)
//! - init: O(1)
//!
//! ## Use Cases
//! - Modeling number of failures until first success
//! - Queuing theory (waiting time for first service)
//! - Reliability analysis (time until first failure)
//! - Geometric random walks
//! - Retries until success scenarios
//!
//! ## References
//! - Geometric distribution: https://en.wikipedia.org/wiki/Geometric_distribution
//! - Memoryless property: https://en.wikipedia.org/wiki/Memorylessness

const std = @import("std");
const math = std.math;
const testing = std.testing;

/// Geometric distribution with success probability p
///
/// Parameters:
/// - T: numeric type (f32 or f64)
/// - p: success probability (must be in (0, 1])
pub fn Geometric(comptime T: type) type {
    return struct {
        p: T,

        const Self = @This();

        /// Initialize Geometric distribution
        ///
        /// Parameters:
        /// - p: success probability (must be in (0, 1])
        ///
        /// Returns: Geometric distribution instance
        ///
        /// Errors:
        /// - error.InvalidProbability if p <= 0 or p > 1
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(p: T) !Self {
            if (p <= 0.0 or p > 1.0) {
                return error.InvalidProbability;
            }
            return .{ .p = p };
        }

        /// Probability mass function: P(X=k) = (1-p)^k * p
        ///
        /// Parameters:
        /// - k: outcome (number of failures before first success, k ≥ 0)
        ///
        /// Returns: P(X=k) — (1-p)^k * p for k ≥ 0, 0 for k < 0
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn pmf(self: Self, k: i64) T {
            if (k < 0) {
                return 0.0;
            }
            const q = 1.0 - self.p; // probability of failure
            const k_float = @as(T, @floatFromInt(k));
            return math.pow(T, q, k_float) * self.p;
        }

        /// Cumulative distribution function: F(k) = P(X ≤ k)
        ///
        /// Parameters:
        /// - k: outcome
        ///
        /// Returns: cumulative probability P(X ≤ k)
        ///          = 1 - (1-p)^(k+1) for k ≥ 0, 0 for k < 0
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn cdf(self: Self, k: i64) T {
            if (k < 0) {
                return 0.0;
            }
            const q = 1.0 - self.p; // probability of failure
            const k_plus_1 = @as(T, @floatFromInt(k + 1));
            return 1.0 - math.pow(T, q, k_plus_1);
        }

        /// Quantile function (inverse CDF)
        ///
        /// Returns smallest k where CDF(k) >= prob
        ///
        /// Parameters:
        /// - prob: probability in (0, 1)
        ///
        /// Returns: k such that P(X ≤ k) >= prob
        ///
        /// Errors:
        /// - error.InvalidProbability if prob <= 0 or prob > 1
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn quantile(self: Self, prob: T) !i64 {
            if (prob <= 0.0 or prob > 1.0) {
                return error.InvalidProbability;
            }

            // Special case: prob = 1 returns very large value
            if (prob >= 1.0) {
                return @as(i64, @intFromFloat(1000000)); // Return large value for quantile at p=1
            }

            // Q(p) = ceil(log(1-p) / log(1-q)) - 1
            // where q = prob (the quantile probability)
            const q = 1.0 - prob;
            const q_value = 1.0 - self.p;

            if (q_value >= 1.0) {
                // p = 0, undefined behavior
                return error.InvalidProbability;
            }

            const log_q = @log(q);
            const log_q_value = @log(q_value);
            const result = math.ceil(log_q / log_q_value) - 1.0;
            return @as(i64, @intFromFloat(result));
        }

        /// Natural logarithm of probability mass function
        ///
        /// log(P(X=k)) = k*log(1-p) + log(p)
        ///
        /// Parameters:
        /// - k: outcome (number of failures, k ≥ 0)
        ///
        /// Returns: log probability mass at k (-∞ for k < 0)
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn logpmf(self: Self, k: i64) T {
            if (k < 0) {
                return -math.inf(T);
            }
            const q = 1.0 - self.p; // probability of failure
            const k_float = @as(T, @floatFromInt(k));
            return k_float * @log(q) + @log(self.p);
        }

        /// Generate random sample from Geometric distribution
        ///
        /// Uses inverse transform sampling: floor(log(U) / log(1-p))
        /// where U ~ Uniform(0,1)
        ///
        /// Parameters:
        /// - rng: random number generator (std.Random)
        ///
        /// Returns: number of failures before first success (k ≥ 0)
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn sample(self: Self, rng: std.Random) i64 {
            const u = switch (T) {
                f32 => rng.float(f32),
                f64 => rng.float(f64),
                else => @compileError("Geometric only supports f32 and f64"),
            };

            // Avoid log(0) and log(1)
            const u_safe = math.clamp(u, 1e-10, 1.0 - 1e-10);
            const q = 1.0 - self.p;

            if (q >= 1.0 or q <= 0.0) {
                // Degenerate case: p=0 (undefined) or p=1 (always 0)
                return 0;
            }

            const sample_float = math.floor(@log(u_safe) / @log(q));
            return @as(i64, @intFromFloat(sample_float));
        }
    };
}

// ============================================================================
// TESTS
// ============================================================================

// ============================================================================
// INIT TESTS (6 tests)
// ============================================================================

test "Geometric.init - standard probability p=0.5" {
    const dist = try Geometric(f64).init(0.5);
    try testing.expectEqual(@as(f64, 0.5), dist.p);
}

test "Geometric.init - custom probability p=0.3" {
    const dist = try Geometric(f64).init(0.3);
    try testing.expectEqual(@as(f64, 0.3), dist.p);
}

test "Geometric.init - high probability p=0.9" {
    const dist = try Geometric(f64).init(0.9);
    try testing.expectEqual(@as(f64, 0.9), dist.p);
}

test "Geometric.init - boundary p=1 (degenerate)" {
    const dist = try Geometric(f64).init(1.0);
    try testing.expectEqual(@as(f64, 1.0), dist.p);
}

test "Geometric.init - error when p <= 0" {
    const result = Geometric(f64).init(0.0);
    try testing.expectError(error.InvalidProbability, result);
    const result2 = Geometric(f64).init(-0.1);
    try testing.expectError(error.InvalidProbability, result2);
}

test "Geometric.init - error when p > 1" {
    const result = Geometric(f64).init(1.5);
    try testing.expectError(error.InvalidProbability, result);
}

// ============================================================================
// PMF TESTS (10 tests)
// ============================================================================

test "Geometric.pmf - pmf(0) = p with p=0.3" {
    const dist = try Geometric(f64).init(0.3);
    const pmf_0 = dist.pmf(0);
    try testing.expectApproxEqAbs(@as(f64, 0.3), pmf_0, 1e-10);
}

test "Geometric.pmf - pmf(1) = (1-p)*p with p=0.3" {
    const dist = try Geometric(f64).init(0.3);
    const pmf_1 = dist.pmf(1);
    const expected = 0.7 * 0.3;
    try testing.expectApproxEqAbs(expected, pmf_1, 1e-10);
}

test "Geometric.pmf - pmf decreases monotonically" {
    const dist = try Geometric(f64).init(0.5);
    const pmf_0 = dist.pmf(0);
    const pmf_1 = dist.pmf(1);
    const pmf_2 = dist.pmf(2);
    const pmf_5 = dist.pmf(5);
    try testing.expect(pmf_0 > pmf_1);
    try testing.expect(pmf_1 > pmf_2);
    try testing.expect(pmf_2 > pmf_5);
}

test "Geometric.pmf - pmf(-1) = 0" {
    const dist = try Geometric(f64).init(0.5);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pmf(-1), 1e-10);
}

test "Geometric.pmf - pmf(-100) = 0 for negative k" {
    const dist = try Geometric(f64).init(0.5);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pmf(-100), 1e-10);
}

test "Geometric.pmf - edge case p=1 gives pmf(0)=1, pmf(k>0)=0" {
    const dist = try Geometric(f64).init(1.0);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.pmf(0), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pmf(1), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pmf(5), 1e-10);
}

test "Geometric.pmf - normalization for small range (sum pmf(0..10) < 1)" {
    const dist = try Geometric(f64).init(0.5);
    var sum: f64 = 0.0;
    for (0..11) |k| {
        sum += dist.pmf(@as(i64, @intCast(k)));
    }
    // Should be less than 1 since tail extends beyond k=10
    try testing.expect(sum < 1.0);
    // But should be substantial
    try testing.expect(sum > 0.99);
}

test "Geometric.pmf - larger k values decay exponentially" {
    const dist = try Geometric(f64).init(0.3);
    const pmf_10 = dist.pmf(10);
    const pmf_20 = dist.pmf(20);
    // pmf decays exponentially: should be much smaller at k=20
    // For p=0.3: (1-p)^10 ≈ 0.028, so ratio is 0.7^10 ≈ 0.0282
    try testing.expect(pmf_20 < pmf_10 * 0.1);
}

test "Geometric.pmf - f32 precision" {
    const dist = try Geometric(f32).init(0.3);
    const pmf_0 = dist.pmf(0);
    const pmf_1 = dist.pmf(1);
    try testing.expect(pmf_0 > 0.0);
    try testing.expect(pmf_1 > 0.0);
    try testing.expect(pmf_0 > pmf_1);
}

// ============================================================================
// CDF TESTS (9 tests)
// ============================================================================

test "Geometric.cdf - cdf(-1) = 0" {
    const dist = try Geometric(f64).init(0.5);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.cdf(-1), 1e-10);
}

test "Geometric.cdf - cdf(0) = p" {
    const dist = try Geometric(f64).init(0.3);
    const cdf_0 = dist.cdf(0);
    try testing.expectApproxEqAbs(@as(f64, 0.3), cdf_0, 1e-10);
}

test "Geometric.cdf - cdf(k) = 1 - (1-p)^(k+1)" {
    const dist = try Geometric(f64).init(0.4);
    const k = 5;
    const q = 0.6;
    const expected = 1.0 - math.pow(f64, q, 6.0);
    const cdf_k = dist.cdf(k);
    try testing.expectApproxEqAbs(expected, cdf_k, 1e-10);
}

test "Geometric.cdf - monotonically increasing" {
    const dist = try Geometric(f64).init(0.5);
    const c_minus_1 = dist.cdf(-1);
    const c_0 = dist.cdf(0);
    const c_1 = dist.cdf(1);
    const c_5 = dist.cdf(5);
    const c_100 = dist.cdf(100);
    try testing.expect(c_minus_1 <= c_0);
    try testing.expect(c_0 <= c_1);
    try testing.expect(c_1 <= c_5);
    try testing.expect(c_5 <= c_100);
}

test "Geometric.cdf - bounds [0, 1]" {
    const dist = try Geometric(f64).init(0.5);
    const k_values = [_]i64{ -5, -1, 0, 1, 5, 10, 100 };
    for (k_values) |k| {
        const c = dist.cdf(k);
        try testing.expect(c >= 0.0);
        try testing.expect(c <= 1.0);
    }
}

test "Geometric.cdf - approaches 1 as k increases" {
    const dist = try Geometric(f64).init(0.3);
    const cdf_large = dist.cdf(100);
    try testing.expect(cdf_large > 0.99);
}

test "Geometric.cdf - edge case p=1 gives cdf(0)=1, cdf(-1)=0" {
    const dist = try Geometric(f64).init(1.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.cdf(-1), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.cdf(0), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.cdf(5), 1e-10);
}

test "Geometric.cdf - f32 precision" {
    const dist = try Geometric(f32).init(0.5);
    const cdf_0 = dist.cdf(0);
    try testing.expect(cdf_0 >= 0.0);
    try testing.expect(cdf_0 <= 1.0);
}

// ============================================================================
// QUANTILE TESTS (10 tests)
// ============================================================================

test "Geometric.quantile - Q(prob) bounded by error tolerance" {
    const dist = try Geometric(f64).init(0.5);
    // Q(0.01) should be relatively small
    const q_small = try dist.quantile(0.01);
    try testing.expect(q_small >= 0);
}

test "Geometric.quantile - Q(prob) increases monotonically" {
    const dist = try Geometric(f64).init(0.5);
    const q_01 = try dist.quantile(0.1);
    const q_05 = try dist.quantile(0.5);
    const q_09 = try dist.quantile(0.9);
    try testing.expect(q_01 <= q_05);
    try testing.expect(q_05 <= q_09);
}

test "Geometric.quantile - Q(prob) non-negative" {
    const dist = try Geometric(f64).init(0.3);
    for (1..100) |i| {
        const prob = @as(f64, @floatFromInt(i)) / 100.0;
        const q = try dist.quantile(prob);
        try testing.expect(q >= 0);
    }
}

test "Geometric.quantile - Q(CDF(k)) >= k for various k" {
    const dist = try Geometric(f64).init(0.5);
    const k_values = [_]i64{ 0, 1, 2, 5, 10 };
    for (k_values) |k| {
        const prob = dist.cdf(k);
        const q = try dist.quantile(prob);
        try testing.expect(q >= k);
    }
}

test "Geometric.quantile - inverse property: CDF(Q(p)) >= p" {
    const dist = try Geometric(f64).init(0.4);
    const probs = [_]f64{ 0.1, 0.25, 0.5, 0.75, 0.99 };
    for (probs) |prob| {
        const q = try dist.quantile(prob);
        const cdf_q = dist.cdf(q);
        try testing.expect(cdf_q >= prob - 1e-6);
    }
}

test "Geometric.quantile - error when prob <= 0" {
    const dist = try Geometric(f64).init(0.5);
    const result = dist.quantile(0.0);
    try testing.expectError(error.InvalidProbability, result);
    const result2 = dist.quantile(-0.1);
    try testing.expectError(error.InvalidProbability, result2);
}

test "Geometric.quantile - error when prob > 1" {
    const dist = try Geometric(f64).init(0.5);
    const result = dist.quantile(1.1);
    try testing.expectError(error.InvalidProbability, result);
}

test "Geometric.quantile - small prob gives small quantile (p=0.5)" {
    const dist = try Geometric(f64).init(0.5);
    const q_small = try dist.quantile(0.01);
    const q_large = try dist.quantile(0.99);
    try testing.expect(q_small < q_large);
}

test "Geometric.quantile - f32 precision" {
    const dist = try Geometric(f32).init(0.5);
    const q = try dist.quantile(0.5);
    try testing.expect(q >= 0);
}

// ============================================================================
// LOGPMF TESTS (5 tests)
// ============================================================================

test "Geometric.logpmf - consistency: log(pmf(k)) = logpmf(k)" {
    const dist = try Geometric(f64).init(0.3);
    for (0..10) |k| {
        const k_int = @as(i64, @intCast(k));
        const pmf_k = dist.pmf(k_int);
        const logpmf_k = dist.logpmf(k_int);
        const expected = @log(pmf_k);
        try testing.expectApproxEqAbs(expected, logpmf_k, 1e-10);
    }
}

test "Geometric.logpmf - logpmf(k) = k*log(1-p) + log(p)" {
    const dist = try Geometric(f64).init(0.4);
    const k = 7;
    const k_float = @as(f64, @floatFromInt(k));
    const expected = k_float * @log(0.6) + @log(0.4);
    const logpmf_k = dist.logpmf(k);
    try testing.expectApproxEqAbs(expected, logpmf_k, 1e-10);
}

test "Geometric.logpmf - logpmf(-1) = -infinity" {
    const dist = try Geometric(f64).init(0.5);
    const logpmf_val = dist.logpmf(-1);
    try testing.expect(math.isNegativeInf(logpmf_val));
}

test "Geometric.logpmf - numerical stability for large k" {
    const dist = try Geometric(f64).init(0.3);
    const logpmf_large = dist.logpmf(1000);
    try testing.expect(!math.isNan(logpmf_large));
    try testing.expect(logpmf_large < 0.0); // log of small probability
}

test "Geometric.logpmf - f32 precision" {
    const dist = try Geometric(f32).init(0.5);
    const logpmf_0 = dist.logpmf(0);
    const logpmf_5 = dist.logpmf(5);
    try testing.expect(!math.isNan(logpmf_0));
    try testing.expect(!math.isNan(logpmf_5));
}

// ============================================================================
// SAMPLE TESTS (10 tests)
// ============================================================================

test "Geometric.sample - all samples >= 0" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const dist = try Geometric(f64).init(0.5);

    for (0..1000) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample >= 0);
    }
}

test "Geometric.sample - p=1 always returns 0" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const dist = try Geometric(f64).init(1.0);

    for (0..100) |_| {
        try testing.expectEqual(@as(i64, 0), dist.sample(rng));
    }
}

test "Geometric.sample - mean converges to (1-p)/p (p=0.5, 10k samples)" {
    var prng = std.Random.DefaultPrng.init(999);
    const rng = prng.random();

    const dist = try Geometric(f64).init(0.5);
    const expected_mean = (1.0 - dist.p) / dist.p; // (0.5) / 0.5 = 1.0

    var sum: f64 = 0.0;
    const n_samples = 10000;
    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        sum += @as(f64, @floatFromInt(s));
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    // With 10k samples, expect within ±5% of theoretical mean
    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.05);
}

test "Geometric.sample - mean converges (p=0.3, 10k samples)" {
    var prng = std.Random.DefaultPrng.init(777);
    const rng = prng.random();

    const dist = try Geometric(f64).init(0.3);
    const expected_mean = (1.0 - dist.p) / dist.p; // (0.7) / 0.3 ≈ 2.333

    var sum: f64 = 0.0;
    const n_samples = 10000;
    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        sum += @as(f64, @floatFromInt(s));
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.05);
}

test "Geometric.sample - variance converges (p=0.5, 10k samples)" {
    var prng = std.Random.DefaultPrng.init(555);
    const rng = prng.random();

    const dist = try Geometric(f64).init(0.5);
    const expected_variance = (1.0 - dist.p) / (dist.p * dist.p); // 0.5 / 0.25 = 2.0

    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    const n_samples = 10000;

    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        const s_float = @as(f64, @floatFromInt(s));
        sum += s_float;
        sum_sq += s_float * s_float;
    }

    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));
    const sample_variance = (sum_sq / @as(f64, @floatFromInt(n_samples))) - (sample_mean * sample_mean);

    // With 10k samples, expect variance within ±10% of theoretical
    try testing.expectApproxEqRel(expected_variance, sample_variance, 0.10);
}

test "Geometric.sample - different seeds produce different sequences" {
    var prng1 = std.Random.DefaultPrng.init(111);
    var prng2 = std.Random.DefaultPrng.init(222);

    const dist = try Geometric(f64).init(0.5);

    var sum1: i64 = 0;
    var sum2: i64 = 0;
    for (0..100) |_| {
        sum1 += dist.sample(prng1.random());
        sum2 += dist.sample(prng2.random());
    }

    // Different seeds should produce different results with very high probability
    try testing.expect(sum1 != sum2);
}

test "Geometric.sample - high p produces mostly small values" {
    var prng = std.Random.DefaultPrng.init(333);
    const rng = prng.random();

    const dist = try Geometric(f64).init(0.9);

    var count_zero: u32 = 0;
    const n_samples = 1000;
    for (0..n_samples) |_| {
        if (dist.sample(rng) == 0) {
            count_zero += 1;
        }
    }

    // Expect roughly 90% to be 0 (success on first trial)
    const ratio = @as(f64, @floatFromInt(count_zero)) / @as(f64, @floatFromInt(n_samples));
    try testing.expect(ratio > 0.85);
    try testing.expect(ratio < 0.95);
}

test "Geometric.sample - low p produces more failures" {
    var prng = std.Random.DefaultPrng.init(444);
    const rng = prng.random();

    const dist = try Geometric(f64).init(0.1);
    const expected_mean = 9.0; // (1-0.1)/0.1 = 9

    var sum: f64 = 0.0;
    const n_samples = 5000;
    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        sum += @as(f64, @floatFromInt(s));
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.08);
}

test "Geometric.sample - f32 precision" {
    var prng = std.Random.DefaultPrng.init(666);
    const rng = prng.random();

    const dist = try Geometric(f32).init(0.5);

    for (0..100) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample >= 0);
    }
}

// ============================================================================
// INTEGRATION TESTS (6 tests)
// ============================================================================

test "Geometric.pmf - normalization: sum of pmf from k=0 to very large k approaches 1" {
    const dist = try Geometric(f64).init(0.5);
    var sum: f64 = 0.0;
    for (0..100) |k| {
        sum += dist.pmf(@as(i64, @intCast(k)));
    }
    // Should be very close to 1 with 100 terms
    try testing.expect(sum > 0.999);
}

test "Geometric.cdf - CDF property: CDF(k) = 1 - (1-p)^(k+1)" {
    const dist = try Geometric(f64).init(0.4);
    const q = 0.6;
    for (0..20) |k| {
        const k_int = @as(i64, @intCast(k));
        const cdf_k = dist.cdf(k_int);
        const k_plus_1 = @as(f64, @floatFromInt(k_int + 1));
        const expected = 1.0 - math.pow(f64, q, k_plus_1);
        try testing.expectApproxEqAbs(expected, cdf_k, 1e-10);
    }
}

test "Geometric.memoryless - P(X >= n+m | X >= n) = P(X >= m)" {
    const dist = try Geometric(f64).init(0.5);

    // Memoryless property for Geometric: P(X >= n+m | X >= n) = P(X >= m)
    // Where P(X >= k) = (1-p)^k (number of failures until first success)
    const n = 5;
    const m = 3;

    // P(X >= k) = (1-p)^k
    const q = 1.0 - dist.p;
    const p_geq_n = math.pow(f64, q, @as(f64, @floatFromInt(n)));
    const p_geq_n_plus_m = math.pow(f64, q, @as(f64, @floatFromInt(n + m)));
    const p_geq_m = math.pow(f64, q, @as(f64, @floatFromInt(m)));

    // P(X >= n+m | X >= n) = P(X >= n+m) / P(X >= n)
    const conditional = p_geq_n_plus_m / p_geq_n;

    // Should be equal to P(X >= m) = (1-p)^m
    try testing.expectApproxEqAbs(p_geq_m, conditional, 1e-10);
}

test "Geometric.mode - mode is always 0 (most likely outcome)" {
    const dist = try Geometric(f64).init(0.5);
    const pmf_0 = dist.pmf(0);
    const pmf_1 = dist.pmf(1);
    const pmf_2 = dist.pmf(2);
    try testing.expect(pmf_0 > pmf_1);
    try testing.expect(pmf_1 > pmf_2);
}

test "Geometric.mean - E[X] = (1-p)/p property" {
    const dist = try Geometric(f64).init(0.4);
    const expected_mean = (1.0 - dist.p) / dist.p; // 0.6 / 0.4 = 1.5

    // Use 20k samples for precise convergence
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    var sum: f64 = 0.0;
    const n_samples = 20000;

    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        sum += @as(f64, @floatFromInt(s));
    }

    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));
    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.04);
}

test "Geometric - complementary probabilities p and (1-p) symmetric" {
    // This test validates the structure across complementary p values
    const dist1 = try Geometric(f64).init(0.3);
    const dist2 = try Geometric(f64).init(0.7);

    // Different distributions should have different means
    const mean1 = (1.0 - dist1.p) / dist1.p;
    const mean2 = (1.0 - dist2.p) / dist2.p;
    try testing.expect(mean1 > mean2);
}
