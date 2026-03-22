//! Bernoulli Distribution
//!
//! Represents a discrete Bernoulli distribution with success probability p.
//! The simplest discrete probability distribution — models a single trial with two outcomes:
//! success (1) with probability p, or failure (0) with probability 1-p.
//!
//! ## Parameters
//! - `p: T` — success probability (must be in [0, 1])
//!
//! ## Mathematical Properties
//! - **PMF**: P(X=1) = p, P(X=0) = 1-p, P(X=k) = 0 for k ≠ 0,1
//! - **CDF**: F(k) = 0 if k<0, (1-p) if 0≤k<1, 1 if k≥1
//! - **Quantile**: Q(α) = 0 if α ≤ (1-p), else 1
//! - **Log-PMF**: log(p) if k=1, log(1-p) if k=0, -∞ else
//! - **Mean**: E[X] = p
//! - **Variance**: Var[X] = p(1-p)
//! - **Mode**: 1 if p > 0.5, 0 if p < 0.5 (both if p=0.5)
//!
//! ## Time Complexity
//! - pmf, cdf, quantile, logpmf: O(1)
//! - sample: O(1)
//! - init: O(1)
//!
//! ## Use Cases
//! - Binary outcomes (coin flips, success/failure experiments)
//! - Basic building block for Binomial distribution (sum of n independent Bernoulli trials)
//! - Modeling probabilities in machine learning (classification)
//! - A/B testing (binary outcomes)
//! - Quality control (pass/fail testing)
//!
//! ## References
//! - Bernoulli distribution: https://en.wikipedia.org/wiki/Bernoulli_distribution
//! - Special case of Binomial with n=1

const std = @import("std");
const math = std.math;
const testing = std.testing;

/// Bernoulli distribution with success probability p
///
/// Parameters:
/// - T: numeric type (f32 or f64)
/// - p: success probability (must be in [0, 1])
pub fn Bernoulli(comptime T: type) type {
    return struct {
        p: T,

        const Self = @This();

        /// Initialize Bernoulli distribution
        ///
        /// Parameters:
        /// - p: success probability (must be in [0, 1])
        ///
        /// Returns: Bernoulli distribution instance
        ///
        /// Errors:
        /// - error.InvalidProbability if p < 0 or p > 1
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(p: T) !Self {
            if (p < 0.0 or p > 1.0) {
                return error.InvalidProbability;
            }
            return .{ .p = p };
        }

        /// Probability mass function: P(X=k)
        ///
        /// Parameters:
        /// - k: outcome (0 or 1)
        ///
        /// Returns: P(X=k) — p for k=1, 1-p for k=0, 0 otherwise
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn pmf(self: Self, k: i64) T {
            if (k == 0) {
                return 1.0 - self.p;
            } else if (k == 1) {
                return self.p;
            } else {
                return 0.0;
            }
        }

        /// Cumulative distribution function: F(k) = P(X ≤ k)
        ///
        /// Parameters:
        /// - k: outcome
        ///
        /// Returns: cumulative probability P(X ≤ k)
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn cdf(self: Self, k: i64) T {
            if (k < 0) {
                return 0.0;
            } else if (k == 0) {
                return 1.0 - self.p;
            } else {
                return 1.0;
            }
        }

        /// Quantile function (inverse CDF)
        ///
        /// Returns smallest k where CDF(k) >= prob
        ///
        /// Parameters:
        /// - prob: probability in [0, 1]
        ///
        /// Returns: k such that P(X ≤ k) >= prob
        ///
        /// Errors:
        /// - error.InvalidProbability if prob < 0 or prob > 1
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn quantile(self: Self, prob: T) !i64 {
            if (prob < 0.0 or prob > 1.0) {
                return error.InvalidProbability;
            }

            if (prob <= 1.0 - self.p) {
                return 0;
            } else {
                return 1;
            }
        }

        /// Natural logarithm of probability mass function
        ///
        /// log(P(X=k)) = log(p) if k=1, log(1-p) if k=0, -∞ else
        ///
        /// Parameters:
        /// - k: outcome (0 or 1)
        ///
        /// Returns: log probability mass at k
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn logpmf(self: Self, k: i64) T {
            if (k == 0) {
                if (self.p == 1.0) {
                    return -math.inf(T);
                }
                return @log(1.0 - self.p);
            } else if (k == 1) {
                if (self.p == 0.0) {
                    return -math.inf(T);
                }
                return @log(self.p);
            } else {
                return -math.inf(T);
            }
        }

        /// Generate random sample from Bernoulli distribution
        ///
        /// Uses inverse transform sampling: return 1 if U ≤ p, else 0
        ///
        /// Parameters:
        /// - rng: random number generator (std.Random)
        ///
        /// Returns: 1 with probability p, 0 with probability 1-p
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn sample(self: Self, rng: std.Random) i64 {
            const u = switch (T) {
                f32 => rng.float(f32),
                f64 => rng.float(f64),
                else => @compileError("Bernoulli only supports f32 and f64"),
            };
            if (u < self.p) {
                return 1;
            } else {
                return 0;
            }
        }
    };
}

// ============================================================================
// TESTS
// ============================================================================

// ============================================================================
// INIT TESTS (6 tests)
// ============================================================================

test "Bernoulli.init - standard probability p=0.5" {
    const dist = try Bernoulli(f64).init(0.5);
    try testing.expectEqual(@as(f64, 0.5), dist.p);
}

test "Bernoulli.init - custom probability p=0.3" {
    const dist = try Bernoulli(f64).init(0.3);
    try testing.expectEqual(@as(f64, 0.3), dist.p);
}

test "Bernoulli.init - boundary p=0" {
    const dist = try Bernoulli(f64).init(0.0);
    try testing.expectEqual(@as(f64, 0.0), dist.p);
}

test "Bernoulli.init - boundary p=1" {
    const dist = try Bernoulli(f64).init(1.0);
    try testing.expectEqual(@as(f64, 1.0), dist.p);
}

test "Bernoulli.init - error when p < 0" {
    const result = Bernoulli(f64).init(-0.1);
    try testing.expectError(error.InvalidProbability, result);
}

test "Bernoulli.init - error when p > 1" {
    const result = Bernoulli(f64).init(1.5);
    try testing.expectError(error.InvalidProbability, result);
}

// ============================================================================
// PMF TESTS (10 tests)
// ============================================================================

test "Bernoulli.pmf - pmf(0) = 1-p with p=0.3" {
    const dist = try Bernoulli(f64).init(0.3);
    const pmf_0 = dist.pmf(0);
    try testing.expectApproxEqAbs(@as(f64, 0.7), pmf_0, 1e-10);
}

test "Bernoulli.pmf - pmf(1) = p with p=0.3" {
    const dist = try Bernoulli(f64).init(0.3);
    const pmf_1 = dist.pmf(1);
    try testing.expectApproxEqAbs(@as(f64, 0.3), pmf_1, 1e-10);
}

test "Bernoulli.pmf - pmf(-1) = 0" {
    const dist = try Bernoulli(f64).init(0.5);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pmf(-1), 1e-10);
}

test "Bernoulli.pmf - pmf(2) = 0" {
    const dist = try Bernoulli(f64).init(0.5);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pmf(2), 1e-10);
}

test "Bernoulli.pmf - pmf(100) = 0 for large k" {
    const dist = try Bernoulli(f64).init(0.5);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pmf(100), 1e-10);
}

test "Bernoulli.pmf - p=0 gives pmf(0)=1, pmf(1)=0" {
    const dist = try Bernoulli(f64).init(0.0);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.pmf(0), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pmf(1), 1e-10);
}

test "Bernoulli.pmf - p=1 gives pmf(0)=0, pmf(1)=1" {
    const dist = try Bernoulli(f64).init(1.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pmf(0), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.pmf(1), 1e-10);
}

test "Bernoulli.pmf - normalization: pmf(0) + pmf(1) = 1" {
    const dist = try Bernoulli(f64).init(0.7);
    const sum = dist.pmf(0) + dist.pmf(1);
    try testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-10);
}

test "Bernoulli.pmf - p=0.5 symmetric pmf(0) = pmf(1)" {
    const dist = try Bernoulli(f64).init(0.5);
    try testing.expectApproxEqAbs(dist.pmf(0), dist.pmf(1), 1e-10);
}

test "Bernoulli.pmf - f32 precision" {
    const dist = try Bernoulli(f32).init(0.3);
    const pmf_0 = dist.pmf(0);
    const pmf_1 = dist.pmf(1);
    try testing.expect(pmf_0 >= 0.0);
    try testing.expect(pmf_1 >= 0.0);
}

// ============================================================================
// CDF TESTS (9 tests)
// ============================================================================

test "Bernoulli.cdf - cdf(-1) = 0" {
    const dist = try Bernoulli(f64).init(0.5);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.cdf(-1), 1e-10);
}

test "Bernoulli.cdf - cdf(0) = 1-p with p=0.3" {
    const dist = try Bernoulli(f64).init(0.3);
    const cdf_0 = dist.cdf(0);
    try testing.expectApproxEqAbs(@as(f64, 0.7), cdf_0, 1e-10);
}

test "Bernoulli.cdf - cdf(1) = 1" {
    const dist = try Bernoulli(f64).init(0.5);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.cdf(1), 1e-10);
}

test "Bernoulli.cdf - cdf(100) = 1 for large k" {
    const dist = try Bernoulli(f64).init(0.5);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.cdf(100), 1e-10);
}

test "Bernoulli.cdf - monotonically increasing" {
    const dist = try Bernoulli(f64).init(0.5);
    const c_minus_1 = dist.cdf(-1);
    const c_0 = dist.cdf(0);
    const c_1 = dist.cdf(1);
    const c_2 = dist.cdf(2);
    try testing.expect(c_minus_1 <= c_0);
    try testing.expect(c_0 <= c_1);
    try testing.expect(c_1 <= c_2);
}

test "Bernoulli.cdf - bounds [0, 1] for various k" {
    const dist = try Bernoulli(f64).init(0.5);
    const k_values = [_]i64{ -5, -1, 0, 1, 2, 5, 10 };
    for (k_values) |k| {
        const c = dist.cdf(k);
        try testing.expect(c >= 0.0);
        try testing.expect(c <= 1.0);
    }
}

test "Bernoulli.cdf - p=0 gives cdf(0)=1, cdf(1)=1" {
    const dist = try Bernoulli(f64).init(0.0);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.cdf(0), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.cdf(1), 1e-10);
}

test "Bernoulli.cdf - p=1 gives cdf(0)=0, cdf(1)=1" {
    const dist = try Bernoulli(f64).init(1.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.cdf(0), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.cdf(1), 1e-10);
}

test "Bernoulli.cdf - f32 precision" {
    const dist = try Bernoulli(f32).init(0.5);
    const cdf_0 = dist.cdf(0);
    try testing.expect(cdf_0 >= 0.0);
    try testing.expect(cdf_0 <= 1.0);
}

// ============================================================================
// QUANTILE TESTS (10 tests)
// ============================================================================

test "Bernoulli.quantile - Q(0) = 0" {
    const dist = try Bernoulli(f64).init(0.5);
    const q = try dist.quantile(0.0);
    try testing.expectEqual(@as(i64, 0), q);
}

test "Bernoulli.quantile - Q(1) = 1" {
    const dist = try Bernoulli(f64).init(0.5);
    const q = try dist.quantile(1.0);
    try testing.expectEqual(@as(i64, 1), q);
}

test "Bernoulli.quantile - Q(α) = 0 if α ≤ 1-p (p=0.3)" {
    const dist = try Bernoulli(f64).init(0.3);
    // 1-p = 0.7, so Q(0.5) should be 0
    const q = try dist.quantile(0.5);
    try testing.expectEqual(@as(i64, 0), q);
}

test "Bernoulli.quantile - Q(α) = 1 if α > 1-p (p=0.3)" {
    const dist = try Bernoulli(f64).init(0.3);
    // 1-p = 0.7, so Q(0.8) should be 1
    const q = try dist.quantile(0.8);
    try testing.expectEqual(@as(i64, 1), q);
}

test "Bernoulli.quantile - Q(0.7) = 0 for p=0.3" {
    const dist = try Bernoulli(f64).init(0.3);
    const q = try dist.quantile(0.7);
    try testing.expectEqual(@as(i64, 0), q);
}

test "Bernoulli.quantile - Q(0.7001) = 1 for p=0.3" {
    const dist = try Bernoulli(f64).init(0.3);
    const q = try dist.quantile(0.7001);
    try testing.expectEqual(@as(i64, 1), q);
}

test "Bernoulli.quantile - error when prob < 0" {
    const dist = try Bernoulli(f64).init(0.5);
    const result = dist.quantile(-0.1);
    try testing.expectError(error.InvalidProbability, result);
}

test "Bernoulli.quantile - error when prob > 1" {
    const dist = try Bernoulli(f64).init(0.5);
    const result = dist.quantile(1.1);
    try testing.expectError(error.InvalidProbability, result);
}

test "Bernoulli.quantile - monotonically non-decreasing" {
    const dist = try Bernoulli(f64).init(0.5);
    const q1 = try dist.quantile(0.1);
    const q2 = try dist.quantile(0.5);
    const q3 = try dist.quantile(0.9);
    try testing.expect(q1 <= q2);
    try testing.expect(q2 <= q3);
}

test "Bernoulli.quantile - f32 precision" {
    const dist = try Bernoulli(f32).init(0.5);
    const q = try dist.quantile(0.5);
    try testing.expect(q >= 0);
    try testing.expect(q <= 1);
}

// ============================================================================
// LOGPMF TESTS (5 tests)
// ============================================================================

test "Bernoulli.logpmf - log(pmf(0)) equals logpmf(0) for p=0.3" {
    const dist = try Bernoulli(f64).init(0.3);
    const pmf_0 = dist.pmf(0);
    const logpmf_0 = dist.logpmf(0);
    const expected = @log(pmf_0);
    try testing.expectApproxEqAbs(expected, logpmf_0, 1e-10);
}

test "Bernoulli.logpmf - log(pmf(1)) equals logpmf(1) for p=0.3" {
    const dist = try Bernoulli(f64).init(0.3);
    const pmf_1 = dist.pmf(1);
    const logpmf_1 = dist.logpmf(1);
    const expected = @log(pmf_1);
    try testing.expectApproxEqAbs(expected, logpmf_1, 1e-10);
}

test "Bernoulli.logpmf - logpmf(-1) = -infinity" {
    const dist = try Bernoulli(f64).init(0.5);
    const logpmf_val = dist.logpmf(-1);
    try testing.expect(math.isNegativeInf(logpmf_val));
}

test "Bernoulli.logpmf - logpmf(2) = -infinity" {
    const dist = try Bernoulli(f64).init(0.5);
    const logpmf_val = dist.logpmf(2);
    try testing.expect(math.isNegativeInf(logpmf_val));
}

test "Bernoulli.logpmf - f32 precision" {
    const dist = try Bernoulli(f32).init(0.5);
    const logpmf_0 = dist.logpmf(0);
    const logpmf_1 = dist.logpmf(1);
    try testing.expect(!math.isNan(logpmf_0));
    try testing.expect(!math.isNan(logpmf_1));
}

// ============================================================================
// SAMPLE TESTS (10 tests)
// ============================================================================

test "Bernoulli.sample - all samples in range {0, 1}" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const dist = try Bernoulli(f64).init(0.5);

    for (0..1000) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample == 0 or sample == 1);
    }
}

test "Bernoulli.sample - p=0 always returns 0" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const dist = try Bernoulli(f64).init(0.0);

    for (0..100) |_| {
        try testing.expectEqual(@as(i64, 0), dist.sample(rng));
    }
}

test "Bernoulli.sample - p=1 always returns 1" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const dist = try Bernoulli(f64).init(1.0);

    for (0..100) |_| {
        try testing.expectEqual(@as(i64, 1), dist.sample(rng));
    }
}

test "Bernoulli.sample - mean approximately p (p=0.5, 10k samples)" {
    var prng = std.Random.DefaultPrng.init(999);
    const rng = prng.random();

    const dist = try Bernoulli(f64).init(0.5);
    const expected_mean = dist.p;

    var sum: f64 = 0.0;
    const n_samples = 10000;
    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        sum += @as(f64, @floatFromInt(s));
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    // With 10k samples, expect within ~5% of theoretical mean
    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.05);
}

test "Bernoulli.sample - mean approximately p (p=0.3, 10k samples)" {
    var prng = std.Random.DefaultPrng.init(777);
    const rng = prng.random();

    const dist = try Bernoulli(f64).init(0.3);
    const expected_mean = dist.p;

    var sum: f64 = 0.0;
    const n_samples = 10000;
    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        sum += @as(f64, @floatFromInt(s));
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.05);
}

test "Bernoulli.sample - variance approximately p(1-p)" {
    var prng = std.Random.DefaultPrng.init(555);
    const rng = prng.random();

    const dist = try Bernoulli(f64).init(0.5);
    const expected_variance = dist.p * (1.0 - dist.p);

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

    // With 10k samples, expect variance within ~8% of theoretical
    try testing.expectApproxEqRel(expected_variance, sample_variance, 0.08);
}

test "Bernoulli.sample - different seeds produce different sequences" {
    var prng1 = std.Random.DefaultPrng.init(111);
    var prng2 = std.Random.DefaultPrng.init(222);

    const dist = try Bernoulli(f64).init(0.5);

    const s1 = dist.sample(prng1.random());
    const s2 = dist.sample(prng2.random());

    // Different seeds should (with very high probability) produce different samples
    try testing.expect(s1 != s2);
}

test "Bernoulli.sample - p=0.7 produces more 1s than 0s" {
    var prng = std.Random.DefaultPrng.init(333);
    const rng = prng.random();

    const dist = try Bernoulli(f64).init(0.7);

    var count_ones: u32 = 0;
    const n_samples = 1000;
    for (0..n_samples) |_| {
        if (dist.sample(rng) == 1) {
            count_ones += 1;
        }
    }

    // Expect roughly 70% ones
    const ratio = @as(f64, @floatFromInt(count_ones)) / @as(f64, @floatFromInt(n_samples));
    try testing.expect(ratio > 0.65); // Allow ±5% variation
    try testing.expect(ratio < 0.75);
}

test "Bernoulli.sample - f32 precision" {
    var prng = std.Random.DefaultPrng.init(444);
    const rng = prng.random();

    const dist = try Bernoulli(f32).init(0.5);

    for (0..100) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample == 0 or sample == 1);
    }
}

// ============================================================================
// INTEGRATION TESTS (5 tests)
// ============================================================================

test "Bernoulli.pmf - normalization sums to 1" {
    const dist = try Bernoulli(f64).init(0.7);
    const sum = dist.pmf(0) + dist.pmf(1);
    try testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-10);
}

test "Bernoulli.cdf - and quantile are inverses" {
    const dist = try Bernoulli(f64).init(0.5);
    // For both outcomes, verify quantile(cdf(k)) >= k
    for (0..2) |k| {
        const k_int = @as(i64, @intCast(k));
        const p = dist.cdf(k_int);
        const q = try dist.quantile(p);
        try testing.expect(q >= k_int);
    }
}

test "Bernoulli.sample - ensemble statistics (p=0.5, 20k samples)" {
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    const dist = try Bernoulli(f64).init(0.5);
    const expected_mean = dist.p;
    const expected_variance = dist.p * (1.0 - dist.p);

    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    const n_samples = 20000;

    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        const s_float = @as(f64, @floatFromInt(s));
        sum += s_float;
        sum_sq += s_float * s_float;
    }

    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));
    const sample_variance = (sum_sq / @as(f64, @floatFromInt(n_samples))) - (sample_mean * sample_mean);

    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.05);
    try testing.expectApproxEqRel(expected_variance, sample_variance, 0.15);
}

test "Bernoulli - mode property: mode=1 if p>0.5, mode=0 if p<0.5" {
    // p=0.3: mode is 0
    const dist1 = try Bernoulli(f64).init(0.3);
    try testing.expect(dist1.pmf(0) > dist1.pmf(1));

    // p=0.7: mode is 1
    const dist2 = try Bernoulli(f64).init(0.7);
    try testing.expect(dist2.pmf(1) > dist2.pmf(0));

    // p=0.5: both equally likely
    const dist3 = try Bernoulli(f64).init(0.5);
    try testing.expectApproxEqAbs(dist3.pmf(0), dist3.pmf(1), 1e-10);
}

test "Bernoulli - compare p=0.2 vs p=0.8 (complementary)" {
    const dist1 = try Bernoulli(f64).init(0.2);
    const dist2 = try Bernoulli(f64).init(0.8);

    // pmf(0) of dist1 should equal pmf(1) of dist2
    try testing.expectApproxEqAbs(dist1.pmf(0), dist2.pmf(1), 1e-10);

    // pmf(1) of dist1 should equal pmf(0) of dist2
    try testing.expectApproxEqAbs(dist1.pmf(1), dist2.pmf(0), 1e-10);
}
