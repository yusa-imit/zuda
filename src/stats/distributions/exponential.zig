//! Exponential Distribution
//!
//! Represents a continuous exponential distribution with rate parameter λ.
//! Models the time between events in a Poisson process.
//!
//! ## Parameters
//! - `lambda: T` — rate parameter (must be > 0)
//!
//! ## Mathematical Properties
//! - **PDF**: f(x; λ) = λ * exp(-λx) for x ≥ 0, else 0
//! - **CDF**: F(x; λ) = 1 - exp(-λx) for x ≥ 0, else 0
//! - **Quantile**: Q(p; λ) = -ln(1-p)/λ for p in [0, 1]
//! - **Log-PDF**: log(f(x)) = log(λ) - λx for x ≥ 0, else -∞
//! - **Mean**: 1/λ
//! - **Variance**: 1/λ²
//! - **Median**: ln(2)/λ
//! - **Mode**: x = 0 (maximum)
//! - **Memoryless Property**: P(X > s+t | X > s) = P(X > t)
//!
//! ## Time Complexity
//! - pdf, cdf, quantile, logpdf: O(1)
//! - sample: O(1)
//! - init: O(1)
//!
//! ## Use Cases
//! - Modeling waiting times between events (radioactive decay, customer arrivals)
//! - Reliability engineering (lifetime of components)
//! - Queueing theory (service times)
//! - Internet traffic modeling
//!
//! ## References
//! - Inverse transform sampling: https://en.wikipedia.org/wiki/Inverse_transform_sampling
//! - Memoryless property: https://en.wikipedia.org/wiki/Memorylessness

const std = @import("std");
const math = std.math;
const testing = std.testing;

/// Exponential distribution with rate parameter λ
///
/// Parameters:
/// - T: numeric type (f32 or f64)
/// - lambda: rate parameter (must be > 0)
pub fn Exponential(comptime T: type) type {
    return struct {
        lambda: T,

        const Self = @This();

        /// Initialize exponential distribution with rate λ
        ///
        /// Parameters:
        /// - lambda: rate parameter (must be > 0)
        ///
        /// Returns: Exponential distribution instance
        ///
        /// Errors:
        /// - error.InvalidRate if lambda <= 0
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(lambda: T) !Self {
            if (lambda <= 0.0) {
                return error.InvalidRate;
            }
            return .{ .lambda = lambda };
        }

        /// Probability density function: f(x; λ) = λ * exp(-λx) for x ≥ 0
        ///
        /// Parameters:
        /// - x: value to evaluate at
        ///
        /// Returns: probability density at x (0 for x < 0)
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn pdf(self: Self, x: T) T {
            if (x < 0.0) {
                return 0.0;
            }
            return self.lambda * @exp(-self.lambda * x);
        }

        /// Cumulative distribution function
        ///
        /// F(x; λ) = 1 - exp(-λx) for x ≥ 0, else 0
        ///
        /// Parameters:
        /// - x: value to evaluate at
        ///
        /// Returns: cumulative probability P(X <= x)
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn cdf(self: Self, x: T) T {
            if (x < 0.0) {
                return 0.0;
            }
            return 1.0 - @exp(-self.lambda * x);
        }

        /// Quantile function (inverse CDF)
        ///
        /// Q(p; λ) = -ln(1-p) / λ for p in [0, 1]
        ///
        /// Parameters:
        /// - p: probability in [0, 1]
        ///
        /// Returns: value x such that P(X <= x) = p
        ///
        /// Errors:
        /// - error.InvalidProbability if p < 0 or p > 1
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn quantile(self: Self, p: T) !T {
            if (p < 0.0 or p > 1.0) {
                return error.InvalidProbability;
            }

            // Handle boundary cases
            if (p == 0.0) {
                return 0.0;
            }
            if (p == 1.0) {
                return math.inf(T);
            }

            return -@log(1.0 - p) / self.lambda;
        }

        /// Natural logarithm of probability density function
        ///
        /// log(f(x; λ)) = log(λ) - λx for x ≥ 0, else -∞
        ///
        /// More numerically stable than log(pdf(x)) for large x.
        ///
        /// Parameters:
        /// - x: value to evaluate at
        ///
        /// Returns: log probability density at x
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn logpdf(self: Self, x: T) T {
            if (x < 0.0) {
                return -math.inf(T);
            }
            return @log(self.lambda) - self.lambda * x;
        }

        /// Generate random sample from distribution
        ///
        /// Uses inverse transform sampling: if U ~ Uniform(0,1), then
        /// X = -ln(U) / λ ~ Exponential(λ)
        ///
        /// Parameters:
        /// - rng: random number generator (std.Random)
        ///
        /// Returns: random value from distribution
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn sample(self: Self, rng: std.Random) T {
            const u = switch (T) {
                f32 => rng.float(f32),
                f64 => rng.float(f64),
                else => @compileError("Exponential distribution only supports f32 and f64"),
            };

            // Ensure u is not exactly 0 (avoid log(0))
            const u_safe = if (u == 0.0) 1e-15 else u;

            // Inverse transform sampling
            return -@log(u_safe) / self.lambda;
        }
    };
}

// ============================================================================
// TESTS
// ============================================================================

// ============================================================================
// INIT TESTS (6 tests)
// ============================================================================

test "Exponential.init - standard rate λ=1" {
    const dist = try Exponential(f64).init(1.0);
    try testing.expectEqual(@as(f64, 1.0), dist.lambda);
}

test "Exponential.init - custom rate λ=0.5" {
    const dist = try Exponential(f64).init(0.5);
    try testing.expectEqual(@as(f64, 0.5), dist.lambda);
}

test "Exponential.init - custom rate λ=2.0" {
    const dist = try Exponential(f64).init(2.0);
    try testing.expectEqual(@as(f64, 2.0), dist.lambda);
}

test "Exponential.init - large rate λ=100" {
    const dist = try Exponential(f64).init(100.0);
    try testing.expectEqual(@as(f64, 100.0), dist.lambda);
}

test "Exponential.init - error when lambda = 0" {
    const result = Exponential(f64).init(0.0);
    try testing.expectError(error.InvalidRate, result);
}

test "Exponential.init - error when lambda < 0" {
    const result = Exponential(f64).init(-1.5);
    try testing.expectError(error.InvalidRate, result);
}

// ============================================================================
// PDF TESTS (10 tests)
// ============================================================================

test "Exponential.pdf - maximum at x=0 (λ=1)" {
    const dist = try Exponential(f64).init(1.0);
    const at_zero = dist.pdf(0.0);
    const at_one = dist.pdf(1.0);
    // PDF should be highest at x=0
    try testing.expect(at_zero > at_one);
}

test "Exponential.pdf - value at x=0 equals lambda" {
    const dist = try Exponential(f64).init(1.0);
    const pdf_at_zero = dist.pdf(0.0);
    // For λ=1: f(0) = 1 * exp(0) = 1
    try testing.expectApproxEqAbs(@as(f64, 1.0), pdf_at_zero, 1e-10);
}

test "Exponential.pdf - exponential decay (λ=1)" {
    const dist = try Exponential(f64).init(1.0);
    const f0 = dist.pdf(0.0);
    const f1 = dist.pdf(1.0);
    // Should satisfy f(x+1) / f(x) ≈ exp(-λ) = 1/e ≈ 0.368
    const ratio = f1 / f0;
    const expected_ratio = 1.0 / math.e;
    try testing.expectApproxEqAbs(expected_ratio, ratio, 1e-10);
}

test "Exponential.pdf - negative x returns 0" {
    const dist = try Exponential(f64).init(1.0);
    const pdf_val = dist.pdf(-5.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), pdf_val, 1e-10);
}

test "Exponential.pdf - custom lambda λ=2" {
    const dist = try Exponential(f64).init(2.0);
    const pdf_at_zero = dist.pdf(0.0);
    // For λ=2: f(0) = 2 * exp(0) = 2
    try testing.expectApproxEqAbs(@as(f64, 2.0), pdf_at_zero, 1e-10);
}

test "Exponential.pdf - custom lambda λ=0.5" {
    const dist = try Exponential(f64).init(0.5);
    const pdf_at_zero = dist.pdf(0.0);
    // For λ=0.5: f(0) = 0.5 * exp(0) = 0.5
    try testing.expectApproxEqAbs(@as(f64, 0.5), pdf_at_zero, 1e-10);
}

test "Exponential.pdf - tails approach zero" {
    const dist = try Exponential(f64).init(1.0);
    const far_out = dist.pdf(50.0);
    try testing.expect(far_out > 0.0);
    try testing.expect(far_out < 1e-15);
}

test "Exponential.pdf - different lambdas: higher rate gives higher peak" {
    const dist1 = try Exponential(f64).init(0.5);
    const dist2 = try Exponential(f64).init(2.0);
    const pdf1_zero = dist1.pdf(0.0);
    const pdf2_zero = dist2.pdf(0.0);
    // Higher λ gives higher peak: f_λ₂(0) > f_λ₁(0)
    try testing.expect(pdf2_zero > pdf1_zero);
}

test "Exponential.pdf - f32 precision (λ=1)" {
    const dist = try Exponential(f32).init(1.0);
    const pdf_at_zero = dist.pdf(0.0);
    try testing.expectApproxEqRel(@as(f32, 1.0), pdf_at_zero, 1e-5);
}

// ============================================================================
// CDF TESTS (9 tests)
// ============================================================================

test "Exponential.cdf - at x=0 returns 0" {
    const dist = try Exponential(f64).init(1.0);
    const cdf_at_zero = dist.cdf(0.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), cdf_at_zero, 1e-10);
}

test "Exponential.cdf - negative x returns 0" {
    const dist = try Exponential(f64).init(1.0);
    const cdf_val = dist.cdf(-5.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), cdf_val, 1e-10);
}

test "Exponential.cdf - monotonically increasing" {
    const dist = try Exponential(f64).init(1.0);
    const c0 = dist.cdf(0.0);
    const c1 = dist.cdf(1.0);
    const c2 = dist.cdf(2.0);
    const c5 = dist.cdf(5.0);
    try testing.expect(c0 < c1);
    try testing.expect(c1 < c2);
    try testing.expect(c2 < c5);
}

test "Exponential.cdf - approaches 1 as x → ∞" {
    const dist = try Exponential(f64).init(1.0);
    const cdf_large = dist.cdf(30.0);
    try testing.expect(cdf_large < 1.0);
    try testing.expect(cdf_large > 1.0 - 1e-12);
}

test "Exponential.cdf - bounds [0, 1]" {
    const dist = try Exponential(f64).init(1.0);
    for ([_]f64{ -10.0, -1.0, 0.0, 0.5, 1.0, 2.0, 10.0, 100.0 }) |x| {
        const c = dist.cdf(x);
        try testing.expect(c >= 0.0);
        try testing.expect(c <= 1.0);
    }
}

test "Exponential.cdf - median at ln(2)/λ (λ=1)" {
    const dist = try Exponential(f64).init(1.0);
    const median_x = @log(2.0);
    const cdf_at_median = dist.cdf(median_x);
    // CDF at median should be 0.5
    try testing.expectApproxEqAbs(@as(f64, 0.5), cdf_at_median, 1e-10);
}

test "Exponential.cdf - median at ln(2)/λ (λ=2)" {
    const dist = try Exponential(f64).init(2.0);
    const median_x = @log(2.0) / 2.0;
    const cdf_at_median = dist.cdf(median_x);
    try testing.expectApproxEqAbs(@as(f64, 0.5), cdf_at_median, 1e-10);
}

test "Exponential.cdf - specific value (λ=1, x=1)" {
    const dist = try Exponential(f64).init(1.0);
    const cdf_val = dist.cdf(1.0);
    // F(1) = 1 - exp(-1) ≈ 0.6321
    const expected = 1.0 - @exp(-1.0);
    try testing.expectApproxEqAbs(expected, cdf_val, 1e-10);
}

test "Exponential.cdf - f32 precision at x=1 (λ=1)" {
    const dist = try Exponential(f32).init(1.0);
    const cdf_val = dist.cdf(1.0);
    const expected = 1.0 - @exp(-1.0);
    try testing.expectApproxEqRel(@as(f32, expected), cdf_val, 1e-5);
}

// ============================================================================
// QUANTILE TESTS (10 tests)
// ============================================================================

test "Exponential.quantile - p=0 returns 0" {
    const dist = try Exponential(f64).init(1.0);
    const q = try dist.quantile(0.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), q, 1e-10);
}

test "Exponential.quantile - p=1 returns infinity" {
    const dist = try Exponential(f64).init(1.0);
    const q = try dist.quantile(1.0);
    try testing.expect(math.isPositiveInf(q));
}

test "Exponential.quantile - p=0.5 returns median ln(2)/λ (λ=1)" {
    const dist = try Exponential(f64).init(1.0);
    const q = try dist.quantile(0.5);
    const expected = @log(2.0);
    try testing.expectApproxEqAbs(expected, q, 1e-10);
}

test "Exponential.quantile - p=0.5 returns median ln(2)/λ (λ=2)" {
    const dist = try Exponential(f64).init(2.0);
    const q = try dist.quantile(0.5);
    const expected = @log(2.0) / 2.0;
    try testing.expectApproxEqAbs(expected, q, 1e-10);
}

test "Exponential.quantile - error when p < 0" {
    const dist = try Exponential(f64).init(1.0);
    const result = dist.quantile(-0.1);
    try testing.expectError(error.InvalidProbability, result);
}

test "Exponential.quantile - error when p > 1" {
    const dist = try Exponential(f64).init(1.0);
    const result = dist.quantile(1.1);
    try testing.expectError(error.InvalidProbability, result);
}

test "Exponential.quantile - monotonically increasing" {
    const dist = try Exponential(f64).init(1.0);
    const q1 = try dist.quantile(0.1);
    const q2 = try dist.quantile(0.25);
    const q3 = try dist.quantile(0.5);
    const q4 = try dist.quantile(0.75);
    const q5 = try dist.quantile(0.9);
    try testing.expect(q1 < q2);
    try testing.expect(q2 < q3);
    try testing.expect(q3 < q4);
    try testing.expect(q4 < q5);
}

test "Exponential.quantile - different lambdas scale inversely" {
    const dist1 = try Exponential(f64).init(1.0);
    const dist2 = try Exponential(f64).init(2.0);
    const q1 = try dist1.quantile(0.5);
    const q2 = try dist2.quantile(0.5);
    // Higher λ should give lower quantile: Q_2(p) = Q_1(p) / 2
    try testing.expectApproxEqAbs(q1 / 2.0, q2, 1e-10);
}

test "Exponential.quantile - f32 precision (p=0.5, λ=1)" {
    const dist = try Exponential(f32).init(1.0);
    const q = try dist.quantile(0.5);
    const expected = @log(2.0);
    try testing.expectApproxEqRel(@as(f32, expected), q, 1e-5);
}

// ============================================================================
// LOGPDF TESTS (5 tests)
// ============================================================================

test "Exponential.logpdf - equals log(pdf) for valid x" {
    const dist = try Exponential(f64).init(1.0);
    const x = 0.5;
    const pdf_val = dist.pdf(x);
    const logpdf_val = dist.logpdf(x);
    const expected = @log(pdf_val);
    try testing.expectApproxEqAbs(expected, logpdf_val, 1e-12);
}

test "Exponential.logpdf - negative x returns -infinity" {
    const dist = try Exponential(f64).init(1.0);
    const logpdf_val = dist.logpdf(-5.0);
    try testing.expect(math.isNegativeInf(logpdf_val));
}

test "Exponential.logpdf - at x=0 equals log(lambda)" {
    const dist = try Exponential(f64).init(2.0);
    const logpdf_val = dist.logpdf(0.0);
    const expected = @log(2.0);
    try testing.expectApproxEqAbs(expected, logpdf_val, 1e-10);
}

test "Exponential.logpdf - numerical stability for large x" {
    const dist = try Exponential(f64).init(1.0);
    // For large x, logpdf should be very negative but finite
    const logpdf_large = dist.logpdf(1000.0);
    try testing.expect(logpdf_large < 0.0);
    try testing.expect(!math.isInf(logpdf_large));
    try testing.expect(!math.isNan(logpdf_large));
}

test "Exponential.logpdf - f32 precision" {
    const dist = try Exponential(f32).init(1.0);
    const logpdf_val = dist.logpdf(0.0);
    try testing.expectApproxEqRel(@as(f32, 0.0), logpdf_val, 1e-5);
}

// ============================================================================
// SAMPLE TESTS (8 tests)
// ============================================================================

test "Exponential.sample - all samples are non-negative" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const dist = try Exponential(f64).init(1.0);

    for (0..1000) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample >= 0.0);
        try testing.expect(!math.isNan(sample));
    }
}

test "Exponential.sample - mean approximately 1/λ (λ=1, 10k samples)" {
    var prng = std.Random.DefaultPrng.init(999);
    const rng = prng.random();

    const dist = try Exponential(f64).init(1.0);
    const expected_mean = 1.0 / dist.lambda;

    var sum: f64 = 0.0;
    const n_samples = 10000;
    for (0..n_samples) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    // With 10k samples, expect mean within ~4% of theoretical mean
    // Standard error = sqrt(variance/n) = sqrt(1/10000) = 0.01, so 4% ≈ 4σ
    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.04);
}

test "Exponential.sample - mean approximately 1/λ (λ=2, 10k samples)" {
    var prng = std.Random.DefaultPrng.init(777);
    const rng = prng.random();

    const dist = try Exponential(f64).init(2.0);
    const expected_mean = 1.0 / dist.lambda;

    var sum: f64 = 0.0;
    const n_samples = 10000;
    for (0..n_samples) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.02);
}

test "Exponential.sample - variance approximately 1/λ² (10k samples)" {
    var prng = std.Random.DefaultPrng.init(555);
    const rng = prng.random();

    const dist = try Exponential(f64).init(1.0);
    const expected_variance = 1.0 / (dist.lambda * dist.lambda);

    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    const n_samples = 10000;

    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        sum += s;
        sum_sq += s * s;
    }

    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));
    const sample_variance = (sum_sq / @as(f64, @floatFromInt(n_samples))) - (sample_mean * sample_mean);

    // With 10k samples, expect variance within ~5% of theoretical variance
    try testing.expectApproxEqRel(expected_variance, sample_variance, 0.05);
}

test "Exponential.sample - different seeds produce different sequences" {
    var prng1 = std.Random.DefaultPrng.init(111);
    var prng2 = std.Random.DefaultPrng.init(222);

    const dist = try Exponential(f64).init(1.0);

    const s1 = dist.sample(prng1.random());
    const s2 = dist.sample(prng2.random());

    // Different seeds should (with very high probability) produce different samples
    try testing.expect(s1 != s2);
}

test "Exponential.sample - small lambda (0.5) spreads samples wider" {
    var prng = std.Random.DefaultPrng.init(333);
    const rng = prng.random();

    const dist = try Exponential(f64).init(0.5);

    var sum: f64 = 0.0;
    const n_samples = 5000;
    for (0..n_samples) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    // Expected mean for λ=0.5 is 2.0
    try testing.expectApproxEqRel(@as(f64, 2.0), sample_mean, 0.03);
}

test "Exponential.sample - f32 precision" {
    var prng = std.Random.DefaultPrng.init(444);
    const rng = prng.random();

    const dist = try Exponential(f32).init(1.0);

    for (0..100) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample >= 0.0);
        try testing.expect(!math.isNan(sample));
    }
}

// ============================================================================
// INTEGRATION TESTS (5+ tests)
// ============================================================================

test "Exponential.pdf - integral over [0,∞) approximately 1" {
    const dist = try Exponential(f64).init(1.0);
    // Use numerical integration (trapezoid rule) over [0, 20]
    // (integral from 0 to 20 captures ~99.9% of probability)
    const n_steps = 5000;
    const a = 0.0;
    const b = 20.0;
    const step = (b - a) / @as(f64, @floatFromInt(n_steps));

    var integral: f64 = 0.0;
    for (0..n_steps) |i| {
        const x = a + step * @as(f64, @floatFromInt(i));
        integral += dist.pdf(x) * step;
    }

    try testing.expectApproxEqRel(@as(f64, 1.0), integral, 0.01);
}

test "Exponential.cdf - inverse relationship with quantile" {
    const dist = try Exponential(f64).init(1.5);
    const x = 1.0;
    const p = dist.cdf(x);
    const q = try dist.quantile(p);
    // quantile(cdf(x)) should equal x
    try testing.expectApproxEqAbs(x, q, 1e-10);
}

test "Exponential.memoryless - P(X > s+t | X > s) = P(X > t)" {
    const dist = try Exponential(f64).init(1.0);
    const s = 2.0;
    const t = 3.0;

    // P(X > s+t) = 1 - F(s+t)
    const p_st = 1.0 - dist.cdf(s + t);
    // P(X > s) = 1 - F(s)
    const p_s = 1.0 - dist.cdf(s);
    // P(X > t) = 1 - F(t)
    const p_t = 1.0 - dist.cdf(t);

    // Memoryless: P(X > s+t | X > s) = P(X > s+t) / P(X > s) = P(X > t)
    const conditional_prob = p_st / p_s;
    try testing.expectApproxEqAbs(p_t, conditional_prob, 1e-10);
}

test "Exponential.mode - PDF is maximum at x=0" {
    const dist = try Exponential(f64).init(1.0);
    const pdf_at_mode = dist.pdf(0.0);

    // Test several points and verify all have lower PDF
    for ([_]f64{ 0.1, 0.5, 1.0, 2.0, 5.0 }) |x| {
        const pdf_x = dist.pdf(x);
        try testing.expect(pdf_at_mode > pdf_x);
    }
}

test "Exponential.sample - ensemble statistics (mean ~1/λ, variance ~1/λ²)" {
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    const dist = try Exponential(f64).init(1.5);
    const expected_mean = 1.0 / dist.lambda;
    const expected_variance = 1.0 / (dist.lambda * dist.lambda);

    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    const n_samples = 10000;

    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        sum += s;
        sum_sq += s * s;
    }

    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));
    const sample_variance = (sum_sq / @as(f64, @floatFromInt(n_samples))) - (sample_mean * sample_mean);

    // Mean should be close to 1/λ
    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.03);

    // Variance should be close to 1/λ²
    try testing.expectApproxEqRel(expected_variance, sample_variance, 0.05);
}

test "Exponential.quantile - CDF-quantile composition for probabilities" {
    const dist = try Exponential(f64).init(2.0);

    for ([_]f64{ 0.1, 0.25, 0.5, 0.75, 0.9 }) |p| {
        const q = try dist.quantile(p);
        const p_back = dist.cdf(q);
        // cdf(quantile(p)) should equal p
        try testing.expectApproxEqAbs(p, p_back, 1e-10);
    }
}
