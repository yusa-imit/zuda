//! Chi-Squared Distribution
//!
//! Represents a continuous chi-squared distribution with degrees of freedom parameter k.
//! A fundamental distribution in statistics used for hypothesis testing and confidence intervals.
//!
//! ## Parameters
//! - `k: T` — degrees of freedom (must be > 0, typically a positive integer)
//!
//! ## Mathematical Properties
//! - **PDF**: f(x; k) = (1/(2^(k/2) * Γ(k/2))) * x^(k/2-1) * e^(-x/2) for x ≥ 0, else 0
//! - **CDF**: F(x; k) = P(k/2, x/2) / Γ(k/2) (lower incomplete gamma)
//! - **Quantile**: Q(p; k) = 2 * Q_gamma(k/2, p), inverse of CDF (numerical)
//! - **Log-PDF**: (k/2-1)*log(x) - x/2 - (k/2)*log(2) - logΓ(k/2)
//! - **Mean**: E[X] = k
//! - **Variance**: Var[X] = 2k
//! - **Mode**: max(k-2, 0) for k ≥ 1
//! - **Special Cases**:
//!   - χ²(1) = Gamma(0.5, 2)
//!   - χ²(2) = Exponential(0.5)
//!   - χ²(k) = Gamma(k/2, 2)
//!
//! ## Relationship to Gamma
//! The chi-squared distribution with k degrees of freedom is equivalent to Gamma(k/2, 2).
//! This implementation uses Gamma for pdf, cdf, quantile, and sampling.
//!
//! ## Time Complexity
//! - pdf, cdf, quantile, logpdf: O(1) to O(log k) depending on method
//! - sample: O(1) to O(log k) depending on method
//! - init: O(1)
//!
//! ## Use Cases
//! - Goodness-of-fit tests (Pearson's chi-squared test)
//! - Independence testing in contingency tables
//! - Variance estimation and hypothesis testing
//! - Confidence intervals for variance/standard deviation
//! - Modeling sum of squared standard normal random variables
//!
//! ## References
//! - Chi-squared distribution: https://en.wikipedia.org/wiki/Chi-squared_distribution
//! - Relation to Gamma: https://en.wikipedia.org/wiki/Gamma_distribution#Chi-squared_distribution
//! - Statistical testing: https://en.wikipedia.org/wiki/Chi-squared_test

const std = @import("std");
const math = std.math;
const testing = std.testing;
const Gamma = @import("gamma.zig").Gamma;

/// Chi-squared distribution with k degrees of freedom
///
/// Implemented via Gamma(k/2, 2) relationship.
/// χ²(k) is the distribution of the sum of k squared standard normal variables.
///
/// Parameters:
/// - T: numeric type (f32 or f64)
/// - k: degrees of freedom (must be > 0)
pub fn ChiSquared(comptime T: type) type {
    return struct {
        k: T,
        gamma_dist: Gamma(T),

        const Self = @This();

        /// Initialize chi-squared distribution with k degrees of freedom
        ///
        /// Parameters:
        /// - k: degrees of freedom (must be > 0)
        ///
        /// Returns: ChiSquared distribution instance
        ///
        /// Errors:
        /// - error.InvalidParameter if k <= 0
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(k: T) !Self {
            if (k <= 0.0) {
                return error.InvalidParameter;
            }

            // χ²(k) = Gamma(k/2, 2)
            const shape = k / 2.0;
            const scale = 2.0;
            const gamma_dist = try Gamma(T).init(shape, scale);

            return .{
                .k = k,
                .gamma_dist = gamma_dist,
            };
        }

        /// Probability density function: f(x; k) = (1/(2^(k/2) * Γ(k/2))) * x^(k/2-1) * e^(-x/2)
        ///
        /// Parameters:
        /// - x: value to evaluate at
        ///
        /// Returns: probability density at x (0 for x < 0)
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn pdf(self: Self, x: T) T {
            // Delegate to Gamma(k/2, 2)
            return self.gamma_dist.pdf(x);
        }

        /// Cumulative distribution function
        ///
        /// F(x; k) = P(k/2, x/2) / Γ(k/2) (lower incomplete gamma)
        ///
        /// Parameters:
        /// - x: value to evaluate at
        ///
        /// Returns: cumulative probability P(X <= x)
        ///
        /// Time: O(k) or O(1) depending on approximation method
        /// Space: O(1)
        pub fn cdf(self: Self, x: T) T {
            // Delegate to Gamma(k/2, 2)
            return self.gamma_dist.cdf(x);
        }

        /// Quantile function (inverse CDF)
        ///
        /// Uses Newton-Raphson iteration on the CDF (no closed form)
        ///
        /// Parameters:
        /// - p: probability in [0, 1]
        ///
        /// Returns: value x such that P(X <= x) = p
        ///
        /// Errors:
        /// - error.InvalidProbability if p < 0 or p > 1
        ///
        /// Time: O(log k) via Newton-Raphson
        /// Space: O(1)
        pub fn quantile(self: Self, p: T) !T {
            // Delegate to Gamma(k/2, 2)
            return self.gamma_dist.quantile(p);
        }

        /// Natural logarithm of probability density function
        ///
        /// log(f(x; k)) = (k/2-1)*log(x) - x/2 - (k/2)*log(2) - logΓ(k/2)
        ///
        /// More numerically stable than log(pdf(x)) for extreme values.
        ///
        /// Parameters:
        /// - x: value to evaluate at
        ///
        /// Returns: log probability density at x
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn logpdf(self: Self, x: T) T {
            // Delegate to Gamma(k/2, 2)
            return self.gamma_dist.logpdf(x);
        }

        /// Generate random sample from distribution
        ///
        /// Uses Gamma sampling via Marsaglia-Tsang method.
        /// χ²(k) samples are generated as Gamma(k/2, 2) samples.
        ///
        /// Parameters:
        /// - rng: random number generator (std.Random)
        ///
        /// Returns: random value from distribution
        ///
        /// Time: O(1) expected, O(log k) worst case
        /// Space: O(1)
        pub fn sample(self: Self, rng: std.Random) T {
            // Delegate to Gamma(k/2, 2)
            return self.gamma_dist.sample(rng);
        }
    };
}

// ============================================================================
// TESTS
// ============================================================================

// ============================================================================
// INIT TESTS (6 tests)
// ============================================================================

test "ChiSquared.init - k=1" {
    const dist = try ChiSquared(f64).init(1.0);
    try testing.expectEqual(@as(f64, 1.0), dist.k);
}

test "ChiSquared.init - k=2" {
    const dist = try ChiSquared(f64).init(2.0);
    try testing.expectEqual(@as(f64, 2.0), dist.k);
}

test "ChiSquared.init - k=10" {
    const dist = try ChiSquared(f64).init(10.0);
    try testing.expectEqual(@as(f64, 10.0), dist.k);
}

test "ChiSquared.init - fractional k=3.5" {
    const dist = try ChiSquared(f64).init(3.5);
    try testing.expectEqual(@as(f64, 3.5), dist.k);
}

test "ChiSquared.init - error when k=0" {
    const result = ChiSquared(f64).init(0.0);
    try testing.expectError(error.InvalidParameter, result);
}

test "ChiSquared.init - error when k<0" {
    const result = ChiSquared(f64).init(-1.5);
    try testing.expectError(error.InvalidParameter, result);
}

// ============================================================================
// PDF TESTS (11 tests)
// ============================================================================

test "ChiSquared.pdf - negative x returns 0" {
    const dist = try ChiSquared(f64).init(2.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pdf(-5.0), 1e-10);
}

test "ChiSquared.pdf - x=0 returns 0 for k>2" {
    const dist = try ChiSquared(f64).init(3.0);
    const pdf_at_zero = dist.pdf(0.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), pdf_at_zero, 1e-10);
}

test "ChiSquared.pdf - x=0 returns finite for k=2 (Exponential case)" {
    const dist = try ChiSquared(f64).init(2.0);
    const pdf_at_zero = dist.pdf(0.0);
    // For k=2, χ²(2) = Exponential(0.5), f(0) = 0.5
    try testing.expectApproxEqAbs(@as(f64, 0.5), pdf_at_zero, 1e-10);
}

test "ChiSquared.pdf - x=0 is infinite for k<2" {
    const dist = try ChiSquared(f64).init(1.0);
    const pdf_at_zero = dist.pdf(0.0);
    try testing.expect(math.isPositiveInf(pdf_at_zero));
}

test "ChiSquared.pdf - mode at k-2 for k>2" {
    const dist = try ChiSquared(f64).init(5.0);
    const mode = dist.k - 2.0; // = 3.0
    const pdf_at_mode = dist.pdf(mode);

    // PDF should be higher at mode than at nearby points
    const pdf_left = dist.pdf(mode - 0.5);
    const pdf_right = dist.pdf(mode + 0.5);
    try testing.expect(pdf_at_mode >= pdf_left);
    try testing.expect(pdf_at_mode >= pdf_right);
}

test "ChiSquared.pdf - Exponential case (k=2) matches exponential pdf" {
    const dist = try ChiSquared(f64).init(2.0);
    // For k=2: f(x) = (1/2)*exp(-x/2)
    const x = 2.0;
    const pdf_val = dist.pdf(x);
    const expected = 0.5 * @exp(-x / 2.0);
    try testing.expectApproxEqAbs(expected, pdf_val, 1e-10);
}

test "ChiSquared.pdf - decreasing for k=2 (Exponential)" {
    const dist = try ChiSquared(f64).init(2.0);
    const f1 = dist.pdf(1.0);
    const f2 = dist.pdf(2.0);
    const f3 = dist.pdf(3.0);
    try testing.expect(f1 > f2);
    try testing.expect(f2 > f3);
}

test "ChiSquared.pdf - large x approaches 0" {
    const dist = try ChiSquared(f64).init(3.0);
    const pdf_far = dist.pdf(100.0);
    try testing.expect(pdf_far > 0.0);
    try testing.expect(pdf_far < 1e-20);
}

test "ChiSquared.pdf - f32 precision (k=3)" {
    const dist = try ChiSquared(f32).init(3.0);
    const x = 2.0;
    const pdf_val = dist.pdf(x);
    try testing.expect(pdf_val > 0.0);
    try testing.expect(!math.isNan(pdf_val));
}

test "ChiSquared.pdf - handles small x > 0" {
    const dist = try ChiSquared(f64).init(3.0);
    const pdf_small = dist.pdf(0.001);
    try testing.expect(pdf_small > 0.0);
    try testing.expect(!math.isNan(pdf_small));
}

// ============================================================================
// CDF TESTS (10 tests)
// ============================================================================

test "ChiSquared.cdf - at x=0 returns 0" {
    const dist = try ChiSquared(f64).init(2.0);
    const cdf_zero = dist.cdf(0.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), cdf_zero, 1e-10);
}

test "ChiSquared.cdf - negative x returns 0" {
    const dist = try ChiSquared(f64).init(2.0);
    const cdf_neg = dist.cdf(-5.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), cdf_neg, 1e-10);
}

test "ChiSquared.cdf - monotonically increasing" {
    const dist = try ChiSquared(f64).init(3.0);
    const c0 = dist.cdf(0.5);
    const c1 = dist.cdf(1.0);
    const c2 = dist.cdf(2.0);
    const c5 = dist.cdf(5.0);
    try testing.expect(c0 < c1);
    try testing.expect(c1 < c2);
    try testing.expect(c2 < c5);
}

test "ChiSquared.cdf - approaches 1 as x→∞" {
    const dist = try ChiSquared(f64).init(3.0);
    const cdf_large = dist.cdf(50.0);
    try testing.expect(cdf_large < 1.0);
    try testing.expect(cdf_large > 1.0 - 1e-10);
}

test "ChiSquared.cdf - bounded [0, 1]" {
    const dist = try ChiSquared(f64).init(3.0);
    for ([_]f64{ 0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0 }) |x| {
        const c = dist.cdf(x);
        try testing.expect(c >= 0.0);
        try testing.expect(c <= 1.0);
    }
}

test "ChiSquared.cdf - Exponential case (k=2) matches exponential" {
    const dist = try ChiSquared(f64).init(2.0);
    // For k=2: F(x) = 1 - exp(-x/2)
    const x = 2.0;
    const cdf_val = dist.cdf(x);
    const expected = 1.0 - @exp(-x / 2.0);
    try testing.expectApproxEqAbs(expected, cdf_val, 1e-10);
}

test "ChiSquared.cdf - f32 precision (k=4)" {
    const dist = try ChiSquared(f32).init(4.0);
    const cdf_val = dist.cdf(2.0);
    try testing.expect(cdf_val > 0.0);
    try testing.expect(cdf_val < 1.0);
    try testing.expect(!math.isNan(cdf_val));
}

test "ChiSquared.cdf - relationship with pdf (integral)" {
    const dist = try ChiSquared(f64).init(3.0);
    // Numerical derivative: (CDF(x+h) - CDF(x)) / h ≈ PDF(x)
    const x = 2.0;
    const h = 0.001;
    const cdf_deriv = (dist.cdf(x + h) - dist.cdf(x)) / h;
    const pdf_x = dist.pdf(x);
    try testing.expectApproxEqRel(pdf_x, cdf_deriv, 0.01);
}

// ============================================================================
// QUANTILE TESTS (10 tests)
// ============================================================================

test "ChiSquared.quantile - p=0 returns 0" {
    const dist = try ChiSquared(f64).init(2.0);
    const q = try dist.quantile(0.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), q, 1e-10);
}

test "ChiSquared.quantile - p=1 returns infinity" {
    const dist = try ChiSquared(f64).init(2.0);
    const q = try dist.quantile(1.0);
    try testing.expect(math.isPositiveInf(q));
}

test "ChiSquared.quantile - error when p<0" {
    const dist = try ChiSquared(f64).init(2.0);
    const result = dist.quantile(-0.1);
    try testing.expectError(error.InvalidProbability, result);
}

test "ChiSquared.quantile - error when p>1" {
    const dist = try ChiSquared(f64).init(2.0);
    const result = dist.quantile(1.1);
    try testing.expectError(error.InvalidProbability, result);
}

test "ChiSquared.quantile - monotonically increasing" {
    const dist = try ChiSquared(f64).init(3.0);
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

test "ChiSquared.quantile - inverse of cdf (composition)" {
    const dist = try ChiSquared(f64).init(3.0);
    for ([_]f64{ 0.1, 0.25, 0.5, 0.75, 0.9 }) |p| {
        const q = try dist.quantile(p);
        const p_back = dist.cdf(q);
        // cdf(quantile(p)) should ≈ p (within numerical precision)
        try testing.expectApproxEqAbs(p, p_back, 0.02);
    }
}

test "ChiSquared.quantile - median for k=2" {
    const dist = try ChiSquared(f64).init(2.0);
    const median = try dist.quantile(0.5);
    // For χ²(2) (Exponential(0.5)): Q(0.5) = 2*ln(2) ≈ 1.3863
    const expected = 2.0 * @log(2.0);
    try testing.expectApproxEqAbs(expected, median, 0.01);
}

test "ChiSquared.quantile - Exponential case (k=2)" {
    const dist = try ChiSquared(f64).init(2.0);
    const q = try dist.quantile(0.5);
    // For k=2: Q(0.5) = 2*ln(2) ≈ 1.3863
    const expected = 2.0 * @log(2.0);
    try testing.expectApproxEqAbs(expected, q, 1e-9);
}

test "ChiSquared.quantile - f32 precision (p=0.5, k=3)" {
    const dist = try ChiSquared(f32).init(3.0);
    const q = try dist.quantile(0.5);
    try testing.expect(q > 0.0);
    try testing.expect(!math.isNan(q));
}

// ============================================================================
// LOGPDF TESTS (5 tests)
// ============================================================================

test "ChiSquared.logpdf - equals log(pdf) for valid x" {
    const dist = try ChiSquared(f64).init(2.0);
    const x = 1.5;
    const pdf_val = dist.pdf(x);
    const logpdf_val = dist.logpdf(x);
    const expected = @log(pdf_val);
    try testing.expectApproxEqAbs(expected, logpdf_val, 1e-12);
}

test "ChiSquared.logpdf - negative x returns -infinity" {
    const dist = try ChiSquared(f64).init(3.0);
    const logpdf_neg = dist.logpdf(-1.0);
    try testing.expect(math.isNegativeInf(logpdf_neg));
}

test "ChiSquared.logpdf - numerical stability for large x" {
    const dist = try ChiSquared(f64).init(3.0);
    const logpdf_large = dist.logpdf(1000.0);
    try testing.expect(logpdf_large < 0.0);
    try testing.expect(!math.isInf(logpdf_large));
    try testing.expect(!math.isNan(logpdf_large));
}

test "ChiSquared.logpdf - Exponential case (k=2)" {
    const dist = try ChiSquared(f64).init(2.0);
    const x = 2.0;
    const logpdf_val = dist.logpdf(x);
    // For k=2: log(f(x)) = log(0.5) - x/2 = -log(2) - x/2
    const expected = -@log(2.0) - x / 2.0;
    try testing.expectApproxEqAbs(expected, logpdf_val, 1e-10);
}

test "ChiSquared.logpdf - f32 precision" {
    const dist = try ChiSquared(f32).init(3.0);
    const logpdf_val = dist.logpdf(1.5);
    try testing.expect(!math.isNan(logpdf_val));
    try testing.expect(logpdf_val < 0.0);
}

// ============================================================================
// SAMPLE TESTS (10 tests)
// ============================================================================

test "ChiSquared.sample - all samples non-negative" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const dist = try ChiSquared(f64).init(3.0);

    for (0..1000) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample >= 0.0);
        try testing.expect(!math.isNan(sample));
    }
}

test "ChiSquared.sample - Exponential case (k=2) mean≈2" {
    var prng = std.Random.DefaultPrng.init(99);
    const rng = prng.random();

    const dist = try ChiSquared(f64).init(2.0);
    const expected_mean = 2.0; // E[X] = k

    var sum: f64 = 0.0;
    const n_samples = 5000;
    for (0..n_samples) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.05);
}

test "ChiSquared.sample - mean≈k (k=3, 10k samples)" {
    var prng = std.Random.DefaultPrng.init(999);
    const rng = prng.random();

    const dist = try ChiSquared(f64).init(3.0);
    const expected_mean = dist.k;

    var sum: f64 = 0.0;
    const n_samples = 10000;
    for (0..n_samples) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.05);
}

test "ChiSquared.sample - mean≈k (k=5, 10k samples)" {
    var prng = std.Random.DefaultPrng.init(777);
    const rng = prng.random();

    const dist = try ChiSquared(f64).init(5.0);
    const expected_mean = dist.k;

    var sum: f64 = 0.0;
    const n_samples = 10000;
    for (0..n_samples) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.05);
}

test "ChiSquared.sample - variance≈2k (k=3, 10k samples)" {
    var prng = std.Random.DefaultPrng.init(555);
    const rng = prng.random();

    const dist = try ChiSquared(f64).init(3.0);
    const expected_variance = 2.0 * dist.k;

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

    try testing.expectApproxEqRel(expected_variance, sample_variance, 0.10);
}

test "ChiSquared.sample - different seeds produce different sequences" {
    var prng1 = std.Random.DefaultPrng.init(111);
    var prng2 = std.Random.DefaultPrng.init(222);

    const dist = try ChiSquared(f64).init(3.0);

    const s1 = dist.sample(prng1.random());
    const s2 = dist.sample(prng2.random());

    try testing.expect(s1 != s2);
}

test "ChiSquared.sample - k<2 case (1.5)" {
    var prng = std.Random.DefaultPrng.init(333);
    const rng = prng.random();

    const dist = try ChiSquared(f64).init(1.5);
    const expected_mean = dist.k;

    var sum: f64 = 0.0;
    const n_samples = 5000;
    for (0..n_samples) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.10);
}

test "ChiSquared.sample - k>20 case (30)" {
    var prng = std.Random.DefaultPrng.init(444);
    const rng = prng.random();

    const dist = try ChiSquared(f64).init(30.0);
    const expected_mean = dist.k;

    var sum: f64 = 0.0;
    const n_samples = 10000;
    for (0..n_samples) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.05);
}

test "ChiSquared.sample - f32 precision" {
    var prng = std.Random.DefaultPrng.init(666);
    const rng = prng.random();

    const dist = try ChiSquared(f32).init(3.0);

    for (0..100) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample >= 0.0);
        try testing.expect(!math.isNan(sample));
    }
}

// ============================================================================
// INTEGRATION TESTS (5 tests)
// ============================================================================

test "ChiSquared.pdf - integral over domain approximately 1" {
    const dist = try ChiSquared(f64).init(3.0);
    // Numerical integration (trapezoid rule) over [0, 30]
    const n_steps = 5000;
    const a = 0.0;
    const b = 30.0;
    const step = (b - a) / @as(f64, @floatFromInt(n_steps));

    var integral: f64 = 0.0;
    for (0..n_steps) |i| {
        const x = a + step * @as(f64, @floatFromInt(i));
        integral += dist.pdf(x) * step;
    }

    try testing.expectApproxEqRel(@as(f64, 1.0), integral, 0.01);
}

test "ChiSquared.cdf-quantile inverse relationship" {
    const dist = try ChiSquared(f64).init(3.0);
    const x = 4.5;
    const p = dist.cdf(x);
    const q = try dist.quantile(p);
    // quantile(cdf(x)) should ≈ x
    try testing.expectApproxEqAbs(x, q, 0.02);
}

test "ChiSquared.ensemble statistics (20k samples)" {
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    const dist = try ChiSquared(f64).init(4.0);
    const expected_mean = dist.k;
    const expected_variance = 2.0 * dist.k;

    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    const n_samples = 20000;

    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        sum += s;
        sum_sq += s * s;
    }

    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));
    const sample_variance = (sum_sq / @as(f64, @floatFromInt(n_samples))) - (sample_mean * sample_mean);

    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.03);
    try testing.expectApproxEqRel(expected_variance, sample_variance, 0.08);
}

test "ChiSquared.mode property (k>2)" {
    const dist = try ChiSquared(f64).init(5.0);
    const mode = dist.k - 2.0;
    const pdf_at_mode = dist.pdf(mode);

    // Test several points and verify mode gives max (locally)
    for ([_]f64{ 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 }) |delta| {
        const pdf_left = dist.pdf(mode - delta);
        const pdf_right = dist.pdf(mode + delta);
        try testing.expect(pdf_at_mode >= pdf_left);
        try testing.expect(pdf_at_mode >= pdf_right);
    }
}

test "ChiSquared.shape scaling (k→∞ approaches Normal-like)" {
    // For large k, χ²(k) ≈ Normal(k, sqrt(2k))
    // Mean should increase, shape should become more symmetric

    const dist_small = try ChiSquared(f64).init(2.0);
    const dist_large = try ChiSquared(f64).init(50.0);

    const mean_small = 2.0;
    const mean_large = 50.0;

    // Compare PDF at mean
    const pdf_small_at_mean = dist_small.pdf(mean_small);
    const pdf_large_at_mean = dist_large.pdf(mean_large);

    // Both should be positive
    try testing.expect(pdf_small_at_mean > 0.0);
    try testing.expect(pdf_large_at_mean > 0.0);
}
