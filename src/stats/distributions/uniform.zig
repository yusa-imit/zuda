//! Continuous Uniform Distribution
//!
//! Represents a uniform (constant) probability distribution over a closed interval [a, b].
//! All values within the interval are equally likely.
//!
//! ## Parameters
//! - `a: T` — lower bound (minimum value)
//! - `b: T` — upper bound (maximum value), must satisfy a < b
//!
//! ## Mathematical Properties
//! - **PDF**: f(x; a, b) = 1/(b-a) for x in [a, b], else 0
//! - **CDF**: F(x; a, b) = 0 if x<a, (x-a)/(b-a) if a≤x≤b, 1 if x>b
//! - **Quantile**: Q(p; a, b) = a + p(b-a) for p in [0, 1]
//! - **Log-PDF**: log(1/(b-a)) = -log(b-a)
//! - **Mean**: (a + b) / 2
//! - **Variance**: (b - a)² / 12
//!
//! ## Time Complexity
//! - pdf, cdf, quantile, logpdf: O(1)
//! - sample: O(1)
//! - init: O(1)
//!
//! ## Use Cases
//! - Baseline hypothesis testing ("assume uniform distribution")
//! - Random sampling from bounded intervals
//! - Monte Carlo integration over [a, b]

const std = @import("std");
const math = std.math;
const testing = std.testing;

/// Continuous uniform distribution over interval [a, b]
///
/// Parameters:
/// - T: numeric type (f32 or f64)
/// - a: lower bound
/// - b: upper bound (must be > a)
pub fn Uniform(comptime T: type) type {
    return struct {
        a: T,
        b: T,

        const Self = @This();

        /// Initialize uniform distribution with bounds [a, b]
        ///
        /// Parameters:
        /// - a: lower bound
        /// - b: upper bound
        ///
        /// Returns: Uniform distribution instance
        ///
        /// Errors:
        /// - error.InvalidBounds if a >= b
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(a: T, b: T) !Self {
            if (a >= b) {
                return error.InvalidBounds;
            }
            return .{ .a = a, .b = b };
        }

        /// Probability density function: f(x; a, b) = 1/(b-a) for x in [a, b]
        ///
        /// Parameters:
        /// - x: value to evaluate at
        ///
        /// Returns: probability density at x
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn pdf(self: Self, x: T) T {
            if (x < self.a or x > self.b) {
                return 0.0;
            }
            return 1.0 / (self.b - self.a);
        }

        /// Cumulative distribution function
        ///
        /// F(x; a, b) = 0 if x < a
        ///             (x - a) / (b - a) if a <= x <= b
        ///             1 if x > b
        ///
        /// Parameters:
        /// - x: value to evaluate at
        ///
        /// Returns: cumulative probability P(X <= x)
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn cdf(self: Self, x: T) T {
            if (x < self.a) {
                return 0.0;
            }
            if (x > self.b) {
                return 1.0;
            }
            return (x - self.a) / (self.b - self.a);
        }

        /// Quantile function (inverse CDF): Q(p; a, b) = a + p(b - a)
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
            return self.a + p * (self.b - self.a);
        }

        /// Natural logarithm of probability density function
        ///
        /// log(f(x; a, b)) = log(1/(b-a)) = -log(b-a)
        ///
        /// Parameters:
        /// - x: value to evaluate at
        ///
        /// Returns: log probability density at x
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn logpdf(self: Self, x: T) T {
            if (x < self.a or x > self.b) {
                return -math.inf(T);
            }
            return -@log(self.b - self.a);
        }

        /// Generate random sample from distribution
        ///
        /// Uses inverse transform sampling: if U ~ Uniform(0,1),
        /// then X = a + U(b-a) ~ Uniform(a,b)
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
                else => @compileError("Uniform distribution only supports f32 and f64"),
            };
            return self.a + u * (self.b - self.a);
        }
    };
}

// ============================================================================
// TESTS
// ============================================================================

test "Uniform.init - standard [0,1]" {
    const dist = try Uniform(f64).init(0.0, 1.0);
    try testing.expectEqual(@as(f64, 0.0), dist.a);
    try testing.expectEqual(@as(f64, 1.0), dist.b);
}

test "Uniform.init - custom bounds [5,10]" {
    const dist = try Uniform(f64).init(5.0, 10.0);
    try testing.expectEqual(@as(f64, 5.0), dist.a);
    try testing.expectEqual(@as(f64, 10.0), dist.b);
}

test "Uniform.init - negative bounds [-10,-5]" {
    const dist = try Uniform(f64).init(-10.0, -5.0);
    try testing.expectEqual(@as(f64, -10.0), dist.a);
    try testing.expectEqual(@as(f64, -5.0), dist.b);
}

test "Uniform.init - error when a >= b (equal bounds)" {
    const result = Uniform(f64).init(5.0, 5.0);
    try testing.expectError(error.InvalidBounds, result);
}

test "Uniform.init - error when a > b (reversed bounds)" {
    const result = Uniform(f64).init(10.0, 5.0);
    try testing.expectError(error.InvalidBounds, result);
}

test "Uniform.init - f32 precision [0.5, 1.5]" {
    const dist = try Uniform(f32).init(0.5, 1.5);
    try testing.expectEqual(@as(f32, 0.5), dist.a);
    try testing.expectEqual(@as(f32, 1.5), dist.b);
}

test "Uniform.pdf - constant value inside [a,b]" {
    const dist = try Uniform(f64).init(0.0, 1.0);
    const pdf_val = dist.pdf(0.25);
    // PDF should be 1/(1-0) = 1 for all x in [0,1]
    try testing.expectApproxEqAbs(@as(f64, 1.0), pdf_val, 1e-10);
}

test "Uniform.pdf - value at lower bound a" {
    const dist = try Uniform(f64).init(2.0, 4.0);
    const pdf_val = dist.pdf(2.0);
    // PDF = 1/(4-2) = 0.5
    try testing.expectApproxEqAbs(@as(f64, 0.5), pdf_val, 1e-10);
}

test "Uniform.pdf - value at upper bound b" {
    const dist = try Uniform(f64).init(2.0, 4.0);
    const pdf_val = dist.pdf(4.0);
    // PDF = 1/(4-2) = 0.5
    try testing.expectApproxEqAbs(@as(f64, 0.5), pdf_val, 1e-10);
}

test "Uniform.pdf - midpoint [0,1]" {
    const dist = try Uniform(f64).init(0.0, 1.0);
    const pdf_val = dist.pdf(0.5);
    try testing.expectApproxEqAbs(@as(f64, 1.0), pdf_val, 1e-10);
}

test "Uniform.pdf - below lower bound" {
    const dist = try Uniform(f64).init(0.0, 1.0);
    const pdf_val = dist.pdf(-0.5);
    // PDF should be 0 for x < a
    try testing.expectApproxEqAbs(@as(f64, 0.0), pdf_val, 1e-10);
}

test "Uniform.pdf - above upper bound" {
    const dist = try Uniform(f64).init(0.0, 1.0);
    const pdf_val = dist.pdf(1.5);
    // PDF should be 0 for x > b
    try testing.expectApproxEqAbs(@as(f64, 0.0), pdf_val, 1e-10);
}

test "Uniform.pdf - custom interval [5,10]" {
    const dist = try Uniform(f64).init(5.0, 10.0);
    const pdf_val = dist.pdf(7.5);
    // PDF = 1/(10-5) = 0.2
    try testing.expectApproxEqAbs(@as(f64, 0.2), pdf_val, 1e-10);
}

test "Uniform.pdf - f32 precision" {
    const dist = try Uniform(f32).init(0.0, 2.0);
    const pdf_val = dist.pdf(1.0);
    // PDF = 1/(2-0) = 0.5
    try testing.expectApproxEqRel(@as(f32, 0.5), pdf_val, 1e-6);
}

test "Uniform.cdf - x below lower bound returns 0" {
    const dist = try Uniform(f64).init(0.0, 1.0);
    const cdf_val = dist.cdf(-0.5);
    try testing.expectApproxEqAbs(@as(f64, 0.0), cdf_val, 1e-10);
}

test "Uniform.cdf - x at lower bound a returns 0" {
    const dist = try Uniform(f64).init(2.0, 4.0);
    const cdf_val = dist.cdf(2.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), cdf_val, 1e-10);
}

test "Uniform.cdf - x at upper bound b returns 1" {
    const dist = try Uniform(f64).init(2.0, 4.0);
    const cdf_val = dist.cdf(4.0);
    try testing.expectApproxEqAbs(@as(f64, 1.0), cdf_val, 1e-10);
}

test "Uniform.cdf - x above upper bound returns 1" {
    const dist = try Uniform(f64).init(0.0, 1.0);
    const cdf_val = dist.cdf(2.0);
    try testing.expectApproxEqAbs(@as(f64, 1.0), cdf_val, 1e-10);
}

test "Uniform.cdf - midpoint returns 0.5" {
    const dist = try Uniform(f64).init(0.0, 1.0);
    const cdf_val = dist.cdf(0.5);
    try testing.expectApproxEqAbs(@as(f64, 0.5), cdf_val, 1e-10);
}

test "Uniform.cdf - monotonically increasing" {
    const dist = try Uniform(f64).init(0.0, 10.0);
    const c1 = dist.cdf(2.0);
    const c2 = dist.cdf(5.0);
    const c3 = dist.cdf(8.0);
    try testing.expect(c1 < c2);
    try testing.expect(c2 < c3);
}

test "Uniform.cdf - [5,10] interval at x=7.5" {
    const dist = try Uniform(f64).init(5.0, 10.0);
    const cdf_val = dist.cdf(7.5);
    // CDF = (7.5 - 5) / (10 - 5) = 2.5 / 5 = 0.5
    try testing.expectApproxEqAbs(@as(f64, 0.5), cdf_val, 1e-10);
}

test "Uniform.cdf - f32 precision" {
    const dist = try Uniform(f32).init(0.0, 4.0);
    const cdf_val = dist.cdf(2.0);
    // CDF = (2 - 0) / (4 - 0) = 0.5
    try testing.expectApproxEqRel(@as(f32, 0.5), cdf_val, 1e-6);
}

test "Uniform.quantile - p=0 returns a" {
    const dist = try Uniform(f64).init(2.0, 4.0);
    const q = try dist.quantile(0.0);
    try testing.expectApproxEqAbs(@as(f64, 2.0), q, 1e-10);
}

test "Uniform.quantile - p=1 returns b" {
    const dist = try Uniform(f64).init(2.0, 4.0);
    const q = try dist.quantile(1.0);
    try testing.expectApproxEqAbs(@as(f64, 4.0), q, 1e-10);
}

test "Uniform.quantile - p=0.5 returns midpoint" {
    const dist = try Uniform(f64).init(0.0, 1.0);
    const q = try dist.quantile(0.5);
    try testing.expectApproxEqAbs(@as(f64, 0.5), q, 1e-10);
}

test "Uniform.quantile - p=0.5 for [5,10]" {
    const dist = try Uniform(f64).init(5.0, 10.0);
    const q = try dist.quantile(0.5);
    // Q(0.5) = 5 + 0.5(10-5) = 5 + 2.5 = 7.5
    try testing.expectApproxEqAbs(@as(f64, 7.5), q, 1e-10);
}

test "Uniform.quantile - p=0.25 for [0,4]" {
    const dist = try Uniform(f64).init(0.0, 4.0);
    const q = try dist.quantile(0.25);
    // Q(0.25) = 0 + 0.25(4-0) = 1.0
    try testing.expectApproxEqAbs(@as(f64, 1.0), q, 1e-10);
}

test "Uniform.quantile - p=0.75 for [0,4]" {
    const dist = try Uniform(f64).init(0.0, 4.0);
    const q = try dist.quantile(0.75);
    // Q(0.75) = 0 + 0.75(4-0) = 3.0
    try testing.expectApproxEqAbs(@as(f64, 3.0), q, 1e-10);
}

test "Uniform.quantile - error when p < 0" {
    const dist = try Uniform(f64).init(0.0, 1.0);
    const result = dist.quantile(-0.1);
    try testing.expectError(error.InvalidProbability, result);
}

test "Uniform.quantile - error when p > 1" {
    const dist = try Uniform(f64).init(0.0, 1.0);
    const result = dist.quantile(1.1);
    try testing.expectError(error.InvalidProbability, result);
}

test "Uniform.quantile - f32 precision" {
    const dist = try Uniform(f32).init(0.0, 10.0);
    const q = try dist.quantile(0.3);
    // Q(0.3) = 0 + 0.3(10-0) = 3.0
    try testing.expectApproxEqRel(@as(f32, 3.0), q, 1e-6);
}

test "Uniform.logpdf - constant for all x in [a,b]" {
    const dist = try Uniform(f64).init(1.0, 3.0);
    const log_pdf_1 = dist.logpdf(1.5);
    const log_pdf_2 = dist.logpdf(2.0);
    const log_pdf_3 = dist.logpdf(2.9);
    // All should equal log(1/(3-1)) = log(0.5) = -ln(2)
    try testing.expectApproxEqAbs(log_pdf_1, log_pdf_2, 1e-10);
    try testing.expectApproxEqAbs(log_pdf_2, log_pdf_3, 1e-10);
}

test "Uniform.logpdf - correct value [0,1]" {
    const dist = try Uniform(f64).init(0.0, 1.0);
    const log_pdf = dist.logpdf(0.5);
    // log(1/(1-0)) = log(1) = 0
    try testing.expectApproxEqAbs(@as(f64, 0.0), log_pdf, 1e-10);
}

test "Uniform.logpdf - correct value [5,10]" {
    const dist = try Uniform(f64).init(5.0, 10.0);
    const log_pdf = dist.logpdf(7.5);
    // log(1/(10-5)) = log(0.2) = -ln(5)
    const expected = -@log(@as(f64, 5.0));
    try testing.expectApproxEqAbs(expected, log_pdf, 1e-10);
}

test "Uniform.logpdf - equals log(pdf(x))" {
    const dist = try Uniform(f64).init(2.0, 8.0);
    const x = 5.0;
    const pdf_val = dist.pdf(x);
    const log_pdf_val = dist.logpdf(x);
    const expected = @log(pdf_val);
    try testing.expectApproxEqAbs(expected, log_pdf_val, 1e-10);
}

test "Uniform.logpdf - f32 precision" {
    const dist = try Uniform(f32).init(0.0, 2.0);
    const log_pdf = dist.logpdf(1.0);
    // log(1/2) = -ln(2) ≈ -0.693147
    const expected = -@log(@as(f32, 2.0));
    try testing.expectApproxEqRel(expected, log_pdf, 1e-5);
}

test "Uniform.sample - all samples in [a,b]" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const dist = try Uniform(f64).init(0.0, 1.0);

    for (0..1000) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample >= dist.a);
        try testing.expect(sample <= dist.b);
    }
}

test "Uniform.sample - all samples in custom [5,10]" {
    var prng = std.Random.DefaultPrng.init(123);
    const rng = prng.random();

    const dist = try Uniform(f64).init(5.0, 10.0);

    for (0..1000) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample >= 5.0);
        try testing.expect(sample <= 10.0);
    }
}

test "Uniform.sample - mean approximately (a+b)/2" {
    var prng = std.Random.DefaultPrng.init(999);
    const rng = prng.random();

    const dist = try Uniform(f64).init(2.0, 8.0);

    var sum: f64 = 0.0;
    const n_samples = 10000;
    const expected_mean = (dist.a + dist.b) / 2.0;
    for (0..n_samples) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    // With 10k samples, expect mean within ~2% of theoretical mean
    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.02);
}

test "Uniform.sample - variance approximately (b-a)²/12" {
    var prng = std.Random.DefaultPrng.init(777);
    const rng = prng.random();

    const dist = try Uniform(f64).init(0.0, 6.0);
    const expected_variance = (dist.b - dist.a) * (dist.b - dist.a) / 12.0;

    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    const n_samples = 10000;

    for (0..n_samples) |_| {
        const sample = dist.sample(rng);
        sum += sample;
        sum_sq += sample * sample;
    }

    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));
    const sample_variance = (sum_sq / @as(f64, @floatFromInt(n_samples))) - (sample_mean * sample_mean);

    // With 10k samples, expect variance within ~2% of theoretical variance
    try testing.expectApproxEqRel(expected_variance, sample_variance, 0.02);
}

test "Uniform.sample - negative bounds [-5,-1]" {
    var prng = std.Random.DefaultPrng.init(555);
    const rng = prng.random();

    const dist = try Uniform(f64).init(-5.0, -1.0);

    for (0..100) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample >= -5.0);
        try testing.expect(sample <= -1.0);
    }
}

test "Uniform.sample - f32 precision" {
    var prng = std.Random.DefaultPrng.init(333);
    const rng = prng.random();

    const dist = try Uniform(f32).init(0.0, 10.0);

    for (0..100) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample >= 0.0);
        try testing.expect(sample <= 10.0);
    }
}

test "Uniform.pdf - integral over [a,b] approximately 1" {
    const dist = try Uniform(f64).init(0.0, 10.0);
    // Use numerical integration (trapezoid rule) to verify PDF integrates to ~1
    const n_steps = 1000;
    const step = (dist.b - dist.a) / @as(f64, @floatFromInt(n_steps));

    var integral: f64 = 0.0;
    for (0..n_steps) |i| {
        const x = dist.a + step * @as(f64, @floatFromInt(i));
        integral += dist.pdf(x) * step;
    }

    try testing.expectApproxEqRel(@as(f64, 1.0), integral, 0.01);
}

test "Uniform.cdf - inverse relationship with quantile" {
    const dist = try Uniform(f64).init(2.0, 8.0);
    const x = 5.0;
    const p = dist.cdf(x);
    const q = try dist.quantile(p);
    // quantile(cdf(x)) should equal x
    try testing.expectApproxEqAbs(x, q, 1e-10);
}

test "Uniform.sample - f64 large interval [0,1e6]" {
    var prng = std.Random.DefaultPrng.init(111);
    const rng = prng.random();

    const dist = try Uniform(f64).init(0.0, 1e6);

    for (0..100) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample >= 0.0);
        try testing.expect(sample <= 1e6);
    }
}

test "Uniform.pdf - narrow interval [1.0, 1.01]" {
    const dist = try Uniform(f64).init(1.0, 1.01);
    const pdf_val = dist.pdf(1.005);
    // PDF = 1/(1.01 - 1) = 1/0.01 = 100
    try testing.expectApproxEqAbs(@as(f64, 100.0), pdf_val, 1e-8);
}

test "Uniform.quantile - multiple probabilities in sequence" {
    const dist = try Uniform(f64).init(0.0, 10.0);
    const q0 = try dist.quantile(0.0);
    const q1 = try dist.quantile(0.1);
    const q2 = try dist.quantile(0.2);
    const q3 = try dist.quantile(0.9);
    const q4 = try dist.quantile(1.0);

    try testing.expectApproxEqAbs(@as(f64, 0.0), q0, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), q1, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 2.0), q2, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 9.0), q3, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 10.0), q4, 1e-10);

    // Verify monotonicity
    try testing.expect(q0 < q1);
    try testing.expect(q1 < q2);
    try testing.expect(q2 < q3);
    try testing.expect(q3 < q4);
}
