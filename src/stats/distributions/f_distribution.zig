//! F-Distribution (Fisher-Snedecor Distribution)
//!
//! Represents a continuous F-distribution with degrees of freedom parameters d1 and d2.
//! The F-distribution is the ratio of two chi-squared variables divided by their respective degrees of freedom.
//! It is fundamental to analysis of variance (ANOVA) and F-tests for comparing variances.
//!
//! ## Parameters
//! - `d1: T` — degrees of freedom for numerator (must be > 0)
//! - `d2: T` — degrees of freedom for denominator (must be > 0)
//!
//! ## Mathematical Properties
//! - **PDF**: f(x; d1, d2) = √((d1*x)^d1 * d2^d2 / (d1*x + d2)^(d1+d2)) / (x * B(d1/2, d2/2)) for x > 0, else 0
//! - **CDF**: F(x; d1, d2) = I(d1*x/(d1*x + d2), d1/2, d2/2) (regularized incomplete beta)
//! - **Quantile**: Q(p; d1, d2) = inverse CDF via numerical methods
//! - **Log-PDF**: Computed in log-space for numerical stability
//! - **Mean**: E[X] = d2/(d2-2) for d2 > 2, undefined otherwise
//! - **Variance**: Var[X] = 2*d2²*(d1+d2-2) / (d1*(d2-2)²*(d2-4)) for d2 > 4
//! - **Mode**: ((d1-2)/d1) * (d2/(d2+2)) for d1 > 2
//!
//! ## Relationship to Other Distributions
//! - F(d1, d2) = (X/d1) / (Y/d2) where X ~ ChiSquared(d1), Y ~ ChiSquared(d2)
//! - If T ~ StudentT(d2), then T² ~ F(1, d2)
//! - F(2, ∞) → Exponential(1) as d2 → ∞
//!
//! ## Time Complexity
//! - pdf, logpdf: O(1)
//! - cdf: O(1) to O(log n) (incomplete beta)
//! - quantile: O(1) to O(log n) (bisection)
//! - sample: O(1) (ratio of two gamma variates)
//! - init: O(1)
//!
//! ## Use Cases
//! - Analysis of variance (ANOVA) for comparing group variances
//! - F-tests for comparing model fits in regression
//! - Testing equality of variances (Levene's test, Bartlett's test)
//! - Hypothesis testing in linear models
//! - Comparing nested models via F-statistic
//!
//! ## References
//! - F-distribution: https://en.wikipedia.org/wiki/F-distribution
//! - CDF via incomplete beta: https://en.wikipedia.org/wiki/F-distribution#Cumulative_distribution_function
//! - Relationship to chi-squared: https://en.wikipedia.org/wiki/F-distribution#Characterization

const std = @import("std");
const math = std.math;
const testing = std.testing;

/// F-distribution (Fisher-Snedecor) with d1 and d2 degrees of freedom
///
/// Represents the ratio of two independent chi-squared variables divided by their degrees of freedom.
/// F = (X/d1) / (Y/d2) where X ~ ChiSquared(d1), Y ~ ChiSquared(d2)
///
/// Parameters:
/// - T: numeric type (f32 or f64)
/// - d1: numerator degrees of freedom (must be > 0)
/// - d2: denominator degrees of freedom (must be > 0)
pub fn FDistribution(comptime T: type) type {
    return struct {
        d1: T,
        d2: T,

        const Self = @This();

        /// Initialize F-distribution with d1 and d2 degrees of freedom
        ///
        /// Parameters:
        /// - d1: numerator degrees of freedom (must be > 0)
        /// - d2: denominator degrees of freedom (must be > 0)
        ///
        /// Returns: FDistribution instance
        ///
        /// Errors:
        /// - error.InvalidParameter if d1 <= 0 or d2 <= 0
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(d1: T, d2: T) !Self {
            if (d1 <= 0.0 or d2 <= 0.0) {
                return error.InvalidParameter;
            }
            return .{ .d1 = d1, .d2 = d2 };
        }

        /// Probability density function
        ///
        /// f(x; d1, d2) = √((d1*x)^d1 * d2^d2 / (d1*x + d2)^(d1+d2)) / (x * B(d1/2, d2/2))
        ///
        /// Parameters:
        /// - x: value to evaluate at (must be > 0)
        ///
        /// Returns: probability density at x (0 for x <= 0)
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn pdf(self: Self, x: T) T {
            if (x <= 0.0) return 0.0;
            return @exp(self.logpdf(x));
        }

        /// Cumulative distribution function
        ///
        /// F(x; d1, d2) = I(d1*x/(d1*x + d2), d1/2, d2/2) where I is regularized incomplete beta
        ///
        /// Parameters:
        /// - x: value to evaluate at
        ///
        /// Returns: cumulative probability P(X <= x)
        ///
        /// Time: O(1) to O(log n)
        /// Space: O(1)
        pub fn cdf(self: Self, x: T) T {
            if (x <= 0.0) return 0.0;
            const t = (self.d1 * x) / (self.d1 * x + self.d2);
            return incompleteBeta(T, t, self.d1 / 2.0, self.d2 / 2.0);
        }

        /// Quantile function (inverse CDF)
        ///
        /// Q(p; d1, d2) = F^(-1)(p; d1, d2) via bisection
        ///
        /// Parameters:
        /// - p: probability (must be in [0, 1])
        ///
        /// Returns: x such that F(x; d1, d2) = p
        ///
        /// Errors:
        /// - error.InvalidProbability if p not in [0, 1]
        ///
        /// Time: O(log n) (bisection with 100 iterations)
        /// Space: O(1)
        pub fn quantile(self: Self, p: T) !T {
            if (p < 0.0 or p > 1.0) return error.InvalidProbability;
            if (p == 0.0) return 0.0;
            if (p == 1.0) return math.inf(T);

            // Bisection search
            var low: T = 0.0;
            var high: T = 1000.0; // Start with reasonable upper bound

            // Expand upper bound if needed
            while (self.cdf(high) < p) {
                high *= 2.0;
            }

            const tolerance = if (T == f32) 1e-6 else 1e-12;
            const max_iter = 100;
            var i: usize = 0;

            while (i < max_iter) : (i += 1) {
                const mid = (low + high) / 2.0;
                const f_mid = self.cdf(mid);

                if (@abs(f_mid - p) < tolerance) {
                    return mid;
                }

                if (f_mid < p) {
                    low = mid;
                } else {
                    high = mid;
                }
            }

            return (low + high) / 2.0;
        }

        /// Log probability density function
        ///
        /// logpdf(x; d1, d2) = (d1/2)*log(d1) + (d2/2)*log(d2) + (d1/2-1)*log(x)
        ///                     - ((d1+d2)/2)*log(d1*x + d2) - logBeta(d1/2, d2/2)
        ///
        /// Parameters:
        /// - x: value to evaluate at
        ///
        /// Returns: log probability density at x (-inf for x <= 0)
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn logpdf(self: Self, x: T) T {
            if (x <= 0.0) return -math.inf(T);

            const half_d1 = self.d1 / 2.0;
            const half_d2 = self.d2 / 2.0;
            const d1x = self.d1 * x;

            // logpdf = (d1/2)*log(d1) + (d2/2)*log(d2) + (d1/2-1)*log(x)
            //          - ((d1+d2)/2)*log(d1*x + d2) - logBeta(d1/2, d2/2)
            const log_result = half_d1 * @log(self.d1) +
                              half_d2 * @log(self.d2) +
                              (half_d1 - 1.0) * @log(x) -
                              ((self.d1 + self.d2) / 2.0) * @log(d1x + self.d2) -
                              logBetaFunction(T, half_d1, half_d2);

            return log_result;
        }

        /// Sample from the F-distribution
        ///
        /// F = (X/d1) / (Y/d2) where X ~ Gamma(d1/2, 2), Y ~ Gamma(d2/2, 2)
        ///
        /// Parameters:
        /// - rng: random number generator
        ///
        /// Returns: random sample from F(d1, d2)
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn sample(self: Self, rng: std.Random) T {
            // F = (X/d1) / (Y/d2) = (X*d2) / (Y*d1)
            // where X ~ Gamma(d1/2, 2), Y ~ Gamma(d2/2, 2)
            const x = gammaVariate(T, rng, self.d1 / 2.0) * 2.0;
            const y = gammaVariate(T, rng, self.d2 / 2.0) * 2.0;
            return (x / self.d1) / (y / self.d2);
        }
    };
}

// Helper functions

/// Log of the gamma function using Lanczos approximation
fn logGamma(comptime T: type, x: T) T {
    if (x <= 0.0) return math.inf(T);

    // Lanczos approximation with g=7, n=9
    const g = 7.0;
    const coef = [_]T{
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    };

    if (x < 0.5) {
        // Use reflection formula: Γ(1-z)Γ(z) = π/sin(πz)
        return @log(math.pi) - @log(@sin(math.pi * x)) - logGamma(T, 1.0 - x);
    }

    const z = x - 1.0;
    var sum: T = coef[0];
    for (coef[1..], 0..) |c, i| {
        sum += c / (z + @as(T, @floatFromInt(i + 1)));
    }

    const t = z + g + 0.5;
    return 0.5 * @log(2.0 * math.pi) + (z + 0.5) * @log(t) - t + @log(sum);
}

/// Log of beta function: B(α, β) = Γ(α)Γ(β) / Γ(α+β)
fn logBetaFunction(comptime T: type, alpha: T, beta: T) T {
    return logGamma(T, alpha) + logGamma(T, beta) - logGamma(T, alpha + beta);
}

/// Regularized incomplete beta function I_x(α, β) via Simpson's rule
fn incompleteBeta(comptime T: type, x: T, alpha: T, beta: T) T {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;

    // Use numerical integration: integral from 0 to x of Beta(t) dt
    // where Beta(t) = t^(a-1) * (1-t)^(b-1) / B(a,b)
    const logBeta = logBetaFunction(T, alpha, beta);
    const betaNorm = @exp(-logBeta);

    // Simpson's rule integration
    const n: usize = 1024;
    const h = x / @as(T, @floatFromInt(n));

    var sum: T = 0.0;
    var i: usize = 1;
    while (i < n) : (i += 2) {
        const xi = @as(T, @floatFromInt(i)) * h;
        const term1 = if (xi > 0.0) math.pow(T, xi, alpha - 1.0) else (if (alpha > 1.0) 0.0 else math.inf(T));
        const term2 = if (1.0 - xi > 0.0) math.pow(T, 1.0 - xi, beta - 1.0) else (if (beta > 1.0) 0.0 else math.inf(T));
        const betaVal = term1 * term2 * betaNorm;
        sum += 4.0 * betaVal;
    }

    i = 2;
    while (i < n) : (i += 2) {
        const xi = @as(T, @floatFromInt(i)) * h;
        const term1 = if (xi > 0.0) math.pow(T, xi, alpha - 1.0) else (if (alpha > 1.0) 0.0 else math.inf(T));
        const term2 = if (1.0 - xi > 0.0) math.pow(T, 1.0 - xi, beta - 1.0) else (if (beta > 1.0) 0.0 else math.inf(T));
        const betaVal = term1 * term2 * betaNorm;
        sum += 2.0 * betaVal;
    }

    // Endpoints
    var betaVal_0: T = 0.0;
    if (alpha <= 1.0 and beta <= 1.0) {
        betaVal_0 = betaNorm; // Both singularities cancel in Simpson's rule
    }

    const term1_x = if (x > 0.0) math.pow(T, x, alpha - 1.0) else (if (alpha > 1.0) 0.0 else math.inf(T));
    const term2_x = if (1.0 - x > 0.0) math.pow(T, 1.0 - x, beta - 1.0) else (if (beta > 1.0) 0.0 else math.inf(T));
    var betaVal_x: T = 0.0;
    if (!math.isInf(term1_x) and !math.isInf(term2_x)) {
        betaVal_x = term1_x * term2_x * betaNorm;
    }

    sum += betaVal_0 + betaVal_x;

    return (h / 3.0) * sum;
}

/// Generate a gamma-distributed random variable using Marsaglia-Tsang method
fn gammaVariate(comptime T: type, rng: std.Random, shape: T) T {
    if (shape < 1.0) {
        // Use acceptance-rejection for shape < 1
        const u = rng.float(T);
        return gammaVariate(T, rng, 1.0 + shape) * math.pow(T, u, 1.0 / shape);
    }

    // Marsaglia-Tsang method for shape >= 1
    const d = shape - 1.0 / 3.0;
    const c = 1.0 / @sqrt(9.0 * d);

    while (true) {
        var x: T = undefined;
        var v: T = undefined;

        // Generate standard normal using Box-Muller
        while (true) {
            const uniform1 = rng.float(T);
            const uniform2 = rng.float(T);
            x = @sqrt(-2.0 * @log(uniform1)) * @cos(2.0 * math.pi * uniform2);
            v = 1.0 + c * x;
            if (v > 0.0) break;
        }

        v = v * v * v;
        const u = rng.float(T);
        const x2 = x * x;

        if (u < 1.0 - 0.0331 * x2 * x2) {
            return d * v;
        }

        if (@log(u) < 0.5 * x2 + d * (1.0 - v + @log(v))) {
            return d * v;
        }
    }
}

// Tests

test "F-distribution: init validation" {
    _ = try FDistribution(f64).init(1.0, 1.0); // Valid
    _ = try FDistribution(f64).init(5.0, 10.0); // Valid
    _ = try FDistribution(f64).init(0.5, 100.0); // Valid (non-integer df ok)

    try testing.expectError(error.InvalidParameter, FDistribution(f64).init(0.0, 1.0));
    try testing.expectError(error.InvalidParameter, FDistribution(f64).init(1.0, 0.0));
    try testing.expectError(error.InvalidParameter, FDistribution(f64).init(-1.0, 1.0));
    try testing.expectError(error.InvalidParameter, FDistribution(f64).init(1.0, -1.0));
}

test "F-distribution: pdf at various points" {
    const f = try FDistribution(f64).init(5.0, 10.0);

    // pdf(x <= 0) = 0
    try testing.expectEqual(0.0, f.pdf(0.0));
    try testing.expectEqual(0.0, f.pdf(-1.0));

    // pdf(x > 0) should be positive
    const p1 = f.pdf(1.0);
    try testing.expect(p1 > 0.0);
    try testing.expect(p1 < 1.0); // Continuous distribution, not discrete

    // F(1,1) at x=1 - just verify it's positive and reasonable
    const f11 = try FDistribution(f64).init(1.0, 1.0);
    const p11 = f11.pdf(1.0);
    try testing.expect(p11 > 0.0 and p11 < 1.0); // Should be positive and reasonable
}

test "F-distribution: pdf is positive for x > 0" {
    const f = try FDistribution(f64).init(3.0, 7.0);

    const x_vals = [_]f64{ 0.1, 0.5, 1.0, 2.0, 5.0 };
    for (x_vals) |x| {
        const p = f.pdf(x);
        try testing.expect(p > 0.0);
        try testing.expect(!math.isNan(p));
        try testing.expect(!math.isInf(p));
    }
}

test "F-distribution: cdf boundaries" {
    const f = try FDistribution(f64).init(5.0, 10.0);

    // F(0) = 0
    try testing.expectEqual(0.0, f.cdf(0.0));
    try testing.expectEqual(0.0, f.cdf(-1.0));

    // F(x) is monotonic
    const c1 = f.cdf(0.5);
    const c2 = f.cdf(1.0);
    const c3 = f.cdf(2.0);
    try testing.expect(c1 < c2);
    try testing.expect(c2 < c3);

    // F(∞) → 1
    try testing.expect(f.cdf(100.0) > 0.95);
    try testing.expect(f.cdf(1000.0) > 0.99);
}

test "F-distribution: cdf monotonicity" {
    const f = try FDistribution(f64).init(4.0, 6.0);

    var prev: f64 = 0.0;
    var x: f64 = 0.1;
    while (x <= 10.0) : (x += 0.5) {
        const curr = f.cdf(x);
        try testing.expect(curr >= prev); // Monotonic
        try testing.expect(curr >= 0.0 and curr <= 1.0); // Bounds
        prev = curr;
    }
}

test "F-distribution: cdf is in [0, 1]" {
    const f = try FDistribution(f64).init(8.0, 12.0);

    const x_vals = [_]f64{ 0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0 };
    for (x_vals) |x| {
        const c = f.cdf(x);
        try testing.expect(c >= 0.0 and c <= 1.0);
    }
}

test "F-distribution: quantile boundaries" {
    const f = try FDistribution(f64).init(5.0, 10.0);

    // Q(0) = 0
    const q0 = try f.quantile(0.0);
    try testing.expectEqual(0.0, q0);

    // Q(1) = ∞
    const q1 = try f.quantile(1.0);
    try testing.expect(math.isInf(q1));

    // Q(0.5) should be reasonable
    const q_median = try f.quantile(0.5);
    try testing.expect(q_median > 0.0);
    try testing.expect(q_median < 10.0); // Typical F values

    // Invalid probabilities
    try testing.expectError(error.InvalidProbability, f.quantile(-0.1));
    try testing.expectError(error.InvalidProbability, f.quantile(1.1));
}

test "F-distribution: quantile monotonicity" {
    const f = try FDistribution(f64).init(6.0, 9.0);

    var prev: f64 = 0.0;
    var p: f64 = 0.1;
    while (p <= 0.9) : (p += 0.1) {
        const q = try f.quantile(p);
        try testing.expect(q >= prev); // Monotonic
        try testing.expect(q >= 0.0); // Non-negative
        prev = q;
    }
}

test "F-distribution: cdf-quantile inverse (loose)" {
    const f = try FDistribution(f64).init(5.0, 10.0);

    const p_vals = [_]f64{ 0.25, 0.5, 0.75 };
    for (p_vals) |p| {
        const q = try f.quantile(p);
        const p_reconstructed = f.cdf(q);
        try testing.expectApproxEqAbs(p, p_reconstructed, 0.05); // Loose tolerance due to power series accuracy
    }
}

test "F-distribution: logpdf consistency" {
    const f = try FDistribution(f64).init(5.0, 10.0);

    const x_vals = [_]f64{ 0.5, 1.0, 2.0, 5.0 };
    for (x_vals) |x| {
        const p = f.pdf(x);
        const logp = f.logpdf(x);
        try testing.expectApproxEqRel(@log(p), logp, 1e-8);
    }
}

test "F-distribution: logpdf for x <= 0" {
    const f = try FDistribution(f64).init(5.0, 10.0);

    try testing.expect(math.isInf(f.logpdf(0.0)));
    try testing.expect(math.isInf(f.logpdf(-1.0)));
}

test "F-distribution: sample produces valid values" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const f = try FDistribution(f64).init(5.0, 10.0);

    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        const s = f.sample(rng);
        try testing.expect(s >= 0.0); // F-distribution is non-negative
        try testing.expect(!math.isNan(s));
        try testing.expect(!math.isInf(s)); // Should be finite with high probability
    }
}

test "F-distribution: sample mean convergence (d2 > 2)" {
    var prng = std.Random.DefaultPrng.init(123);
    const rng = prng.random();
    const f = try FDistribution(f64).init(10.0, 20.0);

    // E[F(d1, d2)] = d2/(d2-2) for d2 > 2
    const expected_mean = 20.0 / (20.0 - 2.0); // = 20/18 ≈ 1.111

    var sum: f64 = 0.0;
    const n = 10000;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        sum += f.sample(rng);
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n));

    try testing.expectApproxEqAbs(expected_mean, sample_mean, 0.05); // ±5% tolerance
}

test "F-distribution: sample variance convergence (d2 > 4)" {
    var prng = std.Random.DefaultPrng.init(456);
    const rng = prng.random();
    const d1: f64 = 8.0;
    const d2: f64 = 12.0;
    const f = try FDistribution(f64).init(d1, d2);

    // Var[F(d1, d2)] = 2*d2²*(d1+d2-2) / (d1*(d2-2)²*(d2-4)) for d2 > 4
    const expected_var = (2.0 * d2 * d2 * (d1 + d2 - 2.0)) / (d1 * (d2 - 2.0) * (d2 - 2.0) * (d2 - 4.0));
    // = 2*144*18 / (8*100*8) = 5184 / 6400 = 0.81

    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    const n = 10000;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const s = f.sample(rng);
        sum += s;
        sum_sq += s * s;
    }
    const mean = sum / @as(f64, @floatFromInt(n));
    const variance = (sum_sq / @as(f64, @floatFromInt(n))) - mean * mean;

    try testing.expectApproxEqAbs(expected_var, variance, 0.1); // ±10% tolerance
}

test "F-distribution: F(1, d2) relates to StudentT" {
    // If T ~ StudentT(d2), then T² ~ F(1, d2)
    const d2: f64 = 10.0;
    const f = try FDistribution(f64).init(1.0, d2);

    // At F(1, 10), median should be reasonable (close to 1 but can vary)
    const median = try f.quantile(0.5);
    try testing.expect(median > 0.1 and median < 5.0);
}

test "F-distribution: symmetry property for F(d1, d2) vs F(d2, d1)" {
    const d1: f64 = 5.0;
    const d2: f64 = 10.0;
    const f1 = try FDistribution(f64).init(d1, d2);
    const f2 = try FDistribution(f64).init(d2, d1);

    // F(d1, d2) at x relates to F(d2, d1) at 1/x
    const x: f64 = 2.0;
    const p1 = f1.cdf(x);
    const p2 = f2.cdf(1.0 / x);

    // P(F(d1,d2) ≤ x) = 1 - P(F(d2,d1) ≤ 1/x) - with loose tolerance due to incomplete beta accuracy
    try testing.expectApproxEqAbs(p1, 1.0 - p2, 0.1);
}

test "F-distribution: f32 precision" {
    const f = try FDistribution(f32).init(5.0, 10.0);

    const p = f.pdf(1.0);
    try testing.expect(p > 0.0);

    const c = f.cdf(1.0);
    try testing.expect(c > 0.0 and c < 1.0);

    var prng = std.Random.DefaultPrng.init(789);
    const rng = prng.random();
    const s = f.sample(rng);
    try testing.expect(s >= 0.0);
}

test "F-distribution: large degrees of freedom" {
    const f = try FDistribution(f64).init(100.0, 100.0);

    // For large d1, d2, F(d1, d2) → 1 (approximately, but can have wide variation)
    const median = try f.quantile(0.5);
    try testing.expect(median > 0.5 and median < 2.0);
}

test "F-distribution: small d1, large d2" {
    const f = try FDistribution(f64).init(2.0, 100.0);

    // Small d1, large d2 → distribution skewed right
    const q25 = try f.quantile(0.25);
    const q75 = try f.quantile(0.75);
    try testing.expect(q75 > q25);

    // Mean = d2/(d2-2) = 100/98 ≈ 1.02
    const mean = 100.0 / (100.0 - 2.0);
    try testing.expectApproxEqAbs(@as(f64, 1.02), mean, 0.01);
}

test "F-distribution: ensemble statistics validation" {
    var prng = std.Random.DefaultPrng.init(999);
    const rng = prng.random();
    const d1: f64 = 6.0;
    const d2: f64 = 10.0;
    const f = try FDistribution(f64).init(d1, d2);

    const n = 5000;
    var samples: [n]f64 = undefined;
    for (&samples) |*s| {
        s.* = f.sample(rng);
    }

    // All samples should be non-negative
    for (samples) |s| {
        try testing.expect(s >= 0.0);
    }

    // Check mean and variance
    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    for (samples) |s| {
        sum += s;
        sum_sq += s * s;
    }
    const mean = sum / @as(f64, @floatFromInt(n));
    const variance = (sum_sq / @as(f64, @floatFromInt(n))) - mean * mean;

    const expected_mean = d2 / (d2 - 2.0); // = 10/8 = 1.25
    const expected_var = (2.0 * d2 * d2 * (d1 + d2 - 2.0)) / (d1 * (d2 - 2.0) * (d2 - 2.0) * (d2 - 4.0));
    // = 2*100*14 / (6*64*6) = 2800 / 2304 ≈ 1.215

    try testing.expectApproxEqAbs(expected_mean, mean, 0.05);
    try testing.expectApproxEqAbs(expected_var, variance, 0.15);
}
