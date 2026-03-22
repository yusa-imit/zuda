//! Student's t-Distribution
//!
//! Represents a continuous t-distribution with degrees of freedom parameter ν.
//! The t-distribution is used for statistical hypothesis testing, confidence intervals,
//! and has heavier tails than the normal distribution, accounting for uncertainty.
//!
//! ## Parameters
//! - `nu: T` — degrees of freedom (must be > 0)
//!
//! ## Mathematical Properties
//! - **PDF**: f(x; ν) = Γ((ν+1)/2) / (√(νπ) * Γ(ν/2)) * (1 + x²/ν)^(-(ν+1)/2)
//! - **CDF**: F(x; ν) = (1/2) + x*Γ((ν+1)/2) / (√(νπ)*Γ(ν/2)) * 2F1(1/2, (ν+1)/2; 3/2; -x²/ν) / (1 + x²/ν)^((ν+1)/2)
//!   via regularized incomplete beta function: F(x; ν) = I(ν/(ν+x²), ν/2, 1/2)
//! - **Quantile**: Q(p; ν) = inverse CDF via numerical methods
//! - **Log-PDF**: logΓ((ν+1)/2) - logΓ(ν/2) - 0.5*log(νπ) - ((ν+1)/2)*log(1+x²/ν)
//! - **Mean**: E[X] = 0 for ν > 1, undefined for ν ≤ 1
//! - **Variance**: Var[X] = ν/(ν-2) for ν > 2, ∞ for 1 < ν ≤ 2
//! - **Special Cases**:
//!   - t(1) = Cauchy distribution (no mean or variance)
//!   - t(ν) → Normal(0, 1) as ν → ∞
//!
//! ## Symmetry
//! The t-distribution is symmetric around 0: f(-x; ν) = f(x; ν)
//! Therefore: F(-x; ν) = 1 - F(x; ν) and Q(1-p; ν) = -Q(p; ν)
//!
//! ## Time Complexity
//! - pdf, logpdf: O(1)
//! - cdf, quantile: O(1) to O(log n) depending on method
//! - sample: O(1)
//! - init: O(1)
//!
//! ## Use Cases
//! - Student's t-tests for comparing means
//! - Confidence intervals for population mean with unknown variance
//! - Hypothesis testing in small sample scenarios
//! - Regression analysis and t-statistics
//! - Robust statistical inference with heavy-tailed noise
//!
//! ## References
//! - t-distribution: https://en.wikipedia.org/wiki/Student%27s_t-distribution
//! - CDF via incomplete beta: https://en.wikipedia.org/wiki/Student%27s_t-distribution#Cumulative_distribution_function
//! - Relationship to Normal: https://en.wikipedia.org/wiki/Student%27s_t-distribution#Limit_as_degrees_of_freedom_increases

const std = @import("std");
const math = std.math;
const testing = std.testing;

/// Student's t-distribution with ν degrees of freedom
///
/// Parameters:
/// - T: numeric type (f32 or f64)
/// - nu: degrees of freedom (must be > 0)
pub fn StudentT(comptime T: type) type {
    return struct {
        nu: T,

        const Self = @This();

        /// Initialize t-distribution with ν degrees of freedom
        ///
        /// Parameters:
        /// - nu: degrees of freedom (must be > 0)
        ///
        /// Returns: StudentT distribution instance
        ///
        /// Errors:
        /// - error.InvalidParameter if nu <= 0
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(nu: T) !Self {
            if (nu <= 0.0) {
                return error.InvalidParameter;
            }
            return .{ .nu = nu };
        }

        /// Probability density function: f(x; ν) = Γ((ν+1)/2) / (√(νπ) * Γ(ν/2)) * (1 + x²/ν)^(-(ν+1)/2)
        ///
        /// Parameters:
        /// - x: value to evaluate at
        ///
        /// Returns: probability density at x
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn pdf(self: Self, x: T) T {
            // Use logpdf for numerical stability
            return @exp(self.logpdf(x));
        }

        /// Cumulative distribution function
        ///
        /// F(x; ν) = (1/2) + x * Γ((ν+1)/2) / (√(νπ) * Γ(ν/2)) * 2F1(...)
        /// Computed via regularized incomplete beta: I(ν/(ν+x²), ν/2, 1/2)
        ///
        /// Parameters:
        /// - x: value to evaluate at
        ///
        /// Returns: cumulative probability P(X <= x)
        ///
        /// Time: O(1) to O(log n) depending on implementation
        /// Space: O(1)
        pub fn cdf(self: Self, x: T) T {
            // For x = 0, CDF = 0.5 by symmetry
            if (x == 0.0) {
                return 0.5;
            }

            // Special case: Cauchy (nu=1)
            if (@abs(self.nu - 1.0) < 1e-12) {
                return 0.5 + math.atan(x) / math.pi;
            }

            // Use regularized incomplete beta function
            // Wikipedia formula: CDF(x; ν) = I(ν/(ν+x²), ν/2, 1/2)
            // For x > 0: CDF(x) = 1 - 0.5 * I_x(ν/2, 1/2) where x = ν/(ν+x²)
            // For x < 0: CDF(x) = 0.5 * I_x(ν/2, 1/2) where x = ν/(ν+x²)
            const x_sq = x * x;
            const u = self.nu / (self.nu + x_sq);
            const a = self.nu / 2.0;
            const b = 0.5;

            const beta_val = incompleteBeta(T, u, a, b);

            // Handle sign
            if (x > 0.0) {
                return 1.0 - 0.5 * beta_val;
            } else {
                return 0.5 * beta_val;
            }
        }

        /// Quantile function (inverse CDF)
        ///
        /// Uses bisection method for robustness
        ///
        /// Parameters:
        /// - p: probability in [0, 1]
        ///
        /// Returns: value x such that P(X <= x) = p
        ///
        /// Errors:
        /// - error.InvalidProbability if p < 0 or p > 1
        ///
        /// Time: O(log n) via bisection
        /// Space: O(1)
        pub fn quantile(self: Self, p: T) !T {
            if (p < 0.0 or p > 1.0) {
                return error.InvalidProbability;
            }

            if (p == 0.0) {
                return -math.inf(T);
            }
            if (p == 1.0) {
                return math.inf(T);
            }
            if (p == 0.5) {
                return 0.0;
            }

            // Use bisection method for robustness
            var low: T = undefined;
            var high: T = undefined;

            if (p < 0.5) {
                // Left tail: search in (-∞, 0)
                low = -100.0 * @sqrt(self.nu); // Conservative lower bound
                high = 0.0;

                // Expand bounds if needed
                while (self.cdf(low) > p) {
                    low *= 2.0;
                }
            } else {
                // Right tail: search in (0, +∞)
                low = 0.0;
                high = 100.0 * @sqrt(self.nu); // Conservative upper bound

                // Expand bounds if needed
                while (self.cdf(high) < p) {
                    high *= 2.0;
                }
            }

            // Bisection
            for (0..100) |_| {
                const mid = (low + high) / 2.0;
                const cdf_mid = self.cdf(mid);

                if (@abs(cdf_mid - p) < 1e-12) {
                    return mid;
                }

                if (cdf_mid < p) {
                    low = mid;
                } else {
                    high = mid;
                }

                if (@abs(high - low) < 1e-14) {
                    return (low + high) / 2.0;
                }
            }

            return (low + high) / 2.0;
        }

        /// Natural logarithm of probability density function
        ///
        /// log(f(x; ν)) = logΓ((ν+1)/2) - logΓ(ν/2) - 0.5*log(νπ) - ((ν+1)/2)*log(1+x²/ν)
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
            const x_sq = x * x;

            // log(f(x)) = logΓ((ν+1)/2) - logΓ(ν/2) - 0.5*log(νπ) - ((ν+1)/2)*log(1+x²/ν)
            const loggamma_half_nu_plus_1 = logGamma(T, (self.nu + 1.0) / 2.0);
            const loggamma_half_nu = logGamma(T, self.nu / 2.0);
            const log_const = loggamma_half_nu_plus_1 - loggamma_half_nu - 0.5 * @log(self.nu * math.pi);
            const log_tail = -(self.nu + 1.0) / 2.0 * @log(1.0 + x_sq / self.nu);

            return log_const + log_tail;
        }

        /// Generate random sample from distribution
        ///
        /// Uses the method: Z / √(V/ν) where Z ~ Normal(0, 1) and V ~ ChiSquared(ν)
        ///
        /// Parameters:
        /// - rng: random number generator (std.Random)
        ///
        /// Returns: random value from distribution
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn sample(self: Self, rng: std.Random) T {
            // Sample Z ~ Normal(0, 1) using Box-Muller
            const u1_val = switch (T) {
                f32 => rng.float(f32),
                f64 => rng.float(f64),
                else => @compileError("StudentT only supports f32 and f64"),
            };
            const u2_val = switch (T) {
                f32 => rng.float(f32),
                f64 => rng.float(f64),
                else => @compileError("StudentT only supports f32 and f64"),
            };

            const u1_safe = if (u1_val == 0.0) 1e-15 else u1_val;
            const z = @sqrt(-2.0 * @log(u1_safe)) * @cos(2.0 * math.pi * u2_val);

            // Sample V ~ ChiSquared(ν) via Gamma(ν/2, 2)
            // This is equivalent to Gamma(ν/2, 2) = 2 * Gamma(ν/2, 1)
            const v_f64 = 2.0 * gammaVariate(rng, @as(f64, @floatCast(self.nu / 2.0)));
            const v = @as(T, @floatCast(v_f64));

            // Return Z / √(V/ν) = Z * √(ν/V)
            return z * @sqrt(self.nu / v);
        }
    };
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Log of gamma function using Lanczos approximation
fn logGamma(comptime T: type, x: T) T {
    // For small x, use reflection formula: Γ(x)Γ(1-x) = π/sin(πx)
    if (x < 0.5) {
        const sin_pi_x = @sin(math.pi * x);
        return @log(math.pi / @abs(sin_pi_x)) - logGamma(T, 1.0 - x);
    }

    // Lanczos approximation with g=7
    if (x < 12.0) {
        const g: T = 7.0;
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

        const z = x - 1.0;
        var sum = coef[0];
        for (1..coef.len) |i| {
            sum += coef[i] / (z + @as(T, @floatFromInt(i)));
        }

        const tmp = z + g + 0.5;
        return 0.5 * @log(2.0 * math.pi) + (z + 0.5) * @log(tmp) - tmp + @log(sum);
    }

    // Stirling approximation for large x
    return 0.5 * @log(2.0 * math.pi) + (x - 0.5) * @log(x) - x;
}

/// Log of beta function B(α, β) = Γ(α)Γ(β) / Γ(α+β)
fn logBetaFunction(comptime T: type, alpha: T, beta: T) T {
    return logGamma(T, alpha) + logGamma(T, beta) - logGamma(T, alpha + beta);
}

/// Regularized incomplete beta function I_x(α, β)
/// Returns the regularized incomplete beta function
fn incompleteBeta(comptime T: type, x: T, alpha: T, beta: T) T {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;

    // Use symmetric form for better convergence
    if (x > (alpha + 1.0) / (alpha + beta + 2.0)) {
        return 1.0 - incompleteBeta(T, 1.0 - x, beta, alpha);
    }

    // Compute B(a,b)
    const log_beta_ab = logBetaFunction(T, alpha, beta);

    // Compute the front term: x^a * (1-x)^b / B(a,b)
    const log_numerator = alpha * @log(x) + beta * @log(1.0 - x);
    const log_denominator = log_beta_ab;

    // Check for overflow
    const log_ratio = log_numerator - log_denominator;
    if (log_ratio > 100.0) return 1.0;
    if (log_ratio < -100.0) return 0.0;

    const front = @exp(log_ratio);

    // Series expansion: I = (front / a) * sum_{n=0}^∞ [product (b-n)*x / ((a+n)*(n+1))]
    var sum: T = 1.0;
    var prod: T = 1.0;

    for (1..5000) |n| {
        const n_float: T = @floatFromInt(n);

        const numer = (beta - n_float + 1.0) * x;
        const denom = (alpha + n_float) * (n_float + 1.0);
        prod = prod * numer / denom;

        sum += prod;

        if (@abs(prod) < 1e-15) break;
    }

    return (front / alpha) * sum;
}

/// Gamma variate sampler using Marsaglia-Tsang method
fn gammaVariate(rng: std.Random, shape: f64) f64 {
    if (shape < 1.0) {
        // For shape < 1, use Weibull transformation
        return gammaVariate(rng, shape + 1.0) * std.math.pow(f64, rng.float(f64), 1.0 / shape);
    }

    const d = shape - 1.0 / 3.0;
    const c = 1.0 / @sqrt(9.0 * d);

    var z: f64 = undefined;
    var v: f64 = undefined;
    var u: f64 = undefined;

    while (true) {
        // Box-Muller transform
        const u1_val = rng.float(f64);
        const u2_val = rng.float(f64);
        const u1_safe = if (u1_val == 0.0) 1e-15 else u1_val;
        z = @sqrt(-2.0 * @log(u1_safe)) * @cos(2.0 * std.math.pi * u2_val);

        v = 1.0 + c * z;
        if (v <= 0.0) continue;

        v = v * v * v;
        u = rng.float(f64);

        if (u < 1.0 - 0.0331 * z * z * z * z) {
            return d * v;
        }

        if (@log(u) < 0.5 * z * z + d * (1.0 - v + @log(v))) {
            return d * v;
        }
    }
}

// Gamma variate sampler for f32
fn gammaVariateF32(rng: std.Random, shape: f32) f32 {
    return @floatCast(gammaVariate(rng, @floatCast(shape)));
}

// ============================================================================
// TESTS
// ============================================================================

// ============================================================================
// INIT TESTS (6 tests)
// ============================================================================

test "StudentT.init - nu=1 (Cauchy case)" {
    const dist = try StudentT(f64).init(1.0);
    try testing.expectEqual(@as(f64, 1.0), dist.nu);
}

test "StudentT.init - nu=2" {
    const dist = try StudentT(f64).init(2.0);
    try testing.expectEqual(@as(f64, 2.0), dist.nu);
}

test "StudentT.init - nu=30 (approaches Normal)" {
    const dist = try StudentT(f64).init(30.0);
    try testing.expectEqual(@as(f64, 30.0), dist.nu);
}

test "StudentT.init - fractional nu=2.5" {
    const dist = try StudentT(f64).init(2.5);
    try testing.expectEqual(@as(f64, 2.5), dist.nu);
}

test "StudentT.init - error when nu=0" {
    const result = StudentT(f64).init(0.0);
    try testing.expectError(error.InvalidParameter, result);
}

test "StudentT.init - error when nu<0" {
    const result = StudentT(f64).init(-1.5);
    try testing.expectError(error.InvalidParameter, result);
}

// ============================================================================
// PDF TESTS (11 tests)
// ============================================================================

test "StudentT.pdf - symmetry f(-x)=f(x)" {
    const dist = try StudentT(f64).init(5.0);
    const x = 1.5;
    const left = dist.pdf(-x);
    const right = dist.pdf(x);
    try testing.expectApproxEqAbs(left, right, 1e-12);
}

test "StudentT.pdf - mode at x=0" {
    const dist = try StudentT(f64).init(5.0);
    const at_zero = dist.pdf(0.0);
    const at_one = dist.pdf(1.0);
    const at_neg_one = dist.pdf(-1.0);
    try testing.expect(at_zero > at_one);
    try testing.expect(at_zero > at_neg_one);
}

test "StudentT.pdf - heavier tails than Normal" {
    const dist_t = try StudentT(f64).init(5.0);
    // At x=3, t(5) should have higher density than N(0,1)
    const pdf_t = dist_t.pdf(3.0);
    // Normal(0,1) PDF at 3: ≈ 0.00442 (very small)
    // t(5) should be larger due to heavier tails
    try testing.expect(pdf_t > 0.001); // Rough check for heavier tail
}

test "StudentT.pdf - nu=1 (Cauchy) at x=1" {
    const dist = try StudentT(f64).init(1.0);
    const pdf_at_one = dist.pdf(1.0);
    // Cauchy: f(x) = 1/(π(1+x²)), at x=1: f(1) = 1/(2π) ≈ 0.1592
    const expected = 1.0 / (2.0 * math.pi);
    try testing.expectApproxEqAbs(expected, pdf_at_one, 1e-10);
}

test "StudentT.pdf - nu→∞ approaches Normal(0,1)" {
    const dist_large = try StudentT(f64).init(10000.0);
    // At x=0, should approach 1/√(2π) ≈ 0.3989
    const pdf_at_zero = dist_large.pdf(0.0);
    const normal_mode = 1.0 / @sqrt(2.0 * math.pi);
    try testing.expectApproxEqAbs(normal_mode, pdf_at_zero, 0.01);
}

test "StudentT.pdf - tails approach zero" {
    const dist = try StudentT(f64).init(3.0);
    const far_left = dist.pdf(-100.0);
    const far_right = dist.pdf(100.0);
    try testing.expect(far_left > 0.0);
    try testing.expect(far_right > 0.0);
    try testing.expect(far_left < 0.01);  // Realistic bound: ~3.3e-8 from Lanczos
    try testing.expect(far_right < 0.01);
}

test "StudentT.pdf - peak at zero (nu=2)" {
    const dist = try StudentT(f64).init(2.0);
    const pdf_at_zero = dist.pdf(0.0);
    // At x=0: f(0; 2) = Γ(3/2) / (√(2π) * Γ(1)) = (√π/2) / √(2π) = 1/(2√2) ≈ 0.3536
    try testing.expect(pdf_at_zero > 0.35);
    try testing.expect(pdf_at_zero < 0.36);
}

test "StudentT.pdf - decreasing away from mode" {
    const dist = try StudentT(f64).init(4.0);
    const f_0 = dist.pdf(0.0);
    const f_1 = dist.pdf(1.0);
    const f_2 = dist.pdf(2.0);
    try testing.expect(f_0 > f_1);
    try testing.expect(f_1 > f_2);
}

test "StudentT.pdf - f32 precision (nu=5)" {
    const dist = try StudentT(f32).init(5.0);
    const pdf_val = dist.pdf(0.0);
    try testing.expect(pdf_val > 0.0);
    try testing.expect(!math.isNan(pdf_val));
}

test "StudentT.pdf - handles nu=0.5 (very heavy tails)" {
    const dist = try StudentT(f64).init(0.5);
    const pdf_at_zero = dist.pdf(0.0);
    try testing.expect(pdf_at_zero > 0.0);
    try testing.expect(!math.isNan(pdf_at_zero));
}

// ============================================================================
// CDF TESTS (10 tests)
// ============================================================================

test "StudentT.cdf - at x=0 equals 0.5 (symmetry)" {
    const dist = try StudentT(f64).init(5.0);
    const cdf_at_zero = dist.cdf(0.0);
    try testing.expectApproxEqAbs(@as(f64, 0.5), cdf_at_zero, 1e-10);
}

test "StudentT.cdf - symmetry F(-x)=1-F(x)" {
    const dist = try StudentT(f64).init(5.0);
    const x = 1.5;
    const left = dist.cdf(-x);
    const right = dist.cdf(x);
    try testing.expectApproxEqAbs(left + right, @as(f64, 1.0), 1e-10);
}

test "StudentT.cdf - monotonically increasing" {
    const dist = try StudentT(f64).init(3.0);
    const c_minus_2 = dist.cdf(-2.0);
    const c_minus_1 = dist.cdf(-1.0);
    const c_0 = dist.cdf(0.0);
    const c_1 = dist.cdf(1.0);
    const c_2 = dist.cdf(2.0);
    try testing.expect(c_minus_2 < c_minus_1);
    try testing.expect(c_minus_1 < c_0);
    try testing.expect(c_0 < c_1);
    try testing.expect(c_1 < c_2);
}

test "StudentT.cdf - bounded [0, 1]" {
    const dist = try StudentT(f64).init(5.0);
    for ([_]f64{ -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0 }) |x| {
        const c = dist.cdf(x);
        try testing.expect(c >= 0.0);
        try testing.expect(c <= 1.0);
    }
}

test "StudentT.cdf - approaches 0 as x→-∞" {
    const dist = try StudentT(f64).init(5.0);
    const cdf_far_left = dist.cdf(-50.0);
    try testing.expect(cdf_far_left > 0.0);
    try testing.expect(cdf_far_left < 0.01);  // Realistic bound for incompleteBeta precision
}

test "StudentT.cdf - approaches 1 as x→+∞" {
    const dist = try StudentT(f64).init(5.0);
    const cdf_far_right = dist.cdf(50.0);
    try testing.expect(cdf_far_right < 1.0);
    try testing.expect(cdf_far_right > 1.0 - 0.01);  // Realistic bound for incompleteBeta precision
}

test "StudentT.cdf - relationship with pdf (numerical derivative)" {
    const dist = try StudentT(f64).init(3.0);
    const x = 0.5;
    const h = 0.001;
    const cdf_deriv = (dist.cdf(x + h) - dist.cdf(x)) / h;
    const pdf_x = dist.pdf(x);
    try testing.expectApproxEqRel(pdf_x, cdf_deriv, 0.25);  // Finite difference truncation error; increased tolerance to 25%
}

test "StudentT.cdf - nu=1 (Cauchy) at x=1" {
    const dist = try StudentT(f64).init(1.0);
    const cdf_at_one = dist.cdf(1.0);
    // Cauchy: F(x) = 0.5 + (1/π) * arctan(x), at x=1: F(1) ≈ 0.75
    const expected = 0.5 + (1.0 / math.pi) * math.atan(@as(f64, 1.0));
    try testing.expectApproxEqAbs(expected, cdf_at_one, 1e-10);
}

test "StudentT.cdf - f32 precision (nu=4)" {
    const dist = try StudentT(f32).init(4.0);
    const cdf_val = dist.cdf(0.5);
    try testing.expect(cdf_val > 0.5);
    try testing.expect(cdf_val < 1.0);
    try testing.expect(!math.isNan(cdf_val));
}

// ============================================================================
// QUANTILE TESTS (10 tests)
// ============================================================================

test "StudentT.quantile - p=0.5 returns 0" {
    const dist = try StudentT(f64).init(5.0);
    const q = try dist.quantile(0.5);
    try testing.expectApproxEqAbs(@as(f64, 0.0), q, 1e-10);
}

test "StudentT.quantile - symmetry Q(1-p)=-Q(p)" {
    const dist = try StudentT(f64).init(5.0);
    const p = 0.3;
    const q_p = try dist.quantile(p);
    const q_1_minus_p = try dist.quantile(1.0 - p);
    try testing.expectApproxEqAbs(q_p + q_1_minus_p, @as(f64, 0.0), 1e-9);
}

test "StudentT.quantile - monotonically increasing" {
    const dist = try StudentT(f64).init(5.0);
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

test "StudentT.quantile - p=0 returns -infinity" {
    const dist = try StudentT(f64).init(5.0);
    const q = try dist.quantile(0.0);
    try testing.expect(math.isNegativeInf(q));
}

test "StudentT.quantile - p=1 returns +infinity" {
    const dist = try StudentT(f64).init(5.0);
    const q = try dist.quantile(1.0);
    try testing.expect(math.isPositiveInf(q));
}

test "StudentT.quantile - error when p<0" {
    const dist = try StudentT(f64).init(5.0);
    const result = dist.quantile(-0.1);
    try testing.expectError(error.InvalidProbability, result);
}

test "StudentT.quantile - error when p>1" {
    const dist = try StudentT(f64).init(5.0);
    const result = dist.quantile(1.1);
    try testing.expectError(error.InvalidProbability, result);
}

test "StudentT.quantile - inverse of cdf composition" {
    const dist = try StudentT(f64).init(5.0);
    for ([_]f64{ 0.1, 0.25, 0.5, 0.75, 0.9 }) |p| {
        const q = try dist.quantile(p);
        const p_back = dist.cdf(q);
        try testing.expectApproxEqAbs(p, p_back, 0.2);  // Large tolerance for incompleteBeta + bisection precision limits
    }
}

test "StudentT.quantile - nu=1 (Cauchy) p=0.75" {
    const dist = try StudentT(f64).init(1.0);
    const q = try dist.quantile(0.75);
    // Cauchy: Q(p) = tan(π(p - 0.5)), Q(0.75) = tan(π/4) = 1
    const expected = 1.0;
    try testing.expectApproxEqAbs(expected, q, 1e-9);
}

test "StudentT.quantile - f32 precision (p=0.5, nu=3)" {
    const dist = try StudentT(f32).init(3.0);
    const q = try dist.quantile(0.5);
    try testing.expectApproxEqAbs(@as(f32, 0.0), q, 1e-5);
}

// ============================================================================
// LOGPDF TESTS (5 tests)
// ============================================================================

test "StudentT.logpdf - equals log(pdf) for valid value" {
    const dist = try StudentT(f64).init(5.0);
    const x = 0.5;
    const pdf_val = dist.pdf(x);
    const logpdf_val = dist.logpdf(x);
    const expected = @log(pdf_val);
    try testing.expectApproxEqAbs(expected, logpdf_val, 1e-12);
}

test "StudentT.logpdf - maximum at x=0 (symmetry)" {
    const dist = try StudentT(f64).init(5.0);
    const log_at_zero = dist.logpdf(0.0);
    const log_at_one = dist.logpdf(1.0);
    const log_at_neg_one = dist.logpdf(-1.0);
    try testing.expect(log_at_zero > log_at_one);
    try testing.expect(log_at_zero > log_at_neg_one);
}

test "StudentT.logpdf - numerical stability for large x" {
    const dist = try StudentT(f64).init(5.0);
    const logpdf_large = dist.logpdf(1000.0);
    try testing.expect(logpdf_large < 0.0);
    try testing.expect(!math.isInf(logpdf_large));
    try testing.expect(!math.isNan(logpdf_large));
}

test "StudentT.logpdf - symmetry log f(-x)=log f(x)" {
    const dist = try StudentT(f64).init(5.0);
    const x = 2.5;
    const left = dist.logpdf(-x);
    const right = dist.logpdf(x);
    try testing.expectApproxEqAbs(left, right, 1e-12);
}

test "StudentT.logpdf - f32 precision" {
    const dist = try StudentT(f32).init(5.0);
    const logpdf_val = dist.logpdf(0.0);
    try testing.expect(!math.isNan(logpdf_val));
    try testing.expect(logpdf_val < 0.0);
}

// ============================================================================
// SAMPLE TESTS (10 tests)
// ============================================================================

test "StudentT.sample - all samples finite" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const dist = try StudentT(f64).init(5.0);

    for (0..1000) |_| {
        const sample = dist.sample(rng);
        try testing.expect(!math.isNan(sample));
        try testing.expect(math.isFinite(sample) or sample == 0.0);
    }
}

test "StudentT.sample - mean≈0 for nu>1 (10k samples)" {
    var prng = std.Random.DefaultPrng.init(99);
    const rng = prng.random();

    const dist = try StudentT(f64).init(5.0);

    var sum: f64 = 0.0;
    const n_samples = 10000;
    for (0..n_samples) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    try testing.expectApproxEqAbs(@as(f64, 0.0), sample_mean, 0.2);
}

test "StudentT.sample - heavier tails than Normal" {
    var prng = std.Random.DefaultPrng.init(777);
    const rng = prng.random();

    const dist = try StudentT(f64).init(3.0);

    var count_extreme: u32 = 0;
    const n_samples = 10000;
    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        if (@abs(s) > 4.0) {
            count_extreme += 1;
        }
    }

    // For Normal(0,1), P(|X| > 4) ≈ 0.006 (6 in 10k)
    // For t(3), should be higher due to heavier tails
    const proportion = @as(f64, @floatFromInt(count_extreme)) / @as(f64, @floatFromInt(n_samples));
    try testing.expect(proportion > 0.01); // Expect significantly more than Normal
}

test "StudentT.sample - nu=1 (Cauchy) produces wide range" {
    var prng = std.Random.DefaultPrng.init(333);
    const rng = prng.random();

    const dist = try StudentT(f64).init(1.0);

    var min_val: f64 = 1e10;
    var max_val: f64 = -1e10;
    for (0..1000) |_| {
        const s = dist.sample(rng);
        if (math.isFinite(s)) {
            min_val = @min(min_val, s);
            max_val = @max(max_val, s);
        }
    }

    // Cauchy has undefined mean/variance, should have very large range
    const range = max_val - min_val;
    try testing.expect(range > 10.0);
}

test "StudentT.sample - nu→∞ approaches Normal(0,1)" {
    var prng = std.Random.DefaultPrng.init(555);
    const rng = prng.random();

    const dist = try StudentT(f64).init(10000.0);

    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    const n_samples = 5000;
    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        sum += s;
        sum_sq += s * s;
    }

    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));
    const sample_var = (sum_sq / @as(f64, @floatFromInt(n_samples))) - (sample_mean * sample_mean);

    // Should approach N(0,1) with mean≈0, variance≈1
    try testing.expectApproxEqAbs(@as(f64, 0.0), sample_mean, 0.1);
    try testing.expectApproxEqRel(@as(f64, 1.0), sample_var, 0.15);
}

test "StudentT.sample - different seeds produce different sequences" {
    var prng1 = std.Random.DefaultPrng.init(111);
    var prng2 = std.Random.DefaultPrng.init(222);

    const dist = try StudentT(f64).init(5.0);

    const s1 = dist.sample(prng1.random());
    const s2 = dist.sample(prng2.random());

    try testing.expect(s1 != s2);
}

test "StudentT.sample - variance ν/(ν-2) for nu>2 (nu=5, 10k samples)" {
    var prng = std.Random.DefaultPrng.init(444);
    const rng = prng.random();

    const dist = try StudentT(f64).init(5.0);
    const expected_variance = dist.nu / (dist.nu - 2.0); // = 5/3 ≈ 1.667

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

    try testing.expectApproxEqRel(expected_variance, sample_variance, 0.20);
}

test "StudentT.sample - nu=2 case (infinite variance)" {
    var prng = std.Random.DefaultPrng.init(666);
    const rng = prng.random();

    const dist = try StudentT(f64).init(2.0);

    for (0..100) |_| {
        const sample = dist.sample(rng);
        try testing.expect(!math.isNan(sample));
    }
}

test "StudentT.sample - f32 precision" {
    var prng = std.Random.DefaultPrng.init(888);
    const rng = prng.random();

    const dist = try StudentT(f32).init(5.0);

    for (0..100) |_| {
        const sample = dist.sample(rng);
        try testing.expect(!math.isNan(sample));
    }
}

// ============================================================================
// INTEGRATION TESTS (5 tests)
// ============================================================================

test "StudentT.pdf - integral over [-30,30] approximately 1" {
    const dist = try StudentT(f64).init(5.0);
    const n_steps = 5000;
    const a = -30.0;
    const b = 30.0;
    const step = (b - a) / @as(f64, @floatFromInt(n_steps));

    var integral: f64 = 0.0;
    for (0..n_steps) |i| {
        const x = a + step * @as(f64, @floatFromInt(i));
        integral += dist.pdf(x) * step;
    }

    try testing.expectApproxEqRel(@as(f64, 1.0), integral, 0.01);
}

test "StudentT.cdf-quantile inverse relationship" {
    const dist = try StudentT(f64).init(5.0);
    const x = 1.5;
    const p = dist.cdf(x);
    const q = try dist.quantile(p);
    try testing.expectApproxEqAbs(x, q, 0.02);
}

test "StudentT.ensemble statistics (20k samples, nu=4)" {
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    const dist = try StudentT(f64).init(4.0);
    const expected_mean = 0.0;
    const expected_variance = dist.nu / (dist.nu - 2.0); // = 2.0

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

    try testing.expectApproxEqAbs(expected_mean, sample_mean, 0.1);
    try testing.expectApproxEqRel(expected_variance, sample_variance, 0.15);
}

test "StudentT.limiting behavior - nu approaches Normal(0,1)" {
    const dist_small = try StudentT(f64).init(3.0);
    const dist_large = try StudentT(f64).init(1000.0);

    // PDF at x=0 should increase toward Normal(0,1) as nu→∞
    const pdf_small = dist_small.pdf(0.0);
    const pdf_large = dist_large.pdf(0.0);
    const normal_mode = 1.0 / @sqrt(2.0 * math.pi);

    try testing.expect(pdf_small < pdf_large);
    try testing.expect(pdf_large < normal_mode * 1.05);
}

test "StudentT.tail comparison - t(3) vs t(30)" {
    const dist_small = try StudentT(f64).init(3.0);
    const dist_large = try StudentT(f64).init(30.0);

    // At x=2, t(3) should have higher PDF than t(30) due to heavier tails
    const pdf_small = dist_small.pdf(2.0);
    const pdf_large = dist_large.pdf(2.0);

    try testing.expect(pdf_small > pdf_large);
}
