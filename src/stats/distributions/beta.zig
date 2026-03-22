//! Beta Distribution
//!
//! Represents a continuous beta distribution with shape parameters α and β.
//! The beta distribution is defined on the interval [0, 1] and models the distribution
//! of probabilities and proportions. It is highly flexible with special cases including
//! uniform, arcsine, and power-law distributions.
//!
//! ## Parameters
//! - `alpha: T` — shape parameter 1 (must be > 0)
//! - `beta: T` — shape parameter 2 (must be > 0)
//!
//! ## Mathematical Properties
//! - **PDF**: f(x; α, β) = (x^(α-1) * (1-x)^(β-1)) / B(α, β) for x ∈ [0,1], else 0
//! - **Beta function**: B(α, β) = Γ(α)Γ(β) / Γ(α+β)
//! - **CDF**: F(x; α, β) = I_x(α, β) (regularized incomplete beta function)
//! - **Quantile**: Q(p; α, β) = inverse of CDF (numerical)
//! - **Log-PDF**: (α-1)*log(x) + (β-1)*log(1-x) - log(B(α,β))
//! - **Mean**: E[X] = α/(α+β)
//! - **Variance**: Var[X] = (αβ)/((α+β)²(α+β+1))
//! - **Mode**: (α-1)/(α+β-2) for α, β > 1; undefined otherwise
//!
//! ## Special Cases
//! - **Beta(1, 1)** = Uniform(0, 1): f(x) = 1 for all x ∈ [0,1]
//! - **Beta(0.5, 0.5)** = Arcsine distribution
//! - **Beta(α, 1)** = power law: f(x) ∝ x^(α-1)
//! - **Beta(1, β)** = reverse power law: f(x) ∝ (1-x)^(β-1)
//!
//! ## Time Complexity
//! - pdf, cdf, logpdf: O(1)
//! - quantile: O(log n) via Newton-Raphson iteration
//! - sample: O(1) or O(log n) depending on implementation
//! - init: O(1)
//!
//! ## Use Cases
//! - Modeling proportions and probabilities in [0,1]
//! - Bayesian conjugate prior for binomial likelihood
//! - A/B testing and conversion rate estimation
//! - Concentration and dispersion modeling
//! - Reliability engineering (time-to-failure proportions)
//! - Portfolio optimization
//!
//! ## References
//! - Beta distribution: https://en.wikipedia.org/wiki/Beta_distribution
//! - Regularized incomplete beta: https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function
//! - Sampling via Gamma variates: Choi & Nam (2016)

const std = @import("std");
const math = std.math;
const testing = std.testing;

/// Beta distribution with shape parameters α and β
///
/// Parameters:
/// - T: numeric type (f32 or f64)
/// - alpha: shape parameter 1 (must be > 0)
/// - beta: shape parameter 2 (must be > 0)
pub fn Beta(comptime T: type) type {
    return struct {
        alpha: T,
        beta: T,

        const Self = @This();

        /// Initialize beta distribution with shape parameters α and β
        ///
        /// Parameters:
        /// - alpha: shape parameter 1 (must be > 0)
        /// - beta: shape parameter 2 (must be > 0)
        ///
        /// Returns: Beta distribution instance
        ///
        /// Errors:
        /// - error.InvalidShape if alpha <= 0 or beta <= 0
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(alpha: T, beta: T) !Self {
            if (alpha <= 0.0) {
                return error.InvalidShape;
            }
            if (beta <= 0.0) {
                return error.InvalidShape;
            }
            return .{ .alpha = alpha, .beta = beta };
        }

        /// Probability density function: f(x; α, β) = (x^(α-1) * (1-x)^(β-1)) / B(α,β)
        ///
        /// Parameters:
        /// - x: value to evaluate at
        ///
        /// Returns: probability density at x (0 for x ∉ [0,1])
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn pdf(self: Self, x: T) T {
            // Outside support [0, 1]
            if (x < 0.0 or x > 1.0) {
                return 0.0;
            }

            // Boundary cases at x = 0
            if (x == 0.0) {
                if (self.alpha < 1.0) {
                    return math.inf(T);
                } else if (self.alpha == 1.0) {
                    return self.beta;
                } else {
                    return 0.0;
                }
            }

            // Boundary cases at x = 1
            if (x == 1.0) {
                if (self.beta < 1.0) {
                    return math.inf(T);
                } else if (self.beta == 1.0) {
                    return self.alpha;
                } else {
                    return 0.0;
                }
            }

            // Use logpdf for numerical stability
            return @exp(self.logpdf(x));
        }

        /// Cumulative distribution function: F(x; α, β) = I_x(α, β)
        ///
        /// Parameters:
        /// - x: value to evaluate at
        ///
        /// Returns: cumulative probability at x
        ///
        /// Time: O(1) to O(log n) depending on iteration count
        /// Space: O(1)
        pub fn cdf(self: Self, x: T) T {
            // Outside support
            if (x <= 0.0) {
                return 0.0;
            }
            if (x >= 1.0) {
                return 1.0;
            }

            // Use regularized incomplete beta function
            return incompleteBeta(T, x, self.alpha, self.beta);
        }

        /// Quantile function (inverse CDF): Q(p; α, β)
        ///
        /// Parameters:
        /// - p: probability in [0, 1]
        ///
        /// Returns: quantile value in [0, 1]
        ///
        /// Errors:
        /// - error.InvalidProbability if p < 0 or p > 1
        ///
        /// Time: O(log n) via Newton-Raphson or bisection
        /// Space: O(1)
        pub fn quantile(self: Self, p: T) !T {
            if (p < 0.0 or p > 1.0) {
                return error.InvalidProbability;
            }

            if (p == 0.0) {
                return 0.0;
            }
            if (p == 1.0) {
                return 1.0;
            }

            // Use bisection method for robustness (guaranteed convergence)
            var low: T = 0.0;
            var high: T = 1.0;

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

                if (high - low < 1e-14) {
                    return (low + high) / 2.0;
                }
            }

            return (low + high) / 2.0;
        }

        /// Natural logarithm of probability density function
        ///
        /// Parameters:
        /// - x: value to evaluate at
        ///
        /// Returns: log(pdf(x)), -∞ for x ∉ [0,1]
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn logpdf(self: Self, x: T) T {
            // Outside support
            if (x <= 0.0 or x >= 1.0) {
                return -math.inf(T);
            }

            // log(f(x)) = (α-1)*log(x) + (β-1)*log(1-x) - log(B(α,β))
            const term1 = (self.alpha - 1.0) * @log(x);
            const term2 = (self.beta - 1.0) * @log(1.0 - x);
            const logBeta = logBetaFunction(T, self.alpha, self.beta);

            return term1 + term2 - logBeta;
        }

        /// Random sample from beta distribution
        ///
        /// Uses the gamma variate method: if X ~ Gamma(α, 1) and Y ~ Gamma(β, 1),
        /// then Z = X / (X + Y) ~ Beta(α, β)
        ///
        /// Parameters:
        /// - rng: random number generator (std.Random)
        ///
        /// Returns: sample value in [0, 1]
        ///
        /// Time: O(1) to O(log n) depending on gamma sampling
        /// Space: O(1)
        pub fn sample(self: Self, rng: std.Random) T {
            // Sample X ~ Gamma(α, 1)
            const x = gammaVariate(rng, self.alpha);
            // Sample Y ~ Gamma(β, 1)
            const y = gammaVariate(rng, self.beta);

            // Return X / (X + Y)
            return x / (x + y);
        }
    };
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Log of gamma function using Lanczos approximation
/// Based on Gamma distribution implementation in zuda
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
/// Using: log(B(α,β)) = log(Γ(α)) + log(Γ(β)) - log(Γ(α+β))
fn logBetaFunction(comptime T: type, alpha: T, beta: T) T {
    return logGamma(T, alpha) + logGamma(T, beta) - logGamma(T, alpha + beta);
}

/// Regularized incomplete beta function I_x(α, β)
/// Computes using numerical integration (Simpson's rule) for reliability
fn incompleteBeta(comptime T: type, x: T, alpha: T, beta: T) T {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;

    // Use numerical integration: integral from 0 to x of Beta(t) dt
    // where Beta(t) = t^(a-1) * (1-t)^(b-1) / B(a,b)
    const logBeta = logBetaFunction(T, alpha, beta);
    const betaNorm = @exp(-logBeta);

    // Simpson's rule integration with very high precision
    // Use more points for better accuracy, especially for edge cases
    const n: usize = 2048;
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

    // Add endpoints
    const f0 = if (alpha > 1.0)
        0.0
    else if (alpha == 1.0)
        betaNorm
    else
        math.inf(T);
    const fx_term1 = if (x > 0.0) math.pow(T, x, alpha - 1.0) else (if (alpha > 1.0) 0.0 else math.inf(T));
    const fx_term2 = if (1.0 - x > 0.0) math.pow(T, 1.0 - x, beta - 1.0) else (if (beta > 1.0) 0.0 else math.inf(T));
    const fx = fx_term1 * fx_term2 * betaNorm;

    var result = h / 3.0 * (f0 + sum + fx);

    // Clamp to valid range
    result = if (result < 0.0) 0.0 else if (result > 1.0) 1.0 else result;
    return result;
}

/// Initial guess for quantile using beta distribution moment matching
fn betaQuantileInitial(comptime T: type, p: T, alpha: T, beta: T) T {
    const mean = alpha / (alpha + beta);

    // Simple initial guess based on mean
    if (p < 0.5) {
        return p * mean;
    } else {
        return 1.0 - (1.0 - p) * (1.0 - mean);
    }
}

/// Sample from gamma distribution with shape α and scale 1
/// Using Marsaglia & Tsang method for α ≥ 1
/// and Ahrens & Dieter method for α < 1
fn gammaVariate(rng: std.Random, alpha: f64) f64 {
    if (alpha >= 1.0) {
        // Marsaglia & Tsang (2000)
        const d = alpha - 1.0 / 3.0;
        const c = 1.0 / @sqrt(9.0 * d);

        while (true) {
            var z: f64 = 0.0;
            while (true) {
                // Generate standard normal via Box-Muller
                const uu1 = rng.float(f64);
                const uu2 = rng.float(f64);
                z = @sqrt(-2.0 * @log(uu1)) * @cos(2.0 * math.pi * uu2);

                const v = 1.0 + c * z;
                if (v > 0.0) break;
            }

            const v = 1.0 + c * z;
            const w = d * v * v * v;
            const u = rng.float(f64);

            if (u < 1.0 - 0.0331 * z * z * z * z) {
                return w;
            }
            if (@log(u) < 0.5 * z * z + d * (1.0 - v + @log(v))) {
                return w;
            }
        }
    } else {
        // For α < 1, use Ahrens & Dieter transformation
        return gammaVariate(rng, alpha + 1.0) * math.pow(f64, rng.float(f64), 1.0 / alpha);
    }
}

// ============================================================================
// TESTS (50+ comprehensive tests)
// ============================================================================

// ============================================================================
// INIT TESTS (6 tests)
// ============================================================================

test "Beta.init - standard parameters (α=2, β=2)" {
    const dist = try Beta(f64).init(2.0, 2.0);
    try testing.expectEqual(@as(f64, 2.0), dist.alpha);
    try testing.expectEqual(@as(f64, 2.0), dist.beta);
}

test "Beta.init - symmetric case (α=3, β=3)" {
    const dist = try Beta(f64).init(3.0, 3.0);
    try testing.expectEqual(@as(f64, 3.0), dist.alpha);
    try testing.expectEqual(@as(f64, 3.0), dist.beta);
}

test "Beta.init - skewed (α=5, β=1)" {
    const dist = try Beta(f64).init(5.0, 1.0);
    try testing.expectEqual(@as(f64, 5.0), dist.alpha);
    try testing.expectEqual(@as(f64, 1.0), dist.beta);
}

test "Beta.init - very small shape (α=0.1, β=0.1)" {
    const dist = try Beta(f64).init(0.1, 0.1);
    try testing.expectEqual(@as(f64, 0.1), dist.alpha);
    try testing.expectEqual(@as(f64, 0.1), dist.beta);
}

test "Beta.init - error when alpha<=0" {
    const result = Beta(f64).init(0.0, 2.0);
    try testing.expectError(error.InvalidShape, result);
}

test "Beta.init - error when beta<=0" {
    const result = Beta(f64).init(2.0, -1.0);
    try testing.expectError(error.InvalidShape, result);
}

// ============================================================================
// PDF TESTS (11 tests)
// ============================================================================

test "Beta.pdf - outside support [0,1]: x<0 returns 0" {
    const dist = try Beta(f64).init(2.0, 2.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pdf(-0.5), 1e-10);
}

test "Beta.pdf - outside support [0,1]: x>1 returns 0" {
    const dist = try Beta(f64).init(2.0, 2.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pdf(1.5), 1e-10);
}

test "Beta.pdf - boundary x=0: α<1 returns infinity" {
    const dist = try Beta(f64).init(0.5, 2.0);
    const pdf_at_zero = dist.pdf(0.0);
    try testing.expect(math.isInf(pdf_at_zero) and pdf_at_zero > 0.0);
}

test "Beta.pdf - boundary x=0: α=1 returns β" {
    const dist = try Beta(f64).init(1.0, 3.0);
    const pdf_at_zero = dist.pdf(0.0);
    try testing.expectApproxEqAbs(@as(f64, 3.0), pdf_at_zero, 1e-10);
}

test "Beta.pdf - boundary x=0: α>1 returns 0" {
    const dist = try Beta(f64).init(2.0, 2.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pdf(0.0), 1e-10);
}

test "Beta.pdf - boundary x=1: β<1 returns infinity" {
    const dist = try Beta(f64).init(2.0, 0.5);
    const pdf_at_one = dist.pdf(1.0);
    try testing.expect(math.isInf(pdf_at_one) and pdf_at_one > 0.0);
}

test "Beta.pdf - boundary x=1: β=1 returns α" {
    const dist = try Beta(f64).init(3.0, 1.0);
    const pdf_at_one = dist.pdf(1.0);
    try testing.expectApproxEqAbs(@as(f64, 3.0), pdf_at_one, 1e-10);
}

test "Beta.pdf - Uniform case: Beta(1,1) pdf(x)=1 everywhere" {
    const dist = try Beta(f64).init(1.0, 1.0);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.pdf(0.25), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.pdf(0.5), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.pdf(0.75), 1e-10);
}

test "Beta.pdf - mode at (α-1)/(α+β-2) for α,β>1" {
    const dist = try Beta(f64).init(2.0, 3.0);
    const mode = (dist.alpha - 1.0) / (dist.alpha + dist.beta - 2.0); // = 1/3
    const pdf_at_mode = dist.pdf(mode);
    const pdf_left = dist.pdf(mode - 0.1);
    const pdf_right = dist.pdf(mode + 0.1);
    try testing.expect(pdf_at_mode >= pdf_left);
    try testing.expect(pdf_at_mode >= pdf_right);
}

test "Beta.pdf - symmetry: Beta(α,β).pdf(x) = Beta(β,α).pdf(1-x)" {
    const dist1 = try Beta(f64).init(2.0, 5.0);
    const dist2 = try Beta(f64).init(5.0, 2.0);
    const x = 0.3;
    const pdf1 = dist1.pdf(x);
    const pdf2 = dist2.pdf(1.0 - x);
    try testing.expectApproxEqAbs(pdf1, pdf2, 1e-10);
}

// ============================================================================
// CDF TESTS (10 tests)
// ============================================================================

test "Beta.cdf - F(0)=0" {
    const dist = try Beta(f64).init(2.0, 2.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.cdf(0.0), 1e-10);
}

test "Beta.cdf - F(1)=1" {
    const dist = try Beta(f64).init(2.0, 2.0);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.cdf(1.0), 1e-10);
}

test "Beta.cdf - monotonically increasing" {
    const dist = try Beta(f64).init(2.0, 3.0);
    const f1 = dist.cdf(0.2);
    const f2 = dist.cdf(0.5);
    const f3 = dist.cdf(0.8);
    try testing.expect(f1 <= f2);
    try testing.expect(f2 <= f3);
}

test "Beta.cdf - x<0 returns 0" {
    const dist = try Beta(f64).init(2.0, 2.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.cdf(-0.5), 1e-10);
}

test "Beta.cdf - x>1 returns 1" {
    const dist = try Beta(f64).init(2.0, 2.0);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.cdf(1.5), 1e-10);
}

test "Beta.cdf - Uniform case: Beta(1,1) cdf(x)=x" {
    const dist = try Beta(f64).init(1.0, 1.0);
    try testing.expectApproxEqAbs(@as(f64, 0.3), dist.cdf(0.3), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.5), dist.cdf(0.5), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.7), dist.cdf(0.7), 1e-10);
}

test "Beta.cdf - symmetric case: F(0.5)≈0.5 for Beta(α,α)" {
    const dist = try Beta(f64).init(3.0, 3.0);
    const cdf_mid = dist.cdf(0.5);
    try testing.expectApproxEqAbs(@as(f64, 0.5), cdf_mid, 1e-2);
}

test "Beta.cdf - values bounded in [0,1]" {
    const dist = try Beta(f64).init(2.0, 5.0);
    for ([_]f64{ 0.0, 0.25, 0.5, 0.75, 1.0 }) |x| {
        const cdf_val = dist.cdf(x);
        try testing.expect(cdf_val >= 0.0);
        try testing.expect(cdf_val <= 1.0);
    }
}

test "Beta.cdf - skewed distribution (α<β)" {
    const dist = try Beta(f64).init(1.0, 5.0);
    // Should have more mass near 0
    const cdf_near_zero = dist.cdf(0.1);
    const cdf_near_one = dist.cdf(0.9);
    try testing.expect(cdf_near_zero > cdf_near_one - 0.5);
}

// ============================================================================
// QUANTILE TESTS (10 tests)
// ============================================================================

test "Beta.quantile - Q(0)=0" {
    const dist = try Beta(f64).init(2.0, 2.0);
    const q = try dist.quantile(0.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), q, 1e-10);
}

test "Beta.quantile - Q(1)=1" {
    const dist = try Beta(f64).init(2.0, 2.0);
    const q = try dist.quantile(1.0);
    try testing.expectApproxEqAbs(@as(f64, 1.0), q, 1e-10);
}

test "Beta.quantile - Q(0.5)≈median" {
    const dist = try Beta(f64).init(2.0, 2.0);
    const q = try dist.quantile(0.5);
    try testing.expectApproxEqAbs(@as(f64, 0.5), q, 1e-3);
}

test "Beta.quantile - Uniform case: Beta(1,1) Q(p)=p" {
    const dist = try Beta(f64).init(1.0, 1.0);
    try testing.expectApproxEqAbs(@as(f64, 0.3), try dist.quantile(0.3), 1e-3);
    try testing.expectApproxEqAbs(@as(f64, 0.5), try dist.quantile(0.5), 1e-3);
    try testing.expectApproxEqAbs(@as(f64, 0.7), try dist.quantile(0.7), 1e-3);
}

test "Beta.quantile - inverse property: cdf(quantile(p))≈p" {
    const dist = try Beta(f64).init(2.0, 3.0);
    for ([_]f64{ 0.05, 0.25, 0.5, 0.75, 0.95 }) |p| {
        const q = try dist.quantile(p);
        const cdf_q = dist.cdf(q);
        try testing.expectApproxEqAbs(p, cdf_q, 1e-3);
    }
}

test "Beta.quantile - error when p<0" {
    const dist = try Beta(f64).init(2.0, 2.0);
    const result = dist.quantile(-0.1);
    try testing.expectError(error.InvalidProbability, result);
}

test "Beta.quantile - error when p>1" {
    const dist = try Beta(f64).init(2.0, 2.0);
    const result = dist.quantile(1.1);
    try testing.expectError(error.InvalidProbability, result);
}

test "Beta.quantile - monotonically increasing" {
    const dist = try Beta(f64).init(3.0, 2.0);
    const q1 = try dist.quantile(0.2);
    const q2 = try dist.quantile(0.5);
    const q3 = try dist.quantile(0.8);
    try testing.expect(q1 <= q2);
    try testing.expect(q2 <= q3);
}

test "Beta.quantile - symmetric case: Q(p)=1-Q(1-p) for Beta(α,α)" {
    const dist = try Beta(f64).init(3.0, 3.0);
    const q_low = try dist.quantile(0.25);
    const q_high = try dist.quantile(0.75);
    try testing.expectApproxEqAbs(q_low, 1.0 - q_high, 1e-2);
}

// ============================================================================
// LOGPDF TESTS (5 tests)
// ============================================================================

test "Beta.logpdf - consistency with pdf" {
    const dist = try Beta(f64).init(2.0, 2.0);
    const x = 0.5;
    const pdf_val = dist.pdf(x);
    const logpdf_val = dist.logpdf(x);
    const expected_logpdf = @log(pdf_val);
    try testing.expectApproxEqAbs(expected_logpdf, logpdf_val, 1e-10);
}

test "Beta.logpdf - x<0 returns -infinity" {
    const dist = try Beta(f64).init(2.0, 2.0);
    const logpdf_val = dist.logpdf(-0.5);
    try testing.expect(math.isNegativeInf(logpdf_val));
}

test "Beta.logpdf - x>1 returns -infinity" {
    const dist = try Beta(f64).init(2.0, 2.0);
    const logpdf_val = dist.logpdf(1.5);
    try testing.expect(math.isNegativeInf(logpdf_val));
}

test "Beta.logpdf - numerical stability for small x" {
    const dist = try Beta(f64).init(3.0, 2.0);
    const x = 1e-6;
    const logpdf_val = dist.logpdf(x);
    try testing.expect(!math.isNan(logpdf_val));
    try testing.expect(logpdf_val < 0.0); // Should be negative
}

test "Beta.logpdf - Uniform case: Beta(1,1) logpdf=0" {
    const dist = try Beta(f64).init(1.0, 1.0);
    const logpdf_val = dist.logpdf(0.5);
    try testing.expectApproxEqAbs(@as(f64, 0.0), logpdf_val, 1e-10);
}

// ============================================================================
// SAMPLE TESTS (10 tests)
// ============================================================================

test "Beta.sample - all samples in [0,1]" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const dist = try Beta(f64).init(2.0, 2.0);

    for (0..1000) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample >= 0.0);
        try testing.expect(sample <= 1.0);
    }
}

test "Beta.sample - mean convergence for Beta(2,2)" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const dist = try Beta(f64).init(2.0, 2.0);
    const expected_mean = 2.0 / (2.0 + 2.0); // 0.5

    var sum: f64 = 0.0;
    for (0..10000) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / 10000.0;
    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.03);
}

test "Beta.sample - mean convergence for Beta(5,1)" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const dist = try Beta(f64).init(5.0, 1.0);
    const expected_mean = 5.0 / (5.0 + 1.0); // ≈0.833

    var sum: f64 = 0.0;
    for (0..10000) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / 10000.0;
    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.05);
}

test "Beta.sample - mean convergence for Beta(1,5)" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const dist = try Beta(f64).init(1.0, 5.0);
    const expected_mean = 1.0 / (1.0 + 5.0); // ≈0.167

    var sum: f64 = 0.0;
    for (0..10000) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / 10000.0;
    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.05);
}

test "Beta.sample - variance convergence" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const dist = try Beta(f64).init(3.0, 4.0);
    const a = dist.alpha;
    const b = dist.beta;
    const expected_var = (a * b) / ((a + b) * (a + b) * (a + b + 1.0));

    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    for (0..10000) |_| {
        const s = dist.sample(rng);
        sum += s;
        sum_sq += s * s;
    }
    const mean = sum / 10000.0;
    const sample_var = sum_sq / 10000.0 - mean * mean;
    try testing.expectApproxEqRel(expected_var, sample_var, 0.10);
}

test "Beta.sample - symmetric case: Beta(3,3) mean≈0.5" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const dist = try Beta(f64).init(3.0, 3.0);

    var sum: f64 = 0.0;
    for (0..10000) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / 10000.0;
    try testing.expectApproxEqRel(@as(f64, 0.5), sample_mean, 0.03);
}

test "Beta.sample - very small alpha (0.5)" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const dist = try Beta(f64).init(0.5, 2.0);

    for (0..1000) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample >= 0.0);
        try testing.expect(sample <= 1.0);
    }
}

test "Beta.sample - very large alpha (100)" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const dist = try Beta(f64).init(100.0, 100.0);
    const expected_mean = 0.5;

    var sum: f64 = 0.0;
    for (0..1000) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample >= 0.0);
        try testing.expect(sample <= 1.0);
        sum += sample;
    }
    const sample_mean = sum / 1000.0;
    // Large alpha should cluster tightly around mean
    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.05);
}

test "Beta.sample - edge case Beta(0.1, 0.1)" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const dist = try Beta(f64).init(0.1, 0.1);

    for (0..100) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample >= 0.0);
        try testing.expect(sample <= 1.0);
    }
}

// ============================================================================
// INTEGRATION TESTS (5 tests)
// ============================================================================

test "Beta integration - PDF normalization via numerical integration" {
    const dist = try Beta(f64).init(2.0, 3.0);

    // Trapezoid rule integration
    var integral: f64 = 0.0;
    const n = 1000;
    const dx = 1.0 / @as(f64, @floatFromInt(n));

    for (0..n) |i| {
        const i_f = @as(f64, @floatFromInt(i));
        const x1 = i_f * dx;
        const x2 = (i_f + 1.0) * dx;
        const y1 = dist.pdf(x1);
        const y2 = dist.pdf(x2);
        integral += (y1 + y2) * dx / 2.0;
    }

    try testing.expectApproxEqAbs(@as(f64, 1.0), integral, 0.01);
}

test "Beta integration - CDF-quantile inverse property" {
    const dist = try Beta(f64).init(2.0, 4.0);

    for ([_]f64{ 0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0 }) |p| {
        const q = try dist.quantile(p);
        const cdf_q = dist.cdf(q);
        try testing.expectApproxEqAbs(p, cdf_q, 1e-2);
    }
}

test "Beta integration - ensemble statistics" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const dist = try Beta(f64).init(4.0, 6.0);
    const expected_mean = 4.0 / 10.0;

    var sum: f64 = 0.0;
    for (0..10000) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / 10000.0;
    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.03);
}

test "Beta integration - Beta(2,2) triangular-like shape with mode at 0.5" {
    const dist = try Beta(f64).init(2.0, 2.0);

    const mode = (dist.alpha - 1.0) / (dist.alpha + dist.beta - 2.0); // = 0.5
    const pdf_at_mode = dist.pdf(mode);

    // PDF should be highest at mode
    for ([_]f64{ 0.1, 0.3, 0.7, 0.9 }) |x| {
        try testing.expect(pdf_at_mode >= dist.pdf(x));
    }
}

test "Beta integration - Beta(1,1) equals Uniform(0,1)" {
    const dist = try Beta(f64).init(1.0, 1.0);

    // PDF should be constant 1.0
    for ([_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 }) |x| {
        try testing.expectApproxEqAbs(@as(f64, 1.0), dist.pdf(x), 1e-10);
    }

    // CDF should be linear
    for ([_]f64{ 0.0, 0.25, 0.5, 0.75, 1.0 }) |x| {
        try testing.expectApproxEqAbs(x, dist.cdf(x), 1e-10);
    }
}
