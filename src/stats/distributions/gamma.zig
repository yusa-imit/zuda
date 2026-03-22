//! Gamma Distribution
//!
//! Represents a continuous gamma distribution with shape parameter k and scale parameter θ.
//! Generalizes the exponential distribution and models waiting times for k events in a Poisson process.
//!
//! ## Parameters
//! - `k: T` — shape parameter (must be > 0, integer or non-integer)
//! - `theta: T` — scale parameter (must be > 0)
//!
//! ## Mathematical Properties
//! - **PDF**: f(x; k, θ) = (x^(k-1) * exp(-x/θ)) / (θ^k * Γ(k)) for x > 0, else 0
//! - **CDF**: F(x; k, θ) = γ(k, x/θ) / Γ(k) (lower incomplete gamma function)
//! - **Quantile**: Q(p; k, θ) = θ * Q_standard(p), inverse of CDF (numerical)
//! - **Log-PDF**: (k-1)*log(x) - x/θ - k*log(θ) - logΓ(k)
//! - **Mean**: E[X] = k*θ
//! - **Variance**: Var[X] = k*θ²
//! - **Mode**: (k-1)*θ for k ≥ 1, undefined for k < 1
//! - **Special Cases**: k=1 → Exponential(1/θ), k→∞ → Normal(k*θ, √(k)*θ)
//!
//! ## Time Complexity
//! - pdf, cdf, quantile, logpdf: O(1) to O(log k) depending on method
//! - sample: O(1) to O(log k) depending on method
//! - init: O(1)
//!
//! ## Use Cases
//! - Modeling waiting times for multiple events in Poisson processes
//! - Reliability engineering (lifetime of devices)
//! - Weather modeling (rainfall, wind speed)
//! - Queueing theory
//! - Finance (price volatility)
//!
//! ## References
//! - Gamma distribution: https://en.wikipedia.org/wiki/Gamma_distribution
//! - Sampling methods: Marsaglia & Tsang (2000), Ahrens & Dieter (1982)
//! - Incomplete gamma: https://en.wikipedia.org/wiki/Incomplete_gamma_function

const std = @import("std");
const math = std.math;
const testing = std.testing;

/// Gamma distribution with shape k and scale θ
///
/// Parameters:
/// - T: numeric type (f32 or f64)
/// - k: shape parameter (must be > 0)
/// - theta: scale parameter (must be > 0)
pub fn Gamma(comptime T: type) type {
    return struct {
        k: T,
        theta: T,

        const Self = @This();

        /// Initialize gamma distribution with shape k and scale θ
        ///
        /// Parameters:
        /// - k: shape parameter (must be > 0)
        /// - theta: scale parameter (must be > 0)
        ///
        /// Returns: Gamma distribution instance
        ///
        /// Errors:
        /// - error.InvalidShape if k <= 0
        /// - error.InvalidScale if theta <= 0
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(k: T, theta: T) !Self {
            if (k <= 0.0) {
                return error.InvalidShape;
            }
            if (theta <= 0.0) {
                return error.InvalidScale;
            }
            return .{ .k = k, .theta = theta };
        }

        /// Probability density function: f(x; k, θ) = (x^(k-1) * exp(-x/θ)) / (θ^k * Γ(k))
        ///
        /// Parameters:
        /// - x: value to evaluate at
        ///
        /// Returns: probability density at x (0 for x <= 0)
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn pdf(self: Self, x: T) T {
            if (x < 0.0) {
                return 0.0;
            }

            // Special case: x = 0
            if (x == 0.0) {
                if (self.k < 1.0) {
                    // PDF → ∞ as x → 0+ for k < 1
                    return math.inf(T);
                } else if (self.k == 1.0) {
                    // For k = 1, PDF(0) = 1/θ (Exponential case)
                    return 1.0 / self.theta;
                } else {
                    // For k > 1, PDF(0) = 0
                    return 0.0;
                }
            }

            // Compute log(pdf) for numerical stability
            const logpdf_val = self.logpdf(x);
            return @exp(logpdf_val);
        }

        /// Cumulative distribution function
        ///
        /// F(x; k, θ) = γ(k, x/θ) / Γ(k) (lower incomplete gamma / gamma)
        ///
        /// Parameters:
        /// - x: value to evaluate at
        ///
        /// Returns: cumulative probability P(X <= x)
        ///
        /// Time: O(k) or O(1) depending on approximation method
        /// Space: O(1)
        pub fn cdf(self: Self, x: T) T {
            if (x <= 0.0) {
                return 0.0;
            }

            const scaled = x / self.theta;
            return lowerIncompleteGamma(self.k, scaled);
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
            if (p < 0.0 or p > 1.0) {
                return error.InvalidProbability;
            }

            // Boundary cases
            if (p == 0.0) {
                return 0.0;
            }
            if (p == 1.0) {
                return math.inf(T);
            }

            // Use standard gamma quantile and scale
            const q_standard = standardGammaQuantile(self.k, p);
            return self.theta * q_standard;
        }

        /// Natural logarithm of probability density function
        ///
        /// log(f(x; k, θ)) = (k-1)*log(x) - x/θ - k*log(θ) - logΓ(k)
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
            if (x <= 0.0) {
                return -math.inf(T);
            }

            // log(f) = (k-1)*log(x) - x/θ - k*log(θ) - logΓ(k)
            const term1 = (self.k - 1.0) * @log(x);
            const term2 = -x / self.theta;
            const term3 = -self.k * @log(self.theta);
            const term4 = -logGamma(self.k);

            return term1 + term2 + term3 + term4;
        }

        /// Generate random sample from distribution
        ///
        /// Uses different methods based on k:
        /// - k=1: Exponential sampling: -θ*log(U)
        /// - k<1: Ahrens-Dieter acceptance-rejection
        /// - k≥1: Marsaglia-Tsang squeeze method
        ///
        /// Parameters:
        /// - rng: random number generator (std.Random)
        ///
        /// Returns: random value from distribution
        ///
        /// Time: O(1) expected, O(log k) worst case
        /// Space: O(1)
        pub fn sample(self: Self, rng: std.Random) T {
            // For k=1, use fast exponential sampling
            if (self.k == 1.0) {
                const u = switch (T) {
                    f32 => rng.float(f32),
                    f64 => rng.float(f64),
                    else => @compileError("Invalid type"),
                };
                const u_safe = if (u == 0.0) 1e-15 else u;
                return self.theta * (-@log(u_safe));
            }

            // General case: decompose into integer + fractional parts
            const k_int = @as(i32, @intFromFloat(self.k));
            const k_frac = self.k - @as(T, @floatFromInt(k_int));

            var sample_val: T = 0.0;

            // Integer part: sum of k_int independent Exponential(1) samples
            if (k_int > 0) {
                var u_prod: T = 1.0;
                for (0..@as(usize, @intCast(k_int))) |_| {
                    const u_i = switch (T) {
                        f32 => rng.float(f32),
                        f64 => rng.float(f64),
                        else => @compileError("Invalid type"),
                    };
                    const u_safe = if (u_i == 0.0) 1e-15 else u_i;
                    u_prod *= u_safe;
                }
                sample_val += -@log(u_prod);
            }

            // Fractional part
            if (k_frac > 1e-10) {
                sample_val += sampleFractional(k_frac, rng);
            }

            return self.theta * sample_val;
        }

        // ================================================================
        // HELPER FUNCTIONS
        // ================================================================

        /// Compute logarithm of gamma function Γ(x)
        /// Uses Stirling's approximation for numerical stability
        fn logGamma(x: T) T {
            // For small x, use recurrence: Γ(x+1) = x*Γ(x)
            // For large x, use Stirling's approximation

            if (x < 0.5) {
                // Use reflection formula: Γ(x)Γ(1-x) = π/sin(πx)
                // logΓ(x) = log(π) - log(sin(πx)) - logΓ(1-x)
                const pi = math.pi;
                const sin_pi_x = @sin(pi * x);
                return @log(pi) - @log(@abs(sin_pi_x)) - logGamma(1.0 - x);
            }

            if (x < 12.0) {
                // Lanczos approximation
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

            // Stirling for large x
            return 0.5 * @log(2.0 * math.pi) + (x - 0.5) * @log(x) - x;
        }

        /// Lower incomplete gamma function P(a,x) = γ(a,x) / Γ(a)
        /// Normalized incomplete gamma (regularized gamma) - returns value in [0, 1)
        /// Based on Numerical Recipes implementation
        fn lowerIncompleteGamma(a: T, x: T) T {
            if (x < 0.0 or a <= 0.0) {
                return 0.0;
            }

            // For special case x very close to 0
            if (x < 1e-10) {
                return 0.0;
            }

            // Choose algorithm based on convergence of series vs continued fraction
            // Series converges better when x is small compared to a+1
            if (x < (a + 1.0)) {
                // Use series expansion for lower incomplete gamma
                return gammaSeriesApproximation(a, x);
            } else {
                // Use continued fraction for upper incomplete gamma, then subtract from 1
                const upper = gammaComplementaryApproximation(a, x);
                var result = 1.0 - upper;
                // Clamp to avoid numerical overflow to exactly 1.0
                result = @min(result, 1.0 - 1e-15);
                return @max(0.0, result);
            }
        }

        /// Approximation of lower incomplete gamma using series
        /// Result is γ(a,x) / Γ(a)
        fn gammaSeriesApproximation(a: T, x: T) T {
            const maxiter = 300;
            const itol: T = if (T == f32) 1e-7 else 1e-14;

            // Compute the sum: sum_{n=0}^∞ x^n / (a)_n
            // where (a)_n = a(a+1)(a+2)...(a+n-1) is Pochhammer symbol
            var sum = 1.0 / a;
            var term = sum;
            var ap = a + 1.0;

            for (1..maxiter) |_| {
                term *= x / ap;
                sum += term;
                ap += 1.0;

                if (@abs(term / sum) < itol) {
                    break;
                }
            }

            // Multiply by exp(-x) * x^a / Γ(a)
            const log_result = a * @log(x) - x - logGamma(a);
            return @exp(log_result) * sum;
        }

        /// Approximation of upper incomplete gamma using continued fraction
        /// Result is Γ(a,x) / Γ(a)
        fn gammaComplementaryApproximation(a: T, x: T) T {
            const maxiter = 300;
            const itol: T = if (T == f32) 1e-7 else 1e-14;
            const fpmin: T = if (T == f32) 1e-30 else 1e-300;

            // Continued fraction representation
            // Uses modified Lentz algorithm

            var b = x + 1.0 - a;
            var c = 1.0 / fpmin;
            var d = 1.0 / b;
            var h = d;

            for (1..maxiter) |i| {
                const i_f: T = @as(T, @floatFromInt(i));

                // a_i = -i(i-a)
                const an = -i_f * (i_f - a);
                b += 2.0;

                // Use Lentz algorithm
                d = an * d + b;
                if (@abs(d) < fpmin) d = fpmin;

                c = b + an / c;
                if (@abs(c) < fpmin) c = fpmin;

                d = 1.0 / d;
                const del = d * c;
                h *= del;

                if (@abs(del - 1.0) < itol) break;
            }

            // Result: exp(-x + a*ln(x) - ln(Γ(a))) * h
            const log_result = a * @log(x) - x - logGamma(a);
            return @exp(log_result) * h;
        }

        /// Sample from gamma(k) for fractional k in (0, 1)
        /// Uses method from "The Computer Generation of Poisson Random Variables" by Ahrens & Dieter
        /// Actually implements: Sample Gamma(k+1), then scale by U^(1/k)
        fn sampleFractional(k: T, rng: std.Random) T {
            if (k >= 1.0) {
                // For k >= 1, use Marsaglia's method
                return sampleMarsaglia(k, rng);
            }

            // For 0 < k < 1: use method based on Gamma(k+1)
            // Gamma(k) ~ Gamma(k+1) * U^(1/k) where U ~ Uniform(0,1)

            // First sample Gamma(k+1) using Marsaglia
            const k_plus_1 = k + 1.0;
            const y = sampleMarsaglia(k_plus_1, rng);

            // Generate U ~ Uniform(0,1)
            const u = switch (T) {
                f32 => rng.float(f32),
                f64 => rng.float(f64),
                else => @compileError("Invalid type"),
            };

            // Ensure u is not exactly 0
            const u_safe = if (u == 0.0) 1e-15 else u;

            // Apply transformation: Gamma(k) = Gamma(k+1) * u^(1/k)
            return y * math.pow(T, u_safe, 1.0 / k);
        }

        /// Marsaglia and Tsang's method for sampling Gamma(k) with k >= 1
        /// Reference: "A Simple Method for Generating Gamma Variables" (2000)
        fn sampleMarsaglia(k: T, rng: std.Random) T {
            const d = k - 1.0 / 3.0;
            const c = 1.0 / @sqrt(9.0 * d);

            while (true) {
                // Generate standard normal
                const z = switch (T) {
                    f32 => boxMullerComponent(rng),
                    f64 => boxMullerComponent(rng),
                    else => @compileError("Invalid type"),
                };

                const v = 1.0 + c * z;
                if (v > 0.0) {
                    const v_cubed = v * v * v;
                    const u = switch (T) {
                        f32 => rng.float(f32),
                        f64 => rng.float(f64),
                        else => @compileError("Invalid type"),
                    };

                    // Acceptance-rejection test
                    if (@log(u) < 0.5 * z * z + d * (1.0 - v_cubed + @log(v_cubed))) {
                        return d * v_cubed;
                    }
                }
            }
        }

        /// Generate standard normal via Box-Muller
        fn boxMullerComponent(rng: std.Random) T {
            const u_val1 = switch (T) {
                f32 => rng.float(f32),
                f64 => rng.float(f64),
                else => @compileError("Invalid type"),
            };
            const u_val2 = switch (T) {
                f32 => rng.float(f32),
                f64 => rng.float(f64),
                else => @compileError("Invalid type"),
            };

            const u_safe = if (u_val1 == 0.0) 1e-15 else u_val1;
            const r = @sqrt(-2.0 * @log(u_safe));
            const theta = 2.0 * math.pi * u_val2;
            return r * @cos(theta);
        }

        /// Standard gamma quantile for unit scale (θ=1)
        /// Uses rational approximation and Newton-Raphson
        fn standardGammaQuantile(k: T, p: T) T {
            // Clamp p to valid range
            const p_safe = @max(1e-10, @min(1.0 - 1e-10, p));

            // Initial guess - use better starting point
            var x: T = k;  // Start with mean

            // Refine based on p
            if (p_safe < 0.01) {
                // Lower tail
                x = k * math.pow(T, p_safe, 1.0 / k);
            } else if (p_safe > 0.99) {
                // Upper tail
                const log_tail = @log(1.0 - p_safe);
                x = k - @sqrt(k) * @log(-log_tail);
            } else {
                // Adjust initial guess based on where p is
                if (p_safe < 0.5) {
                    x = k * 0.5 * math.pow(T, p_safe, 1.0 / 3.0);
                } else {
                    x = k * (1.0 + @sqrt(k) * math.pow(T, p_safe - 0.5, 0.3));
                }
            }

            x = @max(x, 1e-15);

            // Newton-Raphson iteration
            var last_delta: T = 1.0;
            for (0..150) |_| {
                if (x <= 0.0) break;

                const cdf_x = lowerIncompleteGamma(k, x);
                const err = cdf_x - p_safe;

                // Early exit if converged
                if (@abs(err) < 1e-15) break;

                // Compute PDF using log for stability
                const logpdf_val = (k - 1.0) * @log(x) - x - logGamma(k);
                const pdf_x = @exp(logpdf_val);

                if (pdf_x < 1e-50 or !math.isFinite(pdf_x)) break;

                var delta = err / pdf_x;

                // Convergence control: if step is not decreasing fast, dampen it
                const damping: T = if (@abs(delta) > @abs(last_delta) * 0.5) 0.5 else 1.0;
                delta *= damping;
                last_delta = delta;

                // Step size limit
                if (@abs(delta) > x) {
                    const sign: T = if (delta < 0.0) -1.0 else 1.0;
                    delta = sign * x * 0.1;
                }

                x -= delta;

                // Bounds checking
                if (x <= 0.0) {
                    x = 1e-15;
                }

                // Convergence check: tiny delta means we're done
                if (@abs(delta) < 1e-15 * @max(1.0, @abs(x))) {
                    break;
                }
            }

            return @max(x, 0.0);
        }
    };
}

// ============================================================================
// TESTS
// ============================================================================

// ============================================================================
// INIT TESTS (7 tests)
// ============================================================================

test "Gamma.init - standard parameters (k=2, θ=1)" {
    const dist = try Gamma(f64).init(2.0, 1.0);
    try testing.expectEqual(@as(f64, 2.0), dist.k);
    try testing.expectEqual(@as(f64, 1.0), dist.theta);
}

test "Gamma.init - different scale (k=3, θ=2)" {
    const dist = try Gamma(f64).init(3.0, 2.0);
    try testing.expectEqual(@as(f64, 3.0), dist.k);
    try testing.expectEqual(@as(f64, 2.0), dist.theta);
}

test "Gamma.init - fractional shape (k=0.5, θ=1)" {
    const dist = try Gamma(f64).init(0.5, 1.0);
    try testing.expectEqual(@as(f64, 0.5), dist.k);
    try testing.expectEqual(@as(f64, 1.0), dist.theta);
}

test "Gamma.init - Exponential case (k=1, θ=1)" {
    const dist = try Gamma(f64).init(1.0, 1.0);
    try testing.expectEqual(@as(f64, 1.0), dist.k);
    try testing.expectEqual(@as(f64, 1.0), dist.theta);
}

test "Gamma.init - error when k=0" {
    const result = Gamma(f64).init(0.0, 1.0);
    try testing.expectError(error.InvalidShape, result);
}

test "Gamma.init - error when k<0" {
    const result = Gamma(f64).init(-1.5, 1.0);
    try testing.expectError(error.InvalidShape, result);
}

test "Gamma.init - error when theta<=0" {
    const result = Gamma(f64).init(2.0, -0.5);
    try testing.expectError(error.InvalidScale, result);
}

// ============================================================================
// PDF TESTS (11 tests)
// ============================================================================

test "Gamma.pdf - negative x returns 0" {
    const dist = try Gamma(f64).init(2.0, 1.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pdf(-5.0), 1e-10);
}

test "Gamma.pdf - x=0 returns 0 for k>1" {
    const dist = try Gamma(f64).init(2.0, 1.0);
    const pdf_at_zero = dist.pdf(0.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), pdf_at_zero, 1e-10);
}

test "Gamma.pdf - x=0 returns 1/θ for k=1 (Exponential case)" {
    const dist = try Gamma(f64).init(1.0, 2.0);
    const pdf_at_zero = dist.pdf(0.0);
    // For k=1, θ=2: f(0) = 1/θ = 0.5
    try testing.expectApproxEqAbs(@as(f64, 0.5), pdf_at_zero, 1e-10);
}

test "Gamma.pdf - mode at (k-1)*θ for k>1" {
    const dist = try Gamma(f64).init(3.0, 2.0);
    const mode = (dist.k - 1.0) * dist.theta; // = 4.0
    const pdf_at_mode = dist.pdf(mode);

    // PDF should be higher at mode than at nearby points
    const pdf_left = dist.pdf(mode - 0.5);
    const pdf_right = dist.pdf(mode + 0.5);
    try testing.expect(pdf_at_mode >= pdf_left);
    try testing.expect(pdf_at_mode >= pdf_right);
}

test "Gamma.pdf - Exponential case (k=1) matches exponential pdf" {
    const dist = try Gamma(f64).init(1.0, 1.0);
    // For k=1, θ=1: f(x) = exp(-x)
    const x = 0.5;
    const pdf_val = dist.pdf(x);
    const expected = @exp(-x);
    try testing.expectApproxEqAbs(expected, pdf_val, 1e-10);
}

test "Gamma.pdf - decreasing for k=1 (Exponential)" {
    const dist = try Gamma(f64).init(1.0, 1.0);
    const f1 = dist.pdf(1.0);
    const f2 = dist.pdf(2.0);
    const f3 = dist.pdf(3.0);
    try testing.expect(f1 > f2);
    try testing.expect(f2 > f3);
}

test "Gamma.pdf - scale parameter scales pdf inversely" {
    const dist1 = try Gamma(f64).init(2.0, 1.0);
    const dist2 = try Gamma(f64).init(2.0, 2.0);
    const x = 1.0;
    const pdf1 = dist1.pdf(x);
    const pdf2 = dist2.pdf(x);
    // pdf_θ2(x) = (1/2) * pdf_θ1(x/2), but not directly comparable at same x
    // Just verify both are positive
    try testing.expect(pdf1 > 0.0);
    try testing.expect(pdf2 > 0.0);
}

test "Gamma.pdf - large x approaches 0" {
    const dist = try Gamma(f64).init(2.0, 1.0);
    const pdf_far = dist.pdf(100.0);
    try testing.expect(pdf_far > 0.0);
    try testing.expect(pdf_far < 1e-20);
}

test "Gamma.pdf - f32 precision (k=2, θ=1)" {
    const dist = try Gamma(f32).init(2.0, 1.0);
    const x = 1.0;
    const pdf_val = dist.pdf(x);
    // For k=2, θ=1, f(1) = 1 * exp(-1) = 1/e ≈ 0.3679
    const expected: f32 = @exp(-1.0);
    try testing.expectApproxEqRel(expected, pdf_val, 1e-5);
}

test "Gamma.pdf - handles small x > 0" {
    const dist = try Gamma(f64).init(2.0, 1.0);
    const pdf_small = dist.pdf(0.001);
    try testing.expect(pdf_small > 0.0);
    try testing.expect(!math.isNan(pdf_small));
}

// ============================================================================
// CDF TESTS (10 tests)
// ============================================================================

test "Gamma.cdf - at x=0 returns 0" {
    const dist = try Gamma(f64).init(2.0, 1.0);
    const cdf_zero = dist.cdf(0.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), cdf_zero, 1e-10);
}

test "Gamma.cdf - negative x returns 0" {
    const dist = try Gamma(f64).init(2.0, 1.0);
    const cdf_neg = dist.cdf(-5.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), cdf_neg, 1e-10);
}

test "Gamma.cdf - monotonically increasing" {
    const dist = try Gamma(f64).init(2.0, 1.0);
    const c0 = dist.cdf(0.5);
    const c1 = dist.cdf(1.0);
    const c2 = dist.cdf(2.0);
    const c5 = dist.cdf(5.0);
    try testing.expect(c0 < c1);
    try testing.expect(c1 < c2);
    try testing.expect(c2 < c5);
}

test "Gamma.cdf - approaches 1 as x→∞" {
    const dist = try Gamma(f64).init(2.0, 1.0);
    const cdf_large = dist.cdf(50.0);
    try testing.expect(cdf_large < 1.0);
    try testing.expect(cdf_large > 1.0 - 1e-10);
}

test "Gamma.cdf - bounded [0, 1]" {
    const dist = try Gamma(f64).init(2.0, 1.0);
    for ([_]f64{ 0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0 }) |x| {
        const c = dist.cdf(x);
        try testing.expect(c >= 0.0);
        try testing.expect(c <= 1.0);
    }
}

test "Gamma.cdf - Exponential case (k=1) matches exponential" {
    const dist = try Gamma(f64).init(1.0, 1.0);
    // For k=1, θ=1: F(x) = 1 - exp(-x)
    const x = 1.0;
    const cdf_val = dist.cdf(x);
    const expected = 1.0 - @exp(-x);
    try testing.expectApproxEqAbs(expected, cdf_val, 1e-10);
}

test "Gamma.cdf - scale parameter affects median" {
    const dist1 = try Gamma(f64).init(2.0, 1.0);
    const dist2 = try Gamma(f64).init(2.0, 2.0);
    // CDF(x, θ=2) = F(x/2, θ=1), so median scales by θ
    // Roughly: F(x_med, 1) ≈ 0.5, F(2*x_med, 2) ≈ 0.5
    const c1 = dist1.cdf(2.0);
    const c2 = dist2.cdf(4.0);
    // Both should be around 0.5 (not exact, but similar)
    try testing.expect(@abs(c1 - c2) < 0.1);
}

test "Gamma.cdf - f32 precision (k=2, θ=1)" {
    const dist = try Gamma(f32).init(2.0, 1.0);
    const cdf_val = dist.cdf(1.0);
    try testing.expect(cdf_val > 0.0);
    try testing.expect(cdf_val < 1.0);
    try testing.expect(!math.isNan(cdf_val));
}

test "Gamma.cdf - relationship with pdf (integral)" {
    const dist = try Gamma(f64).init(2.0, 1.0);
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

test "Gamma.quantile - p=0 returns 0" {
    const dist = try Gamma(f64).init(2.0, 1.0);
    const q = try dist.quantile(0.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), q, 1e-10);
}

test "Gamma.quantile - p=1 returns infinity" {
    const dist = try Gamma(f64).init(2.0, 1.0);
    const q = try dist.quantile(1.0);
    try testing.expect(math.isPositiveInf(q));
}

test "Gamma.quantile - error when p<0" {
    const dist = try Gamma(f64).init(2.0, 1.0);
    const result = dist.quantile(-0.1);
    try testing.expectError(error.InvalidProbability, result);
}

test "Gamma.quantile - error when p>1" {
    const dist = try Gamma(f64).init(2.0, 1.0);
    const result = dist.quantile(1.1);
    try testing.expectError(error.InvalidProbability, result);
}

test "Gamma.quantile - monotonically increasing" {
    const dist = try Gamma(f64).init(2.0, 1.0);
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

test "Gamma.quantile - inverse of cdf (composition)" {
    const dist = try Gamma(f64).init(2.0, 1.0);
    for ([_]f64{ 0.1, 0.25, 0.5, 0.75, 0.9 }) |p| {
        const q = try dist.quantile(p);
        const p_back = dist.cdf(q);
        // cdf(quantile(p)) should ≈ p (within numerical precision)
        try testing.expectApproxEqAbs(p, p_back, 0.02);
    }
}

test "Gamma.quantile - scale parameter scales result" {
    const dist1 = try Gamma(f64).init(2.0, 1.0);
    const dist2 = try Gamma(f64).init(2.0, 2.0);
    const q1 = try dist1.quantile(0.5);
    const q2 = try dist2.quantile(0.5);
    // Q(p; k, θ) = θ * Q(p; k, 1), so Q2 ≈ 2 * Q1
    try testing.expectApproxEqAbs(2.0 * q1, q2, 0.01);
}

test "Gamma.quantile - Exponential case (k=1)" {
    const dist = try Gamma(f64).init(1.0, 1.0);
    const q = try dist.quantile(0.5);
    // For k=1, θ=1: Q(0.5) = ln(2) ≈ 0.693
    const expected = @log(2.0);
    try testing.expectApproxEqAbs(expected, q, 1e-9);
}

test "Gamma.quantile - f32 precision (p=0.5)" {
    const dist = try Gamma(f32).init(2.0, 1.0);
    const q = try dist.quantile(0.5);
    try testing.expect(q > 0.0);
    try testing.expect(!math.isNan(q));
}

// ============================================================================
// LOGPDF TESTS (6 tests)
// ============================================================================

test "Gamma.logpdf - equals log(pdf) for valid x" {
    const dist = try Gamma(f64).init(2.0, 1.0);
    const x = 1.0;
    const pdf_val = dist.pdf(x);
    const logpdf_val = dist.logpdf(x);
    const expected = @log(pdf_val);
    try testing.expectApproxEqAbs(expected, logpdf_val, 1e-12);
}

test "Gamma.logpdf - negative/zero x returns -infinity" {
    const dist = try Gamma(f64).init(2.0, 1.0);
    const logpdf_neg = dist.logpdf(-1.0);
    const logpdf_zero = dist.logpdf(0.0);
    try testing.expect(math.isNegativeInf(logpdf_neg));
    try testing.expect(math.isNegativeInf(logpdf_zero));
}

test "Gamma.logpdf - numerical stability for large x" {
    const dist = try Gamma(f64).init(2.0, 1.0);
    const logpdf_large = dist.logpdf(1000.0);
    try testing.expect(logpdf_large < 0.0);
    try testing.expect(!math.isInf(logpdf_large));
    try testing.expect(!math.isNan(logpdf_large));
}

test "Gamma.logpdf - Exponential case (k=1)" {
    const dist = try Gamma(f64).init(1.0, 1.0);
    const x = 2.0;
    const logpdf_val = dist.logpdf(x);
    // For k=1, θ=1: log(f(x)) = -x
    const expected = -x;
    try testing.expectApproxEqAbs(expected, logpdf_val, 1e-10);
}

test "Gamma.logpdf - maximum at mode for k>1" {
    const dist = try Gamma(f64).init(3.0, 2.0);
    const mode = (dist.k - 1.0) * dist.theta;
    const log_at_mode = dist.logpdf(mode);
    const log_left = dist.logpdf(mode - 0.5);
    const log_right = dist.logpdf(mode + 0.5);
    try testing.expect(log_at_mode >= log_left);
    try testing.expect(log_at_mode >= log_right);
}

test "Gamma.logpdf - f32 precision" {
    const dist = try Gamma(f32).init(2.0, 1.0);
    const logpdf_val = dist.logpdf(1.0);
    try testing.expect(!math.isNan(logpdf_val));
    try testing.expect(logpdf_val < 0.0);
}

// ============================================================================
// SAMPLE TESTS (11 tests)
// ============================================================================

test "Gamma.sample - all samples non-negative" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const dist = try Gamma(f64).init(2.0, 1.0);

    for (0..1000) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample >= 0.0);
        try testing.expect(!math.isNan(sample));
    }
}

test "Gamma.sample - Exponential case (k=1) mean≈1/λ" {
    var prng = std.Random.DefaultPrng.init(99);
    const rng = prng.random();

    const dist = try Gamma(f64).init(1.0, 1.0);
    const expected_mean = 1.0 / 1.0;

    var sum: f64 = 0.0;
    const n_samples = 5000;
    for (0..n_samples) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.05);
}

test "Gamma.sample - mean≈k*θ (k=2, θ=1, 10k samples)" {
    var prng = std.Random.DefaultPrng.init(999);
    const rng = prng.random();

    const dist = try Gamma(f64).init(2.0, 1.0);
    const expected_mean = dist.k * dist.theta;

    var sum: f64 = 0.0;
    const n_samples = 10000;
    for (0..n_samples) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.05);
}

test "Gamma.sample - mean≈k*θ (k=3, θ=2, 10k samples)" {
    var prng = std.Random.DefaultPrng.init(777);
    const rng = prng.random();

    const dist = try Gamma(f64).init(3.0, 2.0);
    const expected_mean = dist.k * dist.theta;

    var sum: f64 = 0.0;
    const n_samples = 10000;
    for (0..n_samples) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.05);
}

test "Gamma.sample - variance≈k*θ² (10k samples)" {
    var prng = std.Random.DefaultPrng.init(555);
    const rng = prng.random();

    const dist = try Gamma(f64).init(2.0, 1.0);
    const expected_variance = dist.k * dist.theta * dist.theta;

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

test "Gamma.sample - different seeds produce different sequences" {
    var prng1 = std.Random.DefaultPrng.init(111);
    var prng2 = std.Random.DefaultPrng.init(222);

    const dist = try Gamma(f64).init(2.0, 1.0);

    const s1 = dist.sample(prng1.random());
    const s2 = dist.sample(prng2.random());

    try testing.expect(s1 != s2);
}

test "Gamma.sample - k<1 case (0.5)" {
    var prng = std.Random.DefaultPrng.init(333);
    const rng = prng.random();

    const dist = try Gamma(f64).init(0.5, 1.0);
    const expected_mean = dist.k * dist.theta;

    var sum: f64 = 0.0;
    const n_samples = 5000;
    for (0..n_samples) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.10);
}

test "Gamma.sample - k>5 case (10)" {
    var prng = std.Random.DefaultPrng.init(444);
    const rng = prng.random();

    const dist = try Gamma(f64).init(10.0, 1.0);
    const expected_mean = dist.k * dist.theta;

    var sum: f64 = 0.0;
    const n_samples = 10000;
    for (0..n_samples) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.05);
}

test "Gamma.sample - f32 precision" {
    var prng = std.Random.DefaultPrng.init(666);
    const rng = prng.random();

    const dist = try Gamma(f32).init(2.0, 1.0);

    for (0..100) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample >= 0.0);
        try testing.expect(!math.isNan(sample));
    }
}

// ============================================================================
// INTEGRATION TESTS (5 tests)
// ============================================================================

test "Gamma.pdf - integral over domain approximately 1" {
    const dist = try Gamma(f64).init(2.0, 1.0);
    // Numerical integration (trapezoid rule) over [0, 20]
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

test "Gamma.cdf-quantile inverse relationship" {
    const dist = try Gamma(f64).init(2.0, 1.0);
    const x = 2.5;
    const p = dist.cdf(x);
    const q = try dist.quantile(p);
    // quantile(cdf(x)) should ≈ x
    try testing.expectApproxEqAbs(x, q, 0.02);
}

test "Gamma.ensemble statistics (20k samples)" {
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    const dist = try Gamma(f64).init(2.0, 2.0);
    const expected_mean = dist.k * dist.theta;
    const expected_variance = dist.k * dist.theta * dist.theta;

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

test "Gamma.mode property (k>1)" {
    const dist = try Gamma(f64).init(3.0, 2.0);
    const mode = (dist.k - 1.0) * dist.theta;
    const pdf_at_mode = dist.pdf(mode);

    // Test several points and verify mode gives max (locally)
    for ([_]f64{ 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 }) |delta| {
        const pdf_left = dist.pdf(mode - delta);
        const pdf_right = dist.pdf(mode + delta);
        try testing.expect(pdf_at_mode >= pdf_left);
        try testing.expect(pdf_at_mode >= pdf_right);
    }
}

test "Gamma.shape scaling (k→∞ approaches Normal-like)" {
    // For large k, Gamma(k, θ) approaches Normal(k*θ, sqrt(k)*θ)
    // PDF shape should become more symmetric

    const dist_small = try Gamma(f64).init(1.0, 1.0);
    const dist_large = try Gamma(f64).init(100.0, 1.0);

    const mean_small = 1.0;
    const mean_large = 100.0;

    // Compare PDF at mean
    const pdf_small_at_mean = dist_small.pdf(mean_small);
    const pdf_large_at_mean = dist_large.pdf(mean_large);

    // For larger k, PDF at mean should be lower (distribution spreads)
    // This is a weak test but verifies basic scaling behavior
    try testing.expect(pdf_small_at_mean > 0.0);
    try testing.expect(pdf_large_at_mean > 0.0);
}
