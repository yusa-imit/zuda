const std = @import("std");
const math = std.math;

/// Errors for distribution operations
pub const DistributionError = error{
    InvalidParameter,
    InvalidProbability,
    OutOfDomain,
};

// ============================================================================
// Normal (Gaussian) Distribution
// ============================================================================

/// Normal (Gaussian) distribution N(μ, σ²)
///
/// Probability density function (PDF):
///   f(x) = (1 / (σ√(2π))) × exp(-(x-μ)²/(2σ²))
///
/// Cumulative distribution function (CDF):
///   Φ(x) = 0.5 × (1 + erf((x-μ)/(σ√2)))
///
/// Parameters:
///   - mean (μ): Location parameter (-∞ to +∞)
///   - std (σ): Scale parameter (σ > 0)
///
/// Time: O(1) for pdf/cdf/quantile/sample
pub fn Normal(comptime T: type) type {
    return struct {
        mean: T,
        std: T,

        const Self = @This();

        /// Create a normal distribution with given mean and standard deviation
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(mean: T, stddev: T) DistributionError!Self {
            if (stddev <= 0.0) return error.InvalidParameter;
            if (!math.isFinite(mean) or !math.isFinite(stddev)) return error.InvalidParameter;
            return Self{ .mean = mean, .std = stddev };
        }

        /// Probability density function (PDF) at x
        ///
        /// f(x) = (1 / (σ√(2π))) × exp(-(x-μ)²/(2σ²))
        ///
        /// Time: O(1) | Space: O(1)
        pub fn pdf(self: Self, x: T) T {
            const z = (x - self.mean) / self.std;
            const norm_factor = 1.0 / (self.std * @sqrt(2.0 * math.pi));
            return norm_factor * @exp(-0.5 * z * z);
        }

        /// Cumulative distribution function (CDF) at x
        ///
        /// Φ(x) = P(X ≤ x) = 0.5 × (1 + erf((x-μ)/(σ√2)))
        ///
        /// Time: O(1) | Space: O(1)
        pub fn cdf(self: Self, x: T) T {
            const z = (x - self.mean) / (self.std * @sqrt(2.0));
            return 0.5 * (1.0 + erf(z));
        }

        /// Quantile function (inverse CDF) - returns x such that P(X ≤ x) = p
        ///
        /// Uses rational approximation (Beasley-Springer-Moro algorithm)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn quantile(self: Self, p: T) DistributionError!T {
            if (p < 0.0 or p > 1.0) return error.InvalidProbability;
            if (p == 0.0) return -math.inf(T);
            if (p == 1.0) return math.inf(T);

            // Use inverse error function approximation
            const z = erfInv(2.0 * p - 1.0) * @sqrt(2.0);
            return self.mean + self.std * z;
        }

        /// Generate a random sample from this distribution
        ///
        /// Uses Box-Muller transform for standard normal, then scale/shift
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sample(self: Self, rng: std.Random) T {
            // Box-Muller transform
            const uniform1 = rng.float(T);
            const uniform2 = rng.float(T);
            const z = @sqrt(-2.0 * @log(uniform1)) * @cos(2.0 * math.pi * uniform2);
            return self.mean + self.std * z;
        }

        /// Log probability density function (log PDF) at x
        ///
        /// log f(x) = -log(σ√(2π)) - (x-μ)²/(2σ²)
        ///
        /// More numerically stable than log(pdf(x)) for extreme values
        ///
        /// Time: O(1) | Space: O(1)
        pub fn logpdf(self: Self, x: T) T {
            const z = (x - self.mean) / self.std;
            return -@log(self.std * @sqrt(2.0 * math.pi)) - 0.5 * z * z;
        }

        /// Survival function (complementary CDF) - P(X > x)
        ///
        /// S(x) = 1 - Φ(x) = 0.5 × (1 - erf((x-μ)/(σ√2)))
        ///
        /// More accurate than 1 - cdf(x) for extreme values
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sf(self: Self, x: T) T {
            const z = (x - self.mean) / (self.std * @sqrt(2.0));
            return 0.5 * (1.0 - erf(z));
        }

        /// Assert that parameters are valid: std > 0, both finite.
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.std <= 0.0) return DistributionError.InvalidParameter;
            if (!math.isFinite(self.mean) or !math.isFinite(self.std)) return DistributionError.InvalidParameter;
        }
    };
}

// ============================================================================
// Uniform Distribution
// ============================================================================

/// Continuous uniform distribution U(a, b)
///
/// Probability density function (PDF):
///   f(x) = 1/(b-a) for x ∈ [a, b], 0 otherwise
///
/// Cumulative distribution function (CDF):
///   F(x) = (x-a)/(b-a) for x ∈ [a, b]
///
/// Parameters:
///   - a: Lower bound (inclusive)
///   - b: Upper bound (exclusive)
///
/// Time: O(1) for all operations
pub fn Uniform(comptime T: type) type {
    return struct {
        a: T,
        b: T,

        const Self = @This();

        /// Create a uniform distribution on [a, b)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(a: T, b: T) DistributionError!Self {
            if (a >= b) return error.InvalidParameter;
            if (!math.isFinite(a) or !math.isFinite(b)) return error.InvalidParameter;
            return Self{ .a = a, .b = b };
        }

        /// Probability density function (PDF) at x
        ///
        /// Time: O(1) | Space: O(1)
        pub fn pdf(self: Self, x: T) T {
            if (x < self.a or x >= self.b) return 0.0;
            return 1.0 / (self.b - self.a);
        }

        /// Cumulative distribution function (CDF) at x
        ///
        /// Time: O(1) | Space: O(1)
        pub fn cdf(self: Self, x: T) T {
            if (x < self.a) return 0.0;
            if (x >= self.b) return 1.0;
            return (x - self.a) / (self.b - self.a);
        }

        /// Quantile function (inverse CDF)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn quantile(self: Self, p: T) DistributionError!T {
            if (p < 0.0 or p > 1.0) return error.InvalidProbability;
            return self.a + p * (self.b - self.a);
        }

        /// Generate a random sample from this distribution
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sample(self: Self, rng: std.Random) T {
            const u = rng.float(T);
            return self.a + u * (self.b - self.a);
        }

        /// Log probability density function
        ///
        /// Time: O(1) | Space: O(1)
        pub fn logpdf(self: Self, x: T) T {
            if (x < self.a or x >= self.b) return -math.inf(T);
            return -@log(self.b - self.a);
        }

        /// Survival function (1 - CDF)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sf(self: Self, x: T) T {
            if (x < self.a) return 1.0;
            if (x >= self.b) return 0.0;
            return (self.b - x) / (self.b - self.a);
        }

        /// Assert that parameters are valid: a < b, both finite.
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.a >= self.b) return DistributionError.InvalidParameter;
            if (!math.isFinite(self.a) or !math.isFinite(self.b)) return DistributionError.InvalidParameter;
        }
    };
}

// ============================================================================
// Exponential Distribution
// ============================================================================

/// Exponential distribution Exp(λ)
///
/// Probability density function (PDF):
///   f(x) = λ × exp(-λx) for x ≥ 0
///
/// Cumulative distribution function (CDF):
///   F(x) = 1 - exp(-λx) for x ≥ 0
///
/// Parameters:
///   - rate (λ): Rate parameter (λ > 0)
///
/// Time: O(1) for all operations
pub fn Exponential(comptime T: type) type {
    return struct {
        rate: T,

        const Self = @This();

        /// Create an exponential distribution with given rate
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(rate: T) DistributionError!Self {
            if (rate <= 0.0) return error.InvalidParameter;
            if (!math.isFinite(rate)) return error.InvalidParameter;
            return Self{ .rate = rate };
        }

        /// Probability density function (PDF) at x
        ///
        /// Time: O(1) | Space: O(1)
        pub fn pdf(self: Self, x: T) T {
            if (x < 0.0) return 0.0;
            return self.rate * @exp(-self.rate * x);
        }

        /// Cumulative distribution function (CDF) at x
        ///
        /// Time: O(1) | Space: O(1)
        pub fn cdf(self: Self, x: T) T {
            if (x < 0.0) return 0.0;
            return 1.0 - @exp(-self.rate * x);
        }

        /// Quantile function (inverse CDF)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn quantile(self: Self, p: T) DistributionError!T {
            if (p < 0.0 or p > 1.0) return error.InvalidProbability;
            if (p == 0.0) return 0.0;
            if (p == 1.0) return math.inf(T);
            return -@log(1.0 - p) / self.rate;
        }

        /// Generate a random sample from this distribution
        ///
        /// Uses inverse transform method
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sample(self: Self, rng: std.Random) T {
            const u = rng.float(T);
            return -@log(u) / self.rate;
        }

        /// Log probability density function
        ///
        /// Time: O(1) | Space: O(1)
        pub fn logpdf(self: Self, x: T) T {
            if (x < 0.0) return -math.inf(T);
            return @log(self.rate) - self.rate * x;
        }

        /// Survival function (1 - CDF)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sf(self: Self, x: T) T {
            if (x < 0.0) return 1.0;
            return @exp(-self.rate * x);
        }

        /// Mean of the distribution (1/λ)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mean(self: Self) T {
            return 1.0 / self.rate;
        }

        /// Variance of the distribution (1/λ²)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn variance(self: Self) T {
            return 1.0 / (self.rate * self.rate);
        }

        /// Assert that parameters are valid: rate > 0 and finite.
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.rate <= 0.0 or !math.isFinite(self.rate)) return DistributionError.InvalidParameter;
        }
    };
}

// ============================================================================
// Gamma Distribution
// ============================================================================

/// Gamma distribution Gamma(α, β)
///
/// A continuous probability distribution commonly used in Bayesian statistics,
/// queuing theory, and as a conjugate prior for precision parameters.
///
/// Probability density function (PDF):
///   f(x) = (β^α / Γ(α)) × x^(α-1) × e^(-βx) for x > 0
///
/// Parameters:
///   - shape (α): Shape parameter (α > 0)
///   - rate (β): Rate parameter (β > 0)
///
/// Special cases:
///   - Exponential(λ) = Gamma(1, λ)
///   - Chi-squared(k) = Gamma(k/2, 1/2)
///
/// Time: O(1) for pdf/cdf/sample (with numerical approximations)
pub fn Gamma(comptime T: type) type {
    return struct {
        shape: T, // α
        rate: T, // β

        const Self = @This();

        /// Create a gamma distribution with given shape and rate parameters
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(shape: T, rate: T) DistributionError!Self {
            if (shape <= 0.0 or rate <= 0.0) return error.InvalidParameter;
            if (!math.isFinite(shape) or !math.isFinite(rate)) return error.InvalidParameter;
            return Self{ .shape = shape, .rate = rate };
        }

        /// Probability density function (PDF) at x
        ///
        /// f(x) = (β^α / Γ(α)) × x^(α-1) × e^(-βx)
        ///
        /// Uses log-space computation for numerical stability
        ///
        /// Time: O(1) | Space: O(1)
        pub fn pdf(self: Self, x: T) T {
            if (x < 0.0) return 0.0;
            if (x == 0.0) {
                // x=0 edge: only Exponential (shape=1) has non-zero finite density at 0
                if (self.shape > 1.0) return 0.0;
                if (self.shape == 1.0) return @exp(self.shape * @log(self.rate) - logGamma(self.shape));
                return math.inf(T);
            }

            // log f(x) = α×log(β) - log(Γ(α)) + (α-1)×log(x) - β×x
            const log_pdf = self.shape * @log(self.rate) - logGamma(self.shape) + (self.shape - 1.0) * @log(x) - self.rate * x;
            return @exp(log_pdf);
        }

        /// Cumulative distribution function (CDF) at x
        ///
        /// P(X ≤ x) = γ(α, βx) / Γ(α)
        ///
        /// Uses regularized lower incomplete gamma function
        ///
        /// Time: O(1) with approximation | Space: O(1)
        pub fn cdf(self: Self, x: T) T {
            if (x <= 0.0) return 0.0;
            // Regularized lower incomplete gamma: P(α, βx)
            return regularizedGammaP(self.shape, self.rate * x);
        }

        /// Quantile function (inverse CDF) - returns x such that P(X ≤ x) = p
        ///
        /// Uses bisection search on CDF
        ///
        /// Time: O(log(1/ε)) for tolerance ε | Space: O(1)
        pub fn quantile(self: Self, p: T) DistributionError!T {
            if (p < 0.0 or p > 1.0) return error.InvalidProbability;
            if (p == 0.0) return 0.0;
            if (p == 1.0) return math.inf(T);

            // Use bisection on CDF
            // Start with mean as initial guess, then bracket
            var low: T = 0.0;
            var high: T = self.mean() * 10.0; // Upper bound well above mean

            // Ensure high bracket is large enough
            while (self.cdf(high) < p) {
                high *= 2.0;
            }

            const tolerance = 1e-10;
            const max_iter = 100;
            var iter: usize = 0;

            while (iter < max_iter) : (iter += 1) {
                const mid = (low + high) / 2.0;
                const cdf_mid = self.cdf(mid);

                if (@abs(cdf_mid - p) < tolerance) {
                    return mid;
                }

                if (cdf_mid < p) {
                    low = mid;
                } else {
                    high = mid;
                }

                if (high - low < tolerance) {
                    return (low + high) / 2.0;
                }
            }

            return (low + high) / 2.0;
        }

        /// Generate a random sample from this distribution
        ///
        /// Uses Marsaglia & Tsang's method for shape ≥ 1
        /// Uses Ahrens-Dieter acceptance-rejection for shape < 1
        ///
        /// Time: O(1) expected | Space: O(1)
        pub fn sample(self: Self, rng: std.Random) T {
            if (self.shape >= 1.0) {
                // Marsaglia & Tsang (2000) method for α ≥ 1
                const d = self.shape - 1.0 / 3.0;
                const c = 1.0 / @sqrt(9.0 * d);

                while (true) {
                    var x: T = undefined;
                    var v: T = undefined;

                    // Generate x from N(0,1) and v = (1 + cx)³
                    while (true) {
                        const uniform1 = rng.float(T);
                        const uniform2 = rng.float(T);
                        x = @sqrt(-2.0 * @log(uniform1)) * @cos(2.0 * math.pi * uniform2); // Box-Muller
                        v = 1.0 + c * x;
                        if (v > 0.0) break;
                    }

                    v = v * v * v;
                    const u = rng.float(T);

                    // Squeeze acceptance
                    if (u < 1.0 - 0.0331 * x * x * x * x) {
                        return d * v / self.rate;
                    }

                    // Full acceptance test
                    if (@log(u) < 0.5 * x * x + d * (1.0 - v + @log(v))) {
                        return d * v / self.rate;
                    }
                }
            } else {
                // Ahrens-Dieter acceptance-rejection for α < 1
                // Generate Gamma(α+1, β) then multiply by U^(1/α)
                const e = math.e;
                const alpha_plus_1 = self.shape + 1.0;

                while (true) {
                    const uniform1 = rng.float(T);
                    const uniform2 = rng.float(T);
                    const uniform3 = rng.float(T);

                    if (uniform1 <= e / (e + alpha_plus_1)) {
                        const xi = std.math.pow(T, (e + alpha_plus_1) * uniform2 / e, 1.0 / alpha_plus_1);
                        if (uniform3 <= @exp(-xi)) {
                            const gamma_sample = xi * std.math.pow(T, uniform1, 1.0 / self.shape);
                            return gamma_sample / self.rate;
                        }
                    } else {
                        const xi = -@log((e + alpha_plus_1) * (1.0 - uniform2) / (alpha_plus_1 * e));
                        if (uniform3 <= std.math.pow(T, xi, alpha_plus_1 - 1.0)) {
                            const gamma_sample = xi * std.math.pow(T, uniform1, 1.0 / self.shape);
                            return gamma_sample / self.rate;
                        }
                    }
                }
            }
        }

        /// Log probability density function at x
        ///
        /// log f(x) = α×log(β) - log(Γ(α)) + (α-1)×log(x) - β×x
        ///
        /// Time: O(1) | Space: O(1)
        pub fn logpdf(self: Self, x: T) T {
            if (x <= 0.0) return -math.inf(T);
            return self.shape * @log(self.rate) - logGamma(self.shape) + (self.shape - 1.0) * @log(x) - self.rate * x;
        }

        /// Mean of the distribution
        ///
        /// E[X] = α / β
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mean(self: Self) T {
            return self.shape / self.rate;
        }

        /// Variance of the distribution
        ///
        /// Var(X) = α / β²
        ///
        /// Time: O(1) | Space: O(1)
        pub fn variance(self: Self) T {
            return self.shape / (self.rate * self.rate);
        }

        /// Assert that parameters are valid: shape > 0, rate > 0, both finite.
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.shape <= 0.0 or !math.isFinite(self.shape)) return DistributionError.InvalidParameter;
            if (self.rate <= 0.0 or !math.isFinite(self.rate)) return DistributionError.InvalidParameter;
        }
    };
}

// ============================================================================
// Beta Distribution
// ============================================================================

/// Beta distribution Beta(α, β)
///
/// Probability density function (PDF):
///   f(x) = (x^(α-1) × (1-x)^(β-1)) / B(α,β)
///   where B(α,β) = Γ(α)Γ(β)/Γ(α+β) is the beta function
///
/// Cumulative distribution function (CDF):
///   F(x) = I_x(α, β) (regularized incomplete beta function)
///
/// Parameters:
///   - alpha (α): Shape parameter (α > 0)
///   - beta (β): Shape parameter (β > 0)
///
/// Domain: x ∈ [0, 1]
///
/// Time: O(1) for all operations
pub fn Beta(comptime T: type) type {
    return struct {
        alpha: T,
        beta: T,

        const Self = @This();

        /// Create a beta distribution with given shape parameters
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(alpha: T, beta_param: T) DistributionError!Self {
            if (alpha <= 0.0 or beta_param <= 0.0) return error.InvalidParameter;
            if (!math.isFinite(alpha) or !math.isFinite(beta_param)) return error.InvalidParameter;
            return Self{ .alpha = alpha, .beta = beta_param };
        }

        /// Probability density function (PDF) at x
        ///
        /// f(x) = (x^(α-1) × (1-x)^(β-1)) / B(α,β)
        ///
        /// Uses log-space computation for numerical stability
        ///
        /// Time: O(1) | Space: O(1)
        pub fn pdf(self: Self, x: T) T {
            if (x < 0.0 or x > 1.0) return 0.0;
            if (x == 0.0) return if (self.alpha < 1.0) math.inf(T) else if (self.alpha == 1.0) self.beta else 0.0;
            if (x == 1.0) return if (self.beta < 1.0) math.inf(T) else if (self.beta == 1.0) self.alpha else 0.0;

            // f(x) = exp(log f(x)) = exp((α-1)log(x) + (β-1)log(1-x) - log B(α,β))
            const log_pdf = (self.alpha - 1.0) * @log(x) +
                (self.beta - 1.0) * @log(1.0 - x) -
                logBeta(self.alpha, self.beta);
            return @exp(log_pdf);
        }

        /// Cumulative distribution function (CDF) at x
        ///
        /// F(x) = I_x(α, β) (regularized incomplete beta function)
        ///
        /// Time: O(1) with finite iterations | Space: O(1)
        pub fn cdf(self: Self, x: T) T {
            if (x <= 0.0) return 0.0;
            if (x >= 1.0) return 1.0;
            return regularizedBetaI(self.alpha, self.beta, x);
        }

        /// Quantile function (inverse CDF)
        ///
        /// Uses bisection search on the CDF
        ///
        /// Time: O(log(1/ε)) where ε is tolerance | Space: O(1)
        pub fn quantile(self: Self, p: T) DistributionError!T {
            if (p < 0.0 or p > 1.0) return error.InvalidProbability;
            if (p == 0.0) return 0.0;
            if (p == 1.0) return 1.0;

            // Bisection search on [0, 1]
            const tolerance = 1e-10;
            var left: T = 0.0;
            var right: T = 1.0;
            var mid: T = 0.5;

            for (0..100) |_| {
                mid = (left + right) / 2.0;
                const cdf_mid = self.cdf(mid);

                if (@abs(cdf_mid - p) < tolerance) break;

                if (cdf_mid < p) {
                    left = mid;
                } else {
                    right = mid;
                }
            }

            return mid;
        }

        /// Generate a random sample from this distribution
        ///
        /// Uses rejection sampling with Gamma variates: if X ~ Gamma(α,1), Y ~ Gamma(β,1), then X/(X+Y) ~ Beta(α,β)
        ///
        /// Time: O(1) expected | Space: O(1)
        pub fn sample(self: Self, rng: std.Random) T {
            // Generate two gamma variates
            const gamma_alpha = Gamma(T){ .shape = self.alpha, .rate = 1.0 };
            const gamma_beta = Gamma(T){ .shape = self.beta, .rate = 1.0 };

            const x = gamma_alpha.sample(rng);
            const y = gamma_beta.sample(rng);

            // Return X/(X+Y)
            return x / (x + y);
        }

        /// Log probability density function
        ///
        /// log f(x) = (α-1)log(x) + (β-1)log(1-x) - log B(α,β)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn logpdf(self: Self, x: T) T {
            if (x <= 0.0 or x >= 1.0) return -math.inf(T);

            return (self.alpha - 1.0) * @log(x) +
                (self.beta - 1.0) * @log(1.0 - x) -
                logBeta(self.alpha, self.beta);
        }

        /// Mean of the distribution (α / (α+β))
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mean(self: Self) T {
            return self.alpha / (self.alpha + self.beta);
        }

        /// Variance of the distribution (αβ / ((α+β)²(α+β+1)))
        ///
        /// Time: O(1) | Space: O(1)
        pub fn variance(self: Self) T {
            const sum = self.alpha + self.beta;
            return (self.alpha * self.beta) / (sum * sum * (sum + 1.0));
        }

        /// Assert that parameters are valid: alpha > 0, beta > 0, both finite.
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.alpha <= 0.0 or !math.isFinite(self.alpha)) return DistributionError.InvalidParameter;
            if (self.beta <= 0.0 or !math.isFinite(self.beta)) return DistributionError.InvalidParameter;
        }
    };
}

// ============================================================================
// Chi-Squared Distribution
// ============================================================================

/// Chi-squared distribution χ²(k)
///
/// A special case of Gamma distribution: χ²(k) = Gamma(k/2, 1/2)
///
/// Probability density function (PDF):
///   f(x) = (1/(2^(k/2) × Γ(k/2))) × x^(k/2 - 1) × e^(-x/2)  for x ≥ 0
///
/// Cumulative distribution function (CDF):
///   F(x) = P(a, x/2) / Γ(a)  where a = k/2, P is lower incomplete gamma
///
/// Parameters:
///   - k (degrees of freedom): Must be positive integer
///
/// Use cases:
///   - Goodness-of-fit tests
///   - Independence tests in contingency tables
///   - Variance estimation in normal populations
///   - Sum of squared standard normals: if Z_i ~ N(0,1), then Σ Z_i² ~ χ²(k)
///
/// Time: O(1) for all operations
pub fn ChiSquared(comptime T: type) type {
    return struct {
        k: T, // degrees of freedom (stored as float for Gamma compatibility)
        gamma_dist: Gamma(T),

        const Self = @This();

        /// Create a chi-squared distribution with k degrees of freedom
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(k: u64) DistributionError!Self {
            if (k == 0) return error.InvalidParameter;

            const k_f = @as(T, @floatFromInt(k));

            // χ²(k) = Gamma(k/2, 1/2)
            const alpha = k_f / 2.0; // shape
            const beta = 0.5; // rate

            const gamma_dist = try Gamma(T).init(alpha, beta);

            return Self{
                .k = k_f,
                .gamma_dist = gamma_dist,
            };
        }

        /// Probability density function (PDF) at x
        ///
        /// f(x) = (1/(2^(k/2) × Γ(k/2))) × x^(k/2 - 1) × e^(-x/2)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn pdf(self: Self, x: T) T {
            if (x < 0.0) return 0.0;
            return self.gamma_dist.pdf(x);
        }

        /// Cumulative distribution function (CDF) at x
        ///
        /// F(x) = P(X ≤ x)
        ///
        /// Uses regularized lower incomplete gamma function
        ///
        /// Time: O(1) | Space: O(1)
        pub fn cdf(self: Self, x: T) T {
            if (x <= 0.0) return 0.0;
            return self.gamma_dist.cdf(x);
        }

        /// Quantile function (inverse CDF) - returns x such that P(X ≤ x) = p
        ///
        /// Uses bisection search on CDF
        ///
        /// Time: O(1) | Space: O(1)
        pub fn quantile(self: Self, p: T) DistributionError!T {
            return try self.gamma_dist.quantile(p);
        }

        /// Generate a random sample from this distribution
        ///
        /// χ²(k) is the sum of k independent squared standard normals
        /// Implemented via Gamma(k/2, 1/2) sampling
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sample(self: Self, rng: std.Random) T {
            return self.gamma_dist.sample(rng);
        }

        /// Log probability density function (log PDF) at x
        ///
        /// More numerically stable than log(pdf(x)) for extreme values
        ///
        /// Time: O(1) | Space: O(1)
        pub fn logpdf(self: Self, x: T) T {
            if (x < 0.0) return -math.inf(T);
            return self.gamma_dist.logpdf(x);
        }

        /// Mean of the distribution
        ///
        /// E[X] = k (degrees of freedom)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mean(self: Self) T {
            return self.k;
        }

        /// Variance of the distribution
        ///
        /// Var(X) = 2k
        ///
        /// Time: O(1) | Space: O(1)
        pub fn variance(self: Self) T {
            return 2.0 * self.k;
        }

        /// Assert that parameters are valid: k > 0 and finite.
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.k <= 0.0 or !math.isFinite(self.k)) return DistributionError.InvalidParameter;
        }
    };
}

// ============================================================================
// Student's t-Distribution
// ============================================================================

/// Student's t-distribution t(ν)
///
/// Probability density function (PDF):
///   f(x) = Γ((ν+1)/2) / (√(νπ) × Γ(ν/2)) × (1 + x²/ν)^(-(ν+1)/2)
///
/// Cumulative distribution function (CDF):
///   F(x) = 0.5 + x × Γ((ν+1)/2) × ₂F₁(0.5, (ν+1)/2; 1.5; -x²/ν) / (√(νπ) × Γ(ν/2))
///   Computed via regularized incomplete beta function I_z((ν/2), 0.5) with z transformation
///
/// Parameters:
///   - nu (ν): Degrees of freedom (ν > 0)
///
/// Properties:
///   - Symmetric around 0
///   - Heavier tails than normal distribution (more probability in extremes)
///   - As ν → ∞, approaches standard normal N(0,1)
///   - Mean = 0 (for ν > 1), Variance = ν/(ν-2) (for ν > 2)
///
/// Use cases:
///   - Hypothesis testing with unknown variance and small samples
///   - Confidence intervals for population mean
///   - Statistical inference when population variance is unknown
///   - Robust statistical methods (resistant to outliers)
///
/// Time: O(1) for all operations
pub fn StudentT(comptime T: type) type {
    return struct {
        nu: T, // degrees of freedom

        const Self = @This();

        /// Create a Student's t-distribution with ν degrees of freedom
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(nu: T) DistributionError!Self {
            if (nu <= 0.0 or !math.isFinite(nu)) return error.InvalidParameter;
            return Self{ .nu = nu };
        }

        /// Probability density function (PDF) at x
        ///
        /// f(x) = Γ((ν+1)/2) / (√(νπ) × Γ(ν/2)) × (1 + x²/ν)^(-(ν+1)/2)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn pdf(self: Self, x: T) T {
            const nu_half = self.nu / 2.0;
            const log_numerator = logGamma((self.nu + 1.0) / 2.0);
            const log_denominator = 0.5 * @log(self.nu * math.pi) + logGamma(nu_half);
            const log_power = -(self.nu + 1.0) / 2.0 * @log(1.0 + (x * x) / self.nu);

            return @exp(log_numerator - log_denominator + log_power);
        }

        /// Cumulative distribution function (CDF) at x
        ///
        /// F(x) = P(X ≤ x)
        ///
        /// Uses regularized incomplete beta function with transformation:
        /// z = (x + √(x² + ν)) / (2√(x² + ν))
        /// F(x) = I_z(ν/2, ν/2) for x ≥ 0
        ///
        /// Alternative formula (used here):
        /// F(x) = 0.5 + 0.5 × sign(x) × I_z(0.5, ν/2) where z = ν/(ν + x²)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn cdf(self: Self, x: T) T {
            if (x == 0.0) return 0.5; // Symmetric around 0

            // Transform to beta CDF: z = ν/(ν + x²)
            const z = self.nu / (self.nu + x * x);

            // I_z(ν/2, 0.5) using regularized incomplete beta
            const beta_cdf = regularizedBetaI(self.nu / 2.0, 0.5, z);

            // Adjust sign based on x
            if (x > 0.0) {
                return 1.0 - 0.5 * beta_cdf;
            } else {
                return 0.5 * beta_cdf;
            }
        }

        /// Quantile function (inverse CDF) - returns x such that P(X ≤ x) = p
        ///
        /// Uses bisection search on CDF
        ///
        /// Time: O(1) | Space: O(1)
        pub fn quantile(self: Self, p: T) DistributionError!T {
            if (p < 0.0 or p > 1.0) return error.InvalidProbability;
            if (p == 0.0) return -math.inf(T);
            if (p == 1.0) return math.inf(T);
            if (p == 0.5) return 0.0; // Symmetric around 0

            // Bisection search for quantile
            var left: T = if (p < 0.5) -100.0 else 0.0;
            var right: T = if (p < 0.5) 0.0 else 100.0;
            const tolerance = 1e-10;

            // Expand search bounds if needed
            while (self.cdf(left) > p) left *= 2.0;
            while (self.cdf(right) < p) right *= 2.0;

            // Bisection
            var iterations: u32 = 0;
            while (right - left > tolerance and iterations < 100) : (iterations += 1) {
                const mid = (left + right) / 2.0;
                const cdf_mid = self.cdf(mid);

                if (cdf_mid < p) {
                    left = mid;
                } else {
                    right = mid;
                }
            }

            return (left + right) / 2.0;
        }

        /// Generate a random sample from this distribution
        ///
        /// Uses the relationship: T = Z / √(V/ν) where Z ~ N(0,1) and V ~ χ²(ν)
        /// This is the definition of the t-distribution
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sample(self: Self, rng: std.Random) T {
            // Generate standard normal Z ~ N(0,1)
            const uniform1 = rng.float(T);
            const uniform2 = rng.float(T);
            const z = @sqrt(-2.0 * @log(uniform1)) * @cos(2.0 * math.pi * uniform2);

            // Generate chi-squared random variable V ~ χ²(ν)
            // χ²(ν) = Gamma(ν/2, 1/2)
            const chi_squared_dist = ChiSquared(T).init(@intFromFloat(self.nu)) catch unreachable;
            const v = chi_squared_dist.sample(rng);

            // T = Z / √(V/ν)
            return z / @sqrt(v / self.nu);
        }

        /// Log probability density function (log PDF) at x
        ///
        /// More numerically stable than log(pdf(x)) for extreme values
        ///
        /// Time: O(1) | Space: O(1)
        pub fn logpdf(self: Self, x: T) T {
            const nu_half = self.nu / 2.0;
            const log_numerator = logGamma((self.nu + 1.0) / 2.0);
            const log_denominator = 0.5 * @log(self.nu * math.pi) + logGamma(nu_half);
            const log_power = -(self.nu + 1.0) / 2.0 * @log(1.0 + (x * x) / self.nu);

            return log_numerator - log_denominator + log_power;
        }

        /// Mean of the distribution
        ///
        /// E[X] = 0 (for ν > 1)
        /// Undefined for ν ≤ 1
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mean(self: Self) T {
            if (self.nu <= 1.0) return math.nan(T);
            return 0.0; // Symmetric around 0
        }

        /// Variance of the distribution
        ///
        /// Var(X) = ν/(ν-2) (for ν > 2)
        /// Infinite for 1 < ν ≤ 2
        /// Undefined for ν ≤ 1
        ///
        /// Time: O(1) | Space: O(1)
        pub fn variance(self: Self) T {
            if (self.nu <= 1.0) return math.nan(T);
            if (self.nu <= 2.0) return math.inf(T);
            return self.nu / (self.nu - 2.0);
        }

        /// Assert that parameters are valid: nu > 0 and finite.
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.nu <= 0.0 or !math.isFinite(self.nu)) return DistributionError.InvalidParameter;
        }
    };
}

// ============================================================================
// F Distribution (Fisher-Snedecor Distribution)
// ============================================================================

/// F distribution F(d₁, d₂)
///
/// The F-distribution is the ratio of two chi-squared distributions divided by their degrees of freedom.
/// If V₁ ~ χ²(d₁) and V₂ ~ χ²(d₂), then F = (V₁/d₁)/(V₂/d₂) ~ F(d₁, d₂)
///
/// Probability density function (PDF):
///   f(x) = √((d₁x)^d₁ × d₂^d₂ / (d₁x + d₂)^(d₁+d₂)) / (x × B(d₁/2, d₂/2))
///   where B is the beta function
///
/// Cumulative distribution function (CDF):
///   F(x) = I_z(d₁/2, d₂/2) where z = d₁x/(d₁x + d₂)
///   I is the regularized incomplete beta function
///
/// Parameters:
///   - d1: Numerator degrees of freedom (d₁ > 0)
///   - d2: Denominator degrees of freedom (d₂ > 0)
///
/// Properties:
///   - Domain: x ∈ [0, ∞)
///   - Mean: d₂/(d₂-2) for d₂ > 2, undefined otherwise
///   - Variance: 2d₂²(d₁+d₂-2)/(d₁(d₂-2)²(d₂-4)) for d₂ > 4, undefined otherwise
///
/// Use cases:
///   - ANOVA F-tests (comparing variances between groups)
///   - Regression F-tests (overall model significance)
///   - Variance ratio tests
///   - Model comparison in hypothesis testing
///
/// Time: O(1) for all operations
pub fn FDistribution(comptime T: type) type {
    return struct {
        d1: T,
        d2: T,

        const Self = @This();

        /// Create an F distribution with given degrees of freedom
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(d1: T, d2: T) DistributionError!Self {
            if (d1 <= 0.0 or d2 <= 0.0) return error.InvalidParameter;
            if (!math.isFinite(d1) or !math.isFinite(d2)) return error.InvalidParameter;
            return Self{ .d1 = d1, .d2 = d2 };
        }

        /// Probability density function (PDF) at x
        ///
        /// f(x) = √((d₁x)^d₁ × d₂^d₂ / (d₁x + d₂)^(d₁+d₂)) / (x × B(d₁/2, d₂/2))
        ///
        /// Computed in log-space for numerical stability:
        /// log f(x) = 0.5×d₁×log(d₁) + 0.5×d₂×log(d₂) + (d₁/2-1)×log(x)
        ///          - 0.5×(d₁+d₂)×log(d₁x + d₂) - logBeta(d₁/2, d₂/2)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn pdf(self: Self, x: T) T {
            if (x <= 0.0) return 0.0;
            return @exp(self.logpdf(x));
        }

        /// Cumulative distribution function (CDF) at x
        ///
        /// F(x) = I_z(d₁/2, d₂/2) where z = d₁x/(d₁x + d₂)
        ///
        /// Uses regularized incomplete beta function
        ///
        /// Time: O(1) | Space: O(1)
        pub fn cdf(self: Self, x: T) T {
            if (x <= 0.0) return 0.0;

            // Transform to beta distribution variable
            const z = (self.d1 * x) / (self.d1 * x + self.d2);
            return regularizedBetaI(self.d1 / 2.0, self.d2 / 2.0, z);
        }

        /// Quantile function (inverse CDF) - returns x such that P(X ≤ x) = p
        ///
        /// Uses bisection search on the CDF
        ///
        /// Time: O(1) | Space: O(1)
        pub fn quantile(self: Self, p: T) DistributionError!T {
            if (p < 0.0 or p > 1.0) return error.InvalidProbability;
            if (p == 0.0) return 0.0;
            if (p == 1.0) return math.inf(T);

            // Bisection search with adaptive bounds
            var low: T = 0.0;
            var high: T = 10.0;

            // Expand upper bound if needed
            while (self.cdf(high) < p) {
                high *= 2.0;
            }

            // Bisection
            const tol: T = 1e-10;
            const max_iter = 100;
            var iter: usize = 0;

            while (high - low > tol and iter < max_iter) : (iter += 1) {
                const mid = (low + high) / 2.0;
                const cdf_mid = self.cdf(mid);
                if (cdf_mid < p) {
                    low = mid;
                } else {
                    high = mid;
                }
            }

            return (low + high) / 2.0;
        }

        /// Generate a random sample from this distribution
        ///
        /// Uses the definition: F = (V₁/d₁)/(V₂/d₂) where V₁~χ²(d₁), V₂~χ²(d₂)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sample(self: Self, rng: std.Random) T {
            const chi1 = ChiSquared(T).init(@intFromFloat(self.d1)) catch unreachable;
            const chi2 = ChiSquared(T).init(@intFromFloat(self.d2)) catch unreachable;

            const v1 = chi1.sample(rng);
            const v2 = chi2.sample(rng);

            return (v1 / self.d1) / (v2 / self.d2);
        }

        /// Log probability density function (log PDF) at x
        ///
        /// log f(x) = 0.5×d₁×log(d₁) + 0.5×d₂×log(d₂) + (d₁/2-1)×log(x)
        ///          - 0.5×(d₁+d₂)×log(d₁x + d₂) - logBeta(d₁/2, d₂/2)
        ///
        /// More numerically stable than log(pdf(x)) for extreme values
        ///
        /// Time: O(1) | Space: O(1)
        pub fn logpdf(self: Self, x: T) T {
            if (x <= 0.0) return -math.inf(T);

            const half_d1 = self.d1 / 2.0;
            const half_d2 = self.d2 / 2.0;
            const d1x = self.d1 * x;

            const log_numerator = half_d1 * @log(self.d1) + half_d2 * @log(self.d2) + (half_d1 - 1.0) * @log(x);
            const log_denominator = (half_d1 + half_d2) * @log(d1x + self.d2);
            const log_beta_term = logBeta(half_d1, half_d2);

            return log_numerator - log_denominator - log_beta_term;
        }

        /// Mean of the distribution
        ///
        /// E[X] = d₂/(d₂-2) for d₂ > 2, undefined otherwise
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mean(self: Self) T {
            if (self.d2 <= 2.0) {
                return math.nan(T);
            }
            return self.d2 / (self.d2 - 2.0);
        }

        /// Variance of the distribution
        ///
        /// Var[X] = 2d₂²(d₁+d₂-2)/(d₁(d₂-2)²(d₂-4)) for d₂ > 4
        /// Infinite for 2 < d₂ ≤ 4
        /// Undefined for d₂ ≤ 2
        ///
        /// Time: O(1) | Space: O(1)
        pub fn variance(self: Self) T {
            if (self.d2 <= 2.0) {
                return math.nan(T);
            }
            if (self.d2 <= 4.0) {
                return math.inf(T);
            }

            const numerator = 2.0 * self.d2 * self.d2 * (self.d1 + self.d2 - 2.0);
            const denominator = self.d1 * (self.d2 - 2.0) * (self.d2 - 2.0) * (self.d2 - 4.0);
            return numerator / denominator;
        }

        /// Assert that parameters are valid: d1 > 0, d2 > 0, both finite.
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.d1 <= 0.0 or !math.isFinite(self.d1)) return DistributionError.InvalidParameter;
            if (self.d2 <= 0.0 or !math.isFinite(self.d2)) return DistributionError.InvalidParameter;
        }
    };
}

// ============================================================================
// Bernoulli Distribution
// ============================================================================

/// Bernoulli distribution Bernoulli(p)
///
/// Models a single binary trial with success probability p.
/// Support: {0, 1}
///
/// Parameters:
///   p: probability of success (0 < p ≤ 1)
///
/// Time: O(1) for all operations | Space: O(1)
pub fn Bernoulli(comptime T: type) type {
    return struct {
        p: T,

        const Self = @This();

        /// Create a Bernoulli distribution with given success probability
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(p: T) DistributionError!Self {
            if (p <= 0.0 or p > 1.0) return error.InvalidProbability;
            if (!math.isFinite(p)) return error.InvalidProbability;
            return Self{ .p = p };
        }

        /// Probability mass function (PMF) at k
        ///
        /// P(X = k) = p if k=1, (1-p) if k=0, 0 otherwise
        ///
        /// Time: O(1) | Space: O(1)
        pub fn pmf(self: Self, k: u64) T {
            if (k == 0) return 1.0 - self.p;
            if (k == 1) return self.p;
            return 0.0;
        }

        /// Cumulative distribution function (CDF) at k
        ///
        /// P(X ≤ k) = 0 if k<0, (1-p) if k=0, 1.0 if k≥1
        ///
        /// Time: O(1) | Space: O(1)
        pub fn cdf(self: Self, k: i64) T {
            if (k < 0) return 0.0;
            if (k == 0) return 1.0 - self.p;
            return 1.0;
        }

        /// Log probability mass function
        ///
        /// log P(X = k) = log(p) if k=1, log(1-p) if k=0, -inf otherwise
        ///
        /// Time: O(1) | Space: O(1)
        pub fn logpmf(self: Self, k: u64) T {
            if (k == 0) return @log(1.0 - self.p);
            if (k == 1) return @log(self.p);
            return -math.inf(T);
        }

        /// Survival function: P(X > k)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sf(self: Self, k: i64) T {
            return 1.0 - self.cdf(k);
        }

        /// Mean of the distribution (p)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mean(self: Self) T {
            return self.p;
        }

        /// Variance of the distribution (p × (1-p))
        ///
        /// Time: O(1) | Space: O(1)
        pub fn variance(self: Self) T {
            return self.p * (1.0 - self.p);
        }

        /// Generate a random sample from this distribution
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sample(self: Self, rng: std.Random) u64 {
            return if (rng.float(T) < self.p) @as(u64, 1) else @as(u64, 0);
        }

        /// Validate internal invariants: 0 < p ≤ 1 and p is finite
        ///
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.p <= 0.0 or self.p > 1.0) return error.InvalidProbability;
            if (!math.isFinite(self.p)) return error.InvalidProbability;
        }
    };
}

// ============================================================================
// Geometric Distribution
// ============================================================================

/// Geometric distribution Geom(p)
///
/// Models the number of trials until first success.
/// Support: {1, 2, 3, ...}
///
/// Parameters:
///   p: probability of success (0 < p ≤ 1)
///
/// Time: O(1) for most operations, O(log k) for pmf/logpmf | Space: O(1)
pub fn Geometric(comptime T: type) type {
    return struct {
        p: T,

        const Self = @This();

        /// Create a Geometric distribution with given success probability
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(p: T) DistributionError!Self {
            if (p <= 0.0 or p > 1.0) return error.InvalidProbability;
            if (!math.isFinite(p)) return error.InvalidProbability;
            return Self{ .p = p };
        }

        /// Probability mass function (PMF) at k
        ///
        /// P(X = k) = (1-p)^(k-1) × p for k≥1, 0 for k=0
        ///
        /// Time: O(1) | Space: O(1)
        pub fn pmf(self: Self, k: u64) T {
            if (k == 0) return 0.0;
            const k_f = @as(T, @floatFromInt(k - 1));
            return math.pow(T, 1.0 - self.p, k_f) * self.p;
        }

        /// Cumulative distribution function (CDF) at k
        ///
        /// P(X ≤ k) = 1 - (1-p)^k for k≥1, 0 for k≤0
        ///
        /// Time: O(1) | Space: O(1)
        pub fn cdf(self: Self, k: i64) T {
            if (k <= 0) return 0.0;
            const k_f = @as(T, @floatFromInt(k));
            return 1.0 - math.pow(T, 1.0 - self.p, k_f);
        }

        /// Quantile function (inverse CDF)
        ///
        /// Returns k such that P(X ≤ k) ≥ prob
        /// Formula: k = ceil(log(1-prob) / log(1-p))
        ///
        /// Time: O(1) | Space: O(1)
        pub fn quantile(self: Self, prob: T) DistributionError!u64 {
            if (prob <= 0.0 or prob >= 1.0) return error.InvalidProbability;

            // k = ceil(log(1-prob) / log(1-p))
            const numerator = @log(1.0 - prob);
            const denominator = @log(1.0 - self.p);
            const result_f = numerator / denominator;
            const result = @ceil(result_f);

            return @intFromFloat(result);
        }

        /// Log probability mass function
        ///
        /// log P(X = k) = (k-1) × log(1-p) + log(p), return -inf for k=0
        ///
        /// Time: O(1) | Space: O(1)
        pub fn logpmf(self: Self, k: u64) T {
            if (k == 0) return -math.inf(T);
            const k_f = @as(T, @floatFromInt(k - 1));
            return k_f * @log(1.0 - self.p) + @log(self.p);
        }

        /// Survival function: P(X > k)
        ///
        /// sf(k) = (1-p)^k for k≥1, else 1.0
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sf(self: Self, k: i64) T {
            if (k <= 0) return 1.0;
            const k_f = @as(T, @floatFromInt(k));
            return math.pow(T, 1.0 - self.p, k_f);
        }

        /// Mode of the distribution (always 1)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mode(self: Self) u64 {
            _ = self;
            return 1;
        }

        /// Mean of the distribution (1/p)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mean(self: Self) T {
            return 1.0 / self.p;
        }

        /// Variance of the distribution ((1-p)/p²)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn variance(self: Self) T {
            return (1.0 - self.p) / (self.p * self.p);
        }

        /// Generate a random sample from this distribution
        ///
        /// Uses geometric sampling: count trials until first success
        ///
        /// Time: O(k) expected, where k is the sample value | Space: O(1)
        pub fn sample(self: Self, rng: std.Random) u64 {
            var k: u64 = 1;
            while (rng.float(T) >= self.p) : (k += 1) {}
            return k;
        }

        /// Validate internal invariants: 0 < p ≤ 1 and p is finite
        ///
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.p <= 0.0 or self.p > 1.0) return error.InvalidProbability;
            if (!math.isFinite(self.p)) return error.InvalidProbability;
        }
    };
}

// ============================================================================
// Poisson Distribution
// ============================================================================

/// Poisson distribution Pois(λ)
///
/// Probability mass function (PMF):
///   P(X = k) = (λ^k × e^(-λ)) / k! for k = 0, 1, 2, ...
///
/// Parameters:
///   - rate (λ): Expected number of events (λ > 0)
///
/// Time: O(k) for pmf/cdf at value k, O(1) for sample
pub fn Poisson(comptime T: type) type {
    return struct {
        rate: T,

        const Self = @This();

        /// Create a Poisson distribution with given rate
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(rate: T) DistributionError!Self {
            if (rate <= 0.0) return error.InvalidParameter;
            if (!math.isFinite(rate)) return error.InvalidParameter;
            return Self{ .rate = rate };
        }

        /// Probability mass function (PMF) at k
        ///
        /// P(X = k) = (λ^k × e^(-λ)) / k!
        ///
        /// Uses log-space computation to avoid overflow for large k
        ///
        /// Time: O(k) for factorial computation | Space: O(1)
        pub fn pmf(self: Self, k: u64) T {
            // P(X = k) = exp(k × log(λ) - λ - log(k!))
            const k_f = @as(T, @floatFromInt(k));
            return @exp(k_f * @log(self.rate) - self.rate - logFactorial(T, k));
        }

        /// Cumulative distribution function (CDF) at k
        ///
        /// P(X ≤ k) = Σ(i=0 to k) pmf(i)
        ///
        /// Uses regularized incomplete gamma function: P(k+1, λ)
        ///
        /// Time: O(k) via summation | Space: O(1)
        pub fn cdf(self: Self, k: u64) T {
            if (k == 0) return self.pmf(0);

            // Sum PMF values from 0 to k
            var sum: T = 0.0;
            for (0..k + 1) |i| {
                sum += self.pmf(i);
            }
            return @min(sum, 1.0); // Clamp to 1.0 due to floating-point errors
        }

        /// Quantile function (inverse CDF) - returns k such that P(X ≤ k) ≥ p
        ///
        /// Uses search over PMF values
        ///
        /// Time: O(k) where k is the result | Space: O(1)
        pub fn quantile(self: Self, p: T) DistributionError!u64 {
            if (p < 0.0 or p > 1.0) return error.InvalidProbability;
            if (p == 0.0) return 0;

            // Search for smallest k such that CDF(k) >= p
            var cumulative: T = 0.0;
            var k: u64 = 0;
            while (k < 1000) : (k += 1) { // Cap at 1000 to avoid infinite loops
                cumulative += self.pmf(k);
                if (cumulative >= p) return k;
            }
            return k;
        }

        /// Generate a random sample from this distribution
        ///
        /// Uses Knuth's algorithm for λ < 30, otherwise normal approximation
        ///
        /// Time: O(λ) for Knuth, O(1) for normal approximation | Space: O(1)
        pub fn sample(self: Self, rng: std.Random) u64 {
            // For small λ: use Knuth's algorithm
            if (self.rate < 30.0) {
                const L = @exp(-self.rate);
                var k: u64 = 0;
                var p: T = 1.0;
                while (true) {
                    k += 1;
                    const u = rng.float(T);
                    p *= u;
                    if (p <= L) break;
                }
                return k - 1;
            }

            // For large λ: use normal approximation N(λ, λ)
            const normal = Normal(T){ .mean = self.rate, .std = @sqrt(self.rate) };
            const sample_f = normal.sample(rng);
            if (sample_f < 0.0) return 0;
            return @intFromFloat(@floor(sample_f + 0.5));
        }

        /// Log probability mass function
        ///
        /// log P(X = k) = k × log(λ) - λ - log(k!)
        ///
        /// Time: O(k) | Space: O(1)
        pub fn logpmf(self: Self, k: u64) T {
            const k_f = @as(T, @floatFromInt(k));
            return k_f * @log(self.rate) - self.rate - logFactorial(T, k);
        }

        /// Mean of the distribution (λ)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mean(self: Self) T {
            return self.rate;
        }

        /// Variance of the distribution (λ)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn variance(self: Self) T {
            return self.rate;
        }

        /// Assert that parameters are valid: rate (λ) > 0 and finite.
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.rate <= 0.0 or !math.isFinite(self.rate)) return DistributionError.InvalidParameter;
        }
    };
}

// ============================================================================
// Binomial Distribution
// ============================================================================

/// Binomial distribution B(n, p)
///
/// Probability mass function (PMF):
///   P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
///   where C(n,k) = n! / (k! × (n-k)!)
///
/// Parameters:
///   - n: Number of trials (n ≥ 0)
///   - p: Success probability (0 ≤ p ≤ 1)
///
/// Time: O(k) for pmf/cdf, O(n) for sample
pub fn Binomial(comptime T: type) type {
    return struct {
        n: u64,
        p: T,

        const Self = @This();

        /// Create a binomial distribution with given trials and probability
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(n: u64, p: T) DistributionError!Self {
            if (p < 0.0 or p > 1.0) return error.InvalidProbability;
            if (!math.isFinite(p)) return error.InvalidParameter;
            return Self{ .n = n, .p = p };
        }

        /// Probability mass function (PMF) at k
        ///
        /// P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
        ///
        /// Uses log-space computation to avoid overflow
        ///
        /// Time: O(k) for binomial coefficient | Space: O(1)
        pub fn pmf(self: Self, k: u64) T {
            if (k > self.n) return 0.0;
            if (self.p == 0.0) return if (k == 0) 1.0 else 0.0;
            if (self.p == 1.0) return if (k == self.n) 1.0 else 0.0;

            // P(X = k) = exp(log C(n,k) + k log p + (n-k) log(1-p))
            const k_f = @as(T, @floatFromInt(k));
            const n_f = @as(T, @floatFromInt(self.n));
            const log_prob = logBinomialCoeff(T, self.n, k) +
                k_f * @log(self.p) +
                (n_f - k_f) * @log(1.0 - self.p);
            return @exp(log_prob);
        }

        /// Cumulative distribution function (CDF) at k
        ///
        /// P(X ≤ k) = Σ(i=0 to k) pmf(i)
        ///
        /// Time: O(k) via summation | Space: O(1)
        pub fn cdf(self: Self, k: u64) T {
            const k_clamped = @min(k, self.n);

            var sum: T = 0.0;
            for (0..k_clamped + 1) |i| {
                sum += self.pmf(i);
            }
            return @min(sum, 1.0);
        }

        /// Quantile function (inverse CDF) - returns k such that P(X ≤ k) ≥ p
        ///
        /// Time: O(n) worst case | Space: O(1)
        pub fn quantile(self: Self, prob: T) DistributionError!u64 {
            if (prob < 0.0 or prob > 1.0) return error.InvalidProbability;
            if (prob == 0.0) return 0;
            if (prob == 1.0) return self.n;

            // Search for smallest k such that CDF(k) >= prob
            var cumulative: T = 0.0;
            for (0..self.n + 1) |k| {
                cumulative += self.pmf(k);
                if (cumulative >= prob) return k;
            }
            return self.n;
        }

        /// Generate a random sample from this distribution
        ///
        /// Uses binomial sampling: sum of n Bernoulli(p) trials
        ///
        /// Time: O(n) | Space: O(1)
        pub fn sample(self: Self, rng: std.Random) u64 {
            var count: u64 = 0;
            for (0..self.n) |_| {
                if (rng.float(T) < self.p) {
                    count += 1;
                }
            }
            return count;
        }

        /// Log probability mass function
        ///
        /// Time: O(k) | Space: O(1)
        pub fn logpmf(self: Self, k: u64) T {
            if (k > self.n) return -math.inf(T);
            if (self.p == 0.0) return if (k == 0) 0.0 else -math.inf(T);
            if (self.p == 1.0) return if (k == self.n) 0.0 else -math.inf(T);

            const k_f = @as(T, @floatFromInt(k));
            const n_f = @as(T, @floatFromInt(self.n));
            return logBinomialCoeff(T, self.n, k) +
                k_f * @log(self.p) +
                (n_f - k_f) * @log(1.0 - self.p);
        }

        /// Mean of the distribution (n × p)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mean(self: Self) T {
            return @as(T, @floatFromInt(self.n)) * self.p;
        }

        /// Variance of the distribution (n × p × (1-p))
        ///
        /// Time: O(1) | Space: O(1)
        pub fn variance(self: Self) T {
            const n_f = @as(T, @floatFromInt(self.n));
            return n_f * self.p * (1.0 - self.p);
        }

        /// Assert that parameters are valid: n ≥ 1, 0 ≤ p ≤ 1.
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.n == 0) return DistributionError.InvalidParameter;
            if (self.p < 0.0 or self.p > 1.0 or !math.isFinite(self.p)) return DistributionError.InvalidParameter;
        }
    };
}

// ============================================================================
// Helper Functions (Special Functions)
// ============================================================================

/// Log factorial: log(n!)
///
/// Uses Stirling's approximation for large n, exact computation for small n
///
/// Time: O(n) for n < 20, O(1) for n ≥ 20 | Space: O(1)
fn logFactorial(comptime T: type, n: u64) T {
    if (n == 0 or n == 1) return 0.0;

    // For small n, use exact computation
    if (n < 20) {
        var result: T = 0.0;
        for (2..n + 1) |i| {
            result += @log(@as(T, @floatFromInt(i)));
        }
        return result;
    }

    // For large n, use Stirling's approximation:
    // log(n!) ≈ n log(n) - n + 0.5 log(2πn)
    const n_f = @as(T, @floatFromInt(n));
    return n_f * @log(n_f) - n_f + 0.5 * @log(2.0 * math.pi * n_f);
}

/// Log binomial coefficient: log(C(n, k)) = log(n! / (k! × (n-k)!))
///
/// Time: O(k) | Space: O(1)
fn logBinomialCoeff(comptime T: type, n: u64, k: u64) T {
    if (k > n) return -math.inf(T);
    if (k == 0 or k == n) return 0.0;

    // Use symmetry: C(n,k) = C(n, n-k)
    const k_opt = @min(k, n - k);

    return logFactorial(T, n) - logFactorial(T, k_opt) - logFactorial(T, n - k_opt);
}

// ============================================================================
// Helper Functions (Special Functions)
// ============================================================================

/// Log gamma function: log(Γ(x))
///
/// Uses Lanczos approximation for numerical stability
///
/// Time: O(1) | Space: O(1)
fn logGamma(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);

    // Lanczos coefficients for g=7, n=9
    const lanczos_g: T = 7.0;
    const lanczos_coef = [_]T{
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
        // Use reflection formula: Γ(1-x)Γ(x) = π/sin(πx)
        // log(Γ(x)) = log(π) - log(sin(πx)) - log(Γ(1-x))
        return @log(math.pi) - @log(@abs(@sin(math.pi * x))) - logGamma(1.0 - x);
    }

    const z = x - 1.0;
    var base = lanczos_coef[0];
    for (1..lanczos_coef.len) |i| {
        base += lanczos_coef[i] / (z + @as(T, @floatFromInt(i)));
    }

    const t = z + lanczos_g + 0.5;
    const log_sqrt_2pi: T = 0.5 * @log(2.0 * math.pi);

    return log_sqrt_2pi + @log(base) - t + (z + 0.5) * @log(t);
}

/// Regularized lower incomplete gamma function: P(a,x) = γ(a,x)/Γ(a)
///
/// Uses series expansion for x < a+1, continued fraction for x ≥ a+1
///
/// Time: O(1) with finite iterations | Space: O(1)
fn regularizedGammaP(a: anytype, x: anytype) @TypeOf(a) {
    const T = @TypeOf(a);

    if (x <= 0.0) return 0.0;
    if (x == math.inf(T)) return 1.0;

    const max_iterations = 200;
    const tolerance = 1e-10;

    if (x < a + 1.0) {
        // Use series expansion: P(a,x) = e^(-x) x^a Σ(Γ(a)/Γ(a+1+n) x^n)
        var ap = a;
        var del = 1.0 / a;
        var sum = del;

        for (0..max_iterations) |_| {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if (@abs(del) < @abs(sum) * tolerance) {
                const log_result = a * @log(x) - x - logGamma(a) + @log(sum);
                return @exp(log_result);
            }
        }

        // If not converged, return current approximation
        const log_result = a * @log(x) - x - logGamma(a) + @log(sum);
        return @exp(log_result);
    } else {
        // Use continued fraction: Q(a,x) = e^(-x) x^a × CF
        // P(a,x) = 1 - Q(a,x)

        var b: T = x + 1.0 - a;
        var c: T = 1.0 / (1.0e-30); // Large number
        var d: T = 1.0 / b;
        var h: T = d;

        for (1..max_iterations + 1) |i| {
            const i_f = @as(T, @floatFromInt(i));
            const an = -i_f * (i_f - a);
            b += 2.0;
            d = an * d + b;
            if (@abs(d) < 1.0e-30) d = 1.0e-30;
            c = b + an / c;
            if (@abs(c) < 1.0e-30) c = 1.0e-30;
            d = 1.0 / d;
            const del = d * c;
            h *= del;
            if (@abs(del - 1.0) < tolerance) {
                const log_result = a * @log(x) - x - logGamma(a) + @log(h);
                return 1.0 - @exp(log_result);
            }
        }

        // If not converged, return current approximation
        const log_result = a * @log(x) - x - logGamma(a) + @log(h);
        return 1.0 - @exp(log_result);
    }
}

/// Log beta function: log(B(a,b)) = log(Γ(a)) + log(Γ(b)) - log(Γ(a+b))
///
/// Time: O(1) | Space: O(1)
fn logBeta(a: anytype, b: anytype) @TypeOf(a) {
    return logGamma(a) + logGamma(b) - logGamma(a + b);
}

/// Regularized incomplete beta function: I_x(a,b) = B_x(a,b) / B(a,b)
///
/// Uses continued fraction expansion (Lentz's algorithm)
///
/// Time: O(1) with finite iterations | Space: O(1)
fn regularizedBetaI(a_in: anytype, b_in: anytype, x_in: anytype) @TypeOf(a_in + x_in) {
    const T = @TypeOf(a_in + x_in);
    const a: T = @floatCast(a_in);
    const b: T = @floatCast(b_in);
    const x: T = @floatCast(x_in);

    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;

    // Use symmetry I_x(a,b) = 1 - I_(1-x)(b,a) when x is large (improves convergence)
    const flip = x > (a + 1.0) / (a + b + 2.0);
    const ax: T = if (flip) b else a;
    const bx: T = if (flip) a else b;
    const xx: T = if (flip) 1.0 - x else x;

    // bt = x^a * (1-x)^b / (a * B(a,b))
    const log_bt = ax * @log(xx) + bx * @log(1.0 - xx) - logBeta(ax, bx) - @log(ax);
    const bt = @exp(log_bt);

    // Numerical Recipes betacf: continued fraction for I_x(a,b) via modified Lentz
    const fpmin: T = 1.0e-30;
    const qab = ax + bx;
    const qap = ax + 1.0;
    const qam = ax - 1.0;

    var c: T = 1.0;
    var d: T = 1.0 - qab * xx / qap;
    if (@abs(d) < fpmin) d = fpmin;
    d = 1.0 / d;
    var h: T = d;

    for (1..200) |m| {
        const m_f: T = @floatFromInt(m);
        const m2 = 2.0 * m_f;

        // Even step
        var aa: T = m_f * (bx - m_f) * xx / ((qam + m2) * (ax + m2));
        d = 1.0 + aa * d;
        if (@abs(d) < fpmin) d = fpmin;
        c = 1.0 + aa / c;
        if (@abs(c) < fpmin) c = fpmin;
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        aa = -(ax + m_f) * (qab + m_f) * xx / ((ax + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (@abs(d) < fpmin) d = fpmin;
        c = 1.0 + aa / c;
        if (@abs(c) < fpmin) c = fpmin;
        d = 1.0 / d;
        const delta = d * c;
        h *= delta;

        if (@abs(delta - 1.0) < 3.0e-7) break;
    }

    const result = bt * h;
    return if (flip) 1.0 - result else result;
}

/// Error function (erf) using rational approximation
///
/// Abramowitz & Stegun formula 7.1.26 (max error: 1.5e-7)
///
/// Time: O(1) | Space: O(1)
fn erf(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    if (x == 0.0) return 0.0;

    const abs_x = @abs(x);
    const sign: T = if (x >= 0.0) 1.0 else -1.0;

    // Coefficients for rational approximation
    const a1: T = 0.254829592;
    const a2: T = -0.284496736;
    const a3: T = 1.421413741;
    const a4: T = -1.453152027;
    const a5: T = 1.061405429;
    const p: T = 0.3275911;

    const t = 1.0 / (1.0 + p * abs_x);
    const t2 = t * t;
    const t3 = t2 * t;
    const t4 = t3 * t;
    const t5 = t4 * t;

    const poly = a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5;
    return sign * (1.0 - poly * @exp(-abs_x * abs_x));
}

/// Inverse error function (erf⁻¹) using rational approximation
///
/// Uses Beasley-Springer-Moro algorithm (max error: 3e-9)
///
/// Time: O(1) | Space: O(1)
fn erfInv(y: anytype) @TypeOf(y) {
    const T = @TypeOf(y);
    if (y == 0.0) return 0.0;
    if (y == 1.0) return math.inf(T);
    if (y == -1.0) return -math.inf(T);

    const abs_y = @abs(y);
    const sign: T = if (y >= 0.0) 1.0 else -1.0;

    // Initial approximation (two regions for different accuracy characteristics)
    var x0: T = undefined;
    if (abs_y <= 0.7) {
        const y2 = y * y;
        const a0: T = 1.0;
        const a1: T = 0.1975;
        const a2: T = 0.1186;
        const a3: T = 0.0742;

        const b0: T = 1.0;
        const b1: T = 0.257;
        const b2: T = 0.0982;

        const num = a0 + a1 * y2 + a2 * y2 * y2 + a3 * y2 * y2 * y2;
        const den = b0 + b1 * y2 + b2 * y2 * y2;
        x0 = y * num / den;
    } else {
        // Tail region: 0.7 < |y| < 1
        const z = @sqrt(-@log(0.5 * (1.0 - abs_y)));

        const c0: T = 2.515517;
        const c1: T = 0.802853;
        const c2: T = 0.010328;

        const d1: T = 1.432788;
        const d2: T = 0.189269;
        const d3: T = 0.001308;

        const num = c0 + c1 * z + c2 * z * z;
        const den = 1.0 + d1 * z + d2 * z * z + d3 * z * z * z;
        x0 = sign * (z - num / den);
    }

    // Newton-Raphson refinement applied to ALL regions:
    // x = x - (erf(x) - y) / (2/sqrt(π) * exp(-x²))
    // Makes erfInv self-consistent with erf, enabling near-exact roundtrips
    const two_over_sqrt_pi: T = 2.0 / @sqrt(math.pi);
    for (0..5) |_| {
        const residual = erf(x0) - y;
        const deriv = two_over_sqrt_pi * @exp(-x0 * x0);
        x0 = x0 - residual / deriv;
    }
    return x0;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const expectApproxEqRel = testing.expectApproxEqRel;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectEqual = testing.expectEqual;
const expectError = testing.expectError;
const expect = testing.expect;

test "Normal distribution: init" {
    const dist = try Normal(f64).init(0.0, 1.0);
    try expectEqual(0.0, dist.mean);
    try expectEqual(1.0, dist.std);

    // Invalid std (≤ 0)
    try expectError(error.InvalidParameter, Normal(f64).init(0.0, 0.0));
    try expectError(error.InvalidParameter, Normal(f64).init(0.0, -1.0));

    // Invalid (non-finite)
    try expectError(error.InvalidParameter, Normal(f64).init(math.nan(f64), 1.0));
    try expectError(error.InvalidParameter, Normal(f64).init(0.0, math.inf(f64)));
}

test "Normal distribution: pdf" {
    const dist = try Normal(f64).init(0.0, 1.0);

    // Standard normal at mean: f(0) = 1/√(2π) ≈ 0.3989
    try expectApproxEqRel(0.3989422804014327, dist.pdf(0.0), 1e-10);

    // At μ ± σ: f(±1) ≈ 0.2420
    try expectApproxEqRel(0.24197072451914337, dist.pdf(1.0), 1e-10);
    try expectApproxEqRel(0.24197072451914337, dist.pdf(-1.0), 1e-10);

    // Non-standard normal: N(5, 2)
    const dist2 = try Normal(f64).init(5.0, 2.0);
    // At mean: f(5) = 1/(2√(2π)) ≈ 0.1995
    try expectApproxEqRel(0.19947114020071635, dist2.pdf(5.0), 1e-10);
}

test "Normal distribution: cdf" {
    const dist = try Normal(f64).init(0.0, 1.0);

    // Standard normal at mean: Φ(0) = 0.5
    try expectApproxEqRel(0.5, dist.cdf(0.0), 1e-10);

    // At μ ± σ: Φ(±1) ≈ 0.8413, 0.1587
    try expectApproxEqRel(0.8413447460685429, dist.cdf(1.0), 1e-5);
    try expectApproxEqRel(0.15865525393145707, dist.cdf(-1.0), 1e-5);

    // At μ ± 2σ: Φ(±2) ≈ 0.9772, 0.0228
    try expectApproxEqRel(0.9772498680518208, dist.cdf(2.0), 1e-5);
    try expectApproxEqRel(0.022750131948179195, dist.cdf(-2.0), 1e-5);
}

test "Normal distribution: quantile" {
    const dist = try Normal(f64).init(0.0, 1.0);

    // Median: Φ⁻¹(0.5) = 0
    try expectApproxEqRel(0.0, try dist.quantile(0.5), 1e-5);

    // 84th percentile: Φ⁻¹(0.8413) ≈ 1 (use looser tolerance for approximation)
    try expectApproxEqRel(1.0, try dist.quantile(0.8413447460685429), 0.05);

    // 16th percentile: Φ⁻¹(0.1587) ≈ -1
    try expectApproxEqRel(-1.0, try dist.quantile(0.15865525393145707), 0.05);

    // Edge cases
    try expectEqual(math.inf(f64), try dist.quantile(1.0));
    try expectEqual(-math.inf(f64), try dist.quantile(0.0));

    // Invalid probability
    try expectError(error.InvalidProbability, dist.quantile(-0.1));
    try expectError(error.InvalidProbability, dist.quantile(1.1));
}

test "Normal distribution: sample" {
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    const dist = try Normal(f64).init(0.0, 1.0);

    // Generate 1000 samples and check mean/std approximately correct
    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    const n = 1000;

    for (0..n) |_| {
        const x = dist.sample(rng);
        sum += x;
        sum_sq += x * x;
    }

    const sample_mean = sum / @as(f64, @floatFromInt(n));
    const sample_var = (sum_sq / @as(f64, @floatFromInt(n))) - (sample_mean * sample_mean);

    // Mean should be close to 0, std close to 1 (with tolerance for randomness)
    try expectApproxEqAbs(0.0, sample_mean, 0.1);
    try expectApproxEqAbs(1.0, @sqrt(sample_var), 0.1);
}

test "Normal distribution: logpdf and sf" {
    const dist = try Normal(f64).init(0.0, 1.0);

    // logpdf(0) = log(pdf(0))
    try expectApproxEqRel(@log(dist.pdf(0.0)), dist.logpdf(0.0), 1e-10);

    // sf(x) = 1 - cdf(x)
    try expectApproxEqRel(1.0 - dist.cdf(1.5), dist.sf(1.5), 1e-10);
}

test "Uniform distribution: init" {
    const dist = try Uniform(f64).init(0.0, 1.0);
    try expectEqual(0.0, dist.a);
    try expectEqual(1.0, dist.b);

    // Invalid (a >= b)
    try expectError(error.InvalidParameter, Uniform(f64).init(1.0, 0.0));
    try expectError(error.InvalidParameter, Uniform(f64).init(1.0, 1.0));

    // Invalid (non-finite)
    try expectError(error.InvalidParameter, Uniform(f64).init(math.nan(f64), 1.0));
}

test "Uniform distribution: pdf" {
    const dist = try Uniform(f64).init(0.0, 4.0);

    // Inside [0, 4): pdf = 1/4 = 0.25
    try expectEqual(0.25, dist.pdf(0.0));
    try expectEqual(0.25, dist.pdf(2.0));
    try expectEqual(0.25, dist.pdf(3.99));

    // Outside [0, 4): pdf = 0
    try expectEqual(0.0, dist.pdf(-1.0));
    try expectEqual(0.0, dist.pdf(5.0));
}

test "Uniform distribution: cdf" {
    const dist = try Uniform(f64).init(0.0, 4.0);

    // CDF(x) = (x - a) / (b - a) = x / 4
    try expectEqual(0.0, dist.cdf(-1.0)); // Before range
    try expectEqual(0.0, dist.cdf(0.0)); // At start
    try expectEqual(0.5, dist.cdf(2.0)); // Midpoint
    try expectEqual(1.0, dist.cdf(4.0)); // At end
    try expectEqual(1.0, dist.cdf(5.0)); // After range
}

test "Uniform distribution: quantile" {
    const dist = try Uniform(f64).init(0.0, 4.0);

    // Quantile(p) = a + p × (b - a) = 4p
    try expectEqual(0.0, try dist.quantile(0.0));
    try expectEqual(2.0, try dist.quantile(0.5));
    try expectEqual(4.0, try dist.quantile(1.0));

    // Invalid probability
    try expectError(error.InvalidProbability, dist.quantile(-0.1));
    try expectError(error.InvalidProbability, dist.quantile(1.1));
}

test "Uniform distribution: sample" {
    var prng = std.Random.DefaultPrng.init(54321);
    const rng = prng.random();

    const dist = try Uniform(f64).init(0.0, 10.0);

    // Generate 1000 samples and check mean approximately 5.0
    var sum: f64 = 0.0;
    const n = 1000;

    for (0..n) |_| {
        const x = dist.sample(rng);
        sum += x;
        try testing.expect(x >= 0.0 and x < 10.0); // In range
    }

    const sample_mean = sum / @as(f64, @floatFromInt(n));
    try expectApproxEqAbs(5.0, sample_mean, 0.3); // Mean of U(0,10) = 5
}

test "Exponential distribution: init" {
    const dist = try Exponential(f64).init(2.0);
    try expectEqual(2.0, dist.rate);

    // Invalid rate (≤ 0)
    try expectError(error.InvalidParameter, Exponential(f64).init(0.0));
    try expectError(error.InvalidParameter, Exponential(f64).init(-1.0));

    // Invalid (non-finite)
    try expectError(error.InvalidParameter, Exponential(f64).init(math.inf(f64)));
}

test "Exponential distribution: pdf" {
    const dist = try Exponential(f64).init(2.0);

    // pdf(0) = λ = 2.0
    try expectEqual(2.0, dist.pdf(0.0));

    // pdf(x) = 2 × exp(-2x)
    try expectApproxEqRel(2.0 * @exp(-2.0 * 0.5), dist.pdf(0.5), 1e-10);

    // pdf(x < 0) = 0
    try expectEqual(0.0, dist.pdf(-1.0));
}

test "Exponential distribution: cdf" {
    const dist = try Exponential(f64).init(2.0);

    // cdf(0) = 0
    try expectEqual(0.0, dist.cdf(0.0));

    // cdf(x) = 1 - exp(-2x)
    try expectApproxEqRel(1.0 - @exp(-2.0 * 0.5), dist.cdf(0.5), 1e-10);

    // cdf at mean (1/λ = 0.5): 1 - exp(-1) ≈ 0.6321
    try expectApproxEqRel(0.6321205588285577, dist.cdf(0.5), 1e-10);

    // cdf(x < 0) = 0
    try expectEqual(0.0, dist.cdf(-1.0));
}

test "Exponential distribution: quantile" {
    const dist = try Exponential(f64).init(2.0);

    // Quantile(p) = -log(1-p) / λ
    try expectEqual(0.0, try dist.quantile(0.0));

    // Median: quantile(0.5) = -log(0.5) / 2 = log(2) / 2 ≈ 0.3466
    try expectApproxEqRel(0.34657359027997264, try dist.quantile(0.5), 1e-10);

    // quantile(1.0) = +inf
    try expectEqual(math.inf(f64), try dist.quantile(1.0));

    // Invalid probability
    try expectError(error.InvalidProbability, dist.quantile(-0.1));
    try expectError(error.InvalidProbability, dist.quantile(1.1));
}

test "Exponential distribution: sample" {
    var prng = std.Random.DefaultPrng.init(99999);
    const rng = prng.random();

    const dist = try Exponential(f64).init(2.0);

    // Generate 1000 samples and check mean approximately 1/λ = 0.5
    var sum: f64 = 0.0;
    const n = 1000;

    for (0..n) |_| {
        const x = dist.sample(rng);
        sum += x;
        try testing.expect(x >= 0.0); // Non-negative
    }

    const sample_mean = sum / @as(f64, @floatFromInt(n));
    try expectApproxEqAbs(0.5, sample_mean, 0.05); // Mean = 1/λ = 0.5
}

test "Exponential distribution: mean and variance" {
    const dist = try Exponential(f64).init(2.0);

    // Mean = 1/λ = 0.5
    try expectEqual(0.5, dist.mean());

    // Variance = 1/λ² = 0.25
    try expectEqual(0.25, dist.variance());
}

test "Normal distribution: f32 precision" {
    const dist = try Normal(f32).init(0.0, 1.0);
    try expectApproxEqRel(@as(f32, 0.3989423), dist.pdf(0.0), 1e-5);
    try expectApproxEqRel(@as(f32, 0.5), dist.cdf(0.0), 1e-5);
}

test "Uniform distribution: f32 precision" {
    const dist = try Uniform(f32).init(0.0, 4.0);
    try expectEqual(@as(f32, 0.25), dist.pdf(2.0));
}

test "Exponential distribution: f32 precision" {
    const dist = try Exponential(f32).init(2.0);
    try expectEqual(@as(f32, 2.0), dist.pdf(0.0));
}

test "Gamma distribution: init" {
    const dist = try Gamma(f64).init(2.0, 3.0);
    try expectEqual(2.0, dist.shape);
    try expectEqual(3.0, dist.rate);

    // Invalid shape (≤ 0)
    try expectError(error.InvalidParameter, Gamma(f64).init(0.0, 1.0));
    try expectError(error.InvalidParameter, Gamma(f64).init(-1.0, 1.0));

    // Invalid rate (≤ 0)
    try expectError(error.InvalidParameter, Gamma(f64).init(1.0, 0.0));
    try expectError(error.InvalidParameter, Gamma(f64).init(1.0, -1.0));

    // Invalid (non-finite)
    try expectError(error.InvalidParameter, Gamma(f64).init(math.inf(f64), 1.0));
    try expectError(error.InvalidParameter, Gamma(f64).init(1.0, math.inf(f64)));
}

test "Gamma distribution: pdf" {
    // Gamma(2, 3): shape=2, rate=3
    const dist = try Gamma(f64).init(2.0, 3.0);

    // pdf(x) = (3^2 / Γ(2)) × x^1 × e^(-3x)
    // Γ(2) = 1, so pdf(x) = 9x × e^(-3x)
    // pdf(1) = 9 × e^(-3) ≈ 0.4480836
    try expectApproxEqRel(0.4480836, dist.pdf(1.0), 1e-6);

    // pdf(0) = 0 for shape > 1
    try expectEqual(0.0, dist.pdf(0.0));

    // pdf(x < 0) = 0
    try expectEqual(0.0, dist.pdf(-1.0));

    // Exponential special case: Gamma(1, λ) = Exponential(λ)
    const exp_as_gamma = try Gamma(f64).init(1.0, 2.0);
    const exp_dist = try Exponential(f64).init(2.0);
    try expectApproxEqRel(exp_dist.pdf(0.5), exp_as_gamma.pdf(0.5), 1e-10);
}

test "Gamma distribution: cdf" {
    const dist = try Gamma(f64).init(2.0, 1.0);

    // cdf(0) = 0
    try expectEqual(0.0, dist.cdf(0.0));

    // cdf(∞) = 1 (test at large value)
    try expectApproxEqRel(1.0, dist.cdf(100.0), 1e-5);

    // For Gamma(2, 1), cdf(2) = 1 - 3e^(-2) ≈ 0.5940
    try expectApproxEqRel(0.5940, dist.cdf(2.0), 1e-3);

    // cdf(x < 0) = 0
    try expectEqual(0.0, dist.cdf(-1.0));
}

test "Gamma distribution: quantile" {
    const dist = try Gamma(f64).init(2.0, 1.0);

    // quantile(0) = 0
    try expectEqual(0.0, try dist.quantile(0.0));

    // quantile(1) = ∞
    try expectEqual(math.inf(f64), try dist.quantile(1.0));

    // quantile(0.5) should be near median
    const median = try dist.quantile(0.5);
    try expectApproxEqRel(0.5, dist.cdf(median), 1e-6);

    // Roundtrip: cdf(quantile(p)) ≈ p
    const p1 = 0.25;
    const q1 = try dist.quantile(p1);
    try expectApproxEqRel(p1, dist.cdf(q1), 1e-6);

    const p2 = 0.75;
    const q2 = try dist.quantile(p2);
    try expectApproxEqRel(p2, dist.cdf(q2), 1e-6);

    // Invalid probabilities
    try expectError(error.InvalidProbability, dist.quantile(-0.1));
    try expectError(error.InvalidProbability, dist.quantile(1.1));
}

test "Gamma distribution: sample mean validation" {
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    // Gamma(2, 3): mean = 2/3 ≈ 0.6667
    const dist = try Gamma(f64).init(2.0, 3.0);

    const n = 10000;
    var sum: f64 = 0.0;
    for (0..n) |_| {
        const x = dist.sample(rng);
        try testing.expect(x >= 0.0); // All samples non-negative
        sum += x;
    }

    const sample_mean = sum / @as(f64, @floatFromInt(n));
    try expectApproxEqAbs(0.6667, sample_mean, 0.05); // Mean = α/β

    // Test variance: Var = α/β² = 2/9 ≈ 0.2222
    sum = 0.0;
    var sum_sq: f64 = 0.0;
    for (0..n) |_| {
        const x = dist.sample(rng);
        sum += x;
        sum_sq += x * x;
    }
    const mean = sum / @as(f64, @floatFromInt(n));
    const variance = (sum_sq / @as(f64, @floatFromInt(n))) - (mean * mean);
    try expectApproxEqAbs(0.2222, variance, 0.02);
}

test "Gamma distribution: sample with shape < 1" {
    var prng = std.Random.DefaultPrng.init(54321);
    const rng = prng.random();

    // Gamma(0.5, 2.0): shape < 1 tests Ahrens-Dieter method
    const dist = try Gamma(f64).init(0.5, 2.0);

    const n = 5000;
    var sum: f64 = 0.0;
    for (0..n) |_| {
        const x = dist.sample(rng);
        try testing.expect(x >= 0.0);
        sum += x;
    }

    const sample_mean = sum / @as(f64, @floatFromInt(n));
    // Mean = α/β = 0.5/2.0 = 0.25
    try expectApproxEqAbs(0.25, sample_mean, 0.03);
}

test "Gamma distribution: logpdf" {
    const dist = try Gamma(f64).init(2.0, 3.0);

    // logpdf(1) = log(pdf(1))
    try expectApproxEqRel(@log(dist.pdf(1.0)), dist.logpdf(1.0), 1e-10);

    // logpdf(0) = -∞ for shape > 1
    try expectEqual(-math.inf(f64), dist.logpdf(0.0));

    // logpdf(x < 0) = -∞
    try expectEqual(-math.inf(f64), dist.logpdf(-1.0));
}

test "Gamma distribution: mean and variance" {
    const dist = try Gamma(f64).init(2.0, 3.0);

    // Mean = α/β = 2/3
    try expectApproxEqRel(2.0 / 3.0, dist.mean(), 1e-10);

    // Variance = α/β² = 2/9
    try expectApproxEqRel(2.0 / 9.0, dist.variance(), 1e-10);
}

test "Gamma distribution: f32 precision" {
    const dist = try Gamma(f32).init(2.0, 1.0);

    // pdf(1) for Gamma(2, 1) = e^(-1) ≈ 0.3679
    try expectApproxEqRel(@as(f32, 0.3679), dist.pdf(1.0), 1e-3);

    // Mean = 2/1 = 2
    try expectEqual(@as(f32, 2.0), dist.mean());
}

test "Beta distribution: init" {
    const dist = try Beta(f64).init(2.0, 5.0);
    try expectEqual(2.0, dist.alpha);
    try expectEqual(5.0, dist.beta);

    // Invalid parameters (≤ 0)
    try expectError(error.InvalidParameter, Beta(f64).init(0.0, 5.0));
    try expectError(error.InvalidParameter, Beta(f64).init(2.0, -1.0));

    // Invalid (non-finite)
    try expectError(error.InvalidParameter, Beta(f64).init(math.nan(f64), 5.0));
    try expectError(error.InvalidParameter, Beta(f64).init(2.0, math.inf(f64)));
}

test "Beta distribution: pdf" {
    const dist = try Beta(f64).init(2.0, 5.0);

    // Beta(2, 5): pdf(x) = x^(α-1)(1-x)^(β-1)/B(α,β) = x^1*(1-x)^4 / B(2,5)
    // B(2,5) = Γ(2)Γ(5)/Γ(7) = 1!×4!/6! = 24/720 = 1/30
    // pdf(0.5) = 0.5 × (0.5)^4 × 30 = 0.5 × 0.0625 × 30 = 0.9375
    try expectApproxEqRel(0.9375, dist.pdf(0.5), 1e-10);

    // pdf outside [0,1] = 0
    try expectEqual(0.0, dist.pdf(-0.1));
    try expectEqual(0.0, dist.pdf(1.1));

    // Uniform distribution: Beta(1, 1) has constant pdf = 1
    const uniform_beta = try Beta(f64).init(1.0, 1.0);
    try expectApproxEqRel(1.0, uniform_beta.pdf(0.5), 1e-10);
}

test "Beta distribution: cdf" {
    const dist = try Beta(f64).init(2.0, 5.0);

    // CDF(0) = 0, CDF(1) = 1
    try expectEqual(0.0, dist.cdf(0.0));
    try expectEqual(1.0, dist.cdf(1.0));

    // CDF should be monotonically increasing
    const cdf_02 = dist.cdf(0.2);
    const cdf_05 = dist.cdf(0.5);
    const cdf_08 = dist.cdf(0.8);
    try testing.expect(cdf_02 < cdf_05);
    try testing.expect(cdf_05 < cdf_08);

    // CDF values should be in [0, 1]
    try testing.expect(cdf_05 >= 0.0 and cdf_05 <= 1.0);
}

test "Beta distribution: quantile" {
    const dist = try Beta(f64).init(2.0, 5.0);

    // quantile(0) = 0, quantile(1) = 1
    try expectApproxEqAbs(0.0, try dist.quantile(0.0), 1e-10);
    try expectApproxEqAbs(1.0, try dist.quantile(1.0), 1e-10);

    // Roundtrip: quantile(cdf(x)) ≈ x
    const x = 0.3;
    const p = dist.cdf(x);
    const x_reconstructed = try dist.quantile(p);
    try expectApproxEqAbs(x, x_reconstructed, 1e-6);

    // Median should be < mean for α < β (right-skewed)
    const median = try dist.quantile(0.5);
    try testing.expect(median < dist.mean());

    // Invalid probability
    try expectError(error.InvalidProbability, dist.quantile(-0.1));
    try expectError(error.InvalidProbability, dist.quantile(1.1));
}

test "Beta distribution: sample" {
    var prng = std.Random.DefaultPrng.init(33333);
    const rng = prng.random();

    const dist = try Beta(f64).init(2.0, 5.0);

    // Generate 1000 samples and check mean ≈ α/(α+β) = 2/7 ≈ 0.286
    var sum: f64 = 0.0;
    const n = 1000;

    for (0..n) |_| {
        const x = dist.sample(rng);
        try testing.expect(x >= 0.0 and x <= 1.0); // In [0, 1]
        sum += x;
    }

    const sample_mean = sum / @as(f64, @floatFromInt(n));
    const expected_mean = dist.mean(); // 2/(2+5) = 0.286
    try expectApproxEqAbs(expected_mean, sample_mean, 0.05);
}

test "Beta distribution: mean and variance" {
    const dist = try Beta(f64).init(2.0, 5.0);

    // Mean = α/(α+β) = 2/7 ≈ 0.286
    try expectApproxEqRel(2.0 / 7.0, dist.mean(), 1e-10);

    // Variance = αβ/((α+β)²(α+β+1)) = 2×5/(7²×8) = 10/392 ≈ 0.0255
    const expected_var = (2.0 * 5.0) / (7.0 * 7.0 * 8.0);
    try expectApproxEqRel(expected_var, dist.variance(), 1e-10);
}

test "Beta distribution: special cases" {
    // Uniform: Beta(1, 1)
    const uniform = try Beta(f64).init(1.0, 1.0);
    try expectEqual(0.5, uniform.mean());
    try expectApproxEqRel(1.0 / 12.0, uniform.variance(), 1e-10);

    // Symmetric: Beta(3, 3)
    const symmetric = try Beta(f64).init(3.0, 3.0);
    try expectEqual(0.5, symmetric.mean()); // α = β → mean = 0.5
}

test "Beta distribution: logpdf" {
    const dist = try Beta(f64).init(2.0, 5.0);

    const x = 0.3;
    const pdf_val = dist.pdf(x);
    const logpdf_val = dist.logpdf(x);

    // logpdf(x) ≈ log(pdf(x))
    try expectApproxEqRel(@log(pdf_val), logpdf_val, 1e-10);

    // logpdf outside (0,1) = -inf
    try expectEqual(-math.inf(f64), dist.logpdf(0.0));
    try expectEqual(-math.inf(f64), dist.logpdf(1.0));
}

test "Beta distribution: f32 precision" {
    const dist = try Beta(f32).init(2.0, 5.0);

    // Mean = 2/7 ≈ 0.286
    try expectApproxEqRel(@as(f32, 2.0 / 7.0), dist.mean(), 1e-5);

    // pdf(0.5) ≈ 0.9375
    try expectApproxEqRel(@as(f32, 0.9375), dist.pdf(0.5), 1e-3);
}

test "Poisson distribution: init" {
    const dist = try Poisson(f64).init(3.0);
    try expectEqual(3.0, dist.rate);

    // Invalid rate (≤ 0)
    try expectError(error.InvalidParameter, Poisson(f64).init(0.0));
    try expectError(error.InvalidParameter, Poisson(f64).init(-1.0));

    // Invalid (non-finite)
    try expectError(error.InvalidParameter, Poisson(f64).init(math.inf(f64)));
}

test "Poisson distribution: pmf" {
    const dist = try Poisson(f64).init(3.0);

    // Poisson(3): P(X=0) = e^(-3) ≈ 0.0498
    try expectApproxEqRel(0.049787068367863944, dist.pmf(0), 1e-10);

    // P(X=1) = 3 × e^(-3) ≈ 0.1494
    try expectApproxEqRel(0.14936120510259183, dist.pmf(1), 1e-10);

    // P(X=3) = (3^3 / 3!) × e^(-3) = 4.5 × e^(-3) ≈ 0.2240
    try expectApproxEqRel(0.22404180765538775, dist.pmf(3), 1e-10);

    // P(X=10) should be very small
    try testing.expect(dist.pmf(10) < 0.01);
}

test "Poisson distribution: cdf" {
    const dist = try Poisson(f64).init(3.0);

    // CDF(0) = P(X=0) = e^(-3)
    try expectApproxEqRel(0.049787068367863944, dist.cdf(0), 1e-10);

    // CDF(3) = P(X≤3) (sum of pmf 0 to 3)
    const expected_cdf3 = dist.pmf(0) + dist.pmf(1) + dist.pmf(2) + dist.pmf(3);
    try expectApproxEqRel(expected_cdf3, dist.cdf(3), 1e-10);

    // CDF should be monotonically increasing
    try testing.expect(dist.cdf(0) < dist.cdf(1));
    try testing.expect(dist.cdf(1) < dist.cdf(2));
    try testing.expect(dist.cdf(2) < dist.cdf(3));
}

test "Poisson distribution: quantile" {
    const dist = try Poisson(f64).init(3.0);

    // quantile(0) = 0
    try expectEqual(0, try dist.quantile(0.0));

    // quantile(0.5) should be around mean (3)
    const median = try dist.quantile(0.5);
    try testing.expect(median >= 2 and median <= 4);

    // Invalid probability
    try expectError(error.InvalidProbability, dist.quantile(-0.1));
    try expectError(error.InvalidProbability, dist.quantile(1.1));
}

test "Poisson distribution: sample" {
    var prng = std.Random.DefaultPrng.init(11111);
    const rng = prng.random();

    const dist = try Poisson(f64).init(5.0);

    // Generate 1000 samples and check mean approximately λ = 5
    var sum: u64 = 0;
    const n = 1000;

    for (0..n) |_| {
        const x = dist.sample(rng);
        sum += x;
    }

    const sample_mean = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(n));
    try expectApproxEqAbs(5.0, sample_mean, 0.5); // Mean = λ = 5
}

test "Poisson distribution: mean and variance" {
    const dist = try Poisson(f64).init(5.0);

    // Mean = λ = 5.0
    try expectEqual(5.0, dist.mean());

    // Variance = λ = 5.0
    try expectEqual(5.0, dist.variance());
}

test "Binomial distribution: init" {
    const dist = try Binomial(f64).init(10, 0.5);
    try expectEqual(10, dist.n);
    try expectEqual(0.5, dist.p);

    // Invalid p (< 0 or > 1)
    try expectError(error.InvalidProbability, Binomial(f64).init(10, -0.1));
    try expectError(error.InvalidProbability, Binomial(f64).init(10, 1.1));

    // Invalid (non-finite)
    try expectError(error.InvalidParameter, Binomial(f64).init(10, math.nan(f64)));
}

test "Binomial distribution: pmf" {
    const dist = try Binomial(f64).init(10, 0.5);

    // B(10, 0.5): P(X=0) = (0.5)^10 = 0.0009765625
    try expectApproxEqRel(0.0009765625, dist.pmf(0), 1e-10);

    // P(X=5) = C(10,5) × (0.5)^10 = 252 × 0.0009765625 ≈ 0.2461
    try expectApproxEqRel(0.24609375, dist.pmf(5), 1e-10);

    // P(X=10) = (0.5)^10 = 0.0009765625
    try expectApproxEqRel(0.0009765625, dist.pmf(10), 1e-10);

    // P(X=11) = 0 (k > n)
    try expectEqual(0.0, dist.pmf(11));
}

test "Binomial distribution: pmf edge cases" {
    // p = 0: all mass at k=0
    const dist_p0 = try Binomial(f64).init(10, 0.0);
    try expectEqual(1.0, dist_p0.pmf(0));
    try expectEqual(0.0, dist_p0.pmf(5));

    // p = 1: all mass at k=n
    const dist_p1 = try Binomial(f64).init(10, 1.0);
    try expectEqual(0.0, dist_p1.pmf(5));
    try expectEqual(1.0, dist_p1.pmf(10));
}

test "Binomial distribution: cdf" {
    const dist = try Binomial(f64).init(10, 0.5);

    // CDF(0) = P(X=0) = 0.0009765625
    try expectApproxEqRel(0.0009765625, dist.cdf(0), 1e-10);

    // CDF(5) = P(X≤5) (sum of pmf 0 to 5)
    var expected_cdf5: f64 = 0.0;
    for (0..6) |k| {
        expected_cdf5 += dist.pmf(k);
    }
    try expectApproxEqRel(expected_cdf5, dist.cdf(5), 1e-10);

    // CDF(10) = 1.0
    try expectApproxEqRel(1.0, dist.cdf(10), 1e-10);

    // CDF should be monotonically increasing
    try testing.expect(dist.cdf(3) < dist.cdf(5));
    try testing.expect(dist.cdf(5) < dist.cdf(8));
}

test "Binomial distribution: quantile" {
    const dist = try Binomial(f64).init(10, 0.5);

    // quantile(0) = 0
    try expectEqual(0, try dist.quantile(0.0));

    // quantile(1.0) = n = 10
    try expectEqual(10, try dist.quantile(1.0));

    // quantile(0.5) should be around mean (n × p = 5)
    const median = try dist.quantile(0.5);
    try testing.expect(median >= 4 and median <= 6);

    // Invalid probability
    try expectError(error.InvalidProbability, dist.quantile(-0.1));
    try expectError(error.InvalidProbability, dist.quantile(1.1));
}

test "Binomial distribution: sample" {
    var prng = std.Random.DefaultPrng.init(22222);
    const rng = prng.random();

    const dist = try Binomial(f64).init(20, 0.3);

    // Generate 1000 samples and check mean approximately n × p = 6
    var sum: u64 = 0;
    const n = 1000;

    for (0..n) |_| {
        const x = dist.sample(rng);
        try testing.expect(x <= 20); // k ≤ n
        sum += x;
    }

    const sample_mean = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(n));
    try expectApproxEqAbs(6.0, sample_mean, 0.5); // Mean = n × p = 20 × 0.3 = 6
}

test "Binomial distribution: mean and variance" {
    const dist = try Binomial(f64).init(20, 0.3);

    // Mean = n × p = 6.0
    try expectEqual(6.0, dist.mean());

    // Variance = n × p × (1-p) = 20 × 0.3 × 0.7 = 4.2
    try expectApproxEqRel(4.2, dist.variance(), 1e-10);
}

test "Poisson distribution: f32 precision" {
    const dist = try Poisson(f32).init(3.0);
    try expectApproxEqRel(@as(f32, 0.0498), dist.pmf(0), 1e-3);
}

test "Binomial distribution: f32 precision" {
    const dist = try Binomial(f32).init(10, 0.5);
    try expectApproxEqRel(@as(f32, 0.2461), dist.pmf(5), 1e-3);
}

test "ChiSquared distribution: init" {
    const dist = try ChiSquared(f64).init(5);
    try expectEqual(5.0, dist.k);

    // Invalid k = 0
    try expectError(error.InvalidParameter, ChiSquared(f64).init(0));
}

test "ChiSquared distribution: pdf" {
    const dist = try ChiSquared(f64).init(2);

    // χ²(2) at x=2: f(2) = 0.5 × e^(-1) ≈ 0.1839
    // For k=2: f(x) = (1/2) × e^(-x/2)
    const expected = 0.5 * @exp(-1.0);
    try expectApproxEqRel(expected, dist.pdf(2.0), 1e-10);

    // PDF at x=0 for k=2 should be 0.5
    try expectApproxEqRel(0.5, dist.pdf(0.0), 1e-10);

    // Negative x should return 0
    try expectEqual(0.0, dist.pdf(-1.0));
}

test "ChiSquared distribution: pdf k=1 case" {
    const dist = try ChiSquared(f64).init(1);

    // χ²(1) at x=1: f(1) = (1/sqrt(2π)) × e^(-0.5) ≈ 0.2420
    // This is sum of one squared N(0,1), so at x=1: f(1) = 1/(sqrt(2π×1)) × e^(-1/2)
    const expected = 1.0 / @sqrt(2.0 * math.pi) * @exp(-0.5);
    try expectApproxEqRel(expected, dist.pdf(1.0), 1e-4);
}

test "ChiSquared distribution: cdf" {
    const dist = try ChiSquared(f64).init(2);

    // CDF at x=0 should be 0
    try expectEqual(0.0, dist.cdf(0.0));

    // χ²(2) has CDF: F(x) = 1 - e^(-x/2)
    // F(2) = 1 - e^(-1) ≈ 0.6321
    const expected_cdf2 = 1.0 - @exp(-1.0);
    try expectApproxEqRel(expected_cdf2, dist.cdf(2.0), 1e-10);

    // F(4) = 1 - e^(-2) ≈ 0.8647
    const expected_cdf4 = 1.0 - @exp(-2.0);
    try expectApproxEqRel(expected_cdf4, dist.cdf(4.0), 1e-10);

    // Monotonicity check
    try testing.expect(dist.cdf(1.0) < dist.cdf(2.0));
    try testing.expect(dist.cdf(2.0) < dist.cdf(4.0));
}

test "ChiSquared distribution: quantile" {
    const dist = try ChiSquared(f64).init(2);

    // quantile(0.0) should be 0
    try expectApproxEqAbs(0.0, try dist.quantile(0.0), 1e-6);

    // For χ²(2), median is approximately 1.386
    const median = try dist.quantile(0.5);
    try expectApproxEqAbs(1.386, median, 0.01);

    // Roundtrip: cdf(quantile(p)) ≈ p
    const p = 0.7;
    const q = try dist.quantile(p);
    try expectApproxEqRel(p, dist.cdf(q), 1e-6);

    // Invalid probability
    try expectError(error.InvalidProbability, dist.quantile(-0.1));
    try expectError(error.InvalidProbability, dist.quantile(1.1));
}

test "ChiSquared distribution: sample" {
    var prng = std.Random.DefaultPrng.init(33333);
    const rng = prng.random();

    const dist = try ChiSquared(f64).init(10);

    // Generate 1000 samples and check mean ≈ k = 10
    var sum: f64 = 0.0;
    const n = 1000;

    for (0..n) |_| {
        const x = dist.sample(rng);
        try testing.expect(x >= 0.0); // χ² is always non-negative
        sum += x;
    }

    const sample_mean = sum / @as(f64, @floatFromInt(n));
    try expectApproxEqAbs(10.0, sample_mean, 1.0); // Mean = k = 10, allow ±1 for sampling variation
}

test "ChiSquared distribution: mean and variance" {
    const dist = try ChiSquared(f64).init(7);

    // Mean = k = 7
    try expectEqual(7.0, dist.mean());

    // Variance = 2k = 14
    try expectEqual(14.0, dist.variance());
}

test "ChiSquared distribution: relationship to Gamma" {
    // χ²(k) should equal Gamma(k/2, 1/2)
    const dist_chi2 = try ChiSquared(f64).init(6);
    const dist_gamma = try Gamma(f64).init(3.0, 0.5); // k=6 → α=3, β=0.5

    const x = 4.0;

    // PDF should match
    try expectApproxEqRel(dist_gamma.pdf(x), dist_chi2.pdf(x), 1e-10);

    // CDF should match
    try expectApproxEqRel(dist_gamma.cdf(x), dist_chi2.cdf(x), 1e-10);

    // Mean should match
    try expectApproxEqRel(dist_gamma.mean(), dist_chi2.mean(), 1e-10);

    // Variance should match
    try expectApproxEqRel(dist_gamma.variance(), dist_chi2.variance(), 1e-10);
}

test "ChiSquared distribution: logpdf" {
    const dist = try ChiSquared(f64).init(5);

    const x = 3.0;
    const pdf_val = dist.pdf(x);
    const logpdf_val = dist.logpdf(x);

    // log(pdf(x)) should equal logpdf(x)
    try expectApproxEqRel(@log(pdf_val), logpdf_val, 1e-10);

    // Negative x should return -inf
    try expectEqual(-math.inf(f64), dist.logpdf(-1.0));
}

test "ChiSquared distribution: f32 precision" {
    const dist = try ChiSquared(f32).init(4);

    // PDF at x=2 for χ²(4)
    const pdf_val = dist.pdf(2.0);
    try testing.expect(pdf_val > 0.0);

    // Mean = 4
    try expectEqual(@as(f32, 4.0), dist.mean());

    // Variance = 8
    try expectEqual(@as(f32, 8.0), dist.variance());
}

// ============================================================================
// Student's t-Distribution Tests
// ============================================================================

test "StudentT distribution: initialization" {
    // Valid initialization
    const dist1 = try StudentT(f64).init(5.0);
    try expectEqual(@as(f64, 5.0), dist1.nu);

    const dist2 = try StudentT(f64).init(1.0);
    try expectEqual(@as(f64, 1.0), dist2.nu);

    // Invalid parameters
    try testing.expectError(error.InvalidParameter, StudentT(f64).init(0.0));
    try testing.expectError(error.InvalidParameter, StudentT(f64).init(-1.0));
    try testing.expectError(error.InvalidParameter, StudentT(f64).init(math.inf(f64)));
    try testing.expectError(error.InvalidParameter, StudentT(f64).init(math.nan(f64)));
}

test "StudentT distribution: PDF at x=0 (mode)" {
    // PDF at mode x=0 for various degrees of freedom
    const dist1 = try StudentT(f64).init(1.0);
    const pdf1 = dist1.pdf(0.0);
    // t(1) at x=0: 1/π ≈ 0.3183
    try testing.expect(@abs(pdf1 - 0.3183) < 0.001);

    const dist5 = try StudentT(f64).init(5.0);
    const pdf5 = dist5.pdf(0.0);
    // t(5) at x=0: Γ(3)/√(5π)Γ(2.5) ≈ 0.3796
    try testing.expect(@abs(pdf5 - 0.3796) < 0.001);

    const dist30 = try StudentT(f64).init(30.0);
    const pdf30 = dist30.pdf(0.0);
    // t(30) at x=0: approaches N(0,1) at x=0 which is 1/√(2π) ≈ 0.3989
    try testing.expect(@abs(pdf30 - 0.3989) < 0.01);
}

test "StudentT distribution: PDF symmetry" {
    const dist = try StudentT(f64).init(5.0);

    // PDF should be symmetric around 0
    try expectApproxEqRel(dist.pdf(-1.0), dist.pdf(1.0), 1e-10);
    try expectApproxEqRel(dist.pdf(-2.5), dist.pdf(2.5), 1e-10);
    try expectApproxEqRel(dist.pdf(-0.5), dist.pdf(0.5), 1e-10);
}

test "StudentT distribution: CDF at x=0 (median)" {
    const dist = try StudentT(f64).init(5.0);

    // CDF at x=0 should be 0.5 (median)
    const cdf_zero = dist.cdf(0.0);
    try expectApproxEqRel(cdf_zero, 0.5, 1e-10);
}

test "StudentT distribution: CDF symmetry" {
    const dist = try StudentT(f64).init(5.0);

    // CDF(-x) + CDF(x) should equal 1 (symmetry)
    const x = 1.5;
    const cdf_neg = dist.cdf(-x);
    const cdf_pos = dist.cdf(x);
    try expectApproxEqRel(cdf_neg + cdf_pos, 1.0, 1e-10);
}

test "StudentT distribution: CDF boundary values" {
    const dist = try StudentT(f64).init(5.0);

    // CDF at large positive x should approach 1
    const cdf_large = dist.cdf(10.0);
    try testing.expect(cdf_large > 0.999);

    // CDF at large negative x should approach 0
    const cdf_small = dist.cdf(-10.0);
    try testing.expect(cdf_small < 0.001);
}

test "StudentT distribution: quantile at median" {
    const dist = try StudentT(f64).init(5.0);

    // Quantile at p=0.5 should be 0 (median)
    const q = try dist.quantile(0.5);
    try expectApproxEqRel(q, 0.0, 1e-10);
}

test "StudentT distribution: quantile roundtrip" {
    const dist = try StudentT(f64).init(5.0);

    // Test quantile -> CDF roundtrip
    const p_values = [_]f64{ 0.05, 0.25, 0.5, 0.75, 0.95 };
    for (p_values) |p| {
        const q = try dist.quantile(p);
        const cdf_q = dist.cdf(q);
        try expectApproxEqRel(cdf_q, p, 1e-6);
    }
}

test "StudentT distribution: quantile errors" {
    const dist = try StudentT(f64).init(5.0);

    // Invalid probabilities
    try testing.expectError(error.InvalidProbability, dist.quantile(-0.1));
    try testing.expectError(error.InvalidProbability, dist.quantile(1.1));

    // Boundary cases
    const q0 = try dist.quantile(0.0);
    try expectEqual(-math.inf(f64), q0);

    const q1 = try dist.quantile(1.0);
    try expectEqual(math.inf(f64), q1);
}

test "StudentT distribution: sampling mean validation" {
    const dist = try StudentT(f64).init(5.0); // ν=5 > 1, mean should be 0
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    const n = 10000;
    var sum: f64 = 0.0;
    for (0..n) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n));

    // Sample mean should be close to theoretical mean (0)
    // With n=10000, standard error ≈ √(variance/n) = √(5/3/10000) ≈ 0.013
    // Using 5 standard errors for 5-sigma confidence
    try testing.expect(@abs(sample_mean - 0.0) < 0.1);
}

test "StudentT distribution: sampling variance validation" {
    const dist = try StudentT(f64).init(5.0); // ν=5 > 2, variance = 5/(5-2) = 1.667
    var prng = std.Random.DefaultPrng.init(54321);
    const rng = prng.random();

    const n = 10000;
    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    for (0..n) |_| {
        const x = dist.sample(rng);
        sum += x;
        sum_sq += x * x;
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n));
    const sample_var = (sum_sq / @as(f64, @floatFromInt(n))) - (sample_mean * sample_mean);
    const theoretical_var = dist.variance();

    // Sample variance should be close to theoretical variance (5/3 ≈ 1.667)
    // With large n, relative error should be small
    try testing.expect(@abs(sample_var - theoretical_var) / theoretical_var < 0.15);
}

test "StudentT distribution: logpdf" {
    const dist = try StudentT(f64).init(5.0);

    const x = 1.5;
    const pdf_val = dist.pdf(x);
    const logpdf_val = dist.logpdf(x);

    // logpdf should equal log(pdf)
    try expectApproxEqRel(logpdf_val, @log(pdf_val), 1e-10);
}

test "StudentT distribution: mean" {
    // Mean is 0 for ν > 1
    const dist1 = try StudentT(f64).init(2.0);
    try expectEqual(@as(f64, 0.0), dist1.mean());

    const dist2 = try StudentT(f64).init(10.0);
    try expectEqual(@as(f64, 0.0), dist2.mean());

    // Mean is undefined (NaN) for ν ≤ 1
    const dist_undef = try StudentT(f64).init(0.5);
    try testing.expect(math.isNan(dist_undef.mean()));
}

test "StudentT distribution: variance" {
    // Variance = ν/(ν-2) for ν > 2
    const dist5 = try StudentT(f64).init(5.0);
    const var5 = dist5.variance();
    try expectApproxEqRel(var5, 5.0 / 3.0, 1e-10); // 5/(5-2) = 1.667

    const dist10 = try StudentT(f64).init(10.0);
    const var10 = dist10.variance();
    try expectApproxEqRel(var10, 10.0 / 8.0, 1e-10); // 10/(10-2) = 1.25

    // Variance is infinite for 1 < ν ≤ 2
    const dist2 = try StudentT(f64).init(2.0);
    try expectEqual(math.inf(f64), dist2.variance());

    const dist1_5 = try StudentT(f64).init(1.5);
    try expectEqual(math.inf(f64), dist1_5.variance());

    // Variance is undefined (NaN) for ν ≤ 1
    const dist0_5 = try StudentT(f64).init(0.5);
    try testing.expect(math.isNan(dist0_5.variance()));
}

test "StudentT distribution: convergence to normal" {
    // As ν → ∞, t(ν) → N(0,1)
    const dist_large = try StudentT(f64).init(100.0);
    const normal = try Normal(f64).init(0.0, 1.0);

    // Compare PDF at various points
    const x_values = [_]f64{ 0.0, 0.5, 1.0, 1.5, 2.0 };
    for (x_values) |x| {
        const t_pdf = dist_large.pdf(x);
        const n_pdf = normal.pdf(x);
        // t(100) converges to N(0,1) — within 2% at tail (x=2)
        try expectApproxEqRel(t_pdf, n_pdf, 0.02);
    }

    // Compare CDF at various points
    for (x_values) |x| {
        const t_cdf = dist_large.cdf(x);
        const n_cdf = normal.cdf(x);
        try expectApproxEqRel(t_cdf, n_cdf, 0.02);
    }
}

test "StudentT distribution: f32 precision" {
    const dist = try StudentT(f32).init(5.0);

    // PDF at x=0
    const pdf_val = dist.pdf(0.0);
    try testing.expect(pdf_val > 0.35 and pdf_val < 0.40);

    // CDF at x=0 (median)
    const cdf_val = dist.cdf(0.0);
    try expectApproxEqRel(cdf_val, @as(f32, 0.5), 1e-6);

    // Mean = 0
    try expectEqual(@as(f32, 0.0), dist.mean());

    // Variance = 5/3
    try expectApproxEqRel(dist.variance(), @as(f32, 5.0 / 3.0), 1e-6);
}

// ============================================================================
// F Distribution Tests
// ============================================================================

test "F distribution: init" {
    // Valid parameters
    const dist = try FDistribution(f64).init(5.0, 10.0);
    try expectEqual(@as(f64, 5.0), dist.d1);
    try expectEqual(@as(f64, 10.0), dist.d2);

    // Invalid parameters
    try testing.expectError(error.InvalidParameter, FDistribution(f64).init(-1.0, 10.0));
    try testing.expectError(error.InvalidParameter, FDistribution(f64).init(5.0, 0.0));
    try testing.expectError(error.InvalidParameter, FDistribution(f64).init(math.nan(f64), 10.0));
    try testing.expectError(error.InvalidParameter, FDistribution(f64).init(5.0, math.inf(f64)));
}

test "F distribution: PDF" {
    // F(5, 10) distribution
    const dist = try FDistribution(f64).init(5.0, 10.0);

    // PDF should be 0 at x=0 and x<0
    try expectEqual(@as(f64, 0.0), dist.pdf(0.0));
    try expectEqual(@as(f64, 0.0), dist.pdf(-1.0));

    // PDF at x=1: f(1; 5,10) = (5/10)^2.5 * 1^1.5 * 1.5^(-7.5) / B(2.5,5) ≈ 0.4955
    const pdf1 = dist.pdf(1.0);
    try expectApproxEqRel(pdf1, 0.4955, 0.01);

    // PDF is positive for x > 0
    try testing.expect(dist.pdf(0.5) > 0.0);
    try testing.expect(dist.pdf(2.0) > 0.0);

    // Mode of F(d1, d2) is at (d1-2)/d1 × d2/(d2+2) for d1 > 2
    // Mode ≈ 0.6 for F(5, 10)
    const mode_x = ((dist.d1 - 2.0) / dist.d1) * (dist.d2 / (dist.d2 + 2.0));
    const pdf_mode = dist.pdf(mode_x);
    const pdf_nearby = dist.pdf(mode_x + 0.1);
    try testing.expect(pdf_mode >= pdf_nearby); // PDF should be maximum at mode
}

test "F distribution: CDF" {
    const dist = try FDistribution(f64).init(5.0, 10.0);

    // CDF at boundaries
    try expectEqual(@as(f64, 0.0), dist.cdf(0.0));
    try expectEqual(@as(f64, 0.0), dist.cdf(-1.0));

    // CDF at x=1: F(1; 5, 10) = 1 - I_(2/3)(5, 2.5) ≈ 0.5348
    const cdf1 = dist.cdf(1.0);
    try expectApproxEqRel(cdf1, 0.5348, 0.01);

    // CDF should be monotonically increasing
    try testing.expect(dist.cdf(0.5) < dist.cdf(1.0));
    try testing.expect(dist.cdf(1.0) < dist.cdf(2.0));
    try testing.expect(dist.cdf(2.0) < dist.cdf(5.0));

    // CDF approaches 1 as x → ∞
    try testing.expect(dist.cdf(100.0) > 0.99);
}

test "F distribution: quantile" {
    const dist = try FDistribution(f64).init(5.0, 10.0);

    // Quantile at boundaries
    try expectEqual(@as(f64, 0.0), try dist.quantile(0.0));
    try expectEqual(math.inf(f64), try dist.quantile(1.0));

    // Invalid probabilities
    try testing.expectError(error.InvalidProbability, dist.quantile(-0.1));
    try testing.expectError(error.InvalidProbability, dist.quantile(1.1));

    // Roundtrip: cdf(quantile(p)) ≈ p
    const p_values = [_]f64{ 0.1, 0.25, 0.5, 0.75, 0.9, 0.95 };
    for (p_values) |p| {
        const x = try dist.quantile(p);
        const p_check = dist.cdf(x);
        try expectApproxEqRel(p_check, p, 1e-6);
    }

    // 95th percentile for F(5, 10) ≈ 3.326 (critical value for α=0.05)
    const q95 = try dist.quantile(0.95);
    try expectApproxEqRel(q95, 3.326, 0.01);
}

test "F distribution: sampling" {
    const dist = try FDistribution(f64).init(5.0, 10.0);
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Generate samples and verify mean
    // Mean of F(d1, d2) = d2/(d2-2) = 10/8 = 1.25 for d2 > 2
    const n_samples = 10000;
    var sum: f64 = 0.0;
    for (0..n_samples) |_| {
        const x = dist.sample(rng);
        try testing.expect(x >= 0.0); // F-distribution is non-negative
        sum += x;
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));
    const theoretical_mean = dist.mean();
    // Allow 5% relative error due to sampling variance
    try expectApproxEqRel(sample_mean, theoretical_mean, 0.05);
}

test "F distribution: logpdf" {
    const dist = try FDistribution(f64).init(5.0, 10.0);

    // logpdf should be -inf at x <= 0
    try expectEqual(-math.inf(f64), dist.logpdf(0.0));
    try expectEqual(-math.inf(f64), dist.logpdf(-1.0));

    // logpdf(x) = log(pdf(x))
    const x_values = [_]f64{ 0.5, 1.0, 1.5, 2.0 };
    for (x_values) |x| {
        const logpdf_val = dist.logpdf(x);
        const pdf_val = dist.pdf(x);
        const expected_logpdf = @log(pdf_val);
        try expectApproxEqRel(logpdf_val, expected_logpdf, 1e-10);
    }
}

test "F distribution: mean" {
    // Mean = d2/(d2-2) for d2 > 2
    const dist1 = try FDistribution(f64).init(5.0, 10.0);
    const mean1 = dist1.mean();
    try expectApproxEqRel(mean1, 10.0 / 8.0, 1e-10); // 10/(10-2) = 1.25

    const dist2 = try FDistribution(f64).init(3.0, 6.0);
    const mean2 = dist2.mean();
    try expectApproxEqRel(mean2, 6.0 / 4.0, 1e-10); // 6/(6-2) = 1.5

    // Mean is undefined for d2 ≤ 2
    const dist_undef = try FDistribution(f64).init(5.0, 2.0);
    try testing.expect(math.isNan(dist_undef.mean()));

    const dist_undef2 = try FDistribution(f64).init(5.0, 1.5);
    try testing.expect(math.isNan(dist_undef2.mean()));
}

test "F distribution: variance" {
    // Variance = 2×d2²×(d1+d2-2)/(d1×(d2-2)²×(d2-4)) for d2 > 4
    const dist = try FDistribution(f64).init(5.0, 10.0);
    const var_val = dist.variance();
    // Var = 2×100×(5+10-2)/(5×64×6) = 2×100×13/(5×64×6) = 2600/1920 ≈ 1.354
    const expected_var = (2.0 * 100.0 * 13.0) / (5.0 * 64.0 * 6.0);
    try expectApproxEqRel(var_val, expected_var, 1e-10);

    // Variance is infinite for 2 < d2 ≤ 4
    const dist_inf = try FDistribution(f64).init(5.0, 4.0);
    try expectEqual(math.inf(f64), dist_inf.variance());

    const dist_inf2 = try FDistribution(f64).init(5.0, 3.0);
    try expectEqual(math.inf(f64), dist_inf2.variance());

    // Variance is undefined (NaN) for d2 ≤ 2
    const dist_undef = try FDistribution(f64).init(5.0, 2.0);
    try testing.expect(math.isNan(dist_undef.variance()));
}

test "F distribution: relationship to chi-squared" {
    // If X ~ χ²(d1) and Y ~ χ²(d2), then F = (X/d1)/(Y/d2) ~ F(d1, d2)
    const d1: f64 = 5.0;
    const d2: f64 = 10.0;

    const f_dist = try FDistribution(f64).init(d1, d2);
    const chi1 = try ChiSquared(f64).init(@intFromFloat(d1));
    const chi2 = try ChiSquared(f64).init(@intFromFloat(d2));

    var prng = std.Random.DefaultPrng.init(123);
    const rng = prng.random();

    // Generate F samples via chi-squared ratio and direct sampling
    const n_samples = 1000;
    var f_direct_sum: f64 = 0.0;
    var f_derived_sum: f64 = 0.0;

    for (0..n_samples) |_| {
        // Direct F sampling
        const f_sample = f_dist.sample(rng);
        f_direct_sum += f_sample;

        // Derived from chi-squared ratio
        const x = chi1.sample(rng);
        const y = chi2.sample(rng);
        const f_derived = (x / d1) / (y / d2);
        f_derived_sum += f_derived;
    }

    const f_direct_mean = f_direct_sum / @as(f64, @floatFromInt(n_samples));
    const f_derived_mean = f_derived_sum / @as(f64, @floatFromInt(n_samples));

    // Both should be close to theoretical mean
    const theoretical_mean = d2 / (d2 - 2.0);
    try expectApproxEqRel(f_direct_mean, theoretical_mean, 0.1);
    try expectApproxEqRel(f_derived_mean, theoretical_mean, 0.1);
}

test "F distribution: symmetry property" {
    // F(d1, d2) and 1/F(d2, d1) have the same distribution
    const d1: f64 = 5.0;
    const d2: f64 = 10.0;

    const f_dist = try FDistribution(f64).init(d1, d2);
    const f_inv_dist = try FDistribution(f64).init(d2, d1);

    // For F(d1, d2), P(X > x) = P(1/Y < 1/x) where Y ~ F(d2, d1)
    // Therefore: 1 - F_d1,d2(x) = F_d2,d1(1/x)
    const x_values = [_]f64{ 0.5, 1.0, 2.0, 3.0 };
    for (x_values) |x| {
        const cdf_forward = f_dist.cdf(x);
        const cdf_inverse = f_inv_dist.cdf(1.0 / x);
        const sf_forward = 1.0 - cdf_forward; // Survival function
        try expectApproxEqRel(sf_forward, cdf_inverse, 0.01);
    }
}

test "F distribution: f32 precision" {
    const dist = try FDistribution(f32).init(5.0, 10.0);

    // PDF at x=1: ≈ 0.4955
    const pdf_val = dist.pdf(1.0);
    try expectApproxEqRel(pdf_val, @as(f32, 0.4955), 1e-3);

    // CDF at x=1: ≈ 0.5348
    const cdf_val = dist.cdf(1.0);
    try expectApproxEqRel(cdf_val, @as(f32, 0.5348), 1e-3);

    // Mean = 10/8 = 1.25
    try expectApproxEqRel(dist.mean(), @as(f32, 1.25), 1e-6);

    // Variance
    const expected_var = @as(f32, (2.0 * 100.0 * 13.0) / (5.0 * 64.0 * 6.0));
    try expectApproxEqRel(dist.variance(), expected_var, 1e-5);
}

test "distributions: memory safety" {
    const allocator = testing.allocator;
    _ = allocator;

    // Test multiple init/deinit cycles (no memory allocation in distributions)
    for (0..10) |_| {
        const normal = try Normal(f64).init(0.0, 1.0);
        try expectApproxEqRel(normal.pdf(0.0), 0.3989422804014327, 1e-10);
        _ = normal.cdf(0.5);

        const uniform = try Uniform(f64).init(0.0, 1.0);
        try expectApproxEqRel(uniform.pdf(0.5), 1.0, 1e-10);

        const exp = try Exponential(f64).init(1.0);
        _ = exp.pdf(0.5);

        const gamma = try Gamma(f64).init(2.0, 1.0);
        _ = gamma.pdf(1.0);

        const beta = try Beta(f64).init(2.0, 5.0);
        _ = beta.pdf(0.3);

        const poisson = try Poisson(f64).init(3.0);
        _ = poisson.pmf(2);

        const binomial = try Binomial(f64).init(10, 0.5);
        _ = binomial.pmf(5);

        const chi2 = try ChiSquared(f64).init(5);
        _ = chi2.pdf(2.0);

        const student_t = try StudentT(f64).init(5.0);
        _ = student_t.pdf(1.0);

        const f_dist = try FDistribution(f64).init(5.0, 10.0);
        _ = f_dist.pdf(1.0);

        const laplace = try Laplace(f64).init(0.0, 1.0);
        _ = laplace.pdf(0.5);
    }
}

// ============================================================================
// Laplace (Double Exponential) Distribution
// ============================================================================

/// Laplace (Double Exponential) distribution Laplace(μ, b)
///
/// Probability density function (PDF):
///   f(x) = (1/(2b)) × exp(-|x-μ|/b)
///
/// Cumulative distribution function (CDF):
///   F(x) = 0.5 × exp((x-μ)/b)           if x < μ
///   F(x) = 1 - 0.5 × exp(-(x-μ)/b)     if x ≥ μ
///
/// Parameters:
///   - location (μ): Location parameter (-∞ to +∞)
///   - scale (b): Scale parameter (b > 0)
///
/// Properties:
///   - Mean = μ
///   - Variance = 2b²
///   - Mode = μ
///   - Symmetric around μ
///   - Heavier tails than Normal distribution
///
/// Applications:
///   - Lasso regression (Laplace prior on coefficients)
///   - Differential privacy (Laplace mechanism)
///   - Robust statistics (heavy-tailed alternative to Normal)
///   - Signal processing (modeling noise)
///
/// Time: O(1) for all operations
pub fn Laplace(comptime T: type) type {
    return struct {
        location: T,
        scale: T,

        const Self = @This();

        /// Create a Laplace distribution with given location and scale
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(location: T, scale: T) DistributionError!Self {
            if (scale <= 0.0) return error.InvalidParameter;
            if (!math.isFinite(location) or !math.isFinite(scale)) return error.InvalidParameter;
            return Self{ .location = location, .scale = scale };
        }

        /// Probability density function (PDF) at x
        ///
        /// f(x) = (1/(2b)) × exp(-|x-μ|/b)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn pdf(self: Self, x: T) T {
            const diff = @abs(x - self.location);
            return (1.0 / (2.0 * self.scale)) * @exp(-diff / self.scale);
        }

        /// Cumulative distribution function (CDF) at x
        ///
        /// F(x) = P(X ≤ x)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn cdf(self: Self, x: T) T {
            if (x < self.location) {
                // x < μ: F(x) = 0.5 × exp((x-μ)/b)
                return 0.5 * @exp((x - self.location) / self.scale);
            } else {
                // x ≥ μ: F(x) = 1 - 0.5 × exp(-(x-μ)/b)
                return 1.0 - 0.5 * @exp(-(x - self.location) / self.scale);
            }
        }

        /// Quantile function (inverse CDF) - returns x such that P(X ≤ x) = p
        ///
        /// Uses closed-form inversion of CDF
        ///
        /// Time: O(1) | Space: O(1)
        pub fn quantile(self: Self, p: T) DistributionError!T {
            if (p < 0.0 or p > 1.0) return error.InvalidProbability;
            if (p == 0.0) return -math.inf(T);
            if (p == 1.0) return math.inf(T);

            if (p < 0.5) {
                // Left tail: x = μ + b × ln(2p)
                return self.location + self.scale * @log(2.0 * p);
            } else {
                // Right tail: x = μ - b × ln(2(1-p))
                return self.location - self.scale * @log(2.0 * (1.0 - p));
            }
        }

        /// Generate a random sample from this distribution
        ///
        /// Uses inverse transform method: X = μ - b × sgn(U - 0.5) × ln(1 - 2|U - 0.5|)
        /// where U ~ Uniform(0,1)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sample(self: Self, rng: std.Random) T {
            const u = rng.float(T);
            return self.quantile(u) catch unreachable; // u is always valid probability
        }

        /// Log probability density function (log PDF) at x
        ///
        /// log f(x) = -ln(2b) - |x-μ|/b
        ///
        /// More numerically stable than log(pdf(x))
        ///
        /// Time: O(1) | Space: O(1)
        pub fn logpdf(self: Self, x: T) T {
            const diff = @abs(x - self.location);
            return -@log(2.0 * self.scale) - diff / self.scale;
        }

        /// Mean of the distribution
        ///
        /// E[X] = μ
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mean(self: Self) T {
            return self.location;
        }

        /// Variance of the distribution
        ///
        /// Var(X) = 2b²
        ///
        /// Time: O(1) | Space: O(1)
        pub fn variance(self: Self) T {
            return 2.0 * self.scale * self.scale;
        }

        /// Mode of the distribution
        ///
        /// Mode = μ (peak at location)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mode(self: Self) T {
            return self.location;
        }

        /// Median of the distribution
        ///
        /// Median = μ (symmetric distribution)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn median(self: Self) T {
            return self.location;
        }

        /// Survival function (complement of CDF)
        ///
        /// S(x) = P(X > x) = 1 - F(x)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sf(self: Self, x: T) T {
            return 1.0 - self.cdf(x);
        }

        /// Mean absolute deviation (MAD)
        ///
        /// MAD = b (scale parameter)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mad(self: Self) T {
            return self.scale;
        }

        /// Assert that parameters are valid: scale > 0 and finite, location finite.
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.scale <= 0.0 or !math.isFinite(self.scale)) return DistributionError.InvalidParameter;
            if (!math.isFinite(self.location)) return DistributionError.InvalidParameter;
        }
    };
}

/// Weibull distribution — continuous distribution for reliability and survival analysis
///
/// The Weibull distribution is a continuous probability distribution with shape parameter k
/// and scale parameter λ. It generalizes the exponential distribution and is widely used
/// in reliability engineering, survival analysis, and extreme value theory.
///
/// Parameters:
/// - shape (k > 0): determines the distribution's form (k < 1: decreasing hazard, k = 1: constant, k > 1: increasing)
/// - scale (λ > 0): stretches or compresses the distribution
///
/// Special cases:
/// - k = 1: Exponential distribution with rate 1/λ
/// - k = 2: Rayleigh distribution
/// - k = 3.4: approximates Normal distribution
///
/// Domain: [0, ∞)
///
/// Use cases:
/// - Reliability engineering (time-to-failure modeling)
/// - Survival analysis (time-to-event data)
/// - Wind speed distribution
/// - Extreme value analysis
pub fn Weibull(comptime T: type) type {
    return struct {
        shape: T,
        scale: T,

        const Self = @This();

        /// Create a Weibull distribution with given shape and scale parameters
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(shape: T, scale: T) DistributionError!Self {
            if (shape <= 0.0) return error.InvalidParameter;
            if (scale <= 0.0) return error.InvalidParameter;
            if (!math.isFinite(shape) or !math.isFinite(scale)) return error.InvalidParameter;
            return Self{ .shape = shape, .scale = scale };
        }

        /// Probability density function (PDF) at x
        ///
        /// f(x) = (k/λ)(x/λ)^(k-1) × exp(-(x/λ)^k) for x ≥ 0
        ///
        /// Time: O(1) | Space: O(1)
        pub fn pdf(self: Self, x: T) T {
            if (x < 0.0) return 0.0;
            if (x == 0.0) {
                // PDF at x=0: k < 1 → ∞, k = 1 → 1/λ, k > 1 → 0
                if (self.shape < 1.0) return math.inf(T);
                if (self.shape == 1.0) return 1.0 / self.scale;
                return 0.0;
            }

            const x_scaled = x / self.scale;
            const x_pow_k = math.pow(T, x_scaled, self.shape);
            const x_pow_k_minus_1 = math.pow(T, x_scaled, self.shape - 1.0);

            return (self.shape / self.scale) * x_pow_k_minus_1 * @exp(-x_pow_k);
        }

        /// Cumulative distribution function (CDF) at x
        ///
        /// F(x) = 1 - exp(-(x/λ)^k) for x ≥ 0
        ///
        /// Time: O(1) | Space: O(1)
        pub fn cdf(self: Self, x: T) T {
            if (x <= 0.0) return 0.0;

            const x_scaled = x / self.scale;
            const x_pow_k = math.pow(T, x_scaled, self.shape);
            return 1.0 - @exp(-x_pow_k);
        }

        /// Quantile function (inverse CDF) - returns x such that P(X ≤ x) = p
        ///
        /// Q(p) = λ × (-ln(1-p))^(1/k)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn quantile(self: Self, p: T) DistributionError!T {
            if (p < 0.0 or p > 1.0) return error.InvalidProbability;
            if (p == 0.0) return 0.0;
            if (p == 1.0) return math.inf(T);

            return self.scale * math.pow(T, -@log(1.0 - p), 1.0 / self.shape);
        }

        /// Generate a random sample from this distribution
        ///
        /// Uses inverse transform method via quantile function
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sample(self: Self, rng: std.Random) T {
            const u = rng.float(T);
            return self.quantile(u) catch unreachable; // u is always valid probability
        }

        /// Log probability density function (log PDF) at x
        ///
        /// log f(x) = ln(k/λ) + (k-1)×ln(x/λ) - (x/λ)^k
        ///
        /// More numerically stable than log(pdf(x))
        ///
        /// Time: O(1) | Space: O(1)
        pub fn logpdf(self: Self, x: T) T {
            if (x < 0.0) return -math.inf(T);
            if (x == 0.0) {
                if (self.shape < 1.0) return math.inf(T);
                if (self.shape == 1.0) return -@log(self.scale);
                return -math.inf(T);
            }

            const x_scaled = x / self.scale;
            const log_x_scaled = @log(x_scaled);
            const x_pow_k = math.pow(T, x_scaled, self.shape);

            return @log(self.shape / self.scale) + (self.shape - 1.0) * log_x_scaled - x_pow_k;
        }

        /// Mean of the distribution
        ///
        /// E[X] = λΓ(1 + 1/k)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mean(self: Self) T {
            const gamma_val = @exp(logGamma(1.0 + 1.0 / self.shape));
            return self.scale * gamma_val;
        }

        /// Variance of the distribution
        ///
        /// Var(X) = λ²[Γ(1 + 2/k) - Γ²(1 + 1/k)]
        ///
        /// Time: O(1) | Space: O(1)
        pub fn variance(self: Self) T {
            const gamma_1_plus_1_over_k = @exp(logGamma(1.0 + 1.0 / self.shape));
            const gamma_1_plus_2_over_k = @exp(logGamma(1.0 + 2.0 / self.shape));
            return self.scale * self.scale * (gamma_1_plus_2_over_k - gamma_1_plus_1_over_k * gamma_1_plus_1_over_k);
        }

        /// Mode of the distribution
        ///
        /// Mode = λ((k-1)/k)^(1/k) for k > 1, otherwise 0
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mode(self: Self) T {
            if (self.shape <= 1.0) return 0.0;
            return self.scale * math.pow(T, (self.shape - 1.0) / self.shape, 1.0 / self.shape);
        }

        /// Median of the distribution
        ///
        /// Median = λ(ln 2)^(1/k)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn median(self: Self) T {
            return self.scale * math.pow(T, @log(2.0), 1.0 / self.shape);
        }

        /// Survival function (complement of CDF)
        ///
        /// S(x) = P(X > x) = exp(-(x/λ)^k)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sf(self: Self, x: T) T {
            if (x <= 0.0) return 1.0;
            const x_scaled = x / self.scale;
            const x_pow_k = math.pow(T, x_scaled, self.shape);
            return @exp(-x_pow_k);
        }

        /// Hazard function (failure rate)
        ///
        /// h(x) = (k/λ)(x/λ)^(k-1)
        ///
        /// Describes instantaneous failure rate at time x
        ///
        /// Time: O(1) | Space: O(1)
        pub fn hazard(self: Self, x: T) T {
            if (x < 0.0) return 0.0;
            if (x == 0.0) {
                if (self.shape < 1.0) return math.inf(T);
                if (self.shape == 1.0) return 1.0 / self.scale;
                return 0.0;
            }

            const x_scaled = x / self.scale;
            const x_pow_k_minus_1 = math.pow(T, x_scaled, self.shape - 1.0);
            return (self.shape / self.scale) * x_pow_k_minus_1;
        }

        /// Assert that parameters are valid: shape > 0, scale > 0, both finite.
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.shape <= 0.0 or !math.isFinite(self.shape)) return DistributionError.InvalidParameter;
            if (self.scale <= 0.0 or !math.isFinite(self.scale)) return DistributionError.InvalidParameter;
        }
    };
}

// ============================================================================
// Tests - Laplace Distribution
// ============================================================================

test "Laplace: init with valid parameters" {
    const dist = try Laplace(f64).init(0.0, 1.0);
    try testing.expectEqual(@as(f64, 0.0), dist.location);
    try testing.expectEqual(@as(f64, 1.0), dist.scale);

    const dist2 = try Laplace(f64).init(5.0, 2.5);
    try testing.expectEqual(@as(f64, 5.0), dist2.location);
    try testing.expectEqual(@as(f64, 2.5), dist2.scale);
}

test "Laplace: init with invalid parameters" {
    // Negative scale
    try testing.expectError(error.InvalidParameter, Laplace(f64).init(0.0, -1.0));

    // Zero scale
    try testing.expectError(error.InvalidParameter, Laplace(f64).init(0.0, 0.0));

    // Infinite location
    try testing.expectError(error.InvalidParameter, Laplace(f64).init(math.inf(f64), 1.0));

    // NaN scale
    try testing.expectError(error.InvalidParameter, Laplace(f64).init(0.0, math.nan(f64)));
}

test "Laplace: pdf at location (mode)" {
    const dist = try Laplace(f64).init(0.0, 1.0);

    // At mode x = μ = 0: f(0) = 1/(2b) = 1/2 = 0.5
    const pdf_at_mode = dist.pdf(0.0);
    try expectApproxEqRel(pdf_at_mode, 0.5, 1e-10);

    // With scale b = 2: f(0) = 1/(2×2) = 0.25
    const dist2 = try Laplace(f64).init(0.0, 2.0);
    try expectApproxEqRel(dist2.pdf(0.0), 0.25, 1e-10);
}

test "Laplace: pdf symmetry around location" {
    const dist = try Laplace(f64).init(0.0, 1.0);

    // f(μ + x) = f(μ - x) for any x
    const x_values = [_]f64{ 0.5, 1.0, 2.0, 3.0 };
    for (x_values) |x| {
        const pdf_pos = dist.pdf(x);
        const pdf_neg = dist.pdf(-x);
        try expectApproxEqRel(pdf_pos, pdf_neg, 1e-10);
    }

    // For shifted distribution μ = 5
    const dist2 = try Laplace(f64).init(5.0, 1.0);
    for (x_values) |x| {
        const pdf_pos = dist2.pdf(5.0 + x);
        const pdf_neg = dist2.pdf(5.0 - x);
        try expectApproxEqRel(pdf_pos, pdf_neg, 1e-10);
    }
}

test "Laplace: pdf manual calculation" {
    const dist = try Laplace(f64).init(0.0, 1.0);

    // At x = 1: f(1) = (1/2) × exp(-1) ≈ 0.1839
    const pdf_1 = dist.pdf(1.0);
    const expected_1 = 0.5 * @exp(-1.0);
    try expectApproxEqRel(pdf_1, expected_1, 1e-10);

    // At x = -2: f(-2) = (1/2) × exp(-2) ≈ 0.0677
    const pdf_minus2 = dist.pdf(-2.0);
    const expected_minus2 = 0.5 * @exp(-2.0);
    try expectApproxEqRel(pdf_minus2, expected_minus2, 1e-10);
}

test "Laplace: cdf at median" {
    const dist = try Laplace(f64).init(0.0, 1.0);

    // At median x = μ = 0: F(0) = 0.5
    const cdf_median = dist.cdf(0.0);
    try expectApproxEqRel(cdf_median, 0.5, 1e-10);

    // For shifted distribution μ = 3.5
    const dist2 = try Laplace(f64).init(3.5, 2.0);
    try expectApproxEqRel(dist2.cdf(3.5), 0.5, 1e-10);
}

test "Laplace: cdf boundary values" {
    const dist = try Laplace(f64).init(0.0, 1.0);

    // CDF should approach 0 as x → -∞ (use absolute comparison: value ≈ 2.27e-5)
    try testing.expect(dist.cdf(-10.0) < 1e-3);

    // CDF should approach 1 as x → +∞
    try testing.expect(dist.cdf(10.0) > 1.0 - 1e-3);

    // Monotonicity: F(x1) < F(x2) for x1 < x2
    try testing.expect(dist.cdf(-1.0) < dist.cdf(0.0));
    try testing.expect(dist.cdf(0.0) < dist.cdf(1.0));
}

test "Laplace: cdf manual calculation" {
    const dist = try Laplace(f64).init(0.0, 1.0);

    // At x = -1 (left tail): F(-1) = 0.5 × exp(-1) ≈ 0.1839
    const cdf_minus1 = dist.cdf(-1.0);
    const expected_minus1 = 0.5 * @exp(-1.0);
    try expectApproxEqRel(cdf_minus1, expected_minus1, 1e-10);

    // At x = 1 (right tail): F(1) = 1 - 0.5 × exp(-1) ≈ 0.8161
    const cdf_1 = dist.cdf(1.0);
    const expected_1 = 1.0 - 0.5 * @exp(-1.0);
    try expectApproxEqRel(cdf_1, expected_1, 1e-10);

    // Verify symmetry: F(μ + x) = 1 - F(μ - x)
    try expectApproxEqRel(cdf_1, 1.0 - cdf_minus1, 1e-10);
}

test "Laplace: quantile at p=0.5 (median)" {
    const dist = try Laplace(f64).init(0.0, 1.0);

    // Median: q(0.5) = μ = 0
    const q_median = try dist.quantile(0.5);
    try expectApproxEqRel(q_median, 0.0, 1e-10);

    // For shifted distribution
    const dist2 = try Laplace(f64).init(5.0, 2.0);
    const q_median2 = try dist2.quantile(0.5);
    try expectApproxEqRel(q_median2, 5.0, 1e-10);
}

test "Laplace: quantile roundtrip with cdf" {
    const dist = try Laplace(f64).init(0.0, 1.0);

    // For various probabilities, cdf(quantile(p)) should equal p
    const p_values = [_]f64{ 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99 };
    for (p_values) |p| {
        const q = try dist.quantile(p);
        const cdf_q = dist.cdf(q);
        try expectApproxEqRel(cdf_q, p, 1e-8);
    }
}

test "Laplace: quantile manual calculation" {
    const dist = try Laplace(f64).init(0.0, 1.0);

    // At p = 0.25 (left tail): q = μ + b × ln(2×0.25) = 0 + ln(0.5) ≈ -0.6931
    const q_25 = try dist.quantile(0.25);
    const expected_25 = @log(0.5);
    try expectApproxEqRel(q_25, expected_25, 1e-10);

    // At p = 0.75 (right tail): q = μ - b × ln(2×0.25) = 0 - ln(0.5) ≈ 0.6931
    const q_75 = try dist.quantile(0.75);
    const expected_75 = -@log(0.5);
    try expectApproxEqRel(q_75, expected_75, 1e-10);

    // Symmetry: q(0.75) = -q(0.25)
    try expectApproxEqRel(q_75, -q_25, 1e-10);
}

test "Laplace: quantile error handling" {
    const dist = try Laplace(f64).init(0.0, 1.0);

    // Invalid probabilities
    try testing.expectError(error.InvalidProbability, dist.quantile(-0.1));
    try testing.expectError(error.InvalidProbability, dist.quantile(1.5));

    // Boundary cases
    const q_0 = try dist.quantile(0.0);
    try testing.expect(math.isNegativeInf(q_0));

    const q_1 = try dist.quantile(1.0);
    try testing.expect(math.isPositiveInf(q_1));
}

test "Laplace: sampling mean validation" {
    const dist = try Laplace(f64).init(5.0, 2.0);
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    // Sample 10000 values and verify sample mean ≈ theoretical mean
    var sum: f64 = 0.0;
    const n = 10000;
    for (0..n) |_| {
        const sample = dist.sample(rng);
        sum += sample;
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n));

    // Sample mean should be close to μ = 5.0
    // Use larger tolerance due to sampling variability
    try expectApproxEqRel(sample_mean, 5.0, 0.05);
}

test "Laplace: sampling variance validation" {
    const dist = try Laplace(f64).init(0.0, 1.0);
    var prng = std.Random.DefaultPrng.init(54321);
    const rng = prng.random();

    // Sample values and compute variance
    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    const n = 10000;
    for (0..n) |_| {
        const sample = dist.sample(rng);
        sum += sample;
        sum_sq += sample * sample;
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n));
    const sample_var = (sum_sq / @as(f64, @floatFromInt(n))) - (sample_mean * sample_mean);

    // Theoretical variance = 2b² = 2×1² = 2.0
    try expectApproxEqRel(sample_var, 2.0, 0.1);
}

test "Laplace: logpdf consistency with log(pdf)" {
    const dist = try Laplace(f64).init(0.0, 1.0);

    const x_values = [_]f64{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    for (x_values) |x| {
        const logpdf_val = dist.logpdf(x);
        const log_of_pdf = @log(dist.pdf(x));
        try expectApproxEqRel(logpdf_val, log_of_pdf, 1e-10);
    }
}

test "Laplace: mean, variance, mode, median" {
    const dist = try Laplace(f64).init(3.0, 1.5);

    // Mean = μ = 3.0
    try expectApproxEqRel(dist.mean(), 3.0, 1e-10);

    // Variance = 2b² = 2×1.5² = 4.5
    try expectApproxEqRel(dist.variance(), 4.5, 1e-10);

    // Mode = μ = 3.0
    try expectApproxEqRel(dist.mode(), 3.0, 1e-10);

    // Median = μ = 3.0
    try expectApproxEqRel(dist.median(), 3.0, 1e-10);

    // MAD = b = 1.5
    try expectApproxEqRel(dist.mad(), 1.5, 1e-10);
}

test "Laplace: survival function" {
    const dist = try Laplace(f64).init(0.0, 1.0);

    // S(x) = 1 - F(x)
    const x_values = [_]f64{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    for (x_values) |x| {
        const sf_val = dist.sf(x);
        const expected = 1.0 - dist.cdf(x);
        try expectApproxEqRel(sf_val, expected, 1e-10);
    }
}

test "Laplace: comparison with Normal (heavier tails)" {
    const laplace = try Laplace(f64).init(0.0, 1.0);
    const normal = try Normal(f64).init(0.0, @sqrt(2.0)); // Same variance

    // At x = 5 (far tail), Laplace decays as exp(-5) while Normal decays as exp(-25/4)
    const laplace_tail = laplace.pdf(5.0);
    const normal_tail = normal.pdf(5.0);
    try testing.expect(laplace_tail > normal_tail);

    // Laplace has heavier tails than Normal with same variance
}

test "Laplace: f32 precision" {
    const dist = try Laplace(f32).init(0.0, 1.0);

    // PDF at mode
    try expectApproxEqRel(dist.pdf(0.0), @as(f32, 0.5), 1e-6);

    // CDF at median
    try expectApproxEqRel(dist.cdf(0.0), @as(f32, 0.5), 1e-6);

    // Mean and variance
    try expectApproxEqRel(dist.mean(), @as(f32, 0.0), 1e-6);
    try expectApproxEqRel(dist.variance(), @as(f32, 2.0), 1e-6);
}

test "Laplace: memory safety" {
    const allocator = testing.allocator;
    _ = allocator;

    // No allocation in Laplace distribution, just verify init/usage
    for (0..10) |_| {
        const dist = try Laplace(f64).init(0.0, 1.0);
        _ = dist.pdf(0.5);
        try expectApproxEqRel(dist.cdf(0.0), 0.5, 1e-10);
        _ = try dist.quantile(0.25);
    }
}

// ============================================================================
// Tests - Weibull Distribution
// ============================================================================

test "Weibull: init with valid parameters" {
    // Valid Weibull distribution
    const dist = try Weibull(f64).init(2.0, 1.0);
    try testing.expectEqual(@as(f64, 2.0), dist.shape);
    try testing.expectEqual(@as(f64, 1.0), dist.scale);
}

test "Weibull: init with invalid parameters" {
    // Negative shape
    try testing.expectError(error.InvalidParameter, Weibull(f64).init(-1.0, 1.0));

    // Zero shape
    try testing.expectError(error.InvalidParameter, Weibull(f64).init(0.0, 1.0));

    // Negative scale
    try testing.expectError(error.InvalidParameter, Weibull(f64).init(2.0, -1.0));

    // Zero scale
    try testing.expectError(error.InvalidParameter, Weibull(f64).init(2.0, 0.0));

    // Infinite parameters
    try testing.expectError(error.InvalidParameter, Weibull(f64).init(math.inf(f64), 1.0));
    try testing.expectError(error.InvalidParameter, Weibull(f64).init(1.0, math.inf(f64)));

    // NaN parameters
    try testing.expectError(error.InvalidParameter, Weibull(f64).init(math.nan(f64), 1.0));
    try testing.expectError(error.InvalidParameter, Weibull(f64).init(1.0, math.nan(f64)));
}

test "Weibull: pdf at mode for k=2 (Rayleigh)" {
    // Weibull(k=2, λ=1) is Rayleigh distribution
    // Mode at λ/√2 ≈ 0.7071
    const dist = try Weibull(f64).init(2.0, 1.0);
    const mode_x = dist.mode();
    try expectApproxEqRel(mode_x, 0.7071067811865476, 1e-10);

    // PDF at mode should be maximum
    const pdf_mode = dist.pdf(mode_x);
    const pdf_before = dist.pdf(mode_x - 0.1);
    const pdf_after = dist.pdf(mode_x + 0.1);
    try testing.expect(pdf_mode > pdf_before);
    try testing.expect(pdf_mode > pdf_after);
}

test "Weibull: pdf for k=1 (exponential)" {
    // Weibull(k=1, λ=2) = Exponential(rate=1/2)
    // PDF: f(x) = (1/2)exp(-x/2)
    const dist = try Weibull(f64).init(1.0, 2.0);

    // At x=0: f(0) = 1/λ = 0.5
    try expectApproxEqRel(dist.pdf(0.0), 0.5, 1e-10);

    // At x=2: f(2) = 0.5 × exp(-1) ≈ 0.18394
    try expectApproxEqRel(dist.pdf(2.0), 0.5 * @exp(-1.0), 1e-10);

    // At x=4: f(4) = 0.5 × exp(-2) ≈ 0.06767
    try expectApproxEqRel(dist.pdf(4.0), 0.5 * @exp(-2.0), 1e-10);
}

test "Weibull: pdf negative x returns 0" {
    const dist = try Weibull(f64).init(2.0, 1.0);
    try testing.expectEqual(@as(f64, 0.0), dist.pdf(-1.0));
    try testing.expectEqual(@as(f64, 0.0), dist.pdf(-10.0));
}

test "Weibull: cdf at median is 0.5" {
    const dist = try Weibull(f64).init(2.0, 1.0);
    const median_val = dist.median();
    try expectApproxEqRel(dist.cdf(median_val), 0.5, 1e-10);
}

test "Weibull: cdf boundary values" {
    const dist = try Weibull(f64).init(2.0, 1.0);

    // At x=0: CDF = 0
    try testing.expectEqual(@as(f64, 0.0), dist.cdf(0.0));

    // At large x: CDF → 1
    try expectApproxEqRel(dist.cdf(10.0), 1.0, 1e-10);

    // Negative x: CDF = 0
    try testing.expectEqual(@as(f64, 0.0), dist.cdf(-1.0));
}

test "Weibull: cdf manual calculation k=2, λ=1" {
    // Weibull(k=2, λ=1): F(x) = 1 - exp(-x²)
    const dist = try Weibull(f64).init(2.0, 1.0);

    // At x=1: F(1) = 1 - exp(-1) ≈ 0.6321
    try expectApproxEqRel(dist.cdf(1.0), 1.0 - @exp(-1.0), 1e-10);

    // At x=√2: F(√2) = 1 - exp(-2) ≈ 0.8647
    const sqrt2 = @sqrt(2.0);
    try expectApproxEqRel(dist.cdf(sqrt2), 1.0 - @exp(-2.0), 1e-10);
}

test "Weibull: cdf monotonicity" {
    const dist = try Weibull(f64).init(2.0, 1.0);

    // CDF should be strictly increasing
    var x: f64 = 0.0;
    var prev_cdf = dist.cdf(x);

    while (x <= 5.0) : (x += 0.5) {
        const curr_cdf = dist.cdf(x);
        try testing.expect(curr_cdf >= prev_cdf);
        prev_cdf = curr_cdf;
    }
}

test "Weibull: quantile at median" {
    const dist = try Weibull(f64).init(2.0, 1.0);
    const q_50 = try dist.quantile(0.5);
    // Median: λ(ln 2)^(1/k) = 1 × (ln 2)^0.5 ≈ 0.8326
    try expectApproxEqRel(q_50, math.pow(f64, @log(2.0), 0.5), 1e-10);
}

test "Weibull: quantile roundtrip with CDF" {
    const dist = try Weibull(f64).init(2.0, 1.0);

    // For various probabilities, quantile(p) should give x such that cdf(x) ≈ p
    const probs = [_]f64{ 0.1, 0.25, 0.5, 0.75, 0.9 };
    for (probs) |p| {
        const x = try dist.quantile(p);
        const p_back = dist.cdf(x);
        try expectApproxEqRel(p_back, p, 1e-9);
    }
}

test "Weibull: quantile error handling" {
    const dist = try Weibull(f64).init(2.0, 1.0);

    // Invalid probabilities
    try testing.expectError(error.InvalidProbability, dist.quantile(-0.1));
    try testing.expectError(error.InvalidProbability, dist.quantile(1.1));

    // Edge cases
    try testing.expectEqual(@as(f64, 0.0), try dist.quantile(0.0));
    try testing.expect(math.isInf(try dist.quantile(1.0)));
}

test "Weibull: quantile manual calculation k=2, λ=1" {
    // Weibull(k=2, λ=1): Q(p) = √(-ln(1-p))
    const dist = try Weibull(f64).init(2.0, 1.0);

    // At p=0.5: Q(0.5) = √(ln 2) ≈ 0.8326
    try expectApproxEqRel(try dist.quantile(0.5), @sqrt(@log(2.0)), 1e-10);

    // At p=0.9: Q(0.9) = √(-ln 0.1) ≈ 1.5174
    try expectApproxEqRel(try dist.quantile(0.9), @sqrt(-@log(0.1)), 1e-10);
}

test "Weibull: sample mean validation" {
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    // Weibull(k=2, λ=1): E[X] = Γ(1.5) ≈ 0.8862
    const dist = try Weibull(f64).init(2.0, 1.0);
    const expected_mean = dist.mean();

    var sum: f64 = 0.0;
    const n = 10000;
    for (0..n) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n));

    // Sample mean should be close to theoretical mean (within 5% for 10K samples)
    try expectApproxEqRel(sample_mean, expected_mean, 0.05);
}

test "Weibull: sample variance validation" {
    var prng = std.Random.DefaultPrng.init(67890);
    const rng = prng.random();

    // Weibull(k=2, λ=1): Var(X) = 1 × [Γ(2) - Γ²(1.5)] ≈ 0.2146
    const dist = try Weibull(f64).init(2.0, 1.0);
    const expected_var = dist.variance();
    const expected_mean = dist.mean();

    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    const n = 10000;
    for (0..n) |_| {
        const x = dist.sample(rng);
        sum += x;
        sum_sq += x * x;
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n));
    const sample_var = (sum_sq / @as(f64, @floatFromInt(n))) - (sample_mean * sample_mean);

    // Sample variance should be close to theoretical variance (within 10% for 10K samples)
    _ = expected_mean;
    try expectApproxEqRel(sample_var, expected_var, 0.10);
}

test "Weibull: samples non-negative" {
    var prng = std.Random.DefaultPrng.init(99999);
    const rng = prng.random();

    const dist = try Weibull(f64).init(2.0, 1.0);

    // All samples should be non-negative
    for (0..100) |_| {
        const x = dist.sample(rng);
        try testing.expect(x >= 0.0);
    }
}

test "Weibull: logpdf consistency with log(pdf)" {
    const dist = try Weibull(f64).init(2.0, 1.0);

    const x_vals = [_]f64{ 0.5, 1.0, 1.5, 2.0 };
    for (x_vals) |x| {
        const log_pdf_val = dist.logpdf(x);
        const pdf_val = dist.pdf(x);
        const log_pdf_from_pdf = @log(pdf_val);

        try expectApproxEqRel(log_pdf_val, log_pdf_from_pdf, 1e-10);
    }
}

test "Weibull: mean calculation" {
    // Weibull(k=1, λ=2) = Exponential(rate=0.5): E[X] = λ = 2
    const dist1 = try Weibull(f64).init(1.0, 2.0);
    try expectApproxEqRel(dist1.mean(), 2.0, 1e-10);

    // Weibull(k=2, λ=1): E[X] = Γ(1.5) = √π/2 ≈ 0.8862
    const dist2 = try Weibull(f64).init(2.0, 1.0);
    try expectApproxEqRel(dist2.mean(), @sqrt(math.pi) / 2.0, 1e-10);
}

test "Weibull: variance calculation" {
    // Weibull(k=1, λ=2) = Exponential(rate=0.5): Var(X) = λ² = 4
    const dist = try Weibull(f64).init(1.0, 2.0);
    try expectApproxEqRel(dist.variance(), 4.0, 1e-10);
}

test "Weibull: mode calculation" {
    // k < 1: mode = 0
    const dist1 = try Weibull(f64).init(0.5, 1.0);
    try testing.expectEqual(@as(f64, 0.0), dist1.mode());

    // k = 1: mode = 0
    const dist2 = try Weibull(f64).init(1.0, 1.0);
    try testing.expectEqual(@as(f64, 0.0), dist2.mode());

    // k = 2, λ = 1: mode = (1/2)^0.5 ≈ 0.7071
    const dist3 = try Weibull(f64).init(2.0, 1.0);
    try expectApproxEqRel(dist3.mode(), 1.0 / @sqrt(2.0), 1e-10);
}

test "Weibull: median calculation" {
    // Weibull(k=2, λ=1): Median = (ln 2)^0.5 ≈ 0.8326
    const dist = try Weibull(f64).init(2.0, 1.0);
    try expectApproxEqRel(dist.median(), @sqrt(@log(2.0)), 1e-10);
}

test "Weibull: survival function" {
    const dist = try Weibull(f64).init(2.0, 1.0);

    // S(x) = 1 - F(x)
    const x_vals = [_]f64{ 0.5, 1.0, 1.5, 2.0 };
    for (x_vals) |x| {
        const sf_val = dist.sf(x);
        const cdf_val = dist.cdf(x);
        try expectApproxEqRel(sf_val, 1.0 - cdf_val, 1e-10);
    }
}

test "Weibull: survival function direct formula k=2, λ=1" {
    // S(x) = exp(-x²)
    const dist = try Weibull(f64).init(2.0, 1.0);

    try expectApproxEqRel(dist.sf(1.0), @exp(-1.0), 1e-10);
    try expectApproxEqRel(dist.sf(2.0), @exp(-4.0), 1e-10);
}

test "Weibull: hazard function for k=1 (constant)" {
    // Exponential distribution has constant hazard rate = 1/λ
    const dist = try Weibull(f64).init(1.0, 2.0);

    // Hazard should be constant at 0.5
    try expectApproxEqRel(dist.hazard(0.0), 0.5, 1e-10);
    try expectApproxEqRel(dist.hazard(1.0), 0.5, 1e-10);
    try expectApproxEqRel(dist.hazard(2.0), 0.5, 1e-10);
}

test "Weibull: hazard function for k=2 (increasing)" {
    // Rayleigh distribution has increasing hazard rate
    const dist = try Weibull(f64).init(2.0, 1.0);

    // Hazard increases with x
    const h0 = dist.hazard(0.5);
    const h1 = dist.hazard(1.0);
    const h2 = dist.hazard(2.0);

    try testing.expect(h1 > h0);
    try testing.expect(h2 > h1);
}

test "Weibull: special case k=1 matches Exponential" {
    // Weibull(k=1, λ=2) should match Exponential(rate=1/2)
    const weibull_dist = try Weibull(f64).init(1.0, 2.0);
    const exp_dist = try Exponential(f64).init(0.5); // rate = 1/λ

    const x_vals = [_]f64{ 0.5, 1.0, 2.0, 4.0 };
    for (x_vals) |x| {
        // PDF should match
        try expectApproxEqRel(weibull_dist.pdf(x), exp_dist.pdf(x), 1e-10);

        // CDF should match
        try expectApproxEqRel(weibull_dist.cdf(x), exp_dist.cdf(x), 1e-10);
    }

    // Mean should match
    try expectApproxEqRel(weibull_dist.mean(), exp_dist.mean(), 1e-10);

    // Variance should match
    try expectApproxEqRel(weibull_dist.variance(), exp_dist.variance(), 1e-10);
}

test "Weibull: f32 precision support" {
    const dist = try Weibull(f32).init(2.0, 1.0);

    try expectApproxEqRel(dist.pdf(0.5), @as(f32, 0.7788), 1e-4);
    try expectApproxEqRel(dist.cdf(0.5), @as(f32, 0.2212), 1e-4);
    try expectApproxEqRel(try dist.quantile(0.25), @as(f32, 0.53637), 1e-4);
    try expectApproxEqRel(dist.mean(), @as(f32, 0.8862), 1e-4);
    try expectApproxEqRel(dist.variance(), @as(f32, 0.2146), 1e-4);
}

test "Weibull: memory safety" {
    const allocator = testing.allocator;
    _ = allocator;

    // No allocation in Weibull distribution, just verify init/usage
    for (0..10) |_| {
        const dist = try Weibull(f64).init(2.0, 1.0);
        try expectApproxEqRel(dist.pdf(1.0), 0.7357588824, 1e-10);
        _ = dist.cdf(0.5);
        _ = try dist.quantile(0.25);
        _ = dist.mean();
        _ = dist.variance();
        _ = dist.hazard(1.0);
    }
}

// ============================================================================
// Pareto Distribution (Type I)
// ============================================================================

/// Pareto distribution (Type I) Pareto(x_m, α)
///
/// Probability density function (PDF):
///   f(x) = (α × x_m^α) / x^(α+1)  for x ≥ x_m
///        = 0                       for x < x_m
///
/// Cumulative distribution function (CDF):
///   F(x) = 1 - (x_m / x)^α
///
/// Quantile function (inverse CDF):
///   Q(p) = x_m / (1-p)^(1/α)
///
/// Parameters:
///   - x_m: Scale parameter (minimum value, x_m > 0)
///   - alpha (α): Shape parameter (tail index, α > 0)
///
/// Properties:
///   - Domain: [x_m, ∞)
///   - Mean: α×x_m/(α-1) for α > 1, undefined otherwise
///   - Variance: (x_m²×α) / ((α-1)²(α-2)) for α > 2, undefined otherwise
///   - Mode: x_m (always at minimum)
///   - Heavy tail: larger α → lighter tail, smaller α → heavier tail
///
/// Use cases:
///   - Income/wealth distribution (Pareto principle: 80/20 rule)
///   - City population sizes
///   - File sizes, earthquake magnitudes
///   - Any phenomenon following power-law distribution
///
/// Time: O(1) for all operations
pub fn Pareto(comptime T: type) type {
    return struct {
        x_m: T, // scale (minimum value)
        alpha: T, // shape (tail index)

        const Self = @This();

        /// Create a Pareto distribution with given scale and shape
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(x_m: T, alpha: T) DistributionError!Self {
            if (x_m <= 0.0) return error.InvalidParameter;
            if (alpha <= 0.0) return error.InvalidParameter;
            if (!math.isFinite(x_m) or !math.isFinite(alpha)) return error.InvalidParameter;
            return Self{ .x_m = x_m, .alpha = alpha };
        }

        /// Probability density function (PDF) at x
        ///
        /// f(x) = (α × x_m^α) / x^(α+1)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn pdf(self: Self, x: T) T {
            if (x < self.x_m) return 0.0;
            // Use log-space for numerical stability
            // log(pdf) = log(α) + α×log(x_m) - (α+1)×log(x)
            const log_pdf = @log(self.alpha) + self.alpha * @log(self.x_m) - (self.alpha + 1.0) * @log(x);
            return @exp(log_pdf);
        }

        /// Log probability density function
        ///
        /// Time: O(1) | Space: O(1)
        pub fn logpdf(self: Self, x: T) T {
            if (x < self.x_m) return -math.inf(T);
            return @log(self.alpha) + self.alpha * @log(self.x_m) - (self.alpha + 1.0) * @log(x);
        }

        /// Cumulative distribution function (CDF) at x
        ///
        /// F(x) = 1 - (x_m/x)^α
        ///
        /// Time: O(1) | Space: O(1)
        pub fn cdf(self: Self, x: T) T {
            if (x < self.x_m) return 0.0;
            return 1.0 - math.pow(T, self.x_m / x, self.alpha);
        }

        /// Quantile function (inverse CDF)
        ///
        /// Q(p) = x_m / (1-p)^(1/α)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn quantile(self: Self, p: T) DistributionError!T {
            if (p < 0.0 or p > 1.0) return error.InvalidProbability;
            if (p == 0.0) return self.x_m;
            if (p == 1.0) return math.inf(T);
            // Q(p) = x_m / (1-p)^(1/α)
            return self.x_m / math.pow(T, 1.0 - p, 1.0 / self.alpha);
        }

        /// Generate random sample using inverse transform method
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sample(self: Self, rng: std.Random) T {
            const u = rng.float(T);
            return self.x_m / math.pow(T, 1.0 - u, 1.0 / self.alpha);
        }

        /// Survival function (complementary CDF)
        ///
        /// S(x) = 1 - F(x) = (x_m/x)^α
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sf(self: Self, x: T) T {
            if (x < self.x_m) return 1.0;
            return math.pow(T, self.x_m / x, self.alpha);
        }

        /// Mean (expected value)
        ///
        /// E[X] = α×x_m/(α-1) for α > 1, undefined otherwise
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mean(self: Self) T {
            if (self.alpha <= 1.0) return math.nan(T);
            return (self.alpha * self.x_m) / (self.alpha - 1.0);
        }

        /// Variance
        ///
        /// Var(X) = (x_m²×α) / ((α-1)²(α-2)) for α > 2, undefined otherwise
        ///
        /// Time: O(1) | Space: O(1)
        pub fn variance(self: Self) T {
            if (self.alpha <= 2.0) {
                if (self.alpha <= 1.0) return math.nan(T);
                return math.inf(T); // Infinite variance for 1 < α ≤ 2
            }
            const numerator = self.x_m * self.x_m * self.alpha;
            const denominator = (self.alpha - 1.0) * (self.alpha - 1.0) * (self.alpha - 2.0);
            return numerator / denominator;
        }

        /// Mode (most likely value)
        ///
        /// Mode = x_m (always at minimum)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mode(self: Self) T {
            return self.x_m;
        }

        /// Median
        ///
        /// Median = x_m × 2^(1/α)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn median(self: Self) T {
            return self.x_m * math.pow(T, 2.0, 1.0 / self.alpha);
        }

        /// Assert that parameters are valid: x_m > 0, alpha > 0, both finite.
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.x_m <= 0.0 or !math.isFinite(self.x_m)) return DistributionError.InvalidParameter;
            if (self.alpha <= 0.0 or !math.isFinite(self.alpha)) return DistributionError.InvalidParameter;
        }
    };
}

// ============================================================================
// Tests: Pareto Distribution
// ============================================================================

test "Pareto: initialization with valid parameters" {
    const dist = try Pareto(f64).init(1.0, 2.0);
    try expectEqual(@as(f64, 1.0), dist.x_m);
    try expectEqual(@as(f64, 2.0), dist.alpha);
}

test "Pareto: initialization errors" {
    // Negative scale
    try expectError(error.InvalidParameter, Pareto(f64).init(-1.0, 2.0));

    // Zero scale
    try expectError(error.InvalidParameter, Pareto(f64).init(0.0, 2.0));

    // Negative shape
    try expectError(error.InvalidParameter, Pareto(f64).init(1.0, -1.0));

    // Infinite scale
    try expectError(error.InvalidParameter, Pareto(f64).init(math.inf(f64), 2.0));

    // NaN shape
    try expectError(error.InvalidParameter, Pareto(f64).init(1.0, math.nan(f64)));
}

test "Pareto: PDF at mode (x_m)" {
    // PDF is maximized at x_m
    const dist = try Pareto(f64).init(1.0, 2.0);

    // PDF(x_m) = α / x_m = 2.0 / 1.0 = 2.0
    try expectApproxEqRel(dist.pdf(1.0), 2.0, 1e-10);
}

test "Pareto: PDF below x_m returns 0" {
    const dist = try Pareto(f64).init(2.0, 1.5);

    try expectEqual(@as(f64, 0.0), dist.pdf(0.0));
    try expectEqual(@as(f64, 0.0), dist.pdf(1.0));
    try expectEqual(@as(f64, 0.0), dist.pdf(1.99));
}

test "Pareto: PDF manual calculation" {
    // Pareto(x_m=1, α=2): PDF(x) = 2/x³
    const dist = try Pareto(f64).init(1.0, 2.0);

    try expectApproxEqRel(dist.pdf(1.0), 2.0, 1e-10);
    try expectApproxEqRel(dist.pdf(2.0), 2.0 / 8.0, 1e-10); // 2/2³ = 0.25
    try expectApproxEqRel(dist.pdf(4.0), 2.0 / 64.0, 1e-10); // 2/4³ = 0.03125
}

test "Pareto: CDF at boundaries" {
    const dist = try Pareto(f64).init(1.0, 2.0);

    // CDF at x_m should be 0
    try expectEqual(@as(f64, 0.0), dist.cdf(1.0));

    // CDF approaches 1 as x → ∞
    try expectApproxEqRel(dist.cdf(100.0), 1.0, 1e-3);
}

test "Pareto: CDF manual calculation" {
    // Pareto(x_m=1, α=2): CDF(x) = 1 - (1/x)²
    const dist = try Pareto(f64).init(1.0, 2.0);

    try expectApproxEqRel(dist.cdf(2.0), 1.0 - 0.25, 1e-10); // 1 - 1/4 = 0.75
    try expectApproxEqRel(dist.cdf(4.0), 1.0 - 0.0625, 1e-10); // 1 - 1/16 = 0.9375
}

test "Pareto: CDF monotonicity" {
    const dist = try Pareto(f64).init(1.0, 2.0);

    const x1 = dist.cdf(1.5);
    const x2 = dist.cdf(2.0);
    const x3 = dist.cdf(3.0);

    try testing.expect(x2 > x1);
    try testing.expect(x3 > x2);
}

test "Pareto: quantile at median" {
    const dist = try Pareto(f64).init(1.0, 2.0);

    // Median = x_m × 2^(1/α) = 1 × 2^0.5 ≈ 1.414
    const q_median = try dist.quantile(0.5);
    try expectApproxEqRel(q_median, @sqrt(2.0), 1e-10);
}

test "Pareto: quantile roundtrip with CDF" {
    const dist = try Pareto(f64).init(2.0, 1.5);

    const probs = [_]f64{ 0.1, 0.25, 0.5, 0.75, 0.9 };
    for (probs) |p| {
        const x = try dist.quantile(p);
        const p_back = dist.cdf(x);
        try expectApproxEqRel(p_back, p, 1e-10);
    }
}

test "Pareto: quantile error handling" {
    const dist = try Pareto(f64).init(1.0, 2.0);

    // Probability out of range
    try expectError(error.InvalidProbability, dist.quantile(-0.1));
    try expectError(error.InvalidProbability, dist.quantile(1.1));

    // Boundary cases
    try expectEqual(dist.x_m, try dist.quantile(0.0));
    try expectEqual(math.inf(f64), try dist.quantile(1.0));
}

test "Pareto: quantile manual calculation" {
    // Pareto(x_m=1, α=2): Q(p) = 1/(1-p)^0.5
    const dist = try Pareto(f64).init(1.0, 2.0);

    // Q(0.75) = 1/(0.25)^0.5 = 2
    try expectApproxEqRel(try dist.quantile(0.75), 2.0, 1e-10);

    // Q(0.9) = 1/(0.1)^0.5 ≈ 3.162
    try expectApproxEqRel(try dist.quantile(0.9), @sqrt(10.0), 1e-10);
}

test "Pareto: sampling mean validation" {
    const dist = try Pareto(f64).init(1.0, 3.0);
    var prng = std.Random.DefaultPrng.init(11111);
    const rng = prng.random();

    var sum: f64 = 0.0;
    const n_samples = 10000;
    for (0..n_samples) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    // Theoretical mean = α×x_m/(α-1) = 3×1/2 = 1.5
    const theoretical_mean = dist.mean();
    try expectApproxEqRel(sample_mean, theoretical_mean, 0.05); // 5% tolerance
}

test "Pareto: variance formula validation" {
    // Variance = x_m²×α / ((α-1)²(α-2))
    // For x_m=1, α=4: 1×4/(9×2) = 4/18 = 2/9 ≈ 0.2222
    const dist4 = try Pareto(f64).init(1.0, 4.0);
    try expectApproxEqRel(dist4.variance(), 2.0 / 9.0, 1e-10);

    // For x_m=2, α=5: 4×5/(16×3) = 20/48 = 5/12 ≈ 0.4167
    const dist5 = try Pareto(f64).init(2.0, 5.0);
    try expectApproxEqRel(dist5.variance(), 5.0 / 12.0, 1e-10);

    // Variance increases as α decreases (heavier tail)
    const dista = try Pareto(f64).init(1.0, 3.0);
    const distb = try Pareto(f64).init(1.0, 6.0);
    try testing.expect(dista.variance() > distb.variance());
}

test "Pareto: sampling produces values ≥ x_m" {
    const dist = try Pareto(f64).init(2.0, 1.5);
    var prng = std.Random.DefaultPrng.init(22222);
    const rng = prng.random();

    for (0..100) |_| {
        const x = dist.sample(rng);
        try testing.expect(x >= dist.x_m);
    }
}

test "Pareto: survival function consistency" {
    const dist = try Pareto(f64).init(1.0, 2.0);

    // S(x) = 1 - F(x)
    const x_vals = [_]f64{ 1.5, 2.0, 3.0, 5.0 };
    for (x_vals) |x| {
        const sf_val = dist.sf(x);
        const cdf_val = dist.cdf(x);
        try expectApproxEqRel(sf_val, 1.0 - cdf_val, 1e-10);
    }
}

test "Pareto: survival function direct formula" {
    // Pareto(x_m=1, α=2): S(x) = (1/x)²
    const dist = try Pareto(f64).init(1.0, 2.0);

    try expectApproxEqRel(dist.sf(2.0), 0.25, 1e-10);
    try expectApproxEqRel(dist.sf(4.0), 0.0625, 1e-10);
}

test "Pareto: mean for α > 1" {
    // Pareto(x_m=1, α=2): mean = 2×1/(2-1) = 2
    const dist = try Pareto(f64).init(1.0, 2.0);
    try expectApproxEqRel(dist.mean(), 2.0, 1e-10);

    // Pareto(x_m=2, α=3): mean = 3×2/(3-1) = 3
    const dist2 = try Pareto(f64).init(2.0, 3.0);
    try expectApproxEqRel(dist2.mean(), 3.0, 1e-10);
}

test "Pareto: mean undefined for α ≤ 1" {
    const dist = try Pareto(f64).init(1.0, 0.5);
    try testing.expect(math.isNan(dist.mean()));

    const dist2 = try Pareto(f64).init(1.0, 1.0);
    try testing.expect(math.isNan(dist2.mean()));
}

test "Pareto: variance for α > 2" {
    // Pareto(x_m=1, α=3): var = 1×3/((3-1)²×(3-2)) = 3/(4×1) = 0.75
    const dist = try Pareto(f64).init(1.0, 3.0);
    try expectApproxEqRel(dist.variance(), 0.75, 1e-10);
}

test "Pareto: variance infinite for 1 < α ≤ 2" {
    const dist = try Pareto(f64).init(1.0, 1.5);
    try expectEqual(math.inf(f64), dist.variance());

    const dist2 = try Pareto(f64).init(1.0, 2.0);
    try expectEqual(math.inf(f64), dist2.variance());
}

test "Pareto: variance undefined for α ≤ 1" {
    const dist = try Pareto(f64).init(1.0, 0.8);
    try testing.expect(math.isNan(dist.variance()));
}

test "Pareto: mode always at x_m" {
    const dist = try Pareto(f64).init(2.5, 1.8);
    try expectEqual(dist.x_m, dist.mode());
}

test "Pareto: median calculation" {
    // Pareto(x_m=1, α=2): median = 1 × 2^0.5 = √2
    const dist = try Pareto(f64).init(1.0, 2.0);
    try expectApproxEqRel(dist.median(), @sqrt(2.0), 1e-10);

    // Pareto(x_m=3, α=1): median = 3 × 2^1 = 6
    const dist2 = try Pareto(f64).init(3.0, 1.0);
    try expectApproxEqRel(dist2.median(), 6.0, 1e-10);
}

test "Pareto: logpdf consistency" {
    const dist = try Pareto(f64).init(1.0, 2.0);

    const x_vals = [_]f64{ 1.0, 2.0, 5.0, 10.0 };
    for (x_vals) |x| {
        const logpdf_val = dist.logpdf(x);
        const pdf_val = dist.pdf(x);
        try expectApproxEqRel(logpdf_val, @log(pdf_val), 1e-10);
    }
}

test "Pareto: 80/20 rule demonstration" {
    // Classic Pareto: 80% of wealth owned by top 20%
    // Use α ≈ 1.161 (log(5)/log(4)) for exact 80/20
    const alpha = @log(5.0) / @log(4.0);
    const dist = try Pareto(f64).init(1.0, alpha);

    // Top 20% means CDF(threshold) = 0.8
    const threshold = try dist.quantile(0.8);

    // Check that survival function at threshold is 20%
    const top_20_percent = dist.sf(threshold);
    try expectApproxEqRel(top_20_percent, 0.2, 1e-10);
}

test "Pareto: f32 precision support" {
    const dist = try Pareto(f32).init(1.0, 3.0);

    try expectApproxEqRel(dist.pdf(1.5), @as(f32, 0.5926), 1e-4);
    try expectApproxEqRel(dist.cdf(2.0), @as(f32, 0.875), 1e-4);
    try expectApproxEqRel(try dist.quantile(0.5), @as(f32, 1.2599), 1e-4);
    try expectApproxEqRel(dist.mean(), @as(f32, 1.5), 1e-4);
    try expectApproxEqRel(dist.variance(), @as(f32, 0.75), 1e-4);
    try expectApproxEqRel(dist.mode(), @as(f32, 1.0), 1e-4);
    try expectApproxEqRel(dist.median(), @as(f32, 1.2599), 1e-4);
}

test "Pareto: memory safety" {
    const allocator = testing.allocator;
    _ = allocator;

    // No allocation in Pareto distribution, just verify init/usage
    for (0..10) |_| {
        const dist = try Pareto(f64).init(1.0, 3.0);
        _ = dist.pdf(2.0);
        try expectApproxEqRel(dist.cdf(2.0), 0.875, 1e-10);
        _ = try dist.quantile(0.5);
        _ = dist.mean();
        _ = dist.variance();
        _ = dist.mode();
        _ = dist.median();
        _ = dist.sf(2.0);
    }
}

// ============================================================================
// LogNormal Distribution
// ============================================================================

/// LogNormal(μ, σ) — Log-normal distribution for right-skewed positive data
///
/// A continuous probability distribution where log(X) ~ Normal(μ, σ).
/// Used for modeling positive random variables that are products of many
/// independent random factors (multiplicative process).
///
/// Properties:
/// - Domain: (0, ∞)
/// - Parameters: μ (location), σ > 0 (scale)
/// - Mean: exp(μ + σ²/2)
/// - Variance: [exp(σ²) - 1] × exp(2μ + σ²)
/// - Mode: exp(μ - σ²)
/// - Median: exp(μ)
///
/// Use cases:
/// - Finance: stock prices, income distribution, option pricing
/// - Biology: species abundance, growth rates, cell sizes
/// - Environmental: pollutant concentrations, particle sizes
/// - Engineering: time to failure, reliability analysis
/// - Economics: wealth distribution, company sizes
///
/// Parameters:
///   - mu (μ): Location parameter of underlying normal (-∞ to +∞)
///   - sigma (σ): Scale parameter of underlying normal (σ > 0)
///
/// Time: O(1) for all operations
pub fn LogNormal(comptime T: type) type {
    return struct {
        mu: T,
        sigma: T,

        const Self = @This();

        /// Create a log-normal distribution with given parameters
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(mu: T, sigma: T) DistributionError!Self {
            if (sigma <= 0.0) return error.InvalidParameter;
            if (!math.isFinite(mu) or !math.isFinite(sigma)) return error.InvalidParameter;
            return Self{ .mu = mu, .sigma = sigma };
        }

        /// Probability density function (PDF) at x
        ///
        /// f(x) = (1 / (x σ √(2π))) × exp(-(ln(x)-μ)²/(2σ²))  for x > 0
        ///      = 0                                             for x ≤ 0
        ///
        /// Time: O(1) | Space: O(1)
        pub fn pdf(self: Self, x: T) T {
            if (x <= 0.0) return 0.0;

            const log_x = @log(x);
            const z = (log_x - self.mu) / self.sigma;
            const norm_factor = 1.0 / (x * self.sigma * @sqrt(2.0 * math.pi));
            return norm_factor * @exp(-0.5 * z * z);
        }

        /// Cumulative distribution function (CDF) at x
        ///
        /// F(x) = Φ((ln(x) - μ) / σ)  for x > 0
        ///      = 0                    for x ≤ 0
        ///
        /// where Φ is the standard normal CDF
        ///
        /// Time: O(1) | Space: O(1)
        pub fn cdf(self: Self, x: T) T {
            if (x <= 0.0) return 0.0;

            const log_x = @log(x);
            const z = (log_x - self.mu) / (self.sigma * @sqrt(2.0));
            return 0.5 * (1.0 + erf(z));
        }

        /// Quantile function (inverse CDF) - returns x such that P(X ≤ x) = p
        ///
        /// Q(p) = exp(μ + σ × Φ⁻¹(p))
        ///
        /// where Φ⁻¹ is the inverse standard normal CDF
        ///
        /// Time: O(1) | Space: O(1)
        pub fn quantile(self: Self, p: T) DistributionError!T {
            if (p < 0.0 or p > 1.0) return error.InvalidProbability;
            if (p == 0.0) return 0.0;
            if (p == 1.0) return math.inf(T);

            // Use inverse error function approximation for standard normal quantile
            const z = erfInv(2.0 * p - 1.0) * @sqrt(2.0);
            return @exp(self.mu + self.sigma * z);
        }

        /// Generate a random sample from this distribution
        ///
        /// Uses: if Z ~ N(μ, σ²), then X = exp(Z) ~ LogNormal(μ, σ)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sample(self: Self, rng: std.Random) T {
            // Sample from Normal(μ, σ²), then exponentiate
            const uniform1 = rng.float(T);
            const uniform2 = rng.float(T);

            // Box-Muller transform for standard normal
            const z = @sqrt(-2.0 * @log(uniform1)) * @cos(2.0 * math.pi * uniform2);

            // Scale and shift, then exponentiate
            return @exp(self.mu + self.sigma * z);
        }

        /// Log probability density function (log PDF) at x
        ///
        /// log f(x) = -log(x) - log(σ√(2π)) - (ln(x)-μ)²/(2σ²)
        ///
        /// More numerically stable than log(pdf(x)) for extreme values
        ///
        /// Time: O(1) | Space: O(1)
        pub fn logpdf(self: Self, x: T) T {
            if (x <= 0.0) return -math.inf(T);

            const log_x = @log(x);
            const z = (log_x - self.mu) / self.sigma;
            return -log_x - @log(self.sigma * @sqrt(2.0 * math.pi)) - 0.5 * z * z;
        }

        /// Survival function (complementary CDF) - P(X > x)
        ///
        /// S(x) = 1 - F(x) = 1 - Φ((ln(x) - μ) / σ)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sf(self: Self, x: T) T {
            if (x <= 0.0) return 1.0;

            const log_x = @log(x);
            const z = (log_x - self.mu) / (self.sigma * @sqrt(2.0));
            return 0.5 * (1.0 - erf(z));
        }

        /// Expected value (mean)
        ///
        /// E[X] = exp(μ + σ²/2)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mean(self: Self) T {
            return @exp(self.mu + 0.5 * self.sigma * self.sigma);
        }

        /// Variance
        ///
        /// Var(X) = [exp(σ²) - 1] × exp(2μ + σ²)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn variance(self: Self) T {
            const sigma_sq = self.sigma * self.sigma;
            return (@exp(sigma_sq) - 1.0) * @exp(2.0 * self.mu + sigma_sq);
        }

        /// Mode (most probable value)
        ///
        /// mode = exp(μ - σ²)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mode(self: Self) T {
            return @exp(self.mu - self.sigma * self.sigma);
        }

        /// Median (50th percentile)
        ///
        /// median = exp(μ)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn median(self: Self) T {
            return @exp(self.mu);
        }

        /// Assert that parameters are valid: sigma > 0, both finite.
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.sigma <= 0.0 or !math.isFinite(self.sigma)) return DistributionError.InvalidParameter;
            if (!math.isFinite(self.mu)) return DistributionError.InvalidParameter;
        }
    };
}

// ============================================================================
// LogNormal Distribution Tests
// ============================================================================

test "LogNormal: init with valid parameters" {
    const dist = try LogNormal(f64).init(0.0, 1.0);
    try expectEqual(0.0, dist.mu);
    try expectEqual(1.0, dist.sigma);
}

test "LogNormal: init with zero sigma fails" {
    try expectError(error.InvalidParameter, LogNormal(f64).init(0.0, 0.0));
}

test "LogNormal: init with negative sigma fails" {
    try expectError(error.InvalidParameter, LogNormal(f64).init(0.0, -1.0));
}

test "LogNormal: init with infinite mu fails" {
    try expectError(error.InvalidParameter, LogNormal(f64).init(math.inf(f64), 1.0));
}

test "LogNormal: init with NaN sigma fails" {
    try expectError(error.InvalidParameter, LogNormal(f64).init(0.0, math.nan(f64)));
}

test "LogNormal: pdf at mode is maximum" {
    const dist = try LogNormal(f64).init(0.0, 1.0);
    const mode = dist.mode();

    // PDF should be higher at mode than at nearby points
    const pdf_mode = dist.pdf(mode);
    const pdf_lower = dist.pdf(mode * 0.9);
    const pdf_upper = dist.pdf(mode * 1.1);

    try expect(pdf_mode > pdf_lower);
    try expect(pdf_mode > pdf_upper);
}

test "LogNormal: pdf for negative x is zero" {
    const dist = try LogNormal(f64).init(0.0, 1.0);
    try expectEqual(0.0, dist.pdf(-1.0));
    try expectEqual(0.0, dist.pdf(0.0));
}

test "LogNormal: pdf manual calculation" {
    const dist = try LogNormal(f64).init(0.0, 1.0);

    // For LogNormal(0, 1) at x=1: f(1) = 1/(1×1×√(2π)) × exp(0) = 1/√(2π) ≈ 0.3989
    const pdf_at_1 = dist.pdf(1.0);
    const expected = 1.0 / @sqrt(2.0 * math.pi);
    try expectApproxEqRel(pdf_at_1, expected, 1e-10);
}

test "LogNormal: cdf at median is 0.5" {
    const dist = try LogNormal(f64).init(1.5, 0.5);
    const median = dist.median();
    const cdf_median = dist.cdf(median);
    try expectApproxEqRel(cdf_median, 0.5, 1e-10);
}

test "LogNormal: cdf for negative x is zero" {
    const dist = try LogNormal(f64).init(0.0, 1.0);
    try expectEqual(0.0, dist.cdf(-1.0));
    try expectEqual(0.0, dist.cdf(0.0));
}

test "LogNormal: cdf is monotonic" {
    const dist = try LogNormal(f64).init(0.0, 1.0);

    const cdf1 = dist.cdf(0.5);
    const cdf2 = dist.cdf(1.0);
    const cdf3 = dist.cdf(2.0);
    const cdf4 = dist.cdf(5.0);

    try expect(cdf1 < cdf2);
    try expect(cdf2 < cdf3);
    try expect(cdf3 < cdf4);
}

test "LogNormal: cdf manual calculation" {
    const dist = try LogNormal(f64).init(0.0, 1.0);

    // For LogNormal(0, 1) at x=exp(1): cdf(e) = Φ((ln(e)-0)/1) = Φ(1) ≈ 0.8413
    const e = @exp(1.0);
    const cdf_at_e = dist.cdf(e);
    const expected = 0.8413447460685429; // Φ(1)
    try expectApproxEqRel(cdf_at_e, expected, 1e-7); // limited by erf approximation accuracy
}

test "LogNormal: quantile at p=0.5 equals median" {
    const dist = try LogNormal(f64).init(1.0, 0.5);
    const q = try dist.quantile(0.5);
    const median = dist.median();
    try expectApproxEqRel(q, median, 1e-10);
}

test "LogNormal: quantile roundtrip with cdf" {
    const dist = try LogNormal(f64).init(0.5, 1.5);

    const probabilities = [_]f64{ 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99 };
    for (probabilities) |p| {
        const q = try dist.quantile(p);
        const cdf_q = dist.cdf(q);
        try expectApproxEqRel(cdf_q, p, 1e-9);
    }
}

test "LogNormal: quantile error handling" {
    const dist = try LogNormal(f64).init(0.0, 1.0);

    try expectError(error.InvalidProbability, dist.quantile(-0.1));
    try expectError(error.InvalidProbability, dist.quantile(1.1));
}

test "LogNormal: quantile at p=0 is 0" {
    const dist = try LogNormal(f64).init(0.0, 1.0);
    const q = try dist.quantile(0.0);
    try expectEqual(0.0, q);
}

test "LogNormal: quantile at p=1 is infinity" {
    const dist = try LogNormal(f64).init(0.0, 1.0);
    const q = try dist.quantile(1.0);
    try expect(math.isInf(q));
    try expect(q > 0.0);
}

test "LogNormal: quantile manual calculation" {
    const dist = try LogNormal(f64).init(0.0, 1.0);

    // For LogNormal(0, 1): Q(0.5) = exp(0 + 1×Φ⁻¹(0.5)) = exp(0) = 1
    const q_median = try dist.quantile(0.5);
    try expectApproxEqRel(q_median, 1.0, 1e-10);
}

test "LogNormal: sample produces positive values" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const dist = try LogNormal(f64).init(0.0, 1.0);

    for (0..100) |_| {
        const x = dist.sample(rng);
        try expect(x > 0.0);
    }
}

test "LogNormal: sample mean validation" {
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    const dist = try LogNormal(f64).init(0.0, 0.5);
    const expected_mean = dist.mean();

    const n = 10000;
    var sum: f64 = 0.0;
    for (0..n) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n));

    // Within 5% of theoretical mean (log-normal has high variance)
    try expectApproxEqRel(sample_mean, expected_mean, 0.05);
}

test "LogNormal: sample variance validation" {
    var prng = std.Random.DefaultPrng.init(54321);
    const rng = prng.random();

    const dist = try LogNormal(f64).init(0.0, 0.5);
    const expected_mean = dist.mean();
    const expected_variance = dist.variance();

    const n = 10000;
    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    for (0..n) |_| {
        const x = dist.sample(rng);
        sum += x;
        sum_sq += x * x;
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n));
    const sample_variance = sum_sq / @as(f64, @floatFromInt(n)) - sample_mean * sample_mean;

    // Mean and variance within 10% (log-normal has heavy tail)
    try expectApproxEqRel(sample_mean, expected_mean, 0.1);
    try expectApproxEqRel(sample_variance, expected_variance, 0.15);
}

test "LogNormal: logpdf consistency with log(pdf)" {
    const dist = try LogNormal(f64).init(0.0, 1.0);

    const test_points = [_]f64{ 0.1, 0.5, 1.0, 2.0, 5.0, 10.0 };
    for (test_points) |x| {
        const logpdf_val = dist.logpdf(x);
        const pdf_val = dist.pdf(x);
        const log_pdf_val = @log(pdf_val);

        try expectApproxEqRel(logpdf_val, log_pdf_val, 1e-10);
    }
}

test "LogNormal: logpdf for negative x is -inf" {
    const dist = try LogNormal(f64).init(0.0, 1.0);
    const logpdf_neg = dist.logpdf(-1.0);
    try expect(math.isInf(logpdf_neg));
    try expect(logpdf_neg < 0.0);
}

test "LogNormal: survival function consistency with 1 - cdf" {
    const dist = try LogNormal(f64).init(0.0, 1.0);

    const test_points = [_]f64{ 0.1, 0.5, 1.0, 2.0, 5.0, 10.0 };
    for (test_points) |x| {
        const sf_val = dist.sf(x);
        const cdf_val = dist.cdf(x);
        try expectApproxEqRel(sf_val, 1.0 - cdf_val, 1e-10);
    }
}

test "LogNormal: sf for negative x is 1" {
    const dist = try LogNormal(f64).init(0.0, 1.0);
    try expectEqual(1.0, dist.sf(-1.0));
    try expectEqual(1.0, dist.sf(0.0));
}

test "LogNormal: mean formula" {
    const dist = try LogNormal(f64).init(1.0, 0.5);
    const mean_val = dist.mean();
    // E[X] = exp(μ + σ²/2) = exp(1.0 + 0.25/2) = exp(1.125)
    const expected = @exp(1.0 + 0.5 * 0.5 / 2.0);
    try expectApproxEqRel(mean_val, expected, 1e-10);
}

test "LogNormal: variance formula" {
    const dist = try LogNormal(f64).init(0.0, 1.0);
    const var_val = dist.variance();
    // Var(X) = [exp(σ²) - 1] × exp(2μ + σ²) = [exp(1) - 1] × exp(0 + 1) = [e - 1] × e
    const sigma_sq = 1.0;
    const expected = (@exp(sigma_sq) - 1.0) * @exp(2.0 * 0.0 + sigma_sq);
    try expectApproxEqRel(var_val, expected, 1e-10);
}

test "LogNormal: mode formula" {
    const dist = try LogNormal(f64).init(1.0, 0.5);
    const mode_val = dist.mode();
    // mode = exp(μ - σ²) = exp(1.0 - 0.25) = exp(0.75)
    const expected = @exp(1.0 - 0.5 * 0.5);
    try expectApproxEqRel(mode_val, expected, 1e-10);
}

test "LogNormal: median formula" {
    const dist = try LogNormal(f64).init(1.5, 2.0);
    const median_val = dist.median();
    // median = exp(μ) = exp(1.5)
    const expected = @exp(1.5);
    try expectApproxEqRel(median_val, expected, 1e-10);
}

test "LogNormal: median < mean < mode relationship" {
    const dist = try LogNormal(f64).init(0.0, 1.0);

    const mode_val = dist.mode();
    const median_val = dist.median();
    const mean_val = dist.mean();

    // For log-normal: mode < median < mean (right-skewed)
    try expect(mode_val < median_val);
    try expect(median_val < mean_val);
}

test "LogNormal: standard log-normal special case" {
    // LogNormal(0, 1) is the "standard" log-normal
    const dist = try LogNormal(f64).init(0.0, 1.0);

    try expectApproxEqRel(dist.median(), 1.0, 1e-10);
    try expectApproxEqRel(dist.mode(), @exp(-1.0), 1e-10);
}

test "LogNormal: f32 precision support" {
    const dist = try LogNormal(f32).init(0.0, 1.0);

    _ = dist.pdf(1.5);
    try expectApproxEqRel(dist.cdf(2.0), @as(f32, 0.7558), 1e-3);
    try expectApproxEqRel(try dist.quantile(0.5), @as(f32, 1.0), 1e-3);
    try expectApproxEqRel(dist.mean(), @as(f32, 1.6487), 1e-3);
    try expectApproxEqRel(dist.mode(), @as(f32, 0.3679), 1e-3);
    try expectApproxEqRel(dist.median(), @as(f32, 1.0), 1e-3);
    try expectApproxEqRel(dist.logpdf(1.0), @as(f32, -0.9189), 1e-3);
    try expectApproxEqRel(dist.sf(2.0), @as(f32, 0.2442), 1e-3);
}

test "LogNormal: memory safety" {
    const allocator = testing.allocator;
    _ = allocator;

    // No allocation in LogNormal distribution, just verify init/usage
    for (0..10) |_| {
        const dist = try LogNormal(f64).init(0.0, 1.0);
        _ = dist.pdf(1.0);
        _ = dist.cdf(1.0);
        try expectApproxEqRel(try dist.quantile(0.5), 1.0, 1e-10);
        _ = dist.mean();
        _ = dist.variance();
        _ = dist.mode();
        _ = dist.median();
        _ = dist.logpdf(1.0);
        _ = dist.sf(1.0);
    }
}

// ============================================================================
// Cauchy Distribution
// ============================================================================

/// Cauchy distribution (Lorentz distribution) with location x₀ and scale γ
///
/// Probability density function (PDF):
///   f(x) = 1 / (πγ[1 + ((x-x₀)/γ)²])
///
/// Cumulative distribution function (CDF):
///   F(x) = (1/π) × arctan((x-x₀)/γ) + 1/2
///
/// Parameters:
///   - x0 (x₀): Location parameter (median, mode) (-∞ to +∞)
///   - gamma (γ): Scale parameter (half-width at half-maximum, γ > 0)
///
/// Properties:
///   - Mean: undefined (infinite variance prevents mean from existing)
///   - Variance: undefined (infinite)
///   - Mode: x₀
///   - Median: x₀
///   - Heavy-tailed distribution (heavier than Student's t)
///   - Stable distribution (sum of Cauchy RVs is Cauchy)
///   - Ratio of two independent standard normals ~ Cauchy(0,1)
///
/// Use cases:
///   - Resonance phenomena in physics (Lorentzian profile)
///   - Ratio of two normal random variables
///   - Modeling outlier-prone data
///   - Spectral line broadening
///   - Robust statistics (median-based inference)
///
/// Time: O(1) for all operations
pub fn Cauchy(comptime T: type) type {
    return struct {
        x0: T, // location (median, mode)
        gamma: T, // scale (half-width at half-maximum)

        const Self = @This();

        /// Create a Cauchy distribution with given location and scale
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(x0: T, gamma: T) DistributionError!Self {
            if (gamma <= 0.0) return error.InvalidParameter;
            if (!math.isFinite(x0) or !math.isFinite(gamma)) return error.InvalidParameter;
            return Self{ .x0 = x0, .gamma = gamma };
        }

        /// Probability density function (PDF) at x
        ///
        /// f(x) = 1 / (πγ[1 + ((x-x₀)/γ)²])
        ///
        /// Time: O(1) | Space: O(1)
        pub fn pdf(self: Self, x: T) T {
            const z = (x - self.x0) / self.gamma;
            return 1.0 / (math.pi * self.gamma * (1.0 + z * z));
        }

        /// Log probability density function
        ///
        /// Time: O(1) | Space: O(1)
        pub fn logpdf(self: Self, x: T) T {
            const z = (x - self.x0) / self.gamma;
            // log(pdf) = -log(π) - log(γ) - log(1 + z²)
            return -@log(math.pi) - @log(self.gamma) - @log(1.0 + z * z);
        }

        /// Cumulative distribution function (CDF) at x
        ///
        /// F(x) = (1/π) × arctan((x-x₀)/γ) + 1/2
        ///
        /// Time: O(1) | Space: O(1)
        pub fn cdf(self: Self, x: T) T {
            const z = (x - self.x0) / self.gamma;
            return (1.0 / math.pi) * math.atan(z) + 0.5;
        }

        /// Quantile function (inverse CDF)
        ///
        /// Q(p) = x₀ + γ × tan(π(p - 1/2))
        ///
        /// Time: O(1) | Space: O(1)
        pub fn quantile(self: Self, p: T) DistributionError!T {
            if (p < 0.0 or p > 1.0) return error.InvalidProbability;
            if (p == 0.0) return -math.inf(T);
            if (p == 1.0) return math.inf(T);
            // Q(p) = x₀ + γ × tan(π(p - 1/2))
            return self.x0 + self.gamma * @tan(math.pi * (p - 0.5));
        }

        /// Generate random sample using inverse transform method
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sample(self: Self, rng: std.Random) T {
            const u = rng.float(T);
            const safe_u = @max(1e-10, @min(1.0 - 1e-10, u));
            return self.x0 + self.gamma * @tan(math.pi * (safe_u - 0.5));
        }

        /// Survival function (complementary CDF)
        ///
        /// S(x) = 1 - F(x) = 1/2 - (1/π) × arctan((x-x₀)/γ)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sf(self: Self, x: T) T {
            return 1.0 - self.cdf(x);
        }

        /// Mean (expected value) - undefined for Cauchy distribution
        ///
        /// The Cauchy distribution has no defined mean (infinite variance)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mean(self: Self) T {
            _ = self;
            return math.nan(T);
        }

        /// Variance - undefined for Cauchy distribution
        ///
        /// The Cauchy distribution has infinite variance
        ///
        /// Time: O(1) | Space: O(1)
        pub fn variance(self: Self) T {
            _ = self;
            return math.inf(T);
        }

        /// Mode (most likely value)
        ///
        /// Mode = x₀ (location parameter)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mode(self: Self) T {
            return self.x0;
        }

        /// Median (50th percentile)
        ///
        /// Median = x₀ (location parameter)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn median(self: Self) T {
            return self.x0;
        }

        /// Assert that parameters are valid: gamma > 0, both finite.
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.gamma <= 0.0 or !math.isFinite(self.gamma)) return DistributionError.InvalidParameter;
            if (!math.isFinite(self.x0)) return DistributionError.InvalidParameter;
        }
    };
}

// ============================================================================
// Cauchy Distribution Tests
// ============================================================================

test "Cauchy: init with valid parameters" {
    const dist = try Cauchy(f64).init(0.0, 1.0);
    try testing.expectEqual(0.0, dist.x0);
    try testing.expectEqual(1.0, dist.gamma);

    const dist2 = try Cauchy(f64).init(2.5, 0.5);
    try testing.expectEqual(2.5, dist2.x0);
    try testing.expectEqual(0.5, dist2.gamma);
}

test "Cauchy: init rejects invalid parameters" {
    // Negative scale
    try testing.expectError(error.InvalidParameter, Cauchy(f64).init(0.0, -1.0));

    // Zero scale
    try testing.expectError(error.InvalidParameter, Cauchy(f64).init(0.0, 0.0));

    // Infinite parameters
    try testing.expectError(error.InvalidParameter, Cauchy(f64).init(math.inf(f64), 1.0));
    try testing.expectError(error.InvalidParameter, Cauchy(f64).init(0.0, math.inf(f64)));

    // NaN parameters
    try testing.expectError(error.InvalidParameter, Cauchy(f64).init(math.nan(f64), 1.0));
    try testing.expectError(error.InvalidParameter, Cauchy(f64).init(0.0, math.nan(f64)));
}

test "Cauchy: PDF properties" {
    const dist = try Cauchy(f64).init(0.0, 1.0);

    // PDF at mode (x₀) should be maximum = 1/(πγ)
    const pdf_mode = dist.pdf(0.0);
    try testing.expectApproxEqRel(1.0 / math.pi, pdf_mode, 1e-10);

    // PDF is symmetric around x₀
    const pdf_left = dist.pdf(-2.0);
    const pdf_right = dist.pdf(2.0);
    try testing.expectApproxEqRel(pdf_left, pdf_right, 1e-10);

    // PDF at x₀ ± γ should be half of maximum
    const pdf_hwhm = dist.pdf(1.0);
    try testing.expectApproxEqRel(pdf_mode / 2.0, pdf_hwhm, 1e-10);
}

test "Cauchy: PDF manual calculation" {
    const dist = try Cauchy(f64).init(0.0, 1.0);

    // Standard Cauchy at x=1: f(1) = 1/(π(1+1²)) = 1/(2π)
    const pdf_1 = dist.pdf(1.0);
    try testing.expectApproxEqRel(1.0 / (2.0 * math.pi), pdf_1, 1e-10);

    // Non-standard Cauchy(2, 0.5) at x=2.5:
    // z = (2.5-2)/0.5 = 1
    // f(2.5) = 1/(π×0.5×(1+1)) = 1/π ≈ 0.318310
    const dist2 = try Cauchy(f64).init(2.0, 0.5);
    const pdf_2 = dist2.pdf(2.5);
    try testing.expectApproxEqRel(1.0 / math.pi, pdf_2, 1e-6);
}

test "Cauchy: CDF properties" {
    const dist = try Cauchy(f64).init(0.0, 1.0);

    // CDF at median should be 0.5
    const cdf_median = dist.cdf(0.0);
    try testing.expectApproxEqRel(0.5, cdf_median, 1e-10);

    // CDF is symmetric around x₀
    const cdf_left = dist.cdf(-2.0);
    const cdf_right = dist.cdf(2.0);
    try testing.expectApproxEqRel(cdf_left, 1.0 - cdf_right, 1e-10);

    // CDF approaches 0 as x → -∞
    const cdf_neg_large = dist.cdf(-100.0);
    try testing.expect(cdf_neg_large < 0.01);

    // CDF approaches 1 as x → +∞
    const cdf_pos_large = dist.cdf(100.0);
    try testing.expect(cdf_pos_large > 0.99);
}

test "Cauchy: CDF manual calculation" {
    const dist = try Cauchy(f64).init(0.0, 1.0);

    // Standard Cauchy at x=1:
    // F(1) = (1/π)arctan(1) + 1/2 = (1/π)(π/4) + 1/2 = 1/4 + 1/2 = 3/4
    const cdf_1 = dist.cdf(1.0);
    try testing.expectApproxEqRel(0.75, cdf_1, 1e-10);

    // Standard Cauchy at x=-1:
    // F(-1) = (1/π)arctan(-1) + 1/2 = (1/π)(-π/4) + 1/2 = -1/4 + 1/2 = 1/4
    const cdf_neg1 = dist.cdf(-1.0);
    try testing.expectApproxEqRel(0.25, cdf_neg1, 1e-10);
}

test "Cauchy: quantile properties" {
    const dist = try Cauchy(f64).init(0.0, 1.0);

    // Median (p=0.5) should equal x₀
    const q_median = try dist.quantile(0.5);
    try testing.expectApproxEqRel(0.0, q_median, 1e-10);

    // Quantile at p=0.75 should equal 1 for standard Cauchy
    // Q(0.75) = tan(π(0.75-0.5)) = tan(π/4) = 1
    const q_75 = try dist.quantile(0.75);
    try testing.expectApproxEqRel(1.0, q_75, 1e-10);

    // Quantile at p=0.25 should equal -1
    const q_25 = try dist.quantile(0.25);
    try testing.expectApproxEqRel(-1.0, q_25, 1e-10);

    // Boundary cases
    const q_0 = try dist.quantile(0.0);
    try testing.expect(math.isNegativeInf(q_0));

    const q_1 = try dist.quantile(1.0);
    try testing.expect(math.isPositiveInf(q_1));
}

test "Cauchy: quantile error handling" {
    const dist = try Cauchy(f64).init(0.0, 1.0);

    // Invalid probabilities
    try testing.expectError(error.InvalidProbability, dist.quantile(-0.1));
    try testing.expectError(error.InvalidProbability, dist.quantile(1.1));
}

test "Cauchy: quantile-CDF roundtrip" {
    const dist = try Cauchy(f64).init(0.0, 1.0);

    const test_probs = [_]f64{ 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99 };
    for (test_probs) |p| {
        const q = try dist.quantile(p);
        const p_back = dist.cdf(q);
        try testing.expectApproxEqRel(p, p_back, 1e-8);
    }
}

test "Cauchy: sampling produces values in reasonable range" {
    const dist = try Cauchy(f64).init(0.0, 1.0);
    var prng = std.Random.DefaultPrng.init(33333);
    const rng = prng.random();

    // Cauchy has heavy tails, but median should be near x₀
    var count_near_median: usize = 0;
    const n_samples = 1000;

    for (0..n_samples) |_| {
        const x = dist.sample(rng);
        // About 93.5% should be within [-10, 10]
        if (x >= -10.0 and x <= 10.0) {
            count_near_median += 1;
        }
    }

    // At least 80% should be within [-10, 10] (conservative check)
    const ratio = @as(f64, @floatFromInt(count_near_median)) / @as(f64, @floatFromInt(n_samples));
    try testing.expect(ratio >= 0.80);
}

test "Cauchy: logpdf consistency with log(pdf)" {
    const dist = try Cauchy(f64).init(0.0, 1.0);

    const test_x = [_]f64{ -5.0, -1.0, 0.0, 1.0, 5.0 };
    for (test_x) |x| {
        const log_pdf = dist.logpdf(x);
        const pdf = dist.pdf(x);
        try testing.expectApproxEqRel(@log(pdf), log_pdf, 1e-10);
    }
}

test "Cauchy: survival function consistency" {
    const dist = try Cauchy(f64).init(0.0, 1.0);

    const test_x = [_]f64{ -5.0, -1.0, 0.0, 1.0, 5.0 };
    for (test_x) |x| {
        const sf = dist.sf(x);
        const cdf = dist.cdf(x);
        try testing.expectApproxEqRel(1.0 - cdf, sf, 1e-10);
    }
}

test "Cauchy: mean is undefined (NaN)" {
    const dist = try Cauchy(f64).init(0.0, 1.0);
    const m = dist.mean();
    try testing.expect(math.isNan(m));
}

test "Cauchy: variance is infinite" {
    const dist = try Cauchy(f64).init(0.0, 1.0);
    const v = dist.variance();
    try testing.expect(math.isPositiveInf(v));
}

test "Cauchy: mode equals location parameter" {
    const dist = try Cauchy(f64).init(2.5, 1.0);
    const m = dist.mode();
    try testing.expectEqual(2.5, m);
}

test "Cauchy: median equals location parameter" {
    const dist = try Cauchy(f64).init(-1.5, 0.5);
    const m = dist.median();
    try testing.expectEqual(-1.5, m);
}

test "Cauchy: ratio of normals property" {
    // The ratio Z = X/Y where X,Y ~ N(0,1) follows Cauchy(0,1)
    // We can verify this by checking that samples from ratio match Cauchy quantiles
    const allocator = testing.allocator;

    const n_samples = 10000;
    const samples = try allocator.alloc(f64, n_samples);
    defer allocator.free(samples);

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    // Generate ratio samples
    for (samples) |*s| {
        // Box-Muller for standard normal
        const uniform1 = rand.float(f64);
        const uniform2 = rand.float(f64);
        const x = @sqrt(-2.0 * @log(uniform1)) * @cos(2.0 * math.pi * uniform2);
        const y = @sqrt(-2.0 * @log(uniform1)) * @sin(2.0 * math.pi * uniform2);
        s.* = x / y; // Ratio ~ Cauchy(0,1)
    }

    // Sort samples to compute empirical quantiles
    std.mem.sort(f64, samples, {}, std.sort.asc(f64));

    const dist = try Cauchy(f64).init(0.0, 1.0);

    // Check empirical quantiles match theoretical quantiles (skip p=0.5 since theoretical=0
    // and expectApproxEqRel cannot handle expected=0; use absolute tolerance instead)
    const test_probs = [_]f64{ 0.25, 0.75 };
    for (test_probs) |p| {
        const idx = @as(usize, @intFromFloat(p * @as(f64, @floatFromInt(n_samples))));
        const empirical_q = samples[@min(idx, n_samples - 1)];
        const theoretical_q = try dist.quantile(p);
        try testing.expectApproxEqRel(theoretical_q, empirical_q, 0.1);
    }
    // Check median (p=0.5) with absolute tolerance
    const median_idx = n_samples / 2;
    const empirical_median = samples[median_idx];
    try testing.expectApproxEqAbs(@as(f64, 0.0), empirical_median, 0.1);
}

test "Cauchy: f32 precision support" {
    const dist = try Cauchy(f32).init(0.0, 1.0);

    // pdf(1.5) = 1/(π*(1+1.5²)) = 1/(π*3.25) ≈ 0.09794
    try expectApproxEqRel(dist.pdf(1.5), @as(f32, 0.09794), 1e-3);
    try expectApproxEqRel(dist.cdf(2.0), @as(f32, 0.8524), 1e-3);
    try expectApproxEqAbs(try dist.quantile(0.5), @as(f32, 0.0), 1e-5);
    try testing.expect(math.isNan(dist.mean()));
    try testing.expect(math.isInf(dist.variance()));
    try expectApproxEqAbs(dist.mode(), @as(f32, 0.0), 1e-5);
    try expectApproxEqAbs(dist.median(), @as(f32, 0.0), 1e-5);
    try expectApproxEqRel(dist.logpdf(1.0), @as(f32, -1.8379), 1e-3);
    try expectApproxEqRel(dist.sf(2.0), @as(f32, 0.1476), 1e-3);
}

test "Cauchy: memory safety" {
    const allocator = testing.allocator;
    _ = allocator;

    // No allocation in Cauchy distribution, just verify init/usage
    for (0..10) |_| {
        const dist = try Cauchy(f64).init(0.0, 1.0);
        _ = dist.pdf(1.0);
        try expectApproxEqRel(dist.cdf(0.0), 0.5, 1e-10);
        _ = try dist.quantile(0.5);
        _ = dist.mean();
        _ = dist.variance();
        _ = dist.mode();
        _ = dist.median();
        _ = dist.logpdf(1.0);
        _ = dist.sf(1.0);
    }
}

// ============================================================================
// Gumbel Distribution
// ============================================================================

/// Gumbel distribution (Type-I Extreme Value distribution) with location μ and scale β
///
/// Probability density function (PDF):
///   f(x) = (1/β) × exp(-(z + exp(-z)))  where z = (x - μ) / β
///
/// Cumulative distribution function (CDF):
///   F(x) = exp(-exp(-z))  where z = (x - μ) / β
///
/// Parameters:
///   - mu (μ): Location parameter (-∞ to +∞)
///   - beta (β): Scale parameter (β > 0)
///
/// Properties:
///   - Mean: μ + β × γ  where γ ≈ 0.5772156649015329 (Euler-Mascheroni constant)
///   - Variance: π² × β² / 6
///   - Mode: μ
///   - Median: μ - β × ln(ln(2))
///   - Right-skewed distribution (positive skewness = 1.14)
///   - Used for extreme value modeling (maximum of many samples)
///
/// Use cases:
///   - Extreme value theory (flood levels, wind speeds, earthquake magnitudes)
///   - Gumbel-softmax trick for categorical sampling in machine learning
///   - Reliability engineering and survival analysis
///   - Modeling maximum order statistics
///
/// Time: O(1) for all operations
pub fn Gumbel(comptime T: type) type {
    return struct {
        mu: T, // location parameter
        beta: T, // scale parameter (β > 0)

        const Self = @This();
        const euler_mascheroni: T = 0.5772156649015329;

        /// Create a Gumbel distribution with given location and scale
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(mu: T, beta: T) DistributionError!Self {
            if (beta <= 0.0) return error.InvalidParameter;
            if (!math.isFinite(mu) or !math.isFinite(beta)) return error.InvalidParameter;
            return Self{ .mu = mu, .beta = beta };
        }

        /// Probability density function (PDF) at x
        ///
        /// f(x) = (1/β) × exp(-(z + exp(-z)))  where z = (x - μ) / β
        ///
        /// Time: O(1) | Space: O(1)
        pub fn pdf(self: Self, x: T) T {
            const z = (x - self.mu) / self.beta;
            return (1.0 / self.beta) * @exp(-(z + @exp(-z)));
        }

        /// Log probability density function
        ///
        /// log f(x) = -log(β) - z - exp(-z)  where z = (x - μ) / β
        ///
        /// Time: O(1) | Space: O(1)
        pub fn logpdf(self: Self, x: T) T {
            const z = (x - self.mu) / self.beta;
            return -@log(self.beta) - z - @exp(-z);
        }

        /// Cumulative distribution function (CDF) at x
        ///
        /// F(x) = exp(-exp(-z))  where z = (x - μ) / β
        ///
        /// Time: O(1) | Space: O(1)
        pub fn cdf(self: Self, x: T) T {
            const z = (x - self.mu) / self.beta;
            return @exp(-@exp(-z));
        }

        /// Quantile function (inverse CDF)
        ///
        /// Q(p) = μ - β × ln(-ln(p))
        ///
        /// Time: O(1) | Space: O(1)
        pub fn quantile(self: Self, p: T) DistributionError!T {
            if (p < 0.0 or p > 1.0) return error.InvalidProbability;
            if (p == 0.0) return -math.inf(T);
            if (p == 1.0) return math.inf(T);
            return self.mu - self.beta * @log(-@log(p));
        }

        /// Generate random sample using inverse transform method
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sample(self: Self, rng: std.Random) T {
            const u = rng.float(T);
            const safe_u = @max(1e-10, @min(1.0 - 1e-10, u));
            return self.mu - self.beta * @log(-@log(safe_u));
        }

        /// Survival function (complementary CDF)
        ///
        /// S(x) = 1 - F(x) = 1 - exp(-exp(-z))
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sf(self: Self, x: T) T {
            return 1.0 - self.cdf(x);
        }

        /// Mean (expected value)
        ///
        /// E[X] = μ + β × γ  where γ ≈ 0.5772156649 (Euler-Mascheroni constant)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mean(self: Self) T {
            return self.mu + self.beta * euler_mascheroni;
        }

        /// Variance
        ///
        /// Var[X] = π² × β² / 6
        ///
        /// Time: O(1) | Space: O(1)
        pub fn variance(self: Self) T {
            return (math.pi * math.pi * self.beta * self.beta) / 6.0;
        }

        /// Mode (most likely value)
        ///
        /// Mode = μ (location parameter)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mode(self: Self) T {
            return self.mu;
        }

        /// Median (50th percentile)
        ///
        /// Median = μ - β × ln(ln(2))
        ///
        /// Time: O(1) | Space: O(1)
        pub fn median(self: Self) T {
            return self.mu - self.beta * @log(@log(2.0));
        }

        /// Validate internal invariants: β > 0, μ and β are finite
        ///
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.beta <= 0.0) return error.InvalidParameter;
            if (!math.isFinite(self.mu) or !math.isFinite(self.beta)) return error.InvalidParameter;
        }
    };
}

// ============================================================================
// Gumbel Distribution Tests
// ============================================================================

test "Gumbel: init with valid parameters" {
    const dist = try Gumbel(f64).init(0.0, 1.0);
    try testing.expectEqual(0.0, dist.mu);
    try testing.expectEqual(1.0, dist.beta);

    const dist2 = try Gumbel(f64).init(2.0, 0.5);
    try testing.expectEqual(2.0, dist2.mu);
    try testing.expectEqual(0.5, dist2.beta);
}

test "Gumbel: init rejects invalid parameters" {
    try testing.expectError(error.InvalidParameter, Gumbel(f64).init(0.0, -1.0));
    try testing.expectError(error.InvalidParameter, Gumbel(f64).init(0.0, 0.0));
    try testing.expectError(error.InvalidParameter, Gumbel(f64).init(math.inf(f64), 1.0));
    try testing.expectError(error.InvalidParameter, Gumbel(f64).init(0.0, math.inf(f64)));
    try testing.expectError(error.InvalidParameter, Gumbel(f64).init(math.nan(f64), 1.0));
    try testing.expectError(error.InvalidParameter, Gumbel(f64).init(0.0, math.nan(f64)));
}

test "Gumbel: PDF at mode equals maximum" {
    const dist = try Gumbel(f64).init(0.0, 1.0);
    const pdf_mode = dist.pdf(0.0);
    const expected = (1.0 / 1.0) * @exp(-1.0);
    try expectApproxEqRel(expected, pdf_mode, 1e-10);

    const dist2 = try Gumbel(f64).init(2.0, 0.5);
    const pdf_mode2 = dist2.pdf(2.0);
    const expected2 = (1.0 / 0.5) * @exp(-1.0);
    try expectApproxEqRel(expected2, pdf_mode2, 1e-10);
}

test "Gumbel: PDF manual calculation" {
    const dist = try Gumbel(f64).init(0.0, 1.0);

    const pdf_0 = dist.pdf(0.0);
    try expectApproxEqRel(@exp(-1.0), pdf_0, 1e-10);

    const pdf_1 = dist.pdf(1.0);
    const expected_1 = @exp(-(1.0 + @exp(-1.0)));
    try expectApproxEqRel(expected_1, pdf_1, 1e-10);

    const dist2 = try Gumbel(f64).init(2.0, 0.5);
    const pdf_2 = dist2.pdf(2.0);
    const expected_2 = (1.0 / 0.5) * @exp(-1.0);
    try expectApproxEqRel(expected_2, pdf_2, 1e-10);
}

test "Gumbel: CDF properties" {
    const dist = try Gumbel(f64).init(0.0, 1.0);

    const cdf_mode = dist.cdf(0.0);
    try expectApproxEqRel(@exp(-1.0), cdf_mode, 1e-10);

    const cdf_neg_large = dist.cdf(-100.0);
    try testing.expect(cdf_neg_large < 0.01);

    const cdf_pos_large = dist.cdf(100.0);
    try testing.expect(cdf_pos_large > 0.99);

    const cdf_1 = dist.cdf(-5.0);
    const cdf_2 = dist.cdf(0.0);
    const cdf_3 = dist.cdf(5.0);
    try testing.expect(cdf_1 <= cdf_2);
    try testing.expect(cdf_2 <= cdf_3);
}

test "Gumbel: CDF manual calculation" {
    const dist = try Gumbel(f64).init(0.0, 1.0);

    const cdf_0 = dist.cdf(0.0);
    try expectApproxEqRel(@exp(-1.0), cdf_0, 1e-10);

    const cdf_1 = dist.cdf(1.0);
    const expected_1 = @exp(-@exp(-1.0));
    try expectApproxEqRel(expected_1, cdf_1, 1e-10);

    const cdf_neg1 = dist.cdf(-1.0);
    const expected_neg1 = @exp(-@exp(1.0));
    try expectApproxEqRel(expected_neg1, cdf_neg1, 1e-10);
}

test "Gumbel: quantile properties" {
    const dist = try Gumbel(f64).init(0.0, 1.0);

    const q_median = try dist.quantile(0.5);
    const expected_median = -@log(@log(2.0));
    try expectApproxEqRel(expected_median, q_median, 1e-10);

    const q_10 = try dist.quantile(0.1);
    try testing.expect(q_10 < q_median);

    const q_90 = try dist.quantile(0.9);
    try testing.expect(q_90 > q_median);

    const q_0 = try dist.quantile(0.0);
    try testing.expect(math.isNegativeInf(q_0));

    const q_1 = try dist.quantile(1.0);
    try testing.expect(math.isPositiveInf(q_1));
}

test "Gumbel: quantile error handling" {
    const dist = try Gumbel(f64).init(0.0, 1.0);
    try testing.expectError(error.InvalidProbability, dist.quantile(-0.1));
    try testing.expectError(error.InvalidProbability, dist.quantile(1.1));
}

test "Gumbel: quantile-CDF roundtrip" {
    const dist = try Gumbel(f64).init(0.0, 1.0);
    const test_probs = [_]f64{ 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99 };
    for (test_probs) |p| {
        const q = try dist.quantile(p);
        const p_back = dist.cdf(q);
        try expectApproxEqRel(p, p_back, 1e-8);
    }
}

test "Gumbel: mean and variance" {
    const dist = try Gumbel(f64).init(0.0, 1.0);

    const m = dist.mean();
    const expected_mean = 0.5772156649015329;
    try expectApproxEqRel(expected_mean, m, 1e-10);

    const v = dist.variance();
    const expected_variance = (math.pi * math.pi) / 6.0;
    try expectApproxEqRel(expected_variance, v, 1e-10);

    const dist2 = try Gumbel(f64).init(2.0, 0.5);
    try expectApproxEqRel(2.0 + 0.5 * 0.5772156649015329, dist2.mean(), 1e-10);
    try expectApproxEqRel((math.pi * math.pi * 0.25) / 6.0, dist2.variance(), 1e-10);
}

test "Gumbel: mode and median" {
    const dist = try Gumbel(f64).init(0.0, 1.0);
    try testing.expectEqual(0.0, dist.mode());

    const med = dist.median();
    const expected_median = -@log(@log(2.0));
    try expectApproxEqRel(expected_median, med, 1e-10);

    const dist2 = try Gumbel(f64).init(2.5, 0.5);
    try testing.expectEqual(2.5, dist2.mode());
    try expectApproxEqRel(2.5 - 0.5 * @log(@log(2.0)), dist2.median(), 1e-10);
}

test "Gumbel: survival function" {
    const dist = try Gumbel(f64).init(0.0, 1.0);

    const test_x = [_]f64{ -5.0, -1.0, 0.0, 1.0, 5.0 };
    for (test_x) |x| {
        const sf = dist.sf(x);
        const cdf = dist.cdf(x);
        try expectApproxEqRel(1.0 - cdf, sf, 1e-10);
    }

    try expectApproxEqRel(1.0 - @exp(-1.0), dist.sf(0.0), 1e-10);
}

test "Gumbel: logpdf matches log of pdf" {
    const dist = try Gumbel(f64).init(0.0, 1.0);

    const test_x = [_]f64{ -5.0, -1.0, 0.0, 1.0, 5.0 };
    for (test_x) |x| {
        const log_pdf = dist.logpdf(x);
        const pdf = dist.pdf(x);
        try expectApproxEqRel(@log(pdf), log_pdf, 1e-10);
    }
}

test "Gumbel: sampling produces values in reasonable range" {
    const dist = try Gumbel(f64).init(0.0, 1.0);
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();
    var count_reasonable: usize = 0;
    const n_samples = 1000;
    for (0..n_samples) |_| {
        const x = dist.sample(rng);
        if (x >= -5.0 and x <= 10.0) count_reasonable += 1;
    }
    const ratio = @as(f64, @floatFromInt(count_reasonable)) / @as(f64, @floatFromInt(n_samples));
    try testing.expect(ratio >= 0.80);
}

test "Gumbel: f32 precision support" {
    const dist = try Gumbel(f32).init(0.0, 1.0);
    try expectApproxEqRel(dist.pdf(0.0), @as(f32, @exp(-1.0)), 1e-5);
    try expectApproxEqRel(dist.cdf(0.0), @as(f32, @exp(-1.0)), 1e-5);
    _ = try dist.quantile(0.5);
    _ = dist.mean();
    _ = dist.variance();
    try testing.expectEqual(@as(f32, 0.0), dist.mode());
    _ = dist.median();
    _ = dist.logpdf(0.0);
    _ = dist.sf(0.0);
}

test "Gumbel: memory safety" {
    const allocator = testing.allocator;
    _ = allocator;

    for (0..10) |_| {
        const dist = try Gumbel(f64).init(0.0, 1.0);
        _ = dist.pdf(1.0);
        try expectApproxEqRel(dist.cdf(0.0), @exp(-1.0), 1e-10);
        _ = try dist.quantile(0.5);
        _ = dist.mean();
        _ = dist.variance();
        _ = dist.mode();
        _ = dist.median();
        _ = dist.logpdf(1.0);
        _ = dist.sf(1.0);
    }
}

// ============================================================================
// Bernoulli Distribution Tests
// ============================================================================

test "Bernoulli: init with valid parameters" {
    const dist = try Bernoulli(f64).init(0.3);
    try expectEqual(0.3, dist.p);

    const dist2 = try Bernoulli(f64).init(1.0);
    try expectEqual(1.0, dist2.p);

    const dist3 = try Bernoulli(f64).init(0.5);
    try expectEqual(0.5, dist3.p);
}

test "Bernoulli: init rejects invalid probabilities" {
    // p = 0 is invalid
    try expectError(error.InvalidProbability, Bernoulli(f64).init(0.0));

    // p < 0 is invalid
    try expectError(error.InvalidProbability, Bernoulli(f64).init(-0.1));

    // p > 1 is invalid
    try expectError(error.InvalidProbability, Bernoulli(f64).init(1.1));

    // Non-finite values
    try expectError(error.InvalidProbability, Bernoulli(f64).init(math.inf(f64)));
    try expectError(error.InvalidProbability, Bernoulli(f64).init(math.nan(f64)));
}

test "Bernoulli: pmf at k=0" {
    const dist = try Bernoulli(f64).init(0.3);
    // P(X=0) = 1 - p = 0.7
    try expectApproxEqRel(0.7, dist.pmf(0), 1e-10);

    const dist2 = try Bernoulli(f64).init(0.5);
    try expectApproxEqRel(0.5, dist2.pmf(0), 1e-10);
}

test "Bernoulli: pmf at k=1" {
    const dist = try Bernoulli(f64).init(0.3);
    // P(X=1) = p = 0.3
    try expectApproxEqRel(0.3, dist.pmf(1), 1e-10);

    const dist2 = try Bernoulli(f64).init(0.7);
    try expectApproxEqRel(0.7, dist2.pmf(1), 1e-10);
}

test "Bernoulli: pmf outside support" {
    const dist = try Bernoulli(f64).init(0.3);
    // P(X=k) = 0 for k ∉ {0, 1}
    try expectEqual(0.0, dist.pmf(2));
    try expectEqual(0.0, dist.pmf(5));
    try expectEqual(0.0, dist.pmf(100));
}

test "Bernoulli: cdf at k=0" {
    const dist = try Bernoulli(f64).init(0.3);
    // CDF(0) = P(X≤0) = P(X=0) = 1 - p = 0.7
    try expectApproxEqRel(0.7, dist.cdf(0), 1e-10);

    const dist2 = try Bernoulli(f64).init(0.5);
    try expectApproxEqRel(0.5, dist2.cdf(0), 1e-10);
}

test "Bernoulli: cdf at k=1" {
    const dist = try Bernoulli(f64).init(0.3);
    // CDF(1) = P(X≤1) = 1.0
    try expectApproxEqRel(1.0, dist.cdf(1), 1e-10);

    const dist2 = try Bernoulli(f64).init(0.999);
    try expectApproxEqRel(1.0, dist2.cdf(1), 1e-10);
}

test "Bernoulli: cdf at negative k" {
    const dist = try Bernoulli(f64).init(0.3);
    // CDF(-1) = 0 (before support begins)
    try expectEqual(0.0, dist.cdf(-1));
}

test "Bernoulli: cdf at large k" {
    const dist = try Bernoulli(f64).init(0.3);
    // CDF(k) = 1.0 for k >= 1
    try expectApproxEqRel(1.0, dist.cdf(5), 1e-10);
    try expectApproxEqRel(1.0, dist.cdf(100), 1e-10);
}

test "Bernoulli: mean equals p" {
    const dist1 = try Bernoulli(f64).init(0.3);
    try expectApproxEqRel(0.3, dist1.mean(), 1e-10);

    const dist2 = try Bernoulli(f64).init(0.7);
    try expectApproxEqRel(0.7, dist2.mean(), 1e-10);

    const dist3 = try Bernoulli(f64).init(1.0);
    try expectApproxEqRel(1.0, dist3.mean(), 1e-10);
}

test "Bernoulli: variance p*(1-p)" {
    const dist1 = try Bernoulli(f64).init(0.5);
    // Variance = 0.5 * 0.5 = 0.25 (maximum)
    try expectApproxEqRel(0.25, dist1.variance(), 1e-10);

    const dist2 = try Bernoulli(f64).init(0.3);
    // Variance = 0.3 * 0.7 = 0.21
    try expectApproxEqRel(0.21, dist2.variance(), 1e-10);

    const dist3 = try Bernoulli(f64).init(0.8);
    // Variance = 0.8 * 0.2 = 0.16
    try expectApproxEqRel(0.16, dist3.variance(), 1e-10);
}

test "Bernoulli: sample for p=1.0" {
    var prng = std.Random.DefaultPrng.init(99999);
    const rng = prng.random();
    const dist = try Bernoulli(f64).init(1.0);
    // For p=1.0, every sample should be 1 (float(T) is always < 1.0)
    for (0..50) |_| {
        const x = dist.sample(rng);
        try expectEqual(1, x);
    }
}

test "Bernoulli: sample results in {0, 1}" {
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    const dist = try Bernoulli(f64).init(0.5);
    for (0..100) |_| {
        const x = dist.sample(rng);
        try testing.expect(x == 0 or x == 1);
    }
}

test "Bernoulli: logpmf at k=1 for p=0.5" {
    const dist = try Bernoulli(f64).init(0.5);
    // logpmf(1) = log(p) = log(0.5) ≈ -0.693147
    try expectApproxEqRel(@log(0.5), dist.logpmf(1), 1e-10);
}

test "Bernoulli: logpmf at k=0 for p=0.5" {
    const dist = try Bernoulli(f64).init(0.5);
    // logpmf(0) = log(1-p) = log(0.5) ≈ -0.693147
    try expectApproxEqRel(@log(0.5), dist.logpmf(0), 1e-10);
}

test "Bernoulli: survival function sf(0)" {
    const dist = try Bernoulli(f64).init(0.3);
    // sf(0) = P(X > 0) = p = 0.3
    try expectApproxEqRel(0.3, dist.sf(0), 1e-10);

    const dist2 = try Bernoulli(f64).init(0.7);
    try expectApproxEqRel(0.7, dist2.sf(0), 1e-10);
}

test "Bernoulli: survival function sf(1)" {
    const dist = try Bernoulli(f64).init(0.3);
    // sf(1) = P(X > 1) = 0.0 (support ends at 1)
    try expectEqual(0.0, dist.sf(1));

    const dist2 = try Bernoulli(f64).init(0.999);
    try expectEqual(0.0, dist2.sf(1));
}

test "Bernoulli: f32 precision support" {
    const dist = try Bernoulli(f32).init(0.5);
    try expectApproxEqRel(@as(f32, 0.5), dist.pmf(0), 1e-5);
    try expectApproxEqRel(@as(f32, 0.5), dist.pmf(1), 1e-5);
    try expectApproxEqRel(@as(f32, 0.5), dist.cdf(0), 1e-5);
    try expectApproxEqRel(@as(f32, 1.0), dist.cdf(1), 1e-5);
    _ = dist.mean();
    _ = dist.variance();
    _ = dist.logpmf(1);
    _ = dist.sf(0);
}

test "Bernoulli: memory safety" {
    const allocator = testing.allocator;
    _ = allocator;

    for (0..100) |_| {
        const dist = try Bernoulli(f64).init(0.5);
        _ = dist.pmf(0);
        _ = dist.pmf(1);
        _ = dist.cdf(0);
        _ = dist.cdf(1);
        _ = dist.mean();
        _ = dist.variance();
        _ = dist.logpmf(0);
        _ = dist.logpmf(1);
        _ = dist.sf(0);
    }
}

test "Bernoulli: pmf sums to 1.0" {
    const dist = try Bernoulli(f64).init(0.3);
    const sum = dist.pmf(0) + dist.pmf(1);
    try expectApproxEqRel(1.0, sum, 1e-10);

    const dist2 = try Bernoulli(f64).init(0.7);
    const sum2 = dist2.pmf(0) + dist2.pmf(1);
    try expectApproxEqRel(1.0, sum2, 1e-10);
}

// ============================================================================
// Geometric Distribution Tests
// ============================================================================

test "Geometric: init with valid parameters" {
    const dist = try Geometric(f64).init(0.5);
    try expectEqual(0.5, dist.p);

    const dist2 = try Geometric(f64).init(1.0);
    try expectEqual(1.0, dist2.p);

    const dist3 = try Geometric(f64).init(0.25);
    try expectEqual(0.25, dist3.p);
}

test "Geometric: init rejects invalid probabilities" {
    // p = 0 is invalid
    try expectError(error.InvalidProbability, Geometric(f64).init(0.0));

    // p < 0 is invalid
    try expectError(error.InvalidProbability, Geometric(f64).init(-0.1));

    // p > 1 is invalid
    try expectError(error.InvalidProbability, Geometric(f64).init(1.1));

    // Non-finite values
    try expectError(error.InvalidProbability, Geometric(f64).init(math.inf(f64)));
    try expectError(error.InvalidProbability, Geometric(f64).init(math.nan(f64)));
}

test "Geometric: pmf at k=1" {
    const dist = try Geometric(f64).init(0.5);
    // P(X=1) = p = 0.5
    try expectApproxEqRel(0.5, dist.pmf(1), 1e-10);

    const dist2 = try Geometric(f64).init(0.25);
    try expectApproxEqRel(0.25, dist2.pmf(1), 1e-10);
}

test "Geometric: pmf at k=2 and k=3" {
    const dist = try Geometric(f64).init(0.5);
    // P(X=2) = (1-p) * p = 0.5 * 0.5 = 0.25
    try expectApproxEqRel(0.25, dist.pmf(2), 1e-10);

    // P(X=3) = (1-p)^2 * p = 0.25 * 0.5 = 0.125
    try expectApproxEqRel(0.125, dist.pmf(3), 1e-10);
}

test "Geometric: pmf at k=0" {
    const dist = try Geometric(f64).init(0.5);
    // P(X=0) = 0 (support starts at k=1)
    try expectEqual(0.0, dist.pmf(0));

    const dist2 = try Geometric(f64).init(0.25);
    try expectEqual(0.0, dist2.pmf(0));
}

test "Geometric: cdf at k=1" {
    const dist = try Geometric(f64).init(0.5);
    // CDF(1) = P(X≤1) = p = 0.5
    try expectApproxEqRel(0.5, dist.cdf(1), 1e-10);

    const dist2 = try Geometric(f64).init(0.25);
    try expectApproxEqRel(0.25, dist2.cdf(1), 1e-10);
}

test "Geometric: cdf at k=2 and k=3" {
    const dist = try Geometric(f64).init(0.5);
    // CDF(2) = 1 - (1-p)^2 = 1 - 0.25 = 0.75
    try expectApproxEqRel(0.75, dist.cdf(2), 1e-10);

    // CDF(3) = 1 - (1-p)^3 = 1 - 0.125 = 0.875
    try expectApproxEqRel(0.875, dist.cdf(3), 1e-10);
}

test "Geometric: cdf at k=0" {
    const dist = try Geometric(f64).init(0.5);
    // CDF(0) = 0 (before support begins)
    try expectEqual(0.0, dist.cdf(0));

    const dist2 = try Geometric(f64).init(0.999);
    try expectEqual(0.0, dist2.cdf(0));
}

test "Geometric: quantile for p=0.5" {
    const dist = try Geometric(f64).init(0.5);
    // quantile(0.5) should be around mean
    const q50 = try dist.quantile(0.5);
    try testing.expect(q50 >= 1);
}

test "Geometric: quantile values at key points" {
    const dist = try Geometric(f64).init(0.5);
    // Quantiles should be at least 1
    const q25 = try dist.quantile(0.25);
    const q75 = try dist.quantile(0.75);
    try testing.expect(q25 >= 1);
    try testing.expect(q75 >= 1);
    try testing.expect(q25 <= q75);
}

test "Geometric: mean 1/p" {
    const dist1 = try Geometric(f64).init(0.5);
    // Mean = 1/p = 2.0
    try expectApproxEqRel(2.0, dist1.mean(), 1e-10);

    const dist2 = try Geometric(f64).init(0.25);
    // Mean = 1/p = 4.0
    try expectApproxEqRel(4.0, dist2.mean(), 1e-10);

    const dist3 = try Geometric(f64).init(0.1);
    // Mean = 1/p = 10.0
    try expectApproxEqRel(10.0, dist3.mean(), 1e-10);
}

test "Geometric: variance (1-p)/p²" {
    const dist1 = try Geometric(f64).init(0.5);
    // Variance = (1-p)/p² = 0.5 / 0.25 = 2.0
    try expectApproxEqRel(2.0, dist1.variance(), 1e-10);

    const dist2 = try Geometric(f64).init(0.25);
    // Variance = 0.75 / 0.0625 = 12.0
    try expectApproxEqRel(12.0, dist2.variance(), 1e-10);
}

test "Geometric: sample returns k >= 1" {
    var prng = std.Random.DefaultPrng.init(54321);
    const rng = prng.random();

    const dist = try Geometric(f64).init(0.5);
    for (0..100) |_| {
        const x = dist.sample(rng);
        try testing.expect(x >= 1);
    }
}

test "Geometric: logpmf at k=1" {
    const dist = try Geometric(f64).init(0.5);
    // logpmf(1) = log(p) = log(0.5) ≈ -0.693147
    try expectApproxEqRel(@log(0.5), dist.logpmf(1), 1e-10);

    const dist2 = try Geometric(f64).init(0.25);
    try expectApproxEqRel(@log(0.25), dist2.logpmf(1), 1e-10);
}

test "Geometric: survival function sf(1)" {
    const dist = try Geometric(f64).init(0.5);
    // sf(1) = P(X > 1) = 1 - CDF(1) = 1 - 0.5 = 0.5
    try expectApproxEqRel(0.5, dist.sf(1), 1e-10);

    const dist2 = try Geometric(f64).init(0.25);
    // sf(1) = 1 - 0.25 = 0.75
    try expectApproxEqRel(0.75, dist2.sf(1), 1e-10);
}

test "Geometric: survival function sf(2)" {
    const dist = try Geometric(f64).init(0.5);
    // sf(2) = P(X > 2) = 1 - CDF(2) = 1 - 0.75 = 0.25
    try expectApproxEqRel(0.25, dist.sf(2), 1e-10);

    const dist2 = try Geometric(f64).init(0.25);
    // sf(2) = 1 - (1 - (0.75)^2) = (0.75)^2 = 0.5625
    try expectApproxEqRel(0.5625, dist2.sf(2), 1e-10);
}

test "Geometric: mode always 1" {
    const dist1 = try Geometric(f64).init(0.3);
    try expectEqual(1, dist1.mode());

    const dist2 = try Geometric(f64).init(0.7);
    try expectEqual(1, dist2.mode());

    const dist3 = try Geometric(f64).init(0.999);
    try expectEqual(1, dist3.mode());
}

test "Geometric: f32 precision support" {
    const dist = try Geometric(f32).init(0.5);
    try expectApproxEqRel(@as(f32, 0.5), dist.pmf(1), 1e-5);
    try expectApproxEqRel(@as(f32, 0.5), dist.cdf(1), 1e-5);
    _ = try dist.quantile(0.5);
    _ = dist.mean();
    _ = dist.variance();
    _ = dist.mode();
    _ = dist.logpmf(1);
    _ = dist.sf(1);
}

test "Geometric: memory safety" {
    const allocator = testing.allocator;
    _ = allocator;

    for (0..100) |_| {
        const dist = try Geometric(f64).init(0.5);
        _ = dist.pmf(1);
        _ = dist.pmf(2);
        _ = dist.cdf(1);
        _ = dist.cdf(2);
        _ = try dist.quantile(0.5);
        _ = dist.mean();
        _ = dist.variance();
        _ = dist.mode();
        _ = dist.logpmf(1);
        _ = dist.sf(1);
    }
}

test "Geometric: pmf series sums to ~1.0" {
    const dist = try Geometric(f64).init(0.5);
    // Sum pmf(k) for k=1..50 should be approximately 1.0
    var sum: f64 = 0.0;
    for (1..51) |k| {
        sum += dist.pmf(k);
    }
    try expectApproxEqRel(1.0, sum, 1e-6);
}

// ============================================================================
// NegativeBinomial Distribution
// ============================================================================

/// NegativeBinomial distribution — number of failures before r-th success
///
/// Parameters:
///   - r: number of successes (r ≥ 1)
///   - p: success probability (0 < p ≤ 1)
///
/// Support: k ∈ {0, 1, 2, ...} (number of failures)
///
/// PMF: C(k+r-1, k) * p^r * (1-p)^k
/// CDF: Regularized incomplete beta function I_p(r, k+1)
/// Mean: r*(1-p)/p
/// Variance: r*(1-p)/p²
/// Mode: floor((r-1)*(1-p)/p) for r > 1, else 0
///
/// Time: O(1) for pmf/mean/variance/mode; O(k) for cdf/sf/quantile
pub fn NegativeBinomial(comptime T: type) type {
    return struct {
        r: u64,
        p: T,

        const Self = @This();

        /// Initialize NegativeBinomial(r, p)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(r: u64, p: T) error{InvalidParameter}!Self {
            if (r < 1) return error.InvalidParameter;
            if (p <= 0.0 or p > 1.0) return error.InvalidParameter;
            return Self{ .r = r, .p = p };
        }

        /// Probability mass function at k
        /// PMF(k) = C(k+r-1, k) * p^r * (1-p)^k
        ///
        /// Time: O(1) | Space: O(1)
        pub fn pmf(self: Self, k: u64) T {
            // PMF(k) = C(k+r-1, k) * p^r * (1-p)^k
            const log_pmf_val = self.logpmf(k);
            return @exp(log_pmf_val);
        }

        /// Cumulative distribution function at k
        /// CDF(k) = P(X ≤ k)
        ///
        /// Time: O(k) | Space: O(1)
        pub fn cdf(self: Self, k: i64) T {
            if (k < 0) return 0.0;

            // CDF(k) = sum of PMF from j=0 to k
            const k_u: u64 = @intCast(k);
            var sum: T = 0.0;
            var j: u64 = 0;
            while (j <= k_u) : (j += 1) {
                sum += self.pmf(j);
            }
            return @min(sum, 1.0);
        }

        /// Log probability mass function at k
        /// logpmf(k) = log(pmf(k))
        ///
        /// Time: O(1) | Space: O(1)
        pub fn logpmf(self: Self, k: u64) T {
            // log PMF(k) = log C(k+r-1, k) + r*log(p) + k*log(1-p)
            // C(k+r-1, k) = (k+r-1)! / (k! * (r-1)!)
            // log C(n, k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
            const k_f = @as(T, @floatFromInt(k));
            const r_f = @as(T, @floatFromInt(self.r));

            const n = k + self.r - 1;
            const n_f = @as(T, @floatFromInt(n));

            // log_binomial_coeff = lgamma(k+r) - lgamma(k+1) - lgamma(r)
            const log_binom_coeff = logGamma(n_f + 1.0) - logGamma(k_f + 1.0) - logGamma(r_f);

            // When k==0, 0*log(0) is defined as 0 by convention (0^0=1 in combinatorics)
            const log_q_term = if (k == 0) @as(T, 0.0) else k_f * @log(1.0 - self.p);
            const log_pmf_val = log_binom_coeff + r_f * @log(self.p) + log_q_term;
            return log_pmf_val;
        }

        /// Survival function at k
        /// sf(k) = P(X > k) = 1 - CDF(k)
        ///
        /// Time: O(k) | Space: O(1)
        pub fn sf(self: Self, k: i64) T {
            if (k < 0) return 1.0;
            return 1.0 - self.cdf(k);
        }

        /// Quantile function (inverse CDF)
        /// Returns k such that CDF(k) ≥ prob
        ///
        /// Time: O(k) linear search | Space: O(1)
        pub fn quantile(self: Self, prob: T) error{ OutOfDomain }!u64 {
            if (prob < 0.0 or prob > 1.0) return error.OutOfDomain;
            if (prob == 0.0) return 0;

            // Linear search for smallest k such that CDF(k) >= prob
            var k: u64 = 0;
            var cumulative: T = 0.0;
            while (k < 10000) : (k += 1) {
                cumulative += self.pmf(k);
                if (cumulative >= prob) return k;
            }
            return k;
        }

        /// Mean of the distribution
        /// Mean = r*(1-p)/p
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mean(self: Self) T {
            const r_f = @as(T, @floatFromInt(self.r));
            return r_f * (1.0 - self.p) / self.p;
        }

        /// Variance of the distribution
        /// Variance = r*(1-p)/p²
        ///
        /// Time: O(1) | Space: O(1)
        pub fn variance(self: Self) T {
            const r_f = @as(T, @floatFromInt(self.r));
            return r_f * (1.0 - self.p) / (self.p * self.p);
        }

        /// Mode of the distribution
        /// Mode = floor((r-1)*(1-p)/p) for r > 1, else 0
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mode(self: Self) u64 {
            if (self.r <= 1) return 0;
            const r_f = @as(T, @floatFromInt(self.r - 1));
            const mode_f = r_f * (1.0 - self.p) / self.p;
            return @intFromFloat(@floor(mode_f));
        }

        /// Draw a random sample from the distribution
        ///
        /// Time: O(r) | Space: O(1)
        pub fn sample(self: Self, rng: anytype) u64 {
            // NegativeBinomial(r,p) = sum of r independent Geometric(p) samples
            // Each Geometric(p) = number of failures before first success
            var total: u64 = 0;
            var i: u64 = 0;
            while (i < self.r) : (i += 1) {
                while (rng.float(T) >= self.p) {
                    total += 1;
                }
            }
            return total;
        }

        /// Validate internal invariants: r ≥ 1 and 0 < p ≤ 1
        ///
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.r < 1) return error.InvalidParameter;
            if (self.p <= 0.0 or self.p > 1.0) return error.InvalidParameter;
        }
    };
}

// Tests for NegativeBinomial Distribution

test "NegativeBinomial: init valid parameters" {
    const NB = NegativeBinomial(f64);
    const nb = try NB.init(3, 0.5);
    try expectEqual(nb.r, 3);
    try expectEqual(nb.p, 0.5);
}

test "NegativeBinomial: init invalid p=0 returns error" {
    const NB = NegativeBinomial(f64);
    const result = NB.init(3, 0.0);
    try expectEqual(error.InvalidParameter, result);
}

test "NegativeBinomial: init invalid p>1 returns error" {
    const NB = NegativeBinomial(f64);
    const result = NB.init(3, 1.1);
    try expectEqual(error.InvalidParameter, result);
}

test "NegativeBinomial: init invalid r=0 returns error" {
    const NB = NegativeBinomial(f64);
    const result = NB.init(0, 0.5);
    try expectEqual(error.InvalidParameter, result);
}

test "NegativeBinomial: pmf k=0 is p^r" {
    const nb = try NegativeBinomial(f64).init(3, 0.5);
    // pmf(0) = p^r = 0.5^3 = 0.125
    try expectApproxEqRel(0.125, nb.pmf(0), 1e-10);
}

test "NegativeBinomial: pmf k=1" {
    const nb = try NegativeBinomial(f64).init(3, 0.5);
    // pmf(1) = C(3,1) * 0.5^3 * 0.5^1 = 3 * 0.0625 = 0.1875
    try expectApproxEqRel(0.1875, nb.pmf(1), 1e-10);
}

test "NegativeBinomial: pmf k=2" {
    const nb = try NegativeBinomial(f64).init(3, 0.5);
    // pmf(2) = C(4,2) * 0.5^3 * 0.5^2 = 6 * 0.03125 = 0.1875
    try expectApproxEqRel(0.1875, nb.pmf(2), 1e-10);
}

test "NegativeBinomial: pmf probabilities sum to approximately 1" {
    const nb = try NegativeBinomial(f64).init(3, 0.5);
    var sum: f64 = 0.0;
    for (0..100) |k| {
        sum += nb.pmf(k);
    }
    try expectApproxEqRel(1.0, sum, 1e-5);
}

test "NegativeBinomial: cdf negative k returns 0" {
    const nb = try NegativeBinomial(f64).init(3, 0.5);
    try expectEqual(0.0, nb.cdf(-1));
}

test "NegativeBinomial: cdf k=0 equals pmf(0)" {
    const nb = try NegativeBinomial(f64).init(3, 0.5);
    // cdf(0) = pmf(0) = 0.125
    try expectApproxEqRel(0.125, nb.cdf(0), 1e-10);
}

test "NegativeBinomial: cdf is monotone non-decreasing" {
    const nb = try NegativeBinomial(f64).init(3, 0.5);
    for (0..9) |k| {
        const cdf_k = nb.cdf(@intCast(k));
        const cdf_k1 = nb.cdf(@intCast(k + 1));
        try testing.expect(cdf_k <= cdf_k1);
    }
}

test "NegativeBinomial: cdf large k approaches 1" {
    const nb = try NegativeBinomial(f64).init(3, 0.5);
    const cdf_100 = nb.cdf(100);
    try testing.expect(cdf_100 > 0.9999);
}

test "NegativeBinomial: logpmf matches log of pmf" {
    const nb = try NegativeBinomial(f64).init(3, 0.5);
    for (0..5) |k| {
        const expected = @log(nb.pmf(k));
        const actual = nb.logpmf(k);
        try expectApproxEqRel(expected, actual, 1e-10);
    }
}

test "NegativeBinomial: sf k=0 equals 1 - cdf(0)" {
    const nb = try NegativeBinomial(f64).init(3, 0.5);
    // sf(0) = P(X > 0) = 1 - cdf(0) = 1 - 0.125 = 0.875
    try expectApproxEqRel(0.875, nb.sf(0), 1e-10);
}

test "NegativeBinomial: sf plus cdf equals 1" {
    const nb = try NegativeBinomial(f64).init(3, 0.5);
    for (0..10) |k| {
        const sum = nb.sf(@intCast(k)) + nb.cdf(@intCast(k));
        try expectApproxEqRel(1.0, sum, 1e-10);
    }
}

test "NegativeBinomial: mean equals r*(1-p)/p" {
    const nb = try NegativeBinomial(f64).init(3, 0.5);
    // mean = 3*(1-0.5)/0.5 = 3.0
    try expectApproxEqRel(3.0, nb.mean(), 1e-10);
}

test "NegativeBinomial: variance equals r*(1-p)/p²" {
    const nb = try NegativeBinomial(f64).init(3, 0.5);
    // variance = 3*(1-0.5)/0.25 = 6.0
    try expectApproxEqRel(6.0, nb.variance(), 1e-10);
}

test "NegativeBinomial: mode k=0 when r=1" {
    const nb = try NegativeBinomial(f64).init(1, 0.7);
    // mode = floor((1-1)*0.3/0.7) = 0
    try expectEqual(0, nb.mode());
}

test "NegativeBinomial: mode for r=5 p=0.3" {
    const nb = try NegativeBinomial(f64).init(5, 0.3);
    // mode = floor((5-1)*0.7/0.3) = floor(9.33) = 9
    try expectEqual(9, nb.mode());
}

test "NegativeBinomial: quantile roundtrip" {
    const nb = try NegativeBinomial(f64).init(4, 0.4);
    for (3..9) |k_val| {
        const k: u64 = k_val;
        const cdf_k = nb.cdf(@intCast(k));
        const q = try nb.quantile(cdf_k);
        try expectEqual(k, q);
    }
}

test "NegativeBinomial: quantile p=0 returns 0" {
    const nb = try NegativeBinomial(f64).init(2, 0.5);
    const q = try nb.quantile(0.0);
    try expectEqual(0, q);
}

test "NegativeBinomial: quantile p approaching 1 returns large value" {
    const nb = try NegativeBinomial(f64).init(2, 0.3);
    const q = try nb.quantile(0.999);
    try testing.expect(q > 0);
}

test "NegativeBinomial: p=1.0 valid pmf k=0 is 1" {
    const nb = try NegativeBinomial(f64).init(3, 1.0);
    // All probability at k=0
    try expectApproxEqRel(1.0, nb.pmf(0), 1e-10);
    try expectApproxEqRel(0.0, nb.pmf(1), 1e-10);
}

test "NegativeBinomial: sample empirical mean near theoretical" {
    var prng = std.Random.DefaultPrng.init(54321);
    const rng = prng.random();

    // NB(3, 0.5): mean = r*(1-p)/p = 3*0.5/0.5 = 3.0
    const nb = try NegativeBinomial(f64).init(3, 0.5);
    var sum: u64 = 0;
    const n_samples = 2000;
    for (0..n_samples) |_| {
        sum += nb.sample(rng);
    }
    const empirical_mean = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(n_samples));
    // Empirical mean should be within 1.0 of theoretical mean (3.0) with high probability
    try testing.expect(empirical_mean >= 2.0 and empirical_mean <= 4.0);
}

test "NegativeBinomial: sample distribution mean approximates theoretical" {
    var prng = std.Random.DefaultPrng.init(54321);
    const rng = prng.random();

    const nb = try NegativeBinomial(f64).init(2, 0.5);
    const theoretical_mean = nb.mean();
    var sum: f64 = 0.0;
    for (0..200) |_| {
        const x = nb.sample(rng);
        sum += @floatFromInt(x);
    }
    const sample_mean = sum / 200.0;
    // Check within 3.0 of theoretical mean (relaxed for randomness)
    try testing.expect(@abs(sample_mean - theoretical_mean) < 3.0);
}

test "NegativeBinomial: f32 type precision" {
    const nb = try NegativeBinomial(f32).init(2, 0.5);
    try expectApproxEqRel(@as(f32, 0.25), nb.pmf(0), 1e-5);
    try expectApproxEqRel(@as(f32, 2.0), nb.mean(), 1e-5); // NB(2,0.5): mean = 2*(1-0.5)/0.5 = 2.0
}

test "NegativeBinomial: memory safety init-deinit loop" {
    for (0..10) |_| {
        const nb = try NegativeBinomial(f64).init(3, 0.5);
        _ = nb.pmf(1);
        _ = nb.pmf(2);
        _ = nb.cdf(1);
        _ = nb.cdf(2);
        _ = nb.mean();
        _ = nb.variance();
        _ = nb.mode();
        _ = nb.logpmf(1);
        _ = nb.sf(1);
    }
}

test "NegativeBinomial: large r parameter" {
    const nb = try NegativeBinomial(f64).init(10, 0.8);
    // pmf(0) = 0.8^10 ≈ 0.1074
    try expectApproxEqRel(0.1074, nb.pmf(0), 0.001);
    // mean = 10*0.2/0.8 = 2.5
    try expectApproxEqRel(2.5, nb.mean(), 1e-10);
}

// ============================================================================
// Hypergeometric Distribution
// ============================================================================

/// Hypergeometric distribution — number of successes in n draws from a finite population
///
/// Parameters:
///   - N: population size (N ≥ 1)
///   - K: number of success states in population (K ≤ N)
///   - n: number of draws without replacement (n ≤ N)
///
/// Support: k ∈ {max(0, n+K-N), ..., min(n, K)}
///
/// PMF: C(K,k) * C(N-K, n-k) / C(N,n)
/// CDF: sum of PMF from support_min to k
/// Mean: n*K/N
/// Variance: n*(K/N)*((N-K)/N)*(N-n)/(N-1)  [0 if N==1]
/// Mode: floor((n+1)*(K+1)/(N+2)) clamped to support
///
/// Time: O(1) for pmf/mean/variance/mode; O(k) for cdf/sf/quantile
pub fn Hypergeometric(comptime T: type) type {
    return struct {
        N: u64,
        K: u64,
        n: u64,

        const Self = @This();

        /// Initialize Hypergeometric(N, K, n)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(N: u64, K: u64, n: u64) error{InvalidParameter}!Self {
            if (N < 1) return error.InvalidParameter;
            if (K > N) return error.InvalidParameter;
            if (n > N) return error.InvalidParameter;
            return Self{ .N = N, .K = K, .n = n };
        }

        /// Minimum value in support: max(0, n + K - N)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn supportMin(self: Self) u64 {
            // Saturating subtraction: if n+K < N, result is 0
            const nK = self.n + self.K;
            if (nK <= self.N) return 0;
            return nK - self.N;
        }

        /// Maximum value in support: min(n, K)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn supportMax(self: Self) u64 {
            return @min(self.n, self.K);
        }

        /// Log probability mass function at k
        /// logPMF(k) = logBinom(K,k) + logBinom(N-K, n-k) - logBinom(N,n)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn logpmf(self: Self, k: u64) T {
            const s_min = self.supportMin();
            const s_max = self.supportMax();
            if (k < s_min or k > s_max) return @as(T, -std.math.inf(T));

            const K_f = @as(T, @floatFromInt(self.K));
            const k_f = @as(T, @floatFromInt(k));
            const NK_f = @as(T, @floatFromInt(self.N - self.K));
            const nk: u64 = self.n - k;  // safe because k <= min(n,K) <= n
            const nk_f = @as(T, @floatFromInt(nk));
            const N_f = @as(T, @floatFromInt(self.N));
            const n_f = @as(T, @floatFromInt(self.n));

            // logBinom(a, b) = lgamma(a+1) - lgamma(b+1) - lgamma(a-b+1)
            const log_binom_K_k = logGamma(K_f + 1.0) - logGamma(k_f + 1.0) - logGamma(K_f - k_f + 1.0);
            const log_binom_NK_nk = logGamma(NK_f + 1.0) - logGamma(nk_f + 1.0) - logGamma(NK_f - nk_f + 1.0);
            const log_binom_N_n = logGamma(N_f + 1.0) - logGamma(n_f + 1.0) - logGamma(N_f - n_f + 1.0);

            return log_binom_K_k + log_binom_NK_nk - log_binom_N_n;
        }

        /// Probability mass function at k
        /// PMF(k) = C(K,k) * C(N-K, n-k) / C(N,n)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn pmf(self: Self, k: u64) T {
            const lp = self.logpmf(k);
            if (lp == @as(T, -std.math.inf(T))) return 0.0;
            return @exp(lp);
        }

        /// Cumulative distribution function P(X <= k)
        ///
        /// Time: O(k) | Space: O(1)
        pub fn cdf(self: Self, k: i64) T {
            const s_max = self.supportMax();
            if (k < 0) return 0.0;
            const k_u: u64 = @intCast(k);
            if (k_u >= s_max) return 1.0;

            const s_min = self.supportMin();
            var sum: T = 0.0;
            var j = s_min;
            while (j <= k_u) : (j += 1) {
                sum += self.pmf(j);
            }
            return @min(sum, 1.0);
        }

        /// Survival function P(X > k) = 1 - CDF(k)
        ///
        /// Time: O(k) | Space: O(1)
        pub fn sf(self: Self, k: i64) T {
            if (k < 0) return 1.0;
            return 1.0 - self.cdf(k);
        }

        /// Quantile function — smallest k such that CDF(k) >= prob
        ///
        /// Time: O(support range) | Space: O(1)
        pub fn quantile(self: Self, prob: T) error{OutOfDomain}!u64 {
            if (prob < 0.0 or prob > 1.0) return error.OutOfDomain;
            const s_min = self.supportMin();
            const s_max = self.supportMax();
            if (prob == 0.0) return s_min;
            if (prob >= 1.0) return s_max;
            var cumulative: T = 0.0;
            var k = s_min;
            while (k <= s_max) : (k += 1) {
                cumulative += self.pmf(k);
                if (cumulative >= prob) return k;
            }
            return s_max;
        }

        /// Mean: n * K / N
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mean(self: Self) T {
            const n_f = @as(T, @floatFromInt(self.n));
            const K_f = @as(T, @floatFromInt(self.K));
            const N_f = @as(T, @floatFromInt(self.N));
            return n_f * K_f / N_f;
        }

        /// Variance: n * K * (N-K) * (N-n) / (N^2 * (N-1))
        /// Returns 0 when N == 1.
        ///
        /// Time: O(1) | Space: O(1)
        pub fn variance(self: Self) T {
            if (self.N == 1) return 0.0;
            const n_f = @as(T, @floatFromInt(self.n));
            const K_f = @as(T, @floatFromInt(self.K));
            const N_f = @as(T, @floatFromInt(self.N));
            const NK_f = @as(T, @floatFromInt(self.N - self.K));
            const Nn_f = @as(T, @floatFromInt(self.N - self.n));
            return n_f * K_f * NK_f * Nn_f / (N_f * N_f * (N_f - 1.0));
        }

        /// Mode: floor((n+1)*(K+1)/(N+2)) clamped to support
        ///
        /// Time: O(1) | Space: O(1)
        pub fn mode(self: Self) u64 {
            const n1 = @as(T, @floatFromInt(self.n + 1));
            const K1 = @as(T, @floatFromInt(self.K + 1));
            const N2 = @as(T, @floatFromInt(self.N + 2));
            const m = @floor(n1 * K1 / N2);
            const m_u: u64 = @intFromFloat(m);
            const s_min = self.supportMin();
            const s_max = self.supportMax();
            return @min(@max(m_u, s_min), s_max);
        }

        /// Sample using inverse-transform method (quantile of uniform)
        ///
        /// Time: O(support range) | Space: O(1)
        pub fn sample(self: Self, rng: std.Random) u64 {
            const u = rng.float(T);
            return self.quantile(u) catch self.supportMin();
        }

        /// Assert internal invariants
        ///
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: Self) !void {
            std.debug.assert(self.K <= self.N);
            std.debug.assert(self.n <= self.N);
            std.debug.assert(self.N >= 1);
        }
    };
}

// Tests for Hypergeometric Distribution

test "Hypergeometric: init valid parameters" {
    const hg = try Hypergeometric(f64).init(10, 4, 3);
    try expectEqual(hg.N, 10);
    try expectEqual(hg.K, 4);
    try expectEqual(hg.n, 3);
}

test "Hypergeometric: init K > N returns error" {
    const result = Hypergeometric(f64).init(10, 15, 3);
    try expectError(error.InvalidParameter, result);
}

test "Hypergeometric: init n > N returns error" {
    const result = Hypergeometric(f64).init(10, 4, 20);
    try expectError(error.InvalidParameter, result);
}

test "Hypergeometric: pmf basic case N=10 K=4 n=3 k=2" {
    const hg = try Hypergeometric(f64).init(10, 4, 3);
    // PMF(2) = C(4,2)*C(6,1)/C(10,3) = 6*6/120 = 0.3
    try expectApproxEqRel(0.3, hg.pmf(2), 1e-10);
}

test "Hypergeometric: pmf k=0 in basic case" {
    const hg = try Hypergeometric(f64).init(10, 4, 3);
    // PMF(0) = C(4,0)*C(6,3)/C(10,3) = 1*20/120 = 1/6 ≈ 0.1667
    try expectApproxEqRel(1.0 / 6.0, hg.pmf(0), 1e-10);
}

test "Hypergeometric: pmf k=1 in basic case" {
    const hg = try Hypergeometric(f64).init(10, 4, 3);
    // PMF(1) = C(4,1)*C(6,2)/C(10,3) = 4*15/120 = 0.5
    try expectApproxEqRel(0.5, hg.pmf(1), 1e-10);
}

test "Hypergeometric: pmf out of support returns 0" {
    const hg = try Hypergeometric(f64).init(10, 2, 5);
    // support_min = max(0, 5+2-10) = 0, support_max = min(5,2) = 2
    try expectApproxEqRel(0.0, hg.pmf(3), 1e-10);
    try expectApproxEqRel(0.0, hg.pmf(10), 1e-10);
}

test "Hypergeometric: pmf probabilities sum to 1" {
    const hg = try Hypergeometric(f64).init(10, 4, 3);
    var sum: f64 = 0.0;
    // support_min = max(0, 3+4-10) = 0, support_max = min(3,4) = 3
    for (0..4) |k| {
        sum += hg.pmf(k);
    }
    try expectApproxEqRel(1.0, sum, 1e-10);
}

test "Hypergeometric: cdf is monotone non-decreasing" {
    const hg = try Hypergeometric(f64).init(10, 4, 3);
    var prev: f64 = 0.0;
    for (0..5) |k| {
        const cdf_val = hg.cdf(@intCast(k));
        try testing.expect(prev <= cdf_val);
        prev = cdf_val;
    }
}

test "Hypergeometric: cdf k < support_min returns 0" {
    const hg = try Hypergeometric(f64).init(10, 2, 5);
    // support_min = max(0, 5+2-10) = 0
    try expectEqual(0.0, hg.cdf(-1));
}

test "Hypergeometric: cdf k >= support_max returns 1" {
    const hg = try Hypergeometric(f64).init(10, 4, 3);
    // support_max = min(3, 4) = 3
    try expectApproxEqRel(1.0, hg.cdf(@intCast(3)), 1e-10);
    try expectApproxEqRel(1.0, hg.cdf(@intCast(10)), 1e-10);
}

test "Hypergeometric: mean N=10 K=4 n=3 is 1.2" {
    const hg = try Hypergeometric(f64).init(10, 4, 3);
    // mean = n*K/N = 3*4/10 = 1.2
    try expectApproxEqRel(1.2, hg.mean(), 1e-10);
}

test "Hypergeometric: variance N=10 K=4 n=3" {
    const hg = try Hypergeometric(f64).init(10, 4, 3);
    // variance = n*(K/N)*((N-K)/N)*(N-n)/(N-1) = 3*0.4*0.6*7/9 ≈ 0.56
    try expectApproxEqRel(0.56, hg.variance(), 1e-10);
}

test "Hypergeometric: deterministic K=N all draws succeed" {
    const hg = try Hypergeometric(f64).init(5, 5, 3);
    // K=N means all items succeed: P(X=3) = 1.0
    try expectApproxEqRel(1.0, hg.pmf(3), 1e-10);
    try expectApproxEqRel(3.0, hg.mean(), 1e-10);
}

test "Hypergeometric: deterministic K=0 no successes" {
    const hg = try Hypergeometric(f64).init(5, 0, 3);
    // K=0 means no items succeed: P(X=0) = 1.0
    try expectApproxEqRel(1.0, hg.pmf(0), 1e-10);
    try expectApproxEqRel(0.0, hg.mean(), 1e-10);
}

test "Hypergeometric: deterministic n=0 empty draw" {
    const hg = try Hypergeometric(f64).init(5, 3, 0);
    // n=0 means no draws: P(X=0) = 1.0
    try expectApproxEqRel(1.0, hg.pmf(0), 1e-10);
    try expectApproxEqRel(0.0, hg.mean(), 1e-10);
}

test "Hypergeometric: logpmf matches log of pmf" {
    const hg = try Hypergeometric(f64).init(10, 4, 3);
    for (0..4) |k| {
        const pmf_val = hg.pmf(k);
        if (pmf_val > 0.0) {
            const expected = @log(pmf_val);
            const actual = hg.logpmf(k);
            try expectApproxEqRel(expected, actual, 1e-10);
        }
    }
}

test "Hypergeometric: sf equals 1 - cdf" {
    const hg = try Hypergeometric(f64).init(10, 4, 3);
    for (0..5) |k| {
        const cdf_val = hg.cdf(@intCast(k));
        const sf_val = hg.sf(@intCast(k));
        try expectApproxEqRel(1.0 - cdf_val, sf_val, 1e-10);
    }
}

test "Hypergeometric: quantile roundtrip at median" {
    const hg = try Hypergeometric(f64).init(10, 4, 3);
    const median = try hg.quantile(0.5);
    const cdf_median = hg.cdf(@intCast(median));
    try testing.expect(cdf_median >= 0.5);
}

test "Hypergeometric: quantile p=0 returns support_min" {
    const hg = try Hypergeometric(f64).init(10, 4, 3);
    const q = try hg.quantile(0.0);
    // support_min = max(0, 3+4-10) = 0
    try expectEqual(0, q);
}

test "Hypergeometric: quantile p=1.0 returns support_max" {
    const hg = try Hypergeometric(f64).init(10, 4, 3);
    const q = try hg.quantile(1.0);
    // support_max = min(3, 4) = 3
    try expectEqual(3, q);
}

test "Hypergeometric: quantile p < 0 returns error" {
    const hg = try Hypergeometric(f64).init(10, 4, 3);
    const result = hg.quantile(-0.01);
    try expectError(error.OutOfDomain, result);
}

test "Hypergeometric: quantile p > 1 returns error" {
    const hg = try Hypergeometric(f64).init(10, 4, 3);
    const result = hg.quantile(1.01);
    try expectError(error.OutOfDomain, result);
}

test "Hypergeometric: mode N=10 K=4 n=3" {
    const hg = try Hypergeometric(f64).init(10, 4, 3);
    // mode = floor((3+1)*(4+1)/(10+2)) = floor(20/12) = 1
    const mode = hg.mode();
    try expectEqual(1, mode);
    // mode should be in [support_min, support_max]
    try testing.expect(mode >= 0 and mode <= 3);
}

test "Hypergeometric: sample within support range" {
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    const hg = try Hypergeometric(f64).init(10, 4, 3);
    // support = [0, 3]
    for (0..20) |_| {
        const sample_val = hg.sample(rng);
        try testing.expect(sample_val >= 0 and sample_val <= 3);
    }
}

test "Hypergeometric: sample support with offset support_min > 0" {
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    const hg = try Hypergeometric(f64).init(6, 4, 4);
    // support_min = max(0, 4+4-6) = 2, support_max = min(4,4) = 4
    for (0..20) |_| {
        const sample_val = hg.sample(rng);
        try testing.expect(sample_val >= 2 and sample_val <= 4);
    }
}

test "Hypergeometric: validate passes on valid instance" {
    const hg = try Hypergeometric(f64).init(10, 4, 3);
    try hg.validate();
}

test "Hypergeometric: f32 type precision" {
    const hg = try Hypergeometric(f32).init(10, 4, 3);
    // pmf(2) ≈ 0.3
    try expectApproxEqRel(@as(f32, 0.3), hg.pmf(2), 0.001);
}

test "Hypergeometric: large population approximation" {
    const hg = try Hypergeometric(f64).init(1000, 200, 50);
    // mean = n*K/N = 50*200/1000 = 10.0
    try expectApproxEqRel(10.0, hg.mean(), 1e-10);
    // variance ≈ n*p*(1-p)*(N-n)/(N-1) where p=K/N ≈ 8.0
    const var_val = hg.variance();
    try testing.expect(var_val >= 7.5 and var_val <= 8.5);
}

test "Hypergeometric: N=1 deterministic cases" {
    const hg1 = try Hypergeometric(f64).init(1, 1, 1);
    // K=1, n=1: P(X=1) = 1.0
    try expectApproxEqRel(1.0, hg1.pmf(1), 1e-10);

    const hg0 = try Hypergeometric(f64).init(1, 0, 1);
    // K=0, n=1: P(X=0) = 1.0
    try expectApproxEqRel(1.0, hg0.pmf(0), 1e-10);
}

test "Hypergeometric: memory safety cycle loop" {
    for (0..10) |_| {
        const hg = try Hypergeometric(f64).init(10, 4, 3);
        for (0..4) |k| {
            _ = hg.pmf(k);
            _ = hg.logpmf(k);
            _ = hg.cdf(@intCast(k));
            _ = hg.sf(@intCast(k));
        }
        _ = hg.mean();
        _ = hg.variance();
        _ = hg.mode();
        try hg.validate();
    }
}

// ============================================================================
// Categorical Distribution
// ============================================================================
// Categorical distribution — discrete distribution over k categories
// Parameters: weights []const T, arbitrary non-negative values (normalized internally)
// Support: {0, 1, 2, ..., k-1}
// Generalization of Bernoulli to multiple categories.
// Probabilities: p[i] = weights[i] / sum(weights)

/// Categorical distribution — discrete distribution over k mutually exclusive outcomes
///
/// A generalization of the Bernoulli distribution to k > 2 categories.
/// Each category i has probability probs[i] with sum(probs) = 1.
/// Category labels are 0, 1, ..., k-1.
///
/// Parameters:
///   - k: number of categories (≥ 2)
///   - probs: probability vector (must sum to 1.0, all ≥ 0)
///
/// Applications:
///   - Multi-class classification (softmax outputs)
///   - Multinomial sampling in language models
///   - Bayesian categorical data modeling
pub fn Categorical(comptime T: type) type {
    return struct {
        const Self = @This();

        probs: []T,        // normalized probability vector (sum = 1.0)
        cum_probs: []T,    // cumulative probabilities for O(log k) sampling
        k: usize,          // number of categories
        allocator: std.mem.Allocator,

        /// Initialize Categorical distribution from weights.
        /// Weights are normalized to sum to 1. Must have k ≥ 2 non-negative weights
        /// with positive total sum.
        /// Time: O(k) | Space: O(k)
        pub fn init(allocator: std.mem.Allocator, weights: []const T) !Self {
            if (weights.len < 2) return DistributionError.InvalidParameter;

            var sum: T = 0.0;
            for (weights) |w| {
                if (w < 0.0) return DistributionError.InvalidParameter;
                sum += w;
            }
            if (sum <= 0.0) return DistributionError.InvalidParameter;

            const probs = try allocator.alloc(T, weights.len);
            errdefer allocator.free(probs);
            const cum_probs = try allocator.alloc(T, weights.len);
            errdefer allocator.free(cum_probs);

            var cumsum: T = 0.0;
            for (weights, 0..) |w, i| {
                probs[i] = w / sum;
                cumsum += probs[i];
                cum_probs[i] = cumsum;
            }
            cum_probs[weights.len - 1] = 1.0; // fix floating point drift

            return Self{
                .probs = probs,
                .cum_probs = cum_probs,
                .k = weights.len,
                .allocator = allocator,
            };
        }

        /// Free allocated memory
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: Self) void {
            self.allocator.free(self.probs);
            self.allocator.free(self.cum_probs);
        }

        /// Number of categories
        pub fn numCategories(self: Self) usize {
            return self.k;
        }

        /// PMF: probability of category i (0-indexed). Returns 0.0 for i ≥ k.
        /// Time: O(1) | Space: O(1)
        pub fn pmf(self: Self, i: usize) T {
            if (i >= self.k) return 0.0;
            return self.probs[i];
        }

        /// Log PMF: log probability of category i. Returns -inf for i ≥ k or p[i] = 0.
        /// Time: O(1) | Space: O(1)
        pub fn logpmf(self: Self, i: usize) T {
            if (i >= self.k) return -math.inf(T);
            const p = self.probs[i];
            if (p == 0.0) return -math.inf(T);
            return @log(p);
        }

        /// CDF: P(X ≤ i). Returns 0.0 for empty range, 1.0 for i ≥ k-1.
        /// Time: O(1) | Space: O(1)
        pub fn cdf(self: Self, i: usize) T {
            if (i >= self.k) return 1.0;
            return self.cum_probs[i];
        }

        /// Mean: E[X] = Σ(i × probs[i])
        /// Time: O(k) | Space: O(1)
        pub fn mean(self: Self) T {
            var m: T = 0.0;
            for (self.probs, 0..) |p, i| {
                m += @as(T, @floatFromInt(i)) * p;
            }
            return m;
        }

        /// Variance: Var[X] = E[X²] - E[X]²
        /// Time: O(k) | Space: O(1)
        pub fn variance(self: Self) T {
            const m = self.mean();
            var ex2: T = 0.0;
            for (self.probs, 0..) |p, i| {
                const fi: T = @floatFromInt(i);
                ex2 += fi * fi * p;
            }
            return ex2 - m * m;
        }

        /// Mode: index of highest probability (smallest index if tied)
        /// Time: O(k) | Space: O(1)
        pub fn mode(self: Self) usize {
            var best_i: usize = 0;
            var best_p: T = self.probs[0];
            for (self.probs[1..], 1..) |p, i| {
                if (p > best_p) {
                    best_p = p;
                    best_i = i;
                }
            }
            return best_i;
        }

        /// Shannon entropy: H = -Σ(p[i] × log(p[i]))
        /// Time: O(k) | Space: O(1)
        pub fn entropy(self: Self) T {
            var h: T = 0.0;
            for (self.probs) |p| {
                if (p > 0.0) {
                    h -= p * @log(p);
                }
            }
            return h;
        }

        /// Sample a category using inverse CDF with binary search.
        /// Time: O(log k) | Space: O(1)
        pub fn sample(self: Self, rng: std.Random) usize {
            const u = rng.float(T);
            var lo: usize = 0;
            var hi: usize = self.k;
            while (lo < hi) {
                const mid = lo + (hi - lo) / 2;
                if (self.cum_probs[mid] < u) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            return @min(lo, self.k - 1);
        }

        /// Validate invariants: k ≥ 2, probs sum to ≈1.0, all probs ≥ 0.
        /// Time: O(k) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.k < 2) return DistributionError.InvalidParameter;
            var sum: T = 0.0;
            for (self.probs) |p| {
                if (p < 0.0) return DistributionError.InvalidParameter;
                sum += p;
            }
            const eps: T = switch (T) {
                f32 => 1e-5,
                else => 1e-12,
            };
            if (@abs(sum - 1.0) > eps) return DistributionError.InvalidParameter;
        }
    };
}

test "Categorical: init with [0.5, 0.3, 0.2]" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.3, 0.2 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    try expectEqual(3, cat.numCategories());
}

test "Categorical: init unnormalized [2, 6, 2] normalizes correctly" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 2.0, 6.0, 2.0 }; // sum=10, normalize to [0.2, 0.6, 0.2]
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    try expectEqual(3, cat.numCategories());
    try expectApproxEqRel(0.2, cat.pmf(0), 1e-10);
    try expectApproxEqRel(0.6, cat.pmf(1), 1e-10);
    try expectApproxEqRel(0.2, cat.pmf(2), 1e-10);
}

test "Categorical: pmf [0.5, 0.3, 0.2]" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.3, 0.2 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    try expectApproxEqRel(0.5, cat.pmf(0), 1e-10);
    try expectApproxEqRel(0.3, cat.pmf(1), 1e-10);
    try expectApproxEqRel(0.2, cat.pmf(2), 1e-10);
}

test "Categorical: pmf out-of-bounds returns 0.0" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.3, 0.2 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    try expectEqual(0.0, cat.pmf(3));
    try expectEqual(0.0, cat.pmf(100));
}

test "Categorical: cdf [0.5, 0.3, 0.2] monotonic" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.3, 0.2 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    const cdf0 = cat.cdf(0);
    const cdf1 = cat.cdf(1);
    const cdf2 = cat.cdf(2);

    try expectApproxEqRel(0.5, cdf0, 1e-10);
    try expectApproxEqRel(0.8, cdf1, 1e-10);
    try expectApproxEqRel(1.0, cdf2, 1e-10);

    try testing.expect(cdf0 <= cdf1);
    try testing.expect(cdf1 <= cdf2);
}

test "Categorical: cdf out-of-bounds returns 1.0" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.3, 0.2 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    try expectEqual(1.0, cat.cdf(3));
    try expectEqual(1.0, cat.cdf(999));
}

test "Categorical: logpmf values [0.5, 0.3, 0.2]" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.3, 0.2 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    // log(0.5) ≈ -0.693147
    // log(0.3) ≈ -1.203973
    // log(0.2) ≈ -1.609438
    try expectApproxEqRel(@log(0.5), cat.logpmf(0), 1e-10);
    try expectApproxEqRel(@log(0.3), cat.logpmf(1), 1e-10);
    try expectApproxEqRel(@log(0.2), cat.logpmf(2), 1e-10);
}

test "Categorical: logpmf out-of-bounds returns -inf" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.3, 0.2 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    const logpmf3 = cat.logpmf(3);
    try testing.expect(math.isNegativeInf(logpmf3));
}

test "Categorical: mean [0.5, 0.3, 0.2] = 0.7" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.3, 0.2 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    // E[X] = 0*0.5 + 1*0.3 + 2*0.2 = 0.7
    try expectApproxEqRel(0.7, cat.mean(), 1e-10);
}

test "Categorical: variance [0.5, 0.3, 0.2] = 0.61" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.3, 0.2 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    // E[X²] = 0*0.5 + 1*0.3 + 4*0.2 = 1.1
    // Var[X] = 1.1 - 0.7² = 1.1 - 0.49 = 0.61
    try expectApproxEqRel(0.61, cat.variance(), 1e-10);
}

test "Categorical: mean binary uniform [1, 1] = 0.5" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 1.0, 1.0 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    try expectApproxEqRel(0.5, cat.mean(), 1e-10);
}

test "Categorical: variance binary uniform [1, 1] = 0.25" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 1.0, 1.0 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    // p = [0.5, 0.5], E[X] = 0.5, E[X²] = 0.5
    // Var[X] = 0.5 - 0.25 = 0.25
    try expectApproxEqRel(0.25, cat.variance(), 1e-10);
}

test "Categorical: mean 3-category uniform [1, 1, 1] = 1.0" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 1.0, 1.0, 1.0 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    // p = [1/3, 1/3, 1/3], E[X] = (0 + 1 + 2) / 3 = 1.0
    try expectApproxEqRel(1.0, cat.mean(), 1e-10);
}

test "Categorical: variance 3-category uniform [1, 1, 1]" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 1.0, 1.0, 1.0 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    // p = [1/3, 1/3, 1/3], E[X] = 1, E[X²] = (0 + 1 + 4)/3 = 5/3
    // Var[X] = 5/3 - 1 = 2/3 ≈ 0.6667
    try expectApproxEqRel(2.0 / 3.0, cat.variance(), 1e-10);
}

test "Categorical: entropy binary uniform [1, 1] = ln(2)" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 1.0, 1.0 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    // H = -sum(p * log(p)) = -(2 * 0.5 * log(0.5)) = ln(2) ≈ 0.693147
    const expected_entropy = @log(2.0);
    try expectApproxEqRel(expected_entropy, cat.entropy(), 1e-10);
}

test "Categorical: entropy 3-category uniform [1, 1, 1] = ln(3)" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 1.0, 1.0, 1.0 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    // H = -(3 * (1/3) * log(1/3)) = ln(3) ≈ 1.098612
    const expected_entropy = @log(3.0);
    try expectApproxEqRel(expected_entropy, cat.entropy(), 1e-10);
}

test "Categorical: mode [0.5, 0.3, 0.2] = 0" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.3, 0.2 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    try expectEqual(0, cat.mode());
}

test "Categorical: mode [0.2, 0.3, 0.5] = 2" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.2, 0.3, 0.5 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    try expectEqual(2, cat.mode());
}

test "Categorical: mode [0.1, 0.9] = 1" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.1, 0.9 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    try expectEqual(1, cat.mode());
}

test "Categorical: deinit frees memory (no leak)" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.3, 0.2 };
    const cat = try Categorical(f64).init(allocator, &weights);
    cat.deinit(); // Should not leak with testing.allocator
}

test "Categorical: sample returns value in [0, k)" {
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.3, 0.2 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    for (0..100) |_| {
        const sample_val = cat.sample(rng);
        try testing.expect(sample_val >= 0 and sample_val < 3);
    }
}

test "Categorical: sample deterministic [1.0, 0.0, 0.0] always returns 0" {
    var prng = std.Random.DefaultPrng.init(54321);
    const rng = prng.random();

    const allocator = testing.allocator;
    const weights = [_]f64{ 1.0, 0.0, 0.0 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    for (0..50) |_| {
        try expectEqual(0, cat.sample(rng));
    }
}

test "Categorical: sample deterministic [0.0, 0.0, 1.0] always returns 2" {
    var prng = std.Random.DefaultPrng.init(54321);
    const rng = prng.random();

    const allocator = testing.allocator;
    const weights = [_]f64{ 0.0, 0.0, 1.0 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    for (0..50) |_| {
        try expectEqual(2, cat.sample(rng));
    }
}

test "Categorical: validate passes on valid distribution" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.3, 0.2 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    try cat.validate();
}

test "Categorical: invalid empty weights" {
    const allocator = testing.allocator;
    const weights: [0]f64 = undefined;
    const result = Categorical(f64).init(allocator, &weights);
    try expectError(error.InvalidParameter, result);
}

test "Categorical: invalid single weight" {
    const allocator = testing.allocator;
    const weights = [_]f64{0.5};
    const result = Categorical(f64).init(allocator, &weights);
    try expectError(error.InvalidParameter, result);
}

test "Categorical: invalid negative weight" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, -0.3, 0.2 };
    const result = Categorical(f64).init(allocator, &weights);
    try expectError(error.InvalidParameter, result);
}

test "Categorical: invalid all-zero weights" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.0, 0.0, 0.0 };
    const result = Categorical(f64).init(allocator, &weights);
    try expectError(error.InvalidParameter, result);
}

test "Categorical: f32 type works" {
    const allocator = testing.allocator;
    const weights = [_]f32{ 0.5, 0.3, 0.2 };
    const cat = try Categorical(f32).init(allocator, &weights);
    defer cat.deinit();

    try expectApproxEqRel(@as(f32, 0.5), cat.pmf(0), 1e-5);
    try expectApproxEqRel(@as(f32, 0.3), cat.pmf(1), 1e-5);
    _ = cat.mean();
    _ = cat.variance();
    _ = cat.mode();
}

test "Categorical: large k (100 categories) uniform" {
    const allocator = testing.allocator;
    const weights_array = try allocator.alloc(f64, 100);
    defer allocator.free(weights_array);

    for (weights_array) |*w| {
        w.* = 1.0;
    }

    const cat = try Categorical(f64).init(allocator, weights_array);
    defer cat.deinit();

    try expectEqual(100, cat.numCategories());
    // Mean of uniform [0..99] is 49.5
    try expectApproxEqRel(49.5, cat.mean(), 1e-10);
}

test "Categorical: memory safety cycle loop" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.3, 0.2 };

    for (0..10) |_| {
        const cat = try Categorical(f64).init(allocator, &weights);
        _ = cat.pmf(0);
        _ = cat.pmf(1);
        _ = cat.pmf(2);
        _ = cat.cdf(0);
        _ = cat.cdf(1);
        _ = cat.cdf(2);
        _ = cat.logpmf(0);
        _ = cat.mean();
        _ = cat.variance();
        _ = cat.mode();
        _ = cat.entropy();
        try cat.validate();
        cat.deinit();
    }
}

test "Categorical: sample distribution empirical [0.7, 0.3]" {
    var prng = std.Random.DefaultPrng.init(99999);
    const rng = prng.random();

    const allocator = testing.allocator;
    const weights = [_]f64{ 0.7, 0.3 };
    const cat = try Categorical(f64).init(allocator, &weights);
    defer cat.deinit();

    var count0: usize = 0;
    var count1: usize = 0;
    const samples = 1000;

    for (0..samples) |_| {
        const s = cat.sample(rng);
        if (s == 0) count0 += 1 else count1 += 1;
    }

    // Empirical frequency for category 0 should be close to 0.7
    const freq0 = @as(f64, @floatFromInt(count0)) / @as(f64, @floatFromInt(samples));
    try expectApproxEqRel(0.7, freq0, 0.05); // Allow 5% tolerance for stochastic test
}

// ============================================================================
// Multinomial Distribution
// ============================================================================
// Multinomial distribution — multivariate generalization of Binomial
// Parameters: n (number of trials), probs []const T (k ≥ 2 categories)
// Support: {(x0, x1, ..., x_{k-1}) | xi ≥ 0, sum(xi) = n}
// Models outcome of n independent trials each with k possible outcomes.

/// Multinomial distribution — generalization of Binomial to k ≥ 2 categories
///
/// Models n independent trials where each trial results in one of k categories
/// with probabilities p1, p2, ..., pk (sum = 1).
/// The outcome is a count vector (x1, ..., xk) where xi = count of category i, sum(xi) = n.
///
/// Parameters:
///   - n: number of trials (n ≥ 1)
///   - probs: normalized probability vector (k ≥ 2, all probs ≥ 0, sum = 1)
///
/// PMF: n! / (x1! * x2! * ... * xk!) * p1^x1 * p2^x2 * ... * pk^xk
/// logPMF: lgamma(n+1) - sum(lgamma(xi+1)) + sum(xi * log(pi))
///         Convention: 0 * log(0) = 0 (term skipped when xi == 0)
///         Returns -inf when counts don't sum to n (outside support)
///
/// Marginal moments:
///   - Mean_i: n * pi
///   - Variance_i: n * pi * (1 - pi)
///   - Covariance(i,j): -n * pi * pj  [= Variance_i when i == j]
///
/// Time: O(k) for pmf/logpmf/validate; O(1) for mean/variance/covariance; O(k) for sample
pub fn Multinomial(comptime T: type) type {
    return struct {
        const Self = @This();

        n: u64,
        probs: []T,        // normalized probability vector (sum = 1.0)
        allocator: std.mem.Allocator,

        /// Initialize Multinomial distribution from n trials and weights.
        /// Weights are normalized to sum to 1. Must have k ≥ 2 non-negative weights
        /// with positive total sum and n ≥ 1.
        /// Time: O(k) | Space: O(k)
        pub fn init(allocator: std.mem.Allocator, n: u64, weights: []const T) !Self {
            if (n == 0) return DistributionError.InvalidParameter;
            if (weights.len < 2) return DistributionError.InvalidParameter;

            var sum: T = 0.0;
            for (weights) |w| {
                if (w < 0.0) return DistributionError.InvalidParameter;
                sum += w;
            }
            if (sum <= 0.0) return DistributionError.InvalidParameter;

            const probs = try allocator.alloc(T, weights.len);
            errdefer allocator.free(probs);

            for (weights, 0..) |w, i| {
                probs[i] = w / sum;
            }
            probs[weights.len - 1] = 1.0; // fix floating point drift
            // Recompute from scratch to ensure normalization
            var recomputed_sum: T = 0.0;
            for (probs[0 .. weights.len - 1]) |p| {
                recomputed_sum += p;
            }
            probs[weights.len - 1] = 1.0 - recomputed_sum;

            return Self{
                .n = n,
                .probs = probs,
                .allocator = allocator,
            };
        }

        /// Free allocated memory
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: Self) void {
            self.allocator.free(self.probs);
        }

        /// Number of categories
        /// Time: O(1) | Space: O(1)
        pub fn numCategories(self: Self) usize {
            return self.probs.len;
        }

        /// PMF: probability of outcome counts. Returns 0 if counts.len != k or sum(counts) != n.
        /// Time: O(k) | Space: O(1)
        pub fn pmf(self: Self, counts: []const u64) T {
            if (counts.len != self.probs.len) return 0.0;

            var sum: u64 = 0;
            for (counts) |c| {
                sum += c;
            }
            if (sum != self.n) return 0.0;

            const logpmf_val = self.logpmf(counts);
            if (math.isNegativeInf(logpmf_val)) return 0.0;
            return @exp(logpmf_val);
        }

        /// Log PMF: log probability of outcome counts.
        /// Returns -inf if counts don't sum to n or length mismatch.
        /// Time: O(k) | Space: O(1)
        pub fn logpmf(self: Self, counts: []const u64) T {
            if (counts.len != self.probs.len) return -math.inf(T);

            var sum: u64 = 0;
            for (counts) |c| {
                sum += c;
            }
            if (sum != self.n) return -math.inf(T);

            // lgamma(n+1) is log(n!)
            var log_coeff = math.lgamma(T, @as(T, @floatFromInt(self.n)) + 1.0);

            // Subtract sum(lgamma(x_i + 1)) for each count
            for (counts) |xi| {
                log_coeff -= math.lgamma(T, @as(T, @floatFromInt(xi)) + 1.0);
            }

            // Add sum(x_i * log(p_i)) for each count/prob pair (skip if x_i == 0)
            var log_probs: T = 0.0;
            for (counts, self.probs) |xi, pi| {
                if (xi > 0) {
                    log_probs += @as(T, @floatFromInt(xi)) * @log(pi);
                }
            }

            return log_coeff + log_probs;
        }

        /// Marginal mean of category i: E[X_i] = n * p_i
        /// Time: O(1) | Space: O(1)
        pub fn mean(self: Self, i: usize) T {
            return @as(T, @floatFromInt(self.n)) * self.probs[i];
        }

        /// Marginal variance of category i: Var[X_i] = n * p_i * (1 - p_i)
        /// Time: O(1) | Space: O(1)
        pub fn variance(self: Self, i: usize) T {
            return @as(T, @floatFromInt(self.n)) * self.probs[i] * (1.0 - self.probs[i]);
        }

        /// Covariance(i, j) = -n*pi*pj. Equals variance when i == j.
        /// Time: O(1) | Space: O(1)
        pub fn covariance(self: Self, i: usize, j: usize) T {
            if (i == j) return self.variance(i);
            return -@as(T, @floatFromInt(self.n)) * self.probs[i] * self.probs[j];
        }

        /// Sample using conditional Binomial method.
        /// Allocates []u64 counts (caller owns). Time: O(k*n) | Space: O(k)
        pub fn sample(self: Self, rng: std.Random, allocator: std.mem.Allocator) ![]u64 {
            const counts = try allocator.alloc(u64, self.probs.len);
            errdefer allocator.free(counts);

            var remaining: u64 = self.n;
            var mass_left: T = 1.0;

            // For each category 0..k-2, sample from Binomial(remaining, p_i / mass_left).
            // Clamp p_conditional to [0,1] to guard against floating-point drift in mass_left.
            for (0 .. self.probs.len - 1) |i| {
                const p_raw = self.probs[i] / mass_left;
                const p_conditional = @min(@max(p_raw, 0.0), 1.0);
                const xi = @min(binomialSample(rng, remaining, p_conditional), remaining);
                counts[i] = xi;
                remaining -= xi;
                mass_left -= self.probs[i];
            }
            // Last category gets remainder
            counts[self.probs.len - 1] = remaining;

            return counts;
        }

        /// Check invariants: n ≥ 1, probs.len ≥ 2, all probs ≥ 0, sum(probs) ≈ 1.0
        /// Time: O(k) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.n == 0) return DistributionError.InvalidParameter;
            if (self.probs.len < 2) return DistributionError.InvalidParameter;

            var sum: T = 0.0;
            for (self.probs) |p| {
                if (p < 0.0) return DistributionError.InvalidParameter;
                sum += p;
            }

            const eps: T = switch (T) {
                f32 => 1e-5,
                else => 1e-12,
            };
            if (@abs(sum - 1.0) > eps) return DistributionError.InvalidParameter;
        }
    };
}

/// Helper: sample from Binomial(n_trials, p) using Bernoulli sum for correctness
/// Time: O(n_trials)
fn binomialSample(rng: std.Random, n_trials: u64, p: anytype) u64 {
    const T = @TypeOf(p);
    var count: u64 = 0;
    for (0..n_trials) |_| {
        if (rng.float(T) < p) {
            count += 1;
        }
    }
    return count;
}

test "Multinomial: init with n=0 returns error" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.5 };
    const result = Multinomial(f64).init(allocator, 0, &weights);
    try expectError(error.InvalidParameter, result);
}

test "Multinomial: init with k=1 returns error" {
    const allocator = testing.allocator;
    const weights = [_]f64{0.5};
    const result = Multinomial(f64).init(allocator, 5, &weights);
    try expectError(error.InvalidParameter, result);
}

test "Multinomial: init with k=0 returns error" {
    const allocator = testing.allocator;
    const weights: [0]f64 = undefined;
    const result = Multinomial(f64).init(allocator, 5, &weights);
    try expectError(error.InvalidParameter, result);
}

test "Multinomial: init with negative weight returns error" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, -0.3, 0.2 };
    const result = Multinomial(f64).init(allocator, 5, &weights);
    try expectError(error.InvalidParameter, result);
}

test "Multinomial: init with zero sum weights returns error" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.0, 0.0, 0.0 };
    const result = Multinomial(f64).init(allocator, 5, &weights);
    try expectError(error.InvalidParameter, result);
}

test "Multinomial: init with valid parameters succeeds" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.3, 0.5, 0.2 };
    const dist = try Multinomial(f64).init(allocator, 10, &weights);
    defer dist.deinit();
}

test "Multinomial: numCategories returns k" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.3, 0.5, 0.2 };
    const dist = try Multinomial(f64).init(allocator, 10, &weights);
    defer dist.deinit();

    try expectEqual(3, dist.numCategories());
}

test "Multinomial: numCategories k=2" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.5 };
    const dist = try Multinomial(f64).init(allocator, 5, &weights);
    defer dist.deinit();

    try expectEqual(2, dist.numCategories());
}

test "Multinomial: pmf uniform n=2 k=3 [1,1,0]" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 1.0, 1.0, 1.0 };
    const dist = try Multinomial(f64).init(allocator, 2, &weights);
    defer dist.deinit();

    const counts = [_]u64{ 1, 1, 0 };
    const pmf_val = dist.pmf(&counts);
    // 2! / (1! * 1! * 0!) * (1/3)^1 * (1/3)^1 * (1/3)^0
    // = 2 * (1/9) = 2/9 ≈ 0.2222
    try expectApproxEqRel(2.0 / 9.0, pmf_val, 1e-10);
}

test "Multinomial: pmf n=1 k=3 reduces to categorical [1,0,0]" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.3, 0.2 };
    const dist = try Multinomial(f64).init(allocator, 1, &weights);
    defer dist.deinit();

    const counts1 = [_]u64{ 1, 0, 0 };
    const pmf1 = dist.pmf(&counts1);
    try expectApproxEqRel(0.5, pmf1, 1e-10);

    const counts2 = [_]u64{ 0, 1, 0 };
    const pmf2 = dist.pmf(&counts2);
    try expectApproxEqRel(0.3, pmf2, 1e-10);

    const counts3 = [_]u64{ 0, 0, 1 };
    const pmf3 = dist.pmf(&counts3);
    try expectApproxEqRel(0.2, pmf3, 1e-10);
}

test "Multinomial: pmf outside support wrong sum returns 0" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.3, 0.2 };
    const dist = try Multinomial(f64).init(allocator, 2, &weights);
    defer dist.deinit();

    const counts = [_]u64{ 1, 1, 1 }; // sum = 3, but n = 2
    const pmf_val = dist.pmf(&counts);
    try expectEqual(0.0, pmf_val);
}

test "Multinomial: pmf counts all zero when n nonzero returns 0" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.3, 0.2 };
    const dist = try Multinomial(f64).init(allocator, 3, &weights);
    defer dist.deinit();

    const counts = [_]u64{ 0, 0, 0 }; // sum = 0, but n = 3
    const pmf_val = dist.pmf(&counts);
    try expectEqual(0.0, pmf_val);
}

test "Multinomial: logpmf matches log(pmf) for valid counts" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.4, 0.3, 0.3 };
    const dist = try Multinomial(f64).init(allocator, 3, &weights);
    defer dist.deinit();

    const counts = [_]u64{ 1, 1, 1 };
    const pmf_val = dist.pmf(&counts);
    const logpmf_val = dist.logpmf(&counts);

    if (pmf_val > 0.0) {
        try expectApproxEqRel(@log(pmf_val), logpmf_val, 1e-10);
    }
}

test "Multinomial: logpmf outside support returns large negative" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.5 };
    const dist = try Multinomial(f64).init(allocator, 2, &weights);
    defer dist.deinit();

    const counts = [_]u64{ 0, 0 }; // sum = 0, but n = 2
    const logpmf_val = dist.logpmf(&counts);
    try testing.expect(math.isNegativeInf(logpmf_val));
}

test "Multinomial: pmf sums to 1 n=2 k=2" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.5 };
    const dist = try Multinomial(f64).init(allocator, 2, &weights);
    defer dist.deinit();

    var sum: f64 = 0.0;
    // enumerate all (x0, x1) with x0 + x1 = 2
    const outcomes = [_][2]u64{
        [_]u64{ 2, 0 },
        [_]u64{ 1, 1 },
        [_]u64{ 0, 2 },
    };

    for (outcomes) |counts| {
        sum += dist.pmf(&counts);
    }

    try expectApproxEqRel(1.0, sum, 1e-10);
}

test "Multinomial: pmf sums to 1 n=3 k=2" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.3, 0.7 };
    const dist = try Multinomial(f64).init(allocator, 3, &weights);
    defer dist.deinit();

    var sum: f64 = 0.0;
    // enumerate all (x0, x1) with x0 + x1 = 3
    for (0..4) |x0| {
        const x1 = 3 - x0;
        const counts = [_]u64{ x0, x1 };
        sum += dist.pmf(&counts);
    }

    try expectApproxEqRel(1.0, sum, 1e-10);
}

test "Multinomial: mean marginal equals n*pi" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.3, 0.5, 0.2 };
    const dist = try Multinomial(f64).init(allocator, 10, &weights);
    defer dist.deinit();

    try expectApproxEqRel(3.0, dist.mean(0), 1e-10); // 10 * 0.3
    try expectApproxEqRel(5.0, dist.mean(1), 1e-10); // 10 * 0.5
    try expectApproxEqRel(2.0, dist.mean(2), 1e-10); // 10 * 0.2
}

test "Multinomial: variance marginal equals np(1-p)" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.3, 0.5, 0.2 };
    const dist = try Multinomial(f64).init(allocator, 10, &weights);
    defer dist.deinit();

    try expectApproxEqRel(10.0 * 0.3 * 0.7, dist.variance(0), 1e-10);
    try expectApproxEqRel(10.0 * 0.5 * 0.5, dist.variance(1), 1e-10);
    try expectApproxEqRel(10.0 * 0.2 * 0.8, dist.variance(2), 1e-10);
}

test "Multinomial: covariance equals -n*pi*pj" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.3, 0.5, 0.2 };
    const dist = try Multinomial(f64).init(allocator, 10, &weights);
    defer dist.deinit();

    try expectApproxEqRel(-10.0 * 0.3 * 0.5, dist.covariance(0, 1), 1e-10);
    try expectApproxEqRel(-10.0 * 0.5 * 0.2, dist.covariance(1, 2), 1e-10);
    try expectApproxEqRel(-10.0 * 0.3 * 0.2, dist.covariance(0, 2), 1e-10);
}

test "Multinomial: covariance diagonal equals variance" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.3, 0.5, 0.2 };
    const dist = try Multinomial(f64).init(allocator, 10, &weights);
    defer dist.deinit();

    try expectApproxEqRel(dist.variance(0), dist.covariance(0, 0), 1e-10);
    try expectApproxEqRel(dist.variance(1), dist.covariance(1, 1), 1e-10);
    try expectApproxEqRel(dist.variance(2), dist.covariance(2, 2), 1e-10);
}

test "Multinomial: sample returns k-length counts" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.3, 0.5, 0.2 };
    const dist = try Multinomial(f64).init(allocator, 10, &weights);
    defer dist.deinit();

    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    const counts = try dist.sample(rng, allocator);
    defer allocator.free(counts);

    try expectEqual(3, counts.len);
}

test "Multinomial: sample counts sum to n" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.3, 0.5, 0.2 };
    const dist = try Multinomial(f64).init(allocator, 10, &weights);
    defer dist.deinit();

    var prng = std.Random.DefaultPrng.init(54321);
    const rng = prng.random();

    for (0..10) |_| {
        const counts = try dist.sample(rng, allocator);
        defer allocator.free(counts);

        var sum: u64 = 0;
        for (counts) |c| {
            sum += c;
        }
        try expectEqual(10, sum);
    }
}

test "Multinomial: sample all counts nonnegative" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.3, 0.5, 0.2 };
    const dist = try Multinomial(f64).init(allocator, 10, &weights);
    defer dist.deinit();

    var prng = std.Random.DefaultPrng.init(99999);
    const rng = prng.random();

    for (0..10) |_| {
        const counts = try dist.sample(rng, allocator);
        defer allocator.free(counts);

        var total_counts: u64 = 0;
        for (counts) |c| {
            total_counts += c;
        }
        try testing.expectEqual(@as(u64, 10), total_counts);
    }
}

test "Multinomial: sample deterministic p=[1,0] always returns [n,0]" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 1.0, 0.0 };
    const dist = try Multinomial(f64).init(allocator, 5, &weights);
    defer dist.deinit();

    var prng = std.Random.DefaultPrng.init(11111);
    const rng = prng.random();

    for (0..10) |_| {
        const counts = try dist.sample(rng, allocator);
        defer allocator.free(counts);

        try expectEqual(5, counts[0]);
        try expectEqual(0, counts[1]);
    }
}

test "Multinomial: sample deterministic p=[0,1] always returns [0,n]" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.0, 1.0 };
    const dist = try Multinomial(f64).init(allocator, 5, &weights);
    defer dist.deinit();

    var prng = std.Random.DefaultPrng.init(22222);
    const rng = prng.random();

    for (0..10) |_| {
        const counts = try dist.sample(rng, allocator);
        defer allocator.free(counts);

        try expectEqual(0, counts[0]);
        try expectEqual(5, counts[1]);
    }
}

test "Multinomial: empirical mean converges n=100 k=3 [0.2,0.3,0.5]" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.2, 0.3, 0.5 };
    const dist = try Multinomial(f64).init(allocator, 100, &weights);
    defer dist.deinit();

    var prng = std.Random.DefaultPrng.init(33333);
    const rng = prng.random();

    var sum0: f64 = 0.0;
    var sum1: f64 = 0.0;
    var sum2: f64 = 0.0;

    const samples = 500;
    for (0..samples) |_| {
        const counts = try dist.sample(rng, allocator);
        defer allocator.free(counts);

        sum0 += @as(f64, @floatFromInt(counts[0]));
        sum1 += @as(f64, @floatFromInt(counts[1]));
        sum2 += @as(f64, @floatFromInt(counts[2]));
    }

    const empirical_mean0 = sum0 / @as(f64, @floatFromInt(samples));
    const empirical_mean1 = sum1 / @as(f64, @floatFromInt(samples));
    const empirical_mean2 = sum2 / @as(f64, @floatFromInt(samples));

    // Expected: 20, 30, 50; allow 10% tolerance
    try expectApproxEqRel(20.0, empirical_mean0, 0.1);
    try expectApproxEqRel(30.0, empirical_mean1, 0.1);
    try expectApproxEqRel(50.0, empirical_mean2, 0.1);
}

test "Multinomial: empirical variance converges n=100 k=2 [0.4,0.6]" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.4, 0.6 };
    const dist = try Multinomial(f64).init(allocator, 100, &weights);
    defer dist.deinit();

    var prng = std.Random.DefaultPrng.init(44444);
    const rng = prng.random();

    var sum0: f64 = 0.0;
    var sumsq0: f64 = 0.0;

    const samples = 500;
    for (0..samples) |_| {
        const counts = try dist.sample(rng, allocator);
        defer allocator.free(counts);

        const val: f64 = @floatFromInt(counts[0]);
        sum0 += val;
        sumsq0 += val * val;
    }

    const empirical_mean0 = sum0 / @as(f64, @floatFromInt(samples));
    const empirical_var0 = (sumsq0 / @as(f64, @floatFromInt(samples))) - empirical_mean0 * empirical_mean0;

    // Expected variance: 100 * 0.4 * 0.6 = 24
    // Allow 15% tolerance
    try expectApproxEqRel(24.0, empirical_var0, 0.15);
}

test "Multinomial: validate passes on valid distribution" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.3, 0.5, 0.2 };
    const dist = try Multinomial(f64).init(allocator, 10, &weights);
    defer dist.deinit();

    try dist.validate();
}

test "Multinomial: f32 type support" {
    const allocator = testing.allocator;
    const weights = [_]f32{ 0.3, 0.5, 0.2 };
    const dist = try Multinomial(f32).init(allocator, 10, &weights);
    defer dist.deinit();

    try expectEqual(3, dist.numCategories());
    try expectApproxEqRel(@as(f32, 3.0), dist.mean(0), 1e-5);
    try expectApproxEqRel(@as(f32, 2.1), dist.variance(0), 1e-5);
}

test "Multinomial: memory safety cycle loop" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.5 };

    for (0..10) |_| {
        const dist = try Multinomial(f64).init(allocator, 10, &weights);
        defer dist.deinit();

        const counts = [_]u64{ 5, 5 };
        _ = dist.pmf(&counts);
        _ = dist.logpmf(&counts);
        _ = dist.mean(0);
        _ = dist.variance(0);
        _ = dist.covariance(0, 1);

        var prng = std.Random.DefaultPrng.init(55555);
        const rng = prng.random();
        const sample_counts = try dist.sample(rng, allocator);
        allocator.free(sample_counts);

        try dist.validate();
    }
}

test "Multinomial: sample ownership caller allocates and frees" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.3, 0.5, 0.2 };
    const dist = try Multinomial(f64).init(allocator, 10, &weights);
    defer dist.deinit();

    var prng = std.Random.DefaultPrng.init(66666);
    const rng = prng.random();

    const counts = try dist.sample(rng, allocator);
    try expectEqual(3, counts.len);
    allocator.free(counts); // Caller must free
}

test "Multinomial: large n and k boundary" {
    const allocator = testing.allocator;
    const weights_array = try allocator.alloc(f64, 10);
    defer allocator.free(weights_array);

    for (weights_array) |*w| {
        w.* = 1.0;
    }

    const dist = try Multinomial(f64).init(allocator, 1000, weights_array);
    defer dist.deinit();

    var prng = std.Random.DefaultPrng.init(77777);
    const rng = prng.random();

    const counts = try dist.sample(rng, allocator);
    defer allocator.free(counts);

    var sum: u64 = 0;
    for (counts) |c| {
        sum += c;
    }
    try expectEqual(1000, sum);
}

test "Multinomial: pmf with zero-prob zero-count handled" {
    const allocator = testing.allocator;
    const weights = [_]f64{ 0.5, 0.0, 0.5 };
    const dist = try Multinomial(f64).init(allocator, 2, &weights);
    defer dist.deinit();

    // Counts [1, 0, 1]: category 1 has 0 prob but 0 count => 0*log(0) = 0
    const counts = [_]u64{ 1, 0, 1 };
    const pmf_val = dist.pmf(&counts);
    try testing.expect(!math.isNan(pmf_val));
}

test "Multinomial: binomial equivalence k=2" {
    const allocator = testing.allocator;
    const p: f64 = 0.6;
    const weights = [_]f64{ p, 1.0 - p };
    const dist = try Multinomial(f64).init(allocator, 10, &weights);
    defer dist.deinit();

    // Multinomial(n=10, [0.6, 0.4]) marginal mean for category 0 should be 10*0.6=6
    // Compare with Binomial(n=10, p=0.6) mean = 10*0.6=6
    try expectApproxEqRel(6.0, dist.mean(0), 1e-10);
    try expectApproxEqRel(10.0 * 0.6 * 0.4, dist.variance(0), 1e-10);
}

// ============================================================================
// DIRICHLET DISTRIBUTION
// ============================================================================

/// Digamma function ψ(x) = d/dx ln Γ(x).
/// Uses recurrence ψ(x+1) = ψ(x) + 1/x to shift x ≥ 6, then asymptotic expansion.
/// Accurate to ~15 significant digits for x > 0.
/// Time: O(1) | Space: O(1)
fn digamma(comptime T: type, x: T) T {
    var sum: T = 0.0;
    var xi = x;
    while (xi < 6.0) {
        sum -= 1.0 / xi;
        xi += 1.0;
    }
    // Asymptotic: ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - 1/(252x⁶)
    const xi2 = xi * xi;
    const xi4 = xi2 * xi2;
    const xi6 = xi4 * xi2;
    return @log(xi) - 0.5 / xi - 1.0 / (12.0 * xi2) + 1.0 / (120.0 * xi4) - 1.0 / (252.0 * xi6) + sum;
}

/// Dirichlet distribution — multivariate continuous distribution over the probability simplex.
/// Conjugate prior for Categorical and Multinomial distributions.
///
/// Parameters: α = (α₁,...,αₖ), αᵢ > 0, k ≥ 2
/// Support: {x ∈ ℝᵏ : xᵢ ≥ 0, Σxᵢ = 1}
/// PDF: f(x|α) = Γ(α₀)/∏Γ(αᵢ) × ∏xᵢ^(αᵢ-1), α₀ = Σαᵢ
pub fn Dirichlet(comptime T: type) type {
    return struct {
        const Self = @This();

        alphas: []T,
        alpha0: T,
        allocator: std.mem.Allocator,

        /// Initialize Dirichlet distribution. All alphas must be > 0, k ≥ 2.
        /// Time: O(k) | Space: O(k)
        pub fn init(allocator: std.mem.Allocator, alphas: []const T) !Self {
            if (alphas.len < 2) return DistributionError.InvalidParameter;

            var alpha0: T = 0.0;
            for (alphas) |a| {
                if (a <= 0.0) return DistributionError.InvalidParameter;
                alpha0 += a;
            }

            const stored = try allocator.alloc(T, alphas.len);
            errdefer allocator.free(stored);
            @memcpy(stored, alphas);

            return Self{ .alphas = stored, .alpha0 = alpha0, .allocator = allocator };
        }

        /// Free allocated memory.
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: Self) void {
            self.allocator.free(self.alphas);
        }

        /// Number of categories k.
        /// Time: O(1) | Space: O(1)
        pub fn numCategories(self: Self) usize {
            return self.alphas.len;
        }

        /// Log-PDF at x. Returns -inf if x.len ≠ k, any xᵢ < 0, or Σxᵢ ≠ 1.
        /// Time: O(k) | Space: O(1)
        pub fn logpdf(self: Self, x: []const T) T {
            if (x.len != self.alphas.len) return -math.inf(T);

            var sum: T = 0.0;
            for (x) |xi| {
                if (xi < 0.0) return -math.inf(T);
                sum += xi;
            }
            const eps: T = switch (T) {
                f32 => 1e-5,
                else => 1e-10,
            };
            if (@abs(sum - 1.0) > eps) return -math.inf(T);

            // log f = lgamma(α₀) - Σlgamma(αᵢ) + Σ(αᵢ-1)log(xᵢ)
            var log_norm: T = logGamma(self.alpha0);
            for (self.alphas) |ai| {
                log_norm -= logGamma(ai);
            }

            var log_kernel: T = 0.0;
            for (x, self.alphas) |xi, ai| {
                const exp = ai - 1.0;
                if (@abs(exp) < 1e-15) continue;
                if (xi <= 0.0) return -math.inf(T);
                log_kernel += exp * @log(xi);
            }

            return log_norm + log_kernel;
        }

        /// PDF at x. Returns 0 if x is outside the simplex.
        /// Time: O(k) | Space: O(1)
        pub fn pdf(self: Self, x: []const T) T {
            const lp = self.logpdf(x);
            if (math.isNegativeInf(lp)) return 0.0;
            return @exp(lp);
        }

        /// Marginal mean of category i: E[Xᵢ] = αᵢ / α₀
        /// Time: O(1) | Space: O(1)
        pub fn mean(self: Self, i: usize) T {
            return self.alphas[i] / self.alpha0;
        }

        /// Marginal variance of category i: Var[Xᵢ] = αᵢ(α₀-αᵢ) / (α₀²(α₀+1))
        /// Time: O(1) | Space: O(1)
        pub fn variance(self: Self, i: usize) T {
            const ai = self.alphas[i];
            return ai * (self.alpha0 - ai) / (self.alpha0 * self.alpha0 * (self.alpha0 + 1.0));
        }

        /// Covariance(i,j): -αᵢαⱼ/(α₀²(α₀+1)) for i≠j; variance(i) for i==j.
        /// Time: O(1) | Space: O(1)
        pub fn covariance(self: Self, i: usize, j: usize) T {
            if (i == j) return self.variance(i);
            return -(self.alphas[i] * self.alphas[j]) / (self.alpha0 * self.alpha0 * (self.alpha0 + 1.0));
        }

        /// Mode: (αᵢ-1)/(α₀-k) for all αᵢ > 1. Returns error.InvalidParameter if any αᵢ ≤ 1.
        /// Allocates []T (caller owns). Time: O(k) | Space: O(k)
        pub fn mode(self: Self, allocator: std.mem.Allocator) ![]T {
            for (self.alphas) |ai| {
                if (ai <= 1.0) return DistributionError.InvalidParameter;
            }
            const k: T = @floatFromInt(self.alphas.len);
            const denom = self.alpha0 - k;
            const m = try allocator.alloc(T, self.alphas.len);
            for (self.alphas, 0..) |ai, i| {
                m[i] = (ai - 1.0) / denom;
            }
            return m;
        }

        /// Shannon entropy: log B(α) + (α₀-k)ψ(α₀) - Σ(αᵢ-1)ψ(αᵢ)
        /// where log B(α) = Σlgamma(αᵢ) - lgamma(α₀), ψ is the digamma function.
        /// Time: O(k) | Space: O(1)
        pub fn entropy(self: Self) T {
            var log_beta: T = -logGamma(self.alpha0);
            for (self.alphas) |ai| {
                log_beta += logGamma(ai);
            }
            const k: T = @floatFromInt(self.alphas.len);
            var weighted_digamma_sum: T = 0.0;
            for (self.alphas) |ai| {
                weighted_digamma_sum += (ai - 1.0) * digamma(T, ai);
            }
            return log_beta + (self.alpha0 - k) * digamma(T, self.alpha0) - weighted_digamma_sum;
        }

        /// Sample from Dirichlet: draw Gamma(αᵢ, 1) for each i, then normalize.
        /// Allocates []T (caller owns). Time: O(k) | Space: O(k)
        pub fn sample(self: Self, rng: std.Random, allocator: std.mem.Allocator) ![]T {
            const xs = try allocator.alloc(T, self.alphas.len);
            errdefer allocator.free(xs);

            var total: T = 0.0;
            for (self.alphas, 0..) |ai, i| {
                // ai > 0 and rate = 1.0 are guaranteed by init — Gamma.init cannot fail here.
                const g = Gamma(T).init(ai, 1.0) catch unreachable;
                xs[i] = g.sample(rng);
                total += xs[i];
            }

            for (xs) |*xi| {
                xi.* /= total;
            }
            // Fix last element to ensure Σxᵢ = 1 exactly
            var partial: T = 0.0;
            for (xs[0 .. xs.len - 1]) |xi| {
                partial += xi;
            }
            xs[xs.len - 1] = 1.0 - partial;

            return xs;
        }

        /// Validate invariants: k ≥ 2, all αᵢ > 0, α₀ ≈ Σαᵢ.
        /// Time: O(k) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.alphas.len < 2) return DistributionError.InvalidParameter;
            var sum: T = 0.0;
            for (self.alphas) |ai| {
                if (ai <= 0.0) return DistributionError.InvalidParameter;
                sum += ai;
            }
            const eps: T = switch (T) {
                f32 => 1e-5,
                else => 1e-10,
            };
            if (@abs(sum - self.alpha0) > eps) return DistributionError.InvalidParameter;
        }
    };
}

// ============================================================================
// DIRICHLET DISTRIBUTION TESTS
// ============================================================================

test "Dirichlet: init with k=1 returns error" {
    const allocator = testing.allocator;
    const alphas = [_]f64{1.0};
    const result = Dirichlet(f64).init(allocator, &alphas);
    try expectError(error.InvalidParameter, result);
}

test "Dirichlet: init with k=0 returns error" {
    const allocator = testing.allocator;
    const alphas: [0]f64 = undefined;
    const result = Dirichlet(f64).init(allocator, &alphas);
    try expectError(error.InvalidParameter, result);
}

test "Dirichlet: init with zero alpha returns error" {
    const allocator = testing.allocator;
    const alphas = [_]f64{ 1.0, 0.0 };
    const result = Dirichlet(f64).init(allocator, &alphas);
    try expectError(error.InvalidParameter, result);
}

test "Dirichlet: init with negative alpha returns error" {
    const allocator = testing.allocator;
    const alphas = [_]f64{ 1.0, -0.5 };
    const result = Dirichlet(f64).init(allocator, &alphas);
    try expectError(error.InvalidParameter, result);
}

test "Dirichlet: symmetric uniform mean" {
    const allocator = testing.allocator;
    const alphas = [_]f64{ 1.0, 1.0, 1.0 };
    const dist = try Dirichlet(f64).init(allocator, &alphas);
    defer dist.deinit();

    // Dirichlet(1,1,1) is uniform over 2-simplex
    // Mean should be [1/3, 1/3, 1/3]
    try expectApproxEqAbs(1.0 / 3.0, dist.mean(0), 1e-10);
    try expectApproxEqAbs(1.0 / 3.0, dist.mean(1), 1e-10);
    try expectApproxEqAbs(1.0 / 3.0, dist.mean(2), 1e-10);
}

test "Dirichlet: mean formula" {
    const allocator = testing.allocator;
    const alphas = [_]f64{ 2.0, 3.0, 5.0 };
    const dist = try Dirichlet(f64).init(allocator, &alphas);
    defer dist.deinit();

    // alpha0 = 10
    // mean(i) = alpha(i) / alpha0
    try expectApproxEqAbs(0.2, dist.mean(0), 1e-10);
    try expectApproxEqAbs(0.3, dist.mean(1), 1e-10);
    try expectApproxEqAbs(0.5, dist.mean(2), 1e-10);
}

test "Dirichlet: variance formula" {
    const allocator = testing.allocator;
    const alphas = [_]f64{ 2.0, 3.0, 5.0 };
    const dist = try Dirichlet(f64).init(allocator, &alphas);
    defer dist.deinit();

    // alpha0 = 10
    // variance(i) = alpha(i) * (alpha0 - alpha(i)) / (alpha0^2 * (alpha0 + 1))
    // variance(0) = 2 * 8 / (100 * 11) = 16/1100 ≈ 0.01454545...
    try expectApproxEqAbs(16.0 / 1100.0, dist.variance(0), 1e-10);
    // variance(1) = 3 * 7 / (100 * 11) = 21/1100 ≈ 0.01909090...
    try expectApproxEqAbs(21.0 / 1100.0, dist.variance(1), 1e-10);
    // variance(2) = 5 * 5 / (100 * 11) = 25/1100 ≈ 0.02272727...
    try expectApproxEqAbs(25.0 / 1100.0, dist.variance(2), 1e-10);
}

test "Dirichlet: covariance formula" {
    const allocator = testing.allocator;
    const alphas = [_]f64{ 2.0, 3.0, 5.0 };
    const dist = try Dirichlet(f64).init(allocator, &alphas);
    defer dist.deinit();

    // alpha0 = 10
    // covariance(i,j) = -alpha(i) * alpha(j) / (alpha0^2 * (alpha0 + 1))
    // covariance(0,1) = -2 * 3 / (100 * 11) = -6/1100 ≈ -0.00545454...
    try expectApproxEqAbs(-6.0 / 1100.0, dist.covariance(0, 1), 1e-10);
    // covariance(0,2) = -2 * 5 / (100 * 11) = -10/1100 ≈ -0.00909090...
    try expectApproxEqAbs(-10.0 / 1100.0, dist.covariance(0, 2), 1e-10);
}

test "Dirichlet: covariance diagonal equals variance" {
    const allocator = testing.allocator;
    const alphas = [_]f64{ 2.0, 3.0, 5.0 };
    const dist = try Dirichlet(f64).init(allocator, &alphas);
    defer dist.deinit();

    // covariance(i,i) should equal variance(i)
    try expectEqual(dist.variance(0), dist.covariance(0, 0));
    try expectEqual(dist.variance(1), dist.covariance(1, 1));
    try expectEqual(dist.variance(2), dist.covariance(2, 2));
}

test "Dirichlet: logpdf at centroid Dir(2,2,2)" {
    const allocator = testing.allocator;
    const alphas = [_]f64{ 2.0, 2.0, 2.0 };
    const dist = try Dirichlet(f64).init(allocator, &alphas);
    defer dist.deinit();

    const x = [_]f64{ 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 };
    const logpdf_val = dist.logpdf(&x);

    // logpdf = lgamma(6) - 3*lgamma(2) + 3*(2-1)*log(1/3)
    // = lgamma(6) - 3*0 + 3*log(1/3)
    // = log(120) + 3*(-log(3))
    const expected = @log(120.0) - 3.0 * @log(3.0);
    try expectApproxEqAbs(expected, logpdf_val, 1e-6);
}

test "Dirichlet: logpdf outside simplex returns -inf" {
    const allocator = testing.allocator;
    const alphas = [_]f64{ 1.0, 2.0 };
    const dist = try Dirichlet(f64).init(allocator, &alphas);
    defer dist.deinit();

    // x sums to 1.1 > 1, outside simplex
    const x = [_]f64{ 0.5, 0.6 };
    const logpdf_val = dist.logpdf(&x);
    try expect(math.isInf(logpdf_val) and logpdf_val < 0);
}

test "Dirichlet: logpdf wrong length returns -inf" {
    const allocator = testing.allocator;
    const alphas = [_]f64{ 1.0, 2.0 };
    const dist = try Dirichlet(f64).init(allocator, &alphas);
    defer dist.deinit();

    // x has length 3 but distribution has k=2
    const x = [_]f64{ 0.3, 0.3, 0.4 };
    const logpdf_val = dist.logpdf(&x);
    try expect(math.isInf(logpdf_val) and logpdf_val < 0);
}

test "Dirichlet: pdf at valid point" {
    const allocator = testing.allocator;
    const alphas = [_]f64{ 2.0, 2.0 };
    const dist = try Dirichlet(f64).init(allocator, &alphas);
    defer dist.deinit();

    const x = [_]f64{ 0.4, 0.6 };
    const pdf_val = dist.pdf(&x);

    // Dir(2,2) is Beta(2,2) on [0,1]: pdf(x) = 6*x*(1-x) = 6*0.4*0.6 = 1.44
    try expectApproxEqRel(@as(f64, 1.44), pdf_val, 1e-10);
}

test "Dirichlet: mode for concentrated params" {
    const allocator = testing.allocator;
    const alphas = [_]f64{ 3.0, 4.0, 5.0 };
    const dist = try Dirichlet(f64).init(allocator, &alphas);
    defer dist.deinit();

    // alpha0 = 12, k = 3
    // mode(i) = (alpha(i) - 1) / (alpha0 - k)
    // = (alpha(i) - 1) / 9
    const mode_slice = try dist.mode(allocator);
    defer allocator.free(mode_slice);

    // mode(0) = 2/9, mode(1) = 3/9, mode(2) = 4/9
    try expectApproxEqAbs(2.0 / 9.0, mode_slice[0], 1e-10);
    try expectApproxEqAbs(3.0 / 9.0, mode_slice[1], 1e-10);
    try expectApproxEqAbs(4.0 / 9.0, mode_slice[2], 1e-10);
}

test "Dirichlet: mode with alpha <= 1 returns error" {
    const allocator = testing.allocator;
    const alphas = [_]f64{ 1.0, 2.0 };
    const dist = try Dirichlet(f64).init(allocator, &alphas);
    defer dist.deinit();

    // alpha(0) = 1.0, so mode() should fail
    const result = dist.mode(allocator);
    try expectError(error.InvalidParameter, result);
}

test "Dirichlet: entropy symmetric Dir(1,1,1)" {
    const allocator = testing.allocator;
    const alphas = [_]f64{ 1.0, 1.0, 1.0 };
    const dist = try Dirichlet(f64).init(allocator, &alphas);
    defer dist.deinit();

    const entropy_val = dist.entropy();

    // For Dir(1,1,1): log B(1,1,1) + (alpha0 - k)*psi(alpha0) - sum((alpha(i)-1)*psi(alpha(i)))
    // B(1,1,1) = Gamma(1)*Gamma(1)*Gamma(1) / Gamma(3) = 1*1*1 / 2 = 1/2
    // log B = -log(2) ≈ -0.693147...
    // alpha0 = 3, k = 3, so (alpha0 - k) = 0
    // sum((alpha(i)-1)*psi(alpha(i))) = sum(0*psi(1)) = 0
    // entropy = log(1/2) = -log(2) ≈ -0.693147...
    try expectApproxEqAbs(-@log(2.0), entropy_val, 1e-6);
}

test "Dirichlet: sample sums to 1" {
    const allocator = testing.allocator;
    const alphas = [_]f64{ 2.0, 3.0, 5.0 };
    const dist = try Dirichlet(f64).init(allocator, &alphas);
    defer dist.deinit();

    var prng = std.Random.DefaultPrng.init(99999);
    const rng = prng.random();

    // Draw 100 samples and verify each sums to ~1.0
    for (0..100) |_| {
        const sample_x = try dist.sample(rng, allocator);
        defer allocator.free(sample_x);

        var sum: f64 = 0.0;
        for (sample_x) |val| {
            sum += val;
        }
        try expectApproxEqAbs(1.0, sum, 1e-10);
    }
}

test "Dirichlet: sample non-negative" {
    const allocator = testing.allocator;
    const alphas = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const dist = try Dirichlet(f64).init(allocator, &alphas);
    defer dist.deinit();

    var prng = std.Random.DefaultPrng.init(88888);
    const rng = prng.random();

    for (0..50) |_| {
        const sample_x = try dist.sample(rng, allocator);
        defer allocator.free(sample_x);

        for (sample_x) |val| {
            try expect(val >= 0.0);
        }
    }
}

test "Dirichlet: sample empirical mean" {
    const allocator = testing.allocator;
    const alphas = [_]f64{ 2.0, 3.0, 5.0 };
    const dist = try Dirichlet(f64).init(allocator, &alphas);
    defer dist.deinit();

    var prng = std.Random.DefaultPrng.init(77777);
    const rng = prng.random();

    var sum0: f64 = 0.0;
    var sum1: f64 = 0.0;
    var sum2: f64 = 0.0;

    const samples = 5000;
    for (0..samples) |_| {
        const sample_x = try dist.sample(rng, allocator);
        defer allocator.free(sample_x);

        sum0 += sample_x[0];
        sum1 += sample_x[1];
        sum2 += sample_x[2];
    }

    const empirical_mean0 = sum0 / @as(f64, @floatFromInt(samples));
    const empirical_mean1 = sum1 / @as(f64, @floatFromInt(samples));
    const empirical_mean2 = sum2 / @as(f64, @floatFromInt(samples));

    // Expected means: 2/10, 3/10, 5/10
    // Allow 2% tolerance
    try expectApproxEqRel(0.2, empirical_mean0, 0.02);
    try expectApproxEqRel(0.3, empirical_mean1, 0.02);
    try expectApproxEqRel(0.5, empirical_mean2, 0.02);
}

test "Dirichlet: f32 support" {
    const allocator = testing.allocator;
    const alphas = [_]f32{ 1.0, 2.0, 3.0 };
    const dist = try Dirichlet(f32).init(allocator, &alphas);
    defer dist.deinit();

    try expectEqual(3, dist.numCategories());
    try expectApproxEqRel(@as(f32, 1.0 / 6.0), dist.mean(0), 1e-5);
    try expectApproxEqRel(@as(f32, 2.0 / 6.0), dist.mean(1), 1e-5);
    try expectApproxEqRel(@as(f32, 3.0 / 6.0), dist.mean(2), 1e-5);

    // Spot-check logpdf
    const x = [_]f32{ 1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0 };
    const logpdf_val = dist.logpdf(&x);
    try expect(!math.isNan(logpdf_val));
}

test "Dirichlet: memory safety init deinit" {
    const allocator = testing.allocator;

    for (0..1000) |_| {
        const alphas = [_]f64{ 1.0, 2.0, 3.0 };
        const dist = try Dirichlet(f64).init(allocator, &alphas);
        dist.deinit();
    }
}

test "Dirichlet: sample memory safety" {
    const allocator = testing.allocator;
    const alphas = [_]f64{ 2.0, 3.0 };
    const dist = try Dirichlet(f64).init(allocator, &alphas);
    defer dist.deinit();

    var prng = std.Random.DefaultPrng.init(55555);
    const rng = prng.random();

    for (0..100) |_| {
        const sample_x = try dist.sample(rng, allocator);
        allocator.free(sample_x);
    }
}

test "Dirichlet: validate passes" {
    const allocator = testing.allocator;
    const alphas = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const dist = try Dirichlet(f64).init(allocator, &alphas);
    defer dist.deinit();

    try dist.validate();
}

test "Dirichlet: numCategories" {
    const allocator = testing.allocator;
    const alphas = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const dist = try Dirichlet(f64).init(allocator, &alphas);
    defer dist.deinit();

    try expectEqual(4, dist.numCategories());
}
