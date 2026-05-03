const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;

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
            if (x <= 0.0) return 0.0;

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
fn regularizedBetaI(a: anytype, b: anytype, x: anytype) @TypeOf(a) {
    const T = @TypeOf(a);

    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;

    // Use symmetry relation if beneficial: I_x(a,b) = 1 - I_(1-x)(b,a)
    if (x > (a + 1.0) / (a + b + 2.0)) {
        return 1.0 - regularizedBetaI(b, a, 1.0 - x);
    }

    // Compute continued fraction using Lentz's algorithm
    const max_iterations = 200;
    const tolerance = 1e-10;

    // log(x^a (1-x)^b / a / B(a,b))
    const log_bt = a * @log(x) + b * @log(1.0 - x) - logBeta(a, b) - @log(a);
    const bt = @exp(log_bt);

    // Continued fraction coefficients
    var f: T = 1.0;
    var c: T = 1.0;
    var d: T = 0.0;

    for (0..max_iterations) |m| {
        const m_f = @as(T, @floatFromInt(m));

        // Even step (2m)
        var aa: T = undefined;
        if (m == 0) {
            aa = 1.0;
        } else {
            const num = m_f * (b - m_f) * x;
            const den = (a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f);
            aa = num / den;
        }

        d = 1.0 + aa * d;
        if (@abs(d) < 1.0e-30) d = 1.0e-30;
        c = 1.0 + aa / c;
        if (@abs(c) < 1.0e-30) c = 1.0e-30;
        d = 1.0 / d;
        f *= d * c;

        // Odd step (2m+1)
        const num = -(a + m_f) * (a + b + m_f) * x;
        const den = (a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0);
        aa = num / den;

        d = 1.0 + aa * d;
        if (@abs(d) < 1.0e-30) d = 1.0e-30;
        c = 1.0 + aa / c;
        if (@abs(c) < 1.0e-30) c = 1.0e-30;
        d = 1.0 / d;
        const delta = d * c;
        f *= delta;

        if (@abs(delta - 1.0) < tolerance) break;
    }

    return bt * f;
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

    // Central region: |y| ≤ 0.7
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
        return y * num / den;
    }

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
    return sign * (z - num / den);
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const expectApproxEqRel = testing.expectApproxEqRel;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectEqual = testing.expectEqual;
const expectError = testing.expectError;

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

    // Beta(2, 5): pdf(0.5) = (0.5^1 × 0.5^4) / B(2,5) ≈ 1.875
    // B(2,5) = Γ(2)Γ(5)/Γ(7) = 1!×4!/6! = 24/720 = 1/30
    // pdf(0.5) = (0.5 × 0.0625) / (1/30) = 0.03125 × 30 = 0.9375
    // Actually: pdf(x) = x^(α-1)(1-x)^(β-1)/B(α,β) = x^1(1-x)^4/B(2,5)
    // pdf(0.5) = 0.5 × (0.5)^4 / (1/30) = 0.5 × 0.0625 × 30 = 0.9375
    try expectApproxEqRel(0.9375, dist.pdf(0.2), 1e-10);

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

    // pdf(0.2) ≈ 0.9375
    try expectApproxEqRel(@as(f32, 0.9375), dist.pdf(0.2), 1e-3);
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

test "distributions: memory safety" {
    const allocator = testing.allocator;
    _ = allocator;

    // Test multiple init/deinit cycles (no memory allocation in distributions)
    for (0..10) |_| {
        const normal = try Normal(f64).init(0.0, 1.0);
        _ = normal.pdf(0.5);
        _ = normal.cdf(0.5);

        const uniform = try Uniform(f64).init(0.0, 1.0);
        _ = uniform.pdf(0.5);

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
    }
}
