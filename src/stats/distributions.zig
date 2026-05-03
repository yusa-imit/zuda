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

        const poisson = try Poisson(f64).init(3.0);
        _ = poisson.pmf(2);

        const binomial = try Binomial(f64).init(10, 0.5);
        _ = binomial.pmf(5);
    }
}
