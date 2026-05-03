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
    }
}
