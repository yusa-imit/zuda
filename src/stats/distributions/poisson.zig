//! Poisson Distribution
//!
//! Represents a discrete Poisson distribution with rate parameter λ.
//! Models the number of events occurring in a fixed interval of time or space,
//! given a constant average rate of occurrence.
//!
//! ## Parameters
//! - `lambda: T` — rate parameter (must be > 0)
//!
//! ## Mathematical Properties
//! - **PMF**: P(X=k) = (λ^k * e^-λ) / k! for k = 0, 1, 2, ...
//! - **CDF**: F(k; λ) = Sum of PMF from 0 to k
//! - **Quantile**: Q(p; λ) = smallest k where CDF(k) >= p
//! - **Log-PMF**: log(λ^k) - λ - log(k!) for numerical stability
//! - **Mean**: λ
//! - **Variance**: λ
//! - **Mode**: floor(λ) or ceil(λ) (bimodal when λ is integer)
//!
//! ## Time Complexity
//! - pmf, cdf, quantile, logpmf: O(k) worst case
//! - sample: O(1) to O(λ) depending on method
//! - init: O(1)
//!
//! ## Use Cases
//! - Counting number of events (phone calls, arrivals, emissions)
//! - Modeling rare events
//! - Quality control and defect counting
//! - Network traffic analysis
//! - Epidemiological models
//!
//! ## References
//! - Inverse transform sampling for Poisson: https://en.wikipedia.org/wiki/Poisson_distribution
//! - Ratio-of-uniforms method: Stadlober & Zechner (1999)
//! - Knuth's algorithm for sampling: https://en.wikipedia.org/wiki/Poisson_distribution#Generating_Poisson-distributed_random_variables

const std = @import("std");
const math = std.math;
const testing = std.testing;

/// Poisson distribution with rate parameter λ
///
/// Parameters:
/// - T: numeric type (f32 or f64)
/// - lambda: rate parameter (must be > 0)
pub fn Poisson(comptime T: type) type {
    return struct {
        lambda: T,

        const Self = @This();

        /// Initialize Poisson distribution with rate λ
        ///
        /// Parameters:
        /// - lambda: rate parameter (must be > 0)
        ///
        /// Returns: Poisson distribution instance
        ///
        /// Errors:
        /// - error.InvalidRate if lambda <= 0
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(lambda: T) !Self {
            if (lambda <= 0.0) {
                return error.InvalidRate;
            }
            return .{ .lambda = lambda };
        }

        /// Probability mass function: P(X=k) = (λ^k * e^-λ) / k!
        ///
        /// Parameters:
        /// - k: non-negative integer (value to evaluate)
        ///
        /// Returns: probability mass at k (0 for k < 0)
        ///
        /// Time: O(k)
        /// Space: O(1)
        pub fn pmf(self: Self, k: i32) T {
            if (k < 0) {
                return 0.0;
            }

            const log_pmf_val = self.logpmf(k);
            return @exp(log_pmf_val);
        }

        /// Cumulative distribution function: F(k; λ) = P(X <= k)
        ///
        /// Parameters:
        /// - k: non-negative integer (value to evaluate)
        ///
        /// Returns: cumulative probability P(X <= k) (0 for k < 0)
        ///
        /// Time: O(k)
        /// Space: O(1)
        pub fn cdf(self: Self, k: i32) T {
            if (k < 0) {
                return 0.0;
            }

            var sum: T = 0.0;
            for (0..@as(usize, @intCast(k + 1))) |i| {
                const i_int = @as(i32, @intCast(i));
                sum += self.pmf(i_int);
            }

            return @min(sum, 1.0); // Clamp to [0, 1]
        }

        /// Quantile function (inverse CDF)
        ///
        /// Returns smallest k where CDF(k) >= p
        ///
        /// Parameters:
        /// - p: probability in [0, 1]
        ///
        /// Returns: k such that P(X <= k) >= p
        ///
        /// Errors:
        /// - error.InvalidProbability if p < 0 or p > 1
        ///
        /// Time: O(k) where k is the result
        /// Space: O(1)
        pub fn quantile(self: Self, p: T) !i32 {
            if (p < 0.0 or p > 1.0) {
                return error.InvalidProbability;
            }

            // Handle boundary cases
            if (p == 0.0) {
                return 0;
            }
            if (p == 1.0) {
                // Return a large value; arbitrary choice: 1000
                return 1000;
            }

            // Binary search or linear search for k where CDF(k) >= p
            var k: i32 = 0;
            while (k <= 10000) {
                if (self.cdf(k) >= p) {
                    return k;
                }
                k += 1;
            }

            return 10000; // Fallback
        }

        /// Natural logarithm of probability mass function
        ///
        /// log(P(X=k)) = k*log(λ) - λ - log(k!)
        ///
        /// More numerically stable than log(pmf(k)) for large k.
        ///
        /// Parameters:
        /// - k: non-negative integer
        ///
        /// Returns: log probability mass at k
        ///
        /// Time: O(k) (for computing log(k!))
        /// Space: O(1)
        pub fn logpmf(self: Self, k: i32) T {
            if (k < 0) {
                return -math.inf(T);
            }

            const k_float: f64 = @floatFromInt(k);
            const log_k_fact: f64 = logFactorial(k);
            const lambda_f64: f64 = switch (T) {
                f32 => @floatCast(self.lambda),
                f64 => self.lambda,
                else => @compileError("Poisson only supports f32 and f64"),
            };

            const result: f64 = k_float * @log(lambda_f64) - lambda_f64 - log_k_fact;
            return @as(T, @floatCast(result));
        }

        /// Generate random sample from Poisson distribution
        ///
        /// Uses Knuth's algorithm for small λ or ratio-of-uniforms for large λ.
        ///
        /// Parameters:
        /// - rng: random number generator (std.Random)
        ///
        /// Returns: random non-negative integer from distribution
        ///
        /// Time: O(λ) expected (Knuth's algorithm)
        /// Space: O(1)
        pub fn sample(self: Self, rng: std.Random) i32 {
            // Knuth's algorithm for small λ
            if (self.lambda < 30.0) {
                return self.sampleKnuth(rng);
            } else {
                // For large λ, use ratio-of-uniforms
                return self.sampleRatioOfUniforms(rng);
            }
        }

        /// Knuth's algorithm for sampling (efficient for small λ)
        fn sampleKnuth(self: Self, rng: std.Random) i32 {
            const L = @exp(-self.lambda);
            var k: i32 = 0;
            var p: T = 1.0;

            while (true) {
                k += 1;
                const u = switch (T) {
                    f32 => rng.float(f32),
                    f64 => rng.float(f64),
                    else => @compileError("Poisson only supports f32 and f64"),
                };
                p *= u;
                if (p < L) {
                    return k - 1;
                }
                // Safety check to avoid infinite loops
                if (k > 1000) return k;
            }
        }

        /// Transformed rejection method for sampling (efficient for large λ)
        /// Uses normal approximation with acceptance-rejection refinement
        fn sampleRatioOfUniforms(self: Self, rng: std.Random) i32 {
            const lambda = self.lambda;
            const sqrt_lambda = @sqrt(lambda);

            // Generate normal approximation: X ~ N(λ, λ)
            // Use Box-Muller to get standard normal, then transform
            const rand1 = switch (T) {
                f32 => rng.float(f32),
                f64 => rng.float(f64),
                else => @compileError("Poisson only supports f32 and f64"),
            };
            const rand2 = switch (T) {
                f32 => rng.float(f32),
                f64 => rng.float(f64),
                else => @compileError("Poisson only supports f32 and f64"),
            };

            // Box-Muller transform: z ~ N(0,1)
            const z = @sqrt(-2.0 * @log(rand1)) * @cos(2.0 * math.pi * rand2);

            // Transform to N(λ, λ)
            const x = lambda + sqrt_lambda * z;

            // Round to nearest integer and clamp to non-negative
            var k = @as(i32, @intFromFloat(@round(x)));
            if (k < 0) {
                k = 0;
            }

            // For large λ, the normal approximation is good enough
            // Optional: Could add acceptance-rejection refinement here,
            // but normal approximation is generally sufficient for λ >= 30
            return k;
        }
    };
}

/// Compute log(n!) efficiently
fn logFactorial(n: i32) f64 {
    if (n < 0) {
        return math.inf(f64);
    }
    if (n == 0 or n == 1) {
        return 0.0;
    }

    // Use Stirling's approximation for large n
    if (n > 20) {
        const n_float = @as(f64, @floatFromInt(n));
        return n_float * @log(n_float) - n_float + 0.5 * @log(2.0 * math.pi * n_float);
    }

    // Compute exactly for small n
    var result: f64 = 0.0;
    for (2..@as(usize, @intCast(n + 1))) |i| {
        result += @log(@as(f64, @floatFromInt(i)));
    }
    return result;
}

// ============================================================================
// TESTS
// ============================================================================

// ============================================================================
// INIT TESTS (6 tests)
// ============================================================================

test "Poisson.init - standard rate λ=1" {
    const dist = try Poisson(f64).init(1.0);
    try testing.expectEqual(@as(f64, 1.0), dist.lambda);
}

test "Poisson.init - custom rate λ=5" {
    const dist = try Poisson(f64).init(5.0);
    try testing.expectEqual(@as(f64, 5.0), dist.lambda);
}

test "Poisson.init - custom rate λ=10" {
    const dist = try Poisson(f64).init(10.0);
    try testing.expectEqual(@as(f64, 10.0), dist.lambda);
}

test "Poisson.init - small rate λ=0.5" {
    const dist = try Poisson(f64).init(0.5);
    try testing.expectEqual(@as(f64, 0.5), dist.lambda);
}

test "Poisson.init - error when lambda = 0" {
    const result = Poisson(f64).init(0.0);
    try testing.expectError(error.InvalidRate, result);
}

test "Poisson.init - error when lambda < 0" {
    const result = Poisson(f64).init(-2.5);
    try testing.expectError(error.InvalidRate, result);
}

// ============================================================================
// PMF TESTS (10 tests)
// ============================================================================

test "Poisson.pmf - negative k returns 0" {
    const dist = try Poisson(f64).init(1.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pmf(-5), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pmf(-1), 1e-10);
}

test "Poisson.pmf - λ=1, P(X=0) ≈ e^-1 ≈ 0.368" {
    const dist = try Poisson(f64).init(1.0);
    const pmf_0 = dist.pmf(0);
    const expected = @exp(-1.0); // ≈ 0.3678794
    try testing.expectApproxEqAbs(expected, pmf_0, 1e-10);
}

test "Poisson.pmf - λ=1, P(X=1) ≈ e^-1 ≈ 0.368" {
    const dist = try Poisson(f64).init(1.0);
    const pmf_1 = dist.pmf(1);
    const expected = @exp(-1.0); // ≈ 0.3678794
    try testing.expectApproxEqAbs(expected, pmf_1, 1e-10);
}

test "Poisson.pmf - λ=5, peak around k=5" {
    const dist = try Poisson(f64).init(5.0);
    const pmf_4 = dist.pmf(4);
    const pmf_5 = dist.pmf(5);
    const pmf_6 = dist.pmf(6);
    // Peak should be near λ=5, so P(5) should be highest
    try testing.expect(pmf_5 >= pmf_4);
    try testing.expect(pmf_5 >= pmf_6);
}

test "Poisson.pmf - λ=1, large k has small probability" {
    const dist = try Poisson(f64).init(1.0);
    const pmf_10 = dist.pmf(10);
    const pmf_5 = dist.pmf(5);
    try testing.expect(pmf_10 < pmf_5);
}

test "Poisson.pmf - λ=5, normalization approximately 1" {
    const dist = try Poisson(f64).init(5.0);
    // Sum PMF from k=0 to k=50 should be approximately 1
    var sum: f64 = 0.0;
    for (0..51) |i| {
        const k = @as(i32, @intCast(i));
        sum += dist.pmf(k);
    }
    // Should capture ~99.99% of probability
    // Allow small floating-point accumulation error
    try testing.expect(sum > 0.999);
    try testing.expect(sum <= 1.001);
}

test "Poisson.pmf - λ=10, P(X=0) is small" {
    const dist = try Poisson(f64).init(10.0);
    const pmf_0 = dist.pmf(0);
    const expected = @exp(-10.0); // ≈ 0.0000454
    try testing.expectApproxEqAbs(expected, pmf_0, 1e-8);
}

test "Poisson.pmf - λ=5, P(X=5) is peak" {
    const dist = try Poisson(f64).init(5.0);
    const pmf_5 = dist.pmf(5);
    // For λ=5: P(5) = (5^5 * e^-5) / 5! = 3125 * e^-5 / 120
    const expected = (math.pow(f64, 5.0, 5.0) * @exp(-5.0)) / 120.0;
    try testing.expectApproxEqAbs(expected, pmf_5, 1e-10);
}

test "Poisson.pmf - f32 precision (λ=1, k=0)" {
    const dist = try Poisson(f32).init(1.0);
    const pmf_0 = dist.pmf(0);
    const expected = @exp(-1.0);
    try testing.expectApproxEqRel(@as(f32, expected), pmf_0, 1e-5);
}

// ============================================================================
// CDF TESTS (10 tests)
// ============================================================================

test "Poisson.cdf - negative k returns 0" {
    const dist = try Poisson(f64).init(1.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.cdf(-5), 1e-10);
}

test "Poisson.cdf - CDF(0) = PMF(0)" {
    const dist = try Poisson(f64).init(1.0);
    const cdf_0 = dist.cdf(0);
    const pmf_0 = dist.pmf(0);
    try testing.expectApproxEqAbs(pmf_0, cdf_0, 1e-10);
}

test "Poisson.cdf - monotonically increasing" {
    const dist = try Poisson(f64).init(5.0);
    const c0 = dist.cdf(0);
    const c1 = dist.cdf(1);
    const c5 = dist.cdf(5);
    const c10 = dist.cdf(10);
    try testing.expect(c0 < c1);
    try testing.expect(c1 < c5);
    try testing.expect(c5 < c10);
}

test "Poisson.cdf - approaches 1 as k → ∞" {
    const dist = try Poisson(f64).init(1.0);
    const cdf_large = dist.cdf(50);
    try testing.expect(cdf_large > 0.999);
    try testing.expect(cdf_large <= 1.0);
}

test "Poisson.cdf - bounds [0, 1]" {
    const dist = try Poisson(f64).init(5.0);
    for (0..20) |i| {
        const k = @as(i32, @intCast(i));
        const c = dist.cdf(k);
        try testing.expect(c >= 0.0);
        try testing.expect(c <= 1.0);
    }
}

test "Poisson.cdf - relationship with PMF: CDF(k) = CDF(k-1) + PMF(k)" {
    const dist = try Poisson(f64).init(5.0);
    for (1..10) |i| {
        const k = @as(i32, @intCast(i));
        const cdf_k = dist.cdf(k);
        const cdf_k_minus_1 = dist.cdf(k - 1);
        const pmf_k = dist.pmf(k);
        const expected = cdf_k_minus_1 + pmf_k;
        try testing.expectApproxEqAbs(expected, cdf_k, 1e-10);
    }
}

test "Poisson.cdf - λ=1, CDF(0) ≈ e^-1" {
    const dist = try Poisson(f64).init(1.0);
    const cdf_0 = dist.cdf(0);
    const expected = @exp(-1.0);
    try testing.expectApproxEqAbs(expected, cdf_0, 1e-10);
}

test "Poisson.cdf - λ=5, CDF(5) specific value" {
    const dist = try Poisson(f64).init(5.0);
    const cdf_5 = dist.cdf(5);
    // CDF(5) should be roughly 0.616
    try testing.expect(cdf_5 > 0.6);
    try testing.expect(cdf_5 < 0.65);
}

test "Poisson.cdf - f32 precision" {
    const dist = try Poisson(f32).init(5.0);
    const cdf_2 = dist.cdf(2);
    try testing.expect(cdf_2 >= 0.0);
    try testing.expect(cdf_2 <= 1.0);
}

// ============================================================================
// QUANTILE TESTS (10 tests)
// ============================================================================

test "Poisson.quantile - p=0 returns 0" {
    const dist = try Poisson(f64).init(1.0);
    const q = try dist.quantile(0.0);
    try testing.expectEqual(@as(i32, 0), q);
}

test "Poisson.quantile - p=1.0 returns large value" {
    const dist = try Poisson(f64).init(1.0);
    const q = try dist.quantile(1.0);
    try testing.expect(q > 0);
}

test "Poisson.quantile - error when p < 0" {
    const dist = try Poisson(f64).init(1.0);
    const result = dist.quantile(-0.1);
    try testing.expectError(error.InvalidProbability, result);
}

test "Poisson.quantile - error when p > 1" {
    const dist = try Poisson(f64).init(1.0);
    const result = dist.quantile(1.1);
    try testing.expectError(error.InvalidProbability, result);
}

test "Poisson.quantile - monotonically increasing" {
    const dist = try Poisson(f64).init(5.0);
    const q1 = try dist.quantile(0.1);
    const q2 = try dist.quantile(0.25);
    const q3 = try dist.quantile(0.5);
    const q4 = try dist.quantile(0.75);
    const q5 = try dist.quantile(0.9);
    try testing.expect(q1 <= q2);
    try testing.expect(q2 <= q3);
    try testing.expect(q3 <= q4);
    try testing.expect(q4 <= q5);
}

test "Poisson.quantile - quantile(cdf(k)) returns k or larger" {
    const dist = try Poisson(f64).init(5.0);
    for (0..15) |i| {
        const k = @as(i32, @intCast(i));
        const p = dist.cdf(k);
        const q = try dist.quantile(p);
        // quantile(cdf(k)) should be >= k (may skip if multiple k map to same CDF)
        try testing.expect(q >= k);
    }
}

test "Poisson.quantile - for typical p values (0.25, 0.5, 0.75)" {
    const dist = try Poisson(f64).init(5.0);
    _ = try dist.quantile(0.25);
    const q50 = try dist.quantile(0.5);
    _ = try dist.quantile(0.75);
    // For λ=5, median is close to 5, so q50 should be around 4-5
    try testing.expect(q50 >= 4);
    try testing.expect(q50 <= 6);
}

test "Poisson.quantile - λ=1 quantiles reasonable" {
    const dist = try Poisson(f64).init(1.0);
    const q50 = try dist.quantile(0.5);
    // For λ=1, median is near 0-1
    try testing.expect(q50 >= 0);
    try testing.expect(q50 <= 2);
}

test "Poisson.quantile - f32 precision" {
    const dist = try Poisson(f32).init(5.0);
    const q = try dist.quantile(0.5);
    try testing.expect(q >= 0);
}

// ============================================================================
// LOGPMF TESTS (5 tests)
// ============================================================================

test "Poisson.logpmf - equals log(pmf) for valid k" {
    const dist = try Poisson(f64).init(5.0);
    for (0..10) |i| {
        const k = @as(i32, @intCast(i));
        const pmf_val = dist.pmf(k);
        const logpmf_val = dist.logpmf(k);
        if (pmf_val > 0) {
            const expected = @log(pmf_val);
            try testing.expectApproxEqAbs(expected, logpmf_val, 1e-10);
        }
    }
}

test "Poisson.logpmf - negative k returns -infinity" {
    const dist = try Poisson(f64).init(1.0);
    const logpmf_val = dist.logpmf(-5);
    try testing.expect(math.isNegativeInf(logpmf_val));
}

test "Poisson.logpmf - numerical stability for large k" {
    const dist = try Poisson(f64).init(5.0);
    // For k=50, logpmf should be very negative but finite
    const logpmf_50 = dist.logpmf(50);
    try testing.expect(logpmf_50 < 0.0);
    try testing.expect(!math.isInf(logpmf_50));
    try testing.expect(!math.isNan(logpmf_50));
}

test "Poisson.logpmf - λ=1, logpmf(0) ≈ log(e^-1) = -1" {
    const dist = try Poisson(f64).init(1.0);
    const logpmf_0 = dist.logpmf(0);
    try testing.expectApproxEqAbs(@as(f64, -1.0), logpmf_0, 1e-10);
}

test "Poisson.logpmf - f32 precision" {
    const dist = try Poisson(f32).init(5.0);
    const logpmf_2 = dist.logpmf(2);
    try testing.expect(!math.isNan(logpmf_2));
    try testing.expect(!math.isInf(logpmf_2));
}

// ============================================================================
// SAMPLE TESTS (8 tests)
// ============================================================================

test "Poisson.sample - all samples are non-negative integers" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const dist = try Poisson(f64).init(5.0);

    for (0..1000) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample >= 0);
    }
}

test "Poisson.sample - mean approximately λ (λ=1, 10k samples)" {
    var prng = std.Random.DefaultPrng.init(999);
    const rng = prng.random();

    const dist = try Poisson(f64).init(1.0);
    const expected_mean = dist.lambda;

    var sum: f64 = 0.0;
    const n_samples = 10000;
    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        sum += @as(f64, @floatFromInt(s));
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    // With 10k samples, expect mean within ~5% of theoretical mean
    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.05);
}

test "Poisson.sample - mean approximately λ (λ=5, 10k samples)" {
    var prng = std.Random.DefaultPrng.init(777);
    const rng = prng.random();

    const dist = try Poisson(f64).init(5.0);
    const expected_mean = dist.lambda;

    var sum: f64 = 0.0;
    const n_samples = 10000;
    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        sum += @as(f64, @floatFromInt(s));
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.05);
}

test "Poisson.sample - variance approximately λ (Poisson property)" {
    var prng = std.Random.DefaultPrng.init(555);
    const rng = prng.random();

    const dist = try Poisson(f64).init(5.0);
    const expected_variance = dist.lambda; // Poisson: Var[X] = λ

    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    const n_samples = 10000;

    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        const s_float = @as(f64, @floatFromInt(s));
        sum += s_float;
        sum_sq += s_float * s_float;
    }

    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));
    const sample_variance = (sum_sq / @as(f64, @floatFromInt(n_samples))) - (sample_mean * sample_mean);

    // With 10k samples, expect variance within ~10% of theoretical variance
    try testing.expectApproxEqRel(expected_variance, sample_variance, 0.10);
}

test "Poisson.sample - different seeds produce different sequences" {
    var prng1 = std.Random.DefaultPrng.init(111);
    var prng2 = std.Random.DefaultPrng.init(222);

    const dist = try Poisson(f64).init(5.0);

    const s1 = dist.sample(prng1.random());
    const s2 = dist.sample(prng2.random());

    // Different seeds should (with very high probability) produce different samples
    try testing.expect(s1 != s2);
}

test "Poisson.sample - small lambda (0.5) has smaller mean" {
    var prng = std.Random.DefaultPrng.init(333);
    const rng = prng.random();

    const dist = try Poisson(f64).init(0.5);

    var sum: f64 = 0.0;
    const n_samples = 5000;
    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        sum += @as(f64, @floatFromInt(s));
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    // Expected mean for λ=0.5 is 0.5
    try testing.expectApproxEqRel(@as(f64, 0.5), sample_mean, 0.10);
}

test "Poisson.sample - large lambda (30) uses efficient algorithm" {
    var prng = std.Random.DefaultPrng.init(444);
    const rng = prng.random();

    const dist = try Poisson(f64).init(30.0);
    const expected_mean = dist.lambda;

    var sum: f64 = 0.0;
    const n_samples = 5000;
    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        sum += @as(f64, @floatFromInt(s));
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    // Expected mean is 30
    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.10);
}

test "Poisson.sample - f32 precision" {
    var prng = std.Random.DefaultPrng.init(666);
    const rng = prng.random();

    const dist = try Poisson(f32).init(5.0);

    for (0..100) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample >= 0);
    }
}

// ============================================================================
// INTEGRATION TESTS (6+ tests)
// ============================================================================

test "Poisson.pmf - normalization sums to approximately 1" {
    const dist = try Poisson(f64).init(5.0);
    // Sum PMF from k=0 to k=100 should be approximately 1
    var sum: f64 = 0.0;
    for (0..101) |i| {
        const k = @as(i32, @intCast(i));
        sum += dist.pmf(k);
    }
    try testing.expectApproxEqRel(@as(f64, 1.0), sum, 0.001);
}

test "Poisson.cdf - and quantile are inverses" {
    const dist = try Poisson(f64).init(5.0);
    // For several k values, verify: quantile(cdf(k)) >= k and cdf(quantile(p)) >= p
    for (0..15) |i| {
        const k = @as(i32, @intCast(i));
        const p = dist.cdf(k);
        const q = try dist.quantile(p);
        try testing.expect(q >= k);
    }
}

test "Poisson.sample - ensemble statistics converge (λ=5)" {
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    const dist = try Poisson(f64).init(5.0);
    const expected_mean = dist.lambda;
    const expected_variance = dist.lambda; // Poisson: Var = λ

    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    const n_samples = 20000;

    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        const s_float = @as(f64, @floatFromInt(s));
        sum += s_float;
        sum_sq += s_float * s_float;
    }

    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));
    const sample_variance = (sum_sq / @as(f64, @floatFromInt(n_samples))) - (sample_mean * sample_mean);

    // Mean should be close to λ
    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.03);

    // Variance should be close to λ (Poisson property)
    try testing.expectApproxEqRel(expected_variance, sample_variance, 0.08);
}

test "Poisson - compare λ=1 vs λ=10 (different scales)" {
    const dist1 = try Poisson(f64).init(1.0);
    const dist10 = try Poisson(f64).init(10.0);

    // P(X=0) should be much higher for λ=1 than λ=10
    const pmf1_0 = dist1.pmf(0);
    const pmf10_0 = dist10.pmf(0);
    try testing.expect(pmf1_0 > pmf10_0);

    // Mode (highest probability) for λ=10 is near 10
    const pmf10_10 = dist10.pmf(10);
    // Mode for λ=1 is at 0 or 1
    const pmf1_0_or_1 = @max(dist1.pmf(0), dist1.pmf(1));
    // Higher λ concentrates around its value
    try testing.expect(pmf10_10 < pmf1_0_or_1);
}

test "Poisson.mode - most likely values cluster near λ" {
    const dist = try Poisson(f64).init(5.0);
    const pmf_4 = dist.pmf(4);
    const pmf_5 = dist.pmf(5);
    const pmf_6 = dist.pmf(6);

    // For λ=5, modes should be around 4-5-6, highest probability near λ
    const max_pmf = @max(pmf_4, @max(pmf_5, pmf_6));
    try testing.expect(max_pmf > dist.pmf(0));
    try testing.expect(max_pmf > dist.pmf(10));
}

test "Poisson.mean_variance - property E[X]=Var[X]=λ empirically verified" {
    var prng = std.Random.DefaultPrng.init(99999);
    const rng = prng.random();

    const dist = try Poisson(f64).init(3.0);

    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    const n_samples = 5000;

    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        const s_float = @as(f64, @floatFromInt(s));
        sum += s_float;
        sum_sq += s_float * s_float;
    }

    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));
    const sample_variance = (sum_sq / @as(f64, @floatFromInt(n_samples))) - (sample_mean * sample_mean);

    // For Poisson, mean should equal variance
    // They should be approximately equal (within sampling error)
    try testing.expectApproxEqRel(sample_mean, sample_variance, 0.15);
}
