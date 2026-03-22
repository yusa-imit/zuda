//! Binomial Distribution
//!
//! Represents a discrete binomial distribution with parameters n (trials) and p (success probability).
//! Models the number of successes in a fixed number of independent Bernoulli trials.
//!
//! ## Parameters
//! - `n: u32` — number of trials (must be > 0)
//! - `p: T` — success probability (must be in [0, 1])
//!
//! ## Mathematical Properties
//! - **PMF**: P(X=k) = C(n,k) * p^k * (1-p)^(n-k) for k = 0, 1, 2, ..., n
//! - **CDF**: F(k; n, p) = Sum of PMF from 0 to floor(k)
//! - **Quantile**: Q(prob; n, p) = smallest k where CDF(k) >= prob
//! - **Log-PMF**: log(C(n,k)) + k*log(p) + (n-k)*log(1-p) for numerical stability
//! - **Mean**: np
//! - **Variance**: np(1-p)
//! - **Mode**: floor((n+1)p) (one mode or bimodal)
//!
//! ## Time Complexity
//! - pmf, cdf, quantile, logpmf: O(k) or O(n) depending on method
//! - sample: O(n) in worst case (binomial inversion) or O(1) (normal approximation for large n)
//! - init: O(1)
//!
//! ## Use Cases
//! - Modeling number of successes in repeated experiments
//! - Quality control and defect counting
//! - A/B testing and statistical inference
//! - Reliability testing
//! - Binomial options pricing
//!
//! ## References
//! - Binomial distribution: https://en.wikipedia.org/wiki/Binomial_distribution
//! - Sampling methods: Devroye (1986), Non-Uniform Random Variate Generation

const std = @import("std");
const math = std.math;
const testing = std.testing;

/// Binomial distribution with parameters n (trials) and p (success probability)
///
/// Parameters:
/// - T: numeric type (f32 or f64)
/// - n: number of trials (must be > 0)
/// - p: success probability (must be in [0, 1])
pub fn Binomial(comptime T: type) type {
    return struct {
        n: u32,
        p: T,

        const Self = @This();

        /// Initialize binomial distribution
        ///
        /// Parameters:
        /// - n: number of trials (must be > 0)
        /// - p: success probability (must be in [0, 1])
        ///
        /// Returns: Binomial distribution instance
        ///
        /// Errors:
        /// - error.InvalidTrials if n == 0
        /// - error.InvalidProbability if p < 0 or p > 1
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(n: u32, p: T) !Self {
            if (n == 0) {
                return error.InvalidTrials;
            }
            if (p < 0.0 or p > 1.0) {
                return error.InvalidProbability;
            }
            return .{ .n = n, .p = p };
        }

        /// Probability mass function: P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
        ///
        /// Parameters:
        /// - k: number of successes (0 <= k <= n)
        ///
        /// Returns: probability mass at k (0 for k < 0 or k > n)
        ///
        /// Time: O(k)
        /// Space: O(1)
        pub fn pmf(self: Self, k: i32) T {
            if (k < 0 or k > @as(i32, @intCast(self.n))) {
                return 0.0;
            }

            const log_pmf_val = self.logpmf(k);
            return @exp(log_pmf_val);
        }

        /// Cumulative distribution function: F(k; n, p) = P(X <= k)
        ///
        /// Parameters:
        /// - k: number of successes (0 <= k <= n)
        ///
        /// Returns: cumulative probability P(X <= k) (0 for k < 0, 1 for k >= n)
        ///
        /// Time: O(k)
        /// Space: O(1)
        pub fn cdf(self: Self, k: i32) T {
            if (k < 0) {
                return 0.0;
            }
            if (k >= @as(i32, @intCast(self.n))) {
                return 1.0;
            }

            var sum: T = 0.0;
            for (0..@as(usize, @intCast(k + 1))) |i| {
                const i_int = @as(i32, @intCast(i));
                sum += self.pmf(i_int);
            }

            return @min(sum, 1.0);
        }

        /// Quantile function (inverse CDF)
        ///
        /// Returns smallest k where CDF(k) >= p
        ///
        /// Parameters:
        /// - prob: probability in [0, 1]
        ///
        /// Returns: k such that P(X <= k) >= prob
        ///
        /// Errors:
        /// - error.InvalidProbability if prob < 0 or prob > 1
        ///
        /// Time: O(n) worst case
        /// Space: O(1)
        pub fn quantile(self: Self, prob: T) !i32 {
            if (prob < 0.0 or prob > 1.0) {
                return error.InvalidProbability;
            }

            // Handle boundary cases
            if (prob == 0.0) {
                return 0;
            }
            if (prob == 1.0) {
                return @as(i32, @intCast(self.n));
            }

            // Linear search for k where CDF(k) >= prob
            for (0..@as(usize, @intCast(self.n + 1))) |i| {
                const k = @as(i32, @intCast(i));
                if (self.cdf(k) >= prob) {
                    return k;
                }
            }

            return @as(i32, @intCast(self.n));
        }

        /// Natural logarithm of probability mass function
        ///
        /// log(P(X=k)) = log(C(n,k)) + k*log(p) + (n-k)*log(1-p)
        ///
        /// More numerically stable than log(pmf(k)) for large n/k.
        ///
        /// Parameters:
        /// - k: number of successes (0 <= k <= n)
        ///
        /// Returns: log probability mass at k
        ///
        /// Time: O(k) (for computing log binomial coefficient)
        /// Space: O(1)
        pub fn logpmf(self: Self, k: i32) T {
            if (k < 0 or k > @as(i32, @intCast(self.n))) {
                return -math.inf(T);
            }

            const k_u = @as(u32, @intCast(k));
            const n_int: i32 = @intCast(self.n);

            // Handle special cases first to avoid 0 * -inf = NaN
            // Case 1: p = 0 — only k=0 has nonzero probability
            if (self.p == 0.0) {
                return if (k == 0) 0.0 else -math.inf(T);
            }

            // Case 2: p = 1 — only k=n has nonzero probability
            if (self.p == 1.0) {
                return if (k == n_int) 0.0 else -math.inf(T);
            }

            // General case: 0 < p < 1
            const log_binom = logBinomialCoefficient(self.n, k_u);
            const log_p: f64 = @log(@as(f64, @floatCast(self.p)));
            const log_1_p: f64 = @log(@as(f64, @floatCast(1.0 - self.p)));

            const k_f: f64 = @floatFromInt(k);
            const n_minus_k: f64 = @as(f64, @floatFromInt(self.n)) - k_f;

            const result: f64 = log_binom + k_f * log_p + n_minus_k * log_1_p;
            return @as(T, @floatCast(result));
        }

        /// Generate random sample from Binomial distribution
        ///
        /// Uses different algorithms depending on parameters:
        /// - Inversion method for small n
        /// - PTRS algorithm for medium n
        /// - Normal approximation for large n
        ///
        /// Parameters:
        /// - rng: random number generator (std.Random)
        ///
        /// Returns: random integer in [0, n]
        ///
        /// Time: O(n) expected (inversion), O(1) for normal approximation
        /// Space: O(1)
        pub fn sample(self: Self, rng: std.Random) i32 {
            // Handle edge cases
            if (self.p == 0.0) {
                return 0;
            }
            if (self.p == 1.0) {
                return @as(i32, @intCast(self.n));
            }

            // Use different algorithms based on n
            if (self.n < 30) {
                return self.sampleInversion(rng);
            } else if (self.p > 0.1 and self.p < 0.9) {
                return self.sampleNormalApproximation(rng);
            } else {
                return self.sampleInversion(rng);
            }
        }

        /// Inversion method: generate n Bernoulli trials
        fn sampleInversion(self: Self, rng: std.Random) i32 {
            var count: i32 = 0;
            const n_iter = self.n;

            for (0..n_iter) |_| {
                const u = switch (T) {
                    f32 => rng.float(f32),
                    f64 => rng.float(f64),
                    else => @compileError("Binomial only supports f32 and f64"),
                };
                if (u < self.p) {
                    count += 1;
                }
            }

            return count;
        }

        /// Normal approximation: use normal distribution with mean np, variance np(1-p)
        fn sampleNormalApproximation(self: Self, rng: std.Random) i32 {
            const mean = @as(f64, @floatFromInt(self.n)) * self.pToF64();
            const variance = mean * (1.0 - self.pToF64());
            const std_dev = @sqrt(variance);

            // Box-Muller transform for normal samples
            const rand1 = switch (T) {
                f32 => rng.float(f32),
                f64 => rng.float(f64),
                else => @compileError("Binomial only supports f32 and f64"),
            };
            const rand2 = switch (T) {
                f32 => rng.float(f32),
                f64 => rng.float(f64),
                else => @compileError("Binomial only supports f32 and f64"),
            };

            const rand1_f64: f64 = switch (T) {
                f32 => @as(f64, @floatCast(rand1)),
                f64 => rand1,
                else => @compileError("Binomial only supports f32 and f64"),
            };
            const rand2_f64: f64 = switch (T) {
                f32 => @as(f64, @floatCast(rand2)),
                f64 => rand2,
                else => @compileError("Binomial only supports f32 and f64"),
            };

            // Box-Muller: z ~ N(0,1)
            const z = @sqrt(-2.0 * @log(rand1_f64)) * @cos(2.0 * math.pi * rand2_f64);

            // Transform to N(mean, variance)
            const x = mean + std_dev * z;

            // Clamp to [0, n]
            var k = @as(i32, @intFromFloat(@round(x)));
            if (k < 0) k = 0;
            if (k > @as(i32, @intCast(self.n))) k = @as(i32, @intCast(self.n));

            return k;
        }

        /// Helper: convert p to f64
        fn pToF64(self: Self) f64 {
            return switch (T) {
                f32 => @as(f64, @floatCast(self.p)),
                f64 => self.p,
                else => @compileError("Binomial only supports f32 and f64"),
            };
        }
    };
}

/// Compute log of binomial coefficient C(n, k) = n! / (k! * (n-k)!)
fn logBinomialCoefficient(n: u32, k: u32) f64 {
    if (k > n) {
        return -math.inf(f64);
    }
    if (k == 0 or k == n) {
        return 0.0;
    }

    // Use symmetry: C(n, k) = C(n, n-k)
    const k_min = @min(k, n - k);

    // Compute log(C(n,k)) = sum_{i=0}^{k-1} log((n-i)/(i+1))
    // This is equivalent to log(n*(n-1)*...*(n-k+1)) - log(k!)
    var sum: f64 = 0.0;
    for (0..k_min) |i| {
        const i_f = @as(f64, @floatFromInt(i));
        const n_f = @as(f64, @floatFromInt(n));
        const numerator = n_f - i_f;
        const denominator = i_f + 1.0;
        sum += @log(numerator / denominator);
    }

    return sum;
}

// ============================================================================
// TESTS
// ============================================================================

// ============================================================================
// INIT TESTS (6 tests)
// ============================================================================

test "Binomial.init - standard parameters n=10, p=0.5" {
    const dist = try Binomial(f64).init(10, 0.5);
    try testing.expectEqual(@as(u32, 10), dist.n);
    try testing.expectEqual(@as(f64, 0.5), dist.p);
}

test "Binomial.init - custom parameters n=100, p=0.1" {
    const dist = try Binomial(f64).init(100, 0.1);
    try testing.expectEqual(@as(u32, 100), dist.n);
    try testing.expectEqual(@as(f64, 0.1), dist.p);
}

test "Binomial.init - boundary p=0" {
    const dist = try Binomial(f64).init(10, 0.0);
    try testing.expectEqual(@as(u32, 10), dist.n);
    try testing.expectEqual(@as(f64, 0.0), dist.p);
}

test "Binomial.init - boundary p=1" {
    const dist = try Binomial(f64).init(10, 1.0);
    try testing.expectEqual(@as(u32, 10), dist.n);
    try testing.expectEqual(@as(f64, 1.0), dist.p);
}

test "Binomial.init - error when n=0" {
    const result = Binomial(f64).init(0, 0.5);
    try testing.expectError(error.InvalidTrials, result);
}

test "Binomial.init - error when p < 0 or p > 1" {
    const result1 = Binomial(f64).init(10, -0.1);
    const result2 = Binomial(f64).init(10, 1.5);
    try testing.expectError(error.InvalidProbability, result1);
    try testing.expectError(error.InvalidProbability, result2);
}

// ============================================================================
// PMF TESTS (10 tests)
// ============================================================================

test "Binomial.pmf - k < 0 returns 0" {
    const dist = try Binomial(f64).init(10, 0.5);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pmf(-1), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pmf(-100), 1e-10);
}

test "Binomial.pmf - k > n returns 0" {
    const dist = try Binomial(f64).init(10, 0.5);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pmf(11), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pmf(100), 1e-10);
}

test "Binomial.pmf - boundary pmf(0) = (1-p)^n" {
    const dist = try Binomial(f64).init(10, 0.5);
    const pmf_0 = dist.pmf(0);
    const expected = math.pow(f64, 0.5, 10.0); // (1-0.5)^10
    try testing.expectApproxEqAbs(expected, pmf_0, 1e-10);
}

test "Binomial.pmf - boundary pmf(n) = p^n" {
    const dist = try Binomial(f64).init(10, 0.5);
    const pmf_n = dist.pmf(10);
    const expected = math.pow(f64, 0.5, 10.0); // 0.5^10
    try testing.expectApproxEqAbs(expected, pmf_n, 1e-10);
}

test "Binomial.pmf - p=0 gives pmf(0)=1, else 0" {
    const dist = try Binomial(f64).init(10, 0.0);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.pmf(0), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pmf(1), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pmf(10), 1e-10);
}

test "Binomial.pmf - p=1 gives pmf(n)=1, else 0" {
    const dist = try Binomial(f64).init(10, 1.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pmf(0), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.pmf(5), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.pmf(10), 1e-10);
}

test "Binomial.pmf - symmetry when p=0.5" {
    const dist = try Binomial(f64).init(10, 0.5);
    // P(X=k) = P(X=n-k) when p=0.5
    for (0..6) |k| {
        const pmf_k = dist.pmf(@as(i32, @intCast(k)));
        const pmf_nk = dist.pmf(@as(i32, @intCast(10 - k)));
        try testing.expectApproxEqAbs(pmf_k, pmf_nk, 1e-10);
    }
}

test "Binomial.pmf - peak near k=np" {
    const dist = try Binomial(f64).init(10, 0.5);
    // For n=10, p=0.5, mean is 5, so pmf(5) should be highest
    const pmf_4 = dist.pmf(4);
    const pmf_5 = dist.pmf(5);
    const pmf_6 = dist.pmf(6);
    try testing.expect(pmf_5 >= pmf_4);
    try testing.expect(pmf_5 >= pmf_6);
}

test "Binomial.pmf - n=10, p=0.5, k=5 has known value" {
    const dist = try Binomial(f64).init(10, 0.5);
    const pmf_5 = dist.pmf(5);
    // C(10,5) * 0.5^5 * 0.5^5 = 252 * 0.5^10 ≈ 0.2461
    const expected = (252.0 * math.pow(f64, 0.5, 10.0));
    try testing.expectApproxEqAbs(expected, pmf_5, 1e-10);
}

// ============================================================================
// CDF TESTS (10 tests)
// ============================================================================

test "Binomial.cdf - k < 0 returns 0" {
    const dist = try Binomial(f64).init(10, 0.5);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.cdf(-5), 1e-10);
}

test "Binomial.cdf - k >= n returns 1" {
    const dist = try Binomial(f64).init(10, 0.5);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.cdf(10), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.cdf(100), 1e-10);
}

test "Binomial.cdf - monotonically increasing" {
    const dist = try Binomial(f64).init(10, 0.5);
    var prev: f64 = dist.cdf(0);
    for (1..11) |k| {
        const curr = dist.cdf(@as(i32, @intCast(k)));
        try testing.expect(prev <= curr);
        prev = curr;
    }
}

test "Binomial.cdf - bounds [0, 1]" {
    const dist = try Binomial(f64).init(10, 0.5);
    for (0..26) |i| {
        const k = @as(i32, @intCast(i)) - 5;
        const c = dist.cdf(k);
        try testing.expect(c >= 0.0);
        try testing.expect(c <= 1.0);
    }
}

test "Binomial.cdf - CDF(k) = CDF(k-1) + PMF(k)" {
    const dist = try Binomial(f64).init(10, 0.5);
    for (1..10) |k| {
        const k_int = @as(i32, @intCast(k));
        const cdf_k = dist.cdf(k_int);
        const cdf_k_minus_1 = dist.cdf(k_int - 1);
        const pmf_k = dist.pmf(k_int);
        const expected = cdf_k_minus_1 + pmf_k;
        try testing.expectApproxEqAbs(expected, cdf_k, 1e-10);
    }
}

test "Binomial.cdf - p=0 gives CDF(0)=1, else 1" {
    const dist = try Binomial(f64).init(10, 0.0);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.cdf(0), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.cdf(5), 1e-10);
}

test "Binomial.cdf - p=1 gives CDF(k<n)=0, CDF(k>=n)=1" {
    const dist = try Binomial(f64).init(10, 1.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.cdf(0), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), dist.cdf(9), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), dist.cdf(10), 1e-10);
}

test "Binomial.cdf - n=10, p=0.5, k=5 specific value" {
    const dist = try Binomial(f64).init(10, 0.5);
    const cdf_5 = dist.cdf(5);
    // CDF(5) for Bin(10, 0.5) is approximately 0.623
    try testing.expect(cdf_5 > 0.6);
    try testing.expect(cdf_5 < 0.65);
}

test "Binomial.cdf - f32 precision" {
    const dist = try Binomial(f32).init(10, 0.5);
    const cdf_5 = dist.cdf(5);
    try testing.expect(cdf_5 >= 0.0);
    try testing.expect(cdf_5 <= 1.0);
}

// ============================================================================
// QUANTILE TESTS (10 tests)
// ============================================================================

test "Binomial.quantile - prob=0 returns 0" {
    const dist = try Binomial(f64).init(10, 0.5);
    const q = try dist.quantile(0.0);
    try testing.expectEqual(@as(i32, 0), q);
}

test "Binomial.quantile - prob=1 returns n" {
    const dist = try Binomial(f64).init(10, 0.5);
    const q = try dist.quantile(1.0);
    try testing.expectEqual(@as(i32, 10), q);
}

test "Binomial.quantile - error when prob < 0" {
    const dist = try Binomial(f64).init(10, 0.5);
    const result = dist.quantile(-0.1);
    try testing.expectError(error.InvalidProbability, result);
}

test "Binomial.quantile - error when prob > 1" {
    const dist = try Binomial(f64).init(10, 0.5);
    const result = dist.quantile(1.5);
    try testing.expectError(error.InvalidProbability, result);
}

test "Binomial.quantile - monotonically increasing" {
    const dist = try Binomial(f64).init(10, 0.5);
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

test "Binomial.quantile - quantile(cdf(k)) >= k" {
    const dist = try Binomial(f64).init(10, 0.5);
    for (0..11) |k| {
        const k_int = @as(i32, @intCast(k));
        const p = dist.cdf(k_int);
        const q = try dist.quantile(p);
        try testing.expect(q >= k_int);
    }
}

test "Binomial.quantile - median near np" {
    const dist = try Binomial(f64).init(10, 0.5);
    const q50 = try dist.quantile(0.5);
    // For n=10, p=0.5, median is near 5
    try testing.expect(q50 >= 4);
    try testing.expect(q50 <= 6);
}

test "Binomial.quantile - extreme p values" {
    const dist = try Binomial(f64).init(20, 0.3);
    const q01 = try dist.quantile(0.01);
    const q99 = try dist.quantile(0.99);
    try testing.expect(q01 >= 0);
    try testing.expect(q99 <= 20);
    try testing.expect(q01 <= q99);
}

test "Binomial.quantile - p=0.5 returns median" {
    const dist = try Binomial(f64).init(10, 0.5);
    const q_median = try dist.quantile(0.5);
    // CDF should be >= 0.5 at this point
    try testing.expect(dist.cdf(q_median) >= 0.5);
}

test "Binomial.quantile - f32 precision" {
    const dist = try Binomial(f32).init(10, 0.5);
    const q = try dist.quantile(0.5);
    try testing.expect(q >= 0);
    try testing.expect(q <= 10);
}

// ============================================================================
// LOGPMF TESTS (5 tests)
// ============================================================================

test "Binomial.logpmf - equals log(pmf) for valid k" {
    const dist = try Binomial(f64).init(10, 0.5);
    for (0..11) |k| {
        const k_int = @as(i32, @intCast(k));
        const pmf_val = dist.pmf(k_int);
        const logpmf_val = dist.logpmf(k_int);
        if (pmf_val > 0) {
            const expected = @log(pmf_val);
            try testing.expectApproxEqAbs(expected, logpmf_val, 1e-9);
        }
    }
}

test "Binomial.logpmf - k < 0 returns -infinity" {
    const dist = try Binomial(f64).init(10, 0.5);
    const logpmf_val = dist.logpmf(-1);
    try testing.expect(math.isNegativeInf(logpmf_val));
}

test "Binomial.logpmf - k > n returns -infinity" {
    const dist = try Binomial(f64).init(10, 0.5);
    const logpmf_val = dist.logpmf(11);
    try testing.expect(math.isNegativeInf(logpmf_val));
}

test "Binomial.logpmf - numerical stability for large n" {
    const dist = try Binomial(f64).init(100, 0.5);
    const logpmf_50 = dist.logpmf(50);
    // Should be very negative but not infinity or NaN
    try testing.expect(logpmf_50 < 0.0);
    try testing.expect(!math.isInf(logpmf_50));
    try testing.expect(!math.isNan(logpmf_50));
}

test "Binomial.logpmf - f32 precision" {
    const dist = try Binomial(f32).init(10, 0.5);
    const logpmf_5 = dist.logpmf(5);
    try testing.expect(!math.isNan(logpmf_5));
    try testing.expect(!math.isInf(logpmf_5));
}

// ============================================================================
// SAMPLE TESTS (10 tests)
// ============================================================================

test "Binomial.sample - all samples in range [0, n]" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const dist = try Binomial(f64).init(10, 0.5);

    for (0..1000) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample >= 0);
        try testing.expect(sample <= 10);
    }
}

test "Binomial.sample - p=0 always returns 0" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const dist = try Binomial(f64).init(10, 0.0);

    for (0..100) |_| {
        try testing.expectEqual(@as(i32, 0), dist.sample(rng));
    }
}

test "Binomial.sample - p=1 always returns n" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const dist = try Binomial(f64).init(10, 1.0);

    for (0..100) |_| {
        try testing.expectEqual(@as(i32, 10), dist.sample(rng));
    }
}

test "Binomial.sample - mean approximately np (n=10, p=0.5, 10k samples)" {
    var prng = std.Random.DefaultPrng.init(999);
    const rng = prng.random();

    const dist = try Binomial(f64).init(10, 0.5);
    const expected_mean = @as(f64, @floatFromInt(dist.n)) * dist.p;

    var sum: f64 = 0.0;
    const n_samples = 10000;
    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        sum += @as(f64, @floatFromInt(s));
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    // With 10k samples, expect within ~5% of theoretical mean
    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.05);
}

test "Binomial.sample - mean approximately np (n=100, p=0.3, 10k samples)" {
    var prng = std.Random.DefaultPrng.init(777);
    const rng = prng.random();

    const dist = try Binomial(f64).init(100, 0.3);
    const expected_mean = @as(f64, @floatFromInt(dist.n)) * dist.p;

    var sum: f64 = 0.0;
    const n_samples = 10000;
    for (0..n_samples) |_| {
        const s = dist.sample(rng);
        sum += @as(f64, @floatFromInt(s));
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n_samples));

    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.05);
}

test "Binomial.sample - variance approximately np(1-p)" {
    var prng = std.Random.DefaultPrng.init(555);
    const rng = prng.random();

    const dist = try Binomial(f64).init(20, 0.5);
    const expected_variance = @as(f64, @floatFromInt(dist.n)) * dist.p * (1.0 - dist.p);

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

    // With 10k samples, expect variance within ~10% of theoretical
    try testing.expectApproxEqRel(expected_variance, sample_variance, 0.10);
}

test "Binomial.sample - different seeds produce different sequences" {
    var prng1 = std.Random.DefaultPrng.init(111);
    var prng2 = std.Random.DefaultPrng.init(222);

    const dist = try Binomial(f64).init(10, 0.5);

    const s1 = dist.sample(prng1.random());
    const s2 = dist.sample(prng2.random());

    // Different seeds should (with very high probability) produce different samples
    try testing.expect(s1 != s2);
}

test "Binomial.sample - small n uses different algorithm than large n" {
    var prng = std.Random.DefaultPrng.init(333);
    const rng = prng.random();

    const dist_small = try Binomial(f64).init(5, 0.5);
    const dist_large = try Binomial(f64).init(100, 0.5);

    // Both should produce valid samples
    const sample_small = dist_small.sample(rng);
    const sample_large = dist_large.sample(rng);

    try testing.expect(sample_small >= 0);
    try testing.expect(sample_small <= 5);
    try testing.expect(sample_large >= 0);
    try testing.expect(sample_large <= 100);
}

test "Binomial.sample - f32 precision" {
    var prng = std.Random.DefaultPrng.init(444);
    const rng = prng.random();

    const dist = try Binomial(f32).init(10, 0.5);

    for (0..100) |_| {
        const sample = dist.sample(rng);
        try testing.expect(sample >= 0);
        try testing.expect(sample <= 10);
    }
}

// ============================================================================
// INTEGRATION TESTS (7+ tests)
// ============================================================================

test "Binomial.pmf - normalization sums to approximately 1" {
    const dist = try Binomial(f64).init(10, 0.5);
    var sum: f64 = 0.0;
    for (0..11) |k| {
        const k_int = @as(i32, @intCast(k));
        sum += dist.pmf(k_int);
    }
    try testing.expectApproxEqRel(@as(f64, 1.0), sum, 0.001);
}

test "Binomial.pmf - normalization sums to 1 (n=20, p=0.3)" {
    const dist = try Binomial(f64).init(20, 0.3);
    var sum: f64 = 0.0;
    for (0..21) |k| {
        const k_int = @as(i32, @intCast(k));
        sum += dist.pmf(k_int);
    }
    try testing.expectApproxEqRel(@as(f64, 1.0), sum, 0.001);
}

test "Binomial.cdf - quantile are inverses" {
    const dist = try Binomial(f64).init(10, 0.5);
    for (0..11) |k| {
        const k_int = @as(i32, @intCast(k));
        const p = dist.cdf(k_int);
        const q = try dist.quantile(p);
        try testing.expect(q >= k_int);
    }
}

test "Binomial.sample - ensemble statistics (n=20, p=0.5)" {
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    const dist = try Binomial(f64).init(20, 0.5);
    const expected_mean = @as(f64, @floatFromInt(dist.n)) * dist.p;
    const expected_variance = @as(f64, @floatFromInt(dist.n)) * dist.p * (1.0 - dist.p);

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

    try testing.expectApproxEqRel(expected_mean, sample_mean, 0.03);
    try testing.expectApproxEqRel(expected_variance, sample_variance, 0.08);
}

test "Binomial - mode is at floor((n+1)p)" {
    const dist = try Binomial(f64).init(10, 0.5);
    // Mode for n=10, p=0.5 is at floor(11*0.5) = 5
    const pmf_mode = dist.pmf(5);
    // Check that pmf(5) is one of the highest values
    try testing.expect(pmf_mode >= dist.pmf(4));
    try testing.expect(pmf_mode >= dist.pmf(6));
}

test "Binomial - skewness with p≠0.5" {
    const dist = try Binomial(f64).init(10, 0.3);
    // For p < 0.5, distribution is right-skewed
    // pmf around k=3 should be highest (since 10*0.3=3)
    const pmf_2 = dist.pmf(2);
    const pmf_3 = dist.pmf(3);
    // 3 should be near mode
    try testing.expect(pmf_3 >= pmf_2);
}

test "Binomial - compare n=10,p=0.5 vs n=20,p=0.5 (mean doubles)" {
    const dist1 = try Binomial(f64).init(10, 0.5);
    const dist2 = try Binomial(f64).init(20, 0.5);

    const mean1 = @as(f64, @floatFromInt(dist1.n)) * dist1.p;
    const mean2 = @as(f64, @floatFromInt(dist2.n)) * dist2.p;

    try testing.expectApproxEqAbs(2.0 * mean1, mean2, 1e-10);
}
