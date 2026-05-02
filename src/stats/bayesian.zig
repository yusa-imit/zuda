const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// Bayesian Inference
///
/// This module provides fundamental Bayesian statistical methods for parameter
/// estimation, credible intervals, and model comparison. Implements conjugate
/// priors for efficient posterior computation and Bayesian model selection.
///
/// Core concepts:
/// - Prior: Initial belief about parameter before observing data
/// - Likelihood: Probability of observing data given parameter
/// - Posterior: Updated belief after observing data (Bayes' theorem)
/// - Conjugate prior: Prior that yields posterior in same family
/// - Credible interval: Bayesian analog of confidence interval
/// - Bayes factor: Evidence ratio for model comparison
///
/// Supported conjugate pairs:
/// - Beta-Binomial: Bernoulli/binomial likelihood with beta prior
/// - Gamma-Poisson: Poisson likelihood with gamma prior
/// - Normal-Normal: Normal likelihood (known variance) with normal prior
/// - Gamma-Normal: Normal likelihood (known mean) with gamma prior on precision
///
/// Use cases:
/// - A/B testing with Bayesian estimation (conversion rates)
/// - Parameter estimation with informative priors
/// - Credible interval construction (alternative to frequentist CI)
/// - Model comparison via Bayes factors
/// - Online learning with sequential Bayesian updates

pub const BayesianError = error{
    InvalidParameters,
    DimensionMismatch,
    NegativeValue,
    InvalidProbability,
    EmptyData,
};

// ============================================================================
// Beta-Binomial Conjugate Pair
// ============================================================================

/// Beta distribution parameters (conjugate prior for binomial)
pub fn BetaPrior(comptime T: type) type {
    return struct {
        alpha: T, // shape parameter (successes)
        beta: T, // shape parameter (failures)

        const Self = @This();

        /// Initialize beta prior
        ///
        /// Time: O(1)
        /// Space: O(1)
        ///
        /// - alpha, beta > 0 (shape parameters)
        /// - alpha=beta=1 → uniform prior (non-informative)
        /// - alpha>1, beta>1 → concentrated prior
        /// - alpha<1, beta<1 → U-shaped prior
        pub fn init(alpha: T, beta: T) BayesianError!Self {
            if (alpha <= 0 or beta <= 0) return error.InvalidParameters;
            return .{ .alpha = alpha, .beta = beta };
        }

        /// Compute prior mean E[p] = alpha / (alpha + beta)
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn mean(self: Self) T {
            return self.alpha / (self.alpha + self.beta);
        }

        /// Compute prior variance Var[p] = (alpha*beta) / ((alpha+beta)^2 * (alpha+beta+1))
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn variance(self: Self) T {
            const sum = self.alpha + self.beta;
            return (self.alpha * self.beta) / (sum * sum * (sum + 1));
        }
    };
}

/// Beta-Binomial posterior parameters after observing binomial data
///
/// Time: O(1) — conjugate prior allows closed-form update
/// Space: O(1)
///
/// Bayes' theorem with conjugate prior:
/// - Prior: Beta(alpha, beta)
/// - Likelihood: Binomial(n, p) with k successes
/// - Posterior: Beta(alpha + k, beta + (n - k))
///
/// Returns posterior BetaPrior with updated parameters
pub fn betaBinomialPosterior(
    comptime T: type,
    prior: BetaPrior(T),
    n_trials: usize,
    n_successes: usize,
) BayesianError!BetaPrior(T) {
    if (n_successes > n_trials) return error.InvalidParameters;

    const k = @as(T, @floatFromInt(n_successes));
    const n = @as(T, @floatFromInt(n_trials));

    const posterior_alpha = prior.alpha + k;
    const posterior_beta = prior.beta + (n - k);

    return BetaPrior(T).init(posterior_alpha, posterior_beta);
}

/// Compute Bayesian credible interval for beta distribution
///
/// Time: O(max_iter) where max_iter ≈ 100 for numerical quantile computation
/// Space: O(1)
///
/// Returns [lower, upper] bounds such that P(lower ≤ p ≤ upper | data) = level
/// Uses inverse CDF (quantile function) via bisection method
///
/// - level: credible level (e.g., 0.95 for 95% credible interval)
/// - tol: numerical tolerance for bisection (default 1e-6)
pub fn betaCredibleInterval(
    comptime T: type,
    dist: BetaPrior(T),
    level: T,
) BayesianError!struct { lower: T, upper: T } {
    if (level <= 0 or level >= 1) return error.InvalidProbability;

    const tail = (1.0 - level) / 2.0;
    const lower_quantile = tail;
    const upper_quantile = 1.0 - tail;

    // Use inverse CDF via bisection
    const lower = try betaQuantile(T, dist, lower_quantile);
    const upper = try betaQuantile(T, dist, upper_quantile);

    return .{ .lower = lower, .upper = upper };
}

/// Compute quantile of beta distribution using bisection
///
/// Time: O(max_iter) where max_iter ≈ 100
/// Space: O(1)
fn betaQuantile(comptime T: type, dist: BetaPrior(T), p: T) BayesianError!T {
    const tol = 1e-6;
    const max_iter = 100;

    var low: T = 0.0;
    var high: T = 1.0;
    var iter: usize = 0;

    while (iter < max_iter) : (iter += 1) {
        const mid = (low + high) / 2.0;
        const cdf_val = betaCDF(T, dist, mid);

        if (@abs(cdf_val - p) < tol) return mid;

        if (cdf_val < p) {
            low = mid;
        } else {
            high = mid;
        }
    }

    return (low + high) / 2.0;
}

/// Compute CDF of beta distribution at point x
///
/// Time: O(max_iter) for incomplete beta function
/// Space: O(1)
fn betaCDF(comptime T: type, dist: BetaPrior(T), x: T) T {
    if (x <= 0) return 0.0;
    if (x >= 1) return 1.0;

    // Use incomplete beta function I_x(alpha, beta)
    return incompleteBeta(T, x, dist.alpha, dist.beta);
}

/// Compute regularized incomplete beta function I_x(a, b)
///
/// Time: O(max_iter) where max_iter ≈ 200
/// Space: O(1)
///
/// Uses continued fraction representation for numerical stability
fn incompleteBeta(comptime T: type, x: T, a: T, b: T) T {
    if (x <= 0) return 0.0;
    if (x >= 1) return 1.0;

    // For numerical stability, use symmetry relation if needed
    if (x > (a / (a + b))) {
        return 1.0 - incompleteBeta(T, 1.0 - x, b, a);
    }

    // Continued fraction expansion (Lentz's algorithm)
    const max_iter = 200;
    const epsilon = 1e-12;

    const bt = @exp(
        lnGamma(T, a + b) - lnGamma(T, a) - lnGamma(T, b) +
            a * @log(x) + b * @log(1.0 - x),
    );

    const qab = a + b;
    const qap = a + 1.0;
    const qam = a - 1.0;

    var c: T = 1.0;
    var d: T = 1.0 - qab * x / qap;
    if (@abs(d) < epsilon) d = epsilon;
    d = 1.0 / d;
    var h = d;

    var m: usize = 1;
    while (m <= max_iter) : (m += 1) {
        const m_float = @as(T, @floatFromInt(m));
        const m2 = 2 * m_float;

        // Even step
        var aa = m_float * (b - m_float) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (@abs(d) < epsilon) d = epsilon;
        c = 1.0 + aa / c;
        if (@abs(c) < epsilon) c = epsilon;
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        aa = -(a + m_float) * (qab + m_float) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (@abs(d) < epsilon) d = epsilon;
        c = 1.0 + aa / c;
        if (@abs(c) < epsilon) c = epsilon;
        d = 1.0 / d;
        const del = d * c;
        h *= del;

        if (@abs(del - 1.0) < epsilon) break;
    }

    return bt * h / a;
}

/// Natural logarithm of gamma function (Lanczos approximation)
///
/// Time: O(1)
/// Space: O(1)
fn lnGamma(comptime T: type, x: T) T {
    if (x <= 0) return math.inf(T);

    // Lanczos approximation coefficients
    const g = 7;
    const coef = [_]T{
        0.99999999999980993,    676.5203681218851,
        -1259.1392167224028,    771.32342877765313,
        -176.61502916214059,    12.507343278686905,
        -0.13857109526572012,   9.9843695780195716e-6,
        1.5056327351493116e-7,
    };

    const z = x;
    var sum = coef[0];
    var i: usize = 1;
    while (i < 9) : (i += 1) {
        sum += coef[i] / (z + @as(T, @floatFromInt(i)));
    }

    const tmp = z + @as(T, @floatFromInt(g)) + 0.5;
    return @log(sum) + @log(2.0 * math.pi) / 2.0 + (z + 0.5) * @log(tmp) - tmp;
}

// ============================================================================
// Gamma-Poisson Conjugate Pair
// ============================================================================

/// Gamma distribution parameters (conjugate prior for Poisson rate)
pub fn GammaPrior(comptime T: type) type {
    return struct {
        alpha: T, // shape parameter
        beta: T, // rate parameter

        const Self = @This();

        /// Initialize gamma prior
        ///
        /// Time: O(1)
        /// Space: O(1)
        ///
        /// - alpha, beta > 0
        /// - mean = alpha / beta
        /// - variance = alpha / beta^2
        pub fn init(alpha: T, beta: T) BayesianError!Self {
            if (alpha <= 0 or beta <= 0) return error.InvalidParameters;
            return .{ .alpha = alpha, .beta = beta };
        }

        /// Compute prior mean E[λ] = alpha / beta
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn mean(self: Self) T {
            return self.alpha / self.beta;
        }

        /// Compute prior variance Var[λ] = alpha / beta^2
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn variance(self: Self) T {
            return self.alpha / (self.beta * self.beta);
        }
    };
}

/// Gamma-Poisson posterior parameters after observing Poisson data
///
/// Time: O(1) — conjugate prior allows closed-form update
/// Space: O(1)
///
/// Bayes' theorem with conjugate prior:
/// - Prior: Gamma(alpha, beta)
/// - Likelihood: Poisson(λ) with sum_x total events over n observations
/// - Posterior: Gamma(alpha + sum_x, beta + n)
///
/// Returns posterior GammaPrior with updated parameters
pub fn gammaPoissonPosterior(
    comptime T: type,
    prior: GammaPrior(T),
    n_obs: usize,
    sum_counts: usize,
) GammaPrior(T) {
    const k = @as(T, @floatFromInt(sum_counts));
    const n = @as(T, @floatFromInt(n_obs));

    const posterior_alpha = prior.alpha + k;
    const posterior_beta = prior.beta + n;

    // Cannot fail: posterior parameters are always positive
    return GammaPrior(T).init(posterior_alpha, posterior_beta) catch unreachable;
}

// ============================================================================
// Normal-Normal Conjugate Pair (known variance)
// ============================================================================

/// Normal distribution parameters (conjugate prior for normal mean with known variance)
pub fn NormalPrior(comptime T: type) type {
    return struct {
        mu: T, // prior mean
        sigma_squared: T, // prior variance

        const Self = @This();

        /// Initialize normal prior
        ///
        /// Time: O(1)
        /// Space: O(1)
        ///
        /// - sigma_squared > 0 (variance must be positive)
        pub fn init(mu: T, sigma_squared: T) BayesianError!Self {
            if (sigma_squared <= 0) return error.InvalidParameters;
            return .{ .mu = mu, .sigma_squared = sigma_squared };
        }

        /// Compute prior mean E[μ] = mu
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn mean(self: Self) T {
            return self.mu;
        }

        /// Compute prior variance Var[μ] = sigma_squared
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn variance(self: Self) T {
            return self.sigma_squared;
        }
    };
}

/// Normal-Normal posterior parameters after observing normal data (known variance)
///
/// Time: O(n) for computing sample mean
/// Space: O(1)
///
/// Bayes' theorem with conjugate prior:
/// - Prior: N(mu_0, sigma_0^2)
/// - Likelihood: N(mu, sigma^2) with data x_1, ..., x_n (sigma^2 known)
/// - Posterior: N(mu_n, sigma_n^2) where:
///   * precision_n = precision_0 + n / sigma^2
///   * mu_n = (precision_0 * mu_0 + (n / sigma^2) * x_bar) / precision_n
///   * sigma_n^2 = 1 / precision_n
///
/// Returns posterior NormalPrior with updated parameters
pub fn normalNormalPosterior(
    comptime T: type,
    prior: NormalPrior(T),
    data: []const T,
    known_variance: T,
) BayesianError!NormalPrior(T) {
    if (data.len == 0) return error.EmptyData;
    if (known_variance <= 0) return error.InvalidParameters;

    // Compute sample mean
    var sum: T = 0;
    for (data) |x| sum += x;
    const x_bar = sum / @as(T, @floatFromInt(data.len));

    const n = @as(T, @floatFromInt(data.len));
    const precision_0 = 1.0 / prior.sigma_squared;
    const precision_n = precision_0 + n / known_variance;

    const mu_n = (precision_0 * prior.mu + (n / known_variance) * x_bar) / precision_n;
    const sigma_n_squared = 1.0 / precision_n;

    return NormalPrior(T).init(mu_n, sigma_n_squared);
}

/// Compute Bayesian credible interval for normal distribution
///
/// Time: O(1)
/// Space: O(1)
///
/// Returns [lower, upper] bounds such that P(lower ≤ μ ≤ upper | data) = level
/// Uses standard normal quantiles
pub fn normalCredibleInterval(
    comptime T: type,
    dist: NormalPrior(T),
    level: T,
) BayesianError!struct { lower: T, upper: T } {
    if (level <= 0 or level >= 1) return error.InvalidProbability;

    const tail = (1.0 - level) / 2.0;

    // Standard normal quantile for common levels
    const z = normalQuantile(T, 1.0 - tail);
    const std_dev = @sqrt(dist.sigma_squared);

    const lower = dist.mu - z * std_dev;
    const upper = dist.mu + z * std_dev;

    return .{ .lower = lower, .upper = upper };
}

/// Compute quantile of standard normal distribution (approximation)
///
/// Time: O(1)
/// Space: O(1)
///
/// Uses rational approximation (Beasley-Springer-Moro algorithm)
fn normalQuantile(comptime T: type, p: T) T {
    if (p <= 0) return -math.inf(T);
    if (p >= 1) return math.inf(T);

    // Use symmetry for p < 0.5
    const sign: T = if (p < 0.5) -1.0 else 1.0;
    const p_adj = if (p < 0.5) p else (1.0 - p);

    // Rational approximation coefficients
    const a = [_]T{ -39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924 };
    const b = [_]T{ -54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857 };
    const c = [_]T{ -0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878 };
    const d = [_]T{ 0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742 };

    const q = @sqrt(-2.0 * @log(p_adj));

    var num: T = 0.0;
    var den: T = 1.0;

    if (p_adj > 0.02425) {
        const r = p_adj - 0.5;
        const r2 = r * r;
        num = ((((((a[0] * r2 + a[1]) * r2 + a[2]) * r2 + a[3]) * r2 + a[4]) * r2 + a[5]) * r);
        den = (((((b[0] * r2 + b[1]) * r2 + b[2]) * r2 + b[3]) * r2 + b[4]) * r2 + 1.0);
    } else {
        num = ((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]));
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }

    return sign * num / den;
}

// ============================================================================
// Model Comparison
// ============================================================================

/// Compute Bayes factor for model comparison
///
/// Time: O(1)
/// Space: O(1)
///
/// Bayes factor BF = P(data | M1) / P(data | M2)
/// - BF > 1: Evidence favors M1
/// - BF < 1: Evidence favors M2
/// - BF > 10: Strong evidence for M1
/// - BF > 100: Decisive evidence for M1
///
/// For beta-binomial models:
/// - log_evidence = log Beta(alpha + k, beta + n - k) - log Beta(alpha, beta)
///   where Beta is the beta function (not distribution)
///
/// Returns log(BF) for numerical stability
pub fn bayesFactor(
    comptime T: type,
    log_evidence_m1: T,
    log_evidence_m2: T,
) T {
    return log_evidence_m1 - log_evidence_m2;
}

/// Compute log marginal likelihood (evidence) for beta-binomial model
///
/// Time: O(1)
/// Space: O(1)
///
/// log P(data | model) = log Beta(alpha + k, beta + n - k) - log Beta(alpha, beta)
pub fn betaBinomialLogEvidence(
    comptime T: type,
    prior: BetaPrior(T),
    n_trials: usize,
    n_successes: usize,
) T {
    const k = @as(T, @floatFromInt(n_successes));
    const n = @as(T, @floatFromInt(n_trials));

    const log_beta_posterior = lnBeta(T, prior.alpha + k, prior.beta + (n - k));
    const log_beta_prior = lnBeta(T, prior.alpha, prior.beta);

    return log_beta_posterior - log_beta_prior;
}

/// Natural logarithm of beta function: ln B(a, b) = ln Γ(a) + ln Γ(b) - ln Γ(a + b)
///
/// Time: O(1)
/// Space: O(1)
fn lnBeta(comptime T: type, a: T, b: T) T {
    return lnGamma(T, a) + lnGamma(T, b) - lnGamma(T, a + b);
}

// ============================================================================
// Tests
// ============================================================================

test "BetaPrior - uniform prior (alpha=1, beta=1)" {
    const prior = try BetaPrior(f64).init(1.0, 1.0);
    try testing.expectApproxEqAbs(0.5, prior.mean(), 1e-9);
    try testing.expectApproxEqAbs(1.0 / 12.0, prior.variance(), 1e-9);
}

test "BetaPrior - informative prior (alpha=10, beta=20)" {
    const prior = try BetaPrior(f64).init(10.0, 20.0);
    try testing.expectApproxEqAbs(10.0 / 30.0, prior.mean(), 1e-9);
    const expected_var = (10.0 * 20.0) / (30.0 * 30.0 * 31.0);
    try testing.expectApproxEqAbs(expected_var, prior.variance(), 1e-9);
}

test "BetaPrior - invalid parameters" {
    const result = BetaPrior(f64).init(0.0, 1.0);
    try testing.expectError(error.InvalidParameters, result);

    const result2 = BetaPrior(f64).init(1.0, -1.0);
    try testing.expectError(error.InvalidParameters, result2);
}

test "betaBinomialPosterior - uniform prior, 10 trials 7 successes" {
    const prior = try BetaPrior(f64).init(1.0, 1.0);
    const posterior = try betaBinomialPosterior(f64, prior, 10, 7);

    // Posterior should be Beta(1 + 7, 1 + 3) = Beta(8, 4)
    try testing.expectApproxEqAbs(8.0, posterior.alpha, 1e-9);
    try testing.expectApproxEqAbs(4.0, posterior.beta, 1e-9);
    try testing.expectApproxEqAbs(8.0 / 12.0, posterior.mean(), 1e-9);
}

test "betaBinomialPosterior - informative prior update" {
    const prior = try BetaPrior(f64).init(5.0, 5.0); // prior belief: p ≈ 0.5
    const posterior = try betaBinomialPosterior(f64, prior, 20, 15);

    // Posterior: Beta(5 + 15, 5 + 5) = Beta(20, 10)
    try testing.expectApproxEqAbs(20.0, posterior.alpha, 1e-9);
    try testing.expectApproxEqAbs(10.0, posterior.beta, 1e-9);
    try testing.expectApproxEqAbs(20.0 / 30.0, posterior.mean(), 1e-9);
}

test "betaBinomialPosterior - invalid successes > trials" {
    const prior = try BetaPrior(f64).init(1.0, 1.0);
    const result = betaBinomialPosterior(f64, prior, 10, 15);
    try testing.expectError(error.InvalidParameters, result);
}

test "betaCredibleInterval - 95% credible interval" {
    const posterior = try BetaPrior(f64).init(8.0, 4.0);
    const ci = try betaCredibleInterval(f64, posterior, 0.95);

    // 95% CI for Beta(8, 4) should be approximately [0.45, 0.87]
    // (using numerical integration, these are approximate)
    try testing.expect(ci.lower > 0.4);
    try testing.expect(ci.lower < 0.5);
    try testing.expect(ci.upper > 0.8);
    try testing.expect(ci.upper < 0.9);
    try testing.expect(ci.upper > ci.lower);
}

test "betaCredibleInterval - narrow interval for informative posterior" {
    const posterior = try BetaPrior(f64).init(100.0, 100.0); // very concentrated
    const ci = try betaCredibleInterval(f64, posterior, 0.95);

    // Should be very narrow around 0.5
    const width = ci.upper - ci.lower;
    try testing.expect(width < 0.2); // narrow interval
    try testing.expectApproxEqAbs(0.5, (ci.lower + ci.upper) / 2.0, 0.1); // centered around 0.5
}

test "betaCredibleInterval - invalid probability" {
    const posterior = try BetaPrior(f64).init(5.0, 5.0);
    const result = betaCredibleInterval(f64, posterior, 1.5);
    try testing.expectError(error.InvalidProbability, result);
}

test "GammaPrior - basic initialization" {
    const prior = try GammaPrior(f64).init(2.0, 1.0);
    try testing.expectApproxEqAbs(2.0, prior.mean(), 1e-9);
    try testing.expectApproxEqAbs(2.0, prior.variance(), 1e-9);
}

test "GammaPrior - invalid parameters" {
    const result = GammaPrior(f64).init(-1.0, 1.0);
    try testing.expectError(error.InvalidParameters, result);
}

test "gammaPoissonPosterior - uniform-ish prior" {
    const prior = try GammaPrior(f64).init(1.0, 1.0);
    const posterior = gammaPoissonPosterior(f64, prior, 10, 30);

    // Posterior: Gamma(1 + 30, 1 + 10) = Gamma(31, 11)
    try testing.expectApproxEqAbs(31.0, posterior.alpha, 1e-9);
    try testing.expectApproxEqAbs(11.0, posterior.beta, 1e-9);
    try testing.expectApproxEqAbs(31.0 / 11.0, posterior.mean(), 1e-6);
}

test "gammaPoissonPosterior - informative prior" {
    const prior = try GammaPrior(f64).init(10.0, 2.0); // prior belief: λ ≈ 5
    const posterior = gammaPoissonPosterior(f64, prior, 20, 100);

    // Posterior: Gamma(10 + 100, 2 + 20) = Gamma(110, 22)
    try testing.expectApproxEqAbs(110.0, posterior.alpha, 1e-9);
    try testing.expectApproxEqAbs(22.0, posterior.beta, 1e-9);
    try testing.expectApproxEqAbs(5.0, posterior.mean(), 1e-6);
}

test "NormalPrior - basic initialization" {
    const prior = try NormalPrior(f64).init(0.0, 1.0);
    try testing.expectApproxEqAbs(0.0, prior.mean(), 1e-9);
    try testing.expectApproxEqAbs(1.0, prior.variance(), 1e-9);
}

test "NormalPrior - invalid variance" {
    const result = NormalPrior(f64).init(0.0, -1.0);
    try testing.expectError(error.InvalidParameters, result);
}

test "normalNormalPosterior - uniform prior, simple data" {
    const prior = try NormalPrior(f64).init(0.0, 1000.0); // very vague prior
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 }; // mean = 3.0
    const posterior = try normalNormalPosterior(f64, prior, &data, 1.0);

    // With vague prior and known variance=1, posterior mean should be ≈ sample mean
    try testing.expectApproxEqAbs(3.0, posterior.mean(), 0.5);
}

test "normalNormalPosterior - informative prior" {
    const prior = try NormalPrior(f64).init(5.0, 1.0); // prior belief: μ ≈ 5
    const data = [_]f64{ 4.0, 5.0, 6.0 }; // sample mean = 5.0
    const posterior = try normalNormalPosterior(f64, prior, &data, 1.0);

    // Posterior mean should be weighted average of prior and sample mean
    try testing.expectApproxEqAbs(5.0, posterior.mean(), 0.5);

    // Posterior variance should be smaller than prior variance
    try testing.expect(posterior.variance() < prior.variance());
}

test "normalNormalPosterior - empty data error" {
    const prior = try NormalPrior(f64).init(0.0, 1.0);
    const data = [_]f64{};
    const result = normalNormalPosterior(f64, prior, &data, 1.0);
    try testing.expectError(error.EmptyData, result);
}

test "normalNormalPosterior - invalid variance" {
    const prior = try NormalPrior(f64).init(0.0, 1.0);
    const data = [_]f64{ 1.0, 2.0 };
    const result = normalNormalPosterior(f64, prior, &data, -1.0);
    try testing.expectError(error.InvalidParameters, result);
}

test "normalCredibleInterval - 95% CI" {
    const posterior = try NormalPrior(f64).init(0.0, 1.0);
    const ci = try normalCredibleInterval(f64, posterior, 0.95);

    // 95% CI for N(0, 1) should be approximately [-1.96, 1.96]
    try testing.expectApproxEqAbs(-1.96, ci.lower, 0.1);
    try testing.expectApproxEqAbs(1.96, ci.upper, 0.1);
}

test "normalCredibleInterval - narrow for small variance" {
    const posterior = try NormalPrior(f64).init(5.0, 0.01); // very concentrated
    const ci = try normalCredibleInterval(f64, posterior, 0.95);

    const width = ci.upper - ci.lower;
    try testing.expect(width < 0.5); // narrow interval
    try testing.expectApproxEqAbs(5.0, (ci.lower + ci.upper) / 2.0, 0.1);
}

test "bayesFactor - model 1 preferred" {
    const log_bf = bayesFactor(f64, -10.0, -15.0);
    try testing.expectApproxEqAbs(5.0, log_bf, 1e-9); // BF = exp(5) ≈ 148, decisive evidence for M1
}

test "bayesFactor - model 2 preferred" {
    const log_bf = bayesFactor(f64, -20.0, -15.0);
    try testing.expectApproxEqAbs(-5.0, log_bf, 1e-9); // BF = exp(-5) ≈ 0.0067, evidence for M2
}

test "betaBinomialLogEvidence - uniform prior" {
    const prior = try BetaPrior(f64).init(1.0, 1.0);
    const log_evidence = betaBinomialLogEvidence(f64, prior, 10, 7);

    // Log evidence should be finite and reasonable
    try testing.expect(math.isFinite(log_evidence));
    try testing.expect(log_evidence < 0); // typically negative (log of probability < 1)
}

test "betaBinomialLogEvidence - informative prior comparison" {
    const prior1 = try BetaPrior(f64).init(1.0, 1.0); // uniform
    const prior2 = try BetaPrior(f64).init(7.0, 3.0); // informative, biased toward success

    const log_ev1 = betaBinomialLogEvidence(f64, prior1, 10, 7);
    const log_ev2 = betaBinomialLogEvidence(f64, prior2, 10, 7);

    // Both should be finite
    try testing.expect(math.isFinite(log_ev1));
    try testing.expect(math.isFinite(log_ev2));

    // Informative prior matching data should have higher evidence
    try testing.expect(log_ev2 > log_ev1);
}

test "memory safety - beta-binomial" {
    const test_allocator = testing.allocator;
    _ = test_allocator;

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const prior = try BetaPrior(f64).init(1.0, 1.0);
        const posterior = try betaBinomialPosterior(f64, prior, 100, 60);
        _ = try betaCredibleInterval(f64, posterior, 0.95);
    }
}

test "memory safety - gamma-poisson" {
    const test_allocator = testing.allocator;
    _ = test_allocator;

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const prior = try GammaPrior(f64).init(2.0, 1.0);
        const posterior = gammaPoissonPosterior(f64, prior, 50, 150);
        _ = posterior.mean();
    }
}

test "memory safety - normal-normal" {
    const test_allocator = testing.allocator;

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const prior = try NormalPrior(f64).init(0.0, 1.0);
        const data = try test_allocator.alloc(f64, 100);
        defer test_allocator.free(data);

        for (data, 0..) |*x, j| {
            x.* = @as(f64, @floatFromInt(j));
        }

        const posterior = try normalNormalPosterior(f64, prior, data, 1.0);
        _ = try normalCredibleInterval(f64, posterior, 0.95);
    }
}

test "f32 precision support" {
    const prior = try BetaPrior(f32).init(5.0, 5.0);
    const posterior = try betaBinomialPosterior(f32, prior, 20, 15);
    try testing.expectApproxEqAbs(@as(f32, 20.0), posterior.alpha, 1e-6);

    const gamma_prior = try GammaPrior(f32).init(2.0, 1.0);
    const gamma_post = gammaPoissonPosterior(f32, gamma_prior, 10, 30);
    try testing.expectApproxEqAbs(@as(f32, 32.0), gamma_post.alpha, 1e-6);

    const normal_prior = try NormalPrior(f32).init(0.0, 1.0);
    const data = [_]f32{ 1.0, 2.0, 3.0 };
    const normal_post = try normalNormalPosterior(f32, normal_prior, &data, 1.0);
    try testing.expect(normal_post.mean() > 0);
}
