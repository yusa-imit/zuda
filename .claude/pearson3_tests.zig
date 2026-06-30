// Pearson Type III Distribution Tests (128th total, 104th continuous)
// Comprehensive test suite for Pearson Type III (3-parameter gamma family)
// Tests cover: initialization, parameter validation, PDF/CDF/quantile exactness,
// boundary behavior, support constraints, consistency checks, and edge cases

const std = @import("std");
const math = std.math;
const testing = std.testing;
const expect = testing.expect;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectApproxEqRel = testing.expectApproxEqRel;
const expectError = testing.expectError;

// ============================================================================
// INITIALIZATION TESTS
// ============================================================================

test "PearsonIII: init accepts valid positive-skew parameters" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    try expect(dist.mu == 5.0);
    try expect(dist.sigma == 2.0);
    try expect(dist.gamma == 1.0);
}

test "PearsonIII: init accepts valid negative-skew parameters" {
    const dist = try PearsonIII(f64).init(0.0, 1.0, -0.5);
    try expect(dist.mu == 0.0);
    try expect(dist.sigma == 1.0);
    try expect(dist.gamma == -0.5);
}

test "PearsonIII: init accepts zero-skew parameters (normal limit)" {
    const dist = try PearsonIII(f64).init(10.0, 3.0, 0.0);
    try expect(dist.mu == 10.0);
    try expect(dist.sigma == 3.0);
    try expect(dist.gamma == 0.0);
}

test "PearsonIII: init accepts negative mean" {
    const dist = try PearsonIII(f64).init(-5.0, 1.5, 0.8);
    try expect(dist.mu == -5.0);
}

test "PearsonIII: init rejects sigma = 0" {
    try expectError(error.InvalidParameter, PearsonIII(f64).init(0.0, 0.0, 1.0));
}

test "PearsonIII: init rejects sigma < 0" {
    try expectError(error.InvalidParameter, PearsonIII(f64).init(0.0, -1.0, 1.0));
}

test "PearsonIII: init rejects sigma = inf" {
    try expectError(error.InvalidParameter, PearsonIII(f64).init(0.0, math.inf(f64), 1.0));
}

test "PearsonIII: init rejects sigma = nan" {
    try expectError(error.InvalidParameter, PearsonIII(f64).init(0.0, math.nan(f64), 1.0));
}

test "PearsonIII: init rejects mu = inf" {
    try expectError(error.InvalidParameter, PearsonIII(f64).init(math.inf(f64), 1.0, 1.0));
}

test "PearsonIII: init rejects mu = -inf" {
    try expectError(error.InvalidParameter, PearsonIII(f64).init(-math.inf(f64), 1.0, 1.0));
}

test "PearsonIII: init rejects mu = nan" {
    try expectError(error.InvalidParameter, PearsonIII(f64).init(math.nan(f64), 1.0, 1.0));
}

test "PearsonIII: init rejects gamma = inf" {
    try expectError(error.InvalidParameter, PearsonIII(f64).init(0.0, 1.0, math.inf(f64)));
}

test "PearsonIII: init rejects gamma = nan" {
    try expectError(error.InvalidParameter, PearsonIII(f64).init(0.0, 1.0, math.nan(f64)));
}

// ============================================================================
// MEAN, VARIANCE, SKEWNESS
// ============================================================================

test "PearsonIII: mean always returns mu (gamma > 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    try expectApproxEqAbs(dist.mean(), 5.0, 1e-10);
}

test "PearsonIII: mean always returns mu (gamma < 0)" {
    const dist = try PearsonIII(f64).init(-3.5, 1.5, -2.0);
    try expectApproxEqAbs(dist.mean(), -3.5, 1e-10);
}

test "PearsonIII: mean always returns mu (gamma = 0)" {
    const dist = try PearsonIII(f64).init(100.0, 50.0, 0.0);
    try expectApproxEqAbs(dist.mean(), 100.0, 1e-10);
}

test "PearsonIII: variance always returns sigma^2 (gamma > 0)" {
    const dist = try PearsonIII(f64).init(0.0, 2.5, 1.5);
    try expectApproxEqAbs(dist.variance(), 6.25, 1e-10);
}

test "PearsonIII: variance always returns sigma^2 (gamma < 0)" {
    const dist = try PearsonIII(f64).init(10.0, 3.0, -0.5);
    try expectApproxEqAbs(dist.variance(), 9.0, 1e-10);
}

test "PearsonIII: variance always returns sigma^2 (gamma = 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 0.0);
    try expectApproxEqAbs(dist.variance(), 4.0, 1e-10);
}

test "PearsonIII: skewness always returns gamma (gamma > 0)" {
    const dist = try PearsonIII(f64).init(0.0, 1.0, 2.5);
    try expectApproxEqAbs(dist.skewness(), 2.5, 1e-10);
}

test "PearsonIII: skewness always returns gamma (gamma < 0)" {
    const dist = try PearsonIII(f64).init(0.0, 1.0, -1.8);
    try expectApproxEqAbs(dist.skewness(), -1.8, 1e-10);
}

test "PearsonIII: skewness always returns gamma (gamma = 0)" {
    const dist = try PearsonIII(f64).init(0.0, 1.0, 0.0);
    try expectApproxEqAbs(dist.skewness(), 0.0, 1e-10);
}

// ============================================================================
// PDF EXACT VALUES (gamma > 0)
// ============================================================================

test "PearsonIII: PDF exact value for (mu=5, sigma=2, gamma=1) at x=3" {
    // alpha=4, beta=1, xi=1; f(3)=8e^{-2}/6 = 4e^{-2}/3 ≈ 0.180447044
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    const expected = 4.0 * math.exp(-2.0) / 3.0;
    try expectApproxEqAbs(dist.pdf(3.0), expected, 1e-8);
}

test "PearsonIII: PDF exact value for (mu=5, sigma=2, gamma=1) at x=5" {
    // alpha=4, beta=1, xi=1; f(5)=(4*2^3*e^{-4})/6 = 32e^{-4}/6
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    const t = 4.0;
    const expected = (@exp(-t) * math.pow(f64, t, 3.0)) / 6.0; // (alpha-1)=3
    try expectApproxEqAbs(dist.pdf(5.0), expected, 1e-8);
}

test "PearsonIII: PDF exact value for exponential case (mu=0, sigma=1, gamma=2) at x=0" {
    // alpha=1, beta=1, xi=-1; Shifted exponential: f(0) = e^{-1} ≈ 0.367879441
    const dist = try PearsonIII(f64).init(0.0, 1.0, 2.0);
    const expected = math.exp(-1.0);
    try expectApproxEqAbs(dist.pdf(0.0), expected, 1e-8);
}

test "PearsonIII: PDF exact value for exponential case (mu=0, sigma=1, gamma=2) at x=1" {
    // alpha=1, beta=1, xi=-1; t=2, f(1)=e^{-2}
    const dist = try PearsonIII(f64).init(0.0, 1.0, 2.0);
    const expected = math.exp(-2.0);
    try expectApproxEqAbs(dist.pdf(1.0), expected, 1e-8);
}

test "PearsonIII: PDF zero outside support (gamma > 0, x < xi)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    // xi = 5 - 2/1 = 1, support [1, ∞)
    try expectApproxEqAbs(dist.pdf(0.0), 0.0, 1e-15);
    try expectApproxEqAbs(dist.pdf(0.9), 0.0, 1e-15);
    try expectApproxEqAbs(dist.pdf(1.0 - 0.01), 0.0, 1e-15);
}

test "PearsonIII: PDF zero outside support (gamma < 0, x > xi)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, -1.0);
    // xi = 5 + 2/1 = 7, support (-∞, 7]
    try expectApproxEqAbs(dist.pdf(9.0), 0.0, 1e-15);
    try expectApproxEqAbs(dist.pdf(10.0), 0.0, 1e-15);
    try expectApproxEqAbs(dist.pdf(7.0 + 0.01), 0.0, 1e-15);
}

test "PearsonIII: PDF positive in interior (gamma > 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    const xs = [_]f64{ 1.1, 2.0, 3.0, 5.0, 10.0 };
    for (xs) |x| {
        try expect(dist.pdf(x) > 0.0);
    }
}

test "PearsonIII: PDF positive in interior (gamma < 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, -1.0);
    const xs = [_]f64{ -10.0, 0.0, 3.0, 5.0, 6.9 };
    for (xs) |x| {
        try expect(dist.pdf(x) > 0.0);
    }
}

// ============================================================================
// LOG PDF CONSISTENCY
// ============================================================================

test "PearsonIII: logPdf equals log(pdf) at x=3 (gamma > 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    const x = 3.0;
    const log_direct = @log(dist.pdf(x));
    const log_func = dist.logPdf(x);
    try expectApproxEqAbs(log_func, log_direct, 1e-8);
}

test "PearsonIII: logPdf equals log(pdf) at x=5.0 (gamma < 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, -1.0);
    const x = 5.0;
    const log_direct = @log(dist.pdf(x));
    const log_func = dist.logPdf(x);
    try expectApproxEqAbs(log_func, log_direct, 1e-8);
}

test "PearsonIII: logPdf returns -inf outside support (gamma > 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    try expect(math.isNegativeInf(dist.logPdf(0.0)));
}

test "PearsonIII: logPdf returns -inf outside support (gamma < 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, -1.0);
    try expect(math.isNegativeInf(dist.logPdf(9.1)));
}

test "PearsonIII: logPdf equals log(pdf) for normal case (gamma = 0)" {
    const dist = try PearsonIII(f64).init(0.0, 1.0, 0.0);
    const x = 0.5;
    const log_direct = @log(dist.pdf(x));
    const log_func = dist.logPdf(x);
    try expectApproxEqAbs(log_func, log_direct, 1e-8);
}

// ============================================================================
// CDF EXACT VALUES (gamma > 0)
// ============================================================================

test "PearsonIII: CDF exact value for (mu=5, sigma=2, gamma=1) at x=3" {
    // alpha=4, beta=1, xi=1; F(3) = 1 - 19e^{-2}/3 ≈ 0.142876560
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    const expected = 1.0 - 19.0 * math.exp(-2.0) / 3.0;
    try expectApproxEqAbs(dist.cdf(3.0), expected, 1e-7);
}

test "PearsonIII: CDF exact value for (mu=0, sigma=1, gamma=2) at x=0" {
    // alpha=1, beta=1, xi=-1; F(0) = 1 - e^{-1} ≈ 0.632120559
    const dist = try PearsonIII(f64).init(0.0, 1.0, 2.0);
    const expected = 1.0 - math.exp(-1.0);
    try expectApproxEqAbs(dist.cdf(0.0), expected, 1e-8);
}

test "PearsonIII: CDF exact value for normal case (gamma = 0) at x=0" {
    // Normal(0, 1): Φ(0) = 0.5
    const dist = try PearsonIII(f64).init(0.0, 1.0, 0.0);
    try expectApproxEqAbs(dist.cdf(0.0), 0.5, 1e-8);
}

test "PearsonIII: CDF at boundary xi (gamma > 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    // xi = 1, CDF(1) should be 0
    try expectApproxEqAbs(dist.cdf(1.0), 0.0, 1e-10);
}

test "PearsonIII: CDF at boundary xi (gamma < 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, -1.0);
    // xi = 9, CDF(9) should be 1
    try expectApproxEqAbs(dist.cdf(9.0), 1.0, 1e-10);
}

test "PearsonIII: CDF zero for x < xi (gamma > 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    // xi = 1
    try expectApproxEqAbs(dist.cdf(0.0), 0.0, 1e-10);
    try expectApproxEqAbs(dist.cdf(0.5), 0.0, 1e-10);
    try expectApproxEqAbs(dist.cdf(1.0 - 1e-10), 0.0, 1e-10);
}

test "PearsonIII: CDF one for x > xi (gamma < 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, -1.0);
    // xi = 9
    try expectApproxEqAbs(dist.cdf(10.0), 1.0, 1e-10);
    try expectApproxEqAbs(dist.cdf(100.0), 1.0, 1e-10);
    try expectApproxEqAbs(dist.cdf(9.0 + 1e-8), 1.0, 1e-10);
}

test "PearsonIII: CDF monotone increasing (gamma > 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    var prev = dist.cdf(1.0);
    const xs = [_]f64{ 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0 };
    for (xs) |x| {
        const current = dist.cdf(x);
        try expect(current >= prev);
        prev = current;
    }
}

test "PearsonIII: CDF monotone increasing (gamma < 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, -1.0);
    var prev = dist.cdf(-100.0);
    const xs = [_]f64{ -10.0, 0.0, 3.0, 5.0, 6.0, 8.5, 9.0 };
    for (xs) |x| {
        const current = dist.cdf(x);
        try expect(current >= prev);
        prev = current;
    }
}

// ============================================================================
// SF (SURVIVAL FUNCTION) CONSISTENCY
// ============================================================================

test "PearsonIII: SF equals 1 - CDF (gamma > 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    const x = 3.0;
    const sf = dist.sf(x);
    const expected = 1.0 - dist.cdf(x);
    try expectApproxEqAbs(sf, expected, 1e-8);
}

test "PearsonIII: SF equals 1 - CDF (gamma < 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, -1.0);
    const x = 5.0;
    const sf = dist.sf(x);
    const expected = 1.0 - dist.cdf(x);
    try expectApproxEqAbs(sf, expected, 1e-8);
}

test "PearsonIII: SF + CDF = 1 (random points)" {
    const dist = try PearsonIII(f64).init(0.0, 1.0, 0.5);
    const xs = [_]f64{ -10.0, -1.0, 0.0, 1.0, 5.0 };
    for (xs) |x| {
        const sum = dist.cdf(x) + dist.sf(x);
        try expectApproxEqAbs(sum, 1.0, 1e-8);
    }
}

// ============================================================================
// QUANTILE / CDF ROUNDTRIP
// ============================================================================

test "PearsonIII: quantile(cdf(x)) ≈ x (gamma > 0, interior)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    const x = 3.5;
    const p = dist.cdf(x);
    const q = try dist.quantile(p);
    try expectApproxEqAbs(q, x, 1e-6);
}

test "PearsonIII: quantile(cdf(x)) ≈ x (gamma < 0, interior)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, -1.0);
    const x = 3.0;
    const p = dist.cdf(x);
    const q = try dist.quantile(p);
    try expectApproxEqAbs(q, x, 1e-6);
}

test "PearsonIII: quantile(cdf(x)) ≈ x (gamma = 0, normal)" {
    const dist = try PearsonIII(f64).init(0.0, 1.0, 0.0);
    const x = 1.5;
    const p = dist.cdf(x);
    const q = try dist.quantile(p);
    try expectApproxEqAbs(q, x, 1e-6);
}

test "PearsonIII: quantile rejects p < 0" {
    const dist = try PearsonIII(f64).init(0.0, 1.0, 1.0);
    try expectError(error.InvalidProbability, dist.quantile(-0.1));
}

test "PearsonIII: quantile rejects p > 1" {
    const dist = try PearsonIII(f64).init(0.0, 1.0, 1.0);
    try expectError(error.InvalidProbability, dist.quantile(1.1));
}

test "PearsonIII: quantile(0) approaches -infinity (gamma > 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    const q = try dist.quantile(1e-15);
    try expect(q < -1000.0);
}

test "PearsonIII: quantile(1) approaches +infinity (gamma > 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    const q = try dist.quantile(1.0 - 1e-15);
    try expect(q > 1000.0);
}

test "PearsonIII: quantile(0.5) is median (gamma > 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    const q = try dist.quantile(0.5);
    const cdf_q = dist.cdf(q);
    try expectApproxEqAbs(cdf_q, 0.5, 1e-6);
}

test "PearsonIII: quantile(0.25), quantile(0.75) ordering" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    const q25 = try dist.quantile(0.25);
    const q75 = try dist.quantile(0.75);
    try expect(q25 < q75);
}

// ============================================================================
// MODE TESTS
// ============================================================================

test "PearsonIII: mode for alpha >= 1 (|gamma| <= 2)" {
    // gamma = 1, alpha = 4/1 = 4 >= 1
    // mode = mu - sigma*gamma/2 = 5 - 2*1/2 = 4
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    try expectApproxEqAbs(dist.mode(), 4.0, 1e-8);
}

test "PearsonIII: mode for alpha >= 1 (gamma = -1)" {
    // gamma = -1, alpha = 4/1 = 4 >= 1
    // mode = mu - sigma*gamma/2 = 5 - 2*(-1)/2 = 6
    const dist = try PearsonIII(f64).init(5.0, 2.0, -1.0);
    try expectApproxEqAbs(dist.mode(), 6.0, 1e-8);
}

test "PearsonIII: mode for alpha < 1 boundary case (gamma > 2)" {
    // gamma = 3, alpha = 4/9 < 1
    // mode = xi = mu - 2*sigma/gamma = 5 - 2*1/3 = 13/3 ≈ 4.333
    const dist = try PearsonIII(f64).init(5.0, 1.0, 3.0);
    const xi = 5.0 - 2.0 * 1.0 / 3.0;
    try expectApproxEqAbs(dist.mode(), xi, 1e-8);
}

test "PearsonIII: mode for gamma = 0 (normal case)" {
    // gamma = 0 => mode = mu
    const dist = try PearsonIII(f64).init(7.5, 2.0, 0.0);
    try expectApproxEqAbs(dist.mode(), 7.5, 1e-8);
}

test "PearsonIII: mode for gamma = 2 (boundary alpha=1)" {
    // gamma = 2, alpha = 4/4 = 1 (boundary)
    // mode = mu - sigma*gamma/2 = 10 - 1*2/2 = 9
    const dist = try PearsonIII(f64).init(10.0, 1.0, 2.0);
    try expectApproxEqAbs(dist.mode(), 9.0, 1e-8);
}

test "PearsonIII: mode for gamma = -2 (boundary alpha=1)" {
    // gamma = -2, alpha = 4/4 = 1 (boundary)
    // mode = mu - sigma*gamma/2 = 10 - 1*(-2)/2 = 11
    const dist = try PearsonIII(f64).init(10.0, 1.0, -2.0);
    try expectApproxEqAbs(dist.mode(), 11.0, 1e-8);
}

// ============================================================================
// ENTROPY TESTS
// ============================================================================

test "PearsonIII: entropy positive (gamma > 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    try expect(dist.entropy() > 0.0);
}

test "PearsonIII: entropy positive (gamma < 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, -1.0);
    try expect(dist.entropy() > 0.0);
}

test "PearsonIII: entropy finite (gamma = 0, normal)" {
    const dist = try PearsonIII(f64).init(0.0, 1.0, 0.0);
    const h = dist.entropy();
    try expect(math.isFinite(h) and h > 0.0);
}

test "PearsonIII: entropy increases with scale (fixed shape/skew)" {
    const dist1 = try PearsonIII(f64).init(0.0, 1.0, 1.0);
    const dist2 = try PearsonIII(f64).init(0.0, 2.0, 1.0);
    try expect(dist2.entropy() > dist1.entropy());
}

// ============================================================================
// SAMPLE TESTS
// ============================================================================

test "PearsonIII: sample generates finite values (gamma > 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    for (0..20) |_| {
        const s = dist.sample(rng);
        try expect(math.isFinite(s));
        try expect(s >= 1.0); // xi = 1
    }
}

test "PearsonIII: sample generates values in support (gamma > 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    var prng = std.Random.DefaultPrng.init(123);
    const rng = prng.random();
    const xi = 5.0 - 2.0 / 1.0; // xi = 1
    for (0..50) |_| {
        const s = dist.sample(rng);
        try expect(s >= xi - 1e-10); // allow small numerical error
    }
}

test "PearsonIII: sample generates values in support (gamma < 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, -1.0);
    var prng = std.Random.DefaultPrng.init(456);
    const rng = prng.random();
    const xi = 5.0 + 2.0 / 1.0; // xi = 7
    for (0..50) |_| {
        const s = dist.sample(rng);
        try expect(s <= xi + 1e-10); // allow small numerical error
    }
}

test "PearsonIII: sample mean approximates distribution mean (gamma > 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    var prng = std.Random.DefaultPrng.init(789);
    const rng = prng.random();
    var sum: f64 = 0.0;
    const n = 10000;
    for (0..n) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n));
    try expectApproxEqAbs(sample_mean, 5.0, 0.1);
}

// ============================================================================
// VALIDATE METHOD
// ============================================================================

test "PearsonIII: validate passes for valid distribution" {
    const dist = try PearsonIII(f64).init(0.0, 1.0, 1.0);
    try dist.validate();
}

test "PearsonIII: validate passes for all parameter combinations" {
    const params = [_][3]f64{
        [_]f64{ 0.0, 1.0, 0.0 },
        [_]f64{ 5.0, 2.0, 1.0 },
        [_]f64{ 5.0, 2.0, -1.0 },
        [_]f64{ -10.0, 5.0, 3.0 },
        [_]f64{ 100.0, 50.0, -0.5 },
    };
    for (params) |p| {
        const dist = try PearsonIII(f64).init(p[0], p[1], p[2]);
        try dist.validate();
    }
}

// ============================================================================
// F32 TYPE SUPPORT
// ============================================================================

test "PearsonIII: f32 type comprehensive support" {
    const dist = try PearsonIII(f32).init(1.0, 2.0, 1.0);
    try expect(dist.pdf(1.5) > 0.0);
    try expect(dist.cdf(1.5) > 0.0 and dist.cdf(1.5) < 1.0);
    const q = try dist.quantile(0.5);
    try expect(math.isFinite(q) and q > 1.0);
    const m = dist.mode();
    try expect(math.isFinite(m));
    const mn = dist.mean();
    try expectApproxEqAbs(mn, 1.0, 1e-5);
    const v = dist.variance();
    try expectApproxEqAbs(v, 4.0, 1e-5);
    const e = dist.entropy();
    try expect(e > 0.0);
    try dist.validate();
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const s = dist.sample(rng);
    try expect(s > 0.0);
}

// ============================================================================
// EDGE CASES AND NUMERICAL STABILITY
// ============================================================================

test "PearsonIII: handle very small gamma (near-normal)" {
    const dist = try PearsonIII(f64).init(0.0, 1.0, 1e-10);
    const pdf_center = dist.pdf(0.0);
    try expect(pdf_center > 0.0);
    const cdf_center = dist.cdf(0.0);
    try expect(cdf_center > 0.4 and cdf_center < 0.6);
}

test "PearsonIII: handle very large gamma" {
    const dist = try PearsonIII(f64).init(0.0, 1.0, 10.0);
    // alpha = 4/100 = 0.04 (very small)
    // Should still work with alpha < 1
    try expect(dist.pdf(0.0) > 0.0);
    try expect(dist.cdf(0.0) > 0.0);
}

test "PearsonIII: handle very small sigma" {
    const dist = try PearsonIII(f64).init(5.0, 0.001, 1.0);
    // Highly concentrated distribution
    try expect(dist.pdf(5.0) > 0.0);
    const m = dist.mode();
    try expect(@abs(m - 5.0) < 0.01);
}

test "PearsonIII: handle very large sigma" {
    const dist = try PearsonIII(f64).init(0.0, 1000.0, 1.0);
    // Very wide distribution
    try expect(dist.pdf(500.0) > 0.0);
    try expect(dist.cdf(500.0) > 0.0 and dist.cdf(500.0) < 1.0);
}

test "PearsonIII: PDF unimodal (gamma > 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    const mode = dist.mode();
    const p1 = dist.pdf(mode - 1.0);
    const p_mode = dist.pdf(mode);
    const p2 = dist.pdf(mode + 1.0);
    try expect(p1 <= p_mode);
    try expect(p2 <= p_mode);
}

test "PearsonIII: PDF unimodal (gamma < 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, -1.0);
    const mode = dist.mode();
    const p1 = dist.pdf(mode - 1.0);
    const p_mode = dist.pdf(mode);
    const p2 = dist.pdf(mode + 1.0);
    try expect(p1 <= p_mode);
    try expect(p2 <= p_mode);
}

// ============================================================================
// NORMAL LIMIT (gamma -> 0)
// ============================================================================

test "PearsonIII: normal limit for small gamma approaches Normal(mu, sigma^2)" {
    const gamma_small = 1e-8;
    const dist_p3 = try PearsonIII(f64).init(5.0, 2.0, gamma_small);
    const dist_normal = try Normal(f64).init(5.0, 2.0);
    const x = 5.0;
    // PDF should be very close to normal
    try expectApproxEqRel(dist_p3.pdf(x), dist_normal.pdf(x), 1e-4);
    try expectApproxEqAbs(dist_p3.cdf(x), dist_normal.cdf(x), 1e-6);
}

// ============================================================================
// EXTREME VALUE TESTS
// ============================================================================

test "PearsonIII: PDF at very large x (gamma > 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    const pdf_large = dist.pdf(1e6);
    try expect(pdf_large >= 0.0 and pdf_large < 1e-10);
}

test "PearsonIII: CDF approaches 1 for very large x (gamma > 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    const cdf_large = dist.cdf(1e6);
    try expect(cdf_large > 0.9999);
}

test "PearsonIII: CDF approaches 0 for very negative x (gamma < 0)" {
    const dist = try PearsonIII(f64).init(5.0, 2.0, -1.0);
    const cdf_small = dist.cdf(-1e6);
    try expect(cdf_small < 1e-4);
}

// ============================================================================
// PARAMETER SCALE INVARIANCE
// ============================================================================

test "PearsonIII: scale property for CDF" {
    const dist1 = try PearsonIII(f64).init(0.0, 1.0, 1.0);
    const dist2 = try PearsonIII(f64).init(0.0, 2.0, 1.0);
    // CDF at 2x should equal CDF at x for the scaled version
    const cdf1_at_2 = dist1.cdf(2.0);
    const cdf2_at_4 = dist2.cdf(4.0);
    try expectApproxEqAbs(cdf1_at_2, cdf2_at_4, 1e-6);
}

// ============================================================================
// REFLECTION SYMMETRY (gamma -> -gamma)
// ============================================================================

test "PearsonIII: PDF magnitude preserved under reflection" {
    const dist_pos = try PearsonIII(f64).init(5.0, 2.0, 1.0);
    const dist_neg = try PearsonIII(f64).init(5.0, 2.0, -1.0);
    // PDF at symmetric points should have same magnitude
    const x_pos = 5.0 + 2.0; // = 7
    const x_neg = 5.0 - 2.0; // = 3
    const pdf_pos = dist_pos.pdf(x_pos);
    const pdf_neg = dist_neg.pdf(x_neg);
    try expectApproxEqAbs(pdf_pos, pdf_neg, 1e-8);
}
