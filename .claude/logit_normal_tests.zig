// LogitNormal Distribution Tests
// ============================================================================
// Test suite for LogitNormal(mu, sigma) distribution.
// If Y ~ Normal(μ, σ²), then X = sigmoid(Y) ~ LogitNormal(μ, σ)
// Support: (0, 1)
// ============================================================================

const std = @import("std");
const math = std.math;
const testing = std.testing;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectApproxEqRel = testing.expectApproxEqRel;
const expectEqual = testing.expectEqual;
const expectError = testing.expectError;
const expect = testing.expect;

// ============================================================================
// 1. Initialization Tests (8 tests)
// ============================================================================

test "LogitNormal: init valid mu=0 sigma=1" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    try expectEqual(0.0, dist.mu);
    try expectEqual(1.0, dist.sigma);
}

test "LogitNormal: init valid mu=2 sigma=0.5" {
    const dist = try LogitNormal(f64).init(2.0, 0.5);
    try expectEqual(2.0, dist.mu);
    try expectEqual(0.5, dist.sigma);
}

test "LogitNormal: init valid mu=-1 sigma=3" {
    const dist = try LogitNormal(f64).init(-1.0, 3.0);
    try expectEqual(-1.0, dist.mu);
    try expectEqual(3.0, dist.sigma);
}

test "LogitNormal: init fails for sigma=0" {
    try expectError(error.InvalidParameter, LogitNormal(f64).init(0.0, 0.0));
}

test "LogitNormal: init fails for sigma negative" {
    try expectError(error.InvalidParameter, LogitNormal(f64).init(0.0, -1.0));
}

test "LogitNormal: init fails for sigma=NaN" {
    try expectError(error.InvalidParameter, LogitNormal(f64).init(0.0, math.nan(f64)));
}

test "LogitNormal: init fails for sigma=Inf" {
    try expectError(error.InvalidParameter, LogitNormal(f64).init(0.0, math.inf(f64)));
}

test "LogitNormal: init fails for mu=NaN" {
    try expectError(error.InvalidParameter, LogitNormal(f64).init(math.nan(f64), 1.0));
}

// ============================================================================
// 2. PDF Tests (6 tests)
// ============================================================================

test "LogitNormal: pdf(0.5; 0, 1) ≈ 4/sqrt(2π) ≈ 1.59577" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    const expected = 4.0 / @sqrt(2.0 * math.pi);
    const actual = dist.pdf(0.5);
    try expectApproxEqAbs(expected, actual, 1e-4);
}

test "LogitNormal: pdf outside support [0,1] returns 0" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    try expectEqual(0.0, dist.pdf(0.0));
    try expectEqual(0.0, dist.pdf(1.0));
    try expectEqual(0.0, dist.pdf(-0.1));
    try expectEqual(0.0, dist.pdf(1.1));
}

test "LogitNormal: pdf interior points positive" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    try expect(dist.pdf(0.2) > 0.0);
    try expect(dist.pdf(0.5) > 0.0);
    try expect(dist.pdf(0.8) > 0.0);
}

test "LogitNormal: pdf symmetric around 0.5 for mu=0" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    const pdf_low = dist.pdf(0.3);
    const pdf_high = dist.pdf(0.7);
    try expectApproxEqAbs(pdf_low, pdf_high, 1e-10);
}

test "LogitNormal: pdf shifts with mu parameter" {
    const dist1 = try LogitNormal(f64).init(0.0, 1.0);
    const dist2 = try LogitNormal(f64).init(2.0, 1.0);
    const pdf1_at_half = dist1.pdf(0.5);
    const pdf2_at_half = dist2.pdf(0.5);
    try expect(pdf2_at_half > pdf1_at_half);
}

test "LogitNormal: pdf with f32 type support" {
    const dist = try LogitNormal(f32).init(0.0, 1.0);
    const p = dist.pdf(0.5);
    try expect(p > 0.0);
    try expect(math.isFinite(p));
}

// ============================================================================
// 3. Log PDF Tests (5 tests)
// ============================================================================

test "LogitNormal: logpdf(0.5; 0, 1) ≈ 0.46790" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    const expected = @log(4.0 / @sqrt(2.0 * math.pi));
    const actual = dist.logpdf(0.5);
    try expectApproxEqAbs(expected, actual, 1e-4);
}

test "LogitNormal: logpdf outside support returns -inf" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    try expect(math.isNegativeInf(dist.logpdf(0.0)));
    try expect(math.isNegativeInf(dist.logpdf(1.0)));
    try expect(math.isNegativeInf(dist.logpdf(-0.1)));
}

test "LogitNormal: logpdf equals log(pdf) for interior points" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    const xs = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    for (xs) |x| {
        const pdf_val = dist.pdf(x);
        const logpdf_val = dist.logpdf(x);
        const expected = @log(pdf_val);
        try expectApproxEqAbs(expected, logpdf_val, 1e-10);
    }
}

test "LogitNormal: logpdf > 0 near peak" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    const logpdf_at_half = dist.logpdf(0.5);
    try expect(logpdf_at_half > 0.0);
}

test "LogitNormal: logpdf decreases in tails" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    const lp1 = dist.logpdf(0.1);
    const lp2 = dist.logpdf(0.5);
    const lp3 = dist.logpdf(0.9);
    try expect(lp2 > lp1);
    try expect(lp2 > lp3);
}

// ============================================================================
// 4. CDF Tests (6 tests)
// ============================================================================

test "LogitNormal: cdf(0.5; 0, 1) = 0.5 exactly" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    const actual = dist.cdf(0.5);
    try expectApproxEqAbs(0.5, actual, 1e-8);
}

test "LogitNormal: cdf boundary values" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    try expect(dist.cdf(0.0) <= 0.0);
    try expect(dist.cdf(1.0) >= 1.0);
}

test "LogitNormal: cdf(0.7; 0, 1) ≈ Φ(log(7/3)) ≈ 0.80187" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    const logit_val = @log(0.7 / (1.0 - 0.7));
    // Approximate Φ(logit_val) using known value Φ(0.84730) ≈ 0.80187
    const actual = dist.cdf(0.7);
    try expectApproxEqAbs(0.80187, actual, 0.01);
}

test "LogitNormal: cdf symmetry for mu=0: cdf(x) + cdf(1-x) = 1" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    const cdf_low = dist.cdf(0.3);
    const cdf_high = dist.cdf(0.7);
    try expectApproxEqAbs(1.0, cdf_low + cdf_high, 1e-8);
}

test "LogitNormal: cdf monotone increasing" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    const xs = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    var i: usize = 0;
    while (i + 1 < xs.len) : (i += 1) {
        try expect(dist.cdf(xs[i]) < dist.cdf(xs[i + 1]));
    }
}

test "LogitNormal: cdf is well-behaved for various mu values" {
    const dist1 = try LogitNormal(f64).init(0.0, 1.0);
    const dist2 = try LogitNormal(f64).init(2.0, 1.0);
    const cdf1 = dist1.cdf(0.5);
    const cdf2 = dist2.cdf(0.5);
    try expect(cdf1 < cdf2);
}

// ============================================================================
// 5. Quantile Tests (5 tests)
// ============================================================================

test "LogitNormal: quantile invalid p < 0" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    try expectError(error.InvalidProbability, dist.quantile(-0.1));
}

test "LogitNormal: quantile invalid p > 1" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    try expectError(error.InvalidProbability, dist.quantile(1.1));
}

test "LogitNormal: quantile(0.5; mu, sigma) = sigmoid(mu) for any sigma" {
    const mu_vals = [_]f64{ 0.0, 2.0, -1.0 };
    const sigma_vals = [_]f64{ 0.5, 1.0, 3.0 };
    for (mu_vals) |mu| {
        for (sigma_vals) |sigma| {
            const dist = try LogitNormal(f64).init(mu, sigma);
            const q = try dist.quantile(0.5);
            const expected = 1.0 / (1.0 + @exp(-mu));
            try expectApproxEqAbs(expected, q, 1e-10);
        }
    }
}

test "LogitNormal: quantile(0.5; 2, 1) = sigmoid(2) ≈ 0.88080" {
    const dist = try LogitNormal(f64).init(2.0, 1.0);
    const q = try dist.quantile(0.5);
    const expected = 1.0 / (1.0 + @exp(-2.0));
    try expectApproxEqAbs(expected, q, 1e-5);
}

test "LogitNormal: quantile roundtrip with cdf" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    const probs = [_]f64{ 0.1, 0.25, 0.5, 0.75, 0.9 };
    for (probs) |p| {
        const q = try dist.quantile(p);
        const cdf_q = dist.cdf(q);
        try expectApproxEqAbs(p, cdf_q, 0.01);
    }
}

// ============================================================================
// 6. Median Tests (2 tests)
// ============================================================================

test "LogitNormal: median(0, sigma) = 0.5 for any sigma" {
    const sigmas = [_]f64{ 0.5, 1.0, 2.0, 3.0 };
    for (sigmas) |sigma| {
        const dist = try LogitNormal(f64).init(0.0, sigma);
        const m = dist.median();
        try expectApproxEqAbs(0.5, m, 1e-10);
    }
}

test "LogitNormal: median(2, 1) = sigmoid(2) ≈ 0.88080" {
    const dist = try LogitNormal(f64).init(2.0, 1.0);
    const m = dist.median();
    const expected = 1.0 / (1.0 + @exp(-2.0));
    try expectApproxEqAbs(expected, m, 1e-8);
}

// ============================================================================
// 7. Mode Tests (3 tests)
// ============================================================================

test "LogitNormal: mode in (0, 1)" {
    const dist = try LogitNormal(f64).init(0.0, 0.5);
    const m = dist.mode();
    try expect(m > 0.0 and m < 1.0);
}

test "LogitNormal: mode approaches 0.5 as sigma decreases" {
    const dist_small_sigma = try LogitNormal(f64).init(0.0, 0.1);
    const dist_large_sigma = try LogitNormal(f64).init(0.0, 2.0);
    const mode_small = dist_small_sigma.mode();
    const mode_large = dist_large_sigma.mode();
    try expect(@abs(mode_small - 0.5) < @abs(mode_large - 0.5));
}

test "LogitNormal: mode shifts with mu parameter" {
    const dist1 = try LogitNormal(f64).init(0.0, 1.0);
    const dist2 = try LogitNormal(f64).init(1.0, 1.0);
    const mode1 = dist1.mode();
    const mode2 = dist2.mode();
    try expect(mode1 < mode2);
}

// ============================================================================
// 8. Mean/Variance Tests (4 tests)
// ============================================================================

test "LogitNormal: mean(0, 0.5) near 0.5 (symmetric case)" {
    const dist = try LogitNormal(f64).init(0.0, 0.5);
    const m = dist.mean();
    try expect(m > 0.4 and m < 0.6);
}

test "LogitNormal: mean in valid range (0, 1)" {
    const dist = try LogitNormal(f64).init(1.0, 1.5);
    const m = dist.mean();
    try expect(m > 0.0 and m < 1.0);
}

test "LogitNormal: variance positive and finite" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    const v = dist.variance();
    try expect(v > 0.0);
    try expect(math.isFinite(v));
}

test "LogitNormal: variance increases with sigma" {
    const dist_small_sigma = try LogitNormal(f64).init(0.0, 0.5);
    const dist_large_sigma = try LogitNormal(f64).init(0.0, 2.0);
    const var_small = dist_small_sigma.variance();
    const var_large = dist_large_sigma.variance();
    try expect(var_large > var_small);
}

// ============================================================================
// 9. Entropy Tests (2 tests)
// ============================================================================

test "LogitNormal: entropy positive and finite" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    const e = dist.entropy();
    try expect(e > 0.0);
    try expect(math.isFinite(e));
}

test "LogitNormal: entropy increases with sigma" {
    const dist_small_sigma = try LogitNormal(f64).init(0.0, 0.5);
    const dist_large_sigma = try LogitNormal(f64).init(0.0, 2.0);
    const e_small = dist_small_sigma.entropy();
    const e_large = dist_large_sigma.entropy();
    try expect(e_large > e_small);
}

// ============================================================================
// 10. Sample Tests (2 tests)
// ============================================================================

test "LogitNormal: sample returns values in (0, 1)" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const s = dist.sample(rng);
        try expect(s > 0.0 and s < 1.0);
    }
}

test "LogitNormal: sample empirical mean approaches true mean" {
    var prng = std.Random.DefaultPrng.init(54321);
    const rng = prng.random();
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    var sum: f64 = 0.0;
    const n = 1000;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        sum += dist.sample(rng);
    }
    const empirical_mean = sum / @as(f64, @floatFromInt(n));
    const true_mean = dist.mean();
    try expect(@abs(empirical_mean - true_mean) < 0.1);
}

// ============================================================================
// 11. Validate Tests (4 tests)
// ============================================================================

test "LogitNormal: validate passes for valid mu=0 sigma=1" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    try dist.validate();
}

test "LogitNormal: validate passes for valid mu=5 sigma=2" {
    const dist = try LogitNormal(f64).init(5.0, 2.0);
    try dist.validate();
}

test "LogitNormal: validate fails for sigma=0" {
    var dist = try LogitNormal(f64).init(0.0, 1.0);
    dist.sigma = 0.0;
    try expectError(error.InvalidParameter, dist.validate());
}

test "LogitNormal: validate fails for sigma NaN" {
    var dist = try LogitNormal(f64).init(0.0, 1.0);
    dist.sigma = math.nan(f64);
    try expectError(error.InvalidParameter, dist.validate());
}

// ============================================================================
// 12. Additional Edge Cases and Properties (4 tests)
// ============================================================================

test "LogitNormal: pdf integral approximates 1 (midpoint rule)" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    var sum: f64 = 0.0;
    var x: f64 = 0.001;
    const dx = 0.999 / 500.0;
    var i: usize = 0;
    while (i < 500) : (i += 1) {
        sum += dist.pdf(x) * dx;
        x += dx;
    }
    try expect(sum > 0.95 and sum < 1.05);
}

test "LogitNormal: cdf and sf sum to 1" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    const xs = [_]f64{ 0.2, 0.5, 0.8 };
    for (xs) |x| {
        const cdf_val = dist.cdf(x);
        const sf_val = 1.0 - cdf_val;
        try expectApproxEqAbs(1.0, cdf_val + sf_val, 1e-10);
    }
}

test "LogitNormal: pdf decreases away from mode" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    const mode = dist.mode();
    const pdf_at_mode = dist.pdf(mode);
    const pdf_nearby = dist.pdf(mode + 0.1);
    try expect(pdf_at_mode > pdf_nearby);
}

test "LogitNormal: quantile boundary conditions" {
    const dist = try LogitNormal(f64).init(0.0, 1.0);
    const q_small = try dist.quantile(0.001);
    const q_large = try dist.quantile(0.999);
    try expect(q_small > 0.0 and q_small < 0.5);
    try expect(q_large > 0.5 and q_large < 1.0);
}
