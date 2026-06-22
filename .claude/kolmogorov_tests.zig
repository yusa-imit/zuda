// ============================================================================
// Kolmogorov Distribution Tests
// ============================================================================

const std = @import("std");
const math = std.math;
const testing = std.testing;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectApproxEqRel = testing.expectApproxEqRel;
const expectEqual = testing.expectEqual;
const expectError = testing.expectError;

// Initialization Tests

test "Kolmogorov: init creates usable instance" {
    const dist = Kolmogorov(f64).init();
    _ = dist;
}

test "Kolmogorov: init returns zero-sized struct (no parameters)" {
    const dist = Kolmogorov(f64).init();
    const size = @sizeOf(Kolmogorov(f64));
    try testing.expect(size == 0);
}

// PDF Tests

test "Kolmogorov: pdf(0) = 0 (boundary)" {
    const dist = Kolmogorov(f64).init();
    const p = dist.pdf(0.0);
    try expectEqual(0.0, p);
}

test "Kolmogorov: pdf(1.0) is positive" {
    const dist = Kolmogorov(f64).init();
    const p = dist.pdf(1.0);
    try testing.expect(p > 0.0);
}

test "Kolmogorov: pdf(1.0) is approximately 0.6867 (series value)" {
    const dist = Kolmogorov(f64).init();
    const p = dist.pdf(1.0);
    // PDF(1.0) ≈ 0.6867 from series computation
    try expectApproxEqAbs(0.6867, p, 0.01);
}

test "Kolmogorov: pdf(0.5) is positive" {
    const dist = Kolmogorov(f64).init();
    const p = dist.pdf(0.5);
    try testing.expect(p > 0.0);
}

test "Kolmogorov: pdf(2.0) is positive but small" {
    const dist = Kolmogorov(f64).init();
    const p = dist.pdf(2.0);
    try testing.expect(p > 0.0);
    try testing.expect(p < 0.1); // Should be much smaller than pdf(1.0)
}

test "Kolmogorov: pdf(negative x) = 0" {
    const dist = Kolmogorov(f64).init();
    const p = dist.pdf(-1.0);
    try expectEqual(0.0, p);
}

test "Kolmogorov: pdf(negative x) = 0 for various negative values" {
    const dist = Kolmogorov(f64).init();
    try expectEqual(0.0, dist.pdf(-0.5));
    try expectEqual(0.0, dist.pdf(-10.0));
    try expectEqual(0.0, dist.pdf(-100.0));
}

test "Kolmogorov: pdf is zero at infinity (x=10.0 is approximately zero)" {
    const dist = Kolmogorov(f64).init();
    const p = dist.pdf(10.0);
    try testing.expect(p < 1e-6);
}

// LogPDF Tests

test "Kolmogorov: logpdf(1.0) ≈ log(pdf(1.0))" {
    const dist = Kolmogorov(f64).init();
    const pdf_val = dist.pdf(1.0);
    const logpdf_val = dist.logpdf(1.0);
    const expected = @log(pdf_val);
    try expectApproxEqAbs(expected, logpdf_val, 1e-10);
}

test "Kolmogorov: logpdf(0.5) ≈ log(pdf(0.5))" {
    const dist = Kolmogorov(f64).init();
    const pdf_val = dist.pdf(0.5);
    const logpdf_val = dist.logpdf(0.5);
    const expected = @log(pdf_val);
    try expectApproxEqAbs(expected, logpdf_val, 1e-10);
}

test "Kolmogorov: logpdf(0) = -infinity" {
    const dist = Kolmogorov(f64).init();
    const lp = dist.logpdf(0.0);
    try testing.expect(math.isNegativeInf(lp));
}

test "Kolmogorov: logpdf(negative x) = -infinity" {
    const dist = Kolmogorov(f64).init();
    const lp = dist.logpdf(-1.0);
    try testing.expect(math.isNegativeInf(lp));
}

// CDF Tests

test "Kolmogorov: cdf(0) = 0 (boundary)" {
    const dist = Kolmogorov(f64).init();
    const c = dist.cdf(0.0);
    try expectEqual(0.0, c);
}

test "Kolmogorov: cdf(1.0) ≈ 0.7300 (well-known value)" {
    const dist = Kolmogorov(f64).init();
    const c = dist.cdf(1.0);
    // CDF(1.0) is a critical value often used in KS tests
    try expectApproxEqAbs(0.7300, c, 1e-4);
}

test "Kolmogorov: cdf(1.36) ≈ 0.95 (α=0.05 critical value)" {
    const dist = Kolmogorov(f64).init();
    const c = dist.cdf(1.36);
    try expectApproxEqAbs(0.95, c, 1e-3);
}

test "Kolmogorov: cdf(1.628) ≈ 0.99 (α=0.01 critical value)" {
    const dist = Kolmogorov(f64).init();
    const c = dist.cdf(1.628);
    try expectApproxEqAbs(0.99, c, 1e-3);
}

test "Kolmogorov: cdf is monotonically increasing" {
    const dist = Kolmogorov(f64).init();
    var prev_cdf = dist.cdf(0.1);
    var x = 0.2;
    while (x <= 3.0) : (x += 0.2) {
        const cdf_x = dist.cdf(x);
        try testing.expect(cdf_x >= prev_cdf);
        prev_cdf = cdf_x;
    }
}

test "Kolmogorov: cdf(10.0) ≈ 1.0 (limit)" {
    const dist = Kolmogorov(f64).init();
    const c = dist.cdf(10.0);
    try testing.expect(c >= 0.999999);
}

test "Kolmogorov: cdf(100.0) = 1.0 (essentially at limit)" {
    const dist = Kolmogorov(f64).init();
    const c = dist.cdf(100.0);
    try expectApproxEqAbs(1.0, c, 1e-6);
}

test "Kolmogorov: cdf(negative x) = 0" {
    const dist = Kolmogorov(f64).init();
    const c = dist.cdf(-1.0);
    try expectEqual(0.0, c);
}

test "Kolmogorov: cdf is always in [0, 1]" {
    const dist = Kolmogorov(f64).init();
    var x = 0.0;
    while (x <= 5.0) : (x += 0.5) {
        const c = dist.cdf(x);
        try testing.expect(c >= 0.0);
        try testing.expect(c <= 1.0);
    }
}

// Quantile Tests

test "Kolmogorov: quantile(0.5) ≈ 0.8275 (median)" {
    const dist = Kolmogorov(f64).init();
    const q = try dist.quantile(0.5);
    // Median of Kolmogorov is approximately 0.8275
    try expectApproxEqAbs(0.8275, q, 1e-2);
}

test "Kolmogorov: quantile(0.73) ≈ 1.0" {
    const dist = Kolmogorov(f64).init();
    const q = try dist.quantile(0.73);
    try expectApproxEqAbs(1.0, q, 1e-3);
}

test "Kolmogorov: quantile(0.95) ≈ 1.36 (inverse of critical value)" {
    const dist = Kolmogorov(f64).init();
    const q = try dist.quantile(0.95);
    try expectApproxEqAbs(1.36, q, 5e-2);
}

test "Kolmogorov: quantile(0.25) is less than median" {
    const dist = Kolmogorov(f64).init();
    const q = try dist.quantile(0.25);
    try testing.expect(q > 0.0);
    try testing.expect(q < 0.8275);
}

test "Kolmogorov: quantile(0.75) is greater than median" {
    const dist = Kolmogorov(f64).init();
    const q = try dist.quantile(0.75);
    try testing.expect(q > 0.8275);
}

test "Kolmogorov: quantile(0) returns error.InvalidProbability" {
    const dist = Kolmogorov(f64).init();
    const result = dist.quantile(0.0);
    try expectError(error.InvalidProbability, result);
}

test "Kolmogorov: quantile(1) returns error.InvalidProbability" {
    const dist = Kolmogorov(f64).init();
    const result = dist.quantile(1.0);
    try expectError(error.InvalidProbability, result);
}

test "Kolmogorov: quantile(-0.1) returns error.InvalidProbability" {
    const dist = Kolmogorov(f64).init();
    const result = dist.quantile(-0.1);
    try expectError(error.InvalidProbability, result);
}

test "Kolmogorov: quantile(1.1) returns error.InvalidProbability" {
    const dist = Kolmogorov(f64).init();
    const result = dist.quantile(1.1);
    try expectError(error.InvalidProbability, result);
}

test "Kolmogorov: quantile(NaN) returns error.InvalidProbability" {
    const dist = Kolmogorov(f64).init();
    const nan = math.nan(f64);
    const result = dist.quantile(nan);
    try expectError(error.InvalidProbability, result);
}

test "Kolmogorov: quantile(infinity) returns error.InvalidProbability" {
    const dist = Kolmogorov(f64).init();
    const inf = math.inf(f64);
    const result = dist.quantile(inf);
    try expectError(error.InvalidProbability, result);
}

test "Kolmogorov: quantile is monotonically increasing" {
    const dist = Kolmogorov(f64).init();
    var prev_q: f64 = 0.0;
    var p = 0.01;
    while (p <= 0.99) : (p += 0.05) {
        const q = try dist.quantile(p);
        try testing.expect(q >= prev_q);
        prev_q = q;
    }
}

test "Kolmogorov: cdf(quantile(p)) ≈ p for various p values" {
    const dist = Kolmogorov(f64).init();
    var p = 0.1;
    while (p < 1.0) : (p += 0.1) {
        const q = try dist.quantile(p);
        const cdf_q = dist.cdf(q);
        try expectApproxEqAbs(p, cdf_q, 1e-3);
    }
}

// Mean, Variance, Entropy Tests

test "Kolmogorov: mean() ≈ 0.8687 (exact theoretical value)" {
    const dist = Kolmogorov(f64).init();
    const m = dist.mean();
    // Mean = √(π/2)·ln(2) ≈ 0.86868...
    try expectApproxEqAbs(0.8687, m, 1e-3);
}

test "Kolmogorov: variance() ≈ 0.2566 (exact theoretical value)" {
    const dist = Kolmogorov(f64).init();
    const v = dist.variance();
    // Variance = π²/12 - π·ln²(2)/2 ≈ 0.2567...
    try expectApproxEqAbs(0.2566, v, 1e-3);
}

test "Kolmogorov: variance is positive" {
    const dist = Kolmogorov(f64).init();
    const v = dist.variance();
    try testing.expect(v > 0.0);
}

test "Kolmogorov: mean is positive" {
    const dist = Kolmogorov(f64).init();
    const m = dist.mean();
    try testing.expect(m > 0.0);
}

test "Kolmogorov: entropy() is positive" {
    const dist = Kolmogorov(f64).init();
    const e = dist.entropy();
    try testing.expect(e > 0.0);
}

test "Kolmogorov: entropy() is finite" {
    const dist = Kolmogorov(f64).init();
    const e = dist.entropy();
    try testing.expect(math.isFinite(e));
}

test "Kolmogorov: mode() is approximately 0.735" {
    const dist = Kolmogorov(f64).init();
    const mo = dist.mode();
    // Kolmogorov has a unique mode around 0.735
    try testing.expect(mo > 0.7);
    try testing.expect(mo < 0.8);
}

// Sample Tests

test "Kolmogorov: sample() produces positive values" {
    const dist = Kolmogorov(f64).init();
    var rng = std.Random.DefaultPrng.init(42);
    for (0..100) |_| {
        const s = dist.sample(rng.random());
        try testing.expect(s > 0.0);
    }
}

test "Kolmogorov: sample() never produces zero" {
    const dist = Kolmogorov(f64).init();
    var rng = std.Random.DefaultPrng.init(42);
    for (0..1000) |_| {
        const s = dist.sample(rng.random());
        try testing.expect(s != 0.0);
    }
}

test "Kolmogorov: sample() never produces negative values" {
    const dist = Kolmogorov(f64).init();
    var rng = std.Random.DefaultPrng.init(42);
    for (0..1000) |_| {
        const s = dist.sample(rng.random());
        try testing.expect(s >= 0.0);
    }
}

test "Kolmogorov: sample() produces mostly reasonable values (< 5.0)" {
    const dist = Kolmogorov(f64).init();
    var rng = std.Random.DefaultPrng.init(42);
    var count_reasonable = 0;
    const total = 1000;
    for (0..total) |_| {
        const s = dist.sample(rng.random());
        if (s < 5.0) count_reasonable += 1;
    }
    // Most samples should be less than 5.0
    try testing.expect(count_reasonable > 900);
}

test "Kolmogorov: sample mean converges to theoretical mean (N=10000)" {
    const dist = Kolmogorov(f64).init();
    var rng = std.Random.DefaultPrng.init(42);
    var sum: f64 = 0.0;
    const n = 10000;
    for (0..n) |_| {
        sum += dist.sample(rng.random());
    }
    const sample_mean = sum / @as(f64, @floatFromInt(n));
    const theoretical_mean = dist.mean();
    // Allow 5% tolerance due to random variation
    try expectApproxEqRel(theoretical_mean, sample_mean, 0.05);
}

// Validate Tests

test "Kolmogorov: validate() always succeeds (no parameters)" {
    const dist = Kolmogorov(f64).init();
    try dist.validate();
}

// f32 Support Tests

test "Kolmogorov: f32 init works" {
    const dist = Kolmogorov(f32).init();
    const pdf_val = dist.pdf(1.0);
    try testing.expect(pdf_val > 0.0);
}

test "Kolmogorov: f32 cdf(1.0) ≈ 0.7300" {
    const dist = Kolmogorov(f32).init();
    const c = dist.cdf(1.0);
    try expectApproxEqAbs(0.7300, c, 1e-3);
}

test "Kolmogorov: f32 mean ≈ 0.8687" {
    const dist = Kolmogorov(f32).init();
    const m = dist.mean();
    try expectApproxEqAbs(0.8687, m, 1e-3);
}

test "Kolmogorov: f32 variance ≈ 0.2566" {
    const dist = Kolmogorov(f32).init();
    const v = dist.variance();
    try expectApproxEqAbs(0.2566, v, 1e-2);
}

test "Kolmogorov: f32 quantile(0.5) ≈ 0.8275 (median)" {
    const dist = Kolmogorov(f32).init();
    const q = try dist.quantile(0.5);
    try expectApproxEqAbs(0.8275, q, 1e-2);
}

test "Kolmogorov: f32 quantile error handling for invalid probabilities" {
    const dist = Kolmogorov(f32).init();
    const result = dist.quantile(-0.5);
    try expectError(error.InvalidProbability, result);
}

test "Kolmogorov: f32 sample produces positive values" {
    const dist = Kolmogorov(f32).init();
    var rng = std.Random.DefaultPrng.init(42);
    for (0..100) |_| {
        const s = dist.sample(rng.random());
        try testing.expect(s > 0.0);
    }
}
