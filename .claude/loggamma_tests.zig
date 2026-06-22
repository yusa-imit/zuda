// LogGamma(T) Distribution Tests
// These tests are FAILING until LogGamma is implemented
// Copy and paste these into src/stats/distributions.zig

test "LogGamma: init with valid alpha and beta succeeds" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    try std.testing.expect(dist.alpha == 2.0);
    try std.testing.expect(dist.beta == 1.0);
}

test "LogGamma: init with fractional alpha succeeds" {
    const dist = try LogGamma(f64).init(0.5, 1.0);
    try std.testing.expect(dist.alpha == 0.5);
}

test "LogGamma: init rejects alpha=0" {
    const result = LogGamma(f64).init(0.0, 1.0);
    try std.testing.expectError(error.InvalidParameter, result);
}

test "LogGamma: init rejects negative alpha" {
    const result = LogGamma(f64).init(-1.0, 1.0);
    try std.testing.expectError(error.InvalidParameter, result);
}

test "LogGamma: init rejects beta=0" {
    const result = LogGamma(f64).init(2.0, 0.0);
    try std.testing.expectError(error.InvalidParameter, result);
}

test "LogGamma: init rejects negative beta" {
    const result = LogGamma(f64).init(2.0, -1.0);
    try std.testing.expectError(error.InvalidParameter, result);
}

test "LogGamma: pdf at y=0 for alpha=1, beta=1 is exp(-1)" {
    const dist = try LogGamma(f64).init(1.0, 1.0);
    const p = dist.pdf(0.0);
    // PDF at mode y=0: f(0) = β^α/Γ(α) * exp(0 - β*exp(0)) = 1 * exp(-1) ≈ 0.3679
    try std.testing.expectApproxEqAbs(@exp(-1.0), p, 1e-5);
}

test "LogGamma: pdf at mode is higher than at neighbors" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    // Mode at log(2/1) = ln(2) ≈ 0.6931
    const mode = dist.mode();
    const at_mode = dist.pdf(mode);
    const left = dist.pdf(mode - 0.1);
    const right = dist.pdf(mode + 0.1);
    try std.testing.expect(at_mode > left);
    try std.testing.expect(at_mode > right);
}

test "LogGamma: logPdf matches log(pdf)" {
    const dist = try LogGamma(f64).init(2.0, 1.5);
    const y = 0.5;
    const pdf_val = dist.pdf(y);
    const logpdf_val = dist.logPdf(y);
    const expected = @log(pdf_val);
    try std.testing.expectApproxEqAbs(expected, logpdf_val, 1e-10);
}

test "LogGamma: pdf is non-negative on range" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    var y: f64 = -5.0;
    while (y <= 10.0) : (y += 0.5) {
        const p = dist.pdf(y);
        try std.testing.expect(p >= 0.0);
    }
}

test "LogGamma: pdf formula check at y=1 for alpha=1, beta=1" {
    const dist = try LogGamma(f64).init(1.0, 1.0);
    const y = 1.0;
    const p = dist.pdf(y);
    // f(1) = 1^1/Γ(1) * exp(1*1 - 1*exp(1)) = exp(1 - e) ≈ exp(-1.71828) ≈ 0.1790
    const expected = @exp(1.0 - @exp(1.0));
    try std.testing.expectApproxEqAbs(expected, p, 1e-5);
}

test "LogGamma: cdf at negative infinity approaches 0" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    const c = dist.cdf(-100.0);
    try std.testing.expectApproxEqAbs(0.0, c, 1e-6);
}

test "LogGamma: cdf at positive infinity approaches 1" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    const c = dist.cdf(100.0);
    try std.testing.expectApproxEqAbs(1.0, c, 1e-6);
}

test "LogGamma: cdf is strictly increasing" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    const c1 = dist.cdf(-1.0);
    const c2 = dist.cdf(0.0);
    const c3 = dist.cdf(1.0);
    const c4 = dist.cdf(2.0);
    try std.testing.expect(c1 < c2);
    try std.testing.expect(c2 < c3);
    try std.testing.expect(c3 < c4);
}

test "LogGamma: cdf at y=0 for alpha=1, beta=1 is 1-exp(-1)" {
    const dist = try LogGamma(f64).init(1.0, 1.0);
    const c = dist.cdf(0.0);
    // CDF(0) = regularizedGammaP(1, 1) = 1 - exp(-1) ≈ 0.6321
    try std.testing.expectApproxEqAbs(1.0 - @exp(-1.0), c, 1e-5);
}

test "LogGamma: quantile rejects p<0" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    const result = dist.quantile(-0.1);
    try std.testing.expectError(error.InvalidProbability, result);
}

test "LogGamma: quantile rejects p>1" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    const result = dist.quantile(1.1);
    try std.testing.expectError(error.InvalidProbability, result);
}

test "LogGamma: quantile at p=0 returns -infinity" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    const q = try dist.quantile(0.0);
    try std.testing.expect(math.isNegativeInf(q));
}

test "LogGamma: quantile at p=1 returns +infinity" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    const q = try dist.quantile(1.0);
    try std.testing.expect(math.isPositiveInf(q));
}

test "LogGamma: cdf-quantile roundtrip at p=0.3" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    const q = try dist.quantile(0.3);
    const c = dist.cdf(q);
    try std.testing.expectApproxEqAbs(0.3, c, 1e-10);
}

test "LogGamma: quantile-cdf roundtrip at y=0.5" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    const c = dist.cdf(0.5);
    const q = try dist.quantile(c);
    try std.testing.expectApproxEqAbs(0.5, q, 1e-9);
}

test "LogGamma: mean for alpha=1, beta=1 is -gamma_E" {
    const dist = try LogGamma(f64).init(1.0, 1.0);
    const m = dist.mean();
    // ψ(1) - log(1) = -γ_E ≈ -0.5772
    const gamma_e = 0.5772156649;
    try std.testing.expectApproxEqAbs(-gamma_e, m, 1e-6);
}

test "LogGamma: mean for alpha=2, beta=1 is 1-gamma_E" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    const m = dist.mean();
    // ψ(2) - log(1) = 1 - γ_E ≈ 0.4228
    const gamma_e = 0.5772156649;
    try std.testing.expectApproxEqAbs(1.0 - gamma_e, m, 1e-6);
}

test "LogGamma: mean for alpha=3, beta=2 scales correctly" {
    const dist = try LogGamma(f64).init(3.0, 2.0);
    const m = dist.mean();
    // ψ(3) - log(2) = (3/2 - γ_E) - 0.6931
    const gamma_e = 0.5772156649;
    const expected = (1.5 - gamma_e) - @log(2.0);
    try std.testing.expectApproxEqAbs(expected, m, 1e-5);
}

test "LogGamma: variance for alpha=1, beta=1 is pi²/6" {
    const dist = try LogGamma(f64).init(1.0, 1.0);
    const v = dist.variance();
    // trigamma(1) = π²/6 ≈ 1.6449
    const expected = math.pi * math.pi / 6.0;
    try std.testing.expectApproxEqAbs(expected, v, 1e-6);
}

test "LogGamma: variance for alpha=2, beta=1 is pi²/6-1" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    const v = dist.variance();
    // trigamma(2) = π²/6 - 1 ≈ 0.6449
    const expected = math.pi * math.pi / 6.0 - 1.0;
    try std.testing.expectApproxEqAbs(expected, v, 1e-6);
}

test "LogGamma: variance independent of beta" {
    const dist1 = try LogGamma(f64).init(2.0, 1.0);
    const dist2 = try LogGamma(f64).init(2.0, 2.0);
    const v1 = dist1.variance();
    const v2 = dist2.variance();
    try std.testing.expectApproxEqAbs(v1, v2, 1e-10);
}

test "LogGamma: mode for alpha=1, beta=1 is 0" {
    const dist = try LogGamma(f64).init(1.0, 1.0);
    const mode = dist.mode();
    try std.testing.expectApproxEqAbs(0.0, mode, 1e-10);
}

test "LogGamma: mode for alpha=2, beta=1 is ln(2)" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    const mode = dist.mode();
    // log(2/1) = ln(2) ≈ 0.6931
    try std.testing.expectApproxEqAbs(@log(2.0), mode, 1e-6);
}

test "LogGamma: mode for alpha=3, beta=2 is ln(1.5)" {
    const dist = try LogGamma(f64).init(3.0, 2.0);
    const mode = dist.mode();
    // log(3/2) = ln(1.5) ≈ 0.4055
    try std.testing.expectApproxEqAbs(@log(1.5), mode, 1e-6);
}

test "LogGamma: entropy for alpha=1, beta=1 is 1+gamma_E" {
    const dist = try LogGamma(f64).init(1.0, 1.0);
    const e = dist.entropy();
    // lgamma(1) + 1*(1 - ψ(1)) = 0 + (1 + γ_E) ≈ 1.5772
    const gamma_e = 0.5772156649;
    try std.testing.expectApproxEqAbs(1.0 + gamma_e, e, 1e-6);
}

test "LogGamma: entropy for alpha=2, beta=1 is 2*gamma_E" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    const e = dist.entropy();
    // lgamma(2) + 2*(1 - ψ(2)) = 0 + 2*γ_E ≈ 1.1544
    const gamma_e = 0.5772156649;
    try std.testing.expectApproxEqAbs(2.0 * gamma_e, e, 1e-6);
}

test "LogGamma: entropy independent of beta" {
    const dist1 = try LogGamma(f64).init(2.0, 1.0);
    const dist2 = try LogGamma(f64).init(2.0, 2.0);
    const e1 = dist1.entropy();
    const e2 = dist2.entropy();
    try std.testing.expectApproxEqAbs(e1, e2, 1e-10);
}

test "LogGamma: validate passes for valid distribution" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    try dist.validate();
}

test "LogGamma: sample returns finite values" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const s = dist.sample(rng);
    try std.testing.expect(math.isFinite(s));
}

test "LogGamma: sample produces values on entire real line" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    var has_negative = false;
    var has_positive = false;
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const s = dist.sample(rng);
        if (s < 0.0) has_negative = true;
        if (s > 0.0) has_positive = true;
    }
    try std.testing.expect(has_negative);
    try std.testing.expect(has_positive);
}

test "LogGamma: empirical mean converges to theoretical" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    var sum: f64 = 0.0;
    const n = 5000;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const s = dist.sample(rng);
        sum += s;
    }
    const empirical_mean = sum / @as(f64, @floatFromInt(n));
    const theoretical_mean = dist.mean();
    try std.testing.expectApproxEqAbs(theoretical_mean, empirical_mean, 0.1);
}

test "LogGamma: f32 support" {
    const dist = try LogGamma(f32).init(2.0, 1.0);
    try dist.validate();
    const p = dist.pdf(0.5);
    try std.testing.expect(p > 0.0);
}

test "LogGamma: special case alpha=1, beta=lambda: mean is -ln(lambda)" {
    const dist = try LogGamma(f64).init(1.0, 2.0);
    const m = dist.mean();
    // ψ(1) - log(2) = -γ_E - ln(2) ≈ -1.2703
    const expected = -0.5772156649 - @log(2.0);
    try std.testing.expectApproxEqAbs(expected, m, 1e-6);
}

test "LogGamma: pdf integrates to approximately 1" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    // Riemann sum integration from -5 to 10, step=0.01
    var sum: f64 = 0.0;
    const step = 0.01;
    var y: f64 = -5.0;
    while (y < 10.0) : (y += step) {
        sum += dist.pdf(y) * step;
    }
    // Should be close to 1
    try std.testing.expectApproxEqAbs(1.0, sum, 0.01);
}

test "LogGamma: sf (survival function) at y=-10 is close to 1" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    const s = dist.sf(-10.0);
    try std.testing.expectApproxEqAbs(1.0, s, 1e-5);
}

test "LogGamma: sf (survival function) at y=10 is close to 0" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    const s = dist.sf(10.0);
    try std.testing.expectApproxEqAbs(0.0, s, 1e-5);
}

test "LogGamma: cdf + sf equals 1" {
    const dist = try LogGamma(f64).init(2.0, 1.0);
    const y = 0.5;
    const c = dist.cdf(y);
    const s = dist.sf(y);
    try std.testing.expectApproxEqAbs(1.0, c + s, 1e-12);
}
