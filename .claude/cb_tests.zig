// ContinuousBernoulli Distribution Tests
// Place these tests after the ContinuousBernoulli implementation in distributions.zig

test "ContinuousBernoulli: init succeeds with lambda=0.3" {
    const dist = try ContinuousBernoulli(f64).init(0.3);
    try testing.expectEqual(@as(f64, 0.3), dist.lambda);
}

test "ContinuousBernoulli: init succeeds with lambda=0.5" {
    const dist = try ContinuousBernoulli(f64).init(0.5);
    try testing.expectEqual(@as(f64, 0.5), dist.lambda);
}

test "ContinuousBernoulli: init succeeds with lambda=0.7" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    try testing.expectEqual(@as(f64, 0.7), dist.lambda);
}

test "ContinuousBernoulli: init succeeds with lambda=0.001" {
    const dist = try ContinuousBernoulli(f64).init(0.001);
    try testing.expectEqual(@as(f64, 0.001), dist.lambda);
}

test "ContinuousBernoulli: init succeeds with lambda=0.999" {
    const dist = try ContinuousBernoulli(f64).init(0.999);
    try testing.expectEqual(@as(f64, 0.999), dist.lambda);
}

test "ContinuousBernoulli: init fails for lambda=0" {
    try testing.expectError(error.InvalidParameter, ContinuousBernoulli(f64).init(0.0));
}

test "ContinuousBernoulli: init fails for lambda=1" {
    try testing.expectError(error.InvalidParameter, ContinuousBernoulli(f64).init(1.0));
}

test "ContinuousBernoulli: init fails for lambda=-0.1" {
    try testing.expectError(error.InvalidParameter, ContinuousBernoulli(f64).init(-0.1));
}

test "ContinuousBernoulli: init fails for lambda=1.1" {
    try testing.expectError(error.InvalidParameter, ContinuousBernoulli(f64).init(1.1));
}

test "ContinuousBernoulli: init fails for lambda=NaN" {
    try testing.expectError(error.InvalidParameter, ContinuousBernoulli(f64).init(math.nan(f64)));
}

test "ContinuousBernoulli: init fails for lambda=inf" {
    try testing.expectError(error.InvalidParameter, ContinuousBernoulli(f64).init(math.inf(f64)));
}

test "ContinuousBernoulli: init fails for lambda=-inf" {
    try testing.expectError(error.InvalidParameter, ContinuousBernoulli(f64).init(-math.inf(f64)));
}

test "ContinuousBernoulli: pdf(0.5; lambda=0.5) = 1.0 (Uniform limit)" {
    const dist = try ContinuousBernoulli(f64).init(0.5);
    const p = dist.pdf(0.5);
    try testing.expectApproxEqAbs(@as(f64, 1.0), p, 1e-10);
}

test "ContinuousBernoulli: pdf(0.0; lambda=0.5) = 1.0" {
    const dist = try ContinuousBernoulli(f64).init(0.5);
    const p = dist.pdf(0.0);
    try testing.expectApproxEqAbs(@as(f64, 1.0), p, 1e-10);
}

test "ContinuousBernoulli: pdf(1.0; lambda=0.5) = 1.0" {
    const dist = try ContinuousBernoulli(f64).init(0.5);
    const p = dist.pdf(1.0);
    try testing.expectApproxEqAbs(@as(f64, 1.0), p, 1e-10);
}

test "ContinuousBernoulli: pdf(0.25; lambda=0.5) = 1.0" {
    const dist = try ContinuousBernoulli(f64).init(0.5);
    const p = dist.pdf(0.25);
    try testing.expectApproxEqAbs(@as(f64, 1.0), p, 1e-10);
}

test "ContinuousBernoulli: pdf(0.5; lambda=0.7) ≈ 0.9707" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    const p = dist.pdf(0.5);
    // C(0.7) = ln(7/3) / 0.4 ≈ 2.1183
    // pdf = 2.1183 * (0.7)^0.5 * (0.3)^0.5 ≈ 0.9707
    try testing.expectApproxEqAbs(@as(f64, 0.9707), p, 1e-4);
}

test "ContinuousBernoulli: pdf outside [0,1] returns 0" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    try testing.expectEqual(@as(f64, 0.0), dist.pdf(-0.1));
    try testing.expectEqual(@as(f64, 0.0), dist.pdf(-1.0));
    try testing.expectEqual(@as(f64, 0.0), dist.pdf(1.1));
    try testing.expectEqual(@as(f64, 0.0), dist.pdf(2.0));
}

test "ContinuousBernoulli: pdf is positive everywhere on [0,1]" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    try testing.expect(dist.pdf(0.0) > 0.0);
    try testing.expect(dist.pdf(0.25) > 0.0);
    try testing.expect(dist.pdf(0.5) > 0.0);
    try testing.expect(dist.pdf(0.75) > 0.0);
    try testing.expect(dist.pdf(1.0) > 0.0);
}

test "ContinuousBernoulli: pdf integrates to approximately 1 on [0,1] for lambda=0.7" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    var sum: f64 = 0.0;
    const dx = 0.001;
    var x: f64 = 0.0;
    while (x <= 1.0) : (x += dx) {
        sum += dist.pdf(x) * dx;
    }
    try testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-3);
}

test "ContinuousBernoulli: pdf integrates to approximately 1 on [0,1] for lambda=0.3" {
    const dist = try ContinuousBernoulli(f64).init(0.3);
    var sum: f64 = 0.0;
    const dx = 0.001;
    var x: f64 = 0.0;
    while (x <= 1.0) : (x += dx) {
        sum += dist.pdf(x) * dx;
    }
    try testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-3);
}

test "ContinuousBernoulli: logpdf(0.5; lambda=0.5) = 0.0" {
    const dist = try ContinuousBernoulli(f64).init(0.5);
    const lp = dist.logpdf(0.5);
    try testing.expectApproxEqAbs(@as(f64, 0.0), lp, 1e-10);
}

test "ContinuousBernoulli: logpdf outside [0,1] returns -infinity" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    try testing.expect(dist.logpdf(-0.1) == -math.inf(f64));
    try testing.expect(dist.logpdf(1.1) == -math.inf(f64));
}

test "ContinuousBernoulli: logpdf(0.5; lambda=0.7) ≈ ln(0.9707)" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    const lp = dist.logpdf(0.5);
    const p = dist.pdf(0.5);
    const expected = @log(p);
    try testing.expectApproxEqAbs(expected, lp, 1e-10);
}

test "ContinuousBernoulli: cdf(0.0; any lambda) = 0" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    try testing.expectEqual(@as(f64, 0.0), dist.cdf(0.0));
    const dist2 = try ContinuousBernoulli(f64).init(0.3);
    try testing.expectEqual(@as(f64, 0.0), dist2.cdf(0.0));
}

test "ContinuousBernoulli: cdf(1.0; any lambda) = 1" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    try testing.expectEqual(@as(f64, 1.0), dist.cdf(1.0));
    const dist2 = try ContinuousBernoulli(f64).init(0.3);
    try testing.expectEqual(@as(f64, 1.0), dist2.cdf(1.0));
}

test "ContinuousBernoulli: cdf(0.5; lambda=0.5) = 0.5 (Uniform)" {
    const dist = try ContinuousBernoulli(f64).init(0.5);
    const c = dist.cdf(0.5);
    try testing.expectApproxEqAbs(@as(f64, 0.5), c, 1e-10);
}

test "ContinuousBernoulli: cdf(0.5; lambda=0.7) ≈ 0.3957" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    const c = dist.cdf(0.5);
    // F(0.5; 0.7) = 0.3 * (1 - (7/3)^0.5) / (-0.4) ≈ 0.3957
    try testing.expectApproxEqAbs(@as(f64, 0.3957), c, 1e-4);
}

test "ContinuousBernoulli: cdf is monotone increasing" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    const c0 = dist.cdf(0.1);
    const c25 = dist.cdf(0.25);
    const c5 = dist.cdf(0.5);
    const c75 = dist.cdf(0.75);
    const c9 = dist.cdf(0.9);
    try testing.expect(c0 < c25);
    try testing.expect(c25 < c5);
    try testing.expect(c5 < c75);
    try testing.expect(c75 < c9);
}

test "ContinuousBernoulli: cdf below 0 returns 0" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    try testing.expectEqual(@as(f64, 0.0), dist.cdf(-1.0));
    try testing.expectEqual(@as(f64, 0.0), dist.cdf(-0.5));
}

test "ContinuousBernoulli: cdf above 1 returns 1" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    try testing.expectEqual(@as(f64, 1.0), dist.cdf(1.1));
    try testing.expectEqual(@as(f64, 1.0), dist.cdf(2.0));
}

test "ContinuousBernoulli: quantile(0; any lambda) = 0" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    const q = try dist.quantile(0.0);
    try testing.expectEqual(@as(f64, 0.0), q);
    const dist2 = try ContinuousBernoulli(f64).init(0.3);
    const q2 = try dist2.quantile(0.0);
    try testing.expectEqual(@as(f64, 0.0), q2);
}

test "ContinuousBernoulli: quantile(1; any lambda) = 1" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    const q = try dist.quantile(1.0);
    try testing.expectEqual(@as(f64, 1.0), q);
    const dist2 = try ContinuousBernoulli(f64).init(0.3);
    const q2 = try dist2.quantile(1.0);
    try testing.expectEqual(@as(f64, 1.0), q2);
}

test "ContinuousBernoulli: quantile(0.5; lambda=0.5) = 0.5 (Uniform)" {
    const dist = try ContinuousBernoulli(f64).init(0.5);
    const q = try dist.quantile(0.5);
    try testing.expectApproxEqAbs(@as(f64, 0.5), q, 1e-10);
}

test "ContinuousBernoulli: quantile fails for p=-0.1" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    try testing.expectError(error.InvalidProbability, dist.quantile(-0.1));
}

test "ContinuousBernoulli: quantile fails for p=1.1" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    try testing.expectError(error.InvalidProbability, dist.quantile(1.1));
}

test "ContinuousBernoulli: quantile fails for p=NaN" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    const p = math.nan(f64);
    try testing.expectError(error.InvalidProbability, dist.quantile(p));
}

test "ContinuousBernoulli: cdf(quantile(p)) ≈ p roundtrip for p=0.25 lambda=0.7" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    const p = 0.25;
    const q = try dist.quantile(p);
    const c = dist.cdf(q);
    try testing.expectApproxEqAbs(p, c, 1e-10);
}

test "ContinuousBernoulli: cdf(quantile(p)) ≈ p roundtrip for p=0.5 lambda=0.7" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    const p = 0.5;
    const q = try dist.quantile(p);
    const c = dist.cdf(q);
    try testing.expectApproxEqAbs(p, c, 1e-10);
}

test "ContinuousBernoulli: cdf(quantile(p)) ≈ p roundtrip for p=0.75 lambda=0.7" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    const p = 0.75;
    const q = try dist.quantile(p);
    const c = dist.cdf(q);
    try testing.expectApproxEqAbs(p, c, 1e-10);
}

test "ContinuousBernoulli: cdf(quantile(p)) ≈ p roundtrip for p=0.1 lambda=0.3" {
    const dist = try ContinuousBernoulli(f64).init(0.3);
    const p = 0.1;
    const q = try dist.quantile(p);
    const c = dist.cdf(q);
    try testing.expectApproxEqAbs(p, c, 1e-10);
}

test "ContinuousBernoulli: cdf(quantile(p)) ≈ p roundtrip for p=0.9 lambda=0.3" {
    const dist = try ContinuousBernoulli(f64).init(0.3);
    const p = 0.9;
    const q = try dist.quantile(p);
    const c = dist.cdf(q);
    try testing.expectApproxEqAbs(p, c, 1e-10);
}

test "ContinuousBernoulli: mean(lambda=0.5) = 0.5" {
    const dist = try ContinuousBernoulli(f64).init(0.5);
    const m = dist.mean();
    try testing.expectApproxEqAbs(@as(f64, 0.5), m, 1e-10);
}

test "ContinuousBernoulli: mean(lambda=0.7) ≈ 0.5698" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    const m = dist.mean();
    // mean = 0.7 / 0.4 - 1 / ln(7/3) ≈ 1.75 - 1.1802 ≈ 0.5698
    try testing.expectApproxEqAbs(@as(f64, 0.5698), m, 1e-4);
}

test "ContinuousBernoulli: mean(lambda=0.3) ≈ 0.4302" {
    const dist = try ContinuousBernoulli(f64).init(0.3);
    const m = dist.mean();
    // By symmetry: mean(0.3) = 1 - mean(0.7) ≈ 0.4302
    try testing.expectApproxEqAbs(@as(f64, 0.4302), m, 1e-4);
}

test "ContinuousBernoulli: mean(lambda) + mean(1-lambda) = 1 (symmetry)" {
    const dist1 = try ContinuousBernoulli(f64).init(0.7);
    const dist2 = try ContinuousBernoulli(f64).init(0.3);
    const m1 = dist1.mean();
    const m2 = dist2.mean();
    try testing.expectApproxEqAbs(@as(f64, 1.0), m1 + m2, 1e-10);
}

test "ContinuousBernoulli: mode(lambda=0.3) = 0" {
    const dist = try ContinuousBernoulli(f64).init(0.3);
    try testing.expectEqual(@as(f64, 0.0), dist.mode());
}

test "ContinuousBernoulli: mode(lambda=0.5) = 0.5" {
    const dist = try ContinuousBernoulli(f64).init(0.5);
    try testing.expectEqual(@as(f64, 0.5), dist.mode());
}

test "ContinuousBernoulli: mode(lambda=0.7) = 1" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    try testing.expectEqual(@as(f64, 1.0), dist.mode());
}

test "ContinuousBernoulli: variance(lambda=0.5) ≈ 1/12 ≈ 0.08333" {
    const dist = try ContinuousBernoulli(f64).init(0.5);
    const v = dist.variance();
    try testing.expectApproxEqAbs(1.0 / 12.0, v, 1e-4);
}

test "ContinuousBernoulli: variance is positive for all valid lambda" {
    const dist1 = try ContinuousBernoulli(f64).init(0.3);
    const dist2 = try ContinuousBernoulli(f64).init(0.5);
    const dist3 = try ContinuousBernoulli(f64).init(0.7);
    try testing.expect(dist1.variance() > 0.0);
    try testing.expect(dist2.variance() > 0.0);
    try testing.expect(dist3.variance() > 0.0);
}

test "ContinuousBernoulli: entropy(lambda=0.5) ≈ 0" {
    const dist = try ContinuousBernoulli(f64).init(0.5);
    const e = dist.entropy();
    try testing.expectApproxEqAbs(@as(f64, 0.0), e, 1e-10);
}

test "ContinuousBernoulli: entropy(lambda=0.7) ≈ -0.0296" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    const e = dist.entropy();
    try testing.expectApproxEqAbs(@as(f64, -0.0296), e, 1e-3);
}

test "ContinuousBernoulli: entropy is non-positive for all lambda" {
    for ([_]f64{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 }) |lam| {
        const dist = try ContinuousBernoulli(f64).init(lam);
        const e = dist.entropy();
        try testing.expect(e <= 0.0);
    }
}

test "ContinuousBernoulli: pdf is monotone increasing for lambda=0.7" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    const pdf0 = dist.pdf(0.0);
    const pdf25 = dist.pdf(0.25);
    const pdf5 = dist.pdf(0.5);
    const pdf75 = dist.pdf(0.75);
    const pdf1 = dist.pdf(1.0);
    try testing.expect(pdf0 < pdf25);
    try testing.expect(pdf25 < pdf5);
    try testing.expect(pdf5 < pdf75);
    try testing.expect(pdf75 < pdf1);
}

test "ContinuousBernoulli: pdf is monotone decreasing for lambda=0.3" {
    const dist = try ContinuousBernoulli(f64).init(0.3);
    const pdf0 = dist.pdf(0.0);
    const pdf25 = dist.pdf(0.25);
    const pdf5 = dist.pdf(0.5);
    const pdf75 = dist.pdf(0.75);
    const pdf1 = dist.pdf(1.0);
    try testing.expect(pdf0 > pdf25);
    try testing.expect(pdf25 > pdf5);
    try testing.expect(pdf5 > pdf75);
    try testing.expect(pdf75 > pdf1);
}

test "ContinuousBernoulli: validate succeeds for lambda=0.3" {
    const dist = try ContinuousBernoulli(f64).init(0.3);
    try dist.validate();
}

test "ContinuousBernoulli: validate succeeds for lambda=0.5" {
    const dist = try ContinuousBernoulli(f64).init(0.5);
    try dist.validate();
}

test "ContinuousBernoulli: validate succeeds for lambda=0.7" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    try dist.validate();
}

test "ContinuousBernoulli: validate fails for lambda=0" {
    var dist: ContinuousBernoulli(f64) = undefined;
    dist.lambda = 0.0;
    try testing.expectError(error.InvalidParameter, dist.validate());
}

test "ContinuousBernoulli: validate fails for lambda=1" {
    var dist: ContinuousBernoulli(f64) = undefined;
    dist.lambda = 1.0;
    try testing.expectError(error.InvalidParameter, dist.validate());
}

test "ContinuousBernoulli: validate fails for lambda=NaN" {
    var dist: ContinuousBernoulli(f64) = undefined;
    dist.lambda = math.nan(f64);
    try testing.expectError(error.InvalidParameter, dist.validate());
}

test "ContinuousBernoulli(f32): init succeeds" {
    const dist = try ContinuousBernoulli(f32).init(0.7);
    try testing.expectEqual(@as(f32, 0.7), dist.lambda);
}

test "ContinuousBernoulli(f32): pdf produces finite values" {
    const dist = try ContinuousBernoulli(f32).init(0.7);
    for ([_]f32{ 0.0, 0.25, 0.5, 0.75, 1.0 }) |x| {
        const p = dist.pdf(x);
        try testing.expect(math.isFinite(p));
    }
}

test "ContinuousBernoulli(f32): cdf in [0,1]" {
    const dist = try ContinuousBernoulli(f32).init(0.7);
    for ([_]f32{ 0.0, 0.25, 0.5, 0.75, 1.0 }) |x| {
        const c = dist.cdf(x);
        try testing.expect(c >= 0.0 and c <= 1.0);
    }
}

test "ContinuousBernoulli: sample produces values in [0,1]" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    var rng = std.Random.DefaultPrng.init(12345);
    for (0..100) |_| {
        const s = dist.sample(rng.random());
        try testing.expect(s >= 0.0 and s <= 1.0);
    }
}

test "ContinuousBernoulli: sample produces finite values" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    var rng = std.Random.DefaultPrng.init(54321);
    for (0..100) |_| {
        const s = dist.sample(rng.random());
        try testing.expect(math.isFinite(s));
    }
}

test "ContinuousBernoulli: sample mean converges to theoretical mean (N=5000, lambda=0.7)" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    var rng = std.Random.DefaultPrng.init(11111);
    var sum: f64 = 0.0;
    for (0..5000) |_| {
        sum += dist.sample(rng.random());
    }
    const sample_mean = sum / 5000.0;
    const theoretical_mean = dist.mean();
    try testing.expectApproxEqAbs(theoretical_mean, sample_mean, 0.05);
}

test "ContinuousBernoulli: sample mean converges to theoretical mean (N=5000, lambda=0.3)" {
    const dist = try ContinuousBernoulli(f64).init(0.3);
    var rng = std.Random.DefaultPrng.init(22222);
    var sum: f64 = 0.0;
    for (0..5000) |_| {
        sum += dist.sample(rng.random());
    }
    const sample_mean = sum / 5000.0;
    const theoretical_mean = dist.mean();
    try testing.expectApproxEqAbs(theoretical_mean, sample_mean, 0.05);
}

test "ContinuousBernoulli: quantile is monotone increasing for lambda=0.7" {
    const dist = try ContinuousBernoulli(f64).init(0.7);
    const q0 = try dist.quantile(0.0);
    const q25 = try dist.quantile(0.25);
    const q5 = try dist.quantile(0.5);
    const q75 = try dist.quantile(0.75);
    const q1 = try dist.quantile(1.0);
    try testing.expect(q0 <= q25);
    try testing.expect(q25 <= q5);
    try testing.expect(q5 <= q75);
    try testing.expect(q75 <= q1);
}

test "ContinuousBernoulli: as lambda→0.5 from below, entropy→0" {
    const dist = try ContinuousBernoulli(f64).init(0.49);
    const e = dist.entropy();
    try testing.expect(e > -0.01 and e < 0.0);
}

test "ContinuousBernoulli: as lambda→0.5 from above, entropy→0" {
    const dist = try ContinuousBernoulli(f64).init(0.51);
    const e = dist.entropy();
    try testing.expect(e > -0.01 and e < 0.0);
}
