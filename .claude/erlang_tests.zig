test "Erlang: init succeeds with valid parameters k=1, lambda=1.0" {
    const dist = try Erlang(f64).init(1, 1.0);
    try testing.expect(dist.k == 1);
    try testing.expectEqual(@as(f64, 1.0), dist.lambda);
}

test "Erlang: init succeeds with valid parameters k=5, lambda=2.0" {
    const dist = try Erlang(f64).init(5, 2.0);
    try testing.expect(dist.k == 5);
    try testing.expectEqual(@as(f64, 2.0), dist.lambda);
}

test "Erlang: init fails for k=0" {
    try testing.expectError(error.InvalidParameter, Erlang(f64).init(0, 1.0));
}

test "Erlang: init fails for lambda=0" {
    try testing.expectError(error.InvalidParameter, Erlang(f64).init(1, 0.0));
}

test "Erlang: init fails for negative lambda" {
    try testing.expectError(error.InvalidParameter, Erlang(f64).init(2, -1.5));
}

test "Erlang: init fails for infinite lambda" {
    try testing.expectError(error.InvalidParameter, Erlang(f64).init(2, math.inf(f64)));
}

test "Erlang: init fails for NaN lambda" {
    try testing.expectError(error.InvalidParameter, Erlang(f64).init(2, math.nan(f64)));
}

test "Erlang: pdf(x; k=1, lambda=1) matches Exponential formula at x=1" {
    const dist = try Erlang(f64).init(1, 1.0);
    const p = dist.pdf(1.0);
    const expected = 1.0 * @exp(-1.0);
    try testing.expectApproxEqAbs(expected, p, 1e-10);
}

test "Erlang: pdf(x; k=2, lambda=1) = x*exp(-x) at x=1" {
    const dist = try Erlang(f64).init(2, 1.0);
    const p = dist.pdf(1.0);
    const expected = 1.0 * @exp(-1.0);
    try testing.expectApproxEqAbs(expected, p, 1e-4);
}

test "Erlang: pdf(x; k=2, lambda=1) at x=2" {
    const dist = try Erlang(f64).init(2, 1.0);
    const p = dist.pdf(2.0);
    const expected = 2.0 * @exp(-2.0);
    try testing.expectApproxEqAbs(expected, p, 1e-4);
}

test "Erlang: pdf(x; k=3, lambda=2) at x=1" {
    const dist = try Erlang(f64).init(3, 2.0);
    const p = dist.pdf(1.0);
    // λ^k * x^(k-1) * exp(-λx) / (k-1)!
    // = 8 * 1 * exp(-2) / 2 = 4*exp(-2) ≈ 0.54134
    const expected = 4.0 * @exp(-2.0);
    try testing.expectApproxEqAbs(expected, p, 1e-4);
}

test "Erlang: pdf is zero for x<0" {
    const dist = try Erlang(f64).init(2, 1.0);
    try testing.expectEqual(@as(f64, 0.0), dist.pdf(-1.0));
    try testing.expectEqual(@as(f64, 0.0), dist.pdf(-10.0));
}

test "Erlang: pdf(0) = 0 for k>=2" {
    const dist = try Erlang(f64).init(2, 1.0);
    try testing.expectEqual(@as(f64, 0.0), dist.pdf(0.0));
}

test "Erlang: pdf(0) = lambda for k=1" {
    const dist = try Erlang(f64).init(1, 2.0);
    const p = dist.pdf(0.0);
    try testing.expectApproxEqAbs(@as(f64, 2.0), p, 1e-10);
}

test "Erlang: pdf is non-negative everywhere" {
    const dist = try Erlang(f64).init(3, 1.5);
    try testing.expect(dist.pdf(-5.0) >= 0.0);
    try testing.expect(dist.pdf(0.0) >= 0.0);
    try testing.expect(dist.pdf(1.0) >= 0.0);
    try testing.expect(dist.pdf(5.0) >= 0.0);
}

test "Erlang: logpdf(x; k=1, lambda=1) = log(lambda) - lambda*x at x=1" {
    const dist = try Erlang(f64).init(1, 1.0);
    const lp = dist.logpdf(1.0);
    const expected = 0.0 - 1.0;
    try testing.expectApproxEqAbs(expected, lp, 1e-10);
}

test "Erlang: logpdf is -infinity for x<=0 when k>=2" {
    const dist = try Erlang(f64).init(2, 1.0);
    try testing.expect(dist.logpdf(0.0) == -math.inf(f64));
    try testing.expect(dist.logpdf(-1.0) == -math.inf(f64));
}

test "Erlang: logpdf consistent with pdf at x=1" {
    const dist = try Erlang(f64).init(3, 2.0);
    const pdf_val = dist.pdf(1.0);
    const logpdf_val = dist.logpdf(1.0);
    try testing.expectApproxEqAbs(@log(pdf_val), logpdf_val, 1e-10);
}

test "Erlang: logpdf consistent with pdf at x=0.5" {
    const dist = try Erlang(f64).init(4, 1.5);
    const pdf_val = dist.pdf(0.5);
    const logpdf_val = dist.logpdf(0.5);
    try testing.expectApproxEqAbs(@log(pdf_val), logpdf_val, 1e-10);
}

test "Erlang: cdf(0; k=1, lambda=1) = 0" {
    const dist = try Erlang(f64).init(1, 1.0);
    try testing.expectEqual(@as(f64, 0.0), dist.cdf(0.0));
}

test "Erlang: cdf(x; k=1, lambda=1) = 1 - exp(-x) at x=1" {
    const dist = try Erlang(f64).init(1, 1.0);
    const c = dist.cdf(1.0);
    const expected = 1.0 - @exp(-1.0);
    try testing.expectApproxEqAbs(expected, c, 1e-4);
}

test "Erlang: cdf(1; k=2, lambda=1) ≈ 0.26424" {
    const dist = try Erlang(f64).init(2, 1.0);
    const c = dist.cdf(1.0);
    // 1 - exp(-1)*(1 + 1) = 1 - 2*exp(-1)
    const expected = 1.0 - 2.0 * @exp(-1.0);
    try testing.expectApproxEqAbs(expected, c, 1e-4);
}

test "Erlang: cdf(2; k=2, lambda=1) ≈ 0.59399" {
    const dist = try Erlang(f64).init(2, 1.0);
    const c = dist.cdf(2.0);
    // 1 - exp(-2)*(1 + 2) = 1 - 3*exp(-2)
    const expected = 1.0 - 3.0 * @exp(-2.0);
    try testing.expectApproxEqAbs(expected, c, 1e-4);
}

test "Erlang: cdf(1; k=3, lambda=2) ≈ 0.32332" {
    const dist = try Erlang(f64).init(3, 2.0);
    const c = dist.cdf(1.0);
    // 1 - exp(-2)*(1 + 2 + 2) = 1 - 5*exp(-2)
    const expected = 1.0 - 5.0 * @exp(-2.0);
    try testing.expectApproxEqAbs(expected, c, 1e-4);
}

test "Erlang: cdf(0.5; k=1, lambda=2) ≈ 0.63212" {
    const dist = try Erlang(f64).init(1, 2.0);
    const c = dist.cdf(0.5);
    const expected = 1.0 - @exp(-1.0);
    try testing.expectApproxEqAbs(expected, c, 1e-4);
}

test "Erlang: cdf is monotonically increasing" {
    const dist = try Erlang(f64).init(3, 1.0);
    const c0 = dist.cdf(0.0);
    const c1 = dist.cdf(1.0);
    const c2 = dist.cdf(2.0);
    const c5 = dist.cdf(5.0);
    try testing.expect(c0 <= c1);
    try testing.expect(c1 < c2);
    try testing.expect(c2 < c5);
}

test "Erlang: cdf approaches 1 at large x" {
    const dist = try Erlang(f64).init(2, 1.0);
    try testing.expect(dist.cdf(50.0) > 0.9999);
}

test "Erlang: sf(x) + cdf(x) = 1" {
    const dist = try Erlang(f64).init(3, 2.0);
    try testing.expectApproxEqAbs(1.0, dist.cdf(1.0) + dist.sf(1.0), 1e-10);
    try testing.expectApproxEqAbs(1.0, dist.cdf(0.5) + dist.sf(0.5), 1e-10);
}

test "Erlang: sf(0) = 1" {
    const dist = try Erlang(f64).init(2, 1.0);
    try testing.expectEqual(@as(f64, 1.0), dist.sf(0.0));
}

test "Erlang: sf approaches 0 at large x" {
    const dist = try Erlang(f64).init(2, 1.0);
    try testing.expect(dist.sf(50.0) < 1e-4);
}

test "Erlang: quantile(0) = 0" {
    const dist = try Erlang(f64).init(2, 1.0);
    const q = try dist.quantile(0.0);
    try testing.expectEqual(@as(f64, 0.0), q);
}

test "Erlang: quantile(1) = +infinity" {
    const dist = try Erlang(f64).init(2, 1.0);
    const q = try dist.quantile(1.0);
    try testing.expect(q == math.inf(f64));
}

test "Erlang: quantile error for p<0" {
    const dist = try Erlang(f64).init(2, 1.0);
    try testing.expectError(error.InvalidProbability, dist.quantile(-0.1));
    try testing.expectError(error.InvalidProbability, dist.quantile(-0.5));
}

test "Erlang: quantile error for p>1" {
    const dist = try Erlang(f64).init(2, 1.0);
    try testing.expectError(error.InvalidProbability, dist.quantile(1.1));
    try testing.expectError(error.InvalidProbability, dist.quantile(1.5));
}

test "Erlang: quantile error for NaN p" {
    const dist = try Erlang(f64).init(2, 1.0);
    try testing.expectError(error.InvalidProbability, dist.quantile(math.nan(f64)));
}

test "Erlang: quantile is monotonically increasing" {
    const dist = try Erlang(f64).init(3, 1.0);
    const q1 = try dist.quantile(0.25);
    const q2 = try dist.quantile(0.5);
    const q3 = try dist.quantile(0.75);
    try testing.expect(q1 < q2);
    try testing.expect(q2 < q3);
}

test "Erlang: quantile-cdf roundtrip at p=0.25" {
    const dist = try Erlang(f64).init(2, 1.0);
    const q = try dist.quantile(0.25);
    const c = dist.cdf(q);
    try testing.expectApproxEqAbs(@as(f64, 0.25), c, 1e-6);
}

test "Erlang: quantile-cdf roundtrip at p=0.5" {
    const dist = try Erlang(f64).init(3, 2.0);
    const q = try dist.quantile(0.5);
    const c = dist.cdf(q);
    try testing.expectApproxEqAbs(@as(f64, 0.5), c, 1e-6);
}

test "Erlang: quantile-cdf roundtrip at p=0.75" {
    const dist = try Erlang(f64).init(4, 1.5);
    const q = try dist.quantile(0.75);
    const c = dist.cdf(q);
    try testing.expectApproxEqAbs(@as(f64, 0.75), c, 1e-6);
}

test "Erlang: mean(k=1, lambda=1) = 1.0" {
    const dist = try Erlang(f64).init(1, 1.0);
    try testing.expectEqual(@as(f64, 1.0), dist.mean());
}

test "Erlang: mean(k=2, lambda=1) = 2.0" {
    const dist = try Erlang(f64).init(2, 1.0);
    try testing.expectEqual(@as(f64, 2.0), dist.mean());
}

test "Erlang: mean(k=3, lambda=2) = 1.5" {
    const dist = try Erlang(f64).init(3, 2.0);
    try testing.expectEqual(@as(f64, 1.5), dist.mean());
}

test "Erlang: mean(k=5, lambda=2) = 2.5" {
    const dist = try Erlang(f64).init(5, 2.0);
    try testing.expectEqual(@as(f64, 2.5), dist.mean());
}

test "Erlang: variance(k=1, lambda=1) = 1.0" {
    const dist = try Erlang(f64).init(1, 1.0);
    try testing.expectEqual(@as(f64, 1.0), dist.variance());
}

test "Erlang: variance(k=2, lambda=1) = 2.0" {
    const dist = try Erlang(f64).init(2, 1.0);
    try testing.expectEqual(@as(f64, 2.0), dist.variance());
}

test "Erlang: variance(k=3, lambda=2) = 0.75" {
    const dist = try Erlang(f64).init(3, 2.0);
    try testing.expectEqual(@as(f64, 0.75), dist.variance());
}

test "Erlang: variance(k=1, lambda=2) = 0.25" {
    const dist = try Erlang(f64).init(1, 2.0);
    try testing.expectEqual(@as(f64, 0.25), dist.variance());
}

test "Erlang: mode(k=1, lambda=1) = 0.0" {
    const dist = try Erlang(f64).init(1, 1.0);
    try testing.expectEqual(@as(f64, 0.0), dist.mode());
}

test "Erlang: mode(k=2, lambda=1) = 1.0" {
    const dist = try Erlang(f64).init(2, 1.0);
    try testing.expectEqual(@as(f64, 1.0), dist.mode());
}

test "Erlang: mode(k=3, lambda=2) = 1.0" {
    const dist = try Erlang(f64).init(3, 2.0);
    try testing.expectEqual(@as(f64, 1.0), dist.mode());
}

test "Erlang: mode(k=5, lambda=1) = 4.0" {
    const dist = try Erlang(f64).init(5, 1.0);
    try testing.expectEqual(@as(f64, 4.0), dist.mode());
}

test "Erlang: entropy is finite for valid parameters" {
    const dist = try Erlang(f64).init(2, 1.0);
    const h = dist.entropy();
    try testing.expect(math.isFinite(h));
}

test "Erlang: entropy(k=5, lambda=1) ≈ 2.478" {
    const dist = try Erlang(f64).init(5, 1.0);
    const h = dist.entropy();
    try testing.expectApproxEqAbs(@as(f64, 2.478), h, 0.05);
}

test "Erlang: entropy increases with scale parameter" {
    const dist1 = try Erlang(f64).init(3, 1.0);
    const dist2 = try Erlang(f64).init(3, 0.5);
    const h1 = dist1.entropy();
    const h2 = dist2.entropy();
    try testing.expect(h2 > h1);
}

test "Erlang: sample produces non-negative values" {
    const dist = try Erlang(f64).init(2, 1.0);
    var rng = std.Random.DefaultPrng.init(42);
    for (0..50) |_| {
        try testing.expect(dist.sample(rng.random()) >= 0.0);
    }
}

test "Erlang: sample produces finite values" {
    const dist = try Erlang(f64).init(3, 2.0);
    var rng = std.Random.DefaultPrng.init(123);
    for (0..50) |_| {
        try testing.expect(math.isFinite(dist.sample(rng.random())));
    }
}

test "Erlang: sample mean converges to theoretical mean (N=5000)" {
    const dist = try Erlang(f64).init(2, 1.0);
    var rng = std.Random.DefaultPrng.init(9999);
    var sum: f64 = 0.0;
    for (0..5000) |_| {
        sum += dist.sample(rng.random());
    }
    const sample_mean = sum / 5000.0;
    const theoretical_mean = dist.mean();
    try testing.expectApproxEqAbs(theoretical_mean, sample_mean, 0.15);
}

test "Erlang: sample mean converges for k=3, lambda=2 (N=5000)" {
    const dist = try Erlang(f64).init(3, 2.0);
    var rng = std.Random.DefaultPrng.init(54321);
    var sum: f64 = 0.0;
    for (0..5000) |_| {
        sum += dist.sample(rng.random());
    }
    const sample_mean = sum / 5000.0;
    const theoretical_mean = dist.mean();
    try testing.expectApproxEqAbs(theoretical_mean, sample_mean, 0.1);
}

test "Erlang: sample variance converges to theoretical variance (N=5000)" {
    const dist = try Erlang(f64).init(2, 1.0);
    var rng = std.Random.DefaultPrng.init(777);
    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    for (0..5000) |_| {
        const s = dist.sample(rng.random());
        sum += s;
        sum_sq += s * s;
    }
    const mean = sum / 5000.0;
    const sample_variance = sum_sq / 5000.0 - mean * mean;
    const theoretical_variance = dist.variance();
    try testing.expectApproxEqAbs(theoretical_variance, sample_variance, 0.3);
}

test "Erlang(f32): init and pdf with f32" {
    const dist = try Erlang(f32).init(2, 1.0);
    const p = dist.pdf(1.0);
    try testing.expect(math.isFinite(p));
    try testing.expect(p > 0.0);
}

test "Erlang(f32): cdf in [0,1] for f32" {
    const dist = try Erlang(f32).init(3, 2.0);
    const c = dist.cdf(1.0);
    try testing.expect(c >= 0.0 and c <= 1.0);
}

test "Erlang(f32): sample produces finite values" {
    const dist = try Erlang(f32).init(2, 1.0);
    var rng = std.Random.DefaultPrng.init(888);
    for (0..30) |_| {
        try testing.expect(math.isFinite(dist.sample(rng.random())));
    }
}

test "Erlang: validate() succeeds for valid parameters" {
    const dist = try Erlang(f64).init(2, 1.0);
    try dist.validate();
}

test "Erlang: validate() fails when k=0" {
    var dist: Erlang(f64) = undefined;
    dist.k = 0;
    dist.lambda = 1.0;
    try testing.expectError(error.InvalidParameter, dist.validate());
}

test "Erlang: validate() fails when lambda=0" {
    var dist: Erlang(f64) = undefined;
    dist.k = 2;
    dist.lambda = 0.0;
    try testing.expectError(error.InvalidParameter, dist.validate());
}

test "Erlang: validate() fails when lambda<0" {
    var dist: Erlang(f64) = undefined;
    dist.k = 2;
    dist.lambda = -1.0;
    try testing.expectError(error.InvalidParameter, dist.validate());
}

test "Erlang: validate() fails when lambda is infinite" {
    var dist: Erlang(f64) = undefined;
    dist.k = 2;
    dist.lambda = math.inf(f64);
    try testing.expectError(error.InvalidParameter, dist.validate());
}

test "Erlang: validate() fails when lambda is NaN" {
    var dist: Erlang(f64) = undefined;
    dist.k = 2;
    dist.lambda = math.nan(f64);
    try testing.expectError(error.InvalidParameter, dist.validate());
}

test "Erlang: k=1 reduces to Exponential(lambda)" {
    const dist = try Erlang(f64).init(1, 2.0);
    // Erlang(1, λ) should match Exponential(λ)
    // pdf(x; 1, λ) = λ * exp(-λx)
    const x = 0.5;
    const p = dist.pdf(x);
    const expected = 2.0 * @exp(-2.0 * x);
    try testing.expectApproxEqAbs(expected, p, 1e-10);
}

test "Erlang: large k produces approximately normal-like distribution" {
    const dist = try Erlang(f64).init(100, 1.0);
    // For large k, Erlang(k, λ) is approximately Normal(k/λ, k/λ²)
    const mean = dist.mean();
    try testing.expectApproxEqAbs(@as(f64, 100.0), mean, 1e-10);
    const variance = dist.variance();
    try testing.expectApproxEqAbs(@as(f64, 100.0), variance, 1e-10);
}

test "Erlang: pdf integrates to approximately 1 (numerical check)" {
    const dist = try Erlang(f64).init(2, 1.0);
    // Numerical integration using simple rectangular rule over support
    var sum: f64 = 0.0;
    const dx = 0.01;
    var x: f64 = 0.0;
    while (x < 20.0) : (x += dx) {
        sum += dist.pdf(x) * dx;
    }
    try testing.expectApproxEqAbs(@as(f64, 1.0), sum, 0.02);
}
