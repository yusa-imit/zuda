// Landau Distribution Tests
// ============================================================================

test "Landau: init valid mu=0 c=1" {
    const dist = try Landau(f64).init(0.0, 1.0);
    try expect(dist.mu == 0.0);
    try expect(dist.c == 1.0);
}

test "Landau: init valid mu=2 c=3" {
    const dist = try Landau(f64).init(2.0, 3.0);
    try expect(dist.mu == 2.0);
    try expect(dist.c == 3.0);
}

test "Landau: init valid negative mu=-5 c=0.5" {
    const dist = try Landau(f64).init(-5.0, 0.5);
    try expect(dist.mu == -5.0);
    try expect(dist.c == 0.5);
}

test "Landau: init fails for c=0" {
    try expectError(error.InvalidParameter, Landau(f64).init(0.0, 0.0));
}

test "Landau: init fails for c negative" {
    try expectError(error.InvalidParameter, Landau(f64).init(0.0, -1.0));
}

test "Landau: init fails for c NaN" {
    try expectError(error.InvalidParameter, Landau(f64).init(0.0, math.nan(f64)));
}

test "Landau: init fails for c Inf" {
    try expectError(error.InvalidParameter, Landau(f64).init(0.0, math.inf(f64)));
}

test "Landau: init fails for mu NaN" {
    try expectError(error.InvalidParameter, Landau(f64).init(math.nan(f64), 1.0));
}

test "Landau: init fails for mu Inf" {
    try expectError(error.InvalidParameter, Landau(f64).init(math.inf(f64), 1.0));
}

test "Landau: pdf peak near mode (-0.2224 for mu=0 c=1)" {
    const dist = try Landau(f64).init(0.0, 1.0);
    const pdf_at_mode = dist.pdf(-0.2224);
    try expectApproxEqAbs(0.1806, pdf_at_mode, 0.005);
}

test "Landau: pdf near-peak value at x=0 (mu=0 c=1)" {
    const dist = try Landau(f64).init(0.0, 1.0);
    const pdf_val = dist.pdf(0.0);
    try expectApproxEqAbs(0.161, pdf_val, 0.01);
}

test "Landau: pdf positive everywhere" {
    const dist = try Landau(f64).init(0.0, 1.0);
    try expect(dist.pdf(-10.0) > 0.0);
    try expect(dist.pdf(-0.2224) > 0.0);
    try expect(dist.pdf(0.0) > 0.0);
    try expect(dist.pdf(10.0) > 0.0);
    try expect(dist.pdf(100.0) > 0.0);
}

test "Landau: pdf decreases in right tail" {
    const dist = try Landau(f64).init(0.0, 1.0);
    const p1 = dist.pdf(5.0);
    const p2 = dist.pdf(10.0);
    const p3 = dist.pdf(20.0);
    try expect(p1 > p2);
    try expect(p2 > p3);
}

test "Landau: pdf approximate integral to 1 (midpoint rule)" {
    const dist = try Landau(f64).init(0.0, 1.0);
    var sum: f64 = 0.0;
    var x: f64 = -5.0;
    const dx = 65.0 / 300.0;
    var i: usize = 0;
    while (i < 300) : (i += 1) {
        sum += dist.pdf(x) * dx;
        x += dx;
    }
    try expectApproxEqAbs(1.0, sum, 0.02);
}

test "Landau: pdf scale invariance f(x; mu, c) = (1/c)*f((x-mu)/c; 0, 1)" {
    const dist1 = try Landau(f64).init(0.0, 1.0);
    const dist2 = try Landau(f64).init(2.0, 3.0);
    const x = 5.0;
    const pdf1 = dist1.pdf((x - 2.0) / 3.0);
    const pdf2 = dist2.pdf(x) * 3.0;
    try expectApproxEqAbs(pdf1, pdf2, 1e-10);
}

test "Landau: logpdf = log(pdf)" {
    const dist = try Landau(f64).init(0.0, 1.0);
    const x = 1.5;
    const pdf_val = dist.pdf(x);
    const logpdf_val = dist.logpdf(x);
    try expectApproxEqAbs(@log(pdf_val), logpdf_val, 1e-10);
}

test "Landau: logpdf several points" {
    const dist = try Landau(f64).init(0.0, 1.0);
    const xs = [_]f64{ -0.2224, 0.0, 1.0, 5.0 };
    for (xs) |x| {
        const expected = @log(dist.pdf(x));
        const actual = dist.logpdf(x);
        try expectApproxEqAbs(expected, actual, 1e-10);
    }
}

test "Landau: cdf left tail approaches 0" {
    const dist = try Landau(f64).init(0.0, 1.0);
    const cdf_neg5 = dist.cdf(-5.0);
    try expect(cdf_neg5 < 0.001);
}

test "Landau: cdf right tail approaches 1" {
    const dist = try Landau(f64).init(0.0, 1.0);
    const cdf_pos30 = dist.cdf(30.0);
    try expect(cdf_pos30 > 0.995);
}

test "Landau: cdf monotonically non-decreasing" {
    const dist = try Landau(f64).init(0.0, 1.0);
    const xs = [_]f64{ -5.0, -1.0, -0.2224, 0.0, 1.0, 5.0, 10.0 };
    var i: usize = 0;
    while (i + 1 < xs.len) : (i += 1) {
        const cdf1 = dist.cdf(xs[i]);
        const cdf2 = dist.cdf(xs[i + 1]);
        try expect(cdf1 <= cdf2);
    }
}

test "Landau: cdf at mode is less than 0.5 (right-skewed)" {
    const dist = try Landau(f64).init(0.0, 1.0);
    const mode = dist.mode();
    const cdf_at_mode = dist.cdf(mode);
    try expect(cdf_at_mode < 0.5);
}

test "Landau: sf = 1 - cdf" {
    const dist = try Landau(f64).init(0.0, 1.0);
    const x = 2.0;
    const sf = dist.sf(x);
    const cdf = dist.cdf(x);
    try expectApproxEqAbs(1.0, sf + cdf, 1e-10);
}

test "Landau: sf several points" {
    const dist = try Landau(f64).init(0.0, 1.0);
    const xs = [_]f64{ -5.0, 0.0, 1.0, 10.0 };
    for (xs) |x| {
        const sf_val = dist.sf(x);
        const cdf_val = dist.cdf(x);
        try expectApproxEqAbs(1.0, sf_val + cdf_val, 1e-10);
    }
}

test "Landau: mode mu=0 c=1 approximately -0.2224" {
    const dist = try Landau(f64).init(0.0, 1.0);
    const m = dist.mode();
    try expectApproxEqAbs(-0.2224, m, 0.05);
}

test "Landau: mode mu=2 c=3 approximately 2 + 3*(-0.2224) = 1.3328" {
    const dist = try Landau(f64).init(2.0, 3.0);
    const m = dist.mode();
    const expected = 2.0 + 3.0 * (-0.2224);
    try expectApproxEqAbs(expected, m, 0.1);
}

test "Landau: mode location parameter shift" {
    const dist1 = try Landau(f64).init(0.0, 1.0);
    const dist2 = try Landau(f64).init(5.0, 1.0);
    const m1 = dist1.mode();
    const m2 = dist2.mode();
    try expectApproxEqAbs(m1 + 5.0, m2, 1e-10);
}

test "Landau: mean returns positive infinity" {
    const dist = try Landau(f64).init(0.0, 1.0);
    const m = dist.mean();
    try expect(math.isPositiveInf(m));
}

test "Landau: variance returns positive infinity" {
    const dist = try Landau(f64).init(0.0, 1.0);
    const v = dist.variance();
    try expect(math.isPositiveInf(v));
}

test "Landau: entropy finite and positive" {
    const dist = try Landau(f64).init(0.0, 1.0);
    const e = dist.entropy();
    try expect(e > 0.0);
    try expect(math.isFinite(e));
}

test "Landau: entropy scaling law H(c) ~ H(1) + log(c)" {
    const dist1 = try Landau(f64).init(0.0, 1.0);
    const dist2 = try Landau(f64).init(0.0, 2.0);
    const e1 = dist1.entropy();
    const e2 = dist2.entropy();
    try expect(e2 > e1);
    const diff = e2 - e1;
    try expectApproxEqAbs(@log(2.0), diff, 0.2);
}

test "Landau: quantile invalid p < 0" {
    const dist = try Landau(f64).init(0.0, 1.0);
    try expectError(error.InvalidProbability, dist.quantile(-0.1));
}

test "Landau: quantile invalid p > 1" {
    const dist = try Landau(f64).init(0.0, 1.0);
    try expectError(error.InvalidProbability, dist.quantile(1.1));
}

test "Landau: quantile invalid p NaN" {
    const dist = try Landau(f64).init(0.0, 1.0);
    try expectError(error.InvalidProbability, dist.quantile(math.nan(f64)));
}

test "Landau: quantile boundary p=0 returns very negative value" {
    const dist = try Landau(f64).init(0.0, 1.0);
    const q = try dist.quantile(0.0);
    try expect(q < -5.0);
}

test "Landau: quantile boundary p=1 returns very large value" {
    const dist = try Landau(f64).init(0.0, 1.0);
    const q = try dist.quantile(1.0);
    try expect(q > 30.0);
}

test "Landau: quantile monotonic increasing" {
    const dist = try Landau(f64).init(0.0, 1.0);
    const ps = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    var i: usize = 0;
    while (i + 1 < ps.len) : (i += 1) {
        const q1 = try dist.quantile(ps[i]);
        const q2 = try dist.quantile(ps[i + 1]);
        try expect(q1 < q2);
    }
}

test "Landau: quantile roundtrip cdf(quantile(p)) >= p" {
    const dist = try Landau(f64).init(0.0, 1.0);
    const ps = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    for (ps) |p| {
        const q = try dist.quantile(p);
        const cdf_val = dist.cdf(q);
        try expect(cdf_val >= p - 0.01);
    }
}

test "Landau: quantile scale invariance" {
    const dist1 = try Landau(f64).init(0.0, 1.0);
    const dist2 = try Landau(f64).init(2.0, 3.0);
    const p = 0.5;
    const q1 = try dist1.quantile(p);
    const q2 = try dist2.quantile(p);
    const expected = 2.0 + 3.0 * q1;
    try expectApproxEqAbs(expected, q2, 0.01);
}

test "Landau: validate passes for mu=0 c=1" {
    const dist = try Landau(f64).init(0.0, 1.0);
    try dist.validate();
}

test "Landau: validate passes for mu=5 c=2" {
    const dist = try Landau(f64).init(5.0, 2.0);
    try dist.validate();
}

test "Landau: validate fails for c=0" {
    var dist = try Landau(f64).init(0.0, 1.0);
    dist.c = 0.0;
    try expectError(error.InvalidParameter, dist.validate());
}

test "Landau: validate fails for c negative" {
    var dist = try Landau(f64).init(0.0, 1.0);
    dist.c = -1.0;
    try expectError(error.InvalidParameter, dist.validate());
}

test "Landau: validate fails for c NaN" {
    var dist = try Landau(f64).init(0.0, 1.0);
    dist.c = math.nan(f64);
    try expectError(error.InvalidParameter, dist.validate());
}

test "Landau: validate fails for mu NaN" {
    var dist = try Landau(f64).init(0.0, 1.0);
    dist.mu = math.nan(f64);
    try expectError(error.InvalidParameter, dist.validate());
}

test "Landau: sample returns finite f64" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const dist = try Landau(f64).init(0.0, 1.0);
    const s = dist.sample(rng);
    try expect(math.isFinite(s));
}

test "Landau: sample from different seeds gives different values" {
    var prng1 = std.Random.DefaultPrng.init(42);
    const rng1 = prng1.random();
    var prng2 = std.Random.DefaultPrng.init(123);
    const rng2 = prng2.random();
    const dist = try Landau(f64).init(0.0, 1.0);
    const s1 = dist.sample(rng1);
    const s2 = dist.sample(rng2);
    try expect(s1 != s2);
}

test "Landau: sample empirical mean closer to median than mode" {
    var prng = std.Random.DefaultPrng.init(54321);
    const rng = prng.random();
    const dist = try Landau(f64).init(0.0, 1.0);
    var sum: f64 = 0.0;
    const n = 5000;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        sum += dist.sample(rng);
    }
    const empirical_mean = sum / @as(f64, n);
    const mode = dist.mode();
    const median_approx = try dist.quantile(0.5);
    try expect(@abs(empirical_mean - median_approx) < @abs(empirical_mean - mode));
}

test "Landau: f32 type support" {
    const dist = try Landau(f32).init(0.0, 1.0);
    try expect(dist.pdf(0.0) > 0.0);
    try expect(dist.cdf(0.0) > 0.0);
    const q = try dist.quantile(0.5);
    try expect(math.isFinite(q));
    const m = dist.mode();
    try expect(math.isFinite(m));
    try expect(math.isPositiveInf(dist.mean()));
    try expect(math.isPositiveInf(dist.variance()));
    try expect(dist.entropy() > 0.0);
    try dist.validate();
}

test "Landau: format output contains mu and c" {
    var buf: [256]u8 = undefined;
    const dist = try Landau(f64).init(2.0, 1.5);
    var stream = std.io.fixedBufferStream(&buf);
    try stream.writer().print("{}", .{dist});
    const output = stream.getWritten();
    try expect(std.mem.indexOf(u8, output, "Landau") != null);
}
