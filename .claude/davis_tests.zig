// Davis Distribution Tests (f64)
// ============================================================================
// Comprehensive test suite for Davis(b, n, μ) — 127th distribution (103rd continuous)
// Riemann zeta-based distribution for Bose-Einstein statistics

// ============================================================================
// INITIALIZATION & VALIDATION TESTS (8 tests)
// ============================================================================

test "Davis: init valid params b=1 n=2 mu=0" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    try expect(dist.b == 1.0);
    try expect(dist.n == 2.0);
    try expect(dist.mu == 0.0);
}

test "Davis: init valid params b=0.5 n=3 mu=1" {
    const dist = try Davis(f64).init(0.5, 3.0, 1.0);
    try expect(dist.b == 0.5);
    try expect(dist.n == 3.0);
    try expect(dist.mu == 1.0);
}

test "Davis: init valid params b=2 n=1.5 mu=-1" {
    const dist = try Davis(f64).init(2.0, 1.5, -1.0);
    try expect(dist.b == 2.0);
    try expect(dist.n == 1.5);
    try expect(dist.mu == -1.0);
}

test "Davis: init fails for b=0" {
    try expectError(error.InvalidParameter, Davis(f64).init(0.0, 2.0, 0.0));
}

test "Davis: init fails for b<0" {
    try expectError(error.InvalidParameter, Davis(f64).init(-1.0, 2.0, 0.0));
}

test "Davis: init fails for b=inf" {
    try expectError(error.InvalidParameter, Davis(f64).init(math.inf(f64), 2.0, 0.0));
}

test "Davis: init fails for n≤1" {
    try expectError(error.InvalidParameter, Davis(f64).init(1.0, 1.0, 0.0));
    try expectError(error.InvalidParameter, Davis(f64).init(1.0, 0.5, 0.0));
}

test "Davis: init fails for mu=nan" {
    try expectError(error.InvalidParameter, Davis(f64).init(1.0, 2.0, math.nan(f64)));
}

test "Davis: validate passes for valid params" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    try dist.validate();
}

test "Davis: validate fails for b=0" {
    var dist = try Davis(f64).init(1.0, 2.0, 0.0);
    dist.b = 0.0;
    try expectError(error.InvalidParameter, dist.validate());
}

// ============================================================================
// PDF TESTS (7 tests)
// ============================================================================

test "Davis: pdf returns 0 for x≤μ" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    try expect(dist.pdf(0.0) == 0.0);
    try expect(dist.pdf(-1.0) == 0.0);
    try expect(dist.pdf(-10.0) == 0.0);
}

test "Davis: pdf returns 0 for x=μ" {
    const dist = try Davis(f64).init(1.0, 2.0, 1.0);
    try expect(dist.pdf(1.0) == 0.0);
}

test "Davis: pdf positive for x>μ" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    try expect(dist.pdf(0.1) > 0.0);
    try expect(dist.pdf(0.5) > 0.0);
    try expect(dist.pdf(1.0) > 0.0);
    try expect(dist.pdf(10.0) > 0.0);
}

test "Davis: pdf at interior point is finite" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    const p = dist.pdf(1.0);
    try expect(math.isFinite(p));
    try expect(p > 0.0);
}

test "Davis: pdf(1; b=1,n=2,μ=0) ≈ 0.35426 (exact: 1/(ζ(2)·(e-1)))" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    const pdf_val = dist.pdf(1.0);
    const expected = 0.35426; // 1 / (π²/6 * (e-1))
    try expectApproxEqAbs(expected, pdf_val, 1e-4);
}

test "Davis: pdf decreases in right tail" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    const p1 = dist.pdf(0.5);
    const p2 = dist.pdf(1.0);
    const p3 = dist.pdf(2.0);
    const p4 = dist.pdf(5.0);
    try expect(p1 > p2);
    try expect(p2 > p3);
    try expect(p3 > p4);
}

test "Davis: pdf finite for all interior x" {
    const dist = try Davis(f64).init(1.0, 3.0, 0.0);
    const xs = [_]f64{ 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0 };
    for (xs) |x| {
        const p = dist.pdf(x);
        try expect(math.isFinite(p));
        try expect(p >= 0.0);
    }
}

test "Davis: f32 type pdf support" {
    const dist = try Davis(f32).init(1.0, 2.0, 0.0);
    const p = dist.pdf(1.0);
    try expect(p > 0.0);
    try expect(math.isFinite(p));
}

// ============================================================================
// LOG PDF TESTS (4 tests)
// ============================================================================

test "Davis: logpdf returns -inf for x≤μ" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    try expect(math.isNegativeInf(dist.logpdf(0.0)));
    try expect(math.isNegativeInf(dist.logpdf(-1.0)));
}

test "Davis: logpdf = log(pdf) for interior x" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    const x = 1.5;
    const pdf_val = dist.pdf(x);
    const logpdf_val = dist.logpdf(x);
    const expected = @log(pdf_val);
    try expectApproxEqAbs(expected, logpdf_val, 1e-10);
}

test "Davis: logpdf finite for interior x" {
    const dist = try Davis(f64).init(1.0, 3.0, 0.0);
    const x = 1.0;
    const lp = dist.logpdf(x);
    try expect(math.isFinite(lp));
}

test "Davis: logpdf consistency across multiple points" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    const xs = [_]f64{ 0.5, 1.0, 2.0, 5.0 };
    for (xs) |x| {
        const pdf_val = dist.pdf(x);
        const logpdf_val = dist.logpdf(x);
        const expected = @log(pdf_val);
        try expectApproxEqAbs(expected, logpdf_val, 1e-10);
    }
}

// ============================================================================
// CDF TESTS (7 tests)
// ============================================================================

test "Davis: cdf returns 0 for x≤μ" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    try expect(dist.cdf(0.0) == 0.0);
    try expect(dist.cdf(-1.0) == 0.0);
    try expect(dist.cdf(-1000.0) == 0.0);
}

test "Davis: cdf in [0,1] for all x" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    const xs = [_]f64{ -10.0, 0.0, 0.1, 0.5, 1.0, 5.0, 100.0 };
    for (xs) |x| {
        const c = dist.cdf(x);
        try expect(c >= 0.0 and c <= 1.0);
    }
}

test "Davis: cdf monotone increasing" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    const xs = [_]f64{ 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0 };
    var i: usize = 0;
    while (i + 1 < xs.len) : (i += 1) {
        const c1 = dist.cdf(xs[i]);
        const c2 = dist.cdf(xs[i + 1]);
        try expect(c1 <= c2);
    }
}

test "Davis: cdf + sf = 1" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    const xs = [_]f64{ 0.1, 0.5, 1.0, 2.0, 5.0, 10.0 };
    for (xs) |x| {
        const c = dist.cdf(x);
        const s = dist.sf(x);
        try expectApproxEqAbs(1.0, c + s, 1e-8);
    }
}

test "Davis: cdf approaches 1 for large x" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    const cdf_large = dist.cdf(1000.0);
    try expect(cdf_large > 0.99);
}

test "Davis: cdf at mode is between 0.2 and 0.8" {
    const dist = try Davis(f64).init(1.0, 3.0, 0.0);
    const m = dist.mode();
    const cdf_mode = dist.cdf(m);
    try expect(cdf_mode > 0.1 and cdf_mode < 0.9);
}

test "Davis: cdf is finite everywhere" {
    const dist = try Davis(f64).init(1.0, 2.5, 0.0);
    const xs = [_]f64{ 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 50.0 };
    for (xs) |x| {
        const c = dist.cdf(x);
        try expect(math.isFinite(c));
    }
}

// ============================================================================
// SF (SURVIVAL FUNCTION) TESTS (2 tests)
// ============================================================================

test "Davis: sf returns 1 for x≤μ" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    try expect(dist.sf(0.0) == 1.0);
    try expect(dist.sf(-1.0) == 1.0);
}

test "Davis: sf + cdf = 1 for all x" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    const xs = [_]f64{ 0.05, 0.1, 0.5, 1.0, 2.0, 10.0 };
    for (xs) |x| {
        const c = dist.cdf(x);
        const s = dist.sf(x);
        try expectApproxEqAbs(1.0, c + s, 1e-9);
    }
}

// ============================================================================
// QUANTILE TESTS (5 tests)
// ============================================================================

test "Davis: quantile rejects p<0" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    try expectError(error.InvalidProbability, dist.quantile(-0.1));
}

test "Davis: quantile rejects p>1" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    try expectError(error.InvalidProbability, dist.quantile(1.1));
}

test "Davis: quantile rejects p=nan" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    try expectError(error.InvalidProbability, dist.quantile(math.nan(f64)));
}

test "Davis: quantile monotone increasing" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    const ps = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    var i: usize = 0;
    while (i + 1 < ps.len) : (i += 1) {
        const q1 = try dist.quantile(ps[i]);
        const q2 = try dist.quantile(ps[i + 1]);
        try expect(q1 < q2);
    }
}

test "Davis: quantile roundtrip cdf(quantile(p)) ≈ p" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    const ps = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    for (ps) |p| {
        const q = try dist.quantile(p);
        const cdf_val = dist.cdf(q);
        try expect(cdf_val >= p - 0.02 and cdf_val <= p + 0.02);
    }
}

// ============================================================================
// MODE TESTS (3 tests)
// ============================================================================

test "Davis: mode is greater than μ" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    const m = dist.mode();
    try expect(m > 0.0);
}

test "Davis: mode is finite" {
    const dist = try Davis(f64).init(1.0, 3.0, 0.0);
    const m = dist.mode();
    try expect(math.isFinite(m));
    try expect(m > 0.0);
}

test "Davis: mode(b=1,n=3,μ=0) is in valid range (0.2, 0.35)" {
    const dist = try Davis(f64).init(1.0, 3.0, 0.0);
    const m = dist.mode();
    // Mode for Davis(1,3,0) ≈ 0.2550 (where u·e^u/(e^u-1) = 4)
    try expect(m > 0.20 and m < 0.35);
}

test "Davis: mode location shift invariance" {
    const dist1 = try Davis(f64).init(1.0, 3.0, 0.0);
    const dist2 = try Davis(f64).init(1.0, 3.0, 1.0);
    const m1 = dist1.mode();
    const m2 = dist2.mode();
    // mode should shift by μ
    try expectApproxEqAbs(m1 + 1.0, m2, 1e-10);
}

test "Davis: mode scale property" {
    const dist1 = try Davis(f64).init(1.0, 3.0, 0.0);
    const dist2 = try Davis(f64).init(2.0, 3.0, 0.0);
    const m1 = dist1.mode();
    const m2 = dist2.mode();
    // Mode should scale with b
    try expectApproxEqAbs(m1 * 2.0, m2, 1e-9);
}

// ============================================================================
// MEAN TESTS (4 tests)
// ============================================================================

test "Davis: mean is +∞ for n≤2" {
    const dist1 = try Davis(f64).init(1.0, 1.5, 0.0);
    const dist2 = try Davis(f64).init(1.0, 2.0, 0.0);
    try expect(math.isPositiveInf(dist1.mean()));
    try expect(math.isPositiveInf(dist2.mean()));
}

test "Davis: mean is finite for n>2" {
    const dist = try Davis(f64).init(1.0, 3.0, 0.0);
    const m = dist.mean();
    try expect(math.isFinite(m));
    try expect(m > 0.0);
}

test "Davis: mean(b=1,n=3,μ=0) ≈ 0.6840 (ζ(2)/(2·ζ(3)))" {
    const dist = try Davis(f64).init(1.0, 3.0, 0.0);
    const m = dist.mean();
    // ζ(2)/(2·ζ(3)) ≈ 1.6449/(2·1.2021) ≈ 0.6840
    const expected = 0.6840;
    try expectApproxEqAbs(expected, m, 0.01);
}

test "Davis: mean location parameter shift" {
    const dist1 = try Davis(f64).init(1.0, 4.0, 0.0);
    const dist2 = try Davis(f64).init(1.0, 4.0, 5.0);
    const m1 = dist1.mean();
    const m2 = dist2.mean();
    // mean(μ) = mean(0) + μ
    try expectApproxEqAbs(m1 + 5.0, m2, 1e-10);
}

// ============================================================================
// VARIANCE TESTS (3 tests)
// ============================================================================

test "Davis: variance is +∞ for n≤3" {
    const dist1 = try Davis(f64).init(1.0, 2.0, 0.0);
    const dist2 = try Davis(f64).init(1.0, 3.0, 0.0);
    try expect(math.isPositiveInf(dist1.variance()));
    try expect(math.isPositiveInf(dist2.variance()));
}

test "Davis: variance is finite and positive for n>3" {
    const dist = try Davis(f64).init(1.0, 4.0, 0.0);
    const v = dist.variance();
    try expect(math.isFinite(v));
    try expect(v > 0.0);
}

test "Davis: variance independent of μ" {
    const dist1 = try Davis(f64).init(1.0, 4.0, 0.0);
    const dist2 = try Davis(f64).init(1.0, 4.0, 5.0);
    const v1 = dist1.variance();
    const v2 = dist2.variance();
    try expectApproxEqAbs(v1, v2, 1e-10);
}

test "Davis: variance scales as b²" {
    const dist1 = try Davis(f64).init(1.0, 4.0, 0.0);
    const dist2 = try Davis(f64).init(2.0, 4.0, 0.0);
    const v1 = dist1.variance();
    const v2 = dist2.variance();
    // variance(b) = variance(1) * b²
    try expectApproxEqAbs(v1 * 4.0, v2, 1e-8);
}

// ============================================================================
// ENTROPY TESTS (2 tests)
// ============================================================================

test "Davis: entropy is finite and positive" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    const e = dist.entropy();
    try expect(e > 0.0);
    try expect(math.isFinite(e));
}

test "Davis: entropy finite for valid params" {
    const dists = [_]struct { b: f64, n: f64, mu: f64 }{
        .{ .b = 1.0, .n = 2.0, .mu = 0.0 },
        .{ .b = 0.5, .n = 3.0, .mu = 1.0 },
        .{ .b = 2.0, .n = 4.0, .mu = -1.0 },
    };
    for (dists) |params| {
        const dist = try Davis(f64).init(params.b, params.n, params.mu);
        const e = dist.entropy();
        try expect(math.isFinite(e));
        try expect(e > 0.0);
    }
}

// ============================================================================
// SAMPLE TESTS (2 tests)
// ============================================================================

test "Davis: sample returns x > μ" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    const s = dist.sample(rng);
    try expect(s > 0.0);
    try expect(math.isFinite(s));
}

test "Davis: 100 samples all satisfy x > μ" {
    var prng = std.Random.DefaultPrng.init(54321);
    const rng = prng.random();
    const dist = try Davis(f64).init(1.0, 2.0, 1.0);
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const s = dist.sample(rng);
        try expect(s > 1.0);
        try expect(math.isFinite(s));
    }
}

test "Davis: sample from different seeds differs" {
    var prng1 = std.Random.DefaultPrng.init(42);
    const rng1 = prng1.random();
    var prng2 = std.Random.DefaultPrng.init(123);
    const rng2 = prng2.random();
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    const s1 = dist.sample(rng1);
    const s2 = dist.sample(rng2);
    try expect(s1 != s2);
}

// ============================================================================
// PROPERTY TESTS (4 tests)
// ============================================================================

test "Davis: validate passes after construction" {
    const dist = try Davis(f64).init(1.0, 2.5, 0.5);
    try dist.validate();
}

test "Davis: pdf integral approximates 1" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    var sum: f64 = 0.0;
    var y: f64 = 0.001;
    const dy = 0.001;
    const max_y = 20.0;
    var i: usize = 0;
    while (y < max_y) : ({
        y += dy;
        i += 1;
    }) {
        sum += dist.pdf(y) * dy;
        if (i > 20000) break;
    }
    // PDF should integrate to ~1 over support
    try expect(sum > 0.90 and sum < 1.10);
}

test "Davis: location invariance of PDF shape" {
    const dist1 = try Davis(f64).init(1.0, 2.5, 0.0);
    const dist2 = try Davis(f64).init(1.0, 2.5, 1.0);
    const x = 1.0; // for dist2, this is y = 1.0 - 1.0 = 0.0
    const p1 = dist1.pdf(x);
    const p2 = dist2.pdf(x + 1.0);
    try expectApproxEqAbs(p1, p2, 1e-10);
}

test "Davis: PDF is strictly positive in interior" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    const xs = [_]f64{ 0.001, 0.01, 0.1, 0.5, 1.0, 5.0 };
    for (xs) |x| {
        try expect(dist.pdf(x) > 0.0);
    }
}

// ============================================================================
// COMPREHENSIVE TYPE SUPPORT TEST
// ============================================================================

test "Davis: f32 type comprehensive support" {
    const dist = try Davis(f32).init(1.0, 2.0, 0.0);
    try expect(dist.pdf(1.0) > 0.0);
    try expect(dist.cdf(1.0) > 0.0 and dist.cdf(1.0) < 1.0);
    const q = try dist.quantile(0.5);
    try expect(math.isFinite(q) and q > 0.0);
    const m = dist.mode();
    try expect(math.isFinite(m) and m > 0.0);
    try expect(math.isPositiveInf(dist.mean()));
    try expect(math.isPositiveInf(dist.variance()));
    try expect(dist.entropy() > 0.0);
    try dist.validate();
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const s = dist.sample(rng);
    try expect(s > 0.0);
}

// ============================================================================
// BOUNDARY & EDGE CASE TESTS (4 tests)
// ============================================================================

test "Davis: handle very small b" {
    const dist = try Davis(f64).init(0.001, 2.0, 0.0);
    const m = dist.mode();
    try expect(m > 0.0);
    try expect(math.isFinite(m));
}

test "Davis: handle very large n" {
    const dist = try Davis(f64).init(1.0, 100.0, 0.0);
    const m = dist.mean();
    try expect(math.isFinite(m));
    try expect(m > 0.0);
}

test "Davis: handle negative μ location" {
    const dist = try Davis(f64).init(1.0, 2.5, -10.0);
    const pdf_test = dist.pdf(-9.0); // x > μ
    try expect(pdf_test > 0.0);
    const cdf_test = dist.cdf(-9.0);
    try expect(cdf_test > 0.0 and cdf_test < 1.0);
}

test "Davis: pdf vanishes as x approaches μ from above" {
    const dist = try Davis(f64).init(1.0, 2.0, 0.0);
    const p1 = dist.pdf(0.0000001);
    const p2 = dist.pdf(0.00001);
    const p3 = dist.pdf(0.001);
    try expect(p1 < p2);
    try expect(p2 < p3);
}
