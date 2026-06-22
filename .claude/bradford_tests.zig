// Bradford Distribution Tests
// Place these tests after the Erlang tests in distributions.zig

test "Bradford: init succeeds with c=1.0" {
    const dist = try Bradford(f64).init(1.0);
    try testing.expectEqual(@as(f64, 1.0), dist.c);
}

test "Bradford: init succeeds with c=0.5" {
    const dist = try Bradford(f64).init(0.5);
    try testing.expectEqual(@as(f64, 0.5), dist.c);
}

test "Bradford: init succeeds with c=2.0" {
    const dist = try Bradford(f64).init(2.0);
    try testing.expectEqual(@as(f64, 2.0), dist.c);
}

test "Bradford: init fails for c=0" {
    try testing.expectError(error.InvalidParameter, Bradford(f64).init(0.0));
}

test "Bradford: init fails for c=-1.0" {
    try testing.expectError(error.InvalidParameter, Bradford(f64).init(-1.0));
}

test "Bradford: init fails for infinite c" {
    try testing.expectError(error.InvalidParameter, Bradford(f64).init(math.inf(f64)));
}

test "Bradford: init fails for NaN c" {
    try testing.expectError(error.InvalidParameter, Bradford(f64).init(math.nan(f64)));
}

test "Bradford: pdf(0.0; c=1) ≈ 1.44270 (=1/ln(2))" {
    const dist = try Bradford(f64).init(1.0);
    const p = dist.pdf(0.0);
    const expected = 1.0 / @log(2.0);
    try testing.expectApproxEqAbs(expected, p, 1e-5);
}

test "Bradford: pdf(0.5; c=1) ≈ 0.96116" {
    const dist = try Bradford(f64).init(1.0);
    const p = dist.pdf(0.5);
    const expected = 1.0 / (@log(2.0) * 1.5);
    try testing.expectApproxEqAbs(expected, p, 1e-5);
}

test "Bradford: pdf(1.0; c=1) ≈ 0.72135 (=1/(2*ln(2)))" {
    const dist = try Bradford(f64).init(1.0);
    const p = dist.pdf(1.0);
    const expected = 1.0 / (2.0 * @log(2.0));
    try testing.expectApproxEqAbs(expected, p, 1e-5);
}

test "Bradford: pdf(0.0; c=2) ≈ 0.90457 (=2/ln(3))" {
    const dist = try Bradford(f64).init(2.0);
    const p = dist.pdf(0.0);
    const expected = 2.0 / @log(3.0);
    try testing.expectApproxEqAbs(expected, p, 1e-5);
}

test "Bradford: pdf(0.25; c=2)" {
    const dist = try Bradford(f64).init(2.0);
    const p = dist.pdf(0.25);
    // f(x; 2) = 2 / (ln(3) * (1 + 2*0.25)) = 2 / (ln(3) * 1.5)
    const expected = 2.0 / (@log(3.0) * 1.5);
    try testing.expectApproxEqAbs(expected, p, 1e-5);
}

test "Bradford: pdf outside [0,1] returns 0" {
    const dist = try Bradford(f64).init(1.0);
    try testing.expectEqual(@as(f64, 0.0), dist.pdf(-0.1));
    try testing.expectEqual(@as(f64, 0.0), dist.pdf(-1.0));
    try testing.expectEqual(@as(f64, 0.0), dist.pdf(1.1));
    try testing.expectEqual(@as(f64, 0.0), dist.pdf(2.0));
}

test "Bradford: pdf is positive everywhere on (0,1)" {
    const dist = try Bradford(f64).init(1.0);
    try testing.expect(dist.pdf(0.0) > 0.0);
    try testing.expect(dist.pdf(0.25) > 0.0);
    try testing.expect(dist.pdf(0.5) > 0.0);
    try testing.expect(dist.pdf(0.75) > 0.0);
    try testing.expect(dist.pdf(1.0) > 0.0);
}

test "Bradford: pdf is monotone decreasing on [0,1] for c>0" {
    const dist = try Bradford(f64).init(1.0);
    const p0 = dist.pdf(0.0);
    const p25 = dist.pdf(0.25);
    const p5 = dist.pdf(0.5);
    const p75 = dist.pdf(0.75);
    const p1 = dist.pdf(1.0);
    try testing.expect(p0 > p25);
    try testing.expect(p25 > p5);
    try testing.expect(p5 > p75);
    try testing.expect(p75 > p1);
}

test "Bradford: logpdf(0.5; c=1) ≈ ln(pdf(0.5))" {
    const dist = try Bradford(f64).init(1.0);
    const lp = dist.logpdf(0.5);
    const p = dist.pdf(0.5);
    const expected = @log(p);
    try testing.expectApproxEqAbs(expected, lp, 1e-10);
}

test "Bradford: logpdf(0.1; c=2)" {
    const dist = try Bradford(f64).init(2.0);
    const lp = dist.logpdf(0.1);
    const p = dist.pdf(0.1);
    const expected = @log(p);
    try testing.expectApproxEqAbs(expected, lp, 1e-10);
}

test "Bradford: logpdf outside [0,1] returns -infinity" {
    const dist = try Bradford(f64).init(1.0);
    try testing.expect(dist.logpdf(-0.1) == -math.inf(f64));
    try testing.expect(dist.logpdf(1.1) == -math.inf(f64));
}

test "Bradford: pdf integrates to approximately 1 on [0,1] for c=1 (trapezoid)" {
    const dist = try Bradford(f64).init(1.0);
    var sum: f64 = 0.0;
    const dx = 0.001;
    var x: f64 = 0.0;
    while (x <= 1.0) : (x += dx) {
        sum += dist.pdf(x) * dx;
    }
    try testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-4);
}

test "Bradford: pdf integrates to approximately 1 on [0,1] for c=2 (trapezoid)" {
    const dist = try Bradford(f64).init(2.0);
    var sum: f64 = 0.0;
    const dx = 0.001;
    var x: f64 = 0.0;
    while (x <= 1.0) : (x += dx) {
        sum += dist.pdf(x) * dx;
    }
    try testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-4);
}

test "Bradford: cdf(0.0; c=1) = 0" {
    const dist = try Bradford(f64).init(1.0);
    const c = dist.cdf(0.0);
    try testing.expectEqual(@as(f64, 0.0), c);
}

test "Bradford: cdf(1.0; c=1) = 1" {
    const dist = try Bradford(f64).init(1.0);
    const c = dist.cdf(1.0);
    try testing.expectEqual(@as(f64, 1.0), c);
}

test "Bradford: cdf(0.5; c=1) ≈ 0.58496 (ln(1.5)/ln(2))" {
    const dist = try Bradford(f64).init(1.0);
    const c = dist.cdf(0.5);
    const expected = @log(1.5) / @log(2.0);
    try testing.expectApproxEqAbs(expected, c, 1e-5);
}

test "Bradford: cdf(0.25; c=1)" {
    const dist = try Bradford(f64).init(1.0);
    const c = dist.cdf(0.25);
    const expected = @log(1.25) / @log(2.0);
    try testing.expectApproxEqAbs(expected, c, 1e-5);
}

test "Bradford: cdf(0.75; c=1)" {
    const dist = try Bradford(f64).init(1.0);
    const c = dist.cdf(0.75);
    const expected = @log(1.75) / @log(2.0);
    try testing.expectApproxEqAbs(expected, c, 1e-5);
}

test "Bradford: cdf below 0 returns 0" {
    const dist = try Bradford(f64).init(1.0);
    try testing.expectEqual(@as(f64, 0.0), dist.cdf(-1.0));
    try testing.expectEqual(@as(f64, 0.0), dist.cdf(-0.5));
}

test "Bradford: cdf above 1 returns 1" {
    const dist = try Bradford(f64).init(1.0);
    try testing.expectEqual(@as(f64, 1.0), dist.cdf(1.1));
    try testing.expectEqual(@as(f64, 1.0), dist.cdf(2.0));
}

test "Bradford: cdf is monotone increasing" {
    const dist = try Bradford(f64).init(1.0);
    const c0 = dist.cdf(0.1);
    const c25 = dist.cdf(0.25);
    const c5 = dist.cdf(0.5);
    const c75 = dist.cdf(0.75);
    const c1 = dist.cdf(0.9);
    try testing.expect(c0 < c25);
    try testing.expect(c25 < c5);
    try testing.expect(c5 < c75);
    try testing.expect(c75 < c1);
}

test "Bradford: cdf is in [0,1]" {
    const dist = try Bradford(f64).init(1.0);
    for ([_]f64{ 0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0 }) |x| {
        const c = dist.cdf(x);
        try testing.expect(c >= 0.0 and c <= 1.0);
    }
}

test "Bradford: quantile(0; c=1) = 0" {
    const dist = try Bradford(f64).init(1.0);
    const q = try dist.quantile(0.0);
    try testing.expectEqual(@as(f64, 0.0), q);
}

test "Bradford: quantile(1; c=1) = 1" {
    const dist = try Bradford(f64).init(1.0);
    const q = try dist.quantile(1.0);
    try testing.expectEqual(@as(f64, 1.0), q);
}

test "Bradford: quantile(0.5; c=1) ≈ √2 - 1 ≈ 0.41421" {
    const dist = try Bradford(f64).init(1.0);
    const q = try dist.quantile(0.5);
    const expected = @sqrt(2.0) - 1.0;
    try testing.expectApproxEqAbs(expected, q, 1e-5);
}

test "Bradford: quantile(0.25; c=1)" {
    const dist = try Bradford(f64).init(1.0);
    const q = try dist.quantile(0.25);
    // Q(p) = (2^p - 1) / 1 = 2^0.25 - 1
    const expected = @pow(@as(f64, 2.0), 0.25) - 1.0;
    try testing.expectApproxEqAbs(expected, q, 1e-5);
}

test "Bradford: quantile(0.75; c=1)" {
    const dist = try Bradford(f64).init(1.0);
    const q = try dist.quantile(0.75);
    const expected = @pow(@as(f64, 2.0), 0.75) - 1.0;
    try testing.expectApproxEqAbs(expected, q, 1e-5);
}

test "Bradford: quantile(0.1; c=1)" {
    const dist = try Bradford(f64).init(1.0);
    const q = try dist.quantile(0.1);
    const expected = @pow(@as(f64, 2.0), 0.1) - 1.0;
    try testing.expectApproxEqAbs(expected, q, 1e-5);
}

test "Bradford: quantile(0.9; c=1)" {
    const dist = try Bradford(f64).init(1.0);
    const q = try dist.quantile(0.9);
    const expected = @pow(@as(f64, 2.0), 0.9) - 1.0;
    try testing.expectApproxEqAbs(expected, q, 1e-5);
}

test "Bradford: quantile is monotone increasing" {
    const dist = try Bradford(f64).init(1.0);
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

test "Bradford: quantile fails for p=-0.1" {
    const dist = try Bradford(f64).init(1.0);
    try testing.expectError(error.InvalidProbability, dist.quantile(-0.1));
}

test "Bradford: quantile fails for p=1.1" {
    const dist = try Bradford(f64).init(1.0);
    try testing.expectError(error.InvalidProbability, dist.quantile(1.1));
}

test "Bradford: quantile fails for p=NaN" {
    const dist = try Bradford(f64).init(1.0);
    const p = math.nan(f64);
    try testing.expectError(error.InvalidProbability, dist.quantile(p));
}

test "Bradford: cdf(quantile(p)) ≈ p roundtrip for p=0.25" {
    const dist = try Bradford(f64).init(1.0);
    const p = 0.25;
    const q = try dist.quantile(p);
    const c = dist.cdf(q);
    try testing.expectApproxEqAbs(p, c, 1e-10);
}

test "Bradford: cdf(quantile(p)) ≈ p roundtrip for p=0.5" {
    const dist = try Bradford(f64).init(1.0);
    const p = 0.5;
    const q = try dist.quantile(p);
    const c = dist.cdf(q);
    try testing.expectApproxEqAbs(p, c, 1e-10);
}

test "Bradford: cdf(quantile(p)) ≈ p roundtrip for p=0.75" {
    const dist = try Bradford(f64).init(1.0);
    const p = 0.75;
    const q = try dist.quantile(p);
    const c = dist.cdf(q);
    try testing.expectApproxEqAbs(p, c, 1e-10);
}

test "Bradford: cdf(quantile(p)) ≈ p roundtrip for p=0.9" {
    const dist = try Bradford(f64).init(1.0);
    const p = 0.9;
    const q = try dist.quantile(p);
    const c = dist.cdf(q);
    try testing.expectApproxEqAbs(p, c, 1e-10);
}

test "Bradford: mean(c=1) ≈ 0.44269 (=(1-ln(2))/ln(2))" {
    const dist = try Bradford(f64).init(1.0);
    const m = dist.mean();
    const expected = (1.0 - @log(2.0)) / @log(2.0);
    try testing.expectApproxEqAbs(expected, m, 1e-5);
}

test "Bradford: mean(c=0.5)" {
    const dist = try Bradford(f64).init(0.5);
    const m = dist.mean();
    const ln15 = @log(1.5);
    const expected = (0.5 - ln15) / (0.5 * ln15);
    try testing.expectApproxEqAbs(expected, m, 1e-5);
}

test "Bradford: mean(c=2)" {
    const dist = try Bradford(f64).init(2.0);
    const m = dist.mean();
    const ln3 = @log(3.0);
    const expected = (2.0 - ln3) / (2.0 * ln3);
    try testing.expectApproxEqAbs(expected, m, 1e-5);
}

test "Bradford: as c→0, mean→0.5" {
    const dist = try Bradford(f64).init(1e-4);
    const m = dist.mean();
    try testing.expectApproxEqAbs(@as(f64, 0.5), m, 0.01);
}

test "Bradford: mode(c=1) = 0.0" {
    const dist = try Bradford(f64).init(1.0);
    try testing.expectEqual(@as(f64, 0.0), dist.mode());
}

test "Bradford: mode(c=0.5) = 0.0" {
    const dist = try Bradford(f64).init(0.5);
    try testing.expectEqual(@as(f64, 0.0), dist.mode());
}

test "Bradford: variance(c=1) ≈ 0.0826" {
    const dist = try Bradford(f64).init(1.0);
    const v = dist.variance();
    const ln2 = @log(2.0);
    const K = ln2;
    const expected = (K * 3.0 - 2.0) / (2.0 * K * K);
    try testing.expectApproxEqAbs(expected, v, 1e-4);
}

test "Bradford: variance is positive for all valid c" {
    const dist = try Bradford(f64).init(1.0);
    try testing.expect(dist.variance() > 0.0);
    const dist2 = try Bradford(f64).init(0.5);
    try testing.expect(dist2.variance() > 0.0);
    const dist3 = try Bradford(f64).init(2.0);
    try testing.expect(dist3.variance() > 0.0);
}

test "Bradford: as c→0, variance→1/12" {
    const dist = try Bradford(f64).init(1e-4);
    const v = dist.variance();
    try testing.expectApproxEqAbs(1.0 / 12.0, v, 0.005);
}

test "Bradford: entropy(c=1)" {
    const dist = try Bradford(f64).init(1.0);
    const e = dist.entropy();
    const ln2 = @log(2.0);
    const expected = @log(ln2 / 1.0) + ln2 / 2.0;
    try testing.expectApproxEqAbs(expected, e, 1e-4);
}

test "Bradford: entropy(c=0.5)" {
    const dist = try Bradford(f64).init(0.5);
    const e = dist.entropy();
    const K = @log(1.5);
    const expected = @log(K / 0.5) + K / 2.0;
    try testing.expectApproxEqAbs(expected, e, 1e-3);
}

test "Bradford: entropy increases as c→0 (approaching Uniform)" {
    const dist_small = try Bradford(f64).init(1e-4);
    const dist_large = try Bradford(f64).init(1.0);
    const e_small = dist_small.entropy();
    const e_large = dist_large.entropy();
    try testing.expect(e_small > e_large);
}

test "Bradford: sample produces values in [0,1]" {
    const dist = try Bradford(f64).init(1.0);
    var rng = std.Random.DefaultPrng.init(12345);
    for (0..100) |_| {
        const s = dist.sample(rng.random());
        try testing.expect(s >= 0.0 and s <= 1.0);
    }
}

test "Bradford: sample produces finite values" {
    const dist = try Bradford(f64).init(1.0);
    var rng = std.Random.DefaultPrng.init(54321);
    for (0..100) |_| {
        const s = dist.sample(rng.random());
        try testing.expect(math.isFinite(s));
    }
}

test "Bradford: sample mean converges to theoretical mean (N=3000, c=1)" {
    const dist = try Bradford(f64).init(1.0);
    var rng = std.Random.DefaultPrng.init(11111);
    var sum: f64 = 0.0;
    for (0..3000) |_| {
        sum += dist.sample(rng.random());
    }
    const sample_mean = sum / 3000.0;
    const theoretical_mean = dist.mean();
    try testing.expectApproxEqAbs(theoretical_mean, sample_mean, 0.05);
}

test "Bradford: sample mean converges to theoretical mean (N=3000, c=0.5)" {
    const dist = try Bradford(f64).init(0.5);
    var rng = std.Random.DefaultPrng.init(22222);
    var sum: f64 = 0.0;
    for (0..3000) |_| {
        sum += dist.sample(rng.random());
    }
    const sample_mean = sum / 3000.0;
    const theoretical_mean = dist.mean();
    try testing.expectApproxEqAbs(theoretical_mean, sample_mean, 0.05);
}

test "Bradford(f32): init succeeds with f32" {
    const dist = try Bradford(f32).init(1.0);
    try testing.expectEqual(@as(f32, 1.0), dist.c);
}

test "Bradford(f32): pdf produces finite values" {
    const dist = try Bradford(f32).init(1.0);
    for ([_]f32{ 0.0, 0.25, 0.5, 0.75, 1.0 }) |x| {
        const p = dist.pdf(x);
        try testing.expect(math.isFinite(p));
    }
}

test "Bradford(f32): cdf in [0,1]" {
    const dist = try Bradford(f32).init(1.0);
    for ([_]f32{ 0.0, 0.25, 0.5, 0.75, 1.0 }) |x| {
        const c = dist.cdf(x);
        try testing.expect(c >= 0.0 and c <= 1.0);
    }
}

test "Bradford(f32): sample produces finite values" {
    const dist = try Bradford(f32).init(1.0);
    var rng = std.Random.DefaultPrng.init(33333);
    for (0..50) |_| {
        const s = dist.sample(rng.random());
        try testing.expect(math.isFinite(s));
    }
}

test "Bradford: validate succeeds for c=1" {
    const dist = try Bradford(f64).init(1.0);
    try dist.validate();
}

test "Bradford: validate succeeds for c=0.5" {
    const dist = try Bradford(f64).init(0.5);
    try dist.validate();
}

test "Bradford: validate fails for c=0" {
    var dist: Bradford(f64) = undefined;
    dist.c = 0.0;
    try testing.expectError(error.InvalidParameter, dist.validate());
}

test "Bradford: validate fails for c=-1" {
    var dist: Bradford(f64) = undefined;
    dist.c = -1.0;
    try testing.expectError(error.InvalidParameter, dist.validate());
}

test "Bradford: validate fails for c=inf" {
    var dist: Bradford(f64) = undefined;
    dist.c = math.inf(f64);
    try testing.expectError(error.InvalidParameter, dist.validate());
}

test "Bradford: validate fails for c=NaN" {
    var dist: Bradford(f64) = undefined;
    dist.c = math.nan(f64);
    try testing.expectError(error.InvalidParameter, dist.validate());
}

test "Bradford: c=2 has steeper decline than c=1" {
    const dist1 = try Bradford(f64).init(1.0);
    const dist2 = try Bradford(f64).init(2.0);
    const pdf1_at_half = dist1.pdf(0.5);
    const pdf2_at_half = dist2.pdf(0.5);
    try testing.expect(pdf2_at_half < pdf1_at_half);
}

test "Bradford: quantile with c=2" {
    const dist = try Bradford(f64).init(2.0);
    const q5 = try dist.quantile(0.5);
    const expected = (@pow(@as(f64, 3.0), 0.5) - 1.0) / 2.0;
    try testing.expectApproxEqAbs(expected, q5, 1e-5);
}

test "Bradford: mean < 0.5 for all c > 0" {
    const dist1 = try Bradford(f64).init(0.1);
    const dist2 = try Bradford(f64).init(1.0);
    const dist3 = try Bradford(f64).init(10.0);
    try testing.expect(dist1.mean() < 0.5);
    try testing.expect(dist2.mean() < 0.5);
    try testing.expect(dist3.mean() < 0.5);
}

test "Bradford: quantile-CDF composition for multiple probabilities with c=2" {
    const dist = try Bradford(f64).init(2.0);
    for ([_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 }) |p| {
        const q = try dist.quantile(p);
        const c = dist.cdf(q);
        try testing.expectApproxEqAbs(p, c, 1e-10);
    }
}
