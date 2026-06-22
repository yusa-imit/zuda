// ============================================================================
// WRAPPED CAUCHY DISTRIBUTION TESTS
// ============================================================================

test "WrappedCauchy: init with valid parameters (mu=0, rho=0.5)" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    try std.testing.expectApproxEqAbs(0.0, dist.mu, 1e-10);
    try std.testing.expectApproxEqAbs(0.5, dist.rho, 1e-10);
}

test "WrappedCauchy: init with valid parameters (mu=pi/2, rho=0.9)" {
    const dist = try WrappedCauchy(f64).init(math.pi / 2.0, 0.9);
    try std.testing.expectApproxEqAbs(math.pi / 2.0, dist.mu, 1e-10);
    try std.testing.expectApproxEqAbs(0.9, dist.rho, 1e-10);
}

test "WrappedCauchy: init with valid parameters (mu=-pi, rho=0.1)" {
    const dist = try WrappedCauchy(f64).init(-math.pi, 0.1);
    try std.testing.expectApproxEqAbs(-math.pi, dist.mu, 1e-10);
    try std.testing.expectApproxEqAbs(0.1, dist.rho, 1e-10);
}

test "WrappedCauchy: init rejects rho <= 0" {
    const result1 = WrappedCauchy(f64).init(0.0, 0.0);
    try std.testing.expectError(error.InvalidParameter, result1);

    const result2 = WrappedCauchy(f64).init(0.0, -0.5);
    try std.testing.expectError(error.InvalidParameter, result2);
}

test "WrappedCauchy: init rejects rho >= 1" {
    const result1 = WrappedCauchy(f64).init(0.0, 1.0);
    try std.testing.expectError(error.InvalidParameter, result1);

    const result2 = WrappedCauchy(f64).init(0.0, 1.5);
    try std.testing.expectError(error.InvalidParameter, result2);
}

test "WrappedCauchy: init rejects non-finite mu" {
    const result1 = WrappedCauchy(f64).init(math.inf(f64), 0.5);
    try std.testing.expectError(error.InvalidParameter, result1);

    const result2 = WrappedCauchy(f64).init(-math.inf(f64), 0.5);
    try std.testing.expectError(error.InvalidParameter, result2);

    const result3 = WrappedCauchy(f64).init(math.nan(f64), 0.5);
    try std.testing.expectError(error.InvalidParameter, result3);
}

test "WrappedCauchy: init rejects non-finite rho" {
    const result1 = WrappedCauchy(f64).init(0.0, math.inf(f64));
    try std.testing.expectError(error.InvalidParameter, result1);

    const result2 = WrappedCauchy(f64).init(0.0, math.nan(f64));
    try std.testing.expectError(error.InvalidParameter, result2);
}

test "WrappedCauchy: pdf at mode (mu=0, rho=0.5, theta=0) ≈ 0.47746" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const p = dist.pdf(0.0);
    const expected = 1.5 / math.pi;
    try std.testing.expectApproxEqAbs(expected, p, 1e-5);
}

test "WrappedCauchy: pdf at antipode (mu=0, rho=0.5, theta=pi) ≈ 0.05305" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const p = dist.pdf(math.pi);
    const expected = 1.0 / (6.0 * math.pi);
    try std.testing.expectApproxEqAbs(expected, p, 1e-5);
}

test "WrappedCauchy: pdf at theta=-pi same as theta=pi (wrapping)" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const p_pos = dist.pdf(math.pi);
    const p_neg = dist.pdf(-math.pi);
    try std.testing.expectApproxEqAbs(p_pos, p_neg, 1e-10);
}

test "WrappedCauchy: pdf at high concentration (mu=0, rho=0.9, theta=0) ≈ 3.0237" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.9);
    const p = dist.pdf(0.0);
    const expected = 9.5 / math.pi;
    try std.testing.expectApproxEqAbs(expected, p, 1e-4);
}

test "WrappedCauchy: pdf positive everywhere on support" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    var theta: f64 = -math.pi + 0.01;
    while (theta < math.pi) : (theta += 0.1) {
        const p = dist.pdf(theta);
        try std.testing.expect(p > 0.0);
    }
}

test "WrappedCauchy: pdf symmetric around mu when mu=0" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const p_pos = dist.pdf(0.5);
    const p_neg = dist.pdf(-0.5);
    try std.testing.expectApproxEqAbs(p_pos, p_neg, 1e-10);
}

test "WrappedCauchy: pdf with non-zero mu" {
    const dist = try WrappedCauchy(f64).init(1.0, 0.5);
    const p_at_mu = dist.pdf(1.0);
    const p_at_other = dist.pdf(1.5);
    try std.testing.expect(p_at_mu > p_at_other);
}

test "WrappedCauchy: pdf integrates to approximately 1" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const step = 0.00628; // ~1000 points over [-π, π]
    var sum: f64 = 0.0;
    var theta: f64 = -math.pi;
    while (theta < math.pi) : (theta += step) {
        sum += dist.pdf(theta) * step;
    }
    try std.testing.expectApproxEqAbs(1.0, sum, 0.01);
}

test "WrappedCauchy: logpdf equals log(pdf)" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const theta_vals = [_]f64{ 0.0, 0.5, -0.5, 1.5, -2.0 };
    for (theta_vals) |theta| {
        const p = dist.pdf(theta);
        const lp = dist.logpdf(theta);
        const expected = @log(p);
        try std.testing.expectApproxEqAbs(expected, lp, 1e-10);
    }
}

test "WrappedCauchy: cdf at theta=0 (mu=0, rho=0.5) equals 0.5" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const c = dist.cdf(0.0);
    try std.testing.expectApproxEqAbs(0.5, c, 1e-5);
}

test "WrappedCauchy: cdf at pi/2 (mu=0, rho=0.5) ≈ 0.89758" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const c = dist.cdf(math.pi / 2.0);
    const expected = 0.5 + math.atan(3.0) / math.pi;
    try std.testing.expectApproxEqAbs(expected, c, 1e-5);
}

test "WrappedCauchy: cdf at -pi/2 (mu=0, rho=0.5) ≈ 0.10242" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const c = dist.cdf(-math.pi / 2.0);
    const expected = 0.5 - math.atan(3.0) / math.pi;
    try std.testing.expectApproxEqAbs(expected, c, 1e-5);
}

test "WrappedCauchy: cdf at -pi equals 0" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const c = dist.cdf(-math.pi);
    try std.testing.expectApproxEqAbs(0.0, c, 1e-10);
}

test "WrappedCauchy: cdf at pi equals 1" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const c = dist.cdf(math.pi);
    try std.testing.expectApproxEqAbs(1.0, c, 1e-10);
}

test "WrappedCauchy: cdf is monotonically increasing" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    var prev_c: f64 = 0.0;
    var theta: f64 = -math.pi + 0.01;
    while (theta < math.pi) : (theta += 0.1) {
        const c = dist.cdf(theta);
        try std.testing.expect(c >= prev_c - 1e-10);
        prev_c = c;
    }
}

test "WrappedCauchy: sf equals 1 - cdf" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const theta_vals = [_]f64{ 0.0, 0.5, -0.5, 1.5, -2.0 };
    for (theta_vals) |theta| {
        const c = dist.cdf(theta);
        const s = dist.sf(theta);
        try std.testing.expectApproxEqAbs(1.0, c + s, 1e-10);
    }
}

test "WrappedCauchy: sf at 0 equals 0.5" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const s = dist.sf(0.0);
    try std.testing.expectApproxEqAbs(0.5, s, 1e-5);
}

test "WrappedCauchy: quantile at 0.5 equals 0 (mu=0, rho=0.5)" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const q = try dist.quantile(0.5);
    try std.testing.expectApproxEqAbs(0.0, q, 1e-5);
}

test "WrappedCauchy: quantile at 0.75 (mu=0, rho=0.5) ≈ 0.64350" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const q = try dist.quantile(0.75);
    const expected = 2.0 * math.atan(1.0 / 3.0);
    try std.testing.expectApproxEqAbs(expected, q, 1e-5);
}

test "WrappedCauchy: quantile at 0.25 (mu=0, rho=0.5) ≈ -0.64350" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const q = try dist.quantile(0.25);
    const expected = -2.0 * math.atan(1.0 / 3.0);
    try std.testing.expectApproxEqAbs(expected, q, 1e-5);
}

test "WrappedCauchy: quantile rejects p < 0" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const result = dist.quantile(-0.1);
    try std.testing.expectError(error.InvalidProbability, result);
}

test "WrappedCauchy: quantile rejects p > 1" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const result = dist.quantile(1.1);
    try std.testing.expectError(error.InvalidProbability, result);
}

test "WrappedCauchy: quantile CDF roundtrip for various p values" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const p_vals = [_]f64{ 0.1, 0.25, 0.5, 0.75, 0.9 };
    for (p_vals) |p| {
        const q = try dist.quantile(p);
        const c = dist.cdf(q);
        try std.testing.expectApproxEqAbs(p, c, 1e-4);
    }
}

test "WrappedCauchy: circularMean returns mu" {
    const dist = try WrappedCauchy(f64).init(0.5, 0.7);
    const mean = dist.circularMean();
    try std.testing.expectApproxEqAbs(0.5, mean, 1e-10);
}

test "WrappedCauchy: circularMean with various mu values" {
    const mu_vals = [_]f64{ 0.0, 1.0, -1.5, math.pi / 3.0 };
    for (mu_vals) |mu| {
        const dist = try WrappedCauchy(f64).init(mu, 0.5);
        const mean = dist.circularMean();
        try std.testing.expectApproxEqAbs(mu, mean, 1e-10);
    }
}

test "WrappedCauchy: circularVariance equals 1 - rho" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const cvar = dist.circularVariance();
    const expected = 1.0 - 0.5;
    try std.testing.expectApproxEqAbs(expected, cvar, 1e-10);
}

test "WrappedCauchy: circularVariance for various rho" {
    const rho_vals = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    for (rho_vals) |rho| {
        const dist = try WrappedCauchy(f64).init(0.0, rho);
        const cvar = dist.circularVariance();
        const expected = 1.0 - rho;
        try std.testing.expectApproxEqAbs(expected, cvar, 1e-10);
    }
}

test "WrappedCauchy: mode equals mu" {
    const dist = try WrappedCauchy(f64).init(0.5, 0.7);
    const m = dist.mode();
    try std.testing.expectApproxEqAbs(0.5, m, 1e-10);
}

test "WrappedCauchy: pdf at mode is maximum" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const p_at_mode = dist.pdf(0.0);
    const p_at_other_1 = dist.pdf(0.5);
    const p_at_other_2 = dist.pdf(-0.5);
    const p_at_far = dist.pdf(2.0);
    try std.testing.expect(p_at_mode > p_at_other_1);
    try std.testing.expect(p_at_mode > p_at_other_2);
    try std.testing.expect(p_at_mode > p_at_far);
}

test "WrappedCauchy: entropy for rho=0.5 ≈ 1.5508 nats" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const ent = dist.entropy();
    const expected = @log(2.0 * math.pi * 0.75);
    try std.testing.expectApproxEqAbs(expected, ent, 1e-5);
}

test "WrappedCauchy: entropy for rho=0.9 ≈ 0.1772 nats" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.9);
    const ent = dist.entropy();
    const expected = @log(2.0 * math.pi * 0.19);
    try std.testing.expectApproxEqAbs(expected, ent, 1e-4);
}

test "WrappedCauchy: entropy decreases with increasing rho" {
    const rho_vals = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    var prev_entropy = (try WrappedCauchy(f64).init(0.0, rho_vals[0])).entropy();
    for (rho_vals[1..]) |rho| {
        const dist = try WrappedCauchy(f64).init(0.0, rho);
        const ent = dist.entropy();
        try std.testing.expect(ent < prev_entropy + 1e-10);
        prev_entropy = ent;
    }
}

test "WrappedCauchy: entropy is correct formula" {
    const rho_vals = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    for (rho_vals) |rho| {
        const dist = try WrappedCauchy(f64).init(0.0, rho);
        const ent = dist.entropy();
        const expected = @log(2.0 * math.pi * (1.0 - rho * rho));
        try std.testing.expectApproxEqAbs(expected, ent, 1e-10);
    }
}

test "WrappedCauchy: sample returns finite values" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const s = dist.sample(rng);
        try std.testing.expect(math.isFinite(s));
    }
}

test "WrappedCauchy: sample returns values in (-pi, pi]" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const s = dist.sample(rng);
        try std.testing.expect(s > -math.pi and s <= math.pi);
    }
}

test "WrappedCauchy: sample with rho=0.5 has empirical mean direction near mu=0" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    var sum_sin: f64 = 0.0;
    var sum_cos: f64 = 0.0;
    const n = 5000;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const s = dist.sample(rng);
        sum_sin += math.sin(s);
        sum_cos += math.cos(s);
    }
    const mean_sin = sum_sin / @as(f64, @floatFromInt(n));
    const mean_cos = sum_cos / @as(f64, @floatFromInt(n));
    const empirical_mean = math.atan2(mean_sin, mean_cos);
    try std.testing.expectApproxEqAbs(0.0, empirical_mean, 0.1);
}

test "WrappedCauchy: sample with different seed produces different values" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    var prng1 = std.Random.DefaultPrng.init(42);
    var prng2 = std.Random.DefaultPrng.init(43);
    const rng1 = prng1.random();
    const rng2 = prng2.random();
    const s1 = dist.sample(rng1);
    const s2 = dist.sample(rng2);
    try std.testing.expect(@abs(s1 - s2) > 1e-5);
}

test "WrappedCauchy: validate accepts valid parameters" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    try dist.validate();
}

test "WrappedCauchy: validate for various valid parameters" {
    const mu_vals = [_]f64{ 0.0, 1.0, -1.5, math.pi / 3.0 };
    const rho_vals = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    for (mu_vals) |mu| {
        for (rho_vals) |rho| {
            const dist = try WrappedCauchy(f64).init(mu, rho);
            try dist.validate();
        }
    }
}

test "WrappedCauchy: f32 support basic smoke test" {
    const dist = try WrappedCauchy(f32).init(0.0, 0.5);
    try dist.validate();
    const p = dist.pdf(0.5);
    try std.testing.expect(p > 0.0);
}

test "WrappedCauchy: f32 cdf and quantile" {
    const dist = try WrappedCauchy(f32).init(0.0, 0.5);
    const c = dist.cdf(0.0);
    try std.testing.expectApproxEqAbs(0.5, c, 1e-4);
    const q = try dist.quantile(0.5);
    try std.testing.expectApproxEqAbs(0.0, q, 1e-3);
}

test "WrappedCauchy: f32 sample" {
    const dist = try WrappedCauchy(f32).init(0.0, 0.5);
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const s = dist.sample(rng);
    try std.testing.expect(math.isFinite(s));
    try std.testing.expect(s > -math.pi and s <= math.pi);
}

test "WrappedCauchy: symmetry when mu=0: pdf(theta) = pdf(-theta)" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const theta_vals = [_]f64{ 0.1, 0.5, 1.0, 1.5, 2.0 };
    for (theta_vals) |theta| {
        const p_pos = dist.pdf(theta);
        const p_neg = dist.pdf(-theta);
        try std.testing.expectApproxEqAbs(p_pos, p_neg, 1e-10);
    }
}

test "WrappedCauchy: symmetry when mu=0: cdf(theta) + cdf(-theta) ≈ 1" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const theta_vals = [_]f64{ 0.1, 0.5, 1.0, 1.5, 2.0 };
    for (theta_vals) |theta| {
        const c_pos = dist.cdf(theta);
        const c_neg = dist.cdf(-theta);
        try std.testing.expectApproxEqAbs(1.0, c_pos + c_neg, 1e-10);
    }
}

test "WrappedCauchy: pdf ratio at extreme and mode (mu=0, rho=0.5)" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.5);
    const p_mode = dist.pdf(0.0);
    const p_antipode = dist.pdf(math.pi);
    const ratio = p_mode / p_antipode;
    // Expected ratio: (1+ρ)²/(1-ρ)² = (1.5)²/(0.5)² = 2.25/0.25 = 9
    const expected_ratio = (1.0 + 0.5) * (1.0 + 0.5) / ((1.0 - 0.5) * (1.0 - 0.5));
    try std.testing.expectApproxEqAbs(expected_ratio, ratio, 1e-5);
}

test "WrappedCauchy: low concentration (rho near 0) approaches Uniform" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.01);
    const p_mode = dist.pdf(0.0);
    const p_other = dist.pdf(1.0);
    const ratio = p_mode / p_other;
    try std.testing.expectApproxEqAbs(1.0, ratio, 0.1);
}

test "WrappedCauchy: high concentration (rho near 1) becomes peaky" {
    const dist = try WrappedCauchy(f64).init(0.0, 0.99);
    const p_mode = dist.pdf(0.0);
    const p_far = dist.pdf(2.0);
    try std.testing.expect(p_mode > 10.0 * p_far);
}
