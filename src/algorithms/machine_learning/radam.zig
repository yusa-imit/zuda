const std = @import("std");
const testing = std.testing;

/// RAdam (Rectified Adam) Optimizer
///
/// Rectified Adaptive Moment Estimation addresses the variance issue in Adam during early training
/// by rectifying the variance through analytical correction. This eliminates the need for warmup
/// schedules and provides more stable convergence.
///
/// Algorithm (from Liu et al., 2020):
/// 1. First moment: m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
/// 2. Second moment: v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²
/// 3. Bias-corrected first moment: m̂_t = m_t / (1 - β₁^t)
/// 4. Variance rectification:
///    ρ_∞ = 2/(1 - β₂) - 1 (maximum length of approximated SMA)
///    ρ_t = ρ_∞ - 2t × β₂^t / (1 - β₂^t) (length of approximated SMA at time t)
///    If ρ_t > 4 (variance is tractable):
///      r_t = sqrt((ρ_t - 4)(ρ_t - 2)ρ_∞ / ((ρ_∞ - 4)(ρ_∞ - 2)ρ_t))
///      v̂_t = v_t / (1 - β₂^t)
///      θ_t = θ_{t-1} - α × r_t × m̂_t / (√v̂_t + ε)
///    Else (variance not tractable, use bias-corrected first moment only):
///      θ_t = θ_{t-1} - α × m̂_t
///
/// Key features:
/// - Variance rectification: Analytical correction for variance during early training
/// - No warmup needed: Eliminates need for learning rate warmup schedules
/// - Stable convergence: More robust than Adam, especially early in training
/// - Automatic switching: Uses first moment only when variance is not tractable
/// - Better than Adam: Matches or exceeds Adam performance without warmup
///
/// Time complexity: O(n) per update where n = number of parameters
/// Space complexity: O(n) for momentum and second moment vectors
///
/// Reference: Liu et al. (2020) "On the Variance of the Adaptive Learning Rate and Beyond" (ICLR 2020)
/// Use cases: Deep learning without warmup, transformer training, when Adam requires warmup,
///            stable training from cold start, settings where warmup tuning is expensive
pub fn RAdam(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Configuration for RAdam optimizer
        pub const Config = struct {
            /// Learning rate (default: 0.001, typical: 0.0001-0.001)
            learning_rate: T = 0.001,
            /// Exponential decay rate for first moment (default: 0.9, typical: 0.9-0.99)
            beta1: T = 0.9,
            /// Exponential decay rate for second moment (default: 0.999, typical: 0.99-0.999)
            beta2: T = 0.999,
            /// Small constant for numerical stability (default: 1e-8)
            epsilon: T = 1e-8,
        };

        allocator: std.mem.Allocator,
        config: Config,
        /// First moment (momentum) vector
        m: []T,
        /// Second moment (velocity) vector
        v: []T,
        /// Current timestep
        timestep: usize,
        /// Maximum length of approximated SMA: ρ_∞ = 2/(1 - β₂) - 1
        rho_inf: T,

        /// Initialize RAdam optimizer
        ///
        /// Time: O(n) | Space: O(n)
        pub fn init(allocator: std.mem.Allocator, num_params: usize, config: Config) !Self {
            if (num_params == 0) return error.EmptyParameters;
            if (config.learning_rate <= 0) return error.InvalidLearningRate;
            if (config.beta1 < 0 or config.beta1 >= 1) return error.InvalidBeta1;
            if (config.beta2 < 0 or config.beta2 >= 1) return error.InvalidBeta2;
            if (config.epsilon <= 0) return error.InvalidEpsilon;

            const m = try allocator.alloc(T, num_params);
            errdefer allocator.free(m);
            @memset(m, 0);

            const v = try allocator.alloc(T, num_params);
            @memset(v, 0);

            // ρ_∞ = 2/(1 - β₂) - 1
            const rho_inf = 2.0 / (1.0 - config.beta2) - 1.0;

            return Self{
                .allocator = allocator,
                .config = config,
                .m = m,
                .v = v,
                .timestep = 0,
                .rho_inf = rho_inf,
            };
        }

        /// Free optimizer state
        ///
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.m);
            self.allocator.free(self.v);
        }

        /// Perform one optimization step with variance rectification
        ///
        /// Time: O(n) | Space: O(1)
        pub fn step(self: *Self, params: []T, gradients: []const T) !void {
            if (params.len != self.m.len) return error.ParameterLengthMismatch;
            if (gradients.len != self.m.len) return error.GradientLengthMismatch;

            self.timestep += 1;
            const t = @as(T, @floatFromInt(self.timestep));

            // Compute bias correction factors
            const beta1_t = std.math.pow(T, self.config.beta1, t);
            const beta2_t = std.math.pow(T, self.config.beta2, t);
            const one_minus_beta1_t = 1.0 - beta1_t;
            const one_minus_beta2_t = 1.0 - beta2_t;

            // Compute ρ_t (length of approximated SMA at time t)
            // ρ_t = ρ_∞ - 2t × β₂^t / (1 - β₂^t)
            const rho_t = self.rho_inf - 2.0 * t * beta2_t / one_minus_beta2_t;

            // Determine if variance is tractable (ρ_t > 4)
            const variance_tractable = rho_t > 4.0;

            for (params, gradients, self.m, self.v) |*param, grad, *m, *v| {
                // Update biased first moment: m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
                m.* = self.config.beta1 * m.* + (1.0 - self.config.beta1) * grad;

                // Update biased second moment: v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²
                v.* = self.config.beta2 * v.* + (1.0 - self.config.beta2) * grad * grad;

                // Bias-corrected first moment: m̂_t = m_t / (1 - β₁^t)
                const m_hat = m.* / one_minus_beta1_t;

                if (variance_tractable) {
                    // Variance is tractable - use rectified adaptive learning rate
                    // Compute rectification term:
                    // r_t = sqrt((ρ_t - 4)(ρ_t - 2)ρ_∞ / ((ρ_∞ - 4)(ρ_∞ - 2)ρ_t))
                    const numerator = (rho_t - 4.0) * (rho_t - 2.0) * self.rho_inf;
                    const denominator = (self.rho_inf - 4.0) * (self.rho_inf - 2.0) * rho_t;
                    const r_t = @sqrt(numerator / denominator);

                    // Bias-corrected second moment: v̂_t = v_t / (1 - β₂^t)
                    const v_hat = v.* / one_minus_beta2_t;

                    // Update with rectified adaptive learning rate:
                    // θ_t = θ_{t-1} - α × r_t × m̂_t / (√v̂_t + ε)
                    param.* -= self.config.learning_rate * r_t * m_hat / (@sqrt(v_hat) + self.config.epsilon);
                } else {
                    // Variance not tractable - use bias-corrected first moment only (like momentum SGD)
                    // θ_t = θ_{t-1} - α × m̂_t
                    param.* -= self.config.learning_rate * m_hat;
                }
            }
        }

        /// Reset optimizer state (useful for restarting training)
        ///
        /// Time: O(n) | Space: O(1)
        pub fn reset(self: *Self) void {
            @memset(self.m, 0);
            @memset(self.v, 0);
            self.timestep = 0;
        }
    };
}

// Tests

test "RAdam: initialization" {
    const allocator = testing.allocator;
    const config = RAdam(f64).Config{};

    var opt = try RAdam(f64).init(allocator, 10, config);
    defer opt.deinit();

    try testing.expectEqual(@as(usize, 10), opt.m.len);
    try testing.expectEqual(@as(usize, 10), opt.v.len);
    try testing.expectEqual(@as(usize, 0), opt.timestep);

    // Verify rho_inf computation: ρ_∞ = 2/(1 - β₂) - 1
    const expected_rho_inf = 2.0 / (1.0 - config.beta2) - 1.0;
    try testing.expectApproxEqAbs(expected_rho_inf, opt.rho_inf, 1e-10);

    // All moments should be zero-initialized
    for (opt.m) |m| try testing.expectEqual(0.0, m);
    for (opt.v) |v| try testing.expectEqual(0.0, v);
}

test "RAdam: custom configuration" {
    const allocator = testing.allocator;
    const config = RAdam(f32).Config{
        .learning_rate = 0.002,
        .beta1 = 0.95,
        .beta2 = 0.9999,
        .epsilon = 1e-7,
    };

    var opt = try RAdam(f32).init(allocator, 5, config);
    defer opt.deinit();

    try testing.expectEqual(@as(f32, 0.002), opt.config.learning_rate);
    try testing.expectEqual(@as(f32, 0.95), opt.config.beta1);
    try testing.expectEqual(@as(f32, 0.9999), opt.config.beta2);
}

test "RAdam: simple quadratic optimization" {
    const allocator = testing.allocator;
    const config = RAdam(f64).Config{ .learning_rate = 0.1 };

    var opt = try RAdam(f64).init(allocator, 1, config);
    defer opt.deinit();

    // Minimize f(x) = x² (gradient = 2x, minimum at x = 0)
    var params = [_]f64{5.0};

    // Early steps: variance not tractable, uses momentum SGD
    for (0..5) |_| {
        const grad = [_]f64{2.0 * params[0]};
        try opt.step(&params, &grad);
    }

    // Later steps: variance tractable, uses rectified adaptive learning rate
    for (0..100) |_| {
        const grad = [_]f64{2.0 * params[0]};
        try opt.step(&params, &grad);
    }

    // Should converge to near zero
    try testing.expect(@abs(params[0]) < 0.1);
}

test "RAdam: multivariate quadratic" {
    const allocator = testing.allocator;
    const config = RAdam(f64).Config{ .learning_rate = 0.01 };

    var opt = try RAdam(f64).init(allocator, 3, config);
    defer opt.deinit();

    // Minimize f(x,y,z) = x² + 2y² + 3z² (gradients = [2x, 4y, 6z])
    var params = [_]f64{ 10.0, -5.0, 3.0 };

    for (0..500) |_| {
        const grad = [_]f64{ 2.0 * params[0], 4.0 * params[1], 6.0 * params[2] };
        try opt.step(&params, &grad);
    }

    // Should converge to zero
    try testing.expect(@abs(params[0]) < 0.01);
    try testing.expect(@abs(params[1]) < 0.01);
    try testing.expect(@abs(params[2]) < 0.01);
}

test "RAdam: Rosenbrock function" {
    const allocator = testing.allocator;
    const config = RAdam(f64).Config{ .learning_rate = 0.001 };

    var opt = try RAdam(f64).init(allocator, 2, config);
    defer opt.deinit();

    // Minimize Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
    // Minimum at (1, 1)
    var params = [_]f64{ -1.0, -1.0 };

    for (0..10000) |_| {
        const x = params[0];
        const y = params[1];
        // Gradients: ∂f/∂x = -2(1-x) - 400x(y-x²), ∂f/∂y = 200(y-x²)
        const grad_x = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
        const grad_y = 200.0 * (y - x * x);
        const grad = [_]f64{ grad_x, grad_y };
        try opt.step(&params, &grad);
    }

    // Should converge close to (1, 1)
    try testing.expect(@abs(params[0] - 1.0) < 0.1);
    try testing.expect(@abs(params[1] - 1.0) < 0.1);
}

test "RAdam: variance rectification computation" {
    const allocator = testing.allocator;
    const config = RAdam(f64).Config{};

    var opt = try RAdam(f64).init(allocator, 1, config);
    defer opt.deinit();

    // Verify rho_inf: ρ_∞ = 2/(1 - β₂) - 1 = 2/(1 - 0.999) - 1 = 1999
    try testing.expectApproxEqAbs(1999.0, opt.rho_inf, 1e-10);

    // At early timesteps, rho_t < 4 (variance not tractable)
    var params = [_]f64{1.0};
    const grad = [_]f64{0.1};

    // Step 1: rho_t should be very negative (not tractable)
    try opt.step(&params, &grad);
    const t1 = 1.0;
    const beta2_t1 = std.math.pow(f64, config.beta2, t1);
    const rho_t1 = opt.rho_inf - 2.0 * t1 * beta2_t1 / (1.0 - beta2_t1);
    try testing.expect(rho_t1 < 4.0); // Variance not tractable

    // After many steps, rho_t > 4 (variance tractable)
    for (0..100) |_| {
        try opt.step(&params, &grad);
    }
    const t_100 = 101.0;
    const beta2_t100 = std.math.pow(f64, config.beta2, t_100);
    const rho_t100 = opt.rho_inf - 2.0 * t_100 * beta2_t100 / (1.0 - beta2_t100);
    try testing.expect(rho_t100 > 4.0); // Variance tractable
}

test "RAdam: momentum accumulation" {
    const allocator = testing.allocator;
    const config = RAdam(f64).Config{};

    var opt = try RAdam(f64).init(allocator, 1, config);
    defer opt.deinit();

    var params = [_]f64{0.0};
    const grad = [_]f64{1.0};

    // After first step, momentum should be (1 - β₁) × g = 0.1
    try opt.step(&params, &grad);
    try testing.expectApproxEqAbs(0.1, opt.m[0], 1e-10);

    // After second step: m = 0.9 × 0.1 + 0.1 × 1 = 0.19
    try opt.step(&params, &grad);
    try testing.expectApproxEqAbs(0.19, opt.m[0], 1e-10);
}

test "RAdam: bias correction effect" {
    const allocator = testing.allocator;
    const config = RAdam(f64).Config{};

    var opt = try RAdam(f64).init(allocator, 1, config);
    defer opt.deinit();

    var params = [_]f64{1.0};
    const grad = [_]f64{0.5};

    const initial = params[0];
    try opt.step(&params, &grad);

    // Early steps use larger learning rate due to bias correction
    const change = @abs(params[0] - initial);
    try testing.expect(change > 0); // Should have moved
}

test "RAdam: sparse gradients" {
    const allocator = testing.allocator;
    const config = RAdam(f64).Config{};

    var opt = try RAdam(f64).init(allocator, 5, config);
    defer opt.deinit();

    var params = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    // Sparse gradient (only first and last elements)
    const grad = [_]f64{ 0.1, 0.0, 0.0, 0.0, 0.2 };

    try opt.step(&params, &grad);

    // First and last params should have non-zero momentum
    try testing.expect(opt.m[0] != 0);
    try testing.expect(opt.m[4] != 0);

    // Middle params should have zero momentum
    try testing.expectEqual(0.0, opt.m[1]);
    try testing.expectEqual(0.0, opt.m[2]);
    try testing.expectEqual(0.0, opt.m[3]);
}

test "RAdam: reset functionality" {
    const allocator = testing.allocator;
    const config = RAdam(f64).Config{};

    var opt = try RAdam(f64).init(allocator, 3, config);
    defer opt.deinit();

    var params = [_]f64{ 1.0, 2.0, 3.0 };
    const grad = [_]f64{ 0.1, 0.2, 0.3 };

    // Take some steps to accumulate state
    try opt.step(&params, &grad);
    try opt.step(&params, &grad);

    try testing.expect(opt.timestep > 0);
    try testing.expect(opt.m[0] != 0);
    try testing.expect(opt.v[0] != 0);

    // Reset
    opt.reset();

    try testing.expectEqual(@as(usize, 0), opt.timestep);
    for (opt.m) |m| try testing.expectEqual(0.0, m);
    for (opt.v) |v| try testing.expectEqual(0.0, v);
}

test "RAdam: f32 support" {
    const allocator = testing.allocator;
    const config = RAdam(f32).Config{};

    var opt = try RAdam(f32).init(allocator, 2, config);
    defer opt.deinit();

    var params = [_]f32{ 5.0, -3.0 };
    const grad = [_]f32{ 1.0, -0.5 };

    try opt.step(&params, &grad);

    try testing.expect(opt.m[0] != 0);
    try testing.expect(opt.v[0] != 0);
}

test "RAdam: large scale" {
    const allocator = testing.allocator;
    const config = RAdam(f64).Config{};

    var opt = try RAdam(f64).init(allocator, 100, config);
    defer opt.deinit();

    const params = try allocator.alloc(f64, 100);
    defer allocator.free(params);
    const gradients = try allocator.alloc(f64, 100);
    defer allocator.free(gradients);

    // Initialize with random-like values
    for (params, 0..) |*p, i| {
        p.* = @as(f64, @floatFromInt(i % 10)) - 5.0;
    }
    for (gradients, 0..) |*g, i| {
        g.* = @as(f64, @floatFromInt(i % 5)) * 0.1;
    }

    // Should handle large-scale updates
    for (0..50) |_| {
        try opt.step(params, gradients);
    }

    // All moments should be updated
    var all_updated = true;
    for (opt.m) |m| {
        if (m == 0) all_updated = false;
    }
    try testing.expect(all_updated);
}

test "RAdam: error on empty parameters" {
    const allocator = testing.allocator;
    const config = RAdam(f64).Config{};

    try testing.expectError(error.EmptyParameters, RAdam(f64).init(allocator, 0, config));
}

test "RAdam: error on mismatched parameter length" {
    const allocator = testing.allocator;
    const config = RAdam(f64).Config{};

    var opt = try RAdam(f64).init(allocator, 3, config);
    defer opt.deinit();

    var params = [_]f64{ 1.0, 2.0 }; // Wrong size
    const grad = [_]f64{ 0.1, 0.2, 0.3 };

    try testing.expectError(error.ParameterLengthMismatch, opt.step(&params, &grad));
}

test "RAdam: error on mismatched gradient length" {
    const allocator = testing.allocator;
    const config = RAdam(f64).Config{};

    var opt = try RAdam(f64).init(allocator, 3, config);
    defer opt.deinit();

    var params = [_]f64{ 1.0, 2.0, 3.0 };
    const grad = [_]f64{ 0.1, 0.2 }; // Wrong size

    try testing.expectError(error.GradientLengthMismatch, opt.step(&params, &grad));
}

test "RAdam: error on invalid learning rate" {
    const allocator = testing.allocator;
    const config = RAdam(f64).Config{ .learning_rate = -0.01 };

    try testing.expectError(error.InvalidLearningRate, RAdam(f64).init(allocator, 5, config));
}

test "RAdam: error on invalid beta1" {
    const allocator = testing.allocator;
    const config1 = RAdam(f64).Config{ .beta1 = -0.1 };
    const config2 = RAdam(f64).Config{ .beta1 = 1.0 };

    try testing.expectError(error.InvalidBeta1, RAdam(f64).init(allocator, 5, config1));
    try testing.expectError(error.InvalidBeta1, RAdam(f64).init(allocator, 5, config2));
}

test "RAdam: error on invalid beta2" {
    const allocator = testing.allocator;
    const config1 = RAdam(f64).Config{ .beta2 = -0.1 };
    const config2 = RAdam(f64).Config{ .beta2 = 1.0 };

    try testing.expectError(error.InvalidBeta2, RAdam(f64).init(allocator, 5, config1));
    try testing.expectError(error.InvalidBeta2, RAdam(f64).init(allocator, 5, config2));
}

test "RAdam: error on invalid epsilon" {
    const allocator = testing.allocator;
    const config = RAdam(f64).Config{ .epsilon = 0.0 };

    try testing.expectError(error.InvalidEpsilon, RAdam(f64).init(allocator, 5, config));
}

test "RAdam: convergence with varying gradients" {
    const allocator = testing.allocator;
    const config = RAdam(f64).Config{ .learning_rate = 0.01 };

    var opt = try RAdam(f64).init(allocator, 2, config);
    defer opt.deinit();

    // Start far from optimum
    var params = [_]f64{ 10.0, -10.0 };

    // Simulate varying gradients (noisy optimization)
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (0..1000) |_| {
        // True gradients plus noise
        const noise1 = (random.float(f64) - 0.5) * 0.1;
        const noise2 = (random.float(f64) - 0.5) * 0.1;
        const grad = [_]f64{
            2.0 * params[0] + noise1,
            2.0 * params[1] + noise2,
        };
        try opt.step(&params, &grad);
    }

    // Should still converge despite noise (RAdam is robust)
    try testing.expect(@abs(params[0]) < 1.0);
    try testing.expect(@abs(params[1]) < 1.0);
}

test "RAdam: no warmup needed demonstration" {
    const allocator = testing.allocator;

    // RAdam with aggressive learning rate (no warmup)
    const config_radam = RAdam(f64).Config{ .learning_rate = 0.1 };
    var opt_radam = try RAdam(f64).init(allocator, 1, config_radam);
    defer opt_radam.deinit();

    var params_radam = [_]f64{5.0};

    // RAdam handles early training without divergence
    for (0..20) |_| {
        const grad = [_]f64{2.0 * params_radam[0]};
        try opt_radam.step(&params_radam, &grad);
    }

    // Should converge stably even without warmup
    try testing.expect(@abs(params_radam[0]) < 3.0); // More stable than vanilla Adam
    try testing.expect(!std.math.isNan(params_radam[0]));
    try testing.expect(!std.math.isInf(params_radam[0]));
}
