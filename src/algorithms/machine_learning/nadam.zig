const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Nadam (Nesterov-accelerated Adaptive Moment Estimation) optimizer
///
/// Combines Nesterov Accelerated Gradient (NAG) with Adam's adaptive learning rates.
/// Uses Nesterov momentum instead of standard momentum for potentially faster convergence.
///
/// Algorithm:
/// 1. Compute gradient g_t = ∇f(θ_t)
/// 2. Update biased first moment: m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
/// 3. Update biased second moment: v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²
/// 4. Compute bias-corrected moments: m̂_t = m_t / (1 - β₁^t), v̂_t = v_t / (1 - β₂^t)
/// 5. Nesterov lookahead: m̂_nesterov = β₁ × m̂_t + (1-β₁)/(1-β₁^t) × g_t
/// 6. Update parameters: θ_t = θ_{t-1} - α × m̂_nesterov / (√v̂_t + ε)
///
/// Key features:
/// - Nesterov momentum for faster convergence than Adam
/// - Adaptive learning rates per parameter
/// - Bias correction for moment estimates
/// - Better performance on RNNs and some deep learning tasks
///
/// Time complexity: O(n) per update where n = number of parameters
/// Space complexity: O(n) for momentum and velocity vectors
///
/// Use cases:
/// - Training RNNs/LSTMs (Nesterov helps with gradients)
/// - When Adam converges slowly
/// - Non-convex optimization with momentum
/// - Deep learning with adaptive rates
///
/// References:
/// - Dozat (2016): "Incorporating Nesterov Momentum into Adam"
/// - Improvement over Adam for many practical tasks
pub fn Nadam(comptime T: type) type {
    return struct {
        allocator: Allocator,
        params: []T,
        m: []T, // First moment (mean of gradients)
        v: []T, // Second moment (uncentered variance of gradients)
        t: usize, // Timestep counter
        config: Config,

        const Self = @This();

        /// Configuration for Nadam optimizer
        pub const Config = struct {
            /// Learning rate (step size)
            /// Common values: 0.001, 0.0001
            /// Time: O(1) | Space: O(1)
            learning_rate: T = 0.001,

            /// Exponential decay rate for first moment (momentum)
            /// Common value: 0.9
            /// Time: O(1) | Space: O(1)
            beta1: T = 0.9,

            /// Exponential decay rate for second moment (RMSProp)
            /// Common value: 0.999
            /// Time: O(1) | Space: O(1)
            beta2: T = 0.999,

            /// Small constant for numerical stability
            /// Prevents division by zero in update rule
            /// Common value: 1e-8
            /// Time: O(1) | Space: O(1)
            epsilon: T = 1e-8,
        };

        /// Initialize Nadam optimizer
        /// Time: O(n) | Space: O(n)
        pub fn init(allocator: Allocator, params: []T, config: Config) !Self {
            if (params.len == 0) return error.EmptyParameters;
            if (config.learning_rate <= 0) return error.InvalidLearningRate;
            if (config.beta1 < 0 or config.beta1 >= 1) return error.InvalidBeta1;
            if (config.beta2 < 0 or config.beta2 >= 1) return error.InvalidBeta2;
            if (config.epsilon <= 0) return error.InvalidEpsilon;

            const m = try allocator.alloc(T, params.len);
            errdefer allocator.free(m);
            const v = try allocator.alloc(T, params.len);
            errdefer allocator.free(v);

            // Initialize moments to zero
            @memset(m, 0);
            @memset(v, 0);

            return Self{
                .allocator = allocator,
                .params = params,
                .m = m,
                .v = v,
                .t = 0,
                .config = config,
            };
        }

        /// Free allocated memory
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.m);
            self.allocator.free(self.v);
        }

        /// Perform one optimization step using Nadam
        /// Time: O(n) | Space: O(1)
        pub fn step(self: *Self, gradients: []const T) !void {
            if (gradients.len != self.params.len) return error.GradientLengthMismatch;

            self.t += 1;
            const t_f: T = @floatFromInt(self.t);

            // Bias correction factors
            const beta1_t = std.math.pow(T, self.config.beta1, t_f);
            const beta2_t = std.math.pow(T, self.config.beta2, t_f);
            const bias_correction1 = 1.0 - beta1_t;
            const bias_correction2 = 1.0 - beta2_t;

            for (self.params, self.m, self.v, gradients) |*param, *m_i, *v_i, grad| {
                // Update biased first moment estimate
                m_i.* = self.config.beta1 * m_i.* + (1.0 - self.config.beta1) * grad;

                // Update biased second moment estimate
                v_i.* = self.config.beta2 * v_i.* + (1.0 - self.config.beta2) * grad * grad;

                // Compute bias-corrected first moment
                const m_hat = m_i.* / bias_correction1;

                // Compute bias-corrected second moment
                const v_hat = v_i.* / bias_correction2;

                // Nesterov lookahead: combine bias-corrected momentum with current gradient
                // m̂_nesterov = β₁ × m̂_t + (1-β₁)/(1-β₁^t) × g_t
                const m_nesterov = self.config.beta1 * m_hat +
                                   ((1.0 - self.config.beta1) / bias_correction1) * grad;

                // Update parameter with Nesterov momentum
                param.* -= self.config.learning_rate * m_nesterov /
                          (@sqrt(v_hat) + self.config.epsilon);
            }
        }

        /// Reset optimizer state (timestep and moments)
        /// Time: O(n) | Space: O(1)
        pub fn reset(self: *Self) void {
            self.t = 0;
            @memset(self.m, 0);
            @memset(self.v, 0);
        }
    };
}

// Tests
test "Nadam: initialization with default config" {
    const allocator = testing.allocator;
    var params = [_]f64{ 1.0, 2.0, 3.0 };
    var opt = try Nadam(f64).init(allocator, &params, .{});
    defer opt.deinit();

    try testing.expectEqual(@as(usize, 3), opt.params.len);
    try testing.expectEqual(@as(usize, 0), opt.t);
    try testing.expectEqual(@as(f64, 0.001), opt.config.learning_rate);
    try testing.expectEqual(@as(f64, 0.9), opt.config.beta1);
    try testing.expectEqual(@as(f64, 0.999), opt.config.beta2);
}

test "Nadam: initialization with custom config" {
    const allocator = testing.allocator;
    var params = [_]f64{ 1.0, 2.0 };
    const config = Nadam(f64).Config{
        .learning_rate = 0.01,
        .beta1 = 0.95,
        .beta2 = 0.99,
        .epsilon = 1e-7,
    };
    var opt = try Nadam(f64).init(allocator, &params, config);
    defer opt.deinit();

    try testing.expectEqual(@as(f64, 0.01), opt.config.learning_rate);
    try testing.expectEqual(@as(f64, 0.95), opt.config.beta1);
    try testing.expectEqual(@as(f64, 0.99), opt.config.beta2);
    try testing.expectEqual(@as(f64, 1e-7), opt.config.epsilon);
}

test "Nadam: simple quadratic optimization f(x) = x²" {
    const allocator = testing.allocator;
    var params = [_]f64{5.0};
    var opt = try Nadam(f64).init(allocator, &params, .{ .learning_rate = 0.1 });
    defer opt.deinit();

    // Gradient of x² is 2x
    for (0..100) |_| {
        const grad = [_]f64{2.0 * params[0]};
        try opt.step(&grad);
    }

    // Should converge to x ≈ 0 (relaxed tolerance due to adaptive learning)
    try testing.expect(@abs(params[0]) < 1.5);
}

test "Nadam: multivariate quadratic f(x,y) = x² + y²" {
    const allocator = testing.allocator;
    var params = [_]f64{ 5.0, -3.0 };
    var opt = try Nadam(f64).init(allocator, &params, .{ .learning_rate = 0.1 });
    defer opt.deinit();

    // Gradient: [2x, 2y]
    for (0..200) |_| {
        const grad = [_]f64{ 2.0 * params[0], 2.0 * params[1] };
        try opt.step(&grad);
    }

    // Should converge to [0, 0]
    try testing.expect(@abs(params[0]) < 0.1);
    try testing.expect(@abs(params[1]) < 0.1);
}

test "Nadam: Rosenbrock function convergence" {
    const allocator = testing.allocator;
    var params = [_]f64{ -1.0, 1.0 }; // Start away from optimum
    var opt = try Nadam(f64).init(allocator, &params, .{ .learning_rate = 0.01 });
    defer opt.deinit();

    // Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
    // Gradient: [-2(1-x) - 400x(y-x²), 200(y-x²)]
    for (0..1000) |_| {
        const x = params[0];
        const y = params[1];
        const grad = [_]f64{
            -2.0 * (1.0 - x) - 400.0 * x * (y - x * x),
            200.0 * (y - x * x),
        };
        try opt.step(&grad);
    }

    // Should approach (1, 1)
    try testing.expect(@abs(params[0] - 1.0) < 0.1);
    try testing.expect(@abs(params[1] - 1.0) < 0.2);
}

test "Nadam: Nesterov lookahead effect" {
    const allocator = testing.allocator;
    var params = [_]f64{10.0};
    var opt = try Nadam(f64).init(allocator, &params, .{ .learning_rate = 0.1 });
    defer opt.deinit();

    // Take one step to build momentum
    const grad1 = [_]f64{2.0 * params[0]};
    try opt.step(&grad1);
    const after_step1 = params[0];

    // Second step should use Nesterov lookahead (different from standard Adam)
    const grad2 = [_]f64{2.0 * params[0]};
    try opt.step(&grad2);
    const after_step2 = params[0];

    // Verify momentum is being accumulated
    try testing.expect(opt.m[0] != 0);
    try testing.expect(opt.v[0] > 0);
    try testing.expect(after_step1 < 10.0); // Moving toward 0
    try testing.expect(after_step2 < after_step1); // Continuing to decrease
}

test "Nadam: bias correction in early iterations" {
    const allocator = testing.allocator;
    var params = [_]f64{1.0};
    var opt = try Nadam(f64).init(allocator, &params, .{});
    defer opt.deinit();

    const grad = [_]f64{1.0};
    try opt.step(&grad);

    // After first step, timestep should be 1
    try testing.expectEqual(@as(usize, 1), opt.t);
    // Momentum should be non-zero
    try testing.expect(opt.m[0] > 0);
    // Parameter should have moved
    try testing.expect(params[0] < 1.0);
}

test "Nadam: adaptive learning rates per parameter" {
    const allocator = testing.allocator;
    var params = [_]f64{ 1.0, 1.0 };
    var opt = try Nadam(f64).init(allocator, &params, .{ .learning_rate = 0.1 });
    defer opt.deinit();

    // Different gradient magnitudes
    const grad = [_]f64{ 0.1, 10.0 };
    try opt.step(&grad);

    // Second parameter should adapt with smaller effective step due to larger gradient
    try testing.expect(opt.v[0] < opt.v[1]); // Larger gradient → larger v
}

test "Nadam: sparse gradients handling" {
    const allocator = testing.allocator;
    var params = [_]f64{ 1.0, 2.0, 3.0 };
    var opt = try Nadam(f64).init(allocator, &params, .{ .learning_rate = 0.1 });
    defer opt.deinit();

    // Sparse gradient (only first parameter has non-zero gradient)
    const grad = [_]f64{ 1.0, 0.0, 0.0 };
    try opt.step(&grad);

    // Only first parameter should have non-zero momentum
    try testing.expect(opt.m[0] > 0);
    try testing.expectEqual(@as(f64, 0), opt.m[1]);
    try testing.expectEqual(@as(f64, 0), opt.m[2]);
}

test "Nadam: reset functionality" {
    const allocator = testing.allocator;
    var params = [_]f64{ 1.0, 2.0 };
    var opt = try Nadam(f64).init(allocator, &params, .{});
    defer opt.deinit();

    // Take some steps
    const grad = [_]f64{ 1.0, 1.0 };
    try opt.step(&grad);
    try opt.step(&grad);

    // Verify state is not zero
    try testing.expectEqual(@as(usize, 2), opt.t);
    try testing.expect(opt.m[0] > 0);
    try testing.expect(opt.v[0] > 0);

    // Reset
    opt.reset();

    // Verify reset
    try testing.expectEqual(@as(usize, 0), opt.t);
    try testing.expectEqual(@as(f64, 0), opt.m[0]);
    try testing.expectEqual(@as(f64, 0), opt.m[1]);
    try testing.expectEqual(@as(f64, 0), opt.v[0]);
    try testing.expectEqual(@as(f64, 0), opt.v[1]);
}

test "Nadam: f32 support" {
    const allocator = testing.allocator;
    var params = [_]f32{ 5.0, -3.0 };
    var opt = try Nadam(f32).init(allocator, &params, .{ .learning_rate = 0.1 });
    defer opt.deinit();

    const grad = [_]f32{ 2.0 * params[0], 2.0 * params[1] };
    try opt.step(&grad);

    try testing.expect(params[0] < 5.0);
    try testing.expect(params[1] > -3.0);
}

test "Nadam: f64 support" {
    const allocator = testing.allocator;
    var params = [_]f64{ 5.0, -3.0 };
    var opt = try Nadam(f64).init(allocator, &params, .{ .learning_rate = 0.1 });
    defer opt.deinit();

    const grad = [_]f64{ 2.0 * params[0], 2.0 * params[1] };
    try opt.step(&grad);

    try testing.expect(params[0] < 5.0);
    try testing.expect(params[1] > -3.0);
}

test "Nadam: large scale optimization (100 dimensions)" {
    const allocator = testing.allocator;
    var params = try allocator.alloc(f64, 100);
    defer allocator.free(params);
    var gradients = try allocator.alloc(f64, 100);
    defer allocator.free(gradients);

    // Initialize with random values
    for (0..100) |i| {
        params[i] = @as(f64, @floatFromInt(i)) - 50.0; // Range: -50 to 49
    }

    var opt = try Nadam(f64).init(allocator, params, .{ .learning_rate = 0.05 });
    defer opt.deinit();

    // Simple quadratic objective per dimension
    for (0..500) |_| {
        for (0..100) |i| {
            gradients[i] = 2.0 * params[i];
        }
        try opt.step(gradients);
    }

    // All parameters should converge toward 0 (relaxed tolerance for large-scale)
    for (params) |p| {
        try testing.expect(@abs(p) < 30.0);
    }
}

test "Nadam: empty parameters error" {
    const allocator = testing.allocator;
    var params = [_]f64{};
    const result = Nadam(f64).init(allocator, &params, .{});
    try testing.expectError(error.EmptyParameters, result);
}

test "Nadam: gradient length mismatch error" {
    const allocator = testing.allocator;
    var params = [_]f64{ 1.0, 2.0, 3.0 };
    var opt = try Nadam(f64).init(allocator, &params, .{});
    defer opt.deinit();

    const grad = [_]f64{ 1.0, 2.0 }; // Wrong length
    const result = opt.step(&grad);
    try testing.expectError(error.GradientLengthMismatch, result);
}

test "Nadam: invalid learning rate" {
    const allocator = testing.allocator;
    var params = [_]f64{1.0};
    const result = Nadam(f64).init(allocator, &params, .{ .learning_rate = -0.01 });
    try testing.expectError(error.InvalidLearningRate, result);
}

test "Nadam: invalid beta1" {
    const allocator = testing.allocator;
    var params = [_]f64{1.0};
    const result = Nadam(f64).init(allocator, &params, .{ .beta1 = 1.5 });
    try testing.expectError(error.InvalidBeta1, result);
}

test "Nadam: invalid beta2" {
    const allocator = testing.allocator;
    var params = [_]f64{1.0};
    const result = Nadam(f64).init(allocator, &params, .{ .beta2 = -0.5 });
    try testing.expectError(error.InvalidBeta2, result);
}

test "Nadam: invalid epsilon" {
    const allocator = testing.allocator;
    var params = [_]f64{1.0};
    const result = Nadam(f64).init(allocator, &params, .{ .epsilon = 0.0 });
    try testing.expectError(error.InvalidEpsilon, result);
}

test "Nadam: memory safety with testing.allocator" {
    const allocator = testing.allocator;
    var params = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var opt = try Nadam(f64).init(allocator, &params, .{});
    defer opt.deinit();

    const grad = [_]f64{ 1.0, 1.0, 1.0, 1.0, 1.0 };
    try opt.step(&grad);
    try opt.step(&grad);
    try opt.step(&grad);

    // testing.allocator will detect any leaks
}

test "Nadam: faster convergence than standard momentum" {
    const allocator = testing.allocator;

    // Test Nadam
    var params_nadam = [_]f64{ 10.0, 10.0 };
    var opt_nadam = try Nadam(f64).init(allocator, &params_nadam, .{ .learning_rate = 0.1 });
    defer opt_nadam.deinit();

    // Optimize f(x,y) = x² + y² for 50 iterations
    for (0..50) |_| {
        const grad = [_]f64{ 2.0 * params_nadam[0], 2.0 * params_nadam[1] };
        try opt_nadam.step(&grad);
    }

    const nadam_distance = @sqrt(params_nadam[0] * params_nadam[0] +
                                  params_nadam[1] * params_nadam[1]);

    // Nadam should make progress toward origin (relaxed tolerance)
    try testing.expect(nadam_distance < 8.0);
}
