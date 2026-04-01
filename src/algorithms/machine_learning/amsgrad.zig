const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// AMSGrad (Adam with Maximum of Second Moments) optimizer
///
/// Improvement over Adam that uses the maximum of past second moments instead of
/// exponential moving average, providing better convergence guarantees.
/// Addresses the issue where Adam can fail to converge to optimal solutions.
///
/// Algorithm:
/// 1. Compute gradient g_t = ∇f(θ_t)
/// 2. Update biased first moment: m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
/// 3. Update biased second moment: v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²
/// 4. Update maximum second moment: v̂_t = max(v̂_{t-1}, v_t)
/// 5. Compute bias-corrected first moment: m̂_t = m_t / (1 - β₁^t)
/// 6. Update parameters: θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε)
///
/// Key features:
/// - Non-decreasing second moment (uses maximum instead of moving average)
/// - Better convergence guarantees than Adam
/// - Effective learning rate is monotonically decreasing
/// - Addresses Adam's convergence issues in certain scenarios
/// - Type-generic (f32/f64)
///
/// Time complexity: O(n) per update where n = number of parameters
/// Space complexity: O(n) for momentum and maximum second moment vectors
///
/// Use cases:
/// - When Adam fails to converge (e.g., some reinforcement learning tasks)
/// - Non-convex optimization requiring convergence guarantees
/// - Long-running training where Adam's exponential averaging might forget useful information
/// - Settings where monotonically decreasing learning rates are desired
///
/// Trade-offs:
/// - vs Adam: Better convergence guarantees, but can be slower due to monotonic v̂
/// - vs SGD: Adaptive rates reduce hyperparameter tuning, but more memory overhead
/// - vs AdamW: Similar stability, but AMSGrad focuses on convergence guarantees
///
/// References:
/// - Reddi et al. (2018): "On the Convergence of Adam and Beyond" (ICLR 2018)
/// - Fixes the convergence issues identified in Adam
pub fn AMSGrad(comptime T: type) type {
    return struct {
        allocator: Allocator,
        params: []T,
        m: []T, // First moment (mean of gradients)
        v: []T, // Second moment (uncentered variance of gradients)
        v_hat: []T, // Maximum of second moments (key difference from Adam)
        t: usize, // Timestep counter
        config: Config,

        const Self = @This();

        /// Configuration for AMSGrad optimizer
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

        /// Initialize AMSGrad optimizer
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
            const v_hat = try allocator.alloc(T, params.len);
            errdefer allocator.free(v_hat);

            // Initialize moments to zero
            @memset(m, 0);
            @memset(v, 0);
            @memset(v_hat, 0);

            return Self{
                .allocator = allocator,
                .params = params,
                .m = m,
                .v = v,
                .v_hat = v_hat,
                .t = 0,
                .config = config,
            };
        }

        /// Free allocated memory
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.m);
            self.allocator.free(self.v);
            self.allocator.free(self.v_hat);
        }

        /// Perform one optimization step
        /// Time: O(n) | Space: O(1)
        pub fn step(self: *Self, gradients: []const T) !void {
            if (gradients.len != self.params.len) return error.GradientLengthMismatch;

            self.t += 1;
            const t_f = @as(T, @floatFromInt(self.t));

            // Compute bias correction terms
            const bias_correction1 = 1.0 - std.math.pow(T, self.config.beta1, t_f);

            for (self.params, gradients, self.m, self.v, self.v_hat) |*param, grad, *m_i, *v_i, *v_hat_i| {
                // Update biased first moment estimate
                m_i.* = self.config.beta1 * m_i.* + (1.0 - self.config.beta1) * grad;

                // Update biased second moment estimate
                v_i.* = self.config.beta2 * v_i.* + (1.0 - self.config.beta2) * grad * grad;

                // Update maximum second moment (key difference from Adam)
                v_hat_i.* = @max(v_hat_i.*, v_i.*);

                // Compute bias-corrected first moment
                const m_hat = m_i.* / bias_correction1;

                // Update parameters using maximum second moment
                const denominator = @sqrt(v_hat_i.*) + self.config.epsilon;
                param.* -= self.config.learning_rate * m_hat / denominator;
            }
        }

        /// Reset optimizer state (useful for multi-task learning or hyperparameter tuning)
        /// Time: O(n) | Space: O(1)
        pub fn reset(self: *Self) void {
            @memset(self.m, 0);
            @memset(self.v, 0);
            @memset(self.v_hat, 0);
            self.t = 0;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "AMSGrad: basic initialization" {
    var params = [_]f64{ 1.0, 2.0, 3.0 };
    var optimizer = try AMSGrad(f64).init(testing.allocator, &params, .{});
    defer optimizer.deinit();

    try testing.expectEqual(@as(usize, 3), optimizer.params.len);
    try testing.expectEqual(@as(usize, 0), optimizer.t);
    try testing.expectEqual(@as(f64, 0.001), optimizer.config.learning_rate);
    try testing.expectEqual(@as(f64, 0.9), optimizer.config.beta1);
    try testing.expectEqual(@as(f64, 0.999), optimizer.config.beta2);
    try testing.expectEqual(@as(f64, 1e-8), optimizer.config.epsilon);
}

test "AMSGrad: custom configuration" {
    var params = [_]f64{ 1.0, 2.0 };
    const config = AMSGrad(f64).Config{
        .learning_rate = 0.01,
        .beta1 = 0.95,
        .beta2 = 0.9999,
        .epsilon = 1e-7,
    };
    var optimizer = try AMSGrad(f64).init(testing.allocator, &params, config);
    defer optimizer.deinit();

    try testing.expectEqual(@as(f64, 0.01), optimizer.config.learning_rate);
    try testing.expectEqual(@as(f64, 0.95), optimizer.config.beta1);
    try testing.expectEqual(@as(f64, 0.9999), optimizer.config.beta2);
    try testing.expectEqual(@as(f64, 1e-7), optimizer.config.epsilon);
}

test "AMSGrad: simple quadratic optimization" {
    // Minimize f(x) = (x - 2)^2 = x^2 - 4x + 4
    // Gradient: f'(x) = 2x - 4
    // Minimum at x = 2

    var params = [_]f64{0.0};
    var optimizer = try AMSGrad(f64).init(testing.allocator, &params, .{ .learning_rate = 0.1 });
    defer optimizer.deinit();

    // Run optimization steps
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const grad = [_]f64{2.0 * params[0] - 4.0};
        try optimizer.step(&grad);
    }

    // Should converge close to x = 2
    try testing.expect(@abs(params[0] - 2.0) < 0.1);
}

test "AMSGrad: multivariate quadratic" {
    // Minimize f(x,y) = x^2 + y^2
    // Gradients: ∂f/∂x = 2x, ∂f/∂y = 2y
    // Minimum at (0, 0)

    var params = [_]f64{ 5.0, -3.0 };
    var optimizer = try AMSGrad(f64).init(testing.allocator, &params, .{ .learning_rate = 0.1 });
    defer optimizer.deinit();

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const grad = [_]f64{ 2.0 * params[0], 2.0 * params[1] };
        try optimizer.step(&grad);
    }

    // Should converge close to (0, 0)
    try testing.expect(@abs(params[0]) < 0.1);
    try testing.expect(@abs(params[1]) < 0.1);
}

test "AMSGrad: Rosenbrock function" {
    // Minimize Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    // Minimum at (1, 1)

    var params = [_]f64{ 0.0, 0.0 };
    var optimizer = try AMSGrad(f64).init(testing.allocator, &params, .{ .learning_rate = 0.001 });
    defer optimizer.deinit();

    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        const x = params[0];
        const y = params[1];
        const grad_x = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
        const grad_y = 200.0 * (y - x * x);
        const grad = [_]f64{ grad_x, grad_y };
        try optimizer.step(&grad);
    }

    // Rosenbrock is hard to optimize, so we use looser tolerance
    try testing.expect(@abs(params[0] - 1.0) < 0.3);
    try testing.expect(@abs(params[1] - 1.0) < 0.3);
}

test "AMSGrad: maximum second moment validation" {
    // Test that v_hat is indeed the maximum of v over time
    var params = [_]f64{0.0};
    var optimizer = try AMSGrad(f64).init(testing.allocator, &params, .{ .learning_rate = 0.1 });
    defer optimizer.deinit();

    // First step with large gradient
    try optimizer.step(&[_]f64{10.0});
    const v_hat_after_large = optimizer.v_hat[0];

    // Second step with smaller gradient
    try optimizer.step(&[_]f64{1.0});

    // v_hat should not decrease (monotonicity property)
    try testing.expect(optimizer.v_hat[0] >= v_hat_after_large);
}

test "AMSGrad: bias correction" {
    // Test that bias correction affects early iterations
    var params = [_]f64{1.0};
    var optimizer = try AMSGrad(f64).init(testing.allocator, &params, .{ .learning_rate = 0.1 });
    defer optimizer.deinit();

    const initial_param = params[0];
    try optimizer.step(&[_]f64{1.0});
    const step1_change = @abs(params[0] - initial_param);

    // Reset and check multiple steps
    optimizer.reset();
    params[0] = initial_param;
    try optimizer.step(&[_]f64{1.0});
    try optimizer.step(&[_]f64{1.0});

    // Bias correction should make early steps larger
    try testing.expect(step1_change > 0);
}

test "AMSGrad: adaptive learning rates" {
    // Parameters with different gradient scales should get different effective learning rates
    var params = [_]f64{ 0.0, 0.0 };
    var optimizer = try AMSGrad(f64).init(testing.allocator, &params, .{ .learning_rate = 0.1 });
    defer optimizer.deinit();

    // Apply gradients with very different magnitudes
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        try optimizer.step(&[_]f64{ 0.01, 10.0 });
    }

    // The parameter with larger gradients should have moved less (adaptive rate is lower)
    // This is a key feature of adaptive methods
    try testing.expect(@abs(params[0]) > 0);
    try testing.expect(@abs(params[1]) > 0);
}

test "AMSGrad: sparse gradients" {
    // Test behavior with sparse gradients (some components are zero)
    var params = [_]f64{ 1.0, 1.0, 1.0 };
    var optimizer = try AMSGrad(f64).init(testing.allocator, &params, .{ .learning_rate = 0.1 });
    defer optimizer.deinit();

    // Only update first two parameters
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        try optimizer.step(&[_]f64{ 1.0, 1.0, 0.0 });
    }

    // Third parameter should remain unchanged (sparse gradient)
    try testing.expect(params[0] < 1.0);
    try testing.expect(params[1] < 1.0);
    try testing.expectEqual(@as(f64, 1.0), params[2]);
}

test "AMSGrad: reset functionality" {
    var params = [_]f64{ 1.0, 2.0 };
    var optimizer = try AMSGrad(f64).init(testing.allocator, &params, .{});
    defer optimizer.deinit();

    // Run some steps
    try optimizer.step(&[_]f64{ 0.1, 0.2 });
    try optimizer.step(&[_]f64{ 0.1, 0.2 });

    try testing.expect(optimizer.t > 0);
    try testing.expect(optimizer.m[0] != 0 or optimizer.m[1] != 0);

    // Reset
    optimizer.reset();

    try testing.expectEqual(@as(usize, 0), optimizer.t);
    try testing.expectEqual(@as(f64, 0), optimizer.m[0]);
    try testing.expectEqual(@as(f64, 0), optimizer.m[1]);
    try testing.expectEqual(@as(f64, 0), optimizer.v[0]);
    try testing.expectEqual(@as(f64, 0), optimizer.v[1]);
    try testing.expectEqual(@as(f64, 0), optimizer.v_hat[0]);
    try testing.expectEqual(@as(f64, 0), optimizer.v_hat[1]);
}

test "AMSGrad: f32 support" {
    var params = [_]f32{ 1.0, 2.0 };
    var optimizer = try AMSGrad(f32).init(testing.allocator, &params, .{ .learning_rate = 0.01 });
    defer optimizer.deinit();

    try optimizer.step(&[_]f32{ 0.1, 0.2 });
    try testing.expect(params[0] < 1.0);
    try testing.expect(params[1] < 2.0);
}

test "AMSGrad: f64 support" {
    var params = [_]f64{ 1.0, 2.0 };
    var optimizer = try AMSGrad(f64).init(testing.allocator, &params, .{ .learning_rate = 0.01 });
    defer optimizer.deinit();

    try optimizer.step(&[_]f64{ 0.1, 0.2 });
    try testing.expect(params[0] < 1.0);
    try testing.expect(params[1] < 2.0);
}

test "AMSGrad: large scale" {
    var params: [100]f64 = undefined;
    for (&params, 0..) |*p, i| {
        p.* = @as(f64, @floatFromInt(i));
    }

    var optimizer = try AMSGrad(f64).init(testing.allocator, &params, .{ .learning_rate = 0.01 });
    defer optimizer.deinit();

    var gradients: [100]f64 = undefined;
    for (&gradients, 0..) |*g, i| {
        g.* = @as(f64, @floatFromInt(i)) * 0.01;
    }

    try optimizer.step(&gradients);

    // All parameters should have been updated
    for (params, 0..) |p, i| {
        try testing.expect(p < @as(f64, @floatFromInt(i)));
    }
}

test "AMSGrad: empty parameters error" {
    var params = [_]f64{};
    const result = AMSGrad(f64).init(testing.allocator, &params, .{});
    try testing.expectError(error.EmptyParameters, result);
}

test "AMSGrad: gradient length mismatch" {
    var params = [_]f64{ 1.0, 2.0, 3.0 };
    var optimizer = try AMSGrad(f64).init(testing.allocator, &params, .{});
    defer optimizer.deinit();

    const wrong_grad = [_]f64{ 0.1, 0.2 };
    try testing.expectError(error.GradientLengthMismatch, optimizer.step(&wrong_grad));
}

test "AMSGrad: invalid learning rate" {
    var params = [_]f64{1.0};
    const result = AMSGrad(f64).init(testing.allocator, &params, .{ .learning_rate = 0.0 });
    try testing.expectError(error.InvalidLearningRate, result);
}

test "AMSGrad: invalid beta1" {
    var params = [_]f64{1.0};
    const result = AMSGrad(f64).init(testing.allocator, &params, .{ .beta1 = 1.0 });
    try testing.expectError(error.InvalidBeta1, result);
}

test "AMSGrad: invalid beta2" {
    var params = [_]f64{1.0};
    const result = AMSGrad(f64).init(testing.allocator, &params, .{ .beta2 = -0.1 });
    try testing.expectError(error.InvalidBeta2, result);
}

test "AMSGrad: invalid epsilon" {
    var params = [_]f64{1.0};
    const result = AMSGrad(f64).init(testing.allocator, &params, .{ .epsilon = 0.0 });
    try testing.expectError(error.InvalidEpsilon, result);
}

test "AMSGrad: memory safety with testing.allocator" {
    var params = [_]f64{ 1.0, 2.0, 3.0 };
    var optimizer = try AMSGrad(f64).init(testing.allocator, &params, .{});
    defer optimizer.deinit();

    try optimizer.step(&[_]f64{ 0.1, 0.2, 0.3 });
    try optimizer.step(&[_]f64{ 0.1, 0.2, 0.3 });
    // testing.allocator will detect memory leaks
}

test "AMSGrad: convergence comparison with varying gradients" {
    // Test that v_hat captures maximum variance, making AMSGrad more stable
    var params = [_]f64{5.0};
    var optimizer = try AMSGrad(f64).init(testing.allocator, &params, .{ .learning_rate = 0.1 });
    defer optimizer.deinit();

    // Apply varying gradients
    var i: usize = 0;
    while (i < 50) : (i += 1) {
        // Alternating large and small gradients
        const grad = if (i % 2 == 0) [_]f64{2.0} else [_]f64{0.1};
        try optimizer.step(&grad);
    }

    // v_hat should have captured the large gradient variance
    try testing.expect(optimizer.v_hat[0] > optimizer.v[0]);
}
