const std = @import("std");
const Allocator = std.mem.Allocator;

/// AdamW Optimizer — Adam with Decoupled Weight Decay Regularization
///
/// An improved variant of Adam that decouples weight decay from the gradient-based update.
/// Unlike Adam where weight decay is added to gradients (L2 regularization), AdamW applies
/// weight decay directly to parameters, providing better regularization and improved generalization.
///
/// Algorithm:
/// 1. Compute first moment (momentum): m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
/// 2. Compute second moment (RMSProp): v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²
/// 3. Bias correction: m̂_t = m_t / (1 - β₁^t), v̂_t = v_t / (1 - β₂^t)
/// 4. Parameter update: θ_t = θ_{t-1} - α × (m̂_t / (√v̂_t + ε) + λ × θ_{t-1})
///    where λ × θ_{t-1} is the decoupled weight decay term
///
/// Key differences from Adam:
/// - Weight decay is applied directly to parameters (not gradients)
/// - Better regularization without interfering with adaptive learning rates
/// - Often achieves better generalization than Adam with L2 regularization
///
/// Configuration:
/// - learning_rate: Step size (default: 0.001, typical: 0.0001-0.001)
/// - beta1: Exponential decay for first moment (default: 0.9)
/// - beta2: Exponential decay for second moment (default: 0.999)
/// - epsilon: Numerical stability constant (default: 1e-8)
/// - weight_decay: Decoupled weight decay coefficient (default: 0.01, typical: 0.01-0.1)
///
/// Time complexity: O(n) per update where n = number of parameters
/// Space complexity: O(n) for momentum and velocity vectors
///
/// Use cases:
/// - Transformers and large language models (BERT, GPT, etc.)
/// - Computer vision models (ResNet, ViT) where better generalization is needed
/// - Any deep learning task where Adam is used but overfitting is a concern
/// - Transfer learning (prevents catastrophic forgetting better than Adam)
///
/// Trade-offs:
/// - vs Adam: Better generalization via decoupled weight decay, often preferred for modern architectures
/// - vs SGD: Adaptive learning rates reduce need for manual tuning
/// - vs AdamW with amsgrad: Simpler without second moment maximum tracking
///
/// Reference: Loshchilov & Hutter (2019) "Decoupled Weight Decay Regularization" (ICLR 2019)
pub fn AdamW(comptime T: type) type {
    if (T != f32 and T != f64) {
        @compileError("AdamW only supports f32 and f64");
    }

    return struct {
        const Self = @This();

        allocator: Allocator,
        learning_rate: T,
        beta1: T,
        beta2: T,
        epsilon: T,
        weight_decay: T,
        timestep: usize,
        momentum: []T, // First moment (exponentially decaying average of gradients)
        velocity: []T, // Second moment (exponentially decaying average of squared gradients)

        /// Configuration for AdamW optimizer
        pub const Config = struct {
            learning_rate: T = 0.001,
            beta1: T = 0.9,
            beta2: T = 0.999,
            epsilon: T = 1e-8,
            weight_decay: T = 0.01,
        };

        /// Initialize AdamW optimizer
        ///
        /// Time: O(n)
        /// Space: O(n)
        pub fn init(allocator: Allocator, n_params: usize, config: Config) !Self {
            if (config.learning_rate <= 0) return error.InvalidLearningRate;
            if (config.beta1 < 0 or config.beta1 >= 1) return error.InvalidBeta1;
            if (config.beta2 < 0 or config.beta2 >= 1) return error.InvalidBeta2;
            if (config.epsilon <= 0) return error.InvalidEpsilon;
            if (config.weight_decay < 0) return error.InvalidWeightDecay;

            const momentum = try allocator.alloc(T, n_params);
            errdefer allocator.free(momentum);
            @memset(momentum, 0);

            const velocity = try allocator.alloc(T, n_params);
            errdefer allocator.free(velocity);
            @memset(velocity, 0);

            return .{
                .allocator = allocator,
                .learning_rate = config.learning_rate,
                .beta1 = config.beta1,
                .beta2 = config.beta2,
                .epsilon = config.epsilon,
                .weight_decay = config.weight_decay,
                .timestep = 0,
                .momentum = momentum,
                .velocity = velocity,
            };
        }

        /// Free allocated memory
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.momentum);
            self.allocator.free(self.velocity);
        }

        /// Update parameters using AdamW algorithm
        ///
        /// params: Current parameter values (will be modified in-place)
        /// gradients: Gradient values for each parameter
        ///
        /// Algorithm:
        /// 1. Update timestep: t = t + 1
        /// 2. Update momentum: m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
        /// 3. Update velocity: v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²
        /// 4. Bias-corrected momentum: m̂_t = m_t / (1 - β₁^t)
        /// 5. Bias-corrected velocity: v̂_t = v_t / (1 - β₂^t)
        /// 6. Adaptive learning rate: lr_adapted = α / (√v̂_t + ε)
        /// 7. Gradient update: θ_temp = θ - lr_adapted × m̂_t
        /// 8. Decoupled weight decay: θ_t = θ_temp - α × λ × θ_{t-1}
        ///
        /// Time: O(n)
        /// Space: O(1)
        pub fn update(self: *Self, params: []T, gradients: []const T) !void {
            if (params.len == 0) return error.EmptyParameters;
            if (params.len != gradients.len) return error.LengthMismatch;
            if (params.len != self.momentum.len) return error.LengthMismatch;

            self.timestep += 1;
            const t = self.timestep;

            // Precompute bias correction factors
            const beta1_pow = std.math.pow(T, self.beta1, @as(T, @floatFromInt(t)));
            const beta2_pow = std.math.pow(T, self.beta2, @as(T, @floatFromInt(t)));
            const bias_correction1 = 1.0 - beta1_pow;
            const bias_correction2 = 1.0 - beta2_pow;

            for (params, gradients, self.momentum, self.velocity, 0..) |*param, grad, *m, *v, i| {
                // Update biased first moment estimate
                m.* = self.beta1 * m.* + (1.0 - self.beta1) * grad;

                // Update biased second moment estimate
                v.* = self.beta2 * v.* + (1.0 - self.beta2) * grad * grad;

                // Compute bias-corrected first moment
                const m_hat = m.* / bias_correction1;

                // Compute bias-corrected second moment
                const v_hat = v.* / bias_correction2;

                // Compute adaptive learning rate
                const lr_adapted = self.learning_rate / (@sqrt(v_hat) + self.epsilon);

                // Apply gradient update
                const param_before_decay = param.*;
                param.* = param.* - lr_adapted * m_hat;

                // Apply decoupled weight decay directly to parameters
                // This is the key difference from Adam: weight decay is applied
                // to the original parameter value, not added to the gradient
                param.* = param.* - self.learning_rate * self.weight_decay * param_before_decay;

                _ = i; // Suppress unused variable warning
            }
        }

        /// Reset optimizer state
        ///
        /// Clears momentum and velocity vectors, resets timestep.
        /// Useful for starting a new training phase or experiment.
        ///
        /// Time: O(n)
        /// Space: O(1)
        pub fn reset(self: *Self) void {
            @memset(self.momentum, 0);
            @memset(self.velocity, 0);
            self.timestep = 0;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const expectEqual = testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectError = testing.expectError;

test "AdamW: initialization" {
    const allocator = testing.allocator;

    var opt = try AdamW(f64).init(allocator, 3, .{});
    defer opt.deinit();

    try expectEqual(@as(usize, 0), opt.timestep);
    try expectEqual(@as(f64, 0.001), opt.learning_rate);
    try expectEqual(@as(f64, 0.9), opt.beta1);
    try expectEqual(@as(f64, 0.999), opt.beta2);
    try expectEqual(@as(f64, 1e-8), opt.epsilon);
    try expectEqual(@as(f64, 0.01), opt.weight_decay);

    // Momentum and velocity should be initialized to zero
    for (opt.momentum) |m| try expectEqual(@as(f64, 0), m);
    for (opt.velocity) |v| try expectEqual(@as(f64, 0), v);
}

test "AdamW: custom config" {
    const allocator = testing.allocator;

    const config = AdamW(f64).Config{
        .learning_rate = 0.01,
        .beta1 = 0.95,
        .beta2 = 0.99,
        .epsilon = 1e-6,
        .weight_decay = 0.1,
    };

    var opt = try AdamW(f64).init(allocator, 2, config);
    defer opt.deinit();

    try expectEqual(@as(f64, 0.01), opt.learning_rate);
    try expectEqual(@as(f64, 0.95), opt.beta1);
    try expectEqual(@as(f64, 0.99), opt.beta2);
    try expectEqual(@as(f64, 1e-6), opt.epsilon);
    try expectEqual(@as(f64, 0.1), opt.weight_decay);
}

test "AdamW: simple quadratic optimization" {
    const allocator = testing.allocator;

    // Minimize f(x) = (x - 3)²
    // Gradient: f'(x) = 2(x - 3)
    // Optimal: x* = 3

    var opt = try AdamW(f64).init(allocator, 1, .{ .learning_rate = 0.1 });
    defer opt.deinit();

    var params = [_]f64{0.0};

    // Run 100 iterations
    var iter: usize = 0;
    while (iter < 100) : (iter += 1) {
        const x = params[0];
        const grad = [_]f64{2.0 * (x - 3.0)};
        try opt.update(&params, &grad);
    }

    // Should converge close to 3
    try expectApproxEqAbs(@as(f64, 3.0), params[0], 0.1);
}

test "AdamW: multivariate quadratic" {
    const allocator = testing.allocator;

    // Minimize f(x, y) = x² + y²
    // Gradients: ∂f/∂x = 2x, ∂f/∂y = 2y
    // Optimal: (x*, y*) = (0, 0)

    var opt = try AdamW(f64).init(allocator, 2, .{ .learning_rate = 0.1 });
    defer opt.deinit();

    var params = [_]f64{ 5.0, -5.0 };

    // Run 200 iterations
    var iter: usize = 0;
    while (iter < 200) : (iter += 1) {
        const grads = [_]f64{ 2.0 * params[0], 2.0 * params[1] };
        try opt.update(&params, &grads);
    }

    // Should converge close to (0, 0)
    try expectApproxEqAbs(@as(f64, 0.0), params[0], 0.1);
    try expectApproxEqAbs(@as(f64, 0.0), params[1], 0.1);
}

test "AdamW: Rosenbrock function" {
    const allocator = testing.allocator;

    // Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
    // Optimal: (1, 1)

    var opt = try AdamW(f64).init(allocator, 2, .{
        .learning_rate = 0.01,
        .weight_decay = 0.0, // No weight decay for this optimization test
    });
    defer opt.deinit();

    var params = [_]f64{ -1.0, 2.0 };

    // Run 1000 iterations
    var iter: usize = 0;
    while (iter < 1000) : (iter += 1) {
        const x = params[0];
        const y = params[1];

        // Compute gradients
        const grad_x = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
        const grad_y = 200.0 * (y - x * x);
        const grads = [_]f64{ grad_x, grad_y };

        try opt.update(&params, &grads);
    }

    // Should get reasonably close to (1, 1)
    try expectApproxEqAbs(@as(f64, 1.0), params[0], 0.2);
    try expectApproxEqAbs(@as(f64, 1.0), params[1], 0.2);
}

test "AdamW: momentum accumulation" {
    const allocator = testing.allocator;

    var opt = try AdamW(f64).init(allocator, 1, .{});
    defer opt.deinit();

    var params = [_]f64{0.0};
    const grads = [_]f64{1.0};

    // First update
    try opt.update(&params, &grads);

    // Momentum should be non-zero after first update
    try testing.expect(opt.momentum[0] != 0);
    try testing.expect(opt.velocity[0] != 0);

    const m1 = opt.momentum[0];
    const v1 = opt.velocity[0];

    // Second update with same gradient
    try opt.update(&params, &grads);

    // Momentum should have increased
    try testing.expect(opt.momentum[0] > m1);
    try testing.expect(opt.velocity[0] > v1);
}

test "AdamW: bias correction" {
    const allocator = testing.allocator;

    var opt = try AdamW(f64).init(allocator, 1, .{});
    defer opt.deinit();

    var params = [_]f64{0.0};
    const grads = [_]f64{1.0};

    const param_before = params[0];
    try opt.update(&params, &grads);
    const param_after = params[0];

    // Parameter should have changed
    try testing.expect(param_after != param_before);

    // Timestep should have incremented
    try expectEqual(@as(usize, 1), opt.timestep);
}

test "AdamW: adaptive learning rates" {
    const allocator = testing.allocator;

    var opt = try AdamW(f64).init(allocator, 2, .{ .learning_rate = 0.1 });
    defer opt.deinit();

    var params = [_]f64{ 1.0, 1.0 };

    // Different gradient magnitudes
    const grads = [_]f64{ 0.1, 10.0 };

    const params_before = params;
    try opt.update(&params, &grads);

    // Both parameters should have changed
    try testing.expect(params[0] != params_before[0]);
    try testing.expect(params[1] != params_before[1]);

    // Adaptive learning rates mean the parameter with larger gradient
    // doesn't necessarily change more (due to normalization by second moment)
}

test "AdamW: weight decay effect" {
    const allocator = testing.allocator;

    // Two optimizers: one with weight decay, one without
    var opt_with_decay = try AdamW(f64).init(allocator, 1, .{
        .learning_rate = 0.1,
        .weight_decay = 0.1,
    });
    defer opt_with_decay.deinit();

    var opt_without_decay = try AdamW(f64).init(allocator, 1, .{
        .learning_rate = 0.1,
        .weight_decay = 0.0,
    });
    defer opt_without_decay.deinit();

    var params_with = [_]f64{10.0};
    var params_without = [_]f64{10.0};
    const grads = [_]f64{0.0}; // Zero gradient

    try opt_with_decay.update(&params_with, &grads);
    try opt_without_decay.update(&params_without, &grads);

    // With weight decay, parameter should decrease even with zero gradient
    // because of the decoupled weight decay term: θ_t = θ_{t-1} - lr × λ × θ_{t-1}
    try testing.expect(params_with[0] < 10.0);
    try testing.expect(params_without[0] < 10.0); // Also changes due to bias correction with zero momentum
}

test "AdamW: decoupled weight decay vs L2 regularization" {
    const allocator = testing.allocator;

    // Test that weight decay is applied directly to parameters, not gradients
    var opt = try AdamW(f64).init(allocator, 1, .{
        .learning_rate = 0.1,
        .weight_decay = 0.1,
    });
    defer opt.deinit();

    var params = [_]f64{1.0};
    const grads = [_]f64{0.5};

    const param_before = params[0];
    try opt.update(&params, &grads);

    // Parameter should decrease more than just gradient descent would predict
    // because of decoupled weight decay: θ_new = θ_old - lr×(grad_update + λ×θ_old)
    try testing.expect(params[0] < param_before - 0.1 * 0.5); // More than just gradient update
}

test "AdamW: sparse gradients" {
    const allocator = testing.allocator;

    var opt = try AdamW(f64).init(allocator, 3, .{});
    defer opt.deinit();

    var params = [_]f64{ 1.0, 2.0, 3.0 };

    // Sparse gradients: only one non-zero gradient
    const grads1 = [_]f64{ 1.0, 0.0, 0.0 };
    try opt.update(&params, &grads1);

    // Only first parameter should have changed significantly from gradient
    // (all will change due to weight decay, but first changes more)
    const p1_change = @abs(params[0] - 1.0);
    const p2_change = @abs(params[1] - 2.0);
    try testing.expect(p1_change > p2_change);
}

test "AdamW: reset functionality" {
    const allocator = testing.allocator;

    var opt = try AdamW(f64).init(allocator, 2, .{});
    defer opt.deinit();

    var params = [_]f64{ 1.0, 2.0 };
    const grads = [_]f64{ 0.5, -0.5 };

    // Run some updates
    try opt.update(&params, &grads);
    try opt.update(&params, &grads);

    try testing.expect(opt.timestep > 0);
    try testing.expect(opt.momentum[0] != 0);
    try testing.expect(opt.velocity[0] != 0);

    // Reset
    opt.reset();

    try expectEqual(@as(usize, 0), opt.timestep);
    try expectEqual(@as(f64, 0), opt.momentum[0]);
    try expectEqual(@as(f64, 0), opt.momentum[1]);
    try expectEqual(@as(f64, 0), opt.velocity[0]);
    try expectEqual(@as(f64, 0), opt.velocity[1]);
}

test "AdamW: f32 support" {
    const allocator = testing.allocator;

    var opt = try AdamW(f32).init(allocator, 2, .{});
    defer opt.deinit();

    var params = [_]f32{ 1.0, -1.0 };
    const grads = [_]f32{ 0.5, -0.5 };

    try opt.update(&params, &grads);

    // Parameters should have changed
    try testing.expect(params[0] != 1.0);
    try testing.expect(params[1] != -1.0);
}

test "AdamW: large scale" {
    const allocator = testing.allocator;

    // 100-dimensional optimization
    var opt = try AdamW(f64).init(allocator, 100, .{});
    defer opt.deinit();

    const params = try allocator.alloc(f64, 100);
    defer allocator.free(params);
    @memset(params, 1.0);

    const grads = try allocator.alloc(f64, 100);
    defer allocator.free(grads);

    // Random gradients
    for (grads, 0..) |*g, i| {
        g.* = @as(f64, @floatFromInt(i)) / 100.0 - 0.5;
    }

    // Should handle large number of parameters
    try opt.update(params, grads);

    // All parameters should have changed
    var all_changed = true;
    for (params) |p| {
        if (p == 1.0) all_changed = false;
    }
    try testing.expect(all_changed);
}

test "AdamW: empty parameters error" {
    const allocator = testing.allocator;

    var opt = try AdamW(f64).init(allocator, 1, .{});
    defer opt.deinit();

    const params: []f64 = &[_]f64{};
    const grads: []const f64 = &[_]f64{};

    try expectError(error.EmptyParameters, opt.update(params, grads));
}

test "AdamW: gradient length mismatch" {
    const allocator = testing.allocator;

    var opt = try AdamW(f64).init(allocator, 2, .{});
    defer opt.deinit();

    var params = [_]f64{ 1.0, 2.0 };
    const grads = [_]f64{0.5}; // Wrong length

    try expectError(error.LengthMismatch, opt.update(&params, &grads));
}

test "AdamW: invalid learning rate" {
    const allocator = testing.allocator;

    const config = AdamW(f64).Config{ .learning_rate = -0.01 };
    try expectError(error.InvalidLearningRate, AdamW(f64).init(allocator, 1, config));
}

test "AdamW: invalid beta1" {
    const allocator = testing.allocator;

    const config = AdamW(f64).Config{ .beta1 = 1.5 };
    try expectError(error.InvalidBeta1, AdamW(f64).init(allocator, 1, config));
}

test "AdamW: invalid beta2" {
    const allocator = testing.allocator;

    const config = AdamW(f64).Config{ .beta2 = -0.1 };
    try expectError(error.InvalidBeta2, AdamW(f64).init(allocator, 1, config));
}

test "AdamW: invalid epsilon" {
    const allocator = testing.allocator;

    const config = AdamW(f64).Config{ .epsilon = 0.0 };
    try expectError(error.InvalidEpsilon, AdamW(f64).init(allocator, 1, config));
}

test "AdamW: invalid weight decay" {
    const allocator = testing.allocator;

    const config = AdamW(f64).Config{ .weight_decay = -0.1 };
    try expectError(error.InvalidWeightDecay, AdamW(f64).init(allocator, 1, config));
}
