const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// RMSprop (Root Mean Square Propagation) optimizer
///
/// Adaptive learning rate optimizer that divides learning rate by exponentially weighted
/// moving average of squared gradients. Addresses Adagrad's aggressive learning rate decay.
///
/// Algorithm:
/// 1. Compute gradient g_t = ∇f(θ_t)
/// 2. Accumulate squared gradient: v_t = β × v_{t-1} + (1 - β) × g_t²
/// 3. Update parameters: θ_t = θ_{t-1} - α / (√v_t + ε) × g_t
///
/// With momentum (optional):
/// - m_t = μ × m_{t-1} - α / (√v_t + ε) × g_t
/// - θ_t = θ_{t-1} + m_t
///
/// Key features:
/// - Adaptive per-parameter learning rates
/// - Moving average of squared gradients (unlike Adagrad's cumulative sum)
/// - Prevents learning rate from becoming infinitesimally small
/// - Works well with non-stationary objectives
/// - Often performs better than Adam for RNNs
///
/// Time complexity: O(n) per update where n = number of parameters
/// Space complexity: O(n) for squared gradient accumulator (+ O(n) if using momentum)
///
/// Use cases:
/// - Recurrent Neural Networks (RNNs, LSTMs, GRUs)
/// - Non-stationary objectives (online learning, reinforcement learning)
/// - When Adam's bias correction is not needed
/// - Mini-batch training with noisy gradients
///
/// Trade-offs:
/// - vs Adagrad: doesn't accumulate all gradients, prevents learning rate collapse
/// - vs Adam: simpler (no bias correction), often better for RNNs
/// - vs SGD: adaptive rates reduce need for learning rate tuning
///
/// References:
/// - Tieleman & Hinton (2012): "Lecture 6.5 - RMSprop" (Coursera Neural Networks course)
/// - Hinton et al. (2012): "Neural Networks for Machine Learning"
pub fn RMSprop(comptime T: type) type {
    if (T != f32 and T != f64) {
        @compileError("RMSprop only supports f32 and f64 types");
    }

    return struct {
        allocator: Allocator,
        learning_rate: T,
        beta: T, // decay rate for squared gradients
        epsilon: T, // numerical stability constant
        momentum: T, // optional momentum parameter
        centered: bool, // centered variant subtracts mean gradient
        weight_decay: T, // L2 regularization

        // State
        v: []T, // squared gradient accumulator
        m: ?[]T, // momentum buffer (allocated only if momentum > 0)
        g: ?[]T, // gradient mean (allocated only if centered=true)

        const Self = @This();

        /// Configuration for RMSprop optimizer
        pub const Config = struct {
            learning_rate: T = 0.01,
            beta: T = 0.9, // typical: 0.9-0.999
            epsilon: T = 1e-8,
            momentum: T = 0.0, // 0 = no momentum, typical: 0.9
            centered: bool = false, // centered RMSprop variant
            weight_decay: T = 0.0, // L2 penalty
        };

        /// Initialize RMSprop optimizer
        ///
        /// Time: O(n) | Space: O(n) to O(3n) depending on momentum and centered
        pub fn init(allocator: Allocator, num_params: usize, config: Config) !Self {
            if (num_params == 0) {
                return error.EmptyParameters;
            }
            if (config.learning_rate <= 0) {
                return error.InvalidLearningRate;
            }
            if (config.beta < 0 or config.beta >= 1) {
                return error.InvalidBeta;
            }
            if (config.epsilon <= 0) {
                return error.InvalidEpsilon;
            }
            if (config.momentum < 0 or config.momentum >= 1) {
                return error.InvalidMomentum;
            }
            if (config.weight_decay < 0) {
                return error.InvalidWeightDecay;
            }

            const v = try allocator.alloc(T, num_params);
            errdefer allocator.free(v);
            @memset(v, 0);

            // Allocate momentum buffer only if needed
            const m = if (config.momentum > 0)
                try allocator.alloc(T, num_params)
            else
                null;
            errdefer if (m) |buf| allocator.free(buf);
            if (m) |buf| @memset(buf, 0);

            // Allocate gradient mean buffer only if centered
            const g = if (config.centered)
                try allocator.alloc(T, num_params)
            else
                null;
            errdefer if (g) |buf| allocator.free(buf);
            if (g) |buf| @memset(buf, 0);

            return Self{
                .allocator = allocator,
                .learning_rate = config.learning_rate,
                .beta = config.beta,
                .epsilon = config.epsilon,
                .momentum = config.momentum,
                .centered = config.centered,
                .weight_decay = config.weight_decay,
                .v = v,
                .m = m,
                .g = g,
            };
        }

        /// Free optimizer resources
        ///
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.v);
            if (self.m) |buf| self.allocator.free(buf);
            if (self.g) |buf| self.allocator.free(buf);
        }

        /// Perform single optimization step
        ///
        /// params: parameter array to update (modified in-place)
        /// gradients: gradient array (must match params length)
        ///
        /// Algorithm:
        /// 1. Apply weight decay: g = g + λ × θ
        /// 2. Update squared gradient: v = β × v + (1-β) × g²
        /// 3. Centered variant: mean = β × mean + (1-β) × g, v_centered = v - mean²
        /// 4. Compute adaptive learning rate: lr_adapted = α / (√v + ε)
        /// 5. With momentum: m = μ × m - lr_adapted × g, θ += m
        /// 6. Without momentum: θ -= lr_adapted × g
        ///
        /// Time: O(n) | Space: O(1)
        pub fn step(self: *Self, params: []T, gradients: []const T) !void {
            if (params.len != self.v.len) {
                return error.ParameterLengthMismatch;
            }
            if (gradients.len != self.v.len) {
                return error.GradientLengthMismatch;
            }

            const n = params.len;
            const one_minus_beta = 1.0 - self.beta;

            for (0..n) |i| {
                var g = gradients[i];

                // Weight decay (L2 regularization)
                if (self.weight_decay > 0) {
                    g += self.weight_decay * params[i];
                }

                // Update squared gradient accumulator
                self.v[i] = self.beta * self.v[i] + one_minus_beta * g * g;

                // Centered variant: subtract mean gradient squared
                var v_centered = self.v[i];
                if (self.centered) {
                    const g_buf = self.g.?;
                    g_buf[i] = self.beta * g_buf[i] + one_minus_beta * g;
                    v_centered = self.v[i] - g_buf[i] * g_buf[i];
                    // Numerical stability: ensure non-negative
                    v_centered = @max(v_centered, 0.0);
                }

                // Compute adaptive learning rate
                const lr_adapted = self.learning_rate / (@sqrt(v_centered) + self.epsilon);

                // Update with or without momentum
                if (self.momentum > 0) {
                    const m_buf = self.m.?;
                    m_buf[i] = self.momentum * m_buf[i] - lr_adapted * g;
                    params[i] += m_buf[i];
                } else {
                    params[i] -= lr_adapted * g;
                }
            }
        }

        /// Reset optimizer state (zero out accumulators)
        ///
        /// Time: O(n) | Space: O(1)
        pub fn reset(self: *Self) void {
            @memset(self.v, 0);
            if (self.m) |buf| @memset(buf, 0);
            if (self.g) |buf| @memset(buf, 0);
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "RMSprop: initialization" {
    const allocator = testing.allocator;

    var opt = try RMSprop(f64).init(allocator, 10, .{});
    defer opt.deinit();

    try testing.expectEqual(@as(f64, 0.01), opt.learning_rate);
    try testing.expectEqual(@as(f64, 0.9), opt.beta);
    try testing.expectEqual(@as(f64, 1e-8), opt.epsilon);
    try testing.expectEqual(@as(usize, 10), opt.v.len);

    // All accumulators should be zero
    for (opt.v) |val| {
        try testing.expectEqual(@as(f64, 0.0), val);
    }
}

test "RMSprop: custom config" {
    const allocator = testing.allocator;

    var opt = try RMSprop(f32).init(allocator, 5, .{
        .learning_rate = 0.001,
        .beta = 0.99,
        .epsilon = 1e-7,
        .momentum = 0.9,
        .centered = true,
        .weight_decay = 0.0001,
    });
    defer opt.deinit();

    try testing.expectEqual(@as(f32, 0.001), opt.learning_rate);
    try testing.expectEqual(@as(f32, 0.99), opt.beta);
    try testing.expectEqual(@as(f32, 0.9), opt.momentum);
    try testing.expect(opt.centered);
    try testing.expect(opt.m != null); // momentum buffer allocated
    try testing.expect(opt.g != null); // centered buffer allocated
}

test "RMSprop: simple quadratic optimization" {
    const allocator = testing.allocator;

    // Minimize f(x) = x^2, gradient = 2x
    var opt = try RMSprop(f64).init(allocator, 1, .{
        .learning_rate = 0.1,
        .beta = 0.95,
    });
    defer opt.deinit();

    var params = [_]f64{5.0};
    const initial = @abs(params[0]);

    // Run several steps
    for (0..200) |_| {
        const grad = [_]f64{2.0 * params[0]};
        try opt.step(&params, &grad);
    }

    // Should significantly reduce magnitude
    try testing.expect(@abs(params[0]) < 0.5 * initial);
}

test "RMSprop: multivariate quadratic" {
    const allocator = testing.allocator;

    // Minimize f(x,y) = x^2 + 4y^2
    var opt = try RMSprop(f64).init(allocator, 2, .{
        .learning_rate = 0.1,
        .beta = 0.95,
    });
    defer opt.deinit();

    var params = [_]f64{ 3.0, 2.0 };

    const initial_dist = @sqrt(params[0] * params[0] + params[1] * params[1]);

    for (0..300) |_| {
        const grad = [_]f64{
            2.0 * params[0], // ∂f/∂x = 2x
            8.0 * params[1], // ∂f/∂y = 8y
        };
        try opt.step(&params, &grad);
    }

    const final_dist = @sqrt(params[0] * params[0] + params[1] * params[1]);

    // Should significantly reduce distance from origin
    try testing.expect(final_dist < 0.5 * initial_dist);
}

test "RMSprop: with momentum" {
    const allocator = testing.allocator;

    var opt = try RMSprop(f64).init(allocator, 1, .{
        .learning_rate = 0.1,
        .momentum = 0.9,
        .beta = 0.95,
    });
    defer opt.deinit();

    var params = [_]f64{5.0};
    const initial = @abs(params[0]);

    // Momentum should accelerate convergence
    for (0..200) |_| {
        const grad = [_]f64{2.0 * params[0]};
        try opt.step(&params, &grad);
    }

    // Should significantly reduce parameter magnitude
    try testing.expect(@abs(params[0]) < 0.5 * initial);
}

test "RMSprop: centered variant" {
    const allocator = testing.allocator;

    var opt = try RMSprop(f64).init(allocator, 1, .{
        .learning_rate = 0.1,
        .beta = 0.95,
        .centered = true,
    });
    defer opt.deinit();

    var params = [_]f64{5.0};
    const initial = @abs(params[0]);

    for (0..200) |_| {
        const grad = [_]f64{2.0 * params[0]};
        try opt.step(&params, &grad);
    }

    // Centered variant should reduce parameter magnitude
    try testing.expect(@abs(params[0]) < 0.5 * initial);
    try testing.expect(opt.g != null); // gradient mean buffer exists
}

test "RMSprop: weight decay" {
    const allocator = testing.allocator;

    var opt = try RMSprop(f64).init(allocator, 1, .{
        .learning_rate = 0.5,
        .weight_decay = 0.01,
    });
    defer opt.deinit();

    var params = [_]f64{5.0};

    for (0..100) |_| {
        const grad = [_]f64{2.0 * params[0]};
        try opt.step(&params, &grad);
    }

    // Weight decay adds regularization
    try testing.expect(@abs(params[0]) < 0.2);
}

test "RMSprop: adaptive learning rates" {
    const allocator = testing.allocator;

    var opt = try RMSprop(f64).init(allocator, 2, .{
        .learning_rate = 0.1,
    });
    defer opt.deinit();

    var params = [_]f64{ 1.0, 1.0 };

    // Different gradient magnitudes
    const grad1 = [_]f64{ 0.1, 10.0 }; // small vs large gradient

    try opt.step(&params, &grad1);

    // Second gradient step
    try opt.step(&params, &grad1);

    // RMSprop should adapt: large gradients get smaller effective LR
    // Both v[0] and v[1] accumulate, but v[1] >> v[0]
    try testing.expect(opt.v[1] > opt.v[0]);
}

test "RMSprop: handles sparse gradients" {
    const allocator = testing.allocator;

    var opt = try RMSprop(f64).init(allocator, 3, .{
        .learning_rate = 0.1,
    });
    defer opt.deinit();

    var params = [_]f64{ 1.0, 1.0, 1.0 };

    // Sparse gradient: only first parameter has gradient
    const grad_sparse = [_]f64{ 1.0, 0.0, 0.0 };

    for (0..10) |_| {
        try opt.step(&params, &grad_sparse);
    }

    // First parameter should change, others mostly unchanged
    try testing.expect(@abs(params[0] - 1.0) > 0.1);
    try testing.expect(@abs(params[1] - 1.0) < 0.001);
    try testing.expect(@abs(params[2] - 1.0) < 0.001);
}

test "RMSprop: reset functionality" {
    const allocator = testing.allocator;

    var opt = try RMSprop(f64).init(allocator, 2, .{
        .momentum = 0.9,
        .centered = true,
    });
    defer opt.deinit();

    var params = [_]f64{ 1.0, 1.0 };
    const grad = [_]f64{ 0.5, 0.5 };

    // Take a step to populate state
    try opt.step(&params, &grad);

    // State should be non-zero
    try testing.expect(opt.v[0] > 0);
    try testing.expect(opt.m.?[0] != 0);
    try testing.expect(opt.g.?[0] > 0);

    // Reset
    opt.reset();

    // All accumulators should be zero
    for (opt.v) |val| try testing.expectEqual(@as(f64, 0.0), val);
    for (opt.m.?) |val| try testing.expectEqual(@as(f64, 0.0), val);
    for (opt.g.?) |val| try testing.expectEqual(@as(f64, 0.0), val);
}

test "RMSprop: f32 support" {
    const allocator = testing.allocator;

    var opt = try RMSprop(f32).init(allocator, 1, .{
        .learning_rate = 0.1,
        .beta = 0.95,
    });
    defer opt.deinit();

    var params = [_]f32{5.0};
    const initial = @abs(params[0]);

    for (0..200) |_| {
        const grad = [_]f32{2.0 * params[0]};
        try opt.step(&params, &grad);
    }

    try testing.expect(@abs(params[0]) < 0.5 * initial);
}

test "RMSprop: large scale" {
    const allocator = testing.allocator;

    const n = 100;
    var opt = try RMSprop(f64).init(allocator, n, .{
        .learning_rate = 0.1,
        .beta = 0.95,
    });
    defer opt.deinit();

    const params = try allocator.alloc(f64, n);
    defer allocator.free(params);
    @memset(params, 2.0);

    const gradients = try allocator.alloc(f64, n);
    defer allocator.free(gradients);

    // Track initial sum for comparison
    var initial_sum: f64 = 0;
    for (params) |p| initial_sum += @abs(p);

    for (0..300) |_| {
        // Gradient = 2 * param
        for (0..n) |i| {
            gradients[i] = 2.0 * params[i];
        }
        try opt.step(params, gradients);
    }

    // Parameters should significantly reduce
    var final_sum: f64 = 0;
    for (params) |p| final_sum += @abs(p);

    try testing.expect(final_sum < 0.5 * initial_sum);
}

test "RMSprop: comparison with vanilla SGD behavior" {
    const allocator = testing.allocator;

    // RMSprop with beta=0 behaves like SGD with inverse gradient scaling
    var opt = try RMSprop(f64).init(allocator, 1, .{
        .learning_rate = 0.1,
        .beta = 0.0, // no exponential averaging
    });
    defer opt.deinit();

    var params = [_]f64{1.0};
    const grad = [_]f64{0.5};

    try opt.step(&params, &grad);

    // With beta=0: v = g^2, lr_adapted = α / (√g^2 + ε) ≈ α / |g|
    // So update ≈ -α / |g| × g = -α × sign(g)
    try testing.expect(params[0] < 1.0); // should decrease
}

test "RMSprop: error - empty parameters" {
    const allocator = testing.allocator;
    try testing.expectError(error.EmptyParameters, RMSprop(f64).init(allocator, 0, .{}));
}

test "RMSprop: error - invalid learning rate" {
    const allocator = testing.allocator;
    try testing.expectError(error.InvalidLearningRate, RMSprop(f64).init(allocator, 10, .{
        .learning_rate = 0.0,
    }));
    try testing.expectError(error.InvalidLearningRate, RMSprop(f64).init(allocator, 10, .{
        .learning_rate = -0.01,
    }));
}

test "RMSprop: error - invalid beta" {
    const allocator = testing.allocator;
    try testing.expectError(error.InvalidBeta, RMSprop(f64).init(allocator, 10, .{
        .beta = 1.0,
    }));
    try testing.expectError(error.InvalidBeta, RMSprop(f64).init(allocator, 10, .{
        .beta = -0.1,
    }));
}

test "RMSprop: error - gradient length mismatch" {
    const allocator = testing.allocator;

    var opt = try RMSprop(f64).init(allocator, 5, .{});
    defer opt.deinit();

    var params = [_]f64{ 1.0, 1.0, 1.0, 1.0, 1.0 };
    const gradients = [_]f64{ 0.1, 0.1 }; // wrong length

    try testing.expectError(error.GradientLengthMismatch, opt.step(&params, &gradients));
}

test "RMSprop: memory safety" {
    const allocator = testing.allocator;

    var opt = try RMSprop(f64).init(allocator, 10, .{
        .momentum = 0.9,
        .centered = true,
    });
    defer opt.deinit();

    // Multiple allocations should not leak
    const params = try allocator.alloc(f64, 10);
    defer allocator.free(params);
    @memset(params, 1.0);

    const gradients = try allocator.alloc(f64, 10);
    defer allocator.free(gradients);
    @memset(gradients, 0.1);

    for (0..10) |_| {
        try opt.step(params, gradients);
    }
}
