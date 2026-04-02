/// LARS (Layer-wise Adaptive Rate Scaling) Optimizer
///
/// LARS adapts the learning rate for each layer based on the ratio of parameter norm to gradient norm.
/// This allows stable training with very large batch sizes (e.g., batch size 32K for ImageNet).
///
/// Reference: You et al. (2017) "Large Batch Training of Convolutional Networks" (arXiv:1708.03888)
///
/// Algorithm:
/// 1. For each layer: Compute trust ratio r = η × ||θ||₂ / (||∇θ||₂ + λ||θ||₂)
/// 2. Adapted learning rate: lr_adapted = lr × r
/// 3. Update with momentum: v_t = μ × v_{t-1} + lr_adapted × (∇θ + λθ)
/// 4. Parameter update: θ_t = θ_{t-1} - v_t
///
/// Key properties:
/// - Enables large-batch training (batch sizes > 8K) without accuracy loss
/// - Layer-wise adaptation prevents gradient explosion/vanishing across layers
/// - Trust coefficient η controls adaptation strength (typical: 0.001)
/// - Weight decay λ is decoupled (applied before trust ratio)
/// - Momentum μ helps smooth updates (typical: 0.9)
///
/// Typical hyperparameters:
/// - learning_rate: 1.0-10.0 (much larger than SGD due to layer-wise scaling)
/// - trust_coef: 0.001 (η in paper)
/// - momentum: 0.9 (standard momentum coefficient)
/// - weight_decay: 0.0001 (L2 penalty)
/// - exclude_from_layer_adaptation: bias/batchnorm params often excluded
///
/// Time: O(n) per update where n = number of parameters
/// Space: O(n) for momentum vectors
///
/// Use cases:
/// - Large-batch training (batch sizes > 8K)
/// - Distributed training across many GPUs
/// - ImageNet training with reduced time
/// - When training time is critical and hardware allows large batches
/// - Foundation for LAMB (combines LARS with Adam)

const std = @import("std");
const Allocator = std.mem.Allocator;

/// LARS optimizer configuration
pub const LARSConfig = struct {
    /// Base learning rate (much larger than SGD, e.g., 1-10)
    learning_rate: f64 = 1.0,
    /// Trust coefficient (η in paper, controls layer-wise adaptation)
    trust_coef: f64 = 0.001,
    /// Momentum coefficient (μ, smooths updates)
    momentum: f64 = 0.9,
    /// Weight decay coefficient (L2 penalty)
    weight_decay: f64 = 0.0001,
    /// Epsilon for numerical stability
    epsilon: f64 = 1e-8,

    /// Validate configuration parameters
    pub fn validate(self: LARSConfig) !void {
        if (self.learning_rate <= 0) return error.InvalidLearningRate;
        if (self.trust_coef <= 0 or self.trust_coef >= 1) return error.InvalidTrustCoef;
        if (self.momentum < 0 or self.momentum >= 1) return error.InvalidMomentum;
        if (self.weight_decay < 0) return error.InvalidWeightDecay;
        if (self.epsilon <= 0) return error.InvalidEpsilon;
    }
};

/// LARS optimizer
pub fn LARS(comptime T: type) type {
    return struct {
        config: LARSConfig,
        velocity: []T,
        allocator: Allocator,

        const Self = @This();

        /// Initialize LARS optimizer
        ///
        /// Time: O(n)
        /// Space: O(n)
        pub fn init(allocator: Allocator, n_params: usize, config: LARSConfig) !Self {
            try config.validate();

            const velocity = try allocator.alloc(T, n_params);
            @memset(velocity, 0);

            return Self{
                .config = config,
                .velocity = velocity,
                .allocator = allocator,
            };
        }

        /// Free optimizer resources
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.velocity);
        }

        /// Perform one optimization step with LARS
        ///
        /// Time: O(n)
        /// Space: O(1)
        pub fn step(self: *Self, params: []T, gradients: []const T) !void {
            if (params.len != gradients.len) return error.LengthMismatch;
            if (params.len != self.velocity.len) return error.LengthMismatch;
            if (params.len == 0) return error.EmptyParameters;

            const lr = @as(T, @floatCast(self.config.learning_rate));
            const eta = @as(T, @floatCast(self.config.trust_coef));
            const mu = @as(T, @floatCast(self.config.momentum));
            const lambda = @as(T, @floatCast(self.config.weight_decay));
            const eps = @as(T, @floatCast(self.config.epsilon));

            // Compute parameter norm: ||θ||₂
            var param_norm: T = 0;
            for (params) |p| {
                param_norm += p * p;
            }
            param_norm = @sqrt(param_norm);

            // Compute gradient norm with weight decay: ||∇θ + λθ||₂
            var grad_norm: T = 0;
            for (params, gradients) |p, g| {
                const g_wd = g + lambda * p;
                grad_norm += g_wd * g_wd;
            }
            grad_norm = @sqrt(grad_norm);

            // Compute trust ratio: r = η × ||θ||₂ / (||∇θ + λθ||₂ + ε)
            const trust_ratio = eta * param_norm / (grad_norm + eps);

            // Adapted learning rate: lr_adapted = lr × r
            const lr_adapted = lr * trust_ratio;

            // Update with momentum: v_t = μ × v_{t-1} + lr_adapted × (∇θ + λθ)
            for (params, gradients, self.velocity) |p, g, *v| {
                const g_wd = g + lambda * p;
                v.* = mu * v.* + lr_adapted * g_wd;
            }

            // Parameter update: θ_t = θ_{t-1} - v_t
            for (params, self.velocity) |*p, v| {
                p.* -= v;
            }
        }

        /// Reset optimizer state (clear momentum)
        ///
        /// Time: O(n)
        /// Space: O(1)
        pub fn reset(self: *Self) void {
            @memset(self.velocity, 0);
        }

        /// Get current velocity (for debugging/inspection)
        pub fn getVelocity(self: *const Self) []const T {
            return self.velocity;
        }
    };
}

// ===== Tests =====

const testing = std.testing;
const expect = testing.expect;
const expectEqual = testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;

test "LARS: initialization with default config" {
    const allocator = testing.allocator;
    const config = LARSConfig{};

    var optimizer = try LARS(f64).init(allocator, 5, config);
    defer optimizer.deinit();

    try expectEqual(@as(usize, 5), optimizer.velocity.len);
    for (optimizer.velocity) |v| {
        try expectEqual(@as(f64, 0), v);
    }
}

test "LARS: initialization with custom config" {
    const allocator = testing.allocator;
    const config = LARSConfig{
        .learning_rate = 5.0,
        .trust_coef = 0.002,
        .momentum = 0.95,
        .weight_decay = 0.0005,
    };

    var optimizer = try LARS(f64).init(allocator, 3, config);
    defer optimizer.deinit();

    try expectEqual(@as(f64, 5.0), optimizer.config.learning_rate);
    try expectEqual(@as(f64, 0.002), optimizer.config.trust_coef);
}

test "LARS: simple quadratic optimization" {
    const allocator = testing.allocator;
    // Minimize f(x) = x², gradient = 2x
    const config = LARSConfig{
        .learning_rate = 1.0,
        .trust_coef = 0.01, // Higher trust coef for faster convergence
        .momentum = 0.9, // Add momentum to help convergence
        .weight_decay = 0.0,
    };

    var optimizer = try LARS(f64).init(allocator, 1, config);
    defer optimizer.deinit();

    var params = [_]f64{10.0};

    // Run optimization
    var i: usize = 0;
    while (i < 200) : (i += 1) {
        const gradients = [_]f64{2.0 * params[0]};
        try optimizer.step(&params, &gradients);
    }

    // Should converge near zero
    try expect(@abs(params[0]) < 1.0);
}

test "LARS: multivariate quadratic" {
    const allocator = testing.allocator;
    // Minimize f(x,y) = x² + y², gradients = (2x, 2y)
    const config = LARSConfig{
        .learning_rate = 1.0,
        .trust_coef = 0.01, // Higher trust coef for faster convergence
        .momentum = 0.9, // Add momentum to help convergence
        .weight_decay = 0.0,
    };

    var optimizer = try LARS(f64).init(allocator, 2, config);
    defer optimizer.deinit();

    var params = [_]f64{ 5.0, -3.0 };

    var i: usize = 0;
    while (i < 200) : (i += 1) {
        const gradients = [_]f64{ 2.0 * params[0], 2.0 * params[1] };
        try optimizer.step(&params, &gradients);
    }

    try expect(@abs(params[0]) < 0.5);
    try expect(@abs(params[1]) < 0.5);
}

test "LARS: trust ratio computation" {
    const allocator = testing.allocator;
    const config = LARSConfig{
        .learning_rate = 1.0,
        .trust_coef = 0.001,
        .momentum = 0.0,
        .weight_decay = 0.0,
        .epsilon = 1e-8,
    };

    var optimizer = try LARS(f64).init(allocator, 2, config);
    defer optimizer.deinit();

    var params = [_]f64{ 3.0, 4.0 }; // ||θ||₂ = 5.0
    const gradients = [_]f64{ 6.0, 8.0 }; // ||∇θ||₂ = 10.0

    const old_params = params;
    try optimizer.step(&params, &gradients);

    // Trust ratio: r = η × ||θ||₂ / ||∇θ||₂ = 0.001 × 5 / 10 = 0.0005
    // lr_adapted = 1.0 × 0.0005 = 0.0005
    // Δθ = lr_adapted × ∇θ = [0.003, 0.004]
    const expected_delta_0 = 0.0005 * 6.0;
    const expected_delta_1 = 0.0005 * 8.0;

    try expectApproxEqAbs(old_params[0] - expected_delta_0, params[0], 1e-6);
    try expectApproxEqAbs(old_params[1] - expected_delta_1, params[1], 1e-6);
}

test "LARS: momentum accumulation" {
    const allocator = testing.allocator;
    const config = LARSConfig{
        .learning_rate = 1.0,
        .trust_coef = 0.01,
        .momentum = 0.9,
        .weight_decay = 0.0,
    };

    var optimizer = try LARS(f64).init(allocator, 1, config);
    defer optimizer.deinit();

    var params = [_]f64{5.0};
    const gradients = [_]f64{1.0};

    // First step: v = 0.9*0 + lr_adapted*1 = lr_adapted
    try optimizer.step(&params, &gradients);
    const velocity_1 = optimizer.velocity[0];

    // Second step: v = 0.9*v_1 + lr_adapted
    try optimizer.step(&params, &gradients);
    const velocity_2 = optimizer.velocity[0];

    // Velocity should be accumulating (v_2 > v_1)
    try expect(velocity_2 > velocity_1);
}

test "LARS: weight decay effect" {
    const allocator = testing.allocator;
    const config_no_wd = LARSConfig{
        .learning_rate = 1.0,
        .trust_coef = 0.001,
        .momentum = 0.0,
        .weight_decay = 0.0,
    };
    const config_with_wd = LARSConfig{
        .learning_rate = 1.0,
        .trust_coef = 0.001,
        .momentum = 0.0,
        .weight_decay = 0.01,
    };

    var opt_no_wd = try LARS(f64).init(allocator, 1, config_no_wd);
    defer opt_no_wd.deinit();

    var opt_with_wd = try LARS(f64).init(allocator, 1, config_with_wd);
    defer opt_with_wd.deinit();

    var params_no_wd = [_]f64{10.0};
    var params_with_wd = [_]f64{10.0};
    const gradients = [_]f64{1.0};

    try opt_no_wd.step(&params_no_wd, &gradients);
    try opt_with_wd.step(&params_with_wd, &gradients);

    // With weight decay, parameters should shrink more
    try expect(params_with_wd[0] < params_no_wd[0]);
}

test "LARS: large batch training simulation" {
    const allocator = testing.allocator;
    // Simulate large-batch ImageNet-style training
    const config = LARSConfig{
        .learning_rate = 10.0, // Large LR typical for LARS
        .trust_coef = 0.001,
        .momentum = 0.9,
        .weight_decay = 0.0001,
    };

    var optimizer = try LARS(f64).init(allocator, 10, config);
    defer optimizer.deinit();

    var params: [10]f64 = undefined;
    for (&params, 0..) |*p, i| {
        p.* = @as(f64, @floatFromInt(i)) - 5.0; // Initialize around zero
    }

    // Simulate 50 large-batch updates
    var step_count: usize = 0;
    while (step_count < 50) : (step_count += 1) {
        var gradients: [10]f64 = undefined;
        for (params, &gradients) |p, *g| {
            g.* = 2.0 * p; // Quadratic objective
        }
        try optimizer.step(&params, &gradients);
    }

    // Should converge towards zero
    for (params) |p| {
        try expect(@abs(p) < 1.0);
    }
}

test "LARS: sparse gradients" {
    const allocator = testing.allocator;
    const config = LARSConfig{
        .learning_rate = 1.0,
        .trust_coef = 0.001,
        .momentum = 0.0,
        .weight_decay = 0.0,
    };

    var optimizer = try LARS(f64).init(allocator, 5, config);
    defer optimizer.deinit();

    var params = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const gradients = [_]f64{ 1.0, 0.0, 0.0, 0.0, 1.0 }; // Sparse

    try optimizer.step(&params, &gradients);

    // Non-zero gradient parameters should have moved
    try expect(params[0] != 1.0);
    try expect(params[4] != 5.0);
}

test "LARS: reset functionality" {
    const allocator = testing.allocator;
    const config = LARSConfig{
        .learning_rate = 1.0,
        .trust_coef = 0.001,
        .momentum = 0.9,
        .weight_decay = 0.0,
    };

    var optimizer = try LARS(f64).init(allocator, 3, config);
    defer optimizer.deinit();

    var params = [_]f64{ 1.0, 2.0, 3.0 };
    const gradients = [_]f64{ 0.5, 0.5, 0.5 };

    // Accumulate momentum
    try optimizer.step(&params, &gradients);
    try optimizer.step(&params, &gradients);

    // Velocity should be non-zero
    try expect(optimizer.velocity[0] != 0);

    // Reset should clear velocity
    optimizer.reset();
    for (optimizer.velocity) |v| {
        try expectEqual(@as(f64, 0), v);
    }
}

test "LARS: f32 support" {
    const allocator = testing.allocator;
    const config = LARSConfig{
        .learning_rate = 1.0,
        .trust_coef = 0.001,
        .momentum = 0.0,
        .weight_decay = 0.0,
    };

    var optimizer = try LARS(f32).init(allocator, 2, config);
    defer optimizer.deinit();

    var params = [_]f32{ 5.0, -3.0 };
    const gradients = [_]f32{ 10.0, -6.0 };

    try optimizer.step(&params, &gradients);

    // Should move in the right direction
    try expect(params[0] < 5.0);
    try expect(params[1] > -3.0);
}

test "LARS: large scale (100 parameters)" {
    const allocator = testing.allocator;
    const config = LARSConfig{
        .learning_rate = 5.0,
        .trust_coef = 0.001,
        .momentum = 0.9,
        .weight_decay = 0.0001,
    };

    var optimizer = try LARS(f64).init(allocator, 100, config);
    defer optimizer.deinit();

    var params: [100]f64 = undefined;
    for (&params, 0..) |*p, i| {
        p.* = @as(f64, @floatFromInt(i % 10)) - 5.0;
    }

    var gradients: [100]f64 = undefined;
    for (params, &gradients) |p, *g| {
        g.* = 2.0 * p;
    }

    try optimizer.step(&params, &gradients);

    // All parameters should have moved
    var moved_count: usize = 0;
    for (params, 0..) |p, i| {
        const original = @as(f64, @floatFromInt(i % 10)) - 5.0;
        if (@abs(p - original) > 1e-10) moved_count += 1;
    }
    try expect(moved_count > 0);
}

test "LARS: error - empty parameters" {
    const allocator = testing.allocator;
    const config = LARSConfig{};

    var optimizer = try LARS(f64).init(allocator, 0, config);
    defer optimizer.deinit();

    var params: [0]f64 = .{};
    const gradients: [0]f64 = .{};

    try testing.expectError(error.EmptyParameters, optimizer.step(&params, &gradients));
}

test "LARS: error - mismatched lengths" {
    const allocator = testing.allocator;
    const config = LARSConfig{};

    var optimizer = try LARS(f64).init(allocator, 3, config);
    defer optimizer.deinit();

    var params = [_]f64{ 1.0, 2.0, 3.0 };
    const gradients = [_]f64{ 1.0, 2.0 };

    try testing.expectError(error.LengthMismatch, optimizer.step(&params, &gradients));
}

test "LARS: error - invalid learning rate" {
    const allocator = testing.allocator;
    const config = LARSConfig{ .learning_rate = -1.0 };

    try testing.expectError(error.InvalidLearningRate, LARS(f64).init(allocator, 3, config));
}

test "LARS: error - invalid trust coefficient" {
    const allocator = testing.allocator;
    const config1 = LARSConfig{ .trust_coef = 0.0 };
    const config2 = LARSConfig{ .trust_coef = 1.0 };

    try testing.expectError(error.InvalidTrustCoef, LARS(f64).init(allocator, 3, config1));
    try testing.expectError(error.InvalidTrustCoef, LARS(f64).init(allocator, 3, config2));
}

test "LARS: error - invalid momentum" {
    const allocator = testing.allocator;
    const config = LARSConfig{ .momentum = 1.0 };

    try testing.expectError(error.InvalidMomentum, LARS(f64).init(allocator, 3, config));
}

test "LARS: convergence with varying gradients" {
    const allocator = testing.allocator;
    const config = LARSConfig{
        .learning_rate = 2.0,
        .trust_coef = 0.001,
        .momentum = 0.9,
        .weight_decay = 0.0,
    };

    var optimizer = try LARS(f64).init(allocator, 2, config);
    defer optimizer.deinit();

    var params = [_]f64{ 10.0, 10.0 };

    // Simulate noisy gradient descent
    var step: usize = 0;
    while (step < 100) : (step += 1) {
        const noise = @as(f64, @floatFromInt(step % 3)) - 1.0;
        const gradients = [_]f64{ 2.0 * params[0] + noise, 2.0 * params[1] - noise };
        try optimizer.step(&params, &gradients);
    }

    // Should still converge despite noise
    try expect(@abs(params[0]) < 2.0);
    try expect(@abs(params[1]) < 2.0);
}

test "LARS: memory safety with testing.allocator" {
    const allocator = testing.allocator;
    const config = LARSConfig{};

    var optimizer = try LARS(f64).init(allocator, 10, config);
    defer optimizer.deinit();

    var params: [10]f64 = undefined;
    var gradients: [10]f64 = undefined;
    for (&params, &gradients, 0..) |*p, *g, i| {
        p.* = @as(f64, @floatFromInt(i));
        g.* = @as(f64, @floatFromInt(i % 3));
    }

    try optimizer.step(&params, &gradients);
    try optimizer.step(&params, &gradients);
    try optimizer.step(&params, &gradients);

    // testing.allocator will detect leaks on deinit
}
