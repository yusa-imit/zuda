const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// SGD (Stochastic Gradient Descent) optimizer with momentum
///
/// Classic optimization algorithm that updates parameters in the direction of negative gradients.
/// Supports optional momentum to accelerate convergence and dampen oscillations.
///
/// Algorithm (with momentum):
/// 1. Compute gradient g_t = ∇f(θ_t)
/// 2. Update velocity: v_t = μ × v_{t-1} - α × g_t
/// 3. Update parameters: θ_t = θ_{t-1} + v_t
///
/// Without momentum (μ = 0):
/// - θ_t = θ_{t-1} - α × g_t
///
/// With Nesterov momentum:
/// - Compute gradient at "lookahead" position: g_t = ∇f(θ_{t-1} + μ × v_{t-1})
/// - Update velocity: v_t = μ × v_{t-1} - α × g_t
/// - Update parameters: θ_t = θ_{t-1} + v_t
///
/// Key features:
/// - Simple and robust baseline optimizer
/// - Momentum accelerates convergence in relevant directions
/// - Dampens oscillations in high-curvature directions
/// - Nesterov variant provides "lookahead" for better gradients
/// - No adaptive learning rates (all parameters use same rate)
///
/// Time complexity: O(n) per update where n = number of parameters
/// Space complexity: O(n) for velocity vector (with momentum), O(1) otherwise
///
/// Use cases:
/// - Simple optimization problems with well-conditioned objectives
/// - Baseline for comparing against adaptive methods (Adam, RMSprop)
/// - Large-batch training (momentum helps smooth gradients)
/// - Convex optimization where theory is well-established
///
/// Trade-offs:
/// - vs Vanilla GD: momentum accelerates convergence, reduces oscillations
/// - vs Adam: simpler (no adaptive rates), requires more tuning (learning rate critical)
/// - vs RMSprop: no adaptive rates, better for stationary objectives
///
/// References:
/// - Polyak (1964): "Some methods of speeding up the convergence of iteration methods"
/// - Nesterov (1983): "A method for unconstrained convex minimization problem"
/// - Sutskever et al. (2013): "On the importance of initialization and momentum"
pub fn SGD(comptime T: type) type {
    return struct {
        allocator: Allocator,
        params: []T,
        velocity: ?[]T, // Only allocated if momentum > 0
        config: Config,

        const Self = @This();

        /// Configuration for SGD optimizer
        pub const Config = struct {
            /// Learning rate (step size)
            /// Common values: 0.1, 0.01, 0.001
            /// Requires careful tuning (too high = divergence, too low = slow)
            /// Time: O(1) | Space: O(1)
            learning_rate: T = 0.01,

            /// Momentum coefficient
            /// Common values: 0.9, 0.95, 0.99
            /// 0 = no momentum (vanilla SGD)
            /// Higher values = more smoothing, faster convergence
            /// Time: O(1) | Space: O(1)
            momentum: T = 0.0,

            /// Use Nesterov accelerated gradient
            /// Computes gradient at "lookahead" position for better updates
            /// Recommended: true when using momentum
            /// Time: O(1) | Space: O(1)
            nesterov: bool = false,

            /// Weight decay (L2 regularization)
            /// Penalizes large weights: θ_t = θ_t - α × λ × θ_t
            /// Common values: 0.0001, 0.00001
            /// 0 = no weight decay
            /// Time: O(1) | Space: O(1)
            weight_decay: T = 0.0,
        };

        /// Initialize SGD optimizer
        /// Time: O(n) if momentum > 0, O(1) otherwise | Space: O(n) if momentum > 0, O(1) otherwise
        pub fn init(allocator: Allocator, params: []T, config: Config) !Self {
            if (params.len == 0) return error.EmptyParameters;
            if (config.learning_rate <= 0) return error.InvalidLearningRate;
            if (config.momentum < 0 or config.momentum >= 1) return error.InvalidMomentum;
            if (config.weight_decay < 0) return error.InvalidWeightDecay;

            // Only allocate velocity if momentum is used
            const velocity = if (config.momentum > 0) blk: {
                const v = try allocator.alloc(T, params.len);
                @memset(v, 0.0);
                break :blk v;
            } else null;

            return Self{
                .allocator = allocator,
                .params = params,
                .velocity = velocity,
                .config = config,
            };
        }

        /// Free resources
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.velocity) |v| {
                self.allocator.free(v);
                self.velocity = null;
            }
        }

        /// Perform optimization step with given gradients
        /// Time: O(n) | Space: O(1)
        pub fn step(self: *Self, gradients: []const T) !void {
            if (gradients.len != self.params.len) {
                return error.GradientLengthMismatch;
            }

            if (self.config.momentum > 0) {
                // Momentum-based update
                if (self.velocity) |v| {
                    if (self.config.nesterov) {
                        // Nesterov Accelerated Gradient (NAG)
                        // First apply lookahead to params (conceptually)
                        // Then compute gradient at that position
                        // In practice: v_t = μ*v_{t-1} - α*g_t, θ_t = θ_{t-1} + μ*v_t - α*g_t
                        for (self.params, gradients, v, 0..) |*param, grad, *vel, i| {
                            // Apply weight decay
                            const wd_term = if (self.config.weight_decay > 0)
                                self.config.weight_decay * param.*
                            else
                                0.0;

                            // Update velocity with gradient and weight decay
                            vel.* = self.config.momentum * vel.* - self.config.learning_rate * (grad + wd_term);

                            // Nesterov: apply momentum twice (lookahead + update)
                            param.* = param.* + self.config.momentum * vel.* + vel.*;
                            _ = i;
                        }
                    } else {
                        // Standard momentum
                        for (self.params, gradients, v, 0..) |*param, grad, *vel, i| {
                            // Apply weight decay
                            const wd_term = if (self.config.weight_decay > 0)
                                self.config.weight_decay * param.*
                            else
                                0.0;

                            // Update velocity: v_t = μ*v_{t-1} - α*g_t
                            vel.* = self.config.momentum * vel.* - self.config.learning_rate * (grad + wd_term);

                            // Update parameters: θ_t = θ_{t-1} + v_t
                            param.* += vel.*;
                            _ = i;
                        }
                    }
                }
            } else {
                // Vanilla SGD (no momentum)
                for (self.params, gradients, 0..) |*param, grad, i| {
                    // Apply weight decay
                    const wd_term = if (self.config.weight_decay > 0)
                        self.config.weight_decay * param.*
                    else
                        0.0;

                    // Simple update: θ_t = θ_{t-1} - α*(g_t + λ*θ_t)
                    param.* -= self.config.learning_rate * (grad + wd_term);
                    _ = i;
                }
            }
        }

        /// Reset optimizer state (zero out velocity)
        /// Time: O(n) if momentum > 0, O(1) otherwise | Space: O(1)
        pub fn reset(self: *Self) void {
            if (self.velocity) |v| {
                @memset(v, 0.0);
            }
        }

        /// Get current velocity (for momentum-based variants)
        /// Time: O(1) | Space: O(1)
        pub fn getVelocity(self: *const Self) ?[]const T {
            return self.velocity;
        }
    };
}

// =============================================================================
// Tests
// =============================================================================

test "SGD: initialization" {
    var params = [_]f64{ 1.0, 2.0, 3.0 };
    const config = SGD(f64).Config{
        .learning_rate = 0.01,
        .momentum = 0.0,
    };

    var optimizer = try SGD(f64).init(testing.allocator, &params, config);
    defer optimizer.deinit();

    try testing.expectEqual(@as(usize, 3), optimizer.params.len);
    try testing.expectEqual(@as(?[]f64, null), optimizer.velocity);
}

test "SGD: vanilla update (no momentum)" {
    var params = [_]f64{ 1.0, 2.0, 3.0 };
    const gradients = [_]f64{ 0.1, 0.2, 0.3 };
    const config = SGD(f64).Config{
        .learning_rate = 0.1,
        .momentum = 0.0,
    };

    var optimizer = try SGD(f64).init(testing.allocator, &params, config);
    defer optimizer.deinit();

    try optimizer.step(&gradients);

    // Expected: θ_new = θ_old - α * g
    // params[0] = 1.0 - 0.1 * 0.1 = 0.99
    // params[1] = 2.0 - 0.1 * 0.2 = 1.98
    // params[2] = 3.0 - 0.1 * 0.3 = 2.97
    try testing.expectApproxEqAbs(@as(f64, 0.99), params[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 1.98), params[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 2.97), params[2], 1e-6);
}

test "SGD: momentum update" {
    var params = [_]f64{ 1.0, 2.0 };
    const gradients = [_]f64{ 0.5, 1.0 };
    const config = SGD(f64).Config{
        .learning_rate = 0.1,
        .momentum = 0.9,
        .nesterov = false,
    };

    var optimizer = try SGD(f64).init(testing.allocator, &params, config);
    defer optimizer.deinit();

    // Step 1: v_0 = 0, v_1 = 0.9*0 - 0.1*0.5 = -0.05
    try optimizer.step(&gradients);
    try testing.expectApproxEqAbs(@as(f64, 1.0 - 0.05), params[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 2.0 - 0.10), params[1], 1e-6);

    // Step 2: v_1 = 0.9*(-0.05) - 0.1*0.5 = -0.095
    try optimizer.step(&gradients);
    try testing.expectApproxEqAbs(@as(f64, 1.0 - 0.05 - 0.095), params[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 2.0 - 0.10 - 0.19), params[1], 1e-6);
}

test "SGD: Nesterov momentum" {
    var params = [_]f64{ 1.0 };
    const gradients = [_]f64{0.5};
    const config = SGD(f64).Config{
        .learning_rate = 0.1,
        .momentum = 0.9,
        .nesterov = true,
    };

    var optimizer = try SGD(f64).init(testing.allocator, &params, config);
    defer optimizer.deinit();

    // Nesterov: θ_t = θ_{t-1} + μ*v_t + v_t where v_t = μ*v_{t-1} - α*g_t
    // Step 1: v_0 = 0, v_1 = 0.9*0 - 0.1*0.5 = -0.05
    //         θ_1 = 1.0 + 0.9*(-0.05) + (-0.05) = 1.0 - 0.045 - 0.05 = 0.905
    try optimizer.step(&gradients);
    try testing.expectApproxEqAbs(@as(f64, 0.905), params[0], 1e-6);
}

test "SGD: weight decay" {
    var params = [_]f64{ 1.0, 2.0 };
    const gradients = [_]f64{ 0.0, 0.0 }; // Zero gradients to isolate weight decay
    const config = SGD(f64).Config{
        .learning_rate = 0.1,
        .momentum = 0.0,
        .weight_decay = 0.01,
    };

    var optimizer = try SGD(f64).init(testing.allocator, &params, config);
    defer optimizer.deinit();

    // With weight decay: θ_t = θ_{t-1} - α*(g_t + λ*θ_{t-1})
    // params[0] = 1.0 - 0.1*(0.0 + 0.01*1.0) = 1.0 - 0.001 = 0.999
    // params[1] = 2.0 - 0.1*(0.0 + 0.01*2.0) = 2.0 - 0.002 = 1.998
    try optimizer.step(&gradients);
    try testing.expectApproxEqAbs(@as(f64, 0.999), params[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 1.998), params[1], 1e-6);
}

test "SGD: simple quadratic optimization" {
    // Minimize f(x) = x² with gradient g(x) = 2x
    var params = [_]f64{5.0};
    const config = SGD(f64).Config{
        .learning_rate = 0.1,
        .momentum = 0.0,
    };

    var optimizer = try SGD(f64).init(testing.allocator, &params, config);
    defer optimizer.deinit();

    // Run 20 steps
    var i: usize = 0;
    while (i < 20) : (i += 1) {
        const grad = 2.0 * params[0]; // Gradient of x²
        try optimizer.step(&[_]f64{grad});
    }

    // Should converge toward 0
    try testing.expect(@abs(params[0]) < 1.0);
}

test "SGD: momentum accelerates convergence" {
    // Compare vanilla SGD vs SGD with momentum
    var params1 = [_]f64{5.0};
    var params2 = [_]f64{5.0};

    var vanilla = try SGD(f64).init(testing.allocator, &params1, .{ .learning_rate = 0.1, .momentum = 0.0 });
    defer vanilla.deinit();

    var momentum = try SGD(f64).init(testing.allocator, &params2, .{ .learning_rate = 0.1, .momentum = 0.9 });
    defer momentum.deinit();

    // Run 10 steps
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const grad1 = 2.0 * params1[0];
        const grad2 = 2.0 * params2[0];
        try vanilla.step(&[_]f64{grad1});
        try momentum.step(&[_]f64{grad2});
    }

    // Momentum should converge faster (closer to 0)
    try testing.expect(@abs(params2[0]) < @abs(params1[0]));
}

test "SGD: reset clears velocity" {
    var params = [_]f64{ 1.0, 2.0 };
    const gradients = [_]f64{ 0.5, 1.0 };
    const config = SGD(f64).Config{
        .learning_rate = 0.1,
        .momentum = 0.9,
    };

    var optimizer = try SGD(f64).init(testing.allocator, &params, config);
    defer optimizer.deinit();

    // Perform a step to build up velocity
    try optimizer.step(&gradients);

    // Velocity should be non-zero
    if (optimizer.velocity) |v| {
        try testing.expect(v[0] != 0.0);
        try testing.expect(v[1] != 0.0);
    }

    // Reset
    optimizer.reset();

    // Velocity should be zero
    if (optimizer.velocity) |v| {
        try testing.expectEqual(@as(f64, 0.0), v[0]);
        try testing.expectEqual(@as(f64, 0.0), v[1]);
    }
}

test "SGD: multivariate optimization" {
    // Minimize f(x, y) = x² + y² with gradients g_x = 2x, g_y = 2y
    var params = [_]f64{ 3.0, -4.0 };
    const config = SGD(f64).Config{
        .learning_rate = 0.1,
        .momentum = 0.9,
    };

    var optimizer = try SGD(f64).init(testing.allocator, &params, config);
    defer optimizer.deinit();

    // Run 30 steps
    var i: usize = 0;
    while (i < 30) : (i += 1) {
        const gradients = [_]f64{ 2.0 * params[0], 2.0 * params[1] };
        try optimizer.step(&gradients);
    }

    // Should converge toward (0, 0)
    try testing.expect(@abs(params[0]) < 0.5);
    try testing.expect(@abs(params[1]) < 0.5);
}

test "SGD: empty parameters error" {
    var params = [_]f64{};
    const config = SGD(f64).Config{};

    const result = SGD(f64).init(testing.allocator, &params, config);
    try testing.expectError(error.EmptyParameters, result);
}

test "SGD: gradient length mismatch" {
    var params = [_]f64{ 1.0, 2.0, 3.0 };
    const gradients = [_]f64{ 0.1, 0.2 }; // Wrong length
    const config = SGD(f64).Config{};

    var optimizer = try SGD(f64).init(testing.allocator, &params, config);
    defer optimizer.deinit();

    const result = optimizer.step(&gradients);
    try testing.expectError(error.GradientLengthMismatch, result);
}

test "SGD: invalid learning rate" {
    var params = [_]f64{1.0};
    const config = SGD(f64).Config{ .learning_rate = -0.01 };

    const result = SGD(f64).init(testing.allocator, &params, config);
    try testing.expectError(error.InvalidLearningRate, result);
}

test "SGD: invalid momentum" {
    var params = [_]f64{1.0};
    const config = SGD(f64).Config{ .momentum = 1.5 };

    const result = SGD(f64).init(testing.allocator, &params, config);
    try testing.expectError(error.InvalidMomentum, result);
}

test "SGD: f32 support" {
    var params = [_]f32{ 1.0, 2.0 };
    const gradients = [_]f32{ 0.1, 0.2 };
    const config = SGD(f32).Config{
        .learning_rate = 0.1,
        .momentum = 0.9,
    };

    var optimizer = try SGD(f32).init(testing.allocator, &params, config);
    defer optimizer.deinit();

    try optimizer.step(&gradients);

    try testing.expect(params[0] < 1.0);
    try testing.expect(params[1] < 2.0);
}

test "SGD: large scale optimization" {
    const allocator = testing.allocator;

    // 100-dimensional problem
    const params_buf = try allocator.alloc(f64, 100);
    defer allocator.free(params_buf);
    const gradients_buf = try allocator.alloc(f64, 100);
    defer allocator.free(gradients_buf);

    // Initialize to random values
    for (params_buf, 0..) |*p, i| {
        p.* = @as(f64, @floatFromInt(i)) - 50.0;
    }

    const config = SGD(f64).Config{
        .learning_rate = 0.05,
        .momentum = 0.9,
    };

    var optimizer = try SGD(f64).init(allocator, params_buf, config);
    defer optimizer.deinit();

    // Minimize sum of squares
    var step: usize = 0;
    while (step < 50) : (step += 1) {
        for (params_buf, gradients_buf) |p, *g| {
            g.* = 2.0 * p; // Gradient of x²
        }
        try optimizer.step(gradients_buf);
    }

    // Check convergence
    var sum: f64 = 0.0;
    for (params_buf) |p| {
        sum += p * p;
    }
    try testing.expect(sum < 500.0); // Should be much closer to 0
}

test "SGD: memory safety" {
    var params = [_]f64{ 1.0, 2.0, 3.0 };
    const config = SGD(f64).Config{
        .learning_rate = 0.01,
        .momentum = 0.9,
    };

    var optimizer = try SGD(f64).init(testing.allocator, &params, config);
    defer optimizer.deinit();

    // Verify velocity is allocated
    try testing.expect(optimizer.velocity != null);

    // Multiple deallocation should be safe
    optimizer.deinit();
}
