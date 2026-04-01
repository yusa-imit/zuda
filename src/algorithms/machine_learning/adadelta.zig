const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Adadelta optimizer
///
/// An extension of Adagrad that addresses the monotonically decreasing learning rate
/// problem by restricting the window of accumulated past gradients to a fixed size.
/// Instead of accumulating all past squared gradients, Adadelta uses a decaying average
/// of past squared gradients, making the learning rate adapt dynamically without
/// vanishing to zero.
///
/// Algorithm:
/// 1. Compute gradient g_t = ∇f(θ_t)
/// 2. Accumulate squared gradients with decay: E[g²]_t = ρ × E[g²]_{t-1} + (1-ρ) × g_t²
/// 3. Compute update: Δθ_t = -√(E[Δθ²]_{t-1} + ε) / √(E[g²]_t + ε) × g_t
/// 4. Accumulate squared updates: E[Δθ²]_t = ρ × E[Δθ²]_{t-1} + (1-ρ) × Δθ_t²
/// 5. Update parameters: θ_t = θ_{t-1} + Δθ_t
///
/// Key features:
/// - No learning rate hyperparameter needed (self-adaptive)
/// - Uses moving average of past squared gradients (not accumulation)
/// - Learning rate doesn't monotonically decrease
/// - Units correct: RMS[Δθ] / RMS[g] has same units as parameter
/// - Robust to initial conditions and hyperparameter choices
///
/// Time complexity: O(n) per update where n = number of parameters
/// Space complexity: O(2n) for gradient and update accumulators
///
/// Use cases:
/// - When you don't want to tune learning rates manually
/// - Sparse data (like Adagrad, but without aggressive decay)
/// - Non-stationary objectives where learning rate needs to adapt
/// - When Adagrad's learning rate decays too quickly
///
/// Trade-offs:
/// - vs Adagrad: no monotonic decay, doesn't require learning rate, but uses more memory
/// - vs RMSprop: similar moving average approach, but Adadelta doesn't need learning rate
/// - vs Adam: simpler (no bias correction), but Adam often converges faster
/// - vs SGD: more robust to hyperparameter choices, but more expensive per step
///
/// Advantages over Adagrad:
/// - Continues learning even after many iterations (no learning rate collapse)
/// - No need to manually set initial learning rate
/// - Better for deep learning (non-convex optimization)
///
/// References:
/// - Zeiler (2012): "ADADELTA: An Adaptive Learning Rate Method" (arXiv:1212.5701)
/// - Fixes the units problem in Adagrad/RMSprop
pub fn Adadelta(comptime T: type) type {
    if (T != f32 and T != f64) {
        @compileError("Adadelta only supports f32 and f64 types");
    }

    return struct {
        allocator: Allocator,
        rho: T, // decay rate for moving average
        epsilon: T, // numerical stability constant
        weight_decay: T, // L2 regularization

        // State
        E_g2: []T, // accumulated squared gradients (moving average)
        E_delta2: []T, // accumulated squared updates (moving average)

        const Self = @This();

        /// Configuration for Adadelta optimizer
        pub const Config = struct {
            rho: T = 0.95, // decay rate (typical: 0.9-0.99)
            epsilon: T = 1e-6, // numerical stability (larger than Adam's 1e-8)
            weight_decay: T = 0.0, // L2 penalty
        };

        /// Initialize Adadelta optimizer
        ///
        /// Time: O(n) | Space: O(2n)
        pub fn init(allocator: Allocator, num_params: usize, config: Config) !Self {
            if (num_params == 0) {
                return error.EmptyParameters;
            }
            if (config.rho <= 0 or config.rho >= 1) {
                return error.InvalidRho;
            }
            if (config.epsilon <= 0) {
                return error.InvalidEpsilon;
            }
            if (config.weight_decay < 0) {
                return error.InvalidWeightDecay;
            }

            const E_g2 = try allocator.alloc(T, num_params);
            errdefer allocator.free(E_g2);
            @memset(E_g2, 0);

            const E_delta2 = try allocator.alloc(T, num_params);
            errdefer allocator.free(E_delta2);
            @memset(E_delta2, 0);

            return Self{
                .allocator = allocator,
                .rho = config.rho,
                .epsilon = config.epsilon,
                .weight_decay = config.weight_decay,
                .E_g2 = E_g2,
                .E_delta2 = E_delta2,
            };
        }

        /// Free resources
        ///
        /// Time: O(1)
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.E_g2);
            self.allocator.free(self.E_delta2);
        }

        /// Perform one optimization step
        ///
        /// Updates parameters using accumulated gradients and updates.
        /// Automatically adapts learning rate based on gradient history.
        ///
        /// Time: O(n) | Space: O(1)
        pub fn step(self: *Self, params: []T, gradients: []const T) !void {
            if (params.len != gradients.len) {
                return error.GradientLengthMismatch;
            }
            if (params.len != self.E_g2.len) {
                return error.ParameterLengthMismatch;
            }

            for (params, gradients, 0..) |*param, grad, i| {
                var g = grad;

                // Weight decay (L2 regularization)
                if (self.weight_decay > 0) {
                    g += self.weight_decay * param.*;
                }

                // Accumulate squared gradient with decay
                // E[g²]_t = ρ × E[g²]_{t-1} + (1-ρ) × g_t²
                self.E_g2[i] = self.rho * self.E_g2[i] + (1 - self.rho) * g * g;

                // Compute RMS of accumulated gradients
                const rms_g = @sqrt(self.E_g2[i] + self.epsilon);

                // Compute RMS of accumulated updates
                const rms_delta = @sqrt(self.E_delta2[i] + self.epsilon);

                // Compute parameter update
                // Δθ_t = -RMS[Δθ]_{t-1} / RMS[g]_t × g_t
                const delta = -(rms_delta / rms_g) * g;

                // Accumulate squared update with decay
                // E[Δθ²]_t = ρ × E[Δθ²]_{t-1} + (1-ρ) × Δθ_t²
                self.E_delta2[i] = self.rho * self.E_delta2[i] + (1 - self.rho) * delta * delta;

                // Update parameter
                param.* += delta;
            }
        }

        /// Reset optimizer state (clear accumulated gradients and updates)
        ///
        /// Time: O(n)
        pub fn reset(self: *Self) void {
            @memset(self.E_g2, 0);
            @memset(self.E_delta2, 0);
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "Adadelta: initialization" {
    const allocator = testing.allocator;
    const T = f64;

    var opt = try Adadelta(T).init(allocator, 3, .{});
    defer opt.deinit();

    try testing.expectEqual(@as(T, 0.95), opt.rho);
    try testing.expectEqual(@as(T, 1e-6), opt.epsilon);
    try testing.expectEqual(@as(usize, 3), opt.E_g2.len);
    try testing.expectEqual(@as(usize, 3), opt.E_delta2.len);
}

test "Adadelta: custom config" {
    const allocator = testing.allocator;
    const T = f64;

    const config = Adadelta(T).Config{
        .rho = 0.9,
        .epsilon = 1e-7,
        .weight_decay = 0.01,
    };

    var opt = try Adadelta(T).init(allocator, 5, config);
    defer opt.deinit();

    try testing.expectEqual(@as(T, 0.9), opt.rho);
    try testing.expectEqual(@as(T, 1e-7), opt.epsilon);
    try testing.expectEqual(@as(T, 0.01), opt.weight_decay);
}

test "Adadelta: simple quadratic optimization" {
    const allocator = testing.allocator;
    const T = f64;

    // Minimize f(x) = x² with gradient g = 2x
    var params = [_]T{10.0};
    var opt = try Adadelta(T).init(allocator, 1, .{ .rho = 0.95, .epsilon = 1e-6 });
    defer opt.deinit();

    // Run optimization (Adadelta has slow initial convergence but doesn't collapse)
    var i: usize = 0;
    while (i < 2000) : (i += 1) {
        const grad = [_]T{2 * params[0]};
        try opt.step(&params, &grad);
    }

    // Should show significant reduction (Adadelta converges slowly but steadily)
    try testing.expect(@abs(params[0]) < 5.0); // From 10.0, shows progress
}

test "Adadelta: multivariate quadratic" {
    const allocator = testing.allocator;
    const T = f64;

    // Minimize f(x,y) = x² + y² with gradients [2x, 2y]
    var params = [_]T{ 5.0, -3.0 };
    var opt = try Adadelta(T).init(allocator, 2, .{});
    defer opt.deinit();

    // Run optimization (more iterations for Adadelta's slower initial convergence)
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        const grads = [_]T{ 2 * params[0], 2 * params[1] };
        try opt.step(&params, &grads);
    }

    // Should converge reasonably close to origin
    try testing.expect(@abs(params[0]) < 1.0);
    try testing.expect(@abs(params[1]) < 1.0);
}

test "Adadelta: adaptive learning without manual rate" {
    const allocator = testing.allocator;
    const T = f64;

    // Test that Adadelta works without specifying learning rate
    // (unlike SGD, Adam, RMSprop which require it)
    var params = [_]T{8.0};
    var opt = try Adadelta(T).init(allocator, 1, .{});
    defer opt.deinit();

    const initial = params[0];

    // Single step
    const grad = [_]T{2 * params[0]};
    try opt.step(&params, &grad);

    // Parameter should change (no explicit learning rate needed)
    try testing.expect(params[0] != initial);
    try testing.expect(@abs(params[0]) < @abs(initial));
}

test "Adadelta: weight decay (L2 regularization)" {
    const allocator = testing.allocator;
    const T = f64;

    var params_no_decay = [_]T{5.0};
    var params_with_decay = [_]T{5.0};

    var opt_no_decay = try Adadelta(T).init(allocator, 1, .{ .weight_decay = 0.0 });
    defer opt_no_decay.deinit();

    var opt_with_decay = try Adadelta(T).init(allocator, 1, .{ .weight_decay = 0.1 });
    defer opt_with_decay.deinit();

    // Same gradient
    const grad = [_]T{1.0};

    try opt_no_decay.step(&params_no_decay, &grad);
    try opt_with_decay.step(&params_with_decay, &grad);

    // With weight decay, parameter should be pulled toward zero more
    // (effective gradient is larger due to L2 penalty)
    try testing.expect(@abs(params_with_decay[0]) < @abs(params_no_decay[0]));
}

test "Adadelta: continues learning (no decay collapse)" {
    const allocator = testing.allocator;
    const T = f64;

    // Test that learning rate doesn't collapse to zero like Adagrad
    var params = [_]T{10.0};
    var opt = try Adadelta(T).init(allocator, 1, .{});
    defer opt.deinit();

    // Run many iterations with constant gradient
    var i: usize = 0;
    var last_param = params[0];
    const grad = [_]T{0.5}; // Small constant gradient

    while (i < 500) : (i += 1) {
        try opt.step(&params, &grad);
    }

    // In last 100 iterations, check that we still make progress
    i = 0;
    while (i < 100) : (i += 1) {
        last_param = params[0];
        try opt.step(&params, &grad);
        // Should still be updating (not stuck)
        try testing.expect(params[0] != last_param);
    }
}

test "Adadelta: sparse gradients" {
    const allocator = testing.allocator;
    const T = f64;

    var params = [_]T{ 1.0, 2.0, 3.0, 4.0 };
    var opt = try Adadelta(T).init(allocator, 4, .{});
    defer opt.deinit();

    // Sparse gradient (only some parameters updated)
    const grad1 = [_]T{ 1.0, 0.0, 0.0, 0.0 };
    const grad2 = [_]T{ 0.0, 0.0, 1.0, 0.0 };

    try opt.step(&params, &grad1);
    try opt.step(&params, &grad2);

    // Parameters with zero gradient should remain unchanged or change minimally
    // (only through weight decay if enabled, which is 0.0 here)
    try testing.expectEqual(@as(T, 2.0), params[1]);
    try testing.expectEqual(@as(T, 4.0), params[3]);
}

test "Adadelta: reset state" {
    const allocator = testing.allocator;
    const T = f64;

    var opt = try Adadelta(T).init(allocator, 3, .{});
    defer opt.deinit();

    // Accumulate some history
    var params = [_]T{ 1.0, 2.0, 3.0 };
    const grad = [_]T{ 0.5, 0.5, 0.5 };
    try opt.step(&params, &grad);

    // State should be non-zero
    try testing.expect(opt.E_g2[0] > 0);
    try testing.expect(opt.E_delta2[0] > 0);

    // Reset
    opt.reset();

    // State should be zero again
    try testing.expectEqual(@as(T, 0), opt.E_g2[0]);
    try testing.expectEqual(@as(T, 0), opt.E_delta2[0]);
}

test "Adadelta: f32 support" {
    const allocator = testing.allocator;
    const T = f32;

    var params = [_]T{5.0};
    var opt = try Adadelta(T).init(allocator, 1, .{});
    defer opt.deinit();

    const grad = [_]T{1.0};
    try opt.step(&params, &grad);

    try testing.expect(@abs(params[0] - 5.0) > 0.0001);
}

test "Adadelta: large scale (100 parameters)" {
    const allocator = testing.allocator;
    const T = f64;

    const params = try allocator.alloc(T, 100);
    defer allocator.free(params);
    const grads = try allocator.alloc(T, 100);
    defer allocator.free(grads);

    // Initialize with random-like values
    for (params, 0..) |*p, i| {
        p.* = @as(T, @floatFromInt(i)) - 50.0;
    }

    var opt = try Adadelta(T).init(allocator, 100, .{});
    defer opt.deinit();

    // Run optimization (gradient = 2 * param for quadratic)
    var i: usize = 0;
    while (i < 2000) : (i += 1) {
        for (params, grads) |p, *g| {
            g.* = 2 * p;
        }
        try opt.step(params, grads);
    }

    // All parameters should move toward zero (Adadelta converges steadily)
    for (params) |p| {
        try testing.expect(@abs(p) < 36.0); // Significant reduction from initial [-50, 49]
    }
}

test "Adadelta: error - empty parameters" {
    const allocator = testing.allocator;
    const T = f64;

    const result = Adadelta(T).init(allocator, 0, .{});
    try testing.expectError(error.EmptyParameters, result);
}

test "Adadelta: error - gradient length mismatch" {
    const allocator = testing.allocator;
    const T = f64;

    var params = [_]T{ 1.0, 2.0 };
    var opt = try Adadelta(T).init(allocator, 2, .{});
    defer opt.deinit();

    const grad = [_]T{0.5}; // Wrong size
    try testing.expectError(error.GradientLengthMismatch, opt.step(&params, &grad));
}

test "Adadelta: error - invalid rho" {
    const allocator = testing.allocator;
    const T = f64;

    const config1 = Adadelta(T).Config{ .rho = 0.0 };
    try testing.expectError(error.InvalidRho, Adadelta(T).init(allocator, 1, config1));

    const config2 = Adadelta(T).Config{ .rho = 1.0 };
    try testing.expectError(error.InvalidRho, Adadelta(T).init(allocator, 1, config2));

    const config3 = Adadelta(T).Config{ .rho = -0.5 };
    try testing.expectError(error.InvalidRho, Adadelta(T).init(allocator, 1, config3));
}

test "Adadelta: error - invalid epsilon" {
    const allocator = testing.allocator;
    const T = f64;

    const config = Adadelta(T).Config{ .epsilon = 0.0 };
    try testing.expectError(error.InvalidEpsilon, Adadelta(T).init(allocator, 1, config));
}

test "Adadelta: error - invalid weight decay" {
    const allocator = testing.allocator;
    const T = f64;

    const config = Adadelta(T).Config{ .weight_decay = -0.1 };
    try testing.expectError(error.InvalidWeightDecay, Adadelta(T).init(allocator, 1, config));
}

test "Adadelta: memory safety (testing.allocator)" {
    const allocator = testing.allocator;
    const T = f64;

    var params = [_]T{ 1.0, 2.0, 3.0 };
    var opt = try Adadelta(T).init(allocator, 3, .{});
    defer opt.deinit();

    const grad = [_]T{ 0.5, 0.5, 0.5 };
    try opt.step(&params, &grad);

    // testing.allocator will catch leaks
}
