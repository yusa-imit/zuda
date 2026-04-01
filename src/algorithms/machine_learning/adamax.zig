const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// AdaMax optimizer
///
/// Variant of Adam based on the infinity norm (max norm) instead of L2 norm.
/// Uses exponentially weighted infinity norm for the second moment, which can be
/// more stable than Adam's L2-based second moment in certain scenarios.
///
/// Algorithm:
/// 1. Compute gradient g_t = ∇f(θ_t)
/// 2. Update biased first moment: m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
/// 3. Update exponentially weighted infinity norm: u_t = max(β₂ × u_{t-1}, |g_t|)
/// 4. Compute bias-corrected first moment: m̂_t = m_t / (1 - β₁^t)
/// 5. Update parameters: θ_t = θ_{t-1} - (α / (1 - β₁^t)) × (m_t / u_t)
///
/// Key differences from Adam:
/// - Uses infinity norm (max) instead of L2 norm for second moment
/// - No bias correction needed for u_t (already unbiased)
/// - Simpler and sometimes more stable than Adam
/// - Better suited for problems with sparse or unbounded gradients
///
/// Time complexity: O(n) per update where n = number of parameters
/// Space complexity: O(n) for momentum and infinity norm vectors
///
/// Use cases:
/// - Problems with sparse gradients (NLP, embeddings)
/// - When Adam's L2 norm is too sensitive to outliers
/// - Large-scale deep learning (alternative to Adam)
/// - Situations requiring more stable second moment estimates
///
/// References:
/// - Kingma & Ba (2015): "Adam: A Method for Stochastic Optimization" (Section 7)
/// - Often more robust than Adam for certain problem types
pub fn AdaMax(comptime T: type) type {
    return struct {
        allocator: Allocator,
        params: []T,
        m: []T, // First moment (exponentially weighted average of gradients)
        u: []T, // Exponentially weighted infinity norm
        t: usize, // Timestep counter
        config: Config,

        const Self = @This();

        /// Configuration for AdaMax optimizer
        pub const Config = struct {
            /// Learning rate (step size)
            /// Common values: 0.002, 0.001
            /// Typically set slightly higher than Adam
            /// Time: O(1) | Space: O(1)
            learning_rate: T = 0.002,

            /// Exponential decay rate for first moment (momentum)
            /// Common value: 0.9
            /// Time: O(1) | Space: O(1)
            beta1: T = 0.9,

            /// Exponential decay rate for infinity norm
            /// Common value: 0.999
            /// Time: O(1) | Space: O(1)
            beta2: T = 0.999,

            /// Small constant for numerical stability
            /// Prevents division by zero in update rule
            /// Common value: 1e-8
            /// Time: O(1) | Space: O(1)
            epsilon: T = 1e-8,
        };

        /// Initialize AdaMax optimizer
        /// Time: O(n) | Space: O(n)
        pub fn init(allocator: Allocator, params: []T, config: Config) !Self {
            if (params.len == 0) return error.EmptyParameters;
            if (config.learning_rate <= 0) return error.InvalidLearningRate;
            if (config.beta1 < 0 or config.beta1 >= 1) return error.InvalidBeta1;
            if (config.beta2 < 0 or config.beta2 >= 1) return error.InvalidBeta2;
            if (config.epsilon <= 0) return error.InvalidEpsilon;

            const m = try allocator.alloc(T, params.len);
            errdefer allocator.free(m);
            const u = try allocator.alloc(T, params.len);
            errdefer allocator.free(u);

            // Initialize moments to zero
            @memset(m, 0);
            @memset(u, 0);

            return Self{
                .allocator = allocator,
                .params = params,
                .m = m,
                .u = u,
                .t = 0,
                .config = config,
            };
        }

        /// Free allocated memory
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.m);
            self.allocator.free(self.u);
        }

        /// Perform one optimization step
        ///
        /// Updates parameters based on computed gradients using AdaMax algorithm.
        ///
        /// Arguments:
        ///   gradients: Gradient vector (same length as params)
        ///
        /// Time: O(n) where n = number of parameters
        /// Space: O(1) (in-place updates)
        pub fn step(self: *Self, gradients: []const T) !void {
            if (gradients.len != self.params.len) return error.GradientLengthMismatch;

            self.t += 1;
            const t_float = @as(T, @floatFromInt(self.t));

            // Compute bias correction for first moment only
            const bias_correction1 = 1.0 - std.math.pow(T, self.config.beta1, t_float);

            for (0..self.params.len) |i| {
                const g = gradients[i];

                // Update biased first moment estimate: m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
                self.m[i] = self.config.beta1 * self.m[i] + (1.0 - self.config.beta1) * g;

                // Update exponentially weighted infinity norm: u_t = max(β₂ × u_{t-1}, |g_t|)
                self.u[i] = @max(self.config.beta2 * self.u[i], @abs(g));

                // Update parameters: θ_t = θ_{t-1} - (α / (1 - β₁^t)) × (m_t / (u_t + ε))
                const step_size = self.config.learning_rate / bias_correction1;
                self.params[i] -= step_size * self.m[i] / (self.u[i] + self.config.epsilon);
            }
        }

        /// Reset optimizer state (timestep and moments)
        /// Useful when starting a new optimization phase
        /// Time: O(n) | Space: O(1)
        pub fn reset(self: *Self) void {
            self.t = 0;
            @memset(self.m, 0);
            @memset(self.u, 0);
        }

        /// Get current timestep
        /// Time: O(1) | Space: O(1)
        pub fn getTimestep(self: Self) usize {
            return self.t;
        }

        /// Get current effective learning rate (after bias correction)
        /// Time: O(1) | Space: O(1)
        pub fn getEffectiveLearningRate(self: Self) T {
            if (self.t == 0) return self.config.learning_rate;
            const t_float = @as(T, @floatFromInt(self.t));
            const bias_correction1 = 1.0 - std.math.pow(T, self.config.beta1, t_float);
            return self.config.learning_rate / bias_correction1;
        }
    };
}

// Tests

test "AdaMax: initialization" {
    const allocator = testing.allocator;

    var params = [_]f64{ 1.0, 2.0, 3.0 };
    const config = AdaMax(f64).Config{};

    var optimizer = try AdaMax(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    try testing.expectEqual(@as(usize, 3), optimizer.params.len);
    try testing.expectEqual(@as(usize, 0), optimizer.t);
    try testing.expectApproxEqAbs(@as(f64, 0.002), optimizer.config.learning_rate, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.9), optimizer.config.beta1, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.999), optimizer.config.beta2, 1e-10);

    // Check moments initialized to zero
    for (optimizer.m) |val| {
        try testing.expectEqual(@as(f64, 0.0), val);
    }
    for (optimizer.u) |val| {
        try testing.expectEqual(@as(f64, 0.0), val);
    }
}

test "AdaMax: custom config" {
    const allocator = testing.allocator;

    var params = [_]f64{1.0};
    const config = AdaMax(f64).Config{
        .learning_rate = 0.01,
        .beta1 = 0.8,
        .beta2 = 0.99,
        .epsilon = 1e-7,
    };

    var optimizer = try AdaMax(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    try testing.expectApproxEqAbs(@as(f64, 0.01), optimizer.config.learning_rate, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.8), optimizer.config.beta1, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.99), optimizer.config.beta2, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1e-7), optimizer.config.epsilon, 1e-10);
}

test "AdaMax: simple quadratic optimization" {
    const allocator = testing.allocator;

    // Minimize f(x) = x² with x₀ = 10
    var params = [_]f64{10.0};
    const config = AdaMax(f64).Config{ .learning_rate = 0.1 };

    var optimizer = try AdaMax(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    // Gradient of f(x) = x² is g(x) = 2x
    for (0..200) |_| {
        const gradient = [_]f64{2.0 * params[0]};
        try optimizer.step(&gradient);
    }

    // Should converge close to x = 0
    try testing.expect(@abs(params[0]) < 0.1);
}

test "AdaMax: multivariate quadratic" {
    const allocator = testing.allocator;

    // Minimize f(x,y) = x² + y² with (x₀, y₀) = (5, -5)
    var params = [_]f64{ 5.0, -5.0 };
    const config = AdaMax(f64).Config{ .learning_rate = 0.1 };

    var optimizer = try AdaMax(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    // Gradients: ∇f = (2x, 2y)
    for (0..200) |_| {
        const gradients = [_]f64{ 2.0 * params[0], 2.0 * params[1] };
        try optimizer.step(&gradients);
    }

    // Should converge close to (0, 0)
    try testing.expect(@abs(params[0]) < 0.1);
    try testing.expect(@abs(params[1]) < 0.1);
}

test "AdaMax: Rosenbrock function" {
    const allocator = testing.allocator;

    // Minimize Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
    // Optimal: (x*, y*) = (1, 1)
    var params = [_]f64{ 0.0, 0.0 };
    const config = AdaMax(f64).Config{ .learning_rate = 0.01 };

    var optimizer = try AdaMax(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    // ∇f = (-2(1-x) - 400x(y-x²), 200(y-x²))
    for (0..2000) |_| {
        const x = params[0];
        const y = params[1];
        const gradients = [_]f64{
            -2.0 * (1.0 - x) - 400.0 * x * (y - x * x),
            200.0 * (y - x * x),
        };
        try optimizer.step(&gradients);
    }

    // Should converge reasonably close to (1, 1)
    try testing.expect(@abs(params[0] - 1.0) < 0.2);
    try testing.expect(@abs(params[1] - 1.0) < 0.2);
}

test "AdaMax: infinity norm update" {
    const allocator = testing.allocator;

    var params = [_]f64{1.0};
    const config = AdaMax(f64).Config{};

    var optimizer = try AdaMax(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    // First step with small gradient
    try optimizer.step(&[_]f64{0.1});
    const u_first = optimizer.u[0];
    try testing.expect(u_first > 0);

    // Second step with larger gradient - should take max
    try optimizer.step(&[_]f64{1.0});
    const u_second = optimizer.u[0];
    try testing.expect(u_second > u_first); // Should be max(beta2 * u_first, 1.0) = 1.0

    // Third step with smaller gradient - should decay
    try optimizer.step(&[_]f64{0.5});
    const u_third = optimizer.u[0];
    try testing.expect(u_third >= @max(0.999 * u_second, 0.5));
}

test "AdaMax: momentum accumulation" {
    const allocator = testing.allocator;

    var params = [_]f64{0.0};
    const config = AdaMax(f64).Config{};

    var optimizer = try AdaMax(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    // Apply consistent gradient
    for (0..10) |_| {
        try optimizer.step(&[_]f64{1.0});
    }

    // First moment should have accumulated
    try testing.expect(optimizer.m[0] > 0);
    try testing.expect(optimizer.m[0] < 1.0); // But less than raw gradient due to beta1
}

test "AdaMax: bias correction effect" {
    const allocator = testing.allocator;

    var params = [_]f64{10.0};
    const config = AdaMax(f64).Config{ .learning_rate = 0.1 };

    var optimizer = try AdaMax(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    const initial_lr = optimizer.getEffectiveLearningRate();
    try optimizer.step(&[_]f64{1.0});
    const lr_after_1 = optimizer.getEffectiveLearningRate();

    // Effective learning rate should increase as bias correction decreases
    try testing.expect(lr_after_1 > initial_lr);

    // After many steps, should approach base learning rate
    for (0..100) |_| {
        try optimizer.step(&[_]f64{1.0});
    }
    const lr_after_100 = optimizer.getEffectiveLearningRate();
    try testing.expectApproxEqAbs(@as(f64, 0.1), lr_after_100, 0.01);
}

test "AdaMax: sparse gradients" {
    const allocator = testing.allocator;

    var params = [_]f64{ 1.0, 2.0, 3.0 };
    const config = AdaMax(f64).Config{};

    var optimizer = try AdaMax(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    // Only first parameter gets gradient
    for (0..50) |_| {
        try optimizer.step(&[_]f64{ 1.0, 0.0, 0.0 });
    }

    // First parameter should have moved
    try testing.expect(@abs(params[0] - 1.0) > 0.1);

    // Other parameters should be unchanged (no gradient)
    try testing.expectEqual(@as(f64, 2.0), params[1]);
    try testing.expectEqual(@as(f64, 3.0), params[2]);
}

test "AdaMax: reset functionality" {
    const allocator = testing.allocator;

    var params = [_]f64{1.0};
    const config = AdaMax(f64).Config{};

    var optimizer = try AdaMax(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    // Take some steps
    for (0..10) |_| {
        try optimizer.step(&[_]f64{1.0});
    }

    try testing.expect(optimizer.t > 0);
    try testing.expect(optimizer.m[0] != 0);
    try testing.expect(optimizer.u[0] != 0);

    // Reset
    optimizer.reset();

    try testing.expectEqual(@as(usize, 0), optimizer.t);
    try testing.expectEqual(@as(f64, 0.0), optimizer.m[0]);
    try testing.expectEqual(@as(f64, 0.0), optimizer.u[0]);
}

test "AdaMax: f32 support" {
    const allocator = testing.allocator;

    var params = [_]f32{ 5.0, -3.0 };
    const config = AdaMax(f32).Config{ .learning_rate = 0.1 };

    var optimizer = try AdaMax(f32).init(allocator, &params, config);
    defer optimizer.deinit();

    for (0..100) |_| {
        const gradients = [_]f32{ 2.0 * params[0], 2.0 * params[1] };
        try optimizer.step(&gradients);
    }

    try testing.expect(@abs(params[0]) < 0.2);
    try testing.expect(@abs(params[1]) < 0.2);
}

test "AdaMax: f64 support" {
    const allocator = testing.allocator;

    var params = [_]f64{ 5.0, -3.0 };
    const config = AdaMax(f64).Config{ .learning_rate = 0.1 };

    var optimizer = try AdaMax(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    for (0..100) |_| {
        const gradients = [_]f64{ 2.0 * params[0], 2.0 * params[1] };
        try optimizer.step(&gradients);
    }

    try testing.expect(@abs(params[0]) < 0.2);
    try testing.expect(@abs(params[1]) < 0.2);
}

test "AdaMax: large scale" {
    const allocator = testing.allocator;

    var params: [100]f64 = undefined;
    for (0..100) |i| {
        params[i] = @as(f64, @floatFromInt(i)) - 50.0; // Range: -50 to 49
    }

    const config = AdaMax(f64).Config{ .learning_rate = 0.05 };
    var optimizer = try AdaMax(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    // Minimize f(x) = Σ xᵢ² (each parameter independent)
    for (0..500) |_| {
        var gradients: [100]f64 = undefined;
        for (0..100) |i| {
            gradients[i] = 2.0 * params[i];
        }
        try optimizer.step(&gradients);
    }

    // All parameters should converge close to 0
    for (params) |p| {
        try testing.expect(@abs(p) < 1.0);
    }
}

test "AdaMax: empty parameters error" {
    const allocator = testing.allocator;
    var params = [_]f64{};
    const config = AdaMax(f64).Config{};

    const result = AdaMax(f64).init(allocator, &params, config);
    try testing.expectError(error.EmptyParameters, result);
}

test "AdaMax: gradient length mismatch" {
    const allocator = testing.allocator;

    var params = [_]f64{ 1.0, 2.0 };
    const config = AdaMax(f64).Config{};

    var optimizer = try AdaMax(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    const result = optimizer.step(&[_]f64{1.0}); // Wrong length
    try testing.expectError(error.GradientLengthMismatch, result);
}

test "AdaMax: invalid learning rate" {
    const allocator = testing.allocator;
    var params = [_]f64{1.0};
    const config = AdaMax(f64).Config{ .learning_rate = -0.1 };

    const result = AdaMax(f64).init(allocator, &params, config);
    try testing.expectError(error.InvalidLearningRate, result);
}

test "AdaMax: invalid beta1" {
    const allocator = testing.allocator;
    var params = [_]f64{1.0};
    const config = AdaMax(f64).Config{ .beta1 = 1.5 };

    const result = AdaMax(f64).init(allocator, &params, config);
    try testing.expectError(error.InvalidBeta1, result);
}

test "AdaMax: invalid beta2" {
    const allocator = testing.allocator;
    var params = [_]f64{1.0};
    const config = AdaMax(f64).Config{ .beta2 = -0.5 };

    const result = AdaMax(f64).init(allocator, &params, config);
    try testing.expectError(error.InvalidBeta2, result);
}

test "AdaMax: invalid epsilon" {
    const allocator = testing.allocator;
    var params = [_]f64{1.0};
    const config = AdaMax(f64).Config{ .epsilon = 0.0 };

    const result = AdaMax(f64).init(allocator, &params, config);
    try testing.expectError(error.InvalidEpsilon, result);
}

test "AdaMax: memory safety" {
    const allocator = testing.allocator;

    var params = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const config = AdaMax(f64).Config{};

    var optimizer = try AdaMax(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    // Multiple steps to verify no memory issues
    for (0..100) |_| {
        const gradients = [_]f64{ 0.1, 0.2, 0.3, 0.4, 0.5 };
        try optimizer.step(&gradients);
    }

    // Verify optimizer state is reasonable
    try testing.expect(optimizer.t == 100);
}
