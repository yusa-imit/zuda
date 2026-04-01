const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Adam (Adaptive Moment Estimation) optimizer
///
/// Adaptive learning rate optimization algorithm that computes individual learning rates
/// for different parameters from estimates of first and second moments of the gradients.
/// Combines the advantages of AdaGrad (adaptive learning rates) and RMSProp (exponential decay).
///
/// Algorithm:
/// 1. Compute gradient g_t = ∇f(θ_t)
/// 2. Update biased first moment: m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
/// 3. Update biased second moment: v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²
/// 4. Compute bias-corrected moments: m̂_t = m_t / (1 - β₁^t), v̂_t = v_t / (1 - β₂^t)
/// 5. Update parameters: θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε)
///
/// Key features:
/// - Adaptive learning rates per parameter
/// - Bias correction for moment estimates
/// - Effective in practice with default hyperparameters
/// - Handles sparse gradients and non-stationary objectives
///
/// Time complexity: O(n) per update where n = number of parameters
/// Space complexity: O(n) for momentum and velocity vectors
///
/// Use cases:
/// - Deep learning (neural networks, transformers)
/// - Non-convex optimization
/// - High-dimensional parameter spaces
/// - Sparse gradient problems (NLP, recommender systems)
///
/// References:
/// - Kingma & Ba (2015): "Adam: A Method for Stochastic Optimization"
/// - Commonly used in PyTorch, TensorFlow, Keras
pub fn Adam(comptime T: type) type {
    return struct {
        allocator: Allocator,
        params: []T,
        m: []T, // First moment (mean of gradients)
        v: []T, // Second moment (uncentered variance of gradients)
        t: usize, // Timestep counter
        config: Config,

        const Self = @This();

        /// Configuration for Adam optimizer
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

        /// Initialize Adam optimizer
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

        /// Perform one optimization step
        ///
        /// Updates parameters based on computed gradients.
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

            // Compute bias correction terms
            const bias_correction1 = 1.0 - std.math.pow(T, self.config.beta1, t_float);
            const bias_correction2 = 1.0 - std.math.pow(T, self.config.beta2, t_float);

            for (0..self.params.len) |i| {
                const g = gradients[i];

                // Update biased first moment estimate: m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
                self.m[i] = self.config.beta1 * self.m[i] + (1.0 - self.config.beta1) * g;

                // Update biased second moment estimate: v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²
                self.v[i] = self.config.beta2 * self.v[i] + (1.0 - self.config.beta2) * g * g;

                // Compute bias-corrected first moment: m̂_t = m_t / (1 - β₁^t)
                const m_hat = self.m[i] / bias_correction1;

                // Compute bias-corrected second moment: v̂_t = v_t / (1 - β₂^t)
                const v_hat = self.v[i] / bias_correction2;

                // Update parameters: θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε)
                self.params[i] -= self.config.learning_rate * m_hat / (@sqrt(v_hat) + self.config.epsilon);
            }
        }

        /// Reset optimizer state (timestep and moments)
        /// Useful when starting a new optimization phase
        /// Time: O(n) | Space: O(1)
        pub fn reset(self: *Self) void {
            self.t = 0;
            @memset(self.m, 0);
            @memset(self.v, 0);
        }

        /// Get current timestep
        /// Time: O(1) | Space: O(1)
        pub fn getTimestep(self: Self) usize {
            return self.t;
        }

        /// Get current learning rate (effective)
        /// After bias correction, effective learning rate changes over time
        /// Time: O(1) | Space: O(1)
        pub fn getEffectiveLearningRate(self: Self) T {
            if (self.t == 0) return self.config.learning_rate;
            const t_float = @as(T, @floatFromInt(self.t));
            const bias_correction1 = 1.0 - std.math.pow(T, self.config.beta1, t_float);
            const bias_correction2 = 1.0 - std.math.pow(T, self.config.beta2, t_float);
            return self.config.learning_rate * @sqrt(bias_correction2) / bias_correction1;
        }
    };
}

// Tests

test "Adam: initialization" {
    const allocator = testing.allocator;

    var params = [_]f64{ 1.0, 2.0, 3.0 };
    const config = Adam(f64).Config{};

    var optimizer = try Adam(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    try testing.expectEqual(@as(usize, 3), optimizer.params.len);
    try testing.expectEqual(@as(usize, 0), optimizer.t);
    try testing.expectApproxEqAbs(@as(f64, 0.001), optimizer.config.learning_rate, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.9), optimizer.config.beta1, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.999), optimizer.config.beta2, 1e-10);
}

test "Adam: simple quadratic optimization" {
    const allocator = testing.allocator;

    // Minimize f(x) = x² with x₀ = 10
    var params = [_]f64{10.0};
    const config = Adam(f64).Config{ .learning_rate = 0.1 };

    var optimizer = try Adam(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    // Gradient of f(x) = x² is g(x) = 2x
    for (0..100) |_| {
        const gradient = [_]f64{2.0 * params[0]};
        try optimizer.step(&gradient);
    }

    // Should converge close to x = 0
    try testing.expect(@abs(params[0]) < 0.1);
}

test "Adam: multivariate optimization" {
    const allocator = testing.allocator;

    // Minimize f(x,y) = x² + y² with (x₀, y₀) = (5, -5)
    var params = [_]f64{ 5.0, -5.0 };
    const config = Adam(f64).Config{ .learning_rate = 0.1 };

    var optimizer = try Adam(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    // Gradients: ∇f = (2x, 2y)
    for (0..100) |_| {
        const gradients = [_]f64{ 2.0 * params[0], 2.0 * params[1] };
        try optimizer.step(&gradients);
    }

    // Should converge close to (0, 0)
    try testing.expect(@abs(params[0]) < 0.1);
    try testing.expect(@abs(params[1]) < 0.1);
}

test "Adam: Rosenbrock function" {
    const allocator = testing.allocator;

    // Minimize Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
    // Optimal: (x*, y*) = (1, 1)
    var params = [_]f64{ 0.0, 0.0 };
    const config = Adam(f64).Config{ .learning_rate = 0.01 };

    var optimizer = try Adam(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    // ∇f = (-2(1-x) - 400x(y-x²), 200(y-x²))
    for (0..1000) |_| {
        const x = params[0];
        const y = params[1];
        const gradients = [_]f64{
            -2.0 * (1.0 - x) - 400.0 * x * (y - x * x),
            200.0 * (y - x * x),
        };
        try optimizer.step(&gradients);
    }

    // Should converge reasonably close to (1, 1)
    try testing.expectApproxEqAbs(@as(f64, 1.0), params[0], 0.1);
    try testing.expectApproxEqAbs(@as(f64, 1.0), params[1], 0.1);
}

test "Adam: momentum effect" {
    const allocator = testing.allocator;

    var params = [_]f64{1.0};
    const config = Adam(f64).Config{ .learning_rate = 0.1 };

    var optimizer = try Adam(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    // Constant gradient should accumulate momentum
    const gradient = [_]f64{0.5};
    try optimizer.step(&gradient);
    const param1 = params[0];
    try optimizer.step(&gradient);
    const param2 = params[0];

    // Second step should be larger due to momentum accumulation
    const step1 = 1.0 - param1;
    const step2 = param1 - param2;
    try testing.expect(step2 > step1);
}

test "Adam: bias correction" {
    const allocator = testing.allocator;

    var params = [_]f64{1.0};
    const config = Adam(f64).Config{ .learning_rate = 0.1 };

    var optimizer = try Adam(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    // First step: bias correction should increase effective learning rate
    const lr_initial = optimizer.getEffectiveLearningRate();
    try testing.expectApproxEqAbs(@as(f64, 0.1), lr_initial, 1e-10);

    const gradient = [_]f64{1.0};
    try optimizer.step(&gradient);
    const lr_after_step = optimizer.getEffectiveLearningRate();

    // After first step, effective LR should be lower due to bias correction
    try testing.expect(lr_after_step < lr_initial);
}

test "Adam: adaptive learning rates" {
    const allocator = testing.allocator;

    // Two parameters with different gradient magnitudes
    var params = [_]f64{ 1.0, 1.0 };
    const config = Adam(f64).Config{ .learning_rate = 0.1 };

    var optimizer = try Adam(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    // Large gradient for param[0], small for param[1]
    const gradients = [_]f64{ 10.0, 0.1 };
    try optimizer.step(&gradients);

    const step0 = 1.0 - params[0];
    const step1 = 1.0 - params[1];

    // Despite 100x gradient difference, step sizes should be more similar
    // due to adaptive learning rates (second moment normalization)
    const ratio = @abs(step0 / step1);
    try testing.expect(ratio < 50.0); // Much less than 100x difference
}

test "Adam: sparse gradients" {
    const allocator = testing.allocator;

    var params = [_]f64{ 1.0, 2.0, 3.0 };
    const config = Adam(f64).Config{ .learning_rate = 0.1 };

    var optimizer = try Adam(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    // Sparse gradient: only update first parameter
    const sparse_gradients = [_]f64{ 1.0, 0.0, 0.0 };
    try optimizer.step(&sparse_gradients);

    // Only first parameter should change significantly
    try testing.expect(@abs(params[0] - 1.0) > 0.01);
    try testing.expectApproxEqAbs(@as(f64, 2.0), params[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 3.0), params[2], 1e-10);
}

test "Adam: reset functionality" {
    const allocator = testing.allocator;

    var params = [_]f64{1.0};
    const config = Adam(f64).Config{ .learning_rate = 0.1 };

    var optimizer = try Adam(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    const gradient = [_]f64{1.0};
    try optimizer.step(&gradient);
    try testing.expectEqual(@as(usize, 1), optimizer.t);

    optimizer.reset();
    try testing.expectEqual(@as(usize, 0), optimizer.t);
    try testing.expectApproxEqAbs(@as(f64, 0.0), optimizer.m[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), optimizer.v[0], 1e-10);
}

test "Adam: timestep tracking" {
    const allocator = testing.allocator;

    var params = [_]f64{1.0};
    const config = Adam(f64).Config{};

    var optimizer = try Adam(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    try testing.expectEqual(@as(usize, 0), optimizer.getTimestep());

    const gradient = [_]f64{1.0};
    try optimizer.step(&gradient);
    try testing.expectEqual(@as(usize, 1), optimizer.getTimestep());

    try optimizer.step(&gradient);
    try testing.expectEqual(@as(usize, 2), optimizer.getTimestep());
}

test "Adam: f32 support" {
    const allocator = testing.allocator;

    var params = [_]f32{5.0};
    const config = Adam(f32).Config{ .learning_rate = 0.1 };

    var optimizer = try Adam(f32).init(allocator, &params, config);
    defer optimizer.deinit();

    // Minimize f(x) = x²
    for (0..50) |_| {
        const gradient = [_]f32{2.0 * params[0]};
        try optimizer.step(&gradient);
    }

    try testing.expect(@abs(params[0]) < 0.1);
}

test "Adam: empty parameters error" {
    const allocator = testing.allocator;

    var params: [0]f64 = undefined;
    const config = Adam(f64).Config{};

    const result = Adam(f64).init(allocator, &params, config);
    try testing.expectError(error.EmptyParameters, result);
}

test "Adam: gradient length mismatch" {
    const allocator = testing.allocator;

    var params = [_]f64{ 1.0, 2.0 };
    const config = Adam(f64).Config{};

    var optimizer = try Adam(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    const wrong_gradients = [_]f64{1.0}; // Wrong length
    const result = optimizer.step(&wrong_gradients);
    try testing.expectError(error.GradientLengthMismatch, result);
}

test "Adam: invalid config" {
    const allocator = testing.allocator;

    var params = [_]f64{1.0};

    // Invalid learning rate
    const config1 = Adam(f64).Config{ .learning_rate = -0.1 };
    try testing.expectError(error.InvalidLearningRate, Adam(f64).init(allocator, &params, config1));

    // Invalid beta1
    const config2 = Adam(f64).Config{ .beta1 = 1.0 };
    try testing.expectError(error.InvalidBeta1, Adam(f64).init(allocator, &params, config2));

    const config3 = Adam(f64).Config{ .beta1 = -0.1 };
    try testing.expectError(error.InvalidBeta1, Adam(f64).init(allocator, &params, config3));

    // Invalid beta2
    const config4 = Adam(f64).Config{ .beta2 = 1.0 };
    try testing.expectError(error.InvalidBeta2, Adam(f64).init(allocator, &params, config4));

    // Invalid epsilon
    const config5 = Adam(f64).Config{ .epsilon = -1e-8 };
    try testing.expectError(error.InvalidEpsilon, Adam(f64).init(allocator, &params, config5));
}

test "Adam: large scale optimization" {
    const allocator = testing.allocator;

    // 100-dimensional quadratic
    const params = try allocator.alloc(f64, 100);
    defer allocator.free(params);
    for (params) |*p| p.* = 10.0;

    const config = Adam(f64).Config{ .learning_rate = 0.1 };

    var optimizer = try Adam(f64).init(allocator, params, config);
    defer optimizer.deinit();

    const gradients = try allocator.alloc(f64, 100);
    defer allocator.free(gradients);

    for (0..200) |_| {
        // Compute gradients: g_i = 2 × p_i
        for (params, gradients) |p, *g| {
            g.* = 2.0 * p;
        }
        try optimizer.step(gradients);
    }

    // All parameters should converge close to 0
    for (params) |p| {
        try testing.expect(@abs(p) < 0.1);
    }
}

test "Adam: memory safety" {
    const allocator = testing.allocator;

    var params = [_]f64{ 1.0, 2.0, 3.0 };
    const config = Adam(f64).Config{};

    var optimizer = try Adam(f64).init(allocator, &params, config);
    defer optimizer.deinit();

    const gradient = [_]f64{ 0.5, 1.0, 1.5 };
    try optimizer.step(&gradient);

    // No memory leaks if using testing.allocator
}
