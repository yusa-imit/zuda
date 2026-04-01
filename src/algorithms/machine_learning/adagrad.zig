const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Adagrad (Adaptive Gradient) optimizer
///
/// Adaptive learning rate optimizer that accumulates squared gradients over time.
/// Each parameter gets its own learning rate that decreases over time, with faster
/// decrease for parameters with larger gradients.
///
/// Algorithm:
/// 1. Compute gradient g_t = ∇f(θ_t)
/// 2. Accumulate squared gradient: G_t = G_{t-1} + g_t²
/// 3. Update parameters: θ_t = θ_{t-1} - α / (√G_t + ε) × g_t
///
/// Key features:
/// - Adaptive per-parameter learning rates
/// - No manual learning rate tuning needed for sparse features
/// - Monotonically decreasing learning rates (G_t always increases)
/// - Eliminates need for manual learning rate schedules
/// - Performs well on sparse data (e.g., NLP, recommender systems)
///
/// Time complexity: O(n) per update where n = number of parameters
/// Space complexity: O(n) for gradient accumulator
///
/// Use cases:
/// - Sparse data (NLP, text classification, word embeddings)
/// - Convex optimization problems
/// - When different features have very different scales
/// - Baseline for comparing adaptive methods
///
/// Trade-offs:
/// - vs SGD: adaptive rates eliminate manual tuning, but learning can stop too early
/// - vs RMSprop: accumulates all gradients (can decay too aggressively), RMSprop uses moving average
/// - vs Adam: simpler (no momentum), but learning rate can become infinitesimally small
/// - vs Adadelta: requires initial learning rate, Adadelta doesn't
///
/// Limitations:
/// - Learning rate monotonically decreases (can stop learning too early)
/// - Not suitable for non-convex optimization (deep learning)
/// - RMSprop/Adam generally preferred for neural networks
///
/// References:
/// - Duchi et al. (2011): "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"
/// - Used in Google's word2vec implementation
pub fn Adagrad(comptime T: type) type {
    if (T != f32 and T != f64) {
        @compileError("Adagrad only supports f32 and f64 types");
    }

    return struct {
        allocator: Allocator,
        learning_rate: T,
        epsilon: T, // numerical stability constant
        weight_decay: T, // L2 regularization

        // State
        G: []T, // accumulated squared gradients

        const Self = @This();

        /// Configuration for Adagrad optimizer
        pub const Config = struct {
            learning_rate: T = 0.01, // typical: 0.01-0.001
            epsilon: T = 1e-8, // prevents division by zero
            weight_decay: T = 0.0, // L2 penalty
        };

        /// Initialize Adagrad optimizer
        ///
        /// Time: O(n) | Space: O(n)
        pub fn init(allocator: Allocator, num_params: usize, config: Config) !Self {
            if (num_params == 0) {
                return error.EmptyParameters;
            }
            if (config.learning_rate <= 0) {
                return error.InvalidLearningRate;
            }
            if (config.epsilon <= 0) {
                return error.InvalidEpsilon;
            }
            if (config.weight_decay < 0) {
                return error.InvalidWeightDecay;
            }

            const G = try allocator.alloc(T, num_params);
            errdefer allocator.free(G);
            @memset(G, 0);

            return Self{
                .allocator = allocator,
                .learning_rate = config.learning_rate,
                .epsilon = config.epsilon,
                .weight_decay = config.weight_decay,
                .G = G,
            };
        }

        /// Free resources
        ///
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.G);
        }

        /// Perform single optimization step
        ///
        /// Updates parameters in-place using accumulated squared gradients.
        ///
        /// Time: O(n) | Space: O(1)
        pub fn step(self: *Self, params: []T, gradients: []const T) !void {
            if (params.len != self.G.len) {
                return error.ParameterLengthMismatch;
            }
            if (gradients.len != self.G.len) {
                return error.GradientLengthMismatch;
            }

            for (params, gradients, self.G) |*param, grad, *g_accum| {
                // Apply weight decay if configured
                var effective_grad = grad;
                if (self.weight_decay > 0) {
                    effective_grad += self.weight_decay * param.*;
                }

                // Accumulate squared gradient: G_t = G_{t-1} + g_t²
                g_accum.* += effective_grad * effective_grad;

                // Compute adaptive learning rate: α / (√G_t + ε)
                const adapted_lr = self.learning_rate / (@sqrt(g_accum.*) + self.epsilon);

                // Update parameter: θ_t = θ_{t-1} - lr_adapted × g_t
                param.* -= adapted_lr * effective_grad;
            }
        }

        /// Reset optimizer state (clears accumulated gradients)
        ///
        /// Time: O(n) | Space: O(1)
        pub fn reset(self: *Self) void {
            @memset(self.G, 0);
        }

        /// Get current effective learning rate for each parameter
        ///
        /// Returns array of per-parameter learning rates: α / (√G_t + ε)
        /// Caller owns returned memory.
        ///
        /// Time: O(n) | Space: O(n)
        pub fn getEffectiveLearningRates(self: *const Self, allocator: Allocator) ![]T {
            const rates = try allocator.alloc(T, self.G.len);
            errdefer allocator.free(rates);

            for (rates, self.G) |*rate, g_accum| {
                rate.* = self.learning_rate / (@sqrt(g_accum) + self.epsilon);
            }

            return rates;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "Adagrad: initialization" {
    const AdagradF64 = Adagrad(f64);
    var optimizer = try AdagradF64.init(testing.allocator, 3, .{});
    defer optimizer.deinit();

    try testing.expectEqual(@as(f64, 0.01), optimizer.learning_rate);
    try testing.expectEqual(@as(f64, 1e-8), optimizer.epsilon);
    try testing.expectEqual(@as(usize, 3), optimizer.G.len);
    for (optimizer.G) |g| {
        try testing.expectEqual(@as(f64, 0), g);
    }
}

test "Adagrad: custom configuration" {
    const AdagradF32 = Adagrad(f32);
    const config = AdagradF32.Config{
        .learning_rate = 0.001,
        .epsilon = 1e-10,
        .weight_decay = 0.0001,
    };
    var optimizer = try AdagradF32.init(testing.allocator, 5, config);
    defer optimizer.deinit();

    try testing.expectEqual(@as(f32, 0.001), optimizer.learning_rate);
    try testing.expectEqual(@as(f32, 1e-10), optimizer.epsilon);
    try testing.expectEqual(@as(f32, 0.0001), optimizer.weight_decay);
}

test "Adagrad: simple quadratic optimization" {
    const AdagradF64 = Adagrad(f64);
    var optimizer = try AdagradF64.init(testing.allocator, 1, .{ .learning_rate = 0.1 });
    defer optimizer.deinit();

    // Minimize f(x) = x², gradient = 2x, optimal x = 0
    var params = [_]f64{10.0};
    const initial = params[0];

    for (0..50) |_| {
        const grad = [_]f64{2.0 * params[0]};
        try optimizer.step(&params, &grad);
    }

    // Should converge toward zero
    try testing.expect(@abs(params[0]) < @abs(initial));
    try testing.expect(@abs(params[0]) < 1.0);
}

test "Adagrad: multivariate quadratic" {
    const AdagradF64 = Adagrad(f64);
    var optimizer = try AdagradF64.init(testing.allocator, 2, .{ .learning_rate = 0.1 });
    defer optimizer.deinit();

    // Minimize f(x,y) = x² + 4y²
    var params = [_]f64{ 5.0, 5.0 };

    for (0..100) |_| {
        const grads = [_]f64{
            2.0 * params[0], // ∂f/∂x = 2x
            8.0 * params[1], // ∂f/∂y = 8y
        };
        try optimizer.step(&params, &grads);
    }

    // Both should converge toward zero
    try testing.expect(@abs(params[0]) < 0.5);
    try testing.expect(@abs(params[1]) < 0.5);
}

test "Adagrad: adaptive learning rates decrease over time" {
    const AdagradF64 = Adagrad(f64);
    var optimizer = try AdagradF64.init(testing.allocator, 1, .{ .learning_rate = 1.0 });
    defer optimizer.deinit();

    // Check that effective learning rate decreases as G accumulates
    const rates1 = try optimizer.getEffectiveLearningRates(testing.allocator);
    defer testing.allocator.free(rates1);
    const initial_rate = rates1[0];

    // Simulate gradient step
    var params = [_]f64{1.0};
    const grads = [_]f64{2.0};
    try optimizer.step(&params, &grads);

    const rates2 = try optimizer.getEffectiveLearningRates(testing.allocator);
    defer testing.allocator.free(rates2);

    // Learning rate should decrease after accumulating gradients
    try testing.expect(rates2[0] < initial_rate);
}

test "Adagrad: different gradient magnitudes get different rates" {
    const AdagradF64 = Adagrad(f64);
    var optimizer = try AdagradF64.init(testing.allocator, 2, .{ .learning_rate = 1.0 });
    defer optimizer.deinit();

    // Parameter 0: large gradients, parameter 1: small gradients
    var params = [_]f64{ 1.0, 1.0 };

    for (0..10) |_| {
        const grads = [_]f64{ 10.0, 0.1 }; // very different magnitudes
        try optimizer.step(&params, &grads);
    }

    const rates = try optimizer.getEffectiveLearningRates(testing.allocator);
    defer testing.allocator.free(rates);

    // Parameter with larger gradients should have smaller effective learning rate
    try testing.expect(rates[0] < rates[1]);
}

test "Adagrad: sparse gradients" {
    const AdagradF64 = Adagrad(f64);
    var optimizer = try AdagradF64.init(testing.allocator, 10, .{});
    defer optimizer.deinit();

    var params = [_]f64{1.0} ** 10;

    // Sparse update: only update parameters 2 and 7
    for (0..20) |_| {
        var grads = [_]f64{0.0} ** 10;
        grads[2] = 0.5;
        grads[7] = 0.5;
        try optimizer.step(&params, &grads);
    }

    // Parameters with non-zero gradients should have moved
    try testing.expect(@abs(params[2] - 1.0) > 0.01);
    try testing.expect(@abs(params[7] - 1.0) > 0.01);

    // Parameters with zero gradients should be unchanged
    try testing.expectEqual(@as(f64, 1.0), params[0]);
    try testing.expectEqual(@as(f64, 1.0), params[5]);
}

test "Adagrad: weight decay" {
    const AdagradF64 = Adagrad(f64);
    var optimizer = try AdagradF64.init(testing.allocator, 1, .{
        .learning_rate = 0.1,
        .weight_decay = 0.01,
    });
    defer optimizer.deinit();

    var params = [_]f64{10.0};
    const grads = [_]f64{0.0}; // zero gradient, only weight decay active

    const initial = params[0];
    try optimizer.step(&params, &grads);

    // Weight decay should have reduced the parameter (L2 penalty)
    try testing.expect(@abs(params[0]) < @abs(initial));
}

test "Adagrad: reset" {
    const AdagradF64 = Adagrad(f64);
    var optimizer = try AdagradF64.init(testing.allocator, 2, .{});
    defer optimizer.deinit();

    // Accumulate some gradients
    var params = [_]f64{ 1.0, 1.0 };
    const grads = [_]f64{ 2.0, 3.0 };
    try optimizer.step(&params, &grads);

    // Check that G is non-zero
    try testing.expect(optimizer.G[0] > 0);
    try testing.expect(optimizer.G[1] > 0);

    // Reset
    optimizer.reset();

    // Check that G is zero again
    try testing.expectEqual(@as(f64, 0), optimizer.G[0]);
    try testing.expectEqual(@as(f64, 0), optimizer.G[1]);
}

test "Adagrad: f32 support" {
    const AdagradF32 = Adagrad(f32);
    var optimizer = try AdagradF32.init(testing.allocator, 2, .{});
    defer optimizer.deinit();

    var params = [_]f32{ 1.0, 2.0 };
    const grads = [_]f32{ 0.5, 0.5 };
    try optimizer.step(&params, &grads);

    // Just verify it compiles and runs
    try testing.expect(params[0] < 1.0);
    try testing.expect(params[1] < 2.0);
}

test "Adagrad: large scale (100 parameters)" {
    const AdagradF64 = Adagrad(f64);
    var optimizer = try AdagradF64.init(testing.allocator, 100, .{});
    defer optimizer.deinit();

    var params = [_]f64{1.0} ** 100;
    var grads = [_]f64{0.1} ** 100;

    // Simulate training
    for (0..50) |_| {
        try optimizer.step(&params, &grads);
    }

    // All parameters should have moved from initial value
    for (params) |p| {
        try testing.expect(@abs(p - 1.0) > 0.01);
    }
}

test "Adagrad: convergence on convex problem" {
    const AdagradF64 = Adagrad(f64);
    var optimizer = try AdagradF64.init(testing.allocator, 1, .{ .learning_rate = 0.5 });
    defer optimizer.deinit();

    // Minimize f(x) = (x-3)², optimal at x=3
    var params = [_]f64{0.0};

    for (0..100) |_| {
        const grad = [_]f64{2.0 * (params[0] - 3.0)};
        try optimizer.step(&params, &grad);
    }

    // Should converge close to optimal value
    try testing.expectApproxEqAbs(@as(f64, 3.0), params[0], 0.1);
}

test "Adagrad: error handling - empty parameters" {
    const AdagradF64 = Adagrad(f64);
    const result = AdagradF64.init(testing.allocator, 0, .{});
    try testing.expectError(error.EmptyParameters, result);
}

test "Adagrad: error handling - parameter length mismatch" {
    const AdagradF64 = Adagrad(f64);
    var optimizer = try AdagradF64.init(testing.allocator, 3, .{});
    defer optimizer.deinit();

    var params = [_]f64{ 1.0, 2.0 }; // length 2, but optimizer expects 3
    const grads = [_]f64{ 0.1, 0.1 };
    const result = optimizer.step(&params, &grads);
    try testing.expectError(error.ParameterLengthMismatch, result);
}

test "Adagrad: error handling - gradient length mismatch" {
    const AdagradF64 = Adagrad(f64);
    var optimizer = try AdagradF64.init(testing.allocator, 2, .{});
    defer optimizer.deinit();

    var params = [_]f64{ 1.0, 2.0 };
    const grads = [_]f64{ 0.1, 0.1, 0.1 }; // length 3, but optimizer expects 2
    const result = optimizer.step(&params, &grads);
    try testing.expectError(error.GradientLengthMismatch, result);
}

test "Adagrad: error handling - invalid learning rate" {
    const AdagradF64 = Adagrad(f64);
    const result = AdagradF64.init(testing.allocator, 2, .{ .learning_rate = -0.1 });
    try testing.expectError(error.InvalidLearningRate, result);
}

test "Adagrad: error handling - invalid epsilon" {
    const AdagradF64 = Adagrad(f64);
    const result = AdagradF64.init(testing.allocator, 2, .{ .epsilon = 0.0 });
    try testing.expectError(error.InvalidEpsilon, result);
}

test "Adagrad: error handling - invalid weight decay" {
    const AdagradF64 = Adagrad(f64);
    const result = AdagradF64.init(testing.allocator, 2, .{ .weight_decay = -0.1 });
    try testing.expectError(error.InvalidWeightDecay, result);
}

test "Adagrad: memory safety" {
    const AdagradF64 = Adagrad(f64);
    var optimizer = try AdagradF64.init(testing.allocator, 1000, .{});
    defer optimizer.deinit();

    const params = try testing.allocator.alloc(f64, 1000);
    defer testing.allocator.free(params);
    @memset(params, 1.0);

    const grads = try testing.allocator.alloc(f64, 1000);
    defer testing.allocator.free(grads);
    @memset(grads, 0.1);

    // Multiple steps to test memory safety
    for (0..100) |_| {
        try optimizer.step(params, grads);
    }
}
