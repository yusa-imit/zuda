// LAMB (Layer-wise Adaptive Moments optimizer for Batch training)
//
// Algorithm: Layer-wise adaptation for large-batch training, extends LARS to adaptive methods
//
// Reference: You et al. (2019) "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
// Paper: https://arxiv.org/abs/1904.00962
//
// Key innovations:
// - Layer-wise learning rate adaptation (like LARS but for Adam)
// - Trust ratio: r_t = ||θ_{t-1}||₂ / (||m̂_t / (√v̂_t + ε)||₂ + λ||θ_{t-1}||₂)
// - Enables large batch training without loss of accuracy
// - Combines Adam's adaptive moments with LARS's layer-wise scaling
// - Decoupled weight decay (like AdamW)
//
// Algorithm steps:
// 1. First moment: m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
// 2. Second moment: v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²
// 3. Bias correction: m̂_t = m_t / (1 - β₁^t), v̂_t = v_t / (1 - β₂^t)
// 4. Adaptive update: u_t = m̂_t / (√v̂_t + ε)
// 5. Trust ratio: r_t = ||θ_{t-1}||₂ / (||u_t||₂ + λ||θ_{t-1}||₂)
// 6. Layer update: θ_t = θ_{t-1} - α × r_t × u_t - α × λ × θ_{t-1}
//
// Use cases:
// - Large batch training (batch sizes > 1024)
// - BERT, GPT, and other transformer models
// - Distributed training across multiple GPUs
// - When training time is critical
// - When computational efficiency with large batches is needed
//
// Trade-offs:
// - vs Adam: Better scaling to large batches, prevents divergence
// - vs LARS: Adaptive moments instead of plain gradients
// - vs AdamW: Layer-wise adaptation for better large-batch stability
// - vs SGD: More complex, but enables much larger batch sizes
//
// Configuration:
// - learning_rate: 0.001 (default, typical: 0.0001-0.01)
// - beta1: 0.9 (first moment decay)
// - beta2: 0.999 (second moment decay)
// - epsilon: 1e-6 (numerical stability, larger than Adam's 1e-8)
// - weight_decay: 0.01 (L2 penalty, typical: 0.01-0.1)
//
// Time: O(n) per update where n = number of parameters
// Space: O(n) for first and second moment vectors

const std = @import("std");
const Allocator = std.mem.Allocator;

pub fn LAMB(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        learning_rate: T,
        beta1: T,
        beta2: T,
        epsilon: T,
        weight_decay: T,
        timestep: usize,

        // State vectors (allocated on init)
        momentum: []T, // First moment (m_t)
        velocity: []T, // Second moment (v_t)

        /// Configuration for LAMB optimizer
        pub const Config = struct {
            learning_rate: T = 0.001,
            beta1: T = 0.9,
            beta2: T = 0.999,
            epsilon: T = 1e-6,
            weight_decay: T = 0.01,
        };

        /// Initialize LAMB optimizer
        ///
        /// Time: O(n) where n = number of parameters
        /// Space: O(n) for momentum and velocity vectors
        pub fn init(allocator: Allocator, num_params: usize, config: Config) !Self {
            if (num_params == 0) return error.EmptyParameters;
            if (config.learning_rate <= 0) return error.InvalidLearningRate;
            if (config.beta1 < 0 or config.beta1 >= 1) return error.InvalidBeta1;
            if (config.beta2 < 0 or config.beta2 >= 1) return error.InvalidBeta2;
            if (config.epsilon <= 0) return error.InvalidEpsilon;
            if (config.weight_decay < 0) return error.InvalidWeightDecay;

            const momentum = try allocator.alloc(T, num_params);
            errdefer allocator.free(momentum);
            @memset(momentum, 0);

            const velocity = try allocator.alloc(T, num_params);
            errdefer allocator.free(velocity);
            @memset(velocity, 0);

            return Self{
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
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.momentum);
            self.allocator.free(self.velocity);
        }

        /// Perform one optimization step
        ///
        /// Applies LAMB update with layer-wise trust ratio adaptation
        ///
        /// Time: O(n) where n = length of parameters
        /// Space: O(1)
        pub fn step(self: *Self, params: []T, gradients: []const T) !void {
            if (params.len != gradients.len) return error.LengthMismatch;
            if (params.len != self.momentum.len) return error.LengthMismatch;

            self.timestep += 1;
            const t = @as(T, @floatFromInt(self.timestep));

            // Bias correction factors
            const bias_correction1 = 1.0 - std.math.pow(T, self.beta1, t);
            const bias_correction2 = 1.0 - std.math.pow(T, self.beta2, t);

            // Update moments
            for (params, gradients, 0..) |_, g, i| {
                // First moment: m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
                self.momentum[i] = self.beta1 * self.momentum[i] + (1.0 - self.beta1) * g;

                // Second moment: v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²
                self.velocity[i] = self.beta2 * self.velocity[i] + (1.0 - self.beta2) * g * g;
            }

            // Compute adaptive update vector u and norms for trust ratio
            var param_norm: T = 0;
            var update_norm: T = 0;

            // Compute norms: ||θ||₂ and ||u||₂ where u = m̂ / (√v̂ + ε)
            for (params, 0..) |p, i| {
                // Bias-corrected moments
                const m_hat = self.momentum[i] / bias_correction1;
                const v_hat = self.velocity[i] / bias_correction2;

                // Adaptive update: u_t = m̂_t / (√v̂_t + ε)
                const u = m_hat / (@sqrt(v_hat) + self.epsilon);

                // Accumulate norms for trust ratio
                param_norm += p * p;
                update_norm += u * u;
            }

            // Compute trust ratio: r_t = ||θ||₂ / (||u||₂ + λ||θ||₂)
            param_norm = @sqrt(param_norm);
            update_norm = @sqrt(update_norm);

            // Prevent division by zero
            const denominator = update_norm + self.weight_decay * param_norm;
            const trust_ratio = if (denominator > 0) param_norm / denominator else 1.0;

            // Apply update with trust ratio and weight decay
            for (params, 0..) |*p, i| {
                // Recompute u from stored momentum/velocity (no temporary storage needed)
                const m_hat = self.momentum[i] / bias_correction1;
                const v_hat = self.velocity[i] / bias_correction2;
                const u = m_hat / (@sqrt(v_hat) + self.epsilon);

                // Layer update: θ_t = θ_{t-1} - α × r_t × u_t - α × λ × θ_{t-1}
                p.* = p.* - self.learning_rate * trust_ratio * u - self.learning_rate * self.weight_decay * p.*;
            }
        }

        /// Reset optimizer state (clear momentum and velocity)
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

test "LAMB: initialization with default config" {
    const allocator = testing.allocator;
    var opt = try LAMB(f64).init(allocator, 10, .{});
    defer opt.deinit();

    try expectEqual(@as(usize, 10), opt.momentum.len);
    try expectEqual(@as(usize, 10), opt.velocity.len);
    try expectEqual(@as(usize, 0), opt.timestep);
    try expectApproxEqAbs(@as(f64, 0.001), opt.learning_rate, 1e-9);
    try expectApproxEqAbs(@as(f64, 0.9), opt.beta1, 1e-9);
    try expectApproxEqAbs(@as(f64, 0.999), opt.beta2, 1e-9);
    try expectApproxEqAbs(@as(f64, 1e-6), opt.epsilon, 1e-9);
    try expectApproxEqAbs(@as(f64, 0.01), opt.weight_decay, 1e-9);

    // State vectors should be zero-initialized
    for (opt.momentum) |m| try expectApproxEqAbs(@as(f64, 0), m, 1e-9);
    for (opt.velocity) |v| try expectApproxEqAbs(@as(f64, 0), v, 1e-9);
}

test "LAMB: initialization with custom config" {
    const allocator = testing.allocator;
    var opt = try LAMB(f32).init(allocator, 5, .{
        .learning_rate = 0.01,
        .beta1 = 0.95,
        .beta2 = 0.9999,
        .epsilon = 1e-7,
        .weight_decay = 0.1,
    });
    defer opt.deinit();

    try expectApproxEqAbs(@as(f32, 0.01), opt.learning_rate, 1e-6);
    try expectApproxEqAbs(@as(f32, 0.95), opt.beta1, 1e-6);
    try expectApproxEqAbs(@as(f32, 0.9999), opt.beta2, 1e-6);
    try expectApproxEqAbs(@as(f32, 1e-7), opt.epsilon, 1e-6);
    try expectApproxEqAbs(@as(f32, 0.1), opt.weight_decay, 1e-6);
}

test "LAMB: simple quadratic convergence" {
    const allocator = testing.allocator;
    // Minimize f(x) = x² with gradient g = 2x
    var params = [_]f64{5.0};
    var opt = try LAMB(f64).init(allocator, 1, .{ .learning_rate = 0.1 });
    defer opt.deinit();

    // Take optimization steps
    var i: usize = 0;
    while (i < 50) : (i += 1) {
        const grad = [_]f64{2.0 * params[0]};
        try opt.step(&params, &grad);
    }

    // Should converge close to x=0
    try expectApproxEqAbs(@as(f64, 0), params[0], 0.1);
}

test "LAMB: multivariate quadratic optimization" {
    const allocator = testing.allocator;
    // Minimize f(x,y) = x² + y² with gradient ∇f = (2x, 2y)
    var params = [_]f32{ 3.0, -4.0 };
    var opt = try LAMB(f32).init(allocator, 2, .{ .learning_rate = 0.1 });
    defer opt.deinit();

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const grad = [_]f32{ 2.0 * params[0], 2.0 * params[1] };
        try opt.step(&params, &grad);
    }

    // Should converge to (0, 0)
    try expectApproxEqAbs(@as(f32, 0), params[0], 0.1);
    try expectApproxEqAbs(@as(f32, 0), params[1], 0.1);
}

test "LAMB: Rosenbrock function progress" {
    const allocator = testing.allocator;
    // Minimize Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
    // Global minimum at (1, 1)
    var params = [_]f64{ 0.5, 0.5 }; // Start closer to optimum
    var opt = try LAMB(f64).init(allocator, 2, .{
        .learning_rate = 0.005,
        .weight_decay = 0.0,
    });
    defer opt.deinit();

    const initial_x = params[0];
    const initial_y = params[1];

    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        const x = params[0];
        const y = params[1];

        // ∇f = (-2(1-x) - 400x(y-x²), 200(y-x²))
        const grad = [_]f64{
            -2.0 * (1.0 - x) - 400.0 * x * (y - x * x),
            200.0 * (y - x * x),
        };

        try opt.step(&params, &grad);
    }

    // LAMB should make progress - check we moved closer to optimum
    const initial_dist = @sqrt((initial_x - 1.0) * (initial_x - 1.0) + (initial_y - 1.0) * (initial_y - 1.0));
    const final_dist = @sqrt((params[0] - 1.0) * (params[0] - 1.0) + (params[1] - 1.0) * (params[1] - 1.0));
    try testing.expect(final_dist < initial_dist); // Made progress
}

test "LAMB: trust ratio computation" {
    const allocator = testing.allocator;
    var params = [_]f64{ 1.0, 2.0, 3.0 };
    const gradients = [_]f64{ 0.1, 0.2, 0.3 };

    var opt = try LAMB(f64).init(allocator, 3, .{
        .learning_rate = 0.01,
        .weight_decay = 0.01,
    });
    defer opt.deinit();

    const initial_params = params;
    try opt.step(&params, &gradients);

    // Trust ratio should scale the update
    // Verify that parameters changed (non-zero update)
    var changed = false;
    for (params, initial_params) |p, p0| {
        if (@abs(p - p0) > 1e-6) changed = true;
    }
    try testing.expect(changed);
}

test "LAMB: momentum accumulation" {
    const allocator = testing.allocator;
    var opt = try LAMB(f64).init(allocator, 2, .{ .beta1 = 0.9 });
    defer opt.deinit();

    var params = [_]f64{ 1.0, 2.0 };
    const grad = [_]f64{ 1.0, 2.0 };

    // First step
    try opt.step(&params, &grad);
    const m1 = opt.momentum[0];

    // Momentum should be non-zero after first step
    try testing.expect(m1 != 0);

    // Second step with same gradient
    try opt.step(&params, &grad);
    const m2 = opt.momentum[0];

    // Momentum should accumulate (increase)
    try testing.expect(@abs(m2) > @abs(m1));
}

test "LAMB: bias correction effect" {
    const allocator = testing.allocator;
    var opt = try LAMB(f64).init(allocator, 1, .{});
    defer opt.deinit();

    var params = [_]f64{1.0};
    const grad = [_]f64{1.0};

    const p0 = params[0];
    try opt.step(&params, &grad);
    const step1_change = @abs(params[0] - p0);

    // Reset and take another step
    opt.reset();
    params[0] = 1.0;
    try opt.step(&params, &grad);
    const step2_change = @abs(params[0] - 1.0);

    // Changes should be similar (bias correction works)
    try expectApproxEqAbs(step1_change, step2_change, 1e-6);
}

test "LAMB: weight decay effect" {
    const allocator = testing.allocator;
    var params_no_decay = [_]f64{ 1.0, 2.0 };
    var params_with_decay = [_]f64{ 1.0, 2.0 };
    const grad = [_]f64{ 0.1, 0.1 };

    var opt_no_decay = try LAMB(f64).init(allocator, 2, .{ .weight_decay = 0.0 });
    defer opt_no_decay.deinit();

    var opt_with_decay = try LAMB(f64).init(allocator, 2, .{ .weight_decay = 0.5 }); // Higher weight decay
    defer opt_with_decay.deinit();

    // Take multiple steps to see effect
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        try opt_no_decay.step(&params_no_decay, &grad);
        try opt_with_decay.step(&params_with_decay, &grad);
    }

    // Weight decay should shrink parameters more (sum of squared params)
    const norm_no_decay = params_no_decay[0] * params_no_decay[0] + params_no_decay[1] * params_no_decay[1];
    const norm_with_decay = params_with_decay[0] * params_with_decay[0] + params_with_decay[1] * params_with_decay[1];
    try testing.expect(norm_with_decay < norm_no_decay);
}

test "LAMB: sparse gradients" {
    const allocator = testing.allocator;
    var params = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const grad = [_]f64{ 1.0, 0.0, 0.0, 1.0 }; // Sparse

    var opt = try LAMB(f64).init(allocator, 4, .{});
    defer opt.deinit();

    try opt.step(&params, &grad);

    // Only first and last params should change significantly
    try testing.expect(params[0] != 1.0); // Changed
    try testing.expect(@abs(params[1] - 2.0) < 0.01); // Nearly unchanged
    try testing.expect(@abs(params[2] - 3.0) < 0.01); // Nearly unchanged
    try testing.expect(params[3] != 4.0); // Changed
}

test "LAMB: reset functionality" {
    const allocator = testing.allocator;
    var opt = try LAMB(f64).init(allocator, 3, .{});
    defer opt.deinit();

    var params = [_]f64{ 1.0, 2.0, 3.0 };
    const grad = [_]f64{ 0.1, 0.2, 0.3 };

    // Take a step
    try opt.step(&params, &grad);
    try testing.expect(opt.timestep == 1);
    try testing.expect(opt.momentum[0] != 0);

    // Reset
    opt.reset();
    try expectEqual(@as(usize, 0), opt.timestep);
    for (opt.momentum) |m| try expectApproxEqAbs(@as(f64, 0), m, 1e-9);
    for (opt.velocity) |v| try expectApproxEqAbs(@as(f64, 0), v, 1e-9);
}

test "LAMB: f32 support" {
    const allocator = testing.allocator;
    var params = [_]f32{ 2.0, -3.0 };
    const grad = [_]f32{ 0.4, -0.6 };

    var opt = try LAMB(f32).init(allocator, 2, .{});
    defer opt.deinit();

    try opt.step(&params, &grad);

    // Verify parameters changed
    try testing.expect(params[0] != 2.0);
    try testing.expect(params[1] != -3.0);
}

test "LAMB: large scale (100 parameters)" {
    const allocator = testing.allocator;
    const n = 100;
    var params = try allocator.alloc(f64, n);
    defer allocator.free(params);
    var grad = try allocator.alloc(f64, n);
    defer allocator.free(grad);

    // Initialize with smaller values for stability
    for (0..n) |i| {
        params[i] = (@as(f64, @floatFromInt(i)) - 50.0) * 0.1; // Scale down
    }

    var opt = try LAMB(f64).init(allocator, n, .{
        .learning_rate = 0.01,
        .weight_decay = 0.0,
    });
    defer opt.deinit();

    // Compute initial norm
    var initial_norm: f64 = 0;
    for (params) |p| initial_norm += p * p;
    initial_norm = @sqrt(initial_norm);

    // Take steps
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        for (0..n) |j| {
            grad[j] = params[j] * 0.1;
        }
        try opt.step(params, grad);
    }

    // Compute final norm - should be reduced
    var final_norm: f64 = 0;
    for (params) |p| final_norm += p * p;
    final_norm = @sqrt(final_norm);

    try testing.expect(final_norm < initial_norm * 0.95); // At least 5% reduction
}

test "LAMB: error - empty parameters" {
    const allocator = testing.allocator;
    try expectError(error.EmptyParameters, LAMB(f64).init(allocator, 0, .{}));
}

test "LAMB: error - invalid learning rate" {
    const allocator = testing.allocator;
    try expectError(error.InvalidLearningRate, LAMB(f64).init(allocator, 5, .{ .learning_rate = 0.0 }));
    try expectError(error.InvalidLearningRate, LAMB(f64).init(allocator, 5, .{ .learning_rate = -0.1 }));
}

test "LAMB: error - invalid beta1" {
    const allocator = testing.allocator;
    try expectError(error.InvalidBeta1, LAMB(f64).init(allocator, 5, .{ .beta1 = -0.1 }));
    try expectError(error.InvalidBeta1, LAMB(f64).init(allocator, 5, .{ .beta1 = 1.0 }));
    try expectError(error.InvalidBeta1, LAMB(f64).init(allocator, 5, .{ .beta1 = 1.5 }));
}

test "LAMB: error - invalid beta2" {
    const allocator = testing.allocator;
    try expectError(error.InvalidBeta2, LAMB(f64).init(allocator, 5, .{ .beta2 = -0.1 }));
    try expectError(error.InvalidBeta2, LAMB(f64).init(allocator, 5, .{ .beta2 = 1.0 }));
}

test "LAMB: error - invalid epsilon" {
    const allocator = testing.allocator;
    try expectError(error.InvalidEpsilon, LAMB(f64).init(allocator, 5, .{ .epsilon = 0.0 }));
    try expectError(error.InvalidEpsilon, LAMB(f64).init(allocator, 5, .{ .epsilon = -1e-8 }));
}

test "LAMB: error - invalid weight decay" {
    const allocator = testing.allocator;
    try expectError(error.InvalidWeightDecay, LAMB(f64).init(allocator, 5, .{ .weight_decay = -0.1 }));
}

test "LAMB: error - gradient length mismatch" {
    const allocator = testing.allocator;
    var opt = try LAMB(f64).init(allocator, 3, .{});
    defer opt.deinit();

    var params = [_]f64{ 1.0, 2.0, 3.0 };
    const grad = [_]f64{ 0.1, 0.2 }; // Wrong length

    try expectError(error.LengthMismatch, opt.step(&params, &grad));
}

test "LAMB: large batch stability" {
    const allocator = testing.allocator;
    // Simulate large-batch scenario
    var params = [_]f64{ 2.0, -2.0, 1.5, -1.5 }; // Smaller initial values
    var opt = try LAMB(f64).init(allocator, 4, .{
        .learning_rate = 0.05, // Moderate LR
        .weight_decay = 0.0,
    });
    defer opt.deinit();

    // Constant gradients (simulating batch average)
    const grad = [_]f64{ 0.4, -0.4, 0.3, -0.3 };

    const initial_norm = @sqrt(2.0 * 2.0 + 2.0 * 2.0 + 1.5 * 1.5 + 1.5 * 1.5);

    var i: usize = 0;
    while (i < 50) : (i += 1) {
        try opt.step(&params, &grad);

        // Check for stability (no NaN, no explosion)
        for (params) |p| {
            try testing.expect(!std.math.isNan(p));
            try testing.expect(@abs(p) < 100.0); // Reasonable bound
        }
    }

    // Should make progress toward zero
    const final_norm = @sqrt(params[0] * params[0] + params[1] * params[1] + params[2] * params[2] + params[3] * params[3]);
    try testing.expect(final_norm < initial_norm * 0.8); // At least 20% reduction
}

test "LAMB: convergence with varying gradients" {
    const allocator = testing.allocator;
    var params = [_]f64{ 10.0, -10.0 };
    var opt = try LAMB(f64).init(allocator, 2, .{
        .learning_rate = 0.1, // Higher LR
        .weight_decay = 0.0,
    });
    defer opt.deinit();

    // Simulate varying gradient magnitudes
    var i: usize = 0;
    while (i < 500) : (i += 1) {
        const scale: f64 = if (i % 2 == 0) 1.0 else 0.1;
        const grad = [_]f64{
            2.0 * params[0] * scale,
            2.0 * params[1] * scale,
        };
        try opt.step(&params, &grad);
    }

    // Should make significant progress despite varying gradients
    const final_norm = @sqrt(params[0] * params[0] + params[1] * params[1]);
    const initial_norm = @sqrt(10.0 * 10.0 + 10.0 * 10.0);
    try testing.expect(final_norm < initial_norm * 0.3); // At least 70% reduction
}
