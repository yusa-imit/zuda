/// Lookahead Optimizer
///
/// Lookahead maintains two sets of weights: fast weights updated by an inner optimizer,
/// and slow weights that are periodically interpolated towards the fast weights.
///
/// Reference: Zhang et al. (2019) "Lookahead Optimizer: k steps forward, 1 step back" (NeurIPS 2019)
///
/// Algorithm:
/// 1. Fast weights θ_fast updated by inner optimizer (e.g., Adam, SGD)
/// 2. Every k steps, slow weights updated: θ_slow = θ_slow + α(θ_fast - θ_slow)
/// 3. Final parameters use slow weights (more stable, better generalization)
///
/// Key properties:
/// - Reduces variance of gradient-based optimizers
/// - Improves convergence and generalization
/// - Can wrap any optimizer (meta-optimizer)
/// - Negligible overhead (~2 parameter copies)
/// - Less sensitive to hyperparameter choices of inner optimizer
///
/// Typical hyperparameters:
/// - k (sync_period): 5-10 steps between slow weight updates (default: 5)
/// - α (slow_step_size): 0.5 (default, typical range: 0.5-0.8)
///
/// Time: O(n) per update where n = number of parameters (same as inner optimizer)
/// Space: O(n) for slow weights storage (additional to inner optimizer's state)
///
/// Use cases:
/// - Deep learning when inner optimizer has high variance (e.g., Adam with high LR)
/// - Transfer learning / fine-tuning (more stable updates)
/// - Training GANs (stabilizes adversarial optimization)
/// - Any scenario requiring robustness to hyperparameter choices
/// - Research baselines (often improves any optimizer out-of-the-box)

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Lookahead optimizer configuration
pub const LookaheadConfig = struct {
    /// Number of fast weight updates between slow weight updates
    sync_period: u32 = 5,
    /// Step size for slow weight updates (α in paper)
    slow_step_size: f64 = 0.5,

    /// Validate configuration parameters
    pub fn validate(self: LookaheadConfig) !void {
        if (self.sync_period == 0) return error.InvalidSyncPeriod;
        if (self.slow_step_size <= 0 or self.slow_step_size > 1.0) return error.InvalidSlowStepSize;
    }
};

/// Lookahead optimizer state
///
/// Generic over floating-point type T (f32 or f64)
pub fn Lookahead(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        config: LookaheadConfig,
        /// Slow weights (periodically updated)
        slow_weights: []T,
        /// Fast weights (updated by inner optimizer, aliased with parameters)
        fast_weights: []const T,
        /// Step counter for synchronization
        step: u32,

        /// Initialize Lookahead optimizer
        ///
        /// Time: O(n), Space: O(n)
        pub fn init(allocator: Allocator, parameters: []const T, config: LookaheadConfig) !Self {
            try config.validate();

            // Allocate slow weights and initialize to current parameters
            const slow_weights = try allocator.alloc(T, parameters.len);
            errdefer allocator.free(slow_weights);

            @memcpy(slow_weights, parameters);

            return Self{
                .allocator = allocator,
                .config = config,
                .slow_weights = slow_weights,
                .fast_weights = parameters,
                .step = 0,
            };
        }

        /// Free allocated memory
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.slow_weights);
        }

        /// Update parameters after inner optimizer step
        ///
        /// Call this AFTER the inner optimizer has updated the fast weights.
        /// Every k steps, this interpolates slow weights toward fast weights.
        ///
        /// Time: O(1) most steps, O(n) every k steps
        /// Space: O(1)
        ///
        /// Returns: true if slow weight update occurred (every k steps)
        pub fn step_update(self: *Self, parameters: []T) !bool {
            if (parameters.len != self.slow_weights.len) return error.ParameterLengthMismatch;

            self.step += 1;

            // Check if it's time to update slow weights
            if (self.step % self.config.sync_period == 0) {
                // θ_slow = θ_slow + α(θ_fast - θ_slow)
                const alpha = @as(T, @floatCast(self.config.slow_step_size));
                for (self.slow_weights, 0..) |*slow_w, i| {
                    const fast_w = parameters[i];
                    slow_w.* = slow_w.* + alpha * (fast_w - slow_w.*);
                }

                // Copy slow weights back to parameters
                @memcpy(parameters, self.slow_weights);

                return true; // Slow update occurred
            }

            return false; // No slow update
        }

        /// Reset optimizer state
        ///
        /// Time: O(n), Space: O(1)
        pub fn reset(self: *Self, parameters: []const T) !void {
            if (parameters.len != self.slow_weights.len) return error.ParameterLengthMismatch;

            @memcpy(self.slow_weights, parameters);
            self.step = 0;
        }

        /// Get current slow weights (for inspection/checkpointing)
        ///
        /// Time: O(1), Space: O(1)
        pub fn getSlowWeights(self: *const Self) []const T {
            return self.slow_weights;
        }

        /// Get current step count
        ///
        /// Time: O(1), Space: O(1)
        pub fn getStep(self: *const Self) u32 {
            return self.step;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "lookahead: initialization with default config" {
    const allocator = testing.allocator;
    const params = [_]f64{ 1.0, 2.0, 3.0 };

    var lookahead = try Lookahead(f64).init(allocator, &params, .{});
    defer lookahead.deinit();

    try testing.expectEqual(@as(usize, 3), lookahead.slow_weights.len);
    try testing.expectEqual(@as(u32, 0), lookahead.step);
    try testing.expectEqual(@as(u32, 5), lookahead.config.sync_period);
    try testing.expectApproxEqAbs(@as(f64, 0.5), lookahead.config.slow_step_size, 1e-10);

    // Slow weights should be initialized to parameters
    for (params, 0..) |p, i| {
        try testing.expectApproxEqAbs(p, lookahead.slow_weights[i], 1e-10);
    }
}

test "lookahead: initialization with custom config" {
    const allocator = testing.allocator;
    const params = [_]f64{ 1.0, 2.0 };

    const config = LookaheadConfig{
        .sync_period = 10,
        .slow_step_size = 0.7,
    };

    var lookahead = try Lookahead(f64).init(allocator, &params, config);
    defer lookahead.deinit();

    try testing.expectEqual(@as(u32, 10), lookahead.config.sync_period);
    try testing.expectApproxEqAbs(@as(f64, 0.7), lookahead.config.slow_step_size, 1e-10);
}

test "lookahead: step update without synchronization" {
    const allocator = testing.allocator;
    var params = [_]f64{ 1.0, 2.0, 3.0 };

    var lookahead = try Lookahead(f64).init(allocator, &params, .{ .sync_period = 5 });
    defer lookahead.deinit();

    // Simulate inner optimizer updates (steps 1-4, no sync)
    for (0..4) |i| {
        // Inner optimizer moves fast weights
        params[0] += 0.1;
        params[1] += 0.1;
        params[2] += 0.1;

        const updated = try lookahead.step_update(&params);
        try testing.expect(!updated); // No slow update yet
        try testing.expectEqual(@as(u32, @intCast(i + 1)), lookahead.step);
    }

    // Fast weights changed, slow weights unchanged
    try testing.expectApproxEqAbs(1.4, params[0], 1e-10);
    try testing.expectApproxEqAbs(1.0, lookahead.slow_weights[0], 1e-10);
}

test "lookahead: step update with synchronization" {
    const allocator = testing.allocator;
    var params = [_]f64{ 1.0, 2.0, 3.0 };

    var lookahead = try Lookahead(f64).init(allocator, &params, .{ .sync_period = 5, .slow_step_size = 0.5 });
    defer lookahead.deinit();

    // Steps 1-4: no sync
    for (0..4) |_| {
        params[0] += 0.1;
        _ = try lookahead.step_update(&params);
    }

    try testing.expectApproxEqAbs(1.4, params[0], 1e-10);

    // Step 5: synchronization occurs
    params[0] += 0.1; // Fast weight: 1.5
    const updated = try lookahead.step_update(&params);
    try testing.expect(updated); // Slow update occurred
    try testing.expectEqual(@as(u32, 5), lookahead.step);

    // Slow weight update: θ_slow = 1.0 + 0.5 * (1.5 - 1.0) = 1.25
    try testing.expectApproxEqAbs(1.25, lookahead.slow_weights[0], 1e-10);
    // Parameters should be copied from slow weights
    try testing.expectApproxEqAbs(1.25, params[0], 1e-10);
}

test "lookahead: multiple synchronization periods" {
    const allocator = testing.allocator;
    var params = [_]f64{10.0};

    var lookahead = try Lookahead(f64).init(allocator, &params, .{ .sync_period = 3, .slow_step_size = 0.5 });
    defer lookahead.deinit();

    // Period 1: steps 1-3
    params[0] = 11.0;
    _ = try lookahead.step_update(&params); // step 1
    params[0] = 12.0;
    _ = try lookahead.step_update(&params); // step 2
    params[0] = 13.0;
    const updated1 = try lookahead.step_update(&params); // step 3: sync
    try testing.expect(updated1);
    // θ_slow = 10.0 + 0.5 * (13.0 - 10.0) = 11.5
    try testing.expectApproxEqAbs(11.5, params[0], 1e-10);

    // Period 2: steps 4-6
    params[0] = 12.0;
    _ = try lookahead.step_update(&params); // step 4
    params[0] = 13.0;
    _ = try lookahead.step_update(&params); // step 5
    params[0] = 14.0;
    const updated2 = try lookahead.step_update(&params); // step 6: sync
    try testing.expect(updated2);
    // θ_slow = 11.5 + 0.5 * (14.0 - 11.5) = 12.75
    try testing.expectApproxEqAbs(12.75, params[0], 1e-10);
}

test "lookahead: slow step size effect" {
    const allocator = testing.allocator;
    var params = [_]f64{0.0};

    // α = 0.3 (slower interpolation)
    var lookahead = try Lookahead(f64).init(allocator, &params, .{ .sync_period = 1, .slow_step_size = 0.3 });
    defer lookahead.deinit();

    params[0] = 10.0;
    _ = try lookahead.step_update(&params);
    // θ_slow = 0.0 + 0.3 * (10.0 - 0.0) = 3.0
    try testing.expectApproxEqAbs(3.0, params[0], 1e-10);

    // α = 0.8 (faster interpolation)
    try lookahead.reset(&[_]f64{0.0});
    lookahead.config.slow_step_size = 0.8;
    params[0] = 10.0;
    _ = try lookahead.step_update(&params);
    // θ_slow = 0.0 + 0.8 * (10.0 - 0.0) = 8.0
    try testing.expectApproxEqAbs(8.0, params[0], 1e-10);
}

test "lookahead: reset functionality" {
    const allocator = testing.allocator;
    var params = [_]f64{ 1.0, 2.0 };

    var lookahead = try Lookahead(f64).init(allocator, &params, .{});
    defer lookahead.deinit();

    // Simulate some steps
    for (0..7) |_| {
        params[0] += 0.1;
        _ = try lookahead.step_update(&params);
    }

    try testing.expect(lookahead.step > 0);

    // Reset to new parameters
    const new_params = [_]f64{ 5.0, 6.0 };
    try lookahead.reset(&new_params);

    try testing.expectEqual(@as(u32, 0), lookahead.step);
    try testing.expectApproxEqAbs(5.0, lookahead.slow_weights[0], 1e-10);
    try testing.expectApproxEqAbs(6.0, lookahead.slow_weights[1], 1e-10);
}

test "lookahead: f32 support" {
    const allocator = testing.allocator;
    var params = [_]f32{ 1.0, 2.0 };

    var lookahead = try Lookahead(f32).init(allocator, &params, .{ .sync_period = 2 });
    defer lookahead.deinit();

    params[0] = 3.0;
    _ = try lookahead.step_update(&params);
    params[0] = 5.0;
    _ = try lookahead.step_update(&params);

    // θ_slow = 1.0 + 0.5 * (5.0 - 1.0) = 3.0
    try testing.expectApproxEqAbs(@as(f32, 3.0), params[0], 1e-5);
}

test "lookahead: error on empty parameters" {
    const allocator = testing.allocator;
    const params = [_]f64{};

    // Should succeed with empty parameters
    var lookahead = try Lookahead(f64).init(allocator, &params, .{});
    defer lookahead.deinit();

    try testing.expectEqual(@as(usize, 0), lookahead.slow_weights.len);
}

test "lookahead: error on mismatched lengths" {
    const allocator = testing.allocator;
    const params = [_]f64{ 1.0, 2.0 };

    var lookahead = try Lookahead(f64).init(allocator, &params, .{});
    defer lookahead.deinit();

    var wrong_params = [_]f64{ 1.0, 2.0, 3.0 };
    try testing.expectError(error.ParameterLengthMismatch, lookahead.step_update(&wrong_params));
    try testing.expectError(error.ParameterLengthMismatch, lookahead.reset(&wrong_params));
}

test "lookahead: error on invalid config - zero sync period" {
    const allocator = testing.allocator;
    const params = [_]f64{1.0};

    const config = LookaheadConfig{ .sync_period = 0 };
    try testing.expectError(error.InvalidSyncPeriod, Lookahead(f64).init(allocator, &params, config));
}

test "lookahead: error on invalid config - negative slow step size" {
    const allocator = testing.allocator;
    const params = [_]f64{1.0};

    const config = LookaheadConfig{ .slow_step_size = -0.1 };
    try testing.expectError(error.InvalidSlowStepSize, Lookahead(f64).init(allocator, &params, config));
}

test "lookahead: error on invalid config - slow step size > 1" {
    const allocator = testing.allocator;
    const params = [_]f64{1.0};

    const config = LookaheadConfig{ .slow_step_size = 1.1 };
    try testing.expectError(error.InvalidSlowStepSize, Lookahead(f64).init(allocator, &params, config));
}

test "lookahead: large scale optimization (100 parameters)" {
    const allocator = testing.allocator;
    const params = try allocator.alloc(f64, 100);
    defer allocator.free(params);

    // Initialize to [0, 1, 2, ..., 99]
    for (params, 0..) |*p, i| {
        p.* = @floatFromInt(i);
    }

    var lookahead = try Lookahead(f64).init(allocator, params, .{ .sync_period = 10, .slow_step_size = 0.6 });
    defer lookahead.deinit();

    // Simulate 25 steps (2 full periods + 5 steps)
    for (0..25) |step| {
        // Inner optimizer increases all parameters by 0.1
        for (params) |*p| {
            p.* += 0.1;
        }
        _ = try lookahead.step_update(params);

        // Check synchronization at step 10 and 20
        if (step == 9) { // step 10 (0-indexed)
            // After 10 steps: fast = 0 + 10*0.1 = 1.0
            // Slow update: θ_slow = 0.0 + 0.6 * (1.0 - 0.0) = 0.6
            try testing.expectApproxEqAbs(0.6, params[0], 1e-10);
        } else if (step == 19) { // step 20
            // After 20 steps: fast would be 2.0, but we synced at step 10 to 0.6
            // Steps 11-20: 0.6 + 10*0.1 = 1.6
            // Slow update: θ_slow = 0.6 + 0.6 * (1.6 - 0.6) = 1.2
            try testing.expectApproxEqAbs(1.2, params[0], 1e-10);
        }
    }
}

test "lookahead: quadratic optimization with SGD inner optimizer" {
    const allocator = testing.allocator;
    // Minimize f(x) = (x - 5)^2 starting from x = 0
    var params = [_]f64{0.0};

    var lookahead = try Lookahead(f64).init(allocator, &params, .{ .sync_period = 3, .slow_step_size = 0.5 });
    defer lookahead.deinit();

    const lr = 0.1;

    // Simulate 15 steps of gradient descent
    for (0..15) |_| {
        // Gradient: df/dx = 2(x - 5)
        const x = params[0];
        const grad = 2.0 * (x - 5.0);

        // SGD update: x = x - lr * grad
        params[0] = x - lr * grad;

        _ = try lookahead.step_update(&params);
    }

    // Should converge towards 5.0 (with slow weights providing stability)
    // Fast SGD would overshoot, but Lookahead stabilizes
    try testing.expect(params[0] > 3.0 and params[0] < 7.0);
}

test "lookahead: getSlowWeights accessor" {
    const allocator = testing.allocator;
    const params = [_]f64{ 1.0, 2.0, 3.0 };

    var lookahead = try Lookahead(f64).init(allocator, &params, .{});
    defer lookahead.deinit();

    const slow = lookahead.getSlowWeights();
    try testing.expectEqual(@as(usize, 3), slow.len);
    try testing.expectApproxEqAbs(1.0, slow[0], 1e-10);
    try testing.expectApproxEqAbs(2.0, slow[1], 1e-10);
    try testing.expectApproxEqAbs(3.0, slow[2], 1e-10);
}

test "lookahead: getStep accessor" {
    const allocator = testing.allocator;
    var params = [_]f64{1.0};

    var lookahead = try Lookahead(f64).init(allocator, &params, .{});
    defer lookahead.deinit();

    try testing.expectEqual(@as(u32, 0), lookahead.getStep());

    _ = try lookahead.step_update(&params);
    try testing.expectEqual(@as(u32, 1), lookahead.getStep());

    _ = try lookahead.step_update(&params);
    try testing.expectEqual(@as(u32, 2), lookahead.getStep());
}

test "lookahead: convergence with varying gradients" {
    const allocator = testing.allocator;
    var params = [_]f64{0.0};

    var lookahead = try Lookahead(f64).init(allocator, &params, .{ .sync_period = 5, .slow_step_size = 0.5 });
    defer lookahead.deinit();

    // Simulate noisy gradient descent (target = 10.0)
    const target = 10.0;
    const lr = 0.2;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (0..50) |_| {
        const x = params[0];
        // Gradient with noise: 2(x - target) + noise
        const grad = 2.0 * (x - target) + random.floatNorm(f64) * 0.5;

        params[0] = x - lr * grad;
        _ = try lookahead.step_update(&params);
    }

    // Lookahead should reduce variance and converge close to target
    try testing.expect(@abs(params[0] - target) < 2.0);
}

test "lookahead: memory safety with testing allocator" {
    const allocator = testing.allocator;
    const params = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    var lookahead = try Lookahead(f64).init(allocator, &params, .{});
    defer lookahead.deinit();

    var params_mut = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    for (0..20) |_| {
        for (&params_mut) |*p| p.* += 0.05;
        _ = try lookahead.step_update(&params_mut);
    }

    // testing.allocator will detect leaks automatically
}
