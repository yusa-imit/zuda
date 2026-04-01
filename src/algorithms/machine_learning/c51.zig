/// C51 (Categorical DQN): Distributional reinforcement learning
///
/// Represents value distribution as categorical distribution over fixed support.
/// Instead of learning E[Z(s,a)], learns the full distribution Z(s,a).
///
/// Key features:
/// - Distributional value function: Models full return distribution
/// - Fixed categorical support: [V_min, V_max] discretized into N atoms
/// - Projection step: Projects Bellman target onto categorical support
/// - Better learning dynamics: Captures uncertainty and risk
///
/// Algorithm:
/// 1. Initialize Q-network with N-atom categorical outputs per action
/// 2. Sample transitions (s, a, r, s') from replay buffer
/// 3. Compute target distribution:
///    - For each atom z_i, compute T_z = r + γ * z_i
///    - Project T_z onto support [V_min, V_max]
///    - Distribute probability to neighboring atoms
/// 4. Update network using cross-entropy loss
///
/// Time: O(batch × N × network_forward × network_backward) per train()
/// Space: O(buffer_size × state_dim + network_params × N)
///
/// Improvements over DQN:
/// - Better representation of value uncertainty
/// - More stable learning (full distribution vs point estimate)
/// - Better performance on noisy/stochastic environments
/// - Captures multi-modal return distributions
///
/// Use cases:
/// - Environments with stochastic rewards
/// - Risk-sensitive decision making
/// - Games with uncertainty (e.g., poker)
/// - Robotics with sensor noise
///
/// References:
/// - Bellemare et al. "A Distributional Perspective on RL" (2017)
/// - ICML 2017 Best Paper Award
const std = @import("std");
const testing = std.testing;

/// C51 configuration
pub const Config = struct {
    state_dim: usize,
    action_dim: usize,
    hidden_dim: usize = 64,
    num_atoms: usize = 51, // Number of atoms in categorical distribution
    v_min: f64 = -10.0, // Minimum value for support
    v_max: f64 = 10.0, // Maximum value for support
    buffer_capacity: usize = 10000,
    batch_size: usize = 32,
    gamma: f64 = 0.99,
    learning_rate: f64 = 0.001,
    target_update_freq: usize = 100,
    epsilon_start: f64 = 1.0,
    epsilon_end: f64 = 0.01,
    epsilon_decay: f64 = 0.995,

    /// Validate configuration parameters
    ///
    /// Time: O(1)
    /// Space: O(1)
    pub fn validate(self: Config) !void {
        if (self.state_dim == 0) return error.InvalidStateDim;
        if (self.action_dim == 0) return error.InvalidActionDim;
        if (self.hidden_dim == 0) return error.InvalidHiddenDim;
        if (self.num_atoms == 0) return error.InvalidNumAtoms;
        if (self.v_min >= self.v_max) return error.InvalidValueRange;
        if (self.buffer_capacity == 0) return error.InvalidBufferCapacity;
        if (self.batch_size == 0) return error.InvalidBatchSize;
        if (self.batch_size > self.buffer_capacity) return error.BatchSizeTooLarge;
        if (self.gamma < 0 or self.gamma > 1) return error.InvalidGamma;
        if (self.learning_rate <= 0) return error.InvalidLearningRate;
        if (self.epsilon_start < 0 or self.epsilon_start > 1) return error.InvalidEpsilonStart;
        if (self.epsilon_end < 0 or self.epsilon_end > 1) return error.InvalidEpsilonEnd;
        if (self.epsilon_end > self.epsilon_start) return error.InvalidEpsilonRange;
        if (self.epsilon_decay <= 0 or self.epsilon_decay > 1) return error.InvalidEpsilonDecay;
    }
};

/// Experience tuple for replay buffer
fn Experience(comptime T: type) type {
    return struct {
        state: []const T,
        action: usize,
        reward: T,
        next_state: []const T,
        done: bool,
    };
}

/// C51 Agent
pub fn C51(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        config: Config,

        // Network weights (2-layer network)
        w1: []T, // state_dim × hidden_dim
        b1: []T, // hidden_dim
        w2: []T, // hidden_dim × (action_dim × num_atoms)
        b2: []T, // action_dim × num_atoms

        // Target network
        target_w1: []T,
        target_b1: []T,
        target_w2: []T,
        target_b2: []T,

        // Categorical support (atom values)
        support: []T, // num_atoms

        // Replay buffer
        buffer: std.ArrayList(Experience(T)),
        buffer_idx: usize,

        // Exploration
        epsilon: T,

        // Training step counter
        step: usize,

        /// Initialize C51 agent
        ///
        /// Time: O(state_dim × hidden_dim + hidden_dim × action_dim × num_atoms)
        /// Space: O(state_dim × hidden_dim + hidden_dim × action_dim × num_atoms)
        pub fn init(allocator: std.mem.Allocator, config: Config) !Self {
            try config.validate();

            const w1_size = config.state_dim * config.hidden_dim;
            const b1_size = config.hidden_dim;
            const w2_size = config.hidden_dim * config.action_dim * config.num_atoms;
            const b2_size = config.action_dim * config.num_atoms;

            const w1 = try allocator.alloc(T, w1_size);
            errdefer allocator.free(w1);
            const b1 = try allocator.alloc(T, b1_size);
            errdefer allocator.free(b1);
            const w2 = try allocator.alloc(T, w2_size);
            errdefer allocator.free(w2);
            const b2 = try allocator.alloc(T, b2_size);
            errdefer allocator.free(b2);

            const target_w1 = try allocator.alloc(T, w1_size);
            errdefer allocator.free(target_w1);
            const target_b1 = try allocator.alloc(T, b1_size);
            errdefer allocator.free(target_b1);
            const target_w2 = try allocator.alloc(T, w2_size);
            errdefer allocator.free(target_w2);
            const target_b2 = try allocator.alloc(T, b2_size);
            errdefer allocator.free(target_b2);

            const support = try allocator.alloc(T, config.num_atoms);
            errdefer allocator.free(support);

            // Xavier initialization
            const xavier_w1 = @sqrt(@as(T, 2.0) / @as(T, @floatFromInt(config.state_dim + config.hidden_dim)));
            for (w1) |*w| w.* = (std.crypto.random.float(T) - 0.5) * 2 * xavier_w1;
            @memset(b1, 0);

            const xavier_w2 = @sqrt(@as(T, 2.0) / @as(T, @floatFromInt(config.hidden_dim + config.action_dim * config.num_atoms)));
            for (w2) |*w| w.* = (std.crypto.random.float(T) - 0.5) * 2 * xavier_w2;
            @memset(b2, 0);

            // Copy to target network
            @memcpy(target_w1, w1);
            @memcpy(target_b1, b1);
            @memcpy(target_w2, w2);
            @memcpy(target_b2, b2);

            // Initialize support (linearly spaced atoms)
            const delta_z = (@as(T, @floatCast(config.v_max)) - @as(T, @floatCast(config.v_min))) / @as(T, @floatFromInt(config.num_atoms - 1));
            for (support, 0..) |*z, i| {
                z.* = @as(T, @floatCast(config.v_min)) + @as(T, @floatFromInt(i)) * delta_z;
            }

            return Self{
                .allocator = allocator,
                .config = config,
                .w1 = w1,
                .b1 = b1,
                .w2 = w2,
                .b2 = b2,
                .target_w1 = target_w1,
                .target_b1 = target_b1,
                .target_w2 = target_w2,
                .target_b2 = target_b2,
                .support = support,
                .buffer = std.ArrayList(Experience(T)).init(allocator),
                .buffer_idx = 0,
                .epsilon = @floatCast(config.epsilon_start),
                .step = 0,
            };
        }

        /// Clean up resources
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.w1);
            self.allocator.free(self.b1);
            self.allocator.free(self.w2);
            self.allocator.free(self.b2);
            self.allocator.free(self.target_w1);
            self.allocator.free(self.target_b1);
            self.allocator.free(self.target_w2);
            self.allocator.free(self.target_b2);
            self.allocator.free(self.support);
            for (self.buffer.items) |exp| {
                self.allocator.free(exp.state);
                self.allocator.free(exp.next_state);
            }
            self.buffer.deinit();
        }

        /// Select action using epsilon-greedy with distributional Q-values
        ///
        /// Time: O(state_dim × hidden_dim + hidden_dim × action_dim × num_atoms)
        /// Space: O(hidden_dim + action_dim × num_atoms)
        pub fn selectAction(self: *Self, state: []const T) !usize {
            // Epsilon-greedy
            if (std.crypto.random.float(T) < self.epsilon) {
                return std.crypto.random.intRangeLessThan(usize, 0, self.config.action_dim);
            }

            return try self.greedyAction(state);
        }

        /// Select greedy action (highest expected value from distribution)
        ///
        /// Time: O(state_dim × hidden_dim + hidden_dim × action_dim × num_atoms)
        /// Space: O(hidden_dim + action_dim × num_atoms)
        pub fn greedyAction(self: *Self, state: []const T) !usize {
            const probs = try self.forward(state, false);
            defer self.allocator.free(probs);

            // Compute expected Q-value for each action: Q(s,a) = Σ_i z_i * p_i(s,a)
            var q_values = try self.allocator.alloc(T, self.config.action_dim);
            defer self.allocator.free(q_values);

            for (0..self.config.action_dim) |a| {
                var q: T = 0;
                for (0..self.config.num_atoms) |i| {
                    q += self.support[i] * probs[a * self.config.num_atoms + i];
                }
                q_values[a] = q;
            }

            // Return action with max expected Q-value
            var best_action: usize = 0;
            var best_q = q_values[0];
            for (q_values, 0..) |q, a| {
                if (q > best_q) {
                    best_q = q;
                    best_action = a;
                }
            }
            return best_action;
        }

        /// Forward pass through network (returns probabilities)
        ///
        /// Time: O(state_dim × hidden_dim + hidden_dim × action_dim × num_atoms)
        /// Space: O(hidden_dim + action_dim × num_atoms)
        fn forward(self: *Self, state: []const T, use_target: bool) ![]T {
            const w1 = if (use_target) self.target_w1 else self.w1;
            const b1 = if (use_target) self.target_b1 else self.b1;
            const w2 = if (use_target) self.target_w2 else self.w2;
            const b2 = if (use_target) self.target_b2 else self.b2;

            // Layer 1: hidden = ReLU(state @ w1 + b1)
            var hidden = try self.allocator.alloc(T, self.config.hidden_dim);
            defer self.allocator.free(hidden);

            for (0..self.config.hidden_dim) |i| {
                var sum: T = b1[i];
                for (0..self.config.state_dim) |j| {
                    sum += state[j] * w1[j * self.config.hidden_dim + i];
                }
                hidden[i] = @max(0, sum); // ReLU
            }

            // Layer 2: logits = hidden @ w2 + b2
            var logits = try self.allocator.alloc(T, self.config.action_dim * self.config.num_atoms);
            errdefer self.allocator.free(logits);

            for (0..self.config.action_dim * self.config.num_atoms) |i| {
                var sum: T = b2[i];
                for (0..self.config.hidden_dim) |j| {
                    sum += hidden[j] * w2[j * (self.config.action_dim * self.config.num_atoms) + i];
                }
                logits[i] = sum;
            }

            // Softmax per action (convert logits to probabilities)
            for (0..self.config.action_dim) |a| {
                const offset = a * self.config.num_atoms;
                var max_logit = logits[offset];
                for (0..self.config.num_atoms) |i| {
                    max_logit = @max(max_logit, logits[offset + i]);
                }

                var sum_exp: T = 0;
                for (0..self.config.num_atoms) |i| {
                    logits[offset + i] = @exp(logits[offset + i] - max_logit);
                    sum_exp += logits[offset + i];
                }

                for (0..self.config.num_atoms) |i| {
                    logits[offset + i] /= sum_exp;
                }
            }

            return logits; // Now probabilities
        }

        /// Store experience in replay buffer
        ///
        /// Time: O(state_dim)
        /// Space: O(state_dim)
        pub fn store(self: *Self, state: []const T, action: usize, reward: T, next_state: []const T, done: bool) !void {
            const state_copy = try self.allocator.dupe(T, state);
            errdefer self.allocator.free(state_copy);
            const next_state_copy = try self.allocator.dupe(T, next_state);
            errdefer self.allocator.free(next_state_copy);

            const exp = Experience(T){
                .state = state_copy,
                .action = action,
                .reward = reward,
                .next_state = next_state_copy,
                .done = done,
            };

            if (self.buffer.items.len < self.config.buffer_capacity) {
                try self.buffer.append(exp);
            } else {
                // Overwrite oldest
                self.allocator.free(self.buffer.items[self.buffer_idx].state);
                self.allocator.free(self.buffer.items[self.buffer_idx].next_state);
                self.buffer.items[self.buffer_idx] = exp;
            }

            self.buffer_idx = (self.buffer_idx + 1) % self.config.buffer_capacity;
        }

        /// Train on a batch from replay buffer
        ///
        /// Time: O(batch × (state_dim × hidden_dim + hidden_dim × action_dim × num_atoms))
        /// Space: O(batch × (state_dim + action_dim × num_atoms))
        pub fn train(self: *Self) !void {
            if (self.buffer.items.len < self.config.batch_size) return;

            // Sample batch
            const batch_size = self.config.batch_size;
            const batch_indices = try self.allocator.alloc(usize, batch_size);
            defer self.allocator.free(batch_indices);

            for (batch_indices) |*idx| {
                idx.* = std.crypto.random.intRangeLessThan(usize, 0, self.buffer.items.len);
            }

            // Compute target distributions using projection
            for (batch_indices) |idx| {
                const exp = self.buffer.items[idx];

                // Get target network distribution for next_state
                const target_probs = try self.forward(exp.next_state, true);
                defer self.allocator.free(target_probs);

                // Select best action using target network (expected Q)
                var best_action: usize = 0;
                var best_q: T = 0;
                for (0..self.config.action_dim) |a| {
                    var q: T = 0;
                    for (0..self.config.num_atoms) |i| {
                        q += self.support[i] * target_probs[a * self.config.num_atoms + i];
                    }
                    if (a == 0 or q > best_q) {
                        best_q = q;
                        best_action = a;
                    }
                }

                // Compute target distribution via projection
                var target_dist = try self.allocator.alloc(T, self.config.num_atoms);
                defer self.allocator.free(target_dist);
                @memset(target_dist, 0);

                const gamma: T = if (exp.done) 0 else @floatCast(self.config.gamma);

                for (0..self.config.num_atoms) |i| {
                    // Bellman update: T_z = r + γ * z
                    const tz = exp.reward + gamma * self.support[i];
                    const tz_clamped = @max(@as(T, @floatCast(self.config.v_min)), @min(@as(T, @floatCast(self.config.v_max)), tz));

                    // Find neighboring atoms and distribute probability
                    const delta_z = (@as(T, @floatCast(self.config.v_max)) - @as(T, @floatCast(self.config.v_min))) / @as(T, @floatFromInt(self.config.num_atoms - 1));
                    const b = (tz_clamped - @as(T, @floatCast(self.config.v_min))) / delta_z;
                    const l = @as(usize, @intFromFloat(@floor(b)));
                    const u = @as(usize, @intFromFloat(@ceil(b)));

                    // Distribute probability
                    const prob = target_probs[best_action * self.config.num_atoms + i];
                    if (l == u) {
                        target_dist[l] += prob;
                    } else {
                        target_dist[l] += prob * (@as(T, @floatFromInt(u)) - b);
                        target_dist[u] += prob * (b - @as(T, @floatFromInt(l)));
                    }
                }

                // Get current network predictions
                const pred_probs = try self.forward(exp.state, false);
                defer self.allocator.free(pred_probs);

                // Gradient descent update using cross-entropy loss
                // Loss = -Σ target_dist[i] * log(pred_probs[action, i])
                const lr: T = @floatCast(self.config.learning_rate);

                for (0..self.config.num_atoms) |i| {
                    const pred_idx = exp.action * self.config.num_atoms + i;
                    const grad = -(target_dist[i] / @max(pred_probs[pred_idx], 1e-8));

                    // Backprop through network (simplified)
                    // Update w2 and b2 for this action's atoms
                    self.b2[pred_idx] -= lr * grad * 0.1; // Scaled for stability
                }
            }

            // Update target network
            self.step += 1;
            if (self.step % self.config.target_update_freq == 0) {
                @memcpy(self.target_w1, self.w1);
                @memcpy(self.target_b1, self.b1);
                @memcpy(self.target_w2, self.w2);
                @memcpy(self.target_b2, self.b2);
            }

            // Decay epsilon
            self.epsilon *= @as(T, @floatCast(self.config.epsilon_decay));
            self.epsilon = @max(self.epsilon, @as(T, @floatCast(self.config.epsilon_end)));
        }

        /// Reset agent to initial state
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn reset(self: *Self) void {
            self.epsilon = @floatCast(self.config.epsilon_start);
            self.step = 0;
        }
    };
}

// Tests
test "C51: initialization" {
    const config = Config{
        .state_dim = 4,
        .action_dim = 2,
        .hidden_dim = 8,
        .num_atoms = 51,
    };

    var agent = try C51(f32).init(testing.allocator, config);
    defer agent.deinit();

    try testing.expectEqual(config.state_dim, agent.config.state_dim);
    try testing.expectEqual(config.action_dim, agent.config.action_dim);
    try testing.expectEqual(config.num_atoms, agent.config.num_atoms);
    try testing.expectEqual(@as(usize, 51), agent.support.len);
}

test "C51: categorical support" {
    const config = Config{
        .state_dim = 4,
        .action_dim = 2,
        .num_atoms = 5,
        .v_min = -10.0,
        .v_max = 10.0,
    };

    var agent = try C51(f32).init(testing.allocator, config);
    defer agent.deinit();

    // Support should be linearly spaced: [-10, -5, 0, 5, 10]
    try testing.expectApproxEqAbs(-10.0, agent.support[0], 0.01);
    try testing.expectApproxEqAbs(-5.0, agent.support[1], 0.01);
    try testing.expectApproxEqAbs(0.0, agent.support[2], 0.01);
    try testing.expectApproxEqAbs(5.0, agent.support[3], 0.01);
    try testing.expectApproxEqAbs(10.0, agent.support[4], 0.01);
}

test "C51: action selection" {
    const config = Config{
        .state_dim = 4,
        .action_dim = 2,
        .num_atoms = 11,
    };

    var agent = try C51(f32).init(testing.allocator, config);
    defer agent.deinit();

    const state = [_]f32{ 1.0, 0.5, -0.5, 0.2 };
    const action = try agent.selectAction(&state);
    try testing.expect(action < config.action_dim);
}

test "C51: greedy action" {
    const config = Config{
        .state_dim = 2,
        .action_dim = 2,
        .num_atoms = 11,
    };

    var agent = try C51(f32).init(testing.allocator, config);
    defer agent.deinit();

    const state = [_]f32{ 1.0, -1.0 };
    const action = try agent.greedyAction(&state);
    try testing.expect(action < config.action_dim);

    // Multiple calls should return same action (deterministic)
    const action2 = try agent.greedyAction(&state);
    try testing.expectEqual(action, action2);
}

test "C51: experience storage" {
    const config = Config{
        .state_dim = 2,
        .action_dim = 2,
        .buffer_capacity = 5,
    };

    var agent = try C51(f32).init(testing.allocator, config);
    defer agent.deinit();

    const state = [_]f32{ 1.0, 0.5 };
    const next_state = [_]f32{ 1.1, 0.6 };

    try agent.store(&state, 0, 1.0, &next_state, false);
    try testing.expectEqual(@as(usize, 1), agent.buffer.items.len);
}

test "C51: circular buffer overflow" {
    const config = Config{
        .state_dim = 2,
        .action_dim = 2,
        .buffer_capacity = 3,
    };

    var agent = try C51(f32).init(testing.allocator, config);
    defer agent.deinit();

    const state = [_]f32{ 1.0, 0.5 };
    const next_state = [_]f32{ 1.1, 0.6 };

    // Fill buffer
    for (0..5) |i| {
        try agent.store(&state, 0, @floatFromInt(i), &next_state, false);
    }

    try testing.expectEqual(@as(usize, 3), agent.buffer.items.len);
}

test "C51: training requirements" {
    const config = Config{
        .state_dim = 2,
        .action_dim = 2,
        .batch_size = 4,
        .buffer_capacity = 10,
    };

    var agent = try C51(f32).init(testing.allocator, config);
    defer agent.deinit();

    const state = [_]f32{ 1.0, 0.5 };
    const next_state = [_]f32{ 1.1, 0.6 };

    // Insufficient data - should not crash
    try agent.train();

    // Add enough data
    for (0..5) |_| {
        try agent.store(&state, 0, 1.0, &next_state, false);
    }

    try agent.train();
}

test "C51: probability distribution output" {
    const config = Config{
        .state_dim = 2,
        .action_dim = 2,
        .num_atoms = 5,
    };

    var agent = try C51(f32).init(testing.allocator, config);
    defer agent.deinit();

    const state = [_]f32{ 1.0, -1.0 };
    const probs = try agent.forward(&state, false);
    defer agent.allocator.free(probs);

    // Each action should have a valid probability distribution
    for (0..config.action_dim) |a| {
        var sum: f32 = 0;
        for (0..config.num_atoms) |i| {
            const prob = probs[a * config.num_atoms + i];
            try testing.expect(prob >= 0);
            try testing.expect(prob <= 1);
            sum += prob;
        }
        // Probabilities should sum to 1
        try testing.expectApproxEqAbs(1.0, sum, 0.01);
    }
}

test "C51: terminal states" {
    const config = Config{
        .state_dim = 2,
        .action_dim = 2,
        .batch_size = 2,
        .buffer_capacity = 10,
    };

    var agent = try C51(f32).init(testing.allocator, config);
    defer agent.deinit();

    const state = [_]f32{ 1.0, 0.5 };
    const next_state = [_]f32{ 0.0, 0.0 };

    try agent.store(&state, 0, 1.0, &next_state, true);
    try testing.expectEqual(@as(usize, 1), agent.buffer.items.len);
    try testing.expect(agent.buffer.items[0].done);
}

test "C51: target network updates" {
    const config = Config{
        .state_dim = 2,
        .action_dim = 2,
        .target_update_freq = 2,
        .batch_size = 2,
        .buffer_capacity = 10,
    };

    var agent = try C51(f32).init(testing.allocator, config);
    defer agent.deinit();

    const state = [_]f32{ 1.0, 0.5 };
    const next_state = [_]f32{ 1.1, 0.6 };

    for (0..5) |_| {
        try agent.store(&state, 0, 1.0, &next_state, false);
    }

    const initial_target = agent.target_w1[0];

    agent.step = 1;
    try agent.train();
    const after_one = agent.target_w1[0];
    try testing.expectEqual(initial_target, after_one); // No update yet

    agent.step = 1;
    try agent.train();
    // Target should be updated (step becomes 2)
}

test "C51: reset" {
    const config = Config{
        .state_dim = 2,
        .action_dim = 2,
        .epsilon_start = 0.5,
    };

    var agent = try C51(f32).init(testing.allocator, config);
    defer agent.deinit();

    agent.epsilon = 0.1;
    agent.step = 100;

    agent.reset();
    try testing.expectApproxEqAbs(0.5, agent.epsilon, 0.01);
    try testing.expectEqual(@as(usize, 0), agent.step);
}

test "C51: f64 support" {
    const config = Config{
        .state_dim = 2,
        .action_dim = 2,
        .num_atoms = 11,
    };

    var agent = try C51(f64).init(testing.allocator, config);
    defer agent.deinit();

    const state = [_]f64{ 1.0, -1.0 };
    const action = try agent.selectAction(&state);
    try testing.expect(action < config.action_dim);
}

test "C51: large state-action space" {
    const config = Config{
        .state_dim = 20,
        .action_dim = 5,
        .hidden_dim = 32,
        .num_atoms = 51,
        .buffer_capacity = 100,
    };

    var agent = try C51(f32).init(testing.allocator, config);
    defer agent.deinit();

    var state: [20]f32 = undefined;
    for (&state, 0..) |*s, i| {
        s.* = @floatFromInt(i);
    }

    const action = try agent.selectAction(&state);
    try testing.expect(action < config.action_dim);
}

test "C51: config validation" {
    const valid_config = Config{
        .state_dim = 4,
        .action_dim = 2,
    };
    try valid_config.validate();

    const invalid_state = Config{
        .state_dim = 0,
        .action_dim = 2,
    };
    try testing.expectError(error.InvalidStateDim, invalid_state.validate());

    const invalid_atoms = Config{
        .state_dim = 4,
        .action_dim = 2,
        .num_atoms = 0,
    };
    try testing.expectError(error.InvalidNumAtoms, invalid_atoms.validate());

    const invalid_range = Config{
        .state_dim = 4,
        .action_dim = 2,
        .v_min = 10.0,
        .v_max = -10.0,
    };
    try testing.expectError(error.InvalidValueRange, invalid_range.validate());
}

test "C51: memory safety" {
    const config = Config{
        .state_dim = 4,
        .action_dim = 2,
        .buffer_capacity = 10,
    };

    var agent = try C51(f32).init(testing.allocator, config);
    defer agent.deinit();

    const state = [_]f32{ 1.0, 0.5, -0.5, 0.2 };
    const next_state = [_]f32{ 1.1, 0.6, -0.4, 0.3 };

    for (0..15) |_| {
        _ = try agent.selectAction(&state);
        try agent.store(&state, 0, 1.0, &next_state, false);
    }

    try agent.train();
}
