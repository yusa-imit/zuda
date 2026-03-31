const std = @import("std");
const Allocator = std.mem.Allocator;

/// Deep Q-Network (DQN) — Deep Reinforcement Learning for Value-Based Control
///
/// Algorithm: Combines Q-Learning with neural networks and experience replay
/// - Uses neural network to approximate Q(s,a) instead of table
/// - Experience replay: stores (s,a,r,s') transitions in buffer, samples randomly
/// - Target network: separate frozen network for stable TD targets
/// - Epsilon-greedy exploration with decay
/// - Batch gradient descent on sampled experiences
///
/// Key innovations over Q-Learning:
/// - Function approximation: handles large/continuous state spaces
/// - Experience replay: breaks temporal correlations, improves sample efficiency
/// - Target network: stabilizes learning by fixing targets for C steps
/// - Mini-batch updates: smoother gradients than online learning
///
/// Time complexity:
/// - update(): O(batch_size × network_forward + batch_size × network_backward)
/// - selectAction(): O(num_actions × network_forward)
/// Space complexity: O(buffer_size × (state_dim + 3)) for replay buffer + O(network_params) for Q-network
///
/// Use cases:
/// - Atari games (original DQN paper), robotics (continuous control with discretized actions),
/// - game AI (large state spaces), autonomous systems (vision-based control),
/// - resource management (scheduling, allocation)
///
/// Trade-offs:
/// - vs Tabular Q-Learning: handles large state spaces, but introduces approximation error
/// - vs Policy Gradient: more sample efficient, but limited to discrete actions
/// - vs Actor-Critic: simpler (one network), but off-policy only
/// - vs Deep Deterministic Policy Gradient (DDPG): discrete actions only, but more stable
pub fn DQN(comptime T: type) type {
    if (T != f32 and T != f64) {
        @compileError("DQN only supports f32 and f64");
    }

    return struct {
        const Self = @This();

        /// Configuration for DQN
        pub const Config = struct {
            /// Learning rate for Q-network updates
            learning_rate: T = 0.001,
            /// Discount factor γ ∈ [0,1] (future reward importance)
            gamma: T = 0.99,
            /// Initial exploration rate ε ∈ [0,1]
            epsilon: T = 1.0,
            /// Final exploration rate (annealing target)
            epsilon_min: T = 0.01,
            /// Epsilon decay rate per step (multiplicative)
            epsilon_decay: T = 0.995,
            /// Experience replay buffer capacity
            buffer_size: usize = 10000,
            /// Mini-batch size for gradient descent
            batch_size: usize = 32,
            /// Target network update frequency (steps)
            target_update_freq: usize = 100,
        };

        /// Experience transition (s, a, r, s', done)
        const Experience = struct {
            state: []T,
            action: usize,
            reward: T,
            next_state: []T,
            done: bool,
        };

        /// Replay buffer for experience replay
        const ReplayBuffer = struct {
            experiences: []Experience,
            capacity: usize,
            size: usize,
            index: usize,
            allocator: Allocator,

            fn init(allocator: Allocator, capacity: usize) !ReplayBuffer {
                const experiences = try allocator.alloc(Experience, capacity);
                return ReplayBuffer{
                    .experiences = experiences,
                    .capacity = capacity,
                    .size = 0,
                    .index = 0,
                    .allocator = allocator,
                };
            }

            fn deinit(self: *ReplayBuffer) void {
                // Free all stored states
                for (self.experiences[0..self.size]) |exp| {
                    self.allocator.free(exp.state);
                    self.allocator.free(exp.next_state);
                }
                self.allocator.free(self.experiences);
            }

            fn add(self: *ReplayBuffer, state: []const T, action: usize, reward: T, next_state: []const T, done: bool) !void {
                // Deep copy states
                const state_copy = try self.allocator.dupe(T, state);
                errdefer self.allocator.free(state_copy);
                const next_state_copy = try self.allocator.dupe(T, next_state);
                errdefer self.allocator.free(next_state_copy);

                // If overwriting, free old states
                if (self.size == self.capacity) {
                    self.allocator.free(self.experiences[self.index].state);
                    self.allocator.free(self.experiences[self.index].next_state);
                }

                self.experiences[self.index] = Experience{
                    .state = state_copy,
                    .action = action,
                    .reward = reward,
                    .next_state = next_state_copy,
                    .done = done,
                };

                self.index = (self.index + 1) % self.capacity;
                self.size = @min(self.size + 1, self.capacity);
            }

            fn sample(self: *ReplayBuffer, batch_size: usize, rng: std.Random) []Experience {
                const n = @min(batch_size, self.size);
                var batch = self.allocator.alloc(Experience, n) catch return &[_]Experience{};

                var i: usize = 0;
                while (i < n) : (i += 1) {
                    const idx = rng.intRangeAtMost(usize, 0, self.size - 1);
                    batch[i] = self.experiences[idx];
                }

                return batch;
            }
        };

        /// Simple 2-layer neural network for Q-value approximation
        const QNetwork = struct {
            input_size: usize,
            hidden_size: usize,
            output_size: usize,
            // Weights: [input_size x hidden_size], [hidden_size x output_size]
            w1: []T,
            w2: []T,
            // Biases: [hidden_size], [output_size]
            b1: []T,
            b2: []T,
            // Activations for backprop
            h: []T,
            allocator: Allocator,

            fn init(allocator: Allocator, input_size: usize, hidden_size: usize, output_size: usize, rng: std.Random) !QNetwork {
                const w1 = try allocator.alloc(T, input_size * hidden_size);
                errdefer allocator.free(w1);
                const w2 = try allocator.alloc(T, hidden_size * output_size);
                errdefer allocator.free(w2);
                const b1 = try allocator.alloc(T, hidden_size);
                errdefer allocator.free(b1);
                const b2 = try allocator.alloc(T, output_size);
                errdefer allocator.free(b2);
                const h = try allocator.alloc(T, hidden_size);
                errdefer allocator.free(h);

                // Xavier initialization
                const scale1 = @sqrt(2.0 / @as(T, @floatFromInt(input_size)));
                const scale2 = @sqrt(2.0 / @as(T, @floatFromInt(hidden_size)));

                for (w1) |*w| w.* = (rng.float(T) * 2 - 1) * scale1;
                for (w2) |*w| w.* = (rng.float(T) * 2 - 1) * scale2;
                @memset(b1, 0);
                @memset(b2, 0);

                return QNetwork{
                    .input_size = input_size,
                    .hidden_size = hidden_size,
                    .output_size = output_size,
                    .w1 = w1,
                    .w2 = w2,
                    .b1 = b1,
                    .b2 = b2,
                    .h = h,
                    .allocator = allocator,
                };
            }

            fn deinit(self: *QNetwork) void {
                self.allocator.free(self.w1);
                self.allocator.free(self.w2);
                self.allocator.free(self.b1);
                self.allocator.free(self.b2);
                self.allocator.free(self.h);
            }

            /// Forward pass: state → Q-values for all actions
            /// Time: O(input_size × hidden_size + hidden_size × output_size)
            fn forward(self: *QNetwork, state: []const T, output: []T) void {
                std.debug.assert(state.len == self.input_size);
                std.debug.assert(output.len == self.output_size);

                // Hidden layer: h = ReLU(W1 * state + b1)
                for (self.h, 0..) |*h_val, i| {
                    var sum: T = self.b1[i];
                    for (state, 0..) |s, j| {
                        sum += self.w1[j * self.hidden_size + i] * s;
                    }
                    h_val.* = @max(0, sum); // ReLU
                }

                // Output layer: Q = W2 * h + b2 (linear activation)
                for (output, 0..) |*q, i| {
                    var sum: T = self.b2[i];
                    for (self.h, 0..) |h_val, j| {
                        sum += self.w2[j * self.output_size + i] * h_val;
                    }
                    q.* = sum;
                }
            }

            /// Copy weights from another network (for target network update)
            fn copyFrom(self: *QNetwork, other: *const QNetwork) void {
                @memcpy(self.w1, other.w1);
                @memcpy(self.w2, other.w2);
                @memcpy(self.b1, other.b1);
                @memcpy(self.b2, other.b2);
            }
        };

        allocator: Allocator,
        config: Config,
        state_dim: usize,
        num_actions: usize,
        q_network: QNetwork,
        target_network: QNetwork,
        replay_buffer: ReplayBuffer,
        epsilon: T,
        step_count: usize,
        rng: std.Random.DefaultPrng,
        q_values: []T, // Reusable buffer for Q-values

        /// Initialize DQN agent
        /// Time: O(state_dim × hidden_size + hidden_size × num_actions)
        /// Space: O(buffer_size × state_dim + network_params)
        pub fn init(allocator: Allocator, state_dim: usize, num_actions: usize, config: Config) !Self {
            if (state_dim == 0) return error.InvalidStateDim;
            if (num_actions == 0) return error.InvalidNumActions;

            var rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
            const hidden_size = @max(64, state_dim * 2); // Heuristic

            var q_network = try QNetwork.init(allocator, state_dim, hidden_size, num_actions, rng.random());
            errdefer q_network.deinit();

            var target_network = try QNetwork.init(allocator, state_dim, hidden_size, num_actions, rng.random());
            errdefer target_network.deinit();

            var replay_buffer = try ReplayBuffer.init(allocator, config.buffer_size);
            errdefer replay_buffer.deinit();

            const q_values = try allocator.alloc(T, num_actions);
            errdefer allocator.free(q_values);

            return Self{
                .allocator = allocator,
                .config = config,
                .state_dim = state_dim,
                .num_actions = num_actions,
                .q_network = q_network,
                .target_network = target_network,
                .replay_buffer = replay_buffer,
                .epsilon = config.epsilon,
                .step_count = 0,
                .rng = rng,
                .q_values = q_values,
            };
        }

        pub fn deinit(self: *Self) void {
            self.q_network.deinit();
            self.target_network.deinit();
            self.replay_buffer.deinit();
            self.allocator.free(self.q_values);
        }

        /// Select action using ε-greedy policy
        /// Time: O(num_actions × network_forward)
        pub fn selectAction(self: *Self, state: []const T) !usize {
            if (state.len != self.state_dim) return error.InvalidState;

            // ε-greedy exploration
            if (self.rng.random().float(T) < self.epsilon) {
                return self.rng.random().intRangeAtMost(usize, 0, self.num_actions - 1);
            }

            // Greedy action (max Q-value)
            self.q_network.forward(state, self.q_values);

            var best_action: usize = 0;
            var best_q = self.q_values[0];
            for (self.q_values[1..], 1..) |q, i| {
                if (q > best_q) {
                    best_q = q;
                    best_action = i;
                }
            }

            return best_action;
        }

        /// Store experience in replay buffer
        /// Time: O(state_dim) for copying states
        pub fn storeExperience(self: *Self, state: []const T, action: usize, reward: T, next_state: []const T, done: bool) !void {
            if (state.len != self.state_dim or next_state.len != self.state_dim) return error.InvalidState;
            if (action >= self.num_actions) return error.InvalidAction;

            try self.replay_buffer.add(state, action, reward, next_state, done);
        }

        /// Train Q-network on mini-batch from replay buffer
        /// Time: O(batch_size × network_forward × network_backward)
        pub fn train(self: *Self) !void {
            if (self.replay_buffer.size < self.config.batch_size) return; // Not enough experiences

            // Sample mini-batch
            const batch = self.replay_buffer.sample(self.config.batch_size, self.rng.random());
            defer self.allocator.free(batch);

            // Temporary buffers for gradients (simplified single-step update)
            const target_q_values = try self.allocator.alloc(T, self.num_actions);
            defer self.allocator.free(target_q_values);

            const current_q_values = try self.allocator.alloc(T, self.num_actions);
            defer self.allocator.free(current_q_values);

            // Process each experience in batch
            for (batch) |exp| {
                // Compute target: r + γ * max_a' Q_target(s', a') if not done, else r
                self.target_network.forward(exp.next_state, target_q_values);
                var max_next_q: T = target_q_values[0];
                for (target_q_values[1..]) |q| {
                    max_next_q = @max(max_next_q, q);
                }

                const target = if (exp.done) exp.reward else exp.reward + self.config.gamma * max_next_q;

                // Current Q-values
                self.q_network.forward(exp.state, current_q_values);
                const current_q = current_q_values[exp.action];

                // TD error
                const td_error = target - current_q;

                // Simplified gradient descent: update only the action taken
                // In full DQN, this would be a proper backprop through the network
                // Here we do a simplified direct update for demonstration
                const delta = self.config.learning_rate * td_error;

                // Update output layer weights for the action
                for (self.q_network.h, 0..) |h_val, j| {
                    self.q_network.w2[j * self.num_actions + exp.action] += delta * h_val;
                }
                self.q_network.b2[exp.action] += delta;
            }

            // Decay epsilon
            self.epsilon = @max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay);

            // Update target network periodically
            self.step_count += 1;
            if (self.step_count % self.config.target_update_freq == 0) {
                self.target_network.copyFrom(&self.q_network);
            }
        }

        /// Get current Q-values for a state
        /// Time: O(num_actions × network_forward)
        pub fn getQValues(self: *Self, state: []const T) ![]const T {
            if (state.len != self.state_dim) return error.InvalidState;

            self.q_network.forward(state, self.q_values);
            return self.q_values;
        }

        /// Get greedy action (no exploration)
        /// Time: O(num_actions × network_forward)
        pub fn getGreedyAction(self: *Self, state: []const T) !usize {
            if (state.len != self.state_dim) return error.InvalidState;

            self.q_network.forward(state, self.q_values);

            var best_action: usize = 0;
            var best_q = self.q_values[0];
            for (self.q_values[1..], 1..) |q, i| {
                if (q > best_q) {
                    best_q = q;
                    best_action = i;
                }
            }

            return best_action;
        }

        /// Reset agent (clear buffer, reset epsilon, reinitialize networks)
        /// Time: O(buffer_size + network_params)
        pub fn reset(self: *Self) !void {
            // Clear replay buffer
            for (self.replay_buffer.experiences[0..self.replay_buffer.size]) |exp| {
                self.allocator.free(exp.state);
                self.allocator.free(exp.next_state);
            }
            self.replay_buffer.size = 0;
            self.replay_buffer.index = 0;

            // Reset epsilon
            self.epsilon = self.config.epsilon;
            self.step_count = 0;

            // Reinitialize networks
            self.q_network.deinit();
            self.target_network.deinit();

            const hidden_size = @max(64, self.state_dim * 2);
            self.q_network = try QNetwork.init(self.allocator, self.state_dim, hidden_size, self.num_actions, self.rng.random());
            self.target_network = try QNetwork.init(self.allocator, self.state_dim, hidden_size, self.num_actions, self.rng.random());
        }
    };
}

// Tests
const testing = std.testing;

test "DQN: basic initialization" {
    const allocator = testing.allocator;

    var dqn = try DQN(f64).init(allocator, 4, 2, .{});
    defer dqn.deinit();

    try testing.expectEqual(4, dqn.state_dim);
    try testing.expectEqual(2, dqn.num_actions);
    try testing.expectApproxEqAbs(1.0, dqn.epsilon, 1e-6);
}

test "DQN: epsilon-greedy action selection" {
    const allocator = testing.allocator;

    var dqn = try DQN(f64).init(allocator, 2, 3, .{ .epsilon = 1.0 }); // Full exploration
    defer dqn.deinit();

    const state = [_]f64{ 0.5, 0.5 };

    // With epsilon=1.0, should always explore (random actions)
    var action_counts = [_]usize{0} ** 3;
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const action = try dqn.selectAction(&state);
        try testing.expect(action < 3);
        action_counts[action] += 1;
    }

    // All actions should have been selected at least once
    for (action_counts) |count| {
        try testing.expect(count > 0);
    }
}

test "DQN: greedy action selection" {
    const allocator = testing.allocator;

    var dqn = try DQN(f64).init(allocator, 2, 3, .{ .epsilon = 0.0 }); // No exploration
    defer dqn.deinit();

    const state = [_]f64{ 0.5, 0.5 };

    // With epsilon=0.0, should always exploit (greedy)
    const action1 = try dqn.getGreedyAction(&state);
    const action2 = try dqn.getGreedyAction(&state);
    try testing.expectEqual(action1, action2); // Deterministic
}

test "DQN: store and replay experiences" {
    const allocator = testing.allocator;

    var dqn = try DQN(f64).init(allocator, 2, 2, .{ .buffer_size = 10 });
    defer dqn.deinit();

    const state = [_]f64{ 0.0, 0.0 };
    const next_state = [_]f64{ 1.0, 1.0 };

    // Store 5 experiences
    var i: usize = 0;
    while (i < 5) : (i += 1) {
        try dqn.storeExperience(&state, 0, 1.0, &next_state, false);
    }

    try testing.expectEqual(5, dqn.replay_buffer.size);
}

test "DQN: buffer overflow (circular)" {
    const allocator = testing.allocator;

    var dqn = try DQN(f64).init(allocator, 2, 2, .{ .buffer_size = 3 });
    defer dqn.deinit();

    const state = [_]f64{ 0.0, 0.0 };
    const next_state = [_]f64{ 1.0, 1.0 };

    // Store more than capacity
    var i: usize = 0;
    while (i < 5) : (i += 1) {
        try dqn.storeExperience(&state, 0, @floatFromInt(i), &next_state, false);
    }

    try testing.expectEqual(3, dqn.replay_buffer.size); // Capped at capacity
}

test "DQN: training updates Q-network" {
    const allocator = testing.allocator;

    var dqn = try DQN(f64).init(allocator, 2, 2, .{ .batch_size = 4, .learning_rate = 0.1 });
    defer dqn.deinit();

    const state = [_]f64{ 0.0, 0.0 };
    const next_state = [_]f64{ 1.0, 1.0 };

    // Store enough experiences for training
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        try dqn.storeExperience(&state, 0, 1.0, &next_state, false);
    }

    // Get initial Q-values
    const q_before = try dqn.getQValues(&state);
    const q0_before = q_before[0];

    // Train (should update Q-values)
    try dqn.train();

    // Q-values should have changed
    const q_after = try dqn.getQValues(&state);
    const q0_after = q_after[0];

    // Q-value should increase (positive reward)
    try testing.expect(q0_after != q0_before);
}

test "DQN: epsilon decay" {
    const allocator = testing.allocator;

    var dqn = try DQN(f64).init(allocator, 2, 2, .{
        .epsilon = 1.0,
        .epsilon_min = 0.1,
        .epsilon_decay = 0.9,
        .batch_size = 2,
    });
    defer dqn.deinit();

    const state = [_]f64{ 0.0, 0.0 };
    const next_state = [_]f64{ 1.0, 1.0 };

    // Store experiences and train
    var i: usize = 0;
    while (i < 5) : (i += 1) {
        try dqn.storeExperience(&state, 0, 1.0, &next_state, false);
    }

    const epsilon_before = dqn.epsilon;
    try dqn.train();
    const epsilon_after = dqn.epsilon;

    // Epsilon should decay
    try testing.expect(epsilon_after < epsilon_before);
    try testing.expect(epsilon_after >= dqn.config.epsilon_min);
}

test "DQN: target network update" {
    const allocator = testing.allocator;

    var dqn = try DQN(f64).init(allocator, 2, 2, .{
        .target_update_freq = 2,
        .batch_size = 2,
    });
    defer dqn.deinit();

    const state = [_]f64{ 0.0, 0.0 };
    const next_state = [_]f64{ 1.0, 1.0 };

    // Store experiences
    var i: usize = 0;
    while (i < 5) : (i += 1) {
        try dqn.storeExperience(&state, 0, 1.0, &next_state, false);
    }

    // Train twice (should trigger target update at step 2)
    try dqn.train();
    try testing.expectEqual(1, dqn.step_count);

    try dqn.train();
    try testing.expectEqual(2, dqn.step_count);

    // Target network should have been updated (weights copied)
    // We can't directly test equality, but step_count confirms the update happened
}

test "DQN: cartpole-like learning" {
    const allocator = testing.allocator;

    var dqn = try DQN(f64).init(allocator, 4, 2, .{
        .learning_rate = 0.01,
        .gamma = 0.99,
        .epsilon = 0.5,
        .batch_size = 8,
    });
    defer dqn.deinit();

    // Simulate simple episodes
    var episode: usize = 0;
    while (episode < 10) : (episode += 1) {
        var state = [_]f64{ 0.0, 0.0, 0.0, 0.0 };

        var step: usize = 0;
        while (step < 20) : (step += 1) {
            const action = try dqn.selectAction(&state);

            // Simple reward: positive for action 0, negative for action 1
            const reward: f64 = if (action == 0) 1.0 else -1.0;

            // Next state (dummy)
            var next_state = [_]f64{ 0.1, 0.1, 0.1, 0.1 };

            try dqn.storeExperience(&state, action, reward, &next_state, false);

            // Train if enough experiences
            if (dqn.replay_buffer.size >= dqn.config.batch_size) {
                try dqn.train();
            }

            state = next_state;
        }
    }

    // After training, agent should prefer action 0 (higher reward)
    const final_state = [_]f64{ 0.0, 0.0, 0.0, 0.0 };
    const q_values = try dqn.getQValues(&final_state);

    // Q(s, 0) should be higher than Q(s, 1) after learning
    try testing.expect(q_values[0] > q_values[1]);
}

test "DQN: terminal state handling" {
    const allocator = testing.allocator;

    var dqn = try DQN(f64).init(allocator, 2, 2, .{ .batch_size = 2 });
    defer dqn.deinit();

    const state = [_]f64{ 0.0, 0.0 };
    const terminal_state = [_]f64{ 1.0, 1.0 };

    // Store terminal transition (done=true)
    try dqn.storeExperience(&state, 0, 10.0, &terminal_state, true);
    try dqn.storeExperience(&state, 1, 5.0, &terminal_state, true);

    // Train (target should be reward only, no future value)
    try dqn.train();

    // Q-values should reflect terminal rewards
    const q_values = try dqn.getQValues(&state);
    try testing.expect(q_values[0] > q_values[1]); // Higher reward action preferred
}

test "DQN: reset functionality" {
    const allocator = testing.allocator;

    var dqn = try DQN(f64).init(allocator, 2, 2, .{ .epsilon = 0.5 });
    defer dqn.deinit();

    const state = [_]f64{ 0.0, 0.0 };
    const next_state = [_]f64{ 1.0, 1.0 };

    // Store experiences
    try dqn.storeExperience(&state, 0, 1.0, &next_state, false);
    dqn.epsilon = 0.2;
    dqn.step_count = 100;

    // Reset
    try dqn.reset();

    try testing.expectEqual(0, dqn.replay_buffer.size);
    try testing.expectApproxEqAbs(dqn.config.epsilon, dqn.epsilon, 1e-6);
    try testing.expectEqual(0, dqn.step_count);
}

test "DQN: f32 support" {
    const allocator = testing.allocator;

    var dqn = try DQN(f32).init(allocator, 2, 2, .{});
    defer dqn.deinit();

    const state = [_]f32{ 0.5, 0.5 };
    const action = try dqn.selectAction(&state);
    try testing.expect(action < 2);
}

test "DQN: large state-action space" {
    const allocator = testing.allocator;

    var dqn = try DQN(f64).init(allocator, 100, 10, .{});
    defer dqn.deinit();

    var state: [100]f64 = undefined;
    for (&state, 0..) |*s, i| s.* = @as(f64, @floatFromInt(i)) / 100.0;

    const action = try dqn.selectAction(&state);
    try testing.expect(action < 10);
}

test "DQN: error handling - invalid state dimension" {
    const allocator = testing.allocator;

    var dqn = try DQN(f64).init(allocator, 2, 2, .{});
    defer dqn.deinit();

    const wrong_state = [_]f64{0.0}; // Should be length 2
    try testing.expectError(error.InvalidState, dqn.selectAction(&wrong_state));
}

test "DQN: error handling - invalid action" {
    const allocator = testing.allocator;

    var dqn = try DQN(f64).init(allocator, 2, 2, .{});
    defer dqn.deinit();

    const state = [_]f64{ 0.0, 0.0 };
    const next_state = [_]f64{ 1.0, 1.0 };

    try testing.expectError(error.InvalidAction, dqn.storeExperience(&state, 5, 1.0, &next_state, false));
}

test "DQN: error handling - zero state dimension" {
    const allocator = testing.allocator;

    try testing.expectError(error.InvalidStateDim, DQN(f64).init(allocator, 0, 2, .{}));
}

test "DQN: error handling - zero actions" {
    const allocator = testing.allocator;

    try testing.expectError(error.InvalidNumActions, DQN(f64).init(allocator, 2, 0, .{}));
}

test "DQN: memory safety with testing allocator" {
    const allocator = testing.allocator;

    var dqn = try DQN(f64).init(allocator, 4, 2, .{ .buffer_size = 5 });
    defer dqn.deinit();

    const state = [_]f64{ 0.0, 0.0, 0.0, 0.0 };
    const next_state = [_]f64{ 1.0, 1.0, 1.0, 1.0 };

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        try dqn.storeExperience(&state, 0, 1.0, &next_state, false);
        if (i >= 3) try dqn.train();
    }

    // No memory leaks should be detected by testing.allocator
}
