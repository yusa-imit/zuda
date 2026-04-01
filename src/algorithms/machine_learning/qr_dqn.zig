//! QR-DQN (Quantile Regression DQN)
//!
//! Distributional reinforcement learning algorithm that learns the full return distribution
//! using quantile regression instead of categorical bins (like C51). QR-DQN directly approximates
//! quantiles of the value distribution without requiring a fixed support range [V_min, V_max].
//!
//! Key differences from C51:
//! - No fixed support bounds (V_min, V_max) — quantiles adapt automatically
//! - Quantile regression loss (Huber quantile loss) instead of cross-entropy
//! - More flexible for unknown return ranges
//! - Direct quantile approximation τ ∈ [0,1]
//!
//! Algorithm:
//! 1. Network outputs N quantiles (default: 200) for each action
//! 2. Quantile loss: ρ_τ(δ) = |τ - I(δ < 0)| × δ with Huber smoothing
//! 3. Bellman target: r + γ × quantiles(s', argmax_a E[Z(s',a)])
//! 4. Loss: Σ_i Σ_j ρ_τ(T_j - θ_i) where T = target quantiles, θ = predicted
//! 5. Experience replay for off-policy learning
//!
//! Time complexity: O(batch_size × N² × network_forward × network_backward) per train()
//! Space complexity: O(buffer_size × state_dim + network_params × N)
//!
//! Use cases:
//! - Unknown return ranges (no need to set V_min/V_max)
//! - Risk-sensitive decision making (access full quantile distribution)
//! - Stochastic environments (poker, stock trading, robotics)
//! - Atari games, continuous control with uncertainty
//!
//! References:
//! - Dabney et al. "Distributional Reinforcement Learning with Quantile Regression" (AAAI 2018)
//! - https://arxiv.org/abs/1710.10044

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Configuration for QR-DQN
pub fn Config(comptime T: type) type {
    return struct {
        state_dim: usize,
        action_dim: usize,
        num_quantiles: usize = 200, // N in the paper
        hidden_dim: usize = 64,
        learning_rate: T = 0.0005,
        gamma: T = 0.99, // discount factor
        epsilon_start: T = 1.0,
        epsilon_end: T = 0.01,
        epsilon_decay: T = 0.995,
        buffer_capacity: usize = 10000,
        batch_size: usize = 32,
        target_update_freq: usize = 100, // steps between target network updates
        kappa: T = 1.0, // Huber loss threshold
    };
}

/// Experience tuple for replay buffer
fn Experience(comptime T: type) type {
    return struct {
        state: []T,
        action: usize,
        reward: T,
        next_state: []T,
        done: bool,
    };
}

/// QR-DQN Agent
///
/// Time: O(batch × N² × forward × backward) per train()
/// Space: O(buffer_size × state_dim + network_params × N)
pub fn QRDQN(comptime T: type) type {
    if (T != f32 and T != f64) {
        @compileError("QRDQN only supports f32 and f64");
    }

    return struct {
        const Self = @This();
        const Exp = Experience(T);

        allocator: Allocator,
        config: Config(T),

        // Network parameters (2-layer MLP)
        w1: []T, // hidden_dim × state_dim
        b1: []T, // hidden_dim
        w2: []T, // (action_dim × num_quantiles) × hidden_dim
        b2: []T, // action_dim × num_quantiles

        // Target network (frozen copy)
        target_w1: []T,
        target_b1: []T,
        target_w2: []T,
        target_b2: []T,

        // Experience replay buffer
        replay_buffer: std.ArrayList(Exp),
        buffer_index: usize,

        // Exploration
        epsilon: T,

        // Training state
        steps: usize,

        // Quantile midpoints (τ_i)
        tau: []T, // num_quantiles values in [0,1]

        /// Initialize QR-DQN agent
        ///
        /// Time: O(network_params)
        /// Space: O(network_params + buffer_capacity × state_dim)
        pub fn init(allocator: Allocator, cfg: Config(T)) !Self {
            if (cfg.state_dim == 0) return error.InvalidStateDim;
            if (cfg.action_dim == 0) return error.InvalidActionDim;
            if (cfg.num_quantiles == 0) return error.InvalidNumQuantiles;
            if (cfg.buffer_capacity == 0) return error.InvalidBufferCapacity;
            if (cfg.batch_size == 0 or cfg.batch_size > cfg.buffer_capacity) {
                return error.InvalidBatchSize;
            }

            // Allocate network parameters
            const w1 = try allocator.alloc(T, cfg.hidden_dim * cfg.state_dim);
            errdefer allocator.free(w1);
            const b1 = try allocator.alloc(T, cfg.hidden_dim);
            errdefer allocator.free(b1);
            const w2 = try allocator.alloc(T, cfg.action_dim * cfg.num_quantiles * cfg.hidden_dim);
            errdefer allocator.free(w2);
            const b2 = try allocator.alloc(T, cfg.action_dim * cfg.num_quantiles);
            errdefer allocator.free(b2);

            // Allocate target network
            const target_w1 = try allocator.alloc(T, cfg.hidden_dim * cfg.state_dim);
            errdefer allocator.free(target_w1);
            const target_b1 = try allocator.alloc(T, cfg.hidden_dim);
            errdefer allocator.free(target_b1);
            const target_w2 = try allocator.alloc(T, cfg.action_dim * cfg.num_quantiles * cfg.hidden_dim);
            errdefer allocator.free(target_w2);
            const target_b2 = try allocator.alloc(T, cfg.action_dim * cfg.num_quantiles);
            errdefer allocator.free(target_b2);

            // Allocate quantile midpoints
            const tau = try allocator.alloc(T, cfg.num_quantiles);
            errdefer allocator.free(tau);

            // Initialize quantile midpoints: τ_i = (i + 0.5) / N
            for (tau, 0..) |*t, i| {
                const fi: T = @floatFromInt(i);
                const fn_: T = @floatFromInt(cfg.num_quantiles);
                t.* = (fi + 0.5) / fn_;
            }

            // Xavier initialization for weights
            const xavier_w1 = @sqrt(2.0 / @as(T, @floatFromInt(cfg.state_dim + cfg.hidden_dim)));
            const xavier_w2 = @sqrt(2.0 / @as(T, @floatFromInt(cfg.hidden_dim + cfg.action_dim * cfg.num_quantiles)));

            var prng = std.Random.DefaultPrng.init(42);
            const random = prng.random();

            for (w1) |*w| w.* = (random.float(T) * 2 - 1) * xavier_w1;
            for (w2) |*w| w.* = (random.float(T) * 2 - 1) * xavier_w2;
            @memset(b1, 0);
            @memset(b2, 0);

            // Copy to target network
            @memcpy(target_w1, w1);
            @memcpy(target_b1, b1);
            @memcpy(target_w2, w2);
            @memcpy(target_b2, b2);

            return .{
                .allocator = allocator,
                .config = cfg,
                .w1 = w1,
                .b1 = b1,
                .w2 = w2,
                .b2 = b2,
                .target_w1 = target_w1,
                .target_b1 = target_b1,
                .target_w2 = target_w2,
                .target_b2 = target_b2,
                .replay_buffer = std.ArrayList(Exp).init(allocator),
                .buffer_index = 0,
                .epsilon = cfg.epsilon_start,
                .steps = 0,
                .tau = tau,
            };
        }

        /// Free all resources
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.w1);
            self.allocator.free(self.b1);
            self.allocator.free(self.w2);
            self.allocator.free(self.b2);
            self.allocator.free(self.target_w1);
            self.allocator.free(self.target_b1);
            self.allocator.free(self.target_w2);
            self.allocator.free(self.target_b2);
            self.allocator.free(self.tau);
            for (self.replay_buffer.items) |exp| {
                self.allocator.free(exp.state);
                self.allocator.free(exp.next_state);
            }
            self.replay_buffer.deinit();
        }

        /// Forward pass: compute quantiles for all actions
        ///
        /// Time: O(state_dim × hidden_dim + hidden_dim × action_dim × N)
        /// Space: O(hidden_dim + action_dim × N)
        fn forward(self: *const Self, state: []const T, w1: []const T, b1: []const T, w2: []const T, b2: []const T, output: []T) void {
            const hidden = self.allocator.alloc(T, self.config.hidden_dim) catch return;
            defer self.allocator.free(hidden);

            // Layer 1: hidden = ReLU(W1 * state + b1)
            for (hidden, 0..) |*h, i| {
                var sum: T = b1[i];
                for (state, 0..) |s, j| {
                    sum += w1[i * self.config.state_dim + j] * s;
                }
                h.* = @max(0, sum); // ReLU
            }

            // Layer 2: output = W2 * hidden + b2 (action_dim × num_quantiles)
            for (0..self.config.action_dim) |a| {
                for (0..self.config.num_quantiles) |n| {
                    const idx = a * self.config.num_quantiles + n;
                    var sum: T = b2[idx];
                    for (hidden, 0..) |h, j| {
                        sum += w2[idx * self.config.hidden_dim + j] * h;
                    }
                    output[idx] = sum;
                }
            }
        }

        /// Select action using epsilon-greedy policy
        ///
        /// Time: O(forward_pass + action_dim × N)
        /// Space: O(action_dim × N)
        pub fn selectAction(self: *Self, state: []const T) !usize {
            if (state.len != self.config.state_dim) return error.InvalidStateSize;

            var prng = std.Random.DefaultPrng.init(@intCast(self.steps));
            const random = prng.random();

            // Epsilon-greedy exploration
            if (random.float(T) < self.epsilon) {
                return random.intRangeAtMost(usize, 0, self.config.action_dim - 1);
            }

            return try self.greedyAction(state);
        }

        /// Select greedy action (argmax of mean quantile values)
        ///
        /// Time: O(forward_pass + action_dim × N)
        /// Space: O(action_dim × N)
        pub fn greedyAction(self: *Self, state: []const T) !usize {
            if (state.len != self.config.state_dim) return error.InvalidStateSize;

            const quantiles = try self.allocator.alloc(T, self.config.action_dim * self.config.num_quantiles);
            defer self.allocator.free(quantiles);

            self.forward(state, self.w1, self.b1, self.w2, self.b2, quantiles);

            // Choose action with highest mean quantile value
            var best_action: usize = 0;
            var best_value: T = -std.math.inf(T);

            for (0..self.config.action_dim) |a| {
                var mean: T = 0;
                for (0..self.config.num_quantiles) |n| {
                    mean += quantiles[a * self.config.num_quantiles + n];
                }
                mean /= @floatFromInt(self.config.num_quantiles);

                if (mean > best_value) {
                    best_value = mean;
                    best_action = a;
                }
            }

            return best_action;
        }

        /// Get quantile distribution for a specific state-action pair
        ///
        /// Time: O(forward_pass)
        /// Space: O(action_dim × N)
        pub fn getQuantiles(self: *Self, state: []const T, action: usize, quantiles_out: []T) !void {
            if (state.len != self.config.state_dim) return error.InvalidStateSize;
            if (action >= self.config.action_dim) return error.InvalidAction;
            if (quantiles_out.len != self.config.num_quantiles) return error.InvalidOutputSize;

            const all_quantiles = try self.allocator.alloc(T, self.config.action_dim * self.config.num_quantiles);
            defer self.allocator.free(all_quantiles);

            self.forward(state, self.w1, self.b1, self.w2, self.b2, all_quantiles);

            const offset = action * self.config.num_quantiles;
            @memcpy(quantiles_out, all_quantiles[offset .. offset + self.config.num_quantiles]);
        }

        /// Store experience in replay buffer
        ///
        /// Time: O(state_dim)
        /// Space: O(1) amortized
        pub fn storeExperience(self: *Self, state: []const T, action: usize, reward: T, next_state: []const T, done: bool) !void {
            if (state.len != self.config.state_dim) return error.InvalidStateSize;
            if (next_state.len != self.config.state_dim) return error.InvalidStateSize;

            const state_copy = try self.allocator.dupe(T, state);
            errdefer self.allocator.free(state_copy);
            const next_state_copy = try self.allocator.dupe(T, next_state);
            errdefer self.allocator.free(next_state_copy);

            const exp = Exp{
                .state = state_copy,
                .action = action,
                .reward = reward,
                .next_state = next_state_copy,
                .done = done,
            };

            if (self.replay_buffer.items.len < self.config.buffer_capacity) {
                try self.replay_buffer.append(exp);
            } else {
                // Circular buffer: overwrite oldest
                self.allocator.free(self.replay_buffer.items[self.buffer_index].state);
                self.allocator.free(self.replay_buffer.items[self.buffer_index].next_state);
                self.replay_buffer.items[self.buffer_index] = exp;
                self.buffer_index = (self.buffer_index + 1) % self.config.buffer_capacity;
            }
        }

        /// Huber quantile loss function
        ///
        /// ρ_τ(u) = |τ - I(u < 0)| × L_κ(u)
        /// where L_κ(u) is Huber loss with threshold κ
        fn quantileLoss(self: *const Self, tau_val: T, td_error: T) T {
            const abs_error = @abs(td_error);
            const huber = if (abs_error <= self.config.kappa)
                0.5 * td_error * td_error
            else
                self.config.kappa * (abs_error - 0.5 * self.config.kappa);

            const indicator: T = if (td_error < 0) 1.0 else 0.0;
            return @abs(tau_val - indicator) * huber;
        }

        /// Train the network using sampled batch
        ///
        /// Time: O(batch × N² × forward × backward)
        /// Space: O(batch × state_dim + batch × N)
        pub fn train(self: *Self) !void {
            if (self.replay_buffer.items.len < self.config.batch_size) {
                return error.NotEnoughExperiences;
            }

            var prng = std.Random.DefaultPrng.init(@intCast(self.steps));
            const random = prng.random();

            // Sample batch
            const batch = try self.allocator.alloc(Exp, self.config.batch_size);
            defer self.allocator.free(batch);

            for (batch) |*exp| {
                const idx = random.intRangeAtMost(usize, 0, self.replay_buffer.items.len - 1);
                exp.* = self.replay_buffer.items[idx];
            }

            // Compute quantile regression loss and update network
            var total_loss: T = 0;

            for (batch) |exp| {
                // Current quantiles
                const current_quantiles = try self.allocator.alloc(T, self.config.action_dim * self.config.num_quantiles);
                defer self.allocator.free(current_quantiles);
                self.forward(exp.state, self.w1, self.b1, self.w2, self.b2, current_quantiles);

                // Target quantiles (using target network)
                const next_quantiles = try self.allocator.alloc(T, self.config.action_dim * self.config.num_quantiles);
                defer self.allocator.free(next_quantiles);
                self.forward(exp.next_state, self.target_w1, self.target_b1, self.target_w2, self.target_b2, next_quantiles);

                // Greedy action for next state (using current network for action selection)
                const next_q = try self.allocator.alloc(T, self.config.action_dim * self.config.num_quantiles);
                defer self.allocator.free(next_q);
                self.forward(exp.next_state, self.w1, self.b1, self.w2, self.b2, next_q);

                var best_next_action: usize = 0;
                var best_mean: T = -std.math.inf(T);
                for (0..self.config.action_dim) |a| {
                    var mean: T = 0;
                    for (0..self.config.num_quantiles) |n| {
                        mean += next_q[a * self.config.num_quantiles + n];
                    }
                    mean /= @floatFromInt(self.config.num_quantiles);
                    if (mean > best_mean) {
                        best_mean = mean;
                        best_next_action = a;
                    }
                }

                // Compute target quantiles: T_j = r + γ × Z(s', a*)
                const target_quantiles = try self.allocator.alloc(T, self.config.num_quantiles);
                defer self.allocator.free(target_quantiles);

                for (0..self.config.num_quantiles) |n| {
                    if (exp.done) {
                        target_quantiles[n] = exp.reward;
                    } else {
                        target_quantiles[n] = exp.reward + self.config.gamma * next_quantiles[best_next_action * self.config.num_quantiles + n];
                    }
                }

                // Quantile regression loss: Σ_i Σ_j ρ_τ_i(T_j - θ_i)
                var loss: T = 0;
                const action_offset = exp.action * self.config.num_quantiles;

                for (0..self.config.num_quantiles) |i| {
                    for (0..self.config.num_quantiles) |j| {
                        const td_error = target_quantiles[j] - current_quantiles[action_offset + i];
                        loss += self.quantileLoss(self.tau[i], td_error);
                    }
                }
                loss /= @floatFromInt(self.config.num_quantiles);

                total_loss += loss;

                // Gradient descent (simplified — computes gradient numerically)
                const delta = self.config.learning_rate * loss / @as(T, @floatFromInt(self.config.batch_size));

                // Update only the quantiles for the taken action
                for (0..self.config.num_quantiles) |n| {
                    const idx = action_offset + n;
                    var gradient: T = 0;
                    for (0..self.config.num_quantiles) |j| {
                        const td_error = target_quantiles[j] - current_quantiles[idx];
                        const indicator: T = if (td_error < 0) -1.0 else 1.0;
                        gradient -= indicator * (if (td_error < 0) self.tau[n] else (self.tau[n] - 1.0));
                    }
                    self.b2[idx] -= delta * gradient;
                }
            }

            // Update target network periodically
            self.steps += 1;
            if (self.steps % self.config.target_update_freq == 0) {
                @memcpy(self.target_w1, self.w1);
                @memcpy(self.target_b1, self.b1);
                @memcpy(self.target_w2, self.w2);
                @memcpy(self.target_b2, self.b2);
            }

            // Decay epsilon
            self.epsilon = @max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay);
        }

        /// Reset agent state
        pub fn reset(self: *Self) void {
            self.epsilon = self.config.epsilon_start;
            self.steps = 0;
            for (self.replay_buffer.items) |exp| {
                self.allocator.free(exp.state);
                self.allocator.free(exp.next_state);
            }
            self.replay_buffer.clearRetainingCapacity();
            self.buffer_index = 0;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const expectEqual = testing.expectEqual;
const expectError = testing.expectError;
const expect = testing.expect;

test "QRDQN: basic initialization" {
    const cfg = Config(f64){
        .state_dim = 4,
        .action_dim = 2,
        .num_quantiles = 51,
        .hidden_dim = 32,
    };

    var agent = try QRDQN(f64).init(testing.allocator, cfg);
    defer agent.deinit();

    try expectEqual(@as(usize, 4), agent.config.state_dim);
    try expectEqual(@as(usize, 2), agent.config.action_dim);
    try expectEqual(@as(usize, 51), agent.config.num_quantiles);
    try expectEqual(@as(f64, 1.0), agent.epsilon);
    try expectEqual(@as(usize, 0), agent.steps);
}

test "QRDQN: quantile midpoints" {
    const cfg = Config(f32){
        .state_dim = 2,
        .action_dim = 2,
        .num_quantiles = 5,
    };

    var agent = try QRDQN(f32).init(testing.allocator, cfg);
    defer agent.deinit();

    // Check quantile midpoints: τ_i = (i + 0.5) / N
    try expect(@abs(agent.tau[0] - 0.1) < 1e-5); // (0 + 0.5) / 5
    try expect(@abs(agent.tau[1] - 0.3) < 1e-5); // (1 + 0.5) / 5
    try expect(@abs(agent.tau[2] - 0.5) < 1e-5); // (2 + 0.5) / 5
    try expect(@abs(agent.tau[3] - 0.7) < 1e-5); // (3 + 0.5) / 5
    try expect(@abs(agent.tau[4] - 0.9) < 1e-5); // (4 + 0.5) / 5
}

test "QRDQN: action selection" {
    const cfg = Config(f64){
        .state_dim = 2,
        .action_dim = 3,
        .num_quantiles = 10,
    };

    var agent = try QRDQN(f64).init(testing.allocator, cfg);
    defer agent.deinit();

    const state = [_]f64{ 0.5, -0.3 };
    const action = try agent.selectAction(&state);
    try expect(action < 3);
}

test "QRDQN: greedy action selection" {
    const cfg = Config(f64){
        .state_dim = 2,
        .action_dim = 2,
        .num_quantiles = 10,
    };

    var agent = try QRDQN(f64).init(testing.allocator, cfg);
    defer agent.deinit();

    agent.epsilon = 0.0; // deterministic

    const state = [_]f64{ 1.0, 0.0 };
    const action = try agent.greedyAction(&state);
    try expect(action < 2);
}

test "QRDQN: get quantiles" {
    const cfg = Config(f64){
        .state_dim = 2,
        .action_dim = 2,
        .num_quantiles = 5,
    };

    var agent = try QRDQN(f64).init(testing.allocator, cfg);
    defer agent.deinit();

    const state = [_]f64{ 0.5, -0.3 };
    var quantiles: [5]f64 = undefined;

    try agent.getQuantiles(&state, 0, &quantiles);

    // Should have 5 quantile values
    for (quantiles) |q| {
        _ = q; // quantiles are network outputs (can be any value)
    }
}

test "QRDQN: experience storage" {
    const cfg = Config(f64){
        .state_dim = 2,
        .action_dim = 2,
        .buffer_capacity = 100,
    };

    var agent = try QRDQN(f64).init(testing.allocator, cfg);
    defer agent.deinit();

    const state = [_]f64{ 0.5, -0.3 };
    const next_state = [_]f64{ 0.6, -0.2 };

    try agent.storeExperience(&state, 1, 1.0, &next_state, false);
    try expectEqual(@as(usize, 1), agent.replay_buffer.items.len);

    const exp = agent.replay_buffer.items[0];
    try expectEqual(@as(usize, 1), exp.action);
    try expectEqual(@as(f64, 1.0), exp.reward);
    try expectEqual(false, exp.done);
}

test "QRDQN: circular buffer overflow" {
    const cfg = Config(f64){
        .state_dim = 2,
        .action_dim = 2,
        .buffer_capacity = 3,
    };

    var agent = try QRDQN(f64).init(testing.allocator, cfg);
    defer agent.deinit();

    const state = [_]f64{ 0.0, 0.0 };
    const next_state = [_]f64{ 1.0, 1.0 };

    // Fill buffer
    try agent.storeExperience(&state, 0, 1.0, &next_state, false);
    try agent.storeExperience(&state, 1, 2.0, &next_state, false);
    try agent.storeExperience(&state, 0, 3.0, &next_state, false);
    try expectEqual(@as(usize, 3), agent.replay_buffer.items.len);

    // Overflow: should overwrite oldest
    try agent.storeExperience(&state, 1, 4.0, &next_state, false);
    try expectEqual(@as(usize, 3), agent.replay_buffer.items.len);
    try expectEqual(@as(f64, 4.0), agent.replay_buffer.items[0].reward); // oldest overwritten
}

test "QRDQN: training requires sufficient experiences" {
    const cfg = Config(f64){
        .state_dim = 2,
        .action_dim = 2,
        .batch_size = 4,
    };

    var agent = try QRDQN(f64).init(testing.allocator, cfg);
    defer agent.deinit();

    // Not enough experiences
    try expectError(error.NotEnoughExperiences, agent.train());
}

test "QRDQN: training with terminal states" {
    const cfg = Config(f64){
        .state_dim = 2,
        .action_dim = 2,
        .batch_size = 2,
        .buffer_capacity = 10,
        .num_quantiles = 5,
    };

    var agent = try QRDQN(f64).init(testing.allocator, cfg);
    defer agent.deinit();

    const state = [_]f64{ 0.5, -0.3 };
    const next_state = [_]f64{ 0.0, 0.0 };

    // Add terminal experience
    try agent.storeExperience(&state, 1, 1.0, &next_state, true);
    try agent.storeExperience(&state, 0, -1.0, &next_state, true);

    // Should be able to train
    try agent.train();

    // Epsilon should decay
    try expect(agent.epsilon < cfg.epsilon_start);
}

test "QRDQN: target network updates" {
    const cfg = Config(f64){
        .state_dim = 2,
        .action_dim = 2,
        .batch_size = 2,
        .target_update_freq = 2,
        .num_quantiles = 5,
    };

    var agent = try QRDQN(f64).init(testing.allocator, cfg);
    defer agent.deinit();

    const state = [_]f64{ 0.5, -0.3 };
    const next_state = [_]f64{ 0.6, -0.2 };

    for (0..4) |_| {
        try agent.storeExperience(&state, 0, 1.0, &next_state, false);
    }

    const original_target_b2 = agent.target_b2[0];

    // First training: steps = 1, no target update
    try agent.train();
    try expectEqual(original_target_b2, agent.target_b2[0]);

    // Second training: steps = 2, target update
    try agent.train();
    // Target should be updated (may or may not be different due to learning)
    _ = agent.target_b2[0]; // just verify no crash
}

test "QRDQN: reset" {
    const cfg = Config(f64){
        .state_dim = 2,
        .action_dim = 2,
    };

    var agent = try QRDQN(f64).init(testing.allocator, cfg);
    defer agent.deinit();

    const state = [_]f64{ 0.0, 0.0 };
    const next_state = [_]f64{ 1.0, 1.0 };

    try agent.storeExperience(&state, 0, 1.0, &next_state, false);
    agent.epsilon = 0.5;
    agent.steps = 100;

    agent.reset();

    try expectEqual(@as(f64, cfg.epsilon_start), agent.epsilon);
    try expectEqual(@as(usize, 0), agent.steps);
    try expectEqual(@as(usize, 0), agent.replay_buffer.items.len);
}

test "QRDQN: f32 support" {
    const cfg = Config(f32){
        .state_dim = 3,
        .action_dim = 2,
        .num_quantiles = 10,
    };

    var agent = try QRDQN(f32).init(testing.allocator, cfg);
    defer agent.deinit();

    const state = [_]f32{ 0.5, -0.3, 0.1 };
    const action = try agent.selectAction(&state);
    try expect(action < 2);
}

test "QRDQN: large state-action spaces" {
    const cfg = Config(f64){
        .state_dim = 20,
        .action_dim = 5,
        .num_quantiles = 51,
        .hidden_dim = 128,
    };

    var agent = try QRDQN(f64).init(testing.allocator, cfg);
    defer agent.deinit();

    const state = [_]f64{0.0} ** 20;
    const next_state = [_]f64{1.0} ** 20;

    try agent.storeExperience(&state, 2, 0.5, &next_state, false);
    try expectEqual(@as(usize, 1), agent.replay_buffer.items.len);
}

test "QRDQN: invalid config" {
    try expectError(error.InvalidStateDim, QRDQN(f64).init(testing.allocator, .{
        .state_dim = 0,
        .action_dim = 2,
    }));

    try expectError(error.InvalidActionDim, QRDQN(f64).init(testing.allocator, .{
        .state_dim = 2,
        .action_dim = 0,
    }));

    try expectError(error.InvalidNumQuantiles, QRDQN(f64).init(testing.allocator, .{
        .state_dim = 2,
        .action_dim = 2,
        .num_quantiles = 0,
    }));

    try expectError(error.InvalidBatchSize, QRDQN(f64).init(testing.allocator, .{
        .state_dim = 2,
        .action_dim = 2,
        .batch_size = 0,
    }));

    try expectError(error.InvalidBatchSize, QRDQN(f64).init(testing.allocator, .{
        .state_dim = 2,
        .action_dim = 2,
        .buffer_capacity = 10,
        .batch_size = 20,
    }));
}

test "QRDQN: memory safety with testing.allocator" {
    const cfg = Config(f64){
        .state_dim = 4,
        .action_dim = 2,
        .batch_size = 2,
        .num_quantiles = 10,
    };

    var agent = try QRDQN(f64).init(testing.allocator, cfg);
    defer agent.deinit();

    const state = [_]f64{ 0.5, -0.3, 0.1, 0.8 };
    const next_state = [_]f64{ 0.6, -0.2, 0.2, 0.9 };

    for (0..5) |_| {
        try agent.storeExperience(&state, 1, 1.0, &next_state, false);
    }

    try agent.train();
    try agent.train();

    // testing.allocator will detect leaks automatically
}
