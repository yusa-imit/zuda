const std = @import("std");
const Allocator = std.mem.Allocator;

/// Deep Deterministic Policy Gradient (DDPG) — Deep RL for Continuous Control
///
/// Algorithm: Off-policy actor-critic combining DPG with experience replay and target networks
/// - Actor network μ(s): deterministic policy mapping states to continuous actions
/// - Critic network Q(s,a): action-value function for evaluating state-action pairs
/// - Target networks: slowly updated copies for stable learning
/// - Experience replay: break temporal correlations, improve sample efficiency
/// - Ornstein-Uhlenbeck noise: temporally correlated exploration for continuous actions
///
/// Key innovations over discrete methods (DQN, Actor-Critic):
/// - Deterministic policy gradient: no action sampling, direct gradient computation
/// - Continuous action space: outputs real-valued actions (not discrete choices)
/// - Soft target updates: τ-based exponential moving average (smoother than hard copy)
/// - Correlated noise: OU process for physics-based exploration
///
/// Time complexity:
/// - train(): O(batch_size × (actor_forward + critic_forward + backprop))
/// - selectAction(): O(actor_forward) = O(state_dim × hidden_size + hidden_size × action_dim)
/// Space complexity: O(buffer_size × (state_dim + action_dim + 3)) + O(actor_params + critic_params)
///
/// Use cases:
/// - Robotics (continuous joint control, manipulation, locomotion)
/// - Autonomous vehicles (steering, throttle, braking)
/// - Industrial control (manufacturing, energy systems)
/// - Financial trading (continuous position sizing)
/// - Resource allocation (continuous resource levels)
///
/// Trade-offs:
/// - vs DQN: handles continuous actions, but requires two networks (more complex)
/// - vs PPO: more sample efficient (off-policy), but less stable (needs careful tuning)
/// - vs SAC: simpler (no entropy term), but less robust to hyperparameters
/// - vs TD3: original formulation (TD3 adds improvements like double Q-learning)
pub fn DDPG(comptime T: type) type {
    if (T != f32 and T != f64) {
        @compileError("DDPG only supports f32 and f64");
    }

    return struct {
        const Self = @This();

        /// Configuration for DDPG
        pub const Config = struct {
            /// Learning rate for actor network updates
            actor_lr: T = 0.0001,
            /// Learning rate for critic network updates
            critic_lr: T = 0.001,
            /// Discount factor γ ∈ [0,1] (future reward importance)
            gamma: T = 0.99,
            /// Soft update coefficient τ ∈ (0,1) for target networks
            tau: T = 0.001,
            /// Experience replay buffer capacity
            buffer_size: usize = 100000,
            /// Mini-batch size for gradient descent
            batch_size: usize = 64,
            /// OU noise standard deviation (exploration)
            noise_stddev: T = 0.2,
            /// OU noise theta (mean reversion rate)
            noise_theta: T = 0.15,
        };

        /// Experience transition (s, a, r, s', done)
        const Experience = struct {
            state: []T,
            action: []T,
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
                // Free all stored states and actions
                for (self.experiences[0..self.size]) |exp| {
                    self.allocator.free(exp.state);
                    self.allocator.free(exp.action);
                    self.allocator.free(exp.next_state);
                }
                self.allocator.free(self.experiences);
            }

            fn add(self: *ReplayBuffer, state: []const T, action: []const T, reward: T, next_state: []const T, done: bool) !void {
                // Deep copy states and actions
                const state_copy = try self.allocator.dupe(T, state);
                errdefer self.allocator.free(state_copy);
                const action_copy = try self.allocator.dupe(T, action);
                errdefer self.allocator.free(action_copy);
                const next_state_copy = try self.allocator.dupe(T, next_state);
                errdefer self.allocator.free(next_state_copy);

                // If overwriting, free old data
                if (self.size == self.capacity) {
                    self.allocator.free(self.experiences[self.index].state);
                    self.allocator.free(self.experiences[self.index].action);
                    self.allocator.free(self.experiences[self.index].next_state);
                }

                self.experiences[self.index] = Experience{
                    .state = state_copy,
                    .action = action_copy,
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

        /// Actor network: μ(s) → a (deterministic policy)
        const ActorNetwork = struct {
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

            fn init(allocator: Allocator, input_size: usize, hidden_size: usize, output_size: usize, rng: std.Random) !ActorNetwork {
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

                return ActorNetwork{
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

            fn deinit(self: *ActorNetwork) void {
                self.allocator.free(self.w1);
                self.allocator.free(self.w2);
                self.allocator.free(self.b1);
                self.allocator.free(self.b2);
                self.allocator.free(self.h);
            }

            /// Forward pass: state → action (tanh activation for bounded actions)
            /// Time: O(input_size × hidden_size + hidden_size × output_size)
            fn forward(self: *ActorNetwork, state: []const T, action: []T) void {
                std.debug.assert(state.len == self.input_size);
                std.debug.assert(action.len == self.output_size);

                // Hidden layer: h = ReLU(W1 * state + b1)
                for (self.h, 0..) |*h_val, i| {
                    var sum: T = self.b1[i];
                    for (state, 0..) |s, j| {
                        sum += self.w1[j * self.hidden_size + i] * s;
                    }
                    h_val.* = @max(0, sum); // ReLU
                }

                // Output layer: a = tanh(W2 * h + b2) (bounded to [-1, 1])
                for (action, 0..) |*a, i| {
                    var sum: T = self.b2[i];
                    for (self.h, 0..) |h_val, j| {
                        sum += self.w2[j * self.output_size + i] * h_val;
                    }
                    a.* = std.math.tanh(sum);
                }
            }

            /// Soft update: θ' ← τθ + (1-τ)θ'
            fn softUpdate(self: *ActorNetwork, other: *const ActorNetwork, tau: T) void {
                for (self.w1, other.w1) |*target, source| {
                    target.* = tau * source + (1 - tau) * target.*;
                }
                for (self.w2, other.w2) |*target, source| {
                    target.* = tau * source + (1 - tau) * target.*;
                }
                for (self.b1, other.b1) |*target, source| {
                    target.* = tau * source + (1 - tau) * target.*;
                }
                for (self.b2, other.b2) |*target, source| {
                    target.* = tau * source + (1 - tau) * target.*;
                }
            }
        };

        /// Critic network: Q(s, a) → scalar value
        const CriticNetwork = struct {
            state_size: usize,
            action_size: usize,
            hidden_size: usize,
            // Weights: [(state_size+action_size) x hidden_size], [hidden_size x 1]
            w1: []T,
            w2: []T,
            // Biases: [hidden_size], [1]
            b1: []T,
            b2: T,
            // Activations for backprop
            h: []T,
            allocator: Allocator,

            fn init(allocator: Allocator, state_size: usize, action_size: usize, hidden_size: usize, rng: std.Random) !CriticNetwork {
                const input_size = state_size + action_size;
                const w1 = try allocator.alloc(T, input_size * hidden_size);
                errdefer allocator.free(w1);
                const w2 = try allocator.alloc(T, hidden_size);
                errdefer allocator.free(w2);
                const b1 = try allocator.alloc(T, hidden_size);
                errdefer allocator.free(b1);
                const h = try allocator.alloc(T, hidden_size);
                errdefer allocator.free(h);

                // Xavier initialization
                const scale1 = @sqrt(2.0 / @as(T, @floatFromInt(input_size)));
                const scale2 = @sqrt(2.0 / @as(T, @floatFromInt(hidden_size)));

                for (w1) |*w| w.* = (rng.float(T) * 2 - 1) * scale1;
                for (w2) |*w| w.* = (rng.float(T) * 2 - 1) * scale2;
                @memset(b1, 0);

                return CriticNetwork{
                    .state_size = state_size,
                    .action_size = action_size,
                    .hidden_size = hidden_size,
                    .w1 = w1,
                    .w2 = w2,
                    .b1 = b1,
                    .b2 = 0,
                    .h = h,
                    .allocator = allocator,
                };
            }

            fn deinit(self: *CriticNetwork) void {
                self.allocator.free(self.w1);
                self.allocator.free(self.w2);
                self.allocator.free(self.b1);
                self.allocator.free(self.h);
            }

            /// Forward pass: (state, action) → Q-value
            /// Time: O((state_size + action_size) × hidden_size + hidden_size)
            fn forward(self: *CriticNetwork, state: []const T, action: []const T) T {
                std.debug.assert(state.len == self.state_size);
                std.debug.assert(action.len == self.action_size);

                // Hidden layer: h = ReLU(W1 * [state; action] + b1)
                for (self.h, 0..) |*h_val, i| {
                    var sum: T = self.b1[i];
                    // State part
                    for (state, 0..) |s, j| {
                        sum += self.w1[j * self.hidden_size + i] * s;
                    }
                    // Action part
                    for (action, 0..) |a, j| {
                        sum += self.w1[(self.state_size + j) * self.hidden_size + i] * a;
                    }
                    h_val.* = @max(0, sum); // ReLU
                }

                // Output layer: Q = W2 * h + b2 (scalar output)
                var q: T = self.b2;
                for (self.h, 0..) |h_val, j| {
                    q += self.w2[j] * h_val;
                }
                return q;
            }

            /// Soft update: θ' ← τθ + (1-τ)θ'
            fn softUpdate(self: *CriticNetwork, other: *const CriticNetwork, tau: T) void {
                for (self.w1, other.w1) |*target, source| {
                    target.* = tau * source + (1 - tau) * target.*;
                }
                for (self.w2, other.w2) |*target, source| {
                    target.* = tau * source + (1 - tau) * target.*;
                }
                for (self.b1, other.b1) |*target, source| {
                    target.* = tau * source + (1 - tau) * target.*;
                }
                self.b2 = tau * other.b2 + (1 - tau) * self.b2;
            }
        };

        /// Ornstein-Uhlenbeck process for temporally correlated exploration noise
        /// dX = θ(μ - X)dt + σdW
        const OUNoise = struct {
            size: usize,
            theta: T,
            sigma: T,
            state: []T,
            allocator: Allocator,

            fn init(allocator: Allocator, size: usize, theta: T, sigma: T) !OUNoise {
                const state = try allocator.alloc(T, size);
                @memset(state, 0);
                return OUNoise{
                    .size = size,
                    .theta = theta,
                    .sigma = sigma,
                    .state = state,
                    .allocator = allocator,
                };
            }

            fn deinit(self: *OUNoise) void {
                self.allocator.free(self.state);
            }

            fn sample(self: *OUNoise, rng: std.Random) []T {
                for (self.state) |*x| {
                    const dx = self.theta * (0 - x.*) + self.sigma * (rng.float(T) * 2 - 1);
                    x.* += dx;
                }
                return self.state;
            }

            fn reset(self: *OUNoise) void {
                @memset(self.state, 0);
            }
        };

        allocator: Allocator,
        config: Config,
        state_dim: usize,
        action_dim: usize,
        actor: ActorNetwork,
        actor_target: ActorNetwork,
        critic: CriticNetwork,
        critic_target: CriticNetwork,
        replay_buffer: ReplayBuffer,
        noise: OUNoise,
        rng: std.Random.DefaultPrng,
        action_buffer: []T, // Reusable buffer for actions

        /// Initialize DDPG agent
        /// Time: O(state_dim × hidden_size + hidden_size × action_dim + buffer_size)
        /// Space: O(buffer_size × (state_dim + action_dim + 3) + network_params)
        pub fn init(allocator: Allocator, state_dim: usize, action_dim: usize, config: Config) !Self {
            if (state_dim == 0) return error.InvalidStateDim;
            if (action_dim == 0) return error.InvalidActionDim;

            var rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
            const hidden_size = @max(64, (state_dim + action_dim) * 2); // Heuristic

            var actor = try ActorNetwork.init(allocator, state_dim, hidden_size, action_dim, rng.random());
            errdefer actor.deinit();

            var actor_target = try ActorNetwork.init(allocator, state_dim, hidden_size, action_dim, rng.random());
            errdefer actor_target.deinit();
            // Initialize target with same weights as actor
            actor_target.softUpdate(&actor, 1.0);

            var critic = try CriticNetwork.init(allocator, state_dim, action_dim, hidden_size, rng.random());
            errdefer critic.deinit();

            var critic_target = try CriticNetwork.init(allocator, state_dim, action_dim, hidden_size, rng.random());
            errdefer critic_target.deinit();
            // Initialize target with same weights as critic
            critic_target.softUpdate(&critic, 1.0);

            var replay_buffer = try ReplayBuffer.init(allocator, config.buffer_size);
            errdefer replay_buffer.deinit();

            var noise = try OUNoise.init(allocator, action_dim, config.noise_theta, config.noise_stddev);
            errdefer noise.deinit();

            const action_buffer = try allocator.alloc(T, action_dim);
            errdefer allocator.free(action_buffer);

            return Self{
                .allocator = allocator,
                .config = config,
                .state_dim = state_dim,
                .action_dim = action_dim,
                .actor = actor,
                .actor_target = actor_target,
                .critic = critic,
                .critic_target = critic_target,
                .replay_buffer = replay_buffer,
                .noise = noise,
                .rng = rng,
                .action_buffer = action_buffer,
            };
        }

        pub fn deinit(self: *Self) void {
            self.actor.deinit();
            self.actor_target.deinit();
            self.critic.deinit();
            self.critic_target.deinit();
            self.replay_buffer.deinit();
            self.noise.deinit();
            self.allocator.free(self.action_buffer);
        }

        /// Select action using deterministic policy μ(s) (no noise)
        /// Time: O(state_dim × hidden_size + hidden_size × action_dim)
        pub fn selectAction(self: *Self, state: []const T) ![]const T {
            if (state.len != self.state_dim) return error.InvalidState;
            self.actor.forward(state, self.action_buffer);
            return self.action_buffer;
        }

        /// Select action with exploration noise: a = μ(s) + N
        /// Time: O(actor_forward + noise_sample)
        pub fn selectActionNoisy(self: *Self, state: []const T) ![]const T {
            if (state.len != self.state_dim) return error.InvalidState;

            self.actor.forward(state, self.action_buffer);
            const noise_sample = self.noise.sample(self.rng.random());

            // Add noise and clip to [-1, 1]
            for (self.action_buffer, 0..) |*a, i| {
                a.* = std.math.clamp(a.* + noise_sample[i], -1.0, 1.0);
            }

            return self.action_buffer;
        }

        /// Store experience in replay buffer
        /// Time: O(state_dim + action_dim)
        pub fn store(self: *Self, state: []const T, action: []const T, reward: T, next_state: []const T, done: bool) !void {
            try self.replay_buffer.add(state, action, reward, next_state, done);
        }

        /// Train actor and critic networks on a minibatch
        /// Time: O(batch_size × (actor_forward + critic_forward + backprop))
        pub fn train(self: *Self) !void {
            if (self.replay_buffer.size < self.config.batch_size) return; // Not enough samples

            const batch = self.replay_buffer.sample(self.config.batch_size, self.rng.random());
            defer self.allocator.free(batch);

            // Critic update: minimize (Q(s,a) - y)² where y = r + γQ'(s', μ'(s'))
            var critic_loss: T = 0;
            for (batch) |exp| {
                // Compute target Q-value
                const next_action = try self.allocator.alloc(T, self.action_dim);
                defer self.allocator.free(next_action);
                self.actor_target.forward(exp.next_state, next_action);
                const next_q = self.critic_target.forward(exp.next_state, next_action);
                const target_q = if (exp.done) exp.reward else exp.reward + self.config.gamma * next_q;

                // Compute current Q-value
                const current_q = self.critic.forward(exp.state, exp.action);

                // TD error
                const td_error = target_q - current_q;
                critic_loss += td_error * td_error;

                // Simple gradient descent update (simplified for demonstration)
                // In practice, would use proper backprop through both layers
                const grad = -2.0 * td_error * self.config.critic_lr;
                for (self.critic.w2, 0..) |*w, i| {
                    w.* -= grad * self.critic.h[i];
                }
                self.critic.b2 -= grad;
            }

            // Actor update: maximize Q(s, μ(s)) via policy gradient
            for (batch) |exp| {
                const action = try self.allocator.alloc(T, self.action_dim);
                defer self.allocator.free(action);
                self.actor.forward(exp.state, action);
                const q_value = self.critic.forward(exp.state, action);

                // Policy gradient: ∇μ J ≈ ∇μ Q(s, μ(s))
                // Simplified gradient update (in practice, would use chain rule through both networks)
                const grad = self.config.actor_lr * q_value / @as(T, @floatFromInt(batch.len));
                // Update w2: [hidden_size x output_size] matrix
                for (0..self.action_dim) |i| {
                    for (0..self.actor.hidden_size) |j| {
                        self.actor.w2[j * self.action_dim + i] += grad * self.actor.h[j];
                    }
                }
            }

            // Soft update target networks: θ' ← τθ + (1-τ)θ'
            self.actor_target.softUpdate(&self.actor, self.config.tau);
            self.critic_target.softUpdate(&self.critic, self.config.tau);
        }

        /// Reset exploration noise
        pub fn resetNoise(self: *Self) void {
            self.noise.reset();
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "DDPG: initialization" {
    const config = DDPG(f64).Config{};
    var agent = try DDPG(f64).init(testing.allocator, 4, 2, config);
    defer agent.deinit();

    try testing.expectEqual(4, agent.state_dim);
    try testing.expectEqual(2, agent.action_dim);
    try testing.expectEqual(0, agent.replay_buffer.size);
}

test "DDPG: deterministic action selection" {
    const config = DDPG(f64).Config{};
    var agent = try DDPG(f64).init(testing.allocator, 4, 2, config);
    defer agent.deinit();

    const state = [_]f64{ 1.0, 0.5, -0.5, 0.0 };
    const action = try agent.selectAction(&state);

    // Actions should be in [-1, 1] due to tanh
    try testing.expect(action.len == 2);
    for (action) |a| {
        try testing.expect(a >= -1.0 and a <= 1.0);
    }

    // Deterministic: same state → same action
    const action2 = try agent.selectAction(&state);
    for (action, action2) |a1, a2| {
        try testing.expectApproxEqAbs(a1, a2, 1e-10);
    }
}

test "DDPG: noisy action selection" {
    const config = DDPG(f64).Config{ .noise_stddev = 0.1 };
    var agent = try DDPG(f64).init(testing.allocator, 4, 2, config);
    defer agent.deinit();

    const state = [_]f64{ 1.0, 0.5, -0.5, 0.0 };
    const action1 = try agent.selectActionNoisy(&state);
    const action2 = try agent.selectActionNoisy(&state);

    // With noise, actions should differ (though they might be close)
    try testing.expect(action1.len == 2);
    try testing.expect(action2.len == 2);

    // Actions still bounded to [-1, 1]
    for (action1) |a| {
        try testing.expect(a >= -1.0 and a <= 1.0);
    }
}

test "DDPG: replay buffer add and sample" {
    const config = DDPG(f64).Config{ .buffer_size = 100 };
    var agent = try DDPG(f64).init(testing.allocator, 4, 2, config);
    defer agent.deinit();

    const state = [_]f64{ 1.0, 0.5, -0.5, 0.0 };
    const action = [_]f64{ 0.5, -0.3 };
    const next_state = [_]f64{ 1.1, 0.6, -0.4, 0.1 };

    try agent.store(&state, &action, 1.0, &next_state, false);
    try testing.expectEqual(1, agent.replay_buffer.size);

    // Add more experiences
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        try agent.store(&state, &action, @as(f64, @floatFromInt(i)), &next_state, false);
    }
    try testing.expectEqual(11, agent.replay_buffer.size);
}

test "DDPG: replay buffer overflow" {
    const config = DDPG(f64).Config{ .buffer_size = 5 };
    var agent = try DDPG(f64).init(testing.allocator, 2, 1, config);
    defer agent.deinit();

    const state = [_]f64{ 1.0, 0.5 };
    const action = [_]f64{0.5};
    const next_state = [_]f64{ 1.1, 0.6 };

    // Add 10 experiences to buffer of size 5 (circular)
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        try agent.store(&state, &action, @as(f64, @floatFromInt(i)), &next_state, false);
    }

    try testing.expectEqual(5, agent.replay_buffer.size); // Capped at buffer_size
}

test "DDPG: critic Q-value computation" {
    const config = DDPG(f64).Config{};
    var agent = try DDPG(f64).init(testing.allocator, 3, 2, config);
    defer agent.deinit();

    const state = [_]f64{ 1.0, 0.5, -0.5 };
    const action = [_]f64{ 0.8, -0.3 };

    const q = agent.critic.forward(&state, &action);
    // Q-value should be a finite number
    try testing.expect(std.math.isFinite(q));
}

test "DDPG: soft target update" {
    const config = DDPG(f64).Config{ .tau = 0.1 };
    var agent = try DDPG(f64).init(testing.allocator, 3, 2, config);
    defer agent.deinit();

    // Get initial target weights
    const initial_w1_0 = agent.actor_target.w1[0];

    // Modify actor weights
    agent.actor.w1[0] += 10.0;

    // Soft update
    agent.actor_target.softUpdate(&agent.actor, config.tau);

    // Target should move toward actor: new = 0.1 * (initial+10) + 0.9 * initial
    const expected = 0.1 * (initial_w1_0 + 10.0) + 0.9 * initial_w1_0;
    try testing.expectApproxEqAbs(expected, agent.actor_target.w1[0], 1e-6);
}

test "DDPG: training with sufficient samples" {
    const config = DDPG(f64).Config{ .batch_size = 4, .buffer_size = 100 };
    var agent = try DDPG(f64).init(testing.allocator, 3, 2, config);
    defer agent.deinit();

    const state = [_]f64{ 1.0, 0.5, -0.5 };
    const action = [_]f64{ 0.5, -0.3 };
    const next_state = [_]f64{ 1.1, 0.6, -0.4 };

    // Add enough samples for training
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        try agent.store(&state, &action, 1.0, &next_state, false);
    }

    // Training should not error with enough samples
    try agent.train();
}

test "DDPG: training with insufficient samples" {
    const config = DDPG(f64).Config{ .batch_size = 32 };
    var agent = try DDPG(f64).init(testing.allocator, 3, 2, config);
    defer agent.deinit();

    // No samples in buffer
    try agent.train(); // Should return early, not error
}

test "DDPG: terminal state handling" {
    const config = DDPG(f64).Config{ .batch_size = 2 };
    var agent = try DDPG(f64).init(testing.allocator, 2, 1, config);
    defer agent.deinit();

    const state = [_]f64{ 1.0, 0.5 };
    const action = [_]f64{0.5};
    const next_state = [_]f64{ 0.0, 0.0 };

    // Add terminal transition (done=true, no future reward)
    try agent.store(&state, &action, 10.0, &next_state, true);
    try agent.store(&state, &action, 5.0, &next_state, false);

    try agent.train();
    // Should not error when handling terminal states
}

test "DDPG: reset noise" {
    const config = DDPG(f64).Config{};
    var agent = try DDPG(f64).init(testing.allocator, 3, 2, config);
    defer agent.deinit();

    const state = [_]f64{ 1.0, 0.5, -0.5 };

    // Generate noise
    _ = try agent.selectActionNoisy(&state);

    // Reset noise
    agent.resetNoise();

    // Noise state should be zero
    for (agent.noise.state) |x| {
        try testing.expectEqual(0.0, x);
    }
}

test "DDPG: f32 support" {
    const config = DDPG(f32).Config{};
    var agent = try DDPG(f32).init(testing.allocator, 4, 2, config);
    defer agent.deinit();

    const state = [_]f32{ 1.0, 0.5, -0.5, 0.0 };
    const action = try agent.selectAction(&state);

    try testing.expect(action.len == 2);
    for (action) |a| {
        try testing.expect(a >= -1.0 and a <= 1.0);
    }
}

test "DDPG: large state and action spaces" {
    const config = DDPG(f64).Config{ .buffer_size = 100 };
    var agent = try DDPG(f64).init(testing.allocator, 50, 10, config);
    defer agent.deinit();

    var state: [50]f64 = undefined;
    for (&state, 0..) |*s, i| {
        s.* = @as(f64, @floatFromInt(i)) / 50.0;
    }

    const action = try agent.selectAction(&state);
    try testing.expectEqual(10, action.len);

    for (action) |a| {
        try testing.expect(a >= -1.0 and a <= 1.0);
    }
}

test "DDPG: error handling - invalid state dim" {
    const config = DDPG(f64).Config{};
    const result = DDPG(f64).init(testing.allocator, 0, 2, config);
    try testing.expectError(error.InvalidStateDim, result);
}

test "DDPG: error handling - invalid action dim" {
    const config = DDPG(f64).Config{};
    const result = DDPG(f64).init(testing.allocator, 4, 0, config);
    try testing.expectError(error.InvalidActionDim, result);
}

test "DDPG: error handling - invalid state size" {
    const config = DDPG(f64).Config{};
    var agent = try DDPG(f64).init(testing.allocator, 4, 2, config);
    defer agent.deinit();

    const wrong_state = [_]f64{ 1.0, 0.5 }; // Only 2 elements, expected 4
    const result = agent.selectAction(&wrong_state);
    try testing.expectError(error.InvalidState, result);
}

test "DDPG: memory safety - leak detection" {
    const config = DDPG(f64).Config{ .buffer_size = 10 };
    var agent = try DDPG(f64).init(testing.allocator, 3, 2, config);
    defer agent.deinit();

    const state = [_]f64{ 1.0, 0.5, -0.5 };
    const action = [_]f64{ 0.5, -0.3 };
    const next_state = [_]f64{ 1.1, 0.6, -0.4 };

    var i: usize = 0;
    while (i < 20) : (i += 1) {
        try agent.store(&state, &action, 1.0, &next_state, false);
    }

    // Multiple train calls
    try agent.train();
    try agent.train();

    // testing.allocator will detect leaks
}
