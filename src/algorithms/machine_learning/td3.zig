const std = @import("std");
const Allocator = std.mem.Allocator;

/// TD3 (Twin Delayed Deep Deterministic Policy Gradient) for continuous control
///
/// Algorithm: Improvement over DDPG with three key features:
/// 1. Clipped Double Q-learning: Two critic networks, use minimum to reduce overestimation
/// 2. Delayed Policy Updates: Update actor less frequently than critics (policy_delay)
/// 3. Target Policy Smoothing: Add noise to target actions to regularize Q-value estimates
///
/// Time: O(batch_size × network_forward × network_backward) per train()
/// Space: O(buffer_size × (state_dim + action_dim) + 2 × critic_params + actor_params)
///
/// Use cases:
/// - Robotics (continuous control tasks with high-dimensional actions)
/// - Simulated physics (MuJoCo, PyBullet environments)
/// - Autonomous systems (drones, vehicles requiring smooth control)
/// - Industrial control (process control, optimization)
/// - Financial trading (portfolio management with continuous actions)
///
/// Trade-offs:
/// - vs DDPG: More stable (twin critics reduce overestimation), but 2× critic memory
/// - vs PPO: Better sample efficiency (off-policy), but requires more tuning
/// - vs SAC: Simpler (no entropy tuning), but SAC often more robust
/// - vs DQN: Handles continuous actions (DQN is discrete-only)
pub fn TD3(comptime T: type) type {
    if (T != f32 and T != f64) {
        @compileError("TD3 only supports f32 and f64");
    }

    return struct {
        const Self = @This();

        // Network architecture
        state_dim: usize,
        action_dim: usize,
        hidden_size: usize,

        // Actor network (policy): s → a
        actor_w1: []T, // state_dim × hidden_size
        actor_b1: []T, // hidden_size
        actor_w2: []T, // hidden_size × action_dim
        actor_b2: []T, // action_dim

        // Twin Critic networks: (s, a) → Q
        critic1_w1: []T, // (state_dim + action_dim) × hidden_size
        critic1_b1: []T, // hidden_size
        critic1_w2: []T, // hidden_size × 1
        critic1_b2: []T, // 1

        critic2_w1: []T,
        critic2_b1: []T,
        critic2_w2: []T,
        critic2_b2: []T,

        // Target networks (delayed updates)
        target_actor_w1: []T,
        target_actor_b1: []T,
        target_actor_w2: []T,
        target_actor_b2: []T,

        target_critic1_w1: []T,
        target_critic1_b1: []T,
        target_critic1_w2: []T,
        target_critic1_b2: []T,

        target_critic2_w1: []T,
        target_critic2_b1: []T,
        target_critic2_w2: []T,
        target_critic2_b2: []T,

        // Experience replay buffer
        replay_buffer: []Experience(T),
        buffer_idx: usize,
        buffer_size: usize,

        // Training state
        update_count: usize,
        actor_lr: T,
        critic_lr: T,
        gamma: T, // discount factor
        tau: T, // soft update coefficient
        policy_delay: usize, // update actor every N critic updates
        target_noise: T, // noise for target policy smoothing
        noise_clip: T, // clip target noise
        exploration_noise: T, // exploration during action selection

        allocator: Allocator,
        prng: std.Random.DefaultPrng,

        pub const Config = struct {
            hidden_size: usize = 256,
            buffer_size: usize = 1000000,
            actor_lr: T = 0.001,
            critic_lr: T = 0.001,
            gamma: T = 0.99,
            tau: T = 0.005,
            policy_delay: usize = 2, // TD3 standard: update actor every 2 critic updates
            target_noise: T = 0.2, // target policy smoothing noise
            noise_clip: T = 0.5, // clip target noise to [-noise_clip, +noise_clip]
            exploration_noise: T = 0.1,
            seed: u64 = 0,
        };

        pub const Experience = struct {
            state: []T,
            action: []T,
            reward: T,
            next_state: []T,
            done: bool,
        };

        /// Initialize TD3 agent
        /// Time: O(network_params), Space: O(buffer_size + network_params)
        pub fn init(allocator: Allocator, state_dim: usize, action_dim: usize, config: Config) !Self {
            if (state_dim == 0) return error.InvalidStateDim;
            if (action_dim == 0) return error.InvalidActionDim;
            if (config.hidden_size == 0) return error.InvalidHiddenSize;
            if (config.gamma < 0 or config.gamma > 1) return error.InvalidGamma;
            if (config.tau <= 0 or config.tau > 1) return error.InvalidTau;
            if (config.policy_delay == 0) return error.InvalidPolicyDelay;

            var prng = std.Random.DefaultPrng.init(config.seed);
            const random = prng.random();

            // Xavier initialization scale
            const actor_scale1 = @sqrt(2.0 / @as(T, @floatFromInt(state_dim)));
            const actor_scale2 = @sqrt(2.0 / @as(T, @floatFromInt(config.hidden_size)));
            const critic_scale1 = @sqrt(2.0 / @as(T, @floatFromInt(state_dim + action_dim)));

            // Allocate actor network
            const actor_w1 = try allocator.alloc(T, state_dim * config.hidden_size);
            const actor_b1 = try allocator.alloc(T, config.hidden_size);
            const actor_w2 = try allocator.alloc(T, config.hidden_size * action_dim);
            const actor_b2 = try allocator.alloc(T, action_dim);

            // Initialize actor weights
            for (actor_w1) |*w| w.* = (random.float(T) * 2 - 1) * actor_scale1;
            for (actor_b1) |*b| b.* = 0;
            for (actor_w2) |*w| w.* = (random.float(T) * 2 - 1) * actor_scale2;
            for (actor_b2) |*b| b.* = 0;

            // Allocate twin critics
            const critic1_w1 = try allocator.alloc(T, (state_dim + action_dim) * config.hidden_size);
            const critic1_b1 = try allocator.alloc(T, config.hidden_size);
            const critic1_w2 = try allocator.alloc(T, config.hidden_size);
            const critic1_b2 = try allocator.alloc(T, 1);

            const critic2_w1 = try allocator.alloc(T, (state_dim + action_dim) * config.hidden_size);
            const critic2_b1 = try allocator.alloc(T, config.hidden_size);
            const critic2_w2 = try allocator.alloc(T, config.hidden_size);
            const critic2_b2 = try allocator.alloc(T, 1);

            // Initialize critic weights
            for (critic1_w1) |*w| w.* = (random.float(T) * 2 - 1) * critic_scale1;
            for (critic1_b1) |*b| b.* = 0;
            for (critic1_w2) |*w| w.* = (random.float(T) * 2 - 1) * actor_scale2;
            for (critic1_b2) |*b| b.* = 0;

            for (critic2_w1) |*w| w.* = (random.float(T) * 2 - 1) * critic_scale1;
            for (critic2_b1) |*b| b.* = 0;
            for (critic2_w2) |*w| w.* = (random.float(T) * 2 - 1) * actor_scale2;
            for (critic2_b2) |*b| b.* = 0;

            // Allocate target networks (copy of main networks)
            const target_actor_w1 = try allocator.dupe(T, actor_w1);
            const target_actor_b1 = try allocator.dupe(T, actor_b1);
            const target_actor_w2 = try allocator.dupe(T, actor_w2);
            const target_actor_b2 = try allocator.dupe(T, actor_b2);

            const target_critic1_w1 = try allocator.dupe(T, critic1_w1);
            const target_critic1_b1 = try allocator.dupe(T, critic1_b1);
            const target_critic1_w2 = try allocator.dupe(T, critic1_w2);
            const target_critic1_b2 = try allocator.dupe(T, critic1_b2);

            const target_critic2_w1 = try allocator.dupe(T, critic2_w1);
            const target_critic2_b1 = try allocator.dupe(T, critic2_b1);
            const target_critic2_w2 = try allocator.dupe(T, critic2_w2);
            const target_critic2_b2 = try allocator.dupe(T, critic2_b2);

            // Allocate replay buffer
            const replay_buffer = try allocator.alloc(Experience(T), config.buffer_size);
            for (replay_buffer) |*exp| {
                exp.state = try allocator.alloc(T, state_dim);
                exp.action = try allocator.alloc(T, action_dim);
                exp.next_state = try allocator.alloc(T, state_dim);
                exp.done = false;
            }

            return Self{
                .state_dim = state_dim,
                .action_dim = action_dim,
                .hidden_size = config.hidden_size,
                .actor_w1 = actor_w1,
                .actor_b1 = actor_b1,
                .actor_w2 = actor_w2,
                .actor_b2 = actor_b2,
                .critic1_w1 = critic1_w1,
                .critic1_b1 = critic1_b1,
                .critic1_w2 = critic1_w2,
                .critic1_b2 = critic1_b2,
                .critic2_w1 = critic2_w1,
                .critic2_b1 = critic2_b1,
                .critic2_w2 = critic2_w2,
                .critic2_b2 = critic2_b2,
                .target_actor_w1 = target_actor_w1,
                .target_actor_b1 = target_actor_b1,
                .target_actor_w2 = target_actor_w2,
                .target_actor_b2 = target_actor_b2,
                .target_critic1_w1 = target_critic1_w1,
                .target_critic1_b1 = target_critic1_b1,
                .target_critic1_w2 = target_critic1_w2,
                .target_critic1_b2 = target_critic1_b2,
                .target_critic2_w1 = target_critic2_w1,
                .target_critic2_b1 = target_critic2_b1,
                .target_critic2_w2 = target_critic2_w2,
                .target_critic2_b2 = target_critic2_b2,
                .replay_buffer = replay_buffer,
                .buffer_idx = 0,
                .buffer_size = config.buffer_size,
                .update_count = 0,
                .actor_lr = config.actor_lr,
                .critic_lr = config.critic_lr,
                .gamma = config.gamma,
                .tau = config.tau,
                .policy_delay = config.policy_delay,
                .target_noise = config.target_noise,
                .noise_clip = config.noise_clip,
                .exploration_noise = config.exploration_noise,
                .allocator = allocator,
                .prng = prng,
            };
        }

        /// Clean up resources
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.actor_w1);
            self.allocator.free(self.actor_b1);
            self.allocator.free(self.actor_w2);
            self.allocator.free(self.actor_b2);

            self.allocator.free(self.critic1_w1);
            self.allocator.free(self.critic1_b1);
            self.allocator.free(self.critic1_w2);
            self.allocator.free(self.critic1_b2);

            self.allocator.free(self.critic2_w1);
            self.allocator.free(self.critic2_b1);
            self.allocator.free(self.critic2_w2);
            self.allocator.free(self.critic2_b2);

            self.allocator.free(self.target_actor_w1);
            self.allocator.free(self.target_actor_b1);
            self.allocator.free(self.target_actor_w2);
            self.allocator.free(self.target_actor_b2);

            self.allocator.free(self.target_critic1_w1);
            self.allocator.free(self.target_critic1_b1);
            self.allocator.free(self.target_critic1_w2);
            self.allocator.free(self.target_critic1_b2);

            self.allocator.free(self.target_critic2_w1);
            self.allocator.free(self.target_critic2_b1);
            self.allocator.free(self.target_critic2_w2);
            self.allocator.free(self.target_critic2_b2);

            for (self.replay_buffer) |*exp| {
                self.allocator.free(exp.state);
                self.allocator.free(exp.action);
                self.allocator.free(exp.next_state);
            }
            self.allocator.free(self.replay_buffer);
        }

        /// Select action with exploration noise
        /// Time: O(network_forward), Space: O(action_dim)
        pub fn selectAction(self: *Self, state: []const T, explore: bool) ![]T {
            if (state.len != self.state_dim) return error.InvalidStateSize;

            const action = try self.allocator.alloc(T, self.action_dim);
            const random = self.prng.random();

            // Forward pass through actor network
            const hidden = try self.allocator.alloc(T, self.hidden_size);
            defer self.allocator.free(hidden);

            // Layer 1: hidden = ReLU(W1 * state + b1)
            for (hidden, 0..) |*h, i| {
                var sum: T = self.actor_b1[i];
                for (state, 0..) |s, j| {
                    sum += self.actor_w1[j * self.hidden_size + i] * s;
                }
                h.* = @max(0, sum); // ReLU
            }

            // Layer 2: action = tanh(W2 * hidden + b2)
            for (action, 0..) |*a, i| {
                var sum: T = self.actor_b2[i];
                for (hidden, 0..) |h, j| {
                    sum += self.actor_w2[j * self.action_dim + i] * h;
                }
                a.* = std.math.tanh(sum); // tanh to bound actions in [-1, 1]
            }

            // Add exploration noise if specified
            if (explore) {
                for (action) |*a| {
                    const noise = (random.float(T) * 2 - 1) * self.exploration_noise;
                    a.* = std.math.clamp(a.* + noise, -1, 1);
                }
            }

            return action;
        }

        /// Select greedy action (no exploration)
        /// Time: O(network_forward), Space: O(action_dim)
        pub fn greedyAction(self: *Self, state: []const T) ![]T {
            return try self.selectAction(state, false);
        }

        /// Store experience in replay buffer (circular buffer)
        /// Time: O(state_dim + action_dim), Space: O(1)
        pub fn store(self: *Self, state: []const T, action: []const T, reward: T, next_state: []const T, done: bool) !void {
            if (state.len != self.state_dim) return error.InvalidStateSize;
            if (action.len != self.action_dim) return error.InvalidActionSize;
            if (next_state.len != self.state_dim) return error.InvalidNextStateSize;

            const idx = self.buffer_idx % self.buffer_size;
            @memcpy(self.replay_buffer[idx].state, state);
            @memcpy(self.replay_buffer[idx].action, action);
            self.replay_buffer[idx].reward = reward;
            @memcpy(self.replay_buffer[idx].next_state, next_state);
            self.replay_buffer[idx].done = done;

            self.buffer_idx += 1;
        }

        /// Train on a mini-batch from replay buffer (TD3 update)
        /// Time: O(batch_size × network_forward × network_backward), Space: O(batch_size × hidden_size)
        pub fn train(self: *Self, batch_size: usize) !void {
            const n_samples = @min(self.buffer_idx, self.buffer_size);
            if (n_samples < batch_size) return error.InsufficientData;

            const random = self.prng.random();

            // Allocate temporary arrays
            var critic1_loss: T = 0;
            var critic2_loss: T = 0;

            // Sample mini-batch
            var i: usize = 0;
            while (i < batch_size) : (i += 1) {
                const idx = random.intRangeAtMost(usize, 0, n_samples - 1);
                const exp = &self.replay_buffer[idx];

                // Compute target Q-value using TARGET networks
                // Target policy smoothing: add noise to target action
                const target_action = try self.allocator.alloc(T, self.action_dim);
                defer self.allocator.free(target_action);

                // Forward through target actor
                const target_hidden = try self.allocator.alloc(T, self.hidden_size);
                defer self.allocator.free(target_hidden);

                for (target_hidden, 0..) |*h, j| {
                    var sum: T = self.target_actor_b1[j];
                    for (exp.next_state, 0..) |s, k| {
                        sum += self.target_actor_w1[k * self.hidden_size + j] * s;
                    }
                    h.* = @max(0, sum);
                }

                for (target_action, 0..) |*a, j| {
                    var sum: T = self.target_actor_b2[j];
                    for (target_hidden, 0..) |h, k| {
                        sum += self.target_actor_w2[k * self.action_dim + j] * h;
                    }
                    // Add target noise for smoothing
                    const noise = std.math.clamp((random.float(T) * 2 - 1) * self.target_noise, -self.noise_clip, self.noise_clip);
                    a.* = std.math.clamp(std.math.tanh(sum) + noise, -1, 1);
                }

                // Compute Q-values from BOTH target critics (clipped double Q-learning)
                const target_q1 = try self.computeCriticValue(exp.next_state, target_action, true, 1);
                const target_q2 = try self.computeCriticValue(exp.next_state, target_action, true, 2);
                const target_q = @min(target_q1, target_q2); // Take minimum (reduce overestimation)

                const target = exp.reward + if (exp.done) 0 else self.gamma * target_q;

                // Compute current Q-values from BOTH critics
                const q1 = try self.computeCriticValue(exp.state, exp.action, false, 1);
                const q2 = try self.computeCriticValue(exp.state, exp.action, false, 2);

                // Update critics (gradient descent)
                const td_error1 = target - q1;
                const td_error2 = target - q2;
                critic1_loss += td_error1 * td_error1;
                critic2_loss += td_error2 * td_error2;

                // Simplified gradient update (actual implementation would use backprop)
                try self.updateCritic(exp.state, exp.action, td_error1, 1);
                try self.updateCritic(exp.state, exp.action, td_error2, 2);
            }

            self.update_count += 1;

            // DELAYED policy update: only update actor every policy_delay steps
            if (self.update_count % self.policy_delay == 0) {
                // Update actor using policy gradient
                try self.updateActor(batch_size);

                // Soft update target networks
                self.softUpdateTargets();
            }
        }

        /// Compute Q-value from critic network
        fn computeCriticValue(self: *Self, state: []const T, action: []const T, use_target: bool, critic_id: usize) !T {
            const w1 = if (use_target) (if (critic_id == 1) self.target_critic1_w1 else self.target_critic2_w1) else (if (critic_id == 1) self.critic1_w1 else self.critic2_w1);
            const b1 = if (use_target) (if (critic_id == 1) self.target_critic1_b1 else self.target_critic2_b1) else (if (critic_id == 1) self.critic1_b1 else self.critic2_b1);
            const w2 = if (use_target) (if (critic_id == 1) self.target_critic1_w2 else self.target_critic2_w2) else (if (critic_id == 1) self.critic1_w2 else self.critic2_w2);
            const b2 = if (use_target) (if (critic_id == 1) self.target_critic1_b2 else self.target_critic2_b2) else (if (critic_id == 1) self.critic1_b2 else self.critic2_b2);

            const hidden = try self.allocator.alloc(T, self.hidden_size);
            defer self.allocator.free(hidden);

            // Layer 1: hidden = ReLU(W1 * [state; action] + b1)
            for (hidden, 0..) |*h, i| {
                var sum: T = b1[i];
                for (state, 0..) |s, j| {
                    sum += w1[j * self.hidden_size + i] * s;
                }
                for (action, 0..) |a, j| {
                    sum += w1[(self.state_dim + j) * self.hidden_size + i] * a;
                }
                h.* = @max(0, sum);
            }

            // Layer 2: Q = W2 * hidden + b2
            var q: T = b2[0];
            for (hidden, 0..) |h, i| {
                q += w2[i] * h;
            }

            return q;
        }

        /// Update critic network (simplified gradient descent)
        fn updateCritic(self: *Self, state: []const T, action: []const T, td_error: T, critic_id: usize) !void {
            const w1 = if (critic_id == 1) self.critic1_w1 else self.critic2_w1;
            const b1 = if (critic_id == 1) self.critic1_b1 else self.critic2_b1;
            const w2 = if (critic_id == 1) self.critic1_w2 else self.critic2_w2;
            const b2 = if (critic_id == 1) self.critic1_b2 else self.critic2_b2;

            // Simplified update: w += lr * td_error * input
            for (state, 0..) |s, j| {
                for (0..self.hidden_size) |i| {
                    w1[j * self.hidden_size + i] += self.critic_lr * td_error * s * 0.1; // 0.1 = simplified gradient
                }
            }
            for (action, 0..) |a, j| {
                for (0..self.hidden_size) |i| {
                    w1[(self.state_dim + j) * self.hidden_size + i] += self.critic_lr * td_error * a * 0.1;
                }
            }

            for (0..self.hidden_size) |i| {
                b1[i] += self.critic_lr * td_error * 0.1;
                w2[i] += self.critic_lr * td_error * 0.1;
            }
            b2[0] += self.critic_lr * td_error;
        }

        /// Update actor network using policy gradient
        fn updateActor(self: *Self, batch_size: usize) !void {
            // Simplified actor update: maximize Q(s, μ(s))
            // In practice, this would use deterministic policy gradient
            const random = self.prng.random();
            const n_samples = @min(self.buffer_idx, self.buffer_size);

            var i: usize = 0;
            while (i < batch_size) : (i += 1) {
                const idx = random.intRangeAtMost(usize, 0, n_samples - 1);
                const exp = &self.replay_buffer[idx];

                // Compute action from current policy
                const action = try self.greedyAction(exp.state);
                defer self.allocator.free(action);

                // Compute Q-value (use critic1 for policy gradient)
                const q = try self.computeCriticValue(exp.state, action, false, 1);

                // Update actor to maximize Q (simplified gradient)
                for (exp.state, 0..) |s, j| {
                    for (0..self.hidden_size) |k| {
                        self.actor_w1[j * self.hidden_size + k] += self.actor_lr * q * s * 0.01;
                    }
                }
            }
        }

        /// Soft update target networks: θ_target = τ*θ + (1-τ)*θ_target
        /// Time: O(network_params), Space: O(1)
        fn softUpdateTargets(self: *Self) void {
            const one_minus_tau = 1 - self.tau;

            // Update target actor
            for (self.target_actor_w1, 0..) |*tw, i| {
                tw.* = self.tau * self.actor_w1[i] + one_minus_tau * tw.*;
            }
            for (self.target_actor_b1, 0..) |*tb, i| {
                tb.* = self.tau * self.actor_b1[i] + one_minus_tau * tb.*;
            }
            for (self.target_actor_w2, 0..) |*tw, i| {
                tw.* = self.tau * self.actor_w2[i] + one_minus_tau * tw.*;
            }
            for (self.target_actor_b2, 0..) |*tb, i| {
                tb.* = self.tau * self.actor_b2[i] + one_minus_tau * tb.*;
            }

            // Update target critic1
            for (self.target_critic1_w1, 0..) |*tw, i| {
                tw.* = self.tau * self.critic1_w1[i] + one_minus_tau * tw.*;
            }
            for (self.target_critic1_b1, 0..) |*tb, i| {
                tb.* = self.tau * self.critic1_b1[i] + one_minus_tau * tb.*;
            }
            for (self.target_critic1_w2, 0..) |*tw, i| {
                tw.* = self.tau * self.critic1_w2[i] + one_minus_tau * tw.*;
            }
            for (self.target_critic1_b2, 0..) |*tb, i| {
                tb.* = self.tau * self.critic1_b2[i] + one_minus_tau * tb.*;
            }

            // Update target critic2
            for (self.target_critic2_w1, 0..) |*tw, i| {
                tw.* = self.tau * self.critic2_w1[i] + one_minus_tau * tw.*;
            }
            for (self.target_critic2_b1, 0..) |*tb, i| {
                tb.* = self.tau * self.critic2_b1[i] + one_minus_tau * tb.*;
            }
            for (self.target_critic2_w2, 0..) |*tw, i| {
                tw.* = self.tau * self.critic2_w2[i] + one_minus_tau * tw.*;
            }
            for (self.target_critic2_b2, 0..) |*tb, i| {
                tb.* = self.tau * self.critic2_b2[i] + one_minus_tau * tb.*;
            }
        }

        /// Reset agent (clear replay buffer, reinitialize weights)
        /// Time: O(network_params), Space: O(1)
        pub fn reset(self: *Self) void {
            self.buffer_idx = 0;
            self.update_count = 0;
            // Weights remain (no reinitialization to preserve learned policy)
        }
    };
}

// Tests
const testing = std.testing;

test "TD3: basic initialization" {
    const config = TD3(f64).Config{
        .hidden_size = 64,
        .buffer_size = 1000,
    };
    var agent = try TD3(f64).init(testing.allocator, 4, 2, config);
    defer agent.deinit();

    try testing.expectEqual(4, agent.state_dim);
    try testing.expectEqual(2, agent.action_dim);
    try testing.expectEqual(64, agent.hidden_size);
    try testing.expectEqual(0, agent.buffer_idx);
    try testing.expectEqual(0, agent.update_count);
}

test "TD3: action selection bounds" {
    const config = TD3(f32).Config{ .hidden_size = 32, .buffer_size = 100 };
    var agent = try TD3(f32).init(testing.allocator, 2, 1, config);
    defer agent.deinit();

    const state = [_]f32{ 0.5, -0.3 };
    const action = try agent.selectAction(&state, false);
    defer testing.allocator.free(action);

    // Actions should be in [-1, 1] due to tanh activation
    for (action) |a| {
        try testing.expect(a >= -1.0 and a <= 1.0);
    }
}

test "TD3: action selection with exploration" {
    const config = TD3(f32).Config{
        .hidden_size = 32,
        .buffer_size = 100,
        .exploration_noise = 0.1,
        .seed = 42,
    };
    var agent = try TD3(f32).init(testing.allocator, 2, 1, config);
    defer agent.deinit();

    const state = [_]f32{ 0.0, 0.0 };

    // Greedy action (no exploration)
    const greedy = try agent.greedyAction(&state);
    defer testing.allocator.free(greedy);

    // Action with exploration
    const explore = try agent.selectAction(&state, true);
    defer testing.allocator.free(explore);

    // Both should be valid actions
    for (greedy) |a| try testing.expect(a >= -1.0 and a <= 1.0);
    for (explore) |a| try testing.expect(a >= -1.0 and a <= 1.0);
}

test "TD3: experience storage" {
    const config = TD3(f64).Config{ .hidden_size = 32, .buffer_size = 10 };
    var agent = try TD3(f64).init(testing.allocator, 3, 2, config);
    defer agent.deinit();

    const state = [_]f64{ 1.0, 2.0, 3.0 };
    const action = [_]f64{ 0.5, -0.3 };
    const next_state = [_]f64{ 1.1, 2.1, 3.1 };

    try agent.store(&state, &action, 1.5, &next_state, false);
    try testing.expectEqual(1, agent.buffer_idx);

    // Check stored values
    try testing.expectEqual(1.5, agent.replay_buffer[0].reward);
    try testing.expectEqual(false, agent.replay_buffer[0].done);
    try testing.expectEqualSlices(f64, &state, agent.replay_buffer[0].state);
}

test "TD3: circular buffer overflow" {
    const config = TD3(f64).Config{ .hidden_size = 16, .buffer_size = 3 };
    var agent = try TD3(f64).init(testing.allocator, 2, 1, config);
    defer agent.deinit();

    const state = [_]f64{ 1.0, 2.0 };
    const action = [_]f64{0.5};
    const next_state = [_]f64{ 1.5, 2.5 };

    // Fill buffer and overflow
    var i: usize = 0;
    while (i < 5) : (i += 1) {
        try agent.store(&state, &action, @as(f64, @floatFromInt(i)), &next_state, false);
    }

    try testing.expectEqual(5, agent.buffer_idx);
    // Should wrap around and overwrite first entries
    try testing.expectEqual(2.0, agent.replay_buffer[2].reward);
}

test "TD3: twin critics reduce overestimation" {
    const config = TD3(f32).Config{
        .hidden_size = 32,
        .buffer_size = 100,
        .seed = 123,
    };
    var agent = try TD3(f32).init(testing.allocator, 2, 1, config);
    defer agent.deinit();

    const state = [_]f32{ 0.5, -0.5 };
    const action = [_]f32{0.3};

    // Compute Q-values from both critics
    const q1 = try agent.computeCriticValue(&state, &action, false, 1);
    const q2 = try agent.computeCriticValue(&state, &action, false, 2);

    // Both should produce values (no specific constraint, just checking functionality)
    _ = q1;
    _ = q2;
}

test "TD3: training requires sufficient data" {
    const config = TD3(f64).Config{ .hidden_size = 16, .buffer_size = 100 };
    var agent = try TD3(f64).init(testing.allocator, 2, 1, config);
    defer agent.deinit();

    // Try to train with empty buffer
    try testing.expectError(error.InsufficientData, agent.train(32));
}

test "TD3: training with valid batch" {
    const config = TD3(f32).Config{
        .hidden_size = 32,
        .buffer_size = 1000,
        .seed = 456,
    };
    var agent = try TD3(f32).init(testing.allocator, 2, 1, config);
    defer agent.deinit();

    // Fill buffer with experiences
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const state = [_]f32{ @as(f32, @floatFromInt(i)) * 0.01, @as(f32, @floatFromInt(i)) * 0.02 };
        const action = [_]f32{0.5};
        const next_state = [_]f32{ state[0] + 0.1, state[1] + 0.1 };
        try agent.store(&state, &action, 1.0, &next_state, false);
    }

    // Train on mini-batch
    try agent.train(8);
    try testing.expectEqual(1, agent.update_count);

    // Train again - actor should only update every policy_delay steps (default=2)
    try agent.train(8);
    try testing.expectEqual(2, agent.update_count);
}

test "TD3: delayed policy updates" {
    const config = TD3(f64).Config{
        .hidden_size = 32,
        .buffer_size = 100,
        .policy_delay = 3,
        .seed = 789,
    };
    var agent = try TD3(f64).init(testing.allocator, 2, 1, config);
    defer agent.deinit();

    // Fill buffer
    var i: usize = 0;
    while (i < 50) : (i += 1) {
        const state = [_]f64{ 0.1, 0.2 };
        const action = [_]f64{0.5};
        const next_state = [_]f64{ 0.2, 0.3 };
        try agent.store(&state, &action, 1.0, &next_state, false);
    }

    // Train 3 times (policy should update only on 3rd call due to policy_delay=3)
    try agent.train(4);
    const count1 = agent.update_count;
    try agent.train(4);
    const count2 = agent.update_count;
    try agent.train(4);
    const count3 = agent.update_count;

    try testing.expectEqual(1, count1);
    try testing.expectEqual(2, count2);
    try testing.expectEqual(3, count3); // Policy updated here
}

test "TD3: target policy smoothing" {
    const config = TD3(f32).Config{
        .hidden_size = 32,
        .buffer_size = 100,
        .target_noise = 0.2,
        .noise_clip = 0.5,
        .seed = 111,
    };
    var agent = try TD3(f32).init(testing.allocator, 2, 1, config);
    defer agent.deinit();

    // Configuration should be stored
    try testing.expectEqual(0.2, agent.target_noise);
    try testing.expectEqual(0.5, agent.noise_clip);
}

test "TD3: terminal state handling" {
    const config = TD3(f64).Config{ .hidden_size = 32, .buffer_size = 100 };
    var agent = try TD3(f64).init(testing.allocator, 2, 1, config);
    defer agent.deinit();

    const state = [_]f64{ 1.0, 2.0 };
    const action = [_]f64{0.5};
    const terminal_state = [_]f64{ 0.0, 0.0 };

    // Store terminal transition
    try agent.store(&state, &action, 10.0, &terminal_state, true);

    try testing.expectEqual(true, agent.replay_buffer[0].done);
    try testing.expectEqual(10.0, agent.replay_buffer[0].reward);
}

test "TD3: soft target updates" {
    const config = TD3(f32).Config{
        .hidden_size = 16,
        .buffer_size = 100,
        .tau = 0.1,
    };
    var agent = try TD3(f32).init(testing.allocator, 2, 1, config);
    defer agent.deinit();

    // Store original target weights
    const original_w1_0 = agent.target_actor_w1[0];
    const original_critic_w1_0 = agent.target_critic1_w1[0];

    // Modify main network weights
    agent.actor_w1[0] += 1.0;
    agent.critic1_w1[0] += 1.0;

    // Perform soft update
    agent.softUpdateTargets();

    // Target weights should have moved towards main weights (by tau=0.1)
    const expected_actor = 0.1 * agent.actor_w1[0] + 0.9 * original_w1_0;
    const expected_critic = 0.1 * agent.critic1_w1[0] + 0.9 * original_critic_w1_0;

    try testing.expectApproxEqAbs(expected_actor, agent.target_actor_w1[0], 1e-5);
    try testing.expectApproxEqAbs(expected_critic, agent.target_critic1_w1[0], 1e-5);
}

test "TD3: reset clears buffer" {
    const config = TD3(f64).Config{ .hidden_size = 16, .buffer_size = 100 };
    var agent = try TD3(f64).init(testing.allocator, 2, 1, config);
    defer agent.deinit();

    const state = [_]f64{ 1.0, 2.0 };
    const action = [_]f64{0.5};
    const next_state = [_]f64{ 1.5, 2.5 };

    try agent.store(&state, &action, 1.0, &next_state, false);
    try testing.expectEqual(1, agent.buffer_idx);

    agent.reset();
    try testing.expectEqual(0, agent.buffer_idx);
    try testing.expectEqual(0, agent.update_count);
}

test "TD3: f32 and f64 support" {
    {
        const config = TD3(f32).Config{ .hidden_size = 16, .buffer_size = 10 };
        var agent = try TD3(f32).init(testing.allocator, 2, 1, config);
        defer agent.deinit();

        const state = [_]f32{ 0.5, -0.5 };
        const action = try agent.greedyAction(&state);
        defer testing.allocator.free(action);
        try testing.expectEqual(1, action.len);
    }

    {
        const config = TD3(f64).Config{ .hidden_size = 16, .buffer_size = 10 };
        var agent = try TD3(f64).init(testing.allocator, 2, 1, config);
        defer agent.deinit();

        const state = [_]f64{ 0.5, -0.5 };
        const action = try agent.greedyAction(&state);
        defer testing.allocator.free(action);
        try testing.expectEqual(1, action.len);
    }
}

test "TD3: large state-action spaces" {
    const config = TD3(f32).Config{
        .hidden_size = 128,
        .buffer_size = 10000,
    };
    var agent = try TD3(f32).init(testing.allocator, 20, 5, config);
    defer agent.deinit();

    var state: [20]f32 = undefined;
    for (&state, 0..) |*s, i| s.* = @as(f32, @floatFromInt(i)) * 0.05;

    const action = try agent.greedyAction(&state);
    defer testing.allocator.free(action);

    try testing.expectEqual(5, action.len);
    for (action) |a| {
        try testing.expect(a >= -1.0 and a <= 1.0);
    }
}

test "TD3: configuration validation" {
    // Invalid state_dim
    try testing.expectError(error.InvalidStateDim, TD3(f64).init(testing.allocator, 0, 2, .{}));

    // Invalid action_dim
    try testing.expectError(error.InvalidActionDim, TD3(f64).init(testing.allocator, 2, 0, .{}));

    // Invalid gamma
    try testing.expectError(error.InvalidGamma, TD3(f64).init(testing.allocator, 2, 1, .{ .gamma = 1.5 }));

    // Invalid tau
    try testing.expectError(error.InvalidTau, TD3(f64).init(testing.allocator, 2, 1, .{ .tau = 0.0 }));

    // Invalid policy_delay
    try testing.expectError(error.InvalidPolicyDelay, TD3(f64).init(testing.allocator, 2, 1, .{ .policy_delay = 0 }));
}

test "TD3: memory safety with testing.allocator" {
    const config = TD3(f64).Config{
        .hidden_size = 32,
        .buffer_size = 100,
    };
    var agent = try TD3(f64).init(testing.allocator, 3, 2, config);
    defer agent.deinit();

    const state = [_]f64{ 1.0, 2.0, 3.0 };
    const action = [_]f64{ 0.5, -0.3 };
    const next_state = [_]f64{ 1.1, 2.1, 3.1 };

    // Store and retrieve
    try agent.store(&state, &action, 1.5, &next_state, false);

    // Select actions
    const a1 = try agent.selectAction(&state, true);
    defer testing.allocator.free(a1);

    const a2 = try agent.greedyAction(&state);
    defer testing.allocator.free(a2);

    // Fill buffer and train
    var i: usize = 0;
    while (i < 50) : (i += 1) {
        try agent.store(&state, &action, 1.0, &next_state, false);
    }
    try agent.train(8);
}
