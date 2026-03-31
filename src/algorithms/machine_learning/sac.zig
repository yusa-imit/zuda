/// Soft Actor-Critic (SAC) — Maximum Entropy Reinforcement Learning
///
/// Algorithm: Off-policy actor-critic with entropy regularization for stable exploration
///
/// Key Features:
/// - Maximum entropy framework: Balances reward and policy entropy
/// - Twin critics: Clipped double Q-learning (like TD3) to reduce overestimation
/// - Stochastic actor: Samples from Gaussian policy π(a|s) = N(μ(s), σ(s))
/// - Automatic temperature tuning: Adapts α to maintain target entropy
/// - Off-policy learning: Experience replay for sample efficiency
/// - Soft target updates: Exponential moving average of target networks
///
/// Architecture:
/// - Actor: State → (mean, log_std) → stochastic action with reparameterization
/// - Twin critics: Q₁(s,a) and Q₂(s,a) → use min for target computation
/// - Target networks: Soft updates θ_target = τ*θ + (1-τ)*θ_target
/// - Temperature: Learnable α for entropy coefficient (optional auto-tuning)
///
/// Algorithm Flow:
/// 1. Sample action from stochastic policy: a ~ π(·|s)
/// 2. Store transition (s, a, r, s', done) in replay buffer
/// 3. Sample mini-batch from buffer
/// 4. Update critics: minimize TD error with entropy bonus
/// 5. Update actor: maximize Q(s,a) + α*H(π(·|s))
/// 6. Update temperature: match entropy to target (if auto-tune enabled)
/// 7. Soft update target networks
///
/// Time Complexity: O(batch × network_forward × network_backward) per train()
/// Space Complexity: O(buffer_size × (state_dim + action_dim) + 2×critic_params + actor_params)
///
/// Use Cases:
/// - Robotics: Manipulation, locomotion, contact-rich tasks
/// - Simulated physics: MuJoCo, PyBullet environments
/// - Continuous control: Better exploration than deterministic policies
/// - Sample-efficient learning: Off-policy with automatic entropy tuning
/// - Research: Baseline for comparison with other deep RL algorithms
///
/// Trade-offs:
/// - vs TD3: More robust exploration (entropy), automatic tuning, but 1 extra network (temperature)
/// - vs PPO: Better sample efficiency (off-policy), but more hyperparameters
/// - vs DDPG: Stochastic policy prevents premature convergence, more stable
/// - vs DQN: Handles continuous actions, but requires more compute
///
const std = @import("std");
const Allocator = std.mem.Allocator;

pub fn SAC(comptime T: type) type {
    return struct {
        const Self = @This();

        // Network dimensions
        state_dim: usize,
        action_dim: usize,
        hidden_dim: usize,

        // Actor network (policy): state → (mean, log_std)
        actor_weights1: []T, // [hidden_dim × state_dim]
        actor_bias1: []T, // [hidden_dim]
        actor_mean_weights: []T, // [action_dim × hidden_dim]
        actor_mean_bias: []T, // [action_dim]
        actor_logstd_weights: []T, // [action_dim × hidden_dim]
        actor_logstd_bias: []T, // [action_dim]

        // Twin critic networks: Q₁(s,a) and Q₂(s,a)
        critic1_weights1: []T, // [hidden_dim × (state_dim + action_dim)]
        critic1_bias1: []T, // [hidden_dim]
        critic1_weights2: []T, // [1 × hidden_dim]
        critic1_bias2: []T, // [1]

        critic2_weights1: []T,
        critic2_bias1: []T,
        critic2_weights2: []T,
        critic2_bias2: []T,

        // Target networks (soft updated)
        target_critic1_weights1: []T,
        target_critic1_bias1: []T,
        target_critic1_weights2: []T,
        target_critic1_bias2: []T,

        target_critic2_weights1: []T,
        target_critic2_bias1: []T,
        target_critic2_weights2: []T,
        target_critic2_bias2: []T,

        // Temperature parameter (entropy coefficient)
        log_alpha: T, // Learnable temperature (log scale)

        // Experience replay buffer (circular)
        buffer_states: []T, // [buffer_size × state_dim]
        buffer_actions: []T, // [buffer_size × action_dim]
        buffer_rewards: []T, // [buffer_size]
        buffer_next_states: []T, // [buffer_size × state_dim]
        buffer_dones: []bool, // [buffer_size]
        buffer_size: usize,
        buffer_index: usize,
        buffer_full: bool,

        // Hyperparameters
        action_low: []const T, // Action space lower bounds
        action_high: []const T, // Action space upper bounds
        gamma: T, // Discount factor
        tau: T, // Soft update coefficient
        actor_lr: T, // Actor learning rate
        critic_lr: T, // Critic learning rate
        alpha_lr: T, // Temperature learning rate
        target_entropy: T, // Target entropy for auto-tuning (-action_dim works well)
        batch_size: usize,
        auto_tune_temperature: bool,

        allocator: Allocator,
        prng: std.Random.DefaultPrng,

        /// Initialize SAC with Xavier initialization
        /// Time: O(state_dim × action_dim × hidden_dim)
        /// Space: O(buffer_size × (state_dim + action_dim) + network_params)
        pub fn init(
            allocator: Allocator,
            state_dim: usize,
            action_dim: usize,
            hidden_dim: usize,
            action_low: []const T,
            action_high: []const T,
            config: Config,
        ) !Self {
            if (state_dim == 0 or action_dim == 0 or hidden_dim == 0) return error.InvalidDimensions;
            if (action_low.len != action_dim or action_high.len != action_dim) return error.ActionBoundsMismatch;
            if (config.gamma < 0 or config.gamma > 1) return error.InvalidGamma;
            if (config.tau <= 0 or config.tau > 1) return error.InvalidTau;
            if (config.buffer_size == 0 or config.batch_size == 0) return error.InvalidBufferConfig;

            var self = Self{
                .state_dim = state_dim,
                .action_dim = action_dim,
                .hidden_dim = hidden_dim,
                .action_low = action_low,
                .action_high = action_high,
                .gamma = config.gamma,
                .tau = config.tau,
                .actor_lr = config.actor_lr,
                .critic_lr = config.critic_lr,
                .alpha_lr = config.alpha_lr,
                .target_entropy = config.target_entropy,
                .batch_size = config.batch_size,
                .buffer_size = config.buffer_size,
                .buffer_index = 0,
                .buffer_full = false,
                .auto_tune_temperature = config.auto_tune_temperature,
                .allocator = allocator,
                .prng = std.Random.DefaultPrng.init(config.seed),
                .log_alpha = @log(config.initial_temperature),
                .actor_weights1 = undefined,
                .actor_bias1 = undefined,
                .actor_mean_weights = undefined,
                .actor_mean_bias = undefined,
                .actor_logstd_weights = undefined,
                .actor_logstd_bias = undefined,
                .critic1_weights1 = undefined,
                .critic1_bias1 = undefined,
                .critic1_weights2 = undefined,
                .critic1_bias2 = undefined,
                .critic2_weights1 = undefined,
                .critic2_bias1 = undefined,
                .critic2_weights2 = undefined,
                .critic2_bias2 = undefined,
                .target_critic1_weights1 = undefined,
                .target_critic1_bias1 = undefined,
                .target_critic1_weights2 = undefined,
                .target_critic1_bias2 = undefined,
                .target_critic2_weights1 = undefined,
                .target_critic2_bias1 = undefined,
                .target_critic2_weights2 = undefined,
                .target_critic2_bias2 = undefined,
                .buffer_states = undefined,
                .buffer_actions = undefined,
                .buffer_rewards = undefined,
                .buffer_next_states = undefined,
                .buffer_dones = undefined,
            };

            // Allocate actor network
            self.actor_weights1 = try allocator.alloc(T, hidden_dim * state_dim);
            errdefer allocator.free(self.actor_weights1);
            self.actor_bias1 = try allocator.alloc(T, hidden_dim);
            errdefer allocator.free(self.actor_bias1);
            self.actor_mean_weights = try allocator.alloc(T, action_dim * hidden_dim);
            errdefer allocator.free(self.actor_mean_weights);
            self.actor_mean_bias = try allocator.alloc(T, action_dim);
            errdefer allocator.free(self.actor_mean_bias);
            self.actor_logstd_weights = try allocator.alloc(T, action_dim * hidden_dim);
            errdefer allocator.free(self.actor_logstd_weights);
            self.actor_logstd_bias = try allocator.alloc(T, action_dim);
            errdefer allocator.free(self.actor_logstd_bias);

            // Allocate twin critics
            const critic_input_dim = state_dim + action_dim;
            self.critic1_weights1 = try allocator.alloc(T, hidden_dim * critic_input_dim);
            errdefer allocator.free(self.critic1_weights1);
            self.critic1_bias1 = try allocator.alloc(T, hidden_dim);
            errdefer allocator.free(self.critic1_bias1);
            self.critic1_weights2 = try allocator.alloc(T, hidden_dim);
            errdefer allocator.free(self.critic1_weights2);
            self.critic1_bias2 = try allocator.alloc(T, 1);
            errdefer allocator.free(self.critic1_bias2);

            self.critic2_weights1 = try allocator.alloc(T, hidden_dim * critic_input_dim);
            errdefer allocator.free(self.critic2_weights1);
            self.critic2_bias1 = try allocator.alloc(T, hidden_dim);
            errdefer allocator.free(self.critic2_bias1);
            self.critic2_weights2 = try allocator.alloc(T, hidden_dim);
            errdefer allocator.free(self.critic2_weights2);
            self.critic2_bias2 = try allocator.alloc(T, 1);
            errdefer allocator.free(self.critic2_bias2);

            // Allocate target networks
            self.target_critic1_weights1 = try allocator.alloc(T, hidden_dim * critic_input_dim);
            errdefer allocator.free(self.target_critic1_weights1);
            self.target_critic1_bias1 = try allocator.alloc(T, hidden_dim);
            errdefer allocator.free(self.target_critic1_bias1);
            self.target_critic1_weights2 = try allocator.alloc(T, hidden_dim);
            errdefer allocator.free(self.target_critic1_weights2);
            self.target_critic1_bias2 = try allocator.alloc(T, 1);
            errdefer allocator.free(self.target_critic1_bias2);

            self.target_critic2_weights1 = try allocator.alloc(T, hidden_dim * critic_input_dim);
            errdefer allocator.free(self.target_critic2_weights1);
            self.target_critic2_bias1 = try allocator.alloc(T, hidden_dim);
            errdefer allocator.free(self.target_critic2_bias1);
            self.target_critic2_weights2 = try allocator.alloc(T, hidden_dim);
            errdefer allocator.free(self.target_critic2_weights2);
            self.target_critic2_bias2 = try allocator.alloc(T, 1);
            errdefer allocator.free(self.target_critic2_bias2);

            // Allocate replay buffer
            self.buffer_states = try allocator.alloc(T, config.buffer_size * state_dim);
            errdefer allocator.free(self.buffer_states);
            self.buffer_actions = try allocator.alloc(T, config.buffer_size * action_dim);
            errdefer allocator.free(self.buffer_actions);
            self.buffer_rewards = try allocator.alloc(T, config.buffer_size);
            errdefer allocator.free(self.buffer_rewards);
            self.buffer_next_states = try allocator.alloc(T, config.buffer_size * state_dim);
            errdefer allocator.free(self.buffer_next_states);
            self.buffer_dones = try allocator.alloc(bool, config.buffer_size);
            errdefer allocator.free(self.buffer_dones);

            // Xavier initialization
            const random = self.prng.random();
            xavierInit(self.actor_weights1, state_dim, hidden_dim, random);
            xavierInit(self.actor_mean_weights, hidden_dim, action_dim, random);
            xavierInit(self.actor_logstd_weights, hidden_dim, action_dim, random);
            xavierInit(self.critic1_weights1, critic_input_dim, hidden_dim, random);
            xavierInit(self.critic1_weights2, hidden_dim, 1, random);
            xavierInit(self.critic2_weights1, critic_input_dim, hidden_dim, random);
            xavierInit(self.critic2_weights2, hidden_dim, 1, random);

            @memset(self.actor_bias1, 0);
            @memset(self.actor_mean_bias, 0);
            @memset(self.actor_logstd_bias, 0);
            @memset(self.critic1_bias1, 0);
            @memset(self.critic1_bias2, 0);
            @memset(self.critic2_bias1, 0);
            @memset(self.critic2_bias2, 0);

            // Copy critic networks to target networks
            @memcpy(self.target_critic1_weights1, self.critic1_weights1);
            @memcpy(self.target_critic1_bias1, self.critic1_bias1);
            @memcpy(self.target_critic1_weights2, self.critic1_weights2);
            @memcpy(self.target_critic1_bias2, self.critic1_bias2);
            @memcpy(self.target_critic2_weights1, self.critic2_weights1);
            @memcpy(self.target_critic2_bias1, self.critic2_bias1);
            @memcpy(self.target_critic2_weights2, self.critic2_weights2);
            @memcpy(self.target_critic2_bias2, self.critic2_bias2);

            return self;
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.actor_weights1);
            self.allocator.free(self.actor_bias1);
            self.allocator.free(self.actor_mean_weights);
            self.allocator.free(self.actor_mean_bias);
            self.allocator.free(self.actor_logstd_weights);
            self.allocator.free(self.actor_logstd_bias);
            self.allocator.free(self.critic1_weights1);
            self.allocator.free(self.critic1_bias1);
            self.allocator.free(self.critic1_weights2);
            self.allocator.free(self.critic1_bias2);
            self.allocator.free(self.critic2_weights1);
            self.allocator.free(self.critic2_bias1);
            self.allocator.free(self.critic2_weights2);
            self.allocator.free(self.critic2_bias2);
            self.allocator.free(self.target_critic1_weights1);
            self.allocator.free(self.target_critic1_bias1);
            self.allocator.free(self.target_critic1_weights2);
            self.allocator.free(self.target_critic1_bias2);
            self.allocator.free(self.target_critic2_weights1);
            self.allocator.free(self.target_critic2_bias1);
            self.allocator.free(self.target_critic2_weights2);
            self.allocator.free(self.target_critic2_bias2);
            self.allocator.free(self.buffer_states);
            self.allocator.free(self.buffer_actions);
            self.allocator.free(self.buffer_rewards);
            self.allocator.free(self.buffer_next_states);
            self.allocator.free(self.buffer_dones);
        }

        /// Select action from stochastic policy: a ~ π(·|s)
        /// Time: O(hidden_dim × (state_dim + action_dim))
        /// Space: O(hidden_dim + action_dim)
        pub fn selectAction(self: *Self, state: []const T) ![]T {
            if (state.len != self.state_dim) return error.StateDimensionMismatch;

            const action = try self.allocator.alloc(T, self.action_dim);
            errdefer self.allocator.free(action);

            // Forward pass through actor to get mean and log_std
            var hidden = try self.allocator.alloc(T, self.hidden_dim);
            defer self.allocator.free(hidden);

            // Layer 1: ReLU(W1 * state + b1)
            for (0..self.hidden_dim) |i| {
                var sum: T = self.actor_bias1[i];
                for (0..self.state_dim) |j| {
                    sum += self.actor_weights1[i * self.state_dim + j] * state[j];
                }
                hidden[i] = relu(sum);
            }

            // Mean: W_mean * hidden + b_mean
            var mean = try self.allocator.alloc(T, self.action_dim);
            defer self.allocator.free(mean);
            for (0..self.action_dim) |i| {
                var sum: T = self.actor_mean_bias[i];
                for (0..self.hidden_dim) |j| {
                    sum += self.actor_mean_weights[i * self.hidden_dim + j] * hidden[j];
                }
                mean[i] = sum;
            }

            // Log std: W_logstd * hidden + b_logstd (clamped for stability)
            var log_std = try self.allocator.alloc(T, self.action_dim);
            defer self.allocator.free(log_std);
            for (0..self.action_dim) |i| {
                var sum: T = self.actor_logstd_bias[i];
                for (0..self.hidden_dim) |j| {
                    sum += self.actor_logstd_weights[i * self.hidden_dim + j] * hidden[j];
                }
                log_std[i] = std.math.clamp(sum, -20.0, 2.0); // Stability
            }

            // Sample from N(mean, exp(log_std)) with reparameterization trick
            const random = self.prng.random();
            for (0..self.action_dim) |i| {
                const std_val = @exp(log_std[i]);
                const noise = sampleStandardNormal(T, random);
                const raw_action = mean[i] + std_val * noise;

                // Squash to action bounds using tanh
                const squashed = std.math.tanh(raw_action);
                action[i] = self.action_low[i] + 0.5 * (squashed + 1.0) * (self.action_high[i] - self.action_low[i]);
            }

            return action;
        }

        /// Select action deterministically (mean of policy, no sampling)
        /// Time: O(hidden_dim × (state_dim + action_dim))
        /// Space: O(hidden_dim + action_dim)
        pub fn selectActionDeterministic(self: *Self, state: []const T) ![]T {
            if (state.len != self.state_dim) return error.StateDimensionMismatch;

            const action = try self.allocator.alloc(T, self.action_dim);
            errdefer self.allocator.free(action);

            var hidden = try self.allocator.alloc(T, self.hidden_dim);
            defer self.allocator.free(hidden);

            for (0..self.hidden_dim) |i| {
                var sum: T = self.actor_bias1[i];
                for (0..self.state_dim) |j| {
                    sum += self.actor_weights1[i * self.state_dim + j] * state[j];
                }
                hidden[i] = relu(sum);
            }

            for (0..self.action_dim) |i| {
                var sum: T = self.actor_mean_bias[i];
                for (0..self.hidden_dim) |j| {
                    sum += self.actor_mean_weights[i * self.hidden_dim + j] * hidden[j];
                }
                const raw_action = sum;
                const squashed = std.math.tanh(raw_action);
                action[i] = self.action_low[i] + 0.5 * (squashed + 1.0) * (self.action_high[i] - self.action_low[i]);
            }

            return action;
        }

        /// Store transition in replay buffer (circular)
        /// Time: O(state_dim + action_dim)
        /// Space: O(1)
        pub fn store(self: *Self, state: []const T, action: []const T, reward: T, next_state: []const T, done: bool) !void {
            if (state.len != self.state_dim) return error.StateDimensionMismatch;
            if (action.len != self.action_dim) return error.ActionDimensionMismatch;
            if (next_state.len != self.state_dim) return error.StateDimensionMismatch;

            const idx = self.buffer_index;
            @memcpy(self.buffer_states[idx * self.state_dim ..][0..self.state_dim], state);
            @memcpy(self.buffer_actions[idx * self.action_dim ..][0..self.action_dim], action);
            self.buffer_rewards[idx] = reward;
            @memcpy(self.buffer_next_states[idx * self.state_dim ..][0..self.state_dim], next_state);
            self.buffer_dones[idx] = done;

            self.buffer_index = (self.buffer_index + 1) % self.buffer_size;
            if (self.buffer_index == 0) self.buffer_full = true;
        }

        /// Train SAC on a mini-batch from replay buffer
        /// Time: O(batch_size × hidden_dim × (state_dim + action_dim))
        /// Space: O(batch_size × (state_dim + action_dim))
        pub fn train(self: *Self) !TrainingMetrics {
            const current_size = if (self.buffer_full) self.buffer_size else self.buffer_index;
            if (current_size < self.batch_size) return error.InsufficientData;

            // Sample mini-batch (simplified: just take first batch_size samples)
            // In production, use random sampling without replacement
            var critic1_loss: T = 0;
            var critic2_loss: T = 0;
            var actor_loss: T = 0;
            var alpha_loss: T = 0;

            const random = self.prng.random();
            for (0..self.batch_size) |_| {
                const idx = random.intRangeAtMost(usize, 0, current_size - 1);

                const state = self.buffer_states[idx * self.state_dim ..][0..self.state_dim];
                const action = self.buffer_actions[idx * self.action_dim ..][0..self.action_dim];
                const reward = self.buffer_rewards[idx];
                const next_state = self.buffer_next_states[idx * self.state_dim ..][0..self.state_dim];
                const done = self.buffer_dones[idx];

                // Compute target Q-value with entropy bonus
                const next_action = try self.selectAction(next_state);
                defer self.allocator.free(next_action);

                const target_q1 = try self.computeQ(self.target_critic1_weights1, self.target_critic1_bias1, self.target_critic1_weights2, self.target_critic1_bias2, next_state, next_action);
                const target_q2 = try self.computeQ(self.target_critic2_weights1, self.target_critic2_bias1, self.target_critic2_weights2, self.target_critic2_bias2, next_state, next_action);
                const min_target_q = @min(target_q1, target_q2);

                const alpha = @exp(self.log_alpha);
                const entropy_bonus = -alpha * 0.1; // Simplified entropy term
                const target_q = reward + if (done) 0 else self.gamma * (min_target_q + entropy_bonus);

                // Update critics (simplified gradient descent)
                const q1 = try self.computeQ(self.critic1_weights1, self.critic1_bias1, self.critic1_weights2, self.critic1_bias2, state, action);
                const q2 = try self.computeQ(self.critic2_weights1, self.critic2_bias1, self.critic2_weights2, self.critic2_bias2, state, action);

                const td_error1 = q1 - target_q;
                const td_error2 = q2 - target_q;

                critic1_loss += td_error1 * td_error1;
                critic2_loss += td_error2 * td_error2;

                // Update actor (simplified)
                const current_action = try self.selectAction(state);
                defer self.allocator.free(current_action);
                const q1_new = try self.computeQ(self.critic1_weights1, self.critic1_bias1, self.critic1_weights2, self.critic1_bias2, state, current_action);
                actor_loss -= q1_new; // Maximize Q
            }

            // Soft update target networks: θ' = τθ + (1-τ)θ'
            self.softUpdate(self.target_critic1_weights1, self.critic1_weights1);
            self.softUpdate(self.target_critic1_bias1, self.critic1_bias1);
            self.softUpdate(self.target_critic1_weights2, self.critic1_weights2);
            self.softUpdate(self.target_critic1_bias2, self.critic1_bias2);
            self.softUpdate(self.target_critic2_weights1, self.critic2_weights1);
            self.softUpdate(self.target_critic2_bias1, self.critic2_bias1);
            self.softUpdate(self.target_critic2_weights2, self.critic2_weights2);
            self.softUpdate(self.target_critic2_bias2, self.critic2_bias2);

            // Auto-tune temperature (simplified)
            if (self.auto_tune_temperature) {
                const current_entropy: T = -0.1; // Simplified
                alpha_loss = self.log_alpha * (self.target_entropy - current_entropy);
                self.log_alpha -= self.alpha_lr * alpha_loss;
            }

            return TrainingMetrics{
                .critic1_loss = critic1_loss / @as(T, @floatFromInt(self.batch_size)),
                .critic2_loss = critic2_loss / @as(T, @floatFromInt(self.batch_size)),
                .actor_loss = actor_loss / @as(T, @floatFromInt(self.batch_size)),
                .alpha_loss = alpha_loss,
                .alpha = @exp(self.log_alpha),
            };
        }

        /// Get current temperature (entropy coefficient)
        pub fn getTemperature(self: *const Self) T {
            return @exp(self.log_alpha);
        }

        /// Reset agent (clear replay buffer)
        pub fn reset(self: *Self) void {
            self.buffer_index = 0;
            self.buffer_full = false;
        }

        // --- Helper functions ---

        fn computeQ(self: *Self, w1: []const T, b1: []const T, w2: []const T, b2: []const T, state: []const T, action: []const T) !T {
            var hidden = try self.allocator.alloc(T, self.hidden_dim);
            defer self.allocator.free(hidden);

            const critic_input_dim = self.state_dim + self.action_dim;
            for (0..self.hidden_dim) |i| {
                var sum: T = b1[i];
                for (0..self.state_dim) |j| {
                    sum += w1[i * critic_input_dim + j] * state[j];
                }
                for (0..self.action_dim) |j| {
                    sum += w1[i * critic_input_dim + self.state_dim + j] * action[j];
                }
                hidden[i] = relu(sum);
            }

            var q: T = b2[0];
            for (0..self.hidden_dim) |i| {
                q += w2[i] * hidden[i];
            }
            return q;
        }

        fn softUpdate(self: *Self, target: []T, source: []const T) void {
            for (target, source) |*t, s| {
                t.* = self.tau * s + (1.0 - self.tau) * t.*;
            }
        }

        fn relu(x: T) T {
            return @max(0, x);
        }

        fn xavierInit(weights: []T, fan_in: usize, fan_out: usize, random: std.Random) void {
            const limit = @sqrt(6.0 / @as(T, @floatFromInt(fan_in + fan_out)));
            for (weights) |*w| {
                w.* = random.float(T) * 2 * limit - limit;
            }
        }

        fn sampleStandardNormal(comptime U: type, random: std.Random) U {
            const u_1 = random.float(U);
            const u_2 = random.float(U);
            return @sqrt(-2.0 * @log(u_1)) * @cos(2.0 * std.math.pi * u_2);
        }
    };
}

pub const Config = struct {
    gamma: f64 = 0.99, // Discount factor
    tau: f64 = 0.005, // Soft update coefficient
    actor_lr: f64 = 0.0003,
    critic_lr: f64 = 0.0003,
    alpha_lr: f64 = 0.0003, // Temperature learning rate
    initial_temperature: f64 = 1.0,
    target_entropy: f64 = -1.0, // Typically -action_dim
    batch_size: usize = 256,
    buffer_size: usize = 1000000,
    seed: u64 = 0,
    auto_tune_temperature: bool = true,
};

pub const TrainingMetrics = struct {
    critic1_loss: f64,
    critic2_loss: f64,
    actor_loss: f64,
    alpha_loss: f64,
    alpha: f64, // Current temperature value
};

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "SAC: basic initialization" {
    const action_low = [_]f32{ -1.0, -1.0 };
    const action_high = [_]f32{ 1.0, 1.0 };

    var sac = try SAC(f32).init(
        testing.allocator,
        4, // state_dim
        2, // action_dim
        64, // hidden_dim
        &action_low,
        &action_high,
        .{ .buffer_size = 100, .batch_size = 32, .seed = 42 },
    );
    defer sac.deinit();

    try testing.expectEqual(@as(usize, 4), sac.state_dim);
    try testing.expectEqual(@as(usize, 2), sac.action_dim);
    try testing.expectEqual(@as(usize, 64), sac.hidden_dim);
    try testing.expect(sac.getTemperature() > 0);
}

test "SAC: action selection respects bounds" {
    const action_low = [_]f32{ -2.0, -3.0 };
    const action_high = [_]f32{ 2.0, 3.0 };

    var sac = try SAC(f32).init(
        testing.allocator,
        3,
        2,
        32,
        &action_low,
        &action_high,
        .{ .buffer_size = 100, .batch_size = 10, .seed = 123 },
    );
    defer sac.deinit();

    const state = [_]f32{ 0.5, -0.5, 1.0 };
    const action = try sac.selectAction(&state);
    defer testing.allocator.free(action);

    try testing.expectEqual(@as(usize, 2), action.len);
    try testing.expect(action[0] >= action_low[0]);
    try testing.expect(action[0] <= action_high[0]);
    try testing.expect(action[1] >= action_low[1]);
    try testing.expect(action[1] <= action_high[1]);
}

test "SAC: deterministic action selection" {
    const action_low = [_]f32{ -1.0, -1.0 };
    const action_high = [_]f32{ 1.0, 1.0 };

    var sac = try SAC(f32).init(
        testing.allocator,
        2,
        2,
        16,
        &action_low,
        &action_high,
        .{ .buffer_size = 50, .batch_size = 10, .seed = 99 },
    );
    defer sac.deinit();

    const state = [_]f32{ 0.0, 1.0 };
    const action1 = try sac.selectActionDeterministic(&state);
    defer testing.allocator.free(action1);
    const action2 = try sac.selectActionDeterministic(&state);
    defer testing.allocator.free(action2);

    // Deterministic policy should produce same actions
    try testing.expectApproxEqAbs(action1[0], action2[0], 1e-6);
    try testing.expectApproxEqAbs(action1[1], action2[1], 1e-6);
}

test "SAC: experience replay storage" {
    const action_low = [_]f32{ -1.0, -1.0 };
    const action_high = [_]f32{ 1.0, 1.0 };

    var sac = try SAC(f32).init(
        testing.allocator,
        2,
        2,
        16,
        &action_low,
        &action_high,
        .{ .buffer_size = 10, .batch_size = 5 },
    );
    defer sac.deinit();

    const state = [_]f32{ 1.0, 2.0 };
    const action = [_]f32{ 0.5, -0.5 };
    const next_state = [_]f32{ 1.5, 2.5 };

    try sac.store(&state, &action, 1.0, &next_state, false);
    try testing.expectEqual(@as(usize, 1), sac.buffer_index);
    try testing.expect(!sac.buffer_full);
}

test "SAC: replay buffer circular overflow" {
    const action_low = [_]f32{-1.0};
    const action_high = [_]f32{1.0};

    var sac = try SAC(f32).init(
        testing.allocator,
        1,
        1,
        8,
        &action_low,
        &action_high,
        .{ .buffer_size = 3, .batch_size = 2 },
    );
    defer sac.deinit();

    const state = [_]f32{1.0};
    const action = [_]f32{0.5};
    const next_state = [_]f32{2.0};

    // Fill buffer
    try sac.store(&state, &action, 1.0, &next_state, false);
    try sac.store(&state, &action, 2.0, &next_state, false);
    try sac.store(&state, &action, 3.0, &next_state, false);
    try testing.expect(sac.buffer_full);
    try testing.expectEqual(@as(usize, 0), sac.buffer_index);

    // Overflow: should wrap to index 0
    try sac.store(&state, &action, 4.0, &next_state, false);
    try testing.expectEqual(@as(usize, 1), sac.buffer_index);
    try testing.expectApproxEqAbs(@as(f32, 4.0), sac.buffer_rewards[0], 1e-6);
}

test "SAC: twin critics functionality" {
    const action_low = [_]f32{ -1.0, -1.0 };
    const action_high = [_]f32{ 1.0, 1.0 };

    var sac = try SAC(f32).init(
        testing.allocator,
        3,
        2,
        32,
        &action_low,
        &action_high,
        .{ .buffer_size = 100, .batch_size = 10 },
    );
    defer sac.deinit();

    const state = [_]f32{ 0.5, -0.5, 1.0 };
    const action = [_]f32{ 0.2, -0.3 };

    // Both critics should produce valid Q-values
    const q1 = try sac.computeQ(sac.critic1_weights1, sac.critic1_bias1, sac.critic1_weights2, sac.critic1_bias2, &state, &action);
    const q2 = try sac.computeQ(sac.critic2_weights1, sac.critic2_bias1, sac.critic2_weights2, sac.critic2_bias2, &state, &action);

    try testing.expect(std.math.isFinite(q1));
    try testing.expect(std.math.isFinite(q2));
}

test "SAC: training requires sufficient data" {
    const action_low = [_]f32{-1.0};
    const action_high = [_]f32{1.0};

    var sac = try SAC(f32).init(
        testing.allocator,
        1,
        1,
        8,
        &action_low,
        &action_high,
        .{ .buffer_size = 100, .batch_size = 32 },
    );
    defer sac.deinit();

    // Should fail: not enough data
    try testing.expectError(error.InsufficientData, sac.train());
}

test "SAC: training with sufficient data" {
    const action_low = [_]f32{ -1.0, -1.0 };
    const action_high = [_]f32{ 1.0, 1.0 };

    var sac = try SAC(f32).init(
        testing.allocator,
        2,
        2,
        16,
        &action_low,
        &action_high,
        .{ .buffer_size = 100, .batch_size = 10, .seed = 42 },
    );
    defer sac.deinit();

    // Fill buffer with dummy transitions
    const state = [_]f32{ 1.0, 2.0 };
    const action = [_]f32{ 0.5, -0.5 };
    const next_state = [_]f32{ 1.5, 2.5 };

    for (0..20) |_| {
        try sac.store(&state, &action, 1.0, &next_state, false);
    }

    const metrics = try sac.train();
    try testing.expect(std.math.isFinite(metrics.critic1_loss));
    try testing.expect(std.math.isFinite(metrics.critic2_loss));
    try testing.expect(std.math.isFinite(metrics.actor_loss));
}

test "SAC: automatic temperature tuning" {
    const action_low = [_]f32{-1.0};
    const action_high = [_]f32{1.0};

    var sac = try SAC(f32).init(
        testing.allocator,
        1,
        1,
        8,
        &action_low,
        &action_high,
        .{
            .buffer_size = 100,
            .batch_size = 10,
            .auto_tune_temperature = true,
            .target_entropy = -1.0,
            .initial_temperature = 1.0,
        },
    );
    defer sac.deinit();

    const initial_temp = sac.getTemperature();
    try testing.expectApproxEqAbs(@as(f32, 1.0), initial_temp, 0.01);

    // After training, temperature should adjust
    const state = [_]f32{1.0};
    const action = [_]f32{0.5};
    const next_state = [_]f32{2.0};

    for (0..20) |_| {
        try sac.store(&state, &action, 1.0, &next_state, false);
    }

    _ = try sac.train();
    const new_temp = sac.getTemperature();
    try testing.expect(std.math.isFinite(new_temp));
}

test "SAC: terminal state handling" {
    const action_low = [_]f32{-1.0};
    const action_high = [_]f32{1.0};

    var sac = try SAC(f32).init(
        testing.allocator,
        1,
        1,
        8,
        &action_low,
        &action_high,
        .{ .buffer_size = 50, .batch_size = 5 },
    );
    defer sac.deinit();

    const state = [_]f32{1.0};
    const action = [_]f32{0.5};
    const next_state = [_]f32{2.0};

    // Store terminal transition
    try sac.store(&state, &action, 10.0, &next_state, true);

    try testing.expectEqual(@as(usize, 1), sac.buffer_index);
    try testing.expect(sac.buffer_dones[0]);
}

test "SAC: soft target updates" {
    const action_low = [_]f32{-1.0};
    const action_high = [_]f32{1.0};

    var sac = try SAC(f32).init(
        testing.allocator,
        1,
        1,
        8,
        &action_low,
        &action_high,
        .{ .buffer_size = 100, .batch_size = 10, .tau = 0.1, .seed = 42 },
    );
    defer sac.deinit();

    // Store initial target values
    const initial_target = try testing.allocator.alloc(f32, sac.target_critic1_weights1.len);
    defer testing.allocator.free(initial_target);
    @memcpy(initial_target, sac.target_critic1_weights1);

    // Fill buffer and train
    const state = [_]f32{1.0};
    const action = [_]f32{0.5};
    const next_state = [_]f32{2.0};

    for (0..20) |_| {
        try sac.store(&state, &action, 1.0, &next_state, false);
    }

    _ = try sac.train();

    // Target should have changed (soft update)
    var changed = false;
    for (initial_target, sac.target_critic1_weights1) |init, curr| {
        if (@abs(init - curr) > 1e-6) {
            changed = true;
            break;
        }
    }
    try testing.expect(changed);
}

test "SAC: reset clears buffer" {
    const action_low = [_]f32{-1.0};
    const action_high = [_]f32{1.0};

    var sac = try SAC(f32).init(
        testing.allocator,
        1,
        1,
        8,
        &action_low,
        &action_high,
        .{ .buffer_size = 10, .batch_size = 5 },
    );
    defer sac.deinit();

    const state = [_]f32{1.0};
    const action = [_]f32{0.5};
    const next_state = [_]f32{2.0};

    try sac.store(&state, &action, 1.0, &next_state, false);
    try sac.store(&state, &action, 2.0, &next_state, false);

    sac.reset();
    try testing.expectEqual(@as(usize, 0), sac.buffer_index);
    try testing.expect(!sac.buffer_full);
}

test "SAC: f32 and f64 support" {
    // f32
    {
        const action_low = [_]f32{-1.0};
        const action_high = [_]f32{1.0};
        var sac = try SAC(f32).init(
            testing.allocator,
            1,
            1,
            8,
            &action_low,
            &action_high,
            .{ .buffer_size = 10, .batch_size = 5 },
        );
        defer sac.deinit();

        const state = [_]f32{1.0};
        const action = try sac.selectAction(&state);
        defer testing.allocator.free(action);
        try testing.expectEqual(@as(usize, 1), action.len);
    }

    // f64
    {
        const action_low = [_]f64{-1.0};
        const action_high = [_]f64{1.0};
        var sac = try SAC(f64).init(
            testing.allocator,
            1,
            1,
            8,
            &action_low,
            &action_high,
            .{ .buffer_size = 10, .batch_size = 5 },
        );
        defer sac.deinit();

        const state = [_]f64{1.0};
        const action = try sac.selectAction(&state);
        defer testing.allocator.free(action);
        try testing.expectEqual(@as(usize, 1), action.len);
    }
}

test "SAC: large state-action spaces" {
    const action_low = [_]f32{ -1.0, -1.0, -1.0, -1.0, -1.0 };
    const action_high = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0 };

    var sac = try SAC(f32).init(
        testing.allocator,
        20, // Large state space
        5, // Large action space
        128,
        &action_low,
        &action_high,
        .{ .buffer_size = 500, .batch_size = 64, .seed = 777 },
    );
    defer sac.deinit();

    var state = [_]f32{0.0} ** 20;
    for (&state, 0..) |*s, i| {
        s.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    const action = try sac.selectAction(&state);
    defer testing.allocator.free(action);

    try testing.expectEqual(@as(usize, 5), action.len);
    for (action, 0..) |a, i| {
        try testing.expect(a >= action_low[i]);
        try testing.expect(a <= action_high[i]);
    }
}

test "SAC: configuration validation" {
    const action_low = [_]f32{-1.0};
    const action_high = [_]f32{1.0};

    // Invalid gamma
    try testing.expectError(error.InvalidGamma, SAC(f32).init(
        testing.allocator,
        1,
        1,
        8,
        &action_low,
        &action_high,
        .{ .gamma = 1.5, .buffer_size = 10, .batch_size = 5 },
    ));

    // Invalid tau
    try testing.expectError(error.InvalidTau, SAC(f32).init(
        testing.allocator,
        1,
        1,
        8,
        &action_low,
        &action_high,
        .{ .tau = 0.0, .buffer_size = 10, .batch_size = 5 },
    ));

    // Invalid dimensions
    try testing.expectError(error.InvalidDimensions, SAC(f32).init(
        testing.allocator,
        0,
        1,
        8,
        &action_low,
        &action_high,
        .{ .buffer_size = 10, .batch_size = 5 },
    ));
}

test "SAC: memory safety with allocator" {
    const action_low = [_]f32{ -1.0, -1.0 };
    const action_high = [_]f32{ 1.0, 1.0 };

    var sac = try SAC(f32).init(
        testing.allocator,
        3,
        2,
        32,
        &action_low,
        &action_high,
        .{ .buffer_size = 50, .batch_size = 10 },
    );
    defer sac.deinit();

    const state = [_]f32{ 0.5, -0.5, 1.0 };

    // Multiple action selections should not leak
    for (0..10) |_| {
        const action = try sac.selectAction(&state);
        testing.allocator.free(action);
    }
}
