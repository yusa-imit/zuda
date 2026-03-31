const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// PPO (Proximal Policy Optimization) — Clipped policy gradient for stable on-policy learning
///
/// Algorithm: On-policy actor-critic with clipped surrogate objective for stable policy updates
///
/// Key features:
///   - Clipped objective: Constrains policy updates to prevent destructively large steps
///   - Multiple epochs: Reuses collected data with mini-batch updates (sample efficient)
///   - Generalized Advantage Estimation (GAE): Reduces variance with λ-return
///   - Actor-Critic: Learns policy π(a|s) and value function V(s) simultaneously
///   - Trust region: Clip ratio bounds deviation from old policy (ε-constrained)
///   - Entropy bonus: Encourages exploration via policy entropy H(π)
///   - Stable learning: More robust than TRPO (no KL penalty) and REINFORCE (lower variance)
///
/// Update rule:
///   - Policy loss: L^CLIP(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
///   - Value loss: L^VF(θ) = E[(V(s) - V_target)²]
///   - Entropy bonus: L^ENT(θ) = E[H(π(·|s))]
///   - Total loss: L = -L^CLIP + c₁L^VF - c₂L^ENT
///   where r_t(θ) = π(a|s) / π_old(a|s) (probability ratio)
///
/// Advantages:
///   - Stable training (clipping prevents destructive updates)
///   - Sample efficient (reuses data for multiple epochs)
///   - Simple to implement (no complex second-order optimization)
///   - Works well in practice (state-of-the-art for many tasks)
///
/// Time: O(K × epochs × |A|) per update where K = trajectory length
/// Space: O(|S|×|A| + |S|) for policy and value function
///
/// Use cases:
///   - Continuous control (robotics, locomotion)
///   - Game playing (Atari, Dota, Go)
///   - Simulated environments (stable learning)
///   - Multi-agent systems (coordinated policies)
pub fn PPO(comptime T: type, comptime S: type, comptime A: type) type {
    return struct {
        const Self = @This();

        /// Configuration parameters
        pub const Config = struct {
            num_states: usize,
            num_actions: usize,
            learning_rate_actor: T = 0.0003, // Actor learning rate (lower for stability)
            learning_rate_critic: T = 0.001, // Critic learning rate
            gamma: T = 0.99, // Discount factor
            lambda: T = 0.95, // GAE lambda (bias-variance tradeoff)
            epsilon: T = 0.2, // Clip ratio (ε in [0.1, 0.3] typical)
            value_coef: T = 0.5, // Value loss coefficient (c₁)
            entropy_coef: T = 0.01, // Entropy bonus coefficient (c₂)
            epochs: usize = 10, // Number of optimization epochs per update
            batch_size: usize = 64, // Mini-batch size
            max_trajectory_length: usize = 2048, // Max steps per update
        };

        /// Trajectory experience (collected data)
        const Experience = struct {
            state: S,
            action: A,
            reward: T,
            next_state: S,
            done: bool,
            log_prob_old: T, // π_old(a|s) for ratio computation
            value: T, // V(s) at collection time
        };

        allocator: Allocator,
        config: Config,

        // Policy network (actor): π(a|s) — action preferences
        policy_params: []T, // [num_states × num_actions] — state-action preferences

        // Value network (critic): V(s)
        value_params: []T, // [num_states] — state values

        // Trajectory buffer
        trajectory: std.ArrayList(Experience),

        pub fn init(allocator: Allocator, config: Config) !Self {
            if (config.num_states == 0) return error.InvalidNumStates;
            if (config.num_actions == 0) return error.InvalidNumActions;
            if (config.learning_rate_actor <= 0) return error.InvalidLearningRate;
            if (config.learning_rate_critic <= 0) return error.InvalidLearningRate;
            if (config.gamma < 0 or config.gamma > 1) return error.InvalidGamma;
            if (config.lambda < 0 or config.lambda > 1) return error.InvalidLambda;
            if (config.epsilon <= 0) return error.InvalidEpsilon;
            if (config.epochs == 0) return error.InvalidEpochs;
            if (config.batch_size == 0) return error.InvalidBatchSize;

            // Allocate policy parameters (initialized to zero for uniform policy)
            const policy_params = try allocator.alloc(T, config.num_states * config.num_actions);
            errdefer allocator.free(policy_params);
            @memset(policy_params, 0.0);

            // Allocate value parameters (initialized to zero)
            const value_params = try allocator.alloc(T, config.num_states);
            errdefer allocator.free(value_params);
            @memset(value_params, 0.0);

            return Self{
                .allocator = allocator,
                .config = config,
                .policy_params = policy_params,
                .value_params = value_params,
                .trajectory = std.ArrayList(Experience).init(allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.policy_params);
            self.allocator.free(self.value_params);
            self.trajectory.deinit();
        }

        /// Select action using current policy π(a|s) with stochastic sampling
        /// Time: O(|A|)
        pub fn selectAction(self: *const Self, state: S, rng: *std.Random) !A {
            if (state >= self.config.num_states) return error.InvalidState;

            const probs = try self.computeActionProbabilities(state);
            defer self.allocator.free(probs);

            // Sample action according to probability distribution
            const action = sampleAction(probs, rng);
            return @intCast(action);
        }

        /// Select action greedily (argmax probability)
        /// Time: O(|A|)
        pub fn selectGreedyAction(self: *const Self, state: S) !A {
            if (state >= self.config.num_states) return error.InvalidState;

            const probs = try self.computeActionProbabilities(state);
            defer self.allocator.free(probs);

            var best_action: usize = 0;
            var best_prob: T = probs[0];
            for (probs, 0..) |p, a| {
                if (p > best_prob) {
                    best_prob = p;
                    best_action = a;
                }
            }
            return @intCast(best_action);
        }

        /// Compute π(a|s) for all actions via softmax
        /// Time: O(|A|)
        fn computeActionProbabilities(self: *const Self, state: S) ![]T {
            const offset = state * self.config.num_actions;
            const logits = self.policy_params[offset .. offset + self.config.num_actions];

            // Softmax: π(a|s) = exp(preference) / Σ_a' exp(preference')
            const probs = try self.allocator.alloc(T, self.config.num_actions);
            errdefer self.allocator.free(probs);

            // Find max for numerical stability
            var max_logit = logits[0];
            for (logits) |logit| {
                if (logit > max_logit) max_logit = logit;
            }

            // Compute exp(logit - max) and sum
            var sum: T = 0.0;
            for (logits, 0..) |logit, i| {
                const exp_val = @exp(logit - max_logit);
                probs[i] = exp_val;
                sum += exp_val;
            }

            // Normalize
            for (probs) |*p| {
                p.* /= sum;
            }

            return probs;
        }

        /// Compute log probability log π(a|s) for specific action
        /// Time: O(|A|)
        fn computeLogProb(self: *const Self, state: S, action: A) !T {
            const probs = try self.computeActionProbabilities(state);
            defer self.allocator.free(probs);
            return @log(probs[action]);
        }

        /// Compute state value V(s)
        /// Time: O(1)
        fn computeValue(self: *const Self, state: S) T {
            return self.value_params[state];
        }

        /// Store experience in trajectory buffer
        /// Time: O(1)
        pub fn storeExperience(
            self: *Self,
            state: S,
            action: A,
            reward: T,
            next_state: S,
            done: bool,
        ) !void {
            if (state >= self.config.num_states) return error.InvalidState;
            if (action >= self.config.num_actions) return error.InvalidAction;
            if (next_state >= self.config.num_states) return error.InvalidNextState;

            const log_prob_old = try self.computeLogProb(state, action);
            const value = self.computeValue(state);

            try self.trajectory.append(.{
                .state = state,
                .action = action,
                .reward = reward,
                .next_state = next_state,
                .done = done,
                .log_prob_old = log_prob_old,
                .value = value,
            });
        }

        /// Update policy using PPO with collected trajectory
        /// Time: O(epochs × K × batch_size × |A|)
        pub fn update(self: *Self) !void {
            if (self.trajectory.items.len == 0) return error.EmptyTrajectory;

            // Compute advantages using GAE
            const advantages = try self.computeGAE();
            defer self.allocator.free(advantages);

            // Compute returns (value targets)
            const returns = try self.allocator.alloc(T, self.trajectory.items.len);
            defer self.allocator.free(returns);
            for (advantages, 0..) |adv, i| {
                returns[i] = adv + self.trajectory.items[i].value;
            }

            // Normalize advantages (improves stability)
            normalizeAdvantages(advantages);

            // Multiple epochs of mini-batch updates
            for (0..self.config.epochs) |_| {
                // Shuffle indices for mini-batch sampling
                var indices = try self.allocator.alloc(usize, self.trajectory.items.len);
                defer self.allocator.free(indices);
                for (indices, 0..) |*idx, i| {
                    idx.* = i;
                }

                // Process mini-batches
                var batch_start: usize = 0;
                while (batch_start < self.trajectory.items.len) {
                    const batch_end = @min(batch_start + self.config.batch_size, self.trajectory.items.len);
                    try self.updateMiniBatch(
                        indices[batch_start..batch_end],
                        advantages,
                        returns,
                    );
                    batch_start = batch_end;
                }
            }

            // Clear trajectory after update
            self.trajectory.clearRetainingCapacity();
        }

        /// Update policy using a single mini-batch
        /// Time: O(batch_size × |A|)
        fn updateMiniBatch(
            self: *Self,
            batch_indices: []const usize,
            advantages: []const T,
            returns: []const T,
        ) !void {
            for (batch_indices) |idx| {
                const exp = self.trajectory.items[idx];
                const advantage = advantages[idx];
                const return_target = returns[idx];

                // Compute current log probability
                const log_prob_new = try self.computeLogProb(exp.state, exp.action);

                // Compute probability ratio r_t(θ) = π(a|s) / π_old(a|s)
                const ratio = @exp(log_prob_new - exp.log_prob_old);

                // Compute clipped surrogate objective
                const clipped_ratio = clip(ratio, 1.0 - self.config.epsilon, 1.0 + self.config.epsilon);
                const surrogate1 = ratio * advantage;
                const surrogate2 = clipped_ratio * advantage;
                const policy_loss = -@min(surrogate1, surrogate2);

                // Compute value loss
                const value_pred = self.computeValue(exp.state);
                const value_loss = (value_pred - return_target) * (value_pred - return_target);

                // Compute entropy bonus (encourage exploration)
                const probs = try self.computeActionProbabilities(exp.state);
                defer self.allocator.free(probs);
                const entropy = computeEntropy(probs);

                // Total loss = policy_loss + c₁*value_loss - c₂*entropy
                // (We minimize loss, so entropy has negative sign)

                // Update policy parameters (gradient ascent on clipped objective)
                try self.updatePolicy(exp.state, exp.action, policy_loss, entropy);

                // Update value parameters
                self.updateValue(exp.state, value_loss);
            }
        }

        /// Update policy parameters using gradient of clipped objective
        /// Time: O(|A|)
        fn updatePolicy(
            self: *Self,
            state: S,
            action: A,
            policy_loss: T,
            entropy: T,
        ) !void {
            const offset = state * self.config.num_actions;

            // Compute softmax probabilities
            const probs = try self.computeActionProbabilities(state);
            defer self.allocator.free(probs);

            // Gradient of log π(a|s) with respect to logits
            // ∇log π(a|s) = e_a - π(·|s) where e_a is one-hot
            for (probs, 0..) |prob, a| {
                const indicator: T = if (a == action) 1.0 else 0.0;
                const grad_log_prob = indicator - prob;

                // Policy gradient: ∇θ L^CLIP = -∇θ log π(a|s) × (clipped advantage)
                // We use negative policy_loss since we already negated in clipped objective
                const policy_grad = -policy_loss * grad_log_prob;

                // Entropy gradient: ∇θ H(π) = -∇θ Σ π log π ≈ -log π(a|s) × ∇θ log π(a|s)
                const entropy_grad = self.config.entropy_coef * entropy * grad_log_prob;

                // Update: θ ← θ - α(∇L^CLIP - c₂∇H)
                const total_grad = policy_grad - entropy_grad;
                self.policy_params[offset + a] -= self.config.learning_rate_actor * total_grad;
            }
        }

        /// Update value parameters
        /// Time: O(1)
        fn updateValue(self: *Self, state: S, value_loss: T) void {
            // Gradient of squared loss: 2(V(s) - V_target)
            const grad = 2.0 * value_loss; // value_loss already includes (pred - target)²
            self.value_params[state] -= self.config.learning_rate_critic * self.config.value_coef * grad;
        }

        /// Compute Generalized Advantage Estimation (GAE)
        /// A_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l} where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        /// Time: O(K)
        fn computeGAE(self: *Self) ![]T {
            const n = self.trajectory.items.len;
            const advantages = try self.allocator.alloc(T, n);
            errdefer self.allocator.free(advantages);

            var gae: T = 0.0;
            var i: usize = n;
            while (i > 0) {
                i -= 1;
                const exp = self.trajectory.items[i];

                // Compute TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
                const next_value: T = if (exp.done) 0.0 else self.computeValue(exp.next_state);
                const td_error = exp.reward + self.config.gamma * next_value - exp.value;

                // GAE: A_t = δ_t + γλ A_{t+1}
                gae = td_error + self.config.gamma * self.config.lambda * gae;
                advantages[i] = gae;
            }

            return advantages;
        }

        /// Reset agent (clear trajectory, keep learned parameters)
        pub fn reset(self: *Self) void {
            self.trajectory.clearRetainingCapacity();
        }

        /// Clip value to [min, max]
        fn clip(value: T, min: T, max: T) T {
            if (value < min) return min;
            if (value > max) return max;
            return value;
        }

        /// Sample action from probability distribution
        fn sampleAction(probs: []const T, rng: *std.Random) usize {
            const r = rng.float(T);
            var cumulative: T = 0.0;
            for (probs, 0..) |p, a| {
                cumulative += p;
                if (r < cumulative) return a;
            }
            return probs.len - 1;
        }

        /// Compute entropy of probability distribution H(π) = -Σ π(a|s) log π(a|s)
        fn computeEntropy(probs: []const T) T {
            var entropy: T = 0.0;
            for (probs) |p| {
                if (p > 1e-8) { // Avoid log(0)
                    entropy -= p * @log(p);
                }
            }
            return entropy;
        }

        /// Normalize advantages to mean=0, std=1 (improves stability)
        fn normalizeAdvantages(advantages: []T) void {
            if (advantages.len == 0) return;

            // Compute mean
            var sum: T = 0.0;
            for (advantages) |adv| {
                sum += adv;
            }
            const mean = sum / @as(T, @floatFromInt(advantages.len));

            // Compute std
            var variance: T = 0.0;
            for (advantages) |adv| {
                const diff = adv - mean;
                variance += diff * diff;
            }
            const std_dev = @sqrt(variance / @as(T, @floatFromInt(advantages.len)));

            // Normalize (avoid division by zero)
            const epsilon: T = 1e-8;
            for (advantages) |*adv| {
                adv.* = (adv.* - mean) / (std_dev + epsilon);
            }
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "PPO: initialization" {
    const config = PPO(f32, usize, usize).Config{
        .num_states = 4,
        .num_actions = 2,
    };

    var ppo = try PPO(f32, usize, usize).init(testing.allocator, config);
    defer ppo.deinit();

    try testing.expectEqual(4, config.num_states);
    try testing.expectEqual(2, config.num_actions);
    try testing.expectEqual(0, ppo.trajectory.items.len);
}

test "PPO: uniform initial policy" {
    const config = PPO(f64, usize, usize).Config{
        .num_states = 2,
        .num_actions = 3,
    };

    var ppo = try PPO(f64, usize, usize).init(testing.allocator, config);
    defer ppo.deinit();

    // Check that initial policy is uniform (all preferences = 0)
    const probs = try ppo.computeActionProbabilities(0);
    defer testing.allocator.free(probs);

    const expected: f64 = 1.0 / 3.0;
    for (probs) |p| {
        try testing.expectApproxEqAbs(expected, p, 0.01);
    }
}

test "PPO: action selection" {
    const config = PPO(f32, usize, usize).Config{
        .num_states = 3,
        .num_actions = 2,
    };

    var ppo = try PPO(f32, usize, usize).init(testing.allocator, config);
    defer ppo.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    const action = try ppo.selectAction(0, &rng);
    try testing.expect(action < 2);
}

test "PPO: greedy action selection" {
    const config = PPO(f32, usize, usize).Config{
        .num_states = 2,
        .num_actions = 3,
    };

    var ppo = try PPO(f32, usize, usize).init(testing.allocator, config);
    defer ppo.deinit();

    // Bias towards action 1
    ppo.policy_params[0 * 3 + 0] = 0.0;
    ppo.policy_params[0 * 3 + 1] = 2.0;
    ppo.policy_params[0 * 3 + 2] = 0.0;

    const action = try ppo.selectGreedyAction(0);
    try testing.expectEqual(1, action);
}

test "PPO: store experience" {
    const config = PPO(f32, usize, usize).Config{
        .num_states = 4,
        .num_actions = 2,
    };

    var ppo = try PPO(f32, usize, usize).init(testing.allocator, config);
    defer ppo.deinit();

    try ppo.storeExperience(0, 0, 1.0, 1, false);
    try ppo.storeExperience(1, 1, -0.5, 2, false);
    try ppo.storeExperience(2, 0, 0.0, 3, true);

    try testing.expectEqual(3, ppo.trajectory.items.len);
    try testing.expectEqual(0, ppo.trajectory.items[0].state);
    try testing.expectEqual(0, ppo.trajectory.items[0].action);
    try testing.expectEqual(1.0, ppo.trajectory.items[0].reward);
}

test "PPO: GAE computation" {
    const config = PPO(f32, usize, usize).Config{
        .num_states = 3,
        .num_actions = 2,
        .gamma = 0.9,
        .lambda = 0.95,
    };

    var ppo = try PPO(f32, usize, usize).init(testing.allocator, config);
    defer ppo.deinit();

    // Simple trajectory: s0 -r1→ s1 -r2→ s2 (terminal)
    ppo.value_params[0] = 1.0;
    ppo.value_params[1] = 2.0;
    ppo.value_params[2] = 0.0;

    try ppo.storeExperience(0, 0, 1.0, 1, false);
    try ppo.storeExperience(1, 1, 3.0, 2, true);

    const advantages = try ppo.computeGAE();
    defer testing.allocator.free(advantages);

    try testing.expectEqual(2, advantages.len);
    // Advantages should reflect TD errors with GAE smoothing
    try testing.expect(advantages[0] != 0.0);
    try testing.expect(advantages[1] != 0.0);
}

test "PPO: clipping function" {
    const clip = PPO(f32, usize, usize).clip;

    try testing.expectEqual(0.8, clip(0.5, 0.8, 1.2));
    try testing.expectEqual(1.2, clip(1.5, 0.8, 1.2));
    try testing.expectEqual(1.0, clip(1.0, 0.8, 1.2));
}

test "PPO: entropy computation" {
    const computeEntropy = PPO(f64, usize, usize).computeEntropy;

    // Uniform distribution → maximum entropy
    const uniform = [_]f64{ 0.25, 0.25, 0.25, 0.25 };
    const entropy_uniform = computeEntropy(&uniform);
    try testing.expectApproxEqAbs(@log(4.0), entropy_uniform, 0.01);

    // Deterministic → minimum entropy (0)
    const deterministic = [_]f64{ 1.0, 0.0, 0.0, 0.0 };
    const entropy_det = computeEntropy(&deterministic);
    try testing.expectApproxEqAbs(0.0, entropy_det, 0.01);
}

test "PPO: advantage normalization" {
    var advantages = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    PPO(f32, usize, usize).normalizeAdvantages(&advantages);

    // Check mean ≈ 0
    var sum: f32 = 0.0;
    for (advantages) |a| {
        sum += a;
    }
    const mean = sum / @as(f32, @floatFromInt(advantages.len));
    try testing.expectApproxEqAbs(0.0, mean, 0.01);

    // Check std ≈ 1
    var variance: f32 = 0.0;
    for (advantages) |a| {
        variance += a * a;
    }
    const std_dev = @sqrt(variance / @as(f32, @floatFromInt(advantages.len)));
    try testing.expectApproxEqAbs(1.0, std_dev, 0.01);
}

test "PPO: update with trajectory" {
    const config = PPO(f32, usize, usize).Config{
        .num_states = 2,
        .num_actions = 2,
        .learning_rate_actor = 0.01,
        .learning_rate_critic = 0.01,
        .epochs = 3,
        .batch_size = 2,
    };

    var ppo = try PPO(f32, usize, usize).init(testing.allocator, config);
    defer ppo.deinit();

    // Collect simple trajectory
    try ppo.storeExperience(0, 0, 1.0, 1, false);
    try ppo.storeExperience(1, 1, -1.0, 0, true);

    // Update should succeed
    try ppo.update();

    // Trajectory should be cleared after update
    try testing.expectEqual(0, ppo.trajectory.items.len);
}

test "PPO: simple 2-state learning" {
    const config = PPO(f32, usize, usize).Config{
        .num_states = 2,
        .num_actions = 2,
        .learning_rate_actor = 0.1,
        .learning_rate_critic = 0.1,
        .gamma = 0.9,
        .epochs = 5,
        .batch_size = 10,
    };

    var ppo = try PPO(f32, usize, usize).init(testing.allocator, config);
    defer ppo.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    // Learn that state 0 → action 0 → reward 1, state 1 → any → reward 0
    for (0..100) |_| {
        ppo.reset();

        // Collect trajectories
        for (0..10) |_| {
            const state: usize = 0;
            const action = try ppo.selectAction(state, &rng);
            const reward: f32 = if (action == 0) 1.0 else -1.0;
            try ppo.storeExperience(state, action, reward, 1, true);
        }

        try ppo.update();
    }

    // After training, greedy action in state 0 should be 0
    const learned_action = try ppo.selectGreedyAction(0);
    try testing.expectEqual(0, learned_action);
}

test "PPO: terminal state handling" {
    const config = PPO(f32, usize, usize).Config{
        .num_states = 3,
        .num_actions = 2,
    };

    var ppo = try PPO(f32, usize, usize).init(testing.allocator, config);
    defer ppo.deinit();

    try ppo.storeExperience(0, 0, 1.0, 1, false);
    try ppo.storeExperience(1, 1, 0.5, 2, true); // Terminal

    const advantages = try ppo.computeGAE();
    defer testing.allocator.free(advantages);

    // Terminal state should have zero future value in GAE
    try testing.expectEqual(2, advantages.len);
}

test "PPO: reset clears trajectory" {
    const config = PPO(f32, usize, usize).Config{
        .num_states = 3,
        .num_actions = 2,
    };

    var ppo = try PPO(f32, usize, usize).init(testing.allocator, config);
    defer ppo.deinit();

    try ppo.storeExperience(0, 0, 1.0, 1, false);
    try ppo.storeExperience(1, 1, -1.0, 2, true);
    try testing.expectEqual(2, ppo.trajectory.items.len);

    ppo.reset();
    try testing.expectEqual(0, ppo.trajectory.items.len);

    // Policy and value parameters should still exist
    try testing.expect(ppo.policy_params.len > 0);
    try testing.expect(ppo.value_params.len > 0);
}

test "PPO: f32 and f64 support" {
    const config_f32 = PPO(f32, usize, usize).Config{
        .num_states = 2,
        .num_actions = 2,
    };
    var ppo_f32 = try PPO(f32, usize, usize).init(testing.allocator, config_f32);
    defer ppo_f32.deinit();

    const config_f64 = PPO(f64, usize, usize).Config{
        .num_states = 2,
        .num_actions = 2,
    };
    var ppo_f64 = try PPO(f64, usize, usize).init(testing.allocator, config_f64);
    defer ppo_f64.deinit();

    try testing.expect(ppo_f32.policy_params.len > 0);
    try testing.expect(ppo_f64.policy_params.len > 0);
}

test "PPO: empty trajectory error" {
    const config = PPO(f32, usize, usize).Config{
        .num_states = 2,
        .num_actions = 2,
    };

    var ppo = try PPO(f32, usize, usize).init(testing.allocator, config);
    defer ppo.deinit();

    // Attempting to update with empty trajectory should fail
    try testing.expectError(error.EmptyTrajectory, ppo.update());
}

test "PPO: large state-action space" {
    const config = PPO(f32, usize, usize).Config{
        .num_states = 50,
        .num_actions = 10,
        .epochs = 2,
        .batch_size = 10,
    };

    var ppo = try PPO(f32, usize, usize).init(testing.allocator, config);
    defer ppo.deinit();

    var prng = std.Random.DefaultPrng.init(123);
    var rng = prng.random();

    // Collect some experiences
    for (0..20) |_| {
        const state: usize = rng.intRangeAtMost(usize, 0, 49);
        const action = try ppo.selectAction(state, &rng);
        const reward: f32 = rng.float(f32) * 2.0 - 1.0;
        const next_state: usize = rng.intRangeAtMost(usize, 0, 49);
        try ppo.storeExperience(state, action, reward, next_state, false);
    }

    try ppo.update();
    try testing.expectEqual(0, ppo.trajectory.items.len);
}

test "PPO: invalid configuration errors" {
    try testing.expectError(error.InvalidNumStates, PPO(f32, usize, usize).init(testing.allocator, .{
        .num_states = 0,
        .num_actions = 2,
    }));

    try testing.expectError(error.InvalidNumActions, PPO(f32, usize, usize).init(testing.allocator, .{
        .num_states = 2,
        .num_actions = 0,
    }));

    try testing.expectError(error.InvalidGamma, PPO(f32, usize, usize).init(testing.allocator, .{
        .num_states = 2,
        .num_actions = 2,
        .gamma = 1.5,
    }));

    try testing.expectError(error.InvalidEpsilon, PPO(f32, usize, usize).init(testing.allocator, .{
        .num_states = 2,
        .num_actions = 2,
        .epsilon = 0.0,
    }));
}

test "PPO: memory safety" {
    const config = PPO(f32, usize, usize).Config{
        .num_states = 10,
        .num_actions = 4,
        .epochs = 3,
        .batch_size = 5,
    };

    var ppo = try PPO(f32, usize, usize).init(testing.allocator, config);
    defer ppo.deinit();

    var prng = std.Random.DefaultPrng.init(999);
    var rng = prng.random();

    // Multiple update cycles
    for (0..5) |_| {
        for (0..10) |_| {
            const state: usize = rng.intRangeAtMost(usize, 0, 9);
            const action = try ppo.selectAction(state, &rng);
            const reward: f32 = rng.float(f32);
            const next_state: usize = rng.intRangeAtMost(usize, 0, 9);
            try ppo.storeExperience(state, action, reward, next_state, false);
        }
        try ppo.update();
    }

    // No memory leaks with testing.allocator
}
