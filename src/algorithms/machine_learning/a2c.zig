/// A2C (Advantage Actor-Critic)
///
/// Synchronous advantage actor-critic algorithm that improves upon basic actor-critic
/// by explicitly using advantage function A(s,a) = Q(s,a) - V(s) and supporting
/// n-step returns for better bias-variance tradeoff.
///
/// **Algorithm Overview**:
/// A2C extends basic actor-critic with:
/// - Advantage function: Explicitly computes A(s,a) = Q(s,a) - V(s) for policy updates
/// - n-step returns: Uses n-step TD targets for reduced bias (configurable n)
/// - Entropy regularization: Adds policy entropy H(π) to encourage exploration
/// - Value loss: Trains critic with MSE between V(s) and n-step returns
/// - Policy loss: Updates actor using advantage-weighted log-probabilities
///
/// **Key Features**:
/// - Synchronous updates: Single-worker version (multi-worker extension: A3C)
/// - n-step bootstrapping: Balances bias (n→∞: Monte Carlo) vs variance (n=1: TD)
/// - Entropy bonus: Prevents premature convergence to deterministic policy
/// - Value function baseline: Reduces gradient variance compared to REINFORCE
/// - Foundation for A3C: Asynchronous version uses parallel workers
///
/// **Update Rules**:
/// ```
/// Advantage: A(s,a) = Σ(γ^i * r_{t+i}) + γ^n * V(s_{t+n}) - V(s_t)
/// Actor:  θ ← θ + α_actor * A(s,a) * ∇log π(a|s)
/// Critic: w ← w - α_critic * ∇[(R - V(s))²]
/// Entropy: θ ← θ + β * ∇H(π)
/// ```
///
/// **Comparison**:
/// - vs Actor-Critic: Explicit advantage, n-step returns, entropy regularization
/// - vs REINFORCE: Lower variance via value baseline, continuous learning
/// - vs PPO: On-policy but no clipping (PPO more stable for large updates)
/// - vs A3C: Synchronous (A3C = asynchronous parallel workers)
///
/// **Time Complexity**: O(|A|) per update (action space size)
/// **Space Complexity**: O(|S| + |S|×|A|) for value function + policy tables
///
/// **Use Cases**:
/// - Continuous learning environments (robotics, game playing)
/// - Sample-efficient on-policy learning
/// - Foundation for distributed RL (A3C extension)
/// - Research baseline for policy gradient methods

const std = @import("std");
const Allocator = std.mem.Allocator;

/// A2C configuration
pub const Config = struct {
    /// Number of states in environment
    n_states: usize,
    /// Number of actions available
    n_actions: usize,
    /// Actor learning rate (policy updates)
    alpha_actor: f64 = 0.001,
    /// Critic learning rate (value updates)
    alpha_critic: f64 = 0.01,
    /// Discount factor for future rewards (0.0 to 1.0)
    gamma: f64 = 0.99,
    /// Number of steps for n-step returns (1 = TD(0), ∞ = Monte Carlo)
    n_steps: usize = 5,
    /// Entropy coefficient for exploration (β in entropy bonus)
    entropy_coeff: f64 = 0.01,
    /// Initial policy temperature for exploration (higher = more random)
    temperature: f64 = 1.0,
    /// Minimum temperature (prevents collapse to deterministic)
    min_temperature: f64 = 0.1,
    /// Temperature decay per episode (exponential annealing)
    temperature_decay: f64 = 0.995,

    pub fn validate(self: Config) !void {
        if (self.n_states == 0) return error.InvalidNStates;
        if (self.n_actions == 0) return error.InvalidNActions;
        if (self.alpha_actor <= 0.0 or self.alpha_actor > 1.0) return error.InvalidAlphaActor;
        if (self.alpha_critic <= 0.0 or self.alpha_critic > 1.0) return error.InvalidAlphaCritic;
        if (self.gamma < 0.0 or self.gamma > 1.0) return error.InvalidGamma;
        if (self.n_steps == 0) return error.InvalidNSteps;
        if (self.entropy_coeff < 0.0) return error.InvalidEntropyCoeff;
        if (self.temperature <= 0.0) return error.InvalidTemperature;
        if (self.min_temperature <= 0.0 or self.min_temperature > self.temperature) return error.InvalidMinTemperature;
        if (self.temperature_decay <= 0.0 or self.temperature_decay > 1.0) return error.InvalidTemperatureDecay;
    }
};

/// A2C agent
pub fn A2C(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        config: Config,
        /// Policy preferences: θ(s,a) — unnormalized action values
        policy: [][]T,
        /// State value function: V(s)
        value_fn: []T,
        /// Current temperature for softmax
        temperature: T,
        /// Trajectory buffer for n-step returns
        trajectory: std.ArrayList(Step),

        const Step = struct {
            state: usize,
            action: usize,
            reward: T,
            next_state: usize,
            done: bool,
        };

        /// Initialize A2C agent
        /// Time: O(|S| × |A|)
        /// Space: O(|S| × |A| + |S|)
        pub fn init(allocator: Allocator, config: Config) !Self {
            try config.validate();

            // Allocate policy (state × action preferences)
            const policy = try allocator.alloc([]T, config.n_states);
            errdefer allocator.free(policy);

            for (policy) |*state_policy| {
                state_policy.* = try allocator.alloc(T, config.n_actions);
                // Initialize to uniform preferences (zero in log space)
                @memset(state_policy.*, 0.0);
            }

            // Allocate value function
            const value_fn = try allocator.alloc(T, config.n_states);
            errdefer allocator.free(value_fn);
            @memset(value_fn, 0.0);

            const trajectory = std.ArrayList(Step).init(allocator);

            return Self{
                .allocator = allocator,
                .config = config,
                .policy = policy,
                .value_fn = value_fn,
                .temperature = @floatCast(config.temperature),
                .trajectory = trajectory,
            };
        }

        /// Clean up resources
        pub fn deinit(self: *Self) void {
            for (self.policy) |state_policy| {
                self.allocator.free(state_policy);
            }
            self.allocator.free(self.policy);
            self.allocator.free(self.value_fn);
            self.trajectory.deinit();
        }

        /// Select action using softmax policy
        /// Time: O(|A|)
        /// Space: O(1)
        pub fn selectAction(self: *Self, state: usize, rng: std.Random) !usize {
            if (state >= self.config.n_states) return error.InvalidState;

            const probs = try self.actionProbabilities(state);
            defer self.allocator.free(probs);

            // Sample from categorical distribution
            const rand_val: T = @floatCast(rng.float(f64));
            var cumsum: T = 0.0;
            for (probs, 0..) |p, a| {
                cumsum += p;
                if (rand_val <= cumsum) {
                    return a;
                }
            }

            // Fallback (numerical precision)
            return self.config.n_actions - 1;
        }

        /// Get action probabilities under current policy
        /// Time: O(|A|)
        /// Space: O(|A|)
        pub fn actionProbabilities(self: *Self, state: usize) ![]T {
            if (state >= self.config.n_states) return error.InvalidState;

            const probs = try self.allocator.alloc(T, self.config.n_actions);
            errdefer self.allocator.free(probs);

            // Softmax with temperature: π(a|s) = exp(θ(s,a)/τ) / Σ exp(θ(s,a')/τ)
            const preferences = self.policy[state];
            var max_pref: T = preferences[0];
            for (preferences[1..]) |pref| {
                max_pref = @max(max_pref, pref);
            }

            var sum_exp: T = 0.0;
            for (preferences, 0..) |pref, a| {
                const scaled = (pref - max_pref) / self.temperature;
                probs[a] = @exp(scaled);
                sum_exp += probs[a];
            }

            // Normalize
            for (probs) |*p| {
                p.* /= sum_exp;
            }

            return probs;
        }

        /// Select greedy action (highest preference)
        /// Time: O(|A|)
        /// Space: O(1)
        pub fn greedyAction(self: *Self, state: usize) !usize {
            if (state >= self.config.n_states) return error.InvalidState;

            const preferences = self.policy[state];
            var best_action: usize = 0;
            var best_pref: T = preferences[0];

            for (preferences[1..], 1..) |pref, a| {
                if (pref > best_pref) {
                    best_pref = pref;
                    best_action = a;
                }
            }

            return best_action;
        }

        /// Store transition in trajectory buffer
        /// Time: O(1)
        /// Space: O(1)
        pub fn storeTransition(self: *Self, state: usize, action: usize, reward: T, next_state: usize, done: bool) !void {
            if (state >= self.config.n_states) return error.InvalidState;
            if (action >= self.config.n_actions) return error.InvalidAction;
            if (next_state >= self.config.n_states) return error.InvalidState;

            try self.trajectory.append(.{
                .state = state,
                .action = action,
                .reward = reward,
                .next_state = next_state,
                .done = done,
            });
        }

        /// Compute n-step advantage for a trajectory position
        /// Time: O(n)
        /// Space: O(1)
        fn computeAdvantage(self: *Self, start_idx: usize) T {
            const gamma: T = @floatCast(self.config.gamma);
            const traj = self.trajectory.items;

            // Compute n-step return: R = Σ(γ^i * r_{t+i}) + γ^n * V(s_{t+n})
            var n_step_return: T = 0.0;
            var discount: T = 1.0;
            var steps: usize = 0;

            for (start_idx..traj.len) |i| {
                const step = traj[i];
                n_step_return += discount * step.reward;
                discount *= gamma;
                steps += 1;

                if (step.done or steps >= self.config.n_steps) {
                    // Add bootstrap value if not terminal
                    if (!step.done) {
                        n_step_return += discount * self.value_fn[step.next_state];
                    }
                    break;
                }
            }

            // Advantage: A(s,a) = R - V(s)
            const state = traj[start_idx].state;
            return n_step_return - self.value_fn[state];
        }

        /// Compute policy entropy H(π) = -Σ π(a|s) log π(a|s)
        /// Time: O(|A|)
        /// Space: O(|A|)
        fn computeEntropy(self: *Self, state: usize) !T {
            const probs = try self.actionProbabilities(state);
            defer self.allocator.free(probs);

            var entropy: T = 0.0;
            for (probs) |p| {
                if (p > 0.0) {
                    entropy -= p * @log(p);
                }
            }

            return entropy;
        }

        /// Update policy and value function from trajectory
        /// Time: O(T × |A|) where T = trajectory length
        /// Space: O(|A|)
        pub fn update(self: *Self) !void {
            if (self.trajectory.items.len == 0) return;

            const alpha_actor: T = @floatCast(self.config.alpha_actor);
            const alpha_critic: T = @floatCast(self.config.alpha_critic);
            const entropy_coeff: T = @floatCast(self.config.entropy_coeff);

            // Update from each trajectory step
            for (self.trajectory.items, 0..) |step, i| {
                const state = step.state;
                const action = step.action;

                // Compute advantage
                const advantage = self.computeAdvantage(i);

                // Update critic: V(s) ← V(s) + α_critic * advantage
                self.value_fn[state] += alpha_critic * advantage;

                // Compute policy gradient
                const probs = try self.actionProbabilities(state);
                defer self.allocator.free(probs);

                // Update actor: θ(s,a) ← θ(s,a) + α_actor * A(s,a) * (1 - π(a|s))
                // Gradient of log π(a|s): ∇log π(a|s) = (1 - π(a|s)) for chosen action
                const policy_grad = 1.0 - probs[action];
                self.policy[state][action] += alpha_actor * advantage * policy_grad;

                // Entropy regularization: θ(s,a) ← θ(s,a) + β * ∇H(π)
                // Gradient of entropy w.r.t. θ(s,a): ∇H = (π(a|s) - 1) / τ
                if (entropy_coeff > 0.0) {
                    const entropy_grad = (probs[action] - 1.0) / self.temperature;
                    self.policy[state][action] += entropy_coeff * entropy_grad;
                }
            }

            // Clear trajectory buffer
            self.trajectory.clearRetainingCapacity();
        }

        /// Decay temperature for annealing exploration
        /// Time: O(1)
        /// Space: O(1)
        pub fn decayTemperature(self: *Self) void {
            const decay: T = @floatCast(self.config.temperature_decay);
            const min_temp: T = @floatCast(self.config.min_temperature);
            self.temperature = @max(self.temperature * decay, min_temp);
        }

        /// Reset agent state (keeps learned policy/value)
        /// Time: O(1)
        /// Space: O(1)
        pub fn reset(self: *Self) void {
            self.trajectory.clearRetainingCapacity();
        }
    };
}

// Tests
const testing = std.testing;

test "A2C: initialization" {
    const config = Config{
        .n_states = 4,
        .n_actions = 2,
        .alpha_actor = 0.01,
        .alpha_critic = 0.1,
        .gamma = 0.9,
        .n_steps = 3,
    };

    var agent = try A2C(f64).init(testing.allocator, config);
    defer agent.deinit();

    // Check initial policy (uniform preferences = 0.0)
    try testing.expectEqual(@as(usize, 4), agent.policy.len);
    try testing.expectEqual(@as(usize, 2), agent.policy[0].len);
    try testing.expectEqual(@as(f64, 0.0), agent.policy[0][0]);

    // Check initial value function
    try testing.expectEqual(@as(usize, 4), agent.value_fn.len);
    try testing.expectEqual(@as(f64, 0.0), agent.value_fn[0]);
}

test "A2C: action probabilities uniform initially" {
    const config = Config{ .n_states = 3, .n_actions = 3 };
    var agent = try A2C(f64).init(testing.allocator, config);
    defer agent.deinit();

    const probs = try agent.actionProbabilities(0);
    defer testing.allocator.free(probs);

    // Uniform policy: each action ~1/3
    const expected: f64 = 1.0 / 3.0;
    for (probs) |p| {
        try testing.expectApproxEqAbs(expected, p, 0.01);
    }
}

test "A2C: stochastic action selection" {
    const config = Config{ .n_states = 2, .n_actions = 2 };
    var agent = try A2C(f64).init(testing.allocator, config);
    defer agent.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Set policy to prefer action 1
    agent.policy[0][1] = 2.0;

    var counts = [_]usize{0} ** 2;
    for (0..100) |_| {
        const action = try agent.selectAction(0, rng);
        counts[action] += 1;
    }

    // Action 1 should be selected more often
    try testing.expect(counts[1] > counts[0]);
}

test "A2C: greedy action selection" {
    const config = Config{ .n_states = 2, .n_actions = 3 };
    var agent = try A2C(f64).init(testing.allocator, config);
    defer agent.deinit();

    // Set preferences: action 2 is best
    agent.policy[0][0] = 0.5;
    agent.policy[0][1] = 1.0;
    agent.policy[0][2] = 2.0;

    const action = try agent.greedyAction(0);
    try testing.expectEqual(@as(usize, 2), action);
}

test "A2C: store transition" {
    const config = Config{ .n_states = 4, .n_actions = 2, .n_steps = 3 };
    var agent = try A2C(f64).init(testing.allocator, config);
    defer agent.deinit();

    try agent.storeTransition(0, 1, 1.0, 1, false);
    try agent.storeTransition(1, 0, 0.5, 2, false);
    try agent.storeTransition(2, 1, -1.0, 3, true);

    try testing.expectEqual(@as(usize, 3), agent.trajectory.items.len);
    try testing.expectEqual(@as(usize, 0), agent.trajectory.items[0].state);
    try testing.expectEqual(@as(f64, 1.0), agent.trajectory.items[0].reward);
    try testing.expect(agent.trajectory.items[2].done);
}

test "A2C: n-step advantage computation" {
    const config = Config{
        .n_states = 4,
        .n_actions = 2,
        .gamma = 0.9,
        .n_steps = 3,
    };
    var agent = try A2C(f64).init(testing.allocator, config);
    defer agent.deinit();

    // Set known value function
    agent.value_fn[0] = 1.0;
    agent.value_fn[1] = 2.0;
    agent.value_fn[2] = 1.5;

    // Store trajectory: r=1.0, r=0.5, r=2.0
    try agent.storeTransition(0, 0, 1.0, 1, false);
    try agent.storeTransition(1, 1, 0.5, 2, false);
    try agent.storeTransition(2, 0, 2.0, 3, true);

    // Compute advantage for first step
    // R = 1.0 + 0.9*0.5 + 0.9²*2.0 = 1.0 + 0.45 + 1.62 = 3.07
    // A = R - V(s0) = 3.07 - 1.0 = 2.07
    const adv = agent.computeAdvantage(0);
    try testing.expectApproxEqAbs(@as(f64, 2.07), adv, 0.01);
}

test "A2C: advantage with terminal state" {
    const config = Config{
        .n_states = 3,
        .n_actions = 2,
        .gamma = 0.9,
        .n_steps = 5,
    };
    var agent = try A2C(f64).init(testing.allocator, config);
    defer agent.deinit();

    agent.value_fn[0] = 1.0;

    // Terminal state: no bootstrap
    try agent.storeTransition(0, 0, 10.0, 1, true);

    // A = 10.0 - 1.0 = 9.0 (no bootstrap for terminal)
    const adv = agent.computeAdvantage(0);
    try testing.expectApproxEqAbs(@as(f64, 9.0), adv, 0.01);
}

test "A2C: entropy computation" {
    const config = Config{ .n_states = 2, .n_actions = 3 };
    var agent = try A2C(f64).init(testing.allocator, config);
    defer agent.deinit();

    // Uniform distribution: H = -3 * (1/3 * log(1/3)) = log(3) ≈ 1.099
    const entropy = try agent.computeEntropy(0);
    try testing.expectApproxEqAbs(@as(f64, 1.099), entropy, 0.01);
}

test "A2C: entropy decreases with deterministic policy" {
    const config = Config{ .n_states = 2, .n_actions = 3 };
    var agent = try A2C(f64).init(testing.allocator, config);
    defer agent.deinit();

    // Make policy deterministic (very high preference for action 0)
    agent.policy[0][0] = 10.0;

    const entropy = try agent.computeEntropy(0);
    // Entropy should be close to 0 for deterministic policy
    try testing.expect(entropy < 0.1);
}

test "A2C: update increases value for positive advantage" {
    const config = Config{
        .n_states = 3,
        .n_actions = 2,
        .alpha_critic = 0.1,
        .gamma = 0.9,
        .n_steps = 2,
    };
    var agent = try A2C(f64).init(testing.allocator, config);
    defer agent.deinit();

    const initial_value = agent.value_fn[0];

    // Positive reward trajectory
    try agent.storeTransition(0, 0, 5.0, 1, false);
    try agent.storeTransition(1, 1, 3.0, 2, true);

    try agent.update();

    // Value should increase due to positive advantage
    try testing.expect(agent.value_fn[0] > initial_value);
}

test "A2C: update decreases value for negative advantage" {
    const config = Config{
        .n_states = 3,
        .n_actions = 2,
        .alpha_critic = 0.1,
        .gamma = 0.9,
        .n_steps = 2,
    };
    var agent = try A2C(f64).init(testing.allocator, config);
    defer agent.deinit();

    // Set high initial value
    agent.value_fn[0] = 10.0;
    const initial_value = agent.value_fn[0];

    // Low reward trajectory (negative advantage)
    try agent.storeTransition(0, 0, 0.1, 1, false);
    try agent.storeTransition(1, 1, 0.1, 2, true);

    try agent.update();

    // Value should decrease due to negative advantage
    try testing.expect(agent.value_fn[0] < initial_value);
}

test "A2C: policy update increases preference for good actions" {
    const config = Config{
        .n_states = 3,
        .n_actions = 2,
        .alpha_actor = 0.1,
        .alpha_critic = 0.1,
        .gamma = 0.9,
        .n_steps = 1,
        .entropy_coeff = 0.0, // Disable entropy for clearer test
    };
    var agent = try A2C(f64).init(testing.allocator, config);
    defer agent.deinit();

    const initial_pref = agent.policy[0][0];

    // Good action (high reward)
    try agent.storeTransition(0, 0, 10.0, 1, true);

    try agent.update();

    // Preference for action 0 should increase
    try testing.expect(agent.policy[0][0] > initial_pref);
}

test "A2C: temperature decay" {
    const config = Config{
        .n_states = 2,
        .n_actions = 2,
        .temperature = 1.0,
        .min_temperature = 0.1,
        .temperature_decay = 0.9,
    };
    var agent = try A2C(f64).init(testing.allocator, config);
    defer agent.deinit();

    try testing.expectEqual(@as(f64, 1.0), agent.temperature);

    agent.decayTemperature();
    try testing.expectApproxEqAbs(@as(f64, 0.9), agent.temperature, 0.001);

    // Decay multiple times
    for (0..20) |_| {
        agent.decayTemperature();
    }

    // Should not go below minimum
    try testing.expect(agent.temperature >= 0.1);
}

test "A2C: reset clears trajectory" {
    const config = Config{ .n_states = 3, .n_actions = 2, .n_steps = 3 };
    var agent = try A2C(f64).init(testing.allocator, config);
    defer agent.deinit();

    try agent.storeTransition(0, 0, 1.0, 1, false);
    try agent.storeTransition(1, 1, 2.0, 2, false);

    try testing.expectEqual(@as(usize, 2), agent.trajectory.items.len);

    agent.reset();

    try testing.expectEqual(@as(usize, 0), agent.trajectory.items.len);
}

test "A2C: 2-state chain learning" {
    const config = Config{
        .n_states = 2,
        .n_actions = 2,
        .alpha_actor = 0.1,
        .alpha_critic = 0.1,
        .gamma = 0.9,
        .n_steps = 2,
        .entropy_coeff = 0.01,
    };
    var agent = try A2C(f64).init(testing.allocator, config);
    defer agent.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // State 0: action 1 → reward 1, state 1: action 0 → reward 2
    for (0..100) |_| {
        // Episode starting from state 0
        const a0 = try agent.selectAction(0, rng);
        const r0: f64 = if (a0 == 1) 1.0 else 0.0;
        try agent.storeTransition(0, a0, r0, 1, false);

        const a1 = try agent.selectAction(1, rng);
        const r1: f64 = if (a1 == 0) 2.0 else 0.0;
        try agent.storeTransition(1, a1, r1, 0, true);

        try agent.update();
        agent.decayTemperature();
    }

    // Check that policy learned optimal actions
    const best_a0 = try agent.greedyAction(0);
    const best_a1 = try agent.greedyAction(1);

    try testing.expectEqual(@as(usize, 1), best_a0);
    try testing.expectEqual(@as(usize, 0), best_a1);
}

test "A2C: f32 support" {
    const config = Config{
        .n_states = 3,
        .n_actions = 2,
        .alpha_actor = 0.01,
        .alpha_critic = 0.1,
    };

    var agent = try A2C(f32).init(testing.allocator, config);
    defer agent.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const action = try agent.selectAction(0, rng);
    try testing.expect(action < 2);

    try agent.storeTransition(0, action, 1.0, 1, false);
    try agent.update();

    // Should compile and run without errors
}

test "A2C: large state-action space" {
    const config = Config{
        .n_states = 20,
        .n_actions = 5,
        .alpha_actor = 0.001,
        .alpha_critic = 0.01,
        .n_steps = 5,
    };

    var agent = try A2C(f64).init(testing.allocator, config);
    defer agent.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Test basic operations on large space
    for (0..20) |s| {
        const action = try agent.selectAction(s, rng);
        try testing.expect(action < 5);

        const next_s = (s + 1) % 20;
        try agent.storeTransition(s, action, 1.0, next_s, s == 19);
    }

    try agent.update();

    // Should handle large spaces efficiently
    try testing.expectEqual(@as(usize, 0), agent.trajectory.items.len);
}

test "A2C: config validation" {
    // Invalid n_states
    {
        const config = Config{ .n_states = 0, .n_actions = 2 };
        const result = A2C(f64).init(testing.allocator, config);
        try testing.expectError(error.InvalidNStates, result);
    }

    // Invalid alpha_actor
    {
        const config = Config{ .n_states = 2, .n_actions = 2, .alpha_actor = 0.0 };
        const result = A2C(f64).init(testing.allocator, config);
        try testing.expectError(error.InvalidAlphaActor, result);
    }

    // Invalid gamma
    {
        const config = Config{ .n_states = 2, .n_actions = 2, .gamma = 1.5 };
        const result = A2C(f64).init(testing.allocator, config);
        try testing.expectError(error.InvalidGamma, result);
    }

    // Invalid n_steps
    {
        const config = Config{ .n_states = 2, .n_actions = 2, .n_steps = 0 };
        const result = A2C(f64).init(testing.allocator, config);
        try testing.expectError(error.InvalidNSteps, result);
    }

    // Invalid temperature decay
    {
        const config = Config{ .n_states = 2, .n_actions = 2, .temperature_decay = 1.5 };
        const result = A2C(f64).init(testing.allocator, config);
        try testing.expectError(error.InvalidTemperatureDecay, result);
    }
}

test "A2C: error handling" {
    const config = Config{ .n_states = 3, .n_actions = 2 };
    var agent = try A2C(f64).init(testing.allocator, config);
    defer agent.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Invalid state
    try testing.expectError(error.InvalidState, agent.selectAction(10, rng));

    // Invalid action
    try testing.expectError(error.InvalidAction, agent.storeTransition(0, 10, 1.0, 1, false));
}

test "A2C: memory safety" {
    const config = Config{ .n_states = 5, .n_actions = 3, .n_steps = 4 };
    var agent = try A2C(f64).init(testing.allocator, config);
    defer agent.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Multiple episodes
    for (0..10) |_| {
        for (0..5) |s| {
            const action = try agent.selectAction(s, rng);
            try agent.storeTransition(s, action, 1.0, (s + 1) % 5, s == 4);
        }
        try agent.update();
    }

    // No memory leaks (verified by testing.allocator)
}
