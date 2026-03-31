/// Actor-Critic Reinforcement Learning Algorithm
///
/// Actor-Critic is a fundamental on-policy RL algorithm that combines:
/// - **Actor**: Policy π(a|s) that selects actions (learned via policy gradient)
/// - **Critic**: Value function V(s) that evaluates states (learned via TD learning)
///
/// **Algorithm Overview**:
/// 1. Actor maintains stochastic policy π(a|s) using softmax over preferences
/// 2. Critic maintains state value function V(s)
/// 3. After each transition (s,a,r,s'):
///    - Compute TD error: δ = r + γV(s') - V(s)
///    - Update critic: V(s) ← V(s) + α_critic × δ
///    - Update actor: preference(s,a) ← preference(s,a) + α_actor × δ
///
/// **Key Advantages**:
/// - Lower variance than REINFORCE (uses critic baseline)
/// - Naturally stochastic policy (no epsilon-greedy needed)
/// - Continuous learning (updates after every step)
/// - Foundation for advanced methods (A2C, A3C, PPO)
///
/// **Time Complexity**: O(|A|) per update (softmax computation)
/// **Space Complexity**: O(|S| + |S|×|A|) for value function + policy tables
///
/// **Use Cases**:
/// - Continuous online learning
/// - Stochastic environments
/// - Robotics control (with function approximation)
/// - Game playing (learns both policy and value)
/// - Baseline for deep RL (A2C, A3C)
///
/// **Trade-offs**:
/// - vs Q-Learning: More stable (policy gradient), but on-policy (less sample efficient)
/// - vs SARSA: Lower variance (critic baseline), explicitly learned policy
/// - vs Expected SARSA: Similar stability, better for continuous actions
/// - vs REINFORCE: Much lower variance, faster convergence

const std = @import("std");
const Allocator = std.mem.Allocator;
const Random = std.Random;

/// Configuration for Actor-Critic algorithm
pub const Config = struct {
    /// Critic learning rate (0.05-0.2 typical for tabular)
    alpha_critic: f64 = 0.1,

    /// Actor learning rate (0.001-0.05 typical, usually lower than critic)
    alpha_actor: f64 = 0.01,

    /// Discount factor for future rewards (0.9-0.99 typical)
    gamma: f64 = 0.99,

    /// Softmax temperature for action selection (0.1-1.0)
    /// Higher = more exploration, lower = more exploitation
    temperature: f64 = 1.0,
};

/// Actor-Critic agent for discrete state-action spaces
///
/// Combines policy gradient (actor) with temporal difference value learning (critic)
/// for on-policy reinforcement learning.
///
/// Time: O(|A|) per update (softmax probability computation)
/// Space: O(|S| + |S|×|A|) for value function + policy preferences
pub fn ActorCritic(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Number of states in the environment
        num_states: usize,

        /// Number of actions available in each state
        num_actions: usize,

        /// State value function V(s) - critic
        /// Time: O(1) access, Space: O(|S|)
        values: []T,

        /// Policy preferences (logits before softmax) - actor
        /// policy_preferences[s][a] = unnormalized log-probability
        /// Time: O(|A|) for softmax, Space: O(|S|×|A|)
        policy_preferences: [][]T,

        /// Critic learning rate (TD value updates)
        alpha_critic: T,

        /// Actor learning rate (policy gradient updates)
        alpha_actor: T,

        /// Discount factor for future rewards
        gamma: T,

        /// Softmax temperature for action selection
        temperature: T,

        allocator: Allocator,

        /// Initialize Actor-Critic agent
        ///
        /// Time: O(|S|×|A|)
        /// Space: O(|S|×|A|)
        pub fn init(
            allocator: Allocator,
            num_states: usize,
            num_actions: usize,
            config: Config,
        ) !Self {
            if (num_states == 0) return error.InvalidStateCount;
            if (num_actions == 0) return error.InvalidActionCount;
            if (config.alpha_critic <= 0 or config.alpha_critic > 1) return error.InvalidLearningRate;
            if (config.alpha_actor <= 0 or config.alpha_actor > 1) return error.InvalidLearningRate;
            if (config.gamma < 0 or config.gamma > 1) return error.InvalidDiscountFactor;
            if (config.temperature <= 0) return error.InvalidTemperature;

            // Allocate value function (initialized to 0)
            const values = try allocator.alloc(T, num_states);
            @memset(values, 0);

            // Allocate policy preferences (initialized to 0 = uniform after softmax)
            const policy_preferences = try allocator.alloc([]T, num_states);
            errdefer allocator.free(policy_preferences);

            for (policy_preferences, 0..) |*row, i| {
                row.* = try allocator.alloc(T, num_actions);
                errdefer {
                    for (policy_preferences[0..i]) |r| {
                        allocator.free(r);
                    }
                }
                @memset(row.*, 0);
            }

            return Self{
                .num_states = num_states,
                .num_actions = num_actions,
                .values = values,
                .policy_preferences = policy_preferences,
                .alpha_critic = @floatCast(config.alpha_critic),
                .alpha_actor = @floatCast(config.alpha_actor),
                .gamma = @floatCast(config.gamma),
                .temperature = @floatCast(config.temperature),
                .allocator = allocator,
            };
        }

        /// Free all allocated memory
        pub fn deinit(self: *Self) void {
            for (self.policy_preferences) |row| {
                self.allocator.free(row);
            }
            self.allocator.free(self.policy_preferences);
            self.allocator.free(self.values);
        }

        /// Select action from current policy using softmax sampling
        ///
        /// Time: O(|A|)
        pub fn selectAction(self: *const Self, state: usize, rng: Random) !usize {
            if (state >= self.num_states) return error.InvalidState;

            // Compute softmax probabilities
            const probs = try self.getPolicyDistribution(state, self.allocator);
            defer self.allocator.free(probs);

            // Sample from distribution
            const r = rng.float(T);
            var cumsum: T = 0;
            for (probs, 0..) |p, i| {
                cumsum += p;
                if (r <= cumsum) return i;
            }

            // Fallback (numerical precision)
            return self.num_actions - 1;
        }

        /// Get probability of taking action in state under current policy
        ///
        /// Time: O(|A|) (requires softmax normalization)
        pub fn getActionProbability(self: *const Self, state: usize, action: usize) !T {
            if (state >= self.num_states) return error.InvalidState;
            if (action >= self.num_actions) return error.InvalidAction;

            const probs = try self.getPolicyDistribution(state, self.allocator);
            defer self.allocator.free(probs);

            return probs[action];
        }

        /// Get full policy distribution π(·|s) for state
        ///
        /// Returns softmax over preferences: π(a|s) = exp(pref[a]/T) / Σ exp(pref[a']/T)
        ///
        /// Time: O(|A|)
        /// Space: O(|A|) (caller must free)
        pub fn getPolicyDistribution(self: *const Self, state: usize, allocator: Allocator) ![]T {
            if (state >= self.num_states) return error.InvalidState;

            const probs = try allocator.alloc(T, self.num_actions);
            const prefs = self.policy_preferences[state];

            // Compute max for numerical stability
            var max_pref = prefs[0];
            for (prefs[1..]) |p| {
                max_pref = @max(max_pref, p);
            }

            // Compute exp(pref/T - max/T)
            var sum: T = 0;
            for (prefs, 0..) |p, i| {
                probs[i] = @exp((p - max_pref) / self.temperature);
                sum += probs[i];
            }

            // Normalize
            for (probs) |*p| {
                p.* /= sum;
            }

            return probs;
        }

        /// Update actor and critic from transition (s, a, r, s')
        ///
        /// **Critic Update**: V(s) ← V(s) + α_critic × δ
        /// **Actor Update**: pref(s,a) ← pref(s,a) + α_actor × δ
        /// where δ = r + γV(s') - V(s) is the TD error
        ///
        /// Time: O(1)
        pub fn update(
            self: *Self,
            state: usize,
            action: usize,
            reward: T,
            next_state: usize,
            terminal: bool,
        ) !void {
            if (state >= self.num_states) return error.InvalidState;
            if (action >= self.num_actions) return error.InvalidAction;
            if (next_state >= self.num_states) return error.InvalidNextState;

            // Compute TD error: δ = r + γV(s') - V(s)
            const next_value: T = if (terminal) 0 else self.values[next_state];
            const td_error = reward + self.gamma * next_value - self.values[state];

            // Update critic: V(s) ← V(s) + α_critic × δ
            self.values[state] += self.alpha_critic * td_error;

            // Update actor: pref(s,a) ← pref(s,a) + α_actor × δ
            self.policy_preferences[state][action] += self.alpha_actor * td_error;
        }

        /// Get state value V(s) from critic
        ///
        /// Time: O(1)
        pub fn getValue(self: *const Self, state: usize) !T {
            if (state >= self.num_states) return error.InvalidState;
            return self.values[state];
        }

        /// Get greedy action (highest probability under current policy)
        ///
        /// Time: O(|A|)
        pub fn getGreedyAction(self: *const Self, state: usize) !usize {
            if (state >= self.num_states) return error.InvalidState;

            const prefs = self.policy_preferences[state];
            var best_action: usize = 0;
            var best_pref = prefs[0];

            for (prefs[1..], 1..) |p, i| {
                if (p > best_pref) {
                    best_pref = p;
                    best_action = i;
                }
            }

            return best_action;
        }

        /// Reset value function and policy (for new learning session)
        ///
        /// Time: O(|S|×|A|)
        pub fn reset(self: *Self) void {
            @memset(self.values, 0);
            for (self.policy_preferences) |row| {
                @memset(row, 0);
            }
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const expectEqual = testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;

test "ActorCritic: basic initialization" {
    const allocator = testing.allocator;

    const config = Config{
        .alpha_critic = 0.1,
        .alpha_actor = 0.01,
        .gamma = 0.9,
        .temperature = 1.0,
    };

    var agent = try ActorCritic(f64).init(allocator, 5, 3, config);
    defer agent.deinit();

    try expectEqual(@as(usize, 5), agent.num_states);
    try expectEqual(@as(usize, 3), agent.num_actions);
    try expectApproxEqAbs(@as(f64, 0.1), agent.alpha_critic, 1e-10);
    try expectApproxEqAbs(@as(f64, 0.01), agent.alpha_actor, 1e-10);

    // Initial values should be 0
    for (agent.values) |v| {
        try expectApproxEqAbs(@as(f64, 0.0), v, 1e-10);
    }
}

test "ActorCritic: uniform initial policy" {
    const allocator = testing.allocator;

    var agent = try ActorCritic(f64).init(allocator, 3, 4, .{});
    defer agent.deinit();

    // Initial preferences are 0, so softmax should be uniform
    const probs = try agent.getPolicyDistribution(0, allocator);
    defer allocator.free(probs);

    const expected = 1.0 / 4.0;
    for (probs) |p| {
        try expectApproxEqAbs(expected, p, 1e-6);
    }
}

test "ActorCritic: policy distribution sums to 1" {
    const allocator = testing.allocator;

    var agent = try ActorCritic(f64).init(allocator, 2, 5, .{});
    defer agent.deinit();

    // Modify some preferences
    agent.policy_preferences[0][0] = 1.0;
    agent.policy_preferences[0][2] = -0.5;
    agent.policy_preferences[0][4] = 2.0;

    const probs = try agent.getPolicyDistribution(0, allocator);
    defer allocator.free(probs);

    var sum: f64 = 0;
    for (probs) |p| {
        sum += p;
    }
    try expectApproxEqAbs(1.0, sum, 1e-6);
}

test "ActorCritic: temperature effect on policy" {
    const allocator = testing.allocator;

    // High temperature = more uniform
    var agent_hot = try ActorCritic(f64).init(allocator, 1, 3, .{ .temperature = 10.0 });
    defer agent_hot.deinit();

    agent_hot.policy_preferences[0][0] = 5.0;
    agent_hot.policy_preferences[0][1] = 0.0;
    agent_hot.policy_preferences[0][2] = 0.0;

    const probs_hot = try agent_hot.getPolicyDistribution(0, allocator);
    defer allocator.free(probs_hot);

    // Low temperature = more peaked
    var agent_cold = try ActorCritic(f64).init(allocator, 1, 3, .{ .temperature = 0.1 });
    defer agent_cold.deinit();

    agent_cold.policy_preferences[0][0] = 5.0;
    agent_cold.policy_preferences[0][1] = 0.0;
    agent_cold.policy_preferences[0][2] = 0.0;

    const probs_cold = try agent_cold.getPolicyDistribution(0, allocator);
    defer allocator.free(probs_cold);

    // Cold should have higher probability on best action
    try testing.expect(probs_cold[0] > probs_hot[0]);
    try testing.expect(probs_cold[0] > 0.9); // Very peaked
}

test "ActorCritic: 2-state chain learning" {
    const allocator = testing.allocator;

    // Simple chain: S0 --(a0)--> S1 (reward=1, terminal)
    var agent = try ActorCritic(f64).init(allocator, 2, 2, .{
        .alpha_critic = 0.5,
        .alpha_actor = 0.1,
        .gamma = 0.9,
    });
    defer agent.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Train with action 0 leading to reward
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const action = try agent.selectAction(0, rng);
        if (action == 0) {
            try agent.update(0, 0, 1.0, 1, true);
        } else {
            try agent.update(0, 1, 0.0, 0, false); // Stay in S0
        }
    }

    // Value of S0 should increase
    const v0 = try agent.getValue(0);
    try testing.expect(v0 > 0.5);

    // Policy should prefer action 0
    const probs = try agent.getPolicyDistribution(0, allocator);
    defer allocator.free(probs);
    try testing.expect(probs[0] > 0.6);
}

test "ActorCritic: gridworld navigation" {
    const allocator = testing.allocator;

    // 4×4 gridworld, states 0-15, goal at state 15
    // Actions: 0=up, 1=right, 2=down, 3=left
    const num_states = 16;
    const num_actions = 4;

    var agent = try ActorCritic(f64).init(allocator, num_states, num_actions, .{
        .alpha_critic = 0.1,
        .alpha_actor = 0.01,
        .gamma = 0.95,
    });
    defer agent.deinit();

    var prng = std.Random.DefaultPrng.init(123);
    const rng = prng.random();

    // Simulate episodes
    var episode: usize = 0;
    while (episode < 200) : (episode += 1) {
        var state: usize = 0; // Start at top-left
        var steps: usize = 0;

        while (steps < 100) : (steps += 1) {
            const action = try agent.selectAction(state, rng);

            // Simple transitions (no walls)
            var next_state = state;
            if (action == 0 and state >= 4) next_state = state - 4; // up
            if (action == 1 and state % 4 < 3) next_state = state + 1; // right
            if (action == 2 and state < 12) next_state = state + 4; // down
            if (action == 3 and state % 4 > 0) next_state = state - 1; // left

            const reward: f64 = if (next_state == 15) 1.0 else -0.01;
            const terminal = next_state == 15;

            try agent.update(state, action, reward, next_state, terminal);

            if (terminal) break;
            state = next_state;
        }
    }

    // Value of goal state should be high
    const v_goal = try agent.getValue(15);
    try testing.expect(v_goal > 0.5);

    // Value should increase as we get closer to goal
    const v_near = try agent.getValue(14); // One step from goal
    const v_far = try agent.getValue(0); // Start state
    try testing.expect(v_near > v_far);
}

test "ActorCritic: TD error computation" {
    const allocator = testing.allocator;

    var agent = try ActorCritic(f64).init(allocator, 3, 2, .{
        .alpha_critic = 0.1,
        .gamma = 0.9,
    });
    defer agent.deinit();

    // Set some values manually
    agent.values[0] = 5.0;
    agent.values[1] = 8.0;

    const v0_before = agent.values[0];

    // Update: δ = 1.0 + 0.9 * 8.0 - 5.0 = 3.2
    try agent.update(0, 0, 1.0, 1, false);

    const v0_after = agent.values[0];
    const expected = v0_before + 0.1 * (1.0 + 0.9 * 8.0 - 5.0);

    try expectApproxEqAbs(expected, v0_after, 1e-6);
}

test "ActorCritic: terminal state handling" {
    const allocator = testing.allocator;

    var agent = try ActorCritic(f64).init(allocator, 3, 2, .{
        .alpha_critic = 0.1,
        .gamma = 0.9,
    });
    defer agent.deinit();

    agent.values[0] = 5.0;
    agent.values[1] = 8.0;

    const v0_before = agent.values[0];

    // Terminal update: δ = 10.0 + 0 - 5.0 = 5.0 (no next state value)
    try agent.update(0, 0, 10.0, 1, true);

    const v0_after = agent.values[0];
    const expected = v0_before + 0.1 * (10.0 + 0.0 - 5.0);

    try expectApproxEqAbs(expected, v0_after, 1e-6);
}

test "ActorCritic: positive advantage increases probability" {
    const allocator = testing.allocator;

    var agent = try ActorCritic(f64).init(allocator, 2, 3, .{
        .alpha_actor = 0.1,
        .alpha_critic = 0.1,
        .gamma = 0.9,
    });
    defer agent.deinit();

    const pref_before = agent.policy_preferences[0][1];

    // Positive TD error → increase preference
    try agent.update(0, 1, 5.0, 1, false); // High reward

    const pref_after = agent.policy_preferences[0][1];
    try testing.expect(pref_after > pref_before);
}

test "ActorCritic: negative advantage decreases probability" {
    const allocator = testing.allocator;

    var agent = try ActorCritic(f64).init(allocator, 2, 3, .{
        .alpha_actor = 0.1,
        .alpha_critic = 0.1,
        .gamma = 0.9,
    });
    defer agent.deinit();

    agent.values[0] = 10.0; // High current value

    const pref_before = agent.policy_preferences[0][1];

    // Negative TD error → decrease preference
    try agent.update(0, 1, 0.0, 1, false); // Low reward

    const pref_after = agent.policy_preferences[0][1];
    try testing.expect(pref_after < pref_before);
}

test "ActorCritic: greedy action selection" {
    const allocator = testing.allocator;

    var agent = try ActorCritic(f64).init(allocator, 2, 4, .{});
    defer agent.deinit();

    // Make action 2 strongly preferred
    agent.policy_preferences[0][0] = 0.0;
    agent.policy_preferences[0][1] = 1.0;
    agent.policy_preferences[0][2] = 5.0;
    agent.policy_preferences[0][3] = -1.0;

    const action = try agent.getGreedyAction(0);
    try expectEqual(@as(usize, 2), action);
}

test "ActorCritic: large state-action space" {
    const allocator = testing.allocator;

    var agent = try ActorCritic(f64).init(allocator, 100, 10, .{});
    defer agent.deinit();

    try expectEqual(@as(usize, 100), agent.num_states);
    try expectEqual(@as(usize, 10), agent.num_actions);

    // Update all states
    var s: usize = 0;
    while (s < 100) : (s += 1) {
        try agent.update(s, s % 10, 1.0, (s + 1) % 100, false);
    }

    // Verify values updated
    var non_zero: usize = 0;
    for (agent.values) |v| {
        if (v != 0.0) non_zero += 1;
    }
    try testing.expect(non_zero > 0);
}

test "ActorCritic: f32 support" {
    const allocator = testing.allocator;

    var agent = try ActorCritic(f32).init(allocator, 3, 2, .{
        .alpha_critic = 0.1,
        .alpha_actor = 0.01,
        .gamma = 0.9,
    });
    defer agent.deinit();

    try agent.update(0, 0, 1.0, 1, false);

    const v0 = try agent.getValue(0);
    try testing.expect(v0 > 0.0);
}

test "ActorCritic: invalid states error" {
    const allocator = testing.allocator;

    var agent = try ActorCritic(f64).init(allocator, 3, 2, .{});
    defer agent.deinit();

    try testing.expectError(error.InvalidState, agent.getValue(5));
    try testing.expectError(error.InvalidState, agent.update(5, 0, 1.0, 0, false));
}

test "ActorCritic: invalid actions error" {
    const allocator = testing.allocator;

    var agent = try ActorCritic(f64).init(allocator, 3, 2, .{});
    defer agent.deinit();

    try testing.expectError(error.InvalidAction, agent.update(0, 5, 1.0, 1, false));
}

test "ActorCritic: invalid config errors" {
    const allocator = testing.allocator;

    try testing.expectError(error.InvalidStateCount, ActorCritic(f64).init(allocator, 0, 2, .{}));
    try testing.expectError(error.InvalidActionCount, ActorCritic(f64).init(allocator, 2, 0, .{}));
    try testing.expectError(error.InvalidLearningRate, ActorCritic(f64).init(allocator, 2, 2, .{ .alpha_critic = -0.1 }));
    try testing.expectError(error.InvalidDiscountFactor, ActorCritic(f64).init(allocator, 2, 2, .{ .gamma = 1.5 }));
    try testing.expectError(error.InvalidTemperature, ActorCritic(f64).init(allocator, 2, 2, .{ .temperature = 0.0 }));
}

test "ActorCritic: reset functionality" {
    const allocator = testing.allocator;

    var agent = try ActorCritic(f64).init(allocator, 3, 2, .{});
    defer agent.deinit();

    // Update some values
    try agent.update(0, 0, 5.0, 1, false);
    try agent.update(1, 1, 3.0, 2, false);

    // Values should be non-zero
    try testing.expect((try agent.getValue(0)) != 0.0);

    // Reset
    agent.reset();

    // All values should be zero
    for (agent.values) |v| {
        try expectApproxEqAbs(0.0, v, 1e-10);
    }

    // All preferences should be zero
    for (agent.policy_preferences) |row| {
        for (row) |p| {
            try expectApproxEqAbs(0.0, p, 1e-10);
        }
    }
}

test "ActorCritic: convergence validation" {
    const allocator = testing.allocator;

    // Simple deterministic task: state 0, action 0 → reward 1
    var agent = try ActorCritic(f64).init(allocator, 1, 2, .{
        .alpha_critic = 0.3,
        .alpha_actor = 0.1,
        .gamma = 0.0, // No discounting for simplicity
    });
    defer agent.deinit();

    // Repeatedly take action 0 and get reward 1
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        try agent.update(0, 0, 1.0, 0, true);
    }

    // Value should converge to 1.0 (immediate reward)
    const v0 = try agent.getValue(0);
    try expectApproxEqAbs(1.0, v0, 0.1);

    // Policy should strongly prefer action 0
    const pref0 = agent.policy_preferences[0][0];
    const pref1 = agent.policy_preferences[0][1];
    try testing.expect(pref0 > pref1 + 5.0);
}
