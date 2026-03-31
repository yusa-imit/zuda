/// Expected SARSA (State-Action-Reward-State-Action with Expected Update)
///
/// On-policy temporal difference (TD) reinforcement learning algorithm that uses
/// expected value of next action instead of sampled action, reducing variance.
///
/// Algorithm:
/// - Initialize Q(s,a) arbitrarily for all state-action pairs
/// - Repeat for each episode:
///   - Initialize state s
///   - Repeat for each step:
///     - Choose action a from s using policy derived from Q (e.g. epsilon-greedy)
///     - Take action a, observe reward r and next state s'
///     - Compute expected Q value: E[Q(s',·)] = Σ_a' π(a'|s') Q(s',a')
///     - Update: Q(s,a) ← Q(s,a) + α[r + γ E[Q(s',·)] - Q(s,a)]
///     - s ← s'
///
/// Key Properties:
/// - On-policy: learns value of policy being followed (including exploration)
/// - Expected update: uses expectation over all actions (lower variance than SARSA)
/// - Converges to optimal policy under standard conditions
/// - More stable than SARSA, nearly as good as Q-Learning
///
/// Complexity:
/// - Time: O(|A|) per update (compute expected value over actions)
/// - Space: O(|S| × |A|) for Q-table
///
/// Use cases:
/// - General RL problems where stability and sample efficiency matter
/// - Stochastic environments (handles randomness better than SARSA)
/// - Robotics (safer than Q-Learning, more stable than SARSA)
/// - Game AI (balanced exploration-exploitation)

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Expected SARSA agent
pub fn ExpectedSARSA(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        num_states: usize,
        num_actions: usize,
        q_table: []T, // Q(s,a) flattened as [s * num_actions + a]
        alpha: T, // learning rate
        gamma: T, // discount factor
        epsilon: T, // exploration rate (epsilon-greedy)

        /// Initialize Expected SARSA agent with given parameters
        /// Time: O(|S| × |A|), Space: O(|S| × |A|)
        pub fn init(
            allocator: Allocator,
            num_states: usize,
            num_actions: usize,
            alpha: T,
            gamma: T,
            epsilon: T,
        ) !Self {
            if (num_states == 0 or num_actions == 0) return error.InvalidDimensions;
            if (alpha <= 0 or alpha > 1) return error.InvalidLearningRate;
            if (gamma < 0 or gamma > 1) return error.InvalidDiscountFactor;
            if (epsilon < 0 or epsilon > 1) return error.InvalidEpsilon;

            const table_size = num_states * num_actions;
            const q_table = try allocator.alloc(T, table_size);
            @memset(q_table, 0);

            return Self{
                .allocator = allocator,
                .num_states = num_states,
                .num_actions = num_actions,
                .q_table = q_table,
                .alpha = alpha,
                .gamma = gamma,
                .epsilon = epsilon,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.q_table);
        }

        /// Get Q-value for state-action pair
        /// Time: O(1), Space: O(1)
        pub fn getQ(self: Self, state: usize, action: usize) !T {
            if (state >= self.num_states) return error.InvalidState;
            if (action >= self.num_actions) return error.InvalidAction;
            return self.q_table[state * self.num_actions + action];
        }

        /// Set Q-value for state-action pair (for testing/initialization)
        /// Time: O(1), Space: O(1)
        pub fn setQ(self: *Self, state: usize, action: usize, value: T) !void {
            if (state >= self.num_states) return error.InvalidState;
            if (action >= self.num_actions) return error.InvalidAction;
            self.q_table[state * self.num_actions + action] = value;
        }

        /// Select action using epsilon-greedy policy
        /// Time: O(|A|), Space: O(1)
        pub fn selectAction(self: Self, state: usize, rng: std.Random) !usize {
            if (state >= self.num_states) return error.InvalidState;

            // Exploration: random action with probability epsilon
            if (rng.float(T) < self.epsilon) {
                return rng.intRangeLessThan(usize, 0, self.num_actions);
            }

            // Exploitation: greedy action (argmax Q(s,a))
            return self.greedyAction(state);
        }

        /// Select greedy action (argmax Q(s,a))
        /// Time: O(|A|), Space: O(1)
        pub fn greedyAction(self: Self, state: usize) !usize {
            if (state >= self.num_states) return error.InvalidState;

            var best_action: usize = 0;
            var best_value = try self.getQ(state, 0);

            for (1..self.num_actions) |action| {
                const q_value = try self.getQ(state, action);
                if (q_value > best_value) {
                    best_value = q_value;
                    best_action = action;
                }
            }

            return best_action;
        }

        /// Compute expected Q-value for next state using current policy
        /// E[Q(s',·)] = Σ_a' π(a'|s') Q(s',a')
        /// For epsilon-greedy: π(a'|s') = ε/|A| + (1-ε) if a' is greedy, else ε/|A|
        /// Time: O(|A|), Space: O(1)
        fn expectedQValue(self: Self, next_state: usize) !T {
            if (next_state >= self.num_states) return error.InvalidState;

            // Find greedy action for next state
            const greedy_action_idx = try self.greedyAction(next_state);

            // Expected value computation:
            // For epsilon-greedy policy:
            //   Greedy action: probability = epsilon/|A| + (1 - epsilon)
            //   Other actions: probability = epsilon/|A|
            const num_actions_f: T = @floatFromInt(self.num_actions);
            const exploration_prob = self.epsilon / num_actions_f;
            const greedy_prob = exploration_prob + (1 - self.epsilon);

            var expected_value: T = 0;

            for (0..self.num_actions) |action| {
                const q_value = try self.getQ(next_state, action);
                const prob = if (action == greedy_action_idx) greedy_prob else exploration_prob;
                expected_value += prob * q_value;
            }

            return expected_value;
        }

        /// Update Q-value using Expected SARSA rule
        /// Q(s,a) ← Q(s,a) + α[r + γ E[Q(s',·)] - Q(s,a)]
        /// Time: O(|A|), Space: O(1)
        pub fn update(
            self: *Self,
            state: usize,
            action: usize,
            reward: T,
            next_state: usize,
            is_terminal: bool,
        ) !void {
            if (state >= self.num_states) return error.InvalidState;
            if (action >= self.num_actions) return error.InvalidAction;
            if (next_state >= self.num_states) return error.InvalidState;

            const current_q = try self.getQ(state, action);

            // Terminal state: no future rewards (expected value = 0)
            const expected_next_q = if (is_terminal) 0 else try self.expectedQValue(next_state);

            // TD error: δ = r + γ E[Q(s',·)] - Q(s,a)
            const td_error = reward + self.gamma * expected_next_q - current_q;

            // Update: Q(s,a) ← Q(s,a) + α δ
            const new_q = current_q + self.alpha * td_error;
            try self.setQ(state, action, new_q);
        }

        /// Compute state value function V(s) = E[Q(s,a)] under current policy
        /// For epsilon-greedy: V(s) = Σ_a π(a|s) Q(s,a)
        /// Time: O(|A|), Space: O(1)
        pub fn stateValue(self: Self, state: usize) !T {
            if (state >= self.num_states) return error.InvalidState;

            // Find greedy action
            const greedy_action_idx = try self.greedyAction(state);

            // Compute expected value under epsilon-greedy policy
            const num_actions_f: T = @floatFromInt(self.num_actions);
            const exploration_prob = self.epsilon / num_actions_f;
            const greedy_prob = exploration_prob + (1 - self.epsilon);

            var value: T = 0;

            for (0..self.num_actions) |action| {
                const q_value = try self.getQ(state, action);
                const prob = if (action == greedy_action_idx) greedy_prob else exploration_prob;
                value += prob * q_value;
            }

            return value;
        }

        /// Decay epsilon (for annealing exploration)
        /// Time: O(1), Space: O(1)
        pub fn decayEpsilon(self: *Self, decay_rate: T) void {
            self.epsilon *= decay_rate;
            if (self.epsilon < 0.01) self.epsilon = 0.01; // minimum exploration
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "ExpectedSARSA: basic initialization" {
    var agent = try ExpectedSARSA(f64).init(testing.allocator, 4, 2, 0.1, 0.9, 0.1);
    defer agent.deinit();

    try testing.expectEqual(@as(usize, 4), agent.num_states);
    try testing.expectEqual(@as(usize, 2), agent.num_actions);
    try testing.expectEqual(@as(f64, 0.1), agent.alpha);
    try testing.expectEqual(@as(f64, 0.9), agent.gamma);

    // Q-table should be initialized to 0
    for (0..4) |s| {
        for (0..2) |a| {
            try testing.expectEqual(@as(f64, 0), try agent.getQ(s, a));
        }
    }
}

test "ExpectedSARSA: 2-state chain learning" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Simple 2-state chain: S0 -> S1 (terminal)
    // Action 0: reward 1, Action 1: reward 0
    var agent = try ExpectedSARSA(f64).init(testing.allocator, 2, 2, 0.5, 0.9, 0.1);
    defer agent.deinit();

    // Train for multiple episodes
    for (0..100) |_| {
        const state: usize = 0;
        const action = try agent.selectAction(state, rng);
        const reward: f64 = if (action == 0) 1.0 else 0.0;
        try agent.update(state, action, reward, 1, true);
    }

    // Action 0 should have higher Q-value at state 0
    const q0 = try agent.getQ(0, 0);
    const q1 = try agent.getQ(0, 1);
    try testing.expect(q0 > q1);
    try testing.expect(q0 > 0.5); // Should learn positive value
}

test "ExpectedSARSA: gridworld navigation" {
    var prng = std.Random.DefaultPrng.init(123);
    const rng = prng.random();

    // 2x2 gridworld: [0,1]
    //                [2,3]
    // Goal at state 3, actions: 0=up, 1=right, 2=down, 3=left
    var agent = try ExpectedSARSA(f64).init(testing.allocator, 4, 4, 0.2, 0.95, 0.2);
    defer agent.deinit();

    // Train for multiple episodes
    for (0..200) |_| {
        var state: usize = 0; // Start at top-left
        const max_steps: usize = 20;
        var steps: usize = 0;

        while (state != 3 and steps < max_steps) : (steps += 1) {
            const action = try agent.selectAction(state, rng);

            // Simulate gridworld transitions
            var next_state = state;
            if (action == 0 and state >= 2) next_state -= 2; // up
            if (action == 1 and (state % 2) == 0) next_state += 1; // right
            if (action == 2 and state < 2) next_state += 2; // down
            if (action == 3 and (state % 2) == 1) next_state -= 1; // left

            const reward: f64 = if (next_state == 3) 1.0 else -0.1;
            const is_terminal = (next_state == 3);

            try agent.update(state, action, reward, next_state, is_terminal);
            state = next_state;
        }
    }

    // Agent should learn to reach goal (state 3)
    // From state 0, optimal path is right then down (actions 1, 2)
    const best_action_0 = try agent.greedyAction(0);
    try testing.expect(best_action_0 == 1 or best_action_0 == 2);
}

test "ExpectedSARSA: expected value computation" {
    var agent = try ExpectedSARSA(f64).init(testing.allocator, 2, 3, 0.1, 0.9, 0.3);
    defer agent.deinit();

    // Set Q-values for state 1
    try agent.setQ(1, 0, 1.0);
    try agent.setQ(1, 1, 5.0); // greedy action
    try agent.setQ(1, 2, 2.0);

    // Expected value should be weighted average under epsilon-greedy
    // Greedy action (1): prob = 0.3/3 + 0.7 = 0.8
    // Other actions: prob = 0.3/3 = 0.1 each
    const expected = try agent.expectedQValue(1);
    const manual_expected = 0.1 * 1.0 + 0.8 * 5.0 + 0.1 * 2.0;
    try testing.expectApproxEqAbs(manual_expected, expected, 1e-10);
}

test "ExpectedSARSA: state value function" {
    var agent = try ExpectedSARSA(f64).init(testing.allocator, 2, 3, 0.1, 0.9, 0.3);
    defer agent.deinit();

    // Set Q-values for state 0
    try agent.setQ(0, 0, 1.0);
    try agent.setQ(0, 1, 5.0); // greedy action
    try agent.setQ(0, 2, 2.0);

    // State value should match expected value under policy
    const v = try agent.stateValue(0);
    const expected = try agent.expectedQValue(0);
    try testing.expectApproxEqAbs(expected, v, 1e-10);
}

test "ExpectedSARSA: epsilon-greedy action selection" {
    var prng = std.Random.DefaultPrng.init(999);
    const rng = prng.random();

    var agent = try ExpectedSARSA(f64).init(testing.allocator, 2, 3, 0.1, 0.9, 0.5);
    defer agent.deinit();

    // Set Q-values: action 1 is clearly best
    try agent.setQ(0, 0, 0.0);
    try agent.setQ(0, 1, 10.0);
    try agent.setQ(0, 2, 0.0);

    // With high epsilon, should see exploration (non-greedy actions)
    var non_greedy_count: usize = 0;
    for (0..100) |_| {
        const action = try agent.selectAction(0, rng);
        if (action != 1) non_greedy_count += 1;
    }

    // Should have some exploration (roughly 50% epsilon)
    try testing.expect(non_greedy_count > 20);
    try testing.expect(non_greedy_count < 80);
}

test "ExpectedSARSA: greedy action selection" {
    var agent = try ExpectedSARSA(f64).init(testing.allocator, 2, 4, 0.1, 0.9, 0.1);
    defer agent.deinit();

    // Set Q-values: action 2 is best
    try agent.setQ(1, 0, 1.0);
    try agent.setQ(1, 1, 2.0);
    try agent.setQ(1, 2, 5.0);
    try agent.setQ(1, 3, 3.0);

    const action = try agent.greedyAction(1);
    try testing.expectEqual(@as(usize, 2), action);
}

test "ExpectedSARSA: expected SARSA update with expected value" {
    var agent = try ExpectedSARSA(f64).init(testing.allocator, 3, 2, 0.5, 0.9, 0.2);
    defer agent.deinit();

    // Set up Q-values
    try agent.setQ(0, 1, 2.0); // current Q(s,a)
    try agent.setQ(1, 0, 3.0);
    try agent.setQ(1, 1, 5.0); // greedy at next state

    // Compute expected value at next state manually
    // Greedy action (1): prob = 0.2/2 + 0.8 = 0.9
    // Other action (0): prob = 0.2/2 = 0.1
    const expected_next = 0.1 * 3.0 + 0.9 * 5.0;

    // Update with reward 1.0
    try agent.update(0, 1, 1.0, 1, false);

    // Expected new Q = old_q + alpha * (reward + gamma * expected_next - old_q)
    //                = 2.0 + 0.5 * (1.0 + 0.9 * expected_next - 2.0)
    const expected_q = 2.0 + 0.5 * (1.0 + 0.9 * expected_next - 2.0);
    const actual_q = try agent.getQ(0, 1);
    try testing.expectApproxEqAbs(expected_q, actual_q, 1e-10);
}

test "ExpectedSARSA: terminal state update" {
    var agent = try ExpectedSARSA(f64).init(testing.allocator, 2, 2, 0.5, 0.9, 0.1);
    defer agent.deinit();

    try agent.setQ(0, 0, 1.0);

    // Terminal transition: next state value should be 0
    try agent.update(0, 0, 10.0, 1, true);

    // Expected: 1.0 + 0.5 * (10.0 + 0.9 * 0 - 1.0) = 5.5
    const q = try agent.getQ(0, 0);
    try testing.expectApproxEqAbs(5.5, q, 1e-10);
}

test "ExpectedSARSA: epsilon decay" {
    var agent = try ExpectedSARSA(f64).init(testing.allocator, 4, 2, 0.1, 0.9, 0.5);
    defer agent.deinit();

    try testing.expectEqual(@as(f64, 0.5), agent.epsilon);

    agent.decayEpsilon(0.9);
    try testing.expectApproxEqAbs(0.45, agent.epsilon, 1e-10);

    // Decay multiple times
    for (0..100) |_| {
        agent.decayEpsilon(0.9);
    }

    // Should not go below minimum
    try testing.expect(agent.epsilon >= 0.01);
}

test "ExpectedSARSA: error handling - invalid state" {
    var agent = try ExpectedSARSA(f64).init(testing.allocator, 4, 2, 0.1, 0.9, 0.1);
    defer agent.deinit();

    try testing.expectError(error.InvalidState, agent.getQ(4, 0));
    try testing.expectError(error.InvalidState, agent.setQ(5, 0, 1.0));
    try testing.expectError(error.InvalidState, agent.greedyAction(10));
}

test "ExpectedSARSA: error handling - invalid action" {
    var agent = try ExpectedSARSA(f64).init(testing.allocator, 4, 2, 0.1, 0.9, 0.1);
    defer agent.deinit();

    try testing.expectError(error.InvalidAction, agent.getQ(0, 2));
    try testing.expectError(error.InvalidAction, agent.setQ(0, 3, 1.0));
}

test "ExpectedSARSA: error handling - invalid parameters" {
    try testing.expectError(error.InvalidDimensions, ExpectedSARSA(f64).init(testing.allocator, 0, 2, 0.1, 0.9, 0.1));
    try testing.expectError(error.InvalidDimensions, ExpectedSARSA(f64).init(testing.allocator, 4, 0, 0.1, 0.9, 0.1));
    try testing.expectError(error.InvalidLearningRate, ExpectedSARSA(f64).init(testing.allocator, 4, 2, 0.0, 0.9, 0.1));
    try testing.expectError(error.InvalidLearningRate, ExpectedSARSA(f64).init(testing.allocator, 4, 2, 1.5, 0.9, 0.1));
    try testing.expectError(error.InvalidDiscountFactor, ExpectedSARSA(f64).init(testing.allocator, 4, 2, 0.1, -0.1, 0.1));
    try testing.expectError(error.InvalidDiscountFactor, ExpectedSARSA(f64).init(testing.allocator, 4, 2, 0.1, 1.5, 0.1));
    try testing.expectError(error.InvalidEpsilon, ExpectedSARSA(f64).init(testing.allocator, 4, 2, 0.1, 0.9, -0.1));
    try testing.expectError(error.InvalidEpsilon, ExpectedSARSA(f64).init(testing.allocator, 4, 2, 0.1, 0.9, 1.5));
}

test "ExpectedSARSA: f32 support" {
    var agent = try ExpectedSARSA(f32).init(testing.allocator, 4, 2, 0.1, 0.9, 0.1);
    defer agent.deinit();

    try agent.setQ(0, 0, 5.5);
    try testing.expectApproxEqAbs(@as(f32, 5.5), try agent.getQ(0, 0), 1e-6);
}

test "ExpectedSARSA: large state-action space" {
    var agent = try ExpectedSARSA(f64).init(testing.allocator, 100, 10, 0.1, 0.9, 0.1);
    defer agent.deinit();

    // Should handle large spaces without issues
    try agent.setQ(99, 9, 7.5);
    try testing.expectEqual(@as(f64, 7.5), try agent.getQ(99, 9));

    const action = try agent.greedyAction(50);
    try testing.expect(action < 10);
}

test "ExpectedSARSA: convergence validation" {
    var prng = std.Random.DefaultPrng.init(777);
    const rng = prng.random();

    // Simple deterministic environment: action 0 always better
    var agent = try ExpectedSARSA(f64).init(testing.allocator, 2, 2, 0.3, 0.9, 0.1);
    defer agent.deinit();

    // Train extensively
    for (0..500) |_| {
        const action = try agent.selectAction(0, rng);
        const reward: f64 = if (action == 0) 1.0 else 0.0;
        try agent.update(0, action, reward, 1, true);
    }

    // Should strongly prefer action 0
    const q0 = try agent.getQ(0, 0);
    const q1 = try agent.getQ(0, 1);
    try testing.expect(q0 > q1 + 0.5); // Clear preference
}

test "ExpectedSARSA: memory safety with testing.allocator" {
    var agent = try ExpectedSARSA(f64).init(testing.allocator, 10, 5, 0.1, 0.9, 0.1);
    defer agent.deinit();

    // Perform operations
    try agent.setQ(5, 3, 2.5);
    _ = try agent.getQ(5, 3);
    _ = try agent.greedyAction(5);

    // testing.allocator will detect leaks automatically
}
