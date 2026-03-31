const std = @import("std");
const Allocator = std.mem.Allocator;

/// SARSA (State-Action-Reward-State-Action) — On-policy TD control algorithm
///
/// SARSA is an on-policy reinforcement learning algorithm that learns the action-value
/// function Q(s,a) for the policy currently being followed, including exploration.
/// Unlike Q-Learning (off-policy), SARSA updates are based on the action actually taken,
/// making it more conservative but potentially more stable during training.
///
/// Algorithm:
/// 1. Initialize Q(s,a) arbitrarily for all state-action pairs
/// 2. For each episode:
///    - Choose action a from state s using policy derived from Q (e.g., epsilon-greedy)
///    - Take action a, observe reward r and next state s'
///    - Choose next action a' from s' using same policy
///    - Update: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
///    - s ← s', a ← a'
/// 3. Repeat until convergence
///
/// Key differences from Q-Learning:
/// - On-policy: learns value of policy being followed (including exploration)
/// - Uses actual next action a' (from epsilon-greedy) instead of max_a' Q(s',a')
/// - More conservative, converges to safer policies when exploring
/// - Better for stochastic environments or when exploration noise matters
///
/// Time complexity: O(|A|) per update (epsilon-greedy action selection)
/// Space complexity: O(|S| × |A|) for Q-table
///
/// Use cases:
/// - Robotics: Safe exploration in physical systems
/// - Game AI: Learning policies in stochastic games
/// - Control systems: Real-time learning with safety constraints
/// - Trading: Risk-averse strategies that account for exploration costs
pub fn SARSA(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        num_states: usize,
        num_actions: usize,
        learning_rate: T, // α: step size for Q-table updates
        discount_factor: T, // γ: future reward discount
        epsilon: T, // exploration rate for epsilon-greedy
        epsilon_decay: T, // decay rate for epsilon after each episode
        epsilon_min: T, // minimum exploration rate
        q_table: []T, // Q(s,a) values, flattened [num_states × num_actions]
        rng: std.Random.DefaultPrng,

        /// Initialize SARSA agent with given parameters
        ///
        /// Time: O(|S| × |A|)
        /// Space: O(|S| × |A|)
        pub fn init(
            allocator: Allocator,
            num_states: usize,
            num_actions: usize,
            learning_rate: T,
            discount_factor: T,
            epsilon: T,
            epsilon_decay: T,
            epsilon_min: T,
            seed: u64,
        ) !Self {
            if (num_states == 0) return error.InvalidNumStates;
            if (num_actions == 0) return error.InvalidNumActions;
            if (learning_rate <= 0 or learning_rate > 1) return error.InvalidLearningRate;
            if (discount_factor < 0 or discount_factor > 1) return error.InvalidDiscountFactor;
            if (epsilon < 0 or epsilon > 1) return error.InvalidEpsilon;
            if (epsilon_decay < 0 or epsilon_decay > 1) return error.InvalidEpsilonDecay;
            if (epsilon_min < 0 or epsilon_min > 1) return error.InvalidEpsilonMin;

            const size = num_states * num_actions;
            const q_table = try allocator.alloc(T, size);
            @memset(q_table, 0);

            return Self{
                .allocator = allocator,
                .num_states = num_states,
                .num_actions = num_actions,
                .learning_rate = learning_rate,
                .discount_factor = discount_factor,
                .epsilon = epsilon,
                .epsilon_decay = epsilon_decay,
                .epsilon_min = epsilon_min,
                .q_table = q_table,
                .rng = std.Random.DefaultPrng.init(seed),
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.q_table);
        }

        /// Select action using epsilon-greedy policy
        ///
        /// With probability epsilon: explore (random action)
        /// With probability 1-epsilon: exploit (greedy action)
        ///
        /// Time: O(|A|)
        /// Space: O(1)
        pub fn selectAction(self: *Self, state: usize) !usize {
            if (state >= self.num_states) return error.InvalidState;

            const rand = self.rng.random();
            if (rand.float(T) < self.epsilon) {
                // Explore: random action
                return rand.intRangeAtMost(usize, 0, self.num_actions - 1);
            } else {
                // Exploit: greedy action
                return self.greedyAction(state);
            }
        }

        /// Get greedy action for given state (argmax_a Q(s,a))
        ///
        /// Time: O(|A|)
        /// Space: O(1)
        pub fn greedyAction(self: *Self, state: usize) usize {
            var best_action: usize = 0;
            var best_value = self.getQ(state, 0);

            for (1..self.num_actions) |action| {
                const value = self.getQ(state, action);
                if (value > best_value) {
                    best_value = value;
                    best_action = action;
                }
            }

            return best_action;
        }

        /// SARSA update rule: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
        ///
        /// This is the key difference from Q-Learning:
        /// - Uses actual next action a' (from policy) instead of max_a' Q(s',a')
        /// - On-policy: learns value of policy being followed
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn update(
            self: *Self,
            state: usize,
            action: usize,
            reward: T,
            next_state: usize,
            next_action: usize, // Key difference: actual next action taken
            done: bool,
        ) !void {
            if (state >= self.num_states) return error.InvalidState;
            if (action >= self.num_actions) return error.InvalidAction;
            if (next_state >= self.num_states) return error.InvalidState;
            if (next_action >= self.num_actions) return error.InvalidAction;

            const current_q = self.getQ(state, action);
            const next_q = if (done) 0 else self.getQ(next_state, next_action);

            // SARSA update: uses actual next action a' (not max)
            const td_target = reward + self.discount_factor * next_q;
            const td_error = td_target - current_q;
            const new_q = current_q + self.learning_rate * td_error;

            self.setQ(state, action, new_q);
        }

        /// Decay epsilon after episode (reduce exploration over time)
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn decayEpsilon(self: *Self) void {
            self.epsilon = @max(self.epsilon_min, self.epsilon * self.epsilon_decay);
        }

        /// Get Q-value for state-action pair
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn getQ(self: *Self, state: usize, action: usize) T {
            const idx = state * self.num_actions + action;
            return self.q_table[idx];
        }

        /// Set Q-value for state-action pair
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn setQ(self: *Self, state: usize, action: usize, value: T) void {
            const idx = state * self.num_actions + action;
            self.q_table[idx] = value;
        }

        /// Get state value V(s) = E[Q(s,a)] under current policy (epsilon-greedy)
        ///
        /// V(s) = (1-ε) × max_a Q(s,a) + ε × (1/|A|) × Σ_a Q(s,a)
        ///
        /// Time: O(|A|)
        /// Space: O(1)
        pub fn getStateValue(self: *Self, state: usize) T {
            var max_q = self.getQ(state, 0);
            var sum_q: T = 0;

            for (0..self.num_actions) |action| {
                const q = self.getQ(state, action);
                max_q = @max(max_q, q);
                sum_q += q;
            }

            const avg_q = sum_q / @as(T, @floatFromInt(self.num_actions));
            return (1 - self.epsilon) * max_q + self.epsilon * avg_q;
        }
    };
}

// ========================================================================================
// Tests
// ========================================================================================

test "SARSA: basic initialization" {
    const allocator = std.testing.allocator;

    var agent = try SARSA(f64).init(allocator, 4, 2, 0.1, 0.9, 0.1, 0.99, 0.01, 42);
    defer agent.deinit();

    try std.testing.expectEqual(4, agent.num_states);
    try std.testing.expectEqual(2, agent.num_actions);
    try std.testing.expectEqual(0.1, agent.learning_rate);
    try std.testing.expectEqual(0.9, agent.discount_factor);
    try std.testing.expectEqual(0.1, agent.epsilon);

    // Q-table should be zero-initialized
    for (0..agent.num_states) |s| {
        for (0..agent.num_actions) |a| {
            try std.testing.expectEqual(0.0, agent.getQ(s, a));
        }
    }
}

test "SARSA: simple 2-state environment" {
    const allocator = std.testing.allocator;

    // Simple chain: state 0 → state 1 (terminal, reward +1)
    // Two actions: 0=stay, 1=advance
    var agent = try SARSA(f64).init(allocator, 2, 2, 0.5, 0.9, 0.0, 1.0, 0.0, 42);
    defer agent.deinit();

    // Train with greedy policy (epsilon=0) for deterministic behavior
    // Episode: s0 --(a=1, r=1)--> s1 (done)
    // With learning_rate=0.5, after each update Q(0,1) = Q(0,1) + 0.5 × [1.0 - Q(0,1)]
    for (0..20) |_| {
        const s0: usize = 0;
        const a0: usize = 1; // Always take advance action
        const r: f64 = 1.0;
        const s1: usize = 1;
        const a1: usize = 0; // Dummy action for terminal state

        try agent.update(s0, a0, r, s1, a1, true);
    }

    // After 20 updates with lr=0.5, Q(0,1) should converge to reward (1.0)
    const q_advance = agent.getQ(0, 1);
    try std.testing.expect(q_advance > 0.95); // Should be very close to 1.0
}

test "SARSA: gridworld navigation" {
    const allocator = std.testing.allocator;

    // 2x2 gridworld: states 0,1,2,3 (3 is goal with reward +10)
    // Actions: 0=left, 1=right, 2=up, 3=down
    var agent = try SARSA(f64).init(allocator, 4, 4, 0.2, 0.95, 0.3, 0.99, 0.01, 123);
    defer agent.deinit();

    // Simulate more episodes with higher learning rate
    for (0..200) |_| {
        var state: usize = 0; // Start at top-left
        var action = try agent.selectAction(state);

        for (0..10) |_| { // Max 10 steps per episode
            // Simplified gridworld dynamics
            const next_state = blk: {
                if (state == 3) break :blk 3; // Already at goal
                if (action == 1 and state % 2 == 0) break :blk state + 1; // Right
                if (action == 3 and state < 2) break :blk state + 2; // Down
                break :blk state; // Stay in place for invalid moves
            };

            const reward: f64 = if (next_state == 3) 10.0 else -0.1;
            const done = next_state == 3;

            const next_action = if (!done) try agent.selectAction(next_state) else 0;

            try agent.update(state, action, reward, next_state, next_action, done);

            if (done) break;
            state = next_state;
            action = next_action;
        }

        agent.decayEpsilon();
    }

    // Some Q-values should be positive (learned from goal reward)
    var max_q: f64 = 0.0;
    for (0..agent.num_states) |s| {
        for (0..agent.num_actions) |a| {
            max_q = @max(max_q, agent.getQ(s, a));
        }
    }
    try std.testing.expect(max_q > 3.0); // Should learn positive values from reward
}

test "SARSA: epsilon-greedy action selection" {
    const allocator = std.testing.allocator;

    var agent = try SARSA(f64).init(allocator, 2, 3, 0.1, 0.9, 1.0, 1.0, 1.0, 999);
    defer agent.deinit();

    // With epsilon=1.0, should always explore (random)
    var action_counts = [_]usize{0} ** 3;
    for (0..100) |_| {
        const action = try agent.selectAction(0);
        action_counts[action] += 1;
    }

    // All actions should be taken at least once
    try std.testing.expect(action_counts[0] > 0);
    try std.testing.expect(action_counts[1] > 0);
    try std.testing.expect(action_counts[2] > 0);
}

test "SARSA: greedy action selection" {
    const allocator = std.testing.allocator;

    var agent = try SARSA(f64).init(allocator, 1, 3, 0.1, 0.9, 0.0, 1.0, 0.0, 42);
    defer agent.deinit();

    // Set Q-values: Q(0,0)=1, Q(0,1)=5, Q(0,2)=2
    agent.setQ(0, 0, 1.0);
    agent.setQ(0, 1, 5.0);
    agent.setQ(0, 2, 2.0);

    // With epsilon=0, should always pick action 1
    for (0..10) |_| {
        const action = try agent.selectAction(0);
        try std.testing.expectEqual(1, action);
    }
}

test "SARSA: on-policy TD update" {
    const allocator = std.testing.allocator;

    var agent = try SARSA(f64).init(allocator, 2, 2, 0.5, 0.9, 0.0, 1.0, 0.0, 42);
    defer agent.deinit();

    // Initial: Q(0,1) = 0, Q(1,0) = 0
    // Update: s=0, a=1, r=1, s'=1, a'=0, done=false
    // SARSA: Q(0,1) ← 0 + 0.5 × [1 + 0.9 × 0 - 0] = 0.5
    try agent.update(0, 1, 1.0, 1, 0, false);
    try std.testing.expectEqual(0.5, agent.getQ(0, 1));

    // Set Q(1,0) = 2.0 for next update
    agent.setQ(1, 0, 2.0);

    // Update again: s=0, a=1, r=1, s'=1, a'=0, done=false
    // SARSA uses actual next action a'=0: Q(1,0) = 2.0
    // Q(0,1) ← 0.5 + 0.5 × [1 + 0.9 × 2.0 - 0.5]
    //        = 0.5 + 0.5 × [1 + 1.8 - 0.5]
    //        = 0.5 + 0.5 × 2.3
    //        = 0.5 + 1.15
    //        = 1.65
    try agent.update(0, 1, 1.0, 1, 0, false);
    try std.testing.expectApproxEqAbs(1.65, agent.getQ(0, 1), 1e-10);
}

test "SARSA: terminal state handling" {
    const allocator = std.testing.allocator;

    var agent = try SARSA(f64).init(allocator, 2, 1, 0.5, 0.9, 0.0, 1.0, 0.0, 42);
    defer agent.deinit();

    // Update with terminal state (done=true)
    // Q(0,0) ← 0 + 0.5 × [10 + 0.9 × 0 - 0] = 5.0
    try agent.update(0, 0, 10.0, 1, 0, true);
    try std.testing.expectEqual(5.0, agent.getQ(0, 0));
}

test "SARSA: epsilon decay" {
    const allocator = std.testing.allocator;

    var agent = try SARSA(f64).init(allocator, 1, 1, 0.1, 0.9, 1.0, 0.9, 0.1, 42);
    defer agent.deinit();

    try std.testing.expectEqual(1.0, agent.epsilon);

    agent.decayEpsilon();
    try std.testing.expectEqual(0.9, agent.epsilon);

    agent.decayEpsilon();
    try std.testing.expectEqual(0.81, agent.epsilon);

    // Decay to minimum
    for (0..20) |_| {
        agent.decayEpsilon();
    }
    try std.testing.expectApproxEqAbs(0.1, agent.epsilon, 1e-10);
}

test "SARSA: state value function" {
    const allocator = std.testing.allocator;

    var agent = try SARSA(f64).init(allocator, 1, 3, 0.1, 0.9, 0.2, 1.0, 0.0, 42);
    defer agent.deinit();

    // Set Q-values: Q(0,0)=2, Q(0,1)=5, Q(0,2)=3
    agent.setQ(0, 0, 2.0);
    agent.setQ(0, 1, 5.0);
    agent.setQ(0, 2, 3.0);

    // V(0) = (1-0.2) × 5 + 0.2 × (2+5+3)/3 ≈ 4.67
    const state_value = agent.getStateValue(0);
    const expected: f64 = 0.8 * 5.0 + 0.2 * (2.0 + 5.0 + 3.0) / 3.0;
    try std.testing.expectApproxEqAbs(expected, state_value, 1e-10);
}

test "SARSA: error handling - invalid state" {
    const allocator = std.testing.allocator;

    var agent = try SARSA(f64).init(allocator, 2, 2, 0.1, 0.9, 0.1, 0.99, 0.01, 42);
    defer agent.deinit();

    try std.testing.expectError(error.InvalidState, agent.selectAction(5));
    try std.testing.expectError(error.InvalidState, agent.update(5, 0, 1.0, 1, 0, false));
    try std.testing.expectError(error.InvalidState, agent.update(0, 0, 1.0, 5, 0, false));
}

test "SARSA: error handling - invalid action" {
    const allocator = std.testing.allocator;

    var agent = try SARSA(f64).init(allocator, 2, 2, 0.1, 0.9, 0.1, 0.99, 0.01, 42);
    defer agent.deinit();

    try std.testing.expectError(error.InvalidAction, agent.update(0, 5, 1.0, 1, 0, false));
    try std.testing.expectError(error.InvalidAction, agent.update(0, 0, 1.0, 1, 5, false));
}

test "SARSA: error handling - invalid parameters" {
    const allocator = std.testing.allocator;

    try std.testing.expectError(error.InvalidNumStates, SARSA(f64).init(allocator, 0, 2, 0.1, 0.9, 0.1, 0.99, 0.01, 42));
    try std.testing.expectError(error.InvalidNumActions, SARSA(f64).init(allocator, 2, 0, 0.1, 0.9, 0.1, 0.99, 0.01, 42));
    try std.testing.expectError(error.InvalidLearningRate, SARSA(f64).init(allocator, 2, 2, 0.0, 0.9, 0.1, 0.99, 0.01, 42));
    try std.testing.expectError(error.InvalidLearningRate, SARSA(f64).init(allocator, 2, 2, 1.5, 0.9, 0.1, 0.99, 0.01, 42));
    try std.testing.expectError(error.InvalidDiscountFactor, SARSA(f64).init(allocator, 2, 2, 0.1, -0.1, 0.1, 0.99, 0.01, 42));
    try std.testing.expectError(error.InvalidDiscountFactor, SARSA(f64).init(allocator, 2, 2, 0.1, 1.5, 0.1, 0.99, 0.01, 42));
}

test "SARSA: f32 support" {
    const allocator = std.testing.allocator;

    var agent = try SARSA(f32).init(allocator, 2, 2, 0.5, 0.9, 0.1, 0.99, 0.01, 42);
    defer agent.deinit();

    try agent.update(0, 1, 1.0, 1, 0, false);
    try std.testing.expectEqual(@as(f32, 0.5), agent.getQ(0, 1));
}

test "SARSA: large state-action space" {
    const allocator = std.testing.allocator;

    var agent = try SARSA(f64).init(allocator, 100, 10, 0.1, 0.95, 0.2, 0.99, 0.01, 777);
    defer agent.deinit();

    // Test random updates
    for (0..50) |i| {
        const s = i % 100;
        const a = i % 10;
        const r: f64 = @floatFromInt(i % 5);
        const s_next = (i + 1) % 100;
        const a_next = (i + 1) % 10;
        const done = i % 20 == 0;

        try agent.update(s, a, r, s_next, a_next, done);
    }

    // Should complete without crashes
    try std.testing.expect(agent.num_states == 100);
    try std.testing.expect(agent.num_actions == 10);
}

test "SARSA: convergence validation" {
    const allocator = std.testing.allocator;

    // Simple deterministic environment: state 0 → state 1 (reward +5)
    var agent = try SARSA(f64).init(allocator, 2, 1, 0.1, 0.9, 0.0, 1.0, 0.0, 42);
    defer agent.deinit();

    // Repeat updates until convergence
    for (0..100) |_| {
        try agent.update(0, 0, 5.0, 1, 0, true);
    }

    // Should converge to reward value (5.0) with learning rate 0.1
    const final_q = agent.getQ(0, 0);
    try std.testing.expectApproxEqAbs(5.0, final_q, 0.1);
}

test "SARSA: memory safety" {
    const allocator = std.testing.allocator;

    var agent = try SARSA(f64).init(allocator, 10, 5, 0.1, 0.9, 0.1, 0.99, 0.01, 42);
    defer agent.deinit();

    // Perform many operations to stress-test memory
    for (0..1000) |i| {
        const s = i % 10;
        const a = i % 5;
        const s_next = (i + 1) % 10;
        const a_next = (i + 1) % 5;
        try agent.update(s, a, 1.0, s_next, a_next, false);
    }

    // All Q-values should remain finite
    for (0..agent.num_states) |s| {
        for (0..agent.num_actions) |a| {
            const q = agent.getQ(s, a);
            try std.testing.expect(std.math.isFinite(q));
        }
    }
}
