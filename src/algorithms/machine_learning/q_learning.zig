/// Q-Learning
///
/// Model-free reinforcement learning algorithm that learns optimal action-value function Q(s,a).
/// Uses temporal difference (TD) learning to iteratively update Q-values based on observed rewards.
///
/// Algorithm:
/// - Initialize Q(s,a) arbitrarily
/// - For each episode:
///   - Select action using ε-greedy policy (explore vs exploit)
///   - Take action, observe reward r and next state s'
///   - Update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
/// - α = learning rate, γ = discount factor
///
/// Key features:
/// - Off-policy learning (learns optimal policy regardless of behavior policy)
/// - Temporal difference (TD) updates (bootstrap from estimated values)
/// - Epsilon-greedy exploration (balance exploration vs exploitation)
/// - Convergence guarantees with proper α schedule and sufficient exploration
///
/// Time complexity:
/// - Update: O(|A|) per step (requires max over all actions)
/// - Episode: O(T × |A|) where T = episode length
///
/// Space complexity:
/// - O(|S| × |A|) for Q-table
///
/// Use cases:
/// - Robotics (navigation, control)
/// - Game playing (board games, video games)
/// - Resource allocation (network routing, job scheduling)
/// - Finance (trading strategies)
/// - Operations research (inventory management)
///
/// Trade-offs:
/// - vs SARSA: Off-policy (learns optimal even with exploratory actions), but can be more unstable
/// - vs Policy Gradient: Sample efficient for discrete actions, but requires discrete state/action spaces
/// - vs Deep Q-Learning: Handles tabular problems exactly, no function approximation errors
/// - vs Monte Carlo: Lower variance via bootstrapping, but introduces bias
///
/// References:
/// - Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Q-Learning agent for discrete state/action spaces
///
/// Type parameters:
/// - T: Numeric type for Q-values (f32 or f64)
pub fn QLearning(comptime T: type) type {
    if (T != f32 and T != f64) {
        @compileError("QLearning only supports f32 and f64");
    }

    return struct {
        const Self = @This();

        /// Q-table: Q[state][action] = expected return
        q_table: [][]T,
        /// Number of states
        num_states: usize,
        /// Number of actions
        num_actions: usize,
        /// Learning rate (α) - step size for updates
        learning_rate: T,
        /// Discount factor (γ) - importance of future rewards
        discount_factor: T,
        /// Exploration rate (ε) - probability of random action
        epsilon: T,
        /// Random number generator
        prng: std.Random.DefaultPrng,
        /// Allocator
        allocator: Allocator,

        /// Initialize Q-Learning agent
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - num_states: Number of discrete states
        /// - num_actions: Number of discrete actions
        /// - learning_rate: Learning rate α ∈ (0, 1], typically 0.1-0.5
        /// - discount_factor: Discount factor γ ∈ [0, 1], typically 0.9-0.99
        /// - epsilon: Exploration rate ε ∈ [0, 1], typically 0.1
        /// - seed: Random seed for reproducibility
        ///
        /// Time: O(|S| × |A|) for initialization
        /// Space: O(|S| × |A|)
        pub fn init(
            allocator: Allocator,
            num_states: usize,
            num_actions: usize,
            learning_rate: T,
            discount_factor: T,
            epsilon: T,
            seed: u64,
        ) !Self {
            if (num_states == 0) return error.InvalidStates;
            if (num_actions == 0) return error.InvalidActions;
            if (learning_rate <= 0 or learning_rate > 1) return error.InvalidLearningRate;
            if (discount_factor < 0 or discount_factor > 1) return error.InvalidDiscountFactor;
            if (epsilon < 0 or epsilon > 1) return error.InvalidEpsilon;

            // Allocate Q-table
            const q_table = try allocator.alloc([]T, num_states);
            errdefer allocator.free(q_table);

            for (q_table, 0..) |*row, i| {
                row.* = try allocator.alloc(T, num_actions);
                errdefer {
                    // Free previously allocated rows on error
                    for (q_table[0..i]) |prev_row| {
                        allocator.free(prev_row);
                    }
                }
                // Initialize Q-values to 0 (optimistic initialization can use small positive values)
                @memset(row.*, 0);
            }

            return Self{
                .q_table = q_table,
                .num_states = num_states,
                .num_actions = num_actions,
                .learning_rate = learning_rate,
                .discount_factor = discount_factor,
                .epsilon = epsilon,
                .prng = std.Random.DefaultPrng.init(seed),
                .allocator = allocator,
            };
        }

        /// Free resources
        pub fn deinit(self: *Self) void {
            for (self.q_table) |row| {
                self.allocator.free(row);
            }
            self.allocator.free(self.q_table);
        }

        /// Select action using ε-greedy policy
        ///
        /// With probability ε, select random action (exploration)
        /// With probability 1-ε, select greedy action (exploitation)
        ///
        /// Time: O(|A|) for finding max Q-value
        pub fn selectAction(self: *Self, state: usize) !usize {
            if (state >= self.num_states) return error.InvalidState;

            const random = self.prng.random();

            // Exploration: random action
            if (random.float(T) < self.epsilon) {
                return random.intRangeLessThan(usize, 0, self.num_actions);
            }

            // Exploitation: greedy action (argmax_a Q(s,a))
            return self.greedyAction(state);
        }

        /// Get greedy action (argmax_a Q(s,a))
        ///
        /// Time: O(|A|)
        pub fn greedyAction(self: *Self, state: usize) !usize {
            if (state >= self.num_states) return error.InvalidState;

            var best_action: usize = 0;
            var best_value: T = self.q_table[state][0];

            for (self.q_table[state][1..], 1..) |value, action| {
                if (value > best_value) {
                    best_value = value;
                    best_action = action;
                }
            }

            return best_action;
        }

        /// Update Q-value using observed transition
        ///
        /// Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        ///
        /// Parameters:
        /// - state: Current state s
        /// - action: Action taken a
        /// - reward: Observed reward r
        /// - next_state: Next state s'
        /// - done: Whether episode terminated (if true, no future rewards)
        ///
        /// Time: O(|A|) for finding max Q(s',a')
        pub fn update(
            self: *Self,
            state: usize,
            action: usize,
            reward: T,
            next_state: usize,
            done: bool,
        ) !void {
            if (state >= self.num_states) return error.InvalidState;
            if (next_state >= self.num_states) return error.InvalidNextState;
            if (action >= self.num_actions) return error.InvalidAction;

            // Current Q-value
            const q_current = self.q_table[state][action];

            // Maximum Q-value for next state (or 0 if terminal)
            var q_next_max: T = 0;
            if (!done) {
                q_next_max = self.q_table[next_state][0];
                for (self.q_table[next_state][1..]) |value| {
                    if (value > q_next_max) {
                        q_next_max = value;
                    }
                }
            }

            // TD target: r + γ max_a' Q(s',a')
            const td_target = reward + self.discount_factor * q_next_max;

            // TD error: δ = target - current
            const td_error = td_target - q_current;

            // Update: Q(s,a) ← Q(s,a) + α × δ
            self.q_table[state][action] = q_current + self.learning_rate * td_error;
        }

        /// Get Q-value for state-action pair
        ///
        /// Time: O(1)
        pub fn getQValue(self: *const Self, state: usize, action: usize) !T {
            if (state >= self.num_states) return error.InvalidState;
            if (action >= self.num_actions) return error.InvalidAction;
            return self.q_table[state][action];
        }

        /// Set Q-value for state-action pair (for testing/initialization)
        ///
        /// Time: O(1)
        pub fn setQValue(self: *Self, state: usize, action: usize, value: T) !void {
            if (state >= self.num_states) return error.InvalidState;
            if (action >= self.num_actions) return error.InvalidAction;
            self.q_table[state][action] = value;
        }

        /// Decay exploration rate (anneal ε over time)
        ///
        /// Common schedules:
        /// - Linear: ε = max(ε_min, ε - decay)
        /// - Exponential: ε = max(ε_min, ε × decay)
        ///
        /// Time: O(1)
        pub fn decayEpsilon(self: *Self, decay: T, min_epsilon: T) void {
            self.epsilon = @max(min_epsilon, self.epsilon * decay);
        }

        /// Get value function V(s) = max_a Q(s,a)
        ///
        /// Time: O(|A|)
        pub fn getStateValue(self: *const Self, state: usize) !T {
            if (state >= self.num_states) return error.InvalidState;

            var max_value = self.q_table[state][0];
            for (self.q_table[state][1..]) |value| {
                if (value > max_value) {
                    max_value = value;
                }
            }
            return max_value;
        }
    };
}

// ===== Tests =====

test "QLearning: basic initialization" {
    const allocator = std.testing.allocator;

    var agent = try QLearning(f64).init(allocator, 4, 2, 0.1, 0.9, 0.1, 42);
    defer agent.deinit();

    try std.testing.expectEqual(@as(usize, 4), agent.num_states);
    try std.testing.expectEqual(@as(usize, 2), agent.num_actions);
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), agent.learning_rate, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9), agent.discount_factor, 1e-10);

    // Q-values should be initialized to 0
    for (0..4) |s| {
        for (0..2) |a| {
            const q = try agent.getQValue(s, a);
            try std.testing.expectApproxEqAbs(@as(f64, 0), q, 1e-10);
        }
    }
}

test "QLearning: simple gridworld" {
    const allocator = std.testing.allocator;

    // 2×2 gridworld: states 0,1,2,3 (left-to-right, top-to-bottom)
    // Actions: 0=up, 1=down, 2=left, 3=right
    // Goal: state 3, reward +1, terminal
    var agent = try QLearning(f64).init(allocator, 4, 4, 0.5, 0.9, 0.0, 42); // ε=0 for deterministic
    defer agent.deinit();

    // Train: state 2 → right → state 3 (goal)
    try agent.update(2, 3, 1.0, 3, true); // reward +1, terminal
    const q_2_3 = try agent.getQValue(2, 3);
    try std.testing.expect(q_2_3 > 0); // Should learn positive value

    // Train: state 1 → down → state 3 (goal)
    try agent.update(1, 1, 1.0, 3, true);
    const q_1_1 = try agent.getQValue(1, 1);
    try std.testing.expect(q_1_1 > 0);

    // Train: state 0 → right → state 1 → down → state 3
    try agent.update(0, 3, 0.0, 1, false); // no immediate reward
    const q_0_3 = try agent.getQValue(0, 3);
    try std.testing.expect(q_0_3 > 0); // Should learn from bootstrapping V(1)
}

test "QLearning: epsilon-greedy action selection" {
    const allocator = std.testing.allocator;

    var agent = try QLearning(f64).init(allocator, 2, 3, 0.1, 0.9, 0.5, 42);
    defer agent.deinit();

    // Set Q-values: action 1 is best
    try agent.setQValue(0, 0, 0.1);
    try agent.setQValue(0, 1, 0.9);
    try agent.setQValue(0, 2, 0.3);

    // With ε=0.5, should sometimes select action 1 (greedy), sometimes others (explore)
    var action_counts = [_]usize{0} ** 3;
    for (0..1000) |_| {
        const action = try agent.selectAction(0);
        action_counts[action] += 1;
    }

    // Action 1 should be selected most often (greedy + some exploration)
    try std.testing.expect(action_counts[1] > action_counts[0]);
    try std.testing.expect(action_counts[1] > action_counts[2]);

    // But other actions should also be selected (exploration)
    try std.testing.expect(action_counts[0] > 0);
    try std.testing.expect(action_counts[2] > 0);
}

test "QLearning: greedy action selection" {
    const allocator = std.testing.allocator;

    var agent = try QLearning(f64).init(allocator, 2, 3, 0.1, 0.9, 0.1, 42);
    defer agent.deinit();

    // Set Q-values: action 2 is best
    try agent.setQValue(0, 0, 0.1);
    try agent.setQValue(0, 1, 0.3);
    try agent.setQValue(0, 2, 0.9);

    const action = try agent.greedyAction(0);
    try std.testing.expectEqual(@as(usize, 2), action);
}

test "QLearning: TD update" {
    const allocator = std.testing.allocator;

    var agent = try QLearning(f64).init(allocator, 3, 2, 0.5, 0.9, 0.1, 42);
    defer agent.deinit();

    // Set next state Q-values
    try agent.setQValue(1, 0, 0.5);
    try agent.setQValue(1, 1, 0.8); // max

    // Update Q(0,0): r=1, γ=0.9, max Q(1,a)=0.8
    // TD target = 1 + 0.9 × 0.8 = 1.72
    // Q(0,0) ← 0 + 0.5 × (1.72 - 0) = 0.86
    try agent.update(0, 0, 1.0, 1, false);
    const q = try agent.getQValue(0, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.86), q, 1e-10);
}

test "QLearning: terminal state update" {
    const allocator = std.testing.allocator;

    var agent = try QLearning(f64).init(allocator, 2, 2, 0.5, 0.9, 0.1, 42);
    defer agent.deinit();

    // Terminal state: no future rewards
    // TD target = r + 0 = 10
    // Q(0,0) ← 0 + 0.5 × (10 - 0) = 5
    try agent.update(0, 0, 10.0, 1, true);
    const q = try agent.getQValue(0, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), q, 1e-10);
}

test "QLearning: epsilon decay" {
    const allocator = std.testing.allocator;

    var agent = try QLearning(f64).init(allocator, 2, 2, 0.1, 0.9, 1.0, 42);
    defer agent.deinit();

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), agent.epsilon, 1e-10);

    agent.decayEpsilon(0.9, 0.01); // exponential decay
    try std.testing.expectApproxEqAbs(@as(f64, 0.9), agent.epsilon, 1e-10);

    agent.decayEpsilon(0.9, 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 0.81), agent.epsilon, 1e-10);

    // Should not go below min
    for (0..100) |_| {
        agent.decayEpsilon(0.9, 0.01);
    }
    try std.testing.expect(agent.epsilon >= 0.01);
}

test "QLearning: state value function" {
    const allocator = std.testing.allocator;

    var agent = try QLearning(f64).init(allocator, 2, 3, 0.1, 0.9, 0.1, 42);
    defer agent.deinit();

    try agent.setQValue(0, 0, 0.3);
    try agent.setQValue(0, 1, 0.9); // max
    try agent.setQValue(0, 2, 0.5);

    const v = try agent.getStateValue(0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9), v, 1e-10);
}

test "QLearning: error handling - invalid states" {
    const allocator = std.testing.allocator;

    try std.testing.expectError(error.InvalidStates, QLearning(f64).init(allocator, 0, 2, 0.1, 0.9, 0.1, 42));
}

test "QLearning: error handling - invalid actions" {
    const allocator = std.testing.allocator;

    try std.testing.expectError(error.InvalidActions, QLearning(f64).init(allocator, 2, 0, 0.1, 0.9, 0.1, 42));
}

test "QLearning: error handling - invalid learning rate" {
    const allocator = std.testing.allocator;

    try std.testing.expectError(error.InvalidLearningRate, QLearning(f64).init(allocator, 2, 2, 0.0, 0.9, 0.1, 42));
    try std.testing.expectError(error.InvalidLearningRate, QLearning(f64).init(allocator, 2, 2, 1.5, 0.9, 0.1, 42));
}

test "QLearning: error handling - invalid discount factor" {
    const allocator = std.testing.allocator;

    try std.testing.expectError(error.InvalidDiscountFactor, QLearning(f64).init(allocator, 2, 2, 0.1, -0.1, 0.1, 42));
    try std.testing.expectError(error.InvalidDiscountFactor, QLearning(f64).init(allocator, 2, 2, 0.1, 1.1, 0.1, 42));
}

test "QLearning: error handling - invalid epsilon" {
    const allocator = std.testing.allocator;

    try std.testing.expectError(error.InvalidEpsilon, QLearning(f64).init(allocator, 2, 2, 0.1, 0.9, -0.1, 42));
    try std.testing.expectError(error.InvalidEpsilon, QLearning(f64).init(allocator, 2, 2, 0.1, 0.9, 1.1, 42));
}

test "QLearning: error handling - out of bounds state" {
    const allocator = std.testing.allocator;

    var agent = try QLearning(f64).init(allocator, 2, 2, 0.1, 0.9, 0.1, 42);
    defer agent.deinit();

    try std.testing.expectError(error.InvalidState, agent.selectAction(2));
    try std.testing.expectError(error.InvalidState, agent.getQValue(2, 0));
    try std.testing.expectError(error.InvalidState, agent.update(2, 0, 0, 1, false));
}

test "QLearning: error handling - out of bounds action" {
    const allocator = std.testing.allocator;

    var agent = try QLearning(f64).init(allocator, 2, 2, 0.1, 0.9, 0.1, 42);
    defer agent.deinit();

    try std.testing.expectError(error.InvalidAction, agent.getQValue(0, 2));
    try std.testing.expectError(error.InvalidAction, agent.update(0, 2, 0, 1, false));
}

test "QLearning: f32 support" {
    const allocator = std.testing.allocator;

    var agent = try QLearning(f32).init(allocator, 2, 2, 0.1, 0.9, 0.1, 42);
    defer agent.deinit();

    try agent.setQValue(0, 0, 0.5);
    const q = try agent.getQValue(0, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), q, 1e-6);
}

test "QLearning: large state-action space" {
    const allocator = std.testing.allocator;

    var agent = try QLearning(f64).init(allocator, 100, 10, 0.1, 0.9, 0.1, 42);
    defer agent.deinit();

    // Set and retrieve Q-values
    try agent.setQValue(50, 5, 0.75);
    const q = try agent.getQValue(50, 5);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), q, 1e-10);

    // Update and verify
    try agent.update(50, 5, 1.0, 60, false);
    const q_updated = try agent.getQValue(50, 5);
    try std.testing.expect(q_updated > 0.75); // Should increase
}

test "QLearning: convergence on simple task" {
    const allocator = std.testing.allocator;

    // Simple 2-state MDP: state 0 → action 0 → state 1 (reward +1, terminal)
    var agent = try QLearning(f64).init(allocator, 2, 2, 0.5, 0.9, 0.0, 42);
    defer agent.deinit();

    // Train for multiple episodes
    for (0..50) |_| {
        try agent.update(0, 0, 1.0, 1, true);
    }

    // Q(0,0) should converge to 1.0 (immediate reward, no future)
    const q = try agent.getQValue(0, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), q, 0.01);
}

test "QLearning: memory safety" {
    const allocator = std.testing.allocator;

    var agent = try QLearning(f64).init(allocator, 10, 5, 0.1, 0.9, 0.1, 42);
    defer agent.deinit();

    // Perform multiple operations to stress test memory management
    for (0..100) |i| {
        const s = i % 10;
        const a = try agent.selectAction(s);
        const next_s = (s + 1) % 10;
        try agent.update(s, a, 0.1, next_s, false);
    }
}
