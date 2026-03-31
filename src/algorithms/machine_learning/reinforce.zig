//! REINFORCE (Monte Carlo Policy Gradient)
//!
//! REINFORCE is the foundational policy gradient algorithm for reinforcement learning.
//! Unlike value-based methods (Q-Learning, SARSA), it directly optimizes a parameterized
//! policy using gradient ascent on expected returns.
//!
//! Algorithm:
//! 1. Initialize policy parameters (preferences) randomly
//! 2. Generate an episode following policy π(a|s)
//! 3. For each step t in episode:
//!    - Compute return G_t = Σ_{k=t}^{T} γ^{k-t} r_k (cumulative discounted reward)
//!    - Update policy: θ ← θ + α G_t ∇_θ log π(a_t|s_t)
//! 4. Repeat until convergence
//!
//! Key features:
//! - Policy gradient: Direct policy optimization (not value-based)
//! - Monte Carlo: Uses complete episode returns (high variance but unbiased)
//! - Stochastic policy: π(a|s) via softmax over action preferences
//! - Gradient ascent: ∇_θ log π(a|s) × G (REINFORCE trick)
//! - Temperature parameter: Controls exploration vs exploitation
//! - Baseline optional: Reduces variance (use Actor-Critic instead for full baseline)
//!
//! Time complexity: O(|A| × T) per episode where T = episode length
//! Space complexity: O(|S| × |A|) for policy parameters (preferences)
//!
//! Use cases:
//! - Continuous action spaces (after extension to function approximation)
//! - Stochastic policies required (e.g., rock-paper-scissors)
//! - Exploration via policy entropy
//! - Foundation for advanced algorithms (A2C, PPO, TRPO)
//!
//! Trade-offs:
//! - vs Q-Learning: Can handle continuous actions, but high variance, slow convergence
//! - vs Actor-Critic: Simpler (no critic), but much higher variance (full returns)
//! - vs SARSA: Policy gradient (more principled), but sample inefficient
//! - Baseline reduction: Can subtract baseline b(s) from returns (Actor-Critic does this)

const std = @import("std");
const Allocator = std.mem.Allocator;

/// REINFORCE algorithm configuration
pub const Config = struct {
    /// Learning rate (step size) for gradient ascent
    learning_rate: f64 = 0.01,
    /// Discount factor for future rewards (γ ∈ [0,1])
    gamma: f64 = 0.99,
    /// Temperature parameter for softmax policy (higher = more exploration)
    temperature: f64 = 1.0,
    /// Maximum episodes for training
    max_episodes: usize = 1000,
    /// Maximum steps per episode
    max_steps_per_episode: usize = 1000,
};

/// Episode step record
const Step = struct {
    state: usize,
    action: usize,
    reward: f64,
};

/// REINFORCE agent
///
/// Uses Monte Carlo policy gradient for learning optimal stochastic policies.
///
/// Example:
/// ```zig
/// const config = Config{ .learning_rate = 0.1, .gamma = 0.95 };
/// var agent = try REINFORCE(f64).init(allocator, n_states, n_actions, config);
/// defer agent.deinit();
///
/// // Generate episode and learn
/// var episode = std.ArrayList(Step).init(allocator);
/// defer episode.deinit();
/// // ... collect episode steps ...
/// try agent.learn(&episode);
/// ```
pub fn REINFORCE(comptime T: type) type {
    if (T != f32 and T != f64) {
        @compileError("REINFORCE only supports f32 and f64 types");
    }

    return struct {
        const Self = @This();

        allocator: Allocator,
        /// Policy preferences: theta[s][a] (unnormalized log probabilities)
        preferences: [][]T,
        n_states: usize,
        n_actions: usize,
        config: Config,

        /// Initialize REINFORCE agent
        ///
        /// Time: O(|S| × |A|)
        /// Space: O(|S| × |A|)
        pub fn init(allocator: Allocator, n_states: usize, n_actions: usize, config: Config) !Self {
            if (n_states == 0) return error.InvalidNumStates;
            if (n_actions == 0) return error.InvalidNumActions;
            if (config.learning_rate <= 0) return error.InvalidLearningRate;
            if (config.gamma < 0 or config.gamma > 1) return error.InvalidGamma;
            if (config.temperature <= 0) return error.InvalidTemperature;

            const preferences = try allocator.alloc([]T, n_states);
            errdefer allocator.free(preferences);

            for (preferences, 0..) |*pref_row, i| {
                pref_row.* = try allocator.alloc(T, n_actions);
                errdefer {
                    for (preferences[0..i]) |row| allocator.free(row);
                    allocator.free(preferences);
                }

                // Initialize preferences to small random values (near uniform policy)
                for (pref_row.*) |*val| {
                    val.* = 0.0; // Start with uniform policy
                }
            }

            return Self{
                .allocator = allocator,
                .preferences = preferences,
                .n_states = n_states,
                .n_actions = n_actions,
                .config = config,
            };
        }

        /// Free resources
        ///
        /// Time: O(|S|)
        /// Space: O(1)
        pub fn deinit(self: *Self) void {
            for (self.preferences) |row| {
                self.allocator.free(row);
            }
            self.allocator.free(self.preferences);
        }

        /// Get policy distribution π(·|s) for a state
        ///
        /// Returns probabilities for each action via softmax.
        ///
        /// Time: O(|A|)
        /// Space: O(1)
        pub fn getPolicy(self: Self, state: usize, probs: []T) !void {
            if (state >= self.n_states) return error.InvalidState;
            if (probs.len != self.n_actions) return error.InvalidProbsLength;

            const temp: T = @floatCast(self.config.temperature);

            // Compute softmax: π(a|s) = exp(θ(s,a)/τ) / Σ_a' exp(θ(s,a')/τ)
            var max_pref: T = self.preferences[state][0];
            for (self.preferences[state][1..]) |pref| {
                max_pref = @max(max_pref, pref);
            }

            var sum: T = 0.0;
            for (self.preferences[state], 0..) |pref, a| {
                probs[a] = @exp((pref - max_pref) / temp); // Numerical stability
                sum += probs[a];
            }

            // Normalize
            for (probs) |*p| {
                p.* /= sum;
            }
        }

        /// Select action from policy π(·|s)
        ///
        /// Uses softmax probabilities for stochastic sampling.
        ///
        /// Time: O(|A|)
        /// Space: O(1)
        pub fn selectAction(self: Self, state: usize) !usize {
            if (state >= self.n_states) return error.InvalidState;

            const probs = try self.allocator.alloc(T, self.n_actions);
            defer self.allocator.free(probs);

            try self.getPolicy(state, probs);

            // Sample from categorical distribution
            const rand = std.crypto.random;
            const r: T = @floatCast(rand.float(f64));
            var cumulative: T = 0.0;
            for (probs, 0..) |p, a| {
                cumulative += p;
                if (r <= cumulative) return a;
            }

            return self.n_actions - 1; // Fallback due to floating point errors
        }

        /// Select greedy action (highest probability)
        ///
        /// Time: O(|A|)
        /// Space: O(1)
        pub fn greedyAction(self: Self, state: usize) !usize {
            if (state >= self.n_states) return error.InvalidState;

            var best_action: usize = 0;
            var best_pref = self.preferences[state][0];
            for (self.preferences[state][1..], 1..) |pref, a| {
                if (pref > best_pref) {
                    best_pref = pref;
                    best_action = a;
                }
            }
            return best_action;
        }

        /// Learn from a complete episode using REINFORCE algorithm
        ///
        /// Updates policy parameters using Monte Carlo returns.
        ///
        /// Algorithm:
        /// 1. Compute returns G_t for each step (backwards)
        /// 2. For each step: θ(s,a) ← θ(s,a) + α G_t ∇log π(a|s)
        /// 3. Gradient: ∇log π(a|s) = e_a - π(·|s) where e_a is one-hot
        ///
        /// Time: O(T × |A|) where T = episode length
        /// Space: O(T + |A|)
        pub fn learn(self: *Self, episode: *const std.ArrayList(Step)) !void {
            if (episode.items.len == 0) return; // Nothing to learn

            const T_ep = episode.items.len;

            // Compute returns G_t for each step (backwards accumulation)
            const returns = try self.allocator.alloc(T, T_ep);
            defer self.allocator.free(returns);

            var G: T = 0.0;
            const gamma: T = @floatCast(self.config.gamma);
            var t: usize = T_ep;
            while (t > 0) {
                t -= 1;
                G = @as(T, @floatCast(episode.items[t].reward)) + gamma * G;
                returns[t] = G;
            }

            // Update policy parameters using gradient ascent
            const alpha: T = @floatCast(self.config.learning_rate);
            const temp: T = @floatCast(self.config.temperature);

            const probs = try self.allocator.alloc(T, self.n_actions);
            defer self.allocator.free(probs);

            for (episode.items, 0..) |step, idx| {
                if (step.state >= self.n_states) return error.InvalidState;
                if (step.action >= self.n_actions) return error.InvalidAction;

                try self.getPolicy(step.state, probs);

                // Policy gradient: ∇log π(a|s) = (e_a - π(·|s)) / τ
                // Update: θ(s,a') ← θ(s,a') + α G_t (e_a - π(a'|s)) / τ
                const G_t = returns[idx];
                for (probs, 0..) |prob, a| {
                    const indicator: T = if (a == step.action) 1.0 else 0.0;
                    const gradient = (indicator - prob) / temp;
                    self.preferences[step.state][a] += alpha * G_t * gradient;
                }
            }
        }

        /// Compute state value V(s) = E_π[G|s] under current policy
        ///
        /// This is the expected return starting from state s following policy π.
        /// Useful for monitoring learning progress.
        ///
        /// Time: O(|A|)
        /// Space: O(1)
        pub fn stateValue(self: Self, state: usize, Q_estimates: []const []const T) !T {
            if (state >= self.n_states) return error.InvalidState;
            if (Q_estimates.len != self.n_states) return error.InvalidQTable;
            if (Q_estimates[state].len != self.n_actions) return error.InvalidQTable;

            const probs = try self.allocator.alloc(T, self.n_actions);
            defer self.allocator.free(probs);

            try self.getPolicy(state, probs);

            var value: T = 0.0;
            for (probs, 0..) |p, a| {
                value += p * Q_estimates[state][a];
            }
            return value;
        }

        /// Reset policy to uniform (for retraining)
        ///
        /// Time: O(|S| × |A|)
        /// Space: O(1)
        pub fn reset(self: *Self) void {
            for (self.preferences) |row| {
                for (row) |*val| {
                    val.* = 0.0;
                }
            }
        }
    };
}

// Tests
const testing = std.testing;

test "REINFORCE: initialization" {
    const allocator = testing.allocator;

    var agent = try REINFORCE(f64).init(allocator, 3, 4, .{});
    defer agent.deinit();

    try testing.expectEqual(@as(usize, 3), agent.n_states);
    try testing.expectEqual(@as(usize, 4), agent.n_actions);
    try testing.expectEqual(@as(f64, 0.99), agent.config.gamma);
}

test "REINFORCE: uniform initial policy" {
    const allocator = testing.allocator;

    var agent = try REINFORCE(f64).init(allocator, 2, 3, .{});
    defer agent.deinit();

    const probs = try allocator.alloc(f64, 3);
    defer allocator.free(probs);

    try agent.getPolicy(0, probs);

    // All preferences are 0, so softmax should be uniform
    const expected: f64 = 1.0 / 3.0;
    for (probs) |p| {
        try testing.expectApproxEqAbs(expected, p, 1e-6);
    }
}

test "REINFORCE: policy distribution validation" {
    const allocator = testing.allocator;

    var agent = try REINFORCE(f64).init(allocator, 2, 3, .{});
    defer agent.deinit();

    // Manually set preferences for state 0
    agent.preferences[0][0] = 1.0;
    agent.preferences[0][1] = 2.0;
    agent.preferences[0][2] = 0.0;

    const probs = try allocator.alloc(f64, 3);
    defer allocator.free(probs);

    try agent.getPolicy(0, probs);

    // Probabilities should sum to 1
    var sum: f64 = 0.0;
    for (probs) |p| {
        sum += p;
    }
    try testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-6);

    // Higher preference should have higher probability
    try testing.expect(probs[1] > probs[0]);
    try testing.expect(probs[0] > probs[2]);
}

test "REINFORCE: temperature effects" {
    const allocator = testing.allocator;

    var agent_low = try REINFORCE(f64).init(allocator, 1, 3, .{ .temperature = 0.1 });
    defer agent_low.deinit();

    var agent_high = try REINFORCE(f64).init(allocator, 1, 3, .{ .temperature = 10.0 });
    defer agent_high.deinit();

    // Set preferences
    agent_low.preferences[0][0] = 1.0;
    agent_low.preferences[0][1] = 2.0;
    agent_low.preferences[0][2] = 0.0;

    agent_high.preferences[0][0] = 1.0;
    agent_high.preferences[0][1] = 2.0;
    agent_high.preferences[0][2] = 0.0;

    const probs_low = try allocator.alloc(f64, 3);
    defer allocator.free(probs_low);
    const probs_high = try allocator.alloc(f64, 3);
    defer allocator.free(probs_high);

    try agent_low.getPolicy(0, probs_low);
    try agent_high.getPolicy(0, probs_high);

    // Low temperature: more deterministic (higher max probability)
    const max_low = @max(@max(probs_low[0], probs_low[1]), probs_low[2]);
    const max_high = @max(@max(probs_high[0], probs_high[1]), probs_high[2]);
    try testing.expect(max_low > max_high);
}

test "REINFORCE: simple 2-state chain learning" {
    const allocator = testing.allocator;

    var agent = try REINFORCE(f64).init(allocator, 2, 2, .{
        .learning_rate = 0.1,
        .gamma = 0.9,
        .temperature = 1.0,
    });
    defer agent.deinit();

    // Episode: s0 -a0-> r=1 -> s1 (terminal)
    var episode = std.ArrayList(Step).init(allocator);
    defer episode.deinit();

    try episode.append(.{ .state = 0, .action = 0, .reward = 1.0 });

    try agent.learn(&episode);

    // After learning, preference for (s0, a0) should increase
    try testing.expect(agent.preferences[0][0] > 0.0);
}

test "REINFORCE: greedy action selection" {
    const allocator = testing.allocator;

    var agent = try REINFORCE(f64).init(allocator, 2, 3, .{});
    defer agent.deinit();

    // Set preferences for state 0
    agent.preferences[0][0] = 1.0;
    agent.preferences[0][1] = 3.0; // Best
    agent.preferences[0][2] = 0.5;

    const action = try agent.greedyAction(0);
    try testing.expectEqual(@as(usize, 1), action);
}

test "REINFORCE: return computation" {
    const allocator = testing.allocator;

    var agent = try REINFORCE(f64).init(allocator, 3, 2, .{
        .learning_rate = 0.1,
        .gamma = 0.9,
    });
    defer agent.deinit();

    // Episode: r1=1, r2=2 -> Returns: G0=1+0.9*2=2.8, G1=2
    var episode = std.ArrayList(Step).init(allocator);
    defer episode.deinit();

    try episode.append(.{ .state = 0, .action = 0, .reward = 1.0 });
    try episode.append(.{ .state = 1, .action = 1, .reward = 2.0 });

    const initial_pref = agent.preferences[0][0];
    try agent.learn(&episode);

    // Preferences should have changed
    try testing.expect(agent.preferences[0][0] != initial_pref or agent.preferences[1][1] != 0.0);
}

test "REINFORCE: policy convergence on simple task" {
    const allocator = testing.allocator;

    var agent = try REINFORCE(f64).init(allocator, 2, 2, .{
        .learning_rate = 0.5,
        .gamma = 0.9,
        .temperature = 1.0,
    });
    defer agent.deinit();

    // Train on: always choosing action 0 in state 0 gives reward 10
    for (0..100) |_| {
        var episode = std.ArrayList(Step).init(allocator);
        defer episode.deinit();

        try episode.append(.{ .state = 0, .action = 0, .reward = 10.0 });

        try agent.learn(&episode);
    }

    // After many episodes, action 0 should be strongly preferred
    try testing.expect(agent.preferences[0][0] > agent.preferences[0][1]);
}

test "REINFORCE: state value computation" {
    const allocator = testing.allocator;

    var agent = try REINFORCE(f64).init(allocator, 2, 3, .{});
    defer agent.deinit();

    // Set uniform policy
    const Q_table = try allocator.alloc([]f64, 2);
    defer allocator.free(Q_table);
    for (Q_table) |*row| {
        row.* = try allocator.alloc(f64, 3);
    }
    defer {
        for (Q_table) |row| allocator.free(row);
    }

    // Q(s0, a0) = 1, Q(s0, a1) = 2, Q(s0, a2) = 3
    Q_table[0][0] = 1.0;
    Q_table[0][1] = 2.0;
    Q_table[0][2] = 3.0;

    Q_table[1][0] = 0.0;
    Q_table[1][1] = 0.0;
    Q_table[1][2] = 0.0;

    const value = try agent.stateValue(0, Q_table);

    // V(s0) = π(a0|s0)*1 + π(a1|s0)*2 + π(a2|s0)*3 = 1/3 + 2/3 + 1 = 2 (uniform policy)
    try testing.expectApproxEqAbs(@as(f64, 2.0), value, 1e-6);
}

test "REINFORCE: f32 support" {
    const allocator = testing.allocator;

    var agent = try REINFORCE(f32).init(allocator, 3, 4, .{});
    defer agent.deinit();

    try testing.expectEqual(@as(usize, 3), agent.n_states);
    try testing.expectEqual(@as(usize, 4), agent.n_actions);

    const probs = try allocator.alloc(f32, 4);
    defer allocator.free(probs);

    try agent.getPolicy(0, probs);

    var sum: f32 = 0.0;
    for (probs) |p| sum += p;
    try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}

test "REINFORCE: error handling - invalid states" {
    const allocator = testing.allocator;

    var agent = try REINFORCE(f64).init(allocator, 2, 3, .{});
    defer agent.deinit();

    const probs = try allocator.alloc(f64, 3);
    defer allocator.free(probs);

    try testing.expectError(error.InvalidState, agent.getPolicy(5, probs));
    try testing.expectError(error.InvalidState, agent.selectAction(5));
    try testing.expectError(error.InvalidState, agent.greedyAction(5));
}

test "REINFORCE: error handling - invalid actions" {
    const allocator = testing.allocator;

    var agent = try REINFORCE(f64).init(allocator, 2, 3, .{});
    defer agent.deinit();

    var episode = std.ArrayList(Step).init(allocator);
    defer episode.deinit();

    try episode.append(.{ .state = 0, .action = 10, .reward = 1.0 });

    try testing.expectError(error.InvalidAction, agent.learn(&episode));
}

test "REINFORCE: error handling - invalid config" {
    const allocator = testing.allocator;

    try testing.expectError(error.InvalidNumStates, REINFORCE(f64).init(allocator, 0, 3, .{}));
    try testing.expectError(error.InvalidNumActions, REINFORCE(f64).init(allocator, 2, 0, .{}));
    try testing.expectError(error.InvalidLearningRate, REINFORCE(f64).init(allocator, 2, 3, .{ .learning_rate = -0.1 }));
    try testing.expectError(error.InvalidGamma, REINFORCE(f64).init(allocator, 2, 3, .{ .gamma = 1.5 }));
    try testing.expectError(error.InvalidTemperature, REINFORCE(f64).init(allocator, 2, 3, .{ .temperature = 0.0 }));
}

test "REINFORCE: reset functionality" {
    const allocator = testing.allocator;

    var agent = try REINFORCE(f64).init(allocator, 2, 3, .{});
    defer agent.deinit();

    // Modify preferences
    agent.preferences[0][0] = 5.0;
    agent.preferences[1][2] = -3.0;

    agent.reset();

    // All preferences should be 0
    for (agent.preferences) |row| {
        for (row) |val| {
            try testing.expectEqual(@as(f64, 0.0), val);
        }
    }
}

test "REINFORCE: large state-action space" {
    const allocator = testing.allocator;

    var agent = try REINFORCE(f64).init(allocator, 100, 10, .{});
    defer agent.deinit();

    try testing.expectEqual(@as(usize, 100), agent.n_states);
    try testing.expectEqual(@as(usize, 10), agent.n_actions);

    const probs = try allocator.alloc(f64, 10);
    defer allocator.free(probs);

    try agent.getPolicy(50, probs);

    var sum: f64 = 0.0;
    for (probs) |p| sum += p;
    try testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-6);
}

test "REINFORCE: multi-step episode learning" {
    const allocator = testing.allocator;

    var agent = try REINFORCE(f64).init(allocator, 4, 2, .{
        .learning_rate = 0.1,
        .gamma = 0.9,
    });
    defer agent.deinit();

    // Episode: s0 -a0-> r=1 -> s1 -a1-> r=2 -> s2 -a0-> r=3 -> terminal
    var episode = std.ArrayList(Step).init(allocator);
    defer episode.deinit();

    try episode.append(.{ .state = 0, .action = 0, .reward = 1.0 });
    try episode.append(.{ .state = 1, .action = 1, .reward = 2.0 });
    try episode.append(.{ .state = 2, .action = 0, .reward = 3.0 });

    const initial_pref_00 = agent.preferences[0][0];

    try agent.learn(&episode);

    // Preferences should change after learning
    try testing.expect(agent.preferences[0][0] != initial_pref_00);
}

test "REINFORCE: memory safety" {
    const allocator = testing.allocator;

    var agent = try REINFORCE(f64).init(allocator, 5, 3, .{});
    defer agent.deinit();

    var episode = std.ArrayList(Step).init(allocator);
    defer episode.deinit();

    try episode.append(.{ .state = 0, .action = 1, .reward = 5.0 });
    try episode.append(.{ .state = 1, .action = 0, .reward = 3.0 });

    try agent.learn(&episode);

    // No memory leaks should occur
}
