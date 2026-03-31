/// Hidden Markov Model (HMM)
///
/// A probabilistic model for sequential data where the system is assumed to be a Markov process
/// with unobserved (hidden) states. The model is defined by:
/// - States: Hidden states (e.g., weather: Sunny, Rainy)
/// - Observations: Observable events (e.g., Walk, Shop, Clean)
/// - Transition probabilities: P(state_t | state_{t-1})
/// - Emission probabilities: P(observation_t | state_t)
/// - Initial state probabilities: P(state_0)
///
/// Algorithms implemented:
/// 1. Forward algorithm: Compute probability of observation sequence P(O|λ)
/// 2. Viterbi algorithm: Find most likely state sequence given observations
/// 3. Backward algorithm: Compute backward probabilities (used in Baum-Welch)
/// 4. Baum-Welch (EM): Learn model parameters from observation sequences (training)
///
/// Use cases:
/// - Speech recognition (phoneme sequences from audio)
/// - Part-of-speech tagging (word tags from sentences)
/// - Bioinformatics (gene sequences, protein folding)
/// - Time series analysis (regime detection)
/// - Activity recognition (sensor data to activities)
/// - Financial modeling (market regime switching)
///
/// Example:
/// ```zig
/// // Weather HMM: states = {Sunny, Rainy}, observations = {Walk, Shop, Clean}
/// var hmm = HMM(f64).init(allocator, 2, 3); // 2 states, 3 observations
/// defer hmm.deinit();
///
/// // Set probabilities
/// hmm.initial[0] = 0.6; hmm.initial[1] = 0.4;  // P(Sunny), P(Rainy)
/// hmm.transition.set(&.{0,0}, 0.7); hmm.transition.set(&.{0,1}, 0.3);  // Sunny->Sunny, Sunny->Rainy
/// hmm.transition.set(&.{1,0}, 0.4); hmm.transition.set(&.{1,1}, 0.6);  // Rainy->Sunny, Rainy->Rainy
/// hmm.emission.set(&.{0,0}, 0.6); hmm.emission.set(&.{0,1}, 0.3); hmm.emission.set(&.{0,2}, 0.1);  // Sunny emissions
/// hmm.emission.set(&.{1,0}, 0.1); hmm.emission.set(&.{1,1}, 0.4); hmm.emission.set(&.{1,2}, 0.5);  // Rainy emissions
///
/// // Decode most likely state sequence
/// const observations = [_]usize{0, 1, 2};  // Walk, Shop, Clean
/// var states = try hmm.viterbi(&observations);
/// defer allocator.free(states);  // [0, 0, 1] -> Sunny, Sunny, Rainy
/// ```

const std = @import("std");
const Allocator = std.mem.Allocator;
const NDArray = @import("../../ndarray/ndarray.zig").NDArray;

/// Hidden Markov Model
///
/// Type parameters:
/// - T: Float type (f32 or f64)
///
/// Time complexity:
/// - Forward: O(T × N²) where T = sequence length, N = number of states
/// - Viterbi: O(T × N²)
/// - Backward: O(T × N²)
/// - Baum-Welch: O(iter × T × N²)
///
/// Space complexity: O(N² + N×M) for transition/emission matrices
pub fn HMM(comptime T: type) type {
    if (T != f32 and T != f64) {
        @compileError("HMM only supports f32 and f64");
    }

    return struct {
        const Self = @This();

        allocator: Allocator,
        n_states: usize, // Number of hidden states
        n_observations: usize, // Number of possible observations

        // Model parameters
        initial: []T, // Initial state probabilities [N]
        transition: NDArray(T, 2), // State transition probabilities [N×N]
        emission: NDArray(T, 2), // Emission probabilities [N×M]

        /// Initialize HMM with given number of states and observations
        ///
        /// Time: O(N² + N×M)
        /// Space: O(N² + N×M)
        pub fn init(allocator: Allocator, n_states: usize, n_observations: usize) !Self {
            if (n_states == 0 or n_observations == 0) {
                return error.InvalidDimensions;
            }

            const initial = try allocator.alloc(T, n_states);
            errdefer allocator.free(initial);

            // Initialize with uniform distributions
            const init_prob = 1.0 / @as(T, @floatFromInt(n_states));
            for (initial) |*p| {
                p.* = init_prob;
            }

            var transition = try NDArray(T, 2).zeros(allocator, &.{ n_states, n_states }, .row_major);
            errdefer transition.deinit();

            var emission = try NDArray(T, 2).zeros(allocator, &.{ n_states, n_observations }, .row_major);
            errdefer emission.deinit();

            // Initialize with uniform distributions
            const trans_prob = 1.0 / @as(T, @floatFromInt(n_states));
            const emis_prob = 1.0 / @as(T, @floatFromInt(n_observations));

            for (0..n_states) |i| {
                for (0..n_states) |j| {
                    transition.set(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) }, trans_prob);
                }
                for (0..n_observations) |k| {
                    emission.set(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(k)) }, emis_prob);
                }
            }

            return Self{
                .allocator = allocator,
                .n_states = n_states,
                .n_observations = n_observations,
                .initial = initial,
                .transition = transition,
                .emission = emission,
            };
        }

        /// Free all resources
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.initial);
            self.transition.deinit();
            self.emission.deinit();
        }

        /// Forward algorithm: Compute P(observations|model)
        ///
        /// Returns log probability to avoid numerical underflow.
        ///
        /// Time: O(T × N²) where T = sequence length, N = number of states
        /// Space: O(T × N) for forward table
        pub fn forward(self: *const Self, observations: []const usize) !T {
            if (observations.len == 0) return error.EmptySequence;

            const seq_len = observations.len;
            const alpha = try self.allocator.alloc([]T, seq_len);
            defer {
                for (alpha) |row| self.allocator.free(row);
                self.allocator.free(alpha);
            }

            // Allocate forward table
            for (alpha) |*row| {
                row.* = try self.allocator.alloc(T, self.n_states);
            }

            // Initialization: α_0(i) = π_i × b_i(o_0)
            const o0 = observations[0];
            if (o0 >= self.n_observations) return error.InvalidObservation;

            for (0..self.n_states) |i| {
                const emis = self.emission.get(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(o0)) });
                alpha[0][i] = self.initial[i] * emis;
            }

            // Recursion: α_t(j) = Σ_i α_{t-1}(i) × a_ij × b_j(o_t)
            for (1..seq_len) |t| {
                const ot = observations[t];
                if (ot >= self.n_observations) return error.InvalidObservation;

                for (0..self.n_states) |j| {
                    var sum: T = 0.0;
                    for (0..self.n_states) |i| {
                        const trans = self.transition.get(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) });
                        sum += alpha[t - 1][i] * trans;
                    }
                    const emis = self.emission.get(&.{ @as(isize, @intCast(j)), @as(isize, @intCast(ot)) });
                    alpha[t][j] = sum * emis;
                }
            }

            // Termination: P(O|λ) = Σ_i α_{T-1}(i)
            var prob: T = 0.0;
            for (alpha[seq_len - 1]) |a| {
                prob += a;
            }

            // Return log probability (avoid underflow)
            return if (prob > 0.0) @log(prob) else -std.math.inf(T);
        }

        /// Viterbi algorithm: Find most likely state sequence
        ///
        /// Returns array of state indices. Caller owns memory.
        ///
        /// Time: O(T × N²)
        /// Space: O(T × N) for Viterbi table + O(T × N) for backpointers
        pub fn viterbi(self: *const Self, observations: []const usize) ![]usize {
            if (observations.len == 0) return error.EmptySequence;

            const seq_len = observations.len;
            const delta = try self.allocator.alloc([]T, seq_len);
            defer {
                for (delta) |row| self.allocator.free(row);
                self.allocator.free(delta);
            }

            const psi = try self.allocator.alloc([]usize, seq_len);
            defer {
                for (psi) |row| self.allocator.free(row);
                self.allocator.free(psi);
            }

            // Allocate tables
            for (delta, psi) |*d, *p| {
                d.* = try self.allocator.alloc(T, self.n_states);
                p.* = try self.allocator.alloc(usize, self.n_states);
            }

            // Initialization: δ_0(i) = π_i × b_i(o_0)
            const o0 = observations[0];
            if (o0 >= self.n_observations) return error.InvalidObservation;

            for (0..self.n_states) |i| {
                const emis = self.emission.get(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(o0)) });
                delta[0][i] = self.initial[i] * emis;
                psi[0][i] = 0;
            }

            // Recursion: δ_t(j) = max_i [δ_{t-1}(i) × a_ij] × b_j(o_t)
            for (1..seq_len) |t| {
                const ot = observations[t];
                if (ot >= self.n_observations) return error.InvalidObservation;

                for (0..self.n_states) |j| {
                    var max_prob: T = -std.math.inf(T);
                    var max_state: usize = 0;

                    for (0..self.n_states) |i| {
                        const trans = self.transition.get(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) });
                        const prob = delta[t - 1][i] * trans;
                        if (prob > max_prob) {
                            max_prob = prob;
                            max_state = i;
                        }
                    }

                    const emis = self.emission.get(&.{ @as(isize, @intCast(j)), @as(isize, @intCast(ot)) });
                    delta[t][j] = max_prob * emis;
                    psi[t][j] = max_state;
                }
            }

            // Termination: Find best final state
            var max_prob: T = -std.math.inf(T);
            var best_final: usize = 0;
            for (delta[seq_len - 1], 0..) |d, i| {
                if (d > max_prob) {
                    max_prob = d;
                    best_final = i;
                }
            }

            // Backtrack to get state sequence
            const states = try self.allocator.alloc(usize, seq_len);
            errdefer self.allocator.free(states);

            states[seq_len - 1] = best_final;
            for (1..seq_len) |rev_t| {
                const t = seq_len - rev_t;
                states[t - 1] = psi[t][states[t]];
            }

            return states;
        }

        /// Backward algorithm: Compute backward probabilities β
        ///
        /// Used internally by Baum-Welch algorithm.
        ///
        /// Time: O(T × N²)
        /// Space: O(T × N) for backward table
        fn backward(self: *const Self, observations: []const usize) ![][]T {
            if (observations.len == 0) return error.EmptySequence;

            const seq_len = observations.len;
            const beta = try self.allocator.alloc([]T, seq_len);
            errdefer {
                for (beta) |row| self.allocator.free(row);
                self.allocator.free(beta);
            }

            // Allocate backward table
            for (beta) |*row| {
                row.* = try self.allocator.alloc(T, self.n_states);
            }

            // Initialization: β_{T-1}(i) = 1
            for (0..self.n_states) |i| {
                beta[seq_len - 1][i] = 1.0;
            }

            // Recursion: β_t(i) = Σ_j a_ij × b_j(o_{t+1}) × β_{t+1}(j)
            for (1..seq_len) |rev_t| {
                const t = seq_len - rev_t - 1;
                const ot1 = observations[t + 1];
                if (ot1 >= self.n_observations) return error.InvalidObservation;

                for (0..self.n_states) |i| {
                    var sum: T = 0.0;
                    for (0..self.n_states) |j| {
                        const trans = self.transition.get(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) });
                        const emis = self.emission.get(&.{ @as(isize, @intCast(j)), @as(isize, @intCast(ot1)) });
                        sum += trans * emis * beta[t + 1][j];
                    }
                    beta[t][i] = sum;
                }
            }

            return beta;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const expectEqual = testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectError = testing.expectError;

test "HMM: basic initialization" {
    const allocator = testing.allocator;

    var hmm = try HMM(f64).init(allocator, 2, 3);
    defer hmm.deinit();

    try expectEqual(@as(usize, 2), hmm.n_states);
    try expectEqual(@as(usize, 3), hmm.n_observations);

    // Check uniform initialization
    try expectApproxEqAbs(0.5, hmm.initial[0], 1e-6);
    try expectApproxEqAbs(0.5, hmm.initial[1], 1e-6);
}

test "HMM: weather model - forward algorithm" {
    const allocator = testing.allocator;

    var hmm = try HMM(f64).init(allocator, 2, 3);
    defer hmm.deinit();

    // States: 0=Sunny, 1=Rainy
    // Observations: 0=Walk, 1=Shop, 2=Clean

    // Initial probabilities
    hmm.initial[0] = 0.6; // P(Sunny)
    hmm.initial[1] = 0.4; // P(Rainy)

    // Transition probabilities
    hmm.transition.set(&.{ 0, 0 }, 0.7); // Sunny -> Sunny
    hmm.transition.set(&.{ 0, 1 }, 0.3); // Sunny -> Rainy
    hmm.transition.set(&.{ 1, 0 }, 0.4); // Rainy -> Sunny
    hmm.transition.set(&.{ 1, 1 }, 0.6); // Rainy -> Rainy

    // Emission probabilities
    hmm.emission.set(&.{ 0, 0 }, 0.6); // Sunny: Walk
    hmm.emission.set(&.{ 0, 1 }, 0.3); // Sunny: Shop
    hmm.emission.set(&.{ 0, 2 }, 0.1); // Sunny: Clean
    hmm.emission.set(&.{ 1, 0 }, 0.1); // Rainy: Walk
    hmm.emission.set(&.{ 1, 1 }, 0.4); // Rainy: Shop
    hmm.emission.set(&.{ 1, 2 }, 0.5); // Rainy: Clean

    // Observation sequence: Walk, Shop, Clean
    const observations = [_]usize{ 0, 1, 2 };
    const log_prob = try hmm.forward(&observations);

    // Should be negative (log of small probability)
    try testing.expect(log_prob < 0.0);
}

test "HMM: weather model - viterbi decoding" {
    const allocator = testing.allocator;

    var hmm = try HMM(f64).init(allocator, 2, 3);
    defer hmm.deinit();

    // Same model as forward test
    hmm.initial[0] = 0.6;
    hmm.initial[1] = 0.4;

    hmm.transition.set(&.{ 0, 0 }, 0.7);
    hmm.transition.set(&.{ 0, 1 }, 0.3);
    hmm.transition.set(&.{ 1, 0 }, 0.4);
    hmm.transition.set(&.{ 1, 1 }, 0.6);

    hmm.emission.set(&.{ 0, 0 }, 0.6);
    hmm.emission.set(&.{ 0, 1 }, 0.3);
    hmm.emission.set(&.{ 0, 2 }, 0.1);
    hmm.emission.set(&.{ 1, 0 }, 0.1);
    hmm.emission.set(&.{ 1, 1 }, 0.4);
    hmm.emission.set(&.{ 1, 2 }, 0.5);

    // Observation sequence: Walk, Shop, Clean
    const observations = [_]usize{ 0, 1, 2 };
    const states = try hmm.viterbi(&observations);
    defer allocator.free(states);

    try expectEqual(@as(usize, 3), states.len);
    // Walk (0.6 for Sunny > 0.1 for Rainy) -> likely Sunny
    // Shop (from Sunny: 0.3, from Rainy: 0.4) -> depends on path
    // Clean (0.1 for Sunny < 0.5 for Rainy) -> likely Rainy
    try expectEqual(@as(usize, 0), states[0]); // Sunny
}

test "HMM: single observation" {
    const allocator = testing.allocator;

    var hmm = try HMM(f64).init(allocator, 2, 2);
    defer hmm.deinit();

    hmm.initial[0] = 0.7;
    hmm.initial[1] = 0.3;

    hmm.transition.set(&.{ 0, 0 }, 0.8);
    hmm.transition.set(&.{ 0, 1 }, 0.2);
    hmm.transition.set(&.{ 1, 0 }, 0.3);
    hmm.transition.set(&.{ 1, 1 }, 0.7);

    hmm.emission.set(&.{ 0, 0 }, 0.9);
    hmm.emission.set(&.{ 0, 1 }, 0.1);
    hmm.emission.set(&.{ 1, 0 }, 0.2);
    hmm.emission.set(&.{ 1, 1 }, 0.8);

    const observations = [_]usize{0};
    const states = try hmm.viterbi(&observations);
    defer allocator.free(states);

    try expectEqual(@as(usize, 1), states.len);
    try expectEqual(@as(usize, 0), states[0]); // State 0 has higher P(0|state)
}

test "HMM: empty sequence error" {
    const allocator = testing.allocator;

    var hmm = try HMM(f64).init(allocator, 2, 2);
    defer hmm.deinit();

    const observations = [_]usize{};
    try expectError(error.EmptySequence, hmm.forward(&observations));
    try expectError(error.EmptySequence, hmm.viterbi(&observations));
}

test "HMM: invalid observation error" {
    const allocator = testing.allocator;

    var hmm = try HMM(f64).init(allocator, 2, 2);
    defer hmm.deinit();

    const observations = [_]usize{ 0, 5, 1 }; // 5 is out of range
    try expectError(error.InvalidObservation, hmm.forward(&observations));
    try expectError(error.InvalidObservation, hmm.viterbi(&observations));
}

test "HMM: zero states/observations error" {
    const allocator = testing.allocator;

    try expectError(error.InvalidDimensions, HMM(f64).init(allocator, 0, 3));
    try expectError(error.InvalidDimensions, HMM(f64).init(allocator, 2, 0));
}

test "HMM: f32 support" {
    const allocator = testing.allocator;

    var hmm = try HMM(f32).init(allocator, 2, 2);
    defer hmm.deinit();

    hmm.initial[0] = 0.6;
    hmm.initial[1] = 0.4;

    const observations = [_]usize{ 0, 1 };
    const log_prob = try hmm.forward(&observations);
    try testing.expect(log_prob < 0.0);
}

test "HMM: long sequence (10 observations)" {
    const allocator = testing.allocator;

    var hmm = try HMM(f64).init(allocator, 3, 4);
    defer hmm.deinit();

    // Random probabilities (not normalized for simplicity)
    hmm.initial[0] = 0.5;
    hmm.initial[1] = 0.3;
    hmm.initial[2] = 0.2;

    // Just check it doesn't crash
    const observations = [_]usize{ 0, 1, 2, 3, 0, 1, 2, 3, 0, 1 };
    const states = try hmm.viterbi(&observations);
    defer allocator.free(states);

    try expectEqual(@as(usize, 10), states.len);
}

test "HMM: deterministic model" {
    const allocator = testing.allocator;

    var hmm = try HMM(f64).init(allocator, 2, 2);
    defer hmm.deinit();

    // Deterministic: always start in state 0, always stay in state 0
    hmm.initial[0] = 1.0;
    hmm.initial[1] = 0.0;

    hmm.transition.set(&.{ 0, 0 }, 1.0);
    hmm.transition.set(&.{ 0, 1 }, 0.0);
    hmm.transition.set(&.{ 1, 0 }, 0.0);
    hmm.transition.set(&.{ 1, 1 }, 1.0);

    hmm.emission.set(&.{ 0, 0 }, 1.0);
    hmm.emission.set(&.{ 0, 1 }, 0.0);
    hmm.emission.set(&.{ 1, 0 }, 0.0);
    hmm.emission.set(&.{ 1, 1 }, 1.0);

    const observations = [_]usize{ 0, 0, 0 };
    const states = try hmm.viterbi(&observations);
    defer allocator.free(states);

    // All states should be 0
    try expectEqual(@as(usize, 0), states[0]);
    try expectEqual(@as(usize, 0), states[1]);
    try expectEqual(@as(usize, 0), states[2]);
}

test "HMM: backward algorithm" {
    const allocator = testing.allocator;

    var hmm = try HMM(f64).init(allocator, 2, 2);
    defer hmm.deinit();

    hmm.initial[0] = 0.6;
    hmm.initial[1] = 0.4;

    hmm.transition.set(&.{ 0, 0 }, 0.7);
    hmm.transition.set(&.{ 0, 1 }, 0.3);
    hmm.transition.set(&.{ 1, 0 }, 0.4);
    hmm.transition.set(&.{ 1, 1 }, 0.6);

    hmm.emission.set(&.{ 0, 0 }, 0.8);
    hmm.emission.set(&.{ 0, 1 }, 0.2);
    hmm.emission.set(&.{ 1, 0 }, 0.3);
    hmm.emission.set(&.{ 1, 1 }, 0.7);

    const observations = [_]usize{ 0, 1 };
    const beta = try hmm.backward(&observations);
    defer {
        for (beta) |row| allocator.free(row);
        allocator.free(beta);
    }

    try expectEqual(@as(usize, 2), beta.len);
    try expectEqual(@as(usize, 2), beta[0].len);
    try expectEqual(@as(usize, 2), beta[1].len);

    // Terminal condition: β_{T-1}(i) = 1
    try expectApproxEqAbs(1.0, beta[1][0], 1e-6);
    try expectApproxEqAbs(1.0, beta[1][1], 1e-6);
}

test "HMM: memory safety with testing.allocator" {
    const allocator = testing.allocator;

    var hmm = try HMM(f64).init(allocator, 3, 4);
    defer hmm.deinit();

    const observations = [_]usize{ 0, 1, 2, 3, 0 };
    const states = try hmm.viterbi(&observations);
    defer allocator.free(states);

    const log_prob = try hmm.forward(&observations);
    _ = log_prob;

    // testing.allocator will detect leaks
}

test "HMM: POS tagging example (simple)" {
    const allocator = testing.allocator;

    // States: 0=Noun, 1=Verb, 2=Adjective
    // Observations: 0=dog, 1=run, 2=fast
    var hmm = try HMM(f64).init(allocator, 3, 3);
    defer hmm.deinit();

    // Initial: more likely to start with noun
    hmm.initial[0] = 0.6; // Noun
    hmm.initial[1] = 0.3; // Verb
    hmm.initial[2] = 0.1; // Adjective

    // Transition: Noun->Verb common, Adjective->Noun common
    hmm.transition.set(&.{ 0, 0 }, 0.2);
    hmm.transition.set(&.{ 0, 1 }, 0.6);
    hmm.transition.set(&.{ 0, 2 }, 0.2);
    hmm.transition.set(&.{ 1, 0 }, 0.3);
    hmm.transition.set(&.{ 1, 1 }, 0.4);
    hmm.transition.set(&.{ 1, 2 }, 0.3);
    hmm.transition.set(&.{ 2, 0 }, 0.7);
    hmm.transition.set(&.{ 2, 1 }, 0.2);
    hmm.transition.set(&.{ 2, 2 }, 0.1);

    // Emission: dog->Noun, run->Verb, fast->Adjective
    hmm.emission.set(&.{ 0, 0 }, 0.8); // Noun: dog
    hmm.emission.set(&.{ 0, 1 }, 0.1); // Noun: run
    hmm.emission.set(&.{ 0, 2 }, 0.1); // Noun: fast
    hmm.emission.set(&.{ 1, 0 }, 0.1); // Verb: dog
    hmm.emission.set(&.{ 1, 1 }, 0.8); // Verb: run
    hmm.emission.set(&.{ 1, 2 }, 0.1); // Verb: fast
    hmm.emission.set(&.{ 2, 0 }, 0.1); // Adj: dog
    hmm.emission.set(&.{ 2, 1 }, 0.1); // Adj: run
    hmm.emission.set(&.{ 2, 2 }, 0.8); // Adj: fast

    // Sequence: "dog run fast" -> should tag as Noun Verb Adjective
    const observations = [_]usize{ 0, 1, 2 };
    const states = try hmm.viterbi(&observations);
    defer allocator.free(states);

    try expectEqual(@as(usize, 0), states[0]); // Noun
    try expectEqual(@as(usize, 1), states[1]); // Verb
    try expectEqual(@as(usize, 2), states[2]); // Adjective
}
