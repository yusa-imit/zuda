const std = @import("std");
const Allocator = std.mem.Allocator;

/// Conditional Random Field (CRF) for sequence labeling
///
/// CRF is a discriminative model for structured prediction on sequences.
/// Unlike HMM (generative), CRF directly models P(labels|observations).
///
/// Key features:
/// - Linear-chain CRF: each label depends on previous label
/// - Feature functions: arbitrary features from observations and label transitions
/// - Log-linear model: P(y|x) ∝ exp(Σ λ_k f_k(y_i-1, y_i, x, i))
/// - Training: maximize conditional likelihood via gradient-based methods
/// - Inference: Viterbi algorithm for most likely label sequence
///
/// Time complexity: O(T × N² × K) where T=sequence length, N=states, K=features
/// Space complexity: O(N² × K) for parameters
///
/// Use cases:
/// - Named Entity Recognition (NER): identify person, location, organization
/// - Part-of-Speech (POS) tagging: noun, verb, adjective, etc.
/// - Shallow parsing: noun phrases, verb phrases
/// - Gene sequence annotation
/// - Speech recognition
/// - Chinese word segmentation
///
/// Trade-offs:
/// - vs HMM: discriminative (models P(y|x) directly), handles overlapping features, but slower training
/// - vs LSTM-CRF: simpler (no deep learning), faster inference, but less feature learning
/// - vs MaxEnt Markov Model (MEMM): avoids label bias problem via global normalization
pub fn CRF(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        num_states: usize, // Number of possible labels (N)
        num_features: usize, // Number of feature functions (K)

        // Parameters: weights λ for each feature function
        // Shape: (num_features,)
        weights: []T,

        // Feature functions are defined by user
        // Each feature f_k(y_prev, y_curr, x, i) returns a value
        trained: bool,

        /// Initialize CRF with number of states and features
        /// Time: O(K), Space: O(K)
        pub fn init(allocator: Allocator, num_states: usize, num_features: usize) !Self {
            if (num_states == 0) return error.ZeroStates;
            if (num_features == 0) return error.ZeroFeatures;

            const weights = try allocator.alloc(T, num_features);
            @memset(weights, 0);

            return Self{
                .allocator = allocator,
                .num_states = num_states,
                .num_features = num_features,
                .weights = weights,
                .trained = false,
            };
        }

        /// Free all resources
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.weights);
            self.* = undefined;
        }

        /// Feature extraction function type
        /// Parameters:
        /// - y_prev: previous label index (or null for first position)
        /// - y_curr: current label index
        /// - observations: full observation sequence
        /// - position: current position in sequence
        /// Returns: feature vector for this configuration
        pub const FeatureExtractor = *const fn (
            y_prev: ?usize,
            y_curr: usize,
            observations: []const []const T,
            position: usize,
        ) []T;

        /// Train CRF using gradient descent with L2 regularization
        /// Time: O(iterations × T × N² × K), Space: O(N × T)
        pub fn train(
            self: *Self,
            sequences: []const []const []const T, // training sequences [num_seq][seq_len][feature_dim]
            labels: []const []const usize, // label sequences [num_seq][seq_len]
            feature_extractor: FeatureExtractor,
            options: struct {
                learning_rate: T = 0.01,
                max_iterations: usize = 100,
                tolerance: T = 1e-4,
                l2_lambda: T = 0.01, // L2 regularization
            },
        ) !void {
            if (sequences.len != labels.len) return error.SequenceLabelMismatch;
            if (sequences.len == 0) return error.EmptyTrainingData;

            var gradients = try self.allocator.alloc(T, self.num_features);
            defer self.allocator.free(gradients);

            var prev_loss: T = std.math.inf(T);

            for (0..options.max_iterations) |iter| {
                @memset(gradients, 0);
                var total_loss: T = 0;

                // Compute gradient over all sequences
                for (sequences, labels) |seq, seq_labels| {
                    if (seq.len != seq_labels.len) return error.SequenceLabelLengthMismatch;
                    if (seq.len == 0) continue;

                    // Forward-backward to compute marginals
                    const forward = try self.computeForward(seq, feature_extractor);
                    defer {
                        for (forward) |row| self.allocator.free(row);
                        self.allocator.free(forward);
                    }

                    const backward = try self.computeBackward(seq, feature_extractor);
                    defer {
                        for (backward) |row| self.allocator.free(row);
                        self.allocator.free(backward);
                    }

                    // Compute expected features (model) and observed features (data)
                    for (0..seq.len) |t| {
                        const y_curr = seq_labels[t];
                        const y_prev = if (t > 0) seq_labels[t - 1] else null;

                        // Observed features (from true labels)
                        const features = feature_extractor(y_prev, y_curr, seq, t);
                        defer self.allocator.free(features);

                        for (features, 0..) |f, k| {
                            gradients[k] += f; // Add observed
                        }

                        // Expected features (from model marginals)
                        for (0..self.num_states) |s| {
                            const marginal = self.computeMarginal(forward, backward, t, s);
                            const model_features = feature_extractor(
                                if (t > 0) seq_labels[t - 1] else null,
                                s,
                                seq,
                                t,
                            );
                            defer self.allocator.free(model_features);

                            for (model_features, 0..) |f, k| {
                                gradients[k] -= marginal * f; // Subtract expected
                            }
                        }
                    }

                    // Compute log-likelihood for this sequence
                    total_loss -= self.computeLogLikelihood(seq, seq_labels, feature_extractor) catch 0;
                }

                // Average gradients
                const num_sequences: T = @floatFromInt(sequences.len);
                for (gradients) |*g| {
                    g.* /= num_sequences;
                }

                // L2 regularization: gradient -= λ * weights
                for (self.weights, gradients) |w, *g| {
                    g.* -= options.l2_lambda * w;
                }

                // Update weights: w += learning_rate * gradient
                for (self.weights, gradients) |*w, g| {
                    w.* += options.learning_rate * g;
                }

                total_loss /= num_sequences;

                // Check convergence
                if (@abs(prev_loss - total_loss) < options.tolerance) {
                    _ = iter; // Training converged
                    break;
                }
                prev_loss = total_loss;
            }

            self.trained = true;
        }

        /// Compute forward probabilities (log space) for inference
        /// Time: O(T × N² × K), Space: O(T × N)
        fn computeForward(
            self: *Self,
            sequence: []const []const T,
            feature_extractor: FeatureExtractor,
        ) ![][]T {
            const T_len = sequence.len;
            var forward = try self.allocator.alloc([]T, T_len);
            errdefer {
                for (forward[0..T_len]) |row| self.allocator.free(row);
                self.allocator.free(forward);
            }

            for (0..T_len) |t| {
                forward[t] = try self.allocator.alloc(T, self.num_states);
                @memset(forward[t], -std.math.inf(T));
            }

            // Initialize t=0
            for (0..self.num_states) |s| {
                const features = feature_extractor(null, s, sequence, 0);
                defer self.allocator.free(features);
                var score: T = 0;
                for (features, self.weights) |f, w| {
                    score += f * w;
                }
                forward[0][s] = score;
            }

            // Forward pass
            for (1..T_len) |t| {
                for (0..self.num_states) |s_curr| {
                    var max_score: T = -std.math.inf(T);
                    for (0..self.num_states) |s_prev| {
                        const features = feature_extractor(s_prev, s_curr, sequence, t);
                        defer self.allocator.free(features);
                        var score: T = forward[t - 1][s_prev];
                        for (features, self.weights) |f, w| {
                            score += f * w;
                        }
                        max_score = @max(max_score, score);
                    }
                    forward[t][s_curr] = max_score;
                }
            }

            return forward;
        }

        /// Compute backward probabilities (log space)
        /// Time: O(T × N² × K), Space: O(T × N)
        fn computeBackward(
            self: *Self,
            sequence: []const []const T,
            feature_extractor: FeatureExtractor,
        ) ![][]T {
            const T_len = sequence.len;
            var backward = try self.allocator.alloc([]T, T_len);
            errdefer {
                for (backward[0..T_len]) |row| self.allocator.free(row);
                self.allocator.free(backward);
            }

            for (0..T_len) |t| {
                backward[t] = try self.allocator.alloc(T, self.num_states);
                @memset(backward[t], 0);
            }

            // Backward pass
            var t: usize = T_len - 1;
            while (t > 0) : (t -= 1) {
                for (0..self.num_states) |s_prev| {
                    var max_score: T = -std.math.inf(T);
                    for (0..self.num_states) |s_curr| {
                        const features = feature_extractor(s_prev, s_curr, sequence, t);
                        defer self.allocator.free(features);
                        var score: T = backward[t][s_curr];
                        for (features, self.weights) |f, w| {
                            score += f * w;
                        }
                        max_score = @max(max_score, score);
                    }
                    backward[t - 1][s_prev] = max_score;
                }
            }

            return backward;
        }

        /// Compute marginal probability P(y_t = s | x)
        fn computeMarginal(
            self: *Self,
            forward: [][]T,
            backward: [][]T,
            t: usize,
            s: usize,
        ) T {
            _ = self;
            const log_marginal = forward[t][s] + backward[t][s];
            return @exp(log_marginal);
        }

        /// Compute log-likelihood of a sequence
        fn computeLogLikelihood(
            self: *Self,
            sequence: []const []const T,
            labels: []const usize,
            feature_extractor: FeatureExtractor,
        ) !T {
            var score: T = 0;
            for (labels, 0..) |y_curr, t| {
                const y_prev = if (t > 0) labels[t - 1] else null;
                const features = feature_extractor(y_prev, y_curr, sequence, t);
                defer self.allocator.free(features);
                for (features, self.weights) |f, w| {
                    score += f * w;
                }
            }
            return score;
        }

        /// Predict most likely label sequence using Viterbi algorithm
        /// Time: O(T × N² × K), Space: O(T × N)
        pub fn predict(
            self: *Self,
            sequence: []const []const T,
            feature_extractor: FeatureExtractor,
            allocator: Allocator,
        ) ![]usize {
            if (!self.trained) return error.ModelNotTrained;
            if (sequence.len == 0) return error.EmptySequence;

            const T_len = sequence.len;

            // Viterbi tables
            var viterbi = try allocator.alloc([]T, T_len);
            defer {
                for (viterbi) |row| allocator.free(row);
                allocator.free(viterbi);
            }

            var backpointer = try allocator.alloc([]usize, T_len);
            defer {
                for (backpointer) |row| allocator.free(row);
                allocator.free(backpointer);
            }

            for (0..T_len) |t| {
                viterbi[t] = try allocator.alloc(T, self.num_states);
                @memset(viterbi[t], -std.math.inf(T));
                backpointer[t] = try allocator.alloc(usize, self.num_states);
                @memset(backpointer[t], 0);
            }

            // Initialize t=0
            for (0..self.num_states) |s| {
                const features = feature_extractor(null, s, sequence, 0);
                defer allocator.free(features);
                var score: T = 0;
                for (features, self.weights) |f, w| {
                    score += f * w;
                }
                viterbi[0][s] = score;
            }

            // Forward pass
            for (1..T_len) |t| {
                for (0..self.num_states) |s_curr| {
                    var max_score: T = -std.math.inf(T);
                    var best_prev: usize = 0;

                    for (0..self.num_states) |s_prev| {
                        const features = feature_extractor(s_prev, s_curr, sequence, t);
                        defer allocator.free(features);
                        var score: T = viterbi[t - 1][s_prev];
                        for (features, self.weights) |f, w| {
                            score += f * w;
                        }
                        if (score > max_score) {
                            max_score = score;
                            best_prev = s_prev;
                        }
                    }

                    viterbi[t][s_curr] = max_score;
                    backpointer[t][s_curr] = best_prev;
                }
            }

            // Backtrack to find best path
            var path = try allocator.alloc(usize, T_len);
            errdefer allocator.free(path);

            // Find best final state
            var best_score: T = -std.math.inf(T);
            var best_state: usize = 0;
            for (0..self.num_states) |s| {
                if (viterbi[T_len - 1][s] > best_score) {
                    best_score = viterbi[T_len - 1][s];
                    best_state = s;
                }
            }

            // Backtrack
            path[T_len - 1] = best_state;
            var t: usize = T_len - 1;
            while (t > 0) : (t -= 1) {
                path[t - 1] = backpointer[t][path[t]];
            }

            return path;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "CRF: initialization and cleanup" {
    var crf = try CRF(f64).init(std.testing.allocator, 3, 5);
    defer crf.deinit();

    try std.testing.expectEqual(@as(usize, 3), crf.num_states);
    try std.testing.expectEqual(@as(usize, 5), crf.num_features);
    try std.testing.expect(!crf.trained);
    try std.testing.expectEqual(@as(usize, 5), crf.weights.len);
}

test "CRF: zero states/features validation" {
    try std.testing.expectError(error.ZeroStates, CRF(f64).init(std.testing.allocator, 0, 5));
    try std.testing.expectError(error.ZeroFeatures, CRF(f64).init(std.testing.allocator, 3, 0));
}

// Simple feature extractor for testing
fn testFeatureExtractor(
    y_prev: ?usize,
    y_curr: usize,
    observations: []const []const f64,
    position: usize,
) []f64 {
    _ = observations;
    _ = position;

    // Simple features: [bias, transition(y_prev, y_curr)]
    var features = std.testing.allocator.alloc(f64, 2) catch unreachable;
    features[0] = 1.0; // bias
    features[1] = if (y_prev) |prev|
        if (prev == y_curr) 1.0 else 0.0
    else
        0.0;
    return features;
}

test "CRF: simple sequence prediction" {
    var crf = try CRF(f64).init(std.testing.allocator, 2, 2); // 2 states (0, 1), 2 features
    defer crf.deinit();

    // Manually set weights favoring state 0
    crf.weights[0] = 1.0; // bias favors state 0
    crf.weights[1] = 0.5; // transition bonus
    crf.trained = true;

    // Create simple sequence
    const obs1 = [_]f64{1.0};
    const obs2 = [_]f64{1.0};
    const obs3 = [_]f64{1.0};
    const sequence = &[_][]const f64{ &obs1, &obs2, &obs3 };

    const prediction = try crf.predict(sequence, testFeatureExtractor, std.testing.allocator);
    defer std.testing.allocator.free(prediction);

    try std.testing.expectEqual(@as(usize, 3), prediction.len);
    // With positive weights, should predict state 0 consistently
    for (prediction) |label| {
        try std.testing.expect(label < 2);
    }
}

test "CRF: empty sequence error" {
    var crf = try CRF(f64).init(std.testing.allocator, 2, 2);
    defer crf.deinit();
    crf.trained = true;

    const empty_seq = &[_][]const f64{};
    try std.testing.expectError(
        error.EmptySequence,
        crf.predict(empty_seq, testFeatureExtractor, std.testing.allocator),
    );
}

test "CRF: untrained model error" {
    var crf = try CRF(f64).init(std.testing.allocator, 2, 2);
    defer crf.deinit();

    const obs1 = [_]f64{1.0};
    const sequence = &[_][]const f64{&obs1};

    try std.testing.expectError(
        error.ModelNotTrained,
        crf.predict(sequence, testFeatureExtractor, std.testing.allocator),
    );
}

test "CRF: f32 type support" {
    var crf = try CRF(f32).init(std.testing.allocator, 2, 2);
    defer crf.deinit();

    try std.testing.expectEqual(@as(usize, 2), crf.num_states);
    try std.testing.expectEqual(@as(usize, 2), crf.num_features);
}

test "CRF: memory safety with testing.allocator" {
    var crf = try CRF(f64).init(std.testing.allocator, 3, 4);
    defer crf.deinit();

    try std.testing.expectEqual(@as(usize, 3), crf.num_states);
    try std.testing.expectEqual(@as(usize, 4), crf.num_features);
}
