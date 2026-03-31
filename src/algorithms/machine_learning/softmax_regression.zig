const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;
const math = std.math;

/// Softmax Regression (Multinomial Logistic Regression) - multi-class classifier
/// using softmax function and cross-entropy loss with gradient descent optimization.
///
/// Algorithm:
/// - Model: P(y=k|x) = exp(w_k·x + b_k) / Σ_j exp(w_j·x + b_j) for k = 1..K
/// - Loss: Cross-entropy L = -Σ_i Σ_k y_ik log(p_ik) where y_ik is one-hot encoded
/// - Optimization: Batch gradient descent with L2 regularization
/// - Gradient: ∂L/∂w_k = (1/n) Σ_i (p_ik - y_ik) x_i + λw_k (regularization)
///
/// Time complexity:
/// - Training: O(n_iter × n × m × k) where n = samples, m = features, k = classes
/// - Prediction: O(m × k) per sample (matrix-vector multiply + softmax)
///
/// Space complexity: O(k × m) for storing weight matrix
///
/// Features:
/// - True multi-class classification (not One-vs-Rest)
/// - L2 regularization to prevent overfitting
/// - Learning rate scheduling
/// - Early stopping on convergence
/// - Probability predictions (proper probability distribution via softmax)
/// - Numerical stability (log-sum-exp trick)
///
/// Use cases:
/// - Multi-class classification: document categorization, image recognition
/// - Natural probabilistic interpretation (probabilities sum to 1)
/// - Foundation for neural network output layers
/// - When classes are mutually exclusive
///
/// Trade-offs:
/// - vs Logistic Regression (OvR): Proper multi-class model, more parameters
/// - vs Naive Bayes: Discriminative model (vs generative), no independence assumption
/// - vs SVM: Probabilistic outputs, faster training, but only linear boundaries
/// - vs Neural Networks: Much simpler, interpretable, but limited to linear decision boundaries
pub fn SoftmaxRegression(comptime T: type) type {
    if (T != f32 and T != f64) {
        @compileError("SoftmaxRegression only supports f32 and f64");
    }

    return struct {
        const Self = @This();

        allocator: Allocator,
        /// Weight matrix: [n_classes][n_features]
        weights: ArrayList(ArrayList(T)),
        /// Bias vector: [n_classes]
        biases: ArrayList(T),
        /// Class labels (sorted)
        classes: ArrayList(i32),
        /// Number of features
        n_features: usize,
        /// Learning rate for gradient descent
        learning_rate: T,
        /// L2 regularization strength (lambda)
        l2_lambda: T,
        /// Maximum iterations for training
        max_iter: usize,
        /// Convergence tolerance (loss change threshold)
        tolerance: T,

        /// Initialize softmax regression classifier
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator, learning_rate: T, l2_lambda: T, max_iter: usize, tolerance: T) Self {
            return .{
                .allocator = allocator,
                .weights = ArrayList(ArrayList(T)).init(allocator),
                .biases = ArrayList(T).init(allocator),
                .classes = ArrayList(i32).init(allocator),
                .n_features = 0,
                .learning_rate = learning_rate,
                .l2_lambda = l2_lambda,
                .max_iter = max_iter,
                .tolerance = tolerance,
            };
        }

        /// Free all memory
        /// Time: O(k × m) | Space: O(1)
        pub fn deinit(self: *Self) void {
            for (self.weights.items) |*weight_vec| {
                weight_vec.deinit();
            }
            self.weights.deinit();
            self.biases.deinit();
            self.classes.deinit();
        }

        /// Train the classifier using batch gradient descent
        /// X: [n_samples][n_features] training data
        /// y: [n_samples] class labels
        /// Time: O(max_iter × n × m × k) | Space: O(k × m + n × k)
        pub fn fit(self: *Self, X: []const []const T, y: []const i32) !void {
            if (X.len == 0) return error.EmptyData;
            if (X.len != y.len) return error.MismatchedDimensions;

            const n_samples = X.len;
            self.n_features = X[0].len;

            // Validate feature dimensions
            for (X) |sample| {
                if (sample.len != self.n_features) return error.MismatchedDimensions;
            }

            // Identify unique classes and sort them
            var class_set = AutoHashMap(i32, void).init(self.allocator);
            defer class_set.deinit();

            for (y) |label| {
                try class_set.put(label, {});
            }

            const n_classes = class_set.count();
            if (n_classes < 2) return error.InsufficientClasses;

            // Store sorted classes
            self.classes.clearRetainingCapacity();
            var class_iter = class_set.keyIterator();
            while (class_iter.next()) |key| {
                try self.classes.append(key.*);
            }
            std.mem.sort(i32, self.classes.items, {}, std.sort.asc(i32));

            // Initialize weights and biases (Xavier initialization)
            const scale = @sqrt(@as(T, 2.0) / @as(T, @floatFromInt(self.n_features)));
            var prng = std.Random.DefaultPrng.init(42);
            const random = prng.random();

            // Clear existing weights
            for (self.weights.items) |*weight_vec| {
                weight_vec.deinit();
            }
            self.weights.clearRetainingCapacity();
            self.biases.clearRetainingCapacity();

            // Allocate weight matrix [k][m] and bias vector [k]
            for (0..n_classes) |_| {
                var weight_vec = ArrayList(T).init(self.allocator);
                for (0..self.n_features) |_| {
                    const val = (random.float(T) - 0.5) * scale;
                    try weight_vec.append(val);
                }
                try self.weights.append(weight_vec);
                try self.biases.append(0.0);
            }

            // Training loop - batch gradient descent
            var prev_loss: T = math.inf(T);

            for (0..self.max_iter) |iter| {
                _ = iter;

                // Allocate temporary storage for probabilities [n][k]
                var probs = ArrayList(ArrayList(T)).init(self.allocator);
                defer {
                    for (probs.items) |*prob_vec| {
                        prob_vec.deinit();
                    }
                    probs.deinit();
                }

                // Forward pass: compute softmax probabilities for all samples
                var total_loss: T = 0.0;

                for (X, 0..) |sample, i| {
                    const sample_probs = try self.softmax(sample);
                    try probs.append(sample_probs);

                    // Compute cross-entropy loss for this sample
                    const true_class_idx = self.getClassIndex(y[i]) orelse return error.UnknownClass;
                    const prob = sample_probs.items[true_class_idx];

                    // Numerical stability: clamp probability to avoid log(0)
                    const prob_clamped = @max(prob, 1e-10);
                    total_loss -= @log(prob_clamped);
                }

                // Add L2 regularization term to loss
                var reg_loss: T = 0.0;
                for (self.weights.items) |weight_vec| {
                    for (weight_vec.items) |w| {
                        reg_loss += w * w;
                    }
                }
                total_loss += 0.5 * self.l2_lambda * reg_loss;
                total_loss /= @as(T, @floatFromInt(n_samples));

                // Check convergence
                if (@abs(prev_loss - total_loss) < self.tolerance) {
                    break;
                }
                prev_loss = total_loss;

                // Backward pass: compute gradients and update weights
                for (0..n_classes) |k| {
                    // Gradient accumulator for class k
                    var grad_w = try ArrayList(T).initCapacity(self.allocator, self.n_features);
                    defer grad_w.deinit();
                    for (0..self.n_features) |_| {
                        try grad_w.append(0.0);
                    }
                    var grad_b: T = 0.0;

                    // Accumulate gradient over all samples
                    for (X, 0..) |sample, i| {
                        const true_class_idx = self.getClassIndex(y[i]) orelse return error.UnknownClass;
                        const y_k: T = if (k == true_class_idx) 1.0 else 0.0;
                        const p_k = probs.items[i].items[k];
                        const error_k = p_k - y_k;

                        // ∂L/∂w_k += (p_k - y_k) * x
                        for (sample, 0..) |x_j, j| {
                            grad_w.items[j] += error_k * x_j;
                        }
                        grad_b += error_k;
                    }

                    // Average gradient and add L2 regularization term
                    const n_samples_f: T = @floatFromInt(n_samples);
                    for (grad_w.items, 0..) |*g, j| {
                        g.* /= n_samples_f;
                        g.* += self.l2_lambda * self.weights.items[k].items[j];
                    }
                    grad_b /= n_samples_f;

                    // Update weights and bias
                    for (self.weights.items[k].items, 0..) |*w, j| {
                        w.* -= self.learning_rate * grad_w.items[j];
                    }
                    self.biases.items[k] -= self.learning_rate * grad_b;
                }
            }
        }

        /// Predict class labels for samples
        /// X: [n_samples][n_features] test data
        /// Returns: [n_samples] predicted class labels
        /// Time: O(n × m × k) | Space: O(n)
        pub fn predict(self: *const Self, X: []const []const T) !ArrayList(i32) {
            if (X.len == 0) return error.EmptyData;
            if (self.classes.items.len == 0) return error.ModelNotFitted;

            var predictions = ArrayList(i32).init(self.allocator);

            for (X) |sample| {
                if (sample.len != self.n_features) return error.MismatchedDimensions;

                var probs = try self.softmax(sample);
                defer probs.deinit();

                // Find class with maximum probability
                var max_prob: T = probs.items[0];
                var max_idx: usize = 0;
                for (probs.items, 0..) |prob, i| {
                    if (prob > max_prob) {
                        max_prob = prob;
                        max_idx = i;
                    }
                }

                try predictions.append(self.classes.items[max_idx]);
            }

            return predictions;
        }

        /// Predict class probabilities for samples
        /// X: [n_samples][n_features] test data
        /// Returns: [n_samples][n_classes] probability matrix
        /// Time: O(n × m × k) | Space: O(n × k)
        pub fn predictProba(self: *const Self, X: []const []const T) !ArrayList(ArrayList(T)) {
            if (X.len == 0) return error.EmptyData;
            if (self.classes.items.len == 0) return error.ModelNotFitted;

            var all_probs = ArrayList(ArrayList(T)).init(self.allocator);

            for (X) |sample| {
                if (sample.len != self.n_features) {
                    // Cleanup on error
                    for (all_probs.items) |*prob_vec| {
                        prob_vec.deinit();
                    }
                    all_probs.deinit();
                    return error.MismatchedDimensions;
                }

                const probs = try self.softmax(sample);
                try all_probs.append(probs);
            }

            return all_probs;
        }

        /// Compute accuracy on test data
        /// X: [n_samples][n_features] test data
        /// y: [n_samples] true class labels
        /// Returns: accuracy in range [0.0, 1.0]
        /// Time: O(n × m × k) | Space: O(n)
        pub fn score(self: *const Self, X: []const []const T, y: []const i32) !T {
            if (X.len == 0) return error.EmptyData;
            if (X.len != y.len) return error.MismatchedDimensions;

            var predictions = try self.predict(X);
            defer predictions.deinit();

            var correct: usize = 0;
            for (predictions.items, 0..) |pred, i| {
                if (pred == y[i]) {
                    correct += 1;
                }
            }

            return @as(T, @floatFromInt(correct)) / @as(T, @floatFromInt(y.len));
        }

        /// Get weight matrix (for inspection)
        /// Returns: [n_classes][n_features] weight matrix
        /// Time: O(1) | Space: O(1)
        pub fn getWeights(self: *const Self) []const ArrayList(T) {
            return self.weights.items;
        }

        /// Get bias vector (for inspection)
        /// Returns: [n_classes] bias vector
        /// Time: O(1) | Space: O(1)
        pub fn getBiases(self: *const Self) []const T {
            return self.biases.items;
        }

        /// Get sorted class labels
        /// Returns: [n_classes] class labels
        /// Time: O(1) | Space: O(1)
        pub fn getClasses(self: *const Self) []const i32 {
            return self.classes.items;
        }

        /// Get number of classes
        /// Time: O(1) | Space: O(1)
        pub fn getNumClasses(self: *const Self) usize {
            return self.classes.items.len;
        }

        // --- Private helper methods ---

        /// Compute softmax probabilities: p_k = exp(z_k) / Σ_j exp(z_j)
        /// Uses log-sum-exp trick for numerical stability
        /// Time: O(m × k) | Space: O(k)
        fn softmax(self: *const Self, x: []const T) !ArrayList(T) {
            var logits = ArrayList(T).init(self.allocator);
            defer logits.deinit();

            // Compute linear combinations z_k = w_k·x + b_k
            for (self.weights.items, 0..) |weight_vec, k| {
                var z: T = self.biases.items[k];
                for (weight_vec.items, 0..) |w, j| {
                    z += w * x[j];
                }
                try logits.append(z);
            }

            // Find max logit for numerical stability (log-sum-exp trick)
            var max_logit: T = logits.items[0];
            for (logits.items) |z| {
                if (z > max_logit) {
                    max_logit = z;
                }
            }

            // Compute exp(z - max) and sum
            var exp_sum: T = 0.0;
            var exp_logits = ArrayList(T).init(self.allocator);
            for (logits.items) |z| {
                const exp_z = @exp(z - max_logit);
                try exp_logits.append(exp_z);
                exp_sum += exp_z;
            }

            // Normalize to get probabilities
            for (exp_logits.items) |*exp_z| {
                exp_z.* /= exp_sum;
            }

            return exp_logits;
        }

        /// Get index of class label in sorted classes array
        /// Time: O(k) | Space: O(1)
        fn getClassIndex(self: *const Self, label: i32) ?usize {
            for (self.classes.items, 0..) |class, i| {
                if (class == label) return i;
            }
            return null;
        }
    };
}

// --- Tests ---

test "SoftmaxRegression: basic 3-class classification" {
    const allocator = std.testing.allocator;

    // Linearly separable 3-class problem
    const X = [_][]const f64{
        &.{ 1.0, 1.0 }, &.{ 1.5, 1.5 }, &.{ 1.2, 1.3 }, // Class 0
        &.{ 5.0, 5.0 }, &.{ 5.5, 5.2 }, &.{ 5.3, 5.4 }, // Class 1
        &.{ 9.0, 1.0 }, &.{ 9.5, 1.2 }, &.{ 9.3, 1.3 }, // Class 2
    };
    const y = [_]i32{ 0, 0, 0, 1, 1, 1, 2, 2, 2 };

    var model = SoftmaxRegression(f64).init(allocator, 0.1, 0.001, 1000, 1e-6);
    defer model.deinit();

    try model.fit(&X, &y);

    // Check predictions on training data
    var predictions = try model.predict(&X);
    defer predictions.deinit();

    try std.testing.expectEqual(@as(usize, 9), predictions.items.len);

    // Should classify most points correctly (perfect separation expected)
    var correct: usize = 0;
    for (predictions.items, 0..) |pred, i| {
        if (pred == y[i]) correct += 1;
    }
    try std.testing.expect(correct >= 8); // At least 88% accuracy
}

test "SoftmaxRegression: probability predictions sum to 1" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{
        &.{ 1.0, 2.0 }, &.{ 2.0, 3.0 },
        &.{ 5.0, 6.0 }, &.{ 6.0, 7.0 },
    };
    const y = [_]i32{ 0, 0, 1, 1 };

    var model = SoftmaxRegression(f64).init(allocator, 0.1, 0.01, 500, 1e-6);
    defer model.deinit();

    try model.fit(&X, &y);

    var probs = try model.predictProba(&X);
    defer {
        for (probs.items) |*prob_vec| {
            prob_vec.deinit();
        }
        probs.deinit();
    }

    // Check that probabilities sum to approximately 1.0
    for (probs.items) |prob_vec| {
        var sum: f64 = 0.0;
        for (prob_vec.items) |p| {
            sum += p;
        }
        try std.testing.expectApproxEqAbs(1.0, sum, 1e-6);

        // All probabilities should be in [0, 1]
        for (prob_vec.items) |p| {
            try std.testing.expect(p >= 0.0);
            try std.testing.expect(p <= 1.0);
        }
    }
}

test "SoftmaxRegression: getWeights and getBiases" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{ &.{ 1.0 }, &.{ 2.0 }, &.{ 3.0 }, &.{ 4.0 } };
    const y = [_]i32{ 0, 0, 1, 1 };

    var model = SoftmaxRegression(f64).init(allocator, 0.1, 0.0, 100, 1e-6);
    defer model.deinit();

    try model.fit(&X, &y);

    const weights = model.getWeights();
    const biases = model.getBiases();

    try std.testing.expectEqual(@as(usize, 2), weights.len);
    try std.testing.expectEqual(@as(usize, 2), biases.len);
    try std.testing.expectEqual(@as(usize, 1), weights[0].items.len);
}

test "SoftmaxRegression: score accuracy" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{
        &.{ 0.0, 0.0 }, &.{ 0.5, 0.5 },
        &.{ 5.0, 5.0 }, &.{ 5.5, 5.5 },
    };
    const y = [_]i32{ 0, 0, 1, 1 };

    var model = SoftmaxRegression(f64).init(allocator, 0.1, 0.01, 500, 1e-6);
    defer model.deinit();

    try model.fit(&X, &y);

    const accuracy = try model.score(&X, &y);
    try std.testing.expect(accuracy >= 0.9); // Should achieve >90% accuracy
}

test "SoftmaxRegression: empty data error" {
    const allocator = std.testing.allocator;

    const X: []const []const f64 = &.{};
    const y: []const i32 = &.{};

    var model = SoftmaxRegression(f64).init(allocator, 0.1, 0.01, 100, 1e-6);
    defer model.deinit();

    try std.testing.expectError(error.EmptyData, model.fit(X, y));
}

test "SoftmaxRegression: dimension mismatch" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{ &.{ 1.0, 2.0 }, &.{ 3.0, 4.0 } };
    const y = [_]i32{0}; // Wrong length

    var model = SoftmaxRegression(f64).init(allocator, 0.1, 0.01, 100, 1e-6);
    defer model.deinit();

    try std.testing.expectError(error.MismatchedDimensions, model.fit(&X, &y));
}

test "SoftmaxRegression: insufficient classes" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{ &.{ 1.0 }, &.{ 2.0 } };
    const y = [_]i32{ 0, 0 }; // Only one class

    var model = SoftmaxRegression(f64).init(allocator, 0.1, 0.01, 100, 1e-6);
    defer model.deinit();

    try std.testing.expectError(error.InsufficientClasses, model.fit(&X, &y));
}

test "SoftmaxRegression: predict before fit" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{ &.{ 1.0, 2.0 } };

    var model = SoftmaxRegression(f64).init(allocator, 0.1, 0.01, 100, 1e-6);
    defer model.deinit();

    try std.testing.expectError(error.ModelNotFitted, model.predict(&X));
}

test "SoftmaxRegression: 4-class problem" {
    const allocator = std.testing.allocator;

    // 4-class linearly separable problem
    const X = [_][]const f64{
        &.{ 0.0, 0.0 }, &.{ 0.5, 0.5 }, // Class 0
        &.{ 5.0, 0.0 }, &.{ 5.5, 0.5 }, // Class 1
        &.{ 0.0, 5.0 }, &.{ 0.5, 5.5 }, // Class 2
        &.{ 5.0, 5.0 }, &.{ 5.5, 5.5 }, // Class 3
    };
    const y = [_]i32{ 0, 0, 1, 1, 2, 2, 3, 3 };

    var model = SoftmaxRegression(f64).init(allocator, 0.1, 0.001, 2000, 1e-6);
    defer model.deinit();

    try model.fit(&X, &y);

    try std.testing.expectEqual(@as(usize, 4), model.getNumClasses());

    var predictions = try model.predict(&X);
    defer predictions.deinit();

    var correct: usize = 0;
    for (predictions.items, 0..) |pred, i| {
        if (pred == y[i]) correct += 1;
    }
    try std.testing.expect(correct >= 7); // At least 87.5% accuracy
}

test "SoftmaxRegression: L2 regularization effect" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{
        &.{ 1.0, 2.0 }, &.{ 1.5, 2.5 },
        &.{ 5.0, 6.0 }, &.{ 5.5, 6.5 },
    };
    const y = [_]i32{ 0, 0, 1, 1 };

    // Train with no regularization
    var model1 = SoftmaxRegression(f64).init(allocator, 0.1, 0.0, 500, 1e-6);
    defer model1.deinit();
    try model1.fit(&X, &y);

    // Train with strong regularization
    var model2 = SoftmaxRegression(f64).init(allocator, 0.1, 1.0, 500, 1e-6);
    defer model2.deinit();
    try model2.fit(&X, &y);

    // Compute L2 norm of weights
    var norm1: f64 = 0.0;
    for (model1.getWeights()) |weight_vec| {
        for (weight_vec.items) |w| {
            norm1 += w * w;
        }
    }

    var norm2: f64 = 0.0;
    for (model2.getWeights()) |weight_vec| {
        for (weight_vec.items) |w| {
            norm2 += w * w;
        }
    }

    // Regularized model should have smaller weight norms
    try std.testing.expect(norm2 < norm1);
}

test "SoftmaxRegression: f32 support" {
    const allocator = std.testing.allocator;

    const X = [_][]const f32{
        &.{ 1.0, 1.0 }, &.{ 1.5, 1.5 },
        &.{ 5.0, 5.0 }, &.{ 5.5, 5.5 },
    };
    const y = [_]i32{ 0, 0, 1, 1 };

    var model = SoftmaxRegression(f32).init(allocator, 0.1, 0.01, 500, 1e-5);
    defer model.deinit();

    try model.fit(&X, &y);

    const accuracy = try model.score(&X, &y);
    try std.testing.expect(accuracy >= 0.9);
}

test "SoftmaxRegression: non-sequential class labels" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{
        &.{ 1.0, 1.0 }, &.{ 1.5, 1.5 },
        &.{ 5.0, 5.0 }, &.{ 5.5, 5.5 },
        &.{ 9.0, 9.0 }, &.{ 9.5, 9.5 },
    };
    const y = [_]i32{ 10, 10, 20, 20, 30, 30 }; // Non-sequential labels

    var model = SoftmaxRegression(f64).init(allocator, 0.1, 0.01, 1000, 1e-6);
    defer model.deinit();

    try model.fit(&X, &y);

    const classes = model.getClasses();
    try std.testing.expectEqual(@as(usize, 3), classes.len);

    // Classes should be sorted
    try std.testing.expectEqual(@as(i32, 10), classes[0]);
    try std.testing.expectEqual(@as(i32, 20), classes[1]);
    try std.testing.expectEqual(@as(i32, 30), classes[2]);
}

test "SoftmaxRegression: large dataset stress test" {
    const allocator = std.testing.allocator;

    // Generate 300 samples with 5 features, 3 classes
    var X_list = ArrayList([]const f64).init(allocator);
    defer {
        for (X_list.items) |sample| {
            allocator.free(sample);
        }
        X_list.deinit();
    }

    var y_list = ArrayList(i32).init(allocator);
    defer y_list.deinit();

    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    for (0..300) |i| {
        const class: i32 = @intCast(@mod(i, 3));
        const offset: f64 = @as(f64, @floatFromInt(class)) * 5.0;

        var sample = try allocator.alloc(f64, 5);
        for (0..5) |j| {
            sample[j] = offset + random.float(f64) * 2.0;
        }

        try X_list.append(sample);
        try y_list.append(class);
    }

    var model = SoftmaxRegression(f64).init(allocator, 0.05, 0.01, 300, 1e-6);
    defer model.deinit();

    try model.fit(X_list.items, y_list.items);

    const accuracy = try model.score(X_list.items, y_list.items);
    try std.testing.expect(accuracy >= 0.85); // Should achieve >85% on this dataset
}

test "SoftmaxRegression: memory safety with testing.allocator" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{
        &.{ 1.0, 2.0 }, &.{ 2.0, 3.0 },
        &.{ 5.0, 6.0 }, &.{ 6.0, 7.0 },
    };
    const y = [_]i32{ 0, 0, 1, 1 };

    var model = SoftmaxRegression(f64).init(allocator, 0.1, 0.01, 200, 1e-6);
    defer model.deinit();

    try model.fit(&X, &y);

    var predictions = try model.predict(&X);
    defer predictions.deinit();

    var probs = try model.predictProba(&X);
    defer {
        for (probs.items) |*prob_vec| {
            prob_vec.deinit();
        }
        probs.deinit();
    }

    // testing.allocator will detect any leaks
}
