const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;

/// Naive Bayes classifier - probabilistic classification based on Bayes' theorem
/// with strong (naive) independence assumptions between features.
///
/// Supports:
/// - Gaussian Naive Bayes: continuous features with normal distribution
/// - Multinomial Naive Bayes: discrete features (e.g., word counts)
/// - Bernoulli Naive Bayes: binary features
///
/// Time complexity:
/// - Training: O(n × m) where n = samples, m = features
/// - Prediction: O(k × m) where k = classes, m = features
///
/// Space complexity: O(k × m) for storing class statistics
///
/// Use cases:
/// - Text classification (spam detection, sentiment analysis)
/// - Document categorization
/// - Medical diagnosis
/// - Real-time prediction (fast inference)
pub fn GaussianNaiveBayes(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        /// Class labels
        classes: ArrayList(i32),
        /// Class prior probabilities P(class)
        class_priors: ArrayList(T),
        /// Feature means for each class: [class][feature]
        means: ArrayList(ArrayList(T)),
        /// Feature variances for each class: [class][feature]
        variances: ArrayList(ArrayList(T)),
        /// Number of features
        n_features: usize,
        /// Smoothing parameter to avoid zero variance
        var_smoothing: T,

        /// Initialize Gaussian Naive Bayes classifier
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator, var_smoothing: T) Self {
            return .{
                .allocator = allocator,
                .classes = ArrayList(i32).init(allocator),
                .class_priors = ArrayList(T).init(allocator),
                .means = ArrayList(ArrayList(T)).init(allocator),
                .variances = ArrayList(ArrayList(T)).init(allocator),
                .n_features = 0,
                .var_smoothing = var_smoothing,
            };
        }

        /// Free all memory
        /// Time: O(k × m) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.classes.deinit();
            self.class_priors.deinit();
            for (self.means.items) |*mean_list| {
                mean_list.deinit();
            }
            self.means.deinit();
            for (self.variances.items) |*var_list| {
                var_list.deinit();
            }
            self.variances.deinit();
        }

        /// Train the classifier on labeled data
        /// X: [n_samples][n_features] training data
        /// y: [n_samples] class labels
        /// Time: O(n × m) | Space: O(k × m)
        pub fn fit(self: *Self, X: []const []const T, y: []const i32) !void {
            if (X.len == 0) return error.EmptyData;
            if (X.len != y.len) return error.MismatchedDimensions;

            const n_samples = X.len;
            self.n_features = X[0].len;

            // Clear previous training
            self.classes.clearRetainingCapacity();
            self.class_priors.clearRetainingCapacity();
            for (self.means.items) |*mean_list| {
                mean_list.deinit();
            }
            self.means.clearRetainingCapacity();
            for (self.variances.items) |*var_list| {
                var_list.deinit();
            }
            self.variances.clearRetainingCapacity();

            // Find unique classes and count samples per class
            var class_counts = AutoHashMap(i32, usize).init(self.allocator);
            defer class_counts.deinit();

            for (y) |label| {
                const count = class_counts.get(label) orelse 0;
                try class_counts.put(label, count + 1);
            }

            // Store classes sorted
            var class_iter = class_counts.keyIterator();
            while (class_iter.next()) |class_ptr| {
                try self.classes.append(class_ptr.*);
            }
            std.mem.sort(i32, self.classes.items, {}, comptime std.sort.asc(i32));

            const n_classes = self.classes.items.len;

            // Calculate class priors P(class)
            for (self.classes.items) |class| {
                const count = class_counts.get(class).?;
                const prior = @as(T, @floatFromInt(count)) / @as(T, @floatFromInt(n_samples));
                try self.class_priors.append(prior);
            }

            // Calculate mean and variance for each class and feature
            for (0..n_classes) |c| {
                var class_means = ArrayList(T).init(self.allocator);
                var class_vars = ArrayList(T).init(self.allocator);

                for (0..self.n_features) |f| {
                    var sum: T = 0.0;
                    var count: usize = 0;

                    // Calculate mean
                    for (X, y) |sample, label| {
                        if (label == self.classes.items[c]) {
                            sum += sample[f];
                            count += 1;
                        }
                    }

                    const mean = if (count > 0) sum / @as(T, @floatFromInt(count)) else 0.0;
                    try class_means.append(mean);

                    // Calculate variance
                    var var_sum: T = 0.0;
                    for (X, y) |sample, label| {
                        if (label == self.classes.items[c]) {
                            const diff = sample[f] - mean;
                            var_sum += diff * diff;
                        }
                    }

                    const variance = if (count > 0)
                        var_sum / @as(T, @floatFromInt(count)) + self.var_smoothing
                    else
                        self.var_smoothing;
                    try class_vars.append(variance);
                }

                try self.means.append(class_means);
                try self.variances.append(class_vars);
            }
        }

        /// Predict class label for a single sample
        /// Time: O(k × m) | Space: O(k)
        pub fn predict(self: *const Self, x: []const T) !i32 {
            if (x.len != self.n_features) return error.MismatchedDimensions;
            if (self.classes.items.len == 0) return error.NotFitted;

            const n_classes = self.classes.items.len;
            var max_log_prob: T = -std.math.inf(T);
            var best_class: i32 = self.classes.items[0];

            // Calculate log posterior for each class
            for (0..n_classes) |c| {
                var log_prob = @log(self.class_priors.items[c]);

                // Add log likelihood for each feature
                for (0..self.n_features) |f| {
                    const mean = self.means.items[c].items[f];
                    const variance = self.variances.items[c].items[f];

                    // Gaussian probability density function (log form)
                    const diff = x[f] - mean;
                    const exponent = -(diff * diff) / (2.0 * variance);
                    const normalizer = -0.5 * @log(2.0 * std.math.pi * variance);
                    log_prob += normalizer + exponent;
                }

                if (log_prob > max_log_prob) {
                    max_log_prob = log_prob;
                    best_class = self.classes.items[c];
                }
            }

            return best_class;
        }

        /// Predict class labels for multiple samples
        /// Time: O(n × k × m) | Space: O(n)
        pub fn predictBatch(self: *const Self, X: []const []const T, allocator: Allocator) ![]i32 {
            var predictions = try allocator.alloc(i32, X.len);
            errdefer allocator.free(predictions);

            for (X, 0..) |sample, i| {
                predictions[i] = try self.predict(sample);
            }

            return predictions;
        }

        /// Predict class probabilities for a single sample
        /// Returns probability distribution over classes
        /// Time: O(k × m) | Space: O(k)
        pub fn predictProba(self: *const Self, x: []const T, allocator: Allocator) ![]T {
            if (x.len != self.n_features) return error.MismatchedDimensions;
            if (self.classes.items.len == 0) return error.NotFitted;

            const n_classes = self.classes.items.len;
            var log_probs = try allocator.alloc(T, n_classes);
            defer allocator.free(log_probs);

            // Calculate log posterior for each class
            for (0..n_classes) |c| {
                var log_prob = @log(self.class_priors.items[c]);

                for (0..self.n_features) |f| {
                    const mean = self.means.items[c].items[f];
                    const variance = self.variances.items[c].items[f];

                    const diff = x[f] - mean;
                    const exponent = -(diff * diff) / (2.0 * variance);
                    const normalizer = -0.5 * @log(2.0 * std.math.pi * variance);
                    log_prob += normalizer + exponent;
                }

                log_probs[c] = log_prob;
            }

            // Convert log probabilities to probabilities using log-sum-exp trick
            const max_log_prob = blk: {
                var max_val = log_probs[0];
                for (log_probs[1..]) |lp| {
                    if (lp > max_val) max_val = lp;
                }
                break :blk max_val;
            };

            var sum: T = 0.0;
            var probs = try allocator.alloc(T, n_classes);
            errdefer allocator.free(probs);

            for (log_probs, 0..) |lp, i| {
                probs[i] = @exp(lp - max_log_prob);
                sum += probs[i];
            }

            // Normalize
            for (probs) |*p| {
                p.* /= sum;
            }

            return probs;
        }

        /// Calculate accuracy on test data
        /// Time: O(n × k × m) | Space: O(n)
        pub fn score(self: *const Self, X: []const []const T, y: []const i32) !T {
            if (X.len != y.len) return error.MismatchedDimensions;
            if (X.len == 0) return 0.0;

            var correct: usize = 0;
            for (X, y) |sample, true_label| {
                const pred_label = try self.predict(sample);
                if (pred_label == true_label) correct += 1;
            }

            return @as(T, @floatFromInt(correct)) / @as(T, @floatFromInt(X.len));
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "GaussianNaiveBayes: basic binary classification" {
    const allocator = std.testing.allocator;

    // Simple 2D linearly separable data
    const X = [_][]const f64{
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 1.5, 1.8 },
        &[_]f64{ 5.0, 8.0 },
        &[_]f64{ 6.0, 9.0 },
        &[_]f64{ 1.0, 0.6 },
        &[_]f64{ 9.0, 11.0 },
    };
    const y = [_]i32{ 0, 0, 1, 1, 0, 1 };

    var nb = GaussianNaiveBayes(f64).init(allocator, 1e-9);
    defer nb.deinit();

    try nb.fit(&X, &y);

    // Test predictions
    const pred1 = try nb.predict(&[_]f64{ 1.2, 1.9 });
    try std.testing.expectEqual(@as(i32, 0), pred1);

    const pred2 = try nb.predict(&[_]f64{ 5.5, 8.5 });
    try std.testing.expectEqual(@as(i32, 1), pred2);
}

test "GaussianNaiveBayes: multi-class classification" {
    const allocator = std.testing.allocator;

    // 3-class data (Iris-like)
    const X = [_][]const f64{
        &[_]f64{ 5.1, 3.5 }, // class 0
        &[_]f64{ 4.9, 3.0 },
        &[_]f64{ 6.2, 2.9 }, // class 1
        &[_]f64{ 5.9, 3.0 },
        &[_]f64{ 7.7, 3.8 }, // class 2
        &[_]f64{ 7.2, 3.6 },
    };
    const y = [_]i32{ 0, 0, 1, 1, 2, 2 };

    var nb = GaussianNaiveBayes(f64).init(allocator, 1e-9);
    defer nb.deinit();

    try nb.fit(&X, &y);

    try std.testing.expectEqual(@as(usize, 3), nb.classes.items.len);

    const pred = try nb.predict(&[_]f64{ 5.0, 3.2 });
    try std.testing.expectEqual(@as(i32, 0), pred);
}

test "GaussianNaiveBayes: predict batch" {
    const allocator = std.testing.allocator;

    const X_train = [_][]const f64{
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 2.0, 3.0 },
        &[_]f64{ 8.0, 8.0 },
        &[_]f64{ 9.0, 9.0 },
    };
    const y_train = [_]i32{ 0, 0, 1, 1 };

    var nb = GaussianNaiveBayes(f64).init(allocator, 1e-9);
    defer nb.deinit();

    try nb.fit(&X_train, &y_train);

    const X_test = [_][]const f64{
        &[_]f64{ 1.5, 2.5 },
        &[_]f64{ 8.5, 8.5 },
    };

    const predictions = try nb.predictBatch(&X_test, allocator);
    defer allocator.free(predictions);

    try std.testing.expectEqual(@as(usize, 2), predictions.len);
    try std.testing.expectEqual(@as(i32, 0), predictions[0]);
    try std.testing.expectEqual(@as(i32, 1), predictions[1]);
}

test "GaussianNaiveBayes: predict probabilities" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 1.5, 1.8 },
        &[_]f64{ 5.0, 8.0 },
        &[_]f64{ 6.0, 9.0 },
    };
    const y = [_]i32{ 0, 0, 1, 1 };

    var nb = GaussianNaiveBayes(f64).init(allocator, 1e-9);
    defer nb.deinit();

    try nb.fit(&X, &y);

    const probs = try nb.predictProba(&[_]f64{ 1.2, 1.9 }, allocator);
    defer allocator.free(probs);

    try std.testing.expectEqual(@as(usize, 2), probs.len);

    // Probabilities should sum to 1
    const sum = probs[0] + probs[1];
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-6);

    // Class 0 should have higher probability for this point
    try std.testing.expect(probs[0] > probs[1]);
}

test "GaussianNaiveBayes: accuracy score" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 1.5, 1.8 },
        &[_]f64{ 5.0, 8.0 },
        &[_]f64{ 6.0, 9.0 },
    };
    const y = [_]i32{ 0, 0, 1, 1 };

    var nb = GaussianNaiveBayes(f64).init(allocator, 1e-9);
    defer nb.deinit();

    try nb.fit(&X, &y);

    const accuracy = try nb.score(&X, &y);
    try std.testing.expectEqual(@as(f64, 1.0), accuracy); // Should fit training data perfectly
}

test "GaussianNaiveBayes: empty data error" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{};
    const y = [_]i32{};

    var nb = GaussianNaiveBayes(f64).init(allocator, 1e-9);
    defer nb.deinit();

    try std.testing.expectError(error.EmptyData, nb.fit(&X, &y));
}

test "GaussianNaiveBayes: mismatched dimensions error" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 3.0, 4.0 },
    };
    const y = [_]i32{0}; // Wrong length

    var nb = GaussianNaiveBayes(f64).init(allocator, 1e-9);
    defer nb.deinit();

    try std.testing.expectError(error.MismatchedDimensions, nb.fit(&X, &y));
}

test "GaussianNaiveBayes: predict before fit error" {
    const allocator = std.testing.allocator;

    var nb = GaussianNaiveBayes(f64).init(allocator, 1e-9);
    defer nb.deinit();

    try std.testing.expectError(error.NotFitted, nb.predict(&[_]f64{ 1.0, 2.0 }));
}

test "GaussianNaiveBayes: predict wrong feature count" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 3.0, 4.0 },
    };
    const y = [_]i32{ 0, 1 };

    var nb = GaussianNaiveBayes(f64).init(allocator, 1e-9);
    defer nb.deinit();

    try nb.fit(&X, &y);

    try std.testing.expectError(error.MismatchedDimensions, nb.predict(&[_]f64{ 1.0, 2.0, 3.0 }));
}

test "GaussianNaiveBayes: single class" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 1.5, 2.5 },
        &[_]f64{ 2.0, 3.0 },
    };
    const y = [_]i32{ 0, 0, 0 }; // All same class

    var nb = GaussianNaiveBayes(f64).init(allocator, 1e-9);
    defer nb.deinit();

    try nb.fit(&X, &y);

    const pred = try nb.predict(&[_]f64{ 1.7, 2.7 });
    try std.testing.expectEqual(@as(i32, 0), pred);
}

test "GaussianNaiveBayes: variance smoothing" {
    const allocator = std.testing.allocator;

    // Data with zero variance in one feature
    const X = [_][]const f64{
        &[_]f64{ 1.0, 5.0 },
        &[_]f64{ 1.0, 5.0 }, // Same values
        &[_]f64{ 9.0, 10.0 },
        &[_]f64{ 9.0, 10.0 },
    };
    const y = [_]i32{ 0, 0, 1, 1 };

    var nb = GaussianNaiveBayes(f64).init(allocator, 1e-5);
    defer nb.deinit();

    try nb.fit(&X, &y);

    // Should not crash due to zero variance (smoothing prevents division by zero)
    const pred = try nb.predict(&[_]f64{ 1.0, 5.0 });
    try std.testing.expectEqual(@as(i32, 0), pred);
}

test "GaussianNaiveBayes: class priors" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 1.5, 2.5 },
        &[_]f64{ 2.0, 3.0 }, // 3 samples of class 0
        &[_]f64{ 8.0, 9.0 }, // 1 sample of class 1
    };
    const y = [_]i32{ 0, 0, 0, 1 };

    var nb = GaussianNaiveBayes(f64).init(allocator, 1e-9);
    defer nb.deinit();

    try nb.fit(&X, &y);

    try std.testing.expectEqual(@as(usize, 2), nb.class_priors.items.len);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), nb.class_priors.items[0], 1e-6); // 3/4
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), nb.class_priors.items[1], 1e-6); // 1/4
}

test "GaussianNaiveBayes: f32 support" {
    const allocator = std.testing.allocator;

    const X = [_][]const f32{
        &[_]f32{ 1.0, 2.0 },
        &[_]f32{ 1.5, 1.8 },
        &[_]f32{ 5.0, 8.0 },
        &[_]f32{ 6.0, 9.0 },
    };
    const y = [_]i32{ 0, 0, 1, 1 };

    var nb = GaussianNaiveBayes(f32).init(allocator, 1e-6);
    defer nb.deinit();

    try nb.fit(&X, &y);

    const pred = try nb.predict(&[_]f32{ 1.2, 1.9 });
    try std.testing.expectEqual(@as(i32, 0), pred);
}

test "GaussianNaiveBayes: large dataset" {
    const allocator = std.testing.allocator;

    // Generate 100 samples, 10 features, 3 classes
    var X = std.ArrayList([]f64).init(allocator);
    defer {
        for (X.items) |row| {
            allocator.free(row);
        }
        X.deinit();
    }

    var y = std.ArrayList(i32).init(allocator);
    defer y.deinit();

    var prng = std.rand.DefaultPrng.init(42);
    const random = prng.random();

    for (0..100) |i| {
        var row = try allocator.alloc(f64, 10);
        const class = @as(i32, @intCast(i % 3));
        for (0..10) |j| {
            row[j] = random.float(f64) * 10.0 + @as(f64, @floatFromInt(class)) * 5.0;
        }
        try X.append(row);
        try y.append(class);
    }

    var nb = GaussianNaiveBayes(f64).init(allocator, 1e-9);
    defer nb.deinit();

    try nb.fit(X.items, y.items);

    const accuracy = try nb.score(X.items, y.items);
    // Should have reasonable accuracy on training data
    try std.testing.expect(accuracy > 0.5);
}

test "GaussianNaiveBayes: memory safety" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 3.0, 4.0 },
    };
    const y = [_]i32{ 0, 1 };

    var nb = GaussianNaiveBayes(f64).init(allocator, 1e-9);
    defer nb.deinit();

    try nb.fit(&X, &y);

    const probs = try nb.predictProba(&[_]f64{ 2.0, 3.0 }, allocator);
    defer allocator.free(probs);

    const predictions = try nb.predictBatch(&X, allocator);
    defer allocator.free(predictions);
}

test "GaussianNaiveBayes: refit capability" {
    const allocator = std.testing.allocator;

    var nb = GaussianNaiveBayes(f64).init(allocator, 1e-9);
    defer nb.deinit();

    // First fit
    const X1 = [_][]const f64{
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 3.0, 4.0 },
    };
    const y1 = [_]i32{ 0, 1 };
    try nb.fit(&X1, &y1);

    // Refit with different data
    const X2 = [_][]const f64{
        &[_]f64{ 5.0, 6.0 },
        &[_]f64{ 7.0, 8.0 },
        &[_]f64{ 9.0, 10.0 },
    };
    const y2 = [_]i32{ 0, 0, 1 };
    try nb.fit(&X2, &y2);

    try std.testing.expectEqual(@as(usize, 2), nb.n_features);
    try std.testing.expectEqual(@as(usize, 2), nb.classes.items.len);
}
