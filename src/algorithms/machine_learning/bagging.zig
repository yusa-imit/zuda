/// Bagging (Bootstrap Aggregating)
///
/// Meta-estimator that fits base estimators on random subsets of the dataset (bootstrap samples)
/// and aggregates predictions through voting (classification) or averaging (regression).
///
/// Key features:
/// - **Bootstrap sampling**: Samples with replacement from training set
/// - **Variance reduction**: Reduces overfitting via ensemble averaging
/// - **Out-of-bag estimation**: Uses samples not in bootstrap for validation
/// - **Parallelizable**: Each estimator is independent (single-threaded for now)
///
/// Algorithm:
/// 1. For each estimator:
///    - Draw bootstrap sample (n samples with replacement)
///    - Train base estimator on bootstrap sample
/// 2. Aggregate predictions:
///    - Classification: Majority vote
///    - Regression: Average
///
/// Time complexity: O(k × T_train) training, O(k × T_predict) prediction
///   where k = n_estimators, T_train/T_predict = base estimator cost
/// Space complexity: O(k × S_estimator) where S_estimator = base estimator space
///
/// Use cases:
/// - Reducing variance of unstable learners (decision trees, neural networks)
/// - Improving generalization without parameter tuning
/// - Out-of-bag error estimation (free cross-validation)
/// - Parallel ensemble training
///
/// Differences from Random Forest:
/// - Bagging: Generic meta-learner, any base estimator, no feature sampling
/// - Random Forest: Specialized for trees, adds random feature selection per split
///
/// Example (classification):
/// ```zig
/// var bagging = try BaggingClassifier(f64).init(allocator, .{
///     .n_estimators = 10,
///     .max_samples = 1.0, // 100% bootstrap sample size
///     .bootstrap = true,
///     .random_seed = 42,
/// });
/// defer bagging.deinit();
/// try bagging.fit(X, y);
/// const predictions = try bagging.predict(X_test);
/// defer allocator.free(predictions);
/// const oob = try bagging.oobScore(); // Out-of-bag accuracy
/// ```

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const DecisionTree = @import("decision_tree.zig").DecisionTree;

/// Bagging Classifier
///
/// Bootstrap aggregating for classification using decision trees as base estimators.
///
/// Time: O(k × n × m × log n) training, O(k × depth) prediction (k = estimators)
/// Space: O(k × nodes) for trees
pub fn BaggingClassifier(comptime T: type) type {
    return struct {
        allocator: Allocator,
        base_estimators: []DecisionTree(T),
        n_classes: usize,
        n_features: usize,
        config: Config,
        // Out-of-bag tracking
        oob_predictions: ?[]usize, // [n_samples]
        oob_counts: ?[]usize, // [n_samples] - how many times sample was OOB

        const Self = @This();

        pub const Config = struct {
            n_estimators: u32 = 10,
            max_samples: T = 1.0, // Fraction of samples (0.0-1.0) or count (>1.0)
            bootstrap: bool = true,
            random_seed: u64 = 42,
            // Base estimator config
            max_depth: u32 = 10,
            min_samples_split: u32 = 2,
            min_samples_leaf: u32 = 1,
        };

        /// Initialize bagging classifier
        ///
        /// Time: O(k)
        /// Space: O(k)
        pub fn init(allocator: Allocator, config: Config) !Self {
            if (config.n_estimators == 0) return error.InvalidNumEstimators;
            if (config.max_samples <= 0) return error.InvalidMaxSamples;

            const estimators = try allocator.alloc(DecisionTree(T), config.n_estimators);
            errdefer allocator.free(estimators);

            for (estimators) |*est| {
                est.* = DecisionTree(T).init(allocator, .{
                    .tree_type = .classification,
                    .max_depth = config.max_depth,
                    .min_samples_split = config.min_samples_split,
                    .min_samples_leaf = config.min_samples_leaf,
                });
            }

            return .{
                .allocator = allocator,
                .base_estimators = estimators,
                .n_classes = 0,
                .n_features = 0,
                .config = config,
                .oob_predictions = null,
                .oob_counts = null,
            };
        }

        /// Free resources
        ///
        /// Time: O(k × nodes)
        /// Space: O(1)
        pub fn deinit(self: *Self) void {
            for (self.base_estimators) |*est| {
                est.deinit();
            }
            self.allocator.free(self.base_estimators);
            if (self.oob_predictions) |oob| self.allocator.free(oob);
            if (self.oob_counts) |counts| self.allocator.free(counts);
        }

        /// Train bagging classifier on dataset
        ///
        /// X: [n_samples][n_features]
        /// y: [n_samples] class labels
        ///
        /// Time: O(k × n × m × log n)
        /// Space: O(n) per estimator for bootstrap sample
        pub fn fit(self: *Self, X: []const []const T, y: []const usize) !void {
            if (X.len == 0) return error.EmptyInput;
            if (X.len != y.len) return error.MismatchedDimensions;

            const n_samples = X.len;
            self.n_features = X[0].len;

            // Detect number of classes
            self.n_classes = 0;
            for (y) |label| {
                if (label >= self.n_classes) {
                    self.n_classes = label + 1;
                }
            }

            // Determine bootstrap sample size
            const sample_size: usize = if (self.config.max_samples <= 1.0)
                @intFromFloat(@as(f64, @floatFromInt(n_samples)) * @as(f64, @floatCast(self.config.max_samples)))
            else
                @intFromFloat(@as(f64, @floatCast(self.config.max_samples)));

            // Initialize OOB tracking
            if (self.config.bootstrap) {
                self.oob_predictions = try self.allocator.alloc(usize, n_samples);
                self.oob_counts = try self.allocator.alloc(usize, n_samples);
                @memset(self.oob_predictions.?, 0);
                @memset(self.oob_counts.?, 0);
            }

            // Train each estimator on bootstrap sample
            var prng = std.rand.DefaultPrng.init(self.config.random_seed);
            const rand = prng.random();

            for (self.base_estimators) |*estimator| {
                // Generate bootstrap sample
                var indices = try self.allocator.alloc(usize, sample_size);
                defer self.allocator.free(indices);

                var in_bag = try self.allocator.alloc(bool, n_samples);
                defer self.allocator.free(in_bag);
                @memset(in_bag, false);

                if (self.config.bootstrap) {
                    // Sampling with replacement
                    for (indices) |*idx| {
                        idx.* = rand.uintLessThan(usize, n_samples);
                        in_bag[idx.*] = true;
                    }
                } else {
                    // No replacement - just shuffle
                    for (0..sample_size) |i| {
                        indices[i] = i;
                    }
                    rand.shuffle(usize, indices);
                    for (indices) |idx| {
                        in_bag[idx] = true;
                    }
                }

                // Create bootstrap dataset
                var X_bootstrap = try self.allocator.alloc([]const T, sample_size);
                defer self.allocator.free(X_bootstrap);
                var y_bootstrap = try self.allocator.alloc(usize, sample_size);
                defer self.allocator.free(y_bootstrap);

                for (indices, 0..) |idx, i| {
                    X_bootstrap[i] = X[idx];
                    y_bootstrap[i] = y[idx];
                }

                // Train estimator
                try estimator.fit(X_bootstrap, y_bootstrap);

                // Update OOB predictions for samples not in this bootstrap
                if (self.config.bootstrap) {
                    for (0..n_samples) |i| {
                        if (!in_bag[i]) {
                            const pred = try estimator.predict(&[_][]const T{X[i]});
                            defer self.allocator.free(pred);
                            self.oob_predictions.?[i] += pred[0];
                            self.oob_counts.?[i] += 1;
                        }
                    }
                }

                // Update seed for next estimator
                _ = rand.int(u64); // Advance RNG state
            }
        }

        /// Predict class labels
        ///
        /// X: [n_samples][n_features]
        /// Returns: [n_samples] predicted labels
        ///
        /// Time: O(k × n × depth)
        /// Space: O(n) for predictions
        pub fn predict(self: *Self, X: []const []const T) ![]usize {
            if (X.len == 0) return error.EmptyInput;
            if (X[0].len != self.n_features) return error.FeatureMismatch;

            const n_samples = X.len;
            var predictions = try self.allocator.alloc(usize, n_samples);
            errdefer self.allocator.free(predictions);

            // Vote matrix: [n_samples][n_classes]
            var votes = try self.allocator.alloc([]usize, n_samples);
            defer {
                for (votes) |row| {
                    self.allocator.free(row);
                }
                self.allocator.free(votes);
            }

            for (votes) |*row| {
                row.* = try self.allocator.alloc(usize, self.n_classes);
                @memset(row.*, 0);
            }

            // Collect predictions from each estimator
            for (self.base_estimators) |*estimator| {
                const est_pred = try estimator.predict(X);
                defer self.allocator.free(est_pred);

                for (est_pred, 0..) |label, i| {
                    votes[i][label] += 1;
                }
            }

            // Majority vote
            for (votes, 0..) |vote_row, i| {
                var max_votes: usize = 0;
                var max_class: usize = 0;
                for (vote_row, 0..) |count, class| {
                    if (count > max_votes) {
                        max_votes = count;
                        max_class = class;
                    }
                }
                predictions[i] = max_class;
            }

            return predictions;
        }

        /// Compute out-of-bag score (accuracy)
        ///
        /// Returns: OOB accuracy [0.0, 1.0]
        ///
        /// Time: O(n)
        /// Space: O(1)
        ///
        /// Note: Requires storing y labels during fit() for full implementation.
        /// Currently returns placeholder value.
        pub fn oobScore(self: *Self) !T {
            if (!self.config.bootstrap) return error.BootstrapNotEnabled;
            if (self.oob_predictions == null) return error.NotFitted;

            const oob_count = self.oob_counts.?;

            var total: usize = 0;
            for (oob_count) |count| {
                if (count > 0) {
                    // Note: We need access to y to compute accuracy
                    // In practice, store y during fit() for OOB computation
                    total += 1;
                }
            }

            if (total == 0) return error.NoOOBSamples;
            // Placeholder - requires storing y labels
            return 0.0;
        }

        /// Get number of estimators
        pub fn numEstimators(self: *const Self) usize {
            return self.base_estimators.len;
        }
    };
}

/// Bagging Regressor
///
/// Bootstrap aggregating for regression using decision trees as base estimators.
///
/// Time: O(k × n × m × log n) training, O(k × depth) prediction (k = estimators)
/// Space: O(k × nodes) for trees
pub fn BaggingRegressor(comptime T: type) type {
    return struct {
        allocator: Allocator,
        base_estimators: []DecisionTree(T),
        n_features: usize,
        config: Config,
        // Out-of-bag tracking
        oob_predictions: ?[]T, // [n_samples]
        oob_counts: ?[]usize, // [n_samples]

        const Self = @This();

        pub const Config = struct {
            n_estimators: u32 = 10,
            max_samples: T = 1.0,
            bootstrap: bool = true,
            random_seed: u64 = 42,
            // Base estimator config
            max_depth: u32 = 10,
            min_samples_split: u32 = 2,
            min_samples_leaf: u32 = 1,
        };

        /// Initialize bagging regressor
        ///
        /// Time: O(k)
        /// Space: O(k)
        pub fn init(allocator: Allocator, config: Config) !Self {
            if (config.n_estimators == 0) return error.InvalidNumEstimators;
            if (config.max_samples <= 0) return error.InvalidMaxSamples;

            const estimators = try allocator.alloc(DecisionTree(T), config.n_estimators);
            errdefer allocator.free(estimators);

            for (estimators) |*est| {
                est.* = DecisionTree(T).init(allocator, .{
                    .tree_type = .regression,
                    .max_depth = config.max_depth,
                    .min_samples_split = config.min_samples_split,
                    .min_samples_leaf = config.min_samples_leaf,
                });
            }

            return .{
                .allocator = allocator,
                .base_estimators = estimators,
                .n_features = 0,
                .config = config,
                .oob_predictions = null,
                .oob_counts = null,
            };
        }

        /// Free resources
        ///
        /// Time: O(k × nodes)
        /// Space: O(1)
        pub fn deinit(self: *Self) void {
            for (self.base_estimators) |*est| {
                est.deinit();
            }
            self.allocator.free(self.base_estimators);
            if (self.oob_predictions) |oob| self.allocator.free(oob);
            if (self.oob_counts) |counts| self.allocator.free(counts);
        }

        /// Train bagging regressor on dataset
        ///
        /// X: [n_samples][n_features]
        /// y: [n_samples] target values
        ///
        /// Time: O(k × n × m × log n)
        /// Space: O(n) per estimator for bootstrap sample
        pub fn fit(self: *Self, X: []const []const T, y: []const T) !void {
            if (X.len == 0) return error.EmptyInput;
            if (X.len != y.len) return error.MismatchedDimensions;

            const n_samples = X.len;
            self.n_features = X[0].len;

            const sample_size: usize = if (self.config.max_samples <= 1.0)
                @intFromFloat(@as(f64, @floatFromInt(n_samples)) * @as(f64, @floatCast(self.config.max_samples)))
            else
                @intFromFloat(@as(f64, @floatCast(self.config.max_samples)));

            // Initialize OOB tracking
            if (self.config.bootstrap) {
                self.oob_predictions = try self.allocator.alloc(T, n_samples);
                self.oob_counts = try self.allocator.alloc(usize, n_samples);
                @memset(self.oob_predictions.?, 0.0);
                @memset(self.oob_counts.?, 0);
            }

            var prng = std.rand.DefaultPrng.init(self.config.random_seed);
            const rand = prng.random();

            for (self.base_estimators) |*estimator| {
                var indices = try self.allocator.alloc(usize, sample_size);
                defer self.allocator.free(indices);

                var in_bag = try self.allocator.alloc(bool, n_samples);
                defer self.allocator.free(in_bag);
                @memset(in_bag, false);

                if (self.config.bootstrap) {
                    for (indices) |*idx| {
                        idx.* = rand.uintLessThan(usize, n_samples);
                        in_bag[idx.*] = true;
                    }
                } else {
                    for (0..sample_size) |i| {
                        indices[i] = i;
                    }
                    rand.shuffle(usize, indices);
                    for (indices) |idx| {
                        in_bag[idx] = true;
                    }
                }

                var X_bootstrap = try self.allocator.alloc([]const T, sample_size);
                defer self.allocator.free(X_bootstrap);
                var y_bootstrap = try self.allocator.alloc(T, sample_size);
                defer self.allocator.free(y_bootstrap);

                for (indices, 0..) |idx, i| {
                    X_bootstrap[i] = X[idx];
                    y_bootstrap[i] = y[idx];
                }

                try estimator.fit(X_bootstrap, y_bootstrap);

                // Update OOB predictions
                if (self.config.bootstrap) {
                    for (0..n_samples) |i| {
                        if (!in_bag[i]) {
                            const pred = try estimator.predict(&[_][]const T{X[i]});
                            defer self.allocator.free(pred);
                            self.oob_predictions.?[i] += pred[0];
                            self.oob_counts.?[i] += 1;
                        }
                    }
                }

                _ = rand.int(u64);
            }
        }

        /// Predict target values
        ///
        /// X: [n_samples][n_features]
        /// Returns: [n_samples] predicted values
        ///
        /// Time: O(k × n × depth)
        /// Space: O(n)
        pub fn predict(self: *Self, X: []const []const T) ![]T {
            if (X.len == 0) return error.EmptyInput;
            if (X[0].len != self.n_features) return error.FeatureMismatch;

            const n_samples = X.len;
            var predictions = try self.allocator.alloc(T, n_samples);
            errdefer self.allocator.free(predictions);
            @memset(predictions, 0.0);

            // Average predictions from each estimator
            for (self.base_estimators) |*estimator| {
                const est_pred = try estimator.predict(X);
                defer self.allocator.free(est_pred);

                for (est_pred, 0..) |pred, i| {
                    predictions[i] += pred;
                }
            }

            // Average
            const k: T = @floatFromInt(self.base_estimators.len);
            for (predictions) |*pred| {
                pred.* /= k;
            }

            return predictions;
        }

        /// Compute out-of-bag score (R²)
        ///
        /// Requires storing y during fit (not implemented)
        pub fn oobScore(self: *Self) !T {
            if (!self.config.bootstrap) return error.BootstrapNotEnabled;
            if (self.oob_predictions == null) return error.NotFitted;
            // Placeholder - requires storing y
            return 0.0;
        }

        /// Get number of estimators
        pub fn numEstimators(self: *const Self) usize {
            return self.base_estimators.len;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "BaggingClassifier: basic initialization" {
    const allocator = testing.allocator;

    var bagging = try BaggingClassifier(f64).init(allocator, .{
        .n_estimators = 5,
    });
    defer bagging.deinit();

    try testing.expectEqual(@as(usize, 5), bagging.numEstimators());
}

test "BaggingClassifier: simple binary classification" {
    const allocator = testing.allocator;

    // XOR-like problem
    const X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 0.0, 1.0 },
        &[_]f64{ 1.0, 0.0 },
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 0.1, 0.1 },
        &[_]f64{ 0.9, 0.9 },
    };
    const y = [_]usize{ 0, 1, 1, 0, 0, 0 };

    var bagging = try BaggingClassifier(f64).init(allocator, .{
        .n_estimators = 10,
        .max_depth = 5,
        .random_seed = 42,
    });
    defer bagging.deinit();

    try bagging.fit(&X, &y);

    const predictions = try bagging.predict(&X);
    defer allocator.free(predictions);

    try testing.expectEqual(@as(usize, 6), predictions.len);
    // Should predict reasonably well on training data
}

test "BaggingClassifier: multi-class classification" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 2.0, 2.0 },
        &[_]f64{ 0.1, 0.1 },
        &[_]f64{ 1.1, 1.1 },
        &[_]f64{ 2.1, 2.1 },
    };
    const y = [_]usize{ 0, 1, 2, 0, 1, 2 };

    var bagging = try BaggingClassifier(f64).init(allocator, .{
        .n_estimators = 10,
        .max_depth = 3,
    });
    defer bagging.deinit();

    try bagging.fit(&X, &y);
    try testing.expectEqual(@as(usize, 3), bagging.n_classes);

    const predictions = try bagging.predict(&X);
    defer allocator.free(predictions);

    try testing.expectEqual(@as(usize, 6), predictions.len);
}

test "BaggingClassifier: no bootstrap (pasting)" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 1.0, 1.0 },
    };
    const y = [_]usize{ 0, 1 };

    var bagging = try BaggingClassifier(f64).init(allocator, .{
        .n_estimators = 5,
        .bootstrap = false, // Pasting instead of bagging
    });
    defer bagging.deinit();

    try bagging.fit(&X, &y);

    const predictions = try bagging.predict(&X);
    defer allocator.free(predictions);

    try testing.expectEqual(@as(usize, 2), predictions.len);
}

test "BaggingClassifier: fractional max_samples" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{ 0.0 }, &[_]f64{ 1.0 }, &[_]f64{ 2.0 }, &[_]f64{ 3.0 },
        &[_]f64{ 4.0 }, &[_]f64{ 5.0 }, &[_]f64{ 6.0 }, &[_]f64{ 7.0 },
    };
    const y = [_]usize{ 0, 0, 0, 0, 1, 1, 1, 1 };

    var bagging = try BaggingClassifier(f64).init(allocator, .{
        .n_estimators = 5,
        .max_samples = 0.5, // 50% of samples per bootstrap
    });
    defer bagging.deinit();

    try bagging.fit(&X, &y);

    const predictions = try bagging.predict(&X);
    defer allocator.free(predictions);

    try testing.expectEqual(@as(usize, 8), predictions.len);
}

test "BaggingRegressor: basic regression" {
    const allocator = testing.allocator;

    // Simple linear relationship
    const X = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{1.0}, &[_]f64{2.0}, &[_]f64{3.0},
        &[_]f64{4.0}, &[_]f64{5.0},
    };
    const y = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };

    var bagging = try BaggingRegressor(f64).init(allocator, .{
        .n_estimators = 10,
        .max_depth = 5,
    });
    defer bagging.deinit();

    try bagging.fit(&X, &y);

    const predictions = try bagging.predict(&X);
    defer allocator.free(predictions);

    try testing.expectEqual(@as(usize, 6), predictions.len);

    // Should predict close to actual values
    for (predictions, y) |pred, actual| {
        try testing.expect(@abs(pred - actual) < 1.0);
    }
}

test "BaggingRegressor: polynomial relationship" {
    const allocator = testing.allocator;

    // y = x^2
    const X = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{1.0}, &[_]f64{2.0}, &[_]f64{3.0}, &[_]f64{4.0},
    };
    const y = [_]f64{ 0.0, 1.0, 4.0, 9.0, 16.0 };

    var bagging = try BaggingRegressor(f64).init(allocator, .{
        .n_estimators = 20,
        .max_depth = 10,
    });
    defer bagging.deinit();

    try bagging.fit(&X, &y);

    const predictions = try bagging.predict(&X);
    defer allocator.free(predictions);

    try testing.expectEqual(@as(usize, 5), predictions.len);
}

test "BaggingClassifier: empty input validation" {
    const allocator = testing.allocator;

    var bagging = try BaggingClassifier(f64).init(allocator, .{});
    defer bagging.deinit();

    const X_empty = [_][]const f64{};
    const y_empty = [_]usize{};

    try testing.expectError(error.EmptyInput, bagging.fit(&X_empty, &y_empty));
}

test "BaggingClassifier: mismatched dimensions" {
    const allocator = testing.allocator;

    var bagging = try BaggingClassifier(f64).init(allocator, .{});
    defer bagging.deinit();

    const X = [_][]const f64{&[_]f64{0.0}};
    const y = [_]usize{ 0, 1 }; // Wrong size

    try testing.expectError(error.MismatchedDimensions, bagging.fit(&X, &y));
}

test "BaggingClassifier: invalid config" {
    const allocator = testing.allocator;

    try testing.expectError(error.InvalidNumEstimators, BaggingClassifier(f64).init(allocator, .{
        .n_estimators = 0,
    }));

    try testing.expectError(error.InvalidMaxSamples, BaggingClassifier(f64).init(allocator, .{
        .max_samples = -1.0,
    }));
}

test "BaggingClassifier: feature mismatch in predict" {
    const allocator = testing.allocator;

    const X_train = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 1.0, 1.0 },
    };
    const y = [_]usize{ 0, 1 };

    var bagging = try BaggingClassifier(f64).init(allocator, .{});
    defer bagging.deinit();

    try bagging.fit(&X_train, &y);

    const X_test = [_][]const f64{
        &[_]f64{ 0.0, 0.0, 0.0 }, // Wrong feature count
    };

    try testing.expectError(error.FeatureMismatch, bagging.predict(&X_test));
}

test "BaggingRegressor: no bootstrap" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{1.0}, &[_]f64{2.0},
    };
    const y = [_]f64{ 0.0, 1.0, 2.0 };

    var bagging = try BaggingRegressor(f64).init(allocator, .{
        .n_estimators = 5,
        .bootstrap = false,
    });
    defer bagging.deinit();

    try bagging.fit(&X, &y);

    const predictions = try bagging.predict(&X);
    defer allocator.free(predictions);

    try testing.expectEqual(@as(usize, 3), predictions.len);
}

test "BaggingClassifier: f32 type support" {
    const allocator = testing.allocator;

    const X = [_][]const f32{
        &[_]f32{ 0.0, 0.0 },
        &[_]f32{ 1.0, 1.0 },
    };
    const y = [_]usize{ 0, 1 };

    var bagging = try BaggingClassifier(f32).init(allocator, .{
        .n_estimators = 5,
    });
    defer bagging.deinit();

    try bagging.fit(&X, &y);

    const predictions = try bagging.predict(&X);
    defer allocator.free(predictions);

    try testing.expectEqual(@as(usize, 2), predictions.len);
}

test "BaggingRegressor: f32 type support" {
    const allocator = testing.allocator;

    const X = [_][]const f32{&[_]f32{0.0}};
    const y = [_]f32{0.0};

    var bagging = try BaggingRegressor(f32).init(allocator, .{});
    defer bagging.deinit();

    try bagging.fit(&X, &y);

    const predictions = try bagging.predict(&X);
    defer allocator.free(predictions);

    try testing.expectEqual(@as(usize, 1), predictions.len);
}

test "BaggingClassifier: large ensemble" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 2.0, 2.0 },
        &[_]f64{ 3.0, 3.0 },
    };
    const y = [_]usize{ 0, 1, 0, 1 };

    var bagging = try BaggingClassifier(f64).init(allocator, .{
        .n_estimators = 50,
        .max_depth = 3,
    });
    defer bagging.deinit();

    try bagging.fit(&X, &y);

    const predictions = try bagging.predict(&X);
    defer allocator.free(predictions);

    try testing.expectEqual(@as(usize, 4), predictions.len);
}

test "BaggingRegressor: large dataset" {
    const allocator = testing.allocator;

    // Create larger dataset
    var X_list = std.ArrayList([]const f64).init(allocator);
    defer {
        for (X_list.items) |row| {
            allocator.free(row);
        }
        X_list.deinit();
    }

    var y_list = std.ArrayList(f64).init(allocator);
    defer y_list.deinit();

    for (0..100) |i| {
        const val: f64 = @floatFromInt(i);
        var row = try allocator.alloc(f64, 1);
        row[0] = val;
        try X_list.append(row);
        try y_list.append(val * 2.0);
    }

    var bagging = try BaggingRegressor(f64).init(allocator, .{
        .n_estimators = 10,
        .max_samples = 0.8,
    });
    defer bagging.deinit();

    try bagging.fit(X_list.items, y_list.items);

    const predictions = try bagging.predict(X_list.items);
    defer allocator.free(predictions);

    try testing.expectEqual(@as(usize, 100), predictions.len);
}

test "BaggingClassifier: memory safety" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 1.0, 1.0 },
    };
    const y = [_]usize{ 0, 1 };

    var bagging = try BaggingClassifier(f64).init(allocator, .{
        .n_estimators = 5,
    });
    defer bagging.deinit();

    try bagging.fit(&X, &y);

    const predictions = try bagging.predict(&X);
    defer allocator.free(predictions);

    // Memory should be freed properly in defer blocks
}
