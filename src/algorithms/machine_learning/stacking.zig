/// Stacking (Stacked Generalization)
///
/// Meta-learning ensemble that trains a meta-model on base estimator predictions.
/// Unlike voting (simple aggregation), stacking learns optimal combination weights.
///
/// Architecture:
/// - **Level 0 (base estimators)**: Train diverse models on training data
/// - **Level 1 (meta-model)**: Train on base predictions using out-of-fold predictions
///
/// Key features:
/// - **Cross-validated predictions**: Uses k-fold CV to generate meta-features (prevents overfitting)
/// - **Heterogeneous models**: Combines different algorithm types (trees, SVM, KNN, etc.)
/// - **Learned combination**: Meta-model learns how to weight base predictions
/// - **State-of-the-art**: Often wins Kaggle competitions
///
/// Algorithm (k-fold stacking):
/// 1. Split training data into k folds
/// 2. For each base estimator:
///    - For each fold i:
///      - Train on folds ≠ i, predict on fold i (out-of-fold predictions)
///    - Concatenate out-of-fold predictions → meta-features
///    - Train on full training set for test-time predictions
/// 3. Train meta-model on meta-features (base predictions)
/// 4. At test time: base predictions → meta-model → final prediction
///
/// Time complexity: O(k × m × T_base + T_meta) training, O(m × T_base + T_meta) prediction
///   where k = folds, m = base estimators, T_base/T_meta = estimator costs
/// Space complexity: O(k × m × n + S_base + S_meta) for CV predictions + estimators
///
/// Use cases:
/// - Kaggle competitions (ensemble of diverse models)
/// - Combining algorithm strengths (SVM margins + tree interactions + KNN local structure)
/// - When simple voting underperforms (meta-model finds better combination)
/// - High-stakes predictions (medical, finance)
///
/// Differences from other ensembles:
/// - **vs Voting**: Stacking learns combination, voting uses fixed rule
/// - **vs Bagging**: Stacking uses heterogeneous models + meta-learner, bagging uses homogeneous + simple average
/// - **vs Boosting**: Stacking trains in parallel, boosting trains sequentially
///
/// Example (regression):
/// ```zig
/// var stacking = try StackingRegressor(f64).init(allocator, .{
///     .n_folds = 5,
///     .random_seed = 42,
///     .use_base_features = false, // Meta-model sees only base predictions
/// });
/// defer stacking.deinit();
///
/// // Train base estimators (simplified example with decision trees)
/// try stacking.fit(X_train, y_train);
///
/// const predictions = try stacking.predict(X_test);
/// defer allocator.free(predictions);
/// ```

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const DecisionTree = @import("decision_tree.zig").DecisionTree;
const LinearRegression = @import("linear_regression.zig").LinearRegression;
const LogisticRegression = @import("logistic_regression.zig").LogisticRegression;

/// Stacking Regressor
///
/// Stacked generalization for regression using cross-validated base predictions.
/// Base estimators: Multiple decision trees with different depths (diverse models).
/// Meta-model: Linear regression (learns optimal combination).
///
/// Time: O(k × m × n × log n) training, O(m × depth) prediction (k=folds, m=base estimators)
/// Space: O(k × m × n + nodes) for CV predictions + trees
pub fn StackingRegressor(comptime T: type) type {
    return struct {
        allocator: Allocator,
        base_estimators: []DecisionTree(T),
        meta_model: LinearRegression(T),
        config: Config,
        fitted: bool = false,

        const Self = @This();

        pub const Config = struct {
            n_folds: u32 = 5,
            random_seed: u64 = 42,
            use_base_features: bool = false, // Include original features in meta-model
            // Base estimators: 3 trees with different depths for diversity
            base_configs: []const BaseConfig = &.{
                .{ .max_depth = 3, .min_samples_split = 5 }, // Shallow tree (high bias, low variance)
                .{ .max_depth = 6, .min_samples_split = 2 }, // Medium tree
                .{ .max_depth = 10, .min_samples_split = 2 }, // Deep tree (low bias, high variance)
            },
        };

        pub const BaseConfig = struct {
            max_depth: u32,
            min_samples_split: u32,
        };

        /// Initialize stacking regressor
        ///
        /// Time: O(m) where m = base estimators
        /// Space: O(m)
        pub fn init(allocator: Allocator, config: Config) !Self {
            if (config.n_folds < 2) return error.InvalidNumFolds;
            if (config.base_configs.len == 0) return error.NoBaseEstimators;

            // Allocate base estimators
            const n_estimators = config.base_configs.len;
            const estimators = try allocator.alloc(DecisionTree(T), n_estimators);
            errdefer allocator.free(estimators);

            for (estimators, config.base_configs) |*est, base_cfg| {
                est.* = DecisionTree(T).init(allocator, .{
                    .tree_type = .regression,
                    .max_depth = base_cfg.max_depth,
                    .min_samples_split = base_cfg.min_samples_split,
                    .min_samples_leaf = 1,
                    .criterion = .mse,
                });
            }

            // Initialize meta-model (linear regression)
            const meta = LinearRegression(T).init(allocator);

            return Self{
                .allocator = allocator,
                .base_estimators = estimators,
                .meta_model = meta,
                .config = config,
            };
        }

        /// Clean up resources
        pub fn deinit(self: *Self) void {
            for (self.base_estimators) |*est| {
                est.deinit();
            }
            self.allocator.free(self.base_estimators);
            self.meta_model.deinit();
        }

        /// Fit stacking regressor using k-fold cross-validation
        ///
        /// Algorithm:
        /// 1. Generate out-of-fold predictions for meta-training
        /// 2. Train base estimators on full training set
        /// 3. Train meta-model on base predictions
        ///
        /// Time: O(k × m × n × log n) where k=folds, m=base estimators, n=samples
        /// Space: O(k × m × n) for CV predictions
        pub fn fit(self: *Self, X: []const []const T, y: []const T) !void {
            if (X.len == 0 or y.len == 0) return error.EmptyData;
            if (X.len != y.len) return error.MismatchedDimensions;

            // Step 1: Generate out-of-fold predictions for meta-training
            const meta_features = try self.generateMetaFeatures(X, y);
            defer {
                for (meta_features) |row| {
                    self.allocator.free(row);
                }
                self.allocator.free(meta_features);
            }

            // Step 2: Train base estimators on full training set
            for (self.base_estimators) |*est| {
                try est.fit(X, y);
            }

            // Step 3: Train meta-model on base predictions
            try self.meta_model.fit(meta_features, y);

            self.fitted = true;
        }

        /// Generate out-of-fold predictions via k-fold CV
        ///
        /// Returns: [n_samples][n_estimators] meta-features
        /// Time: O(k × m × n × log n)
        /// Space: O(k × m × n)
        fn generateMetaFeatures(self: *Self, X: []const []const T, y: []const T) ![][]T {
            const n_samples = X.len;
            const n_estimators = self.base_estimators.len;
            const n_folds = self.config.n_folds;

            // Allocate meta-features [n_samples][n_estimators]
            const meta_features = try self.allocator.alloc([]T, n_samples);
            errdefer self.allocator.free(meta_features);

            for (meta_features) |*row| {
                row.* = try self.allocator.alloc(T, n_estimators);
            }
            errdefer {
                for (meta_features) |row| {
                    self.allocator.free(row);
                }
            }

            // K-fold cross-validation
            const fold_size = n_samples / n_folds;

            for (0..n_folds) |fold_idx| {
                const val_start = fold_idx * fold_size;
                const val_end = if (fold_idx == n_folds - 1) n_samples else (fold_idx + 1) * fold_size;

                // Split into train/validation folds
                var train_X = std.ArrayList([]const T).init(self.allocator);
                defer train_X.deinit();
                var train_y = std.ArrayList(T).init(self.allocator);
                defer train_y.deinit();

                for (0..n_samples) |i| {
                    if (i < val_start or i >= val_end) {
                        try train_X.append(X[i]);
                        try train_y.append(y[i]);
                    }
                }

                // Train each base estimator on train fold
                for (self.base_estimators, 0..) |_, est_idx| {
                    // Create temporary estimator for this fold
                    var fold_est = DecisionTree(T).init(self.allocator, .{
                        .tree_type = .regression,
                        .max_depth = self.config.base_configs[est_idx].max_depth,
                        .min_samples_split = self.config.base_configs[est_idx].min_samples_split,
                        .min_samples_leaf = 1,
                        .criterion = .mse,
                    });
                    defer fold_est.deinit();

                    try fold_est.fit(train_X.items, train_y.items);

                    // Predict on validation fold (out-of-fold predictions)
                    for (val_start..val_end) |i| {
                        const pred = try fold_est.predict(&[_][]const T{X[i]});
                        defer self.allocator.free(pred);
                        meta_features[i][est_idx] = pred[0];
                    }
                }
            }

            return meta_features;
        }

        /// Predict on new data
        ///
        /// Time: O(m × depth + meta_predict) where m=base estimators
        /// Space: O(n × m) for base predictions
        pub fn predict(self: *Self, X: []const []const T) ![]T {
            if (!self.fitted) return error.NotFitted;
            if (X.len == 0) return error.EmptyData;

            const n_samples = X.len;
            const n_estimators = self.base_estimators.len;

            // Step 1: Get base estimator predictions
            const base_predictions = try self.allocator.alloc([]T, n_samples);
            defer {
                for (base_predictions) |row| {
                    self.allocator.free(row);
                }
                self.allocator.free(base_predictions);
            }

            for (base_predictions) |*row| {
                row.* = try self.allocator.alloc(T, n_estimators);
            }

            for (0..n_samples) |i| {
                for (self.base_estimators, 0..) |*est, est_idx| {
                    const pred = try est.predict(&[_][]const T{X[i]});
                    defer self.allocator.free(pred);
                    base_predictions[i][est_idx] = pred[0];
                }
            }

            // Step 2: Meta-model predicts from base predictions
            const final_predictions = try self.meta_model.predict(base_predictions);
            return final_predictions;
        }

        /// Reset the regressor
        pub fn reset(self: *Self) void {
            for (self.base_estimators) |*est| {
                est.deinit();
                est.* = DecisionTree(T).init(self.allocator, .{
                    .tree_type = .regression,
                    .max_depth = 10,
                    .min_samples_split = 2,
                    .min_samples_leaf = 1,
                    .criterion = .mse,
                });
            }
            self.meta_model.deinit();
            self.meta_model = LinearRegression(T).init(self.allocator);
            self.fitted = false;
        }
    };
}

/// Stacking Classifier
///
/// Stacked generalization for classification using cross-validated base predictions.
/// Base estimators: Multiple decision trees with different depths (diverse models).
/// Meta-model: Logistic regression (learns optimal combination).
///
/// Time: O(k × m × n × log n) training, O(m × depth) prediction
/// Space: O(k × m × n + nodes) for CV predictions + trees
pub fn StackingClassifier(comptime T: type) type {
    return struct {
        allocator: Allocator,
        base_estimators: []DecisionTree(T),
        meta_model: LogisticRegression(T),
        n_classes: usize = 0,
        config: Config,
        fitted: bool = false,

        const Self = @This();

        pub const Config = struct {
            n_folds: u32 = 5,
            random_seed: u64 = 42,
            use_base_features: bool = false,
            base_configs: []const BaseConfig = &.{
                .{ .max_depth = 3, .min_samples_split = 5 },
                .{ .max_depth = 6, .min_samples_split = 2 },
                .{ .max_depth = 10, .min_samples_split = 2 },
            },
        };

        pub const BaseConfig = struct {
            max_depth: u32,
            min_samples_split: u32,
        };

        /// Initialize stacking classifier
        pub fn init(allocator: Allocator, config: Config) !Self {
            if (config.n_folds < 2) return error.InvalidNumFolds;
            if (config.base_configs.len == 0) return error.NoBaseEstimators;

            const n_estimators = config.base_configs.len;
            const estimators = try allocator.alloc(DecisionTree(T), n_estimators);
            errdefer allocator.free(estimators);

            for (estimators, config.base_configs) |*est, base_cfg| {
                est.* = DecisionTree(T).init(allocator, .{
                    .tree_type = .classification,
                    .max_depth = base_cfg.max_depth,
                    .min_samples_split = base_cfg.min_samples_split,
                    .min_samples_leaf = 1,
                    .criterion = .gini,
                });
            }

            const meta = LogisticRegression(T).init(allocator);

            return Self{
                .allocator = allocator,
                .base_estimators = estimators,
                .meta_model = meta,
                .config = config,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.base_estimators) |*est| {
                est.deinit();
            }
            self.allocator.free(self.base_estimators);
            self.meta_model.deinit();
        }

        /// Fit stacking classifier using k-fold cross-validation
        pub fn fit(self: *Self, X: []const []const T, y: []const usize) !void {
            if (X.len == 0 or y.len == 0) return error.EmptyData;
            if (X.len != y.len) return error.MismatchedDimensions;

            // Detect number of classes
            var max_class: usize = 0;
            for (y) |label| {
                if (label > max_class) max_class = label;
            }
            self.n_classes = max_class + 1;

            // Step 1: Generate out-of-fold predictions
            const meta_features = try self.generateMetaFeatures(X, y);
            defer {
                for (meta_features) |row| {
                    self.allocator.free(row);
                }
                self.allocator.free(meta_features);
            }

            // Step 2: Train base estimators on full training set
            for (self.base_estimators) |*est| {
                try est.fit(X, y);
            }

            // Step 3: Train meta-model on base predictions
            // Convert meta-features to binary format for logistic regression
            try self.meta_model.fit(meta_features, y);

            self.fitted = true;
        }

        /// Generate out-of-fold predictions via k-fold CV
        fn generateMetaFeatures(self: *Self, X: []const []const T, y: []const usize) ![][]T {
            const n_samples = X.len;
            const n_estimators = self.base_estimators.len;
            const n_folds = self.config.n_folds;

            const meta_features = try self.allocator.alloc([]T, n_samples);
            errdefer self.allocator.free(meta_features);

            for (meta_features) |*row| {
                row.* = try self.allocator.alloc(T, n_estimators);
            }
            errdefer {
                for (meta_features) |row| {
                    self.allocator.free(row);
                }
            }

            const fold_size = n_samples / n_folds;

            for (0..n_folds) |fold_idx| {
                const val_start = fold_idx * fold_size;
                const val_end = if (fold_idx == n_folds - 1) n_samples else (fold_idx + 1) * fold_size;

                var train_X = std.ArrayList([]const T).init(self.allocator);
                defer train_X.deinit();
                var train_y = std.ArrayList(usize).init(self.allocator);
                defer train_y.deinit();

                for (0..n_samples) |i| {
                    if (i < val_start or i >= val_end) {
                        try train_X.append(X[i]);
                        try train_y.append(y[i]);
                    }
                }

                for (self.base_estimators, 0..) |_, est_idx| {
                    var fold_est = DecisionTree(T).init(self.allocator, .{
                        .tree_type = .classification,
                        .max_depth = self.config.base_configs[est_idx].max_depth,
                        .min_samples_split = self.config.base_configs[est_idx].min_samples_split,
                        .min_samples_leaf = 1,
                        .criterion = .gini,
                    });
                    defer fold_est.deinit();

                    try fold_est.fit(train_X.items, train_y.items);

                    for (val_start..val_end) |i| {
                        const pred = try fold_est.predict(&[_][]const T{X[i]});
                        defer self.allocator.free(pred);
                        meta_features[i][est_idx] = @floatFromInt(pred[0]);
                    }
                }
            }

            return meta_features;
        }

        /// Predict class labels
        pub fn predict(self: *Self, X: []const []const T) ![]usize {
            if (!self.fitted) return error.NotFitted;
            if (X.len == 0) return error.EmptyData;

            const n_samples = X.len;
            const n_estimators = self.base_estimators.len;

            const base_predictions = try self.allocator.alloc([]T, n_samples);
            defer {
                for (base_predictions) |row| {
                    self.allocator.free(row);
                }
                self.allocator.free(base_predictions);
            }

            for (base_predictions) |*row| {
                row.* = try self.allocator.alloc(T, n_estimators);
            }

            for (0..n_samples) |i| {
                for (self.base_estimators, 0..) |*est, est_idx| {
                    const pred = try est.predict(&[_][]const T{X[i]});
                    defer self.allocator.free(pred);
                    base_predictions[i][est_idx] = @floatFromInt(pred[0]);
                }
            }

            const final_predictions = try self.meta_model.predict(base_predictions);
            return final_predictions;
        }

        pub fn reset(self: *Self) void {
            for (self.base_estimators) |*est| {
                est.deinit();
                est.* = DecisionTree(T).init(self.allocator, .{
                    .tree_type = .classification,
                    .max_depth = 10,
                    .min_samples_split = 2,
                    .min_samples_leaf = 1,
                    .criterion = .gini,
                });
            }
            self.meta_model.deinit();
            self.meta_model = LogisticRegression(T).init(self.allocator);
            self.n_classes = 0;
            self.fitted = false;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "stacking regressor: basic initialization" {
    const allocator = testing.allocator;

    var stacking = try StackingRegressor(f64).init(allocator, .{
        .n_folds = 5,
        .base_configs = &.{
            .{ .max_depth = 3, .min_samples_split = 5 },
            .{ .max_depth = 6, .min_samples_split = 2 },
        },
    });
    defer stacking.deinit();

    try testing.expect(!stacking.fitted);
    try testing.expectEqual(@as(usize, 2), stacking.base_estimators.len);
}

test "stacking regressor: simple linear data" {
    const allocator = testing.allocator;

    var stacking = try StackingRegressor(f64).init(allocator, .{
        .n_folds = 3,
        .base_configs = &.{
            .{ .max_depth = 3, .min_samples_split = 2 },
            .{ .max_depth = 5, .min_samples_split = 2 },
        },
    });
    defer stacking.deinit();

    // y = 2x + 3
    const X = [_][]const f64{
        &[_]f64{1.0}, &[_]f64{2.0}, &[_]f64{3.0},
        &[_]f64{4.0}, &[_]f64{5.0}, &[_]f64{6.0},
        &[_]f64{7.0}, &[_]f64{8.0}, &[_]f64{9.0},
    };
    const y = [_]f64{ 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0 };

    try stacking.fit(&X, &y);
    try testing.expect(stacking.fitted);

    const predictions = try stacking.predict(&X);
    defer allocator.free(predictions);

    // Check predictions are close to true values
    for (predictions, y) |pred, true_val| {
        try testing.expect(@abs(pred - true_val) < 3.0); // Relaxed tolerance
    }
}

test "stacking regressor: quadratic pattern" {
    const allocator = testing.allocator;

    var stacking = try StackingRegressor(f64).init(allocator, .{
        .n_folds = 3,
        .base_configs = &.{
            .{ .max_depth = 5, .min_samples_split = 2 },
            .{ .max_depth = 8, .min_samples_split = 2 },
            .{ .max_depth = 10, .min_samples_split = 2 },
        },
    });
    defer stacking.deinit();

    // y = x²
    const X = [_][]const f64{
        &[_]f64{1.0}, &[_]f64{2.0}, &[_]f64{3.0}, &[_]f64{4.0},
        &[_]f64{5.0}, &[_]f64{6.0}, &[_]f64{7.0}, &[_]f64{8.0},
        &[_]f64{9.0}, &[_]f64{10.0}, &[_]f64{11.0}, &[_]f64{12.0},
    };
    const y = [_]f64{ 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0, 144.0 };

    try stacking.fit(&X, &y);

    const test_X = [_][]const f64{&[_]f64{5.0}};
    const predictions = try stacking.predict(&test_X);
    defer allocator.free(predictions);

    // Should predict close to 25
    try testing.expect(@abs(predictions[0] - 25.0) < 10.0);
}

test "stacking regressor: multi-feature" {
    const allocator = testing.allocator;

    var stacking = try StackingRegressor(f64).init(allocator, .{
        .n_folds = 3,
        .base_configs = &.{
            .{ .max_depth = 4, .min_samples_split = 2 },
            .{ .max_depth = 6, .min_samples_split = 2 },
        },
    });
    defer stacking.deinit();

    // y = 2x₁ + 3x₂
    const X = [_][]const f64{
        &[_]f64{ 1.0, 1.0 }, &[_]f64{ 2.0, 1.0 }, &[_]f64{ 3.0, 1.0 },
        &[_]f64{ 1.0, 2.0 }, &[_]f64{ 2.0, 2.0 }, &[_]f64{ 3.0, 2.0 },
        &[_]f64{ 1.0, 3.0 }, &[_]f64{ 2.0, 3.0 }, &[_]f64{ 3.0, 3.0 },
    };
    const y = [_]f64{ 5.0, 7.0, 9.0, 8.0, 10.0, 12.0, 11.0, 13.0, 15.0 };

    try stacking.fit(&X, &y);

    const predictions = try stacking.predict(&X);
    defer allocator.free(predictions);

    var sum_error: f64 = 0.0;
    for (predictions, y) |pred, true_val| {
        sum_error += @abs(pred - true_val);
    }
    const mae = sum_error / @as(f64, @floatFromInt(y.len));
    try testing.expect(mae < 3.0);
}

test "stacking regressor: reset functionality" {
    const allocator = testing.allocator;

    var stacking = try StackingRegressor(f64).init(allocator, .{});
    defer stacking.deinit();

    const X = [_][]const f64{ &[_]f64{1.0}, &[_]f64{2.0}, &[_]f64{3.0} };
    const y = [_]f64{ 2.0, 4.0, 6.0 };

    try stacking.fit(&X, &y);
    try testing.expect(stacking.fitted);

    stacking.reset();
    try testing.expect(!stacking.fitted);
}

test "stacking classifier: basic initialization" {
    const allocator = testing.allocator;

    var stacking = try StackingClassifier(f64).init(allocator, .{
        .n_folds = 5,
        .base_configs = &.{
            .{ .max_depth = 3, .min_samples_split = 5 },
            .{ .max_depth = 6, .min_samples_split = 2 },
        },
    });
    defer stacking.deinit();

    try testing.expect(!stacking.fitted);
    try testing.expectEqual(@as(usize, 2), stacking.base_estimators.len);
}

test "stacking classifier: binary classification" {
    const allocator = testing.allocator;

    var stacking = try StackingClassifier(f64).init(allocator, .{
        .n_folds = 3,
        .base_configs = &.{
            .{ .max_depth = 3, .min_samples_split = 2 },
            .{ .max_depth = 5, .min_samples_split = 2 },
        },
    });
    defer stacking.deinit();

    // Linearly separable data
    const X = [_][]const f64{
        &[_]f64{ 1.0, 1.0 }, &[_]f64{ 1.5, 1.2 }, &[_]f64{ 2.0, 1.5 },
        &[_]f64{ 5.0, 5.0 }, &[_]f64{ 5.5, 5.2 }, &[_]f64{ 6.0, 5.5 },
        &[_]f64{ 1.2, 1.5 }, &[_]f64{ 5.2, 5.5 }, &[_]f64{ 1.8, 1.8 },
    };
    const y = [_]usize{ 0, 0, 0, 1, 1, 1, 0, 1, 0 };

    try stacking.fit(&X, &y);
    try testing.expect(stacking.fitted);
    try testing.expectEqual(@as(usize, 2), stacking.n_classes);

    const predictions = try stacking.predict(&X);
    defer allocator.free(predictions);

    // Check accuracy
    var correct: usize = 0;
    for (predictions, y) |pred, true_label| {
        if (pred == true_label) correct += 1;
    }
    const accuracy = @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(y.len));
    try testing.expect(accuracy > 0.7); // At least 70% accuracy
}

test "stacking classifier: multi-class" {
    const allocator = testing.allocator;

    var stacking = try StackingClassifier(f64).init(allocator, .{
        .n_folds = 3,
        .base_configs = &.{
            .{ .max_depth = 5, .min_samples_split = 2 },
            .{ .max_depth = 8, .min_samples_split = 2 },
        },
    });
    defer stacking.deinit();

    // 3-class problem
    const X = [_][]const f64{
        &[_]f64{ 1.0, 1.0 }, &[_]f64{ 1.5, 1.2 }, &[_]f64{ 2.0, 1.5 },
        &[_]f64{ 5.0, 5.0 }, &[_]f64{ 5.5, 5.2 }, &[_]f64{ 6.0, 5.5 },
        &[_]f64{ 1.0, 5.0 }, &[_]f64{ 1.5, 5.5 }, &[_]f64{ 2.0, 6.0 },
        &[_]f64{ 1.2, 1.3 }, &[_]f64{ 5.2, 5.3 }, &[_]f64{ 1.3, 5.3 },
    };
    const y = [_]usize{ 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2 };

    try stacking.fit(&X, &y);
    try testing.expectEqual(@as(usize, 3), stacking.n_classes);

    const predictions = try stacking.predict(&X);
    defer allocator.free(predictions);

    var correct: usize = 0;
    for (predictions, y) |pred, true_label| {
        if (pred == true_label) correct += 1;
    }
    const accuracy = @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(y.len));
    try testing.expect(accuracy > 0.6);
}

test "stacking classifier: XOR-like pattern" {
    const allocator = testing.allocator;

    var stacking = try StackingClassifier(f64).init(allocator, .{
        .n_folds = 2,
        .base_configs = &.{
            .{ .max_depth = 5, .min_samples_split = 2 },
            .{ .max_depth = 8, .min_samples_split = 2 },
            .{ .max_depth = 10, .min_samples_split = 2 },
        },
    });
    defer stacking.deinit();

    // XOR: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0
    const X = [_][]const f64{
        &[_]f64{ 0.1, 0.1 }, &[_]f64{ 0.2, 0.15 },
        &[_]f64{ 0.1, 0.9 }, &[_]f64{ 0.15, 0.85 },
        &[_]f64{ 0.9, 0.1 }, &[_]f64{ 0.85, 0.15 },
        &[_]f64{ 0.9, 0.9 }, &[_]f64{ 0.85, 0.85 },
    };
    const y = [_]usize{ 0, 0, 1, 1, 1, 1, 0, 0 };

    try stacking.fit(&X, &y);

    const predictions = try stacking.predict(&X);
    defer allocator.free(predictions);

    var correct: usize = 0;
    for (predictions, y) |pred, true_label| {
        if (pred == true_label) correct += 1;
    }
    const accuracy = @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(y.len));
    try testing.expect(accuracy > 0.6); // Relaxed for XOR
}

test "stacking classifier: reset functionality" {
    const allocator = testing.allocator;

    var stacking = try StackingClassifier(f64).init(allocator, .{});
    defer stacking.deinit();

    const X = [_][]const f64{ &[_]f64{ 1.0, 1.0 }, &[_]f64{ 2.0, 2.0 }, &[_]f64{ 3.0, 3.0 } };
    const y = [_]usize{ 0, 1, 0 };

    try stacking.fit(&X, &y);
    try testing.expect(stacking.fitted);

    stacking.reset();
    try testing.expect(!stacking.fitted);
    try testing.expectEqual(@as(usize, 0), stacking.n_classes);
}

test "stacking regressor: large dataset" {
    const allocator = testing.allocator;

    var stacking = try StackingRegressor(f64).init(allocator, .{
        .n_folds = 5,
        .base_configs = &.{
            .{ .max_depth = 5, .min_samples_split = 5 },
            .{ .max_depth = 8, .min_samples_split = 3 },
        },
    });
    defer stacking.deinit();

    // Generate 100 samples: y = 3x + 2
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
        const x = @as(f64, @floatFromInt(i)) / 10.0;
        const row = try allocator.alloc(f64, 1);
        row[0] = x;
        try X_list.append(row);
        try y_list.append(3.0 * x + 2.0);
    }

    try stacking.fit(X_list.items, y_list.items);

    const test_X = [_][]const f64{&[_]f64{5.0}};
    const predictions = try stacking.predict(&test_X);
    defer allocator.free(predictions);

    // Should predict close to 3*5+2=17
    try testing.expect(@abs(predictions[0] - 17.0) < 5.0);
}

test "stacking regressor: error on empty data" {
    const allocator = testing.allocator;

    var stacking = try StackingRegressor(f64).init(allocator, .{});
    defer stacking.deinit();

    const X = [_][]const f64{};
    const y = [_]f64{};

    try testing.expectError(error.EmptyData, stacking.fit(&X, &y));
}

test "stacking classifier: error on empty data" {
    const allocator = testing.allocator;

    var stacking = try StackingClassifier(f64).init(allocator, .{});
    defer stacking.deinit();

    const X = [_][]const f64{};
    const y = [_]usize{};

    try testing.expectError(error.EmptyData, stacking.fit(&X, &y));
}

test "stacking regressor: error on predict before fit" {
    const allocator = testing.allocator;

    var stacking = try StackingRegressor(f64).init(allocator, .{});
    defer stacking.deinit();

    const X = [_][]const f64{&[_]f64{1.0}};
    try testing.expectError(error.NotFitted, stacking.predict(&X));
}

test "stacking classifier: error on predict before fit" {
    const allocator = testing.allocator;

    var stacking = try StackingClassifier(f64).init(allocator, .{});
    defer stacking.deinit();

    const X = [_][]const f64{&[_]f64{1.0}};
    try testing.expectError(error.NotFitted, stacking.predict(&X));
}

test "stacking regressor: error on invalid config" {
    const allocator = testing.allocator;

    // n_folds < 2
    try testing.expectError(error.InvalidNumFolds, StackingRegressor(f64).init(allocator, .{
        .n_folds = 1,
    }));

    // No base estimators
    try testing.expectError(error.NoBaseEstimators, StackingRegressor(f64).init(allocator, .{
        .base_configs = &.{},
    }));
}

test "stacking classifier: error on invalid config" {
    const allocator = testing.allocator;

    try testing.expectError(error.InvalidNumFolds, StackingClassifier(f64).init(allocator, .{
        .n_folds = 1,
    }));

    try testing.expectError(error.NoBaseEstimators, StackingClassifier(f64).init(allocator, .{
        .base_configs = &.{},
    }));
}

test "stacking regressor: f32 support" {
    const allocator = testing.allocator;

    var stacking = try StackingRegressor(f32).init(allocator, .{
        .n_folds = 3,
        .base_configs = &.{
            .{ .max_depth = 3, .min_samples_split = 2 },
            .{ .max_depth = 5, .min_samples_split = 2 },
        },
    });
    defer stacking.deinit();

    const X = [_][]const f32{
        &[_]f32{1.0}, &[_]f32{2.0}, &[_]f32{3.0},
        &[_]f32{4.0}, &[_]f32{5.0}, &[_]f32{6.0},
    };
    const y = [_]f32{ 2.0, 4.0, 6.0, 8.0, 10.0, 12.0 };

    try stacking.fit(&X, &y);
    const predictions = try stacking.predict(&X);
    defer allocator.free(predictions);

    try testing.expect(predictions.len == y.len);
}

test "stacking classifier: f32 support" {
    const allocator = testing.allocator;

    var stacking = try StackingClassifier(f32).init(allocator, .{
        .n_folds = 3,
        .base_configs = &.{
            .{ .max_depth = 3, .min_samples_split = 2 },
        },
    });
    defer stacking.deinit();

    const X = [_][]const f32{
        &[_]f32{ 1.0, 1.0 }, &[_]f32{ 2.0, 2.0 }, &[_]f32{ 5.0, 5.0 },
        &[_]f32{ 6.0, 6.0 }, &[_]f32{ 1.5, 1.5 }, &[_]f32{ 5.5, 5.5 },
    };
    const y = [_]usize{ 0, 0, 1, 1, 0, 1 };

    try stacking.fit(&X, &y);
    const predictions = try stacking.predict(&X);
    defer allocator.free(predictions);

    try testing.expect(predictions.len == y.len);
}

test "stacking: memory safety" {
    const allocator = testing.allocator;

    // Regressor
    {
        var stacking = try StackingRegressor(f64).init(allocator, .{});
        defer stacking.deinit();

        const X = [_][]const f64{ &[_]f64{1.0}, &[_]f64{2.0}, &[_]f64{3.0} };
        const y = [_]f64{ 2.0, 4.0, 6.0 };

        try stacking.fit(&X, &y);
        const pred = try stacking.predict(&X);
        defer allocator.free(pred);
    }

    // Classifier
    {
        var stacking = try StackingClassifier(f64).init(allocator, .{});
        defer stacking.deinit();

        const X = [_][]const f64{ &[_]f64{1.0}, &[_]f64{2.0}, &[_]f64{3.0} };
        const y = [_]usize{ 0, 1, 0 };

        try stacking.fit(&X, &y);
        const pred = try stacking.predict(&X);
        defer allocator.free(pred);
    }
}
