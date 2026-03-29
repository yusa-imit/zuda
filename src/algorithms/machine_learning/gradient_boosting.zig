const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

/// Gradient Boosting Machine for classification and regression.
///
/// Gradient Boosting builds an ensemble of decision trees sequentially, where each tree
/// is trained to correct the errors (residuals) of the previous trees. This creates a
/// strong learner from many weak learners through gradient descent in function space.
///
/// Key differences from Random Forest:
/// - Sequential training (trees correct previous errors) vs. parallel independent trees
/// - Typically uses shallow trees (weak learners) vs. deep trees
/// - Higher risk of overfitting, requires careful tuning
/// - Often higher accuracy on structured/tabular data
///
/// Time complexity:
/// - Training: O(n_trees × n × m × depth) where n = samples, m = features, depth = tree depth
/// - Prediction: O(n_trees × depth) per sample
///
/// Space complexity: O(n_trees × nodes) where nodes ≈ O(depth × m) per tree
///
/// Algorithm:
/// 1. Initialize predictions with base learner (mean for regression, log-odds for classification)
/// 2. For each boosting iteration:
///    - Compute residuals (negative gradients of loss function)
///    - Train weak learner (shallow tree) to predict residuals
///    - Update predictions: F_m = F_{m-1} + learning_rate × tree_m
/// 3. Final prediction is sum of all trees weighted by learning rate
///
/// Loss functions:
/// - Regression: Mean Squared Error (MSE) → residuals = y - F(x)
/// - Classification: Log Loss (cross-entropy) → residuals = y - sigmoid(F(x))
///
/// Hyperparameters:
/// - n_trees: Number of boosting iterations (default: 100)
/// - learning_rate: Shrinkage parameter (default: 0.1) - smaller = more robust but needs more trees
/// - max_depth: Maximum tree depth (default: 3) - shallow trees prevent overfitting
/// - min_samples_split: Minimum samples to split node (default: 2)
/// - subsample: Fraction of samples for stochastic boosting (default: 1.0)
/// - random_seed: For reproducible subsampling
///
/// Use cases:
/// - Tabular data prediction (Kaggle competitions standard)
/// - Ranking systems (LambdaMART for search)
/// - Financial forecasting
/// - Customer churn prediction
/// - Fraud detection
pub fn GradientBoosting(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Task type: regression or binary classification
        pub const TaskType = enum {
            regression,
            classification,
        };

        /// Configuration for gradient boosting
        pub const Config = struct {
            task: TaskType = .regression,
            n_trees: usize = 100,
            learning_rate: T = 0.1,
            max_depth: usize = 3,
            min_samples_split: usize = 2,
            subsample: T = 1.0, // 1.0 = use all data, <1.0 = stochastic gradient boosting
            random_seed: u64 = 42,
        };

        /// Simple decision tree node for weak learners
        const TreeNode = struct {
            is_leaf: bool,
            // Internal node
            feature_idx: usize = 0,
            threshold: T = 0,
            left: ?*TreeNode = null,
            right: ?*TreeNode = null,
            // Leaf node
            value: T = 0,

            fn deinit(self: *TreeNode, allocator: Allocator) void {
                if (self.left) |left| {
                    left.deinit(allocator);
                    allocator.destroy(left);
                }
                if (self.right) |right| {
                    right.deinit(allocator);
                    allocator.destroy(right);
                }
            }
        };

        allocator: Allocator,
        config: Config,
        trees: ArrayList(*TreeNode),
        base_prediction: T, // Initial prediction (mean for regression, log-odds for classification)

        /// Initialize gradient boosting model.
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(allocator: Allocator, config: Config) !Self {
            return Self{
                .allocator = allocator,
                .config = config,
                .trees = ArrayList(*TreeNode).init(allocator),
                .base_prediction = 0,
            };
        }

        /// Free all allocated memory.
        ///
        /// Time: O(n_trees × nodes)
        /// Space: O(1)
        pub fn deinit(self: *Self) void {
            for (self.trees.items) |tree| {
                tree.deinit(self.allocator);
                self.allocator.destroy(tree);
            }
            self.trees.deinit();
        }

        /// Train gradient boosting model.
        ///
        /// Time: O(n_trees × n × m × depth)
        /// Space: O(n + n_trees × nodes)
        ///
        /// Parameters:
        /// - X: Training features [n_samples × n_features]
        /// - y: Training targets [n_samples]
        pub fn fit(self: *Self, X: []const []const T, y: []const T) !void {
            const n_samples = X.len;
            if (n_samples == 0) return error.EmptyData;
            if (y.len != n_samples) return error.ShapeMismatch;

            // Initialize base prediction
            self.base_prediction = switch (self.config.task) {
                .regression => blk: {
                    // Mean of targets
                    var sum: T = 0;
                    for (y) |val| sum += val;
                    break :blk sum / @as(T, @floatFromInt(n_samples));
                },
                .classification => blk: {
                    // Log-odds of positive class proportion
                    var pos_count: usize = 0;
                    for (y) |val| {
                        if (val > 0.5) pos_count += 1;
                    }
                    const p = @as(T, @floatFromInt(pos_count)) / @as(T, @floatFromInt(n_samples));
                    const p_clamped = @max(@min(p, 0.9999), 0.0001); // Avoid log(0)
                    break :blk @log(p_clamped / (1.0 - p_clamped));
                },
            };

            // Allocate working arrays
            var predictions = try self.allocator.alloc(T, n_samples);
            defer self.allocator.free(predictions);
            var residuals = try self.allocator.alloc(T, n_samples);
            defer self.allocator.free(residuals);

            // Initialize predictions with base value
            for (predictions) |*pred| pred.* = self.base_prediction;

            // Random number generator for subsampling
            var prng = std.Random.DefaultPrng.init(self.config.random_seed);
            const random = prng.random();

            // Boosting iterations
            var iter: usize = 0;
            while (iter < self.config.n_trees) : (iter += 1) {
                // Compute residuals (negative gradients)
                switch (self.config.task) {
                    .regression => {
                        // MSE gradient: -(y - F(x))
                        for (y, predictions, 0..) |target, pred, i| {
                            residuals[i] = target - pred;
                        }
                    },
                    .classification => {
                        // Log loss gradient: -(y - sigmoid(F(x)))
                        for (y, predictions, 0..) |target, pred, i| {
                            const prob = sigmoid(pred);
                            residuals[i] = target - prob;
                        }
                    },
                }

                // Subsample data for stochastic gradient boosting
                var sample_indices: []usize = undefined;
                var use_subsample = false;
                if (self.config.subsample < 1.0) {
                    const subsample_size = @as(usize, @intFromFloat(@as(T, @floatFromInt(n_samples)) * self.config.subsample));
                    if (subsample_size > 0) {
                        sample_indices = try self.allocator.alloc(usize, subsample_size);
                        defer self.allocator.free(sample_indices);

                        // Random sampling with replacement
                        for (sample_indices) |*idx| {
                            idx.* = random.intRangeLessThan(usize, 0, n_samples);
                        }
                        use_subsample = true;
                    }
                }

                // Build weak learner (shallow tree) to predict residuals
                const tree = if (use_subsample)
                    try self.buildTree(X, residuals, sample_indices, 0)
                else
                    try self.buildTreeFull(X, residuals, 0);

                try self.trees.append(tree);

                // Update predictions with new tree (scaled by learning rate)
                for (X, 0..) |x, i| {
                    const tree_pred = self.predictTree(tree, x);
                    predictions[i] += self.config.learning_rate * tree_pred;
                }
            }
        }

        /// Build decision tree on full dataset.
        ///
        /// Time: O(n × m × depth)
        /// Space: O(depth) for recursion
        fn buildTreeFull(self: *Self, X: []const []const T, y: []const T, depth: usize) !*TreeNode {
            const node = try self.allocator.create(TreeNode);
            errdefer self.allocator.destroy(node);

            const n_samples = X.len;

            // Check stopping criteria
            if (depth >= self.config.max_depth or n_samples < self.config.min_samples_split) {
                // Create leaf node with mean of residuals
                var sum: T = 0;
                for (y) |val| sum += val;
                node.* = TreeNode{
                    .is_leaf = true,
                    .value = sum / @as(T, @floatFromInt(n_samples)),
                };
                return node;
            }

            // Find best split
            const split = try self.findBestSplit(X, y);

            if (split.gain <= 0) {
                // No beneficial split found, create leaf
                var sum: T = 0;
                for (y) |val| sum += val;
                node.* = TreeNode{
                    .is_leaf = true,
                    .value = sum / @as(T, @floatFromInt(n_samples)),
                };
                return node;
            }

            // Split data
            var left_X = ArrayList([]const T).init(self.allocator);
            defer left_X.deinit();
            var left_y = ArrayList(T).init(self.allocator);
            defer left_y.deinit();
            var right_X = ArrayList([]const T).init(self.allocator);
            defer right_X.deinit();
            var right_y = ArrayList(T).init(self.allocator);
            defer right_y.deinit();

            for (X, y) |x, target| {
                if (x[split.feature_idx] <= split.threshold) {
                    try left_X.append(x);
                    try left_y.append(target);
                } else {
                    try right_X.append(x);
                    try right_y.append(target);
                }
            }

            // Recursively build subtrees
            const left_child = try self.buildTreeFull(left_X.items, left_y.items, depth + 1);
            errdefer {
                left_child.deinit(self.allocator);
                self.allocator.destroy(left_child);
            }

            const right_child = try self.buildTreeFull(right_X.items, right_y.items, depth + 1);
            errdefer {
                right_child.deinit(self.allocator);
                self.allocator.destroy(right_child);
            }

            node.* = TreeNode{
                .is_leaf = false,
                .feature_idx = split.feature_idx,
                .threshold = split.threshold,
                .left = left_child,
                .right = right_child,
            };

            return node;
        }

        /// Build decision tree on subsampled data.
        ///
        /// Time: O(k × m × depth) where k = subsample_size
        /// Space: O(depth) for recursion
        fn buildTree(self: *Self, X: []const []const T, y: []const T, indices: []const usize, depth: usize) !*TreeNode {
            const node = try self.allocator.create(TreeNode);
            errdefer self.allocator.destroy(node);

            const n_samples = indices.len;

            // Check stopping criteria
            if (depth >= self.config.max_depth or n_samples < self.config.min_samples_split) {
                // Create leaf node with mean of sampled residuals
                var sum: T = 0;
                for (indices) |idx| sum += y[idx];
                node.* = TreeNode{
                    .is_leaf = true,
                    .value = sum / @as(T, @floatFromInt(n_samples)),
                };
                return node;
            }

            // Find best split on subsample
            const split = try self.findBestSplitSubsample(X, y, indices);

            if (split.gain <= 0) {
                // No beneficial split found, create leaf
                var sum: T = 0;
                for (indices) |idx| sum += y[idx];
                node.* = TreeNode{
                    .is_leaf = true,
                    .value = sum / @as(T, @floatFromInt(n_samples)),
                };
                return node;
            }

            // Split indices
            var left_indices = ArrayList(usize).init(self.allocator);
            defer left_indices.deinit();
            var right_indices = ArrayList(usize).init(self.allocator);
            defer right_indices.deinit();

            for (indices) |idx| {
                if (X[idx][split.feature_idx] <= split.threshold) {
                    try left_indices.append(idx);
                } else {
                    try right_indices.append(idx);
                }
            }

            // Recursively build subtrees
            const left_child = try self.buildTree(X, y, left_indices.items, depth + 1);
            errdefer {
                left_child.deinit(self.allocator);
                self.allocator.destroy(left_child);
            }

            const right_child = try self.buildTree(X, y, right_indices.items, depth + 1);
            errdefer {
                right_child.deinit(self.allocator);
                self.allocator.destroy(right_child);
            }

            node.* = TreeNode{
                .is_leaf = false,
                .feature_idx = split.feature_idx,
                .threshold = split.threshold,
                .left = left_child,
                .right = right_child,
            };

            return node;
        }

        /// Split information for tree building
        const Split = struct {
            feature_idx: usize,
            threshold: T,
            gain: T,
        };

        /// Find best split point for current node.
        ///
        /// Time: O(n × m) for n samples, m features
        fn findBestSplit(self: *Self, X: []const []const T, y: []const T) !Split {
            const n_samples = X.len;
            const n_features = X[0].len;

            var best_split = Split{
                .feature_idx = 0,
                .threshold = 0,
                .gain = -std.math.inf(T),
            };

            // Compute parent variance
            var parent_sum: T = 0;
            var parent_sum_sq: T = 0;
            for (y) |val| {
                parent_sum += val;
                parent_sum_sq += val * val;
            }
            const parent_mean = parent_sum / @as(T, @floatFromInt(n_samples));
            const parent_var = (parent_sum_sq / @as(T, @floatFromInt(n_samples))) - (parent_mean * parent_mean);

            // Try each feature
            for (0..n_features) |feat_idx| {
                // Get unique sorted values for this feature
                var values = try self.allocator.alloc(T, n_samples);
                defer self.allocator.free(values);
                for (X, 0..) |x, i| values[i] = x[feat_idx];
                std.mem.sort(T, values, {}, comptime std.sort.asc(T));

                // Try splits between consecutive values
                var prev_val = values[0];
                for (values[1..]) |val| {
                    if (val == prev_val) continue;
                    const threshold = (prev_val + val) / 2.0;

                    // Split data
                    var left_sum: T = 0;
                    var left_sum_sq: T = 0;
                    var left_count: usize = 0;
                    var right_sum: T = 0;
                    var right_sum_sq: T = 0;
                    var right_count: usize = 0;

                    for (X, y) |x, target| {
                        if (x[feat_idx] <= threshold) {
                            left_sum += target;
                            left_sum_sq += target * target;
                            left_count += 1;
                        } else {
                            right_sum += target;
                            right_sum_sq += target * target;
                            right_count += 1;
                        }
                    }

                    if (left_count == 0 or right_count == 0) continue;

                    // Compute variance reduction (information gain for regression)
                    const left_mean = left_sum / @as(T, @floatFromInt(left_count));
                    const left_var = (left_sum_sq / @as(T, @floatFromInt(left_count))) - (left_mean * left_mean);
                    const right_mean = right_sum / @as(T, @floatFromInt(right_count));
                    const right_var = (right_sum_sq / @as(T, @floatFromInt(right_count))) - (right_mean * right_mean);

                    const weighted_child_var = (@as(T, @floatFromInt(left_count)) * left_var + @as(T, @floatFromInt(right_count)) * right_var) / @as(T, @floatFromInt(n_samples));
                    const gain = parent_var - weighted_child_var;

                    if (gain > best_split.gain) {
                        best_split = Split{
                            .feature_idx = feat_idx,
                            .threshold = threshold,
                            .gain = gain,
                        };
                    }

                    prev_val = val;
                }
            }

            return best_split;
        }

        /// Find best split point for subsampled data.
        ///
        /// Time: O(k × m) for k samples, m features
        fn findBestSplitSubsample(self: *Self, X: []const []const T, y: []const T, indices: []const usize) !Split {
            const n_samples = indices.len;
            const n_features = X[0].len;

            var best_split = Split{
                .feature_idx = 0,
                .threshold = 0,
                .gain = -std.math.inf(T),
            };

            // Compute parent variance
            var parent_sum: T = 0;
            var parent_sum_sq: T = 0;
            for (indices) |idx| {
                const val = y[idx];
                parent_sum += val;
                parent_sum_sq += val * val;
            }
            const parent_mean = parent_sum / @as(T, @floatFromInt(n_samples));
            const parent_var = (parent_sum_sq / @as(T, @floatFromInt(n_samples))) - (parent_mean * parent_mean);

            // Try each feature
            for (0..n_features) |feat_idx| {
                // Get unique sorted values for this feature from subsample
                var values = try self.allocator.alloc(T, n_samples);
                defer self.allocator.free(values);
                for (indices, 0..) |idx, i| values[i] = X[idx][feat_idx];
                std.mem.sort(T, values, {}, comptime std.sort.asc(T));

                // Try splits between consecutive values
                var prev_val = values[0];
                for (values[1..]) |val| {
                    if (val == prev_val) continue;
                    const threshold = (prev_val + val) / 2.0;

                    // Split data
                    var left_sum: T = 0;
                    var left_sum_sq: T = 0;
                    var left_count: usize = 0;
                    var right_sum: T = 0;
                    var right_sum_sq: T = 0;
                    var right_count: usize = 0;

                    for (indices) |idx| {
                        const target = y[idx];
                        if (X[idx][feat_idx] <= threshold) {
                            left_sum += target;
                            left_sum_sq += target * target;
                            left_count += 1;
                        } else {
                            right_sum += target;
                            right_sum_sq += target * target;
                            right_count += 1;
                        }
                    }

                    if (left_count == 0 or right_count == 0) continue;

                    // Compute variance reduction
                    const left_mean = left_sum / @as(T, @floatFromInt(left_count));
                    const left_var = (left_sum_sq / @as(T, @floatFromInt(left_count))) - (left_mean * left_mean);
                    const right_mean = right_sum / @as(T, @floatFromInt(right_count));
                    const right_var = (right_sum_sq / @as(T, @floatFromInt(right_count))) - (right_mean * right_mean);

                    const weighted_child_var = (@as(T, @floatFromInt(left_count)) * left_var + @as(T, @floatFromInt(right_count)) * right_var) / @as(T, @floatFromInt(n_samples));
                    const gain = parent_var - weighted_child_var;

                    if (gain > best_split.gain) {
                        best_split = Split{
                            .feature_idx = feat_idx,
                            .threshold = threshold,
                            .gain = gain,
                        };
                    }

                    prev_val = val;
                }
            }

            return best_split;
        }

        /// Predict with a single tree.
        ///
        /// Time: O(depth)
        fn predictTree(self: *Self, node: *TreeNode, x: []const T) T {
            if (node.is_leaf) {
                return node.value;
            }

            if (x[node.feature_idx] <= node.threshold) {
                return self.predictTree(node.left.?, x);
            } else {
                return self.predictTree(node.right.?, x);
            }
        }

        /// Predict single sample.
        ///
        /// Time: O(n_trees × depth)
        /// Space: O(1)
        pub fn predict(self: *Self, x: []const T) T {
            var pred = self.base_prediction;
            for (self.trees.items) |tree| {
                pred += self.config.learning_rate * self.predictTree(tree, x);
            }

            return switch (self.config.task) {
                .regression => pred,
                .classification => sigmoid(pred), // Convert log-odds to probability
            };
        }

        /// Predict batch of samples.
        ///
        /// Time: O(n_samples × n_trees × depth)
        /// Space: O(n_samples)
        pub fn predictBatch(self: *Self, X: []const []const T, out: []T) !void {
            if (X.len != out.len) return error.ShapeMismatch;
            for (X, 0..) |x, i| {
                out[i] = self.predict(x);
            }
        }

        /// Compute R² score for regression or accuracy for classification.
        ///
        /// Time: O(n_samples × n_trees × depth)
        /// Space: O(n_samples)
        pub fn score(self: *Self, X: []const []const T, y: []const T) !T {
            const n_samples = X.len;
            if (y.len != n_samples) return error.ShapeMismatch;

            const predictions = try self.allocator.alloc(T, n_samples);
            defer self.allocator.free(predictions);

            try self.predictBatch(X, predictions);

            return switch (self.config.task) {
                .regression => blk: {
                    // R² score
                    var ss_res: T = 0;
                    var ss_tot: T = 0;
                    var y_mean: T = 0;
                    for (y) |val| y_mean += val;
                    y_mean /= @as(T, @floatFromInt(n_samples));

                    for (y, predictions) |true_val, pred_val| {
                        const residual = true_val - pred_val;
                        ss_res += residual * residual;
                        const deviation = true_val - y_mean;
                        ss_tot += deviation * deviation;
                    }

                    if (ss_tot == 0) break :blk 1.0;
                    break :blk 1.0 - (ss_res / ss_tot);
                },
                .classification => blk: {
                    // Accuracy
                    var correct: usize = 0;
                    for (y, predictions) |true_val, pred_val| {
                        const pred_class: T = if (pred_val >= 0.5) 1.0 else 0.0;
                        if (pred_class == true_val) correct += 1;
                    }
                    break :blk @as(T, @floatFromInt(correct)) / @as(T, @floatFromInt(n_samples));
                },
            };
        }

        /// Sigmoid activation function.
        fn sigmoid(x: T) T {
            return 1.0 / (1.0 + @exp(-x));
        }
    };
}

// Tests
test "GradientBoosting - basic regression" {
    const allocator = testing.allocator;

    // Simple regression: y = 2x + 1
    const X = [_][]const f64{
        &[_]f64{0.0},
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
        &[_]f64{4.0},
    };
    const y = [_]f64{ 1.0, 3.0, 5.0, 7.0, 9.0 };

    const config = GradientBoosting(f64).Config{
        .task = .regression,
        .n_trees = 50,
        .learning_rate = 0.1,
        .max_depth = 3,
    };
    var gb = try GradientBoosting(f64).init(allocator, config);
    defer gb.deinit();

    try gb.fit(&X, &y);

    // Test predictions
    const pred0 = gb.predict(&[_]f64{0.0});
    const pred2 = gb.predict(&[_]f64{2.0});
    const pred4 = gb.predict(&[_]f64{4.0});

    try testing.expect(@abs(pred0 - 1.0) < 1.0);
    try testing.expect(@abs(pred2 - 5.0) < 1.0);
    try testing.expect(@abs(pred4 - 9.0) < 1.0);
}

test "GradientBoosting - binary classification" {
    const allocator = testing.allocator;

    // XOR-like problem (linearly non-separable)
    const X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 0.0, 1.0 },
        &[_]f64{ 1.0, 0.0 },
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 0.1, 0.1 },
        &[_]f64{ 0.9, 0.9 },
    };
    const y = [_]f64{ 0.0, 1.0, 1.0, 0.0, 0.0, 0.0 };

    const config = GradientBoosting(f64).Config{
        .task = .classification,
        .n_trees = 100,
        .learning_rate = 0.1,
        .max_depth = 2,
    };
    var gb = try GradientBoosting(f64).init(allocator, config);
    defer gb.deinit();

    try gb.fit(&X, &y);

    // Test predictions
    const pred00 = gb.predict(&[_]f64{ 0.0, 0.0 });
    const pred01 = gb.predict(&[_]f64{ 0.0, 1.0 });
    const pred10 = gb.predict(&[_]f64{ 1.0, 0.0 });
    const pred11 = gb.predict(&[_]f64{ 1.0, 1.0 });

    // Check classification (with some tolerance)
    try testing.expect(pred00 < 0.5); // Class 0
    try testing.expect(pred01 > 0.5); // Class 1
    try testing.expect(pred10 > 0.5); // Class 1
    try testing.expect(pred11 < 0.5); // Class 0
}

test "GradientBoosting - predictBatch" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
    };
    const y = [_]f64{ 2.0, 4.0, 6.0 };

    const config = GradientBoosting(f64).Config{
        .task = .regression,
        .n_trees = 30,
        .learning_rate = 0.1,
    };
    var gb = try GradientBoosting(f64).init(allocator, config);
    defer gb.deinit();

    try gb.fit(&X, &y);

    var predictions = [_]f64{0} ** 3;
    try gb.predictBatch(&X, &predictions);

    for (predictions, y) |pred, true_val| {
        try testing.expect(@abs(pred - true_val) < 1.5);
    }
}

test "GradientBoosting - score regression" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
        &[_]f64{4.0},
    };
    const y = [_]f64{ 2.0, 4.0, 6.0, 8.0 };

    const config = GradientBoosting(f64).Config{
        .task = .regression,
        .n_trees = 50,
        .learning_rate = 0.1,
    };
    var gb = try GradientBoosting(f64).init(allocator, config);
    defer gb.deinit();

    try gb.fit(&X, &y);

    const r2 = try gb.score(&X, &y);
    try testing.expect(r2 > 0.8); // Good fit
}

test "GradientBoosting - score classification" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 0.0, 1.0 },
        &[_]f64{ 1.0, 0.0 },
        &[_]f64{ 1.0, 1.0 },
    };
    const y = [_]f64{ 0.0, 1.0, 1.0, 0.0 };

    const config = GradientBoosting(f64).Config{
        .task = .classification,
        .n_trees = 100,
        .learning_rate = 0.1,
        .max_depth = 2,
    };
    var gb = try GradientBoosting(f64).init(allocator, config);
    defer gb.deinit();

    try gb.fit(&X, &y);

    const accuracy = try gb.score(&X, &y);
    try testing.expect(accuracy >= 0.75); // At least 3/4 correct
}

test "GradientBoosting - stochastic gradient boosting" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{1.0}, &[_]f64{2.0}, &[_]f64{3.0},
        &[_]f64{4.0}, &[_]f64{5.0}, &[_]f64{6.0}, &[_]f64{7.0},
    };
    const y = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

    const config = GradientBoosting(f64).Config{
        .task = .regression,
        .n_trees = 50,
        .learning_rate = 0.1,
        .subsample = 0.8, // Use 80% of data per iteration
        .random_seed = 123,
    };
    var gb = try GradientBoosting(f64).init(allocator, config);
    defer gb.deinit();

    try gb.fit(&X, &y);

    const r2 = try gb.score(&X, &y);
    try testing.expect(r2 > 0.7); // Still good fit with subsampling
}

test "GradientBoosting - learning rate effect" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{1.0}, &[_]f64{2.0}, &[_]f64{3.0},
    };
    const y = [_]f64{ 2.0, 4.0, 6.0 };

    // High learning rate (faster but less stable)
    const config_high = GradientBoosting(f64).Config{
        .task = .regression,
        .n_trees = 20,
        .learning_rate = 0.3,
    };
    var gb_high = try GradientBoosting(f64).init(allocator, config_high);
    defer gb_high.deinit();
    try gb_high.fit(&X, &y);

    // Low learning rate (slower but more stable)
    const config_low = GradientBoosting(f64).Config{
        .task = .regression,
        .n_trees = 20,
        .learning_rate = 0.05,
    };
    var gb_low = try GradientBoosting(f64).init(allocator, config_low);
    defer gb_low.deinit();
    try gb_low.fit(&X, &y);

    // Both should work, but with different convergence
    const score_high = try gb_high.score(&X, &y);
    const score_low = try gb_low.score(&X, &y);
    try testing.expect(score_high > 0.5);
    try testing.expect(score_low > 0.5);
}

test "GradientBoosting - max depth effect" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 }, &[_]f64{ 0.0, 1.0 },
        &[_]f64{ 1.0, 0.0 }, &[_]f64{ 1.0, 1.0 },
    };
    const y = [_]f64{ 0.0, 1.0, 1.0, 0.0 };

    // Shallow trees (more regularization)
    const config_shallow = GradientBoosting(f64).Config{
        .task = .classification,
        .n_trees = 100,
        .max_depth = 1,
    };
    var gb_shallow = try GradientBoosting(f64).init(allocator, config_shallow);
    defer gb_shallow.deinit();
    try gb_shallow.fit(&X, &y);

    // Deeper trees (more capacity)
    const config_deep = GradientBoosting(f64).Config{
        .task = .classification,
        .n_trees = 100,
        .max_depth = 3,
    };
    var gb_deep = try GradientBoosting(f64).init(allocator, config_deep);
    defer gb_deep.deinit();
    try gb_deep.fit(&X, &y);

    const accuracy_shallow = try gb_shallow.score(&X, &y);
    const accuracy_deep = try gb_deep.score(&X, &y);

    // Deeper trees should fit better on this complex pattern
    try testing.expect(accuracy_deep >= accuracy_shallow);
}

test "GradientBoosting - empty data" {
    const allocator = testing.allocator;

    const X: []const []const f64 = &[_][]const f64{};
    const y: []const f64 = &[_]f64{};

    const config = GradientBoosting(f64).Config{};
    var gb = try GradientBoosting(f64).init(allocator, config);
    defer gb.deinit();

    try testing.expectError(error.EmptyData, gb.fit(X, y));
}

test "GradientBoosting - shape mismatch" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
    };
    const y = [_]f64{1.0}; // Wrong length

    const config = GradientBoosting(f64).Config{};
    var gb = try GradientBoosting(f64).init(allocator, config);
    defer gb.deinit();

    try testing.expectError(error.ShapeMismatch, gb.fit(&X, &y));
}

test "GradientBoosting - f32 support" {
    const allocator = testing.allocator;

    const X = [_][]const f32{
        &[_]f32{1.0}, &[_]f32{2.0}, &[_]f32{3.0},
    };
    const y = [_]f32{ 2.0, 4.0, 6.0 };

    const config = GradientBoosting(f32).Config{
        .task = .regression,
        .n_trees = 30,
    };
    var gb = try GradientBoosting(f32).init(allocator, config);
    defer gb.deinit();

    try gb.fit(&X, &y);

    const pred = gb.predict(&[_]f32{2.0});
    try testing.expect(@abs(pred - 4.0) < 1.0);
}

test "GradientBoosting - nonlinear regression" {
    const allocator = testing.allocator;

    // Quadratic function: y = x^2
    const X = [_][]const f64{
        &[_]f64{-2.0}, &[_]f64{-1.0}, &[_]f64{0.0},
        &[_]f64{1.0}, &[_]f64{2.0},
    };
    const y = [_]f64{ 4.0, 1.0, 0.0, 1.0, 4.0 };

    const config = GradientBoosting(f64).Config{
        .task = .regression,
        .n_trees = 50,
        .learning_rate = 0.1,
        .max_depth = 3,
    };
    var gb = try GradientBoosting(f64).init(allocator, config);
    defer gb.deinit();

    try gb.fit(&X, &y);

    const r2 = try gb.score(&X, &y);
    try testing.expect(r2 > 0.8); // Should fit well
}

test "GradientBoosting - multiclass simulation via multiple models" {
    const allocator = testing.allocator;

    // Simple 3-class problem (one-vs-rest approach)
    const X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 }, // Class 0
        &[_]f64{ 1.0, 1.0 }, // Class 1
        &[_]f64{ 2.0, 2.0 }, // Class 2
    };

    // Binary labels for class 0 vs rest
    const y_class0 = [_]f64{ 1.0, 0.0, 0.0 };

    const config = GradientBoosting(f64).Config{
        .task = .classification,
        .n_trees = 50,
        .learning_rate = 0.1,
    };
    var gb = try GradientBoosting(f64).init(allocator, config);
    defer gb.deinit();

    try gb.fit(&X, &y_class0);

    const pred0 = gb.predict(&[_]f64{ 0.0, 0.0 });
    try testing.expect(pred0 > 0.5); // Should predict class 0
}
