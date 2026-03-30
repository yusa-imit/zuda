const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

/// eXtreme Gradient Boosting (XGBoost) for classification and regression.
///
/// XGBoost is an advanced gradient boosting algorithm that improves upon standard gradient
/// boosting with regularization, second-order optimization, and efficient tree construction.
///
/// Key innovations over standard gradient boosting:
/// 1. **Second-order optimization**: Uses both gradient (g) and hessian (h) for better convergence
/// 2. **Regularized objectives**: L1 (alpha) and L2 (lambda) penalties on leaf weights
/// 3. **Gain-based pruning**: Only creates splits with positive gain after regularization
/// 4. **Column subsampling**: Samples features per tree (like random forest)
/// 5. **Shrinkage**: Learning rate to prevent overfitting
///
/// Objective function:
/// ```
/// Obj = Σ loss(y_i, ŷ_i) + Σ Ω(f_t)
/// where Ω(f_t) = γT + 0.5λΣw_j² + αΣ|w_j|
/// T = number of leaves, w_j = leaf weights
/// ```
///
/// Split gain formula:
/// ```
/// Gain = 0.5 * [G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ
/// where G = sum of gradients, H = sum of hessians
/// ```
///
/// Time complexity:
/// - Training: O(n_trees × n × m × depth × log(n)) with histogram-based splits
/// - Prediction: O(n_trees × depth) per sample
///
/// Space complexity: O(n_trees × nodes + n × m) for trees and histograms
///
/// Hyperparameters:
/// - n_trees: Number of boosting rounds (default: 100)
/// - learning_rate (eta): Step size shrinkage (default: 0.3) - higher = faster training, more overfitting
/// - max_depth: Maximum tree depth (default: 6) - deeper than standard GBM
/// - min_child_weight: Minimum sum of hessian in child (default: 1.0) - like min_samples_leaf
/// - gamma: Minimum loss reduction for split (default: 0.0) - regularization
/// - lambda: L2 regularization on weights (default: 1.0)
/// - alpha: L1 regularization on weights (default: 0.0)
/// - subsample: Row sampling ratio (default: 1.0)
/// - colsample_bytree: Column sampling ratio per tree (default: 1.0)
/// - random_seed: For reproducible sampling
///
/// Use cases:
/// - Kaggle competitions (most winning solutions use XGBoost)
/// - Structured/tabular data prediction
/// - Ranking and recommendation systems
/// - Time series forecasting
/// - Credit scoring and fraud detection
///
/// Time: O(n_trees × n × m × depth × log(n)) training, O(n_trees × depth) prediction
/// Space: O(n_trees × nodes + n × m)
pub fn XGBoost(comptime T: type) type {
    if (T != f32 and T != f64) {
        @compileError("XGBoost only supports f32 and f64");
    }

    return struct {
        const Self = @This();

        /// Task type: regression or binary classification
        pub const TaskType = enum {
            regression,
            classification,
        };

        /// Configuration for XGBoost
        pub const Config = struct {
            task: TaskType = .regression,
            n_trees: usize = 100,
            learning_rate: T = 0.3, // eta - higher than standard GBM
            max_depth: usize = 6, // deeper than standard GBM
            min_child_weight: T = 1.0, // minimum sum of hessian
            gamma: T = 0.0, // minimum loss reduction for split
            lambda: T = 1.0, // L2 regularization
            alpha: T = 0.0, // L1 regularization
            subsample: T = 1.0, // row sampling
            colsample_bytree: T = 1.0, // column sampling per tree
            random_seed: u64 = 42,
        };

        /// Gradient and Hessian statistics for a node
        const GradStats = struct {
            sum_grad: T,
            sum_hess: T,
            count: usize,

            fn init() GradStats {
                return .{
                    .sum_grad = 0,
                    .sum_hess = 0,
                    .count = 0,
                };
            }

            fn add(self: *GradStats, grad: T, hess: T) void {
                self.sum_grad += grad;
                self.sum_hess += hess;
                self.count += 1;
            }

            /// Calculate optimal leaf weight: -G / (H + lambda)
            fn optimalWeight(self: GradStats, lambda: T, alpha: T) T {
                const denom = self.sum_hess + lambda;
                if (denom == 0) return 0;

                // Apply L1 regularization (soft thresholding)
                var weight = -self.sum_grad / denom;
                if (alpha > 0) {
                    if (weight > alpha) {
                        weight -= alpha;
                    } else if (weight < -alpha) {
                        weight += alpha;
                    } else {
                        weight = 0;
                    }
                }
                return weight;
            }

            /// Calculate gain for this node: -0.5 * G² / (H + lambda)
            fn gain(self: GradStats, lambda: T, gamma: T) T {
                const denom = self.sum_hess + lambda;
                if (denom == 0) return 0;
                return -0.5 * (self.sum_grad * self.sum_grad) / denom - gamma;
            }
        };

        /// Decision tree node with XGBoost statistics
        const TreeNode = struct {
            is_leaf: bool,
            // Internal node
            feature_idx: usize = 0,
            threshold: T = 0,
            left: ?*TreeNode = null,
            right: ?*TreeNode = null,
            // Leaf node
            weight: T = 0, // Optimal leaf weight from second-order approximation
            stats: GradStats = GradStats.init(),

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
        base_prediction: T,
        selected_features: ArrayList(usize), // For column subsampling
        prng: std.Random.DefaultPrng,

        /// Initialize XGBoost model.
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(allocator: Allocator, config: Config) !Self {
            return Self{
                .allocator = allocator,
                .config = config,
                .trees = ArrayList(*TreeNode).init(allocator),
                .base_prediction = 0,
                .selected_features = ArrayList(usize).init(allocator),
                .prng = std.Random.DefaultPrng.init(config.random_seed),
            };
        }

        /// Free all resources.
        ///
        /// Time: O(n_trees × nodes)
        /// Space: O(1)
        pub fn deinit(self: *Self) void {
            for (self.trees.items) |tree| {
                tree.deinit(self.allocator);
                self.allocator.destroy(tree);
            }
            self.trees.deinit();
            self.selected_features.deinit();
        }

        /// Train XGBoost model on dataset.
        ///
        /// Parameters:
        /// - X: Training features [n_samples × n_features]
        /// - y: Training labels [n_samples]
        ///
        /// Time: O(n_trees × n × m × depth × log(n))
        /// Space: O(n_trees × nodes + n × m)
        pub fn fit(self: *Self, X: []const []const T, y: []const T) !void {
            if (X.len == 0 or y.len == 0 or X.len != y.len) return error.InvalidInput;
            const n_samples = X.len;
            const n_features = X[0].len;

            // Initialize base prediction
            self.base_prediction = switch (self.config.task) {
                .regression => blk: {
                    var sum: T = 0;
                    for (y) |yi| sum += yi;
                    break :blk sum / @as(T, @floatFromInt(n_samples));
                },
                .classification => blk: {
                    // Log-odds: log(p / (1-p)) where p = mean(y)
                    var sum: T = 0;
                    for (y) |yi| sum += yi;
                    const p = sum / @as(T, @floatFromInt(n_samples));
                    const p_clamped = @max(1e-7, @min(1.0 - 1e-7, p));
                    break :blk @log(p_clamped / (1.0 - p_clamped));
                },
            };

            // Initialize predictions
            var predictions = try self.allocator.alloc(T, n_samples);
            defer self.allocator.free(predictions);
            @memset(predictions, self.base_prediction);

            // Allocate gradients and hessians
            const gradients = try self.allocator.alloc(T, n_samples);
            defer self.allocator.free(gradients);
            const hessians = try self.allocator.alloc(T, n_samples);
            defer self.allocator.free(hessians);

            // Sample indices for subsampling
            var sample_indices = try self.allocator.alloc(usize, n_samples);
            defer self.allocator.free(sample_indices);
            for (0..n_samples) |i| sample_indices[i] = i;

            // Boosting iterations
            for (0..self.config.n_trees) |_| {
                // Compute gradients and hessians
                self.computeGradients(predictions, y, gradients, hessians);

                // Row subsampling (stochastic gradient boosting)
                const n_subsample = @as(usize, @intFromFloat(@as(T, @floatFromInt(n_samples)) * self.config.subsample));
                if (n_subsample < n_samples) {
                    // Fisher-Yates shuffle for first n_subsample elements
                    var rand = self.prng.random();
                    for (0..n_subsample) |i| {
                        const j = i + rand.uintLessThan(usize, n_samples - i);
                        std.mem.swap(usize, &sample_indices[i], &sample_indices[j]);
                    }
                }

                // Column subsampling (sample features for this tree)
                self.selected_features.clearRetainingCapacity();
                const n_selected = @as(usize, @intFromFloat(@as(T, @floatFromInt(n_features)) * self.config.colsample_bytree));
                var feature_indices = try self.allocator.alloc(usize, n_features);
                defer self.allocator.free(feature_indices);
                for (0..n_features) |i| feature_indices[i] = i;

                if (n_selected < n_features) {
                    var rand = self.prng.random();
                    for (0..n_selected) |i| {
                        const j = i + rand.uintLessThan(usize, n_features - i);
                        std.mem.swap(usize, &feature_indices[i], &feature_indices[j]);
                    }
                }
                for (0..n_selected) |i| {
                    try self.selected_features.append(self.allocator, feature_indices[i]);
                }

                // Build tree with second-order approximation
                const tree = try self.buildTree(X, gradients, hessians, sample_indices[0..n_subsample], 0);
                try self.trees.append(self.allocator, tree);

                // Update predictions with learning rate (shrinkage)
                for (0..n_samples) |i| {
                    const pred = predictTreeInternal(tree, X[i]);
                    predictions[i] += self.config.learning_rate * pred;
                }
            }
        }

        /// Compute gradients and hessians based on loss function.
        fn computeGradients(self: *const Self, predictions: []const T, y: []const T, gradients: []T, hessians: []T) void {
            switch (self.config.task) {
                .regression => {
                    // MSE loss: L = 0.5 * (y - ŷ)²
                    // gradient = ∂L/∂ŷ = ŷ - y
                    // hessian = ∂²L/∂ŷ² = 1
                    for (0..predictions.len) |i| {
                        gradients[i] = predictions[i] - y[i];
                        hessians[i] = 1.0;
                    }
                },
                .classification => {
                    // Log loss: L = -y*log(p) - (1-y)*log(1-p) where p = sigmoid(ŷ)
                    // gradient = ∂L/∂ŷ = p - y
                    // hessian = ∂²L/∂ŷ² = p * (1 - p)
                    for (0..predictions.len) |i| {
                        const p = sigmoid(predictions[i]);
                        gradients[i] = p - y[i];
                        hessians[i] = p * (1.0 - p);
                    }
                },
            }
        }

        /// Build a single tree using second-order approximation.
        fn buildTree(
            self: *Self,
            X: []const []const T,
            gradients: []const T,
            hessians: []const T,
            indices: []const usize,
            depth: usize,
        ) !*TreeNode {
            const node = try self.allocator.create(TreeNode);
            node.* = TreeNode{
                .is_leaf = false,
                .stats = GradStats.init(),
            };

            // Accumulate gradient statistics
            for (indices) |idx| {
                node.stats.add(gradients[idx], hessians[idx]);
            }

            // Check stopping criteria
            if (depth >= self.config.max_depth or
                indices.len < 2 or
                node.stats.sum_hess < self.config.min_child_weight)
            {
                // Create leaf node with optimal weight
                node.is_leaf = true;
                node.weight = node.stats.optimalWeight(self.config.lambda, self.config.alpha);
                return node;
            }

            // Find best split with gain-based criterion
            var best_gain: T = 0;
            var best_feature: usize = 0;
            var best_threshold: T = 0;
            var best_left_stats = GradStats.init();
            var best_right_stats = GradStats.init();
            var found_split = false;

            for (self.selected_features.items) |feature_idx| {
                const split_result = try self.findBestSplit(X, gradients, hessians, indices, feature_idx);
                if (split_result.gain > best_gain) {
                    best_gain = split_result.gain;
                    best_feature = feature_idx;
                    best_threshold = split_result.threshold;
                    best_left_stats = split_result.left_stats;
                    best_right_stats = split_result.right_stats;
                    found_split = true;
                }
            }

            // If no split improves gain (after regularization), make leaf
            if (!found_split or best_gain <= 0) {
                node.is_leaf = true;
                node.weight = node.stats.optimalWeight(self.config.lambda, self.config.alpha);
                return node;
            }

            // Split data
            var left_indices = ArrayList(usize).init(self.allocator);
            defer left_indices.deinit();
            var right_indices = ArrayList(usize).init(self.allocator);
            defer right_indices.deinit();

            for (indices) |idx| {
                if (X[idx][best_feature] <= best_threshold) {
                    try left_indices.append(self.allocator, idx);
                } else {
                    try right_indices.append(self.allocator, idx);
                }
            }

            // Recursively build children
            node.feature_idx = best_feature;
            node.threshold = best_threshold;
            node.left = try self.buildTree(X, gradients, hessians, left_indices.items, depth + 1);
            node.right = try self.buildTree(X, gradients, hessians, right_indices.items, depth + 1);

            return node;
        }

        /// Find best split for a feature using histogram-based method.
        const SplitResult = struct {
            gain: T,
            threshold: T,
            left_stats: GradStats,
            right_stats: GradStats,
        };

        fn findBestSplit(
            self: *const Self,
            X: []const []const T,
            gradients: []const T,
            hessians: []const T,
            indices: []const usize,
            feature_idx: usize,
        ) !SplitResult {
            // Sort indices by feature value
            var sorted_indices = try self.allocator.alloc(usize, indices.len);
            defer self.allocator.free(sorted_indices);
            @memcpy(sorted_indices, indices);

            // Simple insertion sort for small datasets, otherwise use std.sort
            if (sorted_indices.len < 50) {
                for (1..sorted_indices.len) |i| {
                    const key = sorted_indices[i];
                    const key_val = X[key][feature_idx];
                    var j: usize = i;
                    while (j > 0 and X[sorted_indices[j - 1]][feature_idx] > key_val) {
                        sorted_indices[j] = sorted_indices[j - 1];
                        j -= 1;
                    }
                    sorted_indices[j] = key;
                }
            } else {
                const Context = struct {
                    x: []const []const T,
                    feature: usize,
                };
                const ctx = Context{ .x = X, .feature = feature_idx };
                std.mem.sort(usize, sorted_indices, ctx, struct {
                    fn lessThan(context: Context, a: usize, b: usize) bool {
                        return context.x[a][context.feature] < context.x[b][context.feature];
                    }
                }.lessThan);
            }

            var best_gain: T = 0;
            var best_threshold: T = 0;
            var best_left_stats = GradStats.init();
            var best_right_stats = GradStats.init();

            // Try splits between sorted values
            var left_stats = GradStats.init();
            var right_stats = GradStats.init();

            // Initialize right stats with all samples
            for (sorted_indices) |idx| {
                right_stats.add(gradients[idx], hessians[idx]);
            }

            for (0..sorted_indices.len - 1) |i| {
                const idx = sorted_indices[i];
                const next_idx = sorted_indices[i + 1];

                // Move sample from right to left
                left_stats.add(gradients[idx], hessians[idx]);
                right_stats.sum_grad -= gradients[idx];
                right_stats.sum_hess -= hessians[idx];
                right_stats.count -= 1;

                // Skip if duplicate values
                if (X[idx][feature_idx] == X[next_idx][feature_idx]) continue;

                // Check minimum child weight constraint
                if (left_stats.sum_hess < self.config.min_child_weight or
                    right_stats.sum_hess < self.config.min_child_weight)
                {
                    continue;
                }

                // Calculate gain: Gain_split = 0.5 * [G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ
                const parent_stats = GradStats{
                    .sum_grad = left_stats.sum_grad + right_stats.sum_grad,
                    .sum_hess = left_stats.sum_hess + right_stats.sum_hess,
                    .count = left_stats.count + right_stats.count,
                };

                const left_gain = left_stats.gain(self.config.lambda, 0);
                const right_gain = right_stats.gain(self.config.lambda, 0);
                const parent_gain = parent_stats.gain(self.config.lambda, 0);
                const gain = left_gain + right_gain - parent_gain - self.config.gamma;

                if (gain > best_gain) {
                    best_gain = gain;
                    best_threshold = (X[idx][feature_idx] + X[next_idx][feature_idx]) / 2.0;
                    best_left_stats = left_stats;
                    best_right_stats = right_stats;
                }
            }

            return SplitResult{
                .gain = best_gain,
                .threshold = best_threshold,
                .left_stats = best_left_stats,
                .right_stats = best_right_stats,
            };
        }

        /// Predict single sample (returns raw score, not probability).
        ///
        /// Time: O(n_trees × depth)
        /// Space: O(1)
        pub fn predict(self: *const Self, x: []const T) T {
            var pred = self.base_prediction;
            for (self.trees.items) |tree| {
                pred += self.config.learning_rate * predictTreeInternal(tree, x);
            }
            return pred;
        }

        /// Predict batch of samples.
        ///
        /// Time: O(n × n_trees × depth)
        /// Space: O(n)
        pub fn predictBatch(self: *const Self, X: []const []const T, predictions: []T) void {
            for (X, 0..) |x, i| {
                predictions[i] = self.predict(x);
            }
        }

        /// Predict probability for classification (applies sigmoid).
        ///
        /// Time: O(n_trees × depth)
        /// Space: O(1)
        pub fn predictProba(self: *const Self, x: []const T) T {
            if (self.config.task != .classification) return self.predict(x);
            return sigmoid(self.predict(x));
        }

        /// Predict tree value (leaf weight) without learning rate.
        fn predictTreeInternal(node: *const TreeNode, x: []const T) T {
            if (node.is_leaf) return node.weight;

            if (x[node.feature_idx] <= node.threshold) {
                return predictTreeInternal(node.left.?, x);
            } else {
                return predictTreeInternal(node.right.?, x);
            }
        }

        /// Calculate R² score for regression.
        ///
        /// Time: O(n × n_trees × depth)
        /// Space: O(1)
        pub fn score(self: *const Self, X: []const []const T, y: []const T) !T {
            if (X.len != y.len) return error.InvalidInput;

            var ss_res: T = 0;
            var ss_tot: T = 0;
            var y_mean: T = 0;

            for (y) |yi| y_mean += yi;
            y_mean /= @as(T, @floatFromInt(y.len));

            for (X, 0..) |x, i| {
                const pred = self.predict(x);
                const residual = y[i] - pred;
                ss_res += residual * residual;
                const deviation = y[i] - y_mean;
                ss_tot += deviation * deviation;
            }

            if (ss_tot == 0) return 1.0;
            return 1.0 - (ss_res / ss_tot);
        }

        /// Calculate accuracy for classification.
        ///
        /// Time: O(n × n_trees × depth)
        /// Space: O(1)
        pub fn accuracy(self: *const Self, X: []const []const T, y: []const T) !T {
            if (self.config.task != .classification) return error.InvalidTask;
            if (X.len != y.len) return error.InvalidInput;

            var correct: usize = 0;
            for (X, 0..) |x, i| {
                const pred_proba = self.predictProba(x);
                const pred_class: T = if (pred_proba >= 0.5) 1.0 else 0.0;
                if (pred_class == y[i]) correct += 1;
            }

            return @as(T, @floatFromInt(correct)) / @as(T, @floatFromInt(X.len));
        }

        /// Sigmoid function: 1 / (1 + exp(-x))
        fn sigmoid(x: T) T {
            return 1.0 / (1.0 + @exp(-x));
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "XGBoost: basic regression" {
    const allocator = testing.allocator;

    // Simple linear data: y = 2x + 1
    const X = [_][]const f64{
        &[_]f64{ 1.0, 0.5 },
        &[_]f64{ 2.0, 1.0 },
        &[_]f64{ 3.0, 1.5 },
        &[_]f64{ 4.0, 2.0 },
    };
    const y = [_]f64{ 3.0, 5.0, 7.0, 9.0 };

    var model = try XGBoost(f64).init(allocator, .{
        .task = .regression,
        .n_trees = 10,
        .learning_rate = 0.3,
        .max_depth = 3,
    });
    defer model.deinit();

    try model.fit(&X, &y);

    // Check predictions are close to actual values
    for (X, 0..) |x, i| {
        const pred = model.predict(x);
        try testing.expect(@abs(pred - y[i]) < 1.0); // Within 1.0 error
    }
}

test "XGBoost: regression score" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
        &[_]f64{4.0},
        &[_]f64{5.0},
    };
    const y = [_]f64{ 2.0, 4.0, 6.0, 8.0, 10.0 };

    var model = try XGBoost(f64).init(allocator, .{
        .task = .regression,
        .n_trees = 20,
        .learning_rate = 0.3,
        .max_depth = 4,
    });
    defer model.deinit();

    try model.fit(&X, &y);

    const r2 = try model.score(&X, &y);
    try testing.expect(r2 > 0.9); // Good fit
}

test "XGBoost: binary classification" {
    const allocator = testing.allocator;

    // Linearly separable data
    const X = [_][]const f64{
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 1.5, 1.2 },
        &[_]f64{ 2.0, 1.8 },
        &[_]f64{ 5.0, 5.0 },
        &[_]f64{ 5.5, 5.2 },
        &[_]f64{ 6.0, 5.8 },
    };
    const y = [_]f64{ 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 };

    var model = try XGBoost(f64).init(allocator, .{
        .task = .classification,
        .n_trees = 10,
        .learning_rate = 0.3,
        .max_depth = 3,
    });
    defer model.deinit();

    try model.fit(&X, &y);

    // Check classification accuracy
    const acc = try model.accuracy(&X, &y);
    try testing.expect(acc >= 0.8); // At least 80% accuracy
}

test "XGBoost: probability prediction" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 5.0, 5.0 },
    };
    const y = [_]f64{ 0.0, 1.0 };

    var model = try XGBoost(f64).init(allocator, .{
        .task = .classification,
        .n_trees = 10,
    });
    defer model.deinit();

    try model.fit(&X, &y);

    const p0 = model.predictProba(X[0]);
    const p1 = model.predictProba(X[1]);

    try testing.expect(p0 >= 0 and p0 <= 1);
    try testing.expect(p1 >= 0 and p1 <= 1);
    try testing.expect(p0 < 0.5); // Class 0
    try testing.expect(p1 > 0.5); // Class 1
}

test "XGBoost: regularization prevents overfitting" {
    const allocator = testing.allocator;

    // Small dataset with noise
    const X = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
    };
    const y = [_]f64{ 1.0, 2.1, 2.9 }; // Slight noise

    // Model with strong regularization
    var model = try XGBoost(f64).init(allocator, .{
        .task = .regression,
        .n_trees = 10,
        .learning_rate = 0.1,
        .max_depth = 2,
        .lambda = 10.0, // Strong L2
        .gamma = 1.0, // High minimum gain
    });
    defer model.deinit();

    try model.fit(&X, &y);

    // Predictions should be smooth (not overfitted to noise)
    const pred1 = model.predict(X[0]);
    const pred2 = model.predict(X[1]);
    const pred3 = model.predict(X[2]);

    try testing.expect(@abs(pred1 - y[0]) < 2.0);
    try testing.expect(@abs(pred2 - y[1]) < 2.0);
    try testing.expect(@abs(pred3 - y[2]) < 2.0);
}

test "XGBoost: subsampling configuration" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 2.0, 2.0 },
        &[_]f64{ 3.0, 3.0 },
        &[_]f64{ 4.0, 4.0 },
    };
    const y = [_]f64{ 2.0, 4.0, 6.0, 8.0 };

    var model = try XGBoost(f64).init(allocator, .{
        .task = .regression,
        .n_trees = 10,
        .subsample = 0.8, // 80% row sampling
        .colsample_bytree = 0.8, // 80% column sampling
    });
    defer model.deinit();

    try model.fit(&X, &y);

    // Model should still learn reasonably well
    const r2 = try model.score(&X, &y);
    try testing.expect(r2 > 0.5);
}

test "XGBoost: empty data" {
    const allocator = testing.allocator;

    var model = try XGBoost(f64).init(allocator, .{});
    defer model.deinit();

    const X = [_][]const f64{};
    const y = [_]f64{};

    try testing.expectError(error.InvalidInput, model.fit(&X, &y));
}

test "XGBoost: mismatched X and y" {
    const allocator = testing.allocator;

    var model = try XGBoost(f64).init(allocator, .{});
    defer model.deinit();

    const X = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
    };
    const y = [_]f64{1.0};

    try testing.expectError(error.InvalidInput, model.fit(&X, &y));
}

test "XGBoost: single sample" {
    const allocator = testing.allocator;

    const X = [_][]const f64{&[_]f64{1.0}};
    const y = [_]f64{2.0};

    var model = try XGBoost(f64).init(allocator, .{
        .task = .regression,
        .n_trees = 5,
    });
    defer model.deinit();

    try model.fit(&X, &y);

    const pred = model.predict(X[0]);
    try testing.expect(@abs(pred - y[0]) < 1.0);
}

test "XGBoost: deeper trees" {
    const allocator = testing.allocator;

    // Complex non-linear data
    const X = [_][]const f64{
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 2.0, 1.0 },
        &[_]f64{ 2.0, 2.0 },
        &[_]f64{ 3.0, 3.0 },
        &[_]f64{ 3.0, 4.0 },
        &[_]f64{ 4.0, 3.0 },
        &[_]f64{ 4.0, 4.0 },
    };
    const y = [_]f64{ 1.0, 2.0, 2.0, 3.0, 5.0, 6.0, 6.0, 7.0 };

    var model = try XGBoost(f64).init(allocator, .{
        .task = .regression,
        .n_trees = 20,
        .learning_rate = 0.3,
        .max_depth = 6, // Deeper trees
    });
    defer model.deinit();

    try model.fit(&X, &y);

    const r2 = try model.score(&X, &y);
    try testing.expect(r2 > 0.8);
}

test "XGBoost: L1 regularization" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
    };
    const y = [_]f64{ 1.0, 2.0, 3.0 };

    var model = try XGBoost(f64).init(allocator, .{
        .task = .regression,
        .n_trees = 10,
        .alpha = 5.0, // Strong L1 regularization
    });
    defer model.deinit();

    try model.fit(&X, &y);

    // Model should still work with L1 regularization
    const pred = model.predict(X[1]);
    try testing.expect(@abs(pred - y[1]) < 2.0);
}

test "XGBoost: batch prediction" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
    };
    const y = [_]f64{ 2.0, 4.0, 6.0 };

    var model = try XGBoost(f64).init(allocator, .{
        .task = .regression,
        .n_trees = 10,
    });
    defer model.deinit();

    try model.fit(&X, &y);

    var predictions: [3]f64 = undefined;
    model.predictBatch(&X, &predictions);

    for (predictions, 0..) |pred, i| {
        try testing.expect(@abs(pred - y[i]) < 1.0);
    }
}

test "XGBoost: f32 support" {
    const allocator = testing.allocator;

    const X = [_][]const f32{
        &[_]f32{1.0},
        &[_]f32{2.0},
        &[_]f32{3.0},
    };
    const y = [_]f32{ 2.0, 4.0, 6.0 };

    var model = try XGBoost(f32).init(allocator, .{
        .task = .regression,
        .n_trees = 10,
    });
    defer model.deinit();

    try model.fit(&X, &y);

    const pred = model.predict(X[1]);
    try testing.expect(@abs(pred - y[1]) < 1.0);
}

test "XGBoost: multi-feature classification" {
    const allocator = testing.allocator;

    // XOR-like problem (non-linearly separable)
    const X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 0.0, 1.0 },
        &[_]f64{ 1.0, 0.0 },
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 0.1, 0.1 },
        &[_]f64{ 0.1, 0.9 },
        &[_]f64{ 0.9, 0.1 },
        &[_]f64{ 0.9, 0.9 },
    };
    const y = [_]f64{ 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0 };

    var model = try XGBoost(f64).init(allocator, .{
        .task = .classification,
        .n_trees = 30,
        .learning_rate = 0.3,
        .max_depth = 5,
    });
    defer model.deinit();

    try model.fit(&X, &y);

    // XOR is hard but should achieve reasonable accuracy with enough trees
    const acc = try model.accuracy(&X, &y);
    try testing.expect(acc > 0.5); // Better than random
}

test "XGBoost: min_child_weight constraint" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
    };
    const y = [_]f64{ 1.0, 2.0, 3.0 };

    var model = try XGBoost(f64).init(allocator, .{
        .task = .regression,
        .n_trees = 10,
        .min_child_weight = 2.0, // Requires at least 2 samples in leaf
    });
    defer model.deinit();

    try model.fit(&X, &y);

    // Model should still work with constraint
    const pred = model.predict(X[1]);
    try testing.expect(@abs(pred - y[1]) < 2.0);
}
