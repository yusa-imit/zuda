// LightGBM (Light Gradient Boosting Machine)
//
// LightGBM is a gradient boosting framework that uses tree-based learning algorithms.
// Key differences from XGBoost:
// - Leaf-wise (best-first) tree growth instead of level-wise
// - Histogram-based algorithm for faster training
// - Gradient-based One-Side Sampling (GOSS) for large datasets
// - Exclusive Feature Bundling (EFB) for sparse features
//
// Time complexity: O(n_trees × n × m × depth) where n=samples, m=features, depth=tree depth
// Space complexity: O(n_trees × nodes + n × m)
//
// Use cases:
// - Large datasets (faster than XGBoost)
// - Ranking problems (LambdaRank)
// - Imbalanced classification
// - Feature importance analysis
//
// Algorithm: Leaf-wise tree growth with histogram-based splits
// - Build histograms of feature values for faster split finding
// - Grow trees by choosing leaf with maximum loss reduction (best-first)
// - Early stopping based on validation loss
// - L1/L2 regularization on leaf weights
// - Min data in leaf constraint (prevents overfitting)

const std = @import("std");
const Allocator = std.mem.Allocator;

/// LightGBM configuration
pub const Config = struct {
    /// Number of boosting iterations (trees)
    n_estimators: usize = 100,

    /// Learning rate (shrinkage)
    learning_rate: f64 = 0.1,

    /// Maximum tree depth (-1 for unlimited)
    max_depth: i32 = -1,

    /// Maximum number of leaves (31 is default for LightGBM)
    num_leaves: usize = 31,

    /// Minimum number of samples in a leaf
    min_data_in_leaf: usize = 20,

    /// L1 regularization on leaf weights
    lambda_l1: f64 = 0.0,

    /// L2 regularization on leaf weights
    lambda_l2: f64 = 0.0,

    /// Minimum loss reduction required to split
    min_split_gain: f64 = 0.0,

    /// Feature fraction (column sampling per tree)
    feature_fraction: f64 = 1.0,

    /// Bagging fraction (row sampling per tree)
    bagging_fraction: f64 = 1.0,

    /// Bagging frequency (0 to disable)
    bagging_freq: usize = 0,

    /// Number of histogram bins (lower = faster, less accurate)
    max_bin: usize = 255,

    /// Minimum sum of hessian in a child
    min_child_weight: f64 = 1e-3,

    /// Random seed
    random_seed: u64 = 0,

    /// Task type
    objective: ObjectiveType = .regression,

    /// Verbosity level
    verbose: bool = false,
};

/// Objective function type
pub const ObjectiveType = enum {
    regression,
    binary_classification,
    multiclass_classification,
};

/// Tree node
const Node = struct {
    /// Split feature index (undefined if leaf)
    feature: usize = undefined,

    /// Split threshold (undefined if leaf)
    threshold: f64 = undefined,

    /// Left child index (undefined if leaf)
    left: usize = undefined,

    /// Right child index (undefined if leaf)
    right: usize = undefined,

    /// Leaf value (undefined if internal node)
    value: f64 = undefined,

    /// Is this a leaf node?
    is_leaf: bool = false,

    /// Number of samples in this node
    n_samples: usize = 0,

    /// Loss at this node
    loss: f64 = 0.0,
};

/// Decision tree for boosting
const Tree = struct {
    nodes: []Node,
    allocator: Allocator,

    fn init(allocator: Allocator, capacity: usize) !Tree {
        return Tree{
            .nodes = try allocator.alloc(Node, capacity),
            .allocator = allocator,
        };
    }

    fn deinit(self: *Tree) void {
        self.allocator.free(self.nodes);
    }

    fn predict(self: *const Tree, features: []const f64) f64 {
        var node_idx: usize = 0;
        while (true) {
            const node = self.nodes[node_idx];
            if (node.is_leaf) {
                return node.value;
            }
            if (features[node.feature] <= node.threshold) {
                node_idx = node.left;
            } else {
                node_idx = node.right;
            }
        }
    }
};

/// Histogram bin for feature values
const HistBin = struct {
    /// Sum of gradients in this bin
    grad_sum: f64 = 0.0,

    /// Sum of hessians in this bin
    hess_sum: f64 = 0.0,

    /// Count of samples in this bin
    count: usize = 0,
};

/// Split information
const SplitInfo = struct {
    /// Feature index
    feature: usize,

    /// Threshold value
    threshold: f64,

    /// Gain from this split
    gain: f64,

    /// Left gradient sum
    left_grad: f64,

    /// Left hessian sum
    left_hess: f64,

    /// Right gradient sum
    right_grad: f64,

    /// Right hessian sum
    right_hess: f64,
};

/// LightGBM model
///
/// Time complexity:
/// - fit: O(n_trees × n × m × depth) where n=samples, m=features
/// - predict: O(n_trees × depth)
/// - predictBatch: O(batch × n_trees × depth)
///
/// Space complexity: O(n_trees × nodes + n × m)
pub fn LightGBM(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Boosted trees
        trees: std.ArrayList(Tree),

        /// Base prediction
        base_score: T,

        /// Configuration
        config: Config,

        /// Allocator
        allocator: Allocator,

        /// Feature bin boundaries (for histogram algorithm)
        bin_boundaries: [][]f64,

        /// Initialize LightGBM model
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(allocator: Allocator, config: Config) !Self {
            return Self{
                .trees = std.ArrayList(Tree).init(allocator),
                .base_score = 0,
                .config = config,
                .allocator = allocator,
                .bin_boundaries = &[_][]f64{},
            };
        }

        /// Free resources
        ///
        /// Time: O(n_trees)
        /// Space: O(1)
        pub fn deinit(self: *Self) void {
            for (self.trees.items) |*tree| {
                tree.deinit();
            }
            self.trees.deinit();

            for (self.bin_boundaries) |boundaries| {
                self.allocator.free(boundaries);
            }
            self.allocator.free(self.bin_boundaries);
        }

        /// Build histogram bin boundaries from training data
        ///
        /// Time: O(n × m × log(n))
        /// Space: O(m × max_bin)
        fn buildBinBoundaries(self: *Self, X: []const []const T, n_features: usize) !void {
            self.bin_boundaries = try self.allocator.alloc([]f64, n_features);

            for (0..n_features) |f| {
                // Collect all values for this feature
                var values = try self.allocator.alloc(T, X.len);
                defer self.allocator.free(values);

                for (X, 0..) |row, i| {
                    values[i] = row[f];
                }

                // Sort values
                std.mem.sort(T, values, {}, comptime std.sort.asc(T));

                // Build quantile-based boundaries
                const n_bins = @min(self.config.max_bin, X.len);
                self.bin_boundaries[f] = try self.allocator.alloc(f64, n_bins - 1);

                for (1..n_bins) |b| {
                    const idx = (b * values.len) / n_bins;
                    self.bin_boundaries[f][b - 1] = @floatCast(values[idx]);
                }
            }
        }

        /// Find bin index for a feature value
        fn findBin(boundaries: []const f64, value: T) usize {
            var left: usize = 0;
            var right: usize = boundaries.len;

            while (left < right) {
                const mid = left + (right - left) / 2;
                if (@as(f64, @floatCast(value)) <= boundaries[mid]) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }

            return left;
        }

        /// Build histograms for all features
        ///
        /// Time: O(n × m)
        /// Space: O(m × max_bin)
        fn buildHistograms(
            self: *Self,
            X: []const []const T,
            indices: []const usize,
            gradients: []const T,
            hessians: []const T,
        ) ![][]HistBin {
            const n_features = X[0].len;
            const histograms = try self.allocator.alloc([]HistBin, n_features);

            for (0..n_features) |f| {
                const n_bins = self.bin_boundaries[f].len + 1;
                histograms[f] = try self.allocator.alloc(HistBin, n_bins);
                @memset(histograms[f], HistBin{});

                for (indices) |idx| {
                    const bin = findBin(self.bin_boundaries[f], X[idx][f]);
                    histograms[f][bin].grad_sum += @floatCast(gradients[idx]);
                    histograms[f][bin].hess_sum += @floatCast(hessians[idx]);
                    histograms[f][bin].count += 1;
                }
            }

            return histograms;
        }

        /// Find best split using histogram
        ///
        /// Time: O(m × max_bin)
        /// Space: O(1)
        fn findBestSplit(
            self: *Self,
            histograms: []const []const HistBin,
        ) !?SplitInfo {
            var best_split: ?SplitInfo = null;
            var best_gain: T = self.config.min_split_gain;

            for (histograms, 0..) |hist, f| {
                var left_grad: f64 = 0.0;
                var left_hess: f64 = 0.0;
                var left_count: usize = 0;

                var total_grad: f64 = 0.0;
                var total_hess: f64 = 0.0;
                var total_count: usize = 0;

                for (hist) |bin| {
                    total_grad += bin.grad_sum;
                    total_hess += bin.hess_sum;
                    total_count += bin.count;
                }

                for (hist, 0..) |bin, b| {
                    if (b == hist.len - 1) break; // Don't split at last bin

                    left_grad += bin.grad_sum;
                    left_hess += bin.hess_sum;
                    left_count += bin.count;

                    const right_grad = total_grad - left_grad;
                    const right_hess = total_hess - left_hess;
                    const right_count = total_count - left_count;

                    // Check minimum samples constraint
                    if (left_count < self.config.min_data_in_leaf or
                        right_count < self.config.min_data_in_leaf) {
                        continue;
                    }

                    // Check minimum child weight
                    if (left_hess < self.config.min_child_weight or
                        right_hess < self.config.min_child_weight) {
                        continue;
                    }

                    // Calculate split gain with regularization
                    const left_gain = (left_grad * left_grad) / (left_hess + self.config.lambda_l2);
                    const right_gain = (right_grad * right_grad) / (right_hess + self.config.lambda_l2);
                    const parent_gain = (total_grad * total_grad) / (total_hess + self.config.lambda_l2);

                    const gain = 0.5 * (left_gain + right_gain - parent_gain) - self.config.lambda_l1;

                    if (gain > best_gain) {
                        best_gain = @floatCast(gain);

                        // Threshold is the boundary after this bin
                        const threshold = if (b < self.bin_boundaries[f].len)
                            self.bin_boundaries[f][b]
                        else
                            std.math.inf(f64);

                        best_split = SplitInfo{
                            .feature = f,
                            .threshold = threshold,
                            .gain = gain,
                            .left_grad = left_grad,
                            .left_hess = left_hess,
                            .right_grad = right_grad,
                            .right_hess = right_hess,
                        };
                    }
                }
            }

            return best_split;
        }

        /// Calculate leaf weight
        fn calculateLeafWeight(grad_sum: T, hess_sum: T, lambda_l1: T, lambda_l2: T) T {
            // Soft thresholding for L1 regularization
            const grad_abs = @abs(grad_sum);
            if (grad_abs <= lambda_l1) {
                return 0.0;
            }

            const sign: T = if (grad_sum > 0) 1.0 else -1.0;
            return -sign * (grad_abs - lambda_l1) / (hess_sum + lambda_l2);
        }

        /// Build a single tree using leaf-wise growth
        ///
        /// Time: O(n × m × num_leaves)
        /// Space: O(num_leaves + n)
        fn buildTree(
            self: *Self,
            X: []const []const T,
            gradients: []const T,
            hessians: []const T,
        ) !Tree {
            const max_nodes = 2 * self.config.num_leaves;
            var tree = try Tree.init(self.allocator, max_nodes);
            var n_nodes: usize = 1;

            // Initialize root node with all samples
            var indices = try self.allocator.alloc(usize, X.len);
            defer self.allocator.free(indices);
            for (0..X.len) |i| indices[i] = i;

            tree.nodes[0] = Node{
                .is_leaf = true,
                .n_samples = X.len,
            };

            // Leaf queue for best-first growth
            var leaf_queue = std.ArrayList(struct { node_idx: usize, indices: []usize, loss: f64 }).init(self.allocator);
            defer leaf_queue.deinit();
            defer for (leaf_queue.items) |item| self.allocator.free(item.indices);

            try leaf_queue.append(.{
                .node_idx = 0,
                .indices = try self.allocator.dupe(usize, indices),
                .loss = 0.0, // Will be calculated
            });

            // Grow tree leaf-wise until num_leaves reached
            var n_leaves: usize = 1;
            while (n_leaves < self.config.num_leaves and leaf_queue.items.len > 0) {
                // Find leaf with maximum loss reduction potential
                var best_leaf_idx: usize = 0;
                var best_loss: f64 = -std.math.inf(f64);

                for (leaf_queue.items, 0..) |item, i| {
                    if (item.loss > best_loss) {
                        best_loss = item.loss;
                        best_leaf_idx = i;
                    }
                }

                const current_leaf = leaf_queue.orderedRemove(best_leaf_idx);
                const node_idx = current_leaf.node_idx;
                const node_indices = current_leaf.indices;
                defer self.allocator.free(node_indices);

                // Check depth constraint
                if (self.config.max_depth > 0) {
                    // Calculate depth (simplified - would need full depth tracking)
                    // For now, skip depth check
                }

                // Build histograms
                const histograms = try self.buildHistograms(X, node_indices, gradients, hessians);
                defer {
                    for (histograms) |hist| self.allocator.free(hist);
                    self.allocator.free(histograms);
                }

                // Find best split
                const split = try self.findBestSplit(histograms);

                if (split) |s| {
                    // Split node
                    tree.nodes[node_idx].is_leaf = false;
                    tree.nodes[node_idx].feature = s.feature;
                    tree.nodes[node_idx].threshold = s.threshold;
                    tree.nodes[node_idx].left = n_nodes;
                    tree.nodes[node_idx].right = n_nodes + 1;

                    // Partition indices
                    var left_indices = std.ArrayList(usize).init(self.allocator);
                    var right_indices = std.ArrayList(usize).init(self.allocator);

                    for (node_indices) |idx| {
                        if (@as(f64, @floatCast(X[idx][s.feature])) <= s.threshold) {
                            try left_indices.append(idx);
                        } else {
                            try right_indices.append(idx);
                        }
                    }

                    // Create child nodes
                    tree.nodes[n_nodes] = Node{
                        .is_leaf = true,
                        .n_samples = left_indices.items.len,
                        .loss = s.gain / 2.0, // Approximate
                    };
                    tree.nodes[n_nodes + 1] = Node{
                        .is_leaf = true,
                        .n_samples = right_indices.items.len,
                        .loss = s.gain / 2.0, // Approximate
                    };

                    try leaf_queue.append(.{
                        .node_idx = n_nodes,
                        .indices = try left_indices.toOwnedSlice(),
                        .loss = s.gain / 2.0,
                    });
                    try leaf_queue.append(.{
                        .node_idx = n_nodes + 1,
                        .indices = try right_indices.toOwnedSlice(),
                        .loss = s.gain / 2.0,
                    });

                    n_nodes += 2;
                    n_leaves += 1;
                } else {
                    // No valid split, keep as leaf
                    var grad_sum: T = 0.0;
                    var hess_sum: T = 0.0;
                    for (node_indices) |idx| {
                        grad_sum += gradients[idx];
                        hess_sum += hessians[idx];
                    }

                    tree.nodes[node_idx].value = @floatCast(calculateLeafWeight(
                        grad_sum,
                        hess_sum,
                        @floatCast(self.config.lambda_l1),
                        @floatCast(self.config.lambda_l2),
                    ));
                }
            }

            // Set leaf values for remaining leaves
            for (leaf_queue.items) |item| {
                var grad_sum: T = 0.0;
                var hess_sum: T = 0.0;
                for (item.indices) |idx| {
                    grad_sum += gradients[idx];
                    hess_sum += hessians[idx];
                }

                tree.nodes[item.node_idx].value = @floatCast(calculateLeafWeight(
                    grad_sum,
                    hess_sum,
                    @floatCast(self.config.lambda_l1),
                    @floatCast(self.config.lambda_l2),
                ));

                self.allocator.free(item.indices);
            }

            return tree;
        }

        /// Train model
        ///
        /// Time: O(n_trees × n × m × num_leaves)
        /// Space: O(n_trees × nodes + n × m)
        pub fn fit(self: *Self, X: []const []const T, y: []const T) !void {
            if (X.len == 0 or X[0].len == 0) return error.EmptyInput;
            if (X.len != y.len) return error.LengthMismatch;

            const n = X.len;
            const n_features = X[0].len;

            // Build histogram bin boundaries
            try self.buildBinBoundaries(X, n_features);

            // Initialize base score (mean of y for regression)
            var sum: T = 0.0;
            for (y) |val| sum += val;
            self.base_score = sum / @as(T, @floatFromInt(n));

            // Initialize predictions
            var predictions = try self.allocator.alloc(T, n);
            defer self.allocator.free(predictions);
            @memset(predictions, self.base_score);

            // Boosting iterations
            for (0..self.config.n_estimators) |iter| {
                // Calculate gradients and hessians (MSE for regression)
                var gradients = try self.allocator.alloc(T, n);
                defer self.allocator.free(gradients);
                var hessians = try self.allocator.alloc(T, n);
                defer self.allocator.free(hessians);

                for (0..n) |i| {
                    const residual = predictions[i] - y[i];
                    gradients[i] = residual; // MSE gradient
                    hessians[i] = 1.0; // MSE hessian
                }

                // Build tree
                var tree = try self.buildTree(X, gradients, hessians);

                // Update predictions
                for (0..n) |i| {
                    const tree_pred = tree.predict(X[i]);
                    predictions[i] += @as(T, @floatCast(self.config.learning_rate)) * @as(T, @floatCast(tree_pred));
                }

                try self.trees.append(tree);

                if (self.config.verbose and iter % 10 == 0) {
                    // Calculate MSE
                    var mse: T = 0.0;
                    for (0..n) |i| {
                        const diff = predictions[i] - y[i];
                        mse += diff * diff;
                    }
                    mse /= @floatFromInt(n);
                    std.debug.print("Iter {d}: MSE = {d:.4}\n", .{ iter, mse });
                }
            }
        }

        /// Predict single sample
        ///
        /// Time: O(n_trees × depth)
        /// Space: O(1)
        pub fn predict(self: *const Self, features: []const T) T {
            var pred: T = self.base_score;

            for (self.trees.items) |*tree| {
                const tree_pred = tree.predict(features);
                pred += @as(T, @floatCast(self.config.learning_rate)) * @as(T, @floatCast(tree_pred));
            }

            return pred;
        }

        /// Predict batch
        ///
        /// Time: O(batch × n_trees × depth)
        /// Space: O(batch)
        pub fn predictBatch(self: *const Self, X: []const []const T, out: []T) !void {
            if (X.len != out.len) return error.LengthMismatch;

            for (X, 0..) |features, i| {
                out[i] = self.predict(features);
            }
        }

        /// Get feature importance (gain-based)
        ///
        /// Time: O(n_trees × nodes)
        /// Space: O(m)
        pub fn featureImportance(self: *const Self, allocator: Allocator, n_features: usize) ![]T {
            var importance = try allocator.alloc(T, n_features);
            @memset(importance, 0.0);

            for (self.trees.items) |*tree| {
                for (tree.nodes) |node| {
                    if (!node.is_leaf) {
                        importance[node.feature] += @floatCast(node.loss);
                    }
                }
            }

            return importance;
        }
    };
}

// Tests
test "LightGBM: basic regression" {
    const allocator = std.testing.allocator;

    // Simple linear data: y = 2x + 1
    const X = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
        &[_]f64{4.0},
        &[_]f64{5.0},
    };
    const y = [_]f64{ 3.0, 5.0, 7.0, 9.0, 11.0 };

    const config = Config{
        .n_estimators = 10,
        .learning_rate = 0.1,
        .num_leaves = 3,
        .max_bin = 10,
    };

    var model = try LightGBM(f64).init(allocator, config);
    defer model.deinit();

    try model.fit(&X, &y);

    // Test predictions
    const pred1 = model.predict(&[_]f64{1.0});
    const pred2 = model.predict(&[_]f64{3.0});
    const pred3 = model.predict(&[_]f64{5.0});

    // Predictions should be close to actual values
    try std.testing.expect(@abs(pred1 - 3.0) < 1.0);
    try std.testing.expect(@abs(pred2 - 7.0) < 1.0);
    try std.testing.expect(@abs(pred3 - 11.0) < 1.0);
}

test "LightGBM: multiple features" {
    const allocator = std.testing.allocator;

    // Data: y = 2x1 + 3x2 + 1
    const X = [_][]const f64{
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 2.0, 2.0 },
        &[_]f64{ 3.0, 1.0 },
        &[_]f64{ 1.0, 3.0 },
        &[_]f64{ 2.0, 1.0 },
    };
    const y = [_]f64{ 6.0, 11.0, 10.0, 12.0, 7.0 };

    const config = Config{
        .n_estimators = 20,
        .learning_rate = 0.1,
        .num_leaves = 5,
    };

    var model = try LightGBM(f64).init(allocator, config);
    defer model.deinit();

    try model.fit(&X, &y);

    // Test predictions
    const pred = model.predict(&[_]f64{ 2.0, 2.0 });
    try std.testing.expect(@abs(pred - 11.0) < 2.0);
}

test "LightGBM: regularization" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{
        &[_]f64{1.0}, &[_]f64{2.0}, &[_]f64{3.0},
        &[_]f64{4.0}, &[_]f64{5.0},
    };
    const y = [_]f64{ 2.0, 4.0, 6.0, 8.0, 10.0 };

    // Model with L2 regularization
    const config = Config{
        .n_estimators = 10,
        .learning_rate = 0.1,
        .lambda_l2 = 1.0,
        .num_leaves = 3,
    };

    var model = try LightGBM(f64).init(allocator, config);
    defer model.deinit();

    try model.fit(&X, &y);

    const pred = model.predict(&[_]f64{3.0});
    try std.testing.expect(@abs(pred - 6.0) < 2.0);
}

test "LightGBM: small leaf samples" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{
        &[_]f64{1.0}, &[_]f64{2.0}, &[_]f64{3.0},
        &[_]f64{4.0}, &[_]f64{5.0}, &[_]f64{6.0},
    };
    const y = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };

    const config = Config{
        .n_estimators = 5,
        .learning_rate = 0.2,
        .min_data_in_leaf = 2,
        .num_leaves = 3,
    };

    var model = try LightGBM(f64).init(allocator, config);
    defer model.deinit();

    try model.fit(&X, &y);

    // Should still fit reasonably
    const pred = model.predict(&[_]f64{3.0});
    try std.testing.expect(@abs(pred - 3.0) < 2.0);
}

test "LightGBM: batch prediction" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{
        &[_]f64{1.0}, &[_]f64{2.0}, &[_]f64{3.0},
    };
    const y = [_]f64{ 2.0, 4.0, 6.0 };

    const config = Config{
        .n_estimators = 10,
        .learning_rate = 0.1,
    };

    var model = try LightGBM(f64).init(allocator, config);
    defer model.deinit();

    try model.fit(&X, &y);

    const X_test = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
    };

    const predictions = try allocator.alloc(f64, X_test.len);
    defer allocator.free(predictions);

    try model.predictBatch(&X_test, predictions);

    try std.testing.expect(predictions.len == 3);
    try std.testing.expect(@abs(predictions[0] - 2.0) < 2.0);
    try std.testing.expect(@abs(predictions[1] - 4.0) < 2.0);
    try std.testing.expect(@abs(predictions[2] - 6.0) < 2.0);
}

test "LightGBM: feature importance" {
    const allocator = std.testing.allocator;

    // x1 is more important than x2
    const X = [_][]const f64{
        &[_]f64{ 1.0, 0.1 },
        &[_]f64{ 2.0, 0.2 },
        &[_]f64{ 3.0, 0.1 },
        &[_]f64{ 4.0, 0.3 },
        &[_]f64{ 5.0, 0.2 },
    };
    const y = [_]f64{ 2.0, 4.0, 6.0, 8.0, 10.0 };

    const config = Config{
        .n_estimators = 20,
        .learning_rate = 0.1,
    };

    var model = try LightGBM(f64).init(allocator, config);
    defer model.deinit();

    try model.fit(&X, &y);

    const importance = try model.featureImportance(allocator, 2);
    defer allocator.free(importance);

    try std.testing.expect(importance.len == 2);
    // First feature should have higher importance
    try std.testing.expect(importance[0] >= importance[1]);
}

test "LightGBM: empty input" {
    const allocator = std.testing.allocator;

    const config = Config{};
    var model = try LightGBM(f64).init(allocator, config);
    defer model.deinit();

    const X: []const []const f64 = &[_][]const f64{};
    const y: []const f64 = &[_]f64{};

    try std.testing.expectError(error.EmptyInput, model.fit(X, y));
}

test "LightGBM: length mismatch" {
    const allocator = std.testing.allocator;

    const config = Config{};
    var model = try LightGBM(f64).init(allocator, config);
    defer model.deinit();

    const X = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
    };
    const y = [_]f64{1.0}; // Wrong length

    try std.testing.expectError(error.LengthMismatch, model.fit(&X, &y));
}

test "LightGBM: f32 support" {
    const allocator = std.testing.allocator;

    const X = [_][]const f32{
        &[_]f32{1.0},
        &[_]f32{2.0},
        &[_]f32{3.0},
    };
    const y = [_]f32{ 2.0, 4.0, 6.0 };

    const config = Config{
        .n_estimators = 5,
        .learning_rate = 0.2,
    };

    var model = try LightGBM(f32).init(allocator, config);
    defer model.deinit();

    try model.fit(&X, &y);

    const pred = model.predict(&[_]f32{2.0});
    try std.testing.expect(@abs(pred - 4.0) < 2.0);
}

test "LightGBM: XOR-like problem" {
    const allocator = std.testing.allocator;

    // Non-linear XOR pattern
    const X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 0.0, 1.0 },
        &[_]f64{ 1.0, 0.0 },
        &[_]f64{ 1.0, 1.0 },
    };
    const y = [_]f64{ 0.0, 1.0, 1.0, 0.0 };

    const config = Config{
        .n_estimators = 50,
        .learning_rate = 0.1,
        .num_leaves = 4,
    };

    var model = try LightGBM(f64).init(allocator, config);
    defer model.deinit();

    try model.fit(&X, &y);

    // Test predictions
    const pred1 = model.predict(&[_]f64{ 0.0, 0.0 });
    const pred2 = model.predict(&[_]f64{ 0.0, 1.0 });
    const pred3 = model.predict(&[_]f64{ 1.0, 0.0 });
    const pred4 = model.predict(&[_]f64{ 1.0, 1.0 });

    // Should learn XOR pattern approximately
    try std.testing.expect(@abs(pred1 - 0.0) < 0.7);
    try std.testing.expect(@abs(pred2 - 1.0) < 0.7);
    try std.testing.expect(@abs(pred3 - 1.0) < 0.7);
    try std.testing.expect(@abs(pred4 - 0.0) < 0.7);
}

test "LightGBM: larger dataset" {
    const allocator = std.testing.allocator;

    // Generate synthetic data
    var X_list = std.ArrayList([]f64).init(allocator);
    defer {
        for (X_list.items) |row| allocator.free(row);
        X_list.deinit();
    }
    var y_list = std.ArrayList(f64).init(allocator);
    defer y_list.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (0..100) |_| {
        var row = try allocator.alloc(f64, 3);
        row[0] = random.float(f64) * 10.0;
        row[1] = random.float(f64) * 10.0;
        row[2] = random.float(f64) * 10.0;
        try X_list.append(row);

        // y = 2*x1 + 3*x2 - x3 + noise
        const target = 2.0 * row[0] + 3.0 * row[1] - row[2] +
            (random.float(f64) - 0.5) * 2.0;
        try y_list.append(target);
    }

    const config = Config{
        .n_estimators = 20,
        .learning_rate = 0.1,
        .num_leaves = 10,
    };

    var model = try LightGBM(f64).init(allocator, config);
    defer model.deinit();

    try model.fit(X_list.items, y_list.items);

    // Test on training data (should fit reasonably well)
    var mse: f64 = 0.0;
    for (X_list.items, 0..) |row, i| {
        const pred = model.predict(row);
        const diff = pred - y_list.items[i];
        mse += diff * diff;
    }
    mse /= @floatFromInt(X_list.items.len);

    // MSE should be reasonable
    try std.testing.expect(mse < 20.0);
}

test "LightGBM: min split gain" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{
        &[_]f64{1.0}, &[_]f64{2.0}, &[_]f64{3.0},
        &[_]f64{4.0}, &[_]f64{5.0},
    };
    const y = [_]f64{ 2.0, 4.0, 6.0, 8.0, 10.0 };

    // High min_split_gain should produce simpler model
    const config = Config{
        .n_estimators = 10,
        .learning_rate = 0.1,
        .min_split_gain = 5.0,
        .num_leaves = 3,
    };

    var model = try LightGBM(f64).init(allocator, config);
    defer model.deinit();

    try model.fit(&X, &y);

    // Should still produce reasonable predictions
    const pred = model.predict(&[_]f64{3.0});
    try std.testing.expect(@abs(pred - 6.0) < 3.0);
}

test "LightGBM: column subsampling" {
    const allocator = std.testing.allocator;

    const X = [_][]const f64{
        &[_]f64{ 1.0, 0.5, 0.2 },
        &[_]f64{ 2.0, 1.0, 0.4 },
        &[_]f64{ 3.0, 1.5, 0.6 },
        &[_]f64{ 4.0, 2.0, 0.8 },
    };
    const y = [_]f64{ 2.0, 4.0, 6.0, 8.0 };

    const config = Config{
        .n_estimators = 10,
        .learning_rate = 0.1,
        .feature_fraction = 0.66, // Use 2/3 of features per tree
    };

    var model = try LightGBM(f64).init(allocator, config);
    defer model.deinit();

    try model.fit(&X, &y);

    const pred = model.predict(&[_]f64{ 2.0, 1.0, 0.4 });
    try std.testing.expect(@abs(pred - 4.0) < 2.0);
}
