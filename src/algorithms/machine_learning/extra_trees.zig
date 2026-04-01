const std = @import("std");
const Allocator = std.mem.Allocator;

/// Extra Trees (Extremely Randomized Trees) ensemble method.
///
/// Differences from Random Forest:
/// - No bootstrap sampling (uses entire training set)
/// - Random splits instead of best splits (faster training)
/// - Often achieves better variance reduction than Random Forest
///
/// Algorithm:
/// 1. Build n_trees decision trees using entire dataset
/// 2. At each node, randomly select k features (sqrt(m) default)
/// 3. For each feature, randomly generate a split threshold (not optimized)
/// 4. Select best split among the k random feature-threshold pairs
/// 5. Aggregate predictions: majority vote (classification) or mean (regression)
///
/// Time complexity: O(n_trees × n × k × log n) training, O(n_trees × depth) prediction
/// Space complexity: O(n_trees × nodes) where nodes ≈ 2^depth
///
/// Use cases:
/// - High-dimensional data (faster than Random Forest due to random splits)
/// - Noisy data (increased randomness can improve generalization)
/// - When variance reduction is critical
/// - Real-time training scenarios (faster than RF)
pub fn ExtraTrees(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Task type
        pub const TaskType = enum {
            classification,
            regression,
        };

        /// Configuration
        pub const Config = struct {
            task: TaskType = .classification,
            n_trees: usize = 100,
            max_depth: usize = 10,
            min_samples_split: usize = 2,
            max_features: ?usize = null, // null = sqrt(n_features)
            random_seed: u64 = 42,
        };

        /// Decision tree node
        const Node = struct {
            feature: ?usize = null, // null for leaf
            threshold: ?T = null,
            left: ?*Node = null,
            right: ?*Node = null,
            value: T = 0, // leaf value or class

            fn deinit(self: *Node, allocator: Allocator) void {
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
        trees: []Node,
        prng: std.Random.DefaultPrng,
        n_classes: usize,

        /// Initialize Extra Trees ensemble
        pub fn init(allocator: Allocator, config: Config) !Self {
            const trees = try allocator.alloc(Node, config.n_trees);
            @memset(trees, Node{});

            return Self{
                .allocator = allocator,
                .config = config,
                .trees = trees,
                .prng = std.Random.DefaultPrng.init(config.random_seed),
                .n_classes = 0,
            };
        }

        /// Free resources
        pub fn deinit(self: *Self) void {
            for (self.trees) |*tree| {
                tree.deinit(self.allocator);
            }
            self.allocator.free(self.trees);
        }

        /// Train the ensemble
        ///
        /// Time: O(n_trees × n × k × log n) where k = max_features
        /// Space: O(n_trees × nodes)
        pub fn fit(self: *Self, X: []const []const T, y: []const T) !void {
            if (X.len == 0) return error.EmptyInput;
            if (X.len != y.len) return error.MismatchedDimensions;

            const n_features = X[0].len;
            const max_features = self.config.max_features orelse blk: {
                // Default: sqrt(n_features)
                const sqrt_f = @sqrt(@as(f64, @floatFromInt(n_features)));
                break :blk @max(1, @as(usize, @intFromFloat(sqrt_f)));
            };

            // Detect number of classes for classification
            if (self.config.task == .classification) {
                var max_class: T = y[0];
                for (y) |label| {
                    if (label > max_class) max_class = label;
                }
                self.n_classes = @as(usize, @intFromFloat(max_class)) + 1;
            }

            // Build each tree using entire dataset (no bootstrap)
            for (self.trees) |*tree| {
                tree.* = try self.buildTree(X, y, n_features, max_features, 0);
            }
        }

        /// Build a single tree recursively
        fn buildTree(
            self: *Self,
            X: []const []const T,
            y: []const T,
            n_features: usize,
            max_features: usize,
            depth: usize,
        ) !Node {
            const n = X.len;

            // Stopping criteria
            if (n < self.config.min_samples_split or depth >= self.config.max_depth) {
                return self.makeLeaf(y);
            }

            // Check if all labels are the same
            const all_same = blk: {
                const first = y[0];
                for (y[1..]) |label| {
                    if (label != first) break :blk false;
                }
                break :blk true;
            };
            if (all_same) return self.makeLeaf(y);

            // Randomly select max_features features
            var selected_features = try self.allocator.alloc(usize, max_features);
            defer self.allocator.free(selected_features);

            var available = try self.allocator.alloc(usize, n_features);
            defer self.allocator.free(available);
            for (0..n_features) |i| available[i] = i;

            for (0..max_features) |i| {
                const remaining = n_features - i;
                const idx = self.prng.random().uintLessThan(usize, remaining);
                selected_features[i] = available[idx];
                available[idx] = available[remaining - 1];
            }

            // Find best random split among selected features
            var best_feature: ?usize = null;
            var best_threshold: ?T = null;
            var best_score: T = std.math.inf(T);

            for (selected_features) |feature| {
                // Random threshold: uniformly sample from feature range
                var min_val = X[0][feature];
                var max_val = X[0][feature];
                for (X[1..]) |sample| {
                    if (sample[feature] < min_val) min_val = sample[feature];
                    if (sample[feature] > max_val) max_val = sample[feature];
                }

                if (min_val >= max_val) continue; // constant feature

                const rand_val = self.prng.random().float(T);
                const threshold = min_val + rand_val * (max_val - min_val);

                // Compute score (impurity reduction)
                const split_score = try self.computeScore(X, y, feature, threshold);
                if (split_score < best_score) {
                    best_score = split_score;
                    best_feature = feature;
                    best_threshold = threshold;
                }
            }

            if (best_feature == null) {
                return self.makeLeaf(y);
            }

            // Split data
            var left_X = std.ArrayList([]const T).init(self.allocator);
            defer left_X.deinit();
            var left_y = std.ArrayList(T).init(self.allocator);
            defer left_y.deinit();
            var right_X = std.ArrayList([]const T).init(self.allocator);
            defer right_X.deinit();
            var right_y = std.ArrayList(T).init(self.allocator);
            defer right_y.deinit();

            const feat = best_feature.?;
            const thresh = best_threshold.?;

            for (X, 0..) |sample, i| {
                if (sample[feat] <= thresh) {
                    try left_X.append(sample);
                    try left_y.append(y[i]);
                } else {
                    try right_X.append(sample);
                    try right_y.append(y[i]);
                }
            }

            if (left_X.items.len == 0 or right_X.items.len == 0) {
                return self.makeLeaf(y);
            }

            // Recursively build subtrees
            var node = Node{
                .feature = feat,
                .threshold = thresh,
            };

            const left_node = try self.allocator.create(Node);
            errdefer self.allocator.destroy(left_node);
            left_node.* = try self.buildTree(left_X.items, left_y.items, n_features, max_features, depth + 1);
            node.left = left_node;

            const right_node = try self.allocator.create(Node);
            errdefer self.allocator.destroy(right_node);
            right_node.* = try self.buildTree(right_X.items, right_y.items, n_features, max_features, depth + 1);
            node.right = right_node;

            return node;
        }

        /// Create a leaf node
        fn makeLeaf(self: *Self, y: []const T) Node {
            var value: T = 0;
            if (self.config.task == .classification) {
                // Majority vote
                var counts = self.allocator.alloc(usize, self.n_classes) catch return Node{ .value = y[0] };
                defer self.allocator.free(counts);
                @memset(counts, 0);

                for (y) |label| {
                    const idx = @as(usize, @intFromFloat(label));
                    if (idx < counts.len) counts[idx] += 1;
                }

                var max_count: usize = 0;
                var max_class: usize = 0;
                for (counts, 0..) |count, i| {
                    if (count > max_count) {
                        max_count = count;
                        max_class = i;
                    }
                }
                value = @floatFromInt(max_class);
            } else {
                // Mean for regression
                var sum: T = 0;
                for (y) |label| sum += label;
                value = sum / @as(T, @floatFromInt(y.len));
            }
            return Node{ .value = value };
        }

        /// Compute impurity score for a split
        fn computeScore(self: *Self, X: []const []const T, y: []const T, feature: usize, threshold: T) !T {
            var left_y = std.ArrayList(T).init(self.allocator);
            defer left_y.deinit();
            var right_y = std.ArrayList(T).init(self.allocator);
            defer right_y.deinit();

            for (X, 0..) |sample, i| {
                if (sample[feature] <= threshold) {
                    try left_y.append(y[i]);
                } else {
                    try right_y.append(y[i]);
                }
            }

            if (left_y.items.len == 0 or right_y.items.len == 0) {
                return std.math.inf(T);
            }

            const n = @as(T, @floatFromInt(y.len));
            const n_left = @as(T, @floatFromInt(left_y.items.len));
            const n_right = @as(T, @floatFromInt(right_y.items.len));

            const left_impurity = self.computeImpurity(left_y.items);
            const right_impurity = self.computeImpurity(right_y.items);

            // Weighted impurity
            return (n_left / n) * left_impurity + (n_right / n) * right_impurity;
        }

        /// Compute impurity (Gini for classification, variance for regression)
        fn computeImpurity(self: *Self, y: []const T) T {
            if (y.len == 0) return 0;

            if (self.config.task == .classification) {
                // Gini impurity
                var counts = self.allocator.alloc(usize, self.n_classes) catch return 0;
                defer self.allocator.free(counts);
                @memset(counts, 0);

                for (y) |label| {
                    const idx = @as(usize, @intFromFloat(label));
                    if (idx < counts.len) counts[idx] += 1;
                }

                var gini: T = 1.0;
                const n = @as(T, @floatFromInt(y.len));
                for (counts) |count| {
                    const p = @as(T, @floatFromInt(count)) / n;
                    gini -= p * p;
                }
                return gini;
            } else {
                // Variance
                var sum: T = 0;
                for (y) |label| sum += label;
                const mean = sum / @as(T, @floatFromInt(y.len));

                var variance: T = 0;
                for (y) |label| {
                    const diff = label - mean;
                    variance += diff * diff;
                }
                return variance / @as(T, @floatFromInt(y.len));
            }
        }

        /// Predict a single sample through a tree
        fn predictTree(node: *const Node, x: []const T) T {
            if (node.feature == null) {
                return node.value;
            }

            const feat = node.feature.?;
            const thresh = node.threshold.?;

            if (x[feat] <= thresh) {
                return predictTree(node.left.?, x);
            } else {
                return predictTree(node.right.?, x);
            }
        }

        /// Predict class or value for a single sample
        ///
        /// Time: O(n_trees × depth)
        /// Space: O(1)
        pub fn predict(self: *const Self, x: []const T) T {
            if (self.config.task == .classification) {
                // Majority vote across trees
                var counts = self.allocator.alloc(usize, self.n_classes) catch return 0;
                defer self.allocator.free(counts);
                @memset(counts, 0);

                for (self.trees) |*tree| {
                    const pred = predictTree(tree, x);
                    const idx = @as(usize, @intFromFloat(pred));
                    if (idx < counts.len) counts[idx] += 1;
                }

                var max_count: usize = 0;
                var max_class: usize = 0;
                for (counts, 0..) |count, i| {
                    if (count > max_count) {
                        max_count = count;
                        max_class = i;
                    }
                }
                return @floatFromInt(max_class);
            } else {
                // Mean across trees
                var sum: T = 0;
                for (self.trees) |*tree| {
                    sum += predictTree(tree, x);
                }
                return sum / @as(T, @floatFromInt(self.trees.len));
            }
        }

        /// Predict multiple samples
        ///
        /// Time: O(batch × n_trees × depth)
        /// Space: O(batch)
        pub fn predictBatch(self: *const Self, X: []const []const T, allocator: Allocator) ![]T {
            const predictions = try allocator.alloc(T, X.len);
            for (X, 0..) |x, i| {
                predictions[i] = self.predict(x);
            }
            return predictions;
        }

        /// Compute accuracy for classification
        ///
        /// Time: O(n × n_trees × depth)
        pub fn score(self: *const Self, X: []const []const T, y: []const T) !T {
            if (X.len != y.len) return error.MismatchedDimensions;

            var correct: usize = 0;
            for (X, 0..) |x, i| {
                const pred = self.predict(x);
                if (@abs(pred - y[i]) < 1e-6) correct += 1;
            }

            return @as(T, @floatFromInt(correct)) / @as(T, @floatFromInt(X.len));
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const expect = testing.expect;
const expectEqual = testing.expectEqual;
const expectApproxEqRel = testing.expectApproxEqRel;

test "ExtraTrees: initialization and cleanup" {
    const config = ExtraTrees(f64).Config{
        .task = .classification,
        .n_trees = 10,
        .max_depth = 5,
    };
    var et = try ExtraTrees(f64).init(testing.allocator, config);
    defer et.deinit();

    try expectEqual(@as(usize, 10), et.trees.len);
    try expectEqual(@as(usize, 5), et.config.max_depth);
}

test "ExtraTrees: binary classification - linearly separable" {
    const config = ExtraTrees(f64).Config{
        .task = .classification,
        .n_trees = 50,
        .max_depth = 5,
        .random_seed = 42,
    };
    var et = try ExtraTrees(f64).init(testing.allocator, config);
    defer et.deinit();

    // Simple linearly separable data
    var X_data = [_][2]f64{
        .{ 1.0, 1.0 }, .{ 1.5, 1.5 }, .{ 2.0, 2.0 },
        .{ 5.0, 5.0 }, .{ 5.5, 5.5 }, .{ 6.0, 6.0 },
    };
    var X = [_][]const f64{
        &X_data[0], &X_data[1], &X_data[2],
        &X_data[3], &X_data[4], &X_data[5],
    };
    const y = [_]f64{ 0, 0, 0, 1, 1, 1 };

    try et.fit(&X, &y);

    // Test predictions
    try expectEqual(@as(f64, 0), et.predict(&X_data[0]));
    try expectEqual(@as(f64, 1), et.predict(&X_data[5]));

    // Test accuracy
    const accuracy = try et.score(&X, &y);
    try expect(accuracy >= 0.8); // Should achieve high accuracy
}

test "ExtraTrees: multi-class classification" {
    const config = ExtraTrees(f64).Config{
        .task = .classification,
        .n_trees = 50,
        .max_depth = 5,
        .random_seed = 123,
    };
    var et = try ExtraTrees(f64).init(testing.allocator, config);
    defer et.deinit();

    // Three-class problem
    var X_data = [_][2]f64{
        .{ 0.0, 0.0 }, .{ 0.5, 0.5 }, // class 0
        .{ 3.0, 3.0 }, .{ 3.5, 3.5 }, // class 1
        .{ 6.0, 6.0 }, .{ 6.5, 6.5 }, // class 2
    };
    var X = [_][]const f64{
        &X_data[0], &X_data[1], &X_data[2],
        &X_data[3], &X_data[4], &X_data[5],
    };
    const y = [_]f64{ 0, 0, 1, 1, 2, 2 };

    try et.fit(&X, &y);

    const accuracy = try et.score(&X, &y);
    try expect(accuracy >= 0.7);
}

test "ExtraTrees: regression - linear relationship" {
    const config = ExtraTrees(f64).Config{
        .task = .regression,
        .n_trees = 50,
        .max_depth = 10,
        .random_seed = 42,
    };
    var et = try ExtraTrees(f64).init(testing.allocator, config);
    defer et.deinit();

    // y = 2x + 3
    var X_data = [_][1]f64{
        .{0.0}, .{1.0}, .{2.0}, .{3.0}, .{4.0}, .{5.0},
    };
    var X = [_][]const f64{
        &X_data[0], &X_data[1], &X_data[2],
        &X_data[3], &X_data[4], &X_data[5],
    };
    const y = [_]f64{ 3.0, 5.0, 7.0, 9.0, 11.0, 13.0 };

    try et.fit(&X, &y);

    // Test predictions
    const pred0 = et.predict(&X_data[0]);
    const pred5 = et.predict(&X_data[5]);

    try expectApproxEqRel(@as(f64, 3.0), pred0, 0.2); // Allow 20% error
    try expectApproxEqRel(@as(f64, 13.0), pred5, 0.2);
}

test "ExtraTrees: regression - polynomial relationship" {
    const config = ExtraTrees(f64).Config{
        .task = .regression,
        .n_trees = 100,
        .max_depth = 15,
        .random_seed = 123,
    };
    var et = try ExtraTrees(f64).init(testing.allocator, config);
    defer et.deinit();

    // y = x^2
    var X_data = [_][1]f64{
        .{0.0}, .{1.0}, .{2.0}, .{3.0}, .{4.0}, .{5.0},
    };
    var X = [_][]const f64{
        &X_data[0], &X_data[1], &X_data[2],
        &X_data[3], &X_data[4], &X_data[5],
    };
    const y = [_]f64{ 0.0, 1.0, 4.0, 9.0, 16.0, 25.0 };

    try et.fit(&X, &y);

    const pred4 = et.predict(&X_data[4]);
    try expectApproxEqRel(@as(f64, 16.0), pred4, 0.3); // Trees approximate curves
}

test "ExtraTrees: batch prediction" {
    const config = ExtraTrees(f64).Config{
        .task = .classification,
        .n_trees = 30,
        .max_depth = 5,
    };
    var et = try ExtraTrees(f64).init(testing.allocator, config);
    defer et.deinit();

    var X_data = [_][2]f64{
        .{ 1.0, 1.0 }, .{ 2.0, 2.0 },
        .{ 5.0, 5.0 }, .{ 6.0, 6.0 },
    };
    var X = [_][]const f64{ &X_data[0], &X_data[1], &X_data[2], &X_data[3] };
    const y = [_]f64{ 0, 0, 1, 1 };

    try et.fit(&X, &y);

    const predictions = try et.predictBatch(&X, testing.allocator);
    defer testing.allocator.free(predictions);

    try expectEqual(@as(usize, 4), predictions.len);
}

test "ExtraTrees: max_features parameter" {
    const config = ExtraTrees(f64).Config{
        .task = .classification,
        .n_trees = 20,
        .max_depth = 5,
        .max_features = 1, // Only consider 1 feature per split
    };
    var et = try ExtraTrees(f64).init(testing.allocator, config);
    defer et.deinit();

    var X_data = [_][3]f64{
        .{ 1.0, 0.0, 0.0 }, .{ 2.0, 0.0, 0.0 },
        .{ 5.0, 0.0, 0.0 }, .{ 6.0, 0.0, 0.0 },
    };
    var X = [_][]const f64{ &X_data[0], &X_data[1], &X_data[2], &X_data[3] };
    const y = [_]f64{ 0, 0, 1, 1 };

    try et.fit(&X, &y);

    const accuracy = try et.score(&X, &y);
    try expect(accuracy > 0.5); // Should still learn something
}

test "ExtraTrees: default max_features (sqrt)" {
    const config = ExtraTrees(f64).Config{
        .task = .classification,
        .n_trees = 20,
        .max_depth = 5,
        .max_features = null, // Default: sqrt(n_features)
    };
    var et = try ExtraTrees(f64).init(testing.allocator, config);
    defer et.deinit();

    var X_data = [_][4]f64{
        .{ 1.0, 0.0, 0.0, 0.0 }, .{ 2.0, 0.0, 0.0, 0.0 },
        .{ 5.0, 0.0, 0.0, 0.0 }, .{ 6.0, 0.0, 0.0, 0.0 },
    };
    var X = [_][]const f64{ &X_data[0], &X_data[1], &X_data[2], &X_data[3] };
    const y = [_]f64{ 0, 0, 1, 1 };

    try et.fit(&X, &y);

    const accuracy = try et.score(&X, &y);
    try expect(accuracy > 0.5);
}

test "ExtraTrees: large dataset" {
    const config = ExtraTrees(f64).Config{
        .task = .classification,
        .n_trees = 50,
        .max_depth = 10,
        .random_seed = 42,
    };
    var et = try ExtraTrees(f64).init(testing.allocator, config);
    defer et.deinit();

    // 100 samples
    var X_data: [100][2]f64 = undefined;
    var X_ptrs: [100][]const f64 = undefined;
    var y_data: [100]f64 = undefined;

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (0..100) |i| {
        X_data[i] = .{
            random.float(f64) * 10.0,
            random.float(f64) * 10.0,
        };
        X_ptrs[i] = &X_data[i];
        // Simple rule: class 0 if sum < 10, else class 1
        y_data[i] = if (X_data[i][0] + X_data[i][1] < 10.0) 0.0 else 1.0;
    }

    try et.fit(&X_ptrs, &y_data);

    const accuracy = try et.score(&X_ptrs, &y_data);
    try expect(accuracy >= 0.7); // Should learn the pattern
}

test "ExtraTrees: empty input error" {
    const config = ExtraTrees(f64).Config{};
    var et = try ExtraTrees(f64).init(testing.allocator, config);
    defer et.deinit();

    var X: [0][]const f64 = undefined;
    const y: [0]f64 = undefined;

    try testing.expectError(error.EmptyInput, et.fit(&X, &y));
}

test "ExtraTrees: mismatched dimensions error" {
    const config = ExtraTrees(f64).Config{};
    var et = try ExtraTrees(f64).init(testing.allocator, config);
    defer et.deinit();

    var X_data = [_][2]f64{ .{ 1.0, 1.0 }, .{ 2.0, 2.0 } };
    var X = [_][]const f64{ &X_data[0], &X_data[1] };
    const y = [_]f64{0}; // Wrong length

    try testing.expectError(error.MismatchedDimensions, et.fit(&X, &y));
}

test "ExtraTrees: f32 support" {
    const config = ExtraTrees(f32).Config{
        .task = .classification,
        .n_trees = 20,
        .max_depth = 5,
    };
    var et = try ExtraTrees(f32).init(testing.allocator, config);
    defer et.deinit();

    var X_data = [_][2]f32{
        .{ 1.0, 1.0 }, .{ 2.0, 2.0 },
        .{ 5.0, 5.0 }, .{ 6.0, 6.0 },
    };
    var X = [_][]const f32{ &X_data[0], &X_data[1], &X_data[2], &X_data[3] };
    const y = [_]f32{ 0, 0, 1, 1 };

    try et.fit(&X, &y);

    const accuracy = try et.score(&X, &y);
    try expect(accuracy >= 0.8);
}

test "ExtraTrees: memory safety" {
    const config = ExtraTrees(f64).Config{
        .task = .classification,
        .n_trees = 10,
        .max_depth = 5,
    };
    var et = try ExtraTrees(f64).init(testing.allocator, config);
    defer et.deinit();

    var X_data = [_][2]f64{ .{ 1.0, 1.0 }, .{ 5.0, 5.0 } };
    var X = [_][]const f64{ &X_data[0], &X_data[1] };
    const y = [_]f64{ 0, 1 };

    try et.fit(&X, &y);
    _ = et.predict(&X_data[0]);
    // No memory leaks expected with testing.allocator
}

test "ExtraTrees: single sample per class" {
    const config = ExtraTrees(f64).Config{
        .task = .classification,
        .n_trees = 10,
        .max_depth = 5,
    };
    var et = try ExtraTrees(f64).init(testing.allocator, config);
    defer et.deinit();

    var X_data = [_][2]f64{ .{ 1.0, 1.0 }, .{ 5.0, 5.0 } };
    var X = [_][]const f64{ &X_data[0], &X_data[1] };
    const y = [_]f64{ 0, 1 };

    try et.fit(&X, &y);

    // Should still make reasonable predictions
    const pred0 = et.predict(&X_data[0]);
    const pred1 = et.predict(&X_data[1]);

    try expect(pred0 == 0.0 or pred0 == 1.0);
    try expect(pred1 == 0.0 or pred1 == 1.0);
}
