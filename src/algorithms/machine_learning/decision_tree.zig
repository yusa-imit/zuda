const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Decision Tree for classification and regression tasks.
///
/// Implements recursive partitioning with configurable splitting criteria.
/// Supports both categorical (classification) and continuous (regression) targets.
///
/// Classification uses Gini impurity or entropy for split selection.
/// Regression uses variance reduction (MSE minimization).
///
/// Time: O(n × m × log n) training, O(depth) prediction
/// Space: O(nodes) where nodes ≈ O(n) worst case
pub fn DecisionTree(comptime T: type) type {
    return struct {
        allocator: Allocator,
        root: ?*Node,
        tree_type: TreeType,
        max_depth: u32,
        min_samples_split: u32,
        min_samples_leaf: u32,
        n_features: usize,

        const Self = @This();

        pub const TreeType = enum {
            classification,
            regression,
        };

        pub const SplitCriterion = enum {
            gini, // Classification: Gini impurity
            entropy, // Classification: Information gain
            mse, // Regression: Mean squared error
        };

        pub const Node = struct {
            // Split information (internal nodes)
            feature_idx: ?usize, // null for leaf nodes
            threshold: T,

            // Leaf information
            value: T, // Class label (classification) or mean value (regression)
            samples: u32,
            impurity: T,

            // Tree structure
            left: ?*Node,
            right: ?*Node,

            pub fn isLeaf(self: *const Node) bool {
                return self.feature_idx == null;
            }
        };

        pub const Config = struct {
            tree_type: TreeType = .classification,
            max_depth: u32 = 10,
            min_samples_split: u32 = 2,
            min_samples_leaf: u32 = 1,
        };

        /// Initialize decision tree with configuration.
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(allocator: Allocator, config: Config) Self {
            return .{
                .allocator = allocator,
                .root = null,
                .tree_type = config.tree_type,
                .max_depth = config.max_depth,
                .min_samples_split = config.min_samples_split,
                .min_samples_leaf = config.min_samples_leaf,
                .n_features = 0,
            };
        }

        /// Free all nodes in the tree.
        ///
        /// Time: O(nodes)
        /// Space: O(depth) for recursion stack
        pub fn deinit(self: *Self) void {
            if (self.root) |root| {
                self.freeNode(root);
            }
        }

        fn freeNode(self: *Self, node: *Node) void {
            if (node.left) |left| self.freeNode(left);
            if (node.right) |right| self.freeNode(right);
            self.allocator.destroy(node);
        }

        /// Train the decision tree on the given dataset.
        ///
        /// X: feature matrix (n_samples × n_features)
        /// y: target values (n_samples)
        /// criterion: splitting criterion
        ///
        /// Time: O(n × m × log n)
        /// Space: O(nodes)
        pub fn fit(
            self: *Self,
            X: []const []const T,
            y: []const T,
            criterion: SplitCriterion,
        ) !void {
            if (X.len == 0 or y.len == 0 or X.len != y.len) {
                return error.InvalidInput;
            }

            self.n_features = X[0].len;
            const indices = try self.allocator.alloc(usize, X.len);
            defer self.allocator.free(indices);

            for (0..X.len) |i| {
                indices[i] = i;
            }

            self.root = try self.buildTree(X, y, indices, 0, criterion);
        }

        fn buildTree(
            self: *Self,
            X: []const []const T,
            y: []const T,
            indices: []const usize,
            depth: u32,
            criterion: SplitCriterion,
        ) !*Node {
            const node = try self.allocator.create(Node);
            errdefer self.allocator.destroy(node);

            node.samples = @intCast(indices.len);
            node.left = null;
            node.right = null;
            node.feature_idx = null;

            // Stopping criteria
            const should_stop = depth >= self.max_depth or
                indices.len < self.min_samples_split or
                self.isPure(y, indices);

            if (should_stop) {
                try self.makeLeaf(node, y, indices, criterion);
                return node;
            }

            // Find best split
            const split = try self.findBestSplit(X, y, indices, criterion);

            if (split.gain <= 0 or
                split.left_indices.len < self.min_samples_leaf or
                split.right_indices.len < self.min_samples_leaf)
            {
                self.allocator.free(split.left_indices);
                self.allocator.free(split.right_indices);
                try self.makeLeaf(node, y, indices, criterion);
                return node;
            }

            // Create internal node
            node.feature_idx = split.feature_idx;
            node.threshold = split.threshold;
            node.impurity = split.impurity;
            node.value = self.calculateNodeValue(y, indices);

            // Recursively build subtrees
            node.left = try self.buildTree(X, y, split.left_indices, depth + 1, criterion);
            node.right = try self.buildTree(X, y, split.right_indices, depth + 1, criterion);

            self.allocator.free(split.left_indices);
            self.allocator.free(split.right_indices);

            return node;
        }

        const Split = struct {
            feature_idx: usize,
            threshold: T,
            gain: T,
            impurity: T,
            left_indices: []usize,
            right_indices: []usize,
        };

        fn findBestSplit(
            self: *Self,
            X: []const []const T,
            y: []const T,
            indices: []const usize,
            criterion: SplitCriterion,
        ) !Split {
            var best_gain: T = -std.math.inf(T);
            var best_feature: usize = 0;
            var best_threshold: T = 0;
            var best_left = try std.ArrayList(usize).initCapacity(self.allocator, 0);
            var best_right = try std.ArrayList(usize).initCapacity(self.allocator, 0);

            const parent_impurity = self.calculateImpurity(y, indices, criterion);

            // Try each feature
            for (0..self.n_features) |feature_idx| {
                // Get unique values for this feature
                var values = try std.ArrayList(T).initCapacity(self.allocator, indices.len);
                defer values.deinit(self.allocator);

                for (indices) |idx| {
                    try values.append(self.allocator, X[idx][feature_idx]);
                }

                std.mem.sort(T, values.items, {}, comptime std.sort.asc(T));

                // Try split points between consecutive values
                var i: usize = 0;
                while (i < values.items.len - 1) : (i += 1) {
                    if (values.items[i] == values.items[i + 1]) continue;

                    const threshold = (values.items[i] + values.items[i + 1]) / 2;

                    var left = try std.ArrayList(usize).initCapacity(self.allocator, 0);
                    var right = try std.ArrayList(usize).initCapacity(self.allocator, 0);

                    for (indices) |idx| {
                        if (X[idx][feature_idx] <= threshold) {
                            try left.append(self.allocator, idx);
                        } else {
                            try right.append(self.allocator, idx);
                        }
                    }

                    if (left.items.len == 0 or right.items.len == 0) {
                        left.deinit(self.allocator);
                        right.deinit(self.allocator);
                        continue;
                    }

                    const left_impurity = self.calculateImpurity(y, left.items, criterion);
                    const right_impurity = self.calculateImpurity(y, right.items, criterion);

                    const n_total: T = @floatFromInt(indices.len);
                    const n_left: T = @floatFromInt(left.items.len);
                    const n_right: T = @floatFromInt(right.items.len);

                    const weighted_impurity = (n_left / n_total) * left_impurity +
                        (n_right / n_total) * right_impurity;

                    const gain = parent_impurity - weighted_impurity;

                    if (gain > best_gain) {
                        best_gain = gain;
                        best_feature = feature_idx;
                        best_threshold = threshold;

                        best_left.deinit(self.allocator);
                        best_right.deinit(self.allocator);
                        best_left = left;
                        best_right = right;
                    } else {
                        left.deinit(self.allocator);
                        right.deinit(self.allocator);
                    }
                }
            }

            return .{
                .feature_idx = best_feature,
                .threshold = best_threshold,
                .gain = best_gain,
                .impurity = parent_impurity,
                .left_indices = try best_left.toOwnedSlice(self.allocator),
                .right_indices = try best_right.toOwnedSlice(self.allocator),
            };
        }

        fn calculateImpurity(
            self: *Self,
            y: []const T,
            indices: []const usize,
            criterion: SplitCriterion,
        ) T {
            return switch (criterion) {
                .gini => self.giniImpurity(y, indices),
                .entropy => self.entropy(y, indices),
                .mse => self.mse(y, indices),
            };
        }

        fn giniImpurity(self: *Self, y: []const T, indices: []const usize) T {
            var counts = std.AutoHashMap(i64, usize).init(self.allocator);
            defer counts.deinit();

            for (indices) |idx| {
                const label: i64 = @intFromFloat(y[idx]);
                const entry = counts.getOrPut(label) catch unreachable;
                if (entry.found_existing) {
                    entry.value_ptr.* += 1;
                } else {
                    entry.value_ptr.* = 1;
                }
            }

            var impurity: T = 1.0;
            const n: T = @floatFromInt(indices.len);

            var it = counts.iterator();
            while (it.next()) |entry| {
                const p: T = @as(T, @floatFromInt(entry.value_ptr.*)) / n;
                impurity -= p * p;
            }

            return impurity;
        }

        fn entropy(self: *Self, y: []const T, indices: []const usize) T {
            var counts = std.AutoHashMap(i64, usize).init(self.allocator);
            defer counts.deinit();

            for (indices) |idx| {
                const label: i64 = @intFromFloat(y[idx]);
                const entry = counts.getOrPut(label) catch unreachable;
                if (entry.found_existing) {
                    entry.value_ptr.* += 1;
                } else {
                    entry.value_ptr.* = 1;
                }
            }

            var ent: T = 0.0;
            const n: T = @floatFromInt(indices.len);

            var it = counts.iterator();
            while (it.next()) |entry| {
                const p: T = @as(T, @floatFromInt(entry.value_ptr.*)) / n;
                if (p > 0) {
                    ent -= p * @log(p);
                }
            }

            return ent;
        }

        fn mse(_: *Self, y: []const T, indices: []const usize) T {
            if (indices.len == 0) return 0;

            var sum: T = 0;
            for (indices) |idx| {
                sum += y[idx];
            }
            const mean = sum / @as(T, @floatFromInt(indices.len));

            var variance: T = 0;
            for (indices) |idx| {
                const diff = y[idx] - mean;
                variance += diff * diff;
            }

            return variance / @as(T, @floatFromInt(indices.len));
        }

        fn isPure(_: *Self, y: []const T, indices: []const usize) bool {
            if (indices.len <= 1) return true;

            const first_value = y[indices[0]];
            for (indices[1..]) |idx| {
                if (y[idx] != first_value) return false;
            }
            return true;
        }

        fn makeLeaf(
            self: *Self,
            node: *Node,
            y: []const T,
            indices: []const usize,
            criterion: SplitCriterion,
        ) !void {
            node.value = self.calculateNodeValue(y, indices);
            node.impurity = self.calculateImpurity(y, indices, criterion);
        }

        fn calculateNodeValue(self: *Self, y: []const T, indices: []const usize) T {
            return switch (self.tree_type) {
                .classification => blk: {
                    // Majority class
                    var counts = std.AutoHashMap(i64, usize).init(self.allocator);
                    defer counts.deinit();

                    for (indices) |idx| {
                        const label: i64 = @intFromFloat(y[idx]);
                        const entry = counts.getOrPut(label) catch unreachable;
                        if (entry.found_existing) {
                            entry.value_ptr.* += 1;
                        } else {
                            entry.value_ptr.* = 1;
                        }
                    }

                    var max_count: usize = 0;
                    var majority_class: i64 = 0;

                    var it = counts.iterator();
                    while (it.next()) |entry| {
                        if (entry.value_ptr.* > max_count) {
                            max_count = entry.value_ptr.*;
                            majority_class = entry.key_ptr.*;
                        }
                    }

                    break :blk @floatFromInt(majority_class);
                },
                .regression => blk: {
                    // Mean value
                    var sum: T = 0;
                    for (indices) |idx| {
                        sum += y[idx];
                    }
                    break :blk sum / @as(T, @floatFromInt(indices.len));
                },
            };
        }

        /// Predict class label or value for a single sample.
        ///
        /// Time: O(depth)
        /// Space: O(1)
        pub fn predict(self: *const Self, x: []const T) !T {
            var node = self.root orelse return error.NotTrained;

            if (x.len != self.n_features) {
                return error.InvalidInput;
            }

            while (!node.isLeaf()) {
                const feature_idx = node.feature_idx.?;
                if (x[feature_idx] <= node.threshold) {
                    node = node.left.?;
                } else {
                    node = node.right.?;
                }
            }

            return node.value;
        }

        /// Predict class labels or values for multiple samples.
        ///
        /// Time: O(n × depth)
        /// Space: O(n)
        pub fn predictBatch(self: *const Self, X: []const []const T) ![]T {
            const predictions = try self.allocator.alloc(T, X.len);
            errdefer self.allocator.free(predictions);

            for (X, 0..) |x, i| {
                predictions[i] = try self.predict(x);
            }

            return predictions;
        }

        /// Get the depth of the tree.
        ///
        /// Time: O(nodes)
        /// Space: O(depth) for recursion
        pub fn getDepth(self: *const Self) u32 {
            if (self.root) |root| {
                return self.nodeDepth(root);
            }
            return 0;
        }

        fn nodeDepth(self: *const Self, node: *const Node) u32 {
            if (node.isLeaf()) return 1;

            const left_depth = if (node.left) |left| self.nodeDepth(left) else 0;
            const right_depth = if (node.right) |right| self.nodeDepth(right) else 0;

            return 1 + @max(left_depth, right_depth);
        }

        /// Get the number of nodes in the tree.
        ///
        /// Time: O(nodes)
        /// Space: O(depth) for recursion
        pub fn countNodes(self: *const Self) u32 {
            if (self.root) |root| {
                return self.countNodesRecursive(root);
            }
            return 0;
        }

        fn countNodesRecursive(self: *const Self, node: *const Node) u32 {
            var count: u32 = 1;
            if (node.left) |left| count += self.countNodesRecursive(left);
            if (node.right) |right| count += self.countNodesRecursive(right);
            return count;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "DecisionTree: basic classification" {
    const allocator = testing.allocator;

    // Simple linearly separable dataset
    const X = [_][]const f64{
        &[_]f64{ 0, 0 },
        &[_]f64{ 0, 1 },
        &[_]f64{ 1, 0 },
        &[_]f64{ 1, 1 },
        &[_]f64{ 2, 2 },
        &[_]f64{ 2, 3 },
    };
    const y = [_]f64{ 0, 0, 0, 1, 1, 1 };

    var tree = DecisionTree(f64).init(allocator, .{
        .tree_type = .classification,
        .max_depth = 3,
    });
    defer tree.deinit();

    try tree.fit(&X, &y, .gini);

    // Test predictions - should separate class 0 (low values) from class 1 (high values)
    const pred0 = try tree.predict(&[_]f64{ 0, 0 });
    const pred1 = try tree.predict(&[_]f64{ 2, 2 });

    try testing.expectEqual(@as(f64, 0), pred0);
    try testing.expectEqual(@as(f64, 1), pred1);
}

test "DecisionTree: basic regression" {
    const allocator = testing.allocator;

    // Simple linear relationship: y = 2x
    const X = [_][]const f64{
        &[_]f64{0.0},
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
        &[_]f64{4.0},
    };
    const y = [_]f64{ 0.0, 2.0, 4.0, 6.0, 8.0 };

    var tree = DecisionTree(f64).init(allocator, .{
        .tree_type = .regression,
        .max_depth = 3,
    });
    defer tree.deinit();

    try tree.fit(&X, &y, .mse);

    // Test predictions (should be close to actual values)
    const pred0 = try tree.predict(&[_]f64{0.5});
    const pred1 = try tree.predict(&[_]f64{2.5});

    // Predictions should be reasonable approximations
    try testing.expect(@abs(pred0 - 1.0) < 2.0);
    try testing.expect(@abs(pred1 - 5.0) < 2.0);
}

test "DecisionTree: complex classification" {
    const allocator = testing.allocator;

    // More samples to allow for better tree building
    const X = [_][]const f64{
        &[_]f64{ 0, 0 },
        &[_]f64{ 0, 0.5 },
        &[_]f64{ 0, 1 },
        &[_]f64{ 0.5, 0 },
        &[_]f64{ 0.5, 1 },
        &[_]f64{ 1, 0 },
        &[_]f64{ 1, 0.5 },
        &[_]f64{ 1, 1 },
    };
    const y = [_]f64{ 0, 0, 1, 0, 1, 1, 1, 0 };

    var tree = DecisionTree(f64).init(allocator, .{
        .tree_type = .classification,
        .max_depth = 5,
    });
    defer tree.deinit();

    try tree.fit(&X, &y, .gini);

    // Verify tree structure
    const depth = tree.getDepth();
    const nodes = tree.countNodes();
    try testing.expect(depth >= 1);
    try testing.expect(nodes >= 1);
}

test "DecisionTree: entropy criterion" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{ 0, 0 },
        &[_]f64{ 0, 1 },
        &[_]f64{ 1, 0 },
        &[_]f64{ 1, 1 },
    };
    const y = [_]f64{ 0, 1, 1, 0 };

    var tree = DecisionTree(f64).init(allocator, .{
        .tree_type = .classification,
        .max_depth = 3,
    });
    defer tree.deinit();

    try tree.fit(&X, &y, .entropy);

    const pred = try tree.predict(&[_]f64{ 0, 1 });
    try testing.expectEqual(@as(f64, 1), pred);
}

test "DecisionTree: batch prediction" {
    const allocator = testing.allocator;

    const X_train = [_][]const f64{
        &[_]f64{ 0, 0 },
        &[_]f64{ 1, 1 },
    };
    const y_train = [_]f64{ 0, 1 };

    var tree = DecisionTree(f64).init(allocator, .{
        .tree_type = .classification,
    });
    defer tree.deinit();

    try tree.fit(&X_train, &y_train, .gini);

    const X_test = [_][]const f64{
        &[_]f64{ 0, 0 },
        &[_]f64{ 1, 1 },
    };

    const predictions = try tree.predictBatch(&X_test);
    defer allocator.free(predictions);

    try testing.expectEqual(@as(usize, 2), predictions.len);
    try testing.expectEqual(@as(f64, 0), predictions[0]);
    try testing.expectEqual(@as(f64, 1), predictions[1]);
}

test "DecisionTree: depth and node count" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{ 0, 0 },
        &[_]f64{ 0, 1 },
        &[_]f64{ 1, 0 },
        &[_]f64{ 1, 1 },
    };
    const y = [_]f64{ 0, 1, 1, 0 };

    var tree = DecisionTree(f64).init(allocator, .{
        .tree_type = .classification,
        .max_depth = 5,
    });
    defer tree.deinit();

    try tree.fit(&X, &y, .gini);

    const depth = tree.getDepth();
    const nodes = tree.countNodes();

    try testing.expect(depth > 0);
    try testing.expect(nodes > 0);
}

test "DecisionTree: min_samples_split" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{0.0},
        &[_]f64{1.0},
        &[_]f64{2.0},
    };
    const y = [_]f64{ 0, 1, 0 };

    var tree = DecisionTree(f64).init(allocator, .{
        .tree_type = .classification,
        .max_depth = 10,
        .min_samples_split = 10, // Won't split with only 3 samples
    });
    defer tree.deinit();

    try tree.fit(&X, &y, .gini);

    // Should create a single leaf node
    try testing.expectEqual(@as(u32, 1), tree.countNodes());
}

test "DecisionTree: min_samples_leaf" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{0.0},
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
    };
    const y = [_]f64{ 0, 1, 0, 1 };

    var tree = DecisionTree(f64).init(allocator, .{
        .tree_type = .classification,
        .max_depth = 10,
        .min_samples_leaf = 2,
    });
    defer tree.deinit();

    try tree.fit(&X, &y, .gini);

    // Tree should be limited by min_samples_leaf constraint
    const depth = tree.getDepth();
    try testing.expect(depth <= 3);
}

test "DecisionTree: pure leaf" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{0.0},
        &[_]f64{1.0},
        &[_]f64{2.0},
    };
    const y = [_]f64{ 0, 0, 0 }; // All same class

    var tree = DecisionTree(f64).init(allocator, .{
        .tree_type = .classification,
    });
    defer tree.deinit();

    try tree.fit(&X, &y, .gini);

    // Should create a single leaf (pure node)
    try testing.expectEqual(@as(u32, 1), tree.countNodes());

    const pred = try tree.predict(&[_]f64{1.5});
    try testing.expectEqual(@as(f64, 0), pred);
}

test "DecisionTree: multiclass classification" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{ 0, 0 },
        &[_]f64{ 0, 1 },
        &[_]f64{ 1, 0 },
        &[_]f64{ 1, 1 },
        &[_]f64{ 2, 2 },
        &[_]f64{ 2, 3 },
    };
    const y = [_]f64{ 0, 0, 1, 1, 2, 2 };

    var tree = DecisionTree(f64).init(allocator, .{
        .tree_type = .classification,
        .max_depth = 5,
    });
    defer tree.deinit();

    try tree.fit(&X, &y, .gini);

    // Test predictions for each class
    const pred0 = try tree.predict(&[_]f64{ 0, 0.5 });
    const pred1 = try tree.predict(&[_]f64{ 1, 0.5 });
    const pred2 = try tree.predict(&[_]f64{ 2, 2.5 });

    try testing.expectEqual(@as(f64, 0), pred0);
    try testing.expectEqual(@as(f64, 1), pred1);
    try testing.expectEqual(@as(f64, 2), pred2);
}

test "DecisionTree: empty input error" {
    const allocator = testing.allocator;

    const X: []const []const f64 = &[_][]const f64{};
    const y: []const f64 = &[_]f64{};

    var tree = DecisionTree(f64).init(allocator, .{});
    defer tree.deinit();

    try testing.expectError(error.InvalidInput, tree.fit(X, y, .gini));
}

test "DecisionTree: mismatched dimensions error" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{0.0},
        &[_]f64{1.0},
    };
    const y = [_]f64{0.0}; // Wrong length

    var tree = DecisionTree(f64).init(allocator, .{});
    defer tree.deinit();

    try testing.expectError(error.InvalidInput, tree.fit(&X, &y, .gini));
}

test "DecisionTree: predict without training error" {
    const allocator = testing.allocator;

    var tree = DecisionTree(f64).init(allocator, .{});
    defer tree.deinit();

    try testing.expectError(error.NotTrained, tree.predict(&[_]f64{0.0}));
}

test "DecisionTree: predict wrong dimension error" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &[_]f64{ 0, 0 },
        &[_]f64{ 1, 1 },
    };
    const y = [_]f64{ 0, 1 };

    var tree = DecisionTree(f64).init(allocator, .{});
    defer tree.deinit();

    try tree.fit(&X, &y, .gini);

    // Wrong number of features
    try testing.expectError(error.InvalidInput, tree.predict(&[_]f64{0.0}));
}

test "DecisionTree: regression with noise" {
    const allocator = testing.allocator;

    // y ≈ x + noise
    const X = [_][]const f64{
        &[_]f64{0.0},
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
        &[_]f64{4.0},
        &[_]f64{5.0},
    };
    const y = [_]f64{ 0.1, 1.2, 1.9, 3.1, 4.0, 5.2 };

    var tree = DecisionTree(f64).init(allocator, .{
        .tree_type = .regression,
        .max_depth = 4,
    });
    defer tree.deinit();

    try tree.fit(&X, &y, .mse);

    // Predictions should follow the general trend
    const pred0 = try tree.predict(&[_]f64{0.5});
    const pred1 = try tree.predict(&[_]f64{2.5});
    const pred2 = try tree.predict(&[_]f64{4.5});

    // Check general ordering
    try testing.expect(pred0 < pred1);
    try testing.expect(pred1 < pred2);
}

test "DecisionTree: f32 support" {
    const allocator = testing.allocator;

    const X = [_][]const f32{
        &[_]f32{ 0, 0 },
        &[_]f32{ 1, 1 },
    };
    const y = [_]f32{ 0, 1 };

    var tree = DecisionTree(f32).init(allocator, .{
        .tree_type = .classification,
    });
    defer tree.deinit();

    try tree.fit(&X, &y, .gini);

    const pred = try tree.predict(&[_]f32{ 0, 0 });
    try testing.expectEqual(@as(f32, 0), pred);
}

test "DecisionTree: large dataset" {
    const allocator = testing.allocator;

    // Create a larger dataset
    var X_list = try std.ArrayList([]const f64).initCapacity(allocator, 100);
    defer {
        for (X_list.items) |item| allocator.free(item);
        X_list.deinit(allocator);
    }
    var y_list = try std.ArrayList(f64).initCapacity(allocator, 100);
    defer y_list.deinit(allocator);

    // Generate 100 samples
    for (0..100) |i| {
        const x = try allocator.alloc(f64, 2);
        const val: f64 = @floatFromInt(i);
        x[0] = val;
        x[1] = val * 2;
        try X_list.append(allocator, x);

        // Simple classification rule
        const label: f64 = if (i < 50) 0 else 1;
        try y_list.append(allocator, label);
    }

    var tree = DecisionTree(f64).init(allocator, .{
        .tree_type = .classification,
        .max_depth = 10,
    });
    defer tree.deinit();

    try tree.fit(X_list.items, y_list.items, .gini);

    // Test a few predictions
    const pred0 = try tree.predict(&[_]f64{ 10, 20 });
    const pred1 = try tree.predict(&[_]f64{ 60, 120 });

    try testing.expectEqual(@as(f64, 0), pred0);
    try testing.expectEqual(@as(f64, 1), pred1);

    // Check tree size
    const nodes = tree.countNodes();
    try testing.expect(nodes > 1);
    try testing.expect(nodes < 100); // Should be much smaller than dataset
}
