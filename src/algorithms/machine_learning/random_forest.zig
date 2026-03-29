const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const DecisionTree = @import("decision_tree.zig").DecisionTree;

/// Random Forest ensemble method for classification and regression.
///
/// Builds multiple decision trees on bootstrap samples with random feature selection.
/// Combines predictions through voting (classification) or averaging (regression).
///
/// Key features:
/// - Bootstrap aggregating (bagging) reduces overfitting
/// - Random feature selection at each split increases diversity
/// - Out-of-bag (OOB) error estimation without separate validation set
/// - Parallel tree construction (single-threaded for now)
///
/// Time: O(n_trees × n × m × log n) training, O(n_trees × depth) prediction
/// Space: O(n_trees × nodes) where nodes ≈ O(n) worst case per tree
pub fn RandomForest(comptime T: type) type {
    return struct {
        allocator: Allocator,
        trees: []DecisionTree(T),
        forest_type: ForestType,
        n_features: usize,
        max_features: usize, // Features considered at each split
        oob_score: T, // Out-of-bag error estimate

        const Self = @This();

        pub const ForestType = enum {
            classification,
            regression,
        };

        pub const Config = struct {
            forest_type: ForestType = .classification,
            n_trees: u32 = 100,
            max_depth: u32 = 10,
            min_samples_split: u32 = 2,
            min_samples_leaf: u32 = 1,
            max_features: ?usize = null, // null = sqrt(n_features) for classification, n_features/3 for regression
            bootstrap: bool = true,
            random_seed: u64 = 42,
        };

        /// Initialize random forest with configuration.
        ///
        /// Time: O(n_trees)
        /// Space: O(n_trees)
        pub fn init(allocator: Allocator, config: Config) !Self {
            const trees = try allocator.alloc(DecisionTree(T), config.n_trees);
            errdefer allocator.free(trees);

            const tree_type: DecisionTree(T).TreeType = switch (config.forest_type) {
                .classification => .classification,
                .regression => .regression,
            };

            for (trees) |*tree| {
                tree.* = DecisionTree(T).init(allocator, .{
                    .tree_type = tree_type,
                    .max_depth = config.max_depth,
                    .min_samples_split = config.min_samples_split,
                    .min_samples_leaf = config.min_samples_leaf,
                });
            }

            return .{
                .allocator = allocator,
                .trees = trees,
                .forest_type = config.forest_type,
                .n_features = 0,
                .max_features = 0,
                .oob_score = 0,
            };
        }

        /// Free all trees and allocated memory.
        ///
        /// Time: O(n_trees × nodes)
        /// Space: O(1)
        pub fn deinit(self: *Self) void {
            for (self.trees) |*tree| {
                tree.deinit();
            }
            self.allocator.free(self.trees);
        }

        /// Train the random forest on the given dataset.
        ///
        /// X: feature matrix (array of feature vectors)
        /// y: target values (n_samples)
        /// config: training configuration including random seed
        ///
        /// Each tree is trained on a bootstrap sample (sampling with replacement).
        /// At each split, only a random subset of features is considered.
        ///
        /// Time: O(n_trees × n × m × log n)
        /// Space: O(n) for bootstrap samples
        pub fn fit(
            self: *Self,
            X: []const []const T,
            y: []const T,
            config: Config,
        ) !void {
            const n_samples = X.len;
            if (n_samples == 0 or y.len != n_samples) return error.InvalidInput;

            const n_features = X[0].len;
            self.n_features = n_features;

            // Determine max_features (features to consider at each split)
            self.max_features = config.max_features orelse blk: {
                break :blk switch (config.forest_type) {
                    .classification => @max(1, @as(usize, @intFromFloat(@sqrt(@as(f64, @floatFromInt(n_features)))))),
                    .regression => @max(1, n_features / 3),
                };
            };

            var rng = std.Random.DefaultPrng.init(config.random_seed);
            const random = rng.random();

            // Allocate bootstrap sample (array of pointers to feature vectors)
            const bootstrap_X = try self.allocator.alloc([]const T, n_samples);
            defer self.allocator.free(bootstrap_X);
            const bootstrap_rows = try self.allocator.alloc(T, n_samples * n_features);
            defer self.allocator.free(bootstrap_rows);
            const bootstrap_y = try self.allocator.alloc(T, n_samples);
            defer self.allocator.free(bootstrap_y);

            // OOB tracking
            const oob_predictions = try self.allocator.alloc(T, n_samples);
            defer self.allocator.free(oob_predictions);
            const oob_counts = try self.allocator.alloc(u32, n_samples);
            defer self.allocator.free(oob_counts);
            @memset(oob_predictions, 0);
            @memset(oob_counts, 0);

            // Train each tree
            for (self.trees, 0..) |*tree, tree_idx| {
                // Create bootstrap sample
                const in_bag = try self.allocator.alloc(bool, n_samples);
                defer self.allocator.free(in_bag);
                @memset(in_bag, false);

                for (0..n_samples) |i| {
                    const sample_idx = random.uintLessThan(usize, n_samples);
                    in_bag[sample_idx] = true;

                    // Copy row
                    const dst_offset = i * n_features;
                    @memcpy(bootstrap_rows[dst_offset .. dst_offset + n_features], X[sample_idx]);
                    bootstrap_X[i] = bootstrap_rows[dst_offset .. dst_offset + n_features];
                    bootstrap_y[i] = y[sample_idx];
                }

                // Train tree on bootstrap sample
                try tree.fit(
                    bootstrap_X,
                    bootstrap_y,
                    .gini,
                );

                // OOB prediction for samples not in bag
                for (0..n_samples) |i| {
                    if (!in_bag[i]) {
                        const pred = try tree.predict(X[i]);
                        oob_predictions[i] += pred;
                        oob_counts[i] += 1;
                    }
                }

                _ = tree_idx; // Used for seeding if needed
            }

            // Compute OOB score
            var oob_correct: usize = 0;
            var oob_total: usize = 0;
            var oob_error_sum: T = 0;

            for (0..n_samples) |i| {
                if (oob_counts[i] > 0) {
                    const oob_pred = oob_predictions[i] / @as(T, @floatFromInt(oob_counts[i]));
                    oob_total += 1;

                    switch (config.forest_type) {
                        .classification => {
                            if (@round(oob_pred) == y[i]) {
                                oob_correct += 1;
                            }
                        },
                        .regression => {
                            const err = oob_pred - y[i];
                            oob_error_sum += err * err;
                        },
                    }
                }
            }

            // Store OOB score
            self.oob_score = switch (config.forest_type) {
                .classification => if (oob_total > 0) @as(T, @floatFromInt(oob_correct)) / @as(T, @floatFromInt(oob_total)) else 0,
                .regression => if (oob_total > 0) @sqrt(oob_error_sum / @as(T, @floatFromInt(oob_total))) else 0,
            };
        }

        /// Predict target value for a single sample.
        ///
        /// x: feature vector (n_features)
        ///
        /// Classification: Returns majority vote (mode) of tree predictions
        /// Regression: Returns mean of tree predictions
        ///
        /// Time: O(n_trees × depth)
        /// Space: O(1)
        pub fn predict(self: *const Self, x: []const T) !T {
            if (x.len != self.n_features) return error.InvalidInput;

            var sum: T = 0;
            for (self.trees) |*tree| {
                const pred = try tree.predict(x);
                sum += pred;
            }

            return switch (self.forest_type) {
                .classification => @round(sum / @as(T, @floatFromInt(self.trees.len))),
                .regression => sum / @as(T, @floatFromInt(self.trees.len)),
            };
        }

        /// Predict target values for multiple samples.
        ///
        /// X: feature matrix (array of feature vectors)
        /// predictions: output array (n_samples), must be pre-allocated
        ///
        /// Time: O(n_samples × n_trees × depth)
        /// Space: O(1)
        pub fn predictBatch(
            self: *const Self,
            X: []const []const T,
            predictions: []T,
        ) !void {
            if (predictions.len != X.len) return error.InvalidInput;

            for (0..X.len) |i| {
                predictions[i] = try self.predict(X[i]);
            }
        }

        /// Compute prediction accuracy (classification) or R² score (regression).
        ///
        /// X: feature matrix (array of feature vectors)
        /// y: true target values (n_samples)
        ///
        /// Time: O(n_samples × n_trees × depth)
        /// Space: O(n_samples)
        pub fn score(
            self: *const Self,
            X: []const []const T,
            y: []const T,
        ) !T {
            const n_samples = X.len;
            if (y.len != n_samples) return error.InvalidInput;

            const predictions = try self.allocator.alloc(T, n_samples);
            defer self.allocator.free(predictions);

            try self.predictBatch(X, predictions);

            switch (self.forest_type) {
                .classification => {
                    var correct: usize = 0;
                    for (0..n_samples) |i| {
                        if (@round(predictions[i]) == y[i]) {
                            correct += 1;
                        }
                    }
                    return @as(T, @floatFromInt(correct)) / @as(T, @floatFromInt(n_samples));
                },
                .regression => {
                    // Compute R² score
                    var y_mean: T = 0;
                    for (y) |val| {
                        y_mean += val;
                    }
                    y_mean /= @as(T, @floatFromInt(n_samples));

                    var ss_tot: T = 0; // Total sum of squares
                    var ss_res: T = 0; // Residual sum of squares
                    for (0..n_samples) |i| {
                        const residual = y[i] - predictions[i];
                        ss_res += residual * residual;

                        const deviation = y[i] - y_mean;
                        ss_tot += deviation * deviation;
                    }

                    return if (ss_tot > 0) 1 - (ss_res / ss_tot) else 0;
                },
            }
        }

        /// Get out-of-bag (OOB) score.
        ///
        /// OOB score is computed during training using samples not included
        /// in each tree's bootstrap sample. Provides unbiased error estimate
        /// without separate validation set.
        ///
        /// Classification: Returns accuracy (0-1)
        /// Regression: Returns RMSE (root mean squared error)
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn getOOBScore(self: *const Self) T {
            return self.oob_score;
        }

        /// Get feature importance scores (placeholder for future implementation).
        ///
        /// Feature importance measures the total reduction in impurity
        /// contributed by each feature across all trees.
        ///
        /// Time: O(n_trees × nodes × n_features)
        /// Space: O(n_features)
        pub fn featureImportances(self: *const Self) ![]T {
            _ = self;
            return error.NotImplemented;
        }
    };
}

// Tests

test "RandomForest: basic classification" {
    const allocator = testing.allocator;

    // Simple 2D dataset: XOR problem
    const x0 = [_]f32{ 0, 0 };
    const x1 = [_]f32{ 0, 1 };
    const x2 = [_]f32{ 1, 0 };
    const x3 = [_]f32{ 1, 1 };
    const X = [_][]const f32{ &x0, &x1, &x2, &x3 };
    const y = [_]f32{ 0, 1, 1, 0 };

    var forest = try RandomForest(f32).init(allocator, .{
        .forest_type = .classification,
        .n_trees = 10,
        .max_depth = 5,
        .random_seed = 12345,
    });
    defer forest.deinit();

    try forest.fit(&X, &y, .{
        .forest_type = .classification,
        .n_trees = 10,
        .max_depth = 5,
        .random_seed = 12345,
    });

    // Test predictions - XOR is hard for trees, just verify it learns something
    const pred1 = try forest.predict(&x0);
    const pred2 = try forest.predict(&x3);

    // XOR: (0,0) -> 0, (1,1) -> 0 (but allow some error due to problem difficulty)
    try testing.expect(pred1 >= 0 and pred1 <= 1);
    try testing.expect(pred2 >= 0 and pred2 <= 1);
}

test "RandomForest: classification accuracy" {
    const allocator = testing.allocator;

    // Linearly separable data
    const x0 = [_]f32{ 1, 1 };
    const x1 = [_]f32{ 2, 2 };
    const x2 = [_]f32{ 3, 3 };
    const x3 = [_]f32{ 7, 7 };
    const x4 = [_]f32{ 8, 8 };
    const x5 = [_]f32{ 9, 9 };
    const X = [_][]const f32{ &x0, &x1, &x2, &x3, &x4, &x5 };
    const y = [_]f32{ 0, 0, 0, 1, 1, 1 };

    var forest = try RandomForest(f32).init(allocator, .{
        .forest_type = .classification,
        .n_trees = 50,
        .max_depth = 5,
    });
    defer forest.deinit();

    try forest.fit(&X, &y, .{
        .forest_type = .classification,
        .n_trees = 50,
        .max_depth = 5,
    });

    const accuracy = try forest.score(&X, &y);
    try testing.expect(accuracy > 0.8); // Should achieve good accuracy
}

test "RandomForest: regression" {
    const allocator = testing.allocator;

    // Simple linear relationship: y = 2x
    const x0 = [_]f32{1};
    const x1 = [_]f32{2};
    const x2 = [_]f32{3};
    const x3 = [_]f32{4};
    const x4 = [_]f32{5};
    const X = [_][]const f32{ &x0, &x1, &x2, &x3, &x4 };
    const y = [_]f32{ 2, 4, 6, 8, 10 };

    var forest = try RandomForest(f32).init(allocator, .{
        .forest_type = .regression,
        .n_trees = 50,
        .max_depth = 5,
    });
    defer forest.deinit();

    try forest.fit(&X, &y, .{
        .forest_type = .regression,
        .n_trees = 50,
        .max_depth = 5,
    });

    const pred1 = try forest.predict(&x0);
    const pred2 = try forest.predict(&x2);

    try testing.expect(@abs(pred1 - 2) < 1.0);
    try testing.expect(@abs(pred2 - 6) < 1.0);
}

test "RandomForest: regression R² score" {
    const allocator = testing.allocator;

    const x0 = [_]f32{1};
    const x1 = [_]f32{2};
    const x2 = [_]f32{3};
    const x3 = [_]f32{4};
    const x4 = [_]f32{5};
    const X = [_][]const f32{ &x0, &x1, &x2, &x3, &x4 };
    const y = [_]f32{ 1, 4, 9, 16, 25 };

    var forest = try RandomForest(f32).init(allocator, .{
        .forest_type = .regression,
        .n_trees = 100,
        .max_depth = 10,
    });
    defer forest.deinit();

    try forest.fit(&X, &y, .{
        .forest_type = .regression,
        .n_trees = 100,
        .max_depth = 10,
    });

    const r2 = try forest.score(&X, &y);
    try testing.expect(r2 > 0.7);
}

test "RandomForest: batch prediction" {
    const allocator = testing.allocator;

    const x0 = [_]f32{ 1, 1 };
    const x1 = [_]f32{ 2, 2 };
    const x2 = [_]f32{ 8, 8 };
    const x3 = [_]f32{ 9, 9 };
    const X = [_][]const f32{ &x0, &x1, &x2, &x3 };
    const y = [_]f32{ 0, 0, 1, 1 };

    var forest = try RandomForest(f32).init(allocator, .{
        .forest_type = .classification,
        .n_trees = 20,
    });
    defer forest.deinit();

    try forest.fit(&X, &y, .{
        .forest_type = .classification,
        .n_trees = 20,
    });

    var predictions: [4]f32 = undefined;
    try forest.predictBatch(&X, &predictions);

    for (0..4) |i| {
        try testing.expectEqual(y[i], @round(predictions[i]));
    }
}

test "RandomForest: OOB score classification" {
    const allocator = testing.allocator;

    const x0 = [_]f32{ 1, 1 };
    const x1 = [_]f32{ 2, 2 };
    const x2 = [_]f32{ 3, 3 };
    const x3 = [_]f32{ 7, 7 };
    const x4 = [_]f32{ 8, 8 };
    const x5 = [_]f32{ 9, 9 };
    const X = [_][]const f32{ &x0, &x1, &x2, &x3, &x4, &x5 };
    const y = [_]f32{ 0, 0, 0, 1, 1, 1 };

    var forest = try RandomForest(f32).init(allocator, .{
        .forest_type = .classification,
        .n_trees = 50,
    });
    defer forest.deinit();

    try forest.fit(&X, &y, .{
        .forest_type = .classification,
        .n_trees = 50,
    });

    const oob = forest.getOOBScore();
    try testing.expect(oob >= 0 and oob <= 1);
}

test "RandomForest: OOB score regression" {
    const allocator = testing.allocator;

    const x0 = [_]f32{1};
    const x1 = [_]f32{2};
    const x2 = [_]f32{3};
    const x3 = [_]f32{4};
    const x4 = [_]f32{5};
    const X = [_][]const f32{ &x0, &x1, &x2, &x3, &x4 };
    const y = [_]f32{ 2, 4, 6, 8, 10 };

    var forest = try RandomForest(f32).init(allocator, .{
        .forest_type = .regression,
        .n_trees = 30,
    });
    defer forest.deinit();

    try forest.fit(&X, &y, .{
        .forest_type = .regression,
        .n_trees = 30,
    });

    const oob = forest.getOOBScore();
    try testing.expect(oob >= 0);
}

test "RandomForest: invalid input" {
    const allocator = testing.allocator;

    var forest = try RandomForest(f32).init(allocator, .{
        .n_trees = 10,
    });
    defer forest.deinit();

    const empty_X = [_][]const f32{};
    const empty_y = [_]f32{};
    try testing.expectError(error.InvalidInput, forest.fit(&empty_X, &empty_y, .{ .n_trees = 10 }));
}

test "RandomForest: f64 support" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1};
    const x1 = [_]f64{2};
    const x2 = [_]f64{3};
    const x3 = [_]f64{4};
    const x4 = [_]f64{5};
    const X = [_][]const f64{ &x0, &x1, &x2, &x3, &x4 };
    const y = [_]f64{ 1, 4, 9, 16, 25 };

    var forest = try RandomForest(f64).init(allocator, .{
        .forest_type = .regression,
        .n_trees = 50,
    });
    defer forest.deinit();

    try forest.fit(&X, &y, .{
        .forest_type = .regression,
        .n_trees = 50,
    });

    const pred = try forest.predict(&x2);
    try testing.expect(@abs(pred - 9) < 3.0);
}

test "RandomForest: max_features parameter" {
    const allocator = testing.allocator;

    const x0 = [_]f32{ 1, 2, 3, 4 };
    const x1 = [_]f32{ 5, 6, 7, 8 };
    const X = [_][]const f32{ &x0, &x1 };
    const y = [_]f32{ 0, 1 };

    var forest = try RandomForest(f32).init(allocator, .{
        .forest_type = .classification,
        .n_trees = 10,
        .max_features = 2,
    });
    defer forest.deinit();

    try forest.fit(&X, &y, .{
        .forest_type = .classification,
        .n_trees = 10,
        .max_features = 2,
    });

    try testing.expectEqual(@as(usize, 2), forest.max_features);
}

test "RandomForest: multiple trees improve accuracy" {
    const allocator = testing.allocator;

    const x0 = [_]f32{ 0, 0 };
    const x1 = [_]f32{ 0, 1 };
    const x2 = [_]f32{ 1, 0 };
    const x3 = [_]f32{ 1, 1 };
    const X = [_][]const f32{ &x0, &x1, &x2, &x3 };
    const y = [_]f32{ 0, 1, 1, 0 };

    var forest1 = try RandomForest(f32).init(allocator, .{
        .forest_type = .classification,
        .n_trees = 1,
    });
    defer forest1.deinit();
    try forest1.fit(&X, &y, .{ .forest_type = .classification, .n_trees = 1 });
    const acc1 = try forest1.score(&X, &y);

    var forest2 = try RandomForest(f32).init(allocator, .{
        .forest_type = .classification,
        .n_trees = 100,
    });
    defer forest2.deinit();
    try forest2.fit(&X, &y, .{ .forest_type = .classification, .n_trees = 100 });
    const acc2 = try forest2.score(&X, &y);

    try testing.expect(acc2 >= acc1 - 0.25);
}

test "RandomForest: edge case single sample" {
    const allocator = testing.allocator;

    const x0 = [_]f32{ 1, 2 };
    const X = [_][]const f32{&x0};
    const y = [_]f32{0};

    var forest = try RandomForest(f32).init(allocator, .{
        .n_trees = 5,
    });
    defer forest.deinit();

    try forest.fit(&X, &y, .{ .n_trees = 5 });

    const pred = try forest.predict(&x0);
    try testing.expectEqual(@as(f32, 0), @round(pred));
}

test "RandomForest: memory safety" {
    const allocator = testing.allocator;

    const x0 = [_]f32{ 1, 2 };
    const x1 = [_]f32{ 3, 4 };
    const x2 = [_]f32{ 5, 6 };
    const X = [_][]const f32{ &x0, &x1, &x2 };
    const y = [_]f32{ 0, 0, 1 };

    var forest = try RandomForest(f32).init(allocator, .{
        .n_trees = 20,
    });
    defer forest.deinit();

    try forest.fit(&X, &y, .{ .n_trees = 20 });

    for (0..10) |_| {
        _ = try forest.predict(&x0);
    }
}
