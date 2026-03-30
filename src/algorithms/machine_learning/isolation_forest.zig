const std = @import("std");
const Allocator = std.mem.Allocator;
const Random = std.Random;
const testing = std.testing;

/// Isolation Forest for anomaly detection
///
/// Isolation Forest is an unsupervised anomaly detection algorithm that uses
/// ensemble of isolation trees to identify outliers. The key insight is that
/// anomalies are easier to isolate (require fewer splits) than normal points.
///
/// Algorithm:
/// 1. Training: Build n_trees isolation trees on random subsamples
/// 2. For each tree: recursively split on random attribute and threshold
/// 3. Prediction: Compute average path length for each sample
/// 4. Anomaly score: 2^(-E[h(x)] / c(ψ)) where E[h(x)] is avg path length
///
/// Time complexity:
/// - Training: O(n_trees × ψ × log ψ) where ψ is subsample size
/// - Prediction: O(n_trees × log ψ) per sample
///
/// Space complexity: O(n_trees × ψ) for tree structures
///
/// Use cases:
/// - Fraud detection (credit cards, insurance claims)
/// - Network intrusion detection
/// - Quality control (manufacturing defects)
/// - Medical diagnosis (rare diseases)
/// - Sensor data monitoring (equipment failures)
///
/// Example:
/// ```zig
/// var forest = IsolationForest(f64).init(allocator, 100, 256, 42);
/// defer forest.deinit();
/// try forest.fit(training_data);
/// const scores = try forest.predict(test_data);
/// defer allocator.free(scores);
/// // scores > 0.5 are anomalies
/// ```
pub fn IsolationForest(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        trees: []IsolationTree(T),
        n_trees: usize,
        n_fitted_trees: usize, // number of trees actually built (for safe deinit)
        subsample_size: usize,
        max_depth: usize,
        rng: Random,

        /// Initialize Isolation Forest
        ///
        /// Time: O(1) | Space: O(n_trees)
        pub fn init(allocator: Allocator, n_trees: usize, subsample_size: usize, seed: u64) !Self {
            if (n_trees == 0) return error.InvalidTreeCount;
            if (subsample_size == 0) return error.InvalidSubsampleSize;

            const trees = try allocator.alloc(IsolationTree(T), n_trees);
            errdefer allocator.free(trees);

            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();

            // Compute max depth from subsample size (log2(ψ))
            var max_depth: usize = 0;
            var size = subsample_size;
            while (size > 1) : (size /= 2) {
                max_depth += 1;
            }

            return Self{
                .allocator = allocator,
                .trees = trees,
                .n_trees = n_trees,
                .n_fitted_trees = 0, // no trees built yet
                .subsample_size = subsample_size,
                .max_depth = max_depth,
                .rng = rng,
            };
        }

        /// Free all resources
        ///
        /// Time: O(n_trees × tree_size) | Space: O(1)
        pub fn deinit(self: *Self) void {
            // Only deinit trees that were actually built
            for (self.trees[0..self.n_fitted_trees]) |*tree| {
                tree.deinit(self.allocator);
            }
            self.allocator.free(self.trees);
        }

        /// Train on dataset
        ///
        /// Time: O(n_trees × ψ × log ψ) | Space: O(n_trees × ψ)
        pub fn fit(self: *Self, X: []const []const T) !void {
            if (X.len == 0) return error.EmptyDataset;
            if (X[0].len == 0) return error.ZeroFeatures;

            const n_samples = X.len;
            const n_features = X[0].len;

            // Build each isolation tree
            for (self.trees, 0..) |*tree, i| {
                // Sample random subset
                const sample_size = @min(self.subsample_size, n_samples);
                const indices = try self.randomSample(sample_size, n_samples);
                defer self.allocator.free(indices);

                // Extract subsample
                const subsample = try self.allocator.alloc([]const T, sample_size);
                defer self.allocator.free(subsample);
                for (indices, 0..) |idx, j| {
                    subsample[j] = X[idx];
                }

                // Build tree
                tree.* = try IsolationTree(T).build(
                    self.allocator,
                    subsample,
                    n_features,
                    0,
                    self.max_depth,
                    &self.rng,
                );

                // Mark this tree as fitted
                self.n_fitted_trees = i + 1;
            }
        }

        /// Predict anomaly scores for dataset
        ///
        /// Returns array of scores in range [0, 1] where:
        /// - Score > 0.5: likely anomaly
        /// - Score ≈ 0.5: normal
        /// - Score < 0.5: definitely normal
        ///
        /// Time: O(n × n_trees × log ψ) | Space: O(n)
        pub fn predict(self: *Self, X: []const []const T) ![]T {
            if (X.len == 0) return error.EmptyDataset;

            const n_samples = X.len;
            const scores = try self.allocator.alloc(T, n_samples);
            errdefer self.allocator.free(scores);

            // Compute average path length normalization factor c(n)
            const c_n = averagePathLength(T, self.subsample_size);

            // Compute anomaly score for each sample
            for (X, 0..) |x, i| {
                var sum_path_length: T = 0;

                // Average path length across all fitted trees
                for (self.trees[0..self.n_fitted_trees]) |*tree| {
                    const path_len = tree.pathLength(x, 0);
                    sum_path_length += path_len;
                }

                const avg_path_length = sum_path_length / @as(T, @floatFromInt(self.n_fitted_trees));

                // Anomaly score: 2^(-E[h(x)] / c(n))
                // Special case: if c_n is 0 (single sample), return neutral score
                if (c_n == 0) {
                    scores[i] = 0.5;
                } else {
                    scores[i] = std.math.pow(T, 2.0, -avg_path_length / c_n);
                }
            }

            return scores;
        }

        /// Predict anomaly score for single sample
        ///
        /// Time: O(n_trees × log ψ) | Space: O(1)
        pub fn predictOne(self: *Self, x: []const T) T {
            const c_n = averagePathLength(T, self.subsample_size);
            var sum_path_length: T = 0;

            for (self.trees[0..self.n_fitted_trees]) |*tree| {
                const path_len = tree.pathLength(x, 0);
                sum_path_length += path_len;
            }

            const avg_path_length = sum_path_length / @as(T, @floatFromInt(self.n_fitted_trees));

            // Special case: if c_n is 0 (single sample), return neutral score
            if (c_n == 0) {
                return 0.5;
            }
            return std.math.pow(T, 2.0, -avg_path_length / c_n);
        }

        /// Generate random sample of indices without replacement
        fn randomSample(self: *Self, k: usize, n: usize) ![]usize {
            const indices = try self.allocator.alloc(usize, k);
            errdefer self.allocator.free(indices);

            // Fisher-Yates shuffle of first k elements
            var pool = try self.allocator.alloc(usize, n);
            defer self.allocator.free(pool);
            for (pool, 0..) |*p, i| {
                p.* = i;
            }

            for (0..k) |i| {
                const j = self.rng.intRangeAtMost(usize, i, n - 1);
                const temp = pool[i];
                pool[i] = pool[j];
                pool[j] = temp;
                indices[i] = pool[i];
            }

            return indices;
        }
    };
}

/// Isolation Tree node
fn IsolationTree(comptime T: type) type {
    return struct {
        const Self = @This();

        split_attr: ?usize, // null for leaf
        split_value: T,
        left: ?*Self,
        right: ?*Self,
        size: usize, // number of samples at this node

        /// Build isolation tree recursively
        ///
        /// Time: O(n log n) | Space: O(log n) for recursion
        fn build(
            allocator: Allocator,
            X: []const []const T,
            n_features: usize,
            depth: usize,
            max_depth: usize,
            rng: *Random,
        ) !Self {
            const n_samples = X.len;

            // Leaf condition: max depth reached or single sample
            if (depth >= max_depth or n_samples <= 1) {
                return Self{
                    .split_attr = null,
                    .split_value = 0,
                    .left = null,
                    .right = null,
                    .size = n_samples,
                };
            }

            // Select random attribute
            const attr = rng.intRangeAtMost(usize, 0, n_features - 1);

            // Find min/max values for this attribute
            var min_val = X[0][attr];
            var max_val = X[0][attr];
            for (X[1..]) |x| {
                min_val = @min(min_val, x[attr]);
                max_val = @max(max_val, x[attr]);
            }

            // No split possible if all values are the same
            if (min_val >= max_val) {
                return Self{
                    .split_attr = null,
                    .split_value = 0,
                    .left = null,
                    .right = null,
                    .size = n_samples,
                };
            }

            // Random split value in range [min, max]
            const split_value = if (T == f32 or T == f64)
                min_val + rng.float(T) * (max_val - min_val)
            else
                @as(T, @floatFromInt(rng.intRangeAtMost(i64, @intFromFloat(min_val), @intFromFloat(max_val))));

            // Partition samples
            const left_samples = try allocator.alloc([]const T, n_samples);
            defer allocator.free(left_samples);
            const right_samples = try allocator.alloc([]const T, n_samples);
            defer allocator.free(right_samples);

            var left_count: usize = 0;
            var right_count: usize = 0;

            for (X) |x| {
                if (x[attr] < split_value) {
                    left_samples[left_count] = x;
                    left_count += 1;
                } else {
                    right_samples[right_count] = x;
                    right_count += 1;
                }
            }

            // Create child nodes
            var node = Self{
                .split_attr = attr,
                .split_value = split_value,
                .left = null,
                .right = null,
                .size = n_samples,
            };

            if (left_count > 0) {
                const left_node = try allocator.create(Self);
                errdefer allocator.destroy(left_node);
                left_node.* = try build(
                    allocator,
                    left_samples[0..left_count],
                    n_features,
                    depth + 1,
                    max_depth,
                    rng,
                );
                node.left = left_node;
            }

            if (right_count > 0) {
                const right_node = try allocator.create(Self);
                errdefer allocator.destroy(right_node);
                right_node.* = try build(
                    allocator,
                    right_samples[0..right_count],
                    n_features,
                    depth + 1,
                    max_depth,
                    rng,
                );
                node.right = right_node;
            }

            return node;
        }

        /// Free tree recursively
        fn deinit(self: *Self, allocator: Allocator) void {
            if (self.left) |left| {
                left.deinit(allocator);
                allocator.destroy(left);
            }
            if (self.right) |right| {
                right.deinit(allocator);
                allocator.destroy(right);
            }
        }

        /// Compute path length for a sample
        ///
        /// Time: O(log n) average | Space: O(1)
        fn pathLength(self: *const Self, x: []const T, current_depth: usize) T {
            // Leaf node: add average path length for remaining samples
            if (self.split_attr == null) {
                return @as(T, @floatFromInt(current_depth)) + averagePathLength(T, self.size);
            }

            const attr = self.split_attr.?;
            if (x[attr] < self.split_value) {
                if (self.left) |left| {
                    return left.pathLength(x, current_depth + 1);
                }
            } else {
                if (self.right) |right| {
                    return right.pathLength(x, current_depth + 1);
                }
            }

            // Fallback if child is null
            return @as(T, @floatFromInt(current_depth)) + averagePathLength(T, self.size);
        }
    };
}

/// Compute average path length c(n) for normalization
///
/// c(n) = 2H(n-1) - 2(n-1)/n where H(n) is harmonic number
/// This is the average path length of unsuccessful search in BST
///
/// Time: O(1) | Space: O(1)
fn averagePathLength(comptime T: type, n: usize) T {
    if (n <= 1) return 0;
    if (n == 2) return 1;

    const n_f = @as(T, @floatFromInt(n));
    const h_n = harmonicNumber(T, n - 1);
    return 2.0 * h_n - 2.0 * (n_f - 1.0) / n_f;
}

/// Compute harmonic number H(n) = 1 + 1/2 + 1/3 + ... + 1/n
///
/// Approximation: H(n) ≈ ln(n) + γ where γ ≈ 0.5772 (Euler-Mascheroni constant)
///
/// Time: O(1) | Space: O(1)
fn harmonicNumber(comptime T: type, n: usize) T {
    if (n == 0) return 0;
    const euler_gamma: T = 0.5772156649;
    return @log(@as(T, @floatFromInt(n))) + euler_gamma;
}

// ============================================================================
// Tests
// ============================================================================

test "IsolationForest: basic anomaly detection" {
    const allocator = testing.allocator;

    // Create simple 2D dataset with outliers
    // Normal points: clustered around (0, 0)
    // Outliers: far from cluster at (10, 10) and (-10, -10)
    var data = [_][]const f64{
        &[_]f64{ 0.1, 0.2 },
        &[_]f64{ -0.1, 0.1 },
        &[_]f64{ 0.2, -0.1 },
        &[_]f64{ -0.2, -0.2 },
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 10.0, 10.0 }, // outlier
        &[_]f64{ -10.0, -10.0 }, // outlier
    };

    // Use subsample_size = 4, smaller than dataset (7)
    var forest = try IsolationForest(f64).init(allocator, 100, 4, 42);
    defer forest.deinit();

    try forest.fit(&data);

    const scores = try forest.predict(&data);
    defer allocator.free(scores);

    // Outliers should have higher scores than normal points
    const avg_normal = (scores[0] + scores[1] + scores[2] + scores[3] + scores[4]) / 5.0;
    const avg_outlier = (scores[5] + scores[6]) / 2.0;
    try testing.expect(avg_outlier > avg_normal);
}

test "IsolationForest: predictOne" {
    const allocator = testing.allocator;

    // More samples for better discrimination
    var data = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 0.1, 0.1 },
        &[_]f64{ -0.1, -0.1 },
        &[_]f64{ 0.05, -0.05 },
        &[_]f64{ -0.05, 0.05 },
        &[_]f64{ 0.15, -0.15 },
    };

    var forest = try IsolationForest(f64).init(allocator, 100, 4, 42);
    defer forest.deinit();

    try forest.fit(&data);

    const normal_point = [_]f64{ 0.02, 0.02 };
    const outlier_point = [_]f64{ 10.0, 10.0 };

    const normal_score = forest.predictOne(&normal_point);
    const outlier_score = forest.predictOne(&outlier_point);

    // Both scores should be valid
    try testing.expect(normal_score >= 0.0 and normal_score <= 1.0);
    try testing.expect(outlier_score >= 0.0 and outlier_score <= 1.0);
    try testing.expect(!std.math.isNan(normal_score));
    try testing.expect(!std.math.isNan(outlier_score));
}

test "IsolationForest: reproducibility with same seed" {
    const allocator = testing.allocator;

    var data = [_][]const f64{
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 2.0, 3.0 },
        &[_]f64{ 3.0, 4.0 },
        &[_]f64{ 10.0, 10.0 },
    };

    // Train two forests with same seed
    var forest1 = try IsolationForest(f64).init(allocator, 50, 64, 12345);
    defer forest1.deinit();
    try forest1.fit(&data);

    var forest2 = try IsolationForest(f64).init(allocator, 50, 64, 12345);
    defer forest2.deinit();
    try forest2.fit(&data);

    const scores1 = try forest1.predict(&data);
    defer allocator.free(scores1);
    const scores2 = try forest2.predict(&data);
    defer allocator.free(scores2);

    // Results should be identical
    for (scores1, scores2) |s1, s2| {
        try testing.expectApproxEqAbs(s1, s2, 1e-10);
    }
}

test "IsolationForest: edge case - single sample" {
    const allocator = testing.allocator;

    var data = [_][]const f64{
        &[_]f64{ 1.0, 2.0 },
    };

    var forest = try IsolationForest(f64).init(allocator, 10, 1, 42);
    defer forest.deinit();

    try forest.fit(&data);

    const scores = try forest.predict(&data);
    defer allocator.free(scores);

    // Single sample: score should be valid (0-1 range)
    try testing.expect(scores[0] >= 0.0 and scores[0] <= 1.0);
    try testing.expect(!std.math.isNan(scores[0]));
}

test "IsolationForest: edge case - all identical points" {
    const allocator = testing.allocator;

    var data = [_][]const f64{
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 1.0, 1.0 },
    };

    var forest = try IsolationForest(f64).init(allocator, 50, 3, 42);
    defer forest.deinit();

    try forest.fit(&data);

    const scores = try forest.predict(&data);
    defer allocator.free(scores);

    // All identical points should have similar scores (variance should be low)
    var sum: f64 = 0;
    for (scores) |score| {
        sum += score;
    }
    const mean = sum / @as(f64, @floatFromInt(scores.len));

    var variance: f64 = 0;
    for (scores) |score| {
        const diff = score - mean;
        variance += diff * diff;
    }
    variance /= @as(f64, @floatFromInt(scores.len));

    // Variance should be very small for identical points
    try testing.expect(variance < 0.01);
}

test "IsolationForest: stress test - large dataset" {
    const allocator = testing.allocator;

    // Generate 1000 normal points + 10 outliers
    const n_normal = 1000;
    const n_outliers = 10;
    const n_total = n_normal + n_outliers;

    const data = try allocator.alloc([]f64, n_total);
    defer {
        for (data) |row| {
            allocator.free(row);
        }
        allocator.free(data);
    }

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Normal points: Gaussian around (0, 0)
    for (0..n_normal) |i| {
        const row = try allocator.alloc(f64, 2);
        row[0] = rng.floatNorm(f64) * 0.5;
        row[1] = rng.floatNorm(f64) * 0.5;
        data[i] = row;
    }

    // Outliers: far from center
    for (n_normal..n_total) |i| {
        const row = try allocator.alloc(f64, 2);
        row[0] = 10.0 + rng.float(f64);
        row[1] = 10.0 + rng.float(f64);
        data[i] = row;
    }

    // Convert to const slices
    const const_data = try allocator.alloc([]const f64, n_total);
    defer allocator.free(const_data);
    for (data, 0..) |row, i| {
        const_data[i] = row;
    }

    var forest = try IsolationForest(f64).init(allocator, 100, 256, 42);
    defer forest.deinit();

    try forest.fit(const_data);

    const scores = try forest.predict(const_data);
    defer allocator.free(scores);

    // Count anomalies (score > 0.6 for clear outliers)
    var anomaly_count: usize = 0;
    for (scores) |score| {
        if (score > 0.6) {
            anomaly_count += 1;
        }
    }

    // Should detect some anomalies (at least 5 of the 10 outliers)
    try testing.expect(anomaly_count >= 5);
}

test "IsolationForest: f32 precision" {
    const allocator = testing.allocator;

    // More samples for f32
    var data = [_][]const f32{
        &[_]f32{ 0.0, 0.0 },
        &[_]f32{ 0.1, 0.1 },
        &[_]f32{ -0.1, -0.1 },
        &[_]f32{ 0.05, 0.05 },
        &[_]f32{ -0.05, -0.05 },
        &[_]f32{ 10.0, 10.0 }, // outlier
    };

    var forest = try IsolationForest(f32).init(allocator, 100, 4, 42);
    defer forest.deinit();

    try forest.fit(&data);

    const scores = try forest.predict(&data);
    defer allocator.free(scores);

    // All scores should be valid
    for (scores) |score| {
        try testing.expect(score >= 0.0 and score <= 1.0);
        try testing.expect(!std.math.isNan(score));
    }
}

test "IsolationForest: error - empty dataset" {
    const allocator = testing.allocator;

    var data = [_][]const f64{};

    var forest = try IsolationForest(f64).init(allocator, 10, 64, 42);
    defer forest.deinit();

    try testing.expectError(error.EmptyDataset, forest.fit(&data));
}

test "IsolationForest: error - zero features" {
    const allocator = testing.allocator;

    var data = [_][]const f64{
        &[_]f64{},
        &[_]f64{},
    };

    var forest = try IsolationForest(f64).init(allocator, 10, 64, 42);
    defer forest.deinit();

    try testing.expectError(error.ZeroFeatures, forest.fit(&data));
}

test "IsolationForest: error - invalid parameters" {
    const allocator = testing.allocator;

    try testing.expectError(error.InvalidTreeCount, IsolationForest(f64).init(allocator, 0, 64, 42));
    try testing.expectError(error.InvalidSubsampleSize, IsolationForest(f64).init(allocator, 10, 0, 42));
}

test "IsolationForest: high-dimensional data" {
    const allocator = testing.allocator;

    // 5D data with more samples
    var data = [_][]const f64{
        &[_]f64{ 0.0, 0.0, 0.0, 0.0, 0.0 },
        &[_]f64{ 0.1, 0.1, 0.1, 0.1, 0.1 },
        &[_]f64{ -0.1, -0.1, -0.1, -0.1, -0.1 },
        &[_]f64{ 0.05, -0.05, 0.05, -0.05, 0.05 },
        &[_]f64{ -0.05, 0.05, -0.05, 0.05, -0.05 },
        &[_]f64{ 10.0, 10.0, 10.0, 10.0, 10.0 }, // outlier
    };

    var forest = try IsolationForest(f64).init(allocator, 100, 4, 42);
    defer forest.deinit();

    try forest.fit(&data);

    const scores = try forest.predict(&data);
    defer allocator.free(scores);

    // All scores should be valid
    for (scores) |score| {
        try testing.expect(score >= 0.0 and score <= 1.0);
        try testing.expect(!std.math.isNan(score));
    }
}

test "IsolationForest: subsample smaller than dataset" {
    const allocator = testing.allocator;

    var data = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 0.1, 0.1 },
        &[_]f64{ 0.2, 0.2 },
        &[_]f64{ -0.1, -0.1 },
        &[_]f64{ -0.2, -0.2 },
        &[_]f64{ 0.05, 0.05 },
        &[_]f64{ 10.0, 10.0 }, // outlier
    };

    // Subsample size = 4, smaller than dataset size = 7
    var forest = try IsolationForest(f64).init(allocator, 100, 4, 42);
    defer forest.deinit();

    try forest.fit(&data);

    const scores = try forest.predict(&data);
    defer allocator.free(scores);

    // All scores should be valid
    for (scores) |score| {
        try testing.expect(score >= 0.0 and score <= 1.0);
        try testing.expect(!std.math.isNan(score));
    }
}

test "IsolationForest: memory leak check" {
    const allocator = testing.allocator;

    var data = [_][]const f64{
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 2.0, 3.0 },
        &[_]f64{ 3.0, 4.0 },
    };

    var forest = try IsolationForest(f64).init(allocator, 10, 10, 42);
    defer forest.deinit();

    try forest.fit(&data);

    const scores = try forest.predict(&data);
    defer allocator.free(scores);

    // If we reach here, no memory leaks detected by testing.allocator
}
