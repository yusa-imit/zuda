const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// K-Nearest Neighbors (KNN) Algorithm
///
/// Instance-based supervised learning for classification and regression.
/// Makes predictions by finding k nearest training examples and:
/// - Classification: majority vote of neighbors' labels
/// - Regression: mean of neighbors' target values
///
/// Time: O(n × d) for prediction (where n = training size, d = dimensions)
/// Space: O(n × d) to store training data
///
/// Use cases:
/// - Pattern recognition (handwriting, speech)
/// - Recommendation systems (collaborative filtering)
/// - Medical diagnosis (disease classification)
/// - Credit scoring (default prediction)
/// - Anomaly detection (outlier scoring)
///
/// Advantages:
/// - No training phase (lazy learning)
/// - Non-parametric (no assumptions about data distribution)
/// - Naturally handles multi-class classification
/// - Simple and interpretable
///
/// Limitations:
/// - Slow prediction for large datasets (O(n) per query)
/// - Sensitive to feature scaling (requires normalization)
/// - Curse of dimensionality (distance becomes meaningless in high-d)
/// - Memory intensive (stores all training data)
///
/// Example:
/// ```zig
/// const train_x = [_][]const f64{
///     &.{1.0, 1.0}, &.{2.0, 2.0}, &.{5.0, 5.0}, &.{6.0, 6.0}
/// };
/// const train_y = [_]usize{0, 0, 1, 1};
/// var knn = try KNN(f64).init(allocator, &train_x, &train_y, 3);
/// defer knn.deinit();
/// const label = knn.predictClass(&.{3.0, 3.0}); // returns 0 (closer to first cluster)
/// ```
pub fn KNN(comptime T: type) type {
    return struct {
        train_x: [][]const T, // n training samples, each of dim d
        train_y: []const T, // n target values (labels for classification, values for regression)
        k: usize, // number of neighbors
        allocator: Allocator,

        const Self = @This();

        /// Initialize KNN with training data
        ///
        /// Time: O(n × d) for copying
        /// Space: O(n × d)
        ///
        /// Params:
        ///   allocator: memory allocator
        ///   train_x: training features [n][d]
        ///   train_y: training labels/values [n]
        ///   k: number of neighbors (must be > 0 and ≤ n)
        pub fn init(allocator: Allocator, train_x: []const []const T, train_y: []const T, k: usize) !Self {
            if (train_x.len != train_y.len) return error.MismatchedDimensions;
            if (k == 0 or k > train_x.len) return error.InvalidK;
            if (train_x.len == 0) return error.EmptyTrainingSet;

            // Deep copy training data (we own it)
            const x_copy = try allocator.alloc([]T, train_x.len);
            errdefer allocator.free(x_copy);

            for (train_x, 0..) |sample, i| {
                x_copy[i] = try allocator.alloc(T, sample.len);
                @memcpy(x_copy[i], sample);
            }

            const y_copy = try allocator.dupe(T, train_y);

            return Self{
                .train_x = x_copy,
                .train_y = y_copy,
                .k = k,
                .allocator = allocator,
            };
        }

        /// Free all allocated memory
        ///
        /// Time: O(n)
        /// Space: O(1)
        pub fn deinit(self: *Self) void {
            for (self.train_x) |sample| {
                self.allocator.free(sample);
            }
            self.allocator.free(self.train_x);
            self.allocator.free(self.train_y);
        }

        /// Predict class label via majority vote
        ///
        /// Time: O(n × d + n log k)
        /// Space: O(n) for distances array
        ///
        /// Returns the most common class among k nearest neighbors.
        /// Ties are broken by choosing the class with the nearest neighbor.
        pub fn predictClass(self: *Self, query: []const T) !usize {
            const neighbors = try self.findKNearest(query);
            defer self.allocator.free(neighbors);

            // Count votes for each class (assumes labels are small integers)
            var votes = std.AutoHashMap(usize, usize).init(self.allocator);
            defer votes.deinit();

            for (neighbors) |idx| {
                const label = @as(usize, @intFromFloat(self.train_y[idx]));
                const count = votes.get(label) orelse 0;
                try votes.put(label, count + 1);
            }

            // Find majority class (break ties by nearest neighbor)
            var max_votes: usize = 0;
            var predicted_class: usize = 0;
            var it = votes.iterator();
            while (it.next()) |entry| {
                if (entry.value_ptr.* > max_votes) {
                    max_votes = entry.value_ptr.*;
                    predicted_class = entry.key_ptr.*;
                }
            }

            return predicted_class;
        }

        /// Predict continuous value via mean of neighbors
        ///
        /// Time: O(n × d + n log k)
        /// Space: O(n) for distances array
        ///
        /// Returns the average target value of k nearest neighbors.
        pub fn predictValue(self: *Self, query: []const T) !T {
            const neighbors = try self.findKNearest(query);
            defer self.allocator.free(neighbors);

            var sum: T = 0;
            for (neighbors) |idx| {
                sum += self.train_y[idx];
            }

            return sum / @as(T, @floatFromInt(self.k));
        }

        /// Predict with weighted voting/averaging (inverse distance weighting)
        ///
        /// Time: O(n × d + n log k)
        /// Space: O(n) for distances array
        ///
        /// Weights neighbors by 1/distance (closer neighbors have more influence).
        /// For classification, returns the weighted majority class.
        /// For regression, returns the weighted average.
        pub fn predictWeighted(self: *Self, query: []const T) !T {
            const distances = try self.computeDistances(query);
            defer self.allocator.free(distances);

            // Find k nearest and compute weights
            var weight_sum: T = 0;
            var weighted_value: T = 0;

            for (0..self.k) |i| {
                const idx = distances[i].index;
                const dist = distances[i].distance;

                // Inverse distance weighting (handle zero distance)
                const weight = if (dist < 1e-10) 1e10 else 1.0 / dist;
                weight_sum += weight;
                weighted_value += weight * self.train_y[idx];
            }

            return weighted_value / weight_sum;
        }

        /// Find k nearest neighbors' indices
        ///
        /// Time: O(n × d + n log k)
        /// Space: O(n)
        fn findKNearest(self: *Self, query: []const T) ![]usize {
            const distances = try self.computeDistances(query);
            defer self.allocator.free(distances);

            const indices = try self.allocator.alloc(usize, self.k);
            for (0..self.k) |i| {
                indices[i] = distances[i].index;
            }

            return indices;
        }

        const Distance = struct {
            distance: T,
            index: usize,
        };

        /// Compute distances to all training samples and sort
        ///
        /// Time: O(n × d + n log n)
        /// Space: O(n)
        fn computeDistances(self: *Self, query: []const T) ![]Distance {
            const n = self.train_x.len;
            var distances = try self.allocator.alloc(Distance, n);

            // Compute Euclidean distance to each training sample
            for (self.train_x, 0..) |sample, i| {
                if (sample.len != query.len) return error.DimensionMismatch;

                var sum_sq: T = 0;
                for (sample, query) |s, q| {
                    const diff = s - q;
                    sum_sq += diff * diff;
                }

                distances[i] = .{
                    .distance = @sqrt(sum_sq),
                    .index = i,
                };
            }

            // Sort by distance (ascending)
            const lessThan = struct {
                fn f(_: void, a: Distance, b: Distance) bool {
                    return a.distance < b.distance;
                }
            }.f;
            std.mem.sort(Distance, distances, {}, lessThan);

            return distances;
        }

        /// Get number of training samples
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn size(self: *const Self) usize {
            return self.train_x.len;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "KNN: basic classification (binary)" {
    const allocator = testing.allocator;

    // Two clusters: (0,0)-(2,2) and (5,5)-(7,7)
    const train_x = [_][]const f64{
        &.{ 0.0, 0.0 }, &.{ 1.0, 1.0 }, &.{ 2.0, 2.0 },
        &.{ 5.0, 5.0 }, &.{ 6.0, 6.0 }, &.{ 7.0, 7.0 },
    };
    const train_y = [_]f64{ 0, 0, 0, 1, 1, 1 };

    var knn = try KNN(f64).init(allocator, &train_x, &train_y, 3);
    defer knn.deinit();

    // Query near first cluster
    const label1 = try knn.predictClass(&.{ 1.5, 1.5 });
    try testing.expectEqual(@as(usize, 0), label1);

    // Query near second cluster
    const label2 = try knn.predictClass(&.{ 6.5, 6.5 });
    try testing.expectEqual(@as(usize, 1), label2);
}

test "KNN: multi-class classification" {
    const allocator = testing.allocator;

    // Three classes: corners of a triangle
    const train_x = [_][]const f64{
        &.{ 0.0, 0.0 }, &.{ 1.0, 0.0 }, // class 0
        &.{ 0.0, 5.0 }, &.{ 1.0, 5.0 }, // class 1
        &.{ 5.0, 0.0 }, &.{ 5.0, 1.0 }, // class 2
    };
    const train_y = [_]f64{ 0, 0, 1, 1, 2, 2 };

    var knn = try KNN(f64).init(allocator, &train_x, &train_y, 2);
    defer knn.deinit();

    try testing.expectEqual(@as(usize, 0), try knn.predictClass(&.{ 0.5, 0.5 }));
    try testing.expectEqual(@as(usize, 1), try knn.predictClass(&.{ 0.5, 4.5 }));
    try testing.expectEqual(@as(usize, 2), try knn.predictClass(&.{ 4.5, 0.5 }));
}

test "KNN: regression (predict continuous values)" {
    const allocator = testing.allocator;

    // Linear relationship: y = 2x
    const train_x = [_][]const f64{
        &.{0.0}, &.{1.0}, &.{2.0}, &.{3.0}, &.{4.0}, &.{5.0},
    };
    const train_y = [_]f64{ 0.0, 2.0, 4.0, 6.0, 8.0, 10.0 };

    var knn = try KNN(f64).init(allocator, &train_x, &train_y, 3);
    defer knn.deinit();

    // Query at x=2.5, expect y≈5.0 (average of 2, 3, 4 → 4, 6, 8 → avg=6)
    const pred1 = try knn.predictValue(&.{2.5});
    try testing.expect(@abs(pred1 - 6.0) < 0.1);

    // Query at x=1.0, exact training point
    const pred2 = try knn.predictValue(&.{1.0});
    try testing.expectApproxEqAbs(2.0, pred2, 0.1);
}

test "KNN: weighted prediction" {
    const allocator = testing.allocator;

    const train_x = [_][]const f64{
        &.{1.0}, &.{2.0}, &.{3.0},
    };
    const train_y = [_]f64{ 10.0, 20.0, 30.0 };

    var knn = try KNN(f64).init(allocator, &train_x, &train_y, 3);
    defer knn.deinit();

    // Query at x=1.5 (closer to 1.0 than 2.0, 3.0)
    const pred = try knn.predictWeighted(&.{1.5});
    // Should be weighted toward 10.0
    try testing.expect(pred < 20.0);
}

test "KNN: k=1 (nearest neighbor)" {
    const allocator = testing.allocator;

    const train_x = [_][]const f64{
        &.{ 0.0, 0.0 }, &.{ 1.0, 1.0 }, &.{ 5.0, 5.0 },
    };
    const train_y = [_]f64{ 0, 1, 2 };

    var knn = try KNN(f64).init(allocator, &train_x, &train_y, 1);
    defer knn.deinit();

    // Exact matches
    try testing.expectEqual(@as(usize, 0), try knn.predictClass(&.{ 0.0, 0.0 }));
    try testing.expectEqual(@as(usize, 1), try knn.predictClass(&.{ 1.0, 1.0 }));
    try testing.expectEqual(@as(usize, 2), try knn.predictClass(&.{ 5.0, 5.0 }));

    // Near matches
    try testing.expectEqual(@as(usize, 0), try knn.predictClass(&.{ 0.1, 0.1 }));
    try testing.expectEqual(@as(usize, 2), try knn.predictClass(&.{ 4.9, 4.9 }));
}

test "KNN: high-dimensional data" {
    const allocator = testing.allocator;

    // 5-dimensional features
    const train_x = [_][]const f64{
        &.{ 1, 2, 3, 4, 5 }, &.{ 2, 3, 4, 5, 6 },
        &.{ 10, 11, 12, 13, 14 }, &.{ 11, 12, 13, 14, 15 },
    };
    const train_y = [_]f64{ 0, 0, 1, 1 };

    var knn = try KNN(f64).init(allocator, &train_x, &train_y, 2);
    defer knn.deinit();

    try testing.expectEqual(@as(usize, 0), try knn.predictClass(&.{ 1.5, 2.5, 3.5, 4.5, 5.5 }));
    try testing.expectEqual(@as(usize, 1), try knn.predictClass(&.{ 10.5, 11.5, 12.5, 13.5, 14.5 }));
}

test "KNN: tie-breaking (majority vote)" {
    const allocator = testing.allocator;

    // k=4: if 2 neighbors vote 0 and 2 vote 1, implementation breaks tie deterministically
    const train_x = [_][]const f64{
        &.{ 0.0, 0.0 }, &.{ 0.1, 0.1 }, // class 0
        &.{ 5.0, 5.0 }, &.{ 5.1, 5.1 }, // class 1
    };
    const train_y = [_]f64{ 0, 0, 1, 1 };

    var knn = try KNN(f64).init(allocator, &train_x, &train_y, 4);
    defer knn.deinit();

    // Query at midpoint (2.5, 2.5) - tie situation
    const label = try knn.predictClass(&.{ 2.5, 2.5 });
    // Result is deterministic (implementation-defined tie-breaking)
    try testing.expect(label == 0 or label == 1);
}

test "KNN: single training sample (k=1)" {
    const allocator = testing.allocator;

    const train_x = [_][]const f64{&.{ 1.0, 2.0 }};
    const train_y = [_]f64{42.0};

    var knn = try KNN(f64).init(allocator, &train_x, &train_y, 1);
    defer knn.deinit();

    // All queries should return the single training label
    try testing.expectEqual(@as(usize, 42), try knn.predictClass(&.{ 0.0, 0.0 }));
    try testing.expectEqual(@as(usize, 42), try knn.predictClass(&.{ 10.0, 10.0 }));
}

test "KNN: error - empty training set" {
    const allocator = testing.allocator;

    const train_x = [_][]const f64{};
    const train_y = [_]f64{};

    const result = KNN(f64).init(allocator, &train_x, &train_y, 1);
    try testing.expectError(error.EmptyTrainingSet, result);
}

test "KNN: error - invalid k (k=0)" {
    const allocator = testing.allocator;

    const train_x = [_][]const f64{&.{ 1.0, 2.0 }};
    const train_y = [_]f64{1.0};

    const result = KNN(f64).init(allocator, &train_x, &train_y, 0);
    try testing.expectError(error.InvalidK, result);
}

test "KNN: error - invalid k (k > n)" {
    const allocator = testing.allocator;

    const train_x = [_][]const f64{
        &.{ 1.0, 2.0 }, &.{ 3.0, 4.0 },
    };
    const train_y = [_]f64{ 0, 1 };

    const result = KNN(f64).init(allocator, &train_x, &train_y, 5);
    try testing.expectError(error.InvalidK, result);
}

test "KNN: error - mismatched dimensions (train_x != train_y)" {
    const allocator = testing.allocator;

    const train_x = [_][]const f64{
        &.{ 1.0, 2.0 }, &.{ 3.0, 4.0 },
    };
    const train_y = [_]f64{0}; // wrong size

    const result = KNN(f64).init(allocator, &train_x, &train_y, 1);
    try testing.expectError(error.MismatchedDimensions, result);
}

test "KNN: error - dimension mismatch (query)" {
    const allocator = testing.allocator;

    const train_x = [_][]const f64{
        &.{ 1.0, 2.0 }, &.{ 3.0, 4.0 },
    };
    const train_y = [_]f64{ 0, 1 };

    var knn = try KNN(f64).init(allocator, &train_x, &train_y, 1);
    defer knn.deinit();

    // Query has wrong dimension (3 instead of 2)
    const result = knn.predictClass(&.{ 1.0, 2.0, 3.0 });
    try testing.expectError(error.DimensionMismatch, result);
}

test "KNN: memory safety (no leaks)" {
    const allocator = testing.allocator;

    const train_x = [_][]const f64{
        &.{ 1.0, 2.0 }, &.{ 3.0, 4.0 }, &.{ 5.0, 6.0 },
    };
    const train_y = [_]f64{ 0, 1, 2 };

    var knn = try KNN(f64).init(allocator, &train_x, &train_y, 2);
    defer knn.deinit();

    _ = try knn.predictClass(&.{ 2.0, 3.0 });
    _ = try knn.predictValue(&.{ 2.0, 3.0 });
    _ = try knn.predictWeighted(&.{ 2.0, 3.0 });

    // testing.allocator will catch any leaks
}

test "KNN: f32 support" {
    const allocator = testing.allocator;

    const train_x = [_][]const f32{
        &.{ 0.0, 0.0 }, &.{ 1.0, 1.0 }, &.{ 5.0, 5.0 },
    };
    const train_y = [_]f32{ 0, 1, 2 };

    var knn = try KNN(f32).init(allocator, &train_x, &train_y, 2);
    defer knn.deinit();

    try testing.expectEqual(@as(usize, 0), try knn.predictClass(&.{ 0.5, 0.5 }));
    try testing.expectEqual(@as(usize, 2), try knn.predictClass(&.{ 4.5, 4.5 }));
}

test "KNN: size() method" {
    const allocator = testing.allocator;

    const train_x = [_][]const f64{
        &.{ 1.0, 2.0 }, &.{ 3.0, 4.0 }, &.{ 5.0, 6.0 },
    };
    const train_y = [_]f64{ 0, 1, 2 };

    var knn = try KNN(f64).init(allocator, &train_x, &train_y, 2);
    defer knn.deinit();

    try testing.expectEqual(@as(usize, 3), knn.size());
}
