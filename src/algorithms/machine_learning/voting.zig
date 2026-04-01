/// Voting Ensemble
///
/// Ensemble learning by aggregating predictions from multiple base estimators.
/// Supports both classification (hard/soft voting) and regression (averaging).
///
/// Voting strategies:
/// - **Hard voting (classification)**: Majority vote (argmax of counts)
/// - **Soft voting (classification)**: Average probabilities, then argmax
/// - **Average (regression)**: Mean of predictions
/// - **Weighted**: Apply estimator-specific weights to predictions
///
/// Time complexity: O(k × T_predict) where k = base estimators, T_predict = per-estimator cost
/// Space complexity: O(k × n) for storing predictions
///
/// Use cases:
/// - Combining diverse models (e.g., SVM + Decision Tree + KNN)
/// - Reducing variance (ensemble averaging)
/// - Leveraging strengths of different algorithms
/// - Sklearn VotingClassifier/VotingRegressor equivalent
///
/// Example (classification):
/// ```zig
/// // Assume you have trained base estimators: svm, tree, knn
/// var voting = VotingClassifier(f64).init(allocator, &.{
///     .{ .predict = svm.predict, .weight = 1.0 },
///     .{ .predict = tree.predict, .weight = 1.0 },
///     .{ .predict = knn.predict, .weight = 2.0 }, // higher weight
/// }, .hard);
/// defer voting.deinit();
/// const pred = try voting.predict(X);
/// ```

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Voting strategy for classification
pub const VotingStrategy = enum {
    hard, // Majority vote (predict class labels)
    soft, // Average probabilities (requires predict_proba)
};

/// Base estimator for classification
pub fn ClassifierEstimator(comptime T: type) type {
    return struct {
        /// Predict class labels (required for hard voting)
        predict: ?*const fn ([]const T) anyerror![]const usize = null,
        /// Predict class probabilities (required for soft voting)
        predict_proba: ?*const fn ([]const T) anyerror![]const []const T = null,
        /// Estimator weight (default: 1.0)
        weight: T = 1.0,
    };
}

/// Voting Classifier
///
/// Combines predictions from multiple classifiers using voting.
///
/// Time: O(k × n) per prediction where k = estimators, n = samples
/// Space: O(k × n × c) for soft voting, O(k × n) for hard voting (c = classes)
pub fn VotingClassifier(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        estimators: []const ClassifierEstimator(T),
        strategy: VotingStrategy,
        num_classes: usize = 0, // Detected during first prediction

        /// Initialize voting classifier
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - estimators: Base classifiers with optional weights
        /// - strategy: Voting strategy (hard or soft)
        ///
        /// Returns: Initialized VotingClassifier
        pub fn init(
            allocator: Allocator,
            estimators: []const ClassifierEstimator(T),
            strategy: VotingStrategy,
        ) !Self {
            if (estimators.len == 0) return error.NoEstimators;

            // Validate estimators have required methods
            for (estimators) |est| {
                if (strategy == .hard and est.predict == null) {
                    return error.HardVotingRequiresPredict;
                }
                if (strategy == .soft and est.predict_proba == null) {
                    return error.SoftVotingRequiresPredictProba;
                }
            }

            // Copy estimators
            const est_copy = try allocator.dupe(ClassifierEstimator(T), estimators);

            return Self{
                .allocator = allocator,
                .estimators = est_copy,
                .strategy = strategy,
            };
        }

        /// Free resources
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.estimators);
        }

        /// Predict class labels using voting
        ///
        /// Time: O(k × n × c) for soft, O(k × n) for hard
        /// Space: O(k × n) predictions buffer
        ///
        /// Parameters:
        /// - X: Feature matrix [n_samples × n_features] (flattened row-major)
        ///
        /// Returns: Predicted class labels [n_samples]
        pub fn predict(self: *Self, X: []const T) ![]usize {
            if (X.len == 0) return error.EmptyInput;

            switch (self.strategy) {
                .hard => return self.predictHard(X),
                .soft => return self.predictSoft(X),
            }
        }

        /// Hard voting: majority vote
        fn predictHard(self: *Self, X: []const T) ![]usize {
            const n_samples = X.len; // Simplified: assume 1D for demo
            var predictions = std.ArrayList(usize).init(self.allocator);
            defer predictions.deinit();

            // Get predictions from each estimator
            var all_preds = std.ArrayList([]const usize).init(self.allocator);
            defer {
                for (all_preds.items) |pred| {
                    self.allocator.free(pred);
                }
                all_preds.deinit();
            }

            for (self.estimators) |est| {
                const pred = try est.predict.?(X);
                try all_preds.append(pred);

                // Auto-detect num_classes
                for (pred) |label| {
                    if (label >= self.num_classes) {
                        self.num_classes = label + 1;
                    }
                }
            }

            // Vote for each sample
            const result = try self.allocator.alloc(usize, n_samples);
            var votes = try self.allocator.alloc(T, self.num_classes);
            defer self.allocator.free(votes);

            for (0..n_samples) |i| {
                @memset(votes, 0);

                // Count weighted votes
                for (all_preds.items, self.estimators) |pred, est| {
                    const label = pred[i];
                    votes[label] += est.weight;
                }

                // Find majority class
                var max_votes: T = votes[0];
                var max_label: usize = 0;
                for (votes, 0..) |v, j| {
                    if (v > max_votes) {
                        max_votes = v;
                        max_label = j;
                    }
                }

                result[i] = max_label;
            }

            return result;
        }

        /// Soft voting: average probabilities
        fn predictSoft(self: *Self, X: []const T) ![]usize {
            const n_samples = X.len; // Simplified: assume 1D for demo

            // Get probability predictions from each estimator
            var all_proba = std.ArrayList([]const []const T).init(self.allocator);
            defer {
                for (all_proba.items) |proba| {
                    for (proba) |row| {
                        self.allocator.free(row);
                    }
                    self.allocator.free(proba);
                }
                all_proba.deinit();
            }

            for (self.estimators) |est| {
                const proba = try est.predict_proba.?(X);
                try all_proba.append(proba);

                // Auto-detect num_classes
                if (proba.len > 0 and proba[0].len > self.num_classes) {
                    self.num_classes = proba[0].len;
                }
            }

            // Average probabilities
            const result = try self.allocator.alloc(usize, n_samples);
            var avg_proba = try self.allocator.alloc(T, self.num_classes);
            defer self.allocator.free(avg_proba);

            for (0..n_samples) |i| {
                @memset(avg_proba, 0);

                // Weighted average of probabilities
                var total_weight: T = 0;
                for (all_proba.items, self.estimators) |proba, est| {
                    total_weight += est.weight;
                    for (0..self.num_classes) |c| {
                        avg_proba[c] += proba[i][c] * est.weight;
                    }
                }

                // Normalize
                for (avg_proba) |*p| {
                    p.* /= total_weight;
                }

                // Find max probability class
                var max_prob = avg_proba[0];
                var max_label: usize = 0;
                for (avg_proba, 0..) |p, j| {
                    if (p > max_prob) {
                        max_prob = p;
                        max_label = j;
                    }
                }

                result[i] = max_label;
            }

            return result;
        }
    };
}

/// Base estimator for regression
pub fn RegressorEstimator(comptime T: type) type {
    return struct {
        /// Predict continuous values
        predict: *const fn ([]const T) anyerror![]const T,
        /// Estimator weight (default: 1.0)
        weight: T = 1.0,
    };
}

/// Voting Regressor
///
/// Combines predictions from multiple regressors using weighted averaging.
///
/// Time: O(k × n) per prediction where k = estimators, n = samples
/// Space: O(k × n) for storing predictions
pub fn VotingRegressor(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        estimators: []const RegressorEstimator(T),

        /// Initialize voting regressor
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - estimators: Base regressors with optional weights
        ///
        /// Returns: Initialized VotingRegressor
        pub fn init(
            allocator: Allocator,
            estimators: []const RegressorEstimator(T),
        ) !Self {
            if (estimators.len == 0) return error.NoEstimators;

            // Copy estimators
            const est_copy = try allocator.dupe(RegressorEstimator(T), estimators);

            return Self{
                .allocator = allocator,
                .estimators = est_copy,
            };
        }

        /// Free resources
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.estimators);
        }

        /// Predict continuous values using weighted averaging
        ///
        /// Time: O(k × n)
        /// Space: O(k × n) predictions buffer
        ///
        /// Parameters:
        /// - X: Feature matrix [n_samples × n_features] (flattened row-major)
        ///
        /// Returns: Predicted values [n_samples]
        pub fn predict(self: Self, X: []const T) ![]T {
            if (X.len == 0) return error.EmptyInput;

            const n_samples = X.len; // Simplified: assume 1D for demo

            // Get predictions from each estimator
            var all_preds = std.ArrayList([]const T).init(self.allocator);
            defer {
                for (all_preds.items) |pred| {
                    self.allocator.free(pred);
                }
                all_preds.deinit();
            }

            for (self.estimators) |est| {
                const pred = try est.predict(X);
                try all_preds.append(pred);
            }

            // Weighted average
            const result = try self.allocator.alloc(T, n_samples);
            for (0..n_samples) |i| {
                var sum: T = 0;
                var total_weight: T = 0;

                for (all_preds.items, self.estimators) |pred, est| {
                    sum += pred[i] * est.weight;
                    total_weight += est.weight;
                }

                result[i] = sum / total_weight;
            }

            return result;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

// Mock classifier that predicts constant class
fn mockPredictClass(comptime class: usize) *const fn ([]const f64) anyerror![]const usize {
    const Pred = struct {
        fn predict(X: []const f64) ![]const usize {
            const allocator = testing.allocator;
            const n = X.len;
            const result = try allocator.alloc(usize, n);
            @memset(result, class);
            return result;
        }
    };
    return Pred.predict;
}

// Mock classifier with probabilities
fn mockPredictProba(comptime class: usize, comptime n_classes: usize) *const fn ([]const f64) anyerror![]const []const f64 {
    const Pred = struct {
        fn predict_proba(X: []const f64) ![]const []const f64 {
            const allocator = testing.allocator;
            const n = X.len;
            const result = try allocator.alloc([]const f64, n);

            for (0..n) |i| {
                const proba = try allocator.alloc(f64, n_classes);
                @memset(proba, 0);
                proba[class] = 1.0; // Certain prediction
                result[i] = proba;
            }

            return result;
        }
    };
    return Pred.predict_proba;
}

test "VotingClassifier: initialization" {
    const estimators = &[_]ClassifierEstimator(f64){
        .{ .predict = mockPredictClass(0), .weight = 1.0 },
        .{ .predict = mockPredictClass(1), .weight = 1.0 },
    };

    var voting = try VotingClassifier(f64).init(testing.allocator, estimators, .hard);
    defer voting.deinit();

    try testing.expectEqual(@as(usize, 2), voting.estimators.len);
    try testing.expectEqual(VotingStrategy.hard, voting.strategy);
}

test "VotingClassifier: hard voting - unanimous" {
    // All estimators predict class 0
    const estimators = &[_]ClassifierEstimator(f64){
        .{ .predict = mockPredictClass(0), .weight = 1.0 },
        .{ .predict = mockPredictClass(0), .weight = 1.0 },
        .{ .predict = mockPredictClass(0), .weight = 1.0 },
    };

    var voting = try VotingClassifier(f64).init(testing.allocator, estimators, .hard);
    defer voting.deinit();

    const X = &[_]f64{ 1.0, 2.0, 3.0 };
    const pred = try voting.predict(X);
    defer testing.allocator.free(pred);

    try testing.expectEqual(@as(usize, 3), pred.len);
    for (pred) |label| {
        try testing.expectEqual(@as(usize, 0), label);
    }
}

test "VotingClassifier: hard voting - majority wins" {
    // 2 vote for class 0, 1 votes for class 1 → class 0 wins
    const estimators = &[_]ClassifierEstimator(f64){
        .{ .predict = mockPredictClass(0), .weight = 1.0 },
        .{ .predict = mockPredictClass(0), .weight = 1.0 },
        .{ .predict = mockPredictClass(1), .weight = 1.0 },
    };

    var voting = try VotingClassifier(f64).init(testing.allocator, estimators, .hard);
    defer voting.deinit();

    const X = &[_]f64{ 1.0, 2.0 };
    const pred = try voting.predict(X);
    defer testing.allocator.free(pred);

    try testing.expectEqual(@as(usize, 2), pred.len);
    for (pred) |label| {
        try testing.expectEqual(@as(usize, 0), label);
    }
}

test "VotingClassifier: hard voting - weighted" {
    // Class 0 (weight 1.0) vs Class 1 (weight 2.0) → class 1 wins
    const estimators = &[_]ClassifierEstimator(f64){
        .{ .predict = mockPredictClass(0), .weight = 1.0 },
        .{ .predict = mockPredictClass(1), .weight = 2.0 },
    };

    var voting = try VotingClassifier(f64).init(testing.allocator, estimators, .hard);
    defer voting.deinit();

    const X = &[_]f64{ 1.0, 2.0 };
    const pred = try voting.predict(X);
    defer testing.allocator.free(pred);

    try testing.expectEqual(@as(usize, 2), pred.len);
    for (pred) |label| {
        try testing.expectEqual(@as(usize, 1), label);
    }
}

test "VotingClassifier: soft voting - average probabilities" {
    // Each estimator is certain about different classes
    const estimators = &[_]ClassifierEstimator(f64){
        .{ .predict_proba = mockPredictProba(0, 3), .weight = 1.0 },
        .{ .predict_proba = mockPredictProba(1, 3), .weight = 1.0 },
        .{ .predict_proba = mockPredictProba(1, 3), .weight = 1.0 },
    };

    var voting = try VotingClassifier(f64).init(testing.allocator, estimators, .soft);
    defer voting.deinit();

    const X = &[_]f64{ 1.0, 2.0 };
    const pred = try voting.predict(X);
    defer testing.allocator.free(pred);

    try testing.expectEqual(@as(usize, 2), pred.len);
    // Class 1 has 2/3 probability → wins
    for (pred) |label| {
        try testing.expectEqual(@as(usize, 1), label);
    }
}

test "VotingClassifier: error - no estimators" {
    const estimators = &[_]ClassifierEstimator(f64){};
    try testing.expectError(error.NoEstimators, VotingClassifier(f64).init(testing.allocator, estimators, .hard));
}

test "VotingClassifier: error - empty input" {
    const estimators = &[_]ClassifierEstimator(f64){
        .{ .predict = mockPredictClass(0), .weight = 1.0 },
    };

    var voting = try VotingClassifier(f64).init(testing.allocator, estimators, .hard);
    defer voting.deinit();

    const X = &[_]f64{};
    try testing.expectError(error.EmptyInput, voting.predict(X));
}

// Mock regressor that predicts constant value
fn mockPredictValue(comptime value: f64) *const fn ([]const f64) anyerror![]const f64 {
    const Pred = struct {
        fn predict(X: []const f64) ![]const f64 {
            const allocator = testing.allocator;
            const n = X.len;
            const result = try allocator.alloc(f64, n);
            @memset(result, value);
            return result;
        }
    };
    return Pred.predict;
}

test "VotingRegressor: initialization" {
    const estimators = &[_]RegressorEstimator(f64){
        .{ .predict = mockPredictValue(1.0), .weight = 1.0 },
        .{ .predict = mockPredictValue(2.0), .weight = 1.0 },
    };

    var voting = try VotingRegressor(f64).init(testing.allocator, estimators);
    defer voting.deinit();

    try testing.expectEqual(@as(usize, 2), voting.estimators.len);
}

test "VotingRegressor: average predictions" {
    // Average of 1.0, 2.0, 3.0 = 2.0
    const estimators = &[_]RegressorEstimator(f64){
        .{ .predict = mockPredictValue(1.0), .weight = 1.0 },
        .{ .predict = mockPredictValue(2.0), .weight = 1.0 },
        .{ .predict = mockPredictValue(3.0), .weight = 1.0 },
    };

    var voting = try VotingRegressor(f64).init(testing.allocator, estimators);
    defer voting.deinit();

    const X = &[_]f64{ 1.0, 2.0, 3.0 };
    const pred = try voting.predict(X);
    defer testing.allocator.free(pred);

    try testing.expectEqual(@as(usize, 3), pred.len);
    for (pred) |value| {
        try testing.expectApproxEqAbs(@as(f64, 2.0), value, 1e-6);
    }
}

test "VotingRegressor: weighted average" {
    // Weighted: (1.0×1.0 + 3.0×2.0) / (1.0+2.0) = 7.0/3.0 ≈ 2.333
    const estimators = &[_]RegressorEstimator(f64){
        .{ .predict = mockPredictValue(1.0), .weight = 1.0 },
        .{ .predict = mockPredictValue(3.0), .weight = 2.0 },
    };

    var voting = try VotingRegressor(f64).init(testing.allocator, estimators);
    defer voting.deinit();

    const X = &[_]f64{ 1.0, 2.0 };
    const pred = try voting.predict(X);
    defer testing.allocator.free(pred);

    try testing.expectEqual(@as(usize, 2), pred.len);
    const expected = (1.0 * 1.0 + 3.0 * 2.0) / (1.0 + 2.0);
    for (pred) |value| {
        try testing.expectApproxEqAbs(expected, value, 1e-6);
    }
}

test "VotingRegressor: error - no estimators" {
    const estimators = &[_]RegressorEstimator(f64){};
    try testing.expectError(error.NoEstimators, VotingRegressor(f64).init(testing.allocator, estimators));
}

test "VotingRegressor: error - empty input" {
    const estimators = &[_]RegressorEstimator(f64){
        .{ .predict = mockPredictValue(1.0), .weight = 1.0 },
    };

    var voting = try VotingRegressor(f64).init(testing.allocator, estimators);
    defer voting.deinit();

    const X = &[_]f64{};
    try testing.expectError(error.EmptyInput, voting.predict(X));
}

test "VotingRegressor: f32 support" {
    const estimators = &[_]RegressorEstimator(f32){
        .{ .predict = mockPredictValue(1.0), .weight = 1.0 },
        .{ .predict = mockPredictValue(2.0), .weight = 1.0 },
    };

    var voting = try VotingRegressor(f32).init(testing.allocator, estimators);
    defer voting.deinit();

    const X = &[_]f32{ 1.0, 2.0 };
    const pred = try voting.predict(X);
    defer testing.allocator.free(pred);

    try testing.expectEqual(@as(usize, 2), pred.len);
    for (pred) |value| {
        try testing.expectApproxEqAbs(@as(f32, 1.5), value, 1e-5);
    }
}

test "VotingClassifier: memory safety" {
    const estimators = &[_]ClassifierEstimator(f64){
        .{ .predict = mockPredictClass(0), .weight = 1.0 },
        .{ .predict = mockPredictClass(1), .weight = 1.0 },
    };

    var voting = try VotingClassifier(f64).init(testing.allocator, estimators, .hard);
    defer voting.deinit();

    const X = &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const pred = try voting.predict(X);
    defer testing.allocator.free(pred);

    // No assertions needed - just verify no leaks
}

test "VotingRegressor: memory safety" {
    const estimators = &[_]RegressorEstimator(f64){
        .{ .predict = mockPredictValue(1.0), .weight = 1.0 },
        .{ .predict = mockPredictValue(2.0), .weight = 1.0 },
    };

    var voting = try VotingRegressor(f64).init(testing.allocator, estimators);
    defer voting.deinit();

    const X = &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const pred = try voting.predict(X);
    defer testing.allocator.free(pred);

    // No assertions needed - just verify no leaks
}
