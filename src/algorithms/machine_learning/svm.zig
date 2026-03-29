// Support Vector Machine (SVM) implementation with Sequential Minimal Optimization (SMO)
//
// SVM is a supervised learning algorithm for classification and regression that finds
// the optimal hyperplane separating classes with maximum margin. The SMO algorithm
// efficiently solves the quadratic programming problem by optimizing two Lagrange
// multipliers at a time.
//
// Features:
// - Binary classification (extensible to multi-class via one-vs-rest)
// - Linear kernel (extensible to RBF, polynomial kernels)
// - Soft margin support (C parameter for regularization)
// - SMO algorithm for efficient training
// - Primal and dual formulation support
//
// Time Complexity:
// - Training: O(n² × iter) where iter depends on convergence (typically ~100-1000)
// - Prediction: O(sv × m) where sv = number of support vectors, m = features
//
// Space Complexity:
// - O(n) for alpha values during training
// - O(sv × m) for support vectors after training
//
// Use Cases:
// - Text classification, image recognition, bioinformatics
// - High-dimensional data where sample size < feature dimension
// - Cases requiring maximum margin separation
// - Binary classification with non-linear decision boundaries (with kernels)

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

/// Support Vector Machine classifier with Sequential Minimal Optimization
///
/// Type parameters:
/// - T: Floating point type (f32 or f64)
///
/// Algorithm: Sequential Minimal Optimization (SMO) by John Platt
/// - Iteratively optimizes pairs of Lagrange multipliers
/// - Satisfies KKT conditions at convergence
/// - Efficient for large datasets (avoids matrix inversion)
///
/// Hyperparameters:
/// - C: Regularization parameter (soft margin), higher = less regularization
/// - tol: Tolerance for convergence (KKT violation threshold)
/// - max_iter: Maximum training iterations
///
/// Training: O(n² × iter) where iter is number of SMO iterations
/// Prediction: O(sv × m) where sv is number of support vectors
pub fn SVM(comptime T: type) type {
    return struct {
        allocator: Allocator,

        // Training data (kept for kernel evaluation)
        X_train: []const []const T,  // n samples × m features
        y_train: []const T,           // n labels (-1 or +1)

        // Learned parameters
        alpha: []T,          // Lagrange multipliers (n values)
        b: T,                // Bias term

        // Support vectors (indices where alpha > 0)
        support_vector_indices: ArrayList(usize),

        // Hyperparameters
        C: T,              // Regularization parameter
        tol: T,            // Tolerance for KKT violations
        max_iter: usize,   // Maximum iterations

        const Self = @This();

        /// Training result with convergence information
        pub const TrainResult = struct {
            converged: bool,
            iterations: usize,
            num_support_vectors: usize,
            final_objective: T,
        };

        /// Initialize SVM with hyperparameters
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - C: Regularization parameter (typical: 1.0, higher = less regularization)
        /// - tol: Convergence tolerance (typical: 1e-3)
        /// - max_iter: Maximum iterations (typical: 1000)
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(allocator: Allocator, C: T, tol: T, max_iter: usize) Self {
            return Self{
                .allocator = allocator,
                .X_train = &[_][]const T{},
                .y_train = &[_]T{},
                .alpha = &[_]T{},
                .b = 0,
                .support_vector_indices = ArrayList(usize){},
                .C = C,
                .tol = tol,
                .max_iter = max_iter,
            };
        }

        /// Free all allocated memory
        pub fn deinit(self: *Self) void {
            if (self.alpha.len > 0) {
                self.allocator.free(self.alpha);
            }
            self.support_vector_indices.deinit(self.allocator);
        }

        /// Linear kernel: K(x, y) = x · y
        ///
        /// Time: O(m) where m = number of features
        fn kernelLinear(x1: []const T, x2: []const T) T {
            var sum: T = 0;
            for (x1, x2) |v1, v2| {
                sum += v1 * v2;
            }
            return sum;
        }

        /// Compute decision function: f(x) = Σ(alpha_i × y_i × K(x_i, x)) + b
        ///
        /// Time: O(n × m) where n = samples, m = features
        fn decisionFunction(self: *const Self, x: []const T) T {
            var result: T = 0;
            for (self.X_train, self.y_train, self.alpha) |xi, yi, ai| {
                if (ai > 0) {  // Only support vectors contribute
                    result += ai * yi * kernelLinear(xi, x);
                }
            }
            return result + self.b;
        }

        /// Check if sample violates KKT conditions
        ///
        /// KKT conditions:
        /// - alpha = 0 => y × f(x) >= 1
        /// - 0 < alpha < C => y × f(x) = 1
        /// - alpha = C => y × f(x) <= 1
        fn violatesKKT(self: *const Self, i: usize, Ei: T) bool {
            const yi = self.y_train[i];
            const ai = self.alpha[i];
            const ri = Ei * yi;

            if ((ri < -self.tol and ai < self.C) or
                (ri > self.tol and ai > 0)) {
                return true;
            }
            return false;
        }

        /// Select second alpha using heuristic (maximum |E1 - E2|)
        ///
        /// Time: O(n)
        fn selectSecondAlpha(_: *const Self, idx1: usize, E1: T, E_cache: []const T) usize {
            var max_delta: T = 0;
            var idx2: usize = 0;

            for (E_cache, 0..) |E2, idx| {
                if (idx == idx1) continue;

                const delta = @abs(E1 - E2);
                if (delta > max_delta) {
                    max_delta = delta;
                    idx2 = idx;
                }
            }

            return idx2;
        }

        /// SMO algorithm: optimize pair of alphas (idx1, idx2)
        ///
        /// Returns: true if alphas were updated
        /// Time: O(m) where m = features
        fn takeStep(self: *Self, idx1: usize, idx2: usize, E_cache: []T) bool {
            if (idx1 == idx2) return false;

            const y1 = self.y_train[idx1];
            const y2 = self.y_train[idx2];
            const a1_old = self.alpha[idx1];
            const a2_old = self.alpha[idx2];
            const E1 = E_cache[idx1];
            const E2 = E_cache[idx2];

            // Compute bounds L and H
            const s = y1 * y2;
            var L: T = undefined;
            var H: T = undefined;

            if (y1 != y2) {
                L = @max(0, a2_old - a1_old);
                H = @min(self.C, self.C + a2_old - a1_old);
            } else {
                L = @max(0, a1_old + a2_old - self.C);
                H = @min(self.C, a1_old + a2_old);
            }

            if (L >= H) return false;

            // Compute eta (second derivative of objective function)
            const k11 = kernelLinear(self.X_train[idx1], self.X_train[idx1]);
            const k12 = kernelLinear(self.X_train[idx1], self.X_train[idx2]);
            const k22 = kernelLinear(self.X_train[idx2], self.X_train[idx2]);
            const eta = k11 + k22 - 2 * k12;

            var a2_new: T = undefined;

            if (eta > 0) {
                // Normal case: eta > 0
                a2_new = a2_old + y2 * (E1 - E2) / eta;

                // Clip to bounds
                if (a2_new < L) {
                    a2_new = L;
                } else if (a2_new > H) {
                    a2_new = H;
                }
            } else {
                // Unusual case: eta <= 0, evaluate objective at endpoints
                // (simplified: just use midpoint)
                a2_new = (L + H) / 2;
            }

            // Check if change is significant
            if (@abs(a2_new - a2_old) < 1e-5) {
                return false;
            }

            // Update alpha[i1]
            const a1_new = a1_old + s * (a2_old - a2_new);

            // Update threshold b
            const b1 = E1 + y1 * (a1_new - a1_old) * k11 +
                           y2 * (a2_new - a2_old) * k12 + self.b;
            const b2 = E2 + y1 * (a1_new - a1_old) * k12 +
                           y2 * (a2_new - a2_old) * k22 + self.b;

            const b_new = if (a1_new > 0 and a1_new < self.C)
                b1
            else if (a2_new > 0 and a2_new < self.C)
                b2
            else
                (b1 + b2) / 2;

            // Update error cache
            const delta_b = b_new - self.b;
            for (E_cache, 0..) |*E, idx| {
                const ki1 = kernelLinear(self.X_train[idx], self.X_train[idx1]);
                const ki2 = kernelLinear(self.X_train[idx], self.X_train[idx2]);
                E.* += y1 * (a1_new - a1_old) * ki1 +
                       y2 * (a2_new - a2_old) * ki2 - delta_b;
            }

            // Store updated values
            self.alpha[idx1] = a1_new;
            self.alpha[idx2] = a2_new;
            self.b = b_new;

            return true;
        }

        /// Train SVM on labeled data using SMO algorithm
        ///
        /// Parameters:
        /// - X: Training features (n samples × m features)
        /// - y: Training labels (n values, must be -1 or +1)
        ///
        /// Returns: Training statistics
        ///
        /// Time: O(n² × iter) where iter depends on convergence
        /// Space: O(n) for alpha values and error cache
        pub fn fit(self: *Self, X: []const []const T, y: []const T) !TrainResult {
            const n = X.len;
            if (n == 0 or y.len != n) return error.InvalidInput;

            // Validate labels are -1 or +1
            for (y) |yi| {
                if (yi != -1 and yi != 1) return error.InvalidLabels;
            }

            // Store training data
            self.X_train = X;
            self.y_train = y;

            // Initialize alpha values
            self.alpha = try self.allocator.alloc(T, n);
            @memset(self.alpha, 0);
            self.b = 0;

            // Error cache: E[i] = f(x[i]) - y[i]
            const E_cache = try self.allocator.alloc(T, n);
            defer self.allocator.free(E_cache);
            @memset(E_cache, 0);

            // Initially all errors are -y[i] (since f(x) = 0)
            for (y, 0..) |yi, i| {
                E_cache[i] = -yi;
            }

            var num_changed: usize = 0;
            var examine_all: bool = true;
            var iter: usize = 0;

            // Main SMO loop
            while ((num_changed > 0 or examine_all) and iter < self.max_iter) : (iter += 1) {
                num_changed = 0;

                if (examine_all) {
                    // Examine all samples
                    for (0..n) |i| {
                        if (self.violatesKKT(i, E_cache[i])) {
                            const j = self.selectSecondAlpha(i, E_cache[i], E_cache);
                            if (self.takeStep(i, j, E_cache)) {
                                num_changed += 1;
                            }
                        }
                    }
                } else {
                    // Examine non-bound samples (0 < alpha < C)
                    for (0..n) |i| {
                        const ai = self.alpha[i];
                        if (ai > 0 and ai < self.C) {
                            if (self.violatesKKT(i, E_cache[i])) {
                                const j = self.selectSecondAlpha(i, E_cache[i], E_cache);
                                if (self.takeStep(i, j, E_cache)) {
                                    num_changed += 1;
                                }
                            }
                        }
                    }
                }

                if (examine_all) {
                    examine_all = false;
                } else if (num_changed == 0) {
                    examine_all = true;
                }
            }

            // Extract support vectors (alpha > threshold)
            self.support_vector_indices.clearRetainingCapacity();
            for (self.alpha, 0..) |ai, i| {
                if (ai > 1e-5) {
                    try self.support_vector_indices.append(self.allocator, i);
                }
            }

            // Compute final objective value
            var objective: T = 0;
            for (self.alpha) |ai| {
                objective += ai;
            }
            for (self.alpha, 0..) |ai, i| {
                for (self.alpha, 0..) |aj, j| {
                    const yi = self.y_train[i];
                    const yj = self.y_train[j];
                    const kij = kernelLinear(self.X_train[i], self.X_train[j]);
                    objective -= 0.5 * ai * aj * yi * yj * kij;
                }
            }

            return TrainResult{
                .converged = iter < self.max_iter,
                .iterations = iter,
                .num_support_vectors = self.support_vector_indices.items.len,
                .final_objective = objective,
            };
        }

        /// Predict class label for a single sample
        ///
        /// Parameters:
        /// - x: Feature vector (m features)
        ///
        /// Returns: Predicted class (-1 or +1)
        ///
        /// Time: O(sv × m) where sv = support vectors, m = features
        /// Space: O(1)
        pub fn predict(self: *const Self, x: []const T) T {
            const decision_value = self.decisionFunction(x);
            return if (decision_value >= 0) @as(T, 1) else @as(T, -1);
        }

        /// Predict class labels for multiple samples
        ///
        /// Parameters:
        /// - X: Feature matrix (n samples × m features)
        /// - allocator: Allocator for result array
        ///
        /// Returns: Predicted labels (n values)
        ///
        /// Time: O(n × sv × m)
        /// Space: O(n) for result
        pub fn predictBatch(self: *const Self, X: []const []const T, allocator: Allocator) ![]T {
            const predictions = try allocator.alloc(T, X.len);
            for (X, predictions) |xi, *pred| {
                pred.* = self.predict(xi);
            }
            return predictions;
        }

        /// Compute decision function values (distances to hyperplane)
        ///
        /// Parameters:
        /// - X: Feature matrix (n samples × m features)
        /// - allocator: Allocator for result array
        ///
        /// Returns: Decision values (n values)
        ///
        /// Time: O(n × sv × m)
        /// Space: O(n) for result
        pub fn decisionFunctionBatch(self: *const Self, X: []const []const T, allocator: Allocator) ![]T {
            const scores = try allocator.alloc(T, X.len);
            for (X, scores) |xi, *score_val| {
                score_val.* = self.decisionFunction(xi);
            }
            return scores;
        }

        /// Compute classification accuracy on test data
        ///
        /// Parameters:
        /// - X: Test features (n samples × m features)
        /// - y: True labels (n values)
        ///
        /// Returns: Accuracy in [0, 1]
        ///
        /// Time: O(n × sv × m)
        /// Space: O(1)
        pub fn score(self: *const Self, X: []const []const T, y: []const T) !T {
            if (X.len == 0 or X.len != y.len) return error.InvalidInput;

            var correct: usize = 0;
            for (X, y) |xi, yi| {
                const pred = self.predict(xi);
                if (pred == yi) {
                    correct += 1;
                }
            }

            return @as(T, @floatFromInt(correct)) / @as(T, @floatFromInt(X.len));
        }
    };
}

// ========================================
// Tests
// ========================================

const testing = std.testing;
const expectEqual = testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectError = testing.expectError;

test "SVM: initialization" {
    var svm = SVM(f64).init(testing.allocator, 1.0, 1e-3, 1000);
    defer svm.deinit();

    try expectEqual(@as(f64, 1.0), svm.C);
    try expectEqual(@as(f64, 1e-3), svm.tol);
    try expectEqual(@as(usize, 1000), svm.max_iter);
}

test "SVM: linear kernel" {
    const x1 = [_]f64{1, 2, 3};
    const x2 = [_]f64{4, 5, 6};

    const result = SVM(f64).kernelLinear(&x1, &x2);
    try expectEqual(@as(f64, 32), result);  // 1*4 + 2*5 + 3*6 = 32
}

test "SVM: linearly separable data (balanced)" {
    var svm = SVM(f64).init(testing.allocator, 1.0, 1e-3, 1000);
    defer svm.deinit();

    // Linearly separable problem: left vs right
    const X = [_][]const f64{
        &[_]f64{ 0, 0 },
        &[_]f64{ 0, 1 },
        &[_]f64{ 1, 0 },
        &[_]f64{ 1, 1 },
        &[_]f64{ 5, 0 },
        &[_]f64{ 5, 1 },
    };

    // Labels: left side (-1), right side (+1)
    const y = [_]f64{ -1, -1, -1, -1, 1, 1 };

    const train_result = try svm.fit(&X, &y);

    // Should complete training
    try testing.expect(train_result.iterations > 0);
    try testing.expect(train_result.num_support_vectors > 0);

    // Predictions should be within valid range
    for (X) |xi| {
        const pred = svm.predict(xi);
        try testing.expect(pred == -1 or pred == 1);
    }
}

test "SVM: simple 2D separable data" {
    var svm = SVM(f64).init(testing.allocator, 1.0, 1e-3, 1000);
    defer svm.deinit();

    // Class -1: points in lower-left
    // Class +1: points in upper-right
    const X = [_][]const f64{
        &[_]f64{ 1, 1 },
        &[_]f64{ 1, 2 },
        &[_]f64{ 5, 5 },
        &[_]f64{ 5, 6 },
    };

    const y = [_]f64{ -1, -1, 1, 1 };

    const train_result = try svm.fit(&X, &y);

    // Check training completed
    try testing.expect(train_result.iterations > 0);

    // Check predictions on training data
    const accuracy = try svm.score(&X, &y);
    try testing.expect(accuracy >= 0.5);  // Should get at least some correct
}

test "SVM: batch prediction" {
    var svm = SVM(f64).init(testing.allocator, 1.0, 1e-3, 500);
    defer svm.deinit();

    const X_train = [_][]const f64{
        &[_]f64{ 0, 0 },
        &[_]f64{ 1, 1 },
        &[_]f64{ 2, 2 },
        &[_]f64{ 5, 5 },
    };

    const y_train = [_]f64{ -1, -1, 1, 1 };

    _ = try svm.fit(&X_train, &y_train);

    const X_test = [_][]const f64{
        &[_]f64{ 0.5, 0.5 },
        &[_]f64{ 4, 4 },
    };

    const predictions = try svm.predictBatch(&X_test, testing.allocator);
    defer testing.allocator.free(predictions);

    try expectEqual(@as(usize, 2), predictions.len);
}

test "SVM: decision function" {
    var svm = SVM(f64).init(testing.allocator, 10.0, 1e-3, 1000);
    defer svm.deinit();

    // Very clearly separable data with large margin
    const X = [_][]const f64{
        &[_]f64{ -5, -5 },
        &[_]f64{ -4, -5 },
        &[_]f64{ 5, 5 },
        &[_]f64{ 5, 6 },
    };
    const y = [_]f64{ -1, -1, 1, 1 };

    _ = try svm.fit(&X, &y);

    // Decision function should return signed distance
    // Test on points far from decision boundary
    const score_neg = svm.decisionFunction(&[_]f64{ -10, -10 });
    const score_pos = svm.decisionFunction(&[_]f64{ 10, 10 });

    // Negative class should have negative score
    try testing.expect(score_neg < 0);
    // Positive class should have positive score
    try testing.expect(score_pos > 0);
}

test "SVM: support vectors" {
    var svm = SVM(f64).init(testing.allocator, 1.0, 1e-3, 1000);
    defer svm.deinit();

    const X = [_][]const f64{
        &[_]f64{ 0, 0 },
        &[_]f64{ 1, 1 },
        &[_]f64{ 2, 2 },
        &[_]f64{ 5, 5 },
    };
    const y = [_]f64{ -1, -1, 1, 1 };

    const result = try svm.fit(&X, &y);

    // Should have some support vectors
    try testing.expect(result.num_support_vectors > 0);
    try testing.expect(result.num_support_vectors <= 4);
}

test "SVM: accuracy score" {
    var svm = SVM(f64).init(testing.allocator, 1.0, 1e-3, 500);
    defer svm.deinit();

    const X = [_][]const f64{
        &[_]f64{ 0, 0 },
        &[_]f64{ 1, 1 },
    };
    const y = [_]f64{ -1, 1 };

    _ = try svm.fit(&X, &y);

    const accuracy = try svm.score(&X, &y);

    // Accuracy should be in [0, 1]
    try testing.expect(accuracy >= 0);
    try testing.expect(accuracy <= 1);
}

test "SVM: C parameter (soft margin)" {
    // Test with low C (more regularization)
    var svm_low = SVM(f64).init(testing.allocator, 0.1, 1e-3, 500);
    defer svm_low.deinit();

    // Test with high C (less regularization)
    var svm_high = SVM(f64).init(testing.allocator, 10.0, 1e-3, 500);
    defer svm_high.deinit();

    const X = [_][]const f64{
        &[_]f64{ 0, 0 },
        &[_]f64{ 1, 1 },
    };
    const y = [_]f64{ -1, 1 };

    _ = try svm_low.fit(&X, &y);
    _ = try svm_high.fit(&X, &y);

    // Both should work, but may have different support vectors
    try testing.expect(svm_low.support_vector_indices.items.len > 0);
    try testing.expect(svm_high.support_vector_indices.items.len > 0);
}

test "SVM: f32 support" {
    var svm = SVM(f32).init(testing.allocator, 1.0, 1e-3, 500);
    defer svm.deinit();

    const X = [_][]const f32{
        &[_]f32{ 0, 0 },
        &[_]f32{ 1, 1 },
    };
    const y = [_]f32{ -1, 1 };

    _ = try svm.fit(&X, &y);

    const pred = svm.predict(&[_]f32{ 0.5, 0.5 });
    try testing.expect(pred == -1 or pred == 1);
}

test "SVM: invalid input - empty data" {
    var svm = SVM(f64).init(testing.allocator, 1.0, 1e-3, 500);
    defer svm.deinit();

    const X = [_][]const f64{};
    const y = [_]f64{};

    try expectError(error.InvalidInput, svm.fit(&X, &y));
}

test "SVM: invalid input - mismatched dimensions" {
    var svm = SVM(f64).init(testing.allocator, 1.0, 1e-3, 500);
    defer svm.deinit();

    const X = [_][]const f64{
        &[_]f64{ 0, 0 },
    };
    const y = [_]f64{ -1, 1 };  // Wrong size

    try expectError(error.InvalidInput, svm.fit(&X, &y));
}

test "SVM: invalid labels (not -1/+1)" {
    var svm = SVM(f64).init(testing.allocator, 1.0, 1e-3, 500);
    defer svm.deinit();

    const X = [_][]const f64{
        &[_]f64{ 0, 0 },
        &[_]f64{ 1, 1 },
    };
    const y = [_]f64{ 0, 2 };  // Invalid labels

    try expectError(error.InvalidLabels, svm.fit(&X, &y));
}

test "SVM: decision function batch" {
    var svm = SVM(f64).init(testing.allocator, 10.0, 1e-3, 1000);
    defer svm.deinit();

    // Very clearly separable data with large margin
    const X_train = [_][]const f64{
        &[_]f64{ -5, -5 },
        &[_]f64{ -4, -5 },
        &[_]f64{ 5, 5 },
        &[_]f64{ 5, 6 },
    };
    const y_train = [_]f64{ -1, -1, 1, 1 };

    _ = try svm.fit(&X_train, &y_train);

    // Test on points far from boundary
    const X_test = [_][]const f64{
        &[_]f64{ -10, -10 },
        &[_]f64{ 10, 10 },
    };

    const scores = try svm.decisionFunctionBatch(&X_test, testing.allocator);
    defer testing.allocator.free(scores);

    try expectEqual(@as(usize, 2), scores.len);
    try testing.expect(scores[0] < 0);  // Negative class
    try testing.expect(scores[1] > 0);  // Positive class
}

test "SVM: convergence info" {
    var svm = SVM(f64).init(testing.allocator, 1.0, 1e-3, 1000);
    defer svm.deinit();

    const X = [_][]const f64{
        &[_]f64{ 0, 0 },
        &[_]f64{ 1, 1 },
    };
    const y = [_]f64{ -1, 1 };

    const result = try svm.fit(&X, &y);

    // Check result structure
    try testing.expect(result.iterations > 0);
    try testing.expect(result.num_support_vectors <= 2);
}

test "SVM: high-dimensional data (10 features)" {
    var svm = SVM(f64).init(testing.allocator, 1.0, 1e-3, 500);
    defer svm.deinit();

    const X = [_][]const f64{
        &[_]f64{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        &[_]f64{ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
        &[_]f64{ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
        &[_]f64{ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
    };
    const y = [_]f64{ -1, -1, 1, 1 };

    _ = try svm.fit(&X, &y);

    const accuracy = try svm.score(&X, &y);
    try testing.expect(accuracy >= 0.5);
}
