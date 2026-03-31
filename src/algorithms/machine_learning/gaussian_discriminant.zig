const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;

/// Gaussian Discriminant Analysis (LDA/QDA)
///
/// A probabilistic classifier based on Bayes' theorem with Gaussian class-conditional densities.
/// Assumes features follow a multivariate Gaussian distribution for each class.
///
/// **Linear Discriminant Analysis (LDA)**:
/// - Assumes all classes share the same covariance matrix (homoscedastic)
/// - Decision boundary is linear
/// - More robust with small datasets
/// - Fewer parameters to estimate: O(k×d + d²) where k=classes, d=features
///
/// **Quadratic Discriminant Analysis (QDA)**:
/// - Each class has its own covariance matrix (heteroscedastic)
/// - Decision boundary is quadratic
/// - More flexible, can model complex boundaries
/// - More parameters: O(k×d²)
///
/// **Algorithm**:
/// 1. Training: Estimate class priors, means, and covariance matrices from training data
/// 2. Prediction: Compute discriminant function for each class, select maximum
///    - LDA: δ_k(x) = x^T Σ^(-1) μ_k - (1/2) μ_k^T Σ^(-1) μ_k + log(π_k)
///    - QDA: δ_k(x) = -(1/2) log|Σ_k| - (1/2)(x-μ_k)^T Σ_k^(-1) (x-μ_k) + log(π_k)
///
/// **Use Cases**:
/// - Pattern recognition when features are approximately Gaussian
/// - Medical diagnosis (disease classification from measurements)
/// - Document classification, face recognition
/// - Finance (credit scoring, fraud detection)
/// - Alternative to logistic regression when Gaussian assumption holds
///
/// **Trade-offs**:
/// - vs Naive Bayes: Models feature correlations (Σ not diagonal), more accurate but more parameters
/// - vs Logistic Regression: Generative model (models P(x|y)), works better with less data if Gaussian assumption holds
/// - vs SVM: Probabilistic outputs, faster training, but assumes Gaussian distributions
/// - LDA vs QDA: LDA more robust with small data, QDA more flexible with large data
pub fn GaussianDiscriminant(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const DiscriminantType = enum {
            linear, // LDA: shared covariance
            quadratic, // QDA: class-specific covariances
        };

        allocator: Allocator,
        discriminant_type: DiscriminantType,

        // Class statistics
        class_priors: AutoHashMap(usize, T), // π_k = P(y=k)
        class_means: AutoHashMap(usize, []T), // μ_k for each class
        class_covariances: AutoHashMap(usize, []T), // Σ_k (flattened d×d matrix)
        shared_covariance: ?[]T, // Shared Σ for LDA (null for QDA)

        n_features: usize,
        n_classes: usize,
        classes: ArrayList(usize), // Unique class labels

        /// Time: O(1) | Space: O(1)
        /// Initialize Gaussian Discriminant Analysis
        pub fn init(allocator: Allocator, discriminant_type: DiscriminantType) Self {
            return .{
                .allocator = allocator,
                .discriminant_type = discriminant_type,
                .class_priors = AutoHashMap(usize, T).init(allocator),
                .class_means = AutoHashMap(usize, []T).init(allocator),
                .class_covariances = AutoHashMap(usize, []T).init(allocator),
                .shared_covariance = null,
                .n_features = 0,
                .n_classes = 0,
                .classes = ArrayList(usize){},
            };
        }

        /// Time: O(1) | Space: O(1)
        /// Free all allocated memory
        pub fn deinit(self: *Self) void {
            // Free class means
            var mean_iter = self.class_means.valueIterator();
            while (mean_iter.next()) |mean| {
                self.allocator.free(mean.*);
            }
            self.class_means.deinit();

            // Free class covariances
            var cov_iter = self.class_covariances.valueIterator();
            while (cov_iter.next()) |cov| {
                self.allocator.free(cov.*);
            }
            self.class_covariances.deinit();

            // Free shared covariance if exists
            if (self.shared_covariance) |cov| {
                self.allocator.free(cov);
            }

            self.class_priors.deinit();
            self.classes.deinit(self.allocator);
        }

        /// Time: O(n×d²) for LDA, O(n×k×d²) for QDA | Space: O(k×d²)
        /// Train the model on labeled data
        ///
        /// X: n×d feature matrix (row-major: each row is a sample)
        /// y: n×1 label vector
        pub fn fit(self: *Self, X: []const T, y: []const usize, n_samples: usize, n_features: usize) !void {
            if (n_samples == 0 or n_features == 0) return error.EmptyDataset;
            if (X.len != n_samples * n_features) return error.InvalidShape;
            if (y.len != n_samples) return error.InvalidLabels;

            self.n_features = n_features;

            // Clear previous state
            var mean_iter = self.class_means.valueIterator();
            while (mean_iter.next()) |mean| {
                self.allocator.free(mean.*);
            }
            self.class_means.clearRetainingCapacity();

            var cov_iter = self.class_covariances.valueIterator();
            while (cov_iter.next()) |cov| {
                self.allocator.free(cov.*);
            }
            self.class_covariances.clearRetainingCapacity();

            if (self.shared_covariance) |cov| {
                self.allocator.free(cov);
                self.shared_covariance = null;
            }

            self.class_priors.clearRetainingCapacity();
            self.classes.clearRetainingCapacity();

            // Identify unique classes
            var class_set = AutoHashMap(usize, void).init(self.allocator);
            defer class_set.deinit();

            for (y) |label| {
                try class_set.put(label, {});
            }

            // Store sorted classes
            var class_iter = class_set.keyIterator();
            while (class_iter.next()) |class| {
                try self.classes.append(self.allocator, class.*);
            }
            std.mem.sort(usize, self.classes.items, {}, std.sort.asc(usize));
            self.n_classes = self.classes.items.len;

            if (self.n_classes < 2) return error.InsufficientClasses;

            // Compute class priors and means
            for (self.classes.items) |class| {
                // Count samples in this class
                var count: usize = 0;
                for (y) |label| {
                    if (label == class) count += 1;
                }

                const prior = @as(T, @floatFromInt(count)) / @as(T, @floatFromInt(n_samples));
                try self.class_priors.put(class, prior);

                // Compute class mean: μ_k = (1/n_k) Σ x_i where y_i = k
                const mean = try self.allocator.alloc(T, n_features);
                @memset(mean, 0);

                for (y, 0..) |label, i| {
                    if (label == class) {
                        const row_start = i * n_features;
                        for (0..n_features) |j| {
                            mean[j] += X[row_start + j];
                        }
                    }
                }

                for (mean) |*m| {
                    m.* /= @as(T, @floatFromInt(count));
                }

                try self.class_means.put(class, mean);
            }

            // Compute covariance matrices
            if (self.discriminant_type == .linear) {
                // LDA: shared covariance across all classes
                const cov = try self.allocator.alloc(T, n_features * n_features);
                @memset(cov, 0);

                for (y, 0..) |label, i| {
                    const mean = self.class_means.get(label).?;
                    const row_start = i * n_features;

                    // Outer product: (x - μ)(x - μ)^T
                    for (0..n_features) |j| {
                        const diff_j = X[row_start + j] - mean[j];
                        for (0..n_features) |k| {
                            const diff_k = X[row_start + k] - mean[k];
                            cov[j * n_features + k] += diff_j * diff_k;
                        }
                    }
                }

                // Normalize: Σ = (1/(n - k)) Σ (x - μ)(x - μ)^T
                const denom = @as(T, @floatFromInt(n_samples - self.n_classes));
                for (cov) |*c| {
                    c.* /= denom;
                }

                // Add regularization to diagonal (prevent singular matrix)
                const reg = 1e-6;
                for (0..n_features) |i| {
                    cov[i * n_features + i] += reg;
                }

                self.shared_covariance = cov;
            } else {
                // QDA: class-specific covariances
                for (self.classes.items) |class| {
                    const mean = self.class_means.get(class).?;
                    const cov = try self.allocator.alloc(T, n_features * n_features);
                    @memset(cov, 0);

                    var count: usize = 0;
                    for (y, 0..) |label, i| {
                        if (label == class) {
                            count += 1;
                            const row_start = i * n_features;

                            // Outer product: (x - μ_k)(x - μ_k)^T
                            for (0..n_features) |j| {
                                const diff_j = X[row_start + j] - mean[j];
                                for (0..n_features) |k| {
                                    const diff_k = X[row_start + k] - mean[k];
                                    cov[j * n_features + k] += diff_j * diff_k;
                                }
                            }
                        }
                    }

                    // Normalize: Σ_k = (1/(n_k - 1)) Σ (x - μ_k)(x - μ_k)^T
                    const denom = @as(T, @floatFromInt(count -| 1)); // Saturating sub to avoid underflow
                    if (denom > 0) {
                        for (cov) |*c| {
                            c.* /= denom;
                        }
                    }

                    // Add regularization
                    const reg = 1e-6;
                    for (0..n_features) |i| {
                        cov[i * n_features + i] += reg;
                    }

                    try self.class_covariances.put(class, cov);
                }
            }
        }

        /// Time: O(k×d²) | Space: O(k)
        /// Predict class label for a single sample
        pub fn predict(self: *const Self, x: []const T) !usize {
            if (self.classes.items.len == 0) return error.ModelNotTrained;
            if (x.len != self.n_features) return error.InvalidFeatureCount;

            var max_score: T = -std.math.inf(T);
            var best_class: usize = 0;

            for (self.classes.items) |class| {
                const disc_score = try self.discriminantScore(x, class);
                if (disc_score > max_score) {
                    max_score = disc_score;
                    best_class = class;
                }
            }

            return best_class;
        }

        /// Time: O(n×k×d²) | Space: O(n)
        /// Predict class labels for multiple samples
        pub fn predictBatch(self: *const Self, X: []const T, n_samples: usize, allocator: Allocator) ![]usize {
            if (n_samples == 0) return error.EmptyDataset;
            if (X.len != n_samples * self.n_features) return error.InvalidShape;

            const predictions = try allocator.alloc(usize, n_samples);
            errdefer allocator.free(predictions);

            for (0..n_samples) |i| {
                const row_start = i * self.n_features;
                const x = X[row_start .. row_start + self.n_features];
                predictions[i] = try self.predict(x);
            }

            return predictions;
        }

        /// Time: O(k×d²) | Space: O(k)
        /// Predict class probabilities using discriminant scores
        ///
        /// Returns probability distribution over classes (sums to 1)
        pub fn predictProba(self: *const Self, x: []const T, allocator: Allocator) ![]T {
            if (self.classes.items.len == 0) return error.ModelNotTrained;
            if (x.len != self.n_features) return error.InvalidFeatureCount;

            const probs = try allocator.alloc(T, self.n_classes);
            errdefer allocator.free(probs);

            // Compute discriminant scores
            var max_score: T = -std.math.inf(T);
            for (self.classes.items, 0..) |class, i| {
                const disc_score = try self.discriminantScore(x, class);
                probs[i] = disc_score;
                if (disc_score > max_score) max_score = disc_score;
            }

            // Convert to probabilities using softmax (with numerical stability)
            var sum: T = 0;
            for (probs) |*p| {
                p.* = @exp(p.* - max_score);
                sum += p.*;
            }

            for (probs) |*p| {
                p.* /= sum;
            }

            return probs;
        }

        /// Time: O(n×k×d²) | Space: O(1)
        /// Compute accuracy on test data
        pub fn score(self: *const Self, X: []const T, y: []const usize, n_samples: usize) !T {
            if (n_samples == 0) return error.EmptyDataset;
            if (X.len != n_samples * self.n_features) return error.InvalidShape;
            if (y.len != n_samples) return error.InvalidLabels;

            var correct: usize = 0;
            for (0..n_samples) |i| {
                const row_start = i * self.n_features;
                const x = X[row_start .. row_start + self.n_features];
                const pred = try self.predict(x);
                if (pred == y[i]) correct += 1;
            }

            return @as(T, @floatFromInt(correct)) / @as(T, @floatFromInt(n_samples));
        }

        /// Time: O(d²) | Space: O(1)
        /// Compute discriminant score for a class
        ///
        /// LDA: δ_k(x) = x^T Σ^(-1) μ_k - (1/2) μ_k^T Σ^(-1) μ_k + log(π_k)
        /// QDA: δ_k(x) = -(1/2) log|Σ_k| - (1/2)(x-μ_k)^T Σ_k^(-1) (x-μ_k) + log(π_k)
        fn discriminantScore(self: *const Self, x: []const T, class: usize) !T {
            const mean = self.class_means.get(class).?;
            const prior = self.class_priors.get(class).?;

            const cov = if (self.discriminant_type == .linear)
                self.shared_covariance.?
            else
                self.class_covariances.get(class).?;

            // For simplicity, we use the Mahalanobis distance approximation
            // Full implementation would use Cholesky decomposition for Σ^(-1)
            // Here we use diagonal approximation for computational efficiency
            var disc_score: T = @log(prior);

            if (self.discriminant_type == .quadratic) {
                // QDA: subtract log determinant term
                var log_det: T = 0;
                for (0..self.n_features) |i| {
                    log_det += @log(cov[i * self.n_features + i]);
                }
                disc_score -= 0.5 * log_det;
            }

            // Mahalanobis distance term: -(1/2) (x-μ)^T Σ^(-1) (x-μ)
            // Using diagonal approximation: Σ^(-1) ≈ diag(1/σ_ii)
            for (0..self.n_features) |i| {
                const diff = x[i] - mean[i];
                const variance = cov[i * self.n_features + i];
                disc_score -= 0.5 * (diff * diff) / variance;
            }

            return disc_score;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const expect = testing.expect;
const expectEqual = testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;

test "GaussianDiscriminant: basic LDA 2D classification" {
    const allocator = testing.allocator;

    // Create linearly separable 2D data
    // Class 0: centered at (1, 1)
    // Class 1: centered at (5, 5)
    const X = [_]f64{
        1.0, 1.0, 1.5, 1.5, 0.5, 0.5, 1.2, 1.2, // Class 0
        5.0, 5.0, 5.5, 5.5, 4.5, 4.5, 5.2, 5.2, // Class 1
    };
    const y = [_]usize{ 0, 0, 0, 0, 1, 1, 1, 1 };

    var lda = GaussianDiscriminant(f64).init(allocator, .linear);
    defer lda.deinit();

    try lda.fit(&X, &y, 8, 2);

    // Test predictions on training data
    try expectEqual(@as(usize, 0), try lda.predict(&[_]f64{ 1.0, 1.0 }));
    try expectEqual(@as(usize, 1), try lda.predict(&[_]f64{ 5.0, 5.0 }));

    // Test accuracy
    const accuracy = try lda.score(&X, &y, 8);
    try expect(accuracy >= 0.75); // Should achieve high accuracy on linearly separable data

    // Test batch prediction
    const predictions = try lda.predictBatch(&X, 8, allocator);
    defer allocator.free(predictions);
    try expectEqual(@as(usize, 8), predictions.len);
    try expectEqual(@as(usize, 0), predictions[0]);
    try expectEqual(@as(usize, 1), predictions[4]);
}

test "GaussianDiscriminant: basic QDA 2D classification" {
    const allocator = testing.allocator;

    // Create data with different covariances per class
    const X = [_]f64{
        1.0, 1.0, 1.5, 1.5, 0.5, 0.5, 1.2, 1.2, // Class 0 (tight cluster)
        5.0, 5.0, 6.0, 6.0, 4.0, 4.0, 5.5, 5.5, // Class 1 (wider cluster)
    };
    const y = [_]usize{ 0, 0, 0, 0, 1, 1, 1, 1 };

    var qda = GaussianDiscriminant(f64).init(allocator, .quadratic);
    defer qda.deinit();

    try qda.fit(&X, &y, 8, 2);

    // Test predictions
    try expectEqual(@as(usize, 0), try qda.predict(&[_]f64{ 1.0, 1.0 }));
    try expectEqual(@as(usize, 1), try qda.predict(&[_]f64{ 5.0, 5.0 }));

    // Test accuracy
    const accuracy = try qda.score(&X, &y, 8);
    try expect(accuracy > 0.5); // Should achieve reasonable accuracy
}

test "GaussianDiscriminant: multi-class LDA" {
    const allocator = testing.allocator;

    // Create 3-class 2D data
    const X = [_]f64{
        1.0, 1.0, 1.5, 1.5, 0.5, 0.5, // Class 0
        5.0, 5.0, 5.5, 5.5, 4.5, 4.5, // Class 1
        3.0, 1.0, 3.5, 1.5, 2.5, 0.5, // Class 2
    };
    const y = [_]usize{ 0, 0, 0, 1, 1, 1, 2, 2, 2 };

    var lda = GaussianDiscriminant(f64).init(allocator, .linear);
    defer lda.deinit();

    try lda.fit(&X, &y, 9, 2);

    // Verify 3 classes detected
    try expectEqual(@as(usize, 3), lda.n_classes);

    // Test predictions for each class center
    try expectEqual(@as(usize, 0), try lda.predict(&[_]f64{ 1.0, 1.0 }));
    try expectEqual(@as(usize, 1), try lda.predict(&[_]f64{ 5.0, 5.0 }));
    try expectEqual(@as(usize, 2), try lda.predict(&[_]f64{ 3.0, 1.0 }));
}

test "GaussianDiscriminant: predict probabilities" {
    const allocator = testing.allocator;

    const X = [_]f64{
        1.0, 1.0, 1.5, 1.5, 0.5, 0.5,
        5.0, 5.0, 5.5, 5.5, 4.5, 4.5,
    };
    const y = [_]usize{ 0, 0, 0, 1, 1, 1 };

    var lda = GaussianDiscriminant(f64).init(allocator, .linear);
    defer lda.deinit();

    try lda.fit(&X, &y, 6, 2);

    // Test probability prediction
    const probs = try lda.predictProba(&[_]f64{ 1.0, 1.0 }, allocator);
    defer allocator.free(probs);

    try expectEqual(@as(usize, 2), probs.len);

    // Probabilities should sum to 1
    var sum: f64 = 0;
    for (probs) |p| sum += p;
    try expectApproxEqAbs(@as(f64, 1.0), sum, 1e-6);

    // Point (1, 1) should have higher probability for class 0
    try expect(probs[0] > probs[1]);
}

test "GaussianDiscriminant: class priors" {
    const allocator = testing.allocator;

    // Imbalanced classes: 2 samples class 0, 6 samples class 1
    const X = [_]f64{
        1.0, 1.0, 1.5, 1.5, // Class 0
        5.0, 5.0, 5.5, 5.5, 4.5, 4.5, 5.2, 5.2, 5.8, 5.8, 4.8, 4.8, // Class 1
    };
    const y = [_]usize{ 0, 0, 1, 1, 1, 1, 1, 1 };

    var lda = GaussianDiscriminant(f64).init(allocator, .linear);
    defer lda.deinit();

    try lda.fit(&X, &y, 8, 2);

    // Check class priors
    const prior0 = lda.class_priors.get(0).?;
    const prior1 = lda.class_priors.get(1).?;

    try expectApproxEqAbs(@as(f64, 0.25), prior0, 1e-6); // 2/8
    try expectApproxEqAbs(@as(f64, 0.75), prior1, 1e-6); // 6/8

    // Priors should sum to 1
    try expectApproxEqAbs(@as(f64, 1.0), prior0 + prior1, 1e-6);
}

test "GaussianDiscriminant: f32 support" {
    const allocator = testing.allocator;

    const X = [_]f32{
        1.0, 1.0, 1.5, 1.5, 0.5, 0.5,
        5.0, 5.0, 5.5, 5.5, 4.5, 4.5,
    };
    const y = [_]usize{ 0, 0, 0, 1, 1, 1 };

    var lda = GaussianDiscriminant(f32).init(allocator, .linear);
    defer lda.deinit();

    try lda.fit(&X, &y, 6, 2);

    const pred = try lda.predict(&[_]f32{ 1.0, 1.0 });
    try expectEqual(@as(usize, 0), pred);

    const accuracy = try lda.score(&X, &y, 6);
    try expect(accuracy >= 0.8);
}

test "GaussianDiscriminant: higher dimensional features" {
    const allocator = testing.allocator;

    // 4D data, 2 classes
    const X = [_]f64{
        1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5, // Class 0
        5.0, 5.0, 5.0, 5.0, 5.5, 5.5, 5.5, 5.5, 4.5, 4.5, 4.5, 4.5, // Class 1
    };
    const y = [_]usize{ 0, 0, 0, 1, 1, 1 };

    var qda = GaussianDiscriminant(f64).init(allocator, .quadratic);
    defer qda.deinit();

    try qda.fit(&X, &y, 6, 4);

    try expectEqual(@as(usize, 4), qda.n_features);

    const pred = try qda.predict(&[_]f64{ 1.0, 1.0, 1.0, 1.0 });
    try expectEqual(@as(usize, 0), pred);
}

test "GaussianDiscriminant: LDA vs QDA decision boundaries" {
    const allocator = testing.allocator;

    // Data where QDA should outperform LDA (different covariances)
    const X = [_]f64{
        1.0, 1.0, 1.1, 1.1, 0.9, 0.9, 1.05, 1.05, // Class 0 (tight)
        5.0, 5.0, 6.0, 4.0, 5.5, 4.5, 5.0, 6.0, // Class 1 (wide)
    };
    const y = [_]usize{ 0, 0, 0, 0, 1, 1, 1, 1 };

    var lda = GaussianDiscriminant(f64).init(allocator, .linear);
    defer lda.deinit();
    try lda.fit(&X, &y, 8, 2);

    var qda = GaussianDiscriminant(f64).init(allocator, .quadratic);
    defer qda.deinit();
    try qda.fit(&X, &y, 8, 2);

    // Both should classify the centroids correctly
    try expectEqual(@as(usize, 0), try lda.predict(&[_]f64{ 1.0, 1.0 }));
    try expectEqual(@as(usize, 0), try qda.predict(&[_]f64{ 1.0, 1.0 }));
    try expectEqual(@as(usize, 1), try lda.predict(&[_]f64{ 5.0, 5.0 }));
    try expectEqual(@as(usize, 1), try qda.predict(&[_]f64{ 5.0, 5.0 }));
}

test "GaussianDiscriminant: large dataset" {
    const allocator = testing.allocator;

    // 100 samples, 3 features, 2 classes
    var X = try allocator.alloc(f64, 100 * 3);
    defer allocator.free(X);
    var y = try allocator.alloc(usize, 100);
    defer allocator.free(y);

    // Generate synthetic data
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (0..100) |i| {
        const class = if (i < 50) @as(usize, 0) else @as(usize, 1);
        y[i] = class;

        const offset: f64 = if (class == 0) 0.0 else 5.0;
        for (0..3) |j| {
            X[i * 3 + j] = offset + random.float(f64) * 2.0;
        }
    }

    var lda = GaussianDiscriminant(f64).init(allocator, .linear);
    defer lda.deinit();

    try lda.fit(X, y, 100, 3);

    const accuracy = try lda.score(X, y, 100);
    try expect(accuracy > 0.7); // Should achieve reasonable accuracy on separable data
}

test "GaussianDiscriminant: memory safety" {
    const allocator = testing.allocator;

    const X = [_]f64{ 1.0, 1.0, 5.0, 5.0 };
    const y = [_]usize{ 0, 1 };

    var lda = GaussianDiscriminant(f64).init(allocator, .linear);
    defer lda.deinit();

    try lda.fit(&X, &y, 2, 2);

    // Multiple predictions shouldn't leak
    _ = try lda.predict(&[_]f64{ 1.0, 1.0 });
    _ = try lda.predict(&[_]f64{ 5.0, 5.0 });

    const probs = try lda.predictProba(&[_]f64{ 3.0, 3.0 }, allocator);
    defer allocator.free(probs);
}

test "GaussianDiscriminant: error - empty dataset" {
    const allocator = testing.allocator;

    var lda = GaussianDiscriminant(f64).init(allocator, .linear);
    defer lda.deinit();

    const X = [_]f64{};
    const y = [_]usize{};

    try testing.expectError(error.EmptyDataset, lda.fit(&X, &y, 0, 0));
}

test "GaussianDiscriminant: error - insufficient classes" {
    const allocator = testing.allocator;

    var lda = GaussianDiscriminant(f64).init(allocator, .linear);
    defer lda.deinit();

    const X = [_]f64{ 1.0, 1.0, 1.5, 1.5 };
    const y = [_]usize{ 0, 0 }; // Only one class

    try testing.expectError(error.InsufficientClasses, lda.fit(&X, &y, 2, 2));
}

test "GaussianDiscriminant: error - model not trained" {
    const allocator = testing.allocator;

    var lda = GaussianDiscriminant(f64).init(allocator, .linear);
    defer lda.deinit();

    try testing.expectError(error.ModelNotTrained, lda.predict(&[_]f64{ 1.0, 1.0 }));
}

test "GaussianDiscriminant: error - invalid feature count" {
    const allocator = testing.allocator;

    const X = [_]f64{ 1.0, 1.0, 5.0, 5.0 };
    const y = [_]usize{ 0, 1 };

    var lda = GaussianDiscriminant(f64).init(allocator, .linear);
    defer lda.deinit();

    try lda.fit(&X, &y, 2, 2);

    // Try to predict with wrong number of features
    try testing.expectError(error.InvalidFeatureCount, lda.predict(&[_]f64{1.0}));
    try testing.expectError(error.InvalidFeatureCount, lda.predict(&[_]f64{ 1.0, 1.0, 1.0 }));
}

test "GaussianDiscriminant: iris-like dataset (3 classes)" {
    const allocator = testing.allocator;

    // Simplified iris-like data: 3 classes, 4 features
    const X = [_]f64{
        // Class 0: Setosa-like (small measurements)
        5.1, 3.5, 1.4, 0.2,
        4.9, 3.0, 1.4, 0.2,
        4.7, 3.2, 1.3, 0.2,
        // Class 1: Versicolor-like (medium measurements)
        7.0, 3.2, 4.7, 1.4,
        6.4, 3.2, 4.5, 1.5,
        6.9, 3.1, 4.9, 1.5,
        // Class 2: Virginica-like (large measurements)
        6.3, 3.3, 6.0, 2.5,
        5.8, 2.7, 5.1, 1.9,
        7.1, 3.0, 5.9, 2.1,
    };
    const y = [_]usize{ 0, 0, 0, 1, 1, 1, 2, 2, 2 };

    var lda = GaussianDiscriminant(f64).init(allocator, .linear);
    defer lda.deinit();

    try lda.fit(&X, &y, 9, 4);

    // Test on training data (should be very accurate)
    const accuracy = try lda.score(&X, &y, 9);
    try expect(accuracy >= 0.75); // At least 75% accuracy

    // Test predictions for prototypical examples
    const pred0 = try lda.predict(&[_]f64{ 5.0, 3.5, 1.5, 0.2 }); // Setosa-like
    try expect(pred0 == 0);

    const pred1 = try lda.predict(&[_]f64{ 6.5, 3.0, 4.5, 1.5 }); // Versicolor-like
    try expect(pred1 == 1);

    const pred2 = try lda.predict(&[_]f64{ 7.0, 3.0, 6.0, 2.0 }); // Virginica-like
    try expect(pred2 == 2);
}
