const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;
const math = std.math;

/// Perceptron classifier - simple linear classifier with online learning
///
/// Algorithm:
/// - Linear model: f(x) = sign(w·x + b)
/// - Online learning: Update weights on misclassification
/// - Update rule: w ← w + η·y·x, b ← b + η·y (where y ∈ {-1, +1})
/// - Multi-class: One-vs-Rest (OvR) strategy
///
/// Time complexity:
/// - Training: O(n_epochs × n × m) where n = samples, m = features
/// - Prediction: O(k × m) where k = classes (1 for binary)
///
/// Space complexity: O(k × m) for storing weight matrices
///
/// Features:
/// - Online/incremental learning (can update with single examples)
/// - No probabilistic output (hard predictions only)
/// - Learning rate scheduling
/// - Early stopping on perfect convergence
/// - Averaged perceptron variant for stability
///
/// Advantages:
/// - Simple and fast
/// - Online learning capability
/// - Works well for linearly separable data
/// - Memory efficient
///
/// Limitations:
/// - Only for linearly separable problems
/// - No probabilistic output
/// - Sensitive to learning rate
/// - May not converge on non-separable data
///
/// Use cases:
/// - Binary classification: sentiment analysis, spam detection
/// - Online learning: streaming data classification
/// - Large-scale problems: fast training on millions of samples
/// - Linear decision boundaries
/// - Baseline model for comparison
pub fn Perceptron(comptime T: type) type {
    if (T != f32 and T != f64) {
        @compileError("Perceptron only supports f32 and f64");
    }

    return struct {
        const Self = @This();

        allocator: Allocator,
        /// Weight matrices for each class: [n_classes][n_features]
        weights: ArrayList(ArrayList(T)),
        /// Bias terms for each class: [n_classes]
        biases: ArrayList(T),
        /// Class labels
        classes: ArrayList(i32),
        /// Number of features
        n_features: usize,
        /// Learning rate
        learning_rate: T,
        /// Maximum epochs for training
        max_epochs: usize,
        /// Random seed for shuffling
        seed: u64,
        /// Use averaged perceptron (more stable)
        use_averaged: bool,
        /// Number of classes
        n_classes: usize,

        /// Initialize perceptron classifier
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator, learning_rate: T, max_epochs: usize, seed: u64, use_averaged: bool) Self {
            return .{
                .allocator = allocator,
                .weights = ArrayList(ArrayList(T)).init(allocator),
                .biases = ArrayList(T).init(allocator),
                .classes = ArrayList(i32).init(allocator),
                .n_features = 0,
                .learning_rate = learning_rate,
                .max_epochs = max_epochs,
                .seed = seed,
                .use_averaged = use_averaged,
                .n_classes = 0,
            };
        }

        /// Free all memory
        /// Time: O(k × m) | Space: O(1)
        pub fn deinit(self: *Self) void {
            for (self.weights.items) |*weight_vec| {
                weight_vec.deinit();
            }
            self.weights.deinit();
            self.biases.deinit();
            self.classes.deinit();
        }

        /// Train the perceptron using online updates
        /// X: [n_samples][n_features] training data
        /// y: [n_samples] class labels
        /// Time: O(max_epochs × n × m) | Space: O(k × m)
        pub fn fit(self: *Self, X: []const []const T, y: []const i32) !void {
            if (X.len == 0) return error.EmptyData;
            if (X.len != y.len) return error.MismatchedDimensions;

            const n_samples = X.len;
            self.n_features = X[0].len;

            // Identify unique classes
            var class_set = AutoHashMap(i32, void).init(self.allocator);
            defer class_set.deinit();

            for (y) |label| {
                try class_set.put(label, {});
            }

            // Store sorted classes
            self.classes.clearRetainingCapacity();
            var class_iter = class_set.keyIterator();
            while (class_iter.next()) |class_ptr| {
                try self.classes.append(class_ptr.*);
            }
            std.mem.sort(i32, self.classes.items, {}, std.sort.asc(i32));
            self.n_classes = self.classes.items.len;

            if (self.n_classes < 2) return error.InsufficientClasses;

            // Initialize weights and biases (one per class for OvR)
            for (self.weights.items) |*weight_vec| {
                weight_vec.deinit();
            }
            self.weights.clearRetainingCapacity();
            self.biases.clearRetainingCapacity();

            for (0..self.n_classes) |_| {
                var weight_vec = try ArrayList(T).initCapacity(self.allocator, self.n_features);
                for (0..self.n_features) |_| {
                    try weight_vec.append(0.0);
                }
                try self.weights.append(weight_vec);
                try self.biases.append(0.0);
            }

            // Averaged perceptron: accumulate weights
            var avg_weights: ?ArrayList(ArrayList(T)) = null;
            var avg_biases: ?ArrayList(T) = null;
            var update_count: usize = 0;

            if (self.use_averaged) {
                avg_weights = try ArrayList(ArrayList(T)).initCapacity(self.allocator, self.n_classes);
                avg_biases = try ArrayList(T).initCapacity(self.allocator, self.n_classes);
                
                for (0..self.n_classes) |_| {
                    var weight_vec = try ArrayList(T).initCapacity(self.allocator, self.n_features);
                    for (0..self.n_features) |_| {
                        try weight_vec.append(0.0);
                    }
                    try avg_weights.?.append(weight_vec);
                    try avg_biases.?.append(0.0);
                }
            }
            defer {
                if (avg_weights) |*aw| {
                    for (aw.items) |*weight_vec| {
                        weight_vec.deinit();
                    }
                    aw.deinit();
                }
                if (avg_biases) |*ab| {
                    ab.deinit();
                }
            }

            // Create index array for shuffling
            var indices = try ArrayList(usize).initCapacity(self.allocator, n_samples);
            defer indices.deinit();
            for (0..n_samples) |i| {
                try indices.append(i);
            }

            // Training loop
            var rng = std.rand.DefaultPrng.init(self.seed);
            var random = rng.random();

            for (0..self.max_epochs) |epoch| {
                _ = epoch;
                var n_errors: usize = 0;

                // Shuffle data each epoch
                random.shuffle(usize, indices.items);

                // Online updates
                for (indices.items) |idx| {
                    const x = X[idx];
                    const true_label = y[idx];

                    // For each class (One-vs-Rest)
                    for (self.classes.items, 0..) |class_label, class_idx| {
                        // Convert to binary: +1 if this class, -1 otherwise
                        const binary_label: T = if (true_label == class_label) 1.0 else -1.0;

                        // Compute prediction: sign(w·x + b)
                        var activation: T = self.biases.items[class_idx];
                        for (x, 0..) |x_val, j| {
                            activation += self.weights.items[class_idx].items[j] * x_val;
                        }
                        const prediction: T = if (activation >= 0.0) 1.0 else -1.0;

                        // Update on misclassification
                        if (prediction != binary_label) {
                            n_errors += 1;
                            
                            // Update weights: w ← w + η·y·x
                            for (x, 0..) |x_val, j| {
                                self.weights.items[class_idx].items[j] += self.learning_rate * binary_label * x_val;
                            }
                            // Update bias: b ← b + η·y
                            self.biases.items[class_idx] += self.learning_rate * binary_label;

                            // Accumulate for averaged perceptron
                            if (self.use_averaged) {
                                update_count += 1;
                            }
                        }

                        // Accumulate weights for averaging
                        if (self.use_averaged) {
                            for (0..self.n_features) |j| {
                                avg_weights.?.items[class_idx].items[j] += self.weights.items[class_idx].items[j];
                            }
                            avg_biases.?.items[class_idx] += self.biases.items[class_idx];
                        }
                    }
                }

                // Early stopping if perfect convergence
                if (n_errors == 0) {
                    break;
                }
            }

            // Apply averaging if enabled
            if (self.use_averaged and update_count > 0) {
                const total_updates = @as(T, @floatFromInt(update_count));
                for (0..self.n_classes) |class_idx| {
                    for (0..self.n_features) |j| {
                        self.weights.items[class_idx].items[j] = avg_weights.?.items[class_idx].items[j] / total_updates;
                    }
                    self.biases.items[class_idx] = avg_biases.?.items[class_idx] / total_updates;
                }
            }
        }

        /// Predict class labels for samples
        /// X: [n_samples][n_features] test data
        /// Returns: [n_samples] predicted class labels
        /// Time: O(n × k × m) | Space: O(n)
        pub fn predict(self: *const Self, X: []const []const T) ![]i32 {
            if (X.len == 0) return error.EmptyData;
            if (self.n_features == 0) return error.NotFitted;

            const n_samples = X.len;
            var predictions = try self.allocator.alloc(i32, n_samples);

            for (X, 0..) |x, i| {
                if (x.len != self.n_features) {
                    self.allocator.free(predictions);
                    return error.MismatchedDimensions;
                }

                // Compute scores for each class
                var best_activation: T = -math.inf(T);
                var best_class: i32 = self.classes.items[0];

                for (self.classes.items, 0..) |class_label, class_idx| {
                    var activation: T = self.biases.items[class_idx];
                    for (x, 0..) |x_val, j| {
                        activation += self.weights.items[class_idx].items[j] * x_val;
                    }

                    if (activation > best_activation) {
                        best_activation = activation;
                        best_class = class_label;
                    }
                }

                predictions[i] = best_class;
            }

            return predictions;
        }

        /// Compute accuracy on test set
        /// X: [n_samples][n_features] test data
        /// y: [n_samples] true labels
        /// Returns: accuracy in [0, 1]
        /// Time: O(n × k × m) | Space: O(n)
        pub fn score(self: *const Self, X: []const []const T, y: []const i32) !T {
            const predictions = try self.predict(X);
            defer self.allocator.free(predictions);

            var correct: usize = 0;
            for (predictions, 0..) |pred, i| {
                if (pred == y[i]) {
                    correct += 1;
                }
            }

            return @as(T, @floatFromInt(correct)) / @as(T, @floatFromInt(y.len));
        }

        /// Get learned weights for a specific class
        /// class_idx: index of the class (0 to n_classes-1)
        /// Returns: slice of weights [n_features]
        /// Time: O(1) | Space: O(1)
        pub fn getWeights(self: *const Self, class_idx: usize) ![]const T {
            if (class_idx >= self.n_classes) return error.InvalidClassIndex;
            return self.weights.items[class_idx].items;
        }

        /// Get bias for a specific class
        /// class_idx: index of the class (0 to n_classes-1)
        /// Returns: bias value
        /// Time: O(1) | Space: O(1)
        pub fn getBias(self: *const Self, class_idx: usize) !T {
            if (class_idx >= self.n_classes) return error.InvalidClassIndex;
            return self.biases.items[class_idx];
        }

        /// Get number of classes
        /// Time: O(1) | Space: O(1)
        pub fn getNumClasses(self: *const Self) usize {
            return self.n_classes;
        }

        /// Get class labels
        /// Time: O(1) | Space: O(1)
        pub fn getClasses(self: *const Self) []const i32 {
            return self.classes.items;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const expectEqual = testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectError = testing.expectError;

test "Perceptron: basic binary classification (linearly separable)" {
    const allocator = testing.allocator;

    // AND problem: linearly separable
    var X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 0.0, 1.0 },
        &[_]f64{ 1.0, 0.0 },
        &[_]f64{ 1.0, 1.0 },
    };
    var y = [_]i32{ 0, 0, 0, 1 };

    var model = Perceptron(f64).init(allocator, 0.1, 100, 42, false);
    defer model.deinit();

    try model.fit(&X, &y);

    // Predict
    const predictions = try model.predict(&X);
    defer allocator.free(predictions);

    // Should learn AND function perfectly
    try expectEqual(@as(i32, 0), predictions[0]);
    try expectEqual(@as(i32, 0), predictions[1]);
    try expectEqual(@as(i32, 0), predictions[2]);
    try expectEqual(@as(i32, 1), predictions[3]);

    // Check accuracy
    const accuracy = try model.score(&X, &y);
    try expectApproxEqAbs(@as(f64, 1.0), accuracy, 1e-6);
}

test "Perceptron: binary classification with f32" {
    const allocator = testing.allocator;

    var X = [_][]const f32{
        &[_]f32{ 0.0, 0.0 },
        &[_]f32{ 0.0, 1.0 },
        &[_]f32{ 1.0, 0.0 },
        &[_]f32{ 1.0, 1.0 },
    };
    var y = [_]i32{ 0, 0, 0, 1 };

    var model = Perceptron(f32).init(allocator, 0.1, 100, 42, false);
    defer model.deinit();

    try model.fit(&X, &y);

    const accuracy = try model.score(&X, &y);
    try expectApproxEqAbs(@as(f32, 1.0), accuracy, 1e-5);
}

test "Perceptron: multiclass classification (One-vs-Rest)" {
    const allocator = testing.allocator;

    // 3 classes: cluster around (0,0), (2,2), (4,0)
    var X = [_][]const f64{
        &[_]f64{ 0.1, 0.1 }, // class 0
        &[_]f64{ -0.1, 0.0 }, // class 0
        &[_]f64{ 2.0, 2.1 }, // class 1
        &[_]f64{ 2.1, 1.9 }, // class 1
        &[_]f64{ 4.0, 0.1 }, // class 2
        &[_]f64{ 3.9, -0.1 }, // class 2
    };
    var y = [_]i32{ 0, 0, 1, 1, 2, 2 };

    var model = Perceptron(f64).init(allocator, 0.1, 100, 42, false);
    defer model.deinit();

    try model.fit(&X, &y);

    // Should learn to separate 3 classes
    const accuracy = try model.score(&X, &y);
    try std.testing.expect(accuracy >= 0.8); // Allow some margin for randomness
}

test "Perceptron: averaged perceptron (more stable)" {
    const allocator = testing.allocator;

    var X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 0.0, 1.0 },
        &[_]f64{ 1.0, 0.0 },
        &[_]f64{ 1.0, 1.0 },
    };
    var y = [_]i32{ 0, 0, 0, 1 };

    var model = Perceptron(f64).init(allocator, 0.1, 100, 42, true);
    defer model.deinit();

    try model.fit(&X, &y);

    const accuracy = try model.score(&X, &y);
    try expectApproxEqAbs(@as(f64, 1.0), accuracy, 1e-6);
}

test "Perceptron: learning rate effect" {
    const allocator = testing.allocator;

    var X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 0.0, 1.0 },
        &[_]f64{ 1.0, 0.0 },
        &[_]f64{ 1.0, 1.0 },
    };
    var y = [_]i32{ 0, 0, 0, 1 };

    // High learning rate
    var model1 = Perceptron(f64).init(allocator, 1.0, 100, 42, false);
    defer model1.deinit();
    try model1.fit(&X, &y);
    const acc1 = try model1.score(&X, &y);

    // Low learning rate
    var model2 = Perceptron(f64).init(allocator, 0.01, 100, 42, false);
    defer model2.deinit();
    try model2.fit(&X, &y);
    const acc2 = try model2.score(&X, &y);

    // Both should learn perfectly for linearly separable problem
    try expectApproxEqAbs(@as(f64, 1.0), acc1, 1e-6);
    try expectApproxEqAbs(@as(f64, 1.0), acc2, 1e-6);
}

test "Perceptron: getWeights and getBias" {
    const allocator = testing.allocator;

    var X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 1.0, 1.0 },
    };
    var y = [_]i32{ 0, 1 };

    var model = Perceptron(f64).init(allocator, 0.1, 100, 42, false);
    defer model.deinit();

    try model.fit(&X, &y);

    // Check that we can retrieve weights and bias
    const weights0 = try model.getWeights(0);
    const bias0 = try model.getBias(0);

    try expectEqual(@as(usize, 2), weights0.len);
    try std.testing.expect(@abs(bias0) < 10.0); // Reasonable range

    // Invalid class index should error
    try expectError(error.InvalidClassIndex, model.getWeights(999));
    try expectError(error.InvalidClassIndex, model.getBias(999));
}

test "Perceptron: getNumClasses and getClasses" {
    const allocator = testing.allocator;

    var X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 2.0, 2.0 },
    };
    var y = [_]i32{ 0, 1, 2 };

    var model = Perceptron(f64).init(allocator, 0.1, 100, 42, false);
    defer model.deinit();

    try model.fit(&X, &y);

    try expectEqual(@as(usize, 3), model.getNumClasses());

    const classes = model.getClasses();
    try expectEqual(@as(usize, 3), classes.len);
    try expectEqual(@as(i32, 0), classes[0]);
    try expectEqual(@as(i32, 1), classes[1]);
    try expectEqual(@as(i32, 2), classes[2]);
}

test "Perceptron: empty data error" {
    const allocator = testing.allocator;
    var model = Perceptron(f64).init(allocator, 0.1, 100, 42, false);
    defer model.deinit();

    var X = [_][]const f64{};
    var y = [_]i32{};

    try expectError(error.EmptyData, model.fit(&X, &y));
}

test "Perceptron: mismatched dimensions error" {
    const allocator = testing.allocator;
    var model = Perceptron(f64).init(allocator, 0.1, 100, 42, false);
    defer model.deinit();

    var X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
    };
    var y = [_]i32{ 0, 1 }; // Mismatch: X has 1 sample, y has 2 labels

    try expectError(error.MismatchedDimensions, model.fit(&X, &y));
}

test "Perceptron: predict before fit error" {
    const allocator = testing.allocator;
    var model = Perceptron(f64).init(allocator, 0.1, 100, 42, false);
    defer model.deinit();

    var X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
    };

    try expectError(error.NotFitted, model.predict(&X));
}

test "Perceptron: feature mismatch in predict" {
    const allocator = testing.allocator;

    var X_train = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 1.0, 1.0 },
    };
    var y = [_]i32{ 0, 1 };

    var model = Perceptron(f64).init(allocator, 0.1, 100, 42, false);
    defer model.deinit();

    try model.fit(&X_train, &y);

    // Try to predict with different number of features
    var X_test = [_][]const f64{
        &[_]f64{ 0.0, 0.0, 0.0 }, // 3 features instead of 2
    };

    try expectError(error.MismatchedDimensions, model.predict(&X_test));
}

test "Perceptron: insufficient classes error" {
    const allocator = testing.allocator;

    var X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 1.0, 1.0 },
    };
    var y = [_]i32{ 0, 0 }; // Only one class

    var model = Perceptron(f64).init(allocator, 0.1, 100, 42, false);
    defer model.deinit();

    try expectError(error.InsufficientClasses, model.fit(&X, &y));
}

test "Perceptron: large dataset stress test" {
    const allocator = testing.allocator;

    // Generate 200 samples with 2 linearly separable classes
    var X_list = try ArrayList([]f64).initCapacity(allocator, 200);
    defer {
        for (X_list.items) |x| {
            allocator.free(x);
        }
        X_list.deinit();
    }
    var y_list = try ArrayList(i32).initCapacity(allocator, 200);
    defer y_list.deinit();

    var rng = std.rand.DefaultPrng.init(123);
    var random = rng.random();

    for (0..200) |i| {
        var x = try allocator.alloc(f64, 2);
        if (i < 100) {
            // Class 0: centered at (0, 0)
            x[0] = random.float(f64) - 0.5;
            x[1] = random.float(f64) - 0.5;
            try y_list.append(0);
        } else {
            // Class 1: centered at (2, 2)
            x[0] = 2.0 + random.float(f64) - 0.5;
            x[1] = 2.0 + random.float(f64) - 0.5;
            try y_list.append(1);
        }
        try X_list.append(x);
    }

    var model = Perceptron(f64).init(allocator, 0.1, 50, 42, false);
    defer model.deinit();

    try model.fit(X_list.items, y_list.items);

    const accuracy = try model.score(X_list.items, y_list.items);
    try std.testing.expect(accuracy >= 0.9); // Should achieve high accuracy
}

test "Perceptron: memory safety with testing.allocator" {
    const allocator = testing.allocator;

    var X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 1.0, 1.0 },
    };
    var y = [_]i32{ 0, 1 };

    var model = Perceptron(f64).init(allocator, 0.1, 100, 42, false);
    defer model.deinit();

    try model.fit(&X, &y);

    const predictions = try model.predict(&X);
    defer allocator.free(predictions);
}
