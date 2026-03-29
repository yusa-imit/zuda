const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const math = std.math;

/// Linear Regression - linear model for continuous output prediction using
/// ordinary least squares (OLS) or gradient descent optimization.
///
/// Algorithm:
/// - Model: y = w·x + b where w = weights, b = bias/intercept
/// - OLS solution: w = (X^T X)^(-1) X^T y (closed-form, exact)
/// - Gradient descent: iterative optimization (approximate)
/// - Loss: Mean Squared Error (MSE) = (1/n) Σ(y_pred - y_true)²
///
/// Time complexity:
/// - OLS: O(n × m² + m³) where n = samples, m = features (matrix inversion)
/// - Gradient descent: O(n_iter × n × m)
/// - Prediction: O(m) per sample
///
/// Space complexity: O(m) for storing weights + bias
///
/// Features:
/// - Closed-form OLS solution (exact, fast for small m)
/// - Gradient descent with L2 regularization (Ridge regression)
/// - Multiple evaluation metrics (MSE, RMSE, R², MAE)
/// - Coefficient analysis and feature importance
///
/// Use cases:
/// - Price prediction: real estate, stock prices, commodities
/// - Sales forecasting: revenue, demand prediction
/// - Scientific modeling: physics, chemistry, engineering
/// - Trend analysis: time series, growth rates
/// - Resource estimation: capacity planning, load prediction
pub fn LinearRegression(comptime T: type) type {
    if (T != f32 and T != f64) {
        @compileError("LinearRegression only supports f32 and f64");
    }

    return struct {
        const Self = @This();

        allocator: Allocator,
        /// Weight vector: [n_features]
        weights: ArrayList(T),
        /// Bias/intercept term
        bias: T,
        /// Number of features
        n_features: usize,
        /// Whether model has been trained
        trained: bool,

        /// Configuration for gradient descent training
        pub const GDConfig = struct {
            learning_rate: T = 0.01,
            max_iterations: usize = 1000,
            tolerance: T = 1e-6,
            l2_lambda: T = 0.0, // L2 regularization (Ridge)
        };

        /// Initialize empty linear regression model
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator) Self {
            return .{
                .allocator = allocator,
                .weights = ArrayList(T).init(allocator),
                .bias = 0.0,
                .n_features = 0,
                .trained = false,
            };
        }

        /// Free all allocated memory
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.weights.deinit();
        }

        /// Train using Ordinary Least Squares (closed-form solution)
        /// X: [n_samples][n_features], y: [n_samples]
        /// Time: O(n × m² + m³) | Space: O(m²)
        pub fn fitOLS(self: *Self, X: []const []const T, y: []const T) !void {
            if (X.len == 0 or y.len == 0 or X.len != y.len) {
                return error.InvalidInput;
            }
            const n = X.len;
            const m = X[0].len;
            self.n_features = m;

            // Build normal equation: (X^T X) w = X^T y
            // For simplicity with intercept, we augment X with a column of 1s
            // Alternative: center data and compute intercept separately

            // Compute X^T X (Gram matrix)
            var XtX = try self.allocator.alloc([]T, m);
            defer {
                for (XtX) |row| {
                    self.allocator.free(row);
                }
                self.allocator.free(XtX);
            }
            for (0..m) |i| {
                XtX[i] = try self.allocator.alloc(T, m);
                @memset(XtX[i], 0);
            }

            for (0..m) |i| {
                for (0..m) |j| {
                    var sum: T = 0;
                    for (0..n) |k| {
                        sum += X[k][i] * X[k][j];
                    }
                    XtX[i][j] = sum;
                }
            }

            // Compute X^T y
            var Xty = try self.allocator.alloc(T, m);
            defer self.allocator.free(Xty);
            for (0..m) |i| {
                var sum: T = 0;
                for (0..n) |k| {
                    sum += X[k][i] * y[k];
                }
                Xty[i] = sum;
            }

            // Solve XtX * w = Xty using Gaussian elimination with partial pivoting
            var w = try self.allocator.alloc(T, m);
            defer self.allocator.free(w);

            // Augmented matrix [XtX | Xty]
            var aug = try self.allocator.alloc([]T, m);
            defer {
                for (aug) |row| {
                    self.allocator.free(row);
                }
                self.allocator.free(aug);
            }
            for (0..m) |i| {
                aug[i] = try self.allocator.alloc(T, m + 1);
                for (0..m) |j| {
                    aug[i][j] = XtX[i][j];
                }
                aug[i][m] = Xty[i];
            }

            // Gaussian elimination with partial pivoting
            for (0..m) |k| {
                // Find pivot
                var max_idx = k;
                var max_val = @abs(aug[k][k]);
                for (k + 1..m) |i| {
                    const val = @abs(aug[i][k]);
                    if (val > max_val) {
                        max_val = val;
                        max_idx = i;
                    }
                }

                if (max_val < 1e-10) {
                    return error.SingularMatrix;
                }

                // Swap rows
                if (max_idx != k) {
                    const tmp = aug[k];
                    aug[k] = aug[max_idx];
                    aug[max_idx] = tmp;
                }

                // Eliminate
                for (k + 1..m) |i| {
                    const factor = aug[i][k] / aug[k][k];
                    for (k..m + 1) |j| {
                        aug[i][j] -= factor * aug[k][j];
                    }
                }
            }

            // Back substitution
            var i: usize = m;
            while (i > 0) {
                i -= 1;
                var sum: T = aug[i][m];
                for (i + 1..m) |j| {
                    sum -= aug[i][j] * w[j];
                }
                w[i] = sum / aug[i][i];
            }

            // Store weights
            try self.weights.resize(m);
            @memcpy(self.weights.items, w);

            // Compute bias (intercept) as mean(y) - mean(X) · w
            var mean_y: T = 0;
            for (y) |val| mean_y += val;
            mean_y /= @as(T, @floatFromInt(n));

            var mean_X = try self.allocator.alloc(T, m);
            defer self.allocator.free(mean_X);
            for (0..m) |j| {
                var sum: T = 0;
                for (0..n) |k| sum += X[k][j];
                mean_X[j] = sum / @as(T, @floatFromInt(n));
            }

            var bias_correction: T = 0;
            for (0..m) |j| {
                bias_correction += mean_X[j] * self.weights.items[j];
            }
            self.bias = mean_y - bias_correction;
            self.trained = true;
        }

        /// Train using Gradient Descent
        /// X: [n_samples][n_features], y: [n_samples]
        /// Time: O(n_iter × n × m) | Space: O(m)
        pub fn fitGD(self: *Self, X: []const []const T, y: []const T, config: GDConfig) !void {
            if (X.len == 0 or y.len == 0 or X.len != y.len) {
                return error.InvalidInput;
            }
            const n = X.len;
            const m = X[0].len;
            self.n_features = m;

            // Initialize weights and bias
            try self.weights.resize(m);
            @memset(self.weights.items, 0);
            self.bias = 0;

            const n_float = @as(T, @floatFromInt(n));

            for (0..config.max_iterations) |iter| {
                _ = iter;

                // Compute gradients
                var grad_w = try self.allocator.alloc(T, m);
                defer self.allocator.free(grad_w);
                @memset(grad_w, 0);
                var grad_b: T = 0;

                for (0..n) |i| {
                    const pred = try self.predictSample(X[i]);
                    const error_i = pred - y[i];
                    grad_b += error_i;
                    for (0..m) |j| {
                        grad_w[j] += error_i * X[i][j];
                    }
                }

                // Average gradients and apply L2 regularization
                for (0..m) |j| {
                    grad_w[j] = grad_w[j] / n_float + config.l2_lambda * self.weights.items[j];
                }
                grad_b /= n_float;

                // Update parameters
                for (0..m) |j| {
                    self.weights.items[j] -= config.learning_rate * grad_w[j];
                }
                self.bias -= config.learning_rate * grad_b;

                // Check convergence
                var grad_norm: T = 0;
                for (grad_w) |g| grad_norm += g * g;
                grad_norm += grad_b * grad_b;
                grad_norm = @sqrt(grad_norm);

                if (grad_norm < config.tolerance) {
                    break;
                }
            }

            self.trained = true;
        }

        /// Predict single sample
        /// Time: O(m) | Space: O(1)
        fn predictSample(self: *const Self, x: []const T) !T {
            if (!self.trained) return error.ModelNotTrained;
            if (x.len != self.n_features) return error.FeatureMismatch;

            var result = self.bias;
            for (0..self.n_features) |i| {
                result += self.weights.items[i] * x[i];
            }
            return result;
        }

        /// Predict multiple samples
        /// X: [n_samples][n_features]
        /// Returns: [n_samples] predictions (caller owns memory)
        /// Time: O(n × m) | Space: O(n)
        pub fn predict(self: *const Self, X: []const []const T, allocator: Allocator) ![]T {
            if (!self.trained) return error.ModelNotTrained;
            if (X.len == 0) return error.InvalidInput;

            const predictions = try allocator.alloc(T, X.len);
            for (X, 0..) |sample, i| {
                predictions[i] = try self.predictSample(sample);
            }
            return predictions;
        }

        /// Evaluate model using Mean Squared Error (MSE)
        /// Time: O(n × m) | Space: O(n)
        pub fn score(self: *const Self, X: []const []const T, y: []const T, allocator: Allocator) !T {
            const predictions = try self.predict(X, allocator);
            defer allocator.free(predictions);

            var mse: T = 0;
            for (predictions, y) |pred, true_val| {
                const diff = pred - true_val;
                mse += diff * diff;
            }
            return mse / @as(T, @floatFromInt(predictions.len));
        }

        /// Compute R² (coefficient of determination)
        /// R² = 1 - (SS_res / SS_tot) where SS_res = Σ(y - ŷ)², SS_tot = Σ(y - ȳ)²
        /// R² = 1 means perfect fit, R² = 0 means model = mean, R² < 0 means worse than mean
        /// Time: O(n × m) | Space: O(n)
        pub fn r2Score(self: *const Self, X: []const []const T, y: []const T, allocator: Allocator) !T {
            const predictions = try self.predict(X, allocator);
            defer allocator.free(predictions);

            // Compute mean of y
            var mean_y: T = 0;
            for (y) |val| mean_y += val;
            mean_y /= @as(T, @floatFromInt(y.len));

            // Compute SS_res and SS_tot
            var ss_res: T = 0;
            var ss_tot: T = 0;
            for (predictions, y) |pred, true_val| {
                const res = true_val - pred;
                const tot = true_val - mean_y;
                ss_res += res * res;
                ss_tot += tot * tot;
            }

            if (ss_tot < 1e-10) return 0; // All y values are the same
            return 1.0 - (ss_res / ss_tot);
        }

        /// Compute Mean Absolute Error (MAE)
        /// Time: O(n × m) | Space: O(n)
        pub fn mae(self: *const Self, X: []const []const T, y: []const T, allocator: Allocator) !T {
            const predictions = try self.predict(X, allocator);
            defer allocator.free(predictions);

            var total: T = 0;
            for (predictions, y) |pred, true_val| {
                total += @abs(pred - true_val);
            }
            return total / @as(T, @floatFromInt(predictions.len));
        }

        /// Compute Root Mean Squared Error (RMSE)
        /// Time: O(n × m) | Space: O(n)
        pub fn rmse(self: *const Self, X: []const []const T, y: []const T, allocator: Allocator) !T {
            const mse_val = try self.score(X, y, allocator);
            return @sqrt(mse_val);
        }

        /// Get feature coefficients (weights)
        pub fn coefficients(self: *const Self) []const T {
            return self.weights.items;
        }

        /// Get intercept (bias)
        pub fn intercept(self: *const Self) T {
            return self.bias;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "LinearRegression: basic OLS fit and predict" {
    const allocator = std.testing.allocator;

    // Simple linear relationship: y = 2x + 1
    var X_data = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
        &[_]f64{4.0},
    };
    const y_data = [_]f64{ 3.0, 5.0, 7.0, 9.0 };

    var model = LinearRegression(f64).init(allocator);
    defer model.deinit();

    try model.fitOLS(&X_data, &y_data);

    // Check coefficients: should be close to w=2, b=1
    try std.testing.expect(@abs(model.weights.items[0] - 2.0) < 0.01);
    try std.testing.expect(@abs(model.bias - 1.0) < 0.01);

    // Predict
    const pred = try model.predict(&X_data, allocator);
    defer allocator.free(pred);

    for (pred, y_data) |p, y| {
        try std.testing.expect(@abs(p - y) < 0.01);
    }
}

test "LinearRegression: gradient descent fit" {
    const allocator = std.testing.allocator;

    var X_data = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
        &[_]f64{4.0},
    };
    const y_data = [_]f64{ 3.0, 5.0, 7.0, 9.0 };

    var model = LinearRegression(f64).init(allocator);
    defer model.deinit();

    const config = LinearRegression(f64).GDConfig{
        .learning_rate = 0.01,
        .max_iterations = 1000,
        .tolerance = 1e-6,
        .l2_lambda = 0.0,
    };

    try model.fitGD(&X_data, &y_data, config);

    // Check coefficients: should be close to w=2, b=1
    try std.testing.expect(@abs(model.weights.items[0] - 2.0) < 0.1);
    try std.testing.expect(@abs(model.bias - 1.0) < 0.1);
}

test "LinearRegression: multivariate OLS" {
    const allocator = std.testing.allocator;

    // y = 2x1 + 3x2 + 1
    var X_data = [_][]const f64{
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 2.0, 1.0 },
        &[_]f64{ 3.0, 2.0 },
        &[_]f64{ 4.0, 3.0 },
    };
    const y_data = [_]f64{ 6.0, 8.0, 13.0, 18.0 };

    var model = LinearRegression(f64).init(allocator);
    defer model.deinit();

    try model.fitOLS(&X_data, &y_data);

    // Check coefficients
    try std.testing.expect(@abs(model.weights.items[0] - 2.0) < 0.1);
    try std.testing.expect(@abs(model.weights.items[1] - 3.0) < 0.1);
    try std.testing.expect(@abs(model.bias - 1.0) < 0.1);
}

test "LinearRegression: R² score perfect fit" {
    const allocator = std.testing.allocator;

    var X_data = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
        &[_]f64{4.0},
    };
    const y_data = [_]f64{ 3.0, 5.0, 7.0, 9.0 };

    var model = LinearRegression(f64).init(allocator);
    defer model.deinit();

    try model.fitOLS(&X_data, &y_data);

    const r2 = try model.r2Score(&X_data, &y_data, allocator);
    try std.testing.expect(r2 > 0.99); // Nearly perfect fit
}

test "LinearRegression: MSE and RMSE" {
    const allocator = std.testing.allocator;

    var X_data = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
        &[_]f64{4.0},
    };
    const y_data = [_]f64{ 3.1, 5.0, 6.9, 9.0 }; // Slight noise

    var model = LinearRegression(f64).init(allocator);
    defer model.deinit();

    try model.fitOLS(&X_data, &y_data);

    const mse = try model.score(&X_data, &y_data, allocator);
    const rmse_val = try model.rmse(&X_data, &y_data, allocator);

    try std.testing.expect(mse < 0.01); // Small error
    try std.testing.expect(rmse_val < 0.1);
    try std.testing.expect(@abs(rmse_val - @sqrt(mse)) < 1e-10);
}

test "LinearRegression: MAE" {
    const allocator = std.testing.allocator;

    var X_data = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
        &[_]f64{4.0},
    };
    const y_data = [_]f64{ 3.0, 5.0, 7.0, 9.0 };

    var model = LinearRegression(f64).init(allocator);
    defer model.deinit();

    try model.fitOLS(&X_data, &y_data);

    const mae_val = try model.mae(&X_data, &y_data, allocator);
    try std.testing.expect(mae_val < 0.01);
}

test "LinearRegression: empty input error" {
    const allocator = std.testing.allocator;

    var model = LinearRegression(f64).init(allocator);
    defer model.deinit();

    var X_empty = [_][]const f64{};
    const y_empty = [_]f64{};

    const result = model.fitOLS(&X_empty, &y_empty);
    try std.testing.expectError(error.InvalidInput, result);
}

test "LinearRegression: prediction before training error" {
    const allocator = std.testing.allocator;

    var model = LinearRegression(f64).init(allocator);
    defer model.deinit();

    var X_data = [_][]const f64{&[_]f64{1.0}};
    const result = model.predict(&X_data, allocator);
    try std.testing.expectError(error.ModelNotTrained, result);
}

test "LinearRegression: feature mismatch error" {
    const allocator = std.testing.allocator;

    var X_train = [_][]const f64{
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 2.0, 3.0 },
    };
    const y_train = [_]f64{ 5.0, 8.0 };

    var model = LinearRegression(f64).init(allocator);
    defer model.deinit();

    try model.fitOLS(&X_train, &y_train);

    // Try to predict with wrong number of features
    var X_test = [_][]const f64{&[_]f64{1.0}}; // Only 1 feature
    const result = model.predict(&X_test, allocator);
    try std.testing.expectError(error.FeatureMismatch, result);
}

test "LinearRegression: Ridge regularization (L2)" {
    const allocator = std.testing.allocator;

    // Noisy data with potential overfitting
    var X_data = [_][]const f64{
        &[_]f64{ 1.0, 0.5 },
        &[_]f64{ 2.0, 1.0 },
        &[_]f64{ 3.0, 1.5 },
        &[_]f64{ 4.0, 2.0 },
    };
    const y_data = [_]f64{ 3.2, 5.1, 6.9, 9.1 };

    var model = LinearRegression(f64).init(allocator);
    defer model.deinit();

    const config = LinearRegression(f64).GDConfig{
        .learning_rate = 0.01,
        .max_iterations = 2000,
        .tolerance = 1e-6,
        .l2_lambda = 0.1, // Ridge regularization
    };

    try model.fitGD(&X_data, &y_data, config);

    // With regularization, weights should be smaller
    for (model.weights.items) |w| {
        try std.testing.expect(@abs(w) < 10.0);
    }
}

test "LinearRegression: f32 type support" {
    const allocator = std.testing.allocator;

    var X_data = [_][]const f32{
        &[_]f32{1.0},
        &[_]f32{2.0},
        &[_]f32{3.0},
    };
    const y_data = [_]f32{ 3.0, 5.0, 7.0 };

    var model = LinearRegression(f32).init(allocator);
    defer model.deinit();

    try model.fitOLS(&X_data, &y_data);

    try std.testing.expect(@abs(model.weights.items[0] - 2.0) < 0.1);
}

test "LinearRegression: large dataset stress test" {
    const allocator = std.testing.allocator;

    const n = 1000;
    const m = 5;

    var X_list = std.ArrayList([]f64).init(allocator);
    defer {
        for (X_list.items) |row| {
            allocator.free(row);
        }
        X_list.deinit();
    }

    var y_list = std.ArrayList(f64).init(allocator);
    defer y_list.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // Generate synthetic data: y = Σ(i * x_i) + noise
    for (0..n) |_| {
        const row = try allocator.alloc(f64, m);
        for (0..m) |j| {
            row[j] = random.float(f64) * 10.0;
        }
        try X_list.append(row);

        var y_val: f64 = 0;
        for (0..m) |j| {
            y_val += @as(f64, @floatFromInt(j + 1)) * row[j];
        }
        y_val += random.float(f64) * 0.1; // Small noise
        try y_list.append(y_val);
    }

    var model = LinearRegression(f64).init(allocator);
    defer model.deinit();

    try model.fitOLS(X_list.items, y_list.items);

    const r2 = try model.r2Score(X_list.items, y_list.items, allocator);
    try std.testing.expect(r2 > 0.95); // Good fit despite noise
}

test "LinearRegression: coefficients and intercept getters" {
    const allocator = std.testing.allocator;

    var X_data = [_][]const f64{
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 2.0, 3.0 },
    };
    const y_data = [_]f64{ 8.0, 13.0 };

    var model = LinearRegression(f64).init(allocator);
    defer model.deinit();

    try model.fitOLS(&X_data, &y_data);

    const coef = model.coefficients();
    const inter = model.intercept();

    try std.testing.expect(coef.len == 2);
    try std.testing.expect(inter != 0);
}

test "LinearRegression: convergence check in gradient descent" {
    const allocator = std.testing.allocator;

    var X_data = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
    };
    const y_data = [_]f64{ 2.0, 4.0, 6.0 };

    var model = LinearRegression(f64).init(allocator);
    defer model.deinit();

    const config = LinearRegression(f64).GDConfig{
        .learning_rate = 0.1,
        .max_iterations = 10000,
        .tolerance = 1e-8, // Very tight tolerance
        .l2_lambda = 0.0,
    };

    try model.fitGD(&X_data, &y_data, config);

    // Should converge to near-perfect solution
    try std.testing.expect(@abs(model.weights.items[0] - 2.0) < 0.01);
    try std.testing.expect(@abs(model.bias) < 0.01);
}
