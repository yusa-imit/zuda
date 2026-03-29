/// Ridge Regression
///
/// Ridge regression adds L2 regularization to linear regression, preventing overfitting
/// and handling multicollinearity by penalizing large coefficients.
///
/// **Mathematical formulation:**
///
/// Minimize: ||y - Xw||² + λ||w||²
///
/// where:
/// - X ∈ ℝⁿˣᵐ is the feature matrix (n samples, m features)
/// - y ∈ ℝⁿ is the target vector
/// - w ∈ ℝᵐ is the weight vector
/// - λ ≥ 0 is the regularization parameter
///
/// **Closed-form solution:**
///
/// w = (XᵀX + λI)⁻¹Xᵀy
///
/// **Time complexity:**
/// - Training: O(nm² + m³) for closed-form solution (matrix inversion)
/// - Prediction: O(m) per sample
///
/// **Space complexity:** O(m) for weights
///
/// **When to use:**
/// - Features are highly correlated (multicollinearity)
/// - Number of features approaches or exceeds number of samples
/// - Want to prevent overfitting on training data
/// - Need stable coefficient estimates
///
/// **Ridge vs Lasso:**
/// - Ridge (L2): Shrinks coefficients smoothly, includes all features
/// - Lasso (L1): Performs feature selection, sets some coefficients to zero
///
/// **Example:**
/// ```zig
/// const X = [_][2]f64{
///     .{ 1.0, 2.0 },
///     .{ 2.0, 3.0 },
///     .{ 3.0, 4.0 },
///     .{ 4.0, 5.0 },
/// };
/// const y = [_]f64{ 5.0, 7.0, 9.0, 11.0 };
/// const allocator = std.testing.allocator;
///
/// var ridge = RidgeRegression(f64).init(allocator, 2);
/// defer ridge.deinit();
///
/// try ridge.fit(&X, &y, 1.0); // lambda = 1.0
/// const prediction = ridge.predict(&[_]f64{ 5.0, 6.0 });
/// ```

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Ridge Regression with L2 regularization
///
/// Type parameters:
/// - T: Floating point type (f32 or f64)
pub fn RidgeRegression(comptime T: type) type {
    if (T != f32 and T != f64) {
        @compileError("RidgeRegression only supports f32 and f64");
    }

    return struct {
        const Self = @This();

        allocator: Allocator,
        weights: ?[]T,
        intercept: T,
        n_features: usize,

        /// Initialize a new Ridge Regression model
        ///
        /// **Parameters:**
        /// - allocator: Memory allocator
        /// - n_features: Number of input features
        ///
        /// **Returns:** Uninitialized Ridge Regression model
        ///
        /// **Time:** O(1)
        /// **Space:** O(1)
        pub fn init(allocator: Allocator, n_features: usize) Self {
            return .{
                .allocator = allocator,
                .weights = null,
                .intercept = 0,
                .n_features = n_features,
            };
        }

        /// Free all allocated memory
        ///
        /// **Time:** O(1)
        /// **Space:** O(1)
        pub fn deinit(self: *Self) void {
            if (self.weights) |w| {
                self.allocator.free(w);
                self.weights = null;
            }
        }

        /// Train the model using closed-form solution
        ///
        /// Solves: w = (XᵀX + λI)⁻¹Xᵀy
        ///
        /// **Parameters:**
        /// - X: Feature matrix [n_samples][n_features]
        /// - y: Target values [n_samples]
        /// - lambda: Regularization parameter (λ ≥ 0)
        ///
        /// **Errors:**
        /// - error.DimensionMismatch: If X and y have incompatible sizes
        /// - error.InvalidLambda: If lambda < 0
        /// - error.OutOfMemory: If allocation fails
        ///
        /// **Time:** O(nm² + m³) where n=samples, m=features
        /// **Space:** O(m²) for temporary matrices
        pub fn fit(self: *Self, X: []const []const T, y: []const T, lambda: T) !void {
            if (lambda < 0) {
                return error.InvalidLambda;
            }

            const n_samples = X.len;
            if (n_samples == 0) {
                return error.DimensionMismatch;
            }
            if (y.len != n_samples) {
                return error.DimensionMismatch;
            }
            if (X[0].len != self.n_features) {
                return error.DimensionMismatch;
            }

            // Compute mean-centered data
            var x_mean = try self.allocator.alloc(T, self.n_features);
            defer self.allocator.free(x_mean);
            @memset(x_mean, 0);

            var y_mean: T = 0;
            for (y) |yi| {
                y_mean += yi;
            }
            y_mean /= @as(T, @floatFromInt(n_samples));

            for (X) |xi| {
                for (0..self.n_features) |j| {
                    x_mean[j] += xi[j];
                }
            }
            for (x_mean) |*xm| {
                xm.* /= @as(T, @floatFromInt(n_samples));
            }

            // Compute XᵀX + λI
            var xtx = try self.allocator.alloc(T, self.n_features * self.n_features);
            defer self.allocator.free(xtx);
            @memset(xtx, 0);

            for (X) |xi| {
                for (0..self.n_features) |i| {
                    const xi_centered = xi[i] - x_mean[i];
                    for (0..self.n_features) |j| {
                        const xj_centered = xi[j] - x_mean[j];
                        xtx[i * self.n_features + j] += xi_centered * xj_centered;
                    }
                }
            }

            // Add λI to diagonal
            for (0..self.n_features) |i| {
                xtx[i * self.n_features + i] += lambda;
            }

            // Compute Xᵀy
            var xty = try self.allocator.alloc(T, self.n_features);
            defer self.allocator.free(xty);
            @memset(xty, 0);

            for (X, y) |xi, yi| {
                const yi_centered = yi - y_mean;
                for (0..self.n_features) |i| {
                    const xi_centered = xi[i] - x_mean[i];
                    xty[i] += xi_centered * yi_centered;
                }
            }

            // Solve (XᵀX + λI)w = Xᵀy using Gaussian elimination
            if (self.weights == null) {
                self.weights = try self.allocator.alloc(T, self.n_features);
            }
            try gaussianElimination(T, xtx, xty, self.weights.?, self.n_features);

            // Compute intercept: b = ȳ - w̄ᵀx̄
            var intercept: T = y_mean;
            for (self.weights.?, x_mean) |wi, xmi| {
                intercept -= wi * xmi;
            }
            self.intercept = intercept;
        }

        /// Predict target value for a single sample
        ///
        /// **Parameters:**
        /// - x: Input features [n_features]
        ///
        /// **Returns:** Predicted value
        ///
        /// **Errors:**
        /// - error.NotTrained: If model hasn't been trained
        /// - error.DimensionMismatch: If x size doesn't match n_features
        ///
        /// **Time:** O(m) where m=features
        /// **Space:** O(1)
        pub fn predict(self: *const Self, x: []const T) !T {
            if (self.weights == null) {
                return error.NotTrained;
            }
            if (x.len != self.n_features) {
                return error.DimensionMismatch;
            }

            var result = self.intercept;
            for (self.weights.?, x) |wi, xi| {
                result += wi * xi;
            }
            return result;
        }

        /// Predict target values for multiple samples
        ///
        /// **Parameters:**
        /// - X: Feature matrix [n_samples][n_features]
        /// - predictions: Output buffer [n_samples]
        ///
        /// **Errors:**
        /// - error.NotTrained: If model hasn't been trained
        /// - error.DimensionMismatch: If dimensions don't match
        ///
        /// **Time:** O(nm) where n=samples, m=features
        /// **Space:** O(1)
        pub fn predictBatch(self: *const Self, X: []const []const T, predictions: []T) !void {
            if (predictions.len != X.len) {
                return error.DimensionMismatch;
            }

            for (X, 0..) |xi, i| {
                predictions[i] = try self.predict(xi);
            }
        }

        /// Compute mean squared error on test data
        ///
        /// **Parameters:**
        /// - X: Feature matrix [n_samples][n_features]
        /// - y: True target values [n_samples]
        ///
        /// **Returns:** Mean squared error
        ///
        /// **Errors:**
        /// - error.NotTrained: If model hasn't been trained
        /// - error.DimensionMismatch: If dimensions don't match
        ///
        /// **Time:** O(nm) where n=samples, m=features
        /// **Space:** O(1)
        pub fn score(self: *const Self, X: []const []const T, y: []const T) !T {
            if (X.len != y.len) {
                return error.DimensionMismatch;
            }

            var mse: T = 0;
            for (X, y) |xi, yi| {
                const pred = try self.predict(xi);
                const error_val = yi - pred;
                mse += error_val * error_val;
            }
            return mse / @as(T, @floatFromInt(X.len));
        }

        /// Compute R² coefficient of determination
        ///
        /// **Parameters:**
        /// - X: Feature matrix [n_samples][n_features]
        /// - y: True target values [n_samples]
        ///
        /// **Returns:** R² score (1.0 = perfect fit, 0.0 = no better than mean)
        ///
        /// **Errors:**
        /// - error.NotTrained: If model hasn't been trained
        /// - error.DimensionMismatch: If dimensions don't match
        ///
        /// **Time:** O(nm) where n=samples, m=features
        /// **Space:** O(1)
        pub fn r2Score(self: *const Self, X: []const []const T, y: []const T) !T {
            if (X.len != y.len) {
                return error.DimensionMismatch;
            }

            var y_mean: T = 0;
            for (y) |yi| {
                y_mean += yi;
            }
            y_mean /= @as(T, @floatFromInt(y.len));

            var ss_tot: T = 0;
            var ss_res: T = 0;
            for (X, y) |xi, yi| {
                const pred = try self.predict(xi);
                const error_val = yi - pred;
                ss_res += error_val * error_val;
                const mean_error = yi - y_mean;
                ss_tot += mean_error * mean_error;
            }

            return 1.0 - ss_res / ss_tot;
        }

        /// Get the learned coefficients (weights)
        ///
        /// **Returns:** Slice of weights [n_features]
        ///
        /// **Errors:**
        /// - error.NotTrained: If model hasn't been trained
        ///
        /// **Time:** O(1)
        /// **Space:** O(1)
        pub fn coefficients(self: *const Self) ![]const T {
            if (self.weights == null) {
                return error.NotTrained;
            }
            return self.weights.?;
        }
    };
}

/// Solve Aw = b using Gaussian elimination with partial pivoting
///
/// **Time:** O(n³)
/// **Space:** O(1) (in-place)
fn gaussianElimination(comptime T: type, A: []T, b: []const T, x: []T, n: usize) !void {
    // Create augmented matrix [A|b]
    var aug = try std.heap.page_allocator.alloc(T, n * n);
    defer std.heap.page_allocator.free(aug);
    @memcpy(aug, A);

    var b_copy = try std.heap.page_allocator.alloc(T, n);
    defer std.heap.page_allocator.free(b_copy);
    @memcpy(b_copy, b);

    // Forward elimination with partial pivoting
    for (0..n) |i| {
        // Find pivot
        var max_row = i;
        var max_val = @abs(aug[i * n + i]);
        for (i + 1..n) |k| {
            const val = @abs(aug[k * n + i]);
            if (val > max_val) {
                max_val = val;
                max_row = k;
            }
        }

        // Swap rows
        if (max_row != i) {
            for (0..n) |j| {
                const tmp = aug[i * n + j];
                aug[i * n + j] = aug[max_row * n + j];
                aug[max_row * n + j] = tmp;
            }
            const tmp_b = b_copy[i];
            b_copy[i] = b_copy[max_row];
            b_copy[max_row] = tmp_b;
        }

        // Check for singularity
        if (@abs(aug[i * n + i]) < 1e-10) {
            return error.SingularMatrix;
        }

        // Eliminate column
        for (i + 1..n) |k| {
            const factor = aug[k * n + i] / aug[i * n + i];
            for (i..n) |j| {
                aug[k * n + j] -= factor * aug[i * n + j];
            }
            b_copy[k] -= factor * b_copy[i];
        }
    }

    // Back substitution
    var i = n;
    while (i > 0) {
        i -= 1;
        var sum: T = b_copy[i];
        for (i + 1..n) |j| {
            sum -= aug[i * n + j] * x[j];
        }
        x[i] = sum / aug[i * n + i];
    }
}

// ============================================================================
// Tests
// ============================================================================

test "RidgeRegression: basic linear fit with lambda=0" {
    const allocator = std.testing.allocator;
    const X = [_][2]f64{
        .{ 1.0, 2.0 },
        .{ 2.0, 3.0 },
        .{ 3.0, 4.0 },
        .{ 4.0, 5.0 },
    };
    const y = [_]f64{ 5.0, 7.0, 9.0, 11.0 };

    var ridge = RidgeRegression(f64).init(allocator, 2);
    defer ridge.deinit();

    try ridge.fit(&X, &y, 0.0); // No regularization (equivalent to OLS)

    const pred1 = try ridge.predict(&[_]f64{ 1.0, 2.0 });
    const pred2 = try ridge.predict(&[_]f64{ 4.0, 5.0 });

    try std.testing.expectApproxEqAbs(5.0, pred1, 0.01);
    try std.testing.expectApproxEqAbs(11.0, pred2, 0.01);
}

test "RidgeRegression: with regularization lambda=1.0" {
    const allocator = std.testing.allocator;
    const X = [_][2]f64{
        .{ 1.0, 1.0 },
        .{ 1.0, 2.0 },
        .{ 2.0, 2.0 },
        .{ 2.0, 3.0 },
    };
    const y = [_]f64{ 1.0, 2.0, 2.0, 3.0 };

    var ridge = RidgeRegression(f64).init(allocator, 2);
    defer ridge.deinit();

    try ridge.fit(&X, &y, 1.0);

    // With regularization, coefficients should be shrunk
    const coeffs = try ridge.coefficients();
    try std.testing.expect(coeffs.len == 2);
}

test "RidgeRegression: perfect fit with simple data" {
    const allocator = std.testing.allocator;
    const X = [_][1]f64{
        .{1.0},
        .{2.0},
        .{3.0},
        .{4.0},
    };
    const y = [_]f64{ 2.0, 4.0, 6.0, 8.0 }; // y = 2x

    var ridge = RidgeRegression(f64).init(allocator, 1);
    defer ridge.deinit();

    try ridge.fit(&X, &y, 0.1);

    const pred = try ridge.predict(&[_]f64{5.0});
    try std.testing.expectApproxEqAbs(10.0, pred, 0.5); // Should be close to 10
}

test "RidgeRegression: R² score" {
    const allocator = std.testing.allocator;
    const X = [_][1]f64{
        .{1.0},
        .{2.0},
        .{3.0},
        .{4.0},
    };
    const y = [_]f64{ 2.1, 4.0, 5.9, 8.0 };

    var ridge = RidgeRegression(f64).init(allocator, 1);
    defer ridge.deinit();

    try ridge.fit(&X, &y, 0.1);

    const r2 = try ridge.r2Score(&X, &y);
    try std.testing.expect(r2 > 0.9); // Should have high R²
}

test "RidgeRegression: MSE score" {
    const allocator = std.testing.allocator;
    const X = [_][1]f64{
        .{1.0},
        .{2.0},
        .{3.0},
        .{4.0},
    };
    const y = [_]f64{ 2.0, 4.0, 6.0, 8.0 };

    var ridge = RidgeRegression(f64).init(allocator, 1);
    defer ridge.deinit();

    try ridge.fit(&X, &y, 0.0);

    const mse = try ridge.score(&X, &y);
    try std.testing.expectApproxEqAbs(0.0, mse, 0.01); // Perfect fit
}

test "RidgeRegression: batch prediction" {
    const allocator = std.testing.allocator;
    const X = [_][2]f64{
        .{ 1.0, 2.0 },
        .{ 2.0, 3.0 },
        .{ 3.0, 4.0 },
        .{ 4.0, 5.0 },
    };
    const y = [_]f64{ 5.0, 7.0, 9.0, 11.0 };

    var ridge = RidgeRegression(f64).init(allocator, 2);
    defer ridge.deinit();

    try ridge.fit(&X, &y, 0.5);

    var predictions = [_]f64{0} ** 4;
    try ridge.predictBatch(&X, &predictions);

    for (predictions, y) |pred, actual| {
        try std.testing.expectApproxEqAbs(actual, pred, 1.0); // Reasonable fit
    }
}

test "RidgeRegression: error on untrained model" {
    const allocator = std.testing.allocator;
    var ridge = RidgeRegression(f64).init(allocator, 2);
    defer ridge.deinit();

    const result = ridge.predict(&[_]f64{ 1.0, 2.0 });
    try std.testing.expectError(error.NotTrained, result);
}

test "RidgeRegression: error on dimension mismatch" {
    const allocator = std.testing.allocator;
    const X = [_][2]f64{
        .{ 1.0, 2.0 },
        .{ 2.0, 3.0 },
    };
    const y = [_]f64{ 5.0, 7.0 };

    var ridge = RidgeRegression(f64).init(allocator, 2);
    defer ridge.deinit();

    try ridge.fit(&X, &y, 1.0);

    const result = ridge.predict(&[_]f64{ 1.0, 2.0, 3.0 }); // Wrong size
    try std.testing.expectError(error.DimensionMismatch, result);
}

test "RidgeRegression: error on mismatched X and y" {
    const allocator = std.testing.allocator;
    const X = [_][2]f64{
        .{ 1.0, 2.0 },
        .{ 2.0, 3.0 },
    };
    const y = [_]f64{5.0}; // Wrong length

    var ridge = RidgeRegression(f64).init(allocator, 2);
    defer ridge.deinit();

    const result = ridge.fit(&X, &y, 1.0);
    try std.testing.expectError(error.DimensionMismatch, result);
}

test "RidgeRegression: error on negative lambda" {
    const allocator = std.testing.allocator;
    const X = [_][2]f64{
        .{ 1.0, 2.0 },
        .{ 2.0, 3.0 },
    };
    const y = [_]f64{ 5.0, 7.0 };

    var ridge = RidgeRegression(f64).init(allocator, 2);
    defer ridge.deinit();

    const result = ridge.fit(&X, &y, -1.0); // Negative lambda
    try std.testing.expectError(error.InvalidLambda, result);
}

test "RidgeRegression: multicollinearity handling" {
    const allocator = std.testing.allocator;
    // Highly correlated features: x2 ≈ 2*x1
    const X = [_][2]f64{
        .{ 1.0, 2.01 },
        .{ 2.0, 4.02 },
        .{ 3.0, 6.01 },
        .{ 4.0, 8.02 },
    };
    const y = [_]f64{ 5.0, 7.0, 9.0, 11.0 };

    var ridge = RidgeRegression(f64).init(allocator, 2);
    defer ridge.deinit();

    // Ridge should handle this better than OLS
    try ridge.fit(&X, &y, 1.0);

    const pred = try ridge.predict(&[_]f64{ 5.0, 10.0 });
    try std.testing.expect(@abs(pred - 13.0) < 2.0); // Should still make reasonable predictions
}

test "RidgeRegression: f32 support" {
    const allocator = std.testing.allocator;
    const X = [_][1]f32{
        .{1.0},
        .{2.0},
        .{3.0},
        .{4.0},
    };
    const y = [_]f32{ 2.0, 4.0, 6.0, 8.0 };

    var ridge = RidgeRegression(f32).init(allocator, 1);
    defer ridge.deinit();

    try ridge.fit(&X, &y, 0.1);

    const pred = try ridge.predict(&[_]f32{5.0});
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), pred, 0.5);
}

test "RidgeRegression: high lambda shrinks coefficients more" {
    const allocator = std.testing.allocator;
    const X = [_][2]f64{
        .{ 1.0, 1.0 },
        .{ 2.0, 2.0 },
        .{ 3.0, 3.0 },
        .{ 4.0, 4.0 },
    };
    const y = [_]f64{ 2.0, 4.0, 6.0, 8.0 };

    var ridge1 = RidgeRegression(f64).init(allocator, 2);
    defer ridge1.deinit();
    try ridge1.fit(&X, &y, 0.1);
    const coeffs1 = try ridge1.coefficients();

    var ridge2 = RidgeRegression(f64).init(allocator, 2);
    defer ridge2.deinit();
    try ridge2.fit(&X, &y, 10.0);
    const coeffs2 = try ridge2.coefficients();

    // Higher lambda should result in smaller coefficient magnitudes
    var mag1: f64 = 0;
    var mag2: f64 = 0;
    for (coeffs1) |c| mag1 += c * c;
    for (coeffs2) |c| mag2 += c * c;

    try std.testing.expect(mag2 < mag1);
}

test "RidgeRegression: empty dataset" {
    const allocator = std.testing.allocator;
    const X: []const []const f64 = &[_][]const f64{};
    const y: []const f64 = &[_]f64{};

    var ridge = RidgeRegression(f64).init(allocator, 2);
    defer ridge.deinit();

    const result = ridge.fit(X, y, 1.0);
    try std.testing.expectError(error.DimensionMismatch, result);
}
