/// Lasso Regression (Least Absolute Shrinkage and Selection Operator)
///
/// Linear regression with L1 regularization for feature selection and sparsity.
/// Uses coordinate descent optimization for efficient computation.
///
/// ## Mathematical Model
///
/// Minimize: ||y - Xw||² + α||w||₁
///
/// Where:
/// - y: target values (n×1)
/// - X: feature matrix (n×m)
/// - w: coefficients (m×1)
/// - α: regularization strength (λ in some literature)
/// - ||w||₁ = Σ|wⱼ| (L1 penalty)
///
/// ## Key Properties
///
/// - **Sparsity**: L1 penalty drives some coefficients exactly to zero
/// - **Feature selection**: Automatically selects relevant features
/// - **Handles multicollinearity**: Picks one feature from correlated groups
/// - **Non-differentiable**: |w| not differentiable at zero
///
/// ## Algorithm: Coordinate Descent
///
/// For each coordinate j:
/// 1. Compute partial residual: r⁽ʲ⁾ = y - Σₖ≠ⱼ xₖwₖ
/// 2. Compute correlation: zⱼ = xⱼᵀr⁽ʲ⁾
/// 3. Apply soft-thresholding (proximal operator):
///    - wⱼ = S(zⱼ, α) / ||xⱼ||²
///    - S(z, α) = sign(z) × max(|z| - α, 0)
///
/// Time: O(n_iter × nm) where n_iter ≈ 100-1000
/// Space: O(n + m) for residuals and coefficients
///
/// ## Comparison with Ridge Regression
///
/// | Property | Lasso (L1) | Ridge (L2) |
/// |----------|------------|------------|
/// | Penalty | α||w||₁ | α||w||² |
/// | Sparsity | Yes (some w=0) | No (all w≠0) |
/// | Feature selection | Yes | No |
/// | Multicollinearity | Picks one | Shrinks all |
/// | Solution | Non-unique | Unique |
/// | Solver | Coordinate descent | Closed-form |
///
/// ## Use Cases
///
/// - **High-dimensional data**: Gene expression, text classification
/// - **Feature selection**: Identifying relevant predictors
/// - **Interpretability**: Sparse models are easier to understand
/// - **Compressive sensing**: Signal reconstruction
/// - **Model compression**: Reduce model size for deployment
///
/// ## References
///
/// - Tibshirani, R. (1996). "Regression shrinkage and selection via the lasso"
/// - Friedman, J. et al. (2010). "Regularization paths for GLMs via coordinate descent"
const std = @import("std");
const Allocator = std.mem.Allocator;

/// Lasso Regression model
///
/// Type Parameters:
/// - T: Numeric type (f32 or f64)
pub fn LassoRegression(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Model coefficients (weights)
        coefficients: []T,
        /// Intercept term
        intercept: T,
        /// Regularization strength
        alpha: T,
        /// Memory allocator
        allocator: Allocator,

        /// Initialize Lasso Regression model
        ///
        /// Time: O(1)
        /// Space: O(1)
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - alpha: Regularization strength (>0, typically 0.01-1.0)
        pub fn init(allocator: Allocator, alpha: T) !Self {
            if (alpha <= 0) return error.InvalidAlpha;
            return Self{
                .coefficients = &[_]T{},
                .intercept = 0,
                .alpha = alpha,
                .allocator = allocator,
            };
        }

        /// Free model resources
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.coefficients.len > 0) {
                self.allocator.free(self.coefficients);
            }
        }

        /// Train model using coordinate descent
        ///
        /// Time: O(n_iter × nm) where n_iter typically 100-1000
        /// Space: O(n + m) for residuals and coefficients
        ///
        /// Parameters:
        /// - X: Feature matrix (n samples × m features)
        /// - y: Target values (length n)
        /// - max_iter: Maximum iterations (default 1000)
        /// - tol: Convergence tolerance (default 1e-4)
        /// - fit_intercept: Whether to fit intercept term (default true)
        pub fn fit(self: *Self, X: []const []const T, y: []const T, max_iter: usize, tol: T, fit_intercept: bool) !void {
            const n = y.len;
            if (n == 0 or X.len == 0) return error.EmptyData;
            const m = X[0].len;

            if (m == 0) return error.EmptyData;
            if (X.len != n) return error.ShapeMismatch;

            // Free old coefficients if any
            if (self.coefficients.len > 0) {
                self.allocator.free(self.coefficients);
            }

            // Allocate and initialize coefficients to zero
            self.coefficients = try self.allocator.alloc(T, m);
            @memset(self.coefficients, 0);

            // Center data if fitting intercept
            var X_centered: [][]T = undefined;
            var y_centered: []T = undefined;
            var y_mean: T = 0;
            var X_means: []T = undefined;
            var should_free_centered = false;

            if (fit_intercept) {
                // Compute means
                X_means = try self.allocator.alloc(T, m);
                defer self.allocator.free(X_means);
                @memset(X_means, 0);

                for (X) |row| {
                    for (row, 0..) |val, j| {
                        X_means[j] += val;
                    }
                }
                for (X_means) |*mean| {
                    mean.* /= @as(T, @floatFromInt(n));
                }

                for (y) |val| {
                    y_mean += val;
                }
                y_mean /= @as(T, @floatFromInt(n));

                // Center data
                X_centered = try self.allocator.alloc([]T, n);
                should_free_centered = true;

                for (X_centered, 0..) |*row, i| {
                    row.* = try self.allocator.alloc(T, m);
                    for (row.*, 0..) |*val, j| {
                        val.* = X[i][j] - X_means[j];
                    }
                }

                y_centered = try self.allocator.alloc(T, n);
                for (y_centered, 0..) |*val, i| {
                    val.* = y[i] - y_mean;
                }
            } else {
                // Copy data for non-intercept case
                X_centered = try self.allocator.alloc([]T, n);
                should_free_centered = true;

                for (X_centered, 0..) |*row, i| {
                    row.* = try self.allocator.alloc(T, m);
                    for (row.*, 0..) |*val, j| {
                        val.* = X[i][j];
                    }
                }

                y_centered = try self.allocator.alloc(T, n);
                for (y_centered, 0..) |*val, i| {
                    val.* = y[i];
                }
            }
            defer {
                if (should_free_centered) {
                    for (X_centered) |row| self.allocator.free(row);
                    self.allocator.free(X_centered);
                    self.allocator.free(y_centered);
                }
            }

            // Compute column norms squared (add small epsilon to avoid division by zero)
            const col_norms_sq = try self.allocator.alloc(T, m);
            defer self.allocator.free(col_norms_sq);

            for (0..m) |j| {
                var norm_sq: T = 0;
                for (X_centered) |row| {
                    norm_sq += row[j] * row[j];
                }
                // Add small epsilon to avoid division by zero
                col_norms_sq[j] = if (norm_sq > 0) norm_sq else 1e-10;
            }

            // Initialize residuals: r = y - Xw (w starts at zero, so r = y)
            const residuals = try self.allocator.alloc(T, n);
            defer self.allocator.free(residuals);
            @memcpy(residuals, y_centered);

            // Coordinate descent iterations
            var iter: usize = 0;
            while (iter < max_iter) : (iter += 1) {
                var max_change: T = 0;

                for (0..m) |j| {
                    const old_coef = self.coefficients[j];

                    // Add back contribution of current feature to residuals
                    if (old_coef != 0) {
                        for (residuals, 0..) |*r, i| {
                            r.* += X_centered[i][j] * old_coef;
                        }
                    }

                    // Compute correlation with residual
                    var rho: T = 0;
                    for (X_centered, 0..) |row, i| {
                        rho += row[j] * residuals[i];
                    }

                    // Soft-thresholding operator
                    const threshold = self.alpha * @as(T, @floatFromInt(n));
                    const new_coef = softThreshold(T)(rho, threshold) / col_norms_sq[j];
                    self.coefficients[j] = new_coef;

                    // Update residuals with new coefficient
                    if (new_coef != 0) {
                        for (residuals, 0..) |*r, i| {
                            r.* -= X_centered[i][j] * new_coef;
                        }
                    }

                    // Track convergence
                    const change = @abs(new_coef - old_coef);
                    if (change > max_change) {
                        max_change = change;
                    }
                }

                // Check convergence
                if (max_change < tol) {
                    break;
                }
            }

            // Compute intercept if needed
            if (fit_intercept) {
                self.intercept = y_mean;
                for (0..m) |j| {
                    self.intercept -= self.coefficients[j] * X_means[j];
                }
            } else {
                self.intercept = 0;
            }
        }

        /// Predict single sample
        ///
        /// Time: O(m) where m = number of features
        /// Space: O(1)
        ///
        /// Parameters:
        /// - x: Feature vector (length m)
        pub fn predict(self: Self, x: []const T) T {
            var result = self.intercept;
            for (x, 0..) |val, j| {
                result += self.coefficients[j] * val;
            }
            return result;
        }

        /// Predict multiple samples
        ///
        /// Time: O(nm) where n = samples, m = features
        /// Space: O(n) for predictions array
        ///
        /// Parameters:
        /// - X: Feature matrix (n samples × m features)
        pub fn predictBatch(self: Self, X: []const []const T) ![]T {
            const predictions = try self.allocator.alloc(T, X.len);
            for (X, 0..) |row, i| {
                predictions[i] = self.predict(row);
            }
            return predictions;
        }

        /// Compute R² score (coefficient of determination)
        ///
        /// Time: O(nm)
        /// Space: O(n)
        ///
        /// Parameters:
        /// - X: Feature matrix
        /// - y: True target values
        pub fn score(self: Self, X: []const []const T, y: []const T) !T {
            const predictions = try self.predictBatch(X);
            defer self.allocator.free(predictions);

            var ss_res: T = 0; // Residual sum of squares
            var ss_tot: T = 0; // Total sum of squares
            var y_mean: T = 0;

            for (y) |val| {
                y_mean += val;
            }
            y_mean /= @as(T, @floatFromInt(y.len));

            for (y, 0..) |true_val, i| {
                const pred_val = predictions[i];
                ss_res += (true_val - pred_val) * (true_val - pred_val);
                ss_tot += (true_val - y_mean) * (true_val - y_mean);
            }

            if (ss_tot == 0) return 1.0; // Perfect prediction
            return 1.0 - (ss_res / ss_tot);
        }

        /// Count number of non-zero coefficients (sparsity metric)
        ///
        /// Time: O(m)
        /// Space: O(1)
        pub fn countNonZero(self: Self) usize {
            var count: usize = 0;
            for (self.coefficients) |coef| {
                if (coef != 0) {
                    count += 1;
                }
            }
            return count;
        }

        /// Get L1 norm of coefficients
        ///
        /// Time: O(m)
        /// Space: O(1)
        pub fn l1Norm(self: Self) T {
            var norm: T = 0;
            for (self.coefficients) |coef| {
                norm += @abs(coef);
            }
            return norm;
        }
    };
}

/// Soft-thresholding operator (proximal operator for L1 norm)
///
/// S(z, λ) = sign(z) × max(|z| - λ, 0)
///
/// Time: O(1)
/// Space: O(1)
fn softThreshold(comptime T: type) fn (z: T, lambda: T) T {
    return struct {
        fn apply(z: T, lambda: T) T {
            if (z > lambda) {
                return z - lambda;
            } else if (z < -lambda) {
                return z + lambda;
            } else {
                return 0;
            }
        }
    }.apply;
}

// Tests
const testing = std.testing;

test "Lasso: initialization" {
    var model = try LassoRegression(f64).init(testing.allocator, 0.1);
    defer model.deinit();

    try testing.expectEqual(@as(f64, 0.1), model.alpha);
    try testing.expectEqual(@as(usize, 0), model.coefficients.len);
}

test "Lasso: invalid alpha" {
    const result = LassoRegression(f64).init(testing.allocator, -0.1);
    try testing.expectError(error.InvalidAlpha, result);
}

test "Lasso: simple linear relationship" {
    var model = try LassoRegression(f64).init(testing.allocator, 0.0001);
    defer model.deinit();

    // y = 2x + 1
    const X = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
        &[_]f64{4.0},
        &[_]f64{5.0},
    };
    const y = [_]f64{ 3.0, 5.0, 7.0, 9.0, 11.0 };

    try model.fit(&X, &y, 5000, 1e-6, true);

    // Should recover slope ≈ 2.0, intercept ≈ 1.0
    try testing.expect(@abs(model.coefficients[0] - 2.0) < 0.2);
    try testing.expect(@abs(model.intercept - 1.0) < 0.2);

    // R² should be reasonable (Lasso is biased estimator)
    const r2 = try model.score(&X, &y);
    try testing.expect(r2 > 0.85);
}

test "Lasso: multiple features" {
    var model = try LassoRegression(f64).init(testing.allocator, 0.001);
    defer model.deinit();

    // y = 3x₁ + 2x₂ + 1
    const X = [_][]const f64{
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 2.0, 1.0 },
        &[_]f64{ 3.0, 2.0 },
        &[_]f64{ 4.0, 2.0 },
        &[_]f64{ 5.0, 3.0 },
    };
    const y = [_]f64{ 6.0, 9.0, 14.0, 17.0, 23.0 };

    try model.fit(&X, &y, 5000, 1e-6, true);

    try testing.expectEqual(@as(usize, 2), model.coefficients.len);
    // Both features should be non-zero
    try testing.expect(@abs(model.coefficients[0]) > 0.1);
    try testing.expect(@abs(model.coefficients[1]) > 0.1);

    const r2 = try model.score(&X, &y);
    try testing.expect(r2 > 0.80);
}

test "Lasso: feature selection (irrelevant feature)" {
    var model = try LassoRegression(f64).init(testing.allocator, 0.1);
    defer model.deinit();

    // y = 3x₁ + 0x₂ (second feature is irrelevant)
    const X = [_][]const f64{
        &[_]f64{ 1.0, 0.5 },
        &[_]f64{ 2.0, 1.5 },
        &[_]f64{ 3.0, 2.5 },
        &[_]f64{ 4.0, 3.5 },
        &[_]f64{ 5.0, 4.5 },
    };
    const y = [_]f64{ 3.0, 6.0, 9.0, 12.0, 15.0 };

    try model.fit(&X, &y, 1000, 1e-4, true);

    // First feature should be non-zero
    try testing.expect(@abs(model.coefficients[0]) > 0.5);
    // Second feature should be near zero (sparse solution)
    try testing.expect(@abs(model.coefficients[1]) < 0.5);
}

test "Lasso: high alpha (strong regularization)" {
    var model = try LassoRegression(f64).init(testing.allocator, 10.0);
    defer model.deinit();

    const X = [_][]const f64{
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 2.0, 3.0 },
        &[_]f64{ 3.0, 4.0 },
    };
    const y = [_]f64{ 5.0, 8.0, 11.0 };

    try model.fit(&X, &y, 1000, 1e-4, true);

    // Very strong regularization should drive all coefficients near zero
    const l1 = model.l1Norm();
    try testing.expect(l1 < 1.0);
}

test "Lasso: no intercept" {
    var model = try LassoRegression(f64).init(testing.allocator, 0.01);
    defer model.deinit();

    const X = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
    };
    const y = [_]f64{ 2.0, 4.0, 6.0 }; // y = 2x (no intercept)

    try model.fit(&X, &y, 1000, 1e-4, false);

    try testing.expectEqual(@as(f64, 0), model.intercept);
    try testing.expect(@abs(model.coefficients[0] - 2.0) < 0.2);
}

test "Lasso: prediction" {
    var model = try LassoRegression(f64).init(testing.allocator, 0.0001);
    defer model.deinit();

    const X = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
    };
    const y = [_]f64{ 3.0, 5.0, 7.0 };

    try model.fit(&X, &y, 5000, 1e-6, true);

    const pred = model.predict(&[_]f64{4.0});
    try testing.expect(@abs(pred - 9.0) < 1.0);
}

test "Lasso: batch prediction" {
    var model = try LassoRegression(f64).init(testing.allocator, 0.0001);
    defer model.deinit();

    const X = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
    };
    const y = [_]f64{ 3.0, 5.0, 7.0 };

    try model.fit(&X, &y, 5000, 1e-6, true);

    const X_test = [_][]const f64{
        &[_]f64{4.0},
        &[_]f64{5.0},
    };

    const predictions = try model.predictBatch(&X_test);
    defer testing.allocator.free(predictions);

    try testing.expectEqual(@as(usize, 2), predictions.len);
    try testing.expect(@abs(predictions[0] - 9.0) < 1.0);
    try testing.expect(@abs(predictions[1] - 11.0) < 1.0);
}

test "Lasso: sparsity metric" {
    var model = try LassoRegression(f64).init(testing.allocator, 0.5);
    defer model.deinit();

    const X = [_][]const f64{
        &[_]f64{ 1.0, 0.1, 0.2 },
        &[_]f64{ 2.0, 0.2, 0.3 },
        &[_]f64{ 3.0, 0.3, 0.4 },
    };
    const y = [_]f64{ 3.0, 6.0, 9.0 }; // Only x₁ is relevant

    try model.fit(&X, &y, 1000, 1e-4, true);

    const non_zero = model.countNonZero();
    // Should have sparse solution (fewer than 3 non-zero coefficients)
    try testing.expect(non_zero < 3);
}

test "Lasso: L1 norm" {
    var model = try LassoRegression(f64).init(testing.allocator, 0.1);
    defer model.deinit();

    const X = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
    };
    const y = [_]f64{ 2.0, 4.0 };

    try model.fit(&X, &y, 1000, 1e-4, false);

    const norm = model.l1Norm();
    try testing.expect(norm > 0);
    try testing.expectApproxEqRel(@abs(model.coefficients[0]), norm, 1e-6);
}

test "Lasso: empty data" {
    var model = try LassoRegression(f64).init(testing.allocator, 0.1);
    defer model.deinit();

    const X: []const []const f64 = &[_][]const f64{};
    const y: []const f64 = &[_]f64{};

    const result = model.fit(X, y, 100, 1e-4, true);
    try testing.expectError(error.EmptyData, result);
}

test "Lasso: shape mismatch" {
    var model = try LassoRegression(f64).init(testing.allocator, 0.1);
    defer model.deinit();

    const X = [_][]const f64{
        &[_]f64{1.0},
        &[_]f64{2.0},
    };
    const y = [_]f64{ 1.0, 2.0, 3.0 }; // Wrong length

    const result = model.fit(&X, &y, 100, 1e-4, true);
    try testing.expectError(error.ShapeMismatch, result);
}

test "Lasso: f32 support" {
    var model = try LassoRegression(f32).init(testing.allocator, 0.0001);
    defer model.deinit();

    const X = [_][]const f32{
        &[_]f32{1.0},
        &[_]f32{2.0},
        &[_]f32{3.0},
    };
    const y = [_]f32{ 3.0, 5.0, 7.0 };

    try model.fit(&X, &y, 5000, 1e-5, true);

    const r2 = try model.score(&X, &y);
    try testing.expect(r2 > 0.80);
}
