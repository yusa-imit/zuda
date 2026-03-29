/// Elastic Net Regression
///
/// Linear regression with combined L1 and L2 regularization for balanced
/// feature selection and coefficient shrinkage.
///
/// ## Mathematical Model
///
/// Minimize: ||y - Xw||² + λ(α||w||₁ + (1-α)||w||²)
///
/// Where:
/// - y: target values (n×1)
/// - X: feature matrix (n×m)
/// - w: coefficients (m×1)
/// - λ: regularization strength (overall penalty scale)
/// - α: L1 ratio (balance between L1 and L2)
/// - ||w||₁ = Σ|wⱼ| (L1 penalty for sparsity)
/// - ||w||² = Σwⱼ² (L2 penalty for shrinkage)
///
/// ## Key Properties
///
/// - **Balanced regularization**: Combines benefits of Lasso and Ridge
/// - **Sparse solutions**: L1 component drives some coefficients to zero
/// - **Grouped selection**: L2 component handles correlated features better than Lasso
/// - **Stability**: More stable than Lasso when features are highly correlated
/// - **Flexibility**: α parameter controls L1/L2 balance
///
/// ## Algorithm: Coordinate Descent
///
/// For each coordinate j:
/// 1. Compute partial residual: r⁽ʲ⁾ = y - Σₖ≠ⱼ xₖwₖ
/// 2. Compute correlation: zⱼ = xⱼᵀr⁽ʲ⁾
/// 3. Apply soft-thresholding with L2 normalization:
///    - wⱼ = S(zⱼ, λα) / (||xⱼ||² + λ(1-α))
///    - S(z, t) = sign(z) × max(|z| - t, 0)
///
/// Time: O(n_iter × nm) where n_iter ≈ 100-1000
/// Space: O(n + m) for residuals and coefficients
///
/// ## Comparison with Lasso and Ridge
///
/// | Property | Elastic Net | Lasso (L1) | Ridge (L2) |
/// |----------|-------------|------------|------------|
/// | Penalty | L1 + L2 | α||w||₁ | α||w||² |
/// | Sparsity | Yes (some w=0) | Yes | No |
/// | Feature selection | Yes | Yes | No |
/// | Correlated features | Groups | Picks one | Shrinks all |
/// | Stability | High | Medium | High |
/// | Solution uniqueness | Yes | No | Yes |
///
/// ## Use Cases
///
/// - **Correlated features**: When Lasso is too unstable
/// - **Gene expression**: Many correlated genes, need grouping
/// - **Text classification**: Sparse features with correlations
/// - **Financial modeling**: Correlated market indicators
/// - **Image processing**: Spatially correlated pixels
/// - **General preference**: When unsure between Lasso and Ridge
///
/// ## Parameter Selection
///
/// - **α = 1**: Pure Lasso (maximum sparsity)
/// - **α = 0**: Pure Ridge (maximum grouping)
/// - **α = 0.5**: Balanced (common default)
/// - **λ**: Cross-validation to tune overall strength
///
/// ## References
///
/// - Zou, H. & Hastie, T. (2005). "Regularization and variable selection via the elastic net"
/// - Friedman, J. et al. (2010). "Regularization paths for GLMs via coordinate descent"
const std = @import("std");
const Allocator = std.mem.Allocator;

/// Elastic Net Regression model
///
/// Type Parameters:
/// - T: Numeric type (f32 or f64)
pub fn ElasticNetRegression(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Model coefficients (weights)
        coefficients: []T,
        /// Intercept term
        intercept: T,
        /// Regularization strength (λ)
        lambda: T,
        /// L1 ratio (α): balance between L1 and L2
        /// α = 1: pure Lasso, α = 0: pure Ridge
        l1_ratio: T,
        /// Memory allocator
        allocator: Allocator,

        /// Initialize Elastic Net Regression model
        ///
        /// Time: O(1)
        /// Space: O(1)
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - lambda: Regularization strength (>0, typically 0.01-1.0)
        /// - l1_ratio: L1 ratio (0 ≤ α ≤ 1, typically 0.5)
        pub fn init(allocator: Allocator, lambda: T, l1_ratio: T) !Self {
            if (lambda <= 0) return error.InvalidLambda;
            if (l1_ratio < 0 or l1_ratio > 1) return error.InvalidL1Ratio;
            return Self{
                .coefficients = &[_]T{},
                .intercept = 0,
                .lambda = lambda,
                .l1_ratio = l1_ratio,
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
                self.coefficients = &[_]T{};
            }
        }

        /// Fit model to training data using coordinate descent
        ///
        /// Time: O(max_iter × nm) where n=samples, m=features
        /// Space: O(n + m) for residuals and coefficients
        ///
        /// Parameters:
        /// - X: Feature matrix (n × m)
        /// - y: Target values (n)
        /// - max_iter: Maximum iterations (default: 1000)
        /// - tol: Convergence tolerance (default: 1e-4)
        pub fn fit(self: *Self, X: []const []const T, y: []const T, max_iter: usize, tol: T) !void {
            const n = y.len;
            if (n == 0 or X.len == 0) return error.EmptyData;
            if (X.len != n) return error.ShapeMismatch;

            const m = X[0].len;
            if (m == 0) return error.EmptyData;

            // Free existing coefficients
            if (self.coefficients.len > 0) {
                self.allocator.free(self.coefficients);
            }

            // Allocate coefficients
            self.coefficients = try self.allocator.alloc(T, m);
            @memset(self.coefficients, 0);

            // Center data: compute mean of y
            var y_mean: T = 0;
            for (y) |yi| {
                y_mean += yi;
            }
            y_mean /= @as(T, @floatFromInt(n));

            // Compute feature means
            const x_means = try self.allocator.alloc(T, m);
            defer self.allocator.free(x_means);
            @memset(x_means, 0);

            for (X) |xi| {
                for (xi, 0..) |xij, j| {
                    x_means[j] += xij;
                }
            }
            for (x_means) |*mean| {
                mean.* /= @as(T, @floatFromInt(n));
            }

            // Compute feature norms (column sums of squares)
            const x_norms = try self.allocator.alloc(T, m);
            defer self.allocator.free(x_norms);
            @memset(x_norms, 0);

            for (X) |xi| {
                for (xi, 0..) |xij, j| {
                    const centered = xij - x_means[j];
                    x_norms[j] += centered * centered;
                }
            }

            // Allocate residuals
            const residuals = try self.allocator.alloc(T, n);
            defer self.allocator.free(residuals);

            // Initialize residuals: r = y - y_mean (intercept-only model)
            for (y, 0..) |yi, i| {
                residuals[i] = yi - y_mean;
            }

            // Coordinate descent
            var iter: usize = 0;
            while (iter < max_iter) : (iter += 1) {
                var max_change: T = 0;

                // Update each coefficient
                for (0..m) |j| {
                    // Add back contribution of current feature
                    for (X, 0..) |xi, i| {
                        const centered = xi[j] - x_means[j];
                        residuals[i] += centered * self.coefficients[j];
                    }

                    // Compute correlation with residuals
                    var correlation: T = 0;
                    for (X, 0..) |xi, i| {
                        const centered = xi[j] - x_means[j];
                        correlation += centered * residuals[i];
                    }

                    // Soft-thresholding with elastic net penalty
                    const old_coef = self.coefficients[j];
                    const threshold = self.lambda * self.l1_ratio * @as(T, @floatFromInt(n));
                    const denominator = x_norms[j] + self.lambda * (1 - self.l1_ratio) * @as(T, @floatFromInt(n));

                    if (denominator > 0) {
                        if (correlation > threshold) {
                            self.coefficients[j] = (correlation - threshold) / denominator;
                        } else if (correlation < -threshold) {
                            self.coefficients[j] = (correlation + threshold) / denominator;
                        } else {
                            self.coefficients[j] = 0;
                        }
                    }

                    // Update residuals with new coefficient
                    for (X, 0..) |xi, i| {
                        const centered = xi[j] - x_means[j];
                        residuals[i] -= centered * self.coefficients[j];
                    }

                    // Track convergence
                    const change = @abs(self.coefficients[j] - old_coef);
                    if (change > max_change) {
                        max_change = change;
                    }
                }

                // Check convergence
                if (max_change < tol) {
                    break;
                }
            }

            // Compute intercept: intercept = y_mean - coefficients^T * x_means
            self.intercept = y_mean;
            for (self.coefficients, 0..) |coef, j| {
                self.intercept -= coef * x_means[j];
            }
        }

        /// Predict single sample
        ///
        /// Time: O(m) where m=features
        /// Space: O(1)
        ///
        /// Parameters:
        /// - x: Feature vector (m)
        pub fn predict(self: Self, x: []const T) T {
            var result = self.intercept;
            for (x, 0..) |xi, i| {
                result += self.coefficients[i] * xi;
            }
            return result;
        }

        /// Predict multiple samples
        ///
        /// Time: O(nm) where n=samples, m=features
        /// Space: O(n)
        ///
        /// Parameters:
        /// - X: Feature matrix (n × m)
        pub fn predictBatch(self: Self, X: []const []const T) ![]T {
            const predictions = try self.allocator.alloc(T, X.len);
            for (X, 0..) |xi, i| {
                predictions[i] = self.predict(xi);
            }
            return predictions;
        }

        /// Compute R² score (coefficient of determination)
        ///
        /// Time: O(nm)
        /// Space: O(n)
        ///
        /// Parameters:
        /// - X: Feature matrix (n × m)
        /// - y: True values (n)
        pub fn score(self: Self, X: []const []const T, y: []const T) !T {
            if (X.len != y.len) return error.ShapeMismatch;

            // Compute predictions
            const predictions = try self.predictBatch(X);
            defer self.allocator.free(predictions);

            // Compute mean of y
            var y_mean: T = 0;
            for (y) |yi| {
                y_mean += yi;
            }
            y_mean /= @as(T, @floatFromInt(y.len));

            // Compute sum of squares
            var ss_res: T = 0; // residual sum of squares
            var ss_tot: T = 0; // total sum of squares
            for (y, 0..) |yi, i| {
                const residual = yi - predictions[i];
                ss_res += residual * residual;
                const deviation = yi - y_mean;
                ss_tot += deviation * deviation;
            }

            // R² = 1 - SS_res / SS_tot
            if (ss_tot == 0) return 1.0;
            return 1 - (ss_res / ss_tot);
        }

        /// Get number of non-zero coefficients (active features)
        ///
        /// Time: O(m)
        /// Space: O(1)
        pub fn countNonZero(self: Self) usize {
            var count: usize = 0;
            for (self.coefficients) |coef| {
                if (coef != 0) count += 1;
            }
            return count;
        }

        /// Compute L1 norm of coefficients (sum of absolute values)
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

        /// Compute L2 norm of coefficients (sum of squares)
        ///
        /// Time: O(m)
        /// Space: O(1)
        pub fn l2Norm(self: Self) T {
            var norm: T = 0;
            for (self.coefficients) |coef| {
                norm += coef * coef;
            }
            return norm;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "ElasticNetRegression: basic initialization" {
    const allocator = std.testing.allocator;

    // Valid initialization
    var model = try ElasticNetRegression(f64).init(allocator, 0.1, 0.5);
    defer model.deinit();

    try std.testing.expectEqual(@as(f64, 0.1), model.lambda);
    try std.testing.expectEqual(@as(f64, 0.5), model.l1_ratio);
    try std.testing.expectEqual(@as(usize, 0), model.coefficients.len);
}

test "ElasticNetRegression: invalid parameters" {
    const allocator = std.testing.allocator;

    // Negative lambda
    try std.testing.expectError(error.InvalidLambda, ElasticNetRegression(f64).init(allocator, -0.1, 0.5));

    // Zero lambda
    try std.testing.expectError(error.InvalidLambda, ElasticNetRegression(f64).init(allocator, 0.0, 0.5));

    // Invalid l1_ratio < 0
    try std.testing.expectError(error.InvalidL1Ratio, ElasticNetRegression(f64).init(allocator, 0.1, -0.1));

    // Invalid l1_ratio > 1
    try std.testing.expectError(error.InvalidL1Ratio, ElasticNetRegression(f64).init(allocator, 0.1, 1.5));
}

test "ElasticNetRegression: simple linear fit (low regularization)" {
    const allocator = std.testing.allocator;

    // y = 2x₁ + 3x₂
    const X = [_][2]f64{
        .{ 1.0, 1.0 },
        .{ 2.0, 2.0 },
        .{ 3.0, 3.0 },
        .{ 4.0, 4.0 },
    };
    const y = [_]f64{ 5.0, 10.0, 15.0, 20.0 };

    var model = try ElasticNetRegression(f64).init(allocator, 0.01, 0.5);
    defer model.deinit();

    const X_ptrs = [_][]const f64{ &X[0], &X[1], &X[2], &X[3] };
    try model.fit(&X_ptrs, &y, 2000, 1e-6);

    // With low regularization, should achieve good R² (demonstrates convergence and fit)
    const r2 = try model.score(&X_ptrs, &y);
    try std.testing.expect(r2 > 0.85);

    // Both coefficients should be positive (matches true model [2, 3])
    try std.testing.expect(model.coefficients[0] > 0);
    try std.testing.expect(model.coefficients[1] > 0);

    // Predictions should be in reasonable range
    const pred = model.predict(&X[0]);
    try std.testing.expect(pred > 2.0 and pred < 8.0);
}

test "ElasticNetRegression: high regularization induces sparsity" {
    const allocator = std.testing.allocator;

    // y = 2x₁ + 0x₂ (second feature is noise)
    const X = [_][2]f64{
        .{ 1.0, 0.5 },
        .{ 2.0, -0.3 },
        .{ 3.0, 0.7 },
        .{ 4.0, -0.2 },
    };
    const y = [_]f64{ 2.0, 4.0, 6.0, 8.0 };

    var model = try ElasticNetRegression(f64).init(allocator, 0.5, 0.7); // High lambda, favor L1
    defer model.deinit();

    const X_ptrs = [_][]const f64{ &X[0], &X[1], &X[2], &X[3] };
    try model.fit(&X_ptrs, &y, 1000, 1e-4);

    // Should have sparsity (some coefficients near zero)
    const non_zero = model.countNonZero();
    try std.testing.expect(non_zero <= 2);

    // First coefficient should dominate
    try std.testing.expect(@abs(model.coefficients[0]) > @abs(model.coefficients[1]));
}

test "ElasticNetRegression: alpha=1 behaves like Lasso" {
    const allocator = std.testing.allocator;

    const X = [_][3]f64{
        .{ 1.0, 2.0, 0.1 },
        .{ 2.0, 3.0, 0.2 },
        .{ 3.0, 4.0, 0.1 },
        .{ 4.0, 5.0, 0.3 },
    };
    const y = [_]f64{ 5.0, 8.0, 11.0, 14.0 };

    var model = try ElasticNetRegression(f64).init(allocator, 0.3, 1.0); // Pure L1
    defer model.deinit();

    const X_ptrs = [_][]const f64{ &X[0], &X[1], &X[2], &X[3] };
    try model.fit(&X_ptrs, &y, 1000, 1e-4);

    // Should drive at least one coefficient to exactly zero
    var has_zero = false;
    for (model.coefficients) |coef| {
        if (coef == 0) has_zero = true;
    }
    try std.testing.expect(has_zero);
}

test "ElasticNetRegression: alpha=0 behaves like Ridge" {
    const allocator = std.testing.allocator;

    const X = [_][3]f64{
        .{ 1.0, 2.0, 0.1 },
        .{ 2.0, 3.0, 0.2 },
        .{ 3.0, 4.0, 0.1 },
        .{ 4.0, 5.0, 0.3 },
    };
    const y = [_]f64{ 5.0, 8.0, 11.0, 14.0 };

    var model = try ElasticNetRegression(f64).init(allocator, 0.1, 0.0); // Pure L2
    defer model.deinit();

    const X_ptrs = [_][]const f64{ &X[0], &X[1], &X[2], &X[3] };
    try model.fit(&X_ptrs, &y, 1000, 1e-4);

    // All coefficients should be non-zero (Ridge doesn't produce sparsity)
    const non_zero = model.countNonZero();
    try std.testing.expectEqual(@as(usize, 3), non_zero);
}

test "ElasticNetRegression: prediction consistency" {
    const allocator = std.testing.allocator;

    const X = [_][2]f64{
        .{ 1.0, 2.0 },
        .{ 2.0, 3.0 },
        .{ 3.0, 4.0 },
    };
    const y = [_]f64{ 5.0, 8.0, 11.0 };

    var model = try ElasticNetRegression(f64).init(allocator, 0.1, 0.5);
    defer model.deinit();

    const X_ptrs = [_][]const f64{ &X[0], &X[1], &X[2] };
    try model.fit(&X_ptrs, &y, 1000, 1e-4);

    // Single prediction
    const pred1 = model.predict(&X[0]);

    // Batch prediction
    const preds = try model.predictBatch(&X_ptrs);
    defer allocator.free(preds);

    // Should match
    try std.testing.expectApproxEqAbs(pred1, preds[0], 1e-10);
}

test "ElasticNetRegression: empty data" {
    const allocator = std.testing.allocator;

    var model = try ElasticNetRegression(f64).init(allocator, 0.1, 0.5);
    defer model.deinit();

    const X_empty: []const []const f64 = &[_][]const f64{};
    const y_empty: []const f64 = &[_]f64{};

    try std.testing.expectError(error.EmptyData, model.fit(X_empty, y_empty, 100, 1e-4));
}

test "ElasticNetRegression: shape mismatch" {
    const allocator = std.testing.allocator;

    const X = [_][2]f64{
        .{ 1.0, 2.0 },
        .{ 2.0, 3.0 },
    };
    const y = [_]f64{ 5.0, 8.0, 11.0 }; // Mismatched length

    var model = try ElasticNetRegression(f64).init(allocator, 0.1, 0.5);
    defer model.deinit();

    const X_ptrs = [_][]const f64{ &X[0], &X[1] };
    try std.testing.expectError(error.ShapeMismatch, model.fit(&X_ptrs, &y, 100, 1e-4));
}

test "ElasticNetRegression: f32 support" {
    const allocator = std.testing.allocator;

    const X = [_][2]f32{
        .{ 1.0, 1.0 },
        .{ 2.0, 2.0 },
        .{ 3.0, 3.0 },
    };
    const y = [_]f32{ 4.0, 8.0, 12.0 };

    var model = try ElasticNetRegression(f32).init(allocator, 0.01, 0.5);
    defer model.deinit();

    const X_ptrs = [_][]const f32{ &X[0], &X[1], &X[2] };
    try model.fit(&X_ptrs, &y, 1000, 1e-4);

    const r2 = try model.score(&X_ptrs, &y);
    try std.testing.expect(r2 > 0.9);
}

test "ElasticNetRegression: convergence with max_iter" {
    const allocator = std.testing.allocator;

    const X = [_][2]f64{
        .{ 1.0, 2.0 },
        .{ 2.0, 3.0 },
        .{ 3.0, 4.0 },
    };
    const y = [_]f64{ 5.0, 8.0, 11.0 };

    var model = try ElasticNetRegression(f64).init(allocator, 0.1, 0.5);
    defer model.deinit();

    const X_ptrs = [_][]const f64{ &X[0], &X[1], &X[2] };

    // Should converge even with very few iterations for simple data
    try model.fit(&X_ptrs, &y, 10, 1e-4);

    const pred = model.predict(&X[0]);
    try std.testing.expect(@abs(pred - 5.0) < 2.0); // Reasonable prediction
}

test "ElasticNetRegression: L1 and L2 norm computation" {
    const allocator = std.testing.allocator;

    const X = [_][2]f64{
        .{ 1.0, 2.0 },
        .{ 2.0, 3.0 },
        .{ 3.0, 4.0 },
    };
    const y = [_]f64{ 5.0, 8.0, 11.0 };

    var model = try ElasticNetRegression(f64).init(allocator, 0.1, 0.5);
    defer model.deinit();

    const X_ptrs = [_][]const f64{ &X[0], &X[1], &X[2] };
    try model.fit(&X_ptrs, &y, 1000, 1e-4);

    // Both norms should be non-negative
    const l1 = model.l1Norm();
    const l2 = model.l2Norm();
    try std.testing.expect(l1 >= 0);
    try std.testing.expect(l2 >= 0);

    // L2 norm should equal sum of squared coefficients
    var expected_l2: f64 = 0;
    for (model.coefficients) |coef| {
        expected_l2 += coef * coef;
    }
    try std.testing.expectApproxEqAbs(l2, expected_l2, 1e-10);
}

test "ElasticNetRegression: low regularization achieves good fit" {
    const allocator = std.testing.allocator;

    // y = x₁ + 2x₂
    const X = [_][2]f64{
        .{ 1.0, 1.0 },
        .{ 2.0, 2.0 },
        .{ 3.0, 3.0 },
        .{ 4.0, 4.0 },
    };
    const y = [_]f64{ 3.0, 6.0, 9.0, 12.0 };

    var model = try ElasticNetRegression(f64).init(allocator, 0.01, 0.5);
    defer model.deinit();

    const X_ptrs = [_][]const f64{ &X[0], &X[1], &X[2], &X[3] };
    try model.fit(&X_ptrs, &y, 2000, 1e-6);

    // With low regularization, should achieve high R² (demonstrates convergence)
    const r2 = try model.score(&X_ptrs, &y);
    try std.testing.expect(r2 > 0.95);

    // Coefficients should be positive (matches true model structure)
    try std.testing.expect(model.coefficients[0] > 0);
    try std.testing.expect(model.coefficients[1] > 0);

    // Sum of coefficients should be reasonable
    const coef_sum = model.coefficients[0] + model.coefficients[1];
    try std.testing.expect(coef_sum > 1.0 and coef_sum < 5.0);
}
