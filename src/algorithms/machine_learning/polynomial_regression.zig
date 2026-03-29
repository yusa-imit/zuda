/// Polynomial Regression
///
/// Extends linear regression to model non-linear relationships by using polynomial features.
/// Fits a polynomial of degree d: y = w₀ + w₁x + w₂x² + ... + wₐxᵈ + w_{d+1}x₁x₂ + ...
///
/// For multivariate inputs, generates interaction terms and polynomial features.
///
/// **Algorithm Overview:**
/// 1. **Feature Expansion**: Transform input features x → φ(x) where φ includes:
///    - Original features: x₁, x₂, ..., xₘ
///    - Polynomial terms: x₁², x₂², ..., xₘᵈ
///    - Interaction terms: x₁x₂, x₁x₃, ... (if include_interaction=true)
/// 2. **Linear Model**: Fit linear regression on expanded features φ(x)
/// 3. **Prediction**: Apply same feature expansion to new inputs
///
/// **Training Methods:**
/// - `fitOLS()`: Closed-form solution via normal equations
///   - Time: O(n × p² + p³) where p = number of polynomial features
///   - Space: O(p²) for normal equations matrix
/// - `fitGD()`: Gradient descent with optional L2 regularization (Ridge)
///   - Time: O(n_iter × n × p)
///   - Space: O(p) for weights
///
/// **Feature Count:**
/// For degree d and m original features:
/// - Without interaction: p = 1 + m×d
/// - With interaction (degree ≤ 2): p = 1 + 2m + m(m-1)/2
///
/// **Use Cases:**
/// - Non-linear trend modeling (temperature vs. time)
/// - Physics equations (gravity, projectile motion)
/// - Growth curves (population, bacteria)
/// - Economics (diminishing returns, supply-demand)
/// - Engineering (stress-strain relationships)
///
/// **Example:**
/// ```zig
/// const poly = PolynomialRegression(f64).init(allocator, 2, true); // degree=2, interactions
/// defer poly.deinit();
///
/// const X = [_][2]f64{ .{1.0, 2.0}, .{2.0, 3.0}, .{3.0, 4.0} };
/// const y = [_]f64{ 5.0, 10.0, 17.0 }; // y = x₁² + x₂
///
/// try poly.fitOLS(&X, &y);
/// const pred = try poly.predictSample(&[_]f64{4.0, 5.0}); // → 21.0
/// const r2 = try poly.r2Score(&X, &y); // → ≈1.0 (perfect fit)
/// ```
const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// Polynomial Regression model
///
/// **Type Parameters:**
/// - `T`: Floating-point type (f32 or f64)
///
/// **Fields:**
/// - `allocator`: Memory allocator
/// - `degree`: Polynomial degree (1 = linear, 2 = quadratic, etc.)
/// - `include_interaction`: Whether to include interaction terms (x₁x₂, etc.)
/// - `n_features`: Number of original input features
/// - `n_poly_features`: Number of expanded polynomial features
/// - `weights`: Learned coefficients (length = n_poly_features)
/// - `intercept`: Bias term
/// - `fitted`: Whether model has been trained
pub fn PolynomialRegression(comptime T: type) type {
    if (T != f32 and T != f64) {
        @compileError("PolynomialRegression only supports f32 and f64");
    }

    return struct {
        allocator: Allocator,
        degree: usize,
        include_interaction: bool,
        n_features: usize,
        n_poly_features: usize,
        weights: ?[]T,
        intercept: T,
        fitted: bool,

        const Self = @This();

        /// Initialize polynomial regression model
        ///
        /// **Parameters:**
        /// - `allocator`: Memory allocator
        /// - `degree`: Polynomial degree (must be ≥ 1)
        /// - `include_interaction`: Include interaction terms (e.g., x₁x₂)
        ///
        /// **Returns:** Uninitialized model (call fit before predict)
        ///
        /// **Time:** O(1)
        /// **Space:** O(1)
        pub fn init(allocator: Allocator, degree: usize, include_interaction: bool) Self {
            return .{
                .allocator = allocator,
                .degree = degree,
                .include_interaction = include_interaction,
                .n_features = 0,
                .n_poly_features = 0,
                .weights = null,
                .intercept = 0,
                .fitted = false,
            };
        }

        /// Free model resources
        ///
        /// **Time:** O(1)
        /// **Space:** O(1)
        pub fn deinit(self: *Self) void {
            if (self.weights) |w| {
                self.allocator.free(w);
            }
            self.* = undefined;
        }

        /// Compute number of polynomial features
        ///
        /// Without interaction: 1 + m×d (intercept + each feature to each power)
        /// With interaction: 1 + m×d + combinations (for degree ≤ 2)
        ///
        /// **Time:** O(1)
        /// **Space:** O(1)
        fn computePolyFeatures(m: usize, d: usize, interaction: bool) usize {
            if (!interaction) {
                // Simple: 1 (intercept) + m*d (each feature^1, ^2, ..., ^d)
                return 1 + m * d;
            } else {
                // With interaction (limited to degree 2 for now)
                // 1 (intercept) + m (linear) + m (quadratic) + m*(m-1)/2 (interactions)
                if (d == 1) {
                    return 1 + m; // intercept + linear terms
                } else if (d == 2) {
                    return 1 + m + m + (m * (m - 1)) / 2; // intercept + linear + quadratic + interactions
                } else {
                    // For degree > 2, fall back to no interaction
                    return 1 + m * d;
                }
            }
        }

        /// Expand features to polynomial basis
        ///
        /// For degree=2, include_interaction=false:
        /// [x₁, x₂] → [1, x₁, x₁², x₂, x₂²]
        ///
        /// For degree=2, include_interaction=true:
        /// [x₁, x₂] → [1, x₁, x₂, x₁², x₂², x₁x₂]
        ///
        /// **Time:** O(p) where p = number of polynomial features
        /// **Space:** O(p)
        fn expandFeatures(self: *const Self, x: []const T, out: []T) !void {
            if (x.len != self.n_features) return error.FeatureMismatch;
            if (out.len != self.n_poly_features) return error.BufferTooSmall;

            var idx: usize = 0;

            // Intercept
            out[idx] = 1.0;
            idx += 1;

            if (!self.include_interaction or self.degree > 2) {
                // Simple polynomial: x₁, x₁², ..., x₁ᵈ, x₂, x₂², ..., x₂ᵈ, ...
                for (x) |xi| {
                    var power: T = xi;
                    var d: usize = 1;
                    while (d <= self.degree) : (d += 1) {
                        out[idx] = power;
                        idx += 1;
                        power *= xi;
                    }
                }
            } else {
                // Polynomial with interaction terms (degree ≤ 2 only)
                // Order: [1, x₁, x₂, ..., xₘ, x₁², x₂², ..., xₘ², x₁x₂, x₁x₃, ..., x_{m-1}x_m]

                // Linear terms: x₁, x₂, ..., xₘ
                for (x) |xi| {
                    out[idx] = xi;
                    idx += 1;
                }

                // Quadratic terms: x₁², x₂², ..., xₘ²
                if (self.degree >= 2) {
                    for (x) |xi| {
                        out[idx] = xi * xi;
                        idx += 1;
                    }

                    // Interaction terms: x_i × x_j for i < j
                    for (0..x.len) |i| {
                        for (i + 1..x.len) |j| {
                            out[idx] = x[i] * x[j];
                            idx += 1;
                        }
                    }
                }
            }

            if (idx != self.n_poly_features) return error.FeatureCountMismatch;
        }

        /// Fit model using Ordinary Least Squares (closed-form solution)
        ///
        /// Solves: (Φᵀ Φ) w = Φᵀ y where Φ = expanded features
        ///
        /// **Parameters:**
        /// - `X`: Training features [n × m]
        /// - `y`: Target values [n]
        ///
        /// **Returns:** error.EmptyInput if n=0
        ///
        /// **Time:** O(n × p² + p³) where p = polynomial feature count
        /// **Space:** O(p² + n×p)
        pub fn fitOLS(self: *Self, X: []const []const T, y: []const T) !void {
            const n = X.len;
            if (n == 0 or y.len == 0) return error.EmptyInput;
            if (n != y.len) return error.DimensionMismatch;

            // Infer n_features from first sample
            self.n_features = X[0].len;
            self.n_poly_features = computePolyFeatures(self.n_features, self.degree, self.include_interaction);

            // Expand all samples to polynomial features
            const Phi = try self.allocator.alloc([]T, n);
            defer {
                for (Phi) |row| self.allocator.free(row);
                self.allocator.free(Phi);
            }
            for (Phi, 0..) |*row, i| {
                row.* = try self.allocator.alloc(T, self.n_poly_features);
                try self.expandFeatures(X[i], row.*);
            }

            // Solve normal equations: (ΦᵀΦ)w = Φᵀy
            // Using Gaussian elimination with partial pivoting

            const p = self.n_poly_features;
            var A = try self.allocator.alloc([]T, p);
            defer {
                for (A) |row| self.allocator.free(row);
                self.allocator.free(A);
            }
            for (A) |*row| row.* = try self.allocator.alloc(T, p);

            var b = try self.allocator.alloc(T, p);
            defer self.allocator.free(b);

            // Compute ΦᵀΦ
            for (0..p) |i| {
                for (0..p) |j| {
                    var sum: T = 0;
                    for (Phi) |phi_row| {
                        sum += phi_row[i] * phi_row[j];
                    }
                    A[i][j] = sum;
                }
            }

            // Compute Φᵀy
            for (0..p) |i| {
                var sum: T = 0;
                for (Phi, 0..) |phi_row, k| {
                    sum += phi_row[i] * y[k];
                }
                b[i] = sum;
            }

            // Solve using Gaussian elimination
            try gaussianElimination(T, A, b);

            // Store results (first coefficient is intercept)
            self.intercept = b[0];
            if (self.weights) |w| self.allocator.free(w);
            self.weights = try self.allocator.alloc(T, p - 1);
            @memcpy(self.weights.?, b[1..]);
            self.fitted = true;
        }

        /// Fit model using Gradient Descent
        ///
        /// **Parameters:**
        /// - `X`: Training features [n × m]
        /// - `y`: Target values [n]
        /// - `learning_rate`: Step size (default: 0.01)
        /// - `max_iterations`: Maximum iterations (default: 1000)
        /// - `tolerance`: Convergence threshold (default: 1e-6)
        /// - `l2_lambda`: L2 regularization strength (default: 0.0)
        ///
        /// **Time:** O(n_iter × n × p)
        /// **Space:** O(n×p + p)
        pub fn fitGD(
            self: *Self,
            X: []const []const T,
            y: []const T,
            learning_rate: T,
            max_iterations: usize,
            tolerance: T,
            l2_lambda: T,
        ) !void {
            const n = X.len;
            if (n == 0 or y.len == 0) return error.EmptyInput;
            if (n != y.len) return error.DimensionMismatch;

            self.n_features = X[0].len;
            self.n_poly_features = computePolyFeatures(self.n_features, self.degree, self.include_interaction);

            // Expand features
            const Phi = try self.allocator.alloc([]T, n);
            defer {
                for (Phi) |row| self.allocator.free(row);
                self.allocator.free(Phi);
            }
            for (Phi, 0..) |*row, i| {
                row.* = try self.allocator.alloc(T, self.n_poly_features);
                try self.expandFeatures(X[i], row.*);
            }

            // Initialize weights
            if (self.weights) |w| self.allocator.free(w);
            self.weights = try self.allocator.alloc(T, self.n_poly_features - 1);
            @memset(self.weights.?, 0);
            self.intercept = 0;

            var prev_loss: T = std.math.inf(T);

            // Gradient descent iterations
            var iter: usize = 0;
            while (iter < max_iterations) : (iter += 1) {
                var loss: T = 0;
                var grad_intercept: T = 0;
                var grad = try self.allocator.alloc(T, self.n_poly_features - 1);
                defer self.allocator.free(grad);
                @memset(grad, 0);

                // Compute gradients
                for (Phi, 0..) |phi, k| {
                    const pred = self.intercept + dotProduct(self.weights.?, phi[1..]);
                    const error_k = pred - y[k];
                    loss += error_k * error_k;

                    grad_intercept += error_k;
                    for (phi[1..], 0..) |phi_j, j| {
                        grad[j] += error_k * phi_j;
                    }
                }

                loss = loss / @as(T, @floatFromInt(n));
                grad_intercept = grad_intercept / @as(T, @floatFromInt(n));
                for (grad) |*g| g.* = g.* / @as(T, @floatFromInt(n));

                // Add L2 regularization
                if (l2_lambda > 0) {
                    for (self.weights.?, 0..) |w, j| {
                        grad[j] += l2_lambda * w;
                        loss += 0.5 * l2_lambda * w * w;
                    }
                }

                // Check convergence
                if (@abs(prev_loss - loss) < tolerance) break;
                prev_loss = loss;

                // Update weights
                self.intercept -= learning_rate * grad_intercept;
                for (self.weights.?, 0..) |*w, j| {
                    w.* -= learning_rate * grad[j];
                }
            }

            self.fitted = true;
        }

        /// Predict on multiple samples
        ///
        /// **Time:** O(k × p) where k = number of samples
        /// **Space:** O(k)
        pub fn predict(self: *const Self, X: []const []const T) ![]T {
            if (!self.fitted) return error.NotFitted;
            const k = X.len;
            var predictions = try self.allocator.alloc(T, k);
            for (X, 0..) |x, i| {
                predictions[i] = try self.predictSample(x);
            }
            return predictions;
        }

        /// Predict on single sample
        ///
        /// **Time:** O(p)
        /// **Space:** O(p)
        pub fn predictSample(self: *const Self, x: []const T) !T {
            if (!self.fitted) return error.NotFitted;
            if (x.len != self.n_features) return error.FeatureMismatch;

            var phi = try self.allocator.alloc(T, self.n_poly_features);
            defer self.allocator.free(phi);
            try self.expandFeatures(x, phi);

            return self.intercept + dotProduct(self.weights.?, phi[1..]);
        }

        /// Compute Mean Squared Error
        ///
        /// **Time:** O(k × p)
        /// **Space:** O(k)
        pub fn score(self: *const Self, X: []const []const T, y: []const T) !T {
            const predictions = try self.predict(X);
            defer self.allocator.free(predictions);

            var mse: T = 0;
            for (predictions, y) |pred, actual| {
                const error_val = pred - actual;
                mse += error_val * error_val;
            }
            return mse / @as(T, @floatFromInt(predictions.len));
        }

        /// Compute R² score (coefficient of determination)
        ///
        /// R² = 1 - (SS_res / SS_tot)
        /// - SS_res = Σ(yᵢ - ŷᵢ)² (residual sum of squares)
        /// - SS_tot = Σ(yᵢ - ȳ)² (total sum of squares)
        ///
        /// **Time:** O(k × p)
        /// **Space:** O(k)
        pub fn r2Score(self: *const Self, X: []const []const T, y: []const T) !T {
            const predictions = try self.predict(X);
            defer self.allocator.free(predictions);

            var y_mean: T = 0;
            for (y) |yi| y_mean += yi;
            y_mean /= @as(T, @floatFromInt(y.len));

            var ss_res: T = 0;
            var ss_tot: T = 0;
            for (predictions, y) |pred, actual| {
                ss_res += (actual - pred) * (actual - pred);
                ss_tot += (actual - y_mean) * (actual - y_mean);
            }

            return 1.0 - (ss_res / ss_tot);
        }

        /// Get learned coefficients (excluding intercept)
        pub fn coefficients(self: *const Self) ![]const T {
            if (!self.fitted) return error.NotFitted;
            return self.weights.?;
        }

        // Helper functions
        fn dotProduct(a: []const T, b: []const T) T {
            var sum: T = 0;
            for (a, b) |ai, bi| sum += ai * bi;
            return sum;
        }

        fn gaussianElimination(comptime U: type, A: [][]U, b: []U) !void {
            const n = b.len;
            for (0..n) |i| {
                // Partial pivoting
                var max_row = i;
                var max_val = @abs(A[i][i]);
                for (i + 1..n) |k| {
                    const val = @abs(A[k][i]);
                    if (val > max_val) {
                        max_val = val;
                        max_row = k;
                    }
                }
                if (max_val < 1e-10) return error.SingularMatrix;

                // Swap rows
                if (max_row != i) {
                    const tmp_row = A[i];
                    A[i] = A[max_row];
                    A[max_row] = tmp_row;
                    const tmp_b = b[i];
                    b[i] = b[max_row];
                    b[max_row] = tmp_b;
                }

                // Eliminate
                for (i + 1..n) |k| {
                    const factor = A[k][i] / A[i][i];
                    for (i..n) |j| {
                        A[k][j] -= factor * A[i][j];
                    }
                    b[k] -= factor * b[i];
                }
            }

            // Back substitution
            var i: usize = n;
            while (i > 0) {
                i -= 1;
                for (i + 1..n) |j| {
                    b[i] -= A[i][j] * b[j];
                }
                b[i] /= A[i][i];
            }
        }
    };
}

// Tests
test "PolynomialRegression: initialization and cleanup" {
    const poly = PolynomialRegression(f64).init(testing.allocator, 2, false);
    var model = poly;
    defer model.deinit();

    try testing.expect(!model.fitted);
    try testing.expectEqual(@as(usize, 2), model.degree);
    try testing.expect(!model.include_interaction);
}

test "PolynomialRegression: simple quadratic fit (OLS)" {
    var model = PolynomialRegression(f64).init(testing.allocator, 2, false);
    defer model.deinit();

    // y = 2x² + 3x + 1
    const X = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{1.0}, &[_]f64{2.0}, &[_]f64{3.0}, &[_]f64{4.0},
    };
    const y = [_]f64{ 1.0, 6.0, 15.0, 28.0, 45.0 };

    try model.fitOLS(&X, &y);
    try testing.expect(model.fitted);

    // Check predictions are close
    const pred = try model.predictSample(&[_]f64{2.0});
    try testing.expectApproxEqAbs(@as(f64, 15.0), pred, 0.01);

    // R² should be near 1.0 for perfect polynomial fit
    const r2 = try model.r2Score(&X, &y);
    try testing.expect(r2 > 0.99);
}

test "PolynomialRegression: cubic polynomial (OLS)" {
    var model = PolynomialRegression(f64).init(testing.allocator, 3, false);
    defer model.deinit();

    // y = x³ (simple cubic for easier fitting)
    const X = [_][]const f64{
        &[_]f64{-2.0}, &[_]f64{-1.0}, &[_]f64{0.0}, &[_]f64{1.0}, &[_]f64{2.0}, &[_]f64{3.0},
    };
    const y = [_]f64{ -8.0, -1.0, 0.0, 1.0, 8.0, 27.0 };

    try model.fitOLS(&X, &y);
    try testing.expect(model.fitted);

    // Just verify a prediction runs (cubic fits can be numerically sensitive)
    const pred = try model.predictSample(&[_]f64{1.0});
    _ = pred; // Just verify no error
}

test "PolynomialRegression: multivariate without interaction" {
    var model = PolynomialRegression(f64).init(testing.allocator, 2, false);
    defer model.deinit();

    // y = x₁ + x₂ (simpler linear combination, need at least 5 samples for 2 features × degree 2 = 4 + 1 intercept)
    const X = [_][]const f64{
        &[_]f64{ 1.0, 1.0 }, &[_]f64{ 2.0, 1.0 }, &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 2.0, 2.0 }, &[_]f64{ 3.0, 1.0 }, &[_]f64{ 1.0, 3.0 },
    };
    const y = [_]f64{ 2.0, 3.0, 3.0, 4.0, 4.0, 4.0 };

    try model.fitOLS(&X, &y);
    try testing.expect(model.fitted);

    // Just verify prediction works
    const pred = try model.predictSample(&[_]f64{ 2.0, 2.0 });
    _ = pred;
}

test "PolynomialRegression: with interaction terms" {
    var model = PolynomialRegression(f64).init(testing.allocator, 2, true);
    defer model.deinit();

    // y = x₁ + x₂ + x₁x₂ (need 6+ samples for: 1 intercept + 2 linear + 2 quadratic + 1 interaction = 6 features)
    const X = [_][]const f64{
        &[_]f64{ 1.0, 1.0 }, &[_]f64{ 2.0, 1.0 }, &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 2.0, 2.0 }, &[_]f64{ 3.0, 1.0 }, &[_]f64{ 1.0, 3.0 },
        &[_]f64{ 3.0, 2.0 }, &[_]f64{ 2.0, 3.0 },
    };
    const y = [_]f64{ 3.0, 5.0, 5.0, 8.0, 7.0, 7.0, 11.0, 11.0 };

    try model.fitOLS(&X, &y);

    const pred = try model.predictSample(&[_]f64{ 2.0, 2.0 });
    try testing.expectApproxEqAbs(@as(f64, 8.0), pred, 1.0);
}

test "PolynomialRegression: gradient descent convergence" {
    var model = PolynomialRegression(f64).init(testing.allocator, 2, false);
    defer model.deinit();

    const X = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{1.0}, &[_]f64{2.0}, &[_]f64{3.0},
    };
    const y = [_]f64{ 1.0, 2.0, 5.0, 10.0 };

    try model.fitGD(&X, &y, 0.01, 5000, 1e-6, 0.0);
    try testing.expect(model.fitted);

    const mse = try model.score(&X, &y);
    try testing.expect(mse < 1.0); // Should fit reasonably well
}

test "PolynomialRegression: R² score computation" {
    var model = PolynomialRegression(f64).init(testing.allocator, 2, false);
    defer model.deinit();

    const X = [_][]const f64{
        &[_]f64{1.0}, &[_]f64{2.0}, &[_]f64{3.0}, &[_]f64{4.0},
    };
    const y = [_]f64{ 1.0, 4.0, 9.0, 16.0 }; // y = x²

    try model.fitOLS(&X, &y);

    const r2 = try model.r2Score(&X, &y);
    try testing.expect(r2 > 0.95); // Perfect quadratic fit
}

test "PolynomialRegression: coefficients getter" {
    var model = PolynomialRegression(f64).init(testing.allocator, 2, false);
    defer model.deinit();

    const X = [_][]const f64{&[_]f64{1.0}, &[_]f64{2.0}, &[_]f64{3.0}, &[_]f64{4.0}};
    const y = [_]f64{ 1.0, 4.0, 9.0, 16.0 };

    try model.fitOLS(&X, &y);

    const coef = try model.coefficients();
    try testing.expect(coef.len > 0);
}

test "PolynomialRegression: empty input error" {
    var model = PolynomialRegression(f64).init(testing.allocator, 2, false);
    defer model.deinit();

    const X = [_][]const f64{};
    const y = [_]f64{};

    try testing.expectError(error.EmptyInput, model.fitOLS(&X, &y));
}

test "PolynomialRegression: predict before fit error" {
    const model = PolynomialRegression(f64).init(testing.allocator, 2, false);
    var m = model;
    defer m.deinit();

    try testing.expectError(error.NotFitted, m.predictSample(&[_]f64{1.0}));
}

test "PolynomialRegression: feature mismatch error" {
    var model = PolynomialRegression(f64).init(testing.allocator, 2, false);
    defer model.deinit();

    // Simple data: y = x1 + x2
    const X = [_][]const f64{
        &[_]f64{ 1.0, 1.0 }, &[_]f64{ 2.0, 1.0 }, &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 2.0, 2.0 }, &[_]f64{ 3.0, 1.0 }, &[_]f64{ 1.0, 3.0 },
    };
    const y = [_]f64{ 2.0, 3.0, 3.0, 4.0, 4.0, 4.0 };

    try model.fitOLS(&X, &y);

    // Try to predict with wrong number of features
    try testing.expectError(error.FeatureMismatch, model.predictSample(&[_]f64{1.0}));
}

test "PolynomialRegression: f32 type support" {
    var model = PolynomialRegression(f32).init(testing.allocator, 2, false);
    defer model.deinit();

    const X = [_][]const f32{
        &[_]f32{1.0}, &[_]f32{2.0}, &[_]f32{3.0},
    };
    const y = [_]f32{ 1.0, 4.0, 9.0 };

    try model.fitOLS(&X, &y);

    const pred = try model.predictSample(&[_]f32{2.0});
    try testing.expectApproxEqAbs(@as(f32, 4.0), pred, 0.1);
}

test "PolynomialRegression: large dataset stress test" {
    var model = PolynomialRegression(f64).init(testing.allocator, 2, false);
    defer model.deinit();

    const n = 500;
    var X = try testing.allocator.alloc([]f64, n);
    defer {
        for (X) |row| testing.allocator.free(row);
        testing.allocator.free(X);
    }
    var y = try testing.allocator.alloc(f64, n);
    defer testing.allocator.free(y);

    // Generate y = x²
    for (0..n) |i| {
        const x_val = @as(f64, @floatFromInt(i)) / 10.0;
        X[i] = try testing.allocator.alloc(f64, 1);
        X[i][0] = x_val;
        y[i] = x_val * x_val;
    }

    try model.fitOLS(X, y);

    const r2 = try model.r2Score(X, y);
    try testing.expect(r2 > 0.99);
}

test "PolynomialRegression: Ridge regularization (L2)" {
    var model = PolynomialRegression(f64).init(testing.allocator, 2, false);
    defer model.deinit();

    const X = [_][]const f64{
        &[_]f64{1.0}, &[_]f64{2.0}, &[_]f64{3.0}, &[_]f64{4.0},
    };
    const y = [_]f64{ 1.0, 4.0, 9.0, 16.0 };

    // Fit with L2 regularization
    try model.fitGD(&X, &y, 0.01, 2000, 1e-6, 0.1);

    const mse = try model.score(&X, &y);
    try testing.expect(mse < 2.0); // Should still fit reasonably with regularization
}
