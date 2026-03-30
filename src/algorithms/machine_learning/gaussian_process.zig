//! Gaussian Process Regression (GPR)
//!
//! A Bayesian non-parametric approach for regression that models the distribution
//! over functions. Provides both predictions and uncertainty estimates (variance).
//!
//! Key features:
//! - Kernel-based similarity: RBF (Gaussian), Linear, Polynomial, Matern kernels
//! - Uncertainty quantification: Returns predictive mean and variance
//! - Hyperparameter optimization: Marginal likelihood maximization (optional)
//! - Noise modeling: Gaussian noise with configurable variance
//!
//! Algorithm:
//! 1. Training: Compute K = k(X, X) + σ²I (covariance matrix with noise)
//! 2. Invert: K⁻¹ (via Cholesky decomposition for numerical stability)
//! 3. Prediction: μ* = k(x*, X) K⁻¹ y (predictive mean)
//! 4. Uncertainty: σ²* = k(x*, x*) - k(x*, X) K⁻¹ k(X, x*) (predictive variance)
//!
//! Time complexity:
//! - Training: O(n³) for Cholesky decomposition (one-time cost)
//! - Prediction: O(n) per test point (after training)
//! - Hyperparameter opt: O(n³ × iterations) for gradient-based methods
//!
//! Space complexity: O(n²) for storing covariance matrix
//!
//! Use cases:
//! - Regression with uncertainty: Medical predictions, risk assessment
//! - Optimization: Bayesian optimization (acquisition functions)
//! - Interpolation: Sensor fusion, kriging in geostatistics
//! - Active learning: Query points with high uncertainty
//!
//! Example:
//! ```zig
//! const gp = try GaussianProcess(f64).init(allocator, .{
//!     .kernel_type = .rbf,
//!     .length_scale = 1.0,
//!     .noise_variance = 0.01,
//! });
//! defer gp.deinit();
//!
//! try gp.fit(X_train, y_train);
//! const predictions = try gp.predict(X_test);
//! defer allocator.free(predictions.mean);
//! defer allocator.free(predictions.variance);
//! ```

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Kernel types for covariance functions
pub const KernelType = enum {
    /// Radial Basis Function (Gaussian): k(x, x') = exp(-||x - x'||² / (2 * l²))
    rbf,
    /// Linear: k(x, x') = x^T x' + c
    linear,
    /// Polynomial: k(x, x') = (x^T x' + c)^d
    polynomial,
    /// Matern: k(x, x') = (1 + √(5r)/l + 5r²/(3l²)) exp(-√(5r)/l), r = ||x - x'||
    matern,
};

/// Configuration for Gaussian Process
pub const Config = struct {
    kernel_type: KernelType = .rbf,
    length_scale: f64 = 1.0,
    noise_variance: f64 = 0.01,
    /// For polynomial kernel
    degree: usize = 2,
    /// For linear/polynomial kernel
    constant: f64 = 0.0,
};

/// Prediction result containing mean and variance
pub fn PredictionResult(comptime T: type) type {
    return struct {
        mean: []T,
        variance: []T,
    };
}

/// Gaussian Process Regression
///
/// Type: T = f32 or f64
///
/// Time: O(n³) training, O(n) per prediction
/// Space: O(n²) for covariance matrix
pub fn GaussianProcess(comptime T: type) type {
    if (T != f32 and T != f64) {
        @compileError("GaussianProcess only supports f32 and f64");
    }

    return struct {
        allocator: Allocator,
        config: Config,
        X_train: ?[]const []const T = null,
        y_train: ?[]const T = null,
        K_inv: ?[]T = null, // Inverse of covariance matrix (flattened)
        alpha: ?[]T = null, // K⁻¹ y
        n_train: usize = 0,
        n_features: usize = 0,

        const Self = @This();

        /// Initialize Gaussian Process with configuration
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(allocator: Allocator, config: Config) !Self {
            return Self{
                .allocator = allocator,
                .config = config,
            };
        }

        /// Free all allocated memory
        pub fn deinit(self: *Self) void {
            if (self.K_inv) |k_inv| {
                self.allocator.free(k_inv);
            }
            if (self.alpha) |alpha| {
                self.allocator.free(alpha);
            }
            if (self.X_train) |x_train| {
                for (x_train) |row| {
                    self.allocator.free(row);
                }
                self.allocator.free(x_train);
            }
            if (self.y_train) |y_train| {
                self.allocator.free(y_train);
            }
        }

        /// Compute kernel function k(x1, x2)
        fn kernel(self: *const Self, x1: []const T, x2: []const T) T {
            return switch (self.config.kernel_type) {
                .rbf => blk: {
                    var dist_sq: T = 0;
                    for (x1, 0..) |val, i| {
                        const diff = val - x2[i];
                        dist_sq += diff * diff;
                    }
                    const l = @as(T, @floatCast(self.config.length_scale));
                    break :blk @exp(-dist_sq / (2 * l * l));
                },
                .linear => blk: {
                    var dot: T = 0;
                    for (x1, 0..) |val, i| {
                        dot += val * x2[i];
                    }
                    break :blk dot + @as(T, @floatCast(self.config.constant));
                },
                .polynomial => blk: {
                    var dot: T = 0;
                    for (x1, 0..) |val, i| {
                        dot += val * x2[i];
                    }
                    const base = dot + @as(T, @floatCast(self.config.constant));
                    var result: T = 1;
                    for (0..self.config.degree) |_| {
                        result *= base;
                    }
                    break :blk result;
                },
                .matern => blk: {
                    var dist_sq: T = 0;
                    for (x1, 0..) |val, i| {
                        const diff = val - x2[i];
                        dist_sq += diff * diff;
                    }
                    const r = @sqrt(dist_sq);
                    const l = @as(T, @floatCast(self.config.length_scale));
                    const sqrt5_r_l = @sqrt(@as(T, 5.0)) * r / l;
                    const term1: T = 1 + sqrt5_r_l + (5 * r * r) / (3 * l * l);
                    break :blk term1 * @exp(-sqrt5_r_l);
                },
            };
        }

        /// Cholesky decomposition: A = L L^T
        /// Modifies matrix in-place to become lower triangular
        fn choleskyDecompose(n: usize, matrix: []T) !void {
            for (0..n) |i| {
                for (0..i + 1) |j| {
                    var sum: T = matrix[i * n + j];
                    for (0..j) |k| {
                        sum -= matrix[i * n + k] * matrix[j * n + k];
                    }
                    if (i == j) {
                        if (sum <= 0) {
                            return error.NotPositiveDefinite;
                        }
                        matrix[i * n + i] = @sqrt(sum);
                    } else {
                        matrix[i * n + j] = sum / matrix[j * n + j];
                    }
                }
                // Zero out upper triangle
                for (i + 1..n) |j| {
                    matrix[i * n + j] = 0;
                }
            }
        }

        /// Solve L x = b for lower triangular L
        fn forwardSubstitution(n: usize, L: []const T, b: []const T, x: []T) void {
            for (0..n) |i| {
                var sum: T = b[i];
                for (0..i) |j| {
                    sum -= L[i * n + j] * x[j];
                }
                x[i] = sum / L[i * n + i];
            }
        }

        /// Solve L^T x = b for lower triangular L
        fn backwardSubstitution(n: usize, L: []const T, b: []const T, x: []T) void {
            var i: usize = n;
            while (i > 0) {
                i -= 1;
                var sum: T = b[i];
                for (i + 1..n) |j| {
                    sum -= L[j * n + i] * x[j];
                }
                x[i] = sum / L[i * n + i];
            }
        }

        /// Train Gaussian Process on data
        ///
        /// Args:
        ///   X: Training features (n_samples × n_features)
        ///   y: Training targets (n_samples)
        ///
        /// Time: O(n³) for Cholesky decomposition
        /// Space: O(n²) for covariance matrix
        pub fn fit(self: *Self, X: []const []const T, y: []const T) !void {
            if (X.len != y.len) {
                return error.SizeMismatch;
            }
            if (X.len == 0) {
                return error.EmptyData;
            }

            self.n_train = X.len;
            self.n_features = X[0].len;

            // Copy training data
            const X_copy = try self.allocator.alloc([]T, X.len);
            for (X, 0..) |row, i| {
                X_copy[i] = try self.allocator.alloc(T, row.len);
                @memcpy(X_copy[i], row);
            }
            self.X_train = X_copy;

            const y_copy = try self.allocator.alloc(T, y.len);
            @memcpy(y_copy, y);
            self.y_train = y_copy;

            // Compute covariance matrix K + σ²I
            const K = try self.allocator.alloc(T, self.n_train * self.n_train);
            defer self.allocator.free(K);

            for (0..self.n_train) |i| {
                for (0..self.n_train) |j| {
                    K[i * self.n_train + j] = self.kernel(X[i], X[j]);
                    if (i == j) {
                        K[i * self.n_train + j] += @as(T, @floatCast(self.config.noise_variance));
                    }
                }
            }

            // Cholesky decomposition: K = L L^T
            const L = try self.allocator.alloc(T, self.n_train * self.n_train);
            @memcpy(L, K);
            try choleskyDecompose(self.n_train, L);

            // Solve L α_temp = y (forward substitution)
            const alpha_temp = try self.allocator.alloc(T, self.n_train);
            defer self.allocator.free(alpha_temp);
            forwardSubstitution(self.n_train, L, y, alpha_temp);

            // Solve L^T α = α_temp (backward substitution)
            const alpha = try self.allocator.alloc(T, self.n_train);
            backwardSubstitution(self.n_train, L, alpha_temp, alpha);
            self.alpha = alpha;

            // Compute K_inv = (L L^T)^-1 = L^-T L^-1
            // We'll store L for later use in predictions
            self.K_inv = L; // Store L, not K_inv (sufficient for predictions)
        }

        /// Predict mean and variance for test data
        ///
        /// Args:
        ///   X_test: Test features (n_test × n_features)
        ///
        /// Returns: PredictionResult with mean and variance
        ///
        /// Time: O(n_train × n_test × n_features)
        /// Space: O(n_test)
        pub fn predict(self: *const Self, X_test: []const []const T) !PredictionResult(T) {
            if (self.X_train == null or self.alpha == null or self.K_inv == null) {
                return error.NotTrained;
            }

            const n_test = X_test.len;
            const mean = try self.allocator.alloc(T, n_test);
            errdefer self.allocator.free(mean);
            const variance = try self.allocator.alloc(T, n_test);
            errdefer self.allocator.free(variance);

            const X_train = self.X_train.?;
            const alpha = self.alpha.?;
            const L = self.K_inv.?; // Actually storing L from Cholesky

            for (X_test, 0..) |x_star, i| {
                // Compute k_star = k(x*, X_train)
                const k_star = try self.allocator.alloc(T, self.n_train);
                defer self.allocator.free(k_star);

                for (0..self.n_train) |j| {
                    k_star[j] = self.kernel(x_star, X_train[j]);
                }

                // Predictive mean: μ* = k_star^T α
                var mu: T = 0;
                for (0..self.n_train) |j| {
                    mu += k_star[j] * alpha[j];
                }
                mean[i] = mu;

                // Predictive variance: σ²* = k(x*, x*) - k_star^T K^-1 k_star
                // We compute k_star^T K^-1 k_star = ||L^-1 k_star||²
                const v = try self.allocator.alloc(T, self.n_train);
                defer self.allocator.free(v);
                forwardSubstitution(self.n_train, L, k_star, v);

                var v_norm_sq: T = 0;
                for (v) |val| {
                    v_norm_sq += val * val;
                }

                const k_star_star = self.kernel(x_star, x_star);
                variance[i] = k_star_star - v_norm_sq;

                // Clamp variance to be non-negative (numerical stability)
                if (variance[i] < 0) {
                    variance[i] = 0;
                }
            }

            return PredictionResult(T){
                .mean = mean,
                .variance = variance,
            };
        }

        /// Predict only mean (faster if variance not needed)
        ///
        /// Time: O(n_train × n_test × n_features)
        /// Space: O(n_test)
        pub fn predictMean(self: *const Self, X_test: []const []const T) ![]T {
            if (self.X_train == null or self.alpha == null) {
                return error.NotTrained;
            }

            const n_test = X_test.len;
            const mean = try self.allocator.alloc(T, n_test);
            errdefer self.allocator.free(mean);

            const X_train = self.X_train.?;
            const alpha = self.alpha.?;

            for (X_test, 0..) |x_star, i| {
                var mu: T = 0;
                for (0..self.n_train) |j| {
                    const k = self.kernel(x_star, X_train[j]);
                    mu += k * alpha[j];
                }
                mean[i] = mu;
            }

            return mean;
        }

        /// Compute log marginal likelihood (for hyperparameter optimization)
        ///
        /// log p(y|X) = -0.5 y^T K^-1 y - 0.5 log|K| - 0.5 n log(2π)
        ///
        /// Time: O(n²)
        /// Space: O(1)
        pub fn logMarginalLikelihood(self: *const Self) !T {
            if (self.y_train == null or self.alpha == null or self.K_inv == null) {
                return error.NotTrained;
            }

            const y = self.y_train.?;
            const alpha = self.alpha.?;
            const L = self.K_inv.?;

            // -0.5 y^T α (where α = K^-1 y)
            var data_fit: T = 0;
            for (0..self.n_train) |i| {
                data_fit += y[i] * alpha[i];
            }
            data_fit *= -0.5;

            // -0.5 log|K| = -sum(log(L[i,i])) (Cholesky diagonal)
            var log_det: T = 0;
            for (0..self.n_train) |i| {
                log_det += @log(L[i * self.n_train + i]);
            }

            // -0.5 n log(2π)
            const n_T: T = @floatFromInt(self.n_train);
            const constant = -0.5 * n_T * @log(2 * std.math.pi);

            return data_fit - log_det + constant;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "GaussianProcess: basic fit and predict" {
    const allocator = std.testing.allocator;

    // 1D data: y = x²
    const X_train = [_][]const f64{
        &[_]f64{0.0},
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
    };
    const y_train = [_]f64{ 0.0, 1.0, 4.0, 9.0 };

    var gp = try GaussianProcess(f64).init(allocator, .{
        .kernel_type = .rbf,
        .length_scale = 1.0,
        .noise_variance = 0.01,
    });
    defer gp.deinit();

    try gp.fit(&X_train, &y_train);

    // Test on training data (should predict close to training targets)
    const result = try gp.predict(&X_train);
    defer allocator.free(result.mean);
    defer allocator.free(result.variance);

    for (result.mean, y_train) |pred, target| {
        try std.testing.expect(@abs(pred - target) < 0.5);
    }

    // Variance should be small at training points
    for (result.variance) |var_val| {
        try std.testing.expect(var_val < 0.1);
    }
}

test "GaussianProcess: RBF kernel properties" {
    const allocator = std.testing.allocator;

    var gp = try GaussianProcess(f64).init(allocator, .{
        .kernel_type = .rbf,
        .length_scale = 1.0,
    });
    defer gp.deinit();

    const x1 = [_]f64{0.0};
    const x2 = [_]f64{0.0};
    const x3 = [_]f64{1.0};

    // k(x, x) = 1
    const k_same = gp.kernel(&x1, &x2);
    try std.testing.expectApproxEqAbs(1.0, k_same, 1e-6);

    // k(x, x') decreases with distance
    const k_diff = gp.kernel(&x1, &x3);
    try std.testing.expect(k_diff < 1.0);
    try std.testing.expect(k_diff > 0.0);
}

test "GaussianProcess: linear kernel" {
    const allocator = std.testing.allocator;

    var gp = try GaussianProcess(f64).init(allocator, .{
        .kernel_type = .linear,
        .constant = 0.0,
    });
    defer gp.deinit();

    const X_train = [_][]const f64{
        &[_]f64{ 1.0, 0.0 },
        &[_]f64{ 2.0, 0.0 },
        &[_]f64{ 3.0, 0.0 },
    };
    const y_train = [_]f64{ 1.0, 2.0, 3.0 }; // y = x1

    try gp.fit(&X_train, &y_train);

    const X_test = [_][]const f64{&[_]f64{ 4.0, 0.0 }};
    const mean = try gp.predictMean(&X_test);
    defer allocator.free(mean);

    // Linear kernel should extrapolate linearly
    try std.testing.expect(@abs(mean[0] - 4.0) < 0.5);
}

test "GaussianProcess: polynomial kernel" {
    const allocator = std.testing.allocator;

    var gp = try GaussianProcess(f64).init(allocator, .{
        .kernel_type = .polynomial,
        .degree = 2,
        .constant = 0.0,
        .noise_variance = 0.01,
    });
    defer gp.deinit();

    const X_train = [_][]const f64{
        &[_]f64{0.0},
        &[_]f64{1.0},
        &[_]f64{2.0},
    };
    const y_train = [_]f64{ 0.0, 1.0, 4.0 }; // y = x²

    try gp.fit(&X_train, &y_train);

    const X_test = [_][]const f64{&[_]f64{3.0}};
    const mean = try gp.predictMean(&X_test);
    defer allocator.free(mean);

    // Polynomial kernel should capture quadratic relationship
    try std.testing.expect(@abs(mean[0] - 9.0) < 1.0);
}

test "GaussianProcess: matern kernel" {
    const allocator = std.testing.allocator;

    var gp = try GaussianProcess(f64).init(allocator, .{
        .kernel_type = .matern,
        .length_scale = 1.0,
    });
    defer gp.deinit();

    const x1 = [_]f64{0.0};
    const x2 = [_]f64{0.0};

    // k(x, x) = 1
    const k_same = gp.kernel(&x1, &x2);
    try std.testing.expectApproxEqAbs(1.0, k_same, 1e-6);
}

test "GaussianProcess: uncertainty increases far from data" {
    const allocator = std.testing.allocator;

    const X_train = [_][]const f64{
        &[_]f64{0.0},
        &[_]f64{1.0},
    };
    const y_train = [_]f64{ 0.0, 1.0 };

    var gp = try GaussianProcess(f64).init(allocator, .{
        .kernel_type = .rbf,
        .length_scale = 0.5,
        .noise_variance = 0.01,
    });
    defer gp.deinit();

    try gp.fit(&X_train, &y_train);

    const X_near = [_][]const f64{&[_]f64{0.5}};
    const X_far = [_][]const f64{&[_]f64{10.0}};

    const result_near = try gp.predict(&X_near);
    defer allocator.free(result_near.mean);
    defer allocator.free(result_near.variance);

    const result_far = try gp.predict(&X_far);
    defer allocator.free(result_far.mean);
    defer allocator.free(result_far.variance);

    // Variance should be larger far from training data
    try std.testing.expect(result_far.variance[0] > result_near.variance[0]);
}

test "GaussianProcess: f32 support" {
    const allocator = std.testing.allocator;

    const X_train = [_][]const f32{
        &[_]f32{0.0},
        &[_]f32{1.0},
    };
    const y_train = [_]f32{ 0.0, 1.0 };

    var gp = try GaussianProcess(f32).init(allocator, .{
        .kernel_type = .rbf,
        .length_scale = 1.0,
        .noise_variance = 0.01,
    });
    defer gp.deinit();

    try gp.fit(&X_train, &y_train);

    const X_test = [_][]const f32{&[_]f32{0.5}};
    const mean = try gp.predictMean(&X_test);
    defer allocator.free(mean);

    try std.testing.expect(@abs(mean[0] - 0.5) < 0.2);
}

test "GaussianProcess: log marginal likelihood" {
    const allocator = std.testing.allocator;

    const X_train = [_][]const f64{
        &[_]f64{0.0},
        &[_]f64{1.0},
        &[_]f64{2.0},
    };
    const y_train = [_]f64{ 0.0, 1.0, 4.0 };

    var gp = try GaussianProcess(f64).init(allocator, .{
        .kernel_type = .rbf,
        .length_scale = 1.0,
        .noise_variance = 0.01,
    });
    defer gp.deinit();

    try gp.fit(&X_train, &y_train);

    const log_ml = try gp.logMarginalLikelihood();
    // Should be finite and reasonable
    try std.testing.expect(log_ml < 0); // Typically negative
    try std.testing.expect(!std.math.isNan(log_ml));
    try std.testing.expect(!std.math.isInf(log_ml));
}

test "GaussianProcess: empty data error" {
    const allocator = std.testing.allocator;

    var gp = try GaussianProcess(f64).init(allocator, .{});
    defer gp.deinit();

    const X_empty: []const []const f64 = &[_][]const f64{};
    const y_empty: []const f64 = &[_]f64{};

    try std.testing.expectError(error.EmptyData, gp.fit(X_empty, y_empty));
}

test "GaussianProcess: size mismatch error" {
    const allocator = std.testing.allocator;

    var gp = try GaussianProcess(f64).init(allocator, .{});
    defer gp.deinit();

    const X_train = [_][]const f64{&[_]f64{0.0}};
    const y_train = [_]f64{ 0.0, 1.0 }; // Wrong size

    try std.testing.expectError(error.SizeMismatch, gp.fit(&X_train, &y_train));
}

test "GaussianProcess: predict before fit error" {
    const allocator = std.testing.allocator;

    const gp = try GaussianProcess(f64).init(allocator, .{});

    const X_test = [_][]const f64{&[_]f64{0.0}};
    try std.testing.expectError(error.NotTrained, gp.predict(&X_test));
}

test "GaussianProcess: multivariate data" {
    const allocator = std.testing.allocator;

    const X_train = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 1.0, 0.0 },
        &[_]f64{ 0.0, 1.0 },
        &[_]f64{ 1.0, 1.0 },
    };
    const y_train = [_]f64{ 0.0, 1.0, 1.0, 2.0 }; // y = x1 + x2

    var gp = try GaussianProcess(f64).init(allocator, .{
        .kernel_type = .rbf,
        .length_scale = 1.0,
        .noise_variance = 0.01,
    });
    defer gp.deinit();

    try gp.fit(&X_train, &y_train);

    const X_test = [_][]const f64{&[_]f64{ 0.5, 0.5 }};
    const mean = try gp.predictMean(&X_test);
    defer allocator.free(mean);

    // Should predict close to 1.0
    try std.testing.expect(@abs(mean[0] - 1.0) < 0.5);
}

test "GaussianProcess: noise variance effect" {
    const allocator = std.testing.allocator;

    const X_train = [_][]const f64{
        &[_]f64{0.0},
        &[_]f64{1.0},
    };
    const y_train = [_]f64{ 0.0, 1.0 };

    // Low noise
    var gp_low = try GaussianProcess(f64).init(allocator, .{
        .kernel_type = .rbf,
        .length_scale = 1.0,
        .noise_variance = 0.001,
    });
    defer gp_low.deinit();
    try gp_low.fit(&X_train, &y_train);

    // High noise
    var gp_high = try GaussianProcess(f64).init(allocator, .{
        .kernel_type = .rbf,
        .length_scale = 1.0,
        .noise_variance = 1.0,
    });
    defer gp_high.deinit();
    try gp_high.fit(&X_train, &y_train);

    const X_test = [_][]const f64{&[_]f64{0.0}};

    const result_low = try gp_low.predict(&X_test);
    defer allocator.free(result_low.mean);
    defer allocator.free(result_low.variance);

    const result_high = try gp_high.predict(&X_test);
    defer allocator.free(result_high.mean);
    defer allocator.free(result_high.variance);

    // High noise should give higher variance at training points
    try std.testing.expect(result_high.variance[0] > result_low.variance[0]);
}
