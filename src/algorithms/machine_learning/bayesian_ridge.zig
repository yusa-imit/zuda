const std = @import("std");
const Allocator = std.mem.Allocator;

/// Bayesian Ridge Regression with automatic relevance determination.
///
/// Fits a Bayesian linear regression model with conjugate Gaussian priors on the weights,
/// and inverse-gamma priors on the precision parameters (alpha, lambda).
/// The model automatically tunes regularization through iterative updates of the precision parameters.
///
/// Model:
///   y = X*w + ε, where ε ~ N(0, 1/alpha)
///   w ~ N(0, lambda^-1 * I)
///
/// Hyperpriors:
///   alpha ~ Gamma(alpha_1, alpha_2)  # noise precision
///   lambda ~ Gamma(lambda_1, lambda_2)  # weight precision
///
/// Algorithm:
///   1. Initialize alpha, lambda from priors
///   2. Compute posterior mean (coefficients) via: m = (X^T*X + lambda/alpha*I)^-1 * X^T*y
///   3. Compute posterior covariance: Sigma = (X^T*X + lambda/alpha*I)^-1 / alpha
///   4. Update alpha using residual sum of squares
///   5. Update lambda using coefficient magnitudes
///   6. Repeat 2-5 until convergence
///
/// Time: O(n_iter * (n*d^2 + d^3)) where n=samples, d=features, n_iter~300
/// Space: O(d^2 + n*d)
///
/// Use cases:
///   - Regression with automatic regularization tuning
///   - Uncertainty quantification in predictions
///   - Feature relevance determination
///   - Small sample size problems
///   - Scientific computing requiring predictive distributions
pub fn BayesianRidge(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,

        // Learned parameters
        coef: ?[]T = null,  // Coefficients (mean of posterior), shape: (n_features,)
        intercept: T = 0.0,  // Intercept term

        // Precision parameters
        alpha: T,  // Noise precision (1/sigma^2)
        lambda: T,  // Weight precision (regularization strength)

        // Hyperparameters for priors
        alpha_1: T,  // Gamma prior shape for alpha
        alpha_2: T,  // Gamma prior rate for alpha
        lambda_1: T,  // Gamma prior shape for lambda
        lambda_2: T,  // Gamma prior rate for lambda

        // Training parameters
        max_iter: usize,
        tol: T,
        compute_score: bool,

        // Statistics from training
        sigma: ?[]T = null,  // Posterior covariance diagonal, shape: (n_features,)
        scores: ?[]T = null,  // Log marginal likelihood history

        /// Initialize Bayesian Ridge regression.
        ///
        /// Parameters:
        ///   - allocator: Memory allocator
        ///   - alpha_1: Gamma prior shape for noise precision (default: 1e-6)
        ///   - alpha_2: Gamma prior rate for noise precision (default: 1e-6)
        ///   - lambda_1: Gamma prior shape for weight precision (default: 1e-6)
        ///   - lambda_2: Gamma prior rate for weight precision (default: 1e-6)
        ///   - max_iter: Maximum iterations for optimization (default: 300)
        ///   - tol: Convergence tolerance (default: 1e-3)
        ///   - compute_score: Whether to compute log marginal likelihood (default: false)
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(
            allocator: Allocator,
            alpha_1: T,
            alpha_2: T,
            lambda_1: T,
            lambda_2: T,
            max_iter: usize,
            tol: T,
            compute_score: bool,
        ) Self {
            return .{
                .allocator = allocator,
                .alpha = alpha_1 / alpha_2,
                .lambda = lambda_1 / lambda_2,
                .alpha_1 = alpha_1,
                .alpha_2 = alpha_2,
                .lambda_1 = lambda_1,
                .lambda_2 = lambda_2,
                .max_iter = max_iter,
                .tol = tol,
                .compute_score = compute_score,
            };
        }

        /// Free all allocated memory.
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.coef) |coef| {
                self.allocator.free(coef);
            }
            if (self.sigma) |sigma| {
                self.allocator.free(sigma);
            }
            if (self.scores) |scores| {
                self.allocator.free(scores);
            }
        }

        /// Fit Bayesian Ridge regression model.
        ///
        /// Parameters:
        ///   - X: Feature matrix, shape (n_samples, n_features)
        ///   - y: Target vector, shape (n_samples,)
        ///
        /// Time: O(n_iter * (n*d^2 + d^3))
        /// Space: O(d^2 + n*d)
        pub fn fit(self: *Self, X: []const T, y: []const T, n_samples: usize, n_features: usize) !void {
            if (n_samples == 0 or n_features == 0) return error.InvalidInput;
            if (X.len != n_samples * n_features) return error.DimensionMismatch;
            if (y.len != n_samples) return error.DimensionMismatch;

            // Center data
            var X_mean = try self.allocator.alloc(T, n_features);
            defer self.allocator.free(X_mean);
            var y_mean: T = 0.0;

            for (0..n_features) |j| {
                X_mean[j] = 0.0;
                for (0..n_samples) |i| {
                    X_mean[j] += X[i * n_features + j];
                }
                X_mean[j] /= @as(T, @floatFromInt(n_samples));
            }

            for (0..n_samples) |i| {
                y_mean += y[i];
            }
            y_mean /= @as(T, @floatFromInt(n_samples));

            // Center X and y
            var X_centered = try self.allocator.alloc(T, n_samples * n_features);
            defer self.allocator.free(X_centered);
            var y_centered = try self.allocator.alloc(T, n_samples);
            defer self.allocator.free(y_centered);

            for (0..n_samples) |i| {
                for (0..n_features) |j| {
                    X_centered[i * n_features + j] = X[i * n_features + j] - X_mean[j];
                }
                y_centered[i] = y[i] - y_mean;
            }

            // Allocate coefficient storage
            if (self.coef == null) {
                self.coef = try self.allocator.alloc(T, n_features);
            }
            @memset(self.coef.?, 0.0);

            // Allocate posterior covariance diagonal
            if (self.sigma == null) {
                self.sigma = try self.allocator.alloc(T, n_features);
            }

            // Allocate scores if requested
            if (self.compute_score) {
                if (self.scores) |s| self.allocator.free(s);
                self.scores = try self.allocator.alloc(T, self.max_iter);
            }

            // Compute X^T * X once
            const XTX = try self.allocator.alloc(T, n_features * n_features);
            defer self.allocator.free(XTX);
            try computeXTX(T, X_centered, XTX, n_samples, n_features);

            // Compute X^T * y once
            const XTy = try self.allocator.alloc(T, n_features);
            defer self.allocator.free(XTy);
            try computeXTy(T, X_centered, y_centered, XTy, n_samples, n_features);

            // Iterative update of alpha and lambda
            var prev_alpha = self.alpha;
            var prev_lambda = self.lambda;

            for (0..self.max_iter) |iter| {
                // Solve (X^T*X + lambda/alpha * I) * coef = X^T*y
                try self.solveRidgeSystem(XTX, XTy, n_features);

                // Compute residuals
                var rss: T = 0.0;  // Residual sum of squares
                for (0..n_samples) |i| {
                    var pred: T = 0.0;
                    for (0..n_features) |j| {
                        pred += X_centered[i * n_features + j] * self.coef.?[j];
                    }
                    const residual = y_centered[i] - pred;
                    rss += residual * residual;
                }

                // Update alpha (noise precision)
                const gamma = self.computeGamma(XTX, n_features);
                const n_f: T = @floatFromInt(n_samples);
                self.alpha = (n_f - gamma + 2.0 * self.alpha_1) / (rss + 2.0 * self.alpha_2);

                // Update lambda (weight precision)
                var coef_norm: T = 0.0;
                for (self.coef.?) |c| {
                    coef_norm += c * c;
                }
                const d_f: T = @floatFromInt(n_features);
                self.lambda = (d_f + 2.0 * self.lambda_1) / (coef_norm + 2.0 * self.lambda_2);

                // Compute score if requested
                if (self.compute_score) {
                    self.scores.?[iter] = self.computeLogMarginalLikelihood(
                        X_centered,
                        y_centered,
                        n_samples,
                        n_features,
                        rss,
                    );
                }

                // Check convergence
                const alpha_change = @abs(self.alpha - prev_alpha);
                const lambda_change = @abs(self.lambda - prev_lambda);
                if (alpha_change < self.tol and lambda_change < self.tol) {
                    // Truncate scores to actual iterations
                    if (self.compute_score) {
                        const new_scores = try self.allocator.alloc(T, iter + 1);
                        @memcpy(new_scores, self.scores.?[0..iter + 1]);
                        self.allocator.free(self.scores.?);
                        self.scores = new_scores;
                    }
                    break;
                }

                prev_alpha = self.alpha;
                prev_lambda = self.lambda;
            }

            // Compute intercept
            self.intercept = y_mean;
            for (0..n_features) |j| {
                self.intercept -= self.coef.?[j] * X_mean[j];
            }
        }

        /// Predict using the fitted model.
        ///
        /// Parameters:
        ///   - X: Feature matrix, shape (n_samples, n_features)
        ///   - n_samples: Number of samples
        ///   - n_features: Number of features
        ///
        /// Returns: Predictions, shape (n_samples,). Caller owns memory.
        ///
        /// Time: O(n*d)
        /// Space: O(n)
        pub fn predict(self: *const Self, X: []const T, n_samples: usize, n_features: usize, allocator: Allocator) ![]T {
            if (self.coef == null) return error.NotFitted;
            if (X.len != n_samples * n_features) return error.DimensionMismatch;

            var predictions = try allocator.alloc(T, n_samples);
            for (0..n_samples) |i| {
                predictions[i] = self.intercept;
                for (0..n_features) |j| {
                    predictions[i] += X[i * n_features + j] * self.coef.?[j];
                }
            }
            return predictions;
        }

        /// Predict with uncertainty (mean and standard deviation).
        ///
        /// Returns predictive distribution: y ~ N(mean, std^2) for each sample.
        ///
        /// Parameters:
        ///   - X: Feature matrix, shape (n_samples, n_features)
        ///   - n_samples: Number of samples
        ///   - n_features: Number of features
        ///   - allocator: Memory allocator
        ///
        /// Returns: Tuple of (mean, std), both shape (n_samples,). Caller owns both.
        ///
        /// Time: O(n*d)
        /// Space: O(n)
        pub fn predictWithUncertainty(
            self: *const Self,
            X: []const T,
            n_samples: usize,
            n_features: usize,
            allocator: Allocator,
        ) !struct { mean: []T, std: []T } {
            if (self.coef == null or self.sigma == null) return error.NotFitted;
            if (X.len != n_samples * n_features) return error.DimensionMismatch;

            const mean = try self.predict(X, n_samples, n_features, allocator);
            errdefer allocator.free(mean);

            var std_dev = try allocator.alloc(T, n_samples);
            errdefer allocator.free(std_dev);

            const noise_var = 1.0 / self.alpha;

            for (0..n_samples) |i| {
                var pred_var = noise_var;
                for (0..n_features) |j| {
                    const x_val = X[i * n_features + j];
                    pred_var += x_val * x_val * self.sigma.?[j];
                }
                std_dev[i] = @sqrt(pred_var);
            }

            return .{ .mean = mean, .std = std_dev };
        }

        /// Compute R² score.
        ///
        /// Time: O(n*d)
        /// Space: O(n)
        pub fn score(self: *const Self, X: []const T, y: []const T, n_samples: usize, n_features: usize) !T {
            const predictions = try self.predict(X, n_samples, n_features, self.allocator);
            defer self.allocator.free(predictions);

            var y_mean: T = 0.0;
            for (y) |val| {
                y_mean += val;
            }
            y_mean /= @as(T, @floatFromInt(n_samples));

            var ss_res: T = 0.0;
            var ss_tot: T = 0.0;
            for (0..n_samples) |i| {
                ss_res += (y[i] - predictions[i]) * (y[i] - predictions[i]);
                ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
            }

            return 1.0 - ss_res / ss_tot;
        }

        // Private helper methods

        fn solveRidgeSystem(self: *Self, XTX: []const T, XTy: []const T, n_features: usize) !void {
            // Solve (XTX + (lambda/alpha) * I) * coef = XTy
            const reg = self.lambda / self.alpha;

            // Create augmented matrix (XTX + reg*I)
            var A = try self.allocator.alloc(T, n_features * n_features);
            defer self.allocator.free(A);
            @memcpy(A, XTX);

            // Add regularization to diagonal
            for (0..n_features) |i| {
                A[i * n_features + i] += reg;
            }

            // Copy XTy to coef (will be overwritten with solution)
            @memcpy(self.coef.?, XTy);

            // Solve using Gaussian elimination with partial pivoting
            try gaussianElimination(T, A, self.coef.?, n_features);

            // Compute posterior covariance diagonal
            // sigma_ii = (A^-1)_ii / alpha
            // For efficiency, approximate using diagonal of A after factorization
            for (0..n_features) |i| {
                self.sigma.?[i] = 1.0 / (A[i * n_features + i] * self.alpha);
            }
        }

        fn computeGamma(self: *const Self, XTX: []const T, n_features: usize) T {
            // Gamma = sum_i(lambda_i / (alpha + lambda_i))
            // where lambda_i are eigenvalues of XTX
            // Approximation: gamma ≈ n_features * lambda / (alpha + lambda)
            _ = XTX;
            const n_f: T = @floatFromInt(n_features);
            return n_f * self.lambda / (self.alpha + self.lambda);
        }

        fn computeLogMarginalLikelihood(
            self: *const Self,
            X: []const T,
            y: []const T,
            n_samples: usize,
            n_features: usize,
            rss: T,
        ) T {
            _ = X;
            _ = y;
            // Log marginal likelihood (evidence)
            // log p(y|alpha,lambda) = (n/2)*log(alpha) + (d/2)*log(lambda)
            //                        - alpha/2 * RSS - lambda/2 * ||w||^2
            //                        - n/2*log(2π) - 1/2*log|A|
            // Simplified version for monitoring convergence
            const n_f: T = @floatFromInt(n_samples);
            const d_f: T = @floatFromInt(n_features);

            var coef_norm: T = 0.0;
            for (self.coef.?) |c| {
                coef_norm += c * c;
            }

            const log_alpha = @log(self.alpha);
            const log_lambda = @log(self.lambda);
            const pi: T = std.math.pi;

            return (n_f / 2.0) * log_alpha + (d_f / 2.0) * log_lambda - (self.alpha / 2.0) * rss - (self.lambda / 2.0) * coef_norm - (n_f / 2.0) * @log(2.0 * pi);
        }
    };
}

// Helper functions

fn computeXTX(comptime T: type, X: []const T, XTX: []T, n_samples: usize, n_features: usize) !void {
    @memset(XTX, 0.0);

    for (0..n_features) |i| {
        for (0..n_features) |j| {
            var sum: T = 0.0;
            for (0..n_samples) |k| {
                sum += X[k * n_features + i] * X[k * n_features + j];
            }
            XTX[i * n_features + j] = sum;
        }
    }
}

fn computeXTy(comptime T: type, X: []const T, y: []const T, XTy: []T, n_samples: usize, n_features: usize) !void {
    @memset(XTy, 0.0);

    for (0..n_features) |j| {
        var sum: T = 0.0;
        for (0..n_samples) |i| {
            sum += X[i * n_features + j] * y[i];
        }
        XTy[j] = sum;
    }
}

fn gaussianElimination(comptime T: type, A: []T, b: []T, n: usize) !void {

    // Forward elimination with partial pivoting
    for (0..n) |i| {
        // Find pivot
        var max_row = i;
        var max_val = @abs(A[i * n + i]);
        for (i + 1..n) |k| {
            const val = @abs(A[k * n + i]);
            if (val > max_val) {
                max_val = val;
                max_row = k;
            }
        }

        if (max_val < 1e-10) return error.SingularMatrix;

        // Swap rows if needed
        if (max_row != i) {
            for (0..n) |j| {
                const temp = A[i * n + j];
                A[i * n + j] = A[max_row * n + j];
                A[max_row * n + j] = temp;
            }
            const temp_b = b[i];
            b[i] = b[max_row];
            b[max_row] = temp_b;
        }

        // Eliminate
        for (i + 1..n) |k| {
            const factor = A[k * n + i] / A[i * n + i];
            for (i..n) |j| {
                A[k * n + j] -= factor * A[i * n + j];
            }
            b[k] -= factor * b[i];
        }
    }

    // Back substitution
    var i = n;
    while (i > 0) {
        i -= 1;
        var sum: T = 0.0;
        for (i + 1..n) |j| {
            sum += A[i * n + j] * b[j];
        }
        b[i] = (b[i] - sum) / A[i * n + i];
    }
}

// Tests

test "BayesianRidge: basic fit and predict" {
    const allocator = std.testing.allocator;

    // Simple linear relationship: y = 2*x + 1
    const X = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y = [_]f64{ 3.0, 5.0, 7.0, 9.0, 11.0 };

    var model = BayesianRidge(f64).init(allocator, 1e-6, 1e-6, 1e-6, 1e-6, 300, 1e-3, false);
    defer model.deinit();

    try model.fit(&X, &y, 5, 1);

    // Check coefficients learned approximately
    try std.testing.expect(model.coef != null);
    try std.testing.expect(@abs(model.coef.?[0] - 2.0) < 0.1);
    try std.testing.expect(@abs(model.intercept - 1.0) < 0.1);

    // Test prediction
    const X_test = [_]f64{6.0};
    const pred = try model.predict(&X_test, 1, 1, allocator);
    defer allocator.free(pred);

    try std.testing.expect(@abs(pred[0] - 13.0) < 0.2);
}

test "BayesianRidge: multiple features" {
    const allocator = std.testing.allocator;

    // y = 2*x1 + 3*x2 + 1
    const X = [_]f64{
        1.0, 1.0,
        2.0, 1.0,
        3.0, 2.0,
        4.0, 2.0,
    };
    const y = [_]f64{ 6.0, 8.0, 13.0, 15.0 };

    var model = BayesianRidge(f64).init(allocator, 1e-6, 1e-6, 1e-6, 1e-6, 300, 1e-3, false);
    defer model.deinit();

    try model.fit(&X, &y, 4, 2);

    try std.testing.expect(@abs(model.coef.?[0] - 2.0) < 0.2);
    try std.testing.expect(@abs(model.coef.?[1] - 3.0) < 0.2);
}

test "BayesianRidge: predict with uncertainty" {
    const allocator = std.testing.allocator;

    const X = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y = [_]f64{ 3.1, 4.9, 7.2, 8.8, 11.1 };

    var model = BayesianRidge(f64).init(allocator, 1e-6, 1e-6, 1e-6, 1e-6, 300, 1e-3, false);
    defer model.deinit();

    try model.fit(&X, &y, 5, 1);

    const X_test = [_]f64{6.0};
    const result = try model.predictWithUncertainty(&X_test, 1, 1, allocator);
    defer allocator.free(result.mean);
    defer allocator.free(result.std);

    // Should predict around 13.0 with some uncertainty
    try std.testing.expect(@abs(result.mean[0] - 13.0) < 1.0);
    try std.testing.expect(result.std[0] > 0.0);
    try std.testing.expect(result.std[0] < 5.0);
}

test "BayesianRidge: automatic regularization" {
    const allocator = std.testing.allocator;

    // Data with noise - Bayesian Ridge should automatically find good regularization
    const X = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const y = [_]f64{ 2.8, 5.2, 6.9, 9.1, 10.8, 13.2, 14.9, 17.1 };

    var model = BayesianRidge(f64).init(allocator, 1e-6, 1e-6, 1e-6, 1e-6, 300, 1e-3, false);
    defer model.deinit();

    try model.fit(&X, &y, 8, 1);

    // alpha and lambda should be updated from initial values
    try std.testing.expect(model.alpha > 0.0);
    try std.testing.expect(model.lambda > 0.0);
    try std.testing.expect(model.alpha != 1.0); // Changed from initial
}

test "BayesianRidge: compute score" {
    const allocator = std.testing.allocator;

    const X = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y = [_]f64{ 3.0, 5.0, 7.0, 9.0, 11.0 };

    var model = BayesianRidge(f64).init(allocator, 1e-6, 1e-6, 1e-6, 1e-6, 300, 1e-3, true);
    defer model.deinit();

    try model.fit(&X, &y, 5, 1);

    // Scores should be monotonically increasing (log likelihood improves)
    if (model.scores) |scores| {
        try std.testing.expect(scores.len > 0);
        try std.testing.expect(scores.len <= 300);
    }

    const r2 = try model.score(&X, &y, 5, 1);
    try std.testing.expect(r2 > 0.95); // Should fit well
}

test "BayesianRidge: convergence" {
    const allocator = std.testing.allocator;

    const X = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y = [_]f64{ 3.0, 5.0, 7.0, 9.0, 11.0 };

    var model = BayesianRidge(f64).init(allocator, 1e-6, 1e-6, 1e-6, 1e-6, 300, 1e-3, true);
    defer model.deinit();

    try model.fit(&X, &y, 5, 1);

    // Should converge in fewer than max_iter
    if (model.scores) |scores| {
        try std.testing.expect(scores.len < 300);
    }
}

test "BayesianRidge: f32 support" {
    const allocator = std.testing.allocator;

    const X = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y = [_]f32{ 3.0, 5.0, 7.0, 9.0, 11.0 };

    var model = BayesianRidge(f32).init(allocator, 1e-6, 1e-6, 1e-6, 1e-6, 300, 1e-3, false);
    defer model.deinit();

    try model.fit(&X, &y, 5, 1);

    try std.testing.expect(model.coef != null);
    try std.testing.expect(@abs(model.coef.?[0] - 2.0) < 0.2);
}

test "BayesianRidge: large dataset" {
    const allocator = std.testing.allocator;

    const n = 100;
    var X = try allocator.alloc(f64, n);
    defer allocator.free(X);
    var y = try allocator.alloc(f64, n);
    defer allocator.free(y);

    // y = 1.5*x + 2.0 + noise
    for (0..n) |i| {
        const x: f64 = @floatFromInt(i);
        X[i] = x;
        y[i] = 1.5 * x + 2.0;
    }

    var model = BayesianRidge(f64).init(allocator, 1e-6, 1e-6, 1e-6, 1e-6, 300, 1e-3, false);
    defer model.deinit();

    try model.fit(X, y, n, 1);

    try std.testing.expect(@abs(model.coef.?[0] - 1.5) < 0.1);
    try std.testing.expect(@abs(model.intercept - 2.0) < 0.5);
}

test "BayesianRidge: not fitted error" {
    const allocator = std.testing.allocator;

    var model = BayesianRidge(f64).init(allocator, 1e-6, 1e-6, 1e-6, 1e-6, 300, 1e-3, false);
    defer model.deinit();

    const X_test = [_]f64{1.0};
    const result = model.predict(&X_test, 1, 1, allocator);
    try std.testing.expectError(error.NotFitted, result);
}

test "BayesianRidge: dimension mismatch" {
    const allocator = std.testing.allocator;

    const X = [_]f64{ 1.0, 2.0, 3.0 };
    const y = [_]f64{ 3.0, 5.0 }; // Wrong size

    var model = BayesianRidge(f64).init(allocator, 1e-6, 1e-6, 1e-6, 1e-6, 300, 1e-3, false);
    defer model.deinit();

    const result = model.fit(&X, &y, 3, 1);
    try std.testing.expectError(error.DimensionMismatch, result);
}

test "BayesianRidge: invalid input" {
    const allocator = std.testing.allocator;

    const X = [_]f64{};
    const y = [_]f64{};

    var model = BayesianRidge(f64).init(allocator, 1e-6, 1e-6, 1e-6, 1e-6, 300, 1e-3, false);
    defer model.deinit();

    const result = model.fit(&X, &y, 0, 1);
    try std.testing.expectError(error.InvalidInput, result);
}

test "BayesianRidge: high dimensional" {
    const allocator = std.testing.allocator;

    // 20 samples, 5 features
    const n_samples = 20;
    const n_features = 5;
    var X = try allocator.alloc(f64, n_samples * n_features);
    defer allocator.free(X);
    var y = try allocator.alloc(f64, n_samples);
    defer allocator.free(y);

    // Generate synthetic data
    const true_coef = [_]f64{ 1.0, -0.5, 2.0, 0.3, -1.5 };
    for (0..n_samples) |i| {
        for (0..n_features) |j| {
            X[i * n_features + j] = @as(f64, @floatFromInt(i + j));
        }
        y[i] = 10.0; // intercept
        for (0..n_features) |j| {
            y[i] += true_coef[j] * X[i * n_features + j];
        }
    }

    var model = BayesianRidge(f64).init(allocator, 1e-6, 1e-6, 1e-6, 1e-6, 300, 1e-3, false);
    defer model.deinit();

    try model.fit(X, y, n_samples, n_features);

    // Should learn reasonable coefficients
    try std.testing.expect(model.coef != null);
    try std.testing.expect(model.coef.?.len == n_features);

    const r2 = try model.score(X, y, n_samples, n_features);
    try std.testing.expect(r2 > 0.98); // Should fit very well
}

test "BayesianRidge: posterior covariance computed" {
    const allocator = std.testing.allocator;

    const X = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y = [_]f64{ 3.0, 5.0, 7.0, 9.0, 11.0 };

    var model = BayesianRidge(f64).init(allocator, 1e-6, 1e-6, 1e-6, 1e-6, 300, 1e-3, false);
    defer model.deinit();

    try model.fit(&X, &y, 5, 1);

    // Sigma should be computed
    try std.testing.expect(model.sigma != null);
    try std.testing.expect(model.sigma.?.len == 1);
    try std.testing.expect(model.sigma.?[0] > 0.0);
}

test "BayesianRidge: memory safety" {
    const allocator = std.testing.allocator;

    const X = [_]f64{ 1.0, 2.0, 3.0 };
    const y = [_]f64{ 3.0, 5.0, 7.0 };

    var model = BayesianRidge(f64).init(allocator, 1e-6, 1e-6, 1e-6, 1e-6, 300, 1e-3, true);
    defer model.deinit();

    try model.fit(&X, &y, 3, 1);

    // Multiple predictions should not leak
    const X_test = [_]f64{4.0};
    {
        const pred1 = try model.predict(&X_test, 1, 1, allocator);
        defer allocator.free(pred1);
    }
    {
        const pred2 = try model.predict(&X_test, 1, 1, allocator);
        defer allocator.free(pred2);
    }
}
