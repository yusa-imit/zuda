/// Support Vector Regression (SVR)
///
/// SVR is a regression algorithm based on Support Vector Machines, using an epsilon-insensitive loss function.
/// It finds a function that deviates from the actual targets by at most epsilon, while being as flat as possible.
///
/// Key features:
/// - Epsilon-insensitive loss: only penalize errors larger than epsilon
/// - Kernel support: linear, RBF, polynomial
/// - Sparse solution: only support vectors matter
/// - L2 regularization via C parameter (smaller C = more regularization)
///
/// Algorithm: Sequential Minimal Optimization (SMO) variant for regression
/// - Dual formulation: minimize w.r.t. dual variables α, α*
/// - KKT conditions for optimality
/// - Iteratively update pairs of dual variables
/// - Support vectors: samples with |α_i - α_i*| > 0
///
/// Time: O(n² × iter) training, O(n_sv) prediction where n_sv = support vectors
/// Space: O(n²) for kernel matrix (can be reduced with chunking)
///
/// Use cases:
/// - Non-linear regression with kernel trick
/// - Robust regression with epsilon-tube (outlier tolerance)
/// - Function approximation with sparsity
/// - Time series forecasting
///
/// Trade-offs:
/// - vs Linear Regression: Non-linear via kernels, sparse, robust to outliers, but O(n²) slower
/// - vs Ridge Regression: Sparse solution, epsilon-tube vs L2 loss, kernel support
/// - vs Gaussian Process: Deterministic, no uncertainty quantification, faster for large n
///
const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;

/// SVR Model
pub fn SVR(comptime T: type) type {
    return struct {
        allocator: Allocator,
        kernel_type: KernelType,
        C: T, // regularization parameter (smaller = more regularization)
        epsilon: T, // epsilon-tube tolerance
        gamma: T, // for RBF/polynomial kernels
        degree: usize, // for polynomial kernel
        coef0: T, // for polynomial kernel

        // Training data (needed for kernel evaluation)
        X_train: []const []const T,
        y_train: []const T,

        // Dual variables
        alpha: []T, // α_i (positive slack)
        alpha_star: []T, // α_i* (negative slack)
        b: T, // bias term

        // Support vectors (indices where |α_i - α_i*| > tolerance)
        support_vectors: []usize,
        n_support: usize,

        const Self = @This();

        /// Initialize SVR model
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(
            allocator: Allocator,
            kernel_type: KernelType,
            C: T,
            epsilon: T,
            gamma: T,
            degree: usize,
            coef0: T,
        ) Self {
            return .{
                .allocator = allocator,
                .kernel_type = kernel_type,
                .C = C,
                .epsilon = epsilon,
                .gamma = gamma,
                .degree = degree,
                .coef0 = coef0,
                .X_train = &.{},
                .y_train = &.{},
                .alpha = &.{},
                .alpha_star = &.{},
                .b = 0,
                .support_vectors = &.{},
                .n_support = 0,
            };
        }

        /// Free allocated memory
        ///
        /// Time: O(n)
        /// Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.X_train.len > 0) {
                // Need to cast away const to free
                const X_mut = @constCast(self.X_train);
                for (X_mut) |row| {
                    self.allocator.free(row);
                }
                self.allocator.free(X_mut);
            }
            if (self.y_train.len > 0) {
                const y_mut = @constCast(self.y_train);
                self.allocator.free(y_mut);
            }
            if (self.alpha.len > 0) self.allocator.free(self.alpha);
            if (self.alpha_star.len > 0) self.allocator.free(self.alpha_star);
            if (self.support_vectors.len > 0) self.allocator.free(self.support_vectors);
        }

        /// Compute kernel function between two samples
        ///
        /// Time: O(d) for linear/RBF, O(d × degree) for polynomial
        /// Space: O(1)
        fn kernel(self: *const Self, x1: []const T, x2: []const T) T {
            return switch (self.kernel_type) {
                .linear => linearKernel(T, x1, x2),
                .rbf => rbfKernel(T, x1, x2, self.gamma),
                .polynomial => polynomialKernel(T, x1, x2, self.gamma, self.degree, self.coef0),
            };
        }

        /// Train SVR model using SMO algorithm
        ///
        /// Time: O(n² × iter) where iter depends on convergence
        /// Space: O(n²) for kernel matrix + O(n) for dual variables
        pub fn fit(self: *Self, X: []const []const T, y: []const T, max_iter: usize) !void {
            const n = X.len;
            if (n == 0 or y.len != n) return error.InvalidInput;
            const d = X[0].len;

            // Copy training data
            const X_copy = try self.allocator.alloc([]T, n);
            errdefer {
                for (X_copy[0..n]) |row| self.allocator.free(row);
                self.allocator.free(X_copy);
            }
            for (X, 0..) |row, i| {
                X_copy[i] = try self.allocator.alloc(T, d);
                @memcpy(X_copy[i], row);
            }
            self.X_train = X_copy;

            const y_copy = try self.allocator.alloc(T, n);
            errdefer self.allocator.free(y_copy);
            @memcpy(y_copy, y);
            self.y_train = y_copy;

            // Initialize dual variables
            self.alpha = try self.allocator.alloc(T, n);
            errdefer self.allocator.free(self.alpha);
            @memset(self.alpha, 0);

            self.alpha_star = try self.allocator.alloc(T, n);
            errdefer self.allocator.free(self.alpha_star);
            @memset(self.alpha_star, 0);

            self.b = 0;

            // Precompute kernel matrix
            const K = try self.allocator.alloc([]T, n);
            defer {
                for (K) |row| self.allocator.free(row);
                self.allocator.free(K);
            }
            for (K, 0..) |*row, i| {
                row.* = try self.allocator.alloc(T, n);
                for (0..n) |j| {
                    row.*[j] = self.kernel(X[i], X[j]);
                }
            }

            // SMO optimization
            const tol: T = 1e-3;
            var iter: usize = 0;
            while (iter < max_iter) : (iter += 1) {
                var num_changed: usize = 0;

                // Examine all training examples
                for (0..n) |i| {
                    const E_i = try self.computeError(K, i);
                    const r_i = E_i;

                    // Check KKT conditions
                    const alpha_i = self.alpha[i];
                    const alpha_star_i = self.alpha_star[i];

                    // KKT violations for updating
                    const violates_upper = (alpha_i < self.C and r_i < -tol) or (alpha_star_i < self.C and r_i > tol);
                    const violates_lower = (alpha_i > 0 and r_i > tol) or (alpha_star_i > 0 and r_i < -tol);

                    if (violates_upper or violates_lower) {
                        // Select second index using heuristic
                        const j = try self.selectSecondIndex(K, i, E_i, n);
                        if (j == i) continue;

                        const E_j = try self.computeError(K, j);

                        // Update alpha pair
                        const updated = try self.updateAlphaPair(K, i, j, E_i, E_j);
                        if (updated) {
                            num_changed += 1;
                        }
                    }
                }

                // Check convergence
                if (num_changed == 0) break;
            }

            // Compute bias term
            try self.computeBias(K);

            // Identify support vectors
            try self.identifySupportVectors();
        }

        /// Compute prediction error for sample i
        fn computeError(self: *const Self, K: []const []const T, i: usize) !T {
            var sum: T = 0;
            for (0..self.y_train.len) |j| {
                sum += (self.alpha[j] - self.alpha_star[j]) * K[j][i];
            }
            return sum + self.b - self.y_train[i];
        }

        /// Select second index for optimization (heuristic: max |E_i - E_j|)
        fn selectSecondIndex(self: *const Self, K: []const []const T, i: usize, E_i: T, n: usize) !usize {
            var j: usize = 0;
            var max_diff: T = 0;

            for (0..n) |idx| {
                if (idx == i) continue;
                const E_j = try self.computeError(K, idx);
                const diff = @abs(E_i - E_j);
                if (diff > max_diff) {
                    max_diff = diff;
                    j = idx;
                }
            }

            return j;
        }

        /// Update alpha pair (i, j) using SMO (simplified for regression)
        fn updateAlphaPair(self: *Self, K: []const []const T, i: usize, j: usize, E_i: T, _: T) !bool {
            const eta = K[i][i] + K[j][j] - 2 * K[i][j];
            if (eta <= 0.001) return false; // Non-positive definite kernel or numerical issue

            // Old alphas
            const alpha_i_old = self.alpha[i];
            const alpha_star_i_old = self.alpha_star[i];

            // Simplified update: focus on sample i, use gradient
            const gradient = E_i;

            // Update based on gradient and epsilon-tube
            if (gradient > self.epsilon and alpha_star_i_old < self.C) {
                // Increase alpha_star (prediction too high)
                const delta = @min((gradient - self.epsilon) / eta, self.C - alpha_star_i_old);
                self.alpha_star[i] = alpha_star_i_old + delta * 0.5; // Damped update
                self.alpha[i] = 0;
            } else if (gradient < -self.epsilon and alpha_i_old < self.C) {
                // Increase alpha (prediction too low)
                const delta = @min((-gradient - self.epsilon) / eta, self.C - alpha_i_old);
                self.alpha[i] = alpha_i_old + delta * 0.5; // Damped update
                self.alpha_star[i] = 0;
            } else {
                // Inside epsilon-tube or at bound
                return false;
            }

            // Check if significant change
            const change = @abs(self.alpha[i] - alpha_i_old) + @abs(self.alpha_star[i] - alpha_star_i_old);
            return change > 1e-6;
        }

        /// Compute bias term from support vectors
        fn computeBias(self: *Self, K: []const []const T) !void {
            var sum: T = 0;
            var count: usize = 0;

            for (0..self.y_train.len) |i| {
                // Use samples with 0 < α < C or 0 < α* < C
                const is_sv = (self.alpha[i] > 0 and self.alpha[i] < self.C) or
                    (self.alpha_star[i] > 0 and self.alpha_star[i] < self.C);

                if (is_sv) {
                    var prediction: T = 0;
                    for (0..self.y_train.len) |j| {
                        prediction += (self.alpha[j] - self.alpha_star[j]) * K[j][i];
                    }
                    sum += self.y_train[i] - prediction;
                    count += 1;
                }
            }

            if (count > 0) {
                self.b = sum / @as(T, @floatFromInt(count));
            }
        }

        /// Identify support vectors (samples with |α - α*| > 0)
        fn identifySupportVectors(self: *Self) !void {
            const tol: T = 1e-5;
            var sv_list = try std.ArrayList(usize).initCapacity(self.allocator, self.y_train.len);
            defer sv_list.deinit(self.allocator);

            for (0..self.y_train.len) |i| {
                if (@abs(self.alpha[i] - self.alpha_star[i]) > tol) {
                    sv_list.appendAssumeCapacity(i);
                }
            }

            self.support_vectors = try sv_list.toOwnedSlice(self.allocator);
            self.n_support = self.support_vectors.len;
        }

        /// Predict for a single sample
        ///
        /// Time: O(n_sv × d) where n_sv = number of support vectors
        /// Space: O(1)
        pub fn predictSingle(self: *const Self, x: []const T) !T {
            if (self.X_train.len == 0) return error.ModelNotFitted;

            var sum: T = 0;
            for (0..self.y_train.len) |i| {
                const coef = self.alpha[i] - self.alpha_star[i];
                if (@abs(coef) > 1e-10) { // Only compute for support vectors
                    sum += coef * self.kernel(self.X_train[i], x);
                }
            }
            return sum + self.b;
        }

        /// Predict for multiple samples
        ///
        /// Time: O(m × n_sv × d)
        /// Space: O(m) for predictions
        pub fn predict(self: *const Self, X: []const []const T) ![]T {
            const predictions = try self.allocator.alloc(T, X.len);
            errdefer self.allocator.free(predictions);

            for (X, 0..) |x, i| {
                predictions[i] = try self.predictSingle(x);
            }

            return predictions;
        }

        /// Compute R² score
        ///
        /// Time: O(m × n_sv × d)
        /// Space: O(m)
        pub fn score(self: *const Self, X: []const []const T, y: []const T) !T {
            if (X.len != y.len) return error.DimensionMismatch;

            const predictions = try self.predict(X);
            defer self.allocator.free(predictions);

            // Compute mean of y
            var y_mean: T = 0;
            for (y) |val| y_mean += val;
            y_mean /= @as(T, @floatFromInt(y.len));

            // Compute SS_tot and SS_res
            var ss_tot: T = 0;
            var ss_res: T = 0;
            for (y, predictions) |y_true, y_pred| {
                ss_tot += (y_true - y_mean) * (y_true - y_mean);
                ss_res += (y_true - y_pred) * (y_true - y_pred);
            }

            if (ss_tot == 0) return 1.0; // Perfect prediction
            return 1.0 - ss_res / ss_tot;
        }

        /// Get number of support vectors
        pub fn getSupportVectorCount(self: *const Self) usize {
            return self.n_support;
        }

        /// Get support vector indices
        pub fn getSupportVectors(self: *const Self) []const usize {
            return self.support_vectors;
        }
    };
}

pub const KernelType = enum {
    linear,
    rbf,
    polynomial,
};

/// Linear kernel: x1 · x2
fn linearKernel(comptime T: type, x1: []const T, x2: []const T) T {
    var sum: T = 0;
    for (x1, x2) |a, b| sum += a * b;
    return sum;
}

/// RBF kernel: exp(-gamma * ||x1 - x2||²)
fn rbfKernel(comptime T: type, x1: []const T, x2: []const T, gamma: T) T {
    var sq_dist: T = 0;
    for (x1, x2) |a, b| {
        const diff = a - b;
        sq_dist += diff * diff;
    }
    return @exp(-gamma * sq_dist);
}

/// Polynomial kernel: (gamma * x1·x2 + coef0)^degree
fn polynomialKernel(comptime T: type, x1: []const T, x2: []const T, gamma: T, degree: usize, coef0: T) T {
    var dot: T = 0;
    for (x1, x2) |a, b| dot += a * b;
    return math.pow(T, gamma * dot + coef0, @as(T, @floatFromInt(degree)));
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "SVR: basic linear regression" {
    const allocator = testing.allocator;

    // Linear data: y = 2x + 1
    const X = [_][]const f64{
        &.{0.0}, &.{1.0}, &.{2.0}, &.{3.0}, &.{4.0},
    };
    const y = [_]f64{ 1.0, 3.0, 5.0, 7.0, 9.0 };

    var svr = SVR(f64).init(allocator, .linear, 1.0, 0.1, 1.0, 3, 0.0);
    defer svr.deinit();

    try svr.fit(&X, &y, 100);

    // Test predictions
    const pred = try svr.predictSingle(&.{2.5});
    try testing.expect(@abs(pred - 6.0) < 1.0); // Approximate check

    // Test score
    const r2 = try svr.score(&X, &y);
    try testing.expect(r2 > 0.8); // Good fit
}

test "SVR: non-linear with RBF kernel" {
    const allocator = testing.allocator;

    // Quadratic data: y = x²
    const X = [_][]const f64{
        &.{-2.0}, &.{-1.0}, &.{0.0}, &.{1.0}, &.{2.0},
    };
    const y = [_]f64{ 4.0, 1.0, 0.0, 1.0, 4.0 };

    var svr = SVR(f64).init(allocator, .rbf, 10.0, 0.1, 0.5, 3, 0.0);
    defer svr.deinit();

    try svr.fit(&X, &y, 200);

    // Test predictions on training data
    const pred = try svr.predictSingle(&.{1.5});
    try testing.expect(@abs(pred - 2.25) < 1.0); // y = 1.5² = 2.25

    const r2 = try svr.score(&X, &y);
    try testing.expect(r2 > 0.7); // Decent fit for non-linear
}

test "SVR: polynomial kernel" {
    const allocator = testing.allocator;

    // Linear data
    const X = [_][]const f64{
        &.{1.0}, &.{2.0}, &.{3.0}, &.{4.0},
    };
    const y = [_]f64{ 2.0, 4.0, 6.0, 8.0 };

    var svr = SVR(f64).init(allocator, .polynomial, 1.0, 0.1, 1.0, 2, 0.0);
    defer svr.deinit();

    try svr.fit(&X, &y, 100);

    const pred = try svr.predictSingle(&.{2.5});
    try testing.expect(@abs(pred - 5.0) < 1.0);
}

test "SVR: multiple features" {
    const allocator = testing.allocator;

    // y = 2x₁ + 3x₂
    const X = [_][]const f64{
        &.{ 1.0, 1.0 },
        &.{ 2.0, 1.0 },
        &.{ 1.0, 2.0 },
        &.{ 2.0, 2.0 },
    };
    const y = [_]f64{ 5.0, 7.0, 8.0, 10.0 };

    var svr = SVR(f64).init(allocator, .linear, 1.0, 0.1, 1.0, 3, 0.0);
    defer svr.deinit();

    try svr.fit(&X, &y, 100);

    const pred = try svr.predictSingle(&.{ 1.5, 1.5 });
    try testing.expect(@abs(pred - 7.5) < 1.0); // 2*1.5 + 3*1.5 = 7.5
}

test "SVR: batch prediction" {
    const allocator = testing.allocator;

    const X_train = [_][]const f64{
        &.{1.0}, &.{2.0}, &.{3.0},
    };
    const y_train = [_]f64{ 2.0, 4.0, 6.0 };

    var svr = SVR(f64).init(allocator, .linear, 1.0, 0.1, 1.0, 3, 0.0);
    defer svr.deinit();

    try svr.fit(&X_train, &y_train, 100);

    const X_test = [_][]const f64{
        &.{1.5}, &.{2.5},
    };
    const predictions = try svr.predict(&X_test);
    defer allocator.free(predictions);

    try testing.expect(predictions.len == 2);
    try testing.expect(@abs(predictions[0] - 3.0) < 1.0);
    try testing.expect(@abs(predictions[1] - 5.0) < 1.0);
}

test "SVR: support vectors" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &.{0.0}, &.{1.0}, &.{2.0}, &.{3.0}, &.{4.0},
    };
    const y = [_]f64{ 0.0, 2.0, 4.0, 6.0, 8.0 };

    var svr = SVR(f64).init(allocator, .linear, 1.0, 0.1, 1.0, 3, 0.0);
    defer svr.deinit();

    try svr.fit(&X, &y, 100);

    const n_sv = svr.getSupportVectorCount();
    try testing.expect(n_sv > 0); // Should have support vectors
    try testing.expect(n_sv <= X.len); // At most n support vectors

    const sv_indices = svr.getSupportVectors();
    try testing.expect(sv_indices.len == n_sv);
}

test "SVR: epsilon parameter effect" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &.{1.0}, &.{2.0}, &.{3.0}, &.{4.0},
    };
    const y = [_]f64{ 1.0, 2.0, 3.0, 4.0 };

    // Small epsilon - tighter fit
    var svr_small = SVR(f64).init(allocator, .linear, 1.0, 0.01, 1.0, 3, 0.0);
    defer svr_small.deinit();
    try svr_small.fit(&X, &y, 100);
    const n_sv_small = svr_small.getSupportVectorCount();

    // Large epsilon - looser fit
    var svr_large = SVR(f64).init(allocator, .linear, 1.0, 1.0, 1.0, 3, 0.0);
    defer svr_large.deinit();
    try svr_large.fit(&X, &y, 100);
    const n_sv_large = svr_large.getSupportVectorCount();

    // Larger epsilon should use fewer support vectors (looser constraints)
    try testing.expect(n_sv_large <= n_sv_small);
}

test "SVR: C parameter regularization" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &.{1.0}, &.{2.0}, &.{3.0}, &.{4.0}, &.{5.0},
    };
    const y = [_]f64{ 1.0, 2.1, 2.9, 4.2, 5.1 }; // Slight noise

    // Small C - more regularization
    var svr_reg = SVR(f64).init(allocator, .linear, 0.1, 0.1, 1.0, 3, 0.0);
    defer svr_reg.deinit();
    try svr_reg.fit(&X, &y, 100);

    // Large C - less regularization
    var svr_no_reg = SVR(f64).init(allocator, .linear, 10.0, 0.1, 1.0, 3, 0.0);
    defer svr_no_reg.deinit();
    try svr_no_reg.fit(&X, &y, 100);

    // Both should work
    const r2_reg = try svr_reg.score(&X, &y);
    const r2_no_reg = try svr_no_reg.score(&X, &y);
    try testing.expect(r2_reg > 0.5);
    try testing.expect(r2_no_reg > 0.5);
}

test "SVR: f32 support" {
    const allocator = testing.allocator;

    const X = [_][]const f32{
        &.{1.0}, &.{2.0}, &.{3.0},
    };
    const y = [_]f32{ 2.0, 4.0, 6.0 };

    var svr = SVR(f32).init(allocator, .linear, 1.0, 0.1, 1.0, 3, 0.0);
    defer svr.deinit();

    try svr.fit(&X, &y, 100);

    const pred = try svr.predictSingle(&.{2.5});
    try testing.expect(@abs(pred - 5.0) < 1.0);
}

test "SVR: large dataset stress test" {
    const allocator = testing.allocator;

    // Generate 100 samples: y = 3x + noise
    const n: usize = 100;
    var X = try allocator.alloc([]f64, n);
    defer {
        for (X) |row| allocator.free(row);
        allocator.free(X);
    }
    var y = try allocator.alloc(f64, n);
    defer allocator.free(y);

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (0..n) |i| {
        X[i] = try allocator.alloc(f64, 1);
        const x_val = @as(f64, @floatFromInt(i)) / 10.0;
        X[i][0] = x_val;
        y[i] = 3.0 * x_val + (random.float(f64) - 0.5) * 0.5; // Small noise
    }

    var svr = SVR(f64).init(allocator, .linear, 1.0, 0.1, 1.0, 3, 0.0);
    defer svr.deinit();

    try svr.fit(X, y, 50); // Limited iterations for stress test

    const r2 = try svr.score(X, y);
    try testing.expect(r2 > 0.7); // Decent fit
}

test "SVR: empty data error" {
    const allocator = testing.allocator;

    const X = [_][]const f64{};
    const y = [_]f64{};

    var svr = SVR(f64).init(allocator, .linear, 1.0, 0.1, 1.0, 3, 0.0);
    defer svr.deinit();

    try testing.expectError(error.InvalidInput, svr.fit(&X, &y, 100));
}

test "SVR: dimension mismatch error" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &.{1.0}, &.{2.0},
    };
    const y = [_]f64{1.0}; // Mismatched length

    var svr = SVR(f64).init(allocator, .linear, 1.0, 0.1, 1.0, 3, 0.0);
    defer svr.deinit();

    try testing.expectError(error.InvalidInput, svr.fit(&X, &y, 100));
}

test "SVR: predict before fit error" {
    const allocator = testing.allocator;

    var svr = SVR(f64).init(allocator, .linear, 1.0, 0.1, 1.0, 3, 0.0);
    defer svr.deinit();

    const X_test = [_][]const f64{
        &.{1.0},
    };

    try testing.expectError(error.ModelNotFitted, svr.predict(&X_test));
}

test "SVR: memory safety" {
    const allocator = testing.allocator;

    const X = [_][]const f64{
        &.{1.0}, &.{2.0}, &.{3.0},
    };
    const y = [_]f64{ 2.0, 4.0, 6.0 };

    var svr = SVR(f64).init(allocator, .linear, 1.0, 0.1, 1.0, 3, 0.0);
    try svr.fit(&X, &y, 100);

    const pred = try svr.predict(&X);
    allocator.free(pred);

    svr.deinit();
    // No leaks expected
}
