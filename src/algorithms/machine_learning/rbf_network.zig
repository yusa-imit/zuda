/// Radial Basis Function (RBF) Network
///
/// A neural network that uses radial basis functions (typically Gaussian) as activation functions.
/// Architecture: input layer → RBF hidden layer → linear output layer.
///
/// Algorithm:
/// 1. Select centers (random, K-Means, or specified)
/// 2. Compute widths (σ) from nearest neighbor distances
/// 3. Compute RBF activations: φ(x) = exp(-||x - c||² / (2σ²))
/// 4. Solve linear system: W = (Φ^T Φ + λI)^(-1) Φ^T Y (Ridge regression)
///
/// Time complexity:
/// - Training: O(n_centers × n × d) for K-Means + O(n_centers × n × d) for activations + O(n_centers³) for linear solve
/// - Prediction: O(n_centers × d) per sample
///
/// Space complexity: O(n_centers × d) for centers + O(n_centers × n_outputs) for weights
///
/// Trade-offs:
/// - vs MLP: Simpler (only one hidden layer), faster training, but less expressive
/// - vs SVM: Similar kernel-based approach, but RBF solves regression directly
/// - vs K-Means: RBF provides smooth interpolation, K-Means only assigns clusters
///
/// Use cases:
/// - Function approximation
/// - Time series prediction
/// - Pattern recognition
/// - Control systems
/// - Interpolation with noise

const std = @import("std");
const Allocator = std.mem.Allocator;

/// RBF Network configuration
pub const Config = struct {
    /// Number of RBF centers (hidden layer size)
    n_centers: usize,
    /// Regularization parameter (λ) for Ridge regression, prevents overfitting
    /// Larger values → more regularization → smoother predictions
    lambda: f64 = 1e-6,
    /// Center selection method
    center_method: CenterMethod = .kmeans,
    /// Width calculation method
    width_method: WidthMethod = .nearest_neighbor,
    /// Fixed width value (used if width_method == .fixed)
    fixed_width: ?f64 = null,
    /// Maximum K-Means iterations (if center_method == .kmeans)
    max_kmeans_iter: usize = 100,
    /// K-Means convergence tolerance
    kmeans_tol: f64 = 1e-4,
};

/// Center selection method
pub const CenterMethod = enum {
    /// Random selection from training data
    random,
    /// K-Means clustering
    kmeans,
};

/// Width calculation method
pub const WidthMethod = enum {
    /// Based on average distance to nearest center
    nearest_neighbor,
    /// Fixed width for all centers
    fixed,
};

/// RBF Network
pub fn RBFNetwork(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        config: Config,
        centers: [][]T, // [n_centers][n_features]
        widths: []T, // [n_centers]
        weights: [][]T, // [n_centers + 1][n_outputs] (includes bias)
        n_features: usize,
        n_outputs: usize,
        fitted: bool,

        /// Initialize RBF network
        /// Time: O(1)
        pub fn init(allocator: Allocator, config: Config) Self {
            return .{
                .allocator = allocator,
                .config = config,
                .centers = &[_][]T{},
                .widths = &[_]T{},
                .weights = &[_][]T{},
                .n_features = 0,
                .n_outputs = 0,
                .fitted = false,
            };
        }

        /// Free resources
        pub fn deinit(self: *Self) void {
            for (self.centers) |center| {
                self.allocator.free(center);
            }
            self.allocator.free(self.centers);

            self.allocator.free(self.widths);

            for (self.weights) |weight_row| {
                self.allocator.free(weight_row);
            }
            self.allocator.free(self.weights);
        }

        /// Train the RBF network
        /// X: [n_samples][n_features] training data
        /// y: [n_samples][n_outputs] target values
        /// Time: O(n_centers × n × d + n_centers³)
        /// Space: O(n_centers × d + n_centers × n_outputs)
        pub fn fit(self: *Self, X: []const []const T, y: []const []const T) !void {
            if (X.len == 0 or y.len == 0) return error.EmptyData;
            if (X.len != y.len) return error.DimensionMismatch;
            if (X[0].len == 0 or y[0].len == 0) return error.EmptyData;
            if (self.config.n_centers > X.len) return error.TooManyCenters;

            const n_samples = X.len;
            self.n_features = X[0].len;
            self.n_outputs = y[0].len;

            // 1. Select centers
            try self.selectCenters(X);

            // 2. Compute widths
            try self.computeWidths(X);

            // 3. Compute RBF activations
            const phi = try self.allocator.alloc([]T, n_samples);
            defer {
                for (phi) |row| self.allocator.free(row);
                self.allocator.free(phi);
            }

            for (phi, 0..) |*row, i| {
                row.* = try self.computeActivations(X[i]);
            }

            // 4. Solve linear system: W = (Φ^T Φ + λI)^(-1) Φ^T Y
            try self.solveWeights(phi, y);

            self.fitted = true;
        }

        /// Predict for a single sample
        /// Time: O(n_centers × d)
        pub fn predict(self: *const Self, x: []const T) ![]T {
            if (!self.fitted) return error.NotFitted;
            if (x.len != self.n_features) return error.DimensionMismatch;

            // Compute RBF activations
            const activations = try self.computeActivations(x);
            defer self.allocator.free(activations);

            // Compute output: y = W^T φ(x)
            const output = try self.allocator.alloc(T, self.n_outputs);
            for (output, 0..) |*out, j| {
                out.* = self.weights[self.weights.len - 1][j]; // bias term
                for (activations, 0..) |act, i| {
                    out.* += self.weights[i][j] * act;
                }
            }

            return output;
        }

        /// Predict for multiple samples
        /// Time: O(n_samples × n_centers × d)
        pub fn predictBatch(self: *const Self, X: []const []const T) ![][]T {
            if (!self.fitted) return error.NotFitted;

            var predictions = try self.allocator.alloc([]T, X.len);
            errdefer {
                for (predictions[0..X.len]) |pred| self.allocator.free(pred);
                self.allocator.free(predictions);
            }

            for (X, 0..) |x, i| {
                predictions[i] = try self.predict(x);
            }

            return predictions;
        }

        /// Select centers using specified method
        fn selectCenters(self: *Self, X: []const []const T) !void {
            switch (self.config.center_method) {
                .random => try self.selectRandomCenters(X),
                .kmeans => try self.selectKMeansCenters(X),
            }
        }

        /// Select centers randomly from training data
        fn selectRandomCenters(self: *Self, X: []const []const T) !void {
            var rng = std.Random.DefaultPrng.init(42);
            const random = rng.random();

            self.centers = try self.allocator.alloc([]T, self.config.n_centers);
            errdefer {
                for (self.centers[0..self.config.n_centers]) |center| self.allocator.free(center);
                self.allocator.free(self.centers);
            }

            // Shuffle indices
            const indices = try self.allocator.alloc(usize, X.len);
            defer self.allocator.free(indices);
            for (indices, 0..) |*idx, i| idx.* = i;
            random.shuffle(usize, indices);

            for (self.centers, 0..) |*center, i| {
                center.* = try self.allocator.alloc(T, self.n_features);
                @memcpy(center.*, X[indices[i]]);
            }
        }

        /// Select centers using K-Means clustering
        fn selectKMeansCenters(self: *Self, X: []const []const T) !void {
            // Simple K-Means implementation
            var rng = std.Random.DefaultPrng.init(42);
            const random = rng.random();

            // Initialize centers randomly
            self.centers = try self.allocator.alloc([]T, self.config.n_centers);
            errdefer {
                for (self.centers) |center| self.allocator.free(center);
                self.allocator.free(self.centers);
            }

            for (self.centers) |*center| {
                center.* = try self.allocator.alloc(T, self.n_features);
            }

            // Random initialization
            const indices = try self.allocator.alloc(usize, X.len);
            defer self.allocator.free(indices);
            for (indices, 0..) |*idx, i| idx.* = i;
            random.shuffle(usize, indices);

            for (self.centers, 0..) |center, i| {
                @memcpy(center, X[indices[i]]);
            }

            // K-Means iterations
            var assignments = try self.allocator.alloc(usize, X.len);
            defer self.allocator.free(assignments);

            var iter: usize = 0;
            while (iter < self.config.max_kmeans_iter) : (iter += 1) {
                // Assign points to nearest centers
                for (X, 0..) |x, i| {
                    var min_dist: T = std.math.floatMax(T);
                    var best_center: usize = 0;
                    for (self.centers, 0..) |center, c| {
                        const dist = euclideanDistance(T, x, center);
                        if (dist < min_dist) {
                            min_dist = dist;
                            best_center = c;
                        }
                    }
                    assignments[i] = best_center;
                }

                // Update centers
                var new_centers = try self.allocator.alloc([]T, self.config.n_centers);
                defer {
                    for (new_centers) |nc| self.allocator.free(nc);
                    self.allocator.free(new_centers);
                }

                for (new_centers) |*nc| {
                    nc.* = try self.allocator.alloc(T, self.n_features);
                    @memset(nc.*, 0);
                }

                var counts = try self.allocator.alloc(usize, self.config.n_centers);
                defer self.allocator.free(counts);
                @memset(counts, 0);

                for (X, 0..) |x, i| {
                    const c = assignments[i];
                    for (x, 0..) |val, j| {
                        new_centers[c][j] += val;
                    }
                    counts[c] += 1;
                }

                // Average
                var max_change: T = 0;
                for (self.centers, 0..) |center, c| {
                    if (counts[c] > 0) {
                        for (center, 0..) |*val, j| {
                            const new_val = new_centers[c][j] / @as(T, @floatFromInt(counts[c]));
                            const change = @abs(new_val - val.*);
                            if (change > max_change) max_change = change;
                            val.* = new_val;
                        }
                    }
                }

                if (max_change < self.config.kmeans_tol) break;
            }
        }

        /// Compute widths for RBF centers
        fn computeWidths(self: *Self, X: []const []const T) !void {
            self.widths = try self.allocator.alloc(T, self.config.n_centers);

            switch (self.config.width_method) {
                .fixed => {
                    const width = self.config.fixed_width orelse return error.MissingFixedWidth;
                    @memset(self.widths, width);
                },
                .nearest_neighbor => {
                    // For each center, compute average distance to k nearest points
                    const k: usize = @min(5, X.len); // Use 5 nearest neighbors or less

                    for (self.centers, 0..) |center, i| {
                        var distances = try self.allocator.alloc(T, X.len);
                        defer self.allocator.free(distances);

                        for (X, 0..) |x, j| {
                            distances[j] = euclideanDistance(T, center, x);
                        }

                        std.mem.sort(T, distances, {}, std.sort.asc(T));

                        var sum: T = 0;
                        for (distances[0..k]) |d| sum += d;
                        self.widths[i] = sum / @as(T, @floatFromInt(k));

                        // Ensure non-zero width
                        if (self.widths[i] < 1e-10) self.widths[i] = 1e-10;
                    }
                },
            }
        }

        /// Compute RBF activations for a sample
        /// Returns: [n_centers + 1] (last element is bias = 1)
        fn computeActivations(self: *const Self, x: []const T) ![]T {
            var activations = try self.allocator.alloc(T, self.config.n_centers + 1);
            errdefer self.allocator.free(activations);

            for (self.centers, 0..) |center, i| {
                const dist = euclideanDistance(T, x, center);
                const width_sq = self.widths[i] * self.widths[i];
                activations[i] = @exp(-dist * dist / (2.0 * width_sq));
            }

            activations[self.config.n_centers] = 1.0; // bias term

            return activations;
        }

        /// Solve for weights using Ridge regression
        /// W = (Φ^T Φ + λI)^(-1) Φ^T Y
        fn solveWeights(self: *Self, phi: []const []const T, y: []const []const T) !void {
            _ = phi.len; // n_samples not needed
            const n_basis = self.config.n_centers + 1; // includes bias

            // Compute Φ^T Φ
            var PhiTPhi = try self.allocator.alloc([]T, n_basis);
            defer {
                for (PhiTPhi) |row| self.allocator.free(row);
                self.allocator.free(PhiTPhi);
            }

            for (PhiTPhi) |*row| {
                row.* = try self.allocator.alloc(T, n_basis);
                @memset(row.*, 0);
            }

            for (0..n_basis) |i| {
                for (0..n_basis) |j| {
                    var sum: T = 0;
                    for (phi) |phi_row| {
                        sum += phi_row[i] * phi_row[j];
                    }
                    PhiTPhi[i][j] = sum;
                    // Add regularization to diagonal
                    if (i == j) {
                        PhiTPhi[i][j] += self.config.lambda;
                    }
                }
            }

            // Compute Φ^T Y
            var PhiTY = try self.allocator.alloc([]T, n_basis);
            defer {
                for (PhiTY) |row| self.allocator.free(row);
                self.allocator.free(PhiTY);
            }

            for (PhiTY) |*row| {
                row.* = try self.allocator.alloc(T, self.n_outputs);
                @memset(row.*, 0);
            }

            for (0..n_basis) |i| {
                for (0..self.n_outputs) |j| {
                    var sum: T = 0;
                    for (phi, 0..) |phi_row, k| {
                        sum += phi_row[i] * y[k][j];
                    }
                    PhiTY[i][j] = sum;
                }
            }

            // Solve (Φ^T Φ) W = Φ^T Y using Gaussian elimination
            self.weights = try self.allocator.alloc([]T, n_basis);
            for (self.weights) |*weight_row| {
                weight_row.* = try self.allocator.alloc(T, self.n_outputs);
            }

            // For each output dimension, solve the system
            for (0..self.n_outputs) |out_idx| {
                // Extract b vector for this output
                const b = try self.allocator.alloc(T, n_basis);
                defer self.allocator.free(b);
                for (0..n_basis) |i| {
                    b[i] = PhiTY[i][out_idx];
                }

                // Solve Ax = b
                const x = try gaussianElimination(T, self.allocator, PhiTPhi, b);
                defer self.allocator.free(x);

                // Copy solution
                for (0..n_basis) |i| {
                    self.weights[i][out_idx] = x[i];
                }
            }
        }
    };
}

/// Euclidean distance between two vectors
fn euclideanDistance(comptime T: type, a: []const T, b: []const T) T {
    var sum: T = 0;
    for (a, 0..) |val, i| {
        const diff = val - b[i];
        sum += diff * diff;
    }
    return @sqrt(sum);
}

/// Solve Ax = b using Gaussian elimination with partial pivoting
fn gaussianElimination(comptime T: type, allocator: Allocator, A: []const []const T, b: []const T) ![]T {
    const n = A.len;

    // Create augmented matrix [A | b]
    var aug = try allocator.alloc([]T, n);
    defer {
        for (aug) |row| allocator.free(row);
        allocator.free(aug);
    }

    for (aug, 0..) |*row, i| {
        row.* = try allocator.alloc(T, n + 1);
        @memcpy(row.*[0..n], A[i]);
        row.*[n] = b[i];
    }

    // Forward elimination with partial pivoting
    for (0..n) |k| {
        // Find pivot
        var max_idx = k;
        var max_val = @abs(aug[k][k]);
        for (k + 1..n) |i| {
            const val = @abs(aug[i][k]);
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }

        // Swap rows
        if (max_idx != k) {
            const temp = aug[k];
            aug[k] = aug[max_idx];
            aug[max_idx] = temp;
        }

        // Eliminate
        for (k + 1..n) |i| {
            const factor = aug[i][k] / aug[k][k];
            for (k..n + 1) |j| {
                aug[i][j] -= factor * aug[k][j];
            }
        }
    }

    // Back substitution
    var x = try allocator.alloc(T, n);
    var i: usize = n;
    while (i > 0) {
        i -= 1;
        var sum: T = aug[i][n];
        for (i + 1..n) |j| {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    return x;
}

// ============================================================================
// Tests
// ============================================================================

test "RBF Network - basic regression" {
    const allocator = std.testing.allocator;

    // Simple 1D regression: y = x^2
    var X = [_][]const f64{
        &[_]f64{-2.0}, &[_]f64{-1.0}, &[_]f64{0.0},
        &[_]f64{1.0},  &[_]f64{2.0},
    };
    var y = [_][]const f64{
        &[_]f64{4.0}, &[_]f64{1.0}, &[_]f64{0.0},
        &[_]f64{1.0}, &[_]f64{4.0},
    };

    var rbf = RBFNetwork(f64).init(allocator, .{ .n_centers = 3 });
    defer rbf.deinit();

    try rbf.fit(&X, &y);
    try std.testing.expect(rbf.fitted);

    // Test prediction at training points
    const pred = try rbf.predict(&[_]f64{1.0});
    defer allocator.free(pred);

    // Should approximate y=1 for x=1
    try std.testing.expect(@abs(pred[0] - 1.0) < 1.0);
}

test "RBF Network - 2D input" {
    const allocator = std.testing.allocator;

    // y = x1 + 2*x2
    var X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 }, &[_]f64{ 1.0, 0.0 },
        &[_]f64{ 0.0, 1.0 }, &[_]f64{ 1.0, 1.0 },
    };
    var y = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{1.0},
        &[_]f64{2.0}, &[_]f64{3.0},
    };

    var rbf = RBFNetwork(f64).init(allocator, .{ .n_centers = 2 });
    defer rbf.deinit();

    try rbf.fit(&X, &y);

    const pred = try rbf.predict(&[_]f64{ 0.5, 0.5 });
    defer allocator.free(pred);

    // Should approximate y = 0.5 + 1.0 = 1.5
    try std.testing.expect(@abs(pred[0] - 1.5) < 1.0);
}

test "RBF Network - multiple outputs" {
    const allocator = std.testing.allocator;

    // y1 = x, y2 = 2*x
    var X = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{1.0}, &[_]f64{2.0},
    };
    var y = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 2.0, 4.0 },
    };

    var rbf = RBFNetwork(f64).init(allocator, .{ .n_centers = 2 });
    defer rbf.deinit();

    try rbf.fit(&X, &y);

    const pred = try rbf.predict(&[_]f64{1.5});
    defer allocator.free(pred);

    try std.testing.expect(pred.len == 2);
}

test "RBF Network - batch prediction" {
    const allocator = std.testing.allocator;

    var X = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{1.0}, &[_]f64{2.0},
    };
    var y = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{1.0}, &[_]f64{4.0},
    };

    var rbf = RBFNetwork(f64).init(allocator, .{ .n_centers = 2 });
    defer rbf.deinit();

    try rbf.fit(&X, &y);

    var X_test = [_][]const f64{
        &[_]f64{0.5}, &[_]f64{1.5},
    };

    const preds = try rbf.predictBatch(&X_test);
    defer {
        for (preds) |pred| allocator.free(pred);
        allocator.free(preds);
    }

    try std.testing.expect(preds.len == 2);
}

test "RBF Network - random centers" {
    const allocator = std.testing.allocator;

    var X = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{1.0}, &[_]f64{2.0}, &[_]f64{3.0},
    };
    var y = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{1.0}, &[_]f64{4.0}, &[_]f64{9.0},
    };

    var rbf = RBFNetwork(f64).init(allocator, .{
        .n_centers = 2,
        .center_method = .random,
    });
    defer rbf.deinit();

    try rbf.fit(&X, &y);
    try std.testing.expect(rbf.fitted);
}

test "RBF Network - fixed width" {
    const allocator = std.testing.allocator;

    var X = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{1.0}, &[_]f64{2.0},
    };
    var y = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{1.0}, &[_]f64{4.0},
    };

    var rbf = RBFNetwork(f64).init(allocator, .{
        .n_centers = 2,
        .width_method = .fixed,
        .fixed_width = 1.0,
    });
    defer rbf.deinit();

    try rbf.fit(&X, &y);
    try std.testing.expect(rbf.fitted);
}

test "RBF Network - regularization" {
    const allocator = std.testing.allocator;

    var X = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{1.0}, &[_]f64{2.0},
    };
    var y = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{1.0}, &[_]f64{4.0},
    };

    // Test with different lambda values
    var rbf1 = RBFNetwork(f64).init(allocator, .{
        .n_centers = 2,
        .lambda = 1e-6,
    });
    defer rbf1.deinit();
    try rbf1.fit(&X, &y);

    var rbf2 = RBFNetwork(f64).init(allocator, .{
        .n_centers = 2,
        .lambda = 1.0,
    });
    defer rbf2.deinit();
    try rbf2.fit(&X, &y);

    try std.testing.expect(rbf1.fitted);
    try std.testing.expect(rbf2.fitted);
}

test "RBF Network - f32 support" {
    const allocator = std.testing.allocator;

    var X = [_][]const f32{
        &[_]f32{0.0}, &[_]f32{1.0}, &[_]f32{2.0},
    };
    var y = [_][]const f32{
        &[_]f32{0.0}, &[_]f32{1.0}, &[_]f32{4.0},
    };

    var rbf = RBFNetwork(f32).init(allocator, .{ .n_centers = 2 });
    defer rbf.deinit();

    try rbf.fit(&X, &y);
    try std.testing.expect(rbf.fitted);
}

test "RBF Network - large dataset" {
    const allocator = std.testing.allocator;

    // Generate larger dataset
    var X_data: [50][]const f64 = undefined;
    var y_data: [50][]const f64 = undefined;
    var x_vals: [50]f64 = undefined;
    var y_vals: [50]f64 = undefined;

    for (0..50) |i| {
        const x = @as(f64, @floatFromInt(i)) / 10.0;
        x_vals[i] = x;
        y_vals[i] = x * x; // y = x^2
        X_data[i] = x_vals[i .. i + 1];
        y_data[i] = y_vals[i .. i + 1];
    }

    var rbf = RBFNetwork(f64).init(allocator, .{ .n_centers = 10 });
    defer rbf.deinit();

    try rbf.fit(&X_data, &y_data);
    try std.testing.expect(rbf.fitted);
}

test "RBF Network - empty data" {
    const allocator = std.testing.allocator;

    var X = [_][]const f64{};
    var y = [_][]const f64{};

    var rbf = RBFNetwork(f64).init(allocator, .{ .n_centers = 2 });
    defer rbf.deinit();

    try std.testing.expectError(error.EmptyData, rbf.fit(&X, &y));
}

test "RBF Network - dimension mismatch" {
    const allocator = std.testing.allocator;

    var X = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{1.0},
    };
    var y = [_][]const f64{
        &[_]f64{0.0},
    };

    var rbf = RBFNetwork(f64).init(allocator, .{ .n_centers = 2 });
    defer rbf.deinit();

    try std.testing.expectError(error.DimensionMismatch, rbf.fit(&X, &y));
}

test "RBF Network - predict before fit" {
    const allocator = std.testing.allocator;

    const rbf = RBFNetwork(f64).init(allocator, .{ .n_centers = 2 });

    try std.testing.expectError(error.NotFitted, rbf.predict(&[_]f64{0.0}));
}

test "RBF Network - too many centers" {
    const allocator = std.testing.allocator;

    var X = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{1.0},
    };
    var y = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{1.0},
    };

    var rbf = RBFNetwork(f64).init(allocator, .{ .n_centers = 5 });
    defer rbf.deinit();

    try std.testing.expectError(error.TooManyCenters, rbf.fit(&X, &y));
}

test "RBF Network - memory safety" {
    const allocator = std.testing.allocator;

    var X = [_][]const f64{
        &[_]f64{ 0.0, 0.0 }, &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 2.0, 2.0 }, &[_]f64{ 3.0, 3.0 },
    };
    var y = [_][]const f64{
        &[_]f64{0.0}, &[_]f64{2.0},
        &[_]f64{4.0}, &[_]f64{6.0},
    };

    var rbf = RBFNetwork(f64).init(allocator, .{ .n_centers = 3 });
    defer rbf.deinit();

    try rbf.fit(&X, &y);

    const pred = try rbf.predict(&[_]f64{ 1.5, 1.5 });
    defer allocator.free(pred);

    try std.testing.expect(pred.len == 1);
}
