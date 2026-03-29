const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Principal Component Analysis (PCA)
///
/// Reduces dimensionality by projecting data onto principal components
/// (directions of maximum variance). Based on eigenvalue decomposition
/// of the covariance matrix via Singular Value Decomposition (SVD).
///
/// Algorithm:
/// 1. Center data (subtract mean from each feature)
/// 2. Compute covariance matrix or apply SVD directly
/// 3. Find eigenvectors (principal components) sorted by eigenvalues
/// 4. Project data onto top k components
///
/// Time: O(n × m²) for SVD where n=samples, m=features
/// Space: O(n × m + m²) for data copy and components
///
/// Use cases:
/// - Dimensionality reduction (visualization, compression)
/// - Feature extraction (preprocessing for ML)
/// - Noise reduction (keeping only top components)
/// - Data compression (store fewer components)
/// - Visualization (reduce to 2D/3D for plotting)
///
/// Limitations:
/// - Linear transformation only (no nonlinear patterns)
/// - Sensitive to scaling (features should be standardized)
/// - Assumes directions of max variance are most important
/// - Interpretability: components are linear combinations
///
/// Example:
/// ```zig
/// const data = [_][]const f64{
///     &.{2.5, 2.4}, &.{0.5, 0.7}, &.{2.2, 2.9},
///     &.{1.9, 2.2}, &.{3.1, 3.0}, &.{2.3, 2.7}
/// };
/// var pca_model = PCA(f64).init(allocator);
/// defer pca_model.deinit();
/// try pca_model.fit(&data, 1); // reduce to 1 dimension
/// const transformed = try pca_model.transform(&data);
/// defer allocator.free(transformed);
/// ```
pub fn PCA(comptime T: type) type {
    return struct {
        allocator: Allocator,
        mean: ?[]T = null, // feature means (length m)
        components: ?[][]T = null, // principal components (k × m)
        explained_variance: ?[]T = null, // variance explained by each component
        n_components: usize = 0, // number of components retained

        const Self = @This();

        /// Initialize PCA model
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(allocator: Allocator) Self {
            return .{
                .allocator = allocator,
            };
        }

        /// Free all allocated memory
        /// Time: O(k) where k = n_components
        /// Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.mean) |mean| {
                self.allocator.free(mean);
            }
            if (self.components) |components| {
                for (components) |component| {
                    self.allocator.free(component);
                }
                self.allocator.free(components);
            }
            if (self.explained_variance) |variance| {
                self.allocator.free(variance);
            }
        }

        /// Fit PCA model to training data
        ///
        /// Time: O(n × m² + m³) for SVD
        /// Space: O(n × m + m²)
        ///
        /// Params:
        ///   data: n samples, each []const T of length m
        ///   n_components: number of principal components to retain (≤ m)
        ///
        /// Errors:
        ///   - OutOfMemory: allocation failed
        ///   - EmptyData: data.len == 0
        ///   - InvalidComponents: n_components > m or == 0
        ///   - InconsistentDimensions: samples have different lengths
        pub fn fit(self: *Self, data: []const []const T, n_components: usize) !void {
            if (data.len == 0) return error.EmptyData;
            const m = data[0].len; // number of features
            if (n_components == 0 or n_components > m) return error.InvalidComponents;

            // Check consistent dimensions
            for (data) |sample| {
                if (sample.len != m) return error.InconsistentDimensions;
            }

            // Free previous model if exists
            self.deinit();

            const n = data.len;
            self.n_components = n_components;

            // 1. Compute mean for each feature
            self.mean = try self.allocator.alloc(T, m);
            errdefer self.allocator.free(self.mean.?);

            @memset(self.mean.?, 0);
            for (data) |sample| {
                for (sample, 0..) |value, j| {
                    self.mean.?[j] += value;
                }
            }
            for (self.mean.?) |*val| {
                val.* /= @as(T, @floatFromInt(n));
            }

            // 2. Center data (X_centered = X - mean)
            const centered = try self.allocator.alloc([]T, n);
            errdefer {
                for (centered) |row| self.allocator.free(row);
                self.allocator.free(centered);
            }

            for (centered, 0..) |*row, i| {
                row.* = try self.allocator.alloc(T, m);
                errdefer self.allocator.free(row.*);

                for (row.*, 0..) |*val, j| {
                    val.* = data[i][j] - self.mean.?[j];
                }
            }
            defer {
                for (centered) |row| self.allocator.free(row);
                self.allocator.free(centered);
            }

            // 3. Compute covariance matrix (m × m)
            // Cov = (1/(n-1)) * X^T * X
            const cov = try self.allocator.alloc([]T, m);
            errdefer {
                for (cov) |row| self.allocator.free(row);
                self.allocator.free(cov);
            }

            for (cov, 0..) |*row, i| {
                row.* = try self.allocator.alloc(T, m);
                errdefer self.allocator.free(row.*);

                @memset(row.*, 0);
                for (centered) |sample| {
                    for (row.*, 0..) |*val, j| {
                        val.* += sample[i] * sample[j];
                    }
                }

                const normalizer: T = @floatFromInt(n - 1);
                for (row.*) |*val| {
                    val.* /= normalizer;
                }
            }
            defer {
                for (cov) |row| self.allocator.free(row);
                self.allocator.free(cov);
            }

            // 4. Compute eigenvalues and eigenvectors using power iteration
            // (simplified approach - find top k eigenvectors iteratively)
            self.components = try self.allocator.alloc([]T, n_components);
            errdefer {
                for (self.components.?[0..]) |comp| self.allocator.free(comp);
                self.allocator.free(self.components.?);
            }

            self.explained_variance = try self.allocator.alloc(T, n_components);
            errdefer self.allocator.free(self.explained_variance.?);

            // Deflation method: find eigenvectors one by one
            const residual_cov = try self.allocator.alloc([]T, m);
            defer {
                for (residual_cov) |row| self.allocator.free(row);
                self.allocator.free(residual_cov);
            }

            // Copy covariance to residual
            for (residual_cov, 0..) |*row, i| {
                row.* = try self.allocator.alloc(T, m);
                @memcpy(row.*, cov[i]);
            }

            for (0..n_components) |k| {
                // Power iteration to find dominant eigenvector
                const eigenvec = try self.allocator.alloc(T, m);
                errdefer self.allocator.free(eigenvec);

                // Initialize with random vector
                var prng = std.rand.DefaultPrng.init(42 + k);
                const rand = prng.random();
                for (eigenvec) |*val| {
                    val.* = rand.float(T) * 2 - 1; // [-1, 1]
                }

                // Normalize
                var norm: T = 0;
                for (eigenvec) |val| norm += val * val;
                norm = @sqrt(norm);
                for (eigenvec) |*val| val.* /= norm;

                // Power iteration (up to 100 iterations)
                const max_iter = 100;
                var iter: usize = 0;
                while (iter < max_iter) : (iter += 1) {
                    // v_new = Cov * v_old
                    var new_vec = try self.allocator.alloc(T, m);
                    defer self.allocator.free(new_vec);

                    @memset(new_vec, 0);
                    for (residual_cov, 0..) |row, i| {
                        for (row, 0..) |cov_val, j| {
                            new_vec[i] += cov_val * eigenvec[j];
                        }
                    }

                    // Normalize
                    norm = 0;
                    for (new_vec) |val| norm += val * val;
                    norm = @sqrt(norm);
                    if (norm < 1e-10) break; // degenerate case

                    for (new_vec) |*val| val.* /= norm;

                    // Check convergence
                    var diff: T = 0;
                    for (eigenvec, new_vec) |old, new| {
                        const delta = old - new;
                        diff += delta * delta;
                    }
                    @memcpy(eigenvec, new_vec);

                    if (diff < 1e-8) break;
                }

                self.components.?[k] = eigenvec;

                // Compute eigenvalue (Rayleigh quotient: λ = v^T * Cov * v)
                var lambda: T = 0;
                for (residual_cov, 0..) |row, i| {
                    var sum: T = 0;
                    for (row, 0..) |cov_val, j| {
                        sum += cov_val * eigenvec[j];
                    }
                    lambda += eigenvec[i] * sum;
                }
                self.explained_variance.?[k] = lambda;

                // Deflate: Cov_new = Cov - λ * v * v^T
                for (residual_cov, 0..) |row, i| {
                    for (row, 0..) |*cov_val, j| {
                        cov_val.* -= lambda * eigenvec[i] * eigenvec[j];
                    }
                }
            }
        }

        /// Transform data to principal component space
        ///
        /// Time: O(n × m × k) where k = n_components
        /// Space: O(n × k)
        ///
        /// Params:
        ///   data: n samples to transform
        ///
        /// Returns: n samples projected onto k principal components
        ///
        /// Errors:
        ///   - ModelNotFitted: fit() not called yet
        ///   - OutOfMemory: allocation failed
        ///   - InconsistentDimensions: sample dimension mismatch
        pub fn transform(self: *const Self, data: []const []const T) ![][]T {
            if (self.mean == null or self.components == null) {
                return error.ModelNotFitted;
            }

            const n = data.len;
            const m = self.mean.?.len;
            const k = self.n_components;

            const result = try self.allocator.alloc([]T, n);
            errdefer {
                for (result) |row| self.allocator.free(row);
                self.allocator.free(result);
            }

            for (result, 0..) |*row, i| {
                if (data[i].len != m) return error.InconsistentDimensions;

                row.* = try self.allocator.alloc(T, k);
                errdefer self.allocator.free(row.*);

                // Project: transformed[i][j] = sum((data[i] - mean) * components[j])
                for (row.*, 0..) |*val, j| {
                    val.* = 0;
                    for (0..m) |l| {
                        val.* += (data[i][l] - self.mean.?[l]) * self.components.?[j][l];
                    }
                }
            }

            return result;
        }

        /// Transform single sample
        /// Time: O(m × k)
        /// Space: O(k)
        pub fn transformSample(self: *const Self, sample: []const T) ![]T {
            if (self.mean == null or self.components == null) {
                return error.ModelNotFitted;
            }

            const m = self.mean.?.len;
            const k = self.n_components;

            if (sample.len != m) return error.InconsistentDimensions;

            const result = try self.allocator.alloc(T, k);
            errdefer self.allocator.free(result);

            for (result, 0..) |*val, j| {
                val.* = 0;
                for (0..m) |l| {
                    val.* += (sample[l] - self.mean.?[l]) * self.components.?[j][l];
                }
            }

            return result;
        }

        /// Inverse transform (reconstruct original space from PC space)
        ///
        /// Time: O(n × k × m)
        /// Space: O(n × m)
        ///
        /// Note: Reconstruction loses information if k < m (dimensionality reduction)
        pub fn inverseTransform(self: *const Self, transformed: []const []const T) ![][]T {
            if (self.mean == null or self.components == null) {
                return error.ModelNotFitted;
            }

            const n = transformed.len;
            const m = self.mean.?.len;
            const k = self.n_components;

            const result = try self.allocator.alloc([]T, n);
            errdefer {
                for (result) |row| self.allocator.free(row);
                self.allocator.free(result);
            }

            for (result, 0..) |*row, i| {
                if (transformed[i].len != k) return error.InconsistentDimensions;

                row.* = try self.allocator.alloc(T, m);
                errdefer self.allocator.free(row.*);

                // Reconstruct: original[i][l] = mean[l] + sum(transformed[i][j] * components[j][l])
                for (row.*, 0..) |*val, l| {
                    val.* = self.mean.?[l];
                    for (0..k) |j| {
                        val.* += transformed[i][j] * self.components.?[j][l];
                    }
                }
            }

            return result;
        }

        /// Get total explained variance ratio (sum of explained variance)
        /// Time: O(k)
        /// Space: O(1)
        pub fn explainedVarianceRatio(self: *const Self) ![]T {
            if (self.explained_variance == null) return error.ModelNotFitted;

            var total: T = 0;
            for (self.explained_variance.?) |var_val| {
                total += var_val;
            }

            const ratios = try self.allocator.alloc(T, self.n_components);
            errdefer self.allocator.free(ratios);

            for (ratios, self.explained_variance.?) |*ratio, var_val| {
                ratio.* = if (total > 0) var_val / total else 0;
            }

            return ratios;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "PCA: basic 2D -> 1D reduction" {
    const allocator = testing.allocator;

    // Simple 2D data with clear variance direction
    const data = [_][]const f64{
        &.{ 2.5, 2.4 },
        &.{ 0.5, 0.7 },
        &.{ 2.2, 2.9 },
        &.{ 1.9, 2.2 },
        &.{ 3.1, 3.0 },
        &.{ 2.3, 2.7 },
        &.{ 2.0, 1.6 },
        &.{ 1.0, 1.1 },
        &.{ 1.5, 1.6 },
        &.{ 1.1, 0.9 },
    };

    var pca = PCA(f64).init(allocator);
    defer pca.deinit();

    try pca.fit(&data, 1);

    // Check that mean is computed
    try testing.expect(pca.mean != null);
    try testing.expectEqual(@as(usize, 2), pca.mean.?.len);

    // Check that component is unit vector
    try testing.expect(pca.components != null);
    try testing.expectEqual(@as(usize, 1), pca.components.?.len);

    var comp_norm: f64 = 0;
    for (pca.components.?[0]) |val| {
        comp_norm += val * val;
    }
    try testing.expectApproxEqRel(@as(f64, 1.0), comp_norm, 1e-6);

    // Transform data
    const transformed = try pca.transform(&data);
    defer {
        for (transformed) |row| allocator.free(row);
        allocator.free(transformed);
    }

    try testing.expectEqual(@as(usize, data.len), transformed.len);
    try testing.expectEqual(@as(usize, 1), transformed[0].len);
}

test "PCA: preserve all components (no reduction)" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 1.0, 2.0, 3.0 },
        &.{ 4.0, 5.0, 6.0 },
        &.{ 7.0, 8.0, 9.0 },
    };

    var pca = PCA(f64).init(allocator);
    defer pca.deinit();

    try pca.fit(&data, 3); // keep all 3 components

    try testing.expectEqual(@as(usize, 3), pca.n_components);

    const transformed = try pca.transform(&data);
    defer {
        for (transformed) |row| allocator.free(row);
        allocator.free(transformed);
    }

    // Inverse transform should approximately recover original
    const reconstructed = try pca.inverseTransform(transformed);
    defer {
        for (reconstructed) |row| allocator.free(row);
        allocator.free(reconstructed);
    }

    for (data, reconstructed) |orig, recon| {
        for (orig, recon) |o, r| {
            try testing.expectApproxEqRel(o, r, 1e-3);
        }
    }
}

test "PCA: explained variance" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 1.0, 1.0 },
        &.{ 2.0, 2.0 },
        &.{ 3.0, 3.0 },
        &.{ 4.0, 4.0 },
        &.{ 5.0, 5.0 },
    };

    var pca = PCA(f64).init(allocator);
    defer pca.deinit();

    try pca.fit(&data, 2);

    const ratios = try pca.explainedVarianceRatio();
    defer allocator.free(ratios);

    try testing.expectEqual(@as(usize, 2), ratios.len);

    // First component should explain most variance (data is on diagonal)
    try testing.expect(ratios[0] > 0.95);

    // Total should sum to ~1.0
    var sum: f64 = 0;
    for (ratios) |r| sum += r;
    try testing.expectApproxEqRel(@as(f64, 1.0), sum, 1e-6);
}

test "PCA: single component extraction" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 1.0, 2.0 },
        &.{ 2.0, 4.0 },
        &.{ 3.0, 6.0 },
        &.{ 4.0, 8.0 },
    };

    var pca = PCA(f64).init(allocator);
    defer pca.deinit();

    try pca.fit(&data, 1);

    const transformed = try pca.transform(&data);
    defer {
        for (transformed) |row| allocator.free(row);
        allocator.free(transformed);
    }

    // All samples should have 1D representation
    for (transformed) |sample| {
        try testing.expectEqual(@as(usize, 1), sample.len);
    }
}

test "PCA: transform single sample" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 1.0, 2.0, 3.0 },
        &.{ 4.0, 5.0, 6.0 },
        &.{ 7.0, 8.0, 9.0 },
    };

    var pca = PCA(f64).init(allocator);
    defer pca.deinit();

    try pca.fit(&data, 2);

    const sample = [_]f64{ 2.0, 3.0, 4.0 };
    const transformed = try pca.transformSample(&sample);
    defer allocator.free(transformed);

    try testing.expectEqual(@as(usize, 2), transformed.len);
}

test "PCA: error - empty data" {
    const allocator = testing.allocator;
    const data = [_][]const f64{};

    var pca = PCA(f64).init(allocator);
    defer pca.deinit();

    try testing.expectError(error.EmptyData, pca.fit(&data, 1));
}

test "PCA: error - invalid n_components" {
    const allocator = testing.allocator;
    const data = [_][]const f64{
        &.{ 1.0, 2.0 },
        &.{ 3.0, 4.0 },
    };

    var pca = PCA(f64).init(allocator);
    defer pca.deinit();

    // n_components = 0
    try testing.expectError(error.InvalidComponents, pca.fit(&data, 0));

    // n_components > m
    try testing.expectError(error.InvalidComponents, pca.fit(&data, 3));
}

test "PCA: error - inconsistent dimensions" {
    const allocator = testing.allocator;
    const data = [_][]const f64{
        &.{ 1.0, 2.0 },
        &.{ 3.0, 4.0, 5.0 }, // wrong dimension
    };

    var pca = PCA(f64).init(allocator);
    defer pca.deinit();

    try testing.expectError(error.InconsistentDimensions, pca.fit(&data, 1));
}

test "PCA: error - model not fitted" {
    const allocator = testing.allocator;
    const data = [_][]const f64{
        &.{ 1.0, 2.0 },
    };

    var pca = PCA(f64).init(allocator);
    defer pca.deinit();

    try testing.expectError(error.ModelNotFitted, pca.transform(&data));
    try testing.expectError(error.ModelNotFitted, pca.transformSample(&.{ 1.0, 2.0 }));
    try testing.expectError(error.ModelNotFitted, pca.explainedVarianceRatio());
}

test "PCA: f32 support" {
    const allocator = testing.allocator;

    const data = [_][]const f32{
        &.{ 1.0, 2.0 },
        &.{ 2.0, 4.0 },
        &.{ 3.0, 6.0 },
    };

    var pca = PCA(f32).init(allocator);
    defer pca.deinit();

    try pca.fit(&data, 1);

    const transformed = try pca.transform(&data);
    defer {
        for (transformed) |row| allocator.free(row);
        allocator.free(transformed);
    }

    try testing.expectEqual(@as(usize, data.len), transformed.len);
}

test "PCA: inverse transform with loss" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 1.0, 2.0, 3.0 },
        &.{ 4.0, 5.0, 6.0 },
        &.{ 7.0, 8.0, 9.0 },
    };

    var pca = PCA(f64).init(allocator);
    defer pca.deinit();

    // Reduce to 2 components (lose 1 dimension)
    try pca.fit(&data, 2);

    const transformed = try pca.transform(&data);
    defer {
        for (transformed) |row| allocator.free(row);
        allocator.free(transformed);
    }

    const reconstructed = try pca.inverseTransform(transformed);
    defer {
        for (reconstructed) |row| allocator.free(row);
        allocator.free(reconstructed);
    }

    // Reconstruction should have some error (not exact)
    for (data, reconstructed) |orig, recon| {
        try testing.expectEqual(orig.len, recon.len);
        // Just verify dimensions match, don't check exact values
    }
}

test "PCA: large dataset" {
    const allocator = testing.allocator;

    const n = 100;
    const m = 5;
    const data = try allocator.alloc([]f64, n);
    defer {
        for (data) |row| allocator.free(row);
        allocator.free(data);
    }

    var prng = std.rand.DefaultPrng.init(12345);
    const rand = prng.random();

    for (data) |*row| {
        row.* = try allocator.alloc(f64, m);
        for (row.*) |*val| {
            val.* = rand.float(f64) * 10;
        }
    }

    var pca = PCA(f64).init(allocator);
    defer pca.deinit();

    try pca.fit(data, 2);

    const transformed = try pca.transform(data);
    defer {
        for (transformed) |row| allocator.free(row);
        allocator.free(transformed);
    }

    try testing.expectEqual(@as(usize, n), transformed.len);
    try testing.expectEqual(@as(usize, 2), transformed[0].len);
}

test "PCA: centering is applied" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 10.0, 20.0 },
        &.{ 11.0, 21.0 },
        &.{ 12.0, 22.0 },
    };

    var pca = PCA(f64).init(allocator);
    defer pca.deinit();

    try pca.fit(&data, 2);

    // Mean should be approximately [11, 21]
    try testing.expectApproxEqRel(@as(f64, 11.0), pca.mean.?[0], 1e-6);
    try testing.expectApproxEqRel(@as(f64, 21.0), pca.mean.?[1], 1e-6);
}

test "PCA: memory safety with testing.allocator" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 1.0, 2.0, 3.0 },
        &.{ 4.0, 5.0, 6.0 },
    };

    var pca = PCA(f64).init(allocator);
    defer pca.deinit();

    try pca.fit(&data, 2);

    const transformed = try pca.transform(&data);
    defer {
        for (transformed) |row| allocator.free(row);
        allocator.free(transformed);
    }

    const reconstructed = try pca.inverseTransform(transformed);
    defer {
        for (reconstructed) |row| allocator.free(row);
        allocator.free(reconstructed);
    }

    const ratios = try pca.explainedVarianceRatio();
    defer allocator.free(ratios);
}
