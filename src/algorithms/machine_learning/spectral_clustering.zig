const std = @import("std");
const Allocator = std.mem.Allocator;

/// Spectral Clustering
///
/// Graph-based clustering using eigendecomposition of the Laplacian matrix.
/// Effective for non-convex clusters by embedding data in spectral space.
///
/// Algorithm:
/// 1. Construct similarity graph (k-NN or ε-neighborhood)
/// 2. Compute graph Laplacian: L = D - W (unnormalized) or variants
/// 3. Find first K eigenvectors of Laplacian
/// 4. Apply K-Means clustering on eigenvector embedding
///
/// Time: O(n² × k_nn + n³) where n² for similarity, n³ for eigendecomposition
/// Space: O(n²) for affinity matrix
///
/// Use cases:
/// - Image segmentation (pixels as nodes)
/// - Community detection in networks
/// - Non-convex cluster shapes
/// - Manifold clustering (data lies on low-dim manifold)
///
/// Trade-offs:
/// - vs K-Means: Handles non-convex clusters, but O(n³) vs O(n)
/// - vs DBSCAN: More stable, but requires K parameter
/// - vs GMM: Better for graph-structured data, but expensive
pub fn SpectralClustering(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        n_clusters: usize,
        k_neighbors: usize,
        sigma: T, // RBF kernel bandwidth
        laplacian_type: LaplacianType,
        max_iterations: usize,
        tolerance: T,
        random_seed: u64,

        labels: ?[]usize,
        cluster_centers: ?[][]T, // Centroids in eigenvector space

        pub const LaplacianType = enum {
            unnormalized, // L = D - W
            symmetric, // L_sym = D^(-1/2) L D^(-1/2) = I - D^(-1/2) W D^(-1/2)
            random_walk, // L_rw = D^(-1) L = I - D^(-1) W
        };

        /// Initialize Spectral Clustering
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(
            allocator: Allocator,
            n_clusters: usize,
            k_neighbors: usize,
            sigma: T,
            laplacian_type: LaplacianType,
            max_iterations: usize,
            tolerance: T,
            random_seed: u64,
        ) Self {
            return .{
                .allocator = allocator,
                .n_clusters = n_clusters,
                .k_neighbors = k_neighbors,
                .sigma = sigma,
                .laplacian_type = laplacian_type,
                .max_iterations = max_iterations,
                .tolerance = tolerance,
                .random_seed = random_seed,
                .labels = null,
                .cluster_centers = null,
            };
        }

        /// Free allocated memory
        ///
        /// Time: O(k) | Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.labels) |labels| {
                self.allocator.free(labels);
                self.labels = null;
            }
            if (self.cluster_centers) |centers| {
                for (centers) |center| {
                    self.allocator.free(center);
                }
                self.allocator.free(centers);
                self.cluster_centers = null;
            }
        }

        /// Fit the model to data
        ///
        /// Time: O(n² × k + n³) | Space: O(n²)
        pub fn fit(self: *Self, data: []const []const T) !void {
            if (data.len == 0) return error.EmptyData;
            if (self.n_clusters == 0 or self.n_clusters > data.len) return error.InvalidParameters;

            const n = data.len;

            // Step 1: Compute affinity matrix (k-NN graph with RBF kernel)
            const affinity = try self.allocator.alloc([]T, n);
            defer {
                for (affinity) |row| self.allocator.free(row);
                self.allocator.free(affinity);
            }
            for (affinity) |*row| {
                row.* = try self.allocator.alloc(T, n);
                @memset(row.*, 0);
            }

            try self.computeAffinity(data, affinity);

            // Step 2: Compute degree matrix and Laplacian
            const degree = try self.allocator.alloc(T, n);
            defer self.allocator.free(degree);

            for (degree, 0..) |*deg, i| {
                deg.* = 0;
                for (affinity[i]) |w| {
                    deg.* += w;
                }
            }

            const laplacian = try self.allocator.alloc([]T, n);
            defer {
                for (laplacian) |row| self.allocator.free(row);
                self.allocator.free(laplacian);
            }
            for (laplacian) |*row| {
                row.* = try self.allocator.alloc(T, n);
            }

            try self.computeLaplacian(affinity, degree, laplacian);

            // Step 3: Compute first K eigenvectors
            const eigenvectors = try self.allocator.alloc([]T, n);
            defer {
                for (eigenvectors) |vec| self.allocator.free(vec);
                self.allocator.free(eigenvectors);
            }
            for (eigenvectors) |*vec| {
                vec.* = try self.allocator.alloc(T, self.n_clusters);
            }

            try self.computeEigenvectors(laplacian, eigenvectors);

            // Step 4: Apply K-Means on eigenvector embedding
            const embedding = try self.allocator.alloc([]T, n);
            defer {
                for (embedding) |row| self.allocator.free(row);
                self.allocator.free(embedding);
            }
            for (embedding, 0..) |*row, i| {
                row.* = try self.allocator.alloc(T, self.n_clusters);
                for (row.*, 0..) |*val, j| {
                    val.* = eigenvectors[i][j];
                }
            }

            try self.kmeansClustering(embedding);
        }

        /// Compute affinity matrix using k-NN graph with RBF kernel
        fn computeAffinity(self: *Self, data: []const []const T, affinity: [][]T) !void {
            const n = data.len;

            // For each point, find k nearest neighbors
            for (0..n) |i| {
                // Compute distances to all other points
                const distances = try self.allocator.alloc(struct { idx: usize, dist: T }, n);
                defer self.allocator.free(distances);

                for (distances, 0..) |*entry, j| {
                    entry.idx = j;
                    entry.dist = if (i == j) std.math.inf(T) else self.euclideanDistance(data[i], data[j]);
                }

                // Sort by distance
                std.mem.sort(@TypeOf(distances[0]), distances, {}, struct {
                    fn lessThan(_: void, a: @TypeOf(distances[0]), b: @TypeOf(distances[0])) bool {
                        return a.dist < b.dist;
                    }
                }.lessThan);

                // Connect to k nearest neighbors with RBF kernel
                const k = @min(self.k_neighbors, n - 1);
                for (0..k) |idx| {
                    const j = distances[idx].idx;
                    const dist = distances[idx].dist;
                    const weight = @exp(-dist * dist / (2 * self.sigma * self.sigma));
                    affinity[i][j] = weight;
                    affinity[j][i] = weight; // Symmetric
                }
            }
        }

        /// Compute Laplacian matrix
        fn computeLaplacian(self: *Self, affinity: []const []const T, degree: []const T, laplacian: [][]T) !void {
            const n = affinity.len;

            switch (self.laplacian_type) {
                .unnormalized => {
                    // L = D - W
                    for (0..n) |i| {
                        for (0..n) |j| {
                            laplacian[i][j] = if (i == j) degree[i] - affinity[i][j] else -affinity[i][j];
                        }
                    }
                },
                .symmetric => {
                    // L_sym = I - D^(-1/2) W D^(-1/2)
                    for (0..n) |i| {
                        const d_i_sqrt = @sqrt(degree[i]);
                        for (0..n) |j| {
                            const d_j_sqrt = @sqrt(degree[j]);
                            if (i == j) {
                                laplacian[i][j] = 1.0 - affinity[i][j] / (d_i_sqrt * d_j_sqrt);
                            } else {
                                laplacian[i][j] = -affinity[i][j] / (d_i_sqrt * d_j_sqrt);
                            }
                        }
                    }
                },
                .random_walk => {
                    // L_rw = I - D^(-1) W
                    for (0..n) |i| {
                        for (0..n) |j| {
                            if (i == j) {
                                laplacian[i][j] = 1.0 - affinity[i][j] / degree[i];
                            } else {
                                laplacian[i][j] = -affinity[i][j] / degree[i];
                            }
                        }
                    }
                },
            }
        }

        /// Compute first K eigenvectors using power iteration
        /// (Simplified - real implementation would use LAPACK)
        fn computeEigenvectors(self: *Self, laplacian: []const []const T, eigenvectors: [][]T) !void {
            const n = laplacian.len;
            var prng = std.Random.DefaultPrng.init(self.random_seed);
            const random = prng.random();

            // For each eigenvector (simplified - assumes smallest eigenvalues)
            for (0..self.n_clusters) |k| {
                // Initialize random vector
                var v = try self.allocator.alloc(T, n);
                defer self.allocator.free(v);

                for (v) |*val| {
                    val.* = random.float(T) * 2.0 - 1.0;
                }

                // Normalize
                var norm: T = 0;
                for (v) |val| norm += val * val;
                norm = @sqrt(norm);
                for (v) |*val| val.* /= norm;

                // Power iteration (simplified - finds dominant eigenvector)
                // Real implementation: use inverse iteration for smallest eigenvalues
                for (0..20) |_| {
                    var new_v = try self.allocator.alloc(T, n);
                    defer self.allocator.free(new_v);
                    @memset(new_v, 0);

                    // new_v = L * v
                    for (0..n) |i| {
                        for (0..n) |j| {
                            new_v[i] += laplacian[i][j] * v[j];
                        }
                    }

                    // Normalize
                    norm = 0;
                    for (new_v) |val| norm += val * val;
                    norm = @sqrt(norm);
                    if (norm > 0) {
                        for (new_v, 0..) |val, i| v[i] = val / norm;
                    }
                }

                // Store eigenvector
                for (eigenvectors, 0..) |vec, i| {
                    vec[k] = v[i];
                }
            }
        }

        /// Apply K-Means clustering on embedding
        fn kmeansClustering(self: *Self, embedding: []const []const T) !void {
            const n = embedding.len;

            // Initialize labels
            if (self.labels) |labels| self.allocator.free(labels);
            self.labels = try self.allocator.alloc(usize, n);

            // K-Means++ initialization
            var prng = std.Random.DefaultPrng.init(self.random_seed);
            const random = prng.random();

            if (self.cluster_centers) |centers| {
                for (centers) |center| self.allocator.free(center);
                self.allocator.free(centers);
            }
            self.cluster_centers = try self.allocator.alloc([]T, self.n_clusters);
            for (self.cluster_centers.?) |*center| {
                center.* = try self.allocator.alloc(T, self.n_clusters);
            }

            // First center: random point
            const first_idx = random.intRangeLessThan(usize, 0, n);
            @memcpy(self.cluster_centers.?[0], embedding[first_idx]);

            // Remaining centers: K-Means++
            for (1..self.n_clusters) |k| {
                const distances = try self.allocator.alloc(T, n);
                defer self.allocator.free(distances);

                for (distances, 0..) |*dist, i| {
                    var min_dist: T = std.math.inf(T);
                    for (0..k) |j| {
                        const d = self.euclideanDistance(embedding[i], self.cluster_centers.?[j]);
                        min_dist = @min(min_dist, d);
                    }
                    dist.* = min_dist * min_dist;
                }

                const total: T = blk: {
                    var sum: T = 0;
                    for (distances) |d| sum += d;
                    break :blk sum;
                };

                const threshold = random.float(T) * total;
                var cumsum: T = 0;
                var selected_idx: usize = 0;
                for (distances, 0..) |d, i| {
                    cumsum += d;
                    if (cumsum >= threshold) {
                        selected_idx = i;
                        break;
                    }
                }
                @memcpy(self.cluster_centers.?[k], embedding[selected_idx]);
            }

            // K-Means iterations
            var iteration: usize = 0;
            while (iteration < self.max_iterations) : (iteration += 1) {
                var changed = false;

                // Assignment step
                for (self.labels.?, 0..) |*label, i| {
                    var min_dist: T = std.math.inf(T);
                    var best_cluster: usize = 0;

                    for (self.cluster_centers.?, 0..) |center, k| {
                        const dist = self.euclideanDistance(embedding[i], center);
                        if (dist < min_dist) {
                            min_dist = dist;
                            best_cluster = k;
                        }
                    }

                    if (label.* != best_cluster) {
                        changed = true;
                        label.* = best_cluster;
                    }
                }

                if (!changed) break;

                // Update step
                var counts = try self.allocator.alloc(usize, self.n_clusters);
                defer self.allocator.free(counts);
                @memset(counts, 0);

                for (self.cluster_centers.?) |center| {
                    @memset(center, 0);
                }

                for (self.labels.?, 0..) |label, i| {
                    counts[label] += 1;
                    for (self.cluster_centers.?[label], 0..) |*val, j| {
                        val.* += embedding[i][j];
                    }
                }

                for (self.cluster_centers.?, 0..) |center, k| {
                    if (counts[k] > 0) {
                        for (center) |*val| val.* /= @as(T, @floatFromInt(counts[k]));
                    }
                }
            }
        }

        /// Predict cluster labels for data
        ///
        /// Time: O(n × k) | Space: O(1)
        pub fn predict(self: *const Self, data: []const []const T) ![]usize {
            if (self.labels == null) return error.ModelNotFitted;

            const n = data.len;
            const labels = try self.allocator.alloc(usize, n);

            // Note: For proper prediction, we'd need to project new points
            // into the eigenspace. This is a simplified version.
            for (labels, 0..) |*label, i| {
                var min_dist: T = std.math.inf(T);
                var best_cluster: usize = 0;

                for (self.cluster_centers.?, 0..) |center, k| {
                    const dist = self.euclideanDistance(data[i], center);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_cluster = k;
                    }
                }

                label.* = best_cluster;
            }

            return labels;
        }

        /// Compute Euclidean distance between two points
        fn euclideanDistance(_: *const Self, a: []const T, b: []const T) T {
            var sum: T = 0;
            for (a, b) |a_i, b_i| {
                const diff = a_i - b_i;
                sum += diff * diff;
            }
            return @sqrt(sum);
        }
    };
}

// Tests
const testing = std.testing;

test "SpectralClustering: basic two clusters" {
    const allocator = testing.allocator;

    // Two well-separated clusters
    var data_storage: [6][2]f64 = .{
        .{ 0, 0 }, .{ 0.1, 0.1 }, .{ 0.2, 0 },
        .{ 5, 5 }, .{ 5.1, 5.1 }, .{ 5.2, 5 },
    };
    var data: [6][]const f64 = undefined;
    for (&data, &data_storage) |*ptr, *storage| {
        ptr.* = storage;
    }

    var model = SpectralClustering(f64).init(
        allocator,
        2, // n_clusters
        3, // k_neighbors
        1.0, // sigma
        .symmetric,
        100,
        1e-4,
        42,
    );
    defer model.deinit();

    try model.fit(&data);

    try testing.expect(model.labels != null);
    try testing.expectEqual(@as(usize, 6), model.labels.?.len);

    // Check clusters are valid
    for (model.labels.?) |label| {
        try testing.expect(label < 2);
    }
}

test "SpectralClustering: single cluster" {
    const allocator = testing.allocator;

    var data_storage: [4][2]f64 = .{
        .{ 0, 0 }, .{ 0.1, 0.1 }, .{ 0.2, 0 }, .{ 0.1, 0.2 },
    };
    var data: [4][]const f64 = undefined;
    for (&data, &data_storage) |*ptr, *storage| {
        ptr.* = storage;
    }

    var model = SpectralClustering(f64).init(allocator, 1, 2, 0.5, .unnormalized, 100, 1e-4, 42);
    defer model.deinit();

    try model.fit(&data);

    try testing.expect(model.labels != null);
    for (model.labels.?) |label| {
        try testing.expectEqual(@as(usize, 0), label);
    }
}

test "SpectralClustering: laplacian types" {
    const allocator = testing.allocator;

    var data_storage: [6][2]f64 = .{
        .{ 0, 0 }, .{ 0.1, 0 }, .{ 0.2, 0 },
        .{ 3, 3 }, .{ 3.1, 3 }, .{ 3.2, 3 },
    };
    var data: [6][]const f64 = undefined;
    for (&data, &data_storage) |*ptr, *storage| {
        ptr.* = storage;
    }

    const laplacian_types = [_]SpectralClustering(f64).LaplacianType{
        .unnormalized,
        .symmetric,
        .random_walk,
    };

    for (laplacian_types) |lap_type| {
        var model = SpectralClustering(f64).init(allocator, 2, 3, 1.0, lap_type, 100, 1e-4, 42);
        defer model.deinit();

        try model.fit(&data);
        try testing.expect(model.labels != null);
    }
}

test "SpectralClustering: f32 support" {
    const allocator = testing.allocator;

    var data_storage: [4][2]f32 = .{
        .{ 0, 0 }, .{ 0.1, 0.1 },
        .{ 5, 5 }, .{ 5.1, 5.1 },
    };
    var data: [4][]const f32 = undefined;
    for (&data, &data_storage) |*ptr, *storage| {
        ptr.* = storage;
    }

    var model = SpectralClustering(f32).init(allocator, 2, 2, 1.0, .symmetric, 50, 1e-4, 42);
    defer model.deinit();

    try model.fit(&data);
    try testing.expect(model.labels != null);
}

test "SpectralClustering: 3D data" {
    const allocator = testing.allocator;

    var data_storage: [6][3]f64 = .{
        .{ 0, 0, 0 }, .{ 0.1, 0, 0 }, .{ 0, 0.1, 0 },
        .{ 5, 5, 5 }, .{ 5.1, 5, 5 }, .{ 5, 5.1, 5 },
    };
    var data: [6][]const f64 = undefined;
    for (&data, &data_storage) |*ptr, *storage| {
        ptr.* = storage;
    }

    var model = SpectralClustering(f64).init(allocator, 2, 3, 1.0, .symmetric, 100, 1e-4, 42);
    defer model.deinit();

    try model.fit(&data);
    try testing.expect(model.labels != null);
}

test "SpectralClustering: predict" {
    const allocator = testing.allocator;

    var train_storage: [4][2]f64 = .{
        .{ 0, 0 }, .{ 0.1, 0.1 },
        .{ 5, 5 }, .{ 5.1, 5.1 },
    };
    var train_data: [4][]const f64 = undefined;
    for (&train_data, &train_storage) |*ptr, *storage| {
        ptr.* = storage;
    }

    var model = SpectralClustering(f64).init(allocator, 2, 2, 1.0, .symmetric, 100, 1e-4, 42);
    defer model.deinit();

    try model.fit(&train_data);

    var test_storage: [2][2]f64 = .{ .{ 0.05, 0.05 }, .{ 5.05, 5.05 } };
    var test_data: [2][]const f64 = undefined;
    for (&test_data, &test_storage) |*ptr, *storage| {
        ptr.* = storage;
    }

    const labels = try model.predict(&test_data);
    defer allocator.free(labels);

    try testing.expectEqual(@as(usize, 2), labels.len);
}

test "SpectralClustering: k_neighbors effect" {
    const allocator = testing.allocator;

    var data_storage: [6][2]f64 = .{
        .{ 0, 0 }, .{ 0.1, 0 }, .{ 0.2, 0 },
        .{ 3, 3 }, .{ 3.1, 3 }, .{ 3.2, 3 },
    };
    var data: [6][]const f64 = undefined;
    for (&data, &data_storage) |*ptr, *storage| {
        ptr.* = storage;
    }

    // Test with different k values
    for ([_]usize{ 2, 3, 5 }) |k| {
        var model = SpectralClustering(f64).init(allocator, 2, k, 1.0, .symmetric, 100, 1e-4, 42);
        defer model.deinit();

        try model.fit(&data);
        try testing.expect(model.labels != null);
    }
}

test "SpectralClustering: sigma bandwidth effect" {
    const allocator = testing.allocator;

    var data_storage: [6][2]f64 = .{
        .{ 0, 0 }, .{ 0.1, 0 }, .{ 0.2, 0 },
        .{ 3, 3 }, .{ 3.1, 3 }, .{ 3.2, 3 },
    };
    var data: [6][]const f64 = undefined;
    for (&data, &data_storage) |*ptr, *storage| {
        ptr.* = storage;
    }

    // Test with different sigma values
    for ([_]f64{ 0.5, 1.0, 2.0 }) |sigma| {
        var model = SpectralClustering(f64).init(allocator, 2, 3, sigma, .symmetric, 100, 1e-4, 42);
        defer model.deinit();

        try model.fit(&data);
        try testing.expect(model.labels != null);
    }
}

test "SpectralClustering: large dataset" {
    const allocator = testing.allocator;

    const n: usize = 50; // Smaller than other algorithms due to O(n³)
    const data = try allocator.alloc([]f64, n);
    defer {
        for (data) |row| allocator.free(row);
        allocator.free(data);
    }

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (data, 0..) |*row, i| {
        row.* = try allocator.alloc(f64, 2);
        if (i < n / 2) {
            row.*[0] = random.float(f64);
            row.*[1] = random.float(f64);
        } else {
            row.*[0] = random.float(f64) + 5;
            row.*[1] = random.float(f64) + 5;
        }
    }

    var model = SpectralClustering(f64).init(allocator, 2, 5, 1.0, .symmetric, 100, 1e-4, 42);
    defer model.deinit();

    try model.fit(data);
    try testing.expect(model.labels != null);
}

test "SpectralClustering: empty data error" {
    const allocator = testing.allocator;

    var data: [0][]const f64 = .{};
    var model = SpectralClustering(f64).init(allocator, 2, 3, 1.0, .symmetric, 100, 1e-4, 42);
    defer model.deinit();

    try testing.expectError(error.EmptyData, model.fit(&data));
}

test "SpectralClustering: invalid parameters" {
    const allocator = testing.allocator;

    var data_storage: [4][2]f64 = .{
        .{ 0, 0 }, .{ 0.1, 0.1 }, .{ 5, 5 }, .{ 5.1, 5.1 },
    };
    var data: [4][]const f64 = undefined;
    for (&data, &data_storage) |*ptr, *storage| {
        ptr.* = storage;
    }

    // Zero clusters
    {
        var model = SpectralClustering(f64).init(allocator, 0, 2, 1.0, .symmetric, 100, 1e-4, 42);
        defer model.deinit();
        try testing.expectError(error.InvalidParameters, model.fit(&data));
    }

    // More clusters than data points
    {
        var model = SpectralClustering(f64).init(allocator, 10, 2, 1.0, .symmetric, 100, 1e-4, 42);
        defer model.deinit();
        try testing.expectError(error.InvalidParameters, model.fit(&data));
    }
}

test "SpectralClustering: memory safety" {
    const allocator = testing.allocator;

    var data_storage: [6][2]f64 = .{
        .{ 0, 0 }, .{ 0.1, 0 }, .{ 0.2, 0 },
        .{ 3, 3 }, .{ 3.1, 3 }, .{ 3.2, 3 },
    };
    var data: [6][]const f64 = undefined;
    for (&data, &data_storage) |*ptr, *storage| {
        ptr.* = storage;
    }

    var model = SpectralClustering(f64).init(allocator, 2, 3, 1.0, .symmetric, 100, 1e-4, 42);
    defer model.deinit();

    try model.fit(&data);

    // Multiple fits should not leak
    try model.fit(&data);
    try model.fit(&data);
}

test "SpectralClustering: predict without fit" {
    const allocator = testing.allocator;

    var data_storage: [2][2]f64 = .{ .{ 0, 0 }, .{ 1, 1 } };
    var data: [2][]const f64 = undefined;
    for (&data, &data_storage) |*ptr, *storage| {
        ptr.* = storage;
    }

    const model = SpectralClustering(f64).init(allocator, 2, 2, 1.0, .symmetric, 100, 1e-4, 42);

    try testing.expectError(error.ModelNotFitted, model.predict(&data));
}
