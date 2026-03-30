/// Hierarchical Clustering
///
/// Builds a hierarchy of clusters using bottom-up (agglomerative) or top-down (divisive) approach.
/// This implementation focuses on agglomerative clustering with various linkage methods.
///
/// **Algorithm**: Agglomerative Hierarchical Clustering
/// - Start with each point as its own cluster
/// - Repeatedly merge the two closest clusters
/// - Continue until desired number of clusters or all merged
/// - Produces a dendrogram (tree) showing merge history
///
/// **Linkage Methods**:
/// - Single: Minimum distance between any two points (can create long chains)
/// - Complete: Maximum distance between any two points (compact clusters)
/// - Average: Average distance between all pairs (balanced approach)
/// - Ward: Minimize within-cluster variance (tends to create equal-sized clusters)
///
/// **Time Complexity**: O(n³) for naive implementation, O(n² log n) with priority queue
/// **Space Complexity**: O(n²) for distance matrix + O(n) for merge tree
///
/// **Use Cases**:
/// - Taxonomic classification (biology, linguistics)
/// - Document clustering with dendrogram visualization
/// - Image segmentation hierarchies
/// - Market segmentation with multiple granularities
/// - Gene expression analysis
///
/// **Advantages**:
/// - No need to specify number of clusters upfront
/// - Produces interpretable dendrogram
/// - Works with any distance metric
/// - Can cut tree at any level to get desired number of clusters
///
/// **Disadvantages**:
/// - O(n³) time complexity (expensive for large datasets)
/// - O(n²) space for distance matrix
/// - Cannot undo merges (greedy approach)
/// - Sensitive to noise and outliers
///
/// **Example**:
/// ```zig
/// const data = [_]f64{ 1.0, 1.5, 5.0, 5.5, 9.0 };
/// const points = try std.ArrayList([]const f64).init(allocator);
/// defer points.deinit();
/// for (data) |val| {
///     const point = try allocator.alloc(f64, 1);
///     point[0] = val;
///     try points.append(point);
/// }
///
/// const config = HierarchicalClusteringConfig(f64){
///     .linkage = .complete,
///     .n_clusters = 3,
/// };
///
/// var hc = try HierarchicalClustering(f64).init(allocator, 1, config);
/// defer hc.deinit();
///
/// try hc.fit(points.items);
/// const labels = try hc.predict(allocator);
/// defer allocator.free(labels);
/// // labels might be: [0, 0, 1, 1, 2] (3 clusters)
/// ```

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Linkage method for determining cluster distance
pub const LinkageMethod = enum {
    /// Minimum distance between any two points in different clusters
    /// Can produce long, chain-like clusters
    single,

    /// Maximum distance between any two points in different clusters
    /// Produces compact, spherical clusters
    complete,

    /// Average distance between all pairs of points in different clusters
    /// Balanced approach between single and complete
    average,

    /// Minimize increase in within-cluster variance (sum of squared distances)
    /// Tends to create equal-sized clusters, most commonly used
    ward,
};

/// Configuration for hierarchical clustering
pub fn HierarchicalClusteringConfig(comptime _: type) type {
    return struct {
        /// Linkage method for cluster distance calculation
        linkage: LinkageMethod = .ward,

        /// Number of clusters to extract from dendrogram
        /// If null, returns all merge steps
        n_clusters: ?usize = null,

        /// Distance metric: euclidean or squared euclidean (for Ward)
        /// True = squared euclidean (faster, suitable for Ward)
        /// False = euclidean (standard distance)
        squared_distance: bool = true,
    };
}

/// Merge step in the dendrogram
pub fn MergeStep(comptime T: type) type {
    return struct {
        /// First cluster index being merged
        cluster1: usize,
        /// Second cluster index being merged
        cluster2: usize,
        /// Distance at which merge occurred
        distance: T,
        /// Size of resulting cluster
        size: usize,
    };
}

/// Hierarchical Clustering implementation
///
/// Time: O(n³) naive, O(n² log n) with priority queue
/// Space: O(n²) for distance matrix + O(n) for tree
pub fn HierarchicalClustering(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        n_features: usize,
        config: HierarchicalClusteringConfig(T),

        /// Distance matrix between clusters (triangular storage)
        distance_matrix: ?[]T = null,
        /// Number of points in each cluster
        cluster_sizes: ?[]usize = null,
        /// Parent cluster for each cluster in merge tree
        cluster_parents: ?[]usize = null,
        /// Merge history (dendrogram)
        merges: ?[]MergeStep(T) = null,
        /// Final cluster labels
        labels: ?[]usize = null,
        /// Number of original data points
        n_samples: usize = 0,

        /// Initialize hierarchical clustering
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(allocator: Allocator, n_features: usize, config: HierarchicalClusteringConfig(T)) !Self {
            if (n_features == 0) return error.InvalidDimension;

            return Self{
                .allocator = allocator,
                .n_features = n_features,
                .config = config,
            };
        }

        /// Free all resources
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.distance_matrix) |dm| self.allocator.free(dm);
            if (self.cluster_sizes) |cs| self.allocator.free(cs);
            if (self.cluster_parents) |cp| self.allocator.free(cp);
            if (self.merges) |m| self.allocator.free(m);
            if (self.labels) |l| self.allocator.free(l);
        }

        /// Compute distance matrix for initial points
        ///
        /// Time: O(n² × d) where d = n_features
        /// Space: O(n²)
        fn computeDistanceMatrix(self: *Self, points: []const []const T) !void {
            const n = points.len;
            self.n_samples = n;

            // Allocate triangular distance matrix (only upper triangle)
            const matrix_size = (n * (n - 1)) / 2;
            self.distance_matrix = try self.allocator.alloc(T, matrix_size);

            // Compute pairwise distances
            var idx: usize = 0;
            for (0..n) |i| {
                for (i + 1..n) |j| {
                    self.distance_matrix.?[idx] = try self.computeDistance(points[i], points[j]);
                    idx += 1;
                }
            }
        }

        /// Get distance from triangular matrix
        ///
        /// Time: O(1)
        inline fn getDistance(self: *const Self, i: usize, j: usize) T {
            if (i == j) return 0.0;
            const min_idx = @min(i, j);
            const max_idx = @max(i, j);
            const n = self.n_samples;
            // Triangular indexing: row i, col j (j > i)
            const idx = (min_idx * (2 * n - min_idx - 1)) / 2 + (max_idx - min_idx - 1);
            return self.distance_matrix.?[idx];
        }

        /// Set distance in triangular matrix
        ///
        /// Time: O(1)
        inline fn setDistance(self: *Self, i: usize, j: usize, dist: T) void {
            if (i == j) return;
            const min_idx = @min(i, j);
            const max_idx = @max(i, j);
            const n = self.n_samples;
            const idx = (min_idx * (2 * n - min_idx - 1)) / 2 + (max_idx - min_idx - 1);
            self.distance_matrix.?[idx] = dist;
        }

        /// Compute distance between two points
        ///
        /// Time: O(d) where d = n_features
        fn computeDistance(self: *const Self, p1: []const T, p2: []const T) !T {
            if (p1.len != self.n_features or p2.len != self.n_features) {
                return error.DimensionMismatch;
            }

            var sum: T = 0.0;
            for (p1, p2) |v1, v2| {
                const diff = v1 - v2;
                sum += diff * diff;
            }

            if (self.config.squared_distance) {
                return sum;
            } else {
                return @sqrt(sum);
            }
        }

        /// Fit hierarchical clustering to data
        ///
        /// Time: O(n³) naive implementation
        /// Space: O(n²) for distance matrix
        pub fn fit(self: *Self, points: []const []const T) !void {
            if (points.len == 0) return error.EmptyDataset;

            const n = points.len;

            // Initialize distance matrix
            try self.computeDistanceMatrix(points);

            // Initialize cluster metadata
            self.cluster_sizes = try self.allocator.alloc(usize, n);
            @memset(self.cluster_sizes.?, 1);

            self.cluster_parents = try self.allocator.alloc(usize, n);
            for (self.cluster_parents.?, 0..) |*parent, i| {
                parent.* = i; // Each point is its own cluster initially
            }

            // Allocate merge history
            self.merges = try self.allocator.alloc(MergeStep(T), n - 1);

            // Track active clusters (not yet merged)
            var active = try self.allocator.alloc(bool, n);
            defer self.allocator.free(active);
            @memset(active, true);

            // Agglomerative clustering: merge n-1 times
            var n_clusters = n;
            for (0..n - 1) |merge_idx| {
                // Find closest pair of clusters
                var min_dist: T = std.math.inf(T);
                var best_i: usize = 0;
                var best_j: usize = 1;

                for (0..n) |i| {
                    if (!active[i]) continue;
                    for (i + 1..n) |j| {
                        if (!active[j]) continue;

                        const dist = self.getDistance(i, j);
                        if (dist < min_dist) {
                            min_dist = dist;
                            best_i = i;
                            best_j = j;
                        }
                    }
                }

                // Record merge
                self.merges.?[merge_idx] = MergeStep(T){
                    .cluster1 = best_i,
                    .cluster2 = best_j,
                    .distance = min_dist,
                    .size = self.cluster_sizes.?[best_i] + self.cluster_sizes.?[best_j],
                };

                // Update distances using linkage method
                try self.updateDistances(best_i, best_j, active);

                // Merge cluster j into cluster i
                self.cluster_sizes.?[best_i] += self.cluster_sizes.?[best_j];
                active[best_j] = false;
                n_clusters -= 1;

                // Stop if we reached desired number of clusters
                if (self.config.n_clusters) |target| {
                    if (n_clusters == target) break;
                }
            }

            // Extract cluster labels
            try self.extractLabels(active);
        }

        /// Update distances after merging clusters i and j
        ///
        /// Time: O(n)
        fn updateDistances(self: *Self, i: usize, j: usize, active: []const bool) !void {
            const n = self.n_samples;

            for (0..n) |k| {
                if (k == i or k == j or !active[k]) continue;

                const dist_ik = self.getDistance(i, k);
                const dist_jk = self.getDistance(j, k);

                const new_dist = switch (self.config.linkage) {
                    .single => @min(dist_ik, dist_jk),
                    .complete => @max(dist_ik, dist_jk),
                    .average => blk: {
                        const size_i = @as(T, @floatFromInt(self.cluster_sizes.?[i]));
                        const size_j = @as(T, @floatFromInt(self.cluster_sizes.?[j]));
                        const total = size_i + size_j;
                        break :blk (dist_ik * size_i + dist_jk * size_j) / total;
                    },
                    .ward => blk: {
                        const size_i = @as(T, @floatFromInt(self.cluster_sizes.?[i]));
                        const size_j = @as(T, @floatFromInt(self.cluster_sizes.?[j]));
                        const size_k = @as(T, @floatFromInt(self.cluster_sizes.?[k]));
                        const total = size_i + size_j + size_k;
                        const t1 = ((size_i + size_k) * dist_ik);
                        const t2 = ((size_j + size_k) * dist_jk);
                        const t3 = (size_k * self.getDistance(i, j));
                        break :blk (t1 + t2 - t3) / total;
                    },
                };

                self.setDistance(i, k, new_dist);
            }
        }

        /// Extract cluster labels from merge tree
        ///
        /// Time: O(n²) worst case for finding clusters
        /// Space: O(n)
        fn extractLabels(self: *Self, active: []const bool) !void {
            const n = self.n_samples;
            self.labels = try self.allocator.alloc(usize, n);

            // Find active clusters (cluster representatives)
            var representatives = std.ArrayList(usize).init(self.allocator);
            defer representatives.deinit();

            for (active, 0..) |is_active, i| {
                if (is_active) {
                    try representatives.append(i);
                }
            }

            // If we merged all the way, need to find final clusters by cutting tree
            if (representatives.items.len == 1 and self.config.n_clusters != null) {
                const target = self.config.n_clusters.?;
                if (target >= n) {
                    // Each point is its own cluster
                    for (self.labels.?, 0..) |*label, i| {
                        label.* = i;
                    }
                    return;
                }

                // Cut dendrogram at merge step to get target number of clusters
                const cut_step = n - target - 1;
                representatives.clearRetainingCapacity();

                // Rebuild active clusters up to cut point
                var temp_active = try self.allocator.alloc(bool, n);
                defer self.allocator.free(temp_active);
                @memset(temp_active, true);

                for (0..cut_step + 1) |merge_idx| {
                    const merge = self.merges.?[merge_idx];
                    temp_active[merge.cluster2] = false;
                }

                for (temp_active, 0..) |is_active, i| {
                    if (is_active) {
                        try representatives.append(i);
                    }
                }
            }

            // Assign labels: find which cluster each point belongs to
            for (self.labels.?, 0..) |*label, point_idx| {
                // Find the representative cluster this point merged into
                var cluster_id = point_idx;

                // Trace up merge tree to find representative
                while (true) {
                    var found = false;
                    for (representatives.items) |rep| {
                        if (cluster_id == rep) {
                            found = true;
                            break;
                        }
                    }
                    if (found) break;

                    // Find merge that absorbed this cluster
                    for (self.merges.?[0 .. self.merges.?.len]) |merge| {
                        if (merge.cluster2 == cluster_id) {
                            cluster_id = merge.cluster1;
                            break;
                        }
                    }
                }

                // Assign label based on representative index
                for (representatives.items, 0..) |rep, idx| {
                    if (rep == cluster_id) {
                        label.* = idx;
                        break;
                    }
                }
            }
        }

        /// Get cluster labels
        ///
        /// Time: O(n)
        /// Space: O(n) for returned copy
        pub fn predict(self: *const Self, allocator: Allocator) ![]usize {
            if (self.labels == null) return error.NotFitted;
            const labels = try allocator.alloc(usize, self.labels.?.len);
            @memcpy(labels, self.labels.?);
            return labels;
        }

        /// Get merge history (dendrogram)
        ///
        /// Time: O(n)
        /// Space: O(n) for returned copy
        pub fn getDendrogram(self: *const Self, allocator: Allocator) ![]MergeStep(T) {
            if (self.merges == null) return error.NotFitted;
            const merges = try allocator.alloc(MergeStep(T), self.merges.?.len);
            @memcpy(merges, self.merges.?);
            return merges;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "hierarchical clustering: basic 1D clustering" {
    const allocator = testing.allocator;

    // Create 3 well-separated groups: [1, 1.5], [5, 5.5], [9]
    var points = std.ArrayList([]f64).init(allocator);
    defer {
        for (points.items) |point| allocator.free(point);
        points.deinit();
    }

    const values = [_]f64{ 1.0, 1.5, 5.0, 5.5, 9.0 };
    for (values) |val| {
        const point = try allocator.alloc(f64, 1);
        point[0] = val;
        try points.append(point);
    }

    const config = HierarchicalClusteringConfig(f64){
        .linkage = .complete,
        .n_clusters = 3,
    };

    var hc = try HierarchicalClustering(f64).init(allocator, 1, config);
    defer hc.deinit();

    try hc.fit(points.items);

    const labels = try hc.predict(allocator);
    defer allocator.free(labels);

    // Check that we have 3 clusters
    var unique_labels = std.AutoHashMap(usize, void).init(allocator);
    defer unique_labels.deinit();
    for (labels) |label| {
        try unique_labels.put(label, {});
    }
    try testing.expectEqual(@as(usize, 3), unique_labels.count());

    // Points 0 and 1 should be in same cluster (close together)
    try testing.expectEqual(labels[0], labels[1]);

    // Points 2 and 3 should be in same cluster
    try testing.expectEqual(labels[2], labels[3]);

    // Point 4 should be in different cluster
    try testing.expect(labels[4] != labels[0]);
    try testing.expect(labels[4] != labels[2]);
}

test "hierarchical clustering: 2D data" {
    const allocator = testing.allocator;

    var points = std.ArrayList([]f64).init(allocator);
    defer {
        for (points.items) |point| allocator.free(point);
        points.deinit();
    }

    // Create 2 clusters: [(0,0), (1,1)] and [(10,10), (11,11)]
    const coords = [_][2]f64{
        .{ 0.0, 0.0 },
        .{ 1.0, 1.0 },
        .{ 10.0, 10.0 },
        .{ 11.0, 11.0 },
    };

    for (coords) |coord| {
        const point = try allocator.alloc(f64, 2);
        point[0] = coord[0];
        point[1] = coord[1];
        try points.append(point);
    }

    const config = HierarchicalClusteringConfig(f64){
        .linkage = .average,
        .n_clusters = 2,
    };

    var hc = try HierarchicalClustering(f64).init(allocator, 2, config);
    defer hc.deinit();

    try hc.fit(points.items);

    const labels = try hc.predict(allocator);
    defer allocator.free(labels);

    // Check 2 clusters
    var unique_labels = std.AutoHashMap(usize, void).init(allocator);
    defer unique_labels.deinit();
    for (labels) |label| {
        try unique_labels.put(label, {});
    }
    try testing.expectEqual(@as(usize, 2), unique_labels.count());

    // Points 0 and 1 should be together
    try testing.expectEqual(labels[0], labels[1]);

    // Points 2 and 3 should be together
    try testing.expectEqual(labels[2], labels[3]);

    // The two groups should differ
    try testing.expect(labels[0] != labels[2]);
}

test "hierarchical clustering: single linkage" {
    const allocator = testing.allocator;

    var points = std.ArrayList([]f64).init(allocator);
    defer {
        for (points.items) |point| allocator.free(point);
        points.deinit();
    }

    const values = [_]f64{ 0.0, 1.0, 5.0, 6.0 };
    for (values) |val| {
        const point = try allocator.alloc(f64, 1);
        point[0] = val;
        try points.append(point);
    }

    const config = HierarchicalClusteringConfig(f64){
        .linkage = .single,
        .n_clusters = 2,
    };

    var hc = try HierarchicalClustering(f64).init(allocator, 1, config);
    defer hc.deinit();

    try hc.fit(points.items);

    const labels = try hc.predict(allocator);
    defer allocator.free(labels);

    try testing.expectEqual(@as(usize, 2), blk: {
        var unique = std.AutoHashMap(usize, void).init(allocator);
        defer unique.deinit();
        for (labels) |l| try unique.put(l, {});
        break :blk unique.count();
    });
}

test "hierarchical clustering: ward linkage" {
    const allocator = testing.allocator;

    var points = std.ArrayList([]f64).init(allocator);
    defer {
        for (points.items) |point| allocator.free(point);
        points.deinit();
    }

    const values = [_]f64{ 0.0, 1.0, 5.0, 6.0 };
    for (values) |val| {
        const point = try allocator.alloc(f64, 1);
        point[0] = val;
        try points.append(point);
    }

    const config = HierarchicalClusteringConfig(f64){
        .linkage = .ward,
        .n_clusters = 2,
    };

    var hc = try HierarchicalClustering(f64).init(allocator, 1, config);
    defer hc.deinit();

    try hc.fit(points.items);

    const labels = try hc.predict(allocator);
    defer allocator.free(labels);

    try testing.expectEqual(@as(usize, 2), blk: {
        var unique = std.AutoHashMap(usize, void).init(allocator);
        defer unique.deinit();
        for (labels) |l| try unique.put(l, {});
        break :blk unique.count();
    });
}

test "hierarchical clustering: dendrogram" {
    const allocator = testing.allocator;

    var points = std.ArrayList([]f64).init(allocator);
    defer {
        for (points.items) |point| allocator.free(point);
        points.deinit();
    }

    const values = [_]f64{ 0.0, 1.0, 5.0 };
    for (values) |val| {
        const point = try allocator.alloc(f64, 1);
        point[0] = val;
        try points.append(point);
    }

    const config = HierarchicalClusteringConfig(f64){
        .linkage = .complete,
        .n_clusters = null, // Get full dendrogram
    };

    var hc = try HierarchicalClustering(f64).init(allocator, 1, config);
    defer hc.deinit();

    try hc.fit(points.items);

    const dendrogram = try hc.getDendrogram(allocator);
    defer allocator.free(dendrogram);

    // Should have n-1 merges
    try testing.expectEqual(@as(usize, 2), dendrogram.len);

    // First merge should be closest pair
    try testing.expect(dendrogram[0].distance < dendrogram[1].distance);
}

test "hierarchical clustering: single point" {
    const allocator = testing.allocator;

    var points = std.ArrayList([]f64).init(allocator);
    defer {
        for (points.items) |point| allocator.free(point);
        points.deinit();
    }

    const point = try allocator.alloc(f64, 1);
    point[0] = 5.0;
    try points.append(point);

    const config = HierarchicalClusteringConfig(f64){
        .n_clusters = 1,
    };

    var hc = try HierarchicalClustering(f64).init(allocator, 1, config);
    defer hc.deinit();

    try hc.fit(points.items);

    const labels = try hc.predict(allocator);
    defer allocator.free(labels);

    try testing.expectEqual(@as(usize, 1), labels.len);
    try testing.expectEqual(@as(usize, 0), labels[0]);
}

test "hierarchical clustering: all points same" {
    const allocator = testing.allocator;

    var points = std.ArrayList([]f64).init(allocator);
    defer {
        for (points.items) |point| allocator.free(point);
        points.deinit();
    }

    for (0..5) |_| {
        const point = try allocator.alloc(f64, 2);
        point[0] = 1.0;
        point[1] = 2.0;
        try points.append(point);
    }

    const config = HierarchicalClusteringConfig(f64){
        .n_clusters = 2,
    };

    var hc = try HierarchicalClustering(f64).init(allocator, 2, config);
    defer hc.deinit();

    try hc.fit(points.items);

    const labels = try hc.predict(allocator);
    defer allocator.free(labels);

    // All should be clustered (though arbitrarily since distances are 0)
    try testing.expectEqual(@as(usize, 5), labels.len);
}

test "hierarchical clustering: empty dataset error" {
    const allocator = testing.allocator;

    var points = std.ArrayList([]f64).init(allocator);
    defer points.deinit();

    const config = HierarchicalClusteringConfig(f64){ .n_clusters = 1 };

    var hc = try HierarchicalClustering(f64).init(allocator, 1, config);
    defer hc.deinit();

    try testing.expectError(error.EmptyDataset, hc.fit(points.items));
}

test "hierarchical clustering: dimension mismatch" {
    const allocator = testing.allocator;

    var points = std.ArrayList([]f64).init(allocator);
    defer {
        for (points.items) |point| allocator.free(point);
        points.deinit();
    }

    const point1 = try allocator.alloc(f64, 2);
    point1[0] = 1.0;
    point1[1] = 2.0;
    try points.append(point1);

    const point2 = try allocator.alloc(f64, 3); // Wrong dimension
    point2[0] = 3.0;
    point2[1] = 4.0;
    point2[2] = 5.0;
    try points.append(point2);

    const config = HierarchicalClusteringConfig(f64){ .n_clusters = 2 };

    var hc = try HierarchicalClustering(f64).init(allocator, 2, config);
    defer hc.deinit();

    try testing.expectError(error.DimensionMismatch, hc.fit(points.items));
}

test "hierarchical clustering: f32 support" {
    const allocator = testing.allocator;

    var points = std.ArrayList([]f32).init(allocator);
    defer {
        for (points.items) |point| allocator.free(point);
        points.deinit();
    }

    const values = [_]f32{ 1.0, 1.5, 5.0, 5.5 };
    for (values) |val| {
        const point = try allocator.alloc(f32, 1);
        point[0] = val;
        try points.append(point);
    }

    const config = HierarchicalClusteringConfig(f32){
        .linkage = .average,
        .n_clusters = 2,
    };

    var hc = try HierarchicalClustering(f32).init(allocator, 1, config);
    defer hc.deinit();

    try hc.fit(points.items);

    const labels = try hc.predict(allocator);
    defer allocator.free(labels);

    try testing.expectEqual(@as(usize, 2), blk: {
        var unique = std.AutoHashMap(usize, void).init(allocator);
        defer unique.deinit();
        for (labels) |l| try unique.put(l, {});
        break :blk unique.count();
    });

    try testing.expectEqual(labels[0], labels[1]);
    try testing.expectEqual(labels[2], labels[3]);
}

test "hierarchical clustering: large dataset stress test" {
    const allocator = testing.allocator;

    var points = std.ArrayList([]f64).init(allocator);
    defer {
        for (points.items) |point| allocator.free(point);
        points.deinit();
    }

    // Create 50 points in 3 clear groups
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // Group 1: around (0, 0)
    for (0..15) |_| {
        const point = try allocator.alloc(f64, 2);
        point[0] = random.float(f64) * 2.0;
        point[1] = random.float(f64) * 2.0;
        try points.append(point);
    }

    // Group 2: around (10, 10)
    for (0..15) |_| {
        const point = try allocator.alloc(f64, 2);
        point[0] = 10.0 + random.float(f64) * 2.0;
        point[1] = 10.0 + random.float(f64) * 2.0;
        try points.append(point);
    }

    // Group 3: around (20, 0)
    for (0..20) |_| {
        const point = try allocator.alloc(f64, 2);
        point[0] = 20.0 + random.float(f64) * 2.0;
        point[1] = random.float(f64) * 2.0;
        try points.append(point);
    }

    const config = HierarchicalClusteringConfig(f64){
        .linkage = .ward,
        .n_clusters = 3,
    };

    var hc = try HierarchicalClustering(f64).init(allocator, 2, config);
    defer hc.deinit();

    try hc.fit(points.items);

    const labels = try hc.predict(allocator);
    defer allocator.free(labels);

    try testing.expectEqual(@as(usize, 50), labels.len);

    // Should have 3 clusters
    var unique_labels = std.AutoHashMap(usize, void).init(allocator);
    defer unique_labels.deinit();
    for (labels) |label| {
        try unique_labels.put(label, {});
    }
    try testing.expectEqual(@as(usize, 3), unique_labels.count());
}

test "hierarchical clustering: predict without fit" {
    const allocator = testing.allocator;

    const config = HierarchicalClusteringConfig(f64){ .n_clusters = 2 };

    var hc = try HierarchicalClustering(f64).init(allocator, 1, config);
    defer hc.deinit();

    try testing.expectError(error.NotFitted, hc.predict(allocator));
}

test "hierarchical clustering: dendrogram without fit" {
    const allocator = testing.allocator;

    const config = HierarchicalClusteringConfig(f64){ .n_clusters = 2 };

    var hc = try HierarchicalClustering(f64).init(allocator, 1, config);
    defer hc.deinit();

    try testing.expectError(error.NotFitted, hc.getDendrogram(allocator));
}

test "hierarchical clustering: memory safety" {
    const allocator = testing.allocator;

    var points = std.ArrayList([]f64).init(allocator);
    defer {
        for (points.items) |point| allocator.free(point);
        points.deinit();
    }

    for (0..10) |i| {
        const point = try allocator.alloc(f64, 2);
        point[0] = @floatFromInt(i);
        point[1] = @floatFromInt(i * 2);
        try points.append(point);
    }

    const config = HierarchicalClusteringConfig(f64){ .n_clusters = 3 };

    var hc = try HierarchicalClustering(f64).init(allocator, 2, config);
    defer hc.deinit();

    try hc.fit(points.items);

    const labels = try hc.predict(allocator);
    defer allocator.free(labels);

    // No memory leaks should be detected by testing.allocator
}
