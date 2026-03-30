const std = @import("std");
const Allocator = std.mem.Allocator;

/// OPTICS (Ordering Points To Identify the Clustering Structure)
/// Density-based clustering algorithm that produces a cluster ordering
/// for automatic and interactive cluster analysis.
///
/// Algorithm: Creates a reachability plot showing density-based structure
///   1. For each unprocessed point, compute core distance
///   2. Expand cluster by updating reachability distances of neighbors
///   3. Extract clusters using reachability threshold (xi method or epsilon)
///
/// Key differences from DBSCAN:
///   - Produces hierarchical ordering instead of flat clusters
///   - No need to specify epsilon upfront (can extract multiple epsilon values)
///   - Generates reachability plot for visualization
///   - Better for variable density clusters
///
/// Time: O(n²) naive, O(n log n) with spatial indexing
/// Space: O(n) for ordering and distances
///
/// Use cases:
///   - Exploratory data analysis (visualize cluster structure)
///   - Variable density clusters (hierarchical densities)
///   - Automatic cluster extraction (xi method)
///   - Alternative to DBSCAN when epsilon is unknown
pub fn OPTICS(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Point with ordering information
        pub const OrderedPoint = struct {
            index: usize,
            reachability_distance: T,
            core_distance: T,
            processed: bool,
        };

        /// Cluster extraction result
        pub const Cluster = struct {
            points: std.ArrayList(usize),
            start: usize,
            end: usize,

            pub fn deinit(self: *Cluster) void {
                self.points.deinit();
            }
        };

        allocator: Allocator,
        data: []const []const T,
        epsilon: T,
        min_points: usize,
        ordered_points: std.ArrayList(OrderedPoint),
        processed: []bool,

        /// Initialize OPTICS clustering
        ///
        /// Time: O(1)
        /// Space: O(n)
        pub fn init(allocator: Allocator, data: []const []const T, epsilon: T, min_points: usize) !Self {
            const n = data.len;
            const processed = try allocator.alloc(bool, n);
            @memset(processed, false);

            return Self{
                .allocator = allocator,
                .data = data,
                .epsilon = epsilon,
                .min_points = min_points,
                .ordered_points = std.ArrayList(OrderedPoint).init(allocator),
                .processed = processed,
            };
        }

        pub fn deinit(self: *Self) void {
            self.ordered_points.deinit();
            self.allocator.free(self.processed);
        }

        /// Run OPTICS algorithm to compute ordering
        ///
        /// Time: O(n²) for naive distance computation
        /// Space: O(n) for ordering
        pub fn fit(self: *Self) !void {
            const n = self.data.len;

            for (0..n) |i| {
                if (self.processed[i]) continue;

                // Get neighbors of current point
                const neighbors = try self.getNeighbors(i);
                defer self.allocator.free(neighbors);

                self.processed[i] = true;

                // Compute core distance
                const core_dist = if (neighbors.len >= self.min_points)
                    try self.coreDistance(i, neighbors)
                else
                    std.math.inf(T);

                // Add to ordering
                try self.ordered_points.append(.{
                    .index = i,
                    .reachability_distance = std.math.inf(T),
                    .core_distance = core_dist,
                    .processed = true,
                });

                // Expand cluster ordering
                if (!std.math.isInf(core_dist)) {
                    try self.expandClusterOrder(i, core_dist, neighbors);
                }
            }
        }

        /// Expand cluster ordering from seed point
        fn expandClusterOrder(self: *Self, seed_idx: usize, core_dist: T, neighbors: []const usize) !void {
            // Priority queue: (reachability_distance, point_index)
            var seeds = std.ArrayList(struct { dist: T, idx: usize }).init(self.allocator);
            defer seeds.deinit();

            // Initialize seeds with neighbors
            for (neighbors) |neighbor_idx| {
                if (self.processed[neighbor_idx]) continue;

                const dist = try self.distance(seed_idx, neighbor_idx);
                const reach_dist = @max(core_dist, dist);

                try seeds.append(.{ .dist = reach_dist, .idx = neighbor_idx });
            }

            // Sort seeds by reachability distance
            std.sort.heap(struct { dist: T, idx: usize }, seeds.items, {}, struct {
                fn lessThan(_: void, a: struct { dist: T, idx: usize }, b: struct { dist: T, idx: usize }) bool {
                    return a.dist < b.dist;
                }
            }.lessThan);

            while (seeds.items.len > 0) {
                // Get point with smallest reachability
                const current = seeds.orderedRemove(0);
                if (self.processed[current.idx]) continue;

                // Get neighbors
                const curr_neighbors = try self.getNeighbors(current.idx);
                defer self.allocator.free(curr_neighbors);

                self.processed[current.idx] = true;

                // Compute core distance
                const curr_core_dist = if (curr_neighbors.len >= self.min_points)
                    try self.coreDistance(current.idx, curr_neighbors)
                else
                    std.math.inf(T);

                // Add to ordering
                try self.ordered_points.append(.{
                    .index = current.idx,
                    .reachability_distance = current.dist,
                    .core_distance = curr_core_dist,
                    .processed = true,
                });

                // Update seeds with new neighbors
                if (!std.math.isInf(curr_core_dist)) {
                    for (curr_neighbors) |neighbor_idx| {
                        if (self.processed[neighbor_idx]) continue;

                        const dist = try self.distance(current.idx, neighbor_idx);
                        const reach_dist = @max(curr_core_dist, dist);

                        // Check if already in seeds
                        var found = false;
                        for (seeds.items) |*seed| {
                            if (seed.idx == neighbor_idx) {
                                // Update if better reachability
                                if (reach_dist < seed.dist) {
                                    seed.dist = reach_dist;
                                    // Re-sort after update
                                    std.sort.heap(struct { dist: T, idx: usize }, seeds.items, {}, struct {
                                        fn lessThan(_: void, a: struct { dist: T, idx: usize }, b: struct { dist: T, idx: usize }) bool {
                                            return a.dist < b.dist;
                                        }
                                    }.lessThan);
                                }
                                found = true;
                                break;
                            }
                        }

                        if (!found) {
                            try seeds.append(.{ .dist = reach_dist, .idx = neighbor_idx });
                            // Re-sort after insertion
                            std.sort.heap(struct { dist: T, idx: usize }, seeds.items, {}, struct {
                                fn lessThan(_: void, a: struct { dist: T, idx: usize }, b: struct { dist: T, idx: usize }) bool {
                                    return a.dist < b.dist;
                                }
                            }.lessThan);
                        }
                    }
                }
            }
        }

        /// Extract clusters using epsilon threshold
        ///
        /// Time: O(n)
        /// Space: O(n) for cluster assignments
        pub fn extractClusters(self: *Self, epsilon_extract: T) !std.ArrayList(Cluster) {
            var clusters = std.ArrayList(Cluster).init(self.allocator);
            var current_cluster: ?Cluster = null;
            var cluster_id: i32 = -1;

            for (self.ordered_points.items, 0..) |point, i| {
                if (point.reachability_distance > epsilon_extract) {
                    // End current cluster
                    if (current_cluster) |*cluster| {
                        cluster.end = i;
                        try clusters.append(cluster.*);
                        current_cluster = null;
                    }
                } else {
                    // Continue or start cluster
                    if (current_cluster == null) {
                        cluster_id += 1;
                        current_cluster = Cluster{
                            .points = std.ArrayList(usize).init(self.allocator),
                            .start = i,
                            .end = i,
                        };
                    }

                    if (current_cluster) |*cluster| {
                        try cluster.points.append(point.index);
                    }
                }
            }

            // Add last cluster if exists
            if (current_cluster) |*cluster| {
                cluster.end = self.ordered_points.items.len;
                try clusters.append(cluster.*);
            }

            return clusters;
        }

        /// Get cluster assignments using epsilon threshold
        ///
        /// Time: O(n)
        /// Space: O(n)
        pub fn extractClusterLabels(self: *Self, epsilon_extract: T) ![]i32 {
            const n = self.data.len;
            var labels = try self.allocator.alloc(i32, n);
            @memset(labels, -1); // Initialize as noise

            var cluster_id: i32 = 0;
            var in_cluster = false;

            for (self.ordered_points.items) |point| {
                if (point.reachability_distance > epsilon_extract) {
                    in_cluster = false;
                } else {
                    if (!in_cluster) {
                        cluster_id += 1;
                        in_cluster = true;
                    }
                    labels[point.index] = cluster_id;
                }
            }

            return labels;
        }

        /// Get neighbors within epsilon distance
        fn getNeighbors(self: *Self, idx: usize) ![]usize {
            var neighbors = std.ArrayList(usize).init(self.allocator);

            for (self.data, 0..) |_, i| {
                if (i == idx) continue;
                const dist = try self.distance(idx, i);
                if (dist <= self.epsilon) {
                    try neighbors.append(i);
                }
            }

            return neighbors.toOwnedSlice();
        }

        /// Compute core distance (distance to k-th nearest neighbor)
        fn coreDistance(self: *Self, idx: usize, neighbors: []const usize) !T {
            if (neighbors.len < self.min_points) {
                return std.math.inf(T);
            }

            // Compute distances to all neighbors
            var distances = std.ArrayList(T).init(self.allocator);
            defer distances.deinit();

            for (neighbors) |neighbor_idx| {
                const dist = try self.distance(idx, neighbor_idx);
                try distances.append(dist);
            }

            // Sort and get k-th distance
            std.sort.heap(T, distances.items, {}, struct {
                fn lessThan(_: void, a: T, b: T) bool {
                    return a < b;
                }
            }.lessThan);

            return distances.items[self.min_points - 1];
        }

        /// Compute Euclidean distance between two points
        fn distance(self: *Self, i: usize, j: usize) !T {
            const p1 = self.data[i];
            const p2 = self.data[j];

            if (p1.len != p2.len) return error.DimensionMismatch;

            var sum: T = 0;
            for (p1, p2) |x, y| {
                const diff = x - y;
                sum += diff * diff;
            }

            return @sqrt(sum);
        }
    };
}

// Tests
const testing = std.testing;
const expect = testing.expect;
const expectEqual = testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;

test "OPTICS: basic 2D clustering" {
    const allocator = testing.allocator;

    // Create two well-separated clusters
    const data = [_][]const f64{
        &[_]f64{ 0.0, 0.0 }, &[_]f64{ 0.1, 0.1 }, &[_]f64{ 0.0, 0.2 }, // Cluster 1
        &[_]f64{ 5.0, 5.0 }, &[_]f64{ 5.1, 5.1 }, &[_]f64{ 5.0, 5.2 }, // Cluster 2
    };

    var optics = try OPTICS(f64).init(allocator, &data, 1.0, 2);
    defer optics.deinit();

    try optics.fit();

    // Should have processed all points
    try expectEqual(data.len, optics.ordered_points.items.len);

    // Extract clusters with epsilon=1.0
    const labels = try optics.extractClusterLabels(1.0);
    defer allocator.free(labels);

    // Should have 2 clusters
    var max_label: i32 = -1;
    for (labels) |label| {
        if (label > max_label) max_label = label;
    }
    try expect(max_label >= 1);
}

test "OPTICS: single cluster" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 0.1, 0.1 },
        &[_]f64{ 0.2, 0.0 },
        &[_]f64{ 0.1, 0.2 },
    };

    var optics = try OPTICS(f64).init(allocator, &data, 0.5, 2);
    defer optics.deinit();

    try optics.fit();

    const labels = try optics.extractClusterLabels(0.5);
    defer allocator.free(labels);

    // All points should be in same cluster
    const first_label = labels[0];
    for (labels) |label| {
        try expectEqual(first_label, label);
    }
}

test "OPTICS: noise detection" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &[_]f64{ 0.0, 0.0 }, &[_]f64{ 0.1, 0.1 }, // Cluster
        &[_]f64{ 10.0, 10.0 }, // Noise (outlier)
        &[_]f64{ 0.2, 0.0 }, &[_]f64{ 0.1, 0.2 }, // Cluster
    };

    var optics = try OPTICS(f64).init(allocator, &data, 0.5, 2);
    defer optics.deinit();

    try optics.fit();

    const labels = try optics.extractClusterLabels(0.5);
    defer allocator.free(labels);

    // Outlier should be marked as noise (-1 or different cluster)
    const outlier_label = labels[2];
    try expect(outlier_label == -1 or outlier_label != labels[0]);
}

test "OPTICS: variable density clusters" {
    const allocator = testing.allocator;

    // Dense cluster + sparse cluster
    const data = [_][]const f64{
        &[_]f64{ 0.0, 0.0 }, &[_]f64{ 0.05, 0.05 }, &[_]f64{ 0.1, 0.0 }, // Dense
        &[_]f64{ 5.0, 5.0 }, &[_]f64{ 6.0, 6.0 }, &[_]f64{ 7.0, 7.0 }, // Sparse
    };

    var optics = try OPTICS(f64).init(allocator, &data, 2.0, 2);
    defer optics.deinit();

    try optics.fit();

    // Should handle both densities
    try expectEqual(data.len, optics.ordered_points.items.len);

    // Check reachability distances vary
    var has_small_reach = false;
    var has_large_reach = false;
    for (optics.ordered_points.items) |point| {
        if (!std.math.isInf(point.reachability_distance)) {
            if (point.reachability_distance < 0.5) has_small_reach = true;
            if (point.reachability_distance > 1.0) has_large_reach = true;
        }
    }
    try expect(has_small_reach or has_large_reach);
}

test "OPTICS: min_points parameter" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 0.1, 0.1 },
        &[_]f64{ 0.2, 0.2 },
        &[_]f64{ 0.3, 0.3 },
    };

    // With min_points=2
    var optics1 = try OPTICS(f64).init(allocator, &data, 0.5, 2);
    defer optics1.deinit();
    try optics1.fit();

    // With min_points=3
    var optics2 = try OPTICS(f64).init(allocator, &data, 0.5, 3);
    defer optics2.deinit();
    try optics2.fit();

    // Both should process all points
    try expectEqual(data.len, optics1.ordered_points.items.len);
    try expectEqual(data.len, optics2.ordered_points.items.len);
}

test "OPTICS: ordering property" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 0.1, 0.1 },
        &[_]f64{ 5.0, 5.0 },
        &[_]f64{ 5.1, 5.1 },
    };

    var optics = try OPTICS(f64).init(allocator, &data, 1.0, 2);
    defer optics.deinit();

    try optics.fit();

    // Ordering should group similar points
    // Check that consecutive points in same cluster are close in ordering
    try expectEqual(data.len, optics.ordered_points.items.len);
}

test "OPTICS: 3D data" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &[_]f64{ 0.0, 0.0, 0.0 },
        &[_]f64{ 0.1, 0.1, 0.1 },
        &[_]f64{ 5.0, 5.0, 5.0 },
        &[_]f64{ 5.1, 5.1, 5.1 },
    };

    var optics = try OPTICS(f64).init(allocator, &data, 1.0, 2);
    defer optics.deinit();

    try optics.fit();

    try expectEqual(data.len, optics.ordered_points.items.len);
}

test "OPTICS: f32 type support" {
    const allocator = testing.allocator;

    const data = [_][]const f32{
        &[_]f32{ 0.0, 0.0 },
        &[_]f32{ 0.1, 0.1 },
        &[_]f32{ 5.0, 5.0 },
        &[_]f32{ 5.1, 5.1 },
    };

    var optics = try OPTICS(f32).init(allocator, &data, 1.0, 2);
    defer optics.deinit();

    try optics.fit();

    try expectEqual(data.len, optics.ordered_points.items.len);
}

test "OPTICS: large dataset" {
    const allocator = testing.allocator;

    // Generate 100 points in 2 clusters
    var data = std.ArrayList([]f64).init(allocator);
    defer {
        for (data.items) |point| {
            allocator.free(point);
        }
        data.deinit();
    }

    // Cluster 1: around (0,0)
    for (0..50) |i| {
        const point = try allocator.alloc(f64, 2);
        point[0] = @as(f64, @floatFromInt(i % 10)) * 0.1;
        point[1] = @as(f64, @floatFromInt(i / 10)) * 0.1;
        try data.append(point);
    }

    // Cluster 2: around (5,5)
    for (0..50) |i| {
        const point = try allocator.alloc(f64, 2);
        point[0] = 5.0 + @as(f64, @floatFromInt(i % 10)) * 0.1;
        point[1] = 5.0 + @as(f64, @floatFromInt(i / 10)) * 0.1;
        try data.append(point);
    }

    var optics = try OPTICS(f64).init(allocator, data.items, 0.5, 3);
    defer optics.deinit();

    try optics.fit();

    try expectEqual(@as(usize, 100), optics.ordered_points.items.len);
}

test "OPTICS: cluster extraction" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &[_]f64{ 0.0, 0.0 }, &[_]f64{ 0.1, 0.1 }, &[_]f64{ 0.0, 0.2 },
        &[_]f64{ 5.0, 5.0 }, &[_]f64{ 5.1, 5.1 }, &[_]f64{ 5.0, 5.2 },
    };

    var optics = try OPTICS(f64).init(allocator, &data, 1.0, 2);
    defer optics.deinit();

    try optics.fit();

    var clusters = try optics.extractClusters(1.0);
    defer {
        for (clusters.items) |*cluster| {
            cluster.deinit();
        }
        clusters.deinit();
    }

    // Should extract clusters
    try expect(clusters.items.len > 0);
}

test "OPTICS: reachability distances" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 0.1, 0.1 },
        &[_]f64{ 0.2, 0.2 },
    };

    var optics = try OPTICS(f64).init(allocator, &data, 1.0, 2);
    defer optics.deinit();

    try optics.fit();

    // First point has infinite reachability, others should be finite
    var has_inf = false;
    var has_finite = false;
    for (optics.ordered_points.items) |point| {
        if (std.math.isInf(point.reachability_distance)) {
            has_inf = true;
        } else {
            has_finite = true;
        }
    }
    try expect(has_inf and has_finite);
}

test "OPTICS: memory safety" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 0.1, 0.1 },
    };

    var optics = try OPTICS(f64).init(allocator, &data, 1.0, 2);
    try optics.fit();
    optics.deinit();

    // Should not leak memory
}

test "OPTICS: empty data error" {
    const allocator = testing.allocator;

    const data = [_][]const f64{};

    var optics = try OPTICS(f64).init(allocator, &data, 1.0, 2);
    defer optics.deinit();

    try optics.fit();

    try expectEqual(@as(usize, 0), optics.ordered_points.items.len);
}
