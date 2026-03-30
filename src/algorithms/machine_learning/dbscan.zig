//! DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
//!
//! DBSCAN is a density-based clustering algorithm that groups together points that are closely
//! packed together (points with many nearby neighbors), marking as outliers points that lie
//! alone in low-density regions.
//!
//! Key concepts:
//! - **Core point**: A point with at least `minPts` neighbors within radius `epsilon`
//! - **Border point**: A non-core point within epsilon of a core point
//! - **Noise point**: A point that is neither core nor border (outlier)
//! - **Density-reachable**: Points connected through core points form a cluster
//!
//! Advantages over K-Means:
//! - No need to specify number of clusters in advance
//! - Can find arbitrarily shaped clusters (not just spherical)
//! - Robust to outliers (marks them as noise)
//! - Deterministic (same result each run for same parameters)
//!
//! Time complexity: O(n²) naive, O(n log n) with spatial index
//! Space complexity: O(n) for cluster labels
//!
//! Example:
//! ```zig
//! const allocator = std.testing.allocator;
//! const data = [_][2]f64{
//!     .{ 1.0, 1.0 }, .{ 1.5, 1.5 }, .{ 2.0, 2.0 }, // cluster 1
//!     .{ 10.0, 10.0 }, .{ 10.5, 10.5 }, .{ 11.0, 11.0 }, // cluster 2
//!     .{ 50.0, 50.0 }, // noise
//! };
//! var result = try dbscan(f64, 2, allocator, &data, .{ .epsilon = 1.0, .min_pts = 2 });
//! defer result.deinit(allocator);
//! // result.labels: [0, 0, 0, 1, 1, 1, -1] (cluster 0, cluster 1, noise)
//! ```

const std = @import("std");
const Allocator = std.mem.Allocator;

/// DBSCAN clustering result
///
/// Cluster labels: 0, 1, 2, ... for clusters, -1 for noise points
pub fn DBSCANResult(comptime T: type) type {
    _ = T; // Type parameter used for generic interface consistency
    return struct {
        const Self = @This();

        /// Cluster label for each point (-1 = noise, 0+ = cluster ID)
        labels: []i32,
        /// Number of clusters found (excluding noise)
        n_clusters: usize,
        /// Number of noise points (outliers)
        n_noise: usize,
        /// Core point flags (true if point is a core point)
        core_points: []bool,

        /// Free all allocated memory
        ///
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self, allocator: Allocator) void {
            allocator.free(self.labels);
            allocator.free(self.core_points);
        }

        /// Get cluster sizes (excluding noise)
        ///
        /// Returns an array where index i contains the number of points in cluster i.
        /// Caller owns the returned memory.
        ///
        /// Time: O(n) | Space: O(k) where k = number of clusters
        pub fn getClusterSizes(self: Self, allocator: Allocator) ![]usize {
            var sizes = try allocator.alloc(usize, self.n_clusters);
            @memset(sizes, 0);

            for (self.labels) |label| {
                if (label >= 0) {
                    sizes[@intCast(label)] += 1;
                }
            }

            return sizes;
        }

        /// Get indices of points in a specific cluster
        ///
        /// Returns an ArrayList of indices. Caller must deinit the returned list.
        ///
        /// Time: O(n) | Space: O(m) where m = cluster size
        pub fn getClusterIndices(self: Self, allocator: Allocator, cluster_id: usize) !std.ArrayList(usize) {
            var indices = std.ArrayList(usize).init(allocator);
            errdefer indices.deinit();

            for (self.labels, 0..) |label, i| {
                if (label == @as(i32, @intCast(cluster_id))) {
                    try indices.append(i);
                }
            }

            return indices;
        }

        /// Get indices of noise points (outliers)
        ///
        /// Returns an ArrayList of indices. Caller must deinit the returned list.
        ///
        /// Time: O(n) | Space: O(m) where m = number of noise points
        pub fn getNoiseIndices(self: Self, allocator: Allocator) !std.ArrayList(usize) {
            var indices = std.ArrayList(usize).init(allocator);
            errdefer indices.deinit();

            for (self.labels, 0..) |label, i| {
                if (label == -1) {
                    try indices.append(i);
                }
            }

            return indices;
        }
    };
}

/// DBSCAN clustering options
pub const DBSCANOptions = struct {
    /// Neighborhood radius (maximum distance for two points to be neighbors)
    epsilon: f64 = 0.5,
    /// Minimum number of points required to form a dense region (core point)
    min_pts: usize = 5,
};

/// Perform DBSCAN clustering
///
/// Parameters:
/// - `T`: floating-point type (f32 or f64)
/// - `ndim`: number of dimensions
/// - `allocator`: memory allocator
/// - `data`: array of points (each point is [ndim]T)
/// - `options`: clustering parameters (epsilon, min_pts)
///
/// Returns:
/// - `DBSCANResult(T)`: cluster labels, counts, and core point flags
///
/// Time: O(n²) naive distance computation | Space: O(n)
pub fn dbscan(
    comptime T: type,
    comptime ndim: usize,
    allocator: Allocator,
    data: []const [ndim]T,
    options: DBSCANOptions,
) !DBSCANResult(T) {
    const n = data.len;

    // Initialize labels: -2 = unvisited, -1 = noise, 0+ = cluster ID
    const labels = try allocator.alloc(i32, n);
    errdefer allocator.free(labels);
    @memset(labels, -2);

    var core_points = try allocator.alloc(bool, n);
    errdefer allocator.free(core_points);
    @memset(core_points, false);

    // Find neighbors for all points and mark core points
    var neighbors = try allocator.alloc(std.ArrayList(usize), n);
    defer {
        for (neighbors) |*neighbor_list| {
            neighbor_list.deinit();
        }
        allocator.free(neighbors);
    }

    for (0..n) |i| {
        neighbors[i] = std.ArrayList(usize).init(allocator);
    }

    // Compute neighbors (O(n²) naive approach)
    for (0..n) |i| {
        for (0..n) |j| {
            if (i == j) continue;
            const dist = euclideanDistance(T, ndim, data[i], data[j]);
            if (dist <= options.epsilon) {
                try neighbors[i].append(j);
            }
        }
        // Mark as core point if has enough neighbors
        if (neighbors[i].items.len >= options.min_pts) {
            core_points[i] = true;
        }
    }

    // Cluster core points and their density-reachable neighbors
    var cluster_id: i32 = 0;
    for (0..n) |i| {
        // Skip if already processed or not a core point
        if (labels[i] != -2 or !core_points[i]) continue;

        // Start new cluster
        try expandCluster(allocator, labels, &neighbors, core_points, i, cluster_id);
        cluster_id += 1;
    }

    // Mark remaining unvisited points as noise
    for (labels) |*label| {
        if (label.* == -2) {
            label.* = -1;
        }
    }

    // Count clusters and noise
    var n_clusters: usize = 0;
    var n_noise: usize = 0;
    for (labels) |label| {
        if (label >= 0) {
            const cid: usize = @intCast(label);
            if (cid >= n_clusters) n_clusters = cid + 1;
        } else if (label == -1) {
            n_noise += 1;
        }
    }

    return DBSCANResult(T){
        .labels = labels,
        .n_clusters = n_clusters,
        .n_noise = n_noise,
        .core_points = core_points,
    };
}

/// Expand cluster using breadth-first search from a core point
///
/// Time: O(k×m) where k = cluster size, m = average neighbors | Space: O(k)
fn expandCluster(
    allocator: Allocator,
    labels: []i32,
    neighbors: []const std.ArrayList(usize),
    core_points: []const bool,
    start_idx: usize,
    cluster_id: i32,
) !void {
    var queue = std.ArrayList(usize).init(allocator);
    defer queue.deinit();

    labels[start_idx] = cluster_id;
    try queue.append(start_idx);

    var idx: usize = 0;
    while (idx < queue.items.len) : (idx += 1) {
        const current = queue.items[idx];

        // Only core points can expand the cluster
        if (!core_points[current]) continue;

        for (neighbors[current].items) |neighbor_idx| {
            // Skip already processed points
            if (labels[neighbor_idx] != -2) continue;

            labels[neighbor_idx] = cluster_id;

            // Add to queue if core point (density-reachable)
            if (core_points[neighbor_idx]) {
                try queue.append(neighbor_idx);
            }
        }
    }
}

/// Compute Euclidean distance between two points
///
/// Time: O(d) | Space: O(1)
fn euclideanDistance(comptime T: type, comptime ndim: usize, a: [ndim]T, b: [ndim]T) T {
    var sum: T = 0.0;
    for (0..ndim) |i| {
        const diff = a[i] - b[i];
        sum += diff * diff;
    }
    return @sqrt(sum);
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "DBSCAN - basic two clusters" {
    const allocator = testing.allocator;

    // Two well-separated clusters
    const data = [_][2]f64{
        .{ 1.0, 1.0 }, .{ 1.5, 1.5 }, .{ 2.0, 2.0 }, // cluster 1
        .{ 10.0, 10.0 }, .{ 10.5, 10.5 }, .{ 11.0, 11.0 }, // cluster 2
    };

    var result = try dbscan(f64, 2, allocator, &data, .{ .epsilon = 1.0, .min_pts = 2 });
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 2), result.n_clusters);
    try testing.expectEqual(@as(usize, 0), result.n_noise);

    // Check cluster assignments (exact order depends on iteration, but should be two groups)
    const label0 = result.labels[0];
    const label3 = result.labels[3];
    try testing.expect(label0 != label3); // Different clusters
    try testing.expectEqual(label0, result.labels[1]);
    try testing.expectEqual(label0, result.labels[2]);
    try testing.expectEqual(label3, result.labels[4]);
    try testing.expectEqual(label3, result.labels[5]);
}

test "DBSCAN - noise detection" {
    const allocator = testing.allocator;

    // Two clusters with outliers
    const data = [_][2]f64{
        .{ 1.0, 1.0 }, .{ 1.5, 1.5 }, .{ 2.0, 2.0 }, // cluster 1
        .{ 10.0, 10.0 }, .{ 10.5, 10.5 }, .{ 11.0, 11.0 }, // cluster 2
        .{ 50.0, 50.0 }, // noise
        .{ -50.0, -50.0 }, // noise
    };

    var result = try dbscan(f64, 2, allocator, &data, .{ .epsilon = 1.0, .min_pts = 2 });
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 2), result.n_clusters);
    try testing.expectEqual(@as(usize, 2), result.n_noise);

    // Check noise points
    try testing.expectEqual(@as(i32, -1), result.labels[6]);
    try testing.expectEqual(@as(i32, -1), result.labels[7]);
}

test "DBSCAN - single cluster" {
    const allocator = testing.allocator;

    // Dense single cluster
    const data = [_][2]f64{
        .{ 0.0, 0.0 },
        .{ 0.5, 0.0 },
        .{ 0.0, 0.5 },
        .{ 0.5, 0.5 },
        .{ 1.0, 0.0 },
    };

    var result = try dbscan(f64, 2, allocator, &data, .{ .epsilon = 1.0, .min_pts = 2 });
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 1), result.n_clusters);
    try testing.expectEqual(@as(usize, 0), result.n_noise);

    // All points should have same label
    const label = result.labels[0];
    for (result.labels) |l| {
        try testing.expectEqual(label, l);
    }
}

test "DBSCAN - all noise" {
    const allocator = testing.allocator;

    // Sparse points, all too far apart
    const data = [_][2]f64{
        .{ 0.0, 0.0 },
        .{ 10.0, 10.0 },
        .{ 20.0, 20.0 },
        .{ 30.0, 30.0 },
    };

    var result = try dbscan(f64, 2, allocator, &data, .{ .epsilon = 1.0, .min_pts = 2 });
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), result.n_clusters);
    try testing.expectEqual(@as(usize, 4), result.n_noise);

    // All points should be noise
    for (result.labels) |label| {
        try testing.expectEqual(@as(i32, -1), label);
    }
}

test "DBSCAN - high min_pts" {
    const allocator = testing.allocator;

    const data = [_][2]f64{
        .{ 0.0, 0.0 },
        .{ 0.1, 0.0 },
        .{ 0.0, 0.1 },
    };

    // min_pts too high, no core points
    var result = try dbscan(f64, 2, allocator, &data, .{ .epsilon = 0.2, .min_pts = 5 });
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), result.n_clusters);
    try testing.expectEqual(@as(usize, 3), result.n_noise);
}

test "DBSCAN - different epsilon values" {
    const allocator = testing.allocator;

    const data = [_][2]f64{
        .{ 0.0, 0.0 },
        .{ 1.0, 0.0 },
        .{ 2.0, 0.0 },
        .{ 3.0, 0.0 },
    };

    // Small epsilon: multiple clusters
    {
        var result = try dbscan(f64, 2, allocator, &data, .{ .epsilon = 0.5, .min_pts = 1 });
        defer result.deinit(allocator);
        // Each point is its own cluster (too far apart)
        try testing.expect(result.n_clusters >= 1);
    }

    // Large epsilon: single cluster
    {
        var result = try dbscan(f64, 2, allocator, &data, .{ .epsilon = 2.0, .min_pts = 2 });
        defer result.deinit(allocator);
        try testing.expectEqual(@as(usize, 1), result.n_clusters);
    }
}

test "DBSCAN - core point identification" {
    const allocator = testing.allocator;

    const data = [_][2]f64{
        .{ 0.0, 0.0 }, // core (has 3 neighbors within epsilon)
        .{ 0.5, 0.0 }, // core
        .{ 0.0, 0.5 }, // core
        .{ 0.5, 0.5 }, // core
        .{ 2.0, 0.0 }, // border (neighbor of core but not core itself)
    };

    var result = try dbscan(f64, 2, allocator, &data, .{ .epsilon = 1.0, .min_pts = 2 });
    defer result.deinit(allocator);

    // First 4 points should be core
    try testing.expect(result.core_points[0]);
    try testing.expect(result.core_points[1]);
    try testing.expect(result.core_points[2]);
    try testing.expect(result.core_points[3]);
}

test "DBSCAN - 3D data" {
    const allocator = testing.allocator;

    const data = [_][3]f64{
        .{ 0.0, 0.0, 0.0 },
        .{ 0.5, 0.0, 0.0 },
        .{ 0.0, 0.5, 0.0 },
        .{ 0.0, 0.0, 0.5 },
        .{ 10.0, 10.0, 10.0 },
        .{ 10.5, 10.0, 10.0 },
        .{ 10.0, 10.5, 10.0 },
    };

    var result = try dbscan(f64, 3, allocator, &data, .{ .epsilon = 1.0, .min_pts = 2 });
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 2), result.n_clusters);
}

test "DBSCAN - f32 precision" {
    const allocator = testing.allocator;

    const data = [_][2]f32{
        .{ 1.0, 1.0 },
        .{ 1.5, 1.5 },
        .{ 2.0, 2.0 },
    };

    var result = try dbscan(f32, 2, allocator, &data, .{ .epsilon = 1.0, .min_pts = 2 });
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 1), result.n_clusters);
    try testing.expectEqual(@as(usize, 0), result.n_noise);
}

test "DBSCAN - getClusterSizes" {
    const allocator = testing.allocator;

    const data = [_][2]f64{
        .{ 1.0, 1.0 }, .{ 1.5, 1.5 }, // cluster 1 (2 points)
        .{ 10.0, 10.0 }, .{ 10.5, 10.5 }, .{ 11.0, 11.0 }, // cluster 2 (3 points)
        .{ 50.0, 50.0 }, // noise
    };

    var result = try dbscan(f64, 2, allocator, &data, .{ .epsilon = 1.0, .min_pts = 2 });
    defer result.deinit(allocator);

    const sizes = try result.getClusterSizes(allocator);
    defer allocator.free(sizes);

    try testing.expectEqual(@as(usize, 2), sizes.len);
    // One cluster has 2 points, the other has 3
    const total = sizes[0] + sizes[1];
    try testing.expectEqual(@as(usize, 5), total);
}

test "DBSCAN - getClusterIndices" {
    const allocator = testing.allocator;

    const data = [_][2]f64{
        .{ 1.0, 1.0 }, .{ 1.5, 1.5 }, .{ 2.0, 2.0 },
        .{ 10.0, 10.0 }, .{ 10.5, 10.5 },
    };

    var result = try dbscan(f64, 2, allocator, &data, .{ .epsilon = 1.0, .min_pts = 2 });
    defer result.deinit(allocator);

    // Get indices for first cluster (whichever cluster point 0 is in)
    const cluster_id: usize = @intCast(result.labels[0]);
    var indices = try result.getClusterIndices(allocator, cluster_id);
    defer indices.deinit();

    try testing.expectEqual(@as(usize, 3), indices.items.len);
}

test "DBSCAN - getNoiseIndices" {
    const allocator = testing.allocator;

    const data = [_][2]f64{
        .{ 1.0, 1.0 }, .{ 1.5, 1.5 }, .{ 2.0, 2.0 },
        .{ 50.0, 50.0 }, .{ -50.0, -50.0 },
    };

    var result = try dbscan(f64, 2, allocator, &data, .{ .epsilon = 1.0, .min_pts = 2 });
    defer result.deinit(allocator);

    var noise = try result.getNoiseIndices(allocator);
    defer noise.deinit();

    try testing.expectEqual(@as(usize, 2), noise.items.len);
    try testing.expectEqual(@as(usize, 3), noise.items[0]);
    try testing.expectEqual(@as(usize, 4), noise.items[1]);
}

test "DBSCAN - empty data" {
    const allocator = testing.allocator;

    const data = [_][2]f64{};

    var result = try dbscan(f64, 2, allocator, &data, .{ .epsilon = 1.0, .min_pts = 2 });
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), result.n_clusters);
    try testing.expectEqual(@as(usize, 0), result.n_noise);
}

test "DBSCAN - single point" {
    const allocator = testing.allocator;

    const data = [_][2]f64{.{ 0.0, 0.0 }};

    var result = try dbscan(f64, 2, allocator, &data, .{ .epsilon = 1.0, .min_pts = 2 });
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), result.n_clusters);
    try testing.expectEqual(@as(usize, 1), result.n_noise);
    try testing.expectEqual(@as(i32, -1), result.labels[0]);
}

test "DBSCAN - memory safety with testing.allocator" {
    const allocator = testing.allocator;

    const data = [_][2]f64{
        .{ 0.0, 0.0 }, .{ 1.0, 1.0 }, .{ 2.0, 2.0 },
        .{ 10.0, 10.0 }, .{ 11.0, 11.0 },
    };

    var result = try dbscan(f64, 2, allocator, &data, .{ .epsilon = 2.0, .min_pts = 2 });
    defer result.deinit(allocator);

    const sizes = try result.getClusterSizes(allocator);
    defer allocator.free(sizes);

    var indices = try result.getClusterIndices(allocator, 0);
    defer indices.deinit();

    var noise = try result.getNoiseIndices(allocator);
    defer noise.deinit();

    // If we reach here without leaks, memory safety is verified
}
