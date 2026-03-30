//! Mean Shift Clustering
//!
//! Non-parametric clustering algorithm that doesn't require specifying number of clusters.
//! Iteratively shifts points toward regions of highest density (modes of the distribution).
//!
//! Algorithm:
//! 1. For each point, compute mean shift vector (weighted mean of neighbors within bandwidth)
//! 2. Shift point toward mean until convergence (reaches local density maximum)
//! 3. Points converging to same mode belong to same cluster
//!
//! Time complexity: O(n² × iterations × d) — quadratic in number of points
//! Space complexity: O(n × d) — stores shifted points and labels
//!
//! Key parameters:
//! - bandwidth: Size of search window (larger = fewer, larger clusters)
//! - max_iter: Maximum iterations for convergence (default: 300)
//! - tolerance: Convergence threshold (default: 1e-3)
//!
//! Use cases:
//! - Computer vision: image segmentation, object tracking, blob detection
//! - Mode finding: peak detection in density distributions
//! - Clustering without knowing K: automatic cluster discovery
//! - Non-convex clusters: handles arbitrary shapes like DBSCAN
//!
//! Trade-offs vs other clustering:
//! - vs K-Means: No K needed, finds arbitrary shapes, but O(n²) slower
//! - vs DBSCAN: Smoother clusters (no hard density threshold), but computationally expensive
//! - vs Hierarchical: Automatic cluster count, but doesn't provide hierarchy

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Mean Shift clustering result
pub fn MeanShiftResult(comptime T: type) type {
    return struct {
        /// Cluster labels for each point (-1 for outliers if enabled)
        labels: []i32,
        /// Cluster centers (converged modes)
        centers: [][]const T,
        /// Number of clusters found
        n_clusters: usize,
        
        allocator: Allocator,
        
        const Self = @This();
        
        /// Free allocated memory
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.labels);
            for (self.centers) |center| {
                self.allocator.free(center);
            }
            self.allocator.free(self.centers);
        }
    };
}

/// Mean Shift configuration options
pub const MeanShiftOptions = struct {
    /// Bandwidth (search radius) — larger values produce fewer, larger clusters
    bandwidth: f64 = 1.0,
    
    /// Maximum iterations for convergence
    max_iter: usize = 300,
    
    /// Convergence tolerance (Euclidean distance threshold)
    tolerance: f64 = 1e-3,
    
    /// Random seed for reproducibility (unused in deterministic mean shift)
    seed: ?u64 = null,
    
    /// Minimum cluster size (points in clusters smaller than this are marked as outliers)
    min_cluster_size: usize = 1,
};

/// Perform Mean Shift clustering
///
/// Time: O(n² × iterations × d) where n = points, d = dimensions
/// Space: O(n × d) for shifted points
///
/// Example:
/// ```zig
/// const data = [_][2]f64{ .{0,0}, .{0.5,0.5}, .{10,10}, .{10.5,10.5} };
/// const points = std.mem.sliceAsBytes(&data);
/// const opts = MeanShiftOptions{ .bandwidth = 2.0 };
/// var result = try meanShift(f64, allocator, points, 2, opts);
/// defer result.deinit();
/// // result.labels contains cluster assignments
/// // result.centers contains final cluster centers
/// ```
pub fn meanShift(
    comptime T: type,
    allocator: Allocator,
    data: []const u8,
    n_features: usize,
    options: MeanShiftOptions,
) !MeanShiftResult(T) {
    const n_samples = data.len / (@sizeOf(T) * n_features);
    if (n_samples == 0) return error.EmptyData;
    
    // Reinterpret data as 2D array
    const points_flat = std.mem.bytesAsSlice(T, data);
    
    // Allocate shifted points (start at original positions)
    const shifted = try allocator.alloc(T, n_samples * n_features);
    defer allocator.free(shifted);
    @memcpy(shifted, points_flat);
    
    // Iteratively shift each point toward local density maximum
    for (0..n_samples) |i| {
        const point_start = i * n_features;
        var converged = false;
        var iter: usize = 0;
        
        while (!converged and iter < options.max_iter) : (iter += 1) {
            // Compute weighted mean of neighbors within bandwidth
            var mean = try allocator.alloc(T, n_features);
            defer allocator.free(mean);
            @memset(mean, 0);
            
            var weight_sum: T = 0;
            
            for (0..n_samples) |j| {
                const neighbor_start = j * n_features;
                const dist = euclideanDistance(T, shifted[point_start..point_start+n_features], points_flat[neighbor_start..neighbor_start+n_features]);
                
                if (dist <= options.bandwidth) {
                    // Gaussian kernel (can also use flat kernel)
                    const weight = gaussianKernel(T, dist, options.bandwidth);
                    weight_sum += weight;
                    
                    for (0..n_features) |k| {
                        mean[k] += weight * points_flat[neighbor_start + k];
                    }
                }
            }
            
            // Normalize by total weight
            if (weight_sum > 0) {
                for (mean) |*val| {
                    val.* /= weight_sum;
                }
            }
            
            // Compute shift distance
            const shift_dist = euclideanDistance(T, shifted[point_start..point_start+n_features], mean);
            
            // Update position
            @memcpy(shifted[point_start..point_start+n_features], mean);
            
            // Check convergence
            if (shift_dist < options.tolerance) {
                converged = true;
            }
        }
    }
    
    // Merge nearby converged points into clusters
    var labels = try allocator.alloc(i32, n_samples);
    @memset(labels, -1);
    
    var centers_list = std.ArrayList([]T).init(allocator);
    defer {
        for (centers_list.items) |center| allocator.free(center);
        centers_list.deinit();
    }
    
    var cluster_id: i32 = 0;
    
    for (0..n_samples) |i| {
        if (labels[i] != -1) continue; // Already assigned
        
        // Create new cluster
        const center = try allocator.alloc(T, n_features);
        @memcpy(center, shifted[i * n_features..(i + 1) * n_features]);
        try centers_list.append(center);
        
        // Assign all nearby points to this cluster
        var cluster_size: usize = 0;
        for (0..n_samples) |j| {
            const dist = euclideanDistance(T, shifted[j * n_features..(j + 1) * n_features], center);
            if (dist < options.tolerance * 2) { // Merge threshold
                labels[j] = cluster_id;
                cluster_size += 1;
            }
        }
        
        // Mark small clusters as outliers if min_cluster_size > 1
        if (cluster_size < options.min_cluster_size) {
            for (0..n_samples) |j| {
                if (labels[j] == cluster_id) {
                    labels[j] = -1;
                }
            }
            _ = centers_list.pop();
            allocator.free(center);
        } else {
            cluster_id += 1;
        }
    }
    
    // Convert centers to const slices
    const centers = try allocator.alloc([]const T, centers_list.items.len);
    for (centers_list.items, 0..) |center, i| {
        centers[i] = center;
    }
    
    return MeanShiftResult(T){
        .labels = labels,
        .centers = centers,
        .n_clusters = centers.len,
        .allocator = allocator,
    };
}

/// Euclidean distance between two points
fn euclideanDistance(comptime T: type, a: []const T, b: []const T) T {
    var sum: T = 0;
    for (a, b) |ai, bi| {
        const diff = ai - bi;
        sum += diff * diff;
    }
    return @sqrt(sum);
}

/// Gaussian kernel function
fn gaussianKernel(comptime T: type, distance: T, bandwidth: T) T {
    const x = distance / bandwidth;
    return @exp(-0.5 * x * x);
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "mean shift: basic two clusters" {
    const allocator = testing.allocator;
    
    // Two well-separated clusters
    const data = [_][2]f64{
        .{ 0.0, 0.0 }, .{ 0.5, 0.5 }, .{ 0.3, 0.2 }, // Cluster 1
        .{ 10.0, 10.0 }, .{ 10.5, 10.5 }, .{ 10.2, 10.3 }, // Cluster 2
    };
    const bytes = std.mem.sliceAsBytes(&data);
    
    const opts = MeanShiftOptions{ .bandwidth = 1.5 };
    var result = try meanShift(f64, allocator, bytes, 2, opts);
    defer result.deinit();
    
    try testing.expectEqual(@as(usize, 2), result.n_clusters);
    
    // Check that points in same physical cluster have same label
    try testing.expectEqual(result.labels[0], result.labels[1]);
    try testing.expectEqual(result.labels[0], result.labels[2]);
    try testing.expectEqual(result.labels[3], result.labels[4]);
    try testing.expectEqual(result.labels[3], result.labels[5]);
    
    // Different clusters should have different labels
    try testing.expect(result.labels[0] != result.labels[3]);
}

test "mean shift: single cluster" {
    const allocator = testing.allocator;
    
    const data = [_][2]f64{
        .{ 0.0, 0.0 }, .{ 0.1, 0.1 }, .{ 0.2, 0.0 }, .{ 0.0, 0.2 },
    };
    const bytes = std.mem.sliceAsBytes(&data);
    
    const opts = MeanShiftOptions{ .bandwidth = 0.5 };
    var result = try meanShift(f64, allocator, bytes, 2, opts);
    defer result.deinit();
    
    try testing.expectEqual(@as(usize, 1), result.n_clusters);
    
    // All points should have same label
    for (result.labels) |label| {
        try testing.expectEqual(@as(i32, 0), label);
    }
}

test "mean shift: three clusters" {
    const allocator = testing.allocator;
    
    const data = [_][2]f64{
        .{ 0.0, 0.0 }, .{ 0.5, 0.5 },       // Cluster 1
        .{ 10.0, 10.0 }, .{ 10.5, 10.5 },   // Cluster 2
        .{ 20.0, 20.0 }, .{ 20.5, 20.5 },   // Cluster 3
    };
    const bytes = std.mem.sliceAsBytes(&data);
    
    const opts = MeanShiftOptions{ .bandwidth = 1.5 };
    var result = try meanShift(f64, allocator, bytes, 2, opts);
    defer result.deinit();
    
    try testing.expectEqual(@as(usize, 3), result.n_clusters);
}

test "mean shift: f32 support" {
    const allocator = testing.allocator;
    
    const data = [_][2]f32{
        .{ 0.0, 0.0 }, .{ 0.5, 0.5 },
        .{ 10.0, 10.0 }, .{ 10.5, 10.5 },
    };
    const bytes = std.mem.sliceAsBytes(&data);
    
    const opts = MeanShiftOptions{ .bandwidth = 1.5 };
    var result = try meanShift(f32, allocator, bytes, 2, opts);
    defer result.deinit();
    
    try testing.expectEqual(@as(usize, 2), result.n_clusters);
}

test "mean shift: higher dimensions" {
    const allocator = testing.allocator;
    
    const data = [_][3]f64{
        .{ 0.0, 0.0, 0.0 }, .{ 0.5, 0.5, 0.5 },
        .{ 10.0, 10.0, 10.0 }, .{ 10.5, 10.5, 10.5 },
    };
    const bytes = std.mem.sliceAsBytes(&data);
    
    const opts = MeanShiftOptions{ .bandwidth = 1.5 };
    var result = try meanShift(f64, allocator, bytes, 3, opts);
    defer result.deinit();
    
    try testing.expectEqual(@as(usize, 2), result.n_clusters);
}

test "mean shift: min cluster size filtering" {
    const allocator = testing.allocator;
    
    const data = [_][2]f64{
        .{ 0.0, 0.0 }, .{ 0.5, 0.5 }, .{ 0.3, 0.2 }, // Large cluster
        .{ 20.0, 20.0 }, // Outlier (single point)
    };
    const bytes = std.mem.sliceAsBytes(&data);
    
    const opts = MeanShiftOptions{ .bandwidth = 1.5, .min_cluster_size = 2 };
    var result = try meanShift(f64, allocator, bytes, 2, opts);
    defer result.deinit();
    
    // Should have 1 cluster (outlier filtered)
    try testing.expectEqual(@as(usize, 1), result.n_clusters);
    
    // Outlier should be labeled -1
    try testing.expectEqual(@as(i32, -1), result.labels[3]);
}

test "mean shift: convergence with custom tolerance" {
    const allocator = testing.allocator;
    
    const data = [_][2]f64{
        .{ 0.0, 0.0 }, .{ 0.1, 0.1 },
    };
    const bytes = std.mem.sliceAsBytes(&data);
    
    const opts = MeanShiftOptions{ .bandwidth = 0.5, .tolerance = 0.01 };
    var result = try meanShift(f64, allocator, bytes, 2, opts);
    defer result.deinit();
    
    try testing.expectEqual(@as(usize, 1), result.n_clusters);
}

test "mean shift: bandwidth effect" {
    const allocator = testing.allocator;
    
    const data = [_][2]f64{
        .{ 0.0, 0.0 }, .{ 1.0, 1.0 }, .{ 2.0, 2.0 },
    };
    const bytes = std.mem.sliceAsBytes(&data);
    
    // Small bandwidth → more clusters
    const opts_small = MeanShiftOptions{ .bandwidth = 0.5 };
    var result_small = try meanShift(f64, allocator, bytes, 2, opts_small);
    defer result_small.deinit();
    
    // Large bandwidth → fewer clusters
    const opts_large = MeanShiftOptions{ .bandwidth = 5.0 };
    var result_large = try meanShift(f64, allocator, bytes, 2, opts_large);
    defer result_large.deinit();
    
    try testing.expect(result_small.n_clusters >= result_large.n_clusters);
}

test "mean shift: non-convex clusters" {
    const allocator = testing.allocator;
    
    // Two crescent-shaped clusters (non-convex)
    const data = [_][2]f64{
        .{ 0.0, 0.0 }, .{ 1.0, 0.5 }, .{ 2.0, 0.0 }, // Crescent 1
        .{ 10.0, 10.0 }, .{ 11.0, 10.5 }, .{ 12.0, 10.0 }, // Crescent 2
    };
    const bytes = std.mem.sliceAsBytes(&data);
    
    const opts = MeanShiftOptions{ .bandwidth = 1.5 };
    var result = try meanShift(f64, allocator, bytes, 2, opts);
    defer result.deinit();
    
    // Should still find 2 clusters despite non-convex shape
    try testing.expectEqual(@as(usize, 2), result.n_clusters);
}

test "mean shift: memory safety" {
    const allocator = testing.allocator;
    
    const data = [_][2]f64{
        .{ 0.0, 0.0 }, .{ 10.0, 10.0 },
    };
    const bytes = std.mem.sliceAsBytes(&data);
    
    const opts = MeanShiftOptions{ .bandwidth = 1.0 };
    var result = try meanShift(f64, allocator, bytes, 2, opts);
    result.deinit();
    
    // No memory leaks expected
    try testing.expect(true);
}

test "mean shift: large dataset" {
    const allocator = testing.allocator;
    
    // 100 points in 3 clusters
    var data = std.ArrayList([2]f64).init(allocator);
    defer data.deinit();
    
    // Cluster 1: around (0, 0)
    for (0..33) |i| {
        const x = @as(f64, @floatFromInt(i)) * 0.1;
        try data.append(.{ x, x * 0.5 });
    }
    
    // Cluster 2: around (10, 10)
    for (0..33) |i| {
        const x = 10.0 + @as(f64, @floatFromInt(i)) * 0.1;
        try data.append(.{ x, 10.0 + x * 0.05 });
    }
    
    // Cluster 3: around (20, 20)
    for (0..34) |i| {
        const x = 20.0 + @as(f64, @floatFromInt(i)) * 0.1;
        try data.append(.{ x, 20.0 + x * 0.05 });
    }
    
    const bytes = std.mem.sliceAsBytes(data.items);
    
    const opts = MeanShiftOptions{ .bandwidth = 2.0 };
    var result = try meanShift(f64, allocator, bytes, 2, opts);
    defer result.deinit();
    
    try testing.expectEqual(@as(usize, 3), result.n_clusters);
}

test "mean shift: empty data error" {
    const allocator = testing.allocator;
    
    const data = [_][2]f64{};
    const bytes = std.mem.sliceAsBytes(&data);
    
    const opts = MeanShiftOptions{};
    const result = meanShift(f64, allocator, bytes, 2, opts);
    
    try testing.expectError(error.EmptyData, result);
}
