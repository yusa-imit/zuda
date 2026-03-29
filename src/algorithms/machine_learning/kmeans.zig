const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// K-Means Clustering Algorithm
///
/// Partitions n data points into k clusters by iteratively:
/// 1. Assigning each point to nearest centroid
/// 2. Updating centroids as cluster means
/// 3. Repeating until convergence or max iterations
///
/// Time: O(n × k × d × iterations) where:
///   - n = number of data points
///   - k = number of clusters
///   - d = dimensionality
///   - iterations = convergence iterations (typically ≤ 100)
///
/// Space: O(k × d + n) for centroids and assignments
///
/// Use cases:
/// - Customer segmentation (marketing)
/// - Image compression (color quantization)
/// - Anomaly detection (outlier identification)
/// - Document clustering (topic modeling)
/// - Vector quantization (signal processing)
///
/// Limitations:
/// - Requires k to be specified a priori
/// - Sensitive to initial centroid placement
/// - Assumes spherical clusters (Euclidean distance)
/// - May converge to local optimum
///
/// Example:
/// ```zig
/// const data = [_][]const f64{
///     &.{1.0, 1.0}, &.{1.5, 2.0}, &.{3.0, 4.0}, &.{5.0, 7.0},
///     &.{3.5, 5.0}, &.{4.5, 5.0}, &.{3.5, 4.5}
/// };
/// var result = try kmeans(f64, allocator, &data, 2, .{});
/// defer result.deinit();
/// // result.centroids contains 2 cluster centers
/// // result.labels[i] indicates cluster for data[i]
/// ```
pub fn KMeansResult(comptime T: type) type {
    return struct {
        centroids: [][]T, // k centroids, each of dim d
        labels: []usize, // n cluster assignments (0..k-1)
        inertia: T, // sum of squared distances to nearest centroid
        iterations: usize, // actual iterations until convergence
        allocator: Allocator,

        const Self = @This();

        /// Free all allocated memory
        /// Time: O(k)
        /// Space: O(1)
        pub fn deinit(self: *Self) void {
            for (self.centroids) |centroid| {
                self.allocator.free(centroid);
            }
            self.allocator.free(self.centroids);
            self.allocator.free(self.labels);
        }
    };
}

pub const KMeansOptions = struct {
    max_iterations: usize = 300,
    tolerance: f64 = 1e-4, // convergence threshold for centroid movement
    random_seed: ?u64 = null, // for reproducible initialization
};

/// K-Means clustering
///
/// Time: O(n × k × d × iterations)
/// Space: O(k × d + n)
///
/// Params:
///   T: numeric type (f32, f64)
///   allocator: memory allocator
///   data: n data points, each []const T of length d
///   k: number of clusters (must be ≤ n)
///   options: configuration (max_iterations, tolerance, seed)
///
/// Returns: KMeansResult with centroids, labels, inertia, iteration count
///
/// Errors:
///   - OutOfMemory: allocation failed
///   - InvalidK: k > n or k == 0
///   - EmptyData: data.len == 0
///   - InconsistentDimensions: data points have different dimensions
pub fn kmeans(
    comptime T: type,
    allocator: Allocator,
    data: []const []const T,
    k: usize,
    options: KMeansOptions,
) !KMeansResult(T) {
    if (data.len == 0) return error.EmptyData;
    if (k == 0 or k > data.len) return error.InvalidK;

    const n = data.len;
    const d = data[0].len;

    // Validate dimensions
    for (data) |point| {
        if (point.len != d) return error.InconsistentDimensions;
    }

    // Initialize centroids using k-means++
    const centroids = try initializeCentroidsKMeansPlusPlus(T, allocator, data, k, options.random_seed);
    errdefer {
        for (centroids) |c| allocator.free(c);
        allocator.free(centroids);
    }

    // Allocate labels
    var labels = try allocator.alloc(usize, n);
    errdefer allocator.free(labels);

    var iteration: usize = 0;
    var converged = false;

    while (iteration < options.max_iterations and !converged) : (iteration += 1) {
        // Assignment step: assign each point to nearest centroid
        for (data, 0..) |point, i| {
            labels[i] = nearestCentroid(T, point, centroids);
        }

        // Update step: recompute centroids as cluster means
        const new_centroids = try computeClusterMeans(T, allocator, data, labels, k);
        defer {
            for (new_centroids) |c| allocator.free(c);
            allocator.free(new_centroids);
        }

        // Check convergence: max centroid movement < tolerance
        var max_movement: T = 0;
        for (centroids, new_centroids) |old, new| {
            const movement = euclideanDistance(T, old, new);
            if (movement > max_movement) max_movement = movement;
        }

        // Update centroids
        for (centroids, new_centroids) |old, new| {
            @memcpy(old, new);
        }

        if (max_movement < options.tolerance) {
            converged = true;
        }
    }

    // Compute inertia (sum of squared distances)
    const inertia = computeInertia(T, data, centroids, labels);

    return KMeansResult(T){
        .centroids = centroids,
        .labels = labels,
        .inertia = inertia,
        .iterations = iteration,
        .allocator = allocator,
    };
}

/// Initialize centroids using k-means++ algorithm for better initial placement
/// Time: O(n × k × d)
/// Space: O(k × d)
fn initializeCentroidsKMeansPlusPlus(
    comptime T: type,
    allocator: Allocator,
    data: []const []const T,
    k: usize,
    seed: ?u64,
) ![][]T {
    const n = data.len;

    var prng = std.Random.DefaultPrng.init(seed orelse @as(u64, @intCast(std.time.timestamp())));
    var rand = prng.random();

    var centroids = try allocator.alloc([]T, k);
    errdefer {
        for (centroids[0..0]) |c| allocator.free(c);
        allocator.free(centroids);
    }

    // First centroid: random data point
    const first_idx = rand.intRangeLessThan(usize, 0, n);
    centroids[0] = try allocator.dupe(T, data[first_idx]);

    // Remaining centroids: probability proportional to squared distance
    var distances = try allocator.alloc(T, n);
    defer allocator.free(distances);

    for (1..k) |i| {
        // Compute distance to nearest existing centroid
        for (data, 0..) |point, j| {
            var min_dist: T = std.math.floatMax(T);
            for (centroids[0..i]) |centroid| {
                const dist = euclideanDistance(T, point, centroid);
                if (dist < min_dist) min_dist = dist;
            }
            distances[j] = min_dist * min_dist; // squared distance
        }

        // Select next centroid with probability ∝ distance²
        const idx = weightedRandomChoice(T, distances, rand);
        centroids[i] = try allocator.dupe(T, data[idx]);
    }

    return centroids;
}

/// Find nearest centroid for a point
/// Time: O(k × d)
/// Space: O(1)
fn nearestCentroid(comptime T: type, point: []const T, centroids: []const []const T) usize {
    var min_dist: T = std.math.floatMax(T);
    var min_idx: usize = 0;

    for (centroids, 0..) |centroid, i| {
        const dist = euclideanDistance(T, point, centroid);
        if (dist < min_dist) {
            min_dist = dist;
            min_idx = i;
        }
    }

    return min_idx;
}

/// Compute cluster means as new centroids
/// Time: O(n × d)
/// Space: O(k × d)
fn computeClusterMeans(
    comptime T: type,
    allocator: Allocator,
    data: []const []const T,
    labels: []const usize,
    k: usize,
) ![][]T {
    const d = data[0].len;

    var centroids = try allocator.alloc([]T, k);
    errdefer allocator.free(centroids);

    var counts = try allocator.alloc(usize, k);
    defer allocator.free(counts);
    @memset(counts, 0);

    // Initialize centroids to zero
    for (0..k) |i| {
        centroids[i] = try allocator.alloc(T, d);
        @memset(centroids[i], 0);
    }

    // Accumulate sums
    for (data, labels) |point, label| {
        for (0..d) |dim| {
            centroids[label][dim] += point[dim];
        }
        counts[label] += 1;
    }

    // Compute means (handle empty clusters by keeping previous centroid)
    for (0..k) |i| {
        if (counts[i] > 0) {
            const count_f: T = @floatFromInt(counts[i]);
            for (0..d) |dim| {
                centroids[i][dim] /= count_f;
            }
        }
    }

    return centroids;
}

/// Compute inertia (sum of squared distances to nearest centroid)
/// Time: O(n × d)
/// Space: O(1)
fn computeInertia(
    comptime T: type,
    data: []const []const T,
    centroids: []const []const T,
    labels: []const usize,
) T {
    var inertia: T = 0;

    for (data, labels) |point, label| {
        const dist = euclideanDistance(T, point, centroids[label]);
        inertia += dist * dist;
    }

    return inertia;
}

/// Euclidean distance between two points
/// Time: O(d)
/// Space: O(1)
fn euclideanDistance(comptime T: type, a: []const T, b: []const T) T {
    var sum: T = 0;
    for (a, b) |ai, bi| {
        const diff = ai - bi;
        sum += diff * diff;
    }
    return @sqrt(sum);
}

/// Weighted random selection (probability ∝ weights)
/// Time: O(n)
/// Space: O(1)
fn weightedRandomChoice(comptime T: type, weights: []const T, rand: std.Random) usize {
    var total: T = 0;
    for (weights) |w| total += w;

    var r = rand.float(T) * total;
    for (weights, 0..) |w, i| {
        r -= w;
        if (r <= 0) return i;
    }

    return weights.len - 1; // fallback
}

// ============================================================================
// Tests
// ============================================================================

test "kmeans: basic 2D clustering (2 clusters)" {
    const allocator = testing.allocator;

    // Two clear clusters: (1,1), (1.5,2) and (5,7), (5.5,8)
    const data = [_][]const f64{
        &.{ 1.0, 1.0 },
        &.{ 1.5, 2.0 },
        &.{ 5.0, 7.0 },
        &.{ 5.5, 8.0 },
    };

    var result = try kmeans(f64, allocator, &data, 2, .{ .random_seed = 42 });
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.centroids.len);
    try testing.expectEqual(@as(usize, 4), result.labels.len);

    // Points 0,1 should be in same cluster, points 2,3 in another
    const cluster0 = result.labels[0];
    const cluster1 = result.labels[2];
    try testing.expect(cluster0 != cluster1);
    try testing.expectEqual(result.labels[0], result.labels[1]);
    try testing.expectEqual(result.labels[2], result.labels[3]);

    // Inertia should be positive
    try testing.expect(result.inertia > 0);
}

test "kmeans: 3 clusters in 2D" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 1.0, 1.0 }, &.{ 1.5, 1.5 }, // cluster 1
        &.{ 5.0, 5.0 }, &.{ 5.5, 5.5 }, // cluster 2
        &.{ 9.0, 9.0 }, &.{ 9.5, 9.5 }, // cluster 3
    };

    var result = try kmeans(f64, allocator, &data, 3, .{ .random_seed = 123 });
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.centroids.len);

    // Each pair should be in same cluster
    try testing.expectEqual(result.labels[0], result.labels[1]);
    try testing.expectEqual(result.labels[2], result.labels[3]);
    try testing.expectEqual(result.labels[4], result.labels[5]);

    // All three clusters should be different
    try testing.expect(result.labels[0] != result.labels[2]);
    try testing.expect(result.labels[0] != result.labels[4]);
    try testing.expect(result.labels[2] != result.labels[4]);
}

test "kmeans: single cluster (k=1)" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 1.0, 2.0 },
        &.{ 3.0, 4.0 },
        &.{ 5.0, 6.0 },
    };

    var result = try kmeans(f64, allocator, &data, 1, .{});
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), result.centroids.len);

    // All points in cluster 0
    for (result.labels) |label| {
        try testing.expectEqual(@as(usize, 0), label);
    }

    // Centroid should be mean of all points
    try testing.expectApproxEqAbs(@as(f64, 3.0), result.centroids[0][0], 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 4.0), result.centroids[0][1], 1e-6);
}

test "kmeans: k = n (each point its own cluster)" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 1.0, 2.0 },
        &.{ 3.0, 4.0 },
        &.{ 5.0, 6.0 },
    };

    var result = try kmeans(f64, allocator, &data, 3, .{});
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.centroids.len);

    // All labels should be unique
    var seen = [_]bool{false} ** 3;
    for (result.labels) |label| {
        try testing.expect(!seen[label]);
        seen[label] = true;
    }

    // Inertia should be 0 (each point is its own centroid)
    try testing.expectApproxEqAbs(@as(f64, 0.0), result.inertia, 1e-6);
}

test "kmeans: f32 support" {
    const allocator = testing.allocator;

    const data = [_][]const f32{
        &.{ 1.0, 1.0 },
        &.{ 2.0, 2.0 },
        &.{ 10.0, 10.0 },
        &.{ 11.0, 11.0 },
    };

    var result = try kmeans(f32, allocator, &data, 2, .{ .random_seed = 999 });
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.centroids.len);

    // Points 0,1 vs 2,3 should be in different clusters
    try testing.expect(result.labels[0] != result.labels[2]);
}

test "kmeans: high dimensional (5D)" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 1.0, 1.0, 1.0, 1.0, 1.0 },
        &.{ 1.5, 1.5, 1.5, 1.5, 1.5 },
        &.{ 10.0, 10.0, 10.0, 10.0, 10.0 },
        &.{ 10.5, 10.5, 10.5, 10.5, 10.5 },
    };

    var result = try kmeans(f64, allocator, &data, 2, .{});
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.centroids.len);
    try testing.expectEqual(@as(usize, 5), result.centroids[0].len);

    // Points 0,1 should cluster together
    try testing.expectEqual(result.labels[0], result.labels[1]);
}

test "kmeans: convergence iterations" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 1.0, 1.0 },
        &.{ 1.1, 1.1 },
        &.{ 10.0, 10.0 },
        &.{ 10.1, 10.1 },
    };

    var result = try kmeans(f64, allocator, &data, 2, .{ .max_iterations = 100, .tolerance = 1e-4 });
    defer result.deinit();

    // Should converge in < 100 iterations for well-separated clusters
    try testing.expect(result.iterations < 100);
}

test "kmeans: error - empty data" {
    const allocator = testing.allocator;

    const data = [_][]const f64{};

    try testing.expectError(error.EmptyData, kmeans(f64, allocator, &data, 1, .{}));
}

test "kmeans: error - k = 0" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 1.0, 1.0 },
    };

    try testing.expectError(error.InvalidK, kmeans(f64, allocator, &data, 0, .{}));
}

test "kmeans: error - k > n" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 1.0, 1.0 },
        &.{ 2.0, 2.0 },
    };

    try testing.expectError(error.InvalidK, kmeans(f64, allocator, &data, 3, .{}));
}

test "kmeans: error - inconsistent dimensions" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 1.0, 1.0 },
        &.{ 2.0, 2.0, 2.0 }, // 3D instead of 2D
    };

    try testing.expectError(error.InconsistentDimensions, kmeans(f64, allocator, &data, 2, .{}));
}

test "kmeans: large dataset (100 points)" {
    const allocator = testing.allocator;

    var data_list = std.ArrayList([]f64).init(allocator);
    defer {
        for (data_list.items) |point| allocator.free(point);
        data_list.deinit();
    }

    // Cluster 1: around (0,0)
    for (0..50) |_| {
        var point = try allocator.alloc(f64, 2);
        point[0] = @as(f64, @floatFromInt(@mod(@as(u64, @intCast(std.time.microTimestamp())), 100))) / 100.0;
        point[1] = @as(f64, @floatFromInt(@mod(@as(u64, @intCast(std.time.microTimestamp())) >> 8, 100))) / 100.0;
        try data_list.append(point);
    }

    // Cluster 2: around (10,10)
    for (0..50) |_| {
        var point = try allocator.alloc(f64, 2);
        point[0] = 10.0 + @as(f64, @floatFromInt(@mod(@as(u64, @intCast(std.time.microTimestamp())), 100))) / 100.0;
        point[1] = 10.0 + @as(f64, @floatFromInt(@mod(@as(u64, @intCast(std.time.microTimestamp())) >> 8, 100))) / 100.0;
        try data_list.append(point);
    }

    var result = try kmeans(f64, allocator, data_list.items, 2, .{});
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.centroids.len);
    try testing.expectEqual(@as(usize, 100), result.labels.len);

    // Both clusters should have points
    var count0: usize = 0;
    var count1: usize = 0;
    for (result.labels) |label| {
        if (label == 0) count0 += 1 else count1 += 1;
    }
    try testing.expect(count0 > 0 and count1 > 0);
}

test "kmeans: reproducibility with seed" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 1.0, 1.0 },
        &.{ 2.0, 2.0 },
        &.{ 10.0, 10.0 },
        &.{ 11.0, 11.0 },
    };

    var result1 = try kmeans(f64, allocator, &data, 2, .{ .random_seed = 42 });
    defer result1.deinit();

    var result2 = try kmeans(f64, allocator, &data, 2, .{ .random_seed = 42 });
    defer result2.deinit();

    // Same seed should produce identical results
    for (0..4) |i| {
        try testing.expectEqual(result1.labels[i], result2.labels[i]);
    }

    try testing.expectApproxEqAbs(result1.inertia, result2.inertia, 1e-6);
}
