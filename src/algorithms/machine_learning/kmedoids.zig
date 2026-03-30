/// K-Medoids (PAM - Partitioning Around Medoids) Clustering
///
/// K-Medoids is a clustering algorithm similar to K-Means but uses actual data points
/// (medoids) as cluster centers instead of means. This makes it more robust to outliers
/// and noise, and it works with any distance metric (not limited to Euclidean distance).
///
/// Algorithm: PAM (Partitioning Around Medoids)
/// 1. BUILD phase: Initialize k medoids using k-means++ style selection
/// 2. SWAP phase: Iteratively swap medoids with non-medoids to minimize total cost
/// 3. ASSIGN phase: Assign each point to nearest medoid
///
/// Time: O(k(n-k)²×iter) where n = points, k = clusters, iter = iterations
/// Space: O(n) for assignments and distances
///
/// Properties:
/// - More robust to outliers than K-Means (uses actual data points)
/// - Works with any distance metric (not just Euclidean)
/// - Medoids are interpretable (actual data points)
/// - More expensive than K-Means computationally
///
/// Use cases:
/// - Clustering with non-Euclidean distances (e.g., Manhattan, Cosine)
/// - Robust clustering with outliers
/// - When cluster centers must be actual data points (e.g., selecting representative images)
/// - Time series clustering with DTW distance
/// - Text clustering with edit distance

const std = @import("std");
const Allocator = std.mem.Allocator;
const Random = std.rand.Random;

/// Distance function type: (point_a, point_b, dimensions) -> distance
pub const DistanceFn = *const fn ([]const f64, []const f64, usize) f64;

/// K-Medoids configuration options
pub const KMedoidsOptions = struct {
    /// Maximum number of iterations (default: 300)
    max_iterations: usize = 300,
    /// Random seed for reproducibility (default: 42)
    random_seed: u64 = 42,
    /// Distance function (default: Euclidean)
    distance_fn: DistanceFn = euclideanDistance,
};

/// K-Medoids clustering result
pub const KMedoidsResult = struct {
    /// Medoid indices (k medoids, actual data point indices)
    medoids: []usize,
    /// Cluster assignments for each point
    labels: []usize,
    /// Total cost (sum of distances to assigned medoids)
    cost: f64,
    /// Number of iterations performed
    iterations: usize,

    pub fn deinit(self: *KMedoidsResult, allocator: Allocator) void {
        allocator.free(self.medoids);
        allocator.free(self.labels);
    }
};

/// Euclidean distance function
fn euclideanDistance(a: []const f64, b: []const f64, d: usize) f64 {
    var sum: f64 = 0.0;
    for (0..d) |i| {
        const diff = a[i] - b[i];
        sum += diff * diff;
    }
    return @sqrt(sum);
}

/// Manhattan distance function
pub fn manhattanDistance(a: []const f64, b: []const f64, d: usize) f64 {
    var sum: f64 = 0.0;
    for (0..d) |i| {
        sum += @abs(a[i] - b[i]);
    }
    return sum;
}

/// Compute distance matrix (used for initialization and cost computation)
/// Time: O(n²×d)
/// Space: O(n²)
fn computeDistanceMatrix(
    allocator: Allocator,
    X: []const f64,
    n: usize,
    d: usize,
    distance_fn: DistanceFn,
) ![]f64 {
    const distances = try allocator.alloc(f64, n * n);
    errdefer allocator.free(distances);

    for (0..n) |i| {
        const point_i = X[i * d .. (i + 1) * d];
        for (0..n) |j| {
            if (i == j) {
                distances[i * n + j] = 0.0;
            } else {
                const point_j = X[j * d .. (j + 1) * d];
                distances[i * n + j] = distance_fn(point_i, point_j, d);
            }
        }
    }

    return distances;
}

/// Initialize medoids using k-means++ style selection
/// Time: O(n×k×d)
/// Space: O(n)
fn initializeMedoids(
    allocator: Allocator,
    distances: []const f64,
    n: usize,
    k: usize,
    rng: Random,
) ![]usize {
    const medoids = try allocator.alloc(usize, k);
    errdefer allocator.free(medoids);

    // Choose first medoid randomly
    medoids[0] = rng.intRangeLessThan(usize, 0, n);

    // Choose remaining medoids based on distance from existing medoids
    const min_distances = try allocator.alloc(f64, n);
    defer allocator.free(min_distances);

    for (0..n) |i| {
        min_distances[i] = std.math.floatMax(f64);
    }

    for (1..k) |cluster_idx| {
        // Update minimum distances to existing medoids
        const prev_medoid = medoids[cluster_idx - 1];
        for (0..n) |i| {
            const dist = distances[i * n + prev_medoid];
            if (dist < min_distances[i]) {
                min_distances[i] = dist;
            }
        }

        // Choose next medoid proportional to squared distance
        var sum_sq_dist: f64 = 0.0;
        for (0..n) |i| {
            // Skip already chosen medoids
            var is_medoid = false;
            for (medoids[0..cluster_idx]) |m| {
                if (i == m) {
                    is_medoid = true;
                    break;
                }
            }
            if (!is_medoid) {
                sum_sq_dist += min_distances[i] * min_distances[i];
            }
        }

        const rand_val = rng.float(f64) * sum_sq_dist;
        var cumsum: f64 = 0.0;
        var chosen_idx: usize = 0;
        for (0..n) |i| {
            var is_medoid = false;
            for (medoids[0..cluster_idx]) |m| {
                if (i == m) {
                    is_medoid = true;
                    break;
                }
            }
            if (!is_medoid) {
                cumsum += min_distances[i] * min_distances[i];
                if (cumsum >= rand_val) {
                    chosen_idx = i;
                    break;
                }
            }
        }

        medoids[cluster_idx] = chosen_idx;
    }

    return medoids;
}

/// Assign each point to nearest medoid
/// Time: O(n×k)
/// Space: O(n)
fn assignClusters(
    allocator: Allocator,
    distances: []const f64,
    n: usize,
    medoids: []const usize,
    k: usize,
) ![]usize {
    const labels = try allocator.alloc(usize, n);
    errdefer allocator.free(labels);

    for (0..n) |i| {
        var min_dist = std.math.floatMax(f64);
        var best_cluster: usize = 0;

        for (0..k) |cluster_idx| {
            const medoid = medoids[cluster_idx];
            const dist = distances[i * n + medoid];
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = cluster_idx;
            }
        }

        labels[i] = best_cluster;
    }

    return labels;
}

/// Compute total cost (sum of distances to assigned medoids)
/// Time: O(n)
/// Space: O(1)
fn computeCost(
    distances: []const f64,
    n: usize,
    labels: []const usize,
    medoids: []const usize,
) f64 {
    var total_cost: f64 = 0.0;
    for (0..n) |i| {
        const cluster = labels[i];
        const medoid = medoids[cluster];
        total_cost += distances[i * n + medoid];
    }
    return total_cost;
}

/// K-Medoids clustering using PAM algorithm
///
/// Time: O(k(n-k)²×iter) where n = points, k = clusters, iter = iterations
/// Space: O(n² + n×k) for distance matrix and assignments
///
/// Parameters:
/// - `X`: Input data matrix (n × d, row-major: [point0_dim0, point0_dim1, ..., point1_dim0, ...])
/// - `n`: Number of data points
/// - `d`: Number of dimensions per point
/// - `k`: Number of clusters (must be ≤ n)
/// - `allocator`: Memory allocator
/// - `options`: Configuration options (max iterations, random seed, distance function)
///
/// Returns: KMedoidsResult with medoid indices, cluster assignments, cost, iteration count
///
/// Errors:
/// - `OutOfMemory`: Insufficient memory
/// - `InvalidInput`: k > n or invalid dimensions
pub fn kmedoids(
    X: []const f64,
    n: usize,
    d: usize,
    k: usize,
    allocator: Allocator,
    options: KMedoidsOptions,
) !KMedoidsResult {
    if (k > n) return error.InvalidInput;
    if (n == 0 or d == 0) return error.InvalidInput;

    // Initialize random number generator
    var prng = std.rand.DefaultPrng.init(options.random_seed);
    const rng = prng.random();

    // Compute distance matrix
    const distances = try computeDistanceMatrix(allocator, X, n, d, options.distance_fn);
    defer allocator.free(distances);

    // Initialize medoids
    var medoids = try initializeMedoids(allocator, distances, n, k, rng);
    errdefer allocator.free(medoids);

    // Assign initial clusters
    var labels = try assignClusters(allocator, distances, n, medoids, k);
    errdefer allocator.free(labels);

    var current_cost = computeCost(distances, n, labels, medoids);
    var iteration: usize = 0;

    // PAM SWAP phase: iteratively improve medoids
    while (iteration < options.max_iterations) : (iteration += 1) {
        var improved = false;

        // Try swapping each medoid with each non-medoid
        for (0..k) |cluster_idx| {
            for (0..n) |candidate| {
                // Skip if candidate is already a medoid
                var is_medoid = false;
                for (medoids) |m| {
                    if (candidate == m) {
                        is_medoid = true;
                        break;
                    }
                }
                if (is_medoid) continue;

                // Compute cost if we swap current_medoid with candidate
                const old_medoid = medoids[cluster_idx];
                medoids[cluster_idx] = candidate;

                // Recompute labels with new medoid configuration
                const new_labels = try assignClusters(allocator, distances, n, medoids, k);
                const new_cost = computeCost(distances, n, new_labels, medoids);

                // Keep swap if it improves cost
                if (new_cost < current_cost) {
                    allocator.free(labels);
                    labels = new_labels;
                    current_cost = new_cost;
                    improved = true;
                } else {
                    // Revert swap
                    medoids[cluster_idx] = old_medoid;
                    allocator.free(new_labels);
                }
            }
        }

        // Converged if no improvement
        if (!improved) break;
    }

    return KMedoidsResult{
        .medoids = medoids,
        .labels = labels,
        .cost = current_cost,
        .iterations = iteration,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "K-Medoids: basic clustering (2D, 3 clusters)" {
    const allocator = std.testing.allocator;

    // 3 clear clusters in 2D
    const data = [_]f64{
        // Cluster 0 (bottom-left)
        0.0, 0.0,
        0.1, 0.1,
        0.2, 0.0,
        // Cluster 1 (top-right)
        5.0, 5.0,
        5.1, 5.1,
        5.0, 5.2,
        // Cluster 2 (bottom-right)
        5.0, 0.0,
        5.1, 0.1,
        5.2, 0.0,
    };

    var result = try kmedoids(&data, 9, 2, 3, allocator, .{});
    defer result.deinit(allocator);

    // Should converge
    try std.testing.expect(result.iterations < 300);

    // Cost should be small (points close to medoids)
    try std.testing.expect(result.cost < 5.0);

    // Check that each cluster is cohesive (same labels for nearby points)
    try std.testing.expectEqual(result.labels[0], result.labels[1]);
    try std.testing.expectEqual(result.labels[3], result.labels[4]);
    try std.testing.expectEqual(result.labels[6], result.labels[7]);
}

test "K-Medoids: Manhattan distance" {
    const allocator = std.testing.allocator;

    const data = [_]f64{
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        10.0, 10.0,
        10.0, 11.0,
        11.0, 10.0,
    };

    var result = try kmedoids(&data, 6, 2, 2, allocator, .{
        .distance_fn = manhattanDistance,
    });
    defer result.deinit(allocator);

    // Should find 2 distinct clusters
    try std.testing.expect(result.cost > 0.0);
    try std.testing.expect(result.iterations < 300);

    // Points 0-2 should be in one cluster, 3-5 in another
    const cluster0 = result.labels[0];
    const cluster1 = result.labels[3];
    try std.testing.expectEqual(cluster0, result.labels[1]);
    try std.testing.expectEqual(cluster0, result.labels[2]);
    try std.testing.expectEqual(cluster1, result.labels[4]);
    try std.testing.expectEqual(cluster1, result.labels[5]);
    try std.testing.expect(cluster0 != cluster1);
}

test "K-Medoids: single cluster" {
    const allocator = std.testing.allocator;

    const data = [_]f64{
        1.0, 2.0,
        1.5, 2.5,
        1.2, 2.3,
    };

    var result = try kmedoids(&data, 3, 2, 1, allocator, .{});
    defer result.deinit(allocator);

    // All points should be in cluster 0
    for (result.labels) |label| {
        try std.testing.expectEqual(@as(usize, 0), label);
    }

    // Medoid should be one of the data points
    try std.testing.expect(result.medoids[0] < 3);
}

test "K-Medoids: k equals n (each point is a medoid)" {
    const allocator = std.testing.allocator;

    const data = [_]f64{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    };

    var result = try kmedoids(&data, 3, 2, 3, allocator, .{});
    defer result.deinit(allocator);

    // Each point should be its own medoid
    for (0..3) |i| {
        try std.testing.expectEqual(i, result.labels[i]);
    }

    // Cost should be zero (each point is its own medoid)
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result.cost, 1e-10);
}

test "K-Medoids: robustness to outliers" {
    const allocator = std.testing.allocator;

    // Cluster with outlier
    const data = [_]f64{
        // Main cluster
        0.0,  0.0,
        0.1,  0.1,
        0.2,  0.0,
        0.0,  0.2,
        0.1,  0.2,
        // Outlier
        100.0, 100.0,
    };

    var result = try kmedoids(&data, 6, 2, 2, allocator, .{});
    defer result.deinit(allocator);

    // First 5 points should be in one cluster
    const main_cluster = result.labels[0];
    for (1..5) |i| {
        try std.testing.expectEqual(main_cluster, result.labels[i]);
    }

    // Outlier should be in different cluster
    try std.testing.expect(result.labels[5] != main_cluster);
}

test "K-Medoids: reproducibility with same seed" {
    const allocator = std.testing.allocator;

    const data = [_]f64{
        0.0, 0.0,
        1.0, 1.0,
        5.0, 5.0,
        6.0, 6.0,
    };

    var result1 = try kmedoids(&data, 4, 2, 2, allocator, .{ .random_seed = 123 });
    defer result1.deinit(allocator);

    var result2 = try kmedoids(&data, 4, 2, 2, allocator, .{ .random_seed = 123 });
    defer result2.deinit(allocator);

    // Same seed should produce same results
    try std.testing.expectEqualSlices(usize, result1.labels, result2.labels);
    try std.testing.expectApproxEqAbs(result1.cost, result2.cost, 1e-10);
}

test "K-Medoids: different seeds produce potentially different results" {
    const allocator = std.testing.allocator;

    const data = [_]f64{
        0.0, 0.0,
        1.0, 1.0,
        5.0, 5.0,
        6.0, 6.0,
    };

    var result1 = try kmedoids(&data, 4, 2, 2, allocator, .{ .random_seed = 1 });
    defer result1.deinit(allocator);

    var result2 = try kmedoids(&data, 4, 2, 2, allocator, .{ .random_seed = 9999 });
    defer result2.deinit(allocator);

    // Different seeds may produce different assignments (not guaranteed to differ, but possible)
    // We just check both are valid results
    try std.testing.expect(result1.cost >= 0.0);
    try std.testing.expect(result2.cost >= 0.0);
}

test "K-Medoids: 1D data" {
    const allocator = std.testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 10.0, 11.0, 12.0 };

    var result = try kmedoids(&data, 6, 1, 2, allocator, .{});
    defer result.deinit(allocator);

    // Should find 2 clusters: {1,2,3} and {10,11,12}
    const cluster0 = result.labels[0];
    const cluster1 = result.labels[3];
    try std.testing.expectEqual(cluster0, result.labels[1]);
    try std.testing.expectEqual(cluster0, result.labels[2]);
    try std.testing.expectEqual(cluster1, result.labels[4]);
    try std.testing.expectEqual(cluster1, result.labels[5]);
    try std.testing.expect(cluster0 != cluster1);
}

test "K-Medoids: high-dimensional data" {
    const allocator = std.testing.allocator;

    // 4D data with 2 clusters
    const data = [_]f64{
        // Cluster 0
        0.0, 0.0, 0.0, 0.0,
        0.1, 0.1, 0.1, 0.1,
        0.2, 0.2, 0.2, 0.2,
        // Cluster 1
        5.0, 5.0, 5.0, 5.0,
        5.1, 5.1, 5.1, 5.1,
        5.2, 5.2, 5.2, 5.2,
    };

    var result = try kmedoids(&data, 6, 4, 2, allocator, .{});
    defer result.deinit(allocator);

    // Check cluster cohesion
    const cluster0 = result.labels[0];
    const cluster1 = result.labels[3];
    try std.testing.expectEqual(cluster0, result.labels[1]);
    try std.testing.expectEqual(cluster0, result.labels[2]);
    try std.testing.expectEqual(cluster1, result.labels[4]);
    try std.testing.expectEqual(cluster1, result.labels[5]);
    try std.testing.expect(cluster0 != cluster1);
}

test "K-Medoids: convergence within max iterations" {
    const allocator = std.testing.allocator;

    const data = [_]f64{
        0.0, 0.0,
        0.1, 0.1,
        5.0, 5.0,
        5.1, 5.1,
    };

    var result = try kmedoids(&data, 4, 2, 2, allocator, .{ .max_iterations = 10 });
    defer result.deinit(allocator);

    // Should converge within 10 iterations for simple data
    try std.testing.expect(result.iterations <= 10);
}

test "K-Medoids: error on k > n" {
    const allocator = std.testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };

    const result = kmedoids(&data, 2, 2, 5, allocator, .{});
    try std.testing.expectError(error.InvalidInput, result);
}

test "K-Medoids: error on empty data" {
    const allocator = std.testing.allocator;

    const data = [_]f64{};

    const result = kmedoids(&data, 0, 2, 1, allocator, .{});
    try std.testing.expectError(error.InvalidInput, result);
}

test "K-Medoids: medoids are actual data points" {
    const allocator = std.testing.allocator;

    const data = [_]f64{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0,
    };

    var result = try kmedoids(&data, 4, 2, 2, allocator, .{});
    defer result.deinit(allocator);

    // Each medoid index should be valid
    for (result.medoids) |medoid_idx| {
        try std.testing.expect(medoid_idx < 4);
    }

    // Medoids should be distinct
    try std.testing.expect(result.medoids[0] != result.medoids[1]);
}

test "K-Medoids: cost decreases or stays same over iterations" {
    const allocator = std.testing.allocator;

    const data = [_]f64{
        0.0, 0.0,
        0.5, 0.5,
        1.0, 1.0,
        5.0, 5.0,
        5.5, 5.5,
        6.0, 6.0,
    };

    // We can't directly track intermediate costs, but final cost should be reasonable
    var result = try kmedoids(&data, 6, 2, 2, allocator, .{});
    defer result.deinit(allocator);

    // Cost should be positive and finite
    try std.testing.expect(result.cost > 0.0);
    try std.testing.expect(result.cost < std.math.floatMax(f64));
}
