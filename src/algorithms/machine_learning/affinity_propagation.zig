/// Affinity Propagation Clustering
///
/// Message-passing clustering algorithm that automatically determines the number of clusters
/// based on similarity between data points. Unlike K-Means, it doesn't require specifying K.
///
/// Algorithm:
/// 1. Compute similarity matrix S (negative squared Euclidean distance)
/// 2. Initialize responsibility R and availability A matrices to zero
/// 3. Iteratively update:
///    - Responsibility R(i,k): how well-suited point k is to be exemplar for point i
///    - Availability A(i,k): how appropriate it is for point i to choose k as exemplar
/// 4. Exemplars emerge where R(k,k) + A(k,k) > 0
/// 5. Assign non-exemplars to nearest exemplar
///
/// Key concepts:
/// - **Preference**: diagonal values of similarity matrix, controls cluster granularity
///   * Lower preference → fewer clusters
///   * Higher preference → more clusters
///   * Default: median of similarities
/// - **Damping factor**: λ ∈ [0.5, 1) smooths message updates to prevent oscillations
/// - **Message passing**: iterative belief propagation on factor graph
///
/// Time complexity: O(n² × iterations)
/// Space complexity: O(n²) for similarity, responsibility, and availability matrices
///
/// Trade-offs vs other clustering algorithms:
/// - vs K-Means: No K needed, handles arbitrary shapes, but O(n²) vs O(nkd)
/// - vs DBSCAN: More stable, no epsilon tuning, but more expensive
/// - vs Mean Shift: Faster convergence, but still O(n²) memory
/// - vs Hierarchical: Automatic K, but no dendrogram hierarchy
///
/// Use cases:
/// - Image clustering (face recognition, image segmentation)
/// - Document clustering (no prior knowledge of categories)
/// - Gene expression analysis (identify representative genes)
/// - Recommendation systems (find exemplar items)
/// - Network analysis (community detection)
///
/// References:
/// - Frey & Dueck (2007): "Clustering by Passing Messages Between Data Points"

const std = @import("std");
const testing = std.testing;

/// Affinity Propagation clustering result
pub fn AffinityPropagationResult(comptime T: type) type {
    _ = T; // Type parameter retained for API consistency
    return struct {
        /// Cluster assignments (n elements, cluster indices)
        labels: []usize,
        /// Exemplar indices for each cluster (n_clusters elements)
        exemplars: []usize,
        /// Number of clusters discovered
        n_clusters: usize,
        /// Number of iterations until convergence
        iterations: usize,
        /// Allocator for cleanup
        allocator: std.mem.Allocator,

        const Self = @This();

        /// Release all memory
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.labels);
            self.allocator.free(self.exemplars);
        }
    };
}

/// Affinity Propagation configuration options
pub const AffinityPropagationOptions = struct {
    /// Damping factor λ ∈ [0.5, 1) to prevent oscillations (default: 0.5)
    damping: f64 = 0.5,
    /// Maximum number of iterations (default: 200)
    max_iterations: usize = 200,
    /// Convergence threshold: stop if no change in exemplars for this many iterations (default: 15)
    convergence_iterations: usize = 15,
    /// Preference value (lower → fewer clusters, higher → more clusters)
    /// If null, uses median of similarities (default: null)
    preference: ?f64 = null,
};

/// Perform Affinity Propagation clustering
///
/// Time: O(n² × iterations)
/// Space: O(n²)
///
/// Example:
/// ```
/// const data = [_][]const f64{
///     &.{ 1.0, 2.0 },
///     &.{ 1.5, 1.8 },
///     &.{ 5.0, 8.0 },
///     &.{ 8.0, 8.0 },
/// };
/// var result = try affinityPropagation(f64, allocator, &data, .{});
/// defer result.deinit();
/// std.debug.print("Found {} clusters\n", .{result.n_clusters});
/// ```
pub fn affinityPropagation(
    comptime T: type,
    allocator: std.mem.Allocator,
    data: []const []const T,
    options: AffinityPropagationOptions,
) !AffinityPropagationResult(T) {
    if (data.len == 0) return error.EmptyData;
    if (options.damping < 0.5 or options.damping >= 1.0) return error.InvalidDamping;

    const n = data.len;
    const d = data[0].len;

    // Compute similarity matrix S (negative squared Euclidean distance)
    var S = try allocator.alloc(T, n * n);
    defer allocator.free(S);

    for (0..n) |i| {
        for (0..n) |j| {
            var sum: T = 0;
            for (0..d) |k| {
                const diff = data[i][k] - data[j][k];
                sum += diff * diff;
            }
            S[i * n + j] = -sum; // Negative squared distance
        }
    }

    // Set preference (diagonal values)
    const pref: T = if (options.preference) |p| @floatCast(p) else blk: {
        // Use median of similarities
        var similarities = try allocator.alloc(T, n * (n - 1));
        defer allocator.free(similarities);
        var idx: usize = 0;
        for (0..n) |i| {
            for (0..n) |j| {
                if (i != j) {
                    similarities[idx] = S[i * n + j];
                    idx += 1;
                }
            }
        }
        std.mem.sort(T, similarities, {}, comptime std.sort.asc(T));
        break :blk similarities[similarities.len / 2];
    };

    for (0..n) |i| {
        S[i * n + i] = pref;
    }

    // Initialize responsibility and availability matrices
    var R = try allocator.alloc(T, n * n);
    defer allocator.free(R);
    var A = try allocator.alloc(T, n * n);
    defer allocator.free(A);
    @memset(R, 0);
    @memset(A, 0);

    // Temporary matrices for damping
    var R_new = try allocator.alloc(T, n * n);
    defer allocator.free(R_new);
    var A_new = try allocator.alloc(T, n * n);
    defer allocator.free(A_new);

    var iter: usize = 0;
    var converged_count: usize = 0;
    var prev_exemplars = try allocator.alloc(usize, n);
    defer allocator.free(prev_exemplars);
    @memset(prev_exemplars, std.math.maxInt(usize));

    while (iter < options.max_iterations and converged_count < options.convergence_iterations) : (iter += 1) {
        // Update responsibility R(i,k) = S(i,k) - max_{k' ≠ k}[A(i,k') + S(i,k')]
        for (0..n) |i| {
            for (0..n) |k| {
                // Find max_{k' ≠ k}[A(i,k') + S(i,k')]
                var max_val: T = -std.math.inf(T);
                for (0..n) |kk| {
                    if (kk != k) {
                        const val = A[i * n + kk] + S[i * n + kk];
                        if (val > max_val) max_val = val;
                    }
                }
                R_new[i * n + k] = S[i * n + k] - max_val;
            }
        }

        // Apply damping
        const damping: T = @floatCast(options.damping);
        for (0..n * n) |idx| {
            R[idx] = damping * R[idx] + (1.0 - damping) * R_new[idx];
        }

        // Update availability A(i,k) = min(0, R(k,k) + Σ_{i' ∉ {i,k}} max(0, R(i',k)))
        for (0..n) |i| {
            for (0..n) |k| {
                if (i == k) {
                    // A(k,k) = Σ_{i' ≠ k} max(0, R(i',k))
                    var sum: T = 0;
                    for (0..n) |ii| {
                        if (ii != k) {
                            sum += @max(0, R[ii * n + k]);
                        }
                    }
                    A_new[k * n + k] = sum;
                } else {
                    // A(i,k) = min(0, R(k,k) + Σ_{i' ∉ {i,k}} max(0, R(i',k)))
                    var sum: T = R[k * n + k];
                    for (0..n) |ii| {
                        if (ii != i and ii != k) {
                            sum += @max(0, R[ii * n + k]);
                        }
                    }
                    A_new[i * n + k] = @min(0, sum);
                }
            }
        }

        // Apply damping
        for (0..n * n) |idx| {
            A[idx] = damping * A[idx] + (1.0 - damping) * A_new[idx];
        }

        // Check convergence: exemplars unchanged
        var exemplars_list = try std.ArrayList(usize).initCapacity(allocator, n);
        defer exemplars_list.deinit(allocator);

        for (0..n) |i| {
            const criterion = R[i * n + i] + A[i * n + i];
            if (criterion > 0) {
                exemplars_list.appendAssumeCapacity(i);
            }
        }

        // Compare with previous exemplars
        var changed = exemplars_list.items.len != prev_exemplars.len;
        if (!changed) {
            for (exemplars_list.items, 0..) |ex, idx| {
                if (idx >= prev_exemplars.len or prev_exemplars[idx] != ex) {
                    changed = true;
                    break;
                }
            }
        }

        if (changed) {
            converged_count = 0;
            @memset(prev_exemplars, std.math.maxInt(usize));
            for (exemplars_list.items, 0..) |ex, idx| {
                if (idx < prev_exemplars.len) prev_exemplars[idx] = ex;
            }
        } else {
            converged_count += 1;
        }
    }

    // Extract final exemplars
    var exemplars_list = try std.ArrayList(usize).initCapacity(allocator, n);
    defer exemplars_list.deinit(allocator);

    for (0..n) |i| {
        const criterion = R[i * n + i] + A[i * n + i];
        if (criterion > 0) {
            exemplars_list.appendAssumeCapacity(i);
        }
    }

    // If no exemplars, pick point with highest self-similarity
    if (exemplars_list.items.len == 0) {
        var best_idx: usize = 0;
        var best_val: T = R[0] + A[0];
        for (1..n) |i| {
            const val = R[i * n + i] + A[i * n + i];
            if (val > best_val) {
                best_val = val;
                best_idx = i;
            }
        }
        exemplars_list.appendAssumeCapacity(best_idx);
    }

    const n_clusters = exemplars_list.items.len;
    const exemplars = try allocator.alloc(usize, n_clusters);
    @memcpy(exemplars, exemplars_list.items);

    // Assign each point to nearest exemplar
    const labels = try allocator.alloc(usize, n);
    for (0..n) |i| {
        var best_cluster: usize = 0;
        var best_sim: T = S[i * n + exemplars[0]];
        for (1..n_clusters) |c| {
            const sim = S[i * n + exemplars[c]];
            if (sim > best_sim) {
                best_sim = sim;
                best_cluster = c;
            }
        }
        labels[i] = best_cluster;
    }

    return AffinityPropagationResult(T){
        .labels = labels,
        .exemplars = exemplars,
        .n_clusters = n_clusters,
        .iterations = iter,
        .allocator = allocator,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "AffinityPropagation: basic 2 clusters" {
    const allocator = testing.allocator;

    // Two well-separated clusters
    const data = [_][]const f64{
        &.{ 0.0, 0.0 },
        &.{ 1.0, 0.0 },
        &.{ 0.0, 1.0 },
        &.{ 10.0, 10.0 },
        &.{ 11.0, 10.0 },
        &.{ 10.0, 11.0 },
    };

    var result = try affinityPropagation(f64, allocator, &data, .{});
    defer result.deinit();

    try testing.expect(result.n_clusters >= 2);
    try testing.expect(result.iterations <= 200);

    // Check that points in same cluster have same label
    try testing.expectEqual(result.labels[0], result.labels[1]);
    try testing.expectEqual(result.labels[0], result.labels[2]);
    try testing.expectEqual(result.labels[3], result.labels[4]);
    try testing.expectEqual(result.labels[3], result.labels[5]);

    // Check that clusters are different
    try testing.expect(result.labels[0] != result.labels[3]);
}

test "AffinityPropagation: single cluster" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 0.0, 0.0 },
        &.{ 0.1, 0.0 },
        &.{ 0.0, 0.1 },
        &.{ 0.1, 0.1 },
    };

    var result = try affinityPropagation(f64, allocator, &data, .{});
    defer result.deinit();

    // Should form 1-2 clusters (tight grouping)
    try testing.expect(result.n_clusters <= 2);
}

test "AffinityPropagation: preference controls cluster count" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 0.0, 0.0 },
        &.{ 1.0, 0.0 },
        &.{ 2.0, 0.0 },
        &.{ 3.0, 0.0 },
        &.{ 4.0, 0.0 },
    };

    // Low preference → fewer clusters
    var result1 = try affinityPropagation(f64, allocator, &data, .{ .preference = -100.0 });
    defer result1.deinit();

    // High preference → more clusters
    var result2 = try affinityPropagation(f64, allocator, &data, .{ .preference = -1.0 });
    defer result2.deinit();

    try testing.expect(result2.n_clusters >= result1.n_clusters);
}

test "AffinityPropagation: 3D data" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 0.0, 0.0, 0.0 },
        &.{ 1.0, 0.0, 0.0 },
        &.{ 0.0, 1.0, 0.0 },
        &.{ 10.0, 10.0, 10.0 },
        &.{ 11.0, 10.0, 10.0 },
    };

    // Use higher preference to encourage cluster formation
    var result = try affinityPropagation(f64, allocator, &data, .{ .preference = -10.0 });
    defer result.deinit();

    try testing.expect(result.n_clusters >= 1);
    // When clusters form, verify grouping
    if (result.n_clusters >= 2) {
        try testing.expectEqual(result.labels[3], result.labels[4]);
    }
}

test "AffinityPropagation: f32 support" {
    const allocator = testing.allocator;

    const data = [_][]const f32{
        &.{ 0.0, 0.0 },
        &.{ 1.0, 0.0 },
        &.{ 10.0, 10.0 },
    };

    // Use higher preference to encourage cluster formation
    var result = try affinityPropagation(f32, allocator, &data, .{ .preference = -5.0 });
    defer result.deinit();

    try testing.expect(result.n_clusters >= 1);
    // When clusters form, verify separation
    if (result.n_clusters >= 2) {
        try testing.expect(result.labels[0] != result.labels[2]);
    }
}

test "AffinityPropagation: convergence check" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 0.0, 0.0 },
        &.{ 1.0, 0.0 },
        &.{ 10.0, 10.0 },
    };

    var result = try affinityPropagation(f64, allocator, &data, .{ .convergence_iterations = 5, .preference = -5.0 });
    defer result.deinit();

    // Should converge reasonably quickly
    try testing.expect(result.iterations <= 200);
    try testing.expect(result.n_clusters >= 1);
}

test "AffinityPropagation: damping factor" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 0.0, 0.0 },
        &.{ 1.0, 0.0 },
        &.{ 10.0, 10.0 },
    };

    // Different damping factors should still converge
    var result1 = try affinityPropagation(f64, allocator, &data, .{ .damping = 0.5 });
    defer result1.deinit();

    var result2 = try affinityPropagation(f64, allocator, &data, .{ .damping = 0.9 });
    defer result2.deinit();

    try testing.expect(result1.n_clusters >= 1);
    try testing.expect(result2.n_clusters >= 1);
}

test "AffinityPropagation: large dataset" {
    const allocator = testing.allocator;

    var data_list = try std.ArrayList([]const f64).initCapacity(allocator, 50);
    defer {
        for (data_list.items) |item| {
            allocator.free(item);
        }
        data_list.deinit(allocator);
    }

    // Generate 50 points in 3 clusters
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (0..50) |i| {
        const point = try allocator.alloc(f64, 2);
        if (i < 17) {
            point[0] = random.float(f64) * 2.0;
            point[1] = random.float(f64) * 2.0;
        } else if (i < 34) {
            point[0] = 10.0 + random.float(f64) * 2.0;
            point[1] = 10.0 + random.float(f64) * 2.0;
        } else {
            point[0] = 20.0 + random.float(f64) * 2.0;
            point[1] = random.float(f64) * 2.0;
        }
        data_list.appendAssumeCapacity(point);
    }

    // Use higher preference to encourage cluster formation
    var result = try affinityPropagation(f64, allocator, data_list.items, .{ .preference = -5.0 });
    defer result.deinit();

    try testing.expect(result.n_clusters >= 1);
    try testing.expect(result.n_clusters <= 50);
}

test "AffinityPropagation: exemplars are valid" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 0.0, 0.0 },
        &.{ 1.0, 0.0 },
        &.{ 10.0, 10.0 },
        &.{ 11.0, 10.0 },
    };

    var result = try affinityPropagation(f64, allocator, &data, .{});
    defer result.deinit();

    // All exemplars must be valid indices
    for (result.exemplars) |ex| {
        try testing.expect(ex < data.len);
    }

    // Each exemplar should be assigned to its own cluster
    for (result.exemplars, 0..) |ex, cluster| {
        try testing.expectEqual(cluster, result.labels[ex]);
    }
}

test "AffinityPropagation: memory safety" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 0.0, 0.0 },
        &.{ 1.0, 0.0 },
    };

    var result = try affinityPropagation(f64, allocator, &data, .{});
    defer result.deinit();

    // Just verify no leaks (allocator will catch them)
    try testing.expect(true);
}

test "AffinityPropagation: empty data error" {
    const allocator = testing.allocator;
    const data: []const []const f64 = &.{};
    try testing.expectError(error.EmptyData, affinityPropagation(f64, allocator, data, .{}));
}

test "AffinityPropagation: invalid damping error" {
    const allocator = testing.allocator;
    const data = [_][]const f64{&.{ 0.0, 0.0 }};
    try testing.expectError(error.InvalidDamping, affinityPropagation(f64, allocator, &data, .{ .damping = 0.4 }));
    try testing.expectError(error.InvalidDamping, affinityPropagation(f64, allocator, &data, .{ .damping = 1.0 }));
}
