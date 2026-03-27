const std = @import("std");
const zuda = @import("zuda");

const NDArray = zuda.ndarray.NDArray;
const descriptive = zuda.stats.descriptive;
const distributions = zuda.stats.distributions;

/// K-Means Clustering Example
///
/// Demonstrates:
/// 1. Data generation using Normal distribution
/// 2. K-means clustering algorithm implementation
/// 3. NDArray operations (indexing, slicing, arithmetic)
/// 4. Statistical analysis (mean, variance, distance)
/// 5. ASCII visualization of 2D clusters
///
/// Workflow:
///   Generate synthetic data → K-means clustering → Evaluate inertia → Visualize
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== K-Means Clustering Demonstration ===\n\n", .{});

    // Part 1: Generate synthetic clustered data
    std.debug.print("Part 1: Data Generation\n", .{});
    std.debug.print("------------------------\n", .{});

    const n_samples = 300;
    const n_features = 2;
    const n_clusters = 3;

    // Cluster centers
    const centers = [_][2]f64{
        .{ 0.0, 0.0 },
        .{ 5.0, 5.0 },
        .{ 0.0, 10.0 },
    };

    // Generate data: 100 points per cluster with Normal(center, std=1.0)
    var data = try NDArray(f64, 2).zeros(allocator, &.{ n_samples, n_features }, .row_major);
    defer data.deinit();

    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();

    for (0..n_clusters) |cluster_idx| {
        const normal_x = try distributions.Normal(f64).init(centers[cluster_idx][0], 1.0);
        const normal_y = try distributions.Normal(f64).init(centers[cluster_idx][1], 1.0);

        const start = cluster_idx * 100;
        const end = start + 100;

        for (start..end) |i| {
            const x = normal_x.sample(random);
            const y = normal_y.sample(random);
            data.set(&.{ @intCast(i), 0 }, x);
            data.set(&.{ @intCast(i), 1 }, y);
        }
    }

    std.debug.print("Generated {} samples with {} features\n", .{ n_samples, n_features });
    std.debug.print("True cluster centers:\n", .{});
    for (centers, 0..) |center, i| {
        std.debug.print("  Cluster {}: ({d:.2}, {d:.2})\n", .{ i, center[0], center[1] });
    }
    std.debug.print("\n", .{});

    // Part 2: K-Means Clustering
    std.debug.print("Part 2: K-Means Algorithm\n", .{});
    std.debug.print("--------------------------\n", .{});

    const max_iter = 100;
    const tol = 1e-4;

    // Initialize centroids randomly from data points
    var centroids = try NDArray(f64, 2).zeros(allocator, &.{ n_clusters, n_features }, .row_major);
    defer centroids.deinit();

    // Select random points as initial centroids
    for (0..n_clusters) |k| {
        const idx = random.intRangeAtMost(usize, 0, n_samples - 1);
        for (0..n_features) |j| {
            const val = try data.get(&.{ @intCast(idx), @intCast(j) });
            centroids.set(&.{ @intCast(k), @intCast(j) }, val);
        }
    }

    var labels = try allocator.alloc(usize, n_samples);
    defer allocator.free(labels);

    var converged = false;
    var iter: usize = 0;

    while (iter < max_iter and !converged) : (iter += 1) {
        // Assignment step: assign each point to nearest centroid
        for (0..n_samples) |i| {
            var min_dist: f64 = std.math.inf(f64);
            var best_cluster: usize = 0;

            for (0..n_clusters) |k| {
                var dist: f64 = 0.0;
                for (0..n_features) |j| {
                    const point_val = try data.get(&.{ @intCast(i), @intCast(j) });
                    const centroid_val = try centroids.get(&.{ @intCast(k), @intCast(j) });
                    const diff = point_val - centroid_val;
                    dist += diff * diff;
                }
                dist = @sqrt(dist);

                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = k;
                }
            }

            labels[i] = best_cluster;
        }

        // Update step: recompute centroids
        var new_centroids = try NDArray(f64, 2).zeros(allocator, &.{ n_clusters, n_features }, .row_major);
        defer new_centroids.deinit();

        var counts = try allocator.alloc(usize, n_clusters);
        defer allocator.free(counts);
        @memset(counts, 0);

        for (0..n_samples) |i| {
            const cluster = labels[i];
            counts[cluster] += 1;

            for (0..n_features) |j| {
                const val = try data.get(&.{ @intCast(i), @intCast(j) });
                const current = try new_centroids.get(&.{ @intCast(cluster), @intCast(j) });
                new_centroids.set(&.{ @intCast(cluster), @intCast(j) }, current + val);
            }
        }

        // Average to get new centroids
        for (0..n_clusters) |k| {
            if (counts[k] > 0) {
                for (0..n_features) |j| {
                    const sum = try new_centroids.get(&.{ @intCast(k), @intCast(j) });
                    new_centroids.set(&.{ @intCast(k), @intCast(j) }, sum / @as(f64, @floatFromInt(counts[k])));
                }
            }
        }

        // Check convergence: max centroid movement < tol
        var max_movement: f64 = 0.0;
        for (0..n_clusters) |k| {
            var movement: f64 = 0.0;
            for (0..n_features) |j| {
                const old_val = try centroids.get(&.{ @intCast(k), @intCast(j) });
                const new_val = try new_centroids.get(&.{ @intCast(k), @intCast(j) });
                const diff = old_val - new_val;
                movement += diff * diff;
            }
            movement = @sqrt(movement);
            max_movement = @max(max_movement, movement);
        }

        // Copy new centroids
        for (0..n_clusters) |k| {
            for (0..n_features) |j| {
                const val = try new_centroids.get(&.{ @intCast(k), @intCast(j) });
                centroids.set(&.{ @intCast(k), @intCast(j) }, val);
            }
        }

        if (max_movement < tol) {
            converged = true;
        }
    }

    std.debug.print("Converged: {} after {} iterations\n", .{ converged, iter });
    std.debug.print("\nLearned cluster centers:\n", .{});
    for (0..n_clusters) |k| {
        const x = try centroids.get(&.{ @intCast(k), 0 });
        const y = try centroids.get(&.{ @intCast(k), 1 });
        std.debug.print("  Cluster {}: ({d:.2}, {d:.2})\n", .{ k, x, y });
    }
    std.debug.print("\n", .{});

    // Part 3: Evaluate clustering quality (inertia)
    std.debug.print("Part 3: Clustering Evaluation\n", .{});
    std.debug.print("------------------------------\n", .{});

    // Compute inertia (within-cluster sum of squares)
    var inertia: f64 = 0.0;
    for (0..n_samples) |i| {
        const cluster = labels[i];
        for (0..n_features) |j| {
            const point_val = try data.get(&.{ @intCast(i), @intCast(j) });
            const centroid_val = try centroids.get(&.{ @intCast(cluster), @intCast(j) });
            const diff = point_val - centroid_val;
            inertia += diff * diff;
        }
    }

    std.debug.print("Inertia (within-cluster SS): {d:.2}\n", .{inertia});

    // Compute cluster sizes
    var cluster_sizes = try allocator.alloc(usize, n_clusters);
    defer allocator.free(cluster_sizes);
    @memset(cluster_sizes, 0);

    for (labels) |label| {
        cluster_sizes[label] += 1;
    }

    std.debug.print("Cluster sizes:\n", .{});
    for (0..n_clusters) |k| {
        std.debug.print("  Cluster {}: {} samples\n", .{ k, cluster_sizes[k] });
    }
    std.debug.print("\n", .{});

    // Part 4: Visualize clusters (ASCII scatter plot)
    std.debug.print("Part 4: Cluster Visualization\n", .{});
    std.debug.print("------------------------------\n", .{});

    // Find data range
    var min_x: f64 = std.math.inf(f64);
    var max_x: f64 = -std.math.inf(f64);
    var min_y: f64 = std.math.inf(f64);
    var max_y: f64 = -std.math.inf(f64);

    for (0..n_samples) |i| {
        const x = try data.get(&.{ @intCast(i), 0 });
        const y = try data.get(&.{ @intCast(i), 1 });
        min_x = @min(min_x, x);
        max_x = @max(max_x, x);
        min_y = @min(min_y, y);
        max_y = @max(max_y, y);
    }

    const width = 60;
    const height = 30;

    // Create plot grid
    var grid = try allocator.alloc([]u8, height);
    defer {
        for (grid) |row| {
            allocator.free(row);
        }
        allocator.free(grid);
    }

    for (0..height) |i| {
        grid[i] = try allocator.alloc(u8, width);
        @memset(grid[i], ' ');
    }

    // Plot data points
    const cluster_chars = [_]u8{ 'A', 'B', 'C' };
    for (0..n_samples) |i| {
        const x = try data.get(&.{ @intCast(i), 0 });
        const y = try data.get(&.{ @intCast(i), 1 });
        const cluster = labels[i];

        const px = @as(usize, @intFromFloat((x - min_x) / (max_x - min_x) * @as(f64, @floatFromInt(width - 1))));
        const py = @as(usize, @intFromFloat((y - min_y) / (max_y - min_y) * @as(f64, @floatFromInt(height - 1))));

        const py_inv = height - 1 - py; // Invert y for display

        if (px < width and py_inv < height) {
            grid[py_inv][px] = cluster_chars[cluster];
        }
    }

    // Plot centroids
    for (0..n_clusters) |k| {
        const x = try centroids.get(&.{ @intCast(k), 0 });
        const y = try centroids.get(&.{ @intCast(k), 1 });

        const px = @as(usize, @intFromFloat((x - min_x) / (max_x - min_x) * @as(f64, @floatFromInt(width - 1))));
        const py = @as(usize, @intFromFloat((y - min_y) / (max_y - min_y) * @as(f64, @floatFromInt(height - 1))));

        const py_inv = height - 1 - py;

        if (px < width and py_inv < height) {
            grid[py_inv][px] = '*'; // Centroids marked with *
        }
    }

    std.debug.print("Scatter plot (A/B/C = clusters, * = centroids):\n", .{});
    std.debug.print("Y ({d:.1} to {d:.1})\n", .{ min_y, max_y });
    std.debug.print("|\n", .{});

    for (grid) |row| {
        std.debug.print("|{s}|\n", .{row});
    }

    std.debug.print("|\n", .{});
    std.debug.print("+{s}+ X ({d:.1} to {d:.1})\n", .{ "-" ** width, min_x, max_x });

    std.debug.print("\n=== Summary ===\n", .{});
    std.debug.print("K-Means successfully clustered {} samples into {} groups\n", .{ n_samples, n_clusters });
    std.debug.print("Convergence: {} iterations\n", .{iter});
    std.debug.print("Final inertia: {d:.2}\n", .{inertia});
    std.debug.print("Average cluster size: {d:.1}\n", .{@as(f64, @floatFromInt(n_samples)) / @as(f64, @floatFromInt(n_clusters))});
}
