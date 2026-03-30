const std = @import("std");
const testing = std.testing;

/// Options for t-SNE algorithm
pub const TSNEOptions = struct {
    /// Number of output dimensions (typically 2 or 3 for visualization)
    n_components: usize = 2,
    /// Perplexity: effective number of nearest neighbors (5-50 typical)
    perplexity: f64 = 30.0,
    /// Learning rate for gradient descent (100-1000 typical)
    learning_rate: f64 = 200.0,
    /// Number of iterations
    n_iter: usize = 1000,
    /// Early exaggeration factor (multiply P by this for first iterations)
    early_exaggeration: f64 = 12.0,
    /// Number of iterations with early exaggeration
    early_exaggeration_iter: usize = 250,
    /// Final momentum value
    momentum: f64 = 0.8,
    /// Initial momentum value
    initial_momentum: f64 = 0.5,
    /// Minimum gain value
    min_gain: f64 = 0.01,
    /// Random seed (null = use random seed)
    seed: ?u64 = null,
};

/// t-distributed Stochastic Neighbor Embedding for dimensionality reduction
///
/// Time: O(n² × iter) where n = samples, iter = iterations
/// Space: O(n²) for pairwise similarities + O(n × d_out) for embedding
///
/// t-SNE is a non-linear dimensionality reduction technique particularly good for
/// visualizing high-dimensional data. It preserves local structure by modeling
/// similarities with Student's t-distribution in low-dimensional space.
///
/// Key differences from PCA:
/// - Non-linear (can capture complex manifolds)
/// - Preserves local structure (neighborhoods)
/// - Stochastic (results vary between runs)
/// - Computationally expensive (O(n²) vs O(nm²))
///
/// Use cases:
/// - Visualizing embeddings (word2vec, image features)
/// - Cluster discovery in high-dimensional data
/// - Exploratory data analysis
/// - Genomics and bioinformatics visualization
pub fn TSNE(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        options: TSNEOptions,
        rng: std.Random.DefaultPrng,

        /// Initialize t-SNE
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(allocator: std.mem.Allocator, options: TSNEOptions) Self {
            const seed = options.seed orelse @as(u64, @intCast(std.time.milliTimestamp()));
            return Self{
                .allocator = allocator,
                .options = options,
                .rng = std.Random.DefaultPrng.init(seed),
            };
        }

        /// Clean up resources
        pub fn deinit(self: *Self) void {
            _ = self;
        }

        /// Fit t-SNE and transform data to low-dimensional space
        ///
        /// Time: O(n² × iter) where n = n_samples, iter = n_iter
        /// Space: O(n² + n × d_out)
        ///
        /// X: Input data [n_samples × n_features]
        /// Returns: Embedded data [n_samples × n_components] (caller owns memory)
        pub fn fitTransform(self: *Self, X: []const T, n_samples: usize, n_features: usize) ![]T {
            if (n_samples < 2) return error.InsufficientSamples;
            if (self.options.perplexity >= @as(f64, @floatFromInt(n_samples))) return error.PerplexityTooLarge;

            // Compute pairwise similarities in high-dimensional space
            const P = try self.computePairwiseAffinities(X, n_samples, n_features);
            defer self.allocator.free(P);

            // Initialize low-dimensional embedding randomly
            const Y = try self.allocator.alloc(T, n_samples * self.options.n_components);
            errdefer self.allocator.free(Y);
            const random = self.rng.random();
            for (Y) |*val| {
                val.* = @as(T, @floatCast(random.floatNorm(f64) * 0.0001));
            }

            // Gradient descent with momentum
            const dY = try self.allocator.alloc(T, n_samples * self.options.n_components);
            defer self.allocator.free(dY);
            @memset(dY, 0);

            const gains = try self.allocator.alloc(T, n_samples * self.options.n_components);
            defer self.allocator.free(gains);
            @memset(gains, 1);

            const iY = try self.allocator.alloc(T, n_samples * self.options.n_components);
            defer self.allocator.free(iY);
            @memset(iY, 0);

            // Gradient descent iterations
            for (0..self.options.n_iter) |iter| {
                const momentum = if (iter < self.options.early_exaggeration_iter)
                    self.options.initial_momentum
                else
                    self.options.momentum;

                const exaggeration = if (iter < self.options.early_exaggeration_iter)
                    self.options.early_exaggeration
                else
                    1.0;

                // Compute gradient
                try self.computeGradient(Y, P, dY, n_samples, exaggeration);

                // Update with momentum and adaptive gains
                for (0..n_samples * self.options.n_components) |i| {
                    // Adaptive gains: increase if gradient direction is consistent, decrease otherwise
                    if ((dY[i] > 0) == (iY[i] > 0)) {
                        gains[i] += 0.2;
                    } else {
                        gains[i] *= 0.8;
                        if (gains[i] < self.options.min_gain) gains[i] = @floatCast(self.options.min_gain);
                    }

                    // Momentum update
                    iY[i] = @as(T, @floatCast(momentum)) * iY[i] - @as(T, @floatCast(self.options.learning_rate)) * gains[i] * dY[i];
                    Y[i] += iY[i];
                }

                // Zero-mean the embedding
                if ((iter + 1) % 10 == 0) {
                    self.zeroMean(Y, n_samples);
                }
            }

            return Y;
        }

        fn computePairwiseAffinities(self: *Self, X: []const T, n_samples: usize, n_features: usize) ![]T {
            // Compute squared distances
            const D2 = try self.allocator.alloc(T, n_samples * n_samples);
            defer self.allocator.free(D2);

            for (0..n_samples) |i| {
                for (0..n_samples) |j| {
                    var sum: T = 0;
                    for (0..n_features) |k| {
                        const diff = X[i * n_features + k] - X[j * n_features + k];
                        sum += diff * diff;
                    }
                    D2[i * n_samples + j] = sum;
                }
            }

            // Compute P matrix with binary search for perplexity
            const P = try self.allocator.alloc(T, n_samples * n_samples);
            errdefer self.allocator.free(P);

            const target_entropy = @log(@as(T, @floatCast(self.options.perplexity)));

            for (0..n_samples) |i| {
                // Binary search for beta (precision) that achieves target perplexity
                var beta: T = 1.0;
                var beta_min: T = 0;
                var beta_max: T = std.math.inf(T);

                for (0..50) |_| { // 50 iterations of binary search
                    // Compute P_i|j with current beta
                    var sum_p: T = 0;
                    var entropy: T = 0;
                    for (0..n_samples) |j| {
                        if (i == j) continue;
                        const p_ij = @exp(-D2[i * n_samples + j] * beta);
                        P[i * n_samples + j] = p_ij;
                        sum_p += p_ij;
                    }

                    // Normalize and compute entropy
                    if (sum_p > 0) {
                        for (0..n_samples) |j| {
                            if (i == j) continue;
                            P[i * n_samples + j] /= sum_p;
                            if (P[i * n_samples + j] > 1e-12) {
                                entropy -= P[i * n_samples + j] * @log(P[i * n_samples + j]);
                            }
                        }
                    }

                    // Check if entropy matches target
                    const entropy_diff = entropy - target_entropy;
                    if (@abs(entropy_diff) < 1e-5) break;

                    // Adjust beta
                    if (entropy_diff > 0) {
                        beta_min = beta;
                        if (std.math.isInf(beta_max)) {
                            beta *= 2;
                        } else {
                            beta = (beta + beta_max) / 2;
                        }
                    } else {
                        beta_max = beta;
                        beta = (beta + beta_min) / 2;
                    }
                }
            }

            // Symmetrize P matrix
            for (0..n_samples) |i| {
                for (0..n_samples) |j| {
                    const val = (P[i * n_samples + j] + P[j * n_samples + i]) / @as(T, @floatFromInt(2 * n_samples));
                    P[i * n_samples + j] = val;
                }
            }

            return P;
        }

        fn computeGradient(self: *Self, Y: []const T, P: []const T, dY: []T, n_samples: usize, exaggeration: f64) !void {
            const n_components = self.options.n_components;
            @memset(dY, 0);

            // Compute Q matrix (low-dimensional similarities)
            var sum_Q: T = 0;
            for (0..n_samples) |i| {
                for (0..n_samples) |j| {
                    if (i == j) continue;
                    var dist2: T = 0;
                    for (0..n_components) |d| {
                        const diff = Y[i * n_components + d] - Y[j * n_components + d];
                        dist2 += diff * diff;
                    }
                    sum_Q += 1 / (1 + dist2);
                }
            }

            // Compute gradient
            for (0..n_samples) |i| {
                for (0..n_samples) |j| {
                    if (i == j) continue;

                    var dist2: T = 0;
                    for (0..n_components) |d| {
                        const diff = Y[i * n_components + d] - Y[j * n_components + d];
                        dist2 += diff * diff;
                    }

                    const q_ij = (1 / (1 + dist2)) / sum_Q;
                    const p_ij = P[i * n_samples + j] * @as(T, @floatCast(exaggeration));
                    const mult = (p_ij - q_ij) * (1 / (1 + dist2));

                    for (0..n_components) |d| {
                        const diff = Y[i * n_components + d] - Y[j * n_components + d];
                        dY[i * n_components + d] += 4 * mult * diff;
                    }
                }
            }
        }

        fn zeroMean(self: *Self, Y: []T, n_samples: usize) void {
            const n_components = self.options.n_components;
            for (0..n_components) |d| {
                var mean: T = 0;
                for (0..n_samples) |i| {
                    mean += Y[i * n_components + d];
                }
                mean /= @floatFromInt(n_samples);
                for (0..n_samples) |i| {
                    Y[i * n_components + d] -= mean;
                }
            }
        }
    };
}

// Tests
test "t-SNE: basic 2D embedding" {
    const allocator = testing.allocator;

    // Simple 3-cluster dataset in 4D
    const X = [_]f64{
        // Cluster 1 (around [0,0,0,0])
        0.1,  0.2,  0.1,  0.0,
        0.0,  0.1,  0.2,  0.1,
        0.2,  0.0,  0.1,  0.2,
        // Cluster 2 (around [5,5,5,5])
        5.1,  5.0,  5.2,  5.1,
        5.0,  5.1,  5.0,  5.2,
        5.2,  5.1,  5.1,  5.0,
        // Cluster 3 (around [10,10,10,10])
        10.0, 10.1, 10.2, 10.0,
        10.1, 10.0, 10.1, 10.1,
        10.2, 10.2, 10.0, 10.1,
    };

    var tsne = TSNE(f64).init(allocator, .{
        .n_components = 2,
        .perplexity = 3.0,
        .n_iter = 500,
        .seed = 42,
    });
    defer tsne.deinit();

    const Y = try tsne.fitTransform(&X, 9, 4);
    defer allocator.free(Y);

    // Verify output shape
    try testing.expectEqual(18, Y.len); // 9 samples × 2 components

    // Verify clusters are separated (rough check)
    // Points in same cluster should be closer than points in different clusters
    const dist_within_1 = distance2D(Y[0..2], Y[2..4]); // Cluster 1
    const dist_within_2 = distance2D(Y[6..8], Y[8..10]); // Cluster 2
    const dist_between = distance2D(Y[0..2], Y[6..8]); // Cluster 1 vs 2

    try testing.expect(dist_within_1 < dist_between);
    try testing.expect(dist_within_2 < dist_between);
}

test "t-SNE: 3D embedding" {
    const allocator = testing.allocator;

    const X = [_]f64{
        0.0, 0.0, 1.0, 1.0, 2.0, 2.0,
        3.0, 3.0, 4.0, 4.0, 5.0, 5.0,
    };

    var tsne = TSNE(f64).init(allocator, .{
        .n_components = 3,
        .perplexity = 2.0,
        .n_iter = 300,
        .seed = 42,
    });
    defer tsne.deinit();

    const Y = try tsne.fitTransform(&X, 2, 6);
    defer allocator.free(Y);

    try testing.expectEqual(6, Y.len); // 2 samples × 3 components
}

test "t-SNE: different perplexity values" {
    const allocator = testing.allocator;

    const X = [_]f64{
        0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0,
        4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0,
    };

    // Low perplexity (local structure)
    {
        var tsne = TSNE(f64).init(allocator, .{ .perplexity = 2.0, .n_iter = 300, .seed = 42 });
        defer tsne.deinit();
        const Y = try tsne.fitTransform(&X, 2, 8);
        defer allocator.free(Y);
        try testing.expectEqual(4, Y.len);
    }

    // High perplexity (global structure)
    {
        var tsne = TSNE(f64).init(allocator, .{ .perplexity = 5.0, .n_iter = 300, .seed = 42 });
        defer tsne.deinit();
        const Y = try tsne.fitTransform(&X, 8, 2);
        defer allocator.free(Y);
        try testing.expectEqual(16, Y.len);
    }
}

test "t-SNE: reproducibility with same seed" {
    const allocator = testing.allocator;

    const X = [_]f64{
        0.0, 0.0, 1.0, 1.0,
        2.0, 2.0, 3.0, 3.0,
        4.0, 4.0, 5.0, 5.0,
    };

    var tsne1 = TSNE(f64).init(allocator, .{ .perplexity = 2.0, .n_iter = 200, .seed = 42 });
    defer tsne1.deinit();
    const Y1 = try tsne1.fitTransform(&X, 3, 4);
    defer allocator.free(Y1);

    var tsne2 = TSNE(f64).init(allocator, .{ .perplexity = 2.0, .n_iter = 200, .seed = 42 });
    defer tsne2.deinit();
    const Y2 = try tsne2.fitTransform(&X, 3, 4);
    defer allocator.free(Y2);

    // Same seed should produce identical results
    for (Y1, Y2) |y1, y2| {
        try testing.expectApproxEqAbs(y1, y2, 1e-6);
    }
}

test "t-SNE: f32 support" {
    const allocator = testing.allocator;

    const X = [_]f32{
        0.0, 0.0, 1.0, 1.0,
        2.0, 2.0, 3.0, 3.0,
    };

    var tsne = TSNE(f32).init(allocator, .{ .perplexity = 2.0, .n_iter = 200, .seed = 42 });
    defer tsne.deinit();

    const Y = try tsne.fitTransform(&X, 2, 4);
    defer allocator.free(Y);

    try testing.expectEqual(4, Y.len);
}

test "t-SNE: error on insufficient samples" {
    const allocator = testing.allocator;

    const X = [_]f64{ 0.0, 1.0 };

    var tsne = TSNE(f64).init(allocator, .{});
    defer tsne.deinit();

    const result = tsne.fitTransform(&X, 1, 2);
    try testing.expectError(error.InsufficientSamples, result);
}

test "t-SNE: error on perplexity too large" {
    const allocator = testing.allocator;

    const X = [_]f64{
        0.0, 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0, 7.0,
    };

    var tsne = TSNE(f64).init(allocator, .{ .perplexity = 10.0 }); // > n_samples
    defer tsne.deinit();

    const result = tsne.fitTransform(&X, 2, 4);
    try testing.expectError(error.PerplexityTooLarge, result);
}

test "t-SNE: cluster separation" {
    const allocator = testing.allocator;

    // Two well-separated clusters
    const X = [_]f64{
        // Cluster 1 (around origin)
        0.1,  0.1,  0.0,  0.0,
        0.0,  0.2,  0.1,  0.0,
        0.2,  0.0,  0.1,  0.1,
        // Cluster 2 (far away)
        10.0, 10.1, 10.0, 10.2,
        10.1, 10.0, 10.2, 10.0,
        10.2, 10.2, 10.1, 10.1,
    };

    var tsne = TSNE(f64).init(allocator, .{
        .perplexity = 2.0,
        .n_iter = 500,
        .seed = 42,
    });
    defer tsne.deinit();

    const Y = try tsne.fitTransform(&X, 6, 4);
    defer allocator.free(Y);

    // Compute average distances within and between clusters
    var dist_within: f64 = 0;
    var dist_between: f64 = 0;

    // Within cluster 1
    for (0..3) |i| {
        for (i + 1..3) |j| {
            dist_within += distance2D(Y[i * 2 ..][0..2], Y[j * 2 ..][0..2]);
        }
    }
    dist_within /= 3; // 3 pairs

    // Between clusters
    for (0..3) |i| {
        for (3..6) |j| {
            dist_between += distance2D(Y[i * 2 ..][0..2], Y[j * 2 ..][0..2]);
        }
    }
    dist_between /= 9; // 3×3 pairs

    // Between-cluster distance should be much larger
    try testing.expect(dist_between > 2 * dist_within);
}

test "t-SNE: memory safety" {
    const allocator = testing.allocator;

    const X = [_]f64{
        0.0, 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0, 7.0,
        8.0, 9.0, 10.0, 11.0,
    };

    var tsne = TSNE(f64).init(allocator, .{ .perplexity = 2.0, .n_iter = 200, .seed = 42 });
    defer tsne.deinit();

    const Y = try tsne.fitTransform(&X, 3, 4);
    defer allocator.free(Y);

    try testing.expectEqual(6, Y.len);
}

test "t-SNE: small dataset" {
    const allocator = testing.allocator;

    const X = [_]f64{
        0.0, 1.0, 5.0, 6.0,
    };

    var tsne = TSNE(f64).init(allocator, .{ .perplexity = 1.0, .n_iter = 200, .seed = 42 });
    defer tsne.deinit();

    const Y = try tsne.fitTransform(&X, 2, 2);
    defer allocator.free(Y);

    try testing.expectEqual(4, Y.len);
}

test "t-SNE: learning rate variations" {
    const allocator = testing.allocator;

    const X = [_]f64{
        0.0, 0.0, 1.0, 1.0, 2.0, 2.0,
        3.0, 3.0, 4.0, 4.0, 5.0, 5.0,
    };

    // Low learning rate
    {
        var tsne = TSNE(f64).init(allocator, .{ .learning_rate = 50.0, .n_iter = 300, .seed = 42 });
        defer tsne.deinit();
        const Y = try tsne.fitTransform(&X, 2, 6);
        defer allocator.free(Y);
        try testing.expectEqual(4, Y.len);
    }

    // High learning rate
    {
        var tsne = TSNE(f64).init(allocator, .{ .learning_rate = 500.0, .n_iter = 300, .seed = 42 });
        defer tsne.deinit();
        const Y = try tsne.fitTransform(&X, 2, 6);
        defer allocator.free(Y);
        try testing.expectEqual(4, Y.len);
    }
}

test "t-SNE: early exaggeration effect" {
    const allocator = testing.allocator;

    const X = [_]f64{
        0.0, 0.0, 1.0, 1.0,
        2.0, 2.0, 3.0, 3.0,
        4.0, 4.0, 5.0, 5.0,
    };

    // With early exaggeration (default)
    var tsne1 = TSNE(f64).init(allocator, .{
        .perplexity = 2.0,
        .n_iter = 300,
        .early_exaggeration = 12.0,
        .seed = 42,
    });
    defer tsne1.deinit();
    const Y1 = try tsne1.fitTransform(&X, 3, 4);
    defer allocator.free(Y1);

    // Without early exaggeration
    var tsne2 = TSNE(f64).init(allocator, .{
        .perplexity = 2.0,
        .n_iter = 300,
        .early_exaggeration = 1.0,
        .seed = 42,
    });
    defer tsne2.deinit();
    const Y2 = try tsne2.fitTransform(&X, 3, 4);
    defer allocator.free(Y2);

    // Results should differ (early exaggeration helps separate clusters)
    var different = false;
    for (Y1, Y2) |y1, y2| {
        if (@abs(y1 - y2) > 0.1) {
            different = true;
            break;
        }
    }
    try testing.expect(different);
}

test "t-SNE: larger dataset (stress test)" {
    const allocator = testing.allocator;

    // 50 samples, 10 features
    const X = try allocator.alloc(f64, 50 * 10);
    defer allocator.free(X);

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    for (X) |*val| {
        val.* = random.float(f64) * 10;
    }

    var tsne = TSNE(f64).init(allocator, .{
        .perplexity = 10.0,
        .n_iter = 500,
        .seed = 42,
    });
    defer tsne.deinit();

    const Y = try tsne.fitTransform(X, 50, 10);
    defer allocator.free(Y);

    try testing.expectEqual(100, Y.len); // 50 samples × 2 components

    // Verify no NaN or Inf values
    for (Y) |val| {
        try testing.expect(!std.math.isNan(val));
        try testing.expect(!std.math.isInf(val));
    }
}

// Helper function
fn distance2D(a: []const f64, b: []const f64) f64 {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    return @sqrt(dx * dx + dy * dy);
}
