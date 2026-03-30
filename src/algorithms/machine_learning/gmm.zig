const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const math = std.math;

/// Gaussian Mixture Model (GMM) Clustering
///
/// Models data as a weighted mixture of k Gaussian distributions, allowing:
/// 1. Soft cluster assignments (probabilistic membership)
/// 2. Elliptical cluster shapes (not just spherical like K-Means)
/// 3. Varying cluster sizes and densities
///
/// Algorithm: Expectation-Maximization (EM)
/// 1. E-step: Compute responsibilities (posterior probabilities)
/// 2. M-step: Update means, covariances, and mixing coefficients
/// 3. Repeat until convergence or max iterations
///
/// Time: O(n × k × d² × iterations) where:
///   - n = number of data points
///   - k = number of Gaussian components
///   - d = dimensionality
///   - iterations = EM convergence iterations (typically 10-100)
///
/// Space: O(k × d² + n × k) for covariances and responsibilities
///
/// Use cases:
/// - Soft clustering (overlapping clusters)
/// - Density estimation (probabilistic modeling)
/// - Anomaly detection (low probability samples)
/// - Image segmentation (color/texture modeling)
/// - Speech recognition (phoneme modeling)
///
/// Advantages over K-Means:
/// - Soft assignments (probability distributions)
/// - Flexible cluster shapes (full covariance)
/// - Principled probabilistic framework
///
/// Limitations:
/// - More computationally expensive than K-Means
/// - Requires k to be specified a priori
/// - May converge to local optimum
/// - Sensitive to initialization
/// - Diagonal covariance variant used here for efficiency
///
/// Example:
/// ```zig
/// const data = [_][]const f64{
///     &.{1.0, 1.0}, &.{1.5, 2.0}, &.{3.0, 4.0}, &.{5.0, 7.0},
///     &.{3.5, 5.0}, &.{4.5, 5.0}, &.{3.5, 4.5}
/// };
/// var result = try gmm(f64, allocator, &data, 2, .{});
/// defer result.deinit();
/// // result.means contains k Gaussian means
/// // result.responsibilities[i][j] = P(component j | data[i])
/// ```
pub fn GMMResult(comptime T: type) type {
    return struct {
        means: [][]T, // k means, each of dim d
        covariances: [][]T, // k diagonal covariances, each of dim d
        weights: []T, // k mixing coefficients (sum to 1)
        responsibilities: [][]T, // n × k soft assignments
        log_likelihood: T, // final log-likelihood
        iterations: usize, // actual EM iterations until convergence
        allocator: Allocator,

        const Self = @This();

        /// Free all allocated memory
        /// Time: O(k + n)
        /// Space: O(1)
        pub fn deinit(self: *Self) void {
            for (self.means) |mean| {
                self.allocator.free(mean);
            }
            self.allocator.free(self.means);

            for (self.covariances) |cov| {
                self.allocator.free(cov);
            }
            self.allocator.free(self.covariances);

            self.allocator.free(self.weights);

            for (self.responsibilities) |resp| {
                self.allocator.free(resp);
            }
            self.allocator.free(self.responsibilities);
        }

        /// Predict cluster assignments (hard clustering)
        /// Returns the component with highest responsibility for each point
        /// Time: O(n × k)
        /// Space: O(n)
        pub fn predict(self: Self, allocator: Allocator) ![]usize {
            const labels = try allocator.alloc(usize, self.responsibilities.len);
            for (self.responsibilities, 0..) |resp, i| {
                var max_resp: T = resp[0];
                var max_idx: usize = 0;
                for (resp[1..], 1..) |r, j| {
                    if (r > max_resp) {
                        max_resp = r;
                        max_idx = j;
                    }
                }
                labels[i] = max_idx;
            }
            return labels;
        }

        /// Compute log-likelihood for new data point
        /// Returns log P(x) = log Σ_k π_k N(x | μ_k, Σ_k)
        /// Time: O(k × d)
        /// Space: O(1)
        pub fn logLikelihood(self: Self, x: []const T) T {
            var log_prob_sum: T = -math.inf(T);
            for (self.means, 0..) |mean, k| {
                const log_weight = @log(self.weights[k]);
                const log_gaussian = logGaussianDensity(T, x, mean, self.covariances[k]);
                const log_prob = log_weight + log_gaussian;
                // Log-sum-exp trick for numerical stability
                log_prob_sum = logSumExp(T, log_prob_sum, log_prob);
            }
            return log_prob_sum;
        }
    };
}

pub const GMMOptions = struct {
    max_iterations: usize = 100,
    tolerance: f64 = 1e-6, // convergence threshold for log-likelihood change
    random_seed: ?u64 = null, // for reproducible initialization
    regularization: f64 = 1e-6, // added to diagonal covariance for numerical stability
};

/// Gaussian Mixture Model clustering via EM algorithm
///
/// Time: O(n × k × d² × iterations)
/// Space: O(k × d² + n × k)
///
/// Params:
///   T: numeric type (f32, f64)
///   allocator: memory allocator
///   data: n data points, each []const T of length d
///   k: number of Gaussian components (must be ≤ n)
///   options: configuration (max_iterations, tolerance, seed, regularization)
///
/// Returns: GMMResult with means, covariances, weights, responsibilities, log-likelihood
///
/// Errors:
///   - OutOfMemory: allocation failed
///   - InvalidK: k > n or k == 0
///   - EmptyData: data.len == 0
///   - InconsistentDimensions: data points have different dimensions
pub fn gmm(
    comptime T: type,
    allocator: Allocator,
    data: []const []const T,
    k: usize,
    options: GMMOptions,
) !GMMResult(T) {
    if (data.len == 0) return error.EmptyData;
    if (k == 0 or k > data.len) return error.InvalidK;

    const n = data.len;
    const d = data[0].len;

    // Check consistent dimensions
    for (data) |point| {
        if (point.len != d) return error.InconsistentDimensions;
    }

    // Initialize parameters using K-Means++
    const means = try initializeMeans(T, allocator, data, k, options.random_seed);
    errdefer {
        for (means) |mean| allocator.free(mean);
        allocator.free(means);
    }

    // Initialize covariances to identity (diagonal)
    var covariances = try allocator.alloc([]T, k);
    errdefer allocator.free(covariances);
    for (covariances, 0..) |*cov, i| {
        cov.* = try allocator.alloc(T, d);
        errdefer {
            for (covariances[0..i]) |c| allocator.free(c);
            for (covariances) |c| allocator.free(c);
            allocator.free(covariances);
        }
        @memset(cov.*, 1.0);
    }

    // Initialize weights uniformly
    const weights = try allocator.alloc(T, k);
    errdefer allocator.free(weights);
    const uniform_weight: T = 1.0 / @as(T, @floatFromInt(k));
    @memset(weights, uniform_weight);

    // Allocate responsibilities (n × k)
    var responsibilities = try allocator.alloc([]T, n);
    errdefer allocator.free(responsibilities);
    for (responsibilities, 0..) |*resp, i| {
        resp.* = try allocator.alloc(T, k);
        errdefer {
            for (responsibilities[0..i]) |r| allocator.free(r);
            for (responsibilities) |r| allocator.free(r);
            allocator.free(responsibilities);
        }
    }

    var prev_log_likelihood: T = -math.inf(T);
    var iterations: usize = 0;

    // EM iterations
    while (iterations < options.max_iterations) : (iterations += 1) {
        // E-step: Compute responsibilities
        var log_likelihood: T = 0.0;
        for (data, 0..) |x, i| {
            var log_resp_unnormalized = try allocator.alloc(T, k);
            defer allocator.free(log_resp_unnormalized);

            var log_sum: T = -math.inf(T);
            for (0..k) |j| {
                const log_weight = @log(weights[j]);
                const log_gaussian = logGaussianDensity(T, x, means[j], covariances[j]);
                log_resp_unnormalized[j] = log_weight + log_gaussian;
                log_sum = logSumExp(T, log_sum, log_resp_unnormalized[j]);
            }

            // Normalize responsibilities
            for (0..k) |j| {
                responsibilities[i][j] = @exp(log_resp_unnormalized[j] - log_sum);
            }

            log_likelihood += log_sum;
        }

        log_likelihood /= @as(T, @floatFromInt(n));

        // Check convergence
        if (iterations > 0 and @abs(log_likelihood - prev_log_likelihood) < options.tolerance) {
            break;
        }
        prev_log_likelihood = log_likelihood;

        // M-step: Update parameters
        try updateParameters(T, allocator, data, means, covariances, weights, responsibilities, options.regularization);
    }

    return GMMResult(T){
        .means = means,
        .covariances = covariances,
        .weights = weights,
        .responsibilities = responsibilities,
        .log_likelihood = prev_log_likelihood,
        .iterations = iterations,
        .allocator = allocator,
    };
}

/// Initialize means using K-Means++ algorithm
fn initializeMeans(
    comptime T: type,
    allocator: Allocator,
    data: []const []const T,
    k: usize,
    seed: ?u64,
) ![][]T {
    const n = data.len;
    const d = data[0].len;

    var means = try allocator.alloc([]T, k);
    errdefer allocator.free(means);

    var prng = std.Random.DefaultPrng.init(seed orelse @intCast(std.time.timestamp()));
    const random = prng.random();

    // First centroid: random point
    const first_idx = random.intRangeLessThan(usize, 0, n);
    means[0] = try allocator.alloc(T, d);
    @memcpy(means[0], data[first_idx]);

    var distances = try allocator.alloc(T, n);
    defer allocator.free(distances);

    // Subsequent centroids: K-Means++ initialization
    for (1..k) |c| {
        // Compute squared distances to nearest centroid
        var total_distance: T = 0.0;
        for (data, 0..) |point, i| {
            var min_dist: T = math.inf(T);
            for (means[0..c]) |mean| {
                const dist = squaredEuclideanDistance(T, point, mean);
                min_dist = @min(min_dist, dist);
            }
            distances[i] = min_dist;
            total_distance += min_dist;
        }

        // Select next centroid with probability proportional to squared distance
        const threshold = random.float(T) * total_distance;
        var cumulative: T = 0.0;
        var selected_idx: usize = n - 1;
        for (0..n) |i| {
            cumulative += distances[i];
            if (cumulative >= threshold) {
                selected_idx = i;
                break;
            }
        }

        means[c] = try allocator.alloc(T, d);
        @memcpy(means[c], data[selected_idx]);
    }

    return means;
}

/// Update parameters in M-step
fn updateParameters(
    comptime T: type,
    allocator: Allocator,
    data: []const []const T,
    means: [][]T,
    covariances: [][]T,
    weights: []T,
    responsibilities: [][]T,
    regularization: f64,
) !void {
    const n = data.len;
    const k = means.len;
    _ = means[0].len; // d not needed since we iterate directly

    // Compute effective counts (N_k = Σ_i γ_ik)
    var n_k = try allocator.alloc(T, k);
    defer allocator.free(n_k);
    @memset(n_k, 0.0);

    for (responsibilities) |resp| {
        for (resp, 0..) |r, j| {
            n_k[j] += r;
        }
    }

    // Update means: μ_k = (1/N_k) Σ_i γ_ik x_i
    for (means, 0..) |mean, j| {
        @memset(mean, 0.0);
        for (data, 0..) |x, i| {
            const resp = responsibilities[i][j];
            for (x, 0..) |val, dim| {
                mean[dim] += resp * val;
            }
        }
        for (mean) |*val| {
            val.* /= n_k[j];
        }
    }

    // Update covariances: Σ_k = (1/N_k) Σ_i γ_ik (x_i - μ_k)(x_i - μ_k)^T (diagonal)
    for (covariances, 0..) |cov, j| {
        @memset(cov, 0.0);
        for (data, 0..) |x, i| {
            const resp = responsibilities[i][j];
            for (x, 0..) |val, dim| {
                const diff = val - means[j][dim];
                cov[dim] += resp * diff * diff;
            }
        }
        for (cov) |*val| {
            val.* = val.* / n_k[j] + @as(T, @floatCast(regularization));
        }
    }

    // Update weights: π_k = N_k / n
    for (weights, 0..) |*weight, j| {
        weight.* = n_k[j] / @as(T, @floatFromInt(n));
    }
}

/// Compute log Gaussian density: log N(x | μ, Σ) with diagonal covariance
fn logGaussianDensity(comptime T: type, x: []const T, mean: []const T, cov_diag: []const T) T {
    const d = x.len;
    var log_det: T = 0.0;
    var mahalanobis: T = 0.0;

    for (x, 0..) |val, i| {
        const diff = val - mean[i];
        const var_i = cov_diag[i];
        log_det += @log(var_i);
        mahalanobis += diff * diff / var_i;
    }

    const d_float: T = @floatFromInt(d);
    const log_2pi = @log(2.0 * math.pi);
    return -0.5 * (d_float * log_2pi + log_det + mahalanobis);
}

/// Squared Euclidean distance
fn squaredEuclideanDistance(comptime T: type, a: []const T, b: []const T) T {
    var sum: T = 0.0;
    for (a, b) |ai, bi| {
        const diff = ai - bi;
        sum += diff * diff;
    }
    return sum;
}

/// Log-sum-exp trick: log(exp(a) + exp(b))
fn logSumExp(comptime T: type, a: T, b: T) T {
    if (math.isInf(a)) return b;
    if (math.isInf(b)) return a;
    const max_val = @max(a, b);
    return max_val + @log(@exp(a - max_val) + @exp(b - max_val));
}

// ============================================================================
// Tests
// ============================================================================

test "GMM: basic 2D clustering" {
    const allocator = testing.allocator;

    // Two well-separated clusters
    const data = [_][]const f64{
        &.{ 1.0, 1.0 }, &.{ 1.5, 2.0 }, &.{ 1.2, 1.8 }, // Cluster 1
        &.{ 8.0, 8.0 }, &.{ 8.5, 9.0 }, &.{ 8.2, 8.8 }, // Cluster 2
    };

    var result = try gmm(f64, allocator, &data, 2, .{ .random_seed = 42 });
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.means.len);
    try testing.expectEqual(@as(usize, 6), result.responsibilities.len);

    // Check soft assignments sum to 1
    for (result.responsibilities) |resp| {
        var sum: f64 = 0.0;
        for (resp) |r| sum += r;
        try testing.expectApproxEqAbs(1.0, sum, 1e-9);
    }

    // Check weights sum to 1
    var weight_sum: f64 = 0.0;
    for (result.weights) |w| weight_sum += w;
    try testing.expectApproxEqAbs(1.0, weight_sum, 1e-9);

    // Predict hard assignments
    const labels = try result.predict(allocator);
    defer allocator.free(labels);

    // First 3 points should be in same cluster
    try testing.expectEqual(labels[0], labels[1]);
    try testing.expectEqual(labels[1], labels[2]);

    // Last 3 points should be in same cluster (different from first)
    try testing.expectEqual(labels[3], labels[4]);
    try testing.expectEqual(labels[4], labels[5]);
    try testing.expect(labels[0] != labels[3]);
}

test "GMM: single cluster" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 1.0, 1.0 },
        &.{ 1.5, 1.5 },
        &.{ 2.0, 2.0 },
    };

    var result = try gmm(f64, allocator, &data, 1, .{});
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), result.means.len);
    try testing.expectEqual(@as(usize, 1), result.weights.len);
    try testing.expectApproxEqAbs(1.0, result.weights[0], 1e-9);

    // All responsibilities should be 1.0 (100% in single cluster)
    for (result.responsibilities) |resp| {
        try testing.expectApproxEqAbs(1.0, resp[0], 1e-9);
    }
}

test "GMM: convergence" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 1.0, 1.0 }, &.{ 1.5, 2.0 }, &.{ 2.0, 1.5 },
        &.{ 8.0, 8.0 }, &.{ 8.5, 9.0 }, &.{ 9.0, 8.5 },
    };

    var result = try gmm(f64, allocator, &data, 2, .{ .max_iterations = 100, .tolerance = 1e-6 });
    defer result.deinit();

    // Should converge in reasonable iterations
    try testing.expect(result.iterations <= 100);
    try testing.expect(result.iterations > 0);

    // Log-likelihood should be finite
    try testing.expect(!math.isInf(result.log_likelihood));
    try testing.expect(!math.isNan(result.log_likelihood));
}

test "GMM: log-likelihood for new point" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 1.0, 1.0 }, &.{ 1.5, 2.0 }, &.{ 2.0, 1.5 },
    };

    var result = try gmm(f64, allocator, &data, 1, .{});
    defer result.deinit();

    // Point close to cluster should have high likelihood
    const close_point = [_]f64{ 1.5, 1.5 };
    const ll_close = result.logLikelihood(&close_point);
    try testing.expect(!math.isInf(ll_close));
    try testing.expect(!math.isNan(ll_close));

    // Point far from cluster should have lower likelihood
    const far_point = [_]f64{ 100.0, 100.0 };
    const ll_far = result.logLikelihood(&far_point);
    try testing.expect(ll_close > ll_far);
}

test "GMM: f32 support" {
    const allocator = testing.allocator;

    const data = [_][]const f32{
        &.{ 1.0, 1.0 }, &.{ 1.5, 2.0 },
        &.{ 8.0, 8.0 }, &.{ 8.5, 9.0 },
    };

    var result = try gmm(f32, allocator, &data, 2, .{});
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.means.len);
    try testing.expectEqual(@as(usize, 4), result.responsibilities.len);
}

test "GMM: empty data" {
    const allocator = testing.allocator;
    const data: []const []const f64 = &.{};

    const result = gmm(f64, allocator, data, 2, .{});
    try testing.expectError(error.EmptyData, result);
}

test "GMM: invalid k (zero)" {
    const allocator = testing.allocator;
    const data = [_][]const f64{&.{ 1.0, 1.0 }};

    const result = gmm(f64, allocator, &data, 0, .{});
    try testing.expectError(error.InvalidK, result);
}

test "GMM: invalid k (exceeds n)" {
    const allocator = testing.allocator;
    const data = [_][]const f64{&.{ 1.0, 1.0 }};

    const result = gmm(f64, allocator, &data, 2, .{});
    try testing.expectError(error.InvalidK, result);
}

test "GMM: inconsistent dimensions" {
    const allocator = testing.allocator;
    const data = [_][]const f64{
        &.{ 1.0, 1.0 },
        &.{ 2.0, 2.0, 3.0 }, // Different dimension
    };

    const result = gmm(f64, allocator, &data, 2, .{});
    try testing.expectError(error.InconsistentDimensions, result);
}

test "GMM: overlapping clusters (soft assignment)" {
    const allocator = testing.allocator;

    // Points at cluster boundary
    const data = [_][]const f64{
        &.{ 0.0, 0.0 }, // Clearly in cluster 1
        &.{ 5.0, 5.0 }, // Boundary point
        &.{ 10.0, 10.0 }, // Clearly in cluster 2
    };

    var result = try gmm(f64, allocator, &data, 2, .{ .random_seed = 42 });
    defer result.deinit();

    // Boundary point should have split responsibility
    const boundary_resp = result.responsibilities[1];
    try testing.expect(boundary_resp[0] > 0.1 and boundary_resp[0] < 0.9);
    try testing.expect(boundary_resp[1] > 0.1 and boundary_resp[1] < 0.9);

    // Edge points should have clear assignments
    const edge1_resp = result.responsibilities[0];
    const edge2_resp = result.responsibilities[2];

    // One should be highly assigned to first cluster
    const max1 = @max(edge1_resp[0], edge1_resp[1]);
    try testing.expect(max1 > 0.9);

    // Other should be highly assigned to second cluster
    const max2 = @max(edge2_resp[0], edge2_resp[1]);
    try testing.expect(max2 > 0.9);
}

test "GMM: covariance estimation" {
    const allocator = testing.allocator;

    // Cluster with different variances in x and y
    const data = [_][]const f64{
        &.{ 0.0, 0.0 }, &.{ 1.0, 0.0 }, &.{ 2.0, 0.0 }, // Wide in x
        &.{ 0.0, 0.1 }, &.{ 1.0, 0.1 }, &.{ 2.0, 0.1 }, // Narrow in y
    };

    var result = try gmm(f64, allocator, &data, 1, .{});
    defer result.deinit();

    // Variance in x should be larger than in y
    const cov = result.covariances[0];
    try testing.expect(cov[0] > cov[1]);
}

test "GMM: regularization prevents singular covariance" {
    const allocator = testing.allocator;

    // All points identical (would cause zero variance)
    const data = [_][]const f64{
        &.{ 5.0, 5.0 },
        &.{ 5.0, 5.0 },
        &.{ 5.0, 5.0 },
    };

    var result = try gmm(f64, allocator, &data, 1, .{ .regularization = 1e-6 });
    defer result.deinit();

    // Covariance should be regularization value (not zero)
    for (result.covariances[0]) |cov_val| {
        try testing.expect(cov_val >= 1e-6);
    }
}

test "GMM: reproducibility with seed" {
    const allocator = testing.allocator;

    const data = [_][]const f64{
        &.{ 1.0, 1.0 }, &.{ 1.5, 2.0 }, &.{ 2.0, 1.5 },
        &.{ 8.0, 8.0 }, &.{ 8.5, 9.0 }, &.{ 9.0, 8.5 },
    };

    var result1 = try gmm(f64, allocator, &data, 2, .{ .random_seed = 123 });
    defer result1.deinit();

    var result2 = try gmm(f64, allocator, &data, 2, .{ .random_seed = 123 });
    defer result2.deinit();

    // Results should be identical
    try testing.expectEqual(result1.iterations, result2.iterations);
    try testing.expectApproxEqAbs(result1.log_likelihood, result2.log_likelihood, 1e-9);

    // Means should match
    for (result1.means, result2.means) |m1, m2| {
        for (m1, m2) |v1, v2| {
            try testing.expectApproxEqAbs(v1, v2, 1e-9);
        }
    }
}
