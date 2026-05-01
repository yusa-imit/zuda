const std = @import("std");
const Allocator = std.mem.Allocator;
const math = std.math;
const linalg = @import("../../linalg/decompositions.zig");

/// Multivariate Normal Distribution (MVN)
///
/// N(μ, Σ) where:
/// - μ ∈ ℝⁿ is the mean vector
/// - Σ ∈ ℝⁿˣⁿ is the covariance matrix (symmetric positive definite)
///
/// PDF: f(x) = (2π)⁻ⁿ/² |Σ|⁻¹/² exp(-½(x-μ)ᵀΣ⁻¹(x-μ))
///
/// Use cases:
/// - Gaussian processes and Bayesian inference
/// - Correlated random variable modeling
/// - Financial portfolio risk (correlated returns)
/// - Spatial statistics (geospatial correlation)
pub fn MultivariateNormal(comptime T: type) type {
    return struct {
        mean: []const T, // μ (length n)
        cov: []const T, // Σ (n×n row-major)
        dim: usize, // dimension n
        chol: []T, // Cholesky decomposition L (n×n, lower triangular)
        log_det: T, // log|Σ| for PDF computation
        allocator: Allocator,

        const Self = @This();

        /// Initialize MVN from mean vector and covariance matrix
        ///
        /// Time: O(n³) for Cholesky decomposition | Space: O(n²)
        ///
        /// cov must be symmetric positive definite (SPD)
        /// Returns error.NotPositiveDefinite if Cholesky fails
        pub fn init(allocator: Allocator, mean: []const T, cov: []const T) !Self {
            const n = mean.len;
            if (cov.len != n * n) return error.DimensionMismatch;

            // Allocate Cholesky factor L
            const chol = try allocator.alloc(T, n * n);
            errdefer allocator.free(chol);

            // Copy covariance matrix for Cholesky decomposition
            @memcpy(chol, cov);

            // Compute Cholesky decomposition: Σ = LLᵀ
            try choleskyDecompose(T, chol, n);

            // Compute log|Σ| = 2 × sum(log(L[i,i]))
            var log_det: T = 0.0;
            for (0..n) |i| {
                log_det += @log(chol[i * n + i]);
            }
            log_det *= 2.0;

            return Self{
                .mean = mean,
                .cov = cov,
                .dim = n,
                .chol = chol,
                .log_det = log_det,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.chol);
        }

        /// Probability density function
        ///
        /// Time: O(n²) | Space: O(n)
        pub fn pdf(self: *const Self, x: []const T) !T {
            if (x.len != self.dim) return error.DimensionMismatch;

            const n = self.dim;
            const allocator = self.allocator;

            // Compute (x - μ)
            const diff = try allocator.alloc(T, n);
            defer allocator.free(diff);
            for (0..n) |i| {
                diff[i] = x[i] - self.mean[i];
            }

            // Solve Ly = (x - μ) using forward substitution
            const y = try allocator.alloc(T, n);
            defer allocator.free(y);
            try forwardSubstitution(T, self.chol, diff, y, n);

            // Compute Mahalanobis distance² = yᵀy
            var mahal_sq: T = 0.0;
            for (y) |val| {
                mahal_sq += val * val;
            }

            // PDF = (2π)⁻ⁿ/² |Σ|⁻¹/² exp(-½ Mahalanobis²)
            const log_pdf = -0.5 * (@as(T, @floatFromInt(n)) * @log(2.0 * math.pi) + self.log_det + mahal_sq);
            return @exp(log_pdf);
        }

        /// Log probability density
        ///
        /// Time: O(n²) | Space: O(n)
        ///
        /// More numerically stable than log(pdf(x)) for extreme probabilities
        pub fn logpdf(self: *const Self, x: []const T) !T {
            if (x.len != self.dim) return error.DimensionMismatch;

            const n = self.dim;
            const allocator = self.allocator;

            // Compute (x - μ)
            const diff = try allocator.alloc(T, n);
            defer allocator.free(diff);
            for (0..n) |i| {
                diff[i] = x[i] - self.mean[i];
            }

            // Solve Ly = (x - μ)
            const y = try allocator.alloc(T, n);
            defer allocator.free(y);
            try forwardSubstitution(T, self.chol, diff, y, n);

            // Compute Mahalanobis distance²
            var mahal_sq: T = 0.0;
            for (y) |val| {
                mahal_sq += val * val;
            }

            return -0.5 * (@as(T, @floatFromInt(n)) * @log(2.0 * math.pi) + self.log_det + mahal_sq);
        }

        /// Sample from the distribution using Cholesky method
        ///
        /// Time: O(n²) | Space: O(n)
        ///
        /// Algorithm: x = μ + L·z where z ~ N(0,I)
        /// rng must provide nextFloat() -> T in [0,1)
        pub fn sample(self: *const Self, allocator: Allocator, rng: anytype) ![]T {
            const n = self.dim;

            // Allocate result
            const result = try allocator.alloc(T, n);
            errdefer allocator.free(result);

            // Generate standard normal samples z ~ N(0,1)
            const z = try allocator.alloc(T, n);
            defer allocator.free(z);
            for (z) |*val| {
                val.* = boxMuller(T, rng);
            }

            // Compute x = μ + L·z
            for (0..n) |i| {
                result[i] = self.mean[i];
                for (0..n) |j| {
                    result[i] += self.chol[i * n + j] * z[j];
                }
            }

            return result;
        }

        /// Mahalanobis distance: √((x-μ)ᵀΣ⁻¹(x-μ))
        ///
        /// Time: O(n²) | Space: O(n)
        ///
        /// Measures distance from mean in units of covariance
        pub fn mahalanobis(self: *const Self, x: []const T) !T {
            if (x.len != self.dim) return error.DimensionMismatch;

            const n = self.dim;
            const allocator = self.allocator;

            // Compute (x - μ)
            const diff = try allocator.alloc(T, n);
            defer allocator.free(diff);
            for (0..n) |i| {
                diff[i] = x[i] - self.mean[i];
            }

            // Solve Ly = (x - μ)
            const y = try allocator.alloc(T, n);
            defer allocator.free(y);
            try forwardSubstitution(T, self.chol, diff, y, n);

            // Distance = ||y||
            var dist_sq: T = 0.0;
            for (y) |val| {
                dist_sq += val * val;
            }

            return @sqrt(dist_sq);
        }
    };
}

/// Cholesky decomposition: A = LLᵀ (in-place, lower triangular)
///
/// Time: O(n³) | Space: O(1)
fn choleskyDecompose(comptime T: type, A: []T, n: usize) !void {
    for (0..n) |i| {
        for (0..i + 1) |j| {
            var sum: T = A[i * n + j];

            for (0..j) |k| {
                sum -= A[i * n + k] * A[j * n + k];
            }

            if (i == j) {
                if (sum <= 0.0) return error.NotPositiveDefinite;
                A[i * n + i] = @sqrt(sum);
            } else {
                A[i * n + j] = sum / A[j * n + j];
            }
        }
        // Zero out upper triangular part
        for (i + 1..n) |j| {
            A[i * n + j] = 0.0;
        }
    }
}

/// Forward substitution: solve Ly = b where L is lower triangular
///
/// Time: O(n²) | Space: O(1)
fn forwardSubstitution(comptime T: type, L: []const T, b: []const T, y: []T, n: usize) !void {
    for (0..n) |i| {
        var sum: T = b[i];
        for (0..i) |j| {
            sum -= L[i * n + j] * y[j];
        }
        y[i] = sum / L[i * n + i];
    }
}

/// Box-Muller transform: generate standard normal from uniform [0,1)
///
/// Time: O(1)
fn boxMuller(comptime T: type, rng: anytype) T {
    const uniform1 = rng.random().float(T);
    const uniform2 = rng.random().float(T);
    return @sqrt(-2.0 * @log(uniform1)) * @cos(2.0 * math.pi * uniform2);
}

// ============================================================================
// Tests
// ============================================================================

test "MVN: 1D (reduces to univariate normal)" {
    const allocator = std.testing.allocator;

    const mean = [_]f64{2.0};
    const cov = [_]f64{4.0}; // variance = 4, std = 2

    var mvn = try MultivariateNormal(f64).init(allocator, &mean, &cov);
    defer mvn.deinit();

    // Test PDF at mean (maximum)
    const x_mean = [_]f64{2.0};
    const pdf_mean = try mvn.pdf(&x_mean);
    const expected_pdf = 1.0 / @sqrt(2.0 * math.pi * 4.0); // 1/(2π·4)^(1/2)
    try std.testing.expectApproxEqAbs(expected_pdf, pdf_mean, 1e-10);

    // Test PDF at mean + 1 std
    const x_1std = [_]f64{4.0};
    const pdf_1std = try mvn.pdf(&x_1std);
    // Mahalanobis² = (4-2)²/4 = 1
    const expected_1std = @exp(-0.5) / @sqrt(2.0 * math.pi * 4.0);
    try std.testing.expectApproxEqAbs(expected_1std, pdf_1std, 1e-10);

    // Test Mahalanobis distance
    const dist = try mvn.mahalanobis(&x_1std);
    try std.testing.expectApproxEqAbs(1.0, dist, 1e-10); // (4-2)/2 = 1
}

test "MVN: 2D independent (diagonal covariance)" {
    const allocator = std.testing.allocator;

    const mean = [_]f64{ 0.0, 0.0 };
    const cov = [_]f64{
        1.0, 0.0,
        0.0, 4.0,
    }; // independent, σ₁=1, σ₂=2

    var mvn = try MultivariateNormal(f64).init(allocator, &mean, &cov);
    defer mvn.deinit();

    // Test PDF at mean
    const x_mean = [_]f64{ 0.0, 0.0 };
    const pdf_mean = try mvn.pdf(&x_mean);
    const expected = 1.0 / (2.0 * math.pi * @sqrt(4.0)); // 1/(2π√4)
    try std.testing.expectApproxEqAbs(expected, pdf_mean, 1e-10);

    // Test logpdf (more stable)
    const log_pdf = try mvn.logpdf(&x_mean);
    try std.testing.expectApproxEqAbs(@log(pdf_mean), log_pdf, 1e-10);

    // Test Mahalanobis at (1, 2) → distance² = 1/1 + 4/4 = 2
    const x = [_]f64{ 1.0, 2.0 };
    const dist = try mvn.mahalanobis(&x);
    try std.testing.expectApproxEqAbs(@sqrt(2.0), dist, 1e-10);
}

test "MVN: 2D correlated" {
    const allocator = std.testing.allocator;

    const mean = [_]f64{ 1.0, 2.0 };
    // Σ = [2 1]  (positive definite: eigenvalues 3, 1)
    //     [1 2]
    const cov = [_]f64{
        2.0, 1.0,
        1.0, 2.0,
    };

    var mvn = try MultivariateNormal(f64).init(allocator, &mean, &cov);
    defer mvn.deinit();

    // Verify Cholesky: L = [√2    0  ]
    //                      [1/√2  √1.5]
    try std.testing.expectApproxEqAbs(@sqrt(2.0), mvn.chol[0], 1e-10);
    try std.testing.expectApproxEqAbs(0.0, mvn.chol[1], 1e-10);
    try std.testing.expectApproxEqAbs(1.0 / @sqrt(2.0), mvn.chol[2], 1e-10);
    try std.testing.expectApproxEqAbs(@sqrt(1.5), mvn.chol[3], 1e-10);

    // Test PDF at mean
    const x_mean = [_]f64{ 1.0, 2.0 };
    const pdf_mean = try mvn.pdf(&x_mean);
    // det(Σ) = 2×2 - 1×1 = 3
    const expected = 1.0 / (2.0 * math.pi * @sqrt(3.0));
    try std.testing.expectApproxEqAbs(expected, pdf_mean, 1e-10);

    // Test at different point
    const x = [_]f64{ 2.0, 3.0 };
    const pdf_x = try mvn.pdf(&x);
    try std.testing.expect(pdf_x > 0.0);
    try std.testing.expect(pdf_x < pdf_mean); // farther from mean → lower PDF
}

test "MVN: 3D general case" {
    const allocator = std.testing.allocator;

    const mean = [_]f64{ 0.0, 0.0, 0.0 };
    // SPD matrix with moderate condition number
    const cov = [_]f64{
        4.0, 1.0, 0.5,
        1.0, 3.0, 0.8,
        0.5, 0.8, 2.0,
    };

    var mvn = try MultivariateNormal(f64).init(allocator, &mean, &cov);
    defer mvn.deinit();

    // Test PDF is positive
    const x = [_]f64{ 1.0, -1.0, 0.5 };
    const pdf_val = try mvn.pdf(&x);
    try std.testing.expect(pdf_val > 0.0);

    // Test logpdf consistency
    const log_pdf = try mvn.logpdf(&x);
    try std.testing.expectApproxEqAbs(@log(pdf_val), log_pdf, 1e-9);

    // Test Mahalanobis is positive
    const dist = try mvn.mahalanobis(&x);
    try std.testing.expect(dist > 0.0);
}

test "MVN: sampling produces correct mean (Monte Carlo)" {
    const allocator = std.testing.allocator;

    const mean = [_]f64{ 2.0, -1.0 };
    const cov = [_]f64{
        1.0, 0.0,
        0.0, 1.0,
    }; // independent standard normals shifted

    var mvn = try MultivariateNormal(f64).init(allocator, &mean, &cov);
    defer mvn.deinit();

    var rng = std.Random.DefaultPrng.init(42);

    // Generate 10000 samples and compute empirical mean
    const n_samples = 10000;
    var sum1: f64 = 0.0;
    var sum2: f64 = 0.0;

    for (0..n_samples) |_| {
        const sample = try mvn.sample(allocator, &rng);
        defer allocator.free(sample);
        sum1 += sample[0];
        sum2 += sample[1];
    }

    const emp_mean1 = sum1 / @as(f64, @floatFromInt(n_samples));
    const emp_mean2 = sum2 / @as(f64, @floatFromInt(n_samples));

    // With 10k samples, empirical mean should be close to true mean
    try std.testing.expectApproxEqAbs(2.0, emp_mean1, 0.05); // ~3σ error
    try std.testing.expectApproxEqAbs(-1.0, emp_mean2, 0.05);
}

test "MVN: dimension mismatch errors" {
    const allocator = std.testing.allocator;

    const mean = [_]f64{ 0.0, 0.0 };
    const cov = [_]f64{
        1.0, 0.0,
        0.0, 1.0,
    };

    var mvn = try MultivariateNormal(f64).init(allocator, &mean, &cov);
    defer mvn.deinit();

    // PDF with wrong dimension
    const x_wrong = [_]f64{0.0};
    try std.testing.expectError(error.DimensionMismatch, mvn.pdf(&x_wrong));

    // Mahalanobis with wrong dimension
    try std.testing.expectError(error.DimensionMismatch, mvn.mahalanobis(&x_wrong));
}

test "MVN: not positive definite error" {
    const allocator = std.testing.allocator;

    const mean = [_]f64{ 0.0, 0.0 };
    const cov_bad = [_]f64{
        1.0,  2.0,
        2.0, -1.0, // negative eigenvalue → not PD
    };

    try std.testing.expectError(error.NotPositiveDefinite, MultivariateNormal(f64).init(allocator, &mean, &cov_bad));
}

test "MVN: singular matrix error" {
    const allocator = std.testing.allocator;

    const mean = [_]f64{ 0.0, 0.0 };
    const cov_singular = [_]f64{
        1.0, 1.0,
        1.0, 1.0, // rank 1, det = 0
    };

    try std.testing.expectError(error.NotPositiveDefinite, MultivariateNormal(f64).init(allocator, &mean, &cov_singular));
}

test "MVN: f32 precision" {
    const allocator = std.testing.allocator;

    const mean = [_]f32{ 0.0, 0.0 };
    const cov = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
    };

    var mvn = try MultivariateNormal(f32).init(allocator, &mean, &cov);
    defer mvn.deinit();

    const x = [_]f32{ 0.0, 0.0 };
    const pdf_val = try mvn.pdf(&x);
    const expected = 1.0 / (2.0 * math.pi);
    try std.testing.expectApproxEqAbs(expected, pdf_val, 1e-6);
}

test "MVN: memory safety (10 iterations)" {
    const allocator = std.testing.allocator;

    for (0..10) |_| {
        const mean = [_]f64{ 1.0, 2.0 };
        const cov = [_]f64{
            2.0, 0.5,
            0.5, 3.0,
        };

        var mvn = try MultivariateNormal(f64).init(allocator, &mean, &cov);
        defer mvn.deinit();

        const x = [_]f64{ 1.5, 2.5 };
        _ = try mvn.pdf(&x);
        _ = try mvn.logpdf(&x);
        _ = try mvn.mahalanobis(&x);

        var rng = std.Random.DefaultPrng.init(12345);
        const sample = try mvn.sample(allocator, &rng);
        defer allocator.free(sample);
    }
}
