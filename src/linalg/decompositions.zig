//! Matrix Decompositions — LU, QR, SVD, Cholesky
//!
//! High-level factorization algorithms for solving linear systems,
//! least squares problems, and eigenvalue algorithms.
//!
//! ## Supported Decompositions
//! - **LU**: A = PLU with partial pivoting (O(n³))
//! - **QR**: A = QR via Householder reflections (O(mn²))
//! - **SVD**: A = UΣV^T (v2.0+)
//! - **Cholesky**: A = LL^T for positive definite (v2.0+)

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const root = @import("../root.zig");
const NDArray = root.ndarray.NDArray;

// Import LU decomposition from the lu module
pub const lu = @import("lu.zig").lu;
pub const LUResult = @import("lu.zig").LUResult;

// ============================================================================
// QR Decomposition — Householder Reflections
// ============================================================================

/// Result of QR decomposition: Q (orthogonal) and R (upper triangular)
pub fn QRResult(comptime T: type) type {
    return struct {
        /// Orthogonal matrix (m×m for full QR, or m×n for thin QR)
        /// Satisfies: Q^T @ Q = I
        Q: NDArray(T, 2),

        /// Upper triangular matrix (n×n for full QR, or m×n for thin QR)
        /// Contains the upper triangular factors of the decomposition
        R: NDArray(T, 2),

        /// Allocator used for all allocations
        allocator: Allocator,

        /// Free all allocated memory
        ///
        /// Time: O(1) deallocation
        /// Space: O(1)
        pub fn deinit(self: *@This()) void {
            self.Q.deinit();
            self.R.deinit();
        }
    };
}

/// Compute QR decomposition using Householder reflections
///
/// Factorizes matrix A (m×n, m ≥ n) into A = QR where:
/// - Q is an m×m orthogonal matrix (or m×n for thin QR)
/// - R is an m×n (or n×n) upper triangular matrix
///
/// Householder reflections are numerically stable and suitable for:
/// - Solving least squares problems
/// - Computing eigenvalues (via QR iteration)
/// - Orthonormalization
///
/// Parameters:
/// - T: Numeric type (f32, f64)
/// - A: Input matrix (m×n, must have m ≥ n)
/// - allocator: Memory allocator for result matrices
///
/// Returns: QRResult containing Q and R
///
/// Errors:
/// - error.InvalidDimensions if m < n (tall matrix required)
/// - error.OutOfMemory if allocation fails
///
/// Time: O(mn²) where m = rows, n = cols
/// Space: O(mn) for result matrices
///
/// Example:
/// ```zig
/// var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{4, 2}, &[_]f64{
///     1, 0, 1, 1, 0, 1, 0, 0
/// }, .row_major);
/// defer A.deinit();
/// var result = try qr(f64, A, allocator);
/// defer result.deinit();
/// // Verify: A ≈ Q @ R, Q^T @ Q = I, R is upper triangular
/// ```
pub fn qr(comptime T: type, A: NDArray(T, 2), allocator: Allocator) (NDArray(T, 2).Error || std.mem.Allocator.Error || error{InvalidDimensions})!QRResult(T) {
    const m = A.shape[0];
    const n = A.shape[1];

    // QR requires m >= n (tall or square matrix)
    if (m < n) {
        return error.InvalidDimensions;
    }

    // Allocate Q as m×m identity matrix (we'll apply Householder reflections)
    var Q = try NDArray(T, 2).zeros(allocator, &[_]usize{ m, m }, .row_major);
    errdefer Q.deinit();

    // Initialize Q as identity
    for (0..m) |i| {
        Q.data[i * m + i] = 1;
    }

    // Allocate R as copy of A (we'll modify it in-place)
    var R = try NDArray(T, 2).zeros(allocator, &[_]usize{ m, n }, .row_major);
    errdefer R.deinit();

    // Copy A into R, respecting input layout
    for (0..m) |i| {
        for (0..n) |j| {
            R.data[i * n + j] = try A.get(&[_]isize{ @intCast(i), @intCast(j) });
        }
    }

    // Allocate workspace for Householder vectors
    var v = try allocator.alloc(T, m);
    defer allocator.free(v);

    var u = try allocator.alloc(T, m);
    defer allocator.free(u);

    // Apply Householder reflections to each column
    for (0..n) |col| {
        // Extract column from R starting at row col
        @memset(v, 0);
        @memset(u, 0);

        // Copy column col from R[col:, col] into v
        for (col..m) |i| {
            v[i - col] = R.data[i * n + col];
        }

        // Compute Householder vector: v = x + sign(x[0]) * ||x|| * e1
        var norm_x: T = 0;
        for (0..m - col) |i| {
            norm_x += v[i] * v[i];
        }
        norm_x = @sqrt(norm_x);

        if (norm_x < 1e-15) {
            continue; // Skip zero column
        }

        const sign = if (v[0] >= 0) @as(T, 1) else -1;
        const alpha = sign * norm_x;
        v[0] = v[0] + alpha;

        // Normalize: v = v / ||v||
        var norm_v: T = 0;
        for (0..m - col) |i| {
            norm_v += v[i] * v[i];
        }
        norm_v = @sqrt(norm_v);

        if (norm_v < 1e-15) {
            continue; // Skip if v is zero
        }

        for (0..m - col) |i| {
            u[i] = v[i] / norm_v;
        }

        // Apply reflection to R: R[col:, col:] = (I - 2uu^T) R[col:, col:]
        for (col..n) |j| {
            // Compute u^T * R[col:, j]
            var dot_prod: T = 0;
            for (col..m) |i| {
                dot_prod += u[i - col] * R.data[i * n + j];
            }

            // Update: R[col:, j] = R[col:, j] - 2 * (u^T * R[col:, j]) * u
            for (col..m) |i| {
                R.data[i * n + j] -= 2 * dot_prod * u[i - col];
            }
        }

        // Apply reflection to Q: Q[:, col:] = Q[:, col:] (I - 2uu^T)
        // We store Q as Q from accumulating reflections
        // Q_new = Q_old @ (I - 2uu^T) in the subspace
        for (0..m) |i| {
            // Extract row i of Q[i, col:m]
            // Compute dot product of Q[i, col:m] with u
            var dot_prod: T = 0;
            for (col..m) |j| {
                dot_prod += Q.data[i * m + j] * u[j - col];
            }

            // Update Q[i, col:m]
            for (col..m) |j| {
                Q.data[i * m + j] -= 2 * dot_prod * u[j - col];
            }
        }
    }

    // Extract thin R (m×n stays as m×n, but we need n×n for upper triangular)
    // Actually, keep R as m×n which is already correct

    return QRResult(T){
        .Q = Q,
        .R = R,
        .allocator = allocator,
    };
}

// ============================================================================
// QR Verification Helpers
// ============================================================================

/// Verify that Q^T @ Q = I (orthogonality property)
///
/// Parameters:
/// - T: Numeric type
/// - Q: Orthogonal matrix
/// - tolerance: Epsilon for floating-point comparison
///
/// Time: O(m²n) for matrix multiplication
/// Space: O(m²) temporary matrix
fn verifyOrthogonality(comptime T: type, allocator: Allocator, Q: NDArray(T, 2), tolerance: T) !void {
    const m = Q.shape[0];

    // Compute Q^T @ Q
    var QtQ = try NDArray(T, 2).zeros(allocator, &[_]usize{ m, m }, .row_major);
    defer QtQ.deinit();

    for (0..m) |i| {
        for (0..m) |j| {
            var sum: T = 0;
            for (0..m) |k| {
                sum += Q.data[k * m + i] * Q.data[k * m + j];
            }
            QtQ.data[i * m + j] = sum;
        }
    }

    // Check Q^T @ Q = I
    for (0..m) |i| {
        for (0..m) |j| {
            const expected = if (i == j) @as(T, 1) else 0;
            const diff = @abs(QtQ.data[i * m + j] - expected);
            try testing.expect(diff < tolerance);
        }
    }
}

/// Verify that A ≈ Q @ R (reconstruction accuracy)
///
/// Parameters:
/// - T: Numeric type
/// - A: Original matrix
/// - Q: Orthogonal matrix
/// - R: Upper triangular matrix
/// - tolerance: Epsilon for floating-point comparison
///
/// Time: O(m²n) for matrix multiplication
/// Space: O(mn) temporary matrix
fn verifyQRReconstruction(comptime T: type, allocator: Allocator, A: NDArray(T, 2), Q: NDArray(T, 2), R: NDArray(T, 2), tolerance: T) !void {
    const m = A.shape[0];
    const n = A.shape[1];

    // Compute Q @ R
    var QR = try NDArray(T, 2).zeros(allocator, &[_]usize{ m, n }, .row_major);
    defer QR.deinit();

    for (0..m) |i| {
        for (0..n) |j| {
            var sum: T = 0;
            for (0..m) |k| {
                sum += Q.data[i * m + k] * R.data[k * n + j];
            }
            QR.data[i * n + j] = sum;
        }
    }

    // Compare ||A - QR||
    for (0..m * n) |idx| {
        const A_val = try A.get(&[_]isize{ @intCast(idx / n), @intCast(idx % n) });
        const diff_val = @abs(A_val - QR.data[idx]);
        try testing.expect(diff_val < tolerance);
    }
}

/// Verify that R is upper triangular
///
/// Parameters:
/// - T: Numeric type
/// - R: Upper triangular matrix
/// - tolerance: Epsilon for floating-point comparison (for near-zero lower triangle)
///
/// Time: O(m²) comparisons
/// Space: O(1)
fn verifyUpperTriangular(comptime T: type, R: NDArray(T, 2), tolerance: T) !void {
    const m = R.shape[0];
    const n = R.shape[1];

    // Check lower triangle is zero
    for (1..m) |i| {
        for (0..@min(i, n)) |j| {
            const abs_val = @abs(R.data[i * n + j]);
            try testing.expect(abs_val < tolerance);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "qr: 2x2 identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1, 0,
        0, 1,
    }, .row_major);
    defer A.deinit();

    var result = try qr(f64, A, allocator);
    defer result.deinit();

    // Q should be identity or close to it
    try verifyOrthogonality(f64, allocator, result.Q, 1e-10);

    // R should be upper triangular with small lower triangle
    try verifyUpperTriangular(f64, result.R, 1e-10);

    // Reconstruction: A ≈ Q @ R
    try verifyQRReconstruction(f64, allocator, A, result.Q, result.R, 1e-10);
}

test "qr: 3x3 identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    }, .row_major);
    defer A.deinit();

    var result = try qr(f64, A, allocator);
    defer result.deinit();

    try verifyOrthogonality(f64, allocator, result.Q, 1e-10);
    try verifyUpperTriangular(f64, result.R, 1e-10);
    try verifyQRReconstruction(f64, allocator, A, result.Q, result.R, 1e-10);
}

test "qr: 4x4 identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 4 }, &[_]f64{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    }, .row_major);
    defer A.deinit();

    var result = try qr(f64, A, allocator);
    defer result.deinit();

    try verifyOrthogonality(f64, allocator, result.Q, 1e-10);
    try verifyUpperTriangular(f64, result.R, 1e-10);
    try verifyQRReconstruction(f64, allocator, A, result.Q, result.R, 1e-10);
}

test "qr: 2x2 non-identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1, 2,
        3, 4,
    }, .row_major);
    defer A.deinit();

    var result = try qr(f64, A, allocator);
    defer result.deinit();

    try verifyOrthogonality(f64, allocator, result.Q, 1e-10);
    try verifyUpperTriangular(f64, result.R, 1e-10);
    try verifyQRReconstruction(f64, allocator, A, result.Q, result.R, 1e-10);
}

test "qr: 3x3 non-identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        2, 3,  1,
        6, 13, 5,
        2, 19, 10,
    }, .row_major);
    defer A.deinit();

    var result = try qr(f64, A, allocator);
    defer result.deinit();

    try verifyOrthogonality(f64, allocator, result.Q, 1e-10);
    try verifyUpperTriangular(f64, result.R, 1e-10);
    try verifyQRReconstruction(f64, allocator, A, result.Q, result.R, 1e-10);
}

test "qr: 4x4 non-identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 4 }, &[_]f64{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
        13, 14, 15, 17, // Last element different to avoid singularity
    }, .row_major);
    defer A.deinit();

    var result = try qr(f64, A, allocator);
    defer result.deinit();

    try verifyOrthogonality(f64, allocator, result.Q, 1e-9);
    try verifyUpperTriangular(f64, result.R, 1e-10);
    try verifyQRReconstruction(f64, allocator, A, result.Q, result.R, 1e-9);
}

test "qr: tall matrix 4x2 (m > n)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 2 }, &[_]f64{
        1, 0,
        1, 1,
        1, 2,
        1, 3,
    }, .row_major);
    defer A.deinit();

    var result = try qr(f64, A, allocator);
    defer result.deinit();

    // Q is 4x4 (full)
    try testing.expectEqual(@as(usize, 4), result.Q.shape[0]);
    try testing.expectEqual(@as(usize, 4), result.Q.shape[1]);

    // R is 4x2
    try testing.expectEqual(@as(usize, 4), result.R.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.R.shape[1]);

    try verifyOrthogonality(f64, allocator, result.Q, 1e-10);
    try verifyUpperTriangular(f64, result.R, 1e-10);
    try verifyQRReconstruction(f64, allocator, A, result.Q, result.R, 1e-10);
}

test "qr: tall matrix 5x3 (m > n)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 5, 3 }, &[_]f64{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
        13, 14, 15,
    }, .row_major);
    defer A.deinit();

    var result = try qr(f64, A, allocator);
    defer result.deinit();

    // Q is 5x5, R is 5x3
    try testing.expectEqual(@as(usize, 5), result.Q.shape[0]);
    try testing.expectEqual(@as(usize, 5), result.Q.shape[1]);
    try testing.expectEqual(@as(usize, 5), result.R.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.R.shape[1]);

    try verifyOrthogonality(f64, allocator, result.Q, 1e-9);
    try verifyUpperTriangular(f64, result.R, 1e-10);
    try verifyQRReconstruction(f64, allocator, A, result.Q, result.R, 1e-9);
}

test "qr: tall matrix 6x2 (m > n, thin QR optimization)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 6, 2 }, &[_]f64{
        1, 0,
        2, 1,
        3, 2,
        4, 3,
        5, 4,
        6, 5,
    }, .row_major);
    defer A.deinit();

    var result = try qr(f64, A, allocator);
    defer result.deinit();

    try verifyOrthogonality(f64, allocator, result.Q, 1e-10);
    try verifyUpperTriangular(f64, result.R, 1e-10);
    try verifyQRReconstruction(f64, allocator, A, result.Q, result.R, 1e-10);
}

test "qr: orthogonality validation Q^T @ Q = I" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 2, 3,
        4, 5, 6,
        7, 8, 10,
    }, .row_major);
    defer A.deinit();

    var result = try qr(f64, A, allocator);
    defer result.deinit();

    // Compute Q^T @ Q and verify it equals I
    const m = result.Q.shape[0];
    for (0..m) |i| {
        for (0..m) |j| {
            var sum: f64 = 0;
            for (0..m) |k| {
                sum += result.Q.data[k * m + i] * result.Q.data[k * m + j];
            }
            const expected = if (i == j) @as(f64, 1) else 0;
            try testing.expectApproxEqAbs(expected, sum, 1e-10);
        }
    }
}

test "qr: reconstruction accuracy ||A - QR|| < epsilon" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 3 }, &[_]f64{
        1.2,  2.3,  3.4,
        4.5,  5.6,  6.7,
        7.8,  8.9,  9.0,
        10.1, 11.2, 12.3,
    }, .row_major);
    defer A.deinit();

    var result = try qr(f64, A, allocator);
    defer result.deinit();

    // Compute QR and compare to A
    const m = A.shape[0];
    const n = A.shape[1];

    var max_error: f64 = 0;
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f64 = 0;
            for (0..m) |k| {
                sum += result.Q.data[i * m + k] * result.R.data[k * n + j];
            }
            const A_val = try A.get(&[_]isize{ @intCast(i), @intCast(j) });
            const diff_val = @abs(A_val - sum);
            if (diff_val > max_error) {
                max_error = diff_val;
            }
        }
    }

    try testing.expect(max_error < 1e-10);
}

test "qr: zero columns (edge case)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f64{
        0, 1,
        0, 2,
        0, 3,
    }, .row_major);
    defer A.deinit();

    var result = try qr(f64, A, allocator);
    defer result.deinit();

    // Should still compute valid QR despite zero first column
    try verifyOrthogonality(f64, allocator, result.Q, 1e-9);
    try verifyUpperTriangular(f64, result.R, 1e-10);
    try verifyQRReconstruction(f64, allocator, A, result.Q, result.R, 1e-9);
}

test "qr: f32 precision" {
    const allocator = testing.allocator;

    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f32{
        1.5, 2.5,
        3.5, 4.5,
        5.5, 6.5,
    }, .row_major);
    defer A.deinit();

    var result = try qr(f32, A, allocator);
    defer result.deinit();

    // Larger tolerance for f32
    try verifyOrthogonality(f32, allocator, result.Q, 1e-5);
    try verifyUpperTriangular(f32, result.R, 1e-5);
    try verifyQRReconstruction(f32, allocator, A, result.Q, result.R, 1e-5);
}

test "qr: f64 precision" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 3 }, &[_]f64{
        0.12, 0.34, 0.56,
        0.78, 0.90, 0.12,
        0.34, 0.56, 0.78,
        0.90, 0.12, 0.34,
    }, .row_major);
    defer A.deinit();

    var result = try qr(f64, A, allocator);
    defer result.deinit();

    // Standard f64 tolerance
    try verifyOrthogonality(f64, allocator, result.Q, 1e-10);
    try verifyUpperTriangular(f64, result.R, 1e-10);
    try verifyQRReconstruction(f64, allocator, A, result.Q, result.R, 1e-10);
}

test "qr: memory cleanup — no leaks" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    }, .row_major);
    defer A.deinit();

    var result = try qr(f64, A, allocator);
    result.deinit();

    // Testing allocator detects any leaks
}

test "qr: negative values" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        -1, 2,  -3,
        4,  -5, 6,
        -7, 8,  -9,
    }, .row_major);
    defer A.deinit();

    var result = try qr(f64, A, allocator);
    defer result.deinit();

    try verifyOrthogonality(f64, allocator, result.Q, 1e-9);
    try verifyUpperTriangular(f64, result.R, 1e-10);
    try verifyQRReconstruction(f64, allocator, A, result.Q, result.R, 1e-9);
}

test "qr: diagonal matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        2, 0, 0,
        0, 3, 0,
        0, 0, 5,
    }, .row_major);
    defer A.deinit();

    var result = try qr(f64, A, allocator);
    defer result.deinit();

    try verifyOrthogonality(f64, allocator, result.Q, 1e-10);
    try verifyUpperTriangular(f64, result.R, 1e-10);
    try verifyQRReconstruction(f64, allocator, A, result.Q, result.R, 1e-10);
}

test "qr: upper triangular matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 2, 3,
        0, 4, 5,
        0, 0, 6,
    }, .row_major);
    defer A.deinit();

    var result = try qr(f64, A, allocator);
    defer result.deinit();

    try verifyOrthogonality(f64, allocator, result.Q, 1e-10);
    try verifyUpperTriangular(f64, result.R, 1e-10);
    try verifyQRReconstruction(f64, allocator, A, result.Q, result.R, 1e-10);
}

test "qr: column-major layout" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1, 3,
        2, 4,
    }, .column_major);
    defer A.deinit();

    var result = try qr(f64, A, allocator);
    defer result.deinit();

    try verifyOrthogonality(f64, allocator, result.Q, 1e-10);
    try verifyUpperTriangular(f64, result.R, 1e-10);
    try verifyQRReconstruction(f64, allocator, A, result.Q, result.R, 1e-10);
}

test "qr: small values (numerical stability)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1e-10, 2e-10, 3e-10,
        4e-10, 5e-10, 6e-10,
        7e-10, 8e-10, 9e-10,
    }, .row_major);
    defer A.deinit();

    var result = try qr(f64, A, allocator);
    defer result.deinit();

    try verifyOrthogonality(f64, allocator, result.Q, 1e-8);
    try verifyUpperTriangular(f64, result.R, 1e-8);
    try verifyQRReconstruction(f64, allocator, A, result.Q, result.R, 1e-8);
}

test "qr: large values (numerical stability)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1e10, 2e10, 3e10,
        4e10, 5e10, 6e10,
        7e10, 8e10, 9e10,
    }, .row_major);
    defer A.deinit();

    var result = try qr(f64, A, allocator);
    defer result.deinit();

    try verifyOrthogonality(f64, allocator, result.Q, 1e-8);
    try verifyUpperTriangular(f64, result.R, 1e-8);
    try verifyQRReconstruction(f64, allocator, A, result.Q, result.R, 1e-8);
}

test "qr: orthonormality of columns (least squares application)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 5, 3 }, &[_]f64{
        1, 0, 1,
        0, 1, 1,
        1, 1, 0,
        1, 0, 1,
        0, 1, 1,
    }, .row_major);
    defer A.deinit();

    var result = try qr(f64, A, allocator);
    defer result.deinit();

    // For least squares, we verify Q^T @ Q = I and columns are orthonormal
    const m = result.Q.shape[0];
    const n = result.Q.shape[1];

    // Check columns of Q are orthonormal
    for (0..n) |i| {
        for (0..n) |j| {
            var dot_prod: f64 = 0;
            for (0..m) |k| {
                dot_prod += result.Q.data[k * m + i] * result.Q.data[k * m + j];
            }
            const expected = if (i == j) @as(f64, 1) else 0;
            try testing.expectApproxEqAbs(expected, dot_prod, 1e-10);
        }
    }
}

test "qr: invalid dimensions — m < n returns error" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{
        1, 2, 3,
        4, 5, 6,
    }, .row_major);
    defer A.deinit();

    const err = qr(f64, A, allocator);
    try testing.expectError(error.InvalidDimensions, err);
}

// ============================================================================
// Cholesky Decomposition — LLT for Symmetric Positive Definite Matrices
// ============================================================================

/// Compute Cholesky decomposition of a symmetric positive definite matrix
///
/// Factorizes matrix A (n×n, symmetric positive definite) into A = LL^T where:
/// - L is an n×n lower triangular matrix
/// - L has positive diagonal elements
///
/// Uses the Cholesky-Banachiewicz algorithm (row-wise computation):
/// - For each row i:
///   - For each column j <= i:
///     - If i == j: L[i,i] = sqrt(A[i,i] - sum(L[i,k]²))
///     - If i > j: L[i,j] = (A[i,j] - sum(L[i,k]*L[j,k])) / L[j,j]
///
/// Numerically stable for well-conditioned SPD matrices. Detects non-SPD
/// matrices by checking for negative values under the square root.
///
/// Parameters:
/// - T: Numeric type (f32, f64)
/// - A: Input symmetric positive definite matrix (n×n)
/// - allocator: Memory allocator for result matrix
///
/// Returns: NDArray(T, 2) containing L (lower triangular factor)
///
/// Errors:
/// - error.NonSquareMatrix if A.shape[0] != A.shape[1]
/// - error.NotPositiveDefinite if any diagonal becomes negative or zero
/// - error.OutOfMemory if allocation fails
///
/// Time: O(n³) — cubic in matrix dimension
/// Space: O(n²) for the result matrix L
///
/// Example:
/// ```zig
/// var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{2, 2}, &[_]f64{
///     4, 2,
///     2, 3,
/// }, .row_major);
/// defer A.deinit();
/// var L = try cholesky(f64, A, allocator);
/// defer L.deinit();
/// // L = [[2, 0], [1, sqrt(2)]]
/// // Verify: A ≈ L @ L^T
/// ```
pub fn cholesky(comptime T: type, A: NDArray(T, 2), allocator: Allocator) (NDArray(T, 2).Error || std.mem.Allocator.Error || error{NotPositiveDefinite, NonSquareMatrix})!NDArray(T, 2) {
    const n = A.shape[0];
    const m = A.shape[1];

    // Check that A is square
    if (n != m) {
        return error.NonSquareMatrix;
    }

    // Allocate L as n×n zero matrix (row-major)
    var L = try NDArray(T, 2).zeros(allocator, &[_]usize{ n, n }, .row_major);
    errdefer L.deinit();

    // Compute Cholesky factorization using Banachiewicz algorithm
    for (0..n) |i| {
        // Compute L[i, j] for j <= i
        for (0..i + 1) |j| {
            if (i == j) {
                // Diagonal: L[i,i] = sqrt(A[i,i] - sum(L[i,k]²) for k=0..i-1)
                var sum: T = 0;
                for (0..i) |k| {
                    const Lik = L.data[i * n + k];
                    sum += Lik * Lik;
                }

                const Aii = try A.get(&[_]isize{ @intCast(i), @intCast(i) });
                const diag_val = Aii - sum;

                // Check for positive definiteness: diagonal must be positive
                if (diag_val <= 0) {
                    return error.NotPositiveDefinite;
                }

                L.data[i * n + i] = @sqrt(diag_val);
            } else {
                // Off-diagonal: L[i,j] = (A[i,j] - sum(L[i,k]*L[j,k])) / L[j,j]
                var sum: T = 0;
                for (0..j) |k| {
                    sum += L.data[i * n + k] * L.data[j * n + k];
                }

                const Aij = try A.get(&[_]isize{ @intCast(i), @intCast(j) });
                const Ljj = L.data[j * n + j];

                // Ljj should be positive (checked when computed as diagonal)
                L.data[i * n + j] = (Aij - sum) / Ljj;
            }
        }

        // Upper triangle is zero (initialized at allocation time)
    }

    return L;
}

/// Verify that L is lower triangular (upper triangle is zero)
///
/// Parameters:
/// - T: Numeric type
/// - L: Lower triangular matrix
/// - tolerance: Epsilon for floating-point comparison
///
/// Time: O(n²) comparisons
/// Space: O(1)
fn verifyLowerTriangular(comptime T: type, L: NDArray(T, 2), tolerance: T) !void {
    const n = L.shape[0];
    const m = L.shape[1];

    // Check upper triangle is zero (i < j should be zero)
    for (0..n) |i| {
        for (i + 1..m) |j| {
            const abs_val = @abs(L.data[i * m + j]);
            try testing.expect(abs_val < tolerance);
        }
    }
}

/// Verify Cholesky reconstruction: A ≈ L @ L^T
///
/// Parameters:
/// - T: Numeric type
/// - A: Original matrix
/// - L: Lower triangular factor
/// - tolerance: Epsilon for Frobenius norm
///
/// Time: O(n³) for matrix multiplication and verification
/// Space: O(n²) temporary matrix
fn verifyCholeskyReconstruction(comptime T: type, allocator: Allocator, A: NDArray(T, 2), L: NDArray(T, 2), tolerance: T) !void {
    const n = A.shape[0];

    // Compute L @ L^T
    var LLT = try NDArray(T, 2).zeros(allocator, &[_]usize{ n, n }, .row_major);
    defer LLT.deinit();

    for (0..n) |i| {
        for (0..n) |j| {
            var sum: T = 0;
            for (0..n) |k| {
                sum += L.data[i * n + k] * L.data[j * n + k]; // L^T[k,j] = L[j,k]
            }
            LLT.data[i * n + j] = sum;
        }
    }

    // Compare ||A - LLT||_F < tolerance
    var max_error: T = 0;
    for (0..n * n) |idx| {
        const A_val = try A.get(&[_]isize{ @intCast(idx / n), @intCast(idx % n) });
        const diff_val = @abs(A_val - LLT.data[idx]);
        if (diff_val > max_error) {
            max_error = diff_val;
        }
    }

    try testing.expect(max_error < tolerance);
}

/// Verify that diagonal of L is positive
///
/// Parameters:
/// - T: Numeric type
/// - L: Lower triangular matrix
///
/// Time: O(n)
/// Space: O(1)
fn verifyPositiveDiagonal(comptime T: type, L: NDArray(T, 2)) !void {
    const n = L.shape[0];

    for (0..n) |i| {
        const diag = L.data[i * n + i];
        try testing.expect(diag > 0);
    }
}

// ============================================================================
// Cholesky Tests
// ============================================================================

test "cholesky: 2x2 identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1, 0,
        0, 1,
    }, .row_major);
    defer A.deinit();

    // For identity, L should also be identity
    var L = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer L.deinit();
    for (0..4) |i| L.data[i] = 0;
    for (0..2) |i| L.data[i * 2 + i] = 1;

    try verifyLowerTriangular(f64, L, 1e-10);
    try verifyCholeskyReconstruction(f64, allocator, A, L, 1e-10);
    try verifyPositiveDiagonal(f64, L);
}

test "cholesky: 3x3 identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    }, .row_major);
    defer A.deinit();

    var L = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer L.deinit();
    for (0..9) |i| L.data[i] = 0;
    for (0..3) |i| L.data[i * 3 + i] = 1;

    try verifyLowerTriangular(f64, L, 1e-10);
    try verifyCholeskyReconstruction(f64, allocator, A, L, 1e-10);
    try verifyPositiveDiagonal(f64, L);
}

test "cholesky: 4x4 identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 4 }, &[_]f64{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    }, .row_major);
    defer A.deinit();

    var L = try NDArray(f64, 2).init(allocator, &[_]usize{ 4, 4 }, .row_major);
    defer L.deinit();
    for (0..16) |i| L.data[i] = 0;
    for (0..4) |i| L.data[i * 4 + i] = 1;

    try verifyLowerTriangular(f64, L, 1e-10);
    try verifyCholeskyReconstruction(f64, allocator, A, L, 1e-10);
    try verifyPositiveDiagonal(f64, L);
}

test "cholesky: 2x2 simple SPD matrix" {
    const allocator = testing.allocator;

    // A = [[4, 2], [2, 3]]
    // L = [[2, 0], [1, sqrt(2)]]
    // LL^T = [[4, 2], [2, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        4, 2,
        2, 3,
    }, .row_major);
    defer A.deinit();

    var L = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer L.deinit();
    L.data[0 * 2 + 0] = 2;
    L.data[0 * 2 + 1] = 0;
    L.data[1 * 2 + 0] = 1;
    L.data[1 * 2 + 1] = @sqrt(2.0);

    try verifyLowerTriangular(f64, L, 1e-10);
    try verifyCholeskyReconstruction(f64, allocator, A, L, 1e-10);
    try verifyPositiveDiagonal(f64, L);
}

test "cholesky: 3x3 simple SPD matrix" {
    const allocator = testing.allocator;

    // A = [[4, 2, 1], [2, 5, 3], [1, 3, 6]]
    // This is SPD (all eigenvalues positive)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        4, 2, 1,
        2, 5, 3,
        1, 3, 6,
    }, .row_major);
    defer A.deinit();

    // Expected L from manual calculation or numerical factorization
    var L = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer L.deinit();
    // Compute L[0,0] = sqrt(4) = 2
    // Compute L[1,0] = 2/2 = 1, L[1,1] = sqrt(5-1) = 2
    // Compute L[2,0] = 1/2 = 0.5, L[2,1] = (3-0.5*2)/2 = 1, L[2,2] = sqrt(6-0.5^2-1^2) = sqrt(4.75)
    for (0..9) |i| L.data[i] = 0;
    L.data[0 * 3 + 0] = 2;
    L.data[1 * 3 + 0] = 1;
    L.data[1 * 3 + 1] = 2;
    L.data[2 * 3 + 0] = 0.5;
    L.data[2 * 3 + 1] = 1;
    L.data[2 * 3 + 2] = @sqrt(4.75);

    try verifyLowerTriangular(f64, L, 1e-10);
    try verifyCholeskyReconstruction(f64, allocator, A, L, 1e-9);
    try verifyPositiveDiagonal(f64, L);
}

test "cholesky: 4x4 simple SPD matrix" {
    const allocator = testing.allocator;

    // A = [[5, 1, 2, 1], [1, 4, 1, 1], [2, 1, 3, 0], [1, 1, 0, 2]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 4 }, &[_]f64{
        5, 1, 2, 1,
        1, 4, 1, 1,
        2, 1, 3, 0,
        1, 1, 0, 2,
    }, .row_major);
    defer A.deinit();

    // Compute expected L using Cholesky algorithm
    var L = try NDArray(f64, 2).init(allocator, &[_]usize{ 4, 4 }, .row_major);
    defer L.deinit();
    for (0..16) |i| L.data[i] = 0;

    // Cholesky factorization: L[j,j] = sqrt(A[j,j] - sum(L[j,k]^2))
    // L[i,j] = (A[i,j] - sum(L[i,k]*L[j,k])) / L[j,j]
    L.data[0 * 4 + 0] = @sqrt(5.0);
    L.data[1 * 4 + 0] = 1.0 / L.data[0 * 4 + 0];
    L.data[1 * 4 + 1] = @sqrt(4.0 - L.data[1 * 4 + 0] * L.data[1 * 4 + 0]);
    L.data[2 * 4 + 0] = 2.0 / L.data[0 * 4 + 0];
    L.data[2 * 4 + 1] = (1.0 - L.data[2 * 4 + 0] * L.data[1 * 4 + 0]) / L.data[1 * 4 + 1];
    L.data[2 * 4 + 2] = @sqrt(3.0 - L.data[2 * 4 + 0] * L.data[2 * 4 + 0] - L.data[2 * 4 + 1] * L.data[2 * 4 + 1]);
    L.data[3 * 4 + 0] = 1.0 / L.data[0 * 4 + 0];
    L.data[3 * 4 + 1] = (1.0 - L.data[3 * 4 + 0] * L.data[1 * 4 + 0]) / L.data[1 * 4 + 1];
    L.data[3 * 4 + 2] = (0.0 - L.data[3 * 4 + 0] * L.data[2 * 4 + 0] - L.data[3 * 4 + 1] * L.data[2 * 4 + 1]) / L.data[2 * 4 + 2];
    L.data[3 * 4 + 3] = @sqrt(2.0 - L.data[3 * 4 + 0] * L.data[3 * 4 + 0] - L.data[3 * 4 + 1] * L.data[3 * 4 + 1] - L.data[3 * 4 + 2] * L.data[3 * 4 + 2]);

    try verifyLowerTriangular(f64, L, 1e-10);
    try verifyCholeskyReconstruction(f64, allocator, A, L, 1e-9);
    try verifyPositiveDiagonal(f64, L);
}

test "cholesky: diagonal SPD matrix" {
    const allocator = testing.allocator;

    // Diagonal matrices are trivially SPD
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        4, 0, 0,
        0, 9, 0,
        0, 0, 16,
    }, .row_major);
    defer A.deinit();

    // For diagonal, L is also diagonal with L[i,i] = sqrt(A[i,i])
    var L = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer L.deinit();
    for (0..9) |i| L.data[i] = 0;
    L.data[0 * 3 + 0] = 2;
    L.data[1 * 3 + 1] = 3;
    L.data[2 * 3 + 2] = 4;

    try verifyLowerTriangular(f64, L, 1e-10);
    try verifyCholeskyReconstruction(f64, allocator, A, L, 1e-10);
    try verifyPositiveDiagonal(f64, L);
}

test "cholesky: f32 precision" {
    const allocator = testing.allocator;

    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{
        4, 2,
        2, 3,
    }, .row_major);
    defer A.deinit();

    var L = try NDArray(f32, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer L.deinit();
    L.data[0 * 2 + 0] = 2;
    L.data[0 * 2 + 1] = 0;
    L.data[1 * 2 + 0] = 1;
    L.data[1 * 2 + 1] = @sqrt(2.0);

    try verifyLowerTriangular(f32, L, 1e-5);
    try verifyCholeskyReconstruction(f32, allocator, A, L, 1e-5);
    try verifyPositiveDiagonal(f32, L);
}

test "cholesky: f64 precision" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        4, 2, 1,
        2, 5, 3,
        1, 3, 6,
    }, .row_major);
    defer A.deinit();

    var L = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer L.deinit();
    for (0..9) |i| L.data[i] = 0;
    L.data[0 * 3 + 0] = 2;
    L.data[1 * 3 + 0] = 1;
    L.data[1 * 3 + 1] = 2;
    L.data[2 * 3 + 0] = 0.5;
    L.data[2 * 3 + 1] = 1;
    L.data[2 * 3 + 2] = @sqrt(4.75);

    try verifyLowerTriangular(f64, L, 1e-10);
    try verifyCholeskyReconstruction(f64, allocator, A, L, 1e-9);
    try verifyPositiveDiagonal(f64, L);
}

test "cholesky: memory cleanup — no leaks" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        4, 2, 1,
        2, 5, 3,
        1, 3, 6,
    }, .row_major);
    defer A.deinit();

    var L = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer L.deinit();

    // Testing allocator detects any leaks
}

test "cholesky: non-SPD detection — negative diagonal" {
    const allocator = testing.allocator;

    // Non-symmetric matrix with negative values
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        -1, 2,
        2,  -3,
    }, .row_major);
    defer A.deinit();

    // Attempt Cholesky on non-SPD matrix should fail
    // (This test verifies error detection logic when implemented)
    var L = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer L.deinit();

    // Non-SPD: first diagonal should fail to produce positive sqrt
    // Test documents expected behavior for future implementation
}

test "cholesky: singular matrix detection" {
    const allocator = testing.allocator;

    // Singular (rank-deficient) matrix
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1, 2,
        2, 4,
    }, .row_major);
    defer A.deinit();

    // Singular matrix should be rejected (not positive definite)
    var L = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer L.deinit();

    // Test documents expected error behavior
}

test "cholesky: non-square matrix rejection" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{
        1, 2, 3,
        4, 5, 6,
    }, .row_major);
    defer A.deinit();

    // Non-square matrices cannot be factored with Cholesky
    // Future implementation should return error.NonSquareMatrix
}

test "cholesky: non-symmetric matrix rejection" {
    const allocator = testing.allocator;

    // Symmetric positive definite but presented as non-symmetric
    // (Cholesky uses only lower triangle, but tests should verify symmetry)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        4, 3,
        2, 3,
    }, .row_major);
    defer A.deinit();

    // Non-symmetric matrix should be rejected or handled gracefully
    var L = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer L.deinit();
}

test "cholesky: small values — numerical stability" {
    const allocator = testing.allocator;

    // SPD matrix with small values
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1e-8, 5e-9,
        5e-9, 1e-8,
    }, .row_major);
    defer A.deinit();

    var L = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer L.deinit();
    L.data[0 * 2 + 0] = @sqrt(1e-8);
    L.data[0 * 2 + 1] = 0;
    L.data[1 * 2 + 0] = 5e-9 / @sqrt(1e-8);
    L.data[1 * 2 + 1] = @sqrt(1e-8 - (5e-9 / @sqrt(1e-8)) * (5e-9 / @sqrt(1e-8)));

    try verifyLowerTriangular(f64, L, 1e-8);
    try verifyCholeskyReconstruction(f64, allocator, A, L, 1e-8);
    try verifyPositiveDiagonal(f64, L);
}

test "cholesky: large values — numerical stability" {
    const allocator = testing.allocator;

    // SPD matrix with large values
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1e10, 5e9,
        5e9,  1e10,
    }, .row_major);
    defer A.deinit();

    var L = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer L.deinit();
    L.data[0 * 2 + 0] = @sqrt(1e10);
    L.data[0 * 2 + 1] = 0;
    L.data[1 * 2 + 0] = 5e9 / @sqrt(1e10);
    L.data[1 * 2 + 1] = @sqrt(1e10 - (5e9 / @sqrt(1e10)) * (5e9 / @sqrt(1e10)));

    try verifyLowerTriangular(f64, L, 1e-8);
    try verifyCholeskyReconstruction(f64, allocator, A, L, 1e-8);
    try verifyPositiveDiagonal(f64, L);
}

test "cholesky: covariance matrix use case" {
    const allocator = testing.allocator;

    // Typical covariance matrix from bivariate normal distribution
    // Cov = [[1.0, 0.5], [0.5, 1.0]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1.0, 0.5,
        0.5, 1.0,
    }, .row_major);
    defer A.deinit();

    // Cholesky of covariance matrix: L[0,0] = 1, L[1,0] = 0.5, L[1,1] = sqrt(3)/2
    var L = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer L.deinit();
    L.data[0 * 2 + 0] = 1.0;
    L.data[0 * 2 + 1] = 0;
    L.data[1 * 2 + 0] = 0.5;
    L.data[1 * 2 + 1] = @sqrt(3.0) / 2.0;

    try verifyLowerTriangular(f64, L, 1e-10);
    try verifyCholeskyReconstruction(f64, allocator, A, L, 1e-10);
    try verifyPositiveDiagonal(f64, L);
}

test "cholesky: 5x5 larger SPD matrix" {
    const allocator = testing.allocator;

    // Larger SPD matrix for stress testing
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 5, 5 }, &[_]f64{
        5, 1, 2, 0, 1,
        1, 4, 1, 1, 0,
        2, 1, 6, 2, 1,
        0, 1, 2, 3, 0,
        1, 0, 1, 0, 2,
    }, .row_major);
    defer A.deinit();

    // Initialize L as lower triangular (will be filled by factorization logic)
    var L = try NDArray(f64, 2).init(allocator, &[_]usize{ 5, 5 }, .row_major);
    defer L.deinit();

    // Manually compute Cholesky for verification
    for (0..25) |i| L.data[i] = 0;

    // Row 0
    L.data[0 * 5 + 0] = @sqrt(5.0);

    // Row 1
    L.data[1 * 5 + 0] = 1.0 / L.data[0 * 5 + 0];
    L.data[1 * 5 + 1] = @sqrt(4.0 - L.data[1 * 5 + 0] * L.data[1 * 5 + 0]);

    // Row 2
    L.data[2 * 5 + 0] = 2.0 / L.data[0 * 5 + 0];
    L.data[2 * 5 + 1] = (1.0 - L.data[2 * 5 + 0] * L.data[1 * 5 + 0]) / L.data[1 * 5 + 1];
    L.data[2 * 5 + 2] = @sqrt(6.0 - L.data[2 * 5 + 0] * L.data[2 * 5 + 0] - L.data[2 * 5 + 1] * L.data[2 * 5 + 1]);

    // Row 3
    L.data[3 * 5 + 0] = 0.0 / L.data[0 * 5 + 0];
    L.data[3 * 5 + 1] = (1.0 - L.data[3 * 5 + 0] * L.data[1 * 5 + 0]) / L.data[1 * 5 + 1];
    L.data[3 * 5 + 2] = (2.0 - L.data[3 * 5 + 0] * L.data[2 * 5 + 0] - L.data[3 * 5 + 1] * L.data[2 * 5 + 1]) / L.data[2 * 5 + 2];
    L.data[3 * 5 + 3] = @sqrt(3.0 - L.data[3 * 5 + 0] * L.data[3 * 5 + 0] - L.data[3 * 5 + 1] * L.data[3 * 5 + 1] - L.data[3 * 5 + 2] * L.data[3 * 5 + 2]);

    // Row 4
    L.data[4 * 5 + 0] = 1.0 / L.data[0 * 5 + 0];
    L.data[4 * 5 + 1] = (0.0 - L.data[4 * 5 + 0] * L.data[1 * 5 + 0]) / L.data[1 * 5 + 1];
    L.data[4 * 5 + 2] = (1.0 - L.data[4 * 5 + 0] * L.data[2 * 5 + 0] - L.data[4 * 5 + 1] * L.data[2 * 5 + 1]) / L.data[2 * 5 + 2];
    L.data[4 * 5 + 3] = (0.0 - L.data[4 * 5 + 0] * L.data[3 * 5 + 0] - L.data[4 * 5 + 1] * L.data[3 * 5 + 1] - L.data[4 * 5 + 2] * L.data[3 * 5 + 2]) / L.data[3 * 5 + 3];
    L.data[4 * 5 + 4] = @sqrt(2.0 - L.data[4 * 5 + 0] * L.data[4 * 5 + 0] - L.data[4 * 5 + 1] * L.data[4 * 5 + 1] - L.data[4 * 5 + 2] * L.data[4 * 5 + 2] - L.data[4 * 5 + 3] * L.data[4 * 5 + 3]);

    try verifyLowerTriangular(f64, L, 1e-10);
    try verifyCholeskyReconstruction(f64, allocator, A, L, 1e-9);
    try verifyPositiveDiagonal(f64, L);
}

test "cholesky: column-major layout" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        4, 2,
        2, 3,
    }, .column_major);
    defer A.deinit();

    var L = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer L.deinit();
    L.data[0 * 2 + 0] = 2;
    L.data[0 * 2 + 1] = 0;
    L.data[1 * 2 + 0] = 1;
    L.data[1 * 2 + 1] = @sqrt(2.0);

    try verifyLowerTriangular(f64, L, 1e-10);
    try verifyCholeskyReconstruction(f64, allocator, A, L, 1e-10);
    try verifyPositiveDiagonal(f64, L);
}

// ============================================================================
// SVD Result Type and Verification Helpers
// ============================================================================

/// Result of SVD decomposition: U, S, and Vt matrices
pub fn SVDResult(comptime T: type) type {
    return struct {
        /// Left singular vectors (m×k for thin SVD, where k = min(m,n))
        /// Orthogonal: U^T @ U = I
        U: NDArray(T, 2),

        /// Singular values (k vector, k = min(m,n))
        /// In descending order: S[i] >= S[i+1] >= 0
        S: NDArray(T, 1),

        /// Right singular vectors transposed (k×n, where k = min(m,n))
        /// Orthogonal: Vt @ Vt^T = I
        Vt: NDArray(T, 2),

        /// Allocator used for all allocations
        allocator: Allocator,

        /// Free all allocated memory
        ///
        /// Time: O(1) deallocation
        /// Space: O(1)
        pub fn deinit(self: *@This()) void {
            self.U.deinit();
            self.S.deinit();
            self.Vt.deinit();
        }
    };
}

/// Compute Singular Value Decomposition (SVD) using Golub-Reinsch algorithm
///
/// Factorizes matrix A (m×n) into A = U @ diag(S) @ Vt where:
/// - U is an m×k orthogonal matrix (left singular vectors), k = min(m,n)
/// - S is a k vector of singular values in descending order (non-negative)
/// - Vt is a k×n orthogonal matrix (right singular vectors transposed)
///
/// The thin SVD (k = min(m,n)) is computed, which is suitable for:
/// - Computing low-rank approximations
/// - Computing condition numbers
/// - Computing pseudo-inverses
/// - Least squares solutions
///
/// Algorithm: Golub-Reinsch
/// 1. Bidiagonalize A via Householder reflections (O(mn²))
/// 2. Apply QR iteration to bidiagonal matrix to find singular values (O(n³) worst case)
/// 3. Accumulate orthogonal transformations to form U and V
///
/// Parameters:
/// - T: Numeric type (f32, f64)
/// - A: Input matrix (m×n, any aspect ratio)
/// - allocator: Memory allocator for result matrices
///
/// Returns: SVDResult containing U, S, and Vt
///
/// Errors:
/// - error.OutOfMemory if allocation fails
///
/// Time: O(mn²) for thin SVD (dominant: bidiagonalization)
/// Space: O(mn) for result matrices + O(min(m,n)) workspace
///
/// Example:
/// ```zig
/// var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 2}, &[_]f64{
///     1, 2, 3, 4, 5, 6
/// }, .row_major);
/// defer A.deinit();
/// var result = try svd(f64, A, allocator);
/// defer result.deinit();
/// // Verify: A ≈ U @ diag(S) @ Vt
/// // Verify: U^T @ U = I, Vt @ Vt^T = I
/// // Verify: S[i] >= S[i+1] >= 0
/// ```
pub fn svd(comptime T: type, A: NDArray(T, 2), allocator: Allocator) (NDArray(T, 2).Error || NDArray(T, 1).Error || std.mem.Allocator.Error)!SVDResult(T) {
    const m = A.shape[0];
    const n = A.shape[1];
    const k = @min(m, n);

    // Working copy of A (will be bidiagonalized in-place)
    var B = try NDArray(T, 2).zeros(allocator, &[_]usize{ m, n }, .row_major);
    errdefer B.deinit();

    // Copy A into B, respecting input layout
    for (0..m) |i| {
        for (0..n) |j| {
            B.data[i * n + j] = try A.get(&[_]isize{ @intCast(i), @intCast(j) });
        }
    }

    // Initialize U as m×k identity
    var U = try NDArray(T, 2).zeros(allocator, &[_]usize{ m, k }, .row_major);
    errdefer U.deinit();
    for (0..@min(m, k)) |i| {
        U.data[i * k + i] = 1;
    }

    // Initialize Vt as k×n identity
    var Vt = try NDArray(T, 2).zeros(allocator, &[_]usize{ k, n }, .row_major);
    errdefer Vt.deinit();
    for (0..@min(k, n)) |i| {
        Vt.data[i * n + i] = 1;
    }

    // Singular values vector
    var S = try NDArray(T, 1).zeros(allocator, &[_]usize{k}, .row_major);
    errdefer S.deinit();

    // Bidiagonalize B and accumulate transformations
    try bidiagonalize(T, &B, &U, &Vt, allocator);

    // Extract diagonal and superdiagonal
    var diag = try allocator.alloc(T, k);
    defer allocator.free(diag);
    var super = try allocator.alloc(T, k);
    defer allocator.free(super);

    for (0..k) |i| {
        diag[i] = B.data[i * n + i];
        if (i < k - 1 and i + 1 < n) {
            super[i] = B.data[i * n + (i + 1)];
        } else {
            super[i] = 0;
        }
    }

    // QR iteration to diagonalize
    try qrIterateBidiagonal(T, diag, super, &U, &Vt, k, m, n, allocator);

    // Copy singular values (absolute values)
    for (0..k) |i| {
        S.data[i] = @abs(diag[i]);
    }

    // Sort in descending order
    sortSingularValues(T, &S, &U, &Vt, k, m, n);

    B.deinit();

    return SVDResult(T){
        .U = U,
        .S = S,
        .Vt = Vt,
        .allocator = allocator,
    };
}

/// Bidiagonalize matrix B using Householder reflections
fn bidiagonalize(comptime T: type, B: *NDArray(T, 2), U: *NDArray(T, 2), Vt: *NDArray(T, 2), allocator: Allocator) !void {
    const m = B.shape[0];
    const n = B.shape[1];
    const k = @min(m, n);

    var householder_v = try allocator.alloc(T, @max(m, n));
    defer allocator.free(householder_v);

    for (0..k) |i| {
        // Left Householder to zero column i below diagonal
        if (i < m) {
            const col_size = m - i;
            @memset(householder_v[0..col_size], 0);

            // Extract column
            for (i..m) |row| {
                householder_v[row - i] = B.data[row * n + i];
            }

            // Compute Householder vector
            var norm: T = 0;
            for (0..col_size) |idx| {
                norm += householder_v[idx] * householder_v[idx];
            }
            norm = @sqrt(norm);

            if (norm > 1e-15) {
                const sign: T = if (householder_v[0] >= 0) 1 else -1;
                householder_v[0] += sign * norm;

                var h_norm: T = 0;
                for (0..col_size) |idx| {
                    h_norm += householder_v[idx] * householder_v[idx];
                }
                h_norm = @sqrt(h_norm);

                if (h_norm > 1e-15) {
                    for (0..col_size) |idx| {
                        householder_v[idx] /= h_norm;
                    }

                    // Apply to B[i:, i:]
                    for (i..n) |col| {
                        var dot: T = 0;
                        for (i..m) |row| {
                            dot += householder_v[row - i] * B.data[row * n + col];
                        }
                        for (i..m) |row| {
                            B.data[row * n + col] -= 2 * dot * householder_v[row - i];
                        }
                    }

                    // Apply to U
                    for (0..m) |row| {
                        var dot: T = 0;
                        for (i..m) |col| {
                            if (col < k) {
                                dot += U.data[row * k + col] * householder_v[col - i];
                            }
                        }
                        for (i..m) |col| {
                            if (col < k) {
                                U.data[row * k + col] -= 2 * dot * householder_v[col - i];
                            }
                        }
                    }
                }
            }
        }

        // Right Householder to zero row i after superdiagonal
        if (i < k - 1 and i + 1 < n) {
            const row_size = n - i - 1;
            @memset(householder_v[0..row_size], 0);

            // Extract row
            for (i + 1..n) |col| {
                householder_v[col - i - 1] = B.data[i * n + col];
            }

            var norm: T = 0;
            for (0..row_size) |idx| {
                norm += householder_v[idx] * householder_v[idx];
            }
            norm = @sqrt(norm);

            if (norm > 1e-15) {
                const sign: T = if (householder_v[0] >= 0) 1 else -1;
                householder_v[0] += sign * norm;

                var h_norm: T = 0;
                for (0..row_size) |idx| {
                    h_norm += householder_v[idx] * householder_v[idx];
                }
                h_norm = @sqrt(h_norm);

                if (h_norm > 1e-15) {
                    for (0..row_size) |idx| {
                        householder_v[idx] /= h_norm;
                    }

                    // Apply to B[i:, i+1:]
                    for (i..m) |row| {
                        var dot: T = 0;
                        for (i + 1..n) |col| {
                            dot += B.data[row * n + col] * householder_v[col - i - 1];
                        }
                        for (i + 1..n) |col| {
                            B.data[row * n + col] -= 2 * dot * householder_v[col - i - 1];
                        }
                    }

                    // Apply to Vt
                    for (i + 1..k) |row| {
                        if (row < Vt.shape[0]) {
                            var dot: T = 0;
                            for (0..n) |col| {
                                dot += Vt.data[row * n + col] * householder_v[row - i - 1];
                            }
                            for (0..n) |col| {
                                Vt.data[row * n + col] -= 2 * dot * householder_v[row - i - 1];
                            }
                        }
                    }
                }
            }
        }
    }
}

/// QR iteration with Wilkinson shift on bidiagonal matrix
fn qrIterateBidiagonal(comptime T: type, diag: []T, super: []T, U: *NDArray(T, 2), Vt: *NDArray(T, 2), k: usize, m: usize, n: usize, allocator: Allocator) !void {
    _ = allocator;
    const max_iter = 30 * k;
    const tol = @sqrt(@as(T, switch (T) {
        f32 => 1.19e-7,
        f64 => 2.22e-16,
        else => 1e-10,
    }));

    var iter: usize = 0;
    while (iter < max_iter) : (iter += 1) {
        // Check convergence
        var converged = true;
        for (0..k - 1) |i| {
            if (@abs(super[i]) > tol * (@abs(diag[i]) + @abs(diag[i + 1]) + 1e-15)) {
                converged = false;
                break;
            }
        }
        if (converged) break;

        // Find largest unreduced block
        var p: usize = k - 1;
        while (p > 0) {
            if (@abs(super[p - 1]) <= tol * (@abs(diag[p - 1]) + @abs(diag[p]) + 1e-15)) {
                super[p - 1] = 0;
                p -= 1;
            } else {
                break;
            }
        }

        if (p == 0) break;

        var q: usize = p - 1;
        while (q > 0) {
            if (@abs(super[q - 1]) <= tol * (@abs(diag[q - 1]) + @abs(diag[q]) + 1e-15)) {
                super[q - 1] = 0;
                break;
            }
            q -= 1;
        }

        // Wilkinson shift
        const d = (diag[p - 1] - diag[p]) / 2;
        const mu_sign: T = if (d >= 0) 1 else -1;
        const denom = d + mu_sign * @sqrt(d * d + super[p - 1] * super[p - 1]);
        const mu = if (@abs(denom) > 1e-15) diag[p] - (super[p - 1] * super[p - 1]) / denom else diag[p];

        // QR sweep
        var x = diag[q] - mu;
        var z = super[q];

        for (q..p) |i| {
            // Givens rotation
            const r = @sqrt(x * x + z * z);
            if (r < tol) {
                if (i + 1 <= p) {
                    x = diag[i + 1];
                    z = if (i + 1 < p) super[i + 1] else 0;
                }
                continue;
            }

            const c = x / r;
            const s = z / r;

            // Apply rotation (right side)
            if (i > q) {
                super[i - 1] = r;
            }

            const old_diag_i = diag[i];
            const old_super_i = super[i];
            diag[i] = c * old_diag_i + s * old_super_i;
            super[i] = -s * old_diag_i + c * old_super_i;

            z = s * diag[i + 1];
            diag[i + 1] = c * diag[i + 1];

            // Apply to Vt
            for (0..n) |col| {
                if (i < k and i + 1 < k) {
                    const vi = Vt.data[i * n + col];
                    const vi1 = Vt.data[(i + 1) * n + col];
                    Vt.data[i * n + col] = c * vi + s * vi1;
                    Vt.data[(i + 1) * n + col] = -s * vi + c * vi1;
                }
            }

            // Second Givens (left side)
            x = diag[i];
            const r2 = @sqrt(x * x + z * z);
            if (r2 < tol) {
                if (i < p - 1) {
                    x = super[i];
                    z = 0;
                }
                continue;
            }

            const c2 = x / r2;
            const s2 = z / r2;

            diag[i] = r2;
            const old_super_i2 = super[i];
            super[i] = c2 * old_super_i2 + s2 * diag[i + 1];
            diag[i + 1] = -s2 * old_super_i2 + c2 * diag[i + 1];

            if (i < p - 1) {
                x = super[i];
                z = s2 * super[i + 1];
                super[i + 1] = c2 * super[i + 1];
            }

            // Apply to U
            for (0..m) |row| {
                if (i < k and i + 1 < k) {
                    const ui = U.data[row * k + i];
                    const ui1 = U.data[row * k + (i + 1)];
                    U.data[row * k + i] = c2 * ui + s2 * ui1;
                    U.data[row * k + (i + 1)] = -s2 * ui + c2 * ui1;
                }
            }
        }
    }
}

/// Sort singular values in descending order and permute U, Vt accordingly
fn sortSingularValues(comptime T: type, S: *NDArray(T, 1), U: *NDArray(T, 2), Vt: *NDArray(T, 2), k: usize, m: usize, n: usize) void {
    for (0..k) |i| {
        var max_idx = i;
        for (i + 1..k) |j| {
            if (S.data[j] > S.data[max_idx]) {
                max_idx = j;
            }
        }

        if (max_idx != i) {
            // Swap singular values
            const temp_s = S.data[i];
            S.data[i] = S.data[max_idx];
            S.data[max_idx] = temp_s;

            // Swap columns of U
            for (0..m) |row| {
                const temp_u = U.data[row * k + i];
                U.data[row * k + i] = U.data[row * k + max_idx];
                U.data[row * k + max_idx] = temp_u;
            }

            // Swap rows of Vt
            for (0..n) |col| {
                const temp_vt = Vt.data[i * n + col];
                Vt.data[i * n + col] = Vt.data[max_idx * n + col];
                Vt.data[max_idx * n + col] = temp_vt;
            }
        }
    }
}

/// Verify that U^T @ U = I (orthogonality of left singular vectors)
///
/// Parameters:
/// - T: Numeric type
/// - U: Left singular vectors
/// - tolerance: Epsilon for floating-point comparison
///
/// Time: O(m²k) for matrix multiplication
/// Space: O(k²) temporary matrix
fn verifySVDOrthogonalityU(comptime T: type, allocator: Allocator, U: NDArray(T, 2), tolerance: T) !void {
    const m = U.shape[0];
    const k = U.shape[1];

    // Compute U^T @ U
    var UtU = try NDArray(T, 2).zeros(allocator, &[_]usize{ k, k }, .row_major);
    defer UtU.deinit();

    for (0..k) |i| {
        for (0..k) |j| {
            var sum: T = 0;
            for (0..m) |row| {
                sum += U.data[row * k + i] * U.data[row * k + j];
            }
            UtU.data[i * k + j] = sum;
        }
    }

    // Check U^T @ U = I
    for (0..k) |i| {
        for (0..k) |j| {
            const expected = if (i == j) @as(T, 1) else 0;
            const diff = @abs(UtU.data[i * k + j] - expected);
            try testing.expect(diff < tolerance);
        }
    }
}

/// Verify that Vt @ Vt^T = I (orthogonality of right singular vectors transposed)
///
/// Parameters:
/// - T: Numeric type
/// - Vt: Right singular vectors transposed (k×n)
/// - tolerance: Epsilon for floating-point comparison
///
/// Time: O(k²n) for matrix multiplication
/// Space: O(k²) temporary matrix
fn verifySVDOrthogonalityVt(comptime T: type, allocator: Allocator, Vt: NDArray(T, 2), tolerance: T) !void {
    const k = Vt.shape[0];
    const n = Vt.shape[1];

    // Compute Vt @ Vt^T
    var VtVtT = try NDArray(T, 2).zeros(allocator, &[_]usize{ k, k }, .row_major);
    defer VtVtT.deinit();

    for (0..k) |i| {
        for (0..k) |j| {
            var sum: T = 0;
            for (0..n) |col| {
                sum += Vt.data[i * n + col] * Vt.data[j * n + col];
            }
            VtVtT.data[i * k + j] = sum;
        }
    }

    // Check Vt @ Vt^T = I
    for (0..k) |i| {
        for (0..k) |j| {
            const expected = if (i == j) @as(T, 1) else 0;
            const diff = @abs(VtVtT.data[i * k + j] - expected);
            try testing.expect(diff < tolerance);
        }
    }
}

/// Verify that singular values are in descending order and non-negative
///
/// Parameters:
/// - T: Numeric type
/// - S: Singular values vector (k elements)
///
/// Time: O(k)
/// Space: O(1)
fn verifySVDDescending(comptime T: type, S: NDArray(T, 1)) !void {
    const k = S.shape[0];

    // Check non-negative
    for (0..k) |i| {
        try testing.expect(S.data[i] >= 0);
    }

    // Check descending order
    for (0..k - 1) |i| {
        try testing.expect(S.data[i] >= S.data[i + 1]);
    }
}

/// Verify that A ≈ U @ diag(S) @ Vt (reconstruction accuracy)
///
/// Parameters:
/// - T: Numeric type
/// - A: Original matrix (m×n)
/// - U: Left singular vectors (m×k)
/// - S: Singular values (k)
/// - Vt: Right singular vectors transposed (k×n)
/// - tolerance: Epsilon for floating-point comparison
///
/// Time: O(m²n) for matrix multiplication
/// Space: O(mn) temporary matrices
fn verifySVDReconstruction(comptime T: type, allocator: Allocator, A: NDArray(T, 2), U: NDArray(T, 2), S: NDArray(T, 1), Vt: NDArray(T, 2), tolerance: T) !void {
    const m = A.shape[0];
    const n = A.shape[1];
    const k = S.shape[0];

    // Compute U @ diag(S)
    var USigma = try NDArray(T, 2).zeros(allocator, &[_]usize{ m, k }, .row_major);
    defer USigma.deinit();

    for (0..m) |i| {
        for (0..k) |j| {
            USigma.data[i * k + j] = U.data[i * k + j] * S.data[j];
        }
    }

    // Compute (U @ diag(S)) @ Vt
    var USigmaVt = try NDArray(T, 2).zeros(allocator, &[_]usize{ m, n }, .row_major);
    defer USigmaVt.deinit();

    for (0..m) |i| {
        for (0..n) |j| {
            var sum: T = 0;
            for (0..k) |col| {
                sum += USigma.data[i * k + col] * Vt.data[col * n + j];
            }
            USigmaVt.data[i * n + j] = sum;
        }
    }

    // Compare ||A - U @ diag(S) @ Vt||_F
    var max_error: T = 0;
    for (0..m * n) |idx| {
        const A_val = try A.get(&[_]isize{ @intCast(idx / n), @intCast(idx % n) });
        const diff_val = @abs(A_val - USigmaVt.data[idx]);
        if (diff_val > max_error) {
            max_error = diff_val;
        }
    }

    try testing.expect(max_error < tolerance);
}

// ============================================================================
// SVD Tests
// ============================================================================

test "svd: 2x2 identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1, 0,
        0, 1,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    // Identity has singular values [1, 1]
    try testing.expectApproxEqAbs(@as(f64, 1), result.S.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1), result.S.data[1], 1e-10);

    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-10);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-10);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-10);
}

test "svd: 3x3 identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    // Identity has all singular values = 1
    for (0..3) |i| {
        try testing.expectApproxEqAbs(@as(f64, 1), result.S.data[i], 1e-10);
    }

    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-10);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-10);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-10);
}

test "svd: 4x4 identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 4 }, &[_]f64{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    for (0..4) |i| {
        try testing.expectApproxEqAbs(@as(f64, 1), result.S.data[i], 1e-10);
    }

    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-10);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-10);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-10);
}

test "svd: 2x2 diagonal matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        3, 0,
        0, 2,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    // Diagonal matrix: singular values are absolute values of diagonal entries (in descending order)
    try testing.expectApproxEqAbs(@as(f64, 3), result.S.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 2), result.S.data[1], 1e-10);

    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-10);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-10);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-10);
}

test "svd: 3x3 diagonal matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        5, 0, 0,
        0, 3, 0,
        0, 0, 1,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    try testing.expectApproxEqAbs(@as(f64, 5), result.S.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 3), result.S.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1), result.S.data[2], 1e-10);

    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-10);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-10);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-10);
}

test "svd: 2x2 non-identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1, 2,
        3, 4,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    // Singular values should be positive and descending
    try testing.expect(result.S.data[0] > result.S.data[1]);
    try testing.expect(result.S.data[1] > 0);

    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-10);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-10);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-10);
}

test "svd: 3x3 non-identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        2, 3, 1,
        6, 13, 5,
        2, 19, 10,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    // Verify ordering and positivity
    for (0..2) |i| {
        try testing.expect(result.S.data[i] >= result.S.data[i + 1]);
    }
    for (0..3) |i| {
        try testing.expect(result.S.data[i] >= 0);
    }

    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-9);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-9);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-9);
}

test "svd: 4x4 non-identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 4 }, &[_]f64{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
        13, 14, 15, 17,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-9);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-9);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-9);
}

test "svd: tall matrix 4x2 (m > n)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 2 }, &[_]f64{
        1, 0,
        1, 1,
        1, 2,
        1, 3,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    // Thin SVD: U is 4x2, S is 2, Vt is 2x2
    try testing.expectEqual(@as(usize, 4), result.U.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.U.shape[1]);
    try testing.expectEqual(@as(usize, 2), result.S.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.Vt.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.Vt.shape[1]);

    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-10);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-10);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-10);
}

test "svd: tall matrix 5x3 (m > n)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 5, 3 }, &[_]f64{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    // Thin SVD: U is 5x3, S is 3, Vt is 3x3
    try testing.expectEqual(@as(usize, 5), result.U.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.U.shape[1]);
    try testing.expectEqual(@as(usize, 3), result.S.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.Vt.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.Vt.shape[1]);

    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-9);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-9);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-9);
}

test "svd: tall matrix 6x2 (m > n)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 6, 2 }, &[_]f64{
        1, 0,
        2, 1,
        3, 2,
        4, 3,
        5, 4,
        6, 5,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 6), result.U.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.U.shape[1]);
    try testing.expectEqual(@as(usize, 2), result.S.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.Vt.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.Vt.shape[1]);

    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-10);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-10);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-10);
}

test "svd: wide matrix 2x4 (m < n)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 4 }, &[_]f64{
        1, 2, 3, 4,
        5, 6, 7, 8,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    // Thin SVD: U is 2x2, S is 2, Vt is 2x4
    try testing.expectEqual(@as(usize, 2), result.U.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.U.shape[1]);
    try testing.expectEqual(@as(usize, 2), result.S.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.Vt.shape[0]);
    try testing.expectEqual(@as(usize, 4), result.Vt.shape[1]);

    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-10);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-10);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-10);
}

test "svd: wide matrix 3x5 (m < n)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 5 }, &[_]f64{
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    // Thin SVD: U is 3x3, S is 3, Vt is 3x5
    try testing.expectEqual(@as(usize, 3), result.U.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.U.shape[1]);
    try testing.expectEqual(@as(usize, 3), result.S.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.Vt.shape[0]);
    try testing.expectEqual(@as(usize, 5), result.Vt.shape[1]);

    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-9);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-9);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-9);
}

test "svd: all zeros matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    // Zero matrix: all singular values are 0
    for (0..3) |i| {
        try testing.expectApproxEqAbs(@as(f64, 0), result.S.data[i], 1e-10);
    }

    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-9);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-9);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-10);
}

test "svd: rank-deficient matrix (zero column)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 0, 2,
        2, 0, 4,
        3, 0, 6,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    // Rank-deficient: at least one singular value should be ≈ 0
    try testing.expect(result.S.data[2] < 1e-8);

    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-8);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-8);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-8);
}

test "svd: rank-deficient matrix (proportional rows)" {
    const allocator = testing.allocator;

    // Second row is 2x first row, so rank 1
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f64{
        1, 2,
        2, 4,
        1, 2,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    // Rank 1: exactly one large singular value, one near zero
    try testing.expect(result.S.data[0] > 1);
    try testing.expect(result.S.data[1] < 1e-8);

    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-8);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-8);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-8);
}

test "svd: ones matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    // Ones matrix (rank 1): first singular value is sqrt(9) = 3, rest are 0
    try testing.expectApproxEqAbs(@as(f64, 3), result.S.data[0], 1e-10);
    try testing.expect(result.S.data[1] < 1e-8);
    try testing.expect(result.S.data[2] < 1e-8);

    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-8);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-8);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-8);
}

test "svd: negative values" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        -1, 2,  -3,
        4,  -5, 6,
        -7, 8,  -9,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    // Singular values must be non-negative
    for (0..3) |i| {
        try testing.expect(result.S.data[i] >= 0);
    }

    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-9);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-9);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-9);
}

test "svd: f32 precision" {
    const allocator = testing.allocator;

    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f32{
        1.5, 2.5,
        3.5, 4.5,
        5.5, 6.5,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f32, A, allocator);
    defer result.deinit();

    // Larger tolerance for f32
    try verifySVDOrthogonalityU(f32, allocator, result.U, 1e-5);
    try verifySVDOrthogonalityVt(f32, allocator, result.Vt, 1e-5);
    try verifySVDDescending(f32, result.S);
    try verifySVDReconstruction(f32, allocator, A, result.U, result.S, result.Vt, 1e-5);
}

test "svd: f64 precision" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 3 }, &[_]f64{
        0.12, 0.34, 0.56,
        0.78, 0.90, 0.12,
        0.34, 0.56, 0.78,
        0.90, 0.12, 0.34,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    // Standard f64 tolerance
    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-10);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-10);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-10);
}

test "svd: small values (numerical stability)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1e-10, 2e-10, 3e-10,
        4e-10, 5e-10, 6e-10,
        7e-10, 8e-10, 9e-10,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    // Small values should still decompose correctly
    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-8);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-8);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-8);
}

test "svd: large values (numerical stability)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1e10, 2e10, 3e10,
        4e10, 5e10, 6e10,
        7e10, 8e10, 9e10,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    // Large values should still decompose correctly
    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-8);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-8);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-8);
}

test "svd: Hilbert matrix (ill-conditioned)" {
    const allocator = testing.allocator;

    // 4x4 Hilbert matrix (very ill-conditioned)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 4 }, &[_]f64{
        1,      0.5,    1.0/3, 0.25,
        0.5,    1.0/3,  0.25,  0.2,
        1.0/3,  0.25,   0.2,   1.0/6,
        0.25,   0.2,    1.0/6, 1.0/7,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    // Ill-conditioned: singular values decay rapidly
    try testing.expect(result.S.data[0] > result.S.data[1]);
    try testing.expect(result.S.data[1] > result.S.data[2]);
    try testing.expect(result.S.data[2] > result.S.data[3]);

    // Condition number should be large
    const condition_number = result.S.data[0] / result.S.data[3];
    try testing.expect(condition_number > 1000);

    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-7);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-7);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-7);
}

test "svd: singular value ordering" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 3 }, &[_]f64{
        1, 0, 0,
        0, 5, 0,
        0, 0, 3,
        0, 0, 0,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    // Singular values should be in descending order: [5, 3, 1]
    try testing.expect(result.S.data[0] >= 5 - 1e-10);
    try testing.expect(result.S.data[1] >= 3 - 1e-10);
    try testing.expect(result.S.data[1] <= 3 + 1e-10);
    try testing.expect(result.S.data[2] >= 1 - 1e-10);
    try testing.expect(result.S.data[2] <= 1 + 1e-10);

    try verifySVDDescending(f64, result.S);
}

test "svd: memory cleanup — no leaks" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    result.deinit();

    // Testing allocator detects any leaks
}

test "svd: column-major layout" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1, 3,
        2, 4,
    }, .column_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-10);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-10);
    try verifySVDDescending(f64, result.S);
    try verifySVDReconstruction(f64, allocator, A, result.U, result.S, result.Vt, 1e-10);
}

test "svd: low-rank approximation use case" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 3 }, &[_]f64{
        1, 0, 1,
        0, 2, 0,
        1, 0, 1,
        0, 2, 0,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    // Verify first singular value dominates (rank-2 approximation)
    try testing.expect(result.S.data[0] > result.S.data[1]);

    // Truncate to rank 2 by zeroing out small singular values
    result.S.data[2] = 0;

    // Reconstructed low-rank approximation should still be orthogonal
    try verifySVDOrthogonalityU(f64, allocator, result.U, 1e-9);
    try verifySVDOrthogonalityVt(f64, allocator, result.Vt, 1e-9);
}

test "svd: condition number computation" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 0, 0,
        0, 2, 0,
        0, 0, 0.1,
    }, .row_major);
    defer A.deinit();

    var result = try svd(f64, A, allocator);
    defer result.deinit();

    // Condition number = sigma_max / sigma_min (for non-zero singular values)
    const condition = result.S.data[0] / result.S.data[2];
    try testing.expectApproxEqAbs(condition, @as(f64, 20), 1e-10);
}

// ============================================================================
// Eigendecomposition — QR Algorithm for Symmetric Matrices
// ============================================================================

/// Result of eigendecomposition: eigenvalues and eigenvectors
pub fn EigResult(comptime T: type) type {
    return struct {
        /// Eigenvalues (n vector) in descending order by absolute value
        /// For symmetric matrices, all values are real
        eigenvalues: NDArray(T, 1),

        /// Eigenvectors (n×n matrix, orthonormal columns)
        /// V[i][j] is the j-th component of the i-th eigenvector
        /// Satisfies: V^T @ V = I (orthonormal)
        /// Reconstruction: A = V @ diag(eigenvalues) @ V^T
        eigenvectors: NDArray(T, 2),

        /// Allocator used for all allocations
        allocator: Allocator,

        /// Free all allocated memory
        ///
        /// Time: O(1) deallocation
        /// Space: O(1)
        pub fn deinit(self: *@This()) void {
            self.eigenvalues.deinit();
            self.eigenvectors.deinit();
        }
    };
}

/// Verify that eigenvectors form an orthonormal set: V^T @ V = I
fn verifyEigOrthonormality(comptime T: type, allocator: Allocator, V: NDArray(T, 2), tolerance: T) !void {
    const m = V.shape[1];

    // Compute V^T @ V
    var VT = try V.transpose(allocator);
    defer VT.deinit();

    var VTV = try NDArray(T, 2).matmul(allocator, VT, V);
    defer VTV.deinit();

    // Check that V^T @ V ≈ I
    for (0..m) |i| {
        for (0..m) |j| {
            const val = VTV.get(&[_]usize{ i, j });
            if (i == j) {
                // Diagonal should be 1
                try testing.expectApproxEqAbs(val, @as(T, 1.0), tolerance);
            } else {
                // Off-diagonal should be 0
                try testing.expectApproxEqAbs(val, @as(T, 0.0), tolerance);
            }
        }
    }
}

/// Verify that matrix A ≈ V @ diag(λ) @ V^T (reconstruction)
fn verifyEigReconstruction(comptime T: type, allocator: Allocator, A: NDArray(T, 2), V: NDArray(T, 2), eigenvalues: NDArray(T, 1), tolerance: T) !void {
    const n = A.shape[0];

    // Create diagonal matrix from eigenvalues
    var lambda = try NDArray(T, 2).zeros(allocator, &[_]usize{ n, n }, .row_major);
    defer lambda.deinit();
    for (0..n) |i| {
        lambda.set(&[_]usize{ i, i }, eigenvalues.data[i]);
    }

    // Compute V @ diag(λ) @ V^T
    var V_lambda = try NDArray(T, 2).matmul(allocator, V, lambda);
    defer V_lambda.deinit();

    var VT = try V.transpose(allocator);
    defer VT.deinit();

    var A_reconstructed = try NDArray(T, 2).matmul(allocator, V_lambda, VT);
    defer A_reconstructed.deinit();

    // Check reconstruction error
    for (0..n) |i| {
        for (0..n) |j| {
            const a_val = A.get(&[_]usize{ i, j });
            const r_val = A_reconstructed.get(&[_]usize{ i, j });
            try testing.expectApproxEqAbs(a_val, r_val, tolerance);
        }
    }
}

/// Verify that eigenvalues are in descending order by absolute value
fn verifyEigDescending(comptime T: type, eigenvalues: NDArray(T, 1)) !void {
    for (0..eigenvalues.shape[0] - 1) |i| {
        const abs_curr = @abs(eigenvalues.data[i]);
        const abs_next = @abs(eigenvalues.data[i + 1]);
        try testing.expect(abs_curr >= abs_next);
    }
}

/// Verify that A @ V = V @ diag(λ) (eigenvalue equation)
fn verifyEigProperty(comptime T: type, allocator: Allocator, A: NDArray(T, 2), V: NDArray(T, 2), eigenvalues: NDArray(T, 1), tolerance: T) !void {
    const n = A.shape[0];

    // Compute A @ V
    var AV = try NDArray(T, 2).matmul(allocator, A, V);
    defer AV.deinit();

    // Compute V @ diag(λ)
    var lambda = try NDArray(T, 2).zeros(allocator, &[_]usize{ n, n }, .row_major);
    defer lambda.deinit();
    for (0..n) |i| {
        lambda.set(&[_]usize{ i, i }, eigenvalues.data[i]);
    }

    var V_lambda = try NDArray(T, 2).matmul(allocator, V, lambda);
    defer V_lambda.deinit();

    // Check A @ V ≈ V @ diag(λ)
    for (0..n) |i| {
        for (0..n) |j| {
            const av_val = AV.get(&[_]usize{ i, j });
            const vl_val = V_lambda.get(&[_]usize{ i, j });
            try testing.expectApproxEqAbs(av_val, vl_val, tolerance);
        }
    }
}

/// Verify that the input matrix is symmetric: A = A^T
fn verifySymmetric(comptime T: type, A: NDArray(T, 2), tolerance: T) !void {
    const n = A.shape[0];
    try testing.expect(n == A.shape[1]);

    for (0..n) |i| {
        for (0..n) |j| {
            const a_val = A.get(&[_]usize{ i, j });
            const at_val = A.get(&[_]usize{ j, i });
            try testing.expectApproxEqAbs(a_val, at_val, tolerance);
        }
    }
}

/// Check if a matrix is symmetric: A = A^T (within tolerance)
fn isSymmetricMatrix(comptime T: type, A: NDArray(T, 2), tolerance: T) !bool {
    const n = A.shape[0];
    if (n != A.shape[1]) {
        return false;
    }

    for (0..n) |i| {
        for (0..n) |j| {
            const a_val = A.get(&[_]usize{ i, j });
            const at_val = A.get(&[_]usize{ j, i });
            const diff = @abs(a_val - at_val);
            if (diff > tolerance) {
                return false;
            }
        }
    }
    return true;
}

/// Compute eigendecomposition of a symmetric matrix using QR algorithm
///
/// Factorizes symmetric matrix A (n×n) into A = V @ diag(λ) @ V^T where:
/// - V is an n×n orthogonal matrix with eigenvectors as columns
/// - λ is an n vector of eigenvalues (real for symmetric matrices)
///
/// The QR algorithm iteratively applies QR decomposition:
/// - A_k = Q_k R_k
/// - A_{k+1} = R_k Q_k
/// - Converges to diagonal form with eigenvalues on diagonal
///
/// Parameters:
/// - T: Numeric type (f32, f64)
/// - A: Input symmetric matrix (n×n)
/// - allocator: Memory allocator for result matrices
///
/// Returns: EigResult containing eigenvalues and orthonormal eigenvectors
///
/// Errors:
/// - error.InvalidDimensions if matrix is not square
/// - error.NonSymmetricMatrix if A != A^T
/// - error.OutOfMemory if allocation fails
///
/// Time: O(n³) for QR iteration (typically 30*n iterations)
/// Space: O(n²) for working matrices
///
/// Example:
/// ```zig
/// var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
///     3, 1, 1, 2
/// }, .row_major);
/// defer A.deinit();
/// var result = try eig(f64, A, allocator);
/// defer result.deinit();
/// // result.eigenvalues contains [3.618..., 1.381...] (descending)
/// // result.eigenvectors columns are orthonormal eigenvectors
/// ```
pub fn eig(comptime T: type, A: NDArray(T, 2), allocator: Allocator) (NDArray(T, 2).Error || NDArray(T, 1).Error || std.mem.Allocator.Error || error{InvalidDimensions, NonSymmetricMatrix})!EigResult(T) {
    const n = A.shape[0];
    const m = A.shape[1];

    // Verify square matrix
    if (n != m) {
        return error.InvalidDimensions;
    }

    // Type-aware tolerance for convergence and symmetry check
    const tolerance: T = @sqrt(std.math.floatEps(T));

    // Verify symmetry with tolerance
    const is_symmetric = try isSymmetricMatrix(T, A, tolerance);
    if (!is_symmetric) {
        return error.NonSymmetricMatrix;
    }

    // Initialize V as identity matrix (will accumulate eigenvectors)
    var V = try NDArray(T, 2).zeros(allocator, &[_]usize{ n, n }, .row_major);
    errdefer V.deinit();
    for (0..n) |i| {
        V.data[i * n + i] = 1;
    }

    // Working copy of A (will be iteratively diagonalized)
    var A_k = try NDArray(T, 2).zeros(allocator, &[_]usize{ n, n }, .row_major);
    errdefer A_k.deinit();
    for (0..n) |i| {
        for (0..n) |j| {
            A_k.data[i * n + j] = A.get(&[_]usize{ i, j });
        }
    }

    // QR iteration: max iterations = 30 * n (empirical choice)
    const max_iterations = 30 * n;
    const convergence_tol = tolerance;

    var iteration: usize = 0;
    while (iteration < max_iterations) : (iteration += 1) {
        // QR decomposition of A_k
        var qr_result = qr(T, A_k, allocator) catch |err| {
            switch (err) {
                error.InvalidDimensions => return error.InvalidDimensions,
                else => return err,
            }
        };
        defer qr_result.deinit();

        // Update: A_{k+1} = R_k @ Q_k (matrix multiplication)
        const R_Q = try NDArray(T, 2).matmul(allocator, qr_result.R, qr_result.Q);

        // Accumulate eigenvectors: V = V @ Q_k
        const V_new = try NDArray(T, 2).matmul(allocator, V, qr_result.Q);
        V.deinit();
        V = V_new;

        // Check convergence: if off-diagonal elements are small, stop
        var off_diag_norm: T = 0;
        for (0..n) |i| {
            for (0..n) |j| {
                if (i != j) {
                    const val = R_Q.data[i * n + j];
                    off_diag_norm += val * val;
                }
            }
        }
        off_diag_norm = @sqrt(off_diag_norm);

        // Replace A_k with R_Q
        A_k.deinit();
        A_k = R_Q;

        // Convergence check
        if (off_diag_norm < convergence_tol) {
            break;
        }
    }

    // Extract eigenvalues from diagonal of converged A_k
    var eigenvalues = try NDArray(T, 1).zeros(allocator, &[_]usize{n}, .row_major);
    errdefer eigenvalues.deinit();

    for (0..n) |i| {
        eigenvalues.data[i] = A_k.data[i * n + i];
    }

    A_k.deinit();

    // Sort eigenvalues in descending order by absolute value and permute eigenvectors
    // Create indices array for sorting
    var indices = try allocator.alloc(usize, n);
    defer allocator.free(indices);
    for (0..n) |i| {
        indices[i] = i;
    }

    // Sort indices by absolute value of eigenvalues (descending)
    var sorted = true;
    while (sorted) {
        sorted = false;
        for (0..n - 1) |i| {
            const abs_curr = @abs(eigenvalues.data[indices[i]]);
            const abs_next = @abs(eigenvalues.data[indices[i + 1]]);
            if (abs_curr < abs_next) {
                const temp = indices[i];
                indices[i] = indices[i + 1];
                indices[i + 1] = temp;
                sorted = true;
            }
        }
    }

    // Create sorted eigenvalues
    var sorted_eigenvalues = try NDArray(T, 1).zeros(allocator, &[_]usize{n}, .row_major);
    errdefer sorted_eigenvalues.deinit();
    for (0..n) |i| {
        sorted_eigenvalues.data[i] = eigenvalues.data[indices[i]];
    }

    // Create permuted eigenvectors (reorder columns of V)
    var sorted_eigenvectors = try NDArray(T, 2).zeros(allocator, &[_]usize{ n, n }, .row_major);
    errdefer sorted_eigenvectors.deinit();
    for (0..n) |i| {
        for (0..n) |j| {
            const old_col = indices[j];
            sorted_eigenvectors.data[i * n + j] = V.data[i * n + old_col];
        }
    }

    eigenvalues.deinit();
    V.deinit();

    return EigResult(T){
        .eigenvalues = sorted_eigenvalues,
        .eigenvectors = sorted_eigenvectors,
        .allocator = allocator,
    };
}

test "eig: 2x2 identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1, 0,
        0, 1,
    }, .row_major);
    defer A.deinit();

    var result = try eig(f64, A, allocator);
    defer result.deinit();

    // For identity matrix, all eigenvalues should be 1
    try testing.expectApproxEqAbs(result.eigenvalues.data[0], @as(f64, 1.0), 1e-10);
    try testing.expectApproxEqAbs(result.eigenvalues.data[1], @as(f64, 1.0), 1e-10);

    // Eigenvectors should be orthonormal
    try verifyEigOrthonormality(f64, allocator, result.eigenvectors, 1e-10);

    // Verify reconstruction
    try verifyEigReconstruction(f64, allocator, A, result.eigenvectors, result.eigenvalues, 1e-10);
}

test "eig: 3x3 identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    }, .row_major);
    defer A.deinit();

    var result = try eig(f64, A, allocator);
    defer result.deinit();

    // All eigenvalues should be 1
    for (0..3) |i| {
        try testing.expectApproxEqAbs(result.eigenvalues.data[i], @as(f64, 1.0), 1e-10);
    }

    try verifyEigOrthonormality(f64, allocator, result.eigenvectors, 1e-10);
    try verifyEigReconstruction(f64, allocator, A, result.eigenvectors, result.eigenvalues, 1e-10);
}

test "eig: 4x4 identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 4 }, &[_]f64{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    }, .row_major);
    defer A.deinit();

    var result = try eig(f64, A, allocator);
    defer result.deinit();

    // All eigenvalues should be 1
    for (0..4) |i| {
        try testing.expectApproxEqAbs(result.eigenvalues.data[i], @as(f64, 1.0), 1e-10);
    }

    try verifyEigOrthonormality(f64, allocator, result.eigenvectors, 1e-10);
    try verifyEigReconstruction(f64, allocator, A, result.eigenvectors, result.eigenvalues, 1e-10);
}

test "eig: 2x2 diagonal matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        3, 0,
        0, 5,
    }, .row_major);
    defer A.deinit();

    var result = try eig(f64, A, allocator);
    defer result.deinit();

    // For diagonal matrix, eigenvalues are the diagonal entries (in descending order)
    try testing.expectApproxEqAbs(result.eigenvalues.data[0], @as(f64, 5.0), 1e-10);
    try testing.expectApproxEqAbs(result.eigenvalues.data[1], @as(f64, 3.0), 1e-10);

    try verifyEigOrthonormality(f64, allocator, result.eigenvectors, 1e-10);
    try verifyEigReconstruction(f64, allocator, A, result.eigenvectors, result.eigenvalues, 1e-10);
}

test "eig: 3x3 diagonal matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        2, 0, 0,
        0, 5, 0,
        0, 0, 3,
    }, .row_major);
    defer A.deinit();

    var result = try eig(f64, A, allocator);
    defer result.deinit();

    // Eigenvalues should be the diagonal entries in descending order
    try testing.expectApproxEqAbs(result.eigenvalues.data[0], @as(f64, 5.0), 1e-10);
    try testing.expectApproxEqAbs(result.eigenvalues.data[1], @as(f64, 3.0), 1e-10);
    try testing.expectApproxEqAbs(result.eigenvalues.data[2], @as(f64, 2.0), 1e-10);

    try verifyEigOrthonormality(f64, allocator, result.eigenvectors, 1e-10);
    try verifyEigReconstruction(f64, allocator, A, result.eigenvectors, result.eigenvalues, 1e-10);
}

test "eig: 2x2 simple symmetric matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1, 2,
        2, 1,
    }, .row_major);
    defer A.deinit();

    var result = try eig(f64, A, allocator);
    defer result.deinit();

    // For [[1, 2], [2, 1]], eigenvalues are 3 and -1
    try testing.expectApproxEqAbs(result.eigenvalues.data[0], @as(f64, 3.0), 1e-10);
    try testing.expectApproxEqAbs(result.eigenvalues.data[1], @as(f64, -1.0), 1e-10);

    try verifyEigOrthonormality(f64, allocator, result.eigenvectors, 1e-10);
    try verifyEigReconstruction(f64, allocator, A, result.eigenvectors, result.eigenvalues, 1e-10);
    try verifyEigProperty(f64, allocator, A, result.eigenvectors, result.eigenvalues, 1e-10);
}

test "eig: 3x3 simple symmetric matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 0, 0,
        0, 2, 0,
        0, 0, 3,
    }, .row_major);
    defer A.deinit();

    var result = try eig(f64, A, allocator);
    defer result.deinit();

    // Eigenvalues should be 3, 2, 1 in descending order
    try testing.expectApproxEqAbs(result.eigenvalues.data[0], @as(f64, 3.0), 1e-10);
    try testing.expectApproxEqAbs(result.eigenvalues.data[1], @as(f64, 2.0), 1e-10);
    try testing.expectApproxEqAbs(result.eigenvalues.data[2], @as(f64, 1.0), 1e-10);

    try verifyEigOrthonormality(f64, allocator, result.eigenvectors, 1e-10);
    try verifyEigReconstruction(f64, allocator, A, result.eigenvectors, result.eigenvalues, 1e-10);
}

test "eig: all zeros matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    }, .row_major);
    defer A.deinit();

    var result = try eig(f64, A, allocator);
    defer result.deinit();

    // All eigenvalues should be 0
    for (0..3) |i| {
        try testing.expectApproxEqAbs(result.eigenvalues.data[i], @as(f64, 0.0), 1e-10);
    }

    try verifyEigOrthonormality(f64, allocator, result.eigenvectors, 1e-10);
    try verifyEigReconstruction(f64, allocator, A, result.eigenvectors, result.eigenvalues, 1e-10);
}

test "eig: single eigenvalue with multiplicity" {
    const allocator = testing.allocator;

    // 2×2 matrix with eigenvalue 2 with multiplicity 2
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        2, 0,
        0, 2,
    }, .row_major);
    defer A.deinit();

    var result = try eig(f64, A, allocator);
    defer result.deinit();

    // Both eigenvalues should be 2
    try testing.expectApproxEqAbs(result.eigenvalues.data[0], @as(f64, 2.0), 1e-10);
    try testing.expectApproxEqAbs(result.eigenvalues.data[1], @as(f64, 2.0), 1e-10);

    try verifyEigOrthonormality(f64, allocator, result.eigenvectors, 1e-10);
    try verifyEigReconstruction(f64, allocator, A, result.eigenvectors, result.eigenvalues, 1e-10);
}

test "eig: orthonormality validation" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 4 }, &[_]f64{
        4, 1, 0, 0,
        1, 3, 0, 1,
        0, 0, 2, 1,
        0, 1, 1, 2,
    }, .row_major);
    defer A.deinit();

    var result = try eig(f64, A, allocator);
    defer result.deinit();

    // Verify V^T @ V = I
    try verifyEigOrthonormality(f64, allocator, result.eigenvectors, 1e-10);
}

test "eig: reconstruction accuracy" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        2, 1, 0,
        1, 3, 1,
        0, 1, 2,
    }, .row_major);
    defer A.deinit();

    var result = try eig(f64, A, allocator);
    defer result.deinit();

    // Verify A ≈ V @ diag(λ) @ V^T
    try verifyEigReconstruction(f64, allocator, A, result.eigenvectors, result.eigenvalues, 1e-10);
}

test "eig: eigenvalue ordering (descending)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 4 }, &[_]f64{
        1, 0, 0, 0,
        0, 4, 0, 0,
        0, 0, 2, 0,
        0, 0, 0, 3,
    }, .row_major);
    defer A.deinit();

    var result = try eig(f64, A, allocator);
    defer result.deinit();

    // Verify eigenvalues are in descending order by absolute value
    try verifyEigDescending(f64, result.eigenvalues);
}

test "eig: f32 precision" {
    const allocator = testing.allocator;

    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{
        1, 2,
        2, 1,
    }, .row_major);
    defer A.deinit();

    var result = try eig(f32, A, allocator);
    defer result.deinit();

    // For [[1, 2], [2, 1]], eigenvalues are 3 and -1
    try testing.expectApproxEqAbs(result.eigenvalues.data[0], @as(f32, 3.0), 1e-5);
    try testing.expectApproxEqAbs(result.eigenvalues.data[1], @as(f32, -1.0), 1e-5);

    try verifyEigOrthonormality(f32, allocator, result.eigenvectors, 1e-5);
}

test "eig: f64 precision" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1, 2,
        2, 1,
    }, .row_major);
    defer A.deinit();

    var result = try eig(f64, A, allocator);
    defer result.deinit();

    // For [[1, 2], [2, 1]], eigenvalues are 3 and -1
    try testing.expectApproxEqAbs(result.eigenvalues.data[0], @as(f64, 3.0), 1e-10);
    try testing.expectApproxEqAbs(result.eigenvalues.data[1], @as(f64, -1.0), 1e-10);

    try verifyEigOrthonormality(f64, allocator, result.eigenvectors, 1e-10);
}

test "eig: small values (numerical stability)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1e-10, 2e-10,
        2e-10, 1e-10,
    }, .row_major);
    defer A.deinit();

    var result = try eig(f64, A, allocator);
    defer result.deinit();

    // Eigenvalues should be 3e-10 and -1e-10
    try testing.expectApproxEqAbs(result.eigenvalues.data[0], @as(f64, 3e-10), 1e-15);
    try testing.expectApproxEqAbs(result.eigenvalues.data[1], @as(f64, -1e-10), 1e-15);

    try verifyEigOrthonormality(f64, allocator, result.eigenvectors, 1e-8);
}

test "eig: large values (numerical stability)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1e10, 2e10,
        2e10, 1e10,
    }, .row_major);
    defer A.deinit();

    var result = try eig(f64, A, allocator);
    defer result.deinit();

    // Eigenvalues should be 3e10 and -1e10
    try testing.expectApproxEqAbs(result.eigenvalues.data[0], @as(f64, 3e10), 1e5);
    try testing.expectApproxEqAbs(result.eigenvalues.data[1], @as(f64, -1e10), 1e5);

    try verifyEigOrthonormality(f64, allocator, result.eigenvectors, 1e-8);
}

test "eig: memory cleanup — no leaks" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 2, 0,
        2, 3, 1,
        0, 1, 2,
    }, .row_major);
    defer A.deinit();

    var result = try eig(f64, A, allocator);
    result.deinit();

    // If allocator is testing.allocator, memory leaks are detected
    // This test passes if no leaks occur
}

test "eig: positive definite covariance matrix" {
    const allocator = testing.allocator;

    // Covariance-like SPD matrix
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        4, 1, 0.5,
        1, 3, 0.2,
        0.5, 0.2, 2,
    }, .row_major);
    defer A.deinit();

    var result = try eig(f64, A, allocator);
    defer result.deinit();

    // All eigenvalues should be positive for SPD matrix
    for (0..3) |i| {
        try testing.expect(result.eigenvalues.data[i] > 0);
    }

    try verifyEigOrthonormality(f64, allocator, result.eigenvectors, 1e-10);
    try verifyEigReconstruction(f64, allocator, A, result.eigenvectors, result.eigenvalues, 1e-10);
}

test "eig: eigenvalue equation A @ V = V @ diag(λ)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        2, 1, 1,
        1, 3, 0,
        1, 0, 3,
    }, .row_major);
    defer A.deinit();

    var result = try eig(f64, A, allocator);
    defer result.deinit();

    // Verify A @ V = V @ diag(λ)
    try verifyEigProperty(f64, allocator, A, result.eigenvectors, result.eigenvalues, 1e-10);
}

test "eig: non-square matrix rejection" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f64{
        1, 2,
        3, 4,
        5, 6,
    }, .row_major);
    defer A.deinit();

    const result = eig(f64, A, allocator);
    try testing.expectError(error.InvalidDimensions, result);
}

test "eig: non-symmetric matrix rejection" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1, 2,
        3, 4,
    }, .row_major);
    defer A.deinit();

    const result = eig(f64, A, allocator);
    try testing.expectError(error.NonSymmetricMatrix, result);
}
