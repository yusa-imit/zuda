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
        2, 3, 1,
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
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 17,  // Last element different to avoid singularity
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
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
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
        1.2, 2.3, 3.4,
        4.5, 5.6, 6.7,
        7.8, 8.9, 9.0,
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
        -1, 2, -3,
        4, -5, 6,
        -7, 8, -9,
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
