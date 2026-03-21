//! Linear System Solver — solve(A, b)
//!
//! Solves the linear system Ax = b using appropriate matrix decomposition:
//! - Cholesky for Symmetric Positive Definite (SPD) matrices
//! - LU for general square matrices
//! - QR for overdetermined (tall) systems (least squares)
//!
//! Auto-selects decomposition based on matrix shape and properties.
//!
//! ## Time Complexity
//! - Decomposition: O(n³) for all methods
//! - Back-substitution: O(n²) for dense forward/back substitution
//! - Total: O(n³)
//!
//! ## Numeric Precision
//! - Tested for f32 (tolerance 1e-5) and f64 (tolerance 1e-10)
//! - Backward stability via decomposition-based solution
//!
//! ## Error Handling
//! - error.SingularMatrix: A is rank-deficient
//! - error.DimensionMismatch: b does not match A rows
//! - error.UnderdeterminedSystem: A has more columns than rows
//! - error.NotPositiveDefinite: Auto-detected Cholesky failed

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const root = @import("../root.zig");
const NDArray = root.ndarray.NDArray;
const lu_mod = @import("lu.zig");
const decomp = @import("decompositions.zig");
const blas_mod = @import("blas.zig");

/// Solve linear system Ax = b using appropriate decomposition
///
/// Auto-selects solver:
/// - Cholesky: If A is square and appears SPD
/// - LU: If A is square and general
/// - QR: If A is tall (m > n) — least squares
///
/// Parameters:
/// - T: Numeric type (f32, f64)
/// - A: Coefficient matrix (m×n, usually square or tall)
/// - b: Right-hand side vector (m×1)
/// - allocator: Memory allocator
///
/// Returns: Solution vector x (n×1) such that Ax ≈ b
///
/// Errors:
/// - error.DimensionMismatch: A.shape[0] != b.shape[0]
/// - error.UnderdeterminedSystem: A has more columns than rows (wide)
/// - error.SingularMatrix: A is rank-deficient (LU/Cholesky)
/// - error.NotPositiveDefinite: Cholesky decomposition failed
///
/// Time: O(n³) decomposition + O(n²) back-substitution
/// Space: O(n²) for decomposition matrices
///
/// Example:
/// ```zig
/// var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{2, 2}, &[_]f64{
///     2, 1,
///     1, 2,
/// }, .row_major);
/// defer A.deinit();
///
/// var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{
///     3, 3,
/// }, .row_major);
/// defer b.deinit();
///
/// var x = try solve(f64, A, b, allocator);
/// defer x.deinit();
/// // x should be approximately [1, 1]
/// ```
pub fn solve(
    comptime T: type,
    A: NDArray(T, 2),
    b: NDArray(T, 1),
    allocator: Allocator,
) (NDArray(T, 2).Error || NDArray(T, 1).Error || std.mem.Allocator.Error || error{
    DimensionMismatch,
    SingularMatrix,
    UnderdeterminedSystem,
    NotPositiveDefinite,
    InvalidDimensions,
    NotImplemented,
})!NDArray(T, 1) {
    const m = A.shape[0];
    const n = A.shape[1];

    // Check dimension compatibility: A has m rows, b must have m elements
    if (b.shape[0] != m) {
        return error.DimensionMismatch;
    }

    // Reject underdetermined systems (more columns than rows)
    if (n > m) {
        return error.UnderdeterminedSystem;
    }

    // Dispatch based on system shape and properties
    if (m == n) {
        // Square matrix — try SPD (Cholesky), fall back to LU
        return try solveSquare(T, A, b, allocator);
    } else {
        // Overdetermined system (m > n) — use QR least squares
        return try solveOverdetermined(T, A, b, allocator);
    }
}

/// Solve square system using appropriate method (Cholesky or LU)
fn solveSquare(
    comptime T: type,
    A: NDArray(T, 2),
    b: NDArray(T, 1),
    allocator: Allocator,
) !NDArray(T, 1) {
    // Try SPD detection and Cholesky
    if (try isSPD(T, A, allocator)) {
        return try solveCholesky(T, A, b, allocator);
    }

    // Fall back to LU for general square matrices
    return try solveLU(T, A, b, allocator);
}

/// Solve overdetermined system using QR decomposition (least squares)
fn solveOverdetermined(
    comptime T: type,
    A: NDArray(T, 2),
    b: NDArray(T, 1),
    allocator: Allocator,
) !NDArray(T, 1) {
    const m = A.shape[0];
    const n = A.shape[1];

    // Compute QR factorization
    var result = try decomp.qr(T, A, allocator);
    defer result.deinit();

    // Extract Q (m×m) and R (m×n)
    const Q = result.Q;
    const R = result.R;

    // Compute c = Q^T b (m×1 = m×m @ m×1)
    var c = try NDArray(T, 1).zeros(allocator, &[_]usize{m}, .row_major);
    errdefer c.deinit();

    for (0..m) |i| {
        var sum: T = 0;
        for (0..m) |k| {
            sum += Q.data[k * m + i] * b.data[k]; // Q^T[i,k] = Q[k,i]
        }
        c.data[i] = sum;
    }

    // Back substitution: solve R x = c (R is m×n, but we only use first n rows)
    // For overdetermined least squares, R is stored as m×n upper triangular
    var x = try NDArray(T, 1).zeros(allocator, &[_]usize{n}, .row_major);
    errdefer x.deinit();

    // Back substitution from row n-1 to 0
    for (0..n) |idx| {
        const i = n - 1 - idx;
        var sum: T = 0;

        // Sum R[i,j] * x[j] for j = i+1..n-1
        for (i + 1..n) |j| {
            sum += R.data[i * n + j] * x.data[j];
        }

        // R[i,i] should be non-zero for least squares solution to exist
        const Rii = R.data[i * n + i];
        if (@abs(Rii) < 1e-15) {
            c.deinit();
            x.deinit();
            return error.SingularMatrix;
        }

        x.data[i] = (c.data[i] - sum) / Rii;
    }

    c.deinit();
    return x;
}

/// Solve SPD system using Cholesky decomposition
fn solveCholesky(
    comptime T: type,
    A: NDArray(T, 2),
    b: NDArray(T, 1),
    allocator: Allocator,
) !NDArray(T, 1) {
    const n = A.shape[0];

    // Compute L from Cholesky
    var L = try decomp.cholesky(T, A, allocator);
    defer L.deinit();

    // Forward substitution: solve L y = b
    var y = try NDArray(T, 1).zeros(allocator, &[_]usize{n}, .row_major);
    errdefer y.deinit();

    for (0..n) |i| {
        var sum: T = 0;
        for (0..i) |j| {
            sum += L.data[i * n + j] * y.data[j];
        }

        const Lii = L.data[i * n + i];
        if (@abs(Lii) < 1e-15) {
            y.deinit();
            return error.SingularMatrix;
        }

        y.data[i] = (b.data[i] - sum) / Lii;
    }

    // Back substitution: solve L^T x = y
    var x = try NDArray(T, 1).zeros(allocator, &[_]usize{n}, .row_major);
    errdefer x.deinit();

    for (0..n) |idx| {
        const i = n - 1 - idx;
        var sum: T = 0;

        // L^T[i,j] = L[j,i]
        for (i + 1..n) |j| {
            sum += L.data[j * n + i] * x.data[j];
        }

        const Lii = L.data[i * n + i];
        if (@abs(Lii) < 1e-15) {
            y.deinit();
            x.deinit();
            return error.SingularMatrix;
        }

        x.data[i] = (y.data[i] - sum) / Lii;
    }

    y.deinit();
    return x;
}

/// Solve general square system using LU decomposition
fn solveLU(
    comptime T: type,
    A: NDArray(T, 2),
    b: NDArray(T, 1),
    allocator: Allocator,
) !NDArray(T, 1) {
    const n = A.shape[0];

    // Compute LU factorization
    var result = try lu_mod.lu(T, allocator, A);
    defer result.deinit();

    const P = result.P;
    const L = result.L;
    const U = result.U;

    // Apply permutation: Pb = P b
    var Pb = try NDArray(T, 1).zeros(allocator, &[_]usize{n}, .row_major);
    errdefer Pb.deinit();

    for (0..n) |i| {
        var sum: T = 0;
        for (0..n) |j| {
            sum += P.data[i * n + j] * b.data[j];
        }
        Pb.data[i] = sum;
    }

    // Forward substitution: solve L y = Pb
    var y = try NDArray(T, 1).zeros(allocator, &[_]usize{n}, .row_major);
    errdefer y.deinit();

    for (0..n) |i| {
        var sum: T = 0;
        for (0..i) |j| {
            sum += L.data[i * n + j] * y.data[j];
        }

        // L has unit diagonal, so L[i,i] = 1
        y.data[i] = Pb.data[i] - sum;
    }

    // Back substitution: solve U x = y
    var x = try NDArray(T, 1).zeros(allocator, &[_]usize{n}, .row_major);
    errdefer x.deinit();

    for (0..n) |idx| {
        const i = n - 1 - idx;
        var sum: T = 0;

        for (i + 1..n) |j| {
            sum += U.data[i * n + j] * x.data[j];
        }

        const Uii = U.data[i * n + i];
        if (@abs(Uii) < 1e-15) {
            Pb.deinit();
            y.deinit();
            x.deinit();
            return error.SingularMatrix;
        }

        x.data[i] = (y.data[i] - sum) / Uii;
    }

    Pb.deinit();
    y.deinit();
    return x;
}

/// Check if matrix is symmetric positive definite
/// Uses heuristic: check symmetry and attempt Cholesky
fn isSPD(
    comptime T: type,
    A: NDArray(T, 2),
    allocator: Allocator,
) !bool {
    const n = A.shape[0];

    // Check symmetry: ||A - A^T|| / ||A|| < sqrt(epsilon)
    // Compute Frobenius norms
    var norm_A: T = 0;
    var norm_diff: T = 0;

    for (0..n) |i| {
        for (0..n) |j| {
            const Aij = try A.get(&[_]isize{ @intCast(i), @intCast(j) });
            const Aji = try A.get(&[_]isize{ @intCast(j), @intCast(i) });

            norm_A += Aij * Aij;
            norm_diff += (Aij - Aji) * (Aij - Aji);
        }
    }

    norm_A = @sqrt(norm_A);
    norm_diff = @sqrt(norm_diff);

    // If matrix is not symmetric enough, it's not SPD
    const sqrt_epsilon = switch (T) {
        f32 => 1e-4,
        f64 => 1e-8,
        else => 1e-15,
    };

    if (norm_A > 0 and norm_diff / norm_A > sqrt_epsilon) {
        return false; // Not symmetric
    }

    // Try Cholesky decomposition
    var L = decomp.cholesky(T, A, allocator) catch {
        return false; // Cholesky failed, not SPD
    };
    L.deinit();

    return true; // Passed symmetry and Cholesky checks
}

// ============================================================================
// Tests
// ============================================================================

test "solve: 2x2 SPD matrix (identity)" {
    const allocator = testing.allocator;

    // A = I, b = [1, 2]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1, 0,
        0, 1,
    }, .row_major);
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 2 }, .row_major);
    defer b.deinit();

    var x = try solve(f64, A, b, allocator);
    defer x.deinit();

    // x should equal b = [1, 2]
    try testing.expectApproxEqAbs(1.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(2.0, x.data[1], 1e-10);
}

test "solve: 2x2 SPD matrix (diagonal)" {
    const allocator = testing.allocator;

    // A = [[2, 0], [0, 3]], b = [4, 6]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        2, 0,
        0, 3,
    }, .row_major);
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 4, 6 }, .row_major);
    defer b.deinit();

    var x = try solve(f64, A, b, allocator);
    defer x.deinit();

    // x = [4/2, 6/3] = [2, 2]
    try testing.expectApproxEqAbs(2.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(2.0, x.data[1], 1e-10);
}

test "solve: 2x2 SPD matrix (symmetric positive definite)" {
    const allocator = testing.allocator;

    // A = [[2, 1], [1, 2]] (SPD), b = [3, 3]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        2, 1,
        1, 2,
    }, .row_major);
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 3, 3 }, .row_major);
    defer b.deinit();

    var x = try solve(f64, A, b, allocator);
    defer x.deinit();

    // Expected: x = [1, 1] (since A*[1,1] = [3, 3])
    try testing.expectApproxEqAbs(1.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(1.0, x.data[1], 1e-10);
}

test "solve: 3x3 SPD matrix" {
    const allocator = testing.allocator;

    // A = [[4, 1, 0], [1, 4, 1], [0, 1, 4]] (SPD)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        4, 1, 0,
        1, 4, 1,
        0, 1, 4,
    }, .row_major);
    defer A.deinit();

    // b = [5, 6, 5]
    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 5, 6, 5 }, .row_major);
    defer b.deinit();

    var x = try solve(f64, A, b, allocator);
    defer x.deinit();

    // Verify Ax ≈ b
    var Ax = try NDArray(f64, 1).zeros(allocator, &[_]usize{3}, .row_major);
    defer Ax.deinit();

    for (0..3) |i| {
        var sum: f64 = 0;
        for (0..3) |j| {
            sum += (try A.get(&[_]isize{ @intCast(i), @intCast(j) })) * x.data[j];
        }
        Ax.data[i] = sum;
    }

    for (0..3) |i| {
        try testing.expectApproxEqAbs(b.data[i], Ax.data[i], 1e-9);
    }
}

test "solve: 2x2 general square matrix (non-symmetric)" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4]], b = [5, 11]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1, 2,
        3, 4,
    }, .row_major);
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 5, 11 }, .row_major);
    defer b.deinit();

    var x = try solve(f64, A, b, allocator);
    defer x.deinit();

    // Expected: x = [1, 2] (since A*[1,2] = [5, 11])
    try testing.expectApproxEqAbs(1.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(2.0, x.data[1], 1e-10);
}

test "solve: 3x3 general square matrix" {
    const allocator = testing.allocator;

    // A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], b = [8, -11, -3]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        2, 1, -1,
        -3, -1, 2,
        -2, 1, 2,
    }, .row_major);
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 8, -11, -3 }, .row_major);
    defer b.deinit();

    var x = try solve(f64, A, b, allocator);
    defer x.deinit();

    // Verify Ax ≈ b
    var Ax = try NDArray(f64, 1).zeros(allocator, &[_]usize{3}, .row_major);
    defer Ax.deinit();

    for (0..3) |i| {
        var sum: f64 = 0;
        for (0..3) |j| {
            sum += (try A.get(&[_]isize{ @intCast(i), @intCast(j) })) * x.data[j];
        }
        Ax.data[i] = sum;
    }

    for (0..3) |i| {
        try testing.expectApproxEqAbs(b.data[i], Ax.data[i], 1e-9);
    }
}

test "solve: 1x1 single equation" {
    const allocator = testing.allocator;

    // A = [[5]], b = [10]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 1 }, &[_]f64{5}, .row_major);
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{10}, .row_major);
    defer b.deinit();

    var x = try solve(f64, A, b, allocator);
    defer x.deinit();

    // x = [2]
    try testing.expectApproxEqAbs(2.0, x.data[0], 1e-10);
}

test "solve: zero RHS vector" {
    const allocator = testing.allocator;

    // A = [[2, 1], [1, 2]], b = [0, 0]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        2, 1,
        1, 2,
    }, .row_major);
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 0, 0 }, .row_major);
    defer b.deinit();

    var x = try solve(f64, A, b, allocator);
    defer x.deinit();

    // x should be [0, 0]
    try testing.expectApproxEqAbs(0.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(0.0, x.data[1], 1e-10);
}

test "solve: overdetermined system (3x2, tall matrix, least squares)" {
    const allocator = testing.allocator;

    // A = [[1, 0], [1, 1], [1, 2]] (3x2), b = [1, 2, 3]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f64{
        1, 0,
        1, 1,
        1, 2,
    }, .row_major);
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer b.deinit();

    var x = try solve(f64, A, b, allocator);
    defer x.deinit();

    // QR least squares solution
    // Expected: x ≈ [1, 1] gives [1, 2, 3] exactly (exact fit)
    try testing.expectApproxEqAbs(1.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(1.0, x.data[1], 1e-10);
}

test "solve: tall matrix with overdetermined least squares (4x3)" {
    const allocator = testing.allocator;

    // A = [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]] (4x3)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 3 }, &[_]f64{
        1, 0, 0,
        1, 1, 0,
        1, 1, 1,
        1, 1, 1,
    }, .row_major);
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1, 2, 3, 3.5 }, .row_major);
    defer b.deinit();

    var x = try solve(f64, A, b, allocator);
    defer x.deinit();

    // Verify least squares: ||Ax - b||² is minimal
    var Ax = try NDArray(f64, 1).zeros(allocator, &[_]usize{4}, .row_major);
    defer Ax.deinit();

    for (0..4) |i| {
        var sum: f64 = 0;
        for (0..3) |j| {
            sum += (try A.get(&[_]isize{ @intCast(i), @intCast(j) })) * x.data[j];
        }
        Ax.data[i] = sum;
    }

    // Check that residual is reasonably small
    var residual_norm: f64 = 0;
    for (0..4) |i| {
        const diff = Ax.data[i] - b.data[i];
        residual_norm += diff * diff;
    }
    residual_norm = @sqrt(residual_norm);

    try testing.expect(residual_norm < 1.0); // Should be reasonably small
}

test "solve: singular matrix detection — all zeros" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 2 }, .row_major);
    defer b.deinit();

    const err = solve(f64, A, b, allocator);
    try testing.expectError(error.SingularMatrix, err);
}

test "solve: singular matrix detection — rank deficient" {
    const allocator = testing.allocator;

    // Rank 1: rows are multiples
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 2, 3,
        2, 4, 6,
        3, 6, 9,
    }, .row_major);
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 6, 12, 18 }, .row_major);
    defer b.deinit();

    const err = solve(f64, A, b, allocator);
    try testing.expectError(error.SingularMatrix, err);
}

test "solve: dimension mismatch — b wrong size" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1, 2,
        3, 4,
    }, .row_major);
    defer A.deinit();

    // b has 3 elements but A has 2 rows
    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer b.deinit();

    const err = solve(f64, A, b, allocator);
    try testing.expectError(error.DimensionMismatch, err);
}

test "solve: underdetermined system error (wide matrix)" {
    const allocator = testing.allocator;

    // A = [[1, 2, 3], [4, 5, 6]] (2x3, more columns than rows)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{
        1, 2, 3,
        4, 5, 6,
    }, .row_major);
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 7, 8 }, .row_major);
    defer b.deinit();

    const err = solve(f64, A, b, allocator);
    try testing.expectError(error.UnderdeterminedSystem, err);
}

test "solve: f32 precision SPD" {
    const allocator = testing.allocator;

    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{
        2, 1,
        1, 2,
    }, .row_major);
    defer A.deinit();

    var b = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{2}, &[_]f32{ 3, 3 }, .row_major);
    defer b.deinit();

    var x = try solve(f32, A, b, allocator);
    defer x.deinit();

    // f32 tolerance: 1e-5
    try testing.expectApproxEqAbs(@as(f32, 1.0), x.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.0), x.data[1], 1e-5);
}

test "solve: f32 precision general" {
    const allocator = testing.allocator;

    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f32{
        2, 1, -1,
        -3, -1, 2,
        -2, 1, 2,
    }, .row_major);
    defer A.deinit();

    var b = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{3}, &[_]f32{ 8, -11, -3 }, .row_major);
    defer b.deinit();

    var x = try solve(f32, A, b, allocator);
    defer x.deinit();

    // Verify Ax ≈ b with f32 tolerance
    var Ax = try NDArray(f32, 1).zeros(allocator, &[_]usize{3}, .row_major);
    defer Ax.deinit();

    for (0..3) |i| {
        var sum: f32 = 0;
        for (0..3) |j| {
            sum += (try A.get(&[_]isize{ @intCast(i), @intCast(j) })) * x.data[j];
        }
        Ax.data[i] = sum;
    }

    for (0..3) |i| {
        try testing.expectApproxEqAbs(b.data[i], Ax.data[i], 1e-5);
    }
}

test "solve: reconstruction Ax ≈ b (2x2 SPD)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        5, 2,
        2, 3,
    }, .row_major);
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 7, 5 }, .row_major);
    defer b.deinit();

    var x = try solve(f64, A, b, allocator);
    defer x.deinit();

    // Compute Ax
    var Ax = try NDArray(f64, 1).zeros(allocator, &[_]usize{2}, .row_major);
    defer Ax.deinit();

    for (0..2) |i| {
        var sum: f64 = 0;
        for (0..2) |j| {
            sum += (try A.get(&[_]isize{ @intCast(i), @intCast(j) })) * x.data[j];
        }
        Ax.data[i] = sum;
    }

    // Verify Ax ≈ b
    try testing.expectApproxEqAbs(b.data[0], Ax.data[0], 1e-10);
    try testing.expectApproxEqAbs(b.data[1], Ax.data[1], 1e-10);
}

test "solve: reconstruction Ax ≈ b (3x3 general)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        6, 2, 1,
        2, 4, -1,
        1, -1, 3,
    }, .row_major);
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 5, 6, 4 }, .row_major);
    defer b.deinit();

    var x = try solve(f64, A, b, allocator);
    defer x.deinit();

    // Compute Ax
    var Ax = try NDArray(f64, 1).zeros(allocator, &[_]usize{3}, .row_major);
    defer Ax.deinit();

    for (0..3) |i| {
        var sum: f64 = 0;
        for (0..3) |j| {
            sum += (try A.get(&[_]isize{ @intCast(i), @intCast(j) })) * x.data[j];
        }
        Ax.data[i] = sum;
    }

    // Verify Ax ≈ b
    for (0..3) |i| {
        try testing.expectApproxEqAbs(b.data[i], Ax.data[i], 1e-9);
    }
}

test "solve: memory cleanup — no leaks (SPD)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        4, 1, 0,
        1, 4, 1,
        0, 1, 4,
    }, .row_major);
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 5, 6, 5 }, .row_major);
    defer b.deinit();

    var x = try solve(f64, A, b, allocator);
    x.deinit();

    // Testing allocator detects any leaks
}

test "solve: memory cleanup — no leaks (general)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        2, 1, -1,
        -3, -1, 2,
        -2, 1, 2,
    }, .row_major);
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 8, -11, -3 }, .row_major);
    defer b.deinit();

    var x = try solve(f64, A, b, allocator);
    x.deinit();

    // Testing allocator detects any leaks
}

test "solve: memory cleanup — no leaks (overdetermined)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f64{
        1, 0,
        1, 1,
        1, 2,
    }, .row_major);
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer b.deinit();

    var x = try solve(f64, A, b, allocator);
    x.deinit();

    // Testing allocator detects any leaks
}

test "solve: negative values in system" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        -2, 1,
        1, -2,
    }, .row_major);
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ -3, -3 }, .row_major);
    defer b.deinit();

    var x = try solve(f64, A, b, allocator);
    defer x.deinit();

    // Verify Ax ≈ b
    var Ax = try NDArray(f64, 1).zeros(allocator, &[_]usize{2}, .row_major);
    defer Ax.deinit();

    for (0..2) |i| {
        var sum: f64 = 0;
        for (0..2) |j| {
            sum += (try A.get(&[_]isize{ @intCast(i), @intCast(j) })) * x.data[j];
        }
        Ax.data[i] = sum;
    }

    try testing.expectApproxEqAbs(b.data[0], Ax.data[0], 1e-10);
    try testing.expectApproxEqAbs(b.data[1], Ax.data[1], 1e-10);
}

test "solve: large values in system" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1000, 1,
        1, 1000,
    }, .row_major);
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1001, 1001 }, .row_major);
    defer b.deinit();

    var x = try solve(f64, A, b, allocator);
    defer x.deinit();

    // Verify Ax ≈ b
    var Ax = try NDArray(f64, 1).zeros(allocator, &[_]usize{2}, .row_major);
    defer Ax.deinit();

    for (0..2) |i| {
        var sum: f64 = 0;
        for (0..2) |j| {
            sum += (try A.get(&[_]isize{ @intCast(i), @intCast(j) })) * x.data[j];
        }
        Ax.data[i] = sum;
    }

    try testing.expectApproxEqAbs(b.data[0], Ax.data[0], 1e-8);
    try testing.expectApproxEqAbs(b.data[1], Ax.data[1], 1e-8);
}

test "solve: nearly singular matrix (ill-conditioned)" {
    const allocator = testing.allocator;

    // Hilbert-like matrix (ill-conditioned but invertible)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1.0, 0.5, 1.0 / 3.0,
        0.5, 1.0 / 3.0, 0.25,
        1.0 / 3.0, 0.25, 0.2,
    }, .row_major);
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1.8, 1.08, 0.78 }, .row_major);
    defer b.deinit();

    var x = try solve(f64, A, b, allocator);
    defer x.deinit();

    // For ill-conditioned matrix, use larger tolerance
    var Ax = try NDArray(f64, 1).zeros(allocator, &[_]usize{3}, .row_major);
    defer Ax.deinit();

    for (0..3) |i| {
        var sum: f64 = 0;
        for (0..3) |j| {
            sum += (try A.get(&[_]isize{ @intCast(i), @intCast(j) })) * x.data[j];
        }
        Ax.data[i] = sum;
    }

    for (0..3) |i| {
        try testing.expectApproxEqAbs(b.data[i], Ax.data[i], 1e-6); // Larger tolerance
    }
}
