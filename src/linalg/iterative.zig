//! Iterative Solvers for Sparse Linear Systems
//!
//! Provides iterative methods for solving large sparse linear systems Ax = b
//! where direct methods (LU, Cholesky) are prohibitively expensive.
//!
//! Algorithms:
//! - **Conjugate Gradient (CG)**: For symmetric positive definite matrices
//!   - Optimal Krylov subspace method
//!   - Convergence in ≤ n iterations (theory), often much faster (practice)
//!   - Memory: O(n) vs O(n²) for direct methods
//!
//! Use cases:
//! - Large-scale FEM/FDM simulations (> 10⁶ unknowns)
//! - Graph Laplacian systems (PageRank, spectral clustering)
//! - Diffusion equations, heat transfer, elasticity
//! - Preconditioned systems (incomplete Cholesky, Jacobi)
//!
//! Design:
//! - Works with CSR sparse matrices (efficient matvec)
//! - Explicit allocator passing (allocator-first)
//! - Configurable tolerance and max iterations
//! - Returns convergence info (iterations, residual norm)

const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const testing = std.testing;
const math = std.math;
const sparse = @import("sparse.zig");

/// Result of an iterative solver
pub fn SolverResult(comptime T: type) type {
    return struct {
        /// Solution vector x
        x: []T,
        /// Number of iterations performed
        iterations: usize,
        /// Final residual norm ||b - Ax||₂
        residual_norm: T,
        /// Whether the solver converged within tolerance
        converged: bool,
        /// Allocator used (for deallocation)
        allocator: Allocator,

        pub fn deinit(self: *@This()) void {
            self.allocator.free(self.x);
        }
    };
}

/// Conjugate Gradient (CG) solver for symmetric positive definite systems
///
/// Solves Ax = b where A is sparse, symmetric, and positive definite (SPD).
///
/// Algorithm (Hestenes & Stiefel, 1952):
/// 1. r₀ = b - Ax₀ (initial residual)
/// 2. p₀ = r₀ (initial search direction)
/// 3. For k = 0, 1, 2, ... until convergence:
///    α = <rₖ, rₖ> / <pₖ, Apₖ>
///    xₖ₊₁ = xₖ + α pₖ
///    rₖ₊₁ = rₖ - α Apₖ
///    β = <rₖ₊₁, rₖ₊₁> / <rₖ, rₖ>
///    pₖ₊₁ = rₖ₊₁ + β pₖ
///
/// Convergence: ||rₖ|| < tol × ||r₀|| or k ≥ max_iter
///
/// Time: O(nnz × k) where k = iterations (typically k ≪ n)
/// Space: O(n) for x, r, p, Ap vectors
///
/// Parameters:
/// - A: Sparse CSR matrix (must be SPD, not validated!)
/// - b: Right-hand side vector
/// - x0: Initial guess (optional, can be null for zero init)
/// - tol: Convergence tolerance (relative residual)
/// - max_iter: Maximum iterations (0 = unlimited, use n)
///
/// Returns: SolverResult with solution, iterations, residual norm
///
/// Errors:
/// - DimensionMismatch: A.rows ≠ A.cols or A.rows ≠ b.len or x0.len ≠ b.len
/// - OutOfMemory: Allocation failure
///
/// Example:
/// ```zig
/// var result = try conjugateGradient(f64, allocator, &A, b, null, 1e-6, 1000);
/// defer result.deinit();
/// if (result.converged) {
///     std.debug.print("Solution: {d}\n", .{result.x});
/// }
/// ```
pub fn conjugateGradient(
    comptime T: type,
    allocator: Allocator,
    A: *const sparse.CSR(T),
    b: []const T,
    x0: ?[]const T,
    tol: T,
    max_iter: usize,
) !SolverResult(T) {
    const n = A.rows;

    // Validate dimensions
    if (A.rows != A.cols) return error.DimensionMismatch;
    if (b.len != n) return error.DimensionMismatch;
    if (x0) |x_init| {
        if (x_init.len != n) return error.DimensionMismatch;
    }

    // Determine actual max iterations (0 means n)
    const max_iters = if (max_iter == 0) n else max_iter;

    // Allocate workspace: x, r, p, Ap
    const x = try allocator.alloc(T, n);
    errdefer allocator.free(x);

    const r = try allocator.alloc(T, n);
    errdefer allocator.free(r);

    const p = try allocator.alloc(T, n);
    errdefer allocator.free(p);

    const Ap = try allocator.alloc(T, n);
    errdefer allocator.free(Ap);

    // Initialize x (copy x0 or zero)
    if (x0) |x_init| {
        @memcpy(x, x_init);
    } else {
        @memset(x, 0);
    }

    // Compute initial residual: r₀ = b - Ax₀
    const Ax = try A.matvec(allocator, x);
    defer allocator.free(Ax);

    for (r, 0..) |*r_i, i| {
        r_i.* = b[i] - Ax[i];
    }

    // Initial search direction: p₀ = r₀
    @memcpy(p, r);

    // Initial residual norm
    var r_norm_sq = dot(T, r, r);
    const r0_norm = @sqrt(r_norm_sq);
    const threshold = tol * r0_norm;

    var k: usize = 0;
    while (k < max_iters) : (k += 1) {
        // Check convergence: ||rₖ|| < tol × ||r₀||
        const r_norm = @sqrt(r_norm_sq);
        if (r_norm < threshold) {
            // Converged
            allocator.free(r);
            allocator.free(p);
            allocator.free(Ap);
            return SolverResult(T){
                .x = x,
                .iterations = k,
                .residual_norm = r_norm,
                .converged = true,
                .allocator = allocator,
            };
        }

        // Compute Ap = A × p
        const Ap_temp = try A.matvec(allocator, p);
        @memcpy(Ap, Ap_temp);
        allocator.free(Ap_temp);

        // α = <rₖ, rₖ> / <pₖ, Apₖ>
        const pAp = dot(T, p, Ap);
        if (@abs(pAp) < 1e-14) {
            // Numerical breakdown (should not happen for SPD matrices)
            allocator.free(r);
            allocator.free(p);
            allocator.free(Ap);
            return SolverResult(T){
                .x = x,
                .iterations = k,
                .residual_norm = @sqrt(r_norm_sq),
                .converged = false,
                .allocator = allocator,
            };
        }
        const alpha = r_norm_sq / pAp;

        // xₖ₊₁ = xₖ + α pₖ
        for (x, 0..) |*x_i, i| {
            x_i.* += alpha * p[i];
        }

        // rₖ₊₁ = rₖ - α Apₖ
        for (r, 0..) |*r_i, i| {
            r_i.* -= alpha * Ap[i];
        }

        // <rₖ₊₁, rₖ₊₁>
        const r_norm_sq_new = dot(T, r, r);

        // β = <rₖ₊₁, rₖ₊₁> / <rₖ, rₖ>
        const beta = r_norm_sq_new / r_norm_sq;

        // pₖ₊₁ = rₖ₊₁ + β pₖ
        for (p, 0..) |*p_i, i| {
            p_i.* = r[i] + beta * p_i.*;
        }

        r_norm_sq = r_norm_sq_new;
    }

    // Max iterations reached without convergence
    allocator.free(r);
    allocator.free(p);
    allocator.free(Ap);

    return SolverResult(T){
        .x = x,
        .iterations = k,
        .residual_norm = @sqrt(r_norm_sq),
        .converged = false,
        .allocator = allocator,
    };
}

/// Dot product: <x, y> = Σ xᵢyᵢ
///
/// Time: O(n) | Space: O(1)
fn dot(comptime T: type, x: []const T, y: []const T) T {
    var sum: T = 0;
    for (x, 0..) |x_i, i| {
        sum += x_i * y[i];
    }
    return sum;
}

// ============================================================================
// Tests
// ============================================================================

test "CG: 2×2 SPD identity system" {
    const allocator = testing.allocator;

    // A = I (2×2 identity)
    var coo = sparse.COO(f64).init(allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    // b = [1, 2]ᵀ → solution x = [1, 2]ᵀ
    const b = [_]f64{ 1.0, 2.0 };

    var result = try conjugateGradient(f64, allocator, &A, &b, null, 1e-10, 100);
    defer result.deinit();

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(1.0, result.x[0], 1e-9);
    try testing.expectApproxEqAbs(2.0, result.x[1], 1e-9);
    try testing.expect(result.iterations <= 2); // CG should converge in 1-2 iterations for 2×2 SPD
}

test "CG: 3×3 SPD diagonal system" {
    const allocator = testing.allocator;

    // A = diag(2, 3, 4)
    var coo = sparse.COO(f64).init(allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 2.0);
    try coo.append(1, 1, 3.0);
    try coo.append(2, 2, 4.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    // b = [4, 6, 8]ᵀ → solution x = [2, 2, 2]ᵀ
    const b = [_]f64{ 4.0, 6.0, 8.0 };

    var result = try conjugateGradient(f64, allocator, &A, &b, null, 1e-10, 100);
    defer result.deinit();

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(2.0, result.x[0], 1e-9);
    try testing.expectApproxEqAbs(2.0, result.x[1], 1e-9);
    try testing.expectApproxEqAbs(2.0, result.x[2], 1e-9);
}

test "CG: 3×3 SPD general system" {
    const allocator = testing.allocator;

    // A = [4  1  0]    SPD matrix
    //     [1  3  1]
    //     [0  1  4]
    var coo = sparse.COO(f64).init(allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 4.0);
    try coo.append(0, 1, 1.0);
    try coo.append(1, 0, 1.0);
    try coo.append(1, 1, 3.0);
    try coo.append(1, 2, 1.0);
    try coo.append(2, 1, 1.0);
    try coo.append(2, 2, 4.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    // b = [5, 5, 5]ᵀ → solution approximately x = [1, 1, 1]ᵀ
    // Verify: Ax = [4+1, 1+3+1, 1+4] = [5, 5, 5] ✓
    const b = [_]f64{ 5.0, 5.0, 5.0 };

    var result = try conjugateGradient(f64, allocator, &A, &b, null, 1e-10, 100);
    defer result.deinit();

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(1.0, result.x[0], 1e-8);
    try testing.expectApproxEqAbs(1.0, result.x[1], 1e-8);
    try testing.expectApproxEqAbs(1.0, result.x[2], 1e-8);
}

test "CG: with initial guess" {
    const allocator = testing.allocator;

    // A = I (2×2 identity)
    var coo = sparse.COO(f64).init(allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    const b = [_]f64{ 5.0, 7.0 };
    const x0 = [_]f64{ 4.0, 6.0 }; // Close initial guess

    var result = try conjugateGradient(f64, allocator, &A, &b, &x0, 1e-10, 100);
    defer result.deinit();

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(5.0, result.x[0], 1e-9);
    try testing.expectApproxEqAbs(7.0, result.x[1], 1e-9);
    try testing.expect(result.iterations <= 2); // Should converge faster with good guess
}

test "CG: large sparse tridiagonal system (5×5)" {
    const allocator = testing.allocator;

    // A = tridiag(-1, 2, -1) — classic FDM discretization
    // [2  -1   0   0   0]
    // [-1  2  -1   0   0]
    // [0  -1   2  -1   0]
    // [0   0  -1   2  -1]
    // [0   0   0  -1   2]
    var coo = sparse.COO(f64).init(allocator, 5, 5);
    defer coo.deinit();

    // Diagonal
    for (0..5) |i| {
        try coo.append(i, i, 2.0);
    }
    // Upper diagonal
    for (0..4) |i| {
        try coo.append(i, i + 1, -1.0);
    }
    // Lower diagonal
    for (0..4) |i| {
        try coo.append(i + 1, i, -1.0);
    }

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    // b = [1, 0, 0, 0, 1]ᵀ
    const b = [_]f64{ 1.0, 0.0, 0.0, 0.0, 1.0 };

    var result = try conjugateGradient(f64, allocator, &A, &b, null, 1e-10, 100);
    defer result.deinit();

    try testing.expect(result.converged);
    try testing.expect(result.iterations <= 5); // Should converge in ≤ n iterations

    // Verify Ax = b (residual check)
    const Ax = try A.matvec(allocator, result.x);
    defer allocator.free(Ax);

    for (Ax, 0..) |val, i| {
        try testing.expectApproxEqAbs(b[i], val, 1e-8);
    }
}

test "CG: max iterations limit" {
    const allocator = testing.allocator;

    // A = I
    var coo = sparse.COO(f64).init(allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);
    try coo.append(2, 2, 1.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    const b = [_]f64{ 1.0, 2.0, 3.0 };

    // Force early termination with max_iter = 1
    var result = try conjugateGradient(f64, allocator, &A, &b, null, 1e-10, 1);
    defer result.deinit();

    try testing.expect(result.iterations == 1);
    // May or may not have converged in 1 iteration
}

test "CG: dimension mismatch errors" {
    const allocator = testing.allocator;

    // Non-square matrix (2×3)
    var coo = sparse.COO(f64).init(allocator, 2, 3);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    const b = [_]f64{ 1.0, 2.0 };

    const result = conjugateGradient(f64, allocator, &A, &b, null, 1e-6, 100);
    try testing.expectError(error.DimensionMismatch, result);
}

test "CG: b length mismatch" {
    const allocator = testing.allocator;

    var coo = sparse.COO(f64).init(allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    const b = [_]f64{1.0}; // Wrong length

    const result = conjugateGradient(f64, allocator, &A, &b, null, 1e-6, 100);
    try testing.expectError(error.DimensionMismatch, result);
}

test "CG: x0 length mismatch" {
    const allocator = testing.allocator;

    var coo = sparse.COO(f64).init(allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    const b = [_]f64{ 1.0, 2.0 };
    const x0 = [_]f64{1.0}; // Wrong length

    const result = conjugateGradient(f64, allocator, &A, &b, &x0, 1e-6, 100);
    try testing.expectError(error.DimensionMismatch, result);
}

test "CG: memory safety (10 iterations)" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var coo = sparse.COO(f64).init(allocator, 3, 3);
        defer coo.deinit();
        try coo.append(0, 0, 2.0);
        try coo.append(1, 1, 3.0);
        try coo.append(2, 2, 4.0);

        var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
        defer A.deinit();

        const b = [_]f64{ 2.0, 3.0, 4.0 };

        var result = try conjugateGradient(f64, allocator, &A, &b, null, 1e-10, 100);
        defer result.deinit();

        try testing.expect(result.converged);
    }
}

test "CG: f32 precision" {
    const allocator = testing.allocator;

    var coo = sparse.COO(f32).init(allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);

    var A = try sparse.CSR(f32).fromCOO(allocator, &coo);
    defer A.deinit();

    const b = [_]f32{ 3.0, 4.0 };

    var result = try conjugateGradient(f32, allocator, &A, &b, null, 1e-6, 100);
    defer result.deinit();

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(@as(f32, 3.0), result.x[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 4.0), result.x[1], 1e-5);
}
