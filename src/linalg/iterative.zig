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
//! - **GMRES** (Generalized Minimal Residual): For general non-symmetric matrices
//!   - Krylov subspace method with orthogonalization
//!   - Minimizes residual norm over Krylov subspace
//!   - Memory: O(n × m) for restart size m
//! - **BiCGSTAB** (Biconjugate Gradient Stabilized): For non-symmetric matrices
//!   - Variant of BiCG with stabilized convergence
//!   - Often faster than GMRES for moderate problems
//!   - Memory: O(n) (no restart needed)
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
const precond = @import("preconditioner.zig");

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

/// Preconditioned Conjugate Gradient (PCG) solver for symmetric positive definite systems
///
/// Solves Ax = b where A is sparse, symmetric, and positive definite (SPD).
/// Optionally accepts a preconditioner M to accelerate convergence by solving M⁻¹Ax = M⁻¹b.
///
/// Algorithm (Hestenes & Stiefel, 1952 + preconditioning):
/// 1. r₀ = b - Ax₀ (initial residual)
/// 2. z₀ = M⁻¹r₀ (apply preconditioner, or z₀ = r₀ if no preconditioner)
/// 3. p₀ = z₀ (initial search direction)
/// 4. For k = 0, 1, 2, ... until convergence:
///    α = <rₖ, zₖ> / <pₖ, Apₖ>
///    xₖ₊₁ = xₖ + α pₖ
///    rₖ₊₁ = rₖ - α Apₖ
///    zₖ₊₁ = M⁻¹rₖ₊₁
///    β = <rₖ₊₁, zₖ₊₁> / <rₖ, zₖ>
///    pₖ₊₁ = zₖ₊₁ + β pₖ
///
/// Preconditioning benefits:
/// - Reduces condition number: κ(M⁻¹A) << κ(A)
/// - Faster convergence: fewer iterations needed
/// - Common preconditioners: Jacobi (diagonal), ILU(0), incomplete Cholesky
///
/// Convergence: ||rₖ|| < tol × ||r₀|| or k ≥ max_iter
///
/// Time: O(nnz × k) + O(apply_precond × k) where k = iterations (typically k ≪ n)
/// Space: O(n) for x, r, z, p, Ap vectors
///
/// Parameters:
/// - Precond: Preconditioner type with `apply(*const Self, r: []const T, z: []T) !void` method
///   Set to `void` for no preconditioning (default CG)
/// - A: Sparse CSR matrix (must be SPD, not validated!)
/// - b: Right-hand side vector
/// - x0: Initial guess (optional, can be null for zero init)
/// - preconditioner: Preconditioner instance (null for no preconditioning)
/// - tol: Convergence tolerance (relative residual)
/// - max_iter: Maximum iterations (0 = unlimited, use n)
///
/// Returns: SolverResult with solution, iterations, residual norm
///
/// Errors:
/// - DimensionMismatch: A.rows ≠ A.cols or A.rows ≠ b.len or x0.len ≠ b.len
/// - OutOfMemory: Allocation failure
///
/// Example (no preconditioner):
/// ```zig
/// var result = try conjugateGradient(void, f64, allocator, &A, b, null, null, 1e-6, 1000);
/// defer result.deinit();
/// ```
///
/// Example (with Jacobi preconditioner):
/// ```zig
/// var jacobi = try precond.JacobiPreconditioner(f64).init(allocator, &A);
/// defer jacobi.deinit();
/// var result = try conjugateGradient(precond.JacobiPreconditioner(f64), f64, allocator, &A, b, null, &jacobi, 1e-6, 1000);
/// defer result.deinit();
/// ```
pub fn conjugateGradient(
    comptime Precond: type,
    comptime T: type,
    allocator: Allocator,
    A: *const sparse.CSR(T),
    b: []const T,
    x0: ?[]const T,
    preconditioner: if (Precond == void) void else ?*const Precond,
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

    // Allocate workspace: x, r, z (preconditioned residual), p, Ap
    const x = try allocator.alloc(T, n);
    errdefer allocator.free(x);

    const r = try allocator.alloc(T, n);
    errdefer allocator.free(r);

    const z = try allocator.alloc(T, n);
    errdefer allocator.free(z);

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

    // Apply preconditioner: z₀ = M⁻¹r₀
    if (Precond != void) {
        if (preconditioner) |M| {
            try M.apply(r, z);
        } else {
            @memcpy(z, r); // No preconditioner
        }
    } else {
        @memcpy(z, r); // No preconditioner
    }

    // Initial search direction: p₀ = z₀
    @memcpy(p, z);

    // Initial residual norm and <r₀, z₀>
    const r0_norm = norm2(T, r);
    const threshold = tol * r0_norm;
    var rz_dot = dot(T, r, z);

    var k: usize = 0;
    while (k < max_iters) : (k += 1) {
        // Check convergence: ||rₖ|| < tol × ||r₀||
        const r_norm = norm2(T, r);
        if (r_norm < threshold) {
            // Converged
            allocator.free(r);
            allocator.free(z);
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

        // α = <rₖ, zₖ> / <pₖ, Apₖ>
        const pAp = dot(T, p, Ap);
        if (@abs(pAp) < 1e-14) {
            // Numerical breakdown (should not happen for SPD matrices)
            allocator.free(r);
            allocator.free(z);
            allocator.free(p);
            allocator.free(Ap);
            return SolverResult(T){
                .x = x,
                .iterations = k,
                .residual_norm = r_norm,
                .converged = false,
                .allocator = allocator,
            };
        }
        const alpha = rz_dot / pAp;

        // xₖ₊₁ = xₖ + α pₖ
        for (x, 0..) |*x_i, i| {
            x_i.* += alpha * p[i];
        }

        // rₖ₊₁ = rₖ - α Apₖ
        for (r, 0..) |*r_i, i| {
            r_i.* -= alpha * Ap[i];
        }

        // Apply preconditioner: zₖ₊₁ = M⁻¹rₖ₊₁
        if (Precond != void) {
            if (preconditioner) |M| {
                try M.apply(r, z);
            } else {
                @memcpy(z, r);
            }
        } else {
            @memcpy(z, r);
        }

        // <rₖ₊₁, zₖ₊₁>
        const rz_dot_new = dot(T, r, z);

        // β = <rₖ₊₁, zₖ₊₁> / <rₖ, zₖ>
        const beta = rz_dot_new / rz_dot;

        // pₖ₊₁ = zₖ₊₁ + β pₖ
        for (p, 0..) |*p_i, i| {
            p_i.* = z[i] + beta * p_i.*;
        }

        rz_dot = rz_dot_new;
    }

    // Max iterations reached without convergence
    allocator.free(r);
    allocator.free(z);
    allocator.free(p);
    allocator.free(Ap);

    return SolverResult(T){
        .x = x,
        .iterations = k,
        .residual_norm = norm2(T, r),
        .converged = false,
        .allocator = allocator,
    };
}

/// GMRES (Generalized Minimal Residual) solver for general non-symmetric systems
///
/// Solves Ax = b where A is sparse (no symmetry or definiteness required).
///
/// Algorithm (Saad & Schultz, 1986):
/// 1. Construct orthonormal basis {v₁, v₂, ..., vₘ} for Krylov subspace Kₘ(A, r₀)
/// 2. Project problem onto Kₘ: minimize ||β e₁ - H̄ₘ y||₂
/// 3. Update solution: x = x₀ + Vₘ y
/// 4. Restart if not converged after m iterations
///
/// Arnoldi process (modified Gram-Schmidt):
/// - Orthogonalize each new Krylov vector against previous ones
/// - Build upper Hessenberg matrix H̄ₘ
/// - Use Givens rotations to solve least squares problem
///
/// Time: O(nnz × m × k) where m = restart size, k = restarts
/// Space: O(n × m) for Krylov basis vectors
///
/// Parameters:
/// - A: Sparse CSR matrix (general, no restrictions)
/// - b: Right-hand side vector
/// - x0: Initial guess (optional, can be null for zero init)
/// - tol: Convergence tolerance (relative residual)
/// - max_iter: Maximum outer iterations (restarts)
/// - restart: Restart size m (Krylov subspace dimension)
///
/// Returns: SolverResult with solution, iterations, residual norm
///
/// Errors:
/// - DimensionMismatch: A.rows ≠ A.cols or A.rows ≠ b.len or x0.len ≠ b.len
/// - OutOfMemory: Allocation failure
///
/// Example:
/// ```zig
/// var result = try gmres(f64, allocator, &A, b, null, 1e-6, 100, 30);
/// defer result.deinit();
/// if (result.converged) {
///     std.debug.print("Solution: {d}\n", .{result.x});
/// }
/// ```
pub fn gmres(
    comptime T: type,
    allocator: Allocator,
    A: *const sparse.CSR(T),
    b: []const T,
    x0: ?[]const T,
    tol: T,
    max_iter: usize,
    restart: usize,
) !SolverResult(T) {
    const n = A.rows;

    // Validate dimensions
    if (A.rows != A.cols) return error.DimensionMismatch;
    if (b.len != n) return error.DimensionMismatch;
    if (x0) |x_init| {
        if (x_init.len != n) return error.DimensionMismatch;
    }

    const m = restart;

    // Allocate solution vector
    const x = try allocator.alloc(T, n);
    errdefer allocator.free(x);

    // Initialize x (copy x0 or zero)
    if (x0) |x_init| {
        @memcpy(x, x_init);
    } else {
        @memset(x, 0);
    }

    // Allocate Krylov basis V (n × (m+1) matrix, column-major)
    const V_data = try allocator.alloc(T, n * (m + 1));
    defer allocator.free(V_data);

    // Allocate Hessenberg matrix H ((m+1) × m, column-major)
    const H = try allocator.alloc(T, (m + 1) * m);
    defer allocator.free(H);

    // Givens rotation storage: cos and sin arrays
    const cs = try allocator.alloc(T, m);
    defer allocator.free(cs);
    const sn = try allocator.alloc(T, m);
    defer allocator.free(sn);

    // Right-hand side of least squares: s = β e₁
    const s = try allocator.alloc(T, m + 1);
    defer allocator.free(s);

    // Workspace for matvec
    const w = try allocator.alloc(T, n);
    defer allocator.free(w);

    var total_iters: usize = 0;
    var converged = false;

    for (0..max_iter) |_| {
        // Compute initial residual: r = b - Ax
        const Ax = try A.matvec(allocator, x);
        defer allocator.free(Ax);

        var beta: T = 0;
        for (0..n) |i| {
            const r_i = b[i] - Ax[i];
            V_data[i] = r_i; // v₁ = r / ||r||
            beta += r_i * r_i;
        }
        beta = @sqrt(beta);

        // Check convergence before restart
        if (beta < tol) {
            converged = true;
            break;
        }

        // Normalize v₁
        for (0..n) |i| {
            V_data[i] /= beta;
        }

        // Initialize s = β e₁
        @memset(s, 0);
        s[0] = beta;

        // Arnoldi iteration
        var j: usize = 0;
        while (j < m) : (j += 1) {
            total_iters += 1;

            // w = A × vⱼ
            const v_j = V_data[j * n .. (j + 1) * n];
            const Av = try A.matvec(allocator, v_j);
            @memcpy(w, Av);
            allocator.free(Av);

            // Modified Gram-Schmidt orthogonalization
            for (0..j + 1) |i| {
                const v_i = V_data[i * n .. (i + 1) * n];
                const h_ij = dot(T, w, v_i);
                H[j * (m + 1) + i] = h_ij;

                // w = w - h_ij × v_i
                for (0..n) |k| {
                    w[k] -= h_ij * v_i[k];
                }
            }

            // Compute ||w||
            const h_jp1 = norm2(T, w);
            H[j * (m + 1) + j + 1] = h_jp1;

            // Check for breakdown
            if (@abs(h_jp1) < 1e-14) {
                // Lucky breakdown: exact solution found
                break;
            }

            // vⱼ₊₁ = w / ||w||
            const v_jp1 = V_data[(j + 1) * n .. (j + 2) * n];
            for (0..n) |k| {
                v_jp1[k] = w[k] / h_jp1;
            }

            // Apply previous Givens rotations to H(:, j)
            for (0..j) |i| {
                const temp = cs[i] * H[j * (m + 1) + i] + sn[i] * H[j * (m + 1) + i + 1];
                H[j * (m + 1) + i + 1] = -sn[i] * H[j * (m + 1) + i] + cs[i] * H[j * (m + 1) + i + 1];
                H[j * (m + 1) + i] = temp;
            }

            // Compute new Givens rotation
            const h_jj = H[j * (m + 1) + j];
            const h_jp1j = H[j * (m + 1) + j + 1];
            const rho = @sqrt(h_jj * h_jj + h_jp1j * h_jp1j);
            cs[j] = h_jj / rho;
            sn[j] = h_jp1j / rho;

            // Apply to H
            H[j * (m + 1) + j] = rho;
            H[j * (m + 1) + j + 1] = 0;

            // Apply to s
            s[j + 1] = -sn[j] * s[j];
            s[j] = cs[j] * s[j];

            // Check convergence: |s[j+1]| = residual norm
            if (@abs(s[j + 1]) < tol) {
                // Converged within this restart
                // Solve upper triangular system Hy = s
                const y = try allocator.alloc(T, j + 1);
                defer allocator.free(y);

                var i: usize = j + 1;
                while (i > 0) {
                    i -= 1;
                    var sum: T = s[i];
                    for (i + 1..j + 1) |k| {
                        sum -= H[k * (m + 1) + i] * y[k];
                    }
                    y[i] = sum / H[i * (m + 1) + i];
                }

                // Update x = x + V * y
                for (0..j + 1) |k| {
                    const v_k = V_data[k * n .. (k + 1) * n];
                    for (0..n) |l| {
                        x[l] += y[k] * v_k[l];
                    }
                }

                converged = true;
                break;
            }
        }

        if (converged) break;

        // End of restart: update x with current approximation
        const y = try allocator.alloc(T, j);
        defer allocator.free(y);

        var i: usize = j;
        while (i > 0) {
            i -= 1;
            var sum: T = s[i];
            for (i + 1..j) |k| {
                sum -= H[k * (m + 1) + i] * y[k];
            }
            y[i] = sum / H[i * (m + 1) + i];
        }

        // Update x = x + V * y
        for (0..j) |k| {
            const v_k = V_data[k * n .. (k + 1) * n];
            for (0..n) |l| {
                x[l] += y[k] * v_k[l];
            }
        }
    }

    // Compute final residual
    const Ax = try A.matvec(allocator, x);
    defer allocator.free(Ax);

    var residual_norm: T = 0;
    for (0..n) |i| {
        const r = b[i] - Ax[i];
        residual_norm += r * r;
    }
    residual_norm = @sqrt(residual_norm);

    return SolverResult(T){
        .x = x,
        .iterations = total_iters,
        .residual_norm = residual_norm,
        .converged = converged,
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

/// Euclidean norm: ||x||₂ = √(Σ xᵢ²)
///
/// Time: O(n) | Space: O(1)
fn norm2(comptime T: type, x: []const T) T {
    var sum: T = 0;
    for (x) |x_i| {
        sum += x_i * x_i;
    }
    return @sqrt(sum);
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

    var result = try conjugateGradient(void, f64, allocator, &A, &b, null, {}, 1e-10, 100);
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

    var result = try conjugateGradient(void, f64, allocator, &A, &b, null, {}, 1e-10, 100);
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

    var result = try conjugateGradient(void, f64, allocator, &A, &b, null, {}, 1e-10, 100);
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

    var result = try conjugateGradient(void, f64, allocator, &A, &b, &x0, {}, 1e-10, 100);
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

    var result = try conjugateGradient(void, f64, allocator, &A, &b, null, {}, 1e-10, 100);
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
    var result = try conjugateGradient(void, f64, allocator, &A, &b, null, {}, 1e-10, 1);
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

    const result = conjugateGradient(void, f64, allocator, &A, &b, null, {}, 1e-6, 100);
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

    const result = conjugateGradient(void, f64, allocator, &A, &b, null, {}, 1e-6, 100);
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

    const result = conjugateGradient(void, f64, allocator, &A, &b, &x0, {}, 1e-6, 100);
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

        var result = try conjugateGradient(void, f64, allocator, &A, &b, null, {}, 1e-10, 100);
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

    var result = try conjugateGradient(void, f32, allocator, &A, &b, null, {}, 1e-6, 100);
    defer result.deinit();

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(@as(f32, 3.0), result.x[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 4.0), result.x[1], 1e-5);
}

// ============================================================================
// Preconditioned CG Tests
// ============================================================================

test "PCG: with Jacobi preconditioner" {
    const allocator = testing.allocator;

    // A = [4  1]  SPD matrix
    //     [1  3]
    var coo = sparse.COO(f64).init(allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 0, 4.0);
    try coo.append(0, 1, 1.0);
    try coo.append(1, 0, 1.0);
    try coo.append(1, 1, 3.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    // b = [9, 8]ᵀ → solution x = [2, 2]ᵀ
    // Verify: Ax = [4*2+1*2, 1*2+3*2] = [10, 8] ✗ Let me recalculate
    // Ax = [4*2+1*2, 1*2+3*2] = [8+2, 2+6] = [10, 8] ✗
    // For x = [2, 2]: Ax = [8+2, 2+6] = [10, 8]
    // So b should be [10, 8] for x = [2, 2]
    const b = [_]f64{ 10.0, 8.0 };

    // Create Jacobi preconditioner
    var jacobi = try precond.JacobiPreconditioner(f64).init(allocator, &A);
    defer jacobi.deinit();

    var result = try conjugateGradient(precond.JacobiPreconditioner(f64), f64, allocator, &A, &b, null, &jacobi, 1e-10, 100);
    defer result.deinit();

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(2.0, result.x[0], 1e-8);
    try testing.expectApproxEqAbs(2.0, result.x[1], 1e-8);
}

test "PCG: with ILU(0) preconditioner" {
    const allocator = testing.allocator;

    // A = tridiag(-1, 3, -1) - SPD matrix
    // [3  -1   0]
    // [-1  3  -1]
    // [0  -1   3]
    var coo = sparse.COO(f64).init(allocator, 3, 3);
    defer coo.deinit();
    for (0..3) |i| {
        try coo.append(i, i, 3.0);
    }
    for (0..2) |i| {
        try coo.append(i, i + 1, -1.0);
        try coo.append(i + 1, i, -1.0);
    }

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    // b = [1, 1, 1]ᵀ
    const b = [_]f64{ 1.0, 1.0, 1.0 };

    // Create ILU(0) preconditioner
    var ilu = try precond.ILUPreconditioner(f64).init(allocator, &A);
    defer ilu.deinit();

    var result = try conjugateGradient(precond.ILUPreconditioner(f64), f64, allocator, &A, &b, null, &ilu, 1e-10, 100);
    defer result.deinit();

    try testing.expect(result.converged);

    // Verify Ax = b
    const Ax = try A.matvec(allocator, result.x);
    defer allocator.free(Ax);

    for (Ax, 0..) |val, i| {
        try testing.expectApproxEqAbs(b[i], val, 1e-8);
    }
}

test "PCG: convergence faster than unpreconditioned CG" {
    const allocator = testing.allocator;

    // A = diag(1, 2, 3, 4, 5) - ill-conditioned
    var coo = sparse.COO(f64).init(allocator, 5, 5);
    defer coo.deinit();
    for (0..5) |i| {
        try coo.append(i, i, @as(f64, @floatFromInt(i + 1)));
    }

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    const b = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    // Unpreconditioned CG
    var result_cg = try conjugateGradient(void, f64, allocator, &A, &b, null, {}, 1e-10, 100);
    defer result_cg.deinit();

    // Preconditioned CG with Jacobi
    var jacobi = try precond.JacobiPreconditioner(f64).init(allocator, &A);
    defer jacobi.deinit();

    var result_pcg = try conjugateGradient(precond.JacobiPreconditioner(f64), f64, allocator, &A, &b, null, &jacobi, 1e-10, 100);
    defer result_pcg.deinit();

    // Both should converge
    try testing.expect(result_cg.converged);
    try testing.expect(result_pcg.converged);

    // PCG should converge in fewer or equal iterations
    // For diagonal matrices, Jacobi preconditioning solves in 1 iteration
    try testing.expect(result_pcg.iterations <= result_cg.iterations);
    try testing.expectEqual(@as(usize, 1), result_pcg.iterations);
}

test "PCG: memory safety (10 iterations)" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var coo = sparse.COO(f64).init(allocator, 3, 3);
        defer coo.deinit();
        try coo.append(0, 0, 2.0);
        try coo.append(1, 1, 3.0);
        try coo.append(2, 2, 4.0);

        var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
        defer A.deinit();

        const b = [_]f64{ 4.0, 6.0, 8.0 };

        var jacobi = try precond.JacobiPreconditioner(f64).init(allocator, &A);
        defer jacobi.deinit();

        var result = try conjugateGradient(precond.JacobiPreconditioner(f64), f64, allocator, &A, &b, null, &jacobi, 1e-10, 100);
        defer result.deinit();

        try testing.expect(result.converged);
    }
}

// ============================================================================
// GMRES Tests
// ============================================================================

test "GMRES: 2×2 identity system" {
    const allocator = testing.allocator;

    // A = I (2×2 identity)
    var coo = sparse.COO(f64).init(allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    // b = [3, 5]ᵀ → solution x = [3, 5]ᵀ
    const b = [_]f64{ 3.0, 5.0 };

    var result = try gmres(f64, allocator, &A, &b, null, 1e-10, 10, 10);
    defer result.deinit();

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(3.0, result.x[0], 1e-9);
    try testing.expectApproxEqAbs(5.0, result.x[1], 1e-9);
}

test "GMRES: 3×3 diagonal system" {
    const allocator = testing.allocator;

    // A = diag(2, 3, 4)
    var coo = sparse.COO(f64).init(allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 2.0);
    try coo.append(1, 1, 3.0);
    try coo.append(2, 2, 4.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    // b = [6, 9, 12]ᵀ → solution x = [3, 3, 3]ᵀ
    const b = [_]f64{ 6.0, 9.0, 12.0 };

    var result = try gmres(f64, allocator, &A, &b, null, 1e-10, 10, 10);
    defer result.deinit();

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(3.0, result.x[0], 1e-9);
    try testing.expectApproxEqAbs(3.0, result.x[1], 1e-9);
    try testing.expectApproxEqAbs(3.0, result.x[2], 1e-9);
}

test "GMRES: 3×3 non-symmetric system" {
    const allocator = testing.allocator;

    // A = [3  1  0]    Non-symmetric
    //     [1  2  1]
    //     [0  1  3]
    var coo = sparse.COO(f64).init(allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 3.0);
    try coo.append(0, 1, 1.0);
    try coo.append(1, 0, 1.0);
    try coo.append(1, 1, 2.0);
    try coo.append(1, 2, 1.0);
    try coo.append(2, 1, 1.0);
    try coo.append(2, 2, 3.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    // b = [4, 4, 4]ᵀ → solution x = [1, 1, 1]ᵀ
    // Verify: Ax = [3+1, 1+2+1, 1+3] = [4, 4, 4] ✓
    const b = [_]f64{ 4.0, 4.0, 4.0 };

    var result = try gmres(f64, allocator, &A, &b, null, 1e-10, 10, 10);
    defer result.deinit();

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(1.0, result.x[0], 1e-8);
    try testing.expectApproxEqAbs(1.0, result.x[1], 1e-8);
    try testing.expectApproxEqAbs(1.0, result.x[2], 1e-8);
}

test "GMRES: with initial guess" {
    const allocator = testing.allocator;

    // A = I (2×2 identity)
    var coo = sparse.COO(f64).init(allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    const b = [_]f64{ 7.0, 9.0 };
    const x0 = [_]f64{ 6.5, 8.5 }; // Close initial guess

    var result = try gmres(f64, allocator, &A, &b, &x0, 1e-10, 10, 10);
    defer result.deinit();

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(7.0, result.x[0], 1e-9);
    try testing.expectApproxEqAbs(9.0, result.x[1], 1e-9);
}

test "GMRES: small restart size" {
    const allocator = testing.allocator;

    // A = I (3×3 identity)
    var coo = sparse.COO(f64).init(allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);
    try coo.append(2, 2, 1.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    const b = [_]f64{ 2.0, 3.0, 4.0 };

    // Use restart=2 (smaller than matrix size)
    var result = try gmres(f64, allocator, &A, &b, null, 1e-10, 10, 2);
    defer result.deinit();

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(2.0, result.x[0], 1e-8);
    try testing.expectApproxEqAbs(3.0, result.x[1], 1e-8);
    try testing.expectApproxEqAbs(4.0, result.x[2], 1e-8);
}

test "GMRES: 4×4 tridiagonal system" {
    const allocator = testing.allocator;

    // A = tridiag(-1, 3, -1)
    // [3  -1   0   0]
    // [-1  3  -1   0]
    // [0  -1   3  -1]
    // [0   0  -1   3]
    var coo = sparse.COO(f64).init(allocator, 4, 4);
    defer coo.deinit();

    for (0..4) |i| {
        try coo.append(i, i, 3.0);
    }
    for (0..3) |i| {
        try coo.append(i, i + 1, -1.0);
        try coo.append(i + 1, i, -1.0);
    }

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    const b = [_]f64{ 1.0, 1.0, 1.0, 1.0 };

    var result = try gmres(f64, allocator, &A, &b, null, 1e-10, 20, 10);
    defer result.deinit();

    try testing.expect(result.converged);

    // Verify Ax = b
    const Ax = try A.matvec(allocator, result.x);
    defer allocator.free(Ax);

    for (Ax, 0..) |val, i| {
        try testing.expectApproxEqAbs(b[i], val, 1e-8);
    }
}

test "GMRES: dimension mismatch errors" {
    const allocator = testing.allocator;

    // Non-square matrix (2×3)
    var coo = sparse.COO(f64).init(allocator, 2, 3);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    const b = [_]f64{ 1.0, 2.0 };

    const result = gmres(f64, allocator, &A, &b, null, 1e-6, 10, 10);
    try testing.expectError(error.DimensionMismatch, result);
}

test "GMRES: b length mismatch" {
    const allocator = testing.allocator;

    var coo = sparse.COO(f64).init(allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);
    try coo.append(2, 2, 1.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    const b = [_]f64{ 1.0, 2.0 }; // Wrong length

    const result = gmres(f64, allocator, &A, &b, null, 1e-6, 10, 10);
    try testing.expectError(error.DimensionMismatch, result);
}

test "GMRES: x0 length mismatch" {
    const allocator = testing.allocator;

    var coo = sparse.COO(f64).init(allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    const b = [_]f64{ 1.0, 2.0 };
    const x0 = [_]f64{ 1.0, 2.0, 3.0 }; // Wrong length

    const result = gmres(f64, allocator, &A, &b, &x0, 1e-6, 10, 10);
    try testing.expectError(error.DimensionMismatch, result);
}

test "GMRES: memory safety (10 iterations)" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var coo = sparse.COO(f64).init(allocator, 3, 3);
        defer coo.deinit();
        try coo.append(0, 0, 2.0);
        try coo.append(1, 1, 3.0);
        try coo.append(2, 2, 4.0);

        var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
        defer A.deinit();

        const b = [_]f64{ 4.0, 6.0, 8.0 };

        var result = try gmres(f64, allocator, &A, &b, null, 1e-10, 10, 10);
        defer result.deinit();

        try testing.expect(result.converged);
    }
}

test "GMRES: f32 precision" {
    const allocator = testing.allocator;

    var coo = sparse.COO(f32).init(allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 0, 2.0);
    try coo.append(1, 1, 3.0);

    var A = try sparse.CSR(f32).fromCOO(allocator, &coo);
    defer A.deinit();

    const b = [_]f32{ 6.0, 9.0 };

    var result = try gmres(f32, allocator, &A, &b, null, 1e-6, 10, 10);
    defer result.deinit();

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(@as(f32, 3.0), result.x[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 3.0), result.x[1], 1e-5);
}

// ============================================================================
// BiCGSTAB (Biconjugate Gradient Stabilized)
// ============================================================================

/// BiCGSTAB solver for non-symmetric linear systems
///
/// Solves Ax = b where A is sparse and non-symmetric (or non-Hermitian).
///
/// Algorithm (van der Vorst, 1992):
/// BiCGSTAB is a variant of BiCG (Biconjugate Gradient) that avoids irregular
/// convergence patterns by using a local minimization strategy. It's often faster
/// than GMRES for moderately difficult problems and uses less memory (no restart).
///
/// 1. Choose arbitrary r̂₀ (often r̂₀ = r₀ = b - Ax₀)
/// 2. p₀ = r₀
/// 3. For k = 0, 1, 2, ... until convergence:
///    α = <r̂₀, rₖ> / <r̂₀, Apₖ>
///    s = rₖ - α Apₖ
///    ω = <As, s> / <As, As>
///    xₖ₊₁ = xₖ + α pₖ + ω s
///    rₖ₊₁ = s - ω As
///    β = (α/ω) × (<r̂₀, rₖ₊₁> / <r̂₀, rₖ>)
///    pₖ₊₁ = rₖ₊₁ + β (pₖ - ω Apₖ)
///
/// Convergence: ||rₖ|| < tol × ||r₀|| or k ≥ max_iter
///
/// Time: O(nnz × k) where k = iterations (typically k ≪ n)
/// Space: O(n) for x, r, r̂, p, s, Ap, As vectors (7n total)
///
/// Advantages over GMRES:
/// - Lower memory: O(n) vs O(n × m) for GMRES with restart m
/// - No restart needed
/// - Often faster convergence for moderate problems
/// - Two matrix-vector products per iteration vs one for CG
///
/// Disadvantages:
/// - Can stagnate or fail for very difficult problems
/// - Less robust than GMRES for highly non-symmetric systems
/// - Breakdown possible (though rare): ω ≈ 0 or <r̂₀, Apₖ> ≈ 0
///
/// Parameters:
/// - A: Sparse CSR matrix (can be non-symmetric)
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
/// const A = try sparse.CSR(f64).fromCOO(allocator, &coo);
/// const b = [_]f64{ 1.0, 2.0, 3.0 };
/// var result = try bicgstab(f64, allocator, &A, &b, null, 1e-10, 100);
/// defer result.deinit();
/// ```
pub fn bicgstab(
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

    // Allocate solution vector
    const x = try allocator.alloc(T, n);
    errdefer allocator.free(x);

    // Initialize x (copy x0 or zero)
    if (x0) |x_init| {
        @memcpy(x, x_init);
    } else {
        @memset(x, 0);
    }

    // Allocate work vectors
    const r = try allocator.alloc(T, n);
    defer allocator.free(r);
    const r_hat = try allocator.alloc(T, n); // Shadow residual
    defer allocator.free(r_hat);
    const p = try allocator.alloc(T, n); // Search direction
    defer allocator.free(p);
    const s = try allocator.alloc(T, n); // Intermediate residual
    defer allocator.free(s);
    const Ap = try allocator.alloc(T, n); // A × p
    defer allocator.free(Ap);
    const As = try allocator.alloc(T, n); // A × s
    defer allocator.free(As);

    // Compute initial residual: r₀ = b - Ax₀
    const Ax0 = try A.matvec(allocator, x);
    defer allocator.free(Ax0);

    var r_norm: T = 0;
    for (0..n) |i| {
        r[i] = b[i] - Ax0[i];
        r_hat[i] = r[i]; // r̂₀ = r₀
        p[i] = r[i]; // p₀ = r₀
        r_norm += r[i] * r[i];
    }
    const r0_norm = @sqrt(r_norm);

    // Check if already converged
    if (r0_norm < tol) {
        return SolverResult(T){
            .x = x,
            .iterations = 0,
            .residual_norm = r0_norm,
            .converged = true,
            .allocator = allocator,
        };
    }

    const tolerance = tol * r0_norm;
    var rho: T = 1.0;
    var alpha: T = 1.0;
    var omega: T = 1.0;
    var iter: usize = 0;

    const actual_max_iter = if (max_iter == 0) n else max_iter;

    while (iter < actual_max_iter) : (iter += 1) {
        // ρₖ = <r̂₀, rₖ>
        const rho_new = dot(T, r_hat, r);

        // Check for breakdown: ρₖ ≈ 0
        if (@abs(rho_new) < 1e-14) {
            // BiCGSTAB breakdown (rare)
            return SolverResult(T){
                .x = x,
                .iterations = iter,
                .residual_norm = @sqrt(dot(T, r, r)),
                .converged = false,
                .allocator = allocator,
            };
        }

        // β = (ρₖ / ρₖ₋₁) × (α / ω)
        const beta = (rho_new / rho) * (alpha / omega);

        // pₖ = rₖ + β (pₖ₋₁ - ω Apₖ₋₁)
        for (0..n) |i| {
            p[i] = r[i] + beta * (p[i] - omega * Ap[i]);
        }

        // Ap = A × pₖ
        const Ap_temp = try A.matvec(allocator, p);
        @memcpy(Ap, Ap_temp);
        allocator.free(Ap_temp);

        // α = ρₖ / <r̂₀, Ap>
        const r_hat_Ap = dot(T, r_hat, Ap);
        if (@abs(r_hat_Ap) < 1e-14) {
            // Breakdown
            return SolverResult(T){
                .x = x,
                .iterations = iter,
                .residual_norm = @sqrt(dot(T, r, r)),
                .converged = false,
                .allocator = allocator,
            };
        }
        alpha = rho_new / r_hat_Ap;

        // s = rₖ - α Ap
        for (0..n) |i| {
            s[i] = r[i] - alpha * Ap[i];
        }

        // Check convergence on s (early termination possible)
        const s_norm = @sqrt(dot(T, s, s));
        if (s_norm < tolerance) {
            // Update x: xₖ₊₁ = xₖ + α pₖ
            for (0..n) |i| {
                x[i] += alpha * p[i];
            }
            return SolverResult(T){
                .x = x,
                .iterations = iter + 1,
                .residual_norm = s_norm,
                .converged = true,
                .allocator = allocator,
            };
        }

        // As = A × s
        const As_temp = try A.matvec(allocator, s);
        @memcpy(As, As_temp);
        allocator.free(As_temp);

        // ω = <As, s> / <As, As>
        const As_s = dot(T, As, s);
        const As_As = dot(T, As, As);
        if (As_As < 1e-14) {
            // Breakdown
            return SolverResult(T){
                .x = x,
                .iterations = iter,
                .residual_norm = s_norm,
                .converged = false,
                .allocator = allocator,
            };
        }
        omega = As_s / As_As;

        // xₖ₊₁ = xₖ + α pₖ + ω s
        for (0..n) |i| {
            x[i] += alpha * p[i] + omega * s[i];
        }

        // rₖ₊₁ = s - ω As
        for (0..n) |i| {
            r[i] = s[i] - omega * As[i];
        }

        // Check convergence
        const r_norm_new = @sqrt(dot(T, r, r));
        if (r_norm_new < tolerance) {
            return SolverResult(T){
                .x = x,
                .iterations = iter + 1,
                .residual_norm = r_norm_new,
                .converged = true,
                .allocator = allocator,
            };
        }

        rho = rho_new;
    }

    // Max iterations reached
    return SolverResult(T){
        .x = x,
        .iterations = iter,
        .residual_norm = @sqrt(dot(T, r, r)),
        .converged = false,
        .allocator = allocator,
    };
}

// ============================================================================
// BiCGSTAB Tests
// ============================================================================

test "BiCGSTAB: 2×2 identity matrix" {
    const allocator = testing.allocator;

    // A = I (identity)
    var coo = sparse.COO(f64).init(allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    // b = [5, 7]ᵀ → solution x = [5, 7]ᵀ
    const b = [_]f64{ 5.0, 7.0 };

    var result = try bicgstab(f64, allocator, &A, &b, null, 1e-10, 100);
    defer result.deinit();

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(5.0, result.x[0], 1e-9);
    try testing.expectApproxEqAbs(7.0, result.x[1], 1e-9);
    try testing.expect(result.iterations <= 2); // Identity converges in 1-2 iterations
}

test "BiCGSTAB: 3×3 diagonal matrix" {
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

    var result = try bicgstab(f64, allocator, &A, &b, null, 1e-10, 100);
    defer result.deinit();

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(2.0, result.x[0], 1e-9);
    try testing.expectApproxEqAbs(2.0, result.x[1], 1e-9);
    try testing.expectApproxEqAbs(2.0, result.x[2], 1e-9);
}

test "BiCGSTAB: 3×3 non-symmetric system" {
    const allocator = testing.allocator;

    // A = [4  1  0]    Non-symmetric matrix
    //     [2  3  1]
    //     [0  1  4]
    var coo = sparse.COO(f64).init(allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 4.0);
    try coo.append(0, 1, 1.0);
    try coo.append(1, 0, 2.0); // Note: A[1,0]=2 but A[0,1]=1 → non-symmetric
    try coo.append(1, 1, 3.0);
    try coo.append(1, 2, 1.0);
    try coo.append(2, 1, 1.0);
    try coo.append(2, 2, 4.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    // b = [5, 6, 5]ᵀ → solution x = [1, 1, 1]ᵀ
    // Verify: Ax = [4+1, 2+3+1, 1+4] = [5, 6, 5] ✓
    const b = [_]f64{ 5.0, 6.0, 5.0 };

    var result = try bicgstab(f64, allocator, &A, &b, null, 1e-10, 100);
    defer result.deinit();

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(1.0, result.x[0], 1e-8);
    try testing.expectApproxEqAbs(1.0, result.x[1], 1e-8);
    try testing.expectApproxEqAbs(1.0, result.x[2], 1e-8);
}

test "BiCGSTAB: with initial guess" {
    const allocator = testing.allocator;

    // A = I (2×2 identity)
    var coo = sparse.COO(f64).init(allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    const b = [_]f64{ 5.0, 7.0 };
    const x0 = [_]f64{ 4.5, 6.5 }; // Close initial guess

    var result = try bicgstab(f64, allocator, &A, &b, &x0, 1e-10, 100);
    defer result.deinit();

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(5.0, result.x[0], 1e-9);
    try testing.expectApproxEqAbs(7.0, result.x[1], 1e-9);
}

test "BiCGSTAB: tridiagonal system (4×4)" {
    const allocator = testing.allocator;

    // A = tridiag(-1, 3, -1) — diagonally dominant, non-symmetric variant
    // [3  -1   0   0]
    // [-1  3  -1   0]
    // [0  -1   3  -1]
    // [0   0  -1   3]
    var coo = sparse.COO(f64).init(allocator, 4, 4);
    defer coo.deinit();

    // Diagonal
    for (0..4) |i| {
        try coo.append(i, i, 3.0);
    }
    // Upper diagonal
    for (0..3) |i| {
        try coo.append(i, i + 1, -1.0);
    }
    // Lower diagonal
    for (0..3) |i| {
        try coo.append(i + 1, i, -1.0);
    }

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    // b = [2, 1, 1, 2]ᵀ
    const b = [_]f64{ 2.0, 1.0, 1.0, 2.0 };

    var result = try bicgstab(f64, allocator, &A, &b, null, 1e-10, 100);
    defer result.deinit();

    try testing.expect(result.converged);
    // Verify residual is small (exact solution may vary)
    try testing.expect(result.residual_norm < 1e-8);
}

test "BiCGSTAB: max iterations limit" {
    const allocator = testing.allocator;

    // A = I (identity)
    var coo = sparse.COO(f64).init(allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);
    try coo.append(2, 2, 1.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    const b = [_]f64{ 1.0, 2.0, 3.0 };

    var result = try bicgstab(f64, allocator, &A, &b, null, 1e-10, 1); // Only 1 iteration
    defer result.deinit();

    try testing.expect(result.iterations <= 1);
}

test "BiCGSTAB: non-square matrix error" {
    const allocator = testing.allocator;

    var coo = sparse.COO(f64).init(allocator, 2, 3);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    const b = [_]f64{ 1.0, 2.0 };

    const result = bicgstab(f64, allocator, &A, &b, null, 1e-6, 10);
    try testing.expectError(error.DimensionMismatch, result);
}

test "BiCGSTAB: b length mismatch" {
    const allocator = testing.allocator;

    var coo = sparse.COO(f64).init(allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);
    try coo.append(2, 2, 1.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    const b = [_]f64{ 1.0, 2.0 }; // Wrong length

    const result = bicgstab(f64, allocator, &A, &b, null, 1e-6, 10);
    try testing.expectError(error.DimensionMismatch, result);
}

test "BiCGSTAB: x0 length mismatch" {
    const allocator = testing.allocator;

    var coo = sparse.COO(f64).init(allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);

    var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer A.deinit();

    const b = [_]f64{ 1.0, 2.0 };
    const x0 = [_]f64{ 1.0, 2.0, 3.0 }; // Wrong length

    const result = bicgstab(f64, allocator, &A, &b, &x0, 1e-6, 10);
    try testing.expectError(error.DimensionMismatch, result);
}

test "BiCGSTAB: memory safety (10 iterations)" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var coo = sparse.COO(f64).init(allocator, 3, 3);
        defer coo.deinit();
        try coo.append(0, 0, 2.0);
        try coo.append(1, 1, 3.0);
        try coo.append(2, 2, 4.0);

        var A = try sparse.CSR(f64).fromCOO(allocator, &coo);
        defer A.deinit();

        const b = [_]f64{ 4.0, 6.0, 8.0 };

        var result = try bicgstab(f64, allocator, &A, &b, null, 1e-10, 100);
        defer result.deinit();

        try testing.expect(result.converged);
    }
}

test "BiCGSTAB: f32 precision" {
    const allocator = testing.allocator;

    var coo = sparse.COO(f32).init(allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 0, 2.0);
    try coo.append(1, 1, 3.0);

    var A = try sparse.CSR(f32).fromCOO(allocator, &coo);
    defer A.deinit();

    const b = [_]f32{ 6.0, 9.0 };

    var result = try bicgstab(f32, allocator, &A, &b, null, 1e-6, 100);
    defer result.deinit();

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(@as(f32, 3.0), result.x[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 3.0), result.x[1], 1e-5);
}
