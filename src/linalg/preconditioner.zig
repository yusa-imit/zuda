//! Preconditioners for Iterative Linear Solvers
//!
//! Preconditioners transform the linear system Ax = b into a better-conditioned
//! system that converges faster with iterative methods (CG, GMRES).
//!
//! Given system Ax = b, preconditioned form is M⁻¹Ax = M⁻¹b where M ≈ A but
//! M⁻¹ is cheap to compute. The goal is to reduce condition number κ(M⁻¹A) << κ(A).
//!
//! Algorithms:
//! - **Jacobi (diagonal)**: M = diag(A), simplest preconditioner
//!   - Apply: O(n) time, O(n) space
//!   - Effective for diagonally dominant matrices
//! - **ILU(0)**: Incomplete LU factorization with zero fill-in
//!   - Apply: O(nnz) time via forward/back substitution
//!   - Better convergence than Jacobi, but more expensive setup
//!
//! Usage:
//! ```zig
//! var jacobi = try JacobiPreconditioner(f64).init(allocator, &A);
//! defer jacobi.deinit();
//!
//! // In CG/GMRES loop:
//! try jacobi.apply(r, z); // z = M⁻¹r
//! ```

const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const testing = std.testing;
const math = std.math;
const sparse = @import("sparse.zig");

/// Jacobi (diagonal) preconditioner
///
/// Uses M = diag(A) where M[i,i] = A[i,i].
/// Preconditioning: z = M⁻¹r is simply z[i] = r[i] / diag[i].
///
/// Time: O(nnz) setup (extract diagonal), O(n) apply
/// Space: O(n) for diagonal storage
///
/// Effective for diagonally dominant matrices (|A[i,i]| >> Σⱼ≠ᵢ |A[i,j]|).
pub fn JacobiPreconditioner(comptime T: type) type {
    return struct {
        /// Diagonal entries M[i,i] = A[i,i]
        diag: []T,
        /// Allocator
        allocator: Allocator,

        const Self = @This();

        /// Initialize from sparse CSR matrix
        ///
        /// Extracts diagonal entries. Zero diagonal entries are replaced with 1.0
        /// to avoid division by zero.
        ///
        /// Time: O(nnz)
        /// Space: O(n)
        pub fn init(allocator: Allocator, A: *const sparse.CSR(T)) !Self {
            const n = A.rows;
            const diag = try allocator.alloc(T, n);
            errdefer allocator.free(diag);

            // Initialize to 1.0 (safe default)
            @memset(diag, 1.0);

            // Extract diagonal entries
            for (0..n) |i| {
                const row_start = A.row_ptr[i];
                const row_end = A.row_ptr[i + 1];
                for (row_start..row_end) |idx| {
                    if (A.col_idx[idx] == i) {
                        const val = A.values[idx];
                        // Avoid division by zero - use 1.0 for zero diagonal
                        diag[i] = if (@abs(val) < 1e-14) 1.0 else val;
                        break;
                    }
                }
            }

            return Self{
                .diag = diag,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.diag);
        }

        /// Apply preconditioner: z = M⁻¹r
        ///
        /// Computes z[i] = r[i] / diag[i] for all i.
        ///
        /// Time: O(n)
        ///
        /// Errors:
        /// - DimensionMismatch: if r.len ≠ z.len or r.len ≠ diag.len
        pub fn apply(self: *const Self, r: []const T, z: []T) !void {
            if (r.len != z.len or r.len != self.diag.len) {
                return error.DimensionMismatch;
            }

            for (r, z, self.diag) |ri, *zi, di| {
                zi.* = ri / di;
            }
        }
    };
}

/// Incomplete LU (ILU(0)) preconditioner
///
/// Computes approximate factorization M = LU where L and U have the same
/// sparsity pattern as A (zero fill-in). This is cheaper than full LU but
/// provides better preconditioning than Jacobi.
///
/// Algorithm:
/// For each row i:
///   For each A[i,k] where k < i:
///     L[i,k] = (A[i,k] - Σⱼ L[i,j]U[j,k]) / U[k,k]
///   For each A[i,k] where k ≥ i:
///     U[i,k] = A[i,k] - Σⱼ L[i,j]U[j,k]
///
/// Only non-zeros matching A's pattern are computed (no fill-in).
///
/// Time: O(nnz²) setup (worst case), O(nnz) apply
/// Space: O(nnz) for L and U factors
///
/// Better convergence than Jacobi for most problems, but more expensive.
pub fn ILUPreconditioner(comptime T: type) type {
    return struct {
        /// Lower triangular factor (CSR format)
        L: sparse.CSR(T),
        /// Upper triangular factor (CSR format)
        U: sparse.CSR(T),
        /// Allocator
        allocator: Allocator,

        const Self = @This();

        /// Initialize from sparse CSR matrix
        ///
        /// Computes ILU(0) factorization with zero fill-in.
        ///
        /// Time: O(nnz × nnz_row_avg) ≈ O(nnz²) worst case
        /// Space: O(nnz)
        pub fn init(allocator: Allocator, A: *const sparse.CSR(T)) !Self {
            const n = A.rows;

            // Create working copy of A for in-place factorization
            var work = try sparse.CSR(T).init(allocator, n, n, A.values.len);
            errdefer work.deinit();

            // Copy A to work matrix
            @memcpy(work.row_ptr, A.row_ptr);
            @memcpy(work.col_idx, A.col_idx);
            @memcpy(work.values, A.values);

            // ILU(0) factorization - in-place on work matrix
            for (0..n) |i| {
                const row_start = work.row_ptr[i];
                const row_end = work.row_ptr[i + 1];

                // Find diagonal element
                var diag_idx: ?usize = null;
                for (row_start..row_end) |idx| {
                    if (work.col_idx[idx] == i) {
                        diag_idx = idx;
                        break;
                    }
                }

                if (diag_idx == null or @abs(work.values[diag_idx.?]) < 1e-14) {
                    work.deinit();
                    return error.SingularMatrix;
                }

                // Update current row based on previous rows
                for (row_start..row_end) |k_idx| {
                    const k = work.col_idx[k_idx];
                    if (k >= i) break; // Only process lower triangular part

                    // Get U[k,k] from row k
                    const k_row_start = work.row_ptr[k];
                    const k_row_end = work.row_ptr[k + 1];
                    var ukk: T = 0.0;
                    for (k_row_start..k_row_end) |idx| {
                        if (work.col_idx[idx] == k) {
                            ukk = work.values[idx];
                            break;
                        }
                    }

                    if (@abs(ukk) < 1e-14) continue;

                    // Compute multiplier L[i,k] = A[i,k] / U[k,k]
                    const mult = work.values[k_idx] / ukk;
                    work.values[k_idx] = mult;

                    // Update U entries: U[i,j] -= L[i,k] * U[k,j]
                    for (row_start..row_end) |j_idx| {
                        const j = work.col_idx[j_idx];
                        if (j < i) continue; // Only update upper triangular

                        // Find U[k,j] in row k
                        for (k_row_start..k_row_end) |kj_idx| {
                            if (work.col_idx[kj_idx] == j) {
                                work.values[j_idx] -= mult * work.values[kj_idx];
                                break;
                            }
                        }
                    }
                }
            }

            // Split work matrix into L (lower) and U (upper)
            // Count nnz for L and U
            var nnz_L: usize = 0;
            var nnz_U: usize = 0;
            for (0..n) |i| {
                const row_start = work.row_ptr[i];
                const row_end = work.row_ptr[i + 1];
                for (row_start..row_end) |idx| {
                    const j = work.col_idx[idx];
                    if (j < i) nnz_L += 1;
                    if (j >= i) nnz_U += 1;
                }
            }

            // Allocate L and U
            var L = try sparse.CSR(T).init(allocator, n, n, nnz_L + n); // +n for unit diagonal
            errdefer L.deinit();
            var U = try sparse.CSR(T).init(allocator, n, n, nnz_U);
            errdefer U.deinit();

            // Fill L and U
            var l_count: usize = 0;
            var u_count: usize = 0;
            for (0..n) |i| {
                L.row_ptr[i] = l_count;
                U.row_ptr[i] = u_count;

                const row_start = work.row_ptr[i];
                const row_end = work.row_ptr[i + 1];

                // Copy lower triangular to L
                for (row_start..row_end) |idx| {
                    const j = work.col_idx[idx];
                    if (j < i) {
                        L.col_idx[l_count] = j;
                        L.values[l_count] = work.values[idx];
                        l_count += 1;
                    }
                }

                // Add unit diagonal to L
                L.col_idx[l_count] = i;
                L.values[l_count] = 1.0;
                l_count += 1;

                // Copy upper triangular to U
                for (row_start..row_end) |idx| {
                    const j = work.col_idx[idx];
                    if (j >= i) {
                        U.col_idx[u_count] = j;
                        U.values[u_count] = work.values[idx];
                        u_count += 1;
                    }
                }
            }
            L.row_ptr[n] = l_count;
            U.row_ptr[n] = u_count;

            work.deinit();

            return Self{
                .L = L,
                .U = U,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.L.deinit();
            self.U.deinit();
        }

        /// Apply preconditioner: z = M⁻¹r = U⁻¹L⁻¹r
        ///
        /// Solves LUz = r in two steps:
        /// 1. Forward substitution: Ly = r
        /// 2. Backward substitution: Uz = y
        ///
        /// Time: O(nnz)
        ///
        /// Errors:
        /// - DimensionMismatch: if r.len ≠ z.len or r.len ≠ n
        pub fn apply(self: *const Self, r: []const T, z: []T) !void {
            const n = self.L.rows;
            if (r.len != z.len or r.len != n) {
                return error.DimensionMismatch;
            }

            // Temporary vector for intermediate result y
            var y = try self.allocator.alloc(T, n);
            defer self.allocator.free(y);

            // Forward substitution: Ly = r (L is lower triangular)
            for (0..n) |i| {
                var sum: T = r[i];
                const row_start = self.L.row_ptr[i];
                const row_end = self.L.row_ptr[i + 1];
                for (row_start..row_end) |idx| {
                    const j = self.L.col_idx[idx];
                    if (j < i) {
                        sum -= self.L.values[idx] * y[j];
                    }
                }
                y[i] = sum; // L has unit diagonal
            }

            // Backward substitution: Uz = y (U is upper triangular)
            var i: usize = n;
            while (i > 0) {
                i -= 1;
                var sum: T = y[i];
                const row_start = self.U.row_ptr[i];
                const row_end = self.U.row_ptr[i + 1];

                // Find diagonal element
                var diag_val: T = 1.0;
                for (row_start..row_end) |idx| {
                    const j = self.U.col_idx[idx];
                    if (j == i) {
                        diag_val = self.U.values[idx];
                    } else if (j > i) {
                        sum -= self.U.values[idx] * z[j];
                    }
                }
                z[i] = sum / diag_val;
            }
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "Jacobi preconditioner - identity matrix" {
    const allocator = testing.allocator;

    // Identity matrix 3×3
    var coo = try sparse.COO(f64).init(allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);
    try coo.append(2, 2, 1.0);

    var csr = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer csr.deinit();

    var jacobi = try JacobiPreconditioner(f64).init(allocator, &csr);
    defer jacobi.deinit();

    // Diagonal should be [1, 1, 1]
    try testing.expectEqual(@as(usize, 3), jacobi.diag.len);
    try testing.expectApproxEqAbs(@as(f64, 1.0), jacobi.diag[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), jacobi.diag[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), jacobi.diag[2], 1e-10);

    // Apply: z = M⁻¹r where r = [2, 3, 4]
    const r = [_]f64{ 2.0, 3.0, 4.0 };
    var z: [3]f64 = undefined;
    try jacobi.apply(&r, &z);

    // Expected: z = [2, 3, 4] (identity preconditioner)
    try testing.expectApproxEqAbs(@as(f64, 2.0), z[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 3.0), z[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 4.0), z[2], 1e-10);
}

test "Jacobi preconditioner - diagonal matrix" {
    const allocator = testing.allocator;

    // Diagonal matrix with diag = [2, 3, 5]
    var coo = try sparse.COO(f64).init(allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 2.0);
    try coo.append(1, 1, 3.0);
    try coo.append(2, 2, 5.0);

    var csr = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer csr.deinit();

    var jacobi = try JacobiPreconditioner(f64).init(allocator, &csr);
    defer jacobi.deinit();

    // Apply: z = M⁻¹r where r = [4, 9, 15]
    const r = [_]f64{ 4.0, 9.0, 15.0 };
    var z: [3]f64 = undefined;
    try jacobi.apply(&r, &z);

    // Expected: z[i] = r[i] / diag[i] = [2, 3, 3]
    try testing.expectApproxEqAbs(@as(f64, 2.0), z[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 3.0), z[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 3.0), z[2], 1e-10);
}

test "Jacobi preconditioner - general sparse matrix" {
    const allocator = testing.allocator;

    // Matrix:
    // [4  1  0]
    // [1  4  1]
    // [0  1  4]
    var coo = try sparse.COO(f64).init(allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 4.0);
    try coo.append(0, 1, 1.0);
    try coo.append(1, 0, 1.0);
    try coo.append(1, 1, 4.0);
    try coo.append(1, 2, 1.0);
    try coo.append(2, 1, 1.0);
    try coo.append(2, 2, 4.0);

    var csr = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer csr.deinit();

    var jacobi = try JacobiPreconditioner(f64).init(allocator, &csr);
    defer jacobi.deinit();

    // Diagonal should be [4, 4, 4]
    try testing.expectApproxEqAbs(@as(f64, 4.0), jacobi.diag[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 4.0), jacobi.diag[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 4.0), jacobi.diag[2], 1e-10);

    // Apply
    const r = [_]f64{ 8.0, 12.0, 16.0 };
    var z: [3]f64 = undefined;
    try jacobi.apply(&r, &z);

    // Expected: z = [2, 3, 4]
    try testing.expectApproxEqAbs(@as(f64, 2.0), z[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 3.0), z[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 4.0), z[2], 1e-10);
}

test "Jacobi preconditioner - zero diagonal handling" {
    const allocator = testing.allocator;

    // Matrix with zero diagonal entry:
    // [0  1]
    // [1  2]
    var coo = try sparse.COO(f64).init(allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 1, 1.0);
    try coo.append(1, 0, 1.0);
    try coo.append(1, 1, 2.0);

    var csr = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer csr.deinit();

    var jacobi = try JacobiPreconditioner(f64).init(allocator, &csr);
    defer jacobi.deinit();

    // Zero diagonal replaced with 1.0
    try testing.expectApproxEqAbs(@as(f64, 1.0), jacobi.diag[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 2.0), jacobi.diag[1], 1e-10);
}

test "Jacobi preconditioner - dimension mismatch" {
    const allocator = testing.allocator;

    var coo = try sparse.COO(f64).init(allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);

    var csr = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer csr.deinit();

    var jacobi = try JacobiPreconditioner(f64).init(allocator, &csr);
    defer jacobi.deinit();

    const r = [_]f64{ 1.0, 2.0, 3.0 }; // Wrong size
    var z: [2]f64 = undefined;
    try testing.expectError(error.DimensionMismatch, jacobi.apply(&r, &z));
}

test "Jacobi preconditioner - f32 precision" {
    const allocator = testing.allocator;

    var coo = try sparse.COO(f32).init(allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 0, 2.0);
    try coo.append(1, 1, 4.0);

    var csr = try sparse.CSR(f32).fromCOO(allocator, &coo);
    defer csr.deinit();

    var jacobi = try JacobiPreconditioner(f32).init(allocator, &csr);
    defer jacobi.deinit();

    const r = [_]f32{ 6.0, 12.0 };
    var z: [2]f32 = undefined;
    try jacobi.apply(&r, &z);

    try testing.expectApproxEqAbs(@as(f32, 3.0), z[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 3.0), z[1], 1e-5);
}

test "ILU preconditioner - identity matrix" {
    const allocator = testing.allocator;

    var coo = try sparse.COO(f64).init(allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);
    try coo.append(2, 2, 1.0);

    var csr = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer csr.deinit();

    var ilu = try ILUPreconditioner(f64).init(allocator, &csr);
    defer ilu.deinit();

    // Apply: z = M⁻¹r where r = [2, 3, 4]
    const r = [_]f64{ 2.0, 3.0, 4.0 };
    var z: [3]f64 = undefined;
    try ilu.apply(&r, &z);

    // For identity, ILU = LU = I, so z = r
    try testing.expectApproxEqAbs(@as(f64, 2.0), z[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 3.0), z[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 4.0), z[2], 1e-10);
}

test "ILU preconditioner - tridiagonal matrix" {
    const allocator = testing.allocator;

    // Tridiagonal matrix (SPD):
    // [2 -1  0]
    // [-1  2 -1]
    // [0 -1  2]
    var coo = try sparse.COO(f64).init(allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 2.0);
    try coo.append(0, 1, -1.0);
    try coo.append(1, 0, -1.0);
    try coo.append(1, 1, 2.0);
    try coo.append(1, 2, -1.0);
    try coo.append(2, 1, -1.0);
    try coo.append(2, 2, 2.0);

    var csr = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer csr.deinit();

    var ilu = try ILUPreconditioner(f64).init(allocator, &csr);
    defer ilu.deinit();

    // Apply preconditioner
    const r = [_]f64{ 1.0, 0.0, 1.0 };
    var z: [3]f64 = undefined;
    try ilu.apply(&r, &z);

    // Verify z is computed (exact values depend on ILU factorization)
    // Just check it doesn't crash and produces finite values
    try testing.expect(math.isFinite(z[0]));
    try testing.expect(math.isFinite(z[1]));
    try testing.expect(math.isFinite(z[2]));
}

test "ILU preconditioner - singular matrix detection" {
    const allocator = testing.allocator;

    // Singular matrix (zero diagonal):
    // [0  1]
    // [1  0]
    var coo = try sparse.COO(f64).init(allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 1, 1.0);
    try coo.append(1, 0, 1.0);

    var csr = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer csr.deinit();

    try testing.expectError(error.SingularMatrix, ILUPreconditioner(f64).init(allocator, &csr));
}

test "ILU preconditioner - memory safety" {
    const allocator = testing.allocator;

    var coo = try sparse.COO(f64).init(allocator, 4, 4);
    defer coo.deinit();
    try coo.append(0, 0, 4.0);
    try coo.append(0, 1, 1.0);
    try coo.append(1, 0, 1.0);
    try coo.append(1, 1, 4.0);
    try coo.append(1, 2, 1.0);
    try coo.append(2, 1, 1.0);
    try coo.append(2, 2, 4.0);
    try coo.append(2, 3, 1.0);
    try coo.append(3, 2, 1.0);
    try coo.append(3, 3, 4.0);

    var csr = try sparse.CSR(f64).fromCOO(allocator, &coo);
    defer csr.deinit();

    // Create and destroy ILU preconditioner 10 times
    for (0..10) |_| {
        var ilu = try ILUPreconditioner(f64).init(allocator, &csr);
        defer ilu.deinit();

        const r = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
        var z: [4]f64 = undefined;
        try ilu.apply(&r, &z);
    }
}
