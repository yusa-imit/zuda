//! LU Decomposition with Partial Pivoting
//!
//! Factorizes a matrix A into LU form: A = PLU where:
//! - P is a permutation matrix
//! - L is a lower triangular matrix with unit diagonal
//! - U is an upper triangular matrix
//!
//! Uses partial pivoting for improved numerical stability.
//!
//! ## Time Complexity
//! - Decomposition: O(n³) where n = min(rows, cols)
//! - Reconstruction: O(n²)
//!
//! ## Numeric Precision
//! - Tested for f32 and f64
//! - Numerical stability via partial pivoting
//! - Singularity detection via zero diagonal in U
//!
//! ## Error Handling
//! - error.NonSquareMatrix for non-square input
//! - error.SingularMatrix when pivoting fails (zero diagonal in U)

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const root = @import("../root.zig");
const NDArray = root.ndarray.NDArray;

/// Result of LU decomposition: P, L, U matrices and pivot array
pub fn LUResult(comptime T: type) type {
    return struct {
        /// Permutation matrix (nxn) as NDArray(T, 2)
        /// When applied: P @ A = L @ U
        P: NDArray(T, 2),

        /// Lower triangular matrix (nxn) with unit diagonal
        /// All elements below diagonal contain L values
        /// Diagonal is implicitly 1.0
        L: NDArray(T, 2),

        /// Upper triangular matrix (nxn)
        /// All elements on and above diagonal contain U values
        U: NDArray(T, 2),

        /// Pivot array: indices of row swaps (length = min(rows, cols))
        /// pivots[i] = j means row i was swapped with row j
        pivots: []usize,

        /// Allocator used for all allocations
        allocator: Allocator,

        /// Free all allocated memory
        ///
        /// Time: O(1) deallocation
        /// Space: O(1)
        pub fn deinit(self: *@This()) void {
            self.P.deinit();
            self.L.deinit();
            self.U.deinit();
            self.allocator.free(self.pivots);
        }
    };
}

/// Compute LU decomposition with partial pivoting
///
/// Factorizes matrix A as A = PLU where:
/// - P is a permutation matrix
/// - L is lower triangular (unit diagonal)
/// - U is upper triangular
///
/// Parameters:
/// - T: Numeric type (f32, f64)
/// - allocator: Memory allocator for result matrices
/// - A: Input matrix (must be square, ndim=2)
///
/// Returns: LUResult containing P, L, U, and pivot information
///
/// Errors:
/// - error.NonSquareMatrix if A is not square
/// - error.SingularMatrix if matrix is singular (zero pivot encountered)
/// - error.OutOfMemory if allocation fails
///
/// Time: O(n³) where n = rows = cols
/// Space: O(n²) for result matrices
///
/// Example:
/// ```zig
/// var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{2, 2}, &[_]f64{1, 2, 3, 4}, .row_major);
/// defer A.deinit();
/// var result = try lu(f64, allocator, A);
/// defer result.deinit();
/// // Verify reconstruction: ||A - P@L@U|| < epsilon
/// ```
pub fn lu(comptime T: type, allocator: Allocator, A: NDArray(T, 2)) (NDArray(T, 2).Error || error{
    NonSquareMatrix,
    SingularMatrix,
} || std.mem.Allocator.Error)!LUResult(T) {
    // Validate square matrix
    if (A.shape[0] != A.shape[1]) {
        return error.NonSquareMatrix;
    }

    const n = A.shape[0];

    // Compute singularity tolerance: sqrt(machine epsilon) for the type
    const epsilon = switch (T) {
        f32 => 1.19209e-7,  // sqrt(epsilon_f32 ≈ 1.19209e-7)
        f64 => 1.49012e-8,  // sqrt(epsilon_f64 ≈ 2.22045e-16)
        else => 1e-15,      // fallback for other types
    };

    // Allocate result matrices (copies of A for in-place computation)
    var L = try NDArray(T, 2).zeros(allocator, &[_]usize{ n, n }, .row_major);
    errdefer L.deinit();

    var U = try NDArray(T, 2).zeros(allocator, &[_]usize{ n, n }, .row_major);
    errdefer U.deinit();

    var P = try NDArray(T, 2).zeros(allocator, &[_]usize{ n, n }, .row_major);
    errdefer P.deinit();

    // Allocate pivot array
    var pivots = try allocator.alloc(usize, n);
    errdefer allocator.free(pivots);

    // Initialize P as identity and pivots as identity permutation
    for (0..n) |i| {
        P.data[i * n + i] = 1;
        pivots[i] = i;
    }

    // Copy A into U for in-place factorization, respecting input layout
    for (0..n) |i| {
        for (0..n) |j| {
            // Use row-major indexing for U (output)
            U.data[i * n + j] = try A.get(&[_]isize{ @intCast(i), @intCast(j) });
        }
    }

    // LU factorization with partial pivoting
    for (0..n) |col| {
        // Find pivot (maximum absolute value in column)
        var pivot_row = col;
        var max_val = @abs(U.data[col * n + col]);

        for (col + 1..n) |row| {
            const abs_val = @abs(U.data[row * n + col]);
            if (abs_val > max_val) {
                max_val = abs_val;
                pivot_row = row;
            }
        }

        // Check for singular matrix (zero pivot)
        if (max_val < epsilon) {
            return error.SingularMatrix;
        }

        // Swap rows in U if needed
        if (pivot_row != col) {
            // Swap U rows (row-major storage)
            for (0..n) |j| {
                const temp = U.data[col * n + j];
                U.data[col * n + j] = U.data[pivot_row * n + j];
                U.data[pivot_row * n + j] = temp;
            }

            // Swap rows in L (only the part computed so far)
            for (0..col) |j| {
                const temp = L.data[col * n + j];
                L.data[col * n + j] = L.data[pivot_row * n + j];
                L.data[pivot_row * n + j] = temp;
            }

            // Update permutation
            pivots[col] = pivot_row;
            // Also swap rows in P for the result (row-major storage)
            for (0..n) |j| {
                const temp = P.data[col * n + j];
                P.data[col * n + j] = P.data[pivot_row * n + j];
                P.data[pivot_row * n + j] = temp;
            }
        }

        // Set diagonal of L to 1
        L.data[col * n + col] = 1;

        // Compute L column and U row
        for (col + 1..n) |row| {
            const factor = U.data[row * n + col] / U.data[col * n + col];
            L.data[row * n + col] = factor;

            // Eliminate below pivot
            for (col..n) |j| {
                U.data[row * n + j] -= factor * U.data[col * n + j];
            }
        }
    }

    return LUResult(T){
        .P = P,
        .L = L,
        .U = U,
        .pivots = pivots,
        .allocator = allocator,
    };
}

/// Verify that matrices satisfy PLU decomposition property: A = PLU
///
/// Reconstructs A from P, L, U and checks ||A - PLU|| < tolerance.
///
/// Parameters:
/// - T: Numeric type
/// - original: Original matrix A
/// - result: LU decomposition result
/// - tolerance: Epsilon for floating-point comparison
///
/// Time: O(n³) matrix multiplication + O(n) comparison
/// Space: O(n²) temporary matrices
fn verifyDecomposition(comptime T: type, allocator: Allocator, original: NDArray(T, 2), result: LUResult(T), tolerance: T) !void {
    const n = original.shape[0];

    // Allocate temporary matrices for reconstruction
    var temp1 = try NDArray(T, 2).zeros(allocator, &[_]usize{ n, n }, .row_major);
    defer temp1.deinit();

    var temp2 = try NDArray(T, 2).zeros(allocator, &[_]usize{ n, n }, .row_major);
    defer temp2.deinit();

    // Compute L @ U
    for (0..n) |i| {
        for (0..n) |j| {
            var sum: T = 0;
            for (0..n) |k| {
                const L_val = if (k < i + 1) result.L.data[i * n + k] else if (k == i) 1 else 0;
                const U_val = if (k <= j) result.U.data[k * n + j] else 0;
                sum += L_val * U_val;
            }
            temp1.data[i * n + j] = sum;
        }
    }

    // Compute P @ (L @ U)
    for (0..n) |i| {
        for (0..n) |j| {
            var sum: T = 0;
            for (0..n) |k| {
                sum += result.P.data[i * n + k] * temp1.data[k * n + j];
            }
            temp2.data[i * n + j] = sum;
        }
    }

    // Compare: ||A - P@L@U|| < tolerance
    for (0..n * n) |idx| {
        const diff = @abs(original.data[idx] - temp2.data[idx]);
        try testing.expect(diff < tolerance);
    }
}

// ============================================================================
// Tests
// ============================================================================

test "lu: 2x2 identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{2, 2}, &[_]f64{
        1, 0,
        0, 1,
    }, .row_major);
    defer A.deinit();

    var result = try lu(f64, allocator, A);
    defer result.deinit();

    // P should be identity
    try testing.expectApproxEqAbs(1.0, result.P.data[0], 1e-10);
    try testing.expectApproxEqAbs(0.0, result.P.data[1], 1e-10);
    try testing.expectApproxEqAbs(0.0, result.P.data[2], 1e-10);
    try testing.expectApproxEqAbs(1.0, result.P.data[3], 1e-10);

    // L should be identity
    try testing.expectApproxEqAbs(1.0, result.L.data[0], 1e-10);
    try testing.expectApproxEqAbs(0.0, result.L.data[1], 1e-10);
    try testing.expectApproxEqAbs(0.0, result.L.data[2], 1e-10);
    try testing.expectApproxEqAbs(1.0, result.L.data[3], 1e-10);

    // U should be identity
    try testing.expectApproxEqAbs(1.0, result.U.data[0], 1e-10);
    try testing.expectApproxEqAbs(0.0, result.U.data[1], 1e-10);
    try testing.expectApproxEqAbs(0.0, result.U.data[2], 1e-10);
    try testing.expectApproxEqAbs(1.0, result.U.data[3], 1e-10);

    try verifyDecomposition(f64, allocator, A, result, 1e-10);
}

test "lu: 3x3 identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 3}, &[_]f64{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    }, .row_major);
    defer A.deinit();

    var result = try lu(f64, allocator, A);
    defer result.deinit();

    // Verify decomposition
    try verifyDecomposition(f64, allocator, A, result, 1e-10);
}

test "lu: 2x2 non-identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{2, 2}, &[_]f64{
        1, 2,
        3, 4,
    }, .row_major);
    defer A.deinit();

    var result = try lu(f64, allocator, A);
    defer result.deinit();

    // Verify reconstruction: A = P@L@U
    try verifyDecomposition(f64, allocator, A, result, 1e-10);
}

test "lu: 3x3 non-identity matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 3}, &[_]f64{
        2, 3, 1,
        6, 13, 5,
        2, 19, 10,
    }, .row_major);
    defer A.deinit();

    var result = try lu(f64, allocator, A);
    defer result.deinit();

    // Verify reconstruction
    try verifyDecomposition(f64, allocator, A, result, 1e-10);
}

test "lu: permutation correctness — requires pivoting" {
    const allocator = testing.allocator;

    // Matrix where pivoting is essential
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 3}, &[_]f64{
        0, 1, 2,
        1, 2, 3,
        2, 3, 4,
    }, .row_major);
    defer A.deinit();

    var result = try lu(f64, allocator, A);
    defer result.deinit();

    // Pivot array should be valid permutation (all entries < n, all unique)
    var seen = try allocator.alloc(bool, result.pivots.len);
    defer allocator.free(seen);
    @memset(seen, false);

    for (result.pivots) |pivot| {
        try testing.expect(pivot < result.pivots.len);
        try testing.expect(!seen[pivot]);
        seen[pivot] = true;
    }

    // P should be a valid permutation matrix
    // Each row and column should sum to exactly 1
    for (0..3) |i| {
        var row_sum: f64 = 0;
        var col_sum: f64 = 0;
        for (0..3) |j| {
            row_sum += result.P.data[i * 3 + j];
            col_sum += result.P.data[j * 3 + i];
        }
        try testing.expectApproxEqAbs(1.0, row_sum, 1e-10);
        try testing.expectApproxEqAbs(1.0, col_sum, 1e-10);
    }

    // Verify reconstruction
    try verifyDecomposition(f64, allocator, A, result, 1e-10);
}

test "lu: L properties — lower triangular with unit diagonal" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 3}, &[_]f64{
        4, 3, 2,
        1, 6, 5,
        7, 8, 9,
    }, .row_major);
    defer A.deinit();

    var result = try lu(f64, allocator, A);
    defer result.deinit();

    // Check L is lower triangular with unit diagonal
    for (0..3) |i| {
        // Diagonal must be 1
        try testing.expectApproxEqAbs(1.0, result.L.data[i * 3 + i], 1e-10);

        // Upper triangle must be 0
        for (i + 1..3) |j| {
            try testing.expectApproxEqAbs(0.0, result.L.data[i * 3 + j], 1e-10);
        }
    }
}

test "lu: U properties — upper triangular" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 3}, &[_]f64{
        4, 3, 2,
        1, 6, 5,
        7, 8, 9,
    }, .row_major);
    defer A.deinit();

    var result = try lu(f64, allocator, A);
    defer result.deinit();

    // Check U is upper triangular
    for (1..3) |i| {
        for (0..i) |j| {
            try testing.expectApproxEqAbs(0.0, result.U.data[i * 3 + j], 1e-10);
        }
    }
}

test "lu: P properties — valid permutation matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 3}, &[_]f64{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    }, .row_major);
    defer A.deinit();

    var result = try lu(f64, allocator, A);
    defer result.deinit();

    const n = 3;

    // Each row sum should be 1
    for (0..n) |i| {
        var row_sum: f64 = 0;
        for (0..n) |j| {
            row_sum += result.P.data[i * n + j];
        }
        try testing.expectApproxEqAbs(1.0, row_sum, 1e-10);
    }

    // Each column sum should be 1
    for (0..n) |j| {
        var col_sum: f64 = 0;
        for (0..n) |i| {
            col_sum += result.P.data[i * n + j];
        }
        try testing.expectApproxEqAbs(1.0, col_sum, 1e-10);
    }

    // Each element should be 0 or 1
    for (0..n * n) |idx| {
        const val = result.P.data[idx];
        try testing.expect(val == 0 or val == 1);
    }
}

test "lu: singular matrix detection — all zeros" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{2, 2}, .row_major);
    defer A.deinit();

    const err = lu(f64, allocator, A);
    try testing.expectError(error.SingularMatrix, err);
}

test "lu: singular matrix detection — rank deficient" {
    const allocator = testing.allocator;

    // Rank 1 matrix: all rows are multiples
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 3}, &[_]f64{
        1, 2, 3,
        2, 4, 6,
        3, 6, 9,
    }, .row_major);
    defer A.deinit();

    const err = lu(f64, allocator, A);
    try testing.expectError(error.SingularMatrix, err);
}

test "lu: non-square matrix error" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{2, 3}, &[_]f64{
        1, 2, 3,
        4, 5, 6,
    }, .row_major);
    defer A.deinit();

    const err = lu(f64, allocator, A);
    try testing.expectError(error.NonSquareMatrix, err);
}

test "lu: f32 precision" {
    const allocator = testing.allocator;

    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{2, 2}, &[_]f32{
        1.5, 2.5,
        3.5, 4.5,
    }, .row_major);
    defer A.deinit();

    var result = try lu(f32, allocator, A);
    defer result.deinit();

    try verifyDecomposition(f32, allocator, A, result, 1e-5);
}

test "lu: f64 precision" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 3}, &[_]f64{
        0.12, 0.34, 0.56,
        0.78, 0.90, 0.12,
        0.34, 0.56, 0.78,
    }, .row_major);
    defer A.deinit();

    var result = try lu(f64, allocator, A);
    defer result.deinit();

    try verifyDecomposition(f64, allocator, A, result, 1e-10);
}

test "lu: ill-conditioned matrix" {
    const allocator = testing.allocator;

    // Hilbert matrix: notoriously ill-conditioned
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 3}, &[_]f64{
        1.0, 0.5, 1.0/3.0,
        0.5, 1.0/3.0, 0.25,
        1.0/3.0, 0.25, 0.2,
    }, .row_major);
    defer A.deinit();

    var result = try lu(f64, allocator, A);
    defer result.deinit();

    // Even for ill-conditioned matrix, PLU should be valid (with larger tolerance)
    try verifyDecomposition(f64, allocator, A, result, 1e-8);
}

test "lu: reconstruction accuracy 4x4 random" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{4, 4}, &[_]f64{
        1.2, 2.3, 3.4, 4.5,
        5.6, 6.7, 7.8, 8.9,
        9.0, 8.1, 7.2, 6.3,
        5.4, 4.5, 3.6, 2.7,
    }, .row_major);
    defer A.deinit();

    var result = try lu(f64, allocator, A);
    defer result.deinit();

    try verifyDecomposition(f64, allocator, A, result, 1e-10);
}

test "lu: memory cleanup — no leaks" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 3}, &[_]f64{
        2, 1, 1,
        4, 3, 3,
        8, 7, 9,
    }, .row_major);
    defer A.deinit();

    var result = try lu(f64, allocator, A);
    result.deinit();

    // Testing allocator detects any leaks
}

test "lu: 5x5 matrix decomposition" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{5, 5}, &[_]f64{
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 26,  // Last diagonal element slightly off to avoid singularity
    }, .row_major);
    defer A.deinit();

    var result = try lu(f64, allocator, A);
    defer result.deinit();

    try verifyDecomposition(f64, allocator, A, result, 1e-9);
}

test "lu: diagonal matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 3}, &[_]f64{
        2, 0, 0,
        0, 3, 0,
        0, 0, 5,
    }, .row_major);
    defer A.deinit();

    var result = try lu(f64, allocator, A);
    defer result.deinit();

    // For diagonal matrix, P should be identity
    try testing.expectApproxEqAbs(1.0, result.P.data[0], 1e-10);
    try testing.expectApproxEqAbs(1.0, result.P.data[4], 1e-10);
    try testing.expectApproxEqAbs(1.0, result.P.data[8], 1e-10);

    // L should be identity
    try testing.expectApproxEqAbs(1.0, result.L.data[0], 1e-10);
    try testing.expectApproxEqAbs(1.0, result.L.data[4], 1e-10);
    try testing.expectApproxEqAbs(1.0, result.L.data[8], 1e-10);

    // U should match A
    for (0..9) |i| {
        try testing.expectApproxEqAbs(A.data[i], result.U.data[i], 1e-10);
    }

    try verifyDecomposition(f64, allocator, A, result, 1e-10);
}

test "lu: triangular matrix (already upper triangular)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 3}, &[_]f64{
        1, 2, 3,
        0, 4, 5,
        0, 0, 6,
    }, .row_major);
    defer A.deinit();

    var result = try lu(f64, allocator, A);
    defer result.deinit();

    try verifyDecomposition(f64, allocator, A, result, 1e-10);
}

test "lu: negative values" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 3}, &[_]f64{
        -1, 2, -3,
        4, -5, 6,
        -7, 8, -9,
    }, .row_major);
    defer A.deinit();

    var result = try lu(f64, allocator, A);
    defer result.deinit();

    try verifyDecomposition(f64, allocator, A, result, 1e-9);
}

test "lu: column-major layout" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{2, 2}, &[_]f64{
        1, 3,
        2, 4,
    }, .column_major);
    defer A.deinit();

    var result = try lu(f64, allocator, A);
    defer result.deinit();

    try verifyDecomposition(f64, allocator, A, result, 1e-10);
}

test "lu: small pivots handled correctly" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 3}, &[_]f64{
        0.001, 2, 3,
        1, 5, 6,
        2, 8, 9,
    }, .row_major);
    defer A.deinit();

    var result = try lu(f64, allocator, A);
    defer result.deinit();

    // Should perform pivoting to avoid tiny pivot
    try verifyDecomposition(f64, allocator, A, result, 1e-10);
}

test "lu: 2x2 simple integer matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{2, 2}, &[_]f64{
        2, 1,
        1, 3,
    }, .row_major);
    defer A.deinit();

    var result = try lu(f64, allocator, A);
    defer result.deinit();

    // This is a known matrix: P should swap rows (pivot on larger value)
    try verifyDecomposition(f64, allocator, A, result, 1e-10);
}
