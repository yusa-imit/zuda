//! Matrix Properties — rank, condition number
//!
//! Computes numerical properties of matrices using Singular Value Decomposition (SVD).
//!
//! ## Functions
//! - **rank(A)**: Numerical rank counting singular values above tolerance threshold
//! - **cond(A)**: Condition number κ(A) = σ_max / σ_min (ratio of largest to smallest singular value)
//!
//! ## Numerical Stability
//! - Rank: Uses relative tolerance = max(m,n) × σ_max × machine_epsilon
//! - Condition number: Returns +∞ for singular matrices (σ_min = 0)
//! - All computations use floating-point type's native precision

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const root = @import("../root.zig");
const NDArray = root.ndarray.NDArray;
const decomp = @import("decompositions.zig");

// ============================================================================
// rank(A) — Matrix Numerical Rank
// ============================================================================

/// Compute the numerical rank of a matrix via SVD
///
/// Counts the number of singular values greater than a tolerance threshold:
/// - Tolerance = max(m,n) × σ_max × machine_epsilon
///
/// This represents the effective rank considering finite precision arithmetic.
///
/// Parameters:
/// - T: Numeric type (f32, f64)
/// - A: Input matrix (m×n)
/// - allocator: Memory allocator for SVD computation
///
/// Returns: Number of singular values above tolerance (0 ≤ rank ≤ min(m,n))
///
/// Errors:
/// - error.OutOfMemory if SVD allocation fails
///
/// Time: O(mn²) for SVD computation
/// Space: O(mn) for SVD matrices
///
/// Example:
/// ```zig
/// var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 3}, &[_]f64{
///     1, 0, 0,
///     0, 1, 0,
///     0, 0, 1,
/// }, .row_major);
/// defer A.deinit();
///
/// const r = try rank(f64, A, allocator); // r == 3
/// ```
pub fn rank(
    comptime T: type,
    A: NDArray(T, 2),
    allocator: Allocator,
) (NDArray(T, 2).Error || NDArray(T, 1).Error || std.mem.Allocator.Error)!usize {
    const m = A.shape[0];
    const n = A.shape[1];

    // Compute SVD
    var svd_result = try decomp.svd(T, A, allocator);
    defer svd_result.deinit();

    // Get machine epsilon for type T
    const eps = switch (T) {
        f32 => std.math.floatEps(f32),
        f64 => std.math.floatEps(f64),
        else => @compileError("rank() only supports f32 and f64"),
    };

    // Get maximum singular value
    const max_m_n = @max(m, n);
    var sigma_max: T = 0;
    for (0..svd_result.S.shape[0]) |i| {
        const s = svd_result.S.data[i];
        if (s > sigma_max) {
            sigma_max = s;
        }
    }

    // Tolerance: max(m,n) × σ_max × machine_epsilon
    const tolerance = @as(T, @floatFromInt(max_m_n)) * sigma_max * eps;

    // Count singular values above tolerance
    var r: usize = 0;
    for (0..svd_result.S.shape[0]) |i| {
        const s = svd_result.S.data[i];
        if (s > tolerance) {
            r += 1;
        }
    }

    return r;
}

// ============================================================================
// cond(A) — Condition Number
// ============================================================================

/// Compute the condition number of a matrix via SVD
///
/// κ(A) = σ_max / σ_min
///
/// Measures the sensitivity of the solution to perturbations in the input.
/// - κ ≈ 1: Well-conditioned (small errors don't amplify)
/// - κ ≫ 1: Ill-conditioned (errors amplify significantly)
/// - κ = +∞: Singular matrix (σ_min = 0)
///
/// Parameters:
/// - T: Numeric type (f32, f64)
/// - A: Input matrix (m×n)
/// - allocator: Memory allocator for SVD computation
///
/// Returns: Condition number κ(A) = σ_max / σ_min (or +∞ if singular)
///
/// Errors:
/// - error.OutOfMemory if SVD allocation fails
///
/// Time: O(mn²) for SVD computation
/// Space: O(mn) for SVD matrices
///
/// Example:
/// ```zig
/// var I = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 3}, &[_]f64{
///     1, 0, 0,
///     0, 1, 0,
///     0, 0, 1,
/// }, .row_major);
/// defer I.deinit();
///
/// const c = try cond(f64, I, allocator); // c == 1.0
/// ```
pub fn cond(
    comptime T: type,
    A: NDArray(T, 2),
    allocator: Allocator,
) (NDArray(T, 2).Error || NDArray(T, 1).Error || std.mem.Allocator.Error)!T {
    const m = A.shape[0];
    const n = A.shape[1];

    // Compute SVD
    var svd_result = try decomp.svd(T, A, allocator);
    defer svd_result.deinit();

    // Get machine epsilon for type T
    const eps = switch (T) {
        f32 => std.math.floatEps(f32),
        f64 => std.math.floatEps(f64),
        else => @compileError("cond() only supports f32 and f64"),
    };

    // Get maximum and minimum singular values
    var sigma_max: T = 0;
    var sigma_min: T = std.math.inf(T);
    for (0..svd_result.S.shape[0]) |i| {
        const s = svd_result.S.data[i];
        if (s > sigma_max) {
            sigma_max = s;
        }
        if (s < sigma_min) {
            sigma_min = s;
        }
    }

    // Tolerance for singularity check: max(m,n) × σ_max × machine_epsilon
    const max_m_n = @max(m, n);
    const tolerance = @as(T, @floatFromInt(max_m_n)) * sigma_max * eps;

    // Handle singular matrix (σ_min <= tolerance)
    if (sigma_min <= tolerance) {
        return std.math.inf(T);
    }

    return sigma_max / sigma_min;
}

// ============================================================================
// Tests
// ============================================================================

test "rank: identity matrix 2x2" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1, 0,
        0, 1,
    }, .row_major);
    defer A.deinit();

    const r = try rank(f64, A, allocator);
    try testing.expectEqual(@as(usize, 2), r);
}

test "rank: identity matrix 3x3" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    }, .row_major);
    defer A.deinit();

    const r = try rank(f64, A, allocator);
    try testing.expectEqual(@as(usize, 3), r);
}

test "rank: full rank square 3x3" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 2, 3,
        4, 5, 6,
        7, 8, 10,
    }, .row_major);
    defer A.deinit();

    const r = try rank(f64, A, allocator);
    try testing.expectEqual(@as(usize, 3), r);
}

test "rank: full rank tall 4x2" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 2 }, &[_]f64{
        1, 0,
        0, 1,
        1, 1,
        2, 3,
    }, .row_major);
    defer A.deinit();

    const r = try rank(f64, A, allocator);
    try testing.expectEqual(@as(usize, 2), r);
}

test "rank: full rank wide 2x4" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 4 }, &[_]f64{
        1, 0, 1, 2,
        0, 1, 1, 3,
    }, .row_major);
    defer A.deinit();

    const r = try rank(f64, A, allocator);
    try testing.expectEqual(@as(usize, 2), r);
}

test "rank: rank-1 matrix (outer product)" {
    const allocator = testing.allocator;

    // Outer product of [1, 2, 3] and [1, 1]
    // All rows are scalar multiples of [1, 1]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f64{
        1, 1,
        2, 2,
        3, 3,
    }, .row_major);
    defer A.deinit();

    const r = try rank(f64, A, allocator);
    try testing.expectEqual(@as(usize, 1), r);
}

test "rank: rank-2 matrix 3x3" {
    const allocator = testing.allocator;

    // Third row is sum of first two
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 0, 0,
        0, 1, 0,
        1, 1, 0,
    }, .row_major);
    defer A.deinit();

    const r = try rank(f64, A, allocator);
    try testing.expectEqual(@as(usize, 2), r);
}

test "rank: zero matrix 3x3" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer A.deinit();

    const r = try rank(f64, A, allocator);
    try testing.expectEqual(@as(usize, 0), r);
}

test "rank: zero row 4x3" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 3 }, &[_]f64{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        0, 0, 0,
    }, .row_major);
    defer A.deinit();

    const r = try rank(f64, A, allocator);
    try testing.expectEqual(@as(usize, 3), r);
}

test "rank: zero column 3x4" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 4 }, &[_]f64{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    }, .row_major);
    defer A.deinit();

    const r = try rank(f64, A, allocator);
    try testing.expectEqual(@as(usize, 3), r);
}

test "rank: 1x1 matrix non-zero" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 1 }, &[_]f64{5}, .row_major);
    defer A.deinit();

    const r = try rank(f64, A, allocator);
    try testing.expectEqual(@as(usize, 1), r);
}

test "rank: 1x1 zero matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 1 }, &[_]f64{0}, .row_major);
    defer A.deinit();

    const r = try rank(f64, A, allocator);
    try testing.expectEqual(@as(usize, 0), r);
}

test "rank: diagonal 4x4 with one zero" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 4 }, &[_]f64{
        2, 0, 0, 0,
        0, 3, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 5,
    }, .row_major);
    defer A.deinit();

    const r = try rank(f64, A, allocator);
    try testing.expectEqual(@as(usize, 3), r);
}

test "rank: f32 precision" {
    const allocator = testing.allocator;

    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{
        1, 0,
        0, 1,
    }, .row_major);
    defer A.deinit();

    const r = try rank(f32, A, allocator);
    try testing.expectEqual(@as(usize, 2), r);
}

test "rank: f64 precision" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    }, .row_major);
    defer A.deinit();

    const r = try rank(f64, A, allocator);
    try testing.expectEqual(@as(usize, 3), r);
}

test "rank: memory safety with allocator" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 5, 5 }, &[_]f64{
        1, 0, 0, 0, 0,
        0, 2, 0, 0, 0,
        0, 0, 3, 0, 0,
        0, 0, 0, 4, 0,
        0, 0, 0, 0, 5,
    }, .row_major);
    defer A.deinit();

    const r = try rank(f64, A, allocator);
    try testing.expectEqual(@as(usize, 5), r);
}

test "cond: identity matrix 3x3" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    }, .row_major);
    defer A.deinit();

    const c = try cond(f64, A, allocator);
    try testing.expectApproxEqAbs(@as(f64, 1.0), c, 1e-10);
}

test "cond: diagonal matrix all ones 3x3" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    }, .row_major);
    defer A.deinit();

    const c = try cond(f64, A, allocator);
    try testing.expectApproxEqAbs(@as(f64, 1.0), c, 1e-10);
}

test "cond: diagonal 1 to 10" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 0, 0,
        0, 5, 0,
        0, 0, 10,
    }, .row_major);
    defer A.deinit();

    const c = try cond(f64, A, allocator);
    try testing.expectApproxEqAbs(@as(f64, 10.0), c, 1e-10);
}

test "cond: well-conditioned 3x3" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        2, 1, 0,
        1, 2, 1,
        0, 1, 2,
    }, .row_major);
    defer A.deinit();

    const c = try cond(f64, A, allocator);
    // This is a well-conditioned tridiagonal matrix
    try testing.expect(c < 10);
}

test "cond: Hilbert 3x3" {
    const allocator = testing.allocator;

    // Hilbert matrix: H[i,j] = 1/(i+j+1)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1.0, 1.0 / 2, 1.0 / 3,
        1.0 / 2, 1.0 / 3, 1.0 / 4,
        1.0 / 3, 1.0 / 4, 1.0 / 5,
    }, .row_major);
    defer A.deinit();

    const c = try cond(f64, A, allocator);
    // Hilbert 3x3 has condition number around 524
    try testing.expect(c > 500);
}

test "cond: Hilbert 4x4" {
    const allocator = testing.allocator;

    // Hilbert matrix 4x4: H[i,j] = 1/(i+j+1)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 4 }, &[_]f64{
        1.0, 1.0 / 2, 1.0 / 3, 1.0 / 4,
        1.0 / 2, 1.0 / 3, 1.0 / 4, 1.0 / 5,
        1.0 / 3, 1.0 / 4, 1.0 / 5, 1.0 / 6,
        1.0 / 4, 1.0 / 5, 1.0 / 6, 1.0 / 7,
    }, .row_major);
    defer A.deinit();

    const c = try cond(f64, A, allocator);
    // Hilbert 4x4 has condition number around 15000
    try testing.expect(c > 10000);
}

test "cond: nearly singular matrix" {
    const allocator = testing.allocator;

    // Matrix with singular values 1, 1e-10
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        1, 0,
        0, 1e-10,
    }, .row_major);
    defer A.deinit();

    const c = try cond(f64, A, allocator);
    // Expected: 1 / 1e-10 = 1e10
    try testing.expect(c > 1e9);
}

test "cond: diagonal wide range 1e-6 to 1" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1e-6, 0, 0,
        0, 1e-3, 0,
        0, 0, 1,
    }, .row_major);
    defer A.deinit();

    const c = try cond(f64, A, allocator);
    // Expected: 1 / 1e-6 = 1e6
    try testing.expectApproxEqAbs(@as(f64, 1e6), c, 1e4);
}

test "cond: stretched matrix high condition" {
    const allocator = testing.allocator;

    // Matrix with highly different singular values
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{
        100, 0,
        0, 1,
    }, .row_major);
    defer A.deinit();

    const c = try cond(f64, A, allocator);
    try testing.expectApproxEqAbs(@as(f64, 100.0), c, 1e-10);
}

test "cond: zero matrix returns infinity" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer A.deinit();

    const c = try cond(f64, A, allocator);
    try testing.expect(std.math.isInf(c));
    try testing.expect(c > 0); // positive infinity
}

test "cond: rank deficient (dependent rows) returns infinity" {
    const allocator = testing.allocator;

    // Third row is sum of first two
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 0, 0,
        0, 1, 0,
        1, 1, 0,
    }, .row_major);
    defer A.deinit();

    const c = try cond(f64, A, allocator);
    try testing.expect(std.math.isInf(c));
}

test "cond: zero row returns infinity" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        1, 0, 0,
        0, 1, 0,
        0, 0, 0,
    }, .row_major);
    defer A.deinit();

    const c = try cond(f64, A, allocator);
    try testing.expect(std.math.isInf(c));
}

test "cond: rectangular tall 4x2 full rank" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 2 }, &[_]f64{
        1, 0,
        0, 1,
        1, 1,
        2, 3,
    }, .row_major);
    defer A.deinit();

    const c = try cond(f64, A, allocator);
    // Condition number exists for rectangular matrices
    try testing.expect(c >= 1);
    try testing.expect(!std.math.isInf(c));
}

test "cond: rectangular wide 2x4 full rank" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 4 }, &[_]f64{
        1, 0, 1, 2,
        0, 1, 1, 3,
    }, .row_major);
    defer A.deinit();

    const c = try cond(f64, A, allocator);
    // Condition number exists for rectangular matrices
    try testing.expect(c >= 1);
    try testing.expect(!std.math.isInf(c));
}

test "cond: f64 precision with known cond" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{
        10, 0, 0,
        0, 5, 0,
        0, 0, 1,
    }, .row_major);
    defer A.deinit();

    const c = try cond(f64, A, allocator);
    // Expected: 10 / 1 = 10
    try testing.expectApproxEqAbs(@as(f64, 10.0), c, 1e-10);
}

test "cond: memory safety with allocator" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 5, 5 }, &[_]f64{
        1, 0, 0, 0, 0,
        0, 2, 0, 0, 0,
        0, 0, 3, 0, 0,
        0, 0, 0, 4, 0,
        0, 0, 0, 0, 5,
    }, .row_major);
    defer A.deinit();

    const c = try cond(f64, A, allocator);
    // Expected: 5 / 1 = 5
    try testing.expectApproxEqAbs(@as(f64, 5.0), c, 1e-10);
}
