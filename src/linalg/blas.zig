//! BLAS Level 1 — Vector-Vector Operations
//!
//! Provides fundamental linear algebra operations on 1D vectors (NDArray with rank 1).
//! These are optimized implementations of Basic Linear Algebra Subprograms (BLAS) Level 1.
//!
//! ## Operations
//! - **dot**: Inner product (dot product) of two vectors
//! - **axpy**: Vector update: y = αx + y (in-place)
//! - **nrm2**: Euclidean norm (L2 norm)
//! - **asum**: Sum of absolute values
//! - **scal**: Vector scaling: x = αx (in-place)
//!
//! ## Time Complexity
//! All operations: O(n) where n is the vector length
//!
//! ## Numeric Precision
//! - Tested for f32 and f64
//! - Handles NaN, Infinity correctly
//!
//! ## Error Handling
//! - Dimension mismatch errors for binary operations
//! - Validates vector lengths match for axpy

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const root = @import("../root.zig");
const NDArray = root.ndarray.NDArray;

/// Compute inner product (dot product) of two vectors
///
/// Parameters:
/// - x: First vector (1D NDArray)
/// - y: Second vector (1D NDArray)
///
/// Returns: Scalar result of x·y
///
/// Errors:
/// - error.DimensionMismatch if x and y have different lengths
///
/// Time: O(n) where n = length of vectors
/// Space: O(1)
///
/// Example:
/// ```zig
/// var x = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{3}, &[_]f64{1, 2, 3}, .row_major);
/// defer x.deinit();
/// var y = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{3}, &[_]f64{4, 5, 6}, .row_major);
/// defer y.deinit();
/// const result = try dot(x, y); // 1*4 + 2*5 + 3*6 = 32
/// ```
pub fn dot(comptime T: type, x: NDArray(T, 1), y: NDArray(T, 1)) (NDArray(T, 1).Error)!T {
    // Check dimension mismatch first
    if (x.shape[0] != y.shape[0]) {
        return error.DimensionMismatch;
    }

    // Compute inner product: sum(x[i] * y[i])
    var result: T = 0;
    var x_iter = x.iterator();
    var y_iter = y.iterator();

    while (x_iter.next()) |x_val| {
        if (y_iter.next()) |y_val| {
            result += x_val * y_val;
        }
    }

    return result;
}

/// Vector update: y = αx + y (in-place)
///
/// Scales vector x by scalar alpha and adds it to y, storing result in y.
/// This is the fundamental BLAS axpy operation (a*x plus y).
///
/// Parameters:
/// - alpha: Scalar multiplier for x
/// - x: First vector (1D NDArray) — not modified
/// - y: Second vector (1D NDArray) — modified in-place
///
/// Errors:
/// - error.DimensionMismatch if x and y have different lengths
///
/// Time: O(n) where n = vector length
/// Space: O(1) (modifies y in-place)
///
/// Example:
/// ```zig
/// var x = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{3}, &[_]f64{1, 2, 3}, .row_major);
/// defer x.deinit();
/// var y = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{3}, &[_]f64{4, 5, 6}, .row_major);
/// defer y.deinit();
/// try axpy(2.0, x, &y); // y = 2*{1,2,3} + {4,5,6} = {6,9,12}
/// ```
pub fn axpy(comptime T: type, alpha: T, x: NDArray(T, 1), y: *NDArray(T, 1)) (NDArray(T, 1).Error)!void {
    // Check dimension mismatch
    if (x.shape[0] != y.shape[0]) {
        return error.DimensionMismatch;
    }

    // Compute y = alpha * x + y (in-place)
    var x_iter = x.iterator();
    var idx: usize = 0;

    while (x_iter.next()) |x_val| {
        y.data[idx] = alpha * x_val + y.data[idx];
        idx += 1;
    }
}

/// Compute Euclidean norm (L2 norm) of a vector
///
/// Returns the length of the vector: sqrt(sum(x_i^2))
///
/// Parameters:
/// - x: Vector (1D NDArray)
///
/// Returns: Non-negative scalar norm value
///
/// Time: O(n) where n = vector length
/// Space: O(1)
///
/// Example:
/// ```zig
/// var x = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{3}, &[_]f64{3, 4, 0}, .row_major);
/// defer x.deinit();
/// const norm = try nrm2(x); // sqrt(9 + 16 + 0) = 5
/// ```
pub fn nrm2(comptime T: type, x: NDArray(T, 1)) (NDArray(T, 1).Error)!T {
    // Compute L2 norm: sqrt(sum(x[i]^2))
    var sum_of_squares: T = 0;
    var iter = x.iterator();

    while (iter.next()) |val| {
        sum_of_squares += val * val;
    }

    return @sqrt(sum_of_squares);
}

/// Sum of absolute values of vector elements
///
/// Computes: sum(|x_i|) for all elements in x
///
/// Parameters:
/// - x: Vector (1D NDArray)
///
/// Returns: Non-negative scalar sum of absolute values
///
/// Time: O(n) where n = vector length
/// Space: O(1)
///
/// Example:
/// ```zig
/// var x = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{4}, &[_]f64{1, -2, 3, -4}, .row_major);
/// defer x.deinit();
/// const sum = try asum(x); // |1| + |-2| + |3| + |-4| = 10
/// ```
pub fn asum(comptime T: type, x: NDArray(T, 1)) (NDArray(T, 1).Error)!T {
    // Compute sum of absolute values: sum(|x[i]|)
    var result: T = 0;
    var iter = x.iterator();

    while (iter.next()) |val| {
        result += @abs(val);
    }

    return result;
}

/// Scale vector in-place: x = αx
///
/// Multiplies all elements of x by scalar alpha.
///
/// Parameters:
/// - alpha: Scalar multiplier
/// - x: Vector (1D NDArray) — modified in-place
///
/// Time: O(n) where n = vector length
/// Space: O(1) (modifies x in-place)
///
/// Example:
/// ```zig
/// var x = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{3}, &[_]f64{1, 2, 3}, .row_major);
/// defer x.deinit();
/// try scal(2.5, &x); // x = {2.5, 5.0, 7.5}
/// ```
pub fn scal(comptime T: type, alpha: T, x: *NDArray(T, 1)) (NDArray(T, 1).Error)!void {
    // Scale vector in-place: x = alpha * x
    for (0..x.shape[0]) |i| {
        x.data[i] *= alpha;
    }
}

// ============================================================================
// Tests
// ============================================================================

test "dot: basic 3-element vectors" {
    const allocator = testing.allocator;

    // x = [1, 2, 3], y = [4, 5, 6]
    // dot(x, y) = 1*4 + 2*5 + 3*6 = 32
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 4, 5, 6 }, .row_major);
    defer y.deinit();

    const result = try dot(f64, x, y);
    try testing.expectApproxEqAbs(32.0, result, 1e-10);
}

test "dot: single element vector" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{5}, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{3}, .row_major);
    defer y.deinit();

    const result = try dot(f64, x, y);
    try testing.expectApproxEqAbs(15.0, result, 1e-10);
}

test "dot: zero vector" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).zeros(allocator, &[_]usize{5}, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 1, 2, 3, 4, 5 }, .row_major);
    defer y.deinit();

    const result = try dot(f64, x, y);
    try testing.expectApproxEqAbs(0.0, result, 1e-10);
}

test "dot: negative values" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ -1, -2, 3, 4 }, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 2, 3, -1, -2 }, .row_major);
    defer y.deinit();

    // (-1)*2 + (-2)*3 + 3*(-1) + 4*(-2) = -2 - 6 - 3 - 8 = -19
    const result = try dot(f64, x, y);
    try testing.expectApproxEqAbs(-19.0, result, 1e-10);
}

test "dot: large vectors (1000 elements)" {
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f64, 1000);
    defer allocator.free(data_x);
    var data_y = try allocator.alloc(f64, 1000);
    defer allocator.free(data_y);

    for (0..1000) |i| {
        data_x[i] = @as(f64, @floatFromInt(i));
        data_y[i] = @as(f64, @floatFromInt(i));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1000}, data_x, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1000}, data_y, .row_major);
    defer y.deinit();

    const result = try dot(f64, x, y);
    // Sum of i^2 for i=0 to 999 = 999*1000*1999/6
    const expected = 999.0 * 1000.0 * 1999.0 / 6.0;
    try testing.expectApproxEqAbs(expected, result, expected * 1e-10);
}

test "dot: f32 precision" {
    const allocator = testing.allocator;

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{3}, &[_]f32{ 0.1, 0.2, 0.3 }, .row_major);
    defer x.deinit();
    var y = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{3}, &[_]f32{ 2.0, 3.0, 4.0 }, .row_major);
    defer y.deinit();

    const result = try dot(f32, x, y);
    // 0.1*2 + 0.2*3 + 0.3*4 = 0.2 + 0.6 + 1.2 = 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), result, 1e-5);
}

test "dot: dimension mismatch error" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 1, 2, 3, 4, 5 }, .row_major);
    defer y.deinit();

    const result = dot(f64, x, y);
    try testing.expectError(error.DimensionMismatch, result);
}

test "dot: orthogonal vectors (result = 0)" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 0 }, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 0, 1 }, .row_major);
    defer y.deinit();

    const result = try dot(f64, x, y);
    try testing.expectApproxEqAbs(0.0, result, 1e-10);
}

// ============================================================================
// axpy Tests
// ============================================================================

test "axpy: basic y = αx + y" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 4, 5, 6 }, .row_major);
    defer y.deinit();

    try axpy(f64, 2.0, x, &y);

    // y = 2.0*[1,2,3] + [4,5,6] = [2,4,6] + [4,5,6] = [6,9,12]
    try testing.expectApproxEqAbs(6.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(9.0, y.data[1], 1e-10);
    try testing.expectApproxEqAbs(12.0, y.data[2], 1e-10);
}

test "axpy: alpha = 0 (y unchanged)" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 4, 5, 6 }, .row_major);
    defer y.deinit();

    try axpy(f64, 0.0, x, &y);

    // y = 0*x + y = y
    try testing.expectApproxEqAbs(4.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(5.0, y.data[1], 1e-10);
    try testing.expectApproxEqAbs(6.0, y.data[2], 1e-10);
}

test "axpy: alpha = 1 (y = x + y)" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 4, 5, 6 }, .row_major);
    defer y.deinit();

    try axpy(f64, 1.0, x, &y);

    // y = 1*x + y = [1,2,3] + [4,5,6] = [5,7,9]
    try testing.expectApproxEqAbs(5.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(7.0, y.data[1], 1e-10);
    try testing.expectApproxEqAbs(9.0, y.data[2], 1e-10);
}

test "axpy: negative alpha" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 4, 5, 6 }, .row_major);
    defer y.deinit();

    try axpy(f64, -1.0, x, &y);

    // y = -1*[1,2,3] + [4,5,6] = [-1,-2,-3] + [4,5,6] = [3,3,3]
    try testing.expectApproxEqAbs(3.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(3.0, y.data[1], 1e-10);
    try testing.expectApproxEqAbs(3.0, y.data[2], 1e-10);
}

test "axpy: single element" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{5}, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{3}, .row_major);
    defer y.deinit();

    try axpy(f64, 2.0, x, &y);

    // y = 2*5 + 3 = 13
    try testing.expectApproxEqAbs(13.0, y.data[0], 1e-10);
}

test "axpy: large vectors" {
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f64, 1000);
    defer allocator.free(data_x);
    var data_y = try allocator.alloc(f64, 1000);
    defer allocator.free(data_y);

    for (0..1000) |i| {
        data_x[i] = 1.0;
        data_y[i] = @as(f64, @floatFromInt(i));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1000}, data_x, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1000}, data_y, .row_major);
    defer y.deinit();

    try axpy(f64, 2.0, x, &y);

    // y[i] = 2*1 + i = 2 + i
    for (0..1000) |i| {
        const expected = 2.0 + @as(f64, @floatFromInt(i));
        try testing.expectApproxEqAbs(expected, y.data[i], 1e-10);
    }
}

test "axpy: dimension mismatch" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 1, 2, 3, 4, 5 }, .row_major);
    defer y.deinit();

    const result = axpy(f64, 1.0, x, &y);
    try testing.expectError(error.DimensionMismatch, result);
}

test "axpy: f32 precision" {
    const allocator = testing.allocator;

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{2}, &[_]f32{ 0.5, 1.5 }, .row_major);
    defer x.deinit();
    var y = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{2}, &[_]f32{ 2.0, 3.0 }, .row_major);
    defer y.deinit();

    try axpy(f32, 0.5, x, &y);

    // y = 0.5*[0.5,1.5] + [2.0,3.0] = [0.25,0.75] + [2.0,3.0] = [2.25,3.75]
    try testing.expectApproxEqAbs(@as(f32, 2.25), y.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 3.75), y.data[1], 1e-5);
}

// ============================================================================
// nrm2 Tests
// ============================================================================

test "nrm2: 3-4-5 triangle" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 3, 4, 0 }, .row_major);
    defer x.deinit();

    const norm = try nrm2(f64, x);
    try testing.expectApproxEqAbs(5.0, norm, 1e-10);
}

test "nrm2: unit vector" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{1}, .row_major);
    defer x.deinit();

    const norm = try nrm2(f64, x);
    try testing.expectApproxEqAbs(1.0, norm, 1e-10);
}

test "nrm2: zero vector" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).zeros(allocator, &[_]usize{5}, .row_major);
    defer x.deinit();

    const norm = try nrm2(f64, x);
    try testing.expectApproxEqAbs(0.0, norm, 1e-10);
}

test "nrm2: negative values" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ -1, -1, -1 }, .row_major);
    defer x.deinit();

    const norm = try nrm2(f64, x);
    try testing.expectApproxEqAbs(@sqrt(3.0), norm, 1e-10);
}

test "nrm2: large vector (1000 elements)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 1000);
    defer allocator.free(data);

    for (0..1000) |i| {
        data[i] = 1.0;
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1000}, data, .row_major);
    defer x.deinit();

    const norm = try nrm2(f64, x);
    try testing.expectApproxEqAbs(@sqrt(1000.0), norm, 1e-8);
}

test "nrm2: f32 precision" {
    const allocator = testing.allocator;

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{2}, &[_]f32{ 1.0, 1.0 }, .row_major);
    defer x.deinit();

    const norm = try nrm2(f32, x);
    try testing.expectApproxEqAbs(@sqrt(2.0), norm, 1e-5);
}

test "nrm2: scaled vector" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 6, 8 }, .row_major);
    defer x.deinit();

    const norm = try nrm2(f64, x);
    // sqrt(36 + 64) = sqrt(100) = 10
    try testing.expectApproxEqAbs(10.0, norm, 1e-10);
}

// ============================================================================
// asum Tests
// ============================================================================

test "asum: mixed signs" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1, -2, 3, -4 }, .row_major);
    defer x.deinit();

    const sum = try asum(f64, x);
    // |1| + |-2| + |3| + |-4| = 1 + 2 + 3 + 4 = 10
    try testing.expectApproxEqAbs(10.0, sum, 1e-10);
}

test "asum: all positive" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer x.deinit();

    const sum = try asum(f64, x);
    try testing.expectApproxEqAbs(10.0, sum, 1e-10);
}

test "asum: all negative" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ -1, -2, -3, -4 }, .row_major);
    defer x.deinit();

    const sum = try asum(f64, x);
    try testing.expectApproxEqAbs(10.0, sum, 1e-10);
}

test "asum: zero vector" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).zeros(allocator, &[_]usize{5}, .row_major);
    defer x.deinit();

    const sum = try asum(f64, x);
    try testing.expectApproxEqAbs(0.0, sum, 1e-10);
}

test "asum: single element" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{-5}, .row_major);
    defer x.deinit();

    const sum = try asum(f64, x);
    try testing.expectApproxEqAbs(5.0, sum, 1e-10);
}

test "asum: large vector (1000 elements)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 1000);
    defer allocator.free(data);

    for (0..500) |i| {
        data[i] = 1.0;
    }
    for (500..1000) |i| {
        data[i] = -1.0;
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1000}, data, .row_major);
    defer x.deinit();

    const sum = try asum(f64, x);
    // 500 ones + 500 negatives = 500 + 500 = 1000
    try testing.expectApproxEqAbs(1000.0, sum, 1e-10);
}

test "asum: f32 precision" {
    const allocator = testing.allocator;

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{3}, &[_]f32{ 0.1, -0.2, 0.3 }, .row_major);
    defer x.deinit();

    const sum = try asum(f32, x);
    try testing.expectApproxEqAbs(@as(f32, 0.6), sum, 1e-5);
}

test "asum: fractional values" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 0.5, 0.5 }, .row_major);
    defer x.deinit();

    const sum = try asum(f64, x);
    try testing.expectApproxEqAbs(1.0, sum, 1e-10);
}

// ============================================================================
// scal Tests
// ============================================================================

test "scal: basic scaling" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    try scal(f64, 2.5, &x);

    // x = 2.5 * [1,2,3] = [2.5, 5.0, 7.5]
    try testing.expectApproxEqAbs(2.5, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(5.0, x.data[1], 1e-10);
    try testing.expectApproxEqAbs(7.5, x.data[2], 1e-10);
}

test "scal: alpha = 0" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    try scal(f64, 0.0, &x);

    // x = 0 * [1,2,3] = [0,0,0]
    try testing.expectApproxEqAbs(0.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(0.0, x.data[1], 1e-10);
    try testing.expectApproxEqAbs(0.0, x.data[2], 1e-10);
}

test "scal: alpha = 1 (unchanged)" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    try scal(f64, 1.0, &x);

    try testing.expectApproxEqAbs(1.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(2.0, x.data[1], 1e-10);
    try testing.expectApproxEqAbs(3.0, x.data[2], 1e-10);
}

test "scal: negative alpha" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    try scal(f64, -1.0, &x);

    // x = -1 * [1,2,3] = [-1,-2,-3]
    try testing.expectApproxEqAbs(-1.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(-2.0, x.data[1], 1e-10);
    try testing.expectApproxEqAbs(-3.0, x.data[2], 1e-10);
}

test "scal: single element" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{5}, .row_major);
    defer x.deinit();

    try scal(f64, 3.0, &x);

    try testing.expectApproxEqAbs(15.0, x.data[0], 1e-10);
}

test "scal: large vector (1000 elements)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 1000);
    defer allocator.free(data);

    for (0..1000) |i| {
        data[i] = @as(f64, @floatFromInt(i));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1000}, data, .row_major);
    defer x.deinit();

    try scal(f64, 2.0, &x);

    for (0..1000) |i| {
        const expected = 2.0 * @as(f64, @floatFromInt(i));
        try testing.expectApproxEqAbs(expected, x.data[i], 1e-10);
    }
}

test "scal: f32 precision" {
    const allocator = testing.allocator;

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{2}, &[_]f32{ 1.0, 2.0 }, .row_major);
    defer x.deinit();

    try scal(f32, 0.5, &x);

    try testing.expectApproxEqAbs(@as(f32, 0.5), x.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.0), x.data[1], 1e-5);
}

test "scal: fractional alpha" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 2, 4, 6 }, .row_major);
    defer x.deinit();

    try scal(f64, 0.5, &x);

    // x = 0.5 * [2,4,6] = [1,2,3]
    try testing.expectApproxEqAbs(1.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(2.0, x.data[1], 1e-10);
    try testing.expectApproxEqAbs(3.0, x.data[2], 1e-10);
}

test "scal: zero vector unchanged" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).zeros(allocator, &[_]usize{5}, .row_major);
    defer x.deinit();

    try scal(f64, 999.0, &x);

    for (0..5) |i| {
        try testing.expectApproxEqAbs(0.0, x.data[i], 1e-10);
    }
}
