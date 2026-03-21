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

// ============================================================================
// BLAS Level 2 — Matrix-Vector Operations
// ============================================================================

/// General matrix-vector multiply: y = αAx + βy
///
/// Performs the standard matrix-vector multiplication with scalar scaling.
/// A is an m×n matrix, x is an n×1 vector, y is an m×1 vector.
/// The result is stored in-place in y.
///
/// Parameters:
/// - alpha: Scalar multiplier for Ax
/// - A: Matrix (2D NDArray with shape [m, n]) — not modified
/// - x: Vector (1D NDArray with shape [n]) — not modified
/// - beta: Scalar multiplier for y
/// - y: Result vector (1D NDArray with shape [m]) — modified in-place
///
/// Errors:
/// - error.DimensionMismatch if A.shape[1] != x.shape[0] or A.shape[0] != y.shape[0]
///
/// Time: O(m*n) where A is m×n
/// Space: O(1) (modifies y in-place)
///
/// Example:
/// ```zig
/// // A = [[1, 2], [3, 4], [5, 6]]  (3×2 matrix)
/// // x = [7, 8]  (2×1 vector)
/// // y = [1, 1, 1]  (3×1 vector)
/// // gemv(2.0, A, x, 3.0, &y)
/// // y = 2.0*A*x + 3.0*[1,1,1]
/// //   = 2.0*[23, 53, 83] + [3,3,3]
/// //   = [49, 109, 169]
/// ```
pub fn gemv(comptime T: type, alpha: T, A: NDArray(T, 2), x: NDArray(T, 1), beta: T, y: *NDArray(T, 1)) (NDArray(T, 1).Error)!void {
    // Validate dimensions
    if (A.shape[1] != x.shape[0]) {
        return error.DimensionMismatch;
    }
    if (A.shape[0] != y.shape[0]) {
        return error.DimensionMismatch;
    }

    const m = A.shape[0]; // rows
    const n = A.shape[1]; // columns

    // y = alpha * A * x + beta * y
    // For each row i of A:
    //   y[i] = beta * y[i] + alpha * sum_j(A[i,j] * x[j])
    for (0..m) |i| {
        var sum: T = 0;
        for (0..n) |j| {
            // Row-major layout: A[i,j] is at A.data[i*n + j]
            const a_val = A.data[i * n + j];
            const x_val = x.data[j];
            sum += a_val * x_val;
        }
        y.data[i] = beta * y.data[i] + alpha * sum;
    }
}

/// Rank-1 update: A = A + αxy^T
///
/// Performs an outer product of vectors x and y, scaled by alpha,
/// and adds the result to matrix A in-place.
///
/// Parameters:
/// - alpha: Scalar multiplier for the outer product
/// - x: First vector (1D NDArray with shape [m]) — not modified
/// - y: Second vector (1D NDArray with shape [n]) — not modified
/// - A: Matrix (2D NDArray with shape [m, n]) — modified in-place
///
/// Errors:
/// - error.DimensionMismatch if A.shape[0] != x.shape[0] or A.shape[1] != y.shape[0]
///
/// Time: O(m*n) where A is m×n
/// Space: O(1) (modifies A in-place)
///
/// Example:
/// ```zig
/// // A = [[0, 0], [0, 0]]  (2×2 matrix)
/// // x = [1, 2]  (2×1 vector)
/// // y = [3, 4]  (2×1 vector)
/// // ger(1.0, x, y, &A)
/// // A = A + x*y^T = [[0,0],[0,0]] + [[1*3, 1*4], [2*3, 2*4]]
/// //   = [[3, 4], [6, 8]]
/// ```
pub fn ger(comptime T: type, alpha: T, x: NDArray(T, 1), y: NDArray(T, 1), A: *NDArray(T, 2)) (NDArray(T, 1).Error)!void {
    // Validate dimensions
    if (A.shape[0] != x.shape[0]) {
        return error.DimensionMismatch;
    }
    if (A.shape[1] != y.shape[0]) {
        return error.DimensionMismatch;
    }

    const m = A.shape[0]; // rows
    const n = A.shape[1]; // columns

    // A = A + alpha * x * y^T
    // For each i,j: A[i,j] = A[i,j] + alpha * x[i] * y[j]
    for (0..m) |i| {
        for (0..n) |j| {
            const x_val = x.data[i];
            const y_val = y.data[j];
            A.data[i * n + j] += alpha * x_val * y_val;
        }
    }
}

// ============================================================================
// gemv Tests
// ============================================================================

test "gemv: basic 3x2 matrix-vector multiply" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4], [5, 6]]  (3x2 matrix)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer A.deinit();

    // x = [7, 8]  (2x1 vector)
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 7, 8 }, .row_major);
    defer x.deinit();

    // y = [1, 1, 1]  (3x1 vector)
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 1, 1 }, .row_major);
    defer y.deinit();

    // y = A*x + y
    // A*x = [[1*7 + 2*8], [3*7 + 4*8], [5*7 + 6*8]] = [[23], [53], [83]]
    // y = [23, 53, 83] + [1, 1, 1] = [24, 54, 84]
    try gemv(f64, 1.0, A, x, 1.0, &y);

    try testing.expectApproxEqAbs(24.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(54.0, y.data[1], 1e-10);
    try testing.expectApproxEqAbs(84.0, y.data[2], 1e-10);
}

test "gemv: identity matrix multiply" {
    const allocator = testing.allocator;

    // A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  (3x3 identity)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 2, 3, 4 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{3}, .row_major);
    defer y.deinit();

    // y = I*x + 0*y = [2, 3, 4]
    try gemv(f64, 1.0, A, x, 0.0, &y);

    try testing.expectApproxEqAbs(2.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(3.0, y.data[1], 1e-10);
    try testing.expectApproxEqAbs(4.0, y.data[2], 1e-10);
}

test "gemv: zero alpha (y = beta*y)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 5, 6 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 10, 20 }, .row_major);
    defer y.deinit();

    // y = 0*A*x + 2*[10, 20] = [20, 40]
    try gemv(f64, 0.0, A, x, 2.0, &y);

    try testing.expectApproxEqAbs(20.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(40.0, y.data[1], 1e-10);
}

test "gemv: zero beta (y = alpha*A*x)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 1 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 999, 999 }, .row_major);
    defer y.deinit();

    // y = 2*A*x + 0*[999, 999] = 2*[3, 7] = [6, 14]
    // A*x = [[1*1 + 2*1], [3*1 + 4*1]] = [3, 7]
    try gemv(f64, 2.0, A, x, 0.0, &y);

    try testing.expectApproxEqAbs(6.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(14.0, y.data[1], 1e-10);
}

test "gemv: alpha = 1, beta = 1" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 3, 4, 5 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 2 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 1 }, .row_major);
    defer y.deinit();

    // A*x = [[2*1 + 3*2], [4*1 + 5*2]] = [8, 14]
    // y = [8, 14] + [1, 1] = [9, 15]
    try gemv(f64, 1.0, A, x, 1.0, &y);

    try testing.expectApproxEqAbs(9.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(15.0, y.data[1], 1e-10);
}

test "gemv: negative alpha" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 1, 1, 1 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 2, 3 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 10, 10 }, .row_major);
    defer y.deinit();

    // A*x = [[1*2 + 1*3], [1*2 + 1*3]] = [5, 5]
    // y = -1*[5, 5] + 10*[1, 1] = [-5, -5] + [10, 10] = [5, 5]
    try gemv(f64, -1.0, A, x, 1.0, &y);

    try testing.expectApproxEqAbs(5.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(5.0, y.data[1], 1e-10);
}

test "gemv: negative beta" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 0, 0, 3 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 5, 2 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 4, 4 }, .row_major);
    defer y.deinit();

    // A*x = [[2*5 + 0*2], [0*5 + 3*2]] = [10, 6]
    // y = [10, 6] + -2*[4, 4] = [10, 6] + [-8, -8] = [2, -2]
    try gemv(f64, 1.0, A, x, -2.0, &y);

    try testing.expectApproxEqAbs(2.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(-2.0, y.data[1], 1e-10);
}

test "gemv: 1x1 matrix (scalar case)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 1 }, &[_]f64{5}, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{3}, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{2}, .row_major);
    defer y.deinit();

    // y = 5*3 + 2 = 17
    try gemv(f64, 1.0, A, x, 1.0, &y);

    try testing.expectApproxEqAbs(17.0, y.data[0], 1e-10);
}

test "gemv: dimension mismatch A columns != x length" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 1 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 0, 0 }, .row_major);
    defer y.deinit();

    const result = gemv(f64, 1.0, A, x, 1.0, &y);
    try testing.expectError(error.DimensionMismatch, result);
}

test "gemv: dimension mismatch A rows != y length" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 1 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 0, 0 }, .row_major);
    defer y.deinit();

    const result = gemv(f64, 1.0, A, x, 1.0, &y);
    try testing.expectError(error.DimensionMismatch, result);
}

test "gemv: rectangular 4x3 matrix" {
    const allocator = testing.allocator;

    // A = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]  (4x3)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 3 }, &[_]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 5, 7, 3 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{4}, .row_major);
    defer y.deinit();

    // y = A*x = [5, 7, 3, 0]
    try gemv(f64, 1.0, A, x, 0.0, &y);

    try testing.expectApproxEqAbs(5.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(7.0, y.data[1], 1e-10);
    try testing.expectApproxEqAbs(3.0, y.data[2], 1e-10);
    try testing.expectApproxEqAbs(0.0, y.data[3], 1e-10);
}

test "gemv: large 100x100 matrix" {
    const allocator = testing.allocator;

    // Create 100x100 identity matrix
    var data_A = try allocator.alloc(f64, 10000);
    defer allocator.free(data_A);
    for (0..10000) |i| {
        data_A[i] = if (i % 101 == 0) 1.0 else 0.0;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 100, 100 }, data_A, .row_major);
    defer A.deinit();

    var data_x = try allocator.alloc(f64, 100);
    defer allocator.free(data_x);
    for (0..100) |i| {
        data_x[i] = @as(f64, @floatFromInt(i));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{100}, data_x, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{100}, .row_major);
    defer y.deinit();

    // y = I*x = x
    try gemv(f64, 1.0, A, x, 0.0, &y);

    for (0..100) |i| {
        const expected = @as(f64, @floatFromInt(i));
        try testing.expectApproxEqAbs(expected, y.data[i], 1e-10);
    }
}

test "gemv: f32 precision" {
    const allocator = testing.allocator;

    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{ 0.5, 1.5, 2.5, 3.5 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{2}, &[_]f32{ 1.0, 2.0 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{2}, &[_]f32{ 0.0, 0.0 }, .row_major);
    defer y.deinit();

    // A*x = [[0.5*1 + 1.5*2], [2.5*1 + 3.5*2]] = [3.5, 9.5]
    try gemv(f32, 1.0, A, x, 0.0, &y);

    try testing.expectApproxEqAbs(@as(f32, 3.5), y.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 9.5), y.data[1], 1e-5);
}

test "gemv: all zero matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 5, 6, 7 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer y.deinit();

    // y = 0*[0]*x + 2*[1, 2, 3] = [2, 4, 6]
    try gemv(f64, 0.0, A, x, 2.0, &y);

    try testing.expectApproxEqAbs(2.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(4.0, y.data[1], 1e-10);
    try testing.expectApproxEqAbs(6.0, y.data[2], 1e-10);
}

test "gemv: complex scaling both alpha and beta" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 1 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 2, 3 }, .row_major);
    defer y.deinit();

    // A*x = [3, 7]
    // y = 2.5*[3, 7] + 0.5*[2, 3] = [7.5, 17.5] + [1, 1.5] = [8.5, 19]
    try gemv(f64, 2.5, A, x, 0.5, &y);

    try testing.expectApproxEqAbs(8.5, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(19.0, y.data[1], 1e-10);
}

// ============================================================================
// ger Tests
// ============================================================================

test "ger: basic 2x2 outer product" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 2 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 3, 4 }, .row_major);
    defer y.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer A.deinit();

    // A = 0 + 1*x*y^T = [[1*3, 1*4], [2*3, 2*4]] = [[3, 4], [6, 8]]
    try ger(f64, 1.0, x, y, &A);

    try testing.expectApproxEqAbs(3.0, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(4.0, A.data[1], 1e-10);
    try testing.expectApproxEqAbs(6.0, A.data[2], 1e-10);
    try testing.expectApproxEqAbs(8.0, A.data[3], 1e-10);
}

test "ger: add to existing matrix" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 1 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 1 }, .row_major);
    defer y.deinit();

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    // A = [[1, 2], [3, 4]] + 1*[[1, 1], [1, 1]] = [[2, 3], [4, 5]]
    try ger(f64, 1.0, x, y, &A);

    try testing.expectApproxEqAbs(2.0, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(3.0, A.data[1], 1e-10);
    try testing.expectApproxEqAbs(4.0, A.data[2], 1e-10);
    try testing.expectApproxEqAbs(5.0, A.data[3], 1e-10);
}

test "ger: zero alpha (A unchanged)" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 5, 6 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 7, 8 }, .row_major);
    defer y.deinit();

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    // A = [[1, 2], [3, 4]] + 0*x*y^T = [[1, 2], [3, 4]] (unchanged)
    try ger(f64, 0.0, x, y, &A);

    try testing.expectApproxEqAbs(1.0, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(2.0, A.data[1], 1e-10);
    try testing.expectApproxEqAbs(3.0, A.data[2], 1e-10);
    try testing.expectApproxEqAbs(4.0, A.data[3], 1e-10);
}

test "ger: negative alpha" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 2, 1 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 2 }, .row_major);
    defer y.deinit();

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 5, 5, 5 }, .row_major);
    defer A.deinit();

    // x*y^T = [[2*1, 2*2], [1*1, 1*2]] = [[2, 4], [1, 2]]
    // A = [[5, 5], [5, 5]] + -1*[[2, 4], [1, 2]] = [[3, 1], [4, 3]]
    try ger(f64, -1.0, x, y, &A);

    try testing.expectApproxEqAbs(3.0, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(1.0, A.data[1], 1e-10);
    try testing.expectApproxEqAbs(4.0, A.data[2], 1e-10);
    try testing.expectApproxEqAbs(3.0, A.data[3], 1e-10);
}

test "ger: 1x1 matrix" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{3}, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{4}, .row_major);
    defer y.deinit();

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 1 }, &[_]f64{5}, .row_major);
    defer A.deinit();

    // A = 5 + 2*3*4 = 5 + 24 = 29
    try ger(f64, 2.0, x, y, &A);

    try testing.expectApproxEqAbs(29.0, A.data[0], 1e-10);
}

test "ger: 3x2 matrix (tall matrix)" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 4, 5 }, .row_major);
    defer y.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 2 }, .row_major);
    defer A.deinit();

    // x*y^T = [[1*4, 1*5], [2*4, 2*5], [3*4, 3*5]] = [[4, 5], [8, 10], [12, 15]]
    try ger(f64, 1.0, x, y, &A);

    try testing.expectApproxEqAbs(4.0, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(5.0, A.data[1], 1e-10);
    try testing.expectApproxEqAbs(8.0, A.data[2], 1e-10);
    try testing.expectApproxEqAbs(10.0, A.data[3], 1e-10);
    try testing.expectApproxEqAbs(12.0, A.data[4], 1e-10);
    try testing.expectApproxEqAbs(15.0, A.data[5], 1e-10);
}

test "ger: 2x3 matrix (wide matrix)" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 2 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 3, 4, 5 }, .row_major);
    defer y.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer A.deinit();

    // x*y^T = [[1*3, 1*4, 1*5], [2*3, 2*4, 2*5]] = [[3, 4, 5], [6, 8, 10]]
    try ger(f64, 1.0, x, y, &A);

    try testing.expectApproxEqAbs(3.0, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(4.0, A.data[1], 1e-10);
    try testing.expectApproxEqAbs(5.0, A.data[2], 1e-10);
    try testing.expectApproxEqAbs(6.0, A.data[3], 1e-10);
    try testing.expectApproxEqAbs(8.0, A.data[4], 1e-10);
    try testing.expectApproxEqAbs(10.0, A.data[5], 1e-10);
}

test "ger: dimension mismatch x length != A rows" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 2 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer y.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer A.deinit();

    const result = ger(f64, 1.0, x, y, &A);
    try testing.expectError(error.DimensionMismatch, result);
}

test "ger: dimension mismatch y length != A columns" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 2 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 2 }, .row_major);
    defer y.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer A.deinit();

    const result = ger(f64, 1.0, x, y, &A);
    try testing.expectError(error.DimensionMismatch, result);
}

test "ger: zero vector x" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).zeros(allocator, &[_]usize{3}, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 5, 6 }, .row_major);
    defer y.deinit();

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer A.deinit();

    // x*y^T = [[0, 0], [0, 0], [0, 0]]
    // A = [[1, 2], [3, 4], [5, 6]] + 2*[[0, 0], [0, 0], [0, 0]] = [[1, 2], [3, 4], [5, 6]]
    try ger(f64, 2.0, x, y, &A);

    try testing.expectApproxEqAbs(1.0, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(2.0, A.data[1], 1e-10);
    try testing.expectApproxEqAbs(3.0, A.data[2], 1e-10);
    try testing.expectApproxEqAbs(4.0, A.data[3], 1e-10);
    try testing.expectApproxEqAbs(5.0, A.data[4], 1e-10);
    try testing.expectApproxEqAbs(6.0, A.data[5], 1e-10);
}

test "ger: large 100x100 outer product" {
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f64, 100);
    defer allocator.free(data_x);
    for (0..100) |i| {
        data_x[i] = 1.0;
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{100}, data_x, .row_major);
    defer x.deinit();

    var data_y = try allocator.alloc(f64, 100);
    defer allocator.free(data_y);
    for (0..100) |i| {
        data_y[i] = 1.0;
    }

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{100}, data_y, .row_major);
    defer y.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 100, 100 }, .row_major);
    defer A.deinit();

    // A = 0 + 1*[all 1s]*[all 1s]^T = all 1s matrix
    try ger(f64, 1.0, x, y, &A);

    for (0..10000) |i| {
        try testing.expectApproxEqAbs(1.0, A.data[i], 1e-10);
    }
}

test "ger: f32 precision" {
    const allocator = testing.allocator;

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{2}, &[_]f32{ 0.5, 1.5 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{2}, &[_]f32{ 2.0, 3.0 }, .row_major);
    defer y.deinit();

    var A = try NDArray(f32, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer A.deinit();

    // x*y^T = [[0.5*2, 0.5*3], [1.5*2, 1.5*3]] = [[1, 1.5], [3, 4.5]]
    try ger(f32, 1.0, x, y, &A);

    try testing.expectApproxEqAbs(@as(f32, 1.0), A.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.5), A.data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 3.0), A.data[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 4.5), A.data[3], 1e-5);
}

test "ger: negative values in vectors" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ -1, 2 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 3, -4 }, .row_major);
    defer y.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer A.deinit();

    // x*y^T = [[-1*3, -1*-4], [2*3, 2*-4]] = [[-3, 4], [6, -8]]
    try ger(f64, 1.0, x, y, &A);

    try testing.expectApproxEqAbs(-3.0, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(4.0, A.data[1], 1e-10);
    try testing.expectApproxEqAbs(6.0, A.data[2], 1e-10);
    try testing.expectApproxEqAbs(-8.0, A.data[3], 1e-10);
}

// ============================================================================
// BLAS Level 3 — Matrix-Matrix Operations
// ============================================================================

/// General matrix-matrix multiply: C = αAB + βC
///
/// Performs the standard matrix-matrix multiplication with scalar scaling.
/// A is an m×k matrix, B is a k×n matrix, C is an m×n matrix.
/// The result is stored in-place in C.
///
/// This is the most important BLAS Level 3 operation, fundamental for:
/// - Neural network layer computations (dense layer = gemm)
/// - Scientific computing and numerical methods
/// - Graphics and linear algebra algorithms
/// - Cache-optimized blocking transformations
///
/// Parameters:
/// - alpha: Scalar multiplier for AB
/// - A: Matrix (2D NDArray with shape [m, k]) — not modified
/// - B: Matrix (2D NDArray with shape [k, n]) — not modified
/// - beta: Scalar multiplier for C
/// - C: Result matrix (2D NDArray with shape [m, n]) — modified in-place
///
/// Errors:
/// - error.DimensionMismatch if A.shape[1] != B.shape[0] (k dimension mismatch)
/// - error.DimensionMismatch if C.shape[0] != A.shape[0] (m dimension mismatch)
/// - error.DimensionMismatch if C.shape[1] != B.shape[1] (n dimension mismatch)
///
/// Time: O(m*n*k) where A is m×k, B is k×n, C is m×n
/// Space: O(1) (modifies C in-place)
///
/// Example:
/// ```zig
/// // A = [[1, 2], [3, 4]]  (2×2 matrix)
/// // B = [[5, 6], [7, 8]]  (2×2 matrix)
/// // C = [[1, 1], [1, 1]]  (2×2 matrix)
/// // gemm(1.0, A, B, 1.0, &C)
/// // AB = [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]
/// //    = [[19, 22], [43, 50]]
/// // C = [[19, 22], [43, 50]] + [[1, 1], [1, 1]]
/// //   = [[20, 23], [44, 51]]
/// ```
pub fn gemm(comptime T: type, alpha: T, A: NDArray(T, 2), B: NDArray(T, 2), beta: T, C: *NDArray(T, 2)) (NDArray(T, 2).Error)!void {
    // Validate dimensions
    // A: m×k, B: k×n, C: m×n
    const m = A.shape[0];
    const k = A.shape[1];
    const n = B.shape[1];

    // Check A.columns == B.rows
    if (A.shape[1] != B.shape[0]) {
        return error.DimensionMismatch;
    }

    // Check C.rows == A.rows
    if (C.shape[0] != A.shape[0]) {
        return error.DimensionMismatch;
    }

    // Check C.columns == B.columns
    if (C.shape[1] != B.shape[1]) {
        return error.DimensionMismatch;
    }

    // Operation: C = α*A*B + β*C
    // First scale C by beta, then accumulate α*A*B
    // Loop order: i (rows of C), j (cols of C), k (inner dimension)

    // Step 1: Scale C by beta
    for (0..m * n) |idx| {
        C.data[idx] = beta * C.data[idx];
    }

    // Step 2: Accumulate α*A*B
    // For each row i of C, for each column j of C:
    //   C[i,j] += α * Σ_k (A[i,k] * B[k,j])
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: T = 0;
            for (0..k) |p| {
                // A[i,p] is at flat index: i*k + p
                // B[p,j] is at flat index: p*n + j
                const a_val = A.data[i * k + p];
                const b_val = B.data[p * n + j];
                sum += a_val * b_val;
            }
            // C[i,j] is at flat index: i*n + j
            C.data[i * n + j] += alpha * sum;
        }
    }
}

// ============================================================================
// gemm Tests
// ============================================================================

test "gemm: basic 2x2 matrix multiply" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4]]  (2×2 matrix)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    // B = [[5, 6], [7, 8]]  (2×2 matrix)
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 6, 7, 8 }, .row_major);
    defer B.deinit();

    // C = [[1, 1], [1, 1]]  (2×2 matrix)
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 1, 1, 1 }, .row_major);
    defer C.deinit();

    // AB = [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]
    //    = [[19, 22], [43, 50]]
    // C = 1.0*AB + 1.0*C = [[20, 23], [44, 51]]
    try gemm(f64, 1.0, A, B, 1.0, &C);

    try testing.expectApproxEqAbs(20.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(23.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(44.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(51.0, C.data[3], 1e-10);
}

test "gemm: 3x3 matrix multiply with identity" {
    const allocator = testing.allocator;

    // A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  (3×3 matrix)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .row_major);
    defer A.deinit();

    // I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  (3×3 identity)
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 }, .row_major);
    defer B.deinit();

    // C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer C.deinit();

    // C = A*I + 0*C = A
    try gemm(f64, 1.0, A, B, 0.0, &C);

    try testing.expectApproxEqAbs(1.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(2.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(3.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(4.0, C.data[3], 1e-10);
    try testing.expectApproxEqAbs(5.0, C.data[4], 1e-10);
    try testing.expectApproxEqAbs(6.0, C.data[5], 1e-10);
    try testing.expectApproxEqAbs(7.0, C.data[6], 1e-10);
    try testing.expectApproxEqAbs(8.0, C.data[7], 1e-10);
    try testing.expectApproxEqAbs(9.0, C.data[8], 1e-10);
}

test "gemm: 1x1 matrix (scalar case)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 1 }, &[_]f64{2}, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 1 }, &[_]f64{3}, .row_major);
    defer B.deinit();

    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 1 }, &[_]f64{5}, .row_major);
    defer C.deinit();

    // C = 2*2*3 + 3*5 = 12 + 15 = 27
    try gemm(f64, 2.0, A, B, 3.0, &C);

    try testing.expectApproxEqAbs(27.0, C.data[0], 1e-10);
}

test "gemm: rectangular matrix 2x3 times 3x2" {
    const allocator = testing.allocator;

    // A = [[1, 2, 3], [4, 5, 6]]  (2×3 matrix)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer A.deinit();

    // B = [[7, 8], [9, 10], [11, 12]]  (3×2 matrix)
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f64{ 7, 8, 9, 10, 11, 12 }, .row_major);
    defer B.deinit();

    // C = [[0, 0], [0, 0]]  (2×2 matrix)
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer C.deinit();

    // A*B = [[1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12], [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]]
    //     = [[58, 64], [139, 154]]
    try gemm(f64, 1.0, A, B, 0.0, &C);

    try testing.expectApproxEqAbs(58.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(64.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(139.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(154.0, C.data[3], 1e-10);
}

test "gemm: rectangular matrix 3x2 times 2x3 results in 3x3" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4], [5, 6]]  (3×2 matrix)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer A.deinit();

    // B = [[7, 8, 9], [10, 11, 12]]  (2×3 matrix)
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 7, 8, 9, 10, 11, 12 }, .row_major);
    defer B.deinit();

    // C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  (3×3 matrix)
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer C.deinit();

    // A*B = [[1*7 + 2*10, 1*8 + 2*11, 1*9 + 2*12], ...]
    //     = [[27, 30, 33], [61, 68, 75], [95, 106, 117]]
    try gemm(f64, 1.0, A, B, 0.0, &C);

    try testing.expectApproxEqAbs(27.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(30.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(33.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(61.0, C.data[3], 1e-10);
    try testing.expectApproxEqAbs(68.0, C.data[4], 1e-10);
    try testing.expectApproxEqAbs(75.0, C.data[5], 1e-10);
    try testing.expectApproxEqAbs(95.0, C.data[6], 1e-10);
    try testing.expectApproxEqAbs(106.0, C.data[7], 1e-10);
    try testing.expectApproxEqAbs(117.0, C.data[8], 1e-10);
}

test "gemm: alpha = 0 (C = beta*C only)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 100, 200, 300, 400 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 6, 7, 8 }, .row_major);
    defer B.deinit();

    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 10, 20, 30, 40 }, .row_major);
    defer C.deinit();

    // C = 0*A*B + 2*C = 2*[[10, 20], [30, 40]] = [[20, 40], [60, 80]]
    try gemm(f64, 0.0, A, B, 2.0, &C);

    try testing.expectApproxEqAbs(20.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(40.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(60.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(80.0, C.data[3], 1e-10);
}

test "gemm: beta = 0 (C = alpha*A*B)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 6, 7, 8 }, .row_major);
    defer B.deinit();

    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 999, 999, 999, 999 }, .row_major);
    defer C.deinit();

    // A*B = [[19, 22], [43, 50]]
    // C = 2*[[19, 22], [43, 50]] + 0*C = [[38, 44], [86, 100]]
    try gemm(f64, 2.0, A, B, 0.0, &C);

    try testing.expectApproxEqAbs(38.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(44.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(86.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(100.0, C.data[3], 1e-10);
}

test "gemm: alpha = 1, beta = 1 (standard accumulation)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 1, 1, 2 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 3, 0, 0, 3 }, .row_major);
    defer B.deinit();

    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 1, 1, 1 }, .row_major);
    defer C.deinit();

    // A*B = [[2*3 + 1*0, 2*0 + 1*3], [1*3 + 2*0, 1*0 + 2*3]] = [[6, 3], [3, 6]]
    // C = [[6, 3], [3, 6]] + [[1, 1], [1, 1]] = [[7, 4], [4, 7]]
    try gemm(f64, 1.0, A, B, 1.0, &C);

    try testing.expectApproxEqAbs(7.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(4.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(4.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(7.0, C.data[3], 1e-10);
}

test "gemm: negative alpha" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 1, 1, 1 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 3, 4, 5 }, .row_major);
    defer B.deinit();

    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 10, 10, 10, 10 }, .row_major);
    defer C.deinit();

    // A*B = [[1*2 + 1*4, 1*3 + 1*5], [1*2 + 1*4, 1*3 + 1*5]] = [[6, 8], [6, 8]]
    // C = -1*[[6, 8], [6, 8]] + 1*[[10, 10], [10, 10]] = [[-6, -8], [-6, -8]] + [[10, 10], [10, 10]] = [[4, 2], [4, 2]]
    try gemm(f64, -1.0, A, B, 1.0, &C);

    try testing.expectApproxEqAbs(4.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(2.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(4.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(2.0, C.data[3], 1e-10);
}

test "gemm: negative beta" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 0, 0, 1 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 3, 2, 4 }, .row_major);
    defer B.deinit();

    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 10, 10, 10, 10 }, .row_major);
    defer C.deinit();

    // A*B = [[1*5 + 0*2, 1*3 + 0*4], [0*5 + 1*2, 0*3 + 1*4]] = [[5, 3], [2, 4]]
    // C = 1*[[5, 3], [2, 4]] + -2*[[10, 10], [10, 10]] = [[5, 3], [2, 4]] + [[-20, -20], [-20, -20]] = [[-15, -17], [-18, -16]]
    try gemm(f64, 1.0, A, B, -2.0, &C);

    try testing.expectApproxEqAbs(-15.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(-17.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(-18.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(-16.0, C.data[3], 1e-10);
}

test "gemm: zero matrix B" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer B.deinit();

    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 6, 7, 8 }, .row_major);
    defer C.deinit();

    // A*B = [[0, 0], [0, 0]]
    // C = 0 + 2*[[5, 6], [7, 8]] = [[10, 12], [14, 16]]
    try gemm(f64, 1.0, A, B, 2.0, &C);

    try testing.expectApproxEqAbs(10.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(12.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(14.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(16.0, C.data[3], 1e-10);
}

test "gemm: zero matrix A" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 10, 20, 30, 40 }, .row_major);
    defer C.deinit();

    // A*B = [[0, 0], [0, 0]]
    // C = 0 + 3*[[10, 20], [30, 40]] = [[30, 60], [90, 120]]
    try gemm(f64, 2.0, A, B, 3.0, &C);

    try testing.expectApproxEqAbs(30.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(60.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(90.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(120.0, C.data[3], 1e-10);
}

test "gemm: identity matrices" {
    const allocator = testing.allocator;

    // A = I
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 }, .row_major);
    defer A.deinit();

    // B = I
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 }, .row_major);
    defer B.deinit();

    // C = zeros
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer C.deinit();

    // I*I = I
    // C = 1*I + 0*0 = I
    try gemm(f64, 1.0, A, B, 0.0, &C);

    // Verify C is identity
    for (0..3) |i| {
        for (0..3) |j| {
            const expected = if (i == j) 1.0 else 0.0;
            try testing.expectApproxEqAbs(expected, C.data[i * 3 + j], 1e-10);
        }
    }
}

test "gemm: negative values in matrices" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ -1, 2, 3, -4 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, -2, -3, 1 }, .row_major);
    defer B.deinit();

    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer C.deinit();

    // A*B = [[-1*5 + 2*(-3), -1*(-2) + 2*1], [3*5 + (-4)*(-3), 3*(-2) + (-4)*1]]
    //     = [[-5 - 6, 2 + 2], [15 + 12, -6 - 4]]
    //     = [[-11, 4], [27, -10]]
    try gemm(f64, 1.0, A, B, 0.0, &C);

    try testing.expectApproxEqAbs(-11.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(4.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(27.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(-10.0, C.data[3], 1e-10);
}

test "gemm: dimension mismatch A columns != B rows" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer C.deinit();

    const result = gemm(f64, 1.0, A, B, 0.0, &C);
    try testing.expectError(error.DimensionMismatch, result);
}

test "gemm: dimension mismatch C rows != A rows" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 6, 7, 8 }, .row_major);
    defer B.deinit();

    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 2 }, .row_major);
    defer C.deinit();

    const result = gemm(f64, 1.0, A, B, 0.0, &C);
    try testing.expectError(error.DimensionMismatch, result);
}

test "gemm: dimension mismatch C columns != B columns" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 6, 7, 8 }, .row_major);
    defer B.deinit();

    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer C.deinit();

    const result = gemm(f64, 1.0, A, B, 0.0, &C);
    try testing.expectError(error.DimensionMismatch, result);
}

test "gemm: f32 precision" {
    const allocator = testing.allocator;

    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{ 0.5, 1.5, 2.5, 3.5 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{ 2.0, 1.0, 1.0, 2.0 }, .row_major);
    defer B.deinit();

    var C = try NDArray(f32, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer C.deinit();

    // A*B = [[0.5*2 + 1.5*1, 0.5*1 + 1.5*2], [2.5*2 + 3.5*1, 2.5*1 + 3.5*2]]
    //     = [[2.5, 3.5], [7.5, 9.5]]
    try gemm(f32, 1.0, A, B, 0.0, &C);

    try testing.expectApproxEqAbs(@as(f32, 2.5), C.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 3.5), C.data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 7.5), C.data[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 9.5), C.data[3], 1e-5);
}

test "gemm: 32x32 stress test with random values" {
    const allocator = testing.allocator;

    // Create 32x32 matrices with simple predictable values
    var A_data = try allocator.alloc(f64, 32 * 32);
    defer allocator.free(A_data);
    for (0..32 * 32) |i| {
        A_data[i] = @as(f64, @floatFromInt((i % 32) + 1));
    }

    var B_data = try allocator.alloc(f64, 32 * 32);
    defer allocator.free(B_data);
    for (0..32 * 32) |i| {
        B_data[i] = @as(f64, @floatFromInt((i / 32) + 1));
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 32, 32 }, A_data, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 32, 32 }, B_data, .row_major);
    defer B.deinit();

    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 32, 32 }, .row_major);
    defer C.deinit();

    // Just verify it runs and produces reasonable non-zero results
    try gemm(f64, 1.0, A, B, 0.0, &C);

    // Check that result is non-zero (at least some elements computed)
    var sum: f64 = 0;
    for (C.data) |val| {
        sum += @abs(val);
    }
    try testing.expect(sum > 0);
}

test "gemm: 64x64 stress test for larger matrices" {
    const allocator = testing.allocator;

    var A_data = try allocator.alloc(f64, 64 * 32);
    defer allocator.free(A_data);
    for (0..64 * 32) |i| {
        A_data[i] = 1.0;
    }

    var B_data = try allocator.alloc(f64, 32 * 64);
    defer allocator.free(B_data);
    for (0..32 * 64) |i| {
        B_data[i] = 1.0;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 32 }, A_data, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 32, 64 }, B_data, .row_major);
    defer B.deinit();

    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 64, 64 }, .row_major);
    defer C.deinit();

    // All 1s * all 1s with k=32 should produce all 32s
    try gemm(f64, 1.0, A, B, 0.0, &C);

    // Spot check a few values
    try testing.expectApproxEqAbs(32.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(32.0, C.data[64 * 64 - 1], 1e-10);
}

test "gemm: accumulation pattern (multiple adds to C)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 1, 1, 1 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 1, 1, 1 }, .row_major);
    defer B.deinit();

    // Start with C = [[1, 1], [1, 1]]
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 1, 1, 1 }, .row_major);
    defer C.deinit();

    // First: C = 1*A*B + 1*C = [[2, 2], [2, 2]] + [[1, 1], [1, 1]] = [[3, 3], [3, 3]]
    try gemm(f64, 1.0, A, B, 1.0, &C);

    for (0..4) |i| {
        try testing.expectApproxEqAbs(3.0, C.data[i], 1e-10);
    }

    // Second: C = 1*A*B + 1*C = [[2, 2], [2, 2]] + [[3, 3], [3, 3]] = [[5, 5], [5, 5]]
    try gemm(f64, 1.0, A, B, 1.0, &C);

    for (0..4) |i| {
        try testing.expectApproxEqAbs(5.0, C.data[i], 1e-10);
    }
}

test "gemm: various scalar combinations" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 0, 0, 2 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 3, 0, 0, 3 }, .row_major);
    defer B.deinit();

    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 4, 4, 4, 4 }, .row_major);
    defer C.deinit();

    // A*B = [[6, 0], [0, 6]]
    // C = 0.5*[[6, 0], [0, 6]] + 1.5*[[4, 4], [4, 4]]
    //   = [[3, 0], [0, 3]] + [[6, 6], [6, 6]]
    //   = [[9, 6], [6, 9]]
    try gemm(f64, 0.5, A, B, 1.5, &C);

    try testing.expectApproxEqAbs(9.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(6.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(6.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(9.0, C.data[3], 1e-10);
}

test "gemm: row vector (1xk) times column vector result (k x 1) = 1x1" {
    const allocator = testing.allocator;

    // A = [[1, 2, 3]]  (1×3 matrix, row vector)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 3 }, &[_]f64{ 1, 2, 3 }, .row_major);
    defer A.deinit();

    // B = [[4], [5], [6]]  (3×1 matrix, column vector)
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 1 }, &[_]f64{ 4, 5, 6 }, .row_major);
    defer B.deinit();

    // C = [[0]]  (1×1 matrix, scalar)
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 1 }, &[_]f64{0}, .row_major);
    defer C.deinit();

    // C = 1*([1*4 + 2*5 + 3*6]) + 0*0 = 1*32 = 32
    try gemm(f64, 1.0, A, B, 0.0, &C);

    try testing.expectApproxEqAbs(32.0, C.data[0], 1e-10);
}

test "gemm: column vector (mx1) times row vector (1xn) produces outer product" {
    const allocator = testing.allocator;

    // A = [[1], [2], [3]]  (3×1 matrix, column vector)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 1 }, &[_]f64{ 1, 2, 3 }, .row_major);
    defer A.deinit();

    // B = [[4, 5]]  (1×2 matrix, row vector)
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 2 }, &[_]f64{ 4, 5 }, .row_major);
    defer B.deinit();

    // C = [[0, 0], [0, 0], [0, 0]]  (3×2 matrix)
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 2 }, .row_major);
    defer C.deinit();

    // A*B = [[1*4, 1*5], [2*4, 2*5], [3*4, 3*5]] = [[4, 5], [8, 10], [12, 15]]
    try gemm(f64, 1.0, A, B, 0.0, &C);

    try testing.expectApproxEqAbs(4.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(5.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(8.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(10.0, C.data[3], 1e-10);
    try testing.expectApproxEqAbs(12.0, C.data[4], 1e-10);
    try testing.expectApproxEqAbs(15.0, C.data[5], 1e-10);
}

/// Compute trace of a square matrix
///
/// The trace is the sum of diagonal elements: trace(A) = Σ A[i,i]
///
/// Parameters:
/// - A: Square matrix (2D NDArray with shape[0] == shape[1])
///
/// Returns: Scalar sum of diagonal elements
///
/// Errors:
/// - error.DimensionMismatch if matrix is not square
///
/// Time: O(n) where n = dimension of square matrix
/// Space: O(1)
///
/// Example:
/// ```zig
/// var A = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{2, 2}, &[_]f64{1, 2, 3, 4}, .row_major);
/// defer A.deinit();
/// const tr = try trace(f64, A);  // 1 + 4 = 5
/// ```
pub fn trace(comptime T: type, A: NDArray(T, 2)) (NDArray(T, 2).Error)!T {
    // Validate square matrix
    if (A.shape[0] != A.shape[1]) {
        return error.DimensionMismatch;
    }

    const n = A.shape[0];
    var sum: T = 0;

    // Sum diagonal elements: A[i,i] for i in 0..n
    for (0..n) |i| {
        sum += A.data[i * n + i]; // Flat index: row*cols + col
    }

    return sum;
}

/// Compute determinant of a square matrix via LU decomposition
///
/// Computes det(A) using in-place LU factorization with partial pivoting.
/// The determinant is the product of diagonal elements with appropriate sign correction.
///
/// Parameters:
/// - A: Square matrix (2D NDArray with shape[0] == shape[1])
///
/// Returns: Scalar determinant
///
/// Errors:
/// - error.DimensionMismatch if matrix is not square
///
/// Time: O(n³) for LU decomposition
/// Space: O(n²) for the LU matrix copy (original A is not modified)
///
/// Example:
/// ```zig
/// var A = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{2, 2}, &[_]f64{1, 2, 3, 4}, .row_major);
/// defer A.deinit();
/// const d = try det(f64, A);  // 1*4 - 2*3 = -2
/// ```
pub fn det(comptime T: type, A: NDArray(T, 2)) (NDArray(T, 2).Error)!T {
    // Validate square matrix
    if (A.shape[0] != A.shape[1]) {
        return error.DimensionMismatch;
    }

    const n = A.shape[0];

    // Clone matrix for in-place LU decomposition (don't modify original)
    var LU = try A.clone();
    defer LU.deinit();

    var sign: T = 1.0; // Track row swaps for sign correction

    // Perform LU decomposition with partial pivoting
    for (0..n) |k| {
        // Find pivot (largest absolute value in column k, rows k..n)
        var max_idx = k;
        var max_val = @abs(LU.data[k * n + k]);
        for (k + 1..n) |i| {
            const val = @abs(LU.data[i * n + k]);
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }

        // If pivot is zero, matrix is singular → det = 0
        if (max_val == 0) {
            return 0;
        }

        // Swap rows k and max_idx if needed
        if (max_idx != k) {
            for (0..n) |j| {
                const temp = LU.data[k * n + j];
                LU.data[k * n + j] = LU.data[max_idx * n + j];
                LU.data[max_idx * n + j] = temp;
            }
            sign = -sign; // Row swap flips determinant sign
        }

        // Eliminate column k below diagonal
        for (k + 1..n) |i| {
            const factor = LU.data[i * n + k] / LU.data[k * n + k];
            LU.data[i * n + k] = factor; // Store L multiplier
            for (k + 1..n) |j| {
                LU.data[i * n + j] -= factor * LU.data[k * n + j];
            }
        }
    }

    // Determinant = sign * product of diagonal elements
    var product: T = sign;
    for (0..n) |i| {
        product *= LU.data[i * n + i];
    }

    return product;
}

// ============================================================================
// Tests for trace() and det()
// ============================================================================

// --- trace() tests ---

test "trace: 2x2 identity matrix" {
    const allocator = testing.allocator;

    // I = [[1, 0], [0, 1]]
    var I = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 0, 0, 1 }, .row_major);
    defer I.deinit();

    const tr = try trace(f64, I);
    try testing.expectApproxEqAbs(2.0, tr, 1e-10);
}

test "trace: 2x2 known result" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4]]
    // trace = 1 + 4 = 5
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    const tr = try trace(f64, A);
    try testing.expectApproxEqAbs(5.0, tr, 1e-10);
}

test "trace: 3x3 known result" {
    const allocator = testing.allocator;

    // A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    // trace = 1 + 5 + 9 = 15
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .row_major);
    defer A.deinit();

    const tr = try trace(f64, A);
    try testing.expectApproxEqAbs(15.0, tr, 1e-10);
}

test "trace: 3x3 identity matrix" {
    const allocator = testing.allocator;

    // I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    // trace = 3
    var I = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 }, .row_major);
    defer I.deinit();

    const tr = try trace(f64, I);
    try testing.expectApproxEqAbs(3.0, tr, 1e-10);
}

test "trace: zero matrix" {
    const allocator = testing.allocator;

    // Z = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    // trace = 0
    var Z = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer Z.deinit();

    const tr = try trace(f64, Z);
    try testing.expectApproxEqAbs(0.0, tr, 1e-10);
}

test "trace: diagonal matrix" {
    const allocator = testing.allocator;

    // D = [[2, 0, 0], [0, 3, 0], [0, 0, 5]]
    // trace = 2 + 3 + 5 = 10
    var D = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 2, 0, 0, 0, 3, 0, 0, 0, 5 }, .row_major);
    defer D.deinit();

    const tr = try trace(f64, D);
    try testing.expectApproxEqAbs(10.0, tr, 1e-10);
}

test "trace: negative values" {
    const allocator = testing.allocator;

    // A = [[-1, 2], [3, -4]]
    // trace = -1 + (-4) = -5
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ -1, 2, 3, -4 }, .row_major);
    defer A.deinit();

    const tr = try trace(f64, A);
    try testing.expectApproxEqAbs(-5.0, tr, 1e-10);
}

test "trace: 1x1 scalar matrix" {
    const allocator = testing.allocator;

    // A = [[7]]
    // trace = 7
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 1 }, &[_]f64{7}, .row_major);
    defer A.deinit();

    const tr = try trace(f64, A);
    try testing.expectApproxEqAbs(7.0, tr, 1e-10);
}

test "trace: f32 precision" {
    const allocator = testing.allocator;

    // A = [[1.5, 0], [0, 2.5]] (f32)
    // trace = 1.5 + 2.5 = 4.0
    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{ 1.5, 0, 0, 2.5 }, .row_major);
    defer A.deinit();

    const tr = try trace(f32, A);
    try testing.expectApproxEqAbs(4.0, tr, 1e-5);
}

test "trace: large matrix 10x10" {
    const allocator = testing.allocator;

    // Create 10x10 diagonal matrix with values 1, 2, ..., 10 on diagonal
    var data: [100]f64 = undefined;
    for (data) |*v| v.* = 0;
    for (0..10) |i| {
        data[i * 10 + i] = @floatFromInt(i + 1);
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 10, 10 }, &data, .row_major);
    defer A.deinit();

    const tr = try trace(f64, A);
    // trace = 1 + 2 + ... + 10 = 55
    try testing.expectApproxEqAbs(55.0, tr, 1e-10);
}

test "trace: additive property trace(A+B) = trace(A) + trace(B)" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4]], trace(A) = 5
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    // B = [[5, 6], [7, 8]], trace(B) = 13
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 6, 7, 8 }, .row_major);
    defer B.deinit();

    // A + B = [[6, 8], [10, 12]], trace = 18
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 6, 8, 10, 12 }, .row_major);
    defer C.deinit();

    const trA = try trace(f64, A);
    const trB = try trace(f64, B);
    const trC = try trace(f64, C);

    try testing.expectApproxEqAbs(trA + trB, trC, 1e-10);
}

test "trace: scalar multiplication property trace(cA) = c*trace(A)" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4]], trace(A) = 5
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    // 2A = [[2, 4], [6, 8]], trace = 10
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 4, 6, 8 }, .row_major);
    defer C.deinit();

    const trA = try trace(f64, A);
    const trC = try trace(f64, C);
    const c = 2.0;

    try testing.expectApproxEqAbs(c * trA, trC, 1e-10);
}

test "trace: non-square matrix error (2x3)" {
    const allocator = testing.allocator;

    // A = [[1, 2, 3], [4, 5, 6]]  (2×3 non-square)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer A.deinit();

    const result = trace(f64, A);
    try testing.expectError(error.DimensionMismatch, result);
}

test "trace: non-square matrix error (3x2)" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4], [5, 6]]  (3×2 non-square)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer A.deinit();

    const result = trace(f64, A);
    try testing.expectError(error.DimensionMismatch, result);
}

test "trace: fractional values" {
    const allocator = testing.allocator;

    // A = [[0.5, 1.2], [1.5, 0.3]]
    // trace = 0.5 + 0.3 = 0.8
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 0.5, 1.2, 1.5, 0.3 }, .row_major);
    defer A.deinit();

    const tr = try trace(f64, A);
    try testing.expectApproxEqAbs(0.8, tr, 1e-10);
}

// --- det() tests ---

test "det: 2x2 basic known result" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4]]
    // det = 1*4 - 2*3 = -2
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    const d = try det(f64, A);
    try testing.expectApproxEqAbs(-2.0, d, 1e-10);
}

test "det: 2x2 identity matrix" {
    const allocator = testing.allocator;

    // I = [[1, 0], [0, 1]]
    // det = 1
    var I = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 0, 0, 1 }, .row_major);
    defer I.deinit();

    const d = try det(f64, I);
    try testing.expectApproxEqAbs(1.0, d, 1e-10);
}

test "det: 3x3 identity matrix" {
    const allocator = testing.allocator;

    // I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    // det = 1
    var I = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 }, .row_major);
    defer I.deinit();

    const d = try det(f64, I);
    try testing.expectApproxEqAbs(1.0, d, 1e-10);
}

test "det: 3x3 known result" {
    const allocator = testing.allocator;

    // A = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
    // det = 1*(1*0 - 4*6) - 2*(0*0 - 4*5) + 3*(0*6 - 1*5)
    //     = 1*(-24) - 2*(-20) + 3*(-5)
    //     = -24 + 40 - 15 = 1
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 0, 1, 4, 5, 6, 0 }, .row_major);
    defer A.deinit();

    const d = try det(f64, A);
    try testing.expectApproxEqAbs(1.0, d, 1e-10);
}

test "det: zero matrix" {
    const allocator = testing.allocator;

    // Z = [[0, 0], [0, 0]]
    // det = 0
    var Z = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer Z.deinit();

    const d = try det(f64, Z);
    try testing.expectApproxEqAbs(0.0, d, 1e-10);
}

test "det: singular matrix (2x2)" {
    const allocator = testing.allocator;

    // A = [[1, 2], [2, 4]]  (singular, second row = 2 * first row)
    // det = 1*4 - 2*2 = 0
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 2, 4 }, .row_major);
    defer A.deinit();

    const d = try det(f64, A);
    try testing.expectApproxEqAbs(0.0, d, 1e-10);
}

test "det: diagonal matrix" {
    const allocator = testing.allocator;

    // D = [[2, 0, 0], [0, 3, 0], [0, 0, 5]]
    // det = 2*3*5 = 30
    var D = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 2, 0, 0, 0, 3, 0, 0, 0, 5 }, .row_major);
    defer D.deinit();

    const d = try det(f64, D);
    try testing.expectApproxEqAbs(30.0, d, 1e-10);
}

test "det: upper triangular matrix" {
    const allocator = testing.allocator;

    // U = [[1, 2, 3], [0, 4, 5], [0, 0, 6]]
    // det = 1*4*6 = 24
    var U = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 0, 4, 5, 0, 0, 6 }, .row_major);
    defer U.deinit();

    const d = try det(f64, U);
    try testing.expectApproxEqAbs(24.0, d, 1e-10);
}

test "det: lower triangular matrix" {
    const allocator = testing.allocator;

    // L = [[2, 0, 0], [3, 4, 0], [5, 6, 7]]
    // det = 2*4*7 = 56
    var L = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 2, 0, 0, 3, 4, 0, 5, 6, 7 }, .row_major);
    defer L.deinit();

    const d = try det(f64, L);
    try testing.expectApproxEqAbs(56.0, d, 1e-10);
}

test "det: 1x1 scalar matrix" {
    const allocator = testing.allocator;

    // A = [[5]]
    // det = 5
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 1 }, &[_]f64{5}, .row_major);
    defer A.deinit();

    const d = try det(f64, A);
    try testing.expectApproxEqAbs(5.0, d, 1e-10);
}

test "det: negative determinant" {
    const allocator = testing.allocator;

    // A = [[0, 1], [1, 0]]  (swap rows of identity)
    // det = 0*0 - 1*1 = -1
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 0, 1, 1, 0 }, .row_major);
    defer A.deinit();

    const d = try det(f64, A);
    try testing.expectApproxEqAbs(-1.0, d, 1e-10);
}

test "det: f32 precision" {
    const allocator = testing.allocator;

    // A = [[2, 1], [1, 3]] (f32)
    // det = 2*3 - 1*1 = 5
    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{ 2, 1, 1, 3 }, .row_major);
    defer A.deinit();

    const d = try det(f32, A);
    try testing.expectApproxEqAbs(5.0, d, 1e-5);
}

test "det: multiplicative property det(AB) = det(A)*det(B)" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4]], det(A) = -2
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    // B = [[5, 6], [7, 8]], det(B) = -2
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 6, 7, 8 }, .row_major);
    defer B.deinit();

    // A*B = [[19, 22], [43, 50]], det = 19*50 - 22*43 = 950 - 946 = 4
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 19, 22, 43, 50 }, .row_major);
    defer C.deinit();

    const detA = try det(f64, A);
    const detB = try det(f64, B);
    const detC = try det(f64, C);

    try testing.expectApproxEqAbs(detA * detB, detC, 1e-10);
}

test "det: non-square matrix error (2x3)" {
    const allocator = testing.allocator;

    // A = [[1, 2, 3], [4, 5, 6]]  (2×3 non-square)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer A.deinit();

    const result = det(f64, A);
    try testing.expectError(error.DimensionMismatch, result);
}

test "det: non-square matrix error (3x2)" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4], [5, 6]]  (3×2 non-square)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer A.deinit();

    const result = det(f64, A);
    try testing.expectError(error.DimensionMismatch, result);
}

test "det: large 5x5 matrix" {
    const allocator = testing.allocator;

    // A = [[2, 0, 0, 0, 0],
    //      [0, 3, 0, 0, 0],
    //      [0, 0, 4, 0, 0],
    //      [0, 0, 0, 5, 0],
    //      [0, 0, 0, 0, 6]]
    // det = 2*3*4*5*6 = 720
    var data: [25]f64 = undefined;
    for (data) |*v| v.* = 0;
    data[0] = 2;
    data[6] = 3;
    data[12] = 4;
    data[18] = 5;
    data[24] = 6;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 5, 5 }, &data, .row_major);
    defer A.deinit();

    const d = try det(f64, A);
    try testing.expectApproxEqAbs(720.0, d, 1e-10);
}

test "det: fractional values" {
    const allocator = testing.allocator;

    // A = [[0.5, 0.2], [0.3, 0.4]]
    // det = 0.5*0.4 - 0.2*0.3 = 0.2 - 0.06 = 0.14
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 0.5, 0.2, 0.3, 0.4 }, .row_major);
    defer A.deinit();

    const d = try det(f64, A);
    try testing.expectApproxEqAbs(0.14, d, 1e-10);
}

test "det: scaling property det(cA) = c^n * det(A)" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4]], det(A) = -2
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    // 2A = [[2, 4], [6, 8]], det = 2^2 * (-2) = -8
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 4, 6, 8 }, .row_major);
    defer C.deinit();

    const detA = try det(f64, A);
    const detC = try det(f64, C);
    const c = 2.0;
    const n = 2.0;

    try testing.expectApproxEqAbs(std.math.pow(f64, c, n) * detA, detC, 1e-10);
}

// ============================================================================
// Vector and Matrix Norm Functions
// ============================================================================

/// L1 norm of a vector: sum of absolute values
///
/// Computes: ||x||₁ = Σ|xᵢ|
///
/// Parameters:
/// - x: Vector (1D NDArray)
///
/// Returns: Non-negative scalar, sum of absolute values
///
/// Time: O(n) where n = vector length
/// Space: O(1)
pub fn norm1(comptime T: type, x: NDArray(T, 1)) (NDArray(T, 1).Error)!T {
    // L1 norm is sum of absolute values - reuse BLAS asum
    return asum(T, x);
}

/// L2 norm of a vector: Euclidean norm (magnitude)
///
/// Computes: ||x||₂ = √(Σxᵢ²)
///
/// Parameters:
/// - x: Vector (1D NDArray)
///
/// Returns: Non-negative scalar, Euclidean norm
///
/// Time: O(n) where n = vector length
/// Space: O(1)
///
/// Note: This reuses the existing nrm2() BLAS function
pub fn norm2(comptime T: type, x: NDArray(T, 1)) (NDArray(T, 1).Error)!T {
    // L2 norm is Euclidean norm - reuse BLAS nrm2
    return nrm2(T, x);
}

/// L∞ norm of a vector: maximum absolute value
///
/// Computes: ||x||∞ = max|xᵢ|
///
/// Parameters:
/// - x: Vector (1D NDArray)
///
/// Returns: Non-negative scalar, maximum absolute value
///
/// Time: O(n) where n = vector length
/// Space: O(1)
pub fn normInf(comptime T: type, x: NDArray(T, 1)) (NDArray(T, 1).Error)!T {
    // L∞ norm is maximum absolute value
    const n = x.shape[0];
    if (n == 0) return 0;

    var max_val: T = @abs(x.data[0]);
    for (1..n) |i| {
        const abs_val = @abs(x.data[i]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }
    return max_val;
}

/// Frobenius norm of a matrix: L2 norm treating matrix as vector
///
/// Computes: ||A||_F = √(ΣᵢⱼAᵢⱼ²)
///
/// Parameters:
/// - A: Matrix (2D NDArray, any shape m×n)
///
/// Returns: Non-negative scalar, Frobenius norm
///
/// Time: O(m*n)
/// Space: O(1)
pub fn normFrobenius(comptime T: type, A: NDArray(T, 2)) (NDArray(T, 2).Error)!T {
    // Frobenius norm: sqrt(sum of squares of all elements)
    const m = A.shape[0];
    const n = A.shape[1];
    const total = m * n;

    if (total == 0) return 0;

    var sum_sq: T = 0;
    for (0..total) |i| {
        sum_sq += A.data[i] * A.data[i];
    }
    return @sqrt(sum_sq);
}

// ============================================================================
// Tests for norm1(), norm2(), normInf(), normFrobenius()
// ============================================================================

// --- norm1() tests ---

test "norm1: vector [1, -2, 3, -4]" {
    const allocator = testing.allocator;

    // x = [1, -2, 3, -4]
    // ||x||₁ = |1| + |-2| + |3| + |-4| = 1 + 2 + 3 + 4 = 10
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1, -2, 3, -4 }, .row_major);
    defer x.deinit();

    const n = try norm1(f64, x);
    try testing.expectApproxEqAbs(10.0, n, 1e-10);
}

test "norm1: zero vector" {
    const allocator = testing.allocator;

    // x = [0, 0, 0]
    // ||x||₁ = 0
    var x = try NDArray(f64, 1).zeros(allocator, &[_]usize{3}, .row_major);
    defer x.deinit();

    const n = try norm1(f64, x);
    try testing.expectApproxEqAbs(0.0, n, 1e-10);
}

test "norm1: single element" {
    const allocator = testing.allocator;

    // x = [5]
    // ||x||₁ = |5| = 5
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{5}, .row_major);
    defer x.deinit();

    const n = try norm1(f64, x);
    try testing.expectApproxEqAbs(5.0, n, 1e-10);
}

test "norm1: negative single element" {
    const allocator = testing.allocator;

    // x = [-7]
    // ||x||₁ = |-7| = 7
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{-7}, .row_major);
    defer x.deinit();

    const n = try norm1(f64, x);
    try testing.expectApproxEqAbs(7.0, n, 1e-10);
}

test "norm1: all positive values" {
    const allocator = testing.allocator;

    // x = [1.5, 2.5, 3.5]
    // ||x||₁ = 1.5 + 2.5 + 3.5 = 7.5
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1.5, 2.5, 3.5 }, .row_major);
    defer x.deinit();

    const n = try norm1(f64, x);
    try testing.expectApproxEqAbs(7.5, n, 1e-10);
}

test "norm1: scaling property ||cx||₁ = |c| * ||x||₁" {
    const allocator = testing.allocator;

    // x = [1, 2, 3]
    // ||x||₁ = 6
    // 2x = [2, 4, 6]
    // ||2x||₁ = 12 = 2 * 6
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    var cx = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 2, 4, 6 }, .row_major);
    defer cx.deinit();

    const n1 = try norm1(f64, x);
    const n2 = try norm1(f64, cx);

    try testing.expectApproxEqAbs(2.0 * n1, n2, 1e-10);
}

test "norm1: f32 precision" {
    const allocator = testing.allocator;

    // x = [1.5, -2.5, 3.5] (f32)
    // ||x||₁ = 1.5 + 2.5 + 3.5 = 7.5
    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{3}, &[_]f32{ 1.5, -2.5, 3.5 }, .row_major);
    defer x.deinit();

    const n = try norm1(f32, x);
    try testing.expectApproxEqAbs(7.5, n, 1e-5);
}

test "norm1: large vector n=100" {
    const allocator = testing.allocator;

    // x = [1, 1, 1, ..., 1] (100 ones)
    // ||x||₁ = 100
    var data: [100]f64 = undefined;
    for (data) |*v| v.* = 1.0;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{100}, &data, .row_major);
    defer x.deinit();

    const n = try norm1(f64, x);
    try testing.expectApproxEqAbs(100.0, n, 1e-10);
}

// --- norm2() tests ---

test "norm2: vector [3, 4]" {
    const allocator = testing.allocator;

    // x = [3, 4]
    // ||x||₂ = √(9 + 16) = √25 = 5
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 3, 4 }, .row_major);
    defer x.deinit();

    const n = try norm2(f64, x);
    try testing.expectApproxEqAbs(5.0, n, 1e-10);
}

test "norm2: zero vector" {
    const allocator = testing.allocator;

    // x = [0, 0, 0]
    // ||x||₂ = 0
    var x = try NDArray(f64, 1).zeros(allocator, &[_]usize{3}, .row_major);
    defer x.deinit();

    const n = try norm2(f64, x);
    try testing.expectApproxEqAbs(0.0, n, 1e-10);
}

test "norm2: unit vector [1, 0, 0]" {
    const allocator = testing.allocator;

    // x = [1, 0, 0]
    // ||x||₂ = √(1 + 0 + 0) = 1
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 0, 0 }, .row_major);
    defer x.deinit();

    const n = try norm2(f64, x);
    try testing.expectApproxEqAbs(1.0, n, 1e-10);
}

test "norm2: single element" {
    const allocator = testing.allocator;

    // x = [7]
    // ||x||₂ = √49 = 7
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{7}, .row_major);
    defer x.deinit();

    const n = try norm2(f64, x);
    try testing.expectApproxEqAbs(7.0, n, 1e-10);
}

test "norm2: [1, 1, 1]" {
    const allocator = testing.allocator;

    // x = [1, 1, 1]
    // ||x||₂ = √(1 + 1 + 1) = √3 ≈ 1.732050808...
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 1, 1 }, .row_major);
    defer x.deinit();

    const n = try norm2(f64, x);
    try testing.expectApproxEqAbs(std.math.sqrt(3.0), n, 1e-10);
}

test "norm2: scaling property ||cx||₂ = |c| * ||x||₂" {
    const allocator = testing.allocator;

    // x = [3, 4]
    // ||x||₂ = 5
    // 2x = [6, 8]
    // ||2x||₂ = √(36 + 64) = √100 = 10 = 2 * 5
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 3, 4 }, .row_major);
    defer x.deinit();

    var cx = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 6, 8 }, .row_major);
    defer cx.deinit();

    const n1 = try norm2(f64, x);
    const n2 = try norm2(f64, cx);

    try testing.expectApproxEqAbs(2.0 * n1, n2, 1e-10);
}

test "norm2: f32 precision" {
    const allocator = testing.allocator;

    // x = [1.0, 2.0, 2.0] (f32)
    // ||x||₂ = √(1 + 4 + 4) = √9 = 3
    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{3}, &[_]f32{ 1.0, 2.0, 2.0 }, .row_major);
    defer x.deinit();

    const n = try norm2(f32, x);
    try testing.expectApproxEqAbs(3.0, n, 1e-5);
}

test "norm2: large vector n=100" {
    const allocator = testing.allocator;

    // x = [1, 1, 1, ..., 1] (100 ones)
    // ||x||₂ = √100 = 10
    var data: [100]f64 = undefined;
    for (data) |*v| v.* = 1.0;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{100}, &data, .row_major);
    defer x.deinit();

    const n = try norm2(f64, x);
    try testing.expectApproxEqAbs(10.0, n, 1e-10);
}

// --- normInf() tests ---

test "normInf: vector [1, -5, 3]" {
    const allocator = testing.allocator;

    // x = [1, -5, 3]
    // ||x||∞ = max(|1|, |-5|, |3|) = 5
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, -5, 3 }, .row_major);
    defer x.deinit();

    const n = try normInf(f64, x);
    try testing.expectApproxEqAbs(5.0, n, 1e-10);
}

test "normInf: zero vector" {
    const allocator = testing.allocator;

    // x = [0, 0, 0]
    // ||x||∞ = 0
    var x = try NDArray(f64, 1).zeros(allocator, &[_]usize{3}, .row_major);
    defer x.deinit();

    const n = try normInf(f64, x);
    try testing.expectApproxEqAbs(0.0, n, 1e-10);
}

test "normInf: single element" {
    const allocator = testing.allocator;

    // x = [7]
    // ||x||∞ = |7| = 7
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{7}, .row_major);
    defer x.deinit();

    const n = try normInf(f64, x);
    try testing.expectApproxEqAbs(7.0, n, 1e-10);
}

test "normInf: negative single element" {
    const allocator = testing.allocator;

    // x = [-9]
    // ||x||∞ = |-9| = 9
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{-9}, .row_major);
    defer x.deinit();

    const n = try normInf(f64, x);
    try testing.expectApproxEqAbs(9.0, n, 1e-10);
}

test "normInf: maximum is negative" {
    const allocator = testing.allocator;

    // x = [-10, -3, -7]
    // ||x||∞ = max(|-10|, |-3|, |-7|) = 10
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ -10, -3, -7 }, .row_major);
    defer x.deinit();

    const n = try normInf(f64, x);
    try testing.expectApproxEqAbs(10.0, n, 1e-10);
}

test "normInf: scaling property ||cx||∞ = |c| * ||x||∞" {
    const allocator = testing.allocator;

    // x = [1, 3, 2]
    // ||x||∞ = 3
    // 2x = [2, 6, 4]
    // ||2x||∞ = 6 = 2 * 3
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 3, 2 }, .row_major);
    defer x.deinit();

    var cx = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 2, 6, 4 }, .row_major);
    defer cx.deinit();

    const n1 = try normInf(f64, x);
    const n2 = try normInf(f64, cx);

    try testing.expectApproxEqAbs(2.0 * n1, n2, 1e-10);
}

test "normInf: f32 precision" {
    const allocator = testing.allocator;

    // x = [1.5, -4.2, 3.1] (f32)
    // ||x||∞ = max(1.5, 4.2, 3.1) = 4.2
    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{3}, &[_]f32{ 1.5, -4.2, 3.1 }, .row_major);
    defer x.deinit();

    const n = try normInf(f32, x);
    try testing.expectApproxEqAbs(4.2, n, 1e-5);
}

test "normInf: large vector n=100" {
    const allocator = testing.allocator;

    // x = [1, 2, 3, ..., 100]
    // ||x||∞ = 100
    var data: [100]f64 = undefined;
    for (data, 0..) |*v, i| v.* = @floatFromInt(i + 1);

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{100}, &data, .row_major);
    defer x.deinit();

    const n = try normInf(f64, x);
    try testing.expectApproxEqAbs(100.0, n, 1e-10);
}

// --- normFrobenius() tests ---

test "normFrobenius: 2x2 matrix [[1, 2], [3, 4]]" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4]]
    // ||A||_F = √(1² + 2² + 3² + 4²) = √(1 + 4 + 9 + 16) = √30
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    const n = try normFrobenius(f64, A);
    try testing.expectApproxEqAbs(std.math.sqrt(30.0), n, 1e-10);
}

test "normFrobenius: 2x3 matrix [[1, 0, 0], [0, 1, 0]]" {
    const allocator = testing.allocator;

    // A = [[1, 0, 0], [0, 1, 0]]
    // ||A||_F = √(1² + 0² + 0² + 0² + 1² + 0²) = √2
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 0, 0, 0, 1, 0 }, .row_major);
    defer A.deinit();

    const n = try normFrobenius(f64, A);
    try testing.expectApproxEqAbs(std.math.sqrt(2.0), n, 1e-10);
}

test "normFrobenius: identity matrix 3x3" {
    const allocator = testing.allocator;

    // I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    // ||I||_F = √(1 + 1 + 1) = √3
    var I = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 }, .row_major);
    defer I.deinit();

    const n = try normFrobenius(f64, I);
    try testing.expectApproxEqAbs(std.math.sqrt(3.0), n, 1e-10);
}

test "normFrobenius: zero matrix" {
    const allocator = testing.allocator;

    // Z = [[0, 0], [0, 0]]
    // ||Z||_F = 0
    var Z = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer Z.deinit();

    const n = try normFrobenius(f64, Z);
    try testing.expectApproxEqAbs(0.0, n, 1e-10);
}

test "normFrobenius: single element matrix [[5]]" {
    const allocator = testing.allocator;

    // A = [[5]]
    // ||A||_F = √25 = 5
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 1 }, &[_]f64{5}, .row_major);
    defer A.deinit();

    const n = try normFrobenius(f64, A);
    try testing.expectApproxEqAbs(5.0, n, 1e-10);
}

test "normFrobenius: negative values [[1, -2], [-3, 4]]" {
    const allocator = testing.allocator;

    // A = [[1, -2], [-3, 4]]
    // ||A||_F = √(1² + (-2)² + (-3)² + 4²) = √(1 + 4 + 9 + 16) = √30
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, -2, -3, 4 }, .row_major);
    defer A.deinit();

    const n = try normFrobenius(f64, A);
    try testing.expectApproxEqAbs(std.math.sqrt(30.0), n, 1e-10);
}

test "normFrobenius: scaling property ||cA||_F = |c| * ||A||_F" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4]]
    // ||A||_F = √30
    // 2A = [[2, 4], [6, 8]]
    // ||2A||_F = √(4 + 16 + 36 + 64) = √120 = 2√30
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    var CA = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 4, 6, 8 }, .row_major);
    defer CA.deinit();

    const n1 = try normFrobenius(f64, A);
    const n2 = try normFrobenius(f64, CA);

    try testing.expectApproxEqAbs(2.0 * n1, n2, 1e-10);
}

test "normFrobenius: f32 precision" {
    const allocator = testing.allocator;

    // A = [[1.5, 2.5], [3.5, 4.5]] (f32)
    // ||A||_F = √(2.25 + 6.25 + 12.25 + 20.25) = √41
    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{ 1.5, 2.5, 3.5, 4.5 }, .row_major);
    defer A.deinit();

    const n = try normFrobenius(f32, A);
    try testing.expectApproxEqAbs(std.math.sqrt(41.0), n, 1e-4);
}

test "normFrobenius: rectangular matrix 3x4" {
    const allocator = testing.allocator;

    // A = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    // ||A||_F = √(1 + 1 + 1) = √3
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 4 }, &[_]f64{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 }, .row_major);
    defer A.deinit();

    const n = try normFrobenius(f64, A);
    try testing.expectApproxEqAbs(std.math.sqrt(3.0), n, 1e-10);
}

test "normFrobenius: 10x10 diagonal matrix" {
    const allocator = testing.allocator;

    // D = diag(1, 2, 3, ..., 10)
    // ||D||_F = √(1² + 2² + 3² + ... + 10²) = √(1+4+9+16+25+36+49+64+81+100) = √385
    var data: [100]f64 = undefined;
    for (data) |*v| v.* = 0;
    for (0..10) |i| {
        data[i * 10 + i] = @floatFromInt(i + 1);
    }

    var D = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 10, 10 }, &data, .row_major);
    defer D.deinit();

    const n = try normFrobenius(f64, D);
    // Sum of squares: 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81 + 100 = 385
    try testing.expectApproxEqAbs(std.math.sqrt(385.0), n, 1e-10);
}

test "normFrobenius: equivalence with frobenius as vector norm" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4]] treated as vector [1, 2, 3, 4]
    // ||[1, 2, 3, 4]||₂ = √30
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer x.deinit();

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    const n_vec = try norm2(f64, x);
    const n_frob = try normFrobenius(f64, A);

    try testing.expectApproxEqAbs(n_vec, n_frob, 1e-10);
}
