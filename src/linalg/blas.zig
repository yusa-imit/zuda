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
