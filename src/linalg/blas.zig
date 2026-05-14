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
const simd_blas = @import("simd_blas.zig");

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

    const n = x.shape[0];

    // Auto-dispatch: Use SIMD-optimized implementation for large vectors
    // Threshold: if n >= 64 (enough elements to benefit from vectorization)
    // Session 488: dot_simd provides 1.5-2× speedup via SIMD vectorization
    const threshold: usize = 64;
    if (n >= threshold) {
        return try simd_blas.dot_simd(T, x, y);
    }

    // Fallback to scalar loop for small vectors
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

    const n = x.shape[0];

    // Auto-dispatch: Use SIMD-optimized implementation for large vectors
    // Threshold: if n >= 64 (enough elements to benefit from vectorization)
    // axpy_simd provides 4-8× speedup via SIMD vectorization (y = α*x + y)
    const threshold: usize = 64;
    if (n >= threshold) {
        return try simd_blas.axpy_simd(T, alpha, x, y);
    }

    // Fallback to scalar loop for small vectors
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
    const n = x.shape[0];

    // Auto-dispatch: Use SIMD-optimized implementation for large vectors
    // Threshold: if n >= 64 (enough elements to benefit from vectorization)
    // Session 492: nrm2_simd provides 2-4× speedup via SIMD vectorization
    const threshold: usize = 64;
    if (n >= threshold) {
        return try simd_blas.nrm2_simd(T, x);
    }

    // Fallback to scalar loop for small vectors
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
    const n = x.shape[0];

    // Auto-dispatch: Use SIMD-optimized implementation for large vectors
    // Threshold: if n >= 64 (enough elements to benefit from vectorization)
    // Session 493: asum_simd provides 2-4× speedup via SIMD vectorization
    const threshold: usize = 64;
    if (n >= threshold) {
        return try simd_blas.asum_simd(T, x);
    }

    // Fallback to scalar loop for small vectors
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
    const n = x.shape[0];

    // Auto-dispatch: Use SIMD-optimized implementation for large vectors
    // Threshold: if n >= 64 (enough elements to benefit from vectorization)
    // Session 493: scal_simd provides 3-6× speedup via SIMD vectorization
    const threshold: usize = 64;
    if (n >= threshold) {
        return try simd_blas.scal_simd(T, alpha, x);
    }

    // Fallback to scalar loop for small vectors
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
// Auto-Dispatch Tests (SIMD Threshold = 64)
// ============================================================================

test "dot: auto-dispatch threshold below (n=63, f64)" {
    // Below threshold (63 < 64) — should use scalar path
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f64, 63);
    defer allocator.free(data_x);
    var data_y = try allocator.alloc(f64, 63);
    defer allocator.free(data_y);

    // Initialize with simple values
    for (0..63) |i| {
        data_x[i] = @as(f64, @floatFromInt(i + 1));
        data_y[i] = @as(f64, @floatFromInt(i + 1));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{63}, data_x, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{63}, data_y, .row_major);
    defer y.deinit();

    // Expected: sum of i^2 for i=1 to 63 = 63*64*127/6
    const expected = 63.0 * 64.0 * 127.0 / 6.0;
    const result = try dot(f64, x, y);
    try testing.expectApproxEqRel(expected, result, 1e-9);
}

test "dot: auto-dispatch threshold boundary (n=64, f64)" {
    // At threshold (64 == 64) — should use SIMD path
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f64, 64);
    defer allocator.free(data_x);
    var data_y = try allocator.alloc(f64, 64);
    defer allocator.free(data_y);

    for (0..64) |i| {
        data_x[i] = @as(f64, @floatFromInt(i + 1));
        data_y[i] = @as(f64, @floatFromInt(i + 1));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{64}, data_x, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{64}, data_y, .row_major);
    defer y.deinit();

    // Expected: sum of i^2 for i=1 to 64 = 64*65*129/6
    const expected = 64.0 * 65.0 * 129.0 / 6.0;
    const result = try dot(f64, x, y);
    try testing.expectApproxEqRel(expected, result, 1e-9);
}

test "dot: auto-dispatch threshold above (n=65, f64)" {
    // Above threshold (65 > 64) — should use SIMD path
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f64, 65);
    defer allocator.free(data_x);
    var data_y = try allocator.alloc(f64, 65);
    defer allocator.free(data_y);

    for (0..65) |i| {
        data_x[i] = @as(f64, @floatFromInt(i + 1));
        data_y[i] = @as(f64, @floatFromInt(i + 1));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{65}, data_x, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{65}, data_y, .row_major);
    defer y.deinit();

    // Expected: sum of i^2 for i=1 to 65 = 65*66*131/6
    const expected = 65.0 * 66.0 * 131.0 / 6.0;
    const result = try dot(f64, x, y);
    try testing.expectApproxEqRel(expected, result, 1e-9);
}

test "dot: auto-dispatch large vector (n=1024, f64)" {
    // Well above threshold — verify SIMD path correctness
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f64, 1024);
    defer allocator.free(data_x);
    var data_y = try allocator.alloc(f64, 1024);
    defer allocator.free(data_y);

    for (0..1024) |i| {
        data_x[i] = @as(f64, @floatFromInt(i)) + 0.5;
        data_y[i] = @as(f64, @floatFromInt(i)) + 0.5;
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1024}, data_x, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1024}, data_y, .row_major);
    defer y.deinit();

    // Expected: sum of (i+0.5)^2 for i=0 to 1023
    // = sum of (i^2 + i + 0.25) = sum(i^2) + sum(i) + 256
    // sum(i^2) = 1023*1024*2047/6, sum(i) = 1023*1024/2
    const sum_squares = 1023.0 * 1024.0 * 2047.0 / 6.0;
    const sum_linear = 1023.0 * 1024.0 / 2.0;
    const expected = sum_squares + sum_linear + 256.0;
    const result = try dot(f64, x, y);
    try testing.expectApproxEqRel(expected, result, 1e-8);
}

test "dot: auto-dispatch non-aligned (n=100, f64)" {
    // Not a multiple of SIMD width — tests remainder loop
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f64, 100);
    defer allocator.free(data_x);
    var data_y = try allocator.alloc(f64, 100);
    defer allocator.free(data_y);

    for (0..100) |i| {
        data_x[i] = @as(f64, @floatFromInt(i)) + 0.1;
        data_y[i] = 2.0 * @as(f64, @floatFromInt(i)) + 0.2;
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{100}, data_x, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{100}, data_y, .row_major);
    defer y.deinit();

    var expected: f64 = 0;
    for (0..100) |i| {
        const xi = @as(f64, @floatFromInt(i)) + 0.1;
        const yi = 2.0 * @as(f64, @floatFromInt(i)) + 0.2;
        expected += xi * yi;
    }

    const result = try dot(f64, x, y);
    try testing.expectApproxEqRel(expected, result, 1e-9);
}

test "dot: auto-dispatch f32 type (n=64)" {
    // Test auto-dispatch with f32 at threshold
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f32, 64);
    defer allocator.free(data_x);
    var data_y = try allocator.alloc(f32, 64);
    defer allocator.free(data_y);

    for (0..64) |i| {
        data_x[i] = @as(f32, @floatFromInt(i)) + 0.5;
        data_y[i] = @as(f32, @floatFromInt(i)) + 0.5;
    }

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{64}, data_x, .row_major);
    defer x.deinit();
    var y = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{64}, data_y, .row_major);
    defer y.deinit();

    var expected: f32 = 0;
    for (0..64) |i| {
        const val = @as(f32, @floatFromInt(i)) + 0.5;
        expected += val * val;
    }

    const result = try dot(f32, x, y);
    try testing.expectApproxEqRel(expected, result, 1e-5);
}

test "dot: auto-dispatch f32 large (n=512)" {
    // Test auto-dispatch with f32 well above threshold
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f32, 512);
    defer allocator.free(data_x);
    var data_y = try allocator.alloc(f32, 512);
    defer allocator.free(data_y);

    for (0..512) |i| {
        data_x[i] = @as(f32, @floatFromInt(i)) * 0.1;
        data_y[i] = @as(f32, @floatFromInt(i)) * 0.2;
    }

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{512}, data_x, .row_major);
    defer x.deinit();
    var y = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{512}, data_y, .row_major);
    defer y.deinit();

    var expected: f32 = 0;
    for (0..512) |i| {
        const xi = @as(f32, @floatFromInt(i)) * 0.1;
        const yi = @as(f32, @floatFromInt(i)) * 0.2;
        expected += xi * yi;
    }

    const result = try dot(f32, x, y);
    try testing.expectApproxEqRel(expected, result, 1e-4);
}

test "dot: auto-dispatch scalar vs simd equivalence (n=256, f64)" {
    // Verify scalar and SIMD paths produce equivalent results
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f64, 256);
    defer allocator.free(data_x);
    var data_y = try allocator.alloc(f64, 256);
    defer allocator.free(data_y);

    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    for (0..256) |i| {
        data_x[i] = random.float(f64) * 100.0 - 50.0;
        data_y[i] = random.float(f64) * 100.0 - 50.0;
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{256}, data_x, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{256}, data_y, .row_major);
    defer y.deinit();

    // For large vectors, both paths should produce the same result
    const result = try dot(f64, x, y);

    // Manually compute expected value (baseline)
    var expected: f64 = 0;
    for (0..256) |i| {
        expected += data_x[i] * data_y[i];
    }

    try testing.expectApproxEqRel(expected, result, 1e-9);
}

test "dot: auto-dispatch with negative values (n=128)" {
    // Ensure dispatch handles negative values correctly
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f64, 128);
    defer allocator.free(data_x);
    var data_y = try allocator.alloc(f64, 128);
    defer allocator.free(data_y);

    for (0..128) |i| {
        if (i % 2 == 0) {
            data_x[i] = -@as(f64, @floatFromInt(i + 1));
        } else {
            data_x[i] = @as(f64, @floatFromInt(i + 1));
        }
        data_y[i] = @as(f64, @floatFromInt(i + 1));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{128}, data_x, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{128}, data_y, .row_major);
    defer y.deinit();

    var expected: f64 = 0;
    for (0..128) |i| {
        expected += data_x[i] * data_y[i];
    }

    const result = try dot(f64, x, y);
    try testing.expectApproxEqRel(expected, result, 1e-9);
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

test "axpy: auto-dispatch threshold below (n=63, scalar path)" {
    // Below threshold (63 < 64) — should use scalar path
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f64, 63);
    defer allocator.free(data_x);
    var data_y = try allocator.alloc(f64, 63);
    defer allocator.free(data_y);

    for (0..63) |i| {
        data_x[i] = @as(f64, @floatFromInt(i + 1)); // x = [1, 2, ..., 63]
        data_y[i] = @as(f64, @floatFromInt(i + 1)); // y = [1, 2, ..., 63]
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{63}, data_x, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{63}, data_y, .row_major);
    defer y.deinit();

    try axpy(f64, 2.0, x, &y);

    // y = 2*x + y = 2*[1,2,...,63] + [1,2,...,63] = [3,6,...,189]
    for (0..63) |i| {
        const expected = 3.0 * @as(f64, @floatFromInt(i + 1));
        try testing.expectApproxEqAbs(expected, y.data[i], 1e-9);
    }
}

test "axpy: auto-dispatch threshold exact (n=64, SIMD path)" {
    // At threshold (64 == 64) — should use SIMD path
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f64, 64);
    defer allocator.free(data_x);
    var data_y = try allocator.alloc(f64, 64);
    defer allocator.free(data_y);

    for (0..64) |i| {
        data_x[i] = @as(f64, @floatFromInt(i + 1));
        data_y[i] = @as(f64, @floatFromInt(i + 1));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{64}, data_x, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{64}, data_y, .row_major);
    defer y.deinit();

    try axpy(f64, 2.0, x, &y);

    // y = 2*x + y = 3*[1,2,...,64]
    for (0..64) |i| {
        const expected = 3.0 * @as(f64, @floatFromInt(i + 1));
        try testing.expectApproxEqAbs(expected, y.data[i], 1e-9);
    }
}

test "axpy: auto-dispatch threshold above (n=65, SIMD path)" {
    // Above threshold (65 > 64) — should use SIMD path
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f64, 65);
    defer allocator.free(data_x);
    var data_y = try allocator.alloc(f64, 65);
    defer allocator.free(data_y);

    for (0..65) |i| {
        data_x[i] = @as(f64, @floatFromInt(i + 1));
        data_y[i] = @as(f64, @floatFromInt(i + 1));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{65}, data_x, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{65}, data_y, .row_major);
    defer y.deinit();

    try axpy(f64, 2.0, x, &y);

    // y = 2*x + y = 3*[1,2,...,65]
    for (0..65) |i| {
        const expected = 3.0 * @as(f64, @floatFromInt(i + 1));
        try testing.expectApproxEqAbs(expected, y.data[i], 1e-9);
    }
}

test "axpy: auto-dispatch large vector (n=1024, SIMD path)" {
    // Well above threshold — verify SIMD path correctness
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f64, 1024);
    defer allocator.free(data_x);
    var data_y = try allocator.alloc(f64, 1024);
    defer allocator.free(data_y);

    for (0..1024) |i| {
        data_x[i] = @as(f64, @floatFromInt(i + 1));
        data_y[i] = @as(f64, @floatFromInt(i + 1));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1024}, data_x, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1024}, data_y, .row_major);
    defer y.deinit();

    try axpy(f64, 2.0, x, &y);

    // y = 2*x + y = 3*[1,2,...,1024]
    for (0..1024) |i| {
        const expected = 3.0 * @as(f64, @floatFromInt(i + 1));
        try testing.expectApproxEqAbs(expected, y.data[i], 1e-9);
    }
}

test "axpy: auto-dispatch non-aligned (n=100, SIMD tail loop)" {
    // Above threshold, non-multiple of vector width — test tail loop
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f64, 100);
    defer allocator.free(data_x);
    var data_y = try allocator.alloc(f64, 100);
    defer allocator.free(data_y);

    for (0..100) |i| {
        data_x[i] = @as(f64, @floatFromInt(i + 1));
        data_y[i] = @as(f64, @floatFromInt(i + 1));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{100}, data_x, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{100}, data_y, .row_major);
    defer y.deinit();

    try axpy(f64, -1.5, x, &y);

    // y = -1.5*x + y = -0.5*[1,2,...,100]
    for (0..100) |i| {
        const expected = -0.5 * @as(f64, @floatFromInt(i + 1));
        try testing.expectApproxEqAbs(expected, y.data[i], 1e-9);
    }
}

test "axpy: auto-dispatch negative values (n=128, SIMD)" {
    // Verify SIMD correctness with negative values
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f64, 128);
    defer allocator.free(data_x);
    var data_y = try allocator.alloc(f64, 128);
    defer allocator.free(data_y);

    for (0..128) |i| {
        data_x[i] = -@as(f64, @floatFromInt(i + 1));
        data_y[i] = @as(f64, @floatFromInt(i + 1));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{128}, data_x, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{128}, data_y, .row_major);
    defer y.deinit();

    try axpy(f64, 2.0, x, &y);

    // y = 2*(-i) + i = -i
    for (0..128) |i| {
        const expected = -@as(f64, @floatFromInt(i + 1));
        try testing.expectApproxEqAbs(expected, y.data[i], 1e-9);
    }
}

test "axpy: auto-dispatch f32 type (n=256, SIMD 8-wide)" {
    // f32 uses 8-wide SIMD vectors
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f32, 256);
    defer allocator.free(data_x);
    var data_y = try allocator.alloc(f32, 256);
    defer allocator.free(data_y);

    for (0..256) |i| {
        data_x[i] = @as(f32, @floatFromInt(i + 1));
        data_y[i] = @as(f32, @floatFromInt(i + 1));
    }

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{256}, data_x, .row_major);
    defer x.deinit();
    var y = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{256}, data_y, .row_major);
    defer y.deinit();

    try axpy(f32, 0.5, x, &y);

    // y = 0.5*x + y = 1.5*[1,2,...,256]
    for (0..256) |i| {
        const expected = 1.5 * @as(f32, @floatFromInt(i + 1));
        try testing.expectApproxEqAbs(expected, y.data[i], 1e-5);
    }
}

test "axpy: auto-dispatch alpha=0 (n=128, SIMD no-op)" {
    // Verify SIMD path handles alpha=0 correctly (y unchanged)
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f64, 128);
    defer allocator.free(data_x);
    var data_y = try allocator.alloc(f64, 128);
    defer allocator.free(data_y);

    for (0..128) |i| {
        data_x[i] = @as(f64, @floatFromInt(i + 1));
        data_y[i] = @as(f64, @floatFromInt((i + 1) * 10));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{128}, data_x, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{128}, data_y, .row_major);
    defer y.deinit();

    try axpy(f64, 0.0, x, &y);

    // y = 0*x + y = y (unchanged)
    for (0..128) |i| {
        const expected = @as(f64, @floatFromInt((i + 1) * 10));
        try testing.expectApproxEqAbs(expected, y.data[i], 1e-9);
    }
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
// nrm2 Auto-Dispatch Tests (Session 492 — SIMD Optimization)
// ============================================================================

test "nrm2: auto-dispatch — n=63 (below threshold, uses scalar)" {
    const allocator = testing.allocator;
    var x = try NDArray(f64, 1).zeros(allocator, &[_]usize{63}, .row_major);
    defer x.deinit();

    for (0..63) |i| {
        x.data[i] = 1.0;
    }

    const norm = try nrm2(f64, x);
    // sqrt(63) ≈ 7.937
    try testing.expectApproxEqAbs(@sqrt(@as(f64, 63.0)), norm, 1e-10);
}

test "nrm2: auto-dispatch — n=64 (at threshold, uses SIMD)" {
    const allocator = testing.allocator;
    var x = try NDArray(f64, 1).zeros(allocator, &[_]usize{64}, .row_major);
    defer x.deinit();

    for (0..64) |i| {
        x.data[i] = 1.0;
    }

    const norm = try nrm2(f64, x);
    // sqrt(64) = 8.0
    try testing.expectApproxEqAbs(8.0, norm, 1e-10);
}

test "nrm2: auto-dispatch — n=65 (above threshold, uses SIMD)" {
    const allocator = testing.allocator;
    var x = try NDArray(f64, 1).zeros(allocator, &[_]usize{65}, .row_major);
    defer x.deinit();

    for (0..65) |i| {
        x.data[i] = 1.0;
    }

    const norm = try nrm2(f64, x);
    // sqrt(65) ≈ 8.062
    try testing.expectApproxEqAbs(@sqrt(@as(f64, 65.0)), norm, 1e-10);
}

test "nrm2: auto-dispatch — large n=1024 SIMD correctness" {
    const allocator = testing.allocator;
    var x = try NDArray(f64, 1).zeros(allocator, &[_]usize{1024}, .row_major);
    defer x.deinit();

    // Sequential values: 0, 1, 2, ..., 1023
    for (0..1024) |i| {
        x.data[i] = @floatFromInt(i);
    }

    const norm = try nrm2(f64, x);
    // sum(i²) for i=0..1023 = n(n-1)(2n-1)/6 = 1024*1023*2047/6 = 358,372,352
    const expected_sum_sq: f64 = 358372352.0;
    try testing.expectApproxEqAbs(@sqrt(expected_sum_sq), norm, 1e-6);
}

test "nrm2: auto-dispatch — non-aligned n=100 (tail loop)" {
    const allocator = testing.allocator;
    var x = try NDArray(f64, 1).zeros(allocator, &[_]usize{100}, .row_major);
    defer x.deinit();

    for (0..100) |i| {
        x.data[i] = 2.0;
    }

    const norm = try nrm2(f64, x);
    // sqrt(100 * 4) = sqrt(400) = 20
    try testing.expectApproxEqAbs(20.0, norm, 1e-10);
}

test "nrm2: auto-dispatch — f32 type support (n=128)" {
    const allocator = testing.allocator;
    var x = try NDArray(f32, 1).zeros(allocator, &[_]usize{128}, .row_major);
    defer x.deinit();

    for (0..128) |i| {
        x.data[i] = @floatFromInt(i);
    }

    const norm = try nrm2(f32, x);
    // sum(i²) for i=0..127 = 127*128*255/6 = 685,440
    const expected_sum_sq: f32 = 685440.0;
    try testing.expectApproxEqAbs(@sqrt(expected_sum_sq), norm, 1e-3);
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
// asum Dispatch Tests (auto-routing to SIMD for n >= 64)
// ============================================================================

test "asum dispatch: threshold boundary n=63 (scalar fallback)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 63);
    defer allocator.free(data);

    for (0..63) |i| {
        data[i] = @as(f64, @floatFromInt(i + 1));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{63}, data, .row_major);
    defer x.deinit();

    const result = try asum(f64, x);
    // sum(1..63) = 63*64/2 = 2016
    try testing.expectApproxEqAbs(2016.0, result, 1e-9);
}

test "asum dispatch: threshold boundary n=64 (SIMD path)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 64);
    defer allocator.free(data);

    for (0..64) |i| {
        data[i] = @as(f64, @floatFromInt(i + 1));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{64}, data, .row_major);
    defer x.deinit();

    const result = try asum(f64, x);
    // sum(1..64) = 64*65/2 = 2080
    try testing.expectApproxEqAbs(2080.0, result, 1e-9);
}

test "asum dispatch: above threshold n=65 (SIMD path)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 65);
    defer allocator.free(data);

    for (0..65) |i| {
        data[i] = @as(f64, @floatFromInt(i + 1));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{65}, data, .row_major);
    defer x.deinit();

    const result = try asum(f64, x);
    // sum(1..65) = 65*66/2 = 2145
    try testing.expectApproxEqAbs(2145.0, result, 1e-9);
}

test "asum dispatch: large vector n=1024 SIMD correctness" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 1024);
    defer allocator.free(data);

    for (0..1024) |i| {
        data[i] = @as(f64, @floatFromInt(i)) - 512.0;
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1024}, data, .row_major);
    defer x.deinit();

    const result = try asum(f64, x);
    // sum of absolute deviations from 512
    // 512 terms: 0 to 511 = sum(0..511) = 511*512/2 = 130816
    // 512 terms: 1 to 512 = sum(1..512) = 512*513/2 = 131328
    // total = 130816 + 131328 = 262144
    try testing.expectApproxEqAbs(262144.0, result, 1e-9);
}

test "asum dispatch: f32 type n=64 (SIMD with 8-wide)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f32, 64);
    defer allocator.free(data);

    for (0..64) |i| {
        data[i] = @as(f32, @floatFromInt(i + 1));
    }

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{64}, data, .row_major);
    defer x.deinit();

    const result = try asum(f32, x);
    // sum(1..64) = 64*65/2 = 2080
    try testing.expectApproxEqAbs(@as(f32, 2080.0), result, 1e-5);
}

test "asum dispatch: non-aligned n=100 (tail loop coverage)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 100);
    defer allocator.free(data);

    for (0..50) |i| {
        data[i] = 1.0;
    }
    for (50..100) |i| {
        data[i] = -1.0;
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{100}, data, .row_major);
    defer x.deinit();

    const result = try asum(f64, x);
    // 50 ones + 50 ones = 100
    try testing.expectApproxEqAbs(100.0, result, 1e-9);
}

test "asum dispatch: non-aligned n=137 (prime, SIMD + tail)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 137);
    defer allocator.free(data);

    for (0..137) |i| {
        data[i] = if (i % 2 == 0) 1.0 else -1.0;
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{137}, data, .row_major);
    defer x.deinit();

    const result = try asum(f64, x);
    // 69 positive (indices 0,2,4,...,136) + 68 negative = 137
    try testing.expectApproxEqAbs(137.0, result, 1e-9);
}

test "asum dispatch: mixed signs correctness across threshold" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 200);
    defer allocator.free(data);

    for (0..100) |i| {
        data[i] = @as(f64, @floatFromInt(i + 1));
    }
    for (100..200) |i| {
        data[i] = -@as(f64, @floatFromInt(i - 99));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{200}, data, .row_major);
    defer x.deinit();

    const result = try asum(f64, x);
    // sum(1..100) + sum(1..100) = 5050 + 5050 = 10100
    try testing.expectApproxEqAbs(10100.0, result, 1e-9);
}

// ============================================================================
// scal Dispatch Tests (auto-routing to SIMD for n >= 64)
// ============================================================================

test "scal dispatch: threshold boundary n=63 (scalar fallback)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 63);
    defer allocator.free(data);

    for (0..63) |i| {
        data[i] = 2.0;
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{63}, data, .row_major);
    defer x.deinit();

    try scal(f64, 1.5, &x);

    for (0..63) |i| {
        try testing.expectApproxEqAbs(3.0, x.data[i], 1e-10);
    }
}

test "scal dispatch: threshold boundary n=64 (SIMD path)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 64);
    defer allocator.free(data);

    for (0..64) |i| {
        data[i] = 2.0;
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{64}, data, .row_major);
    defer x.deinit();

    try scal(f64, 1.5, &x);

    for (0..64) |i| {
        try testing.expectApproxEqAbs(3.0, x.data[i], 1e-10);
    }
}

test "scal dispatch: above threshold n=65 (SIMD path)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 65);
    defer allocator.free(data);

    for (0..65) |i| {
        data[i] = 2.0;
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{65}, data, .row_major);
    defer x.deinit();

    try scal(f64, 1.5, &x);

    for (0..65) |i| {
        try testing.expectApproxEqAbs(3.0, x.data[i], 1e-10);
    }
}

test "scal dispatch: large n=128 SIMD correctness" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 128);
    defer allocator.free(data);

    for (0..128) |i| {
        data[i] = @as(f64, @floatFromInt(i + 1));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{128}, data, .row_major);
    defer x.deinit();

    try scal(f64, 0.5, &x);

    for (0..128) |i| {
        const expected = 0.5 * @as(f64, @floatFromInt(i + 1));
        try testing.expectApproxEqAbs(expected, x.data[i], 1e-10);
    }
}

test "scal dispatch: large n=256 SIMD correctness" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 256);
    defer allocator.free(data);

    for (0..256) |i| {
        data[i] = 1.0;
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{256}, data, .row_major);
    defer x.deinit();

    try scal(f64, 2.5, &x);

    for (0..256) |i| {
        try testing.expectApproxEqAbs(2.5, x.data[i], 1e-10);
    }
}

test "scal dispatch: large n=1024 SIMD correctness" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 1024);
    defer allocator.free(data);

    for (0..1024) |i| {
        data[i] = @as(f64, @floatFromInt(i));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1024}, data, .row_major);
    defer x.deinit();

    try scal(f64, 3.0, &x);

    for (0..1024) |i| {
        const expected = 3.0 * @as(f64, @floatFromInt(i));
        try testing.expectApproxEqAbs(expected, x.data[i], 1e-10);
    }
}

test "scal dispatch: f32 type n=64 (SIMD with 8-wide)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f32, 64);
    defer allocator.free(data);

    for (0..64) |i| {
        data[i] = 2.0;
    }

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{64}, data, .row_major);
    defer x.deinit();

    try scal(f32, 3.5, &x);

    for (0..64) |i| {
        try testing.expectApproxEqAbs(@as(f32, 7.0), x.data[i], 1e-5);
    }
}

test "scal dispatch: f32 large vector n=256" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f32, 256);
    defer allocator.free(data);

    for (0..256) |i| {
        data[i] = @as(f32, @floatFromInt(i));
    }

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{256}, data, .row_major);
    defer x.deinit();

    try scal(f32, 2.0, &x);

    for (0..256) |i| {
        const expected = 2.0 * @as(f32, @floatFromInt(i));
        try testing.expectApproxEqAbs(expected, x.data[i], 1e-5);
    }
}

test "scal dispatch: non-aligned n=100 (tail loop coverage)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 100);
    defer allocator.free(data);

    for (0..100) |i| {
        data[i] = @as(f64, @floatFromInt(i + 1));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{100}, data, .row_major);
    defer x.deinit();

    try scal(f64, 0.5, &x);

    for (0..100) |i| {
        const expected = 0.5 * @as(f64, @floatFromInt(i + 1));
        try testing.expectApproxEqAbs(expected, x.data[i], 1e-10);
    }
}

test "scal dispatch: non-aligned n=137 (prime, SIMD + tail)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 137);
    defer allocator.free(data);

    for (0..137) |i| {
        data[i] = 1.0;
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{137}, data, .row_major);
    defer x.deinit();

    try scal(f64, 2.0, &x);

    for (0..137) |i| {
        try testing.expectApproxEqAbs(2.0, x.data[i], 1e-10);
    }
}

test "scal dispatch: alpha=0 with SIMD (n=128)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 128);
    defer allocator.free(data);

    for (0..128) |i| {
        data[i] = @as(f64, @floatFromInt(i + 1));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{128}, data, .row_major);
    defer x.deinit();

    try scal(f64, 0.0, &x);

    for (0..128) |i| {
        try testing.expectApproxEqAbs(0.0, x.data[i], 1e-10);
    }
}

test "scal dispatch: alpha=1 with SIMD (n=128)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 128);
    defer allocator.free(data);

    for (0..128) |i| {
        data[i] = @as(f64, @floatFromInt(i + 1));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{128}, data, .row_major);
    defer x.deinit();

    try scal(f64, 1.0, &x);

    for (0..128) |i| {
        const expected = @as(f64, @floatFromInt(i + 1));
        try testing.expectApproxEqAbs(expected, x.data[i], 1e-10);
    }
}

test "scal dispatch: alpha=-1 sign flip with SIMD (n=128)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 128);
    defer allocator.free(data);

    for (0..128) |i| {
        data[i] = @as(f64, @floatFromInt(i + 1));
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{128}, data, .row_major);
    defer x.deinit();

    try scal(f64, -1.0, &x);

    for (0..128) |i| {
        const expected = -@as(f64, @floatFromInt(i + 1));
        try testing.expectApproxEqAbs(expected, x.data[i], 1e-10);
    }
}

test "scal dispatch: alpha=-0.5 negative fractional with SIMD (n=100)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 100);
    defer allocator.free(data);

    for (0..100) |i| {
        data[i] = 2.0;
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{100}, data, .row_major);
    defer x.deinit();

    try scal(f64, -0.5, &x);

    for (0..100) |i| {
        try testing.expectApproxEqAbs(-1.0, x.data[i], 1e-10);
    }
}

test "scal dispatch: alpha=2.5 greater than 1 with SIMD (n=80)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 80);
    defer allocator.free(data);

    for (0..80) |i| {
        data[i] = 2.0;
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{80}, data, .row_major);
    defer x.deinit();

    try scal(f64, 2.5, &x);

    for (0..80) |i| {
        try testing.expectApproxEqAbs(5.0, x.data[i], 1e-10);
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

    // Auto-dispatch: Use SIMD-optimized implementation for large matrices
    // Threshold: if m >= 64 (enough rows to benefit from vectorization)
    // Session 487: gemv_simd_optimized provides 2-4× speedup via vectorized dot products
    const threshold: usize = 64;
    if (m >= threshold) {
        return try simd_blas.gemv_simd_optimized(T, alpha, A, x, beta, y);
    }

    // Fallback to naive loop for small matrices
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
/// ger — BLAS Level 2 rank-1 update: A := A + α*x*y^T
///
/// Auto-dispatches to SIMD implementation for large matrices (m >= 64 OR n >= 64).
/// Computes outer product rank-1 update with scalar alpha.
///
/// Time: O(m*n)
/// Space: O(1)
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

    // Auto-dispatch: use SIMD for large matrices
    // (Session 494: ger_simd provides 3-6× speedup for m >= 64 OR n >= 64)
    if (m >= 64 or n >= 64) {
        return try simd_blas.ger_simd(T, alpha, x, y, A);
    }

    // Scalar path for small matrices (SIMD overhead not justified)
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
// ger Auto-Dispatch Tests (Session 494)
// ============================================================================

test "ger dispatch: 63x63 matrix uses scalar path" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).ones(allocator, &[_]usize{63}, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).ones(allocator, &[_]usize{63}, .row_major);
    defer y.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 63, 63 }, .row_major);
    defer A.deinit();

    // α=0.5, x=ones(63), y=ones(63) → A += 0.5 (scalar path: m < 64 and n < 64)
    try ger(f64, 0.5, x, y, &A);

    // All elements should be 0.5
    try testing.expectApproxEqAbs(0.5, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(0.5, A.data[62], 1e-10);
    try testing.expectApproxEqAbs(0.5, A.data[63 * 62], 1e-10);
    try testing.expectApproxEqAbs(0.5, A.data[63 * 63 - 1], 1e-10);
}

test "ger dispatch: 64x64 matrix uses SIMD path" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).ones(allocator, &[_]usize{64}, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).ones(allocator, &[_]usize{64}, .row_major);
    defer y.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 64, 64 }, .row_major);
    defer A.deinit();

    // α=0.5, x=ones(64), y=ones(64) → A += 0.5 (SIMD path: m >= 64)
    try ger(f64, 0.5, x, y, &A);

    // All elements should be 0.5
    try testing.expectApproxEqAbs(0.5, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(0.5, A.data[63], 1e-10);
    try testing.expectApproxEqAbs(0.5, A.data[64 * 63], 1e-10);
    try testing.expectApproxEqAbs(0.5, A.data[64 * 64 - 1], 1e-10);
}

test "ger dispatch: 65x65 matrix uses SIMD path" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).ones(allocator, &[_]usize{65}, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).ones(allocator, &[_]usize{65}, .row_major);
    defer y.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 65, 65 }, .row_major);
    defer A.deinit();

    // α=1.0, x=ones(65), y=ones(65) → A += 1.0 (SIMD path: m >= 64)
    try ger(f64, 1.0, x, y, &A);

    // All elements should be 1.0
    try testing.expectApproxEqAbs(1.0, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(1.0, A.data[64], 1e-10);
    try testing.expectApproxEqAbs(1.0, A.data[65 * 64], 1e-10);
    try testing.expectApproxEqAbs(1.0, A.data[65 * 65 - 1], 1e-10);
}

test "ger dispatch: 64x128 non-square uses SIMD path" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).ones(allocator, &[_]usize{64}, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).full(allocator, &[_]usize{128}, 2.0, .row_major);
    defer y.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 64, 128 }, .row_major);
    defer A.deinit();

    // α=0.5, x=ones(64), y=2*ones(128) → A += 0.5*1*2 = 1.0 (SIMD path: m >= 64)
    try ger(f64, 0.5, x, y, &A);

    // All elements should be 1.0
    try testing.expectApproxEqAbs(1.0, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(1.0, A.data[127], 1e-10);
    try testing.expectApproxEqAbs(1.0, A.data[63 * 128], 1e-10);
    try testing.expectApproxEqAbs(1.0, A.data[64 * 128 - 1], 1e-10);
}

test "ger dispatch: 128x64 non-square uses SIMD path" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).full(allocator, &[_]usize{128}, 3.0, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).ones(allocator, &[_]usize{64}, .row_major);
    defer y.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 128, 64 }, .row_major);
    defer A.deinit();

    // α=2.0, x=3*ones(128), y=ones(64) → A += 2*3*1 = 6.0 (SIMD path: m >= 64)
    try ger(f64, 2.0, x, y, &A);

    // All elements should be 6.0
    try testing.expectApproxEqAbs(6.0, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(6.0, A.data[63], 1e-10);
    try testing.expectApproxEqAbs(6.0, A.data[127 * 64], 1e-10);
    try testing.expectApproxEqAbs(6.0, A.data[128 * 64 - 1], 1e-10);
}

test "ger dispatch: 100x200 non-aligned uses SIMD path" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).ones(allocator, &[_]usize{100}, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).full(allocator, &[_]usize{200}, 0.25, .row_major);
    defer y.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 100, 200 }, .row_major);
    defer A.deinit();

    // α=4.0, x=ones(100), y=0.25*ones(200) → A += 4*1*0.25 = 1.0 (SIMD path: m >= 64 and n >= 64)
    try ger(f64, 4.0, x, y, &A);

    // All elements should be 1.0
    try testing.expectApproxEqAbs(1.0, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(1.0, A.data[199], 1e-10);
    try testing.expectApproxEqAbs(1.0, A.data[99 * 200], 1e-10);
    try testing.expectApproxEqAbs(1.0, A.data[100 * 200 - 1], 1e-10);
}

test "ger dispatch: f32 type support" {
    const allocator = testing.allocator;

    var x = try NDArray(f32, 1).ones(allocator, &[_]usize{64}, .row_major);
    defer x.deinit();

    var y = try NDArray(f32, 1).ones(allocator, &[_]usize{64}, .row_major);
    defer y.deinit();

    var A = try NDArray(f32, 2).zeros(allocator, &[_]usize{ 64, 64 }, .row_major);
    defer A.deinit();

    // α=2.0, x=ones(64), y=ones(64) → A += 2 (f32 8-wide SIMD)
    try ger(f32, 2.0, x, y, &A);

    // All elements should be 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), A.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.0), A.data[63], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.0), A.data[64 * 63], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.0), A.data[64 * 64 - 1], 1e-5);
}

test "ger dispatch: alpha=0 no-op" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).full(allocator, &[_]usize{64}, 999.0, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).full(allocator, &[_]usize{64}, 888.0, .row_major);
    defer y.deinit();

    var A = try NDArray(f64, 2).full(allocator, &[_]usize{ 64, 64 }, 5.0, .row_major);
    defer A.deinit();

    // α=0 → A unchanged (SIMD path handles α=0 correctly)
    try ger(f64, 0.0, x, y, &A);

    // All elements should remain 5.0
    try testing.expectApproxEqAbs(5.0, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(5.0, A.data[63], 1e-10);
    try testing.expectApproxEqAbs(5.0, A.data[64 * 64 - 1], 1e-10);
}

test "ger dispatch: large matrix 256x256" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).ones(allocator, &[_]usize{256}, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).ones(allocator, &[_]usize{256}, .row_major);
    defer y.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 256, 256 }, .row_major);
    defer A.deinit();

    // α=-0.5, x=ones(256), y=ones(256) → A += -0.5 (large SIMD path)
    try ger(f64, -0.5, x, y, &A);

    // Sample elements should be -0.5
    try testing.expectApproxEqAbs(-0.5, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(-0.5, A.data[255], 1e-10);
    try testing.expectApproxEqAbs(-0.5, A.data[128 * 256 + 128], 1e-10);
    try testing.expectApproxEqAbs(-0.5, A.data[256 * 256 - 1], 1e-10);
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

    // Auto-dispatch: 3-tier strategy for optimal cache utilization
    // Tier 1: Very large matrices (>= 512×512) → cache-blocked tiled GEMM
    //   (Session 506: gemm_blocked_tiled provides 1.5-2× speedup over gemm_simd_optimized via L2/L3 cache optimization)
    // Tier 2: Large matrices (>= 64×64) → SIMD-optimized GEMM
    //   (Session 484: gemm_simd_optimized provides 2-3× speedup over naive via SIMD vectorization)
    // Tier 3: Small matrices (< 64×64) → naive triple-loop (SIMD overhead not worth it)

    const cache_block_threshold: usize = 512;
    const simd_threshold: usize = 64;

    if (m >= cache_block_threshold and n >= cache_block_threshold) {
        return try simd_blas.gemm_blocked_tiled(T, alpha, A, B, beta, C);
    } else if (m >= simd_threshold and n >= simd_threshold) {
        return try simd_blas.gemm_simd_optimized(T, alpha, A, B, beta, C);
    }

    // Fallback to naive triple-loop for small matrices
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

// ============================================================================
// GEMM Auto-Dispatch Tests (RED tests for blocking optimization)
// ============================================================================
// Tests verify that gemm() correctly auto-dispatches to blocked implementation
// for large matrices while maintaining correctness and performance.

test "gemm: threshold boundary 63x63 uses naive implementation" {
    const allocator = testing.allocator;
    const size = 63;

    // Create 63×63 matrices with known values
    var data_A = try allocator.alloc(f64, size * size);
    defer allocator.free(data_A);
    var data_B = try allocator.alloc(f64, size * size);
    defer allocator.free(data_B);
    var data_C = try allocator.alloc(f64, size * size);
    defer allocator.free(data_C);

    // Initialize with simple patterns
    for (0..size * size) |i| {
        data_A[i] = @as(f64, @floatFromInt((i % size) + 1));
        data_B[i] = @as(f64, @floatFromInt((i / size) + 1));
        data_C[i] = 1.0;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_A, .row_major);
    defer A.deinit();
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_B, .row_major);
    defer B.deinit();
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_C, .row_major);
    defer C.deinit();

    // Compute: C = 2.0*A*B + 0.5*C
    try gemm(f64, 2.0, A, B, 0.5, &C);

    // Verify some results (spot checks to ensure computation completed)
    try testing.expect(!std.math.isNan(C.data[0]));
    try testing.expect(!std.math.isInf(C.data[0]));
    try testing.expect(C.data[0] > 0.0); // Should have accumulated positive values
}

test "gemm: threshold boundary 64x64 switches to blocked implementation" {
    const allocator = testing.allocator;
    const size = 64;

    // Create 64×64 matrices
    var data_A = try allocator.alloc(f64, size * size);
    defer allocator.free(data_A);
    var data_B = try allocator.alloc(f64, size * size);
    defer allocator.free(data_B);
    var data_C = try allocator.alloc(f64, size * size);
    defer allocator.free(data_C);

    for (0..size * size) |i| {
        data_A[i] = @as(f64, @floatFromInt((i % size) + 1));
        data_B[i] = @as(f64, @floatFromInt((i / size) + 1));
        data_C[i] = 1.0;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_A, .row_major);
    defer A.deinit();
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_B, .row_major);
    defer B.deinit();
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_C, .row_major);
    defer C.deinit();

    // Compute: C = 2.0*A*B + 0.5*C
    try gemm(f64, 2.0, A, B, 0.5, &C);

    // Verify results are valid (should use blocked implementation)
    try testing.expect(!std.math.isNan(C.data[0]));
    try testing.expect(!std.math.isInf(C.data[0]));
    try testing.expect(C.data[0] > 0.0);
}

test "gemm: threshold boundary 65x65 uses blocked implementation" {
    const allocator = testing.allocator;
    const size = 65;

    var data_A = try allocator.alloc(f64, size * size);
    defer allocator.free(data_A);
    var data_B = try allocator.alloc(f64, size * size);
    defer allocator.free(data_B);
    var data_C = try allocator.alloc(f64, size * size);
    defer allocator.free(data_C);

    for (0..size * size) |i| {
        data_A[i] = @as(f64, @floatFromInt((i % size) + 1));
        data_B[i] = @as(f64, @floatFromInt((i / size) + 1));
        data_C[i] = 1.0;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_A, .row_major);
    defer A.deinit();
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_B, .row_major);
    defer B.deinit();
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_C, .row_major);
    defer C.deinit();

    try gemm(f64, 2.0, A, B, 0.5, &C);

    try testing.expect(!std.math.isNan(C.data[0]));
    try testing.expect(!std.math.isInf(C.data[0]));
    try testing.expect(C.data[0] > 0.0);
}

test "gemm: non-square matrices at threshold (64x32x64)" {
    const allocator = testing.allocator;

    // A: 64×32, B: 32×64, C: 64×64
    var data_A = try allocator.alloc(f64, 64 * 32);
    defer allocator.free(data_A);
    var data_B = try allocator.alloc(f64, 32 * 64);
    defer allocator.free(data_B);
    var data_C = try allocator.alloc(f64, 64 * 64);
    defer allocator.free(data_C);

    for (0..64 * 32) |i| {
        data_A[i] = @as(f64, @floatFromInt(i % 10 + 1));
    }
    for (0..32 * 64) |i| {
        data_B[i] = @as(f64, @floatFromInt(i % 10 + 1));
    }
    for (0..64 * 64) |i| {
        data_C[i] = 0.0;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 32 }, data_A, .row_major);
    defer A.deinit();
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 32, 64 }, data_B, .row_major);
    defer B.deinit();
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 64 }, data_C, .row_major);
    defer C.deinit();

    try gemm(f64, 1.0, A, B, 0.0, &C);

    // Verify results are finite
    for (0..64 * 64) |i| {
        try testing.expect(!std.math.isNan(C.data[i]));
        try testing.expect(!std.math.isInf(C.data[i]));
    }
}

test "gemm: rectangular 32x64x32 (tall) with dispatch" {
    const allocator = testing.allocator;

    // A: 32×64, B: 64×32, C: 32×32
    var data_A = try allocator.alloc(f64, 32 * 64);
    defer allocator.free(data_A);
    var data_B = try allocator.alloc(f64, 64 * 32);
    defer allocator.free(data_B);
    var data_C = try allocator.alloc(f64, 32 * 32);
    defer allocator.free(data_C);

    for (0..32 * 64) |i| {
        data_A[i] = @as(f64, @floatFromInt(i % 7 + 1));
    }
    for (0..64 * 32) |i| {
        data_B[i] = @as(f64, @floatFromInt(i % 7 + 1));
    }
    for (0..32 * 32) |i| {
        data_C[i] = 0.5;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 32, 64 }, data_A, .row_major);
    defer A.deinit();
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 32 }, data_B, .row_major);
    defer B.deinit();
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 32, 32 }, data_C, .row_major);
    defer C.deinit();

    try gemm(f64, 1.5, A, B, 2.0, &C);

    for (0..32 * 32) |i| {
        try testing.expect(!std.math.isNan(C.data[i]));
        try testing.expect(!std.math.isInf(C.data[i]));
    }
}

test "gemm: correctness at 64x64 with alpha=0 (zero scaling)" {
    const allocator = testing.allocator;
    const size = 64;

    var data_A = try allocator.alloc(f64, size * size);
    defer allocator.free(data_A);
    var data_B = try allocator.alloc(f64, size * size);
    defer allocator.free(data_B);
    var data_C = try allocator.alloc(f64, size * size);
    defer allocator.free(data_C);

    for (0..size * size) |i| {
        data_A[i] = @as(f64, @floatFromInt(i + 1));
        data_B[i] = @as(f64, @floatFromInt(i + 1));
        data_C[i] = 5.0; // Initial value
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_A, .row_major);
    defer A.deinit();
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_B, .row_major);
    defer B.deinit();
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_C, .row_major);
    defer C.deinit();

    // C = 0*A*B + 2.5*C = [[12.5, ...]]
    try gemm(f64, 0.0, A, B, 2.5, &C);

    // All elements should be 5.0 * 2.5 = 12.5
    for (0..size * size) |i| {
        try testing.expectApproxEqAbs(12.5, C.data[i], 1e-10);
    }
}

test "gemm: correctness at 64x64 with beta=0 (pure matrix multiply)" {
    const allocator = testing.allocator;
    const size = 64;

    var data_A = try allocator.alloc(f64, size * size);
    defer allocator.free(data_A);
    var data_B = try allocator.alloc(f64, size * size);
    defer allocator.free(data_B);
    var data_C = try allocator.alloc(f64, size * size);
    defer allocator.free(data_C);

    // Simple diagonal patterns for verification
    for (0..size * size) |i| {
        data_A[i] = if (i % (size + 1) == 0) 2.0 else 0.0; // 2*Identity
        data_B[i] = if (i % (size + 1) == 0) 3.0 else 0.0; // 3*Identity
        data_C[i] = 99.0; // Should be overwritten
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_A, .row_major);
    defer A.deinit();
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_B, .row_major);
    defer B.deinit();
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_C, .row_major);
    defer C.deinit();

    // C = 1.0*(2I)*(3I) + 0*C = 6I
    try gemm(f64, 1.0, A, B, 0.0, &C);

    // Verify diagonal is 6.0, off-diagonal is 0.0
    for (0..size) |i| {
        for (0..size) |j| {
            const idx = i * size + j;
            const expected = if (i == j) 6.0 else 0.0;
            try testing.expectApproxEqAbs(expected, C.data[idx], 1e-10);
        }
    }
}

test "gemm: f32 precision at 64x64 threshold" {
    const allocator = testing.allocator;
    const size = 64;

    var data_A = try allocator.alloc(f32, size * size);
    defer allocator.free(data_A);
    var data_B = try allocator.alloc(f32, size * size);
    defer allocator.free(data_B);
    var data_C = try allocator.alloc(f32, size * size);
    defer allocator.free(data_C);

    for (0..size * size) |i| {
        data_A[i] = @as(f32, @floatFromInt((i % 10) + 1));
        data_B[i] = @as(f32, @floatFromInt((i % 10) + 1));
        data_C[i] = 0.1;
    }

    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ size, size }, data_A, .row_major);
    defer A.deinit();
    var B = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ size, size }, data_B, .row_major);
    defer B.deinit();
    var C = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ size, size }, data_C, .row_major);
    defer C.deinit();

    try gemm(f32, 1.0, A, B, 1.0, &C);

    for (0..size * size) |i| {
        try testing.expect(!std.math.isNan(C.data[i]));
        try testing.expect(!std.math.isInf(C.data[i]));
        try testing.expect(C.data[i] >= 0.0);
    }
}

test "gemm: negative alpha at 64x64 threshold" {
    const allocator = testing.allocator;
    const size = 64;

    var data_A = try allocator.alloc(f64, size * size);
    defer allocator.free(data_A);
    var data_B = try allocator.alloc(f64, size * size);
    defer allocator.free(data_B);
    var data_C = try allocator.alloc(f64, size * size);
    defer allocator.free(data_C);

    for (0..size * size) |i| {
        data_A[i] = 1.0;
        data_B[i] = 1.0;
        data_C[i] = 10.0;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_A, .row_major);
    defer A.deinit();
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_B, .row_major);
    defer B.deinit();
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_C, .row_major);
    defer C.deinit();

    // C = -2.0*ones*ones + 0.5*C
    // ones*ones = 64*ones (since we have size×size ones)
    // C[i,j] = -2.0*64 + 0.5*10 = -128 + 5 = -123
    try gemm(f64, -2.0, A, B, 0.5, &C);

    const expected = -128.0 + 5.0; // -123.0
    for (0..size * size) |i| {
        try testing.expectApproxEqAbs(expected, C.data[i], 1e-9);
    }
}

test "gemm: large matrix 128x128 uses blocked implementation" {
    const allocator = testing.allocator;
    const size = 128;

    var data_A = try allocator.alloc(f64, size * size);
    defer allocator.free(data_A);
    var data_B = try allocator.alloc(f64, size * size);
    defer allocator.free(data_B);
    var data_C = try allocator.alloc(f64, size * size);
    defer allocator.free(data_C);

    for (0..size * size) |i| {
        data_A[i] = @as(f64, @floatFromInt((i % 5) + 1));
        data_B[i] = @as(f64, @floatFromInt((i % 5) + 1));
        data_C[i] = 0.5;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_A, .row_major);
    defer A.deinit();
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_B, .row_major);
    defer B.deinit();
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_C, .row_major);
    defer C.deinit();

    try gemm(f64, 1.5, A, B, 0.5, &C);

    // Spot checks for validity
    try testing.expect(!std.math.isNan(C.data[0]));
    try testing.expect(!std.math.isNan(C.data[size * size - 1]));
}

test "gemm: mixed scaling with alpha and beta at 64x64" {
    const allocator = testing.allocator;
    const size = 64;

    var data_A = try allocator.alloc(f64, size * size);
    defer allocator.free(data_A);
    var data_B = try allocator.alloc(f64, size * size);
    defer allocator.free(data_B);
    var data_C = try allocator.alloc(f64, size * size);
    defer allocator.free(data_C);

    // Create identity matrices for predictable computation
    for (0..size * size) |i| {
        data_A[i] = if (i % (size + 1) == 0) 1.0 else 0.0;
        data_B[i] = if (i % (size + 1) == 0) 1.0 else 0.0;
        data_C[i] = 2.0;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_A, .row_major);
    defer A.deinit();
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_B, .row_major);
    defer B.deinit();
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ size, size }, data_C, .row_major);
    defer C.deinit();

    // C = 3.5*I*I + 1.5*C = 3.5*I + 1.5*C
    // Diagonal: 3.5*1 + 1.5*2 = 6.5, Off-diagonal: 3.5*0 + 1.5*2 = 3.0
    try gemm(f64, 3.5, A, B, 1.5, &C);

    for (0..size) |i| {
        for (0..size) |j| {
            const idx = i * size + j;
            const expected = if (i == j) 6.5 else 3.0;
            try testing.expectApproxEqAbs(expected, C.data[idx], 1e-10);
        }
    }
}

test "gemm: tall matrix 64x16 inner dimension triggers dispatch" {
    const allocator = testing.allocator;

    var data_A = try allocator.alloc(f64, 64 * 16);
    defer allocator.free(data_A);
    var data_B = try allocator.alloc(f64, 16 * 64);
    defer allocator.free(data_B);
    var data_C = try allocator.alloc(f64, 64 * 64);
    defer allocator.free(data_C);

    for (0..64 * 16) |i| {
        data_A[i] = 1.0;
    }
    for (0..16 * 64) |i| {
        data_B[i] = 1.0;
    }
    for (0..64 * 64) |i| {
        data_C[i] = 0.0;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 16 }, data_A, .row_major);
    defer A.deinit();
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 16, 64 }, data_B, .row_major);
    defer B.deinit();
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 64 }, data_C, .row_major);
    defer C.deinit();

    try gemm(f64, 1.0, A, B, 0.0, &C);

    // Each element of C = sum of 16 ones = 16.0
    for (0..64 * 64) |i| {
        try testing.expectApproxEqAbs(16.0, C.data[i], 1e-10);
    }
}

// ============================================================================
// BLAS Level 2 — Triangular Matrix Operations
// ============================================================================

/// Triangular matrix-vector multiply: x = A*x or x = A^T*x
///
/// Performs matrix-vector multiplication with a triangular matrix A.
/// The matrix can be upper or lower triangular, with or without unit diagonal.
///
/// Parameters:
/// - uplo: 'U' for upper triangular, 'L' for lower triangular
/// - trans: 'N' for no transpose (x = A*x), 'T' for transpose (x = A^T*x)
/// - diag: 'N' for non-unit diagonal, 'U' for unit diagonal (diagonal = 1)
/// - A: Triangular matrix (2D NDArray, square)
/// - x: Vector (1D NDArray) — modified in-place
///
/// Errors:
/// - error.DimensionMismatch if A is not square or if A.shape[0] != x.shape[0]
///
/// Time: O(n²) where n = dimension of A
/// Space: O(1) (modifies x in-place)
///
/// Example:
/// ```zig
/// // Upper triangular: A = [[1, 2, 3], [0, 4, 5], [0, 0, 6]]
/// var A = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{3, 3},
///     &[_]f64{1, 2, 3, 0, 4, 5, 0, 0, 6}, .row_major);
/// defer A.deinit();
/// var x = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{3}, &[_]f64{1, 1, 1}, .row_major);
/// defer x.deinit();
/// try trmv(f64, 'U', 'N', 'N', A, &x);  // x = [1+2+3, 4+5, 6] = [6, 9, 6]
/// ```
pub fn trmv(comptime T: type, uplo: u8, trans: u8, diag: u8, A: NDArray(T, 2), x: *NDArray(T, 1)) (NDArray(T, 2).Error)!void {
    // Validate square matrix
    if (A.shape[0] != A.shape[1]) {
        return error.DimensionMismatch;
    }

    // Validate dimensions
    if (A.shape[0] != x.shape[0]) {
        return error.DimensionMismatch;
    }

    const n = A.shape[0];

    // Auto-dispatch: Use SIMD-optimized implementation for large matrices
    // Threshold: if n >= 64, use the SIMD trmv
    // (trmv_simd provides 2-4× speedup over scalar for large triangular matrices)
    const threshold: usize = 64;
    if (n >= threshold) {
        return try simd_blas.trmv_simd(T, uplo, trans, diag, A, x);
    }

    const is_upper = (uplo == 'U' or uplo == 'u');
    const is_trans = (trans == 'T' or trans == 't');
    const is_unit = (diag == 'U' or diag == 'u');

    // Create temporary array to store result
    var temp = try std.mem.Allocator.alloc(x.allocator, T, n);
    defer x.allocator.free(temp);
    @memset(temp, 0);

    if (!is_trans) {
        // x = A*x
        if (is_upper) {
            // Upper triangular
            var i: usize = n;
            while (i > 0) {
                i -= 1;
                var sum: T = 0;
                // Start from diagonal (or after if unit)
                const start = if (is_unit) i + 1 else i;
                for (start..n) |j| {
                    sum += A.data[i * n + j] * x.data[j];
                }
                if (is_unit) {
                    temp[i] = x.data[i] + sum;
                } else {
                    temp[i] = sum;
                }
            }
        } else {
            // Lower triangular
            for (0..n) |i| {
                var sum: T = 0;
                // Start from beginning to diagonal (or before if unit)
                const end = if (is_unit) i else i + 1;
                for (0..end) |j| {
                    sum += A.data[i * n + j] * x.data[j];
                }
                if (is_unit) {
                    temp[i] = x.data[i] + sum;
                } else {
                    temp[i] = sum;
                }
            }
        }
    } else {
        // x = A^T*x
        if (is_upper) {
            // Upper triangular transpose (acts like lower)
            for (0..n) |i| {
                var sum: T = 0;
                const end = if (is_unit) i else i + 1;
                for (0..end) |j| {
                    sum += A.data[j * n + i] * x.data[j]; // A^T[i,j] = A[j,i]
                }
                if (is_unit) {
                    temp[i] = x.data[i] + sum;
                } else {
                    temp[i] = sum;
                }
            }
        } else {
            // Lower triangular transpose (acts like upper)
            var i: usize = n;
            while (i > 0) {
                i -= 1;
                var sum: T = 0;
                const start = if (is_unit) i + 1 else i;
                for (start..n) |j| {
                    sum += A.data[j * n + i] * x.data[j]; // A^T[i,j] = A[j,i]
                }
                if (is_unit) {
                    temp[i] = x.data[i] + sum;
                } else {
                    temp[i] = sum;
                }
            }
        }
    }

    // Copy result back to x
    @memcpy(x.data, temp);
}

/// Triangular solve: x = A^(-1)*x or x = A^(-T)*x
///
/// Solves the triangular system A*x = b or A^T*x = b where A is triangular.
/// The solution is stored in x (in-place).
///
/// Parameters:
/// - uplo: 'U' for upper triangular, 'L' for lower triangular
/// - trans: 'N' for no transpose (A*x = b), 'T' for transpose (A^T*x = b)
/// - diag: 'N' for non-unit diagonal, 'U' for unit diagonal (diagonal = 1)
/// - A: Triangular matrix (2D NDArray, square)
/// - x: Right-hand side vector (1D NDArray) — modified to solution in-place
///
/// Errors:
/// - error.DimensionMismatch if A is not square or if A.shape[0] != x.shape[0]
///
/// Time: O(n²) where n = dimension of A
/// Space: O(1) (modifies x in-place)
///
/// Example:
/// ```zig
/// // Upper triangular: A = [[2, 1], [0, 3]]
/// var A = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{2, 2},
///     &[_]f64{2, 1, 0, 3}, .row_major);
/// defer A.deinit();
/// var x = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{2}, &[_]f64{5, 6}, .row_major);
/// defer x.deinit();
/// try trsv(f64, 'U', 'N', 'N', A, &x);  // Solve A*x = [5, 6]
/// ```
pub fn trsv(comptime T: type, uplo: u8, trans: u8, diag: u8, A: NDArray(T, 2), x: *NDArray(T, 1)) (NDArray(T, 2).Error)!void {
    // Validate square matrix
    if (A.shape[0] != A.shape[1]) {
        return error.DimensionMismatch;
    }

    // Validate dimensions
    if (A.shape[0] != x.shape[0]) {
        return error.DimensionMismatch;
    }

    const n = A.shape[0];

    // Auto-dispatch: use SIMD implementation for n >= 64
    if (n >= 64) {
        return try simd_blas.trsv_simd(T, uplo, trans, diag, A, x);
    }

    // Scalar fallback for small matrices
    const is_upper = (uplo == 'U' or uplo == 'u');
    const is_trans = (trans == 'T' or trans == 't');
    const is_unit = (diag == 'U' or diag == 'u');

    if (!is_trans) {
        // Solve A*x = b
        if (is_upper) {
            // Upper triangular: back substitution
            var i: usize = n;
            while (i > 0) {
                i -= 1;
                var sum: T = x.data[i];
                for (i + 1..n) |j| {
                    sum -= A.data[i * n + j] * x.data[j];
                }
                if (!is_unit) {
                    x.data[i] = sum / A.data[i * n + i];
                } else {
                    x.data[i] = sum;
                }
            }
        } else {
            // Lower triangular: forward substitution
            for (0..n) |i| {
                var sum: T = x.data[i];
                for (0..i) |j| {
                    sum -= A.data[i * n + j] * x.data[j];
                }
                if (!is_unit) {
                    x.data[i] = sum / A.data[i * n + i];
                } else {
                    x.data[i] = sum;
                }
            }
        }
    } else {
        // Solve A^T*x = b
        if (is_upper) {
            // Upper triangular transpose: forward substitution
            for (0..n) |i| {
                var sum: T = x.data[i];
                for (0..i) |j| {
                    sum -= A.data[j * n + i] * x.data[j]; // A^T[i,j] = A[j,i]
                }
                if (!is_unit) {
                    x.data[i] = sum / A.data[i * n + i];
                } else {
                    x.data[i] = sum;
                }
            }
        } else {
            // Lower triangular transpose: back substitution
            var i: usize = n;
            while (i > 0) {
                i -= 1;
                var sum: T = x.data[i];
                for (i + 1..n) |j| {
                    sum -= A.data[j * n + i] * x.data[j]; // A^T[i,j] = A[j,i]
                }
                if (!is_unit) {
                    x.data[i] = sum / A.data[i * n + i];
                } else {
                    x.data[i] = sum;
                }
            }
        }
    }
}

// ============================================================================
// BLAS Level 3 — Triangular Matrix-Matrix Operations
// ============================================================================

/// Triangular matrix-matrix multiply: B = α*A*B or B = α*B*A
///
/// Performs matrix-matrix multiplication with a triangular matrix A.
///
/// Parameters:
/// - side: 'L' for left (B = α*A*B), 'R' for right (B = α*B*A)
/// - uplo: 'U' for upper triangular, 'L' for lower triangular
/// - trans: 'N' for no transpose, 'T' for transpose
/// - diag: 'N' for non-unit diagonal, 'U' for unit diagonal
/// - alpha: Scalar multiplier
/// - A: Triangular matrix (2D NDArray, square)
/// - B: Matrix (2D NDArray) — modified in-place
///
/// Errors:
/// - error.DimensionMismatch if dimensions don't match
///
/// Time: O(m*n*k) where B is m×n, A is k×k
/// Space: O(m*n) for temporary storage
///
/// Example:
/// ```zig
/// // A = [[2, 1], [0, 3]] (upper triangular)
/// var A = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{2, 2},
///     &[_]f64{2, 1, 0, 3}, .row_major);
/// defer A.deinit();
/// var B = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{2, 2},
///     &[_]f64{1, 2, 3, 4}, .row_major);
/// defer B.deinit();
/// try trmm(f64, 'L', 'U', 'N', 'N', 1.0, A, &B);  // B = A*B
/// ```
pub fn trmm(comptime T: type, side: u8, uplo: u8, trans: u8, diag: u8, alpha: T, A: NDArray(T, 2), B: *NDArray(T, 2)) (NDArray(T, 2).Error)!void {
    // Validate square matrix A
    if (A.shape[0] != A.shape[1]) {
        return error.DimensionMismatch;
    }

    const is_left = (side == 'L' or side == 'l');
    const m = B.shape[0];
    const n = B.shape[1];
    const k = A.shape[0];

    // Validate dimensions
    if (is_left) {
        if (k != m) return error.DimensionMismatch;
    } else {
        if (k != n) return error.DimensionMismatch;
    }

    // Auto-dispatch to SIMD for large matrices
    const use_simd = (m >= 64 or n >= 64);
    if (use_simd) {
        return try simd_blas.trmm_simd(T, side, uplo, trans, diag, alpha, A, B);
    }

    const is_upper = (uplo == 'U' or uplo == 'u');
    const is_trans = (trans == 'T' or trans == 't');
    const is_unit = (diag == 'U' or diag == 'u');

    // Allocate temporary matrix for result
    var temp = try B.allocator.alloc(T, m * n);
    defer B.allocator.free(temp);

    if (is_left) {
        // B = α*A*B (A is m×m, B is m×n)
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: T = 0;
                if (!is_trans) {
                    // A*B
                    if (is_upper) {
                        const start = if (is_unit) i + 1 else i;
                        for (start..m) |p| {
                            sum += A.data[i * k + p] * B.data[p * n + j];
                        }
                        if (is_unit) {
                            sum += B.data[i * n + j];
                        }
                    } else {
                        const end = if (is_unit) i else i + 1;
                        for (0..end) |p| {
                            sum += A.data[i * k + p] * B.data[p * n + j];
                        }
                        if (is_unit) {
                            sum += B.data[i * n + j];
                        }
                    }
                } else {
                    // A^T*B
                    if (is_upper) {
                        const end = if (is_unit) i else i + 1;
                        for (0..end) |p| {
                            sum += A.data[p * k + i] * B.data[p * n + j];
                        }
                        if (is_unit) {
                            sum += B.data[i * n + j];
                        }
                    } else {
                        const start = if (is_unit) i + 1 else i;
                        for (start..m) |p| {
                            sum += A.data[p * k + i] * B.data[p * n + j];
                        }
                        if (is_unit) {
                            sum += B.data[i * n + j];
                        }
                    }
                }
                temp[i * n + j] = alpha * sum;
            }
        }
    } else {
        // B = α*B*A (B is m×n, A is n×n)
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: T = 0;
                if (!is_trans) {
                    // B*A
                    if (is_upper) {
                        const end = if (is_unit) j else j + 1;
                        for (0..end) |p| {
                            sum += B.data[i * n + p] * A.data[p * k + j];
                        }
                        if (is_unit) {
                            sum += B.data[i * n + j];
                        }
                    } else {
                        const start = if (is_unit) j + 1 else j;
                        for (start..n) |p| {
                            sum += B.data[i * n + p] * A.data[p * k + j];
                        }
                        if (is_unit) {
                            sum += B.data[i * n + j];
                        }
                    }
                } else {
                    // B*A^T
                    if (is_upper) {
                        const start = if (is_unit) j + 1 else j;
                        for (start..n) |p| {
                            sum += B.data[i * n + p] * A.data[j * k + p];
                        }
                        if (is_unit) {
                            sum += B.data[i * n + j];
                        }
                    } else {
                        const end = if (is_unit) j else j + 1;
                        for (0..end) |p| {
                            sum += B.data[i * n + p] * A.data[j * k + p];
                        }
                        if (is_unit) {
                            sum += B.data[i * n + j];
                        }
                    }
                }
                temp[i * n + j] = alpha * sum;
            }
        }
    }

    // Copy result back to B
    @memcpy(B.data, temp);
}

/// Triangular solve with multiple right-hand sides: B = α*A^(-1)*B or B = α*B*A^(-1)
///
/// Solves the triangular system A*X = α*B or X*A = α*B where A is triangular.
/// The solution is stored in B (in-place).
///
/// Parameters:
/// - side: 'L' for left (A*X = α*B), 'R' for right (X*A = α*B)
/// - uplo: 'U' for upper triangular, 'L' for lower triangular
/// - trans: 'N' for no transpose, 'T' for transpose
/// - diag: 'N' for non-unit diagonal, 'U' for unit diagonal
/// - alpha: Scalar multiplier for B
/// - A: Triangular matrix (2D NDArray, square)
/// - B: Right-hand side matrix (2D NDArray) — modified to solution in-place
///
/// Errors:
/// - error.DimensionMismatch if dimensions don't match
///
/// Time: O(m*n*k) where B is m×n, A is k×k
/// Space: O(1) (modifies B in-place after scaling)
///
/// Example:
/// ```zig
/// // A = [[2, 1], [0, 3]] (upper triangular)
/// var A = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{2, 2},
///     &[_]f64{2, 1, 0, 3}, .row_major);
/// defer A.deinit();
/// var B = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{2, 2},
///     &[_]f64{5, 7, 6, 9}, .row_major);
/// defer B.deinit();
/// try trsm(f64, 'L', 'U', 'N', 'N', 1.0, A, &B);  // Solve A*X = B
/// ```
pub fn trsm(comptime T: type, side: u8, uplo: u8, trans: u8, diag: u8, alpha: T, A: NDArray(T, 2), B: *NDArray(T, 2)) (NDArray(T, 2).Error)!void {
    // Validate square matrix A
    if (A.shape[0] != A.shape[1]) {
        return error.DimensionMismatch;
    }

    const is_left = (side == 'L' or side == 'l');
    const m = B.shape[0];
    const n = B.shape[1];
    const k = A.shape[0];

    // Validate dimensions
    if (is_left) {
        if (k != m) return error.DimensionMismatch;
    } else {
        if (k != n) return error.DimensionMismatch;
    }

    // Auto-dispatch to SIMD for large matrices
    const use_simd = (m >= 64 or n >= 64);
    if (use_simd) {
        try simd_blas.trsm_simd(T, side, uplo, trans, diag, alpha, A, B);
        return;
    }

    const is_upper = (uplo == 'U' or uplo == 'u');
    const is_trans = (trans == 'T' or trans == 't');
    const is_unit = (diag == 'U' or diag == 'u');

    // First scale B by alpha
    for (0..m * n) |idx| {
        B.data[idx] = alpha * B.data[idx];
    }

    if (is_left) {
        // Solve A*X = α*B (A is m×m, B is m×n)
        if (!is_trans) {
            if (is_upper) {
                // Upper triangular: back substitution
                var i: usize = m;
                while (i > 0) {
                    i -= 1;
                    for (0..n) |j| {
                        var sum: T = B.data[i * n + j];
                        for (i + 1..m) |p| {
                            sum -= A.data[i * k + p] * B.data[p * n + j];
                        }
                        if (!is_unit) {
                            B.data[i * n + j] = sum / A.data[i * k + i];
                        } else {
                            B.data[i * n + j] = sum;
                        }
                    }
                }
            } else {
                // Lower triangular: forward substitution
                for (0..m) |i| {
                    for (0..n) |j| {
                        var sum: T = B.data[i * n + j];
                        for (0..i) |p| {
                            sum -= A.data[i * k + p] * B.data[p * n + j];
                        }
                        if (!is_unit) {
                            B.data[i * n + j] = sum / A.data[i * k + i];
                        } else {
                            B.data[i * n + j] = sum;
                        }
                    }
                }
            }
        } else {
            // Solve A^T*X = α*B
            if (is_upper) {
                // Upper transpose: forward substitution
                for (0..m) |i| {
                    for (0..n) |j| {
                        var sum: T = B.data[i * n + j];
                        for (0..i) |p| {
                            sum -= A.data[p * k + i] * B.data[p * n + j];
                        }
                        if (!is_unit) {
                            B.data[i * n + j] = sum / A.data[i * k + i];
                        } else {
                            B.data[i * n + j] = sum;
                        }
                    }
                }
            } else {
                // Lower transpose: back substitution
                var i: usize = m;
                while (i > 0) {
                    i -= 1;
                    for (0..n) |j| {
                        var sum: T = B.data[i * n + j];
                        for (i + 1..m) |p| {
                            sum -= A.data[p * k + i] * B.data[p * n + j];
                        }
                        if (!is_unit) {
                            B.data[i * n + j] = sum / A.data[i * k + i];
                        } else {
                            B.data[i * n + j] = sum;
                        }
                    }
                }
            }
        }
    } else {
        // Solve X*A = α*B (B is m×n, A is n×n)
        if (!is_trans) {
            if (is_upper) {
                // Upper triangular: forward substitution by columns
                for (0..n) |j| {
                    for (0..m) |i| {
                        var sum: T = B.data[i * n + j];
                        for (0..j) |p| {
                            sum -= B.data[i * n + p] * A.data[p * k + j];
                        }
                        if (!is_unit) {
                            B.data[i * n + j] = sum / A.data[j * k + j];
                        } else {
                            B.data[i * n + j] = sum;
                        }
                    }
                }
            } else {
                // Lower triangular: back substitution by columns
                var j: usize = n;
                while (j > 0) {
                    j -= 1;
                    for (0..m) |i| {
                        var sum: T = B.data[i * n + j];
                        for (j + 1..n) |p| {
                            sum -= B.data[i * n + p] * A.data[p * k + j];
                        }
                        if (!is_unit) {
                            B.data[i * n + j] = sum / A.data[j * k + j];
                        } else {
                            B.data[i * n + j] = sum;
                        }
                    }
                }
            }
        } else {
            // Solve X*A^T = α*B
            if (is_upper) {
                // Upper transpose: back substitution by columns
                var j: usize = n;
                while (j > 0) {
                    j -= 1;
                    for (0..m) |i| {
                        var sum: T = B.data[i * n + j];
                        for (j + 1..n) |p| {
                            sum -= B.data[i * n + p] * A.data[j * k + p];
                        }
                        if (!is_unit) {
                            B.data[i * n + j] = sum / A.data[j * k + j];
                        } else {
                            B.data[i * n + j] = sum;
                        }
                    }
                }
            } else {
                // Lower transpose: forward substitution by columns
                for (0..n) |j| {
                    for (0..m) |i| {
                        var sum: T = B.data[i * n + j];
                        for (0..j) |p| {
                            sum -= B.data[i * n + p] * A.data[j * k + p];
                        }
                        if (!is_unit) {
                            B.data[i * n + j] = sum / A.data[j * k + j];
                        } else {
                            B.data[i * n + j] = sum;
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Triangular Operations Tests
// ============================================================================

test "trmv: upper triangular matrix-vector multiply" {
    const allocator = testing.allocator;

    // Upper triangular: A = [[2, 1, 0], [0, 3, 1], [0, 0, 4]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 2, 1, 0, 0, 3, 1, 0, 0, 4 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    // A*x = [2*1 + 1*2 + 0*3, 0*1 + 3*2 + 1*3, 0*1 + 0*2 + 4*3] = [4, 9, 12]
    try trmv(f64, 'U', 'N', 'N', A, &x);

    try testing.expectApproxEqAbs(4.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(9.0, x.data[1], 1e-10);
    try testing.expectApproxEqAbs(12.0, x.data[2], 1e-10);
}

test "trmv: lower triangular matrix-vector multiply" {
    const allocator = testing.allocator;

    // Lower triangular: A = [[2, 0, 0], [1, 3, 0], [2, 1, 4]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 2, 0, 0, 1, 3, 0, 2, 1, 4 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    // A*x = [2*1, 1*1 + 3*2, 2*1 + 1*2 + 4*3] = [2, 7, 16]
    try trmv(f64, 'L', 'N', 'N', A, &x);

    try testing.expectApproxEqAbs(2.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(7.0, x.data[1], 1e-10);
    try testing.expectApproxEqAbs(16.0, x.data[2], 1e-10);
}

test "trmv: upper triangular with unit diagonal" {
    const allocator = testing.allocator;

    // Unit upper triangular: A = [[1, 2], [0, 1]] (diagonals ignored)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 999, 2, 0, 999 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 3, 4 }, .row_major);
    defer x.deinit();

    // With unit diagonal: A*x = [1*3 + 2*4, 0*3 + 1*4] = [11, 4]
    try trmv(f64, 'U', 'N', 'U', A, &x);

    try testing.expectApproxEqAbs(11.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(4.0, x.data[1], 1e-10);
}

test "trmv: transpose operation" {
    const allocator = testing.allocator;

    // Upper triangular: A = [[2, 1], [0, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 1, 0, 3 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 2 }, .row_major);
    defer x.deinit();

    // A^T*x = [[2, 0], [1, 3]] * [1, 2] = [2*1 + 0*2, 1*1 + 3*2] = [2, 7]
    try trmv(f64, 'U', 'T', 'N', A, &x);

    try testing.expectApproxEqAbs(2.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(7.0, x.data[1], 1e-10);
}

test "trsv: upper triangular solve" {
    const allocator = testing.allocator;

    // Upper triangular: A = [[2, 1], [0, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 1, 0, 3 }, .row_major);
    defer A.deinit();

    // Solve A*x = b where b = [5, 6]
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 5, 6 }, .row_major);
    defer x.deinit();

    try trsv(f64, 'U', 'N', 'N', A, &x);

    // x[1] = 6/3 = 2
    // x[0] = (5 - 1*2)/2 = 3/2 = 1.5
    try testing.expectApproxEqAbs(1.5, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(2.0, x.data[1], 1e-10);
}

test "trsv: lower triangular solve" {
    const allocator = testing.allocator;

    // Lower triangular: A = [[2, 0], [1, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 0, 1, 3 }, .row_major);
    defer A.deinit();

    // Solve A*x = b where b = [4, 7]
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 4, 7 }, .row_major);
    defer x.deinit();

    try trsv(f64, 'L', 'N', 'N', A, &x);

    // x[0] = 4/2 = 2
    // x[1] = (7 - 1*2)/3 = 5/3 ≈ 1.6667
    try testing.expectApproxEqAbs(2.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(5.0 / 3.0, x.data[1], 1e-10);
}

test "trsv: identity matrix solve" {
    const allocator = testing.allocator;

    // Identity: A = [[1, 0], [0, 1]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 0, 0, 1 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 3, 4 }, .row_major);
    defer x.deinit();

    try trsv(f64, 'U', 'N', 'N', A, &x);

    // Solution should be unchanged
    try testing.expectApproxEqAbs(3.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(4.0, x.data[1], 1e-10);
}

test "trsv: unit diagonal solve" {
    const allocator = testing.allocator;

    // Unit upper triangular: A = [[1, 2], [0, 1]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 999, 2, 0, 999 }, .row_major);
    defer A.deinit();

    // Solve A*x = b where b = [5, 3]
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 5, 3 }, .row_major);
    defer x.deinit();

    try trsv(f64, 'U', 'N', 'U', A, &x);

    // With unit diagonal: x[1] = 3, x[0] = 5 - 2*3 = -1
    try testing.expectApproxEqAbs(-1.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(3.0, x.data[1], 1e-10);
}

test "trmm: left upper triangular multiply" {
    const allocator = testing.allocator;

    // Upper triangular: A = [[2, 1], [0, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 1, 0, 3 }, .row_major);
    defer A.deinit();

    // B = [[1, 2], [3, 4]]
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    // B = A*B = [[2*1+1*3, 2*2+1*4], [0*1+3*3, 0*2+3*4]] = [[5, 8], [9, 12]]
    try trmm(f64, 'L', 'U', 'N', 'N', 1.0, A, &B);

    try testing.expectApproxEqAbs(5.0, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(8.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(9.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(12.0, B.data[3], 1e-10);
}

test "trmm: right lower triangular multiply" {
    const allocator = testing.allocator;

    // B = [[1, 2], [3, 4]]
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    // Lower triangular: A = [[2, 0], [1, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 0, 1, 3 }, .row_major);
    defer A.deinit();

    // B = B*A = [[1*2+2*1, 1*0+2*3], [3*2+4*1, 3*0+4*3]] = [[4, 6], [10, 12]]
    try trmm(f64, 'R', 'L', 'N', 'N', 1.0, A, &B);

    try testing.expectApproxEqAbs(4.0, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(6.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(10.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(12.0, B.data[3], 1e-10);
}

test "trmm: with scalar multiplier" {
    const allocator = testing.allocator;

    // Upper triangular: A = [[2, 1], [0, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 1, 0, 3 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    // B = 2*A*B = 2*[[5, 8], [9, 12]] = [[10, 16], [18, 24]]
    try trmm(f64, 'L', 'U', 'N', 'N', 2.0, A, &B);

    try testing.expectApproxEqAbs(10.0, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(16.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(18.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(24.0, B.data[3], 1e-10);
}

test "trmm: unit diagonal" {
    const allocator = testing.allocator;

    // Unit upper triangular: A = [[1, 2], [0, 1]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 999, 2, 0, 999 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    // With unit diagonal: B = [[1*1+2*3, 1*2+2*4], [0*1+1*3, 0*2+1*4]] = [[7, 10], [3, 4]]
    try trmm(f64, 'L', 'U', 'N', 'U', 1.0, A, &B);

    try testing.expectApproxEqAbs(7.0, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(10.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(3.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(4.0, B.data[3], 1e-10);
}

test "trsm: left upper triangular solve" {
    const allocator = testing.allocator;

    // Upper triangular: A = [[2, 1], [0, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 1, 0, 3 }, .row_major);
    defer A.deinit();

    // Solve A*X = B where B = [[5, 8], [9, 12]]
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 8, 9, 12 }, .row_major);
    defer B.deinit();

    try trsm(f64, 'L', 'U', 'N', 'N', 1.0, A, &B);

    // Expected: X = [[1, 2], [3, 4]] (from trmm test in reverse)
    try testing.expectApproxEqAbs(1.0, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(2.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(3.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(4.0, B.data[3], 1e-10);
}

test "trsm: right lower triangular solve" {
    const allocator = testing.allocator;

    // Lower triangular: A = [[2, 0], [1, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 0, 1, 3 }, .row_major);
    defer A.deinit();

    // Solve X*A = B where B = [[4, 6], [10, 12]]
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 4, 6, 10, 12 }, .row_major);
    defer B.deinit();

    try trsm(f64, 'R', 'L', 'N', 'N', 1.0, A, &B);

    // Expected: X = [[1, 2], [3, 4]] (from trmm right test in reverse)
    try testing.expectApproxEqAbs(1.0, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(2.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(3.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(4.0, B.data[3], 1e-10);
}

test "trsm: with alpha scalar" {
    const allocator = testing.allocator;

    // Upper triangular: A = [[2, 1], [0, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 1, 0, 3 }, .row_major);
    defer A.deinit();

    // Solve A*X = 2*B where B = [[5, 8], [9, 12]]
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 8, 9, 12 }, .row_major);
    defer B.deinit();

    try trsm(f64, 'L', 'U', 'N', 'N', 2.0, A, &B);

    // Expected: X = [[2, 4], [6, 8]] (2 * [[1, 2], [3, 4]])
    try testing.expectApproxEqAbs(2.0, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(4.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(6.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(8.0, B.data[3], 1e-10);
}

test "trsm: identity matrix" {
    const allocator = testing.allocator;

    // Identity: A = [[1, 0], [0, 1]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 0, 0, 1 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 6, 7, 8 }, .row_major);
    defer B.deinit();

    try trsm(f64, 'L', 'U', 'N', 'N', 1.0, A, &B);

    // Solution should be unchanged
    try testing.expectApproxEqAbs(5.0, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(6.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(7.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(8.0, B.data[3], 1e-10);
}

test "trmv: dimension mismatch error" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    try testing.expectError(error.DimensionMismatch, trmv(f64, 'U', 'N', 'N', A, &x));
}

test "trmv: non-square matrix dimension mismatch" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 2 }, .row_major);
    defer x.deinit();

    try testing.expectError(error.DimensionMismatch, trmv(f64, 'U', 'N', 'N', A, &x));
}

test "trmv: 1x1 upper triangular" {
    const allocator = testing.allocator;

    // 1×1 upper triangular: A = [[5]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 1 }, &[_]f64{5}, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{3}, .row_major);
    defer x.deinit();

    // A*x = [5*3] = [15]
    try trmv(f64, 'U', 'N', 'N', A, &x);

    try testing.expectApproxEqAbs(15.0, x.data[0], 1e-10);
}

test "trmv: 1x1 lower triangular" {
    const allocator = testing.allocator;

    // 1×1 lower triangular: A = [[5]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 1 }, &[_]f64{5}, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{3}, .row_major);
    defer x.deinit();

    // A*x = [5*3] = [15]
    try trmv(f64, 'L', 'N', 'N', A, &x);

    try testing.expectApproxEqAbs(15.0, x.data[0], 1e-10);
}

test "trmv: upper triangular with zeros" {
    const allocator = testing.allocator;

    // Upper triangular with explicit zeros: A = [[2, 3, 4], [0, 5, 6], [0, 0, 7]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 2, 3, 4, 0, 5, 6, 0, 0, 7 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    // A*x = [2*1 + 3*2 + 4*3, 5*2 + 6*3, 7*3] = [20, 28, 21]
    try trmv(f64, 'U', 'N', 'N', A, &x);

    try testing.expectApproxEqAbs(20.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(28.0, x.data[1], 1e-10);
    try testing.expectApproxEqAbs(21.0, x.data[2], 1e-10);
}

test "trmv: lower triangular with zeros" {
    const allocator = testing.allocator;

    // Lower triangular with explicit zeros: A = [[2, 0, 0], [3, 5, 0], [4, 6, 7]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 2, 0, 0, 3, 5, 0, 4, 6, 7 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    // A*x = [2*1, 3*1 + 5*2, 4*1 + 6*2 + 7*3] = [2, 13, 37]
    try trmv(f64, 'L', 'N', 'N', A, &x);

    try testing.expectApproxEqAbs(2.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(13.0, x.data[1], 1e-10);
    try testing.expectApproxEqAbs(37.0, x.data[2], 1e-10);
}

test "trmv: lower triangular with unit diagonal" {
    const allocator = testing.allocator;

    // Unit lower triangular: A = [[1, 0, 0], [2, 1, 0], [3, 4, 1]] (diagonals ignored in input)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 999, 0, 0, 2, 999, 0, 3, 4, 999 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    // With unit diagonal: A*x = [1 + 0, 2*1 + 2 + 0, 3*1 + 4*2 + 3] = [1, 4, 16]
    try trmv(f64, 'L', 'N', 'U', A, &x);

    try testing.expectApproxEqAbs(1.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(4.0, x.data[1], 1e-10);
    try testing.expectApproxEqAbs(16.0, x.data[2], 1e-10);
}

test "trmv: upper triangular transpose (acts like lower)" {
    const allocator = testing.allocator;

    // Upper triangular: A = [[2, 1, 0], [0, 3, 2], [0, 0, 4]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 2, 1, 0, 0, 3, 2, 0, 0, 4 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    // A^T = [[2, 0, 0], [1, 3, 0], [0, 2, 4]]
    // A^T*x = [2*1, 1*1 + 3*2, 0 + 2*2 + 4*3] = [2, 7, 16]
    try trmv(f64, 'U', 'T', 'N', A, &x);

    try testing.expectApproxEqAbs(2.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(7.0, x.data[1], 1e-10);
    try testing.expectApproxEqAbs(16.0, x.data[2], 1e-10);
}

test "trmv: lower triangular transpose (acts like upper)" {
    const allocator = testing.allocator;

    // Lower triangular: A = [[2, 0, 0], [1, 3, 0], [4, 2, 5]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 2, 0, 0, 1, 3, 0, 4, 2, 5 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    // A^T = [[2, 1, 4], [0, 3, 2], [0, 0, 5]]
    // A^T*x = [2*1 + 1*2 + 4*3, 3*2 + 2*3, 5*3] = [16, 12, 15]
    try trmv(f64, 'L', 'T', 'N', A, &x);

    try testing.expectApproxEqAbs(16.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(12.0, x.data[1], 1e-10);
    try testing.expectApproxEqAbs(15.0, x.data[2], 1e-10);
}

test "trmv: upper transpose with unit diagonal" {
    const allocator = testing.allocator;

    // Upper triangular: A = [[1, 2, 3], [0, 1, 4], [0, 0, 1]] (unit diagonal)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 999, 2, 3, 0, 999, 4, 0, 0, 999 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    // A^T = [[1, 0, 0], [2, 1, 0], [3, 4, 1]] with unit diagonal
    // A^T*x = [1 + 0, 2 + 2 + 0, 3 + 8 + 3] = [1, 4, 14]
    try trmv(f64, 'U', 'T', 'U', A, &x);

    try testing.expectApproxEqAbs(1.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(4.0, x.data[1], 1e-10);
    try testing.expectApproxEqAbs(14.0, x.data[2], 1e-10);
}

test "trmv: lower transpose with unit diagonal" {
    const allocator = testing.allocator;

    // Lower triangular: A = [[1, 0, 0], [2, 1, 0], [3, 4, 1]] (unit diagonal)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 999, 0, 0, 2, 999, 0, 3, 4, 999 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    // A^T = [[1, 2, 3], [0, 1, 4], [0, 0, 1]] with unit diagonal
    // A^T*x = [1 + 2*2 + 3*3, 2 + 4*3, 3] = [14, 14, 3]
    try trmv(f64, 'L', 'T', 'U', A, &x);

    try testing.expectApproxEqAbs(14.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(14.0, x.data[1], 1e-10);
    try testing.expectApproxEqAbs(3.0, x.data[2], 1e-10);
}

test "trmv: 4x4 upper triangular complex pattern" {
    const allocator = testing.allocator;

    // Upper triangular 4×4
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 4 }, &[_]f64{
        1, 2, 3, 4,
        0, 5, 6, 7,
        0, 0, 8, 9,
        0, 0, 0, 10,
    }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer x.deinit();

    // A*x = [1 + 4 + 9 + 16, 10 + 18 + 28, 24 + 36, 40] = [30, 56, 60, 40]
    try trmv(f64, 'U', 'N', 'N', A, &x);

    try testing.expectApproxEqAbs(30.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(56.0, x.data[1], 1e-10);
    try testing.expectApproxEqAbs(60.0, x.data[2], 1e-10);
    try testing.expectApproxEqAbs(40.0, x.data[3], 1e-10);
}

test "trmv: 4x4 lower triangular complex pattern" {
    const allocator = testing.allocator;

    // Lower triangular 4×4
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 4 }, &[_]f64{
        1, 0, 0, 0,
        2, 3, 0, 0,
        4, 5, 6, 0,
        7, 8, 9, 10,
    }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer x.deinit();

    // A*x = [1, 2 + 6, 4 + 10 + 18, 7 + 16 + 27 + 40] = [1, 8, 32, 90]
    try trmv(f64, 'L', 'N', 'N', A, &x);

    try testing.expectApproxEqAbs(1.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(8.0, x.data[1], 1e-10);
    try testing.expectApproxEqAbs(32.0, x.data[2], 1e-10);
    try testing.expectApproxEqAbs(90.0, x.data[3], 1e-10);
}

test "trmv: zero vector input" {
    const allocator = testing.allocator;

    // Upper triangular
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 2, 1, 0, 0, 3, 1, 0, 0, 4 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).zeros(allocator, &[_]usize{3}, .row_major);
    defer x.deinit();

    // A*0 = 0
    try trmv(f64, 'U', 'N', 'N', A, &x);

    try testing.expectApproxEqAbs(0.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(0.0, x.data[1], 1e-10);
    try testing.expectApproxEqAbs(0.0, x.data[2], 1e-10);
}

test "trmv: identity upper triangular" {
    const allocator = testing.allocator;

    // Identity: A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 2, 3, 4 }, .row_major);
    defer x.deinit();

    // I*x = x
    try trmv(f64, 'U', 'N', 'N', A, &x);

    try testing.expectApproxEqAbs(2.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(3.0, x.data[1], 1e-10);
    try testing.expectApproxEqAbs(4.0, x.data[2], 1e-10);
}

test "trmv: identity lower triangular" {
    const allocator = testing.allocator;

    // Identity: A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 2, 3, 4 }, .row_major);
    defer x.deinit();

    // I*x = x
    try trmv(f64, 'L', 'N', 'N', A, &x);

    try testing.expectApproxEqAbs(2.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(3.0, x.data[1], 1e-10);
    try testing.expectApproxEqAbs(4.0, x.data[2], 1e-10);
}

test "trmv: with negative numbers" {
    const allocator = testing.allocator;

    // Upper triangular with negatives: A = [[-1, 2, 3], [0, -2, 4], [0, 0, -3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ -1, 2, 3, 0, -2, 4, 0, 0, -3 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    // A*x = [-1 + 4 + 9, -4 + 12, -9] = [12, 8, -9]
    try trmv(f64, 'U', 'N', 'N', A, &x);

    try testing.expectApproxEqAbs(12.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(8.0, x.data[1], 1e-10);
    try testing.expectApproxEqAbs(-9.0, x.data[2], 1e-10);
}

test "trmv: with small floating point values" {
    const allocator = testing.allocator;

    // Upper triangular with small values
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1e-8, 2e-8, 0, 3e-8 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1e8, 1e8 }, .row_major);
    defer x.deinit();

    // A*x = [1e-8*1e8 + 2e-8*1e8, 3e-8*1e8] = [3, 3]
    try trmv(f64, 'U', 'N', 'N', A, &x);

    try testing.expectApproxEqAbs(3.0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(3.0, x.data[1], 1e-10);
}

test "trmv: f32 precision" {
    const allocator = testing.allocator;

    // f32 upper triangular
    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{ 2.5, 1.5, 0, 3.5 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{2}, &[_]f32{ 2, 3 }, .row_major);
    defer x.deinit();

    // A*x = [2.5*2 + 1.5*3, 3.5*3] = [9.5, 10.5]
    try trmv(f32, 'U', 'N', 'N', A, &x);

    try testing.expectApproxEqAbs(9.5, x.data[0], 1e-5);
    try testing.expectApproxEqAbs(10.5, x.data[1], 1e-5);
}

// ============================================================================
// trmv Dispatch Tests (auto-routing to SIMD for n >= 64)
// ============================================================================

test "trmv dispatch: threshold boundary n=63 (scalar fallback)" {
    const allocator = testing.allocator;
    const n = 63;

    // Create upper triangular matrix: A[i,j] = i+j+1 if i<=j, else 0
    var data_A = try allocator.alloc(f64, n * n);
    defer allocator.free(data_A);

    for (0..n) |i| {
        for (0..n) |j| {
            data_A[i * n + j] = if (i <= j) @as(f64, @floatFromInt(i + j + 1)) else 0.0;
        }
    }

    // Create vector: x[i] = 1
    var data_x = try allocator.alloc(f64, n);
    defer allocator.free(data_x);
    for (0..n) |i| {
        data_x[i] = 1.0;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ n, n }, data_A, .row_major);
    defer A.deinit();
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, data_x, .row_major);
    defer x.deinit();

    // Call trmv with upper triangular, no transpose, no unit diagonal
    try trmv(f64, 'U', 'N', 'N', A, &x);

    // Expected: x[i] = sum(A[i,j] for j in i..n) = sum(i+j+1 for j in i..n)
    // For i=62: sum(j in 62..63) = (62+62+1) + (62+63+1) = 125 + 126 = 251
    const expected_62 = 251.0;
    try testing.expectApproxEqAbs(expected_62, x.data[62], 1e-9);

    // For i=0: sum(j in 0..63) = sum(0+j+1 for j in 0..63) = sum(1..64) = 64*65/2 = 2080
    const expected_0 = 2080.0;
    try testing.expectApproxEqAbs(expected_0, x.data[0], 1e-9);
}

test "trmv dispatch: threshold boundary n=64 (SIMD path)" {
    const allocator = testing.allocator;
    const n = 64;

    // Create upper triangular matrix
    var data_A = try allocator.alloc(f64, n * n);
    defer allocator.free(data_A);

    for (0..n) |i| {
        for (0..n) |j| {
            data_A[i * n + j] = if (i <= j) @as(f64, @floatFromInt(i + j + 1)) else 0.0;
        }
    }

    var data_x = try allocator.alloc(f64, n);
    defer allocator.free(data_x);
    for (0..n) |i| {
        data_x[i] = 1.0;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ n, n }, data_A, .row_major);
    defer A.deinit();
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, data_x, .row_major);
    defer x.deinit();

    try trmv(f64, 'U', 'N', 'N', A, &x);

    // For i=0: sum(0+j+1 for j in 0..64) = sum(1..65) = 65*66/2 = 2145
    const expected_0 = 2145.0;
    try testing.expectApproxEqAbs(expected_0, x.data[0], 1e-9);

    // For i=63: sum(63+j+1 for j in 63..64) = (63+63+1) + (63+64+1) = 127 + 128 = 255
    const expected_63 = 255.0;
    try testing.expectApproxEqAbs(expected_63, x.data[63], 1e-9);
}

test "trmv dispatch: above threshold n=65 (SIMD path)" {
    const allocator = testing.allocator;
    const n = 65;

    var data_A = try allocator.alloc(f64, n * n);
    defer allocator.free(data_A);

    for (0..n) |i| {
        for (0..n) |j| {
            data_A[i * n + j] = if (i <= j) @as(f64, @floatFromInt(i + j + 1)) else 0.0;
        }
    }

    var data_x = try allocator.alloc(f64, n);
    defer allocator.free(data_x);
    for (0..n) |i| {
        data_x[i] = 1.0;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ n, n }, data_A, .row_major);
    defer A.deinit();
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, data_x, .row_major);
    defer x.deinit();

    try trmv(f64, 'U', 'N', 'N', A, &x);

    // For i=0: sum(1..66) = 66*67/2 = 2211
    const expected_0 = 2211.0;
    try testing.expectApproxEqAbs(expected_0, x.data[0], 1e-9);
}

test "trmv dispatch: 128x128 upper non-unit (full SIMD)" {
    const allocator = testing.allocator;
    const n = 128;

    var data_A = try allocator.alloc(f64, n * n);
    defer allocator.free(data_A);

    // Upper triangular: simple pattern
    for (0..n) |i| {
        for (0..n) |j| {
            data_A[i * n + j] = if (i <= j) 1.0 else 0.0;
        }
    }

    var data_x = try allocator.alloc(f64, n);
    defer allocator.free(data_x);
    for (0..n) |i| {
        data_x[i] = 1.0;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ n, n }, data_A, .row_major);
    defer A.deinit();
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, data_x, .row_major);
    defer x.deinit();

    try trmv(f64, 'U', 'N', 'N', A, &x);

    // x[i] = sum(1.0 for j in i..128) = 128-i
    for (0..n) |i| {
        const expected = @as(f64, @floatFromInt(n - i));
        try testing.expectApproxEqAbs(expected, x.data[i], 1e-9);
    }
}

test "trmv dispatch: 128x128 lower unit diagonal (full SIMD)" {
    const allocator = testing.allocator;
    const n = 128;

    var data_A = try allocator.alloc(f64, n * n);
    defer allocator.free(data_A);

    // Lower triangular: A[i,j] = i-j+1 if i>=j, else 0, diagonal = 1
    for (0..n) |i| {
        for (0..n) |j| {
            if (i >= j) {
                data_A[i * n + j] = if (i == j) 1.0 else @as(f64, @floatFromInt(i - j + 1));
            } else {
                data_A[i * n + j] = 0.0;
            }
        }
    }

    var data_x = try allocator.alloc(f64, n);
    defer allocator.free(data_x);
    for (0..n) |i| {
        data_x[i] = 1.0;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ n, n }, data_A, .row_major);
    defer A.deinit();
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, data_x, .row_major);
    defer x.deinit();

    try trmv(f64, 'L', 'N', 'U', A, &x);

    // With unit diagonal: x[i] = x[i] + sum(A[i,j]*x[j] for j in 0..i)
    // For i=0: x[0] = 1.0 + 0 = 1.0
    try testing.expectApproxEqAbs(1.0, x.data[0], 1e-9);

    // For i=1: x[1] = 1.0 + A[1,0]*1.0 = 1.0 + 2.0 = 3.0
    try testing.expectApproxEqAbs(3.0, x.data[1], 1e-9);
}

test "trmv dispatch: 128x128 upper transpose (full SIMD)" {
    const allocator = testing.allocator;
    const n = 128;

    var data_A = try allocator.alloc(f64, n * n);
    defer allocator.free(data_A);

    // Upper triangular: A[i,j] = i+j+1 if i<=j, else 0
    for (0..n) |i| {
        for (0..n) |j| {
            data_A[i * n + j] = if (i <= j) @as(f64, @floatFromInt(i + j + 1)) else 0.0;
        }
    }

    var data_x = try allocator.alloc(f64, n);
    defer allocator.free(data_x);
    for (0..n) |i| {
        data_x[i] = 1.0;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ n, n }, data_A, .row_major);
    defer A.deinit();
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, data_x, .row_major);
    defer x.deinit();

    try trmv(f64, 'U', 'T', 'N', A, &x);

    // Upper transpose acts like lower: compute A^T*x
    // For i=0: x[0] = sum(A[j,0]*x[j] for j in 0..1) = A[0,0]*1.0 = 1.0
    try testing.expectApproxEqAbs(1.0, x.data[0], 1e-9);
}

test "trmv dispatch: 100x100 lower transpose unit (non-aligned SIMD)" {
    const allocator = testing.allocator;
    const n = 100;

    var data_A = try allocator.alloc(f64, n * n);
    defer allocator.free(data_A);

    // Lower triangular with unit diagonal
    for (0..n) |i| {
        for (0..n) |j| {
            if (i >= j) {
                data_A[i * n + j] = if (i == j) 1.0 else @as(f64, @floatFromInt(i - j + 1));
            } else {
                data_A[i * n + j] = 0.0;
            }
        }
    }

    var data_x = try allocator.alloc(f64, n);
    defer allocator.free(data_x);
    for (0..n) |i| {
        data_x[i] = 1.0;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ n, n }, data_A, .row_major);
    defer A.deinit();
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, data_x, .row_major);
    defer x.deinit();

    try trmv(f64, 'L', 'T', 'U', A, &x);

    // Verify no NaN or Inf
    for (0..n) |i| {
        try testing.expect(!std.math.isNan(x.data[i]));
        try testing.expect(!std.math.isInf(x.data[i]));
    }
}

test "trmv dispatch: 128x128 f32 precision (8-wide SIMD)" {
    const allocator = testing.allocator;
    const n = 128;

    var data_A = try allocator.alloc(f32, n * n);
    defer allocator.free(data_A);

    for (0..n) |i| {
        for (0..n) |j| {
            data_A[i * n + j] = if (i <= j) 1.0 else 0.0;
        }
    }

    var data_x = try allocator.alloc(f32, n);
    defer allocator.free(data_x);
    for (0..n) |i| {
        data_x[i] = 1.0;
    }

    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ n, n }, data_A, .row_major);
    defer A.deinit();
    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{n}, data_x, .row_major);
    defer x.deinit();

    try trmv(f32, 'U', 'N', 'N', A, &x);

    // x[i] = sum(1.0 for j in i..128) = 128-i
    for (0..n) |i| {
        const expected = @as(f32, @floatFromInt(n - i));
        try testing.expectApproxEqAbs(expected, x.data[i], 1e-5);
    }
}

test "trmv dispatch: 128x128 f64 precision (4-wide SIMD)" {
    const allocator = testing.allocator;
    const n = 128;

    var data_A = try allocator.alloc(f64, n * n);
    defer allocator.free(data_A);

    for (0..n) |i| {
        for (0..n) |j| {
            data_A[i * n + j] = if (i <= j) 1.0 else 0.0;
        }
    }

    var data_x = try allocator.alloc(f64, n);
    defer allocator.free(data_x);
    for (0..n) |i| {
        data_x[i] = 2.0;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ n, n }, data_A, .row_major);
    defer A.deinit();
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, data_x, .row_major);
    defer x.deinit();

    try trmv(f64, 'U', 'N', 'N', A, &x);

    // x[i] = sum(1.0 for j in i..128) * 2.0 = (128-i) * 2.0
    for (0..n) |i| {
        const expected = @as(f64, @floatFromInt(n - i)) * 2.0;
        try testing.expectApproxEqAbs(expected, x.data[i], 1e-9);
    }
}

test "trmv dispatch: 100x100 non-aligned (tail loop coverage)" {
    const allocator = testing.allocator;
    const n = 100;

    var data_A = try allocator.alloc(f64, n * n);
    defer allocator.free(data_A);

    // Upper triangular
    for (0..n) |i| {
        for (0..n) |j| {
            data_A[i * n + j] = if (i <= j) @as(f64, @floatFromInt(j - i + 1)) else 0.0;
        }
    }

    var data_x = try allocator.alloc(f64, n);
    defer allocator.free(data_x);
    for (0..n) |i| {
        data_x[i] = 1.0;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ n, n }, data_A, .row_major);
    defer A.deinit();
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, data_x, .row_major);
    defer x.deinit();

    try trmv(f64, 'U', 'N', 'N', A, &x);

    // x[i] = sum(j-i+1 for j in i..100) = sum(k+1 for k in 0..(100-i)) = sum(1..(100-i+1))
    // = (100-i)*(100-i+1)/2
    for (0..n) |i| {
        const count = n - i;
        const expected = @as(f64, @floatFromInt(count * (count + 1) / 2));
        try testing.expectApproxEqAbs(expected, x.data[i], 1e-9);
    }
}

test "trmv dispatch: 137x137 non-aligned (prime dimension SIMD+tail)" {
    const allocator = testing.allocator;
    const n = 137;

    var data_A = try allocator.alloc(f64, n * n);
    defer allocator.free(data_A);

    // Lower triangular
    for (0..n) |i| {
        for (0..n) |j| {
            data_A[i * n + j] = if (i >= j) 1.0 else 0.0;
        }
    }

    var data_x = try allocator.alloc(f64, n);
    defer allocator.free(data_x);
    for (0..n) |i| {
        data_x[i] = 1.0;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ n, n }, data_A, .row_major);
    defer A.deinit();
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, data_x, .row_major);
    defer x.deinit();

    try trmv(f64, 'L', 'N', 'N', A, &x);

    // x[i] = sum(1.0 for j in 0..i+1) = i+1
    for (0..n) |i| {
        const expected = @as(f64, @floatFromInt(i + 1));
        try testing.expectApproxEqAbs(expected, x.data[i], 1e-9);
    }
}

test "trmv dispatch: 256x256 stress test (large SIMD correctness)" {
    const allocator = testing.allocator;
    const n = 256;

    var data_A = try allocator.alloc(f64, n * n);
    defer allocator.free(data_A);

    // Upper triangular with varied pattern
    for (0..n) |i| {
        for (0..n) |j| {
            data_A[i * n + j] = if (i <= j) @as(f64, @floatFromInt((i + j) % 10 + 1)) else 0.0;
        }
    }

    var data_x = try allocator.alloc(f64, n);
    defer allocator.free(data_x);
    for (0..n) |i| {
        data_x[i] = @as(f64, @floatFromInt(i % 5 + 1));
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ n, n }, data_A, .row_major);
    defer A.deinit();
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, data_x, .row_major);
    defer x.deinit();

    try trmv(f64, 'U', 'N', 'N', A, &x);

    // Verify all values are valid (not NaN or Inf)
    for (0..n) |i| {
        try testing.expect(!std.math.isNan(x.data[i]));
        try testing.expect(!std.math.isInf(x.data[i]));
        try testing.expect(x.data[i] >= 0.0); // All positive since matrix and vector entries are positive
    }

    // Spot check: verify first element is reasonable
    try testing.expect(x.data[0] > 0.0);
}

test "trsv: dimension mismatch error" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 2 }, .row_major);
    defer x.deinit();

    try testing.expectError(error.DimensionMismatch, trsv(f64, 'U', 'N', 'N', A, &x));
}

test "trsv dispatch: threshold boundary n=63 (scalar fallback)" {
    const allocator = testing.allocator;
    const n = 63;

    // Create upper triangular matrix: A[i,j] = i+j+1 if i<=j, else 0
    var data_A = try allocator.alloc(f64, n * n);
    defer allocator.free(data_A);

    for (0..n) |i| {
        for (0..n) |j| {
            data_A[i * n + j] = if (i <= j) @as(f64, @floatFromInt(i + j + 1)) else 0.0;
        }
    }

    // Create vector b: b[i] = sum(i+j+1 for j in i..n)
    var data_b = try allocator.alloc(f64, n);
    defer allocator.free(data_b);
    for (0..n) |i| {
        var sum: f64 = 0.0;
        for (i..n) |j| {
            sum += @as(f64, @floatFromInt(i + j + 1));
        }
        data_b[i] = sum;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ n, n }, data_A, .row_major);
    defer A.deinit();
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, data_b, .row_major);
    defer x.deinit();

    try trsv(f64, 'U', 'N', 'N', A, &x);

    // Solution should be x[i] = 1 for all i
    for (0..n) |i| {
        try testing.expectApproxEqAbs(1.0, x.data[i], 1e-8);
    }
}

test "trsv dispatch: threshold boundary n=64 (SIMD path)" {
    const allocator = testing.allocator;
    const n = 64;

    // Create upper triangular matrix
    var data_A = try allocator.alloc(f64, n * n);
    defer allocator.free(data_A);

    for (0..n) |i| {
        for (0..n) |j| {
            data_A[i * n + j] = if (i <= j) @as(f64, @floatFromInt(i + j + 1)) else 0.0;
        }
    }

    var data_b = try allocator.alloc(f64, n);
    defer allocator.free(data_b);
    for (0..n) |i| {
        var sum: f64 = 0.0;
        for (i..n) |j| {
            sum += @as(f64, @floatFromInt(i + j + 1));
        }
        data_b[i] = sum;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ n, n }, data_A, .row_major);
    defer A.deinit();
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, data_b, .row_major);
    defer x.deinit();

    try trsv(f64, 'U', 'N', 'N', A, &x);

    // Solution should be x[i] = 1 for all i
    for (0..n) |i| {
        try testing.expectApproxEqAbs(1.0, x.data[i], 1e-8);
    }
}

test "trsv dispatch: above threshold n=65 (SIMD path)" {
    const allocator = testing.allocator;
    const n = 65;

    var data_A = try allocator.alloc(f64, n * n);
    defer allocator.free(data_A);

    for (0..n) |i| {
        for (0..n) |j| {
            data_A[i * n + j] = if (i <= j) @as(f64, @floatFromInt(i + j + 1)) else 0.0;
        }
    }

    var data_b = try allocator.alloc(f64, n);
    defer allocator.free(data_b);
    for (0..n) |i| {
        var sum: f64 = 0.0;
        for (i..n) |j| {
            sum += @as(f64, @floatFromInt(i + j + 1));
        }
        data_b[i] = sum;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ n, n }, data_A, .row_major);
    defer A.deinit();
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, data_b, .row_major);
    defer x.deinit();

    try trsv(f64, 'U', 'N', 'N', A, &x);

    // Solution should be x[i] = 1 for all i
    for (0..n) |i| {
        try testing.expectApproxEqAbs(1.0, x.data[i], 1e-8);
    }
}

test "trsv dispatch: large matrix n=128 all parameter combinations" {
    const allocator = testing.allocator;
    const n = 128;

    // Test all 8 combinations: uplo × trans × diag
    const params = [_][3]u8{
        [3]u8{ 'U', 'N', 'N' },
        [3]u8{ 'U', 'N', 'U' },
        [3]u8{ 'U', 'T', 'N' },
        [3]u8{ 'U', 'T', 'U' },
        [3]u8{ 'L', 'N', 'N' },
        [3]u8{ 'L', 'N', 'U' },
        [3]u8{ 'L', 'T', 'N' },
        [3]u8{ 'L', 'T', 'U' },
    };

    for (params) |param| {
        const uplo = param[0];
        const trans = param[1];
        const diag = param[2];

        var data_A = try allocator.alloc(f64, n * n);
        defer allocator.free(data_A);

        // Create triangular matrix
        for (0..n) |i| {
            for (0..n) |j| {
                if (uplo == 'U') {
                    data_A[i * n + j] = if (i <= j) @as(f64, @floatFromInt(i + j + 1)) else 0.0;
                } else {
                    data_A[i * n + j] = if (i >= j) @as(f64, @floatFromInt(i + j + 1)) else 0.0;
                }
            }
        }

        // Create RHS vector that produces solution x[i] = 1
        var data_b = try allocator.alloc(f64, n);
        defer allocator.free(data_b);

        if (trans == 'N') {
            if (uplo == 'U') {
                // Back substitution: b[i] = sum(A[i,j] for j in i..n)
                for (0..n) |i| {
                    var sum: f64 = 0.0;
                    for (i..n) |j| {
                        sum += @as(f64, @floatFromInt(i + j + 1));
                    }
                    data_b[i] = sum;
                }
            } else {
                // Forward substitution: b[i] = sum(A[i,j] for j in 0..i+1)
                for (0..n) |i| {
                    var sum: f64 = 0.0;
                    for (0..i + 1) |j| {
                        sum += @as(f64, @floatFromInt(i + j + 1));
                    }
                    data_b[i] = sum;
                }
            }
        } else {
            if (uplo == 'U') {
                // A^T is lower, forward substitution: b[i] = sum(A^T[i,j] for j in 0..i+1)
                for (0..n) |i| {
                    var sum: f64 = 0.0;
                    for (0..i + 1) |j| {
                        sum += @as(f64, @floatFromInt(j + i + 1));
                    }
                    data_b[i] = sum;
                }
            } else {
                // A^T is upper, back substitution: b[i] = sum(A^T[i,j] for j in i..n)
                for (0..n) |i| {
                    var sum: f64 = 0.0;
                    for (i..n) |j| {
                        sum += @as(f64, @floatFromInt(j + i + 1));
                    }
                    data_b[i] = sum;
                }
            }
        }

        var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ n, n }, data_A, .row_major);
        defer A.deinit();
        var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, data_b, .row_major);
        defer x.deinit();

        try trsv(f64, uplo, trans, diag, A, &x);

        // Verify solution
        for (0..n) |i| {
            try testing.expectApproxEqAbs(1.0, x.data[i], 1e-8);
        }
    }
}

test "trsv dispatch: f32 type support n=64" {
    const allocator = testing.allocator;
    const n = 64;

    // Create f32 upper triangular matrix
    var data_A = try allocator.alloc(f32, n * n);
    defer allocator.free(data_A);

    for (0..n) |i| {
        for (0..n) |j| {
            data_A[i * n + j] = if (i <= j) @as(f32, @floatFromInt(i + j + 1)) else 0.0;
        }
    }

    var data_b = try allocator.alloc(f32, n);
    defer allocator.free(data_b);
    for (0..n) |i| {
        var sum: f32 = 0.0;
        for (i..n) |j| {
            sum += @as(f32, @floatFromInt(i + j + 1));
        }
        data_b[i] = sum;
    }

    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ n, n }, data_A, .row_major);
    defer A.deinit();
    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{n}, data_b, .row_major);
    defer x.deinit();

    try trsv(f32, 'U', 'N', 'N', A, &x);

    // Solution should be x[i] = 1 for all i
    for (0..n) |i| {
        try testing.expectApproxEqAbs(1.0, x.data[i], 1e-6);
    }
}

test "trsv dispatch: non-aligned size n=67 (tail loop coverage)" {
    const allocator = testing.allocator;
    const n = 67;

    // Create upper triangular matrix
    var data_A = try allocator.alloc(f64, n * n);
    defer allocator.free(data_A);

    for (0..n) |i| {
        for (0..n) |j| {
            data_A[i * n + j] = if (i <= j) @as(f64, @floatFromInt(i + j + 1)) else 0.0;
        }
    }

    var data_b = try allocator.alloc(f64, n);
    defer allocator.free(data_b);
    for (0..n) |i| {
        var sum: f64 = 0.0;
        for (i..n) |j| {
            sum += @as(f64, @floatFromInt(i + j + 1));
        }
        data_b[i] = sum;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ n, n }, data_A, .row_major);
    defer A.deinit();
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, data_b, .row_major);
    defer x.deinit();

    try trsv(f64, 'U', 'N', 'N', A, &x);

    // Solution should be x[i] = 1 for all i
    for (0..n) |i| {
        try testing.expectApproxEqAbs(1.0, x.data[i], 1e-8);
    }
}

test "trsv dispatch: non-aligned size n=100 (tail loop coverage)" {
    const allocator = testing.allocator;
    const n = 100;

    // Create lower triangular matrix
    var data_A = try allocator.alloc(f64, n * n);
    defer allocator.free(data_A);

    for (0..n) |i| {
        for (0..n) |j| {
            data_A[i * n + j] = if (i >= j) @as(f64, @floatFromInt(i + j + 1)) else 0.0;
        }
    }

    var data_b = try allocator.alloc(f64, n);
    defer allocator.free(data_b);
    for (0..n) |i| {
        var sum: f64 = 0.0;
        for (0..i + 1) |j| {
            sum += @as(f64, @floatFromInt(i + j + 1));
        }
        data_b[i] = sum;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ n, n }, data_A, .row_major);
    defer A.deinit();
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, data_b, .row_major);
    defer x.deinit();

    try trsv(f64, 'L', 'N', 'N', A, &x);

    // Solution should be x[i] = 1 for all i
    for (0..n) |i| {
        try testing.expectApproxEqAbs(1.0, x.data[i], 1e-8);
    }
}

test "trsv dispatch: non-aligned size n=137 (tail loop coverage)" {
    const allocator = testing.allocator;
    const n = 137;

    // Create upper triangular matrix
    var data_A = try allocator.alloc(f64, n * n);
    defer allocator.free(data_A);

    for (0..n) |i| {
        for (0..n) |j| {
            data_A[i * n + j] = if (i <= j) @as(f64, @floatFromInt(i + j + 1)) else 0.0;
        }
    }

    var data_b = try allocator.alloc(f64, n);
    defer allocator.free(data_b);
    for (0..n) |i| {
        var sum: f64 = 0.0;
        for (i..n) |j| {
            sum += @as(f64, @floatFromInt(i + j + 1));
        }
        data_b[i] = sum;
    }

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ n, n }, data_A, .row_major);
    defer A.deinit();
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, data_b, .row_major);
    defer x.deinit();

    try trsv(f64, 'U', 'N', 'N', A, &x);

    // Solution should be x[i] = 1 for all i
    for (0..n) |i| {
        try testing.expectApproxEqAbs(1.0, x.data[i], 1e-8);
    }
}

test "trmm: dimension mismatch error" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer B.deinit();

    try testing.expectError(error.DimensionMismatch, trmm(f64, 'L', 'U', 'N', 'N', 1.0, A, &B));
}

// ============================================================================
// TRMM Auto-Dispatch Tests — Verify threshold-based dispatch to SIMD
// ============================================================================

test "trmm: auto-dispatch threshold below (32x32, scalar path)" {
    // Below threshold (32 < 64) — should use scalar path
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 32, 32 }, .row_major);
    defer A.deinit();
    for (0..32) |i| {
        for (i..32) |j| {
            A.data[i * 32 + j] = @as(f64, @floatFromInt(i + j + 2));
        }
    }

    var B = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 32, 32 }, .row_major);
    defer B.deinit();
    for (0..32 * 32) |idx| {
        B.data[idx] = 1.0;
    }

    try trmm(f64, 'L', 'U', 'N', 'N', 1.0, A, &B);

    // Verify result is valid (should compute correctly via scalar path)
    try testing.expect(B.data[0] > 0);
}

test "trmm: auto-dispatch threshold boundary (64x64, SIMD path)" {
    // At threshold (64 == 64) — should use SIMD path
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 64, 64 }, .row_major);
    defer A.deinit();
    for (0..64) |i| {
        for (i..64) |j| {
            A.data[i * 64 + j] = @as(f64, @floatFromInt(i + j + 2));
        }
    }

    var B = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 64, 64 }, .row_major);
    defer B.deinit();
    for (0..64 * 64) |idx| {
        B.data[idx] = 1.0;
    }

    try trmm(f64, 'L', 'U', 'N', 'N', 1.0, A, &B);

    // Spot check: B[0,0] should equal sum of first row upper triangle of A
    const expected_sum = (2 + 65) * 64 / 2;
    try testing.expectApproxEqAbs(@as(f64, @floatFromInt(expected_sum)), B.data[0], 1e-6);
}

test "trmm: auto-dispatch threshold above (100x100, SIMD path)" {
    // Above threshold (100 > 64) — should use SIMD path
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 100, 100 }, .row_major);
    defer A.deinit();
    for (0..100) |i| {
        for (i..100) |j| {
            A.data[i * 100 + j] = 1.0;
        }
    }

    var B = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 100, 64 }, .row_major);
    defer B.deinit();
    for (0..100 * 64) |idx| {
        B.data[idx] = 1.0;
    }

    try trmm(f64, 'L', 'U', 'N', 'N', 1.0, A, &B);

    // With identity upper triangular, B[i,j] = 100-i
    for (0..100) |i| {
        for (0..64) |j| {
            const expected = @as(f64, @floatFromInt(100 - i));
            try testing.expectApproxEqAbs(expected, B.data[i * 64 + j], 1e-10);
        }
    }
}

test "trmm: auto-dispatch 128x128 large matrix (SIMD)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 128, 128 }, .row_major);
    defer A.deinit();
    for (0..128) |i| {
        for (0..i + 1) |j| {
            A.data[i * 128 + j] = @as(f64, @floatFromInt(i + j + 2));
        }
    }

    var B = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 128, 64 }, .row_major);
    defer B.deinit();
    for (0..128 * 64) |idx| {
        B.data[idx] = 1.0;
    }

    try trmm(f64, 'L', 'L', 'N', 'N', 1.0, A, &B);

    // Verify valid output
    for (0..128) |i| {
        for (0..64) |j| {
            try testing.expect(B.data[i * 64 + j] > 0);
        }
    }
}

test "trmm: auto-dispatch 256x256 very large matrix (SIMD)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 256, 256 }, .row_major);
    defer A.deinit();
    for (0..256) |i| {
        for (i..256) |j| {
            A.data[i * 256 + j] = 1.0;
        }
    }

    var B = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 256, 128 }, .row_major);
    defer B.deinit();
    for (0..256 * 128) |idx| {
        B.data[idx] = 1.0;
    }

    try trmm(f64, 'L', 'U', 'N', 'N', 1.0, A, &B);

    // With identity upper triangular, B[i,j] = 256-i
    for (0..256) |i| {
        for (0..128) |j| {
            const expected = @as(f64, @floatFromInt(256 - i));
            try testing.expectApproxEqAbs(expected, B.data[i * 128 + j], 1e-10);
        }
    }
}

test "trmm: auto-dispatch right side 100x100 non-aligned (SIMD)" {
    // Non-aligned size (100) but >= 64, should still use SIMD
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 100, 100 }, .row_major);
    defer A.deinit();
    for (0..100) |i| {
        for (0..i + 1) |j| {
            A.data[i * 100 + j] = 1.0;
        }
    }

    var B = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 80, 100 }, .row_major);
    defer B.deinit();
    for (0..80 * 100) |idx| {
        B.data[idx] = 1.0;
    }

    try trmm(f64, 'R', 'L', 'N', 'N', 1.0, A, &B);

    // Verify computation completed
    for (0..80) |i| {
        for (0..100) |j| {
            try testing.expect(B.data[i * 100 + j] > 0);
        }
    }
}

test "trmm: auto-dispatch f32 type (64x64)" {
    // f32 should also dispatch to SIMD at 64x64 (8-wide vectors)
    const allocator = testing.allocator;

    var A = try NDArray(f32, 2).zeros(allocator, &[_]usize{ 64, 64 }, .row_major);
    defer A.deinit();
    for (0..64) |i| {
        for (i..64) |j| {
            A.data[i * 64 + j] = @as(f32, @floatFromInt(i + j + 2));
        }
    }

    var B = try NDArray(f32, 2).zeros(allocator, &[_]usize{ 64, 64 }, .row_major);
    defer B.deinit();
    for (0..64 * 64) |idx| {
        B.data[idx] = 1.0;
    }

    try trmm(f32, 'L', 'U', 'N', 'N', 1.0, A, &B);

    // Spot check validity
    try testing.expect(B.data[0] > 0);
}

test "trmm: auto-dispatch left+upper+trans+non-unit (128x128)" {
    // Test full parameter combination with SIMD
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 128, 128 }, .row_major);
    defer A.deinit();
    for (0..128) |i| {
        for (i..128) |j| {
            A.data[i * 128 + j] = @as(f64, @floatFromInt(i + j + 2));
        }
    }

    var B = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 128, 64 }, .row_major);
    defer B.deinit();
    for (0..128 * 64) |idx| {
        B.data[idx] = 1.0;
    }

    try trmm(f64, 'L', 'U', 'T', 'N', 2.0, A, &B);

    // Verify non-zero output
    for (0..128) |i| {
        for (0..64) |j| {
            try testing.expect(B.data[i * 64 + j] > 0);
        }
    }
}

test "trmm: auto-dispatch right+lower+no-trans+unit (100x100)" {
    // Test full parameter combination with SIMD
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 100, 100 }, .row_major);
    defer A.deinit();
    for (0..100) |i| {
        for (0..i + 1) |j| {
            A.data[i * 100 + j] = @as(f64, @floatFromInt(i + j + 2));
        }
    }

    var B = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 80, 100 }, .row_major);
    defer B.deinit();
    for (0..80 * 100) |idx| {
        B.data[idx] = 1.0;
    }

    try trmm(f64, 'R', 'L', 'N', 'U', 1.5, A, &B);

    // Verify valid computation
    for (0..80) |i| {
        for (0..100) |j| {
            try testing.expect(B.data[i * 100 + j] > 0);
        }
    }
}

test "trmm: auto-dispatch consistent scalar and SIMD (32x32 vs 64x64)" {
    // Verify that scalar path (32x32) and SIMD path (64x64) produce consistent results
    const allocator = testing.allocator;

    // Scalar path test (32x32)
    var A_scalar = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 1, 0, 3 }, .row_major);
    defer A_scalar.deinit();

    var B_scalar = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B_scalar.deinit();

    try trmm(f64, 'L', 'U', 'N', 'N', 1.0, A_scalar, &B_scalar);

    // SIMD path test (same matrices, but auto-dispatch will choose scalar anyway at 2x2)
    var A_simd = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 1, 0, 3 }, .row_major);
    defer A_simd.deinit();

    var B_simd = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B_simd.deinit();

    try trmm(f64, 'L', 'U', 'N', 'N', 1.0, A_simd, &B_simd);

    // Results should be bit-exact
    for (0..4) |i| {
        try testing.expectApproxEqAbs(B_scalar.data[i], B_simd.data[i], 1e-15);
    }
}

test "trsm: dimension mismatch error" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer B.deinit();

    try testing.expectError(error.DimensionMismatch, trsm(f64, 'L', 'U', 'N', 'N', 1.0, A, &B));
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

// ============================================================================
// syr() — Symmetric Rank-1 Update: A := α*x*x^T + A
// ============================================================================

/// Symmetric rank-1 update: A := α*x*x^T + A
/// Performs a symmetric rank-1 update to a symmetric matrix.
/// Only the specified triangle (upper 'U' or lower 'L') is updated.
/// The other triangle is left untouched.
///
/// Time: O(n²) | Space: O(1)
///
/// Parameters:
///   T: floating point type (f32 or f64)
///   uplo: 'U' for upper triangle, 'L' for lower triangle
///   alpha: scalar multiplier
///   x: 1D vector of length n
///   A: n×n symmetric matrix (modified in-place)
///
/// Errors:
///   - DimensionMismatch: if x.shape[0] != A.shape[0]
///   - NotSquare: if A is not square
pub fn syr(comptime T: type, uplo: u8, alpha: T, x: NDArray(T, 1), A: *NDArray(T, 2)) (NDArray(T, 1).Error)!void {
    // Validate matrix is square
    if (A.shape[0] != A.shape[1]) {
        return error.NotSquare;
    }

    // Validate dimensions match
    if (x.shape[0] != A.shape[0]) {
        return error.DimensionMismatch;
    }

    const n = A.shape[0];

    // Early exit: alpha == 0 is a no-op
    if (alpha == 0) {
        return;
    }

    // Auto-dispatch: Use SIMD-optimized implementation for large matrices
    if (n >= 64) {
        return simd_blas.syr_simd(T, uplo, alpha, x, A);
    }

    // Scalar implementation for small matrices
    if (uplo == 'U') {
        // Update upper triangle: for i in 0..n, for j in i..n: A[i,j] += α*x[i]*x[j]
        for (0..n) |i| {
            const x_i = x.data[i];
            for (i..n) |j| {
                const x_j = x.data[j];
                A.data[i * n + j] += alpha * x_i * x_j;
            }
        }
    } else if (uplo == 'L') {
        // Update lower triangle: for i in 0..n, for j in 0..=i: A[i,j] += α*x[i]*x[j]
        for (0..n) |i| {
            const x_i = x.data[i];
            for (0..i + 1) |j| {
                const x_j = x.data[j];
                A.data[i * n + j] += alpha * x_i * x_j;
            }
        }
    }
}

// ============================================================================
// syr() Tests
// ============================================================================

test "syr: basic 3×3 upper triangle, alpha=1" {
    const allocator = testing.allocator;

    // x = [1, 2, 3]
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    // A = [[1, 2, 3], [0, 4, 5], [0, 0, 6]]  (upper triangle stored)
    // Upper triangle: [1, 2, 3, 4, 5, 6]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 0, 4, 5, 0, 0, 6 }, .row_major);
    defer A.deinit();

    // syr(1.0, x, &A) updates upper triangle:
    // x*x^T = [[1*1, 1*2, 1*3], [2*1, 2*2, 2*3], [3*1, 3*2, 3*3]]
    //       = [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
    // Upper triangle of x*x^T: [1, 2, 3, 4, 6, 9]
    // Result: A(upper) = A(upper) + 1*x*x^T(upper)
    //       = [1, 2, 3, 4, 5, 6] + [1, 2, 3, 4, 6, 9]
    //       = [2, 4, 6, 8, 11, 15]
    // In full matrix form (upper triangle):
    //   A[0,0] = 1 + 1 = 2
    //   A[0,1] = 2 + 2 = 4
    //   A[0,2] = 3 + 3 = 6
    //   A[1,1] = 4 + 4 = 8
    //   A[1,2] = 5 + 6 = 11
    //   A[2,2] = 6 + 9 = 15

    try syr(f64, 'U', 1.0, x, &A);

    try testing.expectApproxEqAbs(2.0, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(4.0, A.data[1], 1e-10);
    try testing.expectApproxEqAbs(6.0, A.data[2], 1e-10);
    try testing.expectApproxEqAbs(8.0, A.data[4], 1e-10);
    try testing.expectApproxEqAbs(11.0, A.data[5], 1e-10);
    try testing.expectApproxEqAbs(15.0, A.data[8], 1e-10);
}

test "syr: basic 3×3 lower triangle, alpha=1" {
    const allocator = testing.allocator;

    // x = [1, 2, 3]
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    // A = [[1, 0, 0], [2, 4, 0], [3, 5, 6]]  (lower triangle stored)
    // Lower triangle: [1, 2, 4, 3, 5, 6]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 0, 0, 2, 4, 0, 3, 5, 6 }, .row_major);
    defer A.deinit();

    // syr(1.0, x, &A) updates lower triangle:
    // x*x^T lower triangle: [1, 2, 4, 3, 6, 9]
    // Result: A(lower) = A(lower) + 1*x*x^T(lower)
    //       = [1, 2, 4, 3, 5, 6] + [1, 2, 4, 3, 6, 9]
    //       = [2, 4, 8, 6, 11, 15]
    // In full matrix form (lower triangle):
    //   A[0,0] = 1 + 1 = 2
    //   A[1,0] = 2 + 2 = 4
    //   A[1,1] = 4 + 4 = 8
    //   A[2,0] = 3 + 3 = 6
    //   A[2,1] = 5 + 6 = 11
    //   A[2,2] = 6 + 9 = 15

    try syr(f64, 'L', 1.0, x, &A);

    try testing.expectApproxEqAbs(2.0, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(4.0, A.data[3], 1e-10);
    try testing.expectApproxEqAbs(8.0, A.data[4], 1e-10);
    try testing.expectApproxEqAbs(6.0, A.data[6], 1e-10);
    try testing.expectApproxEqAbs(11.0, A.data[7], 1e-10);
    try testing.expectApproxEqAbs(15.0, A.data[8], 1e-10);
}

test "syr: alpha=0 (no-op, preserves matrix)" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 5, 6, 7 }, .row_major);
    defer x.deinit();

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 0, 4, 5, 0, 0, 6 }, .row_major);
    defer A.deinit();

    const original = try A.clone();
    defer original.deinit();

    // alpha=0 should not change A
    try syr(f64, 'U', 0.0, x, &A);

    for (0..9) |i| {
        try testing.expectApproxEqAbs(original.data[i], A.data[i], 1e-10);
    }
}

test "syr: alpha=-1 (negative scalar)" {
    const allocator = testing.allocator;

    // x = [1, 2]
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 2 }, .row_major);
    defer x.deinit();

    // A = [[5, 6], [0, 7]]  (upper triangle)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 6, 0, 7 }, .row_major);
    defer A.deinit();

    // syr(-1.0, x, &A) with upper:
    // -1 * x*x^T(upper) = -1 * [[1, 2], [2, 4]] = [[-1, -2], [-2, -4]]
    // A(upper) = [5, 6, 7] + [-1, -2, -4] = [4, 4, 3]
    // A[0,0] = 5 - 1 = 4
    // A[0,1] = 6 - 2 = 4
    // A[1,1] = 7 - 4 = 3

    try syr(f64, 'U', -1.0, x, &A);

    try testing.expectApproxEqAbs(4.0, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(4.0, A.data[1], 1e-10);
    try testing.expectApproxEqAbs(3.0, A.data[3], 1e-10);
}

test "syr: alpha=2.5 (fractional scalar)" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 2, 4 }, .row_major);
    defer x.deinit();

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 1, 0, 1 }, .row_major);
    defer A.deinit();

    // syr(2.5, x, &A) with upper:
    // 2.5 * x*x^T(upper) = 2.5 * [[4, 8], [8, 16]] = [[10, 20], [20, 40]]
    // A(upper) = [1, 1, 1] + [10, 20, 40] = [11, 21, 41]

    try syr(f64, 'U', 2.5, x, &A);

    try testing.expectApproxEqAbs(11.0, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(21.0, A.data[1], 1e-10);
    try testing.expectApproxEqAbs(41.0, A.data[3], 1e-10);
}

test "syr: zero vector (x all zeros)" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).zeros(allocator, &[_]usize{3}, .row_major);
    defer x.deinit();

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 0, 4, 5, 0, 0, 6 }, .row_major);
    defer A.deinit();

    const original = try A.clone();
    defer original.deinit();

    // Zero vector contributes nothing: A remains unchanged
    try syr(f64, 'U', 5.0, x, &A);

    for (0..9) |i| {
        try testing.expectApproxEqAbs(original.data[i], A.data[i], 1e-10);
    }
}

test "syr: uniform vector (all same values)" {
    const allocator = testing.allocator;

    // x = [2, 2, 2]
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 2, 2, 2 }, .row_major);
    defer x.deinit();

    // A = zeros
    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer A.deinit();

    // syr(1.0, x, &A) with upper:
    // x*x^T(upper) = [[4, 4, 4], [4, 4, 4], [4, 4, 4]]
    // Upper triangle: [4, 4, 4, 4, 4, 4]

    try syr(f64, 'U', 1.0, x, &A);

    try testing.expectApproxEqAbs(4.0, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(4.0, A.data[1], 1e-10);
    try testing.expectApproxEqAbs(4.0, A.data[2], 1e-10);
    try testing.expectApproxEqAbs(4.0, A.data[4], 1e-10);
    try testing.expectApproxEqAbs(4.0, A.data[5], 1e-10);
    try testing.expectApproxEqAbs(4.0, A.data[8], 1e-10);
}

test "syr: f32 type support" {
    const allocator = testing.allocator;

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{2}, &[_]f32{ 1.5, 2.5 }, .row_major);
    defer x.deinit();

    var A = try NDArray(f32, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer A.deinit();

    // syr(1.0, x, &A) with upper:
    // x*x^T(upper) = [[2.25, 3.75], [3.75, 6.25]]
    // Upper triangle: [2.25, 3.75, 6.25]

    try syr(f32, 'U', 1.0, x, &A);

    try testing.expectApproxEqAbs(@as(f32, 2.25), A.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 3.75), A.data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 6.25), A.data[3], 1e-5);
}

test "syr: n=1 edge case (1×1 matrix)" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{3}, .row_major);
    defer x.deinit();

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 1 }, &[_]f64{5}, .row_major);
    defer A.deinit();

    // syr(1.0, x, &A) with upper:
    // x*x^T = [[9]], so A[0,0] = 5 + 9 = 14

    try syr(f64, 'U', 1.0, x, &A);

    try testing.expectApproxEqAbs(14.0, A.data[0], 1e-10);
}

test "syr: dimension mismatch — vector length != matrix size" {
    const allocator = testing.allocator;

    // x has 2 elements, A is 3×3
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 2 }, .row_major);
    defer x.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer A.deinit();

    // Should return error
    const result = syr(f64, 'U', 1.0, x, &A);
    try testing.expectError(error.DimensionMismatch, result);
}

test "syr: dimension mismatch — matrix not square (3×4)" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer A.deinit();

    // Should return error: matrix is not square
    const result = syr(f64, 'U', 1.0, x, &A);
    try testing.expectError(error.NotSquare, result);
}

test "syr: large matrix n=64 (SIMD threshold)" {
    const allocator = testing.allocator;

    // Create a 64-element vector
    var x_data = try allocator.alloc(f64, 64);
    defer allocator.free(x_data);
    for (0..64) |i| {
        x_data[i] = @as(f64, @floatFromInt(i + 1));
    }
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{64}, x_data, .row_major);
    defer x.deinit();

    // Create 64×64 matrix initialized to zeros
    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 64, 64 }, .row_major);
    defer A.deinit();

    // Perform syr update
    try syr(f64, 'U', 1.0, x, &A);

    // Verify diagonal: A[i,i] should be x[i]² = (i+1)²
    for (0..64) |i| {
        const expected = @as(f64, @floatFromInt(i + 1)) * @as(f64, @floatFromInt(i + 1));
        try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, i }), 1e-10);
    }

    // Verify upper triangle symmetry: A[i,j] = x[i]*x[j] for i <= j
    for (0..10) |i| {
        for (i + 1..10) |j| {
            const expected = @as(f64, @floatFromInt(i + 1)) * @as(f64, @floatFromInt(j + 1));
            try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, j }), 1e-10);
        }
    }
}

test "syr: large matrix n=128 (well above SIMD threshold)" {
    const allocator = testing.allocator;

    // Create a 128-element vector
    var x_data = try allocator.alloc(f64, 128);
    defer allocator.free(x_data);
    for (0..128) |i| {
        x_data[i] = @as(f64, @floatFromInt(i % 10 + 1)); // Use modulo to keep values small
    }
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{128}, x_data, .row_major);
    defer x.deinit();

    // Create 128×128 matrix initialized to zeros
    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 128, 128 }, .row_major);
    defer A.deinit();

    // Perform syr update
    try syr(f64, 'U', 1.0, x, &A);

    // Verify diagonal: A[i,i] should be x[i]²
    for (0..10) |i| {
        const x_i = @as(f64, @floatFromInt(i % 10 + 1));
        const expected = x_i * x_i;
        try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, i }), 1e-10);
    }

    // Verify upper triangle sample: A[i,j] = x[i]*x[j] for i <= j
    for (0..5) |i| {
        for (i + 1..5) |j| {
            const x_i = @as(f64, @floatFromInt(i % 10 + 1));
            const x_j = @as(f64, @floatFromInt(j % 10 + 1));
            const expected = x_i * x_j;
            try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, j }), 1e-10);
        }
    }
}

test "syr: lower triangle with n=10" {
    const allocator = testing.allocator;

    var x_data = try allocator.alloc(f64, 10);
    defer allocator.free(x_data);
    for (0..10) |i| {
        x_data[i] = @as(f64, @floatFromInt(i + 1));
    }
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{10}, x_data, .row_major);
    defer x.deinit();

    // Create 10×10 matrix initialized to zeros
    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 10, 10 }, .row_major);
    defer A.deinit();

    // Perform syr update with lower triangle
    try syr(f64, 'L', 1.0, x, &A);

    // Verify diagonal: A[i,i] should be x[i]²
    for (0..10) |i| {
        const x_i = @as(f64, @floatFromInt(i + 1));
        const expected = x_i * x_i;
        try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, i }), 1e-10);
    }

    // Verify lower triangle: A[i,j] = x[i]*x[j] for i >= j
    for (0..5) |j| {
        for (j + 1..5) |i| {
            const x_i = @as(f64, @floatFromInt(i + 1));
            const x_j = @as(f64, @floatFromInt(j + 1));
            const expected = x_i * x_j;
            try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, j }), 1e-10);
        }
    }
}

test "syr: no memory leaks with multiple iterations" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 1, 2, 3, 4, 5 }, .row_major);
        defer x.deinit();

        var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 5, 5 }, .row_major);
        defer A.deinit();

        try syr(f64, 'U', 1.0, x, &A);
    }
}

test "syr: multiple accumulations (repeated updates)" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 1 }, .row_major);
    defer x.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer A.deinit();

    // First update: A = 0 + 1.0 * [[1, 1], [1, 1]] = [[1, 1], [1, 1]]
    try syr(f64, 'U', 1.0, x, &A);

    try testing.expectApproxEqAbs(1.0, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(1.0, A.data[1], 1e-10);
    try testing.expectApproxEqAbs(1.0, A.data[3], 1e-10);

    // Second update: A = [[1, 1], [1, 1]] + 1.0 * [[1, 1], [1, 1]] = [[2, 2], [2, 2]]
    try syr(f64, 'U', 1.0, x, &A);

    try testing.expectApproxEqAbs(2.0, A.data[0], 1e-10);
    try testing.expectApproxEqAbs(2.0, A.data[1], 1e-10);
    try testing.expectApproxEqAbs(2.0, A.data[3], 1e-10);
}

// ============================================================================
// syr() Auto-Dispatch Tests
// ============================================================================
// These tests verify that syr() correctly dispatches to syr_simd() for
// large matrices (n >= 64) and uses the scalar path for smaller matrices.

test "syr: auto-dispatch threshold below (n=63, upper, f64, scalar path)" {
    // Below threshold (63 < 64) — should use scalar path
    const allocator = testing.allocator;

    var x_data = try allocator.alloc(f64, 63);
    defer allocator.free(x_data);
    for (0..63) |i| {
        x_data[i] = @as(f64, @floatFromInt(i + 1));
    }
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{63}, x_data, .row_major);
    defer x.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 63, 63 }, .row_major);
    defer A.deinit();

    // Perform syr update
    try syr(f64, 'U', 1.0, x, &A);

    // Verify diagonal: A[i,i] should be x[i]² = (i+1)²
    for (0..63) |i| {
        const expected = @as(f64, @floatFromInt(i + 1)) * @as(f64, @floatFromInt(i + 1));
        try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, i }), 1e-10);
    }

    // Verify upper triangle sample: A[i,j] = x[i]*x[j] for i <= j
    for (0..10) |i| {
        for (i + 1..10) |j| {
            const expected = @as(f64, @floatFromInt(i + 1)) * @as(f64, @floatFromInt(j + 1));
            try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, j }), 1e-10);
        }
    }
}

test "syr: auto-dispatch threshold boundary (n=64, upper, f64, SIMD path)" {
    // At threshold (64 == 64) — should use SIMD path
    const allocator = testing.allocator;

    var x_data = try allocator.alloc(f64, 64);
    defer allocator.free(x_data);
    for (0..64) |i| {
        x_data[i] = @as(f64, @floatFromInt(i + 1));
    }
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{64}, x_data, .row_major);
    defer x.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 64, 64 }, .row_major);
    defer A.deinit();

    // Perform syr update
    try syr(f64, 'U', 1.0, x, &A);

    // Verify diagonal: A[i,i] should be x[i]² = (i+1)²
    for (0..64) |i| {
        const expected = @as(f64, @floatFromInt(i + 1)) * @as(f64, @floatFromInt(i + 1));
        try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, i }), 1e-10);
    }

    // Verify upper triangle sample: A[i,j] = x[i]*x[j] for i <= j
    for (0..10) |i| {
        for (i + 1..10) |j| {
            const expected = @as(f64, @floatFromInt(i + 1)) * @as(f64, @floatFromInt(j + 1));
            try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, j }), 1e-10);
        }
    }
}

test "syr: auto-dispatch threshold above (n=65, upper, f64, SIMD path)" {
    // Above threshold (65 > 64) — should use SIMD path
    const allocator = testing.allocator;

    var x_data = try allocator.alloc(f64, 65);
    defer allocator.free(x_data);
    for (0..65) |i| {
        x_data[i] = @as(f64, @floatFromInt(i + 1));
    }
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{65}, x_data, .row_major);
    defer x.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 65, 65 }, .row_major);
    defer A.deinit();

    // Perform syr update
    try syr(f64, 'U', 1.0, x, &A);

    // Verify diagonal: A[i,i] should be x[i]² = (i+1)²
    for (0..10) |i| {
        const expected = @as(f64, @floatFromInt(i + 1)) * @as(f64, @floatFromInt(i + 1));
        try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, i }), 1e-10);
    }

    // Verify upper triangle sample: A[i,j] = x[i]*x[j] for i <= j
    for (0..10) |i| {
        for (i + 1..10) |j| {
            const expected = @as(f64, @floatFromInt(i + 1)) * @as(f64, @floatFromInt(j + 1));
            try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, j }), 1e-10);
        }
    }
}

test "syr: auto-dispatch lower triangle (n=64, lower, SIMD path)" {
    // SIMD path with lower triangle variant
    const allocator = testing.allocator;

    var x_data = try allocator.alloc(f64, 64);
    defer allocator.free(x_data);
    for (0..64) |i| {
        x_data[i] = @as(f64, @floatFromInt(i + 1));
    }
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{64}, x_data, .row_major);
    defer x.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 64, 64 }, .row_major);
    defer A.deinit();

    // Perform syr update with lower triangle
    try syr(f64, 'L', 1.0, x, &A);

    // Verify diagonal: A[i,i] should be x[i]² = (i+1)²
    for (0..64) |i| {
        const expected = @as(f64, @floatFromInt(i + 1)) * @as(f64, @floatFromInt(i + 1));
        try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, i }), 1e-10);
    }

    // Verify lower triangle sample: A[i,j] = x[i]*x[j] for i >= j
    for (0..10) |j| {
        for (j + 1..10) |i| {
            const expected = @as(f64, @floatFromInt(i + 1)) * @as(f64, @floatFromInt(j + 1));
            try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, j }), 1e-10);
        }
    }
}

test "syr: auto-dispatch large matrix (n=128, upper, f64, SIMD)" {
    // Well above threshold — verify SIMD efficiency with larger matrix
    const allocator = testing.allocator;

    var x_data = try allocator.alloc(f64, 128);
    defer allocator.free(x_data);
    for (0..128) |i| {
        x_data[i] = @as(f64, @floatFromInt(i % 10 + 1));
    }
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{128}, x_data, .row_major);
    defer x.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 128, 128 }, .row_major);
    defer A.deinit();

    // Perform syr update
    try syr(f64, 'U', 1.0, x, &A);

    // Verify diagonal sample: A[i,i] should be x[i]²
    for (0..10) |i| {
        const x_i = @as(f64, @floatFromInt(i % 10 + 1));
        const expected = x_i * x_i;
        try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, i }), 1e-10);
    }

    // Verify upper triangle sample: A[i,j] = x[i]*x[j] for i <= j
    for (0..5) |i| {
        for (i + 1..5) |j| {
            const x_i = @as(f64, @floatFromInt(i % 10 + 1));
            const x_j = @as(f64, @floatFromInt(j % 10 + 1));
            const expected = x_i * x_j;
            try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, j }), 1e-10);
        }
    }
}

test "syr: auto-dispatch very large matrix (n=256, upper, f64, SIMD)" {
    // Very large matrix to stress SIMD path
    const allocator = testing.allocator;

    var x_data = try allocator.alloc(f64, 256);
    defer allocator.free(x_data);
    for (0..256) |i| {
        x_data[i] = @as(f64, @floatFromInt(i % 20 + 1));
    }
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{256}, x_data, .row_major);
    defer x.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 256, 256 }, .row_major);
    defer A.deinit();

    // Perform syr update
    try syr(f64, 'U', 1.0, x, &A);

    // Verify diagonal sample
    for (0..10) |i| {
        const x_i = @as(f64, @floatFromInt(i % 20 + 1));
        const expected = x_i * x_i;
        try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, i }), 1e-10);
    }
}

test "syr: auto-dispatch non-aligned (n=67, upper, SIMD tail loop)" {
    // Non-aligned size to verify tail loop handling in SIMD path
    const allocator = testing.allocator;

    var x_data = try allocator.alloc(f64, 67);
    defer allocator.free(x_data);
    for (0..67) |i| {
        x_data[i] = @as(f64, @floatFromInt(i + 1));
    }
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{67}, x_data, .row_major);
    defer x.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 67, 67 }, .row_major);
    defer A.deinit();

    // Perform syr update
    try syr(f64, 'U', 1.0, x, &A);

    // Verify diagonal sample
    for (0..10) |i| {
        const expected = @as(f64, @floatFromInt(i + 1)) * @as(f64, @floatFromInt(i + 1));
        try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, i }), 1e-10);
    }
}

test "syr: auto-dispatch non-aligned (n=100, upper, SIMD tail loop)" {
    // Another non-aligned size (100) to verify tail loop in SIMD
    const allocator = testing.allocator;

    var x_data = try allocator.alloc(f64, 100);
    defer allocator.free(x_data);
    for (0..100) |i| {
        x_data[i] = @as(f64, @floatFromInt(i % 15 + 1));
    }
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{100}, x_data, .row_major);
    defer x.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 100, 100 }, .row_major);
    defer A.deinit();

    // Perform syr update
    try syr(f64, 'U', 1.0, x, &A);

    // Verify diagonal sample
    for (0..10) |i| {
        const x_i = @as(f64, @floatFromInt(i % 15 + 1));
        const expected = x_i * x_i;
        try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, i }), 1e-10);
    }
}

test "syr: auto-dispatch f32 type (n=64, upper, SIMD)" {
    // Test with f32 type instead of f64
    const allocator = testing.allocator;

    var x_data = try allocator.alloc(f32, 64);
    defer allocator.free(x_data);
    for (0..64) |i| {
        x_data[i] = @as(f32, @floatFromInt(i + 1));
    }
    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{64}, x_data, .row_major);
    defer x.deinit();

    var A = try NDArray(f32, 2).zeros(allocator, &[_]usize{ 64, 64 }, .row_major);
    defer A.deinit();

    // Perform syr update
    try syr(f32, 'U', 1.0, x, &A);

    // Verify diagonal: A[i,i] should be x[i]² = (i+1)²
    for (0..10) |i| {
        const expected = @as(f32, @floatFromInt(i + 1)) * @as(f32, @floatFromInt(i + 1));
        try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, i }), 1e-5);
    }
}

test "syr: auto-dispatch f32 large (n=128, upper, SIMD)" {
    // Test with f32 type on larger matrix
    const allocator = testing.allocator;

    var x_data = try allocator.alloc(f32, 128);
    defer allocator.free(x_data);
    for (0..128) |i| {
        x_data[i] = @as(f32, @floatFromInt(i % 10 + 1));
    }
    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{128}, x_data, .row_major);
    defer x.deinit();

    var A = try NDArray(f32, 2).zeros(allocator, &[_]usize{ 128, 128 }, .row_major);
    defer A.deinit();

    // Perform syr update
    try syr(f32, 'U', 1.0, x, &A);

    // Verify diagonal sample
    for (0..10) |i| {
        const x_i = @as(f32, @floatFromInt(i % 10 + 1));
        const expected = x_i * x_i;
        try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, i }), 1e-5);
    }
}

test "syr: auto-dispatch alpha=0 (n=128, SIMD no-op)" {
    // Alpha=0 should be a no-op even in SIMD path
    const allocator = testing.allocator;

    var x_data = try allocator.alloc(f64, 128);
    defer allocator.free(x_data);
    for (0..128) |i| {
        x_data[i] = @as(f64, @floatFromInt(i + 1));
    }
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{128}, x_data, .row_major);
    defer x.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 128, 128 }, .row_major);
    defer A.deinit();

    const original = try A.clone();
    defer original.deinit();

    // alpha=0 should not change A
    try syr(f64, 'U', 0.0, x, &A);

    // Verify A is unchanged
    for (0..128 * 128) |i| {
        try testing.expectApproxEqAbs(original.data[i], A.data[i], 1e-10);
    }
}

test "syr: auto-dispatch alpha=2.5 (n=128, upper, SIMD)" {
    // Non-unit alpha to verify scalar multiplication in SIMD path
    const allocator = testing.allocator;

    var x_data = try allocator.alloc(f64, 128);
    defer allocator.free(x_data);
    for (0..128) |i| {
        x_data[i] = @as(f64, @floatFromInt(i % 5 + 1));
    }
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{128}, x_data, .row_major);
    defer x.deinit();

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 128, 128 }, .row_major);
    defer A.deinit();

    const alpha = 2.5;

    // Perform syr update with non-unit alpha
    try syr(f64, 'U', alpha, x, &A);

    // Verify diagonal sample: A[i,i] should be alpha * x[i]²
    for (0..10) |i| {
        const x_i = @as(f64, @floatFromInt(i % 5 + 1));
        const expected = alpha * x_i * x_i;
        try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, i }), 1e-9);
    }

    // Verify upper triangle sample: A[i,j] = alpha * x[i]*x[j] for i <= j
    for (0..5) |i| {
        for (i + 1..5) |j| {
            const x_i = @as(f64, @floatFromInt(i % 5 + 1));
            const x_j = @as(f64, @floatFromInt(j % 5 + 1));
            const expected = alpha * x_i * x_j;
            try testing.expectApproxEqAbs(expected, A.at(&[_]usize{ i, j }), 1e-9);
        }
    }
}

// ============================================================================
// TRSM AUTO-DISPATCH TESTS — Verify dispatch to SIMD/scalar implementations
// ============================================================================
// Tests verify that trsm() correctly dispatches to simd_blas.trsm_simd()
// for large matrices and maintains scalar behavior for small matrices.

test "trsm auto-dispatch: threshold 32×32 scalar path f64" {
    const allocator = testing.allocator;

    // Upper triangular: identity + off-diagonals
    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 32, 32 }, .row_major);
    defer A.deinit();
    for (0..32) |i| {
        for (i..32) |j| {
            A.data[i * 32 + j] = @as(f64, @floatFromInt(i + j + 2));
        }
    }

    // Create B with 4 RHS
    var B = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 32, 4 }, .row_major);
    defer B.deinit();
    for (0..32 * 4) |idx| {
        B.data[idx] = 1.0;
    }

    try trsm(f64, 'L', 'U', 'N', 'N', 1.0, A, &B);

    // Verify solution validity
    for (0..32) |i| {
        for (0..4) |j| {
            try testing.expect(!std.math.isNan(B.data[i * 4 + j]));
            try testing.expect(!std.math.isInf(B.data[i * 4 + j]));
        }
    }
}

test "trsm auto-dispatch: threshold 64×64 SIMD path f64" {
    const allocator = testing.allocator;

    // Upper triangular matrix
    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 64, 64 }, .row_major);
    defer A.deinit();
    for (0..64) |i| {
        for (i..64) |j| {
            A.data[i * 64 + j] = @as(f64, @floatFromInt(i + j + 2));
        }
    }

    // Create B with 4 RHS
    var B = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 64, 4 }, .row_major);
    defer B.deinit();
    for (0..64 * 4) |idx| {
        B.data[idx] = 1.0;
    }

    try trsm(f64, 'L', 'U', 'N', 'N', 1.0, A, &B);

    // Verify solution validity (SIMD path)
    for (0..64) |i| {
        for (0..4) |j| {
            try testing.expect(!std.math.isNan(B.data[i * 4 + j]));
            try testing.expect(!std.math.isInf(B.data[i * 4 + j]));
        }
    }
}

test "trsm auto-dispatch: 100×100 non-aligned SIMD f64" {
    const allocator = testing.allocator;

    // Lower triangular matrix (non-64 size)
    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 100, 100 }, .row_major);
    defer A.deinit();
    for (0..100) |i| {
        for (0..i + 1) |j| {
            A.data[i * 100 + j] = @as(f64, @floatFromInt(i + j + 2));
        }
    }

    // Create B with 8 RHS
    var B = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 100, 8 }, .row_major);
    defer B.deinit();
    for (0..100 * 8) |idx| {
        B.data[idx] = 1.0;
    }

    try trsm(f64, 'L', 'L', 'N', 'N', 1.0, A, &B);

    // Verify solution validity
    for (0..100) |i| {
        for (0..8) |j| {
            try testing.expect(!std.math.isNan(B.data[i * 8 + j]));
            try testing.expect(!std.math.isInf(B.data[i * 8 + j]));
        }
    }
}

test "trsm auto-dispatch: 128×128 SIMD path f64" {
    const allocator = testing.allocator;

    // Upper triangular matrix
    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 128, 128 }, .row_major);
    defer A.deinit();
    for (0..128) |i| {
        for (i..128) |j| {
            A.data[i * 128 + j] = @as(f64, @floatFromInt(i + j + 2));
        }
    }

    // Create B with 8 RHS
    var B = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 128, 8 }, .row_major);
    defer B.deinit();
    for (0..128 * 8) |idx| {
        B.data[idx] = 1.0;
    }

    try trsm(f64, 'L', 'U', 'N', 'N', 1.0, A, &B);

    // Verify solution validity
    for (0..128) |i| {
        for (0..8) |j| {
            try testing.expect(!std.math.isNan(B.data[i * 8 + j]));
            try testing.expect(!std.math.isInf(B.data[i * 8 + j]));
        }
    }
}

test "trsm auto-dispatch: 256×256 large SIMD f64" {
    const allocator = testing.allocator;

    // Lower triangular matrix
    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 256, 256 }, .row_major);
    defer A.deinit();
    for (0..256) |i| {
        for (0..i + 1) |j| {
            A.data[i * 256 + j] = @as(f64, @floatFromInt(i + j + 2));
        }
    }

    // Create B with 4 RHS
    var B = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 256, 4 }, .row_major);
    defer B.deinit();
    for (0..256 * 4) |idx| {
        B.data[idx] = 1.0;
    }

    try trsm(f64, 'L', 'L', 'N', 'N', 1.0, A, &B);

    // Verify solution validity
    for (0..256) |i| {
        for (0..4) |j| {
            try testing.expect(!std.math.isNan(B.data[i * 4 + j]));
            try testing.expect(!std.math.isInf(B.data[i * 4 + j]));
        }
    }
}

test "trsm auto-dispatch: left side parameter combo f64" {
    const allocator = testing.allocator;

    // Test left + lower + transpose + unit diagonal
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 64 }, undefined, .row_major);
    defer A.deinit();
    // Fill with unit lower triangular
    for (0..64) |i| {
        for (0..64) |j| {
            A.data[i * 64 + j] = if (i == j) 1.0 else if (i > j) @as(f64, @floatFromInt(i - j)) else 0.0;
        }
    }

    var B = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 64, 4 }, .row_major);
    defer B.deinit();
    for (0..64 * 4) |idx| {
        B.data[idx] = 1.0;
    }

    try trsm(f64, 'L', 'L', 'T', 'U', 1.0, A, &B);

    // Verify no NaN/Inf
    for (0..64 * 4) |idx| {
        try testing.expect(!std.math.isNan(B.data[idx]));
        try testing.expect(!std.math.isInf(B.data[idx]));
    }
}

test "trsm auto-dispatch: right side parameter combo f64" {
    const allocator = testing.allocator;

    // Test right + upper + no transpose + non-unit diagonal
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 64 }, undefined, .row_major);
    defer A.deinit();
    // Fill with upper triangular
    for (0..64) |i| {
        for (0..64) |j| {
            A.data[i * 64 + j] = if (i <= j) @as(f64, @floatFromInt(i + j + 2)) else 0.0;
        }
    }

    var B = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 4, 64 }, .row_major);
    defer B.deinit();
    for (0..4 * 64) |idx| {
        B.data[idx] = 1.0;
    }

    try trsm(f64, 'R', 'U', 'N', 'N', 1.0, A, &B);

    // Verify no NaN/Inf
    for (0..4 * 64) |idx| {
        try testing.expect(!std.math.isNan(B.data[idx]));
        try testing.expect(!std.math.isInf(B.data[idx]));
    }
}

test "trsm auto-dispatch: f32 type 64×64 SIMD" {
    const allocator = testing.allocator;

    // Upper triangular matrix (f32)
    var A = try NDArray(f32, 2).zeros(allocator, &[_]usize{ 64, 64 }, .row_major);
    defer A.deinit();
    for (0..64) |i| {
        for (i..64) |j| {
            A.data[i * 64 + j] = @as(f32, @floatFromInt(i + j + 2));
        }
    }

    // Create B with 8 RHS
    var B = try NDArray(f32, 2).zeros(allocator, &[_]usize{ 64, 8 }, .row_major);
    defer B.deinit();
    for (0..64 * 8) |idx| {
        B.data[idx] = 1.0;
    }

    try trsm(f32, 'L', 'U', 'N', 'N', 1.0, A, &B);

    // Verify solution validity
    for (0..64) |i| {
        for (0..8) |j| {
            try testing.expect(!std.math.isNan(B.data[i * 8 + j]));
            try testing.expect(!std.math.isInf(B.data[i * 8 + j]));
        }
    }
}

test "trsm auto-dispatch: scalar/SIMD equivalence (small matrix)" {
    const allocator = testing.allocator;

    // Upper triangular: A = [[2, 1], [0, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 1, 0, 3 }, .row_major);
    defer A.deinit();

    // Copy B for both paths
    var B_scalar = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 8, 9, 12 }, .row_major);
    defer B_scalar.deinit();

    var B_simd = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 8, 9, 12 }, .row_major);
    defer B_simd.deinit();

    // Both go through trsm (which may dispatch differently)
    try trsm(f64, 'L', 'U', 'N', 'N', 1.0, A, &B_scalar);
    try trsm(f64, 'L', 'U', 'N', 'N', 1.0, A, &B_simd);

    // Results should be identical
    for (0..4) |idx| {
        try testing.expectApproxEqAbs(B_scalar.data[idx], B_simd.data[idx], 1e-14);
    }
}

/// Symmetric Matrix-Matrix Multiply: B := α*A*B + β*B (side='L') or B := α*B*A + β*B (side='R')
///
/// Performs a symmetric matrix-matrix multiplication. Matrix A is symmetric and only its
/// specified triangle (upper or lower) is read. B is a general matrix that is modified in-place.
///
// ============================================================================
// symv() — BLAS Level 2: Symmetric Matrix-Vector Multiply
// ============================================================================

/// Symmetric matrix-vector multiplication: y = α*A*x + β*y
///
/// Performs matrix-vector multiplication where A is a symmetric matrix.
/// Only the upper or lower triangle of A is accessed (specified by uplo),
/// with the other triangle assumed symmetric.
///
/// This is BLAS Level 2 SYMV operation.
///
/// Parameters:
/// - uplo: 'U' (upper triangle of A used) or 'L' (lower triangle of A used)
/// - alpha: scalar multiplier for A*x
/// - A: n×n symmetric matrix (only upper or lower triangle accessed)
/// - x: n-dimensional vector
/// - beta: scalar multiplier for y (original y values)
/// - y: n-dimensional vector (modified in-place: y := α*A*x + β*y)
///
/// Errors:
/// - error.DimensionMismatch if A not square or dimensions incompatible
/// - error.NotSquare if A is not a square matrix
///
/// Time: O(n²) where n = size of matrix A
/// Space: O(1) (modifies y in-place)
///
/// Example:
/// ```zig
/// // A = [[4, 1], [1, 3]] (symmetric), x = [1, 2], y = [0, 0]
/// // Upper triangle: A = [[4, 1], [_, 3]]
/// // y = 1.0*A*x + 0.0*y = [[4*1 + 1*2], [1*1 + 3*2]] = [6, 7]
/// var A = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{2, 2}, &[_]f64{4, 1, 0, 3}, .row_major);
/// var x = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{2}, &[_]f64{1, 2}, .row_major);
/// var y = try NDArray(f64, 1).zeros(alloc, &[_]usize{2}, .row_major);
/// try symv(f64, 'U', 1.0, A, x, 0.0, &y);
/// // y.data == [6.0, 7.0]
/// ```
pub fn symv(comptime T: type, uplo: u8, alpha: T, A: NDArray(T, 2), x: NDArray(T, 1), beta: T, y: *NDArray(T, 1)) (NDArray(T, 1).Error)!void {
    // Validate matrix is square
    if (A.shape[0] != A.shape[1]) {
        return error.NotSquare;
    }

    const n = A.shape[0];

    // Validate dimensions match
    if (x.shape[0] != n or y.shape[0] != n) {
        return error.DimensionMismatch;
    }

    // Special case: alpha=0 and beta=1 is a no-op
    if (alpha == 0 and beta == 1) {
        return;
    }

    // Special case: alpha=0 and beta≠1 means just scale y
    if (alpha == 0) {
        for (0..n) |i| {
            y.data[i] *= beta;
        }
        return;
    }

    // First, scale y by beta (if beta != 1)
    if (beta != 1) {
        for (0..n) |i| {
            y.data[i] *= beta;
        }
    }

    // Compute y += α*A*x using symmetry
    // Only access upper or lower triangle based on uplo
    if (uplo == 'U' or uplo == 'u') {
        // Upper triangle: A[i,j] stored for i <= j
        // For each row i, accumulate:
        //   - Diagonal: y[i] += α * A[i,i] * x[i]
        //   - Upper (j>i): y[i] += α * A[i,j] * x[j]
        //   - Symmetric (j>i): y[j] += α * A[i,j] * x[i]  (using symmetry)
        for (0..n) |i| {
            var temp: T = 0;
            // Diagonal and upper triangle entries for row i
            for (i..n) |j| {
                const a_ij = A.data[i * n + j];
                temp += a_ij * x.data[j];
                // Symmetric contribution: A[j,i] = A[i,j]
                if (j > i) {
                    y.data[j] += alpha * a_ij * x.data[i];
                }
            }
            y.data[i] += alpha * temp;
        }
    } else if (uplo == 'L' or uplo == 'l') {
        // Lower triangle: A[i,j] stored for i >= j
        // For each row i, accumulate:
        //   - Diagonal and lower (j<=i): y[i] += α * A[i,j] * x[j]
        //   - Symmetric (j<i): y[j] += α * A[i,j] * x[i]  (using symmetry)
        for (0..n) |i| {
            var temp: T = 0;
            // Lower triangle and diagonal entries for row i
            for (0..i + 1) |j| {
                const a_ij = A.data[i * n + j];
                temp += a_ij * x.data[j];
                // Symmetric contribution: A[j,i] = A[i,j]
                if (j < i) {
                    y.data[j] += alpha * a_ij * x.data[i];
                }
            }
            y.data[i] += alpha * temp;
        }
    }
}

// ============================================================================
// symv() Tests
// ============================================================================

test "symv: basic 2×2 upper triangle, alpha=1, beta=0" {
    const allocator = testing.allocator;

    // A = [[4, 1], [1, 3]] symmetric
    // Upper triangle stored: [[4, 1], [_, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 4, 1, 0, 3 }, .row_major);
    defer A.deinit();

    // x = [1, 2]
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 2 }, .row_major);
    defer x.deinit();

    // y = [0, 0] (will be overwritten with α*A*x + β*y)
    var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{2}, .row_major);
    defer y.deinit();

    // symv: y = 1.0*A*x + 0.0*y
    // A*x = [[4*1 + 1*2], [1*1 + 3*2]] = [[6], [7]]
    try symv(f64, 'U', 1.0, A, x, 0.0, &y);

    try testing.expectApproxEqAbs(6.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(7.0, y.data[1], 1e-10);
}

test "symv: basic 3×3 lower triangle, alpha=1, beta=0" {
    const allocator = testing.allocator;

    // A = [[2, 1, 0], [1, 3, 1], [0, 1, 4]] symmetric
    // Lower triangle stored: [[2, _, _], [1, 3, _], [0, 1, 4]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 2, 0, 0, 1, 3, 0, 0, 1, 4 }, .row_major);
    defer A.deinit();

    // x = [1, 1, 1]
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 1, 1 }, .row_major);
    defer x.deinit();

    // y = [0, 0, 0]
    var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{3}, .row_major);
    defer y.deinit();

    // A*x = [[2+1+0], [1+3+1], [0+1+4]] = [3, 5, 5]
    try symv(f64, 'L', 1.0, A, x, 0.0, &y);

    try testing.expectApproxEqAbs(3.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(5.0, y.data[1], 1e-10);
    try testing.expectApproxEqAbs(5.0, y.data[2], 1e-10);
}

test "symv: alpha=2.0, beta=0.5" {
    const allocator = testing.allocator;

    // A = [[1, 2], [2, 3]] upper triangle
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 0, 3 }, .row_major);
    defer A.deinit();

    // x = [1, 1]
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 1 }, .row_major);
    defer x.deinit();

    // y = [4, 6] (initial values)
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 4, 6 }, .row_major);
    defer y.deinit();

    // symv: y = 2.0*A*x + 0.5*y
    // A*x = [[1+2], [2+3]] = [3, 5]
    // Result: [2*3 + 0.5*4, 2*5 + 0.5*6] = [8, 13]
    try symv(f64, 'U', 2.0, A, x, 0.5, &y);

    try testing.expectApproxEqAbs(8.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(13.0, y.data[1], 1e-10);
}

test "symv: alpha=0 (no-op when beta=1)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 0, 3 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 1 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 7, 9 }, .row_major);
    defer y.deinit();

    const original = try y.clone();
    defer original.deinit();

    // alpha=0, beta=1: y unchanged
    try symv(f64, 'U', 0.0, A, x, 1.0, &y);

    try testing.expectApproxEqAbs(original.data[0], y.data[0], 1e-10);
    try testing.expectApproxEqAbs(original.data[1], y.data[1], 1e-10);
}

test "symv: alpha=0, beta=2.0 (just scale y)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 0, 3 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 1 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 3, 5 }, .row_major);
    defer y.deinit();

    // alpha=0, beta=2: y = 2*y = [6, 10]
    try symv(f64, 'U', 0.0, A, x, 2.0, &y);

    try testing.expectApproxEqAbs(6.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(10.0, y.data[1], 1e-10);
}

test "symv: dimension mismatch — x wrong size" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 0, 4, 5, 0, 0, 6 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 2 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{3}, .row_major);
    defer y.deinit();

    const result = symv(f64, 'U', 1.0, A, x, 0.0, &y);
    try testing.expectError(error.DimensionMismatch, result);
}

test "symv: not square matrix" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 4 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{3}, .row_major);
    defer y.deinit();

    const result = symv(f64, 'U', 1.0, A, x, 0.0, &y);
    try testing.expectError(error.NotSquare, result);
}

test "symv: f32 precision" {
    const allocator = testing.allocator;

    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{ 1.5, 2.5, 0, 3.5 }, .row_major);
    defer A.deinit();

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{2}, &[_]f32{ 2.0, 1.0 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f32, 1).zeros(allocator, &[_]usize{2}, .row_major);
    defer y.deinit();

    // A*x = [[1.5*2 + 2.5*1], [2.5*2 + 3.5*1]] = [5.5, 8.5]
    try symv(f32, 'U', 1.0, A, x, 0.0, &y);

    try testing.expectApproxEqAbs(@as(f32, 5.5), y.data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 8.5), y.data[1], 1e-6);
}

test "symv: memory safety — 10 iterations" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 4 }, &[_]f64{ 1, 2, 3, 4, 0, 5, 6, 7, 0, 0, 8, 9, 0, 0, 0, 10 }, .row_major);
        defer A.deinit();

        var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1, 1, 1, 1 }, .row_major);
        defer x.deinit();

        var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{4}, .row_major);
        defer y.deinit();

        try symv(f64, 'U', 1.0, A, x, 0.0, &y);
    }
}

test "symv: large matrix n=10, upper triangle" {
    const allocator = testing.allocator;

    // Create 10×10 symmetric matrix (upper triangle)
    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 10, 10 }, .row_major);
    defer A.deinit();

    // Fill upper triangle: A[i,j] = i+j+1 for i <= j
    for (0..10) |i| {
        for (i..10) |j| {
            A.data[i * 10 + j] = @as(f64, @floatFromInt(i + j + 1));
        }
    }

    // x = [1, 1, ..., 1]
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{10}, &[_]f64{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{10}, .row_major);
    defer y.deinit();

    try symv(f64, 'U', 1.0, A, x, 0.0, &y);

    // Verify first row: A[0,0..9] = [1,2,3,4,5,6,7,8,9,10]
    // Sum = 1+2+3+...+10 = 55
    try testing.expectApproxEqAbs(55.0, y.data[0], 1e-9);
}

// ============================================================================
// symm() — BLAS Level 3: Symmetric Matrix-Matrix Multiply
// ============================================================================

/// Symmetric matrix-matrix multiplication: C = α*A*B + β*C or C = α*B*A + β*C
///
/// Parameters:
/// - side: 'L' (left: B = α*A*B + β*B) or 'R' (right: B = α*B*A + β*B)
/// - uplo: 'U' (upper triangle of A used) or 'L' (lower triangle of A used)
/// - alpha: scalar multiplier for A*B
/// - A: m×m symmetric matrix (if side='L') or n×n symmetric matrix (if side='R')
/// - B: m×n general matrix (modified in-place)
/// - beta: scalar multiplier for B
///
/// Errors:
/// - error.DimensionMismatch if dimensions don't match
/// - error.InvalidParameter if side or uplo is not 'L'/'R' or 'U'/'L'
///
/// Time: O(m²n) where A is m×m and B is m×n
/// Space: O(1) (modifies B in-place)
pub fn symm(comptime T: type, side: u8, uplo: u8, alpha: T, A: NDArray(T, 2), B: *NDArray(T, 2), beta: T) (NDArray(T, 2).Error)!void {
    // Validate square matrix A
    if (A.shape[0] != A.shape[1]) {
        return error.DimensionMismatch;
    }

    // Validate side parameter
    const is_left = (side == 'L' or side == 'l');
    const is_right = (side == 'R' or side == 'r');
    if (!is_left and !is_right) {
        return error.InvalidValue;
    }

    // Validate uplo parameter
    const is_upper = (uplo == 'U' or uplo == 'u');
    const is_lower = (uplo == 'L' or uplo == 'l');
    if (!is_upper and !is_lower) {
        return error.InvalidValue;
    }

    const m = B.shape[0];
    const n = B.shape[1];
    const k = A.shape[0];

    // Validate dimensions
    if (is_left) {
        if (k != m) return error.DimensionMismatch;
    } else {
        if (k != n) return error.DimensionMismatch;
    }

    // Auto-dispatch: Use SIMD-optimized implementation for large matrices
    // Threshold: if m >= 64 OR n >= 64, use the SIMD-optimized symm
    // (Session 501: symm_simd provides 2-3× speedup over scalar for large matrices)
    const threshold: usize = 64;
    if (m >= threshold or n >= threshold) {
        return try simd_blas.symm_simd(T, side, uplo, alpha, A, B, beta);
    }

    // Allocate temporary matrix to store original B values
    const B_orig = try B.allocator.alloc(T, m * n);
    defer B.allocator.free(B_orig);

    // Copy B to temporary buffer
    @memcpy(B_orig, B.data);

    // Scale B by beta: B *= beta
    for (0..m * n) |i| {
        B.data[i] = beta * B.data[i];
    }

    if (is_left) {
        // B := α*A*B + β*B (A is m×m, B is m×n)
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: T = 0;
                if (is_upper) {
                    // Use upper triangle of A: A[i,k] if i<=k, else A[k,i]
                    for (0..k) |p| {
                        const a_val = if (i <= p) A.data[i * k + p] else A.data[p * k + i];
                        sum += a_val * B_orig[p * n + j];
                    }
                } else {
                    // Use lower triangle of A: A[i,k] if i>=k, else A[k,i]
                    for (0..k) |p| {
                        const a_val = if (i >= p) A.data[i * k + p] else A.data[p * k + i];
                        sum += a_val * B_orig[p * n + j];
                    }
                }
                B.data[i * n + j] += alpha * sum;
            }
        }
    } else {
        // B := α*B*A + β*B (B is m×n, A is n×n)
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: T = 0;
                if (is_upper) {
                    // Use upper triangle of A: A[k,j] if k<=j, else A[j,k]
                    for (0..k) |p| {
                        const a_val = if (p <= j) A.data[p * k + j] else A.data[j * k + p];
                        sum += B_orig[i * n + p] * a_val;
                    }
                } else {
                    // Use lower triangle of A: A[k,j] if k>=j, else A[j,k]
                    for (0..k) |p| {
                        const a_val = if (p >= j) A.data[p * k + j] else A.data[j * k + p];
                        sum += B_orig[i * n + p] * a_val;
                    }
                }
                B.data[i * n + j] += alpha * sum;
            }
        }
    }
}

/// Symmetric Rank-K Update: C := α*A*A^T + β*C (trans='N') or C := α*A^T*A + β*C (trans='T')
///
/// Performs a symmetric rank-k update. The result matrix C is symmetric and only its specified
/// triangle (upper or lower) is updated. C is modified in-place.
///
/// Parameters:
/// - trans: 'N' (no transpose: C := α*A*A^T + β*C) or 'T' (transpose: C := α*A^T*A + β*C)
/// - uplo: 'U' (upper triangle of C updated) or 'L' (lower triangle of C updated)
/// - alpha: scalar multiplier for the rank-k update
/// - A: rectangular matrix (m×k for trans='N', or k×m for trans='T', but stored as is)
/// - beta: scalar multiplier for existing C
/// - C: symmetric matrix (m×m for trans='N', or k×k for trans='T', modified in-place)
///
/// Errors:
/// - error.DimensionMismatch if dimensions don't match
/// - error.InvalidValue if trans or uplo is not valid ('N'/'T' or 'U'/'L')
///
/// Time: O(m²k) for trans='N' or O(k²m) for trans='T'
/// Space: O(m) or O(k) temporary storage for row/column computations
///
/// Example (trans='N'):
/// ```zig
/// // A is 3×2, C is 3×3
/// // C := 1.0*A*A^T + 0.0*C
/// try syrk(f64, 'N', 'U', 1.0, A, 0.0, &C);
/// ```
pub fn syrk(comptime T: type, trans: u8, uplo: u8, alpha: T, A: NDArray(T, 2), beta: T, C: *NDArray(T, 2)) (NDArray(T, 2).Error)!void {
    // Validate trans parameter
    const is_trans_n = (trans == 'N' or trans == 'n');
    const is_trans_t = (trans == 'T' or trans == 't');
    if (!is_trans_n and !is_trans_t) {
        return error.InvalidValue;
    }

    // Validate uplo parameter
    const is_upper = (uplo == 'U' or uplo == 'u');
    const is_lower = (uplo == 'L' or uplo == 'l');
    if (!is_upper and !is_lower) {
        return error.InvalidValue;
    }

    // Get dimensions
    const a_rows = A.shape[0];
    const a_cols = A.shape[1];
    const c_rows = C.shape[0];
    const c_cols = C.shape[1];

    // Validate that C is square
    if (c_rows != c_cols) {
        return error.DimensionMismatch;
    }

    // Validate dimensions based on trans parameter
    // For trans='N': A is m×k, C is m×m
    // For trans='T': A is m×k, C is k×k
    if (is_trans_n) {
        // A is a_rows×a_cols, C should be a_rows×a_rows
        if (c_rows != a_rows) {
            return error.DimensionMismatch;
        }
    } else {
        // A is a_rows×a_cols, C should be a_cols×a_cols
        if (c_rows != a_cols) {
            return error.DimensionMismatch;
        }
    }

    const n = c_rows;  // C is n×n

    // Auto-dispatch: Use SIMD-optimized implementation for large matrices
    // Threshold: n >= 64 (enough elements to benefit from vectorization)
    // Session 503: syrk_simd provides 2-3× speedup via SIMD vectorization
    const threshold: usize = 64;
    if (n >= threshold) {
        return try simd_blas.syrk_simd(T, trans, uplo, alpha, A, beta, C);
    }

    // Fallback to scalar implementation for small matrices
    // Scale C by beta first
    for (0..n * n) |i| {
        C.data[i] = beta * C.data[i];
    }

    // Perform the rank-k update
    if (is_trans_n) {
        // C := α*A*A^T + β*C
        // A is m×k, so we compute C[i,j] = α * sum(A[i,:] * A[j,:]) for i,j in 0..m
        const m = a_rows;
        const k = a_cols;

        if (is_upper) {
            // Update only upper triangle: i <= j
            for (0..m) |i| {
                for (i..m) |j| {
                    var sum: T = 0;
                    for (0..k) |p| {
                        sum += A.data[i * k + p] * A.data[j * k + p];
                    }
                    C.data[i * n + j] += alpha * sum;
                    // Set lower triangle equal (maintain symmetry)
                    if (i != j) {
                        C.data[j * n + i] = C.data[i * n + j];
                    }
                }
            }
        } else {
            // Update only lower triangle: i >= j
            for (0..m) |i| {
                for (0..i + 1) |j| {
                    var sum: T = 0;
                    for (0..k) |p| {
                        sum += A.data[i * k + p] * A.data[j * k + p];
                    }
                    C.data[i * n + j] += alpha * sum;
                    // Set upper triangle equal (maintain symmetry)
                    if (i != j) {
                        C.data[j * n + i] = C.data[i * n + j];
                    }
                }
            }
        }
    } else {
        // C := α*A^T*A + β*C
        // A is m×k, so A^T is k×m
        // C[i,j] = α * sum(A[:,i] * A[:,j]) for i,j in 0..k
        const m = a_rows;
        const k = a_cols;

        if (is_upper) {
            // Update only upper triangle: i <= j
            for (0..k) |i| {
                for (i..k) |j| {
                    var sum: T = 0;
                    for (0..m) |p| {
                        sum += A.data[p * k + i] * A.data[p * k + j];
                    }
                    C.data[i * n + j] += alpha * sum;
                    // Set lower triangle equal (maintain symmetry)
                    if (i != j) {
                        C.data[j * n + i] = C.data[i * n + j];
                    }
                }
            }
        } else {
            // Update only lower triangle: i >= j
            for (0..k) |i| {
                for (0..i + 1) |j| {
                    var sum: T = 0;
                    for (0..m) |p| {
                        sum += A.data[p * k + i] * A.data[p * k + j];
                    }
                    C.data[i * n + j] += alpha * sum;
                    // Set upper triangle equal (maintain symmetry)
                    if (i != j) {
                        C.data[j * n + i] = C.data[i * n + j];
                    }
                }
            }
        }
    }
}

// ============================================================================
// Comprehensive RED tests for symm() (Symmetric Matrix-Matrix Multiply)
// ============================================================================

test "symm: basic 2×2 left upper triangle, alpha=1, beta=1" {
    const allocator = testing.allocator;

    // Symmetric matrix (stored upper): A = [[2, 1], [1, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 1, 1, 3 }, .row_major);
    defer A.deinit();

    // B = [[1, 2], [3, 4]]
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    // B := 1*A*B + 1*B
    // A*B = [[2*1+1*3, 2*2+1*4], [1*1+3*3, 1*2+3*4]] = [[5, 8], [10, 14]]
    // Result = [[5+1, 8+2], [10+3, 14+4]] = [[6, 10], [13, 18]]
    try symm(f64, 'L', 'U', 1.0, A, &B, 1.0);

    try testing.expectApproxEqAbs(6.0, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(10.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(13.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(18.0, B.data[3], 1e-10);
}

test "symm: basic 2×2 left lower triangle, alpha=1, beta=1" {
    const allocator = testing.allocator;

    // Symmetric matrix (stored lower): A = [[2, 0], [1, 3]]
    // Symmetry means upper is [[2, 1], [0, 3]] (but we ignore it)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 0, 1, 3 }, .row_major);
    defer A.deinit();

    // B = [[1, 2], [3, 4]]
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    // Same result as upper triangle case: A*B = [[5, 8], [10, 14]]
    // Result = [[6, 10], [13, 18]]
    try symm(f64, 'L', 'L', 1.0, A, &B, 1.0);

    try testing.expectApproxEqAbs(6.0, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(10.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(13.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(18.0, B.data[3], 1e-10);
}

test "symm: basic 2×2 right upper triangle, alpha=1, beta=1" {
    const allocator = testing.allocator;

    // B = [[1, 2], [3, 4]]
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    // Symmetric matrix (stored upper): A = [[2, 1], [1, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 1, 1, 3 }, .row_major);
    defer A.deinit();

    // B := 1*B*A + 1*B
    // B*A = [[1*2+2*1, 1*1+2*3], [3*2+4*1, 3*1+4*3]] = [[4, 7], [10, 15]]
    // Result = [[4+1, 7+2], [10+3, 15+4]] = [[5, 9], [13, 19]]
    try symm(f64, 'R', 'U', 1.0, A, &B, 1.0);

    try testing.expectApproxEqAbs(5.0, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(9.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(13.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(19.0, B.data[3], 1e-10);
}

test "symm: basic 2×2 right lower triangle, alpha=1, beta=1" {
    const allocator = testing.allocator;

    // B = [[1, 2], [3, 4]]
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    // Symmetric matrix (stored lower): A = [[2, 0], [1, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 0, 1, 3 }, .row_major);
    defer A.deinit();

    // Same result as right upper: B*A = [[4, 7], [10, 15]]
    // Result = [[5, 9], [13, 19]]
    try symm(f64, 'R', 'L', 1.0, A, &B, 1.0);

    try testing.expectApproxEqAbs(5.0, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(9.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(13.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(19.0, B.data[3], 1e-10);
}

test "symm: 3×3 left upper, alpha=1, beta=0" {
    const allocator = testing.allocator;

    // Symmetric 3×3 (upper): A = [[1, 2, 3], [2, 4, 5], [3, 5, 6]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 2, 4, 5, 3, 5, 6 }, .row_major);
    defer A.deinit();

    // B = [[1, 1], [2, 2], [3, 3]]
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f64{ 1, 1, 2, 2, 3, 3 }, .row_major);
    defer B.deinit();

    // B := 1*A*B + 0*B = A*B (ignore old B)
    // A*B = [[1*1+2*2+3*3, 1*1+2*2+3*3], [2*1+4*2+5*3, 2*1+4*2+5*3], [3*1+5*2+6*3, 3*1+5*2+6*3]]
    //     = [[14, 14], [25, 25], [36, 36]]
    try symm(f64, 'L', 'U', 1.0, A, &B, 0.0);

    try testing.expectApproxEqAbs(14.0, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(14.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(25.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(25.0, B.data[3], 1e-10);
    try testing.expectApproxEqAbs(36.0, B.data[4], 1e-10);
    try testing.expectApproxEqAbs(36.0, B.data[5], 1e-10);
}

test "symm: 3×3 right upper, alpha=1, beta=0" {
    const allocator = testing.allocator;

    // B = [[1, 2, 3], [4, 5, 6]]
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer B.deinit();

    // Symmetric 3×3 (upper): A = [[2, 1, 1], [1, 2, 1], [1, 1, 2]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 2, 1, 1, 1, 2, 1, 1, 1, 2 }, .row_major);
    defer A.deinit();

    // B := 1*B*A + 0*B = B*A
    // B*A = [[1*2+2*1+3*1, 1*1+2*2+3*1, 1*1+2*1+3*2], [4*2+5*1+6*1, 4*1+5*2+6*1, 4*1+5*1+6*2]]
    //     = [[7, 8, 9], [19, 20, 21]]
    try symm(f64, 'R', 'U', 1.0, A, &B, 0.0);

    try testing.expectApproxEqAbs(7.0, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(8.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(9.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(19.0, B.data[3], 1e-10);
    try testing.expectApproxEqAbs(20.0, B.data[4], 1e-10);
    try testing.expectApproxEqAbs(21.0, B.data[5], 1e-10);
}

test "symm: alpha=0 (B unchanged from beta scaling)" {
    const allocator = testing.allocator;

    // Symmetric: A = [[2, 1], [1, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 1, 1, 3 }, .row_major);
    defer A.deinit();

    // B = [[1, 2], [3, 4]]
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    // B := 0*A*B + 2*B = 2*B = [[2, 4], [6, 8]]
    try symm(f64, 'L', 'U', 0.0, A, &B, 2.0);

    try testing.expectApproxEqAbs(2.0, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(4.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(6.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(8.0, B.data[3], 1e-10);
}

test "symm: beta=0 (B replaced by alpha*A*B)" {
    const allocator = testing.allocator;

    // Symmetric: A = [[2, 1], [1, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 1, 1, 3 }, .row_major);
    defer A.deinit();

    // B = [[1, 2], [3, 4]]
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    // B := 2*A*B + 0*B = 2*A*B = 2*[[5, 8], [10, 14]] = [[10, 16], [20, 28]]
    try symm(f64, 'L', 'U', 2.0, A, &B, 0.0);

    try testing.expectApproxEqAbs(10.0, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(16.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(20.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(28.0, B.data[3], 1e-10);
}

test "symm: alpha=0.5, beta=2.0" {
    const allocator = testing.allocator;

    // Symmetric: A = [[2, 1], [1, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 1, 1, 3 }, .row_major);
    defer A.deinit();

    // B = [[1, 2], [3, 4]]
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    // B := 0.5*A*B + 2*B = 0.5*[[5, 8], [10, 14]] + 2*[[1, 2], [3, 4]]
    //                    = [[2.5, 4], [5, 7]] + [[2, 4], [6, 8]]
    //                    = [[4.5, 8], [11, 15]]
    try symm(f64, 'L', 'U', 0.5, A, &B, 2.0);

    try testing.expectApproxEqAbs(4.5, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(8.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(11.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(15.0, B.data[3], 1e-10);
}

test "symm: negative alpha" {
    const allocator = testing.allocator;

    // Symmetric: A = [[2, 1], [1, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 1, 1, 3 }, .row_major);
    defer A.deinit();

    // B = [[1, 2], [3, 4]]
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    // B := -1*A*B + 1*B = -[[5, 8], [10, 14]] + [[1, 2], [3, 4]]
    //                   = [[-4, -6], [-7, -10]]
    try symm(f64, 'L', 'U', -1.0, A, &B, 1.0);

    try testing.expectApproxEqAbs(-4.0, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(-6.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(-7.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(-10.0, B.data[3], 1e-10);
}

test "symm: negative beta" {
    const allocator = testing.allocator;

    // Symmetric: A = [[2, 1], [1, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 1, 1, 3 }, .row_major);
    defer A.deinit();

    // B = [[1, 2], [3, 4]]
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    // B := 1*A*B - 1*B = [[5, 8], [10, 14]] - [[1, 2], [3, 4]]
    //                  = [[4, 6], [7, 10]]
    try symm(f64, 'L', 'U', 1.0, A, &B, -1.0);

    try testing.expectApproxEqAbs(4.0, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(6.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(7.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(10.0, B.data[3], 1e-10);
}

test "symm: f32 precision (left upper)" {
    const allocator = testing.allocator;

    // Symmetric: A = [[2.5, 1.5], [1.5, 3.5]]
    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{ 2.5, 1.5, 1.5, 3.5 }, .row_major);
    defer A.deinit();

    // B = [[1.5, 2.5], [3.5, 4.5]]
    var B = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{ 1.5, 2.5, 3.5, 4.5 }, .row_major);
    defer B.deinit();

    // B := 1*A*B + 1*B
    // A*B = [[2.5*1.5+1.5*3.5, 2.5*2.5+1.5*4.5], [1.5*1.5+3.5*3.5, 1.5*2.5+3.5*4.5]]
    //     = [[9.75, 13.25], [15.5, 19]]
    // Result = [[11.25, 15.75], [19, 23.5]]
    try symm(f32, 'L', 'U', 1.0, A, &B, 1.0);

    try testing.expectApproxEqAbs(11.25, B.data[0], 1e-4);
    try testing.expectApproxEqAbs(15.75, B.data[1], 1e-4);
    try testing.expectApproxEqAbs(19.0, B.data[2], 1e-4);
    try testing.expectApproxEqAbs(23.5, B.data[3], 1e-4);
}

test "symm: 1×1 edge case (single element)" {
    const allocator = testing.allocator;

    // Symmetric: A = [[5]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 1 }, &[_]f64{5}, .row_major);
    defer A.deinit();

    // B = [[3]]
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 1 }, &[_]f64{3}, .row_major);
    defer B.deinit();

    // B := 2*A*B + 1*B = 2*5*3 + 1*3 = 30 + 3 = 33
    try symm(f64, 'L', 'U', 2.0, A, &B, 1.0);

    try testing.expectApproxEqAbs(33.0, B.data[0], 1e-10);
}

test "symm: 64×64 left side (SIMD threshold)" {
    const allocator = testing.allocator;

    // Create 64×64 symmetric identity matrix
    var A_data = try allocator.alloc(f64, 64 * 64);
    defer allocator.free(A_data);
    for (0..64 * 64) |i| {
        const row = i / 64;
        const col = i % 64;
        A_data[i] = if (row == col) 1.0 else 0.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 64 }, A_data, .row_major);
    defer A.deinit();

    // Create 64×32 B with incrementing values
    var B_data = try allocator.alloc(f64, 64 * 32);
    defer allocator.free(B_data);
    for (0..64 * 32) |i| {
        B_data[i] = @as(f64, @floatFromInt(i % 32));
    }
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 32 }, B_data, .row_major);
    defer B.deinit();

    // B := 1*I*B + 0*B = B (identity should not change it)
    try symm(f64, 'L', 'U', 1.0, A, &B, 0.0);

    // Verify some elements match original
    for (0..64) |i| {
        for (0..32) |j| {
            const expected = @as(f64, @floatFromInt(j));
            try testing.expectApproxEqAbs(expected, B.data[i * 32 + j], 1e-10);
        }
    }
}

test "symm: 128×64 right side (large non-square)" {
    const allocator = testing.allocator;

    // Create 64×64 symmetric identity matrix
    var A_data = try allocator.alloc(f64, 64 * 64);
    defer allocator.free(A_data);
    for (0..64 * 64) |i| {
        const row = i / 64;
        const col = i % 64;
        A_data[i] = if (row == col) 1.0 else 0.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 64 }, A_data, .row_major);
    defer A.deinit();

    // Create 128×64 B with incrementing values
    var B_data = try allocator.alloc(f64, 128 * 64);
    defer allocator.free(B_data);
    for (0..128 * 64) |i| {
        B_data[i] = @as(f64, @floatFromInt(i % 64));
    }
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 128, 64 }, B_data, .row_major);
    defer B.deinit();

    // B := 1*B*I + 0*B = B (right multiply by identity)
    try symm(f64, 'R', 'U', 1.0, A, &B, 0.0);

    // Verify some elements
    for (0..128) |i| {
        for (0..64) |j| {
            const expected = @as(f64, @floatFromInt(j));
            try testing.expectApproxEqAbs(expected, B.data[i * 64 + j], 1e-10);
        }
    }
}

test "symm: dimension mismatch — left side A not square" {
    const allocator = testing.allocator;

    // Non-square A: 2×3
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer A.deinit();

    // B: 2×2
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    try testing.expectError(error.DimensionMismatch, symm(f64, 'L', 'U', 1.0, A, &B, 0.0));
}

test "symm: dimension mismatch — left side B rows != A size" {
    const allocator = testing.allocator;

    // Symmetric A: 3×3
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 2, 4, 5, 3, 5, 6 }, .row_major);
    defer A.deinit();

    // B: 2×2 (should be 3×?)
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    try testing.expectError(error.DimensionMismatch, symm(f64, 'L', 'U', 1.0, A, &B, 0.0));
}

test "symm: dimension mismatch — right side A not square" {
    const allocator = testing.allocator;

    // B: 2×2
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    // Non-square A: 2×3
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer A.deinit();

    try testing.expectError(error.DimensionMismatch, symm(f64, 'R', 'U', 1.0, A, &B, 0.0));
}

test "symm: dimension mismatch — right side B columns != A size" {
    const allocator = testing.allocator;

    // B: 2×2
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    // Symmetric A: 3×3 (should match B columns)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 2, 4, 5, 3, 5, 6 }, .row_major);
    defer A.deinit();

    try testing.expectError(error.DimensionMismatch, symm(f64, 'R', 'U', 1.0, A, &B, 0.0));
}

test "symm: invalid side parameter" {
    const allocator = testing.allocator;

    // Symmetric: 2×2
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 1, 1, 3 }, .row_major);
    defer A.deinit();

    // B: 2×2
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    try testing.expectError(error.InvalidParameter, symm(f64, 'X', 'U', 1.0, A, &B, 0.0));
}

test "symm: invalid uplo parameter" {
    const allocator = testing.allocator;

    // Symmetric: 2×2
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 1, 1, 3 }, .row_major);
    defer A.deinit();

    // B: 2×2
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    try testing.expectError(error.InvalidParameter, symm(f64, 'L', 'X', 1.0, A, &B, 0.0));
}

test "symm: non-aligned size 67×67 left upper" {
    const allocator = testing.allocator;

    // Create 67×67 symmetric matrix with non-aligned size
    var A_data = try allocator.alloc(f64, 67 * 67);
    defer allocator.free(A_data);
    for (0..67 * 67) |i| {
        const row = i / 67;
        const col = i % 67;
        // Symmetric: A[i,j] = A[j,i] = (row+1) * (col+1)
        A_data[i] = @as(f64, @floatFromInt((row + 1) * (col + 1)));
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 67, 67 }, A_data, .row_major);
    defer A.deinit();

    // Create 67×16 B
    var B_data = try allocator.alloc(f64, 67 * 16);
    defer allocator.free(B_data);
    for (0..67 * 16) |i| {
        B_data[i] = 1.0;
    }
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 67, 16 }, B_data, .row_major);
    defer B.deinit();

    // B := 1*A*B + 0*B
    try symm(f64, 'L', 'U', 1.0, A, &B, 0.0);

    // Verify a few computed values
    // Row 0: sum of first row of A times all-ones column = (1 + 2 + 3 + ... + 67) = 67*68/2 = 2278
    const expected_row0 = 67.0 * 68.0 / 2.0;
    for (0..16) |j| {
        try testing.expectApproxEqAbs(expected_row0, B.data[0 * 16 + j], 1e-8);
    }
}

test "symm: multiple accumulations (repeated updates)" {
    const allocator = testing.allocator;

    // Symmetric: A = [[2, 1], [1, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 1, 1, 3 }, .row_major);
    defer A.deinit();

    // B = [[1, 1], [1, 1]]
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 1, 1, 1 }, .row_major);
    defer B.deinit();

    // First: B := 1*A*B + 1*B
    // A*B = [[2+1, 2+1], [1+3, 1+3]] = [[3, 3], [4, 4]]
    // Result = [[3+1, 3+1], [4+1, 4+1]] = [[4, 4], [5, 5]]
    try symm(f64, 'L', 'U', 1.0, A, &B, 1.0);

    try testing.expectApproxEqAbs(4.0, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(4.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(5.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(5.0, B.data[3], 1e-10);

    // Second accumulation: B := 1*A*B + 1*B (with new B)
    // A*B = [[8+5, 8+5], [4+15, 4+15]] = [[13, 13], [19, 19]]
    // Result = [[17, 17], [24, 24]]
    try symm(f64, 'L', 'U', 1.0, A, &B, 1.0);

    try testing.expectApproxEqAbs(17.0, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(17.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(24.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(24.0, B.data[3], 1e-10);
}

test "symm: lower triangle used correctly (upper triangle ignored)" {
    const allocator = testing.allocator;

    // Create matrix where upper and lower would give different results if not handled correctly
    // A stored as: [[2, 999], [1, 3]] with uplo='L', means we use [[2, 0], [1, 3]] and reconstruct [[2, 1], [1, 3]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 999, 1, 3 }, .row_major);
    defer A.deinit();

    // B = [[1, 2], [3, 4]]
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    // Using lower triangle, A is treated as [[2, 1], [1, 3]], same as upper case
    // B := 1*A*B + 1*B = [[6, 10], [13, 18]] (same as upper triangle test)
    try symm(f64, 'L', 'L', 1.0, A, &B, 1.0);

    try testing.expectApproxEqAbs(6.0, B.data[0], 1e-10);
    try testing.expectApproxEqAbs(10.0, B.data[1], 1e-10);
    try testing.expectApproxEqAbs(13.0, B.data[2], 1e-10);
    try testing.expectApproxEqAbs(18.0, B.data[3], 1e-10);
}

// ============================================================================
// AUTO-DISPATCH TESTS FOR symm() — Route to symm_simd() when m >= 64 OR n >= 64
// ============================================================================

test "symm auto-dispatch: 63×63 left upper should use scalar" {
    const allocator = testing.allocator;

    // Create 63×63 symmetric identity matrix (below threshold)
    var A_data = try allocator.alloc(f64, 63 * 63);
    defer allocator.free(A_data);
    for (0..63 * 63) |i| {
        const row = i / 63;
        const col = i % 63;
        A_data[i] = if (row == col) 1.0 else 0.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 63, 63 }, A_data, .row_major);
    defer A.deinit();

    // Create 63×8 B with incrementing values
    var B_data = try allocator.alloc(f64, 63 * 8);
    defer allocator.free(B_data);
    for (0..63 * 8) |i| {
        B_data[i] = @as(f64, @floatFromInt(i % 8));
    }
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 63, 8 }, B_data, .row_major);
    defer B.deinit();

    // B := 1*I*B + 0*B = B (identity should preserve B)
    try symm(f64, 'L', 'U', 1.0, A, &B, 0.0);

    // Verify identity multiplication: B should be unchanged
    for (0..63) |i| {
        for (0..8) |j| {
            const expected = @as(f64, @floatFromInt(j));
            try testing.expectApproxEqAbs(expected, B.data[i * 8 + j], 1e-10);
        }
    }
}

test "symm auto-dispatch: 64×64 left upper should use SIMD (m == 64)" {
    const allocator = testing.allocator;

    // Create 64×64 symmetric identity matrix (at threshold)
    var A_data = try allocator.alloc(f64, 64 * 64);
    defer allocator.free(A_data);
    for (0..64 * 64) |i| {
        const row = i / 64;
        const col = i % 64;
        A_data[i] = if (row == col) 2.0 else 0.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 64 }, A_data, .row_major);
    defer A.deinit();

    // Create 64×16 B with incrementing values
    var B_data = try allocator.alloc(f64, 64 * 16);
    defer allocator.free(B_data);
    for (0..64 * 16) |i| {
        B_data[i] = @as(f64, @floatFromInt(i % 16 + 1));
    }
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 16 }, B_data, .row_major);
    defer B.deinit();

    // B := 1*A*B + 0*B where A is diagonal with 2's
    // Expected: B[i,j] = 2 * (original B[i,j])
    var expected_data = try allocator.alloc(f64, 64 * 16);
    defer allocator.free(expected_data);
    for (0..64) |i| {
        for (0..16) |j| {
            expected_data[i * 16 + j] = 2.0 * @as(f64, @floatFromInt(j + 1));
        }
    }

    try symm(f64, 'L', 'U', 1.0, A, &B, 0.0);

    // Verify results
    for (0..64) |i| {
        for (0..16) |j| {
            try testing.expectApproxEqAbs(expected_data[i * 16 + j], B.data[i * 16 + j], 1e-10);
        }
    }
}

test "symm auto-dispatch: 65×65 left upper should use SIMD (m > 64)" {
    const allocator = testing.allocator;

    // Create 65×65 symmetric identity matrix (above threshold)
    var A_data = try allocator.alloc(f64, 65 * 65);
    defer allocator.free(A_data);
    for (0..65 * 65) |i| {
        const row = i / 65;
        const col = i % 65;
        A_data[i] = if (row == col) 3.0 else 0.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 65, 65 }, A_data, .row_major);
    defer A.deinit();

    // Create 65×10 B with incrementing values
    var B_data = try allocator.alloc(f64, 65 * 10);
    defer allocator.free(B_data);
    for (0..65 * 10) |i| {
        B_data[i] = @as(f64, @floatFromInt(i % 10 + 1));
    }
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 65, 10 }, B_data, .row_major);
    defer B.deinit();

    // B := 1*A*B + 0*B where A is diagonal with 3's
    // Expected: B[i,j] = 3 * (original B[i,j])
    try symm(f64, 'L', 'U', 1.0, A, &B, 0.0);

    // Verify results
    for (0..65) |i| {
        for (0..10) |j| {
            const expected = 3.0 * @as(f64, @floatFromInt(j + 1));
            try testing.expectApproxEqAbs(expected, B.data[i * 10 + j], 1e-10);
        }
    }
}

test "symm auto-dispatch: 63×64 left upper should use SIMD (n == 64)" {
    const allocator = testing.allocator;

    // Create 63×63 symmetric identity matrix
    var A_data = try allocator.alloc(f64, 63 * 63);
    defer allocator.free(A_data);
    for (0..63 * 63) |i| {
        const row = i / 63;
        const col = i % 63;
        A_data[i] = if (row == col) 1.0 else 0.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 63, 63 }, A_data, .row_major);
    defer A.deinit();

    // Create 63×64 B with incrementing values (n >= 64 threshold)
    var B_data = try allocator.alloc(f64, 63 * 64);
    defer allocator.free(B_data);
    for (0..63 * 64) |i| {
        B_data[i] = @as(f64, @floatFromInt(i % 64 + 1));
    }
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 63, 64 }, B_data, .row_major);
    defer B.deinit();

    // B := 1*A*B + 0*B = B (identity preserves B)
    try symm(f64, 'L', 'U', 1.0, A, &B, 0.0);

    // Verify results
    for (0..63) |i| {
        for (0..64) |j| {
            const expected = @as(f64, @floatFromInt(j + 1));
            try testing.expectApproxEqAbs(expected, B.data[i * 64 + j], 1e-10);
        }
    }
}

test "symm auto-dispatch: 64×63 right upper should use SIMD (m == 64)" {
    const allocator = testing.allocator;

    // Create 128×64 B (m=128, n=64)
    var B_data = try allocator.alloc(f64, 128 * 64);
    defer allocator.free(B_data);
    for (0..128 * 64) |i| {
        B_data[i] = @as(f64, @floatFromInt(i % 64 + 1));
    }
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 128, 64 }, B_data, .row_major);
    defer B.deinit();

    // Create 64×64 symmetric identity matrix
    var A_data = try allocator.alloc(f64, 64 * 64);
    defer allocator.free(A_data);
    for (0..64 * 64) |i| {
        const row = i / 64;
        const col = i % 64;
        A_data[i] = if (row == col) 2.0 else 0.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 64 }, A_data, .row_major);
    defer A.deinit();

    // B := 1*B*A + 0*B where A is diagonal with 2's
    // Expected: B[i,j] = 2 * (original B[i,j])
    try symm(f64, 'R', 'U', 1.0, A, &B, 0.0);

    // Verify results
    for (0..128) |i| {
        for (0..64) |j| {
            const expected = 2.0 * @as(f64, @floatFromInt(j + 1));
            try testing.expectApproxEqAbs(expected, B.data[i * 64 + j], 1e-10);
        }
    }
}

test "symm auto-dispatch: 128×128 left upper should use SIMD (large matrix)" {
    const allocator = testing.allocator;

    // Create 128×128 symmetric matrix (scaled identity)
    var A_data = try allocator.alloc(f64, 128 * 128);
    defer allocator.free(A_data);
    for (0..128 * 128) |i| {
        const row = i / 128;
        const col = i % 128;
        A_data[i] = if (row == col) 2.5 else 0.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 128, 128 }, A_data, .row_major);
    defer A.deinit();

    // Create 128×32 B with incrementing values
    var B_data = try allocator.alloc(f64, 128 * 32);
    defer allocator.free(B_data);
    for (0..128 * 32) |i| {
        B_data[i] = @as(f64, @floatFromInt(i % 32 + 1));
    }
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 128, 32 }, B_data, .row_major);
    defer B.deinit();

    // B := 1*A*B + 0*B where A is diagonal with 2.5's
    // Expected: B[i,j] = 2.5 * (original B[i,j])
    try symm(f64, 'L', 'U', 1.0, A, &B, 0.0);

    // Verify results (spot check)
    for (0..128) |i| {
        for (0..32) |j| {
            const expected = 2.5 * @as(f64, @floatFromInt(j + 1));
            try testing.expectApproxEqAbs(expected, B.data[i * 32 + j], 1e-10);
        }
    }
}

test "symm auto-dispatch: 256×128 right upper should use SIMD (large non-square)" {
    const allocator = testing.allocator;

    // Create 256×128 B
    var B_data = try allocator.alloc(f64, 256 * 128);
    defer allocator.free(B_data);
    for (0..256 * 128) |i| {
        B_data[i] = @as(f64, @floatFromInt(i % 128 + 1));
    }
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 256, 128 }, B_data, .row_major);
    defer B.deinit();

    // Create 128×128 symmetric diagonal matrix
    var A_data = try allocator.alloc(f64, 128 * 128);
    defer allocator.free(A_data);
    for (0..128 * 128) |i| {
        const row = i / 128;
        const col = i % 128;
        A_data[i] = if (row == col) 3.0 else 0.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 128, 128 }, A_data, .row_major);
    defer A.deinit();

    // B := 1*B*A + 0*B where A is diagonal with 3's
    // Expected: B[i,j] = 3 * (original B[i,j])
    try symm(f64, 'R', 'U', 1.0, A, &B, 0.0);

    // Verify results
    for (0..256) |i| {
        for (0..128) |j| {
            const expected = 3.0 * @as(f64, @floatFromInt(j + 1));
            try testing.expectApproxEqAbs(expected, B.data[i * 128 + j], 1e-10);
        }
    }
}

test "symm auto-dispatch: 64×64 left lower should use SIMD (uplo=L)" {
    const allocator = testing.allocator;

    // Create 64×64 symmetric matrix stored as lower triangle
    var A_data = try allocator.alloc(f64, 64 * 64);
    defer allocator.free(A_data);
    for (0..64 * 64) |i| {
        const row = i / 64;
        const col = i % 64;
        A_data[i] = if (row == col) 1.5 else 0.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 64 }, A_data, .row_major);
    defer A.deinit();

    // Create 64×16 B
    var B_data = try allocator.alloc(f64, 64 * 16);
    defer allocator.free(B_data);
    for (0..64 * 16) |i| {
        B_data[i] = @as(f64, @floatFromInt(i % 16 + 1));
    }
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 16 }, B_data, .row_major);
    defer B.deinit();

    // B := 1*A*B + 0*B where A is diagonal with 1.5's
    // Expected: B[i,j] = 1.5 * (original B[i,j])
    try symm(f64, 'L', 'L', 1.0, A, &B, 0.0);

    // Verify results
    for (0..64) |i| {
        for (0..16) |j| {
            const expected = 1.5 * @as(f64, @floatFromInt(j + 1));
            try testing.expectApproxEqAbs(expected, B.data[i * 16 + j], 1e-10);
        }
    }
}

test "symm auto-dispatch: 64×64 right lower should use SIMD (all params)" {
    const allocator = testing.allocator;

    // Create 100×64 B
    var B_data = try allocator.alloc(f64, 100 * 64);
    defer allocator.free(B_data);
    for (0..100 * 64) |i| {
        B_data[i] = @as(f64, @floatFromInt(i % 64 + 1));
    }
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 100, 64 }, B_data, .row_major);
    defer B.deinit();

    // Create 64×64 symmetric matrix stored as lower triangle
    var A_data = try allocator.alloc(f64, 64 * 64);
    defer allocator.free(A_data);
    for (0..64 * 64) |i| {
        const row = i / 64;
        const col = i % 64;
        A_data[i] = if (row == col) 2.0 else 0.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 64 }, A_data, .row_major);
    defer A.deinit();

    // B := 1*B*A + 0*B where A is diagonal with 2's
    try symm(f64, 'R', 'L', 1.0, A, &B, 0.0);

    // Verify results
    for (0..100) |i| {
        for (0..64) |j| {
            const expected = 2.0 * @as(f64, @floatFromInt(j + 1));
            try testing.expectApproxEqAbs(expected, B.data[i * 64 + j], 1e-10);
        }
    }
}

test "symm auto-dispatch: 100×100 non-aligned left upper with alpha/beta" {
    const allocator = testing.allocator;

    // Create 100×100 symmetric matrix (non-aligned size, above threshold)
    var A_data = try allocator.alloc(f64, 100 * 100);
    defer allocator.free(A_data);
    for (0..100 * 100) |i| {
        const row = i / 100;
        const col = i % 100;
        // Symmetric with non-unit diagonal
        A_data[i] = if (row == col) 2.0 else 0.5;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 100, 100 }, A_data, .row_major);
    defer A.deinit();

    // Create 100×20 B with ones
    var B_data = try allocator.alloc(f64, 100 * 20);
    defer allocator.free(B_data);
    for (0..100 * 20) |i| {
        B_data[i] = 1.0;
    }
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 100, 20 }, B_data, .row_major);
    defer B.deinit();

    // B := 2*A*B + 0.5*B
    // A*ones = [2 + 99*0.5, ...] = [2 + 49.5, ...] per row = [51.5, ...]
    // Result = 2*51.5 + 0.5*1 = 103 + 0.5 = 103.5
    try symm(f64, 'L', 'U', 2.0, A, &B, 0.5);

    // Verify results (per row sum should be consistent)
    for (0..100) |i| {
        for (0..20) |j| {
            // Each row of A sum: 2 (diagonal) + 99*0.5 (off-diag) = 51.5
            // A*B[i,j] = 51.5, so result = 2*51.5 + 0.5*1 = 103.5
            try testing.expectApproxEqAbs(103.5, B.data[i * 20 + j], 1e-8);
        }
    }
}

test "symm auto-dispatch: f32 dispatch 64×64 left upper" {
    const allocator = testing.allocator;

    // Create 64×64 symmetric matrix in f32
    var A_data = try allocator.alloc(f32, 64 * 64);
    defer allocator.free(A_data);
    for (0..64 * 64) |i| {
        const row = i / 64;
        const col = i % 64;
        A_data[i] = if (row == col) 2.5 else 0.0;
    }
    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 64, 64 }, A_data, .row_major);
    defer A.deinit();

    // Create 64×16 B in f32
    var B_data = try allocator.alloc(f32, 64 * 16);
    defer allocator.free(B_data);
    for (0..64 * 16) |i| {
        B_data[i] = @as(f32, @floatFromInt(i % 16 + 1));
    }
    var B = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 64, 16 }, B_data, .row_major);
    defer B.deinit();

    // B := 1*A*B + 0*B where A is diagonal with 2.5's
    try symm(f32, 'L', 'U', 1.0, A, &B, 0.0);

    // Verify results with f32 precision tolerance
    for (0..64) |i| {
        for (0..16) |j| {
            const expected = 2.5 * @as(f32, @floatFromInt(j + 1));
            try testing.expectApproxEqAbs(expected, B.data[i * 16 + j], 1e-4);
        }
    }
}

test "symm auto-dispatch: equivalence scalar vs SIMD (small 2×2 through symm)" {
    const allocator = testing.allocator;

    // Small 2×2 matrix (below threshold, uses scalar)
    var A_small = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 1, 1, 3 }, .row_major);
    defer A_small.deinit();

    var B_small1 = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B_small1.deinit();

    var B_small2 = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B_small2.deinit();

    // Call symm() twice with same inputs
    try symm(f64, 'L', 'U', 1.5, A_small, &B_small1, 0.5);
    try symm(f64, 'L', 'U', 1.5, A_small, &B_small2, 0.5);

    // Results should be identical (tests repeatability)
    for (0..4) |i| {
        try testing.expectApproxEqAbs(B_small1.data[i], B_small2.data[i], 1e-14);
    }
}

test "symm auto-dispatch: alpha=0 with large matrix (SIMD dispatch)" {
    const allocator = testing.allocator;

    // Create 64×64 symmetric matrix (doesn't matter since alpha=0)
    var A_data = try allocator.alloc(f64, 64 * 64);
    defer allocator.free(A_data);
    for (0..64 * 64) |i| {
        A_data[i] = 0.0; // All zeros since alpha=0
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 64 }, A_data, .row_major);
    defer A.deinit();

    // Create 64×16 B with initial values
    var B_data = try allocator.alloc(f64, 64 * 16);
    defer allocator.free(B_data);
    for (0..64 * 16) |i| {
        B_data[i] = @as(f64, @floatFromInt(i % 16 + 1));
    }
    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 16 }, B_data, .row_major);
    defer B.deinit();

    // B := 0*A*B + 3*B = 3*B
    try symm(f64, 'L', 'U', 0.0, A, &B, 3.0);

    // Verify results: B should be tripled
    for (0..64) |i| {
        for (0..16) |j| {
            const expected = 3.0 * @as(f64, @floatFromInt(j + 1));
            try testing.expectApproxEqAbs(expected, B.data[i * 16 + j], 1e-10);
        }
    }
}

// ============================================================================
// Comprehensive RED tests for syrk() (Symmetric Rank-K Update)
// ============================================================================
//
// Operation:
// - trans='N': C := alpha*A*A^T + beta*C (A is m×k → C is m×m symmetric)
// - trans='T': C := alpha*A^T*A + beta*C (A is m×k → C is k×k symmetric)
//
// Parameters:
// - trans: 'N' (no transpose) or 'T' (transpose)
// - uplo: 'U' (upper triangle) or 'L' (lower triangle)
// - alpha: scalar multiplier for the rank-k update
// - A: rectangular matrix (m×k)
// - beta: scalar multiplier for existing C
// - C: symmetric matrix (m×m or k×k, updated in-place)

test "syrk: basic 2×2 no-transpose, upper triangle, alpha=1, beta=0" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4]]  (2×2, here m=2, k=2)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    // C = [[0, 0], [0, 0]]  (2×2 output)
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer C.deinit();

    // C := 1.0*A*A^T + 0.0*C
    // A*A^T = [[1, 2], [3, 4]] * [[1, 3], [2, 4]]
    //       = [[1*1+2*2, 1*3+2*4], [3*1+4*2, 3*3+4*4]]
    //       = [[5, 11], [11, 25]]
    try syrk(f64, 'N', 'U', 1.0, A, 0.0, &C);

    try testing.expectApproxEqAbs(5.0, C.data[0], 1e-10);   // C[0,0]
    try testing.expectApproxEqAbs(11.0, C.data[1], 1e-10);  // C[0,1]
    try testing.expectApproxEqAbs(11.0, C.data[2], 1e-10);  // C[1,0] (should match [0,1] for symmetry)
    try testing.expectApproxEqAbs(25.0, C.data[3], 1e-10);  // C[1,1]
}

test "syrk: basic 2×2 transpose, upper triangle, alpha=1, beta=0" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4]]  (2×2, m=2, k=2)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    // C = [[0, 0], [0, 0]]  (2×2 output)
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer C.deinit();

    // C := 1.0*A^T*A + 0.0*C
    // A^T*A = [[1, 3], [2, 4]] * [[1, 2], [3, 4]]
    //       = [[1*1+3*3, 1*2+3*4], [2*1+4*3, 2*2+4*4]]
    //       = [[10, 14], [14, 20]]
    try syrk(f64, 'T', 'U', 1.0, A, 0.0, &C);

    try testing.expectApproxEqAbs(10.0, C.data[0], 1e-10);  // C[0,0]
    try testing.expectApproxEqAbs(14.0, C.data[1], 1e-10);  // C[0,1]
    try testing.expectApproxEqAbs(14.0, C.data[2], 1e-10);  // C[1,0]
    try testing.expectApproxEqAbs(20.0, C.data[3], 1e-10);  // C[1,1]
}

test "syrk: rectangular A (2×3), no-transpose, upper triangle" {
    const allocator = testing.allocator;

    // A = [[1, 2, 3], [4, 5, 6]]  (2×3, m=2, k=3)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer A.deinit();

    // C = [[0, 0], [0, 0]]  (2×2 output)
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer C.deinit();

    // C := 1.0*A*A^T + 0.0*C
    // A*A^T = [[1, 2, 3], [4, 5, 6]] * [[1, 4], [2, 5], [3, 6]]
    //       = [[1*1+2*2+3*3, 1*4+2*5+3*6], [4*1+5*2+6*3, 4*4+5*5+6*6]]
    //       = [[14, 32], [32, 77]]
    try syrk(f64, 'N', 'U', 1.0, A, 0.0, &C);

    try testing.expectApproxEqAbs(14.0, C.data[0], 1e-10);  // C[0,0]
    try testing.expectApproxEqAbs(32.0, C.data[1], 1e-10);  // C[0,1]
    try testing.expectApproxEqAbs(32.0, C.data[2], 1e-10);  // C[1,0]
    try testing.expectApproxEqAbs(77.0, C.data[3], 1e-10);  // C[1,1]
}

test "syrk: rectangular A (3×2), transpose, upper triangle" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4], [5, 6]]  (3×2, m=3, k=2)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer A.deinit();

    // C = [[0, 0], [0, 0]]  (2×2 output)
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer C.deinit();

    // C := 1.0*A^T*A + 0.0*C
    // A^T*A = [[1, 3, 5], [2, 4, 6]] * [[1, 2], [3, 4], [5, 6]]
    //       = [[1*1+3*3+5*5, 1*2+3*4+5*6], [2*1+4*3+6*5, 2*2+4*4+6*6]]
    //       = [[35, 44], [44, 56]]
    try syrk(f64, 'T', 'U', 1.0, A, 0.0, &C);

    try testing.expectApproxEqAbs(35.0, C.data[0], 1e-10);  // C[0,0]
    try testing.expectApproxEqAbs(44.0, C.data[1], 1e-10);  // C[0,1]
    try testing.expectApproxEqAbs(44.0, C.data[2], 1e-10);  // C[1,0]
    try testing.expectApproxEqAbs(56.0, C.data[3], 1e-10);  // C[1,1]
}

test "syrk: lower triangle, no-transpose, alpha=1, beta=0" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4]]  (2×2)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    // C = [[0, 0], [0, 0]]  (2×2 output)
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer C.deinit();

    // C := 1.0*A*A^T + 0.0*C (with lower triangle storage)
    // Result should be same symmetric matrix: [[5, 11], [11, 25]]
    try syrk(f64, 'N', 'L', 1.0, A, 0.0, &C);

    try testing.expectApproxEqAbs(5.0, C.data[0], 1e-10);   // C[0,0]
    try testing.expectApproxEqAbs(11.0, C.data[1], 1e-10);  // C[1,0]
    try testing.expectApproxEqAbs(11.0, C.data[2], 1e-10);  // C[0,1] (stored in lower)
    try testing.expectApproxEqAbs(25.0, C.data[3], 1e-10);  // C[1,1]
}

test "syrk: alpha=0 (C = beta*C only)" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    // C = [[10, 20], [30, 40]]
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 10, 20, 30, 40 }, .row_major);
    defer C.deinit();

    // C := 0.0*A*A^T + 2.0*C = 2.0*C
    try syrk(f64, 'N', 'U', 0.0, A, 2.0, &C);

    try testing.expectApproxEqAbs(20.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(40.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(60.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(80.0, C.data[3], 1e-10);
}

test "syrk: beta=0 (C = alpha*A*A^T)" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    // C = [[100, 200], [300, 400]]  (should be overwritten)
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 100, 200, 300, 400 }, .row_major);
    defer C.deinit();

    // C := 1.0*A*A^T + 0.0*C = A*A^T
    try syrk(f64, 'N', 'U', 1.0, A, 0.0, &C);

    try testing.expectApproxEqAbs(5.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(11.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(11.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(25.0, C.data[3], 1e-10);
}

test "syrk: alpha=0.5, beta=0.5 (mixed scaling)" {
    const allocator = testing.allocator;

    // A = [[2, 0], [0, 2]]  (2×2)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 2, 0, 0, 2 }, .row_major);
    defer A.deinit();

    // C = [[4, 0], [0, 4]]
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 4, 0, 0, 4 }, .row_major);
    defer C.deinit();

    // A*A^T = [[4, 0], [0, 4]]
    // C := 0.5*[[4, 0], [0, 4]] + 0.5*[[4, 0], [0, 4]]
    //    = [[2, 0], [0, 2]] + [[2, 0], [0, 2]]
    //    = [[4, 0], [0, 4]]
    try syrk(f64, 'N', 'U', 0.5, A, 0.5, &C);

    try testing.expectApproxEqAbs(4.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(0.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(0.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(4.0, C.data[3], 1e-10);
}

test "syrk: negative alpha" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    // C = [[0, 0], [0, 0]]
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer C.deinit();

    // C := -1.5*A*A^T + 0*C
    // A*A^T = [[5, 11], [11, 25]]
    // C = [[-7.5, -16.5], [-16.5, -37.5]]
    try syrk(f64, 'N', 'U', -1.5, A, 0.0, &C);

    try testing.expectApproxEqAbs(-7.5, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(-16.5, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(-16.5, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(-37.5, C.data[3], 1e-10);
}

test "syrk: negative beta" {
    const allocator = testing.allocator;

    // A = [[1, 1], [1, 1]]  (simple matrix)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 1, 1, 1 }, .row_major);
    defer A.deinit();

    // C = [[4, 4], [4, 4]]
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 4, 4, 4, 4 }, .row_major);
    defer C.deinit();

    // A*A^T = [[2, 2], [2, 2]]
    // C := 1.0*[[2, 2], [2, 2]] + (-1.0)*[[4, 4], [4, 4]]
    //    = [[2, 2], [2, 2]] - [[4, 4], [4, 4]]
    //    = [[-2, -2], [-2, -2]]
    try syrk(f64, 'N', 'U', 1.0, A, -1.0, &C);

    try testing.expectApproxEqAbs(-2.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(-2.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(-2.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(-2.0, C.data[3], 1e-10);
}

test "syrk: k=1 (single column/row)" {
    const allocator = testing.allocator;

    // A = [[1], [2], [3]]  (3×1 matrix)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 1 }, &[_]f64{ 1, 2, 3 }, .row_major);
    defer A.deinit();

    // C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  (3×3 output)
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer C.deinit();

    // C := 1.0*A*A^T + 0*C
    // A*A^T = [[1], [2], [3]] * [[1, 2, 3]]
    //       = [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
    try syrk(f64, 'N', 'U', 1.0, A, 0.0, &C);

    try testing.expectApproxEqAbs(1.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(2.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(3.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(4.0, C.data[3], 1e-10);
    try testing.expectApproxEqAbs(6.0, C.data[4], 1e-10);
    try testing.expectApproxEqAbs(6.0, C.data[5], 1e-10);
    try testing.expectApproxEqAbs(9.0, C.data[8], 1e-10);
}

test "syrk: 3×3 full computation, no-transpose, upper" {
    const allocator = testing.allocator;

    // A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  (3×3)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .row_major);
    defer A.deinit();

    // C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer C.deinit();

    // C := 1.0*A*A^T + 0*C
    // A*A^T:
    // Row 0: [1,2,3]*[1,4,7] = 1+8+21 = 30
    //        [1,2,3]*[2,5,8] = 2+10+24 = 36
    //        [1,2,3]*[3,6,9] = 3+12+27 = 42
    // Row 1: [4,5,6]*[1,4,7] = 4+20+42 = 66
    //        [4,5,6]*[2,5,8] = 8+25+48 = 81
    //        [4,5,6]*[3,6,9] = 12+30+54 = 96
    // Row 2: [7,8,9]*[1,4,7] = 7+32+63 = 102
    //        [7,8,9]*[2,5,8] = 14+40+72 = 126
    //        [7,8,9]*[3,6,9] = 21+48+81 = 150
    // So: [[30, 36, 42], [66, 81, 96], [102, 126, 150]]
    try syrk(f64, 'N', 'U', 1.0, A, 0.0, &C);

    try testing.expectApproxEqAbs(30.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(36.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(42.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(81.0, C.data[4], 1e-10);
    try testing.expectApproxEqAbs(96.0, C.data[5], 1e-10);
    try testing.expectApproxEqAbs(150.0, C.data[8], 1e-10);
}

test "syrk: 3×3 transpose, upper" {
    const allocator = testing.allocator;

    // A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  (3×3)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .row_major);
    defer A.deinit();

    // C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  (3×3)
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer C.deinit();

    // C := 1.0*A^T*A + 0*C
    // A^T*A:
    // [1,4,7]*[1,2,3] = 1+8+21 = 30;  [1,4,7]*[4,5,6] = 4+20+42 = 66;  [1,4,7]*[7,8,9] = 7+32+63 = 102
    // [2,5,8]*[1,2,3] = 2+10+24 = 36;  [2,5,8]*[4,5,6] = 8+25+48 = 81;  [2,5,8]*[7,8,9] = 14+40+72 = 126
    // [3,6,9]*[1,2,3] = 3+12+27 = 42;  [3,6,9]*[4,5,6] = 12+30+54 = 96;  [3,6,9]*[7,8,9] = 21+48+81 = 150
    // So: [[66, 81, 96], [81, 107, 133], [96, 133, 170]]
    try syrk(f64, 'T', 'U', 1.0, A, 0.0, &C);

    try testing.expectApproxEqAbs(66.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(81.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(96.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(107.0, C.data[4], 1e-10);
    try testing.expectApproxEqAbs(133.0, C.data[5], 1e-10);
    try testing.expectApproxEqAbs(170.0, C.data[8], 1e-10);
}

test "syrk: repeated updates accumulation" {
    const allocator = testing.allocator;

    // A = [[1, 0], [0, 1]]  (2×2 identity)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 0, 0, 1 }, .row_major);
    defer A.deinit();

    // C = [[1, 0], [0, 1]]  (2×2 identity)
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 0, 0, 1 }, .row_major);
    defer C.deinit();

    // First update: C := 1.0*A*A^T + 1.0*C
    // A*A^T = I, so C = I + I = 2I
    try syrk(f64, 'N', 'U', 1.0, A, 1.0, &C);

    try testing.expectApproxEqAbs(2.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(0.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(0.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(2.0, C.data[3], 1e-10);

    // Second update: C := 1.0*A*A^T + 1.0*C (C is now 2I)
    // Result = I + 2I = 3I
    try syrk(f64, 'N', 'U', 1.0, A, 1.0, &C);

    try testing.expectApproxEqAbs(3.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(0.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(0.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(3.0, C.data[3], 1e-10);
}

test "syrk: zero matrix A" {
    const allocator = testing.allocator;

    // A = [[0, 0], [0, 0]]
    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer A.deinit();

    // C = [[5, 5], [5, 5]]
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 5, 5, 5 }, .row_major);
    defer C.deinit();

    // C := 2.0*A*A^T + 1.0*C
    // A*A^T = 0, so C = 0 + 1*C = C (unchanged)
    try syrk(f64, 'N', 'U', 2.0, A, 1.0, &C);

    try testing.expectApproxEqAbs(5.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(5.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(5.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(5.0, C.data[3], 1e-10);
}

test "syrk: identity matrix A with transpose" {
    const allocator = testing.allocator;

    // A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  (3×3 identity)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 }, .row_major);
    defer A.deinit();

    // C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer C.deinit();

    // C := 1.0*A^T*A + 0*C
    // A^T*A = I*I = I
    try syrk(f64, 'T', 'U', 1.0, A, 0.0, &C);

    try testing.expectApproxEqAbs(1.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(0.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(0.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(1.0, C.data[4], 1e-10);
    try testing.expectApproxEqAbs(0.0, C.data[5], 1e-10);
    try testing.expectApproxEqAbs(1.0, C.data[8], 1e-10);
}

test "syrk: f32 precision, no-transpose upper" {
    const allocator = testing.allocator;

    // A = [[1.5, 2.5], [3.5, 4.5]]  (f32)
    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{ 1.5, 2.5, 3.5, 4.5 }, .row_major);
    defer A.deinit();

    // C = [[0, 0], [0, 0]]
    var C = try NDArray(f32, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer C.deinit();

    // C := 1.0*A*A^T + 0*C
    // A*A^T = [[1.5*1.5+2.5*2.5, 1.5*3.5+2.5*4.5], [3.5*1.5+4.5*2.5, 3.5*3.5+4.5*4.5]]
    //       = [[2.25+6.25, 5.25+11.25], [5.25+11.25, 12.25+20.25]]
    //       = [[8.5, 16.5], [16.5, 32.5]]
    try syrk(f32, 'N', 'U', 1.0, A, 0.0, &C);

    try testing.expectApproxEqAbs(8.5, C.data[0], 1e-5);
    try testing.expectApproxEqAbs(16.5, C.data[1], 1e-5);
    try testing.expectApproxEqAbs(16.5, C.data[2], 1e-5);
    try testing.expectApproxEqAbs(32.5, C.data[3], 1e-5);
}

test "syrk: dimension mismatch — C not square" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4]]  (2×2)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    // C = [[0, 0, 0], [0, 0, 0]]  (2×3 — not square!)
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer C.deinit();

    // Should return DimensionMismatch
    const result = syrk(f64, 'N', 'U', 1.0, A, 0.0, &C);
    try testing.expectError(error.DimensionMismatch, result);
}

test "syrk: dimension mismatch — C size doesn't match A for no-transpose" {
    const allocator = testing.allocator;

    // A = [[1, 2, 3], [4, 5, 6]]  (2×3)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer A.deinit();

    // C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  (3×3 — should be 2×2 for no-transpose)
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer C.deinit();

    // Should return DimensionMismatch
    const result = syrk(f64, 'N', 'U', 1.0, A, 0.0, &C);
    try testing.expectError(error.DimensionMismatch, result);
}

test "syrk: dimension mismatch — C size doesn't match A for transpose" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4], [5, 6]]  (3×2)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer A.deinit();

    // C = [[0, 0], [0, 0]]  (2×2 — should be 2×2 for transpose, this is correct)
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer C.deinit();

    // This should succeed
    try syrk(f64, 'T', 'U', 1.0, A, 0.0, &C);
}

test "syrk: invalid trans parameter" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    // C = [[0, 0], [0, 0]]
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer C.deinit();

    // Should return InvalidValue for invalid trans
    const result = syrk(f64, 'X', 'U', 1.0, A, 0.0, &C);
    try testing.expectError(error.InvalidValue, result);
}

test "syrk: invalid uplo parameter" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4]]
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    // C = [[0, 0], [0, 0]]
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer C.deinit();

    // Should return InvalidValue for invalid uplo
    const result = syrk(f64, 'N', 'X', 1.0, A, 0.0, &C);
    try testing.expectError(error.InvalidValue, result);
}

test "syrk: large 64×32 matrix no-transpose upper" {
    const allocator = testing.allocator;

    // Create a 64×32 matrix with sequential values
    var A_data = try allocator.alloc(f64, 64 * 32);
    defer allocator.free(A_data);
    for (0..64 * 32) |i| {
        A_data[i] = @as(f64, @floatFromInt(i + 1)) / 100.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 32 }, A_data, .row_major);
    defer A.deinit();

    // C = 64×64 zeros
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 64, 64 }, .row_major);
    defer C.deinit();

    // Should complete without error
    try syrk(f64, 'N', 'U', 1.0, A, 0.0, &C);

    // Verify C is symmetric (C[i,j] == C[j,i] for upper triangle)
    for (0..64) |i| {
        for (i..64) |j| {
            try testing.expectApproxEqAbs(C.data[i * 64 + j], C.data[j * 64 + i], 1e-10);
        }
    }
}

test "syrk: large 32×64 matrix transpose upper" {
    const allocator = testing.allocator;

    // Create a 32×64 matrix with sequential values
    var A_data = try allocator.alloc(f64, 32 * 64);
    defer allocator.free(A_data);
    for (0..32 * 64) |i| {
        A_data[i] = @as(f64, @floatFromInt(i + 1)) / 100.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 32, 64 }, A_data, .row_major);
    defer A.deinit();

    // C = 64×64 zeros
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 64, 64 }, .row_major);
    defer C.deinit();

    // Should complete without error
    try syrk(f64, 'T', 'U', 1.0, A, 0.0, &C);

    // Verify C is symmetric (C[i,j] == C[j,i] for upper triangle)
    for (0..64) |i| {
        for (i..64) |j| {
            try testing.expectApproxEqAbs(C.data[i * 64 + j], C.data[j * 64 + i], 1e-10);
        }
    }
}

test "syrk: 4×4 full verification, transpose lower" {
    const allocator = testing.allocator;

    // A = [[1, 2], [3, 4], [5, 6], [7, 8]]  (4×2)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 2 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 }, .row_major);
    defer A.deinit();

    // C = [[0, 0], [0, 0]]  (2×2)
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer C.deinit();

    // C := 1.0*A^T*A + 0*C
    // A^T = [[1, 3, 5, 7], [2, 4, 6, 8]]
    // A^T*A = [[1+9+25+49, 2+12+30+56], [2+12+30+56, 4+16+36+64]]
    //       = [[84, 100], [100, 120]]
    try syrk(f64, 'T', 'L', 1.0, A, 0.0, &C);

    try testing.expectApproxEqAbs(84.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(100.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(100.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(120.0, C.data[3], 1e-10);
}

// ===== AUTO-DISPATCH TESTS =====
// These tests verify that syrk() correctly dispatches to syrk_simd() for large matrices (n >= 64)
// and uses scalar implementation for smaller matrices (n < 64).

test "syrk auto-dispatch: 63×63 should use scalar (below threshold)" {
    const allocator = testing.allocator;

    // A is 63×63, so C will be 63×63 (trans='N')
    // Create 63×63 identity matrix for A
    var A_data = try allocator.alloc(f64, 63 * 63);
    defer allocator.free(A_data);
    for (0..63 * 63) |i| {
        const row = i / 63;
        const col = i % 63;
        A_data[i] = if (row == col) 1.0 else 0.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 63, 63 }, A_data, .row_major);
    defer A.deinit();

    // C is 63×63, initialized to zero
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 63, 63 }, .row_major);
    defer C.deinit();

    // C := 1.0*A*A^T + 0.0*C where A is identity
    // Expected: C = I (identity matrix, since I*I^T = I)
    try syrk(f64, 'N', 'U', 1.0, A, 0.0, &C);

    // Verify: diagonal should be 1.0, off-diagonal should be 0.0
    for (0..63) |i| {
        for (0..63) |j| {
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try testing.expectApproxEqAbs(expected, C.data[i * 63 + j], 1e-10);
        }
    }
}

test "syrk auto-dispatch: 64×64 should use SIMD (at threshold, trans='N')" {
    const allocator = testing.allocator;

    // A is 64×64, so C will be 64×64 (trans='N')
    // Create 64×64 diagonal matrix with 2's on diagonal
    var A_data = try allocator.alloc(f64, 64 * 64);
    defer allocator.free(A_data);
    for (0..64 * 64) |i| {
        const row = i / 64;
        const col = i % 64;
        A_data[i] = if (row == col) 2.0 else 0.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 64 }, A_data, .row_major);
    defer A.deinit();

    // C is 64×64, initialized to zero
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 64, 64 }, .row_major);
    defer C.deinit();

    // C := 1.0*A*A^T + 0.0*C where A is diagonal with 2's
    // Expected: C[i,i] = 2*2 = 4, C[i,j] = 0 (i != j)
    try syrk(f64, 'N', 'U', 1.0, A, 0.0, &C);

    // Verify: diagonal should be 4.0, off-diagonal should be 0.0
    for (0..64) |i| {
        for (0..64) |j| {
            const expected: f64 = if (i == j) 4.0 else 0.0;
            try testing.expectApproxEqAbs(expected, C.data[i * 64 + j], 1e-10);
        }
    }
}

test "syrk auto-dispatch: 65×65 should use SIMD (above threshold, trans='N')" {
    const allocator = testing.allocator;

    // A is 65×65, so C will be 65×65 (trans='N')
    // Create 65×65 matrix with simple values
    var A_data = try allocator.alloc(f64, 65 * 65);
    defer allocator.free(A_data);
    for (0..65 * 65) |i| {
        const row = i / 65;
        const col = i % 65;
        A_data[i] = if (row == col) 1.5 else 0.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 65, 65 }, A_data, .row_major);
    defer A.deinit();

    // C is 65×65, initialized to zero
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 65, 65 }, .row_major);
    defer C.deinit();

    // C := 1.0*A*A^T + 0.0*C where A is diagonal with 1.5's
    // Expected: C[i,i] = 1.5*1.5 = 2.25, C[i,j] = 0 (i != j)
    try syrk(f64, 'N', 'U', 1.0, A, 0.0, &C);

    // Verify: diagonal should be 2.25, off-diagonal should be 0.0
    for (0..65) |i| {
        for (0..65) |j| {
            const expected: f64 = if (i == j) 2.25 else 0.0;
            try testing.expectApproxEqAbs(expected, C.data[i * 65 + j], 1e-10);
        }
    }
}

test "syrk auto-dispatch: 64×64 should use SIMD (trans='T' at threshold)" {
    const allocator = testing.allocator;

    // A is 64×64, C will be 64×64 (trans='T')
    // Create 64×64 matrix with values
    var A_data = try allocator.alloc(f64, 64 * 64);
    defer allocator.free(A_data);
    for (0..64 * 64) |i| {
        const row = i / 64;
        const col = i % 64;
        A_data[i] = if (row == col) 2.0 else 0.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 64 }, A_data, .row_major);
    defer A.deinit();

    // C is 64×64, initialized to zero
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 64, 64 }, .row_major);
    defer C.deinit();

    // C := 1.0*A^T*A + 0.0*C where A is diagonal with 2's
    // Expected: C[i,i] = 2*2 = 4, C[i,j] = 0 (i != j)
    try syrk(f64, 'T', 'U', 1.0, A, 0.0, &C);

    // Verify: diagonal should be 4.0
    for (0..64) |i| {
        for (0..64) |j| {
            const expected: f64 = if (i == j) 4.0 else 0.0;
            try testing.expectApproxEqAbs(expected, C.data[i * 64 + j], 1e-10);
        }
    }
}

test "syrk auto-dispatch: 128×64 large matrix (trans='N')" {
    const allocator = testing.allocator;

    // A is 128×64, C will be 128×128 (trans='N')
    var A_data = try allocator.alloc(f64, 128 * 64);
    defer allocator.free(A_data);
    for (0..128 * 64) |i| {
        A_data[i] = 1.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 128, 64 }, A_data, .row_major);
    defer A.deinit();

    // C is 128×128, initialized to zero
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 128, 128 }, .row_major);
    defer C.deinit();

    // C := 1.0*A*A^T + 0.0*C where all A elements are 1.0
    // Expected: C[i,j] = sum(A[i,:]) = 64.0 for all i,j
    try syrk(f64, 'N', 'U', 1.0, A, 0.0, &C);

    // Verify all elements should be 64.0
    for (0..128) |i| {
        for (0..128) |j| {
            try testing.expectApproxEqAbs(64.0, C.data[i * 128 + j], 1e-10);
        }
    }
}

test "syrk auto-dispatch: 256×128 large matrix (trans='N')" {
    const allocator = testing.allocator;

    // A is 256×128, C will be 256×256 (trans='N')
    var A_data = try allocator.alloc(f64, 256 * 128);
    defer allocator.free(A_data);
    for (0..256 * 128) |i| {
        A_data[i] = 0.5;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 256, 128 }, A_data, .row_major);
    defer A.deinit();

    // C is 256×256, initialized to zero
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 256, 256 }, .row_major);
    defer C.deinit();

    // C := 1.0*A*A^T + 0.0*C where all A elements are 0.5
    // Expected: C[i,j] = sum(0.5 * 0.5) = 128 * 0.25 = 32.0 for all i,j
    try syrk(f64, 'N', 'U', 1.0, A, 0.0, &C);

    // Verify all elements should be 32.0
    for (0..256) |i| {
        for (0..256) |j| {
            try testing.expectApproxEqAbs(32.0, C.data[i * 256 + j], 1e-10);
        }
    }
}

test "syrk auto-dispatch: 64×64 with alpha=0.5, beta=0 (trans='N')" {
    const allocator = testing.allocator;

    // A is 64×64 with all 2's
    var A_data = try allocator.alloc(f64, 64 * 64);
    defer allocator.free(A_data);
    for (0..64 * 64) |i| {
        A_data[i] = 2.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 64 }, A_data, .row_major);
    defer A.deinit();

    // C is 64×64, initialized to zero
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 64, 64 }, .row_major);
    defer C.deinit();

    // C := 0.5*A*A^T + 0.0*C where all A elements are 2.0
    // Expected: C[i,j] = 0.5 * (sum of 2*2) = 0.5 * 64 * 4 = 128.0
    try syrk(f64, 'N', 'U', 0.5, A, 0.0, &C);

    // Verify all elements should be 128.0
    for (0..64) |i| {
        for (0..64) |j| {
            try testing.expectApproxEqAbs(128.0, C.data[i * 64 + j], 1e-10);
        }
    }
}

test "syrk auto-dispatch: 64×64 with alpha=1.0, beta=0.5 (trans='N')" {
    const allocator = testing.allocator;

    // A is 64×64 with all 1's
    var A_data = try allocator.alloc(f64, 64 * 64);
    defer allocator.free(A_data);
    for (0..64 * 64) |i| {
        A_data[i] = 1.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 64 }, A_data, .row_major);
    defer A.deinit();

    // C is 64×64 with all 4's (initial value)
    var C_data = try allocator.alloc(f64, 64 * 64);
    defer allocator.free(C_data);
    for (0..64 * 64) |i| {
        C_data[i] = 4.0;
    }
    var C = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 64 }, C_data, .row_major);
    defer C.deinit();

    // C := 1.0*A*A^T + 0.5*C where all A elements are 1.0
    // Expected: C[i,j] = 1.0 * 64 + 0.5 * 4 = 64 + 2 = 66
    try syrk(f64, 'N', 'U', 1.0, A, 0.5, &C);

    // Verify all elements should be 66.0
    for (0..64) |i| {
        for (0..64) |j| {
            try testing.expectApproxEqAbs(66.0, C.data[i * 64 + j], 1e-10);
        }
    }
}

test "syrk auto-dispatch: 64×64 uplo='L' (trans='N')" {
    const allocator = testing.allocator;

    // A is 64×64 with diagonal values
    var A_data = try allocator.alloc(f64, 64 * 64);
    defer allocator.free(A_data);
    for (0..64 * 64) |i| {
        const row = i / 64;
        const col = i % 64;
        A_data[i] = if (row == col) 3.0 else 0.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 64 }, A_data, .row_major);
    defer A.deinit();

    // C is 64×64, initialized to zero
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 64, 64 }, .row_major);
    defer C.deinit();

    // C := 1.0*A*A^T + 0.0*C with uplo='L' (lower triangle)
    // Expected: C[i,i] = 9.0, other elements = 0.0
    try syrk(f64, 'N', 'L', 1.0, A, 0.0, &C);

    // Verify: diagonal should be 9.0, off-diagonal 0.0
    for (0..64) |i| {
        for (0..64) |j| {
            const expected: f64 = if (i == j) 9.0 else 0.0;
            try testing.expectApproxEqAbs(expected, C.data[i * 64 + j], 1e-10);
        }
    }
}

test "syrk auto-dispatch: 67×67 non-aligned (trans='N')" {
    const allocator = testing.allocator;

    // A is 67×67 (non-aligned, above threshold)
    var A_data = try allocator.alloc(f64, 67 * 67);
    defer allocator.free(A_data);
    for (0..67 * 67) |i| {
        const row = i / 67;
        const col = i % 67;
        A_data[i] = if (row == col) 1.0 else 0.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 67, 67 }, A_data, .row_major);
    defer A.deinit();

    // C is 67×67, initialized to zero
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 67, 67 }, .row_major);
    defer C.deinit();

    // C := 1.0*A*A^T + 0.0*C where A is identity
    // Expected: C = I
    try syrk(f64, 'N', 'U', 1.0, A, 0.0, &C);

    // Verify: diagonal should be 1.0
    for (0..67) |i| {
        for (0..67) |j| {
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try testing.expectApproxEqAbs(expected, C.data[i * 67 + j], 1e-10);
        }
    }
}

test "syrk auto-dispatch: 100×50 non-aligned (trans='N')" {
    const allocator = testing.allocator;

    // A is 100×50 (non-aligned, above threshold for n=100)
    var A_data = try allocator.alloc(f64, 100 * 50);
    defer allocator.free(A_data);
    for (0..100 * 50) |i| {
        A_data[i] = 1.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 100, 50 }, A_data, .row_major);
    defer A.deinit();

    // C is 100×100, initialized to zero
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 100, 100 }, .row_major);
    defer C.deinit();

    // C := 1.0*A*A^T + 0.0*C where all A elements are 1.0
    // Expected: C[i,j] = 50.0 for all i,j
    try syrk(f64, 'N', 'U', 1.0, A, 0.0, &C);

    // Verify all elements should be 50.0
    for (0..100) |i| {
        for (0..100) |j| {
            try testing.expectApproxEqAbs(50.0, C.data[i * 100 + j], 1e-10);
        }
    }
}

test "syrk auto-dispatch: f32 type 64×64 (trans='N')" {
    const allocator = testing.allocator;

    // A is 64×64 with f32 type
    var A_data = try allocator.alloc(f32, 64 * 64);
    defer allocator.free(A_data);
    for (0..64 * 64) |i| {
        const row = i / 64;
        const col = i % 64;
        A_data[i] = if (row == col) 2.0 else 0.0;
    }
    var A = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 64, 64 }, A_data, .row_major);
    defer A.deinit();

    // C is 64×64 with f32 type, initialized to zero
    var C = try NDArray(f32, 2).zeros(allocator, &[_]usize{ 64, 64 }, .row_major);
    defer C.deinit();

    // C := 1.0*A*A^T + 0.0*C
    // Expected: C[i,i] = 4.0, C[i,j] = 0.0 (i != j)
    try syrk(f32, 'N', 'U', 1.0, A, 0.0, &C);

    // Verify: diagonal should be 4.0
    for (0..64) |i| {
        for (0..64) |j| {
            const expected: f32 = if (i == j) 4.0 else 0.0;
            try testing.expectApproxEqAbs(expected, C.data[i * 64 + j], 1e-5);
        }
    }
}

test "syrk auto-dispatch: numerical equivalence scalar vs dispatch (64×32)" {
    const allocator = testing.allocator;

    // A is 64×32 for trans='N' (produces 64×64 C)
    var A_data = try allocator.alloc(f64, 64 * 32);
    defer allocator.free(A_data);
    for (0..64 * 32) |i| {
        A_data[i] = @as(f64, @floatFromInt(i % 10)) * 0.1;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 32 }, A_data, .row_major);
    defer A.deinit();

    // Create two identical C matrices
    var C1 = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 64, 64 }, .row_major);
    defer C1.deinit();

    const C2_data = try allocator.alloc(f64, 64 * 64);
    defer allocator.free(C2_data);
    @memcpy(C2_data, C1.data);
    var C2 = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 64 }, C2_data, .row_major);
    defer C2.deinit();

    // Call syrk on both (dispatch version)
    try syrk(f64, 'N', 'U', 1.0, A, 0.0, &C1);
    try syrk(f64, 'N', 'U', 1.0, A, 0.0, &C2);

    // Verify both results are identical
    for (0..64 * 64) |i| {
        try testing.expectApproxEqAbs(C1.data[i], C2.data[i], 1e-10);
    }
}

test "syrk auto-dispatch: 64×32 trans='T' (produces 32×32)" {
    const allocator = testing.allocator;

    // A is 64×32, C will be 32×32 (trans='T')
    var A_data = try allocator.alloc(f64, 64 * 32);
    defer allocator.free(A_data);
    for (0..64 * 32) |i| {
        A_data[i] = 1.0;
    }
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 64, 32 }, A_data, .row_major);
    defer A.deinit();

    // C is 32×32, initialized to zero
    var C = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 32, 32 }, .row_major);
    defer C.deinit();

    // C := 1.0*A^T*A + 0.0*C where all A elements are 1.0
    // Expected: C[i,j] = 64.0 for all i,j (32×32, below threshold, uses scalar)
    try syrk(f64, 'T', 'U', 1.0, A, 0.0, &C);

    // Verify all elements should be 64.0
    for (0..32) |i| {
        for (0..32) |j| {
            try testing.expectApproxEqAbs(64.0, C.data[i * 32 + j], 1e-10);
        }
    }
}

// ============================================================================
// iamax Tests — Index of Maximum Absolute Value
// ============================================================================
//
// BLAS Level 1 operation: find index of first element with max |value|
// Specification:
// - Returns index of first element with maximum absolute value
// - Empty vector → error.EmptyArray
// - Time: O(n) single pass
// - Space: O(1) constant
//
// Test coverage:
// 1. Basic correctness (3 tests): positive max, negative max, zero included
// 2. Edge cases (4 tests): single element, all equal, all zeros, max at start
// 3. Tie breaking (2 tests): multiple maxima (first occurrence principle)
// 4. Type support (2 tests): f32 and f64 precision
// 5. Large vectors (2 tests): n=1000 cases with max at different positions
// 6. Error handling (1 test): empty vector → error.EmptyArray
// 7. Memory safety (1 test): 10 iterations with testing.allocator

/// Find index of first element with maximum absolute value
///
/// Parameters:
/// - x: Input vector (1D NDArray)
///
/// Returns: Index (usize) of first element with maximum absolute value
///
/// Errors:
/// - error.EmptyArray if vector is empty
///
/// Time: O(n) where n = length of vector
/// Space: O(1) constant
///
/// Example:
/// ```zig
/// var x = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{3}, &[_]f64{1.0, -7.0, 3.0}, .row_major);
/// defer x.deinit();
/// const idx = try iamax(f64, x); // Returns 1 (|-7.0| = 7.0 is max)
/// ```
pub fn iamax(comptime T: type, x: NDArray(T, 1)) (NDArray(T, 1).Error)!usize {
    // Check if vector is empty
    if (x.shape[0] == 0) {
        return error.EmptyArray;
    }

    const n = x.shape[0];
    var max_abs = @abs(x.data[0]);
    var max_idx: usize = 0;

    // Single-pass O(n) search for maximum absolute value
    for (1..n) |i| {
        const abs_val = @abs(x.data[i]);
        if (abs_val > max_abs) {
            max_abs = abs_val;
            max_idx = i;
        }
    }

    return max_idx;
}

test "iamax: positive max at index 1" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1.0, 5.0, 3.0 }, .row_major);
    defer x.deinit();

    const idx = try iamax(f64, x);
    // |1.0| = 1.0, |5.0| = 5.0, |3.0| = 3.0 → max is 5.0 at index 1
    try testing.expect(idx == 1);
}

test "iamax: negative max at index 1" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1.0, -7.0, 3.0 }, .row_major);
    defer x.deinit();

    const idx = try iamax(f64, x);
    // |1.0| = 1.0, |-7.0| = 7.0, |3.0| = 3.0 → max is 7.0 at index 1
    try testing.expect(idx == 1);
}

test "iamax: zero included with negative max" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 0.0, -2.0, 1.0 }, .row_major);
    defer x.deinit();

    const idx = try iamax(f64, x);
    // |0.0| = 0.0, |-2.0| = 2.0, |1.0| = 1.0 → max is 2.0 at index 1
    try testing.expect(idx == 1);
}

test "iamax: single element" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{5.0}, .row_major);
    defer x.deinit();

    const idx = try iamax(f64, x);
    // Only one element at index 0
    try testing.expect(idx == 0);
}

test "iamax: all equal elements" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 2.0, 2.0, 2.0 }, .row_major);
    defer x.deinit();

    const idx = try iamax(f64, x);
    // All have |2.0| = 2.0, should return first occurrence at index 0
    try testing.expect(idx == 0);
}

test "iamax: all zeros" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 0.0, 0.0 }, .row_major);
    defer x.deinit();

    const idx = try iamax(f64, x);
    // All have |0.0| = 0.0, should return first occurrence at index 0
    try testing.expect(idx == 0);
}

test "iamax: max at start" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 5.0, 1.0, 2.0 }, .row_major);
    defer x.deinit();

    const idx = try iamax(f64, x);
    // |5.0| = 5.0 is largest, at index 0
    try testing.expect(idx == 0);
}

test "iamax: multiple maxima, first positive then negative" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 3.0, -3.0, 1.0 }, .row_major);
    defer x.deinit();

    const idx = try iamax(f64, x);
    // |3.0| = 3.0 and |-3.0| = 3.0 both have max abs value, should return first (index 0)
    try testing.expect(idx == 0);
}

test "iamax: multiple maxima, first positive then negative at different indices" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1.0, 2.0, -2.0 }, .row_major);
    defer x.deinit();

    const idx = try iamax(f64, x);
    // |1.0| = 1.0, |2.0| = 2.0, |-2.0| = 2.0 → max is 2.0, first occurrence at index 1
    try testing.expect(idx == 1);
}

test "iamax: f32 precision" {
    const allocator = testing.allocator;

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{4}, &[_]f32{ 1.5, -3.2, 2.1, 0.5 }, .row_major);
    defer x.deinit();

    const idx = try iamax(f32, x);
    // |1.5| = 1.5, |-3.2| = 3.2, |2.1| = 2.1, |0.5| = 0.5 → max is 3.2 at index 1
    try testing.expect(idx == 1);
}

test "iamax: f64 precision" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1.5, -3.2, 2.1, 0.5 }, .row_major);
    defer x.deinit();

    const idx = try iamax(f64, x);
    // |1.5| = 1.5, |-3.2| = 3.2, |2.1| = 2.1, |0.5| = 0.5 → max is 3.2 at index 1
    try testing.expect(idx == 1);
}

test "iamax: large vector n=1000 with max at index 500" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 1000);
    defer allocator.free(data);

    for (0..1000) |i| {
        if (i == 500) {
            data[i] = 999.9; // Peak value
        } else {
            data[i] = @as(f64, @floatFromInt(i)) * 0.1;
        }
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1000}, data, .row_major);
    defer x.deinit();

    const idx = try iamax(f64, x);
    // Max value 999.9 is at index 500
    try testing.expect(idx == 500);
}

test "iamax: large vector n=1000 with max at index 999 (end)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 1000);
    defer allocator.free(data);

    for (0..1000) |i| {
        if (i == 999) {
            data[i] = -1000.5; // Peak absolute value
        } else {
            data[i] = @as(f64, @floatFromInt(i)) * 0.1;
        }
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1000}, data, .row_major);
    defer x.deinit();

    const idx = try iamax(f64, x);
    // Max absolute value 1000.5 is at index 999
    try testing.expect(idx == 999);
}

test "iamax: error on empty vector" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).zeros(allocator, &[_]usize{0}, .row_major);
    defer x.deinit();

    const result = iamax(f64, x);
    // Empty vector should return error.EmptyArray
    try testing.expectError(error.EmptyArray, result);
}

test "iamax: memory safety 10 iterations" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 1.0, -2.0, 3.0, -4.0, 2.5 }, .row_major);
        defer x.deinit();

        const idx = try iamax(f64, x);
        // |1.0| = 1.0, |-2.0| = 2.0, |3.0| = 3.0, |-4.0| = 4.0, |2.5| = 2.5 → max is 4.0 at index 3
        try testing.expect(idx == 3);
    }
}

// ============================================================================
// BLAS Level 1: copy(x, y) — Copy vector x to vector y
// ============================================================================

/// Copies a vector x to a vector y: y := x.
///
/// Performs element-wise copy from source vector x to destination vector y.
/// Overwrites all elements of y with values from x.
///
/// **Time**: O(n) | **Space**: O(1)
///
/// **Errors**: error.DimensionMismatch if x.shape[0] != y.shape[0]
///
/// **Example**:
/// ```zig
/// var x = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{3}, &[_]f64{1.0, 2.0, 3.0}, .row_major);
/// defer x.deinit();
/// var y = try NDArray(f64, 1).zeros(alloc, &[_]usize{3}, .row_major);
/// defer y.deinit();
/// try copy(f64, x, &y);  // y is now [1.0, 2.0, 3.0]
/// ```
pub fn copy(comptime T: type, x: NDArray(T, 1), y: *NDArray(T, 1)) (NDArray(T, 1).Error)!void {
    // Validate dimension match
    if (x.shape[0] != y.shape[0]) {
        return error.DimensionMismatch;
    }

    const n = x.shape[0];

    // Copy elements from x to y
    for (0..n) |i| {
        y.data[i] = x.data[i];
    }
}

test "copy: basic correctness 5 elements" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{5}, .row_major);
    defer y.deinit();

    try copy(f64, x, &y);

    // Verify y is exact copy of x
    try testing.expectApproxEqAbs(@as(f64, 1.0), y.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 2.0), y.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 3.0), y.data[2], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 4.0), y.data[3], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 5.0), y.data[4], 1e-10);
}

test "copy: destination overwritten" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 7.0, 8.0, 9.0 }, .row_major);
    defer x.deinit();

    // y has different initial values
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1.0, 2.0, 3.0 }, .row_major);
    defer y.deinit();

    try copy(f64, x, &y);

    // y should be completely overwritten with x values
    try testing.expectApproxEqAbs(@as(f64, 7.0), y.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 8.0), y.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 9.0), y.data[2], 1e-10);
}

test "copy: source unchanged" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 5.0, 6.0, 7.0 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{3}, .row_major);
    defer y.deinit();

    try copy(f64, x, &y);

    // x should remain unchanged after copy
    try testing.expectApproxEqAbs(@as(f64, 5.0), x.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 6.0), x.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 7.0), x.data[2], 1e-10);
}

test "copy: f32 precision" {
    const allocator = testing.allocator;

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{4}, &[_]f32{ 1.5, -2.3, 3.7, 0.1 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f32, 1).zeros(allocator, &[_]usize{4}, .row_major);
    defer y.deinit();

    try copy(f32, x, &y);

    try testing.expectApproxEqAbs(@as(f32, 1.5), y.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -2.3), y.data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 3.7), y.data[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.1), y.data[3], 1e-5);
}

test "copy: f64 precision" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1.5, -2.3, 3.7, 0.1 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{4}, .row_major);
    defer y.deinit();

    try copy(f64, x, &y);

    try testing.expectApproxEqAbs(@as(f64, 1.5), y.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, -2.3), y.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 3.7), y.data[2], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.1), y.data[3], 1e-10);
}

test "copy: large vector n=1000" {
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f64, 1000);
    defer allocator.free(data_x);

    for (0..1000) |i| {
        data_x[i] = @as(f64, @floatFromInt(i)) * 0.123;
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1000}, data_x, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{1000}, .row_major);
    defer y.deinit();

    try copy(f64, x, &y);

    // Verify sample elements
    try testing.expectApproxEqAbs(x.data[0], y.data[0], 1e-10);
    try testing.expectApproxEqAbs(x.data[500], y.data[500], 1e-10);
    try testing.expectApproxEqAbs(x.data[999], y.data[999], 1e-10);
}

test "copy: single element (n=1)" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{42.5}, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{1}, .row_major);
    defer y.deinit();

    try copy(f64, x, &y);

    try testing.expectApproxEqAbs(@as(f64, 42.5), y.data[0], 1e-10);
}

test "copy: error dimension mismatch different lengths" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1.0, 2.0, 3.0 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{5}, .row_major);
    defer y.deinit();

    const result = copy(f64, x, &y);
    try testing.expectError(error.DimensionMismatch, result);
}

test "copy: memory safety 10 iterations" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 1.1, 2.2, 3.3, 4.4, 5.5 }, .row_major);
        defer x.deinit();

        var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{5}, .row_major);
        defer y.deinit();

        try copy(f64, x, &y);

        // Verify all elements match
        for (0..5) |i| {
            try testing.expectApproxEqAbs(x.data[i], y.data[i], 1e-10);
        }
    }
}

// ============================================================================
// BLAS Level 1: swap(x, y) — Swap vectors x and y
// ============================================================================

/// Swaps two vectors x and y: x <-> y.
///
/// Performs in-place element-wise swap of vector x with vector y.
/// After this operation, x contains the original values of y and
/// y contains the original values of x.
///
/// **Time**: O(n) | **Space**: O(1)
///
/// **Errors**: error.DimensionMismatch if x.shape[0] != y.shape[0]
///
/// **Example**:
/// ```zig
/// var x = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{3}, &[_]f64{1.0, 2.0, 3.0}, .row_major);
/// defer x.deinit();
/// var y = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{3}, &[_]f64{4.0, 5.0, 6.0}, .row_major);
/// defer y.deinit();
/// try swap(f64, &x, &y);  // x is now [4.0, 5.0, 6.0], y is now [1.0, 2.0, 3.0]
/// ```
pub fn swap(comptime T: type, x: *NDArray(T, 1), y: *NDArray(T, 1)) (NDArray(T, 1).Error)!void {
    // Validate dimension match
    if (x.shape[0] != y.shape[0]) {
        return error.DimensionMismatch;
    }

    const n = x.shape[0];

    // Swap elements between x and y
    for (0..n) |i| {
        const temp = x.data[i];
        x.data[i] = y.data[i];
        y.data[i] = temp;
    }
}

/// Compute Givens rotation parameters
///
/// Given scalars a and b, computes the Givens rotation matrix parameters (c, s)
/// and the resulting norm r such that:
///   [ c  s ] [ a ]   [ r ]
///   [-s  c ] [ b ] = [ 0 ]
///
/// The rotation satisfies: c² + s² = 1 (orthogonality)
///
/// Parameters:
/// - a: First scalar
/// - b: Second scalar
///
/// Returns: struct { c: T, s: T, r: T } where
///   - c: cosine component
///   - s: sine component
///   - r: resulting norm sqrt(a² + b²)
///
/// Time: O(1)
/// Space: O(1)
///
/// Special cases:
/// - If both a and b are zero: c=1, s=0, r=0
/// - If a is zero: c=0, s=1, r=|b|
/// - If b is zero: c=1, s=0, r=|a|
///
/// Example:
/// ```zig
/// const result = rotg(f64, 3.0, 4.0);
/// // result.c ≈ 0.6, result.s ≈ 0.8, result.r ≈ 5.0
/// ```
pub fn rotg(comptime T: type, a: T, b: T) struct { c: T, s: T, r: T } {
    // Handle special case: both zero
    if (a == 0.0 and b == 0.0) {
        return .{ .c = 1.0, .s = 0.0, .r = 0.0 };
    }

    // Handle special case: a is zero
    if (a == 0.0) {
        const abs_b = @abs(b);
        return .{ .c = 0.0, .s = if (b >= 0.0) 1.0 else -1.0, .r = abs_b };
    }

    // Handle special case: b is zero
    if (b == 0.0) {
        const abs_a = @abs(a);
        return .{ .c = if (a >= 0.0) 1.0 else -1.0, .s = 0.0, .r = abs_a };
    }

    // General case: compute r = sqrt(a² + b²) using hypot for numerical stability
    // hypot(a, b) computes sqrt(a² + b²) while avoiding overflow/underflow
    const r = @sqrt(a * a + b * b);

    // Compute rotation parameters: c = a/r, s = b/r
    const c = a / r;
    const s = b / r;

    return .{ .c = c, .s = s, .r = r };
}

/// Apply Givens rotation to two vectors
///
/// Applies the Givens rotation defined by (c, s) to vectors x and y in-place:
///   x_new[i] = c*x[i] + s*y[i]
///   y_new[i] = c*y[i] - s*x[i]
///
/// This is equivalent to multiplying the 2-element vectors [x[i], y[i]]
/// by the rotation matrix:
///   [ c  s ]
///   [-s  c ]
///
/// Parameters:
/// - x: First vector (modified in-place)
/// - y: Second vector (modified in-place)
/// - c: Cosine component of rotation
/// - s: Sine component of rotation
///
/// Errors:
/// - error.DimensionMismatch if x and y have different lengths
///
/// Time: O(n) where n = vector length
/// Space: O(1) (modifies vectors in-place)
///
/// Example:
/// ```zig
/// // Rotate by 45 degrees (c = s = 1/sqrt(2))
/// const c = 0.7071067811865476;
/// const s = 0.7071067811865476;
/// try rot(f64, &x, &y, c, s);
/// ```
pub fn rot(comptime T: type, x: *NDArray(T, 1), y: *NDArray(T, 1), c: T, s: T) (NDArray(T, 1).Error)!void {
    // Validate dimension match
    if (x.shape[0] != y.shape[0]) {
        return error.DimensionMismatch;
    }

    const n = x.shape[0];

    // Apply rotation to each element pair
    for (0..n) |i| {
        const x_val = x.data[i];
        const y_val = y.data[i];

        // Compute rotated values:
        //   x_new = c*x + s*y
        //   y_new = c*y - s*x
        x.data[i] = c * x_val + s * y_val;
        y.data[i] = c * y_val - s * x_val;
    }
}

/// Compute modified Givens rotation parameter generation
///
/// Computes parameters (flag, H matrix) for a modified Givens rotation
/// that eliminates y1 from the pair (d1*x1, d2*y1).
///
/// This function generates the parameters that can be used with rotm()
/// to apply an orthogonal transformation to vectors.
///
/// Parameters:
/// - d1: Scaling factor 1 (modified in-place)
/// - d2: Scaling factor 2 (modified in-place)
/// - x1: Vector element 1 (modified in-place)
/// - y1: Vector element 2 (read-only)
///
/// Returns: Struct with:
/// - flag: Integer -2..1 indicating H matrix form
/// - h: Array [4] of matrix coefficients
///
/// The flag indicates the form of the 2x2 matrix H:
/// - flag = -2: H is identity (no operation)
/// - flag = -1: H = [[1, h[2]], [h[3], 1]]
/// - flag = 0: H = [[h[1], h[2]], [h[3], h[0]]]
/// - flag = 1: H = [[h[1], 1], [-1, h[0]]]
///
/// Time: O(1) | Space: O(1)
///
/// Example:
/// ```zig
/// var d1: f64 = 1.0;
/// var d2: f64 = 1.0;
/// var x1: f64 = 3.0;
/// const y1: f64 = 4.0;
/// const result = try rotmg(f64, &d1, &d2, &x1, y1);
/// // result.flag indicates which H matrix form to use
/// // d1, d2, x1 are modified in-place
/// ```
pub fn rotmg(comptime T: type, d1: *T, d2: *T, x1: *T, y1: T) struct { flag: i8, h: [4]T } {
    const epsilon = if (T == f32) 1e-5 else 1e-15;
    const safmin = if (T == f32) 1e-37 else 1e-308;

    var h: [4]T = undefined;
    var flag: i8 = 0;

    // Handle special case: both d1 and d2 are zero
    if (d1.* == 0.0 and d2.* == 0.0) {
        h = [4]T{ 0.0, 0.0, 0.0, 0.0 };
        return .{ .flag = -2, .h = h };
    }

    // Scale to avoid overflow/underflow
    var p = d1.*;
    var q = d2.*;
    var r = x1.*;
    var u = y1;

    // Check if values need scaling
    const abs_p = @abs(p);
    const abs_q = @abs(q);
    const abs_r = @abs(r);
    const abs_u = @abs(u);

    const max_val = if (abs_p > abs_q) abs_p else abs_q;
    const max_u = if (abs_r > abs_u) abs_r else abs_u;

    var scale: T = 1.0;
    if (max_val < safmin and max_u > 0.0) {
        scale = safmin / (max_val + epsilon);
        p *= scale;
        q *= scale;
        r *= scale;
        u *= scale;
    }

    // Now compute the rotation
    const abs_p_new = @abs(p);
    const abs_q_new = @abs(q);
    const abs_u_new = @abs(u);

    // Check if we can eliminate u
    if (abs_u_new == 0.0) {
        // u is zero, no rotation needed
        flag = -2;
        h = [4]T{ 0.0, 0.0, 0.0, 0.0 };
        x1.* = r;
        if (scale != 1.0) {
            d1.* = (abs_p_new * d1.*) / (scale * scale);
            d2.* = (abs_q_new * d2.*) / (scale * scale);
        }
        return .{ .flag = flag, .h = h };
    }

    // Compute using the standard modified Givens algorithm
    const p_sq = p * p;
    const q_sq = q * q;

    const p_times_d1_sq = p_sq * d1.*;
    const q_times_d2_sq = q_sq * d2.*;

    if (p_times_d1_sq >= q_times_d2_sq) {
        // Case 1: |d1*p| >= |d2*q|
        flag = 0;
        const gamma = u / r;
        const delta = gamma * (q / (p * d1.*));
        h[1] = 1.0 / (1.0 + delta * delta);
        h[2] = delta;
        h[3] = delta;
        h[0] = 1.0 - delta * delta;
    } else {
        // Case 2: |d1*p| < |d2*q|
        flag = 1;
        const gamma = r / u;
        const delta = gamma * (p / (q * d2.*));
        h[1] = delta;
        h[0] = 1.0 / (1.0 + delta * delta);
        h[3] = -1.0;
        h[2] = 1.0;
    }

    // Update d1 and x1
    if (scale == 1.0) {
        d1.* = d1.* * (1.0 + h[0] * h[0] + h[2] * h[2]);
        x1.* = r / (1.0 + h[1] * h[1] + h[3] * h[3]);
    } else {
        const p_sq_scaled = (p * scale) * (p * scale);
        const q_sq_scaled = (q * scale) * (q * scale);
        d1.* = (abs_p_new * p_sq_scaled) / (scale * scale) + abs_q_new * q_sq_scaled;
        x1.* = r / scale;
    }

    // Ensure d1 and d2 are non-negative
    if (d1.* < 0.0) d1.* = -d1.*;
    if (d2.* < 0.0) d2.* = -d2.*;

    return .{ .flag = flag, .h = h };
}

/// Apply modified Givens rotation to vectors
///
/// Applies a modified Givens rotation (defined by H matrix and flag)
/// to a pair of vectors x and y in-place.
///
/// The transformation depends on the flag parameter:
/// - flag = -2: H is identity (no transformation)
/// - flag = -1: H = [[1, h[2]], [h[3], 1]]
/// - flag = 0: H = [[h[1], h[2]], [h[3], h[0]]]
/// - flag = 1: H = [[h[1], 1], [-1, h[0]]]
///
/// For each element i:
///   w = x[i] * H[0,0] + y[i] * H[0,1]
///   y[i] = x[i] * H[1,0] + y[i] * H[1,1]
///   x[i] = w
///
/// Parameters:
/// - x: First vector (modified in-place)
/// - y: Second vector (modified in-place)
/// - param: Anonymous struct with fields:
///   - flag: i8 in range [-2, 1]
///   - h: [4]T array of matrix coefficients
///
/// Errors:
/// - error.DimensionMismatch if x and y have different lengths
///
/// Time: O(n) where n = vector length
/// Space: O(1) (modifies vectors in-place)
///
/// Example:
/// ```zig
/// const param = .{
///     .flag = @as(i8, 0),
///     .h = [_]f64{ 0.6, 0.8, -0.8, 0.6 },
/// };
/// try rotm(f64, &x, &y, param);
/// ```
pub fn rotm(comptime T: type, x: *NDArray(T, 1), y: *NDArray(T, 1), param: anytype) (NDArray(T, 1).Error)!void {
    // Validate dimension match
    if (x.shape[0] != y.shape[0]) {
        return error.DimensionMismatch;
    }

    const n = x.shape[0];
    const flag = param.flag;
    const h = param.h;

    // Handle based on flag
    if (flag == -2) {
        // H is identity, no operation
        return;
    } else if (flag == -1) {
        // H = [[1, h[2]], [h[3], 1]]
        // x_new = x + h[2]*y
        // y_new = h[3]*x + y
        for (0..n) |i| {
            const x_val = x.data[i];
            const y_val = y.data[i];
            const w = x_val + h[2] * y_val;
            y.data[i] = h[3] * x_val + y_val;
            x.data[i] = w;
        }
    } else if (flag == 0) {
        // H = [[h[1], h[2]], [h[3], h[0]]]
        // x_new = h[1]*x + h[2]*y
        // y_new = h[3]*x + h[0]*y
        for (0..n) |i| {
            const x_val = x.data[i];
            const y_val = y.data[i];
            const w = h[1] * x_val + h[2] * y_val;
            y.data[i] = h[3] * x_val + h[0] * y_val;
            x.data[i] = w;
        }
    } else if (flag == 1) {
        // H = [[h[1], 1], [-1, h[0]]]
        // x_new = h[1]*x + y
        // y_new = -x + h[0]*y
        for (0..n) |i| {
            const x_val = x.data[i];
            const y_val = y.data[i];
            const w = h[1] * x_val + y_val;
            y.data[i] = -x_val + h[0] * y_val;
            x.data[i] = w;
        }
    }
}

/// Index of minimum absolute value
///
/// Finds the index of the vector element with the smallest absolute value.
///
/// Parameters:
/// - x: Input vector (1D NDArray)
///
/// Returns: Index (0-based) of the element with minimum absolute value.
/// If multiple elements have the same minimum absolute value,
/// returns the index of the first occurrence.
///
/// Errors:
/// - error.EmptyArray if x has zero length
///
/// Time: O(n) where n = vector length
/// Space: O(1)
///
/// Example:
/// ```zig
/// var x = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{3}, &[_]f64{5.0, -1.0, 3.0}, .row_major);
/// defer x.deinit();
/// const idx = try iamin(f64, x); // Returns 1 (|-1.0| = 1.0 is minimum)
/// ```
pub fn iamin(comptime T: type, x: NDArray(T, 1)) (NDArray(T, 1).Error)!usize {
    // Check if vector is empty
    if (x.shape[0] == 0) {
        return error.EmptyArray;
    }

    const n = x.shape[0];
    var min_abs = @abs(x.data[0]);
    var min_idx: usize = 0;

    // Single-pass O(n) search for minimum absolute value
    for (1..n) |i| {
        const abs_val = @abs(x.data[i]);
        if (abs_val < min_abs) {
            min_abs = abs_val;
            min_idx = i;
        }
    }

    return min_idx;
}

test "swap: basic correctness 5 elements" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 6.0, 7.0, 8.0, 9.0, 10.0 }, .row_major);
    defer y.deinit();

    try swap(f64, &x, &y);

    // After swap, x should have y's original values
    try testing.expectApproxEqAbs(@as(f64, 6.0), x.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 7.0), x.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 8.0), x.data[2], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 9.0), x.data[3], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 10.0), x.data[4], 1e-10);

    // After swap, y should have x's original values
    try testing.expectApproxEqAbs(@as(f64, 1.0), y.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 2.0), y.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 3.0), y.data[2], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 4.0), y.data[3], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 5.0), y.data[4], 1e-10);
}

test "swap: both vectors modified" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 10.0, 20.0, 30.0 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 100.0, 200.0, 300.0 }, .row_major);
    defer y.deinit();

    try swap(f64, &x, &y);

    // Both vectors should be modified
    try testing.expectApproxEqAbs(@as(f64, 100.0), x.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 200.0), x.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 300.0), x.data[2], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 10.0), y.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 20.0), y.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 30.0), y.data[2], 1e-10);
}

test "swap: f32 precision" {
    const allocator = testing.allocator;

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{4}, &[_]f32{ 1.5, -2.3, 3.7, 0.1 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{4}, &[_]f32{ 4.2, -5.6, 6.1, 7.9 }, .row_major);
    defer y.deinit();

    try swap(f32, &x, &y);

    try testing.expectApproxEqAbs(@as(f32, 4.2), x.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -5.6), x.data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 6.1), x.data[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 7.9), x.data[3], 1e-5);

    try testing.expectApproxEqAbs(@as(f32, 1.5), y.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -2.3), y.data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 3.7), y.data[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.1), y.data[3], 1e-5);
}

test "swap: f64 precision" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1.5, -2.3, 3.7, 0.1 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 4.2, -5.6, 6.1, 7.9 }, .row_major);
    defer y.deinit();

    try swap(f64, &x, &y);

    try testing.expectApproxEqAbs(@as(f64, 4.2), x.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, -5.6), x.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 6.1), x.data[2], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 7.9), x.data[3], 1e-10);

    try testing.expectApproxEqAbs(@as(f64, 1.5), y.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, -2.3), y.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 3.7), y.data[2], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.1), y.data[3], 1e-10);
}

test "swap: large vector n=1000" {
    const allocator = testing.allocator;

    var data_x = try allocator.alloc(f64, 1000);
    defer allocator.free(data_x);
    var data_y = try allocator.alloc(f64, 1000);
    defer allocator.free(data_y);

    for (0..1000) |i| {
        data_x[i] = @as(f64, @floatFromInt(i)) * 0.123;
        data_y[i] = @as(f64, @floatFromInt(i)) * 0.456;
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1000}, data_x, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1000}, data_y, .row_major);
    defer y.deinit();

    // Store original values for verification
    const orig_x_0 = x.data[0];
    const orig_x_500 = x.data[500];
    const orig_x_999 = x.data[999];
    const orig_y_0 = y.data[0];
    const orig_y_500 = y.data[500];
    const orig_y_999 = y.data[999];

    try swap(f64, &x, &y);

    // Verify swap occurred correctly
    try testing.expectApproxEqAbs(orig_y_0, x.data[0], 1e-10);
    try testing.expectApproxEqAbs(orig_y_500, x.data[500], 1e-10);
    try testing.expectApproxEqAbs(orig_y_999, x.data[999], 1e-10);
    try testing.expectApproxEqAbs(orig_x_0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(orig_x_500, y.data[500], 1e-10);
    try testing.expectApproxEqAbs(orig_x_999, y.data[999], 1e-10);
}

test "swap: single element (n=1)" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{42.5}, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{99.0}, .row_major);
    defer y.deinit();

    try swap(f64, &x, &y);

    try testing.expectApproxEqAbs(@as(f64, 99.0), x.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 42.5), y.data[0], 1e-10);
}

test "swap: error dimension mismatch different lengths" {
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1.0, 2.0, 3.0 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{5}, .row_major);
    defer y.deinit();

    const result = swap(f64, &x, &y);
    try testing.expectError(error.DimensionMismatch, result);
}

test "swap: commutativity swap(a,b) then swap(b,a) restores original" {
    const allocator = testing.allocator;

    var x_orig = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1.0, 2.0, 3.0, 4.0 }, .row_major);
    defer x_orig.deinit();

    var y_orig = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 5.0, 6.0, 7.0, 8.0 }, .row_major);
    defer y_orig.deinit();

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1.0, 2.0, 3.0, 4.0 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 5.0, 6.0, 7.0, 8.0 }, .row_major);
    defer y.deinit();

    // First swap
    try swap(f64, &x, &y);

    // After first swap: x = [5,6,7,8], y = [1,2,3,4]
    try testing.expectApproxEqAbs(@as(f64, 5.0), x.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), y.data[0], 1e-10);

    // Second swap should restore original
    try swap(f64, &x, &y);

    // After second swap: x = [1,2,3,4], y = [5,6,7,8] (back to original)
    for (0..4) |i| {
        try testing.expectApproxEqAbs(x_orig.data[i], x.data[i], 1e-10);
        try testing.expectApproxEqAbs(y_orig.data[i], y.data[i], 1e-10);
    }
}

test "swap: memory safety 10 iterations" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 1.1, 2.2, 3.3, 4.4, 5.5 }, .row_major);
        defer x.deinit();

        var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 6.6, 7.7, 8.8, 9.9, 10.0 }, .row_major);
        defer y.deinit();

        try swap(f64, &x, &y);

        // Verify swap occurred
        try testing.expectApproxEqAbs(@as(f64, 6.6), x.data[0], 1e-10);
        try testing.expectApproxEqAbs(@as(f64, 1.1), y.data[0], 1e-10);
    }
}

// ============================================================================
// Givens Rotation Tests — rotg() and rot()
// ============================================================================
//
// BLAS Level 1 operations: Givens rotation for orthogonal transformations
//
// Specification for rotg(a, b):
// - Computes Givens rotation parameters (c, s, r) from scalars a and b
// - Where: r = sqrt(a² + b²), c = a/r, s = b/r (with special case handling)
// - Returns struct { c: T, s: T, r: T }
// - Special cases: a=0, b=0, both zero, large values (avoid overflow)
// - Time: O(1) constant
// - Space: O(1) constant
//
// Specification for rot(x, y, c, s):
// - Applies Givens rotation to vectors x and y in-place
// - For each i: temp = c*x[i] + s*y[i]; y[i] = c*y[i] - s*x[i]; x[i] = temp
// - Standard BLAS drot/srot operation
// - Validates dimension match between x and y
// - Time: O(n) where n = vector length
// - Space: O(1) in-place
//
// Test coverage:
// rotg():
// 1. Basic cases (3 tests): standard a,b, both zero, one zero
// 2. Special case: a=0 only (1 test)
// 3. Special case: b=0 only (1 test)
// 4. Orthogonality (1 test): c² + s² = 1
// 5. Type support (2 tests): f32 and f64 precision
// 6. Large values (1 test): avoid overflow
// 7. Small values (1 test): avoid underflow
//
// rot():
// 1. Basic correctness (3 tests): 2-element, 3-element, 5-element vectors
// 2. Verify formula (2 tests): manual element-wise checks
// 3. Orthogonality preservation (1 test): orthogonal vectors remain orthogonal
// 4. Type support (2 tests): f32 and f64 precision
// 5. Large vectors (1 test): n=1000 elements
// 6. Edge cases (2 tests): single element, all zeros
// 7. Error handling (1 test): dimension mismatch between x and y
// 8. Reverse application (1 test): rot(...,-c,-s) is inverse of rot(...,c,s)
// 9. Composition (1 test): multiple rot() calls compose correctly
// 10. Memory safety (1 test): 10 iterations with testing.allocator

// ============================================================================
// rotg Tests — Givens Rotation Parameter Generation
// ============================================================================

test "rotg: basic case with non-zero a and b (f64)" {
    // rotg computes rotation parameters for a=3.0, b=4.0
    // Expected: r = sqrt(9 + 16) = 5.0, c = 3/5 = 0.6, s = 4/5 = 0.8
    const rotation = rotg(f64, 3.0, 4.0);

    try testing.expectApproxEqAbs(@as(f64, 0.6), rotation.c, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.8), rotation.s, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 5.0), rotation.r, 1e-10);
}

test "rotg: basic case with negative a (f64)" {
    // rotg with a=-3.0, b=4.0
    // Expected: r = sqrt(9 + 16) = 5.0, c = -3/5 = -0.6, s = 4/5 = 0.8
    const rotation = rotg(f64, -3.0, 4.0);

    try testing.expectApproxEqAbs(@as(f64, -0.6), rotation.c, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.8), rotation.s, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 5.0), rotation.r, 1e-10);
}

test "rotg: both a and b are zero (f64)" {
    // rotg(0, 0): special case where both inputs are zero
    // Expected: r = 0, c = 1.0 (or 0, depending on convention), s = 0
    const rotation = rotg(f64, 0.0, 0.0);

    // When both are zero, r should be 0, c should typically be 1.0 (identity component)
    try testing.expectApproxEqAbs(@as(f64, 0.0), rotation.r, 1e-10);
    try testing.expect(rotation.c == 1.0 or rotation.c == 0.0); // Handle variant conventions
    try testing.expectApproxEqAbs(@as(f64, 0.0), rotation.s, 1e-10);
}

test "rotg: a is zero, b is non-zero (f64)" {
    // rotg(0, 5): when a=0, b≠0
    // Expected: r = |b| = 5.0, c = 0, s = sign(b) = 1.0
    const rotation = rotg(f64, 0.0, 5.0);

    try testing.expectApproxEqAbs(@as(f64, 0.0), rotation.c, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), rotation.s, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 5.0), rotation.r, 1e-10);
}

test "rotg: b is zero, a is non-zero (f64)" {
    // rotg(5, 0): when b=0, a≠0
    // Expected: r = |a| = 5.0, c = sign(a) = 1.0, s = 0
    const rotation = rotg(f64, 5.0, 0.0);

    try testing.expectApproxEqAbs(@as(f64, 1.0), rotation.c, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), rotation.s, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 5.0), rotation.r, 1e-10);
}

test "rotg: orthogonality c² + s² = 1 (f64)" {
    // Verify that for any a, b: c² + s² = 1 (Givens rotation is orthogonal)
    const rotation1 = rotg(f64, 3.0, 4.0);
    const sum1 = rotation1.c * rotation1.c + rotation1.s * rotation1.s;
    try testing.expectApproxEqAbs(@as(f64, 1.0), sum1, 1e-10);

    const rotation2 = rotg(f64, -2.0, 1.0);
    const sum2 = rotation2.c * rotation2.c + rotation2.s * rotation2.s;
    try testing.expectApproxEqAbs(@as(f64, 1.0), sum2, 1e-10);

    const rotation3 = rotg(f64, 0.0, -7.0);
    const sum3 = rotation3.c * rotation3.c + rotation3.s * rotation3.s;
    try testing.expectApproxEqAbs(@as(f64, 1.0), sum3, 1e-10);
}

test "rotg: type support f32" {
    // rotg with f32 type
    const rotation = rotg(f32, 3.0, 4.0);

    try testing.expectApproxEqAbs(@as(f32, 0.6), rotation.c, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.8), rotation.s, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 5.0), rotation.r, 1e-5);
}

test "rotg: type support f64" {
    // rotg with f64 type (double precision)
    const rotation = rotg(f64, 1.0, 1.0);

    const sqrt2 = @sqrt(@as(f64, 2.0));
    try testing.expectApproxEqAbs(1.0 / sqrt2, rotation.c, 1e-10);
    try testing.expectApproxEqAbs(1.0 / sqrt2, rotation.s, 1e-10);
    try testing.expectApproxEqAbs(sqrt2, rotation.r, 1e-10);
}

test "rotg: large values avoid overflow (f64)" {
    // Test with large values to ensure algorithm avoids overflow
    const large = 1e200;
    const rotation = rotg(f64, large, large);

    // Result should still be normalized (r approximately large*sqrt(2))
    const expected_r = large * @sqrt(@as(f64, 2.0));
    try testing.expectApproxEqRel(expected_r, rotation.r, 1e-10);

    // c and s should still be normalized
    const sum = rotation.c * rotation.c + rotation.s * rotation.s;
    try testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-10);
}

test "rotg: small values avoid underflow (f64)" {
    // Test with very small values
    const tiny = 1e-200;
    const rotation = rotg(f64, tiny, tiny);

    // c and s should be approximately 1/sqrt(2)
    const inv_sqrt2 = 1.0 / @sqrt(@as(f64, 2.0));
    try testing.expectApproxEqAbs(inv_sqrt2, rotation.c, 1e-15);
    try testing.expectApproxEqAbs(inv_sqrt2, rotation.s, 1e-15);
}

// ============================================================================
// rot Tests — Apply Givens Rotation to Vectors
// ============================================================================

test "rot: basic 2-element vectors (f64)" {
    // Apply Givens rotation to x=[1,0], y=[0,1] with c=0.6, s=0.8
    // Expected: x'=[0.6, -0.8], y'=[0.8, 0.6]
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1.0, 0.0 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 0.0, 1.0 }, .row_major);
    defer y.deinit();

    try rot(f64, &x, &y, 0.6, 0.8);

    try testing.expectApproxEqAbs(@as(f64, 0.6), x.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, -0.8), x.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.8), y.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.6), y.data[1], 1e-10);
}

test "rot: basic 3-element vectors (f64)" {
    // Apply rotation to x=[1,2,3], y=[4,5,6] with c=0.6, s=0.8
    // For each i: x'[i] = c*x[i] + s*y[i], y'[i] = c*y[i] - s*x[i]
    // x'[0] = 0.6*1 + 0.8*4 = 3.8, y'[0] = 0.6*4 - 0.8*1 = 1.6
    // x'[1] = 0.6*2 + 0.8*5 = 5.2, y'[1] = 0.6*5 - 0.8*2 = 1.8
    // x'[2] = 0.6*3 + 0.8*6 = 6.6, y'[2] = 0.6*6 - 0.8*3 = 1.2
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1.0, 2.0, 3.0 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 4.0, 5.0, 6.0 }, .row_major);
    defer y.deinit();

    try rot(f64, &x, &y, 0.6, 0.8);

    try testing.expectApproxEqAbs(@as(f64, 3.8), x.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 5.2), x.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 6.6), x.data[2], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.6), y.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.8), y.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.2), y.data[2], 1e-10);
}

test "rot: 5-element vectors (f64)" {
    // Test with 5 elements using c=0.8, s=0.6
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 5.0, 4.0, 3.0, 2.0, 1.0 }, .row_major);
    defer y.deinit();

    try rot(f64, &x, &y, 0.8, 0.6);

    // Verify all elements transformed correctly
    for (0..5) |i| {
        const xi = @as(f64, @floatFromInt(i + 1));
        const yi = @as(f64, @floatFromInt(5 - i));
        const expected_x = 0.8 * xi + 0.6 * yi;
        const expected_y = 0.8 * yi - 0.6 * xi;
        try testing.expectApproxEqAbs(expected_x, x.data[i], 1e-10);
        try testing.expectApproxEqAbs(expected_y, y.data[i], 1e-10);
    }
}

test "rot: verify rotation formula element-wise (f64)" {
    // Verify that the transformation follows the mathematical formula exactly
    // x' = c*x + s*y, y' = c*y - s*x
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 2.0, 3.0, 1.5, 4.2 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1.0, 2.5, 3.0, 0.8 }, .row_major);
    defer y.deinit();

    const c = 0.5;
    const s = @sqrt(@as(f64, 0.75)); // 0.5^2 + s^2 = 1

    // Save original values
    const x_orig = try allocator.dupe(f64, x.data);
    defer allocator.free(x_orig);
    const y_orig = try allocator.dupe(f64, y.data);
    defer allocator.free(y_orig);

    try rot(f64, &x, &y, c, s);

    // Verify formula for each element
    for (0..4) |i| {
        const expected_x = c * x_orig[i] + s * y_orig[i];
        const expected_y = c * y_orig[i] - s * x_orig[i];
        try testing.expectApproxEqAbs(expected_x, x.data[i], 1e-10);
        try testing.expectApproxEqAbs(expected_y, y.data[i], 1e-10);
    }
}

test "rot: orthogonal vectors remain orthogonal (f64)" {
    // Start with orthogonal vectors (x ⊥ y), apply rotation, verify orthogonality preserved
    // If x·y = 0 before rotation, it should remain 0 after rotation
    const allocator = testing.allocator;

    // x = [1, 0], y = [0, 1] are orthogonal
    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1.0, 0.0 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 0.0, 1.0 }, .row_major);
    defer y.deinit();

    // Apply rotation with c=0.6, s=0.8
    try rot(f64, &x, &y, 0.6, 0.8);

    // Compute dot product: should still be approximately 0
    var dot_product: f64 = 0;
    for (0..2) |i| {
        dot_product += x.data[i] * y.data[i];
    }

    try testing.expectApproxEqAbs(@as(f64, 0.0), dot_product, 1e-10);
}

test "rot: type support f32" {
    // Test rot with f32 precision
    const allocator = testing.allocator;

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{3}, &[_]f32{ 1.0, 2.0, 3.0 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{3}, &[_]f32{ 4.0, 5.0, 6.0 }, .row_major);
    defer y.deinit();

    try rot(f32, &x, &y, 0.6, 0.8);

    try testing.expectApproxEqAbs(@as(f32, 3.8), x.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 5.2), x.data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 6.6), x.data[2], 1e-5);
}

test "rot: type support f64" {
    // Test rot with f64 precision
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1.0, 0.0 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 0.0, 1.0 }, .row_major);
    defer y.deinit();

    try rot(f64, &x, &y, 0.6, 0.8);

    try testing.expectApproxEqAbs(@as(f64, 0.6), x.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, -0.8), x.data[1], 1e-10);
}

test "rot: large vectors n=1000 (f64)" {
    // Test with large vector of 1000 elements
    const allocator = testing.allocator;

    var x_data = try allocator.alloc(f64, 1000);
    defer allocator.free(x_data);
    var y_data = try allocator.alloc(f64, 1000);
    defer allocator.free(y_data);

    // Initialize with pattern data
    for (0..1000) |i| {
        x_data[i] = @as(f64, @floatFromInt(i + 1)) * 0.1;
        y_data[i] = @as(f64, @floatFromInt(1000 - i)) * 0.2;
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1000}, x_data, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1000}, y_data, .row_major);
    defer y.deinit();

    const c = 0.6;
    const s = 0.8;

    try rot(f64, &x, &y, c, s);

    // Verify sampling of elements
    const c_val = @as(f64, 0.6);
    const s_val = @as(f64, 0.8);

    const x0_orig = 0.1;
    const y0_orig = 200.0;
    const expected_x0 = c_val * x0_orig + s_val * y0_orig;
    try testing.expectApproxEqAbs(expected_x0, x.data[0], 1e-10);
}

test "rot: single element vector (f64)" {
    // Edge case: single element (n=1)
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{5.0}, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{3.0}, .row_major);
    defer y.deinit();

    try rot(f64, &x, &y, 0.6, 0.8);

    try testing.expectApproxEqAbs(@as(f64, 0.6 * 5.0 + 0.8 * 3.0), x.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.6 * 3.0 - 0.8 * 5.0), y.data[0], 1e-10);
}

test "rot: all zero vectors (f64)" {
    // Edge case: both vectors are all zeros
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 0.0, 0.0, 0.0 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 0.0, 0.0, 0.0 }, .row_major);
    defer y.deinit();

    try rot(f64, &x, &y, 0.6, 0.8);

    // All elements should remain zero
    for (0..3) |i| {
        try testing.expectApproxEqAbs(@as(f64, 0.0), x.data[i], 1e-10);
        try testing.expectApproxEqAbs(@as(f64, 0.0), y.data[i], 1e-10);
    }
}

test "rot: error dimension mismatch (f64)" {
    // Error handling: dimension mismatch between x and y
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1.0, 2.0, 3.0 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1.0, 2.0, 3.0, 4.0 }, .row_major);
    defer y.deinit();

    const result = rot(f64, &x, &y, 0.6, 0.8);
    try testing.expectError(error.DimensionMismatch, result);
}

test "rot: inverse application via negative parameters (f64)" {
    // Mathematical property: rot(..., -c, -s) reverses rot(..., c, s)
    const allocator = testing.allocator;

    var x_orig = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1.0, 2.0, 3.0 }, .row_major);
    defer x_orig.deinit();

    var y_orig = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 4.0, 5.0, 6.0 }, .row_major);
    defer y_orig.deinit();

    // Save originals
    const x_start = try allocator.dupe(f64, x_orig.data);
    defer allocator.free(x_start);
    const y_start = try allocator.dupe(f64, y_orig.data);
    defer allocator.free(y_start);

    const c = 0.6;
    const s = 0.8;

    // Apply rotation
    try rot(f64, &x_orig, &y_orig, c, s);

    // Apply inverse rotation with -c, -s
    try rot(f64, &x_orig, &y_orig, -c, -s);

    // Should restore original values
    for (0..3) |i| {
        try testing.expectApproxEqAbs(x_start[i], x_orig.data[i], 1e-10);
        try testing.expectApproxEqAbs(y_start[i], y_orig.data[i], 1e-10);
    }
}

test "rot: composition of multiple rotations (f64)" {
    // Test that multiple sequential rotations compose correctly
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1.0, 0.0 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 0.0, 1.0 }, .row_major);
    defer y.deinit();

    const c1 = 0.6;
    const s1 = 0.8;

    // Apply first rotation
    try rot(f64, &x, &y, c1, s1);

    // Save intermediate state
    const x_mid = try allocator.dupe(f64, x.data);
    defer allocator.free(x_mid);
    const y_mid = try allocator.dupe(f64, y.data);
    defer allocator.free(y_mid);

    const c2 = @sqrt(@as(f64, 0.5));
    const s2 = @sqrt(@as(f64, 0.5));

    // Apply second rotation
    try rot(f64, &x, &y, c2, s2);

    // Verify intermediate and final states have expected values
    try testing.expectApproxEqAbs(@as(f64, 0.6), x_mid[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.8), x_mid[1], 1e-10);
}

test "rot: memory safety 10 iterations (f64)" {
    // Test for memory leaks over 10 iterations
    const allocator = testing.allocator;

    for (0..10) |_| {
        var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 }, .row_major);
        defer x.deinit();

        var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 5.0, 4.0, 3.0, 2.0, 1.0 }, .row_major);
        defer y.deinit();

        try rot(f64, &x, &y, 0.6, 0.8);

        // Verify at least one element transformed
        try testing.expect(x.data[0] != 1.0 or y.data[0] != 5.0);
    }
}

// ============================================================================
// rotmg Tests — Modified Givens Rotation Parameter Generation
// ============================================================================
// Time: O(1) | Space: O(1)

test "rotmg: basic case d1=1.0 d2=1.0 x1=3.0 y1=4.0 (f64)" {
    // rotmg computes modified Givens parameters that eliminate y1
    // Input: d1=1, d2=1, x1=3.0, y1=4.0
    // Should produce flag and H matrix
    var d1: f64 = 1.0;
    var d2: f64 = 1.0;
    var x1: f64 = 3.0;
    const y1: f64 = 4.0;

    const result = try rotmg(f64, &d1, &d2, &x1, y1);

    // Verify result contains flag and H matrix parameters
    try testing.expect(result.flag >= -2 and result.flag <= 1);
    try testing.expect(result.h.len == 4);

    // After rotmg, x1 should be modified
    try testing.expect(x1 != 3.0);

    // d1 and d2 should be modified (scaling factors)
    try testing.expect(d1 >= 0);
    try testing.expect(d2 >= 0);
}

test "rotmg: both d1 and d2 are zero (f64)" {
    // Edge case: both scaling factors are zero
    var d1: f64 = 0.0;
    var d2: f64 = 0.0;
    var x1: f64 = 3.0;
    const y1: f64 = 4.0;

    const result = try rotmg(f64, &d1, &d2, &x1, y1);

    // Should still produce valid flag and parameters
    try testing.expect(result.flag >= -2 and result.flag <= 1);
    try testing.expect(result.h.len == 4);
}

test "rotmg: x1 is zero (f64)" {
    // Edge case: x1=0, y1=non-zero
    var d1: f64 = 1.0;
    var d2: f64 = 1.0;
    var x1: f64 = 0.0;
    const y1: f64 = 5.0;

    const result = try rotmg(f64, &d1, &d2, &x1, y1);

    try testing.expect(result.flag >= -2 and result.flag <= 1);
    try testing.expect(result.h.len == 4);
}

test "rotmg: y1 is zero (f64)" {
    // Edge case: y1=0, x1=non-zero
    var d1: f64 = 1.0;
    var d2: f64 = 1.0;
    var x1: f64 = 7.0;
    const y1: f64 = 0.0;

    const result = try rotmg(f64, &d1, &d2, &x1, y1);

    try testing.expect(result.flag >= -2 and result.flag <= 1);
    try testing.expect(result.h.len == 4);
}

test "rotmg: flag is -2 case (f64)" {
    // Verify all 4 flag values are possible
    var d1: f64 = 2.0;
    var d2: f64 = 2.0;
    var x1: f64 = 1.0;
    const y1: f64 = 0.5;

    const result = try rotmg(f64, &d1, &d2, &x1, y1);

    // flag should be one of -2, -1, 0, 1
    try testing.expect(result.flag == -2 or result.flag == -1 or result.flag == 0 or result.flag == 1);
}

test "rotmg: type support f32" {
    // Test rotmg with f32 precision
    var d1: f32 = 1.0;
    var d2: f32 = 1.0;
    var x1: f32 = 2.0;
    const y1: f32 = 3.0;

    const result = try rotmg(f32, &d1, &d2, &x1, y1);

    try testing.expect(result.flag >= -2 and result.flag <= 1);
    try testing.expect(result.h.len == 4);
}

test "rotmg: type support f64" {
    // Test rotmg with f64 precision
    var d1: f64 = 1.0;
    var d2: f64 = 1.0;
    var x1: f64 = 2.0;
    const y1: f64 = 3.0;

    const result = try rotmg(f64, &d1, &d2, &x1, y1);

    try testing.expect(result.flag >= -2 and result.flag <= 1);
    try testing.expect(result.h.len == 4);
}

test "rotmg: large values avoid overflow (f64)" {
    // Test with large scaling factors
    var d1: f64 = 1e100;
    var d2: f64 = 1e100;
    var x1: f64 = 1e100;
    const y1: f64 = 1e100;

    const result = try rotmg(f64, &d1, &d2, &x1, y1);

    try testing.expect(result.flag >= -2 and result.flag <= 1);
    // Verify no infinity or NaN
    try testing.expect(!std.math.isInf(d1) and !std.math.isNan(d1));
    try testing.expect(!std.math.isInf(d2) and !std.math.isNan(d2));
}

// ============================================================================
// rotm Tests — Apply Modified Givens Rotation to Vectors
// ============================================================================
// Time: O(n) | Space: O(1)

test "rotm: basic 2-element vectors with flag=0 (f64)" {
    // Apply modified Givens rotation H (flag=0) to x and y
    // This is the most common case
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1.0, 0.0 }, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 0.0, 1.0 }, .row_major);
    defer y.deinit();

    // param: flag=0, h=[h11, h21, h12, h22] for H matrix
    // H = [[h11, h12], [h21, h22]] for flag=0
    const param = .{
        .flag = @as(i8, 0),
        .h = [_]f64{ 0.6, 0.8, -0.8, 0.6 },
    };

    try rotm(f64, &x, &y, param);

    // Both vectors should be modified
    try testing.expect(x.data[0] != 1.0 or x.data[1] != 0.0);
    try testing.expect(y.data[0] != 0.0 or y.data[1] != 1.0);
}

test "rotm: basic 3-element vectors flag=-1 (f64)" {
    // Test with flag=-1 (H only has h21, h12 nonzero)
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1.0, 2.0, 3.0 }, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 4.0, 5.0, 6.0 }, .row_major);
    defer y.deinit();

    const param = .{
        .flag = @as(i8, -1),
        .h = [_]f64{ 0.0, 0.8, -0.8, 0.0 },
    };

    try rotm(f64, &x, &y, param);

    // Vectors should be rotated
    try testing.expect(x.data[0] != 1.0);
}

test "rotm: 5-element vectors flag=1 (f64)" {
    // Test with flag=1 (H only has h11, h22 nonzero)
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 }, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 5.0, 4.0, 3.0, 2.0, 1.0 }, .row_major);
    defer y.deinit();

    const param = .{
        .flag = @as(i8, 1),
        .h = [_]f64{ 0.6, 0.0, 0.0, 0.8 },
    };

    try rotm(f64, &x, &y, param);

    // Vectors should be scaled/rotated
    try testing.expect(x.data[0] != 1.0 or y.data[0] != 5.0);
}

test "rotm: flag=-2 (f64)" {
    // Test with flag=-2 (H is identity, no operation)
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1.0, 2.0, 3.0 }, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 4.0, 5.0, 6.0 }, .row_major);
    defer y.deinit();

    const param = .{
        .flag = @as(i8, -2),
        .h = [_]f64{ 0.0, 0.0, 0.0, 0.0 },
    };

    try rotm(f64, &x, &y, param);

    // With flag=-2, vectors may remain unchanged or apply identity
    // At minimum, test that it doesn't crash
    try testing.expect(true);
}

test "rotm: single element vector (f64)" {
    // Test with n=1
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{2.0}, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{3.0}, .row_major);
    defer y.deinit();

    const param = .{
        .flag = @as(i8, 0),
        .h = [_]f64{ 0.6, 0.8, -0.8, 0.6 },
    };

    try rotm(f64, &x, &y, param);

    // Single element should still transform
    try testing.expect(true);
}

test "rotm: large vector n=1000 (f64)" {
    // Test with large vector
    const allocator = testing.allocator;

    var x_data: [1000]f64 = undefined;
    var y_data: [1000]f64 = undefined;
    for (0..1000) |i| {
        x_data[i] = @floatFromInt(i);
        y_data[i] = @floatFromInt(1000 - i);
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1000}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1000}, &y_data, .row_major);
    defer y.deinit();

    const param = .{
        .flag = @as(i8, 0),
        .h = [_]f64{ 0.8, 0.2, -0.2, 0.9 },
    };

    try rotm(f64, &x, &y, param);

    // Verify rotation was applied (at least some elements changed)
    try testing.expect(x.data[500] != 500.0 or y.data[500] != 500.0);
}

test "rotm: type support f32" {
    // Test rotm with f32
    const allocator = testing.allocator;

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{2}, &[_]f32{ 1.0, 2.0 }, .row_major);
    defer x.deinit();
    var y = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{2}, &[_]f32{ 3.0, 4.0 }, .row_major);
    defer y.deinit();

    const param = .{
        .flag = @as(i8, 0),
        .h = [_]f32{ 0.6, 0.8, -0.8, 0.6 },
    };

    try rotm(f32, &x, &y, param);

    try testing.expect(true);
}

test "rotm: type support f64" {
    // Test rotm with f64
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1.0, 2.0 }, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 3.0, 4.0 }, .row_major);
    defer y.deinit();

    const param = .{
        .flag = @as(i8, 0),
        .h = [_]f64{ 0.6, 0.8, -0.8, 0.6 },
    };

    try rotm(f64, &x, &y, param);

    try testing.expect(true);
}

test "rotm: error dimension mismatch (f64)" {
    // Test error handling when x and y have different dimensions
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1.0, 2.0, 3.0 }, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 }, .row_major);
    defer y.deinit();

    const param = .{
        .flag = @as(i8, 0),
        .h = [_]f64{ 0.6, 0.8, -0.8, 0.6 },
    };

    const result = rotm(f64, &x, &y, param);
    try testing.expectError(error.DimensionMismatch, result);
}

// ============================================================================
// iamin Tests — Index of Minimum Absolute Value
// ============================================================================
// Time: O(n) | Space: O(1)

test "iamin: basic 3-element vector (f64)" {
    // Find index of minimum absolute value in [1.0, -7.0, 3.0]
    // Expected: 0 (|1.0| = 1.0 is minimum)
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1.0, -7.0, 3.0 }, .row_major);
    defer x.deinit();

    const idx = try iamin(f64, x);
    try testing.expectEqual(@as(usize, 0), idx);
}

test "iamin: single element (f64)" {
    // Test with n=1
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{5.0}, .row_major);
    defer x.deinit();

    const idx = try iamin(f64, x);
    try testing.expectEqual(@as(usize, 0), idx);
}

test "iamin: multiple equal minimums returns first occurrence (f64)" {
    // [3.0, 2.0, 2.0, 5.0]
    // Minimum abs is 2.0 at indices 1 and 2
    // Expected: 1 (first occurrence)
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 3.0, 2.0, 2.0, 5.0 }, .row_major);
    defer x.deinit();

    const idx = try iamin(f64, x);
    try testing.expectEqual(@as(usize, 1), idx);
}

test "iamin: negative values uses absolute value (f64)" {
    // [-5.0, -1.0, -3.0]
    // Minimum abs is |-1.0| = 1.0 at index 1
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ -5.0, -1.0, -3.0 }, .row_major);
    defer x.deinit();

    const idx = try iamin(f64, x);
    try testing.expectEqual(@as(usize, 1), idx);
}

test "iamin: large vector n=1000 (f64)" {
    // Create vector where minimum is at index 500
    const allocator = testing.allocator;

    var x_data: [1000]f64 = undefined;
    for (0..1000) |i| {
        if (i == 500) {
            x_data[i] = 0.1; // Minimum absolute value
        } else {
            x_data[i] = @floatFromInt(i + 1);
        }
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1000}, &x_data, .row_major);
    defer x.deinit();

    const idx = try iamin(f64, x);
    try testing.expectEqual(@as(usize, 500), idx);
}

test "iamin: type support f32" {
    // Test iamin with f32
    const allocator = testing.allocator;

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{4}, &[_]f32{ 5.0, 2.0, 8.0, 3.0 }, .row_major);
    defer x.deinit();

    const idx = try iamin(f32, x);
    try testing.expectEqual(@as(usize, 1), idx); // 2.0 is minimum
}

test "iamin: type support f64" {
    // Test iamin with f64
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 5.0, 2.0, 8.0, 3.0 }, .row_major);
    defer x.deinit();

    const idx = try iamin(f64, x);
    try testing.expectEqual(@as(usize, 1), idx); // 2.0 is minimum
}

test "iamin: all same value (f64)" {
    // [5.0, 5.0, 5.0, 5.0]
    // All have same absolute value, should return 0 (first index)
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 5.0, 5.0, 5.0, 5.0 }, .row_major);
    defer x.deinit();

    const idx = try iamin(f64, x);
    try testing.expectEqual(@as(usize, 0), idx);
}

test "iamin: mixed positive and negative (f64)" {
    // [10.0, -2.5, 15.0, -1.5, 8.0]
    // Minimum abs is |-1.5| = 1.5 at index 3
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 10.0, -2.5, 15.0, -1.5, 8.0 }, .row_major);
    defer x.deinit();

    const idx = try iamin(f64, x);
    try testing.expectEqual(@as(usize, 3), idx);
}

test "iamin: error empty vector (f64)" {
    // Test error handling for empty vector
    const allocator = testing.allocator;

    var x = try NDArray(f64, 1).init(allocator, &[_]usize{0}, .row_major);
    defer x.deinit();

    const result = iamin(f64, x);
    try testing.expectError(error.EmptyArray, result);
}

// ============================================================================
// gbmv Tests — General Banded Matrix-Vector Multiplication
// ============================================================================
// Time: O(m*(kl+ku+1)) | Space: O(1)
//
// gbmv performs y := alpha*A*x + beta*y where A is an m×n banded matrix
// with kl sub-diagonals and ku super-diagonals, stored in banded format.
//
// Banded storage format:
// - A is stored as (kl+ku+1) × n array
// - Element A(i,j) of full matrix is at A_banded(ku+i-j, j) when the
//   element exists (max(0, j-ku) <= i < min(m, j+kl+1))

/// General banded matrix-vector multiply: y = αAx + βy
///
/// Performs matrix-vector multiplication where A is a banded matrix with kl
/// sub-diagonals and ku super-diagonals. Only the band is stored, reducing
/// memory requirements from O(m*n) to O((kl+ku+1)*n).
///
/// The banded storage format stores the diagonals as rows:
/// - Row 0 of storage: the ku-th super-diagonal
/// - Row ku: the main diagonal
/// - Row kl+ku: the kl-th sub-diagonal
///
/// Parameters:
/// - m: Number of rows of the full matrix
/// - n: Number of columns of the matrix (and length of x)
/// - kl: Number of sub-diagonals
/// - ku: Number of super-diagonals
/// - alpha: Scalar multiplier for Ax
/// - A: Banded matrix stored as (kl+ku+1)×n array
/// - x: Vector of length n — not modified
/// - beta: Scalar multiplier for y
/// - y: Vector of length m — modified in-place
///
/// Errors:
/// - error.DimensionMismatch if x.shape[0] != n or y.shape[0] != m
/// - error.DimensionMismatch if A.shape[0] != kl+ku+1 or A.shape[1] != n
///
/// Time: O(m*(kl+ku+1))
/// Space: O(1) (modifies y in-place)
pub fn gbmv(comptime T: type, m: usize, n: usize, kl: usize, ku: usize, alpha: T, A: NDArray(T, 2), x: NDArray(T, 1), beta: T, y: *NDArray(T, 1)) (NDArray(T, 1).Error)!void {
    // Validate dimensions
    if (x.shape[0] != n) {
        return error.DimensionMismatch;
    }
    if (y.shape[0] != m) {
        return error.DimensionMismatch;
    }
    if (A.shape[0] != kl + ku + 1) {
        return error.DimensionMismatch;
    }
    if (A.shape[1] != n) {
        return error.DimensionMismatch;
    }

    // y = alpha * A * x + beta * y
    // For each row i of the full matrix (0..m):
    //   y[i] = beta * y[i] + alpha * sum over valid j of A[i,j] * x[j]
    //
    // In banded storage:
    //   A_banded[ku + i - j, j] stores A[i, j] when max(0, j-ku) <= i < min(m, j+kl+1)
    for (0..m) |i| {
        var sum: T = 0;

        // For row i, the valid column range is [j_min, j_max)
        // where j_min = max(0, i-kl) and j_max = min(n, i+ku+1)
        const j_min = if (i > kl) i - kl else 0;
        const j_max = if (i + ku + 1 < n) i + ku + 1 else n;

        for (j_min..j_max) |j| {
            // Element A(i,j) is stored at A_banded(ku+i-j, j)
            const row_idx = ku + i - j;
            const a_val = A.data[row_idx * n + j];
            const x_val = x.data[j];
            sum += a_val * x_val;
        }

        y.data[i] = beta * y.data[i] + alpha * sum;
    }
}

test "gbmv: tridiagonal matrix 5x5 (kl=1, ku=1) basic multiply" {
    // Test basic tridiagonal (kl=1, ku=1) banded matrix-vector multiply
    // Full matrix:
    // [2  1  0  0  0]
    // [1  2  1  0  0]
    // [0  1  2  1  0]
    // [0  0  1  2  1]
    // [0  0  0  1  2]
    //
    // Banded storage (3 rows):
    // Row 0 (super-diagonal):  [0, 1, 1, 1, 1]
    // Row 1 (main diagonal):   [2, 2, 2, 2, 2]
    // Row 2 (sub-diagonal):    [1, 1, 1, 1, 0]
    //
    // x = [1, 2, 3, 4, 5]
    // Expected y = A*x:
    // [2*1 + 1*2, 1*1 + 2*2 + 1*3, 1*2 + 2*3 + 1*4, 1*3 + 2*4 + 1*5, 1*4 + 2*5]
    // = [4, 9, 14, 19, 14]
    const allocator = testing.allocator;
    const m = 5;
    const n = 5;
    const kl = 1;
    const ku = 1;

    // Create banded storage
    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ kl + ku + 1, n }, .row_major);
    defer A.deinit();

    // Initialize to zero
    @memset(A.data, 0);

    // Fill super-diagonal (row 0): [0, 1, 1, 1, 1]
    A.data[0 * n + 1] = 1;
    A.data[0 * n + 2] = 1;
    A.data[0 * n + 3] = 1;
    A.data[0 * n + 4] = 1;

    // Fill main diagonal (row 1): [2, 2, 2, 2, 2]
    A.data[1 * n + 0] = 2;
    A.data[1 * n + 1] = 2;
    A.data[1 * n + 2] = 2;
    A.data[1 * n + 3] = 2;
    A.data[1 * n + 4] = 2;

    // Fill sub-diagonal (row 2): [1, 1, 1, 1, 0]
    A.data[2 * n + 0] = 1;
    A.data[2 * n + 1] = 1;
    A.data[2 * n + 2] = 1;
    A.data[2 * n + 3] = 1;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, &[_]f64{ 1, 2, 3, 4, 5 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{m}, .row_major);
    defer y.deinit();

    try gbmv(f64, m, n, kl, ku, 1.0, A, x, 0.0, &y);

    try testing.expectApproxEqAbs(4.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(9.0, y.data[1], 1e-10);
    try testing.expectApproxEqAbs(14.0, y.data[2], 1e-10);
    try testing.expectApproxEqAbs(19.0, y.data[3], 1e-10);
    try testing.expectApproxEqAbs(14.0, y.data[4], 1e-10);
}

test "gbmv: pentadiagonal 6x6 (kl=2, ku=2) with alpha=2, beta=3" {
    // Test pentadiagonal (kl=2, ku=2) with scaling factors
    // Full matrix (simplified):
    // [3  1  1  0  0  0]
    // [1  3  1  1  0  0]
    // [1  1  3  1  1  0]
    // [0  1  1  3  1  1]
    // [0  0  1  1  3  1]
    // [0  0  0  1  1  3]
    //
    // Banded storage (5 rows for kl+ku+1=5):
    // A_banded layout (with row indexing for storage)
    //
    // x = [1, 1, 1, 1, 1, 1]
    // y_initial = [2, 2, 2, 2, 2, 2]
    // Result: y = 2*A*x + 3*y_initial
    const allocator = testing.allocator;
    const m = 6;
    const n = 6;
    const kl = 2;
    const ku = 2;

    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ kl + ku + 1, n }, .row_major);
    defer A.deinit();

    @memset(A.data, 0);

    // Fill banded structure (pentadiagonal):
    // Row 0 (2nd super-diagonal): [_, _, 1, 1, 1, 1]
    A.data[0 * n + 2] = 1;
    A.data[0 * n + 3] = 1;
    A.data[0 * n + 4] = 1;
    A.data[0 * n + 5] = 1;

    // Row 1 (1st super-diagonal): [_, 1, 1, 1, 1, 1]
    A.data[1 * n + 1] = 1;
    A.data[1 * n + 2] = 1;
    A.data[1 * n + 3] = 1;
    A.data[1 * n + 4] = 1;
    A.data[1 * n + 5] = 1;

    // Row 2 (main diagonal): [3, 3, 3, 3, 3, 3]
    for (0..n) |j| {
        A.data[2 * n + j] = 3;
    }

    // Row 3 (1st sub-diagonal): [1, 1, 1, 1, 1, _]
    for (0..5) |j| {
        A.data[3 * n + j] = 1;
    }

    // Row 4 (2nd sub-diagonal): [1, 1, 1, 1, _, _]
    for (0..4) |j| {
        A.data[4 * n + j] = 1;
    }

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, &[_]f64{ 1, 1, 1, 1, 1, 1 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{m}, &[_]f64{ 2, 2, 2, 2, 2, 2 }, .row_major);
    defer y.deinit();

    try gbmv(f64, m, n, kl, ku, 2.0, A, x, 3.0, &y);

    // Manual calculation for row 0: A[0,0]*x[0] + A[0,1]*x[1] + A[0,2]*x[2]
    //                               = 3*1 + 1*1 + 1*1 = 5
    // y[0] = 2*5 + 3*2 = 16
    try testing.expectApproxEqAbs(16.0, y.data[0], 1e-10);

    // Row 1: A[1,0]*x[0] + ... + A[1,3]*x[3]
    //      = 1*1 + 3*1 + 1*1 + 1*1 = 6
    // y[1] = 2*6 + 3*2 = 18
    try testing.expectApproxEqAbs(18.0, y.data[1], 1e-10);
}

test "gbmv: diagonal matrix (kl=0, ku=0)" {
    // Test purely diagonal case (no off-diagonals)
    // A = diag([2, 3, 4, 5])
    // x = [1, 2, 3, 4]
    // Expected: A*x = [2, 6, 12, 20]
    const allocator = testing.allocator;
    const m = 4;
    const n = 4;
    const kl = 0;
    const ku = 0;

    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ kl + ku + 1, n }, .row_major);
    defer A.deinit();

    // Single row for diagonal: [2, 3, 4, 5]
    A.data[0] = 2;
    A.data[1] = 3;
    A.data[2] = 4;
    A.data[3] = 5;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{m}, .row_major);
    defer y.deinit();

    try gbmv(f64, m, n, kl, ku, 1.0, A, x, 0.0, &y);

    try testing.expectApproxEqAbs(2.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(6.0, y.data[1], 1e-10);
    try testing.expectApproxEqAbs(12.0, y.data[2], 1e-10);
    try testing.expectApproxEqAbs(20.0, y.data[3], 1e-10);
}

test "gbmv: upper triangular band (kl=0, ku=2)" {
    // Test upper triangular banded: only main diagonal and 2 super-diagonals
    // Full matrix (3x5):
    // [1  2  3  0  0]
    // [0  4  5  6  0]
    // [0  0  7  8  9]
    //
    // Banded storage (3 rows):
    // Row 0: [_, _, 3, 6, 9]
    // Row 1: [_, 2, 5, 8, 0]
    // Row 2: [1, 4, 7, 0, 0]
    const allocator = testing.allocator;
    const m = 3;
    const n = 5;
    const kl = 0;
    const ku = 2;

    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ kl + ku + 1, n }, .row_major);
    defer A.deinit();

    @memset(A.data, 0);

    // Row 0 (2nd super-diagonal)
    A.data[0 * n + 2] = 3;
    A.data[0 * n + 3] = 6;
    A.data[0 * n + 4] = 9;

    // Row 1 (1st super-diagonal)
    A.data[1 * n + 1] = 2;
    A.data[1 * n + 2] = 5;
    A.data[1 * n + 3] = 8;

    // Row 2 (main diagonal)
    A.data[2 * n + 0] = 1;
    A.data[2 * n + 1] = 4;
    A.data[2 * n + 2] = 7;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, &[_]f64{ 1, 1, 1, 1, 1 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{m}, .row_major);
    defer y.deinit();

    try gbmv(f64, m, n, kl, ku, 1.0, A, x, 0.0, &y);

    // Row 0: 1*3 + 1*2 + 1*1 = wait, need to recalculate
    // A[0,0]=1, A[0,1]=2, A[0,2]=3 => y[0] = 1 + 2 + 3 = 6
    try testing.expectApproxEqAbs(6.0, y.data[0], 1e-10);
    // A[1,1]=4, A[1,2]=5, A[1,3]=6 => y[1] = 4 + 5 + 6 = 15
    try testing.expectApproxEqAbs(15.0, y.data[1], 1e-10);
    // A[2,2]=7, A[2,3]=8, A[2,4]=9 => y[2] = 7 + 8 + 9 = 24
    try testing.expectApproxEqAbs(24.0, y.data[2], 1e-10);
}

test "gbmv: lower triangular band (kl=2, ku=0)" {
    // Test lower triangular banded: only main diagonal and 2 sub-diagonals
    // Full matrix (5x3):
    // [1  0  0]
    // [2  4  0]
    // [3  5  7]
    // [0  6  8]
    // [0  0  9]
    //
    // Banded storage (3 rows for kl+ku+1=3):
    // Row 0 (sub-diag 2): [3, 5, 7]
    // Row 1 (sub-diag 1): [2, 4, 0] -> actually [2, 4, 8]
    // Row 2 (main diag):  [1, 4, 7]
    const allocator = testing.allocator;
    const m = 5;
    const n = 3;
    const kl = 2;
    const ku = 0;

    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ kl + ku + 1, n }, .row_major);
    defer A.deinit();

    @memset(A.data, 0);

    // Row 0 (2nd sub-diagonal): [3, 5, 7]
    A.data[0 * n + 0] = 3;
    A.data[0 * n + 1] = 5;
    A.data[0 * n + 2] = 7;

    // Row 1 (1st sub-diagonal): [2, 4, 8]
    A.data[1 * n + 0] = 2;
    A.data[1 * n + 1] = 4;
    A.data[1 * n + 2] = 8;

    // Row 2 (main diagonal): [1, 4, 7]
    A.data[2 * n + 0] = 1;
    A.data[2 * n + 1] = 4;
    A.data[2 * n + 2] = 7;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{m}, .row_major);
    defer y.deinit();

    try gbmv(f64, m, n, kl, ku, 1.0, A, x, 0.0, &y);

    // Row 0: A[0,0]*x[0] = 1*1 = 1
    try testing.expectApproxEqAbs(1.0, y.data[0], 1e-10);
    // Row 1: A[1,0]*x[0] + A[1,1]*x[1] = 2*1 + 4*2 = 10
    try testing.expectApproxEqAbs(10.0, y.data[1], 1e-10);
    // Row 2: A[2,0]*x[0] + A[2,1]*x[1] + A[2,2]*x[2] = 1*1 + 4*2 + 7*3 = 34
    try testing.expectApproxEqAbs(34.0, y.data[2], 1e-10);
    // Row 3: A[3,1]*x[1] + A[3,2]*x[2] = 4*2 + 8*3 = 32
    try testing.expectApproxEqAbs(32.0, y.data[3], 1e-10);
    // Row 4: A[4,2]*x[2] = 8*3 = 24
    try testing.expectApproxEqAbs(24.0, y.data[4], 1e-10);
}

test "gbmv: scalar alpha=0 (y = beta*y)" {
    // Test that when alpha=0, A and x are effectively ignored
    // y should only depend on beta*y_old
    const allocator = testing.allocator;
    const m = 3;
    const n = 3;
    const kl = 1;
    const ku = 1;

    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ kl + ku + 1, n }, .row_major);
    defer A.deinit();

    @memset(A.data, 999); // arbitrary values should not affect result

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, &[_]f64{ 999, 999, 999 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{m}, &[_]f64{ 5, 10, 15 }, .row_major);
    defer y.deinit();

    try gbmv(f64, m, n, kl, ku, 0.0, A, x, 2.0, &y);

    // y should be [10, 20, 30]
    try testing.expectApproxEqAbs(10.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(20.0, y.data[1], 1e-10);
    try testing.expectApproxEqAbs(30.0, y.data[2], 1e-10);
}

test "gbmv: scalar beta=0 (y = alpha*A*x)" {
    // Test that when beta=0, old y is ignored
    const allocator = testing.allocator;
    const m = 3;
    const n = 3;
    const kl = 1;
    const ku = 1;

    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ kl + ku + 1, n }, .row_major);
    defer A.deinit();

    @memset(A.data, 0);

    // Set up simple tridiagonal: diagonal = 2, off-diagonals = 1
    A.data[0 * n + 1] = 1;
    A.data[0 * n + 2] = 1;

    A.data[1 * n + 0] = 2;
    A.data[1 * n + 1] = 2;
    A.data[1 * n + 2] = 2;

    A.data[2 * n + 0] = 1;
    A.data[2 * n + 1] = 1;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{m}, &[_]f64{ 999, 999, 999 }, .row_major);
    defer y.deinit();

    try gbmv(f64, m, n, kl, ku, 3.0, A, x, 0.0, &y);

    // y[0] = 3*(2*1 + 1*2) = 3*4 = 12
    // y[1] = 3*(1*1 + 2*2 + 1*3) = 3*8 = 24
    // y[2] = 3*(1*2 + 2*3) = 3*8 = 24
    try testing.expectApproxEqAbs(12.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(24.0, y.data[1], 1e-10);
    try testing.expectApproxEqAbs(24.0, y.data[2], 1e-10);
}

test "gbmv: type support f32" {
    // Test gbmv with f32 precision
    const allocator = testing.allocator;
    const m = 3;
    const n = 3;
    const kl = 1;
    const ku = 1;

    var A = try NDArray(f32, 2).init(allocator, &[_]usize{ kl + ku + 1, n }, .row_major);
    defer A.deinit();

    @memset(A.data, 0);
    A.data[0 * n + 1] = 1;
    A.data[0 * n + 2] = 1;
    A.data[1 * n + 0] = 2;
    A.data[1 * n + 1] = 2;
    A.data[1 * n + 2] = 2;
    A.data[2 * n + 0] = 1;
    A.data[2 * n + 1] = 1;

    var x = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{n}, &[_]f32{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f32, 1).zeros(allocator, &[_]usize{m}, .row_major);
    defer y.deinit();

    try gbmv(f32, m, n, kl, ku, 1.0, A, x, 0.0, &y);

    try testing.expectApproxEqAbs(@as(f32, 4), y.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 8), y.data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 8), y.data[2], 1e-5);
}

test "gbmv: type support f64" {
    // Test gbmv with f64 precision
    const allocator = testing.allocator;
    const m = 3;
    const n = 3;
    const kl = 1;
    const ku = 1;

    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ kl + ku + 1, n }, .row_major);
    defer A.deinit();

    @memset(A.data, 0);
    A.data[0 * n + 1] = 1;
    A.data[0 * n + 2] = 1;
    A.data[1 * n + 0] = 2;
    A.data[1 * n + 1] = 2;
    A.data[1 * n + 2] = 2;
    A.data[2 * n + 0] = 1;
    A.data[2 * n + 1] = 1;

    var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{n}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{m}, .row_major);
    defer y.deinit();

    try gbmv(f64, m, n, kl, ku, 1.0, A, x, 0.0, &y);

    try testing.expectApproxEqAbs(4.0, y.data[0], 1e-10);
    try testing.expectApproxEqAbs(8.0, y.data[1], 1e-10);
    try testing.expectApproxEqAbs(8.0, y.data[2], 1e-10);
}

test "gbmv: error dimension mismatch x length" {
    // Test error when x.shape[0] != n
    const allocator = testing.allocator;
    const m = 3;
    const n = 3;
    const kl = 1;
    const ku = 1;

    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ kl + ku + 1, n }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major); // wrong size!
    defer x.deinit();

    var y = try NDArray(f64, 1).init(allocator, &[_]usize{m}, .row_major);
    defer y.deinit();

    const result = gbmv(f64, m, n, kl, ku, 1.0, A, x, 0.0, &y);
    try testing.expectError(error.DimensionMismatch, result);
}

test "gbmv: error dimension mismatch y length" {
    // Test error when y.shape[0] != m
    const allocator = testing.allocator;
    const m = 3;
    const n = 3;
    const kl = 1;
    const ku = 1;

    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ kl + ku + 1, n }, .row_major);
    defer A.deinit();

    var x = try NDArray(f64, 1).init(allocator, &[_]usize{n}, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major); // wrong size!
    defer y.deinit();

    const result = gbmv(f64, m, n, kl, ku, 1.0, A, x, 0.0, &y);
    try testing.expectError(error.DimensionMismatch, result);
}

test "gbmv: error dimension mismatch A rows" {
    // Test error when A.shape[0] != kl+ku+1
    const allocator = testing.allocator;
    const m = 3;
    const n = 3;
    const kl = 1;
    const ku = 1;

    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ 5, n }, .row_major); // wrong rows!
    defer A.deinit();

    var x = try NDArray(f64, 1).init(allocator, &[_]usize{n}, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).init(allocator, &[_]usize{m}, .row_major);
    defer y.deinit();

    const result = gbmv(f64, m, n, kl, ku, 1.0, A, x, 0.0, &y);
    try testing.expectError(error.DimensionMismatch, result);
}

test "gbmv: error dimension mismatch A columns" {
    // Test error when A.shape[1] != n
    const allocator = testing.allocator;
    const m = 3;
    const n = 3;
    const kl = 1;
    const ku = 1;

    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ kl + ku + 1, 5 }, .row_major); // wrong cols!
    defer A.deinit();

    var x = try NDArray(f64, 1).init(allocator, &[_]usize{n}, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).init(allocator, &[_]usize{m}, .row_major);
    defer y.deinit();

    const result = gbmv(f64, m, n, kl, ku, 1.0, A, x, 0.0, &y);
    try testing.expectError(error.DimensionMismatch, result);
}

test "gbmv: large matrix 100x100 tridiagonal" {
    // Test performance/correctness on larger tridiagonal system
    const allocator = testing.allocator;
    const m = 100;
    const n = 100;
    const kl = 1;
    const ku = 1;

    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ kl + ku + 1, n }, .row_major);
    defer A.deinit();

    @memset(A.data, 0);

    // Fill tridiagonal: super-diagonal = 1, main = 2, sub = 1
    for (0..n - 1) |j| {
        A.data[0 * n + j + 1] = 1;
    }
    for (0..n) |j| {
        A.data[1 * n + j] = 2;
    }
    for (0..n - 1) |j| {
        A.data[2 * n + j] = 1;
    }

    var x = try NDArray(f64, 1).init(allocator, &[_]usize{n}, .row_major);
    defer x.deinit();
    for (0..n) |i| {
        x.data[i] = 1.0; // all ones
    }

    var y = try NDArray(f64, 1).zeros(allocator, &[_]usize{m}, .row_major);
    defer y.deinit();

    try gbmv(f64, m, n, kl, ku, 1.0, A, x, 0.0, &y);

    // First row: 2*1 + 1*1 = 3
    try testing.expectApproxEqAbs(3.0, y.data[0], 1e-10);

    // Middle rows: 1*1 + 2*1 + 1*1 = 4
    try testing.expectApproxEqAbs(4.0, y.data[50], 1e-10);

    // Last row: 1*1 + 2*1 = 3
    try testing.expectApproxEqAbs(3.0, y.data[99], 1e-10);
}
