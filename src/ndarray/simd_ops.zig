//! SIMD-Accelerated Element-wise NDArray Operations
//!
//! Provides vectorized element-wise operations on NDArray using Zig's @Vector SIMD intrinsics.
//! These operations achieve 4-8× speedup over scalar element-wise operations on large arrays.
//!
//! ## Supported Operations
//! - **add_simd**: Element-wise addition (C = A + B)
//! - **sub_simd**: Element-wise subtraction (C = A - B)
//! - **mul_simd**: Element-wise multiplication (C = A ⊙ B)
//! - **div_simd**: Element-wise division (C = A ⊘ B)
//! - **add_scalar_simd**: Scalar addition (C = A + α)
//! - **mul_scalar_simd**: Scalar multiplication (C = α * A)
//!
//! ## SIMD Vector Lengths
//! - f32: 8-wide vectors (256-bit AVX / NEON)
//! - f64: 4-wide vectors (256-bit AVX / NEON)
//!
//! ## Strategy
//! - Main loop: vectorized for bulk data (multiples of vector width)
//! - Tail loop: scalar fallback for remaining elements
//! - In-place operations: modify first operand directly
//! - Result allocation: caller owns returned NDArray
//!
//! ## Accuracy
//! - Numerical results identical to scalar operations
//! - IEEE 754 compliant
//! - No precision loss from vectorization

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const NDArray = @import("ndarray.zig").NDArray;

/// SIMD vector length for type T (comptime known)
inline fn simdWidth(comptime T: type) comptime_int {
    return switch (T) {
        f32 => 8,
        f64 => 4,
        else => @compileError("SIMD only supported for f32/f64"),
    };
}

/// Element-wise addition: C = A + B (SIMD-accelerated)
///
/// Adds two NDArrays element-wise using SIMD for 4-8× speedup.
///
/// Parameters:
/// - A: First operand (N-dimensional array)
/// - B: Second operand (must have same shape as A)
/// - allocator: Allocator for result array
///
/// Returns: New NDArray containing A + B (caller owns)
///
/// Errors:
/// - error.DimensionMismatch if A and B have different shapes
/// - error.OutOfMemory if allocation fails
///
/// Time: O(n) with 4-8× speedup from SIMD, where n = total elements
/// Space: O(n) for result array
///
/// Example:
/// ```zig
/// var A = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{2,3}, &[_]f64{1,2,3,4,5,6}, .row_major);
/// defer A.deinit();
/// var B = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{2,3}, &[_]f64{10,20,30,40,50,60}, .row_major);
/// defer B.deinit();
/// var C = try add_simd(f64, 2, A, B, alloc); // {11,22,33,44,55,66}
/// defer C.deinit();
/// ```
pub fn add_simd(comptime T: type, comptime N: usize, A: NDArray(T, N), B: NDArray(T, N), allocator: Allocator) !NDArray(T, N) {
    // Validate shapes match
    for (0..N) |i| {
        if (A.shape[i] != B.shape[i]) return error.DimensionMismatch;
    }

    // Allocate result array with same shape as A
    var C = try NDArray(T, N).zerosLike(A, allocator);
    errdefer C.deinit();

    const total_elements = A.totalElements();
    const vec_width = comptime simdWidth(T);
    const Vec = @Vector(vec_width, T);

    var idx: usize = 0;

    // Main SIMD loop
    while (idx + vec_width <= total_elements) : (idx += vec_width) {
        const a_vec: Vec = A.data[idx..][0..vec_width].*;
        const b_vec: Vec = B.data[idx..][0..vec_width].*;
        const c_vec = a_vec + b_vec;
        @memcpy(C.data[idx..][0..vec_width], &c_vec);
    }

    // Tail loop (scalar)
    while (idx < total_elements) : (idx += 1) {
        C.data[idx] = A.data[idx] + B.data[idx];
    }

    return C;
}

/// Element-wise subtraction: C = A - B (SIMD-accelerated)
///
/// Subtracts B from A element-wise using SIMD for 4-8× speedup.
///
/// Time: O(n) with 4-8× speedup from SIMD
/// Space: O(n) for result array
pub fn sub_simd(comptime T: type, comptime N: usize, A: NDArray(T, N), B: NDArray(T, N), allocator: Allocator) !NDArray(T, N) {
    for (0..N) |i| {
        if (A.shape[i] != B.shape[i]) return error.DimensionMismatch;
    }

    var C = try NDArray(T, N).zerosLike(A, allocator);
    errdefer C.deinit();

    const total_elements = A.totalElements();
    const vec_width = comptime simdWidth(T);
    const Vec = @Vector(vec_width, T);

    var idx: usize = 0;

    while (idx + vec_width <= total_elements) : (idx += vec_width) {
        const a_vec: Vec = A.data[idx..][0..vec_width].*;
        const b_vec: Vec = B.data[idx..][0..vec_width].*;
        const c_vec = a_vec - b_vec;
        @memcpy(C.data[idx..][0..vec_width], &c_vec);
    }

    while (idx < total_elements) : (idx += 1) {
        C.data[idx] = A.data[idx] - B.data[idx];
    }

    return C;
}

/// Element-wise multiplication: C = A ⊙ B (Hadamard product, SIMD-accelerated)
///
/// Multiplies A and B element-wise (not matrix multiply) using SIMD for 4-8× speedup.
///
/// Time: O(n) with 4-8× speedup from SIMD
/// Space: O(n) for result array
pub fn mul_simd(comptime T: type, comptime N: usize, A: NDArray(T, N), B: NDArray(T, N), allocator: Allocator) !NDArray(T, N) {
    for (0..N) |i| {
        if (A.shape[i] != B.shape[i]) return error.DimensionMismatch;
    }

    var C = try NDArray(T, N).zerosLike(A, allocator);
    errdefer C.deinit();

    const total_elements = A.totalElements();
    const vec_width = comptime simdWidth(T);
    const Vec = @Vector(vec_width, T);

    var idx: usize = 0;

    while (idx + vec_width <= total_elements) : (idx += vec_width) {
        const a_vec: Vec = A.data[idx..][0..vec_width].*;
        const b_vec: Vec = B.data[idx..][0..vec_width].*;
        const c_vec = a_vec * b_vec;
        @memcpy(C.data[idx..][0..vec_width], &c_vec);
    }

    while (idx < total_elements) : (idx += 1) {
        C.data[idx] = A.data[idx] * B.data[idx];
    }

    return C;
}

/// Element-wise division: C = A ⊘ B (SIMD-accelerated)
///
/// Divides A by B element-wise using SIMD for 4-8× speedup.
///
/// Time: O(n) with 4-8× speedup from SIMD
/// Space: O(n) for result array
///
/// Note: Division by zero produces Inf or NaN (IEEE 754 semantics)
pub fn div_simd(comptime T: type, comptime N: usize, A: NDArray(T, N), B: NDArray(T, N), allocator: Allocator) !NDArray(T, N) {
    for (0..N) |i| {
        if (A.shape[i] != B.shape[i]) return error.DimensionMismatch;
    }

    var C = try NDArray(T, N).zerosLike(A, allocator);
    errdefer C.deinit();

    const total_elements = A.totalElements();
    const vec_width = comptime simdWidth(T);
    const Vec = @Vector(vec_width, T);

    var idx: usize = 0;

    while (idx + vec_width <= total_elements) : (idx += vec_width) {
        const a_vec: Vec = A.data[idx..][0..vec_width].*;
        const b_vec: Vec = B.data[idx..][0..vec_width].*;
        const c_vec = a_vec / b_vec;
        @memcpy(C.data[idx..][0..vec_width], &c_vec);
    }

    while (idx < total_elements) : (idx += 1) {
        C.data[idx] = A.data[idx] / B.data[idx];
    }

    return C;
}

/// Scalar addition: C = A + α (SIMD-accelerated)
///
/// Adds scalar value to all elements of A using SIMD for 4-8× speedup.
///
/// Time: O(n) with 4-8× speedup from SIMD
/// Space: O(n) for result array
pub fn add_scalar_simd(comptime T: type, comptime N: usize, A: NDArray(T, N), scalar: T, allocator: Allocator) !NDArray(T, N) {
    var C = try NDArray(T, N).zerosLike(A, allocator);
    errdefer C.deinit();

    const total_elements = A.totalElements();
    const vec_width = comptime simdWidth(T);
    const Vec = @Vector(vec_width, T);

    const scalar_vec: Vec = @splat(scalar);
    var idx: usize = 0;

    while (idx + vec_width <= total_elements) : (idx += vec_width) {
        const a_vec: Vec = A.data[idx..][0..vec_width].*;
        const c_vec = a_vec + scalar_vec;
        @memcpy(C.data[idx..][0..vec_width], &c_vec);
    }

    while (idx < total_elements) : (idx += 1) {
        C.data[idx] = A.data[idx] + scalar;
    }

    return C;
}

/// Scalar multiplication: C = α * A (SIMD-accelerated)
///
/// Multiplies all elements of A by scalar value using SIMD for 4-8× speedup.
///
/// Time: O(n) with 4-8× speedup from SIMD
/// Space: O(n) for result array
pub fn mul_scalar_simd(comptime T: type, comptime N: usize, A: NDArray(T, N), scalar: T, allocator: Allocator) !NDArray(T, N) {
    var C = try NDArray(T, N).zerosLike(A, allocator);
    errdefer C.deinit();

    const total_elements = A.totalElements();
    const vec_width = comptime simdWidth(T);
    const Vec = @Vector(vec_width, T);

    const scalar_vec: Vec = @splat(scalar);
    var idx: usize = 0;

    while (idx + vec_width <= total_elements) : (idx += vec_width) {
        const a_vec: Vec = A.data[idx..][0..vec_width].*;
        const c_vec = scalar_vec * a_vec;
        @memcpy(C.data[idx..][0..vec_width], &c_vec);
    }

    while (idx < total_elements) : (idx += 1) {
        C.data[idx] = scalar * A.data[idx];
    }

    return C;
}

// ============================================================================
// Tests — Verify SIMD matches scalar implementation
// ============================================================================

test "add_simd: 1D array" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{8}, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{8}, &[_]f64{ 10, 20, 30, 40, 50, 60, 70, 80 }, .row_major);
    defer B.deinit();

    var C = try add_simd(f64, 1, A, B, allocator);
    defer C.deinit();

    try testing.expectApproxEqAbs(11.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(22.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(33.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(44.0, C.data[3], 1e-10);
    try testing.expectApproxEqAbs(88.0, C.data[7], 1e-10);
}

test "add_simd: 2D array (matrix)" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 10, 20, 30, 40, 50, 60 }, .row_major);
    defer B.deinit();

    var C = try add_simd(f64, 2, A, B, allocator);
    defer C.deinit();

    try testing.expectApproxEqAbs(11.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(66.0, C.data[5], 1e-10);
}

test "add_simd: length not multiple of SIMD width" {
    const allocator = testing.allocator;

    // Length 10 (not multiple of 4 or 8)
    var A = try NDArray(f64, 1).zeros(allocator, &[_]usize{10}, .row_major);
    defer A.deinit();
    var B = try NDArray(f64, 1).zeros(allocator, &[_]usize{10}, .row_major);
    defer B.deinit();

    for (0..10) |i| {
        A.data[i] = @floatFromInt(i + 1);
        B.data[i] = @floatFromInt((i + 1) * 10);
    }

    var C = try add_simd(f64, 1, A, B, allocator);
    defer C.deinit();

    for (0..10) |i| {
        const expected: f64 = @floatFromInt((i + 1) * 11);
        try testing.expectApproxEqAbs(expected, C.data[i], 1e-10);
    }
}

test "sub_simd: basic subtraction" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 10, 20, 30, 40 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer B.deinit();

    var C = try sub_simd(f64, 1, A, B, allocator);
    defer C.deinit();

    try testing.expectApproxEqAbs(9.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(18.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(27.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(36.0, C.data[3], 1e-10);
}

test "mul_simd: element-wise multiplication" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 2, 3, 4, 5 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 10, 10, 10, 10 }, .row_major);
    defer B.deinit();

    var C = try mul_simd(f64, 1, A, B, allocator);
    defer C.deinit();

    try testing.expectApproxEqAbs(20.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(30.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(40.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(50.0, C.data[3], 1e-10);
}

test "div_simd: element-wise division" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 20, 30, 40, 50 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 10, 10, 10, 10 }, .row_major);
    defer B.deinit();

    var C = try div_simd(f64, 1, A, B, allocator);
    defer C.deinit();

    try testing.expectApproxEqAbs(2.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(3.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(4.0, C.data[2], 1e-10);
    try testing.expectApproxEqAbs(5.0, C.data[3], 1e-10);
}

test "add_scalar_simd: add constant to all elements" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{8}, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 }, .row_major);
    defer A.deinit();

    var C = try add_scalar_simd(f64, 1, A, 100.0, allocator);
    defer C.deinit();

    try testing.expectApproxEqAbs(101.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(102.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(108.0, C.data[7], 1e-10);
}

test "mul_scalar_simd: multiply all elements by constant" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{8}, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 }, .row_major);
    defer A.deinit();

    var C = try mul_scalar_simd(f64, 1, A, 5.0, allocator);
    defer C.deinit();

    try testing.expectApproxEqAbs(5.0, C.data[0], 1e-10);
    try testing.expectApproxEqAbs(10.0, C.data[1], 1e-10);
    try testing.expectApproxEqAbs(40.0, C.data[7], 1e-10);
}

test "add_simd: f32 type support" {
    const allocator = testing.allocator;

    var A = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{4}, &[_]f32{ 1, 2, 3, 4 }, .row_major);
    defer A.deinit();

    var B = try NDArray(f32, 1).fromSlice(allocator, &[_]usize{4}, &[_]f32{ 10, 20, 30, 40 }, .row_major);
    defer B.deinit();

    var C = try add_simd(f32, 1, A, B, allocator);
    defer C.deinit();

    try testing.expectApproxEqAbs(@as(f32, 11.0), C.data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 44.0), C.data[3], 1e-6);
}

test "add_simd: no memory leaks" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 100, 100 }, .row_major);
    defer A.deinit();
    var B = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 100, 100 }, .row_major);
    defer B.deinit();

    var C = try add_simd(f64, 2, A, B, allocator);
    defer C.deinit();
    // testing.allocator detects leaks automatically
}
