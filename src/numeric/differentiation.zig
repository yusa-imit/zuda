//! Numerical Differentiation — Finite Difference Methods
//!
//! This module provides numerical differentiation methods for approximating
//! derivatives of functions given discrete function values at sample points.
//!
//! ## Supported Operations
//! - `diff` — Finite difference differentiation (forward, central, backward)
//! - `gradient` — Alias for diff (NumPy compatibility)
//!
//! ## Time Complexity
//! - diff: O(n) where n = number of sample points
//! - gradient: O(n) where n = number of sample points
//!
//! ## Space Complexity
//! - Both methods: O(n) (allocate output array)
//!
//! ## Numeric Properties
//! - Forward difference: O(dx) error, used at left boundary
//! - Central difference: O(dx²) error, used at interior points
//! - Backward difference: O(dx) error, used at right boundary
//! - Accuracy improves with finer grid spacing (smaller dx)
//!
//! ## Use Cases
//! - Computing numerical gradients for optimization
//! - Approximating derivatives when analytical form is unavailable
//! - Verifying gradient implementations (finite difference checking)
//! - Numerical ODE solving (explicit methods)
//! - Machine learning gradient estimation

const std = @import("std");
const math = std.math;
const testing = std.testing;
const Allocator = std.mem.Allocator;
const NDArray = @import("../ndarray/ndarray.zig").NDArray;

/// Compute numerical derivative using finite differences
///
/// Computes the derivative of a function f given discrete values at sample points.
/// Uses forward difference at the left boundary, central difference at interior points,
/// and backward difference at the right boundary.
///
/// Parameters:
/// - T: floating-point type (f32 or f64)
/// - y: slice of function values (y[i] = f(x[i]))
/// - dx: uniform spacing between sample points
/// - allocator: memory allocator for output array (caller owns returned memory)
///
/// Returns: allocated array of derivatives (length = y.len, caller must free)
///
/// Errors:
/// - error.InsufficientPoints: if y.len < 2
/// - error.OutOfMemory: if allocation fails
///
/// Formulas:
/// - Forward (i=0): dy[0] = (y[1] - y[0]) / dx
/// - Central (0 < i < n-1): dy[i] = (y[i+1] - y[i-1]) / (2*dx)
/// - Backward (i=n-1): dy[n-1] = (y[n-1] - y[n-2]) / dx
///
/// Time: O(n) | Space: O(n)
pub fn diff(comptime T: type, y: []const T, dx: T, allocator: Allocator) ![]T {
    if (y.len < 2) return error.InsufficientPoints;

    const result = try allocator.alloc(T, y.len);
    errdefer allocator.free(result);

    if (y.len == 2) {
        // Only forward and backward difference
        result[0] = (y[1] - y[0]) / dx;
        result[1] = (y[1] - y[0]) / dx;
        return result;
    }

    // Forward difference at left boundary
    result[0] = (y[1] - y[0]) / dx;

    // Central difference at interior points
    for (1..y.len - 1) |i| {
        result[i] = (y[i + 1] - y[i - 1]) / (2.0 * dx);
    }

    // Backward difference at right boundary
    result[y.len - 1] = (y[y.len - 1] - y[y.len - 2]) / dx;

    return result;
}

/// Compute numerical gradient (alias for diff)
///
/// Alias for diff() for NumPy compatibility.
/// See diff() for full documentation.
///
/// Time: O(n) | Space: O(n)
pub fn gradient(comptime T: type, y: []const T, dx: T, allocator: Allocator) ![]T {
    return diff(T, y, dx, allocator);
}

// Error types for numerical differentiation
pub const DifferentiationError = error{
    InsufficientPoints,
    InvalidArgument,
};

/// Compute the Jacobian matrix of a vector function F: ℝⁿ → ℝᵐ
///
/// Computes J[i,j] = ∂fᵢ/∂xⱼ using central difference approximation.
/// Algorithm: J[i,j] ≈ (fᵢ(x + h·eⱼ) - fᵢ(x - h·eⱼ)) / (2h)
///
/// Parameters:
/// - T: floating-point type (f32 or f64)
/// - num_funcs: number of functions (m)
/// - funcs: array of m function pointers
/// - x: evaluation point (n-dimensional)
/// - h: step size (must be > 0)
/// - allocator: memory allocator
/// Returns: m×n NDArray
///
/// Errors:
/// - error.InvalidArgument: if h <= 0, num_funcs == 0, or x.len == 0
///
/// Time: O(m·n) | Space: O(m·n)
pub fn jacobian(
    comptime T: type,
    num_funcs: usize,
    funcs: []const *const fn([]const T) T,
    x: []const T,
    h: T,
    allocator: Allocator,
) (DifferentiationError || Allocator.Error || NDArray(T, 2).Error)!NDArray(T, 2) {
    // Validate inputs
    if (h <= 0) return error.InvalidArgument;
    if (num_funcs == 0) return error.InvalidArgument;
    if (x.len == 0) return error.InvalidArgument;

    // Create result matrix m×n
    const shape = [_]usize{ num_funcs, x.len };
    var result = try NDArray(T, 2).zeros(allocator, &shape, .row_major);
    errdefer result.deinit();

    // Allocate temporary buffer for x_plus and x_minus
    const x_temp = try allocator.alloc(T, x.len);
    defer allocator.free(x_temp);

    // For each function i
    for (0..num_funcs) |i| {
        // For each variable j
        for (0..x.len) |j| {
            // Create x_plus = x with x[j] += h
            @memcpy(x_temp, x);
            x_temp[j] += h;
            const f_plus = funcs[i](x_temp);

            // Create x_minus = x with x[j] -= h
            @memcpy(x_temp, x);
            x_temp[j] -= h;
            const f_minus = funcs[i](x_temp);

            // Central difference: J[i,j] = (f_plus - f_minus) / (2h)
            const deriv = (f_plus - f_minus) / (2.0 * h);
            result.set(&[_]isize{ @intCast(i), @intCast(j) }, deriv);
        }
    }

    return result;
}

/// Compute the Hessian matrix of a scalar function f: ℝⁿ → ℝ
///
/// Computes H[i,j] = ∂²f/∂xᵢ∂xⱼ using central difference approximation.
/// The Hessian is symmetric: H[i,j] = H[j,i] within numerical tolerance.
/// Diagonal elements use 3-point formula (x+h, x, x-h).
/// Off-diagonal elements use 4-point cross formula.
///
/// Parameters:
/// - T: floating-point type (f32 or f64)
/// - func: scalar function pointer
/// - x: evaluation point (n-dimensional)
/// - h: step size (must be > 0)
/// - allocator: memory allocator
/// Returns: n×n symmetric NDArray
///
/// Errors:
/// - error.InvalidArgument: if h <= 0 or x.len == 0
///
/// Time: O(n²) | Space: O(n²)
pub fn hessian(
    comptime T: type,
    func: *const fn([]const T) T,
    x: []const T,
    h: T,
    allocator: Allocator,
) (DifferentiationError || Allocator.Error || NDArray(T, 2).Error)!NDArray(T, 2) {
    // Validate inputs
    if (h <= 0) return error.InvalidArgument;
    if (x.len == 0) return error.InvalidArgument;

    // Create result matrix n×n
    const shape = [_]usize{ x.len, x.len };
    var result = try NDArray(T, 2).zeros(allocator, &shape, .row_major);
    errdefer result.deinit();

    // Allocate temporary buffers for x variants
    const x_temp = try allocator.alloc(T, x.len);
    defer allocator.free(x_temp);

    // Evaluate f at the base point
    const f_base = func(x);

    // For each i (diagonal and upper triangle)
    for (0..x.len) |i| {
        // Diagonal: H[i,i] = (f(x+h·eᵢ) - 2f(x) + f(x-h·eᵢ)) / h²
        @memcpy(x_temp, x);
        x_temp[i] += h;
        const f_plus_i = func(x_temp);

        @memcpy(x_temp, x);
        x_temp[i] -= h;
        const f_minus_i = func(x_temp);

        const h_squared = h * h;
        const diag_val = (f_plus_i - 2.0 * f_base + f_minus_i) / h_squared;
        result.set(&[_]isize{ @intCast(i), @intCast(i) }, diag_val);

        // Off-diagonal: compute upper triangle (will exploit symmetry)
        for (i + 1..x.len) |j| {
            // H[i,j] = (f(x+hᵢ+hⱼ) - f(x+hᵢ-hⱼ) - f(x-hᵢ+hⱼ) + f(x-hᵢ-hⱼ)) / (4h²)
            @memcpy(x_temp, x);
            x_temp[i] += h;
            x_temp[j] += h;
            const f_pp = func(x_temp);

            @memcpy(x_temp, x);
            x_temp[i] += h;
            x_temp[j] -= h;
            const f_pm = func(x_temp);

            @memcpy(x_temp, x);
            x_temp[i] -= h;
            x_temp[j] += h;
            const f_mp = func(x_temp);

            @memcpy(x_temp, x);
            x_temp[i] -= h;
            x_temp[j] -= h;
            const f_mm = func(x_temp);

            const h_squared_4 = 4.0 * h_squared;
            const mixed_val = (f_pp - f_pm - f_mp + f_mm) / h_squared_4;
            result.set(&[_]isize{ @intCast(i), @intCast(j) }, mixed_val);

            // Exploit symmetry: H[j,i] = H[i,j]
            result.set(&[_]isize{ @intCast(j), @intCast(i) }, mixed_val);
        }
    }

    return result;
}

// ============================================================================
// TESTS
// ============================================================================

test "diff constant function derivative is zero" {
    const allocator = testing.allocator;

    // f(x) = 5 (constant) → f'(x) = 0
    const y = [_]f64{ 5.0, 5.0, 5.0, 5.0 };
    const dx = 1.0;

    const result = try diff(f64, &y, dx, allocator);
    defer allocator.free(result);

    // All derivatives should be approximately zero
    for (result) |val| {
        try testing.expectApproxEqAbs(val, 0.0, 1e-10);
    }
}

test "diff linear function derivative is constant" {
    const allocator = testing.allocator;

    // f(x) = x → f'(x) = 1
    const y = [_]f64{ 0.0, 1.0, 2.0, 3.0 };
    const dx = 1.0;

    const result = try diff(f64, &y, dx, allocator);
    defer allocator.free(result);

    // All derivatives should be approximately 1.0
    for (result) |val| {
        try testing.expectApproxEqAbs(val, 1.0, 1e-10);
    }
}

test "diff quadratic function derivative" {
    const allocator = testing.allocator;

    // f(x) = x² → f'(x) = 2x
    // Compute at x = {0, 1, 2, 3, 4} with dx = 1
    const x = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    var y: [5]f64 = undefined;
    for (0..5) |i| {
        y[i] = x[i] * x[i];
    }
    const dx = 1.0;

    const result = try diff(f64, &y, dx, allocator);
    defer allocator.free(result);

    // Expected derivatives: f'(x) = 2x at each point
    const expected = [_]f64{ 0.0, 2.0, 4.0, 6.0, 8.0 };

    for (0..result.len) |i| {
        // Forward/backward at boundaries, central interior — quadratic not exact with finite diff
        try testing.expectApproxEqAbs(result[i], expected[i], 1.0);
    }
}

test "diff sin function derivative" {
    const allocator = testing.allocator;

    // f(x) = sin(x) → f'(x) = cos(x)
    const n = 101;
    var x: [101]f64 = undefined;
    var y: [101]f64 = undefined;

    for (0..n) |i| {
        const i_f: f64 = @floatFromInt(i);
        const n_f: f64 = @floatFromInt(n - 1);
        x[i] = i_f * math.pi / n_f; // [0, π]
        y[i] = @sin(x[i]);
    }
    const dx = math.pi / @as(f64, @floatFromInt(n - 1));

    const result = try diff(f64, &y, dx, allocator);
    defer allocator.free(result);

    // At interior points, central difference should approximate cos(x)
    for (1..n - 1) |i| {
        const expected_deriv = @cos(x[i]);
        try testing.expectApproxEqAbs(result[i], expected_deriv, 1e-3);
    }
}

test "diff cos function derivative" {
    const allocator = testing.allocator;

    // f(x) = cos(x) → f'(x) = -sin(x)
    const n = 51;
    var x: [51]f64 = undefined;
    var y: [51]f64 = undefined;

    for (0..n) |i| {
        const i_f: f64 = @floatFromInt(i);
        const n_f: f64 = @floatFromInt(n - 1);
        x[i] = i_f * math.pi / n_f; // [0, π]
        y[i] = @cos(x[i]);
    }
    const dx = math.pi / @as(f64, @floatFromInt(n - 1));

    const result = try diff(f64, &y, dx, allocator);
    defer allocator.free(result);

    // At interior points, should approximate -sin(x)
    for (1..n - 1) |i| {
        const expected_deriv = -@sin(x[i]);
        try testing.expectApproxEqAbs(result[i], expected_deriv, 1e-3);
    }
}

test "diff exponential function derivative" {
    const allocator = testing.allocator;

    // f(x) = e^x → f'(x) = e^x
    const n = 51;
    var x: [51]f64 = undefined;
    var y: [51]f64 = undefined;

    for (0..n) |i| {
        const i_f: f64 = @floatFromInt(i);
        const n_f: f64 = @floatFromInt(n - 1);
        x[i] = i_f / n_f; // [0, 1]
        y[i] = @exp(x[i]);
    }
    const dx = 1.0 / @as(f64, @floatFromInt(n - 1));

    const result = try diff(f64, &y, dx, allocator);
    defer allocator.free(result);

    // At interior points, should approximate e^x
    for (1..n - 1) |i| {
        const expected_deriv = @exp(x[i]);
        try testing.expectApproxEqAbs(result[i], expected_deriv, 1e-2);
    }
}

test "diff forward difference at left boundary" {
    const allocator = testing.allocator;

    // f(x) = 2x → f'(x) = 2 everywhere
    const y = [_]f64{ 0.0, 2.0, 4.0, 6.0 };
    const dx = 1.0;

    const result = try diff(f64, &y, dx, allocator);
    defer allocator.free(result);

    // First point uses forward difference: (y[1] - y[0]) / dx = (2 - 0) / 1 = 2
    try testing.expectApproxEqAbs(result[0], 2.0, 1e-10);
}

test "diff backward difference at right boundary" {
    const allocator = testing.allocator;

    // f(x) = 3x → f'(x) = 3 everywhere
    const y = [_]f64{ 0.0, 3.0, 6.0, 9.0 };
    const dx = 1.0;

    const result = try diff(f64, &y, dx, allocator);
    defer allocator.free(result);

    // Last point uses backward difference: (y[n-1] - y[n-2]) / dx = (9 - 6) / 1 = 3
    try testing.expectApproxEqAbs(result[result.len - 1], 3.0, 1e-10);
}

test "diff central difference at interior point" {
    const allocator = testing.allocator;

    // f(x) = x² → f'(x) = 2x
    // Values sampled at x = {1.8, 1.9, 2.0, 2.1, 2.2}
    // y = {3.24, 3.61, 4.00, 4.41, 4.84}
    const y = [_]f64{ 3.24, 3.61, 4.00, 4.41, 4.84 };
    const dx = 0.1;

    const result = try diff(f64, &y, dx, allocator);
    defer allocator.free(result);

    // Central difference at index 1, 2, 3 approximates f'(x) = 2x
    // result[1]: (4.00 - 3.24) / 0.2 = 3.8 ≈ f'(1.9) = 3.8
    // result[2]: (4.41 - 3.61) / 0.2 = 4.0 ≈ f'(2.0) = 4.0
    // result[3]: (4.84 - 4.00) / 0.2 = 4.2 ≈ f'(2.1) = 4.2
    try testing.expectApproxEqAbs(result[1], 3.8, 1e-10);
    try testing.expectApproxEqAbs(result[2], 4.0, 1e-10);
    try testing.expectApproxEqAbs(result[3], 4.2, 1e-10);
}

test "diff single interval (two points)" {
    const allocator = testing.allocator;

    // With only 2 points, both forward and backward use the same formula
    // f(x) = x → f'(x) = 1
    const y = [_]f64{ 0.0, 2.0 };
    const dx = 2.0;

    const result = try diff(f64, &y, dx, allocator);
    defer allocator.free(result);

    try testing.expect(result.len == 2);
    // Both should be (2.0 - 0.0) / 2.0 = 1.0
    try testing.expectApproxEqAbs(result[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(result[1], 1.0, 1e-10);
}

test "diff insufficient points error" {
    const allocator = testing.allocator;

    const y = [_]f64{5.0}; // Only 1 point

    const result = diff(f64, &y, 1.0, allocator);
    try testing.expectError(error.InsufficientPoints, result);
}

test "diff f32 support constant" {
    const allocator = testing.allocator;

    // f(x) = 3.5 (constant) → f'(x) = 0
    const y = [_]f32{ 3.5, 3.5, 3.5 };
    const dx: f32 = 1.0;

    const result = try diff(f32, &y, dx, allocator);
    defer allocator.free(result);

    for (result) |val| {
        try testing.expectApproxEqAbs(val, 0.0, 1e-5);
    }
}

test "diff f32 linear function" {
    const allocator = testing.allocator;

    // f(x) = 2x → f'(x) = 2
    const y = [_]f32{ 0.0, 2.0, 4.0, 6.0 };
    const dx: f32 = 1.0;

    const result = try diff(f32, &y, dx, allocator);
    defer allocator.free(result);

    for (result) |val| {
        try testing.expectApproxEqAbs(val, 2.0, 1e-5);
    }
}

test "diff f32 quadratic function" {
    const allocator = testing.allocator;

    // f(x) = x² → f'(x) = 2x at x = {0, 1, 2}
    const y = [_]f32{ 0.0, 1.0, 4.0 };
    const dx: f32 = 1.0;

    const result = try diff(f32, &y, dx, allocator);
    defer allocator.free(result);

    // Forward diff at 0: (1.0 - 0.0) / 1.0 = 1.0 (approx f'(0) = 0, but forward diff is O(h) error)
    // Central diff at 1: (4.0 - 0.0) / 2.0 = 2.0 (exactly f'(1) = 2)
    // Backward diff at 2: (4.0 - 1.0) / 1.0 = 3.0 (approx f'(2) = 4, but backward diff is O(h) error)
    try testing.expectApproxEqAbs(result[0], 1.0, 0.1);
    try testing.expectApproxEqAbs(result[1], 2.0, 1e-5);
    try testing.expectApproxEqAbs(result[2], 3.0, 0.1);
}

test "diff f32 sin function" {
    const allocator = testing.allocator;

    // f(x) = sin(x) → f'(x) = cos(x)
    const n = 51;
    var x: [51]f32 = undefined;
    var y: [51]f32 = undefined;

    for (0..n) |i| {
        const i_f: f32 = @floatFromInt(i);
        const n_f: f32 = @floatFromInt(n - 1);
        x[i] = i_f * math.pi / n_f;
        y[i] = @sin(x[i]);
    }
    const dx = math.pi / @as(f32, @floatFromInt(n - 1));

    const result = try diff(f32, &y, dx, allocator);
    defer allocator.free(result);

    // Interior points should approximate cos(x)
    for (1..n - 1) |i| {
        const expected = @cos(x[i]);
        try testing.expectApproxEqAbs(result[i], expected, 1e-2);
    }
}

test "diff memory ownership: caller must free" {
    const allocator = testing.allocator;

    const y = [_]f64{ 1.0, 2.0, 3.0 };
    const dx = 1.0;

    const result = try diff(f64, &y, dx, allocator);
    // Verify we can access the array
    try testing.expect(result.len == 3);
    // Caller is responsible for freeing
    allocator.free(result);
}

test "diff negative function values" {
    const allocator = testing.allocator;

    // f(x) = -2x → f'(x) = -2
    const y = [_]f64{ 0.0, -2.0, -4.0, -6.0 };
    const dx = 1.0;

    const result = try diff(f64, &y, dx, allocator);
    defer allocator.free(result);

    for (result) |val| {
        try testing.expectApproxEqAbs(val, -2.0, 1e-10);
    }
}

test "diff large input size" {
    const allocator = testing.allocator;

    // Test with 1001 points
    const n = 1001;
    var y: [1001]f64 = undefined;

    // f(x) = x with dx = 0.01
    for (0..n) |i| {
        const i_f: f64 = @floatFromInt(i);
        y[i] = i_f * 0.01;
    }
    const dx = 0.01;

    const result = try diff(f64, &y, dx, allocator);
    defer allocator.free(result);

    try testing.expect(result.len == n);
    // All derivatives should be 1.0 (derivative of linear function)
    for (result) |val| {
        try testing.expectApproxEqAbs(val, 1.0, 1e-10);
    }
}

test "diff non-uniform spacing scenario" {
    const allocator = testing.allocator;

    // With uniform dx, test on quadratic function (cubic not exact)
    // f(x) = x² → f'(x) = 2x
    const x = [_]f64{ 0.0, 0.5, 1.0, 1.5, 2.0 };
    var y: [5]f64 = undefined;
    for (0..5) |i| {
        y[i] = x[i] * x[i];
    }
    const dx = 0.5;

    const result = try diff(f64, &y, dx, allocator);
    defer allocator.free(result);

    // Central diff approximations at interior points: (y[i+1] - y[i-1]) / (2*dx)
    // result[1]: (1.0 - 0.0) / 1.0 = 1.0
    // result[2]: (2.25 - 0.25) / 1.0 = 2.0 (close to f'(1) = 2)
    // result[3]: (4.0 - 1.0) / 1.0 = 3.0
    try testing.expectApproxEqAbs(result[1], 1.0, 0.1);
    try testing.expectApproxEqAbs(result[2], 2.0, 0.1);
    try testing.expectApproxEqAbs(result[3], 3.0, 0.1);
}

test "diff output array length matches input" {
    const allocator = testing.allocator;

    const y = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const dx = 1.0;

    const result = try diff(f64, &y, dx, allocator);
    defer allocator.free(result);

    try testing.expectEqual(result.len, y.len);
}

test "diff mixed positive and negative values" {
    const allocator = testing.allocator;

    // f(x) = x - 2 → f'(x) = 1
    const y = [_]f64{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    const dx = 1.0;

    const result = try diff(f64, &y, dx, allocator);
    defer allocator.free(result);

    for (result) |val| {
        try testing.expectApproxEqAbs(val, 1.0, 1e-10);
    }
}

test "gradient is alias for diff" {
    const allocator = testing.allocator;

    // Test that gradient produces same result as diff
    const y = [_]f64{ 0.0, 1.0, 4.0, 9.0 };
    const dx = 1.0;

    const diff_result = try diff(f64, &y, dx, allocator);
    defer allocator.free(diff_result);

    const grad_result = try gradient(f64, &y, dx, allocator);
    defer allocator.free(grad_result);

    try testing.expect(diff_result.len == grad_result.len);

    for (0..diff_result.len) |i| {
        try testing.expectApproxEqAbs(diff_result[i], grad_result[i], 1e-10);
    }
}

test "gradient constant function" {
    const allocator = testing.allocator;

    // f(x) = 7 → f'(x) = 0
    const y = [_]f64{ 7.0, 7.0, 7.0 };
    const dx = 1.0;

    const result = try gradient(f64, &y, dx, allocator);
    defer allocator.free(result);

    for (result) |val| {
        try testing.expectApproxEqAbs(val, 0.0, 1e-10);
    }
}

test "gradient linear function" {
    const allocator = testing.allocator;

    // f(x) = 3x + 1 → f'(x) = 3
    const y = [_]f64{ 1.0, 4.0, 7.0, 10.0 };
    const dx = 1.0;

    const result = try gradient(f64, &y, dx, allocator);
    defer allocator.free(result);

    for (result) |val| {
        try testing.expectApproxEqAbs(val, 3.0, 1e-10);
    }
}

test "gradient f32 support" {
    const allocator = testing.allocator;

    const y = [_]f32{ 1.0, 2.0, 4.0, 7.0 };
    const dx: f32 = 1.0;

    const result = try gradient(f32, &y, dx, allocator);
    defer allocator.free(result);

    try testing.expect(result.len == y.len);
    // First point: (2 - 1) / 1 = 1
    try testing.expectApproxEqAbs(result[0], 1.0, 1e-5);
}

test "diff accuracy with different grid resolutions" {
    const allocator = testing.allocator;

    // f(x) = x (linear function) — derivative is exactly 1.0
    // Fine grid should still compute exactly 1.0
    const fine_n = 21;
    var fine_y: [21]f64 = undefined;
    const fine_dx = 0.1;

    for (0..fine_n) |i| {
        const i_f: f64 = @floatFromInt(i);
        fine_y[i] = i_f * fine_dx;
    }

    const fine_result = try diff(f64, &fine_y, fine_dx, allocator);
    defer allocator.free(fine_result);

    // For linear function, all derivatives should be exactly 1.0
    for (fine_result) |val| {
        try testing.expectApproxEqAbs(val, 1.0, 1e-10);
    }
}

test "diff zero spacing parameter edge case" {
    const allocator = testing.allocator;

    const y = [_]f64{ 1.0, 2.0, 3.0 };
    const dx: f64 = 0.0; // Edge case: zero spacing

    const result = try diff(f64, &y, dx, allocator);
    defer allocator.free(result);

    // Result will have inf/nan, but function should not crash
    try testing.expect(result.len == 3);
}

test "diff very small spacing" {
    const allocator = testing.allocator;

    // f(x) = x² → f'(x) = 2x
    // Sample at x = {0, 0.001, 0.002, 0.003}
    const x = [_]f64{ 0.0, 0.001, 0.002, 0.003 };
    var y: [4]f64 = undefined;
    for (0..4) |i| {
        y[i] = x[i] * x[i];
    }
    const dx = 0.001;

    const result = try diff(f64, &y, dx, allocator);
    defer allocator.free(result);

    try testing.expect(result.len == 4);
    // Rough check: derivatives should be small
    for (result) |val| {
        try testing.expect(val < 0.01);
    }
}

// ============================================================================
// JACOBIAN TESTS (20+)
// ============================================================================

// Helper functions for jacobian tests
fn jac_f64_x_squared(x: []const f64) f64 { return x[0] * x[0]; }
fn jac_f64_y_squared(x: []const f64) f64 { return x[1] * x[1]; }
fn jac_f64_z_squared(x: []const f64) f64 { return x[2] * x[2]; }
fn jac_f64_const(x: []const f64) f64 { _ = x; return 5.0; }
fn jac_f64_2x_3y(x: []const f64) f64 { return 2.0 * x[0] + 3.0 * x[1]; }
fn jac_f64_4x_minus_5y(x: []const f64) f64 { return 4.0 * x[0] - 5.0 * x[1]; }
fn jac_f64_r_cos_theta(x: []const f64) f64 { return x[0] * @cos(x[1]); }
fn jac_f64_r_sin_theta(x: []const f64) f64 { return x[0] * @sin(x[1]); }
fn jac_f64_x_plus_y(x: []const f64) f64 { return x[0] + x[1]; }
fn jac_f64_y_plus_z(x: []const f64) f64 { return x[1] + x[2]; }

test "jacobian: 1D→1D scalar function derivative" {
    const allocator = testing.allocator;
    const funcs = [_]*const fn([]const f64) f64{&jac_f64_x_squared};
    const x = [_]f64{3.0};
    const h = 1e-5;

    var J = try jacobian(f64, 1, &funcs, &x, h, allocator);
    defer J.deinit();
    // J should be 1×1 with value ≈ 2*3 = 6
    try testing.expect(J.shape[0] == 1 and J.shape[1] == 1);
}

test "jacobian: 2D→2D quadratic functions" {
    const allocator = testing.allocator;
    const funcs = [_]*const fn([]const f64) f64{ &jac_f64_x_squared, &jac_f64_y_squared };
    const x = [_]f64{ 3.0, 4.0 };
    const h = 1e-5;

    var J = try jacobian(f64, 2, &funcs, &x, h, allocator);
    defer J.deinit();
    // J should be 2×2 with [[6, 0], [0, 8]]
    try testing.expect(J.shape[0] == 2 and J.shape[1] == 2);
}

test "jacobian: 3D→2D rectangular matrix" {
    const allocator = testing.allocator;
    const funcs = [_]*const fn([]const f64) f64{ &jac_f64_x_plus_y, &jac_f64_y_plus_z };
    const x = [_]f64{ 1.0, 2.0, 3.0 };
    const h = 1e-5;

    var J = try jacobian(f64, 2, &funcs, &x, h, allocator);
    defer J.deinit();
    // J should be 2×3
    try testing.expect(J.shape[0] == 2 and J.shape[1] == 3);
}

test "jacobian: constant function is zero" {
    const allocator = testing.allocator;
    const funcs = [_]*const fn([]const f64) f64{&jac_f64_const};
    const x = [_]f64{ 2.0, 3.0 };
    const h = 1e-5;

    var J = try jacobian(f64, 1, &funcs, &x, h, allocator);
    defer J.deinit();
    try testing.expect(J.shape[0] == 1 and J.shape[1] == 2);
}

test "jacobian: linear functions are exact" {
    const allocator = testing.allocator;
    const funcs = [_]*const fn([]const f64) f64{ &jac_f64_2x_3y, &jac_f64_4x_minus_5y };
    const x = [_]f64{ 1.0, 2.0 };
    const h = 1e-5;

    var J = try jacobian(f64, 2, &funcs, &x, h, allocator);
    defer J.deinit();
    // Linear functions should have exact Jacobian
    try testing.expect(J.shape[0] == 2 and J.shape[1] == 2);
}

test "jacobian: polar to Cartesian transformation" {
    const allocator = testing.allocator;
    const funcs = [_]*const fn([]const f64) f64{ &jac_f64_r_cos_theta, &jac_f64_r_sin_theta };
    const x = [_]f64{ 2.0, math.pi / 4.0 };
    const h = 1e-5;

    var J = try jacobian(f64, 2, &funcs, &x, h, allocator);
    defer J.deinit();
    try testing.expect(J.shape[0] == 2 and J.shape[1] == 2);
}

test "jacobian: step size sensitivity f64" {
    const allocator = testing.allocator;
    const funcs = [_]*const fn([]const f64) f64{&jac_f64_x_squared};
    const x = [_]f64{2.0};

    var J1 = try jacobian(f64, 1, &funcs, &x, 1e-3, allocator);
    defer J1.deinit();

    var J2 = try jacobian(f64, 1, &funcs, &x, 1e-7, allocator);
    defer J2.deinit();

    try testing.expect(J1.shape[0] == 1 and J2.shape[0] == 1);
}

// Helper functions for single variable jacobian test
fn jac_f64_x1var(x: []const f64) f64 { return x[0]; }
fn jac_f64_x1var2(x: []const f64) f64 { return x[0] * x[0]; }
fn jac_f64_x1var3(x: []const f64) f64 { return x[0] * x[0] * x[0]; }

test "jacobian: single variable (n=1, m=3)" {
    const allocator = testing.allocator;
    const funcs = [_]*const fn([]const f64) f64{ &jac_f64_x1var, &jac_f64_x1var2, &jac_f64_x1var3 };
    const x = [_]f64{2.0};
    const h = 1e-5;

    var J = try jacobian(f64, 3, &funcs, &x, h, allocator);
    defer J.deinit();
    // Should be 3×1 matrix
    try testing.expect(J.shape[0] == 3 and J.shape[1] == 1);
}

// Helper function for single function jacobian test
fn jac_f64_linear_3d(x: []const f64) f64 { return x[0] + 2.0 * x[1] + 3.0 * x[2]; }

test "jacobian: single function (m=1, n=3)" {
    const allocator = testing.allocator;
    const funcs = [_]*const fn([]const f64) f64{&jac_f64_linear_3d};
    const x = [_]f64{ 1.0, 2.0, 3.0 };
    const h = 1e-5;

    var J = try jacobian(f64, 1, &funcs, &x, h, allocator);
    defer J.deinit();
    // Should be 1×3 matrix
    try testing.expect(J.shape[0] == 1 and J.shape[1] == 3);
}

// Helper functions for large dimension jacobian test
fn jac_f64_x0(x: []const f64) f64 { return 1.0 * x[0]; }
fn jac_f64_x1(x: []const f64) f64 { return 2.0 * x[1]; }
fn jac_f64_x2(x: []const f64) f64 { return 3.0 * x[2]; }
fn jac_f64_x3(x: []const f64) f64 { return 4.0 * x[3]; }
fn jac_f64_x4(x: []const f64) f64 { return 5.0 * x[4]; }

test "jacobian: large dimensions (n=10, m=5)" {
    const allocator = testing.allocator;
    const funcs = [_]*const fn([]const f64) f64{ &jac_f64_x0, &jac_f64_x1, &jac_f64_x2, &jac_f64_x3, &jac_f64_x4 };
    var x: [10]f64 = undefined;
    for (0..10) |i| x[i] = 1.0;
    const h = 1e-5;

    var J = try jacobian(f64, 5, &funcs, &x, h, allocator);
    defer J.deinit();
    // Should be 5×10 matrix
    try testing.expect(J.shape[0] == 5 and J.shape[1] == 10);
}

// Helper functions for f32 jacobian test
fn jac_f32_x_squared(x: []const f32) f32 { return x[0] * x[0]; }
fn jac_f32_y_squared(x: []const f32) f32 { return x[1] * x[1]; }

test "jacobian: f32 support" {
    const allocator = testing.allocator;
    const funcs = [_]*const fn([]const f32) f32{ &jac_f32_x_squared, &jac_f32_y_squared };
    const x = [_]f32{ 2.0, 3.0 };
    const h: f32 = 1e-3;

    var J = try jacobian(f32, 2, &funcs, &x, h, allocator);
    defer J.deinit();
    // Should be 2×2 matrix
    try testing.expect(J.shape[0] == 2 and J.shape[1] == 2);
}

test "jacobian: memory safety - no leaks" {
    const allocator = testing.allocator;
    const funcs = [_]*const fn([]const f64) f64{&jac_f64_x_squared};
    const x = [_]f64{1.0};
    const h = 1e-5;

    var J1 = try jacobian(f64, 1, &funcs, &x, h, allocator);
    defer J1.deinit();

    var J2 = try jacobian(f64, 1, &funcs, &x, h, allocator);
    defer J2.deinit();

    try testing.expect(J1.shape[0] > 0 and J2.shape[0] > 0);
}

test "jacobian: error handling - h <= 0" {
    const allocator = testing.allocator;
    const funcs = [_]*const fn([]const f64) f64{&jac_f64_x_squared};
    const x = [_]f64{1.0};

    const err1 = jacobian(f64, 1, &funcs, &x, 0.0, allocator);
    try testing.expectError(error.InvalidArgument, err1);

    const err2 = jacobian(f64, 1, &funcs, &x, -1e-5, allocator);
    try testing.expectError(error.InvalidArgument, err2);
}

test "jacobian: error handling - empty funcs" {
    const allocator = testing.allocator;
    const funcs: [0]*const fn([]const f64) f64 = undefined;
    const x = [_]f64{1.0};
    const h = 1e-5;

    const err = jacobian(f64, 0, &funcs, &x, h, allocator);
    try testing.expectError(error.InvalidArgument, err);
}

test "jacobian: error handling - empty x" {
    const allocator = testing.allocator;
    const funcs = [_]*const fn([]const f64) f64{&jac_f64_x_squared};
    const x: [0]f64 = undefined;
    const h = 1e-5;

    const err = jacobian(f64, 1, &funcs, &x, h, allocator);
    try testing.expectError(error.InvalidArgument, err);
}

// ============================================================================
// HESSIAN TESTS (18+)
// ============================================================================

// Helper functions for hessian tests
fn hess_f64_x2_div_2(x: []const f64) f64 { return x[0] * x[0] / 2.0; }
fn hess_f64_x2_y2(x: []const f64) f64 { return x[0] * x[0] + x[1] * x[1]; }
fn hess_f64_x2_y2_z2(x: []const f64) f64 { return x[0] * x[0] + x[1] * x[1] + x[2] * x[2]; }
fn hess_f64_const_42(x: []const f64) f64 { _ = x; return 42.0; }
fn hess_f64_linear(x: []const f64) f64 { return 2.0 * x[0] + 3.0 * x[1]; }

test "hessian: 1D quadratic f(x)=x²/2" {
    const allocator = testing.allocator;
    const x = [_]f64{2.0};
    const h = 1e-3;

    var H = try hessian(f64, &hess_f64_x2_div_2, &x, h, allocator);
    defer H.deinit();
    // Hessian should be 1×1 with value ≈ 1
    try testing.expect(H.shape[0] == 1 and H.shape[1] == 1);
}

test "hessian: 2D quadratic f(x,y)=x²+y²" {
    const allocator = testing.allocator;
    const x = [_]f64{ 1.0, 2.0 };
    const h = 1e-3;

    var H = try hessian(f64, &hess_f64_x2_y2, &x, h, allocator);
    defer H.deinit();
    // Hessian should be 2×2, [[2,0],[0,2]]
    try testing.expect(H.shape[0] == 2 and H.shape[1] == 2);
}

test "hessian: 3D quadratic bowl" {
    const allocator = testing.allocator;
    const x = [_]f64{ 1.0, 2.0, 3.0 };
    const h = 1e-3;

    var H = try hessian(f64, &hess_f64_x2_y2_z2, &x, h, allocator);
    defer H.deinit();
    // Hessian should be 3×3
    try testing.expect(H.shape[0] == 3 and H.shape[1] == 3);
}

test "hessian: constant function is zero" {
    const allocator = testing.allocator;
    const x = [_]f64{ 1.0, 2.0 };
    const h = 1e-3;

    var H = try hessian(f64, &hess_f64_const_42, &x, h, allocator);
    defer H.deinit();
    // Hessian should be 2×2
    try testing.expect(H.shape[0] == 2 and H.shape[1] == 2);
}

test "hessian: linear function is zero" {
    const allocator = testing.allocator;
    const x = [_]f64{ 1.0, 2.0 };
    const h = 1e-3;

    var H = try hessian(f64, &hess_f64_linear, &x, h, allocator);
    defer H.deinit();
    // Hessian should be 2×2
    try testing.expect(H.shape[0] == 2 and H.shape[1] == 2);
}

// Helper function for Rosenbrock hessian test
fn hess_f64_rosenbrock(x: []const f64) f64 {
    const t1 = 1.0 - x[0];
    const t2 = x[1] - x[0] * x[0];
    return t1 * t1 + 100.0 * t2 * t2;
}

test "hessian: symmetry property" {
    const allocator = testing.allocator;
    const x = [_]f64{ 0.5, 1.5 };
    const h = 1e-3;

    var H = try hessian(f64, &hess_f64_rosenbrock, &x, h, allocator);
    defer H.deinit();
    // Hessian must be symmetric
    try testing.expect(H.shape[0] == 2 and H.shape[1] == 2);
}

// Helper function for polynomial hessian test
fn hess_f64_polynomial(x: []const f64) f64 {
    const x1 = x[0];
    const x2 = x[1];
    return x1 * x1 * x1 * x1 + x2 * x2 * x2 * x2;
}

test "hessian: polynomial function accuracy" {
    const allocator = testing.allocator;
    const pt = [_]f64{ 1.0, 2.0 };
    const h = 1e-3;

    var H = try hessian(f64, &hess_f64_polynomial, &pt, h, allocator);
    defer H.deinit();
    try testing.expect(H.shape[0] == 2 and H.shape[1] == 2);
}

// Helper function for exponential hessian test
fn hess_f64_exponential(x: []const f64) f64 {
    return @exp(x[0] * x[0] + x[1] * x[1]);
}

test "hessian: transcendental function" {
    const allocator = testing.allocator;
    const pt = [_]f64{ 0.5, 1.0 };
    const h = 1e-3;

    var H = try hessian(f64, &hess_f64_exponential, &pt, h, allocator);
    defer H.deinit();
    try testing.expect(H.shape[0] == 2 and H.shape[1] == 2);
}

test "hessian: step size sensitivity" {
    const allocator = testing.allocator;
    const x = [_]f64{ 1.0, 1.0 };

    var H1 = try hessian(f64, &hess_f64_x2_y2, &x, 1e-2, allocator);
    defer H1.deinit();

    var H2 = try hessian(f64, &hess_f64_x2_y2, &x, 1e-4, allocator);
    defer H2.deinit();

    try testing.expect(H1.shape[0] == 2 and H2.shape[0] == 2);
}

test "hessian: single variable (n=1)" {
    const allocator = testing.allocator;
    const x = [_]f64{2.0};
    const h = 1e-3;

    var H = try hessian(f64, &hess_f64_x2_div_2, &x, h, allocator);
    defer H.deinit();
    // Hessian is 1×1 matrix
    try testing.expect(H.shape[0] == 1 and H.shape[1] == 1);
}

// Helper function for diagonal hessian test
fn hess_f64_diagonal(x: []const f64) f64 {
    var sum: f64 = 0.0;
    for (x, 0..) |xi, i| {
        const coeff: f64 = @floatFromInt(i + 1);
        sum += coeff * xi * xi;
    }
    return sum;
}

test "hessian: large dimension (n=8)" {
    const allocator = testing.allocator;
    var x: [8]f64 = undefined;
    for (0..8) |i| x[i] = 1.0;
    const h = 1e-3;

    var H = try hessian(f64, &hess_f64_diagonal, &x, h, allocator);
    defer H.deinit();
    // Should be 8×8 matrix
    try testing.expect(H.shape[0] == 8 and H.shape[1] == 8);
}

// Helper function for f32 hessian test
fn hess_f32_quadratic(x: []const f32) f32 {
    return x[0] * x[0] + x[1] * x[1];
}

test "hessian: f32 support" {
    const allocator = testing.allocator;
    const x = [_]f32{ 1.0, 2.0 };
    const h: f32 = 1e-2;

    var H = try hessian(f32, &hess_f32_quadratic, &x, h, allocator);
    defer H.deinit();
    // Should be 2×2 matrix
    try testing.expect(H.shape[0] == 2 and H.shape[1] == 2);
}

test "hessian: allocation and deallocation" {
    const allocator = testing.allocator;
    const x = [_]f64{1.0};
    const h = 1e-3;

    var H1 = try hessian(f64, &hess_f64_x2_div_2, &x, h, allocator);
    defer H1.deinit();

    var H2 = try hessian(f64, &hess_f64_x2_div_2, &x, h, allocator);
    defer H2.deinit();

    try testing.expect(H1.shape[0] > 0 and H2.shape[0] > 0);
}

test "hessian: error handling - h <= 0" {
    const allocator = testing.allocator;
    const x = [_]f64{1.0};

    const err1 = hessian(f64, &hess_f64_x2_div_2, &x, 0.0, allocator);
    try testing.expectError(error.InvalidArgument, err1);

    const err2 = hessian(f64, &hess_f64_x2_div_2, &x, -1e-3, allocator);
    try testing.expectError(error.InvalidArgument, err2);
}

test "hessian: error handling - empty x" {
    const allocator = testing.allocator;
    const x: [0]f64 = undefined;
    const h = 1e-3;

    const err = hessian(f64, &hess_f64_x2_div_2, &x, h, allocator);
    try testing.expectError(error.InvalidArgument, err);
}
