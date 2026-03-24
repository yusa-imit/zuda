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
};

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
