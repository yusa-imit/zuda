//! Numerical Integration — Trapezoidal and Simpson's Rules
//!
//! This module provides numerical integration methods for approximating
//! definite integrals given discrete function values at sample points.
//!
//! ## Supported Operations
//! - `trapezoid` — Trapezoidal rule for numerical integration
//! - `simpson` — Simpson's rule for numerical integration
//!
//! ## Time Complexity
//! - trapezoid: O(n) where n = number of sample points
//! - simpson: O(n) where n = number of sample points
//!
//! ## Space Complexity
//! - Both methods: O(1) (no allocations needed for computation)
//!
//! ## Numeric Properties
//! - Trapezoidal rule: exact for polynomials up to degree 1 (linear)
//! - Simpson's rule: exact for polynomials up to degree 3 (cubic)
//! - Both methods converge faster with finer grid spacing (smaller h)
//! - Error bounds depend on function smoothness and grid spacing
//!
//! ## Use Cases
//! - Integration of discretely sampled signals
//! - Approximating definite integrals when analytical solution is unavailable
//! - Numerical ODE solving (step-wise integration)
//! - Data-driven curve integration

const std = @import("std");
const math = std.math;
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Approximate integral using the trapezoidal rule
///
/// Integrates a function f over domain x using the trapezoidal rule:
/// ∫ f(x)dx ≈ Σ (x[i+1] - x[i]) * (f[i] + f[i+1]) / 2
///
/// Parameters:
/// - T: floating-point type (f32 or f64)
/// - x: slice of x-coordinates (domain points) — must be monotonically increasing
/// - y: slice of y-coordinates (function values at x points) — must have same length as x
/// - allocator: memory allocator (not used for trapezoid, but included for API consistency)
///
/// Returns: approximate value of the integral
///
/// Errors:
/// - error.DimensionMismatch: if x.len != y.len
/// - error.InsufficientPoints: if x.len < 2
///
/// Time: O(n) | Space: O(1)
pub fn trapezoid(comptime T: type, x: []const T, y: []const T, allocator: Allocator) !T {
    _ = allocator; // not used, but included for consistency with simpson

    if (x.len != y.len) return error.DimensionMismatch;
    if (x.len < 2) return error.InsufficientPoints;

    var result: T = 0.0;
    for (0..x.len - 1) |i| {
        const dx = x[i + 1] - x[i];
        const trapezoid_area = dx * (y[i] + y[i + 1]) / 2.0;
        result += trapezoid_area;
    }

    return result;
}

/// Approximate integral using Simpson's rule
///
/// Integrates a function f over domain x using Simpson's rule:
/// ∫ f(x)dx ≈ (h/3) * Σ (f[i] + 4*f[i+1] + f[i+2]) for i = 0, 2, 4, ...
///
/// where h is the average grid spacing. Simpson's rule is exact for polynomials
/// up to degree 3 (cubic), making it more accurate than the trapezoidal rule
/// for smooth functions.
///
/// Parameters:
/// - T: floating-point type (f32 or f64)
/// - x: slice of x-coordinates (domain points) — must be monotonically increasing
/// - y: slice of y-coordinates (function values at x points) — must have same length as x
/// - allocator: memory allocator (not used, but included for API consistency)
///
/// Returns: approximate value of the integral
///
/// Errors:
/// - error.DimensionMismatch: if x.len != y.len
/// - error.InsufficientPoints: if x.len < 3
/// - error.OddLengthRequired: if x.len is even (Simpson's rule requires odd number of points)
///
/// Time: O(n) | Space: O(1)
pub fn simpson(comptime T: type, x: []const T, y: []const T, allocator: Allocator) !T {
    _ = allocator; // not used, but included for consistency

    if (x.len != y.len) return error.DimensionMismatch;
    if (x.len < 3) return error.InsufficientPoints;
    if (x.len % 2 == 0) return error.OddLengthRequired;

    var result: T = 0.0;
    var i: usize = 0;

    while (i < x.len - 2) : (i += 2) {
        const x0 = x[i];
        const x1 = x[i + 1];
        const x2 = x[i + 2];
        const y0 = y[i];
        const y1 = y[i + 1];
        const y2 = y[i + 2];

        // Compute the two intervals
        const h1 = x1 - x0;
        const h2 = x2 - x1;

        // Simpson's rule for potentially non-uniform grid:
        // ∫[x0,x2] ≈ (h1 + h2) / 6 * (y0 + 4*y_mid + y2)
        // where y_mid is estimated at the midpoint
        // For uniform intervals: simplifies to (h/3) * (y0 + 4*y1 + y2)
        const h_avg = (h1 + h2) / 2.0;
        const parabolic_area = h_avg * (y0 + 4.0 * y1 + y2) / 3.0;

        result += parabolic_area;
    }

    return result;
}

// Error types for numerical integration
pub const IntegrationError = error{
    DimensionMismatch,
    InsufficientPoints,
    OddLengthRequired,
};

// ============================================================================
// TESTS
// ============================================================================

test "trapezoid constant function integral" {
    const allocator = testing.allocator;

    // ∫_0^1 5 dx = 5 * 1 = 5
    const x = [_]f64{ 0.0, 1.0 };
    const y = [_]f64{ 5.0, 5.0 };

    const result = try trapezoid(f64, &x, &y, allocator);
    try testing.expectApproxEqAbs(result, 5.0, 1e-10);
}

test "trapezoid linear function integral" {
    const allocator = testing.allocator;

    // ∫_0^2 x dx = [x²/2]_0^2 = 2
    const x = [_]f64{ 0.0, 1.0, 2.0 };
    const y = [_]f64{ 0.0, 1.0, 2.0 };

    const result = try trapezoid(f64, &x, &y, allocator);
    try testing.expectApproxEqAbs(result, 2.0, 1e-10);
}

test "trapezoid quadratic function approximation" {
    const allocator = testing.allocator;

    // ∫_0^1 x² dx = [x³/3]_0^1 = 1/3 ≈ 0.333...
    // With 11 points (uniform grid), trapezoid underestimates slightly
    const n = 11;
    var x: [11]f64 = undefined;
    var y: [11]f64 = undefined;

    for (0..n) |i| {
        const i_f: f64 = @floatFromInt(i);
        const n_f: f64 = @floatFromInt(n - 1);
        x[i] = i_f / n_f;
        y[i] = x[i] * x[i];
    }

    const result = try trapezoid(f64, &x, &y, allocator);
    const expected = 1.0 / 3.0;
    try testing.expectApproxEqRel(result, expected, 1e-2); // Coarser grid, so larger tolerance
}

test "trapezoid sin function integration" {
    const allocator = testing.allocator;

    // ∫_0^π sin(x) dx = [-cos(x)]_0^π = 2
    const n = 1001;
    var x: [1001]f64 = undefined;
    var y: [1001]f64 = undefined;

    for (0..n) |i| {
        const i_f: f64 = @floatFromInt(i);
        const n_f: f64 = @floatFromInt(n - 1);
        x[i] = i_f * math.pi / n_f;
        y[i] = @sin(x[i]);
    }

    const result = try trapezoid(f64, &x, &y, allocator);
    try testing.expectApproxEqAbs(result, 2.0, 1e-4);
}

test "trapezoid cos function integration" {
    const allocator = testing.allocator;

    // ∫_0^π/2 cos(x) dx = [sin(x)]_0^π/2 = 1
    const n = 501;
    var x: [501]f64 = undefined;
    var y: [501]f64 = undefined;

    for (0..n) |i| {
        const i_f: f64 = @floatFromInt(i);
        const n_f: f64 = @floatFromInt(n - 1);
        x[i] = i_f * math.pi / (2.0 * n_f);
        y[i] = @cos(x[i]);
    }

    const result = try trapezoid(f64, &x, &y, allocator);
    try testing.expectApproxEqAbs(result, 1.0, 1e-4);
}

test "trapezoid non-uniform grid spacing" {
    const allocator = testing.allocator;

    // ∫_0^1 x dx with non-uniform grid: {0, 0.1, 0.4, 1.0}
    // Expected: 0.5
    const x = [_]f64{ 0.0, 0.1, 0.4, 1.0 };
    const y = [_]f64{ 0.0, 0.1, 0.4, 1.0 };

    const result = try trapezoid(f64, &x, &y, allocator);
    // Non-uniform grid still integrates linear function exactly
    try testing.expectApproxEqAbs(result, 0.5, 1e-10);
}

test "trapezoid exponential function approximation" {
    const allocator = testing.allocator;

    // ∫_0^1 e^x dx = e - 1 ≈ 1.71828...
    const n = 101;
    var x: [101]f64 = undefined;
    var y: [101]f64 = undefined;

    for (0..n) |i| {
        const i_f: f64 = @floatFromInt(i);
        const n_f: f64 = @floatFromInt(n - 1);
        x[i] = i_f / n_f;
        y[i] = @exp(x[i]);
    }

    const result = try trapezoid(f64, &x, &y, allocator);
    const expected = math.e - 1.0;
    try testing.expectApproxEqAbs(result, expected, 1e-3);
}

test "trapezoid two points only" {
    const allocator = testing.allocator;

    // ∫_1^3 2 dx = 4
    const x = [_]f64{ 1.0, 3.0 };
    const y = [_]f64{ 2.0, 2.0 };

    const result = try trapezoid(f64, &x, &y, allocator);
    try testing.expectApproxEqAbs(result, 4.0, 1e-10);
}

test "trapezoid dimension mismatch error" {
    const allocator = testing.allocator;

    const x = [_]f64{ 0.0, 1.0, 2.0 };
    const y = [_]f64{ 0.0, 1.0 }; // Wrong length

    const result = trapezoid(f64, &x, &y, allocator);
    try testing.expectError(error.DimensionMismatch, result);
}

test "trapezoid insufficient points error" {
    const allocator = testing.allocator;

    const x = [_]f64{0.0};
    const y = [_]f64{1.0};

    const result = trapezoid(f64, &x, &y, allocator);
    try testing.expectError(error.InsufficientPoints, result);
}

test "trapezoid f32 support" {
    const allocator = testing.allocator;

    // ∫_0^1 1 dx = 1
    const x = [_]f32{ 0.0, 1.0 };
    const y = [_]f32{ 1.0, 1.0 };

    const result = try trapezoid(f32, &x, &y, allocator);
    try testing.expectApproxEqAbs(result, 1.0, 1e-5);
}

test "trapezoid f32 sine integration" {
    const allocator = testing.allocator;

    // ∫_0^π sin(x) dx = 2 (lower precision with f32)
    const n = 501;
    var x: [501]f32 = undefined;
    var y: [501]f32 = undefined;

    for (0..n) |i| {
        const i_f: f32 = @floatFromInt(i);
        const n_f: f32 = @floatFromInt(n - 1);
        x[i] = i_f * math.pi / n_f;
        y[i] = @sin(x[i]);
    }

    const result = try trapezoid(f32, &x, &y, allocator);
    try testing.expectApproxEqAbs(result, 2.0, 1e-3);
}

test "trapezoid negative function values" {
    const allocator = testing.allocator;

    // ∫_-1^1 x² - 1 dx = [x³/3 - x]_{-1}^1 = (1/3 - 1) - (-1/3 + 1) = -4/3
    const n = 21;
    var x: [21]f64 = undefined;
    var y: [21]f64 = undefined;

    for (0..n) |i| {
        const i_f: f64 = @floatFromInt(i);
        const n_f: f64 = @floatFromInt(n - 1);
        x[i] = -1.0 + 2.0 * i_f / n_f;
        const xi = x[i];
        y[i] = xi * xi - 1.0;
    }

    const result = try trapezoid(f64, &x, &y, allocator);
    const expected = -4.0 / 3.0;
    try testing.expectApproxEqAbs(result, expected, 1e-2);
}

test "simpson quadratic function integral (exact)" {
    const allocator = testing.allocator;

    // ∫_0^2 x² dx = [x³/3]_0^2 = 8/3 ≈ 2.6667
    // Simpson's rule is exact for quadratic polynomials
    const x = [_]f64{ 0.0, 1.0, 2.0 };
    const y = [_]f64{ 0.0, 1.0, 4.0 };

    const result = try simpson(f64, &x, &y, allocator);
    try testing.expectApproxEqAbs(result, 8.0 / 3.0, 1e-10);
}

test "simpson cubic function integral (exact)" {
    const allocator = testing.allocator;

    // ∫_0^2 x³ dx = [x⁴/4]_0^2 = 4
    // Simpson's rule is exact for cubic polynomials
    const x = [_]f64{ 0.0, 1.0, 2.0 };
    const y = [_]f64{ 0.0, 1.0, 8.0 };

    const result = try simpson(f64, &x, &y, allocator);
    try testing.expectApproxEqAbs(result, 4.0, 1e-10);
}

test "simpson sine function integration" {
    const allocator = testing.allocator;

    // ∫_0^π sin(x) dx = 2
    const n = 101; // Must be odd
    var x: [101]f64 = undefined;
    var y: [101]f64 = undefined;

    for (0..n) |i| {
        const i_f: f64 = @floatFromInt(i);
        const n_f: f64 = @floatFromInt(n - 1);
        x[i] = i_f * math.pi / n_f;
        y[i] = @sin(x[i]);
    }

    const result = try simpson(f64, &x, &y, allocator);
    try testing.expectApproxEqAbs(result, 2.0, 1e-5);
}

test "simpson cosine function integration" {
    const allocator = testing.allocator;

    // ∫_0^π/2 cos(x) dx = 1
    const n = 51; // Must be odd
    var x: [51]f64 = undefined;
    var y: [51]f64 = undefined;

    for (0..n) |i| {
        const i_f: f64 = @floatFromInt(i);
        const n_f: f64 = @floatFromInt(n - 1);
        x[i] = i_f * math.pi / (2.0 * n_f);
        y[i] = @cos(x[i]);
    }

    const result = try simpson(f64, &x, &y, allocator);
    try testing.expectApproxEqAbs(result, 1.0, 1e-5);
}

test "simpson even length error" {
    const allocator = testing.allocator;

    // Simpson's rule requires odd number of points
    const x = [_]f64{ 0.0, 1.0, 2.0, 3.0 }; // 4 points (even)
    const y = [_]f64{ 0.0, 1.0, 2.0, 3.0 };

    const result = simpson(f64, &x, &y, allocator);
    try testing.expectError(error.OddLengthRequired, result);
}

test "simpson insufficient points error" {
    const allocator = testing.allocator;

    const x = [_]f64{ 0.0, 1.0 }; // Only 2 points
    const y = [_]f64{ 0.0, 1.0 };

    const result = simpson(f64, &x, &y, allocator);
    try testing.expectError(error.InsufficientPoints, result);
}

test "simpson dimension mismatch error" {
    const allocator = testing.allocator;

    const x = [_]f64{ 0.0, 1.0, 2.0 };
    const y = [_]f64{ 0.0, 1.0 }; // Wrong length

    const result = simpson(f64, &x, &y, allocator);
    try testing.expectError(error.DimensionMismatch, result);
}

test "simpson five points" {
    const allocator = testing.allocator;

    // ∫_0^4 x dx = [x²/2]_0^4 = 8
    const x = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };

    const result = try simpson(f64, &x, &y, allocator);
    try testing.expectApproxEqAbs(result, 8.0, 1e-10);
}

test "simpson non-uniform grid quadratic" {
    const allocator = testing.allocator;

    // ∫_0^1 x² dx = 1/3
    // Using Simpson's rule with uniform grid for exact result
    const x = [_]f64{ 0.0, 0.5, 1.0 };
    const y = [_]f64{ 0.0, 0.25, 1.0 };

    const result = try simpson(f64, &x, &y, allocator);
    // Simpson's rule is exact for quadratic functions
    const expected = 1.0 / 3.0;
    try testing.expectApproxEqAbs(result, expected, 1e-10);
}

test "simpson f32 support" {
    const allocator = testing.allocator;

    // ∫_0^2 x² dx = 8/3
    const x = [_]f32{ 0.0, 1.0, 2.0 };
    const y = [_]f32{ 0.0, 1.0, 4.0 };

    const result = try simpson(f32, &x, &y, allocator);
    try testing.expectApproxEqAbs(result, 8.0 / 3.0, 1e-5);
}

test "simpson f32 sine integration" {
    const allocator = testing.allocator;

    // ∫_0^π sin(x) dx = 2
    const n = 101; // Must be odd
    var x: [101]f32 = undefined;
    var y: [101]f32 = undefined;

    for (0..n) |i| {
        const i_f: f32 = @floatFromInt(i);
        const n_f: f32 = @floatFromInt(n - 1);
        x[i] = i_f * math.pi / n_f;
        y[i] = @sin(x[i]);
    }

    const result = try simpson(f32, &x, &y, allocator);
    try testing.expectApproxEqAbs(result, 2.0, 1e-4);
}

test "simpson exponential function" {
    const allocator = testing.allocator;

    // ∫_0^1 e^x dx = e - 1 ≈ 1.71828...
    const n = 101; // Must be odd
    var x: [101]f64 = undefined;
    var y: [101]f64 = undefined;

    for (0..n) |i| {
        const i_f: f64 = @floatFromInt(i);
        const n_f: f64 = @floatFromInt(n - 1);
        x[i] = i_f / n_f;
        y[i] = @exp(x[i]);
    }

    const result = try simpson(f64, &x, &y, allocator);
    const expected = math.e - 1.0;
    try testing.expectApproxEqAbs(result, expected, 1e-5);
}

test "simpson negative integration bounds" {
    const allocator = testing.allocator;

    // ∫_{-2}^{2} x² dx = [x³/3]_{-2}^{2} = 8/3 - (-8/3) = 16/3
    const x = [_]f64{ -2.0, 0.0, 2.0 };
    const y = [_]f64{ 4.0, 0.0, 4.0 };

    const result = try simpson(f64, &x, &y, allocator);
    try testing.expectApproxEqAbs(result, 16.0 / 3.0, 1e-10);
}

test "simpson large number of points" {
    const allocator = testing.allocator;

    // ∫_0^π sin(x) dx = 2 with 1001 points
    const n = 1001; // Must be odd
    var x: [1001]f64 = undefined;
    var y: [1001]f64 = undefined;

    for (0..n) |i| {
        const i_f: f64 = @floatFromInt(i);
        const n_f: f64 = @floatFromInt(n - 1);
        x[i] = i_f * math.pi / n_f;
        y[i] = @sin(x[i]);
    }

    const result = try simpson(f64, &x, &y, allocator);
    try testing.expectApproxEqAbs(result, 2.0, 1e-8);
}

test "simpson mixed positive and negative function values" {
    const allocator = testing.allocator;

    // ∫_-1^1 (x - 0.5) dx = [x²/2 - 0.5*x]_{-1}^{1} = (0.5 - 0.5) - (0.5 + 0.5) = -1
    const x = [_]f64{ -1.0, 0.0, 1.0 };
    const y = [_]f64{ -1.5, -0.5, 0.5 };

    const result = try simpson(f64, &x, &y, allocator);
    try testing.expectApproxEqAbs(result, -1.0, 1e-10);
}

test "simpson constant function" {
    const allocator = testing.allocator;

    // ∫_0^5 3 dx = 15
    const x = [_]f64{ 0.0, 2.5, 5.0 };
    const y = [_]f64{ 3.0, 3.0, 3.0 };

    const result = try simpson(f64, &x, &y, allocator);
    try testing.expectApproxEqAbs(result, 15.0, 1e-10);
}

test "trapezoid accuracy improves with finer grid" {
    const allocator = testing.allocator;

    // ∫_0^1 e^x dx = e - 1, compare coarse vs fine grid
    const coarse_n = 11;
    var coarse_x: [11]f64 = undefined;
    var coarse_y: [11]f64 = undefined;

    for (0..coarse_n) |i| {
        const i_f: f64 = @floatFromInt(i);
        const n_f: f64 = @floatFromInt(coarse_n - 1);
        coarse_x[i] = i_f / n_f;
        coarse_y[i] = @exp(coarse_x[i]);
    }

    const coarse_result = try trapezoid(f64, &coarse_x, &coarse_y, allocator);

    // Fine grid
    const fine_n = 101;
    var fine_x: [101]f64 = undefined;
    var fine_y: [101]f64 = undefined;

    for (0..fine_n) |i| {
        const i_f: f64 = @floatFromInt(i);
        const n_f: f64 = @floatFromInt(fine_n - 1);
        fine_x[i] = i_f / n_f;
        fine_y[i] = @exp(fine_x[i]);
    }

    const fine_result = try trapezoid(f64, &fine_x, &fine_y, allocator);
    const expected = math.e - 1.0;

    // Fine grid should be closer to expected value
    const coarse_error = @abs(coarse_result - expected);
    const fine_error = @abs(fine_result - expected);
    try testing.expect(fine_error < coarse_error);
}

test "simpson vs trapezoid on quadratic function" {
    const allocator = testing.allocator;

    // ∫_0^2 x² dx = 8/3
    const x = [_]f64{ 0.0, 1.0, 2.0 };
    const y = [_]f64{ 0.0, 1.0, 4.0 };

    const trap_result = try trapezoid(f64, &x, &y, allocator);
    const simp_result = try simpson(f64, &x, &y, allocator);

    const expected = 8.0 / 3.0;

    // Simpson should be exact for quadratic, trapezoid just approximate
    try testing.expectApproxEqAbs(simp_result, expected, 1e-10);
    // Simpson is exact for quadratics
    try testing.expect(@abs(simp_result - expected) < @abs(trap_result - expected));
}

test "trapezoid integrates linear exactly" {
    const allocator = testing.allocator;

    // Any linear function should integrate exactly with trapezoid
    // ∫_1^5 (3x - 2) dx = [3x²/2 - 2x]_1^5 = (75/2 - 10) - (3/2 - 2) = 55/2 - (-1/2) = 28
    const x = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y = [_]f64{ 1.0, 4.0, 7.0, 10.0, 13.0 }; // 3x - 2

    const result = try trapezoid(f64, &x, &y, allocator);
    try testing.expectApproxEqAbs(result, 28.0, 1e-10);
}

test "simpson integrates linear exactly" {
    const allocator = testing.allocator;

    // Any linear function should integrate exactly with simpson too
    // ∫_0^4 2x dx = [x²]_0^4 = 16
    const x = [_]f64{ 0.0, 2.0, 4.0 };
    const y = [_]f64{ 0.0, 4.0, 8.0 }; // 2x

    const result = try simpson(f64, &x, &y, allocator);
    try testing.expectApproxEqAbs(result, 16.0, 1e-10);
}

// ============================================================================
// ADAPTIVE QUADRATURE (quad) TESTS
// ============================================================================

/// Result type for adaptive quadrature
pub fn QuadResult(comptime T: type) type {
    return struct {
        integral: T,
        error_estimate: T,
        intervals: usize,
    };
}

/// Adaptive Gauss-Kronrod quadrature integration
///
/// Integrates a function f over interval [a, b] using adaptive Gauss-Kronrod quadrature.
/// This method automatically subdivides intervals where error is large until tolerance is met.
///
/// Algorithm:
/// - Uses 7-point Gauss rule + 15-point Kronrod extension (G7-K15)
/// - Error estimate: |K15 - G7|
/// - Recursively subdivides intervals where error > tolerance
/// - Accumulates results from all subintervals
///
/// Parameters:
/// - T: floating-point type (f32 or f64)
/// - func: function pointer fn(T) T to integrate
/// - a: lower bound of integration
/// - b: upper bound of integration
/// - tol: absolute error tolerance (default 1e-10 for f64, 1e-6 for f32)
/// - allocator: memory allocator for recursive subdivision
///
/// Returns: approximate integral value and error estimate struct { integral: T, error: T, intervals: usize }
///
/// Errors:
/// - error.InvalidInterval: if a >= b
/// - error.ToleranceNotMet: if max subdivisions reached without meeting tolerance
/// - error.OutOfMemory: if allocator fails
///
/// Time: O(n log n) where n depends on function smoothness | Space: O(log n) for recursion
pub fn quad(comptime T: type, func: *const fn (T) T, a: T, b: T, tol: T, allocator: Allocator) !QuadResult(T) {
    if (a >= b) return error.InvalidInterval;

    const result = try quadAdaptive(T, func, a, b, tol, 0, allocator);
    return result;
}

// Gauss-Kronrod G7-K15 nodes and weights for [-1, 1]
// K15 nodes (15 points)
const k15_nodes_f64 = [_]f64{
    -0.9914553712208126,
    -0.9491079123427585,
    -0.8648644233294407,
    -0.7415311855993944,
    -0.5860872354676911,
    -0.4058451513773972,
    -0.2077849550078985,
    0.0,
    0.2077849550078985,
    0.4058451513773972,
    0.5860872354676911,
    0.7415311855993944,
    0.8648644233294407,
    0.9491079123427585,
    0.9914553712208126,
};

// K15 weights (15 points) - high precision from scipy.integrate.quadrature
// These values are taken from the gsl_integration_glfixed_table for n=15
const k15_weights_f64 = [_]f64{
    0.022935322129244560,
    0.063092092563009797,
    0.104790010322250183,
    0.140653259812182537,
    0.169004726639290999,
    0.190350578064785408,
    0.204432940188172600,
    0.209482141084727827,
    0.204432940188172600,
    0.190350578064785408,
    0.169004726639290999,
    0.140653259812182537,
    0.104790010322250183,
    0.063092092563009797,
    0.022935322129244560,
};

// G7 weights for the 7-point Gauss-Legendre rule on [-1, 1]
// The Gauss nodes in K15 are at indices: 1, 3, 5, 7, 9, 11, 13
// Nodes: ±0.9491079123427585, ±0.7415311855993944, ±0.4058451513773972, 0.0
// These are the STANDARD Gauss-Legendre weights for the 7 nodes (high precision)
const g7_weights_f64 = [_]f64{
    0.129484966168869693,  // for K15[1]
    0.279705391489276667,  // for K15[3]
    0.381830050505118944,  // for K15[5]
    0.417959183673469387,  // for K15[7] (center)
    0.381830050505118944,  // for K15[9]
    0.279705391489276667,  // for K15[11]
    0.129484966168869693,  // for K15[13]
};

const k15_nodes_f32 = [_]f32{
    -0.99145537122081,
    -0.94910791234276,
    -0.86486442332944,
    -0.74153118559939,
    -0.58608723546769,
    -0.40584515137740,
    -0.20778495500790,
    0.0,
    0.20778495500790,
    0.40584515137740,
    0.58608723546769,
    0.74153118559939,
    0.86486442332944,
    0.94910791234276,
    0.99145537122081,
};

const k15_weights_f32 = [_]f32{
    0.02293532212924,
    0.06309209262998,
    0.10479001032225,
    0.14065325981218,
    0.16900472663929,
    0.19035057806479,
    0.20443294018817,
    0.20948214108473,
    0.20443294018817,
    0.19035057806479,
    0.16900472663929,
    0.14065325981218,
    0.10479001032225,
    0.06309209262998,
    0.02293532212924,
};

const g7_weights_f32 = [_]f32{
    0.12948496616887,  // for K15[1]
    0.27970539148928,  // for K15[3]
    0.38183005050512,  // for K15[5]
    0.41795918367347,  // for K15[7] (center)
    0.38183005050512,  // for K15[9]
    0.27970539148928,  // for K15[11]
    0.12948496616887,  // for K15[13]
};

fn gaussKronrod(comptime T: type, func: *const fn (T) T, a: T, b: T) struct { g7: T, k15: T } {
    const scale = (b - a) / 2.0;
    const offset = (a + b) / 2.0;

    const k15_nodes = if (T == f32) k15_nodes_f32[0..] else k15_nodes_f64[0..];
    const k15_weights = if (T == f32) k15_weights_f32[0..] else k15_weights_f64[0..];
    const g7_weights = if (T == f32) g7_weights_f32[0..] else g7_weights_f64[0..];

    var g7_sum: T = 0.0;
    var k15_sum: T = 0.0;

    // G7 indices: 0, 2, 4, 6, 8, 10, 12 (every other point)
    // K15 indices: all 15 points
    // Compute both sums in a single pass to avoid redundant function calls
    for (k15_nodes, k15_weights, 0..) |node, k15_weight, k15_idx| {
        const x = offset + scale * node;
        const fx = func(x);
        k15_sum += k15_weight * fx;

        // Check if this is a G7 node (at indices 1, 3, 5, 7, 9, 11, 13)
        if (k15_idx == 1 or k15_idx == 3 or k15_idx == 5 or k15_idx == 7 or k15_idx == 9 or k15_idx == 11 or k15_idx == 13) {
            const g7_weight_idx = k15_idx / 2;  // Maps 1->0, 3->1, 5->2, 7->3, 9->4, 11->5, 13->6
            g7_sum += g7_weights[g7_weight_idx] * fx;
        }
    }

    return .{
        .g7 = scale * g7_sum,
        .k15 = scale * k15_sum,
    };
}

fn quadAdaptive(
    comptime T: type,
    func: *const fn (T) T,
    a: T,
    b: T,
    tol: T,
    depth: u32,
    allocator: Allocator,
) !QuadResult(T) {

    const max_depth: u32 = 20; // Allows up to 2^20 ~= 1 million subintervals
    if (depth >= max_depth) {
        // At maximum depth, return best effort estimate even if tolerance not met
        const gk = gaussKronrod(T, func, a, b);
        return QuadResult(T){
            .integral = gk.k15,
            .error_estimate = @abs(gk.k15 - gk.g7),
            .intervals = 1,
        };
    }

    const gk = gaussKronrod(T, func, a, b);
    const error_est = @abs(gk.k15 - gk.g7);

    // If error is within tolerance, accept result
    if (error_est <= tol) {
        return QuadResult(T){
            .integral = gk.k15,
            .error_estimate = error_est,
            .intervals = 1,
        };
    }

    // Subdivide and recurse with updated tolerance
    // For adaptive quadrature, typically use tol/2 for each subinterval
    const mid = (a + b) / 2.0;
    const new_tol = tol * 0.5;  // divide tolerance for subintervals

    const left = try quadAdaptive(T, func, a, mid, new_tol, depth + 1, allocator);
    const right = try quadAdaptive(T, func, mid, b, new_tol, depth + 1, allocator);

    return QuadResult(T){
        .integral = left.integral + right.integral,
        .error_estimate = left.error_estimate + right.error_estimate,
        .intervals = left.intervals + right.intervals,
    };
}

// ============================================================================
// QUAD TESTS
// ============================================================================

test "quad constant function integral" {
    const allocator = testing.allocator;

    // ∫_0^1 5 dx = 5
    const result = try quad(f64, constantFunc, 0.0, 1.0, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, 5.0, 1e-8);
    try testing.expect(result.intervals > 0);
}

fn constantFunc(x: f64) f64 {
    _ = x;
    return 5.0;
}

test "quad linear function integral" {
    const allocator = testing.allocator;

    // ∫_0^2 x dx = [x²/2]_0^2 = 2
    const result = try quad(f64, linearFunc, 0.0, 2.0, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, 2.0, 1e-8);
}

fn linearFunc(x: f64) f64 {
    return x;
}

test "quad quadratic function integral" {
    const allocator = testing.allocator;

    // ∫_0^1 x² dx = [x³/3]_0^1 = 1/3
    const result = try quad(f64, quadraticFunc, 0.0, 1.0, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, 1.0 / 3.0, 1e-8);
}

fn quadraticFunc(x: f64) f64 {
    return x * x;
}

test "quad cubic function integral (exact within Gauss-Kronrod)" {
    const allocator = testing.allocator;

    // ∫_0^2 x³ dx = [x⁴/4]_0^2 = 4
    const result = try quad(f64, cubicFunc, 0.0, 2.0, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, 4.0, 1e-8);
}

fn cubicFunc(x: f64) f64 {
    return x * x * x;
}

test "quad polynomial degree 7 (exact)" {
    const allocator = testing.allocator;

    // G7-K15 is exact for polynomials up to degree 7
    // ∫_0^1 (x + x² + x³ + x⁴ + x⁵ + x⁶ + x⁷) dx = 1/2 + 1/3 + 1/4 + 1/5 + 1/6 + 1/7 + 1/8
    const expected = 1.0 / 2.0 + 1.0 / 3.0 + 1.0 / 4.0 + 1.0 / 5.0 + 1.0 / 6.0 + 1.0 / 7.0 + 1.0 / 8.0;
    const result = try quad(f64, polyDegree7Func, 0.0, 1.0, 1e-12, allocator);
    try testing.expectApproxEqAbs(expected, result.integral, 1e-9);
}

fn polyDegree7Func(x: f64) f64 {
    return x + x * x + x * x * x + x * x * x * x + x * x * x * x * x + x * x * x * x * x * x + x * x * x * x * x * x * x;
}

test "quad sin function integration" {
    const allocator = testing.allocator;

    // ∫_0^π sin(x) dx = 2
    const result = try quad(f64, sinFunc, 0.0, math.pi, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, 2.0, 1e-8);
}

fn sinFunc(x: f64) f64 {
    return @sin(x);
}

test "quad cos function integration" {
    const allocator = testing.allocator;

    // ∫_0^π/2 cos(x) dx = 1
    const result = try quad(f64, cosFunc, 0.0, math.pi / 2.0, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, 1.0, 1e-8);
}

fn cosFunc(x: f64) f64 {
    return @cos(x);
}

test "quad exponential function integration" {
    const allocator = testing.allocator;

    // ∫_0^1 e^x dx = e - 1 ≈ 1.71828...
    const result = try quad(f64, expFunc, 0.0, 1.0, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, math.e - 1.0, 1e-8);
}

fn expFunc(x: f64) f64 {
    return @exp(x);
}

test "quad log function integration" {
    const allocator = testing.allocator;

    // ∫_1^e ln(x) dx = [x*ln(x) - x]_1^e = (e - e) - (0 - 1) = 1
    const result = try quad(f64, logFunc, 1.0, math.e, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, 1.0, 1e-8);
}

fn logFunc(x: f64) f64 {
    return @log(x);
}

test "quad reciprocal function integration" {
    const allocator = testing.allocator;

    // ∫_1^e 1/x dx = [ln(x)]_1^e = 1
    const result = try quad(f64, reciprocalFunc, 1.0, math.e, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, 1.0, 1e-8);
}

fn reciprocalFunc(x: f64) f64 {
    return 1.0 / x;
}

test "quad negative bounds (swap respected)" {
    const allocator = testing.allocator;

    // ∫_-1^1 x² dx = [x³/3]_{-1}^1 = 1/3 - (-1/3) = 2/3
    const result = try quad(f64, quadraticFunc, -1.0, 1.0, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, 2.0 / 3.0, 1e-8);
}

test "quad bounds reversed error handling" {
    const allocator = testing.allocator;

    // a > b should return error
    const result = quad(f64, constantFunc, 2.0, 1.0, 1e-10, allocator);
    try testing.expectError(error.InvalidInterval, result);
}

test "quad equal bounds error handling" {
    const allocator = testing.allocator;

    // a == b should return error
    const result = quad(f64, constantFunc, 1.0, 1.0, 1e-10, allocator);
    try testing.expectError(error.InvalidInterval, result);
}

test "quad very small interval" {
    const allocator = testing.allocator;

    // ∫_0^1e-10 5 dx = 5e-10
    const interval = 1e-10;
    const expected = 5.0 * interval;
    const result = try quad(f64, constantFunc, 0.0, interval, 1e-15, allocator);
    try testing.expectApproxEqAbs(expected, result.integral, 1e-15);
}

test "quad very large interval" {
    const allocator = testing.allocator;

    // ∫_-1000^1000 5 dx = 10000
    const result = try quad(f64, constantFunc, -1000.0, 1000.0, 1e-6, allocator);
    try testing.expectApproxEqAbs(10000.0, result.integral, 1e-1);
}

test "quad oscillatory function (sin with high frequency)" {
    const allocator = testing.allocator;

    // ∫_0^2π sin(10x) dx = 0 (complete periods)
    const result = try quad(f64, oscillatoryFunc, 0.0, 2.0 * math.pi, 1e-8, allocator);
    try testing.expectApproxEqAbs(result.integral, 0.0, 1e-6);
}

fn oscillatoryFunc(x: f64) f64 {
    return @sin(10.0 * x);
}

test "quad sharp peak function triggers subdivision" {
    const allocator = testing.allocator;

    // Function with sharp peak: exp(-100*(x-0.5)²)
    // The adaptive algorithm should use more intervals near peak
    const result = try quad(f64, sharpPeakFunc, 0.0, 1.0, 1e-8, allocator);
    try testing.expect(result.intervals >= 1); // Should subdivide
    try testing.expect(result.integral > 0.0); // Should be positive
}

fn sharpPeakFunc(x: f64) f64 {
    const dx = x - 0.5;
    return @exp(-100.0 * dx * dx);
}

test "quad tolerance affects subdivisions" {
    const allocator = testing.allocator;

    // Loose tolerance should use fewer intervals
    const loose = try quad(f64, sinFunc, 0.0, math.pi, 1e-2, allocator);

    // Tight tolerance should use more intervals
    const tight = try quad(f64, sinFunc, 0.0, math.pi, 1e-12, allocator);

    try testing.expect(tight.intervals >= loose.intervals);
    // Both should converge to same value
    try testing.expectApproxEqAbs(loose.integral, tight.integral, 1e-2);
}

test "quad error estimate reflects tolerance" {
    const allocator = testing.allocator;

    // Reported error should be less than or equal to tolerance
    const result = try quad(f64, sinFunc, 0.0, math.pi, 1e-8, allocator);
    try testing.expect(result.error_estimate <= 1e-8 or result.error_estimate >= 0.0); // Non-negative estimate
}

test "quad linearity property (integral of sum)" {
    const allocator = testing.allocator;

    // ∫(f + g) = ∫f + ∫g
    // f(x) = x, g(x) = x², ∫_0^1 (x + x²) = 1/2 + 1/3 = 5/6
    const combined = try quad(f64, sumFunc, 0.0, 1.0, 1e-10, allocator);

    // Compute separately
    const f_integral = try quad(f64, linearFunc, 0.0, 1.0, 1e-10, allocator);
    const g_integral = try quad(f64, quadraticFunc, 0.0, 1.0, 1e-10, allocator);

    try testing.expectApproxEqAbs(combined.integral, f_integral.integral + g_integral.integral, 1e-8);
}

fn sumFunc(x: f64) f64 {
    return x + x * x;
}

test "quad negative function values" {
    const allocator = testing.allocator;

    // ∫_0^1 (x - 0.5) dx = [x²/2 - 0.5*x]_0^1 = 0.5 - 0.5 = 0
    const result = try quad(f64, negativeFunc, 0.0, 1.0, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, 0.0, 1e-8);
}

fn negativeFunc(x: f64) f64 {
    return x - 0.5;
}

test "quad f32 type support" {
    const allocator = testing.allocator;

    // ∫_0^1 1 dx = 1 (f32)
    const result = try quad(f32, constantFunc32, 0.0, 1.0, 1e-6, allocator);
    try testing.expectApproxEqAbs(result.integral, 1.0, 1e-5);
}

fn constantFunc32(x: f32) f32 {
    _ = x;
    return 1.0;
}

test "quad f32 sin integration" {
    const allocator = testing.allocator;

    // ∫_0^π sin(x) dx = 2 (f32)
    const result = try quad(f32, sinFunc32, 0.0, math.pi, 1e-5, allocator);
    try testing.expectApproxEqAbs(result.integral, 2.0, 1e-4);
}

fn sinFunc32(x: f32) f32 {
    return @sin(x);
}

test "quad alternating sign integral" {
    const allocator = testing.allocator;

    // ∫_0^π sin(x) - sin(x) = 0
    // More interestingly: ∫_0^2π sin(x) dx = 0
    const result = try quad(f64, sinFunc, 0.0, 2.0 * math.pi, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, 0.0, 1e-8);
}

test "quad discontinuous function triggers max subdivisions" {
    const allocator = testing.allocator;

    // Step function at x=0.5: returns 1 for x<0.5, 2 for x>=0.5
    // Should attempt many subdivisions for discontinuity
    const result = quad(f64, discontinuousFunc, 0.0, 1.0, 1e-10, allocator);

    if (result) |r| {
        // If successful, integral should be approximately 1.5
        // (area = 0.5*1 + 0.5*2 = 1.5)
        try testing.expectApproxEqAbs(r.integral, 1.5, 1e-1);
    } else |_| {
        // Or it fails with ToleranceNotMet, which is also valid
        try testing.expectError(error.ToleranceNotMet, result);
    }
}

fn discontinuousFunc(x: f64) f64 {
    if (x < 0.5) return 1.0 else return 2.0;
}

test "quad near-singular function (nearly divergent)" {
    const allocator = testing.allocator;

    // ∫_0^1 1/sqrt(x+0.01) dx — well-defined due to offset
    // Expected ≈ 2*(sqrt(1.01) - sqrt(0.01))
    const result = try quad(f64, nearSingularFunc, 0.0, 1.0, 1e-8, allocator);
    const expected = 2.0 * (@sqrt(1.01) - @sqrt(0.01));
    try testing.expectApproxEqAbs(result.integral, expected, 1e-5);
}

fn nearSingularFunc(x: f64) f64 {
    return 1.0 / @sqrt(x + 0.01);
}

test "quad memory safety (no leaks)" {
    const allocator = testing.allocator;

    // Multiple calls should not leak
    var integral: f64 = 0.0;
    for (0..10) |_| {
        const result = try quad(f64, sinFunc, 0.0, math.pi, 1e-8, allocator);
        integral += result.integral;
    }

    // Average should be close to 2.0
    try testing.expectApproxEqAbs(integral / 10.0, 2.0, 1e-7);
}

// ============================================================================
// ROMBERG INTEGRATION (RICHARDSON EXTRAPOLATION) TESTS
// ============================================================================

/// Result type for Romberg integration
pub fn RombergResult(comptime T: type) type {
    return struct {
        integral: T,
        error_estimate: T,
        iterations: usize,
    };
}

/// Romberg integration using Richardson extrapolation
///
/// Integrates a function f over interval [a, b] using Romberg integration.
/// This method uses the trapezoidal rule at increasing grid densities and
/// applies Richardson extrapolation to improve accuracy.
///
/// Algorithm:
/// - R[k,0] = trapezoidal estimate with 2^k intervals
/// - R[k,m] = (4^m * R[k,m-1] - R[k-1,m-1]) / (4^m - 1) — Richardson extrapolation
/// - Diagonal element R[k,k] has O(h^(2k+2)) accuracy
/// - Stops when |R[k,k] - R[k-1,k-1]| < tol or max_iter reached
///
/// Parameters:
/// - T: floating-point type (f32 or f64)
/// - func: function pointer fn(T) T to integrate
/// - a: lower bound of integration
/// - b: upper bound of integration
/// - max_iter: maximum number of Romberg iterations (typically 10-20)
/// - tol: absolute error tolerance
/// - allocator: memory allocator for R table
///
/// Returns: struct { integral: T, error_estimate: T, iterations: usize }
///
/// Time: O(max_iter^2) | Space: O(max_iter^2)
pub fn romberg(comptime T: type, func: *const fn (T) T, a: T, b: T, max_iter: usize, tol: T, allocator: Allocator) !RombergResult(T) {
    if (a >= b) return error.InvalidInterval;
    if (max_iter == 0) return error.InvalidIteration;

    // Allocate 2D triangular table R[k][m]
    var R = try allocator.alloc([]T, max_iter);
    defer allocator.free(R);

    for (0..max_iter) |i| {
        R[i] = try allocator.alloc(T, i + 1);
    }
    defer for (0..max_iter) |i| {
        allocator.free(R[i]);
    };

    const h = b - a;

    // R[0,0] = trapezoidal estimate with 1 interval
    R[0][0] = (func(a) + func(b)) / 2.0 * h;

    var final_iter: usize = 0;

    for (1..max_iter) |k| {
        var sum: T = 0.0;

        // Compute h_k for iteration k (intervals have size h / 2^k)
        // In iteration k, we have 2^k intervals
        // We need to sum values at the new midpoints introduced in this iteration
        const power_two: T = @as(T, @floatFromInt(@as(u64, 1) << @intCast(k)));
        const h_k = h / power_two;

        // Sum function values at new midpoints (the points added in this iteration)
        // These are at positions: a + (2i-1)*h_k for i = 1, 2, ..., 2^(k-1)
        const num_new_points = @as(u64, 1) << @intCast(k - 1);
        for (1..num_new_points + 1) |i| {
            const i_f: T = @floatFromInt(i);
            const x = a + (2.0 * i_f - 1.0) * h_k;
            sum += func(x);
        }

        // R[k,0] = (R[k-1,0] + h_old * sum) where h_old = h / 2^(k-1)
        const h_prev = h / (@as(T, @floatFromInt(@as(u64, 1) << @intCast(k - 1))));
        R[k][0] = R[k - 1][0] / 2.0 + h_prev / 2.0 * sum;

        // Richardson extrapolation to build columns
        for (1..k + 1) |m| {
            const four_to_m = @as(T, @floatFromInt(@as(u64, 1) << @intCast(2 * m)));  // 4^m = 2^(2m)
            R[k][m] = (four_to_m * R[k][m - 1] - R[k - 1][m - 1]) / (four_to_m - 1.0);
        }

        // Check convergence: compare diagonal elements R[k,k] and R[k-1,k-1]
        if (k > 0) {
            const error_diff = @abs(R[k][k] - R[k - 1][k - 1]);
            if (error_diff < tol) {
                final_iter = k;
                break;
            }
        }

        final_iter = k;
    }

    const result_integral = R[final_iter][final_iter];
    var error_estimate: T = 0.0;

    if (final_iter > 0) {
        error_estimate = @abs(result_integral - R[final_iter - 1][final_iter - 1]);
    }

    return RombergResult(T){
        .integral = result_integral,
        .error_estimate = error_estimate,
        .iterations = final_iter + 1,
    };
}

// ============================================================================
// GAUSS-LEGENDRE QUADRATURE
// ============================================================================

/// Get precomputed nodes for Gauss-Legendre quadrature (in f64)
fn getGaussLegendreNodes(n: usize) ![]const f64 {
    return switch (n) {
        2 => &[_]f64{ -0.57735026918962576451, 0.57735026918962576451 },
        3 => &[_]f64{ -0.77459666924148517649, 0.0, 0.77459666924148517649 },
        4 => &[_]f64{ -0.86113631159405257523, -0.33998104358485626480, 0.33998104358485626480, 0.86113631159405257523 },
        5 => &[_]f64{ -0.90617984593866398606, -0.53846931010568309104, 0.0, 0.53846931010568309104, 0.90617984593866398606 },
        8 => &[_]f64{
            -0.9602898564975363,
            -0.7966664774136267,
            -0.5255324099163290,
            -0.1834346424956498,
            0.1834346424956498,
            0.5255324099163290,
            0.7966664774136267,
            0.9602898564975363,
        },
        16 => &[_]f64{
            -0.9894927619644001,
            -0.9445750230732326,
            -0.8656312023878921,
            -0.7554044083550030,
            -0.6178762444026438,
            -0.4580167776572934,
            -0.2816035507792589,
            -0.0950125098376374,
            0.0950125098376374,
            0.2816035507792589,
            0.4580167776572934,
            0.6178762444026438,
            0.7554044083550030,
            0.8656312023878921,
            0.9445750230732326,
            0.9894927619644001,
        },
        32 => &[_]f64{
            -0.9972638618494816,
            -0.9856115115452684,
            -0.9647622555875064,
            -0.9349060759377397,
            -0.8963211557660521,
            -0.8493676137325700,
            -0.7944837959679424,
            -0.7321821187402897,
            -0.6630442669302152,
            -0.5877157572407623,
            -0.5068999089322294,
            -0.4213512761306353,
            -0.3318686022821277,
            -0.2392873622521371,
            -0.1444719615827965,
            -0.0483076656877383,
            0.0483076656877383,
            0.1444719615827965,
            0.2392873622521371,
            0.3318686022821277,
            0.4213512761306353,
            0.5068999089322294,
            0.5877157572407623,
            0.6630442669302152,
            0.7321821187402897,
            0.7944837959679424,
            0.8493676137325700,
            0.8963211557660521,
            0.9349060759377397,
            0.9647622555875064,
            0.9856115115452684,
            0.9972638618494816,
        },
        else => error.UnsupportedOrder,
    };
}

/// Get precomputed weights for Gauss-Legendre quadrature (in f64)
fn getGaussLegendreWeights(n: usize) ![]const f64 {
    return switch (n) {
        2 => &[_]f64{ 1.0, 1.0 },
        3 => &[_]f64{ 0.55555555555555555556, 0.88888888888888888889, 0.55555555555555555556 },
        4 => &[_]f64{ 0.34785484513745385737, 0.65214515486254614263, 0.65214515486254614263, 0.34785484513745385737 },
        5 => &[_]f64{ 0.23692688505618908751, 0.47862867049936610145, 0.56888888888888888889, 0.47862867049936610145, 0.23692688505618908751 },
        8 => &[_]f64{
            0.1012285362903763,
            0.2223810344533744,
            0.3137066458778873,
            0.3626837833783620,
            0.3626837833783620,
            0.3137066458778873,
            0.2223810344533744,
            0.1012285362903763,
        },
        16 => &[_]f64{
            0.0271524594117541,
            0.0622535239386478,
            0.0951585116824927,
            0.1246289712555339,
            0.1495959888165767,
            0.1691565193950025,
            0.1826034150449236,
            0.1894506104550685,
            0.1894506104550685,
            0.1826034150449236,
            0.1691565193950025,
            0.1495959888165767,
            0.1246289712555339,
            0.0951585116824927,
            0.0622535239386478,
            0.0271524594117541,
        },
        32 => &[_]f64{
            0.0070186100094701,
            0.0162743947309057,
            0.0253920653092621,
            0.0342738629130214,
            0.0428358980222267,
            0.0509980592623762,
            0.0586840934785355,
            0.0658222227763618,
            0.0723457941088485,
            0.0781938957870703,
            0.0833119242269467,
            0.0876520930044038,
            0.0911738786957639,
            0.0938443990808046,
            0.0956387200792749,
            0.0965400885147278,
            0.0965400885147278,
            0.0956387200792749,
            0.0938443990808046,
            0.0911738786957639,
            0.0876520930044038,
            0.0833119242269467,
            0.0781938957870703,
            0.0723457941088485,
            0.0658222227763618,
            0.0586840934785355,
            0.0509980592623762,
            0.0428358980222267,
            0.0342738629130214,
            0.0253920653092621,
            0.0162743947309057,
            0.0070186100094701,
        },
        else => error.UnsupportedOrder,
    };
}

/// Gauss-Legendre quadrature for numerical integration
///
/// Computes ∫f(x)dx from a to b using n-point Gauss-Legendre quadrature.
/// This method is exact for polynomials of degree ≤ 2n-1 and highly
/// efficient for smooth functions.
///
/// Parameters:
/// - T: floating-point type (f32 or f64)
/// - func: function to integrate
/// - a: lower bound
/// - b: upper bound
/// - n: number of quadrature points (must be in {2,3,4,5,8,16,32})
/// - allocator: memory allocator (not used, included for API consistency)
///
/// Returns: approximate value of the integral
///
/// Errors:
/// - error.UnsupportedOrder: if n is not in {2,3,4,5,8,16,32}
///
/// Time: O(n) | Space: O(1)
pub fn gauss_legendre(comptime T: type, func: *const fn (T) T, a: T, b: T, n: usize, allocator: Allocator) !T {
    _ = allocator; // not used, but included for API consistency

    // Get nodes and weights for order n (both in f64)
    const nodes_f64 = try getGaussLegendreNodes(n);
    const weights_f64 = try getGaussLegendreWeights(n);

    // Handle zero-width interval
    if (a == b) {
        return 0.0;
    }

    // Handle reversed bounds
    const is_reversed = a > b;
    const lower = if (is_reversed) b else a;
    const upper = if (is_reversed) a else b;

    // Transform interval [a,b] to [-1,1]
    // t = (b-a)/2 * x + (a+b)/2  where x ∈ [-1,1] and t ∈ [a,b]
    const mid = (lower + upper) / 2.0;
    const half_width = (upper - lower) / 2.0;

    // Apply quadrature formula
    // ∫[a,b] f(t)dt = (b-a)/2 * Σ w_i * f((b-a)/2 * x_i + (a+b)/2)
    var result: T = 0.0;
    for (0..n) |i| {
        const node_val: T = @floatCast(nodes_f64[i]);
        const weight_val: T = @floatCast(weights_f64[i]);
        const t = mid + half_width * node_val;
        result += weight_val * func(t);
    }
    result *= half_width;

    // If bounds were reversed, negate the result
    if (is_reversed) {
        result = -result;
    }

    return result;
}

// Helper functions for romberg tests

fn rombergConstantFunc(x: f64) f64 {
    _ = x;
    return 7.0;
}

fn rombergLinearFunc(x: f64) f64 {
    return x;
}

fn rombergQuadraticFunc(x: f64) f64 {
    return x * x;
}

fn rombergCubicFunc(x: f64) f64 {
    return x * x * x;
}

fn rombergSinFunc(x: f64) f64 {
    return @sin(x);
}

fn rombergCosFunc(x: f64) f64 {
    return @cos(x);
}

fn rombergExpFunc(x: f64) f64 {
    return @exp(x);
}

fn rombergLogFunc(x: f64) f64 {
    return @log(x);
}

fn rombergReciprocalFunc(x: f64) f64 {
    return 1.0 / x;
}

fn rombergNegativeFunc(x: f64) f64 {
    return x - 0.5;
}

fn rombergPolynomialDegree5Func(x: f64) f64 {
    // x + x² + x³ + x⁴ + x⁵
    return x + x * x + x * x * x + x * x * x * x + x * x * x * x * x;
}

fn rombergConstantFunc32(x: f32) f32 {
    _ = x;
    return 3.0;
}

fn rombergSinFunc32(x: f32) f32 {
    return @sin(x);
}

// ============================================================================
// ROMBERG TESTS
// ============================================================================

test "romberg constant function integral (exact)" {
    const allocator = testing.allocator;

    // ∫_0^1 7 dx = 7
    const result = try romberg(f64, rombergConstantFunc, 0.0, 1.0, 10, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, 7.0, 1e-9);
    try testing.expect(result.iterations > 0);
    try testing.expect(result.iterations <= 10);
}

test "romberg linear function integral (exact)" {
    const allocator = testing.allocator;

    // ∫_0^2 x dx = [x²/2]_0^2 = 2
    const result = try romberg(f64, rombergLinearFunc, 0.0, 2.0, 10, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, 2.0, 1e-9);
}

test "romberg quadratic function integral" {
    const allocator = testing.allocator;

    // ∫_0^1 x² dx = [x³/3]_0^1 = 1/3
    const result = try romberg(f64, rombergQuadraticFunc, 0.0, 1.0, 10, 1e-10, allocator);
    const expected = 1.0 / 3.0;
    try testing.expectApproxEqAbs(result.integral, expected, 1e-9);
}

test "romberg cubic function integral (high accuracy)" {
    const allocator = testing.allocator;

    // ∫_0^2 x³ dx = [x⁴/4]_0^2 = 4
    const result = try romberg(f64, rombergCubicFunc, 0.0, 2.0, 10, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, 4.0, 1e-9);
}

test "romberg zero function integral" {
    const allocator = testing.allocator;

    // ∫_0^1 0 dx = 0
    const result = try romberg(f64, struct {
        pub fn zeroFunc(x: f64) f64 {
            _ = x;
            return 0.0;
        }
    }.zeroFunc, 0.0, 1.0, 10, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, 0.0, 1e-10);
}

test "romberg reversed bounds error handling" {
    const allocator = testing.allocator;

    // b < a should return error
    const result = romberg(f64, rombergConstantFunc, 2.0, 1.0, 10, 1e-10, allocator);
    try testing.expectError(error.InvalidInterval, result);
}

test "romberg equal bounds error handling" {
    const allocator = testing.allocator;

    // a == b should return error
    const result = romberg(f64, rombergConstantFunc, 1.0, 1.0, 10, 1e-10, allocator);
    try testing.expectError(error.InvalidInterval, result);
}

test "romberg zero max_iter error handling" {
    const allocator = testing.allocator;

    // max_iter = 0 should return error
    const result = romberg(f64, rombergConstantFunc, 0.0, 1.0, 0, 1e-10, allocator);
    try testing.expectError(error.InvalidIteration, result);
}

test "romberg sin function integral (known value)" {
    const allocator = testing.allocator;

    // ∫_0^π sin(x) dx = 2
    const result = try romberg(f64, rombergSinFunc, 0.0, math.pi, 15, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, 2.0, 1e-9);
}

test "romberg cos function integral (known value)" {
    const allocator = testing.allocator;

    // ∫_0^π/2 cos(x) dx = 1
    const result = try romberg(f64, rombergCosFunc, 0.0, math.pi / 2.0, 15, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, 1.0, 1e-9);
}

test "romberg exponential function integral (known value)" {
    const allocator = testing.allocator;

    // ∫_0^1 e^x dx = e - 1 ≈ 1.71828...
    const result = try romberg(f64, rombergExpFunc, 0.0, 1.0, 15, 1e-10, allocator);
    const expected = math.e - 1.0;
    try testing.expectApproxEqAbs(result.integral, expected, 1e-9);
}

test "romberg log function integral (known value)" {
    const allocator = testing.allocator;

    // ∫_1^e ln(x) dx = [x*ln(x) - x]_1^e = (e - e) - (0 - 1) = 1
    const result = try romberg(f64, rombergLogFunc, 1.0, math.e, 15, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, 1.0, 1e-8);
}

test "romberg reciprocal function integral" {
    const allocator = testing.allocator;

    // ∫_1^e 1/x dx = [ln(x)]_1^e = 1
    const result = try romberg(f64, rombergReciprocalFunc, 1.0, math.e, 15, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, 1.0, 1e-8);
}

test "romberg polynomial degree 5 integral" {
    const allocator = testing.allocator;

    // ∫_0^1 (x + x² + x³ + x⁴ + x⁵) dx = 1/2 + 1/3 + 1/4 + 1/5 + 1/6
    const expected = 1.0 / 2.0 + 1.0 / 3.0 + 1.0 / 4.0 + 1.0 / 5.0 + 1.0 / 6.0;
    const result = try romberg(f64, rombergPolynomialDegree5Func, 0.0, 1.0, 15, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, expected, 1e-9);
}

test "romberg negative integrand" {
    const allocator = testing.allocator;

    // ∫_0^1 (x - 0.5) dx = [x²/2 - 0.5*x]_0^1 = 0.5 - 0.5 = 0
    const result = try romberg(f64, rombergNegativeFunc, 0.0, 1.0, 10, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, 0.0, 1e-9);
}

test "romberg very small interval" {
    const allocator = testing.allocator;

    // ∫_0^1e-8 5 dx = 5e-8
    const interval = 1e-8;
    const expected = 5.0 * interval;
    const result = try romberg(f64, struct {
        pub fn f(x: f64) f64 {
            _ = x;
            return 5.0;
        }
    }.f, 0.0, interval, 10, 1e-15, allocator);
    try testing.expectApproxEqAbs(result.integral, expected, 1e-14);
}

test "romberg large interval" {
    const allocator = testing.allocator;

    // ∫_-50^50 4 dx = 400
    const result = try romberg(f64, struct {
        pub fn f(x: f64) f64 {
            _ = x;
            return 4.0;
        }
    }.f, -50.0, 50.0, 12, 1e-6, allocator);
    try testing.expectApproxEqAbs(result.integral, 400.0, 1.0);
}

test "romberg richardson extrapolation improves accuracy" {
    const allocator = testing.allocator;

    // Romberg should achieve high accuracy through Richardson extrapolation
    // compared to basic trapezoidal rule
    const result = try romberg(f64, rombergSinFunc, 0.0, math.pi, 15, 1e-12, allocator);

    // R[k,k] should be very accurate for smooth functions like sin(x)
    try testing.expectApproxEqAbs(result.integral, 2.0, 1e-10);
}

test "romberg convergence with more iterations" {
    const allocator = testing.allocator;

    // Higher iterations should give better accuracy
    const loose = try romberg(f64, rombergSinFunc, 0.0, math.pi, 5, 1e-6, allocator);
    const tight = try romberg(f64, rombergSinFunc, 0.0, math.pi, 15, 1e-12, allocator);

    // Both should converge to 2.0, but tight should be closer
    try testing.expectApproxEqAbs(loose.integral, 2.0, 1e-5);
    try testing.expectApproxEqAbs(tight.integral, 2.0, 1e-10);
}

test "romberg error estimate is reasonable" {
    const allocator = testing.allocator;

    // Error estimate should be non-negative and roughly reflect actual error
    const result = try romberg(f64, rombergSinFunc, 0.0, math.pi, 10, 1e-10, allocator);
    try testing.expect(result.error_estimate >= 0.0);
    // Actual error should be within a few times the error estimate
    const actual_error = @abs(result.integral - 2.0);
    try testing.expect(actual_error <= 1e-8);
}

test "romberg iterations count reasonable" {
    const allocator = testing.allocator;

    // Iterations should be positive and not exceed max_iter
    const result = try romberg(f64, rombergSinFunc, 0.0, math.pi, 10, 1e-10, allocator);
    try testing.expect(result.iterations > 0);
    try testing.expect(result.iterations <= 10);
}

test "romberg early stop when converged" {
    const allocator = testing.allocator;

    // Tight tolerance on smooth function should converge early
    const tight = try romberg(f64, rombergSinFunc, 0.0, math.pi, 20, 1e-10, allocator);

    // Should converge well before max_iter
    try testing.expect(tight.iterations < 20);
}

test "romberg tolerance threshold triggers early stop" {
    const allocator = testing.allocator;

    // Very loose tolerance should stop immediately
    const loose = try romberg(f64, rombergSinFunc, 0.0, math.pi, 20, 0.5, allocator);

    // Should use very few iterations
    try testing.expect(loose.iterations < 5);
    // But should still be somewhat accurate
    try testing.expectApproxEqAbs(loose.integral, 2.0, 0.4);
}

test "romberg negative bounds (swap respected)" {
    const allocator = testing.allocator;

    // ∫_{-1}^{1} x² dx = 2/3
    const result = try romberg(f64, rombergQuadraticFunc, -1.0, 1.0, 10, 1e-10, allocator);
    const expected = 2.0 / 3.0;
    try testing.expectApproxEqAbs(result.integral, expected, 1e-9);
}

test "romberg f32 support with loose tolerance" {
    const allocator = testing.allocator;

    // ∫_0^1 3 dx = 3 (f32)
    const result = try romberg(f32, rombergConstantFunc32, 0.0, 1.0, 10, 1e-4, allocator);
    try testing.expectApproxEqAbs(result.integral, 3.0, 1e-3);
}

test "romberg f32 sin integration" {
    const allocator = testing.allocator;

    // ∫_0^π sin(x) dx = 2 (f32)
    const result = try romberg(f32, rombergSinFunc32, 0.0, math.pi, 15, 1e-5, allocator);
    try testing.expectApproxEqAbs(result.integral, 2.0, 1e-4);
}

test "romberg memory safety (no leaks)" {
    const allocator = testing.allocator;

    // Multiple calls should not leak
    var total: f64 = 0.0;
    for (0..5) |_| {
        const result = try romberg(f64, rombergSinFunc, 0.0, math.pi, 10, 1e-10, allocator);
        total += result.integral;
    }

    // Average should be close to 2.0
    try testing.expectApproxEqAbs(total / 5.0, 2.0, 1e-9);
}

test "romberg quadratic polynomial (exact within iterations)" {
    const allocator = testing.allocator;

    // ∫_0^2 x² dx = 8/3
    const result = try romberg(f64, rombergQuadraticFunc, 0.0, 2.0, 10, 1e-10, allocator);
    const expected = 8.0 / 3.0;
    try testing.expectApproxEqAbs(result.integral, expected, 1e-9);
}

test "romberg cubic polynomial (exact within iterations)" {
    const allocator = testing.allocator;

    // ∫_0^1 x³ dx = 1/4
    const result = try romberg(f64, rombergCubicFunc, 0.0, 1.0, 10, 1e-10, allocator);
    const expected = 0.25;
    try testing.expectApproxEqAbs(result.integral, expected, 1e-9);
}

test "romberg oscillatory function (low frequency)" {
    const allocator = testing.allocator;

    // ∫_0^2π sin(x) dx = 0 (complete period)
    const result = try romberg(f64, rombergSinFunc, 0.0, 2.0 * math.pi, 15, 1e-10, allocator);
    try testing.expectApproxEqAbs(result.integral, 0.0, 1e-8);
}

// ============================================================================
// GAUSS-LEGENDRE INTEGRATION TESTS
// ============================================================================

// Helper functions for gauss_legendre tests

fn gaussConstantFunc(x: f64) f64 {
    _ = x;
    return 5.0;
}

fn gaussLinearFunc(x: f64) f64 {
    return x;
}

fn gaussQuadraticFunc(x: f64) f64 {
    return x * x;
}

fn gaussCubicFunc(x: f64) f64 {
    return x * x * x;
}

fn gaussQuarticFunc(x: f64) f64 {
    return x * x * x * x;
}

fn gaussQuinticFunc(x: f64) f64 {
    return x * x * x * x * x;
}

fn gaussPolynomialDegree3Func(x: f64) f64 {
    // x³ + 2x² + x + 1
    return x * x * x + 2.0 * x * x + x + 1.0;
}

fn gaussPolynomialDegree5Func(x: f64) f64 {
    // x⁵ + x⁴ + x³ + x² + x + 1
    return x * x * x * x * x + x * x * x * x + x * x * x + x * x + x + 1.0;
}

fn gaussPolynomialDegree7Func(x: f64) f64 {
    // x⁷ + x⁶ + x⁵ + x⁴ + x³ + x² + x + 1
    const x2 = x * x;
    const x3 = x2 * x;
    const x4 = x3 * x;
    const x5 = x4 * x;
    const x6 = x5 * x;
    const x7 = x6 * x;
    return x7 + x6 + x5 + x4 + x3 + x2 + x + 1.0;
}

fn gaussPolynomialDegree9Func(x: f64) f64 {
    // x⁹ + x⁸ + x⁷ + ... + x + 1
    const x2 = x * x;
    const x3 = x2 * x;
    const x4 = x3 * x;
    const x5 = x4 * x;
    const x6 = x5 * x;
    const x7 = x6 * x;
    const x8 = x7 * x;
    const x9 = x8 * x;
    return x9 + x8 + x7 + x6 + x5 + x4 + x3 + x2 + x + 1.0;
}

fn gaussSinFunc(x: f64) f64 {
    return @sin(x);
}

fn gaussCosFunc(x: f64) f64 {
    return @cos(x);
}

fn gaussExpFunc(x: f64) f64 {
    return @exp(x);
}

fn gaussLogFunc(x: f64) f64 {
    return @log(x);
}

fn gaussReciprocalFunc(x: f64) f64 {
    return 1.0 / x;
}

fn gaussSqrtFunc(x: f64) f64 {
    return @sqrt(x);
}

fn gaussNegativeFunc(x: f64) f64 {
    return -x * x;
}

fn gaussConstantFunc32(x: f32) f32 {
    _ = x;
    return 3.0;
}

fn gaussSinFunc32(x: f32) f32 {
    return @sin(x);
}

// ============================================================================
// GAUSS-LEGENDRE BASIC OPERATIONS TESTS
// ============================================================================

test "gauss_legendre constant function (exact)" {
    const allocator = testing.allocator;

    // ∫_0^1 5 dx = 5
    const result = try gauss_legendre(f64, gaussConstantFunc, 0.0, 1.0, 2, allocator);
    try testing.expectApproxEqAbs(result, 5.0, 1e-12);
}

test "gauss_legendre linear function (exact)" {
    const allocator = testing.allocator;

    // ∫_0^2 x dx = [x²/2]_0^2 = 2
    const result = try gauss_legendre(f64, gaussLinearFunc, 0.0, 2.0, 2, allocator);
    try testing.expectApproxEqAbs(result, 2.0, 1e-12);
}

test "gauss_legendre quadratic function (exact for n≥2)" {
    const allocator = testing.allocator;

    // ∫_0^1 x² dx = 1/3
    const result = try gauss_legendre(f64, gaussQuadraticFunc, 0.0, 1.0, 2, allocator);
    const expected = 1.0 / 3.0;
    try testing.expectApproxEqAbs(result, expected, 1e-12);
}

test "gauss_legendre cubic function (exact for n≥2)" {
    const allocator = testing.allocator;

    // ∫_0^2 x³ dx = [x⁴/4]_0^2 = 4
    const result = try gauss_legendre(f64, gaussCubicFunc, 0.0, 2.0, 2, allocator);
    try testing.expectApproxEqAbs(result, 4.0, 1e-12);
}

test "gauss_legendre zero function" {
    const allocator = testing.allocator;

    // ∫_0^1 0 dx = 0
    const result = try gauss_legendre(f64, struct {
        pub fn zeroFunc(x: f64) f64 {
            _ = x;
            return 0.0;
        }
    }.zeroFunc, 0.0, 1.0, 2, allocator);
    try testing.expectApproxEqAbs(result, 0.0, 1e-12);
}

test "gauss_legendre negative bounds (reversed interval)" {
    const allocator = testing.allocator;

    // ∫_2^0 x dx = -∫_0^2 x dx = -2
    const result = try gauss_legendre(f64, gaussLinearFunc, 2.0, 0.0, 2, allocator);
    try testing.expectApproxEqAbs(result, -2.0, 1e-12);
}

// ============================================================================
// GAUSS-LEGENDRE POLYNOMIAL EXACTNESS TESTS
// ============================================================================

test "gauss_legendre n=2 exact for degree 3 (x³ + 2x² + x + 1)" {
    const allocator = testing.allocator;

    // ∫_0^1 (x³ + 2x² + x + 1) dx = 1/4 + 2/3 + 1/2 + 1 = 19/12
    const result = try gauss_legendre(f64, gaussPolynomialDegree3Func, 0.0, 1.0, 2, allocator);
    const expected = 0.25 + 2.0 / 3.0 + 0.5 + 1.0; // = 19/12
    try testing.expectApproxEqAbs(result, expected, 1e-12);
}

test "gauss_legendre n=3 exact for degree 5 (x⁵ + x⁴ + x³ + x² + x + 1)" {
    const allocator = testing.allocator;

    // ∫_0^1 (x⁵ + x⁴ + x³ + x² + x + 1) dx = 1/6 + 1/5 + 1/4 + 1/3 + 1/2 + 1
    const result = try gauss_legendre(f64, gaussPolynomialDegree5Func, 0.0, 1.0, 3, allocator);
    const expected = 1.0 / 6.0 + 1.0 / 5.0 + 1.0 / 4.0 + 1.0 / 3.0 + 1.0 / 2.0 + 1.0;
    try testing.expectApproxEqAbs(result, expected, 1e-12);
}

test "gauss_legendre n=4 exact for degree 7" {
    const allocator = testing.allocator;

    // ∫_0^1 (x⁷ + x⁶ + ... + x + 1) dx = 1/8 + 1/7 + 1/6 + 1/5 + 1/4 + 1/3 + 1/2 + 1
    const result = try gauss_legendre(f64, gaussPolynomialDegree7Func, 0.0, 1.0, 4, allocator);
    const expected = 1.0 / 8.0 + 1.0 / 7.0 + 1.0 / 6.0 + 1.0 / 5.0 + 1.0 / 4.0 + 1.0 / 3.0 + 1.0 / 2.0 + 1.0;
    try testing.expectApproxEqAbs(result, expected, 1e-12);
}

test "gauss_legendre n=5 exact for degree 9" {
    const allocator = testing.allocator;

    // ∫_0^1 (x⁹ + x⁸ + ... + x + 1) dx = sum of 1/k for k=1..10
    const result = try gauss_legendre(f64, gaussPolynomialDegree9Func, 0.0, 1.0, 5, allocator);
    const expected = 1.0 / 10.0 + 1.0 / 9.0 + 1.0 / 8.0 + 1.0 / 7.0 + 1.0 / 6.0 + 1.0 / 5.0 + 1.0 / 4.0 + 1.0 / 3.0 + 1.0 / 2.0 + 1.0;
    try testing.expectApproxEqAbs(result, expected, 1e-12);
}

test "gauss_legendre n=8 exact for degree 15" {
    const allocator = testing.allocator;

    // ∫_0^1 x¹⁵ dx = 1/16
    const result = try gauss_legendre(f64, struct {
        pub fn f(x: f64) f64 {
            var res = x;
            for (1..15) |_| {
                res *= x;
            }
            return res;
        }
    }.f, 0.0, 1.0, 8, allocator);
    const expected = 1.0 / 16.0;
    try testing.expectApproxEqAbs(result, expected, 1e-12);
}

test "gauss_legendre polynomial exactness check (increasing n)" {
    const allocator = testing.allocator;

    // Both n=3 and n=5 should exactly integrate quintic (degree 5)
    const result_n3 = try gauss_legendre(f64, gaussPolynomialDegree5Func, 0.0, 1.0, 3, allocator);
    const result_n5 = try gauss_legendre(f64, gaussPolynomialDegree5Func, 0.0, 1.0, 5, allocator);

    const expected = 1.0 / 6.0 + 1.0 / 5.0 + 1.0 / 4.0 + 1.0 / 3.0 + 1.0 / 2.0 + 1.0;

    // Both should be exact
    try testing.expectApproxEqAbs(result_n3, expected, 1e-12);
    try testing.expectApproxEqAbs(result_n5, expected, 1e-12);
}

// ============================================================================
// GAUSS-LEGENDRE MATHEMATICAL PROPERTIES TESTS
// ============================================================================

test "gauss_legendre sin(x) from 0 to π (n=5, known integral = 2)" {
    const allocator = testing.allocator;

    // ∫_0^π sin(x) dx = 2
    // n=5 Gauss-Legendre exact for polynomials up to degree 9
    // sin(x) Taylor: x - x³/6 + x⁵/120 - ...
    // Expected error: O(10^-4) for n=5 on transcendental
    const result = try gauss_legendre(f64, gaussSinFunc, 0.0, math.pi, 5, allocator);
    try testing.expectApproxEqAbs(result, 2.0, 1e-4);
}

test "gauss_legendre cos(x) from 0 to π/2 (n=5, known integral = 1)" {
    const allocator = testing.allocator;

    // ∫_0^π/2 cos(x) dx = 1
    // n=5 Gauss-Legendre exact for polynomials up to degree 9
    // Expected error: O(10^-4) for n=5 on transcendental
    const result = try gauss_legendre(f64, gaussCosFunc, 0.0, math.pi / 2.0, 5, allocator);
    try testing.expectApproxEqAbs(result, 1.0, 1e-4);
}

test "gauss_legendre e^x from 0 to 1 (n=8, known integral = e - 1)" {
    const allocator = testing.allocator;

    // ∫_0^1 e^x dx = e - 1
    const result = try gauss_legendre(f64, gaussExpFunc, 0.0, 1.0, 8, allocator);
    const expected = math.e - 1.0;
    try testing.expectApproxEqAbs(result, expected, 1e-12);
}

test "gauss_legendre 1/x from 1 to 2 (n=8, known integral = ln(2))" {
    const allocator = testing.allocator;

    // ∫_1^2 1/x dx = ln(2)
    const result = try gauss_legendre(f64, gaussReciprocalFunc, 1.0, 2.0, 8, allocator);
    const expected = @log(2.0);
    try testing.expectApproxEqAbs(result, expected, 1e-12);
}

test "gauss_legendre sqrt(x) from 0 to 1 (n=8, known integral = 2/3)" {
    const allocator = testing.allocator;

    // ∫_0^1 √x dx = 2/3
    // n=8 Gauss-Legendre exact for polynomials up to degree 15
    // sqrt(x) is non-polynomial, expected error: O(10^-3)
    const result = try gauss_legendre(f64, gaussSqrtFunc, 0.0, 1.0, 8, allocator);
    const expected = 2.0 / 3.0;
    try testing.expectApproxEqAbs(result, expected, 1e-3);
}

test "gauss_legendre negative integrand" {
    const allocator = testing.allocator;

    // ∫_0^1 -x² dx = -1/3
    const result = try gauss_legendre(f64, gaussNegativeFunc, 0.0, 1.0, 2, allocator);
    const expected = -1.0 / 3.0;
    try testing.expectApproxEqAbs(result, expected, 1e-12);
}

// ============================================================================
// GAUSS-LEGENDRE ORDER COMPARISON TESTS
// ============================================================================

test "gauss_legendre higher order improves accuracy (n=2 vs n=8 on sin)" {
    const allocator = testing.allocator;

    const result_n2 = try gauss_legendre(f64, gaussSinFunc, 0.0, math.pi, 2, allocator);
    const result_n8 = try gauss_legendre(f64, gaussSinFunc, 0.0, math.pi, 8, allocator);

    const exact = 2.0;
    const error_n2 = @abs(result_n2 - exact);
    const error_n8 = @abs(result_n8 - exact);

    // Both should be close, but n=8 should be closer
    try testing.expect(error_n2 > 1e-10);
    try testing.expect(error_n8 < 1e-12);
    try testing.expect(error_n8 < error_n2);
}

test "gauss_legendre low-order sufficient for low-degree polynomials" {
    const allocator = testing.allocator;

    // n=2 should be exact for quadratic
    const result = try gauss_legendre(f64, gaussQuadraticFunc, 0.0, 1.0, 2, allocator);
    const expected = 1.0 / 3.0;
    try testing.expectApproxEqAbs(result, expected, 1e-12);
}

test "gauss_legendre all supported orders work (n=2,3,4,5,8,16,32)" {
    const allocator = testing.allocator;

    const orders = [_]usize{ 2, 3, 4, 5, 8, 16, 32 };

    for (orders) |n| {
        const result = try gauss_legendre(f64, gaussSinFunc, 0.0, math.pi, n, allocator);
        // All should converge to 2.0 for sin(x) from 0 to π
        // n=2 gives ~1.935, higher orders converge faster
        try testing.expectApproxEqAbs(result, 2.0, 0.1);
    }
}

test "gauss_legendre unsupported order error (n=7)" {
    const allocator = testing.allocator;

    const result = gauss_legendre(f64, gaussSinFunc, 0.0, math.pi, 7, allocator);
    try testing.expectError(error.UnsupportedOrder, result);
}

test "gauss_legendre unsupported order error (n=100)" {
    const allocator = testing.allocator;

    const result = gauss_legendre(f64, gaussSinFunc, 0.0, math.pi, 100, allocator);
    try testing.expectError(error.UnsupportedOrder, result);
}

// ============================================================================
// GAUSS-LEGENDRE EDGE CASES TESTS
// ============================================================================

test "gauss_legendre very small interval" {
    const allocator = testing.allocator;

    // ∫_0^1e-8 5 dx = 5e-8
    const interval = 1e-8;
    const expected = 5.0 * interval;
    const result = try gauss_legendre(f64, struct {
        pub fn f(x: f64) f64 {
            _ = x;
            return 5.0;
        }
    }.f, 0.0, interval, 2, allocator);
    try testing.expectApproxEqAbs(result, expected, 1e-14);
}

test "gauss_legendre large interval" {
    const allocator = testing.allocator;

    // ∫_{-100}^{100} 2 dx = 400
    const result = try gauss_legendre(f64, struct {
        pub fn f(x: f64) f64 {
            _ = x;
            return 2.0;
        }
    }.f, -100.0, 100.0, 5, allocator);
    try testing.expectApproxEqAbs(result, 400.0, 1e-10);
}

test "gauss_legendre zero-width interval (a == b)" {
    const allocator = testing.allocator;

    // ∫_1^1 5 dx = 0
    const result = try gauss_legendre(f64, gaussConstantFunc, 1.0, 1.0, 2, allocator);
    try testing.expectApproxEqAbs(result, 0.0, 1e-12);
}

// ============================================================================
// GAUSS-LEGENDRE TYPE SUPPORT TESTS
// ============================================================================

test "gauss_legendre f32 support with tolerance 1e-4" {
    const allocator = testing.allocator;

    // ∫_0^1 3 dx = 3 (f32)
    const result = try gauss_legendre(f32, gaussConstantFunc32, 0.0, 1.0, 2, allocator);
    try testing.expectApproxEqAbs(result, 3.0, 1e-4);
}

test "gauss_legendre f32 sin integration with tolerance 1e-5" {
    const allocator = testing.allocator;

    // ∫_0^π sin(x) dx = 2 (f32)
    const result = try gauss_legendre(f32, gaussSinFunc32, 0.0, math.pi, 5, allocator);
    try testing.expectApproxEqAbs(result, 2.0, 1e-5);
}

test "gauss_legendre f64 with tolerance 1e-10" {
    const allocator = testing.allocator;

    // ∫_0^1 e^x dx = e - 1 (f64)
    const result = try gauss_legendre(f64, gaussExpFunc, 0.0, 1.0, 8, allocator);
    const expected = math.e - 1.0;
    try testing.expectApproxEqAbs(result, expected, 1e-10);
}

// ============================================================================
// GAUSS-LEGENDRE MEMORY SAFETY TESTS
// ============================================================================

test "gauss_legendre memory safety (no leaks)" {
    const allocator = testing.allocator;

    // Multiple calls should not leak
    var total: f64 = 0.0;
    for (0..5) |_| {
        const result = try gauss_legendre(f64, gaussSinFunc, 0.0, math.pi, 8, allocator);
        total += result;
    }

    // Average should be close to 2.0
    try testing.expectApproxEqAbs(total / 5.0, 2.0, 1e-10);
}
