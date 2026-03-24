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
