//! Numerical Interpolation — Linear 1D Interpolation
//!
//! This module provides interpolation methods for estimating function values
//! at arbitrary points given discrete sample points.
//!
//! ## Supported Operations
//! - `interp1d` — 1D linear interpolation with constant extrapolation
//!
//! ## Time Complexity
//! - interp1d: O(m log n + m) where m = number of query points, n = number of sample points
//!   (binary search for each query point, then linear interpolation)
//!
//! ## Space Complexity
//! - interp1d: O(m) for output array (caller owns)
//!
//! ## Numeric Properties
//! - Linear interpolation: exact for linear functions, good approximation for smooth functions
//! - Extrapolation: constant (clamp to first/last value)
//! - Works with uniform and non-uniform grids
//! - Assumes monotonically increasing x coordinates
//!
//! ## Use Cases
//! - Interpolating discrete measurements to arbitrary points
//! - Resampling signals at different rates
//! - Curve fitting and function approximation
//! - Data smoothing and upsampling
//! - Numerical solution refinement

const std = @import("std");
const math = std.math;
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Perform 1D linear interpolation at query points
///
/// Interpolates function values at arbitrary query points given a set of known
/// sample points (x, y). Uses linear interpolation between consecutive points
/// and constant extrapolation for values outside the sample domain.
///
/// Parameters:
/// - T: floating-point type (f32 or f64)
/// - x: slice of x-coordinates (sample points) — must be monotonically increasing
/// - y: slice of y-coordinates (function values) — must have same length as x
/// - x_new: slice of x-coordinates where we want interpolated values
/// - allocator: memory allocator for output array (caller owns returned memory)
///
/// Returns: allocated array of interpolated y values at x_new points (caller must free)
///
/// Errors:
/// - error.DimensionMismatch: if x.len != y.len
/// - error.InsufficientPoints: if x.len < 2
/// - error.NonMonotonicX: if x is not monotonically increasing
/// - error.OutOfMemory: if allocation fails
///
/// Formula (for x_new[i] in [x[j], x[j+1]]):
/// ```
/// y_new[i] = y[j] + (y[j+1] - y[j]) * (x_new[i] - x[j]) / (x[j+1] - x[j])
/// ```
///
/// Extrapolation:
/// - For x_new[i] < x[0]: y_new[i] = y[0]
/// - For x_new[i] > x[n-1]: y_new[i] = y[n-1]
///
/// Time: O(m log n + m) where m = x_new.len, n = x.len | Space: O(m)
pub fn interp1d(comptime T: type, x: []const T, y: []const T, x_new: []const T, allocator: Allocator) ![]T {
    // Validation
    if (x.len != y.len) return error.DimensionMismatch;
    if (x.len < 2) return error.InsufficientPoints;

    // Validate that x is monotonically increasing
    for (1..x.len) |i| {
        if (x[i] <= x[i - 1]) return error.NonMonotonicX;
    }

    // Allocate output array
    const result = try allocator.alloc(T, x_new.len);
    errdefer allocator.free(result);

    // Handle empty query points
    if (x_new.len == 0) return result;

    // Interpolate at each query point
    for (0..x_new.len) |i| {
        const xi = x_new[i];

        // Extrapolation: below minimum
        if (xi <= x[0]) {
            result[i] = y[0];
            continue;
        }

        // Extrapolation: above maximum
        if (xi >= x[x.len - 1]) {
            result[i] = y[y.len - 1];
            continue;
        }

        // Interpolation: find the interval [x[j], x[j+1]] containing xi
        // Using binary search for efficiency
        var left: usize = 0;
        var right: usize = x.len - 1;

        while (left < right) {
            const mid = left + (right - left) / 2;
            if (x[mid] < xi) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // Now x[left-1] < xi <= x[left]
        // But we need x[j] < xi < x[j+1], so j = left - 1
        const j = left - 1;

        // Linear interpolation formula
        const x0 = x[j];
        const x1 = x[j + 1];
        const y0 = y[j];
        const y1 = y[j + 1];

        const dx = x1 - x0;
        const dy = y1 - y0;
        const alpha = (xi - x0) / dx;

        result[i] = y0 + alpha * dy;
    }

    return result;
}

// Error types for numerical interpolation
pub const InterpolationError = error{
    DimensionMismatch,
    InsufficientPoints,
    NonMonotonicX,
};

// ============================================================================
// TESTS
// ============================================================================

test "interp1d exact match - query at sample points" {
    const allocator = testing.allocator;

    const x = [_]f64{ 0.0, 1.0, 2.0, 3.0 };
    const y = [_]f64{ 0.0, 1.0, 4.0, 9.0 };
    const x_new = [_]f64{ 0.0, 1.0, 2.0, 3.0 };

    const result = try interp1d(f64, &x, &y, &x_new, allocator);
    defer allocator.free(result);

    // Should return exact y values
    try testing.expectApproxEqAbs(result[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(result[1], 1.0, 1e-10);
    try testing.expectApproxEqAbs(result[2], 4.0, 1e-10);
    try testing.expectApproxEqAbs(result[3], 9.0, 1e-10);
}

test "interp1d midpoint interpolation" {
    const allocator = testing.allocator;

    const x = [_]f64{ 0.0, 2.0 };
    const y = [_]f64{ 0.0, 4.0 };
    const x_new = [_]f64{ 1.0 };

    const result = try interp1d(f64, &x, &y, &x_new, allocator);
    defer allocator.free(result);

    // Midpoint between (0,0) and (2,4) is (1,2)
    try testing.expectApproxEqAbs(result[0], 2.0, 1e-10);
}

test "interp1d linear function - exact interpolation" {
    const allocator = testing.allocator;

    // f(x) = 2x + 1
    const x = [_]f64{ 0.0, 1.0, 2.0, 3.0 };
    const y = [_]f64{ 1.0, 3.0, 5.0, 7.0 };

    // Query at various points
    const x_new = [_]f64{ 0.5, 1.5, 2.5 };

    const result = try interp1d(f64, &x, &y, &x_new, allocator);
    defer allocator.free(result);

    // f(0.5) = 2 * 0.5 + 1 = 2
    try testing.expectApproxEqAbs(result[0], 2.0, 1e-10);
    // f(1.5) = 2 * 1.5 + 1 = 4
    try testing.expectApproxEqAbs(result[1], 4.0, 1e-10);
    // f(2.5) = 2 * 2.5 + 1 = 6
    try testing.expectApproxEqAbs(result[2], 6.0, 1e-10);
}

test "interp1d quadratic function - good approximation" {
    const allocator = testing.allocator;

    // f(x) = x² with finer grid for better linear approximation
    const x = [_]f64{ 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 };
    var y: [6]f64 = undefined;
    for (0..6) |i| {
        y[i] = x[i] * x[i];
    }

    // Query at intermediate points
    const x_new = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };

    const result = try interp1d(f64, &x, &y, &x_new, allocator);
    defer allocator.free(result);

    // With finer grid, linear approximation is better for quadratic
    // Check that results are within reasonable bounds for linear approximation of quadratic
    try testing.expect(result[0] >= 0.0 and result[0] <= 0.04);    // f(0.1) = 0.01, linear should approximate
    try testing.expect(result[1] >= 0.04 and result[1] <= 0.16);   // f(0.3) = 0.09
    try testing.expect(result[2] >= 0.16 and result[2] <= 0.36);   // f(0.5) = 0.25
    try testing.expect(result[3] >= 0.36 and result[3] <= 0.64);   // f(0.7) = 0.49
    try testing.expect(result[4] >= 0.64 and result[4] <= 1.0);    // f(0.9) = 0.81
}

test "interp1d single interval - two point interpolation" {
    const allocator = testing.allocator;

    const x = [_]f64{ 0.0, 10.0 };
    const y = [_]f64{ 5.0, 15.0 };

    const x_new = [_]f64{ 5.0 };

    const result = try interp1d(f64, &x, &y, &x_new, allocator);
    defer allocator.free(result);

    // Midpoint: (5, 10)
    try testing.expectApproxEqAbs(result[0], 10.0, 1e-10);
}

test "interp1d multiple query points" {
    const allocator = testing.allocator;

    const x = [_]f64{ 0.0, 1.0, 2.0 };
    const y = [_]f64{ 0.0, 1.0, 4.0 };

    const x_new = [_]f64{ 0.2, 0.4, 0.6, 0.8, 1.1, 1.3, 1.5, 1.7, 1.9 };

    const result = try interp1d(f64, &x, &y, &x_new, allocator);
    defer allocator.free(result);

    try testing.expect(result.len == 9);
    // All results should be within bounds
    for (result) |val| {
        try testing.expect(!math.isNan(val));
    }
}

test "interp1d extrapolation below minimum" {
    const allocator = testing.allocator;

    const x = [_]f64{ 1.0, 2.0, 3.0 };
    const y = [_]f64{ 10.0, 20.0, 30.0 };

    const x_new = [_]f64{ -1.0, 0.0, 0.5 };

    const result = try interp1d(f64, &x, &y, &x_new, allocator);
    defer allocator.free(result);

    // All should be clamped to y[0] = 10.0
    try testing.expectApproxEqAbs(result[0], 10.0, 1e-10);
    try testing.expectApproxEqAbs(result[1], 10.0, 1e-10);
    try testing.expectApproxEqAbs(result[2], 10.0, 1e-10);
}

test "interp1d extrapolation above maximum" {
    const allocator = testing.allocator;

    const x = [_]f64{ 0.0, 1.0, 2.0 };
    const y = [_]f64{ 0.0, 10.0, 20.0 };

    const x_new = [_]f64{ 2.5, 3.0, 10.0 };

    const result = try interp1d(f64, &x, &y, &x_new, allocator);
    defer allocator.free(result);

    // All should be clamped to y[n-1] = 20.0
    try testing.expectApproxEqAbs(result[0], 20.0, 1e-10);
    try testing.expectApproxEqAbs(result[1], 20.0, 1e-10);
    try testing.expectApproxEqAbs(result[2], 20.0, 1e-10);
}

test "interp1d mixed interpolation and extrapolation" {
    const allocator = testing.allocator;

    const x = [_]f64{ 0.0, 1.0, 2.0 };
    const y = [_]f64{ 0.0, 10.0, 20.0 };

    const x_new = [_]f64{ -1.0, 0.5, 1.0, 1.5, 3.0 };

    const result = try interp1d(f64, &x, &y, &x_new, allocator);
    defer allocator.free(result);

    try testing.expectApproxEqAbs(result[0], 0.0, 1e-10);      // below min
    try testing.expectApproxEqAbs(result[1], 5.0, 1e-10);      // interpolated
    try testing.expectApproxEqAbs(result[2], 10.0, 1e-10);     // exact
    try testing.expectApproxEqAbs(result[3], 15.0, 1e-10);     // interpolated
    try testing.expectApproxEqAbs(result[4], 20.0, 1e-10);     // above max
}

test "interp1d non-uniform grid" {
    const allocator = testing.allocator;

    // Non-uniform spacing
    const x = [_]f64{ 0.0, 0.1, 0.5, 2.0, 5.0 };
    const y = [_]f64{ 0.0, 0.01, 0.25, 4.0, 25.0 };

    const x_new = [_]f64{ 0.05, 0.3, 1.0, 3.0 };

    const result = try interp1d(f64, &x, &y, &x_new, allocator);
    defer allocator.free(result);

    // Check intermediate points are reasonable
    try testing.expect(result[0] > 0.0 and result[0] < 0.1);  // between 0 and 0.1
    try testing.expect(result[1] > 0.01 and result[1] < 0.25); // between 0.01 and 0.25
    try testing.expect(result[2] > 0.25 and result[2] < 4.0);  // between 0.25 and 4.0
    try testing.expect(result[3] > 4.0 and result[3] < 25.0);  // between 4.0 and 25.0
}

test "interp1d reversed query order" {
    const allocator = testing.allocator;

    const x = [_]f64{ 0.0, 1.0, 2.0 };
    const y = [_]f64{ 0.0, 1.0, 4.0 };

    // Query points not sorted
    const x_new = [_]f64{ 1.5, 0.5, 1.0 };

    const result = try interp1d(f64, &x, &y, &x_new, allocator);
    defer allocator.free(result);

    // Should still work correctly for each query point independently
    try testing.expectApproxEqAbs(result[0], 2.5, 1e-10);  // 0.5 * (1 + 4) = 2.5
    try testing.expectApproxEqAbs(result[1], 0.5, 1e-10);  // 0.5 * (0 + 1) = 0.5
    try testing.expectApproxEqAbs(result[2], 1.0, 1e-10);  // exact
}

test "interp1d empty query points" {
    const allocator = testing.allocator;

    const x = [_]f64{ 0.0, 1.0, 2.0 };
    const y = [_]f64{ 0.0, 1.0, 4.0 };
    const x_new = [_]f64{};

    const result = try interp1d(f64, &x, &y, &x_new, allocator);
    defer allocator.free(result);

    try testing.expectEqual(result.len, 0);
}

test "interp1d dimension mismatch error" {
    const allocator = testing.allocator;

    const x = [_]f64{ 0.0, 1.0, 2.0 };
    const y = [_]f64{ 0.0, 1.0 };  // Wrong length
    const x_new = [_]f64{ 0.5 };

    const result = interp1d(f64, &x, &y, &x_new, allocator);
    try testing.expectError(error.DimensionMismatch, result);
}

test "interp1d insufficient points error" {
    const allocator = testing.allocator;

    const x = [_]f64{ 0.0 };  // Only one point
    const y = [_]f64{ 1.0 };
    const x_new = [_]f64{ 0.5 };

    const result = interp1d(f64, &x, &y, &x_new, allocator);
    try testing.expectError(error.InsufficientPoints, result);
}

test "interp1d non-monotonic x error" {
    const allocator = testing.allocator;

    const x = [_]f64{ 0.0, 2.0, 1.0 };  // Not monotonically increasing
    const y = [_]f64{ 0.0, 4.0, 1.0 };
    const x_new = [_]f64{ 0.5 };

    const result = interp1d(f64, &x, &y, &x_new, allocator);
    try testing.expectError(error.NonMonotonicX, result);
}

test "interp1d f32 precision" {
    const allocator = testing.allocator;

    const x = [_]f32{ 0.0, 1.0, 2.0 };
    const y = [_]f32{ 0.0, 1.0, 4.0 };
    const x_new = [_]f32{ 0.5, 1.5 };

    const result = try interp1d(f32, &x, &y, &x_new, allocator);
    defer allocator.free(result);

    try testing.expectApproxEqAbs(result[0], 0.5, 1e-5);
    try testing.expectApproxEqAbs(result[1], 2.5, 1e-5);
}

test "interp1d f64 precision" {
    const allocator = testing.allocator;

    const x = [_]f64{ 0.0, 1.0, 2.0 };
    const y = [_]f64{ 0.0, 1.0, 4.0 };
    const x_new = [_]f64{ 0.5, 1.5 };

    const result = try interp1d(f64, &x, &y, &x_new, allocator);
    defer allocator.free(result);

    try testing.expectApproxEqAbs(result[0], 0.5, 1e-10);
    try testing.expectApproxEqAbs(result[1], 2.5, 1e-10);
}

test "interp1d memory ownership - caller frees" {
    const allocator = testing.allocator;

    const x = [_]f64{ 0.0, 1.0 };
    const y = [_]f64{ 0.0, 1.0 };
    const x_new = [_]f64{ 0.5 };

    const result = try interp1d(f64, &x, &y, &x_new, allocator);

    // Caller owns the result and must free it
    allocator.free(result);
}

test "interp1d negative values" {
    const allocator = testing.allocator;

    const x = [_]f64{ -2.0, -1.0, 0.0, 1.0 };
    const y = [_]f64{ -4.0, -1.0, 0.0, 1.0 };

    const x_new = [_]f64{ -1.5, -0.5, 0.5 };

    const result = try interp1d(f64, &x, &y, &x_new, allocator);
    defer allocator.free(result);

    // Check reasonable results for negative domain
    try testing.expectApproxEqAbs(result[0], -2.5, 1e-10);
    try testing.expectApproxEqAbs(result[1], -0.5, 1e-10);
    try testing.expectApproxEqAbs(result[2], 0.5, 1e-10);
}

test "interp1d large dataset" {
    const allocator = testing.allocator;

    const n = 1001;
    var x: [1001]f64 = undefined;
    var y: [1001]f64 = undefined;

    // Generate sample points: f(x) = sin(x)
    for (0..n) |i| {
        const i_f: f64 = @floatFromInt(i);
        const n_f: f64 = @floatFromInt(n - 1);
        x[i] = i_f / n_f * math.pi;
        y[i] = @sin(x[i]);
    }

    // Query at many points
    var x_new_buf: [500]f64 = undefined;
    for (0..500) |i| {
        const i_f: f64 = @floatFromInt(i);
        x_new_buf[i] = i_f / 500.0 * math.pi;
    }

    const result = try interp1d(f64, &x, &y, x_new_buf[0..], allocator);
    defer allocator.free(result);

    try testing.expectEqual(result.len, 500);

    // Check that results are reasonable
    for (result) |val| {
        try testing.expect(val >= -1.1 and val <= 1.1);  // sin is bounded by [-1, 1]
    }
}

test "interp1d constant function" {
    const allocator = testing.allocator;

    // f(x) = 5 for all x
    const x = [_]f64{ 0.0, 1.0, 2.0, 3.0 };
    const y = [_]f64{ 5.0, 5.0, 5.0, 5.0 };

    const x_new = [_]f64{ 0.25, 0.75, 1.5, 2.99 };

    const result = try interp1d(f64, &x, &y, &x_new, allocator);
    defer allocator.free(result);

    // All should be 5.0
    for (result) |val| {
        try testing.expectApproxEqAbs(val, 5.0, 1e-10);
    }
}

test "interp1d exponential approximation" {
    const allocator = testing.allocator;

    // f(x) = e^x (approximated)
    const x = [_]f64{ 0.0, 0.5, 1.0, 1.5 };
    var y: [4]f64 = undefined;
    for (0..4) |i| {
        y[i] = @exp(x[i]);
    }

    const x_new = [_]f64{ 0.25, 0.75, 1.25 };

    const result = try interp1d(f64, &x, &y, &x_new, allocator);
    defer allocator.free(result);

    // Check approximations
    const approx_e_025 = @exp(0.25);
    const approx_e_075 = @exp(0.75);
    const approx_e_125 = @exp(1.25);

    try testing.expectApproxEqRel(result[0], approx_e_025, 0.05);  // 5% tolerance
    try testing.expectApproxEqRel(result[1], approx_e_075, 0.05);
    try testing.expectApproxEqRel(result[2], approx_e_125, 0.05);
}

test "interp1d monotonic x validation - equal values rejected" {
    const allocator = testing.allocator;

    const x = [_]f64{ 0.0, 1.0, 1.0, 2.0 };  // Duplicate value
    const y = [_]f64{ 0.0, 1.0, 1.0, 4.0 };
    const x_new = [_]f64{ 0.5 };

    const result = interp1d(f64, &x, &y, &x_new, allocator);
    try testing.expectError(error.NonMonotonicX, result);
}

test "interp1d query at boundaries" {
    const allocator = testing.allocator;

    const x = [_]f64{ 1.0, 2.0, 3.0 };
    const y = [_]f64{ 10.0, 20.0, 30.0 };

    const x_new = [_]f64{ 1.0, 3.0 };

    const result = try interp1d(f64, &x, &y, &x_new, allocator);
    defer allocator.free(result);

    // Boundary queries should return exact values
    try testing.expectApproxEqAbs(result[0], 10.0, 1e-10);
    try testing.expectApproxEqAbs(result[1], 30.0, 1e-10);
}

test "interp1d very small spacing" {
    const allocator = testing.allocator;

    const x = [_]f64{ 0.0, 1e-6, 2e-6 };
    const y = [_]f64{ 0.0, 1e-6, 4e-6 };

    const x_new = [_]f64{ 0.5e-6, 1.5e-6 };

    const result = try interp1d(f64, &x, &y, &x_new, allocator);
    defer allocator.free(result);

    try testing.expectApproxEqAbs(result[0], 0.5e-6, 1e-15);
    try testing.expectApproxEqAbs(result[1], 2.5e-6, 1e-15);
}

test "interp1d large spacing" {
    const allocator = testing.allocator;

    const x = [_]f64{ 0.0, 1e6, 2e6 };
    const y = [_]f64{ 0.0, 1e6, 4e6 };

    const x_new = [_]f64{ 0.5e6, 1.5e6 };

    const result = try interp1d(f64, &x, &y, &x_new, allocator);
    defer allocator.free(result);

    try testing.expectApproxEqAbs(result[0], 0.5e6, 1e-5);
    try testing.expectApproxEqAbs(result[1], 2.5e6, 1e-5);
}
