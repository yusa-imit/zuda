const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Polynomial evaluation and interpolation algorithms.
///
/// Fundamental operations for working with polynomials in various applications
/// including numerical analysis, signal processing, and computer graphics.

/// Evaluate a polynomial at a given point using Horner's method.
///
/// Given polynomial P(x) = a₀ + a₁x + a₂x² + ... + aₙxⁿ
/// represented by coefficients [a₀, a₁, a₂, ..., aₙ],
/// compute P(x) efficiently using nested multiplication.
///
/// Time: O(n) where n = degree of polynomial
/// Space: O(1)
///
/// # Arguments
///
/// * `T` - Floating point type (f32 or f64)
/// * `coeffs` - Polynomial coefficients in ascending degree order (a₀, a₁, ...)
/// * `x` - Point at which to evaluate the polynomial
///
/// # Returns
///
/// * The value P(x)
///
/// # Example
///
/// ```zig
/// // Evaluate P(x) = 1 + 2x + 3x² at x = 2
/// // P(2) = 1 + 2(2) + 3(4) = 1 + 4 + 12 = 17
/// const coeffs = [_]f64{ 1, 2, 3 };
/// const result = horner(f64, &coeffs, 2.0);
/// // result = 17.0
/// ```
pub fn horner(comptime T: type, coeffs: []const T, x: T) T {
    if (coeffs.len == 0) return 0;

    var result = coeffs[coeffs.len - 1];
    var i: usize = coeffs.len - 1;
    while (i > 0) {
        i -= 1;
        result = result * x + coeffs[i];
    }
    return result;
}

/// Evaluate polynomial and its derivative using Horner's method.
///
/// Simultaneously computes P(x) and P'(x) in a single pass.
/// Useful for Newton-Raphson iteration and other root-finding methods.
///
/// Time: O(n)
/// Space: O(1)
///
/// # Returns
///
/// * Tuple of (value, derivative) at point x
pub fn hornerWithDerivative(comptime T: type, coeffs: []const T, x: T) struct { value: T, derivative: T } {
    if (coeffs.len == 0) return .{ .value = 0, .derivative = 0 };
    if (coeffs.len == 1) return .{ .value = coeffs[0], .derivative = 0 };

    var value = coeffs[coeffs.len - 1];
    var deriv: T = 0;

    var i: usize = coeffs.len - 1;
    while (i > 0) {
        i -= 1;
        deriv = deriv * x + value;
        value = value * x + coeffs[i];
    }

    return .{ .value = value, .derivative = deriv };
}

/// Lagrange polynomial interpolation.
///
/// Given n points (x₀, y₀), (x₁, y₁), ..., (xₙ₋₁, yₙ₋₁),
/// compute the unique polynomial of degree at most n-1 that passes through all points.
/// Evaluates the interpolating polynomial at a given point.
///
/// Time: O(n²) for evaluation
/// Space: O(1)
///
/// # Arguments
///
/// * `T` - Floating point type
/// * `xs` - x-coordinates of interpolation points
/// * `ys` - y-coordinates of interpolation points
/// * `x` - Point at which to evaluate interpolating polynomial
///
/// # Returns
///
/// * Value of interpolating polynomial at x
/// * `error.InvalidArguments` if xs and ys have different lengths
/// * `error.EmptyInput` if no points provided
/// * `error.DuplicatePoints` if x-coordinates are not distinct
///
/// # Example
///
/// ```zig
/// // Interpolate through points (1, 2), (2, 5), (3, 10)
/// const xs = [_]f64{ 1, 2, 3 };
/// const ys = [_]f64{ 2, 5, 10 };
/// const result = try lagrangeInterpolate(f64, &xs, &ys, 2.5);
/// // result ≈ 7.25
/// ```
pub fn lagrangeInterpolate(comptime T: type, xs: []const T, ys: []const T, x: T) !T {
    if (xs.len == 0) return error.EmptyInput;
    if (xs.len != ys.len) return error.InvalidArguments;

    // Check for duplicate x-coordinates
    for (xs, 0..) |xi, i| {
        for (xs[i + 1..]) |xj| {
            if (@abs(xi - xj) < std.math.floatEps(T)) {
                return error.DuplicatePoints;
            }
        }
    }

    var result: T = 0;

    // L(x) = Σᵢ yᵢ · Lᵢ(x) where Lᵢ(x) = Πⱼ≠ᵢ (x - xⱼ) / (xᵢ - xⱼ)
    for (xs, ys, 0..) |xi, yi, i| {
        var term: T = 1;
        for (xs, 0..) |xj, j| {
            if (i != j) {
                term *= (x - xj) / (xi - xj);
            }
        }
        result += yi * term;
    }

    return result;
}

/// Newton's divided differences for polynomial interpolation.
///
/// Constructs divided difference table for Newton interpolation polynomial.
/// More numerically stable than Lagrange form for higher degrees.
///
/// Time: O(n²) for table construction
/// Space: O(n) for divided differences
///
/// # Arguments
///
/// * `T` - Floating point type
/// * `allocator` - Memory allocator for divided differences
/// * `xs` - x-coordinates (must be distinct)
/// * `ys` - y-coordinates
///
/// # Returns
///
/// * Slice of divided differences [f[x₀], f[x₀,x₁], f[x₀,x₁,x₂], ...]
/// * Caller owns the returned slice
pub fn newtonDividedDifferences(comptime T: type, allocator: Allocator, xs: []const T, ys: []const T) ![]T {
    if (xs.len == 0) return error.EmptyInput;
    if (xs.len != ys.len) return error.InvalidArguments;

    const n = xs.len;
    var diffs = try allocator.alloc(T, n);
    errdefer allocator.free(diffs);

    // Initialize with y values
    @memcpy(diffs, ys);

    // Compute divided differences table
    for (1..n) |k| {
        var i: usize = n - 1;
        while (i >= k) : (i -= 1) {
            diffs[i] = (diffs[i] - diffs[i - 1]) / (xs[i] - xs[i - k]);
            if (i == k) break;
        }
    }

    return diffs;
}

/// Evaluate Newton interpolation polynomial.
///
/// Uses precomputed divided differences to evaluate the interpolating polynomial.
/// More efficient than Lagrange when evaluating at multiple points.
///
/// Time: O(n)
/// Space: O(1)
///
/// # Arguments
///
/// * `T` - Floating point type
/// * `xs` - x-coordinates of interpolation points
/// * `diffs` - Divided differences from newtonDividedDifferences
/// * `x` - Point at which to evaluate
///
/// # Returns
///
/// * Value of Newton polynomial at x
pub fn newtonEvaluate(comptime T: type, xs: []const T, diffs: []const T, x: T) T {
    if (diffs.len == 0) return 0;

    var result = diffs[0];
    var term: T = 1;

    for (1..diffs.len) |i| {
        term *= (x - xs[i - 1]);
        result += diffs[i] * term;
    }

    return result;
}

/// Polynomial addition.
///
/// Add two polynomials represented as coefficient arrays.
/// Result degree = max(deg(p), deg(q)).
///
/// Time: O(n) where n = max degree
/// Space: O(n)
///
/// # Returns
///
/// * Coefficient array of sum polynomial
/// * Caller owns the returned slice
pub fn add(comptime T: type, allocator: Allocator, p: []const T, q: []const T) ![]T {
    const max_len = @max(p.len, q.len);
    var result = try allocator.alloc(T, max_len);
    errdefer allocator.free(result);

    @memset(result, 0);

    for (p, 0..) |coeff, i| {
        result[i] += coeff;
    }
    for (q, 0..) |coeff, i| {
        result[i] += coeff;
    }

    return result;
}

/// Polynomial multiplication.
///
/// Multiply two polynomials using standard convolution.
/// Result degree = deg(p) + deg(q).
///
/// Time: O(nm) where n, m are degrees
/// Space: O(n + m)
///
/// # Returns
///
/// * Coefficient array of product polynomial
/// * Caller owns the returned slice
pub fn multiply(comptime T: type, allocator: Allocator, p: []const T, q: []const T) ![]T {
    if (p.len == 0 or q.len == 0) {
        return try allocator.alloc(T, 0);
    }

    const result_len = p.len + q.len - 1;
    var result = try allocator.alloc(T, result_len);
    errdefer allocator.free(result);

    @memset(result, 0);

    for (p, 0..) |pi, i| {
        for (q, 0..) |qj, j| {
            result[i + j] += pi * qj;
        }
    }

    return result;
}

/// Polynomial derivative.
///
/// Compute derivative of polynomial P(x).
/// If P(x) = Σᵢ aᵢxⁱ, then P'(x) = Σᵢ iaᵢxⁱ⁻¹.
///
/// Time: O(n)
/// Space: O(n)
///
/// # Returns
///
/// * Coefficient array of derivative polynomial (degree reduced by 1)
/// * Caller owns the returned slice
pub fn derivative(comptime T: type, allocator: Allocator, coeffs: []const T) ![]T {
    if (coeffs.len <= 1) {
        return try allocator.alloc(T, 0);
    }

    const result = try allocator.alloc(T, coeffs.len - 1);
    errdefer allocator.free(result);

    for (result, 0..) |*r, i| {
        r.* = coeffs[i + 1] * @as(T, @floatFromInt(i + 1));
    }

    return result;
}

/// Polynomial integration (indefinite).
///
/// Compute antiderivative with constant term = 0.
/// If P(x) = Σᵢ aᵢxⁱ, then ∫P(x)dx = Σᵢ (aᵢ/(i+1))xⁱ⁺¹.
///
/// Time: O(n)
/// Space: O(n)
///
/// # Returns
///
/// * Coefficient array of antiderivative (degree increased by 1)
/// * Caller owns the returned slice
pub fn integrate(comptime T: type, allocator: Allocator, coeffs: []const T) ![]T {
    const result = try allocator.alloc(T, coeffs.len + 1);
    errdefer allocator.free(result);

    result[0] = 0; // Constant of integration
    for (coeffs, 0..) |coeff, i| {
        result[i + 1] = coeff / @as(T, @floatFromInt(i + 1));
    }

    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "horner: constant polynomial" {
    const coeffs = [_]f64{5.0};
    try testing.expectEqual(5.0, horner(f64, &coeffs, 10.0));
}

test "horner: linear polynomial" {
    // P(x) = 2 + 3x
    const coeffs = [_]f64{ 2.0, 3.0 };
    try testing.expectEqual(11.0, horner(f64, &coeffs, 3.0)); // 2 + 3*3 = 11
}

test "horner: quadratic polynomial" {
    // P(x) = 1 + 2x + 3x²
    const coeffs = [_]f64{ 1.0, 2.0, 3.0 };
    try testing.expectEqual(17.0, horner(f64, &coeffs, 2.0)); // 1 + 4 + 12 = 17
}

test "horner: cubic at zero" {
    const coeffs = [_]f64{ 5.0, 0.0, 0.0, 1.0 };
    try testing.expectEqual(5.0, horner(f64, &coeffs, 0.0));
}

test "horner: empty coefficients" {
    const coeffs = [_]f64{};
    try testing.expectEqual(0.0, horner(f64, &coeffs, 5.0));
}

test "hornerWithDerivative: constant" {
    const coeffs = [_]f64{5.0};
    const result = hornerWithDerivative(f64, &coeffs, 2.0);
    try testing.expectEqual(5.0, result.value);
    try testing.expectEqual(0.0, result.derivative);
}

test "hornerWithDerivative: linear" {
    // P(x) = 2 + 3x, P'(x) = 3
    const coeffs = [_]f64{ 2.0, 3.0 };
    const result = hornerWithDerivative(f64, &coeffs, 5.0);
    try testing.expectEqual(17.0, result.value); // 2 + 3*5
    try testing.expectEqual(3.0, result.derivative);
}

test "hornerWithDerivative: quadratic" {
    // P(x) = 1 + 2x + 3x², P'(x) = 2 + 6x
    const coeffs = [_]f64{ 1.0, 2.0, 3.0 };
    const result = hornerWithDerivative(f64, &coeffs, 2.0);
    try testing.expectEqual(17.0, result.value); // 1 + 4 + 12
    try testing.expectEqual(14.0, result.derivative); // 2 + 12
}

test "lagrangeInterpolate: two points (linear)" {
    const xs = [_]f64{ 1.0, 3.0 };
    const ys = [_]f64{ 2.0, 6.0 };
    // Line through (1,2) and (3,6): y = 2x
    const result = try lagrangeInterpolate(f64, &xs, &ys, 2.0);
    try testing.expectApproxEqAbs(4.0, result, 1e-10);
}

test "lagrangeInterpolate: three points (quadratic)" {
    const xs = [_]f64{ 1.0, 2.0, 3.0 };
    const ys = [_]f64{ 2.0, 5.0, 10.0 };
    // Interpolate at x=2.5
    const result = try lagrangeInterpolate(f64, &xs, &ys, 2.5);
    try testing.expectApproxEqAbs(7.25, result, 1e-10);
}

test "lagrangeInterpolate: at known point" {
    const xs = [_]f64{ 0.0, 1.0, 2.0 };
    const ys = [_]f64{ 1.0, 4.0, 9.0 };
    const result = try lagrangeInterpolate(f64, &xs, &ys, 1.0);
    try testing.expectApproxEqAbs(4.0, result, 1e-10);
}

test "lagrangeInterpolate: errors" {
    const xs = [_]f64{ 1.0, 2.0 };
    const ys = [_]f64{ 2.0, 5.0, 10.0 };
    try testing.expectError(error.InvalidArguments, lagrangeInterpolate(f64, &xs, &ys, 1.5));

    const empty = [_]f64{};
    try testing.expectError(error.EmptyInput, lagrangeInterpolate(f64, &empty, &empty, 0.0));

    const dup_xs = [_]f64{ 1.0, 1.0 };
    const dup_ys = [_]f64{ 2.0, 3.0 };
    try testing.expectError(error.DuplicatePoints, lagrangeInterpolate(f64, &dup_xs, &dup_ys, 1.0));
}

test "newtonDividedDifferences: linear" {
    const xs = [_]f64{ 1.0, 3.0 };
    const ys = [_]f64{ 2.0, 6.0 };
    const diffs = try newtonDividedDifferences(f64, testing.allocator, &xs, &ys);
    defer testing.allocator.free(diffs);

    // f[x₀] = 2, f[x₀,x₁] = (6-2)/(3-1) = 2
    try testing.expectEqual(2.0, diffs[0]);
    try testing.expectEqual(2.0, diffs[1]);
}

test "newtonDividedDifferences: quadratic" {
    const xs = [_]f64{ 1.0, 2.0, 3.0 };
    const ys = [_]f64{ 1.0, 4.0, 9.0 };
    const diffs = try newtonDividedDifferences(f64, testing.allocator, &xs, &ys);
    defer testing.allocator.free(diffs);

    // f[x₀] = 1
    // f[x₀,x₁] = (4-1)/(2-1) = 3
    // f[x₀,x₁,x₂] = ((9-4)/(3-2) - 3)/(3-1) = (5-3)/2 = 1
    try testing.expectEqual(1.0, diffs[0]);
    try testing.expectEqual(3.0, diffs[1]);
    try testing.expectEqual(1.0, diffs[2]);
}

test "newtonEvaluate: matches lagrange" {
    const xs = [_]f64{ 1.0, 2.0, 3.0 };
    const ys = [_]f64{ 2.0, 5.0, 10.0 };

    const diffs = try newtonDividedDifferences(f64, testing.allocator, &xs, &ys);
    defer testing.allocator.free(diffs);

    const x = 2.5;
    const newton_val = newtonEvaluate(f64, &xs, diffs, x);
    const lagrange_val = try lagrangeInterpolate(f64, &xs, &ys, x);

    try testing.expectApproxEqAbs(lagrange_val, newton_val, 1e-10);
}

test "add: same degree" {
    // (1 + 2x) + (3 + 4x) = 4 + 6x
    const p = [_]f64{ 1.0, 2.0 };
    const q = [_]f64{ 3.0, 4.0 };
    const result = try add(f64, testing.allocator, &p, &q);
    defer testing.allocator.free(result);

    try testing.expectEqual(2, result.len);
    try testing.expectEqual(4.0, result[0]);
    try testing.expectEqual(6.0, result[1]);
}

test "add: different degrees" {
    // (1 + 2x) + (3 + 4x + 5x²) = 4 + 6x + 5x²
    const p = [_]f64{ 1.0, 2.0 };
    const q = [_]f64{ 3.0, 4.0, 5.0 };
    const result = try add(f64, testing.allocator, &p, &q);
    defer testing.allocator.free(result);

    try testing.expectEqual(3, result.len);
    try testing.expectEqual(4.0, result[0]);
    try testing.expectEqual(6.0, result[1]);
    try testing.expectEqual(5.0, result[2]);
}

test "multiply: linear times linear" {
    // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x²
    const p = [_]f64{ 1.0, 2.0 };
    const q = [_]f64{ 3.0, 4.0 };
    const result = try multiply(f64, testing.allocator, &p, &q);
    defer testing.allocator.free(result);

    try testing.expectEqual(3, result.len);
    try testing.expectEqual(3.0, result[0]);
    try testing.expectEqual(10.0, result[1]);
    try testing.expectEqual(8.0, result[2]);
}

test "multiply: empty polynomial" {
    const p = [_]f64{ 1.0, 2.0 };
    const q = [_]f64{};
    const result = try multiply(f64, testing.allocator, &p, &q);
    defer testing.allocator.free(result);

    try testing.expectEqual(0, result.len);
}

test "derivative: quadratic" {
    // P(x) = 1 + 2x + 3x², P'(x) = 2 + 6x
    const coeffs = [_]f64{ 1.0, 2.0, 3.0 };
    const result = try derivative(f64, testing.allocator, &coeffs);
    defer testing.allocator.free(result);

    try testing.expectEqual(2, result.len);
    try testing.expectEqual(2.0, result[0]);
    try testing.expectEqual(6.0, result[1]);
}

test "derivative: constant" {
    const coeffs = [_]f64{5.0};
    const result = try derivative(f64, testing.allocator, &coeffs);
    defer testing.allocator.free(result);

    try testing.expectEqual(0, result.len);
}

test "integrate: linear" {
    // P(x) = 2 + 3x, ∫P(x)dx = 2x + (3/2)x²
    const coeffs = [_]f64{ 2.0, 3.0 };
    const result = try integrate(f64, testing.allocator, &coeffs);
    defer testing.allocator.free(result);

    try testing.expectEqual(3, result.len);
    try testing.expectEqual(0.0, result[0]);
    try testing.expectEqual(2.0, result[1]);
    try testing.expectEqual(1.5, result[2]);
}

test "polynomial operations: f32 support" {
    const p = [_]f32{ 1.0, 2.0 };
    const q = [_]f32{ 3.0, 4.0 };

    const sum = try add(f32, testing.allocator, &p, &q);
    defer testing.allocator.free(sum);
    try testing.expectEqual(@as(usize, 2), sum.len);

    const prod = try multiply(f32, testing.allocator, &p, &q);
    defer testing.allocator.free(prod);
    try testing.expectEqual(@as(usize, 3), prod.len);
}

test "polynomial operations: memory safety" {
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const p = [_]f64{ 1.0, 2.0, 3.0 };
        const q = [_]f64{ 4.0, 5.0 };

        const sum = try add(f64, testing.allocator, &p, &q);
        testing.allocator.free(sum);

        const prod = try multiply(f64, testing.allocator, &p, &q);
        testing.allocator.free(prod);

        const deriv = try derivative(f64, testing.allocator, &p);
        testing.allocator.free(deriv);

        const integ = try integrate(f64, testing.allocator, &q);
        testing.allocator.free(integ);
    }
}
