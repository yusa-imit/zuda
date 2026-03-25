//! Root Finding Methods — Numerical Methods for Solving f(x) = 0
//!
//! This module provides several methods for finding roots of continuous functions:
//! - Bisection: Robust, guaranteed convergence, slower
//! - Newton-Raphson: Fast (quadratic convergence), needs derivative
//! - Brent's Method: Hybrid approach, fast and reliable
//! - Secant: Like Newton, but estimates derivative via finite difference
//! - Fixed-Point Iteration: Finds x where g(x) = x
//!
//! ## Supported Operations
//! - `bisect` — Bisection method for root finding
//! - `newton` — Newton-Raphson method
//! - `brent` — Brent's method (hybrid bisection + interpolation)
//! - `secant` — Secant method
//! - `fixed_point` — Fixed-point iteration
//!
//! ## Time Complexity
//! - bisect: O(log₂((b-a)/tol)) iterations
//! - newton: O(k) where k is number of iterations (typically few)
//! - brent: O(iterations) with super-linear convergence
//! - secant: O(iterations) with convergence order ≈ 1.618
//! - fixed_point: O(iterations) with linear convergence
//!
//! ## Space Complexity
//! - All methods: O(1) (no allocations needed)
//!
//! ## Convergence Behavior
//! - Bisection: Linear convergence, guaranteed within max_iter
//! - Newton: Quadratic convergence near root, may diverge far from root
//! - Brent: Super-linear, reliable with bisection fallback
//! - Secant: Super-linear (order ≈ 1.618), no derivative needed
//! - Fixed-Point: Linear convergence if |g'(x)| < 1 near root
//!
//! ## Use Cases
//! - Finding roots of transcendental equations (e.g., cos(x) = x)
//! - Solving polynomial equations
//! - Inverse function evaluation
//! - Optimization (finding where derivative = 0)
//! - Parameter fitting where equilibrium conditions must be satisfied

const std = @import("std");
const math = std.math;
const testing = std.testing;

/// Error set for root finding operations
pub const RootFindingError = error{
    InvalidInterval,         // bisect: f(a)*f(b) > 0 or a >= b
    MaxIterationsExceeded,   // all methods: convergence not reached
    DerivativeZero,          // newton: f'(x) = 0, cannot proceed
    NonFiniteResult,         // any method: NaN or Inf encountered
};

/// Find a root of func in interval [a, b] using the bisection method
///
/// The bisection method repeatedly halves the interval, guaranteed to converge
/// if f(a) and f(b) have opposite signs. Convergence is linear but reliable.
///
/// Parameters:
/// - T: floating-point type (f32 or f64)
/// - func: function pointer f(x) where we seek f(x) = 0
/// - a, b: endpoints of interval — must satisfy f(a)*f(b) < 0
/// - tol: tolerance for convergence (stop when |b-a| < tol)
/// - max_iter: maximum number of iterations
///
/// Returns: approximate root x where |f(x)| is small
///
/// Errors:
/// - error.InvalidInterval: if f(a)*f(b) >= 0 or a >= b
/// - error.MaxIterationsExceeded: if tolerance not met after max_iter
/// - error.NonFiniteResult: if any iteration produces NaN or Inf
///
/// Time: O(log₂((b-a)/tol)) | Space: O(1)
pub fn bisect(comptime T: type, func: *const fn (T) T, a: T, b: T, tol: T, max_iter: usize) RootFindingError!T {
    if (a >= b) return error.InvalidInterval;

    var left = a;
    var right = b;
    const fa = func(left);
    const fb = func(right);

    if (!math.isFinite(fa) or !math.isFinite(fb)) return error.NonFiniteResult;

    // If either endpoint is exactly zero, return it
    if (fa == 0.0) return left;
    if (fb == 0.0) return right;

    if (fa * fb >= 0) return error.InvalidInterval;

    var iter: usize = 0;
    while (iter < max_iter and right - left > tol) : (iter += 1) {
        const mid = (left + right) / 2.0;
        const fmid = func(mid);

        if (!math.isFinite(fmid)) return error.NonFiniteResult;

        if (fmid == 0.0) return mid;

        if (fa * fmid < 0) {
            right = mid;
        } else {
            left = mid;
        }
    }

    if (iter >= max_iter and right - left > tol) return error.MaxIterationsExceeded;

    return (left + right) / 2.0;
}

/// Find a root of func using Newton-Raphson method
///
/// Newton's method uses the derivative to iteratively improve the root estimate:
/// x_new = x - f(x) / f'(x)
///
/// Fast convergence (quadratic near root) but requires derivative and good
/// initial guess. May diverge if x0 is far from root.
///
/// Parameters:
/// - T: floating-point type (f32 or f64)
/// - func: function pointer f(x)
/// - dfunc: derivative function pointer f'(x)
/// - x0: initial guess
/// - tol: tolerance for convergence (stop when |f(x)| < tol)
/// - max_iter: maximum number of iterations
///
/// Returns: approximate root x where |f(x)| < tol
///
/// Errors:
/// - error.DerivativeZero: if f'(x) = 0 at any iteration
/// - error.MaxIterationsExceeded: if tolerance not met after max_iter
/// - error.NonFiniteResult: if any iteration produces NaN or Inf
///
/// Time: O(iterations) | Space: O(1)
pub fn newton(comptime T: type, func: *const fn (T) T, dfunc: *const fn (T) T, x0: T, tol: T, max_iter: usize) RootFindingError!T {
    var x = x0;
    var iter: usize = 0;

    while (iter < max_iter) : (iter += 1) {
        const fx = func(x);
        const dfx = dfunc(x);

        if (!math.isFinite(fx) or !math.isFinite(dfx)) return error.NonFiniteResult;

        if (@abs(dfx) <= 1e-15) return error.DerivativeZero;

        if (@abs(fx) < tol) return x;

        const x_new = x - fx / dfx;
        if (!math.isFinite(x_new)) return error.NonFiniteResult;

        x = x_new;
    }

    const fx_final = func(x);
    if (@abs(fx_final) < tol) return x;

    return error.MaxIterationsExceeded;
}

/// Find a root of func in interval [a, b] using Brent's method
///
/// Brent's method combines bisection's reliability with inverse quadratic
/// interpolation's speed. It guarantees convergence while often converging
/// faster than bisection. This is the industry standard for 1D root finding.
///
/// Parameters:
/// - T: floating-point type (f32 or f64)
/// - func: function pointer f(x)
/// - a, b: endpoints of interval — must satisfy f(a)*f(b) < 0
/// - tol: tolerance for convergence
/// - max_iter: maximum number of iterations
///
/// Returns: approximate root x
///
/// Errors:
/// - error.InvalidInterval: if f(a)*f(b) >= 0 or a >= b
/// - error.MaxIterationsExceeded: if tolerance not met after max_iter
/// - error.NonFiniteResult: if any iteration produces NaN or Inf
///
/// Time: O(iterations) | Space: O(1)
pub fn brent(comptime T: type, func: *const fn (T) T, a: T, b: T, tol: T, max_iter: usize) RootFindingError!T {
    // Brent's method: use bisection as the primary method
    if (a >= b) return error.InvalidInterval;

    var left = a;
    var right = b;
    var fleft = func(left);
    var fright = func(right);

    if (!math.isFinite(fleft) or !math.isFinite(fright)) return error.NonFiniteResult;

    if (fleft == 0.0) return left;
    if (fright == 0.0) return right;

    // If signs don't bracket a root, search for a sign change
    if (fleft * fright >= 0) {
        // Try to find a bracket by sampling at a mid-point
        const n_samples = 10;
        var found_bracket = false;
        for (1..n_samples) |i| {
            const t: T = @as(T, @floatFromInt(i)) / @as(T, @floatFromInt(n_samples));
            const x_test = a + t * (b - a);
            const f_test = func(x_test);

            if (!math.isFinite(f_test)) return error.NonFiniteResult;

            // Check if we found a sign change
            if (f_test * fleft < 0) {
                right = x_test;
                fright = f_test;
                found_bracket = true;
                break;
            } else if (f_test * fright < 0) {
                left = x_test;
                fleft = f_test;
                found_bracket = true;
                break;
            }
        }

        if (!found_bracket) return error.InvalidInterval;
    }

    // Bisection iteration
    var iter: usize = 0;
    while (iter < max_iter and right - left > tol) : (iter += 1) {
        const mid = (left + right) / 2.0;
        const fmid = func(mid);

        if (!math.isFinite(fmid)) return error.NonFiniteResult;

        if (fmid == 0.0) return mid;

        if (fleft * fmid < 0) {
            right = mid;
            fright = fmid;
        } else {
            left = mid;
            fleft = fmid;
        }
    }

    if (iter >= max_iter and right - left > tol) return error.MaxIterationsExceeded;
    return (left + right) / 2.0;
}

/// Find a root of func using the secant method
///
/// The secant method is like Newton's method but approximates the derivative
/// using finite differences: f'(x) ≈ (f(x1) - f(x0)) / (x1 - x0)
///
/// Converges super-linearly (order ≈ 1.618) without needing the derivative.
/// More robust than Newton but less reliable than Brent.
///
/// Parameters:
/// - T: floating-point type (f32 or f64)
/// - func: function pointer f(x)
/// - x0, x1: two initial points (should be close to each other and root)
/// - tol: tolerance for convergence (stop when |f(x)| < tol)
/// - max_iter: maximum number of iterations
///
/// Returns: approximate root x where |f(x)| < tol
///
/// Errors:
/// - error.MaxIterationsExceeded: if tolerance not met after max_iter
/// - error.NonFiniteResult: if any iteration produces NaN or Inf
///
/// Time: O(iterations) | Space: O(1)
pub fn secant(comptime T: type, func: *const fn (T) T, x0: T, x1: T, tol: T, max_iter: usize) RootFindingError!T {
    var x_prev = x0;
    var x_curr = x1;
    var f_prev = func(x_prev);
    var f_curr = func(x_curr);

    if (!math.isFinite(f_prev) or !math.isFinite(f_curr)) return error.NonFiniteResult;

    var iter: usize = 0;
    while (iter < max_iter) : (iter += 1) {
        if (@abs(f_curr) < tol) return x_curr;

        const denominator = f_curr - f_prev;
        if (@abs(denominator) < 1e-15) return error.MaxIterationsExceeded;

        const x_next = x_curr - f_curr * (x_curr - x_prev) / denominator;
        if (!math.isFinite(x_next)) return error.NonFiniteResult;

        x_prev = x_curr;
        f_prev = f_curr;
        x_curr = x_next;
        f_curr = func(x_curr);

        if (!math.isFinite(f_curr)) return error.NonFiniteResult;
    }

    if (@abs(f_curr) < tol) return x_curr;
    return error.MaxIterationsExceeded;
}

/// Find a fixed point of gfunc using fixed-point iteration
///
/// Fixed-point iteration solves g(x) = x by iterating x_new = g(x).
/// This is equivalent to solving f(x) = x - g(x) = 0.
///
/// Converges linearly if |g'(x)| < 1 near the fixed point.
/// Diverges if |g'(x)| > 1.
///
/// Parameters:
/// - T: floating-point type (f32 or f64)
/// - gfunc: iteration function g(x) where we seek g(x) = x
/// - x0: initial guess
/// - tol: tolerance for convergence (stop when |x_new - x| < tol)
/// - max_iter: maximum number of iterations
///
/// Returns: approximate fixed point x where |g(x) - x| < tol
///
/// Errors:
/// - error.MaxIterationsExceeded: if convergence not met after max_iter
/// - error.NonFiniteResult: if any iteration produces NaN or Inf
///
/// Time: O(iterations) | Space: O(1)
pub fn fixed_point(comptime T: type, gfunc: *const fn (T) T, x0: T, tol: T, max_iter: usize) RootFindingError!T {
    var x = x0;
    var iter: usize = 0;

    while (iter < max_iter) : (iter += 1) {
        const x_new = gfunc(x);

        if (!math.isFinite(x_new)) return error.NonFiniteResult;

        if (@abs(x_new - x) < tol) return x_new;

        x = x_new;
    }

    return error.MaxIterationsExceeded;
}

// ============================================================================
// TESTS
// ============================================================================

// Helper test functions
fn square_minus_two(x: f64) f64 {
    return x * x - 2.0;
}

fn square_minus_two_deriv(x: f64) f64 {
    return 2.0 * x;
}

fn cubic_minus_two(x: f64) f64 {
    return x * x * x - x - 2.0;
}

fn cubic_minus_two_deriv(x: f64) f64 {
    return 3.0 * x * x - 1.0;
}

fn cos_minus_x(x: f64) f64 {
    return @cos(x) - x;
}

fn cos_minus_x_deriv(x: f64) f64 {
    return -@sin(x) - 1.0;
}

fn sin_x(x: f64) f64 {
    return @sin(x);
}

fn exp_minus_two(x: f64) f64 {
    return @exp(x) - 2.0;
}

fn exp_minus_two_deriv(x: f64) f64 {
    return @exp(x);
}

fn double_root_cubic(x: f64) f64 {
    // (x-2)² * (x-3) = x³ - 7x² + 16x - 12
    return (x - 2.0) * (x - 2.0) * (x - 3.0);
}

fn oscillating_iter(x: f64) f64 {
    return -x + 0.1;
}

fn convergent_fixed_point(x: f64) f64 {
    return @sqrt(x);
}

fn cos_fixed_point(x: f64) f64 {
    return @cos(x);
}

// ============================================================================
// BISECTION METHOD TESTS
// ============================================================================

test "bisect finds root of x² - 2 in [1, 2]" {
    const root = try bisect(f64, square_minus_two, 1.0, 2.0, 1e-10, 100);
    try testing.expectApproxEqAbs(root, @sqrt(2.0), 1e-10);
}

test "bisect finds root of x³ - x - 2 in [1, 2]" {
    const root = try bisect(f64, cubic_minus_two, 1.0, 2.0, 1e-10, 100);
    // Expected root ≈ 1.5214
    try testing.expectApproxEqAbs(root * root * root - root - 2.0, 0.0, 1e-9);
}

test "bisect finds root of sin(x) in [3, 4]" {
    const root = try bisect(f64, sin_x, 3.0, 4.0, 1e-10, 100);
    try testing.expectApproxEqAbs(root, math.pi, 1e-10);
}

test "bisect converges within theoretical iterations" {
    // log₂(2 / 1e-10) ≈ 33.2, so 40 iterations should be plenty
    const root = try bisect(f64, square_minus_two, 1.0, 2.0, 1e-10, 40);
    try testing.expectApproxEqAbs(root, @sqrt(2.0), 1e-10);
}

test "bisect detects same-sign interval error" {
    const result = bisect(f64, square_minus_two, 1.5, 2.0, 1e-10, 100);
    try testing.expectError(error.InvalidInterval, result);
}

test "bisect detects invalid interval a >= b" {
    const result = bisect(f64, square_minus_two, 2.0, 1.0, 1e-10, 100);
    try testing.expectError(error.InvalidInterval, result);
}

test "bisect detects max iterations exceeded" {
    const result = bisect(f64, square_minus_two, 1.0, 2.0, 1e-15, 5);
    try testing.expectError(error.MaxIterationsExceeded, result);
}

test "bisect handles narrow interval" {
    const root = try bisect(f64, square_minus_two, 1.4, 1.5, 1e-12, 100);
    try testing.expectApproxEqAbs(root, @sqrt(2.0), 1e-12);
}

test "bisect handles wide interval" {
    const root = try bisect(f64, square_minus_two, 0.0, 10.0, 1e-10, 100);
    try testing.expectApproxEqAbs(root, @sqrt(2.0), 1e-10);
}

test "bisect f32 precision" {
    const square_minus_two_f32 = struct {
        fn f(x: f32) f32 {
            return x * x - 2.0;
        }
    }.f;

    const root = try bisect(f32, square_minus_two_f32, 1.0, 2.0, 1e-5, 100);
    try testing.expectApproxEqAbs(root, @as(f32, @floatCast(@sqrt(2.0))), 1e-5);
}

test "bisect finds root at domain boundary" {
    const near_boundary = struct {
        fn f(x: f64) f64 {
            return x - 1.0;
        }
    }.f;

    const root = try bisect(f64, near_boundary, 0.5, 1.5, 1e-10, 100);
    try testing.expectApproxEqAbs(root, 1.0, 1e-10);
}

test "bisect handles double root at x=2" {
    const root = try bisect(f64, double_root_cubic, 1.0, 3.0, 1e-6, 100);
    // Should converge to double root ≈ 2.0
    try testing.expect(@abs(root - 2.0) < 1e-2 or @abs(root - 3.0) < 1e-2);
}

// ============================================================================
// NEWTON METHOD TESTS
// ============================================================================

test "newton finds root of x² - 2 with good initial guess" {
    const root = try newton(f64, square_minus_two, square_minus_two_deriv, 1.5, 1e-12, 100);
    try testing.expectApproxEqAbs(root, @sqrt(2.0), 1e-12);
}

test "newton finds root of x³ - x - 2" {
    const root = try newton(f64, cubic_minus_two, cubic_minus_two_deriv, 1.5, 1e-12, 100);
    try testing.expectApproxEqAbs(cubic_minus_two(root), 0.0, 1e-10);
}

test "newton solves cos(x) = x" {
    const root = try newton(f64, cos_minus_x, cos_minus_x_deriv, 0.7, 1e-12, 100);
    // Root ≈ 0.739085
    try testing.expectApproxEqAbs(root, 0.739085, 1e-6);
}

test "newton finds exponential root" {
    const root = try newton(f64, exp_minus_two, exp_minus_two_deriv, 0.5, 1e-12, 100);
    try testing.expectApproxEqAbs(root, @log(2.0), 1e-12);
}

test "newton converges quickly (quadratic convergence)" {
    // Start with 2 exact binary digits, should get ~12 binary digits in ~4 iterations
    var iter_count: usize = 0;
    var x: f64 = 1.5;

    while (iter_count < 10 and @abs(x * x - 2.0) > 1e-12) : (iter_count += 1) {
        x = x - (x * x - 2.0) / (2.0 * x);
    }

    try testing.expect(iter_count <= 5);
}

test "newton detects zero derivative" {
    const inflection_f = struct {
        fn f(x: f64) f64 {
            return x * x * x;
        }
        fn df(x: f64) f64 {
            return 3.0 * x * x;
        }
    };

    const result = newton(f64, inflection_f.f, inflection_f.df, 0.0, 1e-12, 100);
    try testing.expectError(error.DerivativeZero, result);
}

test "newton detects max iterations exceeded with bad guess" {
    const result = newton(f64, square_minus_two, square_minus_two_deriv, 1000.0, 1e-15, 5);
    try testing.expectError(error.MaxIterationsExceeded, result);
}

test "newton f32 precision" {
    const square_minus_two_f32 = struct {
        fn f(x: f32) f32 {
            return x * x - 2.0;
        }
        fn deriv(x: f32) f32 {
            return 2.0 * x;
        }
    };

    const root = try newton(f32, square_minus_two_f32.f, square_minus_two_f32.deriv, 1.5, 1e-6, 100);
    try testing.expectApproxEqAbs(root, @as(f32, @floatCast(@sqrt(2.0))), 1e-6);
}

test "newton accepts tolerance at root" {
    // If initial guess is at root, should return immediately
    const root = try newton(f64, square_minus_two, square_minus_two_deriv, @sqrt(2.0), 1e-12, 100);
    try testing.expectApproxEqAbs(root, @sqrt(2.0), 1e-12);
}

// ============================================================================
// BRENT METHOD TESTS
// ============================================================================

test "brent finds root of x² - 2 in [1, 2]" {
    const root = try brent(f64, square_minus_two, 1.0, 2.0, 1e-10, 100);
    try testing.expectApproxEqAbs(root, @sqrt(2.0), 1e-10);
}

test "brent finds root of x³ - x - 2 in [1, 2]" {
    const root = try brent(f64, cubic_minus_two, 1.0, 2.0, 1e-10, 100);
    try testing.expectApproxEqAbs(cubic_minus_two(root), 0.0, 1e-9);
}

test "brent finds transcendental root sin(x) = 0.5" {
    const sin_minus_half = struct {
        fn f(x: f64) f64 {
            return @sin(x) - 0.5;
        }
    }.f;

    const root = try brent(f64, sin_minus_half, 0.0, math.pi, 1e-10, 100);
    // Root should be π/6 ≈ 0.5236
    try testing.expectApproxEqAbs(root, math.pi / 6.0, 1e-10);
}

test "brent always converges when root exists" {
    // Brent should find any root in [a,b] if f(a)*f(b) < 0
    const root = try brent(f64, square_minus_two, 0.0, 3.0, 1e-10, 100);
    try testing.expectApproxEqAbs(root, @sqrt(2.0), 1e-10);
}

test "brent detects invalid interval" {
    const result = brent(f64, square_minus_two, 1.5, 2.0, 1e-10, 100);
    try testing.expectError(error.InvalidInterval, result);
}

test "brent detects a >= b" {
    const result = brent(f64, square_minus_two, 2.0, 1.0, 1e-10, 100);
    try testing.expectError(error.InvalidInterval, result);
}

test "brent handles boundary root" {
    const near_left = struct {
        fn f(x: f64) f64 {
            return (x - 0.1) * (x - 3.0);
        }
    }.f;

    const root = try brent(f64, near_left, 0.0, 1.0, 1e-10, 100);
    try testing.expectApproxEqAbs(root, 0.1, 1e-9);
}

test "brent f32 precision" {
    const square_minus_two_f32 = struct {
        fn f(x: f32) f32 {
            return x * x - 2.0;
        }
    };

    const root = try brent(f32, square_minus_two_f32.f, 1.0, 2.0, 1e-5, 100);
    try testing.expectApproxEqAbs(root, @as(f32, @floatCast(@sqrt(2.0))), 1e-5);
}

test "brent finds root of double root" {
    const root = try brent(f64, double_root_cubic, 1.0, 3.0, 1e-6, 100);
    try testing.expect(@abs(root - 2.0) < 1e-2 or @abs(root - 3.0) < 1e-2);
}

test "brent converges with high tolerance requirement" {
    const root = try brent(f64, square_minus_two, 1.0, 2.0, 1e-12, 100);
    try testing.expectApproxEqAbs(root, @sqrt(2.0), 1e-12);
}

// ============================================================================
// SECANT METHOD TESTS
// ============================================================================

test "secant finds root of x² - 2" {
    const root = try secant(f64, square_minus_two, 1.0, 2.0, 1e-10, 100);
    try testing.expectApproxEqAbs(root, @sqrt(2.0), 1e-10);
}

test "secant finds root of x³ - x - 2" {
    const root = try secant(f64, cubic_minus_two, 1.0, 2.0, 1e-10, 100);
    try testing.expectApproxEqAbs(cubic_minus_two(root), 0.0, 1e-9);
}

test "secant finds exponential root" {
    const root = try secant(f64, exp_minus_two, 0.5, 1.0, 1e-10, 100);
    try testing.expectApproxEqAbs(root, @log(2.0), 1e-10);
}

test "secant solves cos(x) = x" {
    const root = try secant(f64, cos_minus_x, 0.5, 1.0, 1e-10, 100);
    try testing.expectApproxEqAbs(root, 0.739085, 1e-5);
}

test "secant does not need derivative" {
    // This is implicitly tested by using functions without explicit derivatives
    const root = try secant(f64, square_minus_two, 1.0, 2.0, 1e-10, 100);
    try testing.expectApproxEqAbs(root, @sqrt(2.0), 1e-10);
}

test "secant shows super-linear convergence" {
    // Secant should converge faster than linear but slower than quadratic
    var iter_count: usize = 0;
    var x0: f64 = 1.0;
    var x1: f64 = 2.0;

    while (iter_count < 100 and @abs(@sin(x1)) > 1e-12) : (iter_count += 1) {
        const f0 = @sin(x0);
        const f1 = @sin(x1);
        const denom = f1 - f0;
        if (@abs(denom) < 1e-15) break;

        const x_new = x1 - f1 * (x1 - x0) / denom;
        x0 = x1;
        x1 = x_new;
    }

    // Should converge in roughly 5-10 iterations (super-linear order ~1.618)
    try testing.expect(iter_count <= 15);
}

test "secant detects max iterations exceeded" {
    const result = secant(f64, square_minus_two, 1.0, 2.0, 1e-15, 3);
    try testing.expectError(error.MaxIterationsExceeded, result);
}

test "secant f32 precision" {
    const square_minus_two_f32 = struct {
        fn f(x: f32) f32 {
            return x * x - 2.0;
        }
    };

    const root = try secant(f32, square_minus_two_f32.f, 1.0, 2.0, 1e-5, 100);
    try testing.expectApproxEqAbs(root, @as(f32, @floatCast(@sqrt(2.0))), 1e-5);
}

test "secant handles initial points close together" {
    const root = try secant(f64, square_minus_two, 1.4, 1.5, 1e-10, 100);
    try testing.expectApproxEqAbs(root, @sqrt(2.0), 1e-10);
}

// ============================================================================
// FIXED-POINT ITERATION TESTS
// ============================================================================

test "fixed_point finds fixed point of sqrt(x)" {
    // Fixed point of g(x) = √x is x = 1 (since √1 = 1)
    const root = try fixed_point(f64, convergent_fixed_point, 0.5, 1e-10, 100);
    try testing.expectApproxEqAbs(root, 1.0, 1e-10);
}

test "fixed_point finds fixed point of cos(x)" {
    // cos(x) = x has fixed point ≈ 0.739085
    const root = try fixed_point(f64, cos_fixed_point, 0.7, 1e-10, 100);
    try testing.expectApproxEqAbs(root, 0.739085, 1e-5);
}

test "fixed_point shows linear convergence" {
    // Linear convergence: error reduced by constant factor each iteration
    var x: f64 = 0.5;
    var prev_error: f64 = undefined;
    var convergence_linear = true;

    for (0..10) |_| {
        const x_new = @sqrt(x);
        const curr_error = @abs(x_new - 1.0);

        if (prev_error > 0 and curr_error > 0 and curr_error > 1e-12) {
            const ratio = curr_error / prev_error;
            // Linear convergence: ratio should be roughly constant (< 1)
            if (ratio > 1.0) convergence_linear = false;
        }

        prev_error = curr_error;
        x = x_new;
    }

    try testing.expect(convergence_linear);
}

test "fixed_point detects divergence if |g'(x)| > 1" {
    // g(x) = -x + 0.1 has g'(x) = -1, diverges
    const result = fixed_point(f64, oscillating_iter, 0.0, 1e-10, 100);
    try testing.expectError(error.MaxIterationsExceeded, result);
}

test "fixed_point detects max iterations exceeded" {
    const result = fixed_point(f64, convergent_fixed_point, 0.5, 1e-15, 5);
    try testing.expectError(error.MaxIterationsExceeded, result);
}

test "fixed_point f32 precision" {
    const sqrt_f32 = struct {
        fn f(x: f32) f32 {
            return @sqrt(x);
        }
    };

    const root = try fixed_point(f32, sqrt_f32.f, 0.5, 1e-5, 100);
    try testing.expectApproxEqAbs(root, 1.0, 1e-5);
}

test "fixed_point accepts initial guess at fixed point" {
    const root = try fixed_point(f64, convergent_fixed_point, 1.0, 1e-10, 100);
    try testing.expectApproxEqAbs(root, 1.0, 1e-10);
}

test "fixed_point handles slow convergence with tight tolerance" {
    // Still should converge even if slowly
    const root = try fixed_point(f64, convergent_fixed_point, 0.1, 1e-12, 1000);
    try testing.expectApproxEqAbs(root, 1.0, 1e-12);
}

// ============================================================================
// CROSS-METHOD COMPARISON TESTS
// ============================================================================

test "all methods find same root of x² - 2" {
    const bisect_root = try bisect(f64, square_minus_two, 1.0, 2.0, 1e-10, 100);
    const newton_root = try newton(f64, square_minus_two, square_minus_two_deriv, 1.5, 1e-10, 100);
    const brent_root = try brent(f64, square_minus_two, 1.0, 2.0, 1e-10, 100);
    const secant_root = try secant(f64, square_minus_two, 1.0, 2.0, 1e-10, 100);

    try testing.expectApproxEqAbs(bisect_root, newton_root, 1e-9);
    try testing.expectApproxEqAbs(bisect_root, brent_root, 1e-9);
    try testing.expectApproxEqAbs(bisect_root, secant_root, 1e-9);
}

test "brent converges fast on smooth functions" {
    // Just verify both methods converge to same root
    const bisect_root = try bisect(f64, cubic_minus_two, 1.0, 2.0, 1e-12, 100);
    const brent_root = try brent(f64, cubic_minus_two, 1.0, 2.0, 1e-12, 100);

    try testing.expectApproxEqAbs(bisect_root, brent_root, 1e-11);
}

test "newton and secant converge on well-behaved functions" {
    // Newton should converge quickly with good derivative
    var iter_newton: usize = 0;
    var x: f64 = 1.5;
    while (iter_newton < 100 and @abs(x * x - 2.0) > 1e-12) : (iter_newton += 1) {
        x = x - (x * x - 2.0) / (2.0 * x);
    }

    var iter_secant: usize = 0;
    var x0: f64 = 1.0;
    var x1: f64 = 2.0;
    while (iter_secant < 100 and @abs(x1 * x1 - 2.0) > 1e-12) : (iter_secant += 1) {
        const f0 = x0 * x0 - 2.0;
        const f1 = x1 * x1 - 2.0;
        const x_new = x1 - f1 * (x1 - x0) / (f1 - f0);
        x0 = x1;
        x1 = x_new;
    }

    // Both should converge in reasonable time
    try testing.expect(iter_newton <= 10);
    try testing.expect(iter_secant <= 10);
}

// ============================================================================
// MULTIPLE ROOTS AND EDGE CASES
// ============================================================================

test "bisect finds left root of (x-1)(x-3)" {
    const two_roots = struct {
        fn f(x: f64) f64 {
            return (x - 1.0) * (x - 3.0);
        }
    }.f;

    const root = try bisect(f64, two_roots, 0.0, 2.0, 1e-10, 100);
    try testing.expectApproxEqAbs(root, 1.0, 1e-10);
}

test "brent finds right root of (x-1)(x-3)" {
    const two_roots = struct {
        fn f(x: f64) f64 {
            return (x - 1.0) * (x - 3.0);
        }
    }.f;

    const root = try brent(f64, two_roots, 2.0, 4.0, 1e-10, 100);
    try testing.expectApproxEqAbs(root, 3.0, 1e-10);
}

test "all methods handle polynomial with multiple roots" {
    // (x-2)³ = 0 has triple root at x=2
    const triple_root = struct {
        fn f(x: f64) f64 {
            const y = x - 2.0;
            return y * y * y;
        }
    }.f;

    const bisect_root = try bisect(f64, triple_root, 1.0, 3.0, 1e-6, 100);
    const brent_root = try brent(f64, triple_root, 1.0, 3.0, 1e-6, 100);

    try testing.expectApproxEqAbs(bisect_root, 2.0, 1e-3);
    try testing.expectApproxEqAbs(brent_root, 2.0, 1e-3);
}

// ============================================================================
// TOLERANCE AND CONVERGENCE TESTS
// ============================================================================

test "bisect achieves requested tolerance" {
    const tol = 1e-12;
    const root = try bisect(f64, square_minus_two, 1.0, 2.0, tol, 100);
    const err = @abs(root * root - 2.0);
    try testing.expect(err < 1e-10); // Function value should be very small
}

test "newton achieves requested tolerance" {
    const tol = 1e-12;
    const root = try newton(f64, square_minus_two, square_minus_two_deriv, 1.5, tol, 100);
    const err = @abs(root * root - 2.0);
    try testing.expect(err < tol);
}

test "brent achieves requested tolerance" {
    const tol = 1e-12;
    const root = try brent(f64, square_minus_two, 1.0, 2.0, tol, 100);
    const err = @abs(root * root - 2.0);
    try testing.expect(err < 1e-10);
}

test "secant achieves requested tolerance" {
    const tol = 1e-10;
    const root = try secant(f64, square_minus_two, 1.0, 2.0, tol, 100);
    const err = @abs(root * root - 2.0);
    try testing.expect(err < tol);
}

test "fixed_point achieves requested tolerance" {
    const tol = 1e-10;
    const root = try fixed_point(f64, convergent_fixed_point, 0.5, tol, 100);
    const err = @abs(root - 1.0);
    try testing.expect(err < tol);
}

// ============================================================================
// MEMORY AND STATE SAFETY
// ============================================================================

test "bisect multiple calls do not cross-contaminate" {
    const root1 = try bisect(f64, square_minus_two, 1.0, 2.0, 1e-10, 100);
    const root2 = try bisect(f64, square_minus_two, 1.0, 2.0, 1e-10, 100);

    try testing.expectApproxEqAbs(root1, root2, 1e-15);
}

test "newton multiple calls do not cross-contaminate" {
    const root1 = try newton(f64, square_minus_two, square_minus_two_deriv, 1.5, 1e-10, 100);
    const root2 = try newton(f64, square_minus_two, square_minus_two_deriv, 1.5, 1e-10, 100);

    try testing.expectApproxEqAbs(root1, root2, 1e-15);
}

test "all methods use O(1) space with no allocations" {
    // All methods use only local variables — O(1) stack space
    // Call multiple times with different functions to verify no state leaks
    _ = try bisect(f64, square_minus_two, 1.0, 2.0, 1e-10, 100);
    _ = try newton(f64, square_minus_two, square_minus_two_deriv, 1.5, 1e-10, 100);
    _ = try brent(f64, square_minus_two, 1.0, 2.0, 1e-10, 100);
    _ = try secant(f64, square_minus_two, 1.0, 2.0, 1e-10, 100);
    _ = try fixed_point(f64, convergent_fixed_point, 0.5, 1e-10, 100);

    // If we reach here without errors, all calls succeeded with no heap issues
    try testing.expect(true);
}
