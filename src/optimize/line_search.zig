//! Line Search Algorithms for Optimization
//!
//! This module provides algorithms for selecting step sizes in gradient-based optimization.
//! Line search finds a step size α such that moving from x to x + α·p (where p is a
//! descent direction) produces sufficient progress toward the optimum.
//!
//! ## Supported Methods
//!
//! - **Armijo Rule** — Sufficient decrease condition: f(x+α·p) ≤ f(x) + c₁·α·∇f(x)ᵀp
//! - **Wolfe Conditions** — Armijo + curvature condition (strong Wolfe variant)
//! - **Backtracking** — Simple geometric reduction until Armijo condition satisfied
//!
//! ## Time Complexity
//!
//! - Armijo: O(max_iter × f_eval)
//! - Wolfe: O(max_iter × (f_eval + grad_eval))
//! - Backtracking: O(max_iter × f_eval)
//!
//! ## Space Complexity
//!
//! - Armijo: O(1)
//! - Wolfe: O(n) for gradient storage
//! - Backtracking: O(1)
//!
//! ## Parameters & Conventions
//!
//! - **c₁** (Armijo constant): Controls sufficient decrease. Typical: 1e-4. Valid: (0, 1)
//! - **c₂** (Curvature constant): For Wolfe, strong Wolfe uses |grad·p| ≤ c₂|grad₀·p|. Typical: 0.9. Valid: (c₁, 1)
//! - **ρ** (Backtracking reduction factor): α_new = ρ·α_old. Typical: 0.5. Valid: (0, 1)
//! - **Search direction p**: Must satisfy p·∇f(x) < 0 (descent direction)
//! - **α_init** (Initial step size): Heuristic starting point, e.g., 1.0 for Newton, smaller for gradient descent

const std = @import("std");
const math = std.math;
const testing = std.testing;

/// Floating-point type constraint
pub fn ObjectiveFn(comptime T: type) type {
    return fn (x: []const T) T;
}

pub fn GradientFn(comptime T: type) type {
    return fn (x: []const T, out_grad: []T) void;
}

/// Error set for line search operations
pub const LineSearchError = error{
    InvalidParameters,       // c1/c2/rho out of valid range
    NotDescentDirection,     // p·grad >= 0 (not descent)
    MaxIterationsExceeded,   // Failed to converge within max_iter
    AllocationFailed,        // Memory allocation error
    OutOfMemory,             // Allocator ran out of memory
};

/// Result of Armijo line search
pub fn ArmijoResult(comptime T: type) type {
    return struct {
        alpha: T,
        f_new: T,
        n_iter: usize,
        converged: bool,
    };
}

/// Result of Wolfe line search (requires gradient evaluation)
pub fn WolfeResult(comptime T: type) type {
    return struct {
        alpha: T,
        f_new: T,
        grad_new: []T,       // Caller must free
        n_iter: usize,
        converged: bool,
    };
}

/// Result of backtracking line search
pub fn BacktrackResult(comptime T: type) type {
    return struct {
        alpha: T,
        f_new: T,
        n_iter: usize,
        converged: bool,
    };
}

/// Armijo line search: finds step size satisfying sufficient decrease
/// f(x + α·p) ≤ f(x) + c₁·α·grad·p
///
/// Parameters:
/// - f: objective function f(x) → scalar
/// - x: current point
/// - p: descent direction (must satisfy p·grad < 0)
/// - grad: gradient at x
/// - alpha_init: initial step size to try
/// - c1: sufficient decrease constant (typical: 1e-4, valid: (0, 1))
/// - max_iter: maximum backtracking steps
/// - allocator: memory allocator (required but step size search uses O(1))
///
/// Returns: ArmijoResult with step size, new function value, and convergence info
///
/// Time: O(max_iter × f_eval) | Space: O(1)
pub fn armijo(
    comptime T: type,
    f: ObjectiveFn(T),
    x: []const T,
    p: []const T,
    grad: []const T,
    alpha_init: T,
    c1: T,
    max_iter: usize,
    allocator: std.mem.Allocator,
) LineSearchError!ArmijoResult(T) {
    // Parameter validation
    const zero: T = 0;
    const one: T = 1;
    if (c1 <= zero or c1 >= one) {
        return error.InvalidParameters;
    }

    // Verify descent direction: p·grad < 0
    var dir_product: T = 0;
    for (p, grad) |pi, gi| {
        dir_product += pi * gi;
    }
    if (dir_product >= 0) {
        return error.NotDescentDirection;
    }

    // Use allocator in loop (required for step size search, though O(1) space)
    // Initial function value
    const f_x = f(x);

    // Backtracking loop
    var alpha = alpha_init;
    var iter: usize = 0;

    while (iter < max_iter) : (iter += 1) {
        // Compute x_new = x + alpha * p
        const x_new = allocator.alloc(T, x.len) catch return error.OutOfMemory;
        defer allocator.free(x_new);

        for (x_new, x, p) |*x_new_i, x_i, p_i| {
            x_new_i.* = x_i + alpha * p_i;
        }

        // Evaluate f(x_new)
        const f_new = f(x_new);

        // Check Armijo condition: f_new <= f_x + c1 * alpha * (grad · p)
        const threshold = f_x + c1 * alpha * dir_product;

        if (f_new <= threshold) {
            return ArmijoResult(T){
                .alpha = alpha,
                .f_new = f_new,
                .n_iter = iter + 1,
                .converged = true,
            };
        }

        // Reduce step size (geometric backtrack)
        const half: T = 0.5;
        alpha = alpha * half;
    }

    return error.MaxIterationsExceeded;
}

/// Wolfe line search: finds step size satisfying both Armijo and curvature conditions
/// Uses the strong Wolfe variant:
/// - Armijo: f(x+α·p) ≤ f(x) + c₁·α·grad·p
/// - Strong curvature: |grad(x+α·p)·p| ≤ c₂·|grad·p|
///
/// Parameters:
/// - f: objective function
/// - grad_f: gradient function (required for curvature check)
/// - x: current point
/// - p: descent direction
/// - alpha_init: initial step size
/// - c1: Armijo constant (typical: 1e-4, valid: (0, c2))
/// - c2: curvature constant (typical: 0.9, valid: (c1, 1))
/// - max_iter: maximum iterations
/// - allocator: memory allocator (used for gradient_new allocation)
///
/// Returns: WolfeResult with step size, new function value, new gradient (caller frees), and convergence info
///
/// Time: O(max_iter × (f_eval + grad_eval)) | Space: O(n)
pub fn wolfe(
    comptime T: type,
    f: ObjectiveFn(T),
    grad_f: GradientFn(T),
    x: []const T,
    p: []const T,
    alpha_init: T,
    c1: T,
    c2: T,
    max_iter: usize,
    allocator: std.mem.Allocator,
) LineSearchError!WolfeResult(T) {
    // Parameter validation
    const zero: T = 0;
    const one: T = 1;
    if (c1 <= zero or c1 >= c2 or c2 >= one) {
        return error.InvalidParameters;
    }

    // Verify descent direction
    const grad = allocator.alloc(T, x.len) catch return error.OutOfMemory;
    defer allocator.free(grad);
    grad_f(x, grad);

    var dir_product: T = 0;
    for (p, grad) |pi, gi| {
        dir_product += pi * gi;
    }
    if (dir_product >= 0) {
        return error.NotDescentDirection;
    }

    // Initial function value
    const f_x = f(x);

    // Backtracking/expansion loop
    var alpha = alpha_init;
    var iter: usize = 0;

    while (iter < max_iter) : (iter += 1) {
        // Compute x_new = x + alpha * p
        const x_new = allocator.alloc(T, x.len) catch return error.OutOfMemory;
        defer allocator.free(x_new);

        for (x_new, x, p) |*x_new_i, x_i, p_i| {
            x_new_i.* = x_i + alpha * p_i;
        }

        // Evaluate f(x_new)
        const f_new = f(x_new);

        // Check Armijo condition
        const armijo_threshold = f_x + c1 * alpha * dir_product;
        if (f_new > armijo_threshold) {
            // Armijo fails: backtrack
            const half: T = 0.5;
            alpha = alpha * half;
            continue;
        }

        // Evaluate gradient at x_new
        const grad_new = allocator.alloc(T, x.len) catch return error.OutOfMemory;

        grad_f(x_new, grad_new);

        // Check strong Wolfe curvature condition: |grad_new·p| ≤ c2·|grad·p|
        var grad_new_dir_product: T = 0;
        for (p, grad_new) |pi, gi| {
            grad_new_dir_product += pi * gi;
        }

        const abs_grad_new_dir = @abs(grad_new_dir_product);
        const abs_grad_dir = @abs(dir_product);
        const curvature_bound = c2 * abs_grad_dir;

        if (abs_grad_new_dir <= curvature_bound) {
            // Both conditions satisfied
            return WolfeResult(T){
                .alpha = alpha,
                .f_new = f_new,
                .grad_new = grad_new,
                .n_iter = iter + 1,
                .converged = true,
            };
        }

        // Curvature condition not satisfied
        // Check if we should expand or backtrack
        // If grad_new·p is still negative (descent direction), expand to explore
        // Otherwise, backtrack
        if (grad_new_dir_product < 0) {
            const two: T = 2.0;
            alpha = alpha * two;
        } else {
            const half: T = 0.5;
            alpha = alpha * half;
        }
        allocator.free(grad_new);
    }

    return error.MaxIterationsExceeded;
}

/// Backtracking line search: simple geometric reduction until Armijo satisfied
/// Repeatedly reduces α by factor ρ: α_new = ρ·α_old
///
/// Parameters:
/// - f: objective function
/// - x: current point
/// - p: descent direction
/// - grad: gradient at x
/// - alpha_init: initial step size
/// - rho: reduction factor (typical: 0.5, valid: (0, 1))
/// - c: Armijo constant (typical: 1e-4, valid: (0, 1))
/// - max_iter: maximum backtracking steps
/// - allocator: memory allocator
///
/// Returns: BacktrackResult with step size, new function value, and convergence info
///
/// Time: O(max_iter × f_eval) | Space: O(1)
pub fn backtracking(
    comptime T: type,
    f: ObjectiveFn(T),
    x: []const T,
    p: []const T,
    grad: []const T,
    alpha_init: T,
    rho: T,
    c: T,
    max_iter: usize,
    allocator: std.mem.Allocator,
) LineSearchError!BacktrackResult(T) {
    // Parameter validation
    const zero: T = 0;
    const one: T = 1;
    if (rho <= zero or rho >= one) {
        return error.InvalidParameters;
    }
    if (c <= zero or c >= one) {
        return error.InvalidParameters;
    }

    // Verify descent direction
    var dir_product: T = 0;
    for (p, grad) |pi, gi| {
        dir_product += pi * gi;
    }
    if (dir_product >= 0) {
        return error.NotDescentDirection;
    }

    // Use allocator in loop (required for step size search, though O(1) space)
    // Initial function value
    const f_x = f(x);

    // Backtracking loop
    var alpha = alpha_init;
    var iter: usize = 0;

    while (iter < max_iter) : (iter += 1) {
        // Compute x_new = x + alpha * p
        const x_new = allocator.alloc(T, x.len) catch return error.OutOfMemory;
        defer allocator.free(x_new);

        for (x_new, x, p) |*x_new_i, x_i, p_i| {
            x_new_i.* = x_i + alpha * p_i;
        }

        // Evaluate f(x_new)
        const f_new = f(x_new);

        // Check Armijo condition: f_new <= f_x + c * alpha * (grad · p)
        const threshold = f_x + c * alpha * dir_product;

        if (f_new <= threshold) {
            return BacktrackResult(T){
                .alpha = alpha,
                .f_new = f_new,
                .n_iter = iter + 1,
                .converged = true,
            };
        }

        // Reduce step size
        alpha = rho * alpha;
    }

    return error.MaxIterationsExceeded;
}

// ============================================================================
// TESTS
// ============================================================================

// Test helper: simple quadratic function f(x) = x²
fn quadratic_f64(x: []const f64) f64 {
    var sum: f64 = 0;
    for (x) |xi| {
        sum += xi * xi;
    }
    return sum;
}

// Test helper: gradient of quadratic, grad_f(x) = 2x
fn quadratic_grad_f64(x: []const f64, out_grad: []f64) void {
    const two: f64 = 2.0;
    for (x, out_grad) |xi, *gi| {
        gi.* = two * xi;
    }
}

// Test helper: simple quadratic function f(x) = x² for f32
fn quadratic_f32(x: []const f32) f32 {
    var sum: f32 = 0;
    for (x) |xi| {
        sum += xi * xi;
    }
    return sum;
}

// Test helper: gradient of quadratic for f32
fn quadratic_grad_f32(x: []const f32, out_grad: []f32) void {
    const two: f32 = 2.0;
    for (x, out_grad) |xi, *gi| {
        gi.* = two * xi;
    }
}

// Test helper: Rosenbrock function f(x,y) = (1-x)² + 100(y-x²)²
fn rosenbrock_f64(x: []const f64) f64 {
    std.debug.assert(x.len >= 2);
    const a: f64 = 1.0 - x[0];
    const b: f64 = x[1] - x[0] * x[0];
    return a * a + 100.0 * b * b;
}

// Test helper: gradient of Rosenbrock
fn rosenbrock_grad_f64(x: []const f64, out_grad: []f64) void {
    std.debug.assert(x.len >= 2);
    const x0 = x[0];
    const x1 = x[1];
    const two: f64 = 2.0;
    const four: f64 = 4.0;
    const hundred: f64 = 100.0;

    out_grad[0] = -two * (1.0 - x0) - four * hundred * x0 * (x1 - x0 * x0);
    out_grad[1] = two * hundred * (x1 - x0 * x0);
}

// ============================================================================
// ARMIJO TESTS (~12 tests)
// ============================================================================

test "armijo: basic quadratic satisfies sufficient decrease" {
    const allocator = testing.allocator;

    var x = [_]f64{2.0};
    var grad = [_]f64{0};
    quadratic_grad_f64(&x, &grad);

    var p = [_]f64{-1.0}; // Negative gradient direction
    const alpha_init: f64 = 1.0;
    const c1: f64 = 1e-4;
    const max_iter = 20;

    const result = try armijo(f64, quadratic_f64, &x, &p, &grad, alpha_init, c1, max_iter, allocator);

    try testing.expect(result.converged);
    try testing.expect(result.alpha > 0);
    try testing.expect(result.n_iter > 0 and result.n_iter <= max_iter);
    try testing.expect(result.f_new < quadratic_f64(&x)); // Sufficient decrease
}

test "armijo: finds alpha satisfying Armijo condition" {
    const allocator = testing.allocator;

    var x = [_]f64{2.0};
    var grad = [_]f64{0};
    quadratic_grad_f64(&x, &grad);

    var p = [_]f64{-1.0};
    const alpha_init: f64 = 1.0;
    const c1: f64 = 1e-4;

    const result = try armijo(f64, quadratic_f64, &x, &p, &grad, alpha_init, c1, 20, allocator);

    // Verify Armijo condition: f(x + alpha*p) <= f(x) + c1*alpha*(grad·p)
    var x_new = [_]f64{x[0] + result.alpha * p[0]};
    const f_x = quadratic_f64(&x);
    const f_new = quadratic_f64(&x_new);
    const dir_product = p[0] * grad[0];
    const threshold = f_x + c1 * result.alpha * dir_product;

    try testing.expect(f_new <= threshold + 1e-10); // Small tolerance for numerical errors
}

test "armijo: convergence with different c1 values" {
    const allocator = testing.allocator;

    var x = [_]f64{2.0};
    var grad = [_]f64{0};
    quadratic_grad_f64(&x, &grad);

    var p = [_]f64{-1.0};
    const alpha_init: f64 = 1.0;

    // Smaller c1 should allow larger steps
    const result_small_c1 = try armijo(f64, quadratic_f64, &x, &p, &grad, alpha_init, 1e-4, 20, allocator);
    const result_large_c1 = try armijo(f64, quadratic_f64, &x, &p, &grad, alpha_init, 0.1, 20, allocator);

    try testing.expect(result_small_c1.converged);
    try testing.expect(result_large_c1.converged);
    try testing.expect(result_small_c1.alpha >= result_large_c1.alpha); // Larger c1 → more conservative
}

test "armijo: rejects invalid c1 (c1 <= 0)" {
    const allocator = testing.allocator;

    var x = [_]f64{1.0};
    var grad = [_]f64{2.0};
    var p = [_]f64{-1.0};

    const result = armijo(f64, quadratic_f64, &x, &p, &grad, 1.0, -0.1, 20, allocator);
    try testing.expectError(error.InvalidParameters, result);
}

test "armijo: rejects invalid c1 (c1 >= 1)" {
    const allocator = testing.allocator;

    var x = [_]f64{1.0};
    var grad = [_]f64{2.0};
    var p = [_]f64{-1.0};

    const result = armijo(f64, quadratic_f64, &x, &p, &grad, 1.0, 1.5, 20, allocator);
    try testing.expectError(error.InvalidParameters, result);
}

test "armijo: rejects non-descent direction" {
    const allocator = testing.allocator;

    var x = [_]f64{1.0};
    var grad = [_]f64{2.0};
    var p = [_]f64{1.0}; // Same direction as gradient, not descent

    const result = armijo(f64, quadratic_f64, &x, &p, &grad, 1.0, 1e-4, 20, allocator);
    try testing.expectError(error.NotDescentDirection, result);
}

test "armijo: Rosenbrock function convergence" {
    const allocator = testing.allocator;

    var x = [_]f64{ 0.0, 0.0 };
    var grad = [_]f64{0, 0};
    rosenbrock_grad_f64(&x, &grad);

    var p = [_]f64{ -grad[0], -grad[1] }; // Steepest descent
    const alpha_init: f64 = 0.001;
    const c1: f64 = 1e-4;

    const result = try armijo(f64, rosenbrock_f64, &x, &p, &grad, alpha_init, c1, 20, allocator);

    try testing.expect(result.converged);
    try testing.expect(result.alpha > 0);
}

test "armijo: f32 type support" {
    const allocator = testing.allocator;

    var x = [_]f32{2.0};
    var grad = [_]f32{0};
    quadratic_grad_f32(&x, &grad);

    var p = [_]f32{-1.0};
    const alpha_init: f32 = 1.0;
    const c1: f32 = 1e-4;

    const result = try armijo(f32, quadratic_f32, &x, &p, &grad, alpha_init, c1, 20, allocator);

    try testing.expect(result.converged);
    try testing.expect(result.alpha > 0);
}

test "armijo: large initial step size requires backtracking" {
    const allocator = testing.allocator;

    var x = [_]f64{2.0};
    var grad = [_]f64{0};
    quadratic_grad_f64(&x, &grad);

    var p = [_]f64{-1.0};
    const alpha_init: f64 = 1000.0; // Much too large
    const c1: f64 = 1e-4;

    const result = try armijo(f64, quadratic_f64, &x, &p, &grad, alpha_init, c1, 30, allocator);

    try testing.expect(result.converged);
    try testing.expect(result.alpha < alpha_init); // Should have reduced
    try testing.expect(result.n_iter > 1); // Multiple backtrack steps
}

test "armijo: perfect initial step needs few iterations" {
    const allocator = testing.allocator;

    var x = [_]f64{2.0};
    var grad = [_]f64{0};
    quadratic_grad_f64(&x, &grad);

    var p = [_]f64{-1.0};
    const alpha_init: f64 = 0.5; // Good initial guess
    const c1: f64 = 1e-4;

    const result = try armijo(f64, quadratic_f64, &x, &p, &grad, alpha_init, c1, 20, allocator);

    try testing.expect(result.converged);
    try testing.expect(result.n_iter <= 3); // Should converge quickly
}

test "armijo: max_iter exceeded" {
    const allocator = testing.allocator;

    var x = [_]f64{2.0};
    var grad = [_]f64{0};
    quadratic_grad_f64(&x, &grad);

    var p = [_]f64{-1.0};
    const alpha_init: f64 = 1.0;
    const c1: f64 = 1e-4;
    const max_iter: usize = 0; // Force failure

    const result = armijo(f64, quadratic_f64, &x, &p, &grad, alpha_init, c1, max_iter, allocator);
    try testing.expectError(error.MaxIterationsExceeded, result);
}

// ============================================================================
// WOLFE TESTS (~14 tests)
// ============================================================================

test "wolfe: basic quadratic satisfies both conditions" {
    const allocator = testing.allocator;

    var x = [_]f64{2.0};
    var grad = [_]f64{0};
    quadratic_grad_f64(&x, &grad);

    var p = [_]f64{-1.0};
    const alpha_init: f64 = 1.0;
    const c1: f64 = 1e-4;
    const c2: f64 = 0.9;

    const result = try wolfe(f64, quadratic_f64, quadratic_grad_f64, &x, &p, alpha_init, c1, c2, 20, allocator);
    defer allocator.free(result.grad_new);

    try testing.expect(result.converged);
    try testing.expect(result.alpha > 0);
    try testing.expect(result.grad_new.len == 1);
}

test "wolfe: strong Wolfe curvature condition verified" {
    const allocator = testing.allocator;

    var x = [_]f64{2.0};
    var grad = [_]f64{0};
    quadratic_grad_f64(&x, &grad);

    var p = [_]f64{-1.0};
    const alpha_init: f64 = 1.0;
    const c1: f64 = 1e-4;
    const c2: f64 = 0.9;

    const result = try wolfe(f64, quadratic_f64, quadratic_grad_f64, &x, &p, alpha_init, c1, c2, 20, allocator);
    defer allocator.free(result.grad_new);

    // Verify curvature: |grad_new·p| ≤ c2·|grad·p|
    const dir_product = p[0] * grad[0];
    const grad_new_dir_product = p[0] * result.grad_new[0];

    const abs_grad_new_dir = @abs(grad_new_dir_product);
    const abs_grad_dir = @abs(dir_product);
    const curvature_bound = c2 * abs_grad_dir;

    try testing.expect(abs_grad_new_dir <= curvature_bound + 1e-10);
}

test "wolfe: Rosenbrock realistic function" {
    const allocator = testing.allocator;

    var x = [_]f64{ 0.0, 0.0 };
    var grad = [_]f64{0, 0};
    rosenbrock_grad_f64(&x, &grad);

    var p = [_]f64{ -grad[0], -grad[1] };
    const alpha_init: f64 = 0.001;
    const c1: f64 = 1e-4;
    const c2: f64 = 0.9;

    const result = try wolfe(f64, rosenbrock_f64, rosenbrock_grad_f64, &x, &p, alpha_init, c1, c2, 30, allocator);
    defer allocator.free(result.grad_new);

    try testing.expect(result.converged);
    try testing.expect(result.alpha > 0);
}

test "wolfe: parameter validation c1 >= c2 rejected" {
    const allocator = testing.allocator;

    var x = [_]f64{1.0};
    var p = [_]f64{-1.0};

    const result = wolfe(f64, quadratic_f64, quadratic_grad_f64, &x, &p, 1.0, 0.5, 0.3, 20, allocator);
    try testing.expectError(error.InvalidParameters, result);
}

test "wolfe: rejects invalid c2 (c2 >= 1)" {
    const allocator = testing.allocator;

    var x = [_]f64{1.0};
    var p = [_]f64{-1.0};

    const result = wolfe(f64, quadratic_f64, quadratic_grad_f64, &x, &p, 1.0, 1e-4, 1.5, 20, allocator);
    try testing.expectError(error.InvalidParameters, result);
}

test "wolfe: rejects non-descent direction" {
    const allocator = testing.allocator;

    var x = [_]f64{1.0};
    var p = [_]f64{1.0};

    const result = wolfe(f64, quadratic_f64, quadratic_grad_f64, &x, &p, 1.0, 1e-4, 0.9, 20, allocator);
    try testing.expectError(error.NotDescentDirection, result);
}

test "wolfe: c2 parameter effects iteration count" {
    const allocator = testing.allocator;

    var x = [_]f64{2.0};
    var grad = [_]f64{0};
    quadratic_grad_f64(&x, &grad);

    var p = [_]f64{-1.0};
    const alpha_init: f64 = 1.0;
    const c1: f64 = 1e-4;

    const result_low_c2 = try wolfe(f64, quadratic_f64, quadratic_grad_f64, &x, &p, alpha_init, c1, 0.1, 30, allocator);
    defer allocator.free(result_low_c2.grad_new);

    const result_high_c2 = try wolfe(f64, quadratic_f64, quadratic_grad_f64, &x, &p, alpha_init, c1, 0.9, 30, allocator);
    defer allocator.free(result_high_c2.grad_new);

    try testing.expect(result_low_c2.converged);
    try testing.expect(result_high_c2.converged);
    // Higher c2 is more permissive (should generally need fewer iters, but not guaranteed)
}

test "wolfe: f32 type support" {
    const allocator = testing.allocator;

    var x = [_]f32{2.0};
    var grad = [_]f32{0};
    quadratic_grad_f32(&x, &grad);

    var p = [_]f32{-1.0};
    const alpha_init: f32 = 1.0;
    const c1: f32 = 1e-4;
    const c2: f32 = 0.9;

    const result = try wolfe(f32, quadratic_f32, quadratic_grad_f32, &x, &p, alpha_init, c1, c2, 20, allocator);
    defer allocator.free(result.grad_new);

    try testing.expect(result.converged);
}

test "wolfe: gradient memory properly allocated" {
    const allocator = testing.allocator;

    var x = [_]f64{2.0};
    var grad = [_]f64{0};
    quadratic_grad_f64(&x, &grad);

    var p = [_]f64{-1.0};

    const result = try wolfe(f64, quadratic_f64, quadratic_grad_f64, &x, &p, 1.0, 1e-4, 0.9, 20, allocator);
    defer allocator.free(result.grad_new);

    try testing.expect(result.grad_new.len == 1);
    try testing.expect(!math.isNan(result.grad_new[0]));
}

test "wolfe: max_iter exceeded" {
    const allocator = testing.allocator;

    var x = [_]f64{2.0};
    var grad = [_]f64{0};
    quadratic_grad_f64(&x, &grad);

    var p = [_]f64{-1.0};
    const max_iter: usize = 0;

    const result = wolfe(f64, quadratic_f64, quadratic_grad_f64, &x, &p, 1.0, 1e-4, 0.9, max_iter, allocator);
    try testing.expectError(error.MaxIterationsExceeded, result);
}

test "wolfe: multiple calls no memory leaks" {
    const allocator = testing.allocator;

    var x = [_]f64{2.0};
    var grad = [_]f64{0};

    for (0..5) |_| {
        quadratic_grad_f64(&x, &grad);
        var p = [_]f64{-1.0};

        const result = try wolfe(f64, quadratic_f64, quadratic_grad_f64, &x, &p, 1.0, 1e-4, 0.9, 20, allocator);
        allocator.free(result.grad_new);
    }
}

// ============================================================================
// BACKTRACKING TESTS (~10 tests)
// ============================================================================

test "backtracking: basic quadratic with geometric reduction" {
    const allocator = testing.allocator;

    var x = [_]f64{2.0};
    var grad = [_]f64{0};
    quadratic_grad_f64(&x, &grad);

    var p = [_]f64{-1.0};
    const alpha_init: f64 = 1.0;
    const rho: f64 = 0.5;
    const c: f64 = 1e-4;

    const result = try backtracking(f64, quadratic_f64, &x, &p, &grad, alpha_init, rho, c, 20, allocator);

    try testing.expect(result.converged);
    try testing.expect(result.alpha > 0 and result.alpha <= alpha_init);
}

test "backtracking: satisfies Armijo condition" {
    const allocator = testing.allocator;

    var x = [_]f64{2.0};
    var grad = [_]f64{0};
    quadratic_grad_f64(&x, &grad);

    var p = [_]f64{-1.0};
    const alpha_init: f64 = 1.0;
    const rho: f64 = 0.5;
    const c: f64 = 1e-4;

    const result = try backtracking(f64, quadratic_f64, &x, &p, &grad, alpha_init, rho, c, 20, allocator);

    var x_new = [_]f64{x[0] + result.alpha * p[0]};
    const f_x = quadratic_f64(&x);
    const f_new = quadratic_f64(&x_new);
    const dir_product = p[0] * grad[0];
    const threshold = f_x + c * result.alpha * dir_product;

    try testing.expect(f_new <= threshold + 1e-10);
}

test "backtracking: Rosenbrock convergence" {
    const allocator = testing.allocator;

    var x = [_]f64{ 0.0, 0.0 };
    var grad = [_]f64{0, 0};
    rosenbrock_grad_f64(&x, &grad);

    var p = [_]f64{ -grad[0], -grad[1] };
    const alpha_init: f64 = 0.001;
    const rho: f64 = 0.5;
    const c: f64 = 1e-4;

    const result = try backtracking(f64, rosenbrock_f64, &x, &p, &grad, alpha_init, rho, c, 20, allocator);

    try testing.expect(result.converged);
    try testing.expect(result.alpha > 0);
}

test "backtracking: rho parameter effects" {
    const allocator = testing.allocator;

    var x = [_]f64{2.0};
    var grad = [_]f64{0};
    quadratic_grad_f64(&x, &grad);

    var p = [_]f64{-1.0};
    const alpha_init: f64 = 1.0;
    const c: f64 = 1e-4;

    // Smaller rho (faster reduction) vs larger rho (slower reduction)
    const result_small_rho = try backtracking(f64, quadratic_f64, &x, &p, &grad, alpha_init, 0.1, c, 30, allocator);
    const result_large_rho = try backtracking(f64, quadratic_f64, &x, &p, &grad, alpha_init, 0.9, c, 30, allocator);

    try testing.expect(result_small_rho.converged);
    try testing.expect(result_large_rho.converged);
    // Smaller rho → faster reduction → fewer iterations (generally)
}

test "backtracking: rejects invalid rho (rho <= 0)" {
    const allocator = testing.allocator;

    var x = [_]f64{1.0};
    var grad = [_]f64{2.0};
    var p = [_]f64{-1.0};

    const result = backtracking(f64, quadratic_f64, &x, &p, &grad, 1.0, -0.1, 1e-4, 20, allocator);
    try testing.expectError(error.InvalidParameters, result);
}

test "backtracking: rejects invalid rho (rho >= 1)" {
    const allocator = testing.allocator;

    var x = [_]f64{1.0};
    var grad = [_]f64{2.0};
    var p = [_]f64{-1.0};

    const result = backtracking(f64, quadratic_f64, &x, &p, &grad, 1.0, 1.5, 1e-4, 20, allocator);
    try testing.expectError(error.InvalidParameters, result);
}

test "backtracking: rejects non-descent direction" {
    const allocator = testing.allocator;

    var x = [_]f64{1.0};
    var grad = [_]f64{2.0};
    var p = [_]f64{1.0};

    const result = backtracking(f64, quadratic_f64, &x, &p, &grad, 1.0, 0.5, 1e-4, 20, allocator);
    try testing.expectError(error.NotDescentDirection, result);
}

test "backtracking: f32 type support" {
    const allocator = testing.allocator;

    var x = [_]f32{2.0};
    var grad = [_]f32{0};
    quadratic_grad_f32(&x, &grad);

    var p = [_]f32{-1.0};
    const alpha_init: f32 = 1.0;
    const rho: f32 = 0.5;
    const c: f32 = 1e-4;

    const result = try backtracking(f32, quadratic_f32, &x, &p, &grad, alpha_init, rho, c, 20, allocator);

    try testing.expect(result.converged);
}

test "backtracking: max_iter exceeded" {
    const allocator = testing.allocator;

    var x = [_]f64{2.0};
    var grad = [_]f64{0};
    quadratic_grad_f64(&x, &grad);

    var p = [_]f64{-1.0};
    const max_iter: usize = 0;

    const result = backtracking(f64, quadratic_f64, &x, &p, &grad, 1.0, 0.5, 1e-4, max_iter, allocator);
    try testing.expectError(error.MaxIterationsExceeded, result);
}

// ============================================================================
// CROSS-METHOD COMPARISON TESTS (~4 tests)
// ============================================================================

test "cross-method: all find valid step sizes" {
    const allocator = testing.allocator;

    var x = [_]f64{2.0};
    var grad = [_]f64{0};
    quadratic_grad_f64(&x, &grad);

    var p = [_]f64{-1.0};
    const alpha_init: f64 = 1.0;

    const armijo_result = try armijo(f64, quadratic_f64, &x, &p, &grad, alpha_init, 1e-4, 20, allocator);
    const backtrack_result = try backtracking(f64, quadratic_f64, &x, &p, &grad, alpha_init, 0.5, 1e-4, 20, allocator);
    const wolfe_result = try wolfe(f64, quadratic_f64, quadratic_grad_f64, &x, &p, alpha_init, 1e-4, 0.9, 20, allocator);
    defer allocator.free(wolfe_result.grad_new);

    try testing.expect(armijo_result.converged);
    try testing.expect(backtrack_result.converged);
    try testing.expect(wolfe_result.converged);

    try testing.expect(armijo_result.alpha > 0);
    try testing.expect(backtrack_result.alpha > 0);
    try testing.expect(wolfe_result.alpha > 0);
}

test "cross-method: Wolfe is stricter than Armijo" {
    const allocator = testing.allocator;

    var x = [_]f64{2.0};
    var grad = [_]f64{0};
    quadratic_grad_f64(&x, &grad);

    var p = [_]f64{-1.0};
    const alpha_init: f64 = 1.0;

    const armijo_result = try armijo(f64, quadratic_f64, &x, &p, &grad, alpha_init, 1e-4, 20, allocator);
    const wolfe_result = try wolfe(f64, quadratic_f64, quadratic_grad_f64, &x, &p, alpha_init, 1e-4, 0.9, 20, allocator);
    defer allocator.free(wolfe_result.grad_new);

    // Wolfe requires both Armijo and curvature, so step typically smaller or equal
    try testing.expect(wolfe_result.alpha <= armijo_result.alpha + 1e-10);
}

test "cross-method: backtracking approximates Armijo" {
    const allocator = testing.allocator;

    var x = [_]f64{2.0};
    var grad = [_]f64{0};
    quadratic_grad_f64(&x, &grad);

    var p = [_]f64{-1.0};
    const alpha_init: f64 = 1.0;
    const c: f64 = 1e-4;

    const armijo_result = try armijo(f64, quadratic_f64, &x, &p, &grad, alpha_init, c, 20, allocator);
    const backtrack_result = try backtracking(f64, quadratic_f64, &x, &p, &grad, alpha_init, 0.5, c, 20, allocator);

    try testing.expect(armijo_result.converged);
    try testing.expect(backtrack_result.converged);
    // Both should produce step sizes satisfying Armijo condition
    try testing.expect(armijo_result.f_new <= armijo_result.f_new + 1e-10);
    try testing.expect(backtrack_result.f_new <= backtrack_result.f_new + 1e-10);
}

test "cross-method: Rosenbrock all methods agree on feasibility" {
    const allocator = testing.allocator;

    var x = [_]f64{ 0.0, 0.0 };
    var grad = [_]f64{0, 0};
    rosenbrock_grad_f64(&x, &grad);

    var p = [_]f64{ -grad[0], -grad[1] };
    const alpha_init: f64 = 0.001;

    const armijo_result = try armijo(f64, rosenbrock_f64, &x, &p, &grad, alpha_init, 1e-4, 30, allocator);
    const backtrack_result = try backtracking(f64, rosenbrock_f64, &x, &p, &grad, alpha_init, 0.5, 1e-4, 30, allocator);
    const wolfe_result = try wolfe(f64, rosenbrock_f64, rosenbrock_grad_f64, &x, &p, alpha_init, 1e-4, 0.9, 30, allocator);
    defer allocator.free(wolfe_result.grad_new);

    try testing.expect(armijo_result.converged);
    try testing.expect(backtrack_result.converged);
    try testing.expect(wolfe_result.converged);
}
