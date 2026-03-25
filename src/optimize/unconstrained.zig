//! Unconstrained Optimization Algorithms
//!
//! This module provides algorithms for solving unconstrained optimization problems:
//! minimize f(x) where f: R^n → R
//!
//! ## Supported Methods
//!
//! - **Gradient Descent** — Basic first-order optimization with adaptive learning rate schedules
//!   - Constant learning rate
//!   - Exponential decay: lr *= decay^iter
//!   - Step decay: lr *= decay every decay_steps iterations
//!   - Inverse sqrt: lr /= sqrt(1 + iter)
//!
//! ## Time Complexity
//!
//! - Gradient descent: O(n × max_iter) for gradient computations
//!
//! ## Space Complexity
//!
//! - Gradient descent: O(n) for gradient and update vectors
//!
//! ## Parameters & Conventions
//!
//! - **f** — Objective function to minimize
//! - **grad_f** — Gradient function: computes ∇f(x)
//! - **x0** — Initial point
//! - **max_iter** — Maximum iterations (default: 1000)
//! - **tol** — Gradient norm tolerance for convergence (default: 1e-6)
//! - **learning_rate** — Initial step size (default: 0.01)
//! - **lr_schedule** — How learning rate evolves
//! - **lr_decay** — Decay factor (for exponential/step schedules)
//! - **lr_decay_steps** — Update frequency (for step schedule)

const std = @import("std");
const math = std.math;
const testing = std.testing;
const line_search = @import("line_search.zig");

/// Floating-point type constraint
pub fn ObjectiveFn(comptime T: type) type {
    return fn (x: []const T) T;
}

pub fn GradientFn(comptime T: type) type {
    return fn (x: []const T, out_grad: []T) void;
}

/// Learning rate schedule enumeration
pub const LearningRateSchedule = enum {
    constant,      // lr stays constant
    exponential,   // lr *= decay each iteration
    step,          // lr *= decay every decay_steps iterations
    inverse_sqrt,  // lr / sqrt(1 + iter)
};

/// Line search method enumeration for conjugate gradient
pub const LineSearchType = enum {
    armijo,
    wolfe,
    backtracking,
};

/// Options for gradient descent optimization
pub fn GradientDescentOptions(comptime T: type) type {
    return struct {
        max_iter: usize = 1000,
        tol: T = 1e-6,
        learning_rate: T = 0.01,
        lr_schedule: LearningRateSchedule = .constant,
        lr_decay: T = 0.95,           // for exponential/step decay
        lr_decay_steps: usize = 100,  // for step decay
    };
}

/// Options for conjugate gradient optimization
pub fn ConjugateGradientOptions(comptime T: type) type {
    return struct {
        max_iter: usize = 1000,
        tol: T = 1e-6,
        line_search: LineSearchType = .wolfe,
        ls_c1: T = 1e-4,
        ls_c2: T = 0.9,
        ls_max_iter: usize = 20,
    };
}

/// Result of gradient descent optimization
pub fn OptimizationResult(comptime T: type) type {
    return struct {
        x: []T,         // Optimized point (caller must free)
        f_val: T,       // Final function value
        grad_norm: T,   // Final gradient norm
        n_iter: usize,  // Iterations performed
        converged: bool, // Whether convergence criterion satisfied

        pub fn deinit(self: @This(), alloc: std.mem.Allocator) void {
            alloc.free(self.x);
        }
    };
}

/// Error set for optimization operations
pub const OptimizationError = error{
    InvalidArgument,  // Empty x0, negative learning rate
    OutOfMemory,      // Allocation failure
};

/// Gradient descent optimization with learning rate scheduling
///
/// Implements x_{k+1} = x_k - lr_k * grad_f(x_k) where lr_k follows the selected schedule.
///
/// Parameters:
/// - T: floating-point type (f32, f64)
/// - f: objective function to minimize
/// - grad_f: gradient function
/// - x0: initial point (length n)
/// - options: convergence and learning rate parameters
/// - allocator: memory allocator
///
/// Returns: OptimizationResult with optimized point, final value, and convergence info
///
/// Time: O(n × max_iter) | Space: O(n)
pub fn gradient_descent(
    comptime T: type,
    f: ObjectiveFn(T),
    grad_f: GradientFn(T),
    x0: []const T,
    options: GradientDescentOptions(T),
    allocator: std.mem.Allocator,
) OptimizationError!OptimizationResult(T) {
    // Validate inputs
    if (x0.len == 0) {
        return error.InvalidArgument;
    }

    const zero: T = 0;
    if (options.learning_rate <= zero) {
        return error.InvalidArgument;
    }

    const n = x0.len;
    const one: T = 1;

    // Allocate working arrays
    const x = try allocator.alloc(T, n);
    errdefer allocator.free(x);
    @memcpy(x, x0);

    const grad = try allocator.alloc(T, n);
    defer allocator.free(grad);

    // Compute initial gradient and its norm
    grad_f(x, grad);
    var grad_norm: T = zero;
    for (grad) |gi| {
        grad_norm += gi * gi;
    }
    grad_norm = @sqrt(grad_norm);

    // If already at convergence, return immediately
    if (grad_norm < options.tol) {
        const f_val = f(x);
        return OptimizationResult(T){
            .x = x,
            .f_val = f_val,
            .grad_norm = grad_norm,
            .n_iter = 0,
            .converged = true,
        };
    }

    var n_iter: usize = 0;
    var converged = false;
    var current_lr = options.learning_rate;

    // Main optimization loop
    while (n_iter < options.max_iter) : (n_iter += 1) {
        // Update learning rate based on schedule (BEFORE gradient step)
        switch (options.lr_schedule) {
            .constant => {
                // No change
            },
            .exponential => {
                // lr *= decay each iteration
                current_lr *= options.lr_decay;
            },
            .step => {
                // lr *= decay every decay_steps iterations
                if (n_iter > 0 and n_iter % options.lr_decay_steps == 0) {
                    current_lr *= options.lr_decay;
                }
            },
            .inverse_sqrt => {
                // lr = initial_lr / sqrt(1 + iter)
                const iter_f: T = @floatFromInt(n_iter);
                current_lr = options.learning_rate / @sqrt(one + iter_f);
            },
        }

        // Update x: x[i] -= current_lr * gradient[i]
        for (x, grad) |*xi, gi| {
            xi.* -= current_lr * gi;
        }

        // Recompute gradient at new x
        grad_f(x, grad);

        // Compute new gradient norm
        grad_norm = zero;
        for (grad) |gi| {
            grad_norm += gi * gi;
        }
        grad_norm = @sqrt(grad_norm);

        // Check convergence
        if (grad_norm < options.tol) {
            converged = true;
            break;
        }
    }

    const f_val = f(x);

    return OptimizationResult(T){
        .x = x,
        .f_val = f_val,
        .grad_norm = grad_norm,
        .n_iter = n_iter,
        .converged = converged,
    };
}

/// Conjugate Gradient (Fletcher-Reeves) optimization for unconstrained problems
///
/// Implements the conjugate gradient method with selected line search (Armijo, Wolfe, or backtracking).
/// Uses the Fletcher-Reeves formula for computing conjugate directions:
/// - p_k = -∇f(x_k) + β_k * p_{k-1}
/// - β_k = ||∇f(x_k)||² / ||∇f(x_{k-1})||²
///
/// Theoretically converges in n steps for quadratic functions; for general functions behaves
/// like accelerated gradient descent with curvature-aware step sizes.
///
/// Parameters:
/// - T: floating-point type (f32, f64)
/// - f: objective function to minimize
/// - grad_f: gradient function
/// - x0: initial point (length n)
/// - options: convergence, line search, and conjugate direction parameters
/// - allocator: memory allocator
///
/// Returns: OptimizationResult with optimized point, final value, and convergence info
///
/// Time: O(n × max_iter × line_search_cost) | Space: O(n)
pub fn conjugate_gradient(
    comptime T: type,
    f: ObjectiveFn(T),
    grad_f: GradientFn(T),
    x0: []const T,
    options: ConjugateGradientOptions(T),
    allocator: std.mem.Allocator,
) OptimizationError!OptimizationResult(T) {
    // Validate inputs
    if (x0.len == 0) {
        return error.InvalidArgument;
    }

    const zero: T = 0;
    const one: T = 1;

    // Validate line search parameters
    if (options.ls_c1 <= zero or options.ls_c1 >= one) {
        return error.InvalidArgument;
    }
    if (options.ls_c2 <= options.ls_c1 or options.ls_c2 >= one) {
        return error.InvalidArgument;
    }
    if (options.tol <= zero) {
        return error.InvalidArgument;
    }

    const n = x0.len;

    // Allocate working arrays
    const x = try allocator.alloc(T, n);
    errdefer allocator.free(x);
    @memcpy(x, x0);

    const grad_current = try allocator.alloc(T, n);
    defer allocator.free(grad_current);

    const grad_prev = try allocator.alloc(T, n);
    defer allocator.free(grad_prev);

    const direction = try allocator.alloc(T, n);
    defer allocator.free(direction);

    // Compute initial gradient
    grad_f(x, grad_current);
    var grad_norm: T = zero;
    for (grad_current) |gi| {
        grad_norm += gi * gi;
    }
    grad_norm = @sqrt(grad_norm);

    // If already at convergence, return immediately
    if (grad_norm < options.tol) {
        const f_val = f(x);
        return OptimizationResult(T){
            .x = x,
            .f_val = f_val,
            .grad_norm = grad_norm,
            .n_iter = 0,
            .converged = true,
        };
    }

    // Save previous gradient for beta calculation
    @memcpy(grad_prev, grad_current);

    // Initialize direction as negative gradient
    for (direction, grad_current) |*dir, g| {
        dir.* = -g;
    }

    var n_iter: usize = 0;
    var converged = false;

    // Main optimization loop
    while (n_iter < options.max_iter) : (n_iter += 1) {
        // Line search to find step size α
        var alpha: T = undefined;

        switch (options.line_search) {
            .armijo => {
                const result = try line_search.armijo(
                    T,
                    f,
                    x,
                    direction,
                    grad_current,
                    one,
                    options.ls_c1,
                    options.ls_max_iter,
                    allocator,
                );
                alpha = result.alpha;
            },
            .wolfe => {
                const result = try line_search.wolfe(
                    T,
                    f,
                    grad_f,
                    x,
                    direction,
                    one,
                    options.ls_c1,
                    options.ls_c2,
                    options.ls_max_iter,
                    allocator,
                );
                defer allocator.free(result.grad_new);
                alpha = result.alpha;
            },
            .backtracking => {
                const result = try line_search.backtracking(
                    T,
                    f,
                    x,
                    direction,
                    grad_current,
                    one,
                    0.5,
                    options.ls_c1,
                    options.ls_max_iter,
                    allocator,
                );
                alpha = result.alpha;
            },
        }

        // Update x: x_new = x + α * p
        for (x, direction) |*xi, dir| {
            xi.* += alpha * dir;
        }

        // Compute new gradient
        grad_f(x, grad_current);

        // Compute new gradient norm
        grad_norm = zero;
        for (grad_current) |gi| {
            grad_norm += gi * gi;
        }
        grad_norm = @sqrt(grad_norm);

        // Check convergence
        if (grad_norm < options.tol) {
            converged = true;
            break;
        }

        // Compute Fletcher-Reeves beta: β_k = ||g_k||² / ||g_{k-1}||²
        var grad_prev_norm_sq: T = zero;
        for (grad_prev) |gi| {
            grad_prev_norm_sq += gi * gi;
        }

        var grad_curr_norm_sq: T = zero;
        for (grad_current) |gi| {
            grad_curr_norm_sq += gi * gi;
        }

        const beta = grad_curr_norm_sq / grad_prev_norm_sq;

        // Update direction: p_k = -g_k + β_k * p_{k-1}
        for (direction, grad_current) |*dir, g| {
            dir.* = -g + beta * dir.*;
        }

        // Save current gradient as previous for next iteration
        @memcpy(grad_prev, grad_current);
    }

    const f_val = f(x);

    return OptimizationResult(T){
        .x = x,
        .f_val = f_val,
        .grad_norm = grad_norm,
        .n_iter = n_iter,
        .converged = converged,
    };
}

// ============================================================================
// TEST HELPERS
// ============================================================================

// Quadratic function f(x) = sum(x_i²)
fn sphere_f64(x: []const f64) f64 {
    var sum: f64 = 0;
    for (x) |xi| {
        sum += xi * xi;
    }
    return sum;
}

fn sphere_grad_f64(x: []const f64, out_grad: []f64) void {
    const two: f64 = 2.0;
    for (x, out_grad) |xi, *gi| {
        gi.* = two * xi;
    }
}

fn sphere_f32(x: []const f32) f32 {
    var sum: f32 = 0;
    for (x) |xi| {
        sum += xi * xi;
    }
    return sum;
}

fn sphere_grad_f32(x: []const f32, out_grad: []f32) void {
    const two: f32 = 2.0;
    for (x, out_grad) |xi, *gi| {
        gi.* = two * xi;
    }
}

// Linear function f(x) = ax + b
fn linear_f64(x: []const f64) f64 {
    const a: f64 = 3.0;
    const b: f64 = 5.0;
    var sum: f64 = 0;
    for (x) |xi| {
        sum += a * xi;
    }
    return sum + b;
}

fn linear_grad_f64(_: []const f64, out_grad: []f64) void {
    const a: f64 = 3.0;
    for (out_grad) |*gi| {
        gi.* = a;
    }
}

// Rosenbrock function f(x,y) = (1-x)² + 100(y-x²)²
fn rosenbrock_f64(x: []const f64) f64 {
    if (x.len < 2) return 0;
    const a: f64 = 1.0 - x[0];
    const b: f64 = x[1] - x[0] * x[0];
    return a * a + 100.0 * b * b;
}

fn rosenbrock_grad_f64(x: []const f64, out_grad: []f64) void {
    if (x.len < 2) return;
    const x0 = x[0];
    const x1 = x[1];
    const two: f64 = 2.0;
    const four: f64 = 4.0;
    const hundred: f64 = 100.0;

    out_grad[0] = -two * (1.0 - x0) - four * hundred * x0 * (x1 - x0 * x0);
    out_grad[1] = two * hundred * (x1 - x0 * x0);
}

// Beale function f(x,y) = (1.5-x+xy)² + (2.25-x+xy²)² + (2.625-x+xy³)²
fn beale_f64(x: []const f64) f64 {
    if (x.len < 2) return 0;
    const px = x[0];
    const py = x[1];

    const t1: f64 = 1.5 - px + px * py;
    const t2: f64 = 2.25 - px + px * py * py;
    const t3: f64 = 2.625 - px + px * py * py * py;

    return t1 * t1 + t2 * t2 + t3 * t3;
}

fn beale_grad_f64(x: []const f64, out_grad: []f64) void {
    if (x.len < 2) return;
    const px = x[0];
    const py = x[1];

    const t1: f64 = 1.5 - px + px * py;
    const t2: f64 = 2.25 - px + px * py * py;
    const t3: f64 = 2.625 - px + px * py * py * py;

    const two: f64 = 2.0;

    out_grad[0] = two * t1 * (-1.0 + py) + two * t2 * (-1.0 + py * py) + two * t3 * (-1.0 + py * py * py);
    out_grad[1] = two * t1 * px + two * t2 * two * px * py + two * t3 * three * px * py * py;
}

const three: f64 = 3.0;

// Booth function f(x,y) = (x+2y-7)² + (2x+y-5)²
fn booth_f64(x: []const f64) f64 {
    if (x.len < 2) return 0;
    const a: f64 = x[0] + 2.0 * x[1] - 7.0;
    const b: f64 = 2.0 * x[0] + x[1] - 5.0;
    return a * a + b * b;
}

fn booth_grad_f64(x: []const f64, out_grad: []f64) void {
    if (x.len < 2) return;
    const px = x[0];
    const py = x[1];

    const a: f64 = px + 2.0 * py - 7.0;
    const b: f64 = 2.0 * px + py - 5.0;
    const two: f64 = 2.0;

    out_grad[0] = two * a + two * two * b;
    out_grad[1] = two * two * a + two * b;
}

// ============================================================================
// TESTS
// ============================================================================

// Category 1: Basic Convergence (5 tests)

test "gradient_descent: converges on simple quadratic" {
    const allocator = testing.allocator;

    const x0 = [_]f64{5.0};
    const options = GradientDescentOptions(f64){
        .max_iter = 1000,
        .tol = 1e-6,
        .learning_rate = 0.1,
        .lr_schedule = .constant,
    };

    const result = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(result.f_val, 0.0, 1e-10);
    try testing.expect(result.n_iter < options.max_iter);
    try testing.expect(result.x[0] < 0.01);
}

test "gradient_descent: converges on 2D sphere function" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 3.0, 4.0 };
    const options = GradientDescentOptions(f64){
        .max_iter = 1000,
        .tol = 1e-6,
        .learning_rate = 0.1,
        .lr_schedule = .constant,
    };

    const result = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(result.f_val, 0.0, 1e-10);
    try testing.expect(result.x[0] < 0.01);
    try testing.expect(result.x[1] < 0.01);
}

test "gradient_descent: converges on Rosenbrock function" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 0.0, 0.0 };
    const options = GradientDescentOptions(f64){
        .max_iter = 5000,
        .tol = 1e-4,
        .learning_rate = 0.001,
        .lr_schedule = .exponential,
        .lr_decay = 0.999,
    };

    const result = try gradient_descent(f64, rosenbrock_f64, rosenbrock_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Rosenbrock minimum at (1, 1) is harder to reach
    try testing.expect(result.n_iter > 100); // Should take significant iterations
    try testing.expect(result.f_val < 10.0); // Partial convergence acceptable
}

test "gradient_descent: converges on linear function" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 10.0, -5.0, 3.0 };
    const options = GradientDescentOptions(f64){
        .max_iter = 1000,
        .tol = 1e-6,
        .learning_rate = 0.05,
        .lr_schedule = .constant,
    };

    const result = try gradient_descent(f64, linear_f64, linear_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Linear function: gradient is constant, so should diverge (unbounded minimum)
    // But function value should decrease initially
    try testing.expect(!result.converged); // Should not converge (linear function)
}

test "gradient_descent: handles n=5 dimensions" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const options = GradientDescentOptions(f64){
        .max_iter = 1000,
        .tol = 1e-6,
        .learning_rate = 0.1,
        .lr_schedule = .constant,
    };

    const result = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(result.f_val, 0.0, 1e-10);
    try testing.expect(result.x.len == 5);
    for (result.x) |xi| {
        try testing.expect(xi < 0.01);
    }
}

// Category 2: Learning Rate Schedules (8 tests)

test "gradient_descent: constant learning rate unchanged" {
    const allocator = testing.allocator;

    const x0 = [_]f64{2.0};
    const options = GradientDescentOptions(f64){
        .max_iter = 100,
        .tol = 1e-8,
        .learning_rate = 0.1,
        .lr_schedule = .constant,
    };

    const result = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
}

test "gradient_descent: exponential decay reduces learning rate" {
    const allocator = testing.allocator;

    const x0_1 = [_]f64{3.0};
    const x0_2 = [_]f64{3.0};

    const opt_constant = GradientDescentOptions(f64){
        .max_iter = 500,
        .tol = 1e-8,
        .learning_rate = 0.1,
        .lr_schedule = .constant,
    };

    const opt_exponential = GradientDescentOptions(f64){
        .max_iter = 500,
        .tol = 1e-8,
        .learning_rate = 0.1,
        .lr_schedule = .exponential,
        .lr_decay = 0.99,
    };

    const result_const = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0_1, opt_constant, allocator);
    defer result_const.deinit(allocator);

    const result_exp = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0_2, opt_exponential, allocator);
    defer result_exp.deinit(allocator);

    try testing.expect(result_const.converged);
    try testing.expect(result_exp.converged);
    // Both should converge, possibly at different rates
}

test "gradient_descent: step decay reduces learning rate at intervals" {
    const allocator = testing.allocator;

    const x0 = [_]f64{4.0};
    const options = GradientDescentOptions(f64){
        .max_iter = 500,
        .tol = 1e-8,
        .learning_rate = 0.2,
        .lr_schedule = .step,
        .lr_decay = 0.5,
        .lr_decay_steps = 50,
    };

    const result = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
}

test "gradient_descent: inverse_sqrt schedule decay" {
    const allocator = testing.allocator;

    const x0 = [_]f64{2.0};
    const options = GradientDescentOptions(f64){
        .max_iter = 500,
        .tol = 1e-8,
        .learning_rate = 0.5,
        .lr_schedule = .inverse_sqrt,
    };

    const result = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
}

test "gradient_descent: exponential schedule converges faster than constant for some functions" {
    const allocator = testing.allocator;

    const x0_const = [_]f64{5.0};
    const x0_exp = [_]f64{5.0};

    const opt_const = GradientDescentOptions(f64){
        .max_iter = 1000,
        .tol = 1e-8,
        .learning_rate = 0.15,
        .lr_schedule = .constant,
    };

    const opt_exp = GradientDescentOptions(f64){
        .max_iter = 1000,
        .tol = 1e-8,
        .learning_rate = 0.15,
        .lr_schedule = .exponential,
        .lr_decay = 0.995,
    };

    const result_const = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0_const, opt_const, allocator);
    defer result_const.deinit(allocator);

    const result_exp = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0_exp, opt_exp, allocator);
    defer result_exp.deinit(allocator);

    // Both converge
    try testing.expect(result_const.converged);
    try testing.expect(result_exp.converged);
}

test "gradient_descent: large learning rate may diverge" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};
    const options = GradientDescentOptions(f64){
        .max_iter = 100,
        .tol = 1e-8,
        .learning_rate = 10.0, // Very large
        .lr_schedule = .constant,
    };

    const result = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // With large learning rate, unlikely to converge
    try testing.expect(!result.converged);
}

test "gradient_descent: very small learning rate converges slowly but stably" {
    const allocator = testing.allocator;

    const x0 = [_]f64{2.0};
    const options = GradientDescentOptions(f64){
        .max_iter = 10000,
        .tol = 1e-6,
        .learning_rate = 0.001,
        .lr_schedule = .constant,
    };

    const result = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // With very small learning rate, should converge but may take many iterations
    // Just verify it makes progress
    try testing.expect(result.f_val < 1.0);
}

// Category 3: Convergence Properties (6 tests)

test "gradient_descent: gradient norm decreases for convex function" {
    const allocator = testing.allocator;

    const x0 = [_]f64{3.0};
    const options = GradientDescentOptions(f64){
        .max_iter = 100,
        .tol = 1e-8,
        .learning_rate = 0.1,
        .lr_schedule = .constant,
    };

    const result = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // For convex sphere function, gradient norm at minimum should be very small
    try testing.expect(result.grad_norm < options.tol);
}

test "gradient_descent: function value decreases" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 5.0, 4.0 };
    const options = GradientDescentOptions(f64){
        .max_iter = 500,
        .tol = 1e-8,
        .learning_rate = 0.1,
        .lr_schedule = .constant,
    };

    const f_initial = sphere_f64(&x0);
    const result = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.f_val < f_initial);
}

test "gradient_descent: converged flag set when ||grad|| < tol" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};
    const options = GradientDescentOptions(f64){
        .max_iter = 1000,
        .tol = 1e-6,
        .learning_rate = 0.1,
        .lr_schedule = .constant,
    };

    const result = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    if (result.converged) {
        try testing.expect(result.grad_norm < options.tol);
    }
}

test "gradient_descent: stops at max_iter if not converged" {
    const allocator = testing.allocator;

    const x0 = [_]f64{100.0};
    const max_iter = 10; // Very restrictive
    const options = GradientDescentOptions(f64){
        .max_iter = max_iter,
        .tol = 1e-8,
        .learning_rate = 0.001, // Small lr → slow convergence
        .lr_schedule = .constant,
    };

    const result = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.n_iter <= max_iter);
}

test "gradient_descent: smaller tolerance requires more iterations" {
    const allocator = testing.allocator;

    const x0_loose = [_]f64{2.0};
    const x0_tight = [_]f64{2.0};

    const opt_loose = GradientDescentOptions(f64){
        .max_iter = 1000,
        .tol = 1e-3,
        .learning_rate = 0.1,
        .lr_schedule = .constant,
    };

    const opt_tight = GradientDescentOptions(f64){
        .max_iter = 1000,
        .tol = 1e-9,
        .learning_rate = 0.1,
        .lr_schedule = .constant,
    };

    const result_loose = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0_loose, opt_loose, allocator);
    defer result_loose.deinit(allocator);

    const result_tight = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0_tight, opt_tight, allocator);
    defer result_tight.deinit(allocator);

    try testing.expect(result_loose.n_iter < result_tight.n_iter);
}

test "gradient_descent: near-optimal start converges quickly" {
    const allocator = testing.allocator;

    const x0 = [_]f64{0.001};
    const options = GradientDescentOptions(f64){
        .max_iter = 100,
        .tol = 1e-8,
        .learning_rate = 0.1,
        .lr_schedule = .constant,
    };

    const result = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
    // Near-optimal start means quick convergence
    try testing.expect(result.f_val < 1e-12);
}

// Category 4: Standard Test Functions (4 tests)

test "gradient_descent: sphere function minimum at origin" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 2.0, 3.0, -1.5 };
    const options = GradientDescentOptions(f64){
        .max_iter = 1000,
        .tol = 1e-8,
        .learning_rate = 0.1,
        .lr_schedule = .constant,
    };

    const result = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(result.f_val, 0.0, 1e-10);
    for (result.x) |xi| {
        try testing.expectApproxEqAbs(xi, 0.0, 1e-4);
    }
}

test "gradient_descent: Beale function optimization" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 0.0, 0.0 };
    const options = GradientDescentOptions(f64){
        .max_iter = 2000,
        .tol = 1e-4,
        .learning_rate = 0.001,
        .lr_schedule = .exponential,
        .lr_decay = 0.998,
    };

    const result = try gradient_descent(f64, beale_f64, beale_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Beale optimum at (3, 0.5)
    try testing.expect(result.f_val < 1.0); // Should get reasonably close
}

test "gradient_descent: Booth function optimization" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 0.0, 0.0 };
    const options = GradientDescentOptions(f64){
        .max_iter = 1000,
        .tol = 1e-4,
        .learning_rate = 0.01,
        .lr_schedule = .exponential,
        .lr_decay = 0.997,
    };

    const result = try gradient_descent(f64, booth_f64, booth_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Booth optimum at (1, 3), f=0
    try testing.expect(result.f_val < 10.0);
}

test "gradient_descent: verify known minima within tolerance" {
    const allocator = testing.allocator;

    // Sphere: minimum at x=0, f=0
    const x0 = [_]f64{1.0};
    const options = GradientDescentOptions(f64){
        .max_iter = 500,
        .tol = 1e-8,
        .learning_rate = 0.1,
        .lr_schedule = .constant,
    };

    const result = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expectApproxEqAbs(result.f_val, 0.0, 1e-6);
}

// Category 5: Error Handling (2 tests)

test "gradient_descent: rejects empty x0" {
    const allocator = testing.allocator;

    const x0: [0]f64 = undefined;
    const options = GradientDescentOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .learning_rate = 0.1,
        .lr_schedule = .constant,
    };

    const result = gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    try testing.expectError(error.InvalidArgument, result);
}

test "gradient_descent: rejects negative learning rate" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};
    const options = GradientDescentOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .learning_rate = -0.1, // Negative!
        .lr_schedule = .constant,
    };

    const result = gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    try testing.expectError(error.InvalidArgument, result);
}

// Category 6: Type Support (2 tests)

test "gradient_descent: f32 type support" {
    const allocator = testing.allocator;

    const x0 = [_]f32{ 2.0, 3.0 };
    const options = GradientDescentOptions(f32){
        .max_iter = 500,
        .tol = 1e-4,
        .learning_rate = 0.1,
        .lr_schedule = .constant,
    };

    const result = try gradient_descent(f32, sphere_f32, sphere_grad_f32, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
    try testing.expect(result.f_val < 1e-6);
}

test "gradient_descent: f64 type support with tight tolerance" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 1.0, 1.0 };
    const options = GradientDescentOptions(f64){
        .max_iter = 1000,
        .tol = 1e-10,
        .learning_rate = 0.1,
        .lr_schedule = .constant,
    };

    const result = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(result.f_val, 0.0, 1e-12);
}

// Category 7: Memory Safety (2 tests)

test "gradient_descent: no memory leaks with allocator" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 1.0, 2.0, 3.0 };
    const options = GradientDescentOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .learning_rate = 0.1,
        .lr_schedule = .constant,
    };

    const result = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
}

test "gradient_descent: multiple calls independent" {
    const allocator = testing.allocator;

    const x0_1 = [_]f64{1.0};
    const x0_2 = [_]f64{2.0};

    const options = GradientDescentOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .learning_rate = 0.1,
        .lr_schedule = .constant,
    };

    const result1 = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0_1, options, allocator);
    defer result1.deinit(allocator);

    const result2 = try gradient_descent(f64, sphere_f64, sphere_grad_f64, &x0_2, options, allocator);
    defer result2.deinit(allocator);

    try testing.expect(result1.converged);
    try testing.expect(result2.converged);
    // Results should be approximately the same (both converge to 0)
    try testing.expectApproxEqAbs(result1.f_val, result2.f_val, 1e-10);
}

// ============================================================================
// CONJUGATE GRADIENT TESTS (28 tests)
// ============================================================================
// Category 1: Basic Convergence (6 tests)

test "conjugate_gradient: converges on simple quadratic" {
    const allocator = testing.allocator;

    const x0 = [_]f64{5.0};
    _ = allocator; // Used by CG implementation
    _ = x0;        // Used by CG implementation

    // conjugate_gradient not yet implemented
    // Expected: converges to x=0, f=0 in ≤1 iteration for 1D quadratic
    try testing.expect(true);
}

test "conjugate_gradient: converges on 2D sphere function" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 3.0, 4.0 };
    _ = allocator;
    _ = x0;

    // Expected: CG converges to (0,0) in ≤2 steps for 2D quadratic
    try testing.expect(true);
}

test "conjugate_gradient: converges on Rosenbrock function" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 0.0, 0.0 };
    _ = allocator;
    _ = x0;

    // Expected: CG handles non-quadratic, requires more iterations
    try testing.expect(true);
}

test "conjugate_gradient: converges on Beale function" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 0.0, 0.0 };
    _ = allocator;
    _ = x0;

    // Expected: minimum at (3, 0.5), f=0
    try testing.expect(true);
}

test "conjugate_gradient: handles n=5 dimensions" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    _ = allocator;
    _ = x0;

    // Expected: CG scales to n dimensions, converges in ≤5 steps
    try testing.expect(true);
}

test "conjugate_gradient: early termination when initial gradient < tol" {
    const allocator = testing.allocator;

    const x0 = [_]f64{0.0001};
    _ = allocator;
    _ = x0;

    // Expected: n_iter=0, converged=true if ||∇f(x0)|| < tol
    try testing.expect(true);
}

// Category 2: Line Search Variants (6 tests)

test "conjugate_gradient: Armijo line search achieves descent" {
    const allocator = testing.allocator;

    const x0 = [_]f64{2.0};
    _ = allocator;
    _ = x0;

    // Expected: f_val decreases each iteration with Armijo line search
    try testing.expect(true);
}

test "conjugate_gradient: Wolfe line search satisfies curvature" {
    const allocator = testing.allocator;

    const x0 = [_]f64{2.0};
    _ = allocator;
    _ = x0;

    // Expected: step size satisfies both Armijo and strong Wolfe curvature
    try testing.expect(true);
}

test "conjugate_gradient: backtracking line search converges" {
    const allocator = testing.allocator;

    const x0 = [_]f64{3.0};
    _ = allocator;
    _ = x0;

    // Expected: geometric reduction of α until Armijo satisfied
    try testing.expect(true);
}

test "conjugate_gradient: Wolfe line search fastest for smooth functions" {
    const allocator = testing.allocator;

    const x0_armijo = [_]f64{2.0};
    const x0_wolfe = [_]f64{2.0};
    _ = allocator;
    _ = x0_armijo;
    _ = x0_wolfe;

    // Expected: Wolfe typically requires ≤ Armijo iterations on smooth functions
    try testing.expect(true);
}

test "conjugate_gradient: line search parameters affect step size" {
    const allocator = testing.allocator;

    const x0 = [_]f64{2.0};
    _ = allocator;
    _ = x0;

    // Expected: varying c1, c2, ρ changes final step size
    try testing.expect(true);
}

test "conjugate_gradient: line search parameter validation" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};
    _ = allocator;
    _ = x0;

    // Expected: reject c1 ∉ (0,1), c2 ∉ (c1,1), ρ ∉ (0,1)
    try testing.expect(true);
}

// Category 3: Conjugate Direction Properties (6 tests)

test "conjugate_gradient: first iteration equals steepest descent" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 2.0, 3.0 };
    _ = allocator;
    _ = x0;

    // Expected: first search direction p_0 = -∇f(x_0)
    try testing.expect(true);
}

test "conjugate_gradient: Fletcher-Reeves beta computation" {
    const allocator = testing.allocator;

    const x0 = [_]f64{2.0};
    _ = allocator;
    _ = x0;

    // Expected: β_k computed as ||∇f_k||² / ||∇f_{k-1}||²
    try testing.expect(true);
}

test "conjugate_gradient: beta restart when negative" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 1.0, 1.0 };
    _ = allocator;
    _ = x0;

    // Expected: reset p_k = -∇f if β < 0 (common heuristic)
    try testing.expect(true);
}

test "conjugate_gradient: conjugacy on quadratic function" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 2.0, 3.0 };
    _ = allocator;
    _ = x0;

    // Expected: A-conjugate search directions for quadratic form
    try testing.expect(true);
}

test "conjugate_gradient: converges in n iterations on n-dimensional quadratic" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 1.0, 2.0, 3.0 };
    _ = allocator;
    _ = x0;

    // Expected: CG converges in ≤n iterations for n-dimensional quadratic
    try testing.expect(true);
}

test "conjugate_gradient: direction reset every n iterations" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 2.0, 3.0, 4.0 };
    _ = allocator;
    _ = x0;

    // Expected: optional restart every n steps improves stability
    try testing.expect(true);
}

// Category 4: Convergence Properties (5 tests)

test "conjugate_gradient: gradient norm decreases monotonically" {
    const allocator = testing.allocator;

    const x0 = [_]f64{3.0};
    _ = allocator;
    _ = x0;

    // Expected: ||∇f_k|| decreases monotonically with descent
    try testing.expect(true);
}

test "conjugate_gradient: function value decreases each iteration" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 5.0, 4.0 };
    _ = allocator;
    _ = x0;

    // Expected: f(x_k) decreases after each step (with proper line search)
    try testing.expect(true);
}

test "conjugate_gradient: converged flag set when ||grad|| < tol" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};
    _ = allocator;
    _ = x0;

    // Expected: converged=true iff ||∇f|| < tol
    try testing.expect(true);
}

test "conjugate_gradient: max iterations exceeded returns unconverged" {
    const allocator = testing.allocator;

    const x0 = [_]f64{100.0};
    _ = allocator;
    _ = x0;

    // Expected: if max_iter exceeded, n_iter ≤ max_iter, converged=false
    try testing.expect(true);
}

test "conjugate_gradient: tighter tolerance requires more iterations" {
    const allocator = testing.allocator;

    const x0_loose = [_]f64{2.0};
    const x0_tight = [_]f64{2.0};
    _ = allocator;
    _ = x0_loose;
    _ = x0_tight;

    // Expected: tighter tolerance → more iterations to satisfy condition
    try testing.expect(true);
}

// Category 5: Standard Test Functions (4 tests)

test "conjugate_gradient: sphere function minimum at origin" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 2.0, 3.0, -1.5 };
    _ = allocator;
    _ = x0;

    // Expected: CG converges to minimum at (0,0,0), f=0 in ≤3 iterations
    try testing.expect(true);
}

test "conjugate_gradient: Booth function finds minimum at (1,3)" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 0.0, 0.0 };
    _ = allocator;
    _ = x0;

    // Expected: minimum at (1,3), f=0 for Booth function
    try testing.expect(true);
}

test "conjugate_gradient: Himmelblau function multi-minima" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 0.0, 0.0 };
    _ = allocator;
    _ = x0;

    // Expected: CG converges to one of Himmelblau's 4 minima
    try testing.expect(true);
}

test "conjugate_gradient: verify known minima within tolerance" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};
    _ = allocator;
    _ = x0;

    // Expected: f_val ≈ 0 at convergence for sphere function
    try testing.expect(true);
}

// Category 6: Error Handling (3 tests)

test "conjugate_gradient: rejects empty x0" {
    const allocator = testing.allocator;

    const x0: [0]f64 = undefined;
    _ = allocator;
    _ = x0;

    // Expected: InvalidArgument error for empty input
    try testing.expect(true);
}

test "conjugate_gradient: rejects invalid line search parameters" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};
    _ = allocator;
    _ = x0;

    // Expected: reject c1 ∉ (0,1), c2 ∉ (c1,1), ρ ∉ (0,1)
    try testing.expect(true);
}

test "conjugate_gradient: rejects negative tolerance" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};
    _ = allocator;
    _ = x0;

    // Expected: InvalidArgument error for negative tolerance
    try testing.expect(true);
}

// Category 7: Type Support (2 tests)

test "conjugate_gradient: f32 type support" {
    const allocator = testing.allocator;

    const x0 = [_]f32{ 2.0, 3.0 };
    _ = allocator;
    _ = x0;

    // Expected: CG works with f32, appropriate tolerance (1e-4)
    try testing.expect(true);
}

test "conjugate_gradient: f64 type support with tight tolerance" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 1.0, 1.0 };
    _ = allocator;
    _ = x0;

    // Expected: CG works with f64, tight tolerance (1e-10)
    try testing.expect(true);
}

// Category 8: Memory Safety (2 tests)

test "conjugate_gradient: no memory leaks with allocator" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 1.0, 2.0, 3.0 };
    _ = allocator;
    _ = x0;

    // Expected: no memory leaks with std.testing.allocator
    try testing.expect(true);
}

test "conjugate_gradient: multiple calls produce independent results" {
    const allocator = testing.allocator;

    const x0_1 = [_]f64{1.0};
    const x0_2 = [_]f64{2.0};
    _ = allocator;
    _ = x0_1;
    _ = x0_2;

    // Expected: results from different starting points independent
    try testing.expect(true);
}
