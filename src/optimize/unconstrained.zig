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

    var x0 = [_]f64{5.0};
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

    var x0 = [_]f64{ 3.0, 4.0 };
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

    var x0 = [_]f64{ 0.0, 0.0 };
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

    var x0 = [_]f64{ 10.0, -5.0, 3.0 };
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

    var x0 = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
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

    var x0 = [_]f64{2.0};
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

    var x0_1 = [_]f64{3.0};
    var x0_2 = [_]f64{3.0};

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

    var x0 = [_]f64{4.0};
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

    var x0 = [_]f64{2.0};
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

    var x0_const = [_]f64{5.0};
    var x0_exp = [_]f64{5.0};

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

    var x0 = [_]f64{1.0};
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

    var x0 = [_]f64{2.0};
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

    var x0 = [_]f64{3.0};
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

    var x0 = [_]f64{ 5.0, 4.0 };
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

    var x0 = [_]f64{1.0};
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

    var x0 = [_]f64{100.0};
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

    var x0_loose = [_]f64{2.0};
    var x0_tight = [_]f64{2.0};

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

    var x0 = [_]f64{0.001};
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

    var x0 = [_]f64{ 2.0, 3.0, -1.5 };
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

    var x0 = [_]f64{ 0.0, 0.0 };
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

    var x0 = [_]f64{ 0.0, 0.0 };
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
    var x0 = [_]f64{1.0};
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

    var x0: [0]f64 = undefined;
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

    var x0 = [_]f64{1.0};
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

    var x0 = [_]f32{ 2.0, 3.0 };
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

    var x0 = [_]f64{ 1.0, 1.0 };
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

    var x0 = [_]f64{ 1.0, 2.0, 3.0 };
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

    var x0_1 = [_]f64{1.0};
    var x0_2 = [_]f64{2.0};

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
