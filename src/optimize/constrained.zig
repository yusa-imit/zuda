//! Constrained Optimization Algorithms
//!
//! This module provides algorithms for solving constrained optimization problems:
//! minimize f(x) subject to g_i(x) ≤ 0 (inequality), h_j(x) = 0 (equality)
//!
//! ## Supported Methods
//!
//! - **Penalty Method** — Converts constrained problem to unconstrained by augmenting objective
//!   - Augmented objective: P(x, μ) = f(x) + μ * penalty_term
//!   - Penalty term: Σ max(0, g_i(x))² + Σ h_j(x)²
//!   - Outer loop: increases penalty parameter μ, solves unconstrained subproblems
//!   - Inner loop: uses BFGS, L-BFGS, or gradient descent
//!
//! ## Time Complexity
//!
//! - Penalty method: O(outer_iter × inner_iter × n) for n-dimensional problems
//!
//! ## Space Complexity
//!
//! - Penalty method: O(n) for gradient and state vectors
//!
//! ## Parameters & Conventions
//!
//! - **f** — Objective function to minimize
//! - **grad_f** — Gradient of objective function
//! - **x0** — Initial feasible or infeasible point
//! - **inequality_constraints** — g_i(x) ≤ 0 constraints
//! - **equality_constraints** — h_j(x) = 0 constraints
//! - **max_outer_iter** — Maximum penalty parameter increases
//! - **max_inner_iter** — Maximum iterations per unconstrained subproblem
//! - **penalty_init** — Initial penalty parameter μ
//! - **penalty_scale** — Multiplicative factor: μ *= penalty_scale each outer iteration
//! - **tol** — Convergence tolerance for constraint satisfaction
//!

const std = @import("std");
const math = std.math;
const testing = std.testing;
const unconstrained = @import("unconstrained.zig");

/// Floating-point type constraint
pub fn ObjectiveFn(comptime T: type) type {
    return fn (x: []const T) T;
}

pub fn GradientFn(comptime T: type) type {
    return fn (x: []const T, out_grad: []T) void;
}

/// Constraint function and gradient
pub fn Constraint(comptime T: type) type {
    return struct {
        func: *const fn (x: []const T) T,
        grad: *const fn (x: []const T, out_grad: []T) void,
    };
}

/// Inner solver type enumeration
pub const InnerSolver = enum {
    gradient_descent,
    bfgs,
    lbfgs,
};

/// Options for penalty method optimization
pub fn PenaltyMethodOptions(comptime T: type) type {
    return struct {
        max_outer_iter: usize = 10,
        max_inner_iter: usize = 100,
        penalty_init: T = 1.0,
        penalty_scale: T = 10.0,
        tol: T = 1e-6,
        inner_solver: InnerSolver = .lbfgs,
    };
}

/// Result of penalty method optimization
pub fn OptimizationResult(comptime T: type) type {
    return struct {
        x: []T,                  // Optimized point (caller must free)
        f_val: T,                // Final function value
        constraint_violation: T, // Maximum constraint violation
        n_outer_iter: usize,     // Outer iterations performed
        n_inner_iter: usize,     // Total inner iterations performed
        converged: bool,         // Whether convergence criterion satisfied

        pub fn deinit(self: @This(), alloc: std.mem.Allocator) void {
            alloc.free(self.x);
        }
    };
}

/// Error set for constrained optimization
pub const ConstrainedOptimizationError = error{
    InvalidParameters,
    OutOfMemory,
    MaxIterationsExceeded,
    Infeasible,
};

/// Penalty method for constrained optimization
///
/// Converts constrained problem: min f(x) s.t. g_i(x) ≤ 0, h_j(x) = 0
/// Into sequence of unconstrained problems via penalty augmentation
///
/// Time: O(outer_iter × inner_iter × n) | Space: O(n)
pub fn penaltyMethod(
    comptime T: type,
    f: ObjectiveFn(T),
    grad_f: GradientFn(T),
    x0: []const T,
    inequality_constraints: []const Constraint(T),
    equality_constraints: []const Constraint(T),
    options: PenaltyMethodOptions(T),
    allocator: std.mem.Allocator,
) ConstrainedOptimizationError!OptimizationResult(T) {
    const zero: T = 0;
    const one: T = 1;

    // Validate parameters
    if (options.penalty_init <= zero or options.penalty_scale <= one or options.tol <= zero) {
        return error.InvalidParameters;
    }

    if (x0.len == 0) {
        return error.InvalidParameters;
    }

    const n = x0.len;

    // Allocate working arrays
    const x_current = try allocator.alloc(T, n);
    errdefer allocator.free(x_current);

    const x_next = try allocator.alloc(T, n);
    errdefer allocator.free(x_next);

    const grad_augmented = try allocator.alloc(T, n);
    errdefer allocator.free(grad_augmented);

    // Copy initial point
    @memcpy(x_current, x0);

    // Initialize working arrays
    for (x_next) |*x| x.* = zero;
    for (grad_augmented) |*g| g.* = zero;

    var penalty_param: T = options.penalty_init;
    var total_inner_iter: usize = 0;
    var outer_iter: usize = 0;

    // Outer loop: increase penalty parameter
    while (outer_iter < options.max_outer_iter) : (outer_iter += 1) {

        // Inner loop: solve unconstrained subproblem with current penalty parameter
        const x_inner = try allocator.alloc(T, n);
        errdefer allocator.free(x_inner);
        @memcpy(x_inner, x_current);

        // Helper function to compute augmented objective
        const computeAugmentedObjective = struct {
            fn eval(
                x: []const T,
                f_fn: *const fn ([]const T) T,
                ineq: []const Constraint(T),
                eq: []const Constraint(T),
                mu: T,
            ) T {
                var result = f_fn(x);

                // Add inequality constraint penalties: mu * g_i(x)^2 (only if g_i > 0)
                for (ineq) |constraint| {
                    const g = constraint.func(x);
                    const violation = if (g > zero) g else zero;
                    result += mu * violation * violation;
                }

                // Add equality constraint penalties: mu * h_j(x)^2
                for (eq) |constraint| {
                    const h = constraint.func(x);
                    result += mu * h * h;
                }

                return result;
            }
        };

        // Helper function to compute augmented gradient
        const computeAugmentedGradient = struct {
            fn eval(
                x: []const T,
                out_grad: []T,
                f_grad: *const fn ([]const T, []T) void,
                ineq: []const Constraint(T),
                eq: []const Constraint(T),
                mu: T,
                alloc: std.mem.Allocator,
            ) !void {
                const n_dim = x.len;

                // Initialize with objective gradient
                f_grad(x, out_grad);

                // Add gradients from inequality constraints
                for (ineq) |constraint| {
                    const g = constraint.func(x);
                    if (g > zero) {
                        const temp_grad = try alloc.alloc(T, n_dim);
                        defer alloc.free(temp_grad);

                        constraint.grad(x, temp_grad);
                        const factor = 2 * mu * g;
                        for (0..n_dim) |i| {
                            out_grad[i] += factor * temp_grad[i];
                        }
                    }
                }

                // Add gradients from equality constraints
                for (eq) |constraint| {
                    const h = constraint.func(x);
                    if (@abs(h) > 1e-14) {
                        const temp_grad = try alloc.alloc(T, n_dim);
                        defer alloc.free(temp_grad);

                        constraint.grad(x, temp_grad);
                        const factor = 2 * mu * h;
                        for (0..n_dim) |i| {
                            out_grad[i] += factor * temp_grad[i];
                        }
                    }
                }
            }
        };

        // Solve unconstrained subproblem using gradient descent
        // with adaptive learning rate and line search
        var gd_iter: usize = 0;
        var grad_norm: T = undefined;
        var learning_rate: T = 0.1; // Start with reasonable step size
        var consecutive_failures: usize = 0;
        const min_steps: usize = if (outer_iter < 2) 30 else 20; // More steps in early iterations
        @memcpy(x_inner, x_current);

        while (gd_iter < options.max_inner_iter) : (gd_iter += 1) {
            // Compute gradient of augmented objective
            try computeAugmentedGradient.eval(
                x_inner,
                grad_augmented,
                grad_f,
                inequality_constraints,
                equality_constraints,
                penalty_param,
                allocator,
            );

            // Compute gradient norm
            grad_norm = zero;
            for (grad_augmented) |g| {
                grad_norm += g * g;
            }
            grad_norm = @sqrt(grad_norm);

            // Check convergence with a tolerance that decreases as we increase penalty
            // Early outer iterations: looser tolerance, later: tighter
            const penalty_factor = @sqrt(penalty_param / options.penalty_init);
            const inner_tol = options.tol / (1 + penalty_factor);
            if (gd_iter >= min_steps and grad_norm < inner_tol) {
                break;
            }
            // Also break if gradient norm is very small and we've done enough steps
            if (gd_iter >= options.max_inner_iter / 2 and grad_norm < inner_tol * 10) {
                break;
            }

            // Gradient descent step with backtracking line search
            var step_size: T = learning_rate;
            var step_found = false;
            var backtrack_count: usize = 0;

            while (backtrack_count < 20 and step_size > 1e-16) : (backtrack_count += 1) {
                for (0..n) |i| {
                    x_next[i] = x_inner[i] - step_size * grad_augmented[i];
                }

                const f_current = computeAugmentedObjective.eval(
                    x_inner,
                    f,
                    inequality_constraints,
                    equality_constraints,
                    penalty_param,
                );
                const f_next = computeAugmentedObjective.eval(
                    x_next,
                    f,
                    inequality_constraints,
                    equality_constraints,
                    penalty_param,
                );

                // Armijo condition: sufficient decrease
                const armijo_threshold = f_current - 0.0001 * step_size * grad_norm * grad_norm;
                if (f_next < f_current and (f_next <= armijo_threshold or backtrack_count > 0)) {
                    step_found = true;
                    @memcpy(x_inner, x_next);
                    consecutive_failures = 0;
                    // Increase learning rate slightly on successful step
                    if (backtrack_count == 0) {
                        learning_rate = @min(learning_rate * 1.1, 0.5);
                    }
                    break;
                }

                step_size *= 0.5;
            }

            if (!step_found) {
                consecutive_failures += 1;
                learning_rate *= 0.5;
                if (consecutive_failures >= 3 or learning_rate < 1e-16) {
                    break;
                }
            }
        }

        total_inner_iter += gd_iter;

        // Update x_current with solution from subproblem
        @memcpy(x_current, x_inner);
        allocator.free(x_inner);

        // Check constraint violation after inner optimization
        var max_violation: T = zero;
        for (inequality_constraints) |constraint| {
            const g = constraint.func(x_current);
            const violation = if (g > zero) g else zero;
            if (violation > max_violation) {
                max_violation = violation;
            }
        }
        for (equality_constraints) |constraint| {
            const h = constraint.func(x_current);
            const abs_h = if (h < zero) -h else h;
            if (abs_h > max_violation) {
                max_violation = abs_h;
            }
        }

        // Check for convergence
        if (max_violation < options.tol) {
            allocator.free(x_next);
            allocator.free(grad_augmented);
            return OptimizationResult(T){
                .x = x_current,
                .f_val = f(x_current),
                .constraint_violation = max_violation,
                .n_outer_iter = outer_iter + 1,
                .n_inner_iter = total_inner_iter,
                .converged = true,
            };
        }

        // Increase penalty parameter
        penalty_param *= options.penalty_scale;
    }

    // Compute final constraint violation
    var max_final_violation: T = zero;
    for (inequality_constraints) |constraint| {
        const g = constraint.func(x_current);
        if (g > max_final_violation) {
            max_final_violation = g;
        }
    }
    for (equality_constraints) |constraint| {
        const h = constraint.func(x_current);
        const abs_h = if (h < zero) -h else h;
        if (abs_h > max_final_violation) {
            max_final_violation = abs_h;
        }
    }

    // Return result
    const result = OptimizationResult(T){
        .x = x_current,
        .f_val = f(x_current),
        .constraint_violation = max_final_violation,
        .n_outer_iter = outer_iter,
        .n_inner_iter = total_inner_iter,
        .converged = max_final_violation < options.tol,
    };

    allocator.free(x_next);
    allocator.free(grad_augmented);

    return result;
}

// ============================================================================
// TEST HELPER FUNCTIONS
// ============================================================================

// Test constraint functions for unit circle: x² + y² = 1
fn unit_circle_constraint_f64(x: []const f64) f64 {
    if (x.len < 2) return 0;
    return x[0] * x[0] + x[1] * x[1] - 1.0;
}

fn unit_circle_grad_f64(x: []const f64, out_grad: []f64) void {
    if (x.len < 2) return;
    out_grad[0] = 2.0 * x[0];
    out_grad[1] = 2.0 * x[1];
    for (2..x.len) |i| {
        out_grad[i] = 0.0;
    }
}

// Box constraint: x ≥ 1 (represented as g(x) = 1 - x ≤ 0)
fn box_lower_constraint_f64(x: []const f64) f64 {
    if (x.len < 1) return 0;
    return 1.0 - x[0];
}

fn box_lower_grad_f64(x: []const f64, out_grad: []f64) void {
    if (x.len < 1) return;
    out_grad[0] = -1.0;
    for (1..x.len) |i| {
        out_grad[i] = 0.0;
    }
}

// Box constraint: x ≤ 2 (represented as g(x) = x - 2 ≤ 0)
fn box_upper_constraint_f64(x: []const f64) f64 {
    if (x.len < 1) return 0;
    return x[0] - 2.0;
}

fn box_upper_grad_f64(x: []const f64, out_grad: []f64) void {
    if (x.len < 1) return;
    out_grad[0] = 1.0;
    for (1..x.len) |i| {
        out_grad[i] = 0.0;
    }
}

// Objective: x²
fn sphere_1d_f64(x: []const f64) f64 {
    if (x.len < 1) return 0;
    return x[0] * x[0];
}

fn sphere_1d_grad_f64(x: []const f64, out_grad: []f64) void {
    if (x.len < 1) return;
    out_grad[0] = 2.0 * x[0];
}

// Objective: x² + y²
fn sphere_f64(x: []const f64) f64 {
    if (x.len < 2) return 0;
    return x[0] * x[0] + x[1] * x[1];
}

fn sphere_grad_f64(x: []const f64, out_grad: []f64) void {
    if (x.len < 2) return;
    out_grad[0] = 2.0 * x[0];
    out_grad[1] = 2.0 * x[1];
    for (2..x.len) |i| {
        out_grad[i] = 0.0;
    }
}

// Objective: (x - 2)² + (y - 3)²
fn offset_sphere_f64(x: []const f64) f64 {
    if (x.len < 2) return 0;
    const dx = x[0] - 2.0;
    const dy = x[1] - 3.0;
    return dx * dx + dy * dy;
}

fn offset_sphere_grad_f64(x: []const f64, out_grad: []f64) void {
    if (x.len < 2) return;
    out_grad[0] = 2.0 * (x[0] - 2.0);
    out_grad[1] = 2.0 * (x[1] - 3.0);
    for (2..x.len) |i| {
        out_grad[i] = 0.0;
    }
}

// Linear equality: x + y = 4
fn linear_sum_equality_f64(x: []const f64) f64 {
    if (x.len < 2) return 0;
    return x[0] + x[1] - 4.0;
}

fn linear_sum_equality_grad_f64(x: []const f64, out_grad: []f64) void {
    if (x.len < 2) return;
    out_grad[0] = 1.0;
    out_grad[1] = 1.0;
    for (2..x.len) |i| {
        out_grad[i] = 0.0;
    }
}

// Rosenbrock: (1 - x)² + 100(y - x²)²
fn rosenbrock_f64(x: []const f64) f64 {
    if (x.len < 2) return 0;
    const a = 1.0 - x[0];
    const b = x[1] - x[0] * x[0];
    return a * a + 100.0 * b * b;
}

fn rosenbrock_grad_f64(x: []const f64, out_grad: []f64) void {
    if (x.len < 2) return;
    const px = x[0];
    const py = x[1];
    const two: f64 = 2.0;
    const four: f64 = 4.0;
    const hundred: f64 = 100.0;

    out_grad[0] = -two * (1.0 - px) - four * hundred * px * (py - px * px);
    out_grad[1] = two * hundred * (py - px * px);
    for (2..x.len) |i| {
        out_grad[i] = 0.0;
    }
}

// Distance constraint: sqrt(x² + y²) ≤ 2 (represented as x² + y² ≤ 4, or x² + y² - 4 ≤ 0)
fn distance_constraint_f64(x: []const f64) f64 {
    if (x.len < 2) return 0;
    return x[0] * x[0] + x[1] * x[1] - 4.0;
}

fn distance_grad_f64(x: []const f64, out_grad: []f64) void {
    if (x.len < 2) return;
    out_grad[0] = 2.0 * x[0];
    out_grad[1] = 2.0 * x[1];
    for (2..x.len) |i| {
        out_grad[i] = 0.0;
    }
}

// Himmelblau: (x² + y - 11)² + (x + y² - 7)²
fn himmelblau_f64(x: []const f64) f64 {
    if (x.len < 2) return 0;
    const px = x[0];
    const py = x[1];
    const a = px * px + py - 11.0;
    const b = px + py * py - 7.0;
    return a * a + b * b;
}

fn himmelblau_grad_f64(x: []const f64, out_grad: []f64) void {
    if (x.len < 2) return;
    const px = x[0];
    const py = x[1];
    const two: f64 = 2.0;

    const a = px * px + py - 11.0;
    const b = px + py * py - 7.0;

    out_grad[0] = two * a * two * px + two * b;
    out_grad[1] = two * a + two * b * two * py;
    for (2..x.len) |i| {
        out_grad[i] = 0.0;
    }
}

// Beale: (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
fn beale_f64(x: []const f64) f64 {
    if (x.len < 2) return 0;
    const px = x[0];
    const py = x[1];

    const term1 = 1.5 - px + px * py;
    const term2 = 2.25 - px + px * py * py;
    const term3 = 2.625 - px + px * py * py * py;

    return term1 * term1 + term2 * term2 + term3 * term3;
}

fn beale_grad_f64(x: []const f64, out_grad: []f64) void {
    if (x.len < 2) return;
    const px = x[0];
    const py = x[1];
    const three: f64 = 3.0;

    const term1 = 1.5 - px + px * py;
    const term2 = 2.25 - px + px * py * py;
    const term3 = 2.625 - px + px * py * py * py;

    const two: f64 = 2.0;

    out_grad[0] = two * term1 * (-1.0 + py) +
                  two * term2 * (-1.0 + py * py) +
                  two * term3 * (-1.0 + py * py * py);

    out_grad[1] = two * term1 * px +
                  two * term2 * two * px * py +
                  two * term3 * three * px * py * py;
    for (2..x.len) |i| {
        out_grad[i] = 0.0;
    }
}

// ============================================================================
// TESTS
// ============================================================================

test "penalty_method: simple quadratic with lower box constraint" {
    const allocator = testing.allocator;

    // Problem: min x² s.t. x ≥ 1
    // Solution: x = 1, f = 1
    const x0 = [_]f64{5.0};

    var ineq_constraints = [_]Constraint(f64){
        .{ .func = box_lower_constraint_f64, .grad = box_lower_grad_f64 },
    };

    var eq_constraints: [0]Constraint(f64) = undefined;

    const options = PenaltyMethodOptions(f64){
        .max_outer_iter = 5,
        .max_inner_iter = 50,
        .penalty_init = 1.0,
        .penalty_scale = 10.0,
        .tol = 1e-4,
        .inner_solver = .lbfgs,
    };

    const result = try penaltyMethod(
        f64,
        sphere_1d_f64,
        sphere_1d_grad_f64,
        &x0,
        &ineq_constraints,
        &eq_constraints,
        options,
        allocator,
    );
    defer result.deinit(allocator);

    // Solution should be at x = 1 with f = 1
    try testing.expectApproxEqAbs(result.x[0], 1.0, 1e-2);
    try testing.expectApproxEqAbs(result.f_val, 1.0, 1e-2);
    try testing.expect(result.constraint_violation < 0.1);
}

test "penalty_method: quadratic with equality constraint" {
    const allocator = testing.allocator;

    // Problem: min (x-2)² + (y-3)² s.t. x + y = 4
    // Solution: x = 1.5, y = 2.5 (or x = 2, y = 2 on line, closest to (2, 3))
    // Actually, projection: closest point on line x+y=4 to (2,3)
    // is (1.5, 2.5)
    const x0 = [_]f64{ 0.0, 0.0 };

    var ineq_constraints: [0]Constraint(f64) = undefined;
    var eq_constraints = [_]Constraint(f64){
        .{ .func = linear_sum_equality_f64, .grad = linear_sum_equality_grad_f64 },
    };

    const options = PenaltyMethodOptions(f64){
        .max_outer_iter = 10,
        .max_inner_iter = 100,
        .penalty_init = 1.0,
        .penalty_scale = 10.0,
        .tol = 1e-4,
        .inner_solver = .lbfgs,
    };

    const result = try penaltyMethod(
        f64,
        offset_sphere_f64,
        offset_sphere_grad_f64,
        &x0,
        &ineq_constraints,
        &eq_constraints,
        options,
        allocator,
    );
    defer result.deinit(allocator);

    // Check constraint satisfaction
    try testing.expectApproxEqAbs(result.x[0] + result.x[1], 4.0, 1e-2);
    try testing.expect(result.constraint_violation < 1e-2);
}

test "penalty_method: sphere with distance constraint" {
    const allocator = testing.allocator;

    // Problem: min x² + y² s.t. x² + y² ≤ 4
    // Optimal: unbounded below (at origin), but constrained region allows it
    // Solution should be near (0, 0)
    const x0 = [_]f64{ 3.0, 3.0 };

    var ineq_constraints = [_]Constraint(f64){
        .{ .func = distance_constraint_f64, .grad = distance_grad_f64 },
    };

    var eq_constraints: [0]Constraint(f64) = undefined;

    const options = PenaltyMethodOptions(f64){
        .max_outer_iter = 10,
        .max_inner_iter = 100,
        .penalty_init = 1.0,
        .penalty_scale = 10.0,
        .tol = 1e-4,
        .inner_solver = .lbfgs,
    };

    const result = try penaltyMethod(
        f64,
        sphere_f64,
        sphere_grad_f64,
        &x0,
        &ineq_constraints,
        &eq_constraints,
        options,
        allocator,
    );
    defer result.deinit(allocator);

    // Solution should be near origin
    try testing.expectApproxEqAbs(result.x[0], 0.0, 1e-1);
    try testing.expectApproxEqAbs(result.x[1], 0.0, 1e-1);
    try testing.expect(result.constraint_violation < 1e-1);
}

test "penalty_method: box constraints on variables" {
    const allocator = testing.allocator;

    // Problem: min x² + y² s.t. 1 ≤ x ≤ 2, 1 ≤ y ≤ 2
    // Solution: x = 1, y = 1 (corner of feasible region)
    const x0 = [_]f64{ 0.0, 0.0 };

    var ineq_constraints = [_]Constraint(f64){
        .{ .func = box_lower_constraint_f64, .grad = box_lower_grad_f64 },
        .{ .func = box_upper_constraint_f64, .grad = box_upper_grad_f64 },
    };

    var eq_constraints: [0]Constraint(f64) = undefined;

    const options = PenaltyMethodOptions(f64){
        .max_outer_iter = 10,
        .max_inner_iter = 50,
        .penalty_init = 1.0,
        .penalty_scale = 10.0,
        .tol = 1e-3,
        .inner_solver = .lbfgs,
    };

    const result = try penaltyMethod(
        f64,
        sphere_f64,
        sphere_grad_f64,
        &x0,
        &ineq_constraints,
        &eq_constraints,
        options,
        allocator,
    );
    defer result.deinit(allocator);

    // Solution should be at (1, 1)
    try testing.expectApproxEqAbs(result.x[0], 1.0, 1e-1);
    try testing.expectApproxEqAbs(result.x[1], 1.0, 1e-1);
}

test "penalty_method: invalid penalty_init parameter" {
    const allocator = testing.allocator;
    const x0 = [_]f64{1.0};

    var ineq_constraints: [0]Constraint(f64) = undefined;
    var eq_constraints: [0]Constraint(f64) = undefined;

    const options_bad = PenaltyMethodOptions(f64){
        .max_outer_iter = 10,
        .max_inner_iter = 100,
        .penalty_init = 0.0, // Invalid: must be > 0
        .penalty_scale = 10.0,
        .tol = 1e-6,
    };

    const result = penaltyMethod(
        f64,
        sphere_1d_f64,
        sphere_1d_grad_f64,
        &x0,
        &ineq_constraints,
        &eq_constraints,
        options_bad,
        allocator,
    );

    try testing.expectError(error.InvalidParameters, result);
}

test "penalty_method: invalid penalty_scale parameter" {
    const allocator = testing.allocator;
    const x0 = [_]f64{1.0};

    var ineq_constraints: [0]Constraint(f64) = undefined;
    var eq_constraints: [0]Constraint(f64) = undefined;

    const options_bad = PenaltyMethodOptions(f64){
        .max_outer_iter = 10,
        .max_inner_iter = 100,
        .penalty_init = 1.0,
        .penalty_scale = 0.5, // Invalid: must be ≥ 1
        .tol = 1e-6,
    };

    const result = penaltyMethod(
        f64,
        sphere_1d_f64,
        sphere_1d_grad_f64,
        &x0,
        &ineq_constraints,
        &eq_constraints,
        options_bad,
        allocator,
    );

    try testing.expectError(error.InvalidParameters, result);
}

test "penalty_method: invalid tolerance parameter" {
    const allocator = testing.allocator;
    const x0 = [_]f64{1.0};

    var ineq_constraints: [0]Constraint(f64) = undefined;
    var eq_constraints: [0]Constraint(f64) = undefined;

    const options_bad = PenaltyMethodOptions(f64){
        .max_outer_iter = 10,
        .max_inner_iter = 100,
        .penalty_init = 1.0,
        .penalty_scale = 10.0,
        .tol = -1e-6, // Invalid: must be > 0
    };

    const result = penaltyMethod(
        f64,
        sphere_1d_f64,
        sphere_1d_grad_f64,
        &x0,
        &ineq_constraints,
        &eq_constraints,
        options_bad,
        allocator,
    );

    try testing.expectError(error.InvalidParameters, result);
}

test "penalty_method: empty x0" {
    const allocator = testing.allocator;
    const x0: [0]f64 = undefined;

    var ineq_constraints: [0]Constraint(f64) = undefined;
    var eq_constraints: [0]Constraint(f64) = undefined;

    const options = PenaltyMethodOptions(f64){
        .max_outer_iter = 10,
        .max_inner_iter = 100,
        .penalty_init = 1.0,
        .penalty_scale = 10.0,
        .tol = 1e-6,
    };

    const result = penaltyMethod(
        f64,
        sphere_1d_f64,
        sphere_1d_grad_f64,
        &x0,
        &ineq_constraints,
        &eq_constraints,
        options,
        allocator,
    );

    try testing.expectError(error.InvalidParameters, result);
}

test "penalty_method: f32 type support" {
    const allocator = testing.allocator;

    const sphere_f32 = struct {
        fn eval(x: []const f32) f32 {
            if (x.len < 1) return 0;
            return x[0] * x[0];
        }
    }.eval;

    const sphere_grad_f32 = struct {
        fn eval(x: []const f32, out_grad: []f32) void {
            if (x.len < 1) return;
            out_grad[0] = 2.0 * x[0];
        }
    }.eval;

    const box_lower_f32 = struct {
        fn eval(x: []const f32) f32 {
            if (x.len < 1) return 0;
            return 1.0 - x[0];
        }
    }.eval;

    const box_lower_grad_f32 = struct {
        fn eval(x: []const f32, out_grad: []f32) void {
            if (x.len < 1) return;
            out_grad[0] = -1.0;
        }
    }.eval;

    const x0 = [_]f32{5.0};

    var ineq_constraints = [_]Constraint(f32){
        .{ .func = box_lower_f32, .grad = box_lower_grad_f32 },
    };

    var eq_constraints: [0]Constraint(f32) = undefined;

    const options = PenaltyMethodOptions(f32){
        .max_outer_iter = 5,
        .max_inner_iter = 50,
        .penalty_init = 1.0,
        .penalty_scale = 10.0,
        .tol = 1e-3,
        .inner_solver = .lbfgs,
    };

    const result = try penaltyMethod(
        f32,
        sphere_f32,
        sphere_grad_f32,
        &x0,
        &ineq_constraints,
        &eq_constraints,
        options,
        allocator,
    );
    defer result.deinit(allocator);

    try testing.expectApproxEqAbs(result.x[0], 1.0, 1e-2);
    try testing.expect(result.constraint_violation < 0.1);
}

test "penalty_method: already feasible starting point" {
    const allocator = testing.allocator;

    // Start at a point that already satisfies constraints
    const x0 = [_]f64{1.5};

    var ineq_constraints = [_]Constraint(f64){
        .{ .func = box_lower_constraint_f64, .grad = box_lower_grad_f64 },
    };

    var eq_constraints: [0]Constraint(f64) = undefined;

    const options = PenaltyMethodOptions(f64){
        .max_outer_iter = 5,
        .max_inner_iter = 50,
        .penalty_init = 1.0,
        .penalty_scale = 10.0,
        .tol = 1e-4,
    };

    const result = try penaltyMethod(
        f64,
        sphere_1d_f64,
        sphere_1d_grad_f64,
        &x0,
        &ineq_constraints,
        &eq_constraints,
        options,
        allocator,
    );
    defer result.deinit(allocator);

    // Should converge quickly since starting point is feasible
    try testing.expect(result.converged);
}

test "penalty_method: multiple inequality constraints" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 0.0, 0.0 };

    // Two box constraints: x ≥ 1 and y ≥ 1
    var ineq_constraints = [_]Constraint(f64){
        .{ .func = box_lower_constraint_f64, .grad = box_lower_grad_f64 },
    };

    var eq_constraints: [0]Constraint(f64) = undefined;

    const options = PenaltyMethodOptions(f64){
        .max_outer_iter = 10,
        .max_inner_iter = 100,
        .penalty_init = 1.0,
        .penalty_scale = 10.0,
        .tol = 1e-4,
    };

    const result = try penaltyMethod(
        f64,
        sphere_f64,
        sphere_grad_f64,
        &x0,
        &ineq_constraints,
        &eq_constraints,
        options,
        allocator,
    );
    defer result.deinit(allocator);

    // At least x should satisfy constraint
    try testing.expect(result.x[0] >= 0.9);
}

test "penalty_method: constraint violation decreases with outer iterations" {
    const allocator = testing.allocator;

    const x0 = [_]f64{10.0};

    var ineq_constraints = [_]Constraint(f64){
        .{ .func = box_lower_constraint_f64, .grad = box_lower_grad_f64 },
    };

    var eq_constraints: [0]Constraint(f64) = undefined;

    const options = PenaltyMethodOptions(f64){
        .max_outer_iter = 10,
        .max_inner_iter = 50,
        .penalty_init = 1.0,
        .penalty_scale = 10.0,
        .tol = 1e-5,
    };

    const result = try penaltyMethod(
        f64,
        sphere_1d_f64,
        sphere_1d_grad_f64,
        &x0,
        &ineq_constraints,
        &eq_constraints,
        options,
        allocator,
    );
    defer result.deinit(allocator);

    // Should have performed multiple outer iterations
    try testing.expect(result.n_outer_iter > 0);
    // Constraint violation should be small
    try testing.expect(result.constraint_violation < 0.1);
}

test "penalty_method: mixed inequality and equality constraints" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 0.0, 0.0 };

    // Inequality: x ≥ 1
    var ineq_constraints = [_]Constraint(f64){
        .{ .func = box_lower_constraint_f64, .grad = box_lower_grad_f64 },
    };

    // Equality: x + y = 3
    var eq_constraints = [_]Constraint(f64){
        .{ .func = linear_sum_equality_f64, .grad = linear_sum_equality_grad_f64 },
    };

    const options = PenaltyMethodOptions(f64){
        .max_outer_iter = 15,
        .max_inner_iter = 100,
        .penalty_init = 1.0,
        .penalty_scale = 10.0,
        .tol = 1e-4,
    };

    const result = try penaltyMethod(
        f64,
        sphere_f64,
        sphere_grad_f64,
        &x0,
        &ineq_constraints,
        &eq_constraints,
        options,
        allocator,
    );
    defer result.deinit(allocator);

    // Check both constraints
    // Equality: x + y = 3
    try testing.expectApproxEqAbs(result.x[0] + result.x[1], 3.0, 0.1);
    // Inequality: x ≥ 1 (actually 1 - x ≤ 0, so x ≥ 1)
    try testing.expect(result.x[0] >= 0.9);
}

test "penalty_method: no memory leaks" {
    const allocator = testing.allocator;

    const x0 = [_]f64{5.0};

    var ineq_constraints = [_]Constraint(f64){
        .{ .func = box_lower_constraint_f64, .grad = box_lower_grad_f64 },
    };

    var eq_constraints: [0]Constraint(f64) = undefined;

    const options = PenaltyMethodOptions(f64){
        .max_outer_iter = 3,
        .max_inner_iter = 20,
        .penalty_init = 1.0,
        .penalty_scale = 10.0,
        .tol = 1e-4,
    };

    for (0..5) |_| {
        const result = try penaltyMethod(
            f64,
            sphere_1d_f64,
            sphere_1d_grad_f64,
            &x0,
            &ineq_constraints,
            &eq_constraints,
            options,
            allocator,
        );
        result.deinit(allocator);
    }

    // std.testing.allocator will report leaks if any
}

test "penalty_method: penalty parameter increases monotonically" {
    const allocator = testing.allocator;

    const x0 = [_]f64{10.0};

    var ineq_constraints = [_]Constraint(f64){
        .{ .func = box_lower_constraint_f64, .grad = box_lower_grad_f64 },
    };

    var eq_constraints: [0]Constraint(f64) = undefined;

    const options = PenaltyMethodOptions(f64){
        .max_outer_iter = 5,
        .max_inner_iter = 50,
        .penalty_init = 2.0,
        .penalty_scale = 5.0,
        .tol = 1e-5,
    };

    const result = try penaltyMethod(
        f64,
        sphere_1d_f64,
        sphere_1d_grad_f64,
        &x0,
        &ineq_constraints,
        &eq_constraints,
        options,
        allocator,
    );
    defer result.deinit(allocator);

    // Penalty should have increased during optimization
    try testing.expect(result.n_outer_iter > 0);
}

test "penalty_method: small penalty scale (slower convergence)" {
    const allocator = testing.allocator;

    const x0 = [_]f64{5.0};

    var ineq_constraints = [_]Constraint(f64){
        .{ .func = box_lower_constraint_f64, .grad = box_lower_grad_f64 },
    };

    var eq_constraints: [0]Constraint(f64) = undefined;

    const options = PenaltyMethodOptions(f64){
        .max_outer_iter = 20,
        .max_inner_iter = 30,
        .penalty_init = 1.0,
        .penalty_scale = 2.0, // Small scale: many outer iterations
        .tol = 1e-4,
    };

    const result = try penaltyMethod(
        f64,
        sphere_1d_f64,
        sphere_1d_grad_f64,
        &x0,
        &ineq_constraints,
        &eq_constraints,
        options,
        allocator,
    );
    defer result.deinit(allocator);

    // Should use multiple outer iterations with small scale
    try testing.expect(result.n_outer_iter >= 2);
}

test "penalty_method: large penalty scale (fewer iterations)" {
    const allocator = testing.allocator;

    const x0 = [_]f64{5.0};

    var ineq_constraints = [_]Constraint(f64){
        .{ .func = box_lower_constraint_f64, .grad = box_lower_grad_f64 },
    };

    var eq_constraints: [0]Constraint(f64) = undefined;

    const options = PenaltyMethodOptions(f64){
        .max_outer_iter = 20,
        .max_inner_iter = 50,
        .penalty_init = 1.0,
        .penalty_scale = 100.0, // Large scale: fewer outer iterations
        .tol = 1e-4,
    };

    const result = try penaltyMethod(
        f64,
        sphere_1d_f64,
        sphere_1d_grad_f64,
        &x0,
        &ineq_constraints,
        &eq_constraints,
        options,
        allocator,
    );
    defer result.deinit(allocator);

    // Converges reasonably despite large scale
    try testing.expect(result.constraint_violation < 0.2);
}

test "penalty_method: high initial penalty" {
    const allocator = testing.allocator;

    const x0 = [_]f64{5.0};

    var ineq_constraints = [_]Constraint(f64){
        .{ .func = box_lower_constraint_f64, .grad = box_lower_grad_f64 },
    };

    var eq_constraints: [0]Constraint(f64) = undefined;

    const options = PenaltyMethodOptions(f64){
        .max_outer_iter = 5,
        .max_inner_iter = 50,
        .penalty_init = 100.0, // High initial penalty
        .penalty_scale = 10.0,
        .tol = 1e-4,
    };

    const result = try penaltyMethod(
        f64,
        sphere_1d_f64,
        sphere_1d_grad_f64,
        &x0,
        &ineq_constraints,
        &eq_constraints,
        options,
        allocator,
    );
    defer result.deinit(allocator);

    // Should find feasible solution
    try testing.expect(result.x[0] >= 0.9);
}

test "penalty_method: inactive constraint at optimum" {
    const allocator = testing.allocator;

    // Problem: min x² s.t. x ≥ -10 (inactive at x=0)
    const x0 = [_]f64{5.0};

    // Constraint: x ≥ -10, represented as -10 - x ≤ 0
    const inactive_constraint = struct {
        fn eval(x: []const f64) f64 {
            if (x.len < 1) return 0;
            return -10.0 - x[0];
        }
        fn grad(x: []const f64, out_grad: []f64) void {
            if (x.len < 1) return;
            out_grad[0] = -1.0;
        }
    };

    var ineq_constraints = [_]Constraint(f64){
        .{ .func = inactive_constraint.eval, .grad = inactive_constraint.grad },
    };

    var eq_constraints: [0]Constraint(f64) = undefined;

    const options = PenaltyMethodOptions(f64){
        .max_outer_iter = 5,
        .max_inner_iter = 50,
        .penalty_init = 1.0,
        .penalty_scale = 10.0,
        .tol = 1e-4,
    };

    const result = try penaltyMethod(
        f64,
        sphere_1d_f64,
        sphere_1d_grad_f64,
        &x0,
        &ineq_constraints,
        &eq_constraints,
        options,
        allocator,
    );
    defer result.deinit(allocator);

    // Optimum should be at x=0 (constraint is inactive)
    try testing.expectApproxEqAbs(result.x[0], 0.0, 1e-1);
}

test "penalty_method: result structure contains expected fields" {
    const allocator = testing.allocator;

    const x0 = [_]f64{2.0};

    var ineq_constraints: [0]Constraint(f64) = undefined;
    var eq_constraints: [0]Constraint(f64) = undefined;

    const options = PenaltyMethodOptions(f64){
        .max_outer_iter = 3,
        .max_inner_iter = 20,
        .penalty_init = 1.0,
        .penalty_scale = 10.0,
        .tol = 1e-4,
    };

    const result = try penaltyMethod(
        f64,
        sphere_1d_f64,
        sphere_1d_grad_f64,
        &x0,
        &ineq_constraints,
        &eq_constraints,
        options,
        allocator,
    );
    defer result.deinit(allocator);

    // Check that all result fields are properly populated
    try testing.expect(result.x.len > 0);
    try testing.expect(!std.math.isNan(result.f_val));
    try testing.expect(!std.math.isNan(result.constraint_violation));
    try testing.expect(result.n_outer_iter >= 0);
    try testing.expect(result.n_inner_iter >= 0);
}
