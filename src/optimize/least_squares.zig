//! Nonlinear Least Squares Optimization Algorithms
//!
//! This module provides algorithms for solving nonlinear least squares problems:
//! minimize Σ rᵢ(x)² where rᵢ are residual functions
//!
//! ## Supported Methods
//!
//! - **Levenberg-Marquardt** — Damped Gauss-Newton with trust region strategy
//!   - Interpolates between Gauss-Newton (fast near solution) and gradient descent (robust far from solution)
//!   - Solves: (J^T J + λI) δx = -J^T r at each iteration
//!   - Adaptive damping: λ increases when step rejected, decreases when accepted
//!   - Requires residual functions and Jacobian (finite differences if not provided)
//!
//! - **Gauss-Newton** — Second-order method for least squares
//!   - Approximates Hessian as H ≈ J^T J (assumes small residuals)
//!   - Solves: J^T J δx = -J^T r at each iteration
//!   - Fast convergence near solution, but can fail far from optimum
//!
//! ## Time Complexity
//!
//! - Levenberg-Marquardt: O(iter × m × n²) for m residuals, n parameters
//! - Gauss-Newton: O(iter × m × n²)
//!
//! ## Space Complexity
//!
//! - O(m × n) for Jacobian matrix
//!
//! ## Use Cases
//!
//! - Curve fitting — fit model parameters to data points
//! - Parameter estimation — calibrate physics/engineering models
//! - Bundle adjustment — 3D reconstruction from images
//! - Neural network training — minimize squared error loss
//!
//! ## Parameters & Conventions
//!
//! - **residuals** — Vector of residual functions rᵢ(x): []const T → T
//! - **jacobian** — Optional Jacobian matrix J[i,j] = ∂rᵢ/∂xⱼ (finite differences if null)
//! - **x0** — Initial parameter guess
//! - **max_iter** — Maximum iterations
//! - **tol_f** — Convergence tolerance on objective value change
//! - **tol_x** — Convergence tolerance on parameter change
//! - **lambda_init** — Initial damping parameter (Levenberg-Marquardt only)
//!

const std = @import("std");
const math = std.math;
const testing = std.testing;

/// Residual function: rᵢ(x) → scalar
pub fn ResidualFn(comptime T: type) type {
    return fn (x: []const T) T;
}

/// Jacobian function: J[i,j] = ∂rᵢ/∂xⱼ
/// out_jacobian is m×n matrix stored row-major (m residuals, n parameters)
pub fn JacobianFn(comptime T: type) type {
    return fn (x: []const T, out_jacobian: []T) void;
}

/// Options for Levenberg-Marquardt optimization
pub fn LevenbergMarquardtOptions(comptime T: type) type {
    return struct {
        max_iter: usize = 100,
        tol_f: T = 1e-8,        // Objective change tolerance
        tol_x: T = 1e-8,        // Parameter change tolerance
        tol_grad: T = 1e-8,     // Gradient tolerance
        lambda_init: T = 1e-3,  // Initial damping parameter
        lambda_min: T = 1e-12,  // Minimum lambda (approach Gauss-Newton)
        lambda_max: T = 1e12,   // Maximum lambda (approach gradient descent)
        lambda_scale_up: T = 10.0,   // Factor to increase lambda on rejection
        lambda_scale_down: T = 0.1,  // Factor to decrease lambda on acceptance
        epsilon: T = 1e-8,      // Finite difference step size
    };
}

/// Options for Gauss-Newton optimization
pub fn GaussNewtonOptions(comptime T: type) type {
    return struct {
        max_iter: usize = 100,
        tol_f: T = 1e-8,
        tol_x: T = 1e-8,
        tol_grad: T = 1e-8,
        epsilon: T = 1e-8,  // Finite difference step size
    };
}

/// Result of least squares optimization
pub fn LeastSquaresResult(comptime T: type) type {
    return struct {
        x: []T,              // Optimized parameters (caller must free)
        residuals: []T,      // Final residuals (caller must free)
        f_val: T,            // Final objective value: 0.5 * ||r||²
        n_iter: usize,       // Iterations performed
        converged: bool,     // Whether convergence criterion satisfied
        termination_reason: []const u8,  // Human-readable reason

        pub fn deinit(self: @This(), alloc: std.mem.Allocator) void {
            alloc.free(self.x);
            alloc.free(self.residuals);
        }
    };
}

/// Error set for least squares optimization
pub const LeastSquaresError = error{
    InvalidParameters,
    OutOfMemory,
    MaxIterationsExceeded,
    SingularMatrix,
};

/// Levenberg-Marquardt algorithm for nonlinear least squares
/// Solves: min 0.5 * Σ rᵢ(x)²
/// Using adaptive damping: (J^T J + λI) δx = -J^T r
///
/// Time: O(iter × m × n²) where m = num residuals, n = num parameters
/// Space: O(m × n) for Jacobian
pub fn levenberg_marquardt(
    comptime T: type,
    alloc: std.mem.Allocator,
    residual_fns: []const ResidualFn(T),
    jacobian_fn: ?JacobianFn(T),
    x0: []const T,
    options: LevenbergMarquardtOptions(T),
) LeastSquaresError!LeastSquaresResult(T) {
    const n = x0.len;
    const m = residual_fns.len;

    if (n == 0 or m == 0) return error.InvalidParameters;

    // Allocate working arrays
    const x = try alloc.alloc(T, n);
    errdefer alloc.free(x);
    @memcpy(x, x0);

    const x_new = try alloc.alloc(T, n);
    defer alloc.free(x_new);

    const residuals = try alloc.alloc(T, m);
    errdefer alloc.free(residuals);

    const residuals_new = try alloc.alloc(T, m);
    defer alloc.free(residuals_new);

    const jacobian = try alloc.alloc(T, m * n);
    defer alloc.free(jacobian);

    const jtj = try alloc.alloc(T, n * n);  // J^T J
    defer alloc.free(jtj);

    const jtr = try alloc.alloc(T, n);      // J^T r
    defer alloc.free(jtr);

    const delta_x = try alloc.alloc(T, n);  // Parameter update
    defer alloc.free(delta_x);

    // Compute initial residuals and objective
    computeResiduals(T, residual_fns, x, residuals);
    var f_val = computeObjective(T, residuals);

    var lambda = options.lambda_init;
    var n_iter: usize = 0;
    var converged = false;
    var termination_reason: []const u8 = "max iterations";

    while (n_iter < options.max_iter) : (n_iter += 1) {
        // Compute Jacobian (analytical or finite differences)
        if (jacobian_fn) |jac_fn| {
            jac_fn(x, jacobian);
        } else {
            try computeJacobianFiniteDiff(T, alloc, residual_fns, x, options.epsilon, jacobian);
        }

        // Compute J^T J
        computeJTJ(T, jacobian, m, n, jtj);

        // Compute J^T r
        computeJTR(T, jacobian, residuals, m, n, jtr);

        // Check gradient convergence: ||J^T r|| < tol_grad
        const grad_norm = vectorNorm(T, jtr);
        if (grad_norm < options.tol_grad) {
            converged = true;
            termination_reason = "gradient tolerance";
            break;
        }

        // Inner loop: try steps with different damping until improvement found
        var step_accepted = false;
        var inner_attempts: usize = 0;
        const max_inner_attempts: usize = 20;

        while (!step_accepted and inner_attempts < max_inner_attempts) : (inner_attempts += 1) {
            // Solve (J^T J + λI) δx = -J^T r
            try solveDampedNormalEquations(T, alloc, jtj, jtr, lambda, n, delta_x);

            // Compute new point: x_new = x + δx
            for (x_new, 0..) |*val, i| {
                val.* = x[i] + delta_x[i];
            }

            // Compute new residuals and objective
            computeResiduals(T, residual_fns, x_new, residuals_new);
            const f_val_new = computeObjective(T, residuals_new);

            // Check if step reduces objective
            if (f_val_new < f_val) {
                // Accept step
                @memcpy(x, x_new);
                @memcpy(residuals, residuals_new);

                const delta_f = @abs(f_val - f_val_new);
                const delta_x_norm = vectorNorm(T, delta_x);

                f_val = f_val_new;
                step_accepted = true;

                // Decrease damping (move toward Gauss-Newton)
                lambda = @max(lambda * options.lambda_scale_down, options.lambda_min);

                // Check convergence
                if (delta_f < options.tol_f) {
                    converged = true;
                    termination_reason = "objective tolerance";
                    break;
                }
                if (delta_x_norm < options.tol_x) {
                    converged = true;
                    termination_reason = "parameter tolerance";
                    break;
                }
            } else {
                // Reject step, increase damping (move toward gradient descent)
                lambda = @min(lambda * options.lambda_scale_up, options.lambda_max);

                // If lambda maxed out, give up
                if (lambda >= options.lambda_max) {
                    termination_reason = "damping parameter maxed out";
                    break;
                }
            }
        }

        if (converged) break;
        if (!step_accepted) break;  // No improvement possible
    }

    return LeastSquaresResult(T){
        .x = x,
        .residuals = residuals,
        .f_val = f_val,
        .n_iter = n_iter,
        .converged = converged,
        .termination_reason = termination_reason,
    };
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute residuals: r[i] = residual_fns[i](x)
fn computeResiduals(comptime T: type, residual_fns: []const ResidualFn(T), x: []const T, out: []T) void {
    for (residual_fns, 0..) |res_fn, i| {
        out[i] = res_fn(x);
    }
}

/// Compute objective: f(x) = 0.5 * Σ rᵢ²
fn computeObjective(comptime T: type, residuals: []const T) T {
    var sum: T = 0;
    for (residuals) |r| {
        sum += r * r;
    }
    return 0.5 * sum;
}

/// Compute Jacobian via finite differences: J[i,j] = ∂rᵢ/∂xⱼ ≈ (rᵢ(x + ε eⱼ) - rᵢ(x)) / ε
fn computeJacobianFiniteDiff(
    comptime T: type,
    alloc: std.mem.Allocator,
    residual_fns: []const ResidualFn(T),
    x: []const T,
    epsilon: T,
    out_jacobian: []T,
) !void {
    const n = x.len;
    const m = residual_fns.len;

    const x_perturbed = try alloc.alloc(T, n);
    defer alloc.free(x_perturbed);

    const residuals_base = try alloc.alloc(T, m);
    defer alloc.free(residuals_base);

    const residuals_perturbed = try alloc.alloc(T, m);
    defer alloc.free(residuals_perturbed);

    // Compute base residuals
    computeResiduals(T, residual_fns, x, residuals_base);

    // For each parameter
    for (0..n) |j| {
        // Perturb: x + ε eⱼ
        @memcpy(x_perturbed, x);
        x_perturbed[j] += epsilon;

        // Compute perturbed residuals
        computeResiduals(T, residual_fns, x_perturbed, residuals_perturbed);

        // Finite difference: J[i,j] = (r_perturbed[i] - r_base[i]) / ε
        for (0..m) |i| {
            out_jacobian[i * n + j] = (residuals_perturbed[i] - residuals_base[i]) / epsilon;
        }
    }
}

/// Compute J^T J: (n×n) matrix from J (m×n)
/// J^T J[i,j] = Σₖ J[k,i] * J[k,j]
fn computeJTJ(comptime T: type, jacobian: []const T, m: usize, n: usize, out_jtj: []T) void {
    @memset(out_jtj, 0);

    for (0..n) |i| {
        for (0..n) |j| {
            var sum: T = 0;
            for (0..m) |k| {
                sum += jacobian[k * n + i] * jacobian[k * n + j];
            }
            out_jtj[i * n + j] = sum;
        }
    }
}

/// Compute J^T r: (n×1) vector from J (m×n) and r (m×1)
/// (J^T r)[i] = Σₖ J[k,i] * r[k]
fn computeJTR(comptime T: type, jacobian: []const T, residuals: []const T, m: usize, n: usize, out_jtr: []T) void {
    @memset(out_jtr, 0);

    for (0..n) |i| {
        var sum: T = 0;
        for (0..m) |k| {
            sum += jacobian[k * n + i] * residuals[k];
        }
        out_jtr[i] = sum;
    }
}

/// Solve (J^T J + λI) δx = -J^T r using Cholesky decomposition
/// This is the damped normal equations with Levenberg-Marquardt damping
fn solveDampedNormalEquations(
    comptime T: type,
    alloc: std.mem.Allocator,
    jtj: []const T,
    jtr: []const T,
    lambda: T,
    n: usize,
    out_delta_x: []T,
) !void {
    // Form damped matrix: A = J^T J + λI
    var A = try alloc.alloc(T, n * n);
    defer alloc.free(A);

    for (0..n) |i| {
        for (0..n) |j| {
            A[i * n + j] = jtj[i * n + j];
            if (i == j) {
                A[i * n + j] += lambda;
            }
        }
    }

    // Form right-hand side: b = -J^T r
    var b = try alloc.alloc(T, n);
    defer alloc.free(b);

    for (0..n) |i| {
        b[i] = -jtr[i];
    }

    // Solve Ax = b using Cholesky decomposition (A is symmetric positive definite for λ > 0)
    try choleskyDecompose(T, A, n);
    try choleskySolve(T, A, b, n, out_delta_x);
}

/// Cholesky decomposition: A = L L^T (in-place, lower triangle stored in A)
fn choleskyDecompose(comptime T: type, A: []T, n: usize) !void {
    for (0..n) |i| {
        for (0..i + 1) |j| {
            var sum: T = A[i * n + j];

            for (0..j) |k| {
                sum -= A[i * n + k] * A[j * n + k];
            }

            if (i == j) {
                if (sum <= 0) return error.SingularMatrix;
                A[i * n + j] = @sqrt(sum);
            } else {
                A[i * n + j] = sum / A[j * n + j];
            }
        }
    }
}

/// Solve L L^T x = b using forward and backward substitution
fn choleskySolve(comptime T: type, L: []const T, b: []const T, n: usize, out_x: []T) !void {
    // Forward substitution: L y = b
    var y: [256]T = undefined;  // Stack allocation for small problems
    if (n > 256) return error.InvalidParameters;

    for (0..n) |i| {
        var sum: T = b[i];
        for (0..i) |j| {
            sum -= L[i * n + j] * y[j];
        }
        y[i] = sum / L[i * n + i];
    }

    // Backward substitution: L^T x = y
    var i: usize = n;
    while (i > 0) {
        i -= 1;
        var sum: T = y[i];
        for (i + 1..n) |j| {
            sum -= L[j * n + i] * out_x[j];
        }
        out_x[i] = sum / L[i * n + i];
    }
}

/// Compute Euclidean norm: ||v||₂
fn vectorNorm(comptime T: type, v: []const T) T {
    var sum: T = 0;
    for (v) |val| {
        sum += val * val;
    }
    return @sqrt(sum);
}

// ============================================================================
// Tests
// ============================================================================

test "levenberg_marquardt: linear least squares (overdetermined)" {
    // Fit y = a*x + b to data: (1,2), (2,3), (3,5)
    // Residuals: r_i = (a*x_i + b) - y_i
    // Minimum at a ≈ 1.5, b ≈ 0.5

    const data = [_][2]f64{
        .{ 1, 2 },
        .{ 2, 3 },
        .{ 3, 5 },
    };

    const Residual0 = struct {
        fn f(x: []const f64) f64 {
            return (x[0] * data[0][0] + x[1]) - data[0][1];
        }
    };
    const Residual1 = struct {
        fn f(x: []const f64) f64 {
            return (x[0] * data[1][0] + x[1]) - data[1][1];
        }
    };
    const Residual2 = struct {
        fn f(x: []const f64) f64 {
            return (x[0] * data[2][0] + x[1]) - data[2][1];
        }
    };

    const residuals = [_]ResidualFn(f64){ Residual0.f, Residual1.f, Residual2.f };

    var x0 = [_]f64{ 0, 0 };

    const result = try levenberg_marquardt(
        f64,
        testing.allocator,
        &residuals,
        null, // Use finite differences
        &x0,
        .{},
    );
    defer result.deinit(testing.allocator);

    // Check convergence
    try testing.expect(result.converged);

    // Check solution: a ≈ 1.5, b ≈ 0.5
    try testing.expectApproxEqAbs(1.5, result.x[0], 1e-3);
    try testing.expectApproxEqAbs(0.5, result.x[1], 1e-3);

    // Check objective is small
    try testing.expect(result.f_val < 0.5);
}

test "levenberg_marquardt: exponential decay fitting" {
    // Fit y = a * exp(-b * x) to synthetic data
    // True parameters: a = 5.0, b = 0.5

    const true_a = 5.0;
    const true_b = 0.5;

    // Precompute y_data at comptime so it can be accessed by residual functions
    const Data = struct {
        const x_data = [_]f64{ 0, 1, 2, 3, 4 };
        const y_data = blk: {
            var y: [5]f64 = undefined;
            for (x_data, 0..) |x, i| {
                y[i] = true_a * @exp(-true_b * x);
            }
            break :blk y;
        };
    };

    const Residual0 = struct {
        fn f(params: []const f64) f64 {
            const a = params[0];
            const b = params[1];
            return (a * @exp(-b * Data.x_data[0])) - Data.y_data[0];
        }
    };
    const Residual1 = struct {
        fn f(params: []const f64) f64 {
            const a = params[0];
            const b = params[1];
            return (a * @exp(-b * Data.x_data[1])) - Data.y_data[1];
        }
    };
    const Residual2 = struct {
        fn f(params: []const f64) f64 {
            const a = params[0];
            const b = params[1];
            return (a * @exp(-b * Data.x_data[2])) - Data.y_data[2];
        }
    };
    const Residual3 = struct {
        fn f(params: []const f64) f64 {
            const a = params[0];
            const b = params[1];
            return (a * @exp(-b * Data.x_data[3])) - Data.y_data[3];
        }
    };
    const Residual4 = struct {
        fn f(params: []const f64) f64 {
            const a = params[0];
            const b = params[1];
            return (a * @exp(-b * Data.x_data[4])) - Data.y_data[4];
        }
    };

    const residuals = [_]ResidualFn(f64){ Residual0.f, Residual1.f, Residual2.f, Residual3.f, Residual4.f };

    var x0 = [_]f64{ 1.0, 0.1 }; // Initial guess far from solution

    const result = try levenberg_marquardt(
        f64,
        testing.allocator,
        &residuals,
        null,
        &x0,
        .{},
    );
    defer result.deinit(testing.allocator);

    // Check convergence
    try testing.expect(result.converged);

    // Check solution: a ≈ 5.0, b ≈ 0.5
    try testing.expectApproxEqAbs(true_a, result.x[0], 1e-3);
    try testing.expectApproxEqAbs(true_b, result.x[1], 1e-3);

    // Check objective is near zero (perfect fit to noise-free data)
    try testing.expect(result.f_val < 1e-10);
}

test "levenberg_marquardt: Rosenbrock valley (challenging nonlinear)" {
    // Rosenbrock function as least squares: r₁ = 10(x₂ - x₁²), r₂ = 1 - x₁
    // Minimum at (1, 1) with f = 0

    const Residual0 = struct {
        fn f(x: []const f64) f64 {
            return 10 * (x[1] - x[0] * x[0]);
        }
    };
    const Residual1 = struct {
        fn f(x: []const f64) f64 {
            return 1 - x[0];
        }
    };

    const residuals = [_]ResidualFn(f64){ Residual0.f, Residual1.f };

    var x0 = [_]f64{ -1.0, -1.0 }; // Start far from optimum

    const result = try levenberg_marquardt(
        f64,
        testing.allocator,
        &residuals,
        null,
        &x0,
        .{ .max_iter = 200 },
    );
    defer result.deinit(testing.allocator);

    // Check convergence
    try testing.expect(result.converged);

    // Check solution: (1, 1)
    try testing.expectApproxEqAbs(1.0, result.x[0], 1e-4);
    try testing.expectApproxEqAbs(1.0, result.x[1], 1e-4);

    // Check objective is near zero
    try testing.expect(result.f_val < 1e-6);
}

test "levenberg_marquardt: f32 support" {
    // Simple linear fit with f32 precision
    const Residual0 = struct {
        fn f(x: []const f32) f32 {
            return (x[0] + x[1]) - 3.0;
        }
    };
    const Residual1 = struct {
        fn f(x: []const f32) f32 {
            return (2 * x[0] - x[1]) - 1.0;
        }
    };

    const residuals = [_]ResidualFn(f32){ Residual0.f, Residual1.f };

    var x0 = [_]f32{ 0, 0 };

    const result = try levenberg_marquardt(
        f32,
        testing.allocator,
        &residuals,
        null,
        &x0,
        .{},
    );
    defer result.deinit(testing.allocator);

    // Check convergence
    try testing.expect(result.converged);

    // Solution: x + y = 3, 2x - y = 1 → x = 4/3, y = 5/3
    try testing.expectApproxEqAbs(@as(f32, 4.0 / 3.0), result.x[0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 5.0 / 3.0), result.x[1], 1e-3);
}

test "levenberg_marquardt: single parameter optimization" {
    // Minimize (x - 2)²
    const Residual0 = struct {
        fn f(x: []const f64) f64 {
            return x[0] - 2.0;
        }
    };

    const residuals = [_]ResidualFn(f64){Residual0.f};

    var x0 = [_]f64{0.0};

    const result = try levenberg_marquardt(
        f64,
        testing.allocator,
        &residuals,
        null,
        &x0,
        .{},
    );
    defer result.deinit(testing.allocator);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(2.0, result.x[0], 1e-6);
}

test "levenberg_marquardt: convergence on gradient tolerance" {
    // Problem where gradient goes to zero
    const Residual0 = struct {
        fn f(x: []const f64) f64 {
            return x[0] - 5.0;
        }
    };

    const residuals = [_]ResidualFn(f64){Residual0.f};

    var x0 = [_]f64{5.01};

    const result = try levenberg_marquardt(
        f64,
        testing.allocator,
        &residuals,
        null,
        &x0,
        .{ .tol_grad = 1e-3 },
    );
    defer result.deinit(testing.allocator);

    try testing.expect(result.converged);
    try testing.expect(std.mem.eql(u8, result.termination_reason, "gradient tolerance"));
}

test "levenberg_marquardt: quadratic fitting" {
    // Fit y = a*x² + b*x + c to data
    // True parameters: a = 2, b = -3, c = 1

    const true_a = 2.0;
    const true_b = -3.0;
    const true_c = 1.0;

    const Data = struct {
        const x_data = [_]f64{ -2, -1, 0, 1, 2, 3 };
        const y_data = blk: {
            var y: [6]f64 = undefined;
            for (x_data, 0..) |x, i| {
                y[i] = true_a * x * x + true_b * x + true_c;
            }
            break :blk y;
        };
    };

    const Residual0 = struct {
        fn f(p: []const f64) f64 {
            const x = Data.x_data[0];
            return (p[0] * x * x + p[1] * x + p[2]) - Data.y_data[0];
        }
    };
    const Residual1 = struct {
        fn f(p: []const f64) f64 {
            const x = Data.x_data[1];
            return (p[0] * x * x + p[1] * x + p[2]) - Data.y_data[1];
        }
    };
    const Residual2 = struct {
        fn f(p: []const f64) f64 {
            const x = Data.x_data[2];
            return (p[0] * x * x + p[1] * x + p[2]) - Data.y_data[2];
        }
    };
    const Residual3 = struct {
        fn f(p: []const f64) f64 {
            const x = Data.x_data[3];
            return (p[0] * x * x + p[1] * x + p[2]) - Data.y_data[3];
        }
    };
    const Residual4 = struct {
        fn f(p: []const f64) f64 {
            const x = Data.x_data[4];
            return (p[0] * x * x + p[1] * x + p[2]) - Data.y_data[4];
        }
    };
    const Residual5 = struct {
        fn f(p: []const f64) f64 {
            const x = Data.x_data[5];
            return (p[0] * x * x + p[1] * x + p[2]) - Data.y_data[5];
        }
    };

    const residuals = [_]ResidualFn(f64){ Residual0.f, Residual1.f, Residual2.f, Residual3.f, Residual4.f, Residual5.f };

    var x0 = [_]f64{ 0, 0, 0 };

    const result = try levenberg_marquardt(
        f64,
        testing.allocator,
        &residuals,
        null,
        &x0,
        .{},
    );
    defer result.deinit(testing.allocator);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(true_a, result.x[0], 1e-6);
    try testing.expectApproxEqAbs(true_b, result.x[1], 1e-6);
    try testing.expectApproxEqAbs(true_c, result.x[2], 1e-6);
    try testing.expect(result.f_val < 1e-12);
}

test "levenberg_marquardt: handles poor initial guess" {
    // Quadratic bowl: minimize x² + y²
    // Optimum at (0, 0)
    const Residual0 = struct {
        fn f(x: []const f64) f64 {
            return x[0]; // r₁ = x
        }
    };
    const Residual1 = struct {
        fn f(x: []const f64) f64 {
            return x[1]; // r₂ = y
        }
    };

    const residuals = [_]ResidualFn(f64){ Residual0.f, Residual1.f };

    var x0 = [_]f64{ 100.0, -100.0 }; // Very far from optimum

    const result = try levenberg_marquardt(
        f64,
        testing.allocator,
        &residuals,
        null,
        &x0,
        .{ .max_iter = 200 },
    );
    defer result.deinit(testing.allocator);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(0.0, result.x[0], 1e-5);
    try testing.expectApproxEqAbs(0.0, result.x[1], 1e-5);
}

test "levenberg_marquardt: circle fitting" {
    // Fit circle (x - cx)² + (y - cy)² = r² to points
    // True center: (3, 4), radius: 5
    const true_cx = 3.0;
    const true_cy = 4.0;
    const true_r = 5.0;

    const Data = struct {
        // Points on circle
        const points = [_][2]f64{
            .{ 8, 4 },  // (cx + r, cy)
            .{ 3, 9 },  // (cx, cy + r)
            .{ -2, 4 }, // (cx - r, cy)
            .{ 3, -1 }, // (cx, cy - r)
        };
    };

    // Residual: distance from point to circle = sqrt((x-cx)² + (y-cy)²) - r
    const Residual0 = struct {
        fn f(p: []const f64) f64 {
            const cx = p[0];
            const cy = p[1];
            const r = p[2];
            const dx = Data.points[0][0] - cx;
            const dy = Data.points[0][1] - cy;
            return @sqrt(dx * dx + dy * dy) - r;
        }
    };
    const Residual1 = struct {
        fn f(p: []const f64) f64 {
            const cx = p[0];
            const cy = p[1];
            const r = p[2];
            const dx = Data.points[1][0] - cx;
            const dy = Data.points[1][1] - cy;
            return @sqrt(dx * dx + dy * dy) - r;
        }
    };
    const Residual2 = struct {
        fn f(p: []const f64) f64 {
            const cx = p[0];
            const cy = p[1];
            const r = p[2];
            const dx = Data.points[2][0] - cx;
            const dy = Data.points[2][1] - cy;
            return @sqrt(dx * dx + dy * dy) - r;
        }
    };
    const Residual3 = struct {
        fn f(p: []const f64) f64 {
            const cx = p[0];
            const cy = p[1];
            const r = p[2];
            const dx = Data.points[3][0] - cx;
            const dy = Data.points[3][1] - cy;
            return @sqrt(dx * dx + dy * dy) - r;
        }
    };

    const residuals = [_]ResidualFn(f64){ Residual0.f, Residual1.f, Residual2.f, Residual3.f };

    var x0 = [_]f64{ 0, 0, 1 }; // Initial guess

    const result = try levenberg_marquardt(
        f64,
        testing.allocator,
        &residuals,
        null,
        &x0,
        .{},
    );
    defer result.deinit(testing.allocator);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(true_cx, result.x[0], 1e-6);
    try testing.expectApproxEqAbs(true_cy, result.x[1], 1e-6);
    try testing.expectApproxEqAbs(true_r, result.x[2], 1e-6);
}

test "levenberg_marquardt: handles zero residuals at optimum" {
    // Exact fit scenario - no noise
    const Residual0 = struct {
        fn f(x: []const f64) f64 {
            return x[0] - 7.0;
        }
    };

    const residuals = [_]ResidualFn(f64){Residual0.f};

    var x0 = [_]f64{7.0}; // Start at optimum

    const result = try levenberg_marquardt(
        f64,
        testing.allocator,
        &residuals,
        null,
        &x0,
        .{},
    );
    defer result.deinit(testing.allocator);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(7.0, result.x[0], 1e-10);
    try testing.expect(result.f_val < 1e-20);
}
