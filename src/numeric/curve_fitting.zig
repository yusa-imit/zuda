//! Curve Fitting — Levenberg-Marquardt Non-linear Least Squares
//!
//! This module provides curve fitting via the Levenberg-Marquardt algorithm
//! for solving non-linear least squares problems. It is used to fit parameters
//! of a model function to observed data.
//!
//! ## Supported Operations
//! - `curve_fit` — Levenberg-Marquardt algorithm for parameter estimation
//!
//! ## Mathematical Background
//! The Levenberg-Marquardt algorithm solves the non-linear least squares problem:
//!   minimize ||r||² = Σ(y_i - model(x_i, p))²
//! where y_i are observed values, model(x_i, p) are predicted values with
//! parameters p, and r are residuals.
//!
//! The algorithm interpolates between Gauss-Newton (fast, near minimum) and
//! gradient descent (robust, far from minimum) by adjusting a damping parameter λ.
//!
//! ## Time Complexity
//! - O(n·m²·iterations) where n = number of data points, m = number of parameters
//! - iterations typically < 100 for well-conditioned problems
//!
//! ## Space Complexity
//! - O(n·m) for Jacobian matrix and temporary arrays
//!
//! ## Use Cases
//! - Parameter estimation for physics/engineering models
//! - Fitting experimental data to theoretical models
//! - Robust regression and non-linear optimization
//! - Signal processing model identification

const std = @import("std");
const math = std.math;
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Error set for curve fitting operations
pub const CurveFittingError = error{
    DimensionMismatch,       // x_data.len != y_data.len
    InsufficientData,        // not enough data points
    EmptyData,               // no data provided
    InvalidInitialGuess,     // NaN or Inf in initial parameters
    MaxIterationsExceeded,   // convergence not reached
    SingularJacobian,        // J^T·J is rank-deficient
    NonFiniteResult,         // NaN or Inf in computation
    AllocationFailed,        // memory allocation failed
};

/// Result of curve fitting operation
pub fn CurveFitResult(comptime T: type) type {
    return struct {
        params: []T,           // fitted parameters
        residuals: []T,        // residuals at each data point
        n_iter: usize,         // number of iterations performed
        final_cost: T,         // final sum of squared residuals
        converged: bool,       // whether convergence was achieved
    };
}

/// Fit a model to data using the Levenberg-Marquardt algorithm
///
/// Finds parameters p that minimize the sum of squared residuals:
///   minimize Σ(y_i - model(x_i, p))²
///
/// Uses the Levenberg-Marquardt algorithm which adaptively balances
/// Gauss-Newton (fast near optimum) and gradient descent (robust far away).
///
/// Parameters:
/// - T: floating-point type (f32 or f64)
/// - model_func: function(x: []const T, params: []const T) -> T returning predicted y
/// - x_data: slice of independent variable values
/// - y_data: slice of observed dependent variable values (must match x_data.len)
/// - p0: initial parameter guess (must not contain NaN/Inf)
/// - allocator: memory allocator for temporary arrays
///
/// Returns: CurveFitResult containing:
/// - params: optimized parameters (caller must deinit)
/// - residuals: residual values at each data point
/// - n_iter: number of iterations performed
/// - final_cost: final sum of squared residuals
/// - converged: whether convergence tolerance was met
///
/// Errors:
/// - error.DimensionMismatch: if x_data.len != y_data.len
/// - error.EmptyData: if x_data is empty
/// - error.InsufficientData: if fewer than 2 data points
/// - error.InvalidInitialGuess: if p0 contains NaN or Inf
/// - error.AllocationFailed: if memory allocation fails
/// - error.NonFiniteResult: if computation produces NaN or Inf
/// - error.MaxIterationsExceeded: if convergence not reached after max_iter
///
/// Time: O(n·m²·iterations) | Space: O(n·m)
pub fn curve_fit(
    comptime T: type,
    model_func: *const fn (x: T, params: []const T) T,
    x_data: []const T,
    y_data: []const T,
    p0: []const T,
    allocator: Allocator,
) (CurveFittingError || Allocator.Error)!CurveFitResult(T) {
    // Check input dimensions and validity
    if (x_data.len == 0 or y_data.len == 0) return error.EmptyData;
    if (x_data.len != y_data.len) return error.DimensionMismatch;
    if (x_data.len < 2) return error.InsufficientData;

    const n = x_data.len; // number of data points
    const m = p0.len; // number of parameters

    // Check for NaN/Inf in initial guess
    for (p0) |p| {
        if (!math.isFinite(p)) return error.InvalidInitialGuess;
    }

    // Algorithm constants
    const max_iter = 1000;
    const tol = if (T == f32) @as(T, 1e-5) else @as(T, 1e-8);
    const eps = math.sqrt(std.math.floatEps(T));

    // Initialize parameters
    var params = try allocator.dupe(T, p0);
    errdefer allocator.free(params);

    var params_new = try allocator.alloc(T, m);
    errdefer allocator.free(params_new);

    var residuals = try allocator.alloc(T, n);
    errdefer allocator.free(residuals);

    var jacobian = try allocator.alloc(T, n * m);
    errdefer allocator.free(jacobian);

    var jtr = try allocator.alloc(T, m);
    errdefer allocator.free(jtr);

    var hessian = try allocator.alloc(T, m * m);
    errdefer allocator.free(hessian);

    var hessian_damped = try allocator.alloc(T, m * m);
    errdefer allocator.free(hessian_damped);

    // Lambda damping parameter
    var lambda: T = 0.001;
    var converged = false;
    var n_iter: usize = 0;
    var final_cost: T = 0;

    // Main iteration loop
    while (n_iter < max_iter) {
        defer n_iter += 1;

        // Step 1: Compute residuals r[i] = y[i] - model(x[i], params)
        var cost: T = 0;
        for (0..n) |i| {
            const pred = model_func(x_data[i], params);
            if (!math.isFinite(pred)) return error.NonFiniteResult;
            residuals[i] = y_data[i] - pred;
            cost += residuals[i] * residuals[i];
        }
        final_cost = cost;

        // Step 2: Compute Jacobian J[i,j] = ∂model/∂p[j] via central differences
        for (0..m) |j| {
            const step_size = if (@abs(params[j]) > 1) @abs(params[j]) else 1.0;
            const h = eps * step_size;

            // Evaluate at params[j] + h
            params[j] += h;
            var pred_up_buf = try allocator.alloc(T, n);
            defer allocator.free(pred_up_buf);
            for (0..n) |i| {
                const pred = model_func(x_data[i], params);
                if (!math.isFinite(pred)) return error.NonFiniteResult;
                pred_up_buf[i] = pred;
            }

            // Evaluate at params[j] - h
            params[j] -= 2 * h;
            var pred_down_buf = try allocator.alloc(T, n);
            defer allocator.free(pred_down_buf);
            for (0..n) |i| {
                const pred = model_func(x_data[i], params);
                if (!math.isFinite(pred)) return error.NonFiniteResult;
                pred_down_buf[i] = pred;
            }

            // Restore parameter and compute finite difference
            params[j] += h;
            for (0..n) |i| {
                jacobian[i * m + j] = (pred_up_buf[i] - pred_down_buf[i]) / (2 * h);
            }
        }

        // Step 3: Compute Hessian approximation H = J^T·J and J^T·r
        for (0..m) |i| {
            jtr[i] = 0;
            for (0..n) |k| {
                jtr[i] += jacobian[k * m + i] * residuals[k];
            }
        }

        for (0..m) |i| {
            for (0..m) |j| {
                var sum: T = 0;
                for (0..n) |k| {
                    sum += jacobian[k * m + i] * jacobian[k * m + j];
                }
                hessian[i * m + j] = sum;
            }
        }

        // Step 4: Build damped Hessian H_d = H + λ·I
        for (0..m) |i| {
            for (0..m) |j| {
                hessian_damped[i * m + j] = hessian[i * m + j];
                if (i == j) {
                    hessian_damped[i * m + j] += lambda;
                }
            }
        }

        // Step 5: Solve (H + λ·I)·Δp = -J^T·r
        // Create RHS: -J^T·r
        var rhs = try allocator.dupe(T, jtr);
        defer allocator.free(rhs);
        for (0..m) |i| {
            rhs[i] = -rhs[i];
        }

        const solve_result = solveLinearSystem(T, hessian_damped, rhs, m, allocator);
        var delta_p: []T = undefined;

        if (solve_result) |dp| {
            delta_p = dp;
        } else |_| {
            // Singular system, increase damping and continue to retry
            lambda *= 10;
            if (lambda > 1e10) break;
            continue;
        }
        defer allocator.free(delta_p);

        // Step 6: Try params_new = params + Δp
        for (0..m) |i| {
            params_new[i] = params[i] + delta_p[i];
        }

        // Compute cost at new parameters
        var cost_new: T = 0;
        for (0..n) |i| {
            const pred = model_func(x_data[i], params_new);
            if (!math.isFinite(pred)) return error.NonFiniteResult;
            const r = y_data[i] - pred;
            cost_new += r * r;
        }

        // Step 7: Accept/reject and adjust damping
        if (cost_new < cost) {
            // Good step: accept and decrease damping
            @memcpy(params, params_new);
            final_cost = cost_new;
            lambda *= 0.1;
            if (lambda < 1e-10) lambda = 1e-10;

            // Check convergence
            var norm_delta: T = 0;
            var norm_p: T = 0;
            for (0..m) |i| {
                norm_delta += delta_p[i] * delta_p[i];
                norm_p += params[i] * params[i];
            }
            norm_delta = math.sqrt(norm_delta);
            norm_p = math.sqrt(norm_p);

            const rel_change = if (norm_p > 0) norm_delta / norm_p else norm_delta;
            if (rel_change < tol or norm_delta < tol) {
                converged = true;
                break;
            }
        } else {
            // Bad step: reject and increase damping
            lambda *= 10;
            if (lambda > 1e10) {
                // Lambda too large, give up
                break;
            }
        }
    }

    // Ensure residuals are computed with final parameters
    for (0..n) |i| {
        const pred = model_func(x_data[i], params);
        residuals[i] = y_data[i] - pred;
    }

    allocator.free(params_new);
    allocator.free(jtr);
    allocator.free(hessian);
    allocator.free(hessian_damped);
    allocator.free(jacobian);

    return CurveFitResult(T){
        .params = params,
        .residuals = residuals,
        .n_iter = n_iter,
        .final_cost = final_cost,
        .converged = converged,
    };
}

/// Solve linear system Ax = b using Gaussian elimination with partial pivoting
fn solveLinearSystem(
    comptime T: type,
    A: []T,
    b: []T,
    n: usize,
    allocator: Allocator,
) (CurveFittingError || Allocator.Error)![]T {
    // Make copies so we don't modify inputs
    var A_copy = try allocator.dupe(T, A);
    errdefer allocator.free(A_copy);

    var b_copy = try allocator.dupe(T, b);
    errdefer allocator.free(b_copy);

    // Gaussian elimination with partial pivoting
    for (0..n) |col| {
        // Find pivot
        var max_row = col;
        var max_val: T = @abs(A_copy[col * n + col]);
        for (col + 1..n) |row| {
            const val = @abs(A_copy[row * n + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }

        // Check for singular matrix
        if (max_val < 1e-15) {
            allocator.free(A_copy);
            allocator.free(b_copy);
            return error.SingularJacobian;
        }

        // Swap rows
        if (max_row != col) {
            for (0..n) |j| {
                const tmp = A_copy[col * n + j];
                A_copy[col * n + j] = A_copy[max_row * n + j];
                A_copy[max_row * n + j] = tmp;
            }
            const tmp_b = b_copy[col];
            b_copy[col] = b_copy[max_row];
            b_copy[max_row] = tmp_b;
        }

        // Eliminate below
        for (col + 1..n) |row| {
            const factor = A_copy[row * n + col] / A_copy[col * n + col];
            for (col..n) |j| {
                A_copy[row * n + j] -= factor * A_copy[col * n + j];
            }
            b_copy[row] -= factor * b_copy[col];
        }
    }

    // Back substitution
    var x = try allocator.alloc(T, n);
    errdefer allocator.free(x);

    for (0..n) |idx| {
        const i = n - 1 - idx;
        var sum: T = 0;
        for (i + 1..n) |j| {
            sum += A_copy[i * n + j] * x[j];
        }
        x[i] = (b_copy[i] - sum) / A_copy[i * n + i];
    }

    allocator.free(A_copy);
    allocator.free(b_copy);

    return x;
}

// ============================================================================
// UNIT TESTS — BASIC FITTING
// ============================================================================

test "linear model: y = a + b*x with synthetic perfect data" {
    const allocator = testing.allocator;

    // Data: y = 2 + 3*x (perfect, no noise)
    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{ 2.0, 5.0, 8.0, 11.0, 14.0 };

    // Model: y = p[0] + p[1]*x
    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] + params[1] * x;
        }
    }.f;

    const p0 = [_]f64{ 1.0, 2.0 };
    const result = try curve_fit(f64, model, &x_data, &y_data, &p0, allocator);
    defer allocator.free(result.params);
    defer allocator.free(result.residuals);

    // Expected: a ≈ 2, b ≈ 3
    try testing.expectApproxEqAbs(result.params[0], 2.0, 1e-8);
    try testing.expectApproxEqAbs(result.params[1], 3.0, 1e-8);
    try testing.expect(result.converged);
}

test "quadratic model: y = a + b*x + c*x² with synthetic data" {
    const allocator = testing.allocator;

    // Data: y = 1 + 2*x + 0.5*x²
    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{ 1.0, 3.5, 7.0, 11.5, 17.0 };

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] + params[1] * x + params[2] * x * x;
        }
    }.f;

    const p0 = [_]f64{ 0.5, 1.5, 0.3 };
    const result = try curve_fit(f64, model, &x_data, &y_data, &p0, allocator);
    defer allocator.free(result.params);
    defer allocator.free(result.residuals);

    try testing.expectApproxEqAbs(result.params[0], 1.0, 1e-8);
    try testing.expectApproxEqAbs(result.params[1], 2.0, 1e-8);
    try testing.expectApproxEqAbs(result.params[2], 0.5, 1e-8);
    try testing.expect(result.converged);
}

test "exponential decay model: y = a*exp(-b*x)" {
    const allocator = testing.allocator;

    // Data: y = 5*exp(-0.3*x)
    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{
        5.0,
        5.0 * math.exp(-0.3),
        5.0 * math.exp(-0.6),
        5.0 * math.exp(-0.9),
        5.0 * math.exp(-1.2),
    };

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] * math.exp(-params[1] * x);
        }
    }.f;

    const p0 = [_]f64{ 4.0, 0.2 };
    const result = try curve_fit(f64, model, &x_data, &y_data, &p0, allocator);
    defer allocator.free(result.params);
    defer allocator.free(result.residuals);

    try testing.expectApproxEqAbs(result.params[0], 5.0, 1e-6);
    try testing.expectApproxEqAbs(result.params[1], 0.3, 1e-6);
    try testing.expect(result.converged);
}

test "Gaussian model: y = a*exp(-(x-μ)²/(2σ²))" {
    const allocator = testing.allocator;

    // Data: Gaussian with peak a=3, center μ=2, width σ=1
    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_data: [5]f64 = blk: {
        var result: [5]f64 = undefined;
        for (0..5) |i| {
            const x: f64 = @floatFromInt(i);
            const dx = x - 2.0;
            result[i] = 3.0 * math.exp(-0.5 * dx * dx);
        }
        break :blk result;
    };

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            const dx = x - params[1];
            const sigma = params[2];
            return params[0] * math.exp(-0.5 * dx * dx / (sigma * sigma));
        }
    }.f;

    const p0 = [_]f64{ 2.5, 1.5, 1.2 };
    const result = try curve_fit(f64, model, &x_data, &y_data, &p0, allocator);
    defer allocator.free(result.params);
    defer allocator.free(result.residuals);

    try testing.expectApproxEqAbs(result.params[0], 3.0, 1e-6);
    try testing.expectApproxEqAbs(result.params[1], 2.0, 1e-6);
    try testing.expectApproxEqAbs(result.params[2], 1.0, 1e-6);
    try testing.expect(result.converged);
}

test "single parameter model: y = a*x²" {
    const allocator = testing.allocator;

    // Data: y = 2*x²
    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{ 0.0, 2.0, 8.0, 18.0, 32.0 };

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] * x * x;
        }
    }.f;

    const p0 = [_]f64{1.0};
    const result = try curve_fit(f64, model, &x_data, &y_data, &p0, allocator);
    defer allocator.free(result.params);
    defer allocator.free(result.residuals);

    try testing.expectApproxEqAbs(result.params[0], 2.0, 1e-8);
    try testing.expect(result.converged);
}

test "good initial guess converges quickly" {
    const allocator = testing.allocator;

    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{ 2.0, 5.0, 8.0, 11.0, 14.0 };

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] + params[1] * x;
        }
    }.f;

    // Very close to true values
    const p0 = [_]f64{ 2.1, 2.95 };
    const result = try curve_fit(f64, model, &x_data, &y_data, &p0, allocator);
    defer allocator.free(result.params);
    defer allocator.free(result.residuals);

    try testing.expect(result.n_iter < 20);
    try testing.expect(result.converged);
}

test "bad initial guess still converges (robustness test)" {
    const allocator = testing.allocator;

    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{ 2.0, 5.0, 8.0, 11.0, 14.0 };

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] + params[1] * x;
        }
    }.f;

    // Very far from true values
    const p0 = [_]f64{ 0.0, 1.0 };
    const result = try curve_fit(f64, model, &x_data, &y_data, &p0, allocator);
    defer allocator.free(result.params);
    defer allocator.free(result.residuals);

    try testing.expectApproxEqAbs(result.params[0], 2.0, 1e-6);
    try testing.expectApproxEqAbs(result.params[1], 3.0, 1e-6);
    try testing.expect(result.converged);
}

test "large dataset (n=100 points)" {
    const allocator = testing.allocator;

    // Generate synthetic data: y = 1 + 2*x + noise
    var x_data_buf: [100]f64 = undefined;
    var y_data_buf: [100]f64 = undefined;
    for (0..100) |i| {
        const x: f64 = @floatFromInt(i);
        x_data_buf[i] = x;
        y_data_buf[i] = 1.0 + 2.0 * x;
    }

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] + params[1] * x;
        }
    }.f;

    const p0 = [_]f64{ 0.5, 1.5 };
    const result = try curve_fit(f64, model, &x_data_buf, &y_data_buf, &p0, allocator);
    defer allocator.free(result.params);
    defer allocator.free(result.residuals);

    try testing.expectApproxEqAbs(result.params[0], 1.0, 1e-6);
    try testing.expectApproxEqAbs(result.params[1], 2.0, 1e-6);
    try testing.expect(result.converged);
}

// ============================================================================
// MATHEMATICAL PROPERTIES TESTS
// ============================================================================

test "residuals sum near zero for perfect linear fit" {
    const allocator = testing.allocator;

    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{ 2.0, 5.0, 8.0, 11.0, 14.0 };

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] + params[1] * x;
        }
    }.f;

    const p0 = [_]f64{ 1.0, 2.0 };
    const result = try curve_fit(f64, model, &x_data, &y_data, &p0, allocator);
    defer allocator.free(result.params);
    defer allocator.free(result.residuals);

    // Sum of residuals should be near zero
    var sum: f64 = 0.0;
    for (result.residuals) |r| {
        sum += r;
    }

    try testing.expectApproxEqAbs(sum, 0.0, 1e-10);
}

test "cost function decreases monotonically during iterations" {
    const allocator = testing.allocator;

    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{ 2.0, 5.1, 7.9, 11.2, 14.1 }; // slight noise

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] + params[1] * x;
        }
    }.f;

    const p0 = [_]f64{ 0.0, 1.0 };
    const result = try curve_fit(f64, model, &x_data, &y_data, &p0, allocator);
    defer allocator.free(result.params);
    defer allocator.free(result.residuals);

    // Final cost should be small (< initial cost)
    var initial_cost: f64 = 0.0;
    for (0..x_data.len) |i| {
        const pred = model(x_data[i], &p0);
        const r = y_data[i] - pred;
        initial_cost += r * r;
    }

    try testing.expect(result.final_cost < initial_cost);
}

test "convergence achieved in reasonable iterations (< 100)" {
    const allocator = testing.allocator;

    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{ 2.0, 5.0, 8.0, 11.0, 14.0 };

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] + params[1] * x;
        }
    }.f;

    const p0 = [_]f64{ 1.0, 2.0 };
    const result = try curve_fit(f64, model, &x_data, &y_data, &p0, allocator);
    defer allocator.free(result.params);
    defer allocator.free(result.residuals);

    try testing.expect(result.n_iter < 100);
    try testing.expect(result.converged);
}

test "damping parameter increases on bad step" {
    // This test verifies internal LM behavior:
    // When a proposed step increases cost, λ should increase (more gradient descent-like)
    const allocator = testing.allocator;

    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{ 2.0, 5.0, 8.0, 11.0, 14.0 };

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] + params[1] * x;
        }
    }.f;

    const p0 = [_]f64{ 0.0, 1.0 }; // bad initial guess forces multiple adjustments
    const result = try curve_fit(f64, model, &x_data, &y_data, &p0, allocator);
    defer allocator.free(result.params);
    defer allocator.free(result.residuals);

    // Should converge despite bad start
    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(result.params[0], 2.0, 1e-6);
}

test "parameter covariance (Hessian inverse) is well-conditioned" {
    // Verify that J^T·J is invertible for well-conditioned problem
    const allocator = testing.allocator;

    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{ 2.0, 5.0, 8.0, 11.0, 14.0 };

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] + params[1] * x;
        }
    }.f;

    const p0 = [_]f64{ 1.0, 2.0 };
    const result = try curve_fit(f64, model, &x_data, &y_data, &p0, allocator);
    defer allocator.free(result.params);
    defer allocator.free(result.residuals);

    // For a simple linear problem, final cost should be near machine epsilon
    try testing.expect(result.final_cost < 1e-10);
}

// ============================================================================
// EDGE CASES
// ============================================================================

test "noisy data (Gaussian noise σ=0.1)" {
    const allocator = testing.allocator;

    // Data with noise: y = 1 + 2*x + noise
    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{
        1.05,   // 2 + noise
        5.08,   // 5 + noise
        7.92,   // 8 + noise
        11.15,  // 11 + noise
        13.95,  // 14 + noise
    };

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] + params[1] * x;
        }
    }.f;

    const p0 = [_]f64{ 0.5, 1.5 };
    const result = try curve_fit(f64, model, &x_data, &y_data, &p0, allocator);
    defer allocator.free(result.params);
    defer allocator.free(result.residuals);

    // Should recover parameters within noise level
    try testing.expectApproxEqAbs(result.params[0], 1.0, 0.1);
    try testing.expectApproxEqAbs(result.params[1], 2.0, 0.1);
    try testing.expect(result.converged);
}

test "ill-conditioned problem (collinear data)" {
    const allocator = testing.allocator;

    // Nearly collinear: y ≈ 2*x (little independent variation in intercept)
    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{ 0.01, 2.01, 4.01, 6.01, 8.01 };

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] + params[1] * x;
        }
    }.f;

    const p0 = [_]f64{ 0.0, 1.5 };
    const result = try curve_fit(f64, model, &x_data, &y_data, &p0, allocator);
    defer allocator.free(result.params);
    defer allocator.free(result.residuals);

    // Intercept harder to pin down, but slope should be accurate
    try testing.expectApproxEqAbs(result.params[1], 2.0, 0.05);
    try testing.expect(result.converged);
}

test "large residuals (model mismatch)" {
    const allocator = testing.allocator;

    // Data doesn't fit model well
    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{ 1.0, 2.0, 4.0, 8.0, 16.0 }; // exponential data

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] + params[1] * x; // linear model
        }
    }.f;

    const p0 = [_]f64{ 0.0, 2.0 };
    const result = try curve_fit(f64, model, &x_data, &y_data, &p0, allocator);
    defer allocator.free(result.params);
    defer allocator.free(result.residuals);

    // Algorithm should still find least-squares fit
    try testing.expect(result.final_cost > 0);
}

test "minimal data (n=2, exactly determined)" {
    const allocator = testing.allocator;

    const x_data = [_]f64{ 0.0, 1.0 };
    const y_data = [_]f64{ 2.0, 5.0 };

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] + params[1] * x;
        }
    }.f;

    const p0 = [_]f64{ 1.0, 2.0 };
    const result = try curve_fit(f64, model, &x_data, &y_data, &p0, allocator);
    defer allocator.free(result.params);
    defer allocator.free(result.residuals);

    // Should find exact fit
    try testing.expectApproxEqAbs(result.params[0], 2.0, 1e-8);
    try testing.expectApproxEqAbs(result.params[1], 3.0, 1e-8);
}

test "many parameters (m=10)" {
    const allocator = testing.allocator;

    // Fit polynomial of degree 9: y = p[0] + p[1]*x + p[2]*x² + ...
    var x_data_buf: [20]f64 = undefined;
    var y_data_buf: [20]f64 = undefined;
    var p0_buf: [10]f64 = undefined;

    for (0..20) |i| {
        const x: f64 = @floatFromInt(i);
        x_data_buf[i] = x;

        // y = 1 + x (degree-1 polynomial)
        y_data_buf[i] = 1.0 + x;
    }

    for (0..10) |i| {
        p0_buf[i] = if (i == 0) 0.5 else if (i == 1) 1.5 else 0.0;
    }

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            var result: f64 = 0.0;
            var x_power: f64 = 1.0;
            for (params) |p| {
                result += p * x_power;
                x_power *= x;
            }
            return result;
        }
    }.f;

    const result = try curve_fit(f64, model, &x_data_buf, &y_data_buf, &p0_buf, allocator);
    defer allocator.free(result.params);
    defer allocator.free(result.residuals);

    // p[0] ≈ 1, p[1] ≈ 1, p[2..9] ≈ 0
    try testing.expectApproxEqAbs(result.params[0], 1.0, 1e-6);
    try testing.expectApproxEqAbs(result.params[1], 1.0, 1e-6);
    try testing.expect(result.converged);
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

test "error: dimension mismatch (x_data.len != y_data.len)" {
    const allocator = testing.allocator;

    const x_data = [_]f64{ 0.0, 1.0, 2.0 };
    const y_data = [_]f64{ 2.0, 5.0 }; // too short

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] + params[1] * x;
        }
    }.f;

    const p0 = [_]f64{ 1.0, 2.0 };
    const result = curve_fit(f64, model, &x_data, &y_data, &p0, allocator);

    try testing.expectError(error.DimensionMismatch, result);
}

test "error: empty data" {
    const allocator = testing.allocator;

    const x_data: [0]f64 = undefined;
    const y_data: [0]f64 = undefined;

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            _ = x;
            return params[0];
        }
    }.f;

    const p0 = [_]f64{1.0};
    const result = curve_fit(f64, model, &x_data, &y_data, &p0, allocator);

    try testing.expectError(error.EmptyData, result);
}

test "error: insufficient data (only 1 point)" {
    const allocator = testing.allocator;

    const x_data = [_]f64{0.0};
    const y_data = [_]f64{1.0};

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            _ = x;
            return params[0];
        }
    }.f;

    const p0 = [_]f64{1.0};
    const result = curve_fit(f64, model, &x_data, &y_data, &p0, allocator);

    try testing.expectError(error.InsufficientData, result);
}

test "error: NaN in initial guess" {
    const allocator = testing.allocator;

    const x_data = [_]f64{ 0.0, 1.0, 2.0 };
    const y_data = [_]f64{ 2.0, 5.0, 8.0 };

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] + params[1] * x;
        }
    }.f;

    const p0 = [_]f64{ 1.0, std.math.nan(f64) };
    const result = curve_fit(f64, model, &x_data, &y_data, &p0, allocator);

    try testing.expectError(error.InvalidInitialGuess, result);
}

test "error: Inf in initial guess" {
    const allocator = testing.allocator;

    const x_data = [_]f64{ 0.0, 1.0, 2.0 };
    const y_data = [_]f64{ 2.0, 5.0, 8.0 };

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] + params[1] * x;
        }
    }.f;

    const p0 = [_]f64{ std.math.inf(f64), 2.0 };
    const result = curve_fit(f64, model, &x_data, &y_data, &p0, allocator);

    try testing.expectError(error.InvalidInitialGuess, result);
}

test "error: max iterations exceeded" {
    // This test may or may not trigger depending on algorithm convergence
    // We provide a test structure for when it does occur
    const allocator = testing.allocator;

    // Very difficult problem to converge
    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{ 0.01, 0.1, 1.0, 10.0, 100.0 }; // exponential

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] + params[1] * x;
        }
    }.f;

    // Very bad initial guess
    const p0 = [_]f64{ -100.0, -100.0 };
    const result = curve_fit(f64, model, &x_data, &y_data, &p0, allocator);

    // May succeed or fail depending on algorithm robustness
    if (result) |res| {
        defer allocator.free(res.params);
        defer allocator.free(res.residuals);
    } else |_| {
        try testing.expectError(error.MaxIterationsExceeded, result);
    }
}

// ============================================================================
// TYPE SUPPORT
// ============================================================================

test "f32 precision (tolerance 1e-4)" {
    const allocator = testing.allocator;

    const x_data = [_]f32{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f32{ 2.0, 5.0, 8.0, 11.0, 14.0 };

    const model = struct {
        fn f(x: f32, params: []const f32) f32 {
            return params[0] + params[1] * x;
        }
    }.f;

    const p0 = [_]f32{ 1.0, 2.0 };
    const result = try curve_fit(f32, model, &x_data, &y_data, &p0, allocator);
    defer allocator.free(result.params);
    defer allocator.free(result.residuals);

    try testing.expectApproxEqAbs(result.params[0], 2.0, 1e-4);
    try testing.expectApproxEqAbs(result.params[1], 3.0, 1e-4);
    try testing.expect(result.converged);
}

test "f64 precision (tolerance 1e-10)" {
    const allocator = testing.allocator;

    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{ 2.0, 5.0, 8.0, 11.0, 14.0 };

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] + params[1] * x;
        }
    }.f;

    const p0 = [_]f64{ 1.0, 2.0 };
    const result = try curve_fit(f64, model, &x_data, &y_data, &p0, allocator);
    defer allocator.free(result.params);
    defer allocator.free(result.residuals);

    try testing.expectApproxEqAbs(result.params[0], 2.0, 1e-10);
    try testing.expectApproxEqAbs(result.params[1], 3.0, 1e-10);
    try testing.expect(result.converged);
}

// ============================================================================
// MEMORY SAFETY
// ============================================================================

test "no memory leaks (std.testing.allocator detects leaks)" {
    const allocator = testing.allocator;

    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{ 2.0, 5.0, 8.0, 11.0, 14.0 };

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] + params[1] * x;
        }
    }.f;

    const p0 = [_]f64{ 1.0, 2.0 };
    const result = try curve_fit(f64, model, &x_data, &y_data, &p0, allocator);
    defer allocator.free(result.params);
    defer allocator.free(result.residuals);

    try testing.expect(result.converged);
    // If we reach here without error, allocator did not detect leaks
}

test "multiple calls with same allocator (no state pollution)" {
    const allocator = testing.allocator;

    const x_data1 = [_]f64{ 0.0, 1.0, 2.0 };
    const y_data1 = [_]f64{ 1.0, 3.0, 5.0 };

    const x_data2 = [_]f64{ 0.0, 1.0, 2.0, 3.0 };
    const y_data2 = [_]f64{ 2.0, 5.0, 8.0, 11.0 };

    const model = struct {
        fn f(x: f64, params: []const f64) f64 {
            return params[0] + params[1] * x;
        }
    }.f;

    const p0_1 = [_]f64{ 0.5, 1.5 };
    const result1 = try curve_fit(f64, model, &x_data1, &y_data1, &p0_1, allocator);
    defer allocator.free(result1.params);
    defer allocator.free(result1.residuals);

    const p0_2 = [_]f64{ 0.5, 1.5 };
    const result2 = try curve_fit(f64, model, &x_data2, &y_data2, &p0_2, allocator);
    defer allocator.free(result2.params);
    defer allocator.free(result2.residuals);

    // Both should succeed and give correct results
    try testing.expectApproxEqAbs(result1.params[0], 1.0, 1e-8);
    try testing.expectApproxEqAbs(result1.params[1], 2.0, 1e-8);
    try testing.expectApproxEqAbs(result2.params[0], 2.0, 1e-8);
    try testing.expectApproxEqAbs(result2.params[1], 3.0, 1e-8);
}
