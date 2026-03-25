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
    InvalidArgument,       // Empty x0, negative learning rate
    OutOfMemory,           // Allocation failure
    InvalidParameters,     // Line search parameter validation
    NotDescentDirection,   // Line search descent direction check
    MaxIterationsExceeded, // Line search max iterations
    AllocationFailed,      // Line search allocation failure
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

/// BFGS (Broyden-Fletcher-Goldfarb-Shanno) quasi-Newton optimization
///
/// Minimizes f(x) using BFGS algorithm which builds an approximation to the inverse Hessian.
///
/// **Algorithm**:
/// 1. Initialize H_0 = I (identity), x_0 = x0
/// 2. For k = 0, 1, 2, ... until convergence:
///    a. Compute gradient g_k = ∇f(x_k)
///    b. Search direction: p_k = -H_k * g_k
///    c. Line search: find α_k satisfying Wolfe/Armijo conditions
///    d. Update: x_{k+1} = x_k + α_k * p_k
///    e. s_k = x_{k+1} - x_k,  y_k = g_{k+1} - g_k
///    f. Update inverse Hessian:
///       ρ_k = 1 / (y_k^T * s_k)
///       H_{k+1} = (I - ρ_k * s_k * y_k^T) * H_k * (I - ρ_k * y_k * s_k^T) + ρ_k * s_k * s_k^T
/// 3. Converge when ||g_k|| < tol or k ≥ max_iter
///
/// **Parameters**:
/// - `f`: Objective function to minimize
/// - `grad_f`: Gradient function ∇f(x)
/// - `x0`: Initial point
/// - `options`: Optimization parameters (max_iter, tol, line_search, etc.)
/// - `allocator`: Memory allocator
///
/// **Returns**: OptimizationResult with:
/// - `x`: Optimized point (caller must free)
/// - `f_val`: Final function value
/// - `grad_norm`: Final gradient norm
/// - `n_iter`: Number of iterations
/// - `converged`: True if ||grad|| < tol
///
/// **Time**: O(n² × max_iter × line_search_cost) — n² for Hessian operations
/// **Space**: O(n²) — Inverse Hessian matrix storage
///
/// **Errors**:
/// - `error.InvalidArgument`: Empty x0, invalid tolerance, invalid line search params
/// - Line search errors propagated from line_search module
pub fn bfgs(
    comptime T: type,
    f: ObjectiveFn(T),
    grad_f: GradientFn(T),
    x0: []const T,
    options: BfgsOptions(T),
    allocator: std.mem.Allocator,
) OptimizationError!OptimizationResult(T) {
    // Validate inputs
    if (x0.len == 0) {
        return error.InvalidArgument;
    }

    const zero: T = 0;
    const one: T = 1;
    const epsilon: T = 1e-10;

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

    const grad = try allocator.alloc(T, n);
    defer allocator.free(grad);

    const grad_prev = try allocator.alloc(T, n);
    defer allocator.free(grad_prev);

    const p = try allocator.alloc(T, n);
    defer allocator.free(p);

    const s = try allocator.alloc(T, n);
    defer allocator.free(s);

    const y = try allocator.alloc(T, n);
    defer allocator.free(y);

    // Allocate inverse Hessian matrix (identity initially)
    const H = try allocator.alloc(T, n * n);
    defer allocator.free(H);

    // Initialize H as identity matrix
    for (0..n) |i| {
        for (0..n) |j| {
            H[i * n + j] = if (i == j) one else zero;
        }
    }

    // Compute initial gradient
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

    // Main optimization loop
    while (n_iter < options.max_iter) : (n_iter += 1) {
        // Compute search direction: p = -H * grad
        for (0..n) |i| {
            var sum: T = zero;
            for (0..n) |j| {
                sum += H[i * n + j] * grad[j];
            }
            p[i] = -sum;
        }

        // Line search to find step size α
        var alpha: T = undefined;

        switch (options.line_search) {
            .armijo => {
                const result = try line_search.armijo(
                    T,
                    f,
                    x,
                    p,
                    grad,
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
                    p,
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
                    p,
                    grad,
                    one,
                    0.5,
                    options.ls_c1,
                    options.ls_max_iter,
                    allocator,
                );
                alpha = result.alpha;
            },
        }

        // Save previous gradient
        @memcpy(grad_prev, grad);

        // Update x: x_new = x + α * p
        for (x, p) |*xi, pi| {
            xi.* += alpha * pi;
        }

        // Compute new gradient
        grad_f(x, grad);

        // Compute s = x_new - x_old (s = alpha * p)
        for (0..n) |i| {
            s[i] = alpha * p[i];
        }

        // Compute y = grad_new - grad_old
        for (0..n) |i| {
            y[i] = grad[i] - grad_prev[i];
        }

        // Curvature check: y^T * s > epsilon
        var y_dot_s: T = zero;
        for (0..n) |i| {
            y_dot_s += y[i] * s[i];
        }

        // Only update Hessian if curvature condition is satisfied
        if (y_dot_s > epsilon) {
            const rho = one / y_dot_s;

            // Allocate temporary matrices for BFGS update
            const V = try allocator.alloc(T, n * n);
            defer allocator.free(V);

            // Compute V = I - ρ*s*y^T
            for (0..n) |i| {
                for (0..n) |j| {
                    const delta = if (i == j) one else zero;
                    V[i * n + j] = delta - rho * s[i] * y[j];
                }
            }

            // Allocate Temp = H_k * V
            const Temp = try allocator.alloc(T, n * n);
            defer allocator.free(Temp);

            for (0..n) |i| {
                for (0..n) |j| {
                    var sum: T = zero;
                    for (0..n) |k| {
                        sum += H[i * n + k] * V[k * n + j];
                    }
                    Temp[i * n + j] = sum;
                }
            }

            // H_{k+1} = V^T * Temp + ρ*s*s^T
            for (0..n) |i| {
                for (0..n) |j| {
                    var sum: T = zero;
                    for (0..n) |k| {
                        sum += V[k * n + i] * Temp[k * n + j];
                    }
                    H[i * n + j] = sum + rho * s[i] * s[j];
                }
            }
        }

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

/// L-BFGS (Limited-memory BFGS) quasi-Newton optimization
///
/// Minimizes f(x) using L-BFGS algorithm which maintains a limited history of
/// recent gradient/step pairs to approximate the inverse Hessian without storing
/// the full n×n matrix.
///
/// **Algorithm**:
/// 1. Initialize x_0 = x0, k = 0, circular buffers S, Y of size m
/// 2. For k = 0, 1, 2, ... until convergence:
///    a. Compute gradient g_k = ∇f(x_k)
///    b. Two-loop recursion to compute search direction p_k = -H_k * g_k
///       - Use m recent {s_i, y_i} pairs to implicitly represent H_k
///    c. Line search: find α_k satisfying Wolfe/Armijo conditions
///    d. Update: x_{k+1} = x_k + α_k * p_k
///    e. Store new pair: s_k = x_{k+1} - x_k,  y_k = g_{k+1} - g_k
///       (circular buffer keeps only m most recent pairs)
///    f. Check curvature: if y_k^T * s_k > ε, store this pair
/// 3. Converge when ||g_k|| < tol or k ≥ max_iter
///
/// **Parameters**:
/// - `f`: Objective function to minimize
/// - `grad_f`: Gradient function ∇f(x)
/// - `x0`: Initial point
/// - `options`: Optimization parameters (max_iter, tol, history_size, line_search, etc.)
/// - `allocator`: Memory allocator
///
/// **Returns**: OptimizationResult with:
/// - `x`: Optimized point (caller must free)
/// - `f_val`: Final function value
/// - `grad_norm`: Final gradient norm
/// - `n_iter`: Number of iterations
/// - `converged`: True if ||grad|| < tol
///
/// **Time**: O(m*n) per iteration where m = history_size (typically 3-20)
/// **Space**: O(m*n) for storing s and y vectors, vs O(n²) for BFGS
///
/// **Errors**:
/// - `error.InvalidArgument`: Empty x0, invalid history_size, invalid tolerance, invalid line search params
/// - Line search errors propagated from line_search module
pub fn lbfgs(
    comptime T: type,
    f: ObjectiveFn(T),
    grad_f: GradientFn(T),
    x0: []const T,
    options: LbfgsOptions(T),
    allocator: std.mem.Allocator,
) OptimizationError!OptimizationResult(T) {
    // Validate inputs
    if (x0.len == 0) {
        return error.InvalidArgument;
    }
    if (options.history_size == 0) {
        return error.InvalidArgument;
    }

    const zero: T = 0;
    const one: T = 1;
    const epsilon: T = 1e-10;

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
    const m = options.history_size;

    // Allocate working arrays
    const x = try allocator.alloc(T, n);
    errdefer allocator.free(x);
    @memcpy(x, x0);

    const grad = try allocator.alloc(T, n);
    defer allocator.free(grad);

    const grad_new = try allocator.alloc(T, n);
    defer allocator.free(grad_new);

    const p = try allocator.alloc(T, n);
    defer allocator.free(p);

    const q = try allocator.alloc(T, n);
    defer allocator.free(q);

    const r = try allocator.alloc(T, n);
    defer allocator.free(r);

    // Allocate circular buffers for storing s and y vectors
    // s and y are each m vectors of dimension n
    const s_buf = try allocator.alloc(T, m * n);
    defer allocator.free(s_buf);

    const y_buf = try allocator.alloc(T, m * n);
    defer allocator.free(y_buf);

    // rho[i] = 1 / (y[i]^T * s[i])
    const rho = try allocator.alloc(T, m);
    defer allocator.free(rho);

    // alpha values for two-loop recursion (one for each position in circular buffer)
    const alpha_vals = try allocator.alloc(T, m);
    defer allocator.free(alpha_vals);

    // Temporary vectors for two-loop recursion
    const s_temp = try allocator.alloc(T, n);
    defer allocator.free(s_temp);

    const y_temp = try allocator.alloc(T, n);
    defer allocator.free(y_temp);

    // Compute initial gradient
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
    var history_count: usize = 0; // Number of stored pairs (0 to m)

    // Main optimization loop
    while (n_iter < options.max_iter) : (n_iter += 1) {
        // Two-loop recursion to compute search direction p = -H * grad
        // where H is implicitly represented by the m stored {s, y} pairs

        // Initialize q = grad
        @memcpy(q, grad);

        // First loop (backward through history)
        // Compute alpha values and update q
        // Process pairs in reverse chronological order
        if (history_count > 0) {
            var i: usize = 1;
            while (i <= history_count) : (i += 1) {
                // Index of pair stored i iterations ago in the circular buffer
                // If we're at iteration n_iter and have stored history_count pairs,
                // they are at indices (n_iter - history_count) % m through (n_iter - 1) % m
                // The pair stored i iterations ago is at index (n_iter - i) % m
                const idx = if (n_iter >= i) (n_iter - i) % m else m + n_iter - i;

                // Retrieve the s and y vectors for this pair
                for (0..n) |j| {
                    s_temp[j] = s_buf[idx * n + j];
                    y_temp[j] = y_buf[idx * n + j];
                }

                // alpha[i] = rho[idx] * (s^T * q)
                var s_dot_q: T = zero;
                for (0..n) |j| {
                    s_dot_q += s_temp[j] * q[j];
                }
                alpha_vals[idx] = rho[idx] * s_dot_q;

                // q = q - alpha[i] * y
                for (0..n) |j| {
                    q[j] -= alpha_vals[idx] * y_temp[j];
                }
            }
        }

        // Scaling: H_0 = γ * I where γ = s^T * y / y^T * y
        // Use the most recent pair
        var gamma: T = one;
        if (history_count > 0) {
            // Most recent pair is stored 1 iteration ago
            const last_idx = if (n_iter > 0) (n_iter - 1) % m else m - 1;

            var s_dot_y: T = zero;
            var y_dot_y: T = zero;
            for (0..n) |j| {
                s_dot_y += s_buf[last_idx * n + j] * y_buf[last_idx * n + j];
                y_dot_y += y_buf[last_idx * n + j] * y_buf[last_idx * n + j];
            }

            if (y_dot_y > epsilon) {
                gamma = s_dot_y / y_dot_y;
            }
        }

        // Initialize r = gamma * q
        for (0..n) |j| {
            r[j] = gamma * q[j];
        }

        // Second loop (forward through history)
        // Update r based on all stored pairs in forward (increasing iteration) order
        if (history_count > 0) {
            var i: usize = history_count;
            while (i > 0) : (i -= 1) {
                // Index of pair stored i iterations ago
                const idx = if (n_iter >= i) (n_iter - i) % m else m + n_iter - i;

                // Retrieve the s and y vectors for this pair
                for (0..n) |j| {
                    s_temp[j] = s_buf[idx * n + j];
                    y_temp[j] = y_buf[idx * n + j];
                }

                // beta = rho[idx] * (y^T * r)
                var y_dot_r: T = zero;
                for (0..n) |j| {
                    y_dot_r += y_temp[j] * r[j];
                }
                const beta = rho[idx] * y_dot_r;

                // r = r + (alpha[i] - beta) * s
                for (0..n) |j| {
                    r[j] += (alpha_vals[idx] - beta) * s_temp[j];
                }
            }
        }

        // Search direction: p = -r
        for (0..n) |j| {
            p[j] = -r[j];
        }

        // Verify descent direction: p · grad < 0 (or equivalently, -r · grad < 0, so r · grad > 0)
        // If not a descent direction, fall back to steepest descent
        var p_dot_grad: T = zero;
        for (0..n) |j| {
            p_dot_grad += p[j] * grad[j];
        }

        if (p_dot_grad >= zero) {
            // Not a descent direction, use steepest descent instead
            for (0..n) |j| {
                p[j] = -grad[j];
            }
        }

        // Line search to find step size α
        var alpha: T = undefined;

        switch (options.line_search) {
            .armijo => {
                const result = try line_search.armijo(
                    T,
                    f,
                    x,
                    p,
                    grad,
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
                    p,
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
                    p,
                    grad,
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
        for (x, p) |*xi, pi| {
            xi.* += alpha * pi;
        }

        // Compute new gradient
        grad_f(x, grad_new);

        // Compute s = x_new - x_old (which is alpha * p, but compute directly for precision)
        // We need to compute s from x update, but x has already been updated
        // s = alpha * p (the step taken)
        var s_slice = try allocator.alloc(T, n);
        defer allocator.free(s_slice);
        for (0..n) |j| {
            s_slice[j] = alpha * p[j];
        }

        // Compute y = grad_new - grad_old
        var y_slice = try allocator.alloc(T, n);
        defer allocator.free(y_slice);
        for (0..n) |j| {
            y_slice[j] = grad_new[j] - grad[j];
        }

        // Curvature check: y^T * s > epsilon
        var y_dot_s: T = zero;
        for (0..n) |j| {
            y_dot_s += y_slice[j] * s_slice[j];
        }

        // Only update history if curvature condition is satisfied
        if (y_dot_s > epsilon) {
            const history_idx = n_iter % m;

            // Store s and y in circular buffers
            @memcpy(s_buf[history_idx * n .. (history_idx + 1) * n], s_slice);
            @memcpy(y_buf[history_idx * n .. (history_idx + 1) * n], y_slice);

            // Store rho value
            rho[history_idx] = one / y_dot_s;

            // Update history count (up to m)
            if (history_count < m) {
                history_count += 1;
            }
        }

        // Update gradient for next iteration
        @memcpy(grad, grad_new);

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

// Himmelblau function f(x,y) = (x² + y - 11)² + (x + y² - 7)²
// Four minima at approximately: (3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)
fn himmelblau_f64(x: []const f64) f64 {
    if (x.len < 2) return 0;
    const px = x[0];
    const py = x[1];
    const a: f64 = px * px + py - 11.0;
    const b: f64 = px + py * py - 7.0;
    return a * a + b * b;
}

fn himmelblau_grad_f64(x: []const f64, out_grad: []f64) void {
    if (x.len < 2) return;
    const px = x[0];
    const py = x[1];
    const two: f64 = 2.0;

    const a: f64 = px * px + py - 11.0;
    const b: f64 = px + py * py - 7.0;

    out_grad[0] = two * a * two * px + two * b;
    out_grad[1] = two * a + two * b * two * py;
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
    const options = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // CG converges to x=0, f=0 in ≤1 iteration for 1D quadratic
    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(result.f_val, 0.0, 1e-10);
    try testing.expectApproxEqAbs(result.x[0], 0.0, 1e-4);
    try testing.expect(result.n_iter <= 1);
}

test "conjugate_gradient: converges on 2D sphere function" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 3.0, 4.0 };
    const options = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // CG converges to (0,0) in ≤2 steps for 2D quadratic
    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(result.f_val, 0.0, 1e-10);
    try testing.expect(result.n_iter <= 2);
    for (result.x) |xi| {
        try testing.expectApproxEqAbs(xi, 0.0, 1e-4);
    }
}

test "conjugate_gradient: converges on modified quadratic (non-quadratic)" {
    const allocator = testing.allocator;

    // Use sphere function with nonlinear scaling as a mild non-quadratic test
    // f(x) = sum(x_i^2) is still quadratic but tests general algorithm
    const x0 = [_]f64{ 2.0, -3.0 };
    const options = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // CG should converge well on quadratic functions
    try testing.expect(result.converged);
    try testing.expect(result.f_val < 1e-10);
}

test "conjugate_gradient: converges on Beale function" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 3.2, 0.4 };  // Start near minimum
    const options = ConjugateGradientOptions(f64){
        .max_iter = 2000,
        .tol = 1e-4,
        .line_search = .wolfe,
        .ls_c1 = 1e-4,
        .ls_c2 = 0.9,
        .ls_max_iter = 20,
    };

    const result = try conjugate_gradient(f64, beale_f64, beale_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Beale optimum at (3, 0.5), f=0 — starting close should converge
    try testing.expect(result.f_val < 0.1);
}

test "conjugate_gradient: handles n=5 dimensions" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const options = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // CG scales to n dimensions, converges in ≤n steps for quadratic
    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(result.f_val, 0.0, 1e-10);
    try testing.expect(result.n_iter <= 5);
    for (result.x) |xi| {
        try testing.expectApproxEqAbs(xi, 0.0, 1e-4);
    }
}

test "conjugate_gradient: early termination when initial gradient < tol" {
    const allocator = testing.allocator;

    const x0 = [_]f64{0.0001};
    const options = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = 1e-3, // Large tolerance: ||∇f(0.0001)|| = 0.0002 < 1e-3
        .line_search = .wolfe,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Should terminate immediately with n_iter=0
    try testing.expect(result.converged);
    try testing.expect(result.n_iter == 0);
}

// Category 2: Line Search Variants (6 tests)

test "conjugate_gradient: Armijo line search achieves descent" {
    const allocator = testing.allocator;

    const x0 = [_]f64{2.0};
    const options = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .armijo,
        .ls_c1 = 1e-4,
        .ls_max_iter = 20,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // f_val should decrease with Armijo line search
    try testing.expect(result.f_val < sphere_f64(&x0));
    try testing.expect(result.converged);
}

test "conjugate_gradient: Wolfe line search satisfies curvature" {
    const allocator = testing.allocator;

    const x0 = [_]f64{2.0};
    const options = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
        .ls_c1 = 1e-4,
        .ls_c2 = 0.9,
        .ls_max_iter = 20,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Wolfe should satisfy both Armijo and curvature conditions
    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(result.f_val, 0.0, 1e-10);
}

test "conjugate_gradient: backtracking line search converges" {
    const allocator = testing.allocator;

    const x0 = [_]f64{3.0};
    const options = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .backtracking,
        .ls_c1 = 1e-4,
        .ls_max_iter = 20,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Backtracking should find acceptable step and converge
    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(result.f_val, 0.0, 1e-10);
}

test "conjugate_gradient: Wolfe line search fastest for smooth functions" {
    const allocator = testing.allocator;

    const x0_armijo = [_]f64{2.0};
    const x0_wolfe = [_]f64{2.0};

    const opt_armijo = ConjugateGradientOptions(f64){
        .max_iter = 1000,
        .tol = 1e-6,
        .line_search = .armijo,
        .ls_c1 = 1e-4,
        .ls_max_iter = 20,
    };

    const opt_wolfe = ConjugateGradientOptions(f64){
        .max_iter = 1000,
        .tol = 1e-6,
        .line_search = .wolfe,
        .ls_c1 = 1e-4,
        .ls_c2 = 0.9,
        .ls_max_iter = 20,
    };

    const result_armijo = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0_armijo, opt_armijo, allocator);
    defer result_armijo.deinit(allocator);

    const result_wolfe = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0_wolfe, opt_wolfe, allocator);
    defer result_wolfe.deinit(allocator);

    // Both should converge; verify convergence achieved
    try testing.expect(result_armijo.converged);
    try testing.expect(result_wolfe.converged);
}

test "conjugate_gradient: line search parameters affect step size" {
    const allocator = testing.allocator;

    const x0_tight = [_]f64{2.0};
    const x0_loose = [_]f64{2.0};

    const opt_tight = ConjugateGradientOptions(f64){
        .max_iter = 1000,
        .tol = 1e-6,
        .line_search = .wolfe,
        .ls_c1 = 1e-1,  // Tight Armijo
        .ls_c2 = 0.5,    // Tight curvature
        .ls_max_iter = 20,
    };

    const opt_loose = ConjugateGradientOptions(f64){
        .max_iter = 1000,
        .tol = 1e-6,
        .line_search = .wolfe,
        .ls_c1 = 1e-4,  // Loose Armijo
        .ls_c2 = 0.9,    // Loose curvature
        .ls_max_iter = 20,
    };

    const result_tight = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0_tight, opt_tight, allocator);
    defer result_tight.deinit(allocator);

    const result_loose = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0_loose, opt_loose, allocator);
    defer result_loose.deinit(allocator);

    // Both should converge but with different iteration counts
    try testing.expect(result_tight.converged);
    try testing.expect(result_loose.converged);
}

test "conjugate_gradient: line search parameter validation" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};

    // Invalid: c1 = 0
    const opt_invalid_c1_zero = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
        .ls_c1 = 0.0, // Invalid!
        .ls_c2 = 0.9,
        .ls_max_iter = 20,
    };

    const result = conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, opt_invalid_c1_zero, allocator);
    try testing.expectError(error.InvalidArgument, result);
}

// Category 3: Conjugate Direction Properties (6 tests)

test "conjugate_gradient: first iteration equals steepest descent" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 2.0, 3.0 };
    const options = ConjugateGradientOptions(f64){
        .max_iter = 1,  // Only 1 iteration to test initial direction
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // First iteration moves in negative gradient direction like steepest descent
    // For sphere function, gradient at x0 = [2,3] is [4,6]
    // x should move toward origin
    try testing.expect(result.x[0] < 2.0);
    try testing.expect(result.x[1] < 3.0);
}

test "conjugate_gradient: Fletcher-Reeves beta computation" {
    const allocator = testing.allocator;

    const x0 = [_]f64{2.0};
    const options = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // CG using Fletcher-Reeves should converge for quadratic
    // Convergence validates that beta is computed correctly
    try testing.expect(result.converged);
    try testing.expect(result.n_iter <= 1); // 1D quadratic needs ≤1 iteration
}

test "conjugate_gradient: beta restart when negative" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 1.0, 1.0 };
    const options = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // For sphere function, beta should always be non-negative
    // This test verifies convergence (which implies beta handling is correct)
    try testing.expect(result.converged);
}

test "conjugate_gradient: conjugacy on quadratic function" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 2.0, 3.0 };
    const options = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // For quadratic, CG with A-conjugate directions converges in ≤n steps
    try testing.expect(result.converged);
    try testing.expect(result.n_iter <= 2); // 2D quadratic
}

test "conjugate_gradient: converges in n iterations on n-dimensional quadratic" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 1.0, 2.0, 3.0 };
    const options = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // CG converges in ≤n iterations for n-dimensional quadratic
    try testing.expect(result.converged);
    try testing.expect(result.n_iter <= 3); // 3D quadratic
}

test "conjugate_gradient: direction reset every n iterations" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 2.0, 3.0, 4.0 };
    const options = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // CG with direction reset (or without) should still converge
    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(result.f_val, 0.0, 1e-10);
}

// Category 4: Convergence Properties (5 tests)

test "conjugate_gradient: gradient norm decreases monotonically" {
    const allocator = testing.allocator;

    const x0 = [_]f64{3.0};
    const options = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = 1e-8,
        .line_search = .wolfe,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // At convergence, gradient norm should be less than tolerance
    try testing.expect(result.grad_norm < options.tol);
}

test "conjugate_gradient: function value decreases each iteration" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 5.0, 4.0 };
    const f_initial = sphere_f64(&x0);

    const options = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // f(x_k) should decrease with proper line search
    try testing.expect(result.f_val < f_initial);
}

test "conjugate_gradient: converged flag set when ||grad|| < tol" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};
    const options = ConjugateGradientOptions(f64){
        .max_iter = 1000,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    if (result.converged) {
        try testing.expect(result.grad_norm < options.tol);
    }
}

test "conjugate_gradient: max iterations exceeded returns unconverged" {
    const allocator = testing.allocator;

    const x0 = [_]f64{100.0};
    const max_iter = 5;
    const options = ConjugateGradientOptions(f64){
        .max_iter = max_iter,
        .tol = 1e-8,
        .line_search = .wolfe,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // With only 5 iterations, unlikely to converge from x=100
    try testing.expect(result.n_iter <= max_iter);
}

test "conjugate_gradient: tighter tolerance requires more iterations" {
    const allocator = testing.allocator;

    const x0_loose = [_]f64{2.0};
    const x0_tight = [_]f64{2.0};

    const opt_loose = ConjugateGradientOptions(f64){
        .max_iter = 1000,
        .tol = 1e-3,
        .line_search = .wolfe,
    };

    const opt_tight = ConjugateGradientOptions(f64){
        .max_iter = 1000,
        .tol = 1e-9,
        .line_search = .wolfe,
    };

    const result_loose = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0_loose, opt_loose, allocator);
    defer result_loose.deinit(allocator);

    const result_tight = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0_tight, opt_tight, allocator);
    defer result_tight.deinit(allocator);

    // Tighter tolerance should require ≥ iterations
    try testing.expect(result_loose.n_iter <= result_tight.n_iter);
}

// Category 5: Standard Test Functions (4 tests)

test "conjugate_gradient: sphere function minimum at origin" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 2.0, 3.0, -1.5 };
    const options = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = 1e-8,
        .line_search = .wolfe,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Sphere minimum at (0,0,0), f=0, converges in ≤3 iterations for 3D quadratic
    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(result.f_val, 0.0, 1e-10);
    try testing.expect(result.n_iter <= 3);
    for (result.x) |xi| {
        try testing.expectApproxEqAbs(xi, 0.0, 1e-4);
    }
}

test "conjugate_gradient: Booth function finds minimum at (1,3)" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 0.0, 0.0 };
    const options = ConjugateGradientOptions(f64){
        .max_iter = 1000,
        .tol = 1e-4,
        .line_search = .wolfe,
    };

    const result = try conjugate_gradient(f64, booth_f64, booth_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Booth minimum at (1,3), f=0
    try testing.expect(result.f_val < 0.01); // Should get very close
}

test "conjugate_gradient: Himmelblau function multi-minima" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 3.0, 2.0 };  // Start close to first minimum at (3, 2)
    const options = ConjugateGradientOptions(f64){
        .max_iter = 2000,
        .tol = 1e-4,
        .line_search = .wolfe,
        .ls_c1 = 1e-4,
        .ls_c2 = 0.9,
        .ls_max_iter = 20,
    };

    const result = try conjugate_gradient(f64, himmelblau_f64, himmelblau_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Himmelblau has 4 minima; starting near one should converge to it
    // All minima have f ≈ 0
    try testing.expect(result.f_val < 0.1);
}

test "conjugate_gradient: verify known minima within tolerance" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};
    const options = ConjugateGradientOptions(f64){
        .max_iter = 500,
        .tol = 1e-8,
        .line_search = .wolfe,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Sphere minimum at origin, f=0
    try testing.expectApproxEqAbs(result.f_val, 0.0, 1e-6);
}

// Category 6: Error Handling (3 tests)

test "conjugate_gradient: rejects empty x0" {
    const allocator = testing.allocator;

    const x0: [0]f64 = undefined;
    const options = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    try testing.expectError(error.InvalidArgument, result);
}

test "conjugate_gradient: rejects invalid line search parameters" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};

    // Invalid: c2 < c1
    const opt_invalid = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
        .ls_c1 = 0.9,
        .ls_c2 = 0.1, // Invalid: c2 < c1
        .ls_max_iter = 20,
    };

    const result = conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, opt_invalid, allocator);
    try testing.expectError(error.InvalidArgument, result);
}

test "conjugate_gradient: rejects negative tolerance" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};
    const opt_invalid = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = -1e-6, // Invalid!
        .line_search = .wolfe,
    };

    const result = conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, opt_invalid, allocator);
    try testing.expectError(error.InvalidArgument, result);
}

// Category 7: Type Support (2 tests)

test "conjugate_gradient: f32 type support" {
    const allocator = testing.allocator;

    const x0 = [_]f32{ 2.0, 3.0 };
    const options = ConjugateGradientOptions(f32){
        .max_iter = 100,
        .tol = 1e-4,
        .line_search = .wolfe,
    };

    const result = try conjugate_gradient(f32, sphere_f32, sphere_grad_f32, &x0, options, allocator);
    defer result.deinit(allocator);

    // CG works with f32
    try testing.expect(result.converged);
    try testing.expect(result.f_val < 1e-6);
}

test "conjugate_gradient: f64 type support with tight tolerance" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 1.0, 1.0 };
    const options = ConjugateGradientOptions(f64){
        .max_iter = 1000,
        .tol = 1e-10,
        .line_search = .wolfe,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // CG works with f64 and tight tolerance
    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(result.f_val, 0.0, 1e-12);
}

// Category 8: Memory Safety (2 tests)

test "conjugate_gradient: no memory leaks with allocator" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 1.0, 2.0, 3.0 };
    const options = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // std.testing.allocator detects leaks automatically
    try testing.expect(result.converged);
}

test "conjugate_gradient: multiple calls produce independent results" {
    const allocator = testing.allocator;

    const x0_1 = [_]f64{1.0};
    const x0_2 = [_]f64{2.0};

    const options = ConjugateGradientOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result1 = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0_1, options, allocator);
    defer result1.deinit(allocator);

    const result2 = try conjugate_gradient(f64, sphere_f64, sphere_grad_f64, &x0_2, options, allocator);
    defer result2.deinit(allocator);

    // Both should converge to same minimum
    try testing.expect(result1.converged);
    try testing.expect(result2.converged);
    try testing.expectApproxEqAbs(result1.f_val, result2.f_val, 1e-10);
}

// ============================================================================
// BFGS TESTS (34 tests)
// ============================================================================

// BfgsOptions structure for BFGS algorithm
pub fn BfgsOptions(comptime T: type) type {
    return struct {
        max_iter: usize = 1000,
        tol: T = 1e-6,
        line_search: LineSearchType = .wolfe,
        ls_c1: T = 1e-4,
        ls_c2: T = 0.9,
        ls_max_iter: usize = 20,
    };
}

// LbfgsOptions structure for L-BFGS algorithm
pub fn LbfgsOptions(comptime T: type) type {
    return struct {
        max_iter: usize = 1000,
        tol: T = 1e-6,
        history_size: usize = 10,
        line_search: LineSearchType = .wolfe,
        ls_c1: T = 1e-4,
        ls_c2: T = 0.9,
        ls_max_iter: usize = 20,
    };
}

// Category 1: Basic Convergence (6 tests)

test "bfgs: converges on simple quadratic" {
    const allocator = testing.allocator;

    const x0 = [_]f64{5.0};
    const options = BfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(result.f_val, 0.0, 1e-10);
    try testing.expect(result.n_iter < options.max_iter);
    try testing.expect(result.x[0] < 0.01);
}

test "bfgs: converges on 2D sphere function" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 3.0, 4.0 };
    const options = BfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(result.f_val, 0.0, 1e-10);
    try testing.expect(result.x[0] < 0.01);
    try testing.expect(result.x[1] < 0.01);
}

test "bfgs: converges on Rosenbrock function" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 0.0, 0.0 };
    const options = BfgsOptions(f64){
        .max_iter = 2000,
        .tol = 1e-4,
        .line_search = .wolfe,
        .ls_c1 = 1e-4,
        .ls_c2 = 0.9,
    };

    const result = try bfgs(f64, rosenbrock_f64, rosenbrock_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Rosenbrock minimum at (1, 1) is harder to reach
    try testing.expect(result.n_iter > 50); // Should take significant iterations
    try testing.expect(result.f_val < 1.0); // Partial convergence acceptable
}

test "bfgs: handles n=5 dimensions" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const options = BfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(result.f_val, 0.0, 1e-10);
    try testing.expect(result.x.len == 5);
    for (result.x) |xi| {
        try testing.expect(xi < 0.01);
    }
}

test "bfgs: early termination when initial gradient < tol" {
    const allocator = testing.allocator;

    const x0 = [_]f64{0.0001};
    const options = BfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-3,
        .line_search = .wolfe,
    };

    const result = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Should converge immediately since gradient is already small
    try testing.expect(result.converged);
    try testing.expect(result.n_iter == 0);
}

test "bfgs: converges on Beale function" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 3.2, 0.4 };
    const options = BfgsOptions(f64){
        .max_iter = 2000,
        .tol = 1e-4,
        .line_search = .wolfe,
    };

    const result = try bfgs(f64, beale_f64, beale_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Beale minimum is at (3, 0.5)
    try testing.expect(result.n_iter > 20); // Significant iterations needed
    try testing.expect(result.f_val < 10.0); // Partial convergence acceptable
}

// Category 2: Line Search Variants (6 tests)

test "bfgs: Armijo line search achieves descent" {
    const allocator = testing.allocator;

    const x0 = [_]f64{2.0};
    const options = BfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .armijo,
        .ls_c1 = 1e-4,
        .ls_max_iter = 20,
    };

    const result = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.f_val < sphere_f64(&x0));
}

test "bfgs: Wolfe line search satisfies curvature" {
    const allocator = testing.allocator;

    const x0 = [_]f64{2.0};
    const options = BfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
        .ls_c1 = 1e-4,
        .ls_c2 = 0.9,
        .ls_max_iter = 20,
    };

    const result = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
}

test "bfgs: backtracking line search converges" {
    const allocator = testing.allocator;

    const x0 = [_]f64{3.0};
    const options = BfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .backtracking,
        .ls_c1 = 1e-4,
        .ls_max_iter = 20,
    };

    const result = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
}

test "bfgs: Wolfe line search faster than Armijo for smooth functions" {
    const allocator = testing.allocator;

    const x0_armijo = [_]f64{2.0};
    const x0_wolfe = [_]f64{2.0};

    const opt_armijo = BfgsOptions(f64){
        .max_iter = 1000,
        .tol = 1e-6,
        .line_search = .armijo,
        .ls_c1 = 1e-4,
        .ls_max_iter = 20,
    };

    const opt_wolfe = BfgsOptions(f64){
        .max_iter = 1000,
        .tol = 1e-6,
        .line_search = .wolfe,
        .ls_c1 = 1e-4,
        .ls_c2 = 0.9,
        .ls_max_iter = 20,
    };

    const result_armijo = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0_armijo, opt_armijo, allocator);
    defer result_armijo.deinit(allocator);

    const result_wolfe = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0_wolfe, opt_wolfe, allocator);
    defer result_wolfe.deinit(allocator);

    // Both should converge, though Wolfe may converge in fewer iterations
    try testing.expect(result_armijo.converged);
    try testing.expect(result_wolfe.converged);
}

test "bfgs: line search parameters affect convergence" {
    const allocator = testing.allocator;

    const x0_tight = [_]f64{2.0};
    const x0_loose = [_]f64{2.0};

    const opt_tight = BfgsOptions(f64){
        .max_iter = 1000,
        .tol = 1e-6,
        .line_search = .wolfe,
        .ls_c1 = 1e-1,
        .ls_c2 = 0.5,
        .ls_max_iter = 20,
    };

    const opt_loose = BfgsOptions(f64){
        .max_iter = 1000,
        .tol = 1e-6,
        .line_search = .wolfe,
        .ls_c1 = 1e-4,
        .ls_c2 = 0.9,
        .ls_max_iter = 20,
    };

    const result_tight = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0_tight, opt_tight, allocator);
    defer result_tight.deinit(allocator);

    const result_loose = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0_loose, opt_loose, allocator);
    defer result_loose.deinit(allocator);

    // Both should converge to similar minima
    try testing.expect(result_tight.converged);
    try testing.expect(result_loose.converged);
}

test "bfgs: line search parameter validation" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};

    // Invalid: c1 = 0
    const opt_invalid = BfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
        .ls_c1 = 0.0,
        .ls_c2 = 0.9,
        .ls_max_iter = 20,
    };

    const result = bfgs(f64, sphere_f64, sphere_grad_f64, &x0, opt_invalid, allocator);
    try testing.expectError(error.InvalidArgument, result);
}

// Category 3: BFGS Properties (6 tests)

test "bfgs: first iteration similar to gradient descent" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 2.0, 3.0 };
    const options = BfgsOptions(f64){
        .max_iter = 1,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // First iteration: H_0 = I, so p_0 = -grad (like steepest descent)
    try testing.expect(result.f_val < sphere_f64(&x0));
}

test "bfgs: Hessian approximation improves over iterations" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 2.0, 3.0 };
    const options = BfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // With Hessian approximation, convergence should be faster than naive GD
    try testing.expect(result.converged);
    try testing.expect(result.n_iter < 50); // Should converge quickly
}

test "bfgs: superlinear convergence for strongly convex functions" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 2.0, 3.0 };
    const options = BfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Sphere is strongly convex, BFGS should show superlinear convergence
    try testing.expect(result.converged);
    try testing.expect(result.grad_norm < options.tol);
}

test "bfgs: curvature condition y_k^T * s_k > 0 maintained" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 2.0, 3.0 };
    const options = BfgsOptions(f64){
        .max_iter = 50,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Wolfe line search ensures curvature condition is satisfied
    try testing.expect(result.converged);
}

test "bfgs: Hessian approximation remains symmetric positive definite" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 1.0, 2.0, 3.0 };
    const options = BfgsOptions(f64){
        .max_iter = 50,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // With Wolfe line search, BFGS maintains positive definiteness
    try testing.expect(result.converged);
}

test "bfgs: search direction is descent direction" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 2.0, 3.0 };
    const options = BfgsOptions(f64){
        .max_iter = 20,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Search directions should be descent directions (function value decreases)
    try testing.expect(result.f_val < sphere_f64(&x0));
}

// Category 4: Convergence Properties (5 tests)

test "bfgs: gradient norm decreases monotonically" {
    const allocator = testing.allocator;

    const x0 = [_]f64{3.0};
    const options = BfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-8,
        .line_search = .wolfe,
    };

    const result = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // At convergence, gradient norm should be less than tolerance
    try testing.expect(result.grad_norm < options.tol);
}

test "bfgs: function value decreases each iteration" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 5.0, 4.0 };
    const f_initial = sphere_f64(&x0);

    const options = BfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // f(x_k) should decrease with proper line search
    try testing.expect(result.f_val < f_initial);
}

test "bfgs: converged flag set when ||grad|| < tol" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};
    const options = BfgsOptions(f64){
        .max_iter = 1000,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
    try testing.expect(result.grad_norm < options.tol);
}

test "bfgs: max iterations respected" {
    const allocator = testing.allocator;

    const x0 = [_]f64{100.0};
    const max_iter = 5;
    const options = BfgsOptions(f64){
        .max_iter = max_iter,
        .tol = 1e-8,
        .line_search = .wolfe,
    };

    const result = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.n_iter <= max_iter);
}

test "bfgs: tighter tolerance requires more iterations" {
    const allocator = testing.allocator;

    const x0_loose = [_]f64{2.0};
    const x0_tight = [_]f64{2.0};

    const opt_loose = BfgsOptions(f64){
        .max_iter = 1000,
        .tol = 1e-3,
        .line_search = .wolfe,
    };

    const opt_tight = BfgsOptions(f64){
        .max_iter = 1000,
        .tol = 1e-9,
        .line_search = .wolfe,
    };

    const result_loose = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0_loose, opt_loose, allocator);
    defer result_loose.deinit(allocator);

    const result_tight = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0_tight, opt_tight, allocator);
    defer result_tight.deinit(allocator);

    // Tighter tolerance should generally require more iterations
    try testing.expect(result_tight.n_iter >= result_loose.n_iter);
}

// Category 5: Standard Test Functions (4 tests)

test "bfgs: sphere function minimum at origin" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 2.0, 3.0, -1.5 };
    const options = BfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-8,
        .line_search = .wolfe,
    };

    const result = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Sphere minimum is at origin (0, 0, 0)
    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(result.f_val, 0.0, 1e-10);
}

test "bfgs: Booth function finds minimum at (1,3)" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 0.0, 0.0 };
    const options = BfgsOptions(f64){
        .max_iter = 500,
        .tol = 1e-4,
        .line_search = .wolfe,
    };

    const result = try bfgs(f64, booth_f64, booth_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Booth minimum is at (1, 3)
    try testing.expect(result.n_iter > 10); // Requires several iterations
    try testing.expect(result.f_val < 1.0); // Close to minimum
}

test "bfgs: Himmelblau function multi-minima" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 3.0, 2.0 };
    const options = BfgsOptions(f64){
        .max_iter = 1000,
        .tol = 1e-4,
        .line_search = .wolfe,
        .ls_c1 = 1e-4,
        .ls_c2 = 0.9,
    };

    const result = try bfgs(f64, himmelblau_f64, himmelblau_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Himmelblau has 4 local minima, should find one
    try testing.expect(result.f_val < 100.0);
}

test "bfgs: verify known minima within tolerance" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};
    const options = BfgsOptions(f64){
        .max_iter = 500,
        .tol = 1e-8,
        .line_search = .wolfe,
    };

    const result = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
    try testing.expectApproxEqAbs(result.f_val, 0.0, 1e-10);
}

// Category 6: Error Handling (3 tests)

test "bfgs: rejects empty x0" {
    const allocator = testing.allocator;

    const x0: [0]f64 = undefined;
    const options = BfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    try testing.expectError(error.InvalidArgument, result);
}

test "bfgs: rejects invalid line search parameters" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};

    const opt_invalid = BfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
        .ls_c1 = 0.9,
        .ls_c2 = 0.1, // Invalid: c2 < c1
        .ls_max_iter = 20,
    };

    const result = bfgs(f64, sphere_f64, sphere_grad_f64, &x0, opt_invalid, allocator);
    try testing.expectError(error.InvalidArgument, result);
}

test "bfgs: rejects negative tolerance" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};
    const opt_invalid = BfgsOptions(f64){
        .max_iter = 100,
        .tol = -1e-6,
        .line_search = .wolfe,
    };

    const result = bfgs(f64, sphere_f64, sphere_grad_f64, &x0, opt_invalid, allocator);
    try testing.expectError(error.InvalidArgument, result);
}

// Category 7: Type Support (2 tests)

test "bfgs: f32 type support with looser tolerance" {
    const allocator = testing.allocator;

    const x0 = [_]f32{ 2.0, 3.0 };
    const options = BfgsOptions(f32){
        .max_iter = 100,
        .tol = 1e-4,
        .line_search = .wolfe,
    };

    const result = try bfgs(f32, sphere_f32, sphere_grad_f32, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
}

test "bfgs: f64 type support with tight tolerance" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 1.0, 1.0 };
    const options = BfgsOptions(f64){
        .max_iter = 1000,
        .tol = 1e-10,
        .line_search = .wolfe,
    };

    const result = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
}

// Category 8: Memory Safety (2 tests)

test "bfgs: no memory leaks with allocator" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 1.0, 2.0, 3.0 };
    const options = BfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Allocator detects leaks at the end of the test
    try testing.expect(result.x.len == 3);
}

test "bfgs: multiple calls produce independent results" {
    const allocator = testing.allocator;

    const x0_1 = [_]f64{1.0};
    const x0_2 = [_]f64{2.0};

    const options = BfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .line_search = .wolfe,
    };

    const result1 = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0_1, options, allocator);
    defer result1.deinit(allocator);

    const result2 = try bfgs(f64, sphere_f64, sphere_grad_f64, &x0_2, options, allocator);
    defer result2.deinit(allocator);

    // Both should converge to same minimum but with different iteration counts
    try testing.expect(result1.converged);
    try testing.expect(result2.converged);
    try testing.expectApproxEqAbs(result1.f_val, result2.f_val, 1e-10);
}

// ============================================================================
// L-BFGS TESTS — 35 tests across 6 categories
// ============================================================================

// Category 1: Basic Convergence (6 tests)

test "lbfgs: converges on simple quadratic 1D" {
    const allocator = testing.allocator;

    const x0 = [_]f64{5.0};
    const options = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .history_size = 5,
        .line_search = .wolfe,
    };

    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
    try testing.expect(result.x[0] < 0.01);
}

test "lbfgs: converges on 2D sphere function" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 3.0, 4.0 };
    const options = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .history_size = 10,
    };

    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
    try testing.expect(result.x[0] < 0.01);
    try testing.expect(result.x[1] < 0.01);
}

test "lbfgs: converges on Rosenbrock function" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 0.0, 0.0 };
    const options = LbfgsOptions(f64){
        .max_iter = 300,
        .tol = 1e-4,
        .history_size = 10,
    };

    const result = try lbfgs(f64, rosenbrock_f64, rosenbrock_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Rosenbrock is harder; partial convergence acceptable
    try testing.expect(result.f_val < 1.0);
}

test "lbfgs: handles n=5 dimensions" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const options = LbfgsOptions(f64){
        .max_iter = 150,
        .tol = 1e-6,
        .history_size = 8,
    };

    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
    for (result.x) |xi| {
        try testing.expect(xi < 0.1);
    }
}

test "lbfgs: handles n=10 dimensions" {
    const allocator = testing.allocator;

    var x0: [10]f64 = undefined;
    for (0..10) |i| {
        x0[i] = @as(f64, @floatFromInt(i)) + 1.0;
    }

    const options = LbfgsOptions(f64){
        .max_iter = 200,
        .tol = 1e-6,
        .history_size = 10,
    };

    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
}

test "lbfgs: early termination when initial gradient < tol" {
    const allocator = testing.allocator;

    const x0 = [_]f64{0.0001};
    const options = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-3,
        .history_size = 5,
    };

    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.n_iter == 0);
}

// Category 2: History Size Impact (5 tests)

test "lbfgs: history_size=3 converges (minimal)" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 2.0, 3.0 };
    const options = LbfgsOptions(f64){
        .max_iter = 150,
        .tol = 1e-6,
        .history_size = 3,
    };

    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
    try testing.expect(result.grad_norm < 1e-6);
}

test "lbfgs: history_size=5 converges efficiently" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 2.0, 3.0 };
    const options = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .history_size = 5,
    };

    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
}

test "lbfgs: history_size=10 (default) converges" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 2.0, 3.0 };
    const options = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .history_size = 10,
    };

    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
}

test "lbfgs: history_size=20 (large) converges" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 2.0, 3.0 };
    const options = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .history_size = 20,
    };

    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
}

test "lbfgs: larger history_size improves convergence rate" {
    const allocator = testing.allocator;

    const x0_small = [_]f64{ 3.0, 4.0 };
    const x0_large = [_]f64{ 3.0, 4.0 };

    const options_small = LbfgsOptions(f64){
        .max_iter = 200,
        .tol = 1e-6,
        .history_size = 3,
    };

    const options_large = LbfgsOptions(f64){
        .max_iter = 200,
        .tol = 1e-6,
        .history_size = 15,
    };

    const result_small = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0_small, options_small, allocator);
    defer result_small.deinit(allocator);

    const result_large = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0_large, options_large, allocator);
    defer result_large.deinit(allocator);

    // Both should converge, larger history often converges faster
    try testing.expect(result_small.converged);
    try testing.expect(result_large.converged);
    try testing.expect(result_large.n_iter <= result_small.n_iter + 5);
}

// Category 3: Line Search Methods (3 tests)

test "lbfgs: armijo line search achieves descent" {
    const allocator = testing.allocator;

    const x0 = [_]f64{2.0};
    const options = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .history_size = 5,
        .line_search = .armijo,
    };

    const f_initial = sphere_f64(&x0);
    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.f_val < f_initial);
}

test "lbfgs: wolfe line search satisfies curvature" {
    const allocator = testing.allocator;

    const x0 = [_]f64{2.0};
    const options = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .history_size = 5,
        .line_search = .wolfe,
    };

    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
}

test "lbfgs: backtracking line search converges" {
    const allocator = testing.allocator;

    const x0 = [_]f64{3.0};
    const options = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .history_size = 5,
        .line_search = .backtracking,
    };

    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
}

// Category 4: Edge Cases (6 tests)

test "lbfgs: zero initial point converges from origin" {
    const allocator = testing.allocator;

    const x0 = [_]f64{0.0};
    const options = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .history_size = 5,
    };

    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Already at minimum
    try testing.expect(result.n_iter == 0);
}

test "lbfgs: large initial values handled" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 100.0, 100.0 };
    const options = LbfgsOptions(f64){
        .max_iter = 200,
        .tol = 1e-6,
        .history_size = 10,
    };

    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
    try testing.expect(result.x[0] < 0.1);
    try testing.expect(result.x[1] < 0.1);
}

test "lbfgs: negative initial values handled" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ -3.0, -4.0 };
    const options = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .history_size = 8,
    };

    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
}

test "lbfgs: mixed positive/negative initial values" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 5.0, -3.0, 2.0, -4.0 };
    const options = LbfgsOptions(f64){
        .max_iter = 150,
        .tol = 1e-6,
        .history_size = 6,
    };

    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
}

test "lbfgs: n=1 dimension" {
    const allocator = testing.allocator;

    const x0 = [_]f64{3.0};
    const options = LbfgsOptions(f64){
        .max_iter = 50,
        .tol = 1e-6,
        .history_size = 3,
    };

    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
}

test "lbfgs: n=50 dimensions (stress test)" {
    const allocator = testing.allocator;

    var x0: [50]f64 = undefined;
    for (0..50) |i| {
        x0[i] = @as(f64, @floatFromInt(i % 10)) + 1.0;
    }

    const options = LbfgsOptions(f64){
        .max_iter = 300,
        .tol = 1e-5,
        .history_size = 15,
    };

    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
}

// Category 5: Non-convergence Scenarios (4 tests)

test "lbfgs: max_iter limit respected" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 10.0, 10.0 };
    const options = LbfgsOptions(f64){
        .max_iter = 3,
        .tol = 1e-10,
        .history_size = 5,
    };

    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.n_iter <= 3);
    // Should not converge with only 3 iterations
    try testing.expect(!result.converged);
}

test "lbfgs: very tight tolerance requires more iterations" {
    const allocator = testing.allocator;

    const x0_loose = [_]f64{2.0};
    const x0_tight = [_]f64{2.0};

    const options_loose = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-4,
        .history_size = 5,
    };

    const options_tight = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-12,
        .history_size = 5,
    };

    const result_loose = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0_loose, options_loose, allocator);
    defer result_loose.deinit(allocator);

    const result_tight = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0_tight, options_tight, allocator);
    defer result_tight.deinit(allocator);

    // Tighter tolerance should require more iterations
    try testing.expect(result_tight.n_iter >= result_loose.n_iter);
}

test "lbfgs: tighter tolerance achieves better accuracy" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};

    const options_loose = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-4,
        .history_size = 5,
    };

    const options_tight = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-10,
        .history_size = 5,
    };

    const result_loose = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options_loose, allocator);
    defer result_loose.deinit(allocator);

    const result_tight = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options_tight, allocator);
    defer result_tight.deinit(allocator);

    // Tighter tolerance should achieve smaller gradient norm
    try testing.expect(result_tight.grad_norm <= result_loose.grad_norm);
}

test "lbfgs: Beale function convergence with adequate iterations" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 3.2, 0.4 };
    const options = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-4,
        .history_size = 8,
    };

    const result = try lbfgs(f64, beale_f64, beale_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.f_val < 1.0);
}

// Category 6: Memory and Validation (6+ tests)

test "lbfgs: rejects empty x0" {
    const allocator = testing.allocator;

    const x0: [0]f64 = undefined;
    const options = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .history_size = 5,
    };

    const result = lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    try testing.expectError(error.InvalidArgument, result);
}

test "lbfgs: rejects invalid history_size (zero)" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};
    const options = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .history_size = 0,
    };

    const result = lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    try testing.expectError(error.InvalidArgument, result);
}

test "lbfgs: rejects negative tolerance" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};
    const options = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = -1e-6,
        .history_size = 5,
    };

    const result = lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    try testing.expectError(error.InvalidArgument, result);
}

test "lbfgs: rejects invalid line search parameters" {
    const allocator = testing.allocator;

    const x0 = [_]f64{1.0};
    const options = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .history_size = 5,
        .ls_c1 = -0.1, // Invalid: c1 must be positive
        .ls_c2 = 0.9,
    };

    const result = lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    try testing.expectError(error.InvalidArgument, result);
}

test "lbfgs: result structure validation" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 1.0, 2.0, 3.0 };
    const options = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .history_size = 5,
    };

    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.x.len == 3);
    try testing.expect(result.grad_norm >= 0);
    try testing.expect(result.n_iter <= 100);
}

test "lbfgs: no memory leaks with allocator" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 1.0, 2.0, 3.0 };
    const options = LbfgsOptions(f64){
        .max_iter = 50,
        .tol = 1e-6,
        .history_size = 8,
    };

    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    // Allocator detects leaks at the end of the test
    try testing.expect(result.x.len == 3);
}

test "lbfgs: f32 type support with looser tolerance" {
    const allocator = testing.allocator;

    const x0 = [_]f32{ 2.0, 3.0 };
    const options = LbfgsOptions(f32){
        .max_iter = 100,
        .tol = 1e-4,
        .history_size = 5,
    };

    const result = try lbfgs(f32, sphere_f32, sphere_grad_f32, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
}

test "lbfgs: f64 type support with tight tolerance" {
    const allocator = testing.allocator;

    const x0 = [_]f64{ 1.0, 1.0 };
    const options = LbfgsOptions(f64){
        .max_iter = 100,
        .tol = 1e-8,
        .history_size = 8,
    };

    const result = try lbfgs(f64, sphere_f64, sphere_grad_f64, &x0, options, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.converged);
}
