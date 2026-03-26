# Optimization API Reference

## Overview

The Optimization module provides algorithms for finding minima/maxima of functions, with support for unconstrained and constrained optimization, line search methods, least squares fitting, and automatic differentiation. All operations are designed for numerical stability and precision, supporting both f32 and f64 floating-point types.

### Import

```zig
const zuda = @import("zuda");
const optimize = zuda.optimize;
const line_search = optimize.line_search;
const unconstrained = optimize.unconstrained;
const constrained = optimize.constrained;
const least_squares = optimize.least_squares;
const autodiff = optimize.autodiff;
```

### Key Features

- **Line Search Methods**: Armijo, Wolfe, and backtracking for step size selection
- **Unconstrained Optimization**: Gradient descent, conjugate gradient, BFGS, L-BFGS, Nelder-Mead
- **Constrained Optimization**: Penalty method and augmented Lagrangian
- **Least Squares**: Gauss-Newton and Levenberg-Marquardt for nonlinear regression
- **Automatic Differentiation**: Forward-mode AD using dual numbers for exact gradients
- **Type Generic**: Support for f32 and f64 floating-point types
- **Numerical Stability**: Carefully designed algorithms with proper convergence checks

---

## Error Types

All optimize operations use consistent error types:

```zig
pub const LineSearchError = error{
    InvalidParameters,       // c1/c2/rho out of valid range
    NotDescentDirection,     // p·grad >= 0 (not descent)
    MaxIterationsExceeded,   // Failed to converge within max_iter
    AllocationFailed,        // Memory allocation error
    OutOfMemory,             // Allocator ran out of memory
};

pub const UnconstrainedError = error{
    InvalidParameters,
    OutOfMemory,
    MaxIterationsExceeded,
    NoConvergence,
};

pub const ConstrainedOptimizationError = error{
    InvalidParameters,
    OutOfMemory,
    MaxIterationsExceeded,
    Infeasible,
};

pub const LeastSquaresError = error{
    InvalidParameters,
    OutOfMemory,
    MaxIterationsExceeded,
    SingularMatrix,
};
```

---

## Line Search Methods

Line search finds a step size α along a descent direction p such that the objective function sufficiently decreases.

### armijo(T, f, x, p, grad, alpha_init, c1, max_iter, allocator)

Armijo line search: finds step size satisfying sufficient decrease condition.

```zig
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
) LineSearchError!ArmijoResult(T)
```

**Description**: Finds α such that f(x + α·p) ≤ f(x) + c₁·α·(grad·p). Uses geometric backtracking to reduce α until the condition is satisfied.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `f`: Objective function f(x) → scalar
- `x`: Current point (length n)
- `p`: Descent direction (must satisfy p·grad < 0)
- `grad`: Gradient at x (length n)
- `alpha_init`: Initial step size (e.g., 1.0)
- `c1`: Sufficient decrease constant (typical: 1e-4, valid: (0, 1))
- `max_iter`: Maximum backtracking iterations
- `allocator`: Memory allocator

**Returns**: `ArmijoResult(T)` with fields:
- `alpha: T` — Step size found
- `f_new: T` — Objective value at x + α·p
- `n_iter: usize` — Backtracking iterations performed
- `converged: bool` — Whether step was found

**Errors**:
- `error.InvalidParameters`: c1 ≤ 0 or c1 ≥ 1
- `error.NotDescentDirection`: p·grad ≥ 0 (not descent)
- `error.MaxIterationsExceeded`: No valid α found after max_iter attempts
- `error.OutOfMemory`: Allocator failure

**Time**: O(max_iter × f_eval)
**Space**: O(n) temporary for computing x_new

**Example**:
```zig
var x = [_]f64{2.0};
var grad = [_]f64{0};
computeGradient(&x, &grad);

var p = [_]f64{-grad[0]};  // Steepest descent direction
const result = try line_search.armijo(f64, objective, &x, &p, &grad, 1.0, 1e-4, 20, allocator);

std.debug.print("Step size: {d}\n", .{result.alpha});
```

---

### wolfe(T, f, grad_f, x, p, alpha_init, c1, c2, max_iter, allocator)

Wolfe line search: finds step size satisfying both Armijo and strong curvature conditions.

```zig
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
) LineSearchError!WolfeResult(T)
```

**Description**: Finds α satisfying:
- Armijo: f(x+α·p) ≤ f(x) + c₁·α·(grad·p)
- Strong curvature: |grad(x+α·p)·p| ≤ c₂·|grad·p|

Suitable for quasi-Newton methods (BFGS, L-BFGS) and conjugate gradient.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `f`: Objective function
- `grad_f`: Gradient function (required for curvature check)
- `x`: Current point (length n)
- `p`: Descent direction
- `alpha_init`: Initial step size
- `c1`: Armijo constant (typical: 1e-4, valid: (0, c2))
- `c2`: Curvature constant (typical: 0.9, valid: (c1, 1))
- `max_iter`: Maximum iterations
- `allocator`: Memory allocator

**Returns**: `WolfeResult(T)` with fields:
- `alpha: T` — Step size found
- `f_new: T` — Objective value at x + α·p
- `grad_new: []T` — Gradient at x + α·p (caller must free)
- `n_iter: usize` — Iterations performed
- `converged: bool` — Whether both conditions satisfied

**Errors**:
- `error.InvalidParameters`: c1, c2 out of valid range
- `error.NotDescentDirection`: p·grad ≥ 0
- `error.MaxIterationsExceeded`: Failed to find valid α
- `error.OutOfMemory`: Allocator failure

**Time**: O(max_iter × (f_eval + grad_eval))
**Space**: O(n) for gradient storage

**Example**:
```zig
const result = try line_search.wolfe(f64, objective, gradient, &x, &p, 1.0, 1e-4, 0.9, 20, allocator);
defer allocator.free(result.grad_new);

std.debug.print("New gradient norm: {d}\n", .{vectorNorm(result.grad_new)});
```

---

### backtracking(T, f, x, p, grad, alpha_init, rho, c, max_iter, allocator)

Backtracking line search: simple geometric reduction until Armijo condition satisfied.

```zig
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
) LineSearchError!BacktrackResult(T)
```

**Description**: Repeatedly reduces step size by factor ρ until Armijo condition satisfied: α_new = ρ·α_old. Simple, robust, and suitable for gradient descent.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `f`: Objective function
- `x`: Current point (length n)
- `p`: Descent direction
- `grad`: Gradient at x (length n)
- `alpha_init`: Initial step size
- `rho`: Reduction factor (typical: 0.5, valid: (0, 1))
- `c`: Armijo constant (typical: 1e-4, valid: (0, 1))
- `max_iter`: Maximum reduction steps
- `allocator`: Memory allocator

**Returns**: `BacktrackResult(T)` with fields:
- `alpha: T` — Step size found
- `f_new: T` — Objective value at x + α·p
- `n_iter: usize` — Reduction steps performed
- `converged: bool` — Whether step was found

**Errors**:
- `error.InvalidParameters`: rho or c out of valid range
- `error.NotDescentDirection`: p·grad ≥ 0
- `error.MaxIterationsExceeded`: Failed to converge
- `error.OutOfMemory`: Allocator failure

**Time**: O(max_iter × f_eval)
**Space**: O(n)

**Example**:
```zig
const result = try line_search.backtracking(f64, objective, &x, &p, &grad, 1.0, 0.5, 1e-4, 20, allocator);
```

---

## Unconstrained Optimization

Minimize f(x) without constraints using gradient-based and derivative-free methods.

### gradient_descent(T, f, grad_f, x0, options, allocator)

Steepest descent with fixed or adaptive learning rate.

```zig
pub fn gradient_descent(
    comptime T: type,
    f: ObjectiveFn(T),
    grad_f: GradientFn(T),
    x0: []const T,
    options: GradientDescentOptions(T),
    allocator: std.mem.Allocator,
) UnconstrainedError!UnconstrainedResult(T)
```

**Description**: Iterates x_{k+1} = x_k - α_k·∇f(x_k) where α_k is determined by line search or fixed learning rate.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `f`: Objective function
- `grad_f`: Gradient function
- `x0`: Initial point (length n, caller keeps ownership)
- `options`: Algorithm options (see `GradientDescentOptions(T)`)
- `allocator`: Memory allocator

**Options** (`GradientDescentOptions(T)`):
```zig
struct {
    max_iter: usize = 1000,
    tol: T = 1e-6,              // Convergence tolerance
    learning_rate: T = 0.01,    // Step size (fixed)
    use_line_search: bool = false,
}
```

**Returns**: `UnconstrainedResult(T)` with fields:
- `x: []T` — Optimized point (caller must free)
- `f_val: T` — Final objective value
- `grad: []T` — Gradient at solution (caller must free)
- `n_iter: usize` — Iterations performed
- `converged: bool` — Convergence achieved

**Errors**:
- `error.InvalidParameters`: Invalid options
- `error.OutOfMemory`: Allocator failure
- `error.MaxIterationsExceeded`: No convergence

**Time**: O(n_iter × n × f_eval)
**Space**: O(n) for gradient and intermediate vectors

**Convergence**: Linear (slow)

**Example**:
```zig
const x0 = [_]f64{-1.0, 2.0};
const options = optimize.GradientDescentOptions(f64){
    .max_iter = 1000,
    .tol = 1e-6,
    .learning_rate = 0.01,
};

var result = try unconstrained.gradient_descent(f64, rosenbrock, rosenbrock_grad, &x0, options, allocator);
defer allocator.free(result.x);
defer allocator.free(result.grad);

std.debug.print("Converged: {}\n", .{result.converged});
```

---

### conjugate_gradient(T, f, grad_f, x0, options, allocator)

Conjugate Gradient method for convex quadratic and nonquadratic problems.

```zig
pub fn conjugate_gradient(
    comptime T: type,
    f: ObjectiveFn(T),
    grad_f: GradientFn(T),
    x0: []const T,
    options: ConjugateGradientOptions(T),
    allocator: std.mem.Allocator,
) UnconstrainedError!UnconstrainedResult(T)
```

**Description**: Constructs orthogonal descent directions using Polak-Ribière update. Converges in n steps for quadratic functions. More efficient than gradient descent for convex problems.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `f`: Objective function
- `grad_f`: Gradient function
- `x0`: Initial point
- `options`: Algorithm options
- `allocator`: Memory allocator

**Options** (`ConjugateGradientOptions(T)`):
```zig
struct {
    max_iter: usize = 1000,
    tol: T = 1e-6,
    line_search_tol: T = 1e-4,
}
```

**Returns**: `UnconstrainedResult(T)` (same as gradient descent)

**Errors**:
- `error.InvalidParameters`: Invalid options
- `error.OutOfMemory`: Allocator failure
- `error.MaxIterationsExceeded`: No convergence

**Time**: O(n) for quadratic; O(n_iter × n × f_eval) for general
**Space**: O(n) for direction and gradient vectors

**Convergence**: Superlinear for quadratic, faster than gradient descent for general convex problems

**Example**:
```zig
const options = optimize.ConjugateGradientOptions(f64){
    .max_iter = 1000,
    .tol = 1e-6,
};

var result = try unconstrained.conjugate_gradient(f64, objective, gradient, &x0, options, allocator);
```

---

### bfgs(T, f, grad_f, x0, options, allocator)

BFGS quasi-Newton method: approximates Hessian without explicit computation.

```zig
pub fn bfgs(
    comptime T: type,
    f: ObjectiveFn(T),
    grad_f: GradientFn(T),
    x0: []const T,
    options: BFGSOptions(T),
    allocator: std.mem.Allocator,
) UnconstrainedError!UnconstrainedResult(T)
```

**Description**: Maintains n×n approximation of inverse Hessian, updated via rank-2 formula after each step. Fast convergence without computing true Hessian.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `f`: Objective function
- `grad_f`: Gradient function
- `x0`: Initial point
- `options`: Algorithm options
- `allocator`: Memory allocator

**Options** (`BFGSOptions(T)`):
```zig
struct {
    max_iter: usize = 1000,
    tol: T = 1e-6,
    line_search_tol: T = 1e-4,
}
```

**Returns**: `UnconstrainedResult(T)`

**Errors**:
- `error.InvalidParameters`: Invalid options
- `error.OutOfMemory`: Allocator failure
- `error.MaxIterationsExceeded`: No convergence

**Time**: O(n² + n_iter × n² × f_eval) (O(n²) for Hessian updates)
**Space**: O(n²) for Hessian approximation

**Convergence**: Superlinear

**Use Cases**: General nonlinear optimization, medium-scale problems (n < 1000)

**Example**:
```zig
const options = optimize.BFGSOptions(f64){
    .max_iter = 1000,
    .tol = 1e-6,
};

var result = try unconstrained.bfgs(f64, objective, gradient, &x0, options, allocator);
defer allocator.free(result.x);
```

---

### lbfgs(T, f, grad_f, x0, options, allocator)

Limited-memory BFGS: memory-efficient variant of BFGS for high-dimensional problems.

```zig
pub fn lbfgs(
    comptime T: type,
    f: ObjectiveFn(T),
    grad_f: GradientFn(T),
    x0: []const T,
    options: LBFGSOptions(T),
    allocator: std.mem.Allocator,
) UnconstrainedError!UnconstrainedResult(T)
```

**Description**: Stores only m recent (step, gradient) pairs instead of full n×n Hessian. Two-loop recursion reconstructs Hessian-vector product. Suitable for large-scale optimization.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `f`: Objective function
- `grad_f`: Gradient function
- `x0`: Initial point
- `options`: Algorithm options
- `allocator`: Memory allocator

**Options** (`LBFGSOptions(T)`):
```zig
struct {
    max_iter: usize = 1000,
    tol: T = 1e-6,
    line_search_tol: T = 1e-4,
    m: usize = 10,  // Memory: number of correction pairs
}
```

**Returns**: `UnconstrainedResult(T)`

**Errors**:
- `error.InvalidParameters`: Invalid options (m ≤ 0, etc.)
- `error.OutOfMemory`: Allocator failure
- `error.MaxIterationsExceeded`: No convergence

**Time**: O(m·n + n_iter × (m·n) × f_eval) (vs O(n²) for full BFGS)
**Space**: O(m·n) (vs O(n²) for full BFGS)

**Convergence**: Superlinear

**Use Cases**: Large-scale optimization (n > 1000), machine learning, high-dimensional problems

**Example**:
```zig
const options = optimize.LBFGSOptions(f64){
    .max_iter = 1000,
    .tol = 1e-6,
    .m = 10,  // Store last 10 correction pairs
};

var result = try unconstrained.lbfgs(f64, objective, gradient, &x0, options, allocator);
defer allocator.free(result.x);
defer allocator.free(result.grad);
```

---

### nelder_mead(T, f, x0, options, allocator)

Nelder-Mead (Simplex) method: derivative-free optimization for noisy/discontinuous functions.

```zig
pub fn nelder_mead(
    comptime T: type,
    f: ObjectiveFn(T),
    x0: []const T,
    options: NelderMeadOptions(T),
    allocator: std.mem.Allocator,
) UnconstrainedError!UnconstrainedResult(T)
```

**Description**: Maintains simplex of n+1 points, performs reflection/expansion/contraction to move toward optimum. No gradient required. Works for noisy objectives but slower than gradient-based methods.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `f`: Objective function
- `x0`: Initial point (simplex formed around x0)
- `options`: Algorithm options
- `allocator`: Memory allocator

**Options** (`NelderMeadOptions(T)`):
```zig
struct {
    max_iter: usize = 1000,
    tol: T = 1e-6,
    alpha: T = 1.0,    // Reflection coefficient
    beta: T = 0.5,     // Contraction coefficient
    gamma: T = 2.0,    // Expansion coefficient
}
```

**Returns**: `UnconstrainedResult(T)` (note: `grad` field may be uninitialized)

**Errors**:
- `error.InvalidParameters`: Invalid options
- `error.OutOfMemory`: Allocator failure
- `error.MaxIterationsExceeded`: No convergence

**Time**: O(n_iter × n × f_eval) (function evaluations dominate)
**Space**: O(n²) for simplex points

**Convergence**: Slow (sublinear); no theoretical guarantees

**Use Cases**: Noisy functions, discontinuous objectives, small problems (n < 100)

**Example**:
```zig
const options = optimize.NelderMeadOptions(f64){
    .max_iter = 1000,
    .tol = 1e-6,
};

var result = try unconstrained.nelder_mead(f64, noisy_objective, &x0, options, allocator);
defer allocator.free(result.x);
```

---

## Constrained Optimization

Minimize f(x) subject to constraints: g_i(x) ≤ 0 (inequality), h_j(x) = 0 (equality).

### penalty_method(T, f, grad_f, x0, ineq_constraints, eq_constraints, options, allocator)

Penalty method: converts constrained problem to sequence of unconstrained subproblems.

```zig
pub fn penalty_method(
    comptime T: type,
    f: ObjectiveFn(T),
    grad_f: GradientFn(T),
    x0: []const T,
    inequality_constraints: []const Constraint(T),
    equality_constraints: []const Constraint(T),
    options: PenaltyMethodOptions(T),
    allocator: std.mem.Allocator,
) ConstrainedOptimizationError!OptimizationResult(T)
```

**Description**: Solves sequence of unconstrained problems with augmented objective:
P(x, μ) = f(x) + μ·[Σ max(0,g_i(x))² + Σ h_j(x)²]

Outer loop increases penalty parameter μ, inner loop solves unconstrained subproblem.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `f`: Objective function
- `grad_f`: Gradient of f
- `x0`: Initial point
- `inequality_constraints`: Array of g_i constraints (can be empty)
- `equality_constraints`: Array of h_j constraints (can be empty)
- `options`: Algorithm options
- `allocator`: Memory allocator

**Constraint Type** (`Constraint(T)`):
```zig
struct {
    func: *const fn (x: []const T) T,
    grad: *const fn (x: []const T, out_grad: []T) void,
}
```

**Options** (`PenaltyMethodOptions(T)`):
```zig
struct {
    max_outer_iter: usize = 10,
    max_inner_iter: usize = 100,
    penalty_init: T = 1.0,      // Initial penalty parameter μ
    penalty_scale: T = 10.0,    // μ *= penalty_scale each outer iteration
    tol: T = 1e-6,              // Constraint violation tolerance
    inner_solver: InnerSolver = .lbfgs,  // gradient_descent, bfgs, or lbfgs
}
```

**Returns**: `OptimizationResult(T)` with fields:
- `x: []T` — Optimal point (caller must free)
- `f_val: T` — Final objective value
- `constraint_violation: T` — Maximum constraint violation
- `n_outer_iter: usize` — Outer iterations performed
- `n_inner_iter: usize` — Total inner iterations
- `converged: bool` — Convergence achieved

**Errors**:
- `error.InvalidParameters`: Invalid options
- `error.OutOfMemory`: Allocator failure
- `error.MaxIterationsExceeded`: No convergence
- `error.Infeasible`: No feasible solution found

**Time**: O(outer_iter × inner_iter × n)
**Space**: O(n) for gradients and state

**Example**:
```zig
// Minimize x₁² + x₂² subject to x₁ + x₂ = 1
fn constraint_eq(x: []const f64) f64 {
    return x[0] + x[1] - 1.0;
}

fn constraint_eq_grad(x: []const f64, grad: []f64) void {
    grad[0] = 1.0;
    grad[1] = 1.0;
}

const constraint = optimize.Constraint(f64){
    .func = constraint_eq,
    .grad = constraint_eq_grad,
};

var constraints_eq = [_]optimize.Constraint(f64){constraint};
var constraints_ineq = [_]optimize.Constraint(f64){};  // Empty

const options = optimize.PenaltyMethodOptions(f64){
    .max_outer_iter = 10,
    .max_inner_iter = 100,
    .penalty_init = 1.0,
    .penalty_scale = 10.0,
    .tol = 1e-6,
};

var result = try constrained.penalty_method(f64, objective, objective_grad, &x0, &constraints_ineq, &constraints_eq, options, allocator);
defer result.deinit(allocator);
```

---

## Least Squares Optimization

Minimize ‖f(x)‖² = Σ fᵢ(x)² where f: ℝⁿ → ℝᵐ (curve fitting, parameter estimation).

### levenberg_marquardt(T, alloc, residual_fns, jacobian_fn, x0, options)

Levenberg-Marquardt algorithm: robust damped Gauss-Newton for nonlinear least squares.

```zig
pub fn levenberg_marquardt(
    comptime T: type,
    alloc: std.mem.Allocator,
    residual_fns: []const ResidualFn(T),
    jacobian_fn: ?JacobianFn(T),
    x0: []const T,
    options: LevenbergMarquardtOptions(T),
) LeastSquaresError!LeastSquaresResult(T)
```

**Description**: Solves (J^T·J + λI)·δx = -J^T·r at each iteration with adaptive damping λ. Combines Gauss-Newton (λ→0) with gradient descent (λ→∞) for robustness. Increases λ on rejection, decreases on acceptance.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `alloc`: Memory allocator
- `residual_fns`: Array of m residual functions rᵢ(x)
- `jacobian_fn`: Optional analytical Jacobian (uses finite differences if null)
- `x0`: Initial parameter guess (length n)
- `options`: Algorithm options

**Residual/Jacobian Types**:
```zig
pub fn ResidualFn(comptime T: type) type {
    return fn (x: []const T) T;  // rᵢ(x) → scalar
}

pub fn JacobianFn(comptime T: type) type {
    return fn (x: []const T, out_jacobian: []T) void;  // J[i,j] = ∂rᵢ/∂xⱼ
}
```

**Options** (`LevenbergMarquardtOptions(T)`):
```zig
struct {
    max_iter: usize = 100,
    tol_f: T = 1e-8,        // Objective change tolerance
    tol_x: T = 1e-8,        // Parameter change tolerance
    tol_grad: T = 1e-8,     // Gradient norm tolerance
    lambda_init: T = 1e-3,  // Initial damping parameter
    lambda_min: T = 1e-12,  // Minimum lambda
    lambda_max: T = 1e12,   // Maximum lambda
    lambda_scale_up: T = 10.0,   // Increase λ on rejection
    lambda_scale_down: T = 0.1,  // Decrease λ on acceptance
    epsilon: T = 1e-8,      // Finite difference step size
}
```

**Returns**: `LeastSquaresResult(T)` with fields:
- `x: []T` — Optimized parameters (caller must free)
- `residuals: []T` — Final residuals (caller must free)
- `f_val: T` — Final objective value: 0.5·‖r‖²
- `n_iter: usize` — Iterations performed
- `converged: bool` — Convergence achieved
- `termination_reason: []const u8` — Human-readable reason

**Errors**:
- `error.InvalidParameters`: Invalid options
- `error.OutOfMemory`: Allocator failure
- `error.MaxIterationsExceeded`: No convergence
- `error.SingularMatrix`: Singular normal equations

**Time**: O(iter × (m·n² + n³)) for LU solve in normal equations
**Space**: O(m·n) for Jacobian

**Convergence**: Fast near solution, robust far from solution

**Example**:
```zig
// Fit exponential: y = a·exp(b·x)
fn residual_exp(params: []const f64) f64 {
    // Compute one residual for current data point (simplified)
    const a = params[0];
    const b = params[1];
    const x = 1.0;  // Example data point
    const y_true = 2.7;
    const y_pred = a * @exp(b * x);
    return y_pred - y_true;
}

fn jacobian_exp(params: []const f64, out_J: []f64) void {
    const a = params[0];
    const b = params[1];
    const x = 1.0;
    const exp_bx = @exp(b * x);
    out_J[0] = exp_bx;           // ∂r/∂a
    out_J[1] = a * x * exp_bx;   // ∂r/∂b
}

var residuals = [_]optimize.ResidualFn(f64){residual_exp};
const options = optimize.LevenbergMarquardtOptions(f64){
    .max_iter = 100,
    .tol_f = 1e-8,
    .lambda_init = 1e-3,
};

var params = [_]f64{1.0, 1.0};  // Initial guess
var result = try least_squares.levenberg_marquardt(f64, allocator, &residuals, jacobian_exp, &params, options);
defer result.deinit(allocator);
```

---

### gauss_newton(T, alloc, residual_fns, jacobian_fn, x0, options)

Gauss-Newton algorithm: second-order method for nonlinear least squares.

```zig
pub fn gauss_newton(
    comptime T: type,
    alloc: std.mem.Allocator,
    residual_fns: []const ResidualFn(T),
    jacobian_fn: ?JacobianFn(T),
    x0: []const T,
    options: GaussNewtonOptions(T),
) LeastSquaresError!LeastSquaresResult(T)
```

**Description**: Solves J^T·J·δx = -J^T·r at each iteration. Faster than Levenberg-Marquardt near solution but can diverge if initial guess is poor or residuals are large. Approximates Hessian as H ≈ J^T·J (assumes small residuals).

**Parameters**:
- `T`: Numeric type (f32, f64)
- `alloc`: Memory allocator
- `residual_fns`: Array of m residual functions
- `jacobian_fn`: Optional analytical Jacobian (uses finite differences if null)
- `x0`: Initial parameter guess
- `options`: Algorithm options

**Options** (`GaussNewtonOptions(T)`):
```zig
struct {
    max_iter: usize = 100,
    tol_f: T = 1e-8,
    tol_x: T = 1e-8,
    tol_grad: T = 1e-8,
    epsilon: T = 1e-8,  // Finite difference step size
}
```

**Returns**: `LeastSquaresResult(T)` (same as Levenberg-Marquardt)

**Errors**:
- `error.InvalidParameters`: Invalid options
- `error.OutOfMemory`: Allocator failure
- `error.MaxIterationsExceeded`: No convergence
- `error.SingularMatrix`: Singular normal equations

**Time**: O(iter × (m·n² + n³))
**Space**: O(m·n) for Jacobian

**Convergence**: Fast near solution; Q-quadratic

**Use Cases**: Well-conditioned problems with good initial guess

**Example**:
```zig
var residuals = [_]optimize.ResidualFn(f64){residual_exp};
const options = optimize.GaussNewtonOptions(f64){
    .max_iter = 100,
    .tol_f = 1e-8,
};

var result = try least_squares.gauss_newton(f64, allocator, &residuals, jacobian_exp, &params, options);
defer result.deinit(allocator);
```

---

## Automatic Differentiation

Forward-mode AD using dual numbers for exact gradient and Jacobian computation.

### Dual(T) Type

Dual number: value + derivative pair for exact differentiation.

```zig
pub fn Dual(comptime T: type) type {
    // Returns struct with:
    // - value: T (real part)
    // - derivative: T (dual part)
    // - Overloaded arithmetic: add, sub, mul, div, etc.
}
```

**Description**: Represents x + εx' where ε² = 0. All arithmetic operations automatically apply chain rule. Enables exact derivative computation with no numerical approximation error.

**Supported Operations**:
```zig
pub fn constant(v: T) Dual(T)       // Constant: derivative = 0
pub fn variable(v: T) Dual(T)       // Variable: derivative = 1
pub fn init(v: T, d: T) Dual(T)     // Explicit (value, derivative)

// Arithmetic (chain rule automatic)
pub fn add(a: Dual(T), b: Dual(T)) Dual(T)
pub fn sub(a: Dual(T), b: Dual(T)) Dual(T)
pub fn mul(a: Dual(T), b: Dual(T)) Dual(T)
pub fn div(a: Dual(T), b: Dual(T)) Dual(T)
pub fn neg(a: Dual(T)) Dual(T)
pub fn scale(a: Dual(T), k: T) Dual(T)

// Elementary functions (with derivatives)
pub fn square(a: Dual(T)) Dual(T)
pub fn pow(a: Dual(T), n: T) Dual(T)
pub fn sqrt(a: Dual(T)) Dual(T)
pub fn exp(a: Dual(T)) Dual(T)
pub fn log(a: Dual(T)) Dual(T)
pub fn sin(a: Dual(T)) Dual(T)
pub fn cos(a: Dual(T)) Dual(T)
pub fn tan(a: Dual(T)) Dual(T)
pub fn abs(a: Dual(T)) Dual(T)
```

**Example**:
```zig
const D = optimize.autodiff.Dual(f64);

// Compute f(x) = x² + 2x and f'(x) at x = 3
const x = D.variable(3.0);
const f = x.square().add(x.scale(2.0));

std.debug.print("f(3) = {d}, f'(3) = {d}\n", .{f.value, f.derivative});
// Output: f(3) = 15, f'(3) = 8
```

---

### gradient(T, f, x, allocator)

Compute gradient of f: ℝⁿ → ℝ via forward-mode AD.

```zig
pub fn gradient(
    comptime T: type,
    f: *const fn ([]const Dual(T)) Dual(T),
    x: []const T,
    allocator: std.mem.Allocator,
) ![]T
```

**Description**: Uses forward-mode AD to compute ∇f(x) = [∂f/∂x₁, ..., ∂f/∂xₙ]. Evaluates f(x + εeᵢ) for each basis vector eᵢ to extract derivative component.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `f`: Function taking dual array, returning dual scalar
- `x`: Point at which to evaluate gradient (length n)
- `allocator`: Memory allocator

**Returns**: Gradient array (caller must free)

**Errors**:
- `error.OutOfMemory`: Allocator failure

**Time**: O(n × cost(f)) — n function evaluations
**Space**: O(n) for gradient array

**Example**:
```zig
const D = optimize.autodiff.Dual(f64);

const QuadraticFn = struct {
    fn eval(x: []const D) D {
        return x[0].square().add(x[1].square());
    }
};

const x = [_]f64{2.0, 3.0};
const grad = try optimize.autodiff.gradient(f64, QuadraticFn.eval, &x, allocator);
defer allocator.free(grad);

// grad[0] = 4, grad[1] = 6 (∇f = [2x₁, 2x₂] at (2,3))
std.debug.print("∇f = [{d}, {d}]\n", .{grad[0], grad[1]});
```

---

### jacobian(T, f, x, m, allocator)

Compute Jacobian J[i,j] = ∂fᵢ/∂xⱼ of f: ℝⁿ → ℝᵐ.

```zig
pub fn jacobian(
    comptime T: type,
    f: *const fn ([]const Dual(T), []Dual(T)) void,
    x: []const T,
    m: usize,
    allocator: std.mem.Allocator,
) ![]T
```

**Description**: Computes m×n Jacobian matrix via forward-mode AD. One forward pass per input dimension (n passes total). Returns Jacobian in row-major order.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `f`: Function f(dual_x, dual_y) → void computing y = f(x)
- `x`: Input point (length n)
- `m`: Output dimension
- `allocator`: Memory allocator

**Returns**: Jacobian matrix (m rows × n cols, row-major, caller must free)

**Errors**:
- `error.OutOfMemory`: Allocator failure

**Time**: O(n × m × cost(f)) — n evaluations
**Space**: O(m × n) for Jacobian

**Example**:
```zig
const D = optimize.autodiff.Dual(f64);

const VectorFn = struct {
    fn eval(x: []const D, y: []D) void {
        // f₁(x,y) = x²
        // f₂(x,y) = xy
        y[0] = x[0].square();
        y[1] = x[0].mul(x[1]);
    }
};

const x = [_]f64{2.0, 3.0};
const J = try optimize.autodiff.jacobian(f64, VectorFn.eval, &x, 2, allocator);
defer allocator.free(J);

// J = [[4, 0], [3, 2]] (row-major)
// J[i*n + j] for row i, col j
std.debug.print("J[0,0] = {d}\n", .{J[0]});  // ∂f₁/∂x = 2x = 4
```

---

## Practical Examples

### Curve Fitting with Levenberg-Marquardt

```zig
const std = @import("std");
const zuda = @import("zuda");
const optimize = zuda.optimize;
const least_squares = optimize.least_squares;

// Data points: (x, y)
const x_data = [_]f64{0.0, 1.0, 2.0, 3.0};
const y_data = [_]f64{1.0, 2.7, 7.4, 20.1};
const n_data = x_data.len;

// Residual: r_i = y_pred(x_i) - y_i for exponential model y = a*exp(b*x)
fn residual(params: []const f64) f64 {
    // Note: This simplified version computes one residual
    // Full implementation needs residual array
    const a = params[0];
    const b = params[1];
    const x = x_data[0];  // First point
    const y_true = y_data[0];
    const y_pred = a * @exp(b * x);
    return y_pred - y_true;
}

fn jacobian(params: []const f64, out_J: []f64) void {
    const a = params[0];
    const b = params[1];
    const x = x_data[0];
    const exp_bx = @exp(b * x);

    out_J[0] = exp_bx;           // ∂r/∂a
    out_J[1] = a * x * exp_bx;   // ∂r/∂b
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var params0 = [_]f64{1.0, 1.0};

    const options = optimize.LevenbergMarquardtOptions(f64){
        .max_iter = 100,
        .tol_f = 1e-8,
        .lambda_init = 1e-3,
    };

    var residuals = [_]optimize.least_squares.ResidualFn(f64){residual};
    var result = try least_squares.levenberg_marquardt(f64, allocator, &residuals, jacobian, &params0, options);
    defer result.deinit(allocator);

    std.debug.print("a = {d:.6}, b = {d:.6}\n", .{result.x[0], result.x[1]});
    std.debug.print("Objective: {e}\n", .{result.f_val});
}
```

---

### Constrained Optimization with Penalty Method

```zig
// Minimize x₁² + x₂² subject to x₁ + x₂ = 1
fn objective(x: []const f64) f64 {
    return x[0] * x[0] + x[1] * x[1];
}

fn objective_grad(x: []const f64, grad: []f64) void {
    grad[0] = 2.0 * x[0];
    grad[1] = 2.0 * x[1];
}

fn constraint_eq(x: []const f64) f64 {
    return x[0] + x[1] - 1.0;
}

fn constraint_eq_grad(x: []const f64, grad: []f64) void {
    grad[0] = 1.0;
    grad[1] = 1.0;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var x0 = [_]f64{0.0, 0.0};

    const eq_constraint = optimize.Constraint(f64){
        .func = constraint_eq,
        .grad = constraint_eq_grad,
    };

    var eq_constraints = [_]optimize.Constraint(f64){eq_constraint};
    var ineq_constraints = [_]optimize.Constraint(f64){};

    const options = optimize.PenaltyMethodOptions(f64){
        .max_outer_iter = 10,
        .max_inner_iter = 100,
        .penalty_init = 1.0,
        .tol = 1e-6,
    };

    var result = try optimize.constrained.penalty_method(
        f64, objective, objective_grad, &x0, &ineq_constraints, &eq_constraints, options, allocator);
    defer result.deinit(allocator);

    std.debug.print("Optimal: x = [{d:.6}, {d:.6}]\n", .{result.x[0], result.x[1]});
    std.debug.print("Constraint violation: {e}\n", .{result.constraint_violation});
}
```

---

### Gradient Computation with Autodiff

```zig
// Compute gradient of Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
const D = optimize.autodiff.Dual(f64);

const Rosenbrock = struct {
    fn eval(x: []const D) D {
        const one = D.constant(1.0);
        const hundred = D.constant(100.0);

        const term1 = one.sub(x[0]).square();
        const term2 = x[1].sub(x[0].square()).square().mul(hundred);
        return term1.add(term2);
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const x = [_]f64{0.0, 0.0};
    const grad = try optimize.autodiff.gradient(f64, Rosenbrock.eval, &x, allocator);
    defer allocator.free(grad);

    std.debug.print("∇f(0,0) = [{d}, {d}]\n", .{grad[0], grad[1]});
    // At (0,0): ∇f = [-2, 0]
}
```

---

## Performance Considerations

1. **Method Selection**:
   - Smooth, unconstrained → BFGS (medium scale) or L-BFGS (large scale)
   - Quadratic → Conjugate Gradient
   - Least squares → Levenberg-Marquardt (robust) or Gauss-Newton (fast near solution)
   - Noisy/discontinuous → Nelder-Mead

2. **Gradient Computation**:
   - Analytical gradients: use line search + quasi-Newton for fast convergence
   - Autodiff gradients: exact but slower than analytical (use for prototyping)
   - Finite differences: avoid if possible (slow and inaccurate)

3. **Line Search**:
   - Wolfe: for quasi-Newton methods (more expensive but better steps)
   - Backtracking: for gradient descent (simpler and sufficient)

4. **Large-Scale Problems**:
   - Use L-BFGS instead of BFGS (O(mn) vs O(n²) memory)
   - Provide analytical Jacobian for least squares (vs finite differences)
   - Consider stochastic methods if batch processing available

5. **Scaling**: Normalize variables to similar magnitudes for better numerical behavior

---

## Convergence Diagnostics

```zig
// Check convergence indicators
if (result.converged) {
    std.debug.print("Converged in {} iterations\n", .{result.n_iter});
} else {
    std.debug.print("Warning: max iterations reached\n", .{});
}

// For unconstrained: check gradient norm
const grad_norm = vectorNorm(result.grad);
if (grad_norm > 1e-6) {
    std.debug.print("Warning: gradient norm = {e} (may not be at minimum)\n", .{grad_norm});
}

// For constrained: check constraint violations
if (result.constraint_violation > 1e-4) {
    std.debug.print("Warning: constraint violated by {e}\n", .{result.constraint_violation});
}
```

---

## Numerical Stability Notes

- **Ill-Conditioned Problems**: Use L-BFGS or constraint scaling
- **Oscillation**: Reduce learning rate or increase line search iterations
- **Divergence**: Start with better initial guess or use damping (L-M algorithm)
- **Singular Jacobian**: Check Jacobian rank, use pseudo-inverse if needed
- **Stagnation**: Verify gradient computation (finite diff error?) or try different solver

---

## References

- Nocedal & Wright. *Numerical Optimization* (2nd ed.). Springer.
- BLAS/LAPACK: https://www.netlib.org/
- Automatic Differentiation: https://en.wikipedia.org/wiki/Automatic_differentiation
- Levenberg-Marquardt: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
