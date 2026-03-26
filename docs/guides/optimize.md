# Optimization — Nonlinear Optimization and Least Squares

## Overview

The `optimize` module provides algorithms for finding minima/maxima of functions, with support for constrained and unconstrained optimization, linear programming, and nonlinear least squares. It's designed for machine learning, parameter estimation, and engineering optimization.

## Module Structure

```zig
const optimize = zuda.optimize;

// Line search methods
const line_search = optimize.line_search;

// Unconstrained optimization
const unconstrained = optimize.unconstrained;

// Constrained optimization
const constrained = optimize.constrained;

// Least squares
const least_squares = optimize.least_squares;

// Linear programming
const linear = optimize.linear;
```

## Line Search Methods

Find step size α that sufficiently decreases objective along direction.

### Backtracking Line Search

Simple and robust method.

```zig
fn rosenbrock(x: []const f64) f64 {
    const a = 1.0 - x[0];
    const b = x[1] - x[0] * x[0];
    return a * a + 100.0 * b * b;
}

fn rosenbrock_grad(x: []const f64, grad: []f64, allocator: Allocator) !void {
    // ∂f/∂x = -2(1-x) - 400x(y-x²)
    // ∂f/∂y = 200(y-x²)
    grad[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
    grad[1] = 200.0 * (x[1] - x[0] * x[0]);
}

const x = [_]f64{0.0, 0.0};
const direction = [_]f64{1.0, 1.0};  // Search direction

var grad = try allocator.alloc(f64, 2);
defer allocator.free(grad);
try rosenbrock_grad(&x, grad, allocator);

const alpha = try line_search.backtracking(f64, rosenbrock, rosenbrock_grad, &x, &direction, 1.0, allocator);
// Returns step size satisfying Armijo condition
```

**Use cases**: General-purpose, gradient descent, Newton's method

### Strong Wolfe Line Search

More sophisticated, ensures curvature condition.

```zig
const alpha = try line_search.strong_wolfe(f64, rosenbrock, rosenbrock_grad, &x, &direction, 1.0, allocator);
// Satisfies both Armijo and curvature conditions
```

**Use cases**: Quasi-Newton methods (BFGS, L-BFGS), conjugate gradient

## Unconstrained Optimization

Minimize f(x) without constraints.

### Gradient Descent

Simplest first-order method.

```zig
const x0 = [_]f64{-1.0, 2.0};  // Initial guess
const options = optimize.GradientDescentOptions(f64){
    .max_iter = 1000,
    .tol = 1e-6,
    .learning_rate = 0.01,
};

var result = try unconstrained.gradient_descent(f64, rosenbrock, rosenbrock_grad, &x0, options, allocator);
defer allocator.free(result.x);

std.debug.print("Minimum at: [{d:.6}, {d:.6}]\n", .{result.x[0], result.x[1]});
std.debug.print("Objective: {e}\n", .{result.f_val});
std.debug.print("Iterations: {}\n", .{result.n_iter});
```

**Convergence**: Linear

**Use cases**: Simple problems, machine learning (with momentum/adaptive rates)

### Conjugate Gradient

More efficient than gradient descent for convex problems.

```zig
const options = optimize.ConjugateGradientOptions(f64){
    .max_iter = 1000,
    .tol = 1e-6,
};

var result = try unconstrained.conjugate_gradient(f64, rosenbrock, rosenbrock_grad, &x0, options, allocator);
defer allocator.free(result.x);

// Converges much faster than gradient descent
```

**Convergence**: Superlinear for quadratic functions

**Use cases**: Quadratic optimization, large-scale problems

**Variants**: Fletcher-Reeves, Polak-Ribière, Hestenes-Stiefel

### BFGS (Broyden-Fletcher-Goldfarb-Shanno)

Quasi-Newton method that approximates Hessian.

```zig
const options = optimize.BFGSOptions(f64){
    .max_iter = 1000,
    .tol = 1e-6,
};

var result = try unconstrained.bfgs(f64, rosenbrock, rosenbrock_grad, &x0, options, allocator);
defer allocator.free(result.x);

// Fast convergence, doesn't need Hessian
```

**Convergence**: Superlinear

**Memory**: O(n²) for n variables

**Use cases**: General nonlinear optimization, medium-scale problems

### L-BFGS (Limited-memory BFGS)

Memory-efficient variant of BFGS.

```zig
const options = optimize.LBFGSOptions(f64){
    .max_iter = 1000,
    .tol = 1e-6,
    .m = 10,  // Memory limit (number of correction pairs)
};

var result = try unconstrained.lbfgs(f64, rosenbrock, rosenbrock_grad, &x0, options, allocator);
defer allocator.free(result.x);

// Suitable for high-dimensional problems
```

**Convergence**: Superlinear

**Memory**: O(mn) where m is memory parameter

**Use cases**: Large-scale optimization, machine learning

### Nelder-Mead (Simplex)

Derivative-free method.

```zig
const options = optimize.NelderMeadOptions(f64){
    .max_iter = 1000,
    .tol = 1e-6,
};

var result = try unconstrained.nelder_mead(f64, rosenbrock, &x0, options, allocator);
defer allocator.free(result.x);

// No gradients needed
```

**Convergence**: Slow (no theoretical guarantees)

**Use cases**: Noisy functions, discontinuous objectives, small problems

## Constrained Optimization

### Penalty Method

Convert constraints to penalty terms in objective.

```zig
// Minimize x₁² + x₂² subject to x₁ + x₂ = 1
fn objective(x: []const f64) f64 {
    return x[0] * x[0] + x[1] * x[1];
}

fn objective_grad(x: []const f64, grad: []f64, allocator: Allocator) !void {
    grad[0] = 2.0 * x[0];
    grad[1] = 2.0 * x[1];
}

const Constraint = optimize.Constraint(f64);
var constraints = [_]Constraint{
    .{
        .type = .equality,
        .func = struct {
            fn c(x: []const f64) f64 {
                return x[0] + x[1] - 1.0;  // x₁ + x₂ - 1 = 0
            }
        }.c,
        .grad = struct {
            fn g(x: []const f64, grad: []f64, allocator: Allocator) !void {
                _ = allocator;
                grad[0] = 1.0;
                grad[1] = 1.0;
            }
        }.g,
    },
};

const x0 = [_]f64{0.0, 0.0};
const options = optimize.PenaltyMethodOptions(f64){
    .max_iter = 1000,
    .tol = 1e-6,
};

var result = try constrained.penalty_method(f64, objective, objective_grad, &constraints, &x0, options, allocator);
defer allocator.free(result.x);

// Solution: x ≈ [0.5, 0.5]
```

**Use cases**: Equality and inequality constraints, simple problems

### Augmented Lagrangian

More efficient than basic penalty method.

```zig
const options = optimize.AugmentedLagrangianOptions(f64){
    .max_iter = 1000,
    .tol = 1e-6,
};

var result = try constrained.augmented_lagrangian(f64, objective, objective_grad, &constraints, &x0, options, allocator);
defer allocator.free(result.x);

// Better convergence than penalty method
```

**Use cases**: Constrained optimization, robotics, control

### Linear Programming (Simplex)

Solve linear programs: minimize c^T x subject to Ax ≤ b, x ≥ 0

```zig
// Minimize -x₁ - 2x₂
const c = [_]f64{-1.0, -2.0};

// Constraints:
//   -x₁ + x₂ ≤ 1
//   x₁ + x₂ ≤ 3
//   x₁ ≤ 2
//   x₂ ≤ 3
const A = [_][_]f64{
    .{-1.0, 1.0},
    .{1.0, 1.0},
    .{1.0, 0.0},
    .{0.0, 1.0},
};
const b = [_]f64{1.0, 3.0, 2.0, 3.0};

var result = try linear.simplex(f64, &c, &A, &b, allocator);
defer allocator.free(result.x);

std.debug.print("Optimal: [{d:.2}, {d:.2}]\n", .{result.x[0], result.x[1]});
std.debug.print("Objective: {d:.2}\n", .{result.obj_val});
```

**Use cases**: Resource allocation, network flows, scheduling

### Linear Programming (Interior Point)

More efficient for large-scale problems.

```zig
const options = optimize.InteriorPointOptions(f64){
    .max_iter = 100,
    .tol = 1e-6,
};

var result = try linear.interior_point(f64, &c, &A, &b, options, allocator);
defer allocator.free(result.x);

// Faster than simplex for large problems
```

**Use cases**: Large-scale LP, semidefinite programming precursor

## Least Squares

Minimize ||f(x)||² where f: ℝⁿ → ℝᵐ

### Gauss-Newton

For nonlinear least squares.

```zig
// Fit exponential model: y = a * exp(b * x)
fn residual(params: []const f64, x_data: []const f64, y_data: []const f64, r: []f64) void {
    const a = params[0];
    const b = params[1];

    for (x_data, 0..) |x, i| {
        const y_pred = a * @exp(b * x);
        r[i] = y_pred - y_data[i];
    }
}

fn jacobian(params: []const f64, x_data: []const f64, J: [][]f64) void {
    const a = params[0];
    const b = params[1];

    for (x_data, 0..) |x, i| {
        const exp_bx = @exp(b * x);
        J[i][0] = exp_bx;           // ∂r/∂a
        J[i][1] = a * x * exp_bx;   // ∂r/∂b
    }
}

const x_data = [_]f64{0.0, 1.0, 2.0, 3.0};
const y_data = [_]f64{1.0, 2.7, 7.4, 20.1};
const params0 = [_]f64{1.0, 1.0};  // Initial guess

const options = optimize.GaussNewtonOptions(f64){
    .max_iter = 100,
    .tol = 1e-6,
};

var result = try least_squares.gauss_newton(f64, residual, jacobian, &params0, &x_data, &y_data, options, allocator);
defer allocator.free(result.x);

std.debug.print("a = {d:.4}, b = {d:.4}\n", .{result.x[0], result.x[1]});
```

**Convergence**: Fast near solution

**Use cases**: Curve fitting, parameter estimation

### Levenberg-Marquardt

More robust than Gauss-Newton.

```zig
const options = optimize.LevenbergMarquardtOptions(f64){
    .max_iter = 100,
    .tol = 1e-6,
    .lambda = 1e-3,  // Damping parameter
};

var result = try least_squares.levenberg_marquardt(f64, residual, jacobian, &params0, &x_data, &y_data, options, allocator);
defer allocator.free(result.x);

// Combines Gauss-Newton and gradient descent
```

**Convergence**: Robust, handles poor initial guesses

**Use cases**: Nonlinear regression, general curve fitting

**Advantage**: Damping parameter prevents divergence

## Common Optimization Problems

### Linear Regression (Analytical)

```zig
// Minimize ||Ax - b||²
const linalg = zuda.linalg;
var A = try NDArray(f64, 2).fromSlice(allocator, &design_matrix, .row_major);
defer A.deinit();

var b = try NDArray(f64, 1).fromSlice(allocator, &targets, .row_major);
defer b.deinit();

var x = try linalg.solve.lstsq(f64, A, b, allocator);
defer x.deinit();

// Exact solution via normal equations: x = (A^T A)^{-1} A^T b
```

### Logistic Regression

```zig
// Maximize log-likelihood (minimize negative log-likelihood)
fn neg_log_likelihood(beta: []const f64, X: []const []const f64, y: []const f64) f64 {
    var loss: f64 = 0.0;
    for (X, 0..) |x_i, i| {
        const z = dot_product(beta, x_i);
        const p = 1.0 / (1.0 + @exp(-z));  // Sigmoid
        loss -= y[i] * @log(p) + (1.0 - y[i]) * @log(1.0 - p);
    }
    return loss;
}

fn gradient(beta: []const f64, X: []const []const f64, y: []const f64, grad: []f64, allocator: Allocator) !void {
    @memset(grad, 0.0);
    for (X, 0..) |x_i, i| {
        const z = dot_product(beta, x_i);
        const p = 1.0 / (1.0 + @exp(-z));
        const error = p - y[i];
        for (x_i, 0..) |x_ij, j| {
            grad[j] += error * x_ij;
        }
    }
}

const beta0 = [_]f64{0.0, 0.0, 0.0};  // Initialize coefficients
var result = try unconstrained.lbfgs(f64, neg_log_likelihood, gradient, &beta0, options, allocator);
defer allocator.free(result.x);
```

### Portfolio Optimization

```zig
// Minimize risk subject to minimum return
// Minimize x^T Σ x subject to μ^T x ≥ r_min, Σ x_i = 1, x ≥ 0

fn portfolio_objective(x: []const f64, cov_matrix: []const []const f64) f64 {
    // Risk: x^T Σ x
    var risk: f64 = 0.0;
    for (x, 0..) |xi, i| {
        for (x, 0..) |xj, j| {
            risk += xi * cov_matrix[i][j] * xj;
        }
    }
    return risk;
}

// Use constrained optimization with return and budget constraints
```

### Support Vector Machine (SVM)

```zig
// Minimize ||w||² subject to y_i(w^T x_i + b) ≥ 1

// Convert to dual problem and solve with quadratic programming
// (Implementation requires QP solver)
```

## Global Optimization

Methods that avoid local minima.

### Multi-start Optimization

```zig
var best_result: ?optimize.Result(f64) = null;
var best_value: f64 = std.math.inf(f64);

var rng = std.rand.DefaultPrng.init(42);
for (0..100) |_| {
    // Random initial guess
    var x0 = [_]f64{0, 0};
    for (&x0) |*xi| {
        xi.* = rng.random().float(f64) * 10.0 - 5.0;  // [-5, 5]
    }

    var result = try unconstrained.lbfgs(f64, rosenbrock, rosenbrock_grad, &x0, options, allocator);
    defer allocator.free(result.x);

    if (result.f_val < best_value) {
        best_value = result.f_val;
        best_result = result;
    }
}
```

### Simulated Annealing

Probabilistic method for global optimization.

```zig
// Coming in future version
// Accepts worse solutions with decreasing probability (temperature)
```

### Genetic Algorithms

Population-based optimization.

```zig
// Coming in future version
// Evolves population through selection, crossover, mutation
```

## Performance Tips

1. **Provide gradients when possible**: Finite differences are slow and inaccurate
2. **Use L-BFGS for large problems**: O(mn) vs O(n²) for BFGS
3. **Choose appropriate method**:
   - Smooth, unconstrained → BFGS/L-BFGS
   - Quadratic → Conjugate gradient
   - Least squares → Gauss-Newton/Levenberg-Marquardt
   - Linear → Simplex/Interior point
   - Noisy/discontinuous → Nelder-Mead
4. **Scale variables**: Normalize inputs to similar magnitudes
5. **Good initial guess**: Can dramatically reduce iterations
6. **Warm starts**: Reuse solution from similar problem

## Convergence Diagnostics

```zig
// Check gradient norm
const grad_norm = vector_norm(&result.grad);
if (grad_norm < 1e-6) {
    std.debug.print("Converged: gradient norm = {e}\n", .{grad_norm});
} else {
    std.debug.print("Warning: gradient norm = {e} (may not be at minimum)\n", .{grad_norm});
}

// Check function value change
if (result.n_iter == options.max_iter) {
    std.debug.print("Warning: reached max iterations without convergence\n", .{});
}

// Verify constraints satisfied
for (constraints) |c| {
    const violation = c.func(result.x);
    if (@abs(violation) > 1e-4) {
        std.debug.print("Warning: constraint violated by {e}\n", .{violation});
    }
}
```

## Error Handling

```zig
const result = unconstrained.lbfgs(f64, obj, grad, x0, options, allocator) catch |err| switch (err) {
    error.NoConvergence => {
        std.debug.print("Failed to converge in {} iterations\n", .{options.max_iter});
        return;
    },
    error.LineSearchFailed => {
        std.debug.print("Line search could not find valid step\n", .{});
        return;
    },
    error.InvalidDimension => {
        std.debug.print("Gradient dimension mismatch\n", .{});
        return;
    },
    else => return err,
};

const lp_result = linear.simplex(f64, c, A, b, allocator) catch |err| switch (err) {
    error.Infeasible => {
        std.debug.print("Problem has no feasible solution\n", .{});
        return;
    },
    error.Unbounded => {
        std.debug.print("Problem is unbounded (objective → -∞)\n", .{});
        return;
    },
    else => return err,
};
```

## See Also

- [Numerical Methods Guide](numeric.md) — Integration, differentiation, root finding
- [Linear Algebra Guide](linalg.md) — Matrix operations for optimization
- [Statistics Guide](stats.md) — Regression and statistical estimation
