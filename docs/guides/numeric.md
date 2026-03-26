# Numerical Methods — Integration, Differentiation, and Root Finding

## Overview

The `numeric` module provides numerical algorithms for calculus operations, root finding, and interpolation. It's designed for scientific simulation, numerical analysis, and engineering applications where analytical solutions are unavailable.

## Module Structure

```zig
const numeric = zuda.numeric;

// Numerical integration
const integrate = numeric.integration;

// Numerical differentiation
const diff = numeric.differentiation;

// Root finding
const roots = numeric.roots;

// Interpolation
const interp = numeric.interpolation;

// Ordinary Differential Equations
const ode = numeric.ode;
```

## Numerical Integration

Compute definite integrals: ∫ₐᵇ f(x) dx

### Trapezoidal Rule

Linear interpolation between points.

```zig
// Integrate f(x) = x² from 0 to 1
fn square(x: f64) f64 {
    return x * x;
}

const result = try integrate.trapz(f64, square, 0.0, 1.0, 1000, allocator);
// result ≈ 1/3 = 0.333...

// More points → better accuracy
const accurate = try integrate.trapz(f64, square, 0.0, 1.0, 100000, allocator);
```

**Error**: O(h²) where h = (b-a)/n

**Use cases**: Smooth integrands, quick estimates

### Simpson's Rule

Quadratic interpolation (more accurate than trapezoidal).

```zig
const result = try integrate.simps(f64, square, 0.0, 1.0, 1000, allocator);
// Even more accurate than trapz for same n
```

**Error**: O(h⁴)

**Requirement**: Number of intervals must be even

**Use cases**: Smooth functions, higher accuracy needed

### Romberg Integration

Adaptive Richardson extrapolation for high accuracy.

```zig
const result = try integrate.romberg(f64, square, 0.0, 1.0, 1e-10, allocator);
// Automatically adapts to achieve tolerance
```

**Error**: Exponential convergence for smooth functions

**Use cases**: High-precision integration, smooth integrands

### Gaussian Quadrature

Uses optimal sampling points for polynomial integrands.

```zig
// 5-point Gauss-Legendre
const result = try integrate.quad(f64, square, 0.0, 1.0, 5, allocator);
// Exact for polynomials up to degree 2n-1 = 9
```

**Use cases**: Polynomial integrands, finite element methods

### Integrating Tabulated Data

When you have discrete data points instead of a function.

```zig
const x = [_]f64{0.0, 0.5, 1.0, 1.5, 2.0};
const y = [_]f64{0.0, 0.25, 1.0, 2.25, 4.0};  // x²

const result = try integrate.trapz_data(f64, &x, &y);
// Integrate using trapezoidal rule on data points
```

## Numerical Differentiation

Approximate derivatives: f'(x), f''(x)

### Forward Difference

```zig
fn sine(x: f64) f64 {
    return @sin(x);
}

const h = 1e-5;
const x0 = std.math.pi / 4.0;

const derivative = try diff.forward(f64, sine, x0, h);
// f'(x) ≈ (f(x+h) - f(x)) / h
// For sin(π/4), derivative ≈ cos(π/4) = 0.707...
```

**Error**: O(h)

### Backward Difference

```zig
const derivative = try diff.backward(f64, sine, x0, h);
// f'(x) ≈ (f(x) - f(x-h)) / h
```

**Error**: O(h)

### Central Difference

More accurate than forward/backward.

```zig
const derivative = try diff.central(f64, sine, x0, h);
// f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
```

**Error**: O(h²)

**Recommended**: Use central difference for better accuracy

### Second Derivative

```zig
const second_derivative = try diff.second(f64, sine, x0, h);
// f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
// For sin(π/4), f'' ≈ -sin(π/4) = -0.707...
```

**Error**: O(h²)

### Gradient (Multivariate)

Compute ∇f for functions f: ℝⁿ → ℝ

```zig
fn rosenbrock(x: []const f64) f64 {
    // f(x,y) = (1-x)² + 100(y-x²)²
    const dx = 1.0 - x[0];
    const dy = x[1] - x[0] * x[0];
    return dx * dx + 100.0 * dy * dy;
}

const point = [_]f64{0.5, 0.5};
const grad = try diff.gradient(f64, rosenbrock, &point, 1e-5, allocator);
defer allocator.free(grad);

// grad[0] = ∂f/∂x, grad[1] = ∂f/∂y
```

**Use cases**: Optimization, gradient descent

## Root Finding

Find x where f(x) = 0

### Bisection Method

Guaranteed convergence for continuous functions.

```zig
fn cubic(x: f64) f64 {
    return x * x * x - 2.0 * x - 5.0;  // x³ - 2x - 5
}

const root = try roots.bisect(f64, cubic, 2.0, 3.0, 1e-6, 100);
// Find root in [2, 3]
// Requires f(a) and f(b) have opposite signs
```

**Convergence**: Linear (slow but guaranteed)

**Requirements**: Continuous function, f(a) × f(b) < 0

### Newton's Method

Fast convergence when derivative is available.

```zig
fn cubic_derivative(x: f64) f64 {
    return 3.0 * x * x - 2.0;  // 3x² - 2
}

const root = try roots.newton(f64, cubic, cubic_derivative, 2.5, 1e-6, 100);
// Initial guess x₀ = 2.5
```

**Convergence**: Quadratic (very fast near root)

**Requirements**: Differentiable function, good initial guess

**Caution**: Can diverge with poor initial guess

### Secant Method

Like Newton's but doesn't require derivative.

```zig
const root = try roots.secant(f64, cubic, 2.0, 3.0, 1e-6, 100);
// Uses two initial points instead of derivative
```

**Convergence**: Superlinear (between bisection and Newton's)

**Advantage**: No derivative needed

### Brent's Method

Combines bisection, secant, and inverse quadratic interpolation.

```zig
const root = try roots.brent(f64, cubic, 2.0, 3.0, 1e-6, 100);
// Best general-purpose method
```

**Convergence**: Guaranteed + fast

**Recommended**: Use Brent's method for general root finding

### Fixed-Point Iteration

Find x where g(x) = x

```zig
fn fixed_point_func(x: f64) f64 {
    // Rearrange x³ - 2x - 5 = 0 to x = (x³ - 5) / 2
    return (x * x * x - 5.0) / 2.0;
}

const root = try roots.fixed_point(f64, fixed_point_func, 2.5, 1e-6, 100);
```

**Convergence**: Requires |g'(x)| < 1 near fixed point

## Interpolation

Estimate values between known data points.

### Linear Interpolation

```zig
const x_data = [_]f64{0.0, 1.0, 2.0, 3.0};
const y_data = [_]f64{0.0, 1.0, 4.0, 9.0};  // x²

const y_interp = try interp.linear(f64, &x_data, &y_data, 1.5);
// Interpolate at x = 1.5
// Linear between (1, 1) and (2, 4): y ≈ 2.5
```

**Use cases**: Simple, fast, no overshoot

### Polynomial Interpolation (Lagrange)

```zig
const y_interp = try interp.lagrange(f64, &x_data, &y_data, 1.5, allocator);
// Fits polynomial through all points
```

**Degree**: n-1 for n points

**Caution**: High-degree polynomials can oscillate (Runge's phenomenon)

### Cubic Spline Interpolation

Smooth piecewise cubic polynomials.

```zig
var spline = try interp.CubicSpline(f64).init(&x_data, &y_data, allocator);
defer spline.deinit();

const y_interp = try spline.eval(1.5);
// Continuous second derivative (smooth)
```

**Use cases**: Smooth curves, natural-looking interpolation

**Advantages**: No oscillation, smooth derivatives

### Nearest-Neighbor Interpolation

```zig
const y_interp = try interp.nearest(f64, &x_data, &y_data, 1.5);
// Returns y_data[i] where x_data[i] is closest to 1.5
```

**Use cases**: Categorical data, fast lookups

## Ordinary Differential Equations (ODEs)

Solve dy/dt = f(t, y)

### Euler's Method

Simplest ODE solver.

```zig
// Solve dy/dt = -y, y(0) = 1
// Analytical solution: y(t) = exp(-t)
fn dydt(t: f64, y: f64) f64 {
    _ = t;
    return -y;
}

const t0 = 0.0;
const y0 = 1.0;
const t_final = 2.0;
const n_steps = 100;

var result = try ode.euler(f64, dydt, t0, y0, t_final, n_steps, allocator);
defer allocator.free(result.t);
defer allocator.free(result.y);

// result.t[i] = time at step i
// result.y[i] = solution at step i
```

**Error**: O(h) per step, O(h) global

**Use cases**: Educational, simple problems

### Runge-Kutta 4th Order (RK4)

Most popular ODE solver.

```zig
var result = try ode.rk4(f64, dydt, t0, y0, t_final, n_steps, allocator);
defer allocator.free(result.t);
defer allocator.free(result.y);

// Much more accurate than Euler for same n_steps
```

**Error**: O(h⁴) per step, O(h⁴) global

**Recommended**: Use RK4 for general ODE solving

### Systems of ODEs

Solve dy₁/dt = f₁(t, y₁, y₂), dy₂/dt = f₂(t, y₁, y₂), ...

```zig
// Lotka-Volterra predator-prey model
fn lotka_volterra(t: f64, y: []const f64, dydt: []f64) void {
    _ = t;
    const prey = y[0];
    const predator = y[1];

    const alpha = 1.0;   // Prey growth rate
    const beta = 0.1;    // Predation rate
    const gamma = 1.5;   // Predator death rate
    const delta = 0.075; // Predator reproduction rate

    dydt[0] = alpha * prey - beta * prey * predator;
    dydt[1] = delta * prey * predator - gamma * predator;
}

const y0 = [_]f64{10.0, 5.0};  // Initial prey, predator
var result = try ode.rk4_system(f64, lotka_volterra, t0, &y0, t_final, n_steps, allocator);
defer allocator.free(result.t);
defer result.y.deinit();

// result.y[i] = state vector at step i
// result.y[i][0] = prey population
// result.y[i][1] = predator population
```

## Common Applications

### Numerical Integration of Real Data

```zig
// Integrate experimental data (e.g., velocity → distance)
const time = [_]f64{0.0, 1.0, 2.0, 3.0, 4.0};
const velocity = [_]f64{0.0, 2.0, 4.0, 6.0, 8.0};  // m/s

const distance = try integrate.trapz_data(f64, &time, &velocity);
std.debug.print("Total distance: {d} meters\n", .{distance});
```

### Numerical Optimization (Gradient Descent)

```zig
fn objective(x: []const f64) f64 {
    return x[0] * x[0] + x[1] * x[1];  // f(x,y) = x² + y²
}

var x = [_]f64{5.0, 5.0};  // Starting point
const learning_rate = 0.1;
const n_iterations = 100;

for (0..n_iterations) |_| {
    const grad = try diff.gradient(f64, objective, &x, 1e-5, allocator);
    defer allocator.free(grad);

    // Update: x -= learning_rate * grad
    for (x, 0..) |*xi, i| {
        xi.* -= learning_rate * grad[i];
    }
}

// x converges to [0, 0] (minimum)
```

### Curve Fitting with Least Squares

```zig
// Fit data to model y = a*exp(b*x)
const x_data = [_]f64{0.0, 1.0, 2.0, 3.0};
const y_data = [_]f64{1.0, 2.7, 7.4, 20.1};

// Linearize: ln(y) = ln(a) + b*x
var ln_y = try std.ArrayList(f64).initCapacity(allocator, y_data.len);
defer ln_y.deinit();

for (y_data) |y| {
    try ln_y.append(@log(y));
}

// Linear regression on ln(y) vs x
const stats = zuda.stats;
var result = try stats.correlation.linregress(f64, &x_data, ln_y.items, allocator);
defer result.deinit(allocator);

const a = @exp(result.intercept);
const b = result.slope;

std.debug.print("Fitted model: y = {d:.2} * exp({d:.2} * x)\n", .{a, b});
```

### Boundary Value Problems

Solve ODE with conditions at both ends.

```zig
// Shooting method: guess initial derivative, iterate until boundary satisfied
fn shooting_method(f: anytype, t0: f64, y0: f64, t_final: f64, y_final: f64) !f64 {
    // Try different initial slopes until endpoint matches
    // (Simplified; production code would use root finding)
    var guess: f64 = 0.0;

    while (true) {
        var result = try ode.rk4(f64, f, t0, y0, t_final, 100, allocator);
        defer allocator.free(result.t);
        defer allocator.free(result.y);

        const endpoint = result.y[result.y.len - 1];
        if (@abs(endpoint - y_final) < 1e-6) {
            return guess;
        }

        guess += 0.1;  // Adjust (use root finding in practice)
    }
}
```

### Stiff ODEs

For stiff systems, use implicit methods (not yet implemented in v1.0).

```zig
// Example of stiff system: dy/dt = -1000*y + 1000*t²
// Explicit methods require tiny steps; implicit methods are stable
// Coming in future version: Backward Euler, BDF methods
```

## Performance Tips

1. **Choose step size carefully**: Balance accuracy vs. computation
2. **Use RK4 over Euler**: Much better accuracy/cost ratio
3. **Cache function evaluations**: If f(x) is expensive, memoize
4. **Use adaptive methods**: Future versions will have adaptive step size
5. **Vectorize when possible**: Batch operations on arrays

## Error Estimation

### Richardson Extrapolation

```zig
// Compute derivative with h and h/2, estimate error
const h = 1e-3;
const deriv1 = try diff.central(f64, sine, x0, h);
const deriv2 = try diff.central(f64, sine, x0, h / 2.0);

// Error estimate (assuming O(h²))
const error_estimate = @abs(deriv2 - deriv1) / 3.0;
```

### Comparing Methods

```zig
// Compare Euler vs RK4 accuracy
const analytical = @exp(-t_final);  // True solution

var euler_result = try ode.euler(f64, dydt, t0, y0, t_final, 100, allocator);
defer allocator.free(euler_result.t);
defer allocator.free(euler_result.y);

var rk4_result = try ode.rk4(f64, dydt, t0, y0, t_final, 100, allocator);
defer allocator.free(rk4_result.t);
defer allocator.free(rk4_result.y);

const euler_error = @abs(euler_result.y[euler_result.y.len - 1] - analytical);
const rk4_error = @abs(rk4_result.y[rk4_result.y.len - 1] - analytical);

std.debug.print("Euler error: {e}\n", .{euler_error});
std.debug.print("RK4 error: {e}\n", .{rk4_error});
```

## Error Handling

```zig
const root = roots.newton(f64, f, df, x0, tol, max_iter) catch |err| switch (err) {
    error.NoConvergence => {
        std.debug.print("Failed to converge in {} iterations\n", .{max_iter});
        return;
    },
    error.DivisionByZero => {
        std.debug.print("Derivative is zero (use different method)\n", .{});
        return;
    },
    else => return err,
};

const spline = interp.CubicSpline(f64).init(x_data, y_data, allocator) catch |err| switch (err) {
    error.InsufficientData => {
        std.debug.print("Need at least 2 data points\n", .{});
        return;
    },
    error.NonMonotonic => {
        std.debug.print("x_data must be strictly increasing\n", .{});
        return;
    },
    else => return err,
};
```

## See Also

- [Optimization Guide](optimize.md) — Nonlinear optimization algorithms
- [Linear Algebra Guide](linalg.md) — Matrix operations for numerical methods
- [Statistics Guide](stats.md) — Statistical analysis and regression
