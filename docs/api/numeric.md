# Numerical Methods API Reference

## Overview

The Numerical Methods module provides algorithms for numerical integration, differentiation, root finding, interpolation, ordinary differential equations, curve fitting, and special functions. All operations are designed for scientific computing applications where analytical solutions are unavailable.

### Import

```zig
const zuda = @import("zuda");
const numeric = zuda.numeric;
const integration = numeric.integration;
const differentiation = numeric.differentiation;
const root_finding = numeric.root_finding;
const interpolation = numeric.interpolation;
const ode = numeric.ode;
const curve_fitting = numeric.curve_fitting;
const special = numeric.special;
```

### Key Features

- **Numerical Integration**: Trapezoidal and Simpson's rules for definite integrals
- **Numerical Differentiation**: Finite difference methods (forward, central, backward)
- **Root Finding**: Bisection, Newton-Raphson, Brent's method, secant, fixed-point iteration
- **Interpolation**: 1D linear interpolation with extrapolation handling
- **Ordinary Differential Equations**: Euler, RK4, RK45 adaptive solvers
- **Curve Fitting**: Levenberg-Marquardt non-linear least squares
- **Special Functions**: Gamma, Beta, Error function, Bessel functions
- **Type Generic**: Support for f32, f64 floating-point types
- **No Allocations**: Most algorithms use O(1) memory

---

## Error Types

All numeric operations use consistent error types:

```zig
pub const IntegrationError = error{
    DimensionMismatch,      // x.len != y.len
    InsufficientPoints,     // fewer than required points
    OddLengthRequired,      // Simpson requires odd number of points
};

pub const DifferentiationError = error{
    InsufficientPoints,     // fewer than required points
    InvalidArgument,        // invalid parameter (h <= 0, etc.)
};

pub const RootFindingError = error{
    InvalidInterval,        // f(a)*f(b) >= 0 or a >= b
    MaxIterationsExceeded,  // convergence not reached
    DerivativeZero,         // f'(x) = 0 (Newton)
    NonFiniteResult,        // NaN or Inf encountered
};

pub const InterpolationError = error{
    DimensionMismatch,      // x.len != y.len
    InsufficientPoints,     // fewer than 2 points
    NonMonotonicX,          // x not monotonically increasing
};

pub const CurveFittingError = error{
    DimensionMismatch,      // x_data.len != y_data.len
    InsufficientData,       // too few points
    EmptyData,              // no data provided
    InvalidInitialGuess,    // NaN/Inf in parameters
    MaxIterationsExceeded,  // convergence not reached
    SingularJacobian,       // J^T·J is rank-deficient
    NonFiniteResult,        // NaN/Inf in computation
    AllocationFailed,       // memory allocation failed
};

pub const SpecialFunctionError = error{
    DomainError,            // argument outside valid domain
    NotImplemented,         // placeholder for future functions
};
```

---

## Numerical Integration

Compute definite integrals: ∫ₐᵇ f(x) dx

### trapezoid(T, x, y, allocator)

Approximate integral using the trapezoidal rule with discrete data points.

```zig
pub fn trapezoid(comptime T: type, x: []const T, y: []const T, allocator: Allocator) !T
```

**Description**: Integrates tabulated function values using piecewise linear interpolation. Each trapezoid under the curve contributes to the total integral.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `x`: Slice of x-coordinates (domain points) — must be monotonically increasing
- `y`: Slice of y-coordinates (function values) — must have same length as x
- `allocator`: Memory allocator (not used, included for API consistency)

**Returns**: Approximate value of the integral

**Errors**:
- `error.DimensionMismatch`: x.len != y.len
- `error.InsufficientPoints`: x.len < 2

**Time**: O(n) where n = number of points
**Space**: O(1)

**Accuracy**: O(h²) where h = average grid spacing. Exact for linear functions.

**Example**:
```zig
const x = [_]f64{0.0, 0.5, 1.0};
const y = [_]f64{0.0, 0.25, 1.0};  // x²
const result = try integration.trapezoid(f64, &x, &y, allocator);
// result ≈ 0.333 (integral of x² from 0 to 1)
```

---

### simpson(T, x, y, allocator)

Approximate integral using Simpson's rule with discrete data points.

```zig
pub fn simpson(comptime T: type, x: []const T, y: []const T, allocator: Allocator) !T
```

**Description**: Integrates using piecewise quadratic interpolation. Simpson's rule is more accurate than trapezoidal for smooth functions.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `x`: Slice of x-coordinates (domain points) — must be monotonically increasing
- `y`: Slice of y-coordinates (function values) — must have same length as x
- `allocator`: Memory allocator (not used, included for API consistency)

**Returns**: Approximate value of the integral

**Errors**:
- `error.DimensionMismatch`: x.len != y.len
- `error.InsufficientPoints`: x.len < 3
- `error.OddLengthRequired`: x.len must be odd (Simpson requires even number of intervals)

**Time**: O(n) where n = number of points
**Space**: O(1)

**Accuracy**: O(h⁴) where h = average grid spacing. Exact for cubic functions.

**Example**:
```zig
const x = [_]f64{0.0, 0.33, 0.67, 1.0};  // 3 intervals (4 points)
const y = [_]f64{0.0, 0.109, 0.448, 1.0};  // x²
const result = try integration.simpson(f64, &x, &y, allocator);
// result ≈ 0.333 (more accurate than trapezoid)
```

---

## Numerical Differentiation

Approximate derivatives: f'(x)

### diff(T, y, dx, allocator)

Compute numerical derivative using finite differences.

```zig
pub fn diff(comptime T: type, y: []const T, dx: T, allocator: Allocator) ![]T
```

**Description**: Computes derivatives at all points using forward difference at boundaries and central difference at interior points. Returns an allocated array of derivatives.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `y`: Slice of function values (y[i] = f(x[i]))
- `dx`: Uniform spacing between sample points (must be > 0)
- `allocator`: Memory allocator for output array (caller owns returned memory)

**Returns**: Allocated array of derivatives (length = y.len, caller must free)

**Errors**:
- `error.InsufficientPoints`: y.len < 2
- `error.OutOfMemory`: allocation fails

**Formulas**:
- Forward (i=0): dy[0] = (y[1] - y[0]) / dx
- Central (0 < i < n-1): dy[i] = (y[i+1] - y[i-1]) / (2*dx)
- Backward (i=n-1): dy[n-1] = (y[n-1] - y[n-2]) / dx

**Time**: O(n) where n = number of points
**Space**: O(n) for output array

**Accuracy**:
- Forward/Backward: O(dx)
- Central: O(dx²)

**Example**:
```zig
const y = [_]f64{0.0, 0.1, 0.4, 0.9, 1.6};  // x² at x = 0, 0.1, 0.2, ...
const dx = 0.1;
const dy = try differentiation.diff(f64, &y, dx, allocator);
defer allocator.free(dy);
// dy[0] ≈ 0.1, dy[1] ≈ 0.2, dy[2] ≈ 0.3, etc.
```

---

### gradient(T, y, dx, allocator)

Alias for `diff()` for NumPy compatibility.

```zig
pub fn gradient(comptime T: type, y: []const T, dx: T, allocator: Allocator) ![]T
```

**Description**: Same as `diff()`. Provided as an alias for API compatibility with NumPy.

**Time**: O(n)
**Space**: O(n)

---

### jacobian(T, num_funcs, funcs, x, h, allocator)

Compute Jacobian matrix of a vector function F: ℝⁿ → ℝᵐ

```zig
pub fn jacobian(
    comptime T: type,
    num_funcs: usize,
    funcs: []const *const fn([]const T) T,
    x: []const T,
    h: T,
    allocator: Allocator,
) !NDArray(T, 2)
```

**Description**: Computes the Jacobian matrix J[i,j] = ∂fᵢ/∂xⱼ using central difference approximation.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `num_funcs`: Number of functions (m)
- `funcs`: Array of m function pointers, each f: ℝⁿ → ℝ
- `x`: Evaluation point (n-dimensional)
- `h`: Step size (must be > 0)
- `allocator`: Memory allocator

**Returns**: m×n NDArray Jacobian matrix

**Errors**:
- `error.InvalidArgument`: if h ≤ 0, num_funcs == 0, or x.len == 0
- `error.OutOfMemory`: allocation fails

**Time**: O(m·n) function evaluations
**Space**: O(m·n) for Jacobian matrix

**Example**:
```zig
fn f1(x: []const f64) f64 { return x[0] * x[0] + x[1]; }
fn f2(x: []const f64) f64 { return x[0] + x[1] * x[1]; }

const funcs = [_]*const fn([]const f64) f64{&f1, &f2};
const point = [_]f64{1.0, 2.0};
const J = try differentiation.jacobian(f64, 2, &funcs, &point, 1e-5, allocator);
defer J.deinit();
// J[0,0] = ∂f1/∂x = 2.0, J[0,1] = ∂f1/∂y = 1.0
// J[1,0] = ∂f2/∂x = 1.0, J[1,1] = ∂f2/∂y = 4.0
```

---

## Root Finding

Find x where f(x) = 0

### bisect(T, func, a, b, tol, max_iter)

Find root using the bisection method.

```zig
pub fn bisect(
    comptime T: type,
    func: *const fn (T) T,
    a: T,
    b: T,
    tol: T,
    max_iter: usize
) RootFindingError!T
```

**Description**: Repeatedly halves the interval, guaranteed to converge if f(a) and f(b) have opposite signs.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `func`: Function pointer where we seek f(x) = 0
- `a, b`: Endpoints of interval — must satisfy f(a)·f(b) < 0
- `tol`: Tolerance for convergence (stop when |b-a| < tol)
- `max_iter`: Maximum number of iterations

**Returns**: Approximate root x

**Errors**:
- `error.InvalidInterval`: f(a)·f(b) ≥ 0 or a ≥ b
- `error.MaxIterationsExceeded`: tolerance not met after max_iter
- `error.NonFiniteResult`: NaN or Inf encountered

**Time**: O(log₂((b-a)/tol)) iterations
**Space**: O(1)

**Convergence**: Linear (guaranteed)

**Example**:
```zig
fn cubic(x: f64) f64 {
    return x * x * x - 2.0 * x - 5.0;  // x³ - 2x - 5
}

const root = try root_finding.bisect(f64, cubic, 2.0, 3.0, 1e-6, 100);
// root ≈ 2.094551 (verified: f(root) ≈ 0)
```

---

### newton(T, func, dfunc, x0, tol, max_iter)

Find root using Newton-Raphson method.

```zig
pub fn newton(
    comptime T: type,
    func: *const fn (T) T,
    dfunc: *const fn (T) T,
    x0: T,
    tol: T,
    max_iter: usize
) RootFindingError!T
```

**Description**: Uses the derivative to iteratively improve root estimate: x_new = x - f(x)/f'(x)

**Parameters**:
- `T`: Numeric type (f32, f64)
- `func`: Function pointer f(x)
- `dfunc`: Derivative function pointer f'(x)
- `x0`: Initial guess
- `tol`: Tolerance for convergence (stop when |f(x)| < tol)
- `max_iter`: Maximum number of iterations

**Returns**: Approximate root x where |f(x)| < tol

**Errors**:
- `error.DerivativeZero`: if f'(x) = 0 at any iteration
- `error.MaxIterationsExceeded`: tolerance not met after max_iter
- `error.NonFiniteResult`: NaN or Inf encountered

**Time**: O(iterations) — typically very fast (few iterations)
**Space**: O(1)

**Convergence**: Quadratic (very fast near root, may diverge far from root)

**Example**:
```zig
fn cubic(x: f64) f64 {
    return x * x * x - 2.0 * x - 5.0;
}

fn cubic_deriv(x: f64) f64 {
    return 3.0 * x * x - 2.0;
}

const root = try root_finding.newton(f64, cubic, cubic_deriv, 2.5, 1e-6, 100);
// root ≈ 2.094551 (converges in ~5 iterations)
```

---

### brent(T, func, a, b, tol, max_iter)

Find root using Brent's method.

```zig
pub fn brent(
    comptime T: type,
    func: *const fn (T) T,
    a: T,
    b: T,
    tol: T,
    max_iter: usize
) RootFindingError!T
```

**Description**: Combines bisection, secant, and inverse quadratic interpolation. Guaranteed convergence with superlinear speed.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `func`: Function pointer f(x)
- `a, b`: Endpoints of interval — must satisfy f(a)·f(b) < 0
- `tol`: Tolerance for convergence
- `max_iter`: Maximum number of iterations

**Returns**: Approximate root x

**Errors**:
- `error.InvalidInterval`: f(a)·f(b) ≥ 0 or a ≥ b
- `error.MaxIterationsExceeded`: tolerance not met after max_iter
- `error.NonFiniteResult`: NaN or Inf encountered

**Time**: O(iterations) — typically 6-15 iterations
**Space**: O(1)

**Convergence**: Superlinear (faster than bisection, guaranteed like bisection)

**Recommended**: Use Brent's method for general root finding

**Example**:
```zig
const root = try root_finding.brent(f64, cubic, 2.0, 3.0, 1e-6, 100);
// root ≈ 2.094551 (fastest among bracket-based methods)
```

---

### secant(T, func, x0, x1, tol, max_iter)

Find root using secant method.

```zig
pub fn secant(
    comptime T: type,
    func: *const fn (T) T,
    x0: T,
    x1: T,
    tol: T,
    max_iter: usize
) RootFindingError!T
```

**Description**: Like Newton's method but estimates derivative via finite difference using two points.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `func`: Function pointer f(x)
- `x0, x1`: Two initial guesses (not required to bracket root)
- `tol`: Tolerance for convergence
- `max_iter`: Maximum number of iterations

**Returns**: Approximate root x

**Errors**:
- `error.MaxIterationsExceeded`: tolerance not met after max_iter
- `error.NonFiniteResult`: NaN or Inf encountered

**Time**: O(iterations)
**Space**: O(1)

**Convergence**: Superlinear (order ≈ 1.618, faster than bisection, no derivative needed)

**Example**:
```zig
const root = try root_finding.secant(f64, cubic, 2.0, 3.0, 1e-6, 100);
// No derivative needed, fast convergence
```

---

### fixed_point(T, func, x0, tol, max_iter)

Find fixed point using fixed-point iteration.

```zig
pub fn fixed_point(
    comptime T: type,
    func: *const fn (T) T,
    x0: T,
    tol: T,
    max_iter: usize
) RootFindingError!T
```

**Description**: Finds x where g(x) = x by iteration x_new = g(x_old).

**Parameters**:
- `T`: Numeric type (f32, f64)
- `func`: Iteration function g(x) where we seek g(x) = x
- `x0`: Initial guess
- `tol`: Tolerance for convergence (stop when |x_new - x| < tol)
- `max_iter`: Maximum number of iterations

**Returns**: Fixed point x where g(x) ≈ x

**Errors**:
- `error.MaxIterationsExceeded`: convergence not reached after max_iter
- `error.NonFiniteResult`: NaN or Inf encountered

**Time**: O(iterations)
**Space**: O(1)

**Convergence**: Linear (requires |g'(x)| < 1 near fixed point)

**Example**:
```zig
// Find x where x = cos(x)
fn fixed_pt(x: f64) f64 {
    return @cos(x);
}

const x = try root_finding.fixed_point(f64, fixed_pt, 0.5, 1e-6, 100);
// x ≈ 0.739085 (Dottie number)
```

---

## Interpolation

Estimate values between known data points.

### interp1d(T, x, y, x_new, allocator)

Perform 1D linear interpolation at query points.

```zig
pub fn interp1d(
    comptime T: type,
    x: []const T,
    y: []const T,
    x_new: []const T,
    allocator: Allocator
) ![]T
```

**Description**: Interpolates function values at arbitrary query points using linear interpolation between consecutive points. Extrapolation uses constant (first/last value).

**Parameters**:
- `T`: Numeric type (f32, f64)
- `x`: Slice of x-coordinates (sample points) — must be monotonically increasing
- `y`: Slice of y-coordinates (function values) — must have same length as x
- `x_new`: Slice of x-coordinates where we want interpolated values
- `allocator`: Memory allocator for output array (caller owns returned memory)

**Returns**: Allocated array of interpolated y values at x_new points (caller must free)

**Errors**:
- `error.DimensionMismatch`: x.len != y.len
- `error.InsufficientPoints`: x.len < 2
- `error.NonMonotonicX`: x is not monotonically increasing
- `error.OutOfMemory`: allocation fails

**Interpolation Formula** (for x_new[i] in [x[j], x[j+1]]):
```
y_new[i] = y[j] + (y[j+1] - y[j]) * (x_new[i] - x[j]) / (x[j+1] - x[j])
```

**Extrapolation**:
- For x_new[i] < x[0]: y_new[i] = y[0]
- For x_new[i] > x[n-1]: y_new[i] = y[n-1]

**Time**: O(m log n + m) where m = x_new.len, n = x.len (binary search + interpolation)
**Space**: O(m) for output array

**Example**:
```zig
const x = [_]f64{0.0, 1.0, 2.0, 3.0};
const y = [_]f64{0.0, 1.0, 4.0, 9.0};  // x²
const x_new = [_]f64{0.5, 1.5, 2.5};

const y_interp = try interpolation.interp1d(f64, &x, &y, &x_new, allocator);
defer allocator.free(y_interp);
// y_interp[0] ≈ 0.5 (linear interpolation between 0 and 1)
// y_interp[1] ≈ 2.5 (linear interpolation between 1 and 4)
// y_interp[2] ≈ 6.5 (linear interpolation between 4 and 9)
```

---

## Ordinary Differential Equations

Solve dy/dt = f(t, y) with initial condition y(t0) = y0

### Solution(T) — Result Structure

```zig
pub fn Solution(comptime T: type) type {
    return struct {
        t: []const T,       // Time points [t0, t1, ..., t_end]
        y: []const T,       // Solution values [y(t0), y(t1), ..., y(t_end)]
        allocator: Allocator,
        pub fn deinit(self: @This()) void;
    };
}
```

**Note**: Always call `deinit()` on Solution to free allocated memory.

---

### euler(T, derivFn, y0, t_span, dt, allocator)

Solve ODE using explicit Euler method.

```zig
pub fn euler(
    comptime T: type,
    derivFn: *const fn (t: T, y: T) T,
    y0: T,
    t_span: [2]T,
    dt: T,
    allocator: Allocator
) !Solution(T)
```

**Description**: Integrates dy/dt = f(t, y) using forward Euler: y_{n+1} = y_n + dt·f(t_n, y_n)

**Parameters**:
- `T`: Numeric type (f32, f64)
- `derivFn`: Function pointer with signature `fn(t: T, y: T) T`
- `y0`: Initial condition at t = t_span[0]
- `t_span`: [t_start, t_end] — integration interval
- `dt`: Fixed timestep (must be positive)
- `allocator`: Memory allocator for output arrays (caller owns returned Solution)

**Returns**: Solution struct containing arrays t and y of equal length

**Errors**:
- `error.InvalidTimestep`: if dt ≤ 0
- `error.InvalidInterval`: if t_end < t_start
- `error.OutOfMemory`: allocation fails

**Time**: O(n) where n = ceil((t_end - t_start) / dt)
**Space**: O(n) for solution arrays

**Accuracy**:
- Local truncation error: O(dt²)
- Global error: O(dt)
- Exact for linear constant-coefficient ODEs

**Example**:
```zig
fn dydt(t: f64, y: f64) f64 {
    _ = t;
    return -y;  // dy/dt = -y, solution: y(t) = exp(-t)
}

var sol = try ode.euler(f64, dydt, 1.0, .{0.0, 2.0}, 0.01, allocator);
defer sol.deinit();
// sol.y[final] ≈ exp(-2) ≈ 0.135
```

---

### rk4(T, derivFn, y0, t_span, dt, allocator)

Solve ODE using 4th-order Runge-Kutta method.

```zig
pub fn rk4(
    comptime T: type,
    derivFn: *const fn (t: T, y: T) T,
    y0: T,
    t_span: [2]T,
    dt: T,
    allocator: Allocator
) !Solution(T)
```

**Description**: Integrates using RK4 with weighted average of four stages.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `derivFn`: Function pointer with signature `fn(t: T, y: T) T`
- `y0`: Initial condition
- `t_span`: [t_start, t_end] — integration interval
- `dt`: Fixed timestep (must be positive)
- `allocator`: Memory allocator for output arrays

**Returns**: Solution struct containing arrays t and y

**Errors**:
- `error.InvalidTimestep`: if dt ≤ 0
- `error.InvalidInterval`: if t_end < t_start
- `error.OutOfMemory`: allocation fails

**Time**: O(n) where n = ceil((t_end - t_start) / dt)
**Space**: O(n) for solution arrays

**Accuracy**:
- Local truncation error: O(dt⁵)
- Global error: O(dt⁴)
- Exact for polynomials up to degree 3
- Typically 4-6 orders of magnitude more accurate than Euler for same dt

**Recommended**: Use RK4 for general ODE solving

**Example**:
```zig
var sol = try ode.rk4(f64, dydt, 1.0, .{0.0, 2.0}, 0.01, allocator);
defer sol.deinit();
// sol.y[final] ≈ exp(-2) ≈ 0.13534 (much more accurate than Euler)
```

---

### rk45(T, derivFn, y0, t_span, tol, allocator)

Solve ODE using adaptive 5th-order Runge-Kutta (Dormand-Prince method).

```zig
pub fn rk45(
    comptime T: type,
    derivFn: *const fn (t: T, y: T) T,
    y0: T,
    t_span: [2]T,
    tol: T,
    allocator: Allocator
) !Solution(T)
```

**Description**: Adaptive ODE solver that adjusts timestep based on local error estimate.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `derivFn`: Function pointer with signature `fn(t: T, y: T) T`
- `y0`: Initial condition
- `t_span`: [t_start, t_end] — integration interval
- `tol`: Tolerance for error control (adaptive timestep adjusts to achieve this)
- `allocator`: Memory allocator

**Returns**: Solution struct containing arrays t and y (adaptive time points)

**Errors**:
- `error.InvalidInterval`: if t_end < t_start
- `error.InvalidTimestep`: if tol ≤ 0
- `error.OutOfMemory`: allocation fails

**Time**: O(n) where n = adaptive number of timesteps
**Space**: O(n) for solution arrays (n depends on problem stiffness)

**Accuracy**: Achieves requested tolerance through adaptive timestep control

**Use Cases**: Stiff problems, wide range of scales, when dt varies significantly

**Example**:
```zig
var sol = try ode.rk45(f64, dydt, 1.0, .{0.0, 2.0}, 1e-6, allocator);
defer sol.deinit();
// Automatically adjusts timestep to maintain 1e-6 error tolerance
// Number of steps adapted to problem requirements
```

---

## Curve Fitting

Fit a model to data using non-linear least squares.

### CurveFitResult(T) — Result Structure

```zig
pub fn CurveFitResult(comptime T: type) type {
    return struct {
        params: []T,           // Fitted parameters
        residuals: []T,        // Residuals at each data point
        n_iter: usize,         // Number of iterations performed
        final_cost: T,         // Final sum of squared residuals
        converged: bool,       // Whether convergence achieved
    };
}
```

---

### curve_fit(T, model_func, x_data, y_data, p0, allocator)

Fit a model to data using Levenberg-Marquardt algorithm.

```zig
pub fn curve_fit(
    comptime T: type,
    model_func: *const fn (x: T, params: []const T) T,
    x_data: []const T,
    y_data: []const T,
    p0: []const T,
    allocator: Allocator
) !CurveFitResult(T)
```

**Description**: Finds parameters p that minimize sum of squared residuals: Σ(y_i - model(x_i, p))²

Uses Levenberg-Marquardt algorithm which adaptively balances Gauss-Newton (fast near optimum) and gradient descent (robust far away).

**Parameters**:
- `T`: Numeric type (f32, f64)
- `model_func`: Function with signature `fn(x: T, params: []const T) T` returning predicted y
- `x_data`: Slice of independent variable values
- `y_data`: Slice of observed dependent variable values (must match x_data.len)
- `p0`: Initial parameter guess (must not contain NaN/Inf)
- `allocator`: Memory allocator for temporary arrays

**Returns**: CurveFitResult containing:
- `params`: Optimized parameters
- `residuals`: Residual values at each data point
- `n_iter`: Number of iterations performed
- `final_cost`: Final sum of squared residuals
- `converged`: Whether convergence tolerance was met

**Errors**:
- `error.DimensionMismatch`: x_data.len != y_data.len
- `error.EmptyData`: x_data is empty
- `error.InsufficientData`: fewer than 2 data points
- `error.InvalidInitialGuess`: p0 contains NaN or Inf
- `error.AllocationFailed`: memory allocation fails
- `error.NonFiniteResult`: computation produces NaN or Inf
- `error.MaxIterationsExceeded`: convergence not reached after 1000 iterations

**Time**: O(n·m²·iterations) where n = data points, m = parameters, iterations < 100 typical
**Space**: O(n·m) for Jacobian and temporary arrays

**Example**:
```zig
// Fit y = a * exp(b*x) to data
fn model(x: f64, params: []const f64) f64 {
    return params[0] * @exp(params[1] * x);
}

const x_data = [_]f64{0.0, 1.0, 2.0, 3.0};
const y_data = [_]f64{1.0, 2.7, 7.4, 20.1};
const p0 = [_]f64{1.0, 1.0};  // initial guess

var result = try curve_fitting.curve_fit(f64, model, &x_data, &y_data, &p0, allocator);

std.debug.print("a = {d:.4}, b = {d:.4}\n", .{result.params[0], result.params[1]});
std.debug.print("Converged: {}\n", .{result.converged});
```

---

## Special Functions

### gamma(T, x)

Compute gamma function Γ(x) using Lanczos approximation.

```zig
pub fn gamma(comptime T: type, x: T) SpecialFunctionError!T
```

**Description**: The gamma function extends the factorial to non-integer values: Γ(n+1) = n!

Uses Lanczos approximation for fast, accurate evaluation.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `x`: Input value (should not be 0 or negative integer)

**Returns**: Γ(x)

**Errors**:
- `error.DomainError`: if x ≤ 0 and x is an integer

**Time**: O(1)
**Space**: O(1)

**Accuracy**:
- f32: ≈5-7 significant digits
- f64: ≈14-15 significant digits

**Properties**:
- Γ(1) = 1
- Γ(n+1) = n! for non-negative integers n
- Γ(x+1) = x·Γ(x)
- Γ(1/2) = √π

**Example**:
```zig
const result = try special.gamma(f64, 5.0);  // Γ(5) = 4! = 24
const half = try special.gamma(f64, 0.5);    // Γ(0.5) = √π ≈ 1.772454
```

---

### beta(T, a, b)

Compute beta function B(a, b) = Γ(a)·Γ(b) / Γ(a+b)

```zig
pub fn beta(comptime T: type, a: T, b: T) SpecialFunctionError!T
```

**Description**: The beta function appears in beta distributions and binomial coefficients.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `a, b`: Shape parameters (should be positive)

**Returns**: B(a, b)

**Errors**:
- `error.DomainError`: if a ≤ 0 or b ≤ 0

**Time**: O(1)
**Space**: O(1)

**Properties**:
- B(a, b) = B(b, a) (symmetric)
- B(a, b) = (a-1)!(b-1)! / (a+b-1)! for positive integers
- Appears in beta distribution, binomial integrals

**Example**:
```zig
const result = try special.beta(f64, 2.0, 3.0);
// B(2, 3) = Γ(2)·Γ(3) / Γ(5) = 1·2 / 24 = 1/12 ≈ 0.0833
```

---

### erf(T, x)

Compute error function erf(x) = (2/√π)∫₀^x e^(-t²)dt

```zig
pub fn erf(comptime T: type, x: T) T
```

**Description**: The error function is fundamental to probability and statistics. Related to normal distribution CDF.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `x`: Input value

**Returns**: erf(x) in range [-1, 1]

**Time**: O(1)
**Space**: O(1)

**Properties**:
- erf(-x) = -erf(x) (odd function)
- erf(0) = 0
- erf(∞) = 1, erf(-∞) = -1
- Φ(x) = (1 + erf(x/√2)) / 2 (normal CDF)

**Accuracy**: Series expansion for small |x|, continued fraction for large |x|

**Example**:
```zig
const e1 = special.erf(f64, 0.0);     // 0
const e2 = special.erf(f64, 1.0);     // ≈ 0.8427
const e3 = special.erf(f64, -1.0);    // ≈ -0.8427
```

---

### erfc(T, x)

Compute complementary error function erfc(x) = 1 - erf(x)

```zig
pub fn erfc(comptime T: type, x: T) T
```

**Description**: Complementary error function, more numerically stable for large x.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `x`: Input value

**Returns**: erfc(x) in range [0, 2]

**Time**: O(1)
**Space**: O(1)

**Example**:
```zig
const e1 = special.erfc(f64, 0.0);     // 1
const e2 = special.erfc(f64, 1.0);     // ≈ 0.1573
```

---

### bessel_j(T, n, x)

Compute Bessel function of the first kind J_n(x)

```zig
pub fn bessel_j(comptime T: type, n: i32, x: T) SpecialFunctionError!T
```

**Description**: Bessel functions appear in wave equations, heat diffusion, and cylindrical harmonics.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `n`: Order (non-negative integer typically)
- `x`: Argument (can be any value)

**Returns**: J_n(x)

**Errors**:
- `error.DomainError`: for invalid argument ranges

**Time**: O(1)
**Space**: O(1)

**Properties**:
- J_n(0) = 1 if n=0, else 0
- J_n(x) oscillates between -1 and 1
- Related to trigonometric functions: J_0(x) related to cos/sin integrals

**Example**:
```zig
const j0 = try special.bessel_j(f64, 0, 1.0);   // J_0(1) ≈ 0.7652
const j1 = try special.bessel_j(f64, 1, 1.0);   // J_1(1) ≈ 0.4401
```

---

### bessel_y(T, n, x)

Compute Bessel function of the second kind Y_n(x) (Neumann function)

```zig
pub fn bessel_y(comptime T: type, n: i32, x: T) SpecialFunctionError!T
```

**Description**: Bessel function of the second kind, linearly independent from J_n.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `n`: Order (non-negative integer typically)
- `x`: Argument (must be > 0)

**Returns**: Y_n(x)

**Errors**:
- `error.DomainError`: if x ≤ 0 or invalid argument ranges

**Time**: O(1)
**Space**: O(1)

**Properties**:
- Y_n(0) = -∞ (singularity at origin)
- General solution: y = c₁·J_n(x) + c₂·Y_n(x)
- Y_n(x) → -∞ as x → 0

**Example**:
```zig
const y0 = try special.bessel_y(f64, 0, 1.0);   // Y_0(1) ≈ 0.6366
const y1 = try special.bessel_y(f64, 1, 1.0);   // Y_1(1) ≈ -0.6364
```

---

## Practical Examples

### Solving an ODE with Initial Condition

```zig
const std = @import("std");
const zuda = @import("zuda");
const numeric = zuda.numeric;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Solve dy/dt = -2ty with y(0) = 1
    // Analytical: y(t) = exp(-t²)
    fn dydt(t: f64, y: f64) f64 {
        _ = y;
        return -2.0 * t;  // simplified for this example
    }

    var sol = try numeric.ode.rk4(f64, dydt, 1.0, .{0.0, 2.0}, 0.01, allocator);
    defer sol.deinit();

    std.debug.print("t={d:.2}, y={d:.6}\n", .{sol.t[sol.t.len - 1], sol.y[sol.y.len - 1]});
    // Output: t=2.00, y=0.018316 (exp(-4) ≈ 0.018316)
}
```

---

### Root Finding with Multiple Methods

```zig
// Find root of f(x) = x³ - 2x - 5
fn cubic(x: f64) f64 {
    return x * x * x - 2.0 * x - 5.0;
}

fn cubic_deriv(x: f64) f64 {
    return 3.0 * x * x - 2.0;
}

const bisect_root = try root_finding.bisect(f64, cubic, 2.0, 3.0, 1e-6, 100);
const newton_root = try root_finding.newton(f64, cubic, cubic_deriv, 2.5, 1e-6, 100);
const brent_root = try root_finding.brent(f64, cubic, 2.0, 3.0, 1e-6, 100);

// All converge to same root ≈ 2.0946
```

---

### Fitting Exponential Data

```zig
// Fit y = a·exp(b·x)
fn exp_model(x: f64, params: []const f64) f64 {
    return params[0] * @exp(params[1] * x);
}

const x_data = [_]f64{0.0, 1.0, 2.0, 3.0};
const y_data = [_]f64{1.0, 2.7, 7.4, 20.1};
const p0 = [_]f64{1.0, 1.0};

var result = try curve_fitting.curve_fit(f64, exp_model, &x_data, &y_data, &p0, allocator);
defer allocator.free(result.params);
defer allocator.free(result.residuals);

// result.params ≈ [1.0, 1.0] with final_cost near 0
```

---

## Performance Considerations

1. **Step Size**: For ODE solvers, smaller dt increases accuracy but computation time
2. **Integration Accuracy**: Simpson's rule O(h⁴) vs Trapezoid O(h²); choose based on smoothness
3. **Root Finding**: Brent's method is best general-purpose; use bisection if guaranteed bracketing
4. **Curve Fitting**: Levenberg-Marquardt requires good initial guess for convergence
5. **Special Functions**: All O(1); use with confidence for repeated evaluation

---

## Numerical Stability Notes

- **Integration**: Larger grids more prone to accumulated rounding error; watch for oscillations
- **Differentiation**: Forward/backward O(h); central O(h²); smaller h can cause cancellation error
- **Root Finding**: Newton requires good initial guess; Brent combines robustness and speed
- **ODE Solvers**: Smaller dt → smaller local error but more steps → more rounding; RK45 adapts
- **Curve Fitting**: Normalize data to improve Jacobian conditioning; check convergence flag

---

## References

- Press et al. *Numerical Recipes in C* (2nd ed.). Cambridge University Press.
- Dormand & Prince. *A family of embedded Runge-Kutta formulae.* Journal of Computational and Applied Mathematics.
- Bevis & Thompson. *Mapping functions for the atmosphere and ionosphere.* J. Geophys. Res.
- Lanczos, C. *A precision approximation of the Gamma function.* Journal of the Society for Industrial and Applied Mathematics.
