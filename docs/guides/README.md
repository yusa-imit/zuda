# zuda Scientific Computing Guides

Welcome to the zuda scientific computing guide collection. These tutorials provide comprehensive examples and best practices for using zuda's v2.0 scientific computing modules.

## Quick Start

```zig
const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a matrix
    const NDArray = zuda.ndarray.NDArray;
    var A = try NDArray(f64, 2).rand(allocator, &.{100, 100}, .row_major, &std.rand.DefaultPrng.init(42).random());
    defer A.deinit();

    // Solve linear system
    var b = try NDArray(f64, 1).ones(allocator, &.{100}, .row_major);
    defer b.deinit();

    var x = try zuda.linalg.solve.solve(f64, A, b, allocator);
    defer x.deinit();

    std.debug.print("Solution computed!\n", .{});
}
```

## Module Guides

### Core Foundation

- **[NDArray](ndarray.md)** — N-dimensional arrays
  - Creating arrays (zeros, ones, random, from data)
  - Element-wise operations (arithmetic, math functions)
  - Shape manipulation (reshape, transpose, slice)
  - Reductions (sum, mean, max, min)
  - SIMD acceleration
  - NumPy compatibility

### Scientific Computing

- **[Linear Algebra](linalg.md)** — Matrix operations and decompositions
  - **BLAS Level 1**: Vector operations (dot, norm, axpy)
  - **BLAS Level 2**: Matrix-vector (gemv, symv, ger)
  - **BLAS Level 3**: Matrix-matrix (gemm, symm, syrk)
  - **Decompositions**: LU, QR, Cholesky, SVD, Eigendecomposition
  - **Solvers**: solve(), lstsq(), inv(), pinv()
  - **Properties**: rank(), cond()

- **[Statistics](stats.md)** — Statistical computing and analysis
  - **Descriptive**: mean, median, mode, variance, std, quantiles, skewness, kurtosis
  - **Distributions**: Normal, Uniform, Exponential, Gamma, Beta, Binomial, Poisson, Chi-Squared, Student's t, F
  - **Hypothesis Testing**: t-tests, chi-squared tests, ANOVA, Kolmogorov-Smirnov
  - **Correlation & Regression**: Pearson, Spearman, linear regression, multiple regression

- **[Signal Processing](signal.md)** — FFT, filtering, and transforms
  - **FFT**: Complex FFT, inverse FFT, real FFT (optimized)
  - **Spectral Analysis**: Power spectral density, periodogram
  - **Windowing**: Hann, Hamming, Blackman, Bartlett
  - **Convolution**: Linear, circular, FFT-based
  - **Applications**: Frequency analysis, filtering, spectrogram, cross-correlation

- **[Numerical Methods](numeric.md)** — Integration, differentiation, and root finding
  - **Integration**: Trapezoidal, Simpson's, Romberg, Gaussian quadrature
  - **Differentiation**: Forward, backward, central, second derivative, gradient
  - **Root Finding**: Bisection, Newton's, secant, Brent's, fixed-point iteration
  - **Interpolation**: Linear, Lagrange, cubic spline, nearest-neighbor
  - **ODEs**: Euler, Runge-Kutta 4th order, systems of ODEs

- **[Optimization](optimize.md)** — Nonlinear optimization and least squares
  - **Line Search**: Backtracking, strong Wolfe
  - **Unconstrained**: Gradient descent, conjugate gradient, BFGS, L-BFGS, Nelder-Mead
  - **Constrained**: Penalty method, augmented Lagrangian
  - **Linear Programming**: Simplex, interior point
  - **Least Squares**: Gauss-Newton, Levenberg-Marquardt

## Integration Examples

### Machine Learning Pipeline

```zig
// 1. Load and normalize data
var X = try NDArray(f64, 2).fromSlice(allocator, data, .row_major);
defer X.deinit();

const mean_val = X.mean();
var X_centered = try X.addScalar(-mean_val);
defer X_centered.deinit();

const std_val = @sqrt(X_centered.mul(&X_centered).mean());
var X_normalized = try X_centered.mulScalar(1.0 / std_val);
defer X_normalized.deinit();

// 2. Fit model with optimization
const optimize = zuda.optimize;
var params = try optimize.unconstrained.lbfgs(f64, loss_fn, grad_fn, &initial_params, options, allocator);
defer allocator.free(params.x);

// 3. Evaluate with statistics
const stats = zuda.stats;
const predictions = predict(params.x, X_normalized);
const r_squared = compute_r_squared(predictions, y_true);
```

### Signal Analysis Workflow

```zig
// 1. Load time-series data
var signal = try std.ArrayList(f64).initCapacity(allocator, n_samples);
defer signal.deinit(allocator);
// ... load data ...

// 2. Apply window
const window = try zuda.signal.windows.hann(f64, n_samples, allocator);
defer allocator.free(window);

for (signal.items, 0..) |*val, i| {
    val.* *= window[i];
}

// 3. Compute spectrum
const spectrum = try zuda.signal.fft.rfft(f64, signal.items, allocator);
defer allocator.free(spectrum);

// 4. Statistical analysis of power
var power = try std.ArrayList(f64).initCapacity(allocator, spectrum.len);
defer power.deinit();

for (spectrum) |coef| {
    try power.append(coef.magnitude() * coef.magnitude());
}

const mean_power = try zuda.stats.descriptive.mean(f64, power.items, 0);
const std_power = try zuda.stats.descriptive.stdDev(f64, power.items, 0);
```

### Numerical Simulation

```zig
// 1. Set up ODE system (Lotka-Volterra)
fn predator_prey(t: f64, y: []const f64, dydt: []f64) void {
    const prey = y[0];
    const predator = y[1];
    dydt[0] = 1.0 * prey - 0.1 * prey * predator;
    dydt[1] = 0.075 * prey * predator - 1.5 * predator;
}

// 2. Solve ODE
const y0 = [_]f64{10.0, 5.0};  // Initial populations
var result = try zuda.numeric.ode.rk4_system(f64, predator_prey, 0.0, &y0, 100.0, 10000, allocator);
defer allocator.free(result.t);
defer result.y.deinit();

// 3. Statistical analysis of populations
var prey_pop = try std.ArrayList(f64).initCapacity(allocator, result.y.items.len);
defer prey_pop.deinit();

for (result.y.items) |state| {
    try prey_pop.append(state[0]);
}

const stats = zuda.stats.descriptive;
const mean_prey = try stats.mean(f64, prey_pop.items, 0);
const max_prey = stats.max(f64, prey_pop.items);
const min_prey = stats.min(f64, prey_pop.items);
```

### Linear System Analysis

```zig
// 1. Construct system matrix
var A = try NDArray(f64, 2).eye(allocator, n, n, 0, .row_major);
defer A.deinit();

// Add off-diagonal elements
for (0..n-1) |i| {
    A.set(&.{@as(isize, @intCast(i)), @as(isize, @intCast(i+1))}, -0.5);
    A.set(&.{@as(isize, @intCast(i+1)), @as(isize, @intCast(i))}, -0.5);
}

// 2. Analyze properties
const linalg = zuda.linalg;
const rank = try linalg.properties.rank(f64, A, allocator);
const cond = try linalg.properties.cond(f64, A, allocator);

std.debug.print("Rank: {}, Condition: {e}\n", .{rank, cond});

// 3. Solve system
var b = try NDArray(f64, 1).rand(allocator, &.{n}, .row_major, &rng.random());
defer b.deinit();

var x = try linalg.solve.solve(f64, A, b, allocator);
defer x.deinit();

// 4. Verify solution
var Ax = try NDArray(f64, 1).zeros(allocator, &.{n}, .row_major);
defer Ax.deinit();
try linalg.blas.gemv(f64, 1.0, A, x, 0.0, &Ax);

// Compute residual
var residual = try Ax.sub(&b);
defer residual.deinit();
const residual_norm = try linalg.blas.nrm2(f64, residual);
```

## Performance Best Practices

### Memory Management

```zig
// ✅ GOOD: Explicit cleanup
var A = try NDArray(f64, 2).zeros(allocator, &.{1000, 1000}, .row_major);
defer A.deinit();  // Always defer immediately after creation

// ❌ BAD: Forgetting defer causes memory leak
var B = try NDArray(f64, 2).zeros(allocator, &.{1000, 1000}, .row_major);
// Missing defer!
```

### Choosing Algorithms

```zig
// For symmetric positive-definite systems, use Cholesky (fastest)
if (is_symmetric_positive_definite(A)) {
    var chol = try linalg.decompositions.cholesky(f64, allocator, A);
    defer chol.L.deinit();
    // Use L for solving
}

// For general systems, use LU
else {
    var lu = try linalg.decompositions.lu(f64, allocator, A);
    defer lu.L.deinit();
    defer lu.U.deinit();
    defer lu.P.deinit();
    // Use LU for solving
}
```

### SIMD Optimization

```zig
// Large arrays automatically use SIMD for element-wise operations
var A = try NDArray(f64, 2).ones(allocator, &.{10000, 10000}, .row_major);
defer A.deinit();

var B = try NDArray(f64, 2).ones(allocator, &.{10000, 10000}, .row_major);
defer B.deinit();

// This will use SIMD automatically
var C = try A.add(&B);
defer C.deinit();
```

### Batch Operations

```zig
// ✅ GOOD: Single allocation for result
var result = try A.add(&B);
defer result.deinit();

// ❌ BAD: Multiple intermediate allocations
var temp1 = try A.add(&B);
defer temp1.deinit();
var temp2 = try temp1.mul(&C);
defer temp2.deinit();
var final = try temp2.sub(&D);
defer final.deinit();

// ✅ BETTER: Reuse buffers when possible (in-place operations)
// (Future optimization opportunity)
```

## Common Pitfalls

### Type Mismatches

```zig
// ❌ WRONG: Missing ndim parameter
var A = try NDArray(f64).zeros(allocator, &.{10, 10}, .row_major);  // Error!

// ✅ CORRECT: Specify ndim
var A = try NDArray(f64, 2).zeros(allocator, &.{10, 10}, .row_major);
```

### Index Types

```zig
const i: usize = 5;
const j: usize = 3;

// ❌ WRONG: usize indices
A.set(&.{i, j}, value);  // Type error!

// ✅ CORRECT: Cast to isize
A.set(&.{@as(isize, @intCast(i)), @as(isize, @intCast(j))}, value);
```

### Shape Mismatches

```zig
var A = try NDArray(f64, 2).zeros(allocator, &.{3, 4}, .row_major);
defer A.deinit();

var B = try NDArray(f64, 2).zeros(allocator, &.{2, 2}, .row_major);
defer B.deinit();

// ❌ WRONG: Incompatible shapes
var C = try A.add(&B);  // Error: ShapeMismatch

// ✅ CORRECT: Ensure compatible shapes
var B_correct = try NDArray(f64, 2).zeros(allocator, &.{3, 4}, .row_major);
defer B_correct.deinit();
var C = try A.add(&B_correct);  // OK
defer C.deinit();
```

### Power-of-2 FFT

```zig
// ❌ WRONG: Non-power-of-2 length
var signal = try std.ArrayList(Complex(f64)).initCapacity(allocator, 1000);
const spectrum = try fft.fft(f64, signal.items, allocator);  // Error!

// ✅ CORRECT: Pad to power of 2
var signal = try std.ArrayList(Complex(f64)).initCapacity(allocator, 1024);  // 2^10
defer signal.deinit(allocator);
// ... fill signal ...
const spectrum = try fft.fft(f64, signal.items, allocator);  // OK
defer allocator.free(spectrum);
```

## Migration from NumPy/SciPy

See [NumPy Compatibility Guide](../NUMPY_COMPATIBILITY.md) for detailed API mappings:

| NumPy/SciPy | zuda |
|-------------|------|
| `numpy.array()` | `NDArray(T, ndim).fromSlice()` |
| `numpy.zeros()` | `NDArray(T, ndim).zeros()` |
| `numpy.linalg.solve()` | `linalg.solve.solve()` |
| `scipy.linalg.svd()` | `linalg.decompositions.svd()` |
| `scipy.fft.fft()` | `signal.fft.fft()` |
| `scipy.stats.norm()` | `stats.distributions.Normal()` |
| `scipy.optimize.minimize()` | `optimize.unconstrained.lbfgs()` |
| `scipy.integrate.quad()` | `numeric.integration.quad()` |

## Additional Resources

- **[API Reference](../API.md)** — Complete function signatures
- **[Getting Started](../GETTING_STARTED.md)** — Installation and setup
- **[PRD](../PRD.md)** — Project roadmap and requirements
- **[Milestones](../milestones.md)** — Version history and progress

## Contributing

Found an issue or want to improve these guides? See the [main repository](https://github.com/yusa-imit/zuda) for contribution guidelines.

## License

zuda is released under the MIT License. See LICENSE file in the repository root.
