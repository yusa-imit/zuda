# Scientific Computing with zuda — Getting Started Guide

> **zuda v2.0** is a Zig-native scientific computing platform providing NumPy/SciPy-like APIs with compile-time safety, explicit memory management, and competitive performance.

---

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Module Overview](#module-overview)
5. [Complete Tutorials](#complete-tutorials)
6. [Performance Tips](#performance-tips)
7. [Migration Guides](#migration-guides)

---

## Installation & Setup

### Adding zuda to Your Project

**1. Add to `build.zig.zon`:**

```zig
.{
    .name = "my-scientific-app",
    .version = "0.1.0",
    .dependencies = .{
        .zuda = .{
            .url = "https://github.com/yusa-imit/zuda/archive/refs/tags/v2.0.0.tar.gz",
            // Replace hash with actual hash from zig fetch
            .hash = "1220...",
        },
    },
}
```

**2. Configure `build.zig`:**

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Get zuda dependency
    const zuda = b.dependency("zuda", .{
        .target = target,
        .optimize = optimize,
    });

    // Your executable
    const exe = b.addExecutable(.{
        .name = "my-app",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Link zuda
    exe.root_module.addImport("zuda", zuda.module("zuda"));
    b.installArtifact(exe);
}
```

**3. Import in your code:**

```zig
const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Use zuda modules
    var arr = try zuda.ndarray.zeros(f64, 2, &[_]usize{3, 3}, allocator);
    defer arr.deinit();

    std.debug.print("Created 3×3 array\n", .{});
}
```

---

## Quick Start

### Hello, NDArray!

```zig
const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a 2×3 matrix
    var A = try zuda.ndarray.fromSlice(f64, 2, &[_]usize{2, 3},
        &[_]f64{1, 2, 3, 4, 5, 6}, allocator);
    defer A.deinit();

    // Element-wise operations
    var B = try A.mul_scalar(2.0, allocator);  // Multiply by 2
    defer B.deinit();

    // Reductions
    const sum = A.sum();  // Sum all elements
    const mean = A.mean();  // Mean of all elements

    std.debug.print("Original:\n", .{});
    A.print();
    std.debug.print("Doubled:\n", .{});
    B.print();
    std.debug.print("Sum: {d}, Mean: {d}\n", .{sum, mean});
}
```

**Output**:
```
Original:
[[1.0, 2.0, 3.0],
 [4.0, 5.0, 6.0]]
Doubled:
[[2.0, 4.0, 6.0],
 [8.0, 10.0, 12.0]]
Sum: 21.0, Mean: 3.5
```

---

## Core Concepts

### 1. Explicit Memory Management

zuda uses **allocator-first design** — every operation that allocates memory requires an `std.mem.Allocator`.

```zig
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
defer _ = gpa.deinit();
const allocator = gpa.allocator();

var arr = try zuda.ndarray.zeros(f64, 2, &[_]usize{100, 100}, allocator);
defer arr.deinit();  // MUST free memory explicitly
```

**Golden rule**: Every `init()` / allocation must have a corresponding `defer` with `deinit()`.

### 2. Compile-Time Rank, Runtime Shape

Arrays are typed as `NDArray(T, ndim)`:
- **Element type `T`**: `f64`, `f32`, `i32`, etc. (compile-time)
- **Rank `ndim`**: Number of dimensions (compile-time)
- **Shape**: Runtime-known sizes (e.g., `&[_]usize{3, 4}` = 3×4)

```zig
var matrix = try zuda.ndarray.zeros(f64, 2, &[_]usize{3, 4}, allocator);  // 3×4 f64 matrix
var vector = try zuda.ndarray.ones(i32, 1, &[_]usize{100}, allocator);    // 100-element i32 vector
var tensor = try zuda.ndarray.zeros(f32, 3, &[_]usize{10, 20, 30}, allocator);  // 10×20×30 f32 tensor
```

### 3. Row-Major Memory Layout (C-style)

zuda uses row-major (C-style) layout by default, matching NumPy's default.

```zig
// 2×3 matrix [[1, 2, 3], [4, 5, 6]]
// Memory layout: [1, 2, 3, 4, 5, 6]
//                 -------  --------
//                  row 0    row 1
```

### 4. Error Handling

All fallible operations return error unions (`!T`). Use `try` to propagate errors:

```zig
var arr = try zuda.ndarray.zeros(f64, 2, &[_]usize{1000, 1000}, allocator);
defer arr.deinit();

// Or handle errors explicitly:
const result = zuda.linalg.solve(f64, A, b, allocator) catch |err| {
    std.debug.print("Solve failed: {}\n", .{err});
    return err;
};
defer result.deinit();
```

---

## Module Overview

### `zuda.ndarray` — N-Dimensional Arrays

**Purpose**: Core data structure for multi-dimensional numerical data

**Key Functions**:
- **Creation**: `zeros()`, `ones()`, `full()`, `arange()`, `linspace()`, `eye()`, `fromSlice()`
- **Indexing**: `get()`, `set()`, `slice()` (creates views)
- **Element-wise**: `add()`, `sub()`, `mul()`, `div()`, `pow()`, `sin()`, `cos()`, `exp()`, `log()`, `sqrt()`
- **Reductions**: `sum()`, `prod()`, `mean()`, `min()`, `max()`, `sum_axis()`, `mean_axis()`
- **Transformations**: `reshape()`, `transpose()`, `permute()`, `flatten()`, `squeeze()`, `unsqueeze()`
- **I/O**: `save()`, `load()`, `fromCSV()`, `toCSV()`

**Example**: See [NDArray Guide](guides/ndarray.md)

### `zuda.linalg` — Linear Algebra

**Purpose**: BLAS operations, matrix decompositions, linear solvers

**Key Functions**:
- **BLAS Level 1**: `dot()`, `axpy()`, `nrm2()`, `asum()`, `scal()`
- **BLAS Level 2**: `gemv()` (matrix-vector), `ger()` (outer product)
- **BLAS Level 3**: `gemm()` (matrix-matrix multiply)
- **Solvers**: `solve()` (Ax=b), `lstsq()` (least squares), `inv()` (inverse), `pinv()` (pseudo-inverse)
- **Decompositions**: `lu()`, `qr()`, `cholesky()`, `svd()`, `eig()` (eigendecomposition)
- **Properties**: `det()` (determinant), `trace()`, `rank()`, `cond()` (condition number)

**Example**: See [Linear Algebra Guide](guides/linalg.md)

### `zuda.stats` — Statistics & Distributions

**Purpose**: Descriptive statistics, hypothesis testing, distributions, random number generation

**Key Functions**:
- **Descriptive**: `mean()`, `median()`, `mode()`, `std()`, `var()`, `quantile()`, `skewness()`, `kurtosis()`
- **Distributions**: `Normal`, `Uniform`, `Exponential`, `Poisson`, `Binomial`, `Gamma`, `Beta`, `ChiSquared`, `StudentT`
  - Each has: `.pdf()`, `.cdf()`, `.quantile()`, `.sample()`
- **Hypothesis Tests**: `ttest_1samp()`, `ttest_ind()`, `ttest_rel()`, `chi2_test()`, `anova_oneway()`, `ks_test()`, `mannwhitney_u()`
- **Correlation**: `pearsonr()`, `spearmanr()`, `kendalltau()`, `covarianceMatrix()`, `crossCorrelation()`
- **Regression**: `ols()` (ordinary least squares), `polyfit()`
- **Random**: `PCG64`, `Xoshiro256**` PRNGs + `uniform()`, `normal()`, `shuffle()`, `choice()`

**Example**: See [Statistics Guide](guides/stats.md)

### `zuda.signal` — Signal Processing

**Purpose**: FFT, filtering, convolution, spectral analysis

**Key Functions**:
- **FFT**: `fft()`, `ifft()`, `rfft()`, `irfft()`, `fft2()`, `ifft2()`, `fftfreq()`
- **DCT**: `dct()`, `idct()`
- **Convolution**: `convolve()`, `fftconvolve()`, `correlate()`, `convolve2d()`
- **Filtering**: `lfilter()`, `filtfilt()`, `butter()`, `cheby1()`, `firwin()`
- **Spectral**: `periodogram()`, `welch()`
- **Windows**: `hamming()`, `hann()`, `blackman()`, `kaiser()`, `bartlett()`

**Example**: See [Signal Processing Guide](guides/signal.md)

### `zuda.numeric` — Numerical Methods

**Purpose**: Integration, differentiation, interpolation, ODE solvers, root finding

**Key Functions**:
- **Integration**: `trapezoid()`, `simpson()`, `quad()`, `romberg()`, `gauss_legendre()`
- **Differentiation**: `diff()`, `gradient()`, `jacobian()`, `hessian()`
- **Interpolation**: `interp1d()`, `cubic_spline()`, `lagrange()`, `pchip()`, `interp2d()`
- **Root Finding**: `bisect()`, `newton()`, `brent()`, `secant()`, `fixed_point()`
- **ODE Solvers**: `euler()`, `rk4()`, `rk45()`, `bdf()`
- **Special Functions**: `gamma()`, `lgamma()`, `beta()`, `erf()`, `erfc()`, `bessel_j()`, `bessel_y()`

**Example**: See [Numerical Methods Guide](guides/numeric.md)

### `zuda.optimize` — Optimization

**Purpose**: Unconstrained/constrained optimization, curve fitting

**Key Functions**:
- **Gradient-based**: `gradient_descent()`, `conjugate_gradient()`, `lbfgs()`, `nelder_mead()`
- **Auto-differentiation**: `Dual(T)`, `autodiff.gradient()`, `autodiff.jacobian()`
- **Line Search**: `armijo()`, `wolfe()`, `backtracking()`
- **Linear Programming**: `simplex()`, `interior_point()`
- **Constrained**: `augmented_lagrangian()`, `quadratic_programming()`
- **Nonlinear Least Squares**: `levenberg_marquardt()`, `gauss_newton()`

**Example**: See [Optimization Guide](guides/optimize.md)

---

## Complete Tutorials

### Tutorial 1: Data Analysis Pipeline

**Scenario**: Analyze sensor data (temperature readings), compute statistics, detect anomalies

```zig
const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Simulate sensor data (100 readings)
    var rng = zuda.stats.random.PCG64.init(42);
    var data = try zuda.stats.random.normal(f64, &rng, 25.0, 2.0, 100, allocator);
    defer data.deinit();

    // Add some outliers
    data.set(&[_]usize{10}, 40.0);  // Anomaly
    data.set(&[_]usize{50}, 10.0);  // Anomaly

    // Compute descriptive statistics
    const raw_data = data.data[0..data.count()];
    const mean = zuda.stats.mean(f64, raw_data);
    const std_dev = zuda.stats.std(f64, raw_data, 1);
    const median = try zuda.stats.median(f64, raw_data, allocator);

    std.debug.print("Temperature Statistics:\n", .{});
    std.debug.print("  Mean: {d:.2}°C\n", .{mean});
    std.debug.print("  Median: {d:.2}°C\n", .{median});
    std.debug.print("  Std Dev: {d:.2}°C\n", .{std_dev});

    // Detect anomalies (> 3 std deviations from mean)
    const threshold = 3.0 * std_dev;
    std.debug.print("\nAnomalies detected:\n", .{});
    for (raw_data, 0..) |value, i| {
        if (@abs(value - mean) > threshold) {
            std.debug.print("  Index {d}: {d:.2}°C (deviation: {d:.2}°C)\n",
                .{i, value, value - mean});
        }
    }

    // Histogram binning
    var hist = try zuda.stats.histogram(f64, raw_data, 10, null, allocator);
    defer hist.counts.deinit();
    defer hist.edges.deinit();

    std.debug.print("\nHistogram (10 bins):\n", .{});
    const counts = hist.counts.data[0..hist.counts.count()];
    const edges = hist.edges.data[0..hist.edges.count()];
    for (counts, 0..) |count, i| {
        std.debug.print("  [{d:.1} - {d:.1}): {d} readings\n",
            .{edges[i], edges[i+1], count});
    }
}
```

### Tutorial 2: Linear Regression

**Scenario**: Fit a linear model to noisy data, evaluate fit quality

```zig
const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Generate synthetic data: y = 2x + 3 + noise
    const n = 50;
    var rng = zuda.stats.random.PCG64.init(42);

    var x = try zuda.ndarray.linspace(f64, 0.0, 10.0, n, allocator);
    defer x.deinit();

    var noise = try zuda.stats.random.normal(f64, &rng, 0.0, 1.0, n, allocator);
    defer noise.deinit();

    // y = 2x + 3 + noise
    var y = try x.mul_scalar(2.0, allocator);
    defer y.deinit();
    var y_temp = try y.add_scalar(3.0, allocator);
    y.deinit();
    y = y_temp;
    y_temp = try y.add(noise, allocator);
    y.deinit();
    y = y_temp;

    // Fit linear model
    const x_data = x.data[0..x.count()];
    const y_data = y.data[0..y.count()];
    var result = try zuda.stats.ols(f64, x_data, y_data, allocator);
    defer result.coefficients.deinit();
    defer result.residuals.deinit();

    const coeffs = result.coefficients.data[0..2];
    std.debug.print("Linear Fit: y = {d:.4}x + {d:.4}\n", .{coeffs[1], coeffs[0]});
    std.debug.print("R²: {d:.4}\n", .{result.r_squared});
    std.debug.print("Residual std: {d:.4}\n", .{zuda.stats.std(f64, result.residuals.data[0..n], 1)});

    // Prediction
    const x_new = 5.0;
    const y_pred = coeffs[0] + coeffs[1] * x_new;
    std.debug.print("\nPrediction at x={d}: y={d:.2}\n", .{x_new, y_pred});
}
```

### Tutorial 3: Image Filtering (2D Convolution)

**Scenario**: Apply Gaussian blur to a 2D image

```zig
const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a simple 8×8 "image" with a bright spot
    var image = try zuda.ndarray.zeros(f64, 2, &[_]usize{8, 8}, allocator);
    defer image.deinit();

    // Add a bright 2×2 square in the center
    image.set(&[_]usize{3, 3}, 100.0);
    image.set(&[_]usize{3, 4}, 100.0);
    image.set(&[_]usize{4, 3}, 100.0);
    image.set(&[_]usize{4, 4}, 100.0);

    std.debug.print("Original image:\n", .{});
    image.print();

    // Create 3×3 Gaussian kernel (approximation)
    const kernel_data = [_]f64{
        1.0/16, 2.0/16, 1.0/16,
        2.0/16, 4.0/16, 2.0/16,
        1.0/16, 2.0/16, 1.0/16,
    };
    var kernel = try zuda.ndarray.fromSlice(f64, 2, &[_]usize{3, 3}, &kernel_data, allocator);
    defer kernel.deinit();

    // Apply convolution (blurring)
    var blurred = try zuda.signal.convolve2d(f64, image, kernel, .same, allocator);
    defer blurred.deinit();

    std.debug.print("\nBlurred image:\n", .{});
    blurred.print();
}
```

### Tutorial 4: FFT-based Signal Analysis

**Scenario**: Analyze frequency components of a composite signal

```zig
const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const n = 1024;
    const sample_rate = 1000.0;  // Hz
    const t_data = try allocator.alloc(f64, n);
    defer allocator.free(t_data);

    // Time vector
    for (t_data, 0..) |*t, i| {
        t.* = @as(f64, @floatFromInt(i)) / sample_rate;
    }

    // Composite signal: 50 Hz + 120 Hz
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);
    const pi = std.math.pi;
    for (signal, 0..) |*s, i| {
        const t = t_data[i];
        s.* = @sin(2.0 * pi * 50.0 * t) + 0.5 * @sin(2.0 * pi * 120.0 * t);
    }

    var signal_arr = try zuda.ndarray.fromSlice(f64, 1, &[_]usize{n}, signal, allocator);
    defer signal_arr.deinit();

    // Compute FFT
    var spectrum = try zuda.signal.fft(f64, signal_arr, allocator);
    defer spectrum.deinit();

    // Compute frequency bins
    var freqs = try zuda.signal.fftfreq(n, 1.0 / sample_rate, allocator);
    defer freqs.deinit();

    // Find peaks (simplified: just print top 5 magnitudes)
    std.debug.print("Top 5 frequency components:\n", .{});
    const spec_data = spectrum.data[0..n];
    const freq_data = freqs.data[0..n];

    // Simple peak detection: print first half (positive frequencies)
    var peaks = std.ArrayList(struct { freq: f64, mag: f64 }).init(allocator);
    defer peaks.deinit();

    for (0..n/2) |i| {
        const real = spec_data[i * 2];
        const imag = spec_data[i * 2 + 1];
        const magnitude = @sqrt(real * real + imag * imag);
        if (magnitude > 10.0) {  // Threshold
            try peaks.append(.{ .freq = freq_data[i], .mag = magnitude });
        }
    }

    for (peaks.items[0..@min(5, peaks.items.len)]) |peak| {
        std.debug.print("  {d:.1} Hz: magnitude {d:.1}\n", .{peak.freq, peak.mag});
    }
}
```

### Tutorial 5: Optimization — Least Squares Curve Fitting

**Scenario**: Fit an exponential decay model to experimental data

```zig
const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Experimental data: exponential decay y = a * exp(-b * t)
    const t_data = [_]f64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const y_data = [_]f64{10.0, 7.5, 5.6, 4.2, 3.2, 2.4, 1.8, 1.4, 1.0, 0.8};

    // Objective function: sum of squared residuals
    const ObjectiveFn = struct {
        t: []const f64,
        y: []const f64,

        pub fn call(self: @This(), params: []const f64) f64 {
            const a = params[0];
            const b = params[1];
            var sse: f64 = 0.0;
            for (self.t, self.y) |t, y| {
                const y_pred = a * @exp(-b * t);
                const residual = y - y_pred;
                sse += residual * residual;
            }
            return sse;
        }
    };

    const objective = ObjectiveFn{ .t = &t_data, .y = &y_data };

    // Initial guess: a=10, b=0.2
    var params = [_]f64{10.0, 0.2};

    // Optimize using L-BFGS
    var result = try zuda.optimize.lbfgs(f64, objective.call, &params, .{
        .max_iterations = 100,
        .tolerance = 1e-6,
    }, allocator);
    defer result.x.deinit();

    const fitted = result.x.data[0..2];
    std.debug.print("Fitted model: y = {d:.4} * exp(-{d:.4} * t)\n", .{fitted[0], fitted[1]});
    std.debug.print("Final SSE: {d:.6}\n", .{result.final_value});

    // Predict at t=10
    const t_new = 10.0;
    const y_pred = fitted[0] * @exp(-fitted[1] * t_new);
    std.debug.print("Prediction at t={d}: y={d:.4}\n", .{t_new, y_pred});
}
```

---

## Performance Tips

### 1. Use Appropriate Allocators

```zig
// General-purpose (default)
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
defer _ = gpa.deinit();
const allocator = gpa.allocator();

// Arena for temporary allocations (fast, bulk dealloc)
var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
defer arena.deinit();
const temp_allocator = arena.allocator();
```

### 2. Reuse Buffers (In-Place Operations)

Instead of:
```zig
var a = try arr.add(b, allocator);  // Allocates new array
```

Consider pre-allocating output buffer and using BLAS:
```zig
var c = try zuda.ndarray.zeros(f64, 2, &[_]usize{n, m}, allocator);
// Use gemm with α=1, β=1 to accumulate: C = 1*A*B + 1*C
```

### 3. Batch Operations

Instead of looping over individual elements:
```zig
// BAD: Element-wise loop
for (0..arr.count()) |i| {
    arr.data[i] = arr.data[i] * 2.0;
}
```

Use vectorized operations:
```zig
// GOOD: Vectorized operation
var doubled = try arr.mul_scalar(2.0, allocator);
defer doubled.deinit();
```

### 4. Contiguous Memory Access

Ensure arrays are contiguous for cache-friendly access:
```zig
if (!arr.isContiguous()) {
    var contig = try arr.contiguous(allocator);
    defer contig.deinit();
    // Use contig for heavy computation
}
```

### 5. Choose Appropriate Solvers

- **SPD matrices**: Use `cholesky()` (fastest for positive-definite)
- **General square matrices**: Use `solve()` (auto-selects LU)
- **Overdetermined systems**: Use `lstsq()` (QR decomposition)
- **Eigenvalues only**: Cheaper than full eigendecomposition

---

## Migration Guides

Migrating from other scientific computing platforms? See our detailed guides:

- **[From NumPy (Python)](migrations/FROM_NUMPY.md)** — NumPy users: syntax comparisons, memory management, broadcasting
- **[From Eigen (C++)](migrations/FROM_EIGEN.md)** — Eigen users: template metaprogramming → comptime, RAII → defer
- **[From MATLAB](migrations/FROM_MATLAB.md)** — MATLAB users: 1-indexed → 0-indexed, workspace → allocator, backslash operator

---

## Further Reading

### Module-Specific Guides
- [NDArray Guide](guides/ndarray.md) — Comprehensive NDArray API reference
- [Linear Algebra Guide](guides/linalg.md) — BLAS, decompositions, solvers
- [Statistics Guide](guides/stats.md) — Distributions, hypothesis tests, regression
- [Signal Processing Guide](guides/signal.md) — FFT, filtering, convolution
- [Numerical Methods Guide](guides/numeric.md) — Integration, ODE solvers, interpolation
- [Optimization Guide](guides/optimize.md) — Gradient descent, L-BFGS, constrained optimization

### Reference
- [NumPy Compatibility Reference](NUMPY_COMPATIBILITY.md) — 50+ function mappings
- [Benchmark Results](BENCHMARKS.md) — Performance comparison with NumPy, SciPy, Eigen, MATLAB
- [API Documentation](API.md) — Full public API reference

### Community
- [GitHub Repository](https://github.com/yusa-imit/zuda) — Source code, issues, discussions
- [Examples Directory](../examples/) — Runnable examples for all modules

---

## Next Steps

1. **Install zuda** in your project (see [Installation](#installation--setup))
2. **Run a tutorial** from this guide (start with Tutorial 1 or 2)
3. **Explore module guides** for deep dives into specific domains
4. **Check migration guides** if coming from NumPy/MATLAB/Eigen
5. **Read examples** in `examples/` directory for real-world patterns

**Happy scientific computing with Zig! 🚀**
