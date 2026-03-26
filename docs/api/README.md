# zuda v2.0 API Reference

> Complete API reference for zuda's scientific computing platform

## Overview

zuda v2.0 provides a comprehensive scientific computing platform for Zig, covering:
- **NDArray**: N-dimensional arrays with broadcasting and slicing
- **Linear Algebra**: BLAS operations, decompositions, solvers
- **Statistics**: Distributions, hypothesis testing, regression
- **Signal Processing**: FFT, convolution, filtering
- **Numerical Methods**: Integration, differentiation, interpolation, root-finding
- **Optimization**: Unconstrained, constrained, least squares, linear programming

---

## Module Index

| Module | Description | Reference |
|--------|-------------|-----------|
| [`ndarray`](ndarray.md) | N-dimensional arrays | NDArray construction, indexing, slicing, broadcasting |
| [`linalg`](linalg.md) | Linear algebra | BLAS, decompositions, solvers, properties |
| [`stats`](stats.md) | Statistics | Descriptive stats, distributions, hypothesis tests, regression |
| [`signal`](signal.md) | Signal processing | FFT, RFFT, convolution, filtering |
| [`numeric`](numeric.md) | Numerical methods | Integration, differentiation, interpolation, root-finding, ODE |
| [`optimize`](optimize.md) | Optimization | Line search, gradient descent, constrained optimization, LP |

---

## Quick Start

```zig
const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a 2D array
    var A = try zuda.ndarray.NDArray(f64, 2).zeros(allocator, &.{ 3, 3 }, .row_major);
    defer A.deinit();

    // Perform linear algebra
    const blas = zuda.linalg.blas;
    var x = try zuda.ndarray.NDArray(f64, 1).ones(allocator, &.{3}, .row_major);
    defer x.deinit();

    var y = try zuda.ndarray.NDArray(f64, 1).zeros(allocator, &.{3}, .row_major);
    defer y.deinit();

    try blas.gemv(f64, 1.0, A, x, 0.0, &y);

    // Statistics
    const data = [_]f64{ 1, 2, 3, 4, 5 };
    const mean = try zuda.stats.descriptive.mean(f64, &data);
    const std_dev = try zuda.stats.descriptive.stdDev(f64, &data, 0);

    // Optimization
    const rosenbrock = struct {
        fn eval(_: *anyopaque, x: []const f64) f64 {
            const a = 1.0 - x[0];
            const b = x[1] - x[0] * x[0];
            return a * a + 100.0 * b * b;
        }
    }.eval;

    var x0 = [_]f64{ 0, 0 };
    const result = try zuda.optimize.unconstrained.bfgs(
        f64,
        rosenbrock,
        null,
        &x0,
        .{},
        allocator,
    );
    defer allocator.free(result.x);
}
```

---

## Common Patterns

### Memory Management

All allocating functions require an `std.mem.Allocator`. Caller owns returned memory:

```zig
// NDArray ownership
var A = try NDArray(f64, 2).zeros(allocator, &.{ 100, 100 }, .row_major);
defer A.deinit();  // Free when done

// Decomposition results
var lu_result = try linalg.decompositions.lu(f64, A, allocator);
defer lu_result.L.deinit();
defer lu_result.U.deinit();
defer lu_result.P.deinit();

// Optimization results
const opt_result = try optimize.unconstrained.bfgs(f64, fn_ptr, null, &x0, .{}, allocator);
defer allocator.free(opt_result.x);  // Slice, use allocator.free()
```

### Type Parameters

Most functions are generic over floating-point types (`f32`, `f64`):

```zig
// f64 (default for most use cases)
var A_f64 = try NDArray(f64, 2).eye(allocator, 10, 10, 0, .row_major);

// f32 (for memory-constrained or performance-critical code)
var A_f32 = try NDArray(f32, 2).eye(allocator, 10, 10, 0, .row_major);
```

### Error Handling

All fallible operations return error unions:

```zig
const SolveError = error{ SingularMatrix, DimensionMismatch, OutOfMemory };

// Handle errors explicitly
const x = linalg.solve.solve(f64, A, b, allocator) catch |err| switch (err) {
    error.SingularMatrix => {
        std.debug.print("Matrix is singular\n", .{});
        return err;
    },
    else => return err,
};
defer allocator.free(x);
```

### Options Structs

Many algorithms accept options for fine-tuning:

```zig
// Use defaults
const result1 = try optimize.unconstrained.bfgs(f64, fn_ptr, null, &x0, .{}, allocator);

// Customize
const opts = optimize.unconstrained.BFGSOptions(f64){
    .max_iter = 1000,
    .tol = 1e-8,
    .m = 10,  // LBFGS memory
};
const result2 = try optimize.unconstrained.bfgs(f64, fn_ptr, null, &x0, opts, allocator);
```

---

## Performance Guidelines

### NDArray Memory Layout

Choose layout based on access patterns:

```zig
// Row-major (C-style): cache-friendly for row iteration
var A_row = try NDArray(f64, 2).zeros(allocator, &.{ m, n }, .row_major);

// Column-major (Fortran-style): cache-friendly for column iteration, BLAS compatibility
var A_col = try NDArray(f64, 2).zeros(allocator, &.{ m, n }, .column_major);
```

### SIMD Acceleration

Several operations use SIMD automatically when beneficial:
- **BLAS**: GEMM (matrix multiplication) uses SIMD for small-medium matrices
- **NDArray**: Element-wise operations (add, mul, etc.) use SIMD
- **FFT**: Butterfly operations use SIMD

No special flags needed—SIMD is applied automatically based on data size.

### Algorithm Selection

Choose the right algorithm for your problem size and structure:

```zig
// Linear systems: auto-selected based on matrix properties
const x = try linalg.solve.solve(f64, A, b, allocator);  // SPD → Cholesky, general → LU

// Optimization: match algorithm to problem structure
// Unconstrained smooth: BFGS (quasi-Newton)
// Unconstrained non-smooth: Nelder-Mead (derivative-free)
// Equality constraints: Augmented Lagrangian
// Inequality constraints: Interior Point
// Linear programming: Simplex (small) or Interior Point (large)
```

---

## Testing

All modules include comprehensive test suites. Run tests:

```bash
zig build test
```

Test categories:
- **Basic operations**: Correctness of core functionality
- **Edge cases**: Empty inputs, zero dimensions, singular matrices
- **Numerical properties**: Orthogonality, symmetry, convergence
- **Type support**: Both f32 and f64
- **Memory safety**: No leaks (tested with `std.testing.allocator`)

---

## Module Documentation

Detailed API references:
- [NDArray API](ndarray.md) — N-dimensional arrays
- [Linear Algebra API](linalg.md) — BLAS, decompositions, solvers
- [Statistics API](stats.md) — Distributions, testing, regression
- [Signal Processing API](signal.md) — FFT, convolution, filtering
- [Numerical Methods API](numeric.md) — Integration, differentiation, ODE
- [Optimization API](optimize.md) — Constrained/unconstrained optimization

---

## Version Compatibility

- **zuda version**: 2.0.0
- **Zig version**: 0.15.2 or later
- **API stability**: v2.0 APIs are stable; future versions will maintain backward compatibility

---

## Further Reading

- [Getting Started Guide](../GETTING_STARTED.md) — Installation and first steps
- [Scientific Computing Guides](../guides/README.md) — Tutorials for each module
- [NumPy Compatibility](../NUMPY_COMPATIBILITY.md) — Migration from NumPy
- [Milestones](../milestones.md) — Development roadmap
