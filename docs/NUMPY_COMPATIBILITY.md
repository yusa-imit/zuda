# NumPy → zuda Migration Guide

> **Target**: Help NumPy users migrate to zuda for native Zig scientific computing

This guide maps the most commonly used NumPy functions to their zuda equivalents, providing side-by-side code comparisons and migration notes.

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Array Creation](#array-creation)
3. [Array Operations](#array-operations)
4. [Linear Algebra](#linear-algebra)
5. [Statistics](#statistics)
6. [Signal Processing](#signal-processing)
7. [Optimization](#optimization)
8. [Migration Checklist](#migration-checklist)

---

## Core Concepts

### Key Differences

| Aspect | NumPy (Python) | zuda (Zig) |
|--------|----------------|------------|
| **Memory Management** | GC-managed | Explicit allocator |
| **Type System** | Dynamic typing | Comptime generic types |
| **Error Handling** | Exceptions | Error unions |
| **Broadcasting** | Implicit | Explicit (planned v2.1) |
| **Mutability** | Mutable by default | Explicit via pointers |
| **API Style** | Object-oriented | Functional |

### Allocator Pattern

zuda requires an allocator for all heap allocations:

**NumPy**:
```python
import numpy as np
A = np.array([1, 2, 3])  # GC manages memory
```

**zuda**:
```zig
const std = @import("std");
const zuda = @import("zuda");
const NDArray = zuda.ndarray.NDArray;

const allocator = std.heap.page_allocator; // or arena allocator
const A = try NDArray(f64).fromSlice(allocator, &[_]f64{1, 2, 3}, &[_]usize{3});
defer A.deinit(); // MUST free memory
```

---

## Array Creation

### Top 20 NumPy Functions

| NumPy | zuda | Notes |
|-------|------|-------|
| `np.array([1,2,3])` | `NDArray(T).fromSlice(allocator, &[_]T{1,2,3}, &[_]usize{3})` | Explicit type, shape |
| `np.zeros((3,4))` | `NDArray(T).zeros(allocator, &[_]usize{3, 4})` | Generic over T (f32/f64) |
| `np.ones((2,2))` | `NDArray(T).ones(allocator, &[_]usize{2, 2})` | Generic over T |
| `np.full((3,), 7.0)` | `NDArray(T).full(allocator, &[_]usize{3}, 7.0)` | Fill with constant |
| `np.eye(3)` | `NDArray(T).eye(allocator, 3)` | Identity matrix |
| `np.arange(10)` | `NDArray(T).arange(allocator, 0, 10, 1)` | start, stop, step |
| `np.linspace(0,1,5)` | `NDArray(T).linspace(allocator, 0, 1, 5)` | Linear spacing |
| `np.empty((2,3))` | `NDArray(T).empty(allocator, &[_]usize{2, 3})` | Uninitialized |
| `np.random.rand(3,4)` | Use `std.Random` + `NDArray.empty()` + fill loop | No built-in RNG |
| `np.copy(A)` | `A.clone(allocator)` | Deep copy |
| `np.reshape(A, (2,3))` | `A.reshape(&[_]usize{2, 3})` | Returns error if incompatible |
| `np.transpose(A)` | `zuda.linalg.transpose(A, allocator)` | Allocates new array |
| `np.astype(A, np.float32)` | Create new `NDArray(f32)` and copy | No in-place type conversion |
| `A.shape` | `A.shape` | Direct field access (slice) |
| `A.ndim` | `A.rank` | Number of dimensions |
| `A.size` | `A.size()` | Total element count |
| `A.dtype` | Compile-time type parameter `T` | No runtime dtype |
| `np.concatenate([A,B])` | `NDArray(T).concatenate(allocator, &[_]NDArray(T){A, B}, 0)` | axis parameter |
| `np.stack([A,B])` | Not yet implemented (v2.1) | Planned |
| `np.split(A, 2)` | Not yet implemented (v2.1) | Planned |

### Code Examples

**NumPy**:
```python
import numpy as np

A = np.zeros((3, 4), dtype=np.float64)
B = np.ones((3, 4))
C = np.eye(3)
D = np.arange(10)
E = np.linspace(0, 1, 5)
```

**zuda**:
```zig
const std = @import("std");
const zuda = @import("zuda");
const NDArray = zuda.ndarray.NDArray;

const allocator = std.heap.page_allocator;

const A = try NDArray(f64).zeros(allocator, &[_]usize{3, 4});
defer A.deinit();

const B = try NDArray(f64).ones(allocator, &[_]usize{3, 4});
defer B.deinit();

const C = try NDArray(f64).eye(allocator, 3);
defer C.deinit();

const D = try NDArray(f64).arange(allocator, 0, 10, 1);
defer D.deinit();

const E = try NDArray(f64).linspace(allocator, 0, 1, 5);
defer E.deinit();
```

---

## Array Operations

### Element-wise Operations

| NumPy | zuda | Notes |
|-------|------|-------|
| `A + B` | `zuda.ndarray.simd_ops.add_simd(A, B, allocator)` | SIMD-accelerated |
| `A - B` | `zuda.ndarray.simd_ops.sub_simd(A, B, allocator)` | SIMD-accelerated |
| `A * B` | `zuda.ndarray.simd_ops.mul_simd(A, B, allocator)` | Hadamard product |
| `A / B` | `zuda.ndarray.simd_ops.div_simd(A, B, allocator)` | Element-wise division |
| `A + 5` | `zuda.ndarray.simd_ops.add_scalar_simd(A, 5, allocator)` | Scalar addition |
| `A * 2` | `zuda.ndarray.simd_ops.mul_scalar_simd(A, 2, allocator)` | Scalar multiplication |
| `np.sqrt(A)` | `A.apply(std.math.sqrt, allocator)` | Generic unary function |
| `np.exp(A)` | `A.apply(std.math.exp, allocator)` | Generic unary function |
| `np.log(A)` | `A.apply(std.math.log, allocator)` | Generic unary function |
| `np.sum(A)` | `A.sum()` | Returns scalar |
| `np.mean(A)` | `A.mean()` | Returns scalar |
| `np.min(A)` | `A.min()` | Returns scalar |
| `np.max(A)` | `A.max()` | Returns scalar |
| `np.argmin(A)` | `A.argmin()` | Returns index |
| `np.argmax(A)` | `A.argmax()` | Returns index |
| `A.flatten()` | `A.reshape(&[_]usize{A.size()})` | 1D view |
| `A.ravel()` | `A.flatten(allocator)` | Returns new 1D array |
| `A @ B` | `zuda.linalg.blas.gemm(1.0, A, B, 0.0, C, allocator)` | Matrix multiply |
| `A.T` | `zuda.linalg.transpose(A, allocator)` | Transpose |
| `np.dot(x, y)` | `zuda.linalg.simd_blas.dot_simd(x, y)` | Vector dot product |

### Code Examples

**NumPy**:
```python
import numpy as np

A = np.array([[1, 2], [3, 4]], dtype=np.float64)
B = np.array([[5, 6], [7, 8]], dtype=np.float64)

C = A + B         # Element-wise addition
D = A * B         # Hadamard product
E = A @ B         # Matrix multiply
F = A + 10        # Scalar addition
G = np.sqrt(A)    # Element-wise sqrt
total = np.sum(A) # Scalar reduction
```

**zuda**:
```zig
const std = @import("std");
const zuda = @import("zuda");
const NDArray = zuda.ndarray.NDArray;
const simd_ops = zuda.ndarray.simd_ops;
const linalg = zuda.linalg;

const allocator = std.heap.page_allocator;

const A = try NDArray(f64).fromSlice(allocator, &[_]f64{1,2,3,4}, &[_]usize{2,2});
defer A.deinit();

const B = try NDArray(f64).fromSlice(allocator, &[_]f64{5,6,7,8}, &[_]usize{2,2});
defer B.deinit();

const C = try simd_ops.add_simd(f64, A, B, allocator); // Element-wise addition
defer C.deinit();

const D = try simd_ops.mul_simd(f64, A, B, allocator); // Hadamard product
defer D.deinit();

const E_result = try NDArray(f64).zeros(allocator, &[_]usize{2, 2});
defer E_result.deinit();
try linalg.blas.gemm(f64, 1.0, A, B, 0.0, E_result); // Matrix multiply

const F = try simd_ops.add_scalar_simd(f64, A, 10, allocator); // Scalar addition
defer F.deinit();

const G = try A.apply(std.math.sqrt, allocator); // Element-wise sqrt
defer G.deinit();

const total = A.sum(); // Scalar reduction
```

---

## Linear Algebra

### Top 15 NumPy.linalg Functions

| NumPy | zuda | Notes |
|-------|------|-------|
| `np.linalg.inv(A)` | `zuda.linalg.inv(A, allocator)` | Matrix inverse via LU |
| `np.linalg.solve(A, b)` | `zuda.linalg.solve(A, b, allocator)` | Linear system Ax = b |
| `np.linalg.lstsq(A, b)` | `zuda.linalg.solve(A, b, allocator)` | Least squares via QR |
| `np.linalg.det(A)` | `zuda.linalg.det(A)` | Determinant |
| `np.linalg.eig(A)` | `zuda.linalg.eig(A, allocator)` | Eigenvalues/vectors |
| `np.linalg.svd(A)` | `zuda.linalg.decompositions.svd(A, allocator)` | Singular value decomposition |
| `np.linalg.qr(A)` | `zuda.linalg.decompositions.qr(A, allocator)` | QR decomposition |
| `np.linalg.cholesky(A)` | `zuda.linalg.decompositions.cholesky(A, allocator)` | Cholesky (SPD matrices) |
| `np.linalg.lu(A)` | `zuda.linalg.decompositions.lu(A, allocator)` | LU with pivoting |
| `np.linalg.norm(x)` | `zuda.linalg.norm(x)` | Vector/matrix norm |
| `np.linalg.cond(A)` | `zuda.linalg.cond(A)` | Condition number |
| `np.linalg.matrix_rank(A)` | `zuda.linalg.rank(A)` | Matrix rank via SVD |
| `np.linalg.pinv(A)` | `zuda.linalg.pinv(A, allocator)` | Moore-Penrose pseudoinverse |
| `np.dot(A, B)` | `zuda.linalg.blas.gemm(1.0, A, B, 0.0, C, allocator)` | Matrix-matrix product |
| `np.outer(x, y)` | `zuda.linalg.blas.ger(1.0, x, y, A)` | Outer product |

### Code Examples

**NumPy**:
```python
import numpy as np

A = np.array([[4, 2], [2, 3]], dtype=np.float64)
b = np.array([1, 2], dtype=np.float64)

x = np.linalg.solve(A, b)      # Solve Ax = b
L = np.linalg.cholesky(A)      # Cholesky decomposition
A_inv = np.linalg.inv(A)       # Matrix inverse
U, S, Vt = np.linalg.svd(A)    # SVD
```

**zuda**:
```zig
const std = @import("std");
const zuda = @import("zuda");
const NDArray = zuda.ndarray.NDArray;
const linalg = zuda.linalg;

const allocator = std.heap.page_allocator;

const A = try NDArray(f64).fromSlice(allocator, &[_]f64{4,2,2,3}, &[_]usize{2,2});
defer A.deinit();

const b = try NDArray(f64).fromSlice(allocator, &[_]f64{1,2}, &[_]usize{2});
defer b.deinit();

const x = try linalg.solve(f64, A, b, allocator); // Solve Ax = b
defer x.deinit();

const L = try linalg.decompositions.cholesky(f64, A, allocator); // Cholesky decomposition
defer L.deinit();

const A_inv = try linalg.inv(f64, A, allocator); // Matrix inverse
defer A_inv.deinit();

const svd_result = try linalg.decompositions.svd(f64, A, allocator); // SVD
defer svd_result.U.deinit();
defer svd_result.S.deinit();
defer svd_result.Vt.deinit();
```

---

## Statistics

### Top 10 NumPy.stats Functions

| NumPy | zuda | Notes |
|-------|------|-------|
| `np.mean(A)` | `zuda.stats.mean(A)` | Arithmetic mean |
| `np.median(A)` | `zuda.stats.median(A, allocator)` | Median (requires sorting) |
| `np.std(A)` | `zuda.stats.std(A)` | Standard deviation |
| `np.var(A)` | `zuda.stats.variance(A)` | Variance |
| `np.corrcoef(x, y)` | `zuda.stats.corrcoef(x, y)` | Correlation coefficient |
| `np.cov(X)` | `zuda.stats.cov(X, allocator)` | Covariance matrix |
| `np.histogram(x, bins)` | `zuda.stats.histogram(x, bins, allocator)` | Histogram counts |
| `np.percentile(x, 50)` | `zuda.stats.percentile(x, 50, allocator)` | Percentile |
| `scipy.stats.norm.pdf(x)` | `zuda.stats.distributions.normal(0, 1).pdf(x)` | Normal PDF |
| `scipy.stats.ttest_ind(x, y)` | `zuda.stats.ttest_ind(x, y)` | Independent t-test |

### Code Examples

**NumPy**:
```python
import numpy as np

data = np.array([1, 2, 3, 4, 5], dtype=np.float64)

avg = np.mean(data)
med = np.median(data)
sd = np.std(data)
var = np.var(data)
```

**zuda**:
```zig
const std = @import("std");
const zuda = @import("zuda");
const NDArray = zuda.ndarray.NDArray;
const stats = zuda.stats;

const allocator = std.heap.page_allocator;

const data = try NDArray(f64).fromSlice(allocator, &[_]f64{1,2,3,4,5}, &[_]usize{5});
defer data.deinit();

const avg = stats.mean(f64, data);
const med = try stats.median(f64, data, allocator);
const sd = stats.std(f64, data);
const variance = stats.variance(f64, data);
```

---

## Signal Processing

### Top 10 SciPy.signal Functions

| SciPy | zuda | Notes |
|-------|------|-------|
| `scipy.fft.fft(x)` | `zuda.signal.simd_fft.fft_simd(x, allocator)` | SIMD-accelerated |
| `scipy.fft.ifft(x)` | `zuda.signal.simd_fft.ifft_simd(x, allocator)` | SIMD-accelerated |
| `scipy.fft.rfft(x)` | `zuda.signal.fft.rfft(x, allocator)` | Real FFT |
| `scipy.fft.fft2(x)` | `zuda.signal.fft2.fft2(x, allocator)` | 2D FFT |
| `scipy.signal.convolve(x, y)` | `zuda.signal.convolve.convolve(x, y, allocator)` | Convolution |
| `scipy.signal.correlate(x, y)` | `zuda.signal.convolve.correlate(x, y, allocator)` | Cross-correlation |
| `scipy.signal.butter(N, Wn)` | `zuda.signal.filter.butter(N, Wn, allocator)` | Butterworth filter |
| `scipy.signal.filtfilt(b, a, x)` | `zuda.signal.filter.filtfilt(b, a, x, allocator)` | Zero-phase filtering |
| `scipy.signal.welch(x)` | `zuda.signal.spectral.welch(x, allocator)` | Power spectral density |
| `scipy.signal.periodogram(x)` | `zuda.signal.spectral.periodogram(x, allocator)` | Periodogram |

### Code Examples

**SciPy**:
```python
import numpy as np
from scipy import fft, signal

x = np.array([1, 2, 3, 4], dtype=np.float64)

X = fft.fft(x)             # FFT
x_back = fft.ifft(X)       # IFFT
psd = signal.welch(x)      # Power spectral density
```

**zuda**:
```zig
const std = @import("std");
const zuda = @import("zuda");
const Complex = zuda.signal.fft.Complex;
const simd_fft = zuda.signal.simd_fft;
const spectral = zuda.signal.spectral;

const allocator = std.heap.page_allocator;

var x_complex = [_]Complex(f64){
    Complex(f64).init(1, 0),
    Complex(f64).init(2, 0),
    Complex(f64).init(3, 0),
    Complex(f64).init(4, 0),
};

const X = try simd_fft.fft_simd(f64, &x_complex, allocator); // FFT
defer allocator.free(X);

const x_back = try simd_fft.ifft_simd(f64, X, allocator); // IFFT
defer allocator.free(x_back);

// For welch, need to create NDArray from real data
const x_real = try zuda.ndarray.NDArray(f64).fromSlice(allocator, &[_]f64{1,2,3,4}, &[_]usize{4});
defer x_real.deinit();

const psd = try spectral.welch(f64, x_real, allocator); // Power spectral density
defer psd.freqs.deinit();
defer psd.psd.deinit();
```

---

## Optimization

### Top 5 SciPy.optimize Functions

| SciPy | zuda | Notes |
|-------|------|-------|
| `scipy.optimize.minimize(f, x0)` | `zuda.optimize.unconstrained.bfgs(f, x0, allocator)` | BFGS optimizer |
| `scipy.optimize.least_squares(f, x0)` | `zuda.optimize.least_squares.levenberg_marquardt(f, x0, allocator)` | Curve fitting |
| `scipy.optimize.curve_fit(f, x, y)` | `zuda.numeric.curve_fit.curve_fit(f, x, y, allocator)` | Curve fitting |
| `scipy.optimize.linprog(c, A_ub, b_ub)` | `zuda.optimize.constrained.simplex(c, A, b, allocator)` | Linear programming |
| `scipy.optimize.minimize(f, x0, constraints)` | `zuda.optimize.constrained.augmented_lagrangian(f, x0, allocator)` | Constrained optimization |

### Code Examples

**SciPy**:
```python
import numpy as np
from scipy.optimize import minimize

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

x0 = np.array([0, 0], dtype=np.float64)
result = minimize(rosenbrock, x0, method='BFGS')
print(result.x)  # [1, 1]
```

**zuda**:
```zig
const std = @import("std");
const zuda = @import("zuda");
const optimize = zuda.optimize;

const allocator = std.heap.page_allocator;

fn rosenbrock(x: []const f64, grad: ?[]f64) f64 {
    const term1 = 1.0 - x[0];
    const term2 = x[1] - x[0] * x[0];
    if (grad) |g| {
        g[0] = -2.0 * term1 - 400.0 * x[0] * term2;
        g[1] = 200.0 * term2;
    }
    return term1 * term1 + 100.0 * term2 * term2;
}

const x0 = [_]f64{0, 0};
const result = try optimize.unconstrained.bfgs(f64, rosenbrock, &x0, allocator, .{});
defer allocator.free(result.x);

// result.x ≈ [1, 1]
```

---

## Migration Checklist

### Before You Start

- [ ] **Understand allocator patterns**: zuda requires explicit memory management
- [ ] **Learn error handling**: Use `try` and `catch` for error unions
- [ ] **Embrace comptime**: Type parameters are resolved at compile time
- [ ] **Plan for deallocation**: Every `try allocator.alloc()` needs `defer allocator.free()`

### Common Pitfalls

1. **Forgetting `defer`**: Always pair allocation with deallocation
   ```zig
   const A = try NDArray(f64).zeros(allocator, &[_]usize{3, 4});
   defer A.deinit(); // REQUIRED
   ```

2. **Type mismatches**: Explicit type parameters (`f32` vs `f64`)
   ```zig
   const A = try NDArray(f64).zeros(allocator, &[_]usize{3, 4});
   const B = try NDArray(f32).zeros(allocator, &[_]usize{3, 4}); // DIFFERENT TYPE
   // Cannot mix A and B in operations without conversion
   ```

3. **Shape specification**: zuda requires explicit shape arrays
   ```zig
   // NumPy: np.zeros((3, 4))
   // zuda: NDArray(f64).zeros(allocator, &[_]usize{3, 4})
   ```

4. **Broadcasting**: Not yet implemented — manual loop required
   ```zig
   // NumPy: A + b (where b is 1D, A is 2D) — broadcasts automatically
   // zuda: Manual loop or wait for v2.1 broadcasting support
   ```

5. **Indexing**: zuda uses `.get()` for bounds-checked access
   ```zig
   // NumPy: A[1, 2]
   // zuda: try A.get(&[_]usize{1, 2})
   ```

### Performance Considerations

- **Use SIMD variants**: `simd_ops`, `simd_blas`, `simd_fft` for 2-4× speedup
- **Prefer in-place operations**: When possible, reuse buffers (e.g., `gemm` writes to `C`)
- **Arena allocator for batch work**: Reduces allocation overhead
- **Batch deallocations**: Use `defer` to ensure cleanup even on error

### Example Migration: Linear Regression

**NumPy**:
```python
import numpy as np

# Generate data
X = np.random.rand(100, 2)
y = 3 * X[:, 0] + 2 * X[:, 1] + 1 + 0.1 * np.random.randn(100)

# Solve normal equations: β = (X^T X)^-1 X^T y
XtX = X.T @ X
Xty = X.T @ y
beta = np.linalg.solve(XtX, Xty)
```

**zuda**:
```zig
const std = @import("std");
const zuda = @import("zuda");
const NDArray = zuda.ndarray.NDArray;
const linalg = zuda.linalg;

const allocator = std.heap.page_allocator;

// Generate data (use std.Random to fill X and y)
const X = try NDArray(f64).empty(allocator, &[_]usize{100, 2});
defer X.deinit();
// ... fill X with random data ...

const y = try NDArray(f64).empty(allocator, &[_]usize{100});
defer y.deinit();
// ... fill y with target values ...

// Solve normal equations: β = (X^T X)^-1 X^T y
const Xt = try linalg.transpose(f64, X, allocator);
defer Xt.deinit();

const XtX_result = try NDArray(f64).zeros(allocator, &[_]usize{2, 2});
defer XtX_result.deinit();
try linalg.blas.gemm(f64, 1.0, Xt, X, 0.0, XtX_result);

const Xty_result = try NDArray(f64).zeros(allocator, &[_]usize{2});
defer Xty_result.deinit();
try linalg.blas.gemv(f64, 1.0, Xt, y, 0.0, Xty_result);

const beta = try linalg.solve(f64, XtX_result, Xty_result, allocator);
defer beta.deinit();
```

---

## Roadmap

### v2.0 (Current)
- ✅ NDArray
- ✅ BLAS (SIMD-accelerated)
- ✅ Linear algebra (decompositions, solvers)
- ✅ Statistics (distributions, hypothesis tests, regression)
- ✅ Signal processing (FFT, filters, spectral analysis)
- ✅ Optimization (unconstrained, constrained, least squares, LP, auto-diff)

### v2.1 (Planned)
- [ ] Broadcasting support (NumPy-style)
- [ ] Advanced indexing (boolean masks, fancy indexing)
- [ ] Lazy evaluation / expression templates
- [ ] GPU acceleration (Vulkan compute shaders)
- [ ] Parallel algorithms (multi-threaded BLAS)

---

## Resources

- **zuda Documentation**: `docs/API.md`
- **NumPy Documentation**: https://numpy.org/doc/stable/
- **Zig Documentation**: https://ziglang.org/documentation/master/
- **Report Issues**: https://github.com/yusa-imit/zuda/issues

---

**Note**: This guide covers the most commonly used NumPy functions. For comprehensive API documentation, see `docs/API.md`. For migration assistance, open an issue on GitHub.
