# Migrating from NumPy to zuda

> **Quick start**: zuda provides a Zig-native scientific computing platform with NumPy-like APIs. This guide shows side-by-side comparisons and migration patterns.

---

## Table of Contents

1. [Philosophy & Design Differences](#philosophy--design-differences)
2. [Array Creation](#array-creation)
3. [Indexing & Slicing](#indexing--slicing)
4. [Element-wise Operations](#element-wise-operations)
5. [Broadcasting](#broadcasting)
6. [Reductions & Statistics](#reductions--statistics)
7. [Linear Algebra](#linear-algebra)
8. [FFT & Signal Processing](#fft--signal-processing)
9. [Random Number Generation](#random-number-generation)
10. [Performance Considerations](#performance-considerations)
11. [Complete Examples](#complete-examples)

---

## Philosophy & Design Differences

### NumPy (Python)
- **Dynamic typing**: Arrays created at runtime, shape/dtype determined dynamically
- **Reference semantics**: Arrays are objects, assignments create views
- **GC-managed**: Automatic memory management
- **Interpreted**: JIT compilation via NumPy C backend

### zuda (Zig)
- **Static typing**: `NDArray(T, ndim)` with compile-time rank, runtime shape
- **Explicit ownership**: `init()` allocates, `deinit()` frees — no GC
- **Compiled**: Native code, zero-cost abstractions
- **Allocator-first**: Every operation takes `std.mem.Allocator`

**Migration mindset**: Think "arrays as values" instead of "arrays as references". Explicit memory management replaces automatic GC.

---

## Array Creation

### Zeros / Ones / Full

**NumPy**:
```python
import numpy as np

a = np.zeros((3, 4))           # 3×4 array of zeros (float64)
b = np.ones((2, 2), dtype=int) # 2×2 array of ones (int)
c = np.full((2, 3), 7.5)       # 2×3 array filled with 7.5
```

**zuda**:
```zig
const zuda = @import("zuda");
const std = @import("std");

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
defer _ = gpa.deinit();
const allocator = gpa.allocator();

// Zeros: f64 is default, rank=2
var a = try zuda.ndarray.zeros(f64, 2, &[_]usize{3, 4}, allocator);
defer a.deinit();

// Ones: explicit type i32
var b = try zuda.ndarray.ones(i32, 2, &[_]usize{2, 2}, allocator);
defer b.deinit();

// Full: fill with 7.5
var c = try zuda.ndarray.full(f64, 2, &[_]usize{2, 3}, 7.5, allocator);
defer c.deinit();
```

### Arange / Linspace

**NumPy**:
```python
a = np.arange(0, 10, 2)      # [0, 2, 4, 6, 8]
b = np.linspace(0, 1, 5)     # [0, 0.25, 0.5, 0.75, 1.0]
```

**zuda**:
```zig
// Arange: start, stop, step
var a = try zuda.ndarray.arange(i32, 0, 10, 2, allocator);
defer a.deinit(); // Shape: (5,)

// Linspace: start, stop, count
var b = try zuda.ndarray.linspace(f64, 0.0, 1.0, 5, allocator);
defer b.deinit(); // Shape: (5,)
```

### From Slice / Identity

**NumPy**:
```python
data = [1, 2, 3, 4, 5, 6]
a = np.array(data).reshape(2, 3)  # 2×3 from flat data
I = np.eye(3)                      # 3×3 identity
```

**zuda**:
```zig
const data = [_]f64{1, 2, 3, 4, 5, 6};
var a = try zuda.ndarray.fromSlice(f64, 2, &[_]usize{2, 3}, &data, allocator);
defer a.deinit();

var I = try zuda.ndarray.eye(f64, 3, allocator);
defer I.deinit();
```

---

## Indexing & Slicing

### Basic Indexing

**NumPy**:
```python
a = np.array([[1, 2, 3], [4, 5, 6]])
x = a[0, 1]      # Element at row 0, col 1 → 2
row = a[1, :]    # Second row → [4, 5, 6]
col = a[:, 0]    # First column → [1, 4]
```

**zuda**:
```zig
var a = try zuda.ndarray.fromSlice(f64, 2, &[_]usize{2, 3},
    &[_]f64{1, 2, 3, 4, 5, 6}, allocator);
defer a.deinit();

const x = a.get(&[_]usize{0, 1});  // Returns f64: 2.0

// Slicing returns views (non-owning)
var row = a.slice(&[_]zuda.ndarray.Range{
    .{ .index = 1 },         // Row 1
    .{ .all = {} },          // All columns
});
defer row.deinit();  // Free view metadata (not data)

var col = a.slice(&[_]zuda.ndarray.Range{
    .{ .all = {} },          // All rows
    .{ .index = 0 },         // Column 0
});
defer col.deinit();
```

### Negative Indexing

**NumPy**:
```python
a = np.array([1, 2, 3, 4, 5])
last = a[-1]        # 5
second_last = a[-2] # 4
```

**zuda**:
```zig
var a = try zuda.ndarray.arange(i32, 1, 6, 1, allocator);
defer a.deinit();

const last = a.get(&[_]usize{@intCast(a.shape[0] - 1)});  // 5
// Negative indexing: use @intCast with manual calculation
```

---

## Element-wise Operations

### Arithmetic

**NumPy**:
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

c = a + b        # [5, 7, 9]
d = a * 2        # [2, 4, 6]
e = a ** 2       # [1, 4, 9]
```

**zuda**:
```zig
var a = try zuda.ndarray.fromSlice(f64, 1, &[_]usize{3},
    &[_]f64{1, 2, 3}, allocator);
defer a.deinit();
var b = try zuda.ndarray.fromSlice(f64, 1, &[_]usize{3},
    &[_]f64{4, 5, 6}, allocator);
defer b.deinit();

var c = try a.add(b, allocator);  // Element-wise add
defer c.deinit();

var d = try a.mul_scalar(2.0, allocator);  // Scalar multiply
defer d.deinit();

var e = try a.pow(2.0, allocator);  // Element-wise power
defer e.deinit();
```

### Math Functions

**NumPy**:
```python
a = np.array([0, np.pi/4, np.pi/2])
b = np.sin(a)    # Element-wise sine
c = np.exp(a)    # Element-wise exponential
d = np.sqrt(a)   # Element-wise square root
```

**zuda**:
```zig
const pi = std.math.pi;
var a = try zuda.ndarray.fromSlice(f64, 1, &[_]usize{3},
    &[_]f64{0, pi/4, pi/2}, allocator);
defer a.deinit();

var b = try a.sin(allocator);
defer b.deinit();

var c = try a.exp(allocator);
defer c.deinit();

var d = try a.sqrt(allocator);
defer d.deinit();
```

---

## Broadcasting

**NumPy**:
```python
a = np.array([[1, 2, 3]])        # (1, 3)
b = np.array([[10], [20], [30]]) # (3, 1)
c = a + b  # Broadcasts to (3, 3)
# [[11, 12, 13],
#  [21, 22, 23],
#  [31, 32, 33]]
```

**zuda**:
```zig
var a = try zuda.ndarray.fromSlice(f64, 2, &[_]usize{1, 3},
    &[_]f64{1, 2, 3}, allocator);
defer a.deinit();

var b = try zuda.ndarray.fromSlice(f64, 2, &[_]usize{3, 1},
    &[_]f64{10, 20, 30}, allocator);
defer b.deinit();

var c = try a.add(b, allocator);  // Broadcasting automatic
defer c.deinit();
// Result shape: (3, 3)
```

---

## Reductions & Statistics

**NumPy**:
```python
a = np.array([[1, 2, 3], [4, 5, 6]])
total = a.sum()              # 21 (all elements)
row_sums = a.sum(axis=1)     # [6, 15] (sum along columns)
col_means = a.mean(axis=0)   # [2.5, 3.5, 4.5] (mean along rows)
```

**zuda**:
```zig
var a = try zuda.ndarray.fromSlice(f64, 2, &[_]usize{2, 3},
    &[_]f64{1, 2, 3, 4, 5, 6}, allocator);
defer a.deinit();

const total = a.sum();  // Returns f64: 21.0

var row_sums = try a.sum_axis(1, allocator);  // Axis 1 (columns)
defer row_sums.deinit();  // Shape: (2,) → [6, 15]

var col_means = try a.mean_axis(0, allocator);  // Axis 0 (rows)
defer col_means.deinit();  // Shape: (3,) → [2.5, 3.5, 4.5]
```

### Descriptive Statistics

**NumPy**:
```python
data = np.array([1, 2, 3, 4, 5])
m = np.mean(data)    # 3.0
s = np.std(data)     # 1.414... (Bessel correction)
q = np.quantile(data, 0.5)  # 3.0 (median)
```

**zuda**:
```zig
const data = [_]f64{1, 2, 3, 4, 5};
const m = zuda.stats.mean(f64, &data);      // 3.0
const s = zuda.stats.std(f64, &data, 1);    // ddof=1 (Bessel)
const q = try zuda.stats.quantile(f64, &data, 0.5, allocator);  // 3.0
```

---

## Linear Algebra

### Matrix Multiplication

**NumPy**:
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B  # Matrix product [[19, 22], [43, 50]]
```

**zuda**:
```zig
var A = try zuda.ndarray.fromSlice(f64, 2, &[_]usize{2, 2},
    &[_]f64{1, 2, 3, 4}, allocator);
defer A.deinit();

var B = try zuda.ndarray.fromSlice(f64, 2, &[_]usize{2, 2},
    &[_]f64{5, 6, 7, 8}, allocator);
defer B.deinit();

var C = try zuda.linalg.gemm(f64, 1.0, A, B, 0.0, null, allocator);
defer C.deinit();  // Result: [[19, 22], [43, 50]]
```

### Solving Linear Systems

**NumPy**:
```python
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(A, b)  # [2, 3]
```

**zuda**:
```zig
var A = try zuda.ndarray.fromSlice(f64, 2, &[_]usize{2, 2},
    &[_]f64{3, 1, 1, 2}, allocator);
defer A.deinit();

var b = try zuda.ndarray.fromSlice(f64, 1, &[_]usize{2},
    &[_]f64{9, 8}, allocator);
defer b.deinit();

var x = try zuda.linalg.solve(f64, A, b, allocator);
defer x.deinit();  // [2, 3]
```

### Eigenvalues / SVD

**NumPy**:
```python
A = np.array([[4, 2], [1, 3]])
eigvals, eigvecs = np.linalg.eig(A)

U, S, Vt = np.linalg.svd(A)
```

**zuda**:
```zig
var A = try zuda.ndarray.fromSlice(f64, 2, &[_]usize{2, 2},
    &[_]f64{4, 2, 1, 3}, allocator);
defer A.deinit();

// Eigendecomposition
var eig_result = try zuda.linalg.eig(f64, A, allocator);
defer eig_result.eigenvalues.deinit();
defer eig_result.eigenvectors.deinit();

// SVD
var svd_result = try zuda.linalg.svd(f64, A, allocator);
defer svd_result.U.deinit();
defer svd_result.Sigma.deinit();
defer svd_result.Vt.deinit();
```

---

## FFT & Signal Processing

**NumPy**:
```python
import numpy.fft as fft

signal = np.array([1, 2, 3, 4])
spectrum = fft.fft(signal)
reconstructed = fft.ifft(spectrum).real
```

**zuda**:
```zig
var signal = try zuda.ndarray.fromSlice(f64, 1, &[_]usize{4},
    &[_]f64{1, 2, 3, 4}, allocator);
defer signal.deinit();

var spectrum = try zuda.signal.fft(f64, signal, allocator);
defer spectrum.deinit();

var reconstructed = try zuda.signal.ifft(f64, spectrum, allocator);
defer reconstructed.deinit();
```

### Convolution

**NumPy**:
```python
from scipy.signal import convolve

a = np.array([1, 2, 3])
b = np.array([0, 1, 0.5])
c = convolve(a, b, mode='same')
```

**zuda**:
```zig
const a = [_]f64{1, 2, 3};
const b = [_]f64{0, 1, 0.5};
var c = try zuda.signal.convolve(f64, &a, &b, .same, allocator);
defer c.deinit();
```

---

## Random Number Generation

**NumPy**:
```python
rng = np.random.default_rng(seed=42)
uniform = rng.uniform(0, 1, size=5)
normal = rng.normal(0, 1, size=5)
choice = rng.choice([1, 2, 3, 4], size=2, replace=False)
```

**zuda**:
```zig
var rng = zuda.stats.random.PCG64.init(42);

var uniform = try zuda.stats.random.uniform(f64, &rng, 0.0, 1.0, 5, allocator);
defer uniform.deinit();

var normal = try zuda.stats.random.normal(f64, &rng, 0.0, 1.0, 5, allocator);
defer normal.deinit();

const population = [_]i32{1, 2, 3, 4};
var choice = try zuda.stats.random.choice(i32, &rng, &population, 2, false, allocator);
defer choice.deinit();
```

---

## Performance Considerations

### Memory Management

**NumPy**: Automatic via Python GC — no explicit control
**zuda**: Explicit `init()` / `deinit()` — use `defer` to prevent leaks

```zig
// GOOD: Defer ensures cleanup
var a = try zuda.ndarray.zeros(f64, 2, &[_]usize{1000, 1000}, allocator);
defer a.deinit();

// BAD: Memory leak if function returns early
var b = try zuda.ndarray.zeros(f64, 2, &[_]usize{1000, 1000}, allocator);
// ... code ...
b.deinit();  // May not execute if error occurs
```

### Views vs Copies

**NumPy**: Slicing creates views (shares data), `.copy()` for deep copy
**zuda**: `.slice()` creates non-owning view, `.clone()` for deep copy

```zig
var a = try zuda.ndarray.zeros(f64, 2, &[_]usize{10, 10}, allocator);
defer a.deinit();

// View: shares data with `a`
var view = a.slice(&[_]zuda.ndarray.Range{
    .{ .range = .{ .start = 0, .stop = 5 } },
    .{ .all = {} },
});
defer view.deinit();  // Only frees view metadata

// Copy: allocates new memory
var copy = try a.clone(allocator);
defer copy.deinit();  // Frees actual data
```

### Contiguity

**NumPy**: Checks with `.flags['C_CONTIGUOUS']`, makes contiguous with `.copy(order='C')`
**zuda**: `.isContiguous()` + `.contiguous()` (copies if needed)

```zig
if (!a.isContiguous()) {
    var contig = try a.contiguous(allocator);
    defer contig.deinit();
    // Use contig for cache-friendly operations
}
```

---

## Complete Examples

### Linear Regression

**NumPy**:
```python
import numpy as np

# Generate data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2.1, 3.9, 6.1, 8.0, 10.2])

# Add bias column
X_with_bias = np.column_stack([np.ones(5), X])

# Solve normal equations: (X^T X)^-1 X^T y
theta = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
print(f"Coefficients: {theta}")  # [0.04, 2.02]
```

**zuda**:
```zig
const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Generate data
    var X = try zuda.ndarray.fromSlice(f64, 2, &[_]usize{5, 1},
        &[_]f64{1, 2, 3, 4, 5}, allocator);
    defer X.deinit();

    var y = try zuda.ndarray.fromSlice(f64, 1, &[_]usize{5},
        &[_]f64{2.1, 3.9, 6.1, 8.0, 10.2}, allocator);
    defer y.deinit();

    // Add bias column (prepend ones)
    var ones = try zuda.ndarray.ones(f64, 2, &[_]usize{5, 1}, allocator);
    defer ones.deinit();
    var X_with_bias = try zuda.ndarray.concatenate(f64, &[_]*zuda.NDArray(f64, 2){&ones, &X}, 1, allocator);
    defer X_with_bias.deinit();

    // Least squares solve
    var theta = try zuda.linalg.lstsq(f64, X_with_bias, y, allocator);
    defer theta.deinit();

    std.debug.print("Coefficients: [{d:.2}, {d:.2}]\n", .{theta.get(&[_]usize{0}), theta.get(&[_]usize{1})});
}
```

### Image Filtering (2D Convolution)

**NumPy**:
```python
import numpy as np
from scipy.signal import convolve2d

# 5×5 image
image = np.array([
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    [1, 3, 5, 3, 1],
    [2, 4, 2, 4, 2],
    [5, 1, 5, 1, 5]
])

# 3×3 Gaussian blur kernel
kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]) / 16.0

blurred = convolve2d(image, kernel, mode='same', boundary='fill')
```

**zuda**:
```zig
const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // 5×5 image
    var image = try zuda.ndarray.fromSlice(f64, 2, &[_]usize{5, 5}, &[_]f64{
        1, 2, 3, 4, 5,
        5, 4, 3, 2, 1,
        1, 3, 5, 3, 1,
        2, 4, 2, 4, 2,
        5, 1, 5, 1, 5,
    }, allocator);
    defer image.deinit();

    // 3×3 Gaussian blur kernel
    const kernel_data = [_]f64{
        1.0/16, 2.0/16, 1.0/16,
        2.0/16, 4.0/16, 2.0/16,
        1.0/16, 2.0/16, 1.0/16,
    };
    var kernel = try zuda.ndarray.fromSlice(f64, 2, &[_]usize{3, 3}, &kernel_data, allocator);
    defer kernel.deinit();

    var blurred = try zuda.signal.convolve2d(f64, image, kernel, .same, allocator);
    defer blurred.deinit();
}
```

---

## Migration Checklist

- [ ] **Replace imports**: `import numpy as np` → `const zuda = @import("zuda")`
- [ ] **Add allocator**: Pass `std.mem.Allocator` to all array-creating functions
- [ ] **Add deinit()**: Use `defer arr.deinit()` after every `init()` / allocation
- [ ] **Explicit types**: `NDArray(T, ndim)` requires compile-time rank and element type
- [ ] **Error handling**: Replace exceptions with `try` / `catch` (Zig error unions)
- [ ] **Views**: `.slice()` creates views (free metadata only), `.clone()` for deep copies
- [ ] **Broadcasting**: Automatic, same rules as NumPy
- [ ] **Linear algebra**: Replace `np.linalg.*` with `zuda.linalg.*`
- [ ] **FFT**: Replace `np.fft.*` with `zuda.signal.fft/ifft`
- [ ] **Random**: Replace `np.random.*` with `zuda.stats.random.*` (explicit RNG state)

---

## Further Reading

- [NumPy Compatibility Reference](../NUMPY_COMPATIBILITY.md) — 50+ function mappings
- [NDArray Guide](../guides/ndarray.md) — Detailed API documentation
- [Linear Algebra Guide](../guides/linalg.md) — BLAS, decompositions, solvers
- [Statistics Guide](../guides/stats.md) — Distributions, hypothesis tests, regression
- [Signal Processing Guide](../guides/signal.md) — FFT, filtering, convolution

---

**TL;DR**: NumPy → zuda migration is straightforward. Key differences: explicit memory management (`defer`), compile-time rank (`NDArray(T, ndim)`), and allocator-first design. Performance is competitive, often faster due to native compilation.
