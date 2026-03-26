# NDArray — N-Dimensional Arrays

## Overview

`NDArray(T, ndim)` provides a generalized multi-dimensional array structure, serving as the foundation for scientific computing in zuda. It offers NumPy-compatible operations with compile-time rank safety and efficient stride-based memory access.

## Key Features

- **Compile-time rank safety**: Number of dimensions (`ndim`) is checked at compile time
- **Runtime shape flexibility**: Actual dimensions are determined at runtime
- **Multiple memory layouts**: Row-major (C order) and column-major (Fortran order)
- **Zero-copy views**: Slicing and reshaping without data copying when possible
- **Element-wise operations**: Arithmetic, comparison, and mathematical functions
- **Broadcasting**: NumPy-style shape broadcasting for mixed-size operations

## Basic Usage

### Creating Arrays

```zig
const std = @import("std");
const zuda = @import("zuda");
const NDArray = zuda.ndarray.NDArray;

// Allocate a 3×4 matrix (row-major)
var matrix = try NDArray(f64, 2).init(allocator, &.{3, 4}, .row_major);
defer matrix.deinit();

// Create from existing data
const data = [_]f32{1, 2, 3, 4, 5, 6};
var vec = try NDArray(f32, 1).fromSlice(allocator, &data, .row_major);
defer vec.deinit();

// Convenience constructors
var zeros = try NDArray(f64, 2).zeros(allocator, &.{100, 100}, .row_major);
defer zeros.deinit();

var ones = try NDArray(f64, 1).ones(allocator, &.{1000}, .row_major);
defer ones.deinit();

var identity = try NDArray(f64, 2).eye(allocator, 5, 5, 0, .row_major);
defer identity.deinit();

// Random arrays (uniform [0,1))
var rng = std.rand.DefaultPrng.init(42);
var random = try NDArray(f64, 2).rand(allocator, &.{10, 10}, .row_major, &rng.random());
defer random.deinit();
```

### Memory Layouts

**Row-major (C order)**: Last dimension varies fastest in memory
```zig
// For shape [3, 4], memory order:
// [0,0] [0,1] [0,2] [0,3] [1,0] [1,1] ...
var row_major = try NDArray(f64, 2).init(allocator, &.{3, 4}, .row_major);
```

**Column-major (Fortran order)**: First dimension varies fastest
```zig
// For shape [3, 4], memory order:
// [0,0] [1,0] [2,0] [0,1] [1,1] [2,1] ...
var col_major = try NDArray(f64, 2).init(allocator, &.{3, 4}, .column_major);
```

*Use row-major for most applications. Use column-major for Fortran interop or when columns are accessed more frequently than rows.*

### Accessing Elements

```zig
// Single element access
const val = matrix.get(&.{1, 2}); // row 1, col 2
matrix.set(&.{1, 2}, 3.14);

// Flat indexing (ignores strides)
const flat = matrix.data[5]; // Direct memory access

// Iteration
var iter = matrix.iterator();
while (iter.next()) |value| {
    std.debug.print("{d} ", .{value});
}
```

**Important**: Indices must be `[]const isize`, not `usize`:
```zig
// CORRECT
const i: usize = 5;
const j: usize = 3;
matrix.set(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) }, value);

// WRONG
matrix.set(&.{ i, j }, value); // Type error
```

## Array Operations

### Element-wise Arithmetic

All operations return new arrays and use the array's own allocator:

```zig
var a = try NDArray(f64, 2).ones(allocator, &.{3, 3}, .row_major);
defer a.deinit();

var b = try NDArray(f64, 2).ones(allocator, &.{3, 3}, .row_major);
defer b.deinit();

// Element-wise operations (create new arrays)
var sum = try a.add(&b);     // a + b
defer sum.deinit();

var diff = try a.sub(&b);    // a - b
defer diff.deinit();

var prod = try a.mul(&b);    // a * b (element-wise)
defer prod.deinit();

var quot = try a.div(&b);    // a / b (element-wise)
defer quot.deinit();
```

### Scalar Operations

```zig
var scaled = try a.mulScalar(2.5);  // a * 2.5
defer scaled.deinit();

var shifted = try a.addScalar(10.0); // a + 10.0
defer shifted.deinit();
```

### Mathematical Functions

```zig
// Trigonometric
var sin_vals = try a.sin();
defer sin_vals.deinit();

var cos_vals = try a.cos();
defer cos_vals.deinit();

// Exponential and logarithmic
var exp_vals = try a.exp();
defer exp_vals.deinit();

var log_vals = try a.log();    // Natural log
defer log_vals.deinit();

// Power and roots
var squared = try a.pow(2.0);
defer squared.deinit();

var roots = try a.sqrt();
defer roots.deinit();

// Absolute value
var abs_vals = try a.abs();
defer abs_vals.deinit();
```

### Reductions

Reductions collapse one or more dimensions:

```zig
const total = a.sum();           // Sum all elements
const avg = a.mean();            // Mean of all elements
const maximum = a.max();         // Maximum element
const minimum = a.min();         // Minimum element

// Axis-specific reductions
var row_sums = try a.sumAxis(0);  // Sum along axis 0
defer row_sums.deinit();

var col_means = try a.meanAxis(1); // Mean along axis 1
defer col_means.deinit();
```

## Shape Manipulation

### Reshaping

```zig
// Reshape to new dimensions (must have same total size)
var reshaped = try a.reshape(&.{9, 1});  // 3×3 → 9×1
defer reshaped.deinit();

// Flatten to 1D
var flat = try a.flatten();
defer flat.deinit();
```

### Transposition

```zig
// Matrix transpose (2D only)
const transposed = matrix.transpose();  // View, no allocation

// General permutation (any ndim)
var permuted = try tensor.permute(&.{2, 0, 1});  // Rearrange axes
defer permuted.deinit();
```

### Slicing

```zig
// Extract sub-array (returns view when possible)
const Slice = NDArray(f64, 2).Slice;
var sub = try matrix.slice(&[_]Slice{
    .{ .start = 0, .end = 2, .step = 1 },  // First 2 rows
    .{ .start = 1, .end = 4, .step = 1 },  // Columns 1-3
});
defer sub.deinit();
```

## Advanced Topics

### SIMD Acceleration

Element-wise operations automatically use SIMD when available:

```zig
// Automatically vectorized on supported platforms
var result = try a.add(&b);  // Uses SIMD for large arrays
defer result.deinit();
```

### Type Support

NDArray works with any copyable numeric type:

```zig
var int_array = try NDArray(i32, 2).zeros(allocator, &.{10, 10}, .row_major);
defer int_array.deinit();

var float_array = try NDArray(f32, 3).ones(allocator, &.{5, 5, 5}, .row_major);
defer float_array.deinit();

var complex_array = try NDArray(std.math.Complex(f64), 1).init(allocator, &.{100}, .row_major);
defer complex_array.deinit();
```

### Performance Tips

1. **Use row-major layout by default** — Better cache locality for most algorithms
2. **Preallocate when possible** — Use `zeros()` or `ones()` instead of `init()` + filling
3. **Avoid unnecessary copies** — Use views (transpose, slice) when read-only access is sufficient
4. **Batch operations** — Combine multiple operations to reduce temporary allocations
5. **Use appropriate types** — `f32` for memory-constrained or GPU-bound workloads, `f64` for numerical precision

### Integration with Linear Algebra

NDArray is designed to work seamlessly with `zuda.linalg`:

```zig
const linalg = zuda.linalg;

var A = try NDArray(f64, 2).rand(allocator, &.{100, 100}, .row_major, &rng.random());
defer A.deinit();

var b = try NDArray(f64, 1).rand(allocator, &.{100}, .row_major, &rng.random());
defer b.deinit();

// Solve Ax = b
var x = try linalg.solve.solve(f64, A, b, allocator);
defer x.deinit();

// Matrix multiplication (BLAS)
var C = try NDArray(f64, 2).zeros(allocator, &.{100, 100}, .row_major);
defer C.deinit();
try linalg.blas.gemm(f64, 1.0, A, A, 0.0, &C);  // C = A * A
```

## Common Patterns

### Matrix Construction

```zig
// Diagonal matrix
var diag = try NDArray(f64, 2).eye(allocator, n, n, 0, .row_major);
defer diag.deinit();

// Block matrix
var top = try NDArray(f64, 2).ones(allocator, &.{5, 10}, .row_major);
defer top.deinit();
var bottom = try NDArray(f64, 2).zeros(allocator, &.{5, 10}, .row_major);
defer bottom.deinit();
var stacked = try top.vstack(&bottom);
defer stacked.deinit();
```

### Data Pipeline

```zig
// Load → normalize → compute
var data = try NDArray(f64, 2).fromSlice(allocator, raw_data, .row_major);
defer data.deinit();

const mean_val = data.mean();
var centered = try data.addScalar(-mean_val);
defer centered.deinit();

const std_val = @sqrt(centered.mul(&centered).mean());
var normalized = try centered.mulScalar(1.0 / std_val);
defer normalized.deinit();

var result = try normalized.exp();
defer result.deinit();
```

## Error Handling

All NDArray operations return errors for invalid operations:

```zig
// Shape mismatch
var a = try NDArray(f64, 2).zeros(allocator, &.{3, 4}, .row_major);
var b = try NDArray(f64, 2).zeros(allocator, &.{2, 2}, .row_major);
const sum = a.add(&b) catch |err| {
    std.debug.print("Error: {}\n", .{err}); // ShapeMismatch
};

// Index out of bounds
matrix.set(&.{100, 200}, 1.0); // IndexOutOfBounds

// Invalid reshape
const bad = matrix.reshape(&.{7, 7}); // CapacityExceeded
```

## See Also

- [Linear Algebra Guide](linalg.md) — Matrix operations and decompositions
- [Statistics Guide](stats.md) — Statistical functions on arrays
- [NumPy Compatibility](../NUMPY_COMPATIBILITY.md) — NumPy → zuda migration reference
- [API Reference](../API.md) — Complete API documentation
