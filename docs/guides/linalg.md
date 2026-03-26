# Linear Algebra — Matrix Operations and Decompositions

## Overview

The `linalg` module provides comprehensive linear algebra capabilities, from basic vector operations (BLAS) to advanced matrix decompositions and solvers. It's designed for numerical computing, scientific simulation, optimization, and machine learning applications.

## Module Structure

```zig
const linalg = zuda.linalg;

// BLAS operations
const blas = linalg.blas;

// Matrix decompositions
const decomp = linalg.decompositions;

// Linear system solvers
const solve = linalg.solve;

// Matrix properties
const props = linalg.properties;
```

## BLAS — Basic Linear Algebra Subprograms

### Level 1 — Vector Operations

**Dot Product**
```zig
var x = try NDArray(f64, 1).fromSlice(allocator, &[_]f64{1, 2, 3}, .row_major);
defer x.deinit();

var y = try NDArray(f64, 1).fromSlice(allocator, &[_]f64{4, 5, 6}, .row_major);
defer y.deinit();

const result = try blas.dot(f64, x, y);  // 1*4 + 2*5 + 3*6 = 32
```

**Vector Scaling (AXPY)**
```zig
// y = αx + y (in-place)
try blas.axpy(f64, 2.0, x, &y);  // y becomes 2*x + y
```

**Vector Norms**
```zig
const l2_norm = try blas.nrm2(f64, x);     // Euclidean norm: sqrt(Σ x_i²)
const l1_norm = try blas.asum(f64, x);     // Sum of absolute values: Σ |x_i|
```

**In-place Scaling**
```zig
try blas.scal(f64, 0.5, &x);  // x = 0.5 * x
```

### Level 2 — Matrix-Vector Operations

**Matrix-Vector Multiply (GEMV)**
```zig
var A = try NDArray(f64, 2).rand(allocator, &.{100, 50}, .row_major, &rng.random());
defer A.deinit();

var x = try NDArray(f64, 1).ones(allocator, &.{50}, .row_major);
defer x.deinit();

var y = try NDArray(f64, 1).zeros(allocator, &.{100}, .row_major);
defer y.deinit();

// y = 1.0 * A*x + 0.0 * y
try blas.gemv(f64, 1.0, A, x, 0.0, &y);
```

**Symmetric Matrix-Vector Multiply (SYMV)**
```zig
// For symmetric matrices (only upper/lower triangle is used)
try blas.symv(f64, .upper, 1.0, A_sym, x, 0.0, &y);
```

**Rank-1 Update (GER)**
```zig
// A = α * x*y^T + A (outer product update)
try blas.ger(f64, 1.0, x, y, &A);
```

### Level 3 — Matrix-Matrix Operations

**General Matrix Multiply (GEMM)**
```zig
var A = try NDArray(f64, 2).rand(allocator, &.{100, 50}, .row_major, &rng.random());
defer A.deinit();

var B = try NDArray(f64, 2).rand(allocator, &.{50, 80}, .row_major, &rng.random());
defer B.deinit();

var C = try NDArray(f64, 2).zeros(allocator, &.{100, 80}, .row_major);
defer C.deinit();

// C = α*A*B + β*C
try blas.gemm(f64, 1.0, A, B, 0.0, &C);  // C = A*B
```

**Symmetric Matrix Multiply (SYMM)**
```zig
// C = α*A*B + β*C where A is symmetric
try blas.symm(f64, .left, .upper, 1.0, A_sym, B, 0.0, &C);
```

**Symmetric Rank-k Update (SYRK)**
```zig
// C = α*A*A^T + β*C
try blas.syrk(f64, .upper, 1.0, A, 0.0, &C);
```

**Triangular Matrix Multiply (TRMM)**
```zig
// B = α*op(A)*B where A is triangular
try blas.trmm(f64, .left, .upper, .no_trans, .non_unit, 1.0, A_tri, &B);
```

## Matrix Decompositions

### LU Decomposition

Factors A = PLU where P is permutation, L is lower triangular, U is upper triangular.

```zig
var A = try NDArray(f64, 2).fromSlice(allocator, &[_]f64{
    4, 3,
    6, 3,
}, .row_major);
defer A.deinit();
A.shape = .{2, 2};

var result = try decomp.lu(f64, allocator, A);
defer result.L.deinit();
defer result.U.deinit();
defer result.P.deinit();

// Use L and U for solving, determinant, etc.
```

**Use cases**: Solving Ax=b, computing determinant, matrix inversion

**Complexity**: O(n³) time, O(n²) space

### QR Decomposition

Factors A = QR where Q is orthogonal, R is upper triangular.

```zig
var A = try NDArray(f64, 2).rand(allocator, &.{100, 50}, .row_major, &rng.random());
defer A.deinit();

var result = try decomp.qr(f64, A, allocator);
defer result.Q.deinit();
defer result.R.deinit();

// Q is orthonormal: Q^T * Q = I
// R is upper triangular
```

**Use cases**: Least squares, orthogonalization, eigenvalue algorithms

**Complexity**: O(mn²) time for m×n matrix

### Cholesky Decomposition

Factors A = LL^T for symmetric positive-definite matrices.

```zig
var A = try NDArray(f64, 2).fromSlice(allocator, &[_]f64{
    4, 2,
    2, 3,
}, .row_major);
defer A.deinit();
A.shape = .{2, 2};

var result = try decomp.cholesky(f64, allocator, A);
defer result.L.deinit();

// A = L * L^T
```

**Use cases**: Covariance matrices, optimization, Gaussian processes

**Complexity**: O(n³) time, O(n²) space

**Requirement**: Matrix must be symmetric positive-definite

### Singular Value Decomposition (SVD)

Factors A = UΣV^T where U, V are orthogonal and Σ is diagonal.

```zig
var A = try NDArray(f64, 2).rand(allocator, &.{100, 50}, .row_major, &rng.random());
defer A.deinit();

var result = try decomp.svd(f64, allocator, A);
defer result.U.deinit();
defer result.S.deinit();  // Singular values
defer result.Vt.deinit(); // V transpose

// Singular values in descending order
```

**Use cases**: Pseudo-inverse, rank estimation, PCA, low-rank approximation

**Complexity**: O(mn²) time for m×n matrix (m ≥ n)

### Eigendecomposition

Computes eigenvalues and eigenvectors: Av = λv

```zig
var A = try NDArray(f64, 2).rand(allocator, &.{50, 50}, .row_major, &rng.random());
defer A.deinit();

var result = try decomp.eig(f64, allocator, A);
defer result.eigenvalues.deinit();
defer result.eigenvectors.deinit();

// eigenvalues[i] corresponds to eigenvectors column i
```

**Use cases**: Stability analysis, PCA, graph algorithms, quantum mechanics

**Complexity**: O(n³) time for n×n matrix

## Linear System Solvers

### General Solver

Auto-selects the best decomposition based on matrix properties.

```zig
var A = try NDArray(f64, 2).rand(allocator, &.{100, 100}, .row_major, &rng.random());
defer A.deinit();

var b = try NDArray(f64, 1).rand(allocator, &.{100}, .row_major, &rng.random());
defer b.deinit();

// Solve Ax = b
var x = try solve.solve(f64, A, b, allocator);
defer x.deinit();

// Automatically uses:
// - Cholesky for symmetric positive-definite
// - QR for overdetermined systems
// - LU for general square matrices
```

**Complexity**: O(n³) for n×n system

### Least Squares

Solves overdetermined systems (more equations than unknowns).

```zig
// A is m×n with m > n (tall matrix)
var A = try NDArray(f64, 2).rand(allocator, &.{200, 50}, .row_major, &rng.random());
defer A.deinit();

var b = try NDArray(f64, 1).rand(allocator, &.{200}, .row_major, &rng.random());
defer b.deinit();

// Find x that minimizes ||Ax - b||₂
var x = try solve.lstsq(f64, A, b, allocator);
defer x.deinit();
```

**Use cases**: Linear regression, curve fitting, parameter estimation

**Complexity**: O(mn²) for m×n matrix

### Matrix Inverse

Computes A⁻¹ using LU decomposition.

```zig
var A = try NDArray(f64, 2).rand(allocator, &.{50, 50}, .row_major, &rng.random());
defer A.deinit();

var A_inv = try solve.inv(f64, A, allocator);
defer A_inv.deinit();

// Verify: A * A_inv ≈ I
```

**Warning**: Direct inversion is numerically unstable. Prefer solving Ax=b directly.

**Complexity**: O(n³)

### Pseudo-Inverse (Moore-Penrose)

Computes A⁺ for rectangular or rank-deficient matrices.

```zig
var A = try NDArray(f64, 2).rand(allocator, &.{100, 50}, .row_major, &rng.random());
defer A.deinit();

var A_pinv = try solve.pinv(f64, A, allocator);
defer A_pinv.deinit();

// Works for any matrix shape
// For overdetermined: A⁺ = (A^T A)⁻¹ A^T
// For underdetermined: A⁺ = A^T (A A^T)⁻¹
```

**Use cases**: Solving under/overdetermined systems, generalized inverse

**Complexity**: O(mn²) via SVD

## Matrix Properties

### Rank

Count of linearly independent rows/columns.

```zig
const r = try props.rank(f64, A, allocator);
std.debug.print("Matrix rank: {}\n", .{r});
```

**Method**: SVD-based (counts singular values above threshold)

**Complexity**: O(mn²)

### Condition Number

Ratio of largest to smallest singular value: κ(A) = σ_max / σ_min

```zig
const kappa = try props.cond(f64, A, allocator);

if (kappa > 1e12) {
    std.debug.print("Warning: Matrix is ill-conditioned (κ = {e})\n", .{kappa});
}
```

**Interpretation**:
- κ ≈ 1: Well-conditioned (stable)
- κ > 10⁶: Ill-conditioned (numerical issues likely)
- κ = ∞: Singular (not invertible)

**Complexity**: O(mn²)

## Common Patterns

### Solving Multiple Right-Hand Sides

```zig
// Solve AX = B where B has multiple columns
var A = try NDArray(f64, 2).rand(allocator, &.{100, 100}, .row_major, &rng.random());
defer A.deinit();

var B = try NDArray(f64, 2).rand(allocator, &.{100, 10}, .row_major, &rng.random());
defer B.deinit();

// Factor once, solve multiple times
var lu_result = try decomp.lu(f64, allocator, A);
defer lu_result.L.deinit();
defer lu_result.U.deinit();
defer lu_result.P.deinit();

// Use L and U to solve for each column of B
// (Implementation depends on back-substitution helper)
```

### Gram Matrix and Normal Equations

```zig
// Gram matrix: G = A^T * A
var G = try NDArray(f64, 2).zeros(allocator, &.{n, n}, .row_major);
defer G.deinit();

var A_T = A.transpose();  // View, no allocation
try blas.gemm(f64, 1.0, A_T, A, 0.0, &G);

// Normal equations: (A^T A) x = A^T b
var A_Tb = try NDArray(f64, 1).zeros(allocator, &.{n}, .row_major);
defer A_Tb.deinit();
try blas.gemv(f64, 1.0, A_T, b, 0.0, &A_Tb);

var x = try solve.solve(f64, G, A_Tb, allocator);
defer x.deinit();
```

### Orthonormalization

```zig
// QR decomposition gives orthonormal basis
var A = try NDArray(f64, 2).rand(allocator, &.{100, 50}, .row_major, &rng.random());
defer A.deinit();

var result = try decomp.qr(f64, A, allocator);
defer result.Q.deinit();
defer result.R.deinit();

// Q columns are orthonormal: Q^T Q = I
```

### Principal Component Analysis (PCA)

```zig
// 1. Center data: X_centered = X - mean(X)
const mean_val = X.mean();
var X_centered = try X.addScalar(-mean_val);
defer X_centered.deinit();

// 2. Compute covariance matrix: C = (X^T X) / (n-1)
var C = try NDArray(f64, 2).zeros(allocator, &.{n_features, n_features}, .row_major);
defer C.deinit();
const X_T = X_centered.transpose();
try blas.gemm(f64, 1.0 / @as(f64, @floatFromInt(n_samples - 1)), X_T, X_centered, 0.0, &C);

// 3. Eigendecomposition of C
var eig_result = try decomp.eig(f64, allocator, C);
defer eig_result.eigenvalues.deinit();
defer eig_result.eigenvectors.deinit();

// 4. Principal components are eigenvectors with largest eigenvalues
// (sorted in descending order)
```

## Performance Tips

1. **Choose the right decomposition**:
   - **Cholesky** for symmetric positive-definite (fastest)
   - **QR** for least squares and orthogonalization
   - **LU** for general square systems
   - **SVD** for pseudo-inverse and rank (slowest but most general)

2. **Reuse decompositions**: If solving multiple systems with the same A, factor once and reuse

3. **Check condition number**: Before solving, verify matrix is well-conditioned (κ < 10⁶)

4. **Use appropriate precision**: `f32` for GPU/memory-constrained, `f64` for numerical stability

5. **Leverage BLAS Level 3**: GEMM operations are highly optimized; restructure algorithms to use them

6. **Memory layout matters**: Row-major (C order) is default; column-major may be faster for some BLAS operations

## Error Handling

All operations validate inputs and return descriptive errors:

```zig
const result = solve.solve(f64, A, b, allocator) catch |err| switch (err) {
    error.DimensionMismatch => {
        std.debug.print("Matrix and vector dimensions incompatible\n", .{});
    },
    error.SingularMatrix => {
        std.debug.print("Matrix is singular (not invertible)\n", .{});
    },
    error.NotPositiveDefinite => {
        std.debug.print("Matrix is not positive-definite (use LU instead)\n", .{});
    },
    else => return err,
};
```

## See Also

- [NDArray Guide](ndarray.md) — N-dimensional array operations
- [Statistics Guide](stats.md) — Statistical functions leveraging linear algebra
- [Optimization Guide](optimize.md) — Nonlinear optimization using linear algebra primitives
- [NumPy Compatibility](../NUMPY_COMPATIBILITY.md) — NumPy linalg → zuda linalg mapping
