# Linear Algebra API Reference

## Overview

The Linear Algebra module provides BLAS operations (Basic Linear Algebra Subprograms), matrix decompositions, linear solvers, and matrix properties computation. All operations are implemented with numerical stability and precision in mind, supporting both f32 and f64 floating-point types.

### Import

```zig
const zuda = @import("zuda");
const linalg = zuda.linalg;
const blas = linalg.blas;
const decompositions = linalg.decompositions;
const solve_mod = linalg.solve;
const properties = linalg.properties;
```

### Key Features

- **BLAS Level 1, 2, 3**: Vector-vector, matrix-vector, and matrix-matrix operations
- **Matrix Decompositions**: LU, QR, Cholesky, SVD, Eigenvalue
- **Linear System Solvers**: Direct solvers for square and overdetermined systems
- **Matrix Inverse**: Standard and pseudo-inverse (Moore-Penrose)
- **Matrix Properties**: Rank, condition number, trace, determinant, norms
- **Numerical Stability**: Backward stable algorithms with careful error handling
- **Type Generic**: Support for f32, f64, and extensible to other numeric types

---

## Error Types

All linalg operations use consistent error types:

```zig
pub const Error = error{
    DimensionMismatch,          // Shape incompatibility
    SingularMatrix,             // Matrix is rank-deficient
    NonSquareMatrix,            // Operation requires square matrix
    NotPositiveDefinite,        // Cholesky requires SPD matrix
    InvalidDimensions,          // Invalid shape for operation
    InvalidFormat,              // File/format parsing error
    OutOfMemory,                // Allocator failure
    NonSymmetricMatrix,         // Eigenvalue requires symmetric
};
```

---

## BLAS Level 1: Vector-Vector Operations

### dot(T, x, y)

Compute inner product (dot product) of two vectors.

```zig
pub fn dot(comptime T: type, x: NDArray(T, 1), y: NDArray(T, 1))
    (NDArray(T, 1).Error)!T
```

**Description**: Computes the inner product x·y = Σ xᵢyᵢ

**Parameters**:
- `T`: Numeric type (f32, f64)
- `x`: First vector (1D NDArray of length n)
- `y`: Second vector (1D NDArray of length n)

**Returns**: Scalar result of x·y

**Errors**:
- `error.DimensionMismatch`: x.shape[0] != y.shape[0]

**Time**: O(n) where n = vector length
**Space**: O(1)

**Example**:
```zig
var x = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{3}, &[_]f64{1, 2, 3}, .row_major);
defer x.deinit();
var y = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{3}, &[_]f64{4, 5, 6}, .row_major);
defer y.deinit();
const result = try dot(f64, x, y); // 1*4 + 2*5 + 3*6 = 32
```

---

### axpy(T, alpha, x, y)

Vector update: y = αx + y (in-place)

```zig
pub fn axpy(comptime T: type, alpha: T, x: NDArray(T, 1), y: *NDArray(T, 1))
    (NDArray(T, 1).Error)!void
```

**Description**: Scales vector x by scalar α and adds it to y, storing result in-place in y.
This is the fundamental BLAS axpy operation (a times x plus y).

**Parameters**:
- `T`: Numeric type (f32, f64)
- `alpha`: Scalar multiplier for x
- `x`: First vector (1D NDArray, not modified)
- `y`: Second vector (1D NDArray, modified in-place)

**Errors**:
- `error.DimensionMismatch`: x.shape[0] != y.shape[0]

**Time**: O(n) where n = vector length
**Space**: O(1) (modifies y in-place)

**Example**:
```zig
var x = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{3}, &[_]f64{1, 2, 3}, .row_major);
defer x.deinit();
var y = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{3}, &[_]f64{4, 5, 6}, .row_major);
defer y.deinit();
try axpy(f64, 2.0, x, &y); // y = 2*{1,2,3} + {4,5,6} = {6,9,12}
```

---

### nrm2(T, x)

Compute Euclidean norm (L2 norm) of a vector.

```zig
pub fn nrm2(comptime T: type, x: NDArray(T, 1))
    (NDArray(T, 1).Error)!T
```

**Description**: Returns the length of the vector: √(Σ xᵢ²)

**Parameters**:
- `T`: Numeric type (f32, f64)
- `x`: Vector (1D NDArray)

**Returns**: Non-negative scalar norm value

**Time**: O(n) where n = vector length
**Space**: O(1)

**Example**:
```zig
var x = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{3}, &[_]f64{3, 4, 0}, .row_major);
defer x.deinit();
const norm = try nrm2(f64, x); // √(9 + 16 + 0) = 5
```

---

### asum(T, x)

Sum of absolute values of vector elements.

```zig
pub fn asum(comptime T: type, x: NDArray(T, 1))
    (NDArray(T, 1).Error)!T
```

**Description**: Computes Σ |xᵢ| for all elements in x

**Parameters**:
- `T`: Numeric type (f32, f64)
- `x`: Vector (1D NDArray)

**Returns**: Non-negative scalar sum of absolute values

**Time**: O(n) where n = vector length
**Space**: O(1)

**Example**:
```zig
var x = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{4}, &[_]f64{1, -2, 3, -4}, .row_major);
defer x.deinit();
const sum = try asum(f64, x); // |1| + |-2| + |3| + |-4| = 10
```

---

### scal(T, alpha, x)

Scale vector in-place: x = αx

```zig
pub fn scal(comptime T: type, alpha: T, x: *NDArray(T, 1))
    (NDArray(T, 1).Error)!void
```

**Description**: Multiplies all elements of x by scalar α in-place.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `alpha`: Scalar multiplier
- `x`: Vector (1D NDArray, modified in-place)

**Time**: O(n) where n = vector length
**Space**: O(1) (modifies x in-place)

**Example**:
```zig
var x = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{3}, &[_]f64{1, 2, 3}, .row_major);
defer x.deinit();
try scal(f64, 2.5, &x); // x = {2.5, 5.0, 7.5}
```

---

## BLAS Level 2: Matrix-Vector Operations

### gemv(T, alpha, A, x, beta, y)

General matrix-vector multiply: y = αAx + βy

```zig
pub fn gemv(comptime T: type, alpha: T, A: NDArray(T, 2), x: NDArray(T, 1),
            beta: T, y: *NDArray(T, 1))
    (NDArray(T, 1).Error)!void
```

**Description**: Performs matrix-vector multiplication with scalar scaling.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `alpha`: Scalar multiplier for Ax
- `A`: Matrix (m×n)
- `x`: Vector (n×1)
- `beta`: Scalar multiplier for y
- `y`: Result vector (m×1, modified in-place)

**Errors**:
- `error.DimensionMismatch`: A.shape[1] != x.shape[0] or A.shape[0] != y.shape[0]

**Time**: O(m·n) where A is m×n
**Space**: O(1) (modifies y in-place)

**Example**:
```zig
// A = [[1, 2], [3, 4], [5, 6]]  (3×2)
// x = [7, 8]  (2×1)
// y = [1, 1, 1]  (3×1)
// gemv(2.0, A, x, 3.0, &y)
// y = 2.0*A*x + 3.0*y = 2.0*[23, 53, 83] + [3,3,3] = [49, 109, 169]
```

---

### ger(T, alpha, x, y, A)

Rank-1 update: A = αxy^T + A

```zig
pub fn ger(comptime T: type, alpha: T, x: NDArray(T, 1), y: NDArray(T, 1),
           A: *NDArray(T, 2))
    (NDArray(T, 1).Error)!void
```

**Description**: Computes rank-1 outer product update to matrix A.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `alpha`: Scalar multiplier
- `x`: Vector (m×1)
- `y`: Vector (n×1)
- `A`: Matrix (m×n, modified in-place)

**Errors**:
- `error.DimensionMismatch`: Dimension incompatibility

**Time**: O(m·n)
**Space**: O(1) (modifies A in-place)

---

## BLAS Level 3: Matrix-Matrix Operations

### gemm(T, alpha, A, B, beta, C)

General matrix-matrix multiply: C = αAB + βC

```zig
pub fn gemm(comptime T: type, alpha: T, A: NDArray(T, 2), B: NDArray(T, 2),
            beta: T, C: *NDArray(T, 2))
    (NDArray(T, 2).Error)!void
```

**Description**: Performs matrix-matrix multiplication with scalar scaling.
Standard Level 3 BLAS operation with O(n³) complexity optimized for cache locality.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `alpha`: Scalar multiplier for AB
- `A`: Matrix (m×k)
- `B`: Matrix (k×n)
- `beta`: Scalar multiplier for C
- `C`: Result matrix (m×n, modified in-place)

**Errors**:
- `error.DimensionMismatch`: Incompatible dimensions

**Time**: O(m·n·k) where A is m×k, B is k×n, C is m×n
**Space**: O(1) (modifies C in-place)

**Example**:
```zig
// A = [[1, 2], [3, 4]]  (2×2)
// B = [[5, 6], [7, 8]]  (2×2)
// C = [[1, 1], [1, 1]]  (2×2)
// AB = [[19, 22], [43, 50]]
// C = 1.0*AB + 1.0*C = [[20, 23], [44, 51]]
try gemm(f64, 1.0, A, B, 1.0, &C);
```

---

## Matrix Decompositions

### LU Decomposition

```zig
pub const LUResult = struct {
    P: NDArray(T, 2),       // Permutation matrix
    L: NDArray(T, 2),       // Lower triangular (unit diagonal)
    U: NDArray(T, 2),       // Upper triangular
    allocator: Allocator,
    pub fn deinit(self: *@This()) void
};

pub fn lu(comptime T: type, allocator: Allocator, A: NDArray(T, 2))
    LUResult(T)
```

**Description**: Computes LU decomposition with partial pivoting: A = PLU

- P: Permutation matrix (row pivoting)
- L: Lower triangular with unit diagonal
- U: Upper triangular

**Parameters**:
- `T`: Numeric type (f32, f64)
- `allocator`: Memory allocator
- `A`: Input matrix (n×n)

**Returns**: LUResult containing P, L, U matrices

**Errors**: `error.SingularMatrix`, `error.NonSquareMatrix`, `error.OutOfMemory`

**Time**: O(n³) via Gaussian elimination
**Space**: O(n²) for P, L, U matrices

**Example**:
```zig
var A = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{2, 2},
    &[_]f64{4, 3, 6, 3}, .row_major);
defer A.deinit();
var result = try lu(f64, alloc, A);
defer result.deinit();
// A ≈ P @ L @ U
```

---

### QR Decomposition

```zig
pub const QRResult = struct {
    Q: NDArray(T, 2),       // Orthogonal matrix (m×m)
    R: NDArray(T, 2),       // Upper triangular (m×n)
    allocator: Allocator,
    pub fn deinit(self: *@This()) void
};

pub fn qr(comptime T: type, A: NDArray(T, 2), allocator: Allocator)
    QRResult(T)
```

**Description**: Computes QR decomposition using Householder reflections: A = QR

- Q: Orthogonal matrix (satisfies Q^T @ Q = I)
- R: Upper triangular matrix

**Parameters**:
- `T`: Numeric type (f32, f64)
- `A`: Input matrix (m×n, must have m ≥ n)
- `allocator`: Memory allocator

**Returns**: QRResult containing Q (m×m) and R (m×n) matrices

**Errors**: `error.InvalidDimensions` if m < n, `error.OutOfMemory`

**Time**: O(m·n²)
**Space**: O(m²) for Q and O(m·n) for R

**Mathematical Properties**:
- Q^T @ Q = I (orthonormality)
- R is upper triangular
- ||A - Q @ R|| < machine epsilon

**Example**:
```zig
var A = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{4, 2},
    &[_]f64{1, 0, 1, 1, 0, 1, 0, 0}, .row_major);
defer A.deinit();
var result = try qr(f64, A, alloc);
defer result.deinit();
// Verify: A ≈ Q @ R, Q^T @ Q = I, R is upper triangular
```

---

### Cholesky Decomposition

```zig
pub fn cholesky(comptime T: type, A: NDArray(T, 2), allocator: Allocator)
    NDArray(T, 2)
```

**Description**: Computes Cholesky decomposition: A = LL^T for SPD matrices

**Parameters**:
- `T`: Numeric type (f32, f64)
- `A`: Symmetric positive definite matrix (n×n)
- `allocator`: Memory allocator

**Returns**: Lower triangular matrix L such that A = LL^T

**Errors**:
- `error.NotPositiveDefinite`: Matrix is not SPD
- `error.NonSquareMatrix`: Matrix is not square
- `error.OutOfMemory`: Allocator failure

**Time**: O(n³)
**Space**: O(n²)

**Stability**: Numerically stable for well-conditioned SPD matrices

**Example**:
```zig
// A = [[4, 2], [2, 3]] (SPD)
var A = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{2, 2},
    &[_]f64{4, 2, 2, 3}, .row_major);
defer A.deinit();
var L = try cholesky(f64, A, alloc);
defer L.deinit();
// L = [[2, 0], [1, √2]], A = L @ L^T
```

---

### SVD (Singular Value Decomposition)

```zig
pub const SVDResult = struct {
    U: NDArray(T, 2),       // Left singular vectors (m×k)
    S: NDArray(T, 1),       // Singular values (k,) sorted descending
    Vt: NDArray(T, 2),      // Right singular vectors transpose (k×n)
    allocator: Allocator,
    pub fn deinit(self: *@This()) void
};

pub fn svd(comptime T: type, A: NDArray(T, 2), allocator: Allocator)
    SVDResult(T)
```

**Description**: Computes Singular Value Decomposition: A = UΣV^T

- U: Left singular vectors (orthonormal columns)
- S: Singular values (non-negative, sorted descending)
- Vt: Right singular vectors (transposed)

**Parameters**:
- `T`: Numeric type (f32, f64)
- `A`: Input matrix (m×n)
- `allocator`: Memory allocator

**Returns**: SVDResult containing U (m×k), S (k), Vt (k×n)

**Time**: O(m·n·min(m,n))
**Space**: O(m·n + min(m,n)²)

**Applications**:
- Rank computation
- Condition number estimation
- Pseudo-inverse (Moore-Penrose)
- Principal Component Analysis (PCA)

**Example**:
```zig
var A = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{3, 2},
    &[_]f64{1, 0, 0, 1, 1, 1}, .row_major);
defer A.deinit();
var result = try svd(f64, A, alloc);
defer result.deinit();
// Verify: ||A - U @ diag(S) @ Vt|| < epsilon
```

---

### Eigenvalue Decomposition

```zig
pub const EigResult = struct {
    values: NDArray(T, 1),  // Eigenvalues (real for symmetric)
    vectors: NDArray(T, 2), // Eigenvectors (n×n)
    allocator: Allocator,
    pub fn deinit(self: *@This()) void
};

pub fn eig(comptime T: type, A: NDArray(T, 2), allocator: Allocator)
    EigResult(T)
```

**Description**: Computes eigenvalues and eigenvectors of a symmetric matrix

**Parameters**:
- `T`: Numeric type (f32, f64)
- `A`: Symmetric input matrix (n×n)
- `allocator`: Memory allocator

**Returns**: EigResult containing eigenvalues and eigenvectors

**Errors**:
- `error.InvalidDimensions`: Matrix is not square
- `error.NonSymmetricMatrix`: Matrix is not symmetric
- `error.OutOfMemory`: Allocator failure

**Time**: O(n³)
**Space**: O(n²)

**Properties**:
- For symmetric A: all eigenvalues are real
- Eigenvectors form orthonormal basis
- A = V @ Λ @ V^T where Λ = diag(eigenvalues)

---

## Linear System Solvers

### solve(T, A, b, allocator)

Solve linear system Ax = b using appropriate decomposition

```zig
pub fn solve(comptime T: type, A: NDArray(T, 2), b: NDArray(T, 1),
             allocator: Allocator)
    NDArray(T, 1)
```

**Description**: Auto-selects solver based on matrix properties:
- Square SPD: Cholesky decomposition
- Square general: LU with partial pivoting
- Tall (m > n): QR least squares

**Parameters**:
- `T`: Numeric type (f32, f64)
- `A`: Coefficient matrix (m×n)
- `b`: Right-hand side vector (m×1)
- `allocator`: Memory allocator

**Returns**: Solution vector x (n×1) such that Ax ≈ b

**Errors**:
- `error.DimensionMismatch`: A.shape[0] != b.shape[0]
- `error.UnderdeterminedSystem`: A has more columns than rows (wide)
- `error.SingularMatrix`: A is rank-deficient
- `error.NotPositiveDefinite`: Cholesky failed for SPD matrix

**Time**: O(n³) decomposition + O(n²) back-substitution
**Space**: O(n²)

**Example**:
```zig
// Solve 2x + y = 3, x + 2y = 3
var A = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{2, 2},
    &[_]f64{2, 1, 1, 2}, .row_major);
defer A.deinit();
var b = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{2}, &[_]f64{3, 3}, .row_major);
defer b.deinit();
var x = try solve(f64, A, b, alloc);
defer x.deinit();
// x ≈ [1, 1]
```

---

### lstsq(T, A, b, allocator)

Solve least squares problem: minimize ||Ax - b||₂

```zig
pub fn lstsq(comptime T: type, A: NDArray(T, 2), b: NDArray(T, 1),
             allocator: Allocator)
    NDArray(T, 1)
```

**Description**: Solves overdetermined system using QR decomposition.
For tall (m ≥ n) matrices, finds x that minimizes the Euclidean norm of residual Ax - b.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `A`: Coefficient matrix (m×n, must have m ≥ n)
- `b`: Right-hand side vector (m×1)
- `allocator`: Memory allocator

**Returns**: Solution vector x (n×1) minimizing ||Ax - b||₂

**Errors**:
- `error.DimensionMismatch`: A.shape[0] != b.shape[0]
- `error.InvalidDimensions`: A has more columns than rows
- `error.SingularMatrix`: A is rank-deficient

**Time**: O(m·n²) for QR + O(n²) for back-substitution
**Space**: O(m·n)

**Example**:
```zig
// Overdetermined system: 3 equations, 2 unknowns
var A = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{3, 2},
    &[_]f64{1, 0, 0, 1, 1, 1}, .row_major);
defer A.deinit();
var b = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{3},
    &[_]f64{1, 1, 2}, .row_major);
defer b.deinit();
var x = try lstsq(f64, A, b, alloc);
defer x.deinit();
// Least squares solution
```

---

## Matrix Inverse

### inv(T, A, allocator)

Compute matrix inverse A⁻¹ via LU decomposition

```zig
pub fn inv(comptime T: type, A: NDArray(T, 2), allocator: Allocator)
    NDArray(T, 2)
```

**Description**: Solves AX = I column-by-column using LU factorization with partial pivoting.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `A`: Square matrix (n×n)
- `allocator`: Memory allocator

**Returns**: Inverse matrix A⁻¹ such that A @ A⁻¹ = I and A⁻¹ @ A = I

**Errors**:
- `error.NonSquareMatrix`: A is not square
- `error.SingularMatrix`: A is not invertible (det(A) = 0)
- `error.OutOfMemory`: Allocator failure

**Time**: O(n³) for LU + O(n³) for n back-substitutions
**Space**: O(n²)

**Precision**:
- f32: tolerance 1e-5
- f64: tolerance 1e-10

**Example**:
```zig
var A = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{2, 2},
    &[_]f64{4, 7, 2, 6}, .row_major);
defer A.deinit();
var A_inv = try inv(f64, A, alloc);
defer A_inv.deinit();
// A @ A_inv ≈ I
```

---

### pinv(T, A, allocator)

Compute Moore-Penrose pseudo-inverse A⁺ via SVD

```zig
pub fn pinv(comptime T: type, A: NDArray(T, 2), allocator: Allocator)
    NDArray(T, 2)
```

**Description**: Computes A⁺ = VΣ⁺U^T where Σ⁺[i,i] = 1/σᵢ if σᵢ > tolerance, else 0

Works for any matrix shape (square, tall, wide, rank-deficient)

**Parameters**:
- `T`: Numeric type (f32, f64)
- `A`: Input matrix (m×n)
- `allocator`: Memory allocator

**Returns**: A⁺ (n×m) pseudo-inverse matrix

**Time**: O(m·n²) for SVD + O(m·n·k) for pseudo-inverse computation
**Space**: O(m·min(m,n) + min(m,n)·n)

**Mathematical Properties**:
- A @ A⁺ @ A = A
- A⁺ @ A @ A⁺ = A⁺
- (A @ A⁺)^T = A @ A⁺ (projection is symmetric)
- (A⁺ @ A)^T = A⁺ @ A (projection is symmetric)

**Applications**:
- Solving rank-deficient systems
- Minimum norm least squares solution
- Generalized matrix inverse

**Example**:
```zig
// Rank-deficient matrix (wide)
var A = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{2, 3},
    &[_]f64{1, 0, 1, 0, 1, 1}, .row_major);
defer A.deinit();
var A_pinv = try pinv(f64, A, alloc); // 3×2 matrix
defer A_pinv.deinit();
```

---

## Matrix Properties

### rank(T, A, allocator)

Compute numerical rank of a matrix via SVD

```zig
pub fn rank(comptime T: type, A: NDArray(T, 2), allocator: Allocator)
    usize
```

**Description**: Counts the number of singular values greater than a tolerance threshold.
Tolerance = max(m,n) × σ_max × machine_epsilon

**Parameters**:
- `T`: Numeric type (f32, f64)
- `A`: Input matrix (m×n)
- `allocator`: Memory allocator for SVD computation

**Returns**: Number of singular values above tolerance (0 ≤ rank ≤ min(m,n))

**Errors**: `error.OutOfMemory` if SVD allocation fails

**Time**: O(m·n²) for SVD computation
**Space**: O(m·n) for SVD matrices

**Example**:
```zig
// Full rank identity matrix
var I = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{3, 3},
    &[_]f64{1, 0, 0, 0, 1, 0, 0, 0, 1}, .row_major);
defer I.deinit();
const r = try rank(f64, I, alloc); // r == 3

// Rank-deficient matrix
var A = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{3, 2},
    &[_]f64{1, 1, 2, 2, 3, 3}, .row_major);
defer A.deinit();
const r2 = try rank(f64, A, alloc); // r2 == 1
```

---

### cond(T, A, allocator)

Compute condition number κ(A) = σ_max / σ_min via SVD

```zig
pub fn cond(comptime T: type, A: NDArray(T, 2), allocator: Allocator)
    T
```

**Description**: Measures sensitivity of solution to perturbations in input.
- κ ≈ 1: Well-conditioned (small errors don't amplify)
- κ ≫ 1: Ill-conditioned (errors amplify significantly)
- κ = +∞: Singular matrix (σ_min = 0)

**Parameters**:
- `T`: Numeric type (f32, f64)
- `A`: Input matrix (m×n)
- `allocator`: Memory allocator for SVD computation

**Returns**: Condition number κ(A) (or +∞ if singular)

**Errors**: `error.OutOfMemory` if SVD allocation fails

**Time**: O(m·n²) for SVD computation
**Space**: O(m·n) for SVD matrices

**Example**:
```zig
// Well-conditioned: identity matrix
var I = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{3, 3},
    &[_]f64{1, 0, 0, 0, 1, 0, 0, 0, 1}, .row_major);
defer I.deinit();
const c1 = try cond(f64, I, alloc); // c1 == 1.0

// Ill-conditioned: nearly singular
var A = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{2, 2},
    &[_]f64{1, 1, 1, 1.0001}, .row_major);
defer A.deinit();
const c2 = try cond(f64, A, alloc); // c2 >> 1
```

---

### trace(T, A)

Compute trace (sum of diagonal elements)

```zig
pub fn trace(comptime T: type, A: NDArray(T, 2))
    (NDArray(T, 2).Error)!T
```

**Description**: Returns the sum of diagonal elements: tr(A) = Σ A[i,i]

**Parameters**:
- `T`: Numeric type (f32, f64)
- `A`: Square matrix (n×n)

**Returns**: Scalar trace value

**Errors**: `error.DimensionMismatch` if matrix is not square

**Time**: O(n) where n = matrix dimension
**Space**: O(1)

**Mathematical Properties**:
- tr(A + B) = tr(A) + tr(B) (additivity)
- tr(cA) = c·tr(A) (homogeneity)
- tr(AB) = tr(BA) (cyclic property)
- tr(A) = Σ λᵢ (sum of eigenvalues)

**Example**:
```zig
var A = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{2, 2},
    &[_]f64{1, 2, 3, 4}, .row_major);
defer A.deinit();
const tr = try trace(f64, A); // 1 + 4 = 5
```

---

### det(T, A)

Compute determinant via LU decomposition

```zig
pub fn det(comptime T: type, A: NDArray(T, 2))
    (NDArray(T, 2).Error)!T
```

**Description**: Computes det(A) using in-place LU factorization with partial pivoting.
Determinant is the product of diagonal elements with sign correction from row swaps.

**Parameters**:
- `T`: Numeric type (f32, f64)
- `A`: Square matrix (n×n)

**Returns**: Scalar determinant value

**Errors**: `error.DimensionMismatch` if matrix is not square

**Time**: O(n³) for LU decomposition
**Space**: O(n²) for LU matrix copy (original A is not modified)

**Special Cases**:
- det(I) = 1 (identity)
- det(A) = 0 ⟺ A is singular
- det(cA) = c^n × det(A) for scalar c and n×n matrix
- det(AB) = det(A) × det(B)

**Example**:
```zig
var A = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{2, 2},
    &[_]f64{1, 2, 3, 4}, .row_major);
defer A.deinit();
const d = try det(f64, A); // 1*4 - 2*3 = -2
```

---

### norm1(T, x), norm2(T, x), normInf(T, x)

Vector norms

```zig
pub fn norm1(comptime T: type, x: NDArray(T, 1))
    (NDArray(T, 1).Error)!T
pub fn norm2(comptime T: type, x: NDArray(T, 1))
    (NDArray(T, 1).Error)!T
pub fn normInf(comptime T: type, x: NDArray(T, 1))
    (NDArray(T, 1).Error)!T
```

**Description**:
- **norm1**: L1 norm = Σ |xᵢ| (Manhattan distance)
- **norm2**: L2 norm = √(Σ xᵢ²) (Euclidean distance)
- **normInf**: L∞ norm = max |xᵢ| (Max absolute value)

**Parameters**:
- `T`: Numeric type (f32, f64)
- `x`: Vector (1D NDArray)

**Returns**: Non-negative scalar norm value

**Time**: O(n)
**Space**: O(1)

**Example**:
```zig
var x = try NDArray(f64, 1).fromSlice(alloc, &[_]usize{3},
    &[_]f64{3, -4, 0}, .row_major);
defer x.deinit();
const n1 = try norm1(f64, x);     // 3 + 4 + 0 = 7
const n2 = try norm2(f64, x);     // √(9+16+0) = 5
const ninf = try normInf(f64, x); // max(3,4,0) = 4
```

---

### normFrobenius(T, A)

Frobenius norm of a matrix

```zig
pub fn normFrobenius(comptime T: type, A: NDArray(T, 2))
    (NDArray(T, 2).Error)!T
```

**Description**: Computes ||A||_F = √(Σ_ij A[i,j]²)

**Parameters**:
- `T`: Numeric type (f32, f64)
- `A`: Matrix (m×n)

**Returns**: Non-negative scalar Frobenius norm

**Time**: O(m·n)
**Space**: O(1)

**Properties**:
- ||A||_F = √(Σ σᵢ²) (sum of squared singular values)
- ||A||_F² = tr(A^T @ A)

**Example**:
```zig
var A = try NDArray(f64, 2).fromSlice(alloc, &[_]usize{2, 2},
    &[_]f64{3, 4, 0, 0}, .row_major);
defer A.deinit();
const nf = try normFrobenius(f64, A); // √(9+16+0+0) = 5
```

---

## SIMD-Accelerated Operations

For high-performance scientific computing, SIMD-accelerated variants are available:

```zig
pub fn gemm_simd(comptime T: type, alpha: T, A: NDArray(T, 2),
                 B: NDArray(T, 2), beta: T, C: *NDArray(T, 2))
pub fn dot_simd(comptime T: type, x: NDArray(T, 1), y: NDArray(T, 1)) T
pub fn axpy_simd(comptime T: type, alpha: T, x: NDArray(T, 1),
                 y: *NDArray(T, 1)) void
```

These use platform-specific SIMD instructions when available, providing 2-8x speedup for large matrices while maintaining numerical equivalence with standard implementations.

---

## Practical Examples

### Solving a Linear System

```zig
const std = @import("std");
const zuda = @import("zuda");
const NDArray = zuda.ndarray.NDArray;
const linalg = zuda.linalg.solve;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Solve: 2x + y = 5, x + 3y = 6
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{2, 2},
        &[_]f64{2, 1, 1, 3}, .row_major);
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2},
        &[_]f64{5, 6}, .row_major);
    defer b.deinit();

    var x = try linalg.solve(f64, A, b, allocator);
    defer x.deinit();

    std.debug.print("x = [{d}, {d}]\n", .{x.data[0], x.data[1]});
    // Output: x = [2.0, 1.0]
}
```

### Computing Matrix Properties

```zig
const properties = zuda.linalg.properties;

// Check if matrix is invertible
const r = try properties.rank(f64, A, allocator);
const is_full_rank = r == @min(A.shape[0], A.shape[1]);

// Assess numerical stability
const condition = try properties.cond(f64, A, allocator);
const is_ill_conditioned = condition > 1e10;
```

### Least Squares Fitting

```zig
const linalg = zuda.linalg.solve;

// Fit line y = mx + b through n points
// A = [[x1, 1], [x2, 1], ..., [xn, 1]]  (n×2)
// b = [y1, y2, ..., yn]  (n×1)
var x = try linalg.lstsq(f64, A, b, allocator);
defer x.deinit();
const m = x.data[0]; // slope
const c = x.data[1]; // intercept
```

---

## Performance Considerations

1. **Matrix Size**: Algorithms are O(n³), so doubling matrix size increases runtime 8x
2. **Data Layout**: Row-major (C) order is default; verify alignment for performance
3. **Decomposition Reuse**: Store decomposition results (LU, QR, SVD) to solve multiple systems
4. **SIMD Variants**: Use `gemm_simd` for matrices > 64×64 for significant speedup
5. **Memory**: SVD requires O(m·n) temporary storage; allocate sufficient memory

---

## Numerical Stability Notes

- **LU**: Partial pivoting provides backward stability
- **QR**: Householder reflections are backward stable
- **Cholesky**: Stable only for well-conditioned SPD matrices
- **SVD**: Backward stable; used for rank and condition estimation
- **Conditioning**: Check `cond(A)` before solving ill-conditioned systems

---

## References

- Golub & Van Loan. *Matrix Computations* (4th ed.). Johns Hopkins University Press.
- BLAS Standard: https://www.netlib.org/blas/
- LAPACK Documentation: https://www.netlib.org/lapack/
