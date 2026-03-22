# pinv(A) Test Suite Summary

## Overview
Comprehensive test suite for **Moore-Penrose Pseudo-Inverse** implementation via SVD decomposition.

- **Total Tests**: 26 tests
- **Function Signature**: `pub fn pinv(comptime T: type, A: NDArray(T, 2), allocator: Allocator) !NDArray(T, 2)`
- **Algorithm**: A⁺ = VΣ⁺U^T where Σ⁺[i,i] = 1/σᵢ if σᵢ > tolerance, else 0
- **Tolerance**: max(m,n) × σ_max × machine_epsilon
- **Input**: Any m×n matrix (full rank or rank-deficient)
- **Output**: n×m Moore-Penrose pseudo-inverse

---

## Test Categories & Specifications

### 1. Basic Functionality (6 tests)
Tests core pinv behavior for well-conditioned matrices.

| Test | Input | Expected Output | Checks |
|------|-------|-----------------|--------|
| Full-rank square (3×3) | diag(4,5,6) | diag(0.25, 0.2, 1/6) | Diagonal inverse correctness |
| Full-rank tall (4×2) | 4×2 rank-2 matrix | A⁺ shape 2×4 | Left inverse: A⁺A = I₂ |
| Full-rank wide (2×4) | 2×4 rank-2 matrix | A⁺ shape 4×2 | Right inverse: AA⁺ = I₂ |
| Identity matrix | I₃ | I₃ | pinv(I) = I |
| Diagonal (2×2) | diag(2,3) | diag(0.5, 1/3) | Diagonal inverse |
| 1×1 scalar | [5] | [0.2] | Scalar reciprocal |

**Why**: Validates basic pinv behavior for simple matrices where properties are analytically verifiable.

---

### 2. Rank-Deficient Matrices (5 tests)
Tests pinv on matrices with rank < min(m,n).

| Test | Structure | Key Property Verified |
|------|-----------|----------------------|
| Rank-1 outer product | 3×2, [u×v^T] | AA⁺A = A (rank-1 reconstruction) |
| Zero rows | 4×2, bottom 2 rows = 0 | Correct handling of zero rows in SVD |
| Zero columns | 2×4, rightmost 2 cols = 0 | Correct handling of zero columns |
| All zeros | 3×2, all elements = 0 | pinv(0) = 0 |
| Rank-k (k<n) | 3×3, rank=2 | Moore-Penrose property 1 with larger tolerance |

**Why**: Validates numerically stable handling of singular/near-singular matrices through SVD's truncation mechanism.

---

### 3. Moore-Penrose Properties (4 tests)
Verify the 4 defining properties of pseudo-inverse.

**Property 1: AA⁺A = A** (Tall matrix test)
- Input: 3×2 full-rank matrix
- Compute: AA⁺ (3×3), then (AA⁺)A (3×2)
- Verify: Result ≈ Original within 1e-10

**Property 2: A⁺AA⁺ = A⁺** (Wide matrix test)
- Input: 2×3 full-rank matrix
- Compute: A⁺A (3×3), then (A⁺A)A⁺ (3×2)
- Verify: Result ≈ A⁺ within 1e-10

**Property 3: (AA⁺)^T = AA⁺** (Symmetry)
- Input: 3×2 matrix
- Compute: AA⁺ (3×3)
- Verify: AA⁺[i,j] = AA⁺[j,i] for all i,j

**Property 4: (A⁺A)^T = A⁺A** (Symmetry)
- Input: 2×3 matrix
- Compute: A⁺A (3×3)
- Verify: A⁺A[i,j] = A⁺A[j,i] for all i,j

**Why**: Fundamental mathematical definition — any pseudo-inverse must satisfy all 4 properties.

---

### 4. Rectangular Matrices (2 tests)
Tests extreme aspect ratios.

| Test | Shape | Property |
|------|-------|----------|
| Very tall | 10×2 | AA⁺A = A (tall case) |
| Very wide | 2×10 | AA⁺A = A (wide case) |

**Why**: Verifies SVD computation and pinv assembly for matrices with large dimension differences.

---

### 5. Numerical Precision & Stability (4 tests)
Tests precision across float types and ill-conditioned matrices.

| Test | Input | Tolerance | Key Check |
|------|-------|-----------|-----------|
| f32 precision (3×2) | f32 matrix | 1e-4 | AA⁺A = A with f32 precision |
| f64 precision (4×3) | f64 matrix | 1e-10 | AA⁺A = A with f64 precision |
| Ill-conditioned (4×4 Hilbert) | H₄ (κ ≈ 29000) | 1e-6 | AA⁺A = A with relaxed tolerance |
| Small singular values | diag(1e-10, 1e-15, 1) | SVD tolerance | Singular value cutoff below threshold |

**Why**: Validates numerical stability through tolerance thresholding and precision-appropriate tolerances.

---

### 6. Use Cases (3 tests)
Practical applications of pinv.

#### Least Squares Solution (Overdetermined)
- **Problem**: Ax = b with 4 equations, 2 unknowns
- **Solution**: x = A⁺b
- **Check**: x exists and residual norm ≥ 0 (reasonable solution)
- **Why**: Core use case — solving overdetermined systems minimizing ||Ax - b||₂

#### Minimum Norm Solution (Underdetermined)
- **Problem**: Ax = b with 2 equations, 4 unknowns
- **Solution**: x = A⁺b (minimum norm among infinite solutions)
- **Check**: Ax = b exactly
- **Why**: Core use case — finding solution with minimum ||x||₂

#### Reconstruction Error
- **Property**: ||A - AA⁺A|| < ε
- **Input**: 3×2 matrix
- **Compute**: Frobenius norm of (A - AA⁺A)
- **Check**: Error < 1e-10
- **Why**: Validates projection quality — AA⁺ is optimal rank-k approximation

---

### 7. Memory Safety (2 tests)
Leak detection via std.testing.allocator.

| Test | Matrix | Check |
|------|--------|-------|
| 3×2 allocation | 3×2 matrix | No leaks on deinit |
| 2×4 allocation | 2×4 matrix | No leaks on deinit |

**Why**: Ensures all temporary allocations (SVD results, intermediate matrices) are properly freed.

---

## Test Implementation Pattern

Each test follows this structure:

```zig
test "pinv: [description]" {
    const allocator = testing.allocator;

    // 1. Create input matrix A (m×n)
    var A = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{m, n}, &[_]f64{...}, .row_major);
    defer A.deinit();

    // 2. Compute pseudo-inverse A⁺ (n×m)
    var A_pinv = try pinv(f64, A, allocator);
    defer A_pinv.deinit();

    // 3. Verify expected property (matrix operations, symmetry, values, etc.)
    // 4. Assert with tolerance (1e-10 for f64, 1e-4 for f32)
    try testing.expectApproxEqAbs(expected, actual, tolerance);
}
```

---

## Algorithm Implementation Requirements

The pinv function must:

1. **Compute SVD**
   ```zig
   var svd_result = try decomp.svd(T, A, allocator);
   defer svd_result.deinit();

   const U = svd_result.U;      // m×k left singular vectors
   const S = svd_result.S;      // k singular values (descending)
   const Vt = svd_result.Vt;    // k×n right singular vectors (transposed)
   ```

2. **Compute tolerance**
   ```zig
   const max_dim = @floatFromInt(@max(m, n));
   const sigma_max = S.data[0];
   const eps = if (T == f32) 1.2e-7 else 2.2e-16;
   const tolerance = max_dim * sigma_max * eps;
   ```

3. **Construct Σ⁺ (k×k diagonal)**
   ```zig
   for (0..k) |i| {
       if (S.data[i] > tolerance) {
           Sigma_pinv.data[i * k + i] = 1.0 / S.data[i];
       } else {
           Sigma_pinv.data[i * k + i] = 0;
       }
   }
   ```

4. **Compute A⁺ = V Σ⁺ U^T**
   - V is n×k (transpose of Vt which is k×n)
   - Σ⁺ is k×k diagonal
   - U^T is k×m (transpose of U which is m×k)
   - Result: n×m

5. **Memory cleanup**
   - deinit SVD result
   - deinit all temporary matrices
   - return A⁺ to caller

---

## Expected Behavior After Implementation

```zig
// Full-rank square
var A = /* 3×3 diagonal matrix */;
var A_pinv = try pinv(f64, A, allocator);
// A_pinv shape: 3×3 (matches A)
// A_pinv.data contains inverse

// Tall matrix (overdetermined)
var A = /* 4×2 full-rank matrix */;
var A_pinv = try pinv(f64, A, allocator);
// A_pinv shape: 2×4
// A⁺A ≈ I₂, AA⁺ ≠ I₄ (left inverse)

// Wide matrix (underdetermined)
var A = /* 2×4 full-rank matrix */;
var A_pinv = try pinv(f64, A, allocator);
// A_pinv shape: 4×2
// A⁺A ≠ I₄, AA⁺ ≈ I₂ (right inverse)

// Rank-deficient
var A = /* rank-1 3×3 matrix */;
var A_pinv = try pinv(f64, A, allocator);
// A_pinv shape: 3×3
// AA⁺A ≈ A (property 1)
```

---

## File Locations

- **Test file**: `/Users/fn/codespace/zuda/pinv_tests_only.zig`
  - 26 test functions ready to append to `src/linalg/solve.zig`

- **Reference implementation**: `/Users/fn/codespace/zuda/pinv_tests.zig`
  - Includes standalone pinv() function for reference
  - Can be run independently if needed

- **Integration point**: End of `src/linalg/solve.zig` (after inv tests, line 2269)

---

## Notes for Implementation

- **Error handling**: No special error cases — pinv is defined for any matrix via SVD
- **Allocation failures**: Propagate from SVD and matrix operations
- **Tolerance formula**: Critical for rank-deficient matrices; ensures numerical stability
- **Test tolerances**:
  - f64: 1e-10 (full precision)
  - f64 (ill-conditioned): 1e-6 (relaxed for κ > 1000)
  - f32: 1e-4 (limited precision)
- **Temporary matrices**: Must be deinitialized to prevent memory leaks (detected by std.testing.allocator)
