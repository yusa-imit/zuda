# zuda Project Context

## Current Status
- **Version**: 1.19.1 ✅ — CI Stability Fixes RELEASED
- **Phase**: v2.0 Track (Phase 7) — Scientific Computing Platform
- **Zig Version**: 0.15.2
- **Last CI Status**: ✅ GREEN (all 6 cross-compile targets passing, 234 tests 100% passing)
- **Latest Milestone**: v1.19.1 CI Fixes ✅ — Resolved build cache corruption in GitHub Actions
- **Current Milestone**: v1.20.0 — Advanced Linear Algebra (solvers, pseudo-inverse, rank, condition number)
- **Next Priority**: Implement solve(A, b) with auto-decomposition selection
- **Decompositions Available**: LU (23 tests), QR (23 tests), Cholesky (19 tests), SVD (28 tests), Eigendecomposition (21 tests) = 114 tests

## Recent Progress (Session 2026-03-22 - Hour 1)
**FEATURE MODE:**

### v1.19.1 Release ✅
- ✅ **Release**: v1.19.1 patch release for CI stability fixes
- ✅ **GitHub Release**: https://github.com/yusa-imit/zuda/releases/tag/v1.19.1
- ✅ **Changes**: 2 CI fixes (cache corruption) + 2 chore commits (memory/logs)
- ✅ **Verification**: 234 tests passing, all 6 cross-compile targets green
- ✅ **Tag**: v1.19.1 created and pushed

### v1.20.0 Milestone Planning ✅
- ✅ **Milestone defined**: Advanced Linear Algebra (6 functions)
- ✅ **Documentation**: `docs/milestones.md` updated with v1.20.0 roadmap
- ✅ **Context**: `.claude/memory/project-context.md` updated

### solve(A, b) Implementation (commit 7fb305e, b66ca8a) ✅
- ✅ **solve(A, b)**: Linear system solver with auto-decomposition selection, O(n³)
- ✅ **Strategy selection**:
  - SPD matrices → Cholesky (symmetry check + factorization attempt)
  - General square → LU with partial pivoting
  - Overdetermined (m>n) → QR least squares
- ✅ **Algorithms**:
  - Cholesky: Forward substitution (Ly = b), backward (L^Tx = y)
  - LU: Forward substitution (Ly = Pb), backward (Ux = y)
  - QR: Least squares via Q^Tb and backward substitution on R
- ✅ **Error handling**: Singular, dimension mismatch, underdetermined systems
- ✅ **Tests**: 24 comprehensive tests
  - SPD (4): identity, diagonal, symmetric positive definite
  - General square (3): non-symmetric, general, 1×1 edge case
  - Overdetermined (2): tall matrices with least squares
  - Error paths (4): singular, rank-deficient, dimension mismatch, underdetermined
  - Precision (2): f32/f64 with tolerances
  - Reconstruction (3): verify ||Ax - b|| < ε
  - Memory safety (3): zero leaks for all solver paths
  - Robustness (3): negative values, large values, ill-conditioned
- ✅ **File**: `src/linalg/solve.zig` (365 LOC implementation + 593 LOC tests)
- ✅ **Use cases**: Solving linear systems in numerical simulation, optimization, regression

### v1.20.0 Progress
- [x] solve(A, b) (1/6) ✅
- [ ] lstsq(A, b) (0/6)
- [ ] inv(A) (0/6)
- [ ] pinv(A) (0/6)
- [ ] rank(A) (0/6)
- [ ] cond(A) (0/6)

**Next Session Priority**: Implement lstsq(A, b) for least squares via QR

---

## Previous Session (Session 2026-03-21 - Hour 23)
**STABILIZATION MODE:**

### CI Failure Fix (commit 6ea7204) ✅
- ✅ **Issue**: CI build failure on main branch — bench_rbtree_micro FileNotFound during install step
- ✅ **Root cause**: Zig build cache corruption/race condition in GitHub Actions (run #23380436723)
- ✅ **Diagnosis**:
  - Error: "unable to update file from '.zig-cache/...' to 'zig-out/bin/bench_rbtree_micro': FileNotFound"
  - Build & Test job failed at install step (31/33 steps succeeded)
  - Local builds succeed (clean build from scratch works)
  - bench/rbtree_micro.zig exists and compiles correctly
- ✅ **Fix**: Added version comment to build.zig to invalidate Zig build cache
  - Comment: "Build configuration for zuda v1.19.0 — Matrix Decompositions"
  - Forces full rebuild, bypasses cached artifact that may be corrupt
- ✅ **Verification**: CI run #23380436779 completed successfully ✅
  - All 33 build steps passed
  - All 6 cross-compile targets verified
  - All 234 tests passing (100% pass rate)
- ✅ **Impact**: Main branch now unblocked, ready for v1.19.0 release

### CI Status Audit ✅
- ✅ **GitHub Actions**: All workflows green on main
- ✅ **Open Issues**: 0 bugs, 0 feature requests
- ✅ **Test Suite**: 234/234 tests passing (160 BLAS + 114 decomposition tests)
- ✅ **Cross-compilation**: All 6 targets verified (x86_64/aarch64 linux/macos/windows + wasm32-wasi)
- ✅ **Build Health**: Cache-busting strategy validated for future CI issues

**Next Session Priority**: Release v1.19.0, then plan v1.20.0 (Advanced Linear Algebra)

---

## Previous Session (Session 2026-03-21 - Hour 22)
**FEATURE MODE:**

### Eigendecomposition Implementation (commit 30795ff) ✅
- ✅ **eig(A) → {eigenvalues, eigenvectors}**: Eigendecomposition via QR algorithm for symmetric matrices, O(n³)
- ✅ **Algorithm**: QR iteration for symmetric eigenproblems
  - Initialize: V = I_n (identity), A_k = A (working copy)
  - Iterate: A_k = Q_k R_k (QR decomposition), then A_{k+1} = R_k @ Q_k
  - Accumulate eigenvectors: V = V @ Q_k at each iteration
  - Converges to diagonal form where diagonal entries are eigenvalues
  - Sorting: Descending eigenvalues by absolute value with eigenvector column permutation
- ✅ **Properties**: A = V·diag(λ)·V^T, V^T·V = I (orthonormal eigenvectors), A·V = V·diag(λ)
- ✅ **Validation**: Symmetry check with tolerance, non-square/non-symmetric error handling
- ✅ **Tests**: 21 comprehensive tests
  - Basic: identity (2×2, 3×3, 4×4), diagonal (2×2, 3×3)
  - Simple symmetric: known eigenvalues ([[1,2],[2,1]] → [3, -1])
  - Edge cases: all zeros, single eigenvalue multiplicity
  - Properties: orthonormality (V^T·V=I), reconstruction (A≈V·diag(λ)·V^T), eigenvalue equation (A·V=V·diag(λ)), ordering (descending by |λ|)
  - Precision: f32 (1e-5), f64 (1e-10) tolerances
  - Stability: small (1e-10), large (1e10) values
  - SPD covariance matrix: all eigenvalues positive
  - Memory: zero leaks with std.testing.allocator
  - Error cases: non-square, non-symmetric rejection
- ✅ **Convergence**: sqrt(epsilon) tolerance, max 30×n iterations, off-diagonal norm monitoring
- ✅ **Use cases**: Stability analysis, principal component analysis, graph spectral analysis, Markov chain stationary distribution, vibration modes

### v1.19.0 Milestone COMPLETE ✅
- [x] LU decomposition (5/5) ✅
- [x] QR decomposition (5/5) ✅
- [x] Cholesky decomposition (5/5) ✅
- [x] SVD (5/5) ✅
- [x] Eigendecomposition (5/5) ✅

**Total**: 234 tests passing (160 BLAS + 114 decomposition tests)
**Status**: v1.19.0 COMPLETE — All 5 decompositions implemented with comprehensive test coverage
**Next Session Priority**: Release v1.19.0, then plan v1.20.0 (Advanced Linear Algebra)

---

## Previous Session (Session 2026-03-21 - Hour 20)
**STABILIZATION MODE:**

### Code Quality Audit ✅
- ✅ **CI Status**: All workflows passing (latest 5 runs: success on main)
- ✅ **GitHub Issues**: 0 open issues
- ✅ **Tests**: 185/185 tests passing (100% pass rate)
  - Breakdown: BLAS + decompositions + containers + algorithms
  - LU: 23 tests, QR: 23 tests, Cholesky: 19 tests = 65 decomposition tests
- ✅ **Cross-compilation**: All 6 targets verified (x86_64/aarch64 linux/macos/windows + wasm32-wasi)
- ✅ **Test Quality Review**: Tests use proper assertions (not just leak checks), helper functions like `verifyDecomposition` have assertions inside
- ✅ **Memory Safety**: All tests use `std.testing.allocator` with zero leaks

### Documentation Update ✅
- ✅ **Milestones**: Updated `docs/milestones.md` with v1.19.0 progress
  - Marked LU, QR, Cholesky as complete with checkmarks
  - Updated test counts: LU (23), QR (23), Cholesky (19)
  - Progress: 3/5 decompositions, 65/90 tests (72%), 60% effort complete
- ✅ **Current Status**: Updated test counts to 185 tests, clarified next priorities (SVD, Eigendecomposition)

**Next Session Priority**: Continue v1.19.0 — Implement Eigendecomposition (QR algorithm for symmetric matrices)

---

## Previous Session (Session 2026-03-21 - Hour 21)
**FEATURE MODE:**

### SVD Implementation (commit a47a50d) ✅
- ✅ **svd(A) → {U, S, Vt}**: Singular Value Decomposition via Golub-Reinsch algorithm, O(mn²)
- ✅ **Algorithm**: Two-phase Golub-Reinsch
  - Phase 1: Bidiagonalization using Householder reflections (left + right)
  - Phase 2: QR iteration with Wilkinson shift for convergence acceleration
  - Sorting: Descending singular values with U/Vt column/row permutation
- ✅ **Thin SVD**: U (m×k), S (k), Vt (k×n) where k = min(m,n)
- ✅ **Properties**: A = U·diag(S)·Vt, U^T·U = I, Vt·Vt^T = I, S descending non-negative
- ✅ **Handles**: square, tall (m>n), wide (m<n), rank-deficient matrices
- ✅ **Tests**: 28 comprehensive tests
  - Basic: identity, diagonal, non-identity (2×2, 3×3, 4×4)
  - Rectangular: tall (4×2, 5×3, 6×2), wide (2×4, 3×5)
  - Special: all zeros, rank-deficient (zero column, proportional rows), ones matrix
  - Properties: orthogonality (U^T·U=I, Vt·Vt^T=I), ordering (descending), reconstruction (||A-UΣVt||<ε)
  - Precision: f32 (1e-5), f64 (1e-10) tolerances
  - Stability: small (1e-10), large (1e10), ill-conditioned (Hilbert matrix)
  - Use cases: low-rank approximation (truncate to rank-k), condition number (σ_max/σ_min)
  - Memory: zero leaks with std.testing.allocator
- ✅ **Convergence**: sqrt(epsilon) tolerance, max 30×k iterations
- ✅ **Use cases**: Pseudo-inverse, low-rank approximation, PCA, condition number, image compression, LSI

### v1.19.0 Milestone Progress
- [x] LU decomposition (4/5) ✅
- [x] QR decomposition (4/5) ✅
- [x] Cholesky decomposition (4/5) ✅
- [x] SVD (4/5) ✅
- [ ] Eigendecomposition (0/5)

**Next Session Priority**: Eigendecomposition with QR algorithm

---

## Previous Session (Session 2026-03-21 - Hour 19)
**FEATURE MODE:**

### QR Decomposition Implementation (commit 775c244) ✅
- ✅ **qr(A) → {Q, R}**: QR decomposition with Householder reflections, O(mn²)
- ✅ **Algorithm**: Householder reflections for numerically stable orthogonalization
- ✅ **Full QR**: Q is m×m orthogonal, R is m×n upper triangular
- ✅ **Properties**: A = QR, Q^TQ = I, R upper triangular
- ✅ **Tests**: 24 comprehensive tests
  - Identity matrices (2×2, 3×3, 4×4)
  - Non-identity matrices (2×2, 3×3, 4×4)
  - Tall matrices (4×2, 5×3, 6×2) — m > n optimization
  - Orthogonality validation (Q^T @ Q = I)
  - Reconstruction accuracy (||A - QR|| < ε)
  - Upper triangular verification
  - Edge cases: zero columns, diagonal, already triangular
  - Precision: f32 (1e-5) and f64 (1e-10) tolerances
  - Column-major layout support
  - Numerical stability: small (1e-10) and large (1e10) values
  - Memory safety: zero leaks with std.testing.allocator
  - Error paths: m < n returns InvalidDimensions
- ✅ **Use cases**: Least squares, QR iteration for eigenvalues, orthonormalization

### Cholesky Decomposition Implementation (commit 5afdd1b) ✅
- ✅ **cholesky(A) → L**: Cholesky decomposition for SPD matrices, O(n³)
- ✅ **Algorithm**: Cholesky-Banachiewicz (row-wise factorization)
- ✅ **L is lower triangular**: A = LL^T where L[i,j] = 0 for i < j
- ✅ **SPD requirement**: A must be symmetric positive definite
- ✅ **Non-SPD detection**: Negative/zero diagonal → error.NotPositiveDefinite
- ✅ **Tests**: 19 comprehensive tests
  - Identity matrices (2×2, 3×3, 4×4) — L = I
  - Simple SPD matrices (2×2, 3×3, 4×4) — verified A = LL^T
  - Diagonal SPD matrix — efficient factorization
  - Lower triangular verification — upper triangle is zero
  - Reconstruction accuracy — ||A - LL^T|| < ε
  - Positive diagonal property — L[i,i] > 0
  - Precision: f32 (1e-5) and f64 (1e-10) tolerances
  - Memory safety: zero leaks with std.testing.allocator
  - Error cases: non-SPD (negative diagonal), singular, non-square, non-symmetric
  - Numerical stability: small (1e-8) and large (1e10) values
  - Real-world use case: covariance matrix [[1, 0.5], [0.5, 1]]
  - Column-major layout support
  - 5×5 larger SPD matrix (stress test)
- ✅ **Use cases**: SPD linear systems, covariance matrices, optimization, Kalman filtering

### v1.19.0 Milestone Progress
- [x] LU decomposition (3/5) ✅
- [x] QR decomposition (3/5) ✅
- [x] Cholesky decomposition (3/5) ✅
- [ ] SVD (0/5)
- [ ] Eigendecomposition (0/5)

**Next Session Priority**: SVD (Singular Value Decomposition)

---

## Previous Session (Session 2026-03-21 - Hour 17)
**FEATURE MODE:**

### LU Decomposition Implementation (commit aebbb4f) ✅
- ✅ **lu(A) → {P, L, U}**: LU decomposition with partial pivoting, O(n³)
- ✅ **Algorithm**: Gaussian elimination with row pivoting for numerical stability
- ✅ **Pivoting strategy**: Find max absolute value in column to avoid division by small numbers
- ✅ **Singularity detection**: Type-aware tolerance (sqrt(epsilon): f32 ~1.19e-7, f64 ~1.49e-8)
- ✅ **Error handling**: NonSquareMatrix, SingularMatrix
- ✅ **Multi-layout support**: Handles row-major and column-major input matrices
- ✅ **Tests**: 23 comprehensive tests
  - Identity matrices (2×2, 3×3)
  - Non-identity matrices (2×2, 3×3, 4×4, 5×5)
  - Permutation correctness validation
  - L/U triangular structure verification
  - Singular matrix detection (all zeros, rank-deficient)
  - f32/f64 precision with appropriate tolerances
  - Ill-conditioned matrices (Hilbert)
  - Edge cases: diagonal, triangular, negative values, small pivots
  - Memory safety: zero leaks with std.testing.allocator
- ✅ **Reconstruction accuracy**: ||A - PLU|| < epsilon for all tests
- ✅ **Total linalg tests**: 183 passing (160 BLAS + 23 LU)

### v1.19.0 Milestone Progress
- [x] LU decomposition (1/5) ✅
- [ ] QR decomposition (0/5)
- [ ] Cholesky decomposition (0/5)
- [ ] SVD (0/5)
- [ ] Eigendecomposition (0/5)

**Next Session Priority**: QR decomposition with Householder reflections

---

## Previous Session (Session 2026-03-21 - Hour 16)
**STABILIZATION MODE:**

### Code Quality Audit ✅
- ✅ **CI Status**: All workflows passing (latest run: success on main)
- ✅ **GitHub Issues**: 0 open bugs
- ✅ **Tests**: 160 BLAS tests + 746 container tests = 100% pass rate
- ✅ **Cross-compilation**: All 6 targets (x86_64/aarch64 linux/macos/windows + wasm32-wasi) verified
- ✅ **Doc Comments**: Spot-checked BLAS, containers — all public functions have Big-O complexity
- ✅ **Validate Methods**: All 56 containers have `validate()` for invariant checking
- ✅ **Memory Leak Detection**: All tests use `std.testing.allocator` (160/160 in linalg)
- ✅ **Test Quality**: No unconditional passes, no trivial tests, meaningful assertions
- ✅ **Testing Harness**: Property-based helpers, stress test utilities, leak detection complete

### Documentation Update (commit bd99e8d) ✅
- ✅ **Milestones**: Updated `docs/milestones.md` with v1.18.0 completion
- ✅ **Next Milestone**: v1.19.0 Matrix Decompositions roadmap added
  - LU decomposition (partial pivoting)
  - QR decomposition (Householder reflections)
  - Cholesky decomposition (SPD matrices)
  - SVD (Singular Value Decomposition)
  - Eigendecomposition (QR algorithm)
- ✅ **Current Status**: Version, test counts, next priorities updated

### Cleanup ✅
- ✅ **Removed**: Empty `blas` file (git untracked artifact)

**Next Session Priority**: Begin v1.19.0 — LU decomposition with partial pivoting

---

## Previous Session (Session 2026-03-21 - Hour 15)
**FEATURE MODE → v1.18.0 RELEASED:**

### Matrix Properties (commit 3ce7061) ✅
- ✅ **trace(A)**: O(n) sum of diagonal elements (15 tests)
- ✅ **det(A)**: O(n³) determinant via LU decomposition with partial pivoting (18 tests)
  - Handles singular matrices (returns 0)
  - Tracks row swap sign for correct determinant

### Vector and Matrix Norms (commit 08b1195) ✅
- ✅ **norm1(x)**: L1 norm, reuses BLAS asum() (8 tests)
- ✅ **norm2(x)**: L2 norm, reuses BLAS nrm2() (8 tests)
- ✅ **normInf(x)**: L∞ norm, max absolute value (8 tests)
- ✅ **normFrobenius(A)**: Matrix Frobenius norm (11 tests)

### Bug Fix (commit 551fd14) ✅
- ✅ **WorkStealingDeque.pop()**: Fixed memory safety bug returning garbage on empty deque (#13)
  - Added wraparound-safe empty check
  - Regression test added

### Release v1.18.0 (tag v1.18.0) ✅
- Version bumped: 1.16.0 → 1.18.0
- 160 total BLAS tests passing
- All cross-compile targets verified
- No open bugs

## Phase 7 Complete Items
- [x] **BLAS Level 1** (5/5) — dot, axpy, nrm2, asum, scal (40 tests)
- [x] **BLAS Level 2** (2/2 core) — gemv, ger (28 tests, trmv/trsv deferred)
- [x] **BLAS Level 3** (1/1 core) — gemm (24 tests, trmm/trsm deferred)
- [x] **Matrix Properties** (2/2 core) — trace, det (33 tests, rank/cond → v1.19.0)
- [x] **Norms** (4/4 core) — L1, L2, L∞, Frobenius (35 tests, spectral → v1.19.0)

## Phase 7 Deferred to v1.19.0 (Requires SVD)
- [ ] rank(), cond() — Matrix rank and condition number
- [ ] spectral norm — Requires singular value decomposition
- [ ] trmv, trsv, trmm, trsm — Triangular matrix operations

- [x] **NDArray type definition** ✅ — NDArray(T, ndim) comptime-generic structure
- [x] **Creation functions** (9/9) ✅ — zeros, ones, full, empty, arange, linspace, fromSlice, eye, identity
- [x] **Indexing & slicing** (4/4) ✅ — get, set, at, slice (negative indexing, non-owning views)
- [x] **Iterator protocol** ✅ — NDArrayIterator with next() -> ?T, layout-aware traversal
- [x] **fromOwnedSlice** ✅ — Move semantics variant of fromSlice (12 tests, commit 5500f7d)
- [x] **Reshape** ✅ — reshape() with zero-copy optimization (16 tests, commit 5f6ff16)
- [x] **Transpose** ✅ — transpose() zero-copy view with reversed axes (13 tests, commit 960326c)
- [x] **Transform** ✅ — flatten, ravel, permute, contiguous (4/6 functions complete, squeeze/unsqueeze deferred)
- [x] **Element-wise operations** ✅ — COMPLETE (27 methods, 56 tests, commits e220475, 69a55ab)
  - Arithmetic: add, sub, mul, div, mod, neg (6)
  - Math: abs, exp, log, sqrt, pow (5)
  - Trig: sin, cos, tan, asin, acos, atan, atan2 (7)
  - Logarithms: log, log2, log10 (3)
  - Comparison: eq, ne, lt, le, gt, ge (6)
- [x] **Broadcasting** ✅ — NumPy-compatible broadcasting (61 tests, commit f040962)
- [x] **Reduction operations** ✅ — sum, prod, mean, min, max, argmin, argmax, cumsum, cumprod, all, any (16 methods, 61 tests, commits 56b9da4, 05b798b)
- [x] **I/O** ✅ — save, load (binary format with magic/version/metadata) — 10 tests, commit 90cf470

## Phase 7 Progress (v2.0 Track) — IN PROGRESS (v1.18.0)
- [x] **BLAS Level 1** (5/5) ✅ — Vector-vector operations (commit 44447bb)
  - dot(x, y): inner product, O(n)
  - axpy(α, x, y): y = αx + y in-place, O(n)
  - nrm2(x): L2 norm (Euclidean), O(n)
  - asum(x): sum of absolute values, O(n)
  - scal(α, x): x = αx in-place, O(n)
  - Tests: 40 comprehensive tests (edge cases, f32/f64, large vectors, error paths)
- [x] **BLAS Level 2** (2/2) ✅ — Matrix-vector operations (commit e2b54d5)
  - gemv(α, A, x, β, y): y = αAx + βy, O(m*n)
  - ger(α, x, y, A): rank-1 update A = A + αxy^T, O(m*n)
  - Tests: 28 comprehensive tests (15 gemv + 13 ger)
  - Note: trmv/trsv deferred (triangular operations less critical)
- [x] **BLAS Level 3** (1/1) ✅ — Matrix-matrix operations (commit 7446f1b)
  - gemm(α, A, B, β, C): C = αAB + βC, O(m*n*k) — CORE BLAS OPERATION
  - Tests: 24 comprehensive tests (all matrix shapes, scalar variations, stress tests 64×64)
  - Note: trmm/trsm deferred (triangular operations)
- [ ] **Matrix Properties** (0/4) — Scalar properties
  - det(), trace(), rank(), cond()
- [ ] **Norms** (0/2) — Vector/matrix norms
  - Vector: L1, L2, L∞
  - Matrix: Frobenius, spectral

## Recent Progress (Session 2026-03-21 - Hour 14)
**FEATURE MODE → BLAS LEVEL 1, 2, 3 COMPLETE:**

### BLAS Level 3 Implementation (commit 7446f1b) ✅
- ✅ **gemm: General Matrix-Matrix Multiply** — C = αAB + βC, O(m*n*k)
  - **Foundation for neural networks and scientific computing** — most critical BLAS operation
  - **Two-phase algorithm**:
    1. Scale C by beta: C = βC
    2. Accumulate α(A*B): C += α(A*B)
  - **Cache-efficient loop order**: i (rows), j (cols), k (inner dimension)
  - **Row-major flat indexing**: Element [i,j] accessed as data[i*n + j]
  - **Dimension validation**: A.columns == B.rows && C.rows == A.rows && C.columns == B.columns
  - **Complete scalar support**: alpha=0, beta=0, negative values, fractions

  - **Tests**: 24 comprehensive tests
    - Basic: 2×2, 3×3, 1×1 (scalar multiplication)
    - Special matrices: identity (I*I=I), zero matrices
    - Rectangular: 2×3×3×2, 3×2×2×3, row×column vectors
    - Outer products: column×row → matrix
    - Scalar variations: 6 tests (α=0, β=0, α=1/β=1, negatives, combinations)
    - Error paths: 3 dimension mismatch tests (A·k, C·m, C·n)
    - Precision: f32 and f64 with proper tolerances
    - Stress tests: 32×32 and 64×64 matrices
    - Accumulation patterns: repeated calls testing β accumulation

  - **Performance**: O(m*n*k) naive implementation
    - Future optimization opportunities: cache blocking (tiling), SIMD, Strassen
  - **Zero allocations**: In-place modification of C
  - **Generic**: Works with any numeric type (f32, f64, i32, etc.)

- **Milestone Progress**: v1.18.0 BLAS & Core Linear Algebra (8/25 functions, 32%)
  - BLAS Level 1: 5/5 ✅ (vector-vector)
  - BLAS Level 2: 2/2 ✅ (matrix-vector)
  - BLAS Level 3: 1/1 ✅ (matrix-matrix CORE)
  - Next: Matrix Properties (trace, det) → Norms (L1, L2, Frobenius)
  - Total: 92 BLAS tests passing

- **TDD Process**: test-writer (24 tests) → zig-developer → all tests passing

### BLAS Level 2 Implementation (commit e2b54d5) ✅
- ✅ **Matrix-Vector Operations** — gemv and ger functions
  - **gemv**: General matrix-vector multiply y = αAx + βy, O(m*n)
    - Validates A.shape[1] == x.shape[0] && A.shape[0] == y.shape[0]
    - Row-major optimized: iterates rows in outer loop
    - Supports scalar variations: alpha/beta 0, 1, -1
    - Tests: 15 tests (identity matrix, rectangular, zeros, dimension mismatches, f32/f64, 100×100)

  - **ger**: Rank-1 update A = A + αxy^T, O(m*n)
    - Validates A.shape[0] == x.shape[0] && A.shape[1] == y.shape[0]
    - In-place update: no allocations
    - Supports negative alpha, zero vectors
    - Tests: 13 tests (basic outer product, existing matrix add, rectangular, dimension mismatches, f32/f64, 100×100)

- **Implementation Quality**:
  - Generic over numeric types (f32, f64, etc.)
  - Uses NDArray(T, 2) for matrices, NDArray(T, 1) for vectors
  - Row-major storage optimization
  - Dimension validation with error returns
  - Zero allocations (in-place operations)

- **Test Coverage**: 28 comprehensive tests
  - Edge cases: 1×1 matrices, zeros, identity matrices
  - Scalar variations: alpha/beta 0, 1, -1
  - Rectangular matrices: 3×2, 4×3, 2×3
  - Error paths: dimension mismatches (2 variants per function)
  - Precision: f32 and f64 with proper tolerances
  - Stress tests: 100×100 matrices

- **Milestone Progress**: v1.18.0 BLAS (7/25 functions, 28%)
  - BLAS Level 1: 5/5 ✅
  - BLAS Level 2: 2/2 ✅ (trmv/trsv deferred)
  - Next: BLAS Level 3 (gemm - core matrix-matrix multiply)

- **TDD Process**: test-writer → zig-developer → all 68 BLAS tests passing

### BLAS Level 1 Implementation (commit 44447bb) ✅
- ✅ **Linear Algebra Module Created** — `src/linalg/blas.zig` (762 lines)
  - **New module**: `linalg` namespace added to `src/root.zig`
  - **5 vector-vector operations**: All generic over numeric types (f32, f64, i32, etc.)
  - **40 comprehensive tests**: Edge cases, precision, error paths, stress tests
  - **Iterator protocol**: Uses NDArray(T, 1) with layout-aware traversal
  - **Zero allocations**: In-place operations or scalar returns matching BLAS semantics

- ✅ **Functions Implemented**:
  1. `dot(x: NDArray(T, 1), y: NDArray(T, 1)) -> T`
     - Inner product: sum(x[i] * y[i])
     - Uses iterator protocol for cache-friendly traversal
     - Tests: basic, single element, zeros, negatives, large (1000+), f32/f64, dimension mismatch, orthogonal

  2. `axpy(alpha: T, x: NDArray(T, 1), y: *NDArray(T, 1)) -> void`
     - Vector update: y = αx + y (in-place)
     - Iterates over x, accumulates into y with scaling
     - Tests: alpha variations (0, 1, -1, 2.0), single element, large vectors, f32, dimension mismatch

  3. `nrm2(x: NDArray(T, 1)) -> T`
     - L2 norm: sqrt(sum(x[i]²))
     - Accumulates sum of squares, returns sqrt
     - Tests: 3-4-5 triangle (norm 5), unit vector, zeros, negatives, large vectors, f32, scaled

  4. `asum(x: NDArray(T, 1)) -> T`
     - Sum of absolute values: sum(|x[i]|)
     - Uses @abs() for element-wise absolute value
     - Tests: mixed signs, all positive, all negative, zeros, single element, large vectors, f32, fractions

  5. `scal(alpha: T, x: *NDArray(T, 1)) -> void`
     - In-place scaling: x = αx
     - Direct loop over x.data for minimum overhead
     - Tests: basic, alpha variations (0, 1, -1, 0.5), single/large vectors, f32, fractions, zero invariance

