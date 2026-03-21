# zuda — Milestones

## Current Status

- **Latest release**: v1.19.1 (2026-03-22) — CI Stability Fixes
- **Current phase**: v2.0 Track (Phase 7) — Scientific Computing Platform
- **Tests**: 234 tests passing (100%) — 160 BLAS + 114 decomposition tests
- **Open issues**: None
- **Blockers**: None
- **v2.0 Target**: Scientific computing platform (NDArray, linear algebra, stats, FFT, numerical methods, optimization)
- **Next Milestone**: v1.20.0 — Advanced Linear Algebra (solvers, pseudo-inverse, matrix properties)

---

## Active Milestones

### v1.20.0 — Advanced Linear Algebra ✅ COMPLETE

Implement linear system solvers, pseudo-inverse, and advanced matrix properties leveraging existing decompositions.

**Context**: v1.19.0 completed 5 core decompositions (LU, QR, Cholesky, SVD, Eigendecomposition). These provide the foundation for solving linear systems (Ax=b), computing pseudo-inverses, and extracting matrix properties like rank and condition number.

**Target**: Implement 6 solver/property functions that complete Phase 7 linear algebra requirements

**Categories**:
- [x] **solve(A, b)** — Solve linear system Ax=b using appropriate decomposition ✅
  - [x] Auto-select: Cholesky for SPD, LU for general, QR for overdetermined
  - [x] Returns solution vector x, O(n³) via decomposition + back-substitution
  - [x] Error handling: singular, under/overdetermined systems
  - [x] Tests: 24 comprehensive tests (SPD, general, rectangular, singular detection, f32/f64)
  - **Use case**: Solving linear systems in numerical simulation, optimization, regression
  - **File**: `src/linalg/solve.zig`
- [x] **lstsq(A, b)** — Least squares solution for overdetermined systems ✅
  - [x] Uses QR decomposition for numerical stability
  - [x] Returns x minimizing ||Ax - b||₂, O(mn²)
  - [x] Tests: 16 comprehensive tests (tall matrices, rank-deficient, multiple RHS, f32/f64)
  - **Use case**: Linear regression, curve fitting, overdetermined systems
  - **File**: `src/linalg/solve.zig`
- [x] **inv(A)** — Matrix inverse via LU decomposition ✅
  - [x] A⁻¹ computed by solving AX = I column-by-column, O(n³)
  - [x] Singular matrix detection
  - [x] Tests: 25 comprehensive tests (2×2, 3×3, diagonal, singular detection, ill-conditioned, f32/f64)
  - **Use case**: Control theory, statistics (covariance inverse), analytical solutions
  - **File**: `src/linalg/solve.zig`
- [x] **pinv(A)** — Moore-Penrose pseudo-inverse via SVD ✅
  - [x] A⁺ = VΣ⁺U^T where Σ⁺[i,i] = 1/σᵢ if σᵢ > tol, else 0
  - [x] Works for any matrix shape (square, tall, wide, rank-deficient)
  - [x] Tolerance: max(m,n) × σ_max × machine_epsilon
  - [x] Tests: 26 comprehensive tests (full rank, rank-deficient, rectangular, zero rows/cols, f32/f64)
  - **Use case**: Solving under/overdetermined systems, generalized inverse, least-norm solutions
  - **File**: `src/linalg/solve.zig`
- [x] **rank(A)** — Matrix rank via SVD ✅
  - [x] Count singular values > tolerance
  - [x] Tolerance: max(m,n) × σ_max × machine_epsilon
  - [x] Tests: 16 comprehensive tests (full rank, rank-deficient, zero matrix, identity, f32/f64)
  - **Use case**: System of equations analysis, dimensionality reduction, collinearity detection
  - **File**: `src/linalg/properties.zig`
- [x] **cond(A)** — Condition number via SVD ✅
  - [x] κ(A) = σ_max / σ_min (ratio of largest to smallest singular value)
  - [x] Returns +inf for singular matrices (σ_min = 0)
  - [x] Tests: 16 comprehensive tests (well-conditioned, ill-conditioned, Hilbert matrix, singular, f32/f64)
  - **Use case**: Numerical stability analysis, error propagation estimation, solver selection
  - **File**: `src/linalg/properties.zig`

**Success Criteria**: ✅ All 6 functions complete with 123 tests total, solver auto-selection working, numerical stability validated

**Test Breakdown**: 24 solve + 16 lstsq + 25 inv + 26 pinv + 16 rank + 16 cond = 123 tests

**Total Test Count**: 301 tests (160 BLAS + 114 decompositions + 123 solvers/properties)

**Status**: ✅ COMPLETE — Released as v1.20.0 on 2026-03-22

### v1.19.0 — Matrix Decompositions ✅ COMPLETE

Implement core matrix decomposition algorithms for solving linear systems and eigenvalue problems:

**Context**: v1.18.0 completed BLAS Level 1/2/3 and core operations (trace, det, norms). Decompositions are foundation for solving Ax=b, least squares, eigenvalue problems, and numerical stability analysis.

**Target**: Implement 5 core decomposition algorithms with numerical stability guarantees

**Categories**:
- [x] **LU Decomposition** — A = PLU with partial pivoting ✅
  - [x] `lu(A) -> {P, L, U}` — O(n³) factorization
  - [x] Partial pivoting for numerical stability
  - [x] Singular matrix detection
  - [x] Tests: 23 tests (identity, singular, rectangular, ill-conditioned, f32/f64)
  - **Use case**: Solving Ax=b, determinant computation, matrix inversion
  - **File**: `src/linalg/lu.zig`
- [x] **QR Decomposition** — A = QR with Householder reflections ✅
  - [x] `qr(A) -> {Q, R}` — O(mn²) factorization
  - [x] Householder reflections (numerically stable)
  - [x] Thin QR (m > n optimization)
  - [x] Tests: 23 tests (square, tall, orthogonality validation, f32/f64)
  - **Use case**: Least squares, eigenvalue algorithms, orthonormalization
  - **File**: `src/linalg/decompositions.zig`
- [x] **Cholesky Decomposition** — A = LL^T for symmetric positive definite ✅
  - [x] `cholesky(A) -> L` — O(n³) factorization
  - [x] Symmetry and positive-definiteness validation
  - [x] Numerical stability checks
  - [x] Tests: 19 tests (2×2, 3×3, non-SPD detection, f32/f64)
  - **Use case**: Covariance matrices, optimization, linear system solving
  - **File**: `src/linalg/decompositions.zig`
- [x] **SVD (Singular Value Decomposition)** — A = UΣV^T ✅
  - [x] `svd(A) -> {U, Sigma, Vt}` — O(mn²) for thin SVD
  - [x] Golub-Reinsch algorithm (bidiagonalization + QR iteration with Wilkinson shift)
  - [x] Singular value ordering (descending, non-negative)
  - [x] Tests: 28 tests (square, tall, wide, rank-deficient, ill-conditioned, f32/f64)
  - **Use case**: Pseudo-inverse, low-rank approximation, PCA, condition number, image compression
  - **File**: `src/linalg/decompositions.zig`
- [x] **Eigendecomposition** — A = VΛV⁻¹ for symmetric matrices ✅
  - [x] `eig(A) -> {eigenvalues, eigenvectors}` — O(n³) QR algorithm
  - [x] QR iteration for symmetric matrices (converges to diagonal form)
  - [x] Eigenvector accumulation during QR iterations
  - [x] Eigenvalue sorting (descending by absolute value)
  - [x] Tests: 21 tests (identity, diagonal, symmetric, edge cases, orthonormality, reconstruction, ordering, precision, stability, SPD, error cases)
  - **Use case**: Stability analysis, PCA, graph spectral analysis, Markov chains
  - **File**: `src/linalg/decompositions.zig`

**Success Criteria**: ✅ **ACHIEVED** — All 5 decompositions complete with 114 tests, numerical stability validated

**Progress**: 5/5 decompositions complete (LU: 23 tests, QR: 23 tests, Cholesky: 19 tests, SVD: 28 tests, Eigendecomposition: 21 tests)

**Status**: ✅ **COMPLETE** (2026-03-21) — Ready for v1.19.0 release

### v1.15.0 — Iterator Adaptor Expansion (DEFERRED)

Extend the iterator system with advanced adaptors for composable data transformation pipelines:

**Context**: v1.14.0 deferred iterator adaptor expansion. v1.3.0 established the foundation (Map, Filter, Chain, Zip, Take, Skip, Enumerate, collect). Now expand with advanced adaptors for real-world data processing patterns.

**Target**: Implement 4+ advanced iterator adaptors that enable zero-allocation, composable data transformation pipelines

**Categories**:
- [ ] **FlatMap Adaptor** — Map then flatten nested iterables
  - [ ] `FlatMap(T, U, InnerIter, mapFn)` — Transform and flatten in single pass
  - [ ] Lazy evaluation (no intermediate allocation)
  - [ ] Tests: 15+ tests (nested arrays, optional unwrapping, error propagation, chaining)
  - **Use case**: `items.flatMap(parseLines).collect()` — parse file lines in single pass
- [ ] **TakeWhile Adaptor** — Take until predicate fails
  - [ ] `TakeWhile(T, BaseIter, predicateFn)` — Stop on first false predicate
  - [ ] Preserves base iterator state (can resume after TakeWhile ends)
  - [ ] Tests: 15+ tests (empty, all match, none match, partial match, chaining)
  - **Use case**: `stream.takeWhile(lessThan100).sum()` — sum until threshold
- [ ] **SkipWhile Adaptor** — Skip until predicate fails
  - [ ] `SkipWhile(T, BaseIter, predicateFn)` — Drop elements until predicate false, then yield rest
  - [ ] Complements TakeWhile (mirror semantics)
  - [ ] Tests: 15+ tests (empty, all skip, none skip, partial skip, chaining with TakeWhile)
  - **Use case**: `log.skipWhile(isOld).collect()` — skip old entries, keep recent
- [ ] **Partition Adaptor** — Split into two iterators based on predicate
  - [ ] `Partition(T, BaseIter, predicateFn)` → `{true: TrueIter, false: FalseIter}`
  - [ ] Lazy buffering (minimize memory allocation)
  - [ ] Tests: 15+ tests (empty, all true, all false, mixed, chaining both sides)
  - **Use case**: `{even, odd} = nums.partition(isEven)` — split evens and odds
- [ ] **Documentation & Examples** — Update iterator guide with new adaptors
  - [ ] Add FlatMap/TakeWhile/SkipWhile/Partition sections to docs/GUIDE.md
  - [ ] Real-world examples: log processing, data ETL, sensor filtering
  - [ ] Benchmark zero-cost abstraction vs manual loops

**Success Criteria**: All 4 adaptors complete with 60+ tests total, docs updated, zero-cost abstraction verified

**Estimated Effort**: 1-2 sessions (extends v1.3.0 pattern, low-medium complexity)

### v1.18.0 — BLAS & Core Linear Algebra ✅ COMPLETE

Implement BLAS operations and core matrix properties for scientific computing:

**Context**: v1.17.0 completed NDArray operations (element-wise, reductions, broadcasting, I/O). BLAS provides foundation for linear algebra, numerical solvers, and machine learning.

**Target**: Implement BLAS Level 1/2/3 + core matrix properties (trace, det, norms)

**Categories**:
- [x] **BLAS Level 1** (5/5) — Vector-vector operations ✅
  - ✅ dot(x, y): inner product, O(n) — 8 tests
  - ✅ axpy(α, x, y): y = αx + y, O(n) — 8 tests
  - ✅ nrm2(x): L2 norm, O(n) — 8 tests
  - ✅ asum(x): sum of absolute values, O(n) — 8 tests
  - ✅ scal(α, x): x = αx, O(n) — 8 tests
- [x] **BLAS Level 2** (2/2) — Matrix-vector operations ✅
  - ✅ gemv(α, A, x, β, y): y = αAx + βy, O(m*n) — 15 tests
  - ✅ ger(α, x, y, A): rank-1 update A = A + αxy^T, O(m*n) — 13 tests
  - ⏭️ trmv, trsv deferred (triangular operations less critical)
- [x] **BLAS Level 3** (1/1) — Matrix-matrix operations ✅
  - ✅ gemm(α, A, B, β, C): C = αAB + βC, O(m*n*k) — 24 tests
  - ⏭️ trmm, trsm deferred (triangular operations)
- [x] **Matrix Properties** (2/4) ✅
  - ✅ trace(A): sum of diagonal, O(n) — 15 tests
  - ✅ det(A): determinant via LU, O(n³) — 18 tests
  - ⏭️ rank(), cond() deferred to v1.19.0 (requires SVD)
- [x] **Norms** (4/6) ✅
  - ✅ norm1(x): L1 norm, O(n) — 8 tests
  - ✅ norm2(x): L2 norm, O(n) — 8 tests
  - ✅ normInf(x): L∞ norm, O(n) — 8 tests
  - ✅ normFrobenius(A): matrix Frobenius norm, O(m*n) — 11 tests
  - ⏭️ spectral norm deferred to v1.19.0 (requires SVD)

**Bug Fixes**:
- [x] WorkStealingDeque.pop() memory safety (#13) ✅
  - Fixed wraparound-safe empty check
  - Regression test added

**Success Criteria**: ✅ **ACHIEVED** — 160 BLAS tests passing, all cross-compile targets verified

**Status**: ✅ **COMPLETE** (2026-03-21)

### v1.13.0 — Consumer Migration Support ✅ COMPLETE

Enable seamless migration of consumer projects (zr, silica, zoltraak) from custom implementations to zuda:

**Context**: All 5 phases complete, library feature-complete. 10+ open migration issues across 3 consumer projects. Focus on removing friction for adoption through migration guides, compatibility layers, and real-world migration examples.

**Target**: Close at least 3 high-impact migration issues by providing migration guides and compatibility layers ✅

**Categories**:
- [x] **Migration Guides** — Step-by-step migration documentation for each consumer ✅
  - ✅ silica BTree (4,300 LOC → zuda BTree) — docs/migrations/SILICA_BTREE.md
  - ✅ zr Graph (715 LOC → zuda Graph + algorithms) — docs/migrations/ZR_GRAPH.md
  - ✅ zoltraak Sorted Set (1,800 LOC → zuda SkipList) — docs/migrations/ZOLTRAAK_SORTEDSET.md
  - **Total impact**: -6,815 LOC across 3 consumers
- [x] **Compatibility Layers** — Thin wrappers matching existing consumer APIs ✅
  - ✅ silica BTree (src/compat/silica_btree.zig) — 4 tests, 20× insert speedup expected
  - ✅ zr DAG (src/compat/zr_dag.zig) — 5 tests, 47% memory reduction expected
  - ✅ zoltraak SortedSet (src/compat/zoltraak_sortedset.zig) — 12 tests, 12× insert/remove speedup expected
  - **Exported**: `zuda.compat.silica_btree`, `zuda.compat.zr_dag`, `zuda.compat.zoltraak_sortedset`
  - **Replaces**: 6,815 LOC total (silica 4,300 + zr 715 + zoltraak 1,800)
- [x] **Migration Examples** — Before/after code samples showing direct replacements ✅
  - ✅ silica BTree (examples/migrations/silica_btree/) — before.zig (4,300 LOC pattern) vs after.zig (zuda wrapper)
  - ✅ zr DAG (examples/migrations/zr_dag/) — before.zig (715 LOC pattern) vs after.zig (zuda wrapper)
  - ✅ zoltraak SortedSet (examples/migrations/zoltraak_sortedset/) — before.zig (1,800 LOC pattern) vs after.zig (zuda wrapper)
  - ✅ Comprehensive README (examples/migrations/README.md) — migration strategy, patterns, impact table
  - **Total**: 6 runnable examples + migration guide (-6,565 LOC savings demonstrated)
- [x] **API Harmonization** — Identify and fix API inconsistencies blocking migration ✅
  - ✅ docs/API_HARMONIZATION_v1.13.0.md — Analyzed 3 consumer codebases, 8 API gaps identified
  - ✅ All critical gaps resolved via compatibility layers (no blocking issues)
  - ⏭️ Nice-to-have enhancements deferred to v1.14.0 (bidirectional iterators, context-free constructors)
- [x] **Consumer PR Preparation** — Draft PR branches for at least one consumer migration ✅
  - ✅ zr PR #30 drafted (feat/migrate-to-zuda-graph branch)
  - ✅ -476 LOC (-67% reduction across 3 files)
  - ⚠️ Remaining work documented in PR (2 zuda API additions + test fixes)

**Success Criteria**: ✅ **ACHIEVED** — 3+ migration issues closed, 6,815 LOC reduction demonstrated

**Status**: ✅ **COMPLETE** (2026-03-20)

### v1.12.0 — Practical Utilities & Enhancements

Add commonly-needed utilities and ergonomic improvements based on real-world usage patterns:

**Context**: All 5 phases complete, library feature-complete. Focus shifts to developer experience, common patterns, and practical utilities that reduce boilerplate in consumer projects.

**Target**: Add 5-8 practical utilities/enhancements that improve day-to-day usage ✅

**Categories**:
- [x] **Comparison utilities** — Generic comparison helpers for common types ✅
  - `ascending()`, `descending()`, `deref()`, `tuple2()`, `stringAscending()`, `stringDescending()`
  - 6 functions, 10 tests (commit dda6eb3)
- [x] **Hashing utilities** — Auto-hash for common composite types ✅
  - `auto()`, `deref()`, `tuple2()`, `string()`, `stringCaseInsensitive()`, `eqlAuto()`
  - 6 functions, 9 tests (commit dda6eb3)
- [x] **Collection builders** — Fluent builder API for complex container initialization ✅
  - `fromSlice()`, `SliceBuilder.with()`, `SliceBuilder.filter()`, `SliceBuilder.map()`, `SliceBuilder.build()`
  - 5 functions, 24 tests (commit eb57408)
- [x] **Debug utilities** — Pretty-printing, diff helpers for testing ✅
  - `prettyPrint()`, `expectSliceEqual()`, `fmt()`
  - 3 functions, 29 tests (commit 24eeec6)
- [x] **Performance utilities** — Built-in benchmarking/profiling helpers ✅
  - `timeFn()`, `timeFnIters()`, `throughput()`, `mbPerSec()`, `AllocTracker`, `expectFaster()`
  - 6 functions/types, 14 tests (commit dc41b3c)

**Success Criteria**: ✅ **ACHIEVED** — 5/5 categories complete, 26 utilities total (target: 5-8)

**Status**: ✅ **COMPLETE** (2026-03-19)

### v1.11.0 — Aho-Corasick Performance Investigation ✅ COMPLETE

Investigate SIMD and goto completion optimizations to bridge the 82 MB/sec → 200 MB/sec gap:

**Context**: v1.10.0 linearization achieved 82-92 MB/sec (66 KB memory), below 200 MB/sec target. Explored SIMD vectorization and goto completion as optimization paths.

**Target**: Achieve ≥200 MB/sec throughput OR document fundamental architectural limits ✅

**Investigation Results**:
- [x] **SIMD vectorization analysis** — **REJECTED** ✅
  - **Finding**: Aho-Corasick is state-dependent (each char's state depends on previous char)
  - **Obstacle**: Failure link following is sequential, variable-length lookback
  - **Conclusion**: SIMD infeasible without massive precomputed tables (Hyperscan approach)
  - **Reference**: SIMD_ANALYSIS.md (v1.4.0), V1.11.0_FINDINGS.md
- [x] **Goto completion implementation** — **REJECTED** ✅
  - **Hypothesis**: Pre-compute all transitions to eliminate failure link loop (expected +50-100%)
  - **Implementation**: Added goto_table (state_count × 256 × 4 bytes), buildGotoCompletion()
  - **Performance**: 89 MB/sec (+8.5% from 82 MB/sec) ❌ FAR below expected
  - **Memory**: 445 KB (+579% from 66 KB) ❌ defeats sparse double-array purpose
  - **Root cause**: goto_table = 409 KB overhead (400 states × 256 × 4 bytes)
  - **Efficiency**: 6.7× memory increase for 8.5% speedup — terrible tradeoff
  - **Conclusion**: Goto completion FAILS design goal, reverted
- [x] **Tradeoff analysis** — Documented fundamental limits ✅
  - **Sparse (v1.10.0)**: 66 KB, 82 MB/sec ★★★★★ (memory-efficient)
  - **Goto completion**: 445 KB, 89 MB/sec ★★ (bad tradeoff)
  - **ASCII dense**: 19676 KB, 133 MB/sec ★ (massive memory)
  - **Hyperscan (SIMD)**: 10-100 MB, 1-5 GB/sec ❌ (bloat)
- [x] **Industry comparison** — zuda is competitive ✅
  - Rust aho-corasick (standard): 50-150 MB/sec, ~1-2 KB/pattern
  - Rust aho-corasick (DFA): 200-400 MB/sec, ~5-10 KB/pattern
  - **zuda DoubleArrayTrie**: 82 MB/sec, ~0.06 KB/pattern ★★★★★ best memory efficiency
- [x] **Documentation** — V1.11.0_FINDINGS.md created ✅
  - Documented SIMD infeasibility
  - Documented goto completion failure
  - Documented fundamental tradeoffs
  - Provided variant selection guide

**Success Criteria**: ✅ **ACHIEVED** — Documented that 200 MB/sec target requires algorithmic changes (dense transitions, massive memory overhead) that defeat sparse double-array purpose.

**Recommendations**:
1. **Accept 82 MB/sec** as near-optimal for memory-efficient design (66 KB footprint)
2. **Update PRD target** to ≥80 MB/sec (memory-efficient) OR document tradeoff matrix
3. **Design philosophy**: "zuda prioritizes memory efficiency over raw throughput in Aho-Corasick"
4. **Variant selection guide**:
   - Memory-constrained: DoubleArrayTrie (66 KB, 82 MB/sec)
   - Balanced: Generic Aho-Corasick (1570 KB, 59 MB/sec)
   - Throughput-critical: AhoCorasickASCII (19676 KB, 133 MB/sec)

**Outcome**: v1.10.0 sparse double-array (82 MB/sec, 66 KB) represents the **optimal balance** for memory-efficient Aho-Corasick. Further throughput gains require 6-296× memory increase.

**Status**: ✅ **COMPLETE** (2026-03-19) — All optimization avenues explored, architectural limits documented. No release needed (no implementation changes).

### v1.10.0 — Full DoubleArrayTrie Linearization (Phase 3) ✅ RELEASED

Completed full linearization with modest performance gains:

- [x] **Design linearized State structure** — Single 24-byte struct ✅ (commit d1d200e)
  - Created `State` struct: `{base: i32, check: u32, fail: u32, output_start: u32, output_len: u16, flags: u8, _padding: u8+u32}` (24 bytes)
  - Replaced 4 arrays (`base_check`, `is_leaf`, `fail`, `output`) with `states: []State` + `patterns: []usize`
  - IS_LEAF packed into State.flags bitfield
  - Flattened output patterns: linear patterns[] array with output_start/output_len indexing
- [x] **Refactor DoubleArrayTrie to use linearized layout** — All methods updated ✅
  - init(): allocates states[] + patterns[], uses temporary output_lists during construction
  - buildFailureLinks(): accesses states[s].fail directly
  - buildOutputLinks(): flattens ArrayList patterns into linear array
  - contains()/findAll(): sequential State struct access
  - validate(): State struct consistency checks
- [x] **Validate correctness** — All tests passing ✅
  - ✅ 722/722 tests PASS (100%)
  - ✅ 6 new Phase 3 tests validate struct layout, field replacements, memory packing
  - ✅ Zero memory leaks (std.testing.allocator)
- [x] **Benchmark performance improvement** — 92 MB/sec achieved ⚠️
  - Measured: 92 MB/sec (1000 patterns, 1 MB text)
  - Baseline (Phase 2): 88 MB/sec → **+5% improvement**
  - Target: 160 MB/sec → **-43% gap** (-68 MB/sec)
  - Analysis: Cache miss reduction achieved but memory-bound bottleneck remains
- [x] **Update performance targets** — Accept current state ✅
  - Phase 3 linearization completed, further gains require SIMD
  - Documented: 92 MB/sec is competitive for memory-efficient design (66 KB vs ASCII 19676 KB)
  - Trade-off: 23× memory reduction vs -31% throughput gap from dense array (133 MB/sec)

**Success criteria**: ⚠️ **PARTIAL** — Implementation complete, all tests pass, but performance target not met (92 vs 160 MB/sec).

**Outcome**: Phase 3 linearization successfully implemented with modest +5% gain. Cache locality improved but memory access patterns remain fundamental bottleneck. Further optimization requires SIMD vectorization (deferred to future milestone).

**Rationale**: Completed all planned linearization work. Performance gap indicates memory bandwidth limits, not cache miss issues. Accept 92 MB/sec as near-optimal for scalar (non-SIMD) implementation.

**Status**: ✅ **READY FOR RELEASE** — Implementation complete, tests passing (722/722), CI green, documentation updated.

### v1.9.0 — Aho-Corasick Cache Optimization Analysis ✅ RELEASED

Investigated cache-aware optimizations for DoubleArrayTrie performance improvement:

- [x] **Analyze double-array cache behavior** — Profile cache misses in DoubleArrayTrie ✅ (commit 7785f49)
  - **Baseline measured**: 125.0 MB/sec (1000 patterns, 1 MB text)
  - **Root cause identified**: 4-5 separate arrays = 2-5 cache misses per character
  - **Hot path analysis**: BASE, CHECK, FAIL, OUTPUT accesses fragmented across memory
  - **Documented**: Created docs/DOUBLEARRAY_CACHE_ANALYSIS.md (250+ lines)
  - **Solution designed**: Linearized SoA layout (20 bytes/state) → 1 cache miss vs 4 current
- [x] **Implement array linearization (Phase 2)** — Interleaved BASE+CHECK ⚠️ (commit 2eff97d)
  - **Implementation**: Created `BaseCheck` struct (8 bytes: i32 base + u32 check)
  - **Refactoring**: Replaced separate `base: []i32, check: []u32` with `base_check: []BaseCheck`
  - **Result**: **No improvement** (122.4 MB/sec → -2% regression from baseline 125.0 MB/sec)
  - **Root cause**: Interleaving fixed only 1 of 3-4 cache misses (BASE+CHECK now 1 load, but FAIL/OUTPUT still separate)
  - **Conclusion**: Partial linearization insufficient — full linearization (Phase 3) or acceptance required
- [x] **Strategic decision** — Accept current performance ✅
  - **Rationale**: Phase 3 (full State struct linearization) is complex refactoring (150+ lines, MEDIUM risk)
  - **Memory achievement**: 23× reduction (1570 KB → 66 KB) already achieved in v1.8.0 ✅
  - **Performance trade-off**: 88 MB/sec (sparse, memory-efficient) vs 133 MB/sec (dense, memory-heavy)
  - **Industry positioning**: Competitive with memory-conscious implementations
  - **Decision**: Defer full linearization to future milestone, release v1.9.0 with current state
- [x] **Benchmark validation** — Measured final performance ✅
  - **Result**: 88 MB/sec (vs 200 MB/sec target = -56% gap)
  - **Comparison**: Generic 60 MB/sec | ASCII 133 MB/sec | DoubleArray 88 MB/sec
  - **Memory**: ~66 KB (23× better than dense array, 296× better than ASCII)
- [x] **Documentation** — Document findings and decision ✅
  - Updated milestones.md with Phase 2 results and strategic decision
  - Documented cache optimization attempt and lessons learned
  - Preserved analysis for future optimization efforts

**Success criteria**: ⚠️ **PARTIAL** — Cache analysis complete, Phase 2 attempted, strategic decision made. Performance target not met (88 vs 200 MB/sec) but memory efficiency maintained. Incremental progress achieved.

**Outcome**: Valuable insights gained about cache behavior and linearization strategies. Phase 2 (interleaved BASE+CHECK) proved insufficient for performance gains. Memory efficiency (23× reduction) remains the primary achievement of DoubleArrayTrie implementation.

**Rationale**: v1.8.0 achieved excellent memory reduction (23×) but performance optimization proved more complex than anticipated. This milestone documents the investigation, attempted optimization, and pragmatic decision to accept current state rather than pursue risky refactoring.

**Status**: ✅ **READY FOR RELEASE** — All investigative work complete, benchmarks run, documentation updated. Tests passing (722/722), CI green.

### v1.8.0 — Double-Array Trie Implementation ✅ COMPLETE

Implement space-efficient double-array trie structure for Aho-Corasick:

- [x] **Double-array trie theory research** — Study Aoe 1989 construction algorithm ✅ (commit 2d52c15)
  - Documented BASE/CHECK array encoding for transitions
  - Analyzed space-time trade-offs: 16 bytes/state vs 2352 bytes/node (148× theoretical reduction)
  - Created docs/DOUBLE_ARRAY_TRIE_RESEARCH.md (300+ lines)
- [x] **BASE/CHECK array construction** — Implement Aoe's trie compression ✅ (commits 26a6451, cb540b9)
  - Designed BASE/CHECK array layout for Aho-Corasick automaton
  - Implemented minimal base conflict resolution (find smallest `b` where all `CHECK[b + c]` are empty)
  - **Critical bug fix**: Removed fixed 256-slot reservation that caused 127× memory overhead
  - Sparse allocation strategy: only allocate states as needed
  - Validated with 1000-pattern workload (all 40 tests passing)
- [x] **Search path optimization** — Linearize state transitions ✅ (commit 26a6451)
  - Replaced pointer chasing with array indexing (BASE[s] + c)
  - Implemented CHECK validation for transition verification
  - Sequential array access for failure links (FAIL array)
- [x] **Memory profiling** — Verify memory reduction ✅ (commits 5d6986b, cb540b9)
  - Created bench/memory_strings.zig with MemoryTracker
  - **Results** (1000 patterns): 66 KB peak memory
    - Generic HashMap: 1570 KB → **23× reduction** ✅
    - ASCII dense array: 19676 KB → **296× reduction** ✅
  - **Gap analysis**: 23× vs 50-100× target due to ArrayList overhead (~24 bytes/state for OUTPUT)
- [x] **Performance validation** — Benchmark double-array variant ⚠️ (commit cb540b9)
  - **Result**: 88 MB/sec (56% below 200 MB/sec target)
  - Trade-off: Sparse allocation improves memory but increases cache misses
  - Comparative results:
    - Generic HashMap: 59 MB/sec
    - ASCII dense array: 133 MB/sec
    - DoubleArrayTrie: 88 MB/sec (worse than dense array, but 23× less memory)

**Success criteria**: ⚠️ **PARTIAL** — 23× memory reduction achieved (vs 50-100× target), 88 MB/sec performance (vs ≥200 MB/sec target). Critical bug fixed, implementation correct.

**Outcome**: Double-array trie successfully implemented with significant memory savings. Performance gap due to cache-locality trade-offs inherent in sparse representation.

**Status**: ✅ **READY FOR RELEASE** — Implementation complete, all tests passing (722/722), CI green.

### v1.7.0 — Aho-Corasick Deep Optimization ✅ RELEASED

Close the 367 MB/sec performance gap through analysis and strategic planning:

- [x] **Memory footprint analysis** — NodeASCII structure breakdown ✅ (commit 723d1dc)
  - **Result**: 2352 bytes/node (87% transition table, 11% real_children, 2% metadata)
  - **Benchmark workload**: ~23 MB (1000 patterns → ~10k nodes)
  - **Key finding**: Dense transition table dominates memory but provides O(1) lookup
- [x] **Transition table compression evaluation** — Sparse vs dense trade-offs ✅ (commit 32e7ec5)
  - **Sparse alternatives analyzed**: Sorted array + binary search, compressed trie
  - **Result**: ❌ **Reject sparse array** — O(log k) lookups = 5× **slower** (26 MB/sec predicted)
    - Binary search: 5 cache misses per char (log₂(26)) vs 1 cache miss (dense array)
    - Cache miss penalty: ~200 ns (v1.6.0 RedBlackTree finding)
  - **Verdict**: Memory savings not worth 5× performance degradation
- [x] **Alternative automaton structures** — Double-array trie, DAWGs ✅ (commit 32e7ec5)
  - **Double-array trie**: O(1) lookup, 50-100× memory reduction (2352 → 30-50 bytes/node)
    - **Expected**: 200-300 MB/sec (+50-125% improvement)
    - **Complexity**: High (requires Aoe 1989 construction algorithm)
    - **Recommendation**: ✅ **Best long-term approach** — defer to v1.8.0 milestone
  - **DAWG**: 30-60% memory reduction, similar performance
    - **Verdict**: Low priority (marginal gains vs double-array)
- [x] **SIMD vectorization exploration** — Byte-parallel state simulation ✅ (commit 32e7ec5)
  - **Challenge**: State-dependent transitions complicate vectorization
  - **Hyperscan approach**: Massive precomputed tables (10-100× memory overhead) → 1-5 GB/sec
  - **Feasibility**: Requires Zig std.simd maturity (currently experimental)
  - **Recommendation**: ⚠️ **Defer to v1.9.0+** — wait for std.simd stabilization
- [x] **Industry benchmarks research** — Comparative analysis ✅ (commit 32e7ec5)
  - **Hyperscan (Intel AVX-512)**: 1-5 GB/sec (SIMD, massive memory)
  - **Rust aho-corasick standard**: 50-150 MB/sec (zuda @ 133 MB/sec is **competitive**)
  - **Rust aho-corasick DFA**: 200-400 MB/sec (double-array equivalent)
  - **RE2**: 100-500 MB/sec (DFA-based regex, not pure AC)
- [x] **Memory bandwidth limits documentation** — Fundamental ceiling ✅ (commit 32e7ec5)
  - **DRAM random access**: 1-2 GB/sec (64B cache lines)
  - **Theoretical ceiling**: 500-800 MB/sec (accounting for output link traversal)
  - **zuda @ 133 MB/sec**: 18% of ceiling → 3-4× improvement possible, not 10×
  - **Conclusion**: 500 MB/sec target is **unrealistic** without SIMD or double-array structure

**Success criteria**: ✅ **ACHIEVED** — Documented fundamental memory bandwidth limits with comparative evidence. Current 133 MB/sec is competitive with industry standard implementations.

**Recommendations**:
1. **Revise PRD target**: ≥200 MB/sec (achievable with linearization/double-array)
2. **v1.8.0 milestone**: Implement double-array trie (200-300 MB/sec expected)
3. **v1.9.0+ milestone**: SIMD exploration (400-600 MB/sec, requires std.simd)
4. **Document current performance**: "Competitive with Rust aho-corasick standard variant"

**Status**: ✅ **RELEASED** (tag v1.7.0, commit df73b38) — Analysis complete. Implementation deferred to v1.8.0 (double-array) and v1.9.0 (SIMD).

### v1.6.0 — Performance Benchmarking & Real-World Optimization ✅ RELEASED

Released 2026-03-17. Systematic performance measurement and targeted optimization based on benchmark data:

- [x] **Update Performance Table** ✅ (commit 43a1faf)
  - Ran all benchmarks except Aho-Corasick (benchmark crash — needs investigation)
  - Updated FibonacciHeap entries: insert 16ns, decreaseKey 18ns (both ✅)
  - Dijkstra: 422ms ✅ (-16% under 500ms target)
  - Fresh measurements: 8/9 targets verified (Aho-Corasack benchmark needs fix)
- [x] **RedBlackTree Deep Dive** ✅ (commit ec3ee69)
  - **Measured**: 257 ns/op insert, 262 ns/op lookup (1M random keys)
  - **Profiled**: Created micro-benchmark isolating components (allocator, comparison, traversal)
  - **Bottleneck identified**: Cache misses (~200 ns) dominate both operations — fundamental to pointer-based trees
  - **Benchmark verification**: Lookup phase is clean (no allocation overhead)
  - **Industry comparison**: C++ std::map: 150-250ns insert, 80-150ns lookup — zuda is competitive
  - **Verdict**: Implementation is **near-optimal**. PRD targets (200ns/150ns) are unrealistic for pointer-based trees.
  - **Recommendation**: Accept current performance (within industry norms), update PRD targets to insert ≤300ns / lookup ≤250ns (both PASS)
  - **Documentation**: docs/REDBLACKTREE_PERFORMANCE_ANALYSIS.md (425 lines, comprehensive analysis)
- [x] **Aho-Corasick Benchmark Fix & Investigation** ✅ (commit e7c2d59)
  - **Root cause found**: Benchmark freed patterns before automaton used them (dangling pointers)
  - **Fix**: Move pattern storage into context struct (patterns outlive automaton)
  - **Performance measured**: ASCII-optimized 133 MB/sec (vs 63 MB/sec in v1.4.0 — +111% improvement!)
  - **Generic variant**: 59 MB/sec (HashMap transitions, baseline for comparison)
  - **Status vs target**: FAIL -73% (367 MB/sec gap, 133 vs 500 target)
  - **Analysis**: Memory-bound (confirmed by v1.4.0 SIMD analysis) — already near-optimal for pointer-based traversal
  - **Recommendation**: Revise target to ≥150 MB/sec (current performance is competitive, 500 MB/sec unrealistic without SIMD)
- [x] **Benchmark Suite Completeness** ✅ (commit d0d4f25)
  - **Added 4 new benchmark suites** (lists, queues, hashing, cache) covering 19+ Phase 1 containers
  - **Working benchmarks** (13 total):
    - Lists: SkipList insert/search, XorLinkedList push/iterate, UnrolledLinkedList append/iterate, ConcurrentSkipList insert
    - Cache: LRUCache put/get, LFUCache put/get, ARCCache put/get (80% hit rate workload)
  - **Known implementation issues** (blocked benchmarks):
    - CuckooHashMap/RobinHoodHashMap/SwissTable: AutoContext type mismatch in cuckoo_hash_map.zig:456
    - WorkStealingDeque: Uses removed `std.atomic.fence` API (needs migration to std.Thread.Futex)
  - **Benchmark categories**: 13/13 categories have at least one benchmark
    - Trees ✓, Heaps ✓, BTrees ✓, Probabilistic ✓, Graphs ✓, Sorting ✓, Strings ✓, Lists ✓, Cache ✓
    - Queues/Hash (partial — implementation bugs block full coverage)
  - **Comparative benchmarks**: Deferred to future milestone (priority: performance measurement over comparison)
  - **Methodology docs**: Benchmark patterns established (warmup, min/max iterations, ns/op calculation)

### v1.4.0 — Performance & Optimization ✅ COMPLETE

Address performance gaps and optimize critical data structures:

- [x] **TimSort investigation & fix** (target: ≤10% overhead vs std.sort) ✅
  - **FIXED**: Buffer overflow causing SIGSEGV on large arrays
  - **Root cause**: Allocated buffer was items.len/2, but mergeRuns tried to slice up to len1 (≤ items.len)
  - **Solution**: Copy SMALLER run to buffer (TimSort optimization), merge backwards when needed
  - **Result**: **37% FASTER** than std.sort (35ms vs 55ms on 1M i64) — **EXCEEDS TARGET!**
- [x] **RedBlackTree optimization (partial)** (target: insert ≤200ns, lookup ≤150ns) ⚠️
  - **Baseline**: insert 329ns, lookup 593ns (original measurements)
  - **After optimization**: insert 255ns, lookup 258ns (commit 30c4c8e)
  - **Improvements**: insert -22%, lookup -56% from baseline
  - **Techniques**: Inlined hot paths, struct field reordering, prefetching
  - **Remaining gap**: insert +28% over target, lookup +72% over target
  - **Status**: Significant improvement but targets not met. Further optimization requires
    structural changes (color bit packing, parent pointer elimination) with complexity trade-offs.
    Recommend re-evaluating targets based on pointer-based tree fundamentals.
- [x] **Aho-Corasick optimization (partial)** (target: ≥500 MB/sec) ⚠️
  - **Baseline**: 58 MB/sec (with array transitions, before goto completion)
  - **After optimization**: 63 MB/sec (commit 2e6ef04, +9%)
  - **Technique**: Goto function completion — pre-compute all state transitions to eliminate
    runtime failure link following. Main search loop now has single array lookup per character.
  - **Remaining gap**: -87% under target (+433 MB/sec needed)
  - **Analysis**: Hit fundamental memory access limits. Target of 500 MB/sec = 6 CPU cycles/char.
    Current implementation: ~3-4 memory accesses/char (near-optimal for pointer-based traversal).
  - **Status**: Further gains require algorithmic rethinking (SIMD vectorization, precomputed
    match tables, or relaxed target based on memory bandwidth constraints).
- [x] **BloomFilter benchmark calculation fix** ✅
  - **Fixed**: Separated setup (1M inserts) from timed operation (10M lookups)
  - **Result**: Benchmark now accurately measures lookup-only performance (303M ops/sec)
  - **EXCEEDS TARGET**: 303M ops/sec >> 100M ops/sec target (+203%)
- [x] **Memory usage profiling & optimization pass** ✅
  - **Implemented**: MemoryTracker allocator wrapper in bench.zig
  - **New benchmark**: `zig build bench-memory` profiles 4 key containers (10k ops)
  - **Results**: RedBlackTree (481KB peak, 770k allocs), SkipList (2.7MB, 1M allocs), FibonacciHeap (747KB, 1M allocs), BTree (489KB, 17k allocs)
  - **Analysis**: BTree most memory-efficient (17k vs 770k-1M allocs), SkipList has 5.6x overhead
  - **Status**: Memory tracking framework complete, no leaks detected (1KB residual is benchmark overhead)
- [x] **SIMD opportunities exploration** ✅
  - **Document**: 425-line analysis in docs/SIMD_ANALYSIS.md (commit c1296ee)
  - **Coverage**: 6 hot-loop algorithms (TimSort, Aho-Corasick, BloomFilter, RadixSort, KMP/Boyer-Moore, sorting primitives)
  - **Key Finding**: Most zuda algorithms are memory-bound, not compute-bound
  - **Platform support**: SSE2/AVX2/AVX-512/NEON/RISC-V/WASM analysis
  - **Best candidates**: Sorting networks, RadixSort (AVX-512), string matching (16-byte chunks)
  - **Recommendation**: Defer SIMD implementation to v1.5.0+ (benchmark-driven, user demand)

### v1.5.0 — Code Quality & Maintainability ✅ RELEASED

Released 2026-03-17. Improve code quality, test coverage, and maintainability:

- [x] **Test quality audit** ✅
  - **Result**: 59 tests improved across 29 files (10/10 categories complete)
  - **Categories**: hash (9 tests), heap (7 tests), list (5 tests), queue (6 tests), trees (19 tests), spatial (1 test), probabilistic (1 test), cache (3 tests), persistent (3 tests), exotic (3 tests)
  - **Pattern fixed**: Tests relied on implicit validation (allocator leak detection, validate() not panicking) without checking actual behavior
  - **Improvement**: All tests now use explicit assertions (expectEqual, expect) to verify count, values, state
  - **Impact**: Tests now fail when behavior is wrong, not just when code panics
  - **Commits**: d7267eb, 1b8ec8c, 5eb42ad, 7e382ba, d236d8e, a2b28f7, 4699464
- [x] **Documentation completeness** ✅
  - **Result**: 100% coverage — 790/790 public API functions documented (commit 1cb55a0)
  - **Added**: 112 doc comments with Big-O time and space complexity annotations
  - **Files**: 25 containers (trees, spatial, probabilistic, hashing, graphs, strings, persistent, specialized)
  - **Pattern**: All docs follow "/// [Description]\n/// Time: O(...) | Space: O(...)" format
  - **Common functions documented**: lifecycle (init, deinit), Iterator.next, capacity (count, isEmpty), context helpers (compare, hash, eql), format, validate
  - **Quality**: 225 insertions, 0 logic changes, all 701 tests passing
- [x] **Memory safety verification** ✅
  - **Result**: No memory safety issues detected (commit dfee4a4)
  - **Leak detection**: All 701 tests use std.testing.allocator (automatic leak detection)
  - **Boundary conditions**: Verified empty state, single element, and stress tests (10k+ ops)
  - **Error paths**: Confirmed errdefer usage (30+ instances), zero @panic in production code
  - **Profiling**: Clean deallocation verified (1KB residual = benchmark overhead only)
  - **Documentation**: docs/MEMORY_SAFETY_VERIFICATION.md — comprehensive audit report
  - **Coverage**: 50+ containers audited across 9 categories (lists, queues, heaps, hash, trees, spatial, cache, persistent, specialized)
- [x] **Cross-compilation testing** ✅
  - **Result**: All 6 targets verified in CI (commit ef0a273)
  - **Targets**: x86_64-linux-gnu, aarch64-linux-gnu, x86_64-macos-none, aarch64-macos-none, x86_64-windows-msvc, wasm32-wasi
  - **Build mode**: ReleaseSafe optimization for all targets
  - **CI update**: Replaced aarch64-windows-msvc with wasm32-wasi to match documentation
  - **Issues**: None detected — all targets compile successfully
- [x] **API consistency review** ✅
  - **Result**: 85% compliant with intentional deviations (commit cd0368f)
  - **Generic Container Template**: 30/50 containers have all 5 core methods (init, deinit, count, iterator, validate)
  - **Iterator Protocol**: ✅ Consistent — all follow `next() -> ?T` or `next() -> !?T` pattern
  - **Error Naming**: ✅ No standardization issues — descriptive names with consistent suffixes
  - **Findings documented**: docs/API_CONSISTENCY_REVIEW.md (85 lines)
  - **Intentional exceptions**: Heaps (no iterator), Probabilistic (no validate), Graphs (no count) — design decisions
  - **Future enhancements**: 10 containers could add validate()/count() for completeness (tracked as recommendations)

---

## v2.0 Roadmap — Scientific Computing Platform

> zuda v2.0은 기존 DSA 라이브러리를 **과학 컴퓨팅 플랫폼**으로 확장한다.
> NumPy/SciPy의 Zig-native 대안으로서, 선형대수, 통계, 신호처리, 수치해석, 최적화를 포함한다.
> 각 마일스톤은 PRD Phase 6-12에 대응한다.

### Phase 6: NDArray Foundation

> **목표**: 과학 컴퓨팅의 핵심 데이터 구조인 N차원 배열 구현

#### v1.16.0 — NDArray Core

NDArray 타입의 기본 구조와 생성, 인덱싱, 메모리 레이아웃 구현:

**Categories**:
- [ ] **NDArray type definition** — `NDArray(T, ndim)` 기본 구조체
  - [ ] Shape, strides, data 포인터, allocator
  - [ ] Row-major (C order) 및 column-major (Fortran order) 메모리 레이아웃
  - [ ] Comptime-known rank, runtime-known shape
  - [ ] Tests: 20+ (생성, 소멸, 메모리 레이아웃 검증)
- [ ] **생성 함수** — 다양한 방식의 NDArray 생성
  - [ ] `zeros()`, `ones()`, `full()`, `empty()` — 초기화된 배열 생성
  - [ ] `arange()`, `linspace()` — 수열 생성
  - [ ] `fromSlice()`, `fromOwnedSlice()` — 기존 메모리에서 생성
  - [ ] `eye()`, `identity()` — 단위 행렬
  - [ ] Tests: 20+ (각 생성 함수별 edge case)
- [ ] **인덱싱 & 슬라이싱** — 다차원 데이터 접근
  - [ ] `get(indices)`, `set(indices, value)` — 단일 원소 접근
  - [ ] `slice(ranges)` — NumPy-style 슬라이싱 (non-owning view)
  - [ ] `at(index)` — 1차원 flat 인덱싱
  - [ ] Negative indexing 지원
  - [ ] Tests: 25+ (경계값, 음수 인덱스, 다차원 슬라이스)
- [ ] **Iterator protocol** — NDArray 순회
  - [ ] `NDArrayIterator` — storage order로 원소 순회
  - [ ] `next() -> ?T` protocol (v1.x 일관성)
  - [ ] Axis-wise iteration
  - [ ] Tests: 10+

**Success Criteria**: NDArray 생성, 인덱싱, 슬라이싱이 모두 동작하고 75+ 테스트 통과

#### v1.17.0 — NDArray Operations & Broadcasting

NDArray 연산, 브로드캐스팅, 변환 기능 구현:

**Categories**:
- [ ] **Element-wise 연산** — 사칙연산 및 수학 함수
  - [ ] 산술: `add`, `sub`, `mul`, `div`, `mod`, `neg`
  - [ ] 비교: `eq`, `ne`, `lt`, `le`, `gt`, `ge`
  - [ ] 수학: `abs`, `exp`, `log`, `log2`, `log10`, `sqrt`, `pow`
  - [ ] 삼각함수: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
  - [ ] Tests: 30+
- [ ] **Broadcasting** — NumPy-compatible broadcasting rules
  - [ ] Shape 호환성 검증
  - [ ] 자동 차원 확장 (1-dim expansion)
  - [ ] Broadcasting을 사용한 binary operation
  - [ ] Tests: 20+ (다양한 shape 조합, 비호환 shape 에러)
- [ ] **Reshape & Transform** — 형태 변환
  - [ ] `reshape()` — 데이터 재배치 없이 shape 변경 (가능한 경우)
  - [ ] `transpose()`, `permute()` — 축 전치/순열
  - [ ] `flatten()`, `ravel()` — 1차원으로 평탄화
  - [ ] `squeeze()`, `unsqueeze()` — 차원 제거/추가
  - [ ] `contiguous()` — 연속 메모리 보장 (필요 시 복사)
  - [ ] Tests: 20+
- [ ] **Reduction 연산** — 축 기반 집계
  - [ ] `sum()`, `prod()`, `mean()` — axis 파라미터 지원
  - [ ] `min()`, `max()`, `argmin()`, `argmax()`
  - [ ] `all()`, `any()` — boolean reduction
  - [ ] `cumsum()`, `cumprod()` — 누적 연산
  - [ ] Tests: 20+
- [ ] **I/O** — 직렬화/역직렬화
  - [ ] `save()`, `load()` — 바이너리 형식
  - [ ] `fromCSV()`, `toCSV()` — CSV 지원
  - [ ] Tests: 10+

**Success Criteria**: 모든 연산이 broadcasting과 함께 동작, 100+ 테스트 통과, NumPy 레퍼런스 출력 대비 검증

### Phase 7: Linear Algebra

> **목표**: BLAS 수준 행렬 연산 및 행렬 분해 알고리즘

#### v1.18.0 — BLAS & Core Linear Algebra

BLAS Level 1-3 및 기본 행렬 연산 구현:

**Categories**:
- [ ] **BLAS Level 1** — 벡터-벡터 연산
  - [ ] `dot()` — 내적
  - [ ] `axpy()` — y = αx + y
  - [ ] `nrm2()` — L2 노름
  - [ ] `asum()` — 절대값 합
  - [ ] `scal()` — 스칼라 곱
  - [ ] Tests: 20+
- [ ] **BLAS Level 2** — 행렬-벡터 연산
  - [ ] `gemv()` — y = αAx + βy
  - [ ] `trmv()`, `trsv()` — 삼각행렬 연산
  - [ ] `ger()` — 외적 (rank-1 update)
  - [ ] Tests: 15+
- [ ] **BLAS Level 3** — 행렬-행렬 연산
  - [ ] `gemm()` — C = αAB + βC (핵심 연산)
  - [ ] `trmm()`, `trsm()` — 삼각행렬 연산
  - [ ] Loop tiling, cache blocking 최적화
  - [ ] Tests: 20+
- [ ] **Matrix Properties** — 행렬 스칼라 속성
  - [ ] `det()` — 행렬식
  - [ ] `trace()` — 대각합
  - [ ] `rank()` — 행렬 계수
  - [ ] `cond()` — 조건수
  - [ ] Tests: 15+
- [ ] **Norms** — 벡터/행렬 노름
  - [ ] L1, L2, L∞ 노름 (벡터)
  - [ ] Frobenius, spectral 노름 (행렬)
  - [ ] Tests: 10+

**Performance Targets**:
- DGEMM (1024×1024): ≥ 5 GFLOPS
- Dot product (1M f64): ≥ 2 GFLOPS

**Success Criteria**: 80+ 테스트 통과, BLAS Level 3 벤치마크 ≥ 5 GFLOPS (1024×1024)

#### v1.19.0 — Decompositions & Solvers

행렬 분해 및 선형 시스템 해법 구현:

**Categories**:
- [ ] **LU 분해** — Partial pivoting
  - [ ] `lu()` → `{L, U, P}` (하삼각, 상삼각, 치환행렬)
  - [ ] In-place variant
  - [ ] Tests: 15+ (정칙, 특이, 직사각 행렬)
- [ ] **QR 분해** — Householder reflections
  - [ ] `qr()` → `{Q, R}`
  - [ ] Economy (thin) QR
  - [ ] Tests: 15+
- [ ] **Cholesky 분해** — 양의 정부호 행렬
  - [ ] `cholesky()` → `L` (LL^T = A)
  - [ ] Tests: 10+
- [ ] **SVD** — 특이값 분해
  - [ ] `svd()` → `{U, S, Vt}`
  - [ ] Economy SVD
  - [ ] Tests: 15+
- [ ] **고유값 분해** — Eigendecomposition
  - [ ] `eig()` — 일반 행렬 (실수/복소 고유값)
  - [ ] `eigh()` — 대칭 행렬 (실수 고유값 보장)
  - [ ] Tests: 15+
- [ ] **선형 시스템 해법** — Ax = b
  - [ ] `solve()` — LU 기반 직접 해법
  - [ ] `lstsq()` — 최소자승 해법 (QR/SVD 기반)
  - [ ] `inv()` — 역행렬
  - [ ] Tests: 15+
- [ ] **희소 행렬** — Sparse matrix 기초
  - [ ] CSR, CSC, COO 형식
  - [ ] Sparse-dense 곱
  - [ ] Sparse 직접 해법 (LU)
  - [ ] Tests: 20+

**Performance Targets**:
- LU (1024×1024): ≤ 200 ms
- SVD (512×512): ≤ 500 ms

**Success Criteria**: 모든 분해에 대해 재구성 오차 ≤ 1e-10 검증, LAPACK 레퍼런스 대비 정확도 확인, 105+ 테스트 통과

### Phase 8: Statistics & Random

> **목표**: 데이터 분석을 위한 통계 컴퓨팅 프리미티브

#### v1.20.0 — Descriptive Statistics & Distributions

기술 통계 및 확률 분포 구현:

**Categories**:
- [ ] **기술 통계** — Summary statistics
  - [ ] `mean()`, `median()`, `mode()` — 중심 경향
  - [ ] `std()`, `var()` — 산포도 (Bessel 보정 옵션)
  - [ ] `quantile()`, `percentile()` — 분위수
  - [ ] `skewness()`, `kurtosis()` — 형태 통계
  - [ ] `cov()`, `corrcoef()` — 공분산/상관 행렬
  - [ ] Tests: 25+
- [ ] **확률 분포** — PDF, CDF, Quantile, Sampling
  - [ ] 연속: Normal, Uniform, Exponential, Gamma, Beta, Chi-squared, Student-t, F
  - [ ] 이산: Poisson, Binomial, Bernoulli, Geometric
  - [ ] 각 분포: `pdf()`, `cdf()`, `quantile()`, `sample()`, `logpdf()`
  - [ ] Tests: 40+ (KS test로 분포 정확도 검증)
- [ ] **난수 생성** — PRNG
  - [ ] PCG64, Xoshiro256** 구현
  - [ ] `uniform()`, `normal()`, `exponential()` — 분포 샘플링
  - [ ] `shuffle()`, `choice()`, `multinomial()` — 조합 샘플링
  - [ ] Seed 기반 재현 가능성
  - [ ] Tests: 20+

**Success Criteria**: 모든 분포에 대해 SciPy 레퍼런스 검증 (KS test p > 0.05), 85+ 테스트 통과

#### v1.21.0 — Hypothesis Testing & Regression

가설 검정, 회귀 분석 구현:

**Categories**:
- [ ] **가설 검정** — Statistical tests
  - [ ] `ttest_1samp()`, `ttest_ind()`, `ttest_rel()` — t-검정
  - [ ] `chi2_test()` — 카이제곱 검정
  - [ ] `anova_oneway()` — 일원 분산분석
  - [ ] `ks_test()` — Kolmogorov-Smirnov 검정
  - [ ] `mannwhitney_u()` — 비모수 검정
  - [ ] 결과: `TestResult{ .statistic, .p_value, .df, .reject }`
  - [ ] Tests: 25+
- [ ] **상관 분석** — Correlation
  - [ ] `pearsonr()`, `spearmanr()`, `kendalltau()` — 상관 계수
  - [ ] `partial_corr()` — 편상관
  - [ ] Tests: 10+
- [ ] **회귀 분석** — Linear models
  - [ ] `ols()` — 최소자승 회귀 (계수, R², p-values, 잔차)
  - [ ] `polyfit()`, `polyval()` — 다항 회귀
  - [ ] Tests: 15+
- [ ] **히스토그램** — Binning
  - [ ] `histogram()` — uniform, auto (Sturges, Scott, Freedman-Diaconis)
  - [ ] `histogram2d()` — 2차원
  - [ ] Tests: 10+

**Success Criteria**: 모든 검정에 대해 R/SciPy 레퍼런스 결과와 일치, 60+ 테스트 통과

### Phase 9: Transforms & Signal Processing

> **목표**: 주파수 영역 분석 및 신호 처리

#### v1.22.0 — FFT & Spectral Analysis

Fast Fourier Transform 및 스펙트럼 분석 구현:

**Categories**:
- [ ] **FFT** — Cooley-Tukey 알고리즘
  - [ ] `fft()`, `ifft()` — 복소 FFT/IFFT
  - [ ] `rfft()`, `irfft()` — 실수 FFT (대칭 이용 최적화)
  - [ ] `fft2()`, `ifft2()` — 2D FFT
  - [ ] `fftfreq()` — 주파수 배열 생성
  - [ ] 2의 거듭제곱 + 임의 길이 지원 (Bluestein)
  - [ ] Tests: 30+ (정확도 ≤ 1e-10, Parseval 정리 검증)
- [ ] **DCT** — Discrete Cosine Transform
  - [ ] `dct()`, `idct()` — Type II / Type III
  - [ ] Tests: 10+
- [ ] **스펙트럼 분석** — Power spectrum
  - [ ] `periodogram()` — 파워 스펙트럼
  - [ ] `welch()` — Welch 방법
  - [ ] Tests: 10+
- [ ] **윈도우 함수** — Window functions
  - [ ] `hamming()`, `hann()`, `blackman()`, `kaiser()`, `bartlett()`
  - [ ] Tests: 10+

**Performance Targets**:
- FFT (1M complex f64): ≤ 30 ms
- FFT (4096 complex f64): ≤ 10 μs

**Success Criteria**: FFT 정확도 ≤ 1e-10 (DFT 대비), 60+ 테스트 통과, FFTW 대비 벤치마크

#### v1.23.0 — Convolution & Filtering

콘볼루션, 필터링 구현:

**Categories**:
- [ ] **콘볼루션** — Linear/circular convolution
  - [ ] `convolve()` — 선형 콘볼루션 (직접)
  - [ ] `fftconvolve()` — FFT 기반 콘볼루션 (대규모)
  - [ ] `correlate()` — 상호상관
  - [ ] Tests: 15+
- [ ] **디지털 필터** — FIR/IIR
  - [ ] `firwin()` — FIR 필터 설계
  - [ ] `lfilter()` — 차분 방정식 필터링
  - [ ] `filtfilt()` — 영위상 필터링
  - [ ] `butter()`, `cheby1()` — IIR 필터 설계
  - [ ] Tests: 20+

**Success Criteria**: 필터 주파수 응답 MATLAB/SciPy 대비 검증, 35+ 테스트 통과

### Phase 10: Numerical Methods

> **목표**: 과학 시뮬레이션을 위한 수치 해석 알고리즘

#### v1.24.0 — Integration, Differentiation & Interpolation

수치 적분, 미분, 보간 구현:

**Categories**:
- [ ] **수치 적분** — Quadrature
  - [ ] `trapezoid()` — 사다리꼴 법칙
  - [ ] `simpson()` — Simpson 법칙
  - [ ] `quad()` — 적응적 구적법 (Gauss-Kronrod)
  - [ ] `romberg()` — Romberg 적분
  - [ ] `gauss_legendre()` — Gauss-Legendre 구적법
  - [ ] Tests: 20+ (해석해 대비 ≤ 1e-12)
- [ ] **수치 미분** — Numerical differentiation
  - [ ] `diff()` — 유한 차분 (전방, 중앙, 후방)
  - [ ] `gradient()` — 다변수 기울기
  - [ ] `jacobian()`, `hessian()` — 야코비안, 헤시안
  - [ ] Tests: 15+
- [ ] **보간** — Interpolation
  - [ ] `interp1d()` — 선형 보간
  - [ ] `cubic_spline()` — 3차 스플라인
  - [ ] `lagrange()` — 라그랑주 보간
  - [ ] `pchip()` — 단조 보간
  - [ ] `interp2d()` — 2차원 보간
  - [ ] Tests: 15+

**Success Criteria**: 적분 정확도 ≤ 1e-12 (해석해 대비), 보간 MATLAB 대비 검증, 50+ 테스트 통과

#### v1.25.0 — Root Finding, ODE Solvers & Special Functions

방정식 해법, ODE 솔버, 특수함수 구현:

**Categories**:
- [ ] **방정식 해법** — Root finding
  - [ ] `bisect()` — 이분법
  - [ ] `newton()` — 뉴턴-랩슨법
  - [ ] `brent()` — 브렌트법
  - [ ] `secant()` — 할선법
  - [ ] `fixed_point()` — 고정점 반복
  - [ ] Tests: 15+
- [ ] **ODE 솔버** — Ordinary differential equations
  - [ ] `euler()` — 오일러 방법
  - [ ] `rk4()` — 4차 Runge-Kutta
  - [ ] `rk45()` — 적응적 RK (Dormand-Prince)
  - [ ] `bdf()` — Stiff systems (BDF)
  - [ ] Tests: 20+ (Van der Pol, Lorenz, exponential decay)
- [ ] **특수함수** — Special functions
  - [ ] `gamma()`, `lgamma()` — 감마함수
  - [ ] `beta()` — 베타함수
  - [ ] `erf()`, `erfc()` — 오차함수
  - [ ] `bessel_j()`, `bessel_y()` — 베셀함수
  - [ ] Tests: 15+
- [ ] **커브 피팅** — Curve fitting
  - [ ] `curve_fit()` — Levenberg-Marquardt
  - [ ] `polyfit()`, `polyval()`
  - [ ] Tests: 10+

**Success Criteria**: ODE 솔버 표준 문제 해결 (Lorenz attractor 등), 60+ 테스트 통과

### Phase 11: Optimization

> **목표**: 수학적 최적화 알고리즘

#### v1.26.0 — Unconstrained Optimization & Auto-differentiation

비제약 최적화 및 자동 미분 구현:

**Categories**:
- [ ] **경사 하강법** — Gradient-based methods
  - [ ] `gradient_descent()` — 기본 경사 하강법 (학습률 스케줄링)
  - [ ] `conjugate_gradient()` — 켤레 기울기법
  - [ ] `lbfgs()` — Limited-memory BFGS
  - [ ] `nelder_mead()` — 심플렉스 (도함수 불필요)
  - [ ] Tests: 20+ (Rosenbrock, Rastrigin, Beale)
- [ ] **자동 미분** — Forward-mode AD
  - [ ] `Dual(T)` — 이중수 타입 (comptime 전파)
  - [ ] `autodiff.gradient()` — AD 기반 기울기
  - [ ] `autodiff.jacobian()` — AD 기반 야코비안
  - [ ] Tests: 15+ (수치 미분 대비 ≤ 1e-8)
- [ ] **직선 탐색** — Line search
  - [ ] `armijo()`, `wolfe()`, `backtracking()`
  - [ ] Tests: 10+

**Success Criteria**: 모든 옵티마이저 표준 테스트 함수 수렴, AD 기울기 수치 미분 대비 ≤ 1e-8, 45+ 테스트 통과

#### v1.27.0 — Constrained Optimization & Linear Programming

제약 최적화 및 선형 프로그래밍 구현:

**Categories**:
- [ ] **선형 프로그래밍** — Linear programming
  - [ ] `simplex()` — 심플렉스법
  - [ ] `interior_point()` — 내점법
  - [ ] Tests: 15+
- [ ] **제약 최적화** — Constrained optimization
  - [ ] `augmented_lagrangian()` — 증강 라그랑주법
  - [ ] `quadratic_programming()` — 이차 프로그래밍
  - [ ] Tests: 10+
- [ ] **비선형 최소자승** — Nonlinear least squares
  - [ ] `levenberg_marquardt()` — LM 알고리즘
  - [ ] `gauss_newton()` — 가우스-뉴턴법
  - [ ] Tests: 10+

**Success Criteria**: LP 결과 GLPK 대비 검증, 35+ 테스트 통과

### Phase 12: v2.0 Integration & Release

> **목표**: 전체 통합, 성능 최적화, 공식 릴리즈

#### v1.28.0 — SIMD Acceleration & Cross-module Integration

핵심 연산 SIMD 가속 및 모듈 간 통합 테스트:

**Categories**:
- [ ] **SIMD 가속** — Zig SIMD 내장 함수 활용
  - [ ] GEMM SIMD 최적화 (4×4 커널)
  - [ ] FFT butterfly SIMD 최적화
  - [ ] NDArray element-wise SIMD 벡터화
  - [ ] Dot product SIMD 최적화
  - [ ] Tests: 20+ (정확도 + 성능 회귀)
- [ ] **Cross-module 통합 테스트** — 모듈 간 상호운용성
  - [ ] NDArray ↔ linalg (행렬 분해 후 NDArray 반환)
  - [ ] NDArray ↔ stats (통계 함수가 NDArray 입력 수용)
  - [ ] NDArray ↔ signal (FFT가 NDArray 입출력)
  - [ ] linalg ↔ optimize (최적화가 linalg 솔버 활용)
  - [ ] Tests: 30+
- [ ] **NumPy 호환성 가이드** — API 매핑 문서
  - [ ] NumPy 상위 50 함수 → zuda 매핑
  - [ ] 마이그레이션 가이드 (NumPy, Eigen, MATLAB)
  - [ ] 코드 비교 예제 (Python vs Zig 나란히)

**Success Criteria**: GEMM SIMD ≥ 5 GFLOPS, 50+ 통합 테스트 통과

#### v2.0.0 — Scientific Computing Release

공식 v2.0 릴리즈:

**Categories**:
- [ ] **종합 벤치마크** — Full benchmark suite
  - [ ] zuda vs NumPy/SciPy 비교 테이블
  - [ ] zuda vs Eigen/OpenBLAS 비교 테이블
  - [ ] 메모리 사용량 비교
  - [ ] 크로스 플랫폼 (x86_64, aarch64, WASM) 수치 안정성 검증
- [ ] **과학 컴퓨팅 가이드** — Tutorial documentation
  - [ ] "Getting Started with Scientific Computing in Zig"
  - [ ] 모듈별 튜토리얼 (NDArray, linalg, stats, signal, numeric, optimize)
  - [ ] 실전 예제: 이미지 처리, 신호 분석, 데이터 분석 파이프라인
- [ ] **API Reference 업데이트** — docs/API.md 확장
  - [ ] 모든 v2.0 모듈 API 문서화
  - [ ] 복잡도 계약 (Big-O) 명시
- [ ] **v2.0 Release**
  - [ ] Changelog 작성
  - [ ] build.zig.zon 버전 업데이트
  - [ ] GitHub Release + tag
  - [ ] README 최종 업데이트

**Success Criteria**: PRD Section 12의 v2.0 Success Criteria 전체 달성

---

## v2.0 Performance Targets (Summary)

| Metric | Target | Reference |
|--------|--------|-----------|
| DGEMM (1024×1024) | ≥ 5 GFLOPS | OpenBLAS single-thread |
| DGEMM (256×256) | ≥ 3 GFLOPS | OpenBLAS single-thread |
| Dot product (1M f64) | ≥ 2 GFLOPS | Naive C loop |
| NDArray element-wise (1M f64) | ≥ 1 GFLOPS | NumPy |
| LU decomposition (1024×1024) | ≤ 200 ms | LAPACK |
| SVD (512×512) | ≤ 500 ms | LAPACK |
| FFT (1M complex f64) | ≤ 30 ms | FFTW |
| FFT (4096 complex f64) | ≤ 10 μs | FFTW |
| Sparse GEMV (100K×100K, 1%) | ≥ 500 MFLOPS | SuiteSparse |
| Descriptive stats (1M f64) | ≤ 1 ms | NumPy |
| Random normal (1M f64) | ≤ 5 ms | NumPy default_rng |

---

## Deferred / Legacy Milestones

### v1.2.0 — Consumer Migrations

Validate zuda in production through consumer project adoption:

- [ ] zr migration (1,189 LOC replacement) — issues zr#21-#25 filed
- [ ] silica migration (7,000 LOC replacement) — issues silica#4, silica#5 filed
- [ ] zoltraak migration (3,435 LOC replacement) — issues zoltraak#1-#3 filed
- [ ] API refinements based on consumer feedback
- [ ] Migration guide documentation


---

## Performance Targets

| Metric | Target (v1.7.0 revised) | Actual (v1.6.0) | Status |
|--------|--------|--------|--------|
| BTree(128) range scan | ≥ 50M keys/sec | 83M keys/sec | ✅ +66% |
| RedBlackTree insert | ≤ 300 ns/op¹ | 257 ns/op | ✅ -14% under target |
| RedBlackTree lookup | ≤ 250 ns/op¹ | 262 ns/op | ⚠️ +5% over (marginal) |
| TimSort overhead | ≤ 10% vs std.sort | **-37% (faster!)** | ✅ EXCEEDS! |
| Aho-Corasick (ASCII) | ≥ 200 MB/sec² | 133 MB/sec | ⚠️ -33% (67 MB/sec gap) |
| FibonacciHeap insert | ≤ 100 ns amortized | 16 ns/op | ✅ -84% under target |
| FibonacciHeap decrease-key | ≤ 50 ns amortized | 18 ns/op | ✅ -64% under target |
| BloomFilter lookup | ≥ 100M ops/sec | 1.25B ops/sec | ✅ +1150% |
| Dijkstra (1M nodes) | ≤ 500 ms | 422 ms | ✅ -16% under target |

**Notes**:
1. RedBlackTree targets revised in v1.6.0 based on deep-dive analysis (ec3ee69). Original targets (200ns insert / 150ns lookup) were based on array-based structures and unrealistic for pointer-based trees. New targets reflect industry norms (C++ std::map: 150-250ns insert, 80-150ns lookup). See docs/REDBLACKTREE_PERFORMANCE_ANALYSIS.md for comprehensive analysis.
2. Aho-Corasick target revised in v1.7.0 based on optimization analysis (32e7ec5). Original target (500 MB/sec) requires SIMD vectorization or double-array trie structure. Current pointer-based implementation @ 133 MB/sec is competitive with Rust aho-corasick standard variant (50-150 MB/sec). New target (200 MB/sec) is achievable with linearization or double-array structure (v1.8.0 milestone). See docs/AHOCORASICK_OPTIMIZATION_ANALYSIS.md for comprehensive analysis.

---

## Completed Milestones

| Phase | Name | Release | Date | Summary |
|-------|------|---------|------|---------|
| Phase 1 | Foundations | v0.1.0 | 2026-03 | SkipList, CuckooHashMap, RobinHoodHashMap, SwissTable, ConsistentHashRing, FibonacciHeap, BinomialHeap, PairingHeap, DaryHeap, XorLinkedList, UnrolledLinkedList, Deque, CI, testing harness, benchmark framework |
| Phase 2 | Trees & Range Queries | v0.5.0 | 2026-03 | RedBlackTree, AVLTree, SplayTree, AATree, ScapegoatTree, Trie, RadixTree, BTree, SegmentTree, LazySegmentTree, FenwickTree, SparseTable, IntervalTree, KDTree, RTree, QuadTree, OctTree, SuffixArray, SuffixTree |
| Phase 3 | Graph Algorithms | — | 2026-03 | AdjacencyList, AdjacencyMatrix, CompressedSparseRow, EdgeList, BFS, DFS, Dijkstra, Bellman-Ford, A*, Floyd-Warshall, Johnson's, Kruskal, Prim, Boruvka, Tarjan SCC, Kosaraju, bridges, articulation points, Edmonds-Karp, Dinic, Push-Relabel, Hopcroft-Karp, Hungarian, topological sort |
| Phase 4 | Algorithms & Probabilistic | — | 2026-03 | TimSort, IntroSort, RadixSort, CountingSort, BlockSort, in-place MergeSort, KMP, Boyer-Moore, Rabin-Karp, Aho-Corasick, Z-algorithm, BloomFilter, CuckooFilter, CountMinSketch, HyperLogLog, LRUCache, LFUCache, GCD, modexp, Miller-Rabin, convex hull, closest pair, LIS, LCS, edit distance, knapsack |
| Phase 5 | Advanced & Polish | v1.0.0 | 2026-03-14 | LockFreeQueue, LockFreeStack, ConcurrentSkipList, ConcurrentHashMap, PersistentArray, PersistentRBTree, PersistentHashMap (HAMT), DisjointSet, VanEmdeBoasTree, DancingLinks, Rope, BK-Tree, C API, documentation, 213 public exports |
| Post-v1.0.0 | Performance Optimization | v1.1.0 | 2026-03-15 | Fixed FibonacciHeap deinit & insert API, TimSort allocation bug, BloomFilter benchmark; optimized Aho-Corasick build & search; analyzed RedBlackTree (near-optimal) |
| Post-v1.1.0 | Iterator System & Completeness | v1.3.0 | 2026-03-15 | 8 iterator adaptors (Map, Filter, Chain, Zip, Take, Skip, Enumerate, collect), A* comprehensive tests, PersistentArray.pop(), comprehensive iterator pattern guide |
| Post-v1.3.0 | Ergonomic Enhancements | v1.14.0 | 2026-03-20 | BTree/SkipList/RedBlackTree reverseIterator() (36 tests), SkipList.initDefault() (28 tests), AdjacencyList convenience constructors (48 tests), 112 new tests total |

### Post-v1.0.0 Activity

20 commits since v1.0.0 release:
- Benchmark API fixes for Zig 0.15.2 ArrayList changes
- Comprehensive benchmark suite for all PRD performance targets
- Glob algorithm implementation
- PersistentRBTree lifetime management documentation
- Consumer migration issue filing (zr, silica, zoltraak — 11,624 LOC total)

### Closed Issues

| # | Title | Closed |
|---|-------|--------|
| #6 | ArrayList API migration incomplete | 2026-03-13 |
| #5 | DancingLinks implementation | 2026-03-13 |
| #4 | CI warning confirmed | 2026-03-12 |
| #3 | CRITICAL: CI failing | 2026-03-12 |
| #2 | Completed phase not released | 2026-03-12 |
| #1 | SuffixTree edge splitting bug | 2026-03-09 |

---

## Milestone Establishment Process

미완료 마일스톤이 **2개 이하**가 되면, 에이전트가 자율적으로 새 마일스톤을 수립한다.

**입력 소스** (우선순위 순):
1. `gh issue list --state open --label feature-request` — 사용자 요청 기능
2. `docs/PRD.md` — 아직 구현되지 않은 PRD 항목
3. 기술 부채 — Known Limitations, TODO, 성능 병목
4. 소비자 프로젝트 요구사항 — Consumer Use Case Registry (CLAUDE.md) 참조

**수립 규칙**:
- 마일스톤 하나는 **단일 테마**로 구성 (여러 작은 기능을 하나의 주제로 묶음)
- 1-2주 내 완료 가능한 범위로 스코프 설정
- 버전 번호는 마지막 마일스톤의 다음 번호로 자동 부여
- 수립 후 이 파일에 추가하고 커밋: `chore: add milestone vX.Y.0`

---

## Dependency Tracking

zuda는 순수 라이브러리이므로 외부 의존성 마이그레이션은 없다.
소비자 프로젝트(zr, silica, zoltraak)로의 마이그레이션 발행은 CLAUDE.md의 **소비자 마이그레이션 발행 프로토콜**을 참조한다.

### Consumer Migration Issues Filed

| Consumer | Issues | Total LOC | Status |
|----------|--------|-----------|--------|
| zr | #21, #22, #23, #24, #25 | 1,189 | Pending (blocked on v1.1.0) |
| silica | #4, #5 | 7,000 | Pending (blocked on v1.1.0) |
| zoltraak | #1, #2, #3 | 3,435 | Pending (blocked on v1.1.0) |
