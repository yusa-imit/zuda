# zuda — Product Requirements Document

> **Zig Universal Datastructures and Algorithms**
> A comprehensive, production-grade collection of data structures and algorithms written in idiomatic Zig.

**Version:** 2.0 (Major Revision)
**Date:** March 20, 2026
**Author:** Yusa

---

## 1. Overview

### 1.1 Vision

zuda is a batteries-included library of data structures, algorithms, and scientific computing for the Zig programming language. While Zig's standard library intentionally ships a minimal set of containers (ArrayList, HashMap, LinkedList, PriorityQueue, BitSet, Treap), real-world projects frequently need structures like balanced BSTs, concurrent skip lists, spatial indices, graph algorithms, advanced string matchers, multi-dimensional arrays, and linear algebra routines. Today, Zig developers either roll their own, wrap C libraries, or go without.

zuda fills this gap. Starting as the **go-to DSA library for Zig** (v1.x), zuda is evolving into a **comprehensive scientific computing platform** (v2.0) — the Zig-native alternative to Python's NumPy/SciPy ecosystem. It combines the breadth of Rust's `std::collections` + `petgraph` + `ndarray`, C++'s Boost + Eigen, and Python's NumPy + SciPy — but designed from the ground up to embrace Zig's philosophy: explicit allocators, comptime generics, zero hidden control flow, and C ABI interoperability.

**v1.x** delivered 100+ data structures and 80+ algorithms. **v2.0** extends this foundation with N-dimensional arrays, linear algebra, statistics, FFT, numerical methods, and optimization — making zuda the first Zig library to offer a unified DSA + scientific computing stack.

### 1.2 Why Zig?

- **Comptime generics without codegen overhead.** Data structures parameterized by type, comparator, and hash function can be fully monomorphized at compile time — no vtables, no type erasure, no runtime dispatch.
- **Explicit allocator passing.** Every container accepts an `std.mem.Allocator`, making it trivial to use arena allocators for batch workloads, fixed-buffer allocators for embedded targets, or testing allocators that detect leaks.
- **No hidden allocations or control flow.** Users can reason about exactly when and how much memory a container will request.
- **C ABI export.** Any zuda container can be wrapped with a thin C API layer for consumption from Python, Node.js, Go, or any language with C FFI.
- **Cross-compilation out of the box.** A single `zig build` produces binaries for Linux, macOS, Windows, WASI, and bare-metal targets.

### 1.3 Project Goals

| Priority | Goal |
|----------|------|
| P0 | Zig-idiomatic API with comptime generics, explicit allocators, and iterator protocol |
| P0 | Correctness — every data structure backed by property-based and fuzz tests |
| P1 | Performance competitive with or exceeding C/C++ equivalents on standard benchmarks |
| P1 | Comprehensive coverage — aim for the broadest DSA + scientific computing collection in any systems language ecosystem |
| P1 | **NDArray with linear algebra** — N-dimensional array with BLAS-level matrix operations, decompositions, and solvers |
| P2 | First-class documentation with complexity annotations, usage examples, and algorithm explanations |
| P2 | C API layer for cross-language consumption |
| P2 | **NumPy-competitive API** — familiar API surface for users migrating from Python scientific computing |
| P3 | SIMD-accelerated paths for sorting, searching, string algorithms, and **matrix operations** |
| P3 | WASM and embedded (no-std) support |
| P3 | **Statistical computing & numerical methods** — distributions, hypothesis testing, interpolation, ODE solvers |

### 1.4 Non-Goals

- **Replacing `std`.** zuda complements the standard library; it does not reimplement ArrayList or HashMap. Where `std` already provides a good solution, zuda defers to it.
- ~~**Becoming a math library.**~~ *(Revised in v2.0)* — Linear algebra, FFT, and numerical computing are now **in scope** as core v2.0 deliverables. zuda aims to be the Zig-native alternative to NumPy/SciPy.
- **Distributed algorithms.** Consensus protocols, CRDTs, and distributed hash tables are out of scope.
- **Deep learning framework.** zuda provides the mathematical primitives (matrix ops, auto-diff) but is not a training framework like PyTorch/TensorFlow. Neural network layers, optimizers, and model serialization are out of scope.
- **Symbolic computation.** Computer algebra systems (symbolic differentiation, polynomial factoring, equation solving) are out of scope. zuda focuses on numerical methods.

---

## 2. Target Users & Use Cases

### 2.1 Primary Users

**v1.x — Data Structures & Algorithms:**
- **Zig application developers** building systems that need more than what `std` provides (e.g., interval trees for a scheduler, tries for an autocomplete engine, graph algorithms for a dependency resolver).
- **Competitive programmers and algorithm enthusiasts** looking for a well-tested Zig reference implementation.
- **Systems programmers** porting C/C++ codebases to Zig who need drop-in replacements for STL / Boost containers.
- **Embedded developers** who need memory-predictable containers with fixed-capacity variants.

**v2.0 — Scientific Computing (New):**
- **Scientific computing practitioners** who need NumPy/SciPy-equivalent functionality in a compiled, low-latency language without Python's GIL or garbage collector overhead.
- **Data engineers & analysts** building high-performance data pipelines in Zig who need matrix operations, statistics, and transforms.
- **Game & simulation developers** who need linear algebra (transform matrices, physics calculations) and numerical methods (ODE solvers, interpolation) in a systems language.
- **ML infrastructure engineers** building inference engines, feature stores, or preprocessing pipelines that require fast numerical primitives without Python overhead.
- **Signal processing engineers** building real-time audio, image, or sensor processing systems in Zig.
- **Researchers** prototyping numerical algorithms who want C-level performance with a modern, safe language.

### 2.2 Key Use Cases

**Data Structures & Algorithms (v1.x):**
1. **Dependency resolution** — Topological sort, cycle detection, DAG shortest path.
2. **Spatial indexing** — R-Tree / k-d tree for game engines, GIS, or collision detection.
3. **Autocomplete / prefix matching** — Trie and radix tree with ranked results.
4. **Task scheduling** — Fibonacci heap for efficient decrease-key in Dijkstra's algorithm.
5. **Text processing** — Aho-Corasick multi-pattern search, suffix arrays for indexing.
6. **In-memory caching** — LRU / LFU eviction policies with O(1) operations.
7. **Network analysis** — Max-flow, strongly connected components, betweenness centrality.
8. **Interval queries** — Segment tree / Fenwick tree for range-sum and range-min queries.

**Scientific Computing (v2.0 — New):**
9. **Linear algebra** — Matrix multiplication, decompositions (LU, QR, SVD, Cholesky), eigenvalue computation, linear system solving — core primitives for ML inference, physics simulation, and engineering.
10. **Statistical analysis** — Descriptive statistics, hypothesis testing, correlation analysis, probability distributions — for data analysis pipelines and A/B testing frameworks.
11. **Signal processing** — FFT/IFFT, convolution, filtering, windowing — for real-time audio processing, image analysis, and sensor data.
12. **Numerical simulation** — ODE solvers, integration, interpolation — for physics engines, financial modeling, and scientific simulation.
13. **Optimization** — Gradient descent, linear programming, root finding — for parameter tuning, resource allocation, and curve fitting.
14. **Image & array processing** — NDArray operations (reshape, broadcast, slice) — for image manipulation, tensor preprocessing, and multi-dimensional data analysis.

---

## 3. Architecture

### 3.1 Module Organization

```
zuda/
├── build.zig
├── build.zig.zon
├── src/
│   ├── zuda.zig                  # Root — re-exports all public types
│   │
│   ├── containers/               # Data Structures (v1.x)
│   │   ├── lists/                # Sequential containers
│   │   ├── trees/                # Tree-based containers
│   │   ├── graphs/               # Graph representations
│   │   ├── heaps/                # Heap variants
│   │   ├── hashing/              # Hash-based containers
│   │   ├── queues/               # Queue / deque variants
│   │   ├── strings/              # String-specialized structures
│   │   ├── spatial/              # Spatial index structures
│   │   └── probabilistic/        # Bloom filter, Count-Min Sketch, etc.
│   │
│   ├── algorithms/               # Algorithms (v1.x)
│   │   ├── sorting/
│   │   ├── searching/
│   │   ├── graph/
│   │   ├── string/
│   │   ├── math/                 # GCD, modexp, primality, combinatorics
│   │   ├── geometry/             # Convex hull, line intersection, etc.
│   │   └── dynamic_programming/  # Common DP utilities
│   │
│   ├── iterators/                # Composable iterator adaptors
│   │
│   ├── ndarray/                  # N-dimensional Array (v2.0 — NEW)
│   │   ├── ndarray.zig           # Core NDArray type (shape, stride, data)
│   │   ├── ops.zig               # Element-wise operations (+, -, *, /)
│   │   ├── broadcast.zig         # NumPy-style broadcasting rules
│   │   ├── slice.zig             # Advanced slicing & indexing
│   │   ├── reshape.zig           # Reshape, transpose, permute
│   │   └── io.zig                # Serialization (CSV, binary)
│   │
│   ├── linalg/                   # Linear Algebra (v2.0 — NEW)
│   │   ├── blas.zig              # BLAS Level 1-3 (dot, gemv, gemm)
│   │   ├── decompose.zig         # LU, QR, Cholesky, SVD, Eigen
│   │   ├── solve.zig             # Linear system solvers (Ax=b)
│   │   ├── sparse.zig            # Sparse matrix (CSR, CSC, COO)
│   │   └── norm.zig              # Vector/matrix norms
│   │
│   ├── stats/                    # Statistics (v2.0 — NEW)
│   │   ├── descriptive.zig       # Mean, median, std, variance, quantile
│   │   ├── distributions.zig     # Normal, Poisson, Binomial, Uniform, etc.
│   │   ├── hypothesis.zig        # t-test, chi-square, ANOVA, p-value
│   │   ├── correlation.zig       # Pearson, Spearman, covariance matrix
│   │   └── random.zig            # PRNG (PCG, Xoshiro), sampling
│   │
│   ├── signal/                   # Signal Processing (v2.0 — NEW)
│   │   ├── fft.zig               # FFT / IFFT (Cooley-Tukey)
│   │   ├── dct.zig               # Discrete Cosine Transform
│   │   ├── conv.zig              # Convolution & cross-correlation
│   │   ├── filter.zig            # FIR / IIR filters
│   │   └── window.zig            # Hamming, Hann, Blackman, Kaiser
│   │
│   ├── numeric/                  # Numerical Methods (v2.0 — NEW)
│   │   ├── integrate.zig         # Quadrature (Simpson, Gauss-Legendre)
│   │   ├── differentiate.zig     # Numerical differentiation
│   │   ├── interpolate.zig       # Linear, cubic spline, Lagrange
│   │   ├── roots.zig             # Newton, bisection, Brent
│   │   ├── ode.zig               # ODE solvers (Euler, RK4, RK45)
│   │   └── optimize.zig          # Gradient descent, Nelder-Mead, L-BFGS
│   │
│   └── internal/                 # Shared utilities (not public API)
│       ├── testing.zig           # Property-based test helpers
│       ├── bench.zig             # Micro-benchmark harness
│       └── simd.zig              # SIMD abstraction layer (v2.0)
│
├── tests/                        # Integration & fuzz tests
├── bench/                        # Benchmark suites
├── examples/                     # Runnable usage examples
└── docs/                         # Generated & hand-written documentation
```

### 3.2 Design Principles

**Principle 1: Allocator-First**

Every heap-allocating container takes `std.mem.Allocator` as its first init parameter. Containers also provide `*Unmanaged` variants that do not store the allocator and instead require it on every method call, matching the pattern established by `std.ArrayListUnmanaged`.

```zig
// Managed — stores allocator internally
var tree = zuda.RedBlackTree(i64, {}, std.math.order).init(allocator);
defer tree.deinit();

// Unmanaged — caller passes allocator per-call
var tree: zuda.RedBlackTreeUnmanaged(i64, {}, std.math.order) = .empty;
defer tree.deinit(allocator);
```

**Principle 2: Comptime Configuration**

Where a data structure's behavior can be parameterized (comparator, hash function, branching factor, fixed capacity), prefer comptime parameters over runtime options.

```zig
// B-Tree with comptime branching factor
const BTree = zuda.BTree(i64, []const u8, .{ .order = 128 });

// Bloom filter with comptime hash count
const Bloom = zuda.BloomFilter([]const u8, .{ .num_hashes = 7, .bit_count = 1 << 20 });
```

**Principle 3: Iterator Protocol**

All iterable containers expose a `next() -> ?T` iterator, compatible with Zig's `while (iter.next()) |item|` pattern. Iterators are lazy and composable.

```zig
var iter = tree.iterator();
while (iter.next()) |entry| {
    std.debug.print("{}: {}\n", .{ entry.key, entry.value });
}
```

**Principle 4: Complexity Contracts**

Every public function's doc comment states its time and space complexity using Big-O notation. These are not aspirational — they are tested via benchmark regression.

```zig
/// Inserts `key` into the red-black tree, maintaining balance.
/// Time: O(log n) | Space: O(1) amortized
pub fn insert(self: *Self, key: K) !void { ... }
```

**Principle 5: Fixed-Capacity Variants**

For embedded and latency-sensitive contexts, heap-allocating containers offer a `Bounded` variant backed by a comptime-known fixed buffer with no allocator needed.

```zig
// Stack-allocated ring buffer — no allocator, no heap
var ring: zuda.BoundedRingBuffer(u8, 4096) = .{};
```

### 3.3 Compatibility with `std`

zuda containers interoperate with the standard library:

- Any zuda container that stores elements contiguously exposes `.items` / `.slice()` returning `[]T`, consumable by `std.mem`, `std.sort`, `std.fmt`.
- Graph algorithms accept a generic `Graph` interface so users can bring their own adjacency representation.
- Sorting algorithms operate on `[]T` slices directly.

---

## 4. Implemented Features (v1.x) — Reference

> 전체 API 상세는 [docs/API.md](API.md) 참조. 아래는 요약.

**100+ Data Structures**: SkipList, XorLinkedList, UnrolledLinkedList, Deque, RedBlackTree, AVLTree, SplayTree, AATree, ScapegoatTree, BTree, Trie, RadixTree, SegmentTree, LazySegmentTree, FenwickTree, SparseTable, IntervalTree, KDTree, RTree, QuadTree, OctTree, SuffixArray, SuffixTree, FibonacciHeap, BinomialHeap, PairingHeap, DaryHeap, CuckooHashMap, RobinHoodHashMap, SwissTable, ConsistentHashRing, AdjacencyList, AdjacencyMatrix, CompressedSparseRow, EdgeList, BloomFilter, CuckooFilter, CountMinSketch, HyperLogLog, MinHash, LRUCache, LFUCache, ARCCache, LockFreeQueue, LockFreeStack, ConcurrentSkipList, WorkStealingDeque, PersistentArray, PersistentRBTree, PersistentHashMap, DisjointSet, VanEmdeBoasTree, DancingLinks, Rope, BK-Tree, WaveletTree, CartesianTree, FusionTree, Link-Cut Tree

**80+ Algorithms**: TimSort, IntroSort, RadixSort, CountingSort, BlockSort, MergeSort, KMP, Boyer-Moore, Rabin-Karp, Aho-Corasick, Z-algorithm, BFS, DFS, Dijkstra, Bellman-Ford, A*, Floyd-Warshall, Johnson, Kruskal, Prim, Borůvka, Tarjan SCC, Kosaraju, Edmonds-Karp, Dinic, Push-Relabel, Hopcroft-Karp, Hungarian, Topological Sort, Convex Hull, Closest Pair, GCD, ModExp, Miller-Rabin, Sieve, NTT, LIS, LCS, Edit Distance, Knapsack

---

## 5. API Design

> 코딩 컨벤션 및 컨테이너 템플릿 상세는 [CLAUDE.md](../CLAUDE.md#coding-standards) 참조.

- **Naming**: PascalCase (types), camelCase (functions), snake_case (constants)
- **Error handling**: `!T` error unions. Never panic. `error.OutOfMemory` for allocation failures.
- **Container pattern**: init/deinit → count/isEmpty → insert/remove → get/contains → iterator → format/validate
- **Complexity contracts**: Every public function documents Big-O in doc comments
- **C FFI**: `@export` 기반 thin wrapper — 상세는 [examples/FFI_README.md](../examples/FFI_README.md) 참조

---

## 6. Non-Functional Requirements

### 6.1 Performance Targets

**v1.x — Data Structures & Algorithms:**

| Metric | Target | Benchmark |
|--------|--------|-----------|
| RedBlackTree insert | ≤ 300 ns/op (1M random keys) | vs. `std.Treap`, C++ `std::map` |
| RedBlackTree lookup | ≤ 250 ns/op (1M random keys) | vs. `std.Treap`, C++ `std::map` |
| BTree(128) range scan | ≥ 50M keys/sec (sequential) | vs. SQLite B-Tree, LMDB |
| FibonacciHeap decrease-key | ≤ 50 ns amortized | vs. binary heap extract+reinsert |
| BloomFilter lookup | ≥ 100M ops/sec | vs. C reference (libbloom) |
| Dijkstra (1M nodes, 5M edges) | ≤ 500 ms | vs. Boost.Graph, igraph |
| TimSort (1M i64, random) | competitive with `std.sort` | ≤ 10% overhead |
| Aho-Corasick (1000 patterns, 1MB text) | ≥ 200 MB/sec throughput | vs. Rust aho-corasick |

**v2.0 — Scientific Computing (New):**

| Metric | Target | Benchmark |
|--------|--------|-----------|
| DGEMM (1024×1024) | ≥ 5 GFLOPS | vs. OpenBLAS (single-thread) |
| DGEMM (256×256) | ≥ 3 GFLOPS | vs. OpenBLAS (single-thread) |
| Dot product (1M f64) | ≥ 2 GFLOPS | vs. naive C loop |
| NDArray element-wise (1M f64) | ≥ 1 GFLOPS | vs. NumPy |
| LU decomposition (1024×1024) | ≤ 200 ms | vs. LAPACK reference |
| SVD (512×512) | ≤ 500 ms | vs. LAPACK reference |
| FFT (1M complex f64) | ≤ 30 ms | vs. FFTW (single-thread) |
| FFT (4096 complex f64) | ≤ 10 μs | vs. FFTW |
| Sparse GEMV (100K×100K, 1% density) | ≥ 500 MFLOPS | vs. SuiteSparse |
| Descriptive stats (1M f64) | ≤ 1 ms | vs. NumPy |
| Random normal sampling (1M f64) | ≤ 5 ms | vs. NumPy default_rng |

### 6.2 Binary Size

| Configuration | Target |
|--------------|--------|
| Single container (e.g., RBTree only) | < 50 KB stripped |
| Full library linked | < 2 MB stripped |
| Unused containers | Zero cost (dead code elimination by Zig) |

### 6.3 Correctness & Reliability

- **Property-based testing** using a custom fuzzer: generate random operation sequences and verify invariants after each operation.
- **Differential testing** against known-good implementations (C++ STL, Python stdlib) for algorithm correctness.
- **Fuzz testing** via `zig build fuzz` integration — target each container's public API.
- **Memory safety** verified by running all tests under `std.testing.allocator` (detects leaks, double-free, use-after-free).
- **Invariant checks** — every container has a `validate()` method that asserts internal invariants (BST property, heap property, balance factor, etc.).

### 6.4 Documentation Requirements

- Every public type and function has a doc comment.
- Doc comments include: one-line summary, time/space complexity, example usage, edge cases.
- Top-level module docs explain when to use which data structure (decision tree / comparison table).
- `examples/` directory contains runnable programs for each major container and algorithm.
- Algorithm explanations include references to original papers where applicable.

### 6.5 Compatibility

| Target | Support Level |
|--------|------|
| Zig 0.14.x+ | Primary (tested in CI) |
| Linux x86_64, aarch64 | Tier 1 |
| macOS x86_64, aarch64 | Tier 1 |
| Windows x86_64 | Tier 1 |
| WASI | Tier 2 |
| Bare-metal (no-std) | Tier 2 (bounded/fixed-capacity variants only) |
| FreeBSD | Tier 3 |

---

## 7. Development Roadmap

| Phase | Version | Summary | Status |
|-------|---------|---------|--------|
| **Phase 1** | v0.1.0 | Lists, queues, heaps, hash tables, CI, benchmark framework | ✅ |
| **Phase 2** | v0.5.0 | BSTs, tries, B-trees, range queries, spatial structures, suffix arrays | ✅ |
| **Phase 3** | — | Graph representations, traversal, shortest paths, MST, flow, matching | ✅ |
| **Phase 4** | — | Sorting, string algorithms, probabilistic, caches, math, geometry, DP | ✅ |
| **Phase 5** | v1.0.0 | Concurrent, persistent, exotic structures, C API, documentation | ✅ |

---

### Phase 6: NDArray Foundation (v2.0 Track)

**Goal:** Multi-dimensional array as the core data structure for scientific computing.

| Component | Description | Key Operations |
|-----------|------------|----------------|
| `NDArray(T, ndim)` | N-dimensional array with comptime-known rank | create, zeros, ones, full, arange, linspace |
| Shape & Stride | Row-major (C) and column-major (Fortran) memory layouts | shape, strides, ndim, size, itemsize |
| Indexing & Slicing | NumPy-style multi-dimensional indexing | `get(indices)`, `slice(ranges)`, `at(index)` |
| Reshape & Transpose | View-based transformations (zero-copy where possible) | reshape, transpose, permute, flatten, squeeze, unsqueeze |
| Broadcasting | NumPy-compatible broadcasting rules | Automatic shape expansion for binary operations |
| Element-wise Ops | Arithmetic, comparison, math functions | +, -, *, /, @(matmul), abs, exp, log, sin, cos, sqrt |
| Reduction Ops | Aggregation along axes | sum, prod, mean, min, max, argmin, argmax |
| Memory Views | Non-owning slices into existing arrays | view, contiguous, astype |
| I/O | Serialization and deserialization | fromSlice, toSlice, save (binary), load, fromCSV |

**Design principles:**
- `NDArray` accepts `std.mem.Allocator` (allocator-first, consistent with v1.x)
- Comptime-known rank (`NDArray(f64, 2)` for matrices) enables zero-overhead dimension checks
- Runtime-known shape within fixed rank
- Views and slices are non-owning (no copy unless explicit)
- Iterator protocol: `NDArrayIterator` yields elements in storage order

**Exit criteria:** NDArray passes 200+ tests including edge cases (empty, 0-dim scalar, high-rank). Broadcasting verified against NumPy reference outputs. Element-wise operations benchmarked against C loops.

### Phase 7: Linear Algebra

**Goal:** BLAS-level matrix operations and decomposition algorithms.

| Component | Description | Operations |
|-----------|------------|------------|
| **BLAS Level 1** | Vector-vector operations | dot, axpy, nrm2, asum, scal, swap |
| **BLAS Level 2** | Matrix-vector operations | gemv, trmv, trsv, ger, syr |
| **BLAS Level 3** | Matrix-matrix operations | gemm, trmm, trsm, syrk |
| **Decompositions** | Matrix factorizations | LU (partial pivoting), QR (Householder), Cholesky, SVD, Eigendecomposition (symmetric, general) |
| **Solvers** | Linear system solving | solve(A, b), lstsq (least squares), inv (matrix inverse) |
| **Matrix Properties** | Scalar properties of matrices | det, trace, rank, cond (condition number) |
| **Norms** | Vector and matrix norms | L1, L2, Linf, Frobenius |
| **Sparse Matrices** | Sparse storage and operations | CSR, CSC, COO formats; sparse-dense multiply; sparse solvers |

**Design principles:**
- All operations work on `NDArray(T, 2)` (matrices) and `NDArray(T, 1)` (vectors)
- Pure Zig implementations first; optional SIMD acceleration
- In-place variants available (`_inplace` suffix) to minimize allocations
- Sparse matrices share API surface with dense where applicable

**Exit criteria:** All decompositions verified against LAPACK reference outputs (tolerance ≤ 1e-10). GEMM benchmarked against OpenBLAS single-thread. Performance targets: Section 6.1 참조.

### Phase 8: Statistics & Random

**Goal:** Statistical computing primitives for data analysis.

| Component | Description | Functions |
|-----------|------------|-----------|
| **Descriptive Stats** | Summary statistics | mean, median, mode, std, var, quantile, percentile, skewness, kurtosis |
| **Distributions** | Probability distributions (PDF, CDF, quantile, sampling) | Normal, Uniform, Exponential, Poisson, Binomial, Gamma, Beta, Chi-squared, Student-t, F |
| **Hypothesis Testing** | Statistical tests | t-test (1-sample, 2-sample, paired), chi-square, ANOVA, Kolmogorov-Smirnov, Mann-Whitney U |
| **Correlation** | Association measures | Pearson, Spearman, Kendall tau, covariance matrix, cross-correlation |
| **Regression** | Linear models | OLS, polynomial fit, R-squared, residuals |
| **Random** | PRNG and sampling | PCG64, Xoshiro256**, seed, uniform, normal, shuffle, choice, multinomial |
| **Histogram** | Binning and frequency | histogram (uniform, auto), kernel density estimation |

**Design principles:**
- Functions operate on `NDArray(f64, 1)` (vectors) or `NDArray(f64, 2)` (matrices)
- Distributions are comptime-parameterized: `Normal(f64){ .mean = 0, .std = 1 }`
- Random uses explicit state (no global state): `var rng = zuda.random.PCG64.init(seed)`
- All statistical tests return a `TestResult{ .statistic, .p_value, .reject }`

**Exit criteria:** All distributions verified against SciPy reference (KS test, p > 0.05 on 10K samples). Hypothesis tests verified against R / SciPy known results. PRNG passes TestU01 SmallCrush.

### Phase 9: Transforms & Signal Processing

**Goal:** Frequency-domain analysis and signal processing primitives.

| Component | Description | Functions |
|-----------|------------|-----------|
| **FFT** | Fast Fourier Transform | fft, ifft, rfft, irfft, fft2, ifft2, fftfreq |
| **DCT** | Discrete Cosine Transform | dct, idct (Type II, III) |
| **Convolution** | Linear and circular convolution | convolve, correlate, fftconvolve |
| **Windowing** | Window functions | hamming, hann, blackman, kaiser, bartlett |
| **Filtering** | Digital filters | firwin, lfilter, filtfilt, butter, cheby1 |
| **Spectral Analysis** | Power spectrum estimation | periodogram, welch, spectrogram |

**Exit criteria:** FFT accuracy verified against DFT brute-force (tolerance ≤ 1e-10). Parseval's theorem verified for all transforms. Performance targets: Section 6.1 참조.

### Phase 10: Numerical Methods

**Goal:** Foundational numerical algorithms for scientific simulation.

| Component | Description | Functions |
|-----------|------------|-----------|
| **Integration** | Numerical quadrature | trapezoid, simpson, quad (adaptive), romberg, gauss_legendre |
| **Differentiation** | Numerical derivatives | diff (finite difference), gradient, jacobian, hessian |
| **Interpolation** | Function approximation | interp1d (linear), cubic_spline, lagrange, pchip, interp2d |
| **Root Finding** | Equation solving | bisect, newton, brent, secant, fixed_point |
| **ODE Solvers** | Ordinary differential equations | euler, rk4, rk45 (adaptive), bdf (stiff systems) |
| **Curve Fitting** | Parametric fitting | curve_fit (Levenberg-Marquardt), polyfit, polyval |
| **Special Functions** | Mathematical special functions | gamma, beta, erf, erfc, bessel_j, bessel_y |

**Exit criteria:** Integration accuracy verified against analytical solutions (tolerance ≤ 1e-12). ODE solvers tested on standard problems (Van der Pol, Lorenz, stiff decay). Interpolation verified against MATLAB interp1.

### Phase 11: Optimization

**Goal:** Mathematical optimization algorithms.

| Component | Description | Functions |
|-----------|------------|-----------|
| **Unconstrained** | Gradient-based optimization | gradient_descent, conjugate_gradient, lbfgs, bfgs, nelder_mead |
| **Constrained** | Constrained optimization | linear_programming (simplex), quadratic_programming, augmented_lagrangian |
| **Least Squares** | Nonlinear least squares | levenberg_marquardt, gauss_newton |
| **Auto-differentiation** | Forward-mode AD | Dual numbers, gradient computation, jacobian via AD |
| **Convex Optimization** | Convex problem solvers | proximal_gradient, admm, interior_point |
| **Line Search** | Step size selection | armijo, wolfe, backtracking |

**Design principles:**
- All optimizers accept a generic objective function: `fn(x: NDArray(f64, 1)) f64`
- Gradient functions are optional — finite differences used as fallback
- Forward-mode AD via dual numbers: `Dual(f64){ .real, .dual }` (comptime-propagated)
- Results returned as `OptimizeResult{ .x, .fun, .niter, .converged }`

**Exit criteria:** All optimizers converge on Rosenbrock, Rastrigin, and Beale test functions. Linear programming verified against GLPK/CBC. AD gradients match numerical derivatives (tolerance ≤ 1e-8).

### Phase 12: v2.0 Integration & Release

**Goal:** Polish, performance optimization, and unified release.

| Component | Description |
|-----------|------------|
| **SIMD Acceleration** | Vectorize GEMM, FFT, element-wise NDArray ops using Zig SIMD builtins |
| **NumPy Compatibility Layer** | API mapping guide: NumPy function → zuda equivalent |
| **Comprehensive Benchmarks** | Full benchmark suite comparing zuda vs NumPy/SciPy/Eigen/OpenBLAS |
| **Migration Guides** | Step-by-step guides for migrating from NumPy, Eigen, MATLAB |
| **Scientific Computing Guide** | Tutorial-style documentation covering all v2.0 modules |
| **Cross-platform Validation** | Numerical accuracy verified on x86_64, aarch64, WASM |
| **v2.0 Release** | Tagged release with full documentation, changelog, and migration notes |

**Exit criteria:**
- All Phase 6-11 modules integrated and tested together (cross-module tests)
- NDArray ↔ linalg ↔ stats ↔ signal interoperability verified
- Performance within 2× of specialized C libraries (OpenBLAS, FFTW) on key benchmarks
- NumPy compatibility guide covers 80%+ of NumPy's most-used functions
- Zero known correctness bugs; 100+ hours cumulative testing

---

## 8. Testing & Quality

- **Unit tests**: `zig build test` — 746+ passing, `std.testing.allocator` (메모리 누수 자동 감지)
- **Property tests**: 무작위 연산 시퀀스로 불변조건 검증
- **Differential tests**: C++/Python 레퍼런스 대비 출력 비교 (v2.0: LAPACK, SciPy, FFTW)
- **Benchmark regression**: CI에서 15% 이상 성능 저하 시 실패
- **v2.0 수치 안정성**: 분해 재구성 오차 ≤ 1e-10, 적분 오차 ≤ 1e-12

> CI 파이프라인, 메모리 테스트, 벤치마크 프레임워크 상세: [CLAUDE.md](../CLAUDE.md) 참조
> 설치 및 패키징: [README.md](../README.md) / [GETTING_STARTED.md](GETTING_STARTED.md) 참조

---

## 11. Future Considerations (Post v2.0)

Items explicitly deferred beyond the v2.0 roadmap:

- **GPU-accelerated computing** — GEMM, FFT, element-wise NDArray ops on GPU via Vulkan compute or WebGPU.
- **Multi-threaded linear algebra** — Thread-pool-based parallel GEMM, parallel FFT (currently single-threaded).
- **Async/io_uring integration** — Containers optimized for async contexts (e.g., async-friendly concurrent queues).
- **Formal verification** — Coq/Lean proofs for critical invariants (RBTree balance, heap property, numerical stability).
- **Language bindings** — First-class Python (`cffi`), Rust (`bindgen`), and Go (`cgo`) packages beyond raw C FFI.
- **Visualization tool** — CLI/web tool that renders container state and plots (matplotlib-like) for debugging and education.
- **Compression-aware structures** — Containers that operate directly on compressed data (e.g., compressed suffix arrays, FM-index).
- **Deep learning primitives** — Convolution layers, activation functions, backpropagation engine (post-v2.0 if demand exists).
- **Distributed computing** — MPI-style distributed NDArray operations for cluster computing.
- **Complex number first-class support** — `NDArray(Complex(f64), N)` with full linalg/FFT support.

---

## 12. Success Criteria

### v1.0 Success Criteria ✅ ACHIEVED

The project was considered successful at v1.0 when:

1. ✅ **Coverage** — All structures and algorithms from Phases 1–5 implemented and passing tests.
2. ✅ **Correctness** — Zero known correctness bugs; comprehensive testing with zero crashes.
3. ✅ **Performance** — Meets or exceeds all v1.x targets in Section 6.1.
4. ✅ **Usability** — A developer can add zuda to their project and use any container within 5 minutes.
5. ✅ **Code quality** — Zero known undefined behavior; 790/790 public APIs documented; 746 tests passing.
6. ✅ **Community readiness** — README, contributing guide, CI/CD, and cross-compilation in place.

### v2.0 Success Criteria

The project is considered successful at v2.0 when:

1. **Scientific computing coverage** — All modules from Phases 6–12 (NDArray, linalg, stats, signal, numeric, optimize) are implemented and passing tests.
2. **Numerical correctness** — All numerical results verified against reference implementations (LAPACK, FFTW, SciPy) with documented tolerances (≤ 1e-10 for decompositions, ≤ 1e-12 for integration).
3. **Performance** — Meets all v2.0 targets in Section 6.1. Within 2× of specialized C libraries (OpenBLAS, FFTW) on key benchmarks — acceptable given pure Zig with no external dependencies.
4. **NumPy parity** — Covers 80%+ of NumPy's 50 most-used functions. Migration guide maps every NumPy operation to a zuda equivalent.
5. **Integration** — All v2.0 modules interoperate seamlessly (e.g., `stats.correlation` accepts `NDArray`, `linalg.solve` returns `NDArray`, `signal.fft` works with `NDArray`).
6. **Cross-platform numerical stability** — Identical results (within floating-point tolerance) on x86_64, aarch64, and WASM targets.
7. **Documentation** — Scientific computing tutorial guide, API reference for all new modules, benchmark comparison tables vs NumPy/SciPy/Eigen.
8. **Test quality** — 1000+ tests across scientific computing modules; property-based tests for numerical invariants (e.g., A = LU, Q'Q = I, FFT(IFFT(x)) = x).

---

## Appendix A: `std` Overlap

zuda는 `std`를 대체하지 않는다. `std.ArrayList`, `std.HashMap`, `std.sort` 등은 재구현하지 않고, **대안적** 자료구조 (Cuckoo, Robin Hood, Swiss Table, Fibonacci Heap, TimSort 등)를 제공하여 보완한다.

---

## Appendix B: Reference Projects

**Data Structures & Algorithms (v1.x references):**

| Project | Language | Relevance |
|---------|----------|-----------|
| [Boost.Container](https://www.boost.org/doc/libs/release/doc/html/container.html) | C++ | Reference for container API breadth |
| [Boost.Graph](https://www.boost.org/doc/libs/release/libs/graph/) | C++ | Reference for generic graph algorithm design |
| [petgraph](https://github.com/petgraph/petgraph) | Rust | Reference for Rust-idiomatic graph library |
| [indexmap](https://github.com/indexmap-rs/indexmap) | Rust | Insertion-ordered hash map — API inspiration |
| [JGraphT](https://jgrapht.org) | Java | Most comprehensive graph library — scope reference |
| [Google Guava](https://github.com/google/guava) | Java | Reference for utility collection breadth |
| [TheAlgorithms/Zig](https://github.com/TheAlgorithms/Zig) | Zig | Educational Zig algorithms — avoid duplication of effort |
| [TigerBeetle](https://tigerbeetle.com) | Zig | Reference for production Zig patterns (allocators, testing) |
| [libstdc++ / libc++](https://gcc.gnu.org/onlinedocs/libstdc++/) | C++ | Reference for STL container semantics and guarantees |
| [LEDA](https://www.algorithmic-solutions.com/leda/) | C++ | Library of Efficient Data Types and Algorithms |

**Scientific Computing (v2.0 references):**

| Project | Language | Relevance |
|---------|----------|-----------|
| [NumPy](https://numpy.org) | Python | Primary API reference for NDArray design and function naming |
| [SciPy](https://scipy.org) | Python | Reference for linalg, signal, optimize, stats, integrate modules |
| [Eigen](https://eigen.tuxfamily.org) | C++ | Reference for template-based linear algebra API design |
| [OpenBLAS](https://www.openblas.net) | C/Fortran | Performance reference for BLAS operations |
| [LAPACK](https://www.netlib.org/lapack/) | Fortran | Reference implementation for decompositions and solvers |
| [FFTW](https://www.fftw.org) | C | Performance reference for FFT implementations |
| [ndarray](https://github.com/rust-ndarray/ndarray) | Rust | Reference for systems-language NDArray design |
| [nalgebra](https://nalgebra.org) | Rust | Reference for Rust linear algebra library design |
| [Armadillo](https://arma.sourceforge.net) | C++ | Reference for user-friendly linear algebra API |
| [GSL](https://www.gnu.org/software/gsl/) | C | GNU Scientific Library — scope reference for numerical methods |

## Appendix D: NumPy Compatibility Mapping (v2.0)

Core NumPy functions and their planned zuda equivalents:

| NumPy | zuda (v2.0) | Module |
|-------|-------------|--------|
| `np.array()` | `NDArray.fromSlice()` | ndarray |
| `np.zeros()`, `np.ones()`, `np.full()` | `NDArray.zeros()`, `.ones()`, `.full()` | ndarray |
| `np.arange()`, `np.linspace()` | `NDArray.arange()`, `.linspace()` | ndarray |
| `np.reshape()` | `ndarray.reshape()` | ndarray |
| `np.transpose()` | `ndarray.transpose()` | ndarray |
| `arr[i:j]`, `arr[i, j]` | `ndarray.slice()`, `.get()` | ndarray |
| `+`, `-`, `*`, `/`, `@` | `ndarray.add()`, `.sub()`, `.mul()`, `.div()`, `linalg.matmul()` | ndarray, linalg |
| `np.sum()`, `np.mean()`, `np.std()` | `ndarray.sum()`, `stats.mean()`, `stats.std()` | ndarray, stats |
| `np.dot()` | `linalg.dot()` | linalg |
| `np.linalg.solve()` | `linalg.solve()` | linalg |
| `np.linalg.eig()` | `linalg.eig()` | linalg |
| `np.linalg.svd()` | `linalg.svd()` | linalg |
| `np.linalg.inv()` | `linalg.inv()` | linalg |
| `np.linalg.det()` | `linalg.det()` | linalg |
| `np.linalg.norm()` | `linalg.norm()` | linalg |
| `np.fft.fft()` | `signal.fft()` | signal |
| `np.random.normal()` | `random.normal()` | stats |
| `scipy.optimize.minimize()` | `optimize.minimize()` | numeric |
| `scipy.interpolate.interp1d()` | `numeric.interp1d()` | numeric |
| `scipy.integrate.quad()` | `numeric.quad()` | numeric |
| `scipy.stats.ttest_ind()` | `stats.ttest_ind()` | stats |

---

## Appendix C: Glossary

| Term | Definition |
|------|-----------|
| **Comptime** | Zig's compile-time evaluation — allows type-level and value-level computation at compile time |
| **Allocator** | Zig's `std.mem.Allocator` interface — an abstraction over memory allocation strategies |
| **Unmanaged** | A container variant that does not store an allocator internally; the caller passes it per-call |
| **Bounded** | A container variant with a comptime-fixed maximum capacity, requiring no heap allocator |
| **Iterator Protocol** | The Zig convention where an iterator exposes `next() -> ?T`, consumed via `while (it.next()) \|v\|` |
| **Property-Based Testing** | Testing by generating random inputs and verifying that invariants hold, rather than testing specific cases |
| **Differential Testing** | Comparing the output of two implementations (e.g., zuda vs. C++) on the same inputs to find discrepancies |
| **SA-IS** | Suffix Array — Induced Sorting; a linear-time suffix array construction algorithm |
| **HAMT** | Hash Array Mapped Trie — a persistent hash map structure popularized by Clojure and Scala |
| **CSR** | Compressed Sparse Row — a compact graph/matrix representation for sparse data |
| **SCC** | Strongly Connected Components — maximal subgraphs where every node is reachable from every other |
| **NDArray** | N-dimensional Array — the core multi-dimensional data container, analogous to NumPy's `ndarray` |
| **BLAS** | Basic Linear Algebra Subprograms — standardized API for vector and matrix operations (Level 1-3) |
| **LAPACK** | Linear Algebra PACKage — reference implementations for matrix decompositions and solvers |
| **FFT** | Fast Fourier Transform — O(n log n) algorithm for computing the Discrete Fourier Transform |
| **GEMM** | General Matrix Multiply — the core BLAS Level 3 operation: C = αAB + βC |
| **Broadcasting** | NumPy-style rule for applying operations between arrays of different shapes by automatic expansion |
| **Stride** | The number of bytes between consecutive elements along each dimension of an NDArray |
| **View** | A non-owning reference to a subset of an NDArray's data — no copy, shares underlying memory |
| **Dual Number** | A number of the form a + bε (ε² = 0) used for forward-mode automatic differentiation |
| **GFLOPS** | Giga Floating-point Operations Per Second — performance metric for numerical computation |
