# zuda Architecture

## Module Organization

```
zuda (root.zig — re-exports all public types)
├── containers/
│   ├── lists/          Sequential containers (SkipList, XorLinkedList, UnrolledLinkedList)
│   ├── trees/          Tree-based containers (RedBlackTree, AVLTree, BTree, Trie, ...)
│   ├── heaps/          Heap variants (FibonacciHeap, BinomialHeap, PairingHeap, DaryHeap)
│   ├── hashing/        Hash containers (CuckooHashMap, RobinHoodHashMap, SwissTable)
│   ├── queues/         Queue variants (Deque, StealingQueue)
│   ├── graphs/         Graph representations (AdjacencyList, AdjacencyMatrix, CSR)
│   ├── strings/        String structures (SuffixArray, SuffixTree, DoubleArrayTrie)
│   ├── spatial/        Spatial indices (KDTree, RTree, QuadTree, OctTree)
│   └── probabilistic/  Probabilistic (BloomFilter, CountMinSketch, HyperLogLog)
├── algorithms/
│   ├── sorting/        Sorting algorithms on []T slices
│   ├── searching/      Search algorithms
│   ├── graph/          Graph algorithms (generic over Graph concept)
│   ├── string/         String matching algorithms
│   ├── math/           Number theory, combinatorics
│   ├── geometry/       Computational geometry
│   └── dynamic_programming/
├── iterators/          Composable iterator adaptors
└── internal/           testing.zig, bench.zig (not public)
```

## Key Design Decisions

### Allocator-First Pattern
Every heap-allocating container accepts `std.mem.Allocator`. Managed variants store the allocator; Unmanaged variants require it per-call. Matches `std.ArrayListUnmanaged` pattern.

### Comptime Generics
All type parameterization happens at comptime. Comparators, hash functions, branching factors — no vtables, no runtime dispatch. Full monomorphization.

### Iterator Protocol
Standard `next() -> ?T` pattern compatible with `while (iter.next()) |item|`. Iterators are lazy and composable.

### Graph Algorithm Interface
Graph algorithms are generic over a duck-typed `Graph` concept via comptime. Any type providing `.neighbors()` and `.nodeCount()` works.

### Complexity Contracts
Every public function's doc comment states Big-O time and space. Verified via benchmark regression tests.

### Double-Array Trie Implementation (v1.8.0)
The DoubleArrayTrie(T) uses Aoe (1989) algorithm for space-efficient pattern storage:
- BASE[state] = transition base address or next unallocated state ID
- CHECK[pos] = parent state verification (0xFFFFFFFF marks empty slots)
- is_leaf[state] = separate array tracking pattern endings (no BASE negation)
- Current implementation uses naive 256-slot allocation per state for simplicity
- Future optimization: bitmap-based conflict resolution for 50-100× memory reduction
- Trade-off: Construction O(|V| × |Σ|) vs search O(1) per character

### SIMD BLAS Auto-Dispatch Architecture (v2.0.x — Session 496+)
Level 2 BLAS routines (gemv, trmv, trsv) include SIMD-optimized implementations with auto-dispatch:

**File organization**:
- `src/linalg/blas.zig` — Public API, contains scalar implementations + dispatch logic
- `src/linalg/simd_blas.zig` — SIMD-optimized implementations (_simd suffix)
- Import: `const simd_blas = @import("simd_blas.zig");` in blas.zig

**Dispatch pattern** (threshold: n >= 64):
```zig
pub fn trmv(...) !void {
    // ... validation ...
    const n = A.shape[0];
    if (n >= 64) {
        return try simd_blas.trmv_simd(...);  // SIMD path
    }
    // ... scalar fallback for n < 64 ...
}
```

**Implementations** (Session 496-497):
1. gemv_simd (Session 481) — Matrix-vector multiply: y = αAx + βy
2. trmv_simd (Session 496) — Triangular matrix-vector multiply: x = Ax
3. trsv_simd (Session 497) — Triangular solve: Ax = b or A^T*x = b
4. ger_simd — Rank-1 update: A += αxy^T

**SIMD vectorization mechanics**:
- Vector width: 4 for f64, 8 for f32 (256-bit SIMD, AVX/AVX2)
- Main loop: Process `vec_width` elements using `@Vector(width, T)` + `@reduce(.Add, ...)`
- Tail loop: Scalar for `n % vec_width` remainder
- Temporary buffers: Used to preserve input during in-place operations (e.g., trsv, trmv)

**Performance characteristics**:
- Threshold n=64 chosen empirically (SIMD overhead ≈ speedup at this point)
- Expected speedup: 2-4× over scalar for n >= 256
- Memory overhead: O(n) for temporary buffers
- Small matrices (n < 64): No performance regression (scalar path has lower overhead)
