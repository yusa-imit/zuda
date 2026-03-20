# zuda

> **Z**ig **U**niversal **D**atastructures and **A**lgorithms

A comprehensive, production-ready library of data structures, algorithms, and scientific computing for Zig 0.15+. The Zig-native alternative to NumPy/SciPy.

[![CI](https://github.com/yusa-imit/zuda/actions/workflows/ci.yml/badge.svg)](https://github.com/yusa-imit/zuda/actions)
[![Zig](https://img.shields.io/badge/zig-0.15.x-orange)](https://ziglang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Features

### v1.x — Data Structures & Algorithms (Stable)
- **100+ Data Structures**: Lists, trees, graphs, heaps, spatial indexes, probabilistic structures
- **80+ Algorithms**: Sorting, graph algorithms, string matching, dynamic programming, geometry, math
- **746 Tests Passing**: Comprehensive test coverage including property-based and fuzz testing

### v2.0 — Scientific Computing (In Development)
- **NDArray**: N-dimensional array with broadcasting, slicing, and element-wise operations
- **Linear Algebra**: BLAS Level 1-3, LU/QR/SVD/Cholesky decompositions, linear solvers
- **Statistics**: Descriptive stats, probability distributions, hypothesis testing, regression
- **Signal Processing**: FFT/IFFT, convolution, digital filters, spectral analysis
- **Numerical Methods**: Integration, interpolation, ODE solvers, root finding
- **Optimization**: Gradient descent, L-BFGS, linear programming, auto-differentiation

### Core Design
- **Allocator-First**: Every container accepts `std.mem.Allocator` for full control
- **Comptime-Optimized**: Parameterize behavior at compile time for zero-overhead abstractions
- **C FFI**: Export to C, Python, Node.js, and other languages
- **Complete Documentation**: API reference, algorithm explainers, decision guides

---

## Quick Start

### Installation

Add to your `build.zig.zon`:

```zig
.{
    .name = "my-project",
    .version = "0.1.0",
    .dependencies = .{
        .zuda = .{
            .url = "https://github.com/yusa-imit/zuda/archive/refs/tags/v0.5.0.tar.gz",
            .hash = "1220...", // Get via `zig fetch <url>`
        },
    },
}
```

Update `build.zig`:

```zig
const zuda = b.dependency("zuda", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("zuda", zuda.module("zuda"));
```

### Example

```zig
const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Use a Red-Black Tree for ordered map
    const RBTree = zuda.containers.trees.RedBlackTree;
    fn cmp(_: void, a: i32, b: i32) std.math.Order {
        return std.math.order(a, b);
    }

    var map = RBTree(i32, []const u8, void, cmp).init(allocator);
    defer map.deinit();

    try map.insert(42, "answer");
    try map.insert(7, "lucky");

    // Iterate in sorted order
    var it = map.iterator();
    while (it.next()) |entry| {
        std.debug.print("{}: {s}\n", .{ entry.key, entry.value });
    }
}
```

**Output**:
```
7: lucky
42: answer
```

---

## v2.0 Preview — Scientific Computing

zuda v2.0 transforms the library into a comprehensive scientific computing platform. Here's a preview of the planned API:

```zig
const std = @import("std");
const zuda = @import("zuda");
const NDArray = zuda.ndarray.NDArray;
const linalg = zuda.linalg;
const stats = zuda.stats;

pub fn main() !void {
    const alloc = std.heap.page_allocator;

    // Create matrices
    var A = try NDArray(f64, 2).fromSlice(alloc, &.{ 3, 3 }, &.{
        1, 2, 3,
        4, 5, 6,
        7, 8, 10,
    });
    defer A.deinit();

    var b = try NDArray(f64, 1).fromSlice(alloc, &.{3}, &.{ 1, 2, 3 });
    defer b.deinit();

    // Solve Ax = b
    var x = try linalg.solve(alloc, A, b);
    defer x.deinit();

    // SVD decomposition
    var result = try linalg.svd(alloc, A);
    defer result.deinit();

    // Statistics
    const data = try NDArray(f64, 1).arange(alloc, 0, 1000, 1);
    defer data.deinit();
    const mu = stats.mean(data);
    const sigma = stats.std(data);
}
```

### NumPy vs zuda Comparison

| NumPy (Python) | zuda (Zig) |
|----------------|------------|
| `np.array([[1,2],[3,4]])` | `NDArray(f64, 2).fromSlice(alloc, &.{2,2}, &.{1,2,3,4})` |
| `np.zeros((3, 3))` | `NDArray(f64, 2).zeros(alloc, &.{3, 3})` |
| `A @ B` | `linalg.matmul(alloc, A, B)` |
| `np.linalg.solve(A, b)` | `linalg.solve(alloc, A, b)` |
| `np.fft.fft(x)` | `signal.fft(alloc, x)` |
| `np.mean(x)` | `stats.mean(x)` |
| `scipy.optimize.minimize(f, x0)` | `optimize.minimize(alloc, f, x0, .{})` |

**Why zuda over NumPy?**
- No GIL — true parallelism without workarounds
- No garbage collector — predictable latency
- Comptime generics — zero-overhead abstractions
- Single binary — no Python runtime dependency
- Explicit memory — allocator-first design, no hidden allocations
- C ABI — easy integration with any language

---

## What's Inside

### 📦 Containers

| Category | Structures |
|----------|------------|
| **Lists & Queues** | `SkipList`, `XorLinkedList`, `UnrolledLinkedList`, `Deque`, `LockFreeQueue`, `LockFreeStack`, `WorkStealingDeque` |
| **Hash Tables** | `CuckooHashMap`, `RobinHoodHashMap`, `SwissTable`, `ConsistentHashRing`, `PersistentHashMap` (HAMT) |
| **Heaps** | `FibonacciHeap`, `BinomialHeap`, `PairingHeap`, `DaryHeap`, `VanEmdeBoasTree` |
| **Trees** | `RedBlackTree`, `AVLTree`, `SplayTree`, `AATree`, `ScapegoatTree`, `BTree`, `Trie`, `RadixTree`, `SegmentTree`, `LazySegmentTree`, `FenwickTree`, `SparseTable`, `IntervalTree` |
| **Graphs** | `AdjacencyList`, `AdjacencyMatrix`, `CompressedSparseRow`, `EdgeList` |
| **Spatial** | `KDTree`, `RTree`, `QuadTree`, `OctTree` |
| **Strings** | `SuffixArray`, `SuffixTree`, `Rope`, `BKTree` |
| **Caches** | `LRUCache`, `LFUCache`, `ARCCache` |
| **Probabilistic** | `BloomFilter`, `CuckooFilter`, `CountMinSketch`, `HyperLogLog`, `MinHash` |
| **Persistent** | `PersistentArray`, `PersistentHashMap`, `PersistentRBTree` |
| **Specialized** | `DisjointSet`, `DancingLinks`, `ConcurrentSkipList` |

### 🧮 Algorithms

| Category | Algorithms |
|----------|------------|
| **Sorting** | TimSort, IntroSort, RadixSort, CountingSort, MergeSort (3 variants), BlockSort |
| **Graph** | BFS, DFS, Dijkstra, Bellman-Ford, A*, Floyd-Warshall, Johnson, Kruskal, Prim, Borůvka, Tarjan SCC, Kosaraju SCC, Bridges, Articulation Points, Topological Sort, Edmonds-Karp, Dinic, Push-Relabel, Hopcroft-Karp, Hungarian |
| **String** | KMP, Boyer-Moore, Rabin-Karp, Aho-Corasick, Z-algorithm |
| **DP** | LIS, LCS, Edit Distance, Knapsack, Binary Search variants |
| **Geometry** | Convex Hull (Graham, Jarvis), Closest Pair, Haversine, Geohash |
| **Math** | GCD/LCM, ModExp, Miller-Rabin, Sieve of Eratosthenes, CRT, NTT |

---

## Documentation

| Document | Description |
|----------|-------------|
| **[Getting Started](docs/GETTING_STARTED.md)** | Installation, examples, common patterns |
| **[API Reference](docs/API.md)** | Complete API documentation for all structures and algorithms |
| **[Algorithm Explainers](docs/ALGORITHMS.md)** | Conceptual guides for how algorithms work |
| **[Decision Guide](docs/GUIDE.md)** | Choose the right data structure for your use case |
| **[PRD](docs/PRD.md)** | Product requirements and development roadmap |

---

## Real-World Usage

zuda is designed to replace hand-rolled data structures in real projects:

| Project | Current Implementation | zuda Replacement | Status |
|---------|------------------------|------------------|--------|
| **[zr](https://github.com/yusa-imit/zr)** (task runner) | Custom DAG + topological sort (323 LOC) | `AdjacencyList` + `TopologicalSort` | Phase 3 ✅ |
| **[silica](https://github.com/yusa-imit/silica)** (RDBMS) | Custom B+Tree (4300 LOC), LRU cache (1237 LOC) | `BTree`, `LRUCache` | Phase 2/4 ✅ |
| **[zoltraak](https://github.com/yusa-imit/zoltraak)** (Redis-compatible server) | Custom sorted set, HyperLogLog | `SkipList`, `HyperLogLog`, `Geohash` | Phase 1/4 ✅ |

See [CLAUDE.md](CLAUDE.md#consumer-use-case-registry) for complete consumer registry.

---

## Performance

Selected benchmarks (see [API.md](docs/API.md#performance-targets-v050) for full targets):

| Operation | Target | Status |
|-----------|--------|--------|
| RedBlackTree insert (1M keys) | ≤ 200 ns/op | ✅ |
| RedBlackTree lookup (1M keys) | ≤ 150 ns/op | ✅ |
| FibonacciHeap decrease-key | ≤ 50 ns amortized | ✅ |
| BloomFilter lookup | ≥ 100M ops/sec | ✅ |

---

## Testing

```bash
# Run all 701 tests
zig build test

# Cross-compile for 6 targets
zig build -Dtarget=x86_64-linux-gnu
zig build -Dtarget=aarch64-linux-gnu
zig build -Dtarget=x86_64-macos
zig build -Dtarget=aarch64-macos
zig build -Dtarget=x86_64-windows
zig build -Dtarget=wasm32-wasi
```

---

## C FFI

Build shared library with C headers:

```bash
zig build -Dshared=true
```

Output:
- `zig-out/lib/libzuda.a`
- `zig-out/include/zuda.h`

**C Example**:
```c
#include <zuda.h>

ZudaHashMap* map = zuda_hash_map_create();
zuda_hash_map_put(map, "key", "value");
const char* val = zuda_hash_map_get(map, "key");
zuda_hash_map_destroy(map);
```

**Python Example**:
```python
from zuda import HashMap
hm = HashMap()
hm.put("key", "value")
print(hm.get("key"))  # "value"
```

See [examples/FFI_README.md](examples/FFI_README.md) for Python, Node.js, and other language bindings.

---

## Roadmap

### v1.x — Data Structures & Algorithms (Complete)
- [x] **Phase 1**: Lists, queues, heaps, hash tables
- [x] **Phase 2**: Trees, spatial structures, strings
- [x] **Phase 3**: Graph algorithms
- [x] **Phase 4**: Sorting, string algorithms, probabilistic, caches, geometry, math, DP
- [x] **Phase 5**: Concurrent, persistent, exotic, C API, FFI
- [x] **v1.0**: Documentation, decision guides, consumer migration support

### v2.0 — Scientific Computing (In Progress)
- [ ] **Phase 6**: NDArray foundation (multi-dimensional array, broadcasting, element-wise ops)
- [ ] **Phase 7**: Linear algebra (BLAS, decompositions, solvers, sparse matrices)
- [ ] **Phase 8**: Statistics & random (distributions, hypothesis testing, regression)
- [ ] **Phase 9**: Transforms & signal processing (FFT, convolution, filtering)
- [ ] **Phase 10**: Numerical methods (integration, interpolation, ODE solvers, root finding)
- [ ] **Phase 11**: Optimization (gradient descent, L-BFGS, linear programming, auto-diff)
- [ ] **Phase 12**: v2.0 integration, SIMD acceleration, NumPy compatibility guide

See [PRD.md](docs/PRD.md) for detailed roadmap and [milestones.md](docs/milestones.md) for progress tracking.

---

## Contributing

Contributions welcome! Please:

1. Check existing issues or create a new one
2. Fork and create a feature branch
3. Follow Zig coding conventions (see [CLAUDE.md](CLAUDE.md#coding-standards))
4. Add tests for new functionality
5. Ensure `zig build test` passes
6. Submit a pull request

**Bug Reports**: https://github.com/yusa-imit/zuda/issues

---

## Design Principles

1. **Allocator-First**: Every container accepts `std.mem.Allocator` - never hardcode allocator
2. **Comptime Configuration**: Parameterize behavior at compile time (comparators, hash functions, branching factors)
3. **Iterator Protocol**: All iterable containers expose `next() -> ?T`
4. **Complexity Contracts**: Every public function documents Big-O in doc comments
5. **Invariant Validation**: Every container provides `validate()` method
6. **No Panics**: Library code returns errors, caller decides
7. **Memory Safety**: Leak-free (verified with `std.testing.allocator`)

See [CLAUDE.md](CLAUDE.md#coding-standards) for complete coding standards.

---

## Version History

- **v1.14.0** (2026-03-20): Ergonomic enhancements — reverse iterators, convenience constructors, 112 new tests
- **v1.0.0** (2026-03-14): Stable release — 100+ data structures, 80+ algorithms, C FFI, 746 tests
- **v0.5.0** (2026-03-13): Phase 5 complete — C API, FFI bindings, persistent structures, exotic containers
- **v0.4.0**: Phase 4 complete — Probabilistic structures, caches, geometry, math, DP utilities
- **v0.3.0**: Phase 3 complete — Graph algorithms (shortest paths, MST, flow, matching)
- **v0.2.0**: Phase 2 complete — Trees, spatial structures, suffix arrays/trees
- **v0.1.0**: Phase 1 complete — Foundations (lists, queues, heaps, hash tables)

**Next**: v2.0.0 — Scientific computing platform (NDArray, linear algebra, statistics, FFT, numerical methods, optimization)

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Inspired by:
- **Zig Standard Library**: `std.ArrayList`, `std.HashMap`, `std.PriorityQueue`
- **Boost C++ Libraries**: Comprehensive STL-style containers
- **NumPy / SciPy**: NDArray design, scientific computing API surface
- **Eigen**: Template-based linear algebra design patterns
- **LAPACK / OpenBLAS / FFTW**: Performance reference implementations
- **CLRS**: *Introduction to Algorithms* (3rd edition)
- **Sedgewick**: *Algorithms* (4th edition)
- **Knuth**: *The Art of Computer Programming*
- **Google Abseil**: SwissTable implementation
- **Clojure**: Persistent data structures (HAMT)

Developed with **[Claude Code](https://github.com/anthropics/claude-code)** - autonomous AI development.

---

**Ready to build?** → [Getting Started](docs/GETTING_STARTED.md) | [API Reference](docs/API.md) | [Choose a Data Structure](docs/GUIDE.md)
