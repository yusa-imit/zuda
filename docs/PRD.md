# zuda — Product Requirements Document

> **Zig Universal Datastructures and Algorithms**
> A comprehensive, production-grade collection of data structures and algorithms written in idiomatic Zig.

**Version:** 0.1 (Draft)
**Date:** March 7, 2026
**Author:** Yusa

---

## 1. Overview

### 1.1 Vision

zuda is a batteries-included library of data structures and algorithms for the Zig programming language. While Zig's standard library intentionally ships a minimal set of containers (ArrayList, HashMap, LinkedList, PriorityQueue, BitSet, Treap), real-world projects frequently need structures like balanced BSTs, concurrent skip lists, spatial indices, graph algorithms, and advanced string matchers. Today, Zig developers either roll their own, wrap C libraries, or go without.

zuda fills this gap. It aims to be the **go-to DSA library for Zig** — the equivalent of Rust's `std::collections` + `petgraph` + `indexmap`, C++'s Boost.Container + Boost.Graph, or Java's Guava + JGraphT — but designed from the ground up to embrace Zig's philosophy: explicit allocators, comptime generics, zero hidden control flow, and C ABI interoperability.

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
| P1 | Comprehensive coverage — aim for the broadest DSA collection in any systems language ecosystem |
| P2 | First-class documentation with complexity annotations, usage examples, and algorithm explanations |
| P2 | C API layer for cross-language consumption |
| P3 | SIMD-accelerated paths for sorting, searching, and string algorithms where beneficial |
| P3 | WASM and embedded (no-std) support |

### 1.4 Non-Goals

- **Replacing `std`.** zuda complements the standard library; it does not reimplement ArrayList or HashMap. Where `std` already provides a good solution, zuda defers to it.
- **Becoming a math library.** Numerical computing, linear algebra, and FFT are out of scope (these deserve their own project).
- **Distributed algorithms.** Consensus protocols, CRDTs, and distributed hash tables are out of scope.

---

## 2. Target Users & Use Cases

### 2.1 Primary Users

- **Zig application developers** building systems that need more than what `std` provides (e.g., interval trees for a scheduler, tries for an autocomplete engine, graph algorithms for a dependency resolver).
- **Competitive programmers and algorithm enthusiasts** looking for a well-tested Zig reference implementation.
- **Systems programmers** porting C/C++ codebases to Zig who need drop-in replacements for STL / Boost containers.
- **Embedded developers** who need memory-predictable containers with fixed-capacity variants.

### 2.2 Key Use Cases

1. **Dependency resolution** — Topological sort, cycle detection, DAG shortest path.
2. **Spatial indexing** — R-Tree / k-d tree for game engines, GIS, or collision detection.
3. **Autocomplete / prefix matching** — Trie and radix tree with ranked results.
4. **Task scheduling** — Fibonacci heap for efficient decrease-key in Dijkstra's algorithm.
5. **Text processing** — Aho-Corasick multi-pattern search, suffix arrays for indexing.
6. **In-memory caching** — LRU / LFU eviction policies with O(1) operations.
7. **Network analysis** — Max-flow, strongly connected components, betweenness centrality.
8. **Interval queries** — Segment tree / Fenwick tree for range-sum and range-min queries.

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
│   ├── containers/               # Data Structures
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
│   ├── algorithms/               # Algorithms (operate on containers or slices)
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
│   └── internal/                 # Shared utilities (not public API)
│       ├── testing.zig           # Property-based test helpers
│       └── bench.zig             # Micro-benchmark harness
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

## 4. Data Structure Catalog

### 4.1 Phase 1 — Foundational Structures

| Category | Structure | Key Operations | Notes |
|----------|-----------|---------------|-------|
| **Lists** | `SkipList(K, V)` | insert, remove, search, range | Lock-free variant planned for Phase 4 |
| | `XorLinkedList(T)` | push, pop, iterate | Memory-efficient doubly-linked list |
| | `UnrolledLinkedList(T, N)` | insert, remove, iterate | Cache-friendly with comptime node size |
| **Trees** | `RedBlackTree(K, V)` | insert, remove, find, rank, select | Order-statistic augmentation |
| | `AVLTree(K, V)` | insert, remove, find, height | Stricter balance than RBT |
| | `BTree(K, V, order)` | insert, remove, search, range_scan | Disk-friendly, comptime order |
| | `Trie(V)` | insert, search, prefix_match, delete | Byte-keyed, compressed path |
| | `RadixTree(V)` | insert, search, longest_prefix | PATRICIA / compact trie |
| **Heaps** | `FibonacciHeap(T)` | insert, extract_min, decrease_key, merge | O(1) amortized insert & decrease-key |
| | `BinomialHeap(T)` | insert, extract_min, merge | Mergeable heap |
| | `PairingHeap(T)` | insert, extract_min, decrease_key, merge | Simpler than Fibonacci, competitive perf |
| | `DaryHeap(T, d)` | insert, extract_min | Comptime `d`; d=4 often faster than binary |
| **Hashing** | `CuckooHashMap(K, V)` | insert, get, remove | Worst-case O(1) lookup |
| | `RobinHoodHashMap(K, V)` | insert, get, remove | Low variance probe lengths |
| | `SwissTable(K, V)` | insert, get, remove | SIMD-accelerated probing |
| | `ConsistentHashRing(K)` | add_node, remove_node, get_node | Virtual nodes, configurable replicas |
| **Queues** | `Deque(T)` | push_front, push_back, pop_front, pop_back | Circular buffer backed |
| | `StealingQueue(T)` | push, pop, steal | Work-stealing for thread pools |

### 4.2 Phase 2 — Trees & Range Queries

| Category | Structure | Key Operations | Notes |
|----------|-----------|---------------|-------|
| **Range** | `SegmentTree(T, merge)` | build, query, update | Comptime merge function |
| | `LazySegmentTree(T, merge, apply)` | build, range_query, range_update | Lazy propagation |
| | `FenwickTree(T)` | update, prefix_sum, range_sum | aka Binary Indexed Tree |
| | `SparseTable(T, op)` | build, query | O(1) query, immutable |
| **Trees** | `SplayTree(K, V)` | insert, remove, find | Self-adjusting, amortized O(log n) |
| | `Treap(K, V)` | insert, remove, split, merge | Randomized BST (complements std) |
| | `ScapegoatTree(K, V)` | insert, remove, find | Weight-balanced, no extra metadata |
| | `AATree(K, V)` | insert, remove, find | Simplified red-black tree |
| | `IntervalTree(T)` | insert, remove, overlap_query | Augmented BST for intervals |
| | `KDTree(T, K)` | build, nearest, k_nearest, range | K-dimensional, comptime `K` |
| **Spatial** | `RTree(T, dims)` | insert, remove, search, nearest | R*-tree variant, comptime dimensions |
| | `QuadTree(T)` / `OctTree(T)` | insert, remove, query_region | 2D / 3D spatial partitioning |
| **Strings** | `SuffixArray` | build, search, lcp_array | SA-IS construction (linear time) |
| | `SuffixTree` | build, search, longest_repeated | Ukkonen's algorithm |

### 4.3 Phase 3 — Graph & Advanced Algorithms

| Category | Structure / Algorithm | Notes |
|----------|----------------------|-------|
| **Graph Repr.** | `AdjacencyList(V, E)` | Directed / undirected, weighted / unweighted |
| | `AdjacencyMatrix(V)` | Dense graphs, O(1) edge query |
| | `CompressedSparseRow(V, E)` | Immutable, cache-friendly for analytics |
| | `EdgeList(V, E)` | Minimal representation for Kruskal, etc. |
| **Traversal** | BFS, DFS, Iterative DFS | Generic over graph interface |
| **Shortest Path** | Dijkstra, Bellman-Ford, A*, Floyd-Warshall, Johnson's | Dijkstra uses zuda.FibonacciHeap |
| **MST** | Kruskal, Prim, Borůvka | |
| **Connectivity** | Tarjan (SCC), Kosaraju, Bridge detection, Articulation points | |
| **Flow** | Edmonds-Karp, Dinic, Push-Relabel | Max-flow / Min-cut |
| **Matching** | Hopcroft-Karp, Hungarian | Bipartite matching |
| **DAG** | Topological sort (Kahn / DFS), Longest path, Critical path | |
| **Cycles** | Cycle detection (directed / undirected), Eulerian path / circuit | |
| **Centrality** | Betweenness, Closeness, PageRank | |

### 4.4 Phase 4 — Algorithms & Probabilistic Structures

| Category | Algorithm / Structure | Notes |
|----------|----------------------|-------|
| **Sorting** | TimSort, IntroSort, RadixSort (LSD/MSD), CountingSort, MergeSort (in-place), BlockSort | All operate on `[]T` |
| **Searching** | Binary search variants, Interpolation search, Exponential search, Ternary search | |
| **String** | KMP, Rabin-Karp, Boyer-Moore, Aho-Corasick, Z-algorithm | Multi-pattern and single-pattern |
| **Geometry** | Convex hull (Graham, Andrew), Line intersection (Bentley-Ottmann), Closest pair, Voronoi | |
| **DP Utilities** | LIS, LCS, Edit distance, Knapsack solvers, Matrix chain multiplication | Reusable building blocks |
| **Math** | GCD/LCM, Modular exponentiation, Miller-Rabin primality, Sieve of Eratosthenes, CRT, NTT | |
| **Probabilistic** | `BloomFilter(T)` | Space-efficient membership test |
| | `CountMinSketch(T)` | Frequency estimation |
| | `HyperLogLog(T)` | Cardinality estimation |
| | `CuckooFilter(T)` | Bloom alternative with deletion support |
| | `MinHash(T)` | Jaccard similarity estimation |
| **Cache** | `LRUCache(K, V, cap)` | O(1) get/put with eviction |
| | `LFUCache(K, V, cap)` | Frequency-based eviction |
| | `ARCCache(K, V, cap)` | Adaptive replacement cache |
| **Concurrent** | `LockFreeQueue(T)` | Michael-Scott queue |
| | `LockFreeStack(T)` | Treiber stack |
| | `ConcurrentSkipList(K, V)` | Lock-free skip list |
| | `ConcurrentHashMap(K, V)` | Striped locking or lock-free |

### 4.5 Phase 5 — Extended & Exotic Structures

| Category | Structure | Notes |
|----------|-----------|-------|
| **Persistent** | `PersistentArray(T)` | Path-copied, O(log n) access |
| | `PersistentRedBlackTree(K, V)` | Functional / immutable BST |
| | `PersistentHashMap(K, V)` | HAMT (Hash Array Mapped Trie) |
| **Succinct** | `WaveletTree(T)` | Rank/Select/Access in compressed space |
| | `BitVector` with rank/select | Succinct index |
| | `FM-Index` | Compressed full-text index |
| **Specialized** | `DisjointSet(T)` (Union-Find) | Path compression + union by rank |
| | `VanEmdeBoasTree(u)` | O(log log u) operations for integer keys |
| | `DancingLinks` | Knuth's Algorithm X for exact cover |
| | `FusionTree(T)` | Theoretical O(log_w n) integer search |
| | `Link-Cut Tree` | Dynamic tree connectivity |
| | `Cartesian Tree(T)` | Min-heap ordered BST from sequence |
| | `Rope(T)` | Efficient string/sequence editing |
| | `BK-Tree(T, dist)` | Metric space search (spell checking) |

---

## 5. API Design

### 5.1 Naming Conventions

Follow Zig's standard library conventions:

- Types: `PascalCase` — `RedBlackTree`, `FibonacciHeap`
- Functions: `camelCase` — `insert`, `extractMin`, `nearestNeighbor`
- Constants: `snake_case` — `default_load_factor`, `max_branching_factor`
- Generic parameters: single uppercase or descriptive — `K`, `V`, `T`, `Context`

### 5.2 Error Handling

- Use Zig error unions (`!T`) for operations that can fail (allocation, capacity exceeded).
- Never panic on recoverable errors.
- Use `error.OutOfMemory` for allocation failures (standard Zig convention).
- Bounds-checked access uses `get(index)` returning `?T`; unchecked access uses `getUnchecked(index)` marked as `@setRuntimeSafety(false)` in release modes.

### 5.3 Generic Container Template

Every container follows this structural pattern:

```zig
pub fn RedBlackTree(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: K, b: K) std.math.Order,
) type {
    return struct {
        const Self = @This();

        pub const Entry = struct { key: K, value: V };
        pub const Iterator = struct { ... };

        // -- Lifecycle --
        pub fn init(allocator: std.mem.Allocator) Self { ... }
        pub fn deinit(self: *Self) void { ... }
        pub fn clone(self: *const Self) !Self { ... }

        // -- Capacity --
        pub fn count(self: *const Self) usize { ... }
        pub fn isEmpty(self: *const Self) bool { ... }

        // -- Modification --
        /// Time: O(log n) | Space: O(1) amortized
        pub fn insert(self: *Self, key: K, value: V) !?V { ... }
        /// Time: O(log n) | Space: O(1)
        pub fn remove(self: *Self, key: K) ?Entry { ... }

        // -- Lookup --
        /// Time: O(log n) | Space: O(1)
        pub fn get(self: *const Self, key: K) ?V { ... }
        pub fn contains(self: *const Self, key: K) bool { ... }

        // -- Range --
        /// Returns an iterator over entries in [low, high].
        /// Time: O(log n + k) where k = number of results
        pub fn range(self: *const Self, low: K, high: K) Iterator { ... }

        // -- Order Statistics --
        /// Time: O(log n) | Space: O(1)
        pub fn rank(self: *const Self, key: K) usize { ... }
        pub fn select(self: *const Self, i: usize) ?Entry { ... }

        // -- Iteration --
        pub fn iterator(self: *const Self) Iterator { ... }

        // -- Bulk --
        pub fn fromSlice(allocator: std.mem.Allocator, items: []const Entry) !Self { ... }
        pub fn toSlice(self: *const Self, allocator: std.mem.Allocator) ![]Entry { ... }

        // -- Debug --
        pub fn format(self: *const Self, ...) !void { ... }
        pub fn validate(self: *const Self) !void { ... } // BST invariant check
    };
}
```

### 5.4 Graph Interface

Graph algorithms are generic over a `Graph` concept (duck-typed via comptime):

```zig
pub fn dijkstra(
    comptime G: type,
    graph: *const G,
    source: G.NodeId,
    allocator: std.mem.Allocator,
) !ShortestPaths(G.NodeId, G.Weight) {
    // G must provide:
    //   .neighbors(node) -> Iterator over { .target: NodeId, .weight: Weight }
    //   .nodeCount() -> usize
    ...
}

// Works with any conforming graph representation:
const adj = zuda.AdjacencyList(u32, f64).init(allocator);
const paths = try zuda.graph.dijkstra(@TypeOf(adj), &adj, 0, allocator);

const csr = zuda.CompressedSparseRow(u32, f64).fromEdges(allocator, edges);
const paths2 = try zuda.graph.dijkstra(@TypeOf(csr), &csr, 0, allocator);
```

### 5.5 C API Layer

A thin C wrapper generated via `@export`:

```c
#include "zuda.h"

zuda_rbtree *tree = zuda_rbtree_create(sizeof(int64_t), int64_compare);
zuda_rbtree_insert(tree, &key, &value);

int64_t *result = (int64_t *)zuda_rbtree_get(tree, &key);

zuda_rbtree_destroy(tree);
```

---

## 6. Non-Functional Requirements

### 6.1 Performance Targets

| Metric | Target | Benchmark |
|--------|--------|-----------|
| RedBlackTree insert | ≤ 200 ns/op (1M random keys) | vs. `std.Treap`, C++ `std::map` |
| RedBlackTree lookup | ≤ 150 ns/op (1M random keys) | vs. `std.Treap`, C++ `std::map` |
| BTree(128) range scan | ≥ 50M keys/sec (sequential) | vs. SQLite B-Tree, LMDB |
| FibonacciHeap decrease-key | ≤ 50 ns amortized | vs. binary heap extract+reinsert |
| BloomFilter lookup | ≥ 100M ops/sec | vs. C reference (libbloom) |
| Dijkstra (1M nodes, 5M edges) | ≤ 500 ms | vs. Boost.Graph, igraph |
| TimSort (1M i64, random) | competitive with `std.sort` | ≤ 10% overhead |
| Aho-Corasick (1000 patterns, 1MB text) | ≥ 200 MB/sec throughput (standard impl) | vs. Rust aho-corasick |

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

### Phase 1: Foundations (Weeks 1–8)

**Goal:** Core containers, project infrastructure, benchmark harness.

| Week | Milestone | Deliverables |
|------|-----------|-------------|
| 1–2 | Project scaffolding | `build.zig`, CI (GitHub Actions), testing harness, benchmark framework, `README.md`, contributing guide |
| 3–4 | Lists & Queues | `SkipList`, `XorLinkedList`, `UnrolledLinkedList`, `Deque` — with full tests |
| 5–6 | Hash containers | `CuckooHashMap`, `RobinHoodHashMap`, `SwissTable`, `ConsistentHashRing` |
| 7–8 | Heaps | `FibonacciHeap`, `BinomialHeap`, `PairingHeap`, `DaryHeap` |

**Exit criteria:** All Phase 1 containers pass invariant tests, fuzz tests (1hr minimum), and beat or match C++ equivalents in benchmarks.

### Phase 2: Trees & Range Queries (Weeks 9–16)

| Week | Milestone | Deliverables |
|------|-----------|-------------|
| 9–10 | Balanced BSTs | `RedBlackTree`, `AVLTree`, `SplayTree`, `AATree`, `ScapegoatTree` |
| 11–12 | Tries & B-Trees | `Trie`, `RadixTree`, `BTree` |
| 13–14 | Range query structures | `SegmentTree`, `LazySegmentTree`, `FenwickTree`, `SparseTable`, `IntervalTree` |
| 15–16 | Spatial structures | `KDTree`, `RTree`, `QuadTree`, `OctTree` |

**Exit criteria:** Order-statistic operations on RBTree verified against brute-force. Range query structures verified against naive O(n) scans. Spatial queries tested with randomized point clouds.

### Phase 3: Graph Algorithms (Weeks 17–24)

| Week | Milestone | Deliverables |
|------|-----------|-------------|
| 17–18 | Graph representations | `AdjacencyList`, `AdjacencyMatrix`, `CompressedSparseRow`, `EdgeList`, generic `Graph` interface |
| 19–20 | Traversal & shortest paths | BFS, DFS, Dijkstra, Bellman-Ford, A*, Floyd-Warshall, Johnson's |
| 21–22 | MST & connectivity | Kruskal, Prim, Borůvka, Tarjan SCC, Kosaraju, bridges, articulation points |
| 23–24 | Flow & matching | Edmonds-Karp, Dinic, Push-Relabel, Hopcroft-Karp, Hungarian, topological sort |

**Exit criteria:** All graph algorithms verified against known results on standard benchmark graphs (SNAP datasets, DIMACS). Dijkstra benchmarked against Boost.Graph.

### Phase 4: Algorithms & Probabilistic (Weeks 25–34)

| Week | Milestone | Deliverables |
|------|-----------|-------------|
| 25–26 | Sorting algorithms | TimSort, IntroSort, RadixSort, CountingSort, BlockSort, in-place MergeSort |
| 27–28 | String algorithms | KMP, Boyer-Moore, Rabin-Karp, Aho-Corasick, Z-algorithm, suffix array (SA-IS), suffix tree |
| 29–30 | Probabilistic & cache | `BloomFilter`, `CuckooFilter`, `CountMinSketch`, `HyperLogLog`, `MinHash`, `LRUCache`, `LFUCache`, `ARCCache` |
| 31–32 | Math & geometry | GCD, modexp, Miller-Rabin, sieve, NTT, convex hull, closest pair, line intersection |
| 33–34 | DP utilities & search | LIS, LCS, edit distance, knapsack, binary search variants, interpolation search |

**Exit criteria:** Sorting benchmarked against `std.sort` and C `qsort`. String algorithms tested on real-world corpora (English text, DNA sequences). Probabilistic structures verified for false positive rate within theoretical bounds.

### Phase 5: Advanced & Polish (Weeks 35–44)

| Week | Milestone | Deliverables |
|------|-----------|-------------|
| 35–36 | Concurrent structures | `LockFreeQueue`, `LockFreeStack`, `ConcurrentSkipList`, `ConcurrentHashMap`, `StealingQueue` |
| 37–38 | Persistent structures | `PersistentArray`, `PersistentRBTree`, `PersistentHashMap` (HAMT) |
| 39–40 | Exotic structures | `DisjointSet`, `VanEmdeBoasTree`, `DancingLinks`, `Rope`, `BK-Tree`, `Link-Cut Tree`, `WaveletTree` |
| 41–42 | C API & FFI | C header generation, Python/Node.js binding examples, pkg-config support |
| 43–44 | Documentation & release | API reference generation, algorithm explainer docs, decision-tree guide ("which container should I use?"), v1.0 release |

**Exit criteria:** C API usable from Python ctypes example. Full documentation coverage. All benchmarks published. v1.0 tagged.

---

## 8. Testing Strategy

### 8.1 Test Pyramid

| Level | Scope | Quantity | Run Time |
|-------|-------|----------|----------|
| **Unit tests** | Per-function correctness | ~5,000+ | < 30 sec |
| **Property tests** | Random operation sequences per container | ~200 scenarios × 10K ops each | < 5 min |
| **Fuzz tests** | AFL-style mutation on serialized ops | Continuous | 1+ hour / container |
| **Differential tests** | Compare output against C++/Python reference | ~50 per algorithm | < 2 min |
| **Benchmark regression** | Detect performance regressions | ~100 benchmarks | < 10 min |
| **Integration tests** | End-to-end scenarios (e.g., build graph → run Dijkstra → verify) | ~30 | < 1 min |

### 8.2 CI Pipeline

```
push / PR → build (debug + release) → unit tests → property tests → fuzz (30 min) → benchmarks → docs build
                                                                                          ↓
                                                                                  compare against baseline
                                                                                  (fail if > 15% regression)
```

### 8.3 Memory Testing

All tests run under `std.testing.allocator` which:
- Tracks every allocation and free.
- Fails the test on memory leak.
- Detects double-free.
- Optionally fails after N allocations to test error paths (`std.testing.FailingAllocator`).

---

## 9. Benchmark Framework

### 9.1 Design

```zig
const bench = @import("zuda").bench;

pub fn main() !void {
    var b = bench.Runner.init(.{
        .warmup_iterations = 100,
        .min_iterations = 1000,
        .max_time_ns = 5 * std.time.ns_per_s,
    });

    try b.add("RBTree insert 1M random", struct {
        fn run(state: *bench.State) void {
            var tree = zuda.RedBlackTree(i64, void, ...).init(state.allocator);
            defer tree.deinit();
            while (state.next()) {
                tree.insert(state.random.int(i64), {}) catch unreachable;
            }
        }
    }.run);

    try b.run();
    try b.report(.{ .format = .markdown }); // outputs comparison table
}
```

### 9.2 Output Format

```
Benchmark                         |   Time/op |  Allocs/op | vs std.Treap | vs C++ std::map
----------------------------------|-----------|------------|--------------|----------------
RBTree insert (1M random i64)     |   187 ns  |     1.0    |    -12%      |     -8%
RBTree lookup (1M random i64)     |   142 ns  |     0.0    |    -15%      |     -5%
BTree(128) seq scan (1M)          |    18 ns  |     0.0    |      —       |    -22%
FibHeap decrease-key              |    38 ns  |     0.0    |      —       |    -31%
```

---

## 10. Packaging & Distribution

### 10.1 Zig Package Manager

```zig
// Consumer's build.zig.zon
.dependencies = .{
    .zuda = .{
        .url = "https://github.com/yusa/zuda/archive/v1.0.0.tar.gz",
        .hash = "...",
    },
},

// Consumer's build.zig
const zuda_dep = b.dependency("zuda", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("zuda", zuda_dep.module("zuda"));
```

### 10.2 Selective Import

Users can import only what they need — Zig's dead code elimination ensures unused containers add zero overhead:

```zig
const rbt = @import("zuda").containers.trees.RedBlackTree;
const dijkstra = @import("zuda").algorithms.graph.dijkstra;
```

### 10.3 C Library Build

```bash
zig build -Dtarget=x86_64-linux -Doptimize=ReleaseFast -Dc-api=true
# Produces: libzuda.a, libzuda.so, zuda.h
```

---

## 11. Future Considerations (Post v1.0)

Items explicitly deferred beyond the initial roadmap:

- **GPU-accelerated algorithms** — Sorting, graph BFS on GPU via Vulkan compute or OpenCL.
- **Async/io_uring integration** — Containers optimized for async contexts (e.g., async-friendly concurrent queues).
- **Formal verification** — Coq/Lean proofs for critical invariants (RBTree balance, heap property).
- **Language bindings** — First-class Python (`cffi`), Rust (`bindgen`), and Go (`cgo`) packages beyond raw C FFI.
- **Visualization tool** — CLI/web tool that renders container state as diagrams for debugging and education.
- **Compression-aware structures** — Containers that operate directly on compressed data (e.g., compressed suffix arrays, FM-index).

---

## 12. Success Criteria

The project is considered successful at v1.0 when:

1. **Coverage** — All structures and algorithms from Phases 1–4 are implemented and passing tests.
2. **Correctness** — Zero known correctness bugs; 72+ hours cumulative fuzz testing with zero crashes.
3. **Performance** — Meets or exceeds all targets in Section 6.1 on reference hardware (AMD Ryzen 7 / Apple M2, 16 GB RAM).
4. **Usability** — A developer can add zuda to their project and use any container within 5 minutes using the documentation.
5. **Code quality** — Zero known undefined behavior; all public APIs documented; test coverage > 80% by line.
6. **Community readiness** — README, contributing guide, issue templates, and CI are in place for open-source collaboration.

---

## Appendix A: Zig `std` Overlap Analysis

zuda intentionally avoids reimplementing these `std` containers:

| `std` Container | zuda Stance |
|----------------|-------------|
| `std.ArrayList` | Not reimplemented. zuda uses it internally. |
| `std.HashMap` / `AutoHashMap` | Not reimplemented. zuda offers *alternative* hash maps (Cuckoo, Robin Hood, Swiss Table) with different trade-offs. |
| `std.SinglyLinkedList` / `DoublyLinkedList` | Not reimplemented. zuda offers *alternative* linked lists (Xor, Unrolled) with different trade-offs. |
| `std.PriorityQueue` / `PriorityDequeue` | Not reimplemented. zuda offers *alternative* heaps (Fibonacci, Binomial, Pairing, D-ary). |
| `std.Treap` | Not reimplemented, but zuda's `Treap` variant adds split/merge and implicit key support for rope-like use. |
| `std.bit_set` | Not reimplemented. zuda's `BitVector` adds rank/select for succinct data structures. |
| `std.sort` | Not reimplemented. zuda offers *additional* sorts (TimSort, RadixSort, etc.) that complement `std.sort`. |

---

## Appendix B: Reference Projects

| Project | Language | Relevance |
|---------|----------|-----------|
| [Boost.Container](https://www.boost.org/doc/libs/release/doc/html/container.html) | C++ | Reference for container API breadth |
| [Boost.Graph](https://www.boost.org/doc/libs/release/libs/graph/) | C++ | Reference for generic graph algorithm design |
| [petgraph](https://github.com/petgraph/petgraph) | Rust | Reference for Rust-idiomatic graph library |
| [indexmap](https://github.com/indexmap-rs/indexmap) | Rust | Insertion-ordered hash map — API inspiration |
| [JGraphT](https://jgrapht.org) | Java | Most comprehensive graph library — scope reference |
| [Google Guava](https://github.com/google/guava) | Java | Reference for utility collection breadth |
| [TheAlgorithms/Zig](https://github.com/TheAlgorithms/Zig) | Zig | Educational Zig algorithms — avoid duplication of effort, focus on production quality |
| [TigerBeetle](https://tigerbeetle.com) | Zig | Reference for production Zig patterns (allocators, testing) |
| [libstdc++ / libc++](https://gcc.gnu.org/onlinedocs/libstdc++/) | C++ | Reference for STL container semantics and guarantees |
| [LEDA](https://www.algorithmic-solutions.com/leda/) | C++ | Library of Efficient Data Types and Algorithms — naming inspiration |

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
| **CSR** | Compressed Sparse Row — a compact graph representation for static graphs |
| **SCC** | Strongly Connected Components — maximal subgraphs where every node is reachable from every other |
