# zuda Project Context

## Current Status
- **Version**: 0.5.0 (released 2026-03-13)
- **Phase**: Phase 5 — Advanced & Polish (In Progress)
- **Zig Version**: 0.15.2
- **Last CI Status**: ✓ GREEN (701/701 tests passing - 100%)

## Phase 1 Progress — ✅ COMPLETE
- [x] Project scaffolding: CI, testing harness, benchmark framework
- [x] Lists & Queues: SkipList, XorLinkedList, UnrolledLinkedList, Deque
- [x] Hash containers: CuckooHashMap, RobinHoodHashMap, SwissTable, ConsistentHashRing
- [x] Heaps: FibonacciHeap, BinomialHeap, PairingHeap, DaryHeap

## Phase 2 Progress — ✅ COMPLETE
- [x] Balanced BSTs (5/5): RedBlackTree ✓, AVLTree ✓, SplayTree ✓, AATree ✓, ScapegoatTree ✓
- [x] Tries & B-Trees (3/3): BTree ✓, Trie ✓, RadixTree ✓
- [x] Range query (5/5): SegmentTree ✓, LazySegmentTree ✓, FenwickTree ✓, SparseTable ✓, IntervalTree ✓
- [x] Spatial (4/4): KDTree ✓, RTree ✓, QuadTree ✓, OctTree ✓
- [x] Strings (2/2): SuffixArray ✓, SuffixTree ✓ (all tests passing)

## Phase 3 Progress — ✅ COMPLETE
- [x] Graph Representations (4/4): AdjacencyList, AdjacencyMatrix, CompressedSparseRow, EdgeList ✓
- [x] Traversal (2/2): BFS ✓, DFS ✓
- [x] DAG Algorithms (1/1): TopologicalSort (Kahn + DFS) ✓
- [x] Shortest paths (5/5): Dijkstra ✓, Bellman-Ford ✓, A* ✓, Floyd-Warshall ✓, Johnson ✓
- [x] MST (3/3): Kruskal ✓, Prim ✓, Borůvka ✓
- [x] Connectivity (4/4): Tarjan SCC ✓, Kosaraju SCC ✓, Bridges ✓, Articulation Points ✓
- [x] Flow & matching (5/5): Edmonds-Karp ✓, Dinic ✓, Push-Relabel ✓, Hopcroft-Karp ✓, Hungarian ✓

## Phase 4 Progress — ✅ COMPLETE
- [x] **Sorting** (6/6): TimSort ✓, IntroSort ✓, RadixSort (LSD/MSD) ✓, CountingSort ✓, MergeSort (3 variants) ✓, BlockSort ✓
- [x] **String algorithms** (5/5): KMP ✓, Boyer-Moore ✓, Rabin-Karp ✓, Aho-Corasick ✓, Z-algorithm ✓
- [x] **Probabilistic** (5/5): BloomFilter ✓, CountMinSketch ✓, HyperLogLog ✓, CuckooFilter ✓, MinHash ✓
- [x] **Cache** (3/3): LRUCache ✓, LFUCache ✓, ARCCache ✓
- [x] **Geometry** (4/4): Convex hull (Graham, Jarvis) ✓, Closest pair ✓, Haversine ✓, Geohash ✓
- [x] **DP Utilities** (5/5): LIS ✓, LCS ✓, Edit distance ✓, Knapsack ✓, Binary search variants ✓
- [x] **Math** (6/6): GCD/LCM ✓, Modexp ✓, Miller-Rabin ✓, Sieve ✓, CRT ✓, NTT ✓

## Phase 5 Progress — In Progress
- [x] **Concurrent (4/4)**: WorkStealingDeque ✓, LockFreeQueue ✓, LockFreeStack ✓, ConcurrentSkipList ✓ — **COMPLETE**
- [x] **Persistent (3/3)**: PersistentArray ✓, PersistentHashMap ✓, PersistentRBTree ✓ — **COMPLETE**
- [x] **Exotic (5/5)**: DisjointSet ✓, Rope ✓, BK-Tree ✓, VanEmdeBoasTree ✓, DancingLinks ✓ — **COMPLETE**
- [x] **C API & FFI**: C header (zuda.h), Python bindings (ctypes), Node.js bindings (ffi-napi), FFI README — **COMPLETE**
- [ ] **Documentation & v1.0**: API reference, algorithm explainers, decision-tree guide

## Recent Progress (Session 2026-03-13 - Hour 21)
**FEATURE MODE (hour % 4 == 1):**
- ✅ Implemented C API and FFI bindings (92de839)
  - C API (src/ffi/c_api.zig): HashMap, SkipList, BloomFilter with opaque handles
  - C header (include/zuda.h): Complete API declarations for FFI consumers
  - Python bindings (examples/python_bindings.py): ctypes wrappers with auto memory management
  - Node.js bindings (examples/nodejs_bindings.js): ffi-napi wrappers for HashMap & BloomFilter
  - FFI README (examples/FFI_README.md): Usage examples, troubleshooting, extension guide
  - Build system: `-Dshared=true` option to generate libzuda.a + header install
  - Zig 0.15 compatibility: Updated to unmanaged ArrayList API (allocator-per-call)
- ✅ **MILESTONE**: Phase 5 C API & FFI COMPLETE ✓
- 📊 Test count: 701 total (100% passing)
- 🎯 Next: Documentation & v1.0 (final Phase 5 item)

## Test Metrics
- Unit tests: 701 passing / 701 total (100%)
- Property tests: SkipList + heap invariants + tree validations
- Fuzz tests: 1
- Benchmarks: 0
- Known issues: PersistentRBTree memory leak with concurrent versions (needs ref-counting)

## Known Issues
1. **PersistentRBTree**: Memory leak when multiple versions are kept alive concurrently
   - Root cause: Shared nodes between versions without reference counting
   - Workaround: Deinit old version immediately after creating new version
   - Proper fix: Implement reference counting or use arena allocator for version sets
   - Priority: Medium (functional but limited usage pattern)
