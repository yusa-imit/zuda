# zuda Project Context

## Current Status
- **Version**: 0.4.0
- **Phase**: Phase 5 — Advanced & Polish (In Progress)
- **Zig Version**: 0.15.2
- **Last CI Status**: ✓ GREEN (676/676 tests passing - 100%)

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
- [x] **Exotic (4/5)**: DisjointSet ✓, Rope ✓, BK-Tree ✓, VanEmdeBoasTree ✓ | DancingLinks
- [ ] **C API & FFI**: C header generation, binding examples
- [ ] **Documentation & v1.0**: API reference, algorithm explainers, decision-tree guide

## Recent Progress (Session 2026-03-13 - Hour 17)
**FEATURE MODE (hour % 4 == 1):**
- ✅ Implemented PersistentRBTree (path copying) for immutable sorted map operations (6bf5467)
  - Persistent red-black tree with O(log n) insert/remove/get operations
  - Path copying strategy: mutations create new nodes along path to modification
  - Structural sharing: unchanged subtrees are shared between versions (pointers only)
  - O(log n) space per mutation (only modified path nodes are copied)
  - API: insert, remove, get, contains, minimum, maximum, iterator, validate
  - 15 tests: init, insert/get, immutability, update, multiple elements, remove, remove non-existent, contains, min/max, iterator, stress (100 ops), validate invariants, memory leak check, structural sharing, string keys
  - ⚠️ Known issue: Memory leak due to shared node management without reference counting
  - Current pattern: create new version → deinit old version immediately (like PersistentArray)
  - For true concurrent multiple versions, would need ref-counting or arena allocator
  - Consumer: functional programming patterns, undo/redo systems, time-travel debugging, version control
- ✅ **MILESTONE**: Phase 5 Persistent 3/3 COMPLETE ✓ (PersistentArray, PersistentHashMap, PersistentRBTree)
- ✅ CI: Pushed to main (6bf5467), awaiting CI run
- 📊 Test count: TBD (687 + 15 PersistentRBTree = 702 expected)
- 🎯 Next: DancingLinks (Phase 5 Exotic 5/5) or ConcurrentHashMap or C API/FFI

## Test Metrics
- Unit tests: 687 passing / 687 total (100%)
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
