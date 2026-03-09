# zuda Project Context

## Current Status
- **Version**: 0.1.0
- **Phase**: Phase 3 — Graph Algorithms 🚧 **IN PROGRESS**
- **Zig Version**: 0.15.2
- **Last CI Status**: ✓ GREEN (231/231 tests passing - 100%)

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

## Implemented Data Structures
### Lists & Queues (Phase 1)
- **SkipList(K, V)** - Probabilistic balanced structure with O(log n) operations
- **XorLinkedList(T)** - Memory-efficient doubly linked list using XOR pointers
- **UnrolledLinkedList(T, N)** - Cache-friendly linked list with array nodes
- **Deque(T)** - Double-ended queue with O(1) amortized push/pop at both ends

### Hash Containers (Phase 1)
- **CuckooHashMap(K, V)** - O(1) worst-case lookup via two-table cuckoo hashing
- **RobinHoodHashMap(K, V)** - Open addressing with Robin Hood variance reduction
- **SwissTable(K, V)** - Group-based probing with control bytes
- **ConsistentHashRing(K)** - Virtual nodes for distributed hash ring

### Heaps (Phase 1)
- **FibonacciHeap(T)** - Lazy merge with O(1) insert/decrease-key amortized
- **BinomialHeap(T)** - Binomial tree forest with O(log n) merge
- **PairingHeap(T)** - Simple multi-pass pairing heap
- **DaryHeap(T, d)** - Generalized d-ary heap with comptime branching factor

### Trees & Range Queries (Phase 2)
- **BTree(K, V, order, Context)** - Self-balancing search tree with variable branching factor
- **RedBlackTree(K, V, Context, compareFn)** - Color-based balanced BST with O(log n) operations
- **AVLTree(K, V, Context, compareFn)** - Strictly balanced BST with height-based balancing (±1 balance factor)
- **SplayTree, AATree, ScapegoatTree** - Additional balanced BSTs with different rebalancing strategies
- **Trie, RadixTree** - Prefix-based string containers
- **SegmentTree, LazySegmentTree, FenwickTree, SparseTable, IntervalTree** - Range query structures
- **KDTree, RTree, QuadTree, OctTree** - Spatial index structures (2D/3D partitioning)
- **SuffixArray(T)** - Space-efficient suffix array with LCP, pattern matching, and longest repeated substring queries
- **SuffixTree(T)** - Compressed trie of all suffixes with pattern matching and LRS queries (basic O(n²) implementation)

## Phase 3 Progress — 🚧 IN PROGRESS
- [x] Graph Representations (1/4): AdjacencyList ✓
- [ ] Graph Representations (3/4): AdjacencyMatrix, CompressedSparseRow, EdgeList
- [ ] Traversal & shortest paths: BFS, DFS, Dijkstra, Bellman-Ford, A*, Floyd-Warshall, Johnson's
- [ ] MST & connectivity: Kruskal, Prim, Borůvka, Tarjan SCC, Kosaraju, bridges, articulation points
- [ ] Flow & matching: Edmonds-Karp, Dinic, Push-Relabel, Hopcroft-Karp, Hungarian, topological sort

## Implemented Data Structures - Phase 3
### Graph Representations (Phase 3)
- **AdjacencyList(V, W, Context, hashFn, eqlFn)** - Space-efficient graph using adjacency lists, O(V + E) space
  - Supports directed/undirected, weighted/unweighted graphs
  - O(1) amortized add vertex/edge, O(deg(v)) edge queries
  - Helper: IntGraph(W) for u32 vertices

## Implemented Algorithms
(none yet — Phase 3-4)

## Test Metrics
- Unit tests: 231 passing / 231 total (100%)
- Property tests: SkipList + heap invariants + tree validations
- Fuzz tests: 1
- Benchmarks: 0
- Known issues: None

## Recent Progress (Session 2026-03-09 - Hour 21)
**STABILIZATION MODE (hour % 4 == 0 branch):**
- ✅ Fixed SuffixTree bug #1: edge splitting and LRS logic (d17ca50)
  - Pattern search now correctly handles edge boundaries
  - LRS detection fixed for nodes with suffix_index AND children
- ✅ CI GREEN: All 219 tests passing

**FEATURE MODE (returned after bug fix):**
- ✅ Implemented AdjacencyList (dbed907)
  - Generic V vertex type, W weight type (void for unweighted)
  - Directed/undirected graph support
  - 12 comprehensive tests (all passing)
  - Consumer: Will replace zr's DAG implementation
- ✅ Phase 3 STARTED: 1/4 graph representations complete
- 🎯 Next: AdjacencyMatrix, then graph algorithms (BFS, DFS, etc.)
