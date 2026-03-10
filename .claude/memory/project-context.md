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
- [x] Graph Representations (4/4): AdjacencyList, AdjacencyMatrix, CompressedSparseRow, EdgeList ✓
- [x] Traversal (2/2): BFS ✓, DFS ✓
- [x] DAG Algorithms (1/1): TopologicalSort (Kahn + DFS) ✓
- [x] Shortest paths (5/5): Dijkstra ✓, Bellman-Ford ✓, A* ✓, Floyd-Warshall ✓, Johnson ✓
- [ ] MST & connectivity: Kruskal, Prim, Borůvka, Tarjan SCC, Kosaraju, bridges, articulation points
- [ ] Flow & matching: Edmonds-Karp, Dinic, Push-Relabel, Hopcroft-Karp, Hungarian

## Implemented Data Structures - Phase 3
### Graph Representations (Phase 3)
- **AdjacencyList(V, W, Context, hashFn, eqlFn)** - Space-efficient graph using adjacency lists, O(V + E) space
  - Supports directed/undirected, weighted/unweighted graphs
  - O(1) amortized add vertex/edge, O(deg(v)) edge queries
  - Helper: IntGraph(W) for u32 vertices
- **AdjacencyMatrix(V)** - Dense graph representation, O(1) edge query, O(V²) space
- **CompressedSparseRow(V, W)** - Immutable, cache-friendly format for analytics
- **EdgeList(V, W)** - Minimal representation for edge-centric algorithms (Kruskal, etc.)

## Implemented Algorithms - Phase 3
### Graph Traversal
- **BFS(V, Context)** - Breadth-first search with shortest path computation
- **DFS(V, Context)** - Depth-first search with pre/post-order traversal
### DAG Algorithms
- **TopologicalSort(V, Context)** - Kahn's algorithm + DFS-based topological ordering
  - Consumer: zr task runner (replaces custom topo_sort.zig)
  - Cycle detection with vertex reporting
### Shortest Paths
- **Dijkstra(V, W, Context)** - Single-source shortest paths for non-negative weights, O((V+E) log V)
- **BellmanFord(V, W, Context)** - Single-source shortest paths with negative weights, O(V*E)
  - Handles negative edge weights correctly
  - Detects and reports negative cycles
  - Path reconstruction via parent pointers
- **AStar(V, W, Context)** - Heuristic-guided shortest path, O(E) worst case, typically O(b^d)
  - Uses admissible heuristic function for efficient pathfinding
  - Early termination when goal is reached
  - Ideal for navigation, routing, game AI pathfinding
- **FloydWarshall(V, W, Context)** - All-pairs shortest paths, O(V³) time, O(V²) space
  - Computes shortest distances between all vertex pairs
  - Handles negative edge weights (detects negative cycles)
  - Distance and parent matrices for path reconstruction
  - Best for small-medium graphs (V < ~500) needing complete distance information
- **Johnson(V, W, Context)** - All-pairs shortest paths via reweighting, O(V²log V + VE)
  - More efficient than Floyd-Warshall for sparse graphs (E << V²)
  - Combines Bellman-Ford (reweighting) + Dijkstra (all sources)
  - Handles negative weights, detects negative cycles
  - Path reconstruction support, best for sparse graphs

## Test Metrics
- Unit tests: 286 passing / 286 total (100%)
- Property tests: SkipList + heap invariants + tree validations
- Fuzz tests: 1
- Benchmarks: 0
- Known issues: None

## Recent Progress (Session 2026-03-10 - Hour 17)
**FEATURE MODE (hour % 4 == 1):**
- ✅ Implemented Johnson's algorithm (98fbe19)
  - All-pairs shortest paths for sparse graphs via reweighting technique
  - Algorithm: temp source + Bellman-Ford reweighting + Dijkstra from each vertex
  - O(V²log V + VE) - better than Floyd-Warshall O(V³) for sparse graphs
  - Handles negative weights, detects negative cycles
  - 8 comprehensive tests: positive weights, negative weights, cycle detection, disconnected, single vertex, path reconstruction, complex validation, sparse characteristics
  - ⚠️ Fixed: `var` vs `const` for mutable HashMap results (deinit requires *Self)
  - ⚠️ Fixed: Optional unwrapping for parent map (??V -> ?V)
- ✅ **MILESTONE**: Phase 3 Shortest Paths COMPLETE (5/5) ✓
  - Dijkstra, Bellman-Ford, A*, Floyd-Warshall, Johnson all implemented and tested
- ✅ CI GREEN: All 286 tests passing (100%)
- 🎯 Next: MST algorithms (Kruskal, Prim, Borůvka)
