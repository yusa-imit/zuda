# zuda Project Context

## Current Status
- **Version**: 0.1.0
- **Phase**: Phase 4 — Algorithms & Probabilistic (In Progress)
- **Zig Version**: 0.15.2
- **Last CI Status**: ✓ GREEN (458/458 tests passing - 100%)

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

## Phase 3 Progress — ✅ COMPLETE
- [x] Graph Representations (4/4): AdjacencyList, AdjacencyMatrix, CompressedSparseRow, EdgeList ✓
- [x] Traversal (2/2): BFS ✓, DFS ✓
- [x] DAG Algorithms (1/1): TopologicalSort (Kahn + DFS) ✓
- [x] Shortest paths (5/5): Dijkstra ✓, Bellman-Ford ✓, A* ✓, Floyd-Warshall ✓, Johnson ✓
- [x] MST (3/3): Kruskal ✓, Prim ✓, Borůvka ✓
- [x] Connectivity (4/4): Tarjan SCC ✓, Kosaraju SCC ✓, Bridges ✓, Articulation Points ✓
- [x] Flow & matching (5/5): Edmonds-Karp ✓, Dinic ✓, Push-Relabel ✓, Hopcroft-Karp ✓, Hungarian ✓

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
### Minimum Spanning Tree (MST)
- **Kruskal(V, W)** - Edge-centric MST via greedy edge addition, O(E log E)
  - Sort edges by weight, union-find for cycle detection
  - Works with edge list representation
- **Prim(V, W)** - Vertex-centric MST via priority queue, O(E log V)
  - Grows single tree from start vertex
  - Requires adjacency list representation
- **Boruvka(V, W)** - Parallel-friendly MST via component merging, O(E log V)
  - Oldest MST algorithm (1926), adds multiple edges per round
  - Each round: find cheapest outgoing edge for each component
  - Naturally parallelizable, at most log V rounds
### Connectivity
- **TarjanSCC(V)** - Strongly connected components via single DFS pass, O(V + E)
  - Discovery time + low-link value tracking
  - Stack discipline for component formation
  - Components in reverse topological order
  - Consumer: zr DAG cycle detection, silica deadlock detection
- **KosarajuSCC(V)** - Strongly connected components via two DFS passes, O(V + E)
  - Phase 1: DFS to compute finish times
  - Phase 2: Transpose graph (reverse edges)
  - Phase 3: DFS on transposed graph in decreasing finish order
  - Conceptually simpler than Tarjan, easier to parallelize
  - Components in topological order (opposite of Tarjan)
- **Bridges(V)** - Find bridges (cut edges) in undirected graph, O(V + E)
  - Bridge = edge whose removal increases connected components
  - DFS with discovery times and low-link tracking
  - Single-pass algorithm, handles disconnected components
  - Consumer: Network reliability (single point of failure), circuit design
- **ArticulationPoints(V)** - Find articulation points (cut vertices) in undirected graph, O(V + E)
  - Articulation point = vertex whose removal increases connected components
  - DFS with low-link values, root has ≥2 children rule
  - Single-pass algorithm, handles disconnected components
  - Consumer: Social network analysis (key influencers), transportation networks (critical hubs)
### Flow Algorithms
- **EdmondsKarp(V, C, Context)** - Max flow via Ford-Fulkerson with BFS, O(VE²)
  - BFS for shortest augmenting paths (guarantees polynomial time)
  - Returns max flow value, flow on each edge, and minimum cut
  - Consumer: network capacity planning, bipartite matching
- **Dinic(V, C, Context)** - Max flow via level graphs and blocking flows, O(V²E)
  - BFS to build level graph (distance layers from source)
  - DFS to find blocking flows with current-edge optimization
  - Unit capacity networks: O(E * min(V^(2/3), E^(1/2)))
  - Faster than Edmonds-Karp for many practical cases
  - Consumer: general purpose max flow, efficient alternative to Edmonds-Karp
- **PushRelabel(V, C, Context)** - Max flow via preflow-push with local operations, O(V³) basic, O(V²E) FIFO
  - Maintains preflow and height labels, uses push/relabel operations
  - Push: send excess flow to lower neighbor, Relabel: increase height
  - Height bound (2V) prevents infinite loops, current-edge optimization
  - Returns max flow, edge flows, and minimum cut
  - Consumer: network capacity planning, parallelizable max flow
- **HopcroftKarp(V, Context)** - Maximum cardinality bipartite matching, O(E * sqrt(V))
  - BFS to build layered graph, DFS to find vertex-disjoint augmenting paths
  - Alternating paths: unmatched-matched edges
  - Returns matching size, pair_u/pair_v bidirectional lookup
  - Consumer: job assignment, resource allocation
- **Hungarian(W)** - Optimal assignment (min-cost perfect matching), O(n³)
  - Kuhn-Munkres algorithm, primal-dual approach
  - Cost matrix reduction + BFS augmentation with slack adjustment
  - Returns total cost and assignment array
  - Consumer: task scheduling, bipartite matching with costs

## Phase 4 Progress — In Progress
- [x] **Sorting** (6/6): TimSort ✓, IntroSort ✓, RadixSort (LSD/MSD) ✓, CountingSort ✓, MergeSort (3 variants) ✓, BlockSort ✓
- [ ] **String algorithms** (2/5): KMP ✓, Boyer-Moore ✓, Rabin-Karp, Aho-Corasick, Z-algorithm
- [ ] **Probabilistic** (0/5): BloomFilter, CountMinSketch, HyperLogLog, CuckooFilter, MinHash
- [ ] **Cache** (0/3): LRUCache, LFUCache, ARCCache
- [ ] **Geometry** (0/4): Convex hull, Line intersection, Closest pair, Voronoi
- [ ] **DP Utilities** (0/4): LIS, LCS, Edit distance, Knapsack
- [ ] **Math** (0/6): GCD/LCM, Modexp, Miller-Rabin, Sieve, CRT, NTT

## Test Metrics
- Unit tests: 527 passing / 527 total (100%)
- Property tests: SkipList + heap invariants + tree validations
- Fuzz tests: 1
- Benchmarks: 0
- Known issues: None

## Recent Progress (Session 2026-03-11 - Hour 15)
**FEATURE MODE (hour % 4 == 3):**
- ✅ Implemented MergeSort family - three variants (53f6199)
  - **MergeSort**: Classic top-down recursive O(n log n) with O(n) space
  - **MergeSortBottomUp**: Iterative variant with better cache locality
  - **NaturalMergeSort**: Adaptive O(n) best case, exploits existing runs
  - All variants stable, generic over T with custom comparator
  - 20 tests passing: empty, sorted, reverse, duplicates, stability, stress (10k)
- ✅ Implemented BlockSort in-place stable sorting (77f0a9b)
  - O(n log n) time with O(1) space via block rotation
  - Triple-reverse rotation algorithm for in-place merging
  - Adaptive: detects natural runs, uses insertion sort for small runs
  - 14 tests passing: basic, edge cases, stability, rotation unit, stress (1k)
- ✅ **MILESTONE**: Phase 4 Sorting COMPLETE (6/6) ✓
  - All sorting algorithms implemented: TimSort, IntroSort, RadixSort, CountingSort, MergeSort (3), BlockSort
- ✅ Implemented KMP pattern matching (9e4a978)
  - O(n + m) linear time, failure function preprocessing
  - Generic over type T, find/findAll/contains/count methods
  - 18 tests passing: basic, overlapping matches, failure function validation
- ✅ Implemented Boyer-Moore pattern matching (b321fbe)
  - O(n / m) best case (sublinear), O(n * m) worst
  - Bad character + good suffix rules for efficient skipping
  - Right-to-left scanning, best for large alphabets
  - 17 tests passing: basic, long patterns, bad char skip demonstration
- ✅ CI GREEN: All 527 tests passing (100%)
- 🎯 Next: String algorithms (Rabin-Karp, Aho-Corasick, Z-algorithm)

## Previous Progress (Session 2026-03-11 - Hour 13)
**FEATURE MODE (hour % 4 == 1):**
- ✅ Implemented IntroSort hybrid sorting algorithm (0cf5274)
  - Quicksort + heapsort + insertion sort hybrid
  - O(n log n) worst case via depth limit (2 * log2(n))
  - Median-of-three pivot selection, switches to heapsort when depth exceeded
  - 12 tests passing: empty, sorted, reverse, random, large (triggers heapsort), strings, stress (10k)
- ✅ Implemented RadixSort (LSD/MSD) non-comparative sorting (453a330)
  - LSD: iterative byte-by-byte processing, O(d * (n + 256))
  - MSD: recursive with insertion sort for small subarrays
  - Handles signed integers via sign bit flipping
  - 20 tests passing: empty, sorted, large values, signed integers, u8/u64 types, stress (10k)
  - ⚠️ Fixed: u8 overflow in count array indices (byte + 1) → cast to usize first
- ✅ Implemented CountingSort linear-time sorting (960335f)
  - O(n + k) where k = max - min + 1
  - Stable via backwards iteration, supports negative integers
  - 17 tests passing: empty, sorted, signed, stability check, stress (10k)
  - ⚠️ Fixed: Integer overflow in range calculation (i8 range [-100, 100] = 200)
    - Solution: use wider integer type (2x bit size) for difference: WiderInt = Int(signedness, bitSize * 2)
- ✅ **MILESTONE**: Phase 4 Sorting 4/6 complete (TimSort, IntroSort, RadixSort, CountingSort)
- ✅ CI GREEN: All 458 tests passing (100%)
- 🎯 Next: Remaining sorting (MergeSort in-place, BlockSort) or String algorithms (KMP, Boyer-Moore, Aho-Corasick)

## Previous Progress (Session 2026-03-11 - Hour 09)
**FEATURE MODE (hour % 4 == 1):**
- ✅ Implemented Push-Relabel max flow algorithm (02a920b)
  - Preflow-push approach with local push/relabel operations
  - Algorithm: O(V³) basic, O(V²E) with FIFO selection
  - Height labeling function guides pushes, height bound (2V) prevents infinite loops
  - 6 tests passing - all max flow scenarios covered
- ✅ Implemented Hopcroft-Karp bipartite matching algorithm (253107e)
  - Maximum cardinality matching in bipartite graphs
  - Algorithm: O(E * sqrt(V)) via layered graphs and augmenting paths
  - BFS builds layered graph, DFS finds vertex-disjoint paths
  - 4 tests passing: simple, empty, complete K_{3,3}, asymmetric
- ✅ Implemented Hungarian optimal assignment algorithm (5ae8199)
  - Kuhn-Munkres for min-cost perfect matching
  - Algorithm: O(n³) primal-dual with cost reduction and slack adjustment
  - Handles both min-cost (direct) and max-weight (negated) problems
  - 5 tests passing: 2x2, 3x3, max-weight, empty, single element
  - ⚠️ Fixed: Integer overflow in cost adjustment - separate visited row/col logic
- ✅ **MILESTONE**: Phase 3 Flow & Matching COMPLETE (5/5) ✓
- ✅ **MILESTONE**: Phase 3 Graph Algorithms COMPLETE ✓
  - All 19 graph representations and algorithms implemented
  - 367 tests passing (100%)
- ✅ CI GREEN: All 367 tests passing (100%)
- 🎯 Next: Phase 4 - Algorithms & Probabilistic Structures (sorting, string algorithms, caching)

## Previous Session (Session 2026-03-11 - Hour 01)
**FEATURE MODE (hour % 4 == 1):**
- ✅ Implemented Bridges (cut edges) algorithm (cb832a5)
  - Tarjan's bridge-finding via DFS with low-link values
  - Algorithm: bridge detected when low[v] > discovery[u] (no back edge from v's subtree)
  - O(V + E) time, O(V) space
  - 11 tests: empty, simple bridge, triangle, square with diagonal, chain, cycle with tail, disconnected, complex, self-loop, stress (100 nodes)
  - Consumer: network reliability, circuit design, road network planning
- ✅ Implemented Articulation Points (cut vertices) algorithm (ab4b970)
  - Tarjan's articulation point finding via DFS with low-link values
  - Algorithm: root with ≥2 children OR non-root with child v where low[v] ≥ discovery[u]
  - O(V + E) time, O(V) space
  - 12 tests: empty, single vertex, simple chain, triangle, star, chain, cycle with tail, two cycles, disconnected, complex, self-loop, stress (100 nodes)
  - Consumer: social network analysis (key influencers), transportation (critical hubs)
- ✅ **MILESTONE**: Phase 3 Connectivity COMPLETE (4/4) ✓
  - Tarjan SCC, Kosaraju SCC, Bridges, Articulation Points all implemented
- ✅ CI GREEN: All 345 tests passing (100%)
- 🎯 Next: Flow & matching algorithms (Edmonds-Karp, Dinic, Push-Relabel, Hopcroft-Karp, Hungarian)

## Previous Session (Session 2026-03-10 - Hour 23)
**FEATURE MODE (hour % 4 == 3):**
- ✅ Implemented Borůvka's MST algorithm (0c5d13d)
  - Parallel-friendly MST via multiple edges per round
  - Algorithm: each round finds cheapest outgoing edge per component, adds all
  - O(E log V) time (at most log V rounds), O(V + E) space
  - 11 tests: triangle, disconnected, K4, parallel edges, weights, stress (100v), multi-round
  - Oldest MST algorithm (1926) - naturally parallelizable
- ✅ Implemented Tarjan's SCC algorithm (52a9586)
  - Single-pass DFS for strongly connected components
  - Algorithm: discovery time + low-link tracking + stack discipline
  - O(V + E) time, O(V) space
  - 10 tests: cycles, DAG, complex graph, self-loop, stress (100v chain/cycle)
  - Consumer: zr DAG cycle detection, silica deadlock detection
  - ⚠️ Fixed: ArrayList API changes in Zig 0.15 (`.{}` not `.init()`, `append(alloc, ...)`, `deinit(alloc)`)
  - ⚠️ Fixed: `pop()` returns `?T`, need `orelse unreachable` for unwrap
- ✅ **MILESTONE**: Phase 3 MST COMPLETE (3/3) ✓
  - Kruskal, Prim, Borůvka all implemented and tested
- ✅ Implemented Kosaraju's SCC algorithm (b9cec0e)
  - Two-pass DFS for strongly connected components
  - Algorithm: finish times → transpose graph → reverse DFS
  - O(V + E) time, O(V + E) space (transposed graph storage)
  - 11 tests: single vertex, cycles, DAG, complex graph, self-loop, stress (100v chain/cycle), edge case
  - Conceptually simpler than Tarjan, easier to parallelize
  - Components in topological order (opposite of Tarjan)
- ✅ CI GREEN: All 345 tests passing (100%)
- 🎯 Next: Connectivity algorithms (bridges, articulation points)
