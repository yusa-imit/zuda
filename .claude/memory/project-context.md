# zuda Project Context

## Current Status
- **Version**: 0.1.0
- **Phase**: Phase 2 — Trees & Range Queries (STARTING)
- **Zig Version**: 0.15.2
- **Last CI Status**: ✓ GREEN (11/11 tests passing)

## Phase 1 Progress — ✅ COMPLETE
- [x] Project scaffolding: CI, testing harness, benchmark framework
- [x] Lists & Queues: SkipList, XorLinkedList, UnrolledLinkedList, Deque
- [x] Hash containers: CuckooHashMap, RobinHoodHashMap, SwissTable, ConsistentHashRing
- [x] Heaps: FibonacciHeap, BinomialHeap, PairingHeap, DaryHeap

## Phase 2 Progress — Trees & Range Queries
- [ ] Balanced BSTs: RedBlackTree, AVLTree, SplayTree, AATree, ScapegoatTree
- [x] Tries & B-Trees (1/3): BTree ✓
- [ ] Tries & B-Trees (remaining): Trie, RadixTree
- [ ] Range query: SegmentTree, LazySegmentTree, FenwickTree, SparseTable, IntervalTree
- [ ] Spatial: KDTree, RTree, QuadTree, OctTree
- [ ] Strings: SuffixArray, SuffixTree

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

## Implemented Algorithms
(none yet — Phase 3-4)

## Test Metrics
- Unit tests: 22 passing (11 BTree + 11 Phase 1)
- Property tests: SkipList + heap invariants + BTree sorted iteration
- Fuzz tests: 1
- Benchmarks: 0
