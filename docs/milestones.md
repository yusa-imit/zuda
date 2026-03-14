# zuda — Milestones

## Current Status

- **Latest release**: No releases yet (pre-Phase 1)
- **Current phase**: Phase 1 — Foundations
- **Blockers**: None

---

## Active Milestones

### Phase 1 — Foundations (Weeks 1-8)
- [ ] Project scaffolding: CI, testing harness, benchmark framework
- [ ] **Lists & Queues**: `SkipList`, `XorLinkedList`, `UnrolledLinkedList`, `Deque`
- [ ] **Hash containers**: `CuckooHashMap`, `RobinHoodHashMap`, `SwissTable`, `ConsistentHashRing`
- [ ] **Heaps**: `FibonacciHeap`, `BinomialHeap`, `PairingHeap`, `DaryHeap`
- [ ] All Phase 1 containers pass invariant tests, fuzz tests (1hr minimum), and benchmarks

### Phase 2 — Trees & Range Queries (Weeks 9-16)
- [ ] **Balanced BSTs**: `RedBlackTree`, `AVLTree`, `SplayTree`, `AATree`, `ScapegoatTree`
- [ ] **Tries & B-Trees**: `Trie`, `RadixTree`, `BTree`
- [ ] **Range query**: `SegmentTree`, `LazySegmentTree`, `FenwickTree`, `SparseTable`, `IntervalTree`
- [ ] **Spatial**: `KDTree`, `RTree`, `QuadTree`, `OctTree`
- [ ] **Strings**: `SuffixArray`, `SuffixTree`

### Phase 3 — Graph Algorithms (Weeks 17-24)
- [ ] **Representations**: `AdjacencyList`, `AdjacencyMatrix`, `CompressedSparseRow`, `EdgeList`
- [ ] **Traversal & shortest paths**: BFS, DFS, Dijkstra, Bellman-Ford, A*, Floyd-Warshall, Johnson's
- [ ] **MST & connectivity**: Kruskal, Prim, Boruvka, Tarjan SCC, Kosaraju, bridges, articulation points
- [ ] **Flow & matching**: Edmonds-Karp, Dinic, Push-Relabel, Hopcroft-Karp, Hungarian, topological sort

### Phase 4 — Algorithms & Probabilistic (Weeks 25-34)
- [ ] **Sorting**: TimSort, IntroSort, RadixSort, CountingSort, BlockSort, in-place MergeSort
- [ ] **String algorithms**: KMP, Boyer-Moore, Rabin-Karp, Aho-Corasick, Z-algorithm
- [ ] **Probabilistic & cache**: `BloomFilter`, `CuckooFilter`, `CountMinSketch`, `HyperLogLog`, `LRUCache`, `LFUCache`
- [ ] **Math & geometry**: GCD, modexp, Miller-Rabin, convex hull, closest pair
- [ ] **DP utilities**: LIS, LCS, edit distance, knapsack, binary search variants

### Phase 5 — Advanced & Polish (Weeks 35-44)
- [ ] **Concurrent**: `LockFreeQueue`, `LockFreeStack`, `ConcurrentSkipList`, `ConcurrentHashMap`
- [ ] **Persistent**: `PersistentArray`, `PersistentRBTree`, `PersistentHashMap` (HAMT)
- [ ] **Exotic**: `DisjointSet`, `VanEmdeBoasTree`, `DancingLinks`, `Rope`, `BK-Tree`
- [ ] **C API & FFI**: C header generation, binding examples
- [ ] **Documentation & v1.0**: API reference, algorithm explainers, decision-tree guide

---

## Performance Targets

| Metric | Target |
|--------|--------|
| RedBlackTree insert | ≤ 200 ns/op (1M random keys) |
| RedBlackTree lookup | ≤ 150 ns/op (1M random keys) |
| BTree(128) range scan | ≥ 50M keys/sec (sequential) |
| FibonacciHeap decrease-key | ≤ 50 ns amortized |
| BloomFilter lookup | ≥ 100M ops/sec |
| Dijkstra (1M nodes, 5M edges) | ≤ 500 ms |
| TimSort (1M i64, random) | ≤ 10% overhead vs `std.sort` |
| Aho-Corasick (1000 patterns, 1MB text) | ≥ 500 MB/sec |

---

## Completed Milestones

No completed milestones yet.

---

## Milestone Establishment Process

미완료 마일스톤이 **2개 이하**가 되면, 에이전트가 자율적으로 새 마일스톤을 수립한다.

**입력 소스** (우선순위 순):
1. `gh issue list --state open --label feature-request` — 사용자 요청 기능
2. `docs/PRD.md` — 아직 구현되지 않은 PRD 항목
3. 기술 부채 — Known Limitations, TODO, 성능 병목
4. 소비자 프로젝트 요구사항 — Consumer Use Case Registry (CLAUDE.md) 참조

**수립 규칙**:
- 마일스톤 하나는 **단일 테마**로 구성 (여러 작은 기능을 하나의 주제로 묶음)
- 1-2주 내 완료 가능한 범위로 스코프 설정
- 버전 번호는 마지막 마일스톤의 다음 번호로 자동 부여
- 수립 후 이 파일에 추가하고 커밋: `chore: add milestone vX.Y.0`

---

## Dependency Tracking

zuda는 순수 라이브러리이므로 외부 의존성 마이그레이션은 없다.
소비자 프로젝트(zr, silica, zoltraak)로의 마이그레이션 발행은 CLAUDE.md의 **소비자 마이그레이션 발행 프로토콜**을 참조한다.
