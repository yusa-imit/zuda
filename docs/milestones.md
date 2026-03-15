# zuda — Milestones

## Current Status

- **Latest release**: v1.1.0 (2026-03-15) — Performance Optimization
- **Current phase**: Consumer Migrations (v1.2.0)
- **Tests**: 701/701 passing (100%)
- **Open issues**: None
- **Blockers**: None

---

## Active Milestones

### v1.2.0 — Consumer Migrations

Validate zuda in production through consumer project adoption:

- [ ] zr migration (1,189 LOC replacement) — issues zr#21-#25 filed
- [ ] silica migration (7,000 LOC replacement) — issues silica#4, silica#5 filed
- [ ] zoltraak migration (3,435 LOC replacement) — issues zoltraak#1-#3 filed
- [ ] API refinements based on consumer feedback
- [ ] Migration guide documentation

---

## Performance Targets

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| BTree(128) range scan | ≥ 50M keys/sec | 83M keys/sec | ✅ +66% |
| RedBlackTree insert | ≤ 200 ns/op | 329 ns | ❌ +64% over |
| RedBlackTree lookup | ≤ 150 ns/op | 593 ns | ❌ +295% over |
| TimSort overhead | ≤ 10% vs std.sort | 176% overhead | ❌ 17x worse |
| Aho-Corasick | ≥ 500 MB/sec | 46 MB/sec | ❌ -91% |
| FibonacciHeap decrease-key | ≤ 50 ns amortized | N/A (double-free bug) | ❌ |
| BloomFilter lookup | ≥ 100M ops/sec | N/A (calculation bug) | ❌ |
| Dijkstra (1M nodes) | ≤ 500 ms | TBD | — |

---

## Completed Milestones

| Phase | Name | Release | Date | Summary |
|-------|------|---------|------|---------|
| Phase 1 | Foundations | v0.1.0 | 2026-03 | SkipList, CuckooHashMap, RobinHoodHashMap, SwissTable, ConsistentHashRing, FibonacciHeap, BinomialHeap, PairingHeap, DaryHeap, XorLinkedList, UnrolledLinkedList, Deque, CI, testing harness, benchmark framework |
| Phase 2 | Trees & Range Queries | v0.5.0 | 2026-03 | RedBlackTree, AVLTree, SplayTree, AATree, ScapegoatTree, Trie, RadixTree, BTree, SegmentTree, LazySegmentTree, FenwickTree, SparseTable, IntervalTree, KDTree, RTree, QuadTree, OctTree, SuffixArray, SuffixTree |
| Phase 3 | Graph Algorithms | — | 2026-03 | AdjacencyList, AdjacencyMatrix, CompressedSparseRow, EdgeList, BFS, DFS, Dijkstra, Bellman-Ford, A*, Floyd-Warshall, Johnson's, Kruskal, Prim, Boruvka, Tarjan SCC, Kosaraju, bridges, articulation points, Edmonds-Karp, Dinic, Push-Relabel, Hopcroft-Karp, Hungarian, topological sort |
| Phase 4 | Algorithms & Probabilistic | — | 2026-03 | TimSort, IntroSort, RadixSort, CountingSort, BlockSort, in-place MergeSort, KMP, Boyer-Moore, Rabin-Karp, Aho-Corasick, Z-algorithm, BloomFilter, CuckooFilter, CountMinSketch, HyperLogLog, LRUCache, LFUCache, GCD, modexp, Miller-Rabin, convex hull, closest pair, LIS, LCS, edit distance, knapsack |
| Phase 5 | Advanced & Polish | v1.0.0 | 2026-03-14 | LockFreeQueue, LockFreeStack, ConcurrentSkipList, ConcurrentHashMap, PersistentArray, PersistentRBTree, PersistentHashMap (HAMT), DisjointSet, VanEmdeBoasTree, DancingLinks, Rope, BK-Tree, C API, documentation, 213 public exports |
| Post-v1.0.0 | Performance Optimization | v1.1.0 | 2026-03-15 | Fixed FibonacciHeap deinit & insert API, TimSort allocation bug, BloomFilter benchmark; optimized Aho-Corasick build & search; analyzed RedBlackTree (near-optimal) |

### Post-v1.0.0 Activity

20 commits since v1.0.0 release:
- Benchmark API fixes for Zig 0.15.2 ArrayList changes
- Comprehensive benchmark suite for all PRD performance targets
- Glob algorithm implementation
- PersistentRBTree lifetime management documentation
- Consumer migration issue filing (zr, silica, zoltraak — 11,624 LOC total)

### Closed Issues

| # | Title | Closed |
|---|-------|--------|
| #6 | ArrayList API migration incomplete | 2026-03-13 |
| #5 | DancingLinks implementation | 2026-03-13 |
| #4 | CI warning confirmed | 2026-03-12 |
| #3 | CRITICAL: CI failing | 2026-03-12 |
| #2 | Completed phase not released | 2026-03-12 |
| #1 | SuffixTree edge splitting bug | 2026-03-09 |

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

### Consumer Migration Issues Filed

| Consumer | Issues | Total LOC | Status |
|----------|--------|-----------|--------|
| zr | #21, #22, #23, #24, #25 | 1,189 | Pending (blocked on v1.1.0) |
| silica | #4, #5 | 7,000 | Pending (blocked on v1.1.0) |
| zoltraak | #1, #2, #3 | 3,435 | Pending (blocked on v1.1.0) |
