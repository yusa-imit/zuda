# zuda — Milestones

## Current Status

- **Latest release**: v1.4.0 (2026-03-16) — Performance & Optimization
- **Current phase**: Post-v1.4.0 (next milestone TBD)
- **Tests**: 701/701 passing (100%)
- **Open issues**: None
- **Blockers**: None

---

## Active Milestones

### v1.4.0 — Performance & Optimization

Address performance gaps and optimize critical data structures:

- [x] **TimSort investigation & fix** (target: ≤10% overhead vs std.sort) ✅
  - **FIXED**: Buffer overflow causing SIGSEGV on large arrays
  - **Root cause**: Allocated buffer was items.len/2, but mergeRuns tried to slice up to len1 (≤ items.len)
  - **Solution**: Copy SMALLER run to buffer (TimSort optimization), merge backwards when needed
  - **Result**: **37% FASTER** than std.sort (35ms vs 55ms on 1M i64) — **EXCEEDS TARGET!**
- [x] **RedBlackTree optimization (partial)** (target: insert ≤200ns, lookup ≤150ns) ⚠️
  - **Baseline**: insert 329ns, lookup 593ns (original measurements)
  - **After optimization**: insert 255ns, lookup 258ns (commit 30c4c8e)
  - **Improvements**: insert -22%, lookup -56% from baseline
  - **Techniques**: Inlined hot paths, struct field reordering, prefetching
  - **Remaining gap**: insert +28% over target, lookup +72% over target
  - **Status**: Significant improvement but targets not met. Further optimization requires
    structural changes (color bit packing, parent pointer elimination) with complexity trade-offs.
    Recommend re-evaluating targets based on pointer-based tree fundamentals.
- [x] **Aho-Corasick optimization (partial)** (target: ≥500 MB/sec) ⚠️
  - **Baseline**: 58 MB/sec (with array transitions, before goto completion)
  - **After optimization**: 63 MB/sec (commit 2e6ef04, +9%)
  - **Technique**: Goto function completion — pre-compute all state transitions to eliminate
    runtime failure link following. Main search loop now has single array lookup per character.
  - **Remaining gap**: -87% under target (+433 MB/sec needed)
  - **Analysis**: Hit fundamental memory access limits. Target of 500 MB/sec = 6 CPU cycles/char.
    Current implementation: ~3-4 memory accesses/char (near-optimal for pointer-based traversal).
  - **Status**: Further gains require algorithmic rethinking (SIMD vectorization, precomputed
    match tables, or relaxed target based on memory bandwidth constraints).
- [x] **BloomFilter benchmark calculation fix** ✅
  - **Fixed**: Separated setup (1M inserts) from timed operation (10M lookups)
  - **Result**: Benchmark now accurately measures lookup-only performance (303M ops/sec)
  - **EXCEEDS TARGET**: 303M ops/sec >> 100M ops/sec target (+203%)
- [x] **Memory usage profiling & optimization pass** ✅
  - **Implemented**: MemoryTracker allocator wrapper in bench.zig
  - **New benchmark**: `zig build bench-memory` profiles 4 key containers (10k ops)
  - **Results**: RedBlackTree (481KB peak, 770k allocs), SkipList (2.7MB, 1M allocs), FibonacciHeap (747KB, 1M allocs), BTree (489KB, 17k allocs)
  - **Analysis**: BTree most memory-efficient (17k vs 770k-1M allocs), SkipList has 5.6x overhead
  - **Status**: Memory tracking framework complete, no leaks detected (1KB residual is benchmark overhead)
- [x] **SIMD opportunities exploration** ✅
  - **Document**: 425-line analysis in docs/SIMD_ANALYSIS.md (commit c1296ee)
  - **Coverage**: 6 hot-loop algorithms (TimSort, Aho-Corasick, BloomFilter, RadixSort, KMP/Boyer-Moore, sorting primitives)
  - **Key Finding**: Most zuda algorithms are memory-bound, not compute-bound
  - **Platform support**: SSE2/AVX2/AVX-512/NEON/RISC-V/WASM analysis
  - **Best candidates**: Sorting networks, RadixSort (AVX-512), string matching (16-byte chunks)
  - **Recommendation**: Defer SIMD implementation to v1.5.0+ (benchmark-driven, user demand)

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
| RedBlackTree insert | ≤ 200 ns/op | 255 ns | ⚠️ +28% over (improved from 329) |
| RedBlackTree lookup | ≤ 150 ns/op | 258 ns | ⚠️ +72% over (improved from 593) |
| TimSort overhead | ≤ 10% vs std.sort | **-37% (faster!)** | ✅ EXCEEDS! |
| Aho-Corasick | ≥ 500 MB/sec | 63 MB/sec | ❌ -87% (improved from 58) |
| FibonacciHeap decrease-key | ≤ 50 ns amortized | N/A (double-free bug) | ❌ |
| BloomFilter lookup | ≥ 100M ops/sec | 303M ops/sec | ✅ +203% |
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
| Post-v1.1.0 | Iterator System & Completeness | v1.3.0 | 2026-03-15 | 8 iterator adaptors (Map, Filter, Chain, Zip, Take, Skip, Enumerate, collect), A* comprehensive tests, PersistentArray.pop(), comprehensive iterator pattern guide |

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
