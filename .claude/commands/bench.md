Run benchmarks for the zuda library.

Steps:
1. Build and run the benchmark suite
2. Compare results against PRD performance targets:
   - RedBlackTree insert: ≤ 200 ns/op
   - RedBlackTree lookup: ≤ 150 ns/op
   - BTree(128) range scan: ≥ 50M keys/sec
   - FibonacciHeap decrease-key: ≤ 50 ns amortized
   - BloomFilter lookup: ≥ 100M ops/sec
   - Dijkstra (1M nodes): ≤ 500 ms
   - TimSort (1M i64): ≤ 10% overhead vs std.sort
   - Aho-Corasick (1000 patterns, 1MB): ≥ 500 MB/sec
3. Report results with pass/fail against targets
4. If any regression > 15%, flag as WARNING

Optional: $ARGUMENTS
- If user specifies a component (e.g., "trees"), only bench that component
- If user says "baseline", save results as the new baseline
- If user says "compare", diff against saved baseline
