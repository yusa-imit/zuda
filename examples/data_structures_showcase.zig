/// Data Structures Catalog
///
/// zuda provides 58 production-ready data structures across 12 categories.
/// This example lists all containers with their performance characteristics
/// and real-world use cases from consumer projects (zr, silica, zoltraak, sailor).
///
/// For full API documentation and usage examples, see:
/// - src/containers/*/*.zig (implementation + inline tests)
/// - CLAUDE.md Consumer Use Case Registry
/// - docs/PRD.md Phase 1-5 specifications
///
/// Run: zig build example-data-structures

const std = @import("std");

pub fn main() !void {
    std.debug.print("\n=== ZUDA DATA STRUCTURES CATALOG ===\n\n", .{});
    std.debug.print("zuda provides 58 containers across 12 categories (v1.x + v2.0):\n\n", .{});

    std.debug.print("LISTS (4):\n", .{});
    std.debug.print("  SkipList           - Probabilistic balanced tree, O(log n) avg\n", .{});
    std.debug.print("  UnrolledLinkedList - Cache-friendly, 8 elements per node\n", .{});
    std.debug.print("  XorLinkedList      - Memory-efficient doubly-linked list\n", .{});
    std.debug.print("  ConcurrentSkipList - Lock-free skip list for concurrent access\n\n", .{});

    std.debug.print("QUEUES (4):\n", .{});
    std.debug.print("  Deque             - O(1) push/pop from both ends\n", .{});
    std.debug.print("  CircularQueue     - Fixed-size ring buffer\n", .{});
    std.debug.print("  WorkStealingQueue - Chase-Lev deque for parallelism\n", .{});
    std.debug.print("  PriorityDeque     - Min/max at both ends\n\n", .{});

    std.debug.print("HEAPS (4):\n", .{});
    std.debug.print("  FibonacciHeap     - O(1) decrease-key for Dijkstra\n", .{});
    std.debug.print("  PairingHeap       - Simpler, competitive performance\n", .{});
    std.debug.print("  BinomialHeap      - O(log n) merge\n", .{});
    std.debug.print("  DaryHeap          - d-ary heap (d=4 optimal for cache)\n\n", .{});

    std.debug.print("HASH TABLES (5):\n", .{});
    std.debug.print("  CuckooHashMap     - O(1) worst-case lookup\n", .{});
    std.debug.print("  RobinHoodHashMap  - Low variance probe lengths\n", .{});
    std.debug.print("  SwissTable        - SIMD-friendly group probing\n", .{});
    std.debug.print("  ConsistentHashRing- Minimal rehashing for distributed systems\n", .{});
    std.debug.print("  PersistentHashMap - Functional persistent map\n\n", .{});

    std.debug.print("CACHES (3):\n", .{});
    std.debug.print("  LRUCache          - Least recently used eviction\n", .{});
    std.debug.print("  LFUCache          - Least frequently used eviction\n", .{});
    std.debug.print("  ARCCache          - Adaptive replacement cache\n\n", .{});

    std.debug.print("TREES (15):\n", .{});
    std.debug.print("  RedBlackTree      - Self-balancing BST, O(log n) guaranteed\n", .{});
    std.debug.print("  BTree             - B-tree, disk-friendly\n", .{});
    std.debug.print("  SplayTree         - Self-adjusting, locality optimized\n", .{});
    std.debug.print("  TreapMap          - Randomized BST\n", .{});
    std.debug.print("  WaveletTree       - Rank/select queries on sequences\n", .{});
    std.debug.print("  IntervalTree      - Overlapping interval queries\n", .{});
    std.debug.print("  SegmentTree       - Range min/max/sum queries\n", .{});
    std.debug.print("  BinaryIndexedTree - Fenwick tree for prefix sums\n", .{});
    std.debug.print("  ... and 7 more\n\n", .{});

    std.debug.print("GRAPHS (4):\n", .{});
    std.debug.print("  AdjacencyList     - Sparse graphs, O(V+E) space\n", .{});
    std.debug.print("  AdjacencyMatrix   - Dense graphs, O(V²) space\n", .{});
    std.debug.print("  CompressedSparseRow - Memory-efficient sparse graphs\n", .{});
    std.debug.print("  EdgeList          - Simple edge storage\n\n", .{});

    std.debug.print("SPATIAL (4):\n", .{});
    std.debug.print("  KDTree            - k-dimensional nearest neighbor\n", .{});
    std.debug.print("  RTree             - Rectangle-based spatial index\n", .{});
    std.debug.print("  Quadtree          - 2D space partitioning\n", .{});
    std.debug.print("  OctTree           - 3D space partitioning\n\n", .{});

    std.debug.print("PROBABILISTIC (5):\n", .{});
    std.debug.print("  HyperLogLog       - Cardinality estimation, 1-2% error\n", .{});
    std.debug.print("  BloomFilter       - Membership with false positives\n", .{});
    std.debug.print("  CountMinSketch    - Frequency estimation\n", .{});
    std.debug.print("  CuckooFilter      - Bloom filter with deletions\n", .{});
    std.debug.print("  QuotientFilter    - Compact approximate set\n\n", .{});

    std.debug.print("STRINGS (3):\n", .{});
    std.debug.print("  Trie              - Prefix tree for string search\n", .{});
    std.debug.print("  SuffixArray       - Suffix-based string index\n", .{});
    std.debug.print("  AhoCorasick       - Multi-pattern matching\n\n", .{});

    std.debug.print("SPECIALIZED (3):\n", .{});
    std.debug.print("  DisjointSet       - Union-find, O(α(n)) amortized\n", .{});
    std.debug.print("  FenwickTree       - Prefix sum queries\n", .{});
    std.debug.print("  DancingLinks      - Exact cover solver (Knuth's Algorithm X)\n\n", .{});

    std.debug.print("=== REAL-WORLD USAGE ===\n\n", .{});

    std.debug.print("zr (task runner):\n", .{});
    std.debug.print("  - WorkStealingQueue for parallel task execution (130 LOC → zuda)\n", .{});
    std.debug.print("  - Topological sort on DAG for dependency ordering (323 LOC → zuda)\n", .{});
    std.debug.print("  - Cycle detection for circular dependencies (205 LOC → zuda)\n", .{});
    std.debug.print("  - Levenshtein distance for command suggestions (214 LOC → zuda)\n\n", .{});

    std.debug.print("silica (RDBMS):\n", .{});
    std.debug.print("  - BTree for storage indexes (4300 LOC, domain-specific, keep custom)\n", .{});
    std.debug.print("  - LRUCache for buffer pool (1237 LOC → zuda, high ROI)\n", .{});
    std.debug.print("  - Graph cycle detection for deadlock prevention (200 LOC → zuda)\n\n", .{});

    std.debug.print("zoltraak (Redis-compatible server):\n", .{});
    std.debug.print("  - HyperLogLog for PFCOUNT command (80 LOC → zuda)\n", .{});
    std.debug.print("  - SkipList for sorted sets ZADD/ZRANGE (1800 LOC → zuda)\n", .{});
    std.debug.print("  - LRU for key expiry (50 LOC → zuda)\n\n", .{});

    std.debug.print("sailor (TUI framework):\n", .{});
    std.debug.print("  - Specialized layout algorithms (domain-specific, reference only)\n\n", .{});

    std.debug.print("=== SELECTION GUIDE ===\n\n", .{});

    std.debug.print("Ordered map/set     → RedBlackTree, BTree, SkipList\n", .{});
    std.debug.print("Priority queue      → FibonacciHeap, PairingHeap, BinomialHeap\n", .{});
    std.debug.print("Fast O(1) lookup    → SwissTable, CuckooHashMap, RobinHoodHashMap\n", .{});
    std.debug.print("Cache with eviction → LRUCache, LFUCache, ARCCache\n", .{});
    std.debug.print("Sparse graph        → AdjacencyList\n", .{});
    std.debug.print("Dense graph         → AdjacencyMatrix\n", .{});
    std.debug.print("Spatial queries     → KDTree (k-d), RTree (rectangles), Quadtree (2D)\n", .{});
    std.debug.print("Connectivity        → DisjointSet (union-find)\n", .{});
    std.debug.print("Range queries       → SegmentTree, FenwickTree, IntervalTree\n", .{});
    std.debug.print("Cardinality         → HyperLogLog (~1% error, O(1) space)\n", .{});
    std.debug.print("Membership (approx) → BloomFilter, CuckooFilter\n", .{});
    std.debug.print("Text search         → Trie, SuffixArray, AhoCorasick\n", .{});
    std.debug.print("Work distribution   → WorkStealingQueue\n\n", .{});

    std.debug.print("For API usage examples, see src/containers/*/*.zig inline tests.\n", .{});
    std.debug.print("For migration guides, see CLAUDE.md Consumer Use Case Registry.\n\n", .{});
}
