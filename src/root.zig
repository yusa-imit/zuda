//! zuda - Zig Universal Datastructures and Algorithms
//!
//! A comprehensive library of data structures and algorithms for Zig.
//!
//! ## Usage
//!
//! Add zuda to your `build.zig.zon`:
//! ```zig
//! .dependencies = .{
//!     .zuda = .{
//!         .url = "https://github.com/yourusername/zuda/archive/refs/tags/v0.1.0.tar.gz",
//!         .hash = "...",
//!     },
//! },
//! ```
//!
//! Then import in your code:
//! ```zig
//! const zuda = @import("zuda");
//! const SkipList = zuda.containers.lists.SkipList;
//! ```

// Increase comptime evaluation budget for large library with many imports
comptime {
    @setEvalBranchQuota(100000);
}

const std = @import("std");

/// Namespace for all container data structures
pub const containers = struct {
    /// Sequential containers (lists, deques, etc.)
    pub const lists = struct {
        pub const SkipList = @import("containers/lists/skip_list.zig").SkipList;
        pub const ConcurrentSkipList = @import("containers/lists/concurrent_skip_list.zig").ConcurrentSkipList;
        pub const XorLinkedList = @import("containers/lists/xor_linked_list.zig").XorLinkedList;
        pub const UnrolledLinkedList = @import("containers/lists/unrolled_linked_list.zig").UnrolledLinkedList;
    };

    /// Queue variants
    pub const queues = struct {
        pub const Deque = @import("containers/queues/deque.zig").Deque;
        pub const WorkStealingDeque = @import("containers/queues/work_stealing_deque.zig").WorkStealingDeque;
        pub const LockFreeQueue = @import("containers/queues/lock_free_queue.zig").LockFreeQueue;
        pub const LockFreeStack = @import("containers/queues/lock_free_stack.zig").LockFreeStack;
    };

    /// Hash-based containers
    pub const hashing = struct {
        pub const CuckooHashMap = @import("containers/hashing/cuckoo_hash_map.zig").CuckooHashMap;
        pub const AutoCuckooHashMap = @import("containers/hashing/cuckoo_hash_map.zig").AutoCuckooHashMap;
        pub const RobinHoodHashMap = @import("containers/hashing/robin_hood_hash_map.zig").RobinHoodHashMap;
        pub const AutoRobinHoodHashMap = @import("containers/hashing/robin_hood_hash_map.zig").AutoRobinHoodHashMap;
        pub const SwissTable = @import("containers/hashing/swiss_table.zig").SwissTable;
        pub const AutoSwissTable = @import("containers/hashing/swiss_table.zig").AutoSwissTable;
        pub const ConsistentHashRing = @import("containers/hashing/consistent_hash_ring.zig").ConsistentHashRing;
        pub const AutoConsistentHashRing = @import("containers/hashing/consistent_hash_ring.zig").AutoConsistentHashRing;
    };

    /// Heap variants
    pub const heaps = struct {
        pub const FibonacciHeap = @import("containers/heaps/fibonacci_heap.zig").FibonacciHeap;
        pub const BinomialHeap = @import("containers/heaps/binomial_heap.zig").BinomialHeap;
        pub const PairingHeap = @import("containers/heaps/pairing_heap.zig").PairingHeap;
        pub const DaryHeap = @import("containers/heaps/dary_heap.zig").DaryHeap;
    };

    /// Tree-based containers
    pub const trees = struct {
        pub const BTree = @import("containers/trees/btree.zig").BTree;
        pub const RedBlackTree = @import("containers/trees/red_black_tree.zig").RedBlackTree;
        pub const AVLTree = @import("containers/trees/avl_tree.zig").AVLTree;
        pub const SplayTree = @import("containers/trees/splay_tree.zig").SplayTree;
        pub const AATree = @import("containers/trees/aa_tree.zig").AATree;
        pub const ScapegoatTree = @import("containers/trees/scapegoat_tree.zig").ScapegoatTree;
        pub const SegmentTree = @import("containers/trees/segment_tree.zig").SegmentTree;
        pub const LazySegmentTree = @import("containers/trees/lazy_segment_tree.zig").LazySegmentTree;
        pub const FenwickTree = @import("containers/trees/fenwick_tree.zig").FenwickTree;
        pub const SparseTable = @import("containers/trees/sparse_table.zig").SparseTable;
        pub const IntervalTree = @import("containers/trees/interval_tree.zig").IntervalTree;
        pub const VanEmdeBoasTree = @import("containers/trees/van_emde_boas_tree.zig").VanEmdeBoasTree;
    };

    /// Graph representations
    pub const graphs = struct {
        pub const AdjacencyList = @import("containers/graphs/adjacency_list.zig").AdjacencyList;
        pub const IntGraph = @import("containers/graphs/adjacency_list.zig").IntGraph;
        pub const AdjacencyMatrix = @import("containers/graphs/adjacency_matrix.zig").AdjacencyMatrix;
        pub const CompressedSparseRow = @import("containers/graphs/compressed_sparse_row.zig").CompressedSparseRow;
        pub const CSREdge = @import("containers/graphs/compressed_sparse_row.zig").Edge;
        pub const EdgeList = @import("containers/graphs/edge_list.zig").EdgeList;
    };

    /// String-specialized structures
    pub const strings = struct {
        pub const Trie = @import("containers/trees/trie.zig").Trie;
        pub const RadixTree = @import("containers/trees/radix_tree.zig").RadixTree;
        pub const SuffixArray = @import("containers/strings/suffix_array.zig").SuffixArray;
        pub const SuffixTree = @import("containers/strings/suffix_tree.zig").SuffixTree;
        pub const DoubleArrayTrie = @import("containers/strings/double_array_trie.zig").DoubleArrayTrie;
    };

    /// Spatial index structures
    pub const spatial = struct {
        pub const KDTree = @import("containers/spatial/kd_tree.zig").KDTree;
        pub const RTree = @import("containers/spatial/r_tree.zig").RTree;
        pub const BoundingBox = @import("containers/spatial/r_tree.zig").BoundingBox;
        pub const QuadTree = @import("containers/spatial/quad_tree.zig").QuadTree;
        pub const OctTree = @import("containers/spatial/octtree.zig").OctTree;
    };

    /// Probabilistic data structures
    pub const probabilistic = struct {
        pub const BloomFilter = @import("containers/probabilistic/bloom_filter.zig").BloomFilter;
        pub const defaultHashInt = @import("containers/probabilistic/bloom_filter.zig").defaultHashInt;
        pub const defaultHashSlice = @import("containers/probabilistic/bloom_filter.zig").defaultHashSlice;
        pub const CountMinSketch = @import("containers/probabilistic/count_min_sketch.zig").CountMinSketch;
        pub const HyperLogLog = @import("containers/probabilistic/hyperloglog.zig").HyperLogLog;
        pub const CuckooFilter = @import("containers/probabilistic/cuckoo_filter.zig").CuckooFilter;
        pub const MinHash = @import("containers/probabilistic/minhash.zig").MinHash;
    };

    /// Cache eviction strategies
    pub const cache = struct {
        pub const LRUCache = @import("containers/cache/lru_cache.zig").LRUCache;
        pub const LFUCache = @import("containers/cache/lfu_cache.zig").LFUCache;
        pub const ARCCache = @import("containers/cache/arc_cache.zig").ARCCache;
    };

    /// Specialized data structures
    pub const specialized = struct {
        pub const DisjointSet = @import("containers/specialized/disjoint_set.zig").DisjointSet;
        pub const Rope = @import("containers/specialized/rope.zig").Rope;
        pub const BKTree = @import("containers/specialized/bk_tree.zig").BKTree;
    };

    /// Exotic/advanced data structures
    pub const exotic = struct {
        pub const DancingLinks = @import("containers/exotic/dancing_links.zig").DancingLinks;
    };

    /// Persistent (immutable) data structures
    pub const persistent = struct {
        pub const PersistentArray = @import("containers/persistent/persistent_array.zig").PersistentArray;
        pub const PersistentHashMap = @import("containers/hashing/persistent_hash_map.zig").PersistentHashMap;
        pub const PersistentRBTree = @import("containers/persistent/persistent_rbtree.zig").PersistentRBTree;
    };
};

/// Namespace for algorithms
pub const algorithms = struct {
    /// Sorting algorithms
    pub const sorting = struct {
        pub const TimSort = @import("algorithms/sorting/timsort.zig").TimSort;
        pub const timsort = @import("algorithms/sorting/timsort.zig").sort;
        pub const IntroSort = @import("algorithms/sorting/introsort.zig").IntroSort;
        pub const introsort = @import("algorithms/sorting/introsort.zig").sort;
        pub const RadixSort = @import("algorithms/sorting/radixsort.zig").RadixSort;
        pub const CountingSort = @import("algorithms/sorting/countingsort.zig").CountingSort;
        pub const MergeSort = @import("algorithms/sorting/mergesort.zig").MergeSort;
        pub const MergeSortBottomUp = @import("algorithms/sorting/mergesort.zig").MergeSortBottomUp;
        pub const NaturalMergeSort = @import("algorithms/sorting/mergesort.zig").NaturalMergeSort;
        pub const BlockSort = @import("algorithms/sorting/blocksort.zig").BlockSort;
    };

    /// Searching algorithms
    pub const searching = struct {
        // TODO: Binary search variants, etc. will be added here
    };

    /// Graph algorithms
    pub const graph = struct {
        pub const BFS = @import("algorithms/graph/bfs.zig").BFS;
        pub const DFS = @import("algorithms/graph/dfs.zig").DFS;
        pub const Dijkstra = @import("algorithms/graph/dijkstra.zig").Dijkstra;
        pub const BellmanFord = @import("algorithms/graph/bellman_ford.zig").BellmanFord;
        pub const AStar = @import("algorithms/graph/a_star.zig").AStar;
        pub const FloydWarshall = @import("algorithms/graph/floyd_warshall.zig").FloydWarshall;
        pub const Johnson = @import("algorithms/graph/johnson.zig").Johnson;
        pub const Kruskal = @import("algorithms/graph/kruskal.zig").Kruskal;
        pub const Prim = @import("algorithms/graph/prim.zig").Prim;
        pub const Boruvka = @import("algorithms/graph/boruvka.zig").Boruvka;
        pub const TarjanSCC = @import("algorithms/graph/tarjan_scc.zig").TarjanSCC;
        pub const KosarajuSCC = @import("algorithms/graph/kosaraju_scc.zig").KosarajuSCC;
        pub const TopologicalSort = @import("algorithms/graph/topological_sort.zig").TopologicalSort;
        pub const Bridges = @import("algorithms/graph/bridges.zig").Bridges;
        pub const ArticulationPoints = @import("algorithms/graph/articulation_points.zig").ArticulationPoints;
        pub const EdmondsKarp = @import("algorithms/graph/edmonds_karp.zig").EdmondsKarp;
        pub const Dinic = @import("algorithms/graph/dinic.zig").Dinic;
        pub const PushRelabel = @import("algorithms/graph/push_relabel.zig").PushRelabel;
        pub const HopcroftKarp = @import("algorithms/graph/hopcroft_karp.zig").HopcroftKarp;
        pub const Hungarian = @import("algorithms/graph/hungarian.zig").Hungarian;
    };

    /// String algorithms
    pub const string = struct {
        pub const KMP = @import("algorithms/string/kmp.zig").KMP;
        pub const kmpSearch = @import("algorithms/string/kmp.zig").search;
        pub const kmpSearchAll = @import("algorithms/string/kmp.zig").searchAll;
        pub const BoyerMoore = @import("algorithms/string/boyer_moore.zig").BoyerMoore;
        pub const boyerMooreSearch = @import("algorithms/string/boyer_moore.zig").search;
        pub const boyerMooreSearchAll = @import("algorithms/string/boyer_moore.zig").searchAll;
        pub const RabinKarp = @import("algorithms/string/rabin_karp.zig").RabinKarp;
        pub const AhoCorasick = @import("algorithms/string/aho_corasick.zig").AhoCorasick;
        pub const AhoCorasickASCII = @import("algorithms/string/aho_corasick.zig").AhoCorasickASCII;
        pub const ZAlgorithm = @import("algorithms/string/z_algorithm.zig").ZAlgorithm;
        pub const globMatch = @import("algorithms/string/glob_match.zig").match;
        pub const globMatchCaseInsensitive = @import("algorithms/string/glob_match.zig").matchCaseInsensitive;
    };

    /// Mathematical algorithms
    pub const math = struct {
        pub const gcd = @import("algorithms/math/gcd.zig").gcd;
        pub const lcm = @import("algorithms/math/gcd.zig").lcm;
        pub const extendedGcd = @import("algorithms/math/gcd.zig").extendedGcd;
        pub const binaryGcd = @import("algorithms/math/gcd.zig").binaryGcd;
        pub const modExp = @import("algorithms/math/modexp.zig").modExp;
        pub const modInverse = @import("algorithms/math/modexp.zig").modInverse;
        pub const millerRabin = @import("algorithms/math/primality.zig").millerRabin;
        pub const millerRabinDeterministic = @import("algorithms/math/primality.zig").millerRabinDeterministic;
        pub const trialDivision = @import("algorithms/math/primality.zig").trialDivision;
        pub const fermat = @import("algorithms/math/primality.zig").fermat;
        pub const sieveOfEratosthenes = @import("algorithms/math/sieve.zig").sieveOfEratosthenes;
        pub const segmentedSieve = @import("algorithms/math/sieve.zig").segmentedSieve;
        pub const countPrimes = @import("algorithms/math/sieve.zig").countPrimes;
        pub const isPrime = @import("algorithms/math/sieve.zig").isPrime;
        pub const nthPrime = @import("algorithms/math/sieve.zig").nthPrime;
        pub const chineseRemainderTheorem = @import("algorithms/math/crt.zig").chineseRemainderTheorem;
        pub const crtTwo = @import("algorithms/math/crt.zig").crtTwo;
        pub const CRTResult = @import("algorithms/math/crt.zig").CRTResult;
        pub const Moduli = @import("algorithms/math/ntt.zig").Moduli;
        pub const ntt = @import("algorithms/math/ntt.zig").ntt;
        pub const intt = @import("algorithms/math/ntt.zig").intt;
        pub const nttMultiply = @import("algorithms/math/ntt.zig").multiply;
    };

    /// Geometric algorithms
    pub const geometry = struct {
        pub const Point = @import("algorithms/geometry/convex_hull.zig").Point;
        pub const grahamScan = @import("algorithms/geometry/convex_hull.zig").grahamScan;
        pub const jarvisMarch = @import("algorithms/geometry/convex_hull.zig").jarvisMarch;
        pub const ClosestPairResult = @import("algorithms/geometry/closest_pair.zig").ClosestPairResult;
        pub const closestPair = @import("algorithms/geometry/closest_pair.zig").closestPair;
        pub const closestPairBruteForce = @import("algorithms/geometry/closest_pair.zig").bruteForce;
        pub const Coord = @import("algorithms/geometry/haversine.zig").Coord;
        pub const haversineDistance = @import("algorithms/geometry/haversine.zig").distance;
        pub const haversineDistanceKm = @import("algorithms/geometry/haversine.zig").distanceKm;
        pub const haversineDistanceMi = @import("algorithms/geometry/haversine.zig").distanceMi;
        pub const haversineDistanceM = @import("algorithms/geometry/haversine.zig").distanceM;
        pub const initialBearing = @import("algorithms/geometry/haversine.zig").initialBearing;
        pub const destination = @import("algorithms/geometry/haversine.zig").destination;
        pub const midpoint = @import("algorithms/geometry/haversine.zig").midpoint;
        pub const EARTH_RADIUS_KM = @import("algorithms/geometry/haversine.zig").EARTH_RADIUS_KM;
        pub const EARTH_RADIUS_MI = @import("algorithms/geometry/haversine.zig").EARTH_RADIUS_MI;
        pub const EARTH_RADIUS_M = @import("algorithms/geometry/haversine.zig").EARTH_RADIUS_M;
        pub const BoundingBox = @import("algorithms/geometry/geohash.zig").BoundingBox;
        pub const geohashEncode = @import("algorithms/geometry/geohash.zig").encode;
        pub const geohashDecode = @import("algorithms/geometry/geohash.zig").decode;
        pub const geohashNeighbor = @import("algorithms/geometry/geohash.zig").neighbor;
        pub const geohashNeighbors = @import("algorithms/geometry/geohash.zig").neighbors;
        pub const GeohashDirection = @import("algorithms/geometry/geohash.zig").Direction;
    };

    /// Dynamic programming utilities
    pub const dynamic_programming = struct {
        // Longest Increasing Subsequence
        pub const lengthLIS_DP = @import("algorithms/dynamic_programming/lis.zig").lengthDP;
        pub const lengthLIS = @import("algorithms/dynamic_programming/lis.zig").lengthBinarySearch;
        pub const findLIS = @import("algorithms/dynamic_programming/lis.zig").findSequence;
        pub const lengthLIS_WithComparator = @import("algorithms/dynamic_programming/lis.zig").lengthWithComparator;
        pub const lengthLNDS = @import("algorithms/dynamic_programming/lis.zig").lengthNonDecreasing;

        // Longest Common Subsequence
        pub const lengthLCS = @import("algorithms/dynamic_programming/lcs.zig").length;
        pub const lengthLCS_Optimized = @import("algorithms/dynamic_programming/lcs.zig").lengthOptimized;
        pub const findLCS = @import("algorithms/dynamic_programming/lcs.zig").findSequence;
        pub const findAllLCS = @import("algorithms/dynamic_programming/lcs.zig").findAllSequences;
        pub const lengthLCS_String = @import("algorithms/dynamic_programming/lcs.zig").lengthString;
        pub const findLCS_String = @import("algorithms/dynamic_programming/lcs.zig").findSequenceString;

        // Edit Distance (Levenshtein)
        pub const EditOp = @import("algorithms/dynamic_programming/edit_distance.zig").EditOp;
        pub const Edit = @import("algorithms/dynamic_programming/edit_distance.zig").Edit;
        pub const editDistance = @import("algorithms/dynamic_programming/edit_distance.zig").distance;
        pub const editDistanceOptimized = @import("algorithms/dynamic_programming/edit_distance.zig").distanceOptimized;
        pub const editDistanceWithEdits = @import("algorithms/dynamic_programming/edit_distance.zig").distanceWithEdits;
        pub const stringSimilarity = @import("algorithms/dynamic_programming/edit_distance.zig").similarity;
        pub const withinEditThreshold = @import("algorithms/dynamic_programming/edit_distance.zig").withinThreshold;

        // Knapsack problems
        pub const Item = @import("algorithms/dynamic_programming/knapsack.zig").Item;
        pub const knapsack01 = @import("algorithms/dynamic_programming/knapsack.zig").zeroOne;
        pub const knapsack01Optimized = @import("algorithms/dynamic_programming/knapsack.zig").zeroOneOptimized;
        pub const knapsack01WithItems = @import("algorithms/dynamic_programming/knapsack.zig").zeroOneWithItems;
        pub const knapsackUnbounded = @import("algorithms/dynamic_programming/knapsack.zig").unbounded;
        pub const knapsackUnboundedWithCounts = @import("algorithms/dynamic_programming/knapsack.zig").unboundedWithCounts;
        pub const knapsackFractional = @import("algorithms/dynamic_programming/knapsack.zig").fractional;
        pub const knapsackBounded = @import("algorithms/dynamic_programming/knapsack.zig").bounded;

        // Binary search variants
        pub const binarySearch = @import("algorithms/dynamic_programming/binary_search.zig").search;
        pub const lowerBound = @import("algorithms/dynamic_programming/binary_search.zig").lowerBound;
        pub const upperBound = @import("algorithms/dynamic_programming/binary_search.zig").upperBound;
        pub const equalRange = @import("algorithms/dynamic_programming/binary_search.zig").equalRange;
        pub const binarySearchWithComparator = @import("algorithms/dynamic_programming/binary_search.zig").searchWithComparator;
        pub const findFirst = @import("algorithms/dynamic_programming/binary_search.zig").findFirst;
        pub const findLast = @import("algorithms/dynamic_programming/binary_search.zig").findLast;
        pub const countOccurrences = @import("algorithms/dynamic_programming/binary_search.zig").count;
        pub const searchOnAnswer = @import("algorithms/dynamic_programming/binary_search.zig").searchOnAnswer;
        pub const searchFloat = @import("algorithms/dynamic_programming/binary_search.zig").searchFloat;
        pub const searchRotated = @import("algorithms/dynamic_programming/binary_search.zig").searchRotated;
        pub const findPeak = @import("algorithms/dynamic_programming/binary_search.zig").findPeak;
        pub const findMinRotated = @import("algorithms/dynamic_programming/binary_search.zig").findMinRotated;
    };
};

/// Iterator adaptors and utilities
pub const iterators = struct {
    pub const Map = @import("iterators/map.zig").Map;
    pub const Filter = @import("iterators/filter.zig").Filter;
    pub const Chain = @import("iterators/chain.zig").Chain;
    pub const Zip = @import("iterators/zip.zig").Zip;
    pub const Take = @import("iterators/take.zig").Take;
    pub const TakeWhile = @import("iterators/take_while.zig").TakeWhile;
    pub const Skip = @import("iterators/skip.zig").Skip;
    pub const SkipWhile = @import("iterators/skip_while.zig").SkipWhile;
    pub const Enumerate = @import("iterators/enumerate.zig").Enumerate;
    pub const FlatMap = @import("iterators/flat_map.zig").FlatMap;
    pub const Partition = @import("iterators/partition.zig").Partition;
    pub const collect = @import("iterators/collect.zig").collect;
};

/// Utility functions for working with containers (comparators, hash functions, etc.)
pub const utils = @import("utils.zig");

/// Compatibility layers for consumer project migration
pub const compat = struct {
    /// Compatibility layer for silica's BTree API (4,300 LOC replacement)
    pub const silica_btree = @import("compat/silica_btree.zig");
    /// Compatibility layer for zr's DAG API (715 LOC replacement)
    pub const zr_dag = @import("compat/zr_dag.zig");
    /// Compatibility layer for zoltraak's SortedSet API (1,800 LOC replacement)
    pub const zoltraak_sortedset = @import("compat/zoltraak_sortedset.zig");
};

/// Scientific computing modules (v2.0 track)
pub const ndarray = struct {
    /// N-dimensional arrays with flexible memory layouts
    pub const NDArray = @import("ndarray/ndarray.zig").NDArray;
    pub const Layout = @import("ndarray/ndarray.zig").Layout;
};

/// Linear algebra operations (v2.0 track)
pub const linalg = struct {
    /// BLAS Level 1 — Vector-vector operations
    pub const blas = @import("linalg/blas.zig");
    /// Matrix decompositions — LU, QR, SVD, Cholesky
    pub const decompositions = @import("linalg/decompositions.zig");
    /// LU decomposition with partial pivoting
    pub const lu = @import("linalg/lu.zig");
    /// Linear system solver — solve(A, b)
    pub const solve = @import("linalg/solve.zig");
    /// Matrix properties — rank, condition number
    pub const properties = @import("linalg/properties.zig");
};

/// Statistics and data analysis
pub const stats = struct {
    /// Descriptive statistics — mean, median, mode, variance, std, quantile, skewness, kurtosis
    pub const descriptive = @import("stats/descriptive.zig");

    /// Hypothesis testing — one-sample, independent samples, paired samples t-tests
    pub const hypothesis = @import("stats/hypothesis.zig");

    /// Probability distributions — Uniform, Normal, Exponential, Poisson, Binomial, Bernoulli, Geometric, Gamma, Beta, ChiSquared, StudentT, F, etc.
    pub const distributions = struct {
        pub const Uniform = @import("stats/distributions/uniform.zig").Uniform;
        pub const Normal = @import("stats/distributions/normal.zig").Normal;
        pub const Exponential = @import("stats/distributions/exponential.zig").Exponential;
        pub const Poisson = @import("stats/distributions/poisson.zig").Poisson;
        pub const Binomial = @import("stats/distributions/binomial.zig").Binomial;
        pub const Bernoulli = @import("stats/distributions/bernoulli.zig").Bernoulli;
        pub const Geometric = @import("stats/distributions/geometric.zig").Geometric;
        pub const Gamma = @import("stats/distributions/gamma.zig").Gamma;
        pub const Beta = @import("stats/distributions/beta.zig").Beta;
        pub const ChiSquared = @import("stats/distributions/chi_squared.zig").ChiSquared;
        pub const StudentT = @import("stats/distributions/student_t.zig").StudentT;
        pub const FDistribution = @import("stats/distributions/f_distribution.zig").FDistribution;
    };
};

/// Internal utilities (not part of public API)
pub const internal = struct {
    pub const testing = @import("internal/testing.zig");
    pub const bench = @import("internal/bench.zig");
};

test {
    // NOTE: We previously had a large test block that referenced all containers/algorithms.
    // This caused CI to hang due to excessive compilation time (compiling 100+ test suites).
    // Individual module tests are already run via `zig build test`, so this refAll is sufficient.
    std.testing.refAllDecls(@This());

    // Explicitly import stats modules to trigger their tests
    _ = @import("stats/descriptive.zig");
    _ = @import("stats/hypothesis.zig");
    _ = @import("stats/distributions/uniform.zig");
    _ = @import("stats/distributions/normal.zig");
    _ = @import("stats/distributions/exponential.zig");
    _ = @import("stats/distributions/poisson.zig");
    _ = @import("stats/distributions/binomial.zig");
    _ = @import("stats/distributions/bernoulli.zig");
    _ = @import("stats/distributions/geometric.zig");
    _ = @import("stats/distributions/gamma.zig");
    _ = @import("stats/distributions/beta.zig");
    _ = @import("stats/distributions/chi_squared.zig");
    _ = @import("stats/distributions/student_t.zig");
    _ = @import("stats/distributions/f_distribution.zig");
}
