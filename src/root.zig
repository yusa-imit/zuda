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

const std = @import("std");

/// Namespace for all container data structures
pub const containers = struct {
    /// Sequential containers (lists, deques, etc.)
    pub const lists = struct {
        pub const SkipList = @import("containers/lists/skip_list.zig").SkipList;
        pub const XorLinkedList = @import("containers/lists/xor_linked_list.zig").XorLinkedList;
        pub const UnrolledLinkedList = @import("containers/lists/unrolled_linked_list.zig").UnrolledLinkedList;
    };

    /// Queue variants
    pub const queues = struct {
        pub const Deque = @import("containers/queues/deque.zig").Deque;
    };

    /// Hash-based containers
    pub const hashing = struct {
        pub const CuckooHashMap = @import("containers/hashing/cuckoo_hash_map.zig").CuckooHashMap;
        pub const AutoCuckooHashMap = @import("containers/hashing/cuckoo_hash_map.zig").AutoCuckooHashMap;
        pub const RobinHoodHashMap = @import("containers/hashing/robin_hood_hash_map.zig").RobinHoodHashMap;
        pub const AutoRobinHoodHashMap = @import("containers/hashing/robin_hood_hash_map.zig").AutoRobinHoodHashMap;
        pub const SwissTable = @import("containers/hashing/swiss_table.zig").SwissTable;
        pub const ConsistentHashRing = @import("containers/hashing/consistent_hash_ring.zig").ConsistentHashRing;
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
        pub const ZAlgorithm = @import("algorithms/string/z_algorithm.zig").ZAlgorithm;
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
    // TODO: Map, Filter, Chain, Zip, etc. will be added here
};

/// Internal utilities (not part of public API)
pub const internal = struct {
    pub const testing = @import("internal/testing.zig");
    pub const bench = @import("internal/bench.zig");
};

test {
    // Import all tests from submodules
    std.testing.refAllDecls(@This());
    _ = internal.testing;
    _ = internal.bench;
    _ = containers.trees.BTree;
    _ = containers.trees.RedBlackTree;
    _ = containers.trees.AVLTree;
    _ = containers.trees.SplayTree;
    _ = containers.trees.AATree;
    _ = containers.trees.ScapegoatTree;
    _ = containers.trees.SegmentTree;
    _ = containers.trees.LazySegmentTree;
    _ = containers.trees.FenwickTree;
    _ = containers.trees.SparseTable;
    _ = containers.trees.IntervalTree;
    _ = containers.strings.Trie;
    _ = containers.strings.RadixTree;
    _ = containers.strings.SuffixArray;
    _ = containers.strings.SuffixTree;
    _ = containers.spatial.KDTree;
    _ = containers.spatial.RTree;
    _ = containers.spatial.QuadTree;
    _ = containers.spatial.OctTree;
    _ = containers.graphs.AdjacencyList;
    _ = containers.graphs.AdjacencyMatrix;
    _ = algorithms.graph.BFS;
    _ = algorithms.graph.DFS;
    _ = algorithms.graph.Dijkstra;
    _ = algorithms.graph.BellmanFord;
    _ = algorithms.graph.AStar;
    _ = algorithms.graph.FloydWarshall;
    _ = algorithms.graph.Johnson;
    _ = algorithms.graph.Kruskal;
    _ = algorithms.graph.Prim;
    _ = algorithms.graph.Boruvka;
    _ = algorithms.graph.TarjanSCC;
    _ = algorithms.graph.KosarajuSCC;
    _ = algorithms.graph.TopologicalSort;
    _ = algorithms.graph.Bridges;
    _ = algorithms.graph.ArticulationPoints;
    _ = algorithms.graph.EdmondsKarp;
    _ = algorithms.graph.Dinic;
    _ = algorithms.graph.PushRelabel;
    _ = algorithms.graph.HopcroftKarp;
    _ = algorithms.graph.Hungarian;
    _ = algorithms.sorting.TimSort;
    _ = algorithms.sorting.IntroSort;
    _ = algorithms.sorting.RadixSort;
    _ = algorithms.sorting.CountingSort;
    _ = algorithms.sorting.MergeSort;
    _ = algorithms.sorting.BlockSort;
    _ = algorithms.string.KMP;
    _ = algorithms.string.BoyerMoore;
    _ = algorithms.string.RabinKarp;
    _ = algorithms.string.AhoCorasick;
    _ = algorithms.string.ZAlgorithm;
    _ = containers.probabilistic.BloomFilter;
    _ = containers.probabilistic.CountMinSketch;
    _ = containers.probabilistic.HyperLogLog;
    _ = containers.probabilistic.CuckooFilter;
    _ = containers.probabilistic.MinHash;
    _ = containers.cache.LRUCache;
    _ = containers.cache.LFUCache;
    _ = containers.cache.ARCCache;
    _ = @import("algorithms/math/gcd.zig");
    _ = @import("algorithms/math/modexp.zig");
    _ = @import("algorithms/math/primality.zig");
    _ = @import("algorithms/math/sieve.zig");
    _ = @import("algorithms/geometry/convex_hull.zig");
    _ = @import("algorithms/geometry/closest_pair.zig");
    _ = @import("algorithms/geometry/haversine.zig");
    _ = @import("algorithms/geometry/geohash.zig");
    _ = @import("algorithms/dynamic_programming/lis.zig");
    _ = @import("algorithms/dynamic_programming/lcs.zig");
    _ = @import("algorithms/dynamic_programming/edit_distance.zig");
    _ = @import("algorithms/dynamic_programming/knapsack.zig");
    _ = @import("algorithms/dynamic_programming/binary_search.zig");
}
