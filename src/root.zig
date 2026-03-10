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
        // TODO: BloomFilter, CuckooFilter, CountMinSketch, HyperLogLog will be added here
    };
};

/// Namespace for algorithms
pub const algorithms = struct {
    /// Sorting algorithms
    pub const sorting = struct {
        // TODO: TimSort, IntroSort, RadixSort, etc. will be added here
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
    };

    /// String algorithms
    pub const string = struct {
        // TODO: KMP, Boyer-Moore, etc. will be added here
    };

    /// Mathematical algorithms
    pub const math = struct {
        // TODO: GCD, modexp, Miller-Rabin, etc. will be added here
    };

    /// Geometric algorithms
    pub const geometry = struct {
        // TODO: Convex hull, closest pair, etc. will be added here
    };

    /// Dynamic programming utilities
    pub const dynamic_programming = struct {
        // TODO: LIS, LCS, edit distance, knapsack, etc. will be added here
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
}
