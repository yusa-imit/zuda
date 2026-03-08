//! zuda - Zig Unofficial Datastructures and Algorithms
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
        // TODO: SwissTable, ConsistentHashRing will be added here
    };

    /// Heap variants
    pub const heaps = struct {
        // TODO: FibonacciHeap, BinomialHeap, PairingHeap, DaryHeap will be added here
    };

    /// Tree-based containers
    pub const trees = struct {
        // TODO: RedBlackTree, AVLTree, BTree, etc. will be added here
    };

    /// Graph representations
    pub const graphs = struct {
        // TODO: AdjacencyList, AdjacencyMatrix, etc. will be added here
    };

    /// String-specialized structures
    pub const strings = struct {
        // TODO: SuffixArray, SuffixTree, Trie, etc. will be added here
    };

    /// Spatial index structures
    pub const spatial = struct {
        // TODO: KDTree, RTree, QuadTree, OctTree will be added here
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
        // TODO: BFS, DFS, Dijkstra, etc. will be added here
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
}
