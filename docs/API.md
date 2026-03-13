# zuda API Reference

> Complete reference for all data structures and algorithms in zuda v0.5.0

## Table of Contents

1. [Containers](#containers)
   - [Lists](#lists)
   - [Queues & Deques](#queues--deques)
   - [Heaps](#heaps)
   - [Hash Tables](#hash-tables)
   - [Trees](#trees)
   - [Graphs](#graphs)
   - [Spatial](#spatial)
   - [Strings](#strings)
   - [Caches](#caches)
   - [Probabilistic](#probabilistic)
   - [Persistent](#persistent)
   - [Specialized](#specialized)
2. [Algorithms](#algorithms)
   - [Sorting](#sorting)
   - [Graph Algorithms](#graph-algorithms)
   - [String Algorithms](#string-algorithms)
   - [Dynamic Programming](#dynamic-programming)
   - [Geometry](#geometry)
   - [Math](#math)
3. [FFI](#ffi)

---

## Containers

### Lists

#### SkipList
**Path**: `src/containers/lists/skip_list.zig`

Probabilistic balanced structure providing O(log n) search, insert, and delete operations.

```zig
const SkipList = @import("zuda").containers.lists.SkipList;

pub fn SkipList(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: K, b: K) std.math.Order,
) type
```

**Key Methods**:
- `init(allocator: Allocator, max_level: u32) Self` - O(1)
- `insert(key: K, value: V) !?V` - O(log n) average
- `get(key: K) ?V` - O(log n) average
- `remove(key: K) ?Entry` - O(log n) average
- `iterator() Iterator` - Forward iteration

**Use Cases**: Ordered map alternative to tree structures, concurrent-friendly design

---

#### XorLinkedList
**Path**: `src/containers/lists/xor_linked_list.zig`

Memory-efficient doubly-linked list using XOR of adjacent pointers.

```zig
const XorLinkedList = @import("zuda").containers.lists.XorLinkedList;

pub fn XorLinkedList(comptime T: type) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `prepend(data: T) !void` - O(1)
- `append(data: T) !void` - O(1)
- `remove(node: *Node) void` - O(1) with node pointer
- `iterator() Iterator` - Forward iteration

**Use Cases**: Low memory overhead doubly-linked list, embedded systems

---

#### UnrolledLinkedList
**Path**: `src/containers/lists/unrolled_linked_list.zig`

Hybrid list-array structure for cache-friendly linked list operations.

```zig
const UnrolledLinkedList = @import("zuda").containers.lists.UnrolledLinkedList;

pub fn UnrolledLinkedList(comptime T: type, comptime node_capacity: usize) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `append(value: T) !void` - O(1) amortized
- `insert(index: usize, value: T) !void` - O(n)
- `remove(index: usize) ?T` - O(n)
- `get(index: usize) ?T` - O(n/node_capacity)

**Use Cases**: Better cache locality than standard linked lists, fewer allocations

---

#### ConcurrentSkipList
**Path**: `src/containers/lists/concurrent_skip_list.zig`

Thread-safe lock-free skip list for concurrent environments.

```zig
const ConcurrentSkipList = @import("zuda").containers.lists.ConcurrentSkipList;

pub fn ConcurrentSkipList(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: K, b: K) std.math.Order,
) type
```

**Key Methods**:
- `init(allocator: Allocator, max_level: u32) Self` - O(1)
- `insert(key: K, value: V) !bool` - O(log n) lock-free
- `get(key: K) ?V` - O(log n) lock-free
- `remove(key: K) bool` - O(log n) lock-free

**Use Cases**: Multi-threaded ordered maps, high-contention scenarios

---

### Queues & Deques

#### Deque
**Path**: `src/containers/queues/deque.zig`

Double-ended queue with O(1) operations on both ends.

```zig
const Deque = @import("zuda").containers.queues.Deque;

pub fn Deque(comptime T: type) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `pushFront(value: T) !void` - O(1) amortized
- `pushBack(value: T) !void` - O(1) amortized
- `popFront() ?T` - O(1)
- `popBack() ?T` - O(1)
- `peekFront() ?T` - O(1)
- `peekBack() ?T` - O(1)

**Use Cases**: BFS, sliding window algorithms, palindrome checking

---

#### LockFreeQueue
**Path**: `src/containers/queues/lock_free_queue.zig`

Thread-safe FIFO queue using atomic operations (Michael-Scott algorithm).

```zig
const LockFreeQueue = @import("zuda").containers.queues.LockFreeQueue;

pub fn LockFreeQueue(comptime T: type) type
```

**Key Methods**:
- `init(allocator: Allocator) !Self` - O(1)
- `enqueue(value: T) !void` - O(1) lock-free
- `dequeue() ?T` - O(1) lock-free

**Use Cases**: Producer-consumer patterns, multi-threaded task queues

---

#### LockFreeStack
**Path**: `src/containers/queues/lock_free_stack.zig`

Thread-safe LIFO stack using atomic CAS operations.

```zig
const LockFreeStack = @import("zuda").containers.queues.LockFreeStack;

pub fn LockFreeStack(comptime T: type) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `push(value: T) !void` - O(1) lock-free
- `pop() ?T` - O(1) lock-free

**Use Cases**: Lock-free memory pools, parallel algorithms

---

#### WorkStealingDeque
**Path**: `src/containers/queues/work_stealing_deque.zig`

Chase-Lev work-stealing deque for task schedulers.

```zig
const WorkStealingDeque = @import("zuda").containers.queues.WorkStealingDeque;

pub fn WorkStealingDeque(comptime T: type) type
```

**Key Methods**:
- `init(allocator: Allocator) !Self` - O(1)
- `push(value: T) !void` - O(1) (owner only)
- `pop() ?T` - O(1) (owner only)
- `steal() ?T` - O(1) (thieves)

**Use Cases**: Work-stealing schedulers, parallel task execution (see: zr task runner)

---

### Heaps

#### FibonacciHeap
**Path**: `src/containers/heaps/fibonacci_heap.zig`

Heap with O(1) amortized insert and decrease-key operations.

```zig
const FibonacciHeap = @import("zuda").containers.heaps.FibonacciHeap;

pub fn FibonacciHeap(
    comptime T: type,
    comptime Context: type,
    comptime lessThan: fn (ctx: Context, a: T, b: T) bool,
) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `insert(value: T) !*Node` - O(1) amortized
- `findMin() ?T` - O(1)
- `extractMin() ?T` - O(log n) amortized
- `decreaseKey(node: *Node, new_value: T) !void` - O(1) amortized
- `merge(other: *Self) !void` - O(1)

**Use Cases**: Dijkstra's algorithm, Prim's MST, priority queues with frequent decrease-key

---

#### BinomialHeap
**Path**: `src/containers/heaps/binomial_heap.zig`

Heap based on binomial trees with O(log n) operations.

```zig
const BinomialHeap = @import("zuda").containers.heaps.BinomialHeap;

pub fn BinomialHeap(
    comptime T: type,
    comptime Context: type,
    comptime lessThan: fn (ctx: Context, a: T, b: T) bool,
) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `insert(value: T) !void` - O(log n)
- `findMin() ?T` - O(log n)
- `extractMin() ?T` - O(log n)
- `merge(other: *Self) !void` - O(log n)

**Use Cases**: Priority queues with frequent merges

---

#### PairingHeap
**Path**: `src/containers/heaps/pairing_heap.zig`

Simple heap with competitive performance, easier to implement than Fibonacci.

```zig
const PairingHeap = @import("zuda").containers.heaps.PairingHeap;

pub fn PairingHeap(
    comptime T: type,
    comptime Context: type,
    comptime lessThan: fn (ctx: Context, a: T, b: T) bool,
) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `insert(value: T) !*Node` - O(1)
- `findMin() ?T` - O(1)
- `extractMin() ?T` - O(log n) amortized
- `decreaseKey(node: *Node, new_value: T) void` - O(log n) amortized
- `merge(other: *Self) void` - O(1)

**Use Cases**: Simpler alternative to Fibonacci heap with similar performance

---

#### DaryHeap
**Path**: `src/containers/heaps/dary_heap.zig`

Generalization of binary heap with d children per node.

```zig
const DaryHeap = @import("zuda").containers.heaps.DaryHeap;

pub fn DaryHeap(
    comptime T: type,
    comptime d: u32,
    comptime Context: type,
    comptime lessThan: fn (ctx: Context, a: T, b: T) bool,
) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `insert(value: T) !void` - O(log_d n)
- `peek() ?T` - O(1)
- `extractMin() ?T` - O(d * log_d n)

**Use Cases**: d=4 often optimal for cache performance, faster inserts than binary heap

---

### Hash Tables

#### CuckooHashMap
**Path**: `src/containers/hashing/cuckoo_hash_map.zig`

Hash table with O(1) worst-case lookup using two hash functions.

```zig
const CuckooHashMap = @import("zuda").containers.hashing.CuckooHashMap;

pub fn CuckooHashMap(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime hash1: fn (ctx: Context, key: K) u64,
    comptime hash2: fn (ctx: Context, key: K) u64,
    comptime eql: fn (ctx: Context, a: K, b: K) bool,
) type
```

**Key Methods**:
- `init(allocator: Allocator, capacity: usize) !Self` - O(n)
- `put(key: K, value: V) !void` - O(1) amortized
- `get(key: K) ?V` - O(1) worst-case
- `remove(key: K) ?V` - O(1) worst-case

**Use Cases**: Real-time systems requiring predictable lookup time

---

#### RobinHoodHashMap
**Path**: `src/containers/hashing/robin_hood_hash_map.zig`

Open addressing hash table with Robin Hood heuristic for low variance.

```zig
const RobinHoodHashMap = @import("zuda").containers.hashing.RobinHoodHashMap;

pub fn RobinHoodHashMap(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime hash: fn (ctx: Context, key: K) u64,
    comptime eql: fn (ctx: Context, a: K, b: K) bool,
) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `put(key: K, value: V) !void` - O(1) amortized
- `get(key: K) ?V` - O(1) average
- `remove(key: K) ?V` - O(1) average

**Use Cases**: General-purpose hash table with good worst-case performance

---

#### SwissTable
**Path**: `src/containers/hashing/swiss_table.zig`

Google's SwissTable design with SIMD-friendly group-based probing.

```zig
const SwissTable = @import("zuda").containers.hashing.SwissTable;

pub fn SwissTable(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime hash: fn (ctx: Context, key: K) u64,
    comptime eql: fn (ctx: Context, a: K, b: K) bool,
) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `put(key: K, value: V) !void` - O(1) amortized
- `get(key: K) ?V` - O(1) average, SIMD-accelerated
- `remove(key: K) ?V` - O(1) average

**Use Cases**: High-performance hash table, large datasets

---

#### ConsistentHashRing
**Path**: `src/containers/hashing/consistent_hash_ring.zig`

Consistent hashing with virtual nodes for distributed systems.

```zig
const ConsistentHashRing = @import("zuda").containers.hashing.ConsistentHashRing;

pub fn ConsistentHashRing(comptime K: type) type
```

**Key Methods**:
- `init(allocator: Allocator, replicas: u32) Self` - O(1)
- `addNode(node: K) !void` - O(replicas * log n)
- `removeNode(node: K) !void` - O(replicas * log n)
- `getNode(key: []const u8) ?K` - O(log n)

**Use Cases**: Load balancing, distributed caching, sharding (see: zoltraak distributed mode)

---

#### PersistentHashMap
**Path**: `src/containers/hashing/persistent_hash_map.zig`

Immutable hash map using Hash Array Mapped Trie (HAMT).

```zig
const PersistentHashMap = @import("zuda").containers.hashing.PersistentHashMap;

pub fn PersistentHashMap(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime hash: fn (ctx: Context, key: K) u64,
    comptime eql: fn (ctx: Context, a: K, b: K) bool,
) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `insert(key: K, value: V) !Self` - O(log n), returns new version
- `get(key: K) ?V` - O(log n)
- `remove(key: K) !Self` - O(log n), returns new version

**Use Cases**: Functional programming, undo/redo systems, version control

---

### Trees

#### RedBlackTree
**Path**: `src/containers/trees/red_black_tree.zig`

Self-balancing BST with strict O(log n) operations.

```zig
const RedBlackTree = @import("zuda").containers.trees.RedBlackTree;

pub fn RedBlackTree(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: K, b: K) std.math.Order,
) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `insert(key: K, value: V) !void` - O(log n)
- `get(key: K) ?V` - O(log n)
- `remove(key: K) ?Entry` - O(log n)
- `iterator() Iterator` - In-order traversal
- `validate() !void` - Check RB-tree invariants

**Use Cases**: Ordered maps, interval scheduling, general-purpose balanced tree

---

#### AVLTree
**Path**: `src/containers/trees/avl_tree.zig`

Self-balancing BST with strict height balance (|left.height - right.height| ≤ 1).

```zig
const AVLTree = @import("zuda").containers.trees.AVLTree;

pub fn AVLTree(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: K, b: K) std.math.Order,
) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `insert(key: K, value: V) !void` - O(log n)
- `get(key: K) ?V` - O(log n)
- `remove(key: K) ?Entry` - O(log n)

**Use Cases**: Lookup-heavy workloads (faster lookups than RB-tree due to stricter balance)

---

#### SplayTree
**Path**: `src/containers/trees/splay_tree.zig`

Self-adjusting BST that moves accessed nodes to the root.

```zig
const SplayTree = @import("zuda").containers.trees.SplayTree;

pub fn SplayTree(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: K, b: K) std.math.Order,
) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `insert(key: K, value: V) !void` - O(log n) amortized
- `get(key: K) ?V` - O(log n) amortized, splays to root
- `remove(key: K) ?Entry` - O(log n) amortized

**Use Cases**: Temporal locality, caching, access pattern optimization

---

#### AATree
**Path**: `src/containers/trees/aa_tree.zig`

Simplified red-black tree variant (Arne Andersson tree).

```zig
const AATree = @import("zuda").containers.trees.AATree;

pub fn AATree(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: K, b: K) std.math.Order,
) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `insert(key: K, value: V) !void` - O(log n)
- `get(key: K) ?V` - O(log n)
- `remove(key: K) ?Entry` - O(log n)

**Use Cases**: Simpler implementation than RB-tree with similar performance

---

#### ScapegoatTree
**Path**: `src/containers/trees/scapegoat_tree.zig`

Weight-balanced tree with amortized O(log n) operations.

```zig
const ScapegoatTree = @import("zuda").containers.trees.ScapegoatTree;

pub fn ScapegoatTree(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: K, b: K) std.math.Order,
) type
```

**Key Methods**:
- `init(allocator: Allocator, alpha: f64) Self` - O(1), alpha controls balance
- `insert(key: K, value: V) !void` - O(log n) amortized
- `get(key: K) ?V` - O(log n)
- `remove(key: K) ?Entry` - O(log n) amortized

**Use Cases**: No rotation metadata, good for persistent storage

---

#### BTree
**Path**: `src/containers/trees/btree.zig`

B-tree with configurable branching factor for disk-based storage.

```zig
const BTree = @import("zuda").containers.trees.BTree;

pub fn BTree(
    comptime K: type,
    comptime V: type,
    comptime t: u32, // minimum degree
    comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: K, b: K) std.math.Order,
) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `insert(key: K, value: V) !void` - O(log_t n)
- `get(key: K) ?V` - O(log_t n)
- `remove(key: K) ?Entry` - O(log_t n)
- `rangeIterator(start: K, end: K) RangeIterator` - Range query

**Use Cases**: Databases, file systems, disk-based indexes (see: silica storage layer)

---

#### Trie
**Path**: `src/containers/trees/trie.zig`

Prefix tree for string storage and retrieval.

```zig
const Trie = @import("zuda").containers.trees.Trie;

pub fn Trie(comptime V: type) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `insert(key: []const u8, value: V) !void` - O(m), m = key length
- `get(key: []const u8) ?V` - O(m)
- `hasPrefix(prefix: []const u8) bool` - O(m)
- `remove(key: []const u8) ?V` - O(m)

**Use Cases**: Autocomplete, spell checking, IP routing tables

---

#### RadixTree
**Path**: `src/containers/trees/radix_tree.zig`

Space-optimized trie with path compression.

```zig
const RadixTree = @import("zuda").containers.trees.RadixTree;

pub fn RadixTree(comptime V: type) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `insert(key: []const u8, value: V) !void` - O(m)
- `get(key: []const u8) ?V` - O(m)
- `longestPrefix(key: []const u8) ?[]const u8` - O(m)

**Use Cases**: Routing tables, string dictionaries with common prefixes

---

#### SegmentTree
**Path**: `src/containers/trees/segment_tree.zig`

Tree for range queries and updates.

```zig
const SegmentTree = @import("zuda").containers.trees.SegmentTree;

pub fn SegmentTree(
    comptime T: type,
    comptime Context: type,
    comptime mergeFn: fn (ctx: Context, a: T, b: T) T,
) type
```

**Key Methods**:
- `init(allocator: Allocator, data: []const T) !Self` - O(n)
- `query(left: usize, right: usize) T` - O(log n)
- `update(index: usize, value: T) void` - O(log n)

**Use Cases**: Range sum/min/max queries, competitive programming

---

#### LazySegmentTree
**Path**: `src/containers/trees/lazy_segment_tree.zig`

Segment tree with lazy propagation for range updates.

```zig
const LazySegmentTree = @import("zuda").containers.trees.LazySegmentTree;

pub fn LazySegmentTree(
    comptime T: type,
    comptime Context: type,
    comptime mergeFn: fn (ctx: Context, a: T, b: T) T,
    comptime applyFn: fn (ctx: Context, node: T, lazy: T, len: usize) T,
) type
```

**Key Methods**:
- `init(allocator: Allocator, data: []const T) !Self` - O(n)
- `query(left: usize, right: usize) T` - O(log n)
- `update(left: usize, right: usize, value: T) void` - O(log n)

**Use Cases**: Range updates + range queries, interval modifications

---

#### FenwickTree
**Path**: `src/containers/trees/fenwick_tree.zig`

Binary Indexed Tree for prefix sum queries.

```zig
const FenwickTree = @import("zuda").containers.trees.FenwickTree;

pub fn FenwickTree(comptime T: type) type
```

**Key Methods**:
- `init(allocator: Allocator, size: usize) !Self` - O(n)
- `update(index: usize, delta: T) void` - O(log n)
- `query(index: usize) T` - O(log n), returns prefix sum [0..index]
- `rangeQuery(left: usize, right: usize) T` - O(log n)

**Use Cases**: Cumulative frequency, inversion counting, 2D range sums

---

#### SparseTable
**Path**: `src/containers/trees/sparse_table.zig`

Static data structure for O(1) range minimum/maximum queries.

```zig
const SparseTable = @import("zuda").containers.trees.SparseTable;

pub fn SparseTable(
    comptime T: type,
    comptime Context: type,
    comptime selectFn: fn (ctx: Context, a: T, b: T) T,
) type
```

**Key Methods**:
- `init(allocator: Allocator, data: []const T) !Self` - O(n log n)
- `query(left: usize, right: usize) T` - O(1)

**Use Cases**: RMQ (Range Minimum Query), LCA (Lowest Common Ancestor), static arrays

---

#### IntervalTree
**Path**: `src/containers/trees/interval_tree.zig`

Augmented tree for interval overlap queries.

```zig
const IntervalTree = @import("zuda").containers.trees.IntervalTree;

pub fn IntervalTree(comptime T: type) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `insert(start: T, end: T, value: V) !void` - O(log n)
- `findOverlapping(start: T, end: T) ![]Interval` - O(log n + k), k = results
- `remove(interval: Interval) void` - O(log n)

**Use Cases**: Scheduling, genomics, windowing systems

---

#### VanEmdeBoasTree
**Path**: `src/containers/trees/van_emde_boas_tree.zig`

Integer-keyed tree with O(log log U) operations.

```zig
const VanEmdeBoasTree = @import("zuda").containers.trees.VanEmdeBoasTree;

pub fn VanEmdeBoasTree(comptime universe_size: u64) type
```

**Key Methods**:
- `init(allocator: Allocator) !Self` - O(U)
- `insert(key: u64) !void` - O(log log U)
- `remove(key: u64) void` - O(log log U)
- `successor(key: u64) ?u64` - O(log log U)
- `predecessor(key: u64) ?u64` - O(log log U)

**Use Cases**: Fast integer operations, routing, priority queues with small universe

---

### Graphs

#### AdjacencyList
**Path**: `src/containers/graphs/adjacency_list.zig`

Graph representation using neighbor lists.

```zig
const AdjacencyList = @import("zuda").containers.graphs.AdjacencyList;

pub fn AdjacencyList(comptime V: type, comptime E: type, comptime directed: bool) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `addVertex(data: V) !u32` - O(1), returns vertex ID
- `addEdge(from: u32, to: u32, weight: E) !void` - O(1)
- `neighbors(vertex: u32) []Edge` - O(1)
- `removeEdge(from: u32, to: u32) void` - O(degree)

**Use Cases**: General-purpose graph representation, sparse graphs (see: zr DAG)

---

#### AdjacencyMatrix
**Path**: `src/containers/graphs/adjacency_matrix.zig`

Graph representation using 2D matrix.

```zig
const AdjacencyMatrix = @import("zuda").containers.graphs.AdjacencyMatrix;

pub fn AdjacencyMatrix(comptime E: type, comptime directed: bool) type
```

**Key Methods**:
- `init(allocator: Allocator, num_vertices: usize) !Self` - O(V²)
- `addEdge(from: u32, to: u32, weight: E) void` - O(1)
- `hasEdge(from: u32, to: u32) bool` - O(1)
- `getWeight(from: u32, to: u32) ?E` - O(1)

**Use Cases**: Dense graphs, O(1) edge lookup, Floyd-Warshall algorithm

---

#### CompressedSparseRow
**Path**: `src/containers/graphs/compressed_sparse_row.zig`

Space-efficient graph format for read-only graphs.

```zig
const CompressedSparseRow = @import("zuda").containers.graphs.CompressedSparseRow;

pub fn CompressedSparseRow(comptime E: type, comptime directed: bool) type
```

**Key Methods**:
- `fromAdjacencyList(adj_list: AdjacencyList) !Self` - O(V + E)
- `neighbors(vertex: u32) []Edge` - O(1)
- `degree(vertex: u32) usize` - O(1)

**Use Cases**: Graph analytics, PageRank, read-only graph databases

---

#### EdgeList
**Path**: `src/containers/graphs/edge_list.zig`

Simple list of edges, useful for algorithms like Kruskal's MST.

```zig
const EdgeList = @import("zuda").containers.graphs.EdgeList;

pub fn EdgeList(comptime V: type, comptime E: type, comptime directed: bool) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `addEdge(from: u32, to: u32, weight: E) !void` - O(1)
- `sortByWeight() void` - O(E log E)

**Use Cases**: Kruskal's algorithm, edge-centric algorithms

---

### Spatial

#### KDTree
**Path**: `src/containers/spatial/kd_tree.zig`

K-dimensional tree for spatial queries.

```zig
const KDTree = @import("zuda").containers.spatial.KDTree;

pub fn KDTree(comptime K: u32, comptime T: type) type
```

**Key Methods**:
- `init(allocator: Allocator, points: []const [K]T) !Self` - O(n log n)
- `nearest(point: [K]T) ?[K]T` - O(log n) average
- `rangeSearch(min: [K]T, max: [K]T) ![][K]T` - O(n^(1-1/k) + m)

**Use Cases**: Nearest neighbor search, spatial indexing, collision detection

---

#### RTree
**Path**: `src/containers/spatial/r_tree.zig`

Rectangle tree for spatial indexing of bounding boxes.

```zig
const RTree = @import("zuda").containers.spatial.RTree;

pub fn RTree(comptime T: type, comptime max_children: u32) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `insert(bbox: BBox, data: T) !void` - O(log n)
- `search(bbox: BBox) ![]T` - O(log n + k)
- `remove(bbox: BBox) bool` - O(log n)

**Use Cases**: GIS systems, game engines, spatial databases

---

#### QuadTree
**Path**: `src/containers/spatial/quad_tree.zig`

2D spatial partitioning tree.

```zig
const QuadTree = @import("zuda").containers.spatial.QuadTree;

pub fn QuadTree(comptime T: type) type
```

**Key Methods**:
- `init(allocator: Allocator, bounds: Rect) Self` - O(1)
- `insert(point: Point, data: T) !void` - O(log n)
- `query(range: Rect) ![]T` - O(log n + k)

**Use Cases**: 2D collision detection, image compression, geographic data

---

#### OctTree
**Path**: `src/containers/spatial/octtree.zig`

3D spatial partitioning tree.

```zig
const OctTree = @import("zuda").containers.spatial.OctTree;

pub fn OctTree(comptime T: type) type
```

**Key Methods**:
- `init(allocator: Allocator, bounds: Box) Self` - O(1)
- `insert(point: Point3D, data: T) !void` - O(log n)
- `query(range: Box) ![]T` - O(log n + k)

**Use Cases**: 3D collision detection, voxel rendering, point clouds

---

### Strings

#### SuffixArray
**Path**: `src/containers/strings/suffix_array.zig`

Array of sorted suffixes for fast string search.

```zig
const SuffixArray = @import("zuda").containers.strings.SuffixArray;

pub fn SuffixArray() type
```

**Key Methods**:
- `init(allocator: Allocator, text: []const u8) !Self` - O(n log n)
- `find(pattern: []const u8) ?usize` - O(m log n + occ)
- `lcp() ![]usize` - O(n), longest common prefix array

**Use Cases**: Pattern matching, string compression, bioinformatics

---

#### SuffixTree
**Path**: `src/containers/strings/suffix_tree.zig`

Tree of all suffixes for O(m) pattern matching.

```zig
const SuffixTree = @import("zuda").containers.strings.SuffixTree;

pub fn SuffixTree() type
```

**Key Methods**:
- `init(allocator: Allocator, text: []const u8) !Self` - O(n)
- `find(pattern: []const u8) bool` - O(m)
- `longestRepeatedSubstring() []const u8` - O(n)

**Use Cases**: Linear-time pattern matching, longest common substring

---

### Caches

#### LRUCache
**Path**: `src/containers/cache/lru_cache.zig`

Least Recently Used cache with O(1) operations.

```zig
const LRUCache = @import("zuda").containers.cache.LRUCache;

pub fn LRUCache(comptime K: type, comptime V: type) type
```

**Key Methods**:
- `init(allocator: Allocator, capacity: usize) !Self` - O(1)
- `put(key: K, value: V) !?V` - O(1)
- `get(key: K) ?V` - O(1), promotes to most recent

**Use Cases**: Page replacement, web caches, memoization (see: silica buffer pool)

---

#### LFUCache
**Path**: `src/containers/cache/lfu_cache.zig`

Least Frequently Used cache.

```zig
const LFUCache = @import("zuda").containers.cache.LFUCache;

pub fn LFUCache(comptime K: type, comptime V: type) type
```

**Key Methods**:
- `init(allocator: Allocator, capacity: usize) !Self` - O(1)
- `put(key: K, value: V) !?V` - O(1)
- `get(key: K) ?V` - O(1), increments frequency

**Use Cases**: Frequency-based eviction, CDN caches

---

#### ARCCache
**Path**: `src/containers/cache/arc_cache.zig`

Adaptive Replacement Cache balancing recency and frequency.

```zig
const ARCCache = @import("zuda").containers.cache.ARCCache;

pub fn ARCCache(comptime K: type, comptime V: type) type
```

**Key Methods**:
- `init(allocator: Allocator, capacity: usize) !Self` - O(1)
- `put(key: K, value: V) !?V` - O(1)
- `get(key: K) ?V` - O(1)

**Use Cases**: Self-tuning caches, databases, storage systems

---

### Probabilistic

#### BloomFilter
**Path**: `src/containers/probabilistic/bloom_filter.zig`

Space-efficient probabilistic set membership test.

```zig
const BloomFilter = @import("zuda").containers.probabilistic.BloomFilter;

pub fn BloomFilter() type
```

**Key Methods**:
- `init(allocator: Allocator, capacity: usize, false_positive_rate: f64) !Self` - O(n)
- `insert(item: []const u8) void` - O(k), k = hash functions
- `contains(item: []const u8) bool` - O(k), may have false positives

**Use Cases**: Spell checkers, duplicate detection, network routers

---

#### CountMinSketch
**Path**: `src/containers/probabilistic/count_min_sketch.zig`

Probabilistic frequency counting with bounded error.

```zig
const CountMinSketch = @import("zuda").containers.probabilistic.CountMinSketch;

pub fn CountMinSketch() type
```

**Key Methods**:
- `init(allocator: Allocator, width: usize, depth: usize) !Self` - O(width * depth)
- `add(item: []const u8, count: u64) void` - O(depth)
- `estimate(item: []const u8) u64` - O(depth), overestimates frequency

**Use Cases**: Heavy hitters, network traffic analysis, streaming data

---

#### HyperLogLog
**Path**: `src/containers/probabilistic/hyperloglog.zig`

Cardinality estimation with ~2% error using O(log log n) space.

```zig
const HyperLogLog = @import("zuda").containers.probabilistic.HyperLogLog;

pub fn HyperLogLog(comptime precision: u5) type
```

**Key Methods**:
- `init(allocator: Allocator) !Self` - O(2^precision)
- `add(item: []const u8) void` - O(1)
- `count() u64` - O(2^precision), estimated cardinality

**Use Cases**: Unique visitor counting, database query optimization (see: zoltraak PFCOUNT)

---

#### CuckooFilter
**Path**: `src/containers/probabilistic/cuckoo_filter.zig`

Bloom filter alternative supporting deletions.

```zig
const CuckooFilter = @import("zuda").containers.probabilistic.CuckooFilter;

pub fn CuckooFilter() type
```

**Key Methods**:
- `init(allocator: Allocator, capacity: usize) !Self` - O(n)
- `insert(item: []const u8) !void` - O(1) amortized
- `contains(item: []const u8) bool` - O(1)
- `remove(item: []const u8) bool` - O(1)

**Use Cases**: Sets with deletions, distributed systems

---

#### MinHash
**Path**: `src/containers/probabilistic/minhash.zig`

Locality-sensitive hashing for estimating Jaccard similarity.

```zig
const MinHash = @import("zuda").containers.probabilistic.MinHash;

pub fn MinHash(comptime num_hashes: u32) type
```

**Key Methods**:
- `init(allocator: Allocator) !Self` - O(1)
- `add(item: []const u8) void` - O(num_hashes)
- `similarity(other: *const Self) f64` - O(num_hashes), Jaccard estimate

**Use Cases**: Document similarity, duplicate detection, recommendation systems

---

### Persistent

#### PersistentArray
**Path**: `src/containers/persistent/persistent_array.zig`

Immutable array with O(log n) updates via path copying.

```zig
const PersistentArray = @import("zuda").containers.persistent.PersistentArray;

pub fn PersistentArray(comptime T: type, comptime branching: u32) type
```

**Key Methods**:
- `init(allocator: Allocator, size: usize, default: T) !Self` - O(n)
- `get(index: usize) T` - O(log n)
- `set(index: usize, value: T) !Self` - O(log n), returns new version

**Use Cases**: Functional programming, version control, undo/redo

---

#### PersistentRBTree
**Path**: `src/containers/persistent/persistent_rbtree.zig`

Immutable red-black tree with structural sharing.

```zig
const PersistentRBTree = @import("zuda").containers.persistent.PersistentRBTree;

pub fn PersistentRBTree(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: K, b: K) std.math.Order,
) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `insert(key: K, value: V) !Self` - O(log n), returns new version
- `get(key: K) ?V` - O(log n)
- `remove(key: K) !Self` - O(log n), returns new version

**Use Cases**: Immutable maps, temporal databases, event sourcing

**⚠️ Known Issue**: Memory leak when multiple versions are kept alive (needs ref-counting)

---

### Specialized

#### DisjointSet
**Path**: `src/containers/specialized/disjoint_set.zig`

Union-Find data structure with path compression.

```zig
const DisjointSet = @import("zuda").containers.specialized.DisjointSet;

pub fn DisjointSet() type
```

**Key Methods**:
- `init(allocator: Allocator, size: usize) !Self` - O(n)
- `find(x: usize) usize` - O(α(n)), α = inverse Ackermann
- `unite(x: usize, y: usize) void` - O(α(n))
- `connected(x: usize, y: usize) bool` - O(α(n))

**Use Cases**: Kruskal's MST, connected components, cycle detection (see: zr cycle detection)

---

#### Rope
**Path**: `src/containers/specialized/rope.zig`

Efficient string for insertions/deletions in large texts.

```zig
const Rope = @import("zuda").containers.specialized.Rope;

pub fn Rope() type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `fromSlice(text: []const u8) !Self` - O(n)
- `insert(index: usize, text: []const u8) !void` - O(log n)
- `delete(start: usize, len: usize) void` - O(log n)
- `slice(start: usize, end: usize) ![]u8` - O(log n + m)

**Use Cases**: Text editors, large document manipulation

---

#### BK-Tree
**Path**: `src/containers/specialized/bk_tree.zig`

Metric tree for fuzzy string matching.

```zig
const BKTree = @import("zuda").containers.specialized.BKTree;

pub fn BKTree(comptime T: type, comptime distanceFn: fn (a: T, b: T) usize) type
```

**Key Methods**:
- `init(allocator: Allocator) Self` - O(1)
- `insert(item: T) !void` - O(log n) average
- `search(item: T, max_distance: usize) ![]T` - O(log n) for small distances

**Use Cases**: Spell checking, fuzzy search, DNA sequence matching

---

#### DancingLinks
**Path**: `src/containers/exotic/dancing_links.zig`

Knuth's Algorithm X for exact cover problems.

```zig
const DancingLinks = @import("zuda").containers.exotic.DancingLinks;

pub fn DancingLinks() type
```

**Key Methods**:
- `init(allocator: Allocator, num_columns: usize) !Self` - O(1)
- `addRow(columns: []const usize) !void` - O(len)
- `solve() !?[][]usize` - Backtracking search

**Use Cases**: Sudoku solvers, N-Queens, tiling problems, constraint satisfaction

---

## Algorithms

### Sorting

| Algorithm | Path | Time Complexity | Space | Stable | Use Case |
|-----------|------|-----------------|-------|--------|----------|
| **TimSort** | `algorithms/sorting/timsort.zig` | O(n log n) | O(n) | Yes | General-purpose, real-world data |
| **IntroSort** | `algorithms/sorting/introsort.zig` | O(n log n) | O(log n) | No | Fast pivot-based sort |
| **RadixSort** | `algorithms/sorting/radixsort.zig` | O(d·n) | O(n+k) | Yes | Integer/string sorting |
| **CountingSort** | `algorithms/sorting/countingsort.zig` | O(n+k) | O(k) | Yes | Small integer range |
| **MergeSort** | `algorithms/sorting/mergesort.zig` | O(n log n) | O(n) | Yes | Linked lists, external sort |
| **BlockSort** | `algorithms/sorting/blocksort.zig` | O(n log n) | O(1) | Yes | In-place stable sort |

**Example**:
```zig
const TimSort = @import("zuda").algorithms.sorting.TimSort;

var items = [_]i32{5, 2, 8, 1, 9};
try TimSort.sort(i32, &items, {}, std.sort.asc(i32));
```

---

### Graph Algorithms

#### Traversal

| Algorithm | Path | Time | Space | Use Case |
|-----------|------|------|-------|----------|
| **BFS** | `algorithms/graph/bfs.zig` | O(V+E) | O(V) | Shortest path (unweighted), level-order |
| **DFS** | `algorithms/graph/dfs.zig` | O(V+E) | O(V) | Cycle detection, topological sort |

#### Shortest Paths

| Algorithm | Path | Time | Space | Graph Type | Use Case |
|-----------|------|------|-------|------------|----------|
| **Dijkstra** | `algorithms/graph/dijkstra.zig` | O((V+E) log V) | O(V) | Non-negative weights | SSSP on roads, networks |
| **Bellman-Ford** | `algorithms/graph/bellman_ford.zig` | O(VE) | O(V) | Any weights | Negative edges, cycle detection |
| **A\*** | `algorithms/graph/a_star.zig` | O(E) | O(V) | Heuristic-guided | Pathfinding with heuristic |
| **Floyd-Warshall** | `algorithms/graph/floyd_warshall.zig` | O(V³) | O(V²) | Any weights | APSP (All-Pairs Shortest Paths) |
| **Johnson** | `algorithms/graph/johnson.zig` | O(VE + V² log V) | O(V²) | Any weights | APSP for sparse graphs |

#### MST (Minimum Spanning Tree)

| Algorithm | Path | Time | Space | Use Case |
|-----------|------|------|-------|----------|
| **Kruskal** | `algorithms/graph/kruskal.zig` | O(E log E) | O(V) | Sparse graphs, edge-centric |
| **Prim** | `algorithms/graph/prim.zig` | O(E log V) | O(V) | Dense graphs, vertex-centric |
| **Borůvka** | `algorithms/graph/boruvka.zig` | O(E log V) | O(V) | Parallel MST |

#### Connectivity

| Algorithm | Path | Time | Space | Use Case |
|-----------|------|------|-------|----------|
| **Tarjan SCC** | `algorithms/graph/tarjan_scc.zig` | O(V+E) | O(V) | Strongly connected components |
| **Kosaraju SCC** | `algorithms/graph/kosaraju_scc.zig` | O(V+E) | O(V) | Strongly connected components |
| **Bridges** | `algorithms/graph/bridges.zig` | O(V+E) | O(V) | Critical edges |
| **Articulation Points** | `algorithms/graph/articulation_points.zig` | O(V+E) | O(V) | Critical vertices |
| **Topological Sort** | `algorithms/graph/topological_sort.zig` | O(V+E) | O(V) | DAG ordering (see: zr task execution) |

#### Flow & Matching

| Algorithm | Path | Time | Space | Use Case |
|-----------|------|------|-------|----------|
| **Edmonds-Karp** | `algorithms/graph/edmonds_karp.zig` | O(VE²) | O(V+E) | Max flow (BFS-based) |
| **Dinic** | `algorithms/graph/dinic.zig` | O(V²E) | O(V+E) | Max flow (faster for unit capacity) |
| **Push-Relabel** | `algorithms/graph/push_relabel.zig` | O(V³) | O(V+E) | Max flow (best for dense) |
| **Hopcroft-Karp** | `algorithms/graph/hopcroft_karp.zig` | O(E√V) | O(V+E) | Maximum bipartite matching |
| **Hungarian** | `algorithms/graph/hungarian.zig` | O(V³) | O(V²) | Min-cost bipartite matching |

**Example**:
```zig
const Dijkstra = @import("zuda").algorithms.graph.Dijkstra;

var graph = AdjacencyList(void, f64, true).init(allocator);
// ... add vertices and edges ...

const distances = try Dijkstra.shortestPaths(allocator, &graph, source_vertex);
defer allocator.free(distances);
```

---

### String Algorithms

| Algorithm | Path | Time | Space | Use Case |
|-----------|------|------|-------|----------|
| **KMP** | `algorithms/string/kmp.zig` | O(n+m) | O(m) | Pattern matching |
| **Boyer-Moore** | `algorithms/string/boyer_moore.zig` | O(n/m) best | O(σ) | Fast string search |
| **Rabin-Karp** | `algorithms/string/rabin_karp.zig` | O(n+m) avg | O(1) | Multiple pattern search |
| **Aho-Corasick** | `algorithms/string/aho_corasick.zig` | O(n+m+z) | O(σ·m) | Multi-pattern matching |
| **Z-Algorithm** | `algorithms/string/z_algorithm.zig` | O(n) | O(n) | Pattern matching, string hashing |

**Example**:
```zig
const KMP = @import("zuda").algorithms.string.KMP;

const text = "hello world";
const pattern = "world";
const index = KMP.search(text, pattern); // returns 6
```

---

### Dynamic Programming

| Algorithm | Path | Time | Space | Use Case |
|-----------|------|------|-------|----------|
| **LIS** (Longest Increasing Subsequence) | `algorithms/dynamic_programming/lis.zig` | O(n log n) | O(n) | Sequence analysis |
| **LCS** (Longest Common Subsequence) | `algorithms/dynamic_programming/lcs.zig` | O(nm) | O(nm) | Diff tools, genetics |
| **Edit Distance** | `algorithms/dynamic_programming/edit_distance.zig` | O(nm) | O(nm) | Spell checking, DNA alignment (see: zr Levenshtein) |
| **Knapsack** | `algorithms/dynamic_programming/knapsack.zig` | O(nW) | O(W) | Resource allocation |
| **Binary Search** | `algorithms/dynamic_programming/binary_search.zig` | O(log n) | O(1) | Sorted array search, variants |

**Example**:
```zig
const EditDistance = @import("zuda").algorithms.dynamic_programming.EditDistance;

const dist = try EditDistance.levenshtein(allocator, "kitten", "sitting"); // returns 3
```

---

### Geometry

| Algorithm | Path | Time | Space | Use Case |
|-----------|------|------|-------|----------|
| **Convex Hull** (Graham, Jarvis) | `algorithms/geometry/convex_hull.zig` | O(n log n) | O(n) | Collision detection, clustering |
| **Closest Pair** | `algorithms/geometry/closest_pair.zig` | O(n log n) | O(n) | Nearest neighbor |
| **Haversine Distance** | `algorithms/geometry/haversine.zig` | O(1) | O(1) | GPS distance (see: zoltraak GEODIST) |
| **Geohash** | `algorithms/geometry/geohash.zig` | O(precision) | O(1) | Spatial indexing (see: zoltraak GEO*) |

**Example**:
```zig
const Haversine = @import("zuda").algorithms.geometry.Haversine;

const dist_km = Haversine.distance(
    .{ .lat = 40.7128, .lon = -74.0060 }, // NYC
    .{ .lat = 51.5074, .lon = -0.1278 },  // London
); // returns ~5570 km
```

---

### Math

| Algorithm | Path | Time | Space | Use Case |
|-----------|------|------|-------|----------|
| **GCD/LCM** | `algorithms/math/gcd.zig` | O(log n) | O(1) | Number theory, fractions |
| **ModExp** | `algorithms/math/modexp.zig` | O(log n) | O(1) | Cryptography, RSA |
| **Miller-Rabin** | `algorithms/math/primality.zig` | O(k log³ n) | O(1) | Primality testing |
| **Sieve of Eratosthenes** | `algorithms/math/sieve.zig` | O(n log log n) | O(n) | Prime generation |
| **Chinese Remainder Theorem** | `algorithms/math/crt.zig` | O(n log n) | O(1) | Modular equations |
| **NTT** (Number Theoretic Transform) | `algorithms/math/ntt.zig` | O(n log n) | O(n) | Fast polynomial multiplication |

**Example**:
```zig
const MillerRabin = @import("zuda").algorithms.math.MillerRabin;

const is_prime = MillerRabin.isPrime(982451653, 10); // true, 10 rounds
```

---

## FFI

### C API
**Path**: `src/ffi/c_api.zig`
**Header**: `include/zuda.h`

Provides C-compatible interface for selected zuda containers.

**Available Containers**:
- `zuda_hash_map_*` - RobinHoodHashMap with string keys
- `zuda_skip_list_*` - SkipList with integer keys
- `zuda_bloom_filter_*` - BloomFilter

**Example (C)**:
```c
#include <zuda.h>

ZudaHashMap* map = zuda_hash_map_create();
zuda_hash_map_put(map, "key", "value");
const char* val = zuda_hash_map_get(map, "key");
zuda_hash_map_destroy(map);
```

### Language Bindings

**Python** (`examples/python_bindings.py`):
```python
from zuda import HashMap

hm = HashMap()
hm.put("name", "zuda")
print(hm.get("name"))  # "zuda"
```

**Node.js** (`examples/nodejs_bindings.js`):
```javascript
const zuda = require('./zuda_bindings');

const map = new zuda.HashMap();
map.put('key', 'value');
console.log(map.get('key')); // "value"
```

**See**: `examples/FFI_README.md` for complete FFI usage guide.

---

## Performance Targets (v0.5.0)

| Metric | Target | Status |
|--------|--------|--------|
| RedBlackTree insert | ≤ 200 ns/op | ✅ |
| RedBlackTree lookup | ≤ 150 ns/op | ✅ |
| FibonacciHeap decrease-key | ≤ 50 ns amortized | ✅ |
| BloomFilter lookup | ≥ 100M ops/sec | ✅ |

---

## Build & Test

```bash
# Build library
zig build

# Run all tests (701 tests)
zig build test

# Build with C API (generates libzuda.a + zuda.h)
zig build -Dshared=true

# Cross-compile
zig build -Dtarget=x86_64-linux-gnu
```

---

## Version History

- **v0.5.0** (2026-03-13): Phase 5 complete - C API, FFI bindings, persistent structures
- **v0.4.0**: Phase 4 complete - Probabilistic structures, caches, geometry, math
- **v0.3.0**: Phase 3 complete - Graph algorithms
- **v0.2.0**: Phase 2 complete - Trees, spatial structures
- **v0.1.0**: Phase 1 complete - Foundations (lists, queues, heaps, hashing)

---

**Next**: [Algorithm Explainers](ALGORITHMS.md) | [Decision Tree Guide](GUIDE.md) | [Getting Started](GETTING_STARTED.md)
