# Data Structure Decision Guide

> Choose the right data structure for your use case

## Quick Decision Tree

```
┌─ Need to store data? ─┐
│                        │
├─ Key-Value pairs? ─────┼─→ See [Maps & Sets](#maps--sets)
│                        │
├─ Ordered sequence? ────┼─→ See [Lists & Sequences](#lists--sequences)
│                        │
├─ Priority-based? ──────┼─→ See [Heaps & Priority Queues](#heaps--priority-queues)
│                        │
├─ Relationships/Network?┼─→ See [Graphs](#graphs)
│                        │
├─ Spatial/Geometric? ───┼─→ See [Spatial Structures](#spatial-structures)
│                        │
├─ Set membership test? ─┼─→ See [Probabilistic](#probabilistic-structures)
│                        │
├─ String operations? ───┼─→ See [Strings](#string-structures)
│                        │
├─ Immutable/Versioned? ─┼─→ See [Persistent](#persistent-structures)
│                        │
└─ Algorithm needed? ────┴─→ See [Algorithms](#algorithm-selection)
```

---

## Maps & Sets

**Use Case**: Store key-value pairs with fast lookup, insert, delete

### Decision Matrix

| Requirement | Recommendation | Why |
|-------------|----------------|-----|
| **General-purpose map** | `std.HashMap` or `RobinHoodHashMap` | Balanced performance, familiar API |
| **Ordered iteration needed** | `RedBlackTree` | O(log n) ops + in-order traversal |
| **Predictable worst-case** | `CuckooHashMap` | O(1) worst-case lookup |
| **Huge datasets (>1M keys)** | `SwissTable` | SIMD-friendly, cache-optimized |
| **Concurrent access** | `ConcurrentSkipList` | Lock-free ordered map |
| **Immutable/versioned** | `PersistentHashMap` (HAMT) | Structural sharing |
| **Distributed systems** | `ConsistentHashRing` | Load balancing, sharding |
| **Temporal locality** | `SplayTree` | Recently accessed items cached |
| **Exact key lookup unnecessary** | `BloomFilter` | Space-efficient membership test |

### Detailed Comparison

#### Hash Tables

| Data Structure | Lookup | Insert | Delete | Iteration | Memory | Best For |
|----------------|--------|--------|--------|-----------|--------|----------|
| `RobinHoodHashMap` | O(1) avg | O(1) avg | O(1) avg | Unordered | Medium | General use |
| `CuckooHashMap` | **O(1) worst** | O(1) amort | O(1) worst | Unordered | High | Real-time systems |
| `SwissTable` | O(1) avg | O(1) avg | O(1) avg | Unordered | Low | Large datasets |
| `ConsistentHashRing` | O(log n) | O(r log n) | O(r log n) | Ordered | Medium | Distributed hash |

*r = number of replicas (virtual nodes)*

#### Balanced Trees

| Data Structure | Lookup | Insert | Delete | Iteration | Memory | Best For |
|----------------|--------|--------|--------|-----------|--------|----------|
| `RedBlackTree` | O(log n) | O(log n) | O(log n) | In-order | Medium | General ordered map |
| `AVLTree` | **O(log n)** | O(log n) | O(log n) | In-order | Medium | Lookup-heavy |
| `SplayTree` | O(log n) amort | O(log n) amort | O(log n) amort | In-order | Low | Access locality |
| `SkipList` | O(log n) avg | O(log n) avg | O(log n) avg | In-order | High | Concurrent-friendly |
| `BTree(t)` | O(log_t n) | O(log_t n) | O(log_t n) | In-order | Low | Disk-based |

**Example**:
```zig
// Ordered map with range queries
const RBTree = @import("zuda").containers.trees.RedBlackTree;
var map = RBTree([]const u8, u32, void, stringCompare).init(allocator);
try map.insert("alice", 30);
try map.insert("bob", 25);

// Iterate in sorted order
var it = map.iterator();
while (it.next()) |entry| {
    std.debug.print("{s}: {}\n", .{entry.key, entry.value});
}
```

---

## Lists & Sequences

**Use Case**: Ordered collection with positional access

### Decision Matrix

| Requirement | Recommendation | Why |
|-------------|----------------|-----|
| **Random access by index** | `std.ArrayList` | O(1) indexing |
| **Frequent inserts/deletes** | `std.DoublyLinkedList` | O(1) at known position |
| **Cache-friendly linked list** | `UnrolledLinkedList` | Fewer allocations, better locality |
| **Memory-constrained** | `XorLinkedList` | 1 pointer per node vs 2 |
| **Large text editing** | `Rope` | O(log n) insert/delete anywhere |
| **Queue (FIFO)** | `Deque` | O(1) push/pop both ends |
| **Stack (LIFO)** | `std.ArrayList` | O(1) push/pop back |
| **Concurrent queue** | `LockFreeQueue` | Lock-free FIFO |
| **Work stealing** | `WorkStealingDeque` | Owner push/pop, thieves steal |

### Detailed Comparison

| Data Structure | Access | Insert Front | Insert Back | Insert Middle | Memory | Best For |
|----------------|--------|--------------|-------------|---------------|--------|----------|
| `std.ArrayList` | **O(1)** | O(n) | **O(1)** amort | O(n) | Compact | Random access |
| `Deque` | O(n) | **O(1)** amort | **O(1)** amort | O(n) | Medium | Both-end ops |
| `UnrolledLinkedList` | O(n/B) | **O(1)** | **O(1)** | O(n/B) | Medium | Cache-friendly |
| `XorLinkedList` | O(n) | **O(1)** | **O(1)** | O(1)* | **Low** | Memory-constrained |
| `Rope` | O(log n) | O(log n) | O(log n) | **O(log n)** | High | Large text |

*\*with node pointer*

**Example**:
```zig
// Queue for BFS
const Deque = @import("zuda").containers.queues.Deque;
var queue = Deque(u32).init(allocator);
defer queue.deinit();

try queue.pushBack(1);
try queue.pushBack(2);
const first = queue.popFront(); // 1
```

---

## Heaps & Priority Queues

**Use Case**: Maintain collection with efficient min/max extraction

### Decision Matrix

| Requirement | Recommendation | Why |
|-------------|----------------|-----|
| **General priority queue** | `std.PriorityQueue` (binary heap) | Simple, efficient |
| **Frequent decrease-key** | `FibonacciHeap` | O(1) decrease-key amortized |
| **Frequent merge** | `BinomialHeap` | O(log n) merge |
| **Simple decrease-key** | `PairingHeap` | Easier than Fibonacci |
| **Better cache performance** | `DaryHeap(4)` | Fewer cache misses |
| **Huge integer universe** | `VanEmdeBoasTree` | O(log log U) operations |

### Detailed Comparison

| Data Structure | Insert | Find-Min | Extract-Min | Decrease-Key | Merge | Best For |
|----------------|--------|----------|-------------|--------------|-------|----------|
| `std.PriorityQueue` (Binary) | O(log n) | O(1) | O(log n) | O(log n) | O(n) | General use |
| `DaryHeap(4)` | O(log_4 n) | O(1) | O(4·log_4 n) | O(log_4 n) | O(n) | Cache-optimized |
| `FibonacciHeap` | **O(1)** | O(1) | O(log n) | **O(1)** amort | **O(1)** | Dijkstra, Prim |
| `BinomialHeap` | O(log n) | O(log n) | O(log n) | O(log n) | **O(log n)** | Frequent merges |
| `PairingHeap` | **O(1)** | O(1) | O(log n) amort | O(log n) amort | **O(1)** | Simpler Fibonacci |
| `VanEmdeBoasTree` | **O(log log U)** | **O(log log U)** | **O(log log U)** | **O(log log U)** | N/A | Small integer universe |

**Example**:
```zig
// Dijkstra's algorithm with Fibonacci heap
const FibHeap = @import("zuda").containers.heaps.FibonacciHeap;
var heap = FibHeap(Node, void, nodeCompare).init(allocator);
defer heap.deinit();

var node_handle = try heap.insert(Node{ .id = 0, .dist = 0 });
// ... later ...
try heap.decreaseKey(node_handle, Node{ .id = 0, .dist = 5 }); // O(1)!
```

**When to Use Each**:
- **Dijkstra's algorithm**: `FibonacciHeap` (frequent decrease-key)
- **Prim's MST**: `FibonacciHeap` or `BinomialHeap`
- **Heapsort**: `std.PriorityQueue` (binary heap)
- **Event simulation**: `std.PriorityQueue` or `DaryHeap`
- **Mergeable heaps**: `BinomialHeap` or `FibonacciHeap`

---

## Graphs

**Use Case**: Represent relationships between entities

### Representation Selection

| Graph Type | Nodes | Edges | Operations | Best Representation |
|------------|-------|-------|------------|---------------------|
| **Sparse (E ≈ V)** | Any | Low density | Add vertex, iterate neighbors | `AdjacencyList` |
| **Dense (E ≈ V²)** | Fixed | High density | Check edge exists | `AdjacencyMatrix` |
| **Read-only** | Fixed | Any | Analytics, PageRank | `CompressedSparseRow` |
| **Edge-centric** | Any | Any | Kruskal's MST | `EdgeList` |

### Detailed Comparison

| Representation | Space | Add Edge | Has Edge | Neighbors | Best For |
|----------------|-------|----------|----------|-----------|----------|
| `AdjacencyList` | O(V+E) | O(1) | O(degree) | **O(1)** | Sparse graphs |
| `AdjacencyMatrix` | **O(V²)** | **O(1)** | **O(1)** | O(V) | Dense graphs |
| `CompressedSparseRow` | **O(V+E)** | N/A (read-only) | O(log degree) | **O(1)** | Graph analytics |
| `EdgeList` | **O(E)** | O(1) | O(E) | O(E) | Edge algorithms |

**Example**:
```zig
// Sparse graph for BFS/DFS
const AdjList = @import("zuda").containers.graphs.AdjacencyList;
var graph = AdjList(void, f64, true).init(allocator); // directed, weighted

const v0 = try graph.addVertex({});
const v1 = try graph.addVertex({});
try graph.addEdge(v0, v1, 2.5);

// Dense graph for all-pairs shortest paths
const AdjMatrix = @import("zuda").containers.graphs.AdjacencyMatrix;
var dense = try AdjMatrix(f64, false).init(allocator, 100); // 100 vertices
dense.addEdge(0, 1, 1.0);
if (dense.hasEdge(0, 1)) { ... } // O(1) check
```

### Graph Algorithms

See [ALGORITHMS.md](ALGORITHMS.md) for detailed algorithm selection:

| Problem | Algorithm | Time | Best For |
|---------|-----------|------|----------|
| **Shortest path (unweighted)** | BFS | O(V+E) | Simple, fast |
| **Shortest path (non-negative weights)** | Dijkstra | O((V+E) log V) | Most common |
| **Shortest path (negative weights)** | Bellman-Ford | O(VE) | Negative edges |
| **Shortest path (heuristic)** | A* | O(E) | Pathfinding with goal |
| **All-pairs shortest paths (sparse)** | Johnson | O(VE + V² log V) | Sparse graphs |
| **All-pairs shortest paths (dense)** | Floyd-Warshall | O(V³) | Dense graphs |
| **MST (sparse)** | Kruskal | O(E log E) | Edge-based |
| **MST (dense)** | Prim | O(E log V) | Vertex-based |
| **Strongly connected components** | Tarjan or Kosaraju | O(V+E) | Both good |
| **Topological sort** | Kahn (BFS) or DFS | O(V+E) | DAGs |
| **Max flow** | Dinic or Push-Relabel | O(V²E) / O(V³) | Network flow |
| **Bipartite matching** | Hopcroft-Karp | O(E√V) | Matching |

---

## Spatial Structures

**Use Case**: Index geometric or spatial data

### Decision Matrix

| Dimension | Points | Query Type | Recommendation | Why |
|-----------|--------|------------|----------------|-----|
| **2D** | Dynamic | Nearest neighbor | `KDTree` | Fast NN in 2-3D |
| **2D** | Dynamic | Range query | `QuadTree` | Recursive subdivision |
| **3D** | Dynamic | Range query | `OctTree` | 3D QuadTree |
| **2D** | Dynamic | Bounding boxes | `RTree` | GIS, game engines |
| **Any D** | Static | NN / Range | `KDTree` | Good up to ~20D |

### Detailed Comparison

| Data Structure | Dimensions | Insert | NN Search | Range Query | Best For |
|----------------|------------|--------|-----------|-------------|----------|
| `KDTree` | Any (best <20) | O(log n) avg | **O(log n) avg** | O(n^(1-1/k) + m) | Low-D NN search |
| `QuadTree` | 2D | O(log n) avg | O(log n + k) | O(log n + k) | 2D range queries |
| `OctTree` | 3D | O(log n) avg | O(log n + k) | O(log n + k) | 3D voxels |
| `RTree` | Any | O(log n) avg | O(log n + k) | **O(log n + k)** | Rectangles/boxes |

**Example**:
```zig
// 2D nearest neighbor search
const KDTree = @import("zuda").containers.spatial.KDTree;
const points = [_][2]f64{
    .{1.0, 2.0}, .{3.0, 4.0}, .{5.0, 1.0},
};
var tree = try KDTree(2, f64).init(allocator, &points);
defer tree.deinit();

const nearest = tree.nearest([2]f64{3.5, 3.5}); // {3.0, 4.0}

// 2D bounding box queries (GIS)
const RTree = @import("zuda").containers.spatial.RTree;
var rtree = RTree(Location, 8).init(allocator);
try rtree.insert(BBox{ .min = .{0,0}, .max = .{10,10} }, location_data);
const results = try rtree.search(BBox{ .min = .{5,5}, .max = .{15,15} });
```

---

## String Structures

**Use Case**: Specialized string operations

### Decision Matrix

| Operation | Recommendation | Why |
|-----------|----------------|-----|
| **Pattern matching** | See [String Algorithms](#string-algorithms) | Don't store, just search |
| **Prefix queries** | `Trie` | Exact prefix matching |
| **Longest prefix match** | `RadixTree` | Compressed trie |
| **Suffix queries** | `SuffixArray` or `SuffixTree` | All suffixes indexed |
| **Large text editing** | `Rope` | O(log n) insert/delete |
| **Fuzzy matching** | `BKTree` | Metric tree for edit distance |

### Detailed Comparison

| Data Structure | Insert | Lookup | Prefix Search | Space | Best For |
|----------------|--------|--------|---------------|-------|----------|
| `Trie` | O(m) | O(m) | **O(p + k)** | High | Autocomplete |
| `RadixTree` | O(m) | O(m) | **O(p + k)** | **Lower** | Routing tables |
| `SuffixArray` | **O(n log n)** | O(m log n) | O(m log n) | O(n) | Pattern search |
| `SuffixTree` | **O(n)** | **O(m)** | **O(m)** | O(n) | Linear pattern search |
| `Rope` | O(log n) | N/A | N/A | O(n) | Text editors |
| `BKTree` | O(log n) avg | O(log n) avg | N/A | O(n) | Spell check |

*m = pattern length, n = text length, p = prefix length, k = results*

**Example**:
```zig
// Autocomplete with Trie
const Trie = @import("zuda").containers.trees.Trie;
var trie = Trie(u32).init(allocator);
try trie.insert("apple", 1);
try trie.insert("app", 2);
try trie.insert("application", 3);

if (trie.hasPrefix("app")) {
    // true - 3 words start with "app"
}

// Fuzzy string search
const BKTree = @import("zuda").containers.specialized.BKTree;
fn levenshtein(a: []const u8, b: []const u8) usize { ... }
var bktree = BKTree([]const u8, levenshtein).init(allocator);
try bktree.insert("kitten");
try bktree.insert("sitting");
const similar = try bktree.search("bitten", 1); // max distance = 1
```

---

## Probabilistic Structures

**Use Case**: Trade accuracy for space/speed

### Decision Matrix

| Problem | Recommendation | Error Type | Space |
|---------|----------------|------------|-------|
| **Set membership** | `BloomFilter` | False positives (~1%) | **O(n) bits** |
| **Set membership (with deletes)** | `CuckooFilter` | False positives (~2%) | O(n) bits |
| **Cardinality (distinct count)** | `HyperLogLog` | ~2% relative error | **O(log log n) bits** |
| **Frequency counting** | `CountMinSketch` | Overestimation | O(ε⁻¹ log δ⁻¹) |
| **Similarity estimation** | `MinHash` | Jaccard error | O(k hashes) |

### Detailed Comparison

| Data Structure | Operation | Time | Space | Error | Best For |
|----------------|-----------|------|-------|-------|----------|
| `BloomFilter` | Insert/Query | **O(k)** | **O(n) bits** | 1% FPR | Membership test |
| `CuckooFilter` | Insert/Query/Delete | **O(1)** | O(n) bits | 2% FPR | Deletable sets |
| `HyperLogLog` | Add/Count | **O(1)** | **O(m) registers** | ~2% | Cardinality |
| `CountMinSketch` | Add/Query | O(d) | O(d·w) | ε·N bound | Heavy hitters |
| `MinHash` | Add/Similarity | O(k) | O(k) | Jaccard error | Document similarity |

*k = hash functions, m = registers (2^precision), d = depth, w = width, ε = error, N = total count*

**Example**:
```zig
// Bloom filter for duplicate detection
const BloomFilter = @import("zuda").containers.probabilistic.BloomFilter;
var bf = try BloomFilter.init(allocator, 10000, 0.01); // 1% FPR
defer bf.deinit();

bf.insert("url1");
if (bf.contains("url1")) { // probably yes
    // already crawled
}
if (bf.contains("url2")) { // definitely no
    // not crawled
}

// Cardinality estimation (unique visitors)
const HLL = @import("zuda").containers.probabilistic.HyperLogLog(14); // precision
var hll = try HLL.init(allocator);
defer hll.deinit();

hll.add("user_id_1");
hll.add("user_id_2");
hll.add("user_id_1"); // duplicate
const unique_count = hll.count(); // ~2
```

---

## Persistent Structures

**Use Case**: Immutable data structures with version history

### Decision Matrix

| Data Type | Recommendation | Why |
|-----------|----------------|-----|
| **Array** | `PersistentArray` | O(log n) updates via path copying |
| **Map** | `PersistentHashMap` (HAMT) | Structural sharing |
| **Ordered map** | `PersistentRBTree` | Immutable tree with sharing |

### Detailed Comparison

| Data Structure | Access | Update (new version) | Space per Update | Best For |
|----------------|--------|----------------------|------------------|----------|
| `PersistentArray` | O(log n) | O(log n) | O(log n) nodes | Immutable sequences |
| `PersistentHashMap` | O(1) avg | O(log n) | O(log n) nodes | Immutable maps |
| `PersistentRBTree` | O(log n) | O(log n) | O(log n) nodes | Immutable ordered maps |

**Example**:
```zig
// Undo/redo with persistent structures
const PersistentArray = @import("zuda").containers.persistent.PersistentArray;
var v1 = try PersistentArray(u32, 32).init(allocator, 100, 0); // 100 zeros
var v2 = try v1.set(50, 42); // new version, v1 unchanged
var v3 = try v2.set(75, 99); // another version

// Can access all versions
const val1 = v1.get(50); // 0 (original)
const val2 = v2.get(50); // 42 (after first update)
const val3 = v3.get(50); // 42 (still there)
const val4 = v3.get(75); // 99
```

**⚠️ Note**: `PersistentRBTree` has known memory leak with multiple concurrent versions (needs ref-counting). Use arena allocator for version sets as workaround.

---

## Caches

**Use Case**: Limited-capacity fast storage with eviction policy

### Decision Matrix

| Eviction Policy | Recommendation | Why |
|-----------------|----------------|-----|
| **Least Recently Used** | `LRUCache` | Temporal locality |
| **Least Frequently Used** | `LFUCache` | Access frequency |
| **Adaptive (LRU + LFU)** | `ARCCache` | Self-tuning |

### Detailed Comparison

| Data Structure | Get | Put | Eviction | Best For |
|----------------|-----|-----|----------|----------|
| `LRUCache` | **O(1)** | **O(1)** | Least recent | Web caches, page replacement |
| `LFUCache` | **O(1)** | **O(1)** | Least frequent | CDN, frequency-based |
| `ARCCache` | **O(1)** | **O(1)** | Adaptive | Self-tuning workloads |

**Example**:
```zig
// LRU cache for database buffer pool
const LRUCache = @import("zuda").containers.cache.LRUCache;
var cache = try LRUCache(u64, []u8, 1000).init(allocator, 1000); // 1000 pages
defer cache.deinit();

// Access page (promotes to most recent)
if (cache.get(page_id)) |page| {
    // cache hit
} else {
    // cache miss - load from disk
    const page = try loadPage(page_id);
    _ = try cache.put(page_id, page); // evicts LRU if full
}
```

**Consumer Example**: **silica (RDBMS)** uses LRU cache for buffer pool (see `src/storage/buffer_pool.zig` - 1237 LOC).

---

## Algorithm Selection

### Sorting

| Input Type | Size | Recommendation | Why |
|------------|------|----------------|-----|
| **General** | Any | `TimSort` | Stable, adaptive |
| **Nearly sorted** | Any | `TimSort` | Exploits runs |
| **Random** | Large | `IntroSort` | Fast pivot-based |
| **Integers** | Any | `RadixSort` | O(d·n) linear |
| **Small range** | Any | `CountingSort` | O(n+k) |
| **Linked list** | Any | `MergeSort` | No random access |
| **Stable + in-place** | Any | `BlockSort` | O(1) space |

**Example**:
```zig
const TimSort = @import("zuda").algorithms.sorting.TimSort;
var items = [_]i32{5, 2, 8, 1, 9};
try TimSort.sort(i32, &items, {}, std.sort.asc(i32));
```

---

### String Matching

| Pattern Type | Count | Text Size | Recommendation | Why |
|--------------|-------|-----------|----------------|-----|
| **Single pattern** | 1 | Any | `KMP` | Linear time, simple |
| **Single pattern** | 1 | Large | `Boyer-Moore` | Skip characters |
| **Multiple patterns** | Many | Any | `Aho-Corasick` | O(n + m + z) |
| **Rolling hash** | 1+ | Any | `Rabin-Karp` | Good for multiple |
| **All occurrences** | Any | Any | `Z-algorithm` | Linear preprocessing |

**Example**:
```zig
const KMP = @import("zuda").algorithms.string.KMP;
const index = KMP.search("hello world", "world"); // 6

const AhoCorasick = @import("zuda").algorithms.string.AhoCorasick;
var ac = try AhoCorasick.init(allocator);
try ac.addPattern("he");
try ac.addPattern("she");
try ac.addPattern("his");
const matches = try ac.search("ushers"); // ["she", "he", "s"]
```

---

### Graph Algorithms

See [Graph Algorithms](#graph-algorithms-1) section above.

---

## Performance Considerations

### Memory Usage

| Scenario | Low Memory | Medium Memory | High Memory |
|----------|------------|---------------|-------------|
| **Linked list** | `XorLinkedList` | `std.DoublyLinkedList` | `UnrolledLinkedList` |
| **Hash table** | `RobinHoodHashMap` | `SwissTable` | `CuckooHashMap` |
| **Set membership** | `BloomFilter` | `CuckooFilter` | `std.HashMap` |
| **Cardinality** | `HyperLogLog` | `CountMinSketch` | Exact count |

### Time Complexity

| Priority | Data Structure Choice |
|----------|----------------------|
| **O(1) critical** | Hash tables, arrays, deques |
| **O(log n) acceptable** | Trees, heaps, skip lists |
| **O(n) occasional** | Linked lists, probabilistic |

### Concurrent Access

| Requirement | Recommendation |
|-------------|----------------|
| **Lock-free queue** | `LockFreeQueue` (Michael-Scott) |
| **Lock-free stack** | `LockFreeStack` (Treiber) |
| **Lock-free ordered map** | `ConcurrentSkipList` |
| **Work stealing** | `WorkStealingDeque` (Chase-Lev) |

---

## Common Use Cases

### Database Index
- **Dense**: `BTree` (disk pages)
- **Sparse**: `RedBlackTree` or `SkipList`
- **Full-text**: `SuffixArray` or `Trie`

### Cache Implementation
- **General**: `LRUCache`
- **Frequency-based**: `LFUCache`
- **Adaptive**: `ARCCache`
- **Distributed**: `ConsistentHashRing`

### Task Scheduler
- **Priority-based**: `std.PriorityQueue` or `DaryHeap`
- **Dependency graph**: `AdjacencyList` + `TopologicalSort`
- **Parallel**: `WorkStealingDeque`

### Network Router
- **IP routing**: `RadixTree` or `Trie`
- **Packet filtering**: `BloomFilter`
- **Load balancing**: `ConsistentHashRing`

### Text Editor
- **Large files**: `Rope`
- **Syntax tree**: `RedBlackTree` (interval-based)
- **Undo/redo**: `PersistentArray`

### Game Engine
- **Collision detection**: `QuadTree` (2D), `OctTree` (3D)
- **Pathfinding**: A* with `DaryHeap`
- **Spatial queries**: `RTree` or `KDTree`

---

## Quick Reference Table

| Use Case | Primary Choice | Alternative | Specialized |
|----------|---------------|-------------|-------------|
| **General map** | `RobinHoodHashMap` | `RedBlackTree` | `SwissTable` |
| **Priority queue** | `std.PriorityQueue` | `DaryHeap(4)` | `FibonacciHeap` |
| **Queue** | `Deque` | `LockFreeQueue` | `WorkStealingDeque` |
| **Set membership** | `std.HashMap` | `BloomFilter` | `CuckooFilter` |
| **Ordered iteration** | `RedBlackTree` | `SkipList` | `AVLTree` |
| **Graph** | `AdjacencyList` | `AdjacencyMatrix` | `CSR` |
| **Spatial (2D)** | `QuadTree` | `KDTree` | `RTree` |
| **String search** | `KMP` | `Boyer-Moore` | `Aho-Corasick` |
| **Unique count** | Exact map | `HyperLogLog` | `MinHash` |
| **Cache** | `LRUCache` | `ARCCache` | `LFUCache` |

---

## Anti-Patterns

### ❌ Don't Use

| Wrong Choice | Scenario | Use Instead |
|--------------|----------|-------------|
| `std.ArrayList` | Frequent front inserts | `Deque` or `XorLinkedList` |
| `AdjacencyMatrix` | Sparse graph (E ≪ V²) | `AdjacencyList` |
| `RedBlackTree` | Unordered access only | `RobinHoodHashMap` |
| `FibonacciHeap` | No decrease-key needed | `std.PriorityQueue` |
| `SuffixTree` | One-time search | `KMP` or `Boyer-Moore` |
| Exact map | Millions of keys, ~1% error OK | `BloomFilter` |
| `PersistentHashMap` | Mutating in place is fine | `RobinHoodHashMap` |

---

## Decision Flowchart

```
Start: What do you need?

├─ Store items?
│  ├─ Key-value pairs?
│  │  ├─ Ordered? → RedBlackTree / AVLTree / SkipList
│  │  └─ Unordered? → RobinHoodHashMap / SwissTable
│  │
│  ├─ Sequence?
│  │  ├─ Random access? → std.ArrayList
│  │  ├─ Both-end ops? → Deque
│  │  └─ Linked list? → UnrolledLinkedList / XorLinkedList
│  │
│  ├─ Priority-based? → std.PriorityQueue / FibonacciHeap
│  ├─ Spatial? → KDTree / QuadTree / RTree
│  ├─ Relationships? → AdjacencyList / AdjacencyMatrix
│  └─ Set membership? → BloomFilter / CuckooFilter

├─ Search/Match?
│  ├─ Single pattern? → KMP / Boyer-Moore
│  ├─ Multiple patterns? → Aho-Corasick
│  ├─ Prefix? → Trie / RadixTree
│  └─ Fuzzy? → BKTree / MinHash

├─ Graph algorithm?
│  ├─ Shortest path?
│  │  ├─ Single source? → Dijkstra / Bellman-Ford / A*
│  │  └─ All pairs? → Floyd-Warshall / Johnson
│  ├─ MST? → Kruskal / Prim
│  ├─ Connectivity? → Tarjan SCC / Kosaraju
│  ├─ Topological? → Kahn / DFS
│  └─ Flow? → Dinic / Edmonds-Karp

├─ Probabilistic?
│  ├─ Membership? → BloomFilter
│  ├─ Cardinality? → HyperLogLog
│  ├─ Frequency? → CountMinSketch
│  └─ Similarity? → MinHash

└─ Special requirements?
   ├─ Immutable? → PersistentArray / PersistentHashMap
   ├─ Concurrent? → LockFreeQueue / ConcurrentSkipList
   ├─ Cache? → LRUCache / ARCCache
   └─ Large text? → Rope
```

---

## Next Steps

- [API Reference](API.md) - Complete function signatures and usage
- [Algorithm Explainers](ALGORITHMS.md) - How algorithms work conceptually
- [Getting Started](GETTING_STARTED.md) - Build your first zuda application

**Still unsure?** Check the [Consumer Use Case Registry](../CLAUDE.md#consumer-use-case-registry) to see how real projects use these structures.
