# Algorithm Explainers

> Conceptual guides for key algorithms in zuda

## Table of Contents

1. [Graph Algorithms](#graph-algorithms)
2. [Tree Balancing](#tree-balancing)
3. [Hashing Techniques](#hashing-techniques)
4. [Heap Operations](#heap-operations)
5. [String Matching](#string-matching)
6. [Probabilistic Data Structures](#probabilistic-data-structures)
7. [Spatial Indexing](#spatial-indexing)
8. [Persistent Data Structures](#persistent-data-structures)

---

## Graph Algorithms

### Dijkstra's Algorithm

**Purpose**: Find shortest paths from a source vertex to all other vertices in a weighted graph with non-negative edge weights.

**How It Works**:
1. Maintain a priority queue of vertices ordered by current shortest distance
2. Start with source vertex at distance 0, all others at infinity
3. Repeatedly extract the vertex with minimum distance
4. For each neighbor, if path through current vertex is shorter, update distance and add to queue
5. Continue until queue is empty

**Why It Works**: Greedy choice property - once a vertex is extracted, its shortest path is finalized because all edge weights are non-negative.

**Example**:
```
Graph:      A --2-- B
            |       |
            1       3
            |       |
            C --1-- D

Starting from A:
Step 1: distances = {A:0, B:∞, C:∞, D:∞}  → Extract A
Step 2: distances = {A:0, B:2, C:1, D:∞}  → Extract C (min = 1)
Step 3: distances = {A:0, B:2, C:1, D:2}  → Extract B (min = 2)
Step 4: distances = {A:0, B:2, C:1, D:2}  → Extract D (min = 2)

Final: A→B=2, A→C=1, A→D=2
```

**Complexity**: O((V + E) log V) with binary heap, O(V²) with array

**Use Cases**: GPS navigation, network routing, resource allocation

**Consumer Example**: zr (task runner) doesn't use weighted graphs, but the pattern applies to any dependency resolution with costs.

---

### Tarjan's Strongly Connected Components

**Purpose**: Find all strongly connected components (SCCs) in a directed graph - maximal sets of vertices where every vertex is reachable from every other vertex in the set.

**How It Works** (single DFS pass):
1. Assign each vertex a unique DFS discovery index
2. Track the lowest index reachable from each vertex (via DFS tree or back edge)
3. Maintain a stack of vertices in current DFS path
4. When a vertex's low-link equals its index → found SCC root
5. Pop stack until root is reached - all popped vertices form an SCC

**Why It Works**: A vertex is an SCC root if and only if it cannot reach any vertex discovered earlier. The stack maintains the current path, ensuring we capture complete SCCs.

**Example**:
```
Graph:  1 → 2 → 3
        ↑       ↓
        5 ← 4 ← ┘

DFS from 1:
Visit 1: index=0, low=0, stack=[1]
Visit 2: index=1, low=1, stack=[1,2]
Visit 3: index=2, low=2, stack=[1,2,3]
Visit 4: index=3, low=3, stack=[1,2,3,4]
Visit 5: index=4, low=0, stack=[1,2,3,4,5] (back edge to 1)
  → Update low-links: 4.low=0, 3.low=0, 2.low=0, 1.low=0
  → 1.low == 1.index → SCC root!
  → Pop stack: SCC = {5,4,3,2,1}
```

**Complexity**: O(V + E) - single DFS traversal

**Use Cases**: Dependency cycles (see: zr cycle detection), compiler optimization, deadlock detection (see: silica lock manager)

---

### Kruskal's Minimum Spanning Tree

**Purpose**: Find a subset of edges that connects all vertices with minimum total weight (no cycles).

**How It Works**:
1. Sort all edges by weight (ascending)
2. Initialize disjoint set (union-find) for all vertices
3. For each edge in sorted order:
   - If endpoints are in different sets → add edge to MST, union the sets
   - Else → skip edge (would create cycle)
4. Stop when MST has V-1 edges

**Why It Works**: Greedy choice - the lightest edge connecting two components is always safe to include (cut property).

**Example**:
```
Graph:    A --1-- B
          |  \    |
          4   2   3
          |    \  |
          D --5-- C

Sorted edges: AB(1), AC(2), BC(3), AD(4), CD(5)

Step 1: Add AB (1) → {A,B} and {C} and {D}
Step 2: Add AC (2) → {A,B,C} and {D}
Step 3: Skip BC (3) → would create cycle
Step 4: Add AD (4) → {A,B,C,D}
Done! MST weight = 1+2+4 = 7
```

**Complexity**: O(E log E) for sorting + O(E α(V)) for union-find ≈ O(E log E)

**Use Cases**: Network design, clustering, image segmentation

**Consumer Example**: While zr doesn't use MST directly, the union-find component is used for cycle detection.

---

### Topological Sort (Kahn's Algorithm)

**Purpose**: Order vertices in a DAG such that for every edge u→v, u comes before v.

**How It Works**:
1. Compute in-degree (number of incoming edges) for each vertex
2. Add all vertices with in-degree 0 to a queue
3. While queue is not empty:
   - Remove vertex from queue, add to result
   - For each neighbor, decrement in-degree
   - If neighbor's in-degree becomes 0, add to queue
4. If result has fewer than V vertices → graph has a cycle

**Why It Works**: A vertex can be processed only after all its dependencies (incoming edges) are processed.

**Example**:
```
Graph:  A → B → D
        ↓       ↑
        C ------┘

In-degrees: A=0, B=1, C=1, D=2

Queue = [A]
Process A: result=[A], in-degrees: B=0, C=0, D=2, queue=[B,C]
Process B: result=[A,B], in-degrees: C=0, D=1, queue=[C]
Process C: result=[A,B,C], in-degrees: D=0, queue=[D]
Process D: result=[A,B,C,D]

Valid order: A → B → C → D (or A → C → B → D)
```

**Complexity**: O(V + E)

**Use Cases**: Task scheduling, build systems, course prerequisites

**Consumer Example**: **zr (task runner)** uses this exact algorithm to execute tasks in dependency order (see `src/graph/topo_sort.zig` - 323 LOC implementation that zuda will replace).

---

## Tree Balancing

### Red-Black Tree Rotations

**Purpose**: Maintain balanced BST with O(log n) height through local restructuring.

**Invariants**:
1. Every node is red or black
2. Root is black
3. Leaves (NIL) are black
4. Red nodes have black children (no two consecutive reds)
5. All paths from root to leaves have the same number of black nodes

**Left Rotation**:
```
    X                Y
   / \              / \
  a   Y     →      X   c
     / \          / \
    b   c        a   b
```

**Right Rotation**: Mirror of left rotation

**Insertion Rebalancing** (when uncle is red):
```
       G(B)              G(R)
      /   \             /   \
    P(R)  U(R)   →    P(B)  U(B)
    /                 /
  N(R)              N(R)
```

**Why It Works**: Rotations preserve BST order while adjusting structure. Color flips maintain black-height invariant.

**Complexity**: Insert/Delete require at most O(log n) rotations and recoloring

**Use Cases**: Balanced maps, database indexes, priority schedulers

---

### AVL Tree Height Balancing

**Purpose**: Stricter balance than RB-tree → faster lookups at cost of more rotations on updates.

**Invariant**: For every node, |height(left) - height(right)| ≤ 1

**Balance Factor**: BF = height(right) - height(left) ∈ {-1, 0, +1}

**Four Cases** (after insertion causing imbalance):

1. **Left-Left** (BF = -2 at node, -1 at left child):
   ```
       Z(-2)           Y
      /               / \
    Y(-1)     →      X   Z
    /
   X
   ```
   Fix: Right rotation at Z

2. **Left-Right** (BF = -2 at node, +1 at left child):
   ```
       Z(-2)         Z           X
      /             /           / \
    Y(+1)    →    X      →    Y   Z
     \           /
      X         Y
   ```
   Fix: Left rotation at Y, then right rotation at Z

3. **Right-Right**: Mirror of Left-Left
4. **Right-Left**: Mirror of Left-Right

**Complexity**: O(log n) for all operations, at most 2 rotations per insert, O(log n) rotations per delete

**Use Cases**: Lookup-intensive workloads, in-memory databases

---

### Splay Tree Self-Adjustment

**Purpose**: Move accessed nodes to root via double rotations → frequently accessed nodes become faster.

**Splaying Operations**:

1. **Zig-Zig** (left-left or right-right):
   ```
       G              X
      /              / \
    P         →     a   P
   /                   / \
  X                   b   G
 / \                     / \
a   b                   c   d
   ```
   Rotate P, then rotate X

2. **Zig-Zag** (left-right or right-left):
   ```
     G              X
    /              / \
   P         →    P   G
    \            / \ / \
     X          a b c  d
    / \
   b   c
   ```
   Rotate X twice

3. **Zig** (root's child): Simple rotation

**Why It Works**: Amortized O(log n) via potential function - deep nodes get pulled up, reducing future access cost.

**Use Cases**: Caches, access pattern optimization, recently-used data

---

## Hashing Techniques

### Cuckoo Hashing

**Purpose**: Achieve O(1) worst-case lookup using two hash functions and two tables.

**How It Works**:
1. Maintain two tables T1 and T2 with hash functions h1 and h2
2. **Lookup**: Check T1[h1(key)] and T2[h2(key)] - at most 2 locations
3. **Insert**: Try T1[h1(key)], if occupied:
   - Evict existing item to T2[h2(old_key)]
   - If T2 is occupied, evict that item back to T1[h1(evicted_key)]
   - Repeat until empty slot found or cycle detected (rehash with new functions)

**Example**:
```
h1(x) = x % 5, h2(x) = (x / 5) % 5
Insert 7:  T1[2] = 7
Insert 14: T1[4] = 14
Insert 2:  T1[2] occupied → evict 7
           T2[1] = 7, T1[2] = 2
```

**Max Loop Iterations**: O(log n) with high probability before rehash

**Use Cases**: Real-time systems, hardware caches, low-latency lookups

---

### Robin Hood Hashing

**Purpose**: Open addressing with "rich steal from poor" heuristic → low variance in probe lengths.

**Probe Sequence Length (PSL)**: Distance from ideal position

**Insertion Rule**: If inserting item has PSL > current slot's PSL, evict current item and continue inserting it.

**Example**:
```
Hash table (size 5), h(x) = x % 5

Insert 5:  [5, _, _, _, _]  PSL of 5 = 0
Insert 10: [5, 10, _, _, _] PSL of 10 = 0
Insert 0:  Ideal slot = 0, occupied by 5
           PSL(0) = 0 at slot 0, PSL(5) = 0 → no swap
           PSL(0) = 1 at slot 1, PSL(10) = 0 → swap!
           [5, 0, 10, _, _]
```

**Why It Works**: Variance in probe lengths is minimized, improving cache performance and worst-case behavior.

**Use Cases**: General-purpose hash tables, embedded systems

---

### SwissTable (Google's Abseil)

**Purpose**: SIMD-friendly hash table using control bytes for fast probing.

**Structure**:
- Groups of 16 slots
- Each group has 16 control bytes (metadata):
  - 0x00-0x7F: H2 hash (low 7 bits of hash)
  - 0x80: Empty
  - 0xFE: Deleted
  - 0xFF: Sentinel

**Lookup**:
1. Compute H1 (group index) and H2 (7-bit hash)
2. Load 16 control bytes for group H1
3. **SIMD**: Compare all 16 bytes to H2 in parallel
4. Check matching slots for key equality
5. If no match and group has EMPTY → not found
6. Else probe next group (quadratic probing)

**Advantages**:
- 1 cache line per group (64 bytes = 16 control + 48 data)
- 16-way parallel comparison with SSE2/NEON
- Tombstone-free deletion (swap with last in chain)

**Use Cases**: High-performance hash tables, large datasets, Google Abseil

---

## Heap Operations

### Fibonacci Heap Lazy Merge

**Purpose**: O(1) insert and decrease-key via deferred consolidation.

**Structure**:
- Forest of min-heap-ordered trees (roots in circular doubly-linked list)
- Each node stores degree (number of children)
- Marked nodes (lost a child since becoming non-root)

**Operations**:

1. **Insert**: Add new tree with single node to root list - O(1)

2. **Find-Min**: Track min pointer in root list - O(1)

3. **Extract-Min**:
   - Remove min node, add its children to root list
   - **Consolidate**: Merge trees of same degree until all degrees are unique
   - Update min pointer - O(log n) amortized

4. **Decrease-Key**:
   - Decrease key value
   - If heap order violated, **cut** node and add to root list
   - If parent becomes marked, **cascade cut** up to root
   - Unmark node - O(1) amortized

**Why It Works**: Amortized analysis - potential function based on number of trees and marks. Expensive operations (consolidate) are paid for by cheap ones (insert).

**Use Cases**: Dijkstra's algorithm, Prim's MST, algorithms needing frequent decrease-key

---

### D-ary Heap Cache Optimization

**Purpose**: Generalize binary heap to d children → fewer levels but more comparisons per level.

**Trade-off**:
- **Binary heap** (d=2): log₂(n) levels, 1 comparison per level → good for decrease-key
- **4-ary heap** (d=4): log₄(n) levels, 3 comparisons per level → better cache locality
- **8-ary heap** (d=8): Even fewer levels but 7 comparisons → diminishing returns

**Array Indexing**:
- Parent of node i: ⌊(i - 1) / d⌋
- Children of node i: [d*i + 1, d*i + d]

**Cache Analysis**:
- Binary heap: log₂(n) cache misses (one per level)
- 4-ary heap: log₄(n) = ½ log₂(n) cache misses, 4 children fit in 1 cache line

**Optimal d**: Usually d = 4 or d = 8 on modern CPUs with 64-byte cache lines.

**Use Cases**: Priority queues in performance-critical code, graph algorithms

---

## String Matching

### Knuth-Morris-Pratt (KMP)

**Purpose**: Linear-time string matching by avoiding re-scanning matched prefix.

**Key Idea**: Precompute "failure function" - longest proper prefix that is also a suffix.

**Failure Function**:
```
Pattern: "ABABC"
Index:    0 1 2 3 4
Failure:  0 0 1 2 0

Explanation:
- Index 0: No proper prefix → 0
- Index 1: "AB" has no matching prefix/suffix → 0
- Index 2: "ABA" has "A" as prefix and suffix → 1
- Index 3: "ABAB" has "AB" as prefix and suffix → 2
- Index 4: "ABABC" has no matching prefix/suffix → 0
```

**Search Algorithm**:
```
Text:    "ABABABC"
Pattern: "ABABC"

i=0, j=0: A=A, match → i++, j++
i=1, j=1: B=B, match → i++, j++
i=2, j=2: A=A, match → i++, j++
i=3, j=3: B=B, match → i++, j++
i=4, j=4: A≠C, mismatch → j = failure[3] = 2 (skip "AB")
i=4, j=2: A=A, match → i++, j++
i=5, j=3: B=B, match → i++, j++
i=6, j=4: C=C, match → found at i-4 = 2
```

**Complexity**: O(n + m) - linear in text + pattern length

**Use Cases**: Text editors, DNA sequence search, log parsing

---

### Boyer-Moore

**Purpose**: Fast string search by skipping chunks of text based on mismatches.

**Two Heuristics**:

1. **Bad Character Rule**: If mismatch at text[i], shift pattern so that:
   - Next occurrence of text[i] in pattern aligns, or
   - Pattern moves past text[i] if character not in pattern

2. **Good Suffix Rule**: If mismatch after matching suffix, shift pattern so that:
   - Next occurrence of suffix aligns with matching part, or
   - Longest prefix of pattern that matches suffix aligns

**Example**:
```
Text:    "TRUST RUSTYRUST"
Pattern: "RUST"

Align at index 0:
T R U S T
R U S T
      ↑ mismatch at 'T'

Bad character 'T' appears at pattern[3]
Skip to align: shift by max(good_suffix_shift, bad_char_shift)

Next alignment at index 6:
T R U S T   R U S T Y   R U S T
            R U S T
                  ↑ match!
```

**Best Case**: O(n/m) - can skip m characters on each mismatch

**Average Case**: O(n)

**Worst Case**: O(nm) - but rare in practice

**Use Cases**: Text editors (grep), genome search, large text corpora

---

### Aho-Corasick Multi-Pattern Matching

**Purpose**: Search for multiple patterns simultaneously in linear time.

**Structure**: Trie + failure links (like KMP for trees)

**Construction**:
1. Build trie of all patterns
2. Add failure links using BFS:
   - If state s has incoming edge 'c' from parent p,
   - Follow failure links from p until finding state with outgoing edge 'c'
   - Link s's failure to that state (or root if none found)
3. Add output links to mark pattern end states

**Example**:
```
Patterns: ["he", "she", "his", "hers"]

Trie:           (root)
               /  |  \
              h   s   ...
             / \   \
            e   i   h
           /     \   \
          r       s   e
         /             \
        s               r
                         \
                          s
Failure links: (shown with dashed arrows in actual implementation)
- 'h' → root
- 'e' (from 'h') → root
- 's' → root
- 'h' (from 's') → 'h' (from root)
...
```

**Search**:
```
Text: "ushers"
State transitions with failure links allow matching "she", "he", "hers" in one pass
```

**Complexity**: O(n + m + z) where n = text length, m = total pattern length, z = matches

**Use Cases**: Intrusion detection, spam filtering, multi-keyword search

---

## Probabilistic Data Structures

### Bloom Filter Mathematics

**Purpose**: Space-efficient set membership test with tunable false positive rate.

**Structure**: Bit array of size m with k hash functions

**Operations**:
- **Insert**: Set bits at positions h₁(x), h₂(x), ..., hₖ(x) to 1
- **Query**: Check if all bits at h₁(x), ..., hₖ(x) are 1
  - If any is 0 → definitely not in set
  - If all are 1 → probably in set (false positive possible)

**False Positive Rate**:
```
After inserting n items:
- Probability a bit is still 0: (1 - 1/m)^(kn) ≈ e^(-kn/m)
- Probability of false positive: (1 - e^(-kn/m))^k

Optimal k = (m/n) ln(2) ≈ 0.69 * m/n

With optimal k, false positive rate ≈ (0.6185)^(m/n)
```

**Example**:
```
m = 100 bits, n = 10 items, k = 7 hashes
FPR ≈ (0.6185)^10 = 0.008 = 0.8%

To achieve 1% FPR with n=10000: m ≈ 95850 bits ≈ 12 KB
```

**Use Cases**: Spell checkers (words in dictionary), web caches, databases

---

### HyperLogLog Cardinality Estimation

**Purpose**: Estimate cardinality (distinct count) of large sets using O(log log n) space.

**How It Works**:
1. Hash each item to uniform random bits
2. Count leading zeros in hash (ρ(h(x)))
3. Maintain m = 2^p registers (buckets) based on first p bits of hash
4. Store max leading zeros seen in each register
5. Estimate cardinality: n ≈ α_m * m² / Σ 2^(-M[j])
   where α_m is correction constant, M[j] = max leading zeros in bucket j

**Intuition**: If max leading zeros is k, we've likely seen ~2^k distinct values in that bucket.

**Example**:
```
m = 16 registers (p = 4)

Insert "hello": hash = 0101_1000... → bucket 5, leading zeros = 1
Insert "world": hash = 1010_0001... → bucket 10, leading zeros = 3
...
After 1000 inserts:
M = [3, 5, 2, 4, 1, 7, ...] (16 values)

Estimate ≈ 0.673 * 16² / (2^-3 + 2^-5 + ... + 2^-7) ≈ 987
Actual: 1000 → ~1.3% error
```

**Accuracy**: Standard error ≈ 1.04 / √m (2% error with m=256)

**Space**: m log log n bits (e.g., m=16384 registers × 5 bits = 10 KB)

**Use Cases**: Web analytics (unique visitors), database query optimization, distributed systems

**Consumer Example**: **zoltraak (Redis-compatible server)** implements PFADD/PFCOUNT using HyperLogLog (see `src/storage/memory.zig` - 80 LOC implementation).

---

### Count-Min Sketch Frequency Estimation

**Purpose**: Approximate frequency counting with bounded error and constant memory.

**Structure**: 2D array (d × w) with d hash functions

**Operations**:
- **Update**: For item x, increment count[i][h_i(x)] for all i ∈ [0, d)
- **Query**: Return min_i(count[i][h_i(x)])

**Error Bounds**:
- With probability ≥ 1 - δ: estimate ≤ true_count + ε * N
- Where N = total count, d = ⌈ln(1/δ)⌉, w = ⌈e/ε⌉

**Why Minimum**: Hash collisions only cause overestimation, so minimum reduces false positives.

**Example**:
```
d = 3, w = 10, ε = 0.1, δ = 0.05

Add "apple" 5 times: count[0][3]++, count[1][7]++, count[2][1]++ (each 5 times)
Add "banana" 3 times: count[0][3]++, count[1][2]++, count[2][1]++ (each 3 times)

Query "apple": min(count[0][3]=8, count[1][7]=5, count[2][1]=8) = 5 ✓
Query "banana": min(count[0][3]=8, count[1][2]=3, count[2][1]=8) = 3 ✓
```

**Use Cases**: Heavy hitters, network traffic analysis, streaming data

---

## Spatial Indexing

### R-Tree Bounding Box Hierarchy

**Purpose**: Index multi-dimensional rectangles for fast spatial queries.

**Structure**: Tree where each node stores bounding box containing all children's boxes.

**Insertion** (ChooseLeaf + Split):
1. **ChooseLeaf**: From root, recursively choose child whose bounding box needs least enlargement
2. **Insert**: Add rectangle to chosen leaf
3. **Split** (if node overflows): Use quadratic split algorithm
   - Find pair of rectangles with max "wasted" area if grouped
   - Assign remaining rectangles to minimize total area

**Search**:
1. Start at root
2. For each node, check if query rectangle overlaps bounding box
3. If yes, recursively search children
4. Return all leaf rectangles that overlap query

**Example**:
```
Rectangles: R1=[0,0,2,2], R2=[1,1,3,3], R3=[5,5,7,7], R4=[6,6,8,8]

R-Tree (max 2 per node):
        [0,0,8,8]
       /         \
  [0,0,3,3]    [5,5,8,8]
   /    \       /     \
  R1    R2     R3     R4

Query [1.5, 1.5, 2.5, 2.5]:
- Overlaps [0,0,8,8] → descend
- Overlaps [0,0,3,3] → descend → check R1, R2 → return R2
- Doesn't overlap [5,5,8,8] → skip
```

**Complexity**: O(log n) average for insert/search, O(n) worst case

**Use Cases**: GIS systems (map queries), CAD, game engines

---

### KD-Tree Splitting Strategy

**Purpose**: Partition k-dimensional space for efficient nearest neighbor search.

**Construction** (recursive):
1. Choose splitting dimension (cycle through x, y, z, ... or choose max variance)
2. Find median point along that dimension
3. Split into two subtrees: points ≤ median and points > median
4. Recurse on each subtree

**Nearest Neighbor Search**:
1. Traverse to leaf containing query point (current best)
2. Backtrack up tree
3. At each node, check if other subtree could contain closer point
   - If distance to splitting plane < current best distance → search other side
4. Update best distance as closer points are found

**Example** (2D):
```
Points: [(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)]

Build tree (alternating x/y splits):
              (7,2) [split on x=7]
             /                    \
        (5,4) [split on y=4]    (9,6)
       /         \
    (2,3)       (4,7)

Nearest to (7, 3):
1. Descend to (9,6) - distance = 3.6
2. Backtrack to (7,2) - distance = 1.0 (better!)
3. Check left subtree (split at x=7, query x=7) - must check!
4. Descend left to (5,4) - distance = 2.2
5. Final best: (7,2) with distance 1.0
```

**Complexity**: O(log n) average for balanced tree, O(n) worst case

**Use Cases**: Nearest neighbor search, clustering, collision detection

---

## Persistent Data Structures

### Path Copying

**Purpose**: Create new version of data structure sharing unchanged parts with old version.

**Technique**: When updating node X:
1. Create new node X' with updated value
2. Copy parent of X to new parent P', pointing to X' instead of X
3. Recursively copy ancestors up to root
4. Old root points to old structure, new root points to new structure

**Example** (binary tree):
```
Original tree:        Update node 3 to 3':

      1                     1'
     / \                   / \
    2   3        →        2   3'
   / \                   / \
  4   5                 4   5

Only copied: 1→1', 3→3' (2, 4, 5 are shared)
Space: O(log n) new nodes per update
```

**Benefits**:
- Old versions remain accessible
- Structural sharing saves space
- Immutability enables safe concurrency

**Complexity**: O(log n) time and space per update for tree structures

**Use Cases**: Undo/redo, version control, functional programming

---

### Hash Array Mapped Trie (HAMT)

**Purpose**: Persistent hash map with efficient structural sharing.

**Structure**:
- Root-to-leaf paths determined by hash bits
- Each level uses 5 bits of hash (32-way branching)
- Nodes store bitmap (32 bits) indicating which children exist
- Actual children stored in compact array (no null gaps)

**Lookup**:
1. Hash key
2. Use bits [0-4] to find child at root (via bitmap)
3. Use bits [5-9] for next level, etc.
4. Reach leaf, compare full key

**Update** (path copying):
1. Navigate to leaf
2. Create new leaf with updated value
3. Copy parent, updating its child pointer
4. Recursively copy up to root

**Example**:
```
Insert "key1" (hash = 0b...10110):
- Root[22] (bits 0-4 = 10110) → create child
- Level 1[...] (bits 5-9) → ...

Insert "key2" (hash = 0b...10101):
- Root[21] (bits 0-4 = 10101) → create child
- Shares root with "key1" except different child pointers

Update "key1":
- Copy leaf for "key1" → leaf'
- Copy path from leaf to root → root'
- Old root still points to old "key1", new root points to leaf'
```

**Complexity**:
- Lookup: O(log₃₂ n) ≈ O(1) for practical sizes
- Update: O(log₃₂ n) time and space

**Use Cases**: Clojure's PersistentHashMap, immutable collections, concurrent data structures

**Consumer Note**: zuda's PersistentHashMap implements HAMT (see `src/containers/hashing/persistent_hash_map.zig`).

---

## Advanced Topics

### Lock-Free Data Structures (CAS-based)

**Compare-And-Swap (CAS)**: Atomic operation `CAS(addr, expected, new_value)` that:
- Reads value at addr
- If value == expected, writes new_value and returns true
- Else returns false

**Lock-Free Stack** (Treiber stack):
```zig
pub fn push(self: *Self, value: T) void {
    const new_node = Node{ .value = value, .next = null };
    while (true) {
        const old_head = @atomicLoad(*Node, &self.head, .Acquire);
        new_node.next = old_head;
        if (@cmpxchgWeak(*Node, &self.head, old_head, &new_node, .Release, .Acquire) == null) {
            break; // Success!
        }
        // CAS failed, retry
    }
}
```

**ABA Problem**: Head = A → thread 1 preempted → head changed A→B→A → thread 1's CAS succeeds incorrectly.

**Solutions**:
- Tagged pointers (version counter)
- Hazard pointers
- Epoch-based reclamation

**Use Cases**: Multi-threaded queues, memory allocators, work-stealing schedulers

**Consumer Example**: **zr (task runner)** uses work-stealing deque for parallel task execution (see `src/exec/workstealing.zig` - 130 LOC).

---

### Dancing Links (Knuth's Algorithm X)

**Purpose**: Efficiently solve exact cover problems using backtracking with O(1) undo.

**Key Insight**: Doubly-linked list allows O(1) remove and restore:
```zig
// Remove node x
x.left.right = x.right;
x.right.left = x.left;

// Restore node x (if pointers unchanged)
x.left.right = x;
x.right.left = x;
```

**Structure**: 2D toroidal linked list
- Rows = candidate solutions
- Columns = constraints to satisfy
- 1 in matrix = link exists

**Algorithm X**:
1. If matrix empty → solution found
2. Choose column c with fewest 1s
3. For each row r where matrix[r][c] = 1:
   - Include r in solution
   - Remove column c and all rows covering c's constraint
   - Recursively solve reduced matrix
   - Backtrack: restore removed rows/columns (O(1) via Dancing Links)

**Example** (Sudoku):
- Rows = possible placements (cell, digit)
- Columns = constraints (cell filled, row has digit, col has digit, box has digit)
- Search for exact cover

**Complexity**: Exponential worst case, but highly pruned search tree in practice

**Use Cases**: Sudoku, N-Queens, pentomino tiling, scheduling

**Consumer Note**: zuda implements DancingLinks (see `src/containers/exotic/dancing_links.zig`).

---

## Summary

This guide covers the core algorithmic concepts behind zuda's data structures. For complete API details and usage examples, see [API.md](API.md). For choosing the right data structure, see [GUIDE.md](GUIDE.md).

**Key Takeaways**:

1. **Graph Algorithms**: Understand trade-offs between algorithms (Dijkstra vs Bellman-Ford, Kruskal vs Prim)
2. **Tree Balancing**: Different balance criteria (RB-tree vs AVL vs Splay) suit different workloads
3. **Hashing**: Open addressing vs chaining, cuckoo vs robin hood vs Swiss table
4. **Heaps**: Lazy operations (Fibonacci) vs eager (binary), d-ary for cache optimization
5. **String Matching**: Preprocessing amortizes cost (KMP, Boyer-Moore, Aho-Corasick)
6. **Probabilistic**: Trade accuracy for space (Bloom filter, HyperLogLog, Count-Min Sketch)
7. **Spatial**: Partitioning strategies (KD-tree vs R-tree vs Quad/Oct-tree)
8. **Persistent**: Path copying and structural sharing for immutability

**Next Steps**:
- [API Reference](API.md) - Complete function signatures
- [Decision Guide](GUIDE.md) - Choose the right data structure
- [Getting Started](GETTING_STARTED.md) - Build your first zuda application
