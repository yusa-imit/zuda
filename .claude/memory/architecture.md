# zuda Architecture

## Module Organization

```
zuda (root.zig — re-exports all public types)
├── containers/
│   ├── lists/          Sequential containers (SkipList, XorLinkedList, UnrolledLinkedList)
│   ├── trees/          Tree-based containers (RedBlackTree, AVLTree, BTree, Trie, ...)
│   ├── heaps/          Heap variants (FibonacciHeap, BinomialHeap, PairingHeap, DaryHeap)
│   ├── hashing/        Hash containers (CuckooHashMap, RobinHoodHashMap, SwissTable)
│   ├── queues/         Queue variants (Deque, StealingQueue)
│   ├── graphs/         Graph representations (AdjacencyList, AdjacencyMatrix, CSR)
│   ├── strings/        String structures (SuffixArray, SuffixTree)
│   ├── spatial/        Spatial indices (KDTree, RTree, QuadTree, OctTree)
│   └── probabilistic/  Probabilistic (BloomFilter, CountMinSketch, HyperLogLog)
├── algorithms/
│   ├── sorting/        Sorting algorithms on []T slices
│   ├── searching/      Search algorithms
│   ├── graph/          Graph algorithms (generic over Graph concept)
│   ├── string/         String matching algorithms
│   ├── math/           Number theory, combinatorics
│   ├── geometry/       Computational geometry
│   └── dynamic_programming/
├── iterators/          Composable iterator adaptors
└── internal/           testing.zig, bench.zig (not public)
```

## Key Design Decisions

### Allocator-First Pattern
Every heap-allocating container accepts `std.mem.Allocator`. Managed variants store the allocator; Unmanaged variants require it per-call. Matches `std.ArrayListUnmanaged` pattern.

### Comptime Generics
All type parameterization happens at comptime. Comparators, hash functions, branching factors — no vtables, no runtime dispatch. Full monomorphization.

### Iterator Protocol
Standard `next() -> ?T` pattern compatible with `while (iter.next()) |item|`. Iterators are lazy and composable.

### Graph Algorithm Interface
Graph algorithms are generic over a duck-typed `Graph` concept via comptime. Any type providing `.neighbors()` and `.nodeCount()` works.

### Complexity Contracts
Every public function's doc comment states Big-O time and space. Verified via benchmark regression tests.
