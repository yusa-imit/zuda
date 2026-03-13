# Getting Started with zuda

> Quick start guide for building applications with zuda

## Installation

### As a Zig Dependency (Recommended)

Add zuda to your `build.zig.zon`:

```zig
.{
    .name = "my-project",
    .version = "0.1.0",
    .dependencies = .{
        .zuda = .{
            .url = "https://github.com/yusa-imit/zuda/archive/refs/tags/v0.5.0.tar.gz",
            // Run `zig fetch <url>` to get the hash
            .hash = "1220abcd...",
        },
    },
}
```

Update your `build.zig`:

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Add zuda dependency
    const zuda = b.dependency("zuda", .{
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "my-app",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Link zuda
    exe.root_module.addImport("zuda", zuda.module("zuda"));

    b.installArtifact(exe);
}
```

### Fetch the Dependency

```bash
zig build
```

---

## Your First zuda Program

### Example 1: Using a Hash Map

```zig
const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a RobinHoodHashMap
    const RobinHoodHashMap = zuda.containers.hashing.RobinHoodHashMap;

    // Context for string keys
    const StringContext = struct {
        pub fn hash(_: @This(), key: []const u8) u64 {
            return std.hash.Wyhash.hash(0, key);
        }
        pub fn eql(_: @This(), a: []const u8, b: []const u8) bool {
            return std.mem.eql(u8, a, b);
        }
    };

    var map = RobinHoodHashMap([]const u8, u32, StringContext, StringContext.hash, StringContext.eql).init(allocator);
    defer map.deinit();

    // Insert key-value pairs
    try map.put("alice", 30);
    try map.put("bob", 25);
    try map.put("charlie", 35);

    // Retrieve values
    if (map.get("alice")) |age| {
        std.debug.print("Alice is {} years old\n", .{age});
    }

    // Update value
    try map.put("alice", 31);

    // Remove value
    _ = map.remove("charlie");

    // Iterate over entries
    var it = map.iterator();
    while (it.next()) |entry| {
        std.debug.print("{s}: {}\n", .{ entry.key, entry.value });
    }
}
```

Output:
```
Alice is 30 years old
alice: 31
bob: 25
```

---

### Example 2: Using a Red-Black Tree

```zig
const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a RedBlackTree for ordered map
    const RedBlackTree = zuda.containers.trees.RedBlackTree;

    fn compareInts(_: void, a: i32, b: i32) std.math.Order {
        return std.math.order(a, b);
    }

    var tree = RedBlackTree(i32, []const u8, void, compareInts).init(allocator);
    defer tree.deinit();

    // Insert entries
    try tree.insert(42, "answer");
    try tree.insert(7, "lucky");
    try tree.insert(100, "century");
    try tree.insert(13, "unlucky");

    // Lookup
    if (tree.get(42)) |value| {
        std.debug.print("42 -> {s}\n", .{value});
    }

    // Iterate in sorted order
    std.debug.print("\nIn-order traversal:\n", .{});
    var it = tree.iterator();
    while (it.next()) |entry| {
        std.debug.print("{}: {s}\n", .{ entry.key, entry.value });
    }

    // Validate tree invariants
    try tree.validate();
    std.debug.print("\nTree is valid!\n", .{});
}
```

Output:
```
42 -> answer

In-order traversal:
7: lucky
13: unlucky
42: answer
100: century

Tree is valid!
```

---

### Example 3: Using a Bloom Filter

```zig
const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a Bloom filter for 1000 items with 1% false positive rate
    const BloomFilter = zuda.containers.probabilistic.BloomFilter;
    var bf = try BloomFilter.init(allocator, 1000, 0.01);
    defer bf.deinit();

    // Add items
    bf.insert("apple");
    bf.insert("banana");
    bf.insert("cherry");

    // Check membership
    std.debug.print("Contains 'apple': {}\n", .{bf.contains("apple")}); // true
    std.debug.print("Contains 'banana': {}\n", .{bf.contains("banana")}); // true
    std.debug.print("Contains 'grape': {}\n", .{bf.contains("grape")}); // false (probably)

    // Note: Bloom filters can have false positives but never false negatives
    std.debug.print("\nBloom filter uses {} bytes\n", .{bf.sizeInBytes()});
}
```

Output:
```
Contains 'apple': true
Contains 'banana': true
Contains 'grape': false

Bloom filter uses 1197 bytes
```

---

### Example 4: Dijkstra's Shortest Path

```zig
const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a weighted directed graph
    const AdjacencyList = zuda.containers.graphs.AdjacencyList;
    var graph = AdjacencyList(void, f64, true).init(allocator);
    defer graph.deinit();

    // Add vertices
    const a = try graph.addVertex({});
    const b = try graph.addVertex({});
    const c = try graph.addVertex({});
    const d = try graph.addVertex({});

    // Add weighted edges
    try graph.addEdge(a, b, 2.0);
    try graph.addEdge(a, c, 1.0);
    try graph.addEdge(b, d, 3.0);
    try graph.addEdge(c, d, 1.0);

    // Run Dijkstra's algorithm
    const Dijkstra = zuda.algorithms.graph.Dijkstra;
    const result = try Dijkstra.shortestPaths(allocator, &graph, a);
    defer allocator.free(result.distances);
    defer allocator.free(result.predecessors);

    // Print shortest distances
    std.debug.print("Shortest distances from vertex {}:\n", .{a});
    for (result.distances, 0..) |dist, i| {
        if (dist < std.math.inf(f64)) {
            std.debug.print("  to {}: {d:.1}\n", .{ i, dist });
        }
    }
}
```

Output:
```
Shortest distances from vertex 0:
  to 0: 0.0
  to 1: 2.0
  to 2: 1.0
  to 3: 2.0
```

---

### Example 5: String Matching with KMP

```zig
const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const KMP = zuda.algorithms.string.KMP;

    const text = "The quick brown fox jumps over the lazy dog";
    const pattern = "fox";

    // Find first occurrence
    if (try KMP.search(allocator, text, pattern)) |index| {
        std.debug.print("Pattern found at index: {}\n", .{index});
        std.debug.print("Matched: {s}\n", .{text[index..index + pattern.len]});
    } else {
        std.debug.print("Pattern not found\n", .{});
    }

    // Find all occurrences
    const matches = try KMP.searchAll(allocator, text, pattern);
    defer allocator.free(matches);

    std.debug.print("\nAll occurrences: {any}\n", .{matches});
}
```

Output:
```
Pattern found at index: 16
Matched: fox

All occurrences: { 16 }
```

---

## Common Patterns

### Pattern 1: Generic Containers with Comptime Parameters

```zig
const std = @import("std");
const zuda = @import("zuda");

// Define a custom comparator at comptime
fn MyType_compare(_: void, a: MyType, b: MyType) std.math.Order {
    return std.math.order(a.priority, b.priority);
}

const MyType = struct {
    id: u32,
    priority: i32,
};

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Use custom comparator
    const RedBlackTree = zuda.containers.trees.RedBlackTree;
    var tree = RedBlackTree(MyType, []const u8, void, MyType_compare).init(allocator);
    defer tree.deinit();

    try tree.insert(.{ .id = 1, .priority = 10 }, "low");
    try tree.insert(.{ .id = 2, .priority = 100 }, "high");
}
```

---

### Pattern 2: Memory-Safe Iteration

```zig
const std = @import("std");
const zuda = @import("zuda");

pub fn processItems(allocator: std.mem.Allocator) !void {
    const SkipList = zuda.containers.lists.SkipList;

    fn compareU32(_: void, a: u32, b: u32) std.math.Order {
        return std.math.order(a, b);
    }

    var list = SkipList(u32, []const u8, void, compareU32).init(allocator, 16);
    defer list.deinit();

    try list.insert(10, "ten");
    try list.insert(20, "twenty");
    try list.insert(5, "five");

    // Safe iteration (iterator doesn't invalidate on container deinit)
    var it = list.iterator();
    while (it.next()) |entry| {
        std.debug.print("{}: {s}\n", .{ entry.key, entry.value });
    }
}
```

---

### Pattern 3: Allocator-First Design

All zuda containers accept `std.mem.Allocator`:

```zig
const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    // Use different allocators
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa_allocator = gpa.allocator();

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    // GPA for long-lived structure
    const RBTree = zuda.containers.trees.RedBlackTree;
    var persistent_map = RBTree(u32, []const u8, void, compareU32).init(gpa_allocator);
    defer persistent_map.deinit();

    // Arena for temporary structure (no need to deinit individually)
    var temp_map = RBTree(u32, []const u8, void, compareU32).init(arena_allocator);
    _ = temp_map;
    // arena.deinit() frees everything
}

fn compareU32(_: void, a: u32, b: u32) std.math.Order {
    return std.math.order(a, b);
}
```

---

### Pattern 4: Validate Invariants During Development

```zig
const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const RedBlackTree = zuda.containers.trees.RedBlackTree;
    fn cmp(_: void, a: i32, b: i32) std.math.Order {
        return std.math.order(a, b);
    }

    var tree = RedBlackTree(i32, void, void, cmp).init(allocator);
    defer tree.deinit();

    try tree.insert(10, {});
    try tree.insert(20, {});
    try tree.insert(5, {});

    // Validate tree structure (debug builds)
    try tree.validate();

    _ = tree.remove(10);

    // Validate after mutation
    try tree.validate();
    std.debug.print("All invariants hold!\n", .{});
}
```

---

## Project Structure Example

Recommended structure for a zuda-based project:

```
my-project/
├── build.zig
├── build.zig.zon          # zuda dependency
├── src/
│   ├── main.zig
│   ├── graph.zig          # Graph algorithms
│   ├── cache.zig          # LRU cache implementation
│   └── spatial.zig        # Spatial indexing
├── tests/
│   ├── graph_test.zig
│   └── cache_test.zig
└── README.md
```

**src/main.zig**:
```zig
const std = @import("std");
const graph = @import("graph.zig");
const cache = @import("cache.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const g = try graph.createGraph(allocator);
    defer g.deinit();

    const c = try cache.createCache(allocator, 100);
    defer c.deinit();

    // Use graph and cache
}
```

**src/graph.zig**:
```zig
const std = @import("std");
const zuda = @import("zuda");

pub const Graph = zuda.containers.graphs.AdjacencyList(void, f64, true);

pub fn createGraph(allocator: std.mem.Allocator) !Graph {
    return Graph.init(allocator);
}
```

---

## Testing with zuda

```zig
const std = @import("std");
const zuda = @import("zuda");

test "RedBlackTree insertion and lookup" {
    const allocator = std.testing.allocator;

    const RBTree = zuda.containers.trees.RedBlackTree;
    fn cmp(_: void, a: i32, b: i32) std.math.Order {
        return std.math.order(a, b);
    }

    var tree = RBTree(i32, []const u8, void, cmp).init(allocator);
    defer tree.deinit();

    try tree.insert(42, "answer");
    try tree.insert(7, "lucky");

    try std.testing.expectEqualStrings("answer", tree.get(42).?);
    try std.testing.expectEqualStrings("lucky", tree.get(7).?);
    try std.testing.expect(tree.get(100) == null);

    // Validate invariants
    try tree.validate();
}

test "BloomFilter false positive rate" {
    const allocator = std.testing.allocator;

    const BloomFilter = zuda.containers.probabilistic.BloomFilter;
    var bf = try BloomFilter.init(allocator, 1000, 0.01);
    defer bf.deinit();

    // Insert 1000 items
    var i: u32 = 0;
    while (i < 1000) : (i += 1) {
        var buf: [16]u8 = undefined;
        const str = try std.fmt.bufPrint(&buf, "item{}", .{i});
        bf.insert(str);
    }

    // Check that inserted items are found
    i = 0;
    while (i < 1000) : (i += 1) {
        var buf: [16]u8 = undefined;
        const str = try std.fmt.bufPrint(&buf, "item{}", .{i});
        try std.testing.expect(bf.contains(str));
    }
}
```

Run tests:
```bash
zig build test
```

---

## Using C API (FFI)

zuda provides C bindings for selected containers:

### Build Shared Library

```bash
zig build -Dshared=true
```

This generates:
- `zig-out/lib/libzuda.a`
- `zig-out/include/zuda.h`

### C Example

```c
#include <zuda.h>
#include <stdio.h>

int main() {
    ZudaHashMap* map = zuda_hash_map_create();

    zuda_hash_map_put(map, "language", "Zig");
    zuda_hash_map_put(map, "library", "zuda");

    const char* value = zuda_hash_map_get(map, "language");
    printf("Language: %s\n", value); // "Zig"

    zuda_hash_map_destroy(map);
    return 0;
}
```

Compile:
```bash
gcc main.c -L./zig-out/lib -lzuda -o main
./main
```

### Python Example (ctypes)

```python
from ctypes import *

libzuda = CDLL("./zig-out/lib/libzuda.so")

# Define function signatures
libzuda.zuda_hash_map_create.restype = c_void_p
libzuda.zuda_hash_map_put.argtypes = [c_void_p, c_char_p, c_char_p]
libzuda.zuda_hash_map_get.argtypes = [c_void_p, c_char_p]
libzuda.zuda_hash_map_get.restype = c_char_p

# Use the library
hm = libzuda.zuda_hash_map_create()
libzuda.zuda_hash_map_put(hm, b"key", b"value")
result = libzuda.zuda_hash_map_get(hm, b"key")
print(result.decode())  # "value"
libzuda.zuda_hash_map_destroy(hm)
```

See `examples/FFI_README.md` for complete FFI documentation.

---

## Performance Tips

### 1. Choose the Right Data Structure

See [GUIDE.md](GUIDE.md) for decision trees.

### 2. Use Comptime Configuration

```zig
// Tune branching factor at compile time
const BTree = zuda.containers.trees.BTree;
var btree = BTree(u64, []const u8, 128, void, cmp).init(allocator); // degree=128
```

### 3. Pre-allocate Capacity

```zig
const Deque = zuda.containers.queues.Deque;
var deque = Deque(u32).init(allocator);
try deque.ensureCapacity(1000); // Avoid incremental reallocations
```

### 4. Use D-ary Heaps for Cache Locality

```zig
// 4-ary heap often faster than binary heap
const DaryHeap = zuda.containers.heaps.DaryHeap;
var heap = DaryHeap(u32, 4, void, lessThan).init(allocator);
```

### 5. Profile Before Optimizing

```zig
const start = std.time.nanoTimestamp();
// ... your code ...
const elapsed = std.time.nanoTimestamp() - start;
std.debug.print("Elapsed: {} ns\n", .{elapsed});
```

---

## Common Errors and Solutions

### Error: `OutOfMemory`
**Cause**: Allocator ran out of memory
**Solution**: Use arena allocator for batch operations, or increase heap size

### Error: `KeyNotFound`
**Cause**: Attempted to access non-existent key
**Solution**: Check with `contains()` before `get()`, or use optional return

### Error: `TreeInvariant`
**Cause**: Tree structure violated (should not happen)
**Solution**: This is a bug - please report at https://github.com/yusa-imit/zuda/issues

### Error: `CapacityExceeded`
**Cause**: Fixed-capacity container is full
**Solution**: Use dynamic container or increase capacity

---

## Next Steps

Now that you've built your first zuda application:

1. **Explore Data Structures**: [API Reference](API.md)
2. **Understand Algorithms**: [Algorithm Explainers](ALGORITHMS.md)
3. **Choose the Right Tool**: [Decision Guide](GUIDE.md)
4. **Read Examples**: Check `examples/` directory in the repository
5. **Contribute**: Found a bug or want a feature? Open an issue!

---

## Getting Help

- **Documentation**: [API.md](API.md), [ALGORITHMS.md](ALGORITHMS.md), [GUIDE.md](GUIDE.md)
- **Issues**: https://github.com/yusa-imit/zuda/issues
- **Examples**: `examples/` directory
- **Consumer Projects**: See how [zr](https://github.com/yusa-imit/zr), [silica](https://github.com/yusa-imit/silica), and [zoltraak](https://github.com/yusa-imit/zoltraak) use zuda

---

## License

MIT License - see LICENSE file for details

---

**Happy coding with zuda!** 🚀
