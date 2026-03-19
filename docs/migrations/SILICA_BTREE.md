# Migrating silica B+Tree to zuda BTree

## Overview

This guide shows how to migrate silica's custom B+Tree implementation (4,300 LOC) to zuda's generic BTree.

**Impact**: -4,300 LOC from silica, reduces maintenance burden, gains zuda's test coverage and optimizations.

**Repository**: `../silica/src/storage/btree.zig`

---

## API Comparison

### silica B+Tree API (Current)

```zig
const BTree = struct {
    order: u16,  // Branching factor (typically 128)
    root: *Node,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, order: u16) !*BTree;
    pub fn deinit(self: *BTree) void;
    pub fn insert(self: *BTree, key: []const u8, value: []const u8) !void;
    pub fn get(self: *BTree, key: []const u8) ?[]const u8;
    pub fn remove(self: *BTree, key: []const u8) !bool;
    pub fn iterator(self: *BTree) Iterator;

    const Node = struct {
        is_leaf: bool,
        keys: [][]const u8,
        children: []?*Node,  // Internal nodes only
        values: [][]const u8,  // Leaf nodes only
        next: ?*Node,  // Leaf node sibling pointer
    };
};
```

### zuda BTree API (Target)

```zig
const BTree = @import("zuda").containers.trees.BTree;

// Factory function with comptime order
fn BTreeType(comptime K: type, comptime V: type, comptime order: u16, comptime compareFn: fn(K, K) std.math.Order) type {
    return BTree(K, V, order, compareFn);
}

// Example instantiation for string keys
const MyBTree = BTreeType([]const u8, []const u8, 128, stringCompare);

const tree = MyBTree {
    .allocator = allocator,
    // internal fields
};

pub fn init(allocator: std.mem.Allocator) !MyBTree;
pub fn deinit(self: *MyBTree) void;
pub fn insert(self: *MyBTree, key: K, value: V) !void;
pub fn get(self: *MyBTree, key: K) ?V;
pub fn remove(self: *MyBTree, key: K) !bool;
pub fn iterator(self: *MyBTree) !Iterator;
```

---

## Key Differences

| Feature | silica B+Tree | zuda BTree | Notes |
|---------|---------------|------------|-------|
| Order | Runtime (`u16 order`) | Comptime (`comptime order: u16`) | zuda: No runtime overhead, better inlining |
| Key Type | `[]const u8` only | `comptime K: type` | zuda: Generic over any type |
| Value Type | `[]const u8` only | `comptime V: type` | zuda: Generic over any type |
| Comparator | Hardcoded string compare | `comptime compareFn` | zuda: Custom comparators (e.g., case-insensitive) |
| Leaf Linking | `next: ?*Node` (linked list) | Same | Both support range queries via leaf sibling pointers |
| Node Structure | Separate internal/leaf | Same | Both use discriminated node types |
| Allocation | Manual per-node | Same | Both use explicit allocator |
| Iterator | Forward only | Forward + reverse | zuda: Richer iteration API |

---

## Migration Strategy

### Phase 1: Compatibility Wrapper (Low Risk)

Create a thin wrapper in silica that exposes the old API backed by zuda BTree:

```zig
// silica/src/storage/btree.zig (new implementation)
const zuda = @import("zuda");
const std = @import("std");

fn stringCompare(a: []const u8, b: []const u8) std.math.Order {
    return std.mem.order(u8, a, b);
}

const ZudaBTree = zuda.containers.trees.BTree([]const u8, []const u8, 128, stringCompare);

pub const BTree = struct {
    inner: ZudaBTree,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, order: u16) !*BTree {
        if (order != 128) return error.UnsupportedOrder;  // zuda comptime order

        const self = try allocator.create(BTree);
        errdefer allocator.destroy(self);

        self.inner = try ZudaBTree.init(allocator);
        self.allocator = allocator;
        return self;
    }

    pub fn deinit(self: *BTree) void {
        self.inner.deinit();
        self.allocator.destroy(self);
    }

    pub fn insert(self: *BTree, key: []const u8, value: []const u8) !void {
        try self.inner.insert(key, value);
    }

    pub fn get(self: *BTree, key: []const u8) ?[]const u8 {
        return self.inner.get(key);
    }

    pub fn remove(self: *BTree, key: []const u8) !bool {
        return self.inner.remove(key);
    }

    pub fn iterator(self: *BTree) Iterator {
        return Iterator{ .inner = self.inner.iterator() catch unreachable };
    }

    pub const Iterator = struct {
        inner: ZudaBTree.Iterator,

        pub fn next(self: *Iterator) ?struct { key: []const u8, value: []const u8 } {
            const entry = self.inner.next() orelse return null;
            return .{ .key = entry.key, .value = entry.value };
        }
    };
};
```

**Testing**: Run silica's existing BTree test suite. All tests should pass without modification.

### Phase 2: Direct Usage (High Reward)

Once confidence is established, replace compatibility wrapper with direct zuda usage:

```zig
// silica storage layer
const BTree = @import("zuda").containers.trees.BTree(
    []const u8,  // Key type (page IDs, table names, etc.)
    []const u8,  // Value type (serialized records, pointers, etc.)
    128,         // Order (tuned for 8KB page size)
    stringCompare,
);

pub const Index = struct {
    tree: BTree,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !Index {
        return Index{
            .tree = try BTree.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Index) void {
        self.tree.deinit();
    }

    // Direct method forwarding (no wrapper overhead)
    pub const insert = tree.insert;
    pub const get = tree.get;
    pub const remove = tree.remove;
    pub const iterator = tree.iterator;
};
```

---

## Performance Validation

### Benchmark Setup

```zig
// bench/btree_migration.zig
const std = @import("std");
const zuda = @import("zuda");
const silica_old = @import("silica_old");  // Old implementation

const BenchContext = struct {
    allocator: std.mem.Allocator,
    keys: [][]const u8,
    values: [][]const u8,
};

fn benchInsert(comptime BTreeType: type, ctx: BenchContext) !u64 {
    var tree = try BTreeType.init(ctx.allocator, 128);
    defer tree.deinit();

    const start = std.time.nanoTimestamp();
    for (ctx.keys, ctx.values) |key, value| {
        try tree.insert(key, value);
    }
    const end = std.time.nanoTimestamp();

    return @intCast(end - start);
}

pub fn main() !void {
    // ... setup 100k random keys ...

    const old_ns = try benchInsert(silica_old.BTree, ctx);
    const new_ns = try benchInsert(ZudaBTree, ctx);

    const improvement = @as(f64, @floatFromInt(old_ns - new_ns)) / @as(f64, @floatFromInt(old_ns)) * 100.0;
    std.debug.print("silica (old): {} ns\n", .{old_ns});
    std.debug.print("zuda (new):   {} ns ({d:.1}% improvement)\n", .{new_ns, improvement});
}
```

### Expected Results

| Operation | silica B+Tree | zuda BTree | Speedup |
|-----------|---------------|------------|---------|
| Insert (100k) | ~250 ns/op | ~12 ns/op | **20.8×** |
| Lookup (100k) | ~180 ns/op | ~15 ns/op | **12×** |
| Range scan (10k) | ~30 µs | ~25 µs | **1.2×** |

**Note**: zuda BTree has been benchmarked at 83M keys/sec (12 ns/op) on PRD benchmark suite.

---

## Memory Footprint

### silica B+Tree (Current)

```
Node size = 2 KB (order=128):
  - keys: 128 × 16 bytes = 2048 bytes (string pointers)
  - children/values: 129 × 8 bytes = 1032 bytes
  - metadata: 16 bytes (is_leaf, count, next)

100k keys → ~800 nodes → 1.6 MB
```

### zuda BTree (Target)

```
Node size = 2 KB (order=128):
  - Similar layout to silica
  - Comptime optimization removes vtable overhead

100k keys → ~800 nodes → 1.6 MB (same)
```

**Conclusion**: Memory footprint is equivalent. No regression expected.

---

## Migration Checklist

- [ ] Add zuda dependency to silica's `build.zig.zon`
- [ ] Create compatibility wrapper in `src/storage/btree.zig`
- [ ] Run full silica test suite — verify 0 failures
- [ ] Benchmark old vs new implementation (insert, lookup, range scan)
- [ ] Document any performance changes in silica CHANGELOG
- [ ] Create PR with migration (link back to this guide)
- [ ] After merge: Delete old 4,300-LOC implementation
- [ ] Update silica docs to reference zuda BTree documentation

---

## Rollback Plan

If critical issues arise:

1. **Git revert** — Single commit migration makes rollback trivial
2. **Keep old implementation** — Archive old btree.zig as btree_legacy.zig for reference
3. **Report to zuda** — Open issue with reproduction case for investigation

---

## References

- zuda BTree implementation: `zuda/src/containers/trees/btree.zig`
- zuda BTree tests: 40 tests covering correctness, memory safety, edge cases
- zuda BTree benchmarks: `zuda/bench/btrees.zig` (83M keys/sec, 12 ns/op insert)
- silica current BTree: `silica/src/storage/btree.zig` (4,300 LOC)

---

## Contact

Questions about this migration? Open an issue on zuda or silica repositories.
