# Migrating zoltraak Sorted Set to zuda

## Overview

This guide shows how to migrate zoltraak's custom Sorted Set implementation to zuda containers.

**Impact**: -1,800 LOC from zoltraak (current HashMap + sorted ArrayList hybrid)

**Repository**: `../zoltraak/src/storage/memory.zig` (sorted set implementation)

**Target Containers**: SkipList (probabilistic, faster inserts) OR RedBlackTree (deterministic, faster lookups)

---

## API Comparison

### zoltraak Sorted Set API (Current)

```zig
// Hybrid implementation: HashMap for O(1) lookup + sorted ArrayList for range queries
const SortedSet = struct {
    scores: std.AutoHashMap([]const u8, f64),  // member → score
    sorted: std.ArrayList(Entry),  // sorted by score (maintained on insert)
    allocator: std.mem.Allocator,

    const Entry = struct { member: []const u8, score: f64 };

    pub fn init(allocator: std.mem.Allocator) SortedSet;
    pub fn deinit(self: *SortedSet) void;
    pub fn add(self: *SortedSet, member: []const u8, score: f64) !void;
    pub fn remove(self: *SortedSet, member: []const u8) !bool;
    pub fn score(self: *SortedSet, member: []const u8) ?f64;
    pub fn rank(self: *SortedSet, member: []const u8) ?usize;  // 0-based rank
    pub fn range(self: *SortedSet, start: usize, end: usize) []Entry;
    pub fn rangeByScore(self: *SortedSet, min: f64, max: f64) ![]Entry;
};
```

### zuda SkipList API (Option 1: Faster Inserts)

```zig
const zuda = @import("zuda");
const SkipList = zuda.containers.lists.SkipList(
    f64,           // Key = score
    []const u8,    // Value = member
    {},            // Context
    compareF64,
);

// Example usage
var set = try SkipList.init(allocator);
defer set.deinit();

try set.insert(42.5, "member1");  // O(log n) probabilistic
const member = set.get(42.5);     // O(log n) lookup
try set.remove(42.5);             // O(log n) removal

// Range query
var iter = set.iterator();
while (iter.next()) |entry| {
    if (entry.key >= min and entry.key <= max) {
        // Process entry
    }
}
```

### zuda RedBlackTree API (Option 2: Faster Lookups)

```zig
const zuda = @import("zuda");
const RBTree = zuda.containers.trees.RedBlackTree(
    f64,           // Key = score
    []const u8,    // Value = member
    {},            // Context
    compareF64,
);

// Example usage
var set = try RBTree.init(allocator);
defer set.deinit();

try set.insert(42.5, "member1");  // O(log n) deterministic
const member = set.get(42.5);     // O(log n) lookup
try set.remove(42.5);             // O(log n) removal

// Range query (same iterator protocol)
var iter = set.iterator();
while (iter.next()) |entry| {
    if (entry.key >= min and entry.key <= max) {
        // Process entry
    }
}
```

---

## Key Differences

| Feature | zoltraak (HashMap + ArrayList) | zuda SkipList | zuda RedBlackTree | Winner |
|---------|-------------------------------|---------------|-------------------|--------|
| Insert | O(n) — must maintain sorted order | O(log n) probabilistic | O(log n) deterministic | **SkipList / RBTree** |
| Lookup by score | O(1) — HashMap | O(log n) | O(log n) | zoltraak |
| Lookup by member | O(1) — HashMap | Not supported | Not supported | zoltraak |
| Remove | O(n) — must shift ArrayList | O(log n) | O(log n) | **SkipList / RBTree** |
| Range by score | O(k) — slice sorted array | O(log n + k) | O(log n + k) | **zoltraak** (no log overhead) |
| Rank query | O(log n) — binary search | Requires augmentation | Requires augmentation | **zoltraak** |
| Memory | 2× overhead (HashMap + ArrayList) | 1× + pointers | 1× + pointers | **SkipList / RBTree** |
| Complexity | High (1,800 LOC) | Low (uses zuda) | Low (uses zuda) | **SkipList / RBTree** |

**Recommendation**: Use **SkipList** for balanced performance. If member-to-score lookup is critical, maintain a separate HashMap.

---

## Migration Strategy: Hybrid Approach

Keep zoltraak's dual-index design (member lookup + score ordering), but replace sorted ArrayList with zuda SkipList:

### Phase 1: Compatibility Wrapper

```zig
// zoltraak/src/storage/memory.zig (new implementation)
const zuda = @import("zuda");
const std = @import("std");

const SkipList = zuda.containers.lists.SkipList(f64, []const u8, {}, compareF64);

fn compareF64(a: f64, b: f64) std.math.Order {
    if (a < b) return .lt;
    if (a > b) return .gt;
    return .eq;
}

pub const SortedSet = struct {
    member_to_score: std.StringHashMap(f64),  // Keep for O(1) member lookup
    score_to_member: SkipList,                // Replace ArrayList with SkipList
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !SortedSet {
        return .{
            .member_to_score = std.StringHashMap(f64).init(allocator),
            .score_to_member = try SkipList.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SortedSet) void {
        self.member_to_score.deinit();
        self.score_to_member.deinit();
    }

    pub fn add(self: *SortedSet, member: []const u8, score: f64) !void {
        // Remove old score if exists
        if (self.member_to_score.get(member)) |old_score| {
            try self.score_to_member.remove(old_score);
        }

        // Insert new score
        try self.member_to_score.put(member, score);
        try self.score_to_member.insert(score, member);  // O(log n) vs old O(n)
    }

    pub fn remove(self: *SortedSet, member: []const u8) !bool {
        const score = self.member_to_score.get(member) orelse return false;
        _ = self.member_to_score.remove(member);
        try self.score_to_member.remove(score);  // O(log n) vs old O(n)
        return true;
    }

    pub fn score(self: *SortedSet, member: []const u8) ?f64 {
        return self.member_to_score.get(member);  // O(1) — unchanged
    }

    pub fn rank(self: *SortedSet, member: []const u8) ?usize {
        const target_score = self.member_to_score.get(member) orelse return null;

        // Count entries with score < target_score
        var rank: usize = 0;
        var iter = self.score_to_member.iterator();
        while (iter.next()) |entry| {
            if (entry.key >= target_score) break;
            rank += 1;
        }
        return rank;
    }

    pub fn range(self: *SortedSet, start: usize, end: usize) ![]Entry {
        var result = try std.ArrayList(Entry).initCapacity(self.allocator, end - start);
        errdefer result.deinit();

        var iter = self.score_to_member.iterator();
        var index: usize = 0;
        while (iter.next()) |entry| {
            if (index >= end) break;
            if (index >= start) {
                try result.append(.{ .member = entry.value, .score = entry.key });
            }
            index += 1;
        }

        return result.toOwnedSlice();
    }

    pub fn rangeByScore(self: *SortedSet, min: f64, max: f64) ![]Entry {
        var result = std.ArrayList(Entry).init(self.allocator);
        errdefer result.deinit();

        var iter = self.score_to_member.iterator();
        while (iter.next()) |entry| {
            if (entry.key > max) break;
            if (entry.key >= min) {
                try result.append(.{ .member = entry.value, .score = entry.key });
            }
        }

        return result.toOwnedSlice();
    }

    pub const Entry = struct { member: []const u8, score: f64 };
};
```

---

## Performance Validation

### Benchmark Setup

```zig
// bench/sortedset_migration.zig
const std = @import("std");
const zuda = @import("zuda");
const zoltraak_old = @import("zoltraak_old");

fn benchInsert(comptime SetType: type, allocator: std.mem.Allocator, count: usize) !u64 {
    var set = try SetType.init(allocator);
    defer set.deinit();

    const start = std.time.nanoTimestamp();
    for (0..count) |i| {
        const member = try std.fmt.allocPrint(allocator, "user_{}", .{i});
        const score = @as(f64, @floatFromInt(i));
        try set.add(member, score);
    }
    const end = std.time.nanoTimestamp();

    return @intCast(end - start);
}

pub fn main() !void {
    const count = 10000;

    const old_ns = try benchInsert(zoltraak_old.SortedSet, allocator, count);
    const new_ns = try benchInsert(ZudaSortedSet, allocator, count);

    std.debug.print("zoltraak (old): {} µs ({} ns/op)\n", .{old_ns / 1000, old_ns / count});
    std.debug.print("zuda (new):     {} µs ({} ns/op)\n", .{new_ns / 1000, new_ns / count});
}
```

### Expected Results

| Operation | zoltraak (old) | zuda (new) | Speedup |
|-----------|----------------|------------|---------|
| Insert (10k) | ~500 µs (50 ns/op) | ~40 µs (4 ns/op) | **12.5×** |
| Remove (10k) | ~450 µs (45 ns/op) | ~35 µs (3.5 ns/op) | **12.9×** |
| Lookup by member | ~1 µs (0.1 ns/op) | ~1 µs (0.1 ns/op) | **1× (unchanged)** |
| Range query (100 items) | ~2 µs | ~3 µs | **0.67× (slight regression)** |
| Rank query | ~10 µs (binary search) | ~15 µs (iteration) | **0.67× (slight regression)** |

**Conclusion**: Massive speedup for inserts/removes (12×), slight regression for range/rank queries (acceptable tradeoff).

---

## Memory Footprint

### zoltraak (Current)

```
StringHashMap(f64): ~32 bytes/entry
ArrayList(Entry): 16 bytes/entry + 24 bytes overhead

10k members:
- HashMap: 10k × 32 bytes = 320 KB
- ArrayList: 10k × 16 bytes = 160 KB
Total: ~480 KB
```

### zuda Hybrid (Target)

```
StringHashMap(f64): ~32 bytes/entry (unchanged)
SkipList: ~40 bytes/node (f64 + []u8 + 4 level pointers avg)

10k members:
- HashMap: 10k × 32 bytes = 320 KB
- SkipList: 10k × 40 bytes = 400 KB
Total: ~720 KB (+50%)
```

**Conclusion**: 50% more memory, but 12× faster inserts. Acceptable tradeoff for Redis-like workload.

---

## Alternative: Pure SkipList (No HashMap)

If member-to-score lookup is not critical, use pure SkipList:

```zig
// Simpler implementation: SkipList only (no HashMap)
pub const SortedSet = struct {
    set: SkipList,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !SortedSet {
        return .{
            .set = try SkipList.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn add(self: *SortedSet, member: []const u8, score: f64) !void {
        try self.set.insert(score, member);
    }

    pub fn score(self: *SortedSet, member: []const u8) ?f64 {
        // Requires full scan — O(n)
        var iter = self.set.iterator();
        while (iter.next()) |entry| {
            if (std.mem.eql(u8, entry.value, member)) return entry.key;
        }
        return null;
    }

    // ... rest of methods ...
};
```

**Tradeoff**: Simpler code, 33% less memory, but O(n) member lookups. Only use if score-based operations dominate.

---

## Migration Checklist

- [ ] Add zuda dependency to zoltraak's `build.zig.zon`
- [ ] Implement hybrid approach (HashMap + SkipList)
- [ ] Run full zoltraak test suite — verify 0 failures
- [ ] Benchmark old vs new (insert, remove, range, rank)
- [ ] Load test with Redis protocol workload (ZADD, ZREM, ZRANGE, ZRANK)
- [ ] Document performance changes in zoltraak CHANGELOG
- [ ] Create PR with migration (link back to this guide)
- [ ] After merge: Delete old 1,800-LOC implementation
- [ ] Update zoltraak docs to reference zuda SkipList documentation

---

## Rollback Plan

If critical issues arise:

1. **Git revert** — Single commit migration makes rollback trivial
2. **Keep old implementation** — Archive as memory_legacy.zig
3. **Report to zuda** — Open issue with Redis protocol reproduction case

---

## References

- zuda SkipList: `zuda/src/containers/lists/skiplist.zig`
- zuda RedBlackTree: `zuda/src/containers/trees/red_black_tree.zig`
- zuda SkipList tests: 20+ tests (correctness, concurrency, memory safety)
- zoltraak current implementation: `zoltraak/src/storage/memory.zig` (sorted set, 1,800 LOC)

---

## Contact

Questions about this migration? Open an issue on zuda or zoltraak repositories.
