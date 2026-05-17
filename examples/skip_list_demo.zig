//! SkipList API Demo — Practical examples of probabilistic balanced tree
//!
//! Demonstrates:
//! - Sorted key-value storage with O(log n) operations
//! - Custom comparator functions
//! - Range queries and iteration
//! - Ordered set semantics (used by zoltraak sorted set)
//!
//! Run: `zig build example-skip-list`

const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== SkipList API Demo ===\n\n", .{});

    try demo1_basic_operations(allocator);
    try demo2_custom_comparator(allocator);
    try demo3_range_queries(allocator);
    try demo4_leaderboard(allocator);

    std.debug.print("\n=== API Summary ===\n", .{});
    std.debug.print("SkipList(K, V, Context):\n", .{});
    std.debug.print("  init(allocator) → O(1)\n", .{});
    std.debug.print("  insert(key, value) → O(log n) amortized\n", .{});
    std.debug.print("  get(key) → O(log n)\n", .{});
    std.debug.print("  remove(key) → O(log n)\n", .{});
    std.debug.print("  contains(key) → O(log n)\n", .{});
    std.debug.print("  iterator() → O(1), yields sorted order\n", .{});
    std.debug.print("  count() → O(1)\n\n", .{});
    std.debug.print("Properties:\n", .{});
    std.debug.print("  - Probabilistic: avg height O(log n), worst O(n)\n", .{});
    std.debug.print("  - Lock-free friendly: tower structure supports concurrent reads\n", .{});
    std.debug.print("  - Cache-efficient: fewer pointer hops than RBTree\n", .{});
    std.debug.print("  - Sorted iteration: inorder traversal via level-0 links\n", .{});
}

/// Demo 1: Basic CRUD operations with integer keys
fn demo1_basic_operations(allocator: std.mem.Allocator) !void {
    std.debug.print("Demo 1: Basic Operations (Integer Key-Value Store)\n", .{});
    std.debug.print("─────────────────────────────────────────────────\n", .{});

    const SkipList = zuda.containers.lists.SkipList(i32, []const u8, i32AscContext, i32AscContext.compare);
    var list = try SkipList.init(allocator, .{});
    defer list.deinit();

    // Insert key-value pairs
    _ = try list.insert(10, "apple");
    _ = try list.insert(5, "banana");
    _ = try list.insert(20, "cherry");
    _ = try list.insert(15, "date");

    std.debug.print("After inserting 4 items:\n", .{});
    std.debug.print("  count() = {}\n", .{list.count()});
    std.debug.print("  contains(10) = {}\n", .{list.contains(10)});
    std.debug.print("  contains(99) = {}\n", .{list.contains(99)});

    // Lookups
    std.debug.print("\nLookups:\n", .{});
    if (list.get(10)) |value| {
        std.debug.print("  get(10) = \"{s}\"\n", .{value});
    }
    if (list.get(5)) |value| {
        std.debug.print("  get(5) = \"{s}\"\n", .{value});
    }

    // Sorted iteration
    std.debug.print("\nSorted iteration (ascending):\n", .{});
    var iter = list.iterator();
    while (iter.next()) |entry| {
        std.debug.print("  {} → \"{s}\"\n", .{ entry.key, entry.value });
    }

    // Remove
    const removed = list.remove(15);
    std.debug.print("\nAfter remove(15): removed = {}\n", .{removed != null});
    std.debug.print("  count() = {}\n", .{list.count()});

    std.debug.print("\n", .{});
}

/// Demo 2: Custom comparator for descending order
fn demo2_custom_comparator(allocator: std.mem.Allocator) !void {
    std.debug.print("Demo 2: Custom Comparator (Descending Order)\n", .{});
    std.debug.print("─────────────────────────────────────────────────\n", .{});

    const SkipList = zuda.containers.lists.SkipList(f64, []const u8, f64DescContext, f64DescContext.compare);
    var list = try SkipList.init(allocator, .{});
    defer list.deinit();

    _ = try list.insert(3.14, "pi");
    _ = try list.insert(2.71, "e");
    _ = try list.insert(1.41, "sqrt2");
    _ = try list.insert(1.61, "phi");

    std.debug.print("Descending iteration (largest to smallest):\n", .{});
    var iter = list.iterator();
    while (iter.next()) |entry| {
        std.debug.print("  {d:.2} → \"{s}\"\n", .{ entry.key, entry.value });
    }

    std.debug.print("\n", .{});
}

/// Demo 3: Range queries (find all keys in [min, max])
fn demo3_range_queries(allocator: std.mem.Allocator) !void {
    std.debug.print("Demo 3: Range Queries (Score Range)\n", .{});
    std.debug.print("─────────────────────────────────────────────────\n", .{});

    const SkipList = zuda.containers.lists.SkipList(i32, []const u8, i32AscContext, i32AscContext.compare);
    var list = try SkipList.init(allocator, .{});
    defer list.deinit();

    // Insert student scores
    _ = try list.insert(95, "Alice");
    _ = try list.insert(72, "Bob");
    _ = try list.insert(88, "Charlie");
    _ = try list.insert(60, "Dave");
    _ = try list.insert(81, "Eve");

    std.debug.print("All students:\n", .{});
    var iter = list.iterator();
    while (iter.next()) |entry| {
        std.debug.print("  Score {} → {s}\n", .{ entry.key, entry.value });
    }

    // Range query: scores in [70, 90]
    std.debug.print("\nStudents with scores in [70, 90]:\n", .{});
    iter = list.iterator();
    while (iter.next()) |entry| {
        if (entry.key >= 70 and entry.key <= 90) {
            std.debug.print("  Score {} → {s}\n", .{ entry.key, entry.value });
        }
    }

    std.debug.print("\n", .{});
}

/// Demo 4: Leaderboard (zoltraak sorted set use case)
fn demo4_leaderboard(allocator: std.mem.Allocator) !void {
    std.debug.print("Demo 4: Leaderboard (Sorted Set for Rankings)\n", .{});
    std.debug.print("─────────────────────────────────────────────────\n", .{});

    // In zoltraak, sorted set uses HashMap + sorted ArrayList
    // SkipList provides a more efficient alternative: O(log n) insert vs O(n) for sorted ArrayList

    const Player = struct {
        name: []const u8,
        level: u32,
    };

    const SkipList = zuda.containers.lists.SkipList(u64, Player, u64DescContext, u64DescContext.compare);
    var leaderboard = try SkipList.init(allocator, .{});
    defer leaderboard.deinit();

    // Insert players with scores (key = score, value = player info)
    _ = try leaderboard.insert(12500, .{ .name = "Alice", .level = 45 });
    _ = try leaderboard.insert(9800, .{ .name = "Bob", .level = 38 });
    _ = try leaderboard.insert(15300, .{ .name = "Charlie", .level = 52 });
    _ = try leaderboard.insert(11200, .{ .name = "Dave", .level = 42 });
    _ = try leaderboard.insert(13700, .{ .name = "Eve", .level = 48 });

    std.debug.print("Top 5 Players (descending by score):\n", .{});
    var iter = leaderboard.iterator();
    var rank: u32 = 1;
    while (iter.next()) |entry| {
        std.debug.print("  #{}: {s} (Level {}) — {} points\n", .{
            rank,
            entry.value.name,
            entry.value.level,
            entry.key,
        });
        rank += 1;
        if (rank > 5) break;
    }

    // Update score: remove old entry, insert new one
    std.debug.print("\n⚡ Bob scores 2000 more points!\n", .{});
    _ = leaderboard.remove(9800);
    _ = try leaderboard.insert(11800, .{ .name = "Bob", .level = 38 });

    std.debug.print("\nUpdated Top 5:\n", .{});
    iter = leaderboard.iterator();
    rank = 1;
    while (iter.next()) |entry| {
        std.debug.print("  #{}: {s} (Level {}) — {} points\n", .{
            rank,
            entry.value.name,
            entry.value.level,
            entry.key,
        });
        rank += 1;
        if (rank > 5) break;
    }

    std.debug.print("\nAdvantages over zoltraak's HashMap+ArrayList approach:\n", .{});
    std.debug.print("  - Insert: O(log n) vs O(n) (no array shifting)\n", .{});
    std.debug.print("  - Remove: O(log n) vs O(n)\n", .{});
    std.debug.print("  - Range queries: O(k + log n) for k results\n", .{});
    std.debug.print("  - Memory: ~2× overhead (tower) vs 2× (hash + array)\n", .{});

    std.debug.print("\n", .{});
}

// Context for ascending integer keys
const i32AscContext = struct {
    pub fn hash(_: @This(), key: i32) u64 {
        return @intCast(@abs(key));
    }
    pub fn eql(_: @This(), a: i32, b: i32) bool {
        return a == b;
    }
    pub fn compare(_: @This(), a: i32, b: i32) std.math.Order {
        return std.math.order(a, b);
    }
};

// Context for descending f64 keys
const f64DescContext = struct {
    pub fn hash(_: @This(), key: f64) u64 {
        return @bitCast(key);
    }
    pub fn eql(_: @This(), a: f64, b: f64) bool {
        return a == b;
    }
    pub fn compare(_: @This(), a: f64, b: f64) std.math.Order {
        return std.math.order(b, a); // Reversed for descending
    }
};

// Context for descending u64 keys
const u64DescContext = struct {
    pub fn hash(_: @This(), key: u64) u64 {
        return key;
    }
    pub fn eql(_: @This(), a: u64, b: u64) bool {
        return a == b;
    }
    pub fn compare(_: @This(), a: u64, b: u64) std.math.Order {
        return std.math.order(b, a); // Reversed for descending
    }
};
