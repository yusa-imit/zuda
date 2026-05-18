const std = @import("std");
const zuda = @import("zuda");

const RedBlackTree = zuda.containers.trees.RedBlackTree;

/// Demo 1: Basic Operations with Integer Keys
/// Demonstrates: init, insert, remove, iteration, min/max
fn demo1BasicOperations() !void {
    std.debug.print("\n=== Demo 1: Basic Operations ===\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Context for integer comparison
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    };

    var tree = RedBlackTree(i32, []const u8, IntContext, IntContext.compare).init(allocator, .{});
    defer tree.deinit();

    // Insert key-value pairs
    std.debug.print("Inserting: (5, \"five\"), (3, \"three\"), (7, \"seven\"), (1, \"one\")\n", .{});
    _ = try tree.insert(5, "five");
    _ = try tree.insert(3, "three");
    _ = try tree.insert(7, "seven");
    _ = try tree.insert(1, "one");

    std.debug.print("Tree size: {}\n", .{tree.count()});

    // Update existing key
    std.debug.print("\nUpdating key 5 to \"FIVE\"\n", .{});
    const old_value = try tree.insert(5, "FIVE");
    if (old_value) |old| {
        std.debug.print("Old value: {s}\n", .{old});
    }

    // Min and max
    if (tree.min()) |min_entry| {
        std.debug.print("\nMin: {} -> {s}\n", .{ min_entry.key, min_entry.value });
    }
    if (tree.max()) |max_entry| {
        std.debug.print("Max: {} -> {s}\n", .{ max_entry.key, max_entry.value });
    }

    // Iterate in sorted order (ascending)
    std.debug.print("\nIn-order traversal:\n", .{});
    var iter = try tree.iterator();
    defer iter.deinit();
    while (try iter.next()) |entry| {
        std.debug.print("  {} -> {s}\n", .{ entry.key, entry.value });
    }

    // Remove a key
    std.debug.print("\nRemoving key 3\n", .{});
    if (tree.remove(3)) |removed| {
        std.debug.print("Removed: {} -> {s}\n", .{ removed.key, removed.value });
    }
    std.debug.print("Tree size after removal: {}\n", .{tree.count()});
}

/// Demo 2: Reverse Iteration (Descending Order)
/// Demonstrates: reverseIterator for descending traversal
fn demo2ReverseIteration() !void {
    std.debug.print("\n=== Demo 2: Reverse Iteration ===\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    };

    var tree = RedBlackTree(i32, f64, IntContext, IntContext.compare).init(allocator, .{});
    defer tree.deinit();

    // Insert scores
    const scores = [_]struct { key: i32, value: f64 }{
        .{ .key = 10, .value = 85.5 },
        .{ .key = 20, .value = 92.3 },
        .{ .key = 15, .value = 78.9 },
        .{ .key = 5, .value = 95.1 },
        .{ .key = 25, .value = 88.7 },
    };

    std.debug.print("Inserting scores: ", .{});
    for (scores) |s| {
        _ = try tree.insert(s.key, s.value);
        std.debug.print("({}, {d:.1}), ", .{ s.key, s.value });
    }
    std.debug.print("\n", .{});

    // Forward iteration (ascending by key)
    std.debug.print("\nAscending order:\n", .{});
    var fwd_iter = try tree.iterator();
    defer fwd_iter.deinit();
    while (try fwd_iter.next()) |entry| {
        std.debug.print("  ID {} -> Score {d:.1}\n", .{ entry.key, entry.value });
    }

    // Reverse iteration (descending by key)
    std.debug.print("\nDescending order:\n", .{});
    var rev_iter = try tree.reverseIterator();
    defer rev_iter.deinit();
    while (try rev_iter.next()) |entry| {
        std.debug.print("  ID {} -> Score {d:.1}\n", .{ entry.key, entry.value });
    }
}

/// Demo 3: Leaderboard with Player Struct Keys
/// Demonstrates: Custom struct keys with custom compare function
fn demo3CustomKeys() !void {
    std.debug.print("\n=== Demo 3: Leaderboard with Custom Keys ===\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const Player = struct {
        id: u32,
        name: []const u8,
        score: i32,
    };

    // Context that compares players by score (descending), then by ID (ascending) for ties
    const PlayerContext = struct {
        pub fn compare(_: @This(), a: Player, b: Player) std.math.Order {
            // Higher score comes first (descending)
            const score_order = std.math.order(b.score, a.score);
            if (score_order != .eq) return score_order;
            // Tie-break by ID (ascending)
            return std.math.order(a.id, b.id);
        }
    };

    var leaderboard = RedBlackTree(Player, []const u8, PlayerContext, PlayerContext.compare).init(allocator, .{});
    defer leaderboard.deinit();

    // Insert players
    _ = try leaderboard.insert(.{ .id = 101, .name = "Alice", .score = 1500 }, "Grand Master");
    _ = try leaderboard.insert(.{ .id = 102, .name = "Bob", .score = 1200 }, "Master");
    _ = try leaderboard.insert(.{ .id = 103, .name = "Charlie", .score = 1800 }, "Legend");
    _ = try leaderboard.insert(.{ .id = 104, .name = "Diana", .score = 1200 }, "Master");
    _ = try leaderboard.insert(.{ .id = 105, .name = "Eve", .score = 900 }, "Expert");

    std.debug.print("Leaderboard (sorted by score descending, ID ascending):\n", .{});
    var iter = try leaderboard.iterator();
    defer iter.deinit();
    var rank: u32 = 1;
    while (try iter.next()) |entry| {
        std.debug.print("  Rank {}: {s} (ID {}, Score {}) -> {s}\n", .{
            rank,
            entry.key.name,
            entry.key.id,
            entry.key.score,
            entry.value,
        });
        rank += 1;
    }

    // Top player
    if (leaderboard.min()) |top| {
        std.debug.print("\nTop Player: {s} with score {}\n", .{ top.key.name, top.key.score });
    }

    // Bottom player
    if (leaderboard.max()) |bottom| {
        std.debug.print("Bottom Player: {s} with score {}\n", .{ bottom.key.name, bottom.key.score });
    }
}

/// Demo 4: Range Query Simulation (Manual Iteration)
/// Demonstrates: Using iterator with manual bounds checking for range queries
fn demo4RangeQuery() !void {
    std.debug.print("\n=== Demo 4: Range Query (Manual Bounds) ===\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    };

    var tree = RedBlackTree(i32, []const u8, IntContext, IntContext.compare).init(allocator, .{});
    defer tree.deinit();

    // Insert timestamp -> event mappings
    const events = [_]struct { timestamp: i32, event: []const u8 }{
        .{ .timestamp = 100, .event = "Start" },
        .{ .timestamp = 250, .event = "Process A" },
        .{ .timestamp = 300, .event = "Process B" },
        .{ .timestamp = 450, .event = "Process C" },
        .{ .timestamp = 600, .event = "Process D" },
        .{ .timestamp = 750, .event = "End" },
    };

    for (events) |e| {
        _ = try tree.insert(e.timestamp, e.event);
    }

    // Query range [200, 500] (inclusive)
    const range_start: i32 = 200;
    const range_end: i32 = 500;
    std.debug.print("Events in range [{}, {}]:\n", .{ range_start, range_end });

    var iter = try tree.iterator();
    defer iter.deinit();
    var found: u32 = 0;
    while (try iter.next()) |entry| {
        // Manual bounds check (tree doesn't have built-in range query)
        if (entry.key < range_start) continue; // Skip keys before range
        if (entry.key > range_end) break; // Stop after range ends
        std.debug.print("  T={} -> {s}\n", .{ entry.key, entry.value });
        found += 1;
    }
    std.debug.print("Found {} events in range\n", .{found});
}

/// Demo 5: Consumer Use Case - Sorted Set for zoltraak
/// Demonstrates: Migration path from zoltraak's HashMap + sorted ArrayList (1800 LOC)
fn demo5SortedSet() !void {
    std.debug.print("\n=== Demo 5: Sorted Set Use Case (zoltraak) ===\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const F64Context = struct {
        pub fn compare(_: @This(), a: f64, b: f64) std.math.Order {
            return std.math.order(a, b);
        }
    };

    // Red-Black Tree as sorted set (key=value, no separate value needed)
    var sorted_set = RedBlackTree(f64, void, F64Context, F64Context.compare).init(allocator, .{});
    defer sorted_set.deinit();

    std.debug.print("Sorted Set Operations (Redis ZADD equivalent):\n", .{});

    // Add members with scores
    const members = [_]f64{ 85.5, 92.3, 78.9, 95.1, 88.7, 92.3 }; // Note: 92.3 appears twice (duplicate)
    for (members) |score| {
        const old = try sorted_set.insert(score, {});
        if (old != null) {
            std.debug.print("  ZADD {d:.1} -> Already exists (duplicate)\n", .{score});
        } else {
            std.debug.print("  ZADD {d:.1} -> Added\n", .{score});
        }
    }

    std.debug.print("\nSet size: {} (duplicates ignored)\n", .{sorted_set.count()});

    // Range by rank (Redis ZRANGE 0 2 equivalent)
    std.debug.print("\nTop 3 scores (ZRANGE 0 2):\n", .{});
    var iter = try sorted_set.iterator();
    defer iter.deinit();
    var rank: u32 = 0;
    while (try iter.next()) |entry| : (rank += 1) {
        if (rank >= 3) break;
        std.debug.print("  Rank {}: {d:.1}\n", .{ rank, entry.key });
    }

    // Min/Max (Redis ZRANGEBYSCORE -inf +inf LIMIT 0 1 / WITHSCORES REV)
    if (sorted_set.min()) |min_entry| {
        std.debug.print("\nMin score (ZRANGEBYSCORE -inf +inf LIMIT 0 1): {d:.1}\n", .{min_entry.key});
    }
    if (sorted_set.max()) |max_entry| {
        std.debug.print("Max score (ZREVRANGE 0 0): {d:.1}\n", .{max_entry.key});
    }

    // Remove (Redis ZREM equivalent)
    std.debug.print("\nRemoving score 92.3 (ZREM):\n", .{});
    if (sorted_set.remove(92.3)) |removed| {
        std.debug.print("  Removed: {d:.1}\n", .{removed.key});
    }
    std.debug.print("Set size after removal: {}\n", .{sorted_set.count()});
}

pub fn main() !void {
    std.debug.print("\n╔════════════════════════════════════════════════╗\n", .{});
    std.debug.print("║   Red-Black Tree API Demonstration            ║\n", .{});
    std.debug.print("╚════════════════════════════════════════════════╝\n", .{});

    try demo1BasicOperations();
    try demo2ReverseIteration();
    try demo3CustomKeys();
    try demo4RangeQuery();
    try demo5SortedSet();

    std.debug.print("\n✓ All demos completed successfully!\n", .{});
    std.debug.print("\nAPI Summary:\n", .{});
    std.debug.print("  - init(allocator, context) → Self\n", .{});
    std.debug.print("  - insert(key, value) → !?V (returns old value if existed)\n", .{});
    std.debug.print("  - remove(key) → ?Entry (returns removed entry)\n", .{});
    std.debug.print("  - min() → ?Entry (smallest key)\n", .{});
    std.debug.print("  - max() → ?Entry (largest key)\n", .{});
    std.debug.print("  - iterator() → Iterator (ascending order)\n", .{});
    std.debug.print("  - reverseIterator() → ReverseIterator (descending order)\n", .{});
    std.debug.print("  - count() → usize\n", .{});
    std.debug.print("  - isEmpty() → bool\n", .{});
    std.debug.print("\nConsumer Use Cases:\n", .{});
    std.debug.print("  - zoltraak: Sorted set (1800 LOC) → RedBlackTree\n", .{});
    std.debug.print("    * Current: HashMap + sorted ArrayList (O(n) insert/remove)\n", .{});
    std.debug.print("    * With zuda: O(log n) insert/remove/search + sorted iteration\n", .{});
    std.debug.print("  - General: Ordered maps, priority queues, range queries\n", .{});
    std.debug.print("\nRun: zig build example-red-black-tree\n\n", .{});
}
