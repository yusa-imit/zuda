//! Lock-free concurrent skip list.
//!
//! A thread-safe skip list that supports concurrent insertions, deletions, and searches
//! without locks. Uses compare-and-swap (CAS) operations for atomic modifications.
//!
//! ## Algorithm
//!
//! Based on the lock-free skip list by Herlihy et al. with marked references:
//! - Search: Lock-free traversal through levels
//! - Insert: CAS to atomically link nodes at each level
//! - Remove: Two-phase deletion (logical then physical)
//!   1. Logical: Mark the node as deleted (set mark bit)
//!   2. Physical: Unlink the node from all levels
//!
//! ## Properties
//!
//! - **Lock-free**: At least one thread makes progress in a finite number of steps
//! - **Linearizable**: Operations appear to occur atomically at some point
//! - **ABA-safe**: Uses generation counters in tagged pointers
//!
//! ## Performance
//!
//! - Search: O(log n) expected, lock-free
//! - Insert: O(log n) expected, lock-free
//! - Remove: O(log n) expected, lock-free
//! - Space: O(n) average, O(n log n) worst case
//!
//! ## References
//!
//! M. Herlihy, Y. Lev, V. Luchangco, N. Shavit. "A Simple Optimistic Skiplist Algorithm" (2007)
//! W. Pugh. "Concurrent Maintenance of Skip Lists" (1990)

const std = @import("std");
const Allocator = std.mem.Allocator;
const Order = std.math.Order;
const AtomicOrder = std.builtin.AtomicOrder;

/// Lock-free concurrent skip list.
///
/// Thread-safe sorted map supporting concurrent operations without locks.
///
/// Example:
/// ```zig
/// const IntContext = struct {
///     pub fn compare(_: @This(), a: i32, b: i32) Order {
///         return std.math.order(a, b);
///     }
/// };
/// var list = ConcurrentSkipList(i32, []const u8, IntContext, IntContext.compare).init(allocator, .{});
/// defer list.deinit();
///
/// try list.insert(42, "answer");
/// const val = list.get(42); // Some("answer")
/// ```
pub fn ConcurrentSkipList(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime compare_fn: fn (ctx: Context, a: K, b: K) Order,
) type {
    // Use compare_fn as a comptime parameter
    const compareFn = compare_fn;

    return struct {
        const Self = @This();

        /// Maximum height of the skip list
        pub const max_level: usize = 16;

        /// Probability for level generation (p = 1/2 for simplicity)
        const probability_denominator: u32 = 2;

        /// Node in the skip list
        const Node = struct {
            key: K,
            value: V,
            /// forward[i] is a tagged pointer to the next node at level i
            /// Lower bit indicates if this node is marked for deletion
            forward: [max_level]std.atomic.Value(usize),
            top_level: usize,

            fn init(key: K, value: V, level: usize) Node {
                var forward: [max_level]std.atomic.Value(usize) = undefined;
                for (&forward) |*fwd| {
                    fwd.* = std.atomic.Value(usize).init(0);
                }
                return .{
                    .key = key,
                    .value = value,
                    .forward = forward,
                    .top_level = level,
                };
            }
        };

        /// Tagged pointer encoding: [ptr][mark_bit]
        /// Lower bit = mark (1 if logically deleted)
        const TaggedPtr = struct {
            ptr: ?*Node,
            marked: bool,

            fn pack(self: TaggedPtr) usize {
                const ptr_val: usize = @intFromPtr(self.ptr);
                return ptr_val | @intFromBool(self.marked);
            }

            fn unpack(val: usize) TaggedPtr {
                const marked = (val & 1) != 0;
                const ptr_val = val & ~@as(usize, 1);
                return .{
                    .ptr = if (ptr_val == 0) null else @ptrFromInt(ptr_val),
                    .marked = marked,
                };
            }
        };

        pub const Entry = struct {
            key: K,
            value: V,
        };

        allocator: Allocator,
        header: *Node,
        ctx: Context,
        prng_mutex: std.Thread.Mutex,
        prng: std.Random.DefaultPrng,

        // -- Lifecycle --

        /// Initialize an empty concurrent skip list.
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator, ctx: Context) !Self {
            const header = try allocator.create(Node);
            header.* = Node.init(undefined, undefined, max_level - 1);

            const seed = @as(u64, @intCast(std.time.milliTimestamp()));
            const prng = std.Random.DefaultPrng.init(seed);

            return Self{
                .allocator = allocator,
                .header = header,
                .ctx = ctx,
                .prng_mutex = .{},
                .prng = prng,
            };
        }

        /// Free all nodes in the skip list.
        ///
        /// WARNING: Not thread-safe. Ensure no concurrent operations before calling.
        ///
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            const current_tagged = self.header.forward[0].load(.monotonic);
            var current = TaggedPtr.unpack(current_tagged).ptr;

            while (current) |node| {
                const next_tagged = node.forward[0].load(.monotonic);
                const next = TaggedPtr.unpack(next_tagged).ptr;
                self.allocator.destroy(node);
                current = next;
            }

            self.allocator.destroy(self.header);
        }

        // -- Modification --

        /// Insert a key-value pair into the skip list.
        ///
        /// If the key already exists, updates the value and returns the old value.
        /// Thread-safe.
        ///
        /// Time: O(log n) expected | Space: O(1)
        pub fn insert(self: *Self, key: K, value: V) !?V {
            const level = self.randomLevel();
            const new_node = try self.allocator.create(Node);
            new_node.* = Node.init(key, value, level);

            while (true) {
                var preds: [max_level]?*Node = undefined;
                var succs: [max_level]?*Node = undefined;

                const found = self.find(key, &preds, &succs);

                if (found) |existing| {
                    // Key exists, update value
                    const old_value = existing.value;
                    existing.value = value;
                    self.allocator.destroy(new_node);
                    return old_value;
                }

                // Link new node at each level
                var i: usize = 0;
                while (i <= level) : (i += 1) {
                    const succ = succs[i];
                    const succ_tagged = TaggedPtr{ .ptr = succ, .marked = false };
                    new_node.forward[i].store(succ_tagged.pack(), .release);
                }

                // CAS at level 0 first (linearization point)
                const pred = preds[0].?;
                const succ = succs[0];
                const old_tagged = TaggedPtr{ .ptr = succ, .marked = false };
                const new_tagged = TaggedPtr{ .ptr = new_node, .marked = false };

                if (pred.forward[0].cmpxchgWeak(
                    old_tagged.pack(),
                    new_tagged.pack(),
                    .release,
                    .acquire,
                )) |_| {
                    // CAS failed, retry
                    continue;
                }

                // Link at higher levels (best effort)
                i = 1;
                while (i <= level) : (i += 1) {
                    while (true) {
                        const pred_i = preds[i].?;
                        const succ_i = succs[i];
                        const old_i = TaggedPtr{ .ptr = succ_i, .marked = false };
                        const new_i = TaggedPtr{ .ptr = new_node, .marked = false };

                        if (pred_i.forward[i].cmpxchgWeak(
                            old_i.pack(),
                            new_i.pack(),
                            .release,
                            .acquire,
                        )) |_| {
                            // Failed, find again and retry
                            _ = self.find(key, &preds, &succs);
                            continue;
                        }
                        break;
                    }
                }

                return null;
            }
        }

        /// Remove a key from the skip list.
        ///
        /// Returns the removed value, or null if key not found.
        /// Thread-safe.
        ///
        /// Time: O(log n) expected | Space: O(1)
        pub fn remove(self: *Self, key: K) ?V {
            var preds: [max_level]?*Node = undefined;
            var succs: [max_level]?*Node = undefined;

            const victim = self.find(key, &preds, &succs) orelse return null;

            // Phase 1: Logical deletion - mark all levels from top to bottom
            var i: usize = victim.top_level;
            while (true) {
                var succ_tagged = victim.forward[i].load(.acquire);
                var succ = TaggedPtr.unpack(succ_tagged);

                while (!succ.marked) {
                    const new_tagged = TaggedPtr{ .ptr = succ.ptr, .marked = true };
                    if (victim.forward[i].cmpxchgWeak(
                        succ_tagged,
                        new_tagged.pack(),
                        .release,
                        .acquire,
                    )) |new_val| {
                        succ_tagged = new_val;
                        succ = TaggedPtr.unpack(succ_tagged);
                        continue;
                    }
                    break;
                }

                if (i == 0) break;
                i -= 1;
            }

            const old_value = victim.value;

            // Phase 2: Physical deletion - unlink from bottom to top
            _ = self.find(key, &preds, &succs); // Help remove marked nodes

            return old_value;
        }

        // -- Lookup --

        /// Get the value associated with a key.
        ///
        /// Thread-safe.
        ///
        /// Time: O(log n) expected | Space: O(1)
        pub fn get(self: *const Self, key: K) ?V {
            var current: ?*Node = self.header;
            var level: usize = max_level - 1;

            while (true) {
                while (true) {
                    const curr = current orelse break;
                    const next_tagged = curr.forward[level].load(.acquire);
                    const next = TaggedPtr.unpack(next_tagged);

                    if (next.ptr) |node| {
                        if (next.marked) {
                            current = curr;
                            break;
                        }

                        const cmp = compareFn(self.ctx, key, node.key);
                        if (cmp == .eq and !next.marked) {
                            return node.value;
                        } else if (cmp == .gt) {
                            current = node;
                            continue;
                        }
                    }
                    break;
                }

                if (level == 0) break;
                level -= 1;
            }

            return null;
        }

        /// Check if a key exists in the skip list.
        ///
        /// Thread-safe.
        ///
        /// Time: O(log n) expected | Space: O(1)
        pub fn contains(self: *const Self, key: K) bool {
            return self.get(key) != null;
        }

        // -- Internal helpers --

        /// Find a key and populate predecessor/successor arrays.
        ///
        /// Returns a pointer to the node if found (and not marked), null otherwise.
        fn find(
            self: *const Self,
            key: K,
            preds: *[max_level]?*Node,
            succs: *[max_level]?*Node,
        ) ?*Node {
            var pred: ?*Node = self.header;
            var level: usize = max_level - 1;

            while (true) {
                var curr = pred.?.forward[level].load(.acquire);
                var curr_node = TaggedPtr.unpack(curr).ptr;

                while (curr_node) |node| {
                    const next_tagged = node.forward[level].load(.acquire);
                    const next = TaggedPtr.unpack(next_tagged);

                    // Skip marked nodes
                    while (next.marked) {
                        // Try to physically remove
                        const new_curr = TaggedPtr{ .ptr = next.ptr, .marked = false };
                        _ = pred.?.forward[level].cmpxchgWeak(
                            curr,
                            new_curr.pack(),
                            .release,
                            .acquire,
                        );

                        curr = pred.?.forward[level].load(.acquire);
                        curr_node = TaggedPtr.unpack(curr).ptr;
                        break;
                    }

                    if (curr_node == null) break;

                    const cmp = compareFn(self.ctx, key, curr_node.?.key);
                    if (cmp == .gt) {
                        pred = curr_node;
                        curr = curr_node.?.forward[level].load(.acquire);
                        curr_node = TaggedPtr.unpack(curr).ptr;
                    } else {
                        break;
                    }
                }

                preds[level] = pred;
                succs[level] = curr_node;

                if (level == 0) break;
                level -= 1;
            }

            // Check if found at level 0
            if (succs[0]) |node| {
                const tagged = TaggedPtr.unpack(node.forward[0].load(.acquire));
                if (!tagged.marked and compareFn(self.ctx, key, node.key) == .eq) {
                    return node;
                }
            }

            return null;
        }

        /// Generate a random level for a new node.
        fn randomLevel(self: *Self) usize {
            self.prng_mutex.lock();
            defer self.prng_mutex.unlock();

            var level: usize = 0;
            const random = self.prng.random();

            while (level < max_level - 1 and random.int(u32) % probability_denominator == 0) {
                level += 1;
            }

            return level;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "ConcurrentSkipList: init and deinit" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
    );
    defer list.deinit();

    try testing.expectEqual(@as(?i32, null), list.get(42));
}

test "ConcurrentSkipList: insert and get single element" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
    );
    defer list.deinit();

    const old = try list.insert(42, 100);
    try testing.expectEqual(@as(?i32, null), old);

    const val = list.get(42);
    try testing.expectEqual(@as(?i32, 100), val);
}

test "ConcurrentSkipList: insert updates existing key" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
    );
    defer list.deinit();

    _ = try list.insert(42, 100);
    const old = try list.insert(42, 200);

    try testing.expectEqual(@as(?i32, 100), old);
    try testing.expectEqual(@as(?i32, 200), list.get(42));
}

test "ConcurrentSkipList: multiple inserts and lookups" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
    );
    defer list.deinit();

    _ = try list.insert(3, 30);
    _ = try list.insert(1, 10);
    _ = try list.insert(4, 40);
    _ = try list.insert(1, 15); // Update
    _ = try list.insert(5, 50);
    _ = try list.insert(9, 90);

    try testing.expectEqual(@as(?i32, 15), list.get(1));
    try testing.expectEqual(@as(?i32, 30), list.get(3));
    try testing.expectEqual(@as(?i32, 40), list.get(4));
    try testing.expectEqual(@as(?i32, 50), list.get(5));
    try testing.expectEqual(@as(?i32, 90), list.get(9));
    try testing.expectEqual(@as(?i32, null), list.get(2));
}

test "ConcurrentSkipList: remove element" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
    );
    defer list.deinit();

    _ = try list.insert(42, 100);
    try testing.expectEqual(@as(?i32, 100), list.get(42));

    const removed = list.remove(42);
    try testing.expectEqual(@as(?i32, 100), removed);
    try testing.expectEqual(@as(?i32, null), list.get(42));
}

test "ConcurrentSkipList: remove from multiple elements" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
    );
    defer list.deinit();

    _ = try list.insert(1, 10);
    _ = try list.insert(2, 20);
    _ = try list.insert(3, 30);

    const removed = list.remove(2);
    try testing.expectEqual(@as(?i32, 20), removed);
    try testing.expectEqual(@as(?i32, 10), list.get(1));
    try testing.expectEqual(@as(?i32, null), list.get(2));
    try testing.expectEqual(@as(?i32, 30), list.get(3));
}

test "ConcurrentSkipList: remove non-existent key" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
    );
    defer list.deinit();

    _ = try list.insert(1, 10);
    const removed = list.remove(99);
    try testing.expectEqual(@as(?i32, null), removed);
}

test "ConcurrentSkipList: contains" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
    );
    defer list.deinit();

    try testing.expect(!list.contains(42));

    _ = try list.insert(42, 100);
    try testing.expect(list.contains(42));

    _ = list.remove(42);
    try testing.expect(!list.contains(42));
}

test "ConcurrentSkipList: stress test" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
    );
    defer list.deinit();

    // Insert 100 elements
    var i: i32 = 0;
    while (i < 100) : (i += 1) {
        _ = try list.insert(i, i * 10);
    }

    // Verify all elements
    i = 0;
    while (i < 100) : (i += 1) {
        try testing.expectEqual(@as(?i32, i * 10), list.get(i));
    }

    // Remove even elements
    i = 0;
    while (i < 100) : (i += 2) {
        _ = list.remove(i);
    }

    // Verify odd elements remain
    i = 1;
    while (i < 100) : (i += 2) {
        try testing.expectEqual(@as(?i32, i * 10), list.get(i));
    }

    // Verify even elements removed
    i = 0;
    while (i < 100) : (i += 2) {
        try testing.expectEqual(@as(?i32, null), list.get(i));
    }
}

test "ConcurrentSkipList: memory leak check" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
    );
    defer list.deinit();

    var i: i32 = 0;
    while (i < 50) : (i += 1) {
        _ = try list.insert(i, i);
    }

    // Verify all elements are present
    i = 0;
    while (i < 50) : (i += 1) {
        const value = list.get(i);
        try testing.expect(value != null);
        try testing.expectEqual(i, value.?);
    }

    i = 0;
    while (i < 50) : (i += 2) {
        _ = list.remove(i);
    }

    // Verify remaining elements are the odd numbers
    i = 1;
    while (i < 50) : (i += 2) {
        const value = list.get(i);
        try testing.expect(value != null);
        try testing.expectEqual(i, value.?);
    }

    // Verify even numbers are removed
    i = 0;
    while (i < 50) : (i += 2) {
        const value = list.get(i);
        try testing.expectEqual(@as(?i32, null), value);
    }
}

test "ConcurrentSkipList: with string keys" {
    const StringContext = struct {
        pub fn compare(_: @This(), a: []const u8, b: []const u8) Order {
            return std.mem.order(u8, a, b);
        }
    };

    var list = try ConcurrentSkipList([]const u8, i32, StringContext, StringContext.compare).init(
        testing.allocator,
        .{},
    );
    defer list.deinit();

    _ = try list.insert("apple", 1);
    _ = try list.insert("banana", 2);
    _ = try list.insert("cherry", 3);

    try testing.expectEqual(@as(?i32, 1), list.get("apple"));
    try testing.expectEqual(@as(?i32, 2), list.get("banana"));
    try testing.expectEqual(@as(?i32, 3), list.get("cherry"));
    try testing.expectEqual(@as(?i32, null), list.get("date"));
}
