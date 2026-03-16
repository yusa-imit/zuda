//! Skip List - Probabilistic balanced search structure
//!
//! A skip list is a probabilistic data structure that allows O(log n) average
//! search, insertion, and deletion. It consists of multiple levels of linked
//! lists, where each level is a subset of the level below it.
//!
//! ## Time Complexity
//! - Search: O(log n) average, O(n) worst case
//! - Insert: O(log n) average, O(n) worst case
//! - Remove: O(log n) average, O(n) worst case
//! - Range iteration: O(log n + k) where k is the range size
//!
//! ## Space Complexity
//! O(n) average, O(n log n) worst case
//!
//! ## Use Cases
//! - Sorted sets with range queries
//! - Alternative to balanced BSTs when lockless concurrency is needed
//! - Databases (e.g., Redis ZSET, LevelDB memtable)

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const Order = std.math.Order;

/// Skip List with key-value pairs
///
/// ## Type Parameters
/// - K: Key type (must be hashable and comparable)
/// - V: Value type
/// - Context: Context type for comparison function
/// - compareFn: Comparison function (ctx, a, b) -> Order
///
/// ## Example
/// ```zig
/// const IntContext = struct {
///     pub fn compare(_: @This(), a: i32, b: i32) Order {
///         return std.math.order(a, b);
///     }
/// };
/// var list = SkipList(i32, []const u8, IntContext, IntContext.compare).init(allocator, .{});
/// defer list.deinit();
/// ```
pub fn SkipList(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: K, b: K) Order,
) type {
    return struct {
        const Self = @This();

        /// Maximum height of the skip list
        pub const max_level: usize = 32;

        /// Probability for level generation (p = 1/4 for good performance)
        const probability_denominator: u32 = 4;

        /// Node in the skip list
        const Node = struct {
            key: K,
            value: V,
            forward: [max_level]?*Node,

            fn init(key: K, value: V) Node {
                return .{
                    .key = key,
                    .value = value,
                    .forward = [_]?*Node{null} ** max_level,
                };
            }
        };

        pub const Entry = struct {
            key: K,
            value: V,
        };

        /// Iterator for traversing the skip list in sorted order
        pub const Iterator = struct {
            current: ?*Node,

            /// Get the next entry
            /// Time: O(1)
            pub fn next(self: *Iterator) ?Entry {
                const node = self.current orelse return null;
                self.current = node.forward[0];
                return Entry{ .key = node.key, .value = node.value };
            }
        };

        /// Range iterator for a specific key range
        pub const RangeIterator = struct {
            current: ?*Node,
            end_key: ?K,
            ctx: Context,
            inclusive: bool,

            /// Get the next entry in the range
            /// Time: O(1)
            pub fn next(self: *RangeIterator) ?Entry {
                const node = self.current orelse return null;

                if (self.end_key) |end| {
                    const cmp = compareFn(self.ctx, node.key, end);
                    if (cmp == .gt or (cmp == .eq and !self.inclusive)) {
                        return null;
                    }
                }

                self.current = node.forward[0];
                return Entry{ .key = node.key, .value = node.value };
            }
        };

        allocator: Allocator,
        header: *Node,
        level: usize,
        len: usize,
        ctx: Context,
        prng: std.Random.DefaultPrng,

        // -- Lifecycle --

        /// Initialize an empty skip list
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator, ctx: Context) !Self {
            const header = try allocator.create(Node);
            header.* = Node.init(undefined, undefined);

            // Initialize with timestamp-based seed for randomness
            const seed = @as(u64, @intCast(std.time.milliTimestamp()));
            const prng = std.Random.DefaultPrng.init(seed);

            return Self{
                .allocator = allocator,
                .header = header,
                .level = 0,
                .len = 0,
                .ctx = ctx,
                .prng = prng,
            };
        }

        /// Initialize with a specific random seed (for deterministic testing)
        /// Time: O(1) | Space: O(1)
        pub fn initWithSeed(allocator: Allocator, ctx: Context, seed: u64) !Self {
            const header = try allocator.create(Node);
            header.* = Node.init(undefined, undefined);

            const prng = std.Random.DefaultPrng.init(seed);

            return Self{
                .allocator = allocator,
                .header = header,
                .level = 0,
                .len = 0,
                .ctx = ctx,
                .prng = prng,
            };
        }

        /// Clean up all allocated memory
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            var current = self.header.forward[0];
            while (current) |node| {
                const next = node.forward[0];
                self.allocator.destroy(node);
                current = next;
            }
            self.allocator.destroy(self.header);
        }

        /// Create a deep copy of the skip list
        /// Time: O(n) | Space: O(n)
        pub fn clone(self: *const Self) !Self {
            var new_list = try Self.init(self.allocator, self.ctx);
            errdefer new_list.deinit();

            var it = self.iterator();
            while (it.next()) |entry| {
                _ = try new_list.insert(entry.key, entry.value);
            }

            return new_list;
        }

        // -- Capacity --

        /// Get the number of entries in the skip list
        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            return self.len;
        }

        /// Check if the skip list is empty
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.len == 0;
        }

        // -- Modification --

        /// Generate a random level for a new node (geometric distribution)
        fn randomLevel(self: *Self) usize {
            var lvl: usize = 0;
            const random = self.prng.random();

            while (lvl < max_level - 1 and random.uintLessThan(u32, probability_denominator) == 0) {
                lvl += 1;
            }

            return lvl;
        }

        /// Insert a key-value pair, returns the old value if key existed
        /// Time: O(log n) average, O(n) worst | Space: O(1) amortized
        pub fn insert(self: *Self, key: K, value: V) !?V {
            var update: [max_level]?*Node = [_]?*Node{null} ** max_level;
            var current = self.header;

            // Find insertion position at all levels
            var i: usize = self.level;
            while (true) : (i -= 1) {
                while (current.forward[i]) |next| {
                    const cmp = compareFn(self.ctx, next.key, key);
                    if (cmp == .lt) {
                        current = next;
                    } else {
                        break;
                    }
                }
                update[i] = current;

                if (i == 0) break;
            }

            // Check if key already exists
            if (current.forward[0]) |next| {
                if (compareFn(self.ctx, next.key, key) == .eq) {
                    const old_value = next.value;
                    next.value = value;
                    return old_value;
                }
            }

            // Create new node with random level
            const new_level = self.randomLevel();
            if (new_level > self.level) {
                i = self.level + 1;
                while (i <= new_level) : (i += 1) {
                    update[i] = self.header;
                }
                self.level = new_level;
            }

            const new_node = try self.allocator.create(Node);
            new_node.* = Node.init(key, value);

            // Update forward pointers
            i = 0;
            while (i <= new_level) : (i += 1) {
                new_node.forward[i] = update[i].?.forward[i];
                update[i].?.forward[i] = new_node;
            }

            self.len += 1;
            return null;
        }

        /// Remove a key from the skip list, returns the entry if it existed
        /// Time: O(log n) average, O(n) worst | Space: O(1)
        pub fn remove(self: *Self, key: K) ?Entry {
            var update: [max_level]?*Node = [_]?*Node{null} ** max_level;
            var current = self.header;

            // Find the node to remove at all levels
            var i: usize = self.level;
            while (true) : (i -= 1) {
                while (current.forward[i]) |next| {
                    const cmp = compareFn(self.ctx, next.key, key);
                    if (cmp == .lt) {
                        current = next;
                    } else {
                        break;
                    }
                }
                update[i] = current;

                if (i == 0) break;
            }

            // Check if the key exists
            const target = current.forward[0] orelse return null;
            if (compareFn(self.ctx, target.key, key) != .eq) {
                return null;
            }

            // Remove the node from all levels
            i = 0;
            while (i <= self.level) : (i += 1) {
                if (update[i].?.forward[i] != target) break;
                update[i].?.forward[i] = target.forward[i];
            }

            const entry = Entry{ .key = target.key, .value = target.value };
            self.allocator.destroy(target);

            // Update skip list level
            while (self.level > 0 and self.header.forward[self.level] == null) {
                self.level -= 1;
            }

            self.len -= 1;
            return entry;
        }

        /// Remove all entries from the skip list
        /// Time: O(n) | Space: O(1)
        pub fn clearRetainingCapacity(self: *Self) void {
            var current = self.header.forward[0];
            while (current) |node| {
                const next = node.forward[0];
                self.allocator.destroy(node);
                current = next;
            }

            self.header.forward = [_]?*Node{null} ** max_level;
            self.level = 0;
            self.len = 0;
        }

        // -- Lookup --

        /// Get the value associated with a key
        /// Time: O(log n) average, O(n) worst | Space: O(1)
        pub fn get(self: *const Self, key: K) ?V {
            var current = self.header;

            var i: usize = self.level;
            while (true) : (i -= 1) {
                while (current.forward[i]) |next| {
                    const cmp = compareFn(self.ctx, next.key, key);
                    if (cmp == .lt) {
                        current = next;
                    } else if (cmp == .eq) {
                        return next.value;
                    } else {
                        break;
                    }
                }

                if (i == 0) break;
            }

            return null;
        }

        /// Check if a key exists in the skip list
        /// Time: O(log n) average, O(n) worst | Space: O(1)
        pub fn contains(self: *const Self, key: K) bool {
            return self.get(key) != null;
        }

        /// Get a pointer to the value associated with a key
        /// Time: O(log n) average, O(n) worst | Space: O(1)
        pub fn getPtr(self: *Self, key: K) ?*V {
            var current = self.header;

            var i: usize = self.level;
            while (true) : (i -= 1) {
                while (current.forward[i]) |next| {
                    const cmp = compareFn(self.ctx, next.key, key);
                    if (cmp == .lt) {
                        current = next;
                    } else if (cmp == .eq) {
                        return &next.value;
                    } else {
                        break;
                    }
                }

                if (i == 0) break;
            }

            return null;
        }

        /// Get the minimum (first) entry
        /// Time: O(1) | Space: O(1)
        pub fn min(self: *const Self) ?Entry {
            const node = self.header.forward[0] orelse return null;
            return Entry{ .key = node.key, .value = node.value };
        }

        /// Get the maximum (last) entry
        /// Time: O(n) worst case, O(log n) average | Space: O(1)
        pub fn max(self: *const Self) ?Entry {
            if (self.len == 0) return null;

            var current = self.header;
            var i: usize = self.level;

            while (true) : (i -= 1) {
                while (current.forward[i] != null) {
                    current = current.forward[i].?;
                }
                if (i == 0) break;
            }

            return Entry{ .key = current.key, .value = current.value };
        }

        // -- Iteration --

        /// Get an iterator over all entries in sorted order
        /// Time: O(1) | Space: O(1)
        pub fn iterator(self: *const Self) Iterator {
            return Iterator{ .current = self.header.forward[0] };
        }

        /// Get a range iterator from start_key to end_key (inclusive)
        /// Time: O(log n) | Space: O(1)
        pub fn rangeIterator(self: *const Self, start_key: K, end_key: K, inclusive: bool) RangeIterator {
            var current = self.header;

            // Find the starting position
            var i: usize = self.level;
            while (true) : (i -= 1) {
                while (current.forward[i]) |next| {
                    const cmp = compareFn(self.ctx, next.key, start_key);
                    if (cmp == .lt) {
                        current = next;
                    } else {
                        break;
                    }
                }

                if (i == 0) break;
            }

            // Move to the first node >= start_key
            const start_node = if (current.forward[0]) |node| blk: {
                if (compareFn(self.ctx, node.key, start_key) != .lt) {
                    break :blk node;
                }
                break :blk null;
            } else null;

            return RangeIterator{
                .current = start_node,
                .end_key = end_key,
                .ctx = self.ctx,
                .inclusive = inclusive,
            };
        }

        // -- Bulk Operations --

        /// Create a skip list from a slice of entries
        /// Time: O(n log n) | Space: O(n)
        pub fn fromSlice(allocator: Allocator, ctx: Context, entries: []const Entry) !Self {
            var list = try Self.init(allocator, ctx);
            errdefer list.deinit();

            for (entries) |entry| {
                _ = try list.insert(entry.key, entry.value);
            }

            return list;
        }

        /// Convert the skip list to a sorted slice
        /// Caller owns the returned memory
        /// Time: O(n) | Space: O(n)
        pub fn toSlice(self: *const Self, allocator: Allocator) ![]Entry {
            const slice = try allocator.alloc(Entry, self.len);
            errdefer allocator.free(slice);

            var it = self.iterator();
            var i: usize = 0;
            while (it.next()) |entry| : (i += 1) {
                slice[i] = entry;
            }

            return slice;
        }

        // -- Debug --

        /// Validate skip list invariants
        /// Time: O(n) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            var node_count: usize = 0;
            var current = self.header.forward[0];
            var prev_key: ?K = null;

            while (current) |node| : (node_count += 1) {
                // Check ordering
                if (prev_key) |pk| {
                    if (compareFn(self.ctx, pk, node.key) != .lt) {
                        return error.InvalidOrder;
                    }
                }
                prev_key = node.key;
                current = node.forward[0];
            }

            // Check count
            if (node_count != self.len) {
                return error.InvalidCount;
            }
        }

        /// Format the skip list for debugging
        pub fn format(
            self: *const Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;

            try writer.print("SkipList(len={d}, level={d})", .{ self.len, self.level });
        }
    };
}

// -- Tests --

const IntContext = struct {
    pub fn compare(_: @This(), a: i32, b: i32) Order {
        return std.math.order(a, b);
    }
};

test "skip list: init and deinit" {
    var list = try SkipList(i32, []const u8, IntContext, IntContext.compare).initWithSeed(testing.allocator, .{}, 42);
    defer list.deinit();

    try testing.expectEqual(0, list.count());
    try testing.expect(list.isEmpty());
}

test "skip list: insert and get" {
    var list = try SkipList(i32, i32, IntContext, IntContext.compare).initWithSeed(testing.allocator, .{}, 42);
    defer list.deinit();

    try testing.expectEqual(null, try list.insert(5, 50));
    try testing.expectEqual(null, try list.insert(3, 30));
    try testing.expectEqual(null, try list.insert(7, 70));

    try testing.expectEqual(50, list.get(5).?);
    try testing.expectEqual(30, list.get(3).?);
    try testing.expectEqual(70, list.get(7).?);
    try testing.expectEqual(null, list.get(10));
    try testing.expectEqual(3, list.count());
}

test "skip list: insert duplicate updates value" {
    var list = try SkipList(i32, i32, IntContext, IntContext.compare).initWithSeed(testing.allocator, .{}, 42);
    defer list.deinit();

    try testing.expectEqual(null, try list.insert(5, 50));
    try testing.expectEqual(50, try list.insert(5, 100));
    try testing.expectEqual(100, list.get(5).?);
    try testing.expectEqual(1, list.count());
}

test "skip list: remove" {
    var list = try SkipList(i32, i32, IntContext, IntContext.compare).initWithSeed(testing.allocator, .{}, 42);
    defer list.deinit();

    _ = try list.insert(5, 50);
    _ = try list.insert(3, 30);
    _ = try list.insert(7, 70);

    const removed = list.remove(5).?;
    try testing.expectEqual(5, removed.key);
    try testing.expectEqual(50, removed.value);
    try testing.expectEqual(null, list.get(5));
    try testing.expectEqual(2, list.count());

    try testing.expectEqual(null, list.remove(10));
}

test "skip list: iterator order" {
    var list = try SkipList(i32, i32, IntContext, IntContext.compare).initWithSeed(testing.allocator, .{}, 42);
    defer list.deinit();

    const keys = [_]i32{ 5, 3, 7, 1, 9, 2, 8, 4, 6 };
    for (keys) |k| {
        _ = try list.insert(k, k * 10);
    }

    var it = list.iterator();
    var prev: i32 = -1;
    while (it.next()) |entry| {
        try testing.expect(entry.key > prev);
        try testing.expectEqual(entry.key * 10, entry.value);
        prev = entry.key;
    }
}

test "skip list: min and max" {
    var list = try SkipList(i32, i32, IntContext, IntContext.compare).initWithSeed(testing.allocator, .{}, 42);
    defer list.deinit();

    try testing.expectEqual(null, list.min());
    try testing.expectEqual(null, list.max());

    _ = try list.insert(5, 50);
    _ = try list.insert(3, 30);
    _ = try list.insert(7, 70);

    try testing.expectEqual(3, list.min().?.key);
    try testing.expectEqual(7, list.max().?.key);
}

test "skip list: range iterator" {
    var list = try SkipList(i32, i32, IntContext, IntContext.compare).initWithSeed(testing.allocator, .{}, 42);
    defer list.deinit();

    for (0..10) |i| {
        _ = try list.insert(@intCast(i), @intCast(i * 10));
    }

    var it = list.rangeIterator(3, 7, true);
    var count: usize = 0;
    while (it.next()) |entry| {
        try testing.expect(entry.key >= 3 and entry.key <= 7);
        count += 1;
    }
    try testing.expectEqual(5, count);

    var it2 = list.rangeIterator(3, 7, false);
    var count2: usize = 0;
    while (it2.next()) |entry| {
        try testing.expect(entry.key >= 3 and entry.key < 7);
        count2 += 1;
    }
    try testing.expectEqual(4, count2);
}

test "skip list: clear" {
    var list = try SkipList(i32, i32, IntContext, IntContext.compare).initWithSeed(testing.allocator, .{}, 42);
    defer list.deinit();

    _ = try list.insert(5, 50);
    _ = try list.insert(3, 30);
    _ = try list.insert(7, 70);

    list.clearRetainingCapacity();
    try testing.expectEqual(0, list.count());
    try testing.expect(list.isEmpty());
}

test "skip list: validate" {
    var list = try SkipList(i32, i32, IntContext, IntContext.compare).initWithSeed(testing.allocator, .{}, 42);
    defer list.deinit();

    for (0..100) |i| {
        _ = try list.insert(@intCast(i), @intCast(i * 10));
    }
    try testing.expectEqual(100, list.count());

    try list.validate();

    // Remove some elements
    for (0..50) |i| {
        _ = list.remove(@intCast(i));
    }
    try testing.expectEqual(50, list.count());

    try list.validate();

    // Verify remaining elements are 50-99
    for (50..100) |i| {
        const value = list.get(@intCast(i));
        try testing.expect(value != null);
        try testing.expectEqual(@as(i32, @intCast(i * 10)), value.?);
    }
}

test "skip list: stress test with random operations" {
    var list = try SkipList(i32, i32, IntContext, IntContext.compare).initWithSeed(testing.allocator, .{}, 42);
    defer list.deinit();

    var reference = std.AutoHashMap(i32, i32).init(testing.allocator);
    defer reference.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // Perform random operations
    for (0..1000) |_| {
        const key = random.intRangeAtMost(i32, 0, 100);
        const value = random.int(i32);

        const op = random.uintLessThan(u32, 3);
        switch (op) {
            0 => {
                // Insert
                _ = try list.insert(key, value);
                try reference.put(key, value);
            },
            1 => {
                // Remove
                _ = list.remove(key);
                _ = reference.remove(key);
            },
            2 => {
                // Get and verify
                const list_value = list.get(key);
                const ref_value = reference.get(key);
                try testing.expectEqual(ref_value, list_value);
            },
            else => unreachable,
        }
    }

    // Verify final state
    try testing.expectEqual(reference.count(), list.count());
    try list.validate();
}

test "skip list: toSlice and fromSlice" {
    var list = try SkipList(i32, i32, IntContext, IntContext.compare).initWithSeed(testing.allocator, .{}, 42);
    defer list.deinit();

    for (0..10) |i| {
        _ = try list.insert(@intCast(i), @intCast(i * 10));
    }

    const slice = try list.toSlice(testing.allocator);
    defer testing.allocator.free(slice);

    try testing.expectEqual(10, slice.len);

    var list2 = try SkipList(i32, i32, IntContext, IntContext.compare).fromSlice(testing.allocator, .{}, slice);
    defer list2.deinit();

    try testing.expectEqual(list.count(), list2.count());

    var it1 = list.iterator();
    var it2 = list2.iterator();
    while (it1.next()) |e1| {
        const e2 = it2.next().?;
        try testing.expectEqual(e1.key, e2.key);
        try testing.expectEqual(e1.value, e2.value);
    }
}
