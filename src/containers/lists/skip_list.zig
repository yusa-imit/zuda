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

        /// Initialize with default comparison function for i32 keys
        /// Time: O(1) | Space: O(1)
        pub fn initDefault(allocator: Allocator) !Self {
            if (K == i32 or K == f64 or K == []const u8) {
                return init(allocator, {});
            } else {
                @compileError("initDefault() is only available for i32, f64, or []const u8 key types");
            }
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

// -- Tests for initDefault() convenience constructors --

/// Default comparison function for i32 keys
/// Time: O(1) | Space: O(1)
fn defaultCompareInt(_: void, a: i32, b: i32) Order {
    return std.math.order(a, b);
}

/// Default comparison function for f64 keys
/// Time: O(1) | Space: O(1)
fn defaultCompareFloat(_: void, a: f64, b: f64) Order {
    return std.math.order(a, b);
}

/// Default comparison function for string ([]const u8) keys
/// Time: O(min(a.len, b.len)) | Space: O(1)
fn defaultCompareString(_: void, a: []const u8, b: []const u8) Order {
    return std.mem.order(u8, a, b);
}

test "skip list: initDefault with i32 keys - basic operations" {
    // Test that initDefault() creates a working SkipList with i32 keys
    var list = try SkipList(i32, []const u8, void, defaultCompareInt).initDefault(testing.allocator);
    defer list.deinit();

    try testing.expectEqual(0, list.count());
    try testing.expect(list.isEmpty());

    // Insert some values
    try testing.expectEqual(null, try list.insert(5, "five"));
    try testing.expectEqual(null, try list.insert(3, "three"));
    try testing.expectEqual(null, try list.insert(7, "seven"));

    // Verify retrieval
    try testing.expectEqualStrings("five", list.get(5).?);
    try testing.expectEqualStrings("three", list.get(3).?);
    try testing.expectEqualStrings("seven", list.get(7).?);
    try testing.expectEqual(null, list.get(10));
    try testing.expectEqual(3, list.count());
}

test "skip list: initDefault with i32 keys - insert and replace" {
    var list = try SkipList(i32, i32, void, defaultCompareInt).initDefault(testing.allocator);
    defer list.deinit();

    try testing.expectEqual(null, try list.insert(42, 100));
    try testing.expectEqual(100, try list.insert(42, 200));
    try testing.expectEqual(200, list.get(42).?);
    try testing.expectEqual(1, list.count());
}

test "skip list: initDefault with i32 keys - remove operations" {
    var list = try SkipList(i32, i32, void, defaultCompareInt).initDefault(testing.allocator);
    defer list.deinit();

    _ = try list.insert(10, 100);
    _ = try list.insert(20, 200);
    _ = try list.insert(30, 300);

    const removed = list.remove(20).?;
    try testing.expectEqual(20, removed.key);
    try testing.expectEqual(200, removed.value);
    try testing.expectEqual(null, list.get(20));
    try testing.expectEqual(2, list.count());

    // Verify removal of non-existent key returns null
    try testing.expectEqual(null, list.remove(99));
}

test "skip list: initDefault with i32 keys - iterator maintains order" {
    var list = try SkipList(i32, i32, void, defaultCompareInt).initDefault(testing.allocator);
    defer list.deinit();

    const keys = [_]i32{ 15, 5, 25, 3, 17, 11, 22, 8 };
    for (keys) |k| {
        _ = try list.insert(k, k * 10);
    }

    // Verify iteration order is sorted
    var it = list.iterator();
    var prev: i32 = std.math.minInt(i32);
    while (it.next()) |entry| {
        try testing.expect(entry.key > prev);
        try testing.expectEqual(entry.key * 10, entry.value);
        prev = entry.key;
    }
}

test "skip list: initDefault with i32 keys - min and max" {
    var list = try SkipList(i32, i32, void, defaultCompareInt).initDefault(testing.allocator);
    defer list.deinit();

    try testing.expectEqual(null, list.min());
    try testing.expectEqual(null, list.max());

    _ = try list.insert(15, 150);
    _ = try list.insert(5, 50);
    _ = try list.insert(25, 250);

    try testing.expectEqual(5, list.min().?.key);
    try testing.expectEqual(25, list.max().?.key);
}

test "skip list: initDefault with i32 keys - clear and validate" {
    var list = try SkipList(i32, i32, void, defaultCompareInt).initDefault(testing.allocator);
    defer list.deinit();

    for (0..50) |i| {
        _ = try list.insert(@intCast(i), @intCast(i * 10));
    }
    try testing.expectEqual(50, list.count());

    try list.validate();

    list.clearRetainingCapacity();
    try testing.expectEqual(0, list.count());
    try testing.expect(list.isEmpty());
    try list.validate();
}

test "skip list: initDefault with f64 keys - basic operations" {
    var list = try SkipList(f64, []const u8, void, defaultCompareFloat).initDefault(testing.allocator);
    defer list.deinit();

    try testing.expectEqual(0, list.count());

    _ = try list.insert(3.14, "pi");
    _ = try list.insert(2.71, "e");
    _ = try list.insert(1.41, "sqrt2");

    try testing.expectEqualStrings("pi", list.get(3.14).?);
    try testing.expectEqualStrings("e", list.get(2.71).?);
    try testing.expectEqualStrings("sqrt2", list.get(1.41).?);
    try testing.expectEqual(null, list.get(0.0));
    try testing.expectEqual(3, list.count());
}

test "skip list: initDefault with f64 keys - insert and replace" {
    var list = try SkipList(f64, f64, void, defaultCompareFloat).initDefault(testing.allocator);
    defer list.deinit();

    try testing.expectEqual(null, try list.insert(1.5, 100.0));
    try testing.expectEqual(100.0, try list.insert(1.5, 200.0));
    try testing.expectEqual(200.0, list.get(1.5).?);
    try testing.expectEqual(1, list.count());
}

test "skip list: initDefault with f64 keys - remove operations" {
    var list = try SkipList(f64, f64, void, defaultCompareFloat).initDefault(testing.allocator);
    defer list.deinit();

    _ = try list.insert(1.1, 11.0);
    _ = try list.insert(2.2, 22.0);
    _ = try list.insert(3.3, 33.0);

    const removed = list.remove(2.2).?;
    try testing.expectEqual(2.2, removed.key);
    try testing.expectEqual(22.0, removed.value);
    try testing.expectEqual(null, list.get(2.2));
    try testing.expectEqual(2, list.count());
}

test "skip list: initDefault with f64 keys - iterator maintains order" {
    var list = try SkipList(f64, f64, void, defaultCompareFloat).initDefault(testing.allocator);
    defer list.deinit();

    const keys = [_]f64{ 1.5, 0.5, 2.5, 0.3, 1.7, 1.1, 2.2, 0.8 };
    for (keys) |k| {
        _ = try list.insert(k, k * 10.0);
    }

    var it = list.iterator();
    var prev: f64 = -1.0e10;
    while (it.next()) |entry| {
        try testing.expect(entry.key > prev);
        try testing.expectEqual(entry.key * 10.0, entry.value);
        prev = entry.key;
    }
}

test "skip list: initDefault with f64 keys - min and max" {
    var list = try SkipList(f64, f64, void, defaultCompareFloat).initDefault(testing.allocator);
    defer list.deinit();

    try testing.expectEqual(null, list.min());
    try testing.expectEqual(null, list.max());

    _ = try list.insert(1.5, 15.0);
    _ = try list.insert(0.5, 5.0);
    _ = try list.insert(2.5, 25.0);

    try testing.expectEqual(0.5, list.min().?.key);
    try testing.expectEqual(2.5, list.max().?.key);
}

test "skip list: initDefault with f64 keys - validate" {
    var list = try SkipList(f64, f64, void, defaultCompareFloat).initDefault(testing.allocator);
    defer list.deinit();

    for (0..30) |i| {
        const k = @as(f64, @floatFromInt(i)) * 0.1;
        _ = try list.insert(k, k);
    }

    try list.validate();

    for (0..15) |i| {
        const k = @as(f64, @floatFromInt(i)) * 0.1;
        _ = list.remove(k);
    }

    try list.validate();
}

test "skip list: initDefault with string keys - basic operations" {
    var list = try SkipList([]const u8, i32, void, defaultCompareString).initDefault(testing.allocator);
    defer list.deinit();

    try testing.expectEqual(0, list.count());

    _ = try list.insert("apple", 1);
    _ = try list.insert("banana", 2);
    _ = try list.insert("cherry", 3);

    try testing.expectEqual(1, list.get("apple").?);
    try testing.expectEqual(2, list.get("banana").?);
    try testing.expectEqual(3, list.get("cherry").?);
    try testing.expectEqual(null, list.get("date"));
    try testing.expectEqual(3, list.count());
}

test "skip list: initDefault with string keys - insert and replace" {
    var list = try SkipList([]const u8, []const u8, void, defaultCompareString).initDefault(testing.allocator);
    defer list.deinit();

    try testing.expectEqual(null, try list.insert("key1", "value1"));
    try testing.expectEqualStrings("value1", (try list.insert("key1", "value2")).?);
    try testing.expectEqualStrings("value2", list.get("key1").?);
    try testing.expectEqual(1, list.count());
}

test "skip list: initDefault with string keys - remove operations" {
    var list = try SkipList([]const u8, i32, void, defaultCompareString).initDefault(testing.allocator);
    defer list.deinit();

    _ = try list.insert("first", 1);
    _ = try list.insert("second", 2);
    _ = try list.insert("third", 3);

    const removed = list.remove("second").?;
    try testing.expectEqualStrings("second", removed.key);
    try testing.expectEqual(2, removed.value);
    try testing.expectEqual(null, list.get("second"));
    try testing.expectEqual(2, list.count());
}

test "skip list: initDefault with string keys - iterator maintains order" {
    var list = try SkipList([]const u8, i32, void, defaultCompareString).initDefault(testing.allocator);
    defer list.deinit();

    const keys = [_][]const u8{ "dog", "cat", "elephant", "ant", "bee", "fox" };
    for (keys, 0..) |k, i| {
        _ = try list.insert(k, @intCast(i));
    }

    var it = list.iterator();
    var prev: []const u8 = "";
    while (it.next()) |entry| {
        try testing.expect(std.mem.order(u8, prev, entry.key) != .gt);
        prev = entry.key;
    }
}

test "skip list: initDefault with string keys - min and max" {
    var list = try SkipList([]const u8, i32, void, defaultCompareString).initDefault(testing.allocator);
    defer list.deinit();

    try testing.expectEqual(null, list.min());
    try testing.expectEqual(null, list.max());

    _ = try list.insert("zebra", 3);
    _ = try list.insert("apple", 1);
    _ = try list.insert("monkey", 2);

    try testing.expectEqualStrings("apple", list.min().?.key);
    try testing.expectEqualStrings("zebra", list.max().?.key);
}

test "skip list: initDefault with string keys - validate" {
    var list = try SkipList([]const u8, i32, void, defaultCompareString).initDefault(testing.allocator);
    defer list.deinit();

    const strings = [_][]const u8{ "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf" };
    for (strings, 0..) |s, i| {
        _ = try list.insert(s, @intCast(i));
    }

    try list.validate();

    for (0..3) |i| {
        _ = list.remove(strings[i]);
    }

    try list.validate();
}

test "skip list: initDefault with i32 - memory safety (no leaks)" {
    var list = try SkipList(i32, i32, void, defaultCompareInt).initDefault(testing.allocator);
    defer list.deinit();

    for (0..200) |i| {
        _ = try list.insert(@intCast(i), @intCast(i * 2));
    }

    for (0..100) |i| {
        _ = list.remove(@intCast(i));
    }

    try list.validate();
    // std.testing.allocator will detect leaks in defer
}

test "skip list: initDefault with f64 - memory safety (no leaks)" {
    var list = try SkipList(f64, f64, void, defaultCompareFloat).initDefault(testing.allocator);
    defer list.deinit();

    for (0..200) |i| {
        const k = @as(f64, @floatFromInt(i)) * 0.1;
        _ = try list.insert(k, k * 2.0);
    }

    for (0..100) |i| {
        const k = @as(f64, @floatFromInt(i)) * 0.1;
        _ = list.remove(k);
    }

    try list.validate();
    // std.testing.allocator will detect leaks in defer
}

test "skip list: initDefault with strings - memory safety (no leaks)" {
    var list = try SkipList([]const u8, i32, void, defaultCompareString).initDefault(testing.allocator);
    defer list.deinit();

    const keys = [_][]const u8{ "aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh" };

    for (keys, 0..) |k, i| {
        _ = try list.insert(k, @intCast(i));
    }

    for (0..4) |i| {
        _ = list.remove(keys[i]);
    }

    try list.validate();
    // std.testing.allocator will detect leaks in defer
}

test "skip list: initDefault behavior identical to explicit context init with i32" {
    // Create list with initDefault
    var list1 = try SkipList(i32, i32, void, defaultCompareInt).initDefault(testing.allocator);
    defer list1.deinit();

    // Create list with explicit context
    var list2 = try SkipList(i32, i32, void, defaultCompareInt).initWithSeed(testing.allocator, {}, 42);
    defer list2.deinit();

    // Both should start empty
    try testing.expectEqual(0, list1.count());
    try testing.expectEqual(0, list2.count());

    // Insert same keys into both
    const test_keys = [_]i32{ 10, 5, 15, 3, 7, 12, 18 };
    for (test_keys) |k| {
        _ = try list1.insert(k, k);
        _ = try list2.insert(k, k);
    }

    // Both should have same count
    try testing.expectEqual(list1.count(), list2.count());

    // Both should return same values for same keys
    for (test_keys) |k| {
        try testing.expectEqual(list1.get(k), list2.get(k));
    }

    // Both should validate successfully
    try list1.validate();
    try list2.validate();

    // Iterator order should be identical
    var it1 = list1.iterator();
    var it2 = list2.iterator();
    while (it1.next()) |e1| {
        const e2 = it2.next().?;
        try testing.expectEqual(e1.key, e2.key);
        try testing.expectEqual(e1.value, e2.value);
    }
}

test "skip list: initDefault behavior identical to explicit context init with f64" {
    var list1 = try SkipList(f64, f64, void, defaultCompareFloat).initDefault(testing.allocator);
    defer list1.deinit();

    var list2 = try SkipList(f64, f64, void, defaultCompareFloat).initWithSeed(testing.allocator, {}, 42);
    defer list2.deinit();

    try testing.expectEqual(0, list1.count());
    try testing.expectEqual(0, list2.count());

    const test_keys = [_]f64{ 1.0, 0.5, 1.5, 0.3, 0.7, 1.2, 1.8 };
    for (test_keys) |k| {
        _ = try list1.insert(k, k);
        _ = try list2.insert(k, k);
    }

    try testing.expectEqual(list1.count(), list2.count());

    for (test_keys) |k| {
        try testing.expectEqual(list1.get(k), list2.get(k));
    }

    try list1.validate();
    try list2.validate();

    var it1 = list1.iterator();
    var it2 = list2.iterator();
    while (it1.next()) |e1| {
        const e2 = it2.next().?;
        try testing.expectEqual(e1.key, e2.key);
        try testing.expectEqual(e1.value, e2.value);
    }
}

test "skip list: initDefault behavior identical to explicit context init with strings" {
    var list1 = try SkipList([]const u8, i32, void, defaultCompareString).initDefault(testing.allocator);
    defer list1.deinit();

    var list2 = try SkipList([]const u8, i32, void, defaultCompareString).initWithSeed(testing.allocator, {}, 42);
    defer list2.deinit();

    try testing.expectEqual(0, list1.count());
    try testing.expectEqual(0, list2.count());

    const test_keys = [_][]const u8{ "apple", "banana", "cherry", "date", "elderberry" };
    for (test_keys, 0..) |k, i| {
        _ = try list1.insert(k, @intCast(i));
        _ = try list2.insert(k, @intCast(i));
    }

    try testing.expectEqual(list1.count(), list2.count());

    for (test_keys) |k| {
        try testing.expectEqual(list1.get(k), list2.get(k));
    }

    try list1.validate();
    try list2.validate();

    var it1 = list1.iterator();
    var it2 = list2.iterator();
    while (it1.next()) |e1| {
        const e2 = it2.next().?;
        try testing.expectEqualStrings(e1.key, e2.key);
        try testing.expectEqual(e1.value, e2.value);
    }
}

test "skip list: initDefault with i32 - stress test" {
    var list = try SkipList(i32, i32, void, defaultCompareInt).initDefault(testing.allocator);
    defer list.deinit();

    var prng = std.Random.DefaultPrng.init(999);
    const random = prng.random();

    for (0..500) |_| {
        const key = random.intRangeAtMost(i32, 0, 200);
        _ = try list.insert(key, key * 10);
    }

    try list.validate();
    try testing.expect(list.count() > 0);
}

test "skip list: initDefault with strings - contains and getPtr" {
    var list = try SkipList([]const u8, i32, void, defaultCompareString).initDefault(testing.allocator);
    defer list.deinit();

    _ = try list.insert("test", 42);

    try testing.expect(list.contains("test"));
    try testing.expect(!list.contains("missing"));

    const ptr = list.getPtr("test");
    try testing.expect(ptr != null);
    try testing.expectEqual(42, ptr.?.* );
}

test "skip list: initDefault range iterator with i32" {
    var list = try SkipList(i32, i32, void, defaultCompareInt).initDefault(testing.allocator);
    defer list.deinit();

    for (0..20) |i| {
        _ = try list.insert(@intCast(i), @intCast(i * 10));
    }

    var it = list.rangeIterator(5, 15, true);
    var count: usize = 0;
    while (it.next()) |entry| {
        try testing.expect(entry.key >= 5 and entry.key <= 15);
        count += 1;
    }
    try testing.expectEqual(11, count);
}

test "skip list: initDefault range iterator with strings" {
    var list = try SkipList([]const u8, i32, void, defaultCompareString).initDefault(testing.allocator);
    defer list.deinit();

    const keys = [_][]const u8{ "aa", "ab", "ac", "ba", "bb", "bc", "ca", "cb", "cc" };
    for (keys, 0..) |k, i| {
        _ = try list.insert(k, @intCast(i));
    }

    var it = list.rangeIterator("ab", "ca", true);
    var count: usize = 0;
    while (it.next()) |_| {
        count += 1;
    }
    try testing.expect(count > 0);
}
