//! Collection Builders
//!
//! Builder utilities provide ergonomic ways to construct containers from slices.
//! These functions support conversion from homogeneous or key-value slices to various
//! container types (ArrayList, sets, maps, sorted lists).
//!
//! ## Examples
//!
//! ```zig
//! const allocator = std.testing.allocator;
//!
//! // Slice to ArrayList
//! var list = try builder.fromSlice(i32, &[_]i32{ 1, 2, 3 }).toArrayList(allocator);
//! defer list.deinit(allocator);
//!
//! // Key-value pairs to HashMap
//! const pairs = [_]struct { key: []const u8, value: i32 }{
//!     .{ .key = "a", .value = 1 },
//!     .{ .key = "b", .value = 2 },
//! };
//! var map = try builder.fromSlice(
//!     Pair([]const u8, i32),
//!     &pairs
//! ).toHashMap(allocator, hash.string, string_eql);
//! defer map.deinit();
//! ```

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const Order = std.math.Order;

// Re-export commonly used types
const SkipList = @import("../containers/lists/skip_list.zig").SkipList;
const RedBlackTree = @import("../containers/trees/red_black_tree.zig").RedBlackTree;
const RobinHoodHashMap = @import("../containers/hashing/robin_hood_hash_map.zig").RobinHoodHashMap;
const cmp = @import("compare.zig");
const h = @import("hash.zig");

/// Pair structure for key-value associations
pub fn Pair(comptime K: type, comptime V: type) type {
    return struct {
        key: K,
        value: V,
    };
}

/// SliceBuilder for homogeneous items
pub fn SliceBuilder(comptime T: type) type {
    return struct {
        const Self = @This();

        items: []const T,

        /// Convert to ArrayList
        /// Time: O(n) | Space: O(n)
        pub fn toArrayList(self: Self, allocator: Allocator) !std.ArrayList(T) {
            var list = std.ArrayList(T).initCapacity(allocator, self.items.len) catch |err| return err;
            errdefer list.deinit(allocator);
            try list.appendSlice(allocator, self.items);
            return list;
        }

        /// Convert to SkipList with custom comparator
        /// Time: O(n log n) average | Space: O(n)
        pub fn toSkipList(
            self: Self,
            allocator: Allocator,
            ctx: anytype,
            compareFn: fn (@TypeOf(ctx), T, T) Order,
        ) !SkipList(T, void, @TypeOf(ctx), compareFn) {
            var list = try SkipList(T, void, @TypeOf(ctx), compareFn).init(allocator, ctx);
            errdefer list.deinit();
            for (self.items) |item| {
                _ = try list.insert(item, {});
            }
            return list;
        }

        /// Convert key-value pairs to RedBlackTree
        /// Time: O(n log n) | Space: O(n)
        pub fn toRedBlackTree(
            self: Self,
            comptime K: type,
            comptime V: type,
            allocator: Allocator,
            ctx: anytype,
            compareFn: fn (@TypeOf(ctx), K, K) Order,
        ) !RedBlackTree(K, V, @TypeOf(ctx), compareFn) {
            var tree = RedBlackTree(K, V, @TypeOf(ctx), compareFn).init(allocator, ctx);
            errdefer tree.deinit();
            for (self.items) |pair| {
                _ = try tree.insert(pair.key, pair.value);
            }
            return tree;
        }

        /// Convert to RobinHoodHashMap
        /// Time: O(n) average | Space: O(n)
        pub fn toHashMap(
            self: Self,
            comptime K: type,
            comptime V: type,
            allocator: Allocator,
            ctx: anytype,
            hashFn: fn (@TypeOf(ctx), K) u64,
            eqlFn: fn (@TypeOf(ctx), K, K) bool,
        ) !RobinHoodHashMap(K, V, @TypeOf(ctx), hashFn, eqlFn) {
            var map = try RobinHoodHashMap(K, V, @TypeOf(ctx), hashFn, eqlFn).init(allocator, ctx);
            errdefer map.deinit();
            for (self.items) |pair| {
                _ = try map.insert(pair.key, pair.value);
            }
            return map;
        }
    };
}

/// Create a builder from a slice of items
/// Time: O(1) | Space: O(1)
pub fn fromSlice(comptime T: type, items: []const T) SliceBuilder(T) {
    return .{ .items = items };
}

// ============================================================================
// TESTS
// ============================================================================

test "fromSlice to ArrayList" {
    const allocator = testing.allocator;
    const items = [_]i32{ 1, 2, 3, 4, 5 };

    var list = try fromSlice(i32, &items).toArrayList(allocator);
    defer list.deinit(allocator);

    try testing.expectEqual(5, list.items.len);
    try testing.expectEqual(1, list.items[0]);
    try testing.expectEqual(5, list.items[4]);
}

test "fromSlice to ArrayList preserves order" {
    const allocator = testing.allocator;
    const items = [_][]const u8{ "apple", "banana", "cherry" };

    var list = try fromSlice([]const u8, &items).toArrayList(allocator);
    defer list.deinit(allocator);

    try testing.expectEqualStrings("apple", list.items[0]);
    try testing.expectEqualStrings("banana", list.items[1]);
    try testing.expectEqualStrings("cherry", list.items[2]);
}

test "fromSlice to RedBlackTree" {
    const allocator = testing.allocator;

    const pairs = [_]Pair(i32, []const u8){
        .{ .key = 3, .value = "three" },
        .{ .key = 1, .value = "one" },
        .{ .key = 2, .value = "two" },
    };

    var tree = try fromSlice(Pair(i32, []const u8), &pairs).toRedBlackTree(
        i32,
        []const u8,
        allocator,
        {},
        cmp.ascending(i32),
    );
    defer tree.deinit();

    try testing.expectEqual(3, tree.count());
    try testing.expect(tree.contains(1));
    try testing.expect(tree.contains(2));
    try testing.expect(tree.contains(3));
}

test "fromSlice to RedBlackTree maintains BST property" {
    const allocator = testing.allocator;

    const pairs = [_]Pair(i32, void){
        .{ .key = 5, .value = {} },
        .{ .key = 3, .value = {} },
        .{ .key = 7, .value = {} },
        .{ .key = 1, .value = {} },
        .{ .key = 9, .value = {} },
    };

    var tree = try fromSlice(Pair(i32, void), &pairs).toRedBlackTree(
        i32,
        void,
        allocator,
        {},
        cmp.ascending(i32),
    );
    defer tree.deinit();

    // Verify in-order traversal is sorted
    var iter = try tree.iterator();
    var prev: ?i32 = null;
    while (try iter.next()) |entry| {
        if (prev) |p| try testing.expect(p <= entry.key);
        prev = entry.key;
    }
    iter.deinit();
}

test "fromSlice to HashMap" {
    const allocator = testing.allocator;

    const StringCtx = struct {
        pub fn hash(_: @This(), key: []const u8) u64 {
            return h.string({}, key);
        }
        pub fn eql(_: @This(), a: []const u8, b: []const u8) bool {
            return std.mem.eql(u8, a, b);
        }
    };

    const pairs = [_]Pair([]const u8, i32){
        .{ .key = "one", .value = 1 },
        .{ .key = "two", .value = 2 },
        .{ .key = "three", .value = 3 },
    };

    var map = try fromSlice(Pair([]const u8, i32), &pairs).toHashMap(
        []const u8,
        i32,
        allocator,
        StringCtx{},
        StringCtx.hash,
        StringCtx.eql,
    );
    defer map.deinit();

    try testing.expectEqual(3, map.count());
    try testing.expect(map.contains("one"));
    try testing.expect(map.contains("two"));
    try testing.expect(map.contains("three"));
}

test "fromSlice to HashMap retrieves correct values" {
    const allocator = testing.allocator;

    const IntCtx = struct {
        pub fn hash(_: @This(), key: i32) u64 {
            var hasher = std.hash.Wyhash.init(0);
            std.hash.autoHash(&hasher, key);
            return hasher.final();
        }
        pub fn eql(_: @This(), a: i32, b: i32) bool {
            return a == b;
        }
    };

    const pairs = [_]Pair(i32, []const u8){
        .{ .key = 1, .value = "one" },
        .{ .key = 2, .value = "two" },
        .{ .key = 3, .value = "three" },
    };

    var map = try fromSlice(Pair(i32, []const u8), &pairs).toHashMap(
        i32,
        []const u8,
        allocator,
        IntCtx{},
        IntCtx.hash,
        IntCtx.eql,
    );
    defer map.deinit();

    if (map.get(1)) |val| {
        try testing.expectEqualStrings("one", val);
    } else {
        try testing.expect(false); // Should find value
    }
}

test "fromSlice to SkipList" {
    const allocator = testing.allocator;
    const items = [_]i32{ 5, 2, 8, 1, 9, 3 };

    var list = try fromSlice(i32, &items).toSkipList(
        allocator,
        {},
        cmp.ascending(i32),
    );
    defer list.deinit();

    try testing.expectEqual(6, list.count());
    try testing.expect(list.contains(1));
    try testing.expect(list.contains(5));
    try testing.expect(list.contains(9));
}

test "fromSlice to SkipList maintains sorted order" {
    const allocator = testing.allocator;
    const items = [_]i32{ 5, 2, 8, 1, 9, 3 };

    var list = try fromSlice(i32, &items).toSkipList(
        allocator,
        {},
        cmp.ascending(i32),
    );
    defer list.deinit();

    // Verify sorted order via iterator
    var iter = list.iterator();
    var prev: ?i32 = null;
    while (iter.next()) |entry| {
        if (prev) |p| try testing.expect(p <= entry.key);
        prev = entry.key;
    }
}

test "fromSlice empty to ArrayList" {
    const allocator = testing.allocator;
    const items: [0]i32 = .{};

    var list = try fromSlice(i32, &items).toArrayList(allocator);
    defer list.deinit(allocator);

    try testing.expectEqual(0, list.items.len);
}

test "fromSlice empty to HashMap" {
    const allocator = testing.allocator;

    const IntCtx = struct {
        pub fn hash(_: @This(), key: i32) u64 {
            var hasher = std.hash.Wyhash.init(0);
            std.hash.autoHash(&hasher, key);
            return hasher.final();
        }
        pub fn eql(_: @This(), a: i32, b: i32) bool {
            return a == b;
        }
    };

    const items: [0]Pair(i32, i32) = .{};

    var map = try fromSlice(Pair(i32, i32), &items).toHashMap(
        i32,
        i32,
        allocator,
        IntCtx{},
        IntCtx.hash,
        IntCtx.eql,
    );
    defer map.deinit();

    try testing.expectEqual(0, map.count());
}

test "fromSlice empty to RedBlackTree" {
    const allocator = testing.allocator;
    const items: [0]Pair(i32, void) = .{};

    var tree = try fromSlice(Pair(i32, void), &items).toRedBlackTree(
        i32,
        void,
        allocator,
        {},
        cmp.ascending(i32),
    );
    defer tree.deinit();

    try testing.expectEqual(0, tree.count());
}

test "fromSlice empty to SkipList" {
    const allocator = testing.allocator;
    const items: [0]i32 = .{};

    var list = try fromSlice(i32, &items).toSkipList(
        allocator,
        {},
        cmp.ascending(i32),
    );
    defer list.deinit();

    try testing.expectEqual(0, list.count());
}

test "fromSlice single element to ArrayList" {
    const allocator = testing.allocator;
    const items = [_]i32{42};

    var list = try fromSlice(i32, &items).toArrayList(allocator);
    defer list.deinit(allocator);

    try testing.expectEqual(1, list.items.len);
    try testing.expectEqual(42, list.items[0]);
}

test "fromSlice single element to HashMap" {
    const allocator = testing.allocator;

    const StringCtx = struct {
        pub fn hash(_: @This(), key: []const u8) u64 {
            return h.string({}, key);
        }
        pub fn eql(_: @This(), a: []const u8, b: []const u8) bool {
            return std.mem.eql(u8, a, b);
        }
    };

    const pairs = [_]Pair([]const u8, i32){
        .{ .key = "answer", .value = 42 },
    };

    var map = try fromSlice(Pair([]const u8, i32), &pairs).toHashMap(
        []const u8,
        i32,
        allocator,
        StringCtx{},
        StringCtx.hash,
        StringCtx.eql,
    );
    defer map.deinit();

    try testing.expectEqual(1, map.count());
    try testing.expect(map.contains("answer"));
}

test "fromSlice single element to RedBlackTree" {
    const allocator = testing.allocator;
    const pairs = [_]Pair(i32, []const u8){
        .{ .key = 1, .value = "only" },
    };

    var tree = try fromSlice(Pair(i32, []const u8), &pairs).toRedBlackTree(
        i32,
        []const u8,
        allocator,
        {},
        cmp.ascending(i32),
    );
    defer tree.deinit();

    try testing.expectEqual(1, tree.count());
}

test "fromSlice single element to SkipList" {
    const allocator = testing.allocator;
    const items = [_]i32{100};

    var list = try fromSlice(i32, &items).toSkipList(
        allocator,
        {},
        cmp.ascending(i32),
    );
    defer list.deinit();

    try testing.expectEqual(1, list.count());
    try testing.expect(list.contains(100));
}

test "fromSlice duplicates in HashMap overwrites" {
    const allocator = testing.allocator;

    const IntCtx = struct {
        pub fn hash(_: @This(), key: i32) u64 {
            var hasher = std.hash.Wyhash.init(0);
            std.hash.autoHash(&hasher, key);
            return hasher.final();
        }
        pub fn eql(_: @This(), a: i32, b: i32) bool {
            return a == b;
        }
    };

    const pairs = [_]Pair(i32, i32){
        .{ .key = 1, .value = 10 },
        .{ .key = 1, .value = 20 }, // Duplicate key
        .{ .key = 2, .value = 30 },
    };

    var map = try fromSlice(Pair(i32, i32), &pairs).toHashMap(
        i32,
        i32,
        allocator,
        IntCtx{},
        IntCtx.hash,
        IntCtx.eql,
    );
    defer map.deinit();

    // HashMap should have 2 entries (key 1 with latest value, key 2)
    try testing.expectEqual(2, map.count());
}

test "fromSlice duplicates in SkipList deduplicates" {
    const allocator = testing.allocator;
    const items = [_]i32{ 5, 3, 5, 3, 5 };

    var list = try fromSlice(i32, &items).toSkipList(
        allocator,
        {},
        cmp.ascending(i32),
    );
    defer list.deinit();

    // SkipList deduplicates by key, stores only unique keys: {3, 5}
    try testing.expectEqual(2, list.count());
    try testing.expect(list.contains(3));
    try testing.expect(list.contains(5));
}

test "fromSlice duplicates in RedBlackTree overwrites" {
    const allocator = testing.allocator;

    const pairs = [_]Pair(i32, []const u8){
        .{ .key = 1, .value = "first" },
        .{ .key = 1, .value = "second" }, // Duplicate key
        .{ .key = 2, .value = "another" },
    };

    var tree = try fromSlice(Pair(i32, []const u8), &pairs).toRedBlackTree(
        i32,
        []const u8,
        allocator,
        {},
        cmp.ascending(i32),
    );
    defer tree.deinit();

    // Tree should have 2 entries (key 1 overwrites, key 2)
    try testing.expectEqual(2, tree.count());
}

test "builder memory cleanup no leaks" {
    const allocator = testing.allocator;
    const items = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

    var list = try fromSlice(i32, &items).toArrayList(allocator);
    defer list.deinit(allocator);
    // Allocator will detect leaks when scope exits
}

test "builder with large dataset no leaks" {
    const allocator = testing.allocator;
    var items_buf: [1000]i32 = undefined;
    for (0..1000) |i| {
        items_buf[i] = @intCast(i);
    }

    var list = try fromSlice(i32, &items_buf).toArrayList(allocator);
    defer list.deinit(allocator);

    try testing.expectEqual(1000, list.items.len);
}

test "fromSlice with custom comparator" {
    const allocator = testing.allocator;

    const DescContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return cmp.descending(i32)({}, a, b);
        }
    };

    const items = [_]i32{ 3, 1, 4, 1, 5 };

    var list = try fromSlice(i32, &items).toSkipList(
        allocator,
        DescContext{},
        DescContext.compare,
    );
    defer list.deinit();

    // Verify descending order via iterator
    var iter = list.iterator();
    var prev: ?i32 = null;
    while (iter.next()) |entry| {
        if (prev) |p| try testing.expect(p >= entry.key);
        prev = entry.key;
    }
}

test "fromSlice HashMap with string keys" {
    const allocator = testing.allocator;

    const StringCtx = struct {
        pub fn hash(_: @This(), key: []const u8) u64 {
            return h.string({}, key);
        }
        pub fn eql(_: @This(), a: []const u8, b: []const u8) bool {
            return std.mem.eql(u8, a, b);
        }
    };

    const pairs = [_]Pair([]const u8, i32){
        .{ .key = "hello", .value = 1 },
        .{ .key = "world", .value = 2 },
        .{ .key = "test", .value = 3 },
    };

    var map = try fromSlice(Pair([]const u8, i32), &pairs).toHashMap(
        []const u8,
        i32,
        allocator,
        StringCtx{},
        StringCtx.hash,
        StringCtx.eql,
    );
    defer map.deinit();

    try testing.expectEqual(3, map.count());
    if (map.get("hello")) |val| {
        try testing.expectEqual(1, val);
    } else {
        try testing.expect(false);
    }
}

test "fromSlice RedBlackTree string keys" {
    const allocator = testing.allocator;

    const StringCtx = struct {
        pub fn compare(_: @This(), a: []const u8, b: []const u8) Order {
            return cmp.stringAscending({}, a, b);
        }
    };

    const pairs = [_]Pair([]const u8, i32){
        .{ .key = "charlie", .value = 3 },
        .{ .key = "alice", .value = 1 },
        .{ .key = "bob", .value = 2 },
    };

    var tree = try fromSlice(Pair([]const u8, i32), &pairs).toRedBlackTree(
        []const u8,
        i32,
        allocator,
        StringCtx{},
        StringCtx.compare,
    );
    defer tree.deinit();

    // Verify in-order iteration is lexicographically sorted
    var iter = try tree.iterator();
    var prev: ?[]const u8 = null;
    while (try iter.next()) |entry| {
        if (prev) |p| try testing.expect(std.mem.lessThan(u8, p, entry.key));
        prev = entry.key;
    }
    iter.deinit();
}

test "fromSlice to ArrayList multiple times" {
    const allocator = testing.allocator;
    const items1 = [_]i32{ 1, 2, 3 };
    const items2 = [_]i32{ 4, 5, 6 };

    var list1 = try fromSlice(i32, &items1).toArrayList(allocator);
    defer list1.deinit(allocator);

    var list2 = try fromSlice(i32, &items2).toArrayList(allocator);
    defer list2.deinit(allocator);

    try testing.expectEqual(3, list1.items.len);
    try testing.expectEqual(3, list2.items.len);
    try testing.expectEqual(1, list1.items[0]);
    try testing.expectEqual(4, list2.items[0]);
}
