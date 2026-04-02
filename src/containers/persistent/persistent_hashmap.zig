//! Persistent Hash Map (HAMT - Hash Array Mapped Trie)
//!
//! A persistent (immutable) hash map using structural sharing via Hash Array Mapped Trie.
//! Used in functional languages (Clojure, Scala) for efficient immutable collections.
//!
//! Time complexity:
//!   - get: O(log₃₂ n) = ~O(1) for practical sizes
//!   - insert/update: O(log₃₂ n) with structural sharing
//!   - remove: O(log₃₂ n) with structural sharing
//!
//! Space complexity:
//!   - O(n) total
//!   - O(log₃₂ n) extra per version (shared nodes)
//!
//! Features:
//!   - Immutable: all operations return new version
//!   - Structural sharing: versions share unmodified nodes
//!   - Memory efficient: only modified paths are copied
//!   - Hash collision handling: chaining for same bitmap index
//!
//! Use cases:
//!   - Functional programming (immutable maps)
//!   - Version control (efficient snapshots)
//!   - Undo/redo systems
//!   - Concurrent data structures (lock-free reads)

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Persistent Hash Map using HAMT (Hash Array Mapped Trie)
///
/// Time: O(log₃₂ n) for get/insert/remove | Space: O(n) total, O(log₃₂ n) per version
pub fn PersistentHashMap(comptime K: type, comptime V: type, comptime Context: type) type {
    return struct {
        const Self = @This();
        const BITS_PER_LEVEL = 5; // 2^5 = 32-way branching
        const FANOUT = 1 << BITS_PER_LEVEL; // 32
        const LEVEL_MASK = FANOUT - 1; // 0x1F

        /// Entry in the hash map
        pub const Entry = struct {
            key: K,
            value: V,
        };

        /// Node types: Branch (internal) or Leaf (entries)
        const NodeType = enum {
            branch,
            leaf,
        };

        /// HAMT Node - either branch or leaf
        const Node = struct {
            type: NodeType,
            data: union {
                branch: Branch,
                leaf: Leaf,
            },

            fn deinit(self: *Node, allocator: Allocator) void {
                switch (self.type) {
                    .branch => {
                        var i: u32 = 0;
                        while (i < FANOUT) : (i += 1) {
                            if (self.data.branch.children[i]) |child| {
                                child.deinit(allocator);
                                allocator.destroy(child);
                            }
                        }
                    },
                    .leaf => {
                        self.data.leaf.entries.deinit(self.data.leaf.allocator);
                    },
                }
            }

            fn clone(self: *const Node, allocator: Allocator) Allocator.Error!*Node {
                const new_node = try allocator.create(Node);
                switch (self.type) {
                    .branch => {
                        new_node.* = Node{
                            .type = .branch,
                            .data = .{ .branch = self.data.branch },
                        };
                    },
                    .leaf => {
                        new_node.* = Node{
                            .type = .leaf,
                            .data = .{
                                .leaf = .{
                                    .entries = try self.data.leaf.entries.clone(allocator),
                                    .allocator = allocator,
                                },
                            },
                        };
                    },
                }
                return new_node;
            }
        };

        /// Branch node - internal node with bitmap and children
        const Branch = struct {
            bitmap: u32, // which children are present (bit i set = child[i] exists)
            children: [FANOUT]?*Node,

            fn init() Branch {
                return .{
                    .bitmap = 0,
                    .children = [_]?*Node{null} ** FANOUT,
                };
            }

            fn hasChild(self: *const Branch, index: u5) bool {
                return (self.bitmap & (@as(u32, 1) << index)) != 0;
            }

            fn setChild(self: *Branch, index: u5, child: *Node) void {
                self.bitmap |= (@as(u32, 1) << index);
                self.children[index] = child;
            }

            fn clearChild(self: *Branch, index: u5) void {
                self.bitmap &= ~(@as(u32, 1) << index);
                self.children[index] = null;
            }
        };

        /// Leaf node - stores actual entries (handles hash collisions)
        const Leaf = struct {
            entries: std.ArrayList(Entry),
            allocator: Allocator,

            fn init(allocator: Allocator) Leaf {
                return .{
                    .entries = std.ArrayList(Entry).initCapacity(allocator, 0) catch unreachable,
                    .allocator = allocator,
                };
            }
        };

        root: ?*Node,
        size: usize,
        ctx: Context,
        allocator: Allocator,

        /// Initialize empty persistent hash map
        pub fn init(allocator: Allocator, ctx: Context) Self {
            return .{
                .root = null,
                .size = 0,
                .ctx = ctx,
                .allocator = allocator,
            };
        }

        /// Clean up all nodes (call on final version only)
        pub fn deinit(self: *Self) void {
            if (self.root) |root| {
                root.deinit(self.allocator);
                self.allocator.destroy(root);
            }
        }

        /// Get value for key (returns null if not found)
        ///
        /// Time: O(log₃₂ n) | Space: O(1)
        pub fn get(self: *const Self, key: K) ?V {
            if (self.root == null) return null;
            const hash = self.ctx.hash(key);
            return self.getRecursive(self.root.?, hash, 0, key);
        }

        fn getRecursive(self: *const Self, node: *Node, hash: u64, shift: u6, key: K) ?V {
            switch (node.type) {
                .branch => {
                    const index = @as(u5, @truncate((hash >> shift) & LEVEL_MASK));
                    if (!node.data.branch.hasChild(index)) return null;
                    return self.getRecursive(node.data.branch.children[index].?, hash, shift + BITS_PER_LEVEL, key);
                },
                .leaf => {
                    for (node.data.leaf.entries.items) |entry| {
                        if (self.ctx.eql(entry.key, key)) {
                            return entry.value;
                        }
                    }
                    return null;
                },
            }
        }

        /// Insert or update key-value pair (returns new version)
        ///
        /// Time: O(log₃₂ n) | Space: O(log₃₂ n) for modified path
        pub fn insert(self: *const Self, key: K, value: V) Allocator.Error!Self {
            const hash = self.ctx.hash(key);
            var new_map = self.*;

            if (self.root == null) {
                // Empty map - create root leaf
                const root_node = try self.allocator.create(Node);
                root_node.* = Node{
                    .type = .leaf,
                    .data = .{ .leaf = Leaf.init(self.allocator) },
                };
                try root_node.data.leaf.entries.append(self.allocator, Entry{ .key = key, .value = value });
                new_map.root = root_node;
                new_map.size = 1;
                return new_map;
            }

            var inserted_new = false;
            new_map.root = try self.insertRecursive(self.root.?, hash, 0, key, value, &inserted_new);
            if (inserted_new) new_map.size += 1;
            return new_map;
        }

        fn insertRecursive(
            self: *const Self,
            node: *Node,
            hash: u64,
            shift: u6,
            key: K,
            value: V,
            inserted_new: *bool,
        ) Allocator.Error!*Node {
            switch (node.type) {
                .branch => {
                    const index = @as(u5, @truncate((hash >> shift) & LEVEL_MASK));
                    const new_node = try node.clone(self.allocator);

                    if (new_node.data.branch.hasChild(index)) {
                        // Recursively insert into child
                        const old_child = new_node.data.branch.children[index].?;
                        const new_child = try self.insertRecursive(old_child, hash, shift + BITS_PER_LEVEL, key, value, inserted_new);
                        new_node.data.branch.setChild(index, new_child);
                    } else {
                        // Create new leaf child
                        const leaf_node = try self.allocator.create(Node);
                        leaf_node.* = Node{
                            .type = .leaf,
                            .data = .{ .leaf = Leaf.init(self.allocator) },
                        };
                        try leaf_node.data.leaf.entries.append(leaf_node.data.leaf.allocator, Entry{ .key = key, .value = value });
                        new_node.data.branch.setChild(index, leaf_node);
                        inserted_new.* = true;
                    }

                    return new_node;
                },
                .leaf => {
                    // Check if key exists - update in place
                    for (node.data.leaf.entries.items, 0..) |entry, i| {
                        if (self.ctx.eql(entry.key, key)) {
                            // Update existing entry
                            const new_node = try node.clone(self.allocator);
                            new_node.data.leaf.entries.items[i].value = value;
                            return new_node;
                        }
                    }

                    // Key doesn't exist - check if we need to split
                    if (node.data.leaf.entries.items.len < 4 or shift >= 60) {
                        // Small enough or max depth - add to leaf
                        const new_node = try node.clone(self.allocator);
                        try new_node.data.leaf.entries.append(self.allocator, Entry{ .key = key, .value = value });
                        inserted_new.* = true;
                        return new_node;
                    }

                    // Split leaf into branch
                    const branch_node = try self.allocator.create(Node);
                    branch_node.* = Node{
                        .type = .branch,
                        .data = .{ .branch = Branch.init() },
                    };

                    // Re-insert existing entries
                    for (node.data.leaf.entries.items) |entry| {
                        const entry_hash = self.ctx.hash(entry.key);
                        const entry_index = @as(u5, @truncate((entry_hash >> shift) & LEVEL_MASK));

                        if (!branch_node.data.branch.hasChild(entry_index)) {
                            const leaf = try self.allocator.create(Node);
                            leaf.* = Node{
                                .type = .leaf,
                                .data = .{ .leaf = Leaf.init(self.allocator) },
                            };
                            branch_node.data.branch.setChild(entry_index, leaf);
                        }

                        try branch_node.data.branch.children[entry_index].?.data.leaf.entries.append(self.allocator, entry);
                    }

                    // Insert new entry
                    const new_index = @as(u5, @truncate((hash >> shift) & LEVEL_MASK));
                    if (!branch_node.data.branch.hasChild(new_index)) {
                        const leaf = try self.allocator.create(Node);
                        leaf.* = Node{
                            .type = .leaf,
                            .data = .{ .leaf = Leaf.init(self.allocator) },
                        };
                        branch_node.data.branch.setChild(new_index, leaf);
                    }
                    try branch_node.data.branch.children[new_index].?.data.leaf.entries.append(self.allocator, Entry{ .key = key, .value = value });
                    inserted_new.* = true;

                    return branch_node;
                },
            }
        }

        /// Remove key (returns new version, or same if key not found)
        ///
        /// Time: O(log₃₂ n) | Space: O(log₃₂ n) for modified path
        pub fn remove(self: *const Self, key: K) Allocator.Error!Self {
            if (self.root == null) return self.*;

            const hash = self.ctx.hash(key);
            var removed = false;
            const new_root = (try self.removeRecursive(self.root.?, hash, 0, key, &removed)) orelse {
                // Root became empty
                var new_map = self.*;
                new_map.root = null;
                new_map.size = 0;
                return new_map;
            };

            var new_map = self.*;
            new_map.root = new_root;
            if (removed) new_map.size -= 1;
            return new_map;
        }

        fn removeRecursive(
            self: *const Self,
            node: *Node,
            hash: u64,
            shift: u6,
            key: K,
            removed: *bool,
        ) Allocator.Error!?*Node {
            switch (node.type) {
                .branch => {
                    const index = @as(u5, @truncate((hash >> shift) & LEVEL_MASK));
                    if (!node.data.branch.hasChild(index)) return node;

                    const new_node = try node.clone(self.allocator);
                    const old_child = new_node.data.branch.children[index].?;
                    const new_child = try self.removeRecursive(old_child, hash, shift + BITS_PER_LEVEL, key, removed);

                    if (new_child == null) {
                        new_node.data.branch.clearChild(index);
                    } else {
                        new_node.data.branch.setChild(index, new_child.?);
                    }

                    // Check if branch has become empty
                    if (new_node.data.branch.bitmap == 0) return null;

                    return new_node;
                },
                .leaf => {
                    for (node.data.leaf.entries.items, 0..) |entry, i| {
                        if (self.ctx.eql(entry.key, key)) {
                            const new_node = try node.clone(self.allocator);
                            _ = new_node.data.leaf.entries.orderedRemove(i);
                            removed.* = true;

                            // If leaf is empty, return null
                            if (new_node.data.leaf.entries.items.len == 0) return null;

                            return new_node;
                        }
                    }
                    return node; // Key not found
                },
            }
        }

        /// Check if key exists
        ///
        /// Time: O(log₃₂ n) | Space: O(1)
        pub fn contains(self: *const Self, key: K) bool {
            return self.get(key) != null;
        }

        /// Get number of entries
        ///
        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            return self.size;
        }

        /// Check if map is empty
        ///
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.size == 0;
        }

        /// Iterator for traversing entries
        pub const Iterator = struct {
            stack: std.ArrayList(*Node),
            leaf_index: usize,
            branch_indices: std.ArrayList(u5),
            allocator: Allocator,

            fn deinitIterator(self: *Iterator) void {
                self.stack.deinit(self.allocator);
                self.branch_indices.deinit(self.allocator);
            }

            pub fn next(self: *Iterator) ?Entry {
                while (self.stack.items.len > 0) {
                    const node = self.stack.items[self.stack.items.len - 1];

                    switch (node.type) {
                        .leaf => {
                            if (self.leaf_index < node.data.leaf.entries.items.len) {
                                const entry = node.data.leaf.entries.items[self.leaf_index];
                                self.leaf_index += 1;
                                return entry;
                            } else {
                                _ = self.stack.pop();
                                _ = self.branch_indices.pop();
                                self.leaf_index = 0;
                            }
                        },
                        .branch => {
                            var idx = self.branch_indices.items[self.branch_indices.items.len - 1];
                            while (idx < FANOUT) : (idx += 1) {
                                if (node.data.branch.hasChild(idx)) {
                                    self.branch_indices.items[self.branch_indices.items.len - 1] = idx + 1;
                                    self.stack.append(self.allocator, node.data.branch.children[idx].?) catch return null;
                                    self.branch_indices.append(self.allocator, 0) catch return null;
                                    break;
                                }
                            } else {
                                _ = self.stack.pop();
                                _ = self.branch_indices.pop();
                            }
                        },
                    }
                }
                return null;
            }
        };

        /// Get iterator for all entries
        ///
        /// Time: O(1) init | Space: O(log n) for stack
        pub fn iterator(self: *const Self) Allocator.Error!Iterator {
            var stack = try std.ArrayList(*Node).initCapacity(self.allocator, 0);
            var branch_indices = try std.ArrayList(u5).initCapacity(self.allocator, 0);

            if (self.root) |root| {
                try stack.append(self.allocator, root);
                try branch_indices.append(self.allocator, 0);
            }

            return Iterator{
                .stack = stack,
                .leaf_index = 0,
                .branch_indices = branch_indices,
                .allocator = self.allocator,
            };
        }
    };
}

// Tests
const testing = std.testing;

fn testHash(key: u32) u64 {
    // Simple hash for testing
    return @as(u64, key) *% 0x9e3779b97f4a7c15;
}

fn testEql(a: u32, b: u32) bool {
    return a == b;
}

const TestContext = struct {
    pub fn hash(_: @This(), key: u32) u64 {
        return testHash(key);
    }

    pub fn eql(_: @This(), a: u32, b: u32) bool {
        return testEql(a, b);
    }
};

test "PersistentHashMap - initialization" {
    const Map = PersistentHashMap(u32, u32, TestContext);
    var map = Map.init(testing.allocator, TestContext{});
    defer map.deinit();

    try testing.expectEqual(@as(usize, 0), map.count());
    try testing.expect(map.isEmpty());
}

test "PersistentHashMap - single insert and get" {
    const Map = PersistentHashMap(u32, u32, TestContext);
    const map1 = Map.init(testing.allocator, TestContext{});

    var map2 = try map1.insert(42, 100);
    defer map2.deinit();

    try testing.expectEqual(@as(usize, 1), map2.count());
    try testing.expectEqual(@as(u32, 100), map2.get(42).?);
    try testing.expect(map2.contains(42));
}

test "PersistentHashMap - multiple inserts" {
    const Map = PersistentHashMap(u32, u32, TestContext);
    var map = Map.init(testing.allocator, TestContext{});

    map = try map.insert(1, 10);
    map = try map.insert(2, 20);
    map = try map.insert(3, 30);
    defer map.deinit();

    try testing.expectEqual(@as(usize, 3), map.count());
    try testing.expectEqual(@as(u32, 10), map.get(1).?);
    try testing.expectEqual(@as(u32, 20), map.get(2).?);
    try testing.expectEqual(@as(u32, 30), map.get(3).?);
}

test "PersistentHashMap - update existing key" {
    const Map = PersistentHashMap(u32, u32, TestContext);
    var map = Map.init(testing.allocator, TestContext{});

    map = try map.insert(42, 100);
    map = try map.insert(42, 200); // Update
    defer map.deinit();

    try testing.expectEqual(@as(usize, 1), map.count());
    try testing.expectEqual(@as(u32, 200), map.get(42).?);
}

test "PersistentHashMap - persistence (structural sharing)" {
    const Map = PersistentHashMap(u32, u32, TestContext);
    var map1 = Map.init(testing.allocator, TestContext{});

    map1 = try map1.insert(1, 10);
    map1 = try map1.insert(2, 20);

    var map2 = try map1.insert(3, 30);
    defer map2.deinit();

    // map1 should still have only 2 elements
    try testing.expectEqual(@as(usize, 2), map1.count());
    try testing.expectEqual(@as(u32, 10), map1.get(1).?);
    try testing.expectEqual(@as(u32, 20), map1.get(2).?);
    try testing.expect(map1.get(3) == null);

    // map2 should have 3 elements
    try testing.expectEqual(@as(usize, 3), map2.count());
    try testing.expectEqual(@as(u32, 10), map2.get(1).?);
    try testing.expectEqual(@as(u32, 20), map2.get(2).?);
    try testing.expectEqual(@as(u32, 30), map2.get(3).?);
}

test "PersistentHashMap - remove key" {
    const Map = PersistentHashMap(u32, u32, TestContext);
    var map = Map.init(testing.allocator, TestContext{});

    map = try map.insert(1, 10);
    map = try map.insert(2, 20);
    map = try map.insert(3, 30);

    map = try map.remove(2);
    defer map.deinit();

    try testing.expectEqual(@as(usize, 2), map.count());
    try testing.expectEqual(@as(u32, 10), map.get(1).?);
    try testing.expect(map.get(2) == null);
    try testing.expectEqual(@as(u32, 30), map.get(3).?);
}

test "PersistentHashMap - remove preserves old version" {
    const Map = PersistentHashMap(u32, u32, TestContext);
    var map1 = Map.init(testing.allocator, TestContext{});

    map1 = try map1.insert(1, 10);
    map1 = try map1.insert(2, 20);

    var map2 = try map1.remove(1);
    defer map2.deinit();

    // map1 still has both
    try testing.expectEqual(@as(usize, 2), map1.count());
    try testing.expectEqual(@as(u32, 10), map1.get(1).?);

    // map2 has only 2
    try testing.expectEqual(@as(usize, 1), map2.count());
    try testing.expect(map2.get(1) == null);
    try testing.expectEqual(@as(u32, 20), map2.get(2).?);
}

test "PersistentHashMap - get nonexistent key" {
    const Map = PersistentHashMap(u32, u32, TestContext);
    var map = Map.init(testing.allocator, TestContext{});

    map = try map.insert(1, 10);
    defer map.deinit();

    try testing.expect(map.get(99) == null);
    try testing.expect(!map.contains(99));
}

test "PersistentHashMap - hash collisions (same index)" {
    const Map = PersistentHashMap(u32, u32, TestContext);
    var map = Map.init(testing.allocator, TestContext{});

    // Insert multiple values that might hash to similar locations
    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        map = try map.insert(i, i * 10);
    }
    defer map.deinit();

    try testing.expectEqual(@as(usize, 10), map.count());

    i = 0;
    while (i < 10) : (i += 1) {
        try testing.expectEqual(i * 10, map.get(i).?);
    }
}

test "PersistentHashMap - iterator" {
    const Map = PersistentHashMap(u32, u32, TestContext);
    var map = Map.init(testing.allocator, TestContext{});

    map = try map.insert(1, 10);
    map = try map.insert(2, 20);
    map = try map.insert(3, 30);
    defer map.deinit();

    var iter = try map.iterator();
    defer iter.deinitIterator();

    var sum: u32 = 0;
    var count: usize = 0;
    while (iter.next()) |entry| {
        sum += entry.value;
        count += 1;
    }

    try testing.expectEqual(@as(usize, 3), count);
    try testing.expectEqual(@as(u32, 60), sum); // 10 + 20 + 30
}

test "PersistentHashMap - large dataset" {
    const Map = PersistentHashMap(u32, u32, TestContext);
    var map = Map.init(testing.allocator, TestContext{});

    // Insert 100 values
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        map = try map.insert(i, i * 2);
    }
    defer map.deinit();

    try testing.expectEqual(@as(usize, 100), map.count());

    // Verify all values
    i = 0;
    while (i < 100) : (i += 1) {
        try testing.expectEqual(i * 2, map.get(i).?);
    }
}

test "PersistentHashMap - remove all elements" {
    const Map = PersistentHashMap(u32, u32, TestContext);
    var map = Map.init(testing.allocator, TestContext{});

    map = try map.insert(1, 10);
    map = try map.insert(2, 20);
    map = try map.remove(1);
    map = try map.remove(2);
    defer map.deinit();

    try testing.expectEqual(@as(usize, 0), map.count());
    try testing.expect(map.isEmpty());
}

test "PersistentHashMap - complex persistence scenario" {
    const Map = PersistentHashMap(u32, u32, TestContext);
    const v0 = Map.init(testing.allocator, TestContext{});

    const v1 = try v0.insert(1, 10);
    const v2 = try v1.insert(2, 20);
    const v3 = try v2.insert(3, 30);

    var v4 = try v2.insert(4, 40); // Branch from v2
    defer v4.deinit();

    // v2 should have {1:10, 2:20}
    try testing.expectEqual(@as(usize, 2), v2.count());
    try testing.expect(v2.get(3) == null);

    // v3 should have {1:10, 2:20, 3:30}
    try testing.expectEqual(@as(usize, 3), v3.count());
    try testing.expect(v3.get(4) == null);

    // v4 should have {1:10, 2:20, 4:40}
    try testing.expectEqual(@as(usize, 3), v4.count());
    try testing.expect(v4.get(3) == null);
    try testing.expectEqual(@as(u32, 40), v4.get(4).?);
}

test "PersistentHashMap - f32 type support" {
    const Map = PersistentHashMap(u32, f32, TestContext);
    var map = Map.init(testing.allocator, TestContext{});

    map = try map.insert(1, 1.5);
    map = try map.insert(2, 2.5);
    defer map.deinit();

    try testing.expectEqual(@as(usize, 2), map.count());
    try testing.expectEqual(@as(f32, 1.5), map.get(1).?);
    try testing.expectEqual(@as(f32, 2.5), map.get(2).?);
}

test "PersistentHashMap - memory safety with testing.allocator" {
    const Map = PersistentHashMap(u32, u32, TestContext);
    var map = Map.init(testing.allocator, TestContext{});

    var i: u32 = 0;
    while (i < 50) : (i += 1) {
        map = try map.insert(i, i);
    }

    i = 0;
    while (i < 25) : (i += 1) {
        map = try map.remove(i);
    }

    defer map.deinit();

    try testing.expectEqual(@as(usize, 25), map.count());
}
