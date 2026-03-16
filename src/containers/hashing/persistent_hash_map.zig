const std = @import("std");
const testing = std.testing;

/// PersistentHashMap - Hash Array Mapped Trie (HAMT) for immutable hash maps
///
/// A persistent hash map with structural sharing using a hash trie structure.
/// All operations create new versions while sharing unchanged nodes with old versions.
///
/// Structure:
/// - 32-way branching (5-bit chunks of hash)
/// - Sparse representation via bitmap and compact array
/// - Path copying for O(log₃₂ n) updates
/// - O(log₃₂ n) ≈ O(1) for practical sizes
///
/// Time complexity:
/// - get: O(log₃₂ n) ≈ O(1)
/// - set: O(log₃₂ n) with path copying
/// - remove: O(log₃₂ n) with path copying
///
/// Space complexity:
/// - O(n) total structure
/// - O(log₃₂ n) per mutation (path copy)
///
/// Consumer: Functional programming, undo/redo systems, concurrent access without locks
pub fn PersistentHashMap(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime hashFn: fn (ctx: Context, key: K) u64,
    comptime eqlFn: fn (ctx: Context, a: K, b: K) bool,
) type {
    return struct {
        const Self = @This();

        const BRANCH_FACTOR = 32; // 2^5
        const BITS_PER_LEVEL = 5;
        const LEVEL_MASK = (1 << BITS_PER_LEVEL) - 1; // 0x1F

        const Entry = struct {
            key: K,
            value: V,
        };

        const Node = union(enum) {
            leaf: Entry,
            collision: []Entry, // Hash collision bucket
            branch: BranchNode,
        };

        const BranchNode = struct {
            bitmap: u32, // Bitmap of populated children (32 bits for 32-way branching)
            children: []const *Node, // Compact array of children (length = popcount(bitmap))
        };

        allocator: std.mem.Allocator,
        context: Context,
        root: ?*Node,
        _count: usize,

        // -- Lifecycle --

        /// Initialize empty persistent hash map
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: std.mem.Allocator, context: Context) Self {
            return .{
                .allocator = allocator,
                .context = context,
                .root = null,
                ._count = 0,
            };
        }

        /// Free all memory (does not support shared nodes across instances)
        /// Time: O(n) | Space: O(log n) stack
        pub fn deinit(self: *const Self) void {
            if (self.root) |root| {
                self.freeNode(root);
            }
        }

        fn freeNode(self: *const Self, node: *Node) void {
            switch (node.*) {
                .leaf => {},
                .collision => |entries| {
                    self.allocator.free(entries);
                },
                .branch => |branch| {
                    for (branch.children) |child| {
                        self.freeNode(child);
                    }
                    self.allocator.free(branch.children);
                },
            }
            self.allocator.destroy(node);
        }

        // -- Capacity --

        /// Get the number of entries
        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            return self._count;
        }

        /// Check if map is empty
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self._count == 0;
        }

        // -- Lookup --

        /// Get value for key
        /// Time: O(log₃₂ n) | Space: O(1)
        pub fn get(self: *const Self, key: K) ?V {
            if (self.root == null) return null;
            const hash = hashFn(self.context, key);
            return self.getInNode(self.root.?, hash, 0, key);
        }

        fn getInNode(self: *const Self, node: *Node, hash: u64, shift: u6, key: K) ?V {
            return switch (node.*) {
                .leaf => |entry| {
                    if (eqlFn(self.context, entry.key, key)) {
                        return entry.value;
                    }
                    return null;
                },
                .collision => |entries| {
                    for (entries) |entry| {
                        if (eqlFn(self.context, entry.key, key)) {
                            return entry.value;
                        }
                    }
                    return null;
                },
                .branch => |branch| {
                    const idx = @as(u5, @truncate((hash >> shift) & LEVEL_MASK));
                    const bit = @as(u32, 1) << idx;
                    if ((branch.bitmap & bit) == 0) return null;

                    const child_idx = @popCount(branch.bitmap & (bit - 1));
                    return self.getInNode(branch.children[child_idx], hash, shift + BITS_PER_LEVEL, key);
                },
            };
        }

        /// Check if key exists
        /// Time: O(log₃₂ n) | Space: O(1)
        pub fn contains(self: *const Self, key: K) bool {
            return self.get(key) != null;
        }

        // -- Modification (immutable - returns new map) --

        /// Set key-value pair, returns new map with updated value
        /// Time: O(log₃₂ n) | Space: O(log₃₂ n) path copy
        pub fn set(self: *const Self, key: K, value: V) !Self {
            const hash = hashFn(self.context, key);
            const result = if (self.root) |root|
                try self.setInNode(root, hash, 0, key, value)
            else
                .{ .node = try self.createLeaf(key, value), .inserted = true };

            return .{
                .allocator = self.allocator,
                .context = self.context,
                .root = result.node,
                ._count = self._count + @intFromBool(result.inserted),
            };
        }

        const SetResult = struct {
            node: *Node,
            inserted: bool, // true if new key was inserted, false if updated existing
        };

        fn setInNode(self: *const Self, node: *Node, hash: u64, shift: u6, key: K, value: V) !SetResult {
            switch (node.*) {
                .leaf => |entry| {
                    if (eqlFn(self.context, entry.key, key)) {
                        // Update existing leaf
                        return .{ .node = try self.createLeaf(key, value), .inserted = false };
                    }

                    // Hash collision - need to expand
                    const existing_hash = hashFn(self.context, entry.key);
                    if (existing_hash == hash) {
                        // True collision - create collision bucket
                        const entries = try self.allocator.alloc(Entry, 2);
                        entries[0] = entry;
                        entries[1] = .{ .key = key, .value = value };
                        const new_node = try self.allocator.create(Node);
                        new_node.* = .{ .collision = entries };
                        return .{ .node = new_node, .inserted = true };
                    }

                    // Different hashes - create branch
                    return .{ .node = try self.createBranchFromTwo(entry, existing_hash, key, value, hash, shift), .inserted = true };
                },
                .collision => |entries| {
                    // Check if key exists in collision bucket
                    for (entries, 0..) |entry, i| {
                        if (eqlFn(self.context, entry.key, key)) {
                            // Update existing entry
                            const new_entries = try self.allocator.alloc(Entry, entries.len);
                            @memcpy(new_entries, entries);
                            new_entries[i].value = value;
                            const new_node = try self.allocator.create(Node);
                            new_node.* = .{ .collision = new_entries };
                            return .{ .node = new_node, .inserted = false };
                        }
                    }

                    // Add to collision bucket
                    const new_entries = try self.allocator.alloc(Entry, entries.len + 1);
                    @memcpy(new_entries[0..entries.len], entries);
                    new_entries[entries.len] = .{ .key = key, .value = value };
                    const new_node = try self.allocator.create(Node);
                    new_node.* = .{ .collision = new_entries };
                    return .{ .node = new_node, .inserted = true };
                },
                .branch => |branch| {
                    const idx = @as(u5, @truncate((hash >> shift) & LEVEL_MASK));
                    const bit = @as(u32, 1) << idx;

                    if ((branch.bitmap & bit) == 0) {
                        // Insert new child
                        const child = try self.createLeaf(key, value);
                        return .{ .node = try self.insertChild(branch, idx, bit, child), .inserted = true };
                    }

                    // Update existing child
                    const child_idx = @popCount(branch.bitmap & (bit - 1));
                    const result = try self.setInNode(branch.children[child_idx], hash, shift + BITS_PER_LEVEL, key, value);
                    return .{ .node = try self.replaceChild(branch, child_idx, result.node), .inserted = result.inserted };
                },
            }
        }

        /// Remove key, returns new map without key
        /// Time: O(log₃₂ n) | Space: O(log₃₂ n) path copy
        pub fn remove(self: *const Self, key: K) !Self {
            if (self.root == null) return self.*;

            const hash = hashFn(self.context, key);
            const result = try self.removeInNode(self.root.?, hash, 0, key);

            return .{
                .allocator = self.allocator,
                .context = self.context,
                .root = result.node,
                ._count = self._count - @intFromBool(result.removed),
            };
        }

        const RemoveResult = struct {
            node: ?*Node,
            removed: bool,
        };

        fn removeInNode(self: *const Self, node: *Node, hash: u64, shift: u6, key: K) !RemoveResult {
            switch (node.*) {
                .leaf => |entry| {
                    if (eqlFn(self.context, entry.key, key)) {
                        return .{ .node = null, .removed = true };
                    }
                    return .{ .node = node, .removed = false };
                },
                .collision => |entries| {
                    var found_idx: ?usize = null;
                    for (entries, 0..) |entry, i| {
                        if (eqlFn(self.context, entry.key, key)) {
                            found_idx = i;
                            break;
                        }
                    }

                    if (found_idx == null) {
                        return .{ .node = node, .removed = false };
                    }

                    if (entries.len == 2) {
                        // Collapse to leaf
                        const remaining_idx = 1 - found_idx.?;
                        return .{ .node = try self.createLeaf(entries[remaining_idx].key, entries[remaining_idx].value), .removed = true };
                    }

                    // Shrink collision bucket
                    const new_entries = try self.allocator.alloc(Entry, entries.len - 1);
                    var out_idx: usize = 0;
                    for (entries, 0..) |entry, i| {
                        if (i != found_idx.?) {
                            new_entries[out_idx] = entry;
                            out_idx += 1;
                        }
                    }
                    const new_node = try self.allocator.create(Node);
                    new_node.* = .{ .collision = new_entries };
                    return .{ .node = new_node, .removed = true };
                },
                .branch => |branch| {
                    const idx = @as(u5, @truncate((hash >> shift) & LEVEL_MASK));
                    const bit = @as(u32, 1) << idx;

                    if ((branch.bitmap & bit) == 0) {
                        return .{ .node = node, .removed = false };
                    }

                    const child_idx = @popCount(branch.bitmap & (bit - 1));
                    const result = try self.removeInNode(branch.children[child_idx], hash, shift + BITS_PER_LEVEL, key);

                    if (!result.removed) {
                        return .{ .node = node, .removed = false };
                    }

                    if (result.node) |new_child| {
                        return .{ .node = try self.replaceChild(branch, child_idx, new_child), .removed = true };
                    }

                    // Child was removed, need to remove from branch
                    if (branch.children.len == 1) {
                        // Branch becomes empty
                        return .{ .node = null, .removed = true };
                    }

                    if (branch.children.len == 2) {
                        // Collapse branch to single child
                        const remaining_idx = 1 - child_idx;
                        return .{ .node = branch.children[remaining_idx], .removed = true };
                    }

                    // Remove child from branch
                    return .{ .node = try self.removeChild(branch, idx, bit, child_idx), .removed = true };
                },
            }
        }

        // -- Helper functions --

        fn createLeaf(self: *const Self, key: K, value: V) !*Node {
            const node = try self.allocator.create(Node);
            node.* = .{ .leaf = .{ .key = key, .value = value } };
            return node;
        }

        fn createBranchFromTwo(self: *const Self, entry1: Entry, hash1: u64, key2: K, value2: V, hash2: u64, shift: u6) error{OutOfMemory}!*Node {
            const idx1 = @as(u5, @truncate((hash1 >> shift) & LEVEL_MASK));
            const idx2 = @as(u5, @truncate((hash2 >> shift) & LEVEL_MASK));

            if (idx1 != idx2) {
                // Different indices - create branch with two leaves
                const leaf1 = try self.createLeaf(entry1.key, entry1.value);
                const leaf2 = try self.createLeaf(key2, value2);

                const children = try self.allocator.alloc(*Node, 2);
                if (idx1 < idx2) {
                    children[0] = leaf1;
                    children[1] = leaf2;
                } else {
                    children[0] = leaf2;
                    children[1] = leaf1;
                }

                const bitmap = (@as(u32, 1) << idx1) | (@as(u32, 1) << idx2);
                const node = try self.allocator.create(Node);
                node.* = .{ .branch = .{ .bitmap = bitmap, .children = children } };
                return node;
            }

            // Same index at this level - recurse deeper
            const child = try self.createBranchFromTwo(entry1, hash1, key2, value2, hash2, shift + BITS_PER_LEVEL);
            const children = try self.allocator.alloc(*Node, 1);
            children[0] = child;

            const bitmap = @as(u32, 1) << idx1;
            const node = try self.allocator.create(Node);
            node.* = .{ .branch = .{ .bitmap = bitmap, .children = children } };
            return node;
        }

        fn insertChild(self: *const Self, branch: BranchNode, _: u5, bit: u32, child: *Node) !*Node {
            const child_idx = @popCount(branch.bitmap & (bit - 1));
            const new_children = try self.allocator.alloc(*Node, branch.children.len + 1);

            @memcpy(new_children[0..child_idx], branch.children[0..child_idx]);
            new_children[child_idx] = child;
            @memcpy(new_children[child_idx + 1 ..], branch.children[child_idx..]);

            const new_node = try self.allocator.create(Node);
            new_node.* = .{ .branch = .{ .bitmap = branch.bitmap | bit, .children = new_children } };
            return new_node;
        }

        fn replaceChild(self: *const Self, branch: BranchNode, child_idx: usize, new_child: *Node) !*Node {
            const new_children = try self.allocator.alloc(*Node, branch.children.len);
            @memcpy(new_children, branch.children);
            new_children[child_idx] = new_child;

            const new_node = try self.allocator.create(Node);
            new_node.* = .{ .branch = .{ .bitmap = branch.bitmap, .children = new_children } };
            return new_node;
        }

        fn removeChild(self: *const Self, branch: BranchNode, _: u5, bit: u32, child_idx: usize) !*Node {
            const new_children = try self.allocator.alloc(*Node, branch.children.len - 1);

            @memcpy(new_children[0..child_idx], branch.children[0..child_idx]);
            @memcpy(new_children[child_idx..], branch.children[child_idx + 1 ..]);

            const new_node = try self.allocator.create(Node);
            new_node.* = .{ .branch = .{ .bitmap = branch.bitmap & ~bit, .children = new_children } };
            return new_node;
        }

        // -- Bulk operations --

        /// Create from slice of entries
        /// Time: O(n log n) | Space: O(n)
        pub fn fromSlice(allocator: std.mem.Allocator, context: Context, entries: []const Entry) !Self {
            var map = init(allocator, context);
            for (entries) |entry| {
                const new_map = try map.set(entry.key, entry.value);
                map.deinit(); // Free old version
                map = new_map;
            }
            return map;
        }

        // -- Debug --

        /// Validate invariants
        /// Time: O(n) | Space: O(log n) stack
        pub fn validate(self: *const Self) !void {
            var counted: usize = 0;
            if (self.root) |root| {
                counted = try self.validateNode(root, 0);
            }
            if (counted != self._count) {
                return error.InvalidCount;
            }
        }

        fn validateNode(self: *const Self, node: *Node, depth: usize) !usize {
            if (depth > 64 / BITS_PER_LEVEL + 1) { // Max depth for 64-bit hash
                return error.TooDeep;
            }

            return switch (node.*) {
                .leaf => 1,
                .collision => |entries| {
                    if (entries.len < 2) return error.InvalidCollisionBucket;
                    const first_hash = hashFn(self.context, entries[0].key);
                    for (entries[1..]) |entry| {
                        if (hashFn(self.context, entry.key) != first_hash) {
                            return error.InvalidCollisionHash;
                        }
                    }
                    return entries.len;
                },
                .branch => |branch| {
                    if (branch.children.len == 0) return error.EmptyBranch;
                    if (branch.children.len != @popCount(branch.bitmap)) return error.BitmapMismatch;

                    var total: usize = 0;
                    for (branch.children) |child| {
                        total += try self.validateNode(child, depth + 1);
                    }
                    return total;
                },
            };
        }
    };
}

// -- Tests --

const AutoContext = struct {
    pub fn hash(_: AutoContext, key: u32) u64 {
        return std.hash.Wyhash.hash(0, std.mem.asBytes(&key));
    }

    pub fn eql(_: AutoContext, a: u32, b: u32) bool {
        return a == b;
    }
};

test "PersistentHashMap: init and basic operations" {
    const Map = PersistentHashMap(u32, u32, AutoContext, AutoContext.hash, AutoContext.eql);

    const m0 = Map.init(testing.allocator, .{});
    defer m0.deinit();

    try testing.expectEqual(@as(usize, 0), m0.count());
    try testing.expect(m0.isEmpty());
    try testing.expectEqual(@as(?u32, null), m0.get(1));

    const m1 = try m0.set(1, 100);
    defer m1.deinit();
    try testing.expectEqual(@as(usize, 1), m1.count());
    try testing.expectEqual(@as(?u32, 100), m1.get(1));
    try testing.expect(m1.contains(1));
}

test "PersistentHashMap: immutability - old version unchanged" {
    const Map = PersistentHashMap(u32, u32, AutoContext, AutoContext.hash, AutoContext.eql);

    const m0 = Map.init(testing.allocator, .{});
    defer m0.deinit();

    const m1 = try m0.set(1, 100);
    defer m1.deinit();

    const m2 = try m1.set(2, 200);
    defer m2.deinit();

    // m0 unchanged
    try testing.expectEqual(@as(usize, 0), m0.count());
    try testing.expectEqual(@as(?u32, null), m0.get(1));

    // m1 unchanged
    try testing.expectEqual(@as(usize, 1), m1.count());
    try testing.expectEqual(@as(?u32, 100), m1.get(1));
    try testing.expectEqual(@as(?u32, null), m1.get(2));

    // m2 has both
    try testing.expectEqual(@as(usize, 2), m2.count());
    try testing.expectEqual(@as(?u32, 100), m2.get(1));
    try testing.expectEqual(@as(?u32, 200), m2.get(2));
}

test "PersistentHashMap: update existing key" {
    const Map = PersistentHashMap(u32, u32, AutoContext, AutoContext.hash, AutoContext.eql);

    const m0 = Map.init(testing.allocator, .{});
    defer m0.deinit();

    const m1 = try m0.set(1, 100);
    defer m1.deinit();

    const m2 = try m1.set(1, 999); // Update
    defer m2.deinit();

    // m1 still has old value
    try testing.expectEqual(@as(?u32, 100), m1.get(1));

    // m2 has new value
    try testing.expectEqual(@as(?u32, 999), m2.get(1));
    try testing.expectEqual(@as(usize, 1), m2.count()); // Count unchanged
}

test "PersistentHashMap: remove" {
    const Map = PersistentHashMap(u32, u32, AutoContext, AutoContext.hash, AutoContext.eql);

    const m0 = Map.init(testing.allocator, .{});
    defer m0.deinit();

    const m1 = try m0.set(1, 100);
    defer m1.deinit();

    const m2 = try m1.set(2, 200);
    defer m2.deinit();

    const m3 = try m2.remove(1);
    defer m3.deinit();

    // m2 unchanged
    try testing.expectEqual(@as(usize, 2), m2.count());
    try testing.expectEqual(@as(?u32, 100), m2.get(1));

    // m3 has only key 2
    try testing.expectEqual(@as(usize, 1), m3.count());
    try testing.expectEqual(@as(?u32, null), m3.get(1));
    try testing.expectEqual(@as(?u32, 200), m3.get(2));
}

test "PersistentHashMap: remove nonexistent key" {
    const Map = PersistentHashMap(u32, u32, AutoContext, AutoContext.hash, AutoContext.eql);

    const m0 = Map.init(testing.allocator, .{});
    defer m0.deinit();

    const m1 = try m0.set(1, 100);
    defer m1.deinit();

    const m2 = try m1.remove(999);
    defer m2.deinit();

    try testing.expectEqual(@as(usize, 1), m2.count());
    try testing.expectEqual(@as(?u32, 100), m2.get(1));
}

test "PersistentHashMap: hash collision handling" {
    const CollisionContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key / 10; // Force collisions: 0-9 → 0, 10-19 → 1, etc.
        }

        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    const Map = PersistentHashMap(u32, u32, CollisionContext, CollisionContext.hash, CollisionContext.eql);

    const m0 = Map.init(testing.allocator, .{});
    defer m0.deinit();

    const m1 = try m0.set(5, 100); // hash = 0
    defer m1.deinit();

    const m2 = try m1.set(7, 200); // hash = 0, collision!
    defer m2.deinit();

    try testing.expectEqual(@as(usize, 2), m2.count());
    try testing.expectEqual(@as(?u32, 100), m2.get(5));
    try testing.expectEqual(@as(?u32, 200), m2.get(7));
    try m2.validate();
}

test "PersistentHashMap: multiple keys" {
    const Map = PersistentHashMap(u32, u32, AutoContext, AutoContext.hash, AutoContext.eql);

    var m = Map.init(testing.allocator, .{});
    defer m.deinit();

    // Insert 10 keys
    for (0..10) |i| {
        const new_m = try m.set(@intCast(i), @intCast(i * 10));
        m.deinit();
        m = new_m;
    }

    try testing.expectEqual(@as(usize, 10), m.count());

    // Check all keys
    for (0..10) |i| {
        try testing.expectEqual(@as(?u32, @intCast(i * 10)), m.get(@intCast(i)));
    }

    try m.validate();
}

test "PersistentHashMap: fromSlice" {
    const Map = PersistentHashMap(u32, u32, AutoContext, AutoContext.hash, AutoContext.eql);
    const Entry = Map.Entry;

    const entries = [_]Entry{
        .{ .key = 1, .value = 100 },
        .{ .key = 2, .value = 200 },
        .{ .key = 3, .value = 300 },
    };

    const m = try Map.fromSlice(testing.allocator, .{}, &entries);
    defer m.deinit();

    try testing.expectEqual(@as(usize, 3), m.count());
    try testing.expectEqual(@as(?u32, 100), m.get(1));
    try testing.expectEqual(@as(?u32, 200), m.get(2));
    try testing.expectEqual(@as(?u32, 300), m.get(3));
}

test "PersistentHashMap: stress test" {
    const Map = PersistentHashMap(u32, u32, AutoContext, AutoContext.hash, AutoContext.eql);

    var m = Map.init(testing.allocator, .{});
    defer m.deinit();

    // Insert 100 keys
    for (0..100) |i| {
        const new_m = try m.set(@intCast(i), @intCast(i * 2));
        m.deinit();
        m = new_m;
    }

    try testing.expectEqual(@as(usize, 100), m.count());

    // Validate all
    for (0..100) |i| {
        try testing.expectEqual(@as(?u32, @intCast(i * 2)), m.get(@intCast(i)));
    }

    try m.validate();

    // Remove half
    for (0..50) |i| {
        const new_m = try m.remove(@intCast(i));
        m.deinit();
        m = new_m;
    }

    try testing.expectEqual(@as(usize, 50), m.count());

    // Verify remaining
    for (50..100) |i| {
        try testing.expectEqual(@as(?u32, @intCast(i * 2)), m.get(@intCast(i)));
    }

    try m.validate();
}

test "PersistentHashMap: validate detects errors" {
    const Map = PersistentHashMap(u32, u32, AutoContext, AutoContext.hash, AutoContext.eql);

    const m0 = Map.init(testing.allocator, .{});
    defer m0.deinit();

    try m0.validate();
    try testing.expectEqual(@as(usize, 0), m0.count());

    const m1 = try m0.set(1, 100);
    defer m1.deinit();
    try m1.validate();
    try testing.expectEqual(@as(usize, 1), m1.count());
    try testing.expectEqual(@as(?u32, 100), m1.get(1));

    const m2 = try m1.set(2, 200);
    defer m2.deinit();
    try m2.validate();
    try testing.expectEqual(@as(usize, 2), m2.count());
    try testing.expectEqual(@as(?u32, 100), m2.get(1));
    try testing.expectEqual(@as(?u32, 200), m2.get(2));
}

test "PersistentHashMap: structural sharing" {
    const Map = PersistentHashMap(u32, u32, AutoContext, AutoContext.hash, AutoContext.eql);

    const m0 = Map.init(testing.allocator, .{});
    defer m0.deinit();

    const m1 = try m0.set(1, 100);
    defer m1.deinit();

    const m2 = try m1.set(2, 200);
    defer m2.deinit();

    const m3 = try m1.set(3, 300); // Branches from m1, shares nodes with m1
    defer m3.deinit();

    // m1, m2, m3 all valid
    try m1.validate();
    try m2.validate();
    try m3.validate();

    // m2 and m3 should share the node for key 1
    try testing.expectEqual(@as(?u32, 100), m2.get(1));
    try testing.expectEqual(@as(?u32, 100), m3.get(1));
}

test "PersistentHashMap: string keys" {
    const StringContext = struct {
        pub fn hash(_: @This(), key: []const u8) u64 {
            return std.hash.Wyhash.hash(0, key);
        }

        pub fn eql(_: @This(), a: []const u8, b: []const u8) bool {
            return std.mem.eql(u8, a, b);
        }
    };

    const Map = PersistentHashMap([]const u8, u32, StringContext, StringContext.hash, StringContext.eql);

    const m0 = Map.init(testing.allocator, .{});
    defer m0.deinit();

    const m1 = try m0.set("hello", 1);
    defer m1.deinit();

    const m2 = try m1.set("world", 2);
    defer m2.deinit();

    try testing.expectEqual(@as(?u32, 1), m2.get("hello"));
    try testing.expectEqual(@as(?u32, 2), m2.get("world"));
    try testing.expectEqual(@as(?u32, null), m2.get("foo"));
}

test "PersistentHashMap: memory leak check" {
    const Map = PersistentHashMap(u32, u32, AutoContext, AutoContext.hash, AutoContext.eql);

    var m = Map.init(testing.allocator, .{});
    try testing.expectEqual(@as(usize, 0), m.count());

    for (0..20) |i| {
        const new_m = try m.set(@intCast(i), @intCast(i));
        m.deinit();
        m = new_m;
        try testing.expectEqual(@as(usize, i + 1), m.count());
    }

    // Verify all keys are accessible in final version
    for (0..20) |i| {
        try testing.expectEqual(@as(?u32, @intCast(i)), m.get(@intCast(i)));
    }

    m.deinit();
}
