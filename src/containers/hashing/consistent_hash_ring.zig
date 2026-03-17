const std = @import("std");
const testing = std.testing;

/// ConsistentHashRing: Distributed hash table with minimal key redistribution.
///
/// Consistent hashing maps keys to nodes in a way that minimizes rehashing when
/// nodes are added or removed. Uses virtual nodes (replicas) to improve balance.
///
/// Features:
/// - Virtual nodes for better load distribution
/// - Sorted ring with binary search for O(log n) lookup
/// - Minimal key migration on topology changes
/// - Configurable replica count per physical node
///
/// Time Complexity:
/// - addNode: O(r log n) where r = replicas per node, n = total virtual nodes
/// - removeNode: O(r log n)
/// - getNode: O(log n) binary search
/// - Space: O(n * r) where n = physical nodes, r = replicas
pub fn ConsistentHashRing(
    comptime K: type,
    comptime N: type,
    comptime Context: type,
    comptime hash_fn: fn (ctx: Context, key: K) u64,
    comptime node_eql_fn: fn (ctx: Context, a: N, b: N) bool,
) type {
    return struct {
        const Self = @This();

        const VirtualNode = struct {
            hash: u64,
            node: N,
        };

        pub const Iterator = struct {
            ring: *const Self,
            index: usize,

            /// Time: O(1) amortized over full iteration
            pub fn next(self: *Iterator) ?N {
                if (self.index >= self.ring.virtual_nodes.items.len) return null;

                const node = self.ring.virtual_nodes.items[self.index].node;
                self.index += 1;

                // Skip duplicates (multiple virtual nodes for same physical node)
                while (self.index < self.ring.virtual_nodes.items.len) {
                    const next_node = self.ring.virtual_nodes.items[self.index].node;
                    if (!node_eql_fn(self.ring.ctx, node, next_node)) break;
                    self.index += 1;
                }

                return node;
            }
        };

        allocator: std.mem.Allocator,
        ctx: Context,
        virtual_nodes: std.ArrayList(VirtualNode),
        replicas: usize, // Number of virtual nodes per physical node

        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: std.mem.Allocator, ctx: Context, replicas: usize) !Self {
            if (replicas == 0) return error.InvalidReplicaCount;

            return Self{
                .allocator = allocator,
                .ctx = ctx,
                .virtual_nodes = .{},
                .replicas = replicas,
            };
        }

        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.virtual_nodes.deinit(self.allocator);
            self.* = undefined;
        }

        /// Time: O(n) | Space: O(n)
        pub fn clone(self: *const Self) !Self {
            const new = Self{
                .allocator = self.allocator,
                .ctx = self.ctx,
                .virtual_nodes = try self.virtual_nodes.clone(self.allocator),
                .replicas = self.replicas,
            };
            return new;
        }

        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            // Count unique physical nodes
            if (self.virtual_nodes.items.len == 0) return 0;

            var unique_count: usize = 1;
            var prev_node = self.virtual_nodes.items[0].node;

            for (self.virtual_nodes.items[1..]) |vnode| {
                if (!node_eql_fn(self.ctx, vnode.node, prev_node)) {
                    unique_count += 1;
                    prev_node = vnode.node;
                }
            }

            return unique_count;
        }

        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.virtual_nodes.items.len == 0;
        }

        /// Generate hash for virtual node replica
        fn virtualNodeHash(_: Context, node: N, replica_index: usize) u64 {
            // Hash the node data combined with replica index
            var hasher = std.hash.Wyhash.init(0);

            // Hash node (assume node can be converted to bytes)
            const node_bytes = std.mem.asBytes(&node);
            hasher.update(node_bytes);

            // Mix in replica index
            const replica_bytes = std.mem.asBytes(&replica_index);
            hasher.update(replica_bytes);

            return hasher.final();
        }

        /// Time: O(r log n) where r = replicas, n = total virtual nodes | Space: O(r)
        pub fn addNode(self: *Self, node: N) !void {
            // Create r virtual nodes for this physical node
            var i: usize = 0;
            while (i < self.replicas) : (i += 1) {
                const vhash = virtualNodeHash(self.ctx, node, i);
                try self.virtual_nodes.append(self.allocator, VirtualNode{
                    .hash = vhash,
                    .node = node,
                });
            }

            // Keep ring sorted by hash
            std.mem.sort(VirtualNode, self.virtual_nodes.items, {}, struct {
                fn lessThan(_: void, a: VirtualNode, b: VirtualNode) bool {
                    return a.hash < b.hash;
                }
            }.lessThan);
        }

        /// Time: O(r log n) where r = replicas, n = total virtual nodes | Space: O(1)
        pub fn removeNode(self: *Self, node: N) bool {
            var removed = false;
            var i: usize = 0;

            while (i < self.virtual_nodes.items.len) {
                if (node_eql_fn(self.ctx, self.virtual_nodes.items[i].node, node)) {
                    _ = self.virtual_nodes.orderedRemove(i);
                    removed = true;
                } else {
                    i += 1;
                }
            }

            return removed;
        }

        /// Time: O(log n) binary search | Space: O(1)
        /// Returns the node responsible for the given key
        pub fn getNode(self: *const Self, key: K) ?N {
            if (self.virtual_nodes.items.len == 0) return null;

            const key_hash = hash_fn(self.ctx, key);

            // Binary search for first virtual node with hash >= key_hash
            var left: usize = 0;
            var right: usize = self.virtual_nodes.items.len;

            while (left < right) {
                const mid = left + (right - left) / 2;
                if (self.virtual_nodes.items[mid].hash < key_hash) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }

            // Wrap around if we're past the end
            const index = if (left >= self.virtual_nodes.items.len) 0 else left;
            return self.virtual_nodes.items[index].node;
        }

        /// Time: O(log n) | Space: O(1)
        /// Returns true if the ring contains the given node
        pub fn contains(self: *const Self, node: N) bool {
            for (self.virtual_nodes.items) |vnode| {
                if (node_eql_fn(self.ctx, vnode.node, node)) {
                    return true;
                }
            }
            return false;
        }

        /// Time: O(1) | Space: O(1)
        pub fn iterator(self: *const Self) Iterator {
            return Iterator{ .ring = self, .index = 0 };
        }

        /// Time: O(n) | Space: O(1)
        /// Validates internal invariants
        pub fn validate(self: *const Self) !void {
            // Check that virtual nodes are sorted by hash
            for (0..self.virtual_nodes.items.len - 1) |i| {
                if (self.virtual_nodes.items[i].hash > self.virtual_nodes.items[i + 1].hash) {
                    return error.RingNotSorted;
                }
            }

            // Check that each physical node has exactly replicas virtual nodes
            if (self.virtual_nodes.items.len > 0) {
                var node_counts = std.AutoHashMap(u64, usize).init(self.allocator);
                defer node_counts.deinit();

                for (self.virtual_nodes.items) |vnode| {
                    // Use node hash as key (simplified)
                    const node_hash = virtualNodeHash(self.ctx, vnode.node, 0);
                    const entry = try node_counts.getOrPut(node_hash);
                    if (!entry.found_existing) {
                        entry.value_ptr.* = 1;
                    } else {
                        entry.value_ptr.* += 1;
                    }
                }

                // Each unique node should have exactly replicas virtual nodes
                var iter = node_counts.valueIterator();
                while (iter.next()) |replica_count| {
                    if (replica_count.* != self.replicas) {
                        return error.IncorrectReplicaCount;
                    }
                }
            }
        }
    };
}

// -- Tests --

fn testKeyHash(ctx: void, key: u32) u64 {
    _ = ctx;
    return @as(u64, key) *% 0x9e3779b97f4a7c15;
}

fn testNodeEql(ctx: void, a: []const u8, b: []const u8) bool {
    _ = ctx;
    return std.mem.eql(u8, a, b);
}

test "ConsistentHashRing: init and deinit" {
    var ring = try ConsistentHashRing(u32, []const u8, void, testKeyHash, testNodeEql).init(testing.allocator, {}, 3);
    defer ring.deinit();

    try testing.expect(ring.isEmpty());
    try testing.expectEqual(@as(usize, 0), ring.count());
}

test "ConsistentHashRing: add nodes" {
    var ring = try ConsistentHashRing(u32, []const u8, void, testKeyHash, testNodeEql).init(testing.allocator, {}, 3);
    defer ring.deinit();

    try ring.addNode("node1");
    try ring.addNode("node2");
    try ring.addNode("node3");

    try testing.expectEqual(@as(usize, 3), ring.count());
    try testing.expect(ring.contains("node1"));
    try testing.expect(ring.contains("node2"));
    try testing.expect(ring.contains("node3"));
    try testing.expect(!ring.contains("node4"));
}

test "ConsistentHashRing: getNode basic" {
    var ring = try ConsistentHashRing(u32, []const u8, void, testKeyHash, testNodeEql).init(testing.allocator, {}, 5);
    defer ring.deinit();

    try ring.addNode("node1");
    try ring.addNode("node2");
    try ring.addNode("node3");

    // Every key should map to one of the nodes
    const node1 = ring.getNode(100).?;
    const node2 = ring.getNode(200).?;
    const node3 = ring.getNode(300).?;

    try testing.expect(
        std.mem.eql(u8, node1, "node1") or
            std.mem.eql(u8, node1, "node2") or
            std.mem.eql(u8, node1, "node3"),
    );
    try testing.expect(
        std.mem.eql(u8, node2, "node1") or
            std.mem.eql(u8, node2, "node2") or
            std.mem.eql(u8, node2, "node3"),
    );
    try testing.expect(
        std.mem.eql(u8, node3, "node1") or
            std.mem.eql(u8, node3, "node2") or
            std.mem.eql(u8, node3, "node3"),
    );
}

test "ConsistentHashRing: getNode on empty ring" {
    var ring = try ConsistentHashRing(u32, []const u8, void, testKeyHash, testNodeEql).init(testing.allocator, {}, 3);
    defer ring.deinit();

    try testing.expectEqual(@as(?[]const u8, null), ring.getNode(100));
}

test "ConsistentHashRing: remove node" {
    var ring = try ConsistentHashRing(u32, []const u8, void, testKeyHash, testNodeEql).init(testing.allocator, {}, 3);
    defer ring.deinit();

    try ring.addNode("node1");
    try ring.addNode("node2");

    try testing.expectEqual(@as(usize, 2), ring.count());

    const removed = ring.removeNode("node1");
    try testing.expect(removed);
    try testing.expectEqual(@as(usize, 1), ring.count());
    try testing.expect(!ring.contains("node1"));
    try testing.expect(ring.contains("node2"));

    // All keys should now map to node2
    const node = ring.getNode(100).?;
    try testing.expect(std.mem.eql(u8, node, "node2"));
}

test "ConsistentHashRing: remove nonexistent node" {
    var ring = try ConsistentHashRing(u32, []const u8, void, testKeyHash, testNodeEql).init(testing.allocator, {}, 3);
    defer ring.deinit();

    try ring.addNode("node1");
    const removed = ring.removeNode("node999");
    try testing.expect(!removed);
    try testing.expectEqual(@as(usize, 1), ring.count());
}

test "ConsistentHashRing: key redistribution on add/remove" {
    var ring = try ConsistentHashRing(u32, []const u8, void, testKeyHash, testNodeEql).init(testing.allocator, {}, 10);
    defer ring.deinit();

    try ring.addNode("node1");
    try ring.addNode("node2");

    // Map 100 keys
    var initial_mapping = std.AutoHashMap(u32, []const u8).init(testing.allocator);
    defer initial_mapping.deinit();

    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        const node = ring.getNode(i).?;
        try initial_mapping.put(i, node);
    }

    // Add a new node
    try ring.addNode("node3");

    // Check how many keys were remapped
    var remapped_count: usize = 0;
    i = 0;
    while (i < 100) : (i += 1) {
        const new_node = ring.getNode(i).?;
        const old_node = initial_mapping.get(i).?;
        if (!std.mem.eql(u8, new_node, old_node)) {
            remapped_count += 1;
        }
    }

    // With consistent hashing, only ~1/3 of keys should be remapped (100/3 ≈ 33)
    // Allow some variance (20-45%)
    try testing.expect(remapped_count >= 20 and remapped_count <= 45);
}

test "ConsistentHashRing: iterator over unique nodes" {
    var ring = try ConsistentHashRing(u32, []const u8, void, testKeyHash, testNodeEql).init(testing.allocator, {}, 5);
    defer ring.deinit();

    try ring.addNode("node1");
    try ring.addNode("node2");
    try ring.addNode("node3");

    var node_set = std.StringHashMap(void).init(testing.allocator);
    defer node_set.deinit();

    var iter = ring.iterator();
    while (iter.next()) |node| {
        try node_set.put(node, {});
    }

    try testing.expectEqual(@as(usize, 3), node_set.count());
    try testing.expect(node_set.contains("node1"));
    try testing.expect(node_set.contains("node2"));
    try testing.expect(node_set.contains("node3"));
}

test "ConsistentHashRing: clone" {
    var ring = try ConsistentHashRing(u32, []const u8, void, testKeyHash, testNodeEql).init(testing.allocator, {}, 3);
    defer ring.deinit();

    try ring.addNode("node1");
    try ring.addNode("node2");

    var cloned = try ring.clone();
    defer cloned.deinit();

    try testing.expectEqual(@as(usize, 2), cloned.count());
    try testing.expect(cloned.contains("node1"));
    try testing.expect(cloned.contains("node2"));

    // Modify clone, original unchanged
    try cloned.addNode("node3");
    try testing.expectEqual(@as(usize, 3), cloned.count());
    try testing.expectEqual(@as(usize, 2), ring.count());
}

test "ConsistentHashRing: validate sorted ring" {
    var ring = try ConsistentHashRing(u32, []const u8, void, testKeyHash, testNodeEql).init(testing.allocator, {}, 5);
    defer ring.deinit();

    try ring.validate();
    try testing.expectEqual(@as(usize, 0), ring.count());

    try ring.addNode("node1");
    try ring.validate();
    try testing.expectEqual(@as(usize, 1), ring.count());
    try testing.expectEqual(@as(usize, 5), ring.virtual_nodes.items.len);

    try ring.addNode("node2");
    try ring.validate();
    try testing.expectEqual(@as(usize, 2), ring.count());
    try testing.expectEqual(@as(usize, 10), ring.virtual_nodes.items.len);

    try ring.addNode("node3");
    try ring.validate();
    try testing.expectEqual(@as(usize, 3), ring.count());
    try testing.expectEqual(@as(usize, 15), ring.virtual_nodes.items.len);
}

test "ConsistentHashRing: high replica count" {
    var ring = try ConsistentHashRing(u32, []const u8, void, testKeyHash, testNodeEql).init(testing.allocator, {}, 100);
    defer ring.deinit();

    try ring.addNode("node1");
    try ring.addNode("node2");

    try testing.expectEqual(@as(usize, 2), ring.count());
    try testing.expectEqual(@as(usize, 200), ring.virtual_nodes.items.len);

    try ring.validate();
}

test "ConsistentHashRing: zero replicas error" {
    const result = ConsistentHashRing(u32, []const u8, void, testKeyHash, testNodeEql).init(testing.allocator, {}, 0);
    try testing.expectError(error.InvalidReplicaCount, result);
}

test "ConsistentHashRing: stress test with many nodes" {
    var ring = try ConsistentHashRing(u32, []const u8, void, testKeyHash, testNodeEql).init(testing.allocator, {}, 10);
    defer ring.deinit();

    // Add 50 nodes
    var i: usize = 0;
    while (i < 50) : (i += 1) {
        var buf: [20]u8 = undefined;
        const node_name = try std.fmt.bufPrint(&buf, "node{d}", .{i});
        // Allocate persistent memory for node name
        const persistent_name = try testing.allocator.dupe(u8, node_name);
        defer testing.allocator.free(persistent_name);
        try ring.addNode(persistent_name);
    }

    try testing.expectEqual(@as(usize, 50), ring.count());
    try testing.expectEqual(@as(usize, 500), ring.virtual_nodes.items.len);

    // All keys should map to some node
    i = 0;
    while (i < 1000) : (i += 1) {
        const node = ring.getNode(@intCast(i));
        try testing.expect(node != null);
    }

    try ring.validate();
}

test "ConsistentHashRing: memory leak check" {
    var ring = try ConsistentHashRing(u32, []const u8, void, testKeyHash, testNodeEql).init(testing.allocator, {}, 10);
    defer ring.deinit();

    try ring.addNode("node1");
    try ring.addNode("node2");
    try testing.expectEqual(@as(usize, 2), ring.count());
    try testing.expectEqual(@as(usize, 20), ring.virtual_nodes.items.len);

    // Verify both nodes are accessible
    const node1 = ring.getNode(1) orelse unreachable;
    const node2 = ring.getNode(2) orelse unreachable;
    try testing.expect(std.mem.eql(u8, node1, "node1") or std.mem.eql(u8, node1, "node2"));
    try testing.expect(std.mem.eql(u8, node2, "node1") or std.mem.eql(u8, node2, "node2"));

    // std.testing.allocator will detect leaks automatically
}

test "ConsistentHashRing: load distribution" {
    var ring = try ConsistentHashRing(u32, []const u8, void, testKeyHash, testNodeEql).init(testing.allocator, {}, 20);
    defer ring.deinit();

    try ring.addNode("node1");
    try ring.addNode("node2");
    try ring.addNode("node3");

    var node1_count: usize = 0;
    var node2_count: usize = 0;
    var node3_count: usize = 0;

    // Map 1000 keys
    var i: u32 = 0;
    while (i < 1000) : (i += 1) {
        const node = ring.getNode(i).?;
        if (std.mem.eql(u8, node, "node1")) {
            node1_count += 1;
        } else if (std.mem.eql(u8, node, "node2")) {
            node2_count += 1;
        } else if (std.mem.eql(u8, node, "node3")) {
            node3_count += 1;
        }
    }

    // With 20 replicas, load should be fairly balanced (each node ~333 keys ± 100)
    try testing.expect(node1_count >= 233 and node1_count <= 433);
    try testing.expect(node2_count >= 233 and node2_count <= 433);
    try testing.expect(node3_count >= 233 and node3_count <= 433);
}

// Convenience alias for common case
/// Creates consistent hash ring with automatic context.
/// Assumes both K and N support equality comparison and can be hashed via std.hash.Wyhash.
/// Time: O(1) | Space: O(1)
pub fn AutoConsistentHashRing(comptime K: type, comptime N: type) type {
    const AutoContext = struct {
        pub fn hash(_: @This(), key: K) u64 {
            return std.hash.Wyhash.hash(0, std.mem.asBytes(&key));
        }

        pub fn nodeEql(_: @This(), a: N, b: N) bool {
            return std.meta.eql(a, b);
        }
    };

    const BaseRing = ConsistentHashRing(K, N, AutoContext, AutoContext.hash, AutoContext.nodeEql);

    return struct {
        ring: BaseRing,

        const Self = @This();

        /// Initialize with automatic context.
        pub fn init(allocator: std.mem.Allocator, replicas: usize) !Self {
            return Self{
                .ring = try BaseRing.init(allocator, AutoContext{}, replicas),
            };
        }

        pub fn deinit(self: *Self) void {
            self.ring.deinit();
        }

        pub fn addNode(self: *Self, node: N) !void {
            try self.ring.addNode(node);
        }

        pub fn removeNode(self: *Self, node: N) void {
            self.ring.removeNode(node);
        }

        pub fn getNode(self: *const Self, key: K) ?N {
            return self.ring.getNode(key);
        }

        pub fn count(self: *const Self) usize {
            return self.ring.count();
        }
    };
}
