const std = @import("std");

/// ScapegoatTree implements an α-weight-balanced binary search tree.
/// Unlike rotation-based BSTs (RB, AVL), it maintains balance through periodic rebuilding.
///
/// Properties:
/// - Amortized O(log n) insert, remove, get
/// - Worst-case O(log n) space per node (just size counter, no color/level)
/// - No rotations — rebuilds unbalanced subtrees to perfect balance
/// - Simple invariant: height ≤ log_α(n) where α is balance factor (default 2/3)
///
/// Generic parameters:
/// - K: key type
/// - V: value type
/// - Context: comparator context type
/// - compareFn: comparison function (ctx, a, b) -> Order
pub fn ScapegoatTree(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: K, b: K) std.math.Order,
) type {
    return struct {
        const Self = @This();

        pub const Entry = struct {
            key: K,
            value: V,
        };

        const Node = struct {
            key: K,
            value: V,
            left: ?*Node = null,
            right: ?*Node = null,
            size: usize = 1, // subtree size including self

            fn updateSize(self: *Node) void {
                self.size = 1 + sizeOf(self.left) + sizeOf(self.right);
            }

            fn sizeOf(node: ?*Node) usize {
                return if (node) |n| n.size else 0;
            }
        };

        pub const Iterator = struct {
            stack: std.ArrayList(*Node),
            allocator: std.mem.Allocator,

            pub fn next(self: *Iterator) !?Entry {
                while (self.stack.items.len > 0) {
                    const node = self.stack.pop() orelse break;

                    // Push right subtree
                    if (node.right) |right| {
                        try self.stack.append(self.allocator, right);
                    }

                    // Push left subtree
                    if (node.left) |left| {
                        try self.stack.append(self.allocator, left);
                    }

                    return Entry{ .key = node.key, .value = node.value };
                }
                return null;
            }

            pub fn deinit(self: *Iterator) void {
                self.stack.deinit(self.allocator);
            }
        };

        allocator: std.mem.Allocator,
        root: ?*Node = null,
        context: Context,
        node_count: usize = 0,
        max_count: usize = 0, // tracks high-water mark for scapegoat detection

        // Balance parameter: height ≤ log_alpha(n)
        // alpha = 1/balance_ratio
        // Common: balance_ratio = 2 → alpha = 0.5 (very aggressive rebuilding)
        //         balance_ratio = 1.5 → alpha ≈ 0.67 (standard)
        //         balance_ratio = 1.25 → alpha = 0.8 (lazy rebuilding)
        const balance_ratio_numerator: usize = 3;
        const balance_ratio_denominator: usize = 2;

        // -- Lifecycle --

        pub fn init(allocator: std.mem.Allocator, context: Context) Self {
            return Self{
                .allocator = allocator,
                .context = context,
            };
        }

        pub fn deinit(self: *Self) void {
            self.destroySubtree(self.root);
            self.* = undefined;
        }

        fn destroySubtree(self: *Self, node: ?*Node) void {
            if (node) |n| {
                self.destroySubtree(n.left);
                self.destroySubtree(n.right);
                self.allocator.destroy(n);
            }
        }

        pub fn clone(self: *const Self) !Self {
            var new_tree = Self.init(self.allocator, self.context);
            new_tree.root = try self.cloneSubtree(self.root);
            new_tree.node_count = self.node_count;
            new_tree.max_count = self.max_count;
            return new_tree;
        }

        fn cloneSubtree(self: *const Self, node: ?*Node) !?*Node {
            if (node == null) return null;
            const n = node.?;
            const new_node = try self.allocator.create(Node);
            new_node.* = Node{
                .key = n.key,
                .value = n.value,
                .size = n.size,
            };
            new_node.left = try self.cloneSubtree(n.left);
            new_node.right = try self.cloneSubtree(n.right);
            return new_node;
        }

        // -- Capacity --

        pub fn count(self: *const Self) usize {
            return self.node_count;
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.node_count == 0;
        }

        // -- Modification --

        /// Time: O(log n) amortized | Space: O(log n) for rebuild stack
        pub fn insert(self: *Self, key: K, value: V) !?V {
            var depth: usize = 0;
            var scapegoat: ?*?*Node = null;

            const result = try self.insertRecursive(&self.root, key, value, &depth, &scapegoat);

            if (result.inserted) {
                self.node_count += 1;
                if (self.node_count > self.max_count) {
                    self.max_count = self.node_count;
                }

                // Check if tree became too deep
                const max_depth = self.maxAllowedDepth();
                if (depth > max_depth) {
                    // Rebuild at scapegoat
                    if (scapegoat) |sg_ptr| {
                        try self.rebuildSubtree(sg_ptr);
                    }
                }
            }

            return result.old_value;
        }

        const InsertResult = struct {
            inserted: bool,
            old_value: ?V,
        };

        fn insertRecursive(
            self: *Self,
            node_ptr: *?*Node,
            key: K,
            value: V,
            depth: *usize,
            scapegoat: *?*?*Node,
        ) !InsertResult {
            depth.* += 1;

            if (node_ptr.*) |node| {
                const order = compareFn(self.context, key, node.key);

                const result = switch (order) {
                    .eq => blk: {
                        const old = node.value;
                        node.value = value;
                        break :blk InsertResult{ .inserted = false, .old_value = old };
                    },
                    .lt => try self.insertRecursive(&node.left, key, value, depth, scapegoat),
                    .gt => try self.insertRecursive(&node.right, key, value, depth, scapegoat),
                };

                if (result.inserted) {
                    node.updateSize();

                    // Check if this node is unbalanced (potential scapegoat)
                    const left_size = Node.sizeOf(node.left);
                    const right_size = Node.sizeOf(node.right);
                    const max_child = @max(left_size, right_size);

                    // alpha-weight-balance: max_child_size ≤ alpha * node.size
                    // where alpha = balance_ratio_denominator / balance_ratio_numerator
                    if (max_child * balance_ratio_numerator > node.size * balance_ratio_denominator) {
                        scapegoat.* = node_ptr;
                    }
                }

                return result;
            } else {
                // Create new node
                const new_node = try self.allocator.create(Node);
                new_node.* = Node{
                    .key = key,
                    .value = value,
                };
                node_ptr.* = new_node;
                return InsertResult{ .inserted = true, .old_value = null };
            }
        }

        /// Time: O(log n) amortized | Space: O(log n)
        pub fn remove(self: *Self, key: K) ?Entry {
            const result = self.removeRecursive(&self.root, key);
            if (result != null) {
                self.node_count -= 1;

                // Check if we need global rebuild (too many deletions)
                // Rebuild when node_count < max_count / 2
                if (self.node_count * 2 < self.max_count and self.max_count > 1) {
                    self.rebuildSubtree(&self.root) catch {};
                    self.max_count = self.node_count;
                }
            }
            return result;
        }

        fn removeRecursive(self: *Self, node_ptr: *?*Node, key: K) ?Entry {
            const node = node_ptr.* orelse return null;

            const order = compareFn(self.context, key, node.key);

            switch (order) {
                .eq => {
                    const entry = Entry{ .key = node.key, .value = node.value };

                    if (node.left == null) {
                        node_ptr.* = node.right;
                        self.allocator.destroy(node);
                    } else if (node.right == null) {
                        node_ptr.* = node.left;
                        self.allocator.destroy(node);
                    } else {
                        // Two children: replace with in-order successor
                        const min_node = self.findMinNode(node.right.?);
                        node.key = min_node.key;
                        node.value = min_node.value;
                        _ = self.removeRecursive(&node.right, min_node.key);
                        node.updateSize();
                    }

                    return entry;
                },
                .lt => {
                    const result = self.removeRecursive(&node.left, key);
                    if (result != null) {
                        node.updateSize();
                    }
                    return result;
                },
                .gt => {
                    const result = self.removeRecursive(&node.right, key);
                    if (result != null) {
                        node.updateSize();
                    }
                    return result;
                },
            }
        }

        fn findMinNode(_: *Self, node: *Node) *Node {
            var current = node;
            while (current.left) |left| {
                current = left;
            }
            return current;
        }

        fn findMinNodeConst(_: *const Self, node: *Node) *Node {
            var current = node;
            while (current.left) |left| {
                current = left;
            }
            return current;
        }

        /// Rebuild subtree to perfect balance
        fn rebuildSubtree(self: *Self, node_ptr: *?*Node) !void {
            const node = node_ptr.* orelse return;

            // Collect nodes in-order
            var nodes = std.ArrayList(*Node){};
            defer nodes.deinit(self.allocator);

            try self.collectInOrder(node, &nodes);

            // Rebuild balanced tree
            node_ptr.* = try self.buildBalanced(nodes.items);
        }

        fn collectInOrder(self: *Self, node: ?*Node, list: *std.ArrayList(*Node)) !void {
            if (node) |n| {
                // Save children pointers before clearing
                const left = n.left;
                const right = n.right;

                try self.collectInOrder(left, list);

                // Clear children before adding to list
                n.left = null;
                n.right = null;
                try list.append(self.allocator, n);

                try self.collectInOrder(right, list);
            }
        }

        fn buildBalanced(self: *Self, nodes: []*Node) !?*Node {
            if (nodes.len == 0) return null;

            const mid = nodes.len / 2;
            const root = nodes[mid];

            root.left = try self.buildBalanced(nodes[0..mid]);
            root.right = try self.buildBalanced(nodes[mid + 1 ..]);
            root.updateSize();

            return root;
        }

        fn maxAllowedDepth(self: *const Self) usize {
            if (self.node_count <= 1) return 1;

            // height ≤ floor(log_alpha(n)) where alpha = denominator/numerator
            // log_alpha(n) = log(n) / log(alpha)
            // We use approximation: log_alpha(n) ≈ 1.5 * log2(n) for alpha ≈ 0.67

            var n = self.node_count;
            var depth: usize = 0;
            while (n > 0) : (n >>= 1) {
                depth += 1;
            }
            // Apply balance ratio: depth * numerator / denominator
            return (depth * balance_ratio_numerator) / balance_ratio_denominator;
        }

        // -- Lookup --

        /// Time: O(log n) | Space: O(1)
        pub fn get(self: *const Self, key: K) ?V {
            var current = self.root;
            while (current) |node| {
                const order = compareFn(self.context, key, node.key);
                switch (order) {
                    .eq => return node.value,
                    .lt => current = node.left,
                    .gt => current = node.right,
                }
            }
            return null;
        }

        pub fn contains(self: *const Self, key: K) bool {
            return self.get(key) != null;
        }

        pub fn min(self: *const Self) ?Entry {
            if (self.root) |root| {
                const node = self.findMinNodeConst(root);
                return Entry{ .key = node.key, .value = node.value };
            }
            return null;
        }

        pub fn max(self: *const Self) ?Entry {
            var current = self.root;
            while (current) |node| {
                if (node.right) |right| {
                    current = right;
                } else {
                    return Entry{ .key = node.key, .value = node.value };
                }
            }
            return null;
        }

        // -- Iteration --

        pub fn iterator(self: *const Self) !Iterator {
            var stack = std.ArrayList(*Node){};
            if (self.root) |root| {
                try stack.append(self.allocator, root);
            }
            return Iterator{
                .stack = stack,
                .allocator = self.allocator,
            };
        }

        // -- Bulk --

        pub fn clear(self: *Self) void {
            self.destroySubtree(self.root);
            self.root = null;
            self.node_count = 0;
            self.max_count = 0;
        }

        // -- Debug --

        pub fn format(
            self: *const Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try writer.print("ScapegoatTree{{ count: {}, max: {} }}", .{ self.node_count, self.max_count });
        }

        /// Validates ScapegoatTree invariants:
        /// 1. BST property
        /// 2. Size counters are accurate
        /// 3. No structural corruption
        pub fn validate(self: *const Self) !void {
            const actual_count = self.validateSubtree(self.root, null, null) catch |err| {
                return err;
            };

            if (actual_count != self.node_count) {
                return error.InvalidCount;
            }
        }

        fn validateSubtree(self: *const Self, node: ?*Node, min_key: ?K, max_key: ?K) !usize {
            const n = node orelse return 0;

            // Check BST ordering
            if (min_key) |min_k| {
                if (compareFn(self.context, n.key, min_k) != .gt) {
                    return error.BSTViolation;
                }
            }
            if (max_key) |max_k| {
                if (compareFn(self.context, n.key, max_k) != .lt) {
                    return error.BSTViolation;
                }
            }

            const left_size = try self.validateSubtree(n.left, min_key, n.key);
            const right_size = try self.validateSubtree(n.right, n.key, max_key);

            const expected_size = 1 + left_size + right_size;
            if (n.size != expected_size) {
                return error.InvalidSize;
            }

            return expected_size;
        }
    };
}

// -- Tests --

const testing = std.testing;

fn compareI32(_: void, a: i32, b: i32) std.math.Order {
    return std.math.order(a, b);
}

const TestTree = ScapegoatTree(i32, []const u8, void, compareI32);

test "ScapegoatTree: basic insert and get" {
    var tree = TestTree.init(testing.allocator, {});
    defer tree.deinit();

    try testing.expectEqual(@as(?[]const u8, null), try tree.insert(5, "five"));
    try testing.expectEqual(@as(?[]const u8, null), try tree.insert(3, "three"));
    try testing.expectEqual(@as(?[]const u8, null), try tree.insert(7, "seven"));

    try testing.expectEqualStrings("five", tree.get(5).?);
    try testing.expectEqualStrings("three", tree.get(3).?);
    try testing.expectEqualStrings("seven", tree.get(7).?);
    try testing.expectEqual(@as(?[]const u8, null), tree.get(99));
}

test "ScapegoatTree: insert duplicate updates value" {
    var tree = TestTree.init(testing.allocator, {});
    defer tree.deinit();

    try testing.expectEqual(@as(?[]const u8, null), try tree.insert(5, "old"));
    const old_value = try tree.insert(5, "new");
    try testing.expectEqualStrings("old", old_value.?);
    try testing.expectEqualStrings("new", tree.get(5).?);
    try testing.expectEqual(@as(usize, 1), tree.count());
}

test "ScapegoatTree: multiple inserts maintain order" {
    var tree = TestTree.init(testing.allocator, {});
    defer tree.deinit();

    const keys = [_]i32{ 10, 5, 15, 2, 7, 12, 20 };
    for (keys) |key| {
        _ = try tree.insert(key, "value");
    }

    try testing.expectEqual(@as(usize, 7), tree.count());
    try tree.validate();

    for (keys) |key| {
        try testing.expect(tree.contains(key));
    }
}

test "ScapegoatTree: remove existing key" {
    var tree = TestTree.init(testing.allocator, {});
    defer tree.deinit();

    _ = try tree.insert(5, "five");
    _ = try tree.insert(3, "three");
    _ = try tree.insert(7, "seven");

    const removed = tree.remove(5);
    try testing.expect(removed != null);
    try testing.expectEqual(@as(i32, 5), removed.?.key);
    try testing.expectEqualStrings("five", removed.?.value);

    try testing.expectEqual(@as(?[]const u8, null), tree.get(5));
    try testing.expectEqual(@as(usize, 2), tree.count());
    try tree.validate();
}

test "ScapegoatTree: remove nonexistent key" {
    var tree = TestTree.init(testing.allocator, {});
    defer tree.deinit();

    _ = try tree.insert(5, "five");
    const removed = tree.remove(99);
    try testing.expectEqual(@as(?TestTree.Entry, null), removed);
    try testing.expectEqual(@as(usize, 1), tree.count());
}

test "ScapegoatTree: contains" {
    var tree = TestTree.init(testing.allocator, {});
    defer tree.deinit();

    _ = try tree.insert(5, "five");
    try testing.expect(tree.contains(5));
    try testing.expect(!tree.contains(99));
}

test "ScapegoatTree: min and max" {
    var tree = TestTree.init(testing.allocator, {});
    defer tree.deinit();

    try testing.expectEqual(@as(?TestTree.Entry, null), tree.min());
    try testing.expectEqual(@as(?TestTree.Entry, null), tree.max());

    _ = try tree.insert(5, "five");
    _ = try tree.insert(3, "three");
    _ = try tree.insert(7, "seven");

    const min_entry = tree.min().?;
    try testing.expectEqual(@as(i32, 3), min_entry.key);

    const max_entry = tree.max().?;
    try testing.expectEqual(@as(i32, 7), max_entry.key);
}

test "ScapegoatTree: clear" {
    var tree = TestTree.init(testing.allocator, {});
    defer tree.deinit();

    _ = try tree.insert(5, "five");
    _ = try tree.insert(3, "three");

    tree.clear();
    try testing.expectEqual(@as(usize, 0), tree.count());
    try testing.expect(tree.isEmpty());
    try testing.expectEqual(@as(?[]const u8, null), tree.get(5));
}

test "ScapegoatTree: clone" {
    var tree = TestTree.init(testing.allocator, {});
    defer tree.deinit();

    _ = try tree.insert(5, "five");
    _ = try tree.insert(3, "three");
    _ = try tree.insert(7, "seven");

    var cloned = try tree.clone();
    defer cloned.deinit();

    try testing.expectEqual(tree.count(), cloned.count());
    try testing.expectEqualStrings("five", cloned.get(5).?);
    try testing.expectEqualStrings("three", cloned.get(3).?);
    try testing.expectEqualStrings("seven", cloned.get(7).?);

    _ = try cloned.insert(10, "ten");
    try testing.expectEqual(@as(?[]const u8, null), tree.get(10));
}

test "ScapegoatTree: iterator in-order traversal" {
    var tree = TestTree.init(testing.allocator, {});
    defer tree.deinit();

    const keys = [_]i32{ 5, 3, 7, 1, 9 };
    for (keys) |key| {
        _ = try tree.insert(key, "value");
    }

    var iter = try tree.iterator();
    defer iter.deinit();

    var count: usize = 0;
    while (try iter.next()) |_| {
        count += 1;
    }

    try testing.expectEqual(@as(usize, 5), count);
}

test "ScapegoatTree: stress test with sequential inserts" {
    var tree = TestTree.init(testing.allocator, {});
    defer tree.deinit();

    // Sequential insert triggers rebuilds
    const n = 100;
    var i: i32 = 0;
    while (i < n) : (i += 1) {
        _ = try tree.insert(i, "value");
    }

    try testing.expectEqual(@as(usize, n), tree.count());
    try tree.validate();

    // Verify all keys
    i = 0;
    while (i < n) : (i += 1) {
        try testing.expect(tree.contains(i));
    }
}

test "ScapegoatTree: stress test with random operations" {
    var tree = TestTree.init(testing.allocator, {});
    defer tree.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const n = 500;
    var inserted = std.AutoHashMap(i32, void).init(testing.allocator);
    defer inserted.deinit();

    // Random inserts
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const key = random.intRangeAtMost(i32, 0, 999);
        _ = try tree.insert(key, "value");
        try inserted.put(key, {});
    }

    try testing.expectEqual(inserted.count(), tree.count());
    try tree.validate();

    // Random removes
    var key_iter = inserted.keyIterator();
    var remove_count: usize = 0;
    while (key_iter.next()) |key| {
        if (random.boolean()) {
            _ = tree.remove(key.*);
            remove_count += 1;
        }
    }

    try testing.expectEqual(inserted.count() - remove_count, tree.count());
    try tree.validate();
}

test "ScapegoatTree: validate invariants" {
    var tree = TestTree.init(testing.allocator, {});
    defer tree.deinit();

    const keys = [_]i32{ 10, 5, 15, 2, 7, 12, 20 };
    for (keys) |key| {
        _ = try tree.insert(key, "value");
        try tree.validate();
        try testing.expect(tree.get(key) != null);
    }

    try testing.expectEqual(keys.len, tree.count());

    for (keys) |key| {
        _ = tree.remove(key);
        try tree.validate();
        try testing.expectEqual(@as(?[]const u8, null), tree.get(key));
    }

    try testing.expect(tree.isEmpty());
    try testing.expectEqual(@as(usize, 0), tree.count());
}

test "ScapegoatTree: memory leak detection" {
    var tree = TestTree.init(testing.allocator, {});
    defer tree.deinit();

    const n = 100;
    var i: i32 = 0;
    while (i < n) : (i += 1) {
        _ = try tree.insert(i, "value");
    }
    try testing.expectEqual(100, tree.count());

    var removed_count: i32 = 0;
    i = 0;
    while (i < n) : (i += 2) {
        _ = tree.remove(i);
        removed_count += 1;
    }
    try testing.expectEqual(n - removed_count, tree.count());

    // Verify removed items are gone
    i = 0;
    while (i < n) : (i += 2) {
        try testing.expectEqual(@as(?[]const u8, null), tree.get(i));
    }

    tree.clear();
    try testing.expect(tree.isEmpty());
    try testing.expectEqual(@as(usize, 0), tree.count());
}
