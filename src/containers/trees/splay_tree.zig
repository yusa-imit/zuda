const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Self-adjusting binary search tree with splay operations.
/// Amortized O(log n) for all operations by moving accessed nodes to root.
/// Excellent for temporal locality — recently accessed items are fast.
///
/// Operations:
/// - insert, remove, get: O(log n) amortized
/// - Spatial locality benefits: repeated access to same key is O(1)
/// - No balance metadata needed (unlike AVL/RB trees)
pub fn SplayTree(
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
        };

        allocator: Allocator,
        context: Context,
        root: ?*Node = null,
        len: usize = 0,

        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator, context: Context) Self {
            return Self{
                .allocator = allocator,
                .context = context,
            };
        }

        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.clear();
        }

        /// Time: O(n) | Space: O(n)
        pub fn clone(self: *const Self) !Self {
            var new_tree = Self.init(self.allocator, self.context);
            errdefer new_tree.deinit();

            if (self.root) |root| {
                new_tree.root = try self.cloneNode(root);
                new_tree.len = self.len;
            }

            return new_tree;
        }

        fn cloneNode(self: *const Self, node: *Node) Allocator.Error!*Node {
            const new_node = try self.allocator.create(Node);
            new_node.* = .{
                .key = node.key,
                .value = node.value,
            };

            if (node.left) |left| {
                new_node.left = try self.cloneNode(left);
            }

            if (node.right) |right| {
                new_node.right = try self.cloneNode(right);
            }

            return new_node;
        }

        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            return self.len;
        }

        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.len == 0;
        }

        /// Time: O(log n) amortized | Space: O(1)
        /// Returns old value if key existed
        pub fn insert(self: *Self, key: K, value: V) !?V {
            // Splay the tree at the key
            self.root = self.splay(self.root, key);

            // Check if key already exists at root
            if (self.root) |root| {
                const cmp = compareFn(self.context, key, root.key);
                if (cmp == .eq) {
                    const old_value = root.value;
                    root.value = value;
                    return old_value;
                }

                // Create new node and make it root
                const new_node = try self.allocator.create(Node);
                new_node.* = .{
                    .key = key,
                    .value = value,
                };

                if (cmp == .lt) {
                    new_node.left = root.left;
                    new_node.right = root;
                    root.left = null;
                } else {
                    new_node.right = root.right;
                    new_node.left = root;
                    root.right = null;
                }

                self.root = new_node;
            } else {
                // Tree is empty
                const new_node = try self.allocator.create(Node);
                new_node.* = .{
                    .key = key,
                    .value = value,
                };
                self.root = new_node;
            }

            self.len += 1;
            return null;
        }

        /// Time: O(log n) amortized | Space: O(1)
        /// Returns removed entry if key existed
        pub fn remove(self: *Self, key: K) ?Entry {
            if (self.root == null) return null;

            // Splay at key
            self.root = self.splay(self.root, key);

            const root = self.root.?;
            const cmp = compareFn(self.context, key, root.key);
            if (cmp != .eq) {
                return null; // Key not found
            }

            const entry = Entry{
                .key = root.key,
                .value = root.value,
            };

            // Remove root
            if (root.left == null) {
                self.root = root.right;
            } else {
                // Splay the maximum in left subtree to root of left subtree
                const left = self.splay(root.left, key); // Splay with key > all in left subtree
                left.?.right = root.right;
                self.root = left;
            }

            self.allocator.destroy(root);
            self.len -= 1;
            return entry;
        }

        /// Time: O(log n) amortized | Space: O(1)
        pub fn get(self: *Self, key: K) ?V {
            if (self.root == null) return null;

            self.root = self.splay(self.root, key);
            const root = self.root.?;

            if (compareFn(self.context, key, root.key) == .eq) {
                return root.value;
            }

            return null;
        }

        /// Time: O(log n) amortized | Space: O(1)
        pub fn contains(self: *Self, key: K) bool {
            return self.get(key) != null;
        }

        /// Time: O(n) | Space: O(1)
        pub fn clear(self: *Self) void {
            self.clearNode(self.root);
            self.root = null;
            self.len = 0;
        }

        fn clearNode(self: *Self, node: ?*Node) void {
            if (node) |n| {
                self.clearNode(n.left);
                self.clearNode(n.right);
                self.allocator.destroy(n);
            }
        }

        /// Splay operation: move node with key (or closest) to root
        /// Uses top-down splaying for O(1) space
        fn splay(self: *Self, node_opt: ?*Node, key: K) ?*Node {
            var node = node_opt orelse return null;

            // Dummy nodes to simplify linking
            var left_dummy = Node{ .key = undefined, .value = undefined };
            var right_dummy = Node{ .key = undefined, .value = undefined };
            var left_max = &left_dummy; // rightmost of left tree
            var right_min = &right_dummy; // leftmost of right tree

            while (true) {
                const cmp = compareFn(self.context, key, node.key);

                if (cmp == .lt) {
                    // Key < node.key: go left
                    const left = node.left orelse break;

                    if (compareFn(self.context, key, left.key) == .lt) {
                        // Zig-zig: rotate right
                        node.left = left.right;
                        left.right = node;
                        node = left;
                        if (node.left == null) break;
                    }

                    // Link right: hang current node on right tree
                    right_min.left = node;
                    right_min = node;
                    node = node.left orelse break;
                } else if (cmp == .gt) {
                    // Key > node.key: go right
                    const right = node.right orelse break;

                    if (compareFn(self.context, key, right.key) == .gt) {
                        // Zig-zig: rotate left
                        node.right = right.left;
                        right.left = node;
                        node = right;
                        if (node.right == null) break;
                    }

                    // Link left: hang current node on left tree
                    left_max.right = node;
                    left_max = node;
                    node = node.right orelse break;
                } else {
                    // Found exact key
                    break;
                }
            }

            // Assemble: left tree + node + right tree
            left_max.right = node.left;
            right_min.left = node.right;
            node.left = left_dummy.right;
            node.right = right_dummy.left;

            return node;
        }

        /// Debug: validate BST property
        /// Time: O(n) | Space: O(log n) for recursion
        pub fn validate(self: *const Self) !void {
            var node_count: usize = 0;
            try self.validateNode(self.root, null, null, &node_count);
            if (node_count != self.len) {
                return error.InvalidCount;
            }
        }

        fn validateNode(
            self: *const Self,
            node: ?*Node,
            min_key: ?K,
            max_key: ?K,
            node_count: *usize,
        ) !void {
            const n = node orelse return;

            if (min_key) |min| {
                if (compareFn(self.context, n.key, min) != .gt) {
                    return error.BSTPropertyViolation;
                }
            }

            if (max_key) |max| {
                if (compareFn(self.context, n.key, max) != .lt) {
                    return error.BSTPropertyViolation;
                }
            }

            node_count.* += 1;
            try self.validateNode(n.left, min_key, n.key, node_count);
            try self.validateNode(n.right, n.key, max_key, node_count);
        }

        pub const Iterator = struct {
            stack: std.ArrayList(*Node),
            current: ?*Node,
            allocator: Allocator,

            pub fn next(self: *Iterator) ?Entry {
                while (self.current != null or self.stack.items.len > 0) {
                    if (self.current) |node| {
                        self.stack.append(self.allocator, node) catch return null;
                        self.current = node.left;
                    } else {
                        const node = self.stack.pop().?;
                        self.current = node.right;
                        return Entry{
                            .key = node.key,
                            .value = node.value,
                        };
                    }
                }
                return null;
            }

            pub fn deinit(self: *Iterator) void {
                self.stack.deinit(self.allocator);
            }
        };

        /// Time: O(1) to create, O(n) to iterate | Space: O(log n)
        pub fn iterator(self: *const Self) Iterator {
            return Iterator{
                .stack = .{},
                .current = self.root,
                .allocator = self.allocator,
            };
        }
    };
}

fn defaultCompare(context: void, a: i64, b: i64) std.math.Order {
    _ = context;
    return std.math.order(a, b);
}

test "SplayTree: basic insert and get" {
    var tree = SplayTree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 0), tree.count());
    try testing.expect(tree.isEmpty());

    const old = try tree.insert(10, 100);
    try testing.expectEqual(@as(?i64, null), old);
    try testing.expectEqual(@as(usize, 1), tree.count());
    try testing.expectEqual(@as(?i64, 100), tree.get(10));
}

test "SplayTree: insert duplicate updates value" {
    var tree = SplayTree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    _ = try tree.insert(5, 50);
    const old = try tree.insert(5, 500);
    try testing.expectEqual(@as(?i64, 50), old);
    try testing.expectEqual(@as(usize, 1), tree.count());
    try testing.expectEqual(@as(?i64, 500), tree.get(5));
}

test "SplayTree: multiple inserts maintain order" {
    var tree = SplayTree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    const keys = [_]i64{ 5, 2, 8, 1, 3, 7, 9 };
    for (keys) |key| {
        _ = try tree.insert(key, key * 10);
    }

    try testing.expectEqual(@as(usize, 7), tree.count());

    for (keys) |key| {
        try testing.expectEqual(@as(?i64, key * 10), tree.get(key));
    }

    try tree.validate();
}

test "SplayTree: remove existing key" {
    var tree = SplayTree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    _ = try tree.insert(10, 100);
    _ = try tree.insert(5, 50);
    _ = try tree.insert(15, 150);

    const removed = tree.remove(10);
    try testing.expect(removed != null);
    try testing.expectEqual(@as(i64, 10), removed.?.key);
    try testing.expectEqual(@as(i64, 100), removed.?.value);
    try testing.expectEqual(@as(usize, 2), tree.count());
    try testing.expectEqual(@as(?i64, null), tree.get(10));
}

test "SplayTree: remove nonexistent key" {
    var tree = SplayTree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    _ = try tree.insert(10, 100);
    const removed = tree.remove(999);
    try testing.expectEqual(@as(?SplayTree(i64, i64, void, defaultCompare).Entry, null), removed);
    try testing.expectEqual(@as(usize, 1), tree.count());
}

test "SplayTree: remove from empty tree" {
    var tree = SplayTree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    const removed = tree.remove(10);
    try testing.expectEqual(@as(?SplayTree(i64, i64, void, defaultCompare).Entry, null), removed);
}

test "SplayTree: contains" {
    var tree = SplayTree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    try testing.expect(!tree.contains(5));
    _ = try tree.insert(5, 50);
    try testing.expect(tree.contains(5));
    try testing.expect(!tree.contains(10));
}

test "SplayTree: clear" {
    var tree = SplayTree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    _ = try tree.insert(1, 10);
    _ = try tree.insert(2, 20);
    _ = try tree.insert(3, 30);

    try testing.expectEqual(@as(usize, 3), tree.count());
    tree.clear();
    try testing.expectEqual(@as(usize, 0), tree.count());
    try testing.expect(tree.isEmpty());
}

test "SplayTree: clone" {
    var tree = SplayTree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    _ = try tree.insert(10, 100);
    _ = try tree.insert(5, 50);
    _ = try tree.insert(15, 150);

    var cloned = try tree.clone();
    defer cloned.deinit();

    try testing.expectEqual(tree.count(), cloned.count());
    try testing.expectEqual(@as(?i64, 100), cloned.get(10));
    try testing.expectEqual(@as(?i64, 50), cloned.get(5));
    try testing.expectEqual(@as(?i64, 150), cloned.get(15));

    // Modify original, cloned should be unaffected
    _ = try tree.insert(20, 200);
    try testing.expectEqual(@as(?i64, 200), tree.get(20));
    try testing.expectEqual(@as(?i64, null), cloned.get(20));
}

test "SplayTree: iterator in-order traversal" {
    var tree = SplayTree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    const keys = [_]i64{ 5, 2, 8, 1, 3, 7, 9 };
    for (keys) |key| {
        _ = try tree.insert(key, key * 10);
    }

    var iter = tree.iterator();
    defer iter.deinit();

    const expected = [_]i64{ 1, 2, 3, 5, 7, 8, 9 };
    var i: usize = 0;
    while (iter.next()) |entry| {
        try testing.expectEqual(expected[i], entry.key);
        try testing.expectEqual(expected[i] * 10, entry.value);
        i += 1;
    }
    try testing.expectEqual(@as(usize, 7), i);
}

test "SplayTree: stress test with random operations" {
    var tree = SplayTree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    const num_ops = 500;
    var expected = std.AutoHashMap(i64, i64).init(testing.allocator);
    defer expected.deinit();

    for (0..num_ops) |_| {
        const key = random.intRangeAtMost(i64, 1, 100);
        const value = random.int(i64);

        _ = try tree.insert(key, value);
        try expected.put(key, value);
    }

    try tree.validate();

    // Verify all keys
    var iter_expected = expected.iterator();
    while (iter_expected.next()) |entry| {
        const tree_value = tree.get(entry.key_ptr.*);
        try testing.expectEqual(@as(?i64, entry.value_ptr.*), tree_value);
    }

    try testing.expectEqual(expected.count(), tree.count());
}

test "SplayTree: temporal locality — repeated access" {
    var tree = SplayTree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    for (1..101) |i| {
        _ = try tree.insert(@intCast(i), @intCast(i * 10));
    }

    // Access key 42 repeatedly — should splay to root
    for (0..10) |_| {
        try testing.expectEqual(@as(?i64, 420), tree.get(42));
    }

    // Root should be 42 after splaying
    try testing.expectEqual(@as(i64, 42), tree.root.?.key);

    try tree.validate();
}

test "SplayTree: remove all elements one by one" {
    var tree = SplayTree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    const keys = [_]i64{ 5, 2, 8, 1, 3, 7, 9 };
    for (keys) |key| {
        _ = try tree.insert(key, key * 10);
    }

    for (keys) |key| {
        const removed = tree.remove(key);
        try testing.expect(removed != null);
        try testing.expectEqual(key, removed.?.key);
    }

    try testing.expect(tree.isEmpty());
    try testing.expectEqual(@as(usize, 0), tree.count());
}

test "SplayTree: validate invariants after random operations" {
    var tree = SplayTree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    var prng = std.Random.DefaultPrng.init(67890);
    const random = prng.random();

    var inserted_keys = std.AutoArrayHashMap(i64, bool).init(testing.allocator);
    defer inserted_keys.deinit();

    for (0..200) |_| {
        const op = random.intRangeAtMost(u8, 0, 2);
        const key = random.intRangeAtMost(i64, 1, 50);

        if (op == 0) {
            _ = try tree.insert(key, key * 10);
            try inserted_keys.put(key, true);
        } else if (op == 1) {
            _ = tree.remove(key);
            _ = inserted_keys.remove(key);
        } else {
            const val = tree.get(key);
            if (inserted_keys.contains(key)) {
                try testing.expectEqual(key * 10, val.?);
            }
        }

        // Validate after every operation
        try tree.validate();
        try testing.expectEqual(inserted_keys.count(), tree.count());
    }
}

test "SplayTree: memory leak detection" {
    var tree = SplayTree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    for (1..51) |i| {
        _ = try tree.insert(@intCast(i), @intCast(i * 10));
    }
    try testing.expectEqual(@as(usize, 50), tree.count());

    for (1..26) |i| {
        _ = tree.remove(@intCast(i));
    }
    try testing.expectEqual(@as(usize, 25), tree.count());

    // Verify removed items are gone and remaining are intact
    for (1..26) |i| {
        try testing.expectEqual(@as(?i64, null), tree.get(@intCast(i)));
    }
    for (26..51) |i| {
        try testing.expectEqual(@as(?i64, @intCast(i * 10)), tree.get(@intCast(i)));
    }

    // Allocator will detect leaks when tree.deinit() is called
}
