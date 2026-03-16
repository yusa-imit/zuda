const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// AA Tree: simplified self-balancing BST variant of Red-Black tree.
/// Invented by Arne Andersson (1993).
/// Simpler invariants than RB tree: right children cannot be red, left children cannot have same level.
///
/// Operations:
/// - insert, remove, get: O(log n) worst-case
/// - Simpler implementation than RB tree (only two rebalancing operations: skew and split)
/// - Uses level instead of color (level = number of black nodes from leaf to node)
pub fn AATree(
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
            level: usize = 1, // Leaf nodes have level 1

            fn isLeaf(self: *const Node) bool {
                return self.left == null and self.right == null;
            }
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
                .level = node.level,
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

        /// Time: O(log n) worst-case | Space: O(log n) for recursion
        /// Returns old value if key existed
        pub fn insert(self: *Self, key: K, value: V) !?V {
            var old_value: ?V = null;
            self.root = try self.insertNode(self.root, key, value, &old_value);
            if (old_value == null) {
                self.len += 1;
            }
            return old_value;
        }

        fn insertNode(
            self: *Self,
            node_opt: ?*Node,
            key: K,
            value: V,
            old_value: *?V,
        ) !?*Node {
            if (node_opt == null) {
                const new_node = try self.allocator.create(Node);
                new_node.* = .{
                    .key = key,
                    .value = value,
                    .level = 1,
                };
                return new_node;
            }

            var node = node_opt.?;
            const cmp = compareFn(self.context, key, node.key);

            if (cmp == .lt) {
                node.left = try self.insertNode(node.left, key, value, old_value);
            } else if (cmp == .gt) {
                node.right = try self.insertNode(node.right, key, value, old_value);
            } else {
                old_value.* = node.value;
                node.value = value;
                return node;
            }

            // Rebalance
            return split(skew(node));
        }

        /// Skew: eliminate left horizontal links (left child with same level as parent)
        /// Right rotation when node.left.level == node.level
        fn skew(node_opt: ?*Node) ?*Node {
            const node = node_opt orelse return null;
            const left = node.left orelse return node;

            if (left.level == node.level) {
                // Rotate right
                node.left = left.right;
                left.right = node;
                return left;
            }

            return node;
        }

        /// Split: eliminate consecutive horizontal links on the right
        /// Left rotation when node.right.right.level == node.level
        fn split(node_opt: ?*Node) ?*Node {
            const node = node_opt orelse return null;
            const right = node.right orelse return node;
            const right_right = right.right orelse return node;

            if (right_right.level == node.level) {
                // Rotate left
                node.right = right.left;
                right.left = node;
                right.level += 1;
                return right;
            }

            return node;
        }

        /// Time: O(log n) worst-case | Space: O(log n) for recursion
        pub fn remove(self: *Self, key: K) ?Entry {
            var removed_entry: ?Entry = null;
            self.root = self.removeNode(self.root, key, &removed_entry);
            if (removed_entry != null) {
                self.len -= 1;
            }
            return removed_entry;
        }

        fn removeNode(
            self: *Self,
            node_opt: ?*Node,
            key: K,
            removed_entry: *?Entry,
        ) ?*Node {
            var node = node_opt orelse return null;
            const cmp = compareFn(self.context, key, node.key);

            if (cmp == .lt) {
                node.left = self.removeNode(node.left, key, removed_entry);
            } else if (cmp == .gt) {
                node.right = self.removeNode(node.right, key, removed_entry);
            } else {
                // Found node to remove
                removed_entry.* = Entry{
                    .key = node.key,
                    .value = node.value,
                };

                if (node.isLeaf()) {
                    self.allocator.destroy(node);
                    return null;
                } else if (node.left == null) {
                    const right = node.right;
                    self.allocator.destroy(node);
                    return right;
                } else if (node.right == null) {
                    const left = node.left;
                    self.allocator.destroy(node);
                    return left;
                } else {
                    // Two children: replace with in-order successor
                    const successor = self.findMin(node.right.?);
                    node.key = successor.key;
                    node.value = successor.value;
                    var dummy_entry: ?Entry = null;
                    node.right = self.removeNode(node.right, successor.key, &dummy_entry);
                }
            }

            // Rebalance
            var balanced = self.decreaseLevel(node);
            balanced = skew(balanced);
            if (balanced) |b| {
                if (b.right) |right| {
                    b.right = skew(right);
                    if (right.right) |right_right| {
                        right.right = skew(right_right);
                    }
                }
            }
            balanced = split(balanced);
            if (balanced) |b| {
                if (b.right) |right| {
                    b.right = split(right);
                }
            }

            return balanced;
        }

        fn findMin(self: *const Self, node: *Node) *Node {
            _ = self;
            var current = node;
            while (current.left) |left| {
                current = left;
            }
            return current;
        }

        fn decreaseLevel(self: *Self, node_opt: ?*Node) ?*Node {
            _ = self;
            const node = node_opt orelse return null;

            const left_level = if (node.left) |left| left.level else 0;
            const right_level = if (node.right) |right| right.level else 0;

            const should_be = @min(left_level, right_level) + 1;

            if (should_be < node.level) {
                node.level = should_be;
                if (node.right) |right| {
                    if (should_be < right.level) {
                        right.level = should_be;
                    }
                }
            }

            return node;
        }

        /// Time: O(log n) worst-case | Space: O(log n) for recursion
        pub fn get(self: *const Self, key: K) ?V {
            return self.getNode(self.root, key);
        }

        fn getNode(self: *const Self, node: ?*Node, key: K) ?V {
            const n = node orelse return null;
            const cmp = compareFn(self.context, key, n.key);

            return switch (cmp) {
                .lt => self.getNode(n.left, key),
                .gt => self.getNode(n.right, key),
                .eq => n.value,
            };
        }

        /// Time: O(log n) worst-case | Space: O(log n)
        pub fn contains(self: *const Self, key: K) bool {
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

        /// Debug: validate AA tree invariants
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

            // BST property
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

            // AA tree invariants:
            // 1. Leaf nodes have level 1
            if (n.isLeaf() and n.level != 1) {
                return error.LeafLevelNotOne;
            }

            // 2. Left child level is exactly one less than parent
            if (n.left) |left| {
                if (left.level != n.level - 1) {
                    return error.LeftChildLevelInvariant;
                }
            }

            // 3. Right child level is equal to or one less than parent
            if (n.right) |right| {
                if (right.level != n.level and right.level != n.level - 1) {
                    return error.RightChildLevelInvariant;
                }

                // 4. Right grandchild level is strictly less than grandparent
                if (right.right) |right_right| {
                    if (right_right.level >= n.level) {
                        return error.RightGrandchildLevelInvariant;
                    }
                }
            }

            // 5. Every node of level > 1 has two children
            if (n.level > 1 and (n.left == null or n.right == null)) {
                return error.NonLeafMustHaveTwoChildren;
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
                        const popped_node = self.stack.pop().?;
                        self.current = popped_node.right;
                        return Entry{
                            .key = popped_node.key,
                            .value = popped_node.value,
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
                .stack = std.ArrayList(*Node){},
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

test "AATree: basic insert and get" {
    var tree = AATree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 0), tree.count());
    try testing.expect(tree.isEmpty());

    const old = try tree.insert(10, 100);
    try testing.expectEqual(@as(?i64, null), old);
    try testing.expectEqual(@as(usize, 1), tree.count());
    try testing.expectEqual(@as(?i64, 100), tree.get(10));
}

test "AATree: insert duplicate updates value" {
    var tree = AATree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    _ = try tree.insert(5, 50);
    const old = try tree.insert(5, 500);
    try testing.expectEqual(@as(?i64, 50), old);
    try testing.expectEqual(@as(usize, 1), tree.count());
    try testing.expectEqual(@as(?i64, 500), tree.get(5));
}

test "AATree: multiple inserts maintain order" {
    var tree = AATree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    const keys = [_]i64{ 5, 2, 8, 1, 3, 7, 9, 4, 6 };
    for (keys) |key| {
        _ = try tree.insert(key, key * 10);
    }

    try testing.expectEqual(@as(usize, 9), tree.count());

    for (keys) |key| {
        try testing.expectEqual(@as(?i64, key * 10), tree.get(key));
    }

    try tree.validate();
}

test "AATree: remove existing key" {
    var tree = AATree(i64, i64, void, defaultCompare).init(testing.allocator, {});
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

test "AATree: remove nonexistent key" {
    var tree = AATree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    _ = try tree.insert(10, 100);
    const removed = tree.remove(999);
    try testing.expectEqual(@as(?AATree(i64, i64, void, defaultCompare).Entry, null), removed);
    try testing.expectEqual(@as(usize, 1), tree.count());
}

test "AATree: contains" {
    var tree = AATree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    try testing.expect(!tree.contains(5));
    _ = try tree.insert(5, 50);
    try testing.expect(tree.contains(5));
    try testing.expect(!tree.contains(10));
}

test "AATree: clear" {
    var tree = AATree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    _ = try tree.insert(1, 10);
    _ = try tree.insert(2, 20);
    _ = try tree.insert(3, 30);

    try testing.expectEqual(@as(usize, 3), tree.count());
    tree.clear();
    try testing.expectEqual(@as(usize, 0), tree.count());
    try testing.expect(tree.isEmpty());
}

test "AATree: clone" {
    var tree = AATree(i64, i64, void, defaultCompare).init(testing.allocator, {});
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

    // Modify original
    _ = try tree.insert(20, 200);
    try testing.expectEqual(@as(?i64, 200), tree.get(20));
    try testing.expectEqual(@as(?i64, null), cloned.get(20));
}

test "AATree: iterator in-order traversal" {
    var tree = AATree(i64, i64, void, defaultCompare).init(testing.allocator, {});
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

test "AATree: stress test with random insertions" {
    var tree = AATree(i64, i64, void, defaultCompare).init(testing.allocator, {});
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

test "AATree: sequential insertions" {
    var tree = AATree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    // Worst case for unbalanced BST
    for (1..51) |i| {
        _ = try tree.insert(@intCast(i), @intCast(i * 10));
    }

    try tree.validate();
    try testing.expectEqual(@as(usize, 50), tree.count());

    for (1..51) |i| {
        const key: i64 = @intCast(i);
        try testing.expectEqual(@as(?i64, key * 10), tree.get(key));
    }
}

test "AATree: remove all elements one by one" {
    var tree = AATree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    const keys = [_]i64{ 5, 2, 8, 1, 3, 7, 9 };
    for (keys) |key| {
        _ = try tree.insert(key, key * 10);
    }

    for (keys) |key| {
        const removed = tree.remove(key);
        try testing.expect(removed != null);
        try testing.expectEqual(key, removed.?.key);
        try tree.validate();
    }

    try testing.expect(tree.isEmpty());
    try testing.expectEqual(@as(usize, 0), tree.count());
}

test "AATree: validate invariants after mixed operations" {
    var tree = AATree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    var prng = std.Random.DefaultPrng.init(67890);
    const random = prng.random();

    var inserted_keys = std.AutoArrayHashMap(i64, bool).init(testing.allocator);
    defer inserted_keys.deinit();

    for (0..300) |_| {
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
            } else {
                try testing.expectEqual(@as(?i64, null), val);
            }
        }

        // Validate after every operation
        try tree.validate();
        // Verify count matches expected
        try testing.expectEqual(inserted_keys.count(), tree.count());
    }
}

test "AATree: memory leak detection" {
    var tree = AATree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    for (1..101) |i| {
        _ = try tree.insert(@intCast(i), @intCast(i * 10));
    }
    try testing.expectEqual(@as(usize, 100), tree.count());

    for (1..51) |i| {
        _ = tree.remove(@intCast(i));
    }
    try testing.expectEqual(@as(usize, 50), tree.count());

    // Verify remaining values
    for (51..101) |i| {
        try testing.expectEqual(@as(?i64, @intCast(i * 10)), tree.get(@intCast(i)));
    }

    // Allocator will detect leaks when tree.deinit() is called
}

test "AATree: skew and split operations" {
    var tree = AATree(i64, i64, void, defaultCompare).init(testing.allocator, {});
    defer tree.deinit();

    // Trigger skew and split during insertions
    _ = try tree.insert(10, 100);
    _ = try tree.insert(5, 50);
    _ = try tree.insert(15, 150);
    _ = try tree.insert(3, 30);
    _ = try tree.insert(7, 70);
    _ = try tree.insert(12, 120);
    _ = try tree.insert(17, 170);

    try tree.validate();

    // Verify structure is balanced
    const root = tree.root.?;
    try testing.expect(root.level > 1);
}
