const std = @import("std");

/// AVL Tree - Strictly balanced binary search tree with height-based balancing
///
/// Properties:
/// 1. For every node, the height difference between left and right subtrees is at most 1
/// 2. Balance factor = height(left) - height(right) ∈ {-1, 0, 1}
/// 3. Stricter balance than Red-Black Trees, guaranteeing O(log n) height
/// 4. Better search performance at the cost of more rotations during insert/remove
///
/// Time Complexity:
/// - insert: O(log n)
/// - remove: O(log n)
/// - search: O(log n)
/// - min/max: O(log n)
///
/// Space Complexity: O(n)
pub fn AVLTree(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: K, b: K) std.math.Order,
) type {
    return struct {
        const Self = @This();

        const Node = struct {
            key: K,
            value: V,
            height: i32, // Height of subtree rooted at this node
            parent: ?*Node,
            left: ?*Node,
            right: ?*Node,

            fn init(key: K, value: V) Node {
                return .{
                    .key = key,
                    .value = value,
                    .height = 1, // Leaf nodes have height 1
                    .parent = null,
                    .left = null,
                    .right = null,
                };
            }

            fn getHeight(node: ?*Node) i32 {
                return if (node) |n| n.height else 0;
            }

            fn updateHeight(node: *Node) void {
                const left_height = getHeight(node.left);
                const right_height = getHeight(node.right);
                node.height = 1 + @max(left_height, right_height);
            }

            fn balanceFactor(node: *Node) i32 {
                return getHeight(node.left) - getHeight(node.right);
            }
        };

        pub const Entry = struct { key: K, value: V };

        pub const Iterator = struct {
            stack: std.ArrayList(*Node),
            allocator: std.mem.Allocator,

            /// Time: O(log n) amortized | Space: O(log n)
            pub fn next(self: *Iterator) !?Entry {
                const node = self.stack.pop() orelse return null;
                const result = Entry{ .key = node.key, .value = node.value };

                // Push right subtree
                if (node.right) |right| {
                    var current: ?*Node = right;
                    while (current) |n| {
                        try self.stack.append(self.allocator, n);
                        current = n.left;
                    }
                }

                return result;
            }

            /// Frees iterator resources.
            /// Time: O(1) | Space: O(1)
            pub fn deinit(self: *Iterator) void {
                self.stack.deinit(self.allocator);
            }
        };

        allocator: std.mem.Allocator,
        root: ?*Node,
        size: usize,
        context: Context,

        // -- Lifecycle --

        /// Initializes an empty tree.
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: std.mem.Allocator, context: Context) Self {
            return .{
                .allocator = allocator,
                .root = null,
                .size = 0,
                .context = context,
            };
        }

        /// Frees all allocated memory. Invalidates all iterators.
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.root) |root| {
                self.destroySubtree(root);
            }
            self.* = undefined;
        }

        fn destroySubtree(self: *Self, node: *Node) void {
            if (node.left) |left| self.destroySubtree(left);
            if (node.right) |right| self.destroySubtree(right);
            self.allocator.destroy(node);
        }

        /// Creates a deep copy of the tree.
        /// Time: O(n log n) | Space: O(n)
        pub fn clone(self: *const Self) !Self {
            var new_tree = Self.init(self.allocator, self.context);
            errdefer new_tree.deinit();

            if (self.root) |root| {
                new_tree.root = try self.cloneSubtree(root, null);
                new_tree.size = self.size;
            }

            return new_tree;
        }

        fn cloneSubtree(self: *const Self, node: *Node, parent: ?*Node) !*Node {
            const new_node = try self.allocator.create(Node);
            new_node.* = .{
                .key = node.key,
                .value = node.value,
                .height = node.height,
                .parent = parent,
                .left = null,
                .right = null,
            };

            if (node.left) |left| {
                new_node.left = try self.cloneSubtree(left, new_node);
            }
            if (node.right) |right| {
                new_node.right = try self.cloneSubtree(right, new_node);
            }

            return new_node;
        }

        // -- Capacity --

        /// Returns number of elements.
        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            return self.size;
        }

        /// Returns true if empty.
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.size == 0;
        }

        // -- Modification --

        /// Insert a key-value pair. Returns old value if key existed.
        /// Time: O(log n) | Space: O(1)
        pub fn insert(self: *Self, key: K, value: V) !?V {
            if (self.root) |root| {
                const result = try self.insertNode(root, key, value);
                if (result.replaced) {
                    return result.old_value;
                }
                self.root = result.new_root;
                self.size += 1;
                return null;
            } else {
                const node = try self.allocator.create(Node);
                node.* = Node.init(key, value);
                self.root = node;
                self.size = 1;
                return null;
            }
        }

        const InsertResult = struct {
            new_root: *Node,
            replaced: bool,
            old_value: ?V,
        };

        fn insertNode(self: *Self, node: *Node, key: K, value: V) !InsertResult {
            const order = compareFn(self.context, key, node.key);

            switch (order) {
                .eq => {
                    const old = node.value;
                    node.value = value;
                    return .{ .new_root = node, .replaced = true, .old_value = old };
                },
                .lt => {
                    if (node.left) |left| {
                        const result = try self.insertNode(left, key, value);
                        node.left = result.new_root;
                        result.new_root.parent = node;
                        if (result.replaced) {
                            return .{ .new_root = node, .replaced = true, .old_value = result.old_value };
                        }
                    } else {
                        const new_node = try self.allocator.create(Node);
                        new_node.* = Node.init(key, value);
                        new_node.parent = node;
                        node.left = new_node;
                    }
                },
                .gt => {
                    if (node.right) |right| {
                        const result = try self.insertNode(right, key, value);
                        node.right = result.new_root;
                        result.new_root.parent = node;
                        if (result.replaced) {
                            return .{ .new_root = node, .replaced = true, .old_value = result.old_value };
                        }
                    } else {
                        const new_node = try self.allocator.create(Node);
                        new_node.* = Node.init(key, value);
                        new_node.parent = node;
                        node.right = new_node;
                    }
                },
            }

            // Update height and rebalance
            Node.updateHeight(node);
            return .{ .new_root = self.rebalance(node), .replaced = false, .old_value = null };
        }

        /// Rebalance node if balance factor is outside [-1, 1]
        /// Returns new root of subtree after rebalancing
        fn rebalance(self: *Self, node: *Node) *Node {
            const bf = Node.balanceFactor(node);

            // Left-heavy
            if (bf > 1) {
                const left = node.left.?;
                // Left-Right case: rotate left child left first
                if (Node.balanceFactor(left) < 0) {
                    node.left = self.rotateLeft(left);
                }
                // Left-Left case: rotate right
                return self.rotateRight(node);
            }

            // Right-heavy
            if (bf < -1) {
                const right = node.right.?;
                // Right-Left case: rotate right child right first
                if (Node.balanceFactor(right) > 0) {
                    node.right = self.rotateRight(right);
                }
                // Right-Right case: rotate left
                return self.rotateLeft(node);
            }

            return node;
        }

        fn rotateLeft(self: *Self, x: *Node) *Node {
            _ = self;
            const y = x.right.?;
            const t2 = y.left;

            // Perform rotation
            y.left = x;
            x.right = t2;

            // Update parents
            y.parent = x.parent;
            x.parent = y;
            if (t2) |t| t.parent = x;

            // Update heights
            Node.updateHeight(x);
            Node.updateHeight(y);

            return y;
        }

        fn rotateRight(self: *Self, y: *Node) *Node {
            _ = self;
            const x = y.left.?;
            const t2 = x.right;

            // Perform rotation
            x.right = y;
            y.left = t2;

            // Update parents
            x.parent = y.parent;
            y.parent = x;
            if (t2) |t| t.parent = y;

            // Update heights
            Node.updateHeight(y);
            Node.updateHeight(x);

            return x;
        }

        /// Remove a key. Returns the removed entry if found.
        /// Time: O(log n) | Space: O(1)
        pub fn remove(self: *Self, key: K) ?Entry {
            if (self.root) |root| {
                const result = self.removeNode(root, key);
                if (result.removed) {
                    self.root = result.new_root;
                    self.size -= 1;
                    return result.entry;
                }
            }
            return null;
        }

        const RemoveResult = struct {
            new_root: ?*Node,
            removed: bool,
            entry: ?Entry,
        };

        fn removeNode(self: *Self, node: *Node, key: K) RemoveResult {
            const order = compareFn(self.context, key, node.key);

            switch (order) {
                .lt => {
                    if (node.left) |left| {
                        const result = self.removeNode(left, key);
                        node.left = result.new_root;
                        if (result.new_root) |new_left| new_left.parent = node;
                        if (!result.removed) {
                            return .{ .new_root = node, .removed = false, .entry = null };
                        }
                        Node.updateHeight(node);
                        return .{ .new_root = self.rebalance(node), .removed = true, .entry = result.entry };
                    }
                    return .{ .new_root = node, .removed = false, .entry = null };
                },
                .gt => {
                    if (node.right) |right| {
                        const result = self.removeNode(right, key);
                        node.right = result.new_root;
                        if (result.new_root) |new_right| new_right.parent = node;
                        if (!result.removed) {
                            return .{ .new_root = node, .removed = false, .entry = null };
                        }
                        Node.updateHeight(node);
                        return .{ .new_root = self.rebalance(node), .removed = true, .entry = result.entry };
                    }
                    return .{ .new_root = node, .removed = false, .entry = null };
                },
                .eq => {
                    const entry = Entry{ .key = node.key, .value = node.value };

                    // Case 1: Leaf node
                    if (node.left == null and node.right == null) {
                        self.allocator.destroy(node);
                        return .{ .new_root = null, .removed = true, .entry = entry };
                    }

                    // Case 2: One child
                    if (node.left == null) {
                        const right = node.right.?;
                        right.parent = node.parent;
                        self.allocator.destroy(node);
                        return .{ .new_root = right, .removed = true, .entry = entry };
                    }
                    if (node.right == null) {
                        const left = node.left.?;
                        left.parent = node.parent;
                        self.allocator.destroy(node);
                        return .{ .new_root = left, .removed = true, .entry = entry };
                    }

                    // Case 3: Two children - replace with in-order successor
                    const successor = self.findMin(node.right.?);
                    node.key = successor.key;
                    node.value = successor.value;

                    const result = self.removeNode(node.right.?, successor.key);
                    node.right = result.new_root;
                    if (result.new_root) |new_right| new_right.parent = node;

                    Node.updateHeight(node);
                    return .{ .new_root = self.rebalance(node), .removed = true, .entry = entry };
                },
            }
        }

        // -- Lookup --

        /// Get value by key.
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

        /// Checks if a key exists in the tree.
        /// Time: O(log n) | Space: O(1)
        pub fn contains(self: *const Self, key: K) bool {
            return self.get(key) != null;
        }

        fn findNode(self: *const Self, key: K) ?*Node {
            var current = self.root;
            while (current) |node| {
                const order = compareFn(self.context, key, node.key);
                switch (order) {
                    .eq => return node,
                    .lt => current = node.left,
                    .gt => current = node.right,
                }
            }
            return null;
        }

        fn findMin(self: *const Self, node: *Node) *Node {
            _ = self;
            var current = node;
            while (current.left) |left| {
                current = left;
            }
            return current;
        }

        fn findMax(self: *const Self, node: *Node) *Node {
            _ = self;
            var current = node;
            while (current.right) |right| {
                current = right;
            }
            return current;
        }

        /// Get minimum entry.
        /// Time: O(log n) | Space: O(1)
        pub fn min(self: *const Self) ?Entry {
            if (self.root) |root| {
                const node = self.findMin(root);
                return Entry{ .key = node.key, .value = node.value };
            }
            return null;
        }

        /// Get maximum entry.
        /// Time: O(log n) | Space: O(1)
        pub fn max(self: *const Self) ?Entry {
            if (self.root) |root| {
                const node = self.findMax(root);
                return Entry{ .key = node.key, .value = node.value };
            }
            return null;
        }

        /// Get height of the tree (root height).
        /// Time: O(1) | Space: O(1)
        pub fn height(self: *const Self) i32 {
            return Node.getHeight(self.root);
        }

        // -- Iteration --

        /// Create an in-order iterator.
        /// Time: O(log n) init | Space: O(log n)
        pub fn iterator(self: *const Self) !Iterator {
            var stack = std.ArrayList(*Node){};
            errdefer stack.deinit(self.allocator);

            // Initialize stack with leftmost path
            var current = self.root;
            while (current) |node| {
                try stack.append(self.allocator, node);
                current = node.left;
            }

            return Iterator{
                .stack = stack,
                .allocator = self.allocator,
            };
        }

        // -- Debug --

        /// Validate AVL tree invariants.
        /// Time: O(n) | Space: O(log n)
        pub fn validate(self: *const Self) !void {
            if (self.root) |root| {
                _ = try self.validateSubtree(root);
            }
        }

        const ValidationError = error{
            InvalidHeight,
            UnbalancedNode,
            InvalidParent,
            BSTViolation,
        };

        fn validateSubtree(self: *const Self, node: *Node) ValidationError!i32 {
            const left_height = if (node.left) |left| blk: {
                // Validate parent pointer
                if (left.parent != node) return error.InvalidParent;
                // Validate BST property
                const order = compareFn(self.context, left.key, node.key);
                if (order != .lt) return error.BSTViolation;
                // Recursively validate left subtree
                break :blk try self.validateSubtree(left);
            } else 0;

            const right_height = if (node.right) |right| blk: {
                // Validate parent pointer
                if (right.parent != node) return error.InvalidParent;
                // Validate BST property
                const order = compareFn(self.context, right.key, node.key);
                if (order != .gt) return error.BSTViolation;
                // Recursively validate right subtree
                break :blk try self.validateSubtree(right);
            } else 0;

            // Validate height
            const expected_height = 1 + @max(left_height, right_height);
            if (node.height != expected_height) return error.InvalidHeight;

            // Validate balance factor
            const bf = left_height - right_height;
            if (bf < -1 or bf > 1) return error.UnbalancedNode;

            return node.height;
        }

        /// Formats tree for debugging output.
        /// Time: O(1) | Space: O(1)
        pub fn format(
            self: *const Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try writer.print("AVLTree(size={d}, height={d})", .{ self.size, self.height() });
        }
    };
}

// -- Tests --

const testing = std.testing;

fn testCompare(_: void, a: i32, b: i32) std.math.Order {
    return std.math.order(a, b);
}

test "AVLTree: basic insert and get" {
    const Tree = AVLTree(i32, []const u8, void, testCompare);
    var tree = Tree.init(testing.allocator, {});
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 0), tree.count());
    try testing.expect(tree.isEmpty());

    _ = try tree.insert(5, "five");
    _ = try tree.insert(3, "three");
    _ = try tree.insert(7, "seven");
    _ = try tree.insert(1, "one");
    _ = try tree.insert(9, "nine");

    try testing.expectEqual(@as(usize, 5), tree.count());
    try testing.expect(!tree.isEmpty());

    try testing.expectEqualStrings("five", tree.get(5).?);
    try testing.expectEqualStrings("three", tree.get(3).?);
    try testing.expectEqualStrings("seven", tree.get(7).?);
    try testing.expectEqualStrings("one", tree.get(1).?);
    try testing.expectEqualStrings("nine", tree.get(9).?);
    try testing.expect(tree.get(99) == null);
}

test "AVLTree: insert replacement" {
    const Tree = AVLTree(i32, i32, void, testCompare);
    var tree = Tree.init(testing.allocator, {});
    defer tree.deinit();

    const old1 = try tree.insert(10, 100);
    try testing.expect(old1 == null);

    const old2 = try tree.insert(10, 200);
    try testing.expectEqual(@as(i32, 100), old2.?);
    try testing.expectEqual(@as(i32, 200), tree.get(10).?);
    try testing.expectEqual(@as(usize, 1), tree.count());
}

test "AVLTree: remove" {
    const Tree = AVLTree(i32, []const u8, void, testCompare);
    var tree = Tree.init(testing.allocator, {});
    defer tree.deinit();

    _ = try tree.insert(5, "five");
    _ = try tree.insert(3, "three");
    _ = try tree.insert(7, "seven");
    _ = try tree.insert(1, "one");
    _ = try tree.insert(9, "nine");
    _ = try tree.insert(4, "four");
    _ = try tree.insert(6, "six");

    const removed = tree.remove(3);
    try testing.expect(removed != null);
    try testing.expectEqualStrings("three", removed.?.value);
    try testing.expectEqual(@as(usize, 6), tree.count());
    try testing.expect(tree.get(3) == null);

    try testing.expect(tree.remove(99) == null);
    try testing.expectEqual(@as(usize, 6), tree.count());
}

test "AVLTree: min and max" {
    const Tree = AVLTree(i32, i32, void, testCompare);
    var tree = Tree.init(testing.allocator, {});
    defer tree.deinit();

    try testing.expect(tree.min() == null);
    try testing.expect(tree.max() == null);

    _ = try tree.insert(5, 50);
    _ = try tree.insert(3, 30);
    _ = try tree.insert(7, 70);
    _ = try tree.insert(1, 10);
    _ = try tree.insert(9, 90);

    const min_entry = tree.min().?;
    try testing.expectEqual(@as(i32, 1), min_entry.key);
    try testing.expectEqual(@as(i32, 10), min_entry.value);

    const max_entry = tree.max().?;
    try testing.expectEqual(@as(i32, 9), max_entry.key);
    try testing.expectEqual(@as(i32, 90), max_entry.value);
}

test "AVLTree: iterator in-order traversal" {
    const Tree = AVLTree(i32, i32, void, testCompare);
    var tree = Tree.init(testing.allocator, {});
    defer tree.deinit();

    _ = try tree.insert(5, 50);
    _ = try tree.insert(3, 30);
    _ = try tree.insert(7, 70);
    _ = try tree.insert(1, 10);
    _ = try tree.insert(9, 90);
    _ = try tree.insert(4, 40);
    _ = try tree.insert(6, 60);

    var iter = try tree.iterator();
    defer iter.deinit();

    const expected = [_]i32{ 1, 3, 4, 5, 6, 7, 9 };
    for (expected) |key| {
        const entry = (try iter.next()).?;
        try testing.expectEqual(key, entry.key);
        try testing.expectEqual(key * 10, entry.value);
    }

    try testing.expect((try iter.next()) == null);
}

test "AVLTree: clone" {
    const Tree = AVLTree(i32, i32, void, testCompare);
    var tree = Tree.init(testing.allocator, {});
    defer tree.deinit();

    _ = try tree.insert(5, 50);
    _ = try tree.insert(3, 30);
    _ = try tree.insert(7, 70);

    var cloned = try tree.clone();
    defer cloned.deinit();

    try testing.expectEqual(tree.count(), cloned.count());
    try testing.expectEqual(tree.get(5).?, cloned.get(5).?);
    try testing.expectEqual(tree.get(3).?, cloned.get(3).?);
    try testing.expectEqual(tree.get(7).?, cloned.get(7).?);

    // Modify clone shouldn't affect original
    _ = try cloned.insert(1, 10);
    try testing.expectEqual(@as(usize, 3), tree.count());
    try testing.expectEqual(@as(usize, 4), cloned.count());
    try testing.expect(tree.get(1) == null);
    try testing.expectEqual(@as(i32, 10), cloned.get(1).?);
}

test "AVLTree: stress test with validation" {
    const Tree = AVLTree(i32, i32, void, testCompare);
    var tree = Tree.init(testing.allocator, {});
    defer tree.deinit();

    // Insert 1000 sequential elements (worst case for unbalanced BST)
    var i: i32 = 0;
    while (i < 1000) : (i += 1) {
        _ = try tree.insert(i, i * 10);
    }

    try testing.expectEqual(@as(usize, 1000), tree.count());
    try tree.validate();

    // Height should be O(log n) due to AVL balancing
    // log2(1000) ≈ 10, AVL allows at most 1.44*log2(n), so ~14
    const tree_height = tree.height();
    try testing.expect(tree_height <= 15);

    // Remove every other element
    i = 0;
    while (i < 1000) : (i += 2) {
        const removed = tree.remove(i);
        try testing.expect(removed != null);
        try testing.expectEqual(i, removed.?.key);
    }

    try testing.expectEqual(@as(usize, 500), tree.count());
    try tree.validate();

    // Verify remaining elements
    i = 1;
    while (i < 1000) : (i += 2) {
        try testing.expectEqual(i * 10, tree.get(i).?);
    }
}

test "AVLTree: height property" {
    const Tree = AVLTree(i32, i32, void, testCompare);
    var tree = Tree.init(testing.allocator, {});
    defer tree.deinit();

    try testing.expectEqual(@as(i32, 0), tree.height());

    _ = try tree.insert(1, 1);
    try testing.expectEqual(@as(i32, 1), tree.height());

    _ = try tree.insert(2, 2);
    try testing.expectEqual(@as(i32, 2), tree.height());

    _ = try tree.insert(3, 3);
    // After inserting 1,2,3, AVL rebalances: 2 becomes root, 1 and 3 are children
    // Height should be 2 (root=2, children=1,3)
    try testing.expectEqual(@as(i32, 2), tree.height());

    _ = try tree.insert(4, 4);
    _ = try tree.insert(5, 5);
    _ = try tree.insert(6, 6);
    _ = try tree.insert(7, 7);
    // With 7 nodes, height should be 3 (perfect binary tree has height ceil(log2(n+1)))
    try testing.expect(tree.height() <= 4);
    try tree.validate();
}

test "AVLTree: contains" {
    const Tree = AVLTree(i32, i32, void, testCompare);
    var tree = Tree.init(testing.allocator, {});
    defer tree.deinit();

    _ = try tree.insert(5, 50);
    _ = try tree.insert(3, 30);
    _ = try tree.insert(7, 70);

    try testing.expect(tree.contains(5));
    try testing.expect(tree.contains(3));
    try testing.expect(tree.contains(7));
    try testing.expect(!tree.contains(1));
    try testing.expect(!tree.contains(99));
}

test "AVLTree: memory leak detection" {
    const Tree = AVLTree(i32, i32, void, testCompare);
    var tree = Tree.init(testing.allocator, {});
    defer tree.deinit();

    var i: i32 = 0;
    while (i < 100) : (i += 1) {
        _ = try tree.insert(i, i * 10);
    }
    try testing.expectEqual(@as(usize, 100), tree.count());

    i = 0;
    while (i < 50) : (i += 1) {
        _ = tree.remove(i);
    }
    try testing.expectEqual(@as(usize, 50), tree.count());

    // Verify remaining values
    i = 50;
    while (i < 100) : (i += 1) {
        try testing.expectEqual(@as(?i32, i * 10), tree.get(i));
    }

    // testing.allocator will detect leaks automatically
}
