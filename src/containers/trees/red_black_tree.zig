const std = @import("std");

/// Red-Black Tree - Self-balancing binary search tree with O(log n) operations
///
/// Properties:
/// 1. Every node is either red or black
/// 2. Root is always black
/// 3. All leaves (nil) are black
/// 4. Red nodes have black children (no consecutive red nodes)
/// 5. All paths from root to leaves contain the same number of black nodes
///
/// Time Complexity:
/// - insert: O(log n)
/// - remove: O(log n)
/// - search: O(log n)
/// - min/max: O(log n)
///
/// Space Complexity: O(n)
pub fn RedBlackTree(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: K, b: K) std.math.Order,
) type {
    return struct {
        const Self = @This();

        const Color = enum { red, black };

        const Node = struct {
            key: K,
            value: V,
            color: Color,
            parent: ?*Node,
            left: ?*Node,
            right: ?*Node,

            fn init(key: K, value: V) Node {
                return .{
                    .key = key,
                    .value = value,
                    .color = .red, // New nodes are always red
                    .parent = null,
                    .left = null,
                    .right = null,
                };
            }
        };

        pub const Entry = struct { key: K, value: V };

        pub const Iterator = struct {
            stack: std.ArrayList(*Node),
            allocator: std.mem.Allocator,

            /// Time: O(log n) amortized | Space: O(log n)
            pub fn next(self: *Iterator) !?Entry {
                if (self.stack.items.len == 0) return null;

                const node = self.stack.pop().?; // pop() returns ?T in Zig 0.15
                const result = Entry{ .key = node.key, .value = node.value };

                // Push right subtree
                if (node.right) |right| {
                    var current: ?*Node = right;
                    while (current) |n| {
                        try self.stack.append(self.allocator, n); // Zig 0.15 API
                        current = n.left;
                    }
                }

                return result;
            }

            pub fn deinit(self: *Iterator) void {
                self.stack.deinit(self.allocator); // Zig 0.15 API
            }
        };

        allocator: std.mem.Allocator,
        root: ?*Node,
        size: usize,
        context: Context,

        // -- Lifecycle --

        pub fn init(allocator: std.mem.Allocator, context: Context) Self {
            return .{
                .allocator = allocator,
                .root = null,
                .size = 0,
                .context = context,
            };
        }

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
                .color = node.color,
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

        pub fn count(self: *const Self) usize {
            return self.size;
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.size == 0;
        }

        // -- Modification --

        /// Insert or update a key-value pair. Returns the old value if key existed.
        /// Time: O(log n) | Space: O(1)
        pub fn insert(self: *Self, key: K, value: V) !?V {
            if (self.root == null) {
                const node = try self.allocator.create(Node);
                node.* = Node.init(key, value);
                node.color = .black; // Root is always black
                self.root = node;
                self.size = 1;
                return null;
            }

            // Find insertion point - cache last comparison to avoid redundant compareFn call
            var current = self.root.?;
            var parent: *Node = undefined;
            var last_order: std.math.Order = undefined;
            while (true) {
                parent = current;
                const order = compareFn(self.context, key, current.key);
                last_order = order;
                switch (order) {
                    .eq => {
                        // Update existing value
                        const old = current.value;
                        current.value = value;
                        return old;
                    },
                    .lt => {
                        if (current.left) |left| {
                            current = left;
                        } else {
                            break;
                        }
                    },
                    .gt => {
                        if (current.right) |right| {
                            current = right;
                        } else {
                            break;
                        }
                    },
                }
            }

            // Insert new node - use cached comparison result
            const node = try self.allocator.create(Node);
            node.* = Node.init(key, value);
            node.parent = parent;

            if (last_order == .lt) {
                parent.left = node;
            } else {
                parent.right = node;
            }

            self.size += 1;
            self.insertFixup(node);
            return null;
        }

        fn insertFixup(self: *Self, node_param: *Node) void {
            var node = node_param;
            while (node.parent) |parent| {
                if (parent.color == .black) break;

                const grandparent = parent.parent orelse break;

                if (parent == grandparent.left) {
                    const uncle = grandparent.right;
                    if (uncle != null and uncle.?.color == .red) {
                        // Case 1: Uncle is red - recolor
                        parent.color = .black;
                        uncle.?.color = .black;
                        grandparent.color = .red;
                        node = grandparent;
                    } else {
                        // Case 2: Uncle is black
                        if (node == parent.right) {
                            // Left-right case: convert to left-left
                            node = parent;
                            self.rotateLeft(node);
                        }
                        // Left-left case: rotate right
                        const p = node.parent orelse break;
                        p.color = .black;
                        if (p.parent) |gp| {
                            gp.color = .red;
                            self.rotateRight(gp);
                        }
                    }
                } else {
                    const uncle = grandparent.left;
                    if (uncle != null and uncle.?.color == .red) {
                        // Case 1: Uncle is red - recolor
                        parent.color = .black;
                        uncle.?.color = .black;
                        grandparent.color = .red;
                        node = grandparent;
                    } else {
                        // Case 2: Uncle is black
                        if (node == parent.left) {
                            // Right-left case: convert to right-right
                            node = parent;
                            self.rotateRight(node);
                        }
                        // Right-right case: rotate left
                        const p = node.parent orelse break;
                        p.color = .black;
                        if (p.parent) |gp| {
                            gp.color = .red;
                            self.rotateLeft(gp);
                        }
                    }
                }
            }

            // Ensure root is black
            if (self.root) |root| root.color = .black;
        }

        /// Remove a key from the tree. Returns the value if found.
        /// Time: O(log n) | Space: O(1)
        pub fn remove(self: *Self, key: K) ?Entry {
            const node = self.findNode(key) orelse return null;
            const entry = Entry{ .key = node.key, .value = node.value };

            self.deleteNode(node);
            self.size -= 1;

            return entry;
        }

        fn deleteNode(self: *Self, node_param: *Node) void {
            const node = node_param;
            var original_color = node.color;
            var fixup_node: ?*Node = null;
            var fixup_parent: ?*Node = null;

            if (node.left == null) {
                fixup_node = node.right;
                fixup_parent = node.parent;
                self.transplant(node, node.right);
            } else if (node.right == null) {
                fixup_node = node.left;
                fixup_parent = node.parent;
                self.transplant(node, node.left);
            } else {
                // Node has two children - find successor
                var successor = node.right.?;
                while (successor.left) |left| {
                    successor = left;
                }

                original_color = successor.color;
                fixup_node = successor.right;

                if (successor.parent == node) {
                    if (fixup_node) |fn_node| fn_node.parent = successor;
                    fixup_parent = successor;
                } else {
                    fixup_parent = successor.parent;
                    self.transplant(successor, successor.right);
                    successor.right = node.right;
                    if (node.right) |right| right.parent = successor;
                }

                self.transplant(node, successor);
                successor.left = node.left;
                if (node.left) |left| left.parent = successor;
                successor.color = node.color;
            }

            self.allocator.destroy(node);

            if (original_color == .black) {
                self.deleteFixup(fixup_node, fixup_parent);
            }
        }

        fn deleteFixup(self: *Self, node_param: ?*Node, parent_param: ?*Node) void {
            var node = node_param;
            var parent = parent_param;

            while (node != self.root and (node == null or node.?.color == .black)) {
                const p = parent orelse break;

                if (node == p.left) {
                    var sibling = p.right orelse break;

                    if (sibling.color == .red) {
                        sibling.color = .black;
                        p.color = .red;
                        self.rotateLeft(p);
                        sibling = p.right orelse break;
                    }

                    const left_black = sibling.left == null or sibling.left.?.color == .black;
                    const right_black = sibling.right == null or sibling.right.?.color == .black;

                    if (left_black and right_black) {
                        sibling.color = .red;
                        node = p;
                        parent = p.parent;
                    } else {
                        if (right_black) {
                            if (sibling.left) |left| left.color = .black;
                            sibling.color = .red;
                            self.rotateRight(sibling);
                            sibling = p.right orelse break;
                        }

                        sibling.color = p.color;
                        p.color = .black;
                        if (sibling.right) |right| right.color = .black;
                        self.rotateLeft(p);
                        node = self.root;
                        break;
                    }
                } else {
                    var sibling = p.left orelse break;

                    if (sibling.color == .red) {
                        sibling.color = .black;
                        p.color = .red;
                        self.rotateRight(p);
                        sibling = p.left orelse break;
                    }

                    const left_black = sibling.left == null or sibling.left.?.color == .black;
                    const right_black = sibling.right == null or sibling.right.?.color == .black;

                    if (left_black and right_black) {
                        sibling.color = .red;
                        node = p;
                        parent = p.parent;
                    } else {
                        if (left_black) {
                            if (sibling.right) |right| right.color = .black;
                            sibling.color = .red;
                            self.rotateLeft(sibling);
                            sibling = p.left orelse break;
                        }

                        sibling.color = p.color;
                        p.color = .black;
                        if (sibling.left) |left| left.color = .black;
                        self.rotateRight(p);
                        node = self.root;
                        break;
                    }
                }
            }

            if (node) |n| n.color = .black;
        }

        fn transplant(self: *Self, u: *Node, v: ?*Node) void {
            if (u.parent) |parent| {
                if (u == parent.left) {
                    parent.left = v;
                } else {
                    parent.right = v;
                }
            } else {
                self.root = v;
            }

            if (v) |node| {
                node.parent = u.parent;
            }
        }

        inline fn rotateLeft(self: *Self, node: *Node) void {
            const right = node.right orelse return;

            node.right = right.left;
            if (right.left) |left| {
                left.parent = node;
            }

            right.parent = node.parent;
            if (node.parent) |parent| {
                if (node == parent.left) {
                    parent.left = right;
                } else {
                    parent.right = right;
                }
            } else {
                self.root = right;
            }

            right.left = node;
            node.parent = right;
        }

        inline fn rotateRight(self: *Self, node: *Node) void {
            const left = node.left orelse return;

            node.left = left.right;
            if (left.right) |right| {
                right.parent = node;
            }

            left.parent = node.parent;
            if (node.parent) |parent| {
                if (node == parent.right) {
                    parent.right = left;
                } else {
                    parent.left = left;
                }
            } else {
                self.root = left;
            }

            left.right = node;
            node.parent = left;
        }

        // -- Lookup --

        /// Find the value associated with a key.
        /// Time: O(log n) | Space: O(1)
        pub fn get(self: *const Self, key: K) ?V {
            const node = self.findNode(key) orelse return null;
            return node.value;
        }

        /// Check if a key exists in the tree.
        /// Time: O(log n) | Space: O(1)
        pub fn contains(self: *const Self, key: K) bool {
            return self.findNode(key) != null;
        }

        inline fn findNode(self: *const Self, key: K) ?*Node {
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

        /// Find the minimum key in the tree.
        /// Time: O(log n) | Space: O(1)
        pub fn min(self: *const Self) ?Entry {
            const node = self.minNode(self.root) orelse return null;
            return Entry{ .key = node.key, .value = node.value };
        }

        /// Find the maximum key in the tree.
        /// Time: O(log n) | Space: O(1)
        pub fn max(self: *const Self) ?Entry {
            const node = self.maxNode(self.root) orelse return null;
            return Entry{ .key = node.key, .value = node.value };
        }

        fn minNode(self: *const Self, node: ?*Node) ?*Node {
            _ = self;
            var current = node orelse return null;
            while (current.left) |left| {
                current = left;
            }
            return current;
        }

        fn maxNode(self: *const Self, node: ?*Node) ?*Node {
            _ = self;
            var current = node orelse return null;
            while (current.right) |right| {
                current = right;
            }
            return current;
        }

        // -- Iteration --

        /// Create an in-order iterator (sorted by key).
        /// Time: O(log n) for initialization | Space: O(log n)
        pub fn iterator(self: *const Self) !Iterator {
            var stack: std.ArrayList(*Node) = .{}; // Zig 0.15 API
            errdefer stack.deinit(self.allocator);

            // Initialize stack with leftmost path
            var current = self.root;
            while (current) |node| {
                try stack.append(self.allocator, node); // Zig 0.15 API
                current = node.left;
            }

            return Iterator{
                .stack = stack,
                .allocator = self.allocator,
            };
        }

        // -- Debug --

        /// Validate red-black tree invariants.
        /// Time: O(n) | Space: O(log n)
        pub fn validate(self: *const Self) !void {
            if (self.root) |root| {
                // Property 2: Root must be black
                if (root.color != .black) {
                    return error.TreeInvariant;
                }

                // Validate subtree and check black-height consistency
                _ = try self.validateSubtree(root, null, null);
            }
        }

        fn validateSubtree(
            self: *const Self,
            node: *Node,
            min_key: ?K,
            max_key: ?K,
        ) error{TreeInvariant}!usize {
            // Check BST property
            if (min_key) |min_k| {
                if (compareFn(self.context, node.key, min_k) != .gt) {
                    return error.TreeInvariant;
                }
            }
            if (max_key) |max_k| {
                if (compareFn(self.context, node.key, max_k) != .lt) {
                    return error.TreeInvariant;
                }
            }

            // Property 4: Red nodes must have black children
            if (node.color == .red) {
                if (node.left) |left| {
                    if (left.color == .red) return error.TreeInvariant;
                }
                if (node.right) |right| {
                    if (right.color == .red) return error.TreeInvariant;
                }
            }

            // Property 5: All paths have same black height
            var left_black_height: usize = 0;
            var right_black_height: usize = 0;

            if (node.left) |left| {
                if (left.parent != node) return error.TreeInvariant;
                left_black_height = try self.validateSubtree(left, min_key, node.key);
            }

            if (node.right) |right| {
                if (right.parent != node) return error.TreeInvariant;
                right_black_height = try self.validateSubtree(right, node.key, max_key);
            }

            if (left_black_height != right_black_height) {
                return error.TreeInvariant;
            }

            return left_black_height + (if (node.color == .black) @as(usize, 1) else 0);
        }

        pub fn format(
            self: *const Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try writer.print("RedBlackTree(size={d})", .{self.size});
        }
    };
}

// -- Tests --

test "RedBlackTree: basic insert and get" {
    const TestContext = struct {
        fn compare(_: @This(), a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    };

    var tree = RedBlackTree(i32, []const u8, TestContext, TestContext.compare).init(
        std.testing.allocator,
        .{},
    );
    defer tree.deinit();

    try std.testing.expectEqual(@as(usize, 0), tree.count());
    try std.testing.expect(tree.isEmpty());

    _ = try tree.insert(10, "ten");
    _ = try tree.insert(5, "five");
    _ = try tree.insert(15, "fifteen");

    try std.testing.expectEqual(@as(usize, 3), tree.count());
    try std.testing.expect(!tree.isEmpty());

    try std.testing.expectEqualStrings("ten", tree.get(10).?);
    try std.testing.expectEqualStrings("five", tree.get(5).?);
    try std.testing.expectEqualStrings("fifteen", tree.get(15).?);
    try std.testing.expect(tree.get(20) == null);
}

test "RedBlackTree: update existing key" {
    const TestContext = struct {
        fn compare(_: @This(), a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    };

    var tree = RedBlackTree(i32, i32, TestContext, TestContext.compare).init(
        std.testing.allocator,
        .{},
    );
    defer tree.deinit();

    const old1 = try tree.insert(10, 100);
    try std.testing.expect(old1 == null);

    const old2 = try tree.insert(10, 200);
    try std.testing.expectEqual(@as(i32, 100), old2.?);

    try std.testing.expectEqual(@as(i32, 200), tree.get(10).?);
    try std.testing.expectEqual(@as(usize, 1), tree.count());
}

test "RedBlackTree: remove" {
    const TestContext = struct {
        fn compare(_: @This(), a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    };

    var tree = RedBlackTree(i32, i32, TestContext, TestContext.compare).init(
        std.testing.allocator,
        .{},
    );
    defer tree.deinit();

    _ = try tree.insert(10, 100);
    _ = try tree.insert(5, 50);
    _ = try tree.insert(15, 150);
    _ = try tree.insert(3, 30);
    _ = try tree.insert(7, 70);

    try std.testing.expectEqual(@as(usize, 5), tree.count());

    const removed = tree.remove(5);
    try std.testing.expect(removed != null);
    try std.testing.expectEqual(@as(i32, 50), removed.?.value);
    try std.testing.expectEqual(@as(usize, 4), tree.count());

    try std.testing.expect(tree.get(5) == null);
    try std.testing.expect(tree.contains(3));
    try std.testing.expect(tree.contains(7));
}

test "RedBlackTree: min and max" {
    const TestContext = struct {
        fn compare(_: @This(), a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    };

    var tree = RedBlackTree(i32, i32, TestContext, TestContext.compare).init(
        std.testing.allocator,
        .{},
    );
    defer tree.deinit();

    try std.testing.expect(tree.min() == null);
    try std.testing.expect(tree.max() == null);

    _ = try tree.insert(10, 100);
    _ = try tree.insert(5, 50);
    _ = try tree.insert(15, 150);
    _ = try tree.insert(3, 30);
    _ = try tree.insert(20, 200);

    const min_entry = tree.min().?;
    try std.testing.expectEqual(@as(i32, 3), min_entry.key);

    const max_entry = tree.max().?;
    try std.testing.expectEqual(@as(i32, 20), max_entry.key);
}

test "RedBlackTree: iterator" {
    const TestContext = struct {
        fn compare(_: @This(), a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    };

    var tree = RedBlackTree(i32, i32, TestContext, TestContext.compare).init(
        std.testing.allocator,
        .{},
    );
    defer tree.deinit();

    _ = try tree.insert(10, 100);
    _ = try tree.insert(5, 50);
    _ = try tree.insert(15, 150);
    _ = try tree.insert(3, 30);
    _ = try tree.insert(7, 70);

    var iter = try tree.iterator();
    defer iter.deinit();

    const expected = [_]i32{ 3, 5, 7, 10, 15 };
    var i: usize = 0;

    while (try iter.next()) |entry| {
        try std.testing.expectEqual(expected[i], entry.key);
        i += 1;
    }

    try std.testing.expectEqual(expected.len, i);
}

test "RedBlackTree: validate invariants" {
    const TestContext = struct {
        fn compare(_: @This(), a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    };

    var tree = RedBlackTree(i32, i32, TestContext, TestContext.compare).init(
        std.testing.allocator,
        .{},
    );
    defer tree.deinit();

    // Insert many elements to trigger rotations and recoloring
    var i: i32 = 0;
    while (i < 100) : (i += 1) {
        _ = try tree.insert(i, i * 10);
        try tree.validate();
    }

    // Remove elements and validate
    i = 0;
    while (i < 50) : (i += 1) {
        _ = tree.remove(i);
        try tree.validate();
    }
}

test "RedBlackTree: stress test with random operations" {
    const TestContext = struct {
        fn compare(_: @This(), a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    };

    var tree = RedBlackTree(i32, i32, TestContext, TestContext.compare).init(
        std.testing.allocator,
        .{},
    );
    defer tree.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // Insert 1000 random elements
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        const key = random.intRangeAtMost(i32, 0, 999);
        _ = try tree.insert(key, key * 2);

        if (i % 100 == 0) {
            try tree.validate();
        }
    }

    try tree.validate();

    // Remove half of them
    i = 0;
    while (i < 500) : (i += 1) {
        const key = random.intRangeAtMost(i32, 0, 999);
        _ = tree.remove(key);

        if (i % 100 == 0) {
            try tree.validate();
        }
    }

    try tree.validate();
}

test "RedBlackTree: clone" {
    const TestContext = struct {
        fn compare(_: @This(), a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    };

    var tree = RedBlackTree(i32, i32, TestContext, TestContext.compare).init(
        std.testing.allocator,
        .{},
    );
    defer tree.deinit();

    _ = try tree.insert(10, 100);
    _ = try tree.insert(5, 50);
    _ = try tree.insert(15, 150);

    var cloned = try tree.clone();
    defer cloned.deinit();

    try std.testing.expectEqual(tree.count(), cloned.count());
    try std.testing.expectEqual(tree.get(10).?, cloned.get(10).?);
    try std.testing.expectEqual(tree.get(5).?, cloned.get(5).?);
    try std.testing.expectEqual(tree.get(15).?, cloned.get(15).?);

    try cloned.validate();
}

test "RedBlackTree: memory leak detection" {
    const TestContext = struct {
        fn compare(_: @This(), a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    };

    var tree = RedBlackTree(i32, i32, TestContext, TestContext.compare).init(
        std.testing.allocator,
        .{},
    );
    defer tree.deinit();

    var i: i32 = 0;
    while (i < 100) : (i += 1) {
        _ = try tree.insert(i, i);
    }

    // Remove all
    i = 0;
    while (i < 100) : (i += 1) {
        _ = tree.remove(i);
    }

    try std.testing.expectEqual(@as(usize, 0), tree.count());
}
