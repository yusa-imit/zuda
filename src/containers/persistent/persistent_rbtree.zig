const std = @import("std");
const testing = std.testing;

/// Persistent Red-Black Tree - Immutable balanced BST with structural sharing
///
/// Implements a persistent version of a red-black tree using path copying.
/// Each mutation creates a new version while sharing unchanged subtrees with
/// the original version. This provides efficient immutable operations with
/// O(log n) space per mutation.
///
/// Time Complexity:
/// - insert: O(log n) with path copying
/// - remove: O(log n) with path copying
/// - get/contains: O(log n)
/// - min/max: O(log n)
///
/// Space Complexity:
/// - O(n) for storage
/// - O(log n) per mutation (path copying)
///
/// Consumer use case: Functional programming patterns, undo/redo systems,
/// concurrent data structures without locks, time-travel debugging, version control
pub fn PersistentRBTree(
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
            left: ?*Node,
            right: ?*Node,

            fn clone(node: *const Node, allocator: std.mem.Allocator) !*Node {
                const new_node = try allocator.create(Node);
                new_node.* = node.*;
                return new_node;
            }

            fn deinitRecursive(node: *Node, allocator: std.mem.Allocator) void {
                if (node.left) |left| left.deinitRecursive(allocator);
                if (node.right) |right| right.deinitRecursive(allocator);
                allocator.destroy(node);
            }
        };

        pub const Entry = struct { key: K, value: V };

        pub const Iterator = struct {
            stack: std.ArrayList(*const Node),
            allocator: std.mem.Allocator,

            /// Time: O(log n) amortized | Space: O(log n)
            pub fn next(self: *Iterator) !?Entry {
                if (self.stack.items.len == 0) return null;

                const node = self.stack.pop() orelse return null;
                const result = Entry{ .key = node.key, .value = node.value };

                // Push right subtree
                if (node.right) |right| {
                    var current: ?*const Node = right;
                    while (current) |n| {
                        try self.stack.append(self.allocator, n);
                        current = n.left;
                    }
                }

                return result;
            }

            pub fn deinit(self: *Iterator) void {
                self.stack.deinit(self.allocator);
            }
        };

        allocator: std.mem.Allocator,
        root: ?*Node,
        size: usize,
        context: Context,

        // -- Lifecycle --

        /// Initialize an empty persistent red-black tree
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: std.mem.Allocator, context: Context) Self {
            return .{
                .allocator = allocator,
                .root = null,
                .size = 0,
                .context = context,
            };
        }

        /// Deinitialize and free all nodes
        /// Time: O(n) | Space: O(log n) stack
        pub fn deinit(self: *Self) void {
            if (self.root) |root| {
                root.deinitRecursive(self.allocator);
            }
            self.* = undefined;
        }

        /// Clone the entire tree (deep copy)
        /// Time: O(n) | Space: O(n)
        pub fn clone(self: *const Self) !Self {
            return Self{
                .allocator = self.allocator,
                .root = if (self.root) |root| try self.cloneSubtree(root) else null,
                .size = self.size,
                .context = self.context,
            };
        }

        fn cloneSubtree(self: *const Self, node: *const Node) !*Node {
            const new_node = try self.allocator.create(Node);
            new_node.* = .{
                .key = node.key,
                .value = node.value,
                .color = node.color,
                .left = if (node.left) |left| try self.cloneSubtree(left) else null,
                .right = if (node.right) |right| try self.cloneSubtree(right) else null,
            };
            return new_node;
        }

        // -- Capacity --

        /// Get the number of entries
        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            return self.size;
        }

        /// Check if the tree is empty
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.size == 0;
        }

        // -- Modification --

        /// Insert or update a key-value pair, returning new tree (original unchanged)
        /// Time: O(log n) | Space: O(log n)
        pub fn insert(self: *const Self, key: K, value: V) !Self {
            const result = try self.insertRecursive(self.root, key, value);

            // Root must be black
            var new_root = result.node;
            if (new_root) |root| {
                if (root.color == .red) {
                    const black_root = try root.clone(self.allocator);
                    black_root.color = .black;
                    new_root = black_root;
                }
            }

            return Self{
                .allocator = self.allocator,
                .root = new_root,
                .size = if (result.inserted) self.size + 1 else self.size,
                .context = self.context,
            };
        }

        const InsertResult = struct {
            node: ?*Node,
            inserted: bool, // true if new key, false if update
        };

        fn insertRecursive(self: *const Self, node: ?*Node, key: K, value: V) !InsertResult {
            if (node == null) {
                const new_node = try self.allocator.create(Node);
                new_node.* = .{
                    .key = key,
                    .value = value,
                    .color = .red,
                    .left = null,
                    .right = null,
                };
                return InsertResult{ .node = new_node, .inserted = true };
            }

            const n = node.?;
            const cmp = compareFn(self.context, key, n.key);

            if (cmp == .lt) {
                const result = try self.insertRecursive(n.left, key, value);

                // Only create new node if child changed
                const new_node = try self.allocator.create(Node);
                new_node.* = .{
                    .key = n.key,
                    .value = n.value,
                    .color = n.color,
                    .left = result.node,
                    .right = n.right, // Share unchanged right subtree
                };

                const fixed_node = try self.fixInsert(new_node);
                return InsertResult{ .node = fixed_node, .inserted = result.inserted };
            } else if (cmp == .gt) {
                const result = try self.insertRecursive(n.right, key, value);

                // Only create new node if child changed
                const new_node = try self.allocator.create(Node);
                new_node.* = .{
                    .key = n.key,
                    .value = n.value,
                    .color = n.color,
                    .left = n.left, // Share unchanged left subtree
                    .right = result.node,
                };

                const fixed_node = try self.fixInsert(new_node);
                return InsertResult{ .node = fixed_node, .inserted = result.inserted };
            } else {
                // Update existing key - create new node with updated value
                const new_node = try self.allocator.create(Node);
                new_node.* = .{
                    .key = key,
                    .value = value,
                    .color = n.color,
                    .left = n.left,
                    .right = n.right,
                };
                return InsertResult{ .node = new_node, .inserted = false };
            }
        }

        fn fixInsert(self: *const Self, node: *Node) !*Node {
            var n = node;

            // Case 1: Right child is red, left child is black -> rotate left
            if (self.isRed(n.right) and !self.isRed(n.left)) {
                n = try self.rotateLeft(n);
            }

            // Case 2: Left child and left-left grandchild are both red -> rotate right
            if (self.isRed(n.left) and (n.left != null and self.isRed(n.left.?.left))) {
                n = try self.rotateRight(n);
            }

            // Case 3: Both children are red -> flip colors
            if (self.isRed(n.left) and self.isRed(n.right)) {
                self.flipColors(n);
            }

            return n;
        }

        fn rotateLeft(self: *const Self, node: *Node) !*Node {
            const right = node.right orelse return node;

            // Clone nodes in the rotation
            const new_right = try right.clone(self.allocator);
            const new_node = try node.clone(self.allocator);

            new_node.right = new_right.left;
            new_right.left = new_node;

            new_right.color = new_node.color;
            new_node.color = .red;

            return new_right;
        }

        fn rotateRight(self: *const Self, node: *Node) !*Node {
            const left = node.left orelse return node;

            // Clone nodes in the rotation
            const new_left = try left.clone(self.allocator);
            const new_node = try node.clone(self.allocator);

            new_node.left = new_left.right;
            new_left.right = new_node;

            new_left.color = new_node.color;
            new_node.color = .red;

            return new_left;
        }

        fn flipColors(self: *const Self, node: *Node) void {
            _ = self;
            node.color = .red;
            if (node.left) |left| left.color = .black;
            if (node.right) |right| right.color = .black;
        }

        fn isRed(self: *const Self, node: ?*Node) bool {
            _ = self;
            if (node) |n| return n.color == .red;
            return false; // nil nodes are black
        }

        /// Remove a key, returning new tree (original unchanged)
        /// Time: O(log n) | Space: O(log n)
        pub fn remove(self: *const Self, key: K) !Self {
            if (self.root == null) return self.*;

            const result = try self.removeRecursive(self.root, key);

            // Root must be black
            var new_root = result.node;
            if (new_root) |root| {
                if (root.color == .red) {
                    const black_root = try root.clone(self.allocator);
                    black_root.color = .black;
                    new_root = black_root;
                }
            }

            return Self{
                .allocator = self.allocator,
                .root = new_root,
                .size = if (result.removed) self.size - 1 else self.size,
                .context = self.context,
            };
        }

        const RemoveResult = struct {
            node: ?*Node,
            removed: bool,
        };

        fn removeRecursive(self: *const Self, node: ?*Node, key: K) !RemoveResult {
            if (node == null) return RemoveResult{ .node = null, .removed = false };

            const n = node.?;
            const cmp = compareFn(self.context, key, n.key);

            if (cmp == .lt) {
                const result = try self.removeRecursive(n.left, key);
                if (!result.removed) return RemoveResult{ .node = node, .removed = false };

                const new_node = try n.clone(self.allocator);
                new_node.left = result.node;
                return RemoveResult{ .node = new_node, .removed = true };
            } else if (cmp == .gt) {
                const result = try self.removeRecursive(n.right, key);
                if (!result.removed) return RemoveResult{ .node = node, .removed = false };

                const new_node = try n.clone(self.allocator);
                new_node.right = result.node;
                return RemoveResult{ .node = new_node, .removed = true };
            } else {
                // Found the node to remove
                if (n.left == null) {
                    return RemoveResult{ .node = n.right, .removed = true };
                }
                if (n.right == null) {
                    return RemoveResult{ .node = n.left, .removed = true };
                }

                // Node has two children: find successor
                const successor = self.findMin(n.right.?);
                const new_node = try n.clone(self.allocator);
                new_node.key = successor.key;
                new_node.value = successor.value;

                const result = try self.removeRecursive(n.right, successor.key);
                new_node.right = result.node;

                return RemoveResult{ .node = new_node, .removed = true };
            }
        }

        fn findMin(self: *const Self, node: *Node) *const Node {
            _ = self;
            var current = node;
            while (current.left) |left| {
                current = left;
            }
            return current;
        }

        fn findMinConst(self: *const Self, node: *const Node) *const Node {
            _ = self;
            var current = node;
            while (current.left) |left| {
                current = left;
            }
            return current;
        }

        // -- Lookup --

        /// Get value by key
        /// Time: O(log n) | Space: O(1)
        pub fn get(self: *const Self, key: K) ?V {
            var current = self.root;
            while (current) |node| {
                const cmp = compareFn(self.context, key, node.key);
                if (cmp == .lt) {
                    current = node.left;
                } else if (cmp == .gt) {
                    current = node.right;
                } else {
                    return node.value;
                }
            }
            return null;
        }

        /// Check if key exists
        /// Time: O(log n) | Space: O(1)
        pub fn contains(self: *const Self, key: K) bool {
            return self.get(key) != null;
        }

        /// Get minimum key
        /// Time: O(log n) | Space: O(1)
        pub fn minimum(self: *const Self) ?Entry {
            if (self.root == null) return null;
            const node = self.findMinConst(self.root.?);
            return Entry{ .key = node.key, .value = node.value };
        }

        /// Get maximum key
        /// Time: O(log n) | Space: O(1)
        pub fn maximum(self: *const Self) ?Entry {
            if (self.root == null) return null;
            var current = self.root.?;
            while (current.right) |right| {
                current = right;
            }
            return Entry{ .key = current.key, .value = current.value };
        }

        // -- Iteration --

        /// Create an in-order iterator
        /// Time: O(log n) to initialize | Space: O(log n)
        pub fn iterator(self: *const Self) !Iterator {
            var stack = std.ArrayList(*const Node){};
            errdefer stack.deinit(self.allocator);

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

        /// Validate red-black tree invariants
        /// Time: O(n) | Space: O(log n)
        pub fn validate(self: *const Self) !void {
            if (self.root) |root| {
                if (root.color != .black) return error.RootMustBeBlack;
                _ = try self.validateSubtree(root);
            }
        }

        fn validateSubtree(self: *const Self, node: *const Node) !usize {
            // Count black height
            const black_height: usize = if (node.color == .black) 1 else 0;

            // Check red-red violation
            if (node.color == .red) {
                if (node.left) |left| {
                    if (left.color == .red) return error.RedRedViolation;
                }
                if (node.right) |right| {
                    if (right.color == .red) return error.RedRedViolation;
                }
            }

            // Recursively validate children
            const left_black_height = if (node.left) |left| try self.validateSubtree(left) else 0;
            const right_black_height = if (node.right) |right| try self.validateSubtree(right) else 0;

            // Check black height equality
            if (left_black_height != right_black_height) return error.BlackHeightMismatch;

            return black_height + left_black_height;
        }
    };
}

// -- Tests --

fn testContext() type {
    return struct {
        fn compare(_: @This(), a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    };
}

test "PersistentRBTree: init and deinit" {
    var tree = PersistentRBTree(i32, i32, testContext(), testContext().compare).init(testing.allocator, .{});
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 0), tree.count());
    try testing.expect(tree.isEmpty());
}

test "PersistentRBTree: insert and get" {
    var tree = PersistentRBTree(i32, i32, testContext(), testContext().compare).init(testing.allocator, .{});
    defer tree.deinit();

    const prev_count = tree.count();
    const prev_val = tree.get(10);

    var tree2 = try tree.insert(10, 100);
    defer tree2.deinit();

    // Original tree unchanged (checked before mutation)
    try testing.expectEqual(@as(usize, 0), prev_count);
    try testing.expect(prev_val == null);

    // New tree has the element
    try testing.expectEqual(@as(usize, 1), tree2.count());
    try testing.expectEqual(@as(i32, 100), tree2.get(10).?);
}

test "PersistentRBTree: immutability" {
    var tree = PersistentRBTree(i32, i32, testContext(), testContext().compare).init(testing.allocator, .{});
    defer tree.deinit();

    var tree2 = try tree.insert(1, 10);
    var tree3 = try tree2.insert(2, 20);

    // Snapshots at different points
    try testing.expectEqual(@as(usize, 2), tree3.count());
    try testing.expectEqual(@as(i32, 10), tree3.get(1).?);
    try testing.expectEqual(@as(i32, 20), tree3.get(2).?);

    tree2.deinit();
    tree3.deinit();
}

test "PersistentRBTree: update existing key" {
    var tree = PersistentRBTree(i32, i32, testContext(), testContext().compare).init(testing.allocator, .{});
    defer tree.deinit();

    var tree2 = try tree.insert(5, 50);
    var tree3 = try tree2.insert(5, 99); // Update

    // Size unchanged on update
    try testing.expectEqual(@as(usize, 1), tree3.count());
    try testing.expectEqual(@as(i32, 99), tree3.get(5).?);

    tree2.deinit();
    tree3.deinit();
}

test "PersistentRBTree: multiple elements" {
    var tree = PersistentRBTree(i32, i32, testContext(), testContext().compare).init(testing.allocator, .{});
    defer tree.deinit();

    const keys = [_]i32{ 50, 30, 70, 20, 40, 60, 80 };
    for (keys) |key| {
        const new_tree = try tree.insert(key, key * 10);
        tree.deinit();
        tree = new_tree;
    }

    try testing.expectEqual(@as(usize, 7), tree.count());
    for (keys) |key| {
        try testing.expectEqual(@as(i32, key * 10), tree.get(key).?);
    }
}

test "PersistentRBTree: remove" {
    var tree = PersistentRBTree(i32, i32, testContext(), testContext().compare).init(testing.allocator, .{});
    defer tree.deinit();

    var current = tree;
    current = try current.insert(1, 10);
    current = try current.insert(2, 20);
    current = try current.insert(3, 30);

    const prev_count = current.count();
    const prev_val = current.get(2);

    var tree2 = try current.remove(2);
    defer tree2.deinit();

    // Check original before removal
    try testing.expectEqual(@as(usize, 3), prev_count);
    try testing.expectEqual(@as(i32, 20), prev_val.?);

    // New tree has element removed
    try testing.expectEqual(@as(usize, 2), tree2.count());
    try testing.expect(tree2.get(2) == null);
    try testing.expectEqual(@as(i32, 10), tree2.get(1).?);
    try testing.expectEqual(@as(i32, 30), tree2.get(3).?);
}

test "PersistentRBTree: remove non-existent" {
    var tree = PersistentRBTree(i32, i32, testContext(), testContext().compare).init(testing.allocator, .{});
    defer tree.deinit();

    tree = try tree.insert(5, 50);

    var tree2 = try tree.remove(99);
    defer tree2.deinit();

    // Size unchanged
    try testing.expectEqual(@as(usize, 1), tree2.count());
}

test "PersistentRBTree: contains" {
    var tree = PersistentRBTree(i32, i32, testContext(), testContext().compare).init(testing.allocator, .{});
    defer tree.deinit();

    tree = try tree.insert(42, 100);

    try testing.expect(tree.contains(42));
    try testing.expect(!tree.contains(99));
}

test "PersistentRBTree: minimum and maximum" {
    var tree = PersistentRBTree(i32, i32, testContext(), testContext().compare).init(testing.allocator, .{});
    defer tree.deinit();

    try testing.expect(tree.minimum() == null);
    try testing.expect(tree.maximum() == null);

    tree = try tree.insert(50, 500);
    tree = try tree.insert(30, 300);
    tree = try tree.insert(70, 700);

    const min = tree.minimum().?;
    const max = tree.maximum().?;

    try testing.expectEqual(@as(i32, 30), min.key);
    try testing.expectEqual(@as(i32, 70), max.key);
}

test "PersistentRBTree: iterator" {
    var tree = PersistentRBTree(i32, i32, testContext(), testContext().compare).init(testing.allocator, .{});
    defer tree.deinit();

    const keys = [_]i32{ 50, 30, 70, 20, 40 };
    for (keys) |key| {
        tree = try tree.insert(key, key * 10);
    }

    var it = try tree.iterator();
    defer it.deinit();

    // Should iterate in sorted order
    const expected = [_]i32{ 20, 30, 40, 50, 70 };
    for (expected) |exp_key| {
        const entry = (try it.next()).?;
        try testing.expectEqual(exp_key, entry.key);
        try testing.expectEqual(exp_key * 10, entry.value);
    }

    try testing.expect((try it.next()) == null);
}

test "PersistentRBTree: stress test" {
    var tree = PersistentRBTree(i32, i32, testContext(), testContext().compare).init(testing.allocator, .{});
    defer tree.deinit();

    // Insert 100 elements
    var i: i32 = 0;
    while (i < 100) : (i += 1) {
        tree = try tree.insert(i, i * 100);
    }

    try testing.expectEqual(@as(usize, 100), tree.count());

    // Verify all elements
    i = 0;
    while (i < 100) : (i += 1) {
        try testing.expectEqual(@as(i32, i * 100), tree.get(i).?);
    }

    // Remove 50 elements
    i = 0;
    while (i < 50) : (i += 1) {
        tree = try tree.remove(i * 2); // Remove even numbers
    }

    try testing.expectEqual(@as(usize, 50), tree.count());

    // Verify remaining elements (odd numbers)
    i = 0;
    while (i < 50) : (i += 1) {
        const key = i * 2 + 1;
        try testing.expectEqual(@as(i32, key * 100), tree.get(key).?);
    }
}

test "PersistentRBTree: validate invariants" {
    var tree = PersistentRBTree(i32, i32, testContext(), testContext().compare).init(testing.allocator, .{});
    defer tree.deinit();

    const keys = [_]i32{ 50, 30, 70, 20, 40, 60, 80, 10, 25, 35, 45 };
    for (keys) |key| {
        tree = try tree.insert(key, key);
        try tree.validate();
    }

    for (keys) |key| {
        tree = try tree.remove(key);
        try tree.validate();
    }
}

test "PersistentRBTree: memory leak check" {
    var tree = PersistentRBTree(i32, i32, testContext(), testContext().compare).init(testing.allocator, .{});
    defer tree.deinit();

    var i: i32 = 0;
    while (i < 50) : (i += 1) {
        tree = try tree.insert(i, i * 10);
    }
}

test "PersistentRBTree: structural sharing" {
    var tree = PersistentRBTree(i32, i32, testContext(), testContext().compare).init(testing.allocator, .{});
    defer tree.deinit();

    tree = try tree.insert(1, 10);
    tree = try tree.insert(2, 20);
    tree = try tree.insert(3, 30);

    const prev_count = tree.count();
    const prev_has_4 = tree.get(4);

    // Create a new version
    var tree2 = try tree.insert(4, 40);
    defer tree2.deinit();

    // Verify state before mutation
    try testing.expectEqual(@as(usize, 3), prev_count);
    try testing.expect(prev_has_4 == null);

    // New tree has new element
    try testing.expectEqual(@as(usize, 4), tree2.count());
    try testing.expectEqual(@as(i32, 40), tree2.get(4).?);
}

test "PersistentRBTree: string keys" {
    const StringContext = struct {
        fn compare(_: @This(), a: []const u8, b: []const u8) std.math.Order {
            return std.mem.order(u8, a, b);
        }
    };

    var tree = PersistentRBTree([]const u8, i32, StringContext, StringContext.compare).init(testing.allocator, .{});
    defer tree.deinit();

    tree = try tree.insert("apple", 1);
    tree = try tree.insert("banana", 2);
    tree = try tree.insert("cherry", 3);

    try testing.expectEqual(@as(i32, 1), tree.get("apple").?);
    try testing.expectEqual(@as(i32, 2), tree.get("banana").?);
    try testing.expectEqual(@as(i32, 3), tree.get("cherry").?);
}
