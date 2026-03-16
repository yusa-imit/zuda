//! BTree - Self-balancing search tree with variable branching factor.
//!
//! A B-tree of order M maintains nodes with between ⌈M/2⌉ and M children,
//! ensuring logarithmic depth and efficient disk/cache access through high
//! branching factor. All leaves are at the same depth.
//!
//! This is an in-memory general-purpose B-tree, not a specialized database
//! implementation. For database-specific B+trees with page layouts and overflow
//! handling, see consumer projects like silica.
//!
//! Time complexity:
//!   - Insert: O(log n)
//!   - Remove: O(log n)
//!   - Get: O(log n)
//!   - Range iteration: O(log n + k) where k is result count
//!
//! Space complexity: O(n)

const std = @import("std");

/// BTree with compile-time order and context-based comparison.
///
/// Order M determines:
///   - Max keys per node: M - 1
///   - Min keys per node (non-root): ⌈M/2⌉ - 1
///   - Max children: M
///   - Min children (non-root): ⌈M/2⌉
///
/// Common orders:
///   - M=4: 2-3-4 tree (min 1 key, max 3 keys)
///   - M=128: Good cache performance for in-memory use
///
/// Context provides comparison function:
///   fn compare(ctx: Context, a: K, b: K) std.math.Order
pub fn BTree(
    comptime K: type,
    comptime V: type,
    comptime order: comptime_int,
    comptime Context: type,
) type {
    if (order < 3) @compileError("BTree order must be at least 3");

    return struct {
        const Self = @This();

        pub const Order = order;
        pub const MaxKeys = order - 1;
        pub const MinKeys = (order + 1) / 2 - 1; // ⌈M/2⌉ - 1

        pub const Entry = struct {
            key: K,
            value: V,
        };

        const Node = struct {
            keys: [MaxKeys]K = undefined,
            values: [MaxKeys]V = undefined,
            children: [order]?*Node = [_]?*Node{null} ** order,
            num_keys: usize = 0,
            is_leaf: bool = true,

            fn isFull(self: *const Node) bool {
                return self.num_keys == MaxKeys;
            }

            fn isMinimal(self: *const Node) bool {
                return self.num_keys == MinKeys;
            }

            fn keyAt(self: *const Node, idx: usize) K {
                std.debug.assert(idx < self.num_keys);
                return self.keys[idx];
            }

            fn valueAt(self: *const Node, idx: usize) V {
                std.debug.assert(idx < self.num_keys);
                return self.values[idx];
            }

            fn childAt(self: *const Node, idx: usize) ?*Node {
                std.debug.assert(idx <= self.num_keys);
                return self.children[idx];
            }
        };

        allocator: std.mem.Allocator,
        context: Context,
        root: ?*Node,
        len: usize,

        // -- Lifecycle --

        /// Initialize an empty B-tree.
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: std.mem.Allocator, context: Context) Self {
            return .{
                .allocator = allocator,
                .context = context,
                .root = null,
                .len = 0,
            };
        }

        /// Free all nodes and reset to empty state.
        /// Time: O(n) | Space: O(h) where h is height
        pub fn deinit(self: *Self) void {
            if (self.root) |root| {
                self.freeNode(root);
                self.root = null;
            }
            self.len = 0;
        }

        fn freeNode(self: *Self, node: *Node) void {
            if (!node.is_leaf) {
                for (node.children[0 .. node.num_keys + 1]) |maybe_child| {
                    if (maybe_child) |child| {
                        self.freeNode(child);
                    }
                }
            }
            self.allocator.destroy(node);
        }

        fn createNode(self: *Self, is_leaf: bool) !*Node {
            const node = try self.allocator.create(Node);
            node.* = .{
                .is_leaf = is_leaf,
            };
            return node;
        }

        // -- Capacity --

        /// Returns the number of key-value pairs in the tree.
        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            return self.len;
        }

        /// Checks if the tree is empty.
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.len == 0;
        }

        // -- Modification --

        /// Insert a key-value pair. Returns the old value if key existed.
        /// Time: O(log n) | Space: O(h) stack where h is height
        pub fn insert(self: *Self, key: K, value: V) !?V {
            if (self.root == null) {
                self.root = try self.createNode(true);
            }

            const root = self.root.?;

            // If root is full, split it and create new root
            if (root.isFull()) {
                const new_root = try self.createNode(false);
                new_root.children[0] = root;
                try self.splitChild(new_root, 0);
                self.root = new_root;
            }

            return self.insertNonFull(self.root.?, key, value);
        }

        fn insertNonFull(self: *Self, node: *Node, key: K, value: V) !?V {
            var idx: usize = node.num_keys;

            // Find insertion point
            while (idx > 0) : (idx -= 1) {
                const cmp = self.context.compare(key, node.keyAt(idx - 1));
                if (cmp == .gt) break;
                if (cmp == .eq) {
                    // Key exists, replace value
                    const old = node.valueAt(idx - 1);
                    node.values[idx - 1] = value;
                    return old;
                }
            }

            if (node.is_leaf) {
                // Shift keys and values to make room
                if (idx < node.num_keys) {
                    std.mem.copyBackwards(K, node.keys[idx + 1 .. node.num_keys + 1], node.keys[idx..node.num_keys]);
                    std.mem.copyBackwards(V, node.values[idx + 1 .. node.num_keys + 1], node.values[idx..node.num_keys]);
                }
                node.keys[idx] = key;
                node.values[idx] = value;
                node.num_keys += 1;
                self.len += 1;
                return null;
            } else {
                // Internal node: recurse to child
                var child = node.childAt(idx).?;
                if (child.isFull()) {
                    try self.splitChild(node, idx);
                    // After split, determine which child to use
                    const cmp = self.context.compare(key, node.keyAt(idx));
                    if (cmp == .gt) {
                        idx += 1;
                        child = node.childAt(idx).?;
                    } else if (cmp == .eq) {
                        // Key is the median that was pushed up
                        const old = node.valueAt(idx);
                        node.values[idx] = value;
                        return old;
                    }
                    // else: cmp == .lt, use same child
                }
                return self.insertNonFull(child, key, value);
            }
        }

        fn splitChild(self: *Self, parent: *Node, child_idx: usize) !void {
            const full_child = parent.childAt(child_idx).?;
            const new_child = try self.createNode(full_child.is_leaf);

            const mid = MaxKeys / 2;

            // Save median key/value before modifying full_child
            const median_key = full_child.keys[mid];
            const median_value = full_child.values[mid];

            // Copy second half of keys/values to new child
            @memcpy(new_child.keys[0 .. MaxKeys - mid - 1], full_child.keys[mid + 1 .. MaxKeys]);
            @memcpy(new_child.values[0 .. MaxKeys - mid - 1], full_child.values[mid + 1 .. MaxKeys]);
            new_child.num_keys = MaxKeys - mid - 1;

            // Copy second half of children if internal
            if (!full_child.is_leaf) {
                @memcpy(new_child.children[0 .. MaxKeys - mid], full_child.children[mid + 1 .. order]);
            }

            full_child.num_keys = mid;

            // Insert median into parent
            if (child_idx < parent.num_keys) {
                std.mem.copyBackwards(K, parent.keys[child_idx + 1 .. parent.num_keys + 1], parent.keys[child_idx..parent.num_keys]);
                std.mem.copyBackwards(V, parent.values[child_idx + 1 .. parent.num_keys + 1], parent.values[child_idx..parent.num_keys]);
                std.mem.copyBackwards(?*Node, parent.children[child_idx + 2 .. parent.num_keys + 2], parent.children[child_idx + 1 .. parent.num_keys + 1]);
            }

            parent.keys[child_idx] = median_key;
            parent.values[child_idx] = median_value;
            parent.children[child_idx + 1] = new_child;
            parent.num_keys += 1;
        }

        /// Remove a key-value pair. Returns the value if key existed.
        /// Time: O(log n) | Space: O(h) stack
        pub fn remove(self: *Self, key: K) ?Entry {
            if (self.root == null) return null;

            const result = self.removeFromNode(self.root.?, key);
            if (result != null) {
                self.len -= 1;

                // If root is now empty, make its only child the new root
                if (self.root.?.num_keys == 0) {
                    const old_root = self.root.?;
                    if (!old_root.is_leaf) {
                        self.root = old_root.childAt(0);
                    } else {
                        self.root = null;
                    }
                    self.allocator.destroy(old_root);
                }
            }
            return result;
        }

        fn removeFromNode(self: *Self, node: *Node, key: K) ?Entry {
            var idx: usize = 0;
            while (idx < node.num_keys and self.context.compare(key, node.keyAt(idx)) == .gt) : (idx += 1) {}

            if (idx < node.num_keys and self.context.compare(key, node.keyAt(idx)) == .eq) {
                // Key found in this node
                if (node.is_leaf) {
                    return self.removeFromLeaf(node, idx);
                } else {
                    return self.removeFromInternal(node, idx);
                }
            } else if (!node.is_leaf) {
                // Key might be in subtree
                const is_in_last_child = (idx == node.num_keys);
                const child = node.childAt(idx).?;

                if (child.num_keys < MinKeys + 1) {
                    self.fixChildMinimal(node, idx) catch return null;
                    // After fix, re-find position
                    idx = 0;
                    while (idx < node.num_keys and self.context.compare(key, node.keyAt(idx)) == .gt) : (idx += 1) {}
                    if (idx < node.num_keys and self.context.compare(key, node.keyAt(idx)) == .eq) {
                        if (node.is_leaf) {
                            return self.removeFromLeaf(node, idx);
                        } else {
                            return self.removeFromInternal(node, idx);
                        }
                    }
                }

                if (is_in_last_child and idx > node.num_keys) {
                    idx = node.num_keys;
                }

                if (node.childAt(idx)) |c| {
                    return self.removeFromNode(c, key);
                }
            }
            return null;
        }

        fn removeFromLeaf(_: *Self, node: *Node, idx: usize) Entry {
            const entry = Entry{ .key = node.keyAt(idx), .value = node.valueAt(idx) };
            if (idx < node.num_keys - 1) {
                std.mem.copyForwards(K, node.keys[idx .. node.num_keys - 1], node.keys[idx + 1 .. node.num_keys]);
                std.mem.copyForwards(V, node.values[idx .. node.num_keys - 1], node.values[idx + 1 .. node.num_keys]);
            }
            node.num_keys -= 1;
            return entry;
        }

        fn removeFromInternal(self: *Self, node: *Node, idx: usize) ?Entry {
            const key = node.keyAt(idx);
            const left_child = node.childAt(idx).?;
            const right_child = node.childAt(idx + 1).?;

            if (left_child.num_keys >= MinKeys + 1) {
                // Get predecessor from left subtree
                const pred = self.getPredecessor(left_child);
                const old_value = node.valueAt(idx);
                node.keys[idx] = pred.key;
                node.values[idx] = pred.value;
                _ = self.removeFromNode(left_child, pred.key);
                self.len += 1; // Compensate since we'll decrement in main remove
                return Entry{ .key = key, .value = old_value };
            } else if (right_child.num_keys >= MinKeys + 1) {
                // Get successor from right subtree
                const succ = self.getSuccessor(right_child);
                const old_value = node.valueAt(idx);
                node.keys[idx] = succ.key;
                node.values[idx] = succ.value;
                _ = self.removeFromNode(right_child, succ.key);
                self.len += 1; // Compensate
                return Entry{ .key = key, .value = old_value };
            } else {
                // Merge children and recurse
                self.mergeChildren(node, idx) catch return null;
                return self.removeFromNode(left_child, key);
            }
        }

        fn getPredecessor(_: *Self, node: *Node) Entry {
            var current = node;
            while (!current.is_leaf) {
                current = current.childAt(current.num_keys).?;
            }
            return Entry{ .key = current.keyAt(current.num_keys - 1), .value = current.valueAt(current.num_keys - 1) };
        }

        fn getSuccessor(_: *Self, node: *Node) Entry {
            var current = node;
            while (!current.is_leaf) {
                current = current.childAt(0).?;
            }
            return Entry{ .key = current.keyAt(0), .value = current.valueAt(0) };
        }

        fn fixChildMinimal(self: *Self, parent: *Node, child_idx: usize) !void {
            if (child_idx > 0 and parent.childAt(child_idx - 1).?.num_keys >= MinKeys + 1) {
                try self.borrowFromLeft(parent, child_idx);
            } else if (child_idx < parent.num_keys and parent.childAt(child_idx + 1).?.num_keys >= MinKeys + 1) {
                try self.borrowFromRight(parent, child_idx);
            } else {
                if (child_idx < parent.num_keys) {
                    try self.mergeChildren(parent, child_idx);
                } else {
                    try self.mergeChildren(parent, child_idx - 1);
                }
            }
        }

        fn borrowFromLeft(_: *Self, parent: *Node, child_idx: usize) !void {
            const child = parent.childAt(child_idx).?;
            const left_sibling = parent.childAt(child_idx - 1).?;

            // Shift child's keys/values/children right
            std.mem.copyBackwards(K, child.keys[1 .. child.num_keys + 1], child.keys[0..child.num_keys]);
            std.mem.copyBackwards(V, child.values[1 .. child.num_keys + 1], child.values[0..child.num_keys]);
            if (!child.is_leaf) {
                std.mem.copyBackwards(?*Node, child.children[1 .. child.num_keys + 2], child.children[0 .. child.num_keys + 1]);
            }

            // Move parent key down to child
            child.keys[0] = parent.keyAt(child_idx - 1);
            child.values[0] = parent.valueAt(child_idx - 1);

            // Move left sibling's last key up to parent
            parent.keys[child_idx - 1] = left_sibling.keyAt(left_sibling.num_keys - 1);
            parent.values[child_idx - 1] = left_sibling.valueAt(left_sibling.num_keys - 1);

            // Move left sibling's last child to child
            if (!child.is_leaf) {
                child.children[0] = left_sibling.childAt(left_sibling.num_keys);
            }

            child.num_keys += 1;
            left_sibling.num_keys -= 1;
        }

        fn borrowFromRight(_: *Self, parent: *Node, child_idx: usize) !void {
            const child = parent.childAt(child_idx).?;
            const right_sibling = parent.childAt(child_idx + 1).?;

            // Move parent key down to child
            child.keys[child.num_keys] = parent.keyAt(child_idx);
            child.values[child.num_keys] = parent.valueAt(child_idx);

            // Move right sibling's first key up to parent
            parent.keys[child_idx] = right_sibling.keyAt(0);
            parent.values[child_idx] = right_sibling.valueAt(0);

            // Move right sibling's first child to child
            if (!child.is_leaf) {
                child.children[child.num_keys + 1] = right_sibling.childAt(0);
            }

            child.num_keys += 1;

            // Shift right sibling's keys/values/children left
            std.mem.copyForwards(K, right_sibling.keys[0 .. right_sibling.num_keys - 1], right_sibling.keys[1..right_sibling.num_keys]);
            std.mem.copyForwards(V, right_sibling.values[0 .. right_sibling.num_keys - 1], right_sibling.values[1..right_sibling.num_keys]);
            if (!right_sibling.is_leaf) {
                std.mem.copyForwards(?*Node, right_sibling.children[0..right_sibling.num_keys], right_sibling.children[1 .. right_sibling.num_keys + 1]);
            }

            right_sibling.num_keys -= 1;
        }

        fn mergeChildren(self: *Self, parent: *Node, idx: usize) !void {
            const left_child = parent.childAt(idx).?;
            const right_child = parent.childAt(idx + 1).?;

            // Pull parent key down to left child
            left_child.keys[left_child.num_keys] = parent.keyAt(idx);
            left_child.values[left_child.num_keys] = parent.valueAt(idx);
            left_child.num_keys += 1;

            // Copy right child's keys/values to left child
            @memcpy(left_child.keys[left_child.num_keys..][0..right_child.num_keys], right_child.keys[0..right_child.num_keys]);
            @memcpy(left_child.values[left_child.num_keys..][0..right_child.num_keys], right_child.values[0..right_child.num_keys]);

            // Copy right child's children if internal
            if (!left_child.is_leaf) {
                @memcpy(left_child.children[left_child.num_keys..][0 .. right_child.num_keys + 1], right_child.children[0 .. right_child.num_keys + 1]);
            }

            left_child.num_keys += right_child.num_keys;

            // Remove parent key and right child pointer
            if (idx < parent.num_keys - 1) {
                std.mem.copyForwards(K, parent.keys[idx .. parent.num_keys - 1], parent.keys[idx + 1 .. parent.num_keys]);
                std.mem.copyForwards(V, parent.values[idx .. parent.num_keys - 1], parent.values[idx + 1 .. parent.num_keys]);
                std.mem.copyForwards(?*Node, parent.children[idx + 1 .. parent.num_keys], parent.children[idx + 2 .. parent.num_keys + 1]);
            }
            parent.num_keys -= 1;

            self.allocator.destroy(right_child);
        }

        // -- Lookup --

        /// Get the value associated with a key.
        /// Time: O(log n) | Space: O(1)
        pub fn get(self: *const Self, key: K) ?V {
            if (self.root == null) return null;
            return self.getFromNode(self.root.?, key);
        }

        fn getFromNode(self: *const Self, node: *const Node, key: K) ?V {
            var idx: usize = 0;
            while (idx < node.num_keys) : (idx += 1) {
                const cmp = self.context.compare(key, node.keyAt(idx));
                if (cmp == .eq) return node.valueAt(idx);
                if (cmp == .lt) break;
            }

            if (node.is_leaf) return null;
            if (node.childAt(idx)) |child| {
                return self.getFromNode(child, key);
            }
            return null;
        }

        /// Check if a key exists in the tree.
        /// Time: O(log n) | Space: O(1)
        pub fn contains(self: *const Self, key: K) bool {
            return self.get(key) != null;
        }

        // -- Iteration --

        pub const Iterator = struct {
            tree: *const Self,
            stack: std.ArrayList(*const Node),
            indices: std.ArrayList(usize),
            allocator: std.mem.Allocator,

            pub fn next(self: *Iterator) ?Entry {
                while (self.stack.items.len > 0) {
                    const node = self.stack.items[self.stack.items.len - 1];
                    const idx_ptr = &self.indices.items[self.indices.items.len - 1];

                    // For internal nodes, we interleave children and keys
                    // Order: child[0], key[0], child[1], key[1], ..., child[n]
                    if (!node.is_leaf) {
                        // idx represents position in the interleaved sequence
                        if (idx_ptr.* <= node.num_keys * 2) {
                            if (idx_ptr.* % 2 == 0) {
                                // Even index: descend to child
                                const child_idx = idx_ptr.* / 2;
                                if (node.childAt(child_idx)) |child| {
                                    idx_ptr.* += 1;
                                    self.stack.append(self.allocator, child) catch return null;
                                    self.indices.append(self.allocator, 0) catch return null;
                                    continue;
                                } else {
                                    idx_ptr.* += 1;
                                    continue;
                                }
                            } else {
                                // Odd index: return key
                                const key_idx = idx_ptr.* / 2;
                                idx_ptr.* += 1;
                                if (key_idx < node.num_keys) {
                                    return Entry{ .key = node.keyAt(key_idx), .value = node.valueAt(key_idx) };
                                }
                            }
                        }
                    } else {
                        // Leaf node: just iterate through keys
                        if (idx_ptr.* < node.num_keys) {
                            const key_idx = idx_ptr.*;
                            idx_ptr.* += 1;
                            return Entry{ .key = node.keyAt(key_idx), .value = node.valueAt(key_idx) };
                        }
                    }

                    // Done with this node, pop
                    _ = self.stack.pop();
                    _ = self.indices.pop();
                }
                return null;
            }

            pub fn deinit(self: *Iterator) void {
                self.stack.deinit(self.allocator);
                self.indices.deinit(self.allocator);
            }
        };

        /// Create an iterator over all entries in sorted order.
        /// Time: O(1) to create | Space: O(h) for iterator stack
        pub fn iterator(self: *const Self) !Iterator {
            var stack = std.ArrayList(*const Node){};
            errdefer stack.deinit(self.allocator);
            var indices = std.ArrayList(usize){};
            errdefer indices.deinit(self.allocator);

            if (self.root) |root| {
                try stack.append(self.allocator, root);
                try indices.append(self.allocator, 0);
            }

            return Iterator{
                .tree = self,
                .stack = stack,
                .indices = indices,
                .allocator = self.allocator,
            };
        }

        // -- Debug --

        /// Validate B-tree invariants. Returns error if tree is corrupted.
        /// Time: O(n) | Space: O(h) stack
        pub fn validate(self: *const Self) !void {
            if (self.root == null) {
                if (self.len != 0) return error.TreeInvariant;
                return;
            }

            var counted: usize = 0;
            try self.validateNode(self.root.?, null, null, &counted, true);

            if (counted != self.len) return error.TreeInvariant;
        }

        fn validateNode(
            self: *const Self,
            node: *const Node,
            min_key: ?K,
            max_key: ?K,
            counted: *usize,
            is_root: bool,
        ) !void {
            // Check key count bounds
            if (!is_root and node.num_keys < MinKeys) return error.TreeInvariant;
            if (node.num_keys > MaxKeys) return error.TreeInvariant;

            // Check keys are sorted and in range
            for (0..node.num_keys) |i| {
                const key = node.keyAt(i);

                if (min_key) |min| {
                    if (self.context.compare(key, min) != .gt) return error.TreeInvariant;
                }
                if (max_key) |max| {
                    if (self.context.compare(key, max) != .lt) return error.TreeInvariant;
                }

                if (i > 0) {
                    if (self.context.compare(key, node.keyAt(i - 1)) != .gt) return error.TreeInvariant;
                }

                counted.* += 1;
            }

            // Recursively validate children
            if (!node.is_leaf) {
                for (0..node.num_keys + 1) |i| {
                    const child = node.childAt(i) orelse return error.TreeInvariant;

                    const child_min = if (i > 0) node.keyAt(i - 1) else min_key;
                    const child_max = if (i < node.num_keys) node.keyAt(i) else max_key;

                    try self.validateNode(child, child_min, child_max, counted, false);
                }
            }
        }
    };
}

// -- Tests --

const testing = std.testing;

const TestContext = struct {
    pub fn compare(_: @This(), a: i32, b: i32) std.math.Order {
        return std.math.order(a, b);
    }
};

test "BTree: basic insert and get" {
    const Tree = BTree(i32, []const u8, 4, TestContext);
    var tree = Tree.init(testing.allocator, .{});
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 0), tree.count());
    try testing.expect(tree.isEmpty());

    _ = try tree.insert(10, "ten");
    _ = try tree.insert(20, "twenty");
    _ = try tree.insert(5, "five");

    try testing.expectEqual(@as(usize, 3), tree.count());
    try testing.expectEqualStrings("ten", tree.get(10).?);
    try testing.expectEqualStrings("twenty", tree.get(20).?);
    try testing.expectEqualStrings("five", tree.get(5).?);
    try testing.expect(tree.get(99) == null);
}

test "BTree: insert duplicate replaces value" {
    const Tree = BTree(i32, i32, 4, TestContext);
    var tree = Tree.init(testing.allocator, .{});
    defer tree.deinit();

    _ = try tree.insert(10, 100);
    const old = try tree.insert(10, 200);

    try testing.expectEqual(@as(i32, 100), old.?);
    try testing.expectEqual(@as(i32, 200), tree.get(10).?);
    try testing.expectEqual(@as(usize, 1), tree.count());
}

test "BTree: remove existing key" {
    const Tree = BTree(i32, []const u8, 4, TestContext);
    var tree = Tree.init(testing.allocator, .{});
    defer tree.deinit();

    _ = try tree.insert(10, "ten");
    _ = try tree.insert(20, "twenty");
    _ = try tree.insert(30, "thirty");

    const removed = tree.remove(20);
    try testing.expect(removed != null);
    try testing.expectEqualStrings("twenty", removed.?.value);
    try testing.expectEqual(@as(usize, 2), tree.count());
    try testing.expect(tree.get(20) == null);
    try testing.expectEqualStrings("ten", tree.get(10).?);
    try testing.expectEqualStrings("thirty", tree.get(30).?);
}

test "BTree: remove non-existing key" {
    const Tree = BTree(i32, i32, 4, TestContext);
    var tree = Tree.init(testing.allocator, .{});
    defer tree.deinit();

    _ = try tree.insert(10, 100);
    const removed = tree.remove(99);
    try testing.expect(removed == null);
    try testing.expectEqual(@as(usize, 1), tree.count());
}

test "BTree: stress test with many insertions" {
    const Tree = BTree(i32, i32, 4, TestContext);
    var tree = Tree.init(testing.allocator, .{});
    defer tree.deinit();

    const n = 1000;
    for (0..n) |i| {
        _ = try tree.insert(@intCast(i), @intCast(i * 10));
    }

    try testing.expectEqual(@as(usize, n), tree.count());

    for (0..n) |i| {
        const val = tree.get(@intCast(i));
        try testing.expect(val != null);
        try testing.expectEqual(@as(i32, @intCast(i * 10)), val.?);
    }

    try tree.validate();
}

test "BTree: stress test with insertions and deletions" {
    const Tree = BTree(i32, i32, 4, TestContext);
    var tree = Tree.init(testing.allocator, .{});
    defer tree.deinit();

    const n = 500;

    // Insert
    for (0..n) |i| {
        _ = try tree.insert(@intCast(i), @intCast(i));
    }

    // Remove every other element
    for (0..n) |i| {
        if (i % 2 == 0) {
            _ = tree.remove(@intCast(i));
        }
    }

    try testing.expectEqual(@as(usize, n / 2), tree.count());

    // Verify remaining elements
    for (0..n) |i| {
        if (i % 2 == 0) {
            try testing.expect(tree.get(@intCast(i)) == null);
        } else {
            try testing.expectEqual(@as(i32, @intCast(i)), tree.get(@intCast(i)).?);
        }
    }

    try tree.validate();
}

test "BTree: iterator over empty tree" {
    const Tree = BTree(i32, i32, 4, TestContext);
    var tree = Tree.init(testing.allocator, .{});
    defer tree.deinit();

    var it = try tree.iterator();
    defer it.deinit();

    try testing.expect(it.next() == null);
}

test "BTree: iterator over elements in sorted order" {
    const Tree = BTree(i32, i32, 4, TestContext);
    var tree = Tree.init(testing.allocator, .{});
    defer tree.deinit();

    const keys = [_]i32{ 50, 30, 70, 20, 40, 60, 80 };
    for (keys) |k| {
        _ = try tree.insert(k, k * 10);
    }

    var it = try tree.iterator();
    defer it.deinit();

    var prev: i32 = -1;
    var count: usize = 0;
    while (it.next()) |entry| {
        try testing.expect(entry.key > prev);
        try testing.expectEqual(entry.key * 10, entry.value);
        prev = entry.key;
        count += 1;
    }

    try testing.expectEqual(keys.len, count);
}

test "BTree: validate empty tree" {
    const Tree = BTree(i32, i32, 4, TestContext);
    var tree = Tree.init(testing.allocator, .{});
    defer tree.deinit();

    try testing.expect(tree.isEmpty());
    try testing.expectEqual(@as(usize, 0), tree.count());
    try tree.validate();
}

test "BTree: validate after operations" {
    const Tree = BTree(i32, i32, 4, TestContext);
    var tree = Tree.init(testing.allocator, .{});
    defer tree.deinit();

    for (0..100) |idx| {
        const i: i32 = @intCast(idx);
        _ = try tree.insert(i, i);
        try tree.validate();
        try testing.expectEqual(@as(usize, idx + 1), tree.count());
        try testing.expectEqual(@as(?i32, i), tree.get(i));
    }

    for (0..50) |idx| {
        const i: i32 = @intCast(idx);
        _ = tree.remove(i);
        try tree.validate();
        try testing.expectEqual(@as(usize, 100 - idx - 1), tree.count());
    }

    // Verify removed items are gone and remaining are intact
    for (0..50) |idx| {
        const i: i32 = @intCast(idx);
        try testing.expectEqual(@as(?i32, null), tree.get(i));
    }
    for (50..100) |idx| {
        const i: i32 = @intCast(idx);
        try testing.expectEqual(@as(?i32, i), tree.get(i));
    }
}

test "BTree: memory leak check" {
    const Tree = BTree(i32, i32, 16, TestContext);
    var tree = Tree.init(testing.allocator, .{});
    defer tree.deinit();

    for (0..1000) |i| {
        _ = try tree.insert(@intCast(i), @intCast(i * 2));
    }
    try testing.expectEqual(@as(usize, 1000), tree.count());

    var removed_count: usize = 0;
    for (0..500) |i| {
        _ = tree.remove(@intCast(i * 2));
        removed_count += 1;
    }
    try testing.expectEqual(@as(usize, 1000 - removed_count), tree.count());

    // Verify removed items are gone
    for (0..500) |i| {
        try testing.expectEqual(@as(?i32, null), tree.get(@intCast(i * 2)));
    }

    // Allocator will detect leaks at deinit
}
