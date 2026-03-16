const std = @import("std");

/// RadixTree (compressed trie) for space-efficient prefix operations.
///
/// Properties:
/// - O(m) insert, search, delete where m = key length
/// - Space-optimized: single-child chains compressed into edge labels
/// - Supports prefix matching and longest common prefix queries
/// - Fewer nodes than Trie for sparse keysets
///
/// Generic parameters:
/// - K: key element type (e.g., u8 for strings)
/// - V: value type
pub fn RadixTree(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();

        pub const Entry = struct {
            key: []const K,
            value: V,
        };

        const Node = struct {
            // Edge label from parent to this node (owned slice)
            prefix: []K,
            children: std.AutoHashMap(K, *Node),
            value: ?V = null,
            is_terminal: bool = false,

            fn init(allocator: std.mem.Allocator, prefix: []const K) !Node {
                return Node{
                    .prefix = try allocator.dupe(K, prefix),
                    .children = std.AutoHashMap(K, *Node).init(allocator),
                };
            }

            fn deinit(self: *Node, allocator: std.mem.Allocator) void {
                var iter = self.children.iterator();
                while (iter.next()) |entry| {
                    entry.value_ptr.*.deinit(allocator);
                    allocator.destroy(entry.value_ptr.*);
                }
                self.children.deinit();
                allocator.free(self.prefix);
            }
        };

        pub const Iterator = struct {
            allocator: std.mem.Allocator,
            stack: std.ArrayList(StackFrame),
            current_key: std.ArrayList(K),

            const StackFrame = struct {
                node: *Node,
                iter: std.AutoHashMap(K, *Node).Iterator,
                yielded_self: bool,
            };

            /// Returns next element or null when exhausted.
            /// Time: O(1) amortized | Space: O(1)
            pub fn next(self: *Iterator) !?Entry {
                while (self.stack.items.len > 0) {
                    const frame_idx = self.stack.items.len - 1;
                    var frame = &self.stack.items[frame_idx];

                    // Yield current node if terminal and not yet yielded
                    if (frame.node.is_terminal and !frame.yielded_self) {
                        frame.yielded_self = true;
                        const key = try self.allocator.dupe(K, self.current_key.items);
                        return Entry{
                            .key = key,
                            .value = frame.node.value.?,
                        };
                    }

                    // Try to descend to children
                    if (frame.iter.next()) |child_entry| {
                        const child = child_entry.value_ptr.*;

                        // Append child's prefix to current key
                        try self.current_key.appendSlice(self.allocator, child.prefix);

                        const child_iter = child.children.iterator();
                        try self.stack.append(self.allocator, StackFrame{
                            .node = child,
                            .iter = child_iter,
                            .yielded_self = false,
                        });
                        continue;
                    }

                    // No more children, backtrack
                    _ = self.stack.pop();
                    if (self.current_key.items.len >= frame.node.prefix.len) {
                        self.current_key.shrinkRetainingCapacity(
                            self.current_key.items.len - frame.node.prefix.len,
                        );
                    }
                }

                return null;
            }

            /// Frees iterator resources.
            /// Time: O(1) | Space: O(1)
            pub fn deinit(self: *Iterator) void {
                self.stack.deinit(self.allocator);
                self.current_key.deinit(self.allocator);
            }
        };

        pub const PrefixIterator = struct {
            allocator: std.mem.Allocator,
            stack: std.ArrayList(StackFrame),
            current_key: std.ArrayList(K),
            prefix_len: usize,

            const StackFrame = struct {
                node: *Node,
                iter: std.AutoHashMap(K, *Node).Iterator,
                yielded_self: bool,
            };

            /// Returns next element with the prefix or null when exhausted.
            /// Time: O(1) amortized | Space: O(1)
            pub fn next(self: *PrefixIterator) !?Entry {
                while (self.stack.items.len > 0) {
                    const frame_idx = self.stack.items.len - 1;
                    var frame = &self.stack.items[frame_idx];

                    // Yield current node if terminal and not yet yielded
                    if (frame.node.is_terminal and !frame.yielded_self) {
                        frame.yielded_self = true;
                        const key = try self.allocator.dupe(K, self.current_key.items);
                        return Entry{
                            .key = key,
                            .value = frame.node.value.?,
                        };
                    }

                    // Try to descend to children
                    if (frame.iter.next()) |child_entry| {
                        const child = child_entry.value_ptr.*;

                        // Append child's prefix to current key
                        try self.current_key.appendSlice(self.allocator, child.prefix);

                        const child_iter = child.children.iterator();
                        try self.stack.append(self.allocator, StackFrame{
                            .node = child,
                            .iter = child_iter,
                            .yielded_self = false,
                        });
                        continue;
                    }

                    // No more children, backtrack
                    _ = self.stack.pop();
                    if (self.current_key.items.len >= frame.node.prefix.len and
                        self.current_key.items.len > self.prefix_len)
                    {
                        self.current_key.shrinkRetainingCapacity(
                            self.current_key.items.len - frame.node.prefix.len,
                        );
                    }
                }

                return null;
            }

            /// Frees iterator resources.
            /// Time: O(1) | Space: O(1)
            pub fn deinit(self: *PrefixIterator) void {
                self.stack.deinit(self.allocator);
                self.current_key.deinit(self.allocator);
            }
        };

        allocator: std.mem.Allocator,
        root: Node,
        size: usize = 0,

        const PathEntry = struct { node: *Node, char: K };

        // -- Lifecycle --

        /// Initializes an empty container.
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: std.mem.Allocator) !Self {
            return Self{
                .allocator = allocator,
                .root = try Node.init(allocator, &[_]K{}),
            };
        }

        /// Frees all allocated memory. Invalidates all iterators.
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.root.deinit(self.allocator);
            self.* = undefined;
        }

        /// Creates a deep copy of the container.
        /// Time: O(n log n) | Space: O(n)
        pub fn clone(self: *const Self) !Self {
            var new_tree = try Self.init(self.allocator);
            try self.cloneNode(&self.root, &new_tree.root);
            new_tree.size = self.size;
            return new_tree;
        }

        fn cloneNode(self: *const Self, src: *const Node, dst: *Node) !void {
            dst.value = src.value;
            dst.is_terminal = src.is_terminal;
            self.allocator.free(dst.prefix);
            dst.prefix = try self.allocator.dupe(K, src.prefix);

            var iter = src.children.iterator();
            while (iter.next()) |entry| {
                const first_char = entry.key_ptr.*;
                const child = entry.value_ptr.*;

                const new_child = try self.allocator.create(Node);
                new_child.* = try Node.init(self.allocator, &[_]K{});

                try dst.children.put(first_char, new_child);
                try self.cloneNode(child, new_child);
            }
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

        /// Time: O(m) where m = key.len | Space: O(m) worst-case
        pub fn insert(self: *Self, key: []const K, value: V) !?V {
            if (key.len == 0) return error.EmptyKey;

            var current = &self.root;
            var remaining = key;

            while (remaining.len > 0) {
                const first_char = remaining[0];
                const child_entry = current.children.getPtr(first_char);

                if (child_entry) |child_ptr| {
                    const child = child_ptr.*;
                    const common_len = commonPrefixLen(child.prefix, remaining);

                    if (common_len == child.prefix.len) {
                        // Full match, continue down
                        current = child;
                        remaining = remaining[common_len..];
                    } else {
                        // Partial match, need to split
                        try self.splitNode(current, child, first_char, common_len);
                        current = current.children.get(first_char).?;
                        remaining = remaining[common_len..];
                    }
                } else {
                    // No matching child, create new leaf
                    const new_node = try self.allocator.create(Node);
                    new_node.* = try Node.init(self.allocator, remaining);
                    new_node.value = value;
                    new_node.is_terminal = true;
                    try current.children.put(first_char, new_node);
                    self.size += 1;
                    return null;
                }
            }

            // Key fully consumed, mark current node as terminal
            const old_value = current.value;
            current.value = value;

            if (!current.is_terminal) {
                current.is_terminal = true;
                self.size += 1;
                return null;
            }

            return old_value;
        }

        fn commonPrefixLen(a: []const K, b: []const K) usize {
            var i: usize = 0;
            const min_len = @min(a.len, b.len);
            while (i < min_len and a[i] == b[i]) : (i += 1) {}
            return i;
        }

        fn splitNode(self: *Self, parent: *Node, child: *Node, first_char: K, split_at: usize) !void {
            // Create new intermediate node with common prefix
            const new_node = try self.allocator.create(Node);
            new_node.* = try Node.init(self.allocator, child.prefix[0..split_at]);

            // Update child's prefix to remaining part
            const old_prefix = child.prefix;
            child.prefix = try self.allocator.dupe(K, old_prefix[split_at..]);
            self.allocator.free(old_prefix);

            // Rewire: parent -> new_node -> child
            const child_first_char = child.prefix[0];
            try new_node.children.put(child_first_char, child);
            try parent.children.put(first_char, new_node);
        }

        /// Time: O(m) | Space: O(1)
        pub fn remove(self: *Self, key: []const K) ?V {
            if (key.len == 0) return null;

            var path = std.ArrayList(PathEntry){};
            defer path.deinit(self.allocator);

            var current = &self.root;
            var remaining = key;

            while (remaining.len > 0) {
                const first_char = remaining[0];
                const child = current.children.get(first_char) orelse return null;

                const common_len = commonPrefixLen(child.prefix, remaining);
                if (common_len < child.prefix.len) {
                    // Prefix mismatch
                    return null;
                }

                path.append(self.allocator, .{ .node = current, .char = first_char }) catch return null;
                current = child;
                remaining = remaining[common_len..];
            }

            if (!current.is_terminal) return null;

            const old_value = current.value.?;
            current.is_terminal = false;
            current.value = null;
            self.size -= 1;

            // Cleanup: merge nodes if child count is 1 and not terminal
            self.compressPath(path.items) catch {};

            return old_value;
        }

        fn compressPath(self: *Self, path: []const PathEntry) !void {
            var i = path.len;
            while (i > 0) {
                i -= 1;
                const parent = path[i].node;
                const char = path[i].char;
                const child = parent.children.get(char) orelse continue;

                // Case 1: Child is non-terminal with no children -> remove it
                if (!child.is_terminal and child.children.count() == 0) {
                    _ = parent.children.remove(char);
                    child.deinit(self.allocator);
                    self.allocator.destroy(child);
                    continue;
                }

                // Case 2: Child is non-terminal with exactly one grandchild -> merge
                if (!child.is_terminal and child.children.count() == 1) {
                    var child_iter = child.children.iterator();
                    const grandchild_entry = child_iter.next().?;
                    const grandchild = grandchild_entry.value_ptr.*;

                    // Merge child and grandchild
                    const merged_prefix = try std.mem.concat(
                        self.allocator,
                        K,
                        &[_][]const K{ child.prefix, grandchild.prefix },
                    );
                    self.allocator.free(grandchild.prefix);
                    grandchild.prefix = merged_prefix;

                    // Replace child with grandchild
                    try parent.children.put(char, grandchild);

                    // Free old child
                    self.allocator.free(child.prefix);
                    child.children.deinit();
                    self.allocator.destroy(child);
                }
            }
        }

        // -- Lookup --

        /// Time: O(m) | Space: O(1)
        pub fn get(self: *const Self, key: []const K) ?V {
            const node = self.findNode(key) orelse return null;
            return if (node.is_terminal) node.value else null;
        }

        /// Checks if a key exists in the container.
        /// Time: O(m) where m = key.len | Space: O(1)
        pub fn contains(self: *const Self, key: []const K) bool {
            return self.get(key) != null;
        }

        fn findNode(self: *const Self, key: []const K) ?*const Node {
            var current: *const Node = &self.root;
            var remaining = key;

            while (remaining.len > 0) {
                const first_char = remaining[0];
                const child = current.children.get(first_char) orelse return null;

                const common_len = commonPrefixLen(child.prefix, remaining);
                if (common_len < child.prefix.len) {
                    return null; // Prefix mismatch
                }

                current = child;
                remaining = remaining[common_len..];
            }

            return current;
        }

        // -- Prefix Operations --

        /// Check if tree contains any key with given prefix.
        /// Time: O(m) | Space: O(1)
        pub fn hasPrefix(self: *const Self, prefix: []const K) bool {
            var current: *const Node = &self.root;
            var remaining = prefix;

            while (remaining.len > 0) {
                const first_char = remaining[0];
                const child = current.children.get(first_char) orelse return false;

                const common_len = commonPrefixLen(child.prefix, remaining);
                if (common_len == 0) return false;

                // If remaining is a prefix of child.prefix, we found it
                if (common_len == remaining.len) return true;

                // If child.prefix is a prefix of remaining, continue
                if (common_len == child.prefix.len) {
                    current = child;
                    remaining = remaining[common_len..];
                } else {
                    // Partial match that doesn't align with either being a prefix
                    return false;
                }
            }

            return true;
        }

        /// Iterate over all entries with given prefix.
        /// Time: O(k + n) where k = results, n = nodes visited | Space: O(h) for stack
        pub fn prefixIterator(self: *Self, prefix: []const K) !PrefixIterator {
            var prefix_key = std.ArrayList(K){};
            errdefer prefix_key.deinit(self.allocator);
            var stack = std.ArrayList(PrefixIterator.StackFrame){};
            errdefer stack.deinit(self.allocator);

            // Navigate to the subtree containing the prefix
            var current = &self.root;
            var remaining = prefix;

            while (remaining.len > 0) {
                const first_char = remaining[0];
                const child = current.children.get(first_char) orelse {
                    // No matches
                    return PrefixIterator{
                        .allocator = self.allocator,
                        .stack = stack,
                        .current_key = prefix_key,
                        .prefix_len = prefix.len,
                    };
                };

                const common_len = commonPrefixLen(child.prefix, remaining);
                if (common_len == 0) {
                    // No match
                    return PrefixIterator{
                        .allocator = self.allocator,
                        .stack = stack,
                        .current_key = prefix_key,
                        .prefix_len = prefix.len,
                    };
                }

                if (common_len == remaining.len) {
                    // Prefix fully consumed, child might extend beyond
                    // Add child's full prefix to current key
                    try prefix_key.appendSlice(self.allocator, child.prefix);

                    // Start iteration from this child
                    const child_mut = @constCast(child);
                    const iter = child_mut.children.iterator();
                    try stack.append(self.allocator, PrefixIterator.StackFrame{
                        .node = child_mut,
                        .iter = iter,
                        .yielded_self = false,
                    });
                    break;
                } else if (common_len == child.prefix.len) {
                    // Child's prefix is a prefix of remaining, continue down
                    try prefix_key.appendSlice(self.allocator, child.prefix);
                    current = child;
                    remaining = remaining[common_len..];
                } else {
                    // Partial match in the middle - no keys with this prefix
                    return PrefixIterator{
                        .allocator = self.allocator,
                        .stack = stack,
                        .current_key = prefix_key,
                        .prefix_len = prefix.len,
                    };
                }
            }

            return PrefixIterator{
                .allocator = self.allocator,
                .stack = stack,
                .current_key = prefix_key,
                .prefix_len = prefix.len,
            };
        }

        /// Find longest common prefix of all keys in tree.
        /// Time: O(m) where m = LCP length | Space: O(1)
        pub fn longestCommonPrefix(self: *const Self) ![]K {
            if (self.size == 0) return &[_]K{};

            var lcp = std.ArrayList(K){};
            errdefer lcp.deinit(self.allocator);

            var current = &self.root;

            while (true) {
                if (current.is_terminal or current.children.count() != 1) break;

                var iter = current.children.iterator();
                const child_entry = iter.next().?;
                const child = child_entry.value_ptr.*;

                try lcp.appendSlice(self.allocator, child.prefix);
                current = child;
            }

            return lcp.toOwnedSlice(self.allocator);
        }

        // -- Iteration --

        /// Creates an iterator for the container.
        /// Time: O(log n) amortized | Space: O(log n)
        pub fn iterator(self: *Self) !Iterator {
            var stack = std.ArrayList(Iterator.StackFrame){};
            const iter = self.root.children.iterator();
            try stack.append(self.allocator, Iterator.StackFrame{
                .node = &self.root,
                .iter = iter,
                .yielded_self = false,
            });

            return Iterator{
                .allocator = self.allocator,
                .stack = stack,
                .current_key = std.ArrayList(K){},
            };
        }

        // -- Debug --

        /// Formats container for debugging output.
        /// Time: O(n) | Space: O(n)
        pub fn format(
            self: *const Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try writer.print("RadixTree{{ size: {d} }}", .{self.size});
        }

        /// Validates internal tree invariants. Returns error if tree is corrupted.
        /// Time: O(n) | Space: O(h) where h is tree height
        pub fn validate(self: *const Self) !void {
            try self.validateNode(&self.root, 0);
        }

        fn validateNode(self: *const Self, node: *const Node, _: usize) !void {
            // Non-root nodes must have non-empty prefix
            const node_addr = @intFromPtr(node);
            const root_addr = @intFromPtr(&self.root);
            const is_root = node_addr == root_addr;

            if (!is_root and node.prefix.len == 0) {
                return error.TreeInvariant;
            }

            // Terminal nodes must have values
            if (node.is_terminal and node.value == null) {
                return error.TreeInvariant;
            }

            // Non-terminal nodes with no children are invalid (except root)
            if (!node.is_terminal and node.children.count() == 0 and !is_root) {
                return error.TreeInvariant;
            }

            // Validate children
            var iter = node.children.iterator();
            while (iter.next()) |entry| {
                const first_char = entry.key_ptr.*;
                const child = entry.value_ptr.*;

                // Child's prefix must start with the key it's stored under
                if (child.prefix.len == 0 or child.prefix[0] != first_char) {
                    return error.TreeInvariant;
                }

                try self.validateNode(child, 0);
            }
        }
    };
}

// -- Tests --

const testing = std.testing;

test "RadixTree: basic insert and get" {
    var tree = try RadixTree(u8, i32).init(testing.allocator);
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 0), tree.count());
    try testing.expect(tree.isEmpty());

    const old1 = try tree.insert("apple", 1);
    try testing.expectEqual(@as(?i32, null), old1);
    try testing.expectEqual(@as(usize, 1), tree.count());
    try testing.expectEqual(@as(?i32, 1), tree.get("apple"));

    const old2 = try tree.insert("app", 2);
    try testing.expectEqual(@as(?i32, null), old2);
    try testing.expectEqual(@as(usize, 2), tree.count());
    try testing.expectEqual(@as(?i32, 2), tree.get("app"));
    try testing.expectEqual(@as(?i32, 1), tree.get("apple"));

    // Update existing
    const old3 = try tree.insert("apple", 10);
    try testing.expectEqual(@as(?i32, 1), old3);
    try testing.expectEqual(@as(usize, 2), tree.count());
    try testing.expectEqual(@as(?i32, 10), tree.get("apple"));

    try tree.validate();
}

test "RadixTree: prefix compression" {
    var tree = try RadixTree(u8, i32).init(testing.allocator);
    defer tree.deinit();

    // Insert keys with common prefix
    _ = try tree.insert("testing", 1);
    _ = try tree.insert("test", 2);
    _ = try tree.insert("tester", 3);
    _ = try tree.insert("team", 4);

    try testing.expectEqual(@as(usize, 4), tree.count());
    try testing.expectEqual(@as(?i32, 1), tree.get("testing"));
    try testing.expectEqual(@as(?i32, 2), tree.get("test"));
    try testing.expectEqual(@as(?i32, 3), tree.get("tester"));
    try testing.expectEqual(@as(?i32, 4), tree.get("team"));

    try testing.expect(!tree.contains("te"));
    try testing.expect(!tree.contains("tes"));

    try tree.validate();
}

test "RadixTree: remove" {
    var tree = try RadixTree(u8, i32).init(testing.allocator);
    defer tree.deinit();

    _ = try tree.insert("test", 1);
    _ = try tree.insert("testing", 2);
    _ = try tree.insert("tester", 3);

    try testing.expectEqual(@as(usize, 3), tree.count());

    const removed1 = tree.remove("test");
    try testing.expectEqual(@as(?i32, 1), removed1);
    try testing.expectEqual(@as(usize, 2), tree.count());
    try testing.expect(!tree.contains("test"));
    try testing.expectEqual(@as(?i32, 2), tree.get("testing"));

    const removed2 = tree.remove("nonexistent");
    try testing.expectEqual(@as(?i32, null), removed2);
    try testing.expectEqual(@as(usize, 2), tree.count());

    try tree.validate();
}

test "RadixTree: prefix operations" {
    var tree = try RadixTree(u8, i32).init(testing.allocator);
    defer tree.deinit();

    _ = try tree.insert("apple", 1);
    _ = try tree.insert("application", 2);
    _ = try tree.insert("apply", 3);
    _ = try tree.insert("banana", 4);

    try testing.expect(tree.hasPrefix("app"));
    try testing.expect(tree.hasPrefix("apple"));
    try testing.expect(tree.hasPrefix("ban"));
    try testing.expect(!tree.hasPrefix("orange"));

    // Test prefix iterator
    var iter = try tree.prefixIterator("app");
    defer iter.deinit();

    var count: usize = 0;
    while (try iter.next()) |entry| {
        defer testing.allocator.free(entry.key);
        count += 1;
        try testing.expect(
            std.mem.eql(u8, entry.key, "apple") or
                std.mem.eql(u8, entry.key, "application") or
                std.mem.eql(u8, entry.key, "apply"),
        );
    }
    try testing.expectEqual(@as(usize, 3), count);
}

test "RadixTree: longest common prefix" {
    var tree = try RadixTree(u8, i32).init(testing.allocator);
    defer tree.deinit();

    _ = try tree.insert("flower", 1);
    _ = try tree.insert("flow", 2);
    _ = try tree.insert("flight", 3);

    const lcp = try tree.longestCommonPrefix();
    defer testing.allocator.free(lcp);

    try testing.expect(std.mem.eql(u8, lcp, "fl"));
}

test "RadixTree: iterator" {
    var tree = try RadixTree(u8, i32).init(testing.allocator);
    defer tree.deinit();

    _ = try tree.insert("cat", 1);
    _ = try tree.insert("car", 2);
    _ = try tree.insert("card", 3);

    var iter = try tree.iterator();
    defer iter.deinit();

    var count: usize = 0;
    while (try iter.next()) |entry| {
        defer testing.allocator.free(entry.key);
        count += 1;
    }
    try testing.expectEqual(@as(usize, 3), count);
}

test "RadixTree: clone" {
    var tree = try RadixTree(u8, i32).init(testing.allocator);
    defer tree.deinit();

    _ = try tree.insert("hello", 1);
    _ = try tree.insert("world", 2);

    var cloned = try tree.clone();
    defer cloned.deinit();

    try testing.expectEqual(tree.count(), cloned.count());
    try testing.expectEqual(tree.get("hello"), cloned.get("hello"));
    try testing.expectEqual(tree.get("world"), cloned.get("world"));

    // Modify original
    _ = try tree.insert("new", 3);
    try testing.expectEqual(@as(usize, 3), tree.count());
    try testing.expectEqual(@as(usize, 2), cloned.count());
}

test "RadixTree: edge cases" {
    var tree = try RadixTree(u8, i32).init(testing.allocator);
    defer tree.deinit();

    // Empty key should error
    try testing.expectError(error.EmptyKey, tree.insert("", 1));

    // Single character
    _ = try tree.insert("a", 1);
    try testing.expectEqual(@as(?i32, 1), tree.get("a"));

    // Very long key
    const long_key = "a" ** 1000;
    _ = try tree.insert(long_key, 99);
    try testing.expectEqual(@as(?i32, 99), tree.get(long_key));

    try tree.validate();
}

test "RadixTree: stress test" {
    var tree = try RadixTree(u8, usize).init(testing.allocator);
    defer tree.deinit();

    const keys = [_][]const u8{
        "apple",    "application", "apply",   "banana",  "band",
        "bandana",  "can",         "candy",   "cane",    "cat",
        "dog",      "dodge",       "door",    "elephant", "email",
        "emotion",  "flower",      "flow",    "flight",  "fly",
    };

    // Insert
    for (keys, 0..) |key, i| {
        _ = try tree.insert(key, i);
    }
    try testing.expectEqual(keys.len, tree.count());

    // Verify all
    for (keys, 0..) |key, i| {
        try testing.expectEqual(@as(?usize, i), tree.get(key));
    }

    // Remove half
    var removed: usize = 0;
    for (keys, 0..) |key, i| {
        if (i % 2 == 0) {
            const val = tree.remove(key);
            try testing.expect(val != null);
            removed += 1;
        }
    }
    try testing.expectEqual(keys.len - removed, tree.count());

    try tree.validate();
}

test "RadixTree: memory leak check" {
    var tree = try RadixTree(u8, i32).init(testing.allocator);
    defer tree.deinit();

    for (0..100) |i| {
        var buf: [32]u8 = undefined;
        const key = try std.fmt.bufPrint(&buf, "key{d}", .{i});
        _ = try tree.insert(key, @intCast(i));
    }

    for (0..50) |i| {
        var buf: [32]u8 = undefined;
        const key = try std.fmt.bufPrint(&buf, "key{d}", .{i});
        _ = tree.remove(key);
    }

    try testing.expectEqual(@as(usize, 50), tree.count());
}
