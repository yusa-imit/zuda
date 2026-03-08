const std = @import("std");

/// Trie (prefix tree) for efficient string operations.
///
/// Properties:
/// - O(m) insert, search, delete where m = key length
/// - O(1) space per character in stored keys (with sharing)
/// - Supports prefix matching and autocomplete
/// - Generic over key type (typically u8 for ASCII/UTF-8)
///
/// Generic parameters:
/// - K: key element type (e.g., u8 for strings)
/// - V: value type
pub fn Trie(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();

        pub const Entry = struct {
            key: []const K,
            value: V,
        };

        const Node = struct {
            children: std.AutoHashMap(K, *Node),
            value: ?V = null,
            is_terminal: bool = false,

            fn init(allocator: std.mem.Allocator) Node {
                return Node{
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
            }
        };

        pub const Iterator = struct {
            allocator: std.mem.Allocator,
            stack: std.ArrayList(StackFrame),
            current_key: std.ArrayList(K),

            const StackFrame = struct {
                node: *Node,
                iter: std.AutoHashMap(K, *Node).Iterator,
                char: ?K,
            };

            pub fn next(self: *Iterator) !?Entry {
                while (self.stack.items.len > 0) {
                    const frame_idx = self.stack.items.len - 1;
                    var frame = &self.stack.items[frame_idx];

                    // Check if current node is terminal
                    if (frame.node.is_terminal and frame.char == null) {
                        frame.char = 0; // Mark as visited
                        const key = try self.allocator.dupe(K, self.current_key.items);
                        return Entry{
                            .key = key,
                            .value = frame.node.value.?,
                        };
                    }

                    // Try to descend to children
                    if (frame.iter.next()) |child_entry| {
                        const char = child_entry.key_ptr.*;
                        const child = child_entry.value_ptr.*;

                        try self.current_key.append(self.allocator, char);

                        const child_iter = child.children.iterator();
                        try self.stack.append(self.allocator, StackFrame{
                            .node = child,
                            .iter = child_iter,
                            .char = null,
                        });
                        continue;
                    }

                    // No more children, backtrack
                    _ = self.stack.pop();
                    if (self.current_key.items.len > 0) {
                        _ = self.current_key.pop();
                    }
                }

                return null;
            }

            pub fn deinit(self: *Iterator) void {
                self.stack.deinit(self.allocator);
                self.current_key.deinit(self.allocator);
            }
        };

        pub const PrefixIterator = struct {
            allocator: std.mem.Allocator,
            stack: std.ArrayList(StackFrame),
            current_key: std.ArrayList(K),
            prefix: []const K,

            const StackFrame = struct {
                node: *Node,
                iter: std.AutoHashMap(K, *Node).Iterator,
                yielded_self: bool,
            };

            pub fn next(self: *PrefixIterator) !?Entry {
                while (self.stack.items.len > 0) {
                    const frame_idx = self.stack.items.len - 1;
                    var frame = &self.stack.items[frame_idx];

                    // Yield current node if it's terminal and not yet yielded
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
                        const char = child_entry.key_ptr.*;
                        const child = child_entry.value_ptr.*;

                        try self.current_key.append(self.allocator, char);

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
                    if (self.current_key.items.len > self.prefix.len) {
                        _ = self.current_key.pop();
                    }
                }

                return null;
            }

            pub fn deinit(self: *PrefixIterator) void {
                self.stack.deinit(self.allocator);
                self.current_key.deinit(self.allocator);
            }
        };

        allocator: std.mem.Allocator,
        root: Node,
        size: usize = 0,

        // -- Lifecycle --

        pub fn init(allocator: std.mem.Allocator) Self {
            return Self{
                .allocator = allocator,
                .root = Node.init(allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            self.root.deinit(self.allocator);
            self.* = undefined;
        }

        pub fn clone(self: *const Self) !Self {
            var new_trie = Self.init(self.allocator);
            try self.cloneNode(&self.root, &new_trie.root);
            new_trie.size = self.size;
            return new_trie;
        }

        fn cloneNode(self: *const Self, src: *const Node, dst: *Node) !void {
            dst.value = src.value;
            dst.is_terminal = src.is_terminal;

            var iter = src.children.iterator();
            while (iter.next()) |entry| {
                const char = entry.key_ptr.*;
                const child = entry.value_ptr.*;

                const new_child = try self.allocator.create(Node);
                new_child.* = Node.init(self.allocator);

                try dst.children.put(char, new_child);
                try self.cloneNode(child, new_child);
            }
        }

        // -- Capacity --

        pub fn count(self: *const Self) usize {
            return self.size;
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.size == 0;
        }

        // -- Modification --

        /// Time: O(m) where m = key.len | Space: O(m) worst-case
        pub fn insert(self: *Self, key: []const K, value: V) !?V {
            var current = &self.root;

            for (key) |char| {
                const result = try current.children.getOrPut(char);
                if (!result.found_existing) {
                    const new_node = try self.allocator.create(Node);
                    new_node.* = Node.init(self.allocator);
                    result.value_ptr.* = new_node;
                }
                current = result.value_ptr.*;
            }

            const old_value = current.value;
            current.value = value;

            if (!current.is_terminal) {
                current.is_terminal = true;
                self.size += 1;
                return null;
            }

            return old_value;
        }

        /// Time: O(m) | Space: O(1)
        pub fn remove(self: *Self, key: []const K) ?V {
            if (key.len == 0) return null;

            // Find node and track path
            var path = std.ArrayList(*Node){};
            defer path.deinit(self.allocator);

            var current = &self.root;
            path.append(self.allocator, current) catch return null;

            for (key) |char| {
                if (current.children.get(char)) |child| {
                    current = child;
                    path.append(self.allocator, current) catch return null;
                } else {
                    return null; // Key not found
                }
            }

            if (!current.is_terminal) {
                return null; // Not a complete key
            }

            const value = current.value.?;
            current.is_terminal = false;
            current.value = null;
            self.size -= 1;

            // Clean up nodes with no children and not terminal
            var i = path.items.len - 1;
            while (i > 0) : (i -= 1) {
                const node = path.items[i];
                if (node.children.count() == 0 and !node.is_terminal) {
                    const parent = path.items[i - 1];
                    const char = key[i - 1];
                    _ = parent.children.remove(char);
                    node.deinit(self.allocator);
                    self.allocator.destroy(node);
                } else {
                    break;
                }
            }

            return value;
        }

        // -- Lookup --

        /// Time: O(m) | Space: O(1)
        pub fn get(self: *const Self, key: []const K) ?V {
            var current: *const Node = &self.root;

            for (key) |char| {
                if (current.children.get(char)) |child| {
                    current = child;
                } else {
                    return null;
                }
            }

            if (current.is_terminal) {
                return current.value;
            }

            return null;
        }

        pub fn contains(self: *const Self, key: []const K) bool {
            return self.get(key) != null;
        }

        /// Check if any key in the trie starts with the given prefix
        /// Time: O(m) | Space: O(1)
        pub fn hasPrefix(self: *const Self, prefix: []const K) bool {
            var current: *const Node = &self.root;

            for (prefix) |char| {
                if (current.children.get(char)) |child| {
                    current = child;
                } else {
                    return false;
                }
            }

            return true;
        }

        /// Find the longest common prefix among all keys in the trie
        /// Time: O(min_key_len) | Space: O(min_key_len)
        pub fn longestCommonPrefix(self: *const Self) ![]K {
            var result = std.ArrayList(K){};
            errdefer result.deinit(self.allocator);

            var current: *const Node = &self.root;

            while (current.children.count() == 1 and !current.is_terminal) {
                var iter = current.children.iterator();
                const entry = iter.next().?;
                try result.append(self.allocator, entry.key_ptr.*);
                current = entry.value_ptr.*;
            }

            return result.toOwnedSlice(self.allocator);
        }

        // -- Iteration --

        pub fn iterator(self: *const Self) !Iterator {
            var stack = std.ArrayList(Iterator.StackFrame){};
            const current_key = std.ArrayList(K){};

            const root_iter = self.root.children.iterator();
            try stack.append(self.allocator, Iterator.StackFrame{
                .node = @constCast(&self.root),
                .iter = root_iter,
                .char = null,
            });

            return Iterator{
                .allocator = self.allocator,
                .stack = stack,
                .current_key = current_key,
            };
        }

        /// Returns an iterator over all keys with the given prefix
        /// Time: O(m) to find prefix node, then O(k) per iteration where k = result key length
        pub fn prefixIterator(self: *const Self, prefix: []const K) !?PrefixIterator {
            var current: *const Node = &self.root;

            // Navigate to prefix node
            for (prefix) |char| {
                if (current.children.get(char)) |child| {
                    current = child;
                } else {
                    return null; // Prefix not found
                }
            }

            var stack = std.ArrayList(PrefixIterator.StackFrame){};
            var current_key = std.ArrayList(K){};
            try current_key.appendSlice(self.allocator, prefix);

            const node_iter = current.children.iterator();
            try stack.append(self.allocator, PrefixIterator.StackFrame{
                .node = @constCast(current),
                .iter = node_iter,
                .yielded_self = false,
            });

            return PrefixIterator{
                .allocator = self.allocator,
                .stack = stack,
                .current_key = current_key,
                .prefix = prefix,
            };
        }

        // -- Bulk --

        pub fn clear(self: *Self) void {
            self.root.deinit(self.allocator);
            self.root = Node.init(self.allocator);
            self.size = 0;
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
            try writer.print("Trie{{ size: {} }}", .{self.size});
        }

        /// Validates Trie invariants:
        /// 1. Size matches number of terminal nodes
        /// 2. No orphaned nodes (all nodes reachable from root)
        pub fn validate(self: *const Self) !void {
            var terminal_count: usize = 0;
            try self.validateNode(&self.root, &terminal_count);

            if (terminal_count != self.size) {
                return error.InvalidSize;
            }
        }

        fn validateNode(self: *const Self, node: *const Node, terminal_count: *usize) !void {
            if (node.is_terminal) {
                if (node.value == null) {
                    return error.TerminalWithoutValue;
                }
                terminal_count.* += 1;
            }

            var iter = node.children.iterator();
            while (iter.next()) |entry| {
                try self.validateNode(entry.value_ptr.*, terminal_count);
            }
        }
    };
}

// -- Tests --

const testing = std.testing;

const StringTrie = Trie(u8, []const u8);

test "Trie: basic insert and get" {
    var trie = StringTrie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("cat", "feline");
    _ = try trie.insert("car", "vehicle");
    _ = try trie.insert("card", "paper");

    try testing.expectEqualStrings("feline", trie.get("cat").?);
    try testing.expectEqualStrings("vehicle", trie.get("car").?);
    try testing.expectEqualStrings("paper", trie.get("card").?);
    try testing.expectEqual(@as(?[]const u8, null), trie.get("ca"));
    try testing.expectEqual(@as(?[]const u8, null), trie.get("cards"));
}

test "Trie: insert duplicate updates value" {
    var trie = StringTrie.init(testing.allocator);
    defer trie.deinit();

    try testing.expectEqual(@as(?[]const u8, null), try trie.insert("cat", "old"));
    const old = try trie.insert("cat", "new");
    try testing.expectEqualStrings("old", old.?);
    try testing.expectEqualStrings("new", trie.get("cat").?);
    try testing.expectEqual(@as(usize, 1), trie.count());
}

test "Trie: contains" {
    var trie = StringTrie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("hello", "world");
    try testing.expect(trie.contains("hello"));
    try testing.expect(!trie.contains("hell"));
    try testing.expect(!trie.contains("hello!"));
}

test "Trie: hasPrefix" {
    var trie = StringTrie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("hello", "world");
    _ = try trie.insert("help", "assist");

    try testing.expect(trie.hasPrefix("hel"));
    try testing.expect(trie.hasPrefix("hello"));
    try testing.expect(trie.hasPrefix("help"));
    try testing.expect(!trie.hasPrefix("world"));
}

test "Trie: remove existing key" {
    var trie = StringTrie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("cat", "feline");
    _ = try trie.insert("car", "vehicle");

    const removed = trie.remove("cat");
    try testing.expectEqualStrings("feline", removed.?);
    try testing.expectEqual(@as(?[]const u8, null), trie.get("cat"));
    try testing.expectEqualStrings("vehicle", trie.get("car").?);
    try testing.expectEqual(@as(usize, 1), trie.count());
}

test "Trie: remove nonexistent key" {
    var trie = StringTrie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("cat", "feline");
    const removed = trie.remove("dog");
    try testing.expectEqual(@as(?[]const u8, null), removed);
    try testing.expectEqual(@as(usize, 1), trie.count());
}

test "Trie: remove cleans up internal nodes" {
    var trie = StringTrie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("card", "paper");
    _ = try trie.insert("car", "vehicle");

    _ = trie.remove("card");
    try testing.expect(trie.hasPrefix("car"));
    try testing.expectEqual(@as(usize, 1), trie.count());
}

test "Trie: longestCommonPrefix" {
    var trie = StringTrie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("flower", "plant");
    _ = try trie.insert("flow", "movement");
    _ = try trie.insert("flight", "flying");

    const lcp = try trie.longestCommonPrefix();
    defer testing.allocator.free(lcp);

    try testing.expectEqualStrings("fl", lcp);
}

test "Trie: prefixIterator" {
    var trie = StringTrie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("cat", "feline");
    _ = try trie.insert("car", "vehicle");
    _ = try trie.insert("card", "paper");
    _ = try trie.insert("dog", "canine");

    const maybe_iter = try trie.prefixIterator("ca");
    try testing.expect(maybe_iter != null);

    var iter = maybe_iter.?;
    defer iter.deinit();

    var count: usize = 0;
    while (try iter.next()) |entry| {
        defer testing.allocator.free(entry.key);
        count += 1;
        try testing.expect(std.mem.startsWith(u8, entry.key, "ca"));
    }

    try testing.expectEqual(@as(usize, 3), count);
}

test "Trie: iterator" {
    var trie = StringTrie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("cat", "feline");
    _ = try trie.insert("car", "vehicle");
    _ = try trie.insert("dog", "canine");

    var iter = try trie.iterator();
    defer iter.deinit();

    var count: usize = 0;
    while (try iter.next()) |entry| {
        defer testing.allocator.free(entry.key);
        count += 1;
    }

    try testing.expectEqual(@as(usize, 3), count);
}

test "Trie: clear" {
    var trie = StringTrie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("cat", "feline");
    _ = try trie.insert("dog", "canine");

    trie.clear();
    try testing.expectEqual(@as(usize, 0), trie.count());
    try testing.expect(trie.isEmpty());
    try testing.expectEqual(@as(?[]const u8, null), trie.get("cat"));
}

test "Trie: clone" {
    var trie = StringTrie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("cat", "feline");
    _ = try trie.insert("dog", "canine");

    var cloned = try trie.clone();
    defer cloned.deinit();

    try testing.expectEqual(trie.count(), cloned.count());
    try testing.expectEqualStrings("feline", cloned.get("cat").?);
    try testing.expectEqualStrings("canine", cloned.get("dog").?);

    _ = try cloned.insert("bird", "avian");
    try testing.expectEqual(@as(?[]const u8, null), trie.get("bird"));
}

test "Trie: stress test with many words" {
    var trie = StringTrie.init(testing.allocator);
    defer trie.deinit();

    const words = [_][]const u8{
        "apple",    "application", "apply",  "banana", "band",
        "bandana",  "can",         "candle", "candy",  "cat",
        "category", "dog",         "dot",    "dotted", "egg",
    };

    for (words) |word| {
        _ = try trie.insert(word, "value");
    }

    try testing.expectEqual(@as(usize, words.len), trie.count());

    for (words) |word| {
        try testing.expect(trie.contains(word));
    }

    for (words, 0..) |word, i| {
        if (i % 2 == 0) {
            _ = trie.remove(word);
        }
    }

    try testing.expectEqual(@as(usize, words.len / 2), trie.count());
}

test "Trie: validate invariants" {
    var trie = StringTrie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("cat", "feline");
    _ = try trie.insert("car", "vehicle");
    try trie.validate();

    _ = trie.remove("cat");
    try trie.validate();

    trie.clear();
    try trie.validate();
}

test "Trie: memory leak detection" {
    var trie = StringTrie.init(testing.allocator);
    defer trie.deinit();

    const n = 50;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var key_buf: [20]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "key_{d}", .{i});
        _ = try trie.insert(key, "value");
    }

    i = 0;
    while (i < n) : (i += 2) {
        var key_buf: [20]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "key_{d}", .{i});
        _ = trie.remove(key);
    }

    trie.clear();
}
