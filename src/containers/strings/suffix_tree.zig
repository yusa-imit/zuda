const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// SuffixTree - Compressed trie of all suffixes of a string.
/// Provides efficient pattern matching, longest repeated substring, and substring queries.
///
/// This implementation uses a simpler O(n²) construction for correctness.
/// Future optimization: Ukkonen's algorithm for O(n) construction.
///
/// Time Complexity:
/// - Construction: O(n²) — current implementation
/// - Pattern search: O(m) where m is pattern length
/// - Longest repeated substring: O(n²)
///
/// Space Complexity: O(n²) worst case, O(n) average
///
/// Generic parameters:
/// - T: Character type (typically u8 for bytes)
pub fn SuffixTree(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Internal representation of a tree node
        const Node = struct {
            /// Children indexed by first character of edge label
            children: std.AutoHashMap(T, *Edge),
            /// Suffix index for leaf nodes (null for internal nodes)
            suffix_index: ?usize,
            allocator: Allocator,

            fn init(allocator: Allocator, suffix_index: ?usize) !*Node {
                const node = try allocator.create(Node);
                node.* = .{
                    .children = std.AutoHashMap(T, *Edge).init(allocator),
                    .suffix_index = suffix_index,
                    .allocator = allocator,
                };
                return node;
            }

            fn deinit(self: *Node) void {
                var it = self.children.iterator();
                while (it.next()) |entry| {
                    entry.value_ptr.*.deinit();
                }
                self.children.deinit();
                self.allocator.destroy(self);
            }
        };

        /// Edge in the suffix tree
        const Edge = struct {
            /// Label of the edge (substring of original text)
            label: []const T,
            /// Target node
            target: *Node,
            allocator: Allocator,

            fn init(allocator: Allocator, label: []const T, target: *Node) !*Edge {
                const edge = try allocator.create(Edge);
                edge.* = .{
                    .label = label,
                    .target = target,
                    .allocator = allocator,
                };
                return edge;
            }

            fn deinit(self: *Edge) void {
                self.target.deinit();
                self.allocator.destroy(self);
            }
        };

        text: []const T,
        root: *Node,
        allocator: Allocator,

        // -- Lifecycle --

        /// Build a suffix tree from the given text.
        /// Time: O(n²) | Space: O(n²) worst case
        pub fn init(allocator: Allocator, text: []const T) !Self {
            if (text.len == 0) {
                return error.EmptyText;
            }

            const root = try Node.init(allocator, null);
            errdefer root.deinit();

            var self = Self{
                .text = text,
                .root = root,
                .allocator = allocator,
            };

            // Insert all suffixes
            for (0..text.len) |i| {
                const suffix = text[i..];
                try self.insertSuffix(suffix, i);
            }

            return self;
        }

        /// Free all memory used by the suffix tree.
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.root.deinit();
        }

        // -- Construction --

        fn insertSuffix(self: *Self, suffix: []const T, suffix_index: usize) !void {
            if (suffix.len == 0) return;

            var node = self.root;
            var i: usize = 0;

            while (i < suffix.len) {
                const ch = suffix[i];

                if (node.children.get(ch)) |edge| {
                    // Find longest common prefix with edge label
                    var j: usize = 0;
                    while (j < edge.label.len and i + j < suffix.len and
                           edge.label[j] == suffix[i + j]) : (j += 1) {}

                    if (j == edge.label.len) {
                        // Consumed entire edge, continue at target node
                        node = edge.target;
                        i += j;
                    } else {
                        // Need to split edge
                        const split_node = try Node.init(self.allocator, null);
                        errdefer split_node.deinit();

                        // Create new edge from split node to old target
                        const rest_label = edge.label[j..];
                        const rest_edge = try Edge.init(self.allocator, rest_label, edge.target);
                        try split_node.children.put(rest_label[0], rest_edge);

                        // Update old edge to point to split node
                        edge.target = split_node;
                        edge.label = edge.label[0..j];

                        // Add new branch for remaining suffix
                        if (i + j < suffix.len) {
                            const new_label = suffix[i + j ..];
                            const leaf = try Node.init(self.allocator, suffix_index);
                            const new_edge = try Edge.init(self.allocator, new_label, leaf);
                            try split_node.children.put(new_label[0], new_edge);
                        } else {
                            split_node.suffix_index = suffix_index;
                        }
                        return;
                    }
                } else {
                    // No matching edge, create new leaf
                    const new_label = suffix[i..];
                    const leaf = try Node.init(self.allocator, suffix_index);
                    const new_edge = try Edge.init(self.allocator, new_label, leaf);
                    try node.children.put(ch, new_edge);
                    return;
                }
            }

            // Suffix exhausted at this node
            node.suffix_index = suffix_index;
        }

        // -- Pattern Matching --

        /// Search for a pattern in the text.
        /// Returns true if pattern exists in the text.
        /// Time: O(m) where m is pattern length | Space: O(1)
        pub fn contains(self: *const Self, pattern: []const T) bool {
            if (pattern.len == 0) return true;
            if (pattern.len > self.text.len) return false;

            var node = self.root;
            var i: usize = 0;

            while (i < pattern.len) {
                const ch = pattern[i];
                const edge = node.children.get(ch) orelse return false;

                var j: usize = 0;
                while (j < edge.label.len and i + j < pattern.len) : (j += 1) {
                    if (edge.label[j] != pattern[i + j]) {
                        return false;
                    }
                }

                i += j;
                if (i < pattern.len) {
                    node = edge.target;
                }
            }

            return true;
        }

        /// Find all occurrences of a pattern in the text.
        /// Returns a list of starting positions (caller owns the memory).
        /// Time: O(m + k) where m is pattern length, k is number of occurrences | Space: O(k)
        pub fn findAll(self: *const Self, allocator: Allocator, pattern: []const T) ![]usize {
            if (pattern.len == 0 or pattern.len > self.text.len) {
                return try allocator.alloc(usize, 0);
            }

            var positions: std.ArrayList(usize) = .empty;
            defer positions.deinit(allocator);

            // Find the node/edge representing the pattern
            var node = self.root;
            var i: usize = 0;
            var found_edge: ?*Edge = null;
            var edge_match_len: usize = 0;

            while (i < pattern.len) {
                const ch = pattern[i];
                const edge = node.children.get(ch) orelse return try positions.toOwnedSlice(allocator);

                var j: usize = 0;
                while (j < edge.label.len and i + j < pattern.len) : (j += 1) {
                    if (edge.label[j] != pattern[i + j]) {
                        return try positions.toOwnedSlice(allocator);
                    }
                }

                i += j;
                if (i >= pattern.len) {
                    // Pattern exhausted
                    if (j < edge.label.len) {
                        // Pattern ends in middle of edge
                        found_edge = edge;
                        edge_match_len = j;
                    } else {
                        // Pattern ends exactly at edge boundary
                        node = edge.target;
                    }
                } else {
                    // Pattern continues, move to target node
                    node = edge.target;
                }
            }

            // Collect all suffix indices
            if (found_edge) |edge| {
                try self.collectLeaves(allocator, edge.target, &positions);
            } else {
                try self.collectLeaves(allocator, node, &positions);
            }

            // Sort positions
            std.mem.sort(usize, positions.items, {}, std.sort.asc(usize));
            return try positions.toOwnedSlice(allocator);
        }

        fn collectLeaves(self: *const Self, allocator: Allocator, node: *const Node, positions: *std.ArrayList(usize)) !void {
            if (node.suffix_index) |idx| {
                try positions.append(allocator, idx);
            }

            var it = node.children.iterator();
            while (it.next()) |entry| {
                try self.collectLeaves(allocator, entry.value_ptr.*.target, positions);
            }
        }

        // -- Longest Repeated Substring --

        /// Find the longest repeated substring in the text.
        /// Returns the substring as a slice (points into original text).
        /// Time: O(n²) | Space: O(1)
        pub fn longestRepeatedSubstring(self: *const Self) ?[]const T {
            if (self.text.len < 2) return null;

            var max_len: usize = 0;
            var max_start: usize = 0;

            self.dfsLongestRepeated(self.root, 0, &max_len, &max_start);

            if (max_len == 0) return null;
            return self.text[max_start .. max_start + max_len];
        }

        fn dfsLongestRepeated(self: *const Self, node: *const Node, depth: usize, max_len: *usize, max_start: *usize) void {
            // Internal nodes with multiple children represent repeated substrings
            // OR nodes with suffix_index AND at least one child (suffix ends here + continues in child)
            const is_repeated = (node.children.count() >= 2) or
                               (node.suffix_index != null and node.children.count() >= 1);

            if (is_repeated and depth > 0 and depth > max_len.*) {
                max_len.* = depth;
                // Find any leaf to get the substring position
                if (self.findAnyLeaf(node)) |idx| {
                    if (idx + depth <= self.text.len) {
                        max_start.* = idx;
                    }
                }
            }

            var it = node.children.iterator();
            while (it.next()) |entry| {
                const edge = entry.value_ptr.*;
                self.dfsLongestRepeated(edge.target, depth + edge.label.len, max_len, max_start);
            }
        }

        fn findAnyLeaf(self: *const Self, node: *const Node) ?usize {
            if (node.suffix_index) |idx| return idx;

            var it = node.children.iterator();
            while (it.next()) |entry| {
                if (self.findAnyLeaf(entry.value_ptr.*.target)) |idx| {
                    return idx;
                }
            }

            return null;
        }

        // -- Capacity --

        /// Returns the length of the text.
        /// Time: O(1) | Space: O(1)
        pub fn len(self: *const Self) usize {
            return self.text.len;
        }

        /// Returns true if the text is empty.
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.text.len == 0;
        }

        // -- Debug & Validation --

        /// Validate the suffix tree structure invariants.
        /// Time: O(n) | Space: O(h) where h is tree height
        pub fn validate(self: *const Self) !void {
            if (self.text.len == 0) return error.InvalidTree;
            try self.validateNode(self.root);
        }

        fn validateNode(self: *const Self, node: *const Node) !void {
            // Internal nodes should have at least one child (except if it's a suffix end)
            if (node.children.count() == 0 and node.suffix_index == null and node != self.root) {
                return error.InvalidInternalNode;
            }

            // Recursively validate children
            var it = node.children.iterator();
            while (it.next()) |entry| {
                const edge = entry.value_ptr.*;

                // Edge label should not be empty
                if (edge.label.len == 0) {
                    return error.InvalidEdgeLabel;
                }

                // First character of edge should match the key
                if (edge.label[0] != entry.key_ptr.*) {
                    return error.InvalidEdgeKey;
                }

                try self.validateNode(edge.target);
            }
        }

        /// Format the suffix tree for debugging.
        /// Time: O(n) | Space: O(h)
        pub fn format(self: *const Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;
            try writer.writeAll("SuffixTree(");
            if (self.text.len <= 50) {
                try writer.print("text=\"{s}\"", .{self.text});
            } else {
                try writer.print("text=\"{s}...\" (len={})", .{ self.text[0..50], self.text.len });
            }
            try writer.print(", nodes={})", .{self.countNodes()});
        }

        fn countNodes(self: *const Self) usize {
            return self.countNodesRecursive(self.root);
        }

        fn countNodesRecursive(self: *const Self, node: *const Node) usize {
            var count: usize = 1;
            var it = node.children.iterator();
            while (it.next()) |entry| {
                count += self.countNodesRecursive(entry.value_ptr.*.target);
            }
            return count;
        }
    };
}

// -- Tests --

test "SuffixTree: basic construction" {
    const text = "banana";
    var tree = try SuffixTree(u8).init(testing.allocator, text);
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 6), tree.len());
    try tree.validate();
}

test "SuffixTree: pattern matching" {
    const text = "banana";
    var tree = try SuffixTree(u8).init(testing.allocator, text);
    defer tree.deinit();

    try testing.expect(tree.contains("ana"));
    try testing.expect(tree.contains("ban"));
    try testing.expect(tree.contains("na"));
    try testing.expect(!tree.contains("ann"));
    try testing.expect(!tree.contains("xyz"));
}

test "SuffixTree: find all occurrences" {
    const text = "banana";
    var tree = try SuffixTree(u8).init(testing.allocator, text);
    defer tree.deinit();

    const positions = try tree.findAll(testing.allocator, "ana");
    defer testing.allocator.free(positions);

    try testing.expectEqual(@as(usize, 2), positions.len);
    try testing.expectEqual(@as(usize, 1), positions[0]); // "banana"
    try testing.expectEqual(@as(usize, 3), positions[1]); // "banana"
}

test "SuffixTree: longest repeated substring" {
    const text = "banana";
    var tree = try SuffixTree(u8).init(testing.allocator, text);
    defer tree.deinit();

    const lrs = tree.longestRepeatedSubstring() orelse return error.TestFailed;
    try testing.expectEqualStrings("ana", lrs);
}

test "SuffixTree: no repeated substring" {
    const text = "abcdef";
    var tree = try SuffixTree(u8).init(testing.allocator, text);
    defer tree.deinit();

    const lrs = tree.longestRepeatedSubstring();
    try testing.expectEqual(@as(?[]const u8, null), lrs);
}

test "SuffixTree: repeated patterns" {
    const text = "abcabcabc";
    var tree = try SuffixTree(u8).init(testing.allocator, text);
    defer tree.deinit();

    try testing.expect(tree.contains("abc"));
    try testing.expect(tree.contains("bcabc"));
    try testing.expect(tree.contains("abcabc"));

    const positions = try tree.findAll(testing.allocator, "abc");
    defer testing.allocator.free(positions);
    try testing.expectEqual(@as(usize, 3), positions.len);
}

test "SuffixTree: single character" {
    const text = "a";
    var tree = try SuffixTree(u8).init(testing.allocator, text);
    defer tree.deinit();

    try testing.expect(tree.contains("a"));
    try testing.expect(!tree.contains("b"));
    try tree.validate();
}

test "SuffixTree: two characters" {
    const text = "ab";
    var tree = try SuffixTree(u8).init(testing.allocator, text);
    defer tree.deinit();

    try testing.expect(tree.contains("a"));
    try testing.expect(tree.contains("b"));
    try testing.expect(tree.contains("ab"));
    try testing.expect(!tree.contains("ba"));
    try tree.validate();
}

test "SuffixTree: memory leak check" {
    const text = "abracadabra";
    var tree = try SuffixTree(u8).init(testing.allocator, text);
    defer tree.deinit();

    const positions = try tree.findAll(testing.allocator, "abra");
    defer testing.allocator.free(positions);

    try testing.expectEqual(@as(usize, 2), positions.len);
}

test "SuffixTree: stress test" {
    const allocator = testing.allocator;
    const text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit.";
    var tree = try SuffixTree(u8).init(allocator, text);
    defer tree.deinit();

    try testing.expect(tree.contains("Lorem"));
    try testing.expect(tree.contains("ipsum"));
    try testing.expect(tree.contains("dolor"));
    try testing.expect(!tree.contains("xyz"));

    const positions = try tree.findAll(allocator, "or");
    defer allocator.free(positions);
    try testing.expect(positions.len > 0);

    try tree.validate();
}
