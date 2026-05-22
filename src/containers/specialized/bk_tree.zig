const std = @import("std");
const testing = std.testing;

/// BK-Tree (Burkhard-Keller Tree) - Metric space search tree.
///
/// A BK-Tree is a tree data structure specialized for discrete metric spaces.
/// It enables efficient searching for elements within a given distance threshold,
/// making it ideal for spell checking, fuzzy matching, and similarity search.
///
/// Time: insert O(log n) average, search O(log n) for small tolerance
/// Space: O(n)
///
/// Properties:
/// - Each node stores an element and a map of child distances
/// - Children are indexed by their distance from the parent
/// - Triangle inequality ensures efficient pruning during search
///
/// Common distance functions:
/// - Levenshtein distance (edit distance) for strings
/// - Hamming distance for fixed-length sequences
/// - Custom metric functions for domain-specific use cases
///
/// Type Parameters:
/// - T: Element type
/// - Context: Distance function context
/// - distanceFn: fn(ctx: Context, a: T, b: T) usize - must satisfy metric axioms:
///   1. d(x, y) >= 0 (non-negativity)
///   2. d(x, y) = 0 iff x == y (identity)
///   3. d(x, y) = d(y, x) (symmetry)
///   4. d(x, z) <= d(x, y) + d(y, z) (triangle inequality)
pub fn BKTree(
    comptime T: type,
    comptime Context: type,
    comptime distanceFn: fn (ctx: Context, a: T, b: T) usize,
) type {
    return struct {
        const Self = @This();

        pub const Node = struct {
            value: T,
            children: std.AutoHashMap(usize, *Node),

            fn init(allocator: std.mem.Allocator, value: T) !*Node {
                const node = try allocator.create(Node);
                node.* = .{
                    .value = value,
                    .children = std.AutoHashMap(usize, *Node).init(allocator),
                };
                return node;
            }

            fn deinit(self: *Node, allocator: std.mem.Allocator) void {
                var it = self.children.valueIterator();
                while (it.next()) |child| {
                    child.*.deinit(allocator);
                }
                self.children.deinit();
                allocator.destroy(self);
            }
        };

        pub const SearchResult = struct {
            value: T,
            distance: usize,
        };

        pub const Iterator = struct {
            results: std.ArrayList(SearchResult),
            index: usize,

            /// Returns next element or null when exhausted.
            /// Time: O(1) amortized | Space: O(1)
            pub fn next(self: *Iterator) ?SearchResult {
                if (self.index >= self.results.items.len) return null;
                const result = self.results.items[self.index];
                self.index += 1;
                return result;
            }

            /// Frees iterator resources.
            /// Time: O(1) | Space: O(1)
            pub fn deinit(self: *Iterator, allocator: std.mem.Allocator) void {
                self.results.deinit(allocator);
            }
        };

        allocator: std.mem.Allocator,
        root: ?*Node,
        context: Context,
        size: usize,

        // -- Lifecycle --

        /// Initialize an empty BK-Tree.
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: std.mem.Allocator, context: Context) Self {
            return .{
                .allocator = allocator,
                .root = null,
                .context = context,
                .size = 0,
            };
        }

        /// Free all allocated memory.
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.root) |root| {
                root.deinit(self.allocator);
            }
        }

        // -- Capacity --

        /// Return the number of elements in the tree.
        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            return self.size;
        }

        /// Check if the tree is empty.
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.size == 0;
        }

        // -- Modification --

        /// Insert a value into the tree.
        /// Time: O(log n) average | Space: O(1) amortized
        pub fn insert(self: *Self, value: T) !void {
            if (self.root == null) {
                self.root = try Node.init(self.allocator, value);
                self.size += 1;
                return;
            }

            var current = self.root.?;
            while (true) {
                const dist = distanceFn(self.context, current.value, value);
                if (dist == 0) {
                    // Duplicate, do not insert
                    return;
                }

                const entry = try current.children.getOrPut(dist);
                if (!entry.found_existing) {
                    entry.value_ptr.* = try Node.init(self.allocator, value);
                    self.size += 1;
                    return;
                }

                current = entry.value_ptr.*;
            }
        }

        // -- Search --

        /// Search for all values within a given distance tolerance.
        /// Time: O(log n) average for small tolerance | Space: O(k) where k = result count
        pub fn search(self: *const Self, query: T, tolerance: usize) !Iterator {
            var results: std.ArrayList(SearchResult) = .{};
            errdefer results.deinit(self.allocator);

            if (self.root) |root| {
                try self.searchNode(root, query, tolerance, &results);
            }

            return Iterator{
                .results = results,
                .index = 0,
            };
        }

        fn searchNode(
            self: *const Self,
            node: *Node,
            query: T,
            tolerance: usize,
            results: *std.ArrayList(SearchResult),
        ) !void {
            const dist = distanceFn(self.context, node.value, query);

            if (dist <= tolerance) {
                try results.append(self.allocator, .{ .value = node.value, .distance = dist });
            }

            // Triangle inequality pruning:
            // If |d(node, query) - d(node, child)| > tolerance, skip child
            const min_dist: usize = if (dist > tolerance) dist - tolerance else 0;
            const max_dist = dist + tolerance;

            var it = node.children.iterator();
            while (it.next()) |entry| {
                const child_dist = entry.key_ptr.*;
                if (child_dist >= min_dist and child_dist <= max_dist) {
                    try self.searchNode(entry.value_ptr.*, query, tolerance, results);
                }
            }
        }

        /// Check if an exact match exists.
        /// Time: O(log n) average | Space: O(1)
        pub fn contains(self: *const Self, value: T) bool {
            if (self.root == null) return false;

            var current = self.root.?;
            while (true) {
                const dist = distanceFn(self.context, current.value, value);
                if (dist == 0) return true;

                if (current.children.get(dist)) |child| {
                    current = child;
                } else {
                    return false;
                }
            }
        }

        // -- Debug --

        /// Validate tree invariants (metric properties).
        /// Time: O(n) | Space: O(h) where h = tree height
        pub fn validate(self: *const Self) !void {
            if (self.root) |root| {
                try self.validateNode(root);
            }
        }

        fn validateNode(self: *const Self, node: *Node) !void {
            var it = node.children.iterator();
            while (it.next()) |entry| {
                const expected_dist = entry.key_ptr.*;
                const child = entry.value_ptr.*;

                // Verify stored distance matches actual distance
                const actual_dist = distanceFn(self.context, node.value, child.value);
                if (expected_dist != actual_dist) {
                    return error.BKTreeInvariant;
                }

                // Recursively validate children
                try self.validateNode(child);
            }
        }
    };
}

// -- Tests --

// Levenshtein distance (edit distance) for strings
fn levenshteinDistance(_: void, a: []const u8, b: []const u8) usize {
    if (a.len == 0) return b.len;
    if (b.len == 0) return a.len;

    const m = a.len;
    const n = b.len;

    // Use stack allocation for small strings, heap for large
    var buffer: [256]usize = undefined;
    var heap_buffer: ?[]usize = null;
    defer if (heap_buffer) |buf| testing.allocator.free(buf);

    const dp: []usize = if ((n + 1) <= buffer.len)
        buffer[0..(n + 1)]
    else blk: {
        const buf = testing.allocator.alloc(usize, n + 1) catch {
            // Fallback: return max distance for OOM in test
            return @max(m, n);
        };
        heap_buffer = buf;
        break :blk buf;
    };

    // Initialize first row
    for (0..n + 1) |i| {
        dp[i] = i;
    }

    var i: usize = 1;
    while (i <= m) : (i += 1) {
        var prev = dp[0];
        dp[0] = i;

        var j: usize = 1;
        while (j <= n) : (j += 1) {
            const temp = dp[j];
            if (a[i - 1] == b[j - 1]) {
                dp[j] = prev;
            } else {
                dp[j] = 1 + @min(prev, @min(dp[j], dp[j - 1]));
            }
            prev = temp;
        }
    }

    return dp[n];
}

test "BKTree - basic insert and search" {
    var tree = BKTree([]const u8, void, levenshteinDistance).init(testing.allocator, {});
    defer tree.deinit();

    try tree.insert("hello");
    try tree.insert("help");
    try tree.insert("world");
    try tree.insert("word");

    try testing.expectEqual(@as(usize, 4), tree.count());
    try testing.expect(tree.contains("hello"));
    try testing.expect(tree.contains("help"));
    try testing.expect(!tree.contains("test"));
}

test "BKTree - search with tolerance" {
    var tree = BKTree([]const u8, void, levenshteinDistance).init(testing.allocator, {});
    defer tree.deinit();

    try tree.insert("kitten");
    try tree.insert("sitten"); // distance 1 from kitten (s -> k)
    try tree.insert("kitchen"); // distance 2 from kitten
    try tree.insert("written"); // distance 3 from kitten

    // Search for "kitten" with tolerance 1
    var iter = try tree.search("kitten", 1);
    defer iter.deinit(testing.allocator);

    var found_count: usize = 0;
    while (iter.next()) |result| {
        found_count += 1;
        // Should find "kitten" (distance 0) and "sitten" (distance 1)
        try testing.expect(result.distance <= 1);
    }
    try testing.expectEqual(@as(usize, 2), found_count);
}

test "BKTree - spell checking use case" {
    var tree = BKTree([]const u8, void, levenshteinDistance).init(testing.allocator, {});
    defer tree.deinit();

    // Build dictionary
    const words = [_][]const u8{ "apple", "application", "apply", "ape", "banana", "bandana", "can", "cat", "dog" };
    for (words) |word| {
        try tree.insert(word);
    }

    // Typo: "aple" -> should find "apple", "ape"
    var iter = try tree.search("aple", 1);
    defer iter.deinit(testing.allocator);

    var suggestions: std.ArrayList([]const u8) = .{};
    defer suggestions.deinit(testing.allocator);

    while (iter.next()) |result| {
        try suggestions.append(testing.allocator, result.value);
    }

    try testing.expect(suggestions.items.len >= 2);
    try testing.expect(std.mem.indexOf(u8, suggestions.items[0], "ap") != null);
}

test "BKTree - empty tree" {
    var tree = BKTree([]const u8, void, levenshteinDistance).init(testing.allocator, {});
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 0), tree.count());
    try testing.expect(tree.isEmpty());
    try testing.expect(!tree.contains("test"));

    var iter = try tree.search("test", 5);
    defer iter.deinit(testing.allocator);
    try testing.expect(iter.next() == null);
}

test "BKTree - duplicates" {
    var tree = BKTree([]const u8, void, levenshteinDistance).init(testing.allocator, {});
    defer tree.deinit();

    try tree.insert("test");
    try tree.insert("test"); // Duplicate
    try tree.insert("test"); // Duplicate

    try testing.expectEqual(@as(usize, 1), tree.count());
}

test "BKTree - hamming distance for fixed-length strings" {
    const Context = struct {
        fn hamming(_: void, a: []const u8, b: []const u8) usize {
            if (a.len != b.len) return a.len + b.len; // Invalid
            var dist: usize = 0;
            for (0..a.len) |i| {
                if (a[i] != b[i]) dist += 1;
            }
            return dist;
        }
    };

    var tree = BKTree([]const u8, void, Context.hamming).init(testing.allocator, {});
    defer tree.deinit();

    try tree.insert("1010");
    try tree.insert("1011");
    try tree.insert("0000");
    try tree.insert("1111");

    var iter = try tree.search("1010", 1);
    defer iter.deinit(testing.allocator);

    var count: usize = 0;
    while (iter.next()) |result| {
        count += 1;
        try testing.expect(result.distance <= 1);
    }
    try testing.expect(count == 2); // "1010" (0) and "1011" (1)
}

test "BKTree - validate" {
    var tree = BKTree([]const u8, void, levenshteinDistance).init(testing.allocator, {});
    defer tree.deinit();

    try tree.insert("alpha");
    try tree.insert("beta");
    try tree.insert("gamma");

    try tree.validate();
}

test "BKTree - large tolerance search" {
    var tree = BKTree([]const u8, void, levenshteinDistance).init(testing.allocator, {});
    defer tree.deinit();

    try tree.insert("a");
    try tree.insert("b");
    try tree.insert("c");
    try tree.insert("d");

    // Large tolerance should find all
    var iter = try tree.search("x", 10);
    defer iter.deinit(testing.allocator);

    var count: usize = 0;
    while (iter.next()) |_| {
        count += 1;
    }
    try testing.expectEqual(@as(usize, 4), count);
}

test "BKTree - stress test" {
    var tree = BKTree([]const u8, void, levenshteinDistance).init(testing.allocator, {});
    defer tree.deinit();

    const words = [_][]const u8{
        "algorithm", "data",      "structure", "tree",   "graph",  "heap",   "queue",
        "stack",     "array",     "list",      "map",    "set",    "hash",   "sort",
        "search",    "insert",    "delete",    "update", "create", "remove", "find",
        "get",       "put",       "add",       "clear",  "empty",  "size",   "count",
        "contains",  "iterator",  "next",      "prev",   "first",  "last",   "begin",
        "end",       "reverse",   "sort",      "filter", "reduce", "map",    "foreach",
    };

    for (words) |word| {
        try tree.insert(word);
    }

    try testing.expect(tree.count() <= words.len); // May have duplicates

    // Search for similar words
    var iter = try tree.search("algorythm", 2);
    defer iter.deinit(testing.allocator);

    var found = false;
    while (iter.next()) |result| {
        if (std.mem.eql(u8, result.value, "algorithm")) {
            found = true;
        }
    }
    try testing.expect(found);
}

test "BKTree - search with tolerance 0 (exact match only)" {
    var tree = BKTree([]const u8, void, levenshteinDistance).init(testing.allocator, {});
    defer tree.deinit();

    try tree.insert("apple");
    try tree.insert("apply");  // distance 2 from apple
    try tree.insert("ape");    // distance 2 from apple
    try tree.insert("append"); // distance 3 from apple

    // Search with tolerance 0 should only find exact matches
    var iter = try tree.search("apple", 0);
    defer iter.deinit(testing.allocator);

    var count: usize = 0;
    while (iter.next()) |result| {
        count += 1;
        // All results must have distance exactly 0
        try testing.expectEqual(@as(usize, 0), result.distance);
        try testing.expect(std.mem.eql(u8, result.value, "apple"));
    }
    try testing.expectEqual(@as(usize, 1), count);

    // Search for non-existent word with tolerance 0 should find nothing
    var iter2 = try tree.search("xyz", 0);
    defer iter2.deinit(testing.allocator);
    try testing.expect(iter2.next() == null);
}

test "BKTree - multiple results at different distances" {
    var tree = BKTree([]const u8, void, levenshteinDistance).init(testing.allocator, {});
    defer tree.deinit();

    try tree.insert("cat");
    try tree.insert("bat");    // distance 1 from cat (c->b)
    try tree.insert("rat");    // distance 1 from cat (c->r)
    try tree.insert("cats");   // distance 1 from cat (insert s)
    try tree.insert("cast");   // distance 2 from cat (c->c, a->a, t->s, insert t)
    try tree.insert("hat");    // distance 1 from cat (c->h)

    // Search with tolerance 1 should find exact match plus distance-1 items
    var iter = try tree.search("cat", 1);
    defer iter.deinit(testing.allocator);

    var distance_counts = [2]usize{ 0, 0 }; // counts for distance 0 and 1
    var found_values: std.ArrayList([]const u8) = .{};
    defer found_values.deinit(testing.allocator);

    while (iter.next()) |result| {
        try testing.expect(result.distance <= 1);
        if (result.distance < 2) {
            distance_counts[result.distance] += 1;
            try found_values.append(testing.allocator, result.value);
        }
    }

    // Should find 1 item at distance 0 (cat itself)
    try testing.expectEqual(@as(usize, 1), distance_counts[0]);
    // Should find multiple items at distance 1 (bat, rat, cats, hat)
    try testing.expectEqual(@as(usize, 4), distance_counts[1]);
}

test "BKTree - iterator exhaustion behavior" {
    var tree = BKTree([]const u8, void, levenshteinDistance).init(testing.allocator, {});
    defer tree.deinit();

    try tree.insert("hello");
    try tree.insert("hell");
    try tree.insert("held");

    var iter = try tree.search("hello", 1);
    defer iter.deinit(testing.allocator);

    // Exhaust the iterator
    var count: usize = 0;
    while (iter.next()) |_| {
        count += 1;
    }
    try testing.expect(count > 0);

    // Call next() multiple times after exhaustion
    // Each subsequent call should return null (not crash or loop)
    try testing.expect(iter.next() == null);
    try testing.expect(iter.next() == null);
    try testing.expect(iter.next() == null);
    try testing.expect(iter.next() == null);

    // Verify iterator index is beyond results length
    try testing.expect(iter.index >= iter.results.items.len);
}

test "BKTree - single element tree" {
    var tree = BKTree([]const u8, void, levenshteinDistance).init(testing.allocator, {});
    defer tree.deinit();

    try tree.insert("single");

    // Count and isEmpty checks
    try testing.expectEqual(@as(usize, 1), tree.count());
    try testing.expect(!tree.isEmpty());

    // Contains check for exact match
    try testing.expect(tree.contains("single"));
    try testing.expect(!tree.contains("single2"));

    // Search with different tolerances
    var iter0 = try tree.search("single", 0);
    defer iter0.deinit(testing.allocator);
    var count0: usize = 0;
    while (iter0.next()) |result| {
        count0 += 1;
        try testing.expectEqual(@as(usize, 0), result.distance);
        try testing.expect(std.mem.eql(u8, result.value, "single"));
    }
    try testing.expectEqual(@as(usize, 1), count0);

    var iter1 = try tree.search("single", 5);
    defer iter1.deinit(testing.allocator);
    var count1: usize = 0;
    while (iter1.next()) |result| {
        count1 += 1;
        try testing.expect(std.mem.eql(u8, result.value, "single"));
    }
    try testing.expectEqual(@as(usize, 1), count1);

    // Search for different word should fail with tolerance 0
    var iter_no_match = try tree.search("double", 0);
    defer iter_no_match.deinit(testing.allocator);
    try testing.expect(iter_no_match.next() == null);

    // Validate single-node tree
    try tree.validate();
}

test "BKTree - sequential similar words (unbalanced insertion)" {
    var tree = BKTree([]const u8, void, levenshteinDistance).init(testing.allocator, {});
    defer tree.deinit();

    // Insert words in a pattern that creates unbalanced tree structure
    // Each word is close to the previous one
    try tree.insert("cat");
    try tree.insert("cats");    // distance 1: append s
    try tree.insert("catty");   // distance 2: append ty
    try tree.insert("cattle");  // distance 3 from cat, but distance 3 from cats
    try tree.insert("catfish"); // distance 4: append fish

    // Verify all insertions worked
    try testing.expectEqual(@as(usize, 5), tree.count());

    // Verify all items are findable via contains
    try testing.expect(tree.contains("cat"));
    try testing.expect(tree.contains("cats"));
    try testing.expect(tree.contains("catty"));
    try testing.expect(tree.contains("cattle"));
    try testing.expect(tree.contains("catfish"));

    // Search from the first item with various tolerances
    var iter1 = try tree.search("cat", 1);
    defer iter1.deinit(testing.allocator);
    var count1: usize = 0;
    while (iter1.next()) |_| {
        count1 += 1;
    }
    try testing.expect(count1 >= 2); // cat and cats

    var iter2 = try tree.search("cat", 2);
    defer iter2.deinit(testing.allocator);
    var count2: usize = 0;
    while (iter2.next()) |_| {
        count2 += 1;
    }
    try testing.expect(count2 >= 3); // cat, cats, catty

    // Search from an intermediate item
    var iter_middle = try tree.search("cats", 2);
    defer iter_middle.deinit(testing.allocator);
    var middle_count: usize = 0;
    while (iter_middle.next()) |_| {
        middle_count += 1;
    }
    try testing.expect(middle_count >= 2); // At least cats itself and nearby words

    // Verify tree invariants hold after unbalanced insertions
    try tree.validate();
}
