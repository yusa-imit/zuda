const std = @import("std");
const Allocator = std.mem.Allocator;

/// Aho-Corasick - Efficient multi-pattern matching algorithm
///
/// Time Complexity:
///   Construction: O(∑mᵢ) where mᵢ = length of pattern i
///   Searching:    O(n + z) where n = text length, z = number of matches
///   Total:        O(∑mᵢ + n + z)
///
/// Space Complexity: O(∑mᵢ × |Σ|) where Σ = alphabet size
///   In practice: O(number of trie nodes × transitions)
///
/// Characteristics:
/// - Multi-pattern: Searches for all patterns simultaneously
/// - Automaton-based: Builds a finite state machine (trie + failure links)
/// - Linear time: Single pass through text regardless of number of patterns
/// - No backtracking: Uses failure links to handle mismatches
/// - Dictionary matching: Ideal for keyword detection, censoring, virus scanning
///
/// Algorithm:
/// 1. Build trie from all patterns
/// 2. Compute failure links (BFS, similar to KMP's failure function)
/// 3. Compute output links (for overlapping pattern detection)
/// 4. Traverse text using the automaton, emit matches via output links
///
/// Example:
///   patterns = ["he", "she", "his", "hers"]
///   text = "ushers"
///   matches = [("she", 1), ("he", 2), ("hers", 2)]
pub fn AhoCorasick(comptime T: type) type {
    return struct {
        allocator: Allocator,
        root: *Node,
        patterns: []const []const T,

        const Self = @This();

        const Node = struct {
            children: std.AutoHashMap(T, *Node),
            failure: ?*Node,
            output: ?*Node, // For reporting overlapping matches
            pattern_indices: std.ArrayList(usize), // Patterns ending at this node
            depth: usize,

            fn init(allocator: Allocator) !*Node {
                const node = try allocator.create(Node);
                node.* = .{
                    .children = std.AutoHashMap(T, *Node).init(allocator),
                    .failure = null,
                    .output = null,
                    .pattern_indices = .{},
                    .depth = 0,
                };
                return node;
            }

            fn deinit(self: *Node, allocator: Allocator) void {
                var it = self.children.valueIterator();
                while (it.next()) |child| {
                    child.*.deinit(allocator);
                }
                self.children.deinit();
                self.pattern_indices.deinit(allocator);
                allocator.destroy(self);
            }
        };

        pub const Match = struct {
            pattern_index: usize,
            position: usize, // Start position in text
        };

        /// Initialize Aho-Corasick automaton with patterns
        /// Time: O(∑mᵢ + number of nodes) | Space: O(∑mᵢ)
        pub fn init(allocator: Allocator, patterns: []const []const T) !Self {
            if (patterns.len == 0) return error.EmptyPatternSet;

            // Validate patterns
            for (patterns) |pattern| {
                if (pattern.len == 0) return error.EmptyPattern;
            }

            const root = try Node.init(allocator);
            errdefer root.deinit(allocator);

            var self = Self{
                .allocator = allocator,
                .root = root,
                .patterns = patterns,
            };

            // Build trie
            try self.buildTrie();

            // Build failure links and output links
            try self.buildFailureLinks();

            return self;
        }

        pub fn deinit(self: *Self) void {
            self.root.deinit(self.allocator);
        }

        /// Build trie from patterns
        fn buildTrie(self: *Self) !void {
            for (self.patterns, 0..) |pattern, pattern_idx| {
                var current = self.root;

                for (pattern) |ch| {
                    const child = try current.children.getOrPut(ch);
                    if (!child.found_existing) {
                        const new_node = try Node.init(self.allocator);
                        new_node.depth = current.depth + 1;
                        child.value_ptr.* = new_node;
                    }
                    current = child.value_ptr.*;
                }

                // Mark pattern end
                try current.pattern_indices.append(self.allocator, pattern_idx);
            }
        }

        /// Build failure links using BFS (similar to KMP preprocessing)
        fn buildFailureLinks(self: *Self) !void {
            // Use Deque for O(1) queue operations
            // (ArrayList.orderedRemove(0) is O(n) and was the bottleneck)
            const Deque = @import("../../containers/queues/deque.zig").Deque;
            var queue = Deque(*Node).init(self.allocator);
            defer queue.deinit();

            // Initialize: root's children have failure link to root
            self.root.failure = self.root;

            var it = self.root.children.iterator();
            while (it.next()) |entry| {
                const child = entry.value_ptr.*;
                child.failure = self.root;
                try queue.push_back(child);
            }

            // BFS to compute failure links for all other nodes
            while (queue.count() > 0) {
                const current = try queue.pop_front();

                var child_it = current.children.iterator();
                while (child_it.next()) |entry| {
                    const ch = entry.key_ptr.*;
                    const child = entry.value_ptr.*;

                    try queue.push_back(child);

                    // Find failure link by following parent's failure chain
                    var failure_candidate = current.failure;
                    while (failure_candidate != null) {
                        if (failure_candidate.?.children.get(ch)) |target| {
                            child.failure = target;
                            break;
                        }
                        if (failure_candidate == self.root) {
                            child.failure = self.root;
                            break;
                        }
                        failure_candidate = failure_candidate.?.failure;
                    }

                    // Build output link for overlapping patterns
                    if (child.failure) |fail_node| {
                        if (fail_node.pattern_indices.items.len > 0) {
                            child.output = fail_node;
                        } else {
                            child.output = fail_node.output;
                        }
                    }
                }
            }
        }

        /// Find first occurrence of any pattern in text
        /// Returns (pattern_index, position) or null if no match
        /// Time: O(n) | Space: O(1)
        pub fn findFirst(self: *const Self, text: []const T) ?Match {
            if (text.len == 0) return null;

            var current = self.root;

            for (text, 0..) |ch, i| {
                // Follow failure links until we find a valid transition
                while (true) {
                    if (current.children.get(ch)) |child| {
                        current = child;
                        break;
                    }
                    if (current == self.root) break;
                    current = current.failure.?;
                }

                // Check if we've matched any patterns
                if (current.pattern_indices.items.len > 0) {
                    const pattern_idx = current.pattern_indices.items[0];
                    return Match{
                        .pattern_index = pattern_idx,
                        .position = i + 1 - self.patterns[pattern_idx].len,
                    };
                }

                // Check output link for overlapping patterns
                var output_node = current.output;
                while (output_node) |node| {
                    if (node.pattern_indices.items.len > 0) {
                        const pattern_idx = node.pattern_indices.items[0];
                        return Match{
                            .pattern_index = pattern_idx,
                            .position = i + 1 - self.patterns[pattern_idx].len,
                        };
                    }
                    output_node = node.output;
                }
            }

            return null;
        }

        /// Find all occurrences of all patterns in text
        /// Returns ArrayList of matches (pattern_index, position)
        /// Time: O(n + z) where z = number of matches | Space: O(z)
        pub fn findAll(self: *const Self, text: []const T, allocator: Allocator) !std.ArrayList(Match) {
            var matches: std.ArrayList(Match) = .{};
            errdefer matches.deinit(allocator);

            if (text.len == 0) return matches;

            var current = self.root;

            for (text, 0..) |ch, i| {
                // Follow failure links until we find a valid transition
                while (true) {
                    if (current.children.get(ch)) |child| {
                        current = child;
                        break;
                    }
                    if (current == self.root) break;
                    current = current.failure.?;
                }

                // Emit all patterns ending at current node
                for (current.pattern_indices.items) |pattern_idx| {
                    try matches.append(allocator, .{
                        .pattern_index = pattern_idx,
                        .position = i + 1 - self.patterns[pattern_idx].len,
                    });
                }

                // Emit all overlapping patterns via output links
                var output_node = current.output;
                while (output_node) |node| {
                    for (node.pattern_indices.items) |pattern_idx| {
                        try matches.append(allocator, .{
                            .pattern_index = pattern_idx,
                            .position = i + 1 - self.patterns[pattern_idx].len,
                        });
                    }
                    output_node = node.output;
                }
            }

            return matches;
        }

        /// Check if text contains any of the patterns
        /// Time: O(n) | Space: O(1)
        pub fn contains(self: *const Self, text: []const T) bool {
            return self.findFirst(text) != null;
        }

        /// Count total occurrences of all patterns in text
        /// Time: O(n + z) | Space: O(1)
        pub fn count(self: *const Self, text: []const T) usize {
            var total: usize = 0;
            if (text.len == 0) return total;

            var current = self.root;

            for (text) |ch| {
                while (true) {
                    if (current.children.get(ch)) |child| {
                        current = child;
                        break;
                    }
                    if (current == self.root) break;
                    current = current.failure.?;
                }

                total += current.pattern_indices.items.len;

                var output_node = current.output;
                while (output_node) |node| {
                    total += node.pattern_indices.items.len;
                    output_node = node.output;
                }
            }

            return total;
        }
    };
}

// ===== Tests =====

test "AhoCorasick - basic match" {
    const AC = AhoCorasick(u8);
    const patterns = [_][]const u8{ "he", "she", "his", "hers" };
    const patterns_slice: []const []const u8 = &patterns;

    var ac = try AC.init(std.testing.allocator, patterns_slice);
    defer ac.deinit();

    const text = "ushers";
    var matches = try ac.findAll(text, std.testing.allocator);
    defer matches.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 3), matches.items.len);

    // Expected: "she" at 1, "he" at 2, "hers" at 2
    try std.testing.expectEqual(@as(usize, 1), matches.items[0].pattern_index); // "she"
    try std.testing.expectEqual(@as(usize, 1), matches.items[0].position);

    try std.testing.expectEqual(@as(usize, 0), matches.items[1].pattern_index); // "he"
    try std.testing.expectEqual(@as(usize, 2), matches.items[1].position);

    try std.testing.expectEqual(@as(usize, 3), matches.items[2].pattern_index); // "hers"
    try std.testing.expectEqual(@as(usize, 2), matches.items[2].position);
}

test "AhoCorasick - single pattern" {
    const AC = AhoCorasick(u8);
    const patterns = [_][]const u8{"pattern"};
    const patterns_slice: []const []const u8 = &patterns;

    var ac = try AC.init(std.testing.allocator, patterns_slice);
    defer ac.deinit();

    const text = "this is a pattern in text pattern";
    var matches = try ac.findAll(text, std.testing.allocator);
    defer matches.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), matches.items.len);
    try std.testing.expectEqual(@as(usize, 10), matches.items[0].position);
    try std.testing.expectEqual(@as(usize, 26), matches.items[1].position);
}

test "AhoCorasick - no matches" {
    const AC = AhoCorasick(u8);
    const patterns = [_][]const u8{ "foo", "bar", "baz" };
    const patterns_slice: []const []const u8 = &patterns;

    var ac = try AC.init(std.testing.allocator, patterns_slice);
    defer ac.deinit();

    const text = "hello world";
    var matches = try ac.findAll(text, std.testing.allocator);
    defer matches.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 0), matches.items.len);
}

test "AhoCorasick - overlapping patterns" {
    const AC = AhoCorasick(u8);
    const patterns = [_][]const u8{ "abc", "bcd", "cde" };
    const patterns_slice: []const []const u8 = &patterns;

    var ac = try AC.init(std.testing.allocator, patterns_slice);
    defer ac.deinit();

    const text = "abcde";
    var matches = try ac.findAll(text, std.testing.allocator);
    defer matches.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 3), matches.items.len);
}

test "AhoCorasick - pattern at beginning" {
    const AC = AhoCorasick(u8);
    const patterns = [_][]const u8{"hello"};
    const patterns_slice: []const []const u8 = &patterns;

    var ac = try AC.init(std.testing.allocator, patterns_slice);
    defer ac.deinit();

    const text = "hello world";
    const match = ac.findFirst(text);
    try std.testing.expect(match != null);
    try std.testing.expectEqual(@as(usize, 0), match.?.position);
}

test "AhoCorasick - pattern at end" {
    const AC = AhoCorasick(u8);
    const patterns = [_][]const u8{"world"};
    const patterns_slice: []const []const u8 = &patterns;

    var ac = try AC.init(std.testing.allocator, patterns_slice);
    defer ac.deinit();

    const text = "hello world";
    const match = ac.findFirst(text);
    try std.testing.expect(match != null);
    try std.testing.expectEqual(@as(usize, 6), match.?.position);
}

test "AhoCorasick - contains" {
    const AC = AhoCorasick(u8);
    const patterns = [_][]const u8{ "needle", "hay" };
    const patterns_slice: []const []const u8 = &patterns;

    var ac = try AC.init(std.testing.allocator, patterns_slice);
    defer ac.deinit();

    try std.testing.expect(ac.contains("needle in haystack"));
    try std.testing.expect(!ac.contains("nothing here"));
}

test "AhoCorasick - count occurrences" {
    const AC = AhoCorasick(u8);
    const patterns = [_][]const u8{ "a", "aa" };
    const patterns_slice: []const []const u8 = &patterns;

    var ac = try AC.init(std.testing.allocator, patterns_slice);
    defer ac.deinit();

    const text = "aaaa";
    const total = ac.count(text);
    // "a" appears at 0,1,2,3 (4 times)
    // "aa" appears at 0,1,2 (3 times)
    // Total: 7
    try std.testing.expectEqual(@as(usize, 7), total);
}

test "AhoCorasick - empty text" {
    const AC = AhoCorasick(u8);
    const patterns = [_][]const u8{"test"};
    const patterns_slice: []const []const u8 = &patterns;

    var ac = try AC.init(std.testing.allocator, patterns_slice);
    defer ac.deinit();

    const text = "";
    var matches = try ac.findAll(text, std.testing.allocator);
    defer matches.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 0), matches.items.len);
}

test "AhoCorasick - error: empty pattern set" {
    const AC = AhoCorasick(u8);
    const patterns: []const []const u8 = &[_][]const u8{};

    const result = AC.init(std.testing.allocator, patterns);
    try std.testing.expectError(error.EmptyPatternSet, result);
}

test "AhoCorasick - error: empty pattern" {
    const AC = AhoCorasick(u8);
    const patterns = [_][]const u8{ "valid", "" };
    const patterns_slice: []const []const u8 = &patterns;

    const result = AC.init(std.testing.allocator, patterns_slice);
    try std.testing.expectError(error.EmptyPattern, result);
}

test "AhoCorasick - multiple repeated patterns" {
    const AC = AhoCorasick(u8);
    const patterns = [_][]const u8{ "ab", "ab", "ab" }; // Duplicate patterns
    const patterns_slice: []const []const u8 = &patterns;

    var ac = try AC.init(std.testing.allocator, patterns_slice);
    defer ac.deinit();

    const text = "xabx";
    var matches = try ac.findAll(text, std.testing.allocator);
    defer matches.deinit(std.testing.allocator);

    // Each duplicate pattern triggers a separate match
    try std.testing.expectEqual(@as(usize, 3), matches.items.len);
}

test "AhoCorasick - case sensitive" {
    const AC = AhoCorasick(u8);
    const patterns = [_][]const u8{"Hello"};
    const patterns_slice: []const []const u8 = &patterns;

    var ac = try AC.init(std.testing.allocator, patterns_slice);
    defer ac.deinit();

    try std.testing.expect(ac.contains("Hello world"));
    try std.testing.expect(!ac.contains("hello world"));
}

test "AhoCorasick - prefix patterns" {
    const AC = AhoCorasick(u8);
    const patterns = [_][]const u8{ "a", "ab", "abc" };
    const patterns_slice: []const []const u8 = &patterns;

    var ac = try AC.init(std.testing.allocator, patterns_slice);
    defer ac.deinit();

    const text = "abc";
    var matches = try ac.findAll(text, std.testing.allocator);
    defer matches.deinit(std.testing.allocator);

    // All three patterns match at overlapping positions
    try std.testing.expectEqual(@as(usize, 3), matches.items.len);
}

test "AhoCorasick - stress test" {
    const AC = AhoCorasick(u8);
    const patterns = [_][]const u8{ "test", "stress", "data", "structure" };
    const patterns_slice: []const []const u8 = &patterns;

    var ac = try AC.init(std.testing.allocator, patterns_slice);
    defer ac.deinit();

    // Build large text with repeated patterns
    var text_buf: [10000]u8 = undefined;
    var i: usize = 0;
    while (i < 10000) : (i += 5) {
        const remaining = 10000 - i;
        if (remaining >= 4) {
            @memcpy(text_buf[i .. i + 4], "test");
        } else {
            @memcpy(text_buf[i..10000], "test"[0..remaining]);
            break;
        }
    }

    const text = text_buf[0..10000];
    var matches = try ac.findAll(text, std.testing.allocator);
    defer matches.deinit(std.testing.allocator);

    // "test" appears approximately every 5 bytes, so ~2000 matches
    try std.testing.expect(matches.items.len > 1000);
}

test "AhoCorasick - unicode support" {
    const AC = AhoCorasick(u8);
    const patterns = [_][]const u8{ "こんにちは", "世界" };
    const patterns_slice: []const []const u8 = &patterns;

    var ac = try AC.init(std.testing.allocator, patterns_slice);
    defer ac.deinit();

    const text = "こんにちは世界";
    var matches = try ac.findAll(text, std.testing.allocator);
    defer matches.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), matches.items.len);
}
