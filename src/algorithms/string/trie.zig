/// Trie (Prefix Tree) — efficient string storage and prefix matching
///
/// A Trie is a tree-based data structure for storing strings where each node represents
/// a character, and paths from root to leaves represent complete strings. Optimized for:
/// - Prefix matching and autocomplete
/// - Dictionary operations
/// - Spell checking
/// - String interning
/// - IP routing tables
///
/// Time Complexity:
/// - insert(word): O(m) where m = word length
/// - search(word): O(m)
/// - startsWith(prefix): O(m)
/// - delete(word): O(m)
/// - getAllWordsWithPrefix(prefix): O(n + k*m) where n = nodes in subtree, k = words, m = avg length
/// - countWordsWithPrefix(prefix): O(n)
/// - longestCommonPrefix(): O(t) where t = total characters in trie
///
/// Space Complexity: O(ALPHABET_SIZE * N * M) where N = nodes, M = avg key length
///
/// References:
/// - Fredkin, Edward (1960). "Trie memory"
/// - Knuth, Donald (1973). "The Art of Computer Programming, Volume 3"
/// - Applications: Google autocomplete, spell checkers, routers
const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Trie node representing a character in the tree
pub const TrieNode = struct {
    /// Child nodes indexed by character (ASCII)
    children: [26]?*TrieNode,
    /// Marks if this node represents end of a word
    is_end_of_word: bool,
    /// Frequency count (how many times this word was inserted)
    count: usize,
    /// Allocator for memory management
    allocator: Allocator,

    /// Create a new TrieNode
    /// Time: O(1)
    /// Space: O(1)
    pub fn init(allocator: Allocator) !*TrieNode {
        const node = try allocator.create(TrieNode);
        node.* = TrieNode{
            .children = [_]?*TrieNode{null} ** 26,
            .is_end_of_word = false,
            .count = 0,
            .allocator = allocator,
        };
        return node;
    }

    /// Recursively free this node and all children
    /// Time: O(n) where n = nodes in subtree
    /// Space: O(h) recursion depth, h = height
    pub fn deinit(self: *TrieNode) void {
        for (self.children) |maybe_child| {
            if (maybe_child) |child| {
                child.deinit();
            }
        }
        self.allocator.destroy(self);
    }

    /// Get child index for a character (lowercase a-z)
    /// Time: O(1)
    /// Space: O(1)
    fn charIndex(c: u8) ?usize {
        if (c >= 'a' and c <= 'z') {
            return c - 'a';
        }
        return null;
    }
};

/// Trie (Prefix Tree) data structure
pub const Trie = struct {
    root: *TrieNode,
    allocator: Allocator,
    word_count: usize, // Total unique words
    total_insertions: usize, // Total insert operations (includes duplicates)

    /// Create a new Trie
    /// Time: O(1)
    /// Space: O(1)
    pub fn init(allocator: Allocator) !Trie {
        const root = try TrieNode.init(allocator);
        return Trie{
            .root = root,
            .allocator = allocator,
            .word_count = 0,
            .total_insertions = 0,
        };
    }

    /// Free all memory used by the Trie
    /// Time: O(n) where n = total nodes
    /// Space: O(h) recursion depth
    pub fn deinit(self: *Trie) void {
        self.root.deinit();
    }

    /// Insert a word into the Trie (lowercase a-z only)
    /// Returns true if word was newly inserted, false if already existed
    /// Time: O(m) where m = word length
    /// Space: O(m) worst case (all new nodes)
    pub fn insert(self: *Trie, word: []const u8) !bool {
        var current = self.root;

        for (word) |c| {
            const idx = TrieNode.charIndex(c) orelse return error.InvalidCharacter;

            if (current.children[idx] == null) {
                current.children[idx] = try TrieNode.init(self.allocator);
            }

            current = current.children[idx].?;
        }

        const was_new = !current.is_end_of_word;
        if (was_new) {
            self.word_count += 1;
        }

        current.is_end_of_word = true;
        current.count += 1;
        self.total_insertions += 1;

        return was_new;
    }

    /// Search for an exact word in the Trie
    /// Time: O(m) where m = word length
    /// Space: O(1)
    pub fn search(self: *Trie, word: []const u8) bool {
        const node = self.findNode(word);
        return if (node) |n| n.is_end_of_word else false;
    }

    /// Check if any word in the Trie starts with the given prefix
    /// Time: O(m) where m = prefix length
    /// Space: O(1)
    pub fn startsWith(self: *Trie, prefix: []const u8) bool {
        return self.findNode(prefix) != null;
    }

    /// Get the frequency count for a word
    /// Time: O(m) where m = word length
    /// Space: O(1)
    pub fn getCount(self: *Trie, word: []const u8) usize {
        const node = self.findNode(word);
        return if (node) |n| (if (n.is_end_of_word) n.count else 0) else 0;
    }

    /// Delete a word from the Trie
    /// Returns true if word was found and deleted, false otherwise
    /// Time: O(m) where m = word length
    /// Space: O(m) recursion depth
    pub fn delete(self: *Trie, word: []const u8) bool {
        const was_deleted = self.deleteHelper(self.root, word, 0);
        if (was_deleted) {
            self.word_count -= 1;
        }
        return was_deleted;
    }

    /// Helper for recursive deletion with node cleanup
    fn deleteHelper(self: *Trie, node: *TrieNode, word: []const u8, depth: usize) bool {
        if (depth == word.len) {
            if (!node.is_end_of_word) {
                return false; // Word not found
            }
            node.is_end_of_word = false;
            node.count = 0;

            // Check if node has no children (can be deleted)
            for (node.children) |child| {
                if (child != null) {
                    return true; // Has children, keep node
                }
            }
            return true;
        }

        const idx = TrieNode.charIndex(word[depth]) orelse return false;
        const child = node.children[idx] orelse return false;

        const should_delete_child = self.deleteHelper(child, word, depth + 1);

        if (should_delete_child) {
            // Check if child can be deleted (no children and not end of another word)
            var can_delete = !child.is_end_of_word;
            if (can_delete) {
                for (child.children) |grandchild| {
                    if (grandchild != null) {
                        can_delete = false;
                        break;
                    }
                }
            }

            if (can_delete) {
                child.deinit();
                node.children[idx] = null;
            }
        }

        return false; // Don't delete parent nodes during traversal
    }

    /// Get all words in the Trie that start with the given prefix
    /// Time: O(n + k*m) where n = nodes in subtree, k = words, m = avg length
    /// Space: O(k*m) for result ArrayList
    pub fn getAllWordsWithPrefix(self: *Trie, allocator: Allocator, prefix: []const u8) !std.ArrayList([]const u8) {
        var results = std.ArrayList([]const u8).init(allocator);
        errdefer {
            for (results.items) |item| {
                allocator.free(item);
            }
            results.deinit();
        }

        const prefix_node = self.findNode(prefix);
        if (prefix_node == null) {
            return results; // Empty list
        }

        var prefix_buf = std.ArrayList(u8).init(allocator);
        defer prefix_buf.deinit();

        try prefix_buf.appendSlice(prefix);
        try self.collectWords(prefix_node.?, &prefix_buf, &results, allocator);

        return results;
    }

    /// Collect all words in subtree (DFS)
    fn collectWords(self: *Trie, node: *TrieNode, current: *std.ArrayList(u8), results: *std.ArrayList([]const u8), allocator: Allocator) !void {
        _ = self;

        if (node.is_end_of_word) {
            const word = try allocator.dupe(u8, current.items);
            try results.append(word);
        }

        for (node.children, 0..) |maybe_child, i| {
            if (maybe_child) |child| {
                const char = @as(u8, @intCast(i)) + 'a';
                try current.append(char);
                try self.collectWords(child, current, results, allocator);
                _ = current.pop();
            }
        }
    }

    /// Count how many words in the Trie start with the given prefix
    /// Time: O(n) where n = nodes in subtree
    /// Space: O(h) recursion depth
    pub fn countWordsWithPrefix(self: *Trie, prefix: []const u8) usize {
        const prefix_node = self.findNode(prefix) orelse return 0;
        return self.countWordsInSubtree(prefix_node);
    }

    /// Count words in a subtree (DFS)
    fn countWordsInSubtree(self: *Trie, node: *TrieNode) usize {
        _ = self;
        var count: usize = if (node.is_end_of_word) 1 else 0;

        for (node.children) |maybe_child| {
            if (maybe_child) |child| {
                count += self.countWordsInSubtree(child);
            }
        }

        return count;
    }

    /// Find the longest common prefix of all words in the Trie
    /// Time: O(m) where m = length of longest common prefix
    /// Space: O(m) for result string
    pub fn longestCommonPrefix(self: *Trie, allocator: Allocator) ![]const u8 {
        var result = std.ArrayList(u8).init(allocator);
        errdefer result.deinit();

        var current = self.root;

        while (true) {
            // Count non-null children
            var child_count: usize = 0;
            var next_node: ?*TrieNode = null;
            var next_char: u8 = 0;

            for (current.children, 0..) |maybe_child, i| {
                if (maybe_child) |child| {
                    child_count += 1;
                    next_node = child;
                    next_char = @as(u8, @intCast(i)) + 'a';
                }
            }

            // Stop if: reached end of word, or more than one branch, or no children
            if (current.is_end_of_word or child_count != 1 or next_node == null) {
                break;
            }

            try result.append(next_char);
            current = next_node.?;
        }

        return result.toOwnedSlice();
    }

    /// Check if the Trie is empty (no words)
    /// Time: O(1)
    /// Space: O(1)
    pub fn isEmpty(self: *Trie) bool {
        return self.word_count == 0;
    }

    /// Get the number of unique words in the Trie
    /// Time: O(1)
    /// Space: O(1)
    pub fn size(self: *Trie) usize {
        return self.word_count;
    }

    /// Clear all words from the Trie
    /// Time: O(n) where n = total nodes
    /// Space: O(h) recursion depth
    pub fn clear(self: *Trie) !void {
        self.root.deinit();
        self.root = try TrieNode.init(self.allocator);
        self.word_count = 0;
        self.total_insertions = 0;
    }

    /// Find the node corresponding to a prefix/word
    /// Returns null if prefix doesn't exist
    /// Time: O(m) where m = prefix length
    /// Space: O(1)
    fn findNode(self: *Trie, prefix: []const u8) ?*TrieNode {
        var current = self.root;

        for (prefix) |c| {
            const idx = TrieNode.charIndex(c) orelse return null;
            current = current.children[idx] orelse return null;
        }

        return current;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "Trie: basic insert and search" {
    var trie = try Trie.init(testing.allocator);
    defer trie.deinit();

    try testing.expect(try trie.insert("hello"));
    try testing.expect(try trie.insert("world"));
    try testing.expect(!try trie.insert("hello")); // Duplicate

    try testing.expect(trie.search("hello"));
    try testing.expect(trie.search("world"));
    try testing.expect(!trie.search("hel"));
    try testing.expect(!trie.search("worlds"));
    try testing.expect(!trie.search(""));
}

test "Trie: prefix matching" {
    var trie = try Trie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("apple");
    _ = try trie.insert("app");
    _ = try trie.insert("application");

    try testing.expect(trie.startsWith("app"));
    try testing.expect(trie.startsWith("appl"));
    try testing.expect(!trie.startsWith("banana"));
    try testing.expect(!trie.startsWith("applications"));
}

test "Trie: delete operation" {
    var trie = try Trie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("hello");
    _ = try trie.insert("help");
    _ = try trie.insert("heap");

    try testing.expect(trie.search("hello"));
    try testing.expect(trie.delete("hello"));
    try testing.expect(!trie.search("hello"));
    try testing.expect(trie.search("help")); // Still exists
    try testing.expect(trie.search("heap")); // Still exists

    try testing.expect(!trie.delete("hello")); // Already deleted
    try testing.expect(!trie.delete("world")); // Never existed
}

test "Trie: getAllWordsWithPrefix" {
    var trie = try Trie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("cat");
    _ = try trie.insert("cats");
    _ = try trie.insert("caterpillar");
    _ = try trie.insert("dog");

    var words = try trie.getAllWordsWithPrefix(testing.allocator, "cat");
    defer {
        for (words.items) |word| {
            testing.allocator.free(word);
        }
        words.deinit();
    }

    try testing.expectEqual(@as(usize, 3), words.items.len);
    try testing.expectEqualStrings("cat", words.items[0]);
    try testing.expectEqualStrings("caterpillar", words.items[1]);
    try testing.expectEqualStrings("cats", words.items[2]);
}

test "Trie: countWordsWithPrefix" {
    var trie = try Trie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("test");
    _ = try trie.insert("testing");
    _ = try trie.insert("tester");
    _ = try trie.insert("tested");
    _ = try trie.insert("team");

    try testing.expectEqual(@as(usize, 4), trie.countWordsWithPrefix("test"));
    try testing.expectEqual(@as(usize, 5), trie.countWordsWithPrefix("te"));
    try testing.expectEqual(@as(usize, 0), trie.countWordsWithPrefix("toast"));
}

test "Trie: longestCommonPrefix" {
    var trie = try Trie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("flower");
    _ = try trie.insert("flow");
    _ = try trie.insert("flight");

    const lcp = try trie.longestCommonPrefix(testing.allocator);
    defer testing.allocator.free(lcp);

    try testing.expectEqualStrings("fl", lcp);
}

test "Trie: longestCommonPrefix with no common prefix" {
    var trie = try Trie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("dog");
    _ = try trie.insert("cat");
    _ = try trie.insert("bird");

    const lcp = try trie.longestCommonPrefix(testing.allocator);
    defer testing.allocator.free(lcp);

    try testing.expectEqualStrings("", lcp);
}

test "Trie: frequency count" {
    var trie = try Trie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("hello");
    _ = try trie.insert("hello");
    _ = try trie.insert("hello");
    _ = try trie.insert("world");

    try testing.expectEqual(@as(usize, 3), trie.getCount("hello"));
    try testing.expectEqual(@as(usize, 1), trie.getCount("world"));
    try testing.expectEqual(@as(usize, 0), trie.getCount("hel"));
}

test "Trie: isEmpty and size" {
    var trie = try Trie.init(testing.allocator);
    defer trie.deinit();

    try testing.expect(trie.isEmpty());
    try testing.expectEqual(@as(usize, 0), trie.size());

    _ = try trie.insert("hello");
    try testing.expect(!trie.isEmpty());
    try testing.expectEqual(@as(usize, 1), trie.size());

    _ = try trie.insert("world");
    try testing.expectEqual(@as(usize, 2), trie.size());

    _ = try trie.delete("hello");
    try testing.expectEqual(@as(usize, 1), trie.size());
}

test "Trie: clear operation" {
    var trie = try Trie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("apple");
    _ = try trie.insert("banana");
    _ = try trie.insert("cherry");

    try testing.expectEqual(@as(usize, 3), trie.size());

    try trie.clear();

    try testing.expect(trie.isEmpty());
    try testing.expectEqual(@as(usize, 0), trie.size());
    try testing.expect(!trie.search("apple"));
}

test "Trie: empty string handling" {
    var trie = try Trie.init(testing.allocator);
    defer trie.deinit();

    // Empty string should work
    try testing.expect(try trie.insert(""));
    try testing.expect(trie.search(""));
    try testing.expect(trie.startsWith(""));
}

test "Trie: single character words" {
    var trie = try Trie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("a");
    _ = try trie.insert("i");

    try testing.expect(trie.search("a"));
    try testing.expect(trie.search("i"));
    try testing.expect(!trie.search("b"));
}

test "Trie: invalid characters" {
    var trie = try Trie.init(testing.allocator);
    defer trie.deinit();

    try testing.expectError(error.InvalidCharacter, trie.insert("Hello")); // Uppercase
    try testing.expectError(error.InvalidCharacter, trie.insert("hello!")); // Punctuation
    try testing.expectError(error.InvalidCharacter, trie.insert("hello123")); // Numbers
}

test "Trie: getAllWordsWithPrefix with no matches" {
    var trie = try Trie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("apple");
    _ = try trie.insert("banana");

    var words = try trie.getAllWordsWithPrefix(testing.allocator, "cherry");
    defer words.deinit();

    try testing.expectEqual(@as(usize, 0), words.items.len);
}

test "Trie: delete with shared prefixes" {
    var trie = try Trie.init(testing.allocator);
    defer trie.deinit();

    _ = try trie.insert("abc");
    _ = try trie.insert("abcd");
    _ = try trie.insert("ab");

    try testing.expect(trie.delete("abcd"));
    try testing.expect(!trie.search("abcd"));
    try testing.expect(trie.search("abc")); // Should still exist
    try testing.expect(trie.search("ab")); // Should still exist

    try testing.expect(trie.delete("ab"));
    try testing.expect(!trie.search("ab"));
    try testing.expect(trie.search("abc")); // Should still exist
}

test "Trie: large dataset stress test" {
    var trie = try Trie.init(testing.allocator);
    defer trie.deinit();

    // Insert many words
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        var buf: [20]u8 = undefined;
        const word = try std.fmt.bufPrint(&buf, "word{d}", .{i});
        _ = try trie.insert(word);
    }

    try testing.expectEqual(@as(usize, 100), trie.size());

    // Verify all words exist
    i = 0;
    while (i < 100) : (i += 1) {
        var buf: [20]u8 = undefined;
        const word = try std.fmt.bufPrint(&buf, "word{d}", .{i});
        try testing.expect(trie.search(word));
    }

    // Count prefix matches
    try testing.expectEqual(@as(usize, 100), trie.countWordsWithPrefix("word"));
}

test "Trie: memory safety with allocator verification" {
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var trie = try Trie.init(testing.allocator);
        defer trie.deinit();

        _ = try trie.insert("memory");
        _ = try trie.insert("safe");
        _ = try trie.insert("test");

        var words = try trie.getAllWordsWithPrefix(testing.allocator, "mem");
        defer {
            for (words.items) |word| {
                testing.allocator.free(word);
            }
            words.deinit();
        }
    }
}
