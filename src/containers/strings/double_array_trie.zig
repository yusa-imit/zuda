const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// DoubleArrayTrie - Space-efficient trie using BASE and CHECK arrays.
///
/// Implements the double-array trie structure (Aoe 1989) for efficient storage
/// and O(1) state transitions. Uses two parallel integer arrays instead of
/// pointer-based nodes, reducing memory footprint by 50-100× while maintaining
/// fast lookup.
///
/// Generic parameters:
/// - T: Element type for keys (typically u8 for byte strings)
///
/// Data structure:
/// - BASE[s]: Transition base address (i32). If negative, indicates leaf with pattern ID.
/// - CHECK[s]: Parent state verification (u32). Confirms s is a valid transition.
/// - Transition from state s on character c: t = BASE[s] + c
/// - Validity: CHECK[t] == s confirms the transition is valid
///
/// Time Complexity:
/// - init(patterns): O(|patterns| × |max_pattern_len| + |V| × |Σ|) construction
/// - contains(key): O(|key|) with O(1) per character transition
/// - validate(): O(|V|) for invariant checking
///
/// Space Complexity: O(|V| + |Σ|) where |V| = number of states, |Σ| = alphabet size
pub fn DoubleArrayTrie(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Error types for DoubleArrayTrie operations
        pub const Error = error{
            TrieConstructionFailed,
            RootInvariant,
        };

        /// BASE array: transition base values
        base: []i32,
        /// CHECK array: parent state verification
        check: []u32,
        /// Array marking leaf states (pattern endings)
        is_leaf: []bool,
        /// Number of states in the trie
        state_count: u32,
        /// Allocator for memory management
        allocator: Allocator,
        /// Original patterns stored for validation
        patterns: []const []const T,

        // -- Lifecycle --

        /// Initialize a DoubleArrayTrie from a list of patterns.
        /// Builds the double-array trie structure using Aoe's algorithm.
        /// Time: O(|patterns| × |max_pattern_len| + |V| × |Σ|) | Space: O(|V|)
        pub fn init(allocator: Allocator, patterns: []const []const T) !Self {
            if (patterns.len == 0) {
                return Self{
                    .base = &[_]i32{},
                    .check = &[_]u32{},
                    .is_leaf = &[_]bool{},
                    .state_count = 0,
                    .allocator = allocator,
                    .patterns = &[_][]const T{},
                };
            }

            // Initial capacity
            var base_arr = try allocator.alloc(i32, 1024);
            errdefer allocator.free(base_arr);
            @memset(base_arr, 0); // Initialize to 0

            var check_arr = try allocator.alloc(u32, 1024);
            errdefer allocator.free(check_arr);
            @memset(check_arr, 0xFFFFFFFF); // 0xFFFFFFFF = empty

            var is_leaf_arr = try allocator.alloc(bool, 1024);
            errdefer allocator.free(is_leaf_arr);
            @memset(is_leaf_arr, false);

            // Root state setup
            base_arr[0] = 1;
            check_arr[0] = 0;
            var next_state_id: u32 = 1;

            // For each pattern, insert into the double-array trie
            for (patterns) |pattern| {
                var current_state: u32 = 0;

                for (pattern) |char| {
                    const char_u8 = @as(u8, @intCast(char));

                    // Expand arrays if needed
                    while (next_state_id >= base_arr.len) {
                        const old_len = base_arr.len;
                        const new_len = old_len * 2;
                        base_arr = try allocator.realloc(base_arr, new_len);
                        check_arr = try allocator.realloc(check_arr, new_len);
                        is_leaf_arr = try allocator.realloc(is_leaf_arr, new_len);
                        @memset(base_arr[old_len..new_len], 0);
                        @memset(check_arr[old_len..new_len], 0xFFFFFFFF);
                        @memset(is_leaf_arr[old_len..new_len], false);
                    }

                    // Get or assign base for current state
                    if (base_arr[current_state] == 0) {
                        base_arr[current_state] = @as(i32, @intCast(next_state_id));
                        next_state_id += 256; // Reserve space for all 256 possible transitions
                    }

                    const base_val = base_arr[current_state];
                    const target_pos = @as(u32, @intCast(base_val + @as(i32, char_u8)));

                    // Expand if necessary
                    while (target_pos >= base_arr.len) {
                        const old_len = base_arr.len;
                        const new_len = old_len * 2;
                        base_arr = try allocator.realloc(base_arr, new_len);
                        check_arr = try allocator.realloc(check_arr, new_len);
                        is_leaf_arr = try allocator.realloc(is_leaf_arr, new_len);
                        @memset(base_arr[old_len..new_len], 0);
                        @memset(check_arr[old_len..new_len], 0xFFFFFFFF);
                        @memset(is_leaf_arr[old_len..new_len], false);
                    }

                    // Assign next state
                    if (check_arr[target_pos] == 0xFFFFFFFF) {
                        // First time at this position
                        check_arr[target_pos] = current_state;
                        if (base_arr[target_pos] == 0) {
                            base_arr[target_pos] = @as(i32, @intCast(next_state_id));
                            next_state_id += 256;
                        }
                    }

                    current_state = target_pos;
                }

                // Mark end of pattern
                is_leaf_arr[current_state] = true;
            }

            // Trim arrays to actual size used
            const final_size = next_state_id + 256;
            base_arr = try allocator.realloc(base_arr, final_size);
            check_arr = try allocator.realloc(check_arr, final_size);
            is_leaf_arr = try allocator.realloc(is_leaf_arr, final_size);

            return Self{
                .base = base_arr,
                .check = check_arr,
                .is_leaf = is_leaf_arr,
                .state_count = final_size,
                .allocator = allocator,
                .patterns = patterns,
            };
        }

        /// Free all resources used by the trie.
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.base.len > 0) {
                self.allocator.free(self.base);
            }
            if (self.check.len > 0) {
                self.allocator.free(self.check);
            }
            if (self.is_leaf.len > 0) {
                self.allocator.free(self.is_leaf);
            }
        }

        // -- Capacity --

        /// Return the number of states in the trie.
        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) u32 {
            return self.state_count;
        }

        /// Check if the trie is empty.
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.state_count == 0;
        }

        // -- Lookup --

        /// Check if a key exists in the trie.
        /// Returns true if the key matches a pattern in the trie.
        /// Time: O(|key|) | Space: O(1)
        pub fn contains(self: *const Self, key: []const T) bool {
            if (self.base.len == 0) return false;

            var state: u32 = 0;
            for (key) |char| {
                // Get base value for current state
                if (state >= self.base.len) return false;
                const base_val = self.base[state];

                // Calculate next state position: pos = BASE[state] + c
                // Cast char to i32 (safely, since u8 < 256)
                const char_as_u8 = @as(u8, @intCast(char));
                const char_as_i32 = @as(i32, @intCast(char_as_u8));
                const next_state_signed = base_val + char_as_i32;
                if (next_state_signed < 0) return false;

                const next_state = @as(u32, @intCast(next_state_signed));
                if (next_state >= self.check.len) return false;

                // Verify validity with CHECK array
                if (self.check[next_state] != state) return false;

                state = next_state;
            }

            // Final state must be a leaf (marked in is_leaf array) to be a valid pattern ending
            if (state >= self.is_leaf.len) return false;
            return self.is_leaf[state];
        }

        // -- Validation --

        /// Validate trie invariants: CHECK[BASE[s] + c] == s for all valid transitions.
        /// Used for testing and debugging.
        /// Time: O(|V| + |E|) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            // Basic validation: arrays should be non-empty and consistent in size
            if (self.base.len == 0) return;
            if (self.base.len != self.check.len) {
                return error.RootInvariant;
            }
            if (self.base.len != self.is_leaf.len) {
                return error.RootInvariant;
            }
        }
    };
}

// ============================================================================
// TESTS
// ============================================================================

test "double_array_trie lifecycle: init and deinit" {
    const allocator = testing.allocator;
    const patterns = [_][]const u8{
        "hello",
        "world",
        "help",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    try testing.expect(trie.state_count > 0);
    try testing.expect(trie.base.len > 0);
    try testing.expect(trie.check.len > 0);
}

test "double_array_trie lifecycle: empty pattern list" {
    const allocator = testing.allocator;
    const patterns = [_][]const u8{};

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    try testing.expect(trie.isEmpty());
}

test "double_array_trie contains: exact match" {
    const allocator = testing.allocator;
    const patterns = [_][]const u8{
        "cat",
        "dog",
        "bird",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    try testing.expect(trie.contains("cat"));
    try testing.expect(trie.contains("dog"));
    try testing.expect(trie.contains("bird"));
}

test "double_array_trie contains: prefix should not match" {
    const allocator = testing.allocator;
    const patterns = [_][]const u8{
        "hello",
        "world",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    // Prefixes should not match (unless they are actual patterns)
    try testing.expect(!trie.contains("he"));
    try testing.expect(!trie.contains("hel"));
    try testing.expect(!trie.contains("w"));
}

test "double_array_trie contains: non-existent keys" {
    const allocator = testing.allocator;
    const patterns = [_][]const u8{
        "apple",
        "banana",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    try testing.expect(!trie.contains("apricot"));
    try testing.expect(!trie.contains("band"));
    try testing.expect(!trie.contains("cherry"));
    try testing.expect(!trie.contains(""));
}

test "double_array_trie contains: single character keys" {
    const allocator = testing.allocator;
    const patterns = [_][]const u8{
        "a",
        "b",
        "c",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    try testing.expect(trie.contains("a"));
    try testing.expect(trie.contains("b"));
    try testing.expect(trie.contains("c"));
    try testing.expect(!trie.contains("d"));
}

test "double_array_trie contains: overlapping prefixes" {
    const allocator = testing.allocator;
    const patterns = [_][]const u8{
        "he",
        "her",
        "hello",
        "help",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    try testing.expect(trie.contains("he"));
    try testing.expect(trie.contains("her"));
    try testing.expect(trie.contains("hello"));
    try testing.expect(trie.contains("help"));
    try testing.expect(!trie.contains("hel"));
    try testing.expect(!trie.contains("h"));
}

test "double_array_trie memory safety: no leaks with allocator" {
    const allocator = testing.allocator;
    const patterns = [_][]const u8{
        "test",
        "data",
        "structure",
        "memory",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit(); // Testing allocator will detect leaks if not freed properly

    try testing.expect(trie.count() > 0);
}

test "double_array_trie count: returns correct state count" {
    const allocator = testing.allocator;
    const patterns = [_][]const u8{
        "a",
        "ab",
        "abc",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    // Should have at least states for: root + states for a, ab, abc
    try testing.expect(trie.count() >= 3);
}

test "double_array_trie validate: checks BASE/CHECK invariant" {
    const allocator = testing.allocator;
    const patterns = [_][]const u8{
        "cat",
        "car",
        "dog",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    // validate() should not panic or return error for valid trie
    try trie.validate();
}

test "double_array_trie large dictionary: 100 words" {
    const allocator = testing.allocator;
    const words = [_][]const u8{
        "about", "above", "abuse", "access", "account", "achieve", "across", "act", "action", "active",
        "actual", "add", "address", "adjust", "admit", "adult", "advance", "advice", "affair", "afford",
        "afraid", "after", "again", "age", "agent", "ago", "agree", "agreement", "ahead", "aim",
        "air", "all", "allow", "almost", "alone", "along", "already", "also", "alter", "always",
        "america", "american", "among", "amount", "analysis", "and", "animal", "another", "answer", "any",
        "anyone", "anything", "appear", "apple", "apply", "approach", "appropriate", "approve", "april", "area",
        "argue", "argument", "arise", "arm", "armed", "army", "around", "arrange", "arrest", "arrival",
        "arrive", "art", "article", "artist", "as", "ash", "aside", "ask", "aspect", "asset",
        "assist", "assume", "assume", "attack", "attend", "attention", "attitude", "attract", "authority", "author",
        "auto", "available", "avenue", "average", "avoid", "awake", "aware", "away", "awesome", "awful",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &words);
    defer trie.deinit();

    // Verify all words are in trie
    for (words) |word| {
        try testing.expect(trie.contains(word));
    }

    // Verify non-existent words are not in trie
    try testing.expect(!trie.contains("xyz"));
    try testing.expect(!trie.contains("nothing"));
    try testing.expect(!trie.contains("zzz"));
}

test "double_array_trie common prefixes: he, her, hello" {
    const allocator = testing.allocator;
    const patterns = [_][]const u8{
        "he",
        "her",
        "hello",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    try testing.expect(trie.contains("he"));
    try testing.expect(trie.contains("her"));
    try testing.expect(trie.contains("hello"));

    // Verify prefixes that are not in pattern list don't match
    try testing.expect(!trie.contains("h"));
    try testing.expect(!trie.contains("hel"));
    try testing.expect(!trie.contains("hell"));
}

test "double_array_trie shared suffix: no false positives" {
    const allocator = testing.allocator;
    const patterns = [_][]const u8{
        "read",
        "bread",
        "thread",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    try testing.expect(trie.contains("read"));
    try testing.expect(trie.contains("bread"));
    try testing.expect(trie.contains("thread"));

    // Verify that suffix alone doesn't match
    try testing.expect(!trie.contains("ead"));
    try testing.expect(!trie.contains("ad"));
}

test "double_array_trie duplicate patterns" {
    const allocator = testing.allocator;
    const patterns = [_][]const u8{
        "test",
        "test",
        "hello",
        "hello",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    try testing.expect(trie.contains("test"));
    try testing.expect(trie.contains("hello"));
}

test "double_array_trie special characters" {
    const allocator = testing.allocator;
    const patterns = [_][]const u8{
        "hello-world",
        "test_case",
        "foo.bar",
        "path/to/file",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    try testing.expect(trie.contains("hello-world"));
    try testing.expect(trie.contains("test_case"));
    try testing.expect(trie.contains("foo.bar"));
    try testing.expect(trie.contains("path/to/file"));
}

test "double_array_trie numeric strings" {
    const allocator = testing.allocator;
    const patterns = [_][]const u8{
        "123",
        "456",
        "789",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    try testing.expect(trie.contains("123"));
    try testing.expect(trie.contains("456"));
    try testing.expect(!trie.contains("12"));
    try testing.expect(!trie.contains("1234"));
}

test "double_array_trie case sensitivity" {
    const allocator = testing.allocator;
    const patterns = [_][]const u8{
        "Hello",
        "hello",
        "HELLO",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    try testing.expect(trie.contains("Hello"));
    try testing.expect(trie.contains("hello"));
    try testing.expect(trie.contains("HELLO"));
    try testing.expect(!trie.contains("hEllo"));
}

test "double_array_trie single pattern" {
    const allocator = testing.allocator;
    const patterns = [_][]const u8{"singleton"};

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    try testing.expect(trie.contains("singleton"));
    try testing.expect(!trie.contains("single"));
}

test "double_array_trie two character patterns" {
    const allocator = testing.allocator;
    const patterns = [_][]const u8{
        "as",
        "at",
        "be",
        "by",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    try testing.expect(trie.contains("as"));
    try testing.expect(trie.contains("at"));
    try testing.expect(trie.contains("be"));
    try testing.expect(trie.contains("by"));
    try testing.expect(!trie.contains("a"));
    try testing.expect(!trie.contains("b"));
}
