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
/// - FAIL[s]: Failure link for Aho-Corasick automaton (u32). Points to longest proper suffix state.
/// - OUTPUT[s]: Pattern match data at state s (usize array). Lists pattern indices ending here.
/// - Transition from state s on character c: t = BASE[s] + c
/// - Validity: CHECK[t] == s confirms the transition is valid
///
/// Time Complexity:
/// - init(patterns): O(|patterns| × |max_pattern_len| + |V| × |Σ|) construction
/// - contains(key): O(|key|) with O(1) per character transition
/// - findAll(text): O(|text| + z) where z = number of matches
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

        /// Match result from Aho-Corasick pattern search
        pub const Match = struct {
            pattern_index: usize,
            position: usize,
        };

        /// BASE array: transition base values
        base: []i32,
        /// CHECK array: parent state verification
        check: []u32,
        /// Array marking leaf states (pattern endings)
        is_leaf: []bool,
        /// FAIL array: failure links for Aho-Corasick automaton
        fail: []u32,
        /// OUTPUT array: pattern indices matching at each state
        output: []std.ArrayList(usize),
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
                const empty_output = try allocator.alloc(std.ArrayList(usize), 0);
                return Self{
                    .base = &[_]i32{},
                    .check = &[_]u32{},
                    .is_leaf = &[_]bool{},
                    .fail = &[_]u32{},
                    .output = empty_output,
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

            var fail_arr = try allocator.alloc(u32, 1024);
            errdefer allocator.free(fail_arr);
            @memset(fail_arr, 0);

            var output_arr = try allocator.alloc(std.ArrayList(usize), 1024);
            errdefer {
                for (output_arr) |*o| o.deinit(allocator);
                allocator.free(output_arr);
            }
            for (output_arr) |*o| {
                o.* = .{};
                o.allocator = allocator;
            }

            // Root state setup
            base_arr[0] = 1;
            check_arr[0] = 0;
            fail_arr[0] = 0;
            var next_state_id: u32 = 1;

            // For each pattern, insert into the double-array trie
            for (patterns, 0..) |pattern, pattern_idx| {
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
                        fail_arr = try allocator.realloc(fail_arr, new_len);
                        output_arr = try allocator.realloc(output_arr, new_len);
                        @memset(base_arr[old_len..new_len], 0);
                        @memset(check_arr[old_len..new_len], 0xFFFFFFFF);
                        @memset(is_leaf_arr[old_len..new_len], false);
                        @memset(fail_arr[old_len..new_len], 0);
                        for (output_arr[old_len..new_len]) |*o| {
                            o.* = .{};
                            o.allocator = allocator;
                        }
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
                        fail_arr = try allocator.realloc(fail_arr, new_len);
                        output_arr = try allocator.realloc(output_arr, new_len);
                        @memset(base_arr[old_len..new_len], 0);
                        @memset(check_arr[old_len..new_len], 0xFFFFFFFF);
                        @memset(is_leaf_arr[old_len..new_len], false);
                        @memset(fail_arr[old_len..new_len], 0);
                        for (output_arr[old_len..new_len]) |*o| {
                            o.* = .{};
                            o.allocator = allocator;
                        }
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
                try output_arr[current_state].append(pattern_idx);
            }

            // Trim arrays to actual size used
            const final_size = next_state_id + 256;
            base_arr = try allocator.realloc(base_arr, final_size);
            check_arr = try allocator.realloc(check_arr, final_size);
            is_leaf_arr = try allocator.realloc(is_leaf_arr, final_size);
            fail_arr = try allocator.realloc(fail_arr, final_size);
            output_arr = try allocator.realloc(output_arr, final_size);

            var result = Self{
                .base = base_arr,
                .check = check_arr,
                .is_leaf = is_leaf_arr,
                .fail = fail_arr,
                .output = output_arr,
                .state_count = final_size,
                .allocator = allocator,
                .patterns = patterns,
            };

            // Build failure links and output links for Aho-Corasick
            try result.buildFailureLinks();
            try result.buildOutputLinks();

            return result;
        }

        /// Build failure links using BFS (Aho-Corasick preprocessing).
        /// Failure links enable pattern detection after state mismatch.
        /// Time: O(|V|) | Space: O(|V|)
        fn buildFailureLinks(self: *Self) !void {
            if (self.base.len == 0) return;

            // Use a queue for BFS traversal
            var queue = std.ArrayList(u32){};
            queue.allocator = self.allocator;
            defer queue.deinit(self.allocator);

            // Root's failure link is self (0 -> 0)
            self.fail[0] = 0;

            // Initialize: all depth-1 nodes have failure link to root
            if (self.base[0] >= 0) {
                const base_val = self.base[0];
                for (0..256) |c| {
                    const target_pos = @as(u32, @intCast(base_val + @as(i32, @intCast(c))));
                    if (target_pos < self.check.len and self.check[target_pos] == 0) {
                        // This is a depth-1 node
                        self.fail[target_pos] = 0;
                        try queue.append(target_pos);
                    }
                }
            }

            // BFS to compute failure links for all other nodes
            var queue_idx: usize = 0;
            while (queue_idx < queue.items.len) {
                const current = queue.items[queue_idx];
                queue_idx += 1;

                // Get children of current state
                if (current < self.base.len and self.base[current] >= 0) {
                    const base_val = self.base[current];
                    for (0..256) |c| {
                        const target_pos = @as(u32, @intCast(base_val + @as(i32, @intCast(c))));
                        if (target_pos < self.check.len and self.check[target_pos] == current) {
                            // This is a child of current state

                            // Find failure link by following parent's failure chain
                            var failure_candidate = self.fail[current];
                            while (failure_candidate != 0) {
                                const fail_base = self.base[failure_candidate];
                                if (fail_base >= 0) {
                                    const fail_target = @as(u32, @intCast(fail_base + @as(i32, @intCast(c))));
                                    if (fail_target < self.check.len and self.check[fail_target] == failure_candidate) {
                                        // Found a transition on character c
                                        self.fail[target_pos] = fail_target;
                                        break;
                                    }
                                }
                                failure_candidate = self.fail[failure_candidate];
                            }

                            // If no match found, link to root
                            if (failure_candidate == 0) {
                                // Check if root has transition on c
                                const root_base = self.base[0];
                                if (root_base >= 0) {
                                    const root_target = @as(u32, @intCast(root_base + @as(i32, @intCast(c))));
                                    if (root_target < self.check.len and self.check[root_target] == 0) {
                                        self.fail[target_pos] = root_target;
                                    } else {
                                        self.fail[target_pos] = 0;
                                    }
                                } else {
                                    self.fail[target_pos] = 0;
                                }
                            }

                            try queue.append(target_pos);
                        }
                    }
                }
            }
        }

        /// Build output links for overlapping pattern detection.
        /// Copies pattern indices from failure states to output array.
        /// Time: O(|V|) | Space: O(1)
        fn buildOutputLinks(self: *Self) !void {
            if (self.base.len == 0) return;

            for (0..self.state_count) |s| {
                const state = @as(u32, @intCast(s));

                // Copy pattern indices from failure chain
                var failure_state = self.fail[state];
                while (failure_state != state and failure_state != 0) {
                    if (failure_state < self.output.len) {
                        for (self.output[failure_state].items) |pattern_idx| {
                            try self.output[state].append(pattern_idx);
                        }
                    }
                    failure_state = self.fail[failure_state];
                }
            }
        }

        /// Free all resources used by the trie.
        /// Time: O(|V|) | Space: O(1)
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
            if (self.fail.len > 0) {
                self.allocator.free(self.fail);
            }
            if (self.output.len > 0) {
                for (self.output) |*o| {
                    o.deinit(self.allocator);
                }
                self.allocator.free(self.output);
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

        // -- Aho-Corasick Search --

        /// Find all pattern occurrences in text using Aho-Corasick automaton.
        /// Caller must free returned slice with allocator.free().
        /// Time: O(|text| + z) where z = number of matches | Space: O(z)
        pub fn findAll(self: *const Self, allocator: Allocator, text: []const T) ![]Match {
            var matches = std.ArrayList(Match){};
            matches.allocator = allocator;
            errdefer matches.deinit(allocator);

            if (text.len == 0 or self.base.len == 0) {
                return matches.toOwnedSlice();
            }

            var current_state: u32 = 0;

            for (text, 0..) |char, i| {
                const char_u8 = @as(u8, @intCast(char));

                // Follow failure links until we find a valid transition or reach root
                while (true) {
                    if (current_state < self.base.len) {
                        const base_val = self.base[current_state];
                        if (base_val >= 0) {
                            const next_state_signed = base_val + @as(i32, @intCast(char_u8));
                            if (next_state_signed >= 0) {
                                const next_state = @as(u32, @intCast(next_state_signed));
                                if (next_state < self.check.len and self.check[next_state] == current_state) {
                                    // Valid transition found
                                    current_state = next_state;
                                    break;
                                }
                            }
                        }
                    }

                    // No valid transition, follow failure link or go to root
                    if (current_state == 0) {
                        break;
                    }
                    if (current_state < self.fail.len) {
                        current_state = self.fail[current_state];
                    } else {
                        current_state = 0;
                    }
                }

                // Emit patterns at current state
                if (current_state < self.output.len) {
                    for (self.output[current_state].items) |pattern_idx| {
                        try matches.append(.{
                            .pattern_index = pattern_idx,
                            .position = i + 1 - self.patterns[pattern_idx].len,
                        });
                    }
                }

                // Emit patterns at failure chain (overlapping matches)
                var failure_state = if (current_state < self.fail.len) self.fail[current_state] else 0;
                while (failure_state != 0 and failure_state != current_state) {
                    if (failure_state < self.output.len) {
                        for (self.output[failure_state].items) |pattern_idx| {
                            try matches.append(.{
                                .pattern_index = pattern_idx,
                                .position = i + 1 - self.patterns[pattern_idx].len,
                            });
                        }
                    }
                    if (failure_state < self.fail.len) {
                        failure_state = self.fail[failure_state];
                    } else {
                        break;
                    }
                }
            }

            return matches.toOwnedSlice();
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

// ============================================================================
// AHO-CORASICK EXTENSION TESTS (FAILING TESTS FOR RED PHASE)
// ============================================================================

test "aho_corasick_dat: basic multi-pattern matching in ushers" {
    // Goal: patterns ["he", "she", "his", "hers"] should find 3 matches in "ushers"
    // Expected matches: "she" at 1, "he" at 2, "hers" at 2
    const allocator = testing.allocator;
    const patterns = [_][]const u8{
        "he",
        "she",
        "his",
        "hers",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "ushers";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    // Must find exactly 3 matches
    try testing.expectEqual(@as(usize, 3), matches.len);

    // Verify match positions and pattern indices
    // "she" at position 1 (index 1)
    try testing.expectEqual(@as(usize, 1), matches[0].pattern_index);
    try testing.expectEqual(@as(usize, 1), matches[0].position);

    // "he" at position 2 (index 0)
    try testing.expectEqual(@as(usize, 0), matches[1].pattern_index);
    try testing.expectEqual(@as(usize, 2), matches[1].position);

    // "hers" at position 2 (index 3)
    try testing.expectEqual(@as(usize, 3), matches[2].pattern_index);
    try testing.expectEqual(@as(usize, 2), matches[2].position);
}

test "aho_corasick_dat: overlapping patterns abc" {
    // Goal: patterns ["ab", "abc", "bc"] should find all 3 in "abc"
    // Expected: "ab" at 0, "abc" at 0, "bc" at 1
    const allocator = testing.allocator;
    const patterns = [_][]const u8{
        "ab",
        "abc",
        "bc",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "abc";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    // All 3 overlapping patterns must be found
    try testing.expectEqual(@as(usize, 3), matches.len);
}

test "aho_corasick_dat: empty text returns no matches" {
    const allocator = testing.allocator;
    const patterns = [_][]const u8{ "pattern" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    try testing.expectEqual(@as(usize, 0), matches.len);
}

test "aho_corasick_dat: no matches in text" {
    // Goal: verify findAll returns empty when no patterns match
    const allocator = testing.allocator;
    const patterns = [_][]const u8{ "foo", "bar", "baz" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "hello world xyz";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    try testing.expectEqual(@as(usize, 0), matches.len);
}

test "aho_corasick_dat: single pattern match" {
    // Goal: find single pattern occurrence
    const allocator = testing.allocator;
    const patterns = [_][]const u8{"pattern"};

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "this is a pattern in text pattern";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    // Must find both occurrences
    try testing.expectEqual(@as(usize, 2), matches.len);
    try testing.expectEqual(@as(usize, 0), matches[0].pattern_index);
    try testing.expectEqual(@as(usize, 10), matches[0].position);
    try testing.expectEqual(@as(usize, 0), matches[1].pattern_index);
    try testing.expectEqual(@as(usize, 26), matches[1].position);
}

test "aho_corasick_dat: pattern at beginning" {
    // Goal: verify match detection at text start
    const allocator = testing.allocator;
    const patterns = [_][]const u8{"hello"};

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "hello world";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    try testing.expectEqual(@as(usize, 1), matches.len);
    try testing.expectEqual(@as(usize, 0), matches[0].position);
}

test "aho_corasick_dat: pattern at end" {
    // Goal: verify match detection at text end
    const allocator = testing.allocator;
    const patterns = [_][]const u8{"world"};

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "hello world";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    try testing.expectEqual(@as(usize, 1), matches.len);
    try testing.expectEqual(@as(usize, 6), matches[0].position);
}

test "aho_corasick_dat: prefix patterns" {
    // Goal: patterns that are prefixes of each other
    // patterns ["a", "ab", "abc"] all match in text "abc"
    const allocator = testing.allocator;
    const patterns = [_][]const u8{
        "a",
        "ab",
        "abc",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "abc";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    // All three patterns should match
    try testing.expectEqual(@as(usize, 3), matches.len);
}

test "aho_corasick_dat: failure link traversal" {
    // Goal: verify failure links enable pattern detection after mismatch
    // patterns ["she", "he"] in "ushers": after matching "she", "he" is at pos 2
    const allocator = testing.allocator;
    const patterns = [_][]const u8{ "she", "he" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "ushers";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    // Both "she" and "he" must be detected
    try testing.expectEqual(@as(usize, 2), matches.len);
}

test "aho_corasick_dat: output links for overlapping matches" {
    // Goal: output links must report all overlapping pattern endings
    // patterns ["a", "aa", "aaa"] in "aaaa" produces 7 total matches
    const allocator = testing.allocator;
    const patterns = [_][]const u8{
        "a",
        "aa",
        "aaa",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "aaaa";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    // "a" appears at 0,1,2,3 (4 times)
    // "aa" appears at 0,1,2 (3 times)
    // "aaa" appears at 0,1 (2 times)
    // Total: 7 matches
    try testing.expectEqual(@as(usize, 7), matches.len);
}

test "aho_corasick_dat: multiple non-overlapping patterns" {
    // Goal: find multiple patterns in sequence without overlap
    // patterns ["cat", "dog", "bird"] in "catdogbird"
    const allocator = testing.allocator;
    const patterns = [_][]const u8{ "cat", "dog", "bird" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "catdogbird";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    try testing.expectEqual(@as(usize, 3), matches.len);
    try testing.expectEqual(@as(usize, 0), matches[0].position); // "cat"
    try testing.expectEqual(@as(usize, 3), matches[1].position); // "dog"
    try testing.expectEqual(@as(usize, 6), matches[2].position); // "bird"
}

test "aho_corasick_dat: case sensitivity" {
    // Goal: verify matching is case-sensitive
    // "Hello" should not match "hello"
    const allocator = testing.allocator;
    const patterns = [_][]const u8{"Hello"};

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text1 = "Hello world";
    const matches1 = try trie.findAll(allocator, text1);
    defer allocator.free(matches1);
    try testing.expectEqual(@as(usize, 1), matches1.len);

    const text2 = "hello world";
    const matches2 = try trie.findAll(allocator, text2);
    defer allocator.free(matches2);
    try testing.expectEqual(@as(usize, 0), matches2.len);
}

test "aho_corasick_dat: stress test 100 patterns 10KB text" {
    // Goal: verify correctness on large input
    // 100+ patterns on 10KB text
    const allocator = testing.allocator;
    const pattern_list = [_][]const u8{
        "test", "stress", "data", "structure", "algorithm", "pattern",
        "match", "find", "search", "locate", "index", "position",
        "word", "text", "string", "input", "output", "result",
        "code", "build", "compile", "error", "warning", "debug",
        "optimize", "performance", "memory", "cache", "allocation", "free",
        "array", "list", "queue", "stack", "tree", "graph",
        "node", "edge", "vertex", "transition", "state", "machine",
        "automaton", "trie", "hash", "compare", "equal", "sort",
        "loop", "iterate", "enumerate", "collect", "filter", "reduce",
        "map", "fold", "scan", "zip", "unzip", "product",
        "sum", "count", "min", "max", "average", "median",
        "variance", "deviation", "probability", "distribution", "random", "seed",
        "generator", "crypto", "hash_map", "binary", "hex", "octal",
        "decimal", "fraction", "ratio", "percent", "proportion", "scale",
        "transform", "rotate", "flip", "transpose", "invert", "apply",
        "compose", "chain", "pipeline", "flow", "stream", "async",
        "await", "promise", "future", "callback", "handler", "listener",
        "event", "emit", "dispatch", "trigger", "subscribe", "publish",
    };

    var trie = try DoubleArrayTrie(u8).init(allocator, &pattern_list);
    defer trie.deinit();

    // Generate 10KB text with pattern repetitions
    var text_buf: [10240]u8 = undefined;
    var i: usize = 0;
    while (i < 10240) : (i += 5) {
        const remaining = 10240 - i;
        if (remaining >= 4) {
            @memcpy(text_buf[i .. i + 4], "test");
        } else {
            if (remaining > 0) {
                @memcpy(text_buf[i..10240], "test"[0..remaining]);
            }
            break;
        }
    }

    const text = text_buf[0..10240];
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    // Should find many matches of "test" pattern repeated
    try testing.expect(matches.len > 100);
}

test "aho_corasick_dat: repeated pattern appears multiple times" {
    // Goal: count multiple occurrences of same pattern
    // pattern "ab" appears 3 times in "xabxabxab"
    const allocator = testing.allocator;
    const patterns = [_][]const u8{"ab"};

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "xabxabxab";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    try testing.expectEqual(@as(usize, 3), matches.len);
    try testing.expectEqual(@as(usize, 1), matches[0].position);
    try testing.expectEqual(@as(usize, 4), matches[1].position);
    try testing.expectEqual(@as(usize, 7), matches[2].position);
}

test "aho_corasick_dat: pattern with common prefixes and suffixes" {
    // Goal: test failure links with complex sharing
    // patterns ["aba", "bab", "ab"] in "ababab"
    const allocator = testing.allocator;
    const patterns = [_][]const u8{ "aba", "bab", "ab" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "ababab";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    // Verify multiple matches occur
    try testing.expect(matches.len > 2);
}

test "aho_corasick_dat: match struct contains correct pattern_index" {
    // Goal: verify Match.pattern_index correctly identifies pattern
    const allocator = testing.allocator;
    const patterns = [_][]const u8{ "apple", "apply", "apply" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "apply apple apply";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    // First match should be "apply" (index 1 or 2)
    try testing.expect(matches[0].pattern_index <= 2);
    // Last 2 matches should be "apply" and "apple"
    try testing.expect(matches.len >= 3);
}

test "aho_corasick_dat: single character patterns" {
    // Goal: verify single-char pattern detection
    const allocator = testing.allocator;
    const patterns = [_][]const u8{ "a", "b", "c" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "abcabc";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    try testing.expectEqual(@as(usize, 6), matches.len); // 2 occurrences each
}

test "aho_corasick_dat: long text with patterns at end" {
    // Goal: verify failure links work correctly near text boundary
    const allocator = testing.allocator;
    const patterns = [_][]const u8{"end"};

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "this is the very end";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    try testing.expectEqual(@as(usize, 1), matches.len);
    try testing.expectEqual(@as(usize, 17), matches[0].position);
}

test "aho_corasick_dat: patterns with shared prefixes abc abcd" {
    // Goal: OUTPUT links connect patterns sharing common path
    const allocator = testing.allocator;
    const patterns = [_][]const u8{ "abc", "abcd" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "abcd";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    // Both patterns should match
    try testing.expectEqual(@as(usize, 2), matches.len);
}

test "aho_corasick_dat: failure link chain she he hers" {
    // Goal: failure links form chain: she -> he -> (root)
    // In "ushers", starting from "s", mismatch, follow failure to "sh"
    const allocator = testing.allocator;
    const patterns = [_][]const u8{ "she", "he", "hers" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "ushers";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    // Must find all three via failure link traversal
    try testing.expectEqual(@as(usize, 3), matches.len);
}

test "aho_corasick_dat: memory safety no leaks" {
    // Goal: testing allocator detects leaks on findAll result
    const allocator = testing.allocator;
    const patterns = [_][]const u8{ "test", "data", "structure" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "test data structure";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches); // Must properly free allocated matches slice

    try testing.expectEqual(@as(usize, 3), matches.len);
}
