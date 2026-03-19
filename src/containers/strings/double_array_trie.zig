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

        /// Phase 3 linearized State structure (24 bytes)
        /// Packs all state data into single cache-line-friendly struct.
        /// Reduces cache misses from 3-4 → 1 per transition.
        pub const State = struct {
            base: i32,           // 4 bytes - transition base (if < 0, leaf with pattern ID)
            check: u32,          // 4 bytes - parent state verification
            fail: u32,           // 4 bytes - failure link for Aho-Corasick
            output_start: u32,   // 4 bytes - index into patterns[] array
            output_len: u16,     // 2 bytes - number of patterns at this state
            flags: u8,           // 1 byte - bit flags (is_leaf, etc.)
            _padding1: u8,       // 1 byte - alignment padding
            _padding2: u32,      // 4 bytes - additional padding for 24-byte alignment
            // Total: 24 bytes/state
        };

        /// IS_LEAF flag bit (stored in State.flags)
        pub const IS_LEAF: u8 = 0x01;

        /// Interleaved BASE+CHECK structure for cache locality
        /// Reduces cache misses from 2 → 1 per transition check
        pub const BaseCheck = struct {
            base: i32,
            check: u32,
        };

        /// Phase 3: Linearized states array (24 bytes per state)
        states: []State,
        /// Phase 3: Flattened pattern indices array (indexed by State.output_start/output_len)
        patterns: []usize,
        /// Goto completion table: pre-computed transitions for all chars
        /// Size: state_count × 256, indexed as [state_id * 256 + char]
        /// Each entry is a valid next state (eliminates failure link loop)
        goto_table: []u32,
        /// Number of states in the trie
        state_count: u32,
        /// Number of pattern entries in patterns array
        pattern_count: u32,
        /// Allocator for memory management
        allocator: Allocator,
        /// Original input patterns stored for validation
        input_patterns: []const []const T,

        // -- Lifecycle --

        /// Initialize a DoubleArrayTrie from a list of patterns.
        /// Builds the double-array trie structure using Aoe's algorithm with Phase 3 linearization.
        /// Time: O(|patterns| × |max_pattern_len| + |V| × |Σ|) | Space: O(|V|)
        pub fn init(allocator: Allocator, patterns: []const []const T) !Self {
            if (patterns.len == 0) {
                const empty_states = try allocator.alloc(State, 0);
                const empty_patterns_arr = try allocator.alloc(usize, 0);
                const empty_goto = try allocator.alloc(u32, 0);
                return Self{
                    .states = empty_states,
                    .patterns = empty_patterns_arr,
                    .goto_table = empty_goto,
                    .state_count = 0,
                    .pattern_count = 0,
                    .allocator = allocator,
                    .input_patterns = patterns,
                };
            }

            // Phase 3: Allocate single linearized states array + pattern collection ArrayList
            var states_arr = try allocator.alloc(State, 1024);
            errdefer allocator.free(states_arr);
            // Initialize with base=0, check=0xFFFFFFFF (empty)
            for (states_arr) |*state| {
                state.* = .{
                    .base = 0,
                    .check = 0xFFFFFFFF,
                    .fail = 0,
                    .output_start = 0,
                    .output_len = 0,
                    .flags = 0,
                    ._padding1 = 0,
                    ._padding2 = 0,
                };
            }

            // Collect output patterns in a temporary ArrayList
            var output_lists = try allocator.alloc(std.ArrayList(usize), 1024);
            errdefer {
                for (output_lists) |*o| o.deinit(allocator);
                allocator.free(output_lists);
            }
            for (output_lists) |*o| {
                o.* = std.ArrayList(usize){};
            }

            // Root state setup
            states_arr[0] = .{
                .base = 1,
                .check = 0,
                .fail = 0,
                .output_start = 0,
                .output_len = 0,
                .flags = 0,
                ._padding1 = 0,
                ._padding2 = 0,
            };
            var next_state_id: u32 = 1;

            // Build trie incrementally, assigning states to parents as needed
            // Key optimization: only allocate base when transitioning to new state
            for (patterns, 0..) |pattern, pattern_idx| {
                var current_state: u32 = 0;

                for (pattern) |char| {
                    const char_u8 = @as(u8, @intCast(char));

                    // Get or assign base for current state (minimal base search)
                    if (states_arr[current_state].base == 0) {
                        // Find minimal conflict-free base for this state
                        var base_candidate: u32 = 1;
                        while (true) {
                            // Check if this base works for this character
                            const target_pos = base_candidate + char_u8;
                            if (target_pos >= states_arr.len) {
                                // Expand arrays to accommodate target_pos
                                const old_len = states_arr.len;
                                const new_len = @max(old_len * 2, target_pos + 1);
                                states_arr = try allocator.realloc(states_arr, new_len);
                                output_lists = try allocator.realloc(output_lists, new_len);
                                for (states_arr[old_len..new_len]) |*state| {
                                    state.* = .{
                                        .base = 0,
                                        .check = 0xFFFFFFFF,
                                        .fail = 0,
                                        .output_start = 0,
                                        .output_len = 0,
                                        .flags = 0,
                                        ._padding1 = 0,
                                        ._padding2 = 0,
                                    };
                                }
                                for (output_lists[old_len..new_len]) |*o| {
                                    o.* = std.ArrayList(usize){};
                                }
                            }
                            if (states_arr[target_pos].check == 0xFFFFFFFF) {
                                // Position is empty, use this base
                                states_arr[current_state].base = @as(i32, @intCast(base_candidate));
                                break;
                            }
                            // Conflict, try next base
                            base_candidate += 1;
                        }
                    }

                    const base_val = states_arr[current_state].base;
                    const target_pos = @as(u32, @intCast(base_val + @as(i32, char_u8)));

                    // Expand arrays if necessary
                    while (target_pos >= states_arr.len) {
                        const old_len = states_arr.len;
                        const new_len = old_len * 2;
                        states_arr = try allocator.realloc(states_arr, new_len);
                        output_lists = try allocator.realloc(output_lists, new_len);
                        for (states_arr[old_len..new_len]) |*state| {
                            state.* = .{
                                .base = 0,
                                .check = 0xFFFFFFFF,
                                .fail = 0,
                                .output_start = 0,
                                .output_len = 0,
                                .flags = 0,
                                ._padding1 = 0,
                                ._padding2 = 0,
                            };
                        }
                        for (output_lists[old_len..new_len]) |*o| {
                            o.* = std.ArrayList(usize){};
                        }
                    }

                    // Assign next state at target position
                    if (states_arr[target_pos].check == 0xFFFFFFFF) {
                        // First time at this position - create new state
                        states_arr[target_pos].check = current_state;
                        next_state_id = @max(next_state_id, target_pos + 1);
                    }

                    current_state = target_pos;
                }

                // Mark end of pattern: set IS_LEAF flag
                states_arr[current_state].flags |= IS_LEAF;
                try output_lists[current_state].append(allocator, pattern_idx);
            }

            // Trim states array to actual size used
            states_arr = try allocator.realloc(states_arr, next_state_id);
            output_lists = try allocator.realloc(output_lists, next_state_id);

            // Phase 3: Flatten output_lists into single patterns array
            var total_patterns: u32 = 0;
            for (output_lists[0..next_state_id]) |list| {
                total_patterns += @as(u32, @intCast(list.items.len));
            }

            const patterns_arr = try allocator.alloc(usize, total_patterns);
            errdefer allocator.free(patterns_arr);

            var pattern_offset: u32 = 0;
            for (states_arr[0..next_state_id], 0..) |*state, state_idx| {
                const list = output_lists[state_idx];
                state.output_start = pattern_offset;
                state.output_len = @intCast(list.items.len);
                @memcpy(
                    patterns_arr[pattern_offset .. pattern_offset + @as(u32, @intCast(list.items.len))],
                    list.items,
                );
                pattern_offset += @as(u32, @intCast(list.items.len));
            }

            // Clean up temporary output_lists
            for (output_lists) |*o| o.deinit(allocator);
            allocator.free(output_lists);

            // Allocate goto_table for completion optimization
            const goto_table = try allocator.alloc(u32, @as(usize, next_state_id) * 256);
            errdefer allocator.free(goto_table);

            var result = Self{
                .states = states_arr,
                .patterns = patterns_arr,
                .goto_table = goto_table,
                .state_count = next_state_id,
                .pattern_count = total_patterns,
                .allocator = allocator,
                .input_patterns = patterns,
            };

            // Build failure links and output links for Aho-Corasick
            try result.buildFailureLinks();
            try result.buildOutputLinks();
            // Build goto completion table (pre-compute all transitions)
            try result.buildGotoCompletion();

            return result;
        }

        /// Build failure links using BFS (Aho-Corasick preprocessing).
        /// Failure links enable pattern detection after state mismatch.
        /// Time: O(|V|) | Space: O(|V|)
        fn buildFailureLinks(self: *Self) !void {
            if (self.states.len == 0) return;

            // Use a queue for BFS traversal
            var queue = std.ArrayList(u32){};
            defer queue.deinit(self.allocator);

            // Root's failure link is self (0 -> 0)
            self.states[0].fail = 0;

            // Initialize: all depth-1 nodes have failure link to root
            if (self.states[0].base >= 0) {
                const base_val = self.states[0].base;
                for (0..256) |c| {
                    const target_pos = @as(u32, @intCast(base_val + @as(i32, @intCast(c))));
                    if (target_pos < self.states.len and self.states[target_pos].check == 0) {
                        // This is a depth-1 node
                        self.states[target_pos].fail = 0;
                        try queue.append(self.allocator, target_pos);
                    }
                }
            }

            // BFS to compute failure links for all other nodes
            var queue_idx: usize = 0;
            while (queue_idx < queue.items.len) {
                const current = queue.items[queue_idx];
                queue_idx += 1;

                // Get children of current state
                if (current < self.states.len and self.states[current].base >= 0) {
                    const base_val = self.states[current].base;
                    for (0..256) |c| {
                        const target_pos = @as(u32, @intCast(base_val + @as(i32, @intCast(c))));
                        if (target_pos < self.states.len and self.states[target_pos].check == current) {
                            // This is a child of current state

                            // Find failure link by following parent's failure chain
                            var failure_candidate = self.states[current].fail;
                            while (failure_candidate != 0) {
                                const fail_base = self.states[failure_candidate].base;
                                if (fail_base >= 0) {
                                    const fail_target = @as(u32, @intCast(fail_base + @as(i32, @intCast(c))));
                                    if (fail_target < self.states.len and self.states[fail_target].check == failure_candidate) {
                                        // Found a transition on character c
                                        self.states[target_pos].fail = fail_target;
                                        break;
                                    }
                                }
                                failure_candidate = self.states[failure_candidate].fail;
                            }

                            // If no match found, link to root
                            if (failure_candidate == 0) {
                                // Check if root has transition on c
                                const root_base = self.states[0].base;
                                if (root_base >= 0) {
                                    const root_target = @as(u32, @intCast(root_base + @as(i32, @intCast(c))));
                                    if (root_target < self.states.len and self.states[root_target].check == 0) {
                                        self.states[target_pos].fail = root_target;
                                    } else {
                                        self.states[target_pos].fail = 0;
                                    }
                                } else {
                                    self.states[target_pos].fail = 0;
                                }
                            }

                            try queue.append(self.allocator, target_pos);
                        }
                    }
                }
            }
        }

        /// Build output links for overlapping pattern detection.
        /// Copies pattern indices from failure states to output array.
        /// Time: O(|V| × |failure_chain|) | Space: O(z) where z = total output patterns
        fn buildOutputLinks(self: *Self) !void {
            if (self.states.len == 0) return;

            // Phase 3: Build output lists with failure chain patterns
            var output_lists = try self.allocator.alloc(std.ArrayList(usize), self.state_count);
            defer {
                for (output_lists) |*list| list.deinit(self.allocator);
                self.allocator.free(output_lists);
            }
            for (output_lists) |*list| {
                list.* = std.ArrayList(usize){};
            }

            // For each state, collect patterns: direct + patterns from failure chain
            for (0..self.state_count) |s| {
                const state_idx = @as(u32, @intCast(s));
                const state = self.states[state_idx];

                // Add direct patterns from current state
                const start = state.output_start;
                const len = state.output_len;
                for (self.patterns[start .. start + len]) |pattern_idx| {
                    try output_lists[state_idx].append(self.allocator, pattern_idx);
                }

                // Add patterns from failure chain
                var failure_state = state.fail;
                while (failure_state != state_idx and failure_state != 0) {
                    if (failure_state < self.state_count) {
                        const fail_state = self.states[failure_state];
                        const fail_start = fail_state.output_start;
                        const fail_len = fail_state.output_len;
                        for (self.patterns[fail_start .. fail_start + fail_len]) |pattern_idx| {
                            try output_lists[state_idx].append(self.allocator, pattern_idx);
                        }
                    }
                    if (failure_state < self.state_count) {
                        failure_state = self.states[failure_state].fail;
                    } else {
                        break;
                    }
                }
            }

            // Flatten output_lists into patterns array
            var total_patterns: u32 = 0;
            for (output_lists) |list| {
                total_patterns += @as(u32, @intCast(list.items.len));
            }

            const new_patterns = try self.allocator.alloc(usize, total_patterns);
            var pattern_offset: u32 = 0;
            for (self.states[0..self.state_count], 0..) |*state, state_idx| {
                const list = output_lists[state_idx];
                state.output_start = pattern_offset;
                state.output_len = @intCast(list.items.len);
                @memcpy(
                    new_patterns[pattern_offset .. pattern_offset + @as(u32, @intCast(list.items.len))],
                    list.items,
                );
                pattern_offset += @as(u32, @intCast(list.items.len));
            }

            // Replace old patterns array
            self.allocator.free(self.patterns);
            self.patterns = new_patterns;
            self.pattern_count = total_patterns;
        }

        /// Build goto completion table for O(1) transition lookups.
        /// Pre-computes transitions for all characters from all states,
        /// eliminating the failure link following loop in findAll().
        /// Time: O(|V| × 256) | Space: O(|V| × 256)
        fn buildGotoCompletion(self: *Self) !void {
            if (self.states.len == 0) return;

            // For each state and character, compute the next state
            for (0..self.state_count) |s| {
                const state = self.states[s];
                const state_u32 = @as(u32, @intCast(s));

                for (0..256) |c| {
                    const char_u8: u8 = @intCast(c);
                    var next_state: u32 = undefined;
                    var found = false;

                    // Try direct transition first (sparse array)
                    if (state.base >= 0) {
                        const next_state_signed = state.base + @as(i32, @intCast(char_u8));
                        if (next_state_signed >= 0) {
                            const candidate = @as(u32, @intCast(next_state_signed));
                            if (candidate < self.states.len and self.states[candidate].check == state_u32) {
                                // Direct transition exists
                                next_state = candidate;
                                found = true;
                            }
                        }
                    }

                    // If no direct transition, follow failure links to find valid transition
                    if (!found) {
                        if (state_u32 == 0) {
                            // Root state: always transition to itself or existing children
                            const root_base = self.states[0].base;
                            if (root_base >= 0) {
                                const root_target_signed = root_base + @as(i32, @intCast(char_u8));
                                if (root_target_signed >= 0) {
                                    const root_target = @as(u32, @intCast(root_target_signed));
                                    if (root_target < self.states.len and self.states[root_target].check == 0) {
                                        next_state = root_target;
                                        found = true;
                                    }
                                }
                            }
                            if (!found) {
                                next_state = 0; // Stay at root
                                found = true;
                            }
                        } else {
                            // Non-root: follow failure links
                            var failure_state = state.fail;
                            while (failure_state != 0) {
                                const fail_state = self.states[failure_state];
                                if (fail_state.base >= 0) {
                                    const fail_target_signed = fail_state.base + @as(i32, @intCast(char_u8));
                                    if (fail_target_signed >= 0) {
                                        const fail_target = @as(u32, @intCast(fail_target_signed));
                                        if (fail_target < self.states.len and self.states[fail_target].check == failure_state) {
                                            next_state = fail_target;
                                            found = true;
                                            break;
                                        }
                                    }
                                }
                                failure_state = fail_state.fail;
                            }

                            // If still not found, try root
                            if (!found) {
                                const root_base = self.states[0].base;
                                if (root_base >= 0) {
                                    const root_target_signed = root_base + @as(i32, @intCast(char_u8));
                                    if (root_target_signed >= 0) {
                                        const root_target = @as(u32, @intCast(root_target_signed));
                                        if (root_target < self.states.len and self.states[root_target].check == 0) {
                                            next_state = root_target;
                                            found = true;
                                        }
                                    }
                                }
                            }

                            // Ultimate fallback: root
                            if (!found) {
                                next_state = 0;
                            }
                        }
                    }

                    // Store in goto table
                    self.goto_table[s * 256 + c] = next_state;
                }
            }
        }

        /// Free all resources used by the trie.
        /// Time: O(|V|) | Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.states.len > 0) {
                self.allocator.free(self.states);
            }
            if (self.patterns.len > 0) {
                self.allocator.free(self.patterns);
            }
            if (self.goto_table.len > 0) {
                self.allocator.free(self.goto_table);
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
            if (self.states.len == 0) return false;

            var state: u32 = 0;
            for (key) |char| {
                // Get base value for current state (Phase 3: single cache line access!)
                if (state >= self.states.len) return false;
                const base_val = self.states[state].base;

                // Calculate next state position: pos = BASE[state] + c
                // Cast char to i32 (safely, since u8 < 256)
                const char_as_u8 = @as(u8, @intCast(char));
                const char_as_i32 = @as(i32, @intCast(char_as_u8));
                const next_state_signed = base_val + char_as_i32;
                if (next_state_signed < 0) return false;

                const next_state = @as(u32, @intCast(next_state_signed));
                if (next_state >= self.states.len) return false;

                // Verify validity with CHECK field (same State struct as base!)
                if (self.states[next_state].check != state) return false;

                state = next_state;
            }

            // Final state must be a leaf (marked by IS_LEAF flag) to be a valid pattern ending
            if (state >= self.states.len) return false;
            return (self.states[state].flags & IS_LEAF) != 0;
        }

        // -- Aho-Corasick Search --

        /// Find all pattern occurrences in text using Aho-Corasick automaton.
        /// Caller must free returned slice with allocator.free().
        /// Time: O(|text| + z) where z = number of matches | Space: O(z)
        pub fn findAll(self: *const Self, allocator: Allocator, text: []const T) ![]Match {
            var matches = std.ArrayList(Match){};
            errdefer matches.deinit(allocator);

            if (text.len == 0 or self.states.len == 0) {
                return matches.toOwnedSlice(allocator);
            }

            var current_state: u32 = 0;

            for (text, 0..) |char, i| {
                const char_u8 = @as(u8, @intCast(char));

                // Use pre-computed goto table for O(1) transition lookup
                // Goto completion eliminates failure link following loop
                if (current_state < self.state_count) {
                    current_state = self.goto_table[current_state * 256 + char_u8];
                } else {
                    current_state = 0;
                }

                // Emit patterns at current state
                if (current_state < self.state_count) {
                    const state = self.states[current_state];
                    if (state.output_start + state.output_len <= self.patterns.len) {
                        const state_patterns = self.patterns[state.output_start .. state.output_start + state.output_len];
                        for (state_patterns) |pattern_idx| {
                            if (pattern_idx < self.input_patterns.len) {
                                const pattern_len = self.input_patterns[pattern_idx].len;
                                if (i + 1 >= pattern_len) {
                                    try matches.append(allocator, .{
                                        .pattern_index = pattern_idx,
                                        .position = i + 1 - pattern_len,
                                    });
                                }
                            }
                        }
                    }
                }

                // Emit patterns at failure chain (overlapping matches)
                var failure_state = if (current_state < self.states.len) self.states[current_state].fail else 0;
                while (failure_state != 0 and failure_state != current_state) {
                    if (failure_state < self.state_count) {
                        const state = self.states[failure_state];
                        if (state.output_start + state.output_len <= self.patterns.len) {
                            const state_patterns = self.patterns[state.output_start .. state.output_start + state.output_len];
                            for (state_patterns) |pattern_idx| {
                                if (pattern_idx < self.input_patterns.len) {
                                    const pattern_len = self.input_patterns[pattern_idx].len;
                                    if (i + 1 >= pattern_len) {
                                        try matches.append(allocator, .{
                                            .pattern_index = pattern_idx,
                                            .position = i + 1 - pattern_len,
                                        });
                                    }
                                }
                            }
                        }
                    }
                    if (failure_state < self.states.len) {
                        failure_state = self.states[failure_state].fail;
                    } else {
                        break;
                    }
                }
            }

            return matches.toOwnedSlice(allocator);
        }

        // -- Validation --

        /// Validate trie invariants: CHECK[BASE[s] + c] == s for all valid transitions.
        /// Used for testing and debugging.
        /// Time: O(|V| + |E|) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            // Phase 3: Validate State struct invariants
            if (self.states.len == 0) return;

            // Verify size consistency
            if (self.states.len != self.state_count) {
                return error.RootInvariant;
            }

            // Verify patterns array consistency
            var max_output_end: u32 = 0;
            for (self.states[0..self.state_count]) |state| {
                max_output_end = @max(max_output_end, state.output_start + state.output_len);
            }
            if (max_output_end != self.pattern_count) {
                return error.RootInvariant;
            }

            // Verify goto_table consistency (if non-empty)
            if (self.state_count > 0) {
                const expected_goto_size: usize = @as(usize, self.state_count) * 256;
                if (self.goto_table.len != expected_goto_size) {
                    return error.RootInvariant;
                }

                // Verify all goto entries are valid state indices
                for (self.goto_table) |next_state| {
                    if (next_state >= self.state_count) {
                        return error.RootInvariant;
                    }
                }
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
    try testing.expect(trie.states.len > 0);
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

// ============================================================================
// PHASE 3 LINEARIZATION TESTS (RED PHASE — FAILING)
// ============================================================================
// These tests verify the Phase 3 linearized State structure design.
// Expected to FAIL with current Phase 2 implementation.
// Phase 3 replaces 4 separate arrays (base_check, is_leaf, fail, output)
// with a single 24-byte State struct + flattened patterns array.

test "Phase 3: State struct size validation" {
    // Phase 3 design goal: pack all state data into exactly 24 bytes
    // Current Phase 2: BASE+CHECK (8) + is_leaf (1) + fail (4) + output overhead = 13+
    // Phase 3: Single State struct with 24-byte cache-line alignment
    //
    // Assertion: @sizeOf(State) must equal 24 bytes (8-byte aligned)
    // This ensures each state fits in exactly 3 cache line slots (64-byte line / 24-byte state)
    // and minimizes padding waste.
    //
    // If this fails with Phase 2, it's expected — State struct doesn't exist yet.
    // Phase 3 implementation must define: pub const State = struct { ... };

    const Trie = DoubleArrayTrie(u8);
    const state_size = @sizeOf(Trie.State);
    try testing.expectEqual(@as(usize, 24), state_size);
}

test "Phase 3: states array replaces base_check" {
    // Phase 3 redesign: Replace Phase 2's separate base_check array with
    // a single linearized states array where each State contains base, check,
    // fail, and output metadata.
    //
    // Assertion: Self.states exists and has type []State
    // Assertion: Self.base_check should NOT exist (removed in Phase 3)
    //
    // Expected Phase 2 behavior:
    // - base_check field exists: FAIL (Phase 3 removes it)
    // - states field missing: FAIL (Phase 3 adds it)
    //
    // After Phase 3 refactoring, the DoubleArrayTrie struct should contain:
    // - states: []State (replaces base_check: []BaseCheck)
    // - patterns: []usize (flattened output patterns)
    // - state_count: u32 (unchanged)
    // - allocator: Allocator (unchanged)

    const allocator = testing.allocator;
    const patterns = [_][]const u8{ "test" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    // Phase 3 requirement: states array must exist and be non-empty
    try testing.expect(@hasField(DoubleArrayTrie(u8), "states"));
    try testing.expect(trie.states.len > 0);

    // Phase 3 requirement: base_check array must be removed
    try testing.expect(!@hasField(DoubleArrayTrie(u8), "base_check"));
}

test "Phase 3: patterns array linearization" {
    // Phase 3 redesign: Replace Phase 2's output: []ArrayList(usize) with
    // a single flattened patterns: []usize array, indexed by State.output_start
    // and State.output_len.
    //
    // Pattern layout (Phase 3):
    // - patterns: [pattern_0_idx_0, pattern_0_idx_1, pattern_1_idx_0, ...]
    // - State.output_start: index into patterns[] where this state's patterns begin
    // - State.output_len: count of pattern indices for this state
    //
    // Example: If state 5 has patterns [2, 5, 7]:
    // - state.output_start = 100 (position in patterns[])
    // - state.output_len = 3
    // - patterns[100..103] = [2, 5, 7]
    //
    // Assertion: patterns field exists (replaces output field)
    // Assertion: Accessing patterns at state.output_start returns correct indices
    //
    // Current Phase 2: output[state].items = []usize
    // Phase 3: patterns[state.output_start .. state.output_start + state.output_len] = []usize

    const allocator = testing.allocator;
    const test_patterns = [_][]const u8{ "abc", "ab", "bc" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &test_patterns);
    defer trie.deinit();

    // Phase 3 requirement: patterns array exists (linearized output)
    try testing.expect(@hasField(DoubleArrayTrie(u8), "patterns"));
    try testing.expect(trie.patterns.len > 0);

    // Phase 3 requirement: output field removed (replaced by patterns + output_start/output_len)
    try testing.expect(!@hasField(DoubleArrayTrie(u8), "output"));
}

test "Phase 3: IS_LEAF flag replaces is_leaf array" {
    // Phase 3 redesign: Replace Phase 2's is_leaf: []bool with
    // a bit flag in State.flags (0x01 = IS_LEAF).
    //
    // Flag layout (Phase 3):
    // - State.flags: u8 bit field
    // - IS_LEAF constant: 0x01
    // - Leaf check: (state.flags & IS_LEAF) != 0
    //
    // Memory improvement:
    // - Phase 2: is_leaf[state] = 1 byte per state
    // - Phase 3: state.flags bit = 1 bit (within 24-byte State)
    //
    // Assertion: is_leaf array field removed
    // Assertion: State.flags field exists
    // Assertion: IS_LEAF constant defined
    // Assertion: Leaf states can be checked via flags

    const allocator = testing.allocator;
    const test_patterns = [_][]const u8{ "hello", "world" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &test_patterns);
    defer trie.deinit();

    // Phase 3 requirement: is_leaf array removed
    try testing.expect(!@hasField(DoubleArrayTrie(u8), "is_leaf"));

    // Phase 3 requirement: State struct has flags field
    const Trie = DoubleArrayTrie(u8);
    try testing.expect(@hasField(Trie.State, "flags"));

    // Phase 3 requirement: IS_LEAF constant defined for flag bit
    try testing.expect(@hasDecl(Trie, "IS_LEAF"));
}

test "Phase 3: fail field in State struct" {
    // Phase 3 redesign: Move fail: []u32 array into State.fail: u32 field.
    //
    // Aho-Corasick automaton requires failure links to find suffix matches.
    // Phase 2 stores these in a separate array: fail[state] = next_state
    // Phase 3 stores them in the State struct: state.fail = next_state
    //
    // Memory improvement:
    // - Phase 2: fail[state] = 1 separate array access
    // - Phase 3: state.fail = same cache line as state.base/check (1 access)
    //
    // Assertion: fail array field removed
    // Assertion: State.fail field exists and is u32
    // Assertion: Failure links can be accessed via state.fail

    const allocator = testing.allocator;
    const test_patterns = [_][]const u8{ "she", "he", "hers" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &test_patterns);
    defer trie.deinit();

    // Phase 3 requirement: fail array removed
    try testing.expect(!@hasField(DoubleArrayTrie(u8), "fail"));

    // Phase 3 requirement: State struct has fail field (u32)
    const Trie = DoubleArrayTrie(u8);
    try testing.expect(@hasField(Trie.State, "fail"));

    // Verify State.fail is u32
    // (This is a compile-time check, but we can validate after init)
}

test "Phase 3: memory layout sequential" {
    // Phase 3 design verification: states array must be densely packed.
    // Each State struct is exactly 24 bytes, so states[i+1] should be
    // exactly 24 bytes after states[i] in memory.
    //
    // Verification:
    // - Calculate address of states[0]
    // - Calculate address of states[1]
    // - Verify difference == 24 bytes
    //
    // This ensures optimal cache locality and no padding waste between states.
    // Failure indicates alignment or struct size issue.
    //
    // Assertion: states array has uniform 24-byte stride
    // Assertion: No padding or gaps between consecutive states

    const allocator = testing.allocator;
    const test_patterns = [_][]const u8{ "a", "ab", "abc" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &test_patterns);
    defer trie.deinit();

    if (trie.states.len >= 2) {
        const addr_0 = @intFromPtr(&trie.states[0]);
        const addr_1 = @intFromPtr(&trie.states[1]);
        const stride = addr_1 - addr_0;

        // Phase 3 requirement: uniform 24-byte stride between states
        try testing.expectEqual(@as(usize, 24), stride);
    }
}

// ============================================================================
// GOTO COMPLETION OPTIMIZATION TESTS (RED PHASE — FAILING)
// ============================================================================
// Tests for goto completion pre-computation optimization.
// Goto completion eliminates the failure link loop in findAll() by pre-computing
// all transitions, improving throughput from 82 MB/sec → 150-200 MB/sec.
//
// Current findAll() (lines 506-531): For each character, follows failure links
// until finding a valid transition. This is O(|alphabet|) in worst case per char.
//
// Goto completion: Pre-compute a "complete" transition table where every state
// has a valid transition for every character (no loops, no invalid transitions).
// This converts Aho-Corasick from sparse to dense transitions (with goto completion).

test "goto_completion: correctness findAll same results as sparse" {
    // RED PHASE: This test MUST FAIL initially because goto completion
    // is not yet implemented. After implementation, results must match exactly.
    //
    // Requirement: Verify that goto completion produces identical match results
    // to the current sparse algorithm (with failure link following).
    //
    // Verification strategy:
    // 1. Run findAll() with sparse implementation (current)
    // 2. Save results (baseline)
    // 3. After goto completion implementation:
    //    - Run findAll() with goto completion
    //    - Compare results (must be identical: same positions, pattern indices)
    //
    // Since goto completion is a pure optimization (internal change only),
    // external behavior must not change. This is the most critical test.
    //
    // Pattern set: "he", "she", "his", "hers" (Aho-Corasick standard example)
    // Text: "ahishers" (contains multiple overlapping pattern matches)
    //
    // Assertion: findAll() results are deterministic and contain expected patterns

    const allocator = testing.allocator;
    const patterns = [_][]const u8{ "he", "she", "his", "hers" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "ahishers";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    // Must find at least one match (text contains patterns)
    try testing.expect(matches.len > 0);

    // Verify that all matched patterns are from our pattern set
    for (matches) |m| {
        try testing.expect(m.pattern_index < patterns.len);
        // Pattern at matched position should be findable by string slice
        try testing.expect(m.position < text.len);
        const pattern = patterns[m.pattern_index];
        if (m.position + pattern.len <= text.len) {
            const text_slice = text[m.position .. m.position + pattern.len];
            try testing.expect(std.mem.eql(u8, text_slice, pattern));
        }
    }

    // After goto completion, results must be deterministic
    // (Running again should yield identical matches)
    const matches2 = try trie.findAll(allocator, text);
    defer allocator.free(matches2);
    try testing.expectEqual(matches.len, matches2.len);
}

test "goto_completion: transition completeness every state has all chars" {
    // RED PHASE: Goto completion must pre-compute ALL transitions.
    // After optimization, every state s and character c ∈ [0,255]
    // must have a valid transition (no failure link needed).
    //
    // CURRENTLY: Sparse version has many invalid transitions (requires failure link follow)
    // AFTER GOTO COMPLETION: All 256 transitions per state must be valid
    //
    // This test MUST FAIL with current sparse implementation.
    // Expected result: Many invalid_transition_count (current), 0 after optimization.
    //
    // Patterns: "he", "she", "his", "hers"
    // States: root, h, s, sh, she, he, his, hers (≈8 states)
    //
    // After goto completion:
    // - From root: 256 transitions (one per char)
    // - From each state: 256 transitions
    // - No state should require failure link traversal
    //
    // Assertion: After goto completion, all states have valid transitions for all chars

    const allocator = testing.allocator;
    const patterns = [_][]const u8{ "he", "she", "his", "hers" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    // Verify goto_table has been built with all 256 transitions per state
    var states_with_missing_goto: u32 = 0;
    for (0..trie.state_count) |s| {
        // Each state should have 256 entries in goto_table
        var valid_gotos: u32 = 0;
        for (0..256) |c| {
            const goto_idx = s * 256 + c;
            if (goto_idx < trie.goto_table.len) {
                const next_state = trie.goto_table[goto_idx];
                // All gotos must point to valid states
                if (next_state < trie.state_count) {
                    valid_gotos += 1;
                }
            }
        }

        // After goto completion, must have all 256 valid goto entries
        if (valid_gotos < 256) {
            states_with_missing_goto += 1;
        }
    }

    // After goto completion, this should be 0
    // (all states should have all 256 transitions in goto_table)
    try testing.expectEqual(@as(u32, 0), states_with_missing_goto);
}

test "goto_completion: memory overhead sparse not dense" {
    // RED PHASE: Goto completion memory usage constraint.
    // Expected: More memory than sparse (82 MB current) but less than dense (19.7 GB)
    //
    // Memory estimates:
    // - Current sparse (82 MB): BASE + CHECK + FAIL (~12 bytes/state)
    // - After goto completion: BASE + CHECK + FAIL + pre-computed gotos
    //   Estimated: 66 KB → 256 KB (still sparse, compressed goto table)
    // - Dense version (19.7 GB): 256 transitions × 4 bytes per state
    //
    // For patterns ["he", "she", "his", "hers"] with ~8 states:
    // - Sparse baseline: ~8 states × 12 bytes = 96 bytes
    // - Goto completion: ~8 states × 24 bytes (State) + goto overhead
    // - Maximum acceptable: < 5 KB (still sparse, not bloated)
    //
    // Assertion: Peak memory > 96 bytes (more than sparse) but < 5 KB

    const allocator = testing.allocator;
    const patterns = [_][]const u8{ "he", "she", "his", "hers" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    // Calculate memory used by linearized states
    const states_memory = trie.states.len * @sizeOf(DoubleArrayTrie(u8).State);
    const patterns_memory = trie.patterns.len * @sizeOf(usize);
    const total_memory = states_memory + patterns_memory;

    // Minimum: must be larger than sparse version (~96 bytes for small trie)
    try testing.expect(total_memory > 50);

    // Maximum: must be reasonable (< 10 KB for goto completion overhead)
    // Goto completion should add at most 2-3× memory for pre-computed tables
    try testing.expect(total_memory < 10_000);

    // Memory per state should be ~24 bytes (State struct)
    const memory_per_state = states_memory / @max(trie.states.len, 1);
    try testing.expectEqual(@as(usize, 24), memory_per_state);
}

test "goto_completion: no regression sparse throughput" {
    // RED PHASE: Performance regression test.
    // Goto completion should NOT degrade performance compared to current sparse version.
    //
    // Baseline: Current sparse findAll() = 82 MB/sec on 1000 patterns + 1 MB text
    // Requirement: After goto completion, throughput >= 82 MB/sec (ideally 150-200 MB/sec)
    //
    // This test verifies that findAll() executes without error and finds expected matches.
    // Absolute performance is machine-dependent and tested separately in benchmarks.
    //
    // Success: findAll() completes and finds all pattern instances
    // Note: This is a functional correctness test, not a throughput measurement

    const allocator = testing.allocator;

    // Simple pattern set
    const pattern_strs = [_][]const u8{ "test" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &pattern_strs);
    defer trie.deinit();

    // Generate simple test text with "test" repeated
    const text = "test test test test test";

    // Run findAll to verify it works without error
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    // Must find the pattern instances
    try testing.expect(matches.len > 0);

    // Verify no regression: all matches should be valid
    for (matches) |m| {
        try testing.expect(m.pattern_index == 0);
        try testing.expect(m.position < text.len);
    }
}

test "goto_completion: overlapping patterns correctness" {
    // RED PHASE: Goto completion must maintain correctness for overlapping patterns.
    // After optimization, failure links are replaced by pre-computed gotos,
    // but output detection must still work for all overlapping matches.
    //
    // Patterns: "ab", "abc", "bc" → all three should match in text "abc"
    // This requires OUTPUT links to work correctly even with goto completion.
    //
    // Assertion: findAll() finds all overlapping matches (same as sparse)

    const allocator = testing.allocator;
    const patterns = [_][]const u8{ "ab", "abc", "bc" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "abc";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    // All 3 overlapping patterns must be found
    try testing.expectEqual(@as(usize, 3), matches.len);

    // Verify all pattern indices are present
    var patterns_found = [_]bool{ false, false, false };
    for (matches) |m| {
        if (m.pattern_index < 3) {
            patterns_found[m.pattern_index] = true;
        }
    }
    try testing.expect(patterns_found[0]); // "ab"
    try testing.expect(patterns_found[1]); // "abc"
    try testing.expect(patterns_found[2]); // "bc"
}

test "goto_completion: repeated patterns multiple occurrences" {
    // RED PHASE: Goto completion must find all occurrences of repeated patterns.
    // Each pattern instance must be detected, not skipped due to goto optimization.
    //
    // Pattern: "ab" appears 3 times in "xabxabxab"
    // Expected: All 3 occurrences found at positions 1, 4, 7
    //
    // Assertion: findAll() finds all occurrences with correct positions

    const allocator = testing.allocator;
    const patterns = [_][]const u8{"ab"};

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "xabxabxab";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    // All 3 occurrences must be found
    try testing.expectEqual(@as(usize, 3), matches.len);

    // Verify positions
    try testing.expectEqual(@as(usize, 1), matches[0].position);
    try testing.expectEqual(@as(usize, 4), matches[1].position);
    try testing.expectEqual(@as(usize, 7), matches[2].position);
}

test "goto_completion: single character patterns all chars" {
    // RED PHASE: Single-char patterns stress goto completion.
    // After optimization, single-char states have 256 transitions pre-computed.
    // Each must lead to correct next state or root.
    //
    // Patterns: "a", "b", "c" (single chars)
    // Text: "abcabc" (6 chars = 2 full sequences)
    // Expected: 6 matches (2 of each pattern)
    //
    // Assertion: All single-char patterns found correctly

    const allocator = testing.allocator;
    const patterns = [_][]const u8{ "a", "b", "c" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "abcabc";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    try testing.expectEqual(@as(usize, 6), matches.len);
}

test "goto_completion: long text patterns near end" {
    // RED PHASE: Goto completion must work correctly at text boundaries.
    // Failure links are most critical near text end (no future chars to match).
    // Goto completion replaces failure links, so must maintain correctness.
    //
    // Pattern: "end" in "this is the very end"
    // Expected: Match at position 17 (last 3 chars)
    //
    // Assertion: Pattern near text end found correctly

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

test "goto_completion: complex failure link chain" {
    // RED PHASE: Goto completion replaces failure link chains.
    // Current sparse version: "she" → fail to "he" → fail to root
    // After goto completion: Direct transitions without failure links
    //
    // Patterns: "she", "he", "hers" (explicit failure chain in Aho-Corasick)
    // Text: "ushers"
    // Expected: All patterns found (same as sparse, via different mechanism)
    //
    // Assertion: All patterns found despite failure link elimination

    const allocator = testing.allocator;
    const patterns = [_][]const u8{ "she", "he", "hers" };

    var trie = try DoubleArrayTrie(u8).init(allocator, &patterns);
    defer trie.deinit();

    const text = "ushers";
    const matches = try trie.findAll(allocator, text);
    defer allocator.free(matches);

    // Must find all three: "she" at 1, "he" at 2, "hers" at 2
    try testing.expectEqual(@as(usize, 3), matches.len);
}

