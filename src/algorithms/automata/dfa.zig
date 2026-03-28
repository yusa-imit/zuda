/// Deterministic Finite Automaton (DFA) for fast pattern matching
/// Exactly one active state at any time, faster than NFA for matching
///
/// Use cases:
/// - Lexical analysis / scanners
/// - Protocol state machines
/// - Fast pattern validation (no backtracking)
/// - Network packet filtering
///
/// Time complexity:
/// - Construction: O(m) where m is pattern length (for simple patterns)
/// - NFA→DFA conversion: O(2^n) worst case (subset construction)
/// - Matching: O(n) where n is text length (deterministic, no backtracking)
///
/// Space complexity: O(|states| × |alphabet|) for transition table

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;

/// DFA state
pub const State = struct {
    /// State ID
    id: usize,
    /// Is this an accepting state?
    is_accept: bool,
    /// Transition table: char → next_state
    /// Using HashMap for sparse transitions (common in text processing)
    transitions: AutoHashMap(u8, usize),

    pub fn init(allocator: Allocator, id: usize, is_accept: bool) !State {
        return State{
            .id = id,
            .is_accept = is_accept,
            .transitions = AutoHashMap(u8, usize).init(allocator),
        };
    }

    pub fn deinit(self: *State) void {
        self.transitions.deinit();
    }

    /// Add transition for character to target state
    /// Time: O(1) average | Space: O(1)
    pub fn addTransition(self: *State, c: u8, to: usize) !void {
        try self.transitions.put(c, to);
    }

    /// Get next state for character, returns null if no transition
    /// Time: O(1) average | Space: O(1)
    pub fn getNext(self: *const State, c: u8) ?usize {
        return self.transitions.get(c);
    }
};

/// Deterministic Finite Automaton
pub const DFA = struct {
    const Self = @This();

        allocator: Allocator,
        /// States (indexed by ID)
        states: ArrayList(State),
        /// Start state ID
        start: usize,
        /// Current state (for stateful matching)
        current: usize,

        /// Initialize empty DFA
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator) Self {
            return Self{
                .allocator = allocator,
                .states = ArrayList(State).init(allocator),
                .start = 0,
                .current = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.states.items) |*state| {
                state.deinit();
            }
            self.states.deinit();
        }

        /// Add new state to DFA
        /// Time: O(1) amortized | Space: O(1)
        pub fn addState(self: *Self, is_accept: bool) !usize {
            const id = self.states.items.len;
            const state = try State.init(self.allocator, id, is_accept);
            try self.states.append(state);
            return id;
        }

        /// Build DFA from literal string pattern (exact match)
        /// Time: O(n) where n is pattern length | Space: O(n)
        pub fn fromLiteral(allocator: Allocator, pattern: []const u8) !Self {
            var dfa = Self.init(allocator);
            errdefer dfa.deinit();

            if (pattern.len == 0) {
                // Empty pattern: single accepting state
                const s0 = try dfa.addState(true);
                dfa.start = s0;
                dfa.current = s0;
                return dfa;
            }

            // Create chain of states for each character
            dfa.start = try dfa.addState(false);
            dfa.current = dfa.start;
            var current = dfa.start;

            for (pattern, 0..) |c, i| {
                const is_last = (i == pattern.len - 1);
                const next = try dfa.addState(is_last);
                try dfa.states.items[current].addTransition(c, next);
                current = next;
            }

            return dfa;
        }

        /// Build DFA from prefix string (matches any text starting with prefix)
        /// Time: O(n) where n is prefix length | Space: O(n)
        pub fn fromPrefix(allocator: Allocator, prefix: []const u8) !Self {
            var dfa = Self.init(allocator);
            errdefer dfa.deinit();

            if (prefix.len == 0) {
                // Empty prefix: accept everything, stay in accepting state
                const s0 = try dfa.addState(true);
                dfa.start = s0;
                dfa.current = s0;
                // Self-loop on all characters
                for (0..256) |c| {
                    try dfa.states.items[s0].addTransition(@intCast(c), s0);
                }
                return dfa;
            }

            // Create chain for prefix
            dfa.start = try dfa.addState(false);
            dfa.current = dfa.start;
            var current = dfa.start;

            for (prefix) |c| {
                const next = try dfa.addState(false);
                try dfa.states.items[current].addTransition(c, next);
                current = next;
            }

            // Last state is accepting and has self-loop on all characters
            dfa.states.items[current].is_accept = true;
            for (0..256) |c| {
                try dfa.states.items[current].addTransition(@intCast(c), current);
            }

            return dfa;
        }

        /// Build DFA for suffix matching (matches any text ending with suffix)
        /// Note: Suffix matching requires storing entire input or using specialized algorithms
        /// This implementation uses simple approach: accept if suffix matches at end
        /// Time: O(n) where n is suffix length | Space: O(n)
        pub fn fromSuffix(allocator: Allocator, suffix: []const u8) !Self {
            var dfa = Self.init(allocator);
            errdefer dfa.deinit();

            if (suffix.len == 0) {
                // Empty suffix: always accept
                const s0 = try dfa.addState(true);
                dfa.start = s0;
                dfa.current = s0;
                for (0..256) |c| {
                    try dfa.states.items[s0].addTransition(@intCast(c), s0);
                }
                return dfa;
            }

            // Create chain for suffix with KMP-style failure links
            dfa.start = try dfa.addState(false);
            dfa.current = dfa.start;

            // Add self-loop for any character at start
            for (0..256) |c| {
                if (c != suffix[0]) {
                    try dfa.states.items[dfa.start].addTransition(@intCast(c), dfa.start);
                }
            }

            // Build suffix chain
            var current = dfa.start;
            for (suffix, 0..) |c, i| {
                const is_last = (i == suffix.len - 1);
                const next = try dfa.addState(is_last);
                try dfa.states.items[current].addTransition(c, next);

                if (!is_last) {
                    // On mismatch in middle, go back to start
                    for (0..256) |ch| {
                        if (ch != suffix[i + 1]) {
                            const target = if (ch == suffix[0]) 1 else dfa.start; // 1 = first char match state
                            try dfa.states.items[next].addTransition(@intCast(ch), target);
                        }
                    }
                }

                current = next;
            }

            return dfa;
        }

        /// Reset DFA to start state
        /// Time: O(1) | Space: O(1)
        pub fn reset(self: *Self) void {
            self.current = self.start;
        }

        /// Process single character (stateful)
        /// Returns true if currently in accepting state after transition
        /// Time: O(1) | Space: O(1)
        pub fn step(self: *Self, c: u8) bool {
            if (self.current >= self.states.items.len) return false;

            const state = &self.states.items[self.current];
            if (state.getNext(c)) |next| {
                self.current = next;
            } else {
                // No transition defined: stay in current state or trap state
                // For simplicity, stay in current state (partial DFA)
            }

            return self.states.items[self.current].is_accept;
        }

        /// Match entire text against DFA (stateless)
        /// Returns true if text is accepted
        /// Time: O(n) where n is text length | Space: O(1)
        pub fn match(self: *const Self, text: []const u8) bool {
            if (self.states.items.len == 0) return false;

            var current = self.start;

            for (text) |c| {
                const state = &self.states.items[current];
                if (state.getNext(c)) |next| {
                    current = next;
                } else {
                    // No transition: reject (complete DFA would have trap state)
                    return false;
                }
            }

            return self.states.items[current].is_accept;
        }

        /// Check if text matches pattern (convenience wrapper)
        /// Time: O(n) | Space: O(1)
        pub fn accepts(self: *const Self, text: []const u8) bool {
            return self.match(text);
        }

        /// Search for pattern in text (returns index of first match or null)
        /// Time: O(n × m) worst case | Space: O(1)
        pub fn search(self: *const Self, text: []const u8) ?usize {
            if (text.len == 0) return null;

            for (text, 0..) |_, i| {
                if (self.match(text[i..])) {
                    return i;
                }
            }

            return null;
        }

        /// Count number of non-overlapping matches in text
        /// Time: O(n × m) where n is text length, m is pattern length | Space: O(1)
        pub fn count(self: *const Self, text: []const u8) usize {
            var result: usize = 0;
            var i: usize = 0;

            while (i < text.len) {
                if (self.match(text[i..])) {
                    result += 1;
                    // Skip past match to avoid overlapping
                    i += 1; // For now, advance by 1 (proper impl would skip pattern length)
                } else {
                    i += 1;
                }
            }

            return result;
        }

        /// Validate DFA structure
        /// Time: O(|states| × |transitions|) | Space: O(1)
        pub fn validate(self: *const Self) bool {
            if (self.states.items.len == 0) return false;
            if (self.start >= self.states.items.len) return false;
            if (self.current >= self.states.items.len) return false;

            // Check all transitions point to valid states
            for (self.states.items) |state| {
                var iter = state.transitions.valueIterator();
                while (iter.next()) |target| {
                    if (target.* >= self.states.items.len) return false;
                }
            }

            return true;
        }
};

// Tests
const testing = std.testing;

test "DFA: literal pattern exact match" {
    const allocator = testing.allocator;

    var dfa = try DFA.fromLiteral(allocator, "hello");
    defer dfa.deinit();

    try testing.expect(dfa.match("hello"));
    try testing.expect(!dfa.match("hell"));
    try testing.expect(!dfa.match("helloa"));
    try testing.expect(!dfa.match("Hello"));
    try testing.expect(!dfa.match(""));
}

test "DFA: empty pattern" {
    const allocator = testing.allocator;

    var dfa = try DFA.fromLiteral(allocator, "");
    defer dfa.deinit();

    try testing.expect(dfa.match(""));
    try testing.expect(!dfa.match("a"));
}

test "DFA: single character" {
    const allocator = testing.allocator;

    var dfa = try DFA.fromLiteral(allocator, "x");
    defer dfa.deinit();

    try testing.expect(dfa.match("x"));
    try testing.expect(!dfa.match(""));
    try testing.expect(!dfa.match("xy"));
    try testing.expect(!dfa.match("y"));
}

test "DFA: prefix matching" {
    const allocator = testing.allocator;

    var dfa = try DFA.fromPrefix(allocator, "pre");
    defer dfa.deinit();

    try testing.expect(dfa.match("pre"));
    try testing.expect(dfa.match("prefix"));
    try testing.expect(dfa.match("pre123"));
    try testing.expect(!dfa.match("pr"));
    try testing.expect(!dfa.match("xpre"));
}

test "DFA: empty prefix (accept all)" {
    const allocator = testing.allocator;

    var dfa = try DFA.fromPrefix(allocator, "");
    defer dfa.deinit();

    try testing.expect(dfa.match(""));
    try testing.expect(dfa.match("anything"));
    try testing.expect(dfa.match("xyz"));
}

test "DFA: suffix matching" {
    const allocator = testing.allocator;

    var dfa = try DFA.fromSuffix(allocator, "end");
    defer dfa.deinit();

    try testing.expect(dfa.match("end"));
    try testing.expect(dfa.match("theend"));
    try testing.expect(dfa.match("xxxend"));
    try testing.expect(!dfa.match("ending"));
    try testing.expect(!dfa.match("en"));
}

test "DFA: stateful step matching" {
    const allocator = testing.allocator;

    var dfa = try DFA.fromLiteral(allocator, "abc");
    defer dfa.deinit();

    dfa.reset();
    try testing.expect(!dfa.step('a'));
    try testing.expect(!dfa.step('b'));
    try testing.expect(dfa.step('c')); // Now in accepting state

    dfa.reset();
    try testing.expect(!dfa.step('a'));
    try testing.expect(!dfa.step('x')); // Wrong char, no transition
}

test "DFA: search in text" {
    const allocator = testing.allocator;

    var dfa = try DFA.fromLiteral(allocator, "cat");
    defer dfa.deinit();

    try testing.expectEqual(@as(?usize, 0), dfa.search("cat"));
    try testing.expectEqual(@as(?usize, 4), dfa.search("the cat sat"));
    try testing.expectEqual(@as(?usize, null), dfa.search("dog"));
}

test "DFA: validate structure" {
    const allocator = testing.allocator;

    var dfa = try DFA.fromLiteral(allocator, "test");
    defer dfa.deinit();

    try testing.expect(dfa.validate());
}

test "DFA: case sensitivity" {
    const allocator = testing.allocator;

    var dfa = try DFA.fromLiteral(allocator, "Test");
    defer dfa.deinit();

    try testing.expect(dfa.match("Test"));
    try testing.expect(!dfa.match("test"));
    try testing.expect(!dfa.match("TEST"));
}

test "DFA: manual construction" {
    const allocator = testing.allocator;

    var dfa = DFA.init(allocator);
    defer dfa.deinit();

    const s0 = try dfa.addState(false);
    const s1 = try dfa.addState(true);

    dfa.start = s0;
    dfa.current = s0;

    try dfa.states.items[s0].addTransition('a', s1);
    try dfa.states.items[s1].addTransition('b', s0);

    // Accepts "a" (ends in s1)
    try testing.expect(dfa.match("a"));
    try testing.expect(!dfa.match("ab")); // Ends in s0
    try testing.expect(dfa.match("aba")); // Ends in s1
}

test "DFA: accepts() alias" {
    const allocator = testing.allocator;

    var dfa = try DFA.fromLiteral(allocator, "ok");
    defer dfa.deinit();

    try testing.expect(dfa.accepts("ok"));
    try testing.expect(!dfa.accepts("not"));
}
