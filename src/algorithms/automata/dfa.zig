//! Deterministic Finite Automaton (DFA) for fast pattern matching.
//!
//! Exactly one active state at any time, faster than an NFA for matching.
//!
//! Use cases: lexical analysis, protocol state machines, pattern validation without
//! backtracking, network packet filtering.
//!
//! Time: construction O(m) for a pattern of length m (suffix patterns O(256 m)); NFA to DFA
//! conversion O(2^n) worst case (subset construction); matching O(n) for text of length n.
//! Space: O(|states| x |alphabet|) for the sparse transition tables.
//!
//! Ownership: a `DFA` keeps the allocator it was initialised with because `addState` grows
//! it; the owner releases everything with `deinit`. Constructors free partial work on error.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;
const assert = std.debug.assert;

/// DFA state
pub const State = struct {
    /// State ID
    id: usize,
    /// Is this an accepting state?
    is_accept: bool,
    /// Transition table: char → next_state
    /// Using HashMap for sparse transitions (common in text processing)
    transitions: AutoHashMap(u8, usize),

    pub fn init(gpa: Allocator, id: usize, is_accept: bool) !State {
        return State{
            .id = id,
            .is_accept = is_accept,
            .transitions = AutoHashMap(u8, usize).init(gpa),
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
    allocator: Allocator,
    /// States (indexed by ID)
    states: ArrayList(State),
    /// Start state ID
    start: usize,
    /// Current state (for stateful matching)
    current: usize,

    /// Initialize empty DFA
    /// Time: O(1) | Space: O(1)
    pub fn init(gpa: Allocator) DFA {
        return DFA{
            .allocator = gpa,
            .states = .empty,
            .start = 0,
            .current = 0,
        };
    }

    pub fn deinit(self: *DFA) void {
        for (self.states.items) |*state| {
            state.deinit();
        }
        self.states.deinit(self.allocator);
    }

    /// Add new state to DFA
    /// Time: O(1) amortized | Space: O(1)
    pub fn addState(self: *DFA, is_accept: bool) !usize {
        const id = self.states.items.len;
        const state = try State.init(self.allocator, id, is_accept);
        try self.states.append(self.allocator, state);
        return id;
    }

    /// Build DFA from literal string pattern (exact match)
    /// Time: O(n) where n is pattern length | Space: O(n)
    pub fn fromLiteral(gpa: Allocator, pattern: []const u8) !DFA {
        var dfa = DFA.init(gpa);
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
    pub fn fromPrefix(gpa: Allocator, prefix: []const u8) !DFA {
        var dfa = DFA.init(gpa);
        errdefer dfa.deinit();

        if (prefix.len == 0) {
            // Empty prefix: accept everything, stay in accepting state
            const s0 = try dfa.addState(true);
            dfa.start = s0;
            dfa.current = s0;
            // DFA-loop on all characters
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

    /// Build DFA for suffix matching (accepts any text ending with `suffix`)
    ///
    /// This is the KMP automaton: state j means "the longest suffix of the text read so far
    /// that is a prefix of `suffix` has length j", and state `suffix.len` accepts. Every
    /// state has a transition for all 256 bytes, so overlapping occurrences such as
    /// "endend" or "aaab" for "aab" are recognised.
    /// Time: O(256 n) where n is suffix length | Space: O(256 n)
    pub fn fromSuffix(gpa: Allocator, suffix: []const u8) !DFA {
        var dfa = DFA.init(gpa);
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

        // State 0: only the first suffix byte makes progress.
        dfa.start = try dfa.addState(false);
        dfa.current = dfa.start;
        for (0..256) |c| {
            const to: usize = if (c == suffix[0]) 1 else 0;
            try dfa.states.items[dfa.start].addTransition(@intCast(c), to);
        }

        // `restart` is the state reached by reading suffix[1..j]; a state on a mismatch
        // behaves exactly like it, and then gains its own forward edge.
        var restart: usize = 0;
        for (1..suffix.len + 1) |j| {
            const state_j = try dfa.addState(j == suffix.len);
            assert(state_j == j);
            assert(restart < j);

            var iter = dfa.states.items[restart].transitions.iterator();
            for (0..256) |_| {
                const entry = iter.next() orelse break;
                try dfa.states.items[j].addTransition(entry.key_ptr.*, entry.value_ptr.*);
            }
            if (j == suffix.len) break;

            try dfa.states.items[j].addTransition(suffix[j], j + 1);
            restart = dfa.states.items[restart].getNext(suffix[j]).?;
        }

        return dfa;
    }

    /// Reset DFA to start state
    /// Time: O(1) | Space: O(1)
    pub fn reset(self: *DFA) void {
        self.current = self.start;
    }

    /// Process single character (stateful)
    /// Returns true if currently in accepting state after transition
    /// Time: O(1) | Space: O(1)
    pub fn step(self: *DFA, c: u8) bool {
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
    pub fn match(self: *const DFA, text: []const u8) bool {
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
    pub fn accepts(self: *const DFA, text: []const u8) bool {
        return self.match(text);
    }

    /// Length of the shortest prefix of `text` that the DFA accepts, or null if none.
    /// A start state that accepts yields 0 (the empty match).
    /// Time: O(n) where n is text length | Space: O(1)
    pub fn matchLen(self: *const DFA, text: []const u8) ?usize {
        if (self.states.items.len == 0) return null;
        assert(self.start < self.states.items.len);

        var current = self.start;
        if (self.states.items[current].is_accept) return 0;

        for (text, 1..) |c, len| {
            // No transition rejects (a complete DFA would have a trap state).
            current = self.states.items[current].getNext(c) orelse return null;
            if (self.states.items[current].is_accept) return len;
        }

        return null;
    }

    /// Search for pattern in text (returns index of first match or null)
    ///
    /// A match starts at the leftmost index from which some prefix of the rest of the text
    /// is accepted.
    /// Time: O(n × m) worst case | Space: O(1)
    pub fn search(self: *const DFA, text: []const u8) ?usize {
        if (text.len == 0) return null;

        for (0..text.len) |i| {
            if (self.matchLen(text[i..]) != null) return i;
        }

        return null;
    }

    /// Count number of non-overlapping, non-empty matches in text
    ///
    /// Scanning left to right, each match is the shortest accepted prefix at its start
    /// index and the scan resumes right after it.
    /// Time: O(n × m) where n is text length, m is pattern length | Space: O(1)
    pub fn count(self: *const DFA, text: []const u8) usize {
        var result: usize = 0;
        var i: usize = 0;

        // Each pass advances `i` by at least one, so `text.len` passes suffice.
        for (0..text.len) |_| {
            if (i >= text.len) break;

            const len = self.matchLen(text[i..]) orelse 0;
            if (len > 0) result += 1;
            i += @max(len, 1);
        }

        assert(i >= text.len);
        return result;
    }

    /// Validate DFA structure
    /// Time: O(|states| × |transitions|) | Space: O(1)
    pub fn validate(self: *const DFA) bool {
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
    const gpa = testing.allocator;

    var dfa = try DFA.fromLiteral(gpa, "hello");
    defer dfa.deinit();

    try testing.expect(dfa.match("hello"));
    try testing.expect(!dfa.match("hell"));
    try testing.expect(!dfa.match("helloa"));
    try testing.expect(!dfa.match("Hello"));
    try testing.expect(!dfa.match(""));
}

test "DFA: empty pattern" {
    const gpa = testing.allocator;

    var dfa = try DFA.fromLiteral(gpa, "");
    defer dfa.deinit();

    try testing.expect(dfa.match(""));
    try testing.expect(!dfa.match("a"));
}

test "DFA: single character" {
    const gpa = testing.allocator;

    var dfa = try DFA.fromLiteral(gpa, "x");
    defer dfa.deinit();

    try testing.expect(dfa.match("x"));
    try testing.expect(!dfa.match(""));
    try testing.expect(!dfa.match("xy"));
    try testing.expect(!dfa.match("y"));
}

test "DFA: prefix matching" {
    const gpa = testing.allocator;

    var dfa = try DFA.fromPrefix(gpa, "pre");
    defer dfa.deinit();

    try testing.expect(dfa.match("pre"));
    try testing.expect(dfa.match("prefix"));
    try testing.expect(dfa.match("pre123"));
    try testing.expect(!dfa.match("pr"));
    try testing.expect(!dfa.match("xpre"));
}

test "DFA: empty prefix (accept all)" {
    const gpa = testing.allocator;

    var dfa = try DFA.fromPrefix(gpa, "");
    defer dfa.deinit();

    try testing.expect(dfa.match(""));
    try testing.expect(dfa.match("anything"));
    try testing.expect(dfa.match("xyz"));
}

test "DFA: suffix matching" {
    const gpa = testing.allocator;

    var dfa = try DFA.fromSuffix(gpa, "end");
    defer dfa.deinit();

    try testing.expect(dfa.match("end"));
    try testing.expect(dfa.match("theend"));
    try testing.expect(dfa.match("xxxend"));
    try testing.expect(!dfa.match("ending"));
    try testing.expect(!dfa.match("en"));
}

test "DFA: stateful step matching" {
    const gpa = testing.allocator;

    var dfa = try DFA.fromLiteral(gpa, "abc");
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
    const gpa = testing.allocator;

    var dfa = try DFA.fromLiteral(gpa, "cat");
    defer dfa.deinit();

    try testing.expectEqual(@as(?usize, 0), dfa.search("cat"));
    try testing.expectEqual(@as(?usize, 4), dfa.search("the cat sat"));
    try testing.expectEqual(@as(?usize, null), dfa.search("dog"));
}

test "DFA: validate structure" {
    const gpa = testing.allocator;

    var dfa = try DFA.fromLiteral(gpa, "test");
    defer dfa.deinit();

    try testing.expect(dfa.validate());
}

test "DFA: case sensitivity" {
    const gpa = testing.allocator;

    var dfa = try DFA.fromLiteral(gpa, "Test");
    defer dfa.deinit();

    try testing.expect(dfa.match("Test"));
    try testing.expect(!dfa.match("test"));
    try testing.expect(!dfa.match("TEST"));
}

test "DFA: manual construction" {
    const gpa = testing.allocator;

    var dfa = DFA.init(gpa);
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
    const gpa = testing.allocator;

    var dfa = try DFA.fromLiteral(gpa, "ok");
    defer dfa.deinit();

    try testing.expect(dfa.accepts("ok"));
    try testing.expect(!dfa.accepts("not"));
}

test "DFA: suffix matching recognises overlapping occurrences" {
    const gpa = testing.allocator;

    var end = try DFA.fromSuffix(gpa, "end");
    defer end.deinit();

    try testing.expect(end.match("endend"));
    try testing.expect(end.match("eend"));
    try testing.expect(!end.match("endx"));
    try testing.expect(!end.match(""));

    // The text "aaab" ends in "aab" even though the third 'a' restarts a partial match.
    var aab = try DFA.fromSuffix(gpa, "aab");
    defer aab.deinit();

    try testing.expect(aab.match("aaab"));
    try testing.expect(aab.match("abaab"));
    try testing.expect(!aab.match("aaba"));
    try testing.expect(aab.validate());
}

test "DFA: empty suffix accepts everything" {
    const gpa = testing.allocator;

    var dfa = try DFA.fromSuffix(gpa, "");
    defer dfa.deinit();

    try testing.expect(dfa.match(""));
    try testing.expect(dfa.match("anything"));
}

test "DFA: matchLen is the shortest accepted prefix" {
    const gpa = testing.allocator;

    var dfa = try DFA.fromPrefix(gpa, "ab");
    defer dfa.deinit();

    try testing.expectEqual(@as(?usize, 2), dfa.matchLen("abzzz"));
    try testing.expectEqual(@as(?usize, null), dfa.matchLen("a"));
    try testing.expectEqual(@as(?usize, null), dfa.matchLen("ba"));

    var empty = try DFA.fromLiteral(gpa, "");
    defer empty.deinit();

    try testing.expectEqual(@as(?usize, 0), empty.matchLen("x"));
}

test "DFA: count skips past each match" {
    const gpa = testing.allocator;

    var dfa = try DFA.fromLiteral(gpa, "cat");
    defer dfa.deinit();

    try testing.expectEqual(@as(usize, 0), dfa.count(""));
    try testing.expectEqual(@as(usize, 1), dfa.count("cat"));
    try testing.expectEqual(@as(usize, 2), dfa.count("a cat and a cat."));
    try testing.expectEqual(@as(usize, 2), dfa.count("catcat"));
    try testing.expectEqual(@as(usize, 0), dfa.count("ca t"));
}

test "DFA: no leaks when an allocation fails" {
    const Runner = struct {
        fn run(gpa: Allocator) !void {
            var literal = try DFA.fromLiteral(gpa, "ab");
            defer literal.deinit();

            var prefix = try DFA.fromPrefix(gpa, "a");
            defer prefix.deinit();

            var suffix = try DFA.fromSuffix(gpa, "ab");
            defer suffix.deinit();

            try testing.expect(literal.match("ab"));
            try testing.expect(prefix.match("abc"));
            try testing.expect(suffix.match("xab"));
        }
    };

    try testing.checkAllAllocationFailures(testing.allocator, Runner.run, .{});
}
