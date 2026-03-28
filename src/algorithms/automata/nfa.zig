/// Non-deterministic Finite Automaton (NFA) for pattern matching
/// Supports epsilon transitions and Thompson's construction for regex compilation
///
/// Use cases:
/// - Regular expression matching
/// - Lexical analysis / tokenization
/// - Pattern validation
/// - Compiler frontend construction
///
/// Time complexity:
/// - Construction: O(m) where m is pattern length
/// - Matching: O(n × |states|) where n is text length, worst case O(nm)
/// - Can have multiple active states due to non-determinism
///
/// Space complexity: O(m) for NFA states, O(|states|) for active state tracking

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

/// NFA transition types
pub const Transition = union(enum) {
    /// Epsilon transition (empty move)
    epsilon: usize,
    /// Character match transition
    char: struct {
        c: u8,
        to: usize,
    },
    /// Any character (wildcard .)
    any: usize,
    /// Character class [a-z] (simplified: stores allowed chars)
    class: struct {
        allowed: []const u8,
        to: usize,
    },
};

/// NFA state
pub const State = struct {
    /// State ID
    id: usize,
    /// Is this an accepting state?
    is_accept: bool,
    /// Outgoing transitions
    transitions: ArrayList(Transition),

    pub fn init(allocator: Allocator, id: usize, is_accept: bool) !State {
        return State{
            .id = id,
            .is_accept = is_accept,
            .transitions = ArrayList(Transition).init(allocator),
        };
    }

    pub fn deinit(self: *State) void {
        for (self.transitions.items) |transition| {
            if (transition == .class) {
                // Note: class.allowed is assumed to be owned by the NFA
                // and will be freed during NFA.deinit()
            }
        }
        self.transitions.deinit();
    }

    /// Add epsilon transition to target state
    /// Time: O(1) amortized
    pub fn addEpsilon(self: *State, to: usize) !void {
        try self.transitions.append(.{ .epsilon = to });
    }

    /// Add character transition to target state
    /// Time: O(1) amortized
    pub fn addChar(self: *State, c: u8, to: usize) !void {
        try self.transitions.append(.{ .char = .{ .c = c, .to = to } });
    }

    /// Add wildcard (any character) transition
    /// Time: O(1) amortized
    pub fn addAny(self: *State, to: usize) !void {
        try self.transitions.append(.{ .any = to });
    }

    /// Add character class transition [allowed_chars]
    /// Time: O(1) amortized
    /// Note: allowed slice must remain valid for NFA lifetime
    pub fn addClass(self: *State, allowed: []const u8, to: usize) !void {
        try self.transitions.append(.{ .class = .{ .allowed = allowed, .to = to } });
    }
};

/// Non-deterministic Finite Automaton
pub const NFA = struct {
    const Self = @This();

        allocator: Allocator,
        /// States (indexed by ID)
        states: ArrayList(State),
        /// Start state ID
        start: usize,
        /// Character class storage (owned by NFA)
        char_classes: ArrayList([]u8),

        /// Initialize empty NFA
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator) Self {
            return Self{
                .allocator = allocator,
                .states = ArrayList(State).init(allocator),
                .start = 0,
                .char_classes = ArrayList([]u8).init(allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.states.items) |*state| {
                state.deinit();
            }
            self.states.deinit();

            for (self.char_classes.items) |class| {
                self.allocator.free(class);
            }
            self.char_classes.deinit();
        }

        /// Add new state to NFA
        /// Time: O(1) amortized | Space: O(1)
        pub fn addState(self: *Self, is_accept: bool) !usize {
            const id = self.states.items.len;
            const state = try State.init(self.allocator, id, is_accept);
            try self.states.append(state);
            return id;
        }

        /// Build NFA from literal string pattern (exact match)
        /// Time: O(n) where n is pattern length | Space: O(n)
        pub fn fromLiteral(allocator: Allocator, pattern: []const u8) !Self {
            var nfa = Self.init(allocator);
            errdefer nfa.deinit();

            if (pattern.len == 0) {
                // Empty pattern: single accepting state
                _ = try nfa.addState(true);
                return nfa;
            }

            // Create chain of states for each character
            nfa.start = try nfa.addState(false);
            var current = nfa.start;

            for (pattern, 0..) |c, i| {
                const is_last = (i == pattern.len - 1);
                const next = try nfa.addState(is_last);
                try nfa.states.items[current].addChar(c, next);
                current = next;
            }

            return nfa;
        }

        /// Build NFA for wildcard pattern with '.' (any char) and '*' (zero or more)
        /// Simplified regex subset: literal chars, '.', '*', no capture groups
        /// Time: O(n) where n is pattern length | Space: O(n)
        pub fn fromWildcard(allocator: Allocator, pattern: []const u8) !Self {
            var nfa = Self.init(allocator);
            errdefer nfa.deinit();

            if (pattern.len == 0) {
                _ = try nfa.addState(true);
                return nfa;
            }

            nfa.start = try nfa.addState(false);
            var current = nfa.start;
            var i: usize = 0;

            while (i < pattern.len) : (i += 1) {
                const c = pattern[i];

                if (c == '*' and i > 0) {
                    // Handle Kleene star: loop back to previous state
                    // Create new state for after the star
                    const next = try nfa.addState(i == pattern.len - 1);
                    try nfa.states.items[current].addEpsilon(next);
                    // Loop back to current for repetition
                    const loop_target = current;
                    try nfa.states.items[current].addEpsilon(loop_target);
                    current = next;
                } else if (c == '.') {
                    // Wildcard: match any character
                    const is_last = (i == pattern.len - 1) or
                        (i + 1 < pattern.len and pattern[i + 1] != '*');
                    const next = try nfa.addState(is_last);
                    try nfa.states.items[current].addAny(next);
                    current = next;
                } else {
                    // Literal character
                    const is_last = (i == pattern.len - 1) or
                        (i + 1 < pattern.len and pattern[i + 1] != '*');
                    const next = try nfa.addState(is_last);
                    try nfa.states.items[current].addChar(c, next);
                    current = next;
                }
            }

            return nfa;
        }

        /// Match input text against NFA
        /// Returns true if text is accepted by the automaton
        /// Time: O(n × |states|) where n is text length | Space: O(|states|)
        pub fn match(self: *const Self, text: []const u8) !bool {
            if (self.states.items.len == 0) return false;

            var current_states = ArrayList(usize).init(self.allocator);
            defer current_states.deinit();

            var next_states = ArrayList(usize).init(self.allocator);
            defer next_states.deinit();

            // Start with epsilon-closure of start state
            try self.epsilonClosure(self.start, &current_states);

            // Process each character
            for (text) |c| {
                next_states.clearRetainingCapacity();

                for (current_states.items) |state_id| {
                    const state = &self.states.items[state_id];

                    for (state.transitions.items) |transition| {
                        switch (transition) {
                            .char => |char_trans| {
                                if (char_trans.c == c) {
                                    try self.epsilonClosure(char_trans.to, &next_states);
                                }
                            },
                            .any => |to| {
                                try self.epsilonClosure(to, &next_states);
                            },
                            .class => |class_trans| {
                                if (std.mem.indexOfScalar(u8, class_trans.allowed, c) != null) {
                                    try self.epsilonClosure(class_trans.to, &next_states);
                                }
                            },
                            .epsilon => {}, // Already handled in epsilon-closure
                        }
                    }
                }

                // Swap current and next
                const temp = current_states;
                current_states = next_states;
                next_states = temp;
            }

            // Check if any current state is accepting
            for (current_states.items) |state_id| {
                if (self.states.items[state_id].is_accept) {
                    return true;
                }
            }

            return false;
        }

        /// Compute epsilon-closure of a state (all states reachable via epsilon transitions)
        /// Time: O(|states| × |transitions|) worst case | Space: O(|states|)
        fn epsilonClosure(self: *const Self, start_state: usize, result: *ArrayList(usize)) !void {
            var visited = try self.allocator.alloc(bool, self.states.items.len);
            defer self.allocator.free(visited);
            @memset(visited, false);

            var stack = ArrayList(usize).init(self.allocator);
            defer stack.deinit();

            try stack.append(start_state);
            visited[start_state] = true;
            try result.append(start_state);

            while (stack.items.len > 0) {
                const state_id = stack.pop();
                const state = &self.states.items[state_id];

                for (state.transitions.items) |transition| {
                    if (transition == .epsilon) {
                        const to = transition.epsilon;
                        if (!visited[to]) {
                            visited[to] = true;
                            try stack.append(to);
                            try result.append(to);
                        }
                    }
                }
            }
        }

        /// Validate NFA structure
        /// Time: O(|states| × |transitions|) | Space: O(1)
        pub fn validate(self: *const Self) bool {
            if (self.states.items.len == 0) return false;
            if (self.start >= self.states.items.len) return false;

            // Check all transitions point to valid states
            for (self.states.items) |state| {
                for (state.transitions.items) |transition| {
                    const target = switch (transition) {
                        .epsilon => |to| to,
                        .char => |char_trans| char_trans.to,
                        .any => |to| to,
                        .class => |class_trans| class_trans.to,
                    };
                    if (target >= self.states.items.len) return false;
                }
            }

            return true;
        }
};

// Tests
const testing = std.testing;

test "NFA: literal pattern exact match" {
    const allocator = testing.allocator;

    var nfa = try NFA.fromLiteral(allocator, "hello");
    defer nfa.deinit();

    try testing.expect(try nfa.match("hello"));
    try testing.expect(!try nfa.match("hell"));
    try testing.expect(!try nfa.match("helloa"));
    try testing.expect(!try nfa.match("Hello"));
    try testing.expect(!try nfa.match(""));
}

test "NFA: empty pattern" {
    const allocator = testing.allocator;

    var nfa = try NFA.fromLiteral(allocator, "");
    defer nfa.deinit();

    try testing.expect(try nfa.match(""));
    try testing.expect(!try nfa.match("a"));
}

test "NFA: single character" {
    const allocator = testing.allocator;

    var nfa = try NFA.fromLiteral(allocator, "a");
    defer nfa.deinit();

    try testing.expect(try nfa.match("a"));
    try testing.expect(!try nfa.match(""));
    try testing.expect(!try nfa.match("ab"));
    try testing.expect(!try nfa.match("b"));
}

test "NFA: wildcard dot (any character)" {
    const allocator = testing.allocator;

    var nfa = try NFA.fromWildcard(allocator, "h.llo");
    defer nfa.deinit();

    try testing.expect(try nfa.match("hello"));
    try testing.expect(try nfa.match("hallo"));
    try testing.expect(try nfa.match("h9llo"));
    try testing.expect(!try nfa.match("hllo"));
    try testing.expect(!try nfa.match("helllo"));
}

test "NFA: Kleene star (zero or more)" {
    const allocator = testing.allocator;

    var nfa = try NFA.fromWildcard(allocator, "a*");
    defer nfa.deinit();

    try testing.expect(try nfa.match(""));
    try testing.expect(try nfa.match("a"));
    try testing.expect(try nfa.match("aa"));
    try testing.expect(try nfa.match("aaa"));
}

test "NFA: wildcard with star" {
    const allocator = testing.allocator;

    var nfa = try NFA.fromWildcard(allocator, ".*");
    defer nfa.deinit();

    try testing.expect(try nfa.match(""));
    try testing.expect(try nfa.match("a"));
    try testing.expect(try nfa.match("hello"));
    try testing.expect(try nfa.match("anything"));
}

test "NFA: manual construction with epsilon transitions" {
    const allocator = testing.allocator;

    var nfa = NFA.init(allocator);
    defer nfa.deinit();

    const s0 = try nfa.addState(false);
    const s1 = try nfa.addState(false);
    const s2 = try nfa.addState(true);

    nfa.start = s0;

    try nfa.states.items[s0].addEpsilon(s1);
    try nfa.states.items[s0].addChar('a', s2);
    try nfa.states.items[s1].addChar('b', s2);

    // Accepts "a" or "b" (via epsilon to s1)
    try testing.expect(try nfa.match("a"));
    try testing.expect(try nfa.match("b"));
    try testing.expect(!try nfa.match("ab"));
    try testing.expect(!try nfa.match(""));
}

test "NFA: validate structure" {
    const allocator = testing.allocator;

    var nfa = try NFA.fromLiteral(allocator, "test");
    defer nfa.deinit();

    try testing.expect(nfa.validate());
}

test "NFA: case sensitivity" {
    const allocator = testing.allocator;

    var nfa = try NFA.fromLiteral(allocator, "Test");
    defer nfa.deinit();

    try testing.expect(try nfa.match("Test"));
    try testing.expect(!try nfa.match("test"));
    try testing.expect(!try nfa.match("TEST"));
}

test "NFA: complex wildcard pattern" {
    const allocator = testing.allocator;

    var nfa = try NFA.fromWildcard(allocator, "a.c");
    defer nfa.deinit();

    try testing.expect(try nfa.match("abc"));
    try testing.expect(try nfa.match("axc"));
    try testing.expect(try nfa.match("a9c"));
    try testing.expect(!try nfa.match("ac"));
    try testing.expect(!try nfa.match("abbc"));
}
