/// Automata algorithms: finite state machines and pattern recognition
///
/// This module provides implementations of deterministic and non-deterministic
/// finite automata for efficient pattern matching and text processing.
///
/// ## Algorithms
///
/// ### NFA (Non-deterministic Finite Automaton)
/// - **Thompson's construction** for regex compilation
/// - Supports epsilon transitions, wildcards, and Kleene star
/// - Multiple active states during matching
/// - Time: O(n × |states|) for matching, Space: O(|states|)
/// - Use cases: regex engines, flexible pattern matching, theoretical CS
///
/// ### DFA (Deterministic Finite Automaton)
/// - **Subset construction** for determinization
/// - Exactly one active state at any time
/// - Faster matching than NFA (no backtracking)
/// - Time: O(n) for matching, Space: O(|states| × |alphabet|)
/// - Use cases: lexical analysis, protocol state machines, fast validation
///
/// ## When to Use
///
/// **Use NFA when**:
/// - Pattern has many epsilon transitions (e.g., complex regex)
/// - Construction time is critical (DFA conversion can be exponential)
/// - Memory is limited (NFA is often smaller than equivalent DFA)
/// - Need theoretical model for regex semantics
///
/// **Use DFA when**:
/// - Matching performance is critical (hot path in scanner)
/// - Pattern is simple (e.g., literal string, prefix/suffix)
/// - Multiple matches against same pattern (amortize construction cost)
/// - Need predictable O(n) worst-case matching time
///
/// ## Examples
///
/// ```zig
/// const automata = @import("zuda").algorithms.automata;
///
/// // Example 1: NFA with regex-like pattern
/// var nfa = try automata.NFA.fromWildcard(allocator, "a.*b");
/// defer nfa.deinit();
/// try testing.expect(try nfa.match("aXYZb")); // true
/// try testing.expect(!try nfa.match("aXYZ")); // false
///
/// // Example 2: DFA for exact matching
/// var dfa = try automata.DFA.fromLiteral(allocator, "hello");
/// defer dfa.deinit();
/// try testing.expect(dfa.match("hello")); // true
/// try testing.expect(!dfa.match("hell")); // false
///
/// // Example 3: DFA for prefix matching
/// var dfa_prefix = try automata.DFA.fromPrefix(allocator, "http");
/// defer dfa_prefix.deinit();
/// try testing.expect(dfa_prefix.match("http")); // true
/// try testing.expect(dfa_prefix.match("https")); // true
/// try testing.expect(!dfa_prefix.match("ftp")); // false
///
/// // Example 4: Stateful DFA (streaming)
/// var dfa_stream = try automata.DFA.fromLiteral(allocator, "abc");
/// defer dfa_stream.deinit();
/// dfa_stream.reset();
/// _ = dfa_stream.step('a'); // false (not accepting yet)
/// _ = dfa_stream.step('b'); // false
/// const accepted = dfa_stream.step('c'); // true (accepting state)
/// ```
///
/// ## Algorithm Selection Guide
///
/// | Feature | NFA | DFA |
/// |---------|-----|-----|
/// | Match Time | O(n × \|states\|) | O(n) |
/// | Space | O(m) | O(m × 256) worst case |
/// | Construction | O(m) | O(2^m) worst case |
/// | Epsilon Transitions | Yes | No |
/// | Backtracking | Yes | No |
/// | Streaming | Possible | Easy |
/// | Predictable | No | Yes |
///
/// Where m = pattern length, n = text length
///
/// ## Implementation Notes
///
/// - NFA uses epsilon-closure for state exploration
/// - DFA uses transition table (HashMap for sparse, array for dense)
/// - Both support validate() for structural integrity checks
/// - Thread-safety: automata are immutable after construction (match is const)
/// - Memory: use testing.allocator to detect leaks in tests

const std = @import("std");

/// Non-deterministic Finite Automaton
pub const NFA = @import("automata/nfa.zig").NFA;

/// Deterministic Finite Automaton
pub const DFA = @import("automata/dfa.zig").DFA;

// Re-export types for convenience
pub const NFAState = @import("automata/nfa.zig").State;
pub const NFATransition = @import("automata/nfa.zig").Transition;
pub const DFAState = @import("automata/dfa.zig").State;

// Tests
const testing = std.testing;

test "automata: NFA basic usage" {
    const allocator = testing.allocator;

    var nfa = try NFA.fromLiteral(allocator, "test");
    defer nfa.deinit();

    try testing.expect(try nfa.match("test"));
    try testing.expect(!try nfa.match("testing"));
}

test "automata: DFA basic usage" {
    const allocator = testing.allocator;

    var dfa = try DFA.fromLiteral(allocator, "test");
    defer dfa.deinit();

    try testing.expect(dfa.match("test"));
    try testing.expect(!dfa.match("testing"));
}

test "automata: NFA vs DFA equivalence" {
    const allocator = testing.allocator;

    const pattern = "hello";

    var nfa = try NFA.fromLiteral(allocator, pattern);
    defer nfa.deinit();

    var dfa = try DFA.fromLiteral(allocator, pattern);
    defer dfa.deinit();

    // Both should accept same inputs
    try testing.expect(try nfa.match("hello") == dfa.match("hello"));
    try testing.expect(try nfa.match("hell") == dfa.match("hell"));
    try testing.expect(try nfa.match("") == dfa.match(""));
}
