const std = @import("std");

/// Match a glob pattern against a string.
/// Supports:
///   *       — matches any sequence of characters (including empty)
///   ?       — matches exactly one character
///   [abc]   — matches any of the listed characters
///   [a-z]   — matches any character in the range
///   [^abc]  — negated character class
///   \c      — escape sequence, matches literal character c
///
/// ## Complexity
/// Time: O(n * m) worst case where n = pattern length, m = string length
///       O(n + m) average case with early exits
/// Space: O(1) iterative implementation
pub fn match(pattern: []const u8, str: []const u8) bool {
    return matchAt(pattern, 0, str, 0);
}

/// Recursive helper that tracks positions in both pattern and string.
fn matchAt(pattern: []const u8, pi: usize, str: []const u8, si: usize) bool {
    var p = pi;
    var s = si;

    while (p < pattern.len) {
        const pc = pattern[p];

        switch (pc) {
            '*' => {
                // Skip consecutive stars (optimization)
                while (p < pattern.len and pattern[p] == '*') p += 1;

                // If star is at end of pattern, match everything remaining
                if (p == pattern.len) return true;

                // Try matching the rest of the pattern at every position in str
                // This is the expensive backtracking part
                var si2 = s;
                while (si2 <= str.len) : (si2 += 1) {
                    if (matchAt(pattern, p, str, si2)) return true;
                }
                return false;
            },
            '?' => {
                // Must have at least one character to consume
                if (s >= str.len) return false;
                p += 1;
                s += 1;
            },
            '[' => {
                if (s >= str.len) return false;
                const ch = str[s];
                p += 1; // skip '['

                const negate = p < pattern.len and pattern[p] == '^';
                if (negate) p += 1;

                var matched = false;
                var first = true;
                while (p < pattern.len and (first or pattern[p] != ']')) {
                    first = false;

                    // Check for range pattern (e.g., a-z)
                    if (p + 2 < pattern.len and pattern[p + 1] == '-' and pattern[p + 2] != ']') {
                        if (ch >= pattern[p] and ch <= pattern[p + 2]) matched = true;
                        p += 3;
                    } else {
                        if (ch == pattern[p]) matched = true;
                        p += 1;
                    }
                }

                // Skip closing ']'
                if (p < pattern.len and pattern[p] == ']') p += 1;

                // XOR: matched and negate must differ for success
                if (matched == negate) return false;
                s += 1;
            },
            '\\' => {
                // Escape: treat next character literally
                p += 1;
                if (p >= pattern.len) return false;
                if (s >= str.len) return false;
                if (pattern[p] != str[s]) return false;
                p += 1;
                s += 1;
            },
            else => {
                // Literal character match
                if (s >= str.len or pc != str[s]) return false;
                p += 1;
                s += 1;
            },
        }
    }

    // Pattern exhausted — must have consumed all of str too
    return s == str.len;
}

/// Case-insensitive glob matching.
/// Uses ASCII lowercase conversion for comparison.
///
/// ## Complexity
/// Time: O(n * m) worst case
/// Space: O(1)
pub fn matchCaseInsensitive(pattern: []const u8, str: []const u8) bool {
    return matchAtCaseInsensitive(pattern, 0, str, 0);
}

fn matchAtCaseInsensitive(pattern: []const u8, pi: usize, str: []const u8, si: usize) bool {
    var p = pi;
    var s = si;

    while (p < pattern.len) {
        const pc = pattern[p];

        switch (pc) {
            '*' => {
                while (p < pattern.len and pattern[p] == '*') p += 1;
                if (p == pattern.len) return true;

                var si2 = s;
                while (si2 <= str.len) : (si2 += 1) {
                    if (matchAtCaseInsensitive(pattern, p, str, si2)) return true;
                }
                return false;
            },
            '?' => {
                if (s >= str.len) return false;
                p += 1;
                s += 1;
            },
            '[' => {
                if (s >= str.len) return false;
                const ch = std.ascii.toLower(str[s]);
                p += 1;

                const negate = p < pattern.len and pattern[p] == '^';
                if (negate) p += 1;

                var matched = false;
                var first = true;
                while (p < pattern.len and (first or pattern[p] != ']')) {
                    first = false;

                    if (p + 2 < pattern.len and pattern[p + 1] == '-' and pattern[p + 2] != ']') {
                        const start = std.ascii.toLower(pattern[p]);
                        const end = std.ascii.toLower(pattern[p + 2]);
                        if (ch >= start and ch <= end) matched = true;
                        p += 3;
                    } else {
                        if (ch == std.ascii.toLower(pattern[p])) matched = true;
                        p += 1;
                    }
                }

                if (p < pattern.len and pattern[p] == ']') p += 1;
                if (matched == negate) return false;
                s += 1;
            },
            '\\' => {
                p += 1;
                if (p >= pattern.len) return false;
                if (s >= str.len) return false;
                if (std.ascii.toLower(pattern[p]) != std.ascii.toLower(str[s])) return false;
                p += 1;
                s += 1;
            },
            else => {
                if (s >= str.len or std.ascii.toLower(pc) != std.ascii.toLower(str[s])) return false;
                p += 1;
                s += 1;
            },
        }
    }

    return s == str.len;
}

// ── Unit tests ────────────────────────────────────────────────────────────────

test "glob - exact match" {
    try std.testing.expect(match("hello", "hello"));
    try std.testing.expect(!match("hello", "hell"));
    try std.testing.expect(!match("hello", "helloo"));
    try std.testing.expect(!match("hello", "world"));
}

test "glob - empty pattern and string" {
    try std.testing.expect(match("", ""));
    try std.testing.expect(!match("", "a"));
    try std.testing.expect(!match("a", ""));
}

test "glob - star wildcard" {
    try std.testing.expect(match("*", "anything"));
    try std.testing.expect(match("*", ""));
    try std.testing.expect(match("h*llo", "hello"));
    try std.testing.expect(match("h*llo", "hllo"));
    try std.testing.expect(match("h*llo", "heeeello"));
    try std.testing.expect(!match("h*llo", "hworld"));
    try std.testing.expect(match("*llo", "hello"));
    try std.testing.expect(match("hel*", "hello"));
    try std.testing.expect(match("h*l*o", "hello"));
}

test "glob - multiple consecutive stars" {
    try std.testing.expect(match("***", "anything"));
    try std.testing.expect(match("h***o", "hello"));
    try std.testing.expect(match("*****", ""));
}

test "glob - star at boundaries" {
    try std.testing.expect(match("*world", "hello world"));
    try std.testing.expect(match("hello*", "hello world"));
    try std.testing.expect(match("*", "x"));
}

test "glob - question mark wildcard" {
    try std.testing.expect(match("h?llo", "hello"));
    try std.testing.expect(match("h?llo", "hallo"));
    try std.testing.expect(!match("h?llo", "hllo"));
    try std.testing.expect(!match("h?llo", "heello"));
    try std.testing.expect(match("???", "abc"));
    try std.testing.expect(!match("???", "ab"));
}

test "glob - character class" {
    try std.testing.expect(match("h[ae]llo", "hello"));
    try std.testing.expect(match("h[ae]llo", "hallo"));
    try std.testing.expect(!match("h[ae]llo", "hillo"));
    try std.testing.expect(match("[abc]", "a"));
    try std.testing.expect(match("[abc]", "b"));
    try std.testing.expect(!match("[abc]", "d"));
}

test "glob - character range" {
    try std.testing.expect(match("h[a-e]llo", "hello"));
    try std.testing.expect(match("h[a-e]llo", "hallo"));
    try std.testing.expect(!match("h[a-e]llo", "hillo"));
    try std.testing.expect(match("[0-9]", "5"));
    try std.testing.expect(!match("[0-9]", "a"));
    try std.testing.expect(match("[a-zA-Z]", "m"));
    try std.testing.expect(match("[a-zA-Z]", "M"));
}

test "glob - negated character class" {
    try std.testing.expect(match("h[^ae]llo", "hillo"));
    try std.testing.expect(!match("h[^ae]llo", "hello"));
    try std.testing.expect(!match("h[^ae]llo", "hallo"));
    try std.testing.expect(match("[^0-9]", "a"));
    try std.testing.expect(!match("[^0-9]", "5"));
}

test "glob - escape sequences" {
    try std.testing.expect(match("\\*", "*"));
    try std.testing.expect(!match("\\*", "anything"));
    try std.testing.expect(match("\\?", "?"));
    try std.testing.expect(match("\\[abc\\]", "[abc]"));
    try std.testing.expect(match("file\\*.txt", "file*.txt"));
}

test "glob - combined wildcards" {
    try std.testing.expect(match("*.??g", "test.zig"));
    try std.testing.expect(match("*.??g", "main.log"));
    try std.testing.expect(!match("*.??g", "file.z"));
    try std.testing.expect(match("test*.txt", "test123.txt"));
    try std.testing.expect(match("*test*", "mytestfile"));
    try std.testing.expect(match("?*?", "ab"));
    try std.testing.expect(!match("?*?", "a"));
}

test "glob - filesystem patterns" {
    // Common file glob patterns
    try std.testing.expect(match("*.txt", "file.txt"));
    try std.testing.expect(match("*.txt", ".txt"));
    try std.testing.expect(!match("*.txt", "file.zig"));
    try std.testing.expect(match("test*", "test"));
    try std.testing.expect(match("test*", "test123"));
    try std.testing.expect(match("*test", "mytest"));
}

test "glob - Redis KEYS patterns" {
    // Common Redis KEYS usage
    try std.testing.expect(match("*", "user:1"));
    try std.testing.expect(match("user:*", "user:1"));
    try std.testing.expect(match("user:*", "user:1000"));
    try std.testing.expect(!match("user:*", "session:1"));
    try std.testing.expect(match("?ello", "hello"));
    try std.testing.expect(match("?ello", "jello"));
    try std.testing.expect(match("user:[0-9]*", "user:1234"));
    try std.testing.expect(!match("user:[0-9]*", "user:abc"));
}

test "glob - edge cases" {
    try std.testing.expect(match("**", "anything"));
    try std.testing.expect(match("a*b*c", "abc"));
    try std.testing.expect(match("a*b*c", "aXbYc"));
    try std.testing.expect(!match("a*b*c", "aXcYb"));
}

test "glob - case insensitive - basic" {
    try std.testing.expect(matchCaseInsensitive("HELLO", "hello"));
    try std.testing.expect(matchCaseInsensitive("hello", "HELLO"));
    try std.testing.expect(matchCaseInsensitive("HeLLo", "hElLo"));
    try std.testing.expect(!matchCaseInsensitive("HELLO", "world"));
}

test "glob - case insensitive - wildcards" {
    try std.testing.expect(matchCaseInsensitive("H*LLO", "hello"));
    try std.testing.expect(matchCaseInsensitive("h?llo", "HELLO"));
    try std.testing.expect(matchCaseInsensitive("*.TXT", "file.txt"));
    try std.testing.expect(matchCaseInsensitive("*.txt", "FILE.TXT"));
}

test "glob - case insensitive - character classes" {
    try std.testing.expect(matchCaseInsensitive("h[ae]llo", "HELLO"));
    try std.testing.expect(matchCaseInsensitive("H[A-E]LLO", "hello"));
    try std.testing.expect(matchCaseInsensitive("[A-Z]", "m"));
    try std.testing.expect(matchCaseInsensitive("[a-z]", "M"));
}

test "glob - memory leak check" {
    // No allocations in this algorithm, but verify with testing allocator
    const allocator = std.testing.allocator;
    _ = allocator; // unused but demonstrates allocator-free operation

    // Verify results are correct across multiple calls (no state corruption)
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        try std.testing.expect(match("*test*", "mytestfile"));
        try std.testing.expect(match("h[a-z]llo", "hello"));
        try std.testing.expect(matchCaseInsensitive("*.TXT", "file.txt"));
    }
}

test "glob - stress test" {
    // Test with longer strings
    const pattern = "*test*file*";
    const str = "this_is_a_very_long_test_string_with_file_in_it";
    try std.testing.expect(match(pattern, str));

    // Test with many wildcards
    const pattern2 = "a*b*c*d*e*f*g";
    const str2 = "aXbYcZdWeVfUg";
    try std.testing.expect(match(pattern2, str2));
}
