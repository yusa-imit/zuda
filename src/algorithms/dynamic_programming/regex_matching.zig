const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Regular Expression Matching
///
/// Given a string s and a pattern p, implement regular expression matching with support for:
/// - '.' matches any single character
/// - '*' matches zero or more of the preceding element
///
/// The matching should cover the entire input string (not partial).
///
/// Algorithm: Dynamic Programming with three implementation variants
/// - isMatch(): Full DP table — O(n×m) time, O(n×m) space
/// - isMatchOptimized(): Rolling buffer — O(n×m) time, O(m) space
/// - isMatchRecursive(): Recursive DP with memoization — O(n×m) time, O(n×m) space
///
/// Pattern syntax:
/// - '.' matches any single character
/// - '*' matches zero or more of the preceding element (not a standalone wildcard)
/// - All other characters match themselves
///
/// Example:
/// - "aa" matches "a*" (true: '*' matches two 'a')
/// - "ab" matches ".*" (true: '.' matches 'a', '*' matches zero or more of any char)
/// - "aab" matches "c*a*b" (true: 'c*' matches zero 'c', 'a*' matches two 'a', 'b' matches 'b')
///
/// Use cases:
/// - Pattern matching in text editors (regex search)
/// - Compiler lexical analysis (token matching)
/// - Input validation (email, phone format)
/// - Text processing pipelines (sed, grep functionality)
/// - String parsing and extraction
///
/// Time complexity: O(n×m) where n = string length, m = pattern length
/// Space complexity: O(n×m) for full DP, O(m) for optimized, O(n×m) for recursive memo
///
/// Reference: LeetCode #10 (Regular Expression Matching)
pub fn isMatch(s: []const u8, p: []const u8) bool {
    const n = s.len;
    const m = p.len;

    // dp[i][j] = true if s[0..i] matches p[0..j]
    // Allocate 2D DP table using page allocator
    const allocator = std.heap.page_allocator;
    var dp = allocator.alloc([]bool, n + 1) catch unreachable;
    defer allocator.free(dp);

    // Initialize DP table
    var i: usize = 0;
    while (i <= n) : (i += 1) {
        dp[i] = allocator.alloc(bool, m + 1) catch unreachable;
        var j: usize = 0;
        while (j <= m) : (j += 1) {
            dp[i][j] = false;
        }
    }
    defer {
        i = 0;
        while (i <= n) : (i += 1) {
            allocator.free(dp[i]);
        }
    }

    // Base case: empty string matches empty pattern
    dp[0][0] = true;

    // Handle patterns like "a*", "a*b*" that can match empty string
    i = 0;
    while (i < m) : (i += 1) {
        if (p[i] == '*') {
            // '*' matches the previous character zero times
            // So p[0..i+1] can match empty if p[0..i-1] can match empty
            if (i > 0) {
                dp[0][i + 1] = dp[0][i - 1];
            }
        }
    }

    // Fill DP table
    i = 0;
    while (i < n) : (i += 1) {
        var j: usize = 0;
        while (j < m) : (j += 1) {
            if (p[j] == '*') {
                // '*' matches zero or more of the previous character
                if (j > 0) {
                    // Case 1: Match zero of the previous character
                    // s[0..i+1] matches p[0..j-1] (skip current char and '*')
                    dp[i + 1][j + 1] = dp[i + 1][j - 1];

                    // Case 2: Match one or more of the previous character
                    // If current char matches, we can use the '*' to match it
                    // and check if s[0..i] matches p[0..j+1] (same pattern)
                    if (p[j - 1] == '.' or p[j - 1] == s[i]) {
                        dp[i + 1][j + 1] = dp[i + 1][j + 1] or dp[i][j + 1];
                    }
                }
            } else if (p[j] == '.' or p[j] == s[i]) {
                // '.' matches any character, or exact match
                dp[i + 1][j + 1] = dp[i][j];
            }
        }
    }

    return dp[n][m];
}

/// Regular Expression Matching (Space-Optimized)
///
/// Same as isMatch() but uses rolling buffer to reduce space complexity from O(n×m) to O(m).
/// Only keeps current and previous row of DP table.
///
/// Time: O(n×m)
/// Space: O(m)
pub fn isMatchOptimized(s: []const u8, p: []const u8) bool {
    const n = s.len;
    const m = p.len;

    // Use two rows: previous and current
    const allocator = std.heap.page_allocator;
    var prev = allocator.alloc(bool, m + 1) catch unreachable;
    defer allocator.free(prev);
    var curr = allocator.alloc(bool, m + 1) catch unreachable;
    defer allocator.free(curr);

    // Initialize rows
    var j: usize = 0;
    while (j <= m) : (j += 1) {
        prev[j] = false;
        curr[j] = false;
    }

    // Base case: empty string matches empty pattern
    prev[0] = true;

    // Handle patterns like "a*", "a*b*" that can match empty string
    var i: usize = 0;
    while (i < m) : (i += 1) {
        if (p[i] == '*') {
            if (i > 0) {
                prev[i + 1] = prev[i - 1];
            }
        }
    }

    // Fill DP table row by row
    i = 0;
    while (i < n) : (i += 1) {
        // Reset current row
        curr[0] = false;

        j = 0;
        while (j < m) : (j += 1) {
            if (p[j] == '*') {
                if (j > 0) {
                    // Case 1: Match zero of the previous character
                    curr[j + 1] = curr[j - 1];

                    // Case 2: Match one or more of the previous character
                    if (p[j - 1] == '.' or p[j - 1] == s[i]) {
                        curr[j + 1] = curr[j + 1] or prev[j + 1];
                    }
                }
            } else if (p[j] == '.' or p[j] == s[i]) {
                curr[j + 1] = prev[j];
            } else {
                curr[j + 1] = false;
            }
        }

        // Swap rows
        const temp = prev;
        prev = curr;
        curr = temp;
    }

    return prev[m];
}

/// Regular Expression Matching (Recursive with Memoization)
///
/// Recursive approach with memoization for intuitive understanding.
/// Explores all possible matches recursively, caches results to avoid recomputation.
///
/// Time: O(n×m)
/// Space: O(n×m) for memoization table
pub fn isMatchRecursive(allocator: Allocator, s: []const u8, p: []const u8) !bool {
    const n = s.len;
    const m = p.len;

    // Memoization table: -1 = not computed, 0 = false, 1 = true
    var memo = try allocator.alloc([]i8, n + 1);
    defer allocator.free(memo);

    var i: usize = 0;
    while (i <= n) : (i += 1) {
        memo[i] = try allocator.alloc(i8, m + 1);
        var j: usize = 0;
        while (j <= m) : (j += 1) {
            memo[i][j] = -1;
        }
    }
    defer {
        i = 0;
        while (i <= n) : (i += 1) {
            allocator.free(memo[i]);
        }
    }

    return isMatchHelper(s, p, 0, 0, memo);
}

fn isMatchHelper(s: []const u8, p: []const u8, i: usize, j: usize, memo: [][]i8) bool {
    // Check memo
    if (memo[i][j] != -1) {
        return memo[i][j] == 1;
    }

    var result: bool = undefined;

    // Base case: reached end of pattern
    if (j == p.len) {
        result = (i == s.len);
    } else {
        // Check if current characters match
        const first_match = i < s.len and (p[j] == s[i] or p[j] == '.');

        // Check if next character is '*'
        if (j + 1 < p.len and p[j + 1] == '*') {
            // Case 1: Match zero occurrences (skip current char and '*')
            // Case 2: Match one or more occurrences (if current char matches, advance string index)
            result = isMatchHelper(s, p, i, j + 2, memo) or
                (first_match and isMatchHelper(s, p, i + 1, j, memo));
        } else {
            // No '*', must match current character and advance both indices
            result = first_match and isMatchHelper(s, p, i + 1, j + 1, memo);
        }
    }

    // Store in memo
    memo[i][j] = if (result) @as(i8, 1) else @as(i8, 0);
    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "regex matching - basic examples" {
    // Example 1: "aa" vs "a"
    try testing.expect(!isMatch("aa", "a"));

    // Example 2: "aa" vs "a*"
    try testing.expect(isMatch("aa", "a*"));

    // Example 3: "ab" vs ".*"
    try testing.expect(isMatch("ab", ".*"));
}

test "regex matching - empty string and pattern" {
    // Empty string matches empty pattern
    try testing.expect(isMatch("", ""));

    // Empty string doesn't match non-empty pattern (unless pattern can match empty)
    try testing.expect(!isMatch("", "a"));

    // Empty string can match pattern with '*'
    try testing.expect(isMatch("", "a*"));
    try testing.expect(isMatch("", "a*b*"));
    try testing.expect(isMatch("", ".*"));
}

test "regex matching - single character" {
    // Exact match
    try testing.expect(isMatch("a", "a"));

    // '.' matches any character
    try testing.expect(isMatch("a", "."));

    // Mismatch
    try testing.expect(!isMatch("a", "b"));

    // Pattern longer than string
    try testing.expect(!isMatch("a", "ab"));
}

test "regex matching - dot wildcard" {
    // '.' matches any single character
    try testing.expect(isMatch("a", "."));
    try testing.expect(isMatch("ab", ".."));
    try testing.expect(isMatch("abc", "..."));

    // '.' doesn't match empty
    try testing.expect(!isMatch("", "."));

    // '.' doesn't match multiple characters (unless with '*')
    try testing.expect(!isMatch("aa", "."));
}

test "regex matching - star quantifier" {
    // '*' matches zero occurrences
    try testing.expect(isMatch("", "a*"));
    try testing.expect(isMatch("b", "a*b"));

    // '*' matches one occurrence
    try testing.expect(isMatch("a", "a*"));
    try testing.expect(isMatch("ab", "a*b"));

    // '*' matches multiple occurrences
    try testing.expect(isMatch("aa", "a*"));
    try testing.expect(isMatch("aaa", "a*"));
    try testing.expect(isMatch("aaab", "a*b"));

    // Multiple '*' in pattern
    try testing.expect(isMatch("aab", "a*b*"));
    try testing.expect(isMatch("aabb", "a*b*"));
}

test "regex matching - dot-star combination" {
    // '.*' matches zero or more of any character
    try testing.expect(isMatch("", ".*"));
    try testing.expect(isMatch("a", ".*"));
    try testing.expect(isMatch("ab", ".*"));
    try testing.expect(isMatch("abc", ".*"));

    // '.*' in middle of pattern
    try testing.expect(isMatch("abc", "a.*c"));
    try testing.expect(isMatch("ac", "a.*c"));
    try testing.expect(isMatch("abbbbc", "a.*c"));
}

test "regex matching - complex patterns" {
    // "aab" vs "c*a*b"
    try testing.expect(isMatch("aab", "c*a*b"));

    // "mississippi" vs "mis*is*p*."
    try testing.expect(!isMatch("mississippi", "mis*is*p*."));

    // "mississippi" vs "mis*is*ip*."
    try testing.expect(isMatch("mississippi", "mis*is*ip*."));

    // Mixed wildcards and literals
    try testing.expect(isMatch("aaa", "a*a"));
    try testing.expect(isMatch("aaa", "ab*a*c*a"));
}

test "regex matching - edge cases" {
    // Pattern longer than string (no match)
    try testing.expect(!isMatch("a", "ab"));
    try testing.expect(!isMatch("ab", "abc"));

    // Pattern can match due to '*'
    try testing.expect(isMatch("a", "ab*"));
    try testing.expect(isMatch("a", "a.*"));

    // String longer than pattern (no match unless pattern has wildcards)
    try testing.expect(!isMatch("ab", "a"));
    // "a*" means zero or more 'a', so "ab" doesn't match (has 'b')
    try testing.expect(!isMatch("ab", "a*"));
    // But "ab" matches "a*b" (zero or more 'a' followed by 'b')
    try testing.expect(isMatch("ab", "a*b"));
    // And "ab" matches ".*" (any sequence)
    try testing.expect(isMatch("ab", ".*"));

    // All wildcards
    try testing.expect(isMatch("abcdefg", ".*"));
}

test "regex matching - star at beginning" {
    // Invalid patterns (star must have preceding element)
    // In our implementation, we handle this gracefully
    // Real regex engines would reject these patterns

    // Valid patterns with star at position 1
    try testing.expect(isMatch("", "a*"));
    try testing.expect(isMatch("aa", "a*"));
}

test "regex matching - backtracking scenarios" {
    // Requires backtracking to find correct match
    try testing.expect(isMatch("aaa", "a*a"));
    try testing.expect(isMatch("aaa", "ab*a*c*a"));
    try testing.expect(!isMatch("aaa", "aaaa"));

    // Complex backtracking
    try testing.expect(isMatch("aaaab", "a*ab"));
    try testing.expect(isMatch("bbbba", ".*a*a"));
}

test "regex matching - optimized variant" {
    // Test space-optimized version with same cases
    try testing.expect(isMatchOptimized("aa", "a*"));
    try testing.expect(isMatchOptimized("ab", ".*"));
    try testing.expect(isMatchOptimized("aab", "c*a*b"));
    try testing.expect(!isMatchOptimized("aa", "a"));
    try testing.expect(isMatchOptimized("", "a*b*c*"));

    // Large string
    const s = "a" ** 100;
    const p = "a*";
    try testing.expect(isMatchOptimized(s, p));
}

test "regex matching - recursive variant" {
    const allocator = testing.allocator;

    // Test recursive version with same cases
    try testing.expect(try isMatchRecursive(allocator, "aa", "a*"));
    try testing.expect(try isMatchRecursive(allocator, "ab", ".*"));
    try testing.expect(try isMatchRecursive(allocator, "aab", "c*a*b"));
    try testing.expect(!try isMatchRecursive(allocator, "aa", "a"));
    try testing.expect(try isMatchRecursive(allocator, "", "a*b*c*"));

    // Complex pattern
    try testing.expect(try isMatchRecursive(allocator, "mississippi", "mis*is*ip*."));
}

test "regex matching - consistency across variants" {
    const allocator = testing.allocator;

    const test_cases = [_]struct { s: []const u8, p: []const u8 }{
        .{ .s = "aa", .p = "a" },
        .{ .s = "aa", .p = "a*" },
        .{ .s = "ab", .p = ".*" },
        .{ .s = "aab", .p = "c*a*b" },
        .{ .s = "mississippi", .p = "mis*is*p*." },
        .{ .s = "", .p = "" },
        .{ .s = "", .p = "a*" },
        .{ .s = "a", .p = "ab*" },
    };

    for (test_cases) |tc| {
        const result1 = isMatch(tc.s, tc.p);
        const result2 = isMatchOptimized(tc.s, tc.p);
        const result3 = try isMatchRecursive(allocator, tc.s, tc.p);

        try testing.expectEqual(result1, result2);
        try testing.expectEqual(result1, result3);
    }
}

test "regex matching - memory safety" {
    const allocator = testing.allocator;

    // Recursive version uses allocator
    const result = try isMatchRecursive(allocator, "test", "t.*t");
    try testing.expect(result);
}
