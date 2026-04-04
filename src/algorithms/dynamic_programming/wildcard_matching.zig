const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Wildcard Matching — determines if a string matches a pattern with wildcards
///
/// Algorithm: Dynamic programming with bottom-up approach
///
/// Pattern syntax:
/// - '?' matches any single character
/// - '*' matches zero or more characters (greedy)
/// - All other characters match literally
///
/// DP recurrence:
/// - dp[i][j] = true if s[0..i-1] matches p[0..j-1]
/// - If p[j-1] == '?': dp[i][j] = dp[i-1][j-1]
/// - If p[j-1] == '*': dp[i][j] = dp[i-1][j] (match one or more) OR dp[i][j-1] (match zero)
/// - Else: dp[i][j] = (s[i-1] == p[j-1]) AND dp[i-1][j-1]
///
/// Base cases:
/// - dp[0][0] = true (empty string matches empty pattern)
/// - dp[0][j] = true if p[0..j-1] is all '*' characters
/// - dp[i][0] = false for i > 0 (non-empty string doesn't match empty pattern)
///
/// Time: O(n × m) where n = len(s), m = len(p)
/// Space: O(n × m) for full table, O(m) for space-optimized version

/// Checks if string matches pattern with wildcards using full DP table
///
/// Time: O(n × m)
/// Space: O(n × m)
///
/// Example:
/// ```zig
/// const result = try isMatch(allocator, "aa", "a"); // false
/// const result2 = try isMatch(allocator, "aa", "*"); // true
/// const result3 = try isMatch(allocator, "cb", "?a"); // false
/// const result4 = try isMatch(allocator, "adceb", "*a*b"); // true
/// ```
pub fn isMatch(allocator: Allocator, s: []const u8, p: []const u8) !bool {
    const n = s.len;
    const m = p.len;

    // Allocate DP table: dp[i][j] = s[0..i] matches p[0..j]
    const dp = try allocator.alloc([]bool, n + 1);
    defer allocator.free(dp);
    for (dp) |*row| {
        row.* = try allocator.alloc(bool, m + 1);
    }
    defer for (dp) |row| {
        allocator.free(row);
    };

    // Initialize to false
    for (dp) |row| {
        @memset(row, false);
    }

    // Base case: empty string matches empty pattern
    dp[0][0] = true;

    // Base case: empty string matches pattern with all '*'
    for (1..m + 1) |j| {
        if (p[j - 1] == '*') {
            dp[0][j] = dp[0][j - 1];
        }
    }

    // Fill DP table
    for (1..n + 1) |i| {
        for (1..m + 1) |j| {
            if (p[j - 1] == '*') {
                // '*' matches zero or more characters
                dp[i][j] = dp[i - 1][j] or dp[i][j - 1];
            } else if (p[j - 1] == '?' or p[j - 1] == s[i - 1]) {
                // '?' matches single char, or literal match
                dp[i][j] = dp[i - 1][j - 1];
            }
        }
    }

    return dp[n][m];
}

/// Space-optimized wildcard matching using rolling buffer
///
/// Time: O(n × m)
/// Space: O(m)
///
/// Example:
/// ```zig
/// const result = try isMatchOptimized(allocator, "aa", "*"); // true
/// ```
pub fn isMatchOptimized(allocator: Allocator, s: []const u8, p: []const u8) !bool {
    const n = s.len;
    const m = p.len;

    // Two-row rolling buffer
    var prev = try allocator.alloc(bool, m + 1);
    defer allocator.free(prev);
    var curr = try allocator.alloc(bool, m + 1);
    defer allocator.free(curr);

    // Initialize to false
    @memset(prev, false);
    @memset(curr, false);

    // Base case: empty string matches empty pattern
    prev[0] = true;

    // Base case: empty string matches pattern with all '*'
    for (1..m + 1) |j| {
        if (p[j - 1] == '*') {
            prev[j] = prev[j - 1];
        }
    }

    // Fill DP table with rolling buffer
    for (1..n + 1) |i| {
        curr[0] = false; // Non-empty string doesn't match empty pattern

        for (1..m + 1) |j| {
            if (p[j - 1] == '*') {
                curr[j] = prev[j] or curr[j - 1];
            } else if (p[j - 1] == '?' or p[j - 1] == s[i - 1]) {
                curr[j] = prev[j - 1];
            } else {
                curr[j] = false;
            }
        }

        // Swap buffers
        std.mem.swap([]bool, &prev, &curr);
    }

    return prev[m];
}

/// Greedy optimization for specific pattern types
///
/// Time: O(n + m) best case, O(n × m) worst case
/// Space: O(1)
///
/// Uses two-pointer approach when pattern has simple structure.
/// Falls back to DP for complex patterns.
pub fn isMatchGreedy(s: []const u8, p: []const u8) bool {
    var si: usize = 0;
    var pi: usize = 0;
    var star_idx: ?usize = null;
    var match_idx: usize = 0;

    while (si < s.len) {
        // Characters match or '?'
        if (pi < p.len and (p[pi] == '?' or p[pi] == s[si])) {
            si += 1;
            pi += 1;
        }
        // '*' found — save position
        else if (pi < p.len and p[pi] == '*') {
            star_idx = pi;
            match_idx = si;
            pi += 1;
        }
        // No match, backtrack to last '*'
        else if (star_idx != null) {
            pi = star_idx.? + 1;
            match_idx += 1;
            si = match_idx;
        }
        // No match possible
        else {
            return false;
        }
    }

    // Check remaining pattern characters (must all be '*')
    while (pi < p.len) {
        if (p[pi] != '*') return false;
        pi += 1;
    }

    return true;
}

// ============================================================================
// Tests
// ============================================================================

test "wildcard matching - basic examples" {
    const allocator = testing.allocator;

    // Basic non-match
    try testing.expect(try isMatch(allocator, "aa", "a") == false);

    // '*' matches everything
    try testing.expect(try isMatch(allocator, "aa", "*") == true);

    // '?' doesn't match 'a'
    try testing.expect(try isMatch(allocator, "cb", "?a") == false);

    // Complex pattern
    try testing.expect(try isMatch(allocator, "adceb", "*a*b") == true);

    // No match
    try testing.expect(try isMatch(allocator, "acdcb", "a*c?b") == false);
}

test "wildcard matching - empty strings" {
    const allocator = testing.allocator;

    // Empty matches empty
    try testing.expect(try isMatch(allocator, "", "") == true);

    // Empty matches all '*'
    try testing.expect(try isMatch(allocator, "", "*") == true);
    try testing.expect(try isMatch(allocator, "", "***") == true);

    // Empty doesn't match non-empty pattern
    try testing.expect(try isMatch(allocator, "", "a") == false);
    try testing.expect(try isMatch(allocator, "", "?") == false);

    // Non-empty doesn't match empty pattern
    try testing.expect(try isMatch(allocator, "a", "") == false);
}

test "wildcard matching - single character" {
    const allocator = testing.allocator;

    // Exact match
    try testing.expect(try isMatch(allocator, "a", "a") == true);

    // No match
    try testing.expect(try isMatch(allocator, "a", "b") == false);

    // '?' matches
    try testing.expect(try isMatch(allocator, "a", "?") == true);

    // '*' matches
    try testing.expect(try isMatch(allocator, "a", "*") == true);
}

test "wildcard matching - question mark" {
    const allocator = testing.allocator;

    // Multiple '?'
    try testing.expect(try isMatch(allocator, "abc", "???") == true);

    // Mixed with literals
    try testing.expect(try isMatch(allocator, "abc", "a?c") == true);
    try testing.expect(try isMatch(allocator, "abc", "a?d") == false);

    // Too few '?'
    try testing.expect(try isMatch(allocator, "abc", "??") == false);

    // Too many '?'
    try testing.expect(try isMatch(allocator, "abc", "????") == false);
}

test "wildcard matching - asterisk" {
    const allocator = testing.allocator;

    // '*' matches zero characters
    try testing.expect(try isMatch(allocator, "abc", "abc*") == true);

    // '*' matches multiple characters
    try testing.expect(try isMatch(allocator, "abcdef", "abc*") == true);

    // Multiple '*'
    try testing.expect(try isMatch(allocator, "abcdef", "*abc*def*") == true);

    // '*' at start
    try testing.expect(try isMatch(allocator, "abcdef", "*def") == true);

    // '*' in middle
    try testing.expect(try isMatch(allocator, "abcdef", "ab*ef") == true);
}

test "wildcard matching - complex patterns" {
    const allocator = testing.allocator;

    // Mixed wildcards
    try testing.expect(try isMatch(allocator, "abcdef", "a?c*f") == true);
    try testing.expect(try isMatch(allocator, "abcdef", "a*c?e*") == true);

    // Multiple '*' with literals
    try testing.expect(try isMatch(allocator, "mississippi", "m*si*p*") == true);
    try testing.expect(try isMatch(allocator, "mississippi", "m*si*p?") == false);

    // Adjacent '*'
    try testing.expect(try isMatch(allocator, "abc", "**abc") == true);
    try testing.expect(try isMatch(allocator, "abc", "a**c") == true);
}

test "wildcard matching - optimized version" {
    const allocator = testing.allocator;

    // Same tests with optimized version
    try testing.expect(try isMatchOptimized(allocator, "aa", "a") == false);
    try testing.expect(try isMatchOptimized(allocator, "aa", "*") == true);
    try testing.expect(try isMatchOptimized(allocator, "cb", "?a") == false);
    try testing.expect(try isMatchOptimized(allocator, "adceb", "*a*b") == true);
    try testing.expect(try isMatchOptimized(allocator, "acdcb", "a*c?b") == false);

    // Consistency check
    const s = "mississippi";
    const p = "m*si*p*";
    const result1 = try isMatch(allocator, s, p);
    const result2 = try isMatchOptimized(allocator, s, p);
    try testing.expect(result1 == result2);
}

test "wildcard matching - greedy version" {
    // Same tests with greedy version (no allocator needed)
    try testing.expect(isMatchGreedy("aa", "a") == false);
    try testing.expect(isMatchGreedy("aa", "*") == true);
    try testing.expect(isMatchGreedy("cb", "?a") == false);
    try testing.expect(isMatchGreedy("adceb", "*a*b") == true);
    try testing.expect(isMatchGreedy("acdcb", "a*c?b") == false);

    // Additional greedy tests
    try testing.expect(isMatchGreedy("", "") == true);
    try testing.expect(isMatchGreedy("", "*") == true);
    try testing.expect(isMatchGreedy("abc", "???") == true);
    try testing.expect(isMatchGreedy("mississippi", "m*si*p*") == true);
}

test "wildcard matching - large strings" {
    const allocator = testing.allocator;

    // Long string with simple pattern
    const s1 = "a" ** 100 ++ "b";
    const p1 = "*b";
    try testing.expect(try isMatch(allocator, s1, p1) == true);

    // Long pattern
    const s2 = "abcdefghijklmnopqrstuvwxyz";
    const p2 = "a*b*c*d*e*f*g*h*i*j*k*l*m*n*o*p*q*r*s*t*u*v*w*x*y*z";
    try testing.expect(try isMatch(allocator, s2, p2) == true);
}

test "wildcard matching - edge cases" {
    const allocator = testing.allocator;

    // Pattern longer than string
    try testing.expect(try isMatch(allocator, "a", "ab") == false);

    // String longer than pattern (without '*')
    try testing.expect(try isMatch(allocator, "ab", "a") == false);

    // All wildcards
    try testing.expect(try isMatch(allocator, "abc", "***") == true);

    // Pattern with trailing '*'
    try testing.expect(try isMatch(allocator, "abc", "abc*****") == true);
}

test "wildcard matching - backtracking scenarios" {
    const allocator = testing.allocator;

    // Requires backtracking
    try testing.expect(try isMatch(allocator, "aaaa", "*a") == true);
    try testing.expect(try isMatch(allocator, "aaaa", "*aa") == true);

    // Complex backtracking
    try testing.expect(try isMatch(allocator, "abefcdgiescdfimde", "ab*cd?i*de") == true);
}

test "wildcard matching - memory safety" {
    const allocator = testing.allocator;

    // Multiple allocations/deallocations
    for (0..10) |_| {
        _ = try isMatch(allocator, "test", "t*t");
        _ = try isMatchOptimized(allocator, "test", "t*t");
    }
}
