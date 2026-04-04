// Longest Palindromic Subsequence (LPS)
//
// Given a string, find the length of the longest palindromic subsequence (not substring).
// A subsequence is a sequence derived by deleting some (or zero) characters without changing
// the order of remaining characters.
//
// Example: For "bbbab", the LPS is "bbbb" with length 4.
//          For "cbbd", the LPS is "bb" with length 2.
//
// Key differences from Longest Palindromic Substring:
// - Substring: consecutive characters (e.g., "aba" in "xabay")
// - Subsequence: non-consecutive characters allowed (e.g., "aca" in "xabacy")
//
// Time complexity: O(n²) where n = string length
// Space complexity: O(n²) for DP table (can be optimized to O(n) with rolling array)
//
// Reference: Classic DP problem, similar structure to LCS
// Applications: DNA sequence analysis, text comparison, palindrome detection

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Find the length of the longest palindromic subsequence
/// Time: O(n²), Space: O(n²)
pub fn longestPalindromicSubsequence(allocator: Allocator, s: []const u8) !usize {
    if (s.len == 0) return 0;
    if (s.len == 1) return 1;

    const n = s.len;

    // Create DP table: dp[i][j] = length of LPS in s[i..j+1]
    var dp = try allocator.alloc([]usize, n);
    defer allocator.free(dp);
    for (dp) |*row| {
        row.* = try allocator.alloc(usize, n);
    }
    defer for (dp) |row| {
        allocator.free(row);
    };

    // Initialize diagonal (single characters are palindromes of length 1)
    for (0..n) |i| {
        dp[i][i] = 1;
    }

    // Fill DP table bottom-up by increasing substring length
    var len: usize = 2;
    while (len <= n) : (len += 1) {
        var i: usize = 0;
        while (i + len - 1 < n) : (i += 1) {
            const j = i + len - 1;

            if (s[i] == s[j]) {
                // Characters match: add 2 to inner substring
                if (len == 2) {
                    dp[i][j] = 2;
                } else {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                }
            } else {
                // Characters don't match: take max of excluding either character
                dp[i][j] = @max(dp[i + 1][j], dp[i][j - 1]);
            }
        }
    }

    return dp[0][n - 1];
}

/// Find the length of LPS with space optimization (rolling array)
/// Time: O(n²), Space: O(n)
pub fn longestPalindromicSubsequenceOptimized(allocator: Allocator, s: []const u8) !usize {
    if (s.len == 0) return 0;
    if (s.len == 1) return 1;

    const n = s.len;

    // Use two rows: current and previous
    var prev = try allocator.alloc(usize, n);
    defer allocator.free(prev);
    var curr = try allocator.alloc(usize, n);
    defer allocator.free(curr);

    // Initialize diagonal
    for (0..n) |i| {
        prev[i] = 1;
    }

    // Fill DP table bottom-up
    var len: usize = 2;
    while (len <= n) : (len += 1) {
        @memset(curr, 0);
        var i: usize = 0;
        while (i + len - 1 < n) : (i += 1) {
            const j = i + len - 1;

            if (s[i] == s[j]) {
                if (len == 2) {
                    curr[i] = 2;
                } else {
                    curr[i] = prev[i + 1] + 2;
                }
            } else {
                curr[i] = @max(prev[i], curr[i]);
                if (i + 1 < n) {
                    curr[i] = @max(curr[i], prev[i + 1]);
                }
            }
        }
        // Swap rows
        const temp = prev;
        prev = curr;
        curr = temp;
    }

    return prev[0];
}

/// Find the actual longest palindromic subsequence (not just length)
/// Time: O(n²), Space: O(n²)
pub fn findLongestPalindromicSubsequence(allocator: Allocator, s: []const u8) ![]u8 {
    if (s.len == 0) return try allocator.alloc(u8, 0);
    if (s.len == 1) return try allocator.dupe(u8, s);

    const n = s.len;

    // Build DP table
    var dp = try allocator.alloc([]usize, n);
    defer allocator.free(dp);
    for (dp) |*row| {
        row.* = try allocator.alloc(usize, n);
    }
    defer for (dp) |row| {
        allocator.free(row);
    };

    for (0..n) |i| {
        dp[i][i] = 1;
    }

    var len: usize = 2;
    while (len <= n) : (len += 1) {
        var i: usize = 0;
        while (i + len - 1 < n) : (i += 1) {
            const j = i + len - 1;

            if (s[i] == s[j]) {
                if (len == 2) {
                    dp[i][j] = 2;
                } else {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                }
            } else {
                dp[i][j] = @max(dp[i + 1][j], dp[i][j - 1]);
            }
        }
    }

    // Backtrack to construct the subsequence
    const lps_len = dp[0][n - 1];
    var result = try allocator.alloc(u8, lps_len);
    errdefer allocator.free(result);

    var i: usize = 0;
    var j: usize = n - 1;
    var left: usize = 0;
    var right: usize = lps_len - 1;

    while (i <= j and left <= right) {
        if (s[i] == s[j]) {
            result[left] = s[i];
            if (left != right) {
                result[right] = s[i];
            }
            left += 1;
            if (right > 0) {
                right -= 1;
            }
            i += 1;
            if (j > 0) {
                j -= 1;
            }
        } else if (i + 1 <= j and dp[i + 1][j] > dp[i][j - 1]) {
            i += 1;
        } else if (j > 0) {
            j -= 1;
        } else {
            break;
        }
    }

    return result;
}

/// Count the number of distinct longest palindromic subsequences
/// Time: O(n²), Space: O(n²)
pub fn countLongestPalindromicSubsequences(allocator: Allocator, s: []const u8) !usize {
    if (s.len == 0) return 0;
    if (s.len == 1) return 1;

    const n = s.len;

    // DP table for length
    var dp_len = try allocator.alloc([]usize, n);
    defer allocator.free(dp_len);
    for (dp_len) |*row| {
        row.* = try allocator.alloc(usize, n);
    }
    defer for (dp_len) |row| {
        allocator.free(row);
    };

    // DP table for count
    var dp_count = try allocator.alloc([]usize, n);
    defer allocator.free(dp_count);
    for (dp_count) |*row| {
        row.* = try allocator.alloc(usize, n);
    }
    defer for (dp_count) |row| {
        allocator.free(row);
    };

    // Initialize
    for (0..n) |i| {
        dp_len[i][i] = 1;
        dp_count[i][i] = 1;
    }

    // Fill tables
    var len: usize = 2;
    while (len <= n) : (len += 1) {
        var i: usize = 0;
        while (i + len - 1 < n) : (i += 1) {
            const j = i + len - 1;

            if (s[i] == s[j]) {
                if (len == 2) {
                    dp_len[i][j] = 2;
                    dp_count[i][j] = 1;
                } else {
                    dp_len[i][j] = dp_len[i + 1][j - 1] + 2;
                    dp_count[i][j] = dp_count[i + 1][j - 1];
                }
            } else {
                const left_len = dp_len[i + 1][j];
                const right_len = dp_len[i][j - 1];

                if (left_len > right_len) {
                    dp_len[i][j] = left_len;
                    dp_count[i][j] = dp_count[i + 1][j];
                } else if (right_len > left_len) {
                    dp_len[i][j] = right_len;
                    dp_count[i][j] = dp_count[i][j - 1];
                } else {
                    dp_len[i][j] = left_len;
                    dp_count[i][j] = dp_count[i + 1][j] + dp_count[i][j - 1];
                }
            }
        }
    }

    return dp_count[0][n - 1];
}

/// Minimum number of deletions to make a string palindrome
/// (Equivalent to n - LPS length)
/// Time: O(n²), Space: O(n²)
pub fn minDeletionsToMakePalindrome(allocator: Allocator, s: []const u8) !usize {
    const lps_len = try longestPalindromicSubsequence(allocator, s);
    return s.len - lps_len;
}

/// Minimum number of insertions to make a string palindrome
/// (Same as minimum deletions for LPS approach)
/// Time: O(n²), Space: O(n²)
pub fn minInsertionsToMakePalindrome(allocator: Allocator, s: []const u8) !usize {
    return minDeletionsToMakePalindrome(allocator, s);
}

// ============================================================================
// Tests
// ============================================================================

test "LPS - empty string" {
    const len = try longestPalindromicSubsequence(testing.allocator, "");
    try testing.expectEqual(@as(usize, 0), len);
}

test "LPS - single character" {
    const len = try longestPalindromicSubsequence(testing.allocator, "a");
    try testing.expectEqual(@as(usize, 1), len);
}

test "LPS - two same characters" {
    const len = try longestPalindromicSubsequence(testing.allocator, "aa");
    try testing.expectEqual(@as(usize, 2), len);
}

test "LPS - two different characters" {
    const len = try longestPalindromicSubsequence(testing.allocator, "ab");
    try testing.expectEqual(@as(usize, 1), len);
}

test "LPS - bbbab" {
    const len = try longestPalindromicSubsequence(testing.allocator, "bbbab");
    try testing.expectEqual(@as(usize, 4), len); // "bbbb"
}

test "LPS - cbbd" {
    const len = try longestPalindromicSubsequence(testing.allocator, "cbbd");
    try testing.expectEqual(@as(usize, 2), len); // "bb"
}

test "LPS - entire string is palindrome" {
    const len = try longestPalindromicSubsequence(testing.allocator, "racecar");
    try testing.expectEqual(@as(usize, 7), len);
}

test "LPS - no palindrome except single chars" {
    const len = try longestPalindromicSubsequence(testing.allocator, "abcdef");
    try testing.expectEqual(@as(usize, 1), len);
}

test "LPS - long palindrome subsequence" {
    const len = try longestPalindromicSubsequence(testing.allocator, "agbdba");
    try testing.expectEqual(@as(usize, 5), len); // "abdba"
}

test "LPS - space optimized version" {
    const len = try longestPalindromicSubsequenceOptimized(testing.allocator, "bbbab");
    try testing.expectEqual(@as(usize, 4), len);
}

test "LPS - optimized vs standard" {
    const test_cases = [_][]const u8{
        "bbbab",
        "cbbd",
        "agbdba",
        "racecar",
        "abcdef",
        "",
        "a",
        "aa",
    };

    for (test_cases) |s| {
        const standard = try longestPalindromicSubsequence(testing.allocator, s);
        const optimized = try longestPalindromicSubsequenceOptimized(testing.allocator, s);
        try testing.expectEqual(standard, optimized);
    }
}

test "LPS - find actual subsequence" {
    const result = try findLongestPalindromicSubsequence(testing.allocator, "bbbab");
    defer testing.allocator.free(result);
    try testing.expectEqual(@as(usize, 4), result.len);
    // Should be "bbbb"
    for (result) |ch| {
        try testing.expectEqual(@as(u8, 'b'), ch);
    }
}

test "LPS - find subsequence cbbd" {
    const result = try findLongestPalindromicSubsequence(testing.allocator, "cbbd");
    defer testing.allocator.free(result);
    try testing.expectEqual(@as(usize, 2), result.len);
    try testing.expectEqual(@as(u8, 'b'), result[0]);
    try testing.expectEqual(@as(u8, 'b'), result[1]);
}

test "LPS - find subsequence for palindrome" {
    const result = try findLongestPalindromicSubsequence(testing.allocator, "racecar");
    defer testing.allocator.free(result);
    try testing.expectEqual(@as(usize, 7), result.len);
    try testing.expectEqualStrings("racecar", result);
}

test "LPS - count distinct subsequences" {
    const count = try countLongestPalindromicSubsequences(testing.allocator, "bbbab");
    try testing.expect(count >= 1); // At least one LPS exists
}

test "LPS - minimum deletions" {
    const deletions = try minDeletionsToMakePalindrome(testing.allocator, "agbdba");
    try testing.expectEqual(@as(usize, 1), deletions); // "agbdba" -> "abdba" (delete 'g')
}

test "LPS - minimum deletions for palindrome" {
    const deletions = try minDeletionsToMakePalindrome(testing.allocator, "racecar");
    try testing.expectEqual(@as(usize, 0), deletions); // Already palindrome
}

test "LPS - minimum insertions" {
    const insertions = try minInsertionsToMakePalindrome(testing.allocator, "abc");
    try testing.expectEqual(@as(usize, 2), insertions); // "abc" -> "cabac" (insert 2)
}

test "LPS - large string" {
    const s = "abcdefghijklmnopqrstuvwxyz" ++ "zyxwvutsrqponmlkjihgfedcba";
    const len = try longestPalindromicSubsequence(testing.allocator, s);
    try testing.expectEqual(@as(usize, 51), len); // Almost entire string forms palindrome
}

test "LPS - repeated characters" {
    const len = try longestPalindromicSubsequence(testing.allocator, "aaabbbccc");
    try testing.expectEqual(@as(usize, 5), len); // Any 5 same chars like "aaaaa" or "bbbbb"
}

test "LPS - memory safety" {
    // Test that all allocations are properly freed
    _ = try longestPalindromicSubsequence(testing.allocator, "test");
    _ = try longestPalindromicSubsequenceOptimized(testing.allocator, "test");
    const result = try findLongestPalindromicSubsequence(testing.allocator, "test");
    testing.allocator.free(result);
    _ = try countLongestPalindromicSubsequences(testing.allocator, "test");
    _ = try minDeletionsToMakePalindrome(testing.allocator, "test");
    _ = try minInsertionsToMakePalindrome(testing.allocator, "test");
}
