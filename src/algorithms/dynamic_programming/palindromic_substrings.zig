const std = @import("std");
const Allocator = std.mem.Allocator;

/// Count all palindromic substrings in a string
/// Uses center expansion approach for O(n²) time and O(1) space
///
/// Time: O(n²) where n = string length
/// Space: O(1) - no additional space beyond input
///
/// Use cases:
/// - Text analysis (palindrome density)
/// - Pattern recognition in sequences
/// - DNA sequence analysis (palindromic motifs)
/// - String validation and preprocessing
///
/// Example:
/// ```zig
/// const count = countPalindromicSubstrings("aaa");
/// // Returns 6: "a", "a", "a", "aa", "aa", "aaa"
/// ```
pub fn countPalindromicSubstrings(s: []const u8) u64 {
    if (s.len == 0) return 0;

    var count: u64 = 0;

    // Expand around each possible center
    for (0..s.len) |i| {
        // Odd length palindromes (single character center)
        count += expandAroundCenter(s, i, i);

        // Even length palindromes (two character center)
        if (i + 1 < s.len) {
            count += expandAroundCenter(s, i, i + 1);
        }
    }

    return count;
}

/// Count palindromic substrings using dynamic programming
/// Uses 2D DP table for explicit state tracking
///
/// Time: O(n²) where n = string length
/// Space: O(n²) for DP table
///
/// DP state: dp[i][j] = true if s[i..j+1] is palindrome
/// Recurrence:
/// - dp[i][i] = true (single char)
/// - dp[i][i+1] = (s[i] == s[i+1]) (two chars)
/// - dp[i][j] = (s[i] == s[j]) and dp[i+1][j-1] (longer)
///
/// Example:
/// ```zig
/// const count = try countPalindromicSubstringsDP(allocator, "aba");
/// // Returns 4: "a", "b", "a", "aba"
/// ```
pub fn countPalindromicSubstringsDP(allocator: Allocator, s: []const u8) !u64 {
    if (s.len == 0) return 0;

    const n = s.len;

    // Allocate DP table
    var dp = try allocator.alloc([]bool, n);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }

    for (0..n) |i| {
        dp[i] = try allocator.alloc(bool, n);
        @memset(dp[i], false);
    }

    var count: u64 = 0;

    // Fill DP table by substring length
    var len: usize = 1;
    while (len <= n) : (len += 1) {
        var i: usize = 0;
        while (i + len <= n) : (i += 1) {
            const j = i + len - 1;

            if (len == 1) {
                // Single character is always palindrome
                dp[i][j] = true;
                count += 1;
            } else if (len == 2) {
                // Two characters: check if equal
                if (s[i] == s[j]) {
                    dp[i][j] = true;
                    count += 1;
                }
            } else {
                // Longer substring: check ends match and inside is palindrome
                if (s[i] == s[j] and dp[i + 1][j - 1]) {
                    dp[i][j] = true;
                    count += 1;
                }
            }
        }
    }

    return count;
}

/// Find all palindromic substrings in a string
/// Returns ArrayList of substring slices (borrowed references)
///
/// Time: O(n²) where n = string length
/// Space: O(k) where k = number of palindromes (output size)
///
/// Example:
/// ```zig
/// var result = try findAllPalindromicSubstrings(allocator, "aba");
/// defer result.deinit(allocator);
/// // Result: ["a", "b", "a", "aba"]
/// ```
pub fn findAllPalindromicSubstrings(allocator: Allocator, s: []const u8) !std.ArrayList([]const u8) {
    var result = try std.ArrayList([]const u8).initCapacity(allocator, 0);
    errdefer result.deinit(allocator);

    if (s.len == 0) return result;

    const n = s.len;

    // Allocate DP table
    var dp = try allocator.alloc([]bool, n);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }

    for (0..n) |i| {
        dp[i] = try allocator.alloc(bool, n);
        @memset(dp[i], false);
    }

    // Fill DP table and collect palindromes
    var len: usize = 1;
    while (len <= n) : (len += 1) {
        var i: usize = 0;
        while (i + len <= n) : (i += 1) {
            const j = i + len - 1;

            var is_palindrome = false;
            if (len == 1) {
                is_palindrome = true;
            } else if (len == 2) {
                is_palindrome = (s[i] == s[j]);
            } else {
                is_palindrome = (s[i] == s[j] and dp[i + 1][j - 1]);
            }

            if (is_palindrome) {
                dp[i][j] = true;
                try result.append(allocator, s[i..j+1]);
            }
        }
    }

    return result;
}

/// Count longest palindromic substrings
/// Returns count of substrings that have maximum length
///
/// Time: O(n²) where n = string length
/// Space: O(1)
///
/// Example:
/// ```zig
/// const count = countLongestPalindromicSubstrings("abba");
/// // Returns 1: only "abba" has max length 4
/// ```
pub fn countLongestPalindromicSubstrings(s: []const u8) u64 {
    if (s.len == 0) return 0;

    var max_len: usize = 0;
    var count: u64 = 0;

    // Check all possible centers
    for (0..s.len) |i| {
        // Odd length
        const odd_info = getPalindromeInfo(s, i, i);
        if (odd_info.length > max_len) {
            max_len = odd_info.length;
            count = 1;
        } else if (odd_info.length == max_len) {
            count += 1;
        }

        // Even length
        if (i + 1 < s.len) {
            const even_info = getPalindromeInfo(s, i, i + 1);
            if (even_info.length > max_len) {
                max_len = even_info.length;
                count = 1;
            } else if (even_info.length == max_len) {
                count += 1;
            }
        }
    }

    return count;
}

/// Get length of distinct palindromic substrings (no duplicates)
/// Uses hash set to track unique palindromes
///
/// Time: O(n²) where n = string length
/// Space: O(k) where k = number of unique palindromes
///
/// Example:
/// ```zig
/// const count = try countDistinctPalindromicSubstrings(allocator, "aaa");
/// // Returns 3: "a", "aa", "aaa" (not 6)
/// ```
pub fn countDistinctPalindromicSubstrings(allocator: Allocator, s: []const u8) !u64 {
    if (s.len == 0) return 0;

    var seen = std.StringHashMap(void).init(allocator);
    defer seen.deinit();

    const n = s.len;

    // Allocate DP table
    var dp = try allocator.alloc([]bool, n);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }

    for (0..n) |i| {
        dp[i] = try allocator.alloc(bool, n);
        @memset(dp[i], false);
    }

    // Fill DP table and track unique palindromes
    var len: usize = 1;
    while (len <= n) : (len += 1) {
        var i: usize = 0;
        while (i + len <= n) : (i += 1) {
            const j = i + len - 1;

            var is_palindrome = false;
            if (len == 1) {
                is_palindrome = true;
            } else if (len == 2) {
                is_palindrome = (s[i] == s[j]);
            } else {
                is_palindrome = (s[i] == s[j] and dp[i + 1][j - 1]);
            }

            if (is_palindrome) {
                dp[i][j] = true;
                const substring = s[i..j+1];
                try seen.put(substring, {});
            }
        }
    }

    return @as(u64, @intCast(seen.count()));
}

/// Information about a palindrome centered at given indices
const PalindromeInfo = struct {
    start: usize,
    end: usize,
    length: usize,
};

/// Get information about palindrome centered at given indices
fn getPalindromeInfo(s: []const u8, left_init: usize, right_init: usize) PalindromeInfo {
    // Check if initial center is valid
    if (left_init >= s.len or right_init >= s.len or s[left_init] != s[right_init]) {
        return PalindromeInfo{
            .start = left_init,
            .end = right_init,
            .length = 0,
        };
    }

    var left = left_init;
    var right = right_init;

    // Expand while valid palindrome
    while (true) {
        if (left == 0 or right >= s.len - 1) break;

        const next_left = left - 1;
        const next_right = right + 1;

        if (next_left >= s.len or next_right >= s.len or s[next_left] != s[next_right]) {
            break;
        }

        left = next_left;
        right = next_right;
    }

    return PalindromeInfo{
        .start = left,
        .end = right,
        .length = right - left + 1,
    };
}

/// Helper: expand around center and count palindromes
fn expandAroundCenter(s: []const u8, left_init: usize, right_init: usize) u64 {
    var left = left_init;
    var right = right_init;
    var count: u64 = 0;

    // Expand while characters match
    while (left < s.len and right < s.len and s[left] == s[right]) {
        count += 1;

        if (left == 0) break;
        left -= 1;
        right += 1;
    }

    return count;
}

// ============================================================================
// Tests
// ============================================================================

test "palindromic_substrings: basic examples" {
    const allocator = std.testing.allocator;

    // "abc" - 3 single chars
    try std.testing.expectEqual(@as(u64, 3), countPalindromicSubstrings("abc"));
    try std.testing.expectEqual(@as(u64, 3), try countPalindromicSubstringsDP(allocator, "abc"));

    // "aaa" - "a", "a", "a", "aa", "aa", "aaa" = 6
    try std.testing.expectEqual(@as(u64, 6), countPalindromicSubstrings("aaa"));
    try std.testing.expectEqual(@as(u64, 6), try countPalindromicSubstringsDP(allocator, "aaa"));

    // "aba" - "a", "b", "a", "aba" = 4
    try std.testing.expectEqual(@as(u64, 4), countPalindromicSubstrings("aba"));
    try std.testing.expectEqual(@as(u64, 4), try countPalindromicSubstringsDP(allocator, "aba"));
}

test "palindromic_substrings: empty and single" {
    const allocator = std.testing.allocator;

    try std.testing.expectEqual(@as(u64, 0), countPalindromicSubstrings(""));
    try std.testing.expectEqual(@as(u64, 0), try countPalindromicSubstringsDP(allocator, ""));

    try std.testing.expectEqual(@as(u64, 1), countPalindromicSubstrings("a"));
    try std.testing.expectEqual(@as(u64, 1), try countPalindromicSubstringsDP(allocator, "a"));
}

test "palindromic_substrings: two characters" {
    const allocator = std.testing.allocator;

    // "aa" - "a", "a", "aa" = 3
    try std.testing.expectEqual(@as(u64, 3), countPalindromicSubstrings("aa"));
    try std.testing.expectEqual(@as(u64, 3), try countPalindromicSubstringsDP(allocator, "aa"));

    // "ab" - "a", "b" = 2
    try std.testing.expectEqual(@as(u64, 2), countPalindromicSubstrings("ab"));
    try std.testing.expectEqual(@as(u64, 2), try countPalindromicSubstringsDP(allocator, "ab"));
}

test "palindromic_substrings: longer examples" {
    const allocator = std.testing.allocator;

    // "abba" - "a", "b", "b", "a", "bb", "abba" = 6
    try std.testing.expectEqual(@as(u64, 6), countPalindromicSubstrings("abba"));
    try std.testing.expectEqual(@as(u64, 6), try countPalindromicSubstringsDP(allocator, "abba"));

    // "racecar" - many palindromes
    const count = countPalindromicSubstrings("racecar");
    try std.testing.expectEqual(count, try countPalindromicSubstringsDP(allocator, "racecar"));
    try std.testing.expect(count > 7); // At least single chars
}

test "palindromic_substrings: find all palindromes" {
    const allocator = std.testing.allocator;

    var result = try findAllPalindromicSubstrings(allocator, "aba");
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 4), result.items.len);

    // Check contents (order may vary by implementation)
    var has_a = false;
    var has_b = false;
    var has_aba = false;

    for (result.items) |item| {
        if (std.mem.eql(u8, item, "a")) has_a = true;
        if (std.mem.eql(u8, item, "b")) has_b = true;
        if (std.mem.eql(u8, item, "aba")) has_aba = true;
    }

    try std.testing.expect(has_a);
    try std.testing.expect(has_b);
    try std.testing.expect(has_aba);
}

test "palindromic_substrings: count longest" {
    // "abba" - only "abba" has length 4
    try std.testing.expectEqual(@as(u64, 1), countLongestPalindromicSubstrings("abba"));

    // "aaa" - "aaa" has length 3
    try std.testing.expectEqual(@as(u64, 1), countLongestPalindromicSubstrings("aaa"));

    // "aba" - only "aba" has length 3
    try std.testing.expectEqual(@as(u64, 1), countLongestPalindromicSubstrings("aba"));

    // "abc" - all have length 1
    try std.testing.expectEqual(@as(u64, 3), countLongestPalindromicSubstrings("abc"));
}

test "palindromic_substrings: distinct palindromes" {
    const allocator = std.testing.allocator;

    // "aaa" - distinct: "a", "aa", "aaa" = 3 (not 6)
    try std.testing.expectEqual(@as(u64, 3), try countDistinctPalindromicSubstrings(allocator, "aaa"));

    // "aba" - distinct: "a", "b", "aba" = 3
    try std.testing.expectEqual(@as(u64, 3), try countDistinctPalindromicSubstrings(allocator, "aba"));

    // "abba" - distinct: "a", "b", "bb", "abba" = 4
    try std.testing.expectEqual(@as(u64, 4), try countDistinctPalindromicSubstrings(allocator, "abba"));
}

test "palindromic_substrings: all equal characters" {
    const allocator = std.testing.allocator;

    // "aaaa" - n*(n+1)/2 = 4*5/2 = 10
    try std.testing.expectEqual(@as(u64, 10), countPalindromicSubstrings("aaaa"));
    try std.testing.expectEqual(@as(u64, 10), try countPalindromicSubstringsDP(allocator, "aaaa"));

    // Distinct: "a", "aa", "aaa", "aaaa" = 4
    try std.testing.expectEqual(@as(u64, 4), try countDistinctPalindromicSubstrings(allocator, "aaaa"));
}

test "palindromic_substrings: mixed characters" {
    const allocator = std.testing.allocator;

    const s = "abcba";
    const count1 = countPalindromicSubstrings(s);
    const count2 = try countPalindromicSubstringsDP(allocator, s);

    try std.testing.expectEqual(count1, count2);
    try std.testing.expect(count1 >= 5); // At least single chars
}

test "palindromic_substrings: large string" {
    const allocator = std.testing.allocator;

    var s: [100]u8 = undefined;
    for (0..100) |i| {
        s[i] = @as(u8, @intCast((i % 26) + 'a'));
    }

    const count1 = countPalindromicSubstrings(&s);
    const count2 = try countPalindromicSubstringsDP(allocator, &s);

    try std.testing.expectEqual(count1, count2);
    try std.testing.expect(count1 >= 100); // At least single chars
}

test "palindromic_substrings: consistency between methods" {
    const allocator = std.testing.allocator;

    const test_cases = [_][]const u8{
        "a",
        "aa",
        "aba",
        "abba",
        "aaa",
        "abc",
        "racecar",
        "abcdefg",
        "aabbaa",
    };

    for (test_cases) |s| {
        const count1 = countPalindromicSubstrings(s);
        const count2 = try countPalindromicSubstringsDP(allocator, s);
        try std.testing.expectEqual(count1, count2);
    }
}

test "palindromic_substrings: find all with duplicates" {
    const allocator = std.testing.allocator;

    var result = try findAllPalindromicSubstrings(allocator, "aaa");
    defer result.deinit(allocator);

    // Should have 6 total: "a", "a", "a", "aa", "aa", "aaa"
    try std.testing.expectEqual(@as(usize, 6), result.items.len);
}

test "palindromic_substrings: memory safety" {
    const allocator = std.testing.allocator;

    for (0..10) |_| {
        _ = try countPalindromicSubstringsDP(allocator, "abcba");

        var result = try findAllPalindromicSubstrings(allocator, "aba");
        result.deinit(allocator);

        _ = try countDistinctPalindromicSubstrings(allocator, "aaa");
    }
}
