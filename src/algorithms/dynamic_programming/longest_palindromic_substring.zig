const std = @import("std");
const Allocator = std.mem.Allocator;

/// Longest Palindromic Substring (LPS)
///
/// Finds the longest contiguous substring that reads the same forwards and backwards.
/// This is DIFFERENT from Longest Palindromic Subsequence (allows non-contiguous chars).
///
/// Algorithm:
/// - Expand Around Center: O(n²) time, O(1) space
/// - Dynamic Programming: O(n²) time, O(n²) space
/// - Manacher's Algorithm: O(n) time, O(n) space (advanced)
///
/// Use cases:
/// - Text analysis (DNA sequences, pattern recognition)
/// - String processing (compression, encoding)
/// - Interview questions (LeetCode #5)
/// - Linguistic analysis (palindrome detection)

/// Find longest palindromic substring using expand-around-center approach.
///
/// Time: O(n²) — for each center, expand outwards
/// Space: O(1) — only stores indices
///
/// Returns: substring slice pointing into original string
///
/// Algorithm:
/// 1. For each possible center (n + n-1 centers for odd/even palindromes)
/// 2. Expand while chars match
/// 3. Track longest palindrome found
///
/// Example:
/// ```zig
/// const s = "babad";
/// const result = longestPalindrome(s);
/// // result = "bab" or "aba" (both valid)
/// ```
pub fn longestPalindrome(s: []const u8) []const u8 {
    if (s.len == 0) return s;

    var start: usize = 0;
    var max_len: usize = 1;

    for (0..s.len) |i| {
        // Odd length palindromes (center = single char)
        const len1 = expandAroundCenter(s, i, i);
        // Even length palindromes (center = between two chars)
        const len2 = expandAroundCenter(s, i, i + 1);

        const len = @max(len1, len2);
        if (len > max_len) {
            max_len = len;
            start = i - (len - 1) / 2;
        }
    }

    return s[start..start + max_len];
}

fn expandAroundCenter(s: []const u8, left_init: usize, right_init: usize) usize {
    if (right_init >= s.len) return 0;

    var left: isize = @intCast(left_init);
    var right: isize = @intCast(right_init);
    const n: isize = @intCast(s.len);

    while (left >= 0 and right < n and s[@intCast(left)] == s[@intCast(right)]) {
        left -= 1;
        right += 1;
    }

    // Now left and right are one position too far
    left += 1;
    right -= 1;

    return @intCast(right - left + 1);
}

/// Find all palindromic substrings using DP table.
///
/// Time: O(n²) — fill DP table
/// Space: O(n²) — DP table
///
/// Returns: ArrayList of all palindromic substrings (caller owns, must deinit)
///
/// DP table: dp[i][j] = true if s[i..j+1] is palindrome
/// Recurrence:
/// - dp[i][i] = true (single char)
/// - dp[i][i+1] = (s[i] == s[i+1]) (two chars)
/// - dp[i][j] = (s[i] == s[j]) and dp[i+1][j-1] (general)
///
/// Example:
/// ```zig
/// var result = try allPalindromes(allocator, "aab");
/// defer result.deinit(allocator);
/// // result contains: "a", "a", "aa", "b"
/// ```
pub fn allPalindromes(allocator: Allocator, s: []const u8) !std.ArrayList([]const u8) {
    var result = try std.ArrayList([]const u8).initCapacity(allocator, s.len);
    errdefer result.deinit(allocator);

    if (s.len == 0) return result;

    const n = s.len;

    // Allocate DP table
    const dp = try allocator.alloc([]bool, n);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }

    for (0..n) |i| {
        dp[i] = try allocator.alloc(bool, n);
        @memset(dp[i], false);
    }

    // Fill DP table
    // Length 1: single characters
    for (0..n) |i| {
        dp[i][i] = true;
        try result.append(allocator,s[i..i+1]);
    }

    // Length 2: two characters
    for (0..n-1) |i| {
        if (s[i] == s[i+1]) {
            dp[i][i+1] = true;
            try result.append(allocator, s[i..i+2]);
        }
    }

    // Length 3+: use recurrence
    var len: usize = 3;
    while (len <= n) : (len += 1) {
        var i: usize = 0;
        while (i + len <= n) : (i += 1) {
            const j = i + len - 1;
            if (s[i] == s[j] and dp[i+1][j-1]) {
                dp[i][j] = true;
                try result.append(allocator, s[i..j+1]);
            }
        }
    }

    return result;
}

/// Count total number of palindromic substrings.
///
/// Time: O(n²) — expand around each center
/// Space: O(1) — no additional storage
///
/// Example:
/// ```zig
/// const count = countPalindromes("aaa");
/// // count = 6 ("a", "a", "a", "aa", "aa", "aaa")
/// ```
pub fn countPalindromes(s: []const u8) usize {
    if (s.len == 0) return 0;

    var count: usize = 0;

    for (0..s.len) |i| {
        // Odd length palindromes
        count += countPalindromesAroundCenter(s, i, i);
        // Even length palindromes
        count += countPalindromesAroundCenter(s, i, i + 1);
    }

    return count;
}

fn countPalindromesAroundCenter(s: []const u8, left_init: usize, right_init: usize) usize {
    var count: usize = 0;
    var left = left_init;
    var right = right_init;

    while (right < s.len and s[left] == s[right]) {
        count += 1;
        if (left == 0) break;
        left -= 1;
        right += 1;
    }

    return count;
}

/// Check if entire string is a palindrome.
///
/// Time: O(n) — single pass comparison
/// Space: O(1) — no allocation
///
/// Example:
/// ```zig
/// const is_pal = isPalindrome("racecar"); // true
/// const not_pal = isPalindrome("hello"); // false
/// ```
pub fn isPalindrome(s: []const u8) bool {
    if (s.len <= 1) return true;

    var left: usize = 0;
    var right: usize = s.len - 1;

    while (left < right) {
        if (s[left] != s[right]) return false;
        left += 1;
        right -= 1;
    }

    return true;
}

/// Find longest palindromic substring using DP table (alternative approach).
///
/// Time: O(n²) — fill DP table
/// Space: O(n²) — DP table
///
/// This is similar to allPalindromes but optimized to return only the longest.
/// Uses DP approach for educational purposes (shows alternative to expand-around-center).
///
/// Example:
/// ```zig
/// const result = try longestPalindromeDP(allocator, "babad");
/// defer allocator.free(result);
/// // result = "bab" or "aba"
/// ```
pub fn longestPalindromeDP(allocator: Allocator, s: []const u8) ![]const u8 {
    if (s.len == 0) {
        const result = try allocator.alloc(u8, 0);
        return result;
    }
    if (s.len == 1) {
        const result = try allocator.alloc(u8, 1);
        result[0] = s[0];
        return result;
    }

    const n = s.len;

    // Allocate DP table
    const dp = try allocator.alloc([]bool, n);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }

    for (0..n) |i| {
        dp[i] = try allocator.alloc(bool, n);
        @memset(dp[i], false);
    }

    var start: usize = 0;
    var max_len: usize = 1;

    // Length 1: single characters
    for (0..n) |i| {
        dp[i][i] = true;
    }

    // Length 2: two characters
    for (0..n-1) |i| {
        if (s[i] == s[i+1]) {
            dp[i][i+1] = true;
            start = i;
            max_len = 2;
        }
    }

    // Length 3+: use recurrence
    var len: usize = 3;
    while (len <= n) : (len += 1) {
        var i: usize = 0;
        while (i + len <= n) : (i += 1) {
            const j = i + len - 1;
            if (s[i] == s[j] and dp[i+1][j-1]) {
                dp[i][j] = true;
                start = i;
                max_len = len;
            }
        }
    }

    const result = try allocator.alloc(u8, max_len);
    @memcpy(result, s[start..start + max_len]);
    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "longest palindrome - basic cases" {
    const testing = std.testing;

    // Single char
    try testing.expectEqualSlices(u8, "a", longestPalindrome("a"));

    // Two chars
    try testing.expectEqualSlices(u8, "aa", longestPalindrome("aa"));
    try testing.expect(longestPalindrome("ab").len == 1); // Either "a" or "b"

    // Classic examples
    const s1 = longestPalindrome("babad");
    try testing.expect(std.mem.eql(u8, s1, "bab") or std.mem.eql(u8, s1, "aba"));

    try testing.expectEqualSlices(u8, "bb", longestPalindrome("cbbd"));
}

test "longest palindrome - edge cases" {
    const testing = std.testing;

    // Empty
    try testing.expectEqualSlices(u8, "", longestPalindrome(""));

    // All same chars
    try testing.expectEqualSlices(u8, "aaaa", longestPalindrome("aaaa"));

    // No palindrome longer than 1
    const result = longestPalindrome("abcdef");
    try testing.expect(result.len == 1);
}

test "longest palindrome - even vs odd length" {
    const testing = std.testing;

    // Odd length palindrome
    try testing.expectEqualSlices(u8, "racecar", longestPalindrome("racecar"));

    // Even length palindrome
    try testing.expectEqualSlices(u8, "abba", longestPalindrome("abba"));

    // Mixed
    try testing.expectEqualSlices(u8, "anana", longestPalindrome("banana"));
}

test "all palindromes - basic" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var result = try allPalindromes(allocator, "aab");
    defer result.deinit(allocator);

    // Should find: "a", "a", "aa", "b"
    try testing.expect(result.items.len == 4);

    var found_aa = false;
    for (result.items) |p| {
        if (std.mem.eql(u8, p, "aa")) found_aa = true;
    }
    try testing.expect(found_aa);
}

test "all palindromes - empty" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var result = try allPalindromes(allocator, "");
    defer result.deinit(allocator);

    try testing.expect(result.items.len == 0);
}

test "all palindromes - single char" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var result = try allPalindromes(allocator, "x");
    defer result.deinit(allocator);

    try testing.expect(result.items.len == 1);
    try testing.expectEqualSlices(u8, "x", result.items[0]);
}

test "count palindromes - basic" {
    const testing = std.testing;

    // "aaa" has 6 palindromes: "a", "a", "a", "aa", "aa", "aaa"
    try testing.expectEqual(@as(usize, 6), countPalindromes("aaa"));

    // "abc" has 3 palindromes: "a", "b", "c"
    try testing.expectEqual(@as(usize, 3), countPalindromes("abc"));

    // "aab" has 4 palindromes: "a", "a", "aa", "b"
    try testing.expectEqual(@as(usize, 4), countPalindromes("aab"));
}

test "count palindromes - edge cases" {
    const testing = std.testing;

    // Empty
    try testing.expectEqual(@as(usize, 0), countPalindromes(""));

    // Single
    try testing.expectEqual(@as(usize, 1), countPalindromes("x"));

    // Two same
    try testing.expectEqual(@as(usize, 3), countPalindromes("aa")); // "a", "a", "aa"

    // Two different
    try testing.expectEqual(@as(usize, 2), countPalindromes("ab")); // "a", "b"
}

test "is palindrome - basic" {
    const testing = std.testing;

    // True cases
    try testing.expect(isPalindrome(""));
    try testing.expect(isPalindrome("a"));
    try testing.expect(isPalindrome("aa"));
    try testing.expect(isPalindrome("aba"));
    try testing.expect(isPalindrome("abba"));
    try testing.expect(isPalindrome("racecar"));

    // False cases
    try testing.expect(!isPalindrome("ab"));
    try testing.expect(!isPalindrome("abc"));
    try testing.expect(!isPalindrome("hello"));
}

test "DP approach - basic cases" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Single char
    const r1 = try longestPalindromeDP(allocator, "a");
    defer allocator.free(r1);
    try testing.expectEqualSlices(u8, "a", r1);

    // Two chars
    const r2 = try longestPalindromeDP(allocator, "aa");
    defer allocator.free(r2);
    try testing.expectEqualSlices(u8, "aa", r2);

    // Classic example
    const r3 = try longestPalindromeDP(allocator, "babad");
    defer allocator.free(r3);
    try testing.expect(std.mem.eql(u8, r3, "bab") or std.mem.eql(u8, r3, "aba"));

    // Even palindrome
    const r4 = try longestPalindromeDP(allocator, "cbbd");
    defer allocator.free(r4);
    try testing.expectEqualSlices(u8, "bb", r4);
}

test "DP approach - complex palindromes" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Odd length
    const r1 = try longestPalindromeDP(allocator, "racecar");
    defer allocator.free(r1);
    try testing.expectEqualSlices(u8, "racecar", r1);

    // Even length
    const r2 = try longestPalindromeDP(allocator, "abba");
    defer allocator.free(r2);
    try testing.expectEqualSlices(u8, "abba", r2);

    // Multiple palindromes
    const r3 = try longestPalindromeDP(allocator, "banana");
    defer allocator.free(r3);
    try testing.expectEqualSlices(u8, "anana", r3);
}

test "consistency - expand vs DP methods" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const test_strings = [_][]const u8{
        "babad",
        "cbbd",
        "racecar",
        "banana",
        "abcdef",
    };

    for (test_strings) |s| {
        const expand = longestPalindrome(s);
        const dp_result = try longestPalindromeDP(allocator, s);
        defer allocator.free(dp_result);

        // Both should find palindromes of same length
        try testing.expectEqual(expand.len, dp_result.len);
    }
}

test "large string performance" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Build large string with embedded palindrome
    const n = 1000;
    var s = try allocator.alloc(u8, n);
    defer allocator.free(s);

    // Pattern: "abc...abc[racecar]abc...abc"
    for (0..n) |i| {
        if (i >= 450 and i < 457) {
            // Embed "racecar" in middle
            const pal = "racecar";
            s[i] = pal[i - 450];
        } else {
            s[i] = @as(u8, @intCast((i % 26) + 'a'));
        }
    }

    const result = longestPalindrome(s);
    try testing.expect(result.len >= 7); // Should find "racecar" or longer
}

test "memory safety - all allocating functions" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // allPalindromes
    {
        var result = try allPalindromes(allocator, "test");
        defer result.deinit(allocator);
        try testing.expect(result.items.len > 0);
    }

    // longestPalindromeDP
    {
        const result = try longestPalindromeDP(allocator, "palindrome");
        defer allocator.free(result);
        try testing.expect(result.len > 0);
    }
}
