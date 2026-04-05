/// Manacher's Algorithm for finding longest palindromic substrings in O(n) time.
///
/// Classic algorithm that uses symmetry properties to avoid redundant comparisons.
/// Works by expanding palindromes around centers while utilizing previously computed results.
///
/// ## Algorithm
///
/// 1. **Preprocessing**: Insert separators (#) between characters for uniform handling
///    - "abc" → "#a#b#c#" (handles even/odd length palindromes uniformly)
/// 2. **Expansion with mirroring**: For each position, expand palindrome using:
///    - Mirror position's radius (if within right boundary)
///    - Manual expansion only when necessary
/// 3. **Track rightmost boundary**: Update center and right when finding longer palindrome
///
/// ## Time Complexity
///
/// - `longestPalindromicSubstring()`: O(n) where n = string length
/// - `allPalindromes()`: O(n) for radii, O(n²) worst case for extraction
/// - `countPalindromes()`: O(n)
/// - `longestPalindromeLength()`: O(1) after preprocessing O(n)
///
/// ## Space Complexity
///
/// - O(n) for transformed string and radius array
///
/// ## References
///
/// - Manacher, Glenn (1975) - "A New Linear-Time On-Line Algorithm for Finding the Smallest Initial Palindrome of a String"
/// - LeetCode #5 (Longest Palindromic Substring)

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Result containing the longest palindromic substring and its position.
pub const PalindromeResult = struct {
    /// The palindromic substring (owned by caller, must free)
    substring: []const u8,
    /// Starting index in original string
    start: usize,
    /// Length of palindrome
    length: usize,
};

/// Find the longest palindromic substring using Manacher's algorithm.
///
/// Returns the palindrome itself along with its position in the original string.
///
/// Time: O(n) | Space: O(n)
///
/// ## Example
///
/// ```zig
/// const result = try longestPalindromicSubstring(allocator, "babad");
/// defer allocator.free(result.substring);
/// // result.substring = "bab" or "aba" (both valid, algorithm finds first)
/// // result.start = 0, result.length = 3
/// ```
pub fn longestPalindromicSubstring(allocator: Allocator, s: []const u8) !PalindromeResult {
    if (s.len == 0) return error.EmptyString;
    if (s.len == 1) {
        const substr = try allocator.dupe(u8, s);
        return PalindromeResult{
            .substring = substr,
            .start = 0,
            .length = 1,
        };
    }

    // Transform: "abc" → "#a#b#c#"
    const transformed = try transform(allocator, s);
    defer allocator.free(transformed);

    // Radius array: radius[i] = radius of palindrome centered at i
    const radius = try allocator.alloc(usize, transformed.len);
    defer allocator.free(radius);
    @memset(radius, 0);

    var center: usize = 0; // Center of rightmost palindrome
    var right: usize = 0; // Rightmost boundary

    var max_len: usize = 0;
    var max_center: usize = 0;

    for (0..transformed.len) |i| {
        // Mirror position relative to center
        const mirror = if (i < center) 0 else 2 * center - i;

        // Initialize radius using mirror if within right boundary
        if (i < right and mirror < transformed.len) {
            radius[i] = @min(radius[mirror], right - i);
        }

        // Expand palindrome at position i
        while (true) {
            const left_pos = if (i > radius[i]) i - radius[i] - 1 else null;
            const right_pos = i + radius[i] + 1;

            if (left_pos == null or right_pos >= transformed.len) break;
            if (transformed[left_pos.?] != transformed[right_pos]) break;

            radius[i] += 1;
        }

        // Update rightmost boundary
        if (i + radius[i] > right) {
            center = i;
            right = i + radius[i];
        }

        // Track maximum palindrome
        if (radius[i] > max_len) {
            max_len = radius[i];
            max_center = i;
        }
    }

    // Convert back to original string indices
    const start = (max_center - max_len) / 2;
    const substr = try allocator.dupe(u8, s[start .. start + max_len]);

    return PalindromeResult{
        .substring = substr,
        .start = start,
        .length = max_len,
    };
}

/// Get just the length of the longest palindromic substring.
///
/// More efficient than `longestPalindromicSubstring` when only length is needed.
///
/// Time: O(n) | Space: O(n)
///
/// ## Example
///
/// ```zig
/// const len = try longestPalindromeLength(allocator, "babad");
/// // len = 3 ("bab" or "aba")
/// ```
pub fn longestPalindromeLength(allocator: Allocator, s: []const u8) !usize {
    if (s.len == 0) return 0;
    if (s.len == 1) return 1;

    const transformed = try transform(allocator, s);
    defer allocator.free(transformed);

    const radius = try allocator.alloc(usize, transformed.len);
    defer allocator.free(radius);
    @memset(radius, 0);

    var center: usize = 0;
    var right: usize = 0;
    var max_len: usize = 0;

    for (0..transformed.len) |i| {
        const mirror = if (i < center) 0 else 2 * center - i;

        if (i < right and mirror < transformed.len) {
            radius[i] = @min(radius[mirror], right - i);
        }

        while (true) {
            const left_pos = if (i > radius[i]) i - radius[i] - 1 else null;
            const right_pos = i + radius[i] + 1;

            if (left_pos == null or right_pos >= transformed.len) break;
            if (transformed[left_pos.?] != transformed[right_pos]) break;

            radius[i] += 1;
        }

        if (i + radius[i] > right) {
            center = i;
            right = i + radius[i];
        }

        max_len = @max(max_len, radius[i]);
    }

    return max_len;
}

/// Count total number of palindromic substrings.
///
/// Counts all palindromes (including overlapping ones).
/// Single characters are counted as palindromes.
///
/// Time: O(n) | Space: O(n)
///
/// ## Example
///
/// ```zig
/// const count = try countPalindromes(allocator, "aaa");
/// // count = 6: "a"(3) + "aa"(2) + "aaa"(1)
/// ```
pub fn countPalindromes(allocator: Allocator, s: []const u8) !usize {
    if (s.len == 0) return 0;

    const transformed = try transform(allocator, s);
    defer allocator.free(transformed);

    const radius = try allocator.alloc(usize, transformed.len);
    defer allocator.free(radius);
    @memset(radius, 0);

    var center: usize = 0;
    var right: usize = 0;

    for (0..transformed.len) |i| {
        const mirror = if (i < center) 0 else 2 * center - i;

        if (i < right and mirror < transformed.len) {
            radius[i] = @min(radius[mirror], right - i);
        }

        while (true) {
            const left_pos = if (i > radius[i]) i - radius[i] - 1 else null;
            const right_pos = i + radius[i] + 1;

            if (left_pos == null or right_pos >= transformed.len) break;
            if (transformed[left_pos.?] != transformed[right_pos]) break;

            radius[i] += 1;
        }

        if (i + radius[i] > right) {
            center = i;
            right = i + radius[i];
        }
    }

    // Sum radii: each radius[i] represents (radius[i] + 1) / 2 palindromes
    // For odd positions (original chars): count palindromes
    // For even positions (separators): count even-length palindromes
    var total: usize = 0;
    for (radius) |r| {
        total += (r + 1) / 2;
    }

    return total;
}

/// Find all palindromic substrings.
///
/// Returns array of all palindromes (caller must free each string and the array).
/// May contain duplicates if same palindrome appears at different positions.
///
/// Time: O(n) for radii computation, O(n²) worst case for extraction
/// Space: O(n²) worst case (e.g., "aaa" has O(n²) palindromes)
///
/// ## Example
///
/// ```zig
/// const palindromes = try allPalindromes(allocator, "abc");
/// defer {
///     for (palindromes) |p| allocator.free(p);
///     allocator.free(palindromes);
/// }
/// // palindromes = ["a", "b", "c"]
/// ```
pub fn allPalindromes(allocator: Allocator, s: []const u8) ![][]const u8 {
    if (s.len == 0) return &[_][]const u8{};

    const transformed = try transform(allocator, s);
    defer allocator.free(transformed);

    const radius = try allocator.alloc(usize, transformed.len);
    defer allocator.free(radius);
    @memset(radius, 0);

    var center: usize = 0;
    var right: usize = 0;

    for (0..transformed.len) |i| {
        const mirror = if (i < center) 0 else 2 * center - i;

        if (i < right and mirror < transformed.len) {
            radius[i] = @min(radius[mirror], right - i);
        }

        while (true) {
            const left_pos = if (i > radius[i]) i - radius[i] - 1 else null;
            const right_pos = i + radius[i] + 1;

            if (left_pos == null or right_pos >= transformed.len) break;
            if (transformed[left_pos.?] != transformed[right_pos]) break;

            radius[i] += 1;
        }

        if (i + radius[i] > right) {
            center = i;
            right = i + radius[i];
        }
    }

    // Extract all palindromes
    var results = std.ArrayList([]const u8).init(allocator);
    errdefer {
        for (results.items) |p| allocator.free(p);
        results.deinit();
    }

    for (0..transformed.len) |i| {
        // Skip separators with radius 0
        if (radius[i] == 0 and transformed[i] == '#') continue;

        // Extract palindrome
        const start = (i - radius[i]) / 2;
        const len = radius[i];
        if (len > 0) {
            const palindrome = try allocator.dupe(u8, s[start .. start + len]);
            try results.append(palindrome);
        }
    }

    return try results.toOwnedSlice();
}

/// Transform string by inserting separators for uniform palindrome handling.
///
/// Internal helper. Converts "abc" → "#a#b#c#" so even and odd length
/// palindromes can be handled uniformly (all have odd length in transformed).
///
/// Time: O(n) | Space: O(n)
fn transform(allocator: Allocator, s: []const u8) ![]u8 {
    // Length: original + separators + boundary = 2n + 1
    const result = try allocator.alloc(u8, 2 * s.len + 1);

    for (0..result.len) |i| {
        if (i % 2 == 0) {
            result[i] = '#';
        } else {
            result[i] = s[i / 2];
        }
    }

    return result;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const test_allocator = testing.allocator;

test "manacher: basic palindromes" {
    {
        const result = try longestPalindromicSubstring(test_allocator, "babad");
        defer test_allocator.free(result.substring);
        try testing.expectEqual(@as(usize, 3), result.length);
        // Either "bab" or "aba" is valid
        try testing.expect(std.mem.eql(u8, result.substring, "bab") or
            std.mem.eql(u8, result.substring, "aba"));
    }

    {
        const result = try longestPalindromicSubstring(test_allocator, "cbbd");
        defer test_allocator.free(result.substring);
        try testing.expectEqualStrings("bb", result.substring);
        try testing.expectEqual(@as(usize, 1), result.start);
        try testing.expectEqual(@as(usize, 2), result.length);
    }
}

test "manacher: entire string is palindrome" {
    const result = try longestPalindromicSubstring(test_allocator, "racecar");
    defer test_allocator.free(result.substring);
    try testing.expectEqualStrings("racecar", result.substring);
    try testing.expectEqual(@as(usize, 0), result.start);
    try testing.expectEqual(@as(usize, 7), result.length);
}

test "manacher: no palindrome longer than 1" {
    const result = try longestPalindromicSubstring(test_allocator, "abcdef");
    defer test_allocator.free(result.substring);
    try testing.expectEqual(@as(usize, 1), result.length);
}

test "manacher: single character" {
    const result = try longestPalindromicSubstring(test_allocator, "a");
    defer test_allocator.free(result.substring);
    try testing.expectEqualStrings("a", result.substring);
    try testing.expectEqual(@as(usize, 0), result.start);
    try testing.expectEqual(@as(usize, 1), result.length);
}

test "manacher: empty string" {
    try testing.expectError(error.EmptyString, longestPalindromicSubstring(test_allocator, ""));
}

test "manacher: two characters same" {
    const result = try longestPalindromicSubstring(test_allocator, "aa");
    defer test_allocator.free(result.substring);
    try testing.expectEqualStrings("aa", result.substring);
    try testing.expectEqual(@as(usize, 2), result.length);
}

test "manacher: two characters different" {
    const result = try longestPalindromicSubstring(test_allocator, "ab");
    defer test_allocator.free(result.substring);
    try testing.expectEqual(@as(usize, 1), result.length);
}

test "manacher: multiple palindromes" {
    const result = try longestPalindromicSubstring(test_allocator, "abacabad");
    defer test_allocator.free(result.substring);
    try testing.expectEqual(@as(usize, 5), result.length);
    try testing.expectEqualStrings("abaca", result.substring);
}

test "manacher: even length palindrome" {
    const result = try longestPalindromicSubstring(test_allocator, "abba");
    defer test_allocator.free(result.substring);
    try testing.expectEqualStrings("abba", result.substring);
    try testing.expectEqual(@as(usize, 4), result.length);
}

test "manacher: longestPalindromeLength" {
    try testing.expectEqual(@as(usize, 3), try longestPalindromeLength(test_allocator, "babad"));
    try testing.expectEqual(@as(usize, 2), try longestPalindromeLength(test_allocator, "cbbd"));
    try testing.expectEqual(@as(usize, 7), try longestPalindromeLength(test_allocator, "racecar"));
    try testing.expectEqual(@as(usize, 1), try longestPalindromeLength(test_allocator, "abcdef"));
    try testing.expectEqual(@as(usize, 1), try longestPalindromeLength(test_allocator, "a"));
    try testing.expectEqual(@as(usize, 0), try longestPalindromeLength(test_allocator, ""));
}

test "manacher: countPalindromes" {
    // "aaa": "a"(3) + "aa"(2) + "aaa"(1) = 6
    try testing.expectEqual(@as(usize, 6), try countPalindromes(test_allocator, "aaa"));

    // "abc": "a"(1) + "b"(1) + "c"(1) = 3
    try testing.expectEqual(@as(usize, 3), try countPalindromes(test_allocator, "abc"));

    // "aba": "a"(2) + "b"(1) + "aba"(1) = 4
    try testing.expectEqual(@as(usize, 4), try countPalindromes(test_allocator, "aba"));

    try testing.expectEqual(@as(usize, 0), try countPalindromes(test_allocator, ""));
}

test "manacher: allPalindromes single char" {
    const palindromes = try allPalindromes(test_allocator, "a");
    defer {
        for (palindromes) |p| test_allocator.free(p);
        test_allocator.free(palindromes);
    }
    try testing.expectEqual(@as(usize, 1), palindromes.len);
    try testing.expectEqualStrings("a", palindromes[0]);
}

test "manacher: allPalindromes no repeats" {
    const palindromes = try allPalindromes(test_allocator, "abc");
    defer {
        for (palindromes) |p| test_allocator.free(p);
        test_allocator.free(palindromes);
    }
    try testing.expectEqual(@as(usize, 3), palindromes.len);
}

test "manacher: allPalindromes with repeats" {
    const palindromes = try allPalindromes(test_allocator, "aaa");
    defer {
        for (palindromes) |p| test_allocator.free(p);
        test_allocator.free(palindromes);
    }
    // "a"(3) + "aa"(2) + "aaa"(1) = 6
    try testing.expectEqual(@as(usize, 6), palindromes.len);
}

test "manacher: allPalindromes empty" {
    const palindromes = try allPalindromes(test_allocator, "");
    defer test_allocator.free(palindromes);
    try testing.expectEqual(@as(usize, 0), palindromes.len);
}

test "manacher: large string" {
    const s = "abcdefghijklmnopqrstuvwxyzzyxwvutsrqponmlkjihgfedcba";
    const result = try longestPalindromicSubstring(test_allocator, s);
    defer test_allocator.free(result.substring);
    try testing.expectEqual(@as(usize, 26), result.length);
}

test "manacher: repeated characters" {
    const result = try longestPalindromicSubstring(test_allocator, "aaaaaaa");
    defer test_allocator.free(result.substring);
    try testing.expectEqualStrings("aaaaaaa", result.substring);
    try testing.expectEqual(@as(usize, 7), result.length);
}

test "manacher: memory safety" {
    // Run multiple times to check for leaks
    for (0..10) |_| {
        const result = try longestPalindromicSubstring(test_allocator, "abacabad");
        test_allocator.free(result.substring);

        _ = try longestPalindromeLength(test_allocator, "racecar");
        _ = try countPalindromes(test_allocator, "aaa");

        const palindromes = try allPalindromes(test_allocator, "aba");
        for (palindromes) |p| test_allocator.free(p);
        test_allocator.free(palindromes);
    }
}
