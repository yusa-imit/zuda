const std = @import("std");
const testing = std.testing;

/// Find the longest common prefix (LCP) of an array of strings.
/// Uses horizontal scanning approach - compare first string with each subsequent string.
///
/// Time: O(S) where S = sum of all characters in all strings (worst case: all identical)
/// Space: O(1) - no allocation
///
/// Returns: slice of the first string containing the common prefix (empty if no common prefix)
pub fn longestCommonPrefix(strings: []const []const u8) []const u8 {
    if (strings.len == 0) return "";
    if (strings.len == 1) return strings[0];

    var prefix = strings[0];
    for (strings[1..]) |str| {
        // Reduce prefix until it matches the start of str
        while (prefix.len > 0 and !std.mem.startsWith(u8, str, prefix)) {
            prefix = prefix[0 .. prefix.len - 1];
        }
        if (prefix.len == 0) return "";
    }
    return prefix;
}

/// Find LCP using vertical scanning (column-by-column comparison).
/// Compares characters at the same position across all strings.
///
/// Time: O(S) where S = sum of all characters (early termination on first mismatch)
/// Space: O(1) - no allocation
///
/// Returns: slice of the first string containing the common prefix
pub fn longestCommonPrefixVertical(strings: []const []const u8) []const u8 {
    if (strings.len == 0) return "";
    if (strings.len == 1) return strings[0];

    const first = strings[0];
    for (first, 0..) |char, i| {
        // Check if character at position i matches in all strings
        for (strings[1..]) |str| {
            if (i >= str.len or str[i] != char) {
                return first[0..i];
            }
        }
    }
    return first;
}

/// Find LCP using divide and conquer approach.
/// Recursively finds LCP of left and right halves, then merges.
///
/// Time: O(S) where S = sum of all characters
/// Space: O(log n) for recursion stack where n = number of strings
///
/// Returns: allocated string containing the common prefix (caller owns memory)
pub fn longestCommonPrefixDivideConquer(
    allocator: std.mem.Allocator,
    strings: []const []const u8,
) ![]u8 {
    if (strings.len == 0) return try allocator.dupe(u8, "");
    if (strings.len == 1) return try allocator.dupe(u8, strings[0]);

    return divideConquerHelper(strings);
}

fn divideConquerHelper(strings: []const []const u8) ![]u8 {
    if (strings.len == 1) return strings[0];
    if (strings.len == 2) return commonPrefixOfTwo(strings[0], strings[1]);

    const mid = strings.len / 2;
    const left_prefix = try divideConquerHelper(strings[0..mid]);
    const right_prefix = try divideConquerHelper(strings[mid..]);
    return commonPrefixOfTwo(left_prefix, right_prefix);
}

fn commonPrefixOfTwo(str1: []const u8, str2: []const u8) []const u8 {
    const min_len = @min(str1.len, str2.len);
    for (0..min_len) |i| {
        if (str1[i] != str2[i]) return str1[0..i];
    }
    return str1[0..min_len];
}

/// Find LCP using binary search on the length of the shortest string.
/// Binary searches for the longest prefix length that is common to all strings.
///
/// Time: O(S * log m) where S = sum of all characters, m = length of shortest string
/// Space: O(1) - no allocation
///
/// Returns: slice of the first string containing the common prefix
pub fn longestCommonPrefixBinarySearch(strings: []const []const u8) []const u8 {
    if (strings.len == 0) return "";
    if (strings.len == 1) return strings[0];

    // Find minimum length string
    var min_len = strings[0].len;
    for (strings[1..]) |str| {
        min_len = @min(min_len, str.len);
    }

    var left: usize = 0;
    var right: usize = min_len;

    while (left < right) {
        const mid = left + (right - left + 1) / 2; // Round up to avoid infinite loop
        if (isCommonPrefix(strings, mid)) {
            left = mid;
        } else {
            right = mid - 1;
        }
    }

    return strings[0][0..left];
}

fn isCommonPrefix(strings: []const []const u8, length: usize) bool {
    const prefix = strings[0][0..length];
    for (strings[1..]) |str| {
        if (str.len < length) return false;
        if (!std.mem.eql(u8, str[0..length], prefix)) return false;
    }
    return true;
}

/// Find all common prefixes of length >= min_length.
/// Returns an ArrayList of prefix lengths that are common to all strings.
///
/// Time: O(m * n) where m = length of shortest string, n = number of strings
/// Space: O(m) for result array
pub fn findAllCommonPrefixLengths(
    allocator: std.mem.Allocator,
    strings: []const []const u8,
    min_length: usize,
) !std.ArrayList(usize) {
    var result = std.ArrayList(usize).init(allocator);
    errdefer result.deinit();

    if (strings.len == 0) return result;

    const lcp = longestCommonPrefixVertical(strings);
    for (min_length..lcp.len + 1) |len| {
        try result.append(len);
    }
    return result;
}

/// Count the number of strings that share a prefix of at least min_length.
///
/// Time: O(n * m) where n = number of strings, m = min_length
/// Space: O(1)
pub fn countStringsWithPrefix(
    strings: []const []const u8,
    prefix: []const u8,
) usize {
    var count: usize = 0;
    for (strings) |str| {
        if (std.mem.startsWith(u8, str, prefix)) {
            count += 1;
        }
    }
    return count;
}

/// Find the longest common prefix of all suffixes starting at given positions.
/// Useful for suffix array construction and analysis.
///
/// Time: O(m) where m = length of shortest suffix
/// Space: O(1)
pub fn longestCommonPrefixOfSuffixes(
    text: []const u8,
    positions: []const usize,
) []const u8 {
    if (positions.len == 0) return "";
    if (positions.len == 1) {
        const pos = positions[0];
        if (pos >= text.len) return "";
        return text[pos..];
    }

    // Get first suffix
    const first_pos = positions[0];
    if (first_pos >= text.len) return "";
    var prefix = text[first_pos..];

    // Compare with all other suffixes
    for (positions[1..]) |pos| {
        if (pos >= text.len) return "";
        const suffix = text[pos..];

        // Find common prefix of current prefix and this suffix
        const min_len = @min(prefix.len, suffix.len);
        var common_len: usize = 0;
        for (0..min_len) |i| {
            if (prefix[i] != suffix[i]) break;
            common_len += 1;
        }
        prefix = prefix[0..common_len];
        if (prefix.len == 0) return "";
    }

    return prefix;
}

// ============================================================================
// Tests
// ============================================================================

test "LCP: empty array" {
    const strings: []const []const u8 = &.{};
    const lcp = longestCommonPrefix(strings);
    try testing.expectEqualStrings("", lcp);
}

test "LCP: single string" {
    const strings = &[_][]const u8{"hello"};
    const lcp = longestCommonPrefix(strings);
    try testing.expectEqualStrings("hello", lcp);
}

test "LCP: two identical strings" {
    const strings = &[_][]const u8{ "hello", "hello" };
    const lcp = longestCommonPrefix(strings);
    try testing.expectEqualStrings("hello", lcp);
}

test "LCP: common prefix" {
    const strings = &[_][]const u8{ "flower", "flow", "flight" };
    const lcp = longestCommonPrefix(strings);
    try testing.expectEqualStrings("fl", lcp);
}

test "LCP: no common prefix" {
    const strings = &[_][]const u8{ "dog", "racecar", "car" };
    const lcp = longestCommonPrefix(strings);
    try testing.expectEqualStrings("", lcp);
}

test "LCP: one empty string" {
    const strings = &[_][]const u8{ "hello", "", "help" };
    const lcp = longestCommonPrefix(strings);
    try testing.expectEqualStrings("", lcp);
}

test "LCP: all empty strings" {
    const strings = &[_][]const u8{ "", "", "" };
    const lcp = longestCommonPrefix(strings);
    try testing.expectEqualStrings("", lcp);
}

test "LCP: prefix is entire first string" {
    const strings = &[_][]const u8{ "abc", "abcdef", "abcxyz" };
    const lcp = longestCommonPrefix(strings);
    try testing.expectEqualStrings("abc", lcp);
}

test "LCP vertical: common prefix" {
    const strings = &[_][]const u8{ "flower", "flow", "flight" };
    const lcp = longestCommonPrefixVertical(strings);
    try testing.expectEqualStrings("fl", lcp);
}

test "LCP vertical: early termination" {
    const strings = &[_][]const u8{ "abc", "xyz", "def" };
    const lcp = longestCommonPrefixVertical(strings);
    try testing.expectEqualStrings("", lcp);
}

test "LCP vertical: different lengths" {
    const strings = &[_][]const u8{ "a", "ab", "abc" };
    const lcp = longestCommonPrefixVertical(strings);
    try testing.expectEqualStrings("a", lcp);
}

test "LCP divide and conquer: basic" {
    const strings = &[_][]const u8{ "flower", "flow", "flight" };
    const lcp = try longestCommonPrefixDivideConquer(testing.allocator, strings);
    defer testing.allocator.free(lcp);
    try testing.expectEqualStrings("fl", lcp);
}

test "LCP divide and conquer: single string" {
    const strings = &[_][]const u8{"hello"};
    const lcp = try longestCommonPrefixDivideConquer(testing.allocator, strings);
    defer testing.allocator.free(lcp);
    try testing.expectEqualStrings("hello", lcp);
}

test "LCP divide and conquer: empty" {
    const strings: []const []const u8 = &.{};
    const lcp = try longestCommonPrefixDivideConquer(testing.allocator, strings);
    defer testing.allocator.free(lcp);
    try testing.expectEqualStrings("", lcp);
}

test "LCP binary search: common prefix" {
    const strings = &[_][]const u8{ "flower", "flow", "flight" };
    const lcp = longestCommonPrefixBinarySearch(strings);
    try testing.expectEqualStrings("fl", lcp);
}

test "LCP binary search: no common prefix" {
    const strings = &[_][]const u8{ "dog", "racecar", "car" };
    const lcp = longestCommonPrefixBinarySearch(strings);
    try testing.expectEqualStrings("", lcp);
}

test "LCP binary search: all identical" {
    const strings = &[_][]const u8{ "test", "test", "test" };
    const lcp = longestCommonPrefixBinarySearch(strings);
    try testing.expectEqualStrings("test", lcp);
}

test "all common prefix lengths" {
    const strings = &[_][]const u8{ "flower", "flow", "flight" };
    var lengths = try findAllCommonPrefixLengths(testing.allocator, strings, 1);
    defer lengths.deinit();

    try testing.expectEqual(@as(usize, 2), lengths.items.len);
    try testing.expectEqual(@as(usize, 1), lengths.items[0]);
    try testing.expectEqual(@as(usize, 2), lengths.items[1]);
}

test "count strings with prefix" {
    const strings = &[_][]const u8{ "flower", "flow", "flight", "dog" };
    const count = countStringsWithPrefix(strings, "fl");
    try testing.expectEqual(@as(usize, 3), count);
}

test "count strings with prefix: no matches" {
    const strings = &[_][]const u8{ "dog", "racecar", "car" };
    const count = countStringsWithPrefix(strings, "fl");
    try testing.expectEqual(@as(usize, 0), count);
}

test "LCP of suffixes: basic" {
    const text = "banana";
    const positions = &[_]usize{ 1, 3 }; // "anana" and "ana"
    const lcp = longestCommonPrefixOfSuffixes(text, positions);
    try testing.expectEqualStrings("ana", lcp);
}

test "LCP of suffixes: no common prefix" {
    const text = "banana";
    const positions = &[_]usize{ 0, 1 }; // "banana" and "anana"
    const lcp = longestCommonPrefixOfSuffixes(text, positions);
    try testing.expectEqualStrings("", lcp);
}

test "LCP of suffixes: empty positions" {
    const text = "banana";
    const positions: []const usize = &.{};
    const lcp = longestCommonPrefixOfSuffixes(text, positions);
    try testing.expectEqualStrings("", lcp);
}

test "LCP of suffixes: single position" {
    const text = "banana";
    const positions = &[_]usize{2}; // "nana"
    const lcp = longestCommonPrefixOfSuffixes(text, positions);
    try testing.expectEqualStrings("nana", lcp);
}

test "LCP: consistency across methods" {
    const strings = &[_][]const u8{ "flower", "flow", "flight" };

    const lcp1 = longestCommonPrefix(strings);
    const lcp2 = longestCommonPrefixVertical(strings);
    const lcp3 = try longestCommonPrefixDivideConquer(testing.allocator, strings);
    defer testing.allocator.free(lcp3);
    const lcp4 = longestCommonPrefixBinarySearch(strings);

    try testing.expectEqualStrings(lcp1, lcp2);
    try testing.expectEqualStrings(lcp1, lcp3);
    try testing.expectEqualStrings(lcp1, lcp4);
}

test "LCP: large dataset" {
    const allocator = testing.allocator;
    var strings = std.ArrayList([]u8).init(allocator);
    defer {
        for (strings.items) |str| allocator.free(str);
        strings.deinit();
    }

    // Create 100 strings with common prefix "prefix_"
    for (0..100) |i| {
        const str = try std.fmt.allocPrint(allocator, "prefix_{d}_suffix", .{i});
        try strings.append(str);
    }

    const lcp = longestCommonPrefix(strings.items);
    try testing.expectEqualStrings("prefix_", lcp);
}

test "LCP: unicode strings" {
    const strings = &[_][]const u8{ "café", "cafard", "cabaret" };
    const lcp = longestCommonPrefix(strings);
    try testing.expectEqualStrings("ca", lcp);
}

test "LCP: memory safety" {
    const allocator = testing.allocator;
    for (0..10) |_| {
        const strings = &[_][]const u8{ "test", "testing", "tester" };
        const lcp = try longestCommonPrefixDivideConquer(allocator, strings);
        allocator.free(lcp);
    }
}
