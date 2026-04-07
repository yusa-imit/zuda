/// Anagram detection and manipulation algorithms.
///
/// An anagram is a word or phrase formed by rearranging the letters of another word or phrase,
/// typically using all the original letters exactly once.
///
/// ## Algorithms
///
/// 1. **Character Frequency Matching**:
///    - Count character frequencies for both strings
///    - Time: O(n), Space: O(1) for ASCII (fixed 256), O(k) for Unicode
///    - Best for single comparison
///
/// 2. **Sorted String Comparison**:
///    - Sort both strings and compare
///    - Time: O(n log n), Space: O(n) for sorting buffer
///    - Simple and effective
///
/// 3. **Hash-based Grouping**:
///    - Use canonical form (sorted string or frequency signature) as hash key
///    - Time: O(n × m) for n strings of length m, Space: O(n × m)
///    - Best for grouping multiple strings
///
/// ## Time Complexity
///
/// - `areAnagrams()`: O(n) where n = string length
/// - `areAnagramsSorted()`: O(n log n)
/// - `findAllAnagrams()`: O(n × m) where n = list size, m = average string length
/// - `groupAnagrams()`: O(n × m log m) for n strings of length m (sorting-based)
/// - `countAnagramPairs()`: O(n × m) for n strings
///
/// ## Space Complexity
///
/// - O(1) for frequency-based comparisons (fixed 256 for ASCII)
/// - O(n) for sorted comparison buffer
/// - O(n × m) for grouping hash maps
///
/// ## Use Cases
///
/// - Word games (Scrabble, word puzzles)
/// - Spell checkers (suggesting alternatives)
/// - Text analysis (finding related words)
/// - Data deduplication (grouping similar entries)
/// - Cryptography (transposition ciphers)
///
/// ## References
///
/// - LeetCode #242 (Valid Anagram)
/// - LeetCode #49 (Group Anagrams)
/// - LeetCode #438 (Find All Anagrams in a String)

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Check if two strings are anagrams using character frequency counting.
///
/// Two strings are anagrams if they contain the same characters with the same frequencies.
/// Case-sensitive by default.
///
/// Time: O(n + m) where n, m = string lengths | Space: O(1) for ASCII
///
/// ## Example
///
/// ```zig
/// const result1 = areAnagrams("listen", "silent");  // true
/// const result2 = areAnagrams("hello", "world");    // false
/// const result3 = areAnagrams("Astronomer", "Moon starer");  // false (spaces differ)
/// ```
pub fn areAnagrams(s1: []const u8, s2: []const u8) bool {
    // Quick length check
    if (s1.len != s2.len) return false;

    // Count character frequencies (ASCII)
    var freq: [256]i32 = [_]i32{0} ** 256;

    for (s1) |c| {
        freq[c] += 1;
    }

    for (s2) |c| {
        freq[c] -= 1;
    }

    // All frequencies should be zero
    for (freq) |count| {
        if (count != 0) return false;
    }

    return true;
}

/// Check if two strings are anagrams using sorting approach.
///
/// Sorts both strings and compares the results. Allocates temporary buffers.
///
/// Time: O(n log n) where n = string length | Space: O(n)
///
/// ## Example
///
/// ```zig
/// var result1 = try areAnagramsSorted(allocator, "anagram", "nagaram");  // true
/// var result2 = try areAnagramsSorted(allocator, "rat", "car");  // false
/// ```
pub fn areAnagramsSorted(allocator: Allocator, s1: []const u8, s2: []const u8) !bool {
    if (s1.len != s2.len) return false;

    // Create sorted copies
    var sorted1 = try allocator.dupe(u8, s1);
    defer allocator.free(sorted1);
    var sorted2 = try allocator.dupe(u8, s2);
    defer allocator.free(sorted2);

    std.mem.sort(u8, sorted1, {}, std.sort.asc(u8));
    std.mem.sort(u8, sorted2, {}, std.sort.asc(u8));

    return std.mem.eql(u8, sorted1, sorted2);
}

/// Find all anagrams of a pattern in a text (sliding window approach).
///
/// Returns indices where anagram substrings start.
///
/// Time: O(n) where n = text length | Space: O(1) for frequency arrays + O(k) for results
///
/// ## Example
///
/// ```zig
/// const text = "cbaebabacd";
/// const pattern = "abc";
/// var indices = try findAllAnagrams(allocator, text, pattern);
/// defer indices.deinit();
/// // indices = [0, 6] ("cba" at 0, "bac" at 6)
/// ```
pub fn findAllAnagrams(allocator: Allocator, text: []const u8, pattern: []const u8) !std.ArrayList(usize) {
    var result = std.ArrayList(usize).init(allocator);
    errdefer result.deinit();

    if (pattern.len > text.len) return result;

    // Count pattern frequencies
    var pattern_freq: [256]i32 = [_]i32{0} ** 256;
    for (pattern) |c| {
        pattern_freq[c] += 1;
    }

    // Sliding window
    var window_freq: [256]i32 = [_]i32{0} ** 256;

    // Initialize first window
    for (text[0..pattern.len]) |c| {
        window_freq[c] += 1;
    }

    // Check first window
    if (frequenciesMatch(&pattern_freq, &window_freq)) {
        try result.append(0);
    }

    // Slide window
    var i: usize = pattern.len;
    while (i < text.len) : (i += 1) {
        // Add new character
        window_freq[text[i]] += 1;
        // Remove old character
        window_freq[text[i - pattern.len]] -= 1;

        if (frequenciesMatch(&pattern_freq, &window_freq)) {
            try result.append(i - pattern.len + 1);
        }
    }

    return result;
}

/// Helper function to compare two frequency arrays.
fn frequenciesMatch(freq1: *const [256]i32, freq2: *const [256]i32) bool {
    for (freq1, 0..) |count, idx| {
        if (count != freq2[idx]) return false;
    }
    return true;
}

/// Group strings that are anagrams together.
///
/// Returns a list of groups, where each group contains anagram strings.
///
/// Time: O(n × m log m) where n = number of strings, m = average length | Space: O(n × m)
///
/// ## Example
///
/// ```zig
/// const words = [_][]const u8{ "eat", "tea", "tan", "ate", "nat", "bat" };
/// var groups = try groupAnagrams(allocator, &words);
/// defer {
///     for (groups.items) |group| group.deinit();
///     groups.deinit();
/// }
/// // groups = [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
/// ```
pub fn groupAnagrams(allocator: Allocator, strings: []const []const u8) !std.ArrayList(std.ArrayList([]const u8)) {
    var groups = std.ArrayList(std.ArrayList([]const u8)).init(allocator);
    errdefer {
        for (groups.items) |group| group.deinit();
        groups.deinit();
    }

    // Map from canonical form (sorted string) to group index
    var map = std.StringHashMap(usize).init(allocator);
    defer {
        var it = map.keyIterator();
        while (it.next()) |key| {
            allocator.free(key.*);
        }
        map.deinit();
    }

    for (strings) |str| {
        // Create canonical form (sorted string)
        var sorted = try allocator.dupe(u8, str);
        errdefer allocator.free(sorted);
        std.mem.sort(u8, sorted, {}, std.sort.asc(u8));

        if (map.get(sorted)) |group_idx| {
            // Add to existing group
            allocator.free(sorted);
            try groups.items[group_idx].append(str);
        } else {
            // Create new group
            var new_group = std.ArrayList([]const u8).init(allocator);
            errdefer new_group.deinit();
            try new_group.append(str);
            const group_idx = groups.items.len;
            try groups.append(new_group);
            try map.put(sorted, group_idx);
        }
    }

    return groups;
}

/// Count the number of anagram pairs in a list of strings.
///
/// A pair (i, j) is counted if i < j and strings[i] and strings[j] are anagrams.
///
/// Time: O(n² × m) where n = number of strings, m = average length | Space: O(1)
///
/// ## Example
///
/// ```zig
/// const words = [_][]const u8{ "listen", "silent", "enlist", "rat" };
/// const count = countAnagramPairs(&words);  // 3 (listen-silent, listen-enlist, silent-enlist)
/// ```
pub fn countAnagramPairs(strings: []const []const u8) usize {
    var count: usize = 0;
    for (strings, 0..) |s1, i| {
        for (strings[i + 1 ..]) |s2| {
            if (areAnagrams(s1, s2)) {
                count += 1;
            }
        }
    }
    return count;
}

/// Check if two strings are anagrams ignoring case and spaces.
///
/// Useful for phrase anagrams like "Astronomer" and "Moon starer".
///
/// Time: O(n) where n = string length | Space: O(1)
///
/// ## Example
///
/// ```zig
/// const result1 = areAnagramsIgnoreCaseAndSpaces("Astronomer", "Moon starer");  // true
/// const result2 = areAnagramsIgnoreCaseAndSpaces("Hello World", "world hello");  // true
/// ```
pub fn areAnagramsIgnoreCaseAndSpaces(s1: []const u8, s2: []const u8) bool {
    var freq: [256]i32 = [_]i32{0} ** 256;

    // Count s1 (lowercase, skip spaces)
    for (s1) |c| {
        if (c == ' ') continue;
        const lower = if (c >= 'A' and c <= 'Z') c + 32 else c;
        freq[lower] += 1;
    }

    // Count s2 (lowercase, skip spaces)
    for (s2) |c| {
        if (c == ' ') continue;
        const lower = if (c >= 'A' and c <= 'Z') c + 32 else c;
        freq[lower] -= 1;
    }

    // All frequencies should be zero
    for (freq) |count| {
        if (count != 0) return false;
    }

    return true;
}

/// Get the canonical form of a string (sorted characters).
///
/// Two strings have the same canonical form if and only if they are anagrams.
///
/// Time: O(n log n) | Space: O(n)
///
/// ## Example
///
/// ```zig
/// var canonical = try getCanonicalForm(allocator, "listen");
/// defer allocator.free(canonical);
/// // canonical = "eilnst"
/// ```
pub fn getCanonicalForm(allocator: Allocator, s: []const u8) ![]u8 {
    var result = try allocator.dupe(u8, s);
    std.mem.sort(u8, result, {}, std.sort.asc(u8));
    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "areAnagrams - basic examples" {
    try std.testing.expect(areAnagrams("listen", "silent"));
    try std.testing.expect(areAnagrams("anagram", "nagaram"));
    try std.testing.expect(areAnagrams("", ""));
    try std.testing.expect(!areAnagrams("hello", "world"));
    try std.testing.expect(!areAnagrams("rat", "car"));
}

test "areAnagrams - edge cases" {
    try std.testing.expect(!areAnagrams("a", "ab")); // different lengths
    try std.testing.expect(areAnagrams("a", "a")); // single character
    try std.testing.expect(!areAnagrams("ab", "aab")); // different frequencies
    try std.testing.expect(areAnagrams("aabbcc", "abcabc"));
}

test "areAnagramsSorted - basic examples" {
    const allocator = std.testing.allocator;
    try std.testing.expect(try areAnagramsSorted(allocator, "listen", "silent"));
    try std.testing.expect(try areAnagramsSorted(allocator, "anagram", "nagaram"));
    try std.testing.expect(!try areAnagramsSorted(allocator, "rat", "car"));
}

test "findAllAnagrams - basic examples" {
    const allocator = std.testing.allocator;

    {
        var result = try findAllAnagrams(allocator, "cbaebabacd", "abc");
        defer result.deinit();
        try std.testing.expectEqual(@as(usize, 2), result.items.len);
        try std.testing.expectEqual(@as(usize, 0), result.items[0]); // "cba"
        try std.testing.expectEqual(@as(usize, 6), result.items[1]); // "bac"
    }

    {
        var result = try findAllAnagrams(allocator, "abab", "ab");
        defer result.deinit();
        try std.testing.expectEqual(@as(usize, 3), result.items.len);
        try std.testing.expectEqual(@as(usize, 0), result.items[0]); // "ab"
        try std.testing.expectEqual(@as(usize, 1), result.items[1]); // "ba"
        try std.testing.expectEqual(@as(usize, 2), result.items[2]); // "ab"
    }
}

test "findAllAnagrams - edge cases" {
    const allocator = std.testing.allocator;

    {
        // Pattern longer than text
        var result = try findAllAnagrams(allocator, "abc", "abcd");
        defer result.deinit();
        try std.testing.expectEqual(@as(usize, 0), result.items.len);
    }

    {
        // No anagrams found
        var result = try findAllAnagrams(allocator, "abcdefgh", "xyz");
        defer result.deinit();
        try std.testing.expectEqual(@as(usize, 0), result.items.len);
    }

    {
        // Entire text is anagram
        var result = try findAllAnagrams(allocator, "abc", "abc");
        defer result.deinit();
        try std.testing.expectEqual(@as(usize, 1), result.items.len);
        try std.testing.expectEqual(@as(usize, 0), result.items[0]);
    }
}

test "groupAnagrams - basic examples" {
    const allocator = std.testing.allocator;

    const words = [_][]const u8{ "eat", "tea", "tan", "ate", "nat", "bat" };
    var groups = try groupAnagrams(allocator, &words);
    defer {
        for (groups.items) |group| group.deinit();
        groups.deinit();
    }

    try std.testing.expectEqual(@as(usize, 3), groups.items.len);

    // Check that groups contain expected anagrams (order may vary)
    var found_eat_group = false;
    var found_tan_group = false;
    var found_bat_group = false;

    for (groups.items) |group| {
        if (group.items.len == 3) {
            // eat, tea, ate group
            found_eat_group = true;
        } else if (group.items.len == 2) {
            // tan, nat group
            found_tan_group = true;
        } else if (group.items.len == 1) {
            // bat group
            found_bat_group = true;
        }
    }

    try std.testing.expect(found_eat_group);
    try std.testing.expect(found_tan_group);
    try std.testing.expect(found_bat_group);
}

test "groupAnagrams - edge cases" {
    const allocator = std.testing.allocator;

    {
        // Single string
        const words = [_][]const u8{"hello"};
        var groups = try groupAnagrams(allocator, &words);
        defer {
            for (groups.items) |group| group.deinit();
            groups.deinit();
        }
        try std.testing.expectEqual(@as(usize, 1), groups.items.len);
        try std.testing.expectEqual(@as(usize, 1), groups.items[0].items.len);
    }

    {
        // All anagrams
        const words = [_][]const u8{ "abc", "bca", "cab" };
        var groups = try groupAnagrams(allocator, &words);
        defer {
            for (groups.items) |group| group.deinit();
            groups.deinit();
        }
        try std.testing.expectEqual(@as(usize, 1), groups.items.len);
        try std.testing.expectEqual(@as(usize, 3), groups.items[0].items.len);
    }

    {
        // No anagrams
        const words = [_][]const u8{ "abc", "def", "ghi" };
        var groups = try groupAnagrams(allocator, &words);
        defer {
            for (groups.items) |group| group.deinit();
            groups.deinit();
        }
        try std.testing.expectEqual(@as(usize, 3), groups.items.len);
    }
}

test "countAnagramPairs - basic examples" {
    const words1 = [_][]const u8{ "listen", "silent", "enlist", "rat" };
    try std.testing.expectEqual(@as(usize, 3), countAnagramPairs(&words1));

    const words2 = [_][]const u8{ "abc", "def", "ghi" };
    try std.testing.expectEqual(@as(usize, 0), countAnagramPairs(&words2));

    const words3 = [_][]const u8{ "ab", "ba", "ab" };
    try std.testing.expectEqual(@as(usize, 3), countAnagramPairs(&words3));
}

test "areAnagramsIgnoreCaseAndSpaces - basic examples" {
    try std.testing.expect(areAnagramsIgnoreCaseAndSpaces("Astronomer", "Moon starer"));
    try std.testing.expect(areAnagramsIgnoreCaseAndSpaces("The Eyes", "They See"));
    try std.testing.expect(areAnagramsIgnoreCaseAndSpaces("Hello World", "world hello"));
    try std.testing.expect(!areAnagramsIgnoreCaseAndSpaces("Hello", "World"));
}

test "getCanonicalForm - basic examples" {
    const allocator = std.testing.allocator;

    {
        var canonical = try getCanonicalForm(allocator, "listen");
        defer allocator.free(canonical);
        try std.testing.expectEqualStrings("eilnst", canonical);
    }

    {
        var canonical = try getCanonicalForm(allocator, "silent");
        defer allocator.free(canonical);
        try std.testing.expectEqualStrings("eilnst", canonical);
    }

    {
        var canonical = try getCanonicalForm(allocator, "abc");
        defer allocator.free(canonical);
        try std.testing.expectEqualStrings("abc", canonical);
    }
}

test "anagrams - large input" {
    const allocator = std.testing.allocator;

    // Create large string with known anagram substring
    var text = try allocator.alloc(u8, 1000);
    defer allocator.free(text);
    @memset(text, 'a');
    text[100] = 'b';
    text[101] = 'c';
    text[102] = 'a';

    var result = try findAllAnagrams(allocator, text, "abc");
    defer result.deinit();
    try std.testing.expect(result.items.len > 0);
}

test "anagrams - memory safety" {
    const allocator = std.testing.allocator;

    // Test multiple allocations and deallocations
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var result = try findAllAnagrams(allocator, "abcabc", "abc");
        defer result.deinit();
    }
}
