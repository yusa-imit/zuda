// Longest Common Substring
// Dynamic programming algorithm for finding the longest contiguous substring common to two strings.
//
// Unlike Longest Common Subsequence (LCS) which finds the longest subsequence (not necessarily contiguous),
// this algorithm finds the longest contiguous substring that appears in both input strings.
//
// Algorithm:
// - DP recurrence: if s1[i] == s2[j], then dp[i][j] = dp[i-1][j-1] + 1
//                  else dp[i][j] = 0 (reset for non-contiguous)
// - Track maximum length and ending position during DP computation
// - Result is the substring s1[end-maxLen+1..end+1]
//
// Time complexity: O(n×m) where n = len(s1), m = len(s2)
// Space complexity: O(n×m) for tabulation, O(min(n,m)) for space-optimized

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Result of longest common substring search.
pub const SubstringResult = struct {
    /// Maximum length of common substring
    length: usize,
    /// Starting index in first string (0 if length == 0)
    start1: usize,
    /// Starting index in second string (0 if length == 0)
    start2: usize,

    /// Extract the actual substring from the first string.
    /// Caller does not own the returned slice (it's a view into the original).
    ///
    /// Time: O(1)
    /// Space: O(1)
    pub fn substring1(self: SubstringResult, s1: []const u8) []const u8 {
        if (self.length == 0) return "";
        return s1[self.start1 .. self.start1 + self.length];
    }

    /// Extract the actual substring from the second string.
    /// Caller does not own the returned slice (it's a view into the original).
    ///
    /// Time: O(1)
    /// Space: O(1)
    pub fn substring2(self: SubstringResult, s2: []const u8) []const u8 {
        if (self.length == 0) return "";
        return s2[self.start2 .. self.start2 + self.length];
    }
};

/// Find the longest common substring between two strings using tabulation (2D DP table).
/// Returns length and starting indices in both strings.
///
/// Algorithm:
/// - Build n×m table where dp[i][j] represents length of common substring ending at s1[i-1] and s2[j-1]
/// - If s1[i-1] == s2[j-1], dp[i][j] = dp[i-1][j-1] + 1, else dp[i][j] = 0
/// - Track maximum length and its ending position
/// - Starting indices are computed from ending position and length
///
/// Time: O(n×m)
/// Space: O(n×m)
pub fn longestCommonSubstring(allocator: Allocator, s1: []const u8, s2: []const u8) !SubstringResult {
    if (s1.len == 0 or s2.len == 0) {
        return SubstringResult{ .length = 0, .start1 = 0, .start2 = 0 };
    }

    const n = s1.len;
    const m = s2.len;

    // Allocate DP table: (n+1) × (m+1)
    const table = try allocator.alloc([]usize, n + 1);
    defer allocator.free(table);

    for (table) |*row| {
        row.* = try allocator.alloc(usize, m + 1);
    }
    defer for (table) |row| {
        allocator.free(row);
    };

    // Initialize first row and column to 0
    @memset(table[0], 0);
    for (1..n + 1) |i| {
        table[i][0] = 0;
    }

    var maxLen: usize = 0;
    var endIndex1: usize = 0; // ending index in s1 (exclusive)

    // Fill DP table
    for (1..n + 1) |i| {
        for (1..m + 1) |j| {
            if (s1[i - 1] == s2[j - 1]) {
                table[i][j] = table[i - 1][j - 1] + 1;
                if (table[i][j] > maxLen) {
                    maxLen = table[i][j];
                    endIndex1 = i;
                }
            } else {
                table[i][j] = 0; // Reset for non-contiguous
            }
        }
    }

    if (maxLen == 0) {
        return SubstringResult{ .length = 0, .start1 = 0, .start2 = 0 };
    }

    const start1 = endIndex1 - maxLen;

    // Find start2 by searching for the substring in s2
    const substr = s1[start1..endIndex1];
    const start2 = std.mem.indexOf(u8, s2, substr) orelse 0;

    return SubstringResult{
        .length = maxLen,
        .start1 = start1,
        .start2 = start2,
    };
}

/// Space-optimized version using only two rows (current and previous).
/// Returns length and starting indices.
///
/// Algorithm:
/// - Use rolling buffer with two rows instead of full table
/// - Same recurrence relation but only keep previous row
/// - Track maximum and ending position as before
/// - To find start2, we need to search s2 after computing the result
///
/// Time: O(n×m)
/// Space: O(min(n,m)) — uses smaller dimension for rows
pub fn longestCommonSubstringOptimized(allocator: Allocator, s1: []const u8, s2: []const u8) !SubstringResult {
    if (s1.len == 0 or s2.len == 0) {
        return SubstringResult{ .length = 0, .start1 = 0, .start2 = 0 };
    }

    // Ensure s2 is the smaller string (for space efficiency)
    if (s1.len < s2.len) {
        const result = try longestCommonSubstringOptimized(allocator, s2, s1);
        return SubstringResult{
            .length = result.length,
            .start1 = result.start2,
            .start2 = result.start1,
        };
    }

    const n = s1.len;
    const m = s2.len;

    // Two rows: previous and current
    var prev = try allocator.alloc(usize, m + 1);
    defer allocator.free(prev);
    var curr = try allocator.alloc(usize, m + 1);
    defer allocator.free(curr);

    @memset(prev, 0);
    @memset(curr, 0);

    var maxLen: usize = 0;
    var endIndex1: usize = 0;

    for (1..n + 1) |i| {
        curr[0] = 0;
        for (1..m + 1) |j| {
            if (s1[i - 1] == s2[j - 1]) {
                curr[j] = prev[j - 1] + 1;
                if (curr[j] > maxLen) {
                    maxLen = curr[j];
                    endIndex1 = i;
                }
            } else {
                curr[j] = 0;
            }
        }
        // Swap rows
        std.mem.swap([]usize, &prev, &curr);
    }

    if (maxLen == 0) {
        return SubstringResult{ .length = 0, .start1 = 0, .start2 = 0 };
    }

    const start1 = endIndex1 - maxLen;
    const substr = s1[start1..endIndex1];
    const start2 = std.mem.indexOf(u8, s2, substr) orelse 0;

    return SubstringResult{
        .length = maxLen,
        .start1 = start1,
        .start2 = start2,
    };
}

/// Find all common substrings of a given minimum length.
/// Returns an ArrayList of SubstringResult (caller owns the list).
///
/// Algorithm:
/// - Build full DP table as in longestCommonSubstring
/// - Collect all positions where table[i][j] >= minLength and table[i][j] > table[i+1][j] (end of substring)
/// - Deduplicate based on actual substring content
///
/// Time: O(n×m + k×L) where k = number of results, L = average substring length
/// Space: O(n×m + k)
pub fn allCommonSubstrings(allocator: Allocator, s1: []const u8, s2: []const u8, minLength: usize) !std.ArrayList(SubstringResult) {
    var results = std.ArrayList(SubstringResult).init(allocator);
    errdefer results.deinit();

    if (s1.len == 0 or s2.len == 0 or minLength == 0) {
        return results;
    }

    const n = s1.len;
    const m = s2.len;

    // Build DP table
    const table = try allocator.alloc([]usize, n + 1);
    defer allocator.free(table);

    for (table) |*row| {
        row.* = try allocator.alloc(usize, m + 1);
    }
    defer for (table) |row| {
        allocator.free(row);
    };

    @memset(table[0], 0);
    for (1..n + 1) |i| {
        table[i][0] = 0;
    }

    // Fill table and collect endpoints
    for (1..n + 1) |i| {
        for (1..m + 1) |j| {
            if (s1[i - 1] == s2[j - 1]) {
                table[i][j] = table[i - 1][j - 1] + 1;
            } else {
                table[i][j] = 0;
            }
        }
    }

    // Collect all substrings of length >= minLength
    // A substring ends at (i, j) if table[i][j] > 0 and (i == n or j == m or table[i+1][j] == 0 or table[i][j+1] == 0)
    var seen = std.StringHashMap(void).init(allocator);
    defer seen.deinit();

    for (1..n + 1) |i| {
        for (1..m + 1) |j| {
            const len = table[i][j];
            if (len >= minLength) {
                // Check if this is an endpoint (next cells are 0 or out of bounds)
                const isEnd = (i == n or j == m or table[i + 1][j] == 0 or table[i][j + 1] == 0);
                if (isEnd) {
                    const start1 = i - len;
                    const substr = s1[start1..i];

                    // Deduplicate by substring content
                    const gop = try seen.getOrPut(substr);
                    if (!gop.found_existing) {
                        const start2 = std.mem.indexOf(u8, s2, substr) orelse continue;
                        try results.append(SubstringResult{
                            .length = len,
                            .start1 = start1,
                            .start2 = start2,
                        });
                    }
                }
            }
        }
    }

    return results;
}

// Tests
const testing = std.testing;

test "longest common substring - basic" {
    const allocator = testing.allocator;

    const result = try longestCommonSubstring(allocator, "abcdef", "xbcde");
    try testing.expectEqual(@as(usize, 4), result.length);
    try testing.expectEqualStrings("bcde", result.substring1("abcdef"));
    try testing.expectEqualStrings("bcde", result.substring2("xbcde"));
}

test "longest common substring - multiple matches" {
    const allocator = testing.allocator;

    const result = try longestCommonSubstring(allocator, "abcxyzabc", "xyzabcxyz");
    try testing.expectEqual(@as(usize, 3), result.length);
    // Should find "abc" or "xyz" (both length 3)
    const substr = result.substring1("abcxyzabc");
    try testing.expect(std.mem.eql(u8, substr, "abc") or std.mem.eql(u8, substr, "xyz"));
}

test "longest common substring - no match" {
    const allocator = testing.allocator;

    const result = try longestCommonSubstring(allocator, "abc", "def");
    try testing.expectEqual(@as(usize, 0), result.length);
}

test "longest common substring - empty strings" {
    const allocator = testing.allocator;

    const r1 = try longestCommonSubstring(allocator, "", "abc");
    try testing.expectEqual(@as(usize, 0), r1.length);

    const r2 = try longestCommonSubstring(allocator, "abc", "");
    try testing.expectEqual(@as(usize, 0), r2.length);

    const r3 = try longestCommonSubstring(allocator, "", "");
    try testing.expectEqual(@as(usize, 0), r3.length);
}

test "longest common substring - single char match" {
    const allocator = testing.allocator;

    const result = try longestCommonSubstring(allocator, "a", "a");
    try testing.expectEqual(@as(usize, 1), result.length);
    try testing.expectEqualStrings("a", result.substring1("a"));
}

test "longest common substring - full match" {
    const allocator = testing.allocator;

    const result = try longestCommonSubstring(allocator, "hello", "hello");
    try testing.expectEqual(@as(usize, 5), result.length);
    try testing.expectEqualStrings("hello", result.substring1("hello"));
}

test "longest common substring - prefix/suffix" {
    const allocator = testing.allocator;

    const r1 = try longestCommonSubstring(allocator, "prefix_suffix", "prefix_different");
    try testing.expectEqual(@as(usize, 7), r1.length); // "prefix_"
    try testing.expectEqualStrings("prefix_", r1.substring1("prefix_suffix"));

    const r2 = try longestCommonSubstring(allocator, "different_suffix", "another_suffix");
    try testing.expectEqual(@as(usize, 7), r2.length); // "_suffix"
    try testing.expectEqualStrings("_suffix", r2.substring1("different_suffix"));
}

test "longest common substring - case sensitive" {
    const allocator = testing.allocator;

    const result = try longestCommonSubstring(allocator, "ABCD", "abcd");
    try testing.expectEqual(@as(usize, 0), result.length); // Case-sensitive
}

test "longest common substring - optimized basic" {
    const allocator = testing.allocator;

    const result = try longestCommonSubstringOptimized(allocator, "abcdef", "xbcde");
    try testing.expectEqual(@as(usize, 4), result.length);
    try testing.expectEqualStrings("bcde", result.substring1("abcdef"));
}

test "longest common substring - optimized swapped inputs" {
    const allocator = testing.allocator;

    const r1 = try longestCommonSubstringOptimized(allocator, "short", "very_long_string_short");
    try testing.expectEqual(@as(usize, 5), r1.length);
    try testing.expectEqualStrings("short", r1.substring1("short"));

    const r2 = try longestCommonSubstringOptimized(allocator, "very_long_string_short", "short");
    try testing.expectEqual(@as(usize, 5), r2.length);
    try testing.expectEqualStrings("short", r2.substring2("short"));
}

test "longest common substring - optimized empty" {
    const allocator = testing.allocator;

    const result = try longestCommonSubstringOptimized(allocator, "", "abc");
    try testing.expectEqual(@as(usize, 0), result.length);
}

test "longest common substring - large strings" {
    const allocator = testing.allocator;

    const s1 = "a" ** 50 ++ "common_part" ++ "b" ** 50;
    const s2 = "x" ** 30 ++ "common_part" ++ "y" ** 30;

    const result = try longestCommonSubstring(allocator, s1, s2);
    try testing.expectEqual(@as(usize, 11), result.length); // "common_part"
    try testing.expectEqualStrings("common_part", result.substring1(s1));

    const resultOpt = try longestCommonSubstringOptimized(allocator, s1, s2);
    try testing.expectEqual(@as(usize, 11), resultOpt.length);
}

test "longest common substring - all common (min length 2)" {
    const allocator = testing.allocator;

    var results = try allCommonSubstrings(allocator, "abcabc", "xabcy", 2);
    defer results.deinit();

    try testing.expect(results.items.len >= 1);
    // Should find "ab" and "bc" (both length 2)
    var foundAb = false;
    var foundBc = false;
    for (results.items) |r| {
        const substr = r.substring1("abcabc");
        if (std.mem.eql(u8, substr, "ab")) foundAb = true;
        if (std.mem.eql(u8, substr, "bc")) foundBc = true;
    }
    try testing.expect(foundAb);
    try testing.expect(foundBc);
}

test "longest common substring - all common (min length 3)" {
    const allocator = testing.allocator;

    var results = try allCommonSubstrings(allocator, "abcxyzabc", "xyzabcxyz", 3);
    defer results.deinit();

    try testing.expect(results.items.len >= 2);
    // Should find "abc" and "xyz" (both length 3)
    var foundAbc = false;
    var foundXyz = false;
    for (results.items) |r| {
        const substr = r.substring1("abcxyzabc");
        if (std.mem.eql(u8, substr, "abc")) foundAbc = true;
        if (std.mem.eql(u8, substr, "xyz")) foundXyz = true;
    }
    try testing.expect(foundAbc);
    try testing.expect(foundXyz);
}

test "longest common substring - all common empty" {
    const allocator = testing.allocator;

    var results = try allCommonSubstrings(allocator, "abc", "def", 1);
    defer results.deinit();

    try testing.expectEqual(@as(usize, 0), results.items.len);
}

test "longest common substring - all common no duplicates" {
    const allocator = testing.allocator;

    var results = try allCommonSubstrings(allocator, "aaaa", "aaaa", 2);
    defer results.deinit();

    // Should find "aa", "aaa", "aaaa" but deduplicate same substrings
    try testing.expect(results.items.len <= 3); // At most 3 unique substrings
}

test "longest common substring - indices validation" {
    const allocator = testing.allocator;

    const result = try longestCommonSubstring(allocator, "test_common_test", "another_common_end");
    try testing.expectEqual(@as(usize, 8), result.length); // "_common_"

    const substr1 = result.substring1("test_common_test");
    const substr2 = result.substring2("another_common_end");
    try testing.expectEqualStrings(substr1, substr2);
    try testing.expectEqualStrings("_common_", substr1);
}

test "longest common substring - memory safety" {
    const allocator = testing.allocator;

    // Multiple allocations and deallocations
    for (0..10) |_| {
        const result = try longestCommonSubstring(allocator, "memory", "test_memory_test");
        try testing.expectEqual(@as(usize, 6), result.length);

        var results = try allCommonSubstrings(allocator, "abc", "xabc", 1);
        defer results.deinit();
        try testing.expect(results.items.len > 0);
    }
}
