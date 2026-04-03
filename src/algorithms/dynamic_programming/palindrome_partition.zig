const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

/// Palindrome Partitioning — Minimum cuts to partition string into palindromes
///
/// Given a string, partition it such that every substring is a palindrome.
/// This module provides three dynamic programming variants:
///
/// 1. minCuts(): Find minimum cuts needed (O(n²) time, O(n²) space)
/// 2. allPartitions(): Get all possible palindrome partitions (O(n×2ⁿ) time)
/// 3. isPalindrome(): Helper to check if substring is palindrome (O(1) with memoization)
///
/// ## Algorithm
/// Classic DP problem with two stages:
/// 1. Build palindrome lookup table (isPalin[i][j] = true if s[i..j] is palindrome)
/// 2. Compute minimum cuts: cuts[i] = min(cuts[j-1] + 1) for all j where s[j..i] is palindrome
///
/// ## Key Concepts
/// - **Optimal Substructure**: If s[0..i] has minimum k cuts, and s[j..i] is palindrome,
///   then s[0..j-1] must have k-1 cuts (optimal for that prefix)
/// - **Palindrome Table**: isPalin[i][j] = (s[i] == s[j]) && (j-i < 2 || isPalin[i+1][j-1])
///   Base case: single char is palindrome, two same chars are palindrome
/// - **Backtracking**: Reconstruct all partitions by exploring all valid cuts
///
/// ## Time Complexity
/// - minCuts(): O(n²) — build palindrome table O(n²) + compute cuts O(n²)
/// - allPartitions(): O(n×2ⁿ) — exponential in worst case (all chars different)
///
/// ## Space Complexity
/// - minCuts(): O(n²) for palindrome table + O(n) for cuts array
/// - allPartitions(): O(n²) for palindrome table + O(n) recursion depth
///
/// ## Use Cases
/// - Text processing: sentence segmentation with palindrome constraints
/// - Bioinformatics: DNA sequence analysis (palindromic repeats)
/// - Compression: palindrome-based encoding schemes
/// - Pattern recognition: finding palindromic structures
///
/// ## References
/// - LeetCode #131 (Palindrome Partitioning)
/// - LeetCode #132 (Palindrome Partitioning II)

/// Find minimum number of cuts needed to partition string into palindromes
///
/// Uses dynamic programming with palindrome memoization table.
/// For each position i, try all possible last palindromes ending at i.
///
/// ## Algorithm
/// 1. Build palindrome table: O(n²) time, O(n²) space
///    isPalin[i][j] = true if substring s[i..j+1] is palindrome
/// 2. DP cuts array: cuts[i] = min cuts for prefix s[0..i+1]
///    For each i, try all j where s[j..i+1] is palindrome:
///    cuts[i] = min(cuts[i], cuts[j-1] + 1) if j > 0
///    cuts[i] = 0 if entire prefix is palindrome (j == 0)
///
/// ## Example
/// ```
/// s = "aab" → min cuts = 1
/// Partition: "aa" | "b" (1 cut)
///
/// s = "aba" → min cuts = 0
/// No cuts needed: entire string is palindrome
/// ```
///
/// ## Time: O(n²) — palindrome table + DP computation
/// ## Space: O(n²) — palindrome table + O(n) cuts array
pub fn minCuts(allocator: Allocator, s: []const u8) !usize {
    const n = s.len;
    if (n <= 1) return 0;

    // Build palindrome table
    var isPalin = try allocator.alloc([]bool, n);
    defer allocator.free(isPalin);
    for (isPalin) |*row| {
        row.* = try allocator.alloc(bool, n);
    }
    defer for (isPalin) |row| {
        allocator.free(row);
    };

    // Initialize: single chars and adjacent same chars
    for (0..n) |i| {
        isPalin[i][i] = true;
        if (i + 1 < n) {
            isPalin[i][i + 1] = (s[i] == s[i + 1]);
        }
    }

    // Fill palindrome table: length 3 to n
    var len: usize = 3;
    while (len <= n) : (len += 1) {
        var i: usize = 0;
        while (i + len <= n) : (i += 1) {
            const j = i + len - 1;
            isPalin[i][j] = (s[i] == s[j]) and isPalin[i + 1][j - 1];
        }
    }

    // DP for minimum cuts
    var cuts = try allocator.alloc(usize, n);
    defer allocator.free(cuts);

    for (0..n) |i| {
        if (isPalin[0][i]) {
            cuts[i] = 0; // Entire prefix is palindrome
            continue;
        }

        cuts[i] = i; // Worst case: i cuts (every char separate)
        for (0..i) |j| {
            if (isPalin[j + 1][i]) {
                cuts[i] = @min(cuts[i], cuts[j] + 1);
            }
        }
    }

    return cuts[n - 1];
}

/// Result for all palindrome partitions
pub const PartitionResult = struct {
    /// List of all possible partitions
    /// Each partition is a list of palindrome strings
    partitions: ArrayList(ArrayList([]const u8)),

    pub fn deinit(self: *PartitionResult) void {
        for (self.partitions.items) |part| {
            // Strings are slices into original, don't free them
            part.deinit();
        }
        self.partitions.deinit();
    }
};

/// Get all possible palindrome partitions of string
///
/// Uses backtracking with palindrome memoization table.
/// Explores all possible cuts that result in valid palindrome partitions.
///
/// ## Algorithm
/// 1. Build palindrome table: O(n²) time
/// 2. Backtrack from start position:
///    - Try all possible end positions where s[start..end] is palindrome
///    - Recursively partition remaining substring
///    - Collect all valid complete partitions
///
/// ## Example
/// ```
/// s = "aab"
/// All partitions:
/// 1. ["a", "a", "b"]
/// 2. ["aa", "b"]
/// ```
///
/// ## Time: O(n×2ⁿ) — worst case all chars different, 2ⁿ partitions
/// ## Space: O(n²) for palindrome table + O(n) recursion depth
pub fn allPartitions(allocator: Allocator, s: []const u8) !PartitionResult {
    const n = s.len;
    var result = PartitionResult{
        .partitions = ArrayList(ArrayList([]const u8)).init(allocator),
    };

    if (n == 0) {
        return result;
    }

    // Build palindrome table
    var isPalin = try allocator.alloc([]bool, n);
    defer allocator.free(isPalin);
    for (isPalin) |*row| {
        row.* = try allocator.alloc(bool, n);
    }
    defer for (isPalin) |row| {
        allocator.free(row);
    };

    // Initialize palindrome table
    for (0..n) |i| {
        isPalin[i][i] = true;
        if (i + 1 < n) {
            isPalin[i][i + 1] = (s[i] == s[i + 1]);
        }
    }

    var len: usize = 3;
    while (len <= n) : (len += 1) {
        var i: usize = 0;
        while (i + len <= n) : (i += 1) {
            const j = i + len - 1;
            isPalin[i][j] = (s[i] == s[j]) and isPalin[i + 1][j - 1];
        }
    }

    // Backtrack to find all partitions
    var current = ArrayList([]const u8).init(allocator);
    defer current.deinit();

    try backtrack(allocator, s, 0, &isPalin, &current, &result.partitions);

    return result;
}

fn backtrack(
    allocator: Allocator,
    s: []const u8,
    start: usize,
    isPalin: *const [][]bool,
    current: *ArrayList([]const u8),
    result: *ArrayList(ArrayList([]const u8)),
) !void {
    if (start == s.len) {
        // Found complete partition
        var partition = ArrayList([]const u8).init(allocator);
        try partition.appendSlice(current.items);
        try result.append(partition);
        return;
    }

    // Try all possible end positions
    for (start..s.len) |end| {
        if (isPalin.*[start][end]) {
            try current.append(s[start .. end + 1]);
            try backtrack(allocator, s, end + 1, isPalin, current, result);
            _ = current.pop();
        }
    }
}

/// Check if substring s[i..j+1] is palindrome
///
/// Helper function for testing and validation.
///
/// ## Time: O(n) where n = j - i + 1
/// ## Space: O(1)
pub fn isPalindrome(s: []const u8, i: usize, j: usize) bool {
    if (i > j) return false;
    var left = i;
    var right = j;
    while (left < right) {
        if (s[left] != s[right]) return false;
        left += 1;
        right -= 1;
    }
    return true;
}

// ============================================================================
// Tests
// ============================================================================

test "palindrome partition - minCuts basic" {
    const allocator = std.testing.allocator;

    // "aab" → 1 cut: "aa" | "b"
    {
        const cuts = try minCuts(allocator, "aab");
        try std.testing.expectEqual(@as(usize, 1), cuts);
    }

    // "aba" → 0 cuts: entire string is palindrome
    {
        const cuts = try minCuts(allocator, "aba");
        try std.testing.expectEqual(@as(usize, 0), cuts);
    }

    // "abcde" → 4 cuts: all chars separate
    {
        const cuts = try minCuts(allocator, "abcde");
        try std.testing.expectEqual(@as(usize, 4), cuts);
    }
}

test "palindrome partition - minCuts edge cases" {
    const allocator = std.testing.allocator;

    // Empty string
    {
        const cuts = try minCuts(allocator, "");
        try std.testing.expectEqual(@as(usize, 0), cuts);
    }

    // Single character
    {
        const cuts = try minCuts(allocator, "a");
        try std.testing.expectEqual(@as(usize, 0), cuts);
    }

    // Two same characters
    {
        const cuts = try minCuts(allocator, "aa");
        try std.testing.expectEqual(@as(usize, 0), cuts);
    }

    // Two different characters
    {
        const cuts = try minCuts(allocator, "ab");
        try std.testing.expectEqual(@as(usize, 1), cuts);
    }
}

test "palindrome partition - minCuts complex" {
    const allocator = std.testing.allocator;

    // "racecar" → 0 cuts: entire string is palindrome
    {
        const cuts = try minCuts(allocator, "racecar");
        try std.testing.expectEqual(@as(usize, 0), cuts);
    }

    // "abacabad" → 3 cuts: "aba" | "c" | "aba" | "d"
    {
        const cuts = try minCuts(allocator, "abacabad");
        try std.testing.expectEqual(@as(usize, 3), cuts);
    }

    // "abaabcd" → 3 cuts: "aba" | "a" | "b" | "cd" or "abaa" | "b" | "c" | "d"
    {
        const cuts = try minCuts(allocator, "abaabcd");
        try std.testing.expectEqual(@as(usize, 3), cuts);
    }
}

test "palindrome partition - allPartitions basic" {
    const allocator = std.testing.allocator;

    // "aab" → 2 partitions
    {
        var result = try allPartitions(allocator, "aab");
        defer result.deinit();

        try std.testing.expectEqual(@as(usize, 2), result.partitions.items.len);

        // Check first partition: ["a", "a", "b"]
        const p1 = result.partitions.items[0];
        try std.testing.expectEqual(@as(usize, 3), p1.items.len);
        try std.testing.expectEqualStrings("a", p1.items[0]);
        try std.testing.expectEqualStrings("a", p1.items[1]);
        try std.testing.expectEqualStrings("b", p1.items[2]);

        // Check second partition: ["aa", "b"]
        const p2 = result.partitions.items[1];
        try std.testing.expectEqual(@as(usize, 2), p2.items.len);
        try std.testing.expectEqualStrings("aa", p2.items[0]);
        try std.testing.expectEqualStrings("b", p2.items[1]);
    }
}

test "palindrome partition - allPartitions single palindrome" {
    const allocator = std.testing.allocator;

    // "aba" → 1 partition: entire string
    {
        var result = try allPartitions(allocator, "aba");
        defer result.deinit();

        try std.testing.expectEqual(@as(usize, 1), result.partitions.items.len);

        const p = result.partitions.items[0];
        try std.testing.expectEqual(@as(usize, 1), p.items.len);
        try std.testing.expectEqualStrings("aba", p.items[0]);
    }
}

test "palindrome partition - allPartitions no palindromes" {
    const allocator = std.testing.allocator;

    // "abc" → 1 partition: all chars separate
    {
        var result = try allPartitions(allocator, "abc");
        defer result.deinit();

        try std.testing.expectEqual(@as(usize, 1), result.partitions.items.len);

        const p = result.partitions.items[0];
        try std.testing.expectEqual(@as(usize, 3), p.items.len);
        try std.testing.expectEqualStrings("a", p.items[0]);
        try std.testing.expectEqualStrings("b", p.items[1]);
        try std.testing.expectEqualStrings("c", p.items[2]);
    }
}

test "palindrome partition - allPartitions edge cases" {
    const allocator = std.testing.allocator;

    // Empty string
    {
        var result = try allPartitions(allocator, "");
        defer result.deinit();
        try std.testing.expectEqual(@as(usize, 0), result.partitions.items.len);
    }

    // Single character
    {
        var result = try allPartitions(allocator, "a");
        defer result.deinit();
        try std.testing.expectEqual(@as(usize, 1), result.partitions.items.len);
        try std.testing.expectEqual(@as(usize, 1), result.partitions.items[0].items.len);
        try std.testing.expectEqualStrings("a", result.partitions.items[0].items[0]);
    }

    // Two same characters
    {
        var result = try allPartitions(allocator, "aa");
        defer result.deinit();
        try std.testing.expectEqual(@as(usize, 2), result.partitions.items.len);
    }
}

test "palindrome partition - allPartitions multiple palindromes" {
    const allocator = std.testing.allocator;

    // "abaaba" → multiple partitions including full palindrome
    {
        var result = try allPartitions(allocator, "abaaba");
        defer result.deinit();

        // Should have multiple partitions
        try std.testing.expect(result.partitions.items.len > 1);

        // One partition should be entire string
        var foundFull = false;
        for (result.partitions.items) |p| {
            if (p.items.len == 1 and std.mem.eql(u8, p.items[0], "abaaba")) {
                foundFull = true;
                break;
            }
        }
        try std.testing.expect(foundFull);
    }
}

test "palindrome partition - isPalindrome helper" {
    // Basic palindromes
    try std.testing.expect(isPalindrome("aba", 0, 2));
    try std.testing.expect(isPalindrome("racecar", 0, 6));
    try std.testing.expect(isPalindrome("aa", 0, 1));
    try std.testing.expect(isPalindrome("a", 0, 0));

    // Non-palindromes
    try std.testing.expect(!isPalindrome("abc", 0, 2));
    try std.testing.expect(!isPalindrome("ab", 0, 1));

    // Substrings
    try std.testing.expect(isPalindrome("abcba", 0, 4));
    try std.testing.expect(isPalindrome("abcba", 1, 3)); // "bcb"
    try std.testing.expect(!isPalindrome("abcba", 0, 3)); // "abcb"
}

test "palindrome partition - allPartitions count validation" {
    const allocator = std.testing.allocator;

    // "aa" → 2 partitions: ["a", "a"] and ["aa"]
    {
        var result = try allPartitions(allocator, "aa");
        defer result.deinit();
        try std.testing.expectEqual(@as(usize, 2), result.partitions.items.len);
    }

    // "aaa" → 4 partitions: ["a","a","a"], ["aa","a"], ["a","aa"], ["aaa"]
    {
        var result = try allPartitions(allocator, "aaa");
        defer result.deinit();
        try std.testing.expectEqual(@as(usize, 4), result.partitions.items.len);
    }
}

test "palindrome partition - minCuts long string" {
    const allocator = std.testing.allocator;

    // Alternating pattern
    {
        const cuts = try minCuts(allocator, "abababab");
        try std.testing.expect(cuts >= 3); // At least some cuts needed
    }

    // All same character → 0 cuts
    {
        const cuts = try minCuts(allocator, "aaaaaaa");
        try std.testing.expectEqual(@as(usize, 0), cuts);
    }
}

test "palindrome partition - allPartitions palindrome validation" {
    const allocator = std.testing.allocator;

    // Verify all partitions contain only palindromes
    var result = try allPartitions(allocator, "aab");
    defer result.deinit();

    for (result.partitions.items) |partition| {
        for (partition.items) |substr| {
            try std.testing.expect(isPalindrome(substr, 0, substr.len - 1));
        }
    }
}

test "palindrome partition - memory safety" {
    const allocator = std.testing.allocator;

    // Test minCuts doesn't leak
    {
        const cuts = try minCuts(allocator, "abcdefgh");
        _ = cuts;
    }

    // Test allPartitions doesn't leak
    {
        var result = try allPartitions(allocator, "abab");
        defer result.deinit();
    }
}
