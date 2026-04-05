//! Palindrome Partitioning - Partition string into palindromic substrings
//!
//! ## Overview
//!
//! Given a string s, partition s such that every substring of the partition is a palindrome.
//! Return all possible palindrome partitioning of s.
//!
//! ## Example
//!
//! ```
//! Input: "aab"
//! Output: [["a","a","b"], ["aa","b"]]
//! ```
//!
//! ## Algorithm
//!
//! Uses backtracking to explore all possible partitions:
//! 1. Start from index 0
//! 2. Try all possible end positions that form palindromes
//! 3. Recursively partition the remaining substring
//! 4. Backtrack when reaching end or no valid partitions
//!
//! ## Time Complexity
//!
//! - Time: O(N × 2^N) where N is string length
//!   * 2^N possible partitions (each position can split or not)
//!   * N to check palindrome for each partition
//! - Space: O(N) for recursion stack + result storage
//!
//! ## Use Cases
//!
//! - String processing (text segmentation)
//! - DNA sequence analysis (finding palindromic patterns)
//! - Natural language processing (word boundary detection)
//! - Combinatorial generation (enumerate all valid partitions)

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

/// Partition string into all possible palindromic substrings.
///
/// Returns a list of all possible palindrome partitionings.
/// Caller owns the returned ArrayList and all inner ArrayLists.
///
/// Time: O(N × 2^N) | Space: O(N)
///
/// ## Example
///
/// ```zig
/// var result = try partition(allocator, "aab");
/// defer {
///     for (result.items) |part| {
///         for (part.items) |str| allocator.free(str);
///         part.deinit();
///     }
///     result.deinit();
/// }
/// // result = [["a","a","b"], ["aa","b"]]
/// ```
pub fn partition(allocator: Allocator, s: []const u8) !ArrayList(ArrayList([]const u8)) {
    var result = try ArrayList(ArrayList([]const u8)).initCapacity(allocator, 0);
    errdefer {
        for (result.items) |*part| {
            for (part.items) |str| allocator.free(str);
            part.deinit(allocator);
        }
        result.deinit(allocator);
    }

    var current = try ArrayList([]const u8).initCapacity(allocator, 0);
    defer {
        for (current.items) |str| allocator.free(str);
        current.deinit(allocator);
    }

    try backtrack(allocator, s, 0, &current, &result);
    return result;
}

/// Count total number of palindrome partitions without generating them.
///
/// More memory efficient than partition() when only count is needed.
///
/// Time: O(N × 2^N) | Space: O(N)
pub fn countPartitions(allocator: Allocator, s: []const u8) !usize {
    var count: usize = 0;
    var current = try ArrayList([]const u8).initCapacity(allocator, 0);
    defer {
        for (current.items) |str| allocator.free(str);
        current.deinit(allocator);
    }

    try backtrackCount(allocator, s, 0, &current, &count);
    return count;
}

/// Find minimum cuts needed to partition string into palindromes.
///
/// Uses dynamic programming to find minimum number of cuts.
/// Returns 0 if the entire string is already a palindrome.
///
/// Time: O(N²) | Space: O(N²)
///
/// ## Example
///
/// ```zig
/// const cuts = try minCut(allocator, "aab"); // 1 cut: "aa" | "b"
/// ```
pub fn minCut(allocator: Allocator, s: []const u8) !usize {
    if (s.len == 0) return 0;

    const n = s.len;

    // dp[i] = minimum cuts for s[0..i]
    var dp = try allocator.alloc(usize, n);
    defer allocator.free(dp);

    // is_palindrome[i][j] = true if s[i..j+1] is palindrome
    var is_palindrome = try allocator.alloc([]bool, n);
    defer {
        for (is_palindrome) |row| allocator.free(row);
        allocator.free(is_palindrome);
    }
    for (0..n) |i| {
        is_palindrome[i] = try allocator.alloc(bool, n);
        @memset(is_palindrome[i], false);
    }

    // Build palindrome table
    for (0..n) |i| {
        is_palindrome[i][i] = true;
    }
    for (0..n - 1) |i| {
        is_palindrome[i][i + 1] = s[i] == s[i + 1];
    }
    var len: usize = 3;
    while (len <= n) : (len += 1) {
        var i: usize = 0;
        while (i + len - 1 < n) : (i += 1) {
            const j = i + len - 1;
            is_palindrome[i][j] = (s[i] == s[j]) and is_palindrome[i + 1][j - 1];
        }
    }

    // DP for minimum cuts
    for (0..n) |i| {
        if (is_palindrome[0][i]) {
            dp[i] = 0;
        } else {
            dp[i] = i; // worst case: cut after every character
            for (0..i) |j| {
                if (is_palindrome[j + 1][i]) {
                    dp[i] = @min(dp[i], dp[j] + 1);
                }
            }
        }
    }

    return dp[n - 1];
}

/// Check if all partitions are palindromes.
///
/// Validates a partition result without allocating.
///
/// Time: O(K × L) where K is number of partitions, L is average length
pub fn isValidPartition(s: []const u8, parts: []const []const u8) bool {
    // Check concatenation equals original
    var total_len: usize = 0;
    for (parts) |part| total_len += part.len;
    if (total_len != s.len) return false;

    var pos: usize = 0;
    for (parts) |part| {
        // Check substring match
        if (!std.mem.eql(u8, part, s[pos .. pos + part.len])) return false;
        // Check palindrome
        if (!isPalindrome(part)) return false;
        pos += part.len;
    }
    return true;
}

// Helper: Backtracking for partition generation
fn backtrack(
    allocator: Allocator,
    s: []const u8,
    start: usize,
    current: *ArrayList([]const u8),
    result: *ArrayList(ArrayList([]const u8)),
) !void {
    // Base case: reached end of string
    if (start == s.len) {
        var part = try ArrayList([]const u8).initCapacity(allocator, current.items.len);
        for (current.items) |str| {
            const copy = try allocator.dupe(u8, str);
            part.appendAssumeCapacity(copy);
        }
        try result.append(allocator, part);
        return;
    }

    // Try all possible end positions
    var end = start;
    while (end < s.len) : (end += 1) {
        const substr = s[start .. end + 1];
        if (isPalindrome(substr)) {
            // Make choice
            const copy = try allocator.dupe(u8, substr);
            try current.append(allocator, copy);

            // Recurse
            try backtrack(allocator, s, end + 1, current, result);

            // Backtrack
            const removed = current.pop().?;
            allocator.free(removed);
        }
    }
}

// Helper: Backtracking for counting only
fn backtrackCount(
    allocator: Allocator,
    s: []const u8,
    start: usize,
    current: *ArrayList([]const u8),
    count: *usize,
) !void {
    if (start == s.len) {
        count.* += 1;
        return;
    }

    var end = start;
    while (end < s.len) : (end += 1) {
        const substr = s[start .. end + 1];
        if (isPalindrome(substr)) {
            const copy = try allocator.dupe(u8, substr);
            try current.append(allocator, copy);
            try backtrackCount(allocator, s, end + 1, current, count);
            const removed = current.pop().?;
            allocator.free(removed);
        }
    }
}

// Helper: Check if string is palindrome
fn isPalindrome(s: []const u8) bool {
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

// Tests

test "partition - basic example 'aab'" {
    const allocator = std.testing.allocator;

    var result = try partition(allocator, "aab");
    defer {
        for (result.items) |*part| {
            for (part.items) |str| allocator.free(str);
            part.deinit(allocator);
        }
        result.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 2), result.items.len);

    // Expected: [["a","a","b"], ["aa","b"]]
    // Sort for consistent comparison
    const first = result.items[0].items;
    const second = result.items[1].items;

    if (first.len == 3) {
        try std.testing.expectEqualStrings("a", first[0]);
        try std.testing.expectEqualStrings("a", first[1]);
        try std.testing.expectEqualStrings("b", first[2]);

        try std.testing.expectEqualStrings("aa", second[0]);
        try std.testing.expectEqualStrings("b", second[1]);
    } else {
        try std.testing.expectEqualStrings("aa", first[0]);
        try std.testing.expectEqualStrings("b", first[1]);

        try std.testing.expectEqualStrings("a", second[0]);
        try std.testing.expectEqualStrings("a", second[1]);
        try std.testing.expectEqualStrings("b", second[2]);
    }
}

test "partition - single character" {
    const allocator = std.testing.allocator;

    var result = try partition(allocator, "a");
    defer {
        for (result.items) |*part| {
            for (part.items) |str| allocator.free(str);
            part.deinit(allocator);
        }
        result.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), result.items.len);
    try std.testing.expectEqual(@as(usize, 1), result.items[0].items.len);
    try std.testing.expectEqualStrings("a", result.items[0].items[0]);
}

test "partition - empty string" {
    const allocator = std.testing.allocator;

    var result = try partition(allocator, "");
    defer {
        for (result.items) |*part| {
            for (part.items) |str| allocator.free(str);
            part.deinit(allocator);
        }
        result.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), result.items.len);
    try std.testing.expectEqual(@as(usize, 0), result.items[0].items.len);
}

test "partition - all palindrome 'aba'" {
    const allocator = std.testing.allocator;

    var result = try partition(allocator, "aba");
    defer {
        for (result.items) |*part| {
            for (part.items) |str| allocator.free(str);
            part.deinit(allocator);
        }
        result.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 2), result.items.len);

    // Expected: [["a","b","a"], ["aba"]]
    var found_single = false;
    var found_whole = false;

    for (result.items) |part| {
        if (part.items.len == 1) {
            try std.testing.expectEqualStrings("aba", part.items[0]);
            found_whole = true;
        } else if (part.items.len == 3) {
            try std.testing.expectEqualStrings("a", part.items[0]);
            try std.testing.expectEqualStrings("b", part.items[1]);
            try std.testing.expectEqualStrings("a", part.items[2]);
            found_single = true;
        }
    }

    try std.testing.expect(found_single and found_whole);
}

test "partition - no palindrome 'abc'" {
    const allocator = std.testing.allocator;

    var result = try partition(allocator, "abc");
    defer {
        for (result.items) |*part| {
            for (part.items) |str| allocator.free(str);
            part.deinit(allocator);
        }
        result.deinit(allocator);
    }

    // Only single characters are palindromes
    try std.testing.expectEqual(@as(usize, 1), result.items.len);
    try std.testing.expectEqual(@as(usize, 3), result.items[0].items.len);
    try std.testing.expectEqualStrings("a", result.items[0].items[0]);
    try std.testing.expectEqualStrings("b", result.items[0].items[1]);
    try std.testing.expectEqualStrings("c", result.items[0].items[2]);
}

test "partition - repeated characters 'aaa'" {
    const allocator = std.testing.allocator;

    var result = try partition(allocator, "aaa");
    defer {
        for (result.items) |*part| {
            for (part.items) |str| allocator.free(str);
            part.deinit(allocator);
        }
        result.deinit(allocator);
    }

    // Expected: [["a","a","a"], ["a","aa"], ["aa","a"], ["aaa"]]
    try std.testing.expectEqual(@as(usize, 4), result.items.len);
}

test "countPartitions - matches partition count" {
    const allocator = std.testing.allocator;

    const count1 = try countPartitions(allocator, "aab");
    try std.testing.expectEqual(@as(usize, 2), count1);

    const count2 = try countPartitions(allocator, "aba");
    try std.testing.expectEqual(@as(usize, 2), count2);

    const count3 = try countPartitions(allocator, "aaa");
    try std.testing.expectEqual(@as(usize, 4), count3);

    const count4 = try countPartitions(allocator, "abc");
    try std.testing.expectEqual(@as(usize, 1), count4);
}

test "countPartitions - empty string" {
    const allocator = std.testing.allocator;

    const count = try countPartitions(allocator, "");
    try std.testing.expectEqual(@as(usize, 1), count);
}

test "minCut - basic examples" {
    const allocator = std.testing.allocator;

    const cuts1 = try minCut(allocator, "aab");
    try std.testing.expectEqual(@as(usize, 1), cuts1); // "aa" | "b"

    const cuts2 = try minCut(allocator, "aba");
    try std.testing.expectEqual(@as(usize, 0), cuts2); // already palindrome

    const cuts3 = try minCut(allocator, "abc");
    try std.testing.expectEqual(@as(usize, 2), cuts3); // "a" | "b" | "c"

    const cuts4 = try minCut(allocator, "aaa");
    try std.testing.expectEqual(@as(usize, 0), cuts4); // already palindrome
}

test "minCut - empty and single char" {
    const allocator = std.testing.allocator;

    const cuts1 = try minCut(allocator, "");
    try std.testing.expectEqual(@as(usize, 0), cuts1);

    const cuts2 = try minCut(allocator, "a");
    try std.testing.expectEqual(@as(usize, 0), cuts2);
}

test "minCut - longer string" {
    const allocator = std.testing.allocator;

    const cuts = try minCut(allocator, "abcba");
    try std.testing.expectEqual(@as(usize, 0), cuts); // already palindrome
}

test "minCut - complex case" {
    const allocator = std.testing.allocator;

    const cuts = try minCut(allocator, "apjesgpsxoeiokmqmfgvjslcjukbqxpsobyhjpbgdfruqdkeiszrlmtwgfxyfostpqczidfljwfbbrflkgdvtytbgqalguewnhvvmcgxboycffopmtmhtfizxkmeftcucxpobxmelmjtuzigsxnncxpaibgpuijwhankxbplpyejxmrrjgeoevqozwdtgospohznkoyzocjlracchjqnggbfeebmuvbicbvmpuleywrpzwsihivnrwtxcukwplgtobhgxukwrdlszfaiqxwjvrgxnsveedxseeyeykarqnjrtlaliyudpacctzizcftjlunlgnfwcqqxcqikocqffsjyurzwysfjmswvhbrmshjuzsgpwyubtfbnwajuvrfhlccvfwhxfqthkcwhatktymgxostjlztwdxritygbrbibdgkezvzajizxasjnrcjwzdfvdnwwqeyumkamhzoqhnqjfzwzbixclcxqrtniznemxeahfozp");
    try std.testing.expect(cuts > 0); // Should require cuts
}

test "isValidPartition - valid partitions" {
    const parts1 = [_][]const u8{ "a", "a", "b" };
    try std.testing.expect(isValidPartition("aab", &parts1));

    const parts2 = [_][]const u8{ "aa", "b" };
    try std.testing.expect(isValidPartition("aab", &parts2));

    const parts3 = [_][]const u8{"aba"};
    try std.testing.expect(isValidPartition("aba", &parts3));
}

test "isValidPartition - invalid partitions" {
    const parts1 = [_][]const u8{ "ab", "b" }; // "ab" not palindrome
    try std.testing.expect(!isValidPartition("aab", &parts1));

    const parts2 = [_][]const u8{ "a", "b" }; // doesn't cover all
    try std.testing.expect(!isValidPartition("aab", &parts2));

    const parts3 = [_][]const u8{ "a", "a", "a" }; // too many
    try std.testing.expect(!isValidPartition("aa", &parts3));
}

test "isPalindrome - helper" {
    try std.testing.expect(isPalindrome(""));
    try std.testing.expect(isPalindrome("a"));
    try std.testing.expect(isPalindrome("aa"));
    try std.testing.expect(isPalindrome("aba"));
    try std.testing.expect(isPalindrome("abba"));
    try std.testing.expect(!isPalindrome("ab"));
    try std.testing.expect(!isPalindrome("abc"));
    try std.testing.expect(!isPalindrome("abca"));
}

test "partition - longer palindromic string" {
    const allocator = std.testing.allocator;

    var result = try partition(allocator, "racecar");
    defer {
        for (result.items) |*part| {
            for (part.items) |str| allocator.free(str);
            part.deinit(allocator);
        }
        result.deinit(allocator);
    }

    // Should include full palindrome partition
    var found_full = false;
    for (result.items) |part| {
        if (part.items.len == 1 and std.mem.eql(u8, part.items[0], "racecar")) {
            found_full = true;
        }
    }
    try std.testing.expect(found_full);
}

test "partition - memory safety" {
    const allocator = std.testing.allocator;

    // Multiple allocations and deallocations
    for (0..5) |_| {
        var result = try partition(allocator, "aab");
        for (result.items) |*part| {
            for (part.items) |str| allocator.free(str);
            part.deinit(allocator);
        }
        result.deinit(allocator);
    }
}
