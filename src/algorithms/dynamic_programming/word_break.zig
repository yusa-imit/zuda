const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Word Break: Determine if a string can be segmented into dictionary words
///
/// Given a string s and a dictionary of words, determine if s can be segmented
/// into a space-separated sequence of one or more dictionary words.
///
/// Example:
///   s = "leetcode", dict = ["leet", "code"] → true
///   s = "applepenapple", dict = ["apple", "pen"] → true
///   s = "catsandog", dict = ["cats", "dog", "sand", "and", "cat"] → false
///
/// This module provides three variants:
/// 1. canBreak(): Boolean decision (can the string be segmented?)
/// 2. countBreaks(): Count all possible segmentations
/// 3. allBreaks(): Return all possible segmentations
///
/// Time: O(n²×m) where n = string length, m = dictionary size
/// Space: O(n) for DP table, O(n×k) for segmentation reconstruction where k = number of segmentations

/// Check if a string can be segmented into dictionary words
///
/// Uses dynamic programming: dp[i] = true if s[0..i] can be segmented
/// Recurrence: dp[i] = true if any dp[j] is true AND s[j..i] is in dictionary
///
/// Time: O(n²×m) where n = string length, m = dictionary size (worst case checks all substrings)
/// Space: O(n) for DP table
///
/// Example:
/// ```zig
/// var gpa = std.heap.GeneralPurposeAllocator(.{}){};
/// defer _ = gpa.deinit();
/// const allocator = gpa.allocator();
///
/// var dict = std.StringHashMap(void).init(allocator);
/// defer dict.deinit();
/// try dict.put("leet", {});
/// try dict.put("code", {});
///
/// const result = try canBreak(allocator, "leetcode", dict);
/// // result == true
/// ```
pub fn canBreak(allocator: Allocator, s: []const u8, dict: std.StringHashMap(void)) !bool {
    const n = s.len;
    if (n == 0) return true;

    // dp[i] = true if s[0..i] can be segmented
    var dp = try allocator.alloc(bool, n + 1);
    defer allocator.free(dp);
    @memset(dp, false);
    dp[0] = true; // Empty string can always be segmented

    // For each position i
    for (1..n + 1) |i| {
        // Try all possible last words ending at i
        for (0..i) |j| {
            if (dp[j] and dict.contains(s[j..i])) {
                dp[i] = true;
                break;
            }
        }
    }

    return dp[n];
}

/// Count the number of ways to segment a string into dictionary words
///
/// Uses dynamic programming: dp[i] = number of ways to segment s[0..i]
/// Recurrence: dp[i] = sum of dp[j] for all j where s[j..i] is in dictionary
///
/// Time: O(n²×m) where n = string length, m = dictionary size
/// Space: O(n) for DP table
///
/// Example:
/// ```zig
/// var dict = std.StringHashMap(void).init(allocator);
/// defer dict.deinit();
/// try dict.put("cat", {});
/// try dict.put("cats", {});
/// try dict.put("and", {});
/// try dict.put("sand", {});
/// try dict.put("dog", {});
///
/// const count = try countBreaks(allocator, "catsanddog", dict);
/// // count == 2 (ways: "cats and dog" or "cat sand dog")
/// ```
pub fn countBreaks(allocator: Allocator, s: []const u8, dict: std.StringHashMap(void)) !usize {
    const n = s.len;
    if (n == 0) return 1;

    // dp[i] = number of ways to segment s[0..i]
    var dp = try allocator.alloc(usize, n + 1);
    defer allocator.free(dp);
    @memset(dp, 0);
    dp[0] = 1; // Empty string has one way

    // For each position i
    for (1..n + 1) |i| {
        // Try all possible last words ending at i
        for (0..i) |j| {
            if (dict.contains(s[j..i])) {
                dp[i] += dp[j];
            }
        }
    }

    return dp[n];
}

/// Return all possible segmentations of a string into dictionary words
///
/// Uses dynamic programming with backtracking to reconstruct all solutions.
/// Returns an ArrayList of segmentations, where each segmentation is an
/// ArrayList of word slices into the original string.
///
/// Time: O(n²×m + n×k) where n = string length, m = dictionary size, k = number of segmentations
/// Space: O(n×k) for storing all segmentations
///
/// Example:
/// ```zig
/// var dict = std.StringHashMap(void).init(allocator);
/// defer dict.deinit();
/// try dict.put("cat", {});
/// try dict.put("cats", {});
/// try dict.put("and", {});
/// try dict.put("sand", {});
/// try dict.put("dog", {});
///
/// var segmentations = try allBreaks(allocator, "catsanddog", dict);
/// defer {
///     for (segmentations.items) |seg| seg.deinit(allocator);
///     segmentations.deinit(allocator);
/// }
/// // segmentations.items.len == 2
/// // segmentations.items[0] = ["cats", "and", "dog"]
/// // segmentations.items[1] = ["cat", "sand", "dog"]
/// ```
pub fn allBreaks(allocator: Allocator, s: []const u8, dict: std.StringHashMap(void)) !std.ArrayList(std.ArrayList([]const u8)) {
    const n = s.len;
    var result = try std.ArrayList(std.ArrayList([]const u8)).initCapacity(allocator, 0);
    errdefer {
        for (result.items) |*seg| seg.deinit(allocator);
        result.deinit(allocator);
    }

    if (n == 0) {
        const empty_seg = try std.ArrayList([]const u8).initCapacity(allocator, 0);
        try result.append(allocator,empty_seg);
        return result;
    }

    // dp[i] = list of (j, word) pairs where s[j..i] is a valid word and dp[j] is true
    var dp = try allocator.alloc(std.ArrayList(usize), n + 1);
    defer {
        for (dp) |*list| list.deinit(allocator);
        allocator.free(dp);
    }
    for (dp) |*list| list.* = try std.ArrayList(usize).initCapacity(allocator, 0);

    // For each position i
    for (1..n + 1) |i| {
        // Try all possible last words ending at i
        for (0..i) |j| {
            if (dict.contains(s[j..i])) {
                // Either j == 0 (start of string) or dp[j] has solutions
                if (j == 0 or dp[j].items.len > 0) {
                    try dp[i].append(allocator, j);
                }
            }
        }
    }

    // If no valid segmentation exists, return empty result
    if (dp[n].items.len == 0) {
        return result;
    }

    // Backtrack to reconstruct all segmentations
    var current = try std.ArrayList([]const u8).initCapacity(allocator, 0);
    defer current.deinit(allocator);
    try backtrack(allocator, s, &dp, n, &current, &result);

    return result;
}

fn backtrack(
    allocator: Allocator,
    s: []const u8,
    dp: *[]std.ArrayList(usize),
    pos: usize,
    current: *std.ArrayList([]const u8),
    result: *std.ArrayList(std.ArrayList([]const u8)),
) !void {
    if (pos == 0) {
        // Found a complete segmentation, add it to result (in reverse order)
        var seg = try std.ArrayList([]const u8).initCapacity(allocator, current.items.len);
        errdefer seg.deinit(allocator);

        var i: usize = current.items.len;
        while (i > 0) {
            i -= 1;
            try seg.append(allocator, current.items[i]);
        }

        try result.append(allocator,seg);
        return;
    }

    // Try all possible previous positions
    for (dp.*[pos].items) |prev_pos| {
        try current.append(allocator, s[prev_pos..pos]);
        try backtrack(allocator, s, dp, prev_pos, current, result);
        _ = current.pop();
    }
}

// Tests
test "word break - basic true case" {
    var dict = std.StringHashMap(void).init(testing.allocator);
    defer dict.deinit();
    try dict.put("leet", {});
    try dict.put("code", {});

    const result = try canBreak(testing.allocator, "leetcode", dict);
    try testing.expect(result);
}

test "word break - basic false case" {
    var dict = std.StringHashMap(void).init(testing.allocator);
    defer dict.deinit();
    try dict.put("cats", {});
    try dict.put("dog", {});
    try dict.put("sand", {});
    try dict.put("and", {});
    try dict.put("cat", {});

    const result = try canBreak(testing.allocator, "catsandog", dict);
    try testing.expect(!result);
}

test "word break - multiple words" {
    var dict = std.StringHashMap(void).init(testing.allocator);
    defer dict.deinit();
    try dict.put("apple", {});
    try dict.put("pen", {});

    const result = try canBreak(testing.allocator, "applepenapple", dict);
    try testing.expect(result);
}

test "word break - empty string" {
    var dict = std.StringHashMap(void).init(testing.allocator);
    defer dict.deinit();

    const result = try canBreak(testing.allocator, "", dict);
    try testing.expect(result);
}

test "word break - single character" {
    var dict = std.StringHashMap(void).init(testing.allocator);
    defer dict.deinit();
    try dict.put("a", {});

    const result1 = try canBreak(testing.allocator, "a", dict);
    try testing.expect(result1);

    const result2 = try canBreak(testing.allocator, "b", dict);
    try testing.expect(!result2);
}

test "word break - overlapping words" {
    var dict = std.StringHashMap(void).init(testing.allocator);
    defer dict.deinit();
    try dict.put("cat", {});
    try dict.put("cats", {});
    try dict.put("and", {});
    try dict.put("sand", {});
    try dict.put("dog", {});

    const result = try canBreak(testing.allocator, "catsanddog", dict);
    try testing.expect(result);
}

test "word break - repeated word" {
    var dict = std.StringHashMap(void).init(testing.allocator);
    defer dict.deinit();
    try dict.put("a", {});

    const result = try canBreak(testing.allocator, "aaaaaaa", dict);
    try testing.expect(result);
}

test "word break - no dictionary match" {
    var dict = std.StringHashMap(void).init(testing.allocator);
    defer dict.deinit();
    try dict.put("hello", {});
    try dict.put("world", {});

    const result = try canBreak(testing.allocator, "helloworl", dict);
    try testing.expect(!result);
}

test "word break - count basic" {
    var dict = std.StringHashMap(void).init(testing.allocator);
    defer dict.deinit();
    try dict.put("cat", {});
    try dict.put("cats", {});
    try dict.put("and", {});
    try dict.put("sand", {});
    try dict.put("dog", {});

    const count = try countBreaks(testing.allocator, "catsanddog", dict);
    try testing.expectEqual(@as(usize, 2), count); // "cats and dog" or "cat sand dog"
}

test "word break - count single way" {
    var dict = std.StringHashMap(void).init(testing.allocator);
    defer dict.deinit();
    try dict.put("leet", {});
    try dict.put("code", {});

    const count = try countBreaks(testing.allocator, "leetcode", dict);
    try testing.expectEqual(@as(usize, 1), count);
}

test "word break - count zero ways" {
    var dict = std.StringHashMap(void).init(testing.allocator);
    defer dict.deinit();
    try dict.put("hello", {});

    const count = try countBreaks(testing.allocator, "world", dict);
    try testing.expectEqual(@as(usize, 0), count);
}

test "word break - count empty string" {
    var dict = std.StringHashMap(void).init(testing.allocator);
    defer dict.deinit();

    const count = try countBreaks(testing.allocator, "", dict);
    try testing.expectEqual(@as(usize, 1), count);
}

test "word break - count repeated word" {
    var dict = std.StringHashMap(void).init(testing.allocator);
    defer dict.deinit();
    try dict.put("a", {});

    const count = try countBreaks(testing.allocator, "aaa", dict);
    try testing.expectEqual(@as(usize, 1), count); // Only one way: "a" + "a" + "a"
}

test "word break - all breaks basic" {
    var dict = std.StringHashMap(void).init(testing.allocator);
    defer dict.deinit();
    try dict.put("cat", {});
    try dict.put("cats", {});
    try dict.put("and", {});
    try dict.put("sand", {});
    try dict.put("dog", {});

    var segmentations = try allBreaks(testing.allocator, "catsanddog", dict);
    defer {
        for (segmentations.items) |*seg| seg.deinit(testing.allocator);
        segmentations.deinit(testing.allocator);
    }

    try testing.expectEqual(@as(usize, 2), segmentations.items.len);

    // Check segmentations (order may vary)
    // One should be ["cats", "and", "dog"], the other ["cat", "sand", "dog"]
    var found_cats = false;
    var found_cat = false;

    for (segmentations.items) |seg| {
        try testing.expectEqual(@as(usize, 3), seg.items.len);
        try testing.expectEqualStrings("dog", seg.items[2]); // Both end with "dog"

        if (std.mem.eql(u8, seg.items[0], "cats")) {
            found_cats = true;
            try testing.expectEqualStrings("and", seg.items[1]);
        } else if (std.mem.eql(u8, seg.items[0], "cat")) {
            found_cat = true;
            try testing.expectEqualStrings("sand", seg.items[1]);
        }
    }

    try testing.expect(found_cats);
    try testing.expect(found_cat);
}

test "word break - all breaks single way" {
    var dict = std.StringHashMap(void).init(testing.allocator);
    defer dict.deinit();
    try dict.put("leet", {});
    try dict.put("code", {});

    var segmentations = try allBreaks(testing.allocator, "leetcode", dict);
    defer {
        for (segmentations.items) |*seg| seg.deinit(testing.allocator);
        segmentations.deinit(testing.allocator);
    }

    try testing.expectEqual(@as(usize, 1), segmentations.items.len);
    try testing.expectEqual(@as(usize, 2), segmentations.items[0].items.len);
    try testing.expectEqualStrings("leet", segmentations.items[0].items[0]);
    try testing.expectEqualStrings("code", segmentations.items[0].items[1]);
}

test "word break - all breaks zero ways" {
    var dict = std.StringHashMap(void).init(testing.allocator);
    defer dict.deinit();
    try dict.put("hello", {});

    var segmentations = try allBreaks(testing.allocator, "world", dict);
    defer {
        for (segmentations.items) |*seg| seg.deinit(testing.allocator);
        segmentations.deinit(testing.allocator);
    }

    try testing.expectEqual(@as(usize, 0), segmentations.items.len);
}

test "word break - all breaks empty string" {
    var dict = std.StringHashMap(void).init(testing.allocator);
    defer dict.deinit();

    var segmentations = try allBreaks(testing.allocator, "", dict);
    defer {
        for (segmentations.items) |*seg| seg.deinit(testing.allocator);
        segmentations.deinit(testing.allocator);
    }

    try testing.expectEqual(@as(usize, 1), segmentations.items.len);
    try testing.expectEqual(@as(usize, 0), segmentations.items[0].items.len);
}

test "word break - all breaks complex" {
    var dict = std.StringHashMap(void).init(testing.allocator);
    defer dict.deinit();
    try dict.put("pine", {});
    try dict.put("apple", {});
    try dict.put("pen", {});
    try dict.put("applepen", {});
    try dict.put("pineapple", {});

    var segmentations = try allBreaks(testing.allocator, "pineapplepenapple", dict);
    defer {
        for (segmentations.items) |*seg| seg.deinit(testing.allocator);
        segmentations.deinit(testing.allocator);
    }

    // Multiple valid segmentations:
    // 1. ["pine", "apple", "pen", "apple"]
    // 2. ["pineapple", "pen", "apple"]
    // 3. ["pine", "applepen", "apple"]
    try testing.expect(segmentations.items.len >= 3);
}

test "word break - memory safety" {
    var dict = std.StringHashMap(void).init(testing.allocator);
    defer dict.deinit();
    try dict.put("a", {});
    try dict.put("aa", {});
    try dict.put("aaa", {});

    const result = try canBreak(testing.allocator, "aaaaaaa", dict);
    try testing.expect(result);

    const count = try countBreaks(testing.allocator, "aaaa", dict);
    try testing.expect(count > 0);

    var segmentations = try allBreaks(testing.allocator, "aaa", dict);
    defer {
        for (segmentations.items) |*seg| seg.deinit(testing.allocator);
        segmentations.deinit(testing.allocator);
    }
    try testing.expect(segmentations.items.len > 0);
}
