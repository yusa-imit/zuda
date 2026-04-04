const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Checks if s3 is formed by interleaving s1 and s2.
/// An interleaving of two strings s1 and s2 is a string s3 that uses all
/// characters from s1 and s2 exactly once while maintaining their relative order.
///
/// Time: O(n×m) where n=len(s1), m=len(s2)
/// Space: O(n×m) for the DP table
///
/// Example:
///   s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac" → true
///   s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc" → false
pub fn isInterleave(s1: []const u8, s2: []const u8, s3: []const u8) bool {
    const n = s1.len;
    const m = s2.len;

    // Quick check: lengths must match
    if (n + m != s3.len) return false;

    // Empty strings edge cases
    if (n == 0) return std.mem.eql(u8, s2, s3);
    if (m == 0) return std.mem.eql(u8, s1, s3);

    // Use static array for reasonable string lengths (up to 100×100)
    // For production, would use allocator
    var dp: [101][101]bool = undefined;

    // Base case: empty strings can form empty string
    dp[0][0] = true;

    // Fill first row: only using s1
    for (1..n + 1) |i| {
        dp[i][0] = dp[i - 1][0] and s1[i - 1] == s3[i - 1];
    }

    // Fill first column: only using s2
    for (1..m + 1) |j| {
        dp[0][j] = dp[0][j - 1] and s2[j - 1] == s3[j - 1];
    }

    // Fill the rest of the table
    for (1..n + 1) |i| {
        for (1..m + 1) |j| {
            const k = i + j - 1; // Current position in s3
            dp[i][j] = (dp[i - 1][j] and s1[i - 1] == s3[k]) or
                (dp[i][j - 1] and s2[j - 1] == s3[k]);
        }
    }

    return dp[n][m];
}

/// Space-optimized version using rolling array.
///
/// Time: O(n×m)
/// Space: O(m) - only stores current and previous row
pub fn isInterleaveOptimized(s1: []const u8, s2: []const u8, s3: []const u8) bool {
    const n = s1.len;
    const m = s2.len;

    if (n + m != s3.len) return false;
    if (n == 0) return std.mem.eql(u8, s2, s3);
    if (m == 0) return std.mem.eql(u8, s1, s3);

    // Use only one row of size m+1
    var dp: [101]bool = undefined;

    dp[0] = true;

    // Initialize first row
    for (1..m + 1) |j| {
        dp[j] = dp[j - 1] and s2[j - 1] == s3[j - 1];
    }

    // Process each row
    for (1..n + 1) |i| {
        dp[0] = dp[0] and s1[i - 1] == s3[i - 1];
        for (1..m + 1) |j| {
            const k = i + j - 1;
            dp[j] = (dp[j] and s1[i - 1] == s3[k]) or
                (dp[j - 1] and s2[j - 1] == s3[k]);
        }
    }

    return dp[m];
}

/// Allocating version that works with arbitrary string lengths.
///
/// Time: O(n×m)
/// Space: O(n×m) allocated on heap
pub fn isInterleaveAlloc(allocator: Allocator, s1: []const u8, s2: []const u8, s3: []const u8) !bool {
    const n = s1.len;
    const m = s2.len;

    if (n + m != s3.len) return false;
    if (n == 0) return std.mem.eql(u8, s2, s3);
    if (m == 0) return std.mem.eql(u8, s1, s3);

    // Allocate 2D DP table
    const dp = try allocator.alloc([]bool, n + 1);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }

    for (dp) |*row| {
        row.* = try allocator.alloc(bool, m + 1);
        @memset(row.*, false);
    }

    dp[0][0] = true;

    for (1..n + 1) |i| {
        dp[i][0] = dp[i - 1][0] and s1[i - 1] == s3[i - 1];
    }

    for (1..m + 1) |j| {
        dp[0][j] = dp[0][j - 1] and s2[j - 1] == s3[j - 1];
    }

    for (1..n + 1) |i| {
        for (1..m + 1) |j| {
            const k = i + j - 1;
            dp[i][j] = (dp[i - 1][j] and s1[i - 1] == s3[k]) or
                (dp[i][j - 1] and s2[j - 1] == s3[k]);
        }
    }

    return dp[n][m];
}

/// Returns one valid interleaving path if it exists.
/// Path is represented as a string of 's1' and 's2' characters indicating
/// which source string each character came from.
///
/// Time: O(n×m)
/// Space: O(n×m) for DP table + O(n+m) for path
pub fn getInterleavePath(allocator: Allocator, s1: []const u8, s2: []const u8, s3: []const u8) !?[]const u8 {
    const n = s1.len;
    const m = s2.len;

    if (n + m != s3.len) return null;

    const dp = try allocator.alloc([]bool, n + 1);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }

    for (dp) |*row| {
        row.* = try allocator.alloc(bool, m + 1);
        @memset(row.*, false);
    }

    dp[0][0] = true;

    for (1..n + 1) |i| {
        dp[i][0] = dp[i - 1][0] and s1[i - 1] == s3[i - 1];
    }

    for (1..m + 1) |j| {
        dp[0][j] = dp[0][j - 1] and s2[j - 1] == s3[j - 1];
    }

    for (1..n + 1) |i| {
        for (1..m + 1) |j| {
            const k = i + j - 1;
            dp[i][j] = (dp[i - 1][j] and s1[i - 1] == s3[k]) or
                (dp[i][j - 1] and s2[j - 1] == s3[k]);
        }
    }

    if (!dp[n][m]) return null;

    // Backtrack to build path
    var path = try allocator.alloc(u8, n + m);
    var i = n;
    var j = m;
    var pos = n + m;

    while (i > 0 or j > 0) {
        pos -= 1;
        const k = i + j - 1;

        if (i > 0 and dp[i - 1][j] and s1[i - 1] == s3[k]) {
            path[pos] = '1'; // Took from s1
            i -= 1;
        } else {
            path[pos] = '2'; // Took from s2
            j -= 1;
        }
    }

    return path;
}

/// Counts the number of distinct ways to interleave s1 and s2 to form s3.
///
/// Time: O(n×m)
/// Space: O(n×m)
pub fn countInterleaveWays(s1: []const u8, s2: []const u8, s3: []const u8) usize {
    const n = s1.len;
    const m = s2.len;

    if (n + m != s3.len) return 0;

    var dp: [101][101]usize = undefined;
    for (&dp) |*row| {
        @memset(row, 0);
    }

    dp[0][0] = 1;

    for (1..n + 1) |i| {
        if (s1[i - 1] == s3[i - 1]) {
            dp[i][0] = dp[i - 1][0];
        }
    }

    for (1..m + 1) |j| {
        if (s2[j - 1] == s3[j - 1]) {
            dp[0][j] = dp[0][j - 1];
        }
    }

    for (1..n + 1) |i| {
        for (1..m + 1) |j| {
            const k = i + j - 1;
            var ways: usize = 0;
            if (s1[i - 1] == s3[k]) {
                ways += dp[i - 1][j];
            }
            if (s2[j - 1] == s3[k]) {
                ways += dp[i][j - 1];
            }
            dp[i][j] = ways;
        }
    }

    return dp[n][m];
}

// ============================================================================
// Tests
// ============================================================================

test "interleaving string - basic valid case" {
    const s1 = "aabcc";
    const s2 = "dbbca";
    const s3 = "aadbbcbcac";
    try testing.expect(isInterleave(s1, s2, s3));
    try testing.expect(isInterleaveOptimized(s1, s2, s3));
}

test "interleaving string - basic invalid case" {
    const s1 = "aabcc";
    const s2 = "dbbca";
    const s3 = "aadbbbaccc";
    try testing.expect(!isInterleave(s1, s2, s3));
    try testing.expect(!isInterleaveOptimized(s1, s2, s3));
}

test "interleaving string - empty strings" {
    try testing.expect(isInterleave("", "", ""));
    try testing.expect(isInterleave("a", "", "a"));
    try testing.expect(isInterleave("", "b", "b"));
    try testing.expect(!isInterleave("a", "", "b"));
}

test "interleaving string - single characters" {
    try testing.expect(isInterleave("a", "b", "ab"));
    try testing.expect(isInterleave("a", "b", "ba"));
    try testing.expect(!isInterleave("a", "b", "aa"));
    try testing.expect(!isInterleave("a", "b", "bb"));
}

test "interleaving string - same characters" {
    try testing.expect(isInterleave("aa", "aa", "aaaa"));
    try testing.expect(isInterleave("aa", "aa", "aaa") == false); // Length mismatch
}

test "interleaving string - order matters" {
    try testing.expect(isInterleave("abc", "def", "adbecf"));
    try testing.expect(isInterleave("abc", "def", "abdecf"));
    try testing.expect(isInterleave("abc", "def", "defabc")); // Valid: all of s2, then all of s1
    try testing.expect(!isInterleave("abc", "def", "abcfed")); // Invalid: reverses s2 order (fed)
}

test "interleaving string - repeated patterns" {
    try testing.expect(isInterleave("ab", "ab", "abab")); // a(s1),b(s2),a(s2),b(s1) or others
    try testing.expect(isInterleave("ab", "ab", "aabb")); // a(s1),a(s2),b(s1),b(s2)
    try testing.expect(!isInterleave("ab", "ab", "abba")); // Invalid: would require s2 to give 'b' before 'a'
    try testing.expect(!isInterleave("ab", "cd", "abdc")); // Invalid: 'd' before 'c'
}

test "interleaving string - length mismatch" {
    try testing.expect(!isInterleave("a", "b", "abc"));
    try testing.expect(!isInterleave("abc", "def", "ab"));
}

test "interleaving string - all from one string" {
    try testing.expect(isInterleave("abc", "", "abc"));
    try testing.expect(isInterleave("", "def", "def"));
}

test "interleaving string - complex pattern" {
    const s1 = "db";
    const s2 = "b";
    const s3 = "cbb";
    try testing.expect(!isInterleave(s1, s2, s3));
}

test "interleaving string - allocating version" {
    const allocator = testing.allocator;
    const s1 = "aabcc";
    const s2 = "dbbca";
    const s3 = "aadbbcbcac";
    try testing.expect(try isInterleaveAlloc(allocator, s1, s2, s3));

    const s4 = "aadbbbaccc";
    try testing.expect(!try isInterleaveAlloc(allocator, s1, s2, s4));
}

test "interleaving string - get path" {
    const allocator = testing.allocator;
    const s1 = "aabcc";
    const s2 = "dbbca";
    const s3 = "aadbbcbcac";

    const path = try getInterleavePath(allocator, s1, s2, s3);
    try testing.expect(path != null);
    defer allocator.free(path.?);

    // Verify path is valid (10 characters: 5 from s1, 5 from s2)
    try testing.expectEqual(@as(usize, 10), path.?.len);

    var count1: usize = 0;
    var count2: usize = 0;
    for (path.?) |c| {
        if (c == '1') count1 += 1 else count2 += 1;
    }
    try testing.expectEqual(@as(usize, 5), count1);
    try testing.expectEqual(@as(usize, 5), count2);
}

test "interleaving string - get path for invalid" {
    const allocator = testing.allocator;
    const s1 = "aabcc";
    const s2 = "dbbca";
    const s3 = "aadbbbaccc";

    const path = try getInterleavePath(allocator, s1, s2, s3);
    try testing.expect(path == null);
}

test "interleaving string - count ways" {
    const s1 = "ab";
    const s2 = "cd";
    const s3 = "abcd";

    // Only one way: take all from s1 first, then all from s2
    try testing.expectEqual(@as(usize, 1), countInterleaveWays(s1, s2, s3));
}

test "interleaving string - count ways multiple" {
    const s1 = "a";
    const s2 = "b";
    const s3 = "ab";

    // Two ways: "a" then "b" OR both at same position
    // Actually just 1 way for "ab": must be s1[0]=a, s2[0]=b
    try testing.expectEqual(@as(usize, 1), countInterleaveWays(s1, s2, s3));
}

test "interleaving string - count ways with repetitions" {
    const s1 = "aa";
    const s2 = "aa";
    const s3 = "aaaa";

    // Multiple ways to interleave identical characters
    // C(4,2) = 6 ways to choose positions for s1's two 'a's
    try testing.expectEqual(@as(usize, 6), countInterleaveWays(s1, s2, s3));
}

test "interleaving string - count ways zero" {
    const s1 = "a";
    const s2 = "b";
    const s3 = "ba";

    // Only 1 way: s2[0]=b, then s1[0]=a
    try testing.expectEqual(@as(usize, 1), countInterleaveWays(s1, s2, s3));
}

test "interleaving string - optimized vs standard consistency" {
    const s1 = "abc";
    const s2 = "def";
    const s3 = "adbecf";

    const result1 = isInterleave(s1, s2, s3);
    const result2 = isInterleaveOptimized(s1, s2, s3);
    try testing.expectEqual(result1, result2);
}

test "interleaving string - large input" {
    const s1 = "a" ** 50;
    const s2 = "b" ** 50;
    const s3 = "ab" ** 50;

    try testing.expect(isInterleave(s1, s2, s3));
    try testing.expect(isInterleaveOptimized(s1, s2, s3));
}

test "interleaving string - memory safety" {
    const allocator = testing.allocator;
    const s1 = "test";
    const s2 = "case";
    const s3 = "tecasste";

    _ = try isInterleaveAlloc(allocator, s1, s2, s3);
    const path = try getInterleavePath(allocator, s1, s2, s3);
    if (path) |p| allocator.free(p);
}
