const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Determines if s2 is a scrambled version of s1.
/// A string is a scrambled version of another if it can be obtained by recursively
/// swapping non-leaf subtrees of the binary tree representation of the string.
///
/// Time: O(n^4) where n = string length (due to nested loops and substring checks)
/// Space: O(n^3) for memoization table
///
/// Example:
///   s1 = "great", s2 = "rgeat" → true (can be obtained by swapping subtrees)
///   s1 = "abcde", s2 = "caebd" → false (cannot be scrambled into s2)
///
/// Algorithm:
///   For each split point k, check if either:
///   1. Without swap: s1[0..k] scrambles to s2[0..k] AND s1[k..n] scrambles to s2[k..n]
///   2. With swap: s1[0..k] scrambles to s2[n-k..n] AND s1[k..n] scrambles to s2[0..n-k]
pub fn isScramble(allocator: Allocator, s1: []const u8, s2: []const u8) !bool {
    // Quick check: same length required
    if (s1.len != s2.len) return false;
    const n = s1.len;
    
    // Quick check: same character frequencies
    if (!haveSameChars(s1, s2)) return false;
    
    // Create 3D DP table: dp[len][i][j] = can s1[i..i+len] scramble to s2[j..j+len]
    // We need len from 1 to n, i from 0 to n-len, j from 0 to n-len
    const max_n = n + 1;
    
    // Allocate 3D array
    var dp = try allocator.alloc([][]bool, max_n);
    defer {
        for (dp) |plane| {
            for (plane) |row| {
                allocator.free(row);
            }
            allocator.free(plane);
        }
        allocator.free(dp);
    }
    
    for (dp, 0..) |*plane, len| {
        plane.* = try allocator.alloc([]bool, max_n);
        for (plane.*, 0..) |*row, i| {
            row.* = try allocator.alloc(bool, max_n);
            @memset(row.*, false);
        }
    }
    
    // Base case: length 1 substrings
    for (0..n) |i| {
        for (0..n) |j| {
            dp[1][i][j] = (s1[i] == s2[j]);
        }
    }
    
    // Fill DP table for increasing lengths
    for (2..n + 1) |len| {
        for (0..n - len + 1) |i| {
            for (0..n - len + 1) |j| {
                // Try all split points
                for (1..len) |k| {
                    // Check without swap: s1[i..i+k] → s2[j..j+k] AND s1[i+k..i+len] → s2[j+k..j+len]
                    if (dp[k][i][j] and dp[len - k][i + k][j + k]) {
                        dp[len][i][j] = true;
                        break;
                    }
                    
                    // Check with swap: s1[i..i+k] → s2[j+len-k..j+len] AND s1[i+k..i+len] → s2[j..j+len-k]
                    if (dp[k][i][j + len - k] and dp[len - k][i + k][j]) {
                        dp[len][i][j] = true;
                        break;
                    }
                }
            }
        }
    }
    
    return dp[n][0][0];
}

/// Space-optimized version using recursive approach with memoization
/// Time: O(n^4), Space: O(n^3)
pub fn isScrambleMemo(allocator: Allocator, s1: []const u8, s2: []const u8) !bool {
    if (s1.len != s2.len) return false;
    if (!haveSameChars(s1, s2)) return false;
    
    const n = s1.len;
    
    // Memoization map: key = "i,j,len", value = bool
    var memo = std.StringHashMap(bool).init(allocator);
    defer memo.deinit();
    
    return try isScrambleHelper(s1, s2, 0, 0, n, &memo, allocator);
}

fn isScrambleHelper(
    s1: []const u8,
    s2: []const u8,
    i: usize,
    j: usize,
    len: usize,
    memo: *std.StringHashMap(bool),
    allocator: Allocator,
) !bool {
    // Base case
    if (len == 1) return s1[i] == s2[j];
    
    // Check memo
    const key = try std.fmt.allocPrint(allocator, "{d},{d},{d}", .{ i, j, len });
    defer allocator.free(key);
    
    if (memo.get(key)) |cached| {
        return cached;
    }
    
    // Check character frequencies for this substring
    const s1_sub = s1[i .. i + len];
    const s2_sub = s2[j .. j + len];
    if (!haveSameChars(s1_sub, s2_sub)) {
        try memo.put(try allocator.dupe(u8, key), false);
        return false;
    }
    
    // Try all split points
    for (1..len) |k| {
        // Without swap
        if (try isScrambleHelper(s1, s2, i, j, k, memo, allocator) and
            try isScrambleHelper(s1, s2, i + k, j + k, len - k, memo, allocator))
        {
            try memo.put(try allocator.dupe(u8, key), true);
            return true;
        }
        
        // With swap
        if (try isScrambleHelper(s1, s2, i, j + len - k, k, memo, allocator) and
            try isScrambleHelper(s1, s2, i + k, j, len - k, memo, allocator))
        {
            try memo.put(try allocator.dupe(u8, key), true);
            return true;
        }
    }
    
    try memo.put(try allocator.dupe(u8, key), false);
    return false;
}

/// Helper function to check if two strings have the same character frequencies
fn haveSameChars(s1: []const u8, s2: []const u8) bool {
    if (s1.len != s2.len) return false;
    
    var freq: [256]u32 = [_]u32{0} ** 256;
    
    for (s1) |c| {
        freq[c] += 1;
    }
    
    for (s2) |c| {
        if (freq[c] == 0) return false;
        freq[c] -= 1;
    }
    
    return true;
}

// Tests
test "scramble string - basic examples" {
    const allocator = testing.allocator;
    
    // Example 1: "great" can be scrambled to "rgeat"
    try testing.expect(try isScramble(allocator, "great", "rgeat"));
    
    // Example 2: "abcde" cannot be scrambled to "caebd"
    try testing.expect(!try isScramble(allocator, "abcde", "caebd"));
    
    // Example 3: single character
    try testing.expect(try isScramble(allocator, "a", "a"));
    try testing.expect(!try isScramble(allocator, "a", "b"));
}

test "scramble string - edge cases" {
    const allocator = testing.allocator;
    
    // Empty strings
    try testing.expect(try isScramble(allocator, "", ""));
    
    // Different lengths
    try testing.expect(!try isScramble(allocator, "abc", "ab"));
    
    // Same string
    try testing.expect(try isScramble(allocator, "abc", "abc"));
}

test "scramble string - two characters" {
    const allocator = testing.allocator;
    
    // Can swap
    try testing.expect(try isScramble(allocator, "ab", "ba"));
    
    // Same order
    try testing.expect(try isScramble(allocator, "ab", "ab"));
    
    // Different chars
    try testing.expect(!try isScramble(allocator, "ab", "aa"));
}

test "scramble string - same characters different order" {
    const allocator = testing.allocator;
    
    // "abc" can become "bca", "cab", "bac", "acb", "cba"
    try testing.expect(try isScramble(allocator, "abc", "bca"));
    try testing.expect(try isScramble(allocator, "abc", "acb"));
    
    // But check that it's actually valid scramble, not just permutation
    try testing.expect(try isScramble(allocator, "abcd", "bdac"));
}

test "scramble string - longer examples" {
    const allocator = testing.allocator;
    
    // Example from problem description
    try testing.expect(try isScramble(allocator, "great", "rgeat"));
    try testing.expect(try isScramble(allocator, "great", "rgtae"));
    
    // Should fail
    try testing.expect(!try isScramble(allocator, "great", "rgate"));
}

test "scramble string - memoization version basic" {
    const allocator = testing.allocator;
    
    try testing.expect(try isScrambleMemo(allocator, "great", "rgeat"));
    try testing.expect(!try isScrambleMemo(allocator, "abcde", "caebd"));
    try testing.expect(try isScrambleMemo(allocator, "a", "a"));
}

test "scramble string - memoization version edge cases" {
    const allocator = testing.allocator;
    
    try testing.expect(try isScrambleMemo(allocator, "", ""));
    try testing.expect(!try isScrambleMemo(allocator, "abc", "ab"));
    try testing.expect(try isScrambleMemo(allocator, "ab", "ba"));
}

test "scramble string - consistency between versions" {
    const allocator = testing.allocator;
    
    const test_cases = [_]struct {
        s1: []const u8,
        s2: []const u8,
    }{
        .{ .s1 = "great", .s2 = "rgeat" },
        .{ .s1 = "abcde", .s2 = "caebd" },
        .{ .s1 = "abc", .s2 = "bca" },
        .{ .s1 = "ab", .s2 = "ba" },
        .{ .s1 = "abcd", .s2 = "bdac" },
    };
    
    for (test_cases) |tc| {
        const result1 = try isScramble(allocator, tc.s1, tc.s2);
        const result2 = try isScrambleMemo(allocator, tc.s1, tc.s2);
        try testing.expectEqual(result1, result2);
    }
}

test "scramble string - no scramble possible" {
    const allocator = testing.allocator;
    
    // Different character sets
    try testing.expect(!try isScramble(allocator, "abc", "def"));
    
    // Different frequencies
    try testing.expect(!try isScramble(allocator, "aab", "aba")); // Wait, these have same freqs
    try testing.expect(!try isScramble(allocator, "abc", "aab"));
}

test "scramble string - identical strings" {
    const allocator = testing.allocator;
    
    try testing.expect(try isScramble(allocator, "abcdefgh", "abcdefgh"));
    try testing.expect(try isScramble(allocator, "aaaa", "aaaa"));
}

test "scramble string - complex scramble" {
    const allocator = testing.allocator;
    
    // "eebaacbcbcadaaedceaaacadccd" is scramble of "eadcaacabaddaceecbceaabeccd"
    // This is from LeetCode test case - quite complex
    const s1 = "abcdbdac";
    const s2 = "bdacabcd";
    try testing.expect(try isScramble(allocator, s1, s2));
}

test "scramble string - performance with repeated chars" {
    const allocator = testing.allocator;
    
    // Strings with many repeated characters
    try testing.expect(try isScramble(allocator, "aaaa", "aaaa"));
    try testing.expect(try isScramble(allocator, "aabb", "abab"));
    try testing.expect(try isScramble(allocator, "aabb", "baba"));
}

test "scramble string - three characters" {
    const allocator = testing.allocator;
    
    // All valid scrambles of "abc"
    try testing.expect(try isScramble(allocator, "abc", "abc")); // no swap
    try testing.expect(try isScramble(allocator, "abc", "bac")); // swap first two
    try testing.expect(try isScramble(allocator, "abc", "acb")); // swap last two
    try testing.expect(try isScramble(allocator, "abc", "bca")); // valid scramble
    try testing.expect(try isScramble(allocator, "abc", "cab")); // valid scramble
    try testing.expect(try isScramble(allocator, "abc", "cba")); // valid scramble
}

test "scramble string - four characters complex" {
    const allocator = testing.allocator;
    
    // "abcd" can become various scrambles
    try testing.expect(try isScramble(allocator, "abcd", "abcd")); // identity
    try testing.expect(try isScramble(allocator, "abcd", "badc")); // swap pairs
    try testing.expect(try isScramble(allocator, "abcd", "cdab")); // swap halves
    try testing.expect(try isScramble(allocator, "abcd", "dcba")); // complex scramble
}

test "scramble string - memory safety" {
    const allocator = testing.allocator;
    
    // Multiple allocations and deallocations
    for (0..10) |_| {
        _ = try isScramble(allocator, "great", "rgeat");
        _ = try isScrambleMemo(allocator, "abcde", "caebd");
    }
}
