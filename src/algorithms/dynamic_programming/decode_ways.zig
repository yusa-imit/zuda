const std = @import("std");
const Allocator = std.mem.Allocator;

/// Decode Ways - Count the number of ways to decode a digit string to letters
///
/// Problem: Given a string of digits, count how many ways it can be decoded where:
/// 'A' = "1", 'B' = "2", ..., 'Z' = "26"
///
/// Examples:
/// - "12" → 2 ways: "AB" (1,2) or "L" (12)
/// - "226" → 3 ways: "BZ" (2,26), "VF" (22,6), "BBF" (2,2,6)
/// - "06" → 0 ways: no valid decoding (leading zero)
///
/// Algorithm: Dynamic Programming
/// - dp[i] = number of ways to decode s[0..i-1]
/// - Base: dp[0] = 1 (empty string), dp[1] = 1 if s[0] != '0'
/// - Recurrence:
///   * Single digit: if s[i-1] in '1'-'9', add dp[i-1]
///   * Two digits: if s[i-2..i-1] in "10"-"26", add dp[i-2]
/// - Key constraint: '0' can only be decoded as part of "10" or "20"
///
/// Time Complexity: O(n) where n = length of string
/// Space Complexity: O(1) - space-optimized with rolling variables

/// Count the number of ways to decode a digit string.
/// Time: O(n) | Space: O(1)
///
/// Returns 0 if the string cannot be decoded (e.g., leading zero, invalid digits).
///
/// Example:
/// ```zig
/// const count = countWays("226"); // 3
/// ```
pub fn countWays(s: []const u8) u64 {
    if (s.len == 0) return 0;
    if (s[0] == '0') return 0; // Leading zero - invalid

    // dp[i] represents ways to decode s[0..i-1]
    // We only need the last two values, so use rolling variables
    var prev2: u64 = 1; // dp[0] = 1 (base case: empty string)
    var prev1: u64 = 1; // dp[1] = 1 (first character is valid)

    for (1..s.len) |i| {
        var curr: u64 = 0;

        // Single digit decode: s[i] must be '1'-'9'
        if (s[i] >= '1' and s[i] <= '9') {
            curr += prev1;
        }

        // Two digit decode: s[i-1..i] must be "10"-"26"
        const two_digit = (s[i - 1] - '0') * 10 + (s[i] - '0');
        if (two_digit >= 10 and two_digit <= 26) {
            curr += prev2;
        }

        // Update rolling variables
        prev2 = prev1;
        prev1 = curr;
    }

    return prev1;
}

/// Count ways with explicit DP table (for debugging/understanding).
/// Time: O(n) | Space: O(n)
///
/// Returns the DP table where dp[i] = ways to decode s[0..i-1].
/// Caller owns the returned memory.
///
/// Example:
/// ```zig
/// const dp = try countWaysTable(allocator, "12");
/// defer allocator.free(dp);
/// // dp = [1, 1, 2] → 2 ways
/// ```
pub fn countWaysTable(allocator: Allocator, s: []const u8) ![]u64 {
    if (s.len == 0) {
        const result = try allocator.alloc(u64, 1);
        result[0] = 0;
        return result;
    }
    if (s[0] == '0') {
        const result = try allocator.alloc(u64, s.len + 1);
        @memset(result, 0);
        return result;
    }

    const dp = try allocator.alloc(u64, s.len + 1);
    dp[0] = 1; // Empty string
    dp[1] = 1; // First character

    for (1..s.len) |i| {
        dp[i + 1] = 0;

        // Single digit
        if (s[i] >= '1' and s[i] <= '9') {
            dp[i + 1] += dp[i];
        }

        // Two digits
        const two_digit = (s[i - 1] - '0') * 10 + (s[i] - '0');
        if (two_digit >= 10 and two_digit <= 26) {
            dp[i + 1] += dp[i - 1];
        }
    }

    return dp;
}

/// Count ways with wildcard support ('*' represents any digit '1'-'9').
/// Time: O(n) | Space: O(1)
///
/// Wildcards:
/// - Single '*' can be 1-9 (9 ways)
/// - Two digit "1*" can be 10-19 (10 ways)
/// - Two digit "2*" can be 20-26 (7 ways)
/// - Two digit "*1"-"*6" can be 11/21 or 12/22 ... (2 ways each)
/// - Two digit "*7"-"*9" can only be 17/18/19 (1 way each)
/// - Two digit "**" can be 11-19 and 21-26 (15 ways)
///
/// Example:
/// ```zig
/// const count = countWaysWildcard("2*"); // 15 (20-29, but only 20-26 valid)
/// ```
pub fn countWaysWildcard(s: []const u8) u64 {
    if (s.len == 0) return 0;

    const MOD: u64 = 1_000_000_007; // Use modulo for large numbers

    var prev2: u64 = 1;
    var prev1: u64 = if (s[0] == '*') 9 else if (s[0] == '0') 0 else 1;

    for (1..s.len) |i| {
        var curr: u64 = 0;

        // Single digit decode
        if (s[i] == '*') {
            curr = (curr + prev1 * 9) % MOD; // '*' can be 1-9
        } else if (s[i] >= '1' and s[i] <= '9') {
            curr = (curr + prev1) % MOD;
        }

        // Two digit decode
        if (s[i - 1] == '*') {
            if (s[i] == '*') {
                // "**" → 11-19 (9) + 21-26 (6) = 15
                curr = (curr + prev2 * 15) % MOD;
            } else if (s[i] >= '0' and s[i] <= '6') {
                // "*0"-"*6" → 10/20, 11/21, ..., 16/26 = 2 ways each
                curr = (curr + prev2 * 2) % MOD;
            } else {
                // "*7"-"*9" → only 17, 18, 19 = 1 way each
                curr = (curr + prev2) % MOD;
            }
        } else if (s[i - 1] == '1') {
            if (s[i] == '*') {
                // "1*" → 11-19 = 9 ways
                curr = (curr + prev2 * 9) % MOD;
            } else {
                // "10"-"19"
                curr = (curr + prev2) % MOD;
            }
        } else if (s[i - 1] == '2') {
            if (s[i] == '*') {
                // "2*" → 21-26 = 6 ways (not 27-29)
                curr = (curr + prev2 * 6) % MOD;
            } else if (s[i] >= '0' and s[i] <= '6') {
                // "20"-"26"
                curr = (curr + prev2) % MOD;
            }
        }

        prev2 = prev1;
        prev1 = curr;
    }

    return prev1;
}

/// Check if a string can be decoded (has at least one valid decoding).
/// Time: O(n) | Space: O(1)
///
/// Example:
/// ```zig
/// const valid = canDecode("06"); // false (leading zero in "6")
/// ```
pub fn canDecode(s: []const u8) bool {
    return countWays(s) > 0;
}

/// Find all possible decoded strings (exponential output).
/// Time: O(2^n) worst case | Space: O(n × output)
///
/// Returns an ArrayList of all valid decoded strings.
/// Caller owns the returned list and must deinit() all strings.
///
/// Example:
/// ```zig
/// var decodings = try allDecodings(allocator, "12");
/// defer {
///     for (decodings.items) |str| allocator.free(str);
///     decodings.deinit();
/// }
/// // decodings = ["AB", "L"]
/// ```
pub fn allDecodings(allocator: Allocator, s: []const u8) !std.ArrayList([]u8) {
    var result = try std.ArrayList([]u8).initCapacity(allocator, 0);
    if (s.len == 0) return result;
    if (s[0] == '0') return result; // Invalid

    var current = try std.ArrayList(u8).initCapacity(allocator, s.len);
    defer current.deinit(allocator);

    try decodeHelper(allocator, s, 0, &current, &result);
    return result;
}

fn decodeHelper(
    allocator: Allocator,
    s: []const u8,
    idx: usize,
    current: *std.ArrayList(u8),
    result: *std.ArrayList([]u8),
) !void {
    if (idx == s.len) {
        // Found a complete decoding
        const decoded = try allocator.dupe(u8, current.items);
        try result.append(allocator, decoded);
        return;
    }

    // Single digit decode
    if (s[idx] >= '1' and s[idx] <= '9') {
        const digit = s[idx] - '0';
        const letter = 'A' + digit - 1;
        try current.append(allocator, letter);
        try decodeHelper(allocator, s, idx + 1, current, result);
        _ = current.pop();
    }

    // Two digit decode
    if (idx + 1 < s.len) {
        const two_digit = (s[idx] - '0') * 10 + (s[idx + 1] - '0');
        if (two_digit >= 10 and two_digit <= 26) {
            const letter = 'A' + @as(u8, @intCast(two_digit)) - 1;
            try current.append(allocator, letter);
            try decodeHelper(allocator, s, idx + 2, current, result);
            _ = current.pop();
        }
    }
}

// ========================================
// Tests
// ========================================

const testing = std.testing;

test "decode_ways: basic examples" {
    try testing.expectEqual(@as(u64, 0), countWays("")); // Empty
    try testing.expectEqual(@as(u64, 1), countWays("1")); // Single digit
    try testing.expectEqual(@as(u64, 2), countWays("12")); // "AB" or "L"
    try testing.expectEqual(@as(u64, 3), countWays("226")); // "BZ", "VF", "BBF"
    try testing.expectEqual(@as(u64, 1), countWays("10")); // Only "J"
    try testing.expectEqual(@as(u64, 0), countWays("06")); // Leading zero in second position
}

test "decode_ways: leading zeros" {
    try testing.expectEqual(@as(u64, 0), countWays("0")); // Invalid
    try testing.expectEqual(@as(u64, 0), countWays("01")); // Leading zero
    try testing.expectEqual(@as(u64, 0), countWays("001")); // Multiple leading zeros
}

test "decode_ways: zeros in middle" {
    try testing.expectEqual(@as(u64, 1), countWays("10")); // Valid "J"
    try testing.expectEqual(@as(u64, 1), countWays("20")); // Valid "T"
    try testing.expectEqual(@as(u64, 0), countWays("30")); // Invalid (30 > 26)
    try testing.expectEqual(@as(u64, 1), countWays("1101")); // "KA"
    try testing.expectEqual(@as(u64, 0), countWays("100")); // "J" + "0" invalid
}

test "decode_ways: boundary values" {
    try testing.expectEqual(@as(u64, 2), countWays("11")); // "AA" or "K"
    try testing.expectEqual(@as(u64, 2), countWays("19")); // "AI" or "S"
    try testing.expectEqual(@as(u64, 2), countWays("26")); // "BF" or "Z"
    try testing.expectEqual(@as(u64, 1), countWays("27")); // Only "BG" (27 > 26)
    try testing.expectEqual(@as(u64, 1), countWays("99")); // Only "II"
}

test "decode_ways: long strings" {
    // "111" → "AAA" (1,1,1), "KA" (11,1), "AK" (1,11) = 3 ways
    try testing.expectEqual(@as(u64, 3), countWays("111"));
    // "1111" → 5 ways (Fibonacci pattern)
    try testing.expectEqual(@as(u64, 5), countWays("1111"));
    // "11111" → 8 ways (Fibonacci continues)
    try testing.expectEqual(@as(u64, 8), countWays("11111"));
}

test "decode_ways: single digit patterns" {
    // "123" → "ABC" (1,2,3), "LC" (12,3), "AW" (1,23) = 3 ways
    try testing.expectEqual(@as(u64, 3), countWays("123"));
    // "9999" → only "IIII" (9,9,9,9) since 99 > 26
    try testing.expectEqual(@as(u64, 1), countWays("9999"));
}

test "decode_ways: mixed patterns" {
    // "2111" → dp[0]=1, dp[1]=1, dp[2]=2 (21 or 2,1), dp[3]=3 (21,1 or 2,11 or 2,1,1), dp[4]=5
    try testing.expectEqual(@as(u64, 5), countWays("2111"));
}

test "decode_ways: table variant" {
    const allocator = testing.allocator;

    const dp = try countWaysTable(allocator, "226");
    defer allocator.free(dp);

    try testing.expectEqual(@as(usize, 4), dp.len); // dp[0..3]
    try testing.expectEqual(@as(u64, 1), dp[0]); // Base
    try testing.expectEqual(@as(u64, 1), dp[1]); // "2"
    try testing.expectEqual(@as(u64, 2), dp[2]); // "22" or "2,2"
    try testing.expectEqual(@as(u64, 3), dp[3]); // Final count
}

test "decode_ways: table empty string" {
    const allocator = testing.allocator;
    const dp = try countWaysTable(allocator, "");
    defer allocator.free(dp);
    try testing.expectEqual(@as(u64, 0), dp[0]);
}

test "decode_ways: table leading zero" {
    const allocator = testing.allocator;
    const dp = try countWaysTable(allocator, "06");
    defer allocator.free(dp);
    try testing.expectEqual(@as(u64, 0), dp[0]);
    try testing.expectEqual(@as(u64, 0), dp[1]);
    try testing.expectEqual(@as(u64, 0), dp[2]);
}

test "decode_ways: can decode" {
    try testing.expect(canDecode("12"));
    try testing.expect(canDecode("226"));
    try testing.expect(!canDecode("06"));
    try testing.expect(!canDecode("0"));
    try testing.expect(canDecode("10"));
}

test "decode_ways: wildcard basic" {
    try testing.expectEqual(@as(u64, 9), countWaysWildcard("*")); // 1-9
    try testing.expectEqual(@as(u64, 18), countWaysWildcard("1*")); // 10-19: 1 way single + 9 ways two-digit? No...
    // Let me recalculate: "1*" at position 1:
    // Single: if * is 1-9, we have 9 ways to decode as single digit
    // Two: "1*" can be 10-19 (9 ways from previous position)
    // So for a string of length 2: prev2=1, prev1=9 (from first *)
    // At position 1: curr = 9*1 (single) + 1*9 (two) = 9+9=18. Correct!
    try testing.expectEqual(@as(u64, 15), countWaysWildcard("2*")); // 20-29, but only 20-26 valid
    // prev2=1, prev1=9, at position 1: curr = 9*1 (single) + 1*6 (two digit 20-26) = 9+6=15. Correct!
}

test "decode_ways: wildcard complex" {
    try testing.expectEqual(@as(u64, 96), countWaysWildcard("**")); // Many combinations
    // "**": prev2=1, prev1=9 (first *), at position 1:
    // curr = 9*9 (single) + 1*15 (two digit 11-19,21-26) = 81+15=96. Correct!
}

test "decode_ways: all decodings" {
    const allocator = testing.allocator;

    var decodings = try allDecodings(allocator, "12");
    defer {
        for (decodings.items) |str| allocator.free(str);
        decodings.deinit(allocator);
    }

    try testing.expectEqual(@as(usize, 2), decodings.items.len);
    // Should have "AB" and "L"
    var found_ab = false;
    var found_l = false;
    for (decodings.items) |str| {
        if (std.mem.eql(u8, str, "AB")) found_ab = true;
        if (std.mem.eql(u8, str, "L")) found_l = true;
    }
    try testing.expect(found_ab);
    try testing.expect(found_l);
}

test "decode_ways: all decodings complex" {
    const allocator = testing.allocator;

    var decodings = try allDecodings(allocator, "226");
    defer {
        for (decodings.items) |str| allocator.free(str);
        decodings.deinit(allocator);
    }

    try testing.expectEqual(@as(usize, 3), decodings.items.len);
    // Should have "BBF" (2,2,6), "BZ" (2,26), "VF" (22,6)
}

test "decode_ways: all decodings empty" {
    const allocator = testing.allocator;
    var decodings = try allDecodings(allocator, "");
    defer decodings.deinit(allocator);
    try testing.expectEqual(@as(usize, 0), decodings.items.len);
}

test "decode_ways: all decodings invalid" {
    const allocator = testing.allocator;
    var decodings = try allDecodings(allocator, "06");
    defer decodings.deinit(allocator);
    try testing.expectEqual(@as(usize, 0), decodings.items.len);
}

test "decode_ways: memory safety" {
    const allocator = testing.allocator;

    // Table variant
    const dp = try countWaysTable(allocator, "111111");
    allocator.free(dp);

    // All decodings variant
    var decodings = try allDecodings(allocator, "111");
    for (decodings.items) |str| allocator.free(str);
    decodings.deinit(allocator);
}
