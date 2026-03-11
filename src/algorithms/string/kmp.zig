const std = @import("std");
const Allocator = std.mem.Allocator;

/// KMP (Knuth-Morris-Pratt) - Efficient pattern matching algorithm
///
/// Time Complexity:
///   Preprocessing: O(m) where m = pattern length
///   Searching:     O(n) where n = text length
///   Total:         O(n + m)
///
/// Space Complexity: O(m) for failure function table
///
/// Characteristics:
/// - Linear time: Never backtracks in the text
/// - Preprocesses pattern to build failure function
/// - Failure function: longest proper prefix that is also a suffix
/// - Optimal for single pattern matching in long texts
/// - No false positives (exact matching)
///
/// Algorithm:
/// 1. Build failure function (partial match table) from pattern
/// 2. Scan text, using failure function to skip unnecessary comparisons
/// 3. When mismatch occurs, use failure function to determine next position
pub fn KMP(comptime T: type) type {
    return struct {
        allocator: Allocator,
        pattern: []const T,
        failure: []usize,

        const Self = @This();

        /// Initialize KMP matcher with pattern
        /// Time: O(m) | Space: O(m)
        pub fn init(allocator: Allocator, pattern: []const T) !Self {
            if (pattern.len == 0) return error.EmptyPattern;

            const failure = try allocator.alloc(usize, pattern.len);
            errdefer allocator.free(failure);

            computeFailure(pattern, failure);

            return .{
                .allocator = allocator,
                .pattern = pattern,
                .failure = failure,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.failure);
        }

        /// Find first occurrence of pattern in text
        /// Returns index of first match, or null if not found
        /// Time: O(n) | Space: O(1)
        pub fn find(self: *const Self, text: []const T) ?usize {
            if (text.len < self.pattern.len) return null;

            var i: usize = 0; // text index
            var j: usize = 0; // pattern index

            while (i < text.len) {
                if (text[i] == self.pattern[j]) {
                    i += 1;
                    j += 1;

                    if (j == self.pattern.len) {
                        return i - j; // Match found
                    }
                } else if (j > 0) {
                    j = self.failure[j - 1]; // Use failure function
                } else {
                    i += 1;
                }
            }

            return null;
        }

        /// Find all occurrences of pattern in text
        /// Returns ArrayList of match indices
        /// Time: O(n) | Space: O(k) where k = number of matches
        pub fn findAll(self: *const Self, text: []const T, allocator: Allocator) !std.ArrayList(usize) {
            var matches = std.ArrayList(usize).init(allocator);
            errdefer matches.deinit();

            if (text.len < self.pattern.len) return matches;

            var i: usize = 0;
            var j: usize = 0;

            while (i < text.len) {
                if (text[i] == self.pattern[j]) {
                    i += 1;
                    j += 1;

                    if (j == self.pattern.len) {
                        try matches.append(i - j);
                        j = self.failure[j - 1]; // Continue searching for overlapping matches
                    }
                } else if (j > 0) {
                    j = self.failure[j - 1];
                } else {
                    i += 1;
                }
            }

            return matches;
        }

        /// Check if pattern exists in text
        /// Time: O(n) | Space: O(1)
        pub fn contains(self: *const Self, text: []const T) bool {
            return self.find(text) != null;
        }

        /// Count occurrences of pattern in text (including overlapping)
        /// Time: O(n) | Space: O(1)
        pub fn count(self: *const Self, text: []const T) usize {
            var result: usize = 0;

            if (text.len < self.pattern.len) return 0;

            var i: usize = 0;
            var j: usize = 0;

            while (i < text.len) {
                if (text[i] == self.pattern[j]) {
                    i += 1;
                    j += 1;

                    if (j == self.pattern.len) {
                        result += 1;
                        j = self.failure[j - 1];
                    }
                } else if (j > 0) {
                    j = self.failure[j - 1];
                } else {
                    i += 1;
                }
            }

            return result;
        }

        /// Compute failure function (prefix function) for pattern
        /// failure[i] = length of longest proper prefix of pattern[0..i] that is also a suffix
        fn computeFailure(pattern: []const T, failure: []usize) void {
            failure[0] = 0;
            var len: usize = 0; // Length of previous longest prefix suffix
            var i: usize = 1;

            while (i < pattern.len) {
                if (pattern[i] == pattern[len]) {
                    len += 1;
                    failure[i] = len;
                    i += 1;
                } else {
                    if (len != 0) {
                        len = failure[len - 1];
                    } else {
                        failure[i] = 0;
                        i += 1;
                    }
                }
            }
        }
    };
}

/// Convenient string search function using KMP
/// Time: O(n + m) | Space: O(m)
pub fn search(allocator: Allocator, text: []const u8, pattern: []const u8) !?usize {
    if (pattern.len == 0) return error.EmptyPattern;
    var kmp = try KMP(u8).init(allocator, pattern);
    defer kmp.deinit();
    return kmp.find(text);
}

/// Convenient string search all function using KMP
/// Time: O(n + m) | Space: O(m + k)
pub fn searchAll(allocator: Allocator, text: []const u8, pattern: []const u8) !std.ArrayList(usize) {
    if (pattern.len == 0) return error.EmptyPattern;
    var kmp = try KMP(u8).init(allocator, pattern);
    defer kmp.deinit();
    return try kmp.findAll(text, allocator);
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "KMP: simple match" {
    var kmp = try KMP(u8).init(testing.allocator, "abc");
    defer kmp.deinit();

    const text = "xyzabcdef";
    const result = kmp.find(text);
    try testing.expect(result != null);
    try testing.expectEqual(3, result.?);
}

test "KMP: no match" {
    var kmp = try KMP(u8).init(testing.allocator, "xyz");
    defer kmp.deinit();

    const text = "abcdefgh";
    const result = kmp.find(text);
    try testing.expect(result == null);
}

test "KMP: pattern at start" {
    var kmp = try KMP(u8).init(testing.allocator, "abc");
    defer kmp.deinit();

    const text = "abcdefgh";
    const result = kmp.find(text);
    try testing.expectEqual(0, result.?);
}

test "KMP: pattern at end" {
    var kmp = try KMP(u8).init(testing.allocator, "gh");
    defer kmp.deinit();

    const text = "abcdefgh";
    const result = kmp.find(text);
    try testing.expectEqual(6, result.?);
}

test "KMP: repeating pattern" {
    var kmp = try KMP(u8).init(testing.allocator, "aa");
    defer kmp.deinit();

    const text = "aaaaaa";
    const result = kmp.find(text);
    try testing.expectEqual(0, result.?);
}

test "KMP: overlapping occurrences" {
    var kmp = try KMP(u8).init(testing.allocator, "aba");
    defer kmp.deinit();

    const text = "abababa";
    const matches = try kmp.findAll(text, testing.allocator);
    defer matches.deinit();

    try testing.expectEqual(3, matches.items.len);
    try testing.expectEqual(0, matches.items[0]); // "aba"baba
    try testing.expectEqual(2, matches.items[1]); // ab"aba"ba
    try testing.expectEqual(4, matches.items[2]); // abab"aba"
}

test "KMP: single character pattern" {
    var kmp = try KMP(u8).init(testing.allocator, "x");
    defer kmp.deinit();

    const text = "axbxcxd";
    const matches = try kmp.findAll(text, testing.allocator);
    defer matches.deinit();

    try testing.expectEqual(3, matches.items.len);
    try testing.expectEqual(1, matches.items[0]);
    try testing.expectEqual(3, matches.items[1]);
    try testing.expectEqual(5, matches.items[2]);
}

test "KMP: pattern longer than text" {
    var kmp = try KMP(u8).init(testing.allocator, "abcdefgh");
    defer kmp.deinit();

    const text = "abc";
    const result = kmp.find(text);
    try testing.expect(result == null);
}

test "KMP: pattern equals text" {
    var kmp = try KMP(u8).init(testing.allocator, "hello");
    defer kmp.deinit();

    const text = "hello";
    const result = kmp.find(text);
    try testing.expectEqual(0, result.?);
}

test "KMP: contains method" {
    var kmp = try KMP(u8).init(testing.allocator, "world");
    defer kmp.deinit();

    try testing.expect(kmp.contains("hello world!"));
    try testing.expect(!kmp.contains("hello universe!"));
}

test "KMP: count method" {
    var kmp = try KMP(u8).init(testing.allocator, "ab");
    defer kmp.deinit();

    try testing.expectEqual(0, kmp.count("xyz"));
    try testing.expectEqual(1, kmp.count("xyzab"));
    try testing.expectEqual(2, kmp.count("ababab"));
    try testing.expectEqual(3, kmp.count("abababab"));
}

test "KMP: failure function correctness" {
    var kmp = try KMP(u8).init(testing.allocator, "ababaca");
    defer kmp.deinit();

    // Expected failure function for "ababaca":
    // Index:   0 1 2 3 4 5 6
    // Pattern: a b a b a c a
    // Failure: 0 0 1 2 3 0 1
    const expected = [_]usize{ 0, 0, 1, 2, 3, 0, 1 };
    try testing.expectEqualSlices(usize, &expected, kmp.failure);
}

test "KMP: repeated pattern" {
    var kmp = try KMP(u8).init(testing.allocator, "aaa");
    defer kmp.deinit();

    const text = "aaaaaaa";
    const matches = try kmp.findAll(text, testing.allocator);
    defer matches.deinit();

    try testing.expectEqual(5, matches.items.len); // 7 - 3 + 1 = 5 overlapping matches
}

test "KMP: convenient search function" {
    const result = try search(testing.allocator, "hello world", "world");
    try testing.expectEqual(6, result.?);
}

test "KMP: convenient searchAll function" {
    const matches = try searchAll(testing.allocator, "abcabcabc", "abc");
    defer matches.deinit();

    try testing.expectEqual(3, matches.items.len);
    try testing.expectEqual(0, matches.items[0]);
    try testing.expectEqual(3, matches.items[1]);
    try testing.expectEqual(6, matches.items[2]);
}

test "KMP: empty pattern error" {
    const result = KMP(u8).init(testing.allocator, "");
    try testing.expectError(error.EmptyPattern, result);
}

test "KMP: generic types (u32)" {
    const pattern = [_]u32{ 1, 2, 3 };
    const text = [_]u32{ 0, 1, 2, 3, 4 };

    var kmp = try KMP(u32).init(testing.allocator, &pattern);
    defer kmp.deinit();

    const result = kmp.find(&text);
    try testing.expectEqual(1, result.?);
}
