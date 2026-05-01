const std = @import("std");
const Allocator = std.mem.Allocator;

/// Boyer-Moore - Fast pattern matching with right-to-left scanning
///
/// Time Complexity:
///   Preprocessing: O(m + σ) where m = pattern length, σ = alphabet size
///   Searching:     O(n * m) worst case, O(n / m) best case
///   Average case:  Sublinear, often faster than KMP in practice
///
/// Space Complexity: O(m + σ)
///
/// Characteristics:
/// - Right-to-left scanning of pattern (unlike most algorithms)
/// - Bad character rule: skip based on rightmost occurrence
/// - Good suffix rule: skip based on suffix matches
/// - Sublinear average case: can skip multiple characters
/// - Best for large alphabets and long patterns
///
/// Algorithm:
/// 1. Preprocess pattern to build:
///    - Bad character table (rightmost occurrence of each character)
///    - Good suffix table (suffix shift amounts)
/// 2. Scan text from left to right, compare pattern right to left
/// 3. On mismatch, use max shift from bad char and good suffix rules
pub fn BoyerMoore(comptime T: type) type {
    return struct {
        allocator: Allocator,
        pattern: []const T,
        bad_char: std.AutoHashMap(T, usize),
        good_suffix: []usize,

        const Self = @This();

        /// Initialize Boyer-Moore matcher with pattern
        /// Time: O(m + σ) | Space: O(m + σ)
        pub fn init(allocator: Allocator, pattern: []const T) !Self {
            if (pattern.len == 0) return error.EmptyPattern;

            var bad_char = std.AutoHashMap(T, usize).init(allocator);
            errdefer bad_char.deinit();

            // Build bad character table
            for (pattern, 0..) |c, i| {
                try bad_char.put(c, i);
            }

            // Build good suffix table
            const good_suffix = try allocator.alloc(usize, pattern.len);
            errdefer allocator.free(good_suffix);

            try computeGoodSuffix(allocator, pattern, good_suffix);

            return .{
                .allocator = allocator,
                .pattern = pattern,
                .bad_char = bad_char,
                .good_suffix = good_suffix,
            };
        }

        pub fn deinit(self: *Self) void {
            self.bad_char.deinit();
            self.allocator.free(self.good_suffix);
        }

        /// Find first occurrence of pattern in text
        /// Time: O(n * m) worst, O(n / m) best | Space: O(1)
        pub fn find(self: *const Self, text: []const T) ?usize {
            if (text.len < self.pattern.len) return null;

            var s: usize = 0; // Shift of pattern relative to text

            while (s <= text.len - self.pattern.len) {
                var j: usize = self.pattern.len;

                // Scan pattern from right to left
                while (j > 0 and self.pattern[j - 1] == text[s + j - 1]) {
                    j -= 1;
                }

                if (j == 0) {
                    return s; // Match found
                }

                // Compute shift using bad character and good suffix rules
                const bad_char_shift = self.getBadCharShift(text[s + j - 1], j - 1);
                const good_suffix_shift = self.good_suffix[j - 1];
                s += @max(bad_char_shift, good_suffix_shift);
            }

            return null;
        }

        /// Find all occurrences of pattern in text
        /// Time: O(n * m) worst | Space: O(k)
        pub fn findAll(self: *const Self, text: []const T, allocator: Allocator) !std.ArrayList(usize) {
            var matches: std.ArrayList(usize) = .{};
            errdefer matches.deinit(allocator);

            if (text.len < self.pattern.len) return matches;

            var s: usize = 0;

            while (s <= text.len - self.pattern.len) {
                var j: usize = self.pattern.len;

                while (j > 0 and self.pattern[j - 1] == text[s + j - 1]) {
                    j -= 1;
                }

                if (j == 0) {
                    try matches.append(allocator, s);
                    s += if (self.pattern.len > 1) self.good_suffix[0] else 1;
                } else {
                    const bad_char_shift = self.getBadCharShift(text[s + j - 1], j - 1);
                    const good_suffix_shift = self.good_suffix[j - 1];
                    s += @max(bad_char_shift, good_suffix_shift);
                }
            }

            return matches;
        }

        /// Check if pattern exists in text
        /// Time: O(n * m) worst, O(n / m) best | Space: O(1)
        pub fn contains(self: *const Self, text: []const T) bool {
            return self.find(text) != null;
        }

        /// Count occurrences of pattern in text
        /// Time: O(n * m) worst | Space: O(1)
        pub fn count(self: *const Self, text: []const T) usize {
            var result: usize = 0;

            if (text.len < self.pattern.len) return 0;

            var s: usize = 0;

            while (s <= text.len - self.pattern.len) {
                var j: usize = self.pattern.len;

                while (j > 0 and self.pattern[j - 1] == text[s + j - 1]) {
                    j -= 1;
                }

                if (j == 0) {
                    result += 1;
                    s += if (self.pattern.len > 1) self.good_suffix[0] else 1;
                } else {
                    const bad_char_shift = self.getBadCharShift(text[s + j - 1], j - 1);
                    const good_suffix_shift = self.good_suffix[j - 1];
                    s += @max(bad_char_shift, good_suffix_shift);
                }
            }

            return result;
        }

        /// Compute shift based on bad character rule
        fn getBadCharShift(self: *const Self, c: T, pos: usize) usize {
            const last_occurrence = self.bad_char.get(c);
            if (last_occurrence) |occ| {
                if (occ < pos) {
                    return pos - occ;
                }
                return 1;
            }
            return pos + 1;
        }

        /// Compute good suffix table
        fn computeGoodSuffix(allocator: Allocator, pattern: []const T, good_suffix: []usize) !void {
            const m = pattern.len;

            // Initialize with pattern length (no match)
            for (good_suffix) |*shift| {
                shift.* = m;
            }

            // Compute shifts for each position
            var i: usize = m;
            var j: usize = m + 1;
            var border: std.ArrayList(usize) = .{};
            defer border.deinit(allocator);
            try border.appendNTimes(allocator, 0, m + 1);

            border.items[i - 1] = j;

            while (i > 0) {
                i -= 1;

                while (j <= m and pattern[i] != pattern[j - 1]) {
                    if (good_suffix[j - 1] == m) {
                        good_suffix[j - 1] = j - i - 1;
                    }
                    j = border.items[j - 1];
                }

                j -= 1;
                border.items[i] = j;
            }

            // Case 2: suffix matches prefix
            j = border.items[0];
            for (good_suffix) |*shift| {
                if (shift.* == m) {
                    shift.* = j;
                }
                if (i == j) {
                    j = border.items[j];
                }
                i += 1;
            }
        }
    };
}

/// Convenient string search function using Boyer-Moore
/// Time: O(n + m) average | Space: O(m + σ)
pub fn search(allocator: Allocator, text: []const u8, pattern: []const u8) !?usize {
    if (pattern.len == 0) return error.EmptyPattern;
    var bm = try BoyerMoore(u8).init(allocator, pattern);
    defer bm.deinit();
    return bm.find(text);
}

/// Convenient string search all function using Boyer-Moore
/// Time: O(n + m) average | Space: O(m + σ + k)
pub fn searchAll(allocator: Allocator, text: []const u8, pattern: []const u8) !std.ArrayList(usize) {
    if (pattern.len == 0) return error.EmptyPattern;
    var bm = try BoyerMoore(u8).init(allocator, pattern);
    defer bm.deinit();
    return try bm.findAll(text, allocator);
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "BoyerMoore: simple match" {
    var bm = try BoyerMoore(u8).init(testing.allocator, "abc");
    defer bm.deinit();

    const text = "xyzabcdef";
    const result = bm.find(text);
    try testing.expect(result != null);
    try testing.expectEqual(3, result.?);
}

test "BoyerMoore: no match" {
    var bm = try BoyerMoore(u8).init(testing.allocator, "xyz");
    defer bm.deinit();

    const text = "abcdefgh";
    const result = bm.find(text);
    try testing.expect(result == null);
}

test "BoyerMoore: pattern at start" {
    var bm = try BoyerMoore(u8).init(testing.allocator, "abc");
    defer bm.deinit();

    const text = "abcdefgh";
    const result = bm.find(text);
    try testing.expectEqual(0, result.?);
}

test "BoyerMoore: pattern at end" {
    var bm = try BoyerMoore(u8).init(testing.allocator, "gh");
    defer bm.deinit();

    const text = "abcdefgh";
    const result = bm.find(text);
    try testing.expectEqual(6, result.?);
}

test "BoyerMoore: repeating pattern" {
    var bm = try BoyerMoore(u8).init(testing.allocator, "aa");
    defer bm.deinit();

    const text = "aaaaaa";
    const result = bm.find(text);
    try testing.expectEqual(0, result.?);
}

test "BoyerMoore: multiple occurrences" {
    var bm = try BoyerMoore(u8).init(testing.allocator, "ab");
    defer bm.deinit();

    const text = "ababab";
    var matches = try bm.findAll(text, testing.allocator);
    defer matches.deinit(testing.allocator);

    try testing.expectEqual(3, matches.items.len);
    try testing.expectEqual(0, matches.items[0]);
    try testing.expectEqual(2, matches.items[1]);
    try testing.expectEqual(4, matches.items[2]);
}

test "BoyerMoore: single character pattern" {
    var bm = try BoyerMoore(u8).init(testing.allocator, "x");
    defer bm.deinit();

    const text = "axbxcxd";
    var matches = try bm.findAll(text, testing.allocator);
    defer matches.deinit(testing.allocator);

    try testing.expectEqual(3, matches.items.len);
    try testing.expectEqual(1, matches.items[0]);
    try testing.expectEqual(3, matches.items[1]);
    try testing.expectEqual(5, matches.items[2]);
}

test "BoyerMoore: pattern longer than text" {
    var bm = try BoyerMoore(u8).init(testing.allocator, "abcdefgh");
    defer bm.deinit();

    const text = "abc";
    const result = bm.find(text);
    try testing.expect(result == null);
}

test "BoyerMoore: pattern equals text" {
    var bm = try BoyerMoore(u8).init(testing.allocator, "hello");
    defer bm.deinit();

    const text = "hello";
    const result = bm.find(text);
    try testing.expectEqual(0, result.?);
}

test "BoyerMoore: contains method" {
    var bm = try BoyerMoore(u8).init(testing.allocator, "world");
    defer bm.deinit();

    try testing.expect(bm.contains("hello world!"));
    try testing.expect(!bm.contains("hello universe!"));
}

test "BoyerMoore: count method" {
    var bm = try BoyerMoore(u8).init(testing.allocator, "ab");
    defer bm.deinit();

    try testing.expectEqual(0, bm.count("xyz"));
    try testing.expectEqual(1, bm.count("xyzab"));
    try testing.expectEqual(3, bm.count("ababab"));
}

test "BoyerMoore: long pattern" {
    var bm = try BoyerMoore(u8).init(testing.allocator, "pattern");
    defer bm.deinit();

    const text = "this is a test pattern for matching";
    const result = bm.find(text);
    try testing.expectEqual(15, result.?);
}

test "BoyerMoore: convenient search function" {
    const result = try search(testing.allocator, "hello world", "world");
    try testing.expectEqual(6, result.?);
}

test "BoyerMoore: convenient searchAll function" {
    var matches = try searchAll(testing.allocator, "abcabcabc", "abc");
    defer matches.deinit(testing.allocator);

    try testing.expectEqual(3, matches.items.len);
    try testing.expectEqual(0, matches.items[0]);
    try testing.expectEqual(3, matches.items[1]);
    try testing.expectEqual(6, matches.items[2]);
}

test "BoyerMoore: empty pattern error" {
    const result = BoyerMoore(u8).init(testing.allocator, "");
    try testing.expectError(error.EmptyPattern, result);
}

test "BoyerMoore: case sensitivity" {
    var bm = try BoyerMoore(u8).init(testing.allocator, "Hello");
    defer bm.deinit();

    try testing.expect(!bm.contains("hello world"));
    try testing.expect(bm.contains("Hello world"));
}

test "BoyerMoore: bad character skip" {
    var bm = try BoyerMoore(u8).init(testing.allocator, "example");
    defer bm.deinit();

    // Pattern "example" in "this is an example text"
    // Should skip efficiently using bad character rule
    const text = "this is an example text";
    const result = bm.find(text);
    try testing.expectEqual(11, result.?);
}
