const std = @import("std");
const Allocator = std.mem.Allocator;

/// Z-algorithm - Linear time pattern matching using Z-array
///
/// Time Complexity:
///   Z-array construction: O(n) where n = string length
///   Pattern matching:     O(n + m) where m = pattern length
///   Total:                O(n + m)
///
/// Space Complexity: O(n + m) for the combined string
///
/// Characteristics:
/// - Linear time: Single pass through string with no backtracking
/// - Z-array: Z[i] = length of longest substring starting at i that matches prefix
/// - Efficient: Uses previously computed values to avoid redundant comparisons
/// - Pattern matching: Concatenate pattern$text and find Z[i] == |pattern|
/// - Simple: No failure function or state machine required
///
/// Algorithm:
/// 1. For pattern matching: S = pattern + sentinel + text (sentinel not in alphabet)
/// 2. Compute Z-array where Z[i] is longest substring starting at i matching prefix
/// 3. Use [L, R] interval: rightmost segment that matches prefix
/// 4. For each position, use previously computed Z values within [L, R] to skip work
/// 5. Pattern found at position i if Z[i] == pattern length
///
/// Example:
///   text = "ababcababa"
///   pattern = "aba"
///   S = "aba$ababcababa"
///   Z = [14, 0, 1, 0, 3, 0, 1, 0, 3, 0, 1, 0, 3, 0]
///   matches at positions: 0 (Z[4]=3), 5 (Z[8]=3), 7 (Z[12]=3) → indices 0, 5, 7 in text
pub fn ZAlgorithm(comptime T: type) type {
    return struct {
        allocator: Allocator,

        const Self = @This();

        pub const Match = struct {
            index: usize,
        };

        /// Initialize Z-algorithm instance
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator) Self {
            return .{
                .allocator = allocator,
            };
        }

        /// No resources to clean up
        pub fn deinit(self: *Self) void {
            _ = self;
        }

        /// Compute Z-array for given string
        /// Z[i] = length of longest substring starting at i that matches prefix
        /// Time: O(n) | Space: O(n)
        fn computeZArray(self: *Self, s: []const T) ![]usize {
            const n = s.len;
            if (n == 0) return try self.allocator.alloc(usize, 0);

            const z = try self.allocator.alloc(usize, n);
            @memset(z, 0);
            z[0] = n; // By definition, entire string matches itself at position 0

            var l: usize = 0;
            var r: usize = 0;

            var i: usize = 1;
            while (i < n) : (i += 1) {
                // Case 1: i > r, compute Z[i] from scratch
                if (i > r) {
                    l = i;
                    r = i;

                    // Extend r as far as possible
                    while (r < n and s[r] == s[r - l]) {
                        r += 1;
                    }

                    z[i] = r - l;
                    r -|= 1; // Back to last matching position
                } else {
                    // Case 2: i <= r, use previously computed values
                    const k = i - l;

                    // If Z[k] < remaining window, reuse it
                    if (z[k] < r - i + 1) {
                        z[i] = z[k];
                    } else {
                        // Z[k] reaches or exceeds window, need to extend
                        l = i;
                        while (r + 1 < n and s[r + 1] == s[r + 1 - l]) {
                            r += 1;
                        }
                        z[i] = r - l + 1;
                    }
                }
            }

            return z;
        }

        /// Find first occurrence of pattern in text
        /// Returns index of first match, or null if not found
        /// Time: O(n + m) | Space: O(n + m)
        pub fn findFirst(self: *Self, text: []const T, pattern: []const T) !?usize {
            if (pattern.len == 0) return 0;
            if (text.len < pattern.len) return null;

            // Create sentinel value (use max value for T)
            const sentinel = switch (@typeInfo(T)) {
                .int => std.math.maxInt(T),
                .comptime_int => std.math.maxInt(i32),
                else => @compileError("Z-algorithm requires integer or comparable type"),
            };

            // Check if sentinel appears in pattern or text
            for (pattern) |c| {
                if (c == sentinel) return error.SentinelConflict;
            }
            for (text) |c| {
                if (c == sentinel) return error.SentinelConflict;
            }

            // Construct S = pattern + sentinel + text
            const s_len = pattern.len + 1 + text.len;
            const s = try self.allocator.alloc(T, s_len);
            defer self.allocator.free(s);

            @memcpy(s[0..pattern.len], pattern);
            s[pattern.len] = sentinel;
            @memcpy(s[pattern.len + 1 ..], text);

            // Compute Z-array
            const z = try self.computeZArray(s);
            defer self.allocator.free(z);

            // Find first position where Z[i] == pattern.len
            const offset = pattern.len + 1;
            for (offset..s_len) |i| {
                if (z[i] == pattern.len) {
                    return i - offset;
                }
            }

            return null;
        }

        /// Find all occurrences of pattern in text
        /// Returns list of match indices
        /// Time: O(n + m) | Space: O(n + m + k) where k = number of matches
        pub fn findAll(self: *Self, text: []const T, pattern: []const T) ![]Match {
            if (pattern.len == 0) {
                // Empty pattern matches at every position
                const matches = try self.allocator.alloc(Match, text.len + 1);
                for (matches, 0..) |*m, i| {
                    m.* = .{ .index = i };
                }
                return matches;
            }

            if (text.len < pattern.len) {
                return try self.allocator.alloc(Match, 0);
            }

            // Create sentinel value
            const sentinel = switch (@typeInfo(T)) {
                .int => std.math.maxInt(T),
                .comptime_int => std.math.maxInt(i32),
                else => @compileError("Z-algorithm requires integer or comparable type"),
            };

            // Check for sentinel conflicts
            for (pattern) |c| {
                if (c == sentinel) return error.SentinelConflict;
            }
            for (text) |c| {
                if (c == sentinel) return error.SentinelConflict;
            }

            // Construct S = pattern + sentinel + text
            const s_len = pattern.len + 1 + text.len;
            const s = try self.allocator.alloc(T, s_len);
            defer self.allocator.free(s);

            @memcpy(s[0..pattern.len], pattern);
            s[pattern.len] = sentinel;
            @memcpy(s[pattern.len + 1 ..], text);

            // Compute Z-array
            const z = try self.computeZArray(s);
            defer self.allocator.free(z);

            // Collect all matches
            var matches = std.array_list.AlignedManaged(Self.Match, null).init(self.allocator);
            defer matches.deinit();

            const offset = pattern.len + 1;
            for (offset..s_len) |i| {
                if (z[i] == pattern.len) {
                    try matches.append(.{ .index = i - offset });
                }
            }

            return matches.toOwnedSlice();
        }

        /// Check if pattern exists in text
        /// Time: O(n + m) | Space: O(n + m)
        pub fn contains(self: *Self, text: []const T, pattern: []const T) !bool {
            const first_match = try self.findFirst(text, pattern);
            return first_match != null;
        }

        /// Count occurrences of pattern in text
        /// Time: O(n + m) | Space: O(n + m)
        pub fn count(self: *Self, text: []const T, pattern: []const T) !usize {
            const matches = try self.findAll(text, pattern);
            defer self.allocator.free(matches);
            return matches.len;
        }

        /// Compute Z-array directly (useful for substring problems)
        /// Returns owned Z-array, caller must free
        /// Time: O(n) | Space: O(n)
        pub fn getZArray(self: *Self, s: []const T) ![]usize {
            return try self.computeZArray(s);
        }
    };
}

// ============================================================================
// TESTS
// ============================================================================

test "ZAlgorithm: basic single match" {
    const allocator = std.testing.allocator;

    var z = ZAlgorithm(u8).init(allocator);
    defer z.deinit();

    const text = "hello world";
    const pattern = "world";

    const idx = try z.findFirst(text, pattern);
    try std.testing.expectEqual(@as(?usize, 6), idx);
}

test "ZAlgorithm: multiple matches" {
    const allocator = std.testing.allocator;

    var z = ZAlgorithm(u8).init(allocator);
    defer z.deinit();

    const text = "ababcababa";
    const pattern = "aba";

    const matches = try z.findAll(text, pattern);
    defer allocator.free(matches);

    try std.testing.expectEqual(@as(usize, 3), matches.len);
    try std.testing.expectEqual(@as(usize, 0), matches[0].index);
    try std.testing.expectEqual(@as(usize, 5), matches[1].index);
    try std.testing.expectEqual(@as(usize, 7), matches[2].index);
}

test "ZAlgorithm: overlapping matches" {
    const allocator = std.testing.allocator;

    var z = ZAlgorithm(u8).init(allocator);
    defer z.deinit();

    const text = "aaaa";
    const pattern = "aa";

    const matches = try z.findAll(text, pattern);
    defer allocator.free(matches);

    try std.testing.expectEqual(@as(usize, 3), matches.len);
    try std.testing.expectEqual(@as(usize, 0), matches[0].index);
    try std.testing.expectEqual(@as(usize, 1), matches[1].index);
    try std.testing.expectEqual(@as(usize, 2), matches[2].index);
}

test "ZAlgorithm: no match" {
    const allocator = std.testing.allocator;

    var z = ZAlgorithm(u8).init(allocator);
    defer z.deinit();

    const text = "hello world";
    const pattern = "xyz";

    const idx = try z.findFirst(text, pattern);
    try std.testing.expectEqual(@as(?usize, null), idx);

    const matches = try z.findAll(text, pattern);
    defer allocator.free(matches);
    try std.testing.expectEqual(@as(usize, 0), matches.len);
}

test "ZAlgorithm: empty pattern" {
    const allocator = std.testing.allocator;

    var z = ZAlgorithm(u8).init(allocator);
    defer z.deinit();

    const text = "hello";
    const pattern = "";

    const idx = try z.findFirst(text, pattern);
    try std.testing.expectEqual(@as(?usize, 0), idx);

    const matches = try z.findAll(text, pattern);
    defer allocator.free(matches);
    try std.testing.expectEqual(@as(usize, 6), matches.len); // Matches at every position including end
}

test "ZAlgorithm: empty text" {
    const allocator = std.testing.allocator;

    var z = ZAlgorithm(u8).init(allocator);
    defer z.deinit();

    const text = "";
    const pattern = "hello";

    const idx = try z.findFirst(text, pattern);
    try std.testing.expectEqual(@as(?usize, null), idx);

    const matches = try z.findAll(text, pattern);
    defer allocator.free(matches);
    try std.testing.expectEqual(@as(usize, 0), matches.len);
}

test "ZAlgorithm: pattern longer than text" {
    const allocator = std.testing.allocator;

    var z = ZAlgorithm(u8).init(allocator);
    defer z.deinit();

    const text = "abc";
    const pattern = "abcdef";

    const idx = try z.findFirst(text, pattern);
    try std.testing.expectEqual(@as(?usize, null), idx);
}

test "ZAlgorithm: contains" {
    const allocator = std.testing.allocator;

    var z = ZAlgorithm(u8).init(allocator);
    defer z.deinit();

    try std.testing.expect(try z.contains("hello world", "world"));
    try std.testing.expect(!try z.contains("hello world", "xyz"));
}

test "ZAlgorithm: count" {
    const allocator = std.testing.allocator;

    var z = ZAlgorithm(u8).init(allocator);
    defer z.deinit();

    try std.testing.expectEqual(@as(usize, 3), try z.count("ababcababa", "aba"));
    try std.testing.expectEqual(@as(usize, 0), try z.count("hello", "xyz"));
    try std.testing.expectEqual(@as(usize, 1), try z.count("hello", "hello"));
}

test "ZAlgorithm: Z-array computation" {
    const allocator = std.testing.allocator;

    var z = ZAlgorithm(u8).init(allocator);
    defer z.deinit();

    const s = "aabcaabxaaaz";
    const z_array = try z.getZArray(s);
    defer allocator.free(z_array);

    try std.testing.expectEqual(@as(usize, 12), z_array[0]); // Entire string
    try std.testing.expectEqual(@as(usize, 1), z_array[1]); // "a" matches prefix "a"
    try std.testing.expectEqual(@as(usize, 0), z_array[2]); // "b" doesn't match "a"
    try std.testing.expectEqual(@as(usize, 0), z_array[3]); // "c" doesn't match "a"
    try std.testing.expectEqual(@as(usize, 3), z_array[4]); // "aab" matches prefix "aab"
}

test "ZAlgorithm: single character pattern" {
    const allocator = std.testing.allocator;

    var z = ZAlgorithm(u8).init(allocator);
    defer z.deinit();

    const text = "ababa";
    const pattern = "a";

    const matches = try z.findAll(text, pattern);
    defer allocator.free(matches);

    try std.testing.expectEqual(@as(usize, 3), matches.len);
    try std.testing.expectEqual(@as(usize, 0), matches[0].index);
    try std.testing.expectEqual(@as(usize, 2), matches[1].index);
    try std.testing.expectEqual(@as(usize, 4), matches[2].index);
}

test "ZAlgorithm: text equals pattern" {
    const allocator = std.testing.allocator;

    var z = ZAlgorithm(u8).init(allocator);
    defer z.deinit();

    const text = "hello";
    const pattern = "hello";

    const idx = try z.findFirst(text, pattern);
    try std.testing.expectEqual(@as(?usize, 0), idx);

    const matches = try z.findAll(text, pattern);
    defer allocator.free(matches);
    try std.testing.expectEqual(@as(usize, 1), matches.len);
}

test "ZAlgorithm: unicode support" {
    const allocator = std.testing.allocator;

    var z = ZAlgorithm(u32).init(allocator);
    defer z.deinit();

    const text = [_]u32{ 'h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd' };
    const pattern = [_]u32{ 'w', 'o', 'r', 'l', 'd' };

    const idx = try z.findFirst(&text, &pattern);
    try std.testing.expectEqual(@as(?usize, 6), idx);
}

test "ZAlgorithm: stress test" {
    const allocator = std.testing.allocator;

    var z = ZAlgorithm(u8).init(allocator);
    defer z.deinit();

    // Generate large text with known pattern occurrences
    const text_len = 10000;
    const text = try allocator.alloc(u8, text_len);
    defer allocator.free(text);

    // Fill with pattern "abc" repeated
    for (0..text_len) |i| {
        text[i] = @as(u8, @intCast((i % 3) + 'a'));
    }

    const pattern = "abc";
    const matches = try z.findAll(text, pattern);
    defer allocator.free(matches);

    // Pattern "abc" should appear every 3 characters
    // At positions 0, 3, 6, ..., up to text_len - pattern.len
    const expected_count = ((text_len - pattern.len) / 3) + 1;
    try std.testing.expectEqual(expected_count, matches.len);

    // Verify first few matches
    for (0..@min(10, matches.len)) |i| {
        try std.testing.expectEqual(i * 3, matches[i].index);
    }
}

test "ZAlgorithm: memory leak check" {
    const allocator = std.testing.allocator;

    var z = ZAlgorithm(u8).init(allocator);
    defer z.deinit();

    // Perform multiple operations to ensure no leaks
    for (0..100) |_| {
        const matches = try z.findAll("abcabcabc", "abc");
        allocator.free(matches);

        const z_array = try z.getZArray("aabcaab");
        allocator.free(z_array);

        _ = try z.findFirst("hello world", "world");
        _ = try z.contains("test", "es");
        _ = try z.count("aaaa", "aa");
    }
}
