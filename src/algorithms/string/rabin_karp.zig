const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Rabin-Karp - Rolling hash pattern matching algorithm
///
/// Uses rolling hash to efficiently find pattern occurrences in text.
/// The hash function allows O(1) hash updates when sliding the window.
///
/// Time Complexity:
///   Average: O(n + m) where n = text length, m = pattern length
///   Worst:   O(n*m) when many hash collisions occur
///
/// Space Complexity: O(1)
///
/// Characteristics:
/// - Probabilistic: uses hashing, may have false positives (requires verification)
/// - Rolling hash: efficient window sliding with O(1) hash update
/// - Multiple pattern search: can search for multiple patterns simultaneously
/// - Simple implementation: easier than KMP or Boyer-Moore
///
/// Use cases:
/// - Plagiarism detection
/// - Multiple pattern matching
/// - Substring search in large texts
///
/// Example:
/// ```zig
/// const rk = RabinKarp.init(std.testing.allocator);
/// defer rk.deinit();
/// const matches = try rk.findAll("hello world hello", "hello");
/// // matches = [0, 12]
/// ```
pub const RabinKarp = struct {
    allocator: Allocator,

    /// Base for polynomial rolling hash (prime number)
    const BASE: u64 = 257;

    /// Modulus for hash computation (large prime to reduce collisions)
    const MOD: u64 = 1_000_000_007;

    pub fn init(allocator: Allocator) RabinKarp {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *RabinKarp) void {
        _ = self;
    }

    /// Find the first occurrence of pattern in text.
    /// Returns the index of the match, or null if not found.
    ///
    /// Time: O(n + m) average, O(n*m) worst | Space: O(1)
    pub fn find(self: *RabinKarp, text: []const u8, pattern: []const u8) ?usize {
        _ = self;

        if (pattern.len == 0) return 0;
        if (text.len < pattern.len) return null;

        const pattern_hash = computeHash(pattern);
        const base_power = modPow(BASE, pattern.len - 1, MOD);

        var text_hash = computeHash(text[0..pattern.len]);

        // Check first window
        if (text_hash == pattern_hash and std.mem.eql(u8, text[0..pattern.len], pattern)) {
            return 0;
        }

        // Slide window
        var i: usize = 1;
        while (i <= text.len - pattern.len) : (i += 1) {
            // Remove leftmost character
            text_hash = (text_hash + MOD - (text[i - 1] * base_power) % MOD) % MOD;

            // Shift left and add rightmost character
            text_hash = (text_hash * BASE) % MOD;
            text_hash = (text_hash + text[i + pattern.len - 1]) % MOD;

            // Check hash match
            if (text_hash == pattern_hash) {
                // Verify actual match (handle hash collisions)
                if (std.mem.eql(u8, text[i..i + pattern.len], pattern)) {
                    return i;
                }
            }
        }

        return null;
    }

    /// Find all occurrences of pattern in text.
    /// Returns a slice of indices where matches occur.
    /// Caller owns the returned slice and must free it.
    ///
    /// Time: O(n + m) average, O(n*m) worst | Space: O(k) where k = number of matches
    pub fn findAll(self: *RabinKarp, text: []const u8, pattern: []const u8) ![]usize {
        if (pattern.len == 0) {
            // Empty pattern matches at every position
            const result = try self.allocator.alloc(usize, text.len + 1);
            for (0..text.len + 1) |i| {
                result[i] = i;
            }
            return result;
        }

        if (text.len < pattern.len) {
            return &[_]usize{};
        }

        var matches: std.ArrayListUnmanaged(usize) = .{};
        errdefer matches.deinit(self.allocator);

        const pattern_hash = computeHash(pattern);
        const base_power = modPow(BASE, pattern.len - 1, MOD);

        var text_hash = computeHash(text[0..pattern.len]);

        // Check first window
        if (text_hash == pattern_hash and std.mem.eql(u8, text[0..pattern.len], pattern)) {
            try matches.append(self.allocator, 0);
        }

        // Slide window
        var i: usize = 1;
        while (i <= text.len - pattern.len) : (i += 1) {
            // Remove leftmost character
            text_hash = (text_hash + MOD - (text[i - 1] * base_power) % MOD) % MOD;

            // Shift left and add rightmost character
            text_hash = (text_hash * BASE) % MOD;
            text_hash = (text_hash + text[i + pattern.len - 1]) % MOD;

            // Check hash match
            if (text_hash == pattern_hash) {
                // Verify actual match
                if (std.mem.eql(u8, text[i..i + pattern.len], pattern)) {
                    try matches.append(self.allocator, i);
                }
            }
        }

        return matches.toOwnedSlice(self.allocator);
    }

    /// Find occurrences of multiple patterns in text simultaneously.
    /// Returns a map from pattern index to slice of match indices.
    /// Caller must free each match slice and the result map.
    ///
    /// Time: O(n * k + Σm_i) where k = pattern count, m_i = pattern lengths
    /// Space: O(k + total matches)
    pub fn findMultiple(
        self: *RabinKarp,
        text: []const u8,
        patterns: []const []const u8,
    ) !std.AutoHashMapUnmanaged(usize, []usize) {
        var result: std.AutoHashMapUnmanaged(usize, []usize) = .{};
        errdefer {
            var iter = result.valueIterator();
            while (iter.next()) |matches| {
                self.allocator.free(matches.*);
            }
            result.deinit(self.allocator);
        }

        for (patterns, 0..) |pattern, idx| {
            const matches = try self.findAll(text, pattern);
            try result.put(self.allocator, idx, matches);
        }

        return result;
    }

    /// Compute hash of a string
    fn computeHash(s: []const u8) u64 {
        var hash: u64 = 0;
        for (s) |c| {
            hash = (hash * BASE + c) % MOD;
        }
        return hash;
    }

    /// Compute (base^exp) % mod using fast exponentiation
    fn modPow(base: u64, exp: usize, mod: u64) u64 {
        if (exp == 0) return 1;

        var result: u64 = 1;
        var b = base % mod;
        var e = exp;

        while (e > 0) {
            if (e & 1 == 1) {
                result = (result * b) % mod;
            }
            b = (b * b) % mod;
            e >>= 1;
        }

        return result;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "RabinKarp - basic find" {
    var rk = RabinKarp.init(testing.allocator);
    defer rk.deinit();

    const text = "hello world";
    const pattern = "world";

    const result = rk.find(text, pattern);
    try testing.expectEqual(@as(?usize, 6), result);
}

test "RabinKarp - pattern not found" {
    var rk = RabinKarp.init(testing.allocator);
    defer rk.deinit();

    const text = "hello world";
    const pattern = "xyz";

    const result = rk.find(text, pattern);
    try testing.expectEqual(@as(?usize, null), result);
}

test "RabinKarp - pattern at start" {
    var rk = RabinKarp.init(testing.allocator);
    defer rk.deinit();

    const text = "hello world";
    const pattern = "hello";

    const result = rk.find(text, pattern);
    try testing.expectEqual(@as(?usize, 0), result);
}

test "RabinKarp - empty pattern" {
    var rk = RabinKarp.init(testing.allocator);
    defer rk.deinit();

    const text = "hello";
    const pattern = "";

    const result = rk.find(text, pattern);
    try testing.expectEqual(@as(?usize, 0), result);
}

test "RabinKarp - pattern longer than text" {
    var rk = RabinKarp.init(testing.allocator);
    defer rk.deinit();

    const text = "hi";
    const pattern = "hello";

    const result = rk.find(text, pattern);
    try testing.expectEqual(@as(?usize, null), result);
}

test "RabinKarp - findAll single match" {
    var rk = RabinKarp.init(testing.allocator);
    defer rk.deinit();

    const text = "hello world";
    const pattern = "world";

    const matches = try rk.findAll(text, pattern);
    defer testing.allocator.free(matches);

    try testing.expectEqual(@as(usize, 1), matches.len);
    try testing.expectEqual(@as(usize, 6), matches[0]);
}

test "RabinKarp - findAll multiple matches" {
    var rk = RabinKarp.init(testing.allocator);
    defer rk.deinit();

    const text = "abababa";
    const pattern = "aba";

    const matches = try rk.findAll(text, pattern);
    defer testing.allocator.free(matches);

    try testing.expectEqual(@as(usize, 3), matches.len);
    try testing.expectEqual(@as(usize, 0), matches[0]);
    try testing.expectEqual(@as(usize, 2), matches[1]);
    try testing.expectEqual(@as(usize, 4), matches[2]);
}

test "RabinKarp - findAll no matches" {
    var rk = RabinKarp.init(testing.allocator);
    defer rk.deinit();

    const text = "hello world";
    const pattern = "xyz";

    const matches = try rk.findAll(text, pattern);
    defer testing.allocator.free(matches);

    try testing.expectEqual(@as(usize, 0), matches.len);
}

test "RabinKarp - findAll overlapping matches" {
    var rk = RabinKarp.init(testing.allocator);
    defer rk.deinit();

    const text = "aaa";
    const pattern = "aa";

    const matches = try rk.findAll(text, pattern);
    defer testing.allocator.free(matches);

    try testing.expectEqual(@as(usize, 2), matches.len);
    try testing.expectEqual(@as(usize, 0), matches[0]);
    try testing.expectEqual(@as(usize, 1), matches[1]);
}

test "RabinKarp - case sensitive" {
    var rk = RabinKarp.init(testing.allocator);
    defer rk.deinit();

    const text = "Hello World";
    const pattern = "hello";

    const result = rk.find(text, pattern);
    try testing.expectEqual(@as(?usize, null), result);
}

test "RabinKarp - full text match" {
    var rk = RabinKarp.init(testing.allocator);
    defer rk.deinit();

    const text = "test";
    const pattern = "test";

    const result = rk.find(text, pattern);
    try testing.expectEqual(@as(?usize, 0), result);
}

test "RabinKarp - findMultiple basic" {
    var rk = RabinKarp.init(testing.allocator);
    defer rk.deinit();

    const text = "the quick brown fox jumps over the lazy dog";
    const patterns = [_][]const u8{ "the", "fox", "dog" };

    var result = try rk.findMultiple(text, &patterns);
    defer {
        var iter = result.valueIterator();
        while (iter.next()) |matches| {
            testing.allocator.free(matches.*);
        }
        result.deinit(testing.allocator);
    }

    // Check "the" matches
    const the_matches = result.get(0).?;
    try testing.expectEqual(@as(usize, 2), the_matches.len);
    try testing.expectEqual(@as(usize, 0), the_matches[0]);
    try testing.expectEqual(@as(usize, 31), the_matches[1]);

    // Check "fox" matches
    const fox_matches = result.get(1).?;
    try testing.expectEqual(@as(usize, 1), fox_matches.len);
    try testing.expectEqual(@as(usize, 16), fox_matches[0]);

    // Check "dog" matches
    const dog_matches = result.get(2).?;
    try testing.expectEqual(@as(usize, 1), dog_matches.len);
    try testing.expectEqual(@as(usize, 40), dog_matches[0]);
}

test "RabinKarp - stress test with large text" {
    var rk = RabinKarp.init(testing.allocator);
    defer rk.deinit();

    // Create large text with pattern embedded
    const pattern = "needle";
    const chunk = "hay hay hay ";
    const count = 100;

    var text_list: std.ArrayListUnmanaged(u8) = .{};
    defer text_list.deinit(testing.allocator);

    for (0..count) |_| {
        try text_list.appendSlice(testing.allocator, chunk);
    }
    try text_list.appendSlice(testing.allocator, pattern);
    for (0..count) |_| {
        try text_list.appendSlice(testing.allocator, chunk);
    }

    const text = text_list.items;
    const result = rk.find(text, pattern);

    try testing.expect(result != null);
    try testing.expect(std.mem.eql(u8, text[result.?..result.? + pattern.len], pattern));
}

test "RabinKarp - hash collision handling" {
    var rk = RabinKarp.init(testing.allocator);
    defer rk.deinit();

    // These strings might have hash collisions but should still work correctly
    const text = "abcdefghijklmnopqrstuvwxyz";
    const pattern = "xyz";

    const result = rk.find(text, pattern);
    try testing.expectEqual(@as(?usize, 23), result);
}

test "RabinKarp - repeated pattern" {
    var rk = RabinKarp.init(testing.allocator);
    defer rk.deinit();

    const text = "aaaaaaaaaa";
    const pattern = "aaa";

    const matches = try rk.findAll(text, pattern);
    defer testing.allocator.free(matches);

    try testing.expectEqual(@as(usize, 8), matches.len);
    for (0..8) |i| {
        try testing.expectEqual(@as(usize, i), matches[i]);
    }
}
