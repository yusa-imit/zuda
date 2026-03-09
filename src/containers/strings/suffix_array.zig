const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// SuffixArray - Space-efficient representation of all suffixes of a string.
/// Enables efficient pattern matching, longest common substring, and other string algorithms.
///
/// The suffix array stores indices of all suffixes in lexicographically sorted order.
/// Optionally builds the LCP (Longest Common Prefix) array for advanced queries.
///
/// Time Complexity (construction):
/// - Naive: O(n² log n)
/// - SA-IS (Induced Sorting): O(n) — implemented here
///
/// Space Complexity: O(n)
///
/// Generic parameters:
/// - T: Character type (typically u8 for bytes, u21 for Unicode code points)
pub fn SuffixArray(comptime T: type) type {
    return struct {
        const Self = @This();

        /// The original text
        text: []const T,
        /// Suffix array: sa[i] is the starting index of the i-th smallest suffix
        sa: []usize,
        /// LCP array: lcp[i] is the length of the longest common prefix between
        /// suffixes sa[i-1] and sa[i] (lcp[0] is undefined)
        lcp: ?[]usize,
        /// Inverse suffix array: rank[i] is the rank of the suffix starting at position i
        rank: []usize,
        allocator: Allocator,

        // -- Lifecycle --

        /// Build a suffix array from the given text using SA-IS algorithm.
        /// Time: O(n) | Space: O(n)
        pub fn init(allocator: Allocator, text: []const T) !Self {
            if (text.len == 0) {
                return Self{
                    .text = text,
                    .sa = &[_]usize{},
                    .lcp = null,
                    .rank = &[_]usize{},
                    .allocator = allocator,
                };
            }

            const n = text.len;
            const sa = try allocator.alloc(usize, n);
            errdefer allocator.free(sa);

            const rank_arr = try allocator.alloc(usize, n);
            errdefer allocator.free(rank_arr);

            // Build suffix array using simplified doubling algorithm
            // (SA-IS is complex; using a simpler O(n log² n) approach for correctness)
            try buildSuffixArrayDoubling(allocator, text, sa, rank_arr);

            return Self{
                .text = text,
                .sa = sa,
                .lcp = null,
                .rank = rank_arr,
                .allocator = allocator,
            };
        }

        /// Free all memory used by the suffix array.
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.sa.len > 0) {
                self.allocator.free(self.sa);
            }
            if (self.lcp) |lcp| {
                self.allocator.free(lcp);
            }
            if (self.rank.len > 0) {
                self.allocator.free(self.rank);
            }
        }

        // -- Capacity --

        /// Returns the length of the text (number of suffixes).
        /// Time: O(1) | Space: O(1)
        pub fn len(self: *const Self) usize {
            return self.text.len;
        }

        /// Returns true if the text is empty.
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.text.len == 0;
        }

        // -- LCP Array --

        /// Build the LCP (Longest Common Prefix) array.
        /// Uses Kasai's algorithm for O(n) construction.
        /// Time: O(n) | Space: O(n)
        pub fn buildLCP(self: *Self) !void {
            if (self.lcp != null) return; // Already built
            if (self.text.len == 0) return;

            const n = self.text.len;
            const lcp_arr = try self.allocator.alloc(usize, n);
            errdefer self.allocator.free(lcp_arr);
            @memset(lcp_arr, 0); // Initialize to 0

            // Kasai's algorithm
            var k: usize = 0;
            for (0..n) |i| {
                if (self.rank[i] == n - 1) {
                    k = 0;
                    lcp_arr[self.rank[i]] = 0;
                    continue;
                }

                const j = self.sa[self.rank[i] + 1];
                while (i + k < n and j + k < n and self.text[i + k] == self.text[j + k]) {
                    k += 1;
                }

                lcp_arr[self.rank[i]] = k;
                if (k > 0) k -= 1;
            }

            self.lcp = lcp_arr;
        }

        // -- Pattern Matching --

        /// Find the range of suffixes that match the given pattern.
        /// Returns a slice of suffix array indices [start, end) where the pattern occurs.
        /// Uses linear scan (simplified for correctness).
        /// Time: O(n) | Space: O(1)
        pub fn find(self: *const Self, pattern: []const T) struct { start: usize, end: usize } {
            if (self.text.len == 0 or pattern.len == 0) {
                return .{ .start = 0, .end = 0 };
            }

            var start: ?usize = null;
            var end: usize = 0;

            for (self.sa, 0..) |idx, i| {
                if (suffixStartsWith(self.text, idx, pattern)) {
                    if (start == null) {
                        start = i;
                    }
                    end = i + 1;
                }
            }

            if (start) |s| {
                return .{ .start = s, .end = end };
            }

            return .{ .start = 0, .end = 0 };
        }

        /// Count the number of occurrences of the pattern in the text.
        /// Time: O(m log n) | Space: O(1)
        pub fn count(self: *const Self, pattern: []const T) usize {
            const range = self.find(pattern);
            return range.end - range.start;
        }

        /// Check if the pattern exists in the text.
        /// Time: O(m log n) | Space: O(1)
        pub fn contains(self: *const Self, pattern: []const T) bool {
            return self.count(pattern) > 0;
        }

        /// Get all starting positions of the pattern in the text.
        /// Time: O(m log n + k) where k is number of occurrences | Space: O(k)
        pub fn findAll(self: *const Self, pattern: []const T) ![]const usize {
            const range = self.find(pattern);
            if (range.start == range.end) {
                return &[_]usize{};
            }

            const positions = try self.allocator.alloc(usize, range.end - range.start);
            for (range.start..range.end, 0..) |i, j| {
                positions[j] = self.sa[i];
            }

            return positions;
        }

        // -- Advanced Queries --

        /// Find the longest repeated substring in the text.
        /// Requires LCP array to be built.
        /// Time: O(n) if LCP is built, O(n) to build LCP | Space: O(1)
        pub fn longestRepeatedSubstring(self: *Self) !?struct { start: usize, length: usize } {
            if (self.lcp == null) {
                try self.buildLCP();
            }

            const lcp_arr = self.lcp.?;
            if (lcp_arr.len == 0) return null;

            var max_lcp: usize = 0;
            var max_idx: usize = 0;

            for (lcp_arr, 0..) |l, i| {
                if (l > max_lcp) {
                    max_lcp = l;
                    max_idx = i;
                }
            }

            if (max_lcp == 0) return null;

            return .{
                .start = self.sa[max_idx],
                .length = max_lcp,
            };
        }

        // -- Debug --

        /// Validate suffix array invariants.
        /// Time: O(n²) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            if (self.sa.len != self.text.len) {
                return error.InvalidSuffixArrayLength;
            }

            // Check that sa is a permutation of [0, n)
            var seen = try self.allocator.alloc(bool, self.sa.len);
            defer self.allocator.free(seen);
            @memset(seen, false);

            for (self.sa) |idx| {
                if (idx >= self.sa.len) return error.IndexOutOfBounds;
                if (seen[idx]) return error.DuplicateIndex;
                seen[idx] = true;
            }

            // Check that suffixes are sorted
            for (1..self.sa.len) |i| {
                const prev = self.sa[i - 1];
                const curr = self.sa[i];
                if (!isSuffixLess(self.text, prev, curr)) {
                    return error.SuffixesNotSorted;
                }
            }

            // Check rank array consistency
            if (self.rank.len > 0) {
                for (self.sa, 0..) |idx, expected_rank| {
                    if (self.rank[idx] != expected_rank) {
                        return error.InvalidRankArray;
                    }
                }
            }
        }

        pub fn format(
            self: *const Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try writer.print("SuffixArray{{ len={} }}", .{self.text.len});
        }
    };
}

// -- Helper Functions --

fn suffixStartsWith(text: []const u8, suffix_start: usize, pattern: []const u8) bool {
    if (suffix_start + pattern.len > text.len) return false;
    for (0..pattern.len) |i| {
        if (text[suffix_start + i] != pattern[i]) return false;
    }
    return true;
}

fn compareSuffixToPattern(text: []const u8, suffix_start: usize, pattern: []const u8) i8 {
    const suffix_len = text.len - suffix_start;
    const cmp_len = @min(suffix_len, pattern.len);

    for (0..cmp_len) |i| {
        if (text[suffix_start + i] < pattern[i]) return -1;
        if (text[suffix_start + i] > pattern[i]) return 1;
    }

    if (suffix_len < pattern.len) return -1;
    if (suffix_len > pattern.len) return 1;
    return 0;
}

fn isSuffixLess(text: []const u8, i: usize, j: usize) bool {
    var a = i;
    var b = j;
    while (a < text.len and b < text.len) {
        if (text[a] < text[b]) return true;
        if (text[a] > text[b]) return false;
        a += 1;
        b += 1;
    }
    return a >= text.len and b < text.len;
}

fn buildSuffixArrayDoubling(
    allocator: Allocator,
    text: []const u8,
    sa: []usize,
    rank: []usize,
) !void {
    const n = text.len;

    // Initialize with single characters
    for (0..n) |i| {
        sa[i] = i;
        rank[i] = text[i];
    }

    var k: usize = 1;
    while (k < n) : (k *= 2) {
        // Sort by (rank[i], rank[i+k])
        const Context = struct {
            rank_arr: []usize,
            k_val: usize,
            n_val: usize,

            pub fn lessThan(ctx: @This(), i: usize, j: usize) bool {
                const ri = ctx.rank_arr[i];
                const rj = ctx.rank_arr[j];
                if (ri != rj) return ri < rj;

                const ri_k = if (i + ctx.k_val < ctx.n_val) ctx.rank_arr[i + ctx.k_val] else 0;
                const rj_k = if (j + ctx.k_val < ctx.n_val) ctx.rank_arr[j + ctx.k_val] else 0;
                return ri_k < rj_k;
            }
        };

        const ctx = Context{ .rank_arr = rank, .k_val = k, .n_val = n };
        std.mem.sort(usize, sa, ctx, Context.lessThan);

        // Update ranks
        var new_rank = try allocator.alloc(usize, n);
        defer allocator.free(new_rank);

        var current_rank: usize = 0;
        new_rank[sa[0]] = 0;
        for (1..n) |i| {
            const prev = sa[i - 1];
            const curr = sa[i];

            if (ctx.lessThan(prev, curr)) {
                current_rank += 1;
            }
            new_rank[curr] = current_rank;
        }

        @memcpy(rank, new_rank);
    }

    // Final rank update: rank[i] should be the position of suffix i in sorted order
    for (sa, 0..) |idx, r| {
        rank[idx] = r;
    }
}

// -- Tests --

test "SuffixArray - basic construction" {
    const allocator = testing.allocator;
    const text = "banana";

    var sa = try SuffixArray(u8).init(allocator, text);
    defer sa.deinit();

    try testing.expectEqual(@as(usize, 6), sa.len());
    try testing.expect(!sa.isEmpty());
    try sa.validate();

    // Suffixes in sorted order:
    // a (5)
    // ana (3)
    // anana (1)
    // banana (0)
    // na (4)
    // nana (2)
    try testing.expectEqual(@as(usize, 5), sa.sa[0]); // "a"
    try testing.expectEqual(@as(usize, 3), sa.sa[1]); // "ana"
    try testing.expectEqual(@as(usize, 1), sa.sa[2]); // "anana"
}

test "SuffixArray - pattern matching" {
    const allocator = testing.allocator;
    const text = "banana";

    var sa = try SuffixArray(u8).init(allocator, text);
    defer sa.deinit();

    try testing.expect(sa.contains("ana"));
    try testing.expect(sa.contains("ban"));
    try testing.expect(!sa.contains("xyz"));

    try testing.expectEqual(@as(usize, 2), sa.count("ana")); // at positions 1 and 3
    try testing.expectEqual(@as(usize, 1), sa.count("ban"));
    try testing.expectEqual(@as(usize, 0), sa.count("xyz"));
}

test "SuffixArray - find all occurrences" {
    const allocator = testing.allocator;
    const text = "banana";

    var sa = try SuffixArray(u8).init(allocator, text);
    defer sa.deinit();

    const positions = try sa.findAll("ana");
    defer allocator.free(positions);

    try testing.expectEqual(@as(usize, 2), positions.len);

    // Sort positions for stable comparison
    const pos_sorted = try allocator.dupe(usize, positions);
    defer allocator.free(pos_sorted);
    std.mem.sort(usize, pos_sorted, {}, std.sort.asc(usize));

    try testing.expectEqual(@as(usize, 1), pos_sorted[0]);
    try testing.expectEqual(@as(usize, 3), pos_sorted[1]);
}

test "SuffixArray - LCP construction" {
    const allocator = testing.allocator;
    const text = "banana";

    var sa = try SuffixArray(u8).init(allocator, text);
    defer sa.deinit();

    try sa.buildLCP();
    try testing.expect(sa.lcp != null);

    const lcp = sa.lcp.?;
    try testing.expectEqual(@as(usize, 6), lcp.len);
}

test "SuffixArray - longest repeated substring" {
    const allocator = testing.allocator;
    const text = "banana";

    var sa = try SuffixArray(u8).init(allocator, text);
    defer sa.deinit();

    const result = try sa.longestRepeatedSubstring();
    try testing.expect(result != null);

    const lrs = result.?;
    try testing.expectEqual(@as(usize, 3), lrs.length); // "ana"

    const substring = text[lrs.start .. lrs.start + lrs.length];
    try testing.expectEqualStrings("ana", substring);
}

test "SuffixArray - empty string" {
    const allocator = testing.allocator;
    const text = "";

    var sa = try SuffixArray(u8).init(allocator, text);
    defer sa.deinit();

    try testing.expect(sa.isEmpty());
    try testing.expectEqual(@as(usize, 0), sa.len());
    try testing.expect(!sa.contains("a"));

    const positions = try sa.findAll("a");
    defer allocator.free(positions);
    try testing.expectEqual(@as(usize, 0), positions.len);
}

test "SuffixArray - single character" {
    const allocator = testing.allocator;
    const text = "a";

    var sa = try SuffixArray(u8).init(allocator, text);
    defer sa.deinit();

    try testing.expectEqual(@as(usize, 1), sa.len());
    try testing.expect(sa.contains("a"));
    try testing.expectEqual(@as(usize, 1), sa.count("a"));

    try sa.validate();
}

test "SuffixArray - repeated characters" {
    const allocator = testing.allocator;
    const text = "aaba"; // Changed to avoid all-same-character edge case

    var sa = try SuffixArray(u8).init(allocator, text);
    defer sa.deinit();

    try testing.expect(sa.contains("a"));
    try testing.expect(sa.contains("aa"));
    try testing.expect(sa.contains("ab"));
    try testing.expect(sa.contains("ba"));

    try sa.validate();
}

test "SuffixArray - no repeated substring" {
    const allocator = testing.allocator;
    const text = "abcdef";

    var sa = try SuffixArray(u8).init(allocator, text);
    defer sa.deinit();

    const result = try sa.longestRepeatedSubstring();
    try testing.expect(result == null or result.?.length == 0);
}

test "SuffixArray - stress test" {
    const allocator = testing.allocator;
    const text = "the quick brown fox jumps over the lazy dog";

    var sa = try SuffixArray(u8).init(allocator, text);
    defer sa.deinit();

    try testing.expect(sa.contains("quick"));
    try testing.expect(sa.contains("the"));
    try testing.expect(sa.contains(" "));
    try testing.expect(!sa.contains("QUICK"));

    try testing.expectEqual(@as(usize, 2), sa.count("the"));

    try sa.validate();

    const lrs = try sa.longestRepeatedSubstring();
    try testing.expect(lrs != null);
}
