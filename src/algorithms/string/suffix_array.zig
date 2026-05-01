//! Suffix Array Construction with LCP (Longest Common Prefix) Array
//!
//! Suffix array is a sorted array of all suffixes of a string. It's a space-efficient
//! alternative to suffix trees with many applications in string processing.
//!
//! This implementation provides:
//! - O(n log n) suffix array construction (DC3/Skew algorithm would be O(n) but more complex)
//! - O(n) LCP array construction (Kasai's algorithm)
//! - Pattern searching in O(m log n) time
//! - Longest repeated substring finding
//!
//! Time complexity:
//! - buildSuffixArray: O(n log² n) using counting sort + prefix doubling
//! - buildLCP: O(n) using Kasai's algorithm
//! - search: O(m log n) binary search where m = pattern length
//!
//! Space complexity: O(n) for suffix array and auxiliary arrays
//!
//! Use cases:
//! - Pattern matching (multiple occurrences)
//! - Longest repeated substring
//! - Longest common substring of multiple strings
//! - Data compression (BWT construction)
//! - Bioinformatics (DNA sequence analysis)
//!
//! Reference: Manber & Myers (1990), Kasai et al. (2001)

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Suffix array result containing sorted suffix indices
pub fn SuffixArrayResult(comptime T: type) type {
    return struct {
        /// Sorted array of suffix starting positions
        sa: []usize,
        /// Rank array (inverse of suffix array)
        rank: []usize,
        /// Allocator used for memory
        allocator: Allocator,
        
        const Self = @This();
        
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.sa);
            self.allocator.free(self.rank);
        }
    };
}

/// LCP (Longest Common Prefix) array result
pub fn LCPResult(comptime T: type) type {
    return struct {
        /// LCP[i] = length of longest common prefix between suffixes sa[i] and sa[i+1]
        lcp: []usize,
        allocator: Allocator,
        
        const Self = @This();
        
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.lcp);
        }
    };
}

/// Build suffix array using prefix doubling algorithm with counting sort
/// Time: O(n log² n), Space: O(n)
pub fn buildSuffixArray(comptime T: type, text: []const T, allocator: Allocator) !SuffixArrayResult(T) {
    if (text.len == 0) return error.EmptyText;
    
    const n = text.len;
    
    // Allocate arrays
    var sa = try allocator.alloc(usize, n);
    errdefer allocator.free(sa);
    var rank = try allocator.alloc(usize, n);
    errdefer allocator.free(rank);
    var tmp = try allocator.alloc(usize, n);
    defer allocator.free(tmp);
    
    // Initialize suffix array and rank
    for (0..n) |i| {
        sa[i] = i;
        rank[i] = @as(usize, @intCast(text[i]));
    }
    
    // Prefix doubling: sort by first k characters, then 2k, 4k, ...
    var k: usize = 1;
    while (k < n) : (k *= 2) {
        // Count sort by second half (rank[i + k])
        try countingSort(allocator, sa, rank, tmp, n, k);
        // Count sort by first half (rank[i])
        try countingSort(allocator, sa, rank, tmp, n, 0);
        
        // Update ranks based on new order
        @memcpy(tmp, rank);
        rank[sa[0]] = 0;
        var r: usize = 0;
        for (1..n) |i| {
            const prev = sa[i - 1];
            const curr = sa[i];
            
            // Check if pair (rank[curr], rank[curr+k]) differs from previous
            const prev_first = tmp[prev];
            const prev_second = if (prev + k < n) tmp[prev + k] else 0;
            const curr_first = tmp[curr];
            const curr_second = if (curr + k < n) tmp[curr + k] else 0;
            
            if (curr_first != prev_first or curr_second != prev_second) {
                r += 1;
            }
            rank[curr] = r;
        }
        
        // If all ranks are unique, we're done
        if (r == n - 1) break;
    }
    
    return SuffixArrayResult(T){
        .sa = sa,
        .rank = rank,
        .allocator = allocator,
    };
}

/// Helper: counting sort for suffix array construction
fn countingSort(allocator: Allocator, sa: []usize, rank: []const usize, tmp: []usize, n: usize, k: usize) !void {
    const max_rank = blk: {
        var max: usize = 0;
        for (rank) |r| {
            if (r > max) max = r;
        }
        break :blk max + 1;
    };

    // Count array
    var count = try allocator.alloc(usize, max_rank);
    defer allocator.free(count);
    @memset(count, 0);
    
    // Count occurrences
    for (sa) |i| {
        const key = if (i + k < n) rank[i + k] else 0;
        count[key] += 1;
    }
    
    // Cumulative count
    for (1..max_rank) |i| {
        count[i] += count[i - 1];
    }
    
    // Build sorted array (right to left for stability)
    var i: usize = n;
    while (i > 0) {
        i -= 1;
        const key = if (sa[i] + k < n) rank[sa[i] + k] else 0;
        count[key] -= 1;
        tmp[count[key]] = sa[i];
    }
    
    @memcpy(sa, tmp);
}

/// Build LCP array using Kasai's algorithm
/// LCP[i] = longest common prefix length between suffixes sa[i] and sa[i+1]
/// Time: O(n), Space: O(n)
pub fn buildLCP(comptime T: type, text: []const T, sa: []const usize, rank: []const usize, allocator: Allocator) !LCPResult(T) {
    const n = text.len;
    if (n == 0) return error.EmptyText;
    if (sa.len != n or rank.len != n) return error.LengthMismatch;
    
    var lcp = try allocator.alloc(usize, n);
    errdefer allocator.free(lcp);
    
    var h: usize = 0; // Height (LCP length)
    
    for (0..n) |i| {
        if (rank[i] == 0) {
            lcp[0] = 0;
            continue;
        }
        
        const j = sa[rank[i] - 1]; // Previous suffix in sorted order
        
        // Compute LCP between suffixes i and j
        while (i + h < n and j + h < n and text[i + h] == text[j + h]) {
            h += 1;
        }
        
        lcp[rank[i]] = h;
        
        // Decrease height by 1 for next iteration (Kasai's optimization)
        if (h > 0) h -= 1;
    }
    
    return LCPResult(T){
        .lcp = lcp,
        .allocator = allocator,
    };
}

/// Search for pattern in text using suffix array
/// Returns all starting positions where pattern occurs
/// Time: O(m log n) where m = pattern length
pub fn search(comptime T: type, text: []const T, sa: []const usize, pattern: []const T, allocator: Allocator) ![]usize {
    if (text.len == 0 or pattern.len == 0) return error.EmptyInput;
    if (sa.len != text.len) return error.LengthMismatch;
    
    const n = text.len;
    const m = pattern.len;
    
    // Binary search for first occurrence
    var left: usize = 0;
    var right: usize = n;
    
    while (left < right) {
        const mid = left + (right - left) / 2;
        const suffix_start = sa[mid];
        const cmp_len = @min(m, n - suffix_start);
        
        const cmp = std.mem.order(T, pattern[0..@min(m, cmp_len)], text[suffix_start..][0..cmp_len]);
        
        if (cmp == .lt or (cmp == .eq and m > cmp_len)) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    const start = left;
    
    // Binary search for last occurrence
    left = start;
    right = n;
    
    while (left < right) {
        const mid = left + (right - left) / 2;
        const suffix_start = sa[mid];
        const cmp_len = @min(m, n - suffix_start);
        
        const cmp = std.mem.order(T, pattern[0..@min(m, cmp_len)], text[suffix_start..][0..cmp_len]);
        
        if (cmp == .gt or (cmp == .eq and m <= cmp_len)) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    
    const end = left;
    
    if (start >= n) return allocator.alloc(usize, 0);
    
    // Collect all positions
    const count = end - start;
    var positions = try allocator.alloc(usize, count);
    for (start..end, 0..) |i, idx| {
        positions[idx] = sa[i];
    }
    
    return positions;
}

/// Find longest repeated substring in text
/// Returns (position, length) of the longest repeated substring
/// Time: O(n) after building suffix array and LCP
pub fn longestRepeatedSubstring(comptime T: type, text: []const T, sa: []const usize, lcp: []const usize) !struct { pos: usize, len: usize } {
    if (text.len == 0) return error.EmptyText;
    if (sa.len != text.len or lcp.len != text.len) return error.LengthMismatch;
    
    var max_len: usize = 0;
    var max_pos: usize = 0;
    
    for (lcp, 0..) |len, i| {
        if (len > max_len) {
            max_len = len;
            max_pos = sa[i];
        }
    }
    
    return .{ .pos = max_pos, .len = max_len };
}

/// Count number of distinct substrings
/// Time: O(n) after building suffix array and LCP
pub fn countDistinctSubstrings(text_len: usize, lcp: []const usize) usize {
    if (text_len == 0) return 0;
    
    // Total substrings = n(n+1)/2
    const total = text_len * (text_len + 1) / 2;
    
    // Subtract common prefixes (duplicates)
    var duplicates: usize = 0;
    for (lcp) |len| {
        duplicates += len;
    }
    
    return total - duplicates;
}

// Tests
const testing = std.testing;

test "suffix array: basic construction" {
    const text = "banana";
    var result = try buildSuffixArray(u8, text, testing.allocator);
    defer result.deinit();
    
    // Expected: ["a", "ana", "anana", "banana", "na", "nana"]
    // Positions: [5, 3, 1, 0, 4, 2]
    try testing.expectEqualSlices(usize, &.{ 5, 3, 1, 0, 4, 2 }, result.sa);
}

test "suffix array: single character" {
    const text = "a";
    var result = try buildSuffixArray(u8, text, testing.allocator);
    defer result.deinit();
    
    try testing.expectEqualSlices(usize, &.{0}, result.sa);
}

test "suffix array: repeated characters" {
    const text = "aaaa";
    var result = try buildSuffixArray(u8, text, testing.allocator);
    defer result.deinit();
    
    // All suffixes are lexicographically sorted by length
    try testing.expectEqualSlices(usize, &.{ 3, 2, 1, 0 }, result.sa);
}

test "suffix array: empty text error" {
    const text = "";
    try testing.expectError(error.EmptyText, buildSuffixArray(u8, text, testing.allocator));
}

test "LCP array: basic construction" {
    const text = "banana";
    var sa_result = try buildSuffixArray(u8, text, testing.allocator);
    defer sa_result.deinit();
    
    var lcp_result = try buildLCP(u8, text, sa_result.sa, sa_result.rank, testing.allocator);
    defer lcp_result.deinit();
    
    // LCP values between consecutive suffixes
    // Expected: [0, 1, 3, 0, 0, 2] based on suffix comparisons
    try testing.expect(lcp_result.lcp[0] == 0); // First LCP always 0
    try testing.expect(lcp_result.lcp.len == text.len);
}

test "LCP array: all same characters" {
    const text = "aaaa";
    var sa_result = try buildSuffixArray(u8, text, testing.allocator);
    defer sa_result.deinit();
    
    var lcp_result = try buildLCP(u8, text, sa_result.sa, sa_result.rank, testing.allocator);
    defer lcp_result.deinit();
    
    // Each suffix shares n-1, n-2, n-3 characters with next
    try testing.expectEqualSlices(usize, &.{ 0, 3, 2, 1 }, lcp_result.lcp);
}

test "pattern search: found multiple occurrences" {
    const text = "abracadabra";
    var sa_result = try buildSuffixArray(u8, text, testing.allocator);
    defer sa_result.deinit();
    
    const pattern = "abra";
    const positions = try search(u8, text, sa_result.sa, pattern, testing.allocator);
    defer testing.allocator.free(positions);
    
    // "abra" appears at positions 0 and 7
    try testing.expectEqual(@as(usize, 2), positions.len);
    
    // Sort positions for consistent comparison
    std.mem.sort(usize, positions, {}, std.sort.asc(usize));
    try testing.expectEqualSlices(usize, &.{ 0, 7 }, positions);
}

test "pattern search: single occurrence" {
    const text = "hello world";
    var sa_result = try buildSuffixArray(u8, text, testing.allocator);
    defer sa_result.deinit();
    
    const pattern = "world";
    const positions = try search(u8, text, sa_result.sa, pattern, testing.allocator);
    defer testing.allocator.free(positions);
    
    try testing.expectEqual(@as(usize, 1), positions.len);
    try testing.expectEqual(@as(usize, 6), positions[0]);
}

test "pattern search: not found" {
    const text = "hello world";
    var sa_result = try buildSuffixArray(u8, text, testing.allocator);
    defer sa_result.deinit();
    
    const pattern = "xyz";
    const positions = try search(u8, text, sa_result.sa, pattern, testing.allocator);
    defer testing.allocator.free(positions);
    
    try testing.expectEqual(@as(usize, 0), positions.len);
}

test "longest repeated substring: basic" {
    const text = "banana";
    var sa_result = try buildSuffixArray(u8, text, testing.allocator);
    defer sa_result.deinit();
    
    var lcp_result = try buildLCP(u8, text, sa_result.sa, sa_result.rank, testing.allocator);
    defer lcp_result.deinit();
    
    const result = try longestRepeatedSubstring(u8, text, sa_result.sa, lcp_result.lcp);
    
    // "ana" repeats at positions 1 and 3 (length 3)
    try testing.expectEqual(@as(usize, 3), result.len);
    try testing.expect(result.pos == 1 or result.pos == 3);
}

test "longest repeated substring: no repetition" {
    const text = "abcd";
    var sa_result = try buildSuffixArray(u8, text, testing.allocator);
    defer sa_result.deinit();
    
    var lcp_result = try buildLCP(u8, text, sa_result.sa, sa_result.rank, testing.allocator);
    defer lcp_result.deinit();
    
    const result = try longestRepeatedSubstring(u8, text, sa_result.sa, lcp_result.lcp);
    
    // No repeated substring (except empty)
    try testing.expectEqual(@as(usize, 0), result.len);
}

test "count distinct substrings: basic" {
    const text = "abab";
    var sa_result = try buildSuffixArray(u8, text, testing.allocator);
    defer sa_result.deinit();
    
    var lcp_result = try buildLCP(u8, text, sa_result.sa, sa_result.rank, testing.allocator);
    defer lcp_result.deinit();
    
    const count = countDistinctSubstrings(text.len, lcp_result.lcp);
    
    // Distinct substrings: "a", "b", "ab", "ba", "aba", "bab", "abab" = 7
    // (Empty string not counted)
    try testing.expectEqual(@as(usize, 7), count);
}

test "count distinct substrings: all unique" {
    const text = "abcd";
    var sa_result = try buildSuffixArray(u8, text, testing.allocator);
    defer sa_result.deinit();
    
    var lcp_result = try buildLCP(u8, text, sa_result.sa, sa_result.rank, testing.allocator);
    defer lcp_result.deinit();
    
    const count = countDistinctSubstrings(text.len, lcp_result.lcp);
    
    // All substrings are unique: 4*5/2 = 10
    try testing.expectEqual(@as(usize, 10), count);
}

test "suffix array: large text" {
    const text = "mississippi";
    var sa_result = try buildSuffixArray(u8, text, testing.allocator);
    defer sa_result.deinit();
    
    var lcp_result = try buildLCP(u8, text, sa_result.sa, sa_result.rank, testing.allocator);
    defer lcp_result.deinit();
    
    // Verify basic properties
    try testing.expectEqual(text.len, sa_result.sa.len);
    try testing.expectEqual(text.len, lcp_result.lcp.len);
    
    // First element should be the lexicographically smallest suffix
    const first_suffix = sa_result.sa[0];
    try testing.expect(text[first_suffix] == 'i'); // "i" or "ippi" or similar
}

test "pattern search: overlapping occurrences" {
    const text = "aaaa";
    var sa_result = try buildSuffixArray(u8, text, testing.allocator);
    defer sa_result.deinit();
    
    const pattern = "aa";
    const positions = try search(u8, text, sa_result.sa, pattern, testing.allocator);
    defer testing.allocator.free(positions);
    
    // "aa" appears at positions 0, 1, 2 (overlapping)
    try testing.expectEqual(@as(usize, 3), positions.len);
    
    std.mem.sort(usize, positions, {}, std.sort.asc(usize));
    try testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, positions);
}

test "suffix array: integer type" {
    const text = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    var result = try buildSuffixArray(i32, &text, testing.allocator);
    defer result.deinit();
    
    try testing.expectEqual(text.len, result.sa.len);
    // Verify all indices are valid
    for (result.sa) |idx| {
        try testing.expect(idx < text.len);
    }
}

test "memory safety: multiple allocations" {
    for (0..10) |_| {
        const text = "abracadabra";
        var sa_result = try buildSuffixArray(u8, text, testing.allocator);
        defer sa_result.deinit();
        
        var lcp_result = try buildLCP(u8, text, sa_result.sa, sa_result.rank, testing.allocator);
        defer lcp_result.deinit();
        
        const positions = try search(u8, text, sa_result.sa, "abra", testing.allocator);
        defer testing.allocator.free(positions);
    }
}
