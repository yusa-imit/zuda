//! LZ77: dictionary compression with a sliding window, emitting `Token` values.
//!
//! Purpose: `encode` turns bytes into (offset, length, next byte) tokens, `decode` replays them,
//! including overlapping matches. Malformed tokens (offset beyond the output, zero offset with a
//! length) return `error.CorruptedData`; they are data errors, not asserted.
//! Ownership: `encode` returns an unmanaged list and `decode` an owned slice, both freed by
//! the caller with the allocator passed to the call; no state is kept between calls.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const testing = std.testing;

/// LZ77 - Dictionary-based lossless compression
///
/// Foundation for gzip, PNG (DEFLATE), and many compression formats.
/// Uses a sliding window to find repeated sequences and encodes them
/// as (offset, length, next_char) tuples.
///
/// Format: Each token is either:
/// - Literal: (0, 0, char) - single character
/// - Match: (offset, length, next_char) - reference to previous occurrence
///
/// Time: O(n × w) where n is input length, w is window size
/// Space: O(n) for output, O(w) for sliding window
/// Default window size (32KB - common for gzip)
pub const DEFAULT_WINDOW_SIZE: usize = 32 * 1024;

/// Default lookahead buffer size
pub const DEFAULT_LOOKAHEAD_SIZE: usize = 258;

/// Minimum match length to be worth encoding
pub const MIN_MATCH_LENGTH: usize = 3;

/// LZ77 compression token
pub const Token = struct {
    /// Offset back in the sliding window (0 = literal)
    offset: u16,
    /// Length of match (0 = literal)
    length: u16,
    /// Next character after match (or the literal)
    next_char: u8,
};

/// Encode data using LZ77 compression
///
/// Uses sliding window to find repeated sequences and encode them
/// as (offset, length, next_char) tuples.
///
/// Time: O(n × w) where n = data.len, w = window_size
/// Space: O(n) for tokens, O(w) for window
///
/// Example:
/// ```zig
/// const input = "ababababc";
/// var result = try encode(allocator, input, 256, 128);
/// defer result.deinit();
/// ```
pub fn encode(
    allocator: Allocator,
    data: []const u8,
    window_size: usize,
    lookahead_size: usize,
) Allocator.Error!std.ArrayList(Token) {
    assert(window_size > 0);
    assert(window_size <= std.math.maxInt(u16));
    assert(lookahead_size <= std.math.maxInt(u16));

    var tokens = std.ArrayList(Token).empty;
    errdefer tokens.deinit(allocator);

    if (data.len == 0) return tokens;

    var pos: usize = 0;
    while (pos < data.len) {
        const match = findLongestMatch(data, pos, window_size, lookahead_size);

        if (match.length >= MIN_MATCH_LENGTH) {
            // Encode as match
            // The match search stops one byte short of the end, so next_char exists.
            assert(pos + match.length < data.len);
            const next_char = data[pos + match.length];

            try tokens.append(allocator, .{
                .offset = @intCast(match.offset),
                .length = @intCast(match.length),
                .next_char = next_char,
            });

            pos += match.length + 1; // Include next_char
        } else {
            // Encode as literal
            try tokens.append(allocator, .{
                .offset = 0,
                .length = 0,
                .next_char = data[pos],
            });
            pos += 1;
        }
    }

    return tokens;
}

/// Decode LZ77-compressed data
///
/// Reconstructs original data from (offset, length, next_char) tokens.
///
/// Time: O(n) where n is output length
/// Space: O(n) for output buffer
///
/// Example:
/// ```zig
/// const tokens = try encode(allocator, input, 256, 128);
/// defer tokens.deinit();
/// const decoded = try decode(allocator, tokens.items);
/// defer allocator.free(decoded);
/// ```
pub fn decode(allocator: Allocator, tokens: []const Token) error{ CorruptedData, OutOfMemory }![]u8 {
    // Calculate output size
    var output_size: usize = 0;
    for (tokens) |token| {
        output_size += token.length + 1;
    }

    var output = try allocator.alloc(u8, output_size);
    errdefer allocator.free(output);

    var out_pos: usize = 0;
    for (tokens) |token| {
        if (token.length > 0) {
            // A match must point back into already decoded output.
            if (token.offset == 0 or token.offset > out_pos) return error.CorruptedData;
        }

        if (token.length > 0) {
            // Copy from earlier in output (match)
            const start = out_pos - token.offset;
            var i: usize = 0;
            while (i < token.length) : (i += 1) {
                output[out_pos] = output[start + i];
                out_pos += 1;
            }
        }

        // Append next_char (or literal if length=0)
        if (out_pos < output.len) {
            output[out_pos] = token.next_char;
            out_pos += 1;
        }
    }

    return output[0..out_pos];
}

/// Find longest match in sliding window
fn findLongestMatch(
    data: []const u8,
    pos: usize,
    window_size: usize,
    lookahead_size: usize,
) Match {
    const window_start = if (pos > window_size) pos - window_size else 0;
    // Keep one byte back for the token's next_char.
    const lookahead_end = @min(pos + lookahead_size, data.len - 1);

    var best_match = Match{ .offset = 0, .length = 0 };

    // Search window for matches
    var search_pos = window_start;
    while (search_pos < pos) : (search_pos += 1) {
        var match_len: usize = 0;

        // Count matching characters
        while (pos + match_len < lookahead_end and
            data[search_pos + match_len] == data[pos + match_len])
        {
            match_len += 1;
        }

        // Update best match
        if (match_len > best_match.length) {
            best_match.offset = pos - search_pos;
            best_match.length = match_len;
        }
    }

    return best_match;
}

const Match = struct {
    offset: usize,
    length: usize,
};

/// Calculate compression ratio
///
/// Time: O(1)
/// Space: O(1)
pub fn compressionRatio(original_size: usize, token_count: usize) f64 {
    if (token_count == 0) return 0.0;

    const compressed_size = token_count * @sizeOf(Token);
    return @as(f64, @floatFromInt(original_size)) / @as(f64, @floatFromInt(compressed_size));
}

// ============================================================================
// Tests
// ============================================================================

test "LZ77: corrupted tokens are rejected" {
    const offset_too_far = [_]Token{.{ .offset = 1, .length = 2, .next_char = 'a' }};
    try testing.expectError(error.CorruptedData, decode(testing.allocator, &offset_too_far));

    const zero_offset = [_]Token{
        .{ .offset = 0, .length = 0, .next_char = 'a' },
        .{ .offset = 0, .length = 1, .next_char = 'b' },
    };
    try testing.expectError(error.CorruptedData, decode(testing.allocator, &zero_offset));
}

test "LZ77: match ending at end of input does not append a byte" {
    const input = "abcabc";
    var tokens = try encode(testing.allocator, input, 256, 128);
    defer tokens.deinit(testing.allocator);

    const decoded = try decode(testing.allocator, tokens.items);
    defer testing.allocator.free(decoded);
    try testing.expectEqualSlices(u8, input, decoded);
}

test "LZ77: overlapping match encodes a run" {
    const input = "aaaaaaaa";
    var tokens = try encode(testing.allocator, input, 256, 128);
    defer tokens.deinit(testing.allocator);

    try testing.expect(tokens.items.len < 4);
    const decoded = try decode(testing.allocator, tokens.items);
    defer testing.allocator.free(decoded);
    try testing.expectEqualSlices(u8, input, decoded);
}

test "LZ77: simple literal encoding" {
    const input = "abc";
    var tokens = try encode(testing.allocator, input, 256, 128);
    defer tokens.deinit(testing.allocator);

    // All literals (no repeats)
    try testing.expectEqual(@as(usize, 3), tokens.items.len);
    for (tokens.items) |token| {
        try testing.expectEqual(@as(u16, 0), token.offset);
        try testing.expectEqual(@as(u16, 0), token.length);
    }
}

test "LZ77: simple pattern matching" {
    const input = "ababab";
    var tokens = try encode(testing.allocator, input, 256, 128);
    defer tokens.deinit(testing.allocator);

    // Should find "ab" pattern repeating
    try testing.expect(tokens.items.len < 6); // Better than all literals
}

test "LZ77: roundtrip encode-decode" {
    const original = "abcabcabcxyzxyzxyz";
    var tokens = try encode(testing.allocator, original, 256, 128);
    defer tokens.deinit(testing.allocator);

    const decoded = try decode(testing.allocator, tokens.items);
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(u8, original, decoded);
}

test "LZ77: empty input" {
    const input: []const u8 = "";
    var tokens = try encode(testing.allocator, input, 256, 128);
    defer tokens.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 0), tokens.items.len);

    const decoded = try decode(testing.allocator, tokens.items);
    defer testing.allocator.free(decoded);
    try testing.expectEqual(@as(usize, 0), decoded.len);
}

test "LZ77: single character" {
    const input = "a";
    var tokens = try encode(testing.allocator, input, 256, 128);
    defer tokens.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), tokens.items.len);
    try testing.expectEqual(@as(u8, 'a'), tokens.items[0].next_char);
}

test "LZ77: all same character" {
    const input = "aaaaaaa";
    var tokens = try encode(testing.allocator, input, 256, 128);
    defer tokens.deinit(testing.allocator);

    // Should compress well (find long matches)
    try testing.expect(tokens.items.len < 7);

    const decoded = try decode(testing.allocator, tokens.items);
    defer testing.allocator.free(decoded);
    try testing.expectEqualSlices(u8, input, decoded);
}

test "LZ77: repeated pattern" {
    const pattern = "hello";
    const input = pattern ** 10; // "hellohellohello..."
    var tokens = try encode(testing.allocator, input, 256, 128);
    defer tokens.deinit(testing.allocator);

    // Should find pattern repeats
    try testing.expect(tokens.items.len < input.len);

    const decoded = try decode(testing.allocator, tokens.items);
    defer testing.allocator.free(decoded);
    try testing.expectEqualSlices(u8, input, decoded);
}

test "LZ77: no compression (random-like)" {
    const input = "abcdefghijklmnop"; // All unique
    var tokens = try encode(testing.allocator, input, 256, 128);
    defer tokens.deinit(testing.allocator);

    // Should be mostly literals
    try testing.expectEqual(@as(usize, input.len), tokens.items.len);
}

test "LZ77: small window size" {
    const input = "ababababab";
    var tokens = try encode(testing.allocator, input, 4, 8); // Small window
    defer tokens.deinit(testing.allocator);

    const decoded = try decode(testing.allocator, tokens.items);
    defer testing.allocator.free(decoded);
    try testing.expectEqualSlices(u8, input, decoded);
}

test "LZ77: match at window boundary" {
    const input = "abcdefabcdef";
    var tokens = try encode(testing.allocator, input, 6, 8);
    defer tokens.deinit(testing.allocator);

    const decoded = try decode(testing.allocator, tokens.items);
    defer testing.allocator.free(decoded);
    try testing.expectEqualSlices(u8, input, decoded);
}

test "LZ77: overlapping matches" {
    const input = "aaaaaa"; // Self-overlapping
    var tokens = try encode(testing.allocator, input, 256, 128);
    defer tokens.deinit(testing.allocator);

    const decoded = try decode(testing.allocator, tokens.items);
    defer testing.allocator.free(decoded);
    try testing.expectEqualSlices(u8, input, decoded);
}

test "LZ77: compression ratio" {
    const input = "abcabcabcabcabc";
    var tokens = try encode(testing.allocator, input, 256, 128);
    defer tokens.deinit(testing.allocator);

    const ratio = compressionRatio(input.len, tokens.items.len);
    // Should compress (ratio < 1 in token count, but > 1 in bytes for good compression)
    try testing.expect(ratio > 0.0);
}

test "LZ77: long repeated sequence" {
    const allocator = testing.allocator;
    const base = "0123456789";
    const repetitions = 100;

    var input_list = std.ArrayList(u8).empty;
    defer input_list.deinit(allocator);

    var i: usize = 0;
    while (i < repetitions) : (i += 1) {
        try input_list.appendSlice(testing.allocator, base);
    }

    var tokens = try encode(allocator, input_list.items, 1024, 256);
    defer tokens.deinit(allocator);

    // Should compress significantly
    try testing.expect(tokens.items.len < input_list.items.len / 2);

    const decoded = try decode(allocator, tokens.items);
    defer allocator.free(decoded);
    try testing.expectEqualSlices(u8, input_list.items, decoded);
}

test "LZ77: binary data" {
    const input = [_]u8{ 0, 1, 2, 0, 1, 2, 0, 1, 2 };
    var tokens = try encode(testing.allocator, &input, 256, 128);
    defer tokens.deinit(testing.allocator);

    const decoded = try decode(testing.allocator, tokens.items);
    defer testing.allocator.free(decoded);
    try testing.expectEqualSlices(u8, &input, decoded);
}

test "LZ77: memory safety" {
    const input = "testdata";
    var tokens = try encode(testing.allocator, input, 256, 128);
    defer tokens.deinit(testing.allocator);

    const decoded = try decode(testing.allocator, tokens.items);
    defer testing.allocator.free(decoded);

    // No memory leaks detected by testing.allocator
}
