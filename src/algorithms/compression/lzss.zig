/// LZSS (Lempel-Ziv-Storer-Szymanski) Compression
///
/// LZSS is an improved variant of LZ77 that addresses a key inefficiency:
/// - LZ77 always outputs (offset, length, literal) triples
/// - LZSS uses flag bits to distinguish literals from match references
/// - Only outputs (offset, length) when the match is long enough to save space
/// - Uses 1-bit flags: 0 = literal byte, 1 = (offset, length) reference
///
/// ## Algorithm
///
/// **Encoding**:
/// - Maintain sliding window of last W bytes (default: 4096)
/// - Look-ahead buffer of L bytes (default: 18)
/// - For each position:
///   * Find longest match in window (length ≥ threshold, default: 3)
///   * If match found and saves space: output flag=1 + (offset, length)
///   * Otherwise: output flag=0 + literal byte
/// - Flag bits are packed into bytes (8 flags per byte)
///
/// **Decoding**:
/// - Read flag bits to determine literal vs. reference
/// - For literal (flag=0): copy byte directly
/// - For reference (flag=1): read (offset, length) and copy from window
///
/// ## Improvements over LZ77
///
/// - **Space Efficiency**: No overhead for single characters
/// - **Better Compression**: Flag bits add minimal overhead (~12.5% for literals)
/// - **Match Threshold**: Only encode matches that save space (length ≥ 3 typical)
/// - **Bit Packing**: Flags are bit-packed for efficiency
///
/// ## Time Complexity
///
/// - Encoding: O(n × w) where n = input length, w = window size
///   * Naive search: O(w) per position
///   * Can be optimized with hash table or suffix tree to O(n log w)
/// - Decoding: O(m) where m = output length
///
/// ## Space Complexity
///
/// - Encoding: O(w) for sliding window + O(L) for look-ahead buffer
/// - Decoding: O(w) for reconstruction window
/// - Output: Variable, typically 40-60% of input for text
///
/// ## Use Cases
///
/// - **General-purpose compression**: Better than LZ77 for most data
/// - **Embedded systems**: Simple decode, low memory (used in many ROM compressions)
/// - **Game assets**: Fast decompression, moderate compression ratio
/// - **Network protocols**: Used in various historical protocols
/// - **Archive formats**: Foundation for many compression tools (ARJ, LHA)
///
/// ## Configuration Parameters
///
/// - `window_size`: Sliding window size (default: 4096, typical: 2048-8192)
/// - `lookahead_size`: Look-ahead buffer (default: 18, typical: 16-32)
/// - `min_match_length`: Minimum match to encode (default: 3, typical: 2-4)
///
/// ## Format
///
/// ```
/// Compressed stream:
/// [flag_byte] [data_0..7]
///
/// flag_byte: 8 bits, one per following data item
/// - bit = 0: next item is 1 byte (literal)
/// - bit = 1: next item is 2 bytes (offset, length) packed
///
/// (offset, length) encoding:
/// - 12 bits for offset (0-4095 for window_size=4096)
/// - 4 bits for length (3-18 for min=3, lookahead=18)
/// - Packed as: [offset_high:4 | length:4] [offset_low:8]
/// ```
///
/// ## Example
///
/// ```zig
/// const lzss = @import("zuda").algorithms.compression.lzss;
///
/// // Encode
/// const data = "ABCABCABCABC";
/// var compressed = try lzss.encode(allocator, data);
/// defer compressed.deinit();
///
/// // Decode
/// const decompressed = try lzss.decode(allocator, compressed.items);
/// defer allocator.free(decompressed);
///
/// // Custom configuration
/// var result = try lzss.encodeWithConfig(allocator, data, .{
///     .window_size = 2048,
///     .lookahead_size = 16,
///     .min_match_length = 2,
/// });
/// defer result.deinit();
/// ```
///
/// ## Performance Notes
///
/// - Encoding is slower than RLE but faster than BWT
/// - Decoding is very fast (O(m), simple loop)
/// - Works well on text, moderate on binary data
/// - Compression ratio: typically 40-60% for text, 70-90% for binary
///
/// ## Trade-offs
///
/// vs. LZ77:
/// - Better space efficiency (no overhead for single chars)
/// - Slightly more complex encoding (flag bit management)
/// - Same decoding complexity
///
/// vs. LZW:
/// - LZSS: simpler, faster decode, better for streams
/// - LZW: better compression ratio, requires dictionary
///
/// vs. Deflate (LZ77+Huffman):
/// - LZSS: simpler, faster decode
/// - Deflate: better compression, widely supported

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

/// Configuration for LZSS compression
pub const Config = struct {
    /// Sliding window size (must be power of 2 for efficient masking)
    window_size: u16 = 4096,
    /// Look-ahead buffer size
    lookahead_size: u8 = 18,
    /// Minimum match length to encode (2-4 typical)
    min_match_length: u8 = 3,
};

/// Result of LZSS encoding
pub const EncodeResult = struct {
    data: []u8,
    allocator: Allocator,

    pub fn deinit(self: *EncodeResult) void {
        self.allocator.free(self.data);
    }
};

/// Encode data using LZSS compression
///
/// Time: O(n × w) where n = input length, w = window size
/// Space: O(w + L) where L = lookahead size
pub fn encode(allocator: Allocator, input: []const u8) !ArrayList(u8) {
    return encodeWithConfig(allocator, input, .{});
}

/// Encode data with custom LZSS configuration
///
/// Time: O(n × w)
/// Space: O(w + L)
pub fn encodeWithConfig(allocator: Allocator, input: []const u8, config: Config) !ArrayList(u8) {
    if (input.len == 0) {
        return ArrayList(u8).init(allocator);
    }

    var output = ArrayList(u8).init(allocator);
    errdefer output.deinit();

    var pos: usize = 0;

    while (pos < input.len) {
        // Collect up to 8 items for one flag byte
        var flag_byte: u8 = 0;
        var flag_pos: u3 = 0;
        var items = ArrayList(u8).init(allocator);
        defer items.deinit();

        while (flag_pos < 8 and pos < input.len) {
            // Find longest match in sliding window
            const match = findLongestMatch(input, pos, config);

            if (match.length >= config.min_match_length) {
                // Encode as reference
                flag_byte |= (@as(u8, 1) << flag_pos);

                // Pack (offset, length) into 2 bytes
                // offset: 12 bits (0-4095), length: 4 bits (encoded as length - min_match_length)
                const encoded_length = match.length - config.min_match_length;
                const byte1 = @as(u8, @intCast((match.offset >> 4) & 0xFF));
                const byte2 = @as(u8, @intCast(((match.offset & 0x0F) << 4) | (encoded_length & 0x0F)));

                try items.append(byte1);
                try items.append(byte2);

                pos += match.length;
            } else {
                // Encode as literal
                try items.append(input[pos]);
                pos += 1;
            }

            flag_pos += 1;
        }

        // Write flag byte followed by items
        try output.append(flag_byte);
        try output.appendSlice(items.items);
    }

    return output;
}

/// Match found in sliding window
const Match = struct {
    offset: u16,
    length: u8,
};

/// Find longest match in sliding window
///
/// Time: O(w × L) where w = window size, L = lookahead size
/// Space: O(1)
fn findLongestMatch(input: []const u8, pos: usize, config: Config) Match {
    const window_start = if (pos > config.window_size) pos - config.window_size else 0;
    const lookahead_end = @min(pos + config.lookahead_size, input.len);

    var best_match = Match{ .offset = 0, .length = 0 };

    // Search backward in window for matches
    var search_pos = window_start;
    while (search_pos < pos) : (search_pos += 1) {
        var match_len: u8 = 0;

        // Count matching bytes
        while (search_pos + match_len < pos and
            pos + match_len < lookahead_end and
            input[search_pos + match_len] == input[pos + match_len])
        {
            match_len += 1;
        }

        // Update best match if longer
        if (match_len > best_match.length) {
            best_match.offset = @intCast(pos - search_pos);
            best_match.length = match_len;
        }
    }

    return best_match;
}

/// Decode LZSS compressed data
///
/// Time: O(m) where m = output length
/// Space: O(w) for reconstruction window
pub fn decode(allocator: Allocator, compressed: []const u8) ![]u8 {
    return decodeWithConfig(allocator, compressed, .{});
}

/// Decode LZSS compressed data with custom configuration
///
/// Time: O(m)
/// Space: O(w)
pub fn decodeWithConfig(allocator: Allocator, compressed: []const u8, config: Config) ![]u8 {
    if (compressed.len == 0) {
        return try allocator.alloc(u8, 0);
    }

    var output = ArrayList(u8).init(allocator);
    errdefer output.deinit();

    var pos: usize = 0;

    while (pos < compressed.len) {
        // Read flag byte
        const flag_byte = compressed[pos];
        pos += 1;

        // Process up to 8 items
        var flag_pos: u3 = 0;
        while (flag_pos < 8 and pos < compressed.len) {
            const is_reference = (flag_byte & (@as(u8, 1) << flag_pos)) != 0;

            if (is_reference) {
                // Decode reference (offset, length)
                if (pos + 1 >= compressed.len) break;

                const byte1 = compressed[pos];
                const byte2 = compressed[pos + 1];
                pos += 2;

                // Unpack (offset, length)
                const offset = (@as(u16, byte1) << 4) | (@as(u16, byte2) >> 4);
                const encoded_length = byte2 & 0x0F;
                const length = encoded_length + config.min_match_length;

                // Copy from window
                const copy_start = output.items.len - offset;
                var i: u8 = 0;
                while (i < length) : (i += 1) {
                    try output.append(output.items[copy_start + i]);
                }
            } else {
                // Literal byte
                try output.append(compressed[pos]);
                pos += 1;
            }

            flag_pos += 1;
        }
    }

    return output.toOwnedSlice();
}

/// Compute compression ratio
///
/// Time: O(1)
/// Space: O(1)
pub fn compressionRatio(original_len: usize, compressed_len: usize) f64 {
    if (compressed_len == 0) return 0.0;
    return @as(f64, @floatFromInt(original_len)) / @as(f64, @floatFromInt(compressed_len));
}

// Tests

const testing = std.testing;
const expect = testing.expect;
const expectEqual = testing.expectEqual;
const expectEqualSlices = testing.expectEqualSlices;

test "LZSS: empty input" {
    const allocator = testing.allocator;

    var compressed = try encode(allocator, "");
    defer compressed.deinit();
    try expectEqual(@as(usize, 0), compressed.items.len);

    const decompressed = try decode(allocator, compressed.items);
    defer allocator.free(decompressed);
    try expectEqual(@as(usize, 0), decompressed.len);
}

test "LZSS: single character" {
    const allocator = testing.allocator;

    const input = "A";
    var compressed = try encode(allocator, input);
    defer compressed.deinit();

    const decompressed = try decode(allocator, compressed.items);
    defer allocator.free(decompressed);

    try expectEqualSlices(u8, input, decompressed);
}

test "LZSS: no repetition (all literals)" {
    const allocator = testing.allocator;

    const input = "ABCDEFGH";
    var compressed = try encode(allocator, input);
    defer compressed.deinit();

    const decompressed = try decode(allocator, compressed.items);
    defer allocator.free(decompressed);

    try expectEqualSlices(u8, input, decompressed);
    // Should be slightly larger due to flag bytes
}

test "LZSS: simple repetition" {
    const allocator = testing.allocator;

    const input = "ABCABCABC";
    var compressed = try encode(allocator, input);
    defer compressed.deinit();

    const decompressed = try decode(allocator, compressed.items);
    defer allocator.free(decompressed);

    try expectEqualSlices(u8, input, decompressed);
    // Should be smaller than input
    try expect(compressed.items.len < input.len);
}

test "LZSS: long repetition" {
    const allocator = testing.allocator;

    const input = "AAAAAAAAAAAAAAAA"; // 16 A's
    var compressed = try encode(allocator, input);
    defer compressed.deinit();

    const decompressed = try decode(allocator, compressed.items);
    defer allocator.free(decompressed);

    try expectEqualSlices(u8, input, decompressed);
    // Should achieve good compression
    try expect(compressed.items.len < input.len);
}

test "LZSS: pattern repetition" {
    const allocator = testing.allocator;

    const input = "ABCDABCDABCDABCD";
    var compressed = try encode(allocator, input);
    defer compressed.deinit();

    const decompressed = try decode(allocator, compressed.items);
    defer allocator.free(decompressed);

    try expectEqualSlices(u8, input, decompressed);
}

test "LZSS: mixed literals and references" {
    const allocator = testing.allocator;

    const input = "Hello world! Hello again!";
    var compressed = try encode(allocator, input);
    defer compressed.deinit();

    const decompressed = try decode(allocator, compressed.items);
    defer allocator.free(decompressed);

    try expectEqualSlices(u8, input, decompressed);
}

test "LZSS: overlapping matches" {
    const allocator = testing.allocator;

    const input = "ABABABABABAB"; // Overlapping AB pattern
    var compressed = try encode(allocator, input);
    defer compressed.deinit();

    const decompressed = try decode(allocator, compressed.items);
    defer allocator.free(decompressed);

    try expectEqualSlices(u8, input, decompressed);
}

test "LZSS: binary data" {
    const allocator = testing.allocator;

    const input = &[_]u8{ 0x00, 0x01, 0x02, 0x00, 0x01, 0x02, 0xFF, 0xFE };
    var compressed = try encode(allocator, input);
    defer compressed.deinit();

    const decompressed = try decode(allocator, compressed.items);
    defer allocator.free(decompressed);

    try expectEqualSlices(u8, input, decompressed);
}

test "LZSS: custom config - smaller window" {
    const allocator = testing.allocator;

    const input = "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGH";
    const config = Config{
        .window_size = 256,
        .lookahead_size = 16,
        .min_match_length = 3,
    };

    var compressed = try encodeWithConfig(allocator, input, config);
    defer compressed.deinit();

    const decompressed = try decodeWithConfig(allocator, compressed.items, config);
    defer allocator.free(decompressed);

    try expectEqualSlices(u8, input, decompressed);
}

test "LZSS: custom config - min match 2" {
    const allocator = testing.allocator;

    const input = "ABABCDCDEFEFGH";
    const config = Config{
        .window_size = 4096,
        .lookahead_size = 18,
        .min_match_length = 2, // Encode 2-byte matches
    };

    var compressed = try encodeWithConfig(allocator, input, config);
    defer compressed.deinit();

    const decompressed = try decodeWithConfig(allocator, compressed.items, config);
    defer allocator.free(decompressed);

    try expectEqualSlices(u8, input, decompressed);
}

test "LZSS: large text" {
    const allocator = testing.allocator;

    // Create large repetitive text
    var input = ArrayList(u8).init(allocator);
    defer input.deinit();

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        try input.appendSlice("Lorem ipsum dolor sit amet. ");
    }

    var compressed = try encode(allocator, input.items);
    defer compressed.deinit();

    const decompressed = try decode(allocator, compressed.items);
    defer allocator.free(decompressed);

    try expectEqualSlices(u8, input.items, decompressed);
    // Should achieve significant compression
    try expect(compressed.items.len < input.items.len);
}

test "LZSS: compression ratio helper" {
    const ratio = compressionRatio(1000, 400);
    try expect(ratio > 2.4 and ratio < 2.6); // ~2.5
}

test "LZSS: edge case - maximum lookahead match" {
    const allocator = testing.allocator;

    // Create pattern longer than lookahead buffer
    const pattern = "0123456789ABCDEFGHIJ"; // 20 bytes > default lookahead (18)
    var input = ArrayList(u8).init(allocator);
    defer input.deinit();

    try input.appendSlice(pattern);
    try input.appendSlice(pattern); // Repeat

    var compressed = try encode(allocator, input.items);
    defer compressed.deinit();

    const decompressed = try decode(allocator, compressed.items);
    defer allocator.free(decompressed);

    try expectEqualSlices(u8, input.items, decompressed);
}

test "LZSS: memory safety - 10 iterations" {
    const allocator = testing.allocator;

    const input = "Test data with some repetition. Test data again.";

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var compressed = try encode(allocator, input);
        defer compressed.deinit();

        const decompressed = try decode(allocator, compressed.items);
        defer allocator.free(decompressed);

        try expectEqualSlices(u8, input, decompressed);
    }
}
