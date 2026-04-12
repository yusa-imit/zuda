const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Snappy Compression Algorithm
///
/// Fast compression algorithm developed by Google (2011).
/// Prioritizes speed over compression ratio.
///
/// Algorithm: LZ77-style with fixed Huffman codes and block structure.
///
/// Properties:
/// - Very fast compression (~250 MB/s) and decompression (~500 MB/s)
/// - Moderate compression ratio (typically 1.5x-2x)
/// - Stream-oriented format with length-prefixed chunks
/// - No entropy coding (unlike LZ4 which uses token-based encoding)
///
/// Use cases:
/// - Google BigTable, LevelDB, RocksDB
/// - Apache Kafka, Hadoop
/// - Protocol Buffers over network
/// - In-memory data compression
///
/// Format:
/// - Varint-encoded uncompressed length
/// - Sequence of chunks:
///   * Literal: tag (00) + length + data
///   * Copy (1-byte offset): tag (01) + length + 1-byte offset
///   * Copy (2-byte offset): tag (10) + length + 2-byte offset
///   * Copy (4-byte offset): tag (11) + length + 4-byte offset
///
/// Trade-offs:
/// - vs LZ4: Snappy is simpler, LZ4 is often faster on modern CPUs
/// - vs Deflate: Much faster, lower compression ratio
/// - vs Zstd: Much faster, much lower compression ratio
///
/// Reference: Google Snappy specification (2011)

/// Encode data using Snappy compression.
///
/// Time: O(n) average where n = input length
/// Space: O(n) for output buffer + O(1) for hash table
///
/// Returns compressed data. Caller owns returned slice.
pub fn encode(allocator: Allocator, input: []const u8) (Allocator.Error || error{EmptyInput})![]u8 {
    if (input.len == 0) return error.EmptyInput;

    // Allocate output buffer (worst case: input.len + varint overhead + chunk overhead)
    // Worst case: no compression, all literals with chunk headers
    const max_output = input.len + 10 + (input.len / 60 + 1) * 4;
    var output = try allocator.alloc(u8, max_output);
    errdefer allocator.free(output);

    var pos: usize = 0;

    // Write uncompressed length as varint
    pos += putVarint(output[pos..], input.len);

    // Hash table for finding matches (16-bit hash -> input position)
    var hash_table = [_]u32{0} ** 4096;
    const hash_shift = 32 - 12; // 12-bit hash

    var input_pos: usize = 0;
    while (input_pos < input.len) {
        // Find longest match
        var match_offset: usize = 0;
        var match_length: usize = 0;

        if (input_pos + 4 <= input.len) {
            const hash = hashBytes(input[input_pos..][0..4]);
            const hash_idx = hash >> hash_shift;
            const candidate_pos = hash_table[hash_idx];

            if (candidate_pos > 0 and input_pos >= candidate_pos and input_pos - candidate_pos <= 65535) {
                const offset = input_pos - candidate_pos;
                const len = findMatchLength(input, candidate_pos, input_pos);
                if (len >= 4) {
                    match_offset = offset;
                    match_length = len;
                }
            }

            hash_table[hash_idx] = @intCast(input_pos);
        }

        if (match_length >= 4) {
            // Emit copy chunk
            pos += putCopyChunk(output[pos..], match_offset, match_length);
            input_pos += match_length;
        } else {
            // Emit literal chunk
            const literal_start = input_pos;
            var literal_len: usize = 1;

            // Accumulate consecutive literals (max 60 bytes per chunk)
            while (literal_len < 60 and input_pos + literal_len < input.len) {
                literal_len += 1;
            }

            pos += putLiteralChunk(output[pos..], input[literal_start..][0..literal_len]);
            input_pos += literal_len;
        }
    }

    return allocator.realloc(output, pos);
}

/// Decode Snappy compressed data.
///
/// Time: O(m) where m = output (uncompressed) length
/// Space: O(m) for output buffer
///
/// Returns decompressed data. Caller owns returned slice.
pub fn decode(allocator: Allocator, input: []const u8) (Allocator.Error || error{ InvalidInput, CorruptedData })![]u8 {
    if (input.len == 0) return error.InvalidInput;

    // Read uncompressed length
    const varint_result = getVarint(input) catch return error.InvalidInput;
    const uncompressed_len = varint_result.value;
    var pos = varint_result.bytes_read;

    var output = try allocator.alloc(u8, uncompressed_len);
    errdefer allocator.free(output);

    var output_pos: usize = 0;

    while (pos < input.len) {
        const tag = input[pos];
        pos += 1;

        const chunk_type = tag & 0x03;
        const chunk_len = (tag >> 2) + 1;

        switch (chunk_type) {
            0 => { // Literal
                if (pos + chunk_len > input.len) return error.CorruptedData;
                if (output_pos + chunk_len > output.len) return error.CorruptedData;
                @memcpy(output[output_pos..][0..chunk_len], input[pos..][0..chunk_len]);
                pos += chunk_len;
                output_pos += chunk_len;
            },
            1 => { // Copy with 1-byte offset
                if (pos >= input.len) return error.CorruptedData;
                const offset = @as(usize, input[pos]);
                pos += 1;
                if (offset == 0 or offset > output_pos) return error.CorruptedData;
                try copyWithOverlap(output, output_pos, offset, chunk_len);
                output_pos += chunk_len;
            },
            2 => { // Copy with 2-byte offset
                if (pos + 1 >= input.len) return error.CorruptedData;
                const offset = @as(usize, input[pos]) | (@as(usize, input[pos + 1]) << 8);
                pos += 2;
                if (offset == 0 or offset > output_pos) return error.CorruptedData;
                try copyWithOverlap(output, output_pos, offset, chunk_len);
                output_pos += chunk_len;
            },
            3 => { // Copy with 4-byte offset
                if (pos + 3 >= input.len) return error.CorruptedData;
                const offset = @as(usize, input[pos]) |
                    (@as(usize, input[pos + 1]) << 8) |
                    (@as(usize, input[pos + 2]) << 16) |
                    (@as(usize, input[pos + 3]) << 24);
                pos += 4;
                if (offset == 0 or offset > output_pos) return error.CorruptedData;
                try copyWithOverlap(output, output_pos, offset, chunk_len);
                output_pos += chunk_len;
            },
        }
    }

    if (output_pos != uncompressed_len) return error.CorruptedData;

    return output;
}

/// Encode with custom configuration.
///
/// Time: O(n) average
/// Space: O(n)
pub fn encodeWithConfig(allocator: Allocator, input: []const u8, config: Config) (Allocator.Error || error{EmptyInput})![]u8 {
    _ = config; // For future use
    return encode(allocator, input);
}

/// Decode with validation.
///
/// Time: O(m) where m = output length
/// Space: O(m)
pub fn decodeWithConfig(allocator: Allocator, input: []const u8, config: Config) (Allocator.Error || error{ InvalidInput, CorruptedData })![]u8 {
    _ = config; // For future use
    return decode(allocator, input);
}

/// Calculate compression ratio.
///
/// Time: O(1)
/// Space: O(1)
pub fn compressionRatio(original_size: usize, compressed_size: usize) f64 {
    if (compressed_size == 0) return 0.0;
    return @as(f64, @floatFromInt(original_size)) / @as(f64, @floatFromInt(compressed_size));
}

pub const Config = struct {
    // Future: compression level, block size, etc.
};

// --- Internal Helper Functions ---

fn putVarint(buf: []u8, value: usize) usize {
    var v = value;
    var pos: usize = 0;
    while (v >= 128) {
        buf[pos] = @intCast((v & 0x7F) | 0x80);
        v >>= 7;
        pos += 1;
    }
    buf[pos] = @intCast(v);
    return pos + 1;
}

fn getVarint(buf: []const u8) error{InvalidInput}!struct { value: usize, bytes_read: usize } {
    var value: usize = 0;
    var shift: u6 = 0;
    var pos: usize = 0;

    while (pos < buf.len) {
        const byte = buf[pos];
        value |= (@as(usize, byte & 0x7F) << shift);
        pos += 1;
        if (byte & 0x80 == 0) {
            return .{ .value = value, .bytes_read = pos };
        }
        shift += 7;
        if (shift >= 64) return error.InvalidInput;
    }

    return error.InvalidInput;
}

fn hashBytes(bytes: []const u8) u32 {
    std.debug.assert(bytes.len >= 4);
    const val = @as(u32, bytes[0]) |
        (@as(u32, bytes[1]) << 8) |
        (@as(u32, bytes[2]) << 16) |
        (@as(u32, bytes[3]) << 24);
    return val *% 0x1e35a7bd; // Multiplicative hash
}

fn findMatchLength(input: []const u8, pos1: usize, pos2: usize) usize {
    var len: usize = 0;
    while (pos1 + len < pos2 and pos2 + len < input.len and input[pos1 + len] == input[pos2 + len]) {
        len += 1;
    }
    return len;
}

fn putLiteralChunk(buf: []u8, literal: []const u8) usize {
    std.debug.assert(literal.len > 0 and literal.len <= 60);
    const tag = @as(u8, @intCast((literal.len - 1) << 2)) | 0x00;
    buf[0] = tag;
    @memcpy(buf[1..][0..literal.len], literal);
    return 1 + literal.len;
}

fn putCopyChunk(buf: []u8, offset: usize, length: usize) usize {
    std.debug.assert(length >= 4);
    const len_minus_1 = length - 1;

    if (offset <= 255 and len_minus_1 <= 63) {
        // 1-byte offset
        const tag = @as(u8, @intCast(len_minus_1 << 2)) | 0x01;
        buf[0] = tag;
        buf[1] = @intCast(offset);
        return 2;
    } else if (offset <= 65535 and len_minus_1 <= 63) {
        // 2-byte offset
        const tag = @as(u8, @intCast(len_minus_1 << 2)) | 0x02;
        buf[0] = tag;
        buf[1] = @intCast(offset & 0xFF);
        buf[2] = @intCast((offset >> 8) & 0xFF);
        return 3;
    } else {
        // 4-byte offset
        const tag = @as(u8, @intCast(@min(len_minus_1, 63) << 2)) | 0x03;
        buf[0] = tag;
        buf[1] = @intCast(offset & 0xFF);
        buf[2] = @intCast((offset >> 8) & 0xFF);
        buf[3] = @intCast((offset >> 16) & 0xFF);
        buf[4] = @intCast((offset >> 24) & 0xFF);
        return 5;
    }
}

fn copyWithOverlap(output: []u8, pos: usize, offset: usize, length: usize) error{CorruptedData}!void {
    if (pos + length > output.len) return error.CorruptedData;
    if (offset > pos) return error.CorruptedData;

    const src_start = pos - offset;
    var i: usize = 0;
    while (i < length) : (i += 1) {
        output[pos + i] = output[src_start + i];
    }
}

// --- Tests ---

test "snappy: empty input" {
    const allocator = testing.allocator;
    const input: []const u8 = "";
    try testing.expectError(error.EmptyInput, encode(allocator, input));
}

test "snappy: single character" {
    const allocator = testing.allocator;
    const input = "A";

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "snappy: no repetition" {
    const allocator = testing.allocator;
    const input = "ABCDEFGH";

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "snappy: simple repetition" {
    const allocator = testing.allocator;
    const input = "AAAABBBB";

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    try testing.expect(compressed.len < input.len);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "snappy: long repetition" {
    const allocator = testing.allocator;
    const input = "A" ** 100;

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    try testing.expect(compressed.len < input.len);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "snappy: pattern repetition" {
    const allocator = testing.allocator;
    const input = "ABCD" ** 25;

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    try testing.expect(compressed.len < input.len);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "snappy: mixed literals and matches" {
    const allocator = testing.allocator;
    const input = "Hello world! Hello world!";

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "snappy: binary data" {
    const allocator = testing.allocator;
    const input = [_]u8{ 0x00, 0xFF, 0xAA, 0x55, 0x00, 0xFF, 0xAA, 0x55 };

    const compressed = try encode(allocator, &input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, &input, decompressed);
}

test "snappy: large text with repetition" {
    const allocator = testing.allocator;
    var input_list = std.ArrayList(u8).init(allocator);
    defer input_list.deinit();

    const pattern = "The quick brown fox jumps over the lazy dog. ";
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        try input_list.appendSlice(pattern);
    }

    const input = input_list.items;

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    try testing.expect(compressed.len < input.len);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);

    const ratio = compressionRatio(input.len, compressed.len);
    try testing.expect(ratio > 1.0);
}

test "snappy: compression ratio helper" {
    try testing.expectEqual(@as(f64, 2.0), compressionRatio(100, 50));
    try testing.expectEqual(@as(f64, 1.5), compressionRatio(90, 60));
    try testing.expectEqual(@as(f64, 0.0), compressionRatio(100, 0));
}

test "snappy: corrupted data - truncated" {
    const allocator = testing.allocator;
    const input = "Hello, world!";
    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    // Truncate compressed data
    const truncated = compressed[0 .. compressed.len - 3];
    try testing.expectError(error.CorruptedData, decode(allocator, truncated));
}

test "snappy: invalid varint" {
    const allocator = testing.allocator;
    const invalid = [_]u8{ 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
    try testing.expectError(error.InvalidInput, decode(allocator, &invalid));
}

test "snappy: memory safety" {
    const allocator = testing.allocator;
    const input = "Snappy compression test with reasonable length for memory verification.";

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const compressed = try encode(allocator, input);
        defer allocator.free(compressed);

        const decompressed = try decode(allocator, compressed);
        defer allocator.free(decompressed);

        try testing.expectEqualSlices(u8, input, decompressed);
    }
}

test "snappy: stress test" {
    const allocator = testing.allocator;

    // Test various patterns
    const patterns = [_][]const u8{
        "A" ** 50,
        "AB" ** 25,
        "ABC" ** 17,
        "The quick brown fox jumps over the lazy dog",
        &[_]u8{ 0x00, 0x01, 0x02 } ** 20,
    };

    for (patterns) |pattern| {
        const compressed = try encode(allocator, pattern);
        defer allocator.free(compressed);

        const decompressed = try decode(allocator, compressed);
        defer allocator.free(decompressed);

        try testing.expectEqualSlices(u8, pattern, decompressed);
    }
}
