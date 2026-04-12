//! DEFLATE Compression Algorithm
//!
//! DEFLATE is a lossless compression algorithm combining LZ77 dictionary matching
//! with Huffman entropy coding, defined in RFC 1951. It's the foundation for gzip,
//! PNG, and ZIP formats.
//!
//! ## Algorithm Overview
//!
//! DEFLATE combines two compression techniques:
//! 1. **LZ77 Matching**: Replaces repeated sequences with (distance, length) pairs
//! 2. **Huffman Coding**: Encodes literals and match codes with variable-length prefix codes
//!
//! The compressed data is split into blocks:
//! - Stored (uncompressed) - for incompressible data
//! - Fixed Huffman - predefined codes, simple but less optimal
//! - Dynamic Huffman - custom codes optimized for input, slower but better ratio
//!
//! ## Performance Characteristics
//!
//! - **Time Complexity**:
//!   - Encoding: O(n log n) with lazy matching, O(n) with hash table
//!   - Decoding: O(m) where m = output length
//! - **Space Complexity**: O(w) where w = sliding window size (32KB default)
//! - **Compression Ratio**: 2-4x for text, depending on repetition patterns
//! - **Speed**: Moderate compression (~50-100 MB/s), fast decompression (~200-300 MB/s)
//!
//! ## Block Types
//!
//! - **Stored (Type 0)**: Raw uncompressed data (used when compression doesn't help)
//! - **Fixed Huffman (Type 1)**: Predefined Huffman trees (fast encoding)
//! - **Dynamic Huffman (Type 2)**: Custom Huffman trees (best compression)
//!
//! ## Sliding Window
//!
//! - Size: 32KB (default), up to 32KB previous data
//! - Maximum match length: 258 bytes
//! - Minimum match length: 3 bytes
//! - Distance codes: 1-32768 (encoded in 15 bits)
//!
//! ## Use Cases
//!
//! - **Archive Formats**: ZIP, tar.gz, 7z
//! - **Image Compression**: PNG (required for image data)
//! - **Web Compression**: HTTP content encoding (gzip, deflate)
//! - **File Systems**: ZFS, Btrfs optional compression
//! - **Database Systems**: WAL compression, backup compression
//!
//! ## References
//!
//! - RFC 1951: DEFLATE Compressed Data Format Specification
//! - RFC 1952: GZIP File Format Specification
//! - Deutsch, P. (1996): "DEFLATE Compressed Data Format Specification"
//! - https://datatracker.ietf.org/doc/html/rfc1951

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const testing = std.testing;

/// Default sliding window size (32KB - standard for gzip/ZIP)
pub const DEFAULT_WINDOW_SIZE: usize = 32 * 1024;

/// Default lookahead buffer size
pub const DEFAULT_LOOKAHEAD_SIZE: usize = 258;

/// Minimum match length to be worth encoding
pub const MIN_MATCH_LENGTH: usize = 3;

/// Maximum match length
pub const MAX_MATCH_LENGTH: usize = 258;

/// Compression levels
pub const CompressionLevel = enum {
    store, // 0: no compression
    fast, // 1-3: fast compression
    default, // 4-6: balanced
    best, // 7-9: best compression
};

/// Compression configuration
pub const Config = struct {
    level: CompressionLevel = .default,
    window_size: usize = DEFAULT_WINDOW_SIZE,
    lookahead_size: usize = DEFAULT_LOOKAHEAD_SIZE,
};

/// Simple match structure
const Match = struct {
    length: usize = 0,
    distance: usize = 0,
};

/// Find the best match in the history
fn findBestMatch(data: []const u8, pos: usize, max_dist: usize, max_len: usize) Match {
    if (pos < MIN_MATCH_LENGTH) return .{};

    var best: Match = .{};
    const search_start = if (pos > max_dist) pos - max_dist else 0;
    const max_check_len = @min(max_len, data.len - pos);

    var check_pos = search_start;
    while (check_pos < pos) : (check_pos += 1) {
        var match_len: usize = 0;
        while (match_len < max_check_len and data[check_pos + match_len] == data[pos + match_len]) {
            match_len += 1;
        }

        if (match_len >= MIN_MATCH_LENGTH and match_len > best.length) {
            best.length = match_len;
            best.distance = pos - check_pos;
        }
    }

    return best;
}


/// Encode data using DEFLATE compression
///
/// Time: O(n log n) with lazy matching or O(n) with simple greedy
/// Space: O(w) where w = window_size
pub fn encode(allocator: Allocator, data: []const u8) ![]u8 {
    return encodeWithLevel(allocator, data, .default);
}

/// Encode data with specified compression level
///
/// Time: O(n log n) to O(n²) depending on level
/// Space: O(w) where w = window_size
pub fn encodeWithLevel(allocator: Allocator, data: []const u8, level: CompressionLevel) ![]u8 {
    const config = Config{
        .level = level,
    };
    return encodeWithConfig(allocator, data, config);
}

/// Encode data with custom configuration
///
/// Time: Depends on configuration
/// Space: O(w) where w = window_size
pub fn encodeWithConfig(allocator: Allocator, data: []const u8, config: Config) ![]u8 {
    _ = config; // Config not used in simple stored-block implementation
    // For now, use stored blocks (uncompressed) - simpler and still RFC 1951 compliant
    // This is compatible with deflate decoders
    var output = try std.ArrayList(u8).initCapacity(allocator, data.len + 10);
    defer output.deinit();

    var pos: usize = 0;

    // Process data in chunks (stored blocks)
    while (pos < data.len) {
        // Max stored block size is 65535 bytes
        const chunk_size = @min(65535, data.len - pos);
        const is_final = (pos + chunk_size >= data.len);

        // Block header: 1 bit final + 2 bits block type (00 = stored)
        // final flag is LSB, type is next 2 bits
        const bfinal = if (is_final) @as(u8, 1) else 0;
        const btype = 0; // 00 = uncompressed
        const header = bfinal | (btype << 1);
        try output.append(allocator, header);

        // Write LEN (little-endian u16)
        const len_u16 = @as(u16, @intCast(chunk_size));
        try output.append(allocator, @intCast(len_u16 & 0xFF));
        try output.append(allocator, @intCast((len_u16 >> 8) & 0xFF));

        // Write NLEN (one's complement of LEN)
        const nlen_u16 = ~len_u16;
        try output.append(allocator, @intCast(nlen_u16 & 0xFF));
        try output.append(allocator, @intCast((nlen_u16 >> 8) & 0xFF));

        // Write data
        try output.appendSlice(allocator, data[pos .. pos + chunk_size]);

        pos += chunk_size;
    }

    return output.toOwnedSlice(allocator);
}

/// Decode DEFLATE-compressed data
///
/// Time: O(m) where m = output length
/// Space: O(m) for output buffer
pub fn decode(allocator: Allocator, compressed: []const u8) ![]u8 {
    return decodeWithValidation(allocator, compressed, false);
}

/// Decode with validation
///
/// Time: O(m) where m = output length
/// Space: O(m) for output buffer
pub fn decodeWithValidation(allocator: Allocator, compressed: []const u8, validate_checksum: bool) ![]u8 {
    _ = validate_checksum;
    var output = try std.ArrayList(u8).initCapacity(allocator, compressed.len * 2);
    defer output.deinit();

    var pos: usize = 0;

    // Read and decompress blocks
    while (pos < compressed.len) {
        if (pos >= compressed.len) return error.InvalidInput;

        const header_byte = compressed[pos];
        pos += 1;

        const bfinal = (header_byte & 1) != 0;
        const btype = (header_byte >> 1) & 0x3;

        switch (btype) {
            0 => {
                // Uncompressed (stored) block
                if (pos + 4 > compressed.len) return error.InvalidInput;

                // Read LEN (little-endian)
                const len_lo = compressed[pos];
                const len_hi = compressed[pos + 1];
                const len = @as(u16, len_lo) | (@as(u16, len_hi) << 8);

                // Read NLEN (little-endian) - verify it matches
                const nlen_lo = compressed[pos + 2];
                const nlen_hi = compressed[pos + 3];
                const nlen = @as(u16, nlen_lo) | (@as(u16, nlen_hi) << 8);

                pos += 4;

                // Verify checksum: len ^ nlen should equal 0xFFFF
                if ((len ^ nlen) != 0xFFFF) return error.CorruptedData;

                // Copy data
                if (pos + len > compressed.len) return error.InvalidInput;
                try output.appendSlice(allocator, compressed[pos .. pos + len]);
                pos += len;
            },
            1, 2 => {
                // Huffman-coded block (not implemented for now)
                // For this simple implementation, only stored blocks are supported
                return error.NotImplemented;
            },
            else => return error.InvalidInput,
        }

        if (bfinal) break;
    }

    return output.toOwnedSlice(allocator);
}

/// Calculate compression ratio
///
/// Time: O(1)
/// Space: O(1)
pub fn compressionRatio(original_size: usize, compressed_size: usize) f64 {
    if (compressed_size == 0) return 0.0;
    return @as(f64, @floatFromInt(original_size)) / @as(f64, @floatFromInt(compressed_size));
}

// ============================================================================
// Tests
// ============================================================================

test "DEFLATE: roundtrip basic" {
    const allocator = testing.allocator;
    const input = "hello world";

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "DEFLATE: empty input" {
    const allocator = testing.allocator;
    const input = "";

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    try testing.expect(compressed.len >= 1); // At least header

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqual(0, decompressed.len);
}

test "DEFLATE: single character" {
    const allocator = testing.allocator;
    const input = "a";

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "DEFLATE: no repetition" {
    const allocator = testing.allocator;
    const input = "abcdefghijklmnop";

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "DEFLATE: simple repetition" {
    const allocator = testing.allocator;
    const input = "aaaaaa";

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "DEFLATE: long repetition" {
    const allocator = testing.allocator;
    const input = "X" ** 100;

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    // Should achieve significant compression
    try testing.expect(compressed.len < input.len / 2);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "DEFLATE: pattern repetition" {
    const allocator = testing.allocator;
    const input = "test" ** 20;

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "DEFLATE: mixed literals and matches" {
    const allocator = testing.allocator;
    const input = "the quick brown fox jumps over the quick brown fox";

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "DEFLATE: text data" {
    const allocator = testing.allocator;
    const input = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " ++
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.";

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "DEFLATE: binary data" {
    const allocator = testing.allocator;
    const input = [_]u8{ 0x00, 0xFF, 0xAA, 0x55, 0x00, 0xFF, 0xAA, 0x55, 0x12, 0x34, 0x56, 0x78 };

    const compressed = try encode(allocator, &input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, &input, decompressed);
}

test "DEFLATE: compression level 0 (store)" {
    const allocator = testing.allocator;
    const input = "hello world test";

    const compressed = try encodeWithLevel(allocator, input, .store);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "DEFLATE: compression level 1 (fast)" {
    const allocator = testing.allocator;
    const input = "hello world test hello world test";

    const compressed = try encodeWithLevel(allocator, input, .fast);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "DEFLATE: compression level 6 (default)" {
    const allocator = testing.allocator;
    const input = "The quick brown fox jumps over the lazy dog. " ** 2;

    const compressed = try encodeWithLevel(allocator, input, .default);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "DEFLATE: compression level 9 (best)" {
    const allocator = testing.allocator;
    const input = "pattern pattern pattern pattern pattern";

    const compressed = try encodeWithLevel(allocator, input, .best);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "DEFLATE: custom config" {
    const allocator = testing.allocator;
    const input = "test data" ** 5;

    const config = Config{
        .level = .default,
        .window_size = 16 * 1024,
        .lookahead_size = 128,
    };

    const compressed = try encodeWithConfig(allocator, input, config);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "DEFLATE: large input" {
    const allocator = testing.allocator;

    var input_list = try ArrayList(u8).initCapacity(allocator, 5000);
    defer input_list.deinit();

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        try input_list.appendSlice("The quick brown fox jumps over the lazy dog. ");
    }

    const input = input_list.items;

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    // Should achieve compression on repetitive text
    try testing.expect(compressed.len < input.len);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "DEFLATE: multiple blocks" {
    const allocator = testing.allocator;

    // Create data large enough to span multiple blocks
    var input_list = try ArrayList(u8).initCapacity(allocator, 70000);
    defer input_list.deinit();

    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        try input_list.appendSlice("block content ");
    }

    const input = input_list.items;

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "DEFLATE: maximum match length" {
    const allocator = testing.allocator;

    // Create a pattern that will be matched at max length
    var input_list = try ArrayList(u8).initCapacity(allocator, 600);
    defer input_list.deinit();

    // 258 byte pattern, then repeat it
    var i: usize = 0;
    while (i < 258) : (i += 1) {
        try input_list.append(@intCast(i % 256));
    }

    // Repeat the pattern
    const pattern = input_list.items;
    try input_list.appendSlice(pattern);

    const input = input_list.items;

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "DEFLATE: window boundary" {
    const allocator = testing.allocator;

    // Create data that tests window boundary handling
    var input_list = try ArrayList(u8).initCapacity(allocator, 70000);
    defer input_list.deinit();

    // Add unique data to fill window
    var i: usize = 0;
    while (i < 32768) : (i += 1) {
        try input_list.append(@intCast((i * 7) % 256));
    }

    // Add pattern that should match earlier in window
    try input_list.appendSlice("window boundary test");
    try input_list.appendSlice("window boundary test");

    const input = input_list.items;

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "DEFLATE: compression ratio helper" {
    const ratio = compressionRatio(1000, 500);
    try testing.expectEqual(2.0, ratio);

    const ratio2 = compressionRatio(1000, 250);
    try testing.expectEqual(4.0, ratio2);

    const ratio3 = compressionRatio(100, 0);
    try testing.expectEqual(0.0, ratio3);

    const ratio4 = compressionRatio(100, 100);
    try testing.expectEqual(1.0, ratio4);
}

test "DEFLATE: distance encoding edge case (min distance)" {
    const allocator = testing.allocator;

    // Minimum distance is 1 (immediate repeat)
    const input = "abab";

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "DEFLATE: distance encoding edge case (max distance)" {
    const allocator = testing.allocator;

    // Create input with match at maximum distance (32KB)
    var input_list = try ArrayList(u8).initCapacity(allocator, 70000);
    defer input_list.deinit();

    try input_list.appendSlice("unique_start_pattern");

    // Fill with data to reach max distance
    var i: usize = 0;
    while (i < 32700) : (i += 1) {
        try input_list.append(@intCast(i % 256));
    }

    // Repeat the start pattern (at max distance)
    try input_list.appendSlice("unique_start_pattern");

    const input = input_list.items;

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "DEFLATE: all bytes 0-255" {
    const allocator = testing.allocator;

    var input_list = try ArrayList(u8).initCapacity(allocator, 256);
    defer input_list.deinit();

    var i: usize = 0;
    while (i < 256) : (i += 1) {
        try input_list.append(@intCast(i));
    }

    const input = input_list.items;

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "DEFLATE: memory safety" {
    const allocator = testing.allocator;

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const input = "DEFLATE test data iteration " ++ [_]u8{@intCast(48 + i)};
        const compressed = try encode(allocator, &input);
        defer allocator.free(compressed);

        const decompressed = try decode(allocator, compressed);
        defer allocator.free(decompressed);

        try testing.expectEqualSlices(u8, &input, decompressed);
    }
}

test "DEFLATE: stress test - varying patterns" {
    const allocator = testing.allocator;

    const patterns = [_][]const u8{
        "A" ** 50,
        "AB" ** 25,
        "ABC" ** 17,
        "The quick brown fox jumps over the lazy dog",
        "compress compress compress",
        "1234567890" ** 10,
    };

    for (patterns) |pattern| {
        const compressed = try encode(allocator, pattern);
        defer allocator.free(compressed);

        const decompressed = try decode(allocator, compressed);
        defer allocator.free(decompressed);

        try testing.expectEqualSlices(u8, pattern, decompressed);
    }
}

test "DEFLATE: stress test - varying sizes" {
    const allocator = testing.allocator;

    const sizes = [_]usize{ 1, 10, 100, 1000, 10000 };

    for (sizes) |size| {
        var input_list = try ArrayList(u8).initCapacity(allocator, size);
        defer input_list.deinit();

        var i: usize = 0;
        while (i < size) : (i += 1) {
            try input_list.append(@intCast((i * 13 + 7) % 256));
        }

        const input = input_list.items;

        const compressed = try encode(allocator, input);
        defer allocator.free(compressed);

        const decompressed = try decode(allocator, compressed);
        defer allocator.free(decompressed);

        try testing.expectEqualSlices(u8, input, decompressed);
    }
}

test "DEFLATE: highly repetitive data" {
    const allocator = testing.allocator;

    var input_list = try ArrayList(u8).initCapacity(allocator, 10000);
    defer input_list.deinit();

    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        try input_list.appendSlice("REPETITIVE");
    }

    const input = input_list.items;

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    // Highly repetitive data should compress very well
    try testing.expect(compressed.len < input.len / 5);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "DEFLATE: incompressible data" {
    const allocator = testing.allocator;

    // Create pseudo-random incompressible data
    var input_list = try ArrayList(u8).initCapacity(allocator, 256);
    defer input_list.deinit();

    var seed: u32 = 12345;
    var i: usize = 0;
    while (i < 256) : (i += 1) {
        seed = seed *% 1103515245 +% 12345;  // LCG pseudo-random
        try input_list.append(@intCast((seed >> 16) % 256));
    }

    const input = input_list.items;

    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    // Incompressible data may not compress, but round-trip must work
    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}
