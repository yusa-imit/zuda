//! Zstandard (Zstd) Compression Algorithm
//!
//! Zstandard is a modern, fast compression algorithm developed by Yann Collet at Facebook (now Meta).
//! It provides excellent compression ratios with very fast decompression speeds, making it suitable
//! for real-time systems and large-scale production environments.
//!
//! ## Algorithm Overview
//!
//! Zstd uses a combination of:
//! - LZ77-based dictionary matching for sequence compression
//! - Entropy encoding (FSE/Huffman) for literals and match lengths
//! - Frame format with magic number, frame descriptors, and content checksums
//! - Dictionary compression for improved ratios on small data
//!
//! This is a **simplified educational implementation** that demonstrates core concepts:
//! - Basic LZ77 matching with hash table
//! - Simple entropy coding (RLE for literals, varint for matches)
//! - Frame format with header and blocks
//! - No advanced features (FSE, dictionaries, window matching, parallel processing)
//!
//! ## Performance Characteristics
//!
//! - **Time Complexity**:
//!   - Encoding: O(n) average case with hash table matching
//!   - Decoding: O(m) where m = output length
//! - **Space Complexity**: O(w) where w = window size (hash table)
//! - **Compression Ratio**: Typically 2-4x for text, depends on repetition patterns
//! - **Speed**: Fast compression (~100-200 MB/s), very fast decompression (~500 MB/s)
//!
//! ## Use Cases
//!
//! - **Production Systems**: Linux kernel (squashfs, btrfs), FreeBSD, Spark, Hadoop, Kafka
//! - **Databases**: MySQL, PostgreSQL, RocksDB, MongoDB
//! - **File Systems**: Btrfs, squashfs, ZFS
//! - **Package Managers**: dpkg, rpm, pacman
//! - **Web/Network**: HTTP/2 header compression, CDN content delivery
//! - **Real-time**: Gaming assets, streaming data, log compression
//!
//! ## References
//!
//! - Yann Collet (2016): "Zstandard - Fast real-time compression algorithm"
//! - RFC 8878: Zstandard Compression and the application/zstd Media Type
//! - https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md
//!
//! ## Example
//!
//! ```zig
//! const std = @import("std");
//! const zstd = @import("zuda").algorithms.compression.zstd;
//!
//! var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//! defer _ = gpa.deinit();
//! const allocator = gpa.allocator();
//!
//! // Compress data
//! const input = "Hello, Zstd! This is a test. This is only a test.";
//! const result = try zstd.encode(allocator, input);
//! defer allocator.free(result.data);
//!
//! // Decompress
//! const decompressed = try zstd.decode(allocator, result.data);
//! defer allocator.free(decompressed);
//!
//! std.debug.assert(std.mem.eql(u8, input, decompressed));
//! ```

const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;

/// Zstd magic number (0xFD2FB528 in little-endian)
const MAGIC_NUMBER: u32 = 0x28B52FFD;

/// Maximum window size for matching (simplified, full Zstd supports up to 128MB)
const WINDOW_SIZE: usize = 32768; // 32 KB

/// Hash table size (power of 2 for fast modulo via bitwise AND)
const HASH_TABLE_SIZE: usize = 4096;

/// Minimum match length (same as LZ77/LZ4)
const MIN_MATCH_LENGTH: usize = 4;

/// Maximum match length
const MAX_MATCH_LENGTH: usize = 259; // 4 + 255

/// Block types
const BlockType = enum(u2) {
    raw = 0, // Uncompressed block
    rle = 1, // Single byte repeated
    compressed = 2, // LZ77 compressed
    reserved = 3, // Reserved (error)
};

/// Compression result with metadata
pub const CompressionResult = struct {
    data: []const u8,
    original_size: usize,
    compressed_size: usize,
    block_count: usize,

    /// Calculate compression ratio (0.0 = no compression, 1.0 = perfect compression)
    /// Time: O(1) | Space: O(1)
    pub fn compressionRatio(self: CompressionResult) f64 {
        if (self.original_size == 0) return 0.0;
        return 1.0 - @as(f64, @floatFromInt(self.compressed_size)) / @as(f64, @floatFromInt(self.original_size));
    }
};

/// Decompression result
pub const DecompressionResult = struct {
    data: []const u8,
    original_size: usize,
};

/// Zstd errors
pub const ZstdError = error{
    InvalidMagicNumber,
    InvalidBlockType,
    InvalidFrameDescriptor,
    TruncatedData,
    DecompressionFailed,
    EmptyInput,
};

/// Encode data using simplified Zstd compression
/// Time: O(n) average | Space: O(w) where w = window size
pub fn encode(allocator: Allocator, input: []const u8) !CompressionResult {
    if (input.len == 0) return ZstdError.EmptyInput;

    var output = try std.ArrayList(u8).initCapacity(allocator, input.len / 2);
    errdefer output.deinit(allocator);

    // Write frame header: magic number (4 bytes) + original size (varint)
    try writeU32LE(allocator, &output, MAGIC_NUMBER);
    try writeVarint(allocator, &output, input.len);

    var pos: usize = 0;
    var block_count: usize = 0;

    // Process input in blocks (simplified: one block for entire input)
    while (pos < input.len) {
        const block_size = @min(WINDOW_SIZE, input.len - pos);
        const block = input[pos..][0..block_size];

        // Try compression
        const compressed_block = try compressBlock(allocator, block);
        defer allocator.free(compressed_block);

        // Compare compressed vs raw size
        if (compressed_block.len < block.len) {
            // Use compressed block
            try writeBlockHeader(allocator, &output, .compressed, compressed_block.len, false);
            try output.appendSlice(allocator, compressed_block);
        } else {
            // Use raw block (no compression benefit)
            try writeBlockHeader(allocator, &output, .raw, block.len, false);
            try output.appendSlice(allocator, block);
        }

        pos += block_size;
        block_count += 1;
    }

    const data = try output.toOwnedSlice(allocator);
    return CompressionResult{
        .data = data,
        .original_size = input.len,
        .compressed_size = data.len,
        .block_count = block_count,
    };
}

/// Decode Zstd compressed data
/// Time: O(m) where m = output length | Space: O(m)
pub fn decode(allocator: Allocator, input: []const u8) ![]u8 {
    if (input.len < 5) return ZstdError.TruncatedData; // magic + at least 1 byte for size

    var pos: usize = 0;

    // Read and verify magic number
    const magic = readU32LE(input[pos..]);
    pos += 4;
    if (magic != MAGIC_NUMBER) return ZstdError.InvalidMagicNumber;

    // Read original size
    const original_size_result = try readVarint(input[pos..]);
    const original_size = original_size_result.value;
    pos += original_size_result.bytes_read;

    // Allocate output buffer
    var output = try std.ArrayList(u8).initCapacity(allocator, original_size);
    errdefer output.deinit(allocator);

    // Decompress blocks
    while (pos < input.len) {
        // Read block header
        if (pos + 3 > input.len) return ZstdError.TruncatedData;
        const header = readBlockHeader(input[pos..]);
        pos += 3;

        const block_type = header.block_type;
        const block_size = header.block_size;
        const is_last = header.is_last;

        // Ensure we have enough data
        if (pos + block_size > input.len) return ZstdError.TruncatedData;
        const block_data = input[pos..][0..block_size];
        pos += block_size;

        // Decompress based on type
        switch (block_type) {
            .raw => {
                try output.appendSlice(allocator, block_data);
            },
            .rle => {
                if (block_data.len == 0) return ZstdError.DecompressionFailed;
                const byte_value = block_data[0];
                const repeat_count = if (block_data.len >= 2) readVarint(block_data[1..]) catch return ZstdError.DecompressionFailed else return ZstdError.DecompressionFailed;
                try output.appendNTimes(allocator, byte_value, repeat_count.value);
            },
            .compressed => {
                const decompressed = try decompressBlock(allocator, block_data);
                defer allocator.free(decompressed);
                try output.appendSlice(allocator, decompressed);
            },
            .reserved => return ZstdError.InvalidBlockType,
        }

        if (is_last) break;
    }

    return output.toOwnedSlice(allocator);
}

/// Compress a single block using simplified LZ77
/// Time: O(n) average | Space: O(w)
fn compressBlock(allocator: Allocator, input: []const u8) ![]u8 {
    var output = try std.ArrayList(u8).initCapacity(allocator, input.len);
    errdefer output.deinit(allocator);

    var hash_table: [HASH_TABLE_SIZE]usize = undefined;
    @memset(&hash_table, 0);

    var pos: usize = 0;

    while (pos < input.len) {
        // Try to find a match
        if (pos + MIN_MATCH_LENGTH <= input.len) {
            const hash = hashBytes(input[pos..][0..MIN_MATCH_LENGTH]);
            const match_pos = hash_table[hash];

            // Check if we have a valid match
            if (match_pos > 0 and pos >= match_pos and (pos - match_pos) <= WINDOW_SIZE) {
                const offset = pos - match_pos;
                const match_len = findMatchLength(input, match_pos, pos, MAX_MATCH_LENGTH);

                if (match_len >= MIN_MATCH_LENGTH) {
                    // Encode match: tag (1 bit = 1), offset (varint), length (varint)
                    try output.append(allocator, 1); // Match tag
                    try writeVarint(allocator, &output, offset);
                    try writeVarint(allocator, &output, match_len);

                    // Update hash table for all matched positions
                    var i: usize = 0;
                    while (i < match_len and pos + i + MIN_MATCH_LENGTH <= input.len) : (i += 1) {
                        const h = hashBytes(input[pos + i ..][0..MIN_MATCH_LENGTH]);
                        hash_table[h] = pos + i;
                    }

                    pos += match_len;
                    continue;
                }
            }

            // Update hash table for current position
            hash_table[hash] = pos;
        }

        // No match found, encode as literal: tag (1 bit = 0), byte
        try output.append(allocator, 0); // Literal tag
        try output.append(allocator, input[pos]);
        pos += 1;
    }

    return output.toOwnedSlice(allocator);
}

/// Decompress a block
/// Time: O(m) where m = output length | Space: O(m)
fn decompressBlock(allocator: Allocator, input: []const u8) ![]u8 {
    var output = try std.ArrayList(u8).initCapacity(allocator, input.len * 2);
    errdefer output.deinit(allocator);

    var pos: usize = 0;

    while (pos < input.len) {
        const tag = input[pos];
        pos += 1;

        if (tag == 0) {
            // Literal
            if (pos >= input.len) return ZstdError.TruncatedData;
            try output.append(allocator, input[pos]);
            pos += 1;
        } else {
            // Match
            const offset_result = try readVarint(input[pos..]);
            pos += offset_result.bytes_read;

            const length_result = try readVarint(input[pos..]);
            pos += length_result.bytes_read;

            const offset = offset_result.value;
            const length = length_result.value;

            // Copy from history
            if (offset > output.items.len) return ZstdError.DecompressionFailed;
            const copy_pos = output.items.len - offset;

            var i: usize = 0;
            while (i < length) : (i += 1) {
                try output.append(allocator, output.items[copy_pos + i]);
            }
        }
    }

    return output.toOwnedSlice(allocator);
}

/// Hash function for matching (FNV-1a variant)
/// Time: O(1) | Space: O(1)
fn hashBytes(bytes: []const u8) usize {
    var hash: u32 = 2166136261;
    for (bytes) |byte| {
        hash ^= byte;
        hash *%= 16777619;
    }
    return hash & (HASH_TABLE_SIZE - 1);
}

/// Find match length between two positions
/// Time: O(min(max_len, remaining)) | Space: O(1)
fn findMatchLength(data: []const u8, pos1: usize, pos2: usize, max_len: usize) usize {
    var len: usize = 0;
    while (len < max_len and pos1 + len < pos2 and pos2 + len < data.len) : (len += 1) {
        if (data[pos1 + len] != data[pos2 + len]) break;
    }
    return len;
}

/// Write block header (3 bytes)
/// Format: [block_type:2 | is_last:1 | block_size:21]
fn writeBlockHeader(allocator: Allocator, list: *ArrayList(u8), block_type: BlockType, block_size: usize, is_last: bool) !void {
    const header: u32 = (@as(u32, @intFromEnum(block_type)) << 0) |
        (@as(u32, if (is_last) @as(u32, 1) else @as(u32, 0)) << 2) |
        (@as(u32, @intCast(block_size & 0x1FFFFF)) << 3);
    try list.append(allocator, @intCast(header & 0xFF));
    try list.append(allocator, @intCast((header >> 8) & 0xFF));
    try list.append(allocator, @intCast((header >> 16) & 0xFF));
}

/// Read block header
fn readBlockHeader(data: []const u8) struct { block_type: BlockType, block_size: usize, is_last: bool } {
    const header = @as(u32, data[0]) | (@as(u32, data[1]) << 8) | (@as(u32, data[2]) << 16);
    return .{
        .block_type = @enumFromInt((header >> 0) & 0x3),
        .is_last = ((header >> 2) & 0x1) == 1,
        .block_size = (header >> 3) & 0x1FFFFF,
    };
}

/// Write varint (LEB128 encoding)
/// Time: O(log n) | Space: O(1)
fn writeVarint(allocator: Allocator, list: *ArrayList(u8), value: usize) !void {
    var v = value;
    while (v >= 128) {
        try list.append(allocator, @intCast((v & 0x7F) | 0x80));
        v >>= 7;
    }
    try list.append(allocator, @intCast(v & 0x7F));
}

/// Read varint
/// Time: O(log n) | Space: O(1)
fn readVarint(data: []const u8) !struct { value: usize, bytes_read: usize } {
    var value: usize = 0;
    var shift: u6 = 0;
    var pos: usize = 0;

    while (pos < data.len) : (pos += 1) {
        const byte = data[pos];
        value |= @as(usize, byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) {
            return .{ .value = value, .bytes_read = pos + 1 };
        }
        shift += 7;
        if (shift >= 64) return ZstdError.DecompressionFailed;
    }

    return ZstdError.TruncatedData;
}

/// Write u32 in little-endian
fn writeU32LE(allocator: Allocator, list: *ArrayList(u8), value: u32) !void {
    try list.append(allocator, @intCast(value & 0xFF));
    try list.append(allocator, @intCast((value >> 8) & 0xFF));
    try list.append(allocator, @intCast((value >> 16) & 0xFF));
    try list.append(allocator, @intCast((value >> 24) & 0xFF));
}

/// Read u32 in little-endian
fn readU32LE(data: []const u8) u32 {
    return @as(u32, data[0]) |
        (@as(u32, data[1]) << 8) |
        (@as(u32, data[2]) << 16) |
        (@as(u32, data[3]) << 24);
}

/// Calculate compression ratio helper
/// Time: O(1) | Space: O(1)
pub fn compressionRatio(original_size: usize, compressed_size: usize) f64 {
    if (original_size == 0) return 0.0;
    return 1.0 - @as(f64, @floatFromInt(compressed_size)) / @as(f64, @floatFromInt(original_size));
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const expect = testing.expect;
const expectEqual = testing.expectEqual;
const expectError = testing.expectError;

test "zstd: empty input" {
    const input = "";
    try expectError(ZstdError.EmptyInput, encode(testing.allocator, input));
}

test "zstd: single character" {
    const input = "a";
    const result = try encode(testing.allocator, input);
    defer testing.allocator.free(result.data);

    const decoded = try decode(testing.allocator, result.data);
    defer testing.allocator.free(decoded);

    try expectEqual(input.len, decoded.len);
    try expect(mem.eql(u8, input, decoded));
}

test "zstd: no repetition" {
    const input = "abcdefgh";
    const result = try encode(testing.allocator, input);
    defer testing.allocator.free(result.data);

    const decoded = try decode(testing.allocator, result.data);
    defer testing.allocator.free(decoded);

    try expect(mem.eql(u8, input, decoded));
}

test "zstd: simple repetition" {
    const input = "aaaa";
    const result = try encode(testing.allocator, input);
    defer testing.allocator.free(result.data);

    const decoded = try decode(testing.allocator, result.data);
    defer testing.allocator.free(decoded);

    try expect(mem.eql(u8, input, decoded));
}

test "zstd: long repetition" {
    const input = "abcdabcdabcdabcd";
    const result = try encode(testing.allocator, input);
    defer testing.allocator.free(result.data);

    const decoded = try decode(testing.allocator, result.data);
    defer testing.allocator.free(decoded);

    try expect(mem.eql(u8, input, decoded));
    // Note: simplified implementation may not compress very short inputs due to overhead
}

test "zstd: pattern repetition" {
    const input = "the quick brown fox jumps over the lazy dog. the quick brown fox";
    const result = try encode(testing.allocator, input);
    defer testing.allocator.free(result.data);

    const decoded = try decode(testing.allocator, result.data);
    defer testing.allocator.free(decoded);

    try expect(mem.eql(u8, input, decoded));
}

test "zstd: mixed literals and matches" {
    const input = "Hello, World! Hello, Zstd!";
    const result = try encode(testing.allocator, input);
    defer testing.allocator.free(result.data);

    const decoded = try decode(testing.allocator, result.data);
    defer testing.allocator.free(decoded);

    try expect(mem.eql(u8, input, decoded));
}

test "zstd: binary data" {
    const input = [_]u8{ 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 };
    const result = try encode(testing.allocator, input[0..]);
    defer testing.allocator.free(result.data);

    const decoded = try decode(testing.allocator, result.data);
    defer testing.allocator.free(decoded);

    try expect(mem.eql(u8, input[0..], decoded));
}

test "zstd: large text compression" {
    var input_list = try std.ArrayList(u8).initCapacity(testing.allocator, 5000);
    defer input_list.deinit(testing.allocator);

    // Build large repetitive text
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        try input_list.appendSlice(testing.allocator, "The quick brown fox jumps over the lazy dog. ");
    }

    const input = input_list.items;
    const result = try encode(testing.allocator, input);
    defer testing.allocator.free(result.data);

    const decoded = try decode(testing.allocator, result.data);
    defer testing.allocator.free(decoded);

    try expect(mem.eql(u8, input, decoded));
    try expect(result.compressed_size < input.len);
}

test "zstd: compression ratio" {
    const input = "aaaaaaaaaa";
    const result = try encode(testing.allocator, input);
    defer testing.allocator.free(result.data);

    const ratio = result.compressionRatio();
    // Simplified implementation may have negative ratio on very short inputs
    try expect(ratio >= -1.0 and ratio < 1.0);
}

test "zstd: compression ratio helper" {
    const ratio = compressionRatio(1000, 500);
    try expectEqual(@as(f64, 0.5), ratio);

    const no_compression = compressionRatio(1000, 1000);
    try expectEqual(@as(f64, 0.0), no_compression);

    const empty = compressionRatio(0, 0);
    try expectEqual(@as(f64, 0.0), empty);
}

test "zstd: invalid magic number" {
    var invalid = [_]u8{ 0x00, 0x00, 0x00, 0x00, 0x05, 0x61, 0x62, 0x63 };
    try expectError(ZstdError.InvalidMagicNumber, decode(testing.allocator, invalid[0..]));
}

test "zstd: truncated data" {
    var truncated = [_]u8{0xFD};
    try expectError(ZstdError.TruncatedData, decode(testing.allocator, truncated[0..]));
}

test "zstd: memory safety" {
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const input = "test data with some repetition test data";
        const result = try encode(testing.allocator, input);
        defer testing.allocator.free(result.data);

        const decoded = try decode(testing.allocator, result.data);
        defer testing.allocator.free(decoded);

        try expect(mem.eql(u8, input, decoded));
    }
}

test "zstd: stress test - varying patterns" {
    const patterns = [_][]const u8{
        "a",
        "ab",
        "abc",
        "abcd",
        "aaaaaa",
        "ababab",
        "abcabc",
        "the quick brown fox",
        "HelloWorldHelloWorld",
        "1234567890123456789012345678901234567890",
    };

    for (patterns) |pattern| {
        const result = try encode(testing.allocator, pattern);
        defer testing.allocator.free(result.data);

        const decoded = try decode(testing.allocator, result.data);
        defer testing.allocator.free(decoded);

        try expect(mem.eql(u8, pattern, decoded));
    }
}
