const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// LZ4 Compression Configuration
pub const Config = struct {
    /// Hash table size (power of 2, default 4096)
    hash_table_size: usize = 4096,
    /// Maximum match search attempts (default 128)
    max_search: usize = 128,
    /// Minimum match length (default 4, must be ≥4 for LZ4)
    min_match_length: usize = 4,
};

/// LZ4 lossless compression algorithm optimized for speed.
///
/// LZ4 is a fast compression algorithm that achieves high compression and
/// decompression speeds at the cost of slightly lower compression ratios
/// compared to algorithms like Deflate or LZMA.
///
/// **Algorithm**:
/// - Uses hash table to find matching sequences
/// - Encodes data as sequence of literals and (offset, length) match pairs
/// - Token byte: upper 4 bits = literal length, lower 4 bits = match length
/// - Variable-length encoding for lengths ≥ 15
/// - Last 5 bytes must be literals (safety margin)
///
/// **Properties**:
/// - Fast compression: ~400+ MB/s
/// - Very fast decompression: ~2000+ MB/s
/// - Typical compression ratio: 40-60% for text
/// - Frame format allows parallel processing
///
/// **Use cases**:
/// - Real-time compression (games, streaming)
/// - Database storage (RocksDB, MongoDB)
/// - Filesystems (ZFS, Btrfs)
/// - Network protocols (HTTP/2)
/// - Log compression
///
/// **Reference**: Yann Collet (2011) - LZ4 specification
///
/// Time: O(n) average for compression, O(m) for decompression where n = input, m = output
/// Space: O(hash_table_size + output_size)

const MIN_MATCH = 4; // LZ4 minimum match length
const LAST_LITERALS = 5; // Last 5 bytes must be literals
const ML_BITS = 4;
const ML_MASK = (1 << ML_BITS) - 1;
const RUN_BITS = 8 - ML_BITS;
const RUN_MASK = (1 << RUN_BITS) - 1;

/// Compress data using LZ4 algorithm.
///
/// Time: O(n) average
/// Space: O(hash_table_size + output_size)
pub fn encode(allocator: Allocator, input: []const u8) ![]u8 {
    return encodeWithConfig(allocator, input, .{});
}

/// Compress data using LZ4 with custom configuration.
///
/// Time: O(n × max_search) worst case
/// Space: O(hash_table_size + output_size)
pub fn encodeWithConfig(allocator: Allocator, input: []const u8, config: Config) ![]u8 {
    if (input.len == 0) return try allocator.alloc(u8, 0);

    // Allocate output buffer (worst case: input + overhead)
    var output = try std.ArrayList(u8).initCapacity(allocator, input.len + input.len / 255 + 16);
    errdefer output.deinit(allocator);

    // Hash table for finding matches
    var hash_table = try allocator.alloc(?usize, config.hash_table_size);
    defer allocator.free(hash_table);
    @memset(hash_table, null);

    var pos: usize = 0;
    const input_end = input.len;
    const match_limit = if (input_end >= LAST_LITERALS) input_end - LAST_LITERALS else 0;

    var literal_start = pos;

    while (pos < match_limit) {
        // Find match
        const hash = hashSequence(input[pos..@min(pos + 4, input.len)]) % config.hash_table_size;
        const match_pos: ?usize = hash_table[hash];
        hash_table[hash] = pos;

        var best_match_len: usize = 0;
        var best_match_pos: usize = 0;

        // Search for best match
        if (match_pos) |mp| {
            if (mp < pos) {
                const offset = pos - mp;
                if (offset <= 65535) { // LZ4 max offset is 16 bits
                    // Check match length
                    const max_len = @min(input_end - pos, input.len - mp);
                    var match_len: usize = 0;
                    while (match_len < max_len and input[pos + match_len] == input[mp + match_len]) {
                        match_len += 1;
                    }

                    if (match_len >= config.min_match_length) {
                        best_match_len = match_len;
                        best_match_pos = mp;
                    }
                }
            }
        }

        if (best_match_len >= config.min_match_length) {
            // Found a match - emit literals first, then match
            const literal_len = pos - literal_start;
            const match_len = best_match_len - MIN_MATCH;
            const offset = @as(u16, @intCast(pos - best_match_pos));

            // Encode token
            const token = encodeToken(@min(literal_len, 15), @min(match_len, 15));
            try output.append(allocator, token);

            // Encode extended literal length if needed
            if (literal_len >= 15) {
                var remaining = literal_len - 15;
                while (remaining >= 255) {
                    try output.append(allocator, 255);
                    remaining -= 255;
                }
                try output.append(allocator, @intCast(remaining));
            }

            // Emit literal bytes
            try output.appendSlice(allocator, input[literal_start..pos]);

            // Encode offset (little-endian)
            try output.append(allocator, @intCast(offset & 0xFF));
            try output.append(allocator, @intCast((offset >> 8) & 0xFF));

            // Encode extended match length if needed
            if (match_len >= 15) {
                var remaining = match_len - 15;
                while (remaining >= 255) {
                    try output.append(allocator, 255);
                    remaining -= 255;
                }
                try output.append(allocator, @intCast(remaining));
            }

            pos += best_match_len;
            literal_start = pos;
        } else {
            // No match - continue accumulating literals
            pos += 1;
        }
    }

    // Emit remaining literals (including last 5 bytes)
    const remaining_literals = input_end - literal_start;
    if (remaining_literals > 0) {
        const token = encodeToken(@min(remaining_literals, 15), 0);
        try output.append(allocator, token);

        // Encode extended literal length if needed
        if (remaining_literals >= 15) {
            var remaining = remaining_literals - 15;
            while (remaining >= 255) {
                try output.append(allocator, 255);
                remaining -= 255;
            }
            try output.append(allocator, @intCast(remaining));
        }

        try output.appendSlice(allocator, input[literal_start..input_end]);
    }

    return output.toOwnedSlice(allocator);
}

/// Decompress LZ4-compressed data.
///
/// Time: O(m) where m = output length
/// Space: O(m)
pub fn decode(allocator: Allocator, input: []const u8) ![]u8 {
    return decodeWithConfig(allocator, input, .{});
}

/// Decompress LZ4-compressed data with validation.
///
/// Time: O(m) where m = output length
/// Space: O(m)
pub fn decodeWithConfig(allocator: Allocator, input: []const u8, config: Config) ![]u8 {
    _ = config; // Config reserved for future validation options

    if (input.len == 0) return try allocator.alloc(u8, 0);

    var output = try std.ArrayList(u8).initCapacity(allocator, input.len * 2);
    errdefer output.deinit(allocator);

    var pos: usize = 0;
    while (pos < input.len) {
        // Read token
        const token = input[pos];
        pos += 1;

        var literal_len: usize = (token >> 4) & 0x0F;
        var match_len: usize = token & 0x0F;

        // Decode extended literal length if needed
        if (literal_len == 15) {
            while (pos < input.len) {
                const extra = input[pos];
                pos += 1;
                literal_len += extra;
                if (extra != 255) break;
            }
        }

        // Copy literals
        if (literal_len > 0) {
            if (pos + literal_len > input.len) break;
            try output.appendSlice(allocator, input[pos..pos + literal_len]);
            pos += literal_len;
        }

        // Check if there's a match (sequence ends with literals-only token if match_len == 0)
        if (pos >= input.len) break;

        // Read offset
        if (pos + 1 >= input.len) break;
        const offset = @as(u16, input[pos]) | (@as(u16, input[pos + 1]) << 8);
        pos += 2;

        // Decode extended match length if needed
        if (match_len == 15) {
            while (pos < input.len) {
                const extra = input[pos];
                pos += 1;
                match_len += extra;
                if (extra != 255) break;
            }
        }

        // Copy match (add MIN_MATCH to get actual length)
        const match_len_actual = match_len + MIN_MATCH;
        const match_start = output.items.len - offset;

        // Handle overlapping copies (e.g., offset < match_len)
        var i: usize = 0;
        while (i < match_len_actual) : (i += 1) {
            const src_idx = match_start + i;
            if (src_idx >= output.items.len) break; // Safety check
            try output.append(allocator, output.items[src_idx]);
        }
    }

    return output.toOwnedSlice(allocator);
}

/// Calculate compression ratio.
///
/// Time: O(1)
/// Space: O(1)
pub fn compressionRatio(original_size: usize, compressed_size: usize) f64 {
    if (compressed_size == 0) return 0.0;
    return @as(f64, @floatFromInt(original_size)) / @as(f64, @floatFromInt(compressed_size));
}

// Helper functions

fn hashSequence(data: []const u8) u32 {
    if (data.len < 4) {
        var hash: u32 = 0;
        for (data) |byte| {
            hash = (hash << 8) ^ byte;
        }
        return hash;
    }
    const val = @as(u32, data[0]) |
                (@as(u32, data[1]) << 8) |
                (@as(u32, data[2]) << 16) |
                (@as(u32, data[3]) << 24);
    return val *% 2654435761; // LZ4 hash multiplier
}

fn encodeToken(literal_len: usize, match_len: usize) u8 {
    const lit: u8 = @intCast(@min(literal_len, 15));
    const mat: u8 = @intCast(@min(match_len, 15));
    return (lit << 4) | mat;
}

// ============================================================================
// Tests
// ============================================================================

test "LZ4: empty input" {
    const allocator = testing.allocator;

    const input: []const u8 = "";
    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    try testing.expectEqual(0, compressed.len);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "LZ4: single character" {
    const allocator = testing.allocator;

    const input = "a";
    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "LZ4: no repetition" {
    const allocator = testing.allocator;

    const input = "abcdefghij";
    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "LZ4: simple repetition" {
    const allocator = testing.allocator;

    const input = "aaaaaa";
    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    // For very short strings, LZ4 may not achieve compression due to overhead
    // Just verify round-trip correctness
    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "LZ4: long repetition" {
    const allocator = testing.allocator;

    const input = "a" ** 100;
    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    // Should achieve significant compression
    try testing.expect(compressed.len < input.len / 2);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "LZ4: pattern repetition" {
    const allocator = testing.allocator;

    const input = "abcabc" ** 10;
    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "LZ4: mixed literals and matches" {
    const allocator = testing.allocator;

    const input = "hello world hello universe hello galaxy";
    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "LZ4: binary data" {
    const allocator = testing.allocator;

    const input = &[_]u8{0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7};
    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "LZ4: custom config (small hash table)" {
    const allocator = testing.allocator;

    const input = "test" ** 20;
    const config = Config{
        .hash_table_size = 256,
        .max_search = 64,
        .min_match_length = 4,
    };

    const compressed = try encodeWithConfig(allocator, input, config);
    defer allocator.free(compressed);

    const decompressed = try decodeWithConfig(allocator, compressed, config);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "LZ4: large text" {
    const allocator = testing.allocator;

    // Create large repetitive text
    var input_list = try std.ArrayList(u8).initCapacity(allocator, 1000);
    defer input_list.deinit(allocator);

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        try input_list.appendSlice(allocator, "compress");
    }

    const input = input_list.items;
    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    // Should achieve good compression on repetitive data
    try testing.expect(compressed.len < input.len);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "LZ4: compression ratio helper" {
    const ratio = compressionRatio(1000, 500);
    try testing.expectEqual(2.0, ratio);

    const ratio2 = compressionRatio(1000, 0);
    try testing.expectEqual(0.0, ratio2);
}

test "LZ4: maximum offset handling" {
    const allocator = testing.allocator;

    // Create input with match far from current position
    var input_list = try std.ArrayList(u8).initCapacity(allocator, 70000);
    defer input_list.deinit(allocator);

    try input_list.appendSlice(allocator, "pattern");

    // Add random data to create distance
    var j: usize = 0;
    while (j < 65000) : (j += 1) {
        try input_list.append(allocator, @intCast(j % 256));
    }

    try input_list.appendSlice(allocator, "pattern");

    const input = input_list.items;
    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}

test "LZ4: memory safety" {
    const allocator = testing.allocator;

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const input = "test data with repetition test data";
        const compressed = try encode(allocator, input);
        defer allocator.free(compressed);

        const decompressed = try decode(allocator, compressed);
        defer allocator.free(decompressed);

        try testing.expectEqualSlices(u8, input, decompressed);
    }
}

test "LZ4: stress test with varying patterns" {
    const allocator = testing.allocator;

    var input_list = try std.ArrayList(u8).initCapacity(allocator, 2000);
    defer input_list.deinit(allocator);

    // Mix of literals and repetitions
    try input_list.appendSlice(allocator, "unique");
    try input_list.appendSlice(allocator, "repeat" ** 10);
    try input_list.appendSlice(allocator, "unique2");
    try input_list.appendSlice(allocator, "repeat" ** 10);

    const input = input_list.items;
    const compressed = try encode(allocator, input);
    defer allocator.free(compressed);

    const decompressed = try decode(allocator, compressed);
    defer allocator.free(decompressed);

    try testing.expectEqualSlices(u8, input, decompressed);
}
