/// Run-Length Encoding (RLE) - Simple lossless compression algorithm
///
/// RLE compresses data by replacing consecutive identical elements with a count and the element.
/// For example, "AAAABBBCCDAA" becomes "4A3B2C1D2A" in simple RLE format.
///
/// ## Overview
///
/// RLE is one of the simplest compression algorithms, ideal for data with long runs of repeated values.
/// It's widely used in fax machines, simple graphics (PCX, BMP), and as a preprocessing step
/// for more complex compression.
///
/// ## Features
///
/// - **encode()**: Compress string to RLE format - O(n) time, O(n) space
/// - **decode()**: Decompress RLE back to original - O(n) time, O(n) space
/// - **encodeBytes()**: Binary RLE for arbitrary byte sequences
/// - **decodeBytes()**: Binary RLE decompression
/// - **compressionRatio()**: Calculate space savings (0-1, higher = better)
/// - **wouldCompress()**: Check if RLE would reduce size
///
/// ## Algorithm
///
/// **Encoding**: Scan input, count consecutive runs of same character, emit "count+char"
/// **Decoding**: Parse "count" (digits), read char, repeat char count times
///
/// ## Time Complexity
///
/// - **encode()**: O(n) where n = input length (single pass)
/// - **decode()**: O(m) where m = encoded length (parsing + expansion)
/// - **compressionRatio()**: O(n + m) for both encode and original size
///
/// ## Space Complexity
///
/// - **encode()**: O(k) where k = number of runs (worst case: O(n) for alternating chars)
/// - **decode()**: O(n) for reconstructed data
/// - **In-place variants**: Not implemented (requires mutable buffer)
///
/// ## Use Cases
///
/// - Simple graphics compression (icons, fax images)
/// - Data with long repetitive sequences
/// - Preprocessing for complex compression (Burrows-Wheeler Transform)
/// - Network protocol encoding (reduce bandwidth for sparse data)
/// - Test data generation (compact representation)
///
/// ## Trade-offs
///
/// - **Pros**: Simple, fast, no memory overhead, deterministic
/// - **Cons**: Can expand data if no repetition (worst case: 2x size for alternating chars)
/// - **vs LZ77**: RLE simpler but less effective for general text
/// - **vs Huffman**: RLE doesn't need frequency table, but worse compression for varied data
///
/// ## References
///
/// - PCX image format (1985)
/// - BMP RLE compression
/// - ITU-T T.4 (fax standard)
/// - "Data Compression: The Complete Reference" by Salomon (2007)

const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;

/// Encode string using run-length encoding.
/// Returns encoded string as "count1char1count2char2...".
/// Caller owns returned memory.
///
/// Time: O(n) where n = input length
/// Space: O(k) where k = number of runs (worst case O(n))
///
/// Example:
/// ```zig
/// const encoded = try encode(allocator, "AAAABBBCCDAA");
/// defer allocator.free(encoded);
/// // Result: "4A3B2C1D2A"
/// ```
pub fn encode(allocator: Allocator, input: []const u8) ![]u8 {
    if (input.len == 0) return try allocator.dupe(u8, "");

    var result = try ArrayList(u8).initCapacity(allocator, input.len);
    errdefer result.deinit(allocator);

    var i: usize = 0;
    while (i < input.len) {
        const current = input[i];
        var count: usize = 1;

        // Count consecutive identical characters
        while (i + count < input.len and input[i + count] == current) {
            count += 1;
        }

        // Append count as decimal digits
        var count_buf: [20]u8 = undefined;
        const count_str = try std.fmt.bufPrint(&count_buf, "{d}", .{count});
        try result.appendSlice(allocator, count_str);

        // Append character
        try result.append(allocator, current);

        i += count;
    }

    return try result.toOwnedSlice(allocator);
}

/// Decode run-length encoded string back to original.
/// Input format: "count1char1count2char2...".
/// Caller owns returned memory.
///
/// Time: O(m) where m = encoded length
/// Space: O(n) where n = decoded length
///
/// Example:
/// ```zig
/// const decoded = try decode(allocator, "4A3B2C1D2A");
/// defer allocator.free(decoded);
/// // Result: "AAAABBBCCDAA"
/// ```
pub fn decode(allocator: Allocator, encoded: []const u8) ![]u8 {
    if (encoded.len == 0) return try allocator.dupe(u8, "");

    var result = try ArrayList(u8).initCapacity(allocator, encoded.len);
    errdefer result.deinit(allocator);

    var i: usize = 0;
    while (i < encoded.len) {
        // Parse count (all digits until non-digit)
        var count: usize = 0;
        var digit_count: usize = 0;
        while (i < encoded.len and std.ascii.isDigit(encoded[i])) {
            count = count * 10 + (encoded[i] - '0');
            i += 1;
            digit_count += 1;
        }

        if (digit_count == 0 or i >= encoded.len) return error.InvalidRLEFormat;
        if (count == 0) return error.ZeroRunLength;

        // Read character
        const char = encoded[i];
        i += 1;

        // Append character 'count' times
        try result.appendNTimes(allocator, char, count);
    }

    return try result.toOwnedSlice(allocator);
}

/// Encode byte sequence using run-length encoding.
/// Similar to encode() but for arbitrary binary data.
/// Returns encoded bytes as sequence of (count_byte, value_byte) pairs.
///
/// Time: O(n) where n = input length
/// Space: O(k*2) where k = number of runs
///
/// Example:
/// ```zig
/// const data = [_]u8{0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF};
/// const encoded = try encodeBytes(allocator, &data);
/// defer allocator.free(encoded);
/// // Result: [3, 0xFF, 2, 0x00, 1, 0xFF]
/// ```
pub fn encodeBytes(allocator: Allocator, input: []const u8) ![]u8 {
    if (input.len == 0) return try allocator.dupe(u8, "");

    var result = try ArrayList(u8).initCapacity(allocator, input.len);
    errdefer result.deinit(allocator);

    var i: usize = 0;
    while (i < input.len) {
        const current = input[i];
        var count: usize = 1;

        // Count consecutive identical bytes (max 255 per run for single-byte count)
        while (i + count < input.len and input[i + count] == current and count < 255) {
            count += 1;
        }

        // Append count byte
        try result.append(allocator, @as(u8, @intCast(count)));
        // Append value byte
        try result.append(allocator, current);

        i += count;
    }

    return try result.toOwnedSlice(allocator);
}

/// Decode byte sequence from run-length encoding.
/// Input format: sequence of (count_byte, value_byte) pairs.
/// Caller owns returned memory.
///
/// Time: O(m) where m = encoded length
/// Space: O(n) where n = decoded length
///
/// Example:
/// ```zig
/// const encoded = [_]u8{3, 0xFF, 2, 0x00, 1, 0xFF};
/// const decoded = try decodeBytes(allocator, &encoded);
/// defer allocator.free(decoded);
/// // Result: [0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF]
/// ```
pub fn decodeBytes(allocator: Allocator, encoded: []const u8) ![]u8 {
    if (encoded.len == 0) return try allocator.dupe(u8, "");
    if (encoded.len % 2 != 0) return error.InvalidRLEFormat;

    var result = try ArrayList(u8).initCapacity(allocator, encoded.len);
    errdefer result.deinit(allocator);

    var i: usize = 0;
    while (i < encoded.len) {
        const count = encoded[i];
        i += 1;
        if (i >= encoded.len) return error.InvalidRLEFormat;
        if (count == 0) return error.ZeroRunLength;

        const value = encoded[i];
        i += 1;

        // Append value 'count' times
        try result.appendNTimes(allocator, value, count);
    }

    return try result.toOwnedSlice(allocator);
}

/// Calculate compression ratio for given input.
/// Returns ratio in range [0, 1] where higher = better compression.
/// Ratio = 1 - (encoded_size / original_size)
///
/// Time: O(n) - performs full encoding to measure size
/// Space: O(k) for temporary encoding
///
/// Example:
/// ```zig
/// const ratio = try compressionRatio(allocator, "AAAABBBCCDAA");
/// // Result: ~0.17 (12 bytes -> 10 bytes, 17% reduction)
/// ```
pub fn compressionRatio(allocator: Allocator, input: []const u8) !f64 {
    if (input.len == 0) return 0.0;

    const encoded = try encode(allocator, input);
    defer allocator.free(encoded);

    const original: f64 = @floatFromInt(input.len);
    const compressed: f64 = @floatFromInt(encoded.len);

    return 1.0 - (compressed / original);
}

/// Check if RLE would compress the input (save space).
/// Returns true if encoded size < original size.
///
/// Time: O(n) - performs full encoding
/// Space: O(k) for temporary encoding
///
/// Example:
/// ```zig
/// const will_compress = try wouldCompress(allocator, "AAAABBBCCDAA");
/// // Result: true
/// const wont_compress = try wouldCompress(allocator, "ABCDEFG");
/// // Result: false (alternating chars expand data)
/// ```
pub fn wouldCompress(allocator: Allocator, input: []const u8) !bool {
    if (input.len == 0) return false;

    const encoded = try encode(allocator, input);
    defer allocator.free(encoded);

    return encoded.len < input.len;
}

/// Estimate run count without allocating (for analysis).
/// Counts number of consecutive runs in input.
///
/// Time: O(n)
/// Space: O(1)
///
/// Example:
/// ```zig
/// const runs = countRuns("AAAABBBCCDAA");
/// // Result: 5 (AAAA, BBB, CC, D, AA)
/// ```
pub fn countRuns(input: []const u8) usize {
    if (input.len == 0) return 0;

    var count: usize = 1;
    var i: usize = 1;

    while (i < input.len) : (i += 1) {
        if (input[i] != input[i - 1]) {
            count += 1;
        }
    }

    return count;
}

/// Calculate average run length (for analysis).
/// Returns mean length of consecutive runs.
///
/// Time: O(n)
/// Space: O(1)
///
/// Example:
/// ```zig
/// const avg = avgRunLength("AAAABBBCCDAA");
/// // Result: 2.4 (12 chars / 5 runs)
/// ```
pub fn avgRunLength(input: []const u8) f64 {
    if (input.len == 0) return 0.0;

    const runs = countRuns(input);
    const total: f64 = @floatFromInt(input.len);
    const run_count: f64 = @floatFromInt(runs);

    return total / run_count;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const expect = testing.expect;
const expectEqual = testing.expectEqual;
const expectEqualSlices = testing.expectEqualSlices;
const expectError = testing.expectError;

test "encode basic" {
    const allocator = testing.allocator;

    {
        const result = try encode(allocator, "AAAABBBCCDAA");
        defer allocator.free(result);
        try expectEqualSlices(u8, "4A3B2C1D2A", result);
    }

    {
        const result = try encode(allocator, "WWWWWWWWWWWWBWWWWWWWWWWWWBBBWWWWWWWWWWWWWWWWWWWWWWWWB");
        defer allocator.free(result);
        try expectEqualSlices(u8, "12W1B12W3B24W1B", result);
    }
}

test "encode single character" {
    const allocator = testing.allocator;

    const result = try encode(allocator, "AAAAA");
    defer allocator.free(result);
    try expectEqualSlices(u8, "5A", result);
}

test "encode no repetition (worst case)" {
    const allocator = testing.allocator;

    const result = try encode(allocator, "ABCDEF");
    defer allocator.free(result);
    try expectEqualSlices(u8, "1A1B1C1D1E1F", result);
    // Encoded is 2x larger than original (worst case expansion)
    try expect(result.len > "ABCDEF".len);
}

test "encode empty string" {
    const allocator = testing.allocator;

    const result = try encode(allocator, "");
    defer allocator.free(result);
    try expectEqualSlices(u8, "", result);
}

test "decode basic" {
    const allocator = testing.allocator;

    {
        const result = try decode(allocator, "4A3B2C1D2A");
        defer allocator.free(result);
        try expectEqualSlices(u8, "AAAABBBCCDAA", result);
    }

    {
        const result = try decode(allocator, "12W1B12W3B24W1B");
        defer allocator.free(result);
        try expectEqualSlices(u8, "WWWWWWWWWWWWBWWWWWWWWWWWWBBBWWWWWWWWWWWWWWWWWWWWWWWWB", result);
    }
}

test "decode single run" {
    const allocator = testing.allocator;

    const result = try decode(allocator, "5A");
    defer allocator.free(result);
    try expectEqualSlices(u8, "AAAAA", result);
}

test "decode empty string" {
    const allocator = testing.allocator;

    const result = try decode(allocator, "");
    defer allocator.free(result);
    try expectEqualSlices(u8, "", result);
}

test "encode-decode roundtrip" {
    const allocator = testing.allocator;

    const original = "AAAABBBCCDAA";
    const encoded = try encode(allocator, original);
    defer allocator.free(encoded);

    const decoded = try decode(allocator, encoded);
    defer allocator.free(decoded);

    try expectEqualSlices(u8, original, decoded);
}

test "encode-decode roundtrip no repetition" {
    const allocator = testing.allocator;

    const original = "ABCDEFGHIJ";
    const encoded = try encode(allocator, original);
    defer allocator.free(encoded);

    const decoded = try decode(allocator, encoded);
    defer allocator.free(decoded);

    try expectEqualSlices(u8, original, decoded);
}

test "decode invalid format (no char after digits)" {
    const allocator = testing.allocator;

    try expectError(error.InvalidRLEFormat, decode(allocator, "4A3"));
}

test "decode invalid format (no digits)" {
    const allocator = testing.allocator;

    try expectError(error.InvalidRLEFormat, decode(allocator, "AB"));
}

test "decode zero run length" {
    const allocator = testing.allocator;

    try expectError(error.ZeroRunLength, decode(allocator, "0A"));
}

test "encodeBytes basic" {
    const allocator = testing.allocator;

    const input = [_]u8{ 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF };
    const result = try encodeBytes(allocator, &input);
    defer allocator.free(result);

    const expected = [_]u8{ 3, 0xFF, 2, 0x00, 1, 0xFF };
    try expectEqualSlices(u8, &expected, result);
}

test "encodeBytes max run length (255)" {
    const allocator = testing.allocator;

    var input: [300]u8 = undefined;
    @memset(&input, 0xAA);

    const result = try encodeBytes(allocator, &input);
    defer allocator.free(result);

    // Should split into 255 + 45
    const expected = [_]u8{ 255, 0xAA, 45, 0xAA };
    try expectEqualSlices(u8, &expected, result);
}

test "decodeBytes basic" {
    const allocator = testing.allocator;

    const encoded = [_]u8{ 3, 0xFF, 2, 0x00, 1, 0xFF };
    const result = try decodeBytes(allocator, &encoded);
    defer allocator.free(result);

    const expected = [_]u8{ 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF };
    try expectEqualSlices(u8, &expected, result);
}

test "encodeBytes-decodeBytes roundtrip" {
    const allocator = testing.allocator;

    const original = [_]u8{ 0xFF, 0xFF, 0x00, 0x00, 0x00, 0xAA, 0xAA };
    const encoded = try encodeBytes(allocator, &original);
    defer allocator.free(encoded);

    const decoded = try decodeBytes(allocator, encoded);
    defer allocator.free(decoded);

    try expectEqualSlices(u8, &original, decoded);
}

test "decodeBytes invalid format (odd length)" {
    const allocator = testing.allocator;

    const encoded = [_]u8{ 3, 0xFF, 2 };
    try expectError(error.InvalidRLEFormat, decodeBytes(allocator, &encoded));
}

test "decodeBytes zero run length" {
    const allocator = testing.allocator;

    const encoded = [_]u8{ 0, 0xFF };
    try expectError(error.ZeroRunLength, decodeBytes(allocator, &encoded));
}

test "compressionRatio good compression" {
    const allocator = testing.allocator;

    const ratio = try compressionRatio(allocator, "AAAABBBCCDAA");
    // 12 bytes -> "4A3B2C1D2A" = 10 bytes, ratio = 1 - 10/12 = 0.166...
    try expect(ratio > 0.1);
    try expect(ratio < 0.2);
}

test "compressionRatio no compression (expansion)" {
    const allocator = testing.allocator;

    const ratio = try compressionRatio(allocator, "ABCDEF");
    // 6 bytes -> "1A1B1C1D1E1F" = 12 bytes, ratio = 1 - 12/6 = -1.0 (negative = expansion)
    try expect(ratio < 0.0);
}

test "wouldCompress positive" {
    const allocator = testing.allocator;

    try expect(try wouldCompress(allocator, "AAAABBBCCDAA"));
}

test "wouldCompress negative (alternating)" {
    const allocator = testing.allocator;

    try expect(!try wouldCompress(allocator, "ABCDEFGHIJ"));
}

test "countRuns basic" {
    try expectEqual(@as(usize, 5), countRuns("AAAABBBCCDAA"));
    try expectEqual(@as(usize, 6), countRuns("ABCDEF"));
    try expectEqual(@as(usize, 1), countRuns("AAAAAAA"));
    try expectEqual(@as(usize, 0), countRuns(""));
}

test "avgRunLength basic" {
    const avg = avgRunLength("AAAABBBCCDAA");
    // 12 chars / 5 runs = 2.4
    try expect(avg > 2.3);
    try expect(avg < 2.5);
}

test "avgRunLength no repetition" {
    const avg = avgRunLength("ABCDEF");
    // 6 chars / 6 runs = 1.0
    try expect(avg > 0.9);
    try expect(avg < 1.1);
}

test "large input compression" {
    const allocator = testing.allocator;

    var input: [1000]u8 = undefined;
    @memset(&input, 'A');

    const encoded = try encode(allocator, &input);
    defer allocator.free(encoded);

    // "1000A" = 5 bytes vs 1000 bytes
    try expect(encoded.len < 10);

    const decoded = try decode(allocator, encoded);
    defer allocator.free(decoded);

    try expectEqualSlices(u8, &input, decoded);
}

test "memory safety" {
    const allocator = testing.allocator;

    // Run multiple iterations to catch leaks
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const original = "AAAABBBCCDAA";

        const encoded = try encode(allocator, original);
        defer allocator.free(encoded);

        const decoded = try decode(allocator, encoded);
        defer allocator.free(decoded);

        try expectEqualSlices(u8, original, decoded);
    }
}
