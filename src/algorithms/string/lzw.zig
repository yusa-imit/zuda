const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;

/// LZW (Lempel-Ziv-Welch) Compression
///
/// Dictionary-based adaptive compression algorithm that builds a dictionary
/// of strings dynamically during encoding/decoding. Widely used in GIF, TIFF,
/// PDF, and Unix compress utility.
///
/// Algorithm:
/// - Encoding: Build dictionary of string sequences, output codes
/// - Decoding: Rebuild dictionary from codes, output original strings
/// - Dictionary starts with all single-byte values (0-255)
/// - New entries added as sequences are encountered
///
/// Time complexity:
/// - encode(): O(n) where n = input length
/// - decode(): O(m) where m = compressed length
///
/// Space complexity:
/// - O(d) where d = dictionary size (typically 4096 entries max)
///
/// Use cases:
/// - GIF image compression (patented until 2003, now free)
/// - TIFF image format
/// - PDF document compression
/// - Unix compress utility
/// - Text file compression
///
/// Properties:
/// - Adaptive: dictionary built from input data
/// - No prior knowledge of input needed
/// - Works well on text with repeated patterns
/// - Lossless compression
///
/// Trade-offs:
/// - vs RLE: Better on varied patterns, worse on simple runs
/// - vs LZ77: Simpler, no sliding window overhead
/// - vs Huffman: No frequency analysis needed, adaptive
///
/// Reference:
/// - Welch, T. (1984). "A Technique for High-Performance Data Compression"
/// - IEEE Computer, 17(6), 8-19
/// - GIF89a specification

/// LZW compression result with metadata
pub const CompressionResult = struct {
    codes: []u16,
    dictionary_size: usize,
    compression_ratio: f64,

    pub fn deinit(self: @This(), allocator: Allocator) void {
        allocator.free(self.codes);
    }
};

/// LZW decompression result
pub const DecompressionResult = struct {
    data: []u8,

    pub fn deinit(self: @This(), allocator: Allocator) void {
        allocator.free(self.data);
    }
};

/// Maximum dictionary size (12-bit codes = 4096 entries)
/// First 256 entries reserved for single bytes
pub const MAX_DICT_SIZE: usize = 4096;
pub const INITIAL_DICT_SIZE: usize = 256;

pub const LZWError = error{
    InvalidCode,
    DictionaryFull,
    EmptyInput,
};

/// Encode data using LZW compression
///
/// Builds dictionary adaptively while encoding. Returns codes array.
///
/// Time: O(n) where n = input length
/// Space: O(d) where d = dictionary size
///
/// Example:
/// ```zig
/// const data = "TOBEORNOTTOBEORTOBEORNOT";
/// var result = try encode(allocator, data);
/// defer result.deinit(allocator);
/// // result.codes contains compressed representation
/// ```
pub fn encode(allocator: Allocator, data: []const u8) !CompressionResult {
    if (data.len == 0) return LZWError.EmptyInput;

    var codes = ArrayList(u16).init(allocator);
    errdefer codes.deinit();

    // Dictionary: string -> code
    // Use StringHashMap with owned keys
    var dictionary = StringHashMap(u16).init(allocator);
    defer {
        // Free all owned keys
        var it = dictionary.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
        }
        dictionary.deinit();
    }

    // Initialize dictionary with single-byte values (0-255)
    var i: u16 = 0;
    while (i < INITIAL_DICT_SIZE) : (i += 1) {
        const key = try allocator.alloc(u8, 1);
        key[0] = @intCast(i);
        try dictionary.put(key, i);
    }

    var next_code: u16 = INITIAL_DICT_SIZE;
    var current = ArrayList(u8).init(allocator);
    defer current.deinit();

    for (data) |byte| {
        // Try extending current string
        try current.append(byte);

        // Check if extended string is in dictionary
        if (!dictionary.contains(current.items)) {
            // Output code for current string without last byte
            const prev_len = current.items.len - 1;
            const prev = current.items[0..prev_len];
            const code = dictionary.get(prev) orelse unreachable;
            try codes.append(code);

            // Add new string to dictionary if not full
            if (next_code < MAX_DICT_SIZE) {
                const key = try allocator.dupe(u8, current.items);
                try dictionary.put(key, next_code);
                next_code += 1;
            }

            // Reset current to last byte only
            current.clearRetainingCapacity();
            try current.append(byte);
        }
    }

    // Output code for remaining string
    if (current.items.len > 0) {
        const code = dictionary.get(current.items) orelse unreachable;
        try codes.append(code);
    }

    const codes_slice = try codes.toOwnedSlice();
    const ratio = if (data.len > 0)
        1.0 - (@as(f64, @floatFromInt(codes_slice.len * 2)) / @as(f64, @floatFromInt(data.len)))
    else
        0.0;

    return CompressionResult{
        .codes = codes_slice,
        .dictionary_size = next_code,
        .compression_ratio = ratio,
    };
}

/// Decode LZW compressed data
///
/// Rebuilds dictionary while decoding codes back to original data.
///
/// Time: O(m) where m = number of codes
/// Space: O(d) where d = dictionary size
///
/// Example:
/// ```zig
/// var result = try decode(allocator, codes);
/// defer result.deinit(allocator);
/// // result.data contains decompressed data
/// ```
pub fn decode(allocator: Allocator, codes: []const u16) !DecompressionResult {
    if (codes.len == 0) return LZWError.EmptyInput;

    var output = ArrayList(u8).init(allocator);
    errdefer output.deinit();

    // Dictionary: code -> string
    var dictionary = ArrayList([]u8).init(allocator);
    defer {
        for (dictionary.items) |entry| {
            allocator.free(entry);
        }
        dictionary.deinit();
    }

    // Initialize dictionary with single-byte values (0-255)
    var i: usize = 0;
    while (i < INITIAL_DICT_SIZE) : (i += 1) {
        const entry = try allocator.alloc(u8, 1);
        entry[0] = @intCast(i);
        try dictionary.append(entry);
    }

    // Process first code
    const first_code = codes[0];
    if (first_code >= dictionary.items.len) return LZWError.InvalidCode;

    var previous = try allocator.dupe(u8, dictionary.items[first_code]);
    defer allocator.free(previous);
    try output.appendSlice(previous);

    // Process remaining codes
    for (codes[1..]) |code| {
        var current: []u8 = undefined;
        var need_free = false;
        defer if (need_free) allocator.free(current);

        if (code < dictionary.items.len) {
            // Code exists in dictionary
            current = dictionary.items[code];
        } else if (code == dictionary.items.len) {
            // Special case: code = next_code (pattern like "xyx" where x=previous)
            // Create string = previous + first char of previous
            current = try allocator.alloc(u8, previous.len + 1);
            need_free = true;
            @memcpy(current[0..previous.len], previous);
            current[previous.len] = previous[0];
        } else {
            return LZWError.InvalidCode;
        }

        try output.appendSlice(current);

        // Add new dictionary entry: previous + first char of current
        if (dictionary.items.len < MAX_DICT_SIZE) {
            const new_entry = try allocator.alloc(u8, previous.len + 1);
            @memcpy(new_entry[0..previous.len], previous);
            new_entry[previous.len] = current[0];
            try dictionary.append(new_entry);
        }

        // Update previous
        allocator.free(previous);
        previous = try allocator.dupe(u8, current);
    }

    return DecompressionResult{
        .data = try output.toOwnedSlice(),
    };
}

/// Calculate compression ratio
///
/// Returns value between -∞ and 1.0:
/// - 1.0 = perfect compression (100% size reduction)
/// - 0.0 = no compression
/// - negative = expansion (compressed larger than original)
///
/// Time: O(1)
/// Space: O(1)
pub fn compressionRatio(original_size: usize, compressed_codes: usize) f64 {
    if (original_size == 0) return 0.0;
    const compressed_bytes = compressed_codes * 2; // u16 codes = 2 bytes each
    return 1.0 - (@as(f64, @floatFromInt(compressed_bytes)) / @as(f64, @floatFromInt(original_size)));
}

/// Check if LZW compression would reduce size
///
/// Returns true if encoding would save space.
///
/// Time: O(1)
/// Space: O(1)
pub fn wouldCompress(original_size: usize, compressed_codes: usize) bool {
    return compressionRatio(original_size, compressed_codes) > 0.0;
}

/// Get dictionary utilization percentage
///
/// Returns how full the dictionary is (0.0 to 1.0).
///
/// Time: O(1)
/// Space: O(1)
pub fn dictionaryUtilization(dictionary_size: usize) f64 {
    return @as(f64, @floatFromInt(dictionary_size)) / @as(f64, @floatFromInt(MAX_DICT_SIZE));
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const expect = testing.expect;
const expectEqual = testing.expectEqual;
const expectError = testing.expectError;
const expectApproxEqRel = testing.expectApproxEqRel;

test "LZW: basic encode" {
    const allocator = testing.allocator;
    const data = "TOBEORNOTTOBEORTOBEORNOT";

    var result = try encode(allocator, data);
    defer result.deinit(allocator);

    try expect(result.codes.len > 0);
    try expect(result.codes.len < data.len); // Should compress
    try expect(result.dictionary_size > INITIAL_DICT_SIZE);
    try expect(result.dictionary_size <= MAX_DICT_SIZE);
}

test "LZW: basic decode" {
    const allocator = testing.allocator;
    const data = "TOBEORNOTTOBEORTOBEORNOT";

    var enc_result = try encode(allocator, data);
    defer enc_result.deinit(allocator);

    var dec_result = try decode(allocator, enc_result.codes);
    defer dec_result.deinit(allocator);

    try expectEqual(data.len, dec_result.data.len);
    try expect(std.mem.eql(u8, data, dec_result.data));
}

test "LZW: roundtrip with various strings" {
    const allocator = testing.allocator;

    const test_cases = [_][]const u8{
        "ABABABA",
        "AAAAAAA",
        "ABCDEFGHIJKLMNOP",
        "The quick brown fox jumps over the lazy dog",
        "compression compression compression",
        "1234567890123456789012345678901234567890",
    };

    for (test_cases) |data| {
        var enc = try encode(allocator, data);
        defer enc.deinit(allocator);

        var dec = try decode(allocator, enc.codes);
        defer dec.deinit(allocator);

        try expectEqual(data.len, dec.data.len);
        try expect(std.mem.eql(u8, data, dec.data));
    }
}

test "LZW: single byte" {
    const allocator = testing.allocator;
    const data = "A";

    var enc = try encode(allocator, data);
    defer enc.deinit(allocator);

    try expectEqual(@as(usize, 1), enc.codes.len);
    try expectEqual(@as(u16, 'A'), enc.codes[0]);

    var dec = try decode(allocator, enc.codes);
    defer dec.deinit(allocator);

    try expectEqual(@as(usize, 1), dec.data.len);
    try expectEqual(@as(u8, 'A'), dec.data[0]);
}

test "LZW: repeated pattern" {
    const allocator = testing.allocator;
    const data = "ABABABABABABABABABABAB";

    var enc = try encode(allocator, data);
    defer enc.deinit(allocator);

    // Should compress well due to repetition
    try expect(enc.codes.len < data.len);

    var dec = try decode(allocator, enc.codes);
    defer dec.deinit(allocator);

    try expect(std.mem.eql(u8, data, dec.data));
}

test "LZW: no repetition" {
    const allocator = testing.allocator;
    const data = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    var enc = try encode(allocator, data);
    defer enc.deinit(allocator);

    // May not compress well - mostly single char codes
    try expect(enc.codes.len > 0);

    var dec = try decode(allocator, enc.codes);
    defer dec.deinit(allocator);

    try expect(std.mem.eql(u8, data, dec.data));
}

test "LZW: compression ratio calculation" {
    try expectApproxEqRel(@as(f64, 0.5), compressionRatio(100, 25), 0.01);
    try expectApproxEqRel(@as(f64, 0.0), compressionRatio(100, 50), 0.01);
    try expectApproxEqRel(@as(f64, -1.0), compressionRatio(100, 100), 0.01);
    try expectApproxEqRel(@as(f64, 0.0), compressionRatio(0, 0), 0.01);
}

test "LZW: would compress check" {
    try expect(wouldCompress(100, 25));
    try expect(!wouldCompress(100, 50));
    try expect(!wouldCompress(100, 100));
    try expect(!wouldCompress(0, 0));
}

test "LZW: dictionary utilization" {
    try expectApproxEqRel(@as(f64, 0.25), dictionaryUtilization(1024), 0.01);
    try expectApproxEqRel(@as(f64, 0.5), dictionaryUtilization(2048), 0.01);
    try expectApproxEqRel(@as(f64, 1.0), dictionaryUtilization(4096), 0.01);
}

test "LZW: empty input error" {
    const allocator = testing.allocator;
    try expectError(LZWError.EmptyInput, encode(allocator, ""));
}

test "LZW: invalid code error" {
    const allocator = testing.allocator;
    const codes = [_]u16{ 65, 5000 }; // 5000 is invalid
    try expectError(LZWError.InvalidCode, decode(allocator, &codes));
}

test "LZW: special case xyx pattern" {
    const allocator = testing.allocator;
    // This tests the special case where next code refers to itself
    const data = "ABABAB";

    var enc = try encode(allocator, data);
    defer enc.deinit(allocator);

    var dec = try decode(allocator, enc.codes);
    defer dec.deinit(allocator);

    try expect(std.mem.eql(u8, data, dec.data));
}

test "LZW: large dictionary usage" {
    const allocator = testing.allocator;
    // Create data that will use many dictionary entries
    var data_list = ArrayList(u8).init(allocator);
    defer data_list.deinit();

    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        try data_list.append(@intCast(i % 26 + 'A'));
    }

    var enc = try encode(allocator, data_list.items);
    defer enc.deinit(allocator);

    try expect(enc.dictionary_size > INITIAL_DICT_SIZE);
    try expect(enc.dictionary_size <= MAX_DICT_SIZE);

    var dec = try decode(allocator, enc.codes);
    defer dec.deinit(allocator);

    try expect(std.mem.eql(u8, data_list.items, dec.data));
}

test "LZW: binary data" {
    const allocator = testing.allocator;
    const data = [_]u8{ 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 };

    var enc = try encode(allocator, &data);
    defer enc.deinit(allocator);

    var dec = try decode(allocator, enc.codes);
    defer dec.deinit(allocator);

    try expectEqual(data.len, dec.data.len);
    try expect(std.mem.eql(u8, &data, dec.data));
}

test "LZW: all identical bytes" {
    const allocator = testing.allocator;
    const data = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";

    var enc = try encode(allocator, data);
    defer enc.deinit(allocator);

    // Should compress very well
    try expect(enc.codes.len < data.len / 2);

    var dec = try decode(allocator, enc.codes);
    defer dec.deinit(allocator);

    try expect(std.mem.eql(u8, data, dec.data));
}

test "LZW: long text" {
    const allocator = testing.allocator;
    const data = "The LZW algorithm compresses data by replacing repeated " ++
        "sequences with codes. The algorithm builds a dictionary " ++
        "dynamically during compression. The dictionary starts with " ++
        "all single-byte values and grows as new sequences are found.";

    var enc = try encode(allocator, data);
    defer enc.deinit(allocator);

    try expect(enc.codes.len < data.len); // Should compress

    var dec = try decode(allocator, enc.codes);
    defer dec.deinit(allocator);

    try expect(std.mem.eql(u8, data, dec.data));
}

test "LZW: actual compression on repetitive text" {
    const allocator = testing.allocator;
    const data = "TOBEORNOTTOBEORTOBEORNOT";

    var enc = try encode(allocator, data);
    defer enc.deinit(allocator);

    const ratio = enc.compression_ratio;
    try expect(ratio > 0.0); // Should achieve some compression
}

test "LZW: memory safety check" {
    const allocator = testing.allocator;

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const data = "REPEATING PATTERN REPEATING PATTERN";

        var enc = try encode(allocator, data);
        defer enc.deinit(allocator);

        var dec = try decode(allocator, enc.codes);
        defer dec.deinit(allocator);

        try expect(std.mem.eql(u8, data, dec.data));
    }
}
