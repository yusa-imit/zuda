const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// Run-Length Encoding (RLE) - Lossless compression for repetitive data
///
/// RLE compresses sequences by replacing consecutive identical values with
/// a count-value pair. Optimal for data with long runs (e.g., simple images,
/// fax transmissions, DNA sequences).
///
/// Format: Each run is encoded as (count: u32, value: T)
///
/// Time: O(n) for both encode and decode
/// Space: O(n) worst case (all unique values), O(k) best case (k runs)

/// Encode data using Run-Length Encoding
///
/// Returns ArrayList of (count, value) pairs where each pair represents
/// a run of consecutive identical values.
///
/// Time: O(n) where n is input length
/// Space: O(k) where k is number of runs (k ≤ n)
///
/// Example:
/// ```zig
/// const input = [_]u8{ 'A', 'A', 'A', 'B', 'B', 'C' };
/// var result = try encode(u8, allocator, &input);
/// defer result.deinit();
/// // result = [(3, 'A'), (2, 'B'), (1, 'C')]
/// ```
pub fn encode(comptime T: type, allocator: Allocator, data: []const T) !std.ArrayList(Run(T)) {
    var runs = std.ArrayList(Run(T)).init(allocator);
    errdefer runs.deinit();

    if (data.len == 0) return runs;

    var current_value = data[0];
    var current_count: u32 = 1;

    for (data[1..]) |value| {
        if (std.meta.eql(value, current_value)) {
            current_count += 1;
        } else {
            try runs.append(.{ .count = current_count, .value = current_value });
            current_value = value;
            current_count = 1;
        }
    }

    // Append final run
    try runs.append(.{ .count = current_count, .value = current_value });

    return runs;
}

/// Decode RLE-encoded data back to original form
///
/// Takes ArrayList of (count, value) pairs and expands them back to
/// the original sequence.
///
/// Time: O(n) where n is total count of all runs
/// Space: O(n) for output buffer
///
/// Example:
/// ```zig
/// const runs = [_]Run(u8){ .{ .count = 3, .value = 'A' }, .{ .count = 2, .value = 'B' } };
/// var result = try decode(u8, allocator, &runs);
/// defer allocator.free(result);
/// // result = "AAABB"
/// ```
pub fn decode(comptime T: type, allocator: Allocator, runs: []const Run(T)) ![]T {
    // Calculate total output size
    var total_size: usize = 0;
    for (runs) |run| {
        total_size += run.count;
    }

    var output = try allocator.alloc(T, total_size);
    errdefer allocator.free(output);

    var pos: usize = 0;
    for (runs) |run| {
        var i: u32 = 0;
        while (i < run.count) : (i += 1) {
            output[pos] = run.value;
            pos += 1;
        }
    }

    return output;
}

/// Compression ratio for given data and runs
///
/// Returns ratio = original_size / compressed_size
/// Ratio > 1 means compression, < 1 means expansion
///
/// Time: O(1)
/// Space: O(1)
pub fn compressionRatio(comptime T: type, original_len: usize, runs_len: usize) f64 {
    if (runs_len == 0) return 0.0;

    const original_bytes = original_len * @sizeOf(T);
    const compressed_bytes = runs_len * @sizeOf(Run(T));

    return @as(f64, @floatFromInt(original_bytes)) / @as(f64, @floatFromInt(compressed_bytes));
}

/// Check if RLE would compress the data (ratio > 1)
///
/// Time: O(n) - must scan to count runs
/// Space: O(1)
pub fn wouldCompress(comptime T: type, data: []const T) bool {
    if (data.len == 0) return false;

    var run_count: usize = 1;
    for (data[1..]) |value| {
        if (!std.meta.eql(value, data[run_count - 1])) {
            run_count += 1;
        }
    }

    return compressionRatio(T, data.len, run_count) > 1.0;
}

/// Run representation: (count, value) pair
pub fn Run(comptime T: type) type {
    return struct {
        count: u32,
        value: T,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "RLE: basic encoding" {
    const input = [_]u8{ 'A', 'A', 'A', 'B', 'B', 'C' };
    var runs = try encode(u8, testing.allocator, &input);
    defer runs.deinit();

    try testing.expectEqual(@as(usize, 3), runs.items.len);
    try testing.expectEqual(@as(u32, 3), runs.items[0].count);
    try testing.expectEqual(@as(u8, 'A'), runs.items[0].value);
    try testing.expectEqual(@as(u32, 2), runs.items[1].count);
    try testing.expectEqual(@as(u8, 'B'), runs.items[1].value);
    try testing.expectEqual(@as(u32, 1), runs.items[2].count);
    try testing.expectEqual(@as(u8, 'C'), runs.items[2].value);
}

test "RLE: basic decoding" {
    const runs = [_]Run(u8){
        .{ .count = 3, .value = 'A' },
        .{ .count = 2, .value = 'B' },
        .{ .count = 1, .value = 'C' },
    };
    const result = try decode(u8, testing.allocator, &runs);
    defer testing.allocator.free(result);

    const expected = [_]u8{ 'A', 'A', 'A', 'B', 'B', 'C' };
    try testing.expectEqualSlices(u8, &expected, result);
}

test "RLE: roundtrip encode-decode" {
    const original = [_]u8{ 1, 1, 1, 2, 2, 3, 3, 3, 3, 4 };
    var runs = try encode(u8, testing.allocator, &original);
    defer runs.deinit();

    const decoded = try decode(u8, testing.allocator, runs.items);
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(u8, &original, decoded);
}

test "RLE: empty input" {
    const input: []const u8 = &[_]u8{};
    var runs = try encode(u8, testing.allocator, input);
    defer runs.deinit();

    try testing.expectEqual(@as(usize, 0), runs.items.len);

    const decoded = try decode(u8, testing.allocator, runs.items);
    defer testing.allocator.free(decoded);
    try testing.expectEqual(@as(usize, 0), decoded.len);
}

test "RLE: single element" {
    const input = [_]u8{'X'};
    var runs = try encode(u8, testing.allocator, &input);
    defer runs.deinit();

    try testing.expectEqual(@as(usize, 1), runs.items.len);
    try testing.expectEqual(@as(u32, 1), runs.items[0].count);
    try testing.expectEqual(@as(u8, 'X'), runs.items[0].value);
}

test "RLE: all identical" {
    const input = [_]u8{ 'Z', 'Z', 'Z', 'Z', 'Z' };
    var runs = try encode(u8, testing.allocator, &input);
    defer runs.deinit();

    try testing.expectEqual(@as(usize, 1), runs.items.len);
    try testing.expectEqual(@as(u32, 5), runs.items[0].count);
    try testing.expectEqual(@as(u8, 'Z'), runs.items[0].value);
}

test "RLE: all unique (worst case)" {
    const input = [_]u8{ 'A', 'B', 'C', 'D', 'E' };
    var runs = try encode(u8, testing.allocator, &input);
    defer runs.deinit();

    try testing.expectEqual(@as(usize, 5), runs.items.len);
    for (runs.items) |run| {
        try testing.expectEqual(@as(u32, 1), run.count);
    }
}

test "RLE: compression ratio - highly compressible" {
    const input = [_]u8{ 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A' };
    var runs = try encode(u8, testing.allocator, &input);
    defer runs.deinit();

    const ratio = compressionRatio(u8, input.len, runs.items.len);
    try testing.expect(ratio > 1.0); // Should compress well
}

test "RLE: compression ratio - not compressible" {
    const input = [_]u8{ 'A', 'B', 'C', 'D' };
    var runs = try encode(u8, testing.allocator, &input);
    defer runs.deinit();

    const ratio = compressionRatio(u8, input.len, runs.items.len);
    try testing.expect(ratio < 1.0); // Should expand (worse than original)
}

test "RLE: wouldCompress check" {
    const compressible = [_]u8{ 'A', 'A', 'A', 'B', 'B', 'B' };
    try testing.expect(wouldCompress(u8, &compressible));

    const not_compressible = [_]u8{ 'A', 'B', 'C', 'D' };
    try testing.expect(!wouldCompress(u8, &not_compressible));
}

test "RLE: numeric types" {
    const input = [_]i32{ 10, 10, 20, 20, 20, 30 };
    var runs = try encode(i32, testing.allocator, &input);
    defer runs.deinit();

    try testing.expectEqual(@as(usize, 3), runs.items.len);
    try testing.expectEqual(@as(u32, 2), runs.items[0].count);
    try testing.expectEqual(@as(i32, 10), runs.items[0].value);
}

test "RLE: large run count" {
    const allocator = testing.allocator;
    const size = 10000;
    const data = try allocator.alloc(u8, size);
    defer allocator.free(data);

    @memset(data, 'X');

    var runs = try encode(u8, allocator, data);
    defer runs.deinit();

    try testing.expectEqual(@as(usize, 1), runs.items.len);
    try testing.expectEqual(@as(u32, 10000), runs.items[0].count);
}

test "RLE: memory safety" {
    const input = [_]u8{ 'A', 'A', 'B' };
    var runs = try encode(u8, testing.allocator, &input);
    defer runs.deinit();

    const decoded = try decode(u8, testing.allocator, runs.items);
    defer testing.allocator.free(decoded);

    // No memory leaks detected by testing.allocator
}
