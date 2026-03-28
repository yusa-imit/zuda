const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// Burrows-Wheeler Transform (BWT) - Reversible permutation for compression
///
/// BWT rearranges data to group similar characters together, making it
/// highly compressible with RLE or other methods. Used in bzip2.
///
/// Algorithm:
/// 1. Create all rotations of input string
/// 2. Sort rotations lexicographically
/// 3. Output last column of sorted matrix + original row index
///
/// Time: O(n² log n) naive, O(n) with suffix array optimization
/// Space: O(n²) naive, O(n) with suffix array

/// BWT encoding result
pub const BWTResult = struct {
    /// Transformed data (last column of sorted rotations)
    data: []u8,
    /// Index of original string in sorted rotations
    index: usize,

    pub fn deinit(self: *BWTResult, allocator: Allocator) void {
        allocator.free(self.data);
    }
};

/// Encode data using Burrows-Wheeler Transform
///
/// Creates all rotations, sorts them, returns last column and index.
/// Naive O(n² log n) implementation suitable for moderate-sized inputs.
///
/// Time: O(n² log n) where n is data length
/// Space: O(n²) for rotation matrix
///
/// Example:
/// ```zig
/// const input = "banana";
/// var result = try encode(allocator, input);
/// defer result.deinit(allocator);
/// // result.data ≈ "nnbaaa", result.index = position of original
/// ```
pub fn encode(allocator: Allocator, data: []const u8) !BWTResult {
    if (data.len == 0) {
        return BWTResult{
            .data = try allocator.alloc(u8, 0),
            .index = 0,
        };
    }

    const n = data.len;

    // Create all rotations
    const rotations = try allocator.alloc([]u8, n);
    defer {
        for (rotations) |rot| allocator.free(rot);
        allocator.free(rotations);
    }

    for (rotations, 0..) |*rot, i| {
        rot.* = try allocator.alloc(u8, n);
        // Rotation i: data[i..] ++ data[0..i]
        @memcpy(rot.*[0 .. n - i], data[i..]);
        @memcpy(rot.*[n - i ..], data[0..i]);
    }

    // Create indices for indirect sorting
    const indices = try allocator.alloc(usize, n);
    defer allocator.free(indices);
    for (indices, 0..) |*idx, i| idx.* = i;

    // Sort indices by comparing rotations
    const Context = struct {
        rots: [][]u8,

        pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
            return std.mem.order(u8, ctx.rots[a], ctx.rots[b]) == .lt;
        }
    };

    std.mem.sort(usize, indices, Context{ .rots = rotations }, Context.lessThan);

    // Find index of original string (rotation 0)
    var original_index: usize = 0;
    for (indices, 0..) |idx, i| {
        if (idx == 0) {
            original_index = i;
            break;
        }
    }

    // Extract last column
    var output = try allocator.alloc(u8, n);
    for (indices, 0..) |idx, i| {
        output[i] = rotations[idx][n - 1];
    }

    return BWTResult{
        .data = output,
        .index = original_index,
    };
}

/// Decode BWT-transformed data
///
/// Reconstructs original string from last column and index using
/// the first-last property of BWT.
///
/// Time: O(n²) naive implementation
/// Space: O(n²) for reconstruction matrix
///
/// Example:
/// ```zig
/// const result = try encode(allocator, "banana");
/// defer result.deinit(allocator);
/// const decoded = try decode(allocator, result.data, result.index);
/// defer allocator.free(decoded);
/// // decoded = "banana"
/// ```
pub fn decode(allocator: Allocator, data: []const u8, index: usize) ![]u8 {
    if (data.len == 0) return try allocator.alloc(u8, 0);

    const n = data.len;

    // Reconstruct the rotation matrix by iterative construction
    const matrix = try allocator.alloc([]u8, n);
    defer {
        for (matrix) |row| allocator.free(row);
        allocator.free(matrix);
    }

    for (matrix) |*row| {
        row.* = try allocator.alloc(u8, n);
        @memset(row.*, 0);
    }

    // Build matrix: n iterations of prepending last column and sorting
    var i: usize = 0;
    while (i < n) : (i += 1) {
        // Prepend last column (data) to each row
        for (matrix, 0..) |row, j| {
            // Shift right
            var k: usize = n - 1;
            while (k > 0) : (k -= 1) {
                row[k] = row[k - 1];
            }
            row[0] = data[j];
        }

        // Sort rows lexicographically
        std.mem.sort([]u8, matrix, {}, struct {
            fn lessThan(_: void, a: []u8, b: []u8) bool {
                return std.mem.order(u8, a, b) == .lt;
            }
        }.lessThan);
    }

    // Original string is at the specified index
    const output = try allocator.alloc(u8, n);
    @memcpy(output, matrix[index]);

    return output;
}

/// More efficient decode using first-last property
///
/// Uses the fact that BWT preserves character ordering.
/// Each occurrence of a character in the last column corresponds
/// to the same occurrence in the first column.
///
/// Time: O(n)
/// Space: O(n)
pub fn decodeFast(allocator: Allocator, data: []const u8, index: usize) ![]u8 {
    if (data.len == 0) return try allocator.alloc(u8, 0);

    const n = data.len;

    // Build first column (sorted version of last column)
    const first = try allocator.dupe(u8, data);
    defer allocator.free(first);
    std.mem.sort(u8, first, {}, std.sort.asc(u8));

    // Build next[] array: next[i] = position in L where F[i] appears
    const next = try allocator.alloc(usize, n);
    defer allocator.free(next);

    // Count occurrences of each character
    var counts = [_]usize{0} ** 256;
    for (data) |c| counts[c] += 1;

    // Build cumulative counts (position of first occurrence)
    var cumulative = [_]usize{0} ** 256;
    var sum: usize = 0;
    for (cumulative, 0..) |*cum, c| {
        cum.* = sum;
        sum += counts[c];
    }

    // Build next array
    for (data, 0..) |c, i| {
        next[cumulative[c]] = i;
        cumulative[c] += 1;
    }

    // Reconstruct original string by following next pointers
    var output = try allocator.alloc(u8, n);
    var pos = index;
    for (0..n) |i| {
        output[i] = first[pos];
        pos = next[pos];
    }

    return output;
}

/// Calculate BWT compressibility (clustering metric)
///
/// Measures how well BWT groups similar characters together.
/// Higher values indicate better compressibility.
///
/// Time: O(n)
/// Space: O(1)
pub fn compressibility(data: []const u8) f64 {
    if (data.len < 2) return 0.0;

    // Count adjacent identical characters
    var identical_count: usize = 0;
    for (data[1..], 1..) |c, i| {
        if (c == data[i - 1]) identical_count += 1;
    }

    return @as(f64, @floatFromInt(identical_count)) / @as(f64, @floatFromInt(data.len - 1));
}

// ============================================================================
// Tests
// ============================================================================

test "BWT: simple encode" {
    const input = "banana";
    var result = try encode(testing.allocator, input);
    defer result.deinit(testing.allocator);

    // BWT of "banana" should group 'a's together
    try testing.expectEqual(@as(usize, 6), result.data.len);
}

test "BWT: roundtrip encode-decode (naive)" {
    const original = "banana";
    var encoded = try encode(testing.allocator, original);
    defer encoded.deinit(testing.allocator);

    const decoded = try decode(testing.allocator, encoded.data, encoded.index);
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(u8, original, decoded);
}

test "BWT: roundtrip encode-decode (fast)" {
    const original = "banana";
    var encoded = try encode(testing.allocator, original);
    defer encoded.deinit(testing.allocator);

    const decoded = try decodeFast(testing.allocator, encoded.data, encoded.index);
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(u8, original, decoded);
}

test "BWT: empty input" {
    var encoded = try encode(testing.allocator, "");
    defer encoded.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 0), encoded.data.len);
    try testing.expectEqual(@as(usize, 0), encoded.index);

    const decoded = try decodeFast(testing.allocator, encoded.data, encoded.index);
    defer testing.allocator.free(decoded);
    try testing.expectEqual(@as(usize, 0), decoded.len);
}

test "BWT: single character" {
    const input = "a";
    var encoded = try encode(testing.allocator, input);
    defer encoded.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), encoded.data.len);
    try testing.expectEqual(@as(u8, 'a'), encoded.data[0]);

    const decoded = try decodeFast(testing.allocator, encoded.data, encoded.index);
    defer testing.allocator.free(decoded);
    try testing.expectEqualSlices(u8, input, decoded);
}

test "BWT: all same character" {
    const input = "aaaaa";
    var encoded = try encode(testing.allocator, input);
    defer encoded.deinit(testing.allocator);

    const decoded = try decodeFast(testing.allocator, encoded.data, encoded.index);
    defer testing.allocator.free(decoded);
    try testing.expectEqualSlices(u8, input, decoded);
}

test "BWT: repeated pattern" {
    const input = "ababab";
    var encoded = try encode(testing.allocator, input);
    defer encoded.deinit(testing.allocator);

    const decoded = try decodeFast(testing.allocator, encoded.data, encoded.index);
    defer testing.allocator.free(decoded);
    try testing.expectEqualSlices(u8, input, decoded);
}

test "BWT: compressibility metric" {
    const input1 = "banana";
    var encoded1 = try encode(testing.allocator, input1);
    defer encoded1.deinit(testing.allocator);

    const comp1 = compressibility(encoded1.data);

    const input2 = "abcdef";
    var encoded2 = try encode(testing.allocator, input2);
    defer encoded2.deinit(testing.allocator);

    const comp2 = compressibility(encoded2.data);

    // "banana" should have higher compressibility after BWT
    try testing.expect(comp1 > comp2);
}

test "BWT: numeric data" {
    const input = [_]u8{ 1, 2, 3, 1, 2, 3 };
    var encoded = try encode(testing.allocator, &input);
    defer encoded.deinit(testing.allocator);

    const decoded = try decodeFast(testing.allocator, encoded.data, encoded.index);
    defer testing.allocator.free(decoded);
    try testing.expectEqualSlices(u8, &input, decoded);
}

test "BWT: long repeated string" {
    const allocator = testing.allocator;
    const base = "hello";
    const repetitions = 20;

    var input_list = std.ArrayList(u8).init(allocator);
    defer input_list.deinit();

    var i: usize = 0;
    while (i < repetitions) : (i += 1) {
        try input_list.appendSlice(base);
    }

    var encoded = try encode(allocator, input_list.items);
    defer encoded.deinit(allocator);

    // BWT should group identical characters
    const comp = compressibility(encoded.data);
    try testing.expect(comp > 0.5); // High clustering

    const decoded = try decodeFast(allocator, encoded.data, encoded.index);
    defer allocator.free(decoded);
    try testing.expectEqualSlices(u8, input_list.items, decoded);
}

test "BWT: both decode methods agree" {
    const input = "compression";
    var encoded = try encode(testing.allocator, input);
    defer encoded.deinit(testing.allocator);

    const decoded1 = try decode(testing.allocator, encoded.data, encoded.index);
    defer testing.allocator.free(decoded1);

    const decoded2 = try decodeFast(testing.allocator, encoded.data, encoded.index);
    defer testing.allocator.free(decoded2);

    try testing.expectEqualSlices(u8, decoded1, decoded2);
    try testing.expectEqualSlices(u8, input, decoded1);
}

test "BWT: binary data" {
    const input = [_]u8{ 0, 255, 128, 0, 255, 128 };
    var encoded = try encode(testing.allocator, &input);
    defer encoded.deinit(testing.allocator);

    const decoded = try decodeFast(testing.allocator, encoded.data, encoded.index);
    defer testing.allocator.free(decoded);
    try testing.expectEqualSlices(u8, &input, decoded);
}

test "BWT: memory safety" {
    const input = "testdata";
    var encoded = try encode(testing.allocator, input);
    defer encoded.deinit(testing.allocator);

    const decoded = try decodeFast(testing.allocator, encoded.data, encoded.index);
    defer testing.allocator.free(decoded);

    // No memory leaks detected by testing.allocator
}
