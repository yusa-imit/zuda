const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// Delta Encoding - Lossless compression for sequential numerical data
///
/// Stores first value and differences between consecutive values instead
/// of absolute values. Effective for slowly-changing sequences (time series,
/// sensor data, audio samples).
///
/// Format: [base_value, delta1, delta2, ..., deltaN-1]
///
/// Time: O(n) for both encode and decode
/// Space: O(n) - same size but often smaller values (better for entropy coding)

/// Encode data using Delta Encoding
///
/// Returns ArrayList where first element is the base value and remaining
/// elements are deltas (differences) between consecutive values.
///
/// Time: O(n) where n is input length
/// Space: O(n)
///
/// Example:
/// ```zig
/// const input = [_]i32{ 100, 102, 101, 105, 103 };
/// var result = try encode(i32, allocator, &input);
/// defer result.deinit();
/// // result = [100, 2, -1, 4, -2]
/// ```
pub fn encode(comptime T: type, allocator: Allocator, data: []const T) !std.ArrayList(T) {
    comptime {
        if (@typeInfo(T) != .int and @typeInfo(T) != .float) {
            @compileError("Delta encoding requires numeric type (int or float)");
        }
    }

    var result = std.ArrayList(T).init(allocator);
    errdefer result.deinit();

    if (data.len == 0) return result;

    // First element is the base value
    try result.append(data[0]);

    // Remaining elements are deltas
    for (data[1..], 1..) |value, i| {
        const delta = value - data[i - 1];
        try result.append(delta);
    }

    return result;
}

/// Decode delta-encoded data back to original form
///
/// Takes ArrayList where first element is base and remaining are deltas,
/// reconstructs original sequence by accumulating differences.
///
/// Time: O(n) where n is input length
/// Space: O(n) for output buffer
///
/// Example:
/// ```zig
/// const encoded = [_]i32{ 100, 2, -1, 4, -2 };
/// var result = try decode(i32, allocator, &encoded);
/// defer allocator.free(result);
/// // result = [100, 102, 101, 105, 103]
/// ```
pub fn decode(comptime T: type, allocator: Allocator, encoded: []const T) ![]T {
    comptime {
        if (@typeInfo(T) != .int and @typeInfo(T) != .float) {
            @compileError("Delta decoding requires numeric type (int or float)");
        }
    }

    var output = try allocator.alloc(T, encoded.len);
    errdefer allocator.free(output);

    if (encoded.len == 0) return output;

    // First element is the base value
    output[0] = encoded[0];

    // Reconstruct by accumulating deltas
    for (encoded[1..], 1..) |delta, i| {
        output[i] = output[i - 1] + delta;
    }

    return output;
}

/// Second-order delta encoding (delta of deltas)
///
/// Stores base, first delta, then delta of deltas. Effective for
/// sequences with constant rate of change (linear trends, polynomial curves).
///
/// Time: O(n)
/// Space: O(n)
pub fn encodeSecondOrder(comptime T: type, allocator: Allocator, data: []const T) !std.ArrayList(T) {
    comptime {
        if (@typeInfo(T) != .int and @typeInfo(T) != .float) {
            @compileError("Second-order delta encoding requires numeric type");
        }
    }

    var result = std.ArrayList(T).init(allocator);
    errdefer result.deinit();

    if (data.len == 0) return result;
    if (data.len == 1) {
        try result.append(data[0]);
        return result;
    }

    // Base value and first delta
    try result.append(data[0]);
    try result.append(data[1] - data[0]);

    // Second-order deltas (delta of deltas)
    var prev_delta = data[1] - data[0];
    for (data[2..], 1..) |value, i| {
        const current_delta = value - data[i];
        const delta_delta = current_delta - prev_delta;
        try result.append(delta_delta);
        prev_delta = current_delta;
    }

    return result;
}

/// Decode second-order delta-encoded data
///
/// Time: O(n)
/// Space: O(n)
pub fn decodeSecondOrder(comptime T: type, allocator: Allocator, encoded: []const T) ![]T {
    comptime {
        if (@typeInfo(T) != .int and @typeInfo(T) != .float) {
            @compileError("Second-order delta decoding requires numeric type");
        }
    }

    var output = try allocator.alloc(T, encoded.len);
    errdefer allocator.free(output);

    if (encoded.len == 0) return output;
    if (encoded.len == 1) {
        output[0] = encoded[0];
        return output;
    }

    // Reconstruct base and first value
    output[0] = encoded[0];
    output[1] = output[0] + encoded[1];

    // Reconstruct from second-order deltas
    var current_delta = encoded[1];
    for (encoded[2..], 2..) |delta_delta, i| {
        current_delta = current_delta + delta_delta;
        output[i] = output[i - 1] + current_delta;
    }

    return output;
}

/// Calculate average delta magnitude for analysis
///
/// Time: O(n)
/// Space: O(1)
pub fn averageDelta(comptime T: type, data: []const T) f64 {
    if (data.len < 2) return 0.0;

    var sum: f64 = 0.0;
    for (data[1..], 1..) |value, i| {
        const delta = @abs(value - data[i - 1]);
        sum += @as(f64, @floatFromInt(delta));
    }

    return sum / @as(f64, @floatFromInt(data.len - 1));
}

// ============================================================================
// Tests
// ============================================================================

test "Delta: basic encoding" {
    const input = [_]i32{ 100, 102, 101, 105, 103 };
    var result = try encode(i32, testing.allocator, &input);
    defer result.deinit();

    const expected = [_]i32{ 100, 2, -1, 4, -2 };
    try testing.expectEqualSlices(i32, &expected, result.items);
}

test "Delta: basic decoding" {
    const encoded = [_]i32{ 100, 2, -1, 4, -2 };
    const result = try decode(i32, testing.allocator, &encoded);
    defer testing.allocator.free(result);

    const expected = [_]i32{ 100, 102, 101, 105, 103 };
    try testing.expectEqualSlices(i32, &expected, result);
}

test "Delta: roundtrip encode-decode" {
    const original = [_]i32{ 10, 15, 12, 20, 18, 25 };
    var encoded = try encode(i32, testing.allocator, &original);
    defer encoded.deinit();

    const decoded = try decode(i32, testing.allocator, encoded.items);
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(i32, &original, decoded);
}

test "Delta: empty input" {
    const input: []const i32 = &[_]i32{};
    var encoded = try encode(i32, testing.allocator, input);
    defer encoded.deinit();

    try testing.expectEqual(@as(usize, 0), encoded.items.len);

    const decoded = try decode(i32, testing.allocator, encoded.items);
    defer testing.allocator.free(decoded);
    try testing.expectEqual(@as(usize, 0), decoded.len);
}

test "Delta: single element" {
    const input = [_]i32{42};
    var encoded = try encode(i32, testing.allocator, &input);
    defer encoded.deinit();

    try testing.expectEqual(@as(usize, 1), encoded.items.len);
    try testing.expectEqual(@as(i32, 42), encoded.items[0]);
}

test "Delta: constant sequence (all zeros)" {
    const input = [_]i32{ 100, 100, 100, 100 };
    var encoded = try encode(i32, testing.allocator, &input);
    defer encoded.deinit();

    const expected = [_]i32{ 100, 0, 0, 0 };
    try testing.expectEqualSlices(i32, &expected, encoded.items);
}

test "Delta: linear sequence" {
    const input = [_]i32{ 0, 5, 10, 15, 20 };
    var encoded = try encode(i32, testing.allocator, &input);
    defer encoded.deinit();

    const expected = [_]i32{ 0, 5, 5, 5, 5 };
    try testing.expectEqualSlices(i32, &expected, encoded.items);
}

test "Delta: negative values" {
    const input = [_]i32{ -10, -5, -8, -2 };
    var encoded = try encode(i32, testing.allocator, &input);
    defer encoded.deinit();

    const expected = [_]i32{ -10, 5, -3, 6 };
    try testing.expectEqualSlices(i32, &expected, encoded.items);
}

test "Delta: unsigned integers" {
    const input = [_]u32{ 100, 105, 102, 110 };
    var encoded = try encode(u32, testing.allocator, &input);
    defer encoded.deinit();

    // Note: deltas can wrap for unsigned (5, -3 wraps, 8)
    try testing.expectEqual(@as(u32, 100), encoded.items[0]);
    try testing.expectEqual(@as(u32, 5), encoded.items[1]);
}

test "Delta: floating point" {
    const input = [_]f32{ 1.0, 1.5, 1.3, 2.0 };
    var encoded = try encode(f32, testing.allocator, &input);
    defer encoded.deinit();

    const decoded = try decode(f32, testing.allocator, encoded.items);
    defer testing.allocator.free(decoded);

    for (input, decoded) |orig, dec| {
        try testing.expectApproxEqAbs(orig, dec, 1e-6);
    }
}

test "Delta: second-order encoding" {
    const input = [_]i32{ 0, 1, 4, 9, 16 }; // y = x^2
    var encoded = try encodeSecondOrder(i32, testing.allocator, &input);
    defer encoded.deinit();

    // Second-order deltas should be constant (2) for quadratic
    const expected = [_]i32{ 0, 1, 2, 2, 2 };
    try testing.expectEqualSlices(i32, &expected, encoded.items);
}

test "Delta: second-order roundtrip" {
    const original = [_]i32{ 5, 8, 13, 20, 29 };
    var encoded = try encodeSecondOrder(i32, testing.allocator, &original);
    defer encoded.deinit();

    const decoded = try decodeSecondOrder(i32, testing.allocator, encoded.items);
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(i32, &original, decoded);
}

test "Delta: second-order empty" {
    const input: []const i32 = &[_]i32{};
    var encoded = try encodeSecondOrder(i32, testing.allocator, input);
    defer encoded.deinit();

    try testing.expectEqual(@as(usize, 0), encoded.items.len);
}

test "Delta: second-order single element" {
    const input = [_]i32{42};
    var encoded = try encodeSecondOrder(i32, testing.allocator, &input);
    defer encoded.deinit();

    try testing.expectEqual(@as(usize, 1), encoded.items.len);
    try testing.expectEqual(@as(i32, 42), encoded.items[0]);
}

test "Delta: average delta calculation" {
    const input = [_]i32{ 10, 15, 12, 20 };
    const avg = averageDelta(i32, &input);

    // Deltas: |15-10|=5, |12-15|=3, |20-12|=8, avg = 16/3 ≈ 5.33
    try testing.expectApproxEqAbs(5.33, avg, 0.01);
}

test "Delta: large sequence" {
    const allocator = testing.allocator;
    const size = 1000;
    const data = try allocator.alloc(i32, size);
    defer allocator.free(data);

    // Linear sequence
    for (data, 0..) |*val, i| {
        val.* = @intCast(i * 10);
    }

    var encoded = try encode(i32, allocator, data);
    defer encoded.deinit();

    const decoded = try decode(i32, allocator, encoded.items);
    defer allocator.free(decoded);

    try testing.expectEqualSlices(i32, data, decoded);
}

test "Delta: memory safety" {
    const input = [_]i32{ 1, 2, 3 };
    var encoded = try encode(i32, testing.allocator, &input);
    defer encoded.deinit();

    const decoded = try decode(i32, testing.allocator, encoded.items);
    defer testing.allocator.free(decoded);

    // No memory leaks detected by testing.allocator
}
