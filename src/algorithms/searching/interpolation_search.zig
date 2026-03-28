//! Interpolation Search Algorithm
//!
//! Interpolation search is an improvement over binary search for uniformly distributed sorted data.
//! It estimates the position of the target value based on linear interpolation.
//! For uniformly distributed data, it achieves O(log log n) average time complexity.

const std = @import("std");
const testing = std.testing;

/// Interpolation search - finds an element equal to the target in uniformly distributed data.
/// Returns the index of the target, or null if not found.
///
/// Requirements:
/// - The slice must be sorted in ascending order
/// - Works best with uniformly distributed numeric data
/// - For non-uniform data, may degrade to O(n) worst case
///
/// Time: O(log log n) average, O(n) worst case | Space: O(1)
pub fn interpolationSearch(
    comptime T: type,
    slice: []const T,
    target: T,
) ?usize {
    comptime {
        const type_info = @typeInfo(T);
        if (type_info != .int and type_info != .float) {
            @compileError("interpolationSearch requires numeric type (int or float)");
        }
    }

    if (slice.len == 0) return null;

    var left: usize = 0;
    var right: usize = slice.len - 1;

    while (left <= right and target >= slice[left] and target <= slice[right]) {
        // Single element check
        if (left == right) {
            if (slice[left] == target) return left;
            return null;
        }

        // Prevent division by zero
        if (slice[right] == slice[left]) {
            if (slice[left] == target) return left;
            return null;
        }

        // Interpolation formula to estimate position
        const pos = interpolatePosition(T, slice, left, right, target);

        if (slice[pos] == target) {
            return pos;
        }

        if (slice[pos] < target) {
            left = pos + 1;
        } else {
            if (pos == 0) return null;
            right = pos - 1;
        }
    }

    return null;
}

/// Helper function to calculate interpolated position
fn interpolatePosition(
    comptime T: type,
    slice: []const T,
    left: usize,
    right: usize,
    target: T,
) usize {
    const type_info = @typeInfo(T);

    if (type_info == .float) {
        // For floating-point types
        const range: f64 = @floatCast(slice[right] - slice[left]);
        const target_offset: f64 = @floatCast(target - slice[left]);
        const ratio = target_offset / range;
        const pos_offset = ratio * @as(f64, @floatFromInt(right - left));
        const pos = left + @as(usize, @intFromFloat(pos_offset));
        return @min(pos, right);
    } else {
        // For integer types - use integer arithmetic to avoid overflow
        const range = slice[right] - slice[left];
        const target_offset = target - slice[left];

        // Calculate position using 64-bit intermediate to prevent overflow
        const left_64: i64 = @intCast(left);
        const right_64: i64 = @intCast(right);
        const range_64: i64 = @intCast(range);
        const offset_64: i64 = @intCast(target_offset);

        const pos_offset = @divTrunc((right_64 - left_64) * offset_64, range_64);
        const pos = left_64 + pos_offset;

        return @intCast(@min(@max(pos, left_64), right_64));
    }
}

// ============================================================================
// Tests
// ============================================================================

test "interpolationSearch - basic integers" {
    const arr = [_]i32{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };

    try testing.expectEqual(@as(?usize, 0), interpolationSearch(i32, &arr, 10));
    try testing.expectEqual(@as(?usize, 4), interpolationSearch(i32, &arr, 50));
    try testing.expectEqual(@as(?usize, 9), interpolationSearch(i32, &arr, 100));
    try testing.expectEqual(@as(?usize, null), interpolationSearch(i32, &arr, 25));
    try testing.expectEqual(@as(?usize, null), interpolationSearch(i32, &arr, 5));
    try testing.expectEqual(@as(?usize, null), interpolationSearch(i32, &arr, 105));
}

test "interpolationSearch - uniform distribution" {
    // Uniformly distributed data: 0, 10, 20, 30, ..., 990
    const allocator = testing.allocator;
    const n = 100;

    const arr = try allocator.alloc(i32, n);
    defer allocator.free(arr);

    for (arr, 0..) |*val, i| {
        val.* = @as(i32, @intCast(i * 10));
    }

    // Search for values
    try testing.expectEqual(@as(?usize, 0), interpolationSearch(i32, arr, 0));
    try testing.expectEqual(@as(?usize, 25), interpolationSearch(i32, arr, 250));
    try testing.expectEqual(@as(?usize, 50), interpolationSearch(i32, arr, 500));
    try testing.expectEqual(@as(?usize, 99), interpolationSearch(i32, arr, 990));

    // Non-existing values
    try testing.expectEqual(@as(?usize, null), interpolationSearch(i32, arr, 15));
    try testing.expectEqual(@as(?usize, null), interpolationSearch(i32, arr, 995));
}

test "interpolationSearch - empty array" {
    const arr = [_]i32{};
    try testing.expectEqual(@as(?usize, null), interpolationSearch(i32, &arr, 1));
}

test "interpolationSearch - single element" {
    const arr = [_]i32{42};
    try testing.expectEqual(@as(?usize, 0), interpolationSearch(i32, &arr, 42));
    try testing.expectEqual(@as(?usize, null), interpolationSearch(i32, &arr, 1));
    try testing.expectEqual(@as(?usize, null), interpolationSearch(i32, &arr, 100));
}

test "interpolationSearch - two elements" {
    const arr = [_]i32{ 10, 20 };
    try testing.expectEqual(@as(?usize, 0), interpolationSearch(i32, &arr, 10));
    try testing.expectEqual(@as(?usize, 1), interpolationSearch(i32, &arr, 20));
    try testing.expectEqual(@as(?usize, null), interpolationSearch(i32, &arr, 15));
}

test "interpolationSearch - duplicates" {
    const arr = [_]i32{ 10, 20, 20, 20, 30, 40 };
    const result = interpolationSearch(i32, &arr, 20);
    try testing.expect(result != null);
    try testing.expectEqual(@as(i32, 20), arr[result.?]);
}

test "interpolationSearch - all equal" {
    const arr = [_]i32{ 5, 5, 5, 5, 5 };
    try testing.expectEqual(@as(?usize, 0), interpolationSearch(i32, &arr, 5));
    try testing.expectEqual(@as(?usize, null), interpolationSearch(i32, &arr, 3));
    try testing.expectEqual(@as(?usize, null), interpolationSearch(i32, &arr, 7));
}

test "interpolationSearch - floats" {
    const arr = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };

    try testing.expectEqual(@as(?usize, 0), interpolationSearch(f64, &arr, 1.0));
    try testing.expectEqual(@as(?usize, 4), interpolationSearch(f64, &arr, 5.0));
    try testing.expectEqual(@as(?usize, 9), interpolationSearch(f64, &arr, 10.0));
    try testing.expectEqual(@as(?usize, null), interpolationSearch(f64, &arr, 5.5));
}

test "interpolationSearch - large uniform array" {
    const allocator = testing.allocator;
    const n = 10000;

    const arr = try allocator.alloc(i32, n);
    defer allocator.free(arr);

    // Fill with multiples of 10: 0, 10, 20, ...
    for (arr, 0..) |*val, i| {
        val.* = @as(i32, @intCast(i * 10));
    }

    // Should find elements efficiently
    try testing.expectEqual(@as(?usize, 0), interpolationSearch(i32, arr, 0));
    try testing.expectEqual(@as(?usize, 5000), interpolationSearch(i32, arr, 50000));
    try testing.expectEqual(@as(?usize, 9999), interpolationSearch(i32, arr, 99990));

    // Non-existing values
    try testing.expectEqual(@as(?usize, null), interpolationSearch(i32, arr, 5));
    try testing.expectEqual(@as(?usize, null), interpolationSearch(i32, arr, 100000));
}

test "interpolationSearch - negative numbers" {
    const arr = [_]i32{ -100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100 };

    try testing.expectEqual(@as(?usize, 0), interpolationSearch(i32, &arr, -100));
    try testing.expectEqual(@as(?usize, 5), interpolationSearch(i32, &arr, 0));
    try testing.expectEqual(@as(?usize, 10), interpolationSearch(i32, &arr, 100));
    try testing.expectEqual(@as(?usize, null), interpolationSearch(i32, &arr, -50));
}

test "interpolationSearch - non-uniform distribution degrades gracefully" {
    // Quadratic growth: 1, 4, 9, 16, 25, ...
    const arr = [_]i32{ 1, 4, 9, 16, 25, 36, 49, 64, 81, 100 };

    // Should still find values, though not as efficiently
    try testing.expectEqual(@as(?usize, 0), interpolationSearch(i32, &arr, 1));
    try testing.expectEqual(@as(?usize, 4), interpolationSearch(i32, &arr, 25));
    try testing.expectEqual(@as(?usize, 9), interpolationSearch(i32, &arr, 100));
    try testing.expectEqual(@as(?usize, null), interpolationSearch(i32, &arr, 10));
}

test "interpolationSearch - edge cases near boundaries" {
    const arr = [_]i32{ 1, 2, 3, 4, 5, 100 };

    try testing.expectEqual(@as(?usize, 0), interpolationSearch(i32, &arr, 1));
    try testing.expectEqual(@as(?usize, 5), interpolationSearch(i32, &arr, 100));
    try testing.expectEqual(@as(?usize, null), interpolationSearch(i32, &arr, 0));
    try testing.expectEqual(@as(?usize, null), interpolationSearch(i32, &arr, 101));
}
