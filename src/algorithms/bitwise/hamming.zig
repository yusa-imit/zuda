//! Hamming Distance Algorithms
//!
//! This module provides algorithms for computing Hamming distance between bit patterns.
//! Hamming distance is the number of positions at which corresponding bits differ.
//! It's widely used in error detection/correction, cryptography, and pattern matching.

const std = @import("std");
const testing = std.testing;

/// Compute Hamming distance between two unsigned integers.
/// Time: O(1) | Space: O(1)
///
/// The Hamming distance is the number of bit positions where the two values differ.
/// It's computed by XORing the values and counting the set bits in the result.
///
/// Example:
/// ```zig
/// const dist = hammingDistance(u8, 0b10110100, 0b10010110); // Returns 3
/// // Positions differ at: ___X__X_  ______X_  = 3 differences
/// ```
pub fn hammingDistance(comptime T: type, a: T, b: T) u8 {
    comptime {
        if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned) {
            @compileError("hammingDistance requires unsigned integer type");
        }
    }

    const xor_result = a ^ b;
    return @popCount(xor_result);
}

/// Compute total Hamming distance between all pairs in a slice.
/// Time: O(n²) | Space: O(1)
///
/// Returns the sum of Hamming distances between all unique pairs of elements.
/// Useful for analyzing pattern similarity in a collection.
///
/// Example:
/// ```zig
/// const values = [_]u8{ 0b1010, 0b1100, 0b0011 };
/// const total = totalHammingDistance(u8, &values); // Returns 8
/// ```
pub fn totalHammingDistance(comptime T: type, values: []const T) usize {
    comptime {
        if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned) {
            @compileError("totalHammingDistance requires unsigned integer type");
        }
    }

    var total: usize = 0;
    for (values, 0..) |val_i, i| {
        for (values[i + 1 ..]) |val_j| {
            total += hammingDistance(T, val_i, val_j);
        }
    }
    return total;
}

/// Find the element with minimum Hamming distance to a target.
/// Time: O(n) | Space: O(1)
///
/// Returns the index of the element in values that has the smallest Hamming
/// distance to the target value. If multiple elements have the same minimum
/// distance, returns the first one found.
///
/// Returns error.EmptySlice if values is empty.
///
/// Example:
/// ```zig
/// const values = [_]u8{ 0b1010, 0b1100, 0b0011 };
/// const idx = try findNearestHamming(u8, &values, 0b1000); // Returns 0 (0b1010)
/// ```
pub fn findNearestHamming(comptime T: type, values: []const T, target: T) !usize {
    comptime {
        if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned) {
            @compileError("findNearestHamming requires unsigned integer type");
        }
    }

    if (values.len == 0) return error.EmptySlice;

    var min_distance = hammingDistance(T, values[0], target);
    var min_index: usize = 0;

    for (values[1..], 1..) |val, i| {
        const dist = hammingDistance(T, val, target);
        if (dist < min_distance) {
            min_distance = dist;
            min_index = i;
        }
    }

    return min_index;
}

/// Count elements within a given Hamming distance threshold.
/// Time: O(n) | Space: O(1)
///
/// Returns the number of elements in values that have a Hamming distance
/// to the target that is less than or equal to the threshold.
///
/// Example:
/// ```zig
/// const values = [_]u8{ 0b1010, 0b1100, 0b0011, 0b1001 };
/// const count = countWithinHammingRadius(u8, &values, 0b1000, 2); // Returns 2
/// ```
pub fn countWithinHammingRadius(comptime T: type, values: []const T, target: T, threshold: u8) usize {
    comptime {
        if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned) {
            @compileError("countWithinHammingRadius requires unsigned integer type");
        }
    }

    var count: usize = 0;
    for (values) |val| {
        if (hammingDistance(T, val, target) <= threshold) {
            count += 1;
        }
    }
    return count;
}

/// Compute Hamming distance between two byte slices.
/// Time: O(n) where n = min(a.len, b.len) | Space: O(1)
///
/// Returns error.LengthMismatch if the slices have different lengths.
/// For strings or byte arrays, this is useful for spell-checking, DNA sequence
/// comparison, and error detection.
///
/// Example:
/// ```zig
/// const a = "kitten";
/// const b = "sitten";
/// const dist = try hammingDistanceBytes(a, b); // Returns 1
/// ```
pub fn hammingDistanceBytes(a: []const u8, b: []const u8) !usize {
    if (a.len != b.len) return error.LengthMismatch;

    var distance: usize = 0;
    for (a, b) |byte_a, byte_b| {
        if (byte_a != byte_b) {
            distance += 1;
        }
    }
    return distance;
}

/// Compute bitwise Hamming distance between two byte slices.
/// Time: O(n) where n = min(a.len, b.len) | Space: O(1)
///
/// Unlike hammingDistanceBytes, this counts differing bits, not differing bytes.
/// Returns error.LengthMismatch if the slices have different lengths.
///
/// Example:
/// ```zig
/// const a = [_]u8{ 0b10110100, 0b11001100 };
/// const b = [_]u8{ 0b10010110, 0b11101100 };
/// const dist = try hammingDistanceBitwise(&a, &b); // Returns 4 (3 + 1 bits differ)
/// ```
pub fn hammingDistanceBitwise(a: []const u8, b: []const u8) !usize {
    if (a.len != b.len) return error.LengthMismatch;

    var distance: usize = 0;
    for (a, b) |byte_a, byte_b| {
        distance += hammingDistance(u8, byte_a, byte_b);
    }
    return distance;
}

// ============================================================================
// Tests
// ============================================================================

test "hammingDistance - basic examples" {
    // Same values have distance 0
    try testing.expectEqual(0, hammingDistance(u8, 0b10110100, 0b10110100));

    // Example from docstring
    try testing.expectEqual(3, hammingDistance(u8, 0b10110100, 0b10010110));

    // All bits differ
    try testing.expectEqual(8, hammingDistance(u8, 0b11111111, 0b00000000));

    // Single bit differs
    try testing.expectEqual(1, hammingDistance(u8, 0b10000000, 0b00000000));
}

test "hammingDistance - different types" {
    try testing.expectEqual(0, hammingDistance(u16, 0xFFFF, 0xFFFF));
    try testing.expectEqual(8, hammingDistance(u16, 0xFF00, 0x00FF));
    try testing.expectEqual(16, hammingDistance(u16, 0xFFFF, 0x0000));

    try testing.expectEqual(0, hammingDistance(u32, 0xDEADBEEF, 0xDEADBEEF));
    try testing.expectEqual(1, hammingDistance(u32, 0x00000000, 0x00000001));
}

test "hammingDistance - edge cases" {
    try testing.expectEqual(0, hammingDistance(u8, 0, 0));
    try testing.expectEqual(1, hammingDistance(u64, 0, 1));
}

test "totalHammingDistance - basic examples" {
    const values1 = [_]u8{ 0b1010, 0b1100, 0b0011 };
    // Pairs: (0b1010, 0b1100) = 2, (0b1010, 0b0011) = 3, (0b1100, 0b0011) = 3
    // Total: 2 + 3 + 3 = 8
    try testing.expectEqual(8, totalHammingDistance(u8, &values1));

    const values2 = [_]u8{ 0b1111, 0b1111, 0b1111 };
    try testing.expectEqual(0, totalHammingDistance(u8, &values2));
}

test "totalHammingDistance - empty and single" {
    const empty = [_]u8{};
    try testing.expectEqual(0, totalHammingDistance(u8, &empty));

    const single = [_]u8{0b1010};
    try testing.expectEqual(0, totalHammingDistance(u8, &single));
}

test "findNearestHamming - basic examples" {
    const values = [_]u8{ 0b1010, 0b1100, 0b0011 };

    // Target 0b1000: distances are [1, 2, 3] -> index 0
    try testing.expectEqual(0, try findNearestHamming(u8, &values, 0b1000));

    // Target 0b1111: distances are [2, 1, 2] -> index 1
    try testing.expectEqual(1, try findNearestHamming(u8, &values, 0b1111));

    // Target 0b0000: distances are [2, 2, 2] -> index 0 (first)
    try testing.expectEqual(0, try findNearestHamming(u8, &values, 0b0000));
}

test "findNearestHamming - exact match" {
    const values = [_]u8{ 0b1010, 0b1100, 0b0011 };
    try testing.expectEqual(1, try findNearestHamming(u8, &values, 0b1100));
}

test "findNearestHamming - empty slice error" {
    const empty = [_]u8{};
    try testing.expectError(error.EmptySlice, findNearestHamming(u8, &empty, 0b1010));
}

test "countWithinHammingRadius - basic examples" {
    const values = [_]u8{ 0b1010, 0b1100, 0b0011, 0b1001 };

    // Target 0b1000, threshold 2: distances are [1, 2, 3, 2]
    // Within radius: 0b1010 (1), 0b1100 (2), 0b1001 (2) = 3 elements
    try testing.expectEqual(3, countWithinHammingRadius(u8, &values, 0b1000, 2));

    // Threshold 0: only exact matches
    try testing.expectEqual(0, countWithinHammingRadius(u8, &values, 0b1000, 0));

    // Threshold 8: all elements (max distance for u8 is 8)
    try testing.expectEqual(4, countWithinHammingRadius(u8, &values, 0b1000, 8));
}

test "countWithinHammingRadius - edge cases" {
    const empty = [_]u8{};
    try testing.expectEqual(0, countWithinHammingRadius(u8, &empty, 0b1010, 2));

    const single = [_]u8{0b1010};
    try testing.expectEqual(1, countWithinHammingRadius(u8, &single, 0b1010, 0));
    try testing.expectEqual(0, countWithinHammingRadius(u8, &single, 0b0000, 1));
}

test "hammingDistanceBytes - basic examples" {
    const a1 = "kitten";
    const b1 = "sitten";
    try testing.expectEqual(1, try hammingDistanceBytes(a1, b1));

    const a2 = "abc";
    const b2 = "abc";
    try testing.expectEqual(0, try hammingDistanceBytes(a2, b2));

    const a3 = "abc";
    const b3 = "xyz";
    try testing.expectEqual(3, try hammingDistanceBytes(a3, b3));
}

test "hammingDistanceBytes - length mismatch error" {
    const a = "abc";
    const b = "abcd";
    try testing.expectError(error.LengthMismatch, hammingDistanceBytes(a, b));
}

test "hammingDistanceBitwise - basic examples" {
    const a1 = [_]u8{ 0b10110100, 0b11001100 };
    const b1 = [_]u8{ 0b10010110, 0b11101100 };
    // First byte: 3 bits differ, Second byte: 1 bit differs
    try testing.expectEqual(4, try hammingDistanceBitwise(&a1, &b1));

    const a2 = [_]u8{ 0xFF, 0xFF };
    const b2 = [_]u8{ 0xFF, 0xFF };
    try testing.expectEqual(0, try hammingDistanceBitwise(&a2, &b2));

    const a3 = [_]u8{ 0xFF, 0xFF };
    const b3 = [_]u8{ 0x00, 0x00 };
    try testing.expectEqual(16, try hammingDistanceBitwise(&a3, &b3));
}

test "hammingDistanceBitwise - length mismatch error" {
    const a = [_]u8{0xFF};
    const b = [_]u8{ 0xFF, 0xFF };
    try testing.expectError(error.LengthMismatch, hammingDistanceBitwise(&a, &b));
}

test "hammingDistanceBitwise - empty slices" {
    const empty = [_]u8{};
    try testing.expectEqual(0, try hammingDistanceBitwise(&empty, &empty));
}
