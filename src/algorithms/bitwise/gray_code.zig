//! Gray Code algorithms for binary-to-Gray and Gray-to-binary conversions.
//!
//! Gray code (also known as reflected binary code) is a binary numeral system
//! where consecutive values differ by only one bit. This property is useful in
//! digital systems, error correction, and position encoding.
//!
//! ## Key Properties
//!
//! - Adjacent values differ by exactly one bit (unit distance property)
//! - Cyclic: wraps around (2^n-1 and 0 differ by one bit)
//! - Reflected: second half is reverse of first half with MSB set
//! - Bijective: one-to-one mapping between binary and Gray code
//!
//! ## Algorithms
//!
//! - Binary to Gray: g = n XOR (n >> 1)
//! - Gray to Binary: accumulate XORs from MSB to LSB
//! - Generate sequence: iterative or recursive construction
//! - Next/Previous: single bit flip
//!
//! ## Use Cases
//!
//! - Position encoders (rotary, linear)
//! - Error correction in digital systems
//! - Genetic algorithms (mutation operators)
//! - Karnaugh maps (logic minimization)
//! - Analog-to-digital converters
//! - Puzzle solving (Tower of Hanoi)
//!
//! ## Reference
//!
//! Frank Gray (1953) - "Pulse Code Communication"

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Convert binary to Gray code.
///
/// Time: O(1)
/// Space: O(1)
pub fn binaryToGray(comptime T: type, n: T) T {
    return n ^ (n >> 1);
}

/// Convert Gray code to binary.
///
/// Time: O(log n) where n is number of bits
/// Space: O(1)
pub fn grayToBinary(comptime T: type, gray: T) T {
    var binary = gray;
    var shift: u6 = 1;
    while (shift < @bitSizeOf(T)) : (shift *= 2) {
        binary ^= binary >> shift;
    }
    return binary;
}

/// Generate Gray code sequence of n bits (2^n values).
///
/// Returns ArrayList containing the sequence.
///
/// Time: O(2^n)
/// Space: O(2^n)
pub fn generateSequence(comptime T: type, allocator: Allocator, n: u8) !std.ArrayList(T) {
    if (n > @bitSizeOf(T)) return error.OutOfRange;
    
    const count = @as(usize, 1) << @intCast(n);
    var result = try std.ArrayList(T).initCapacity(allocator, count);
    
    var i: usize = 0;
    while (i < count) : (i += 1) {
        result.appendAssumeCapacity(binaryToGray(T, @intCast(i)));
    }
    
    return result;
}

/// Get the next Gray code value (single bit flip).
///
/// Returns the next value in Gray code sequence and the bit position that changed.
///
/// Time: O(log n)
/// Space: O(1)
pub fn nextGray(comptime T: type, current: T) struct { value: T, bit_pos: u8 } {
    const binary = grayToBinary(T, current);
    const next_binary = binary +% 1; // wrapping add
    const next = binaryToGray(T, next_binary);
    
    // Find which bit changed
    const diff = current ^ next;
    const bit_pos = @ctz(diff);
    
    return .{ .value = next, .bit_pos = @intCast(bit_pos) };
}

/// Get the previous Gray code value (single bit flip).
///
/// Returns the previous value in Gray code sequence and the bit position that changed.
///
/// Time: O(log n)
/// Space: O(1)
pub fn previousGray(comptime T: type, current: T) struct { value: T, bit_pos: u8 } {
    const binary = grayToBinary(T, current);
    const prev_binary = binary -% 1; // wrapping subtract
    const prev = binaryToGray(T, prev_binary);
    
    // Find which bit changed
    const diff = current ^ prev;
    const bit_pos = @ctz(diff);
    
    return .{ .value = prev, .bit_pos = @intCast(bit_pos) };
}

/// Verify if two values are adjacent in Gray code (differ by one bit).
///
/// Time: O(1)
/// Space: O(1)
pub fn areAdjacent(comptime T: type, a: T, b: T) bool {
    const diff = a ^ b;
    return @popCount(diff) == 1;
}

/// Generate reflected Gray code recursively.
///
/// This demonstrates the reflective property: sequence for n bits
/// is formed by prefixing 0 to (n-1) sequence, then 1 to reversed (n-1) sequence.
///
/// Time: O(2^n)
/// Space: O(2^n)
pub fn generateReflected(comptime T: type, allocator: Allocator, n: u8) !std.ArrayList(T) {
    if (n > @bitSizeOf(T)) return error.OutOfRange;
    if (n == 0) {
        var result = try std.ArrayList(T).initCapacity(allocator, 1);
        result.appendAssumeCapacity(0);
        return result;
    }
    
    // Get (n-1) bit sequence
    var prev = try generateReflected(T, allocator, n - 1);
    defer prev.deinit();
    
    const count = @as(usize, 1) << @intCast(n);
    var result = try std.ArrayList(T).initCapacity(allocator, count);
    
    // First half: prefix 0 (no change needed)
    for (prev.items) |val| {
        result.appendAssumeCapacity(val);
    }
    
    // Second half: prefix 1 (set MSB) and reverse order
    const msb: T = @as(T, 1) << @intCast(n - 1);
    var i = prev.items.len;
    while (i > 0) {
        i -= 1;
        result.appendAssumeCapacity(prev.items[i] | msb);
    }
    
    return result;
}

/// Get the bit position that changes between position i and i+1 in Gray code.
///
/// This is useful for efficient iteration.
///
/// Time: O(1)
/// Space: O(1)
pub fn changingBit(comptime T: type, i: T) u8 {
    // The changing bit is the position of the rightmost 0 bit in i
    return @intCast(@ctz(~i));
}

/// Rank: get position of Gray code value in sequence.
///
/// Time: O(log n)
/// Space: O(1)
pub fn rank(comptime T: type, gray: T) T {
    return grayToBinary(T, gray);
}

/// Unrank: get Gray code value at position.
///
/// Time: O(1)
/// Space: O(1)
pub fn unrank(comptime T: type, pos: T) T {
    return binaryToGray(T, pos);
}

// ============================================================================
// Tests
// ============================================================================

test "binaryToGray - basic conversions" {
    try testing.expectEqual(@as(u8, 0b0000), binaryToGray(u8, 0b0000));
    try testing.expectEqual(@as(u8, 0b0001), binaryToGray(u8, 0b0001));
    try testing.expectEqual(@as(u8, 0b0011), binaryToGray(u8, 0b0010));
    try testing.expectEqual(@as(u8, 0b0010), binaryToGray(u8, 0b0011));
    try testing.expectEqual(@as(u8, 0b0110), binaryToGray(u8, 0b0100));
    try testing.expectEqual(@as(u8, 0b1111), binaryToGray(u8, 0b1010));
}

test "grayToBinary - basic conversions" {
    try testing.expectEqual(@as(u8, 0b0000), grayToBinary(u8, 0b0000));
    try testing.expectEqual(@as(u8, 0b0001), grayToBinary(u8, 0b0001));
    try testing.expectEqual(@as(u8, 0b0010), grayToBinary(u8, 0b0011));
    try testing.expectEqual(@as(u8, 0b0011), grayToBinary(u8, 0b0010));
    try testing.expectEqual(@as(u8, 0b0100), grayToBinary(u8, 0b0110));
    try testing.expectEqual(@as(u8, 0b1010), grayToBinary(u8, 0b1111));
}

test "binaryToGray and grayToBinary - roundtrip" {
    var i: u8 = 0;
    while (i < 255) : (i += 1) {
        const gray = binaryToGray(u8, i);
        const binary = grayToBinary(u8, gray);
        try testing.expectEqual(i, binary);
    }
}

test "generateSequence - 3 bits" {
    var seq = try generateSequence(u8, testing.allocator, 3);
    defer seq.deinit();
    
    const expected = [_]u8{ 0, 1, 3, 2, 6, 7, 5, 4 };
    try testing.expectEqual(expected.len, seq.items.len);
    
    for (expected, 0..) |exp, idx| {
        try testing.expectEqual(exp, seq.items[idx]);
    }
}

test "generateSequence - adjacent values differ by one bit" {
    var seq = try generateSequence(u8, testing.allocator, 4);
    defer seq.deinit();
    
    var i: usize = 0;
    while (i < seq.items.len - 1) : (i += 1) {
        const diff = seq.items[i] ^ seq.items[i + 1];
        try testing.expectEqual(@as(u8, 1), @popCount(diff));
    }
}

test "generateSequence - cyclic property" {
    var seq = try generateSequence(u8, testing.allocator, 4);
    defer seq.deinit();
    
    // Last and first should differ by one bit (cyclic)
    const diff = seq.items[seq.items.len - 1] ^ seq.items[0];
    try testing.expectEqual(@as(u8, 1), @popCount(diff));
}

test "nextGray - basic progression" {
    var current: u8 = 0;
    const expected_sequence = [_]u8{ 0, 1, 3, 2, 6, 7, 5, 4 };
    
    for (expected_sequence[0 .. expected_sequence.len - 1], expected_sequence[1..]) |exp_curr, exp_next| {
        if (current == exp_curr) {
            const result = nextGray(u8, current);
            try testing.expectEqual(exp_next, result.value);
            current = result.value;
        }
    }
}

test "previousGray - basic regression" {
    var current: u8 = 4; // Last in 3-bit sequence
    const expected_sequence = [_]u8{ 4, 5, 7, 6, 2, 3, 1, 0 };
    
    for (expected_sequence[0 .. expected_sequence.len - 1], expected_sequence[1..]) |exp_curr, exp_prev| {
        if (current == exp_curr) {
            const result = previousGray(u8, current);
            try testing.expectEqual(exp_prev, result.value);
            current = result.value;
        }
    }
}

test "nextGray and previousGray - roundtrip" {
    var current: u8 = 0;
    var i: usize = 0;
    while (i < 16) : (i += 1) {
        const next = nextGray(u8, current);
        const prev = previousGray(u8, next.value);
        try testing.expectEqual(current, prev.value);
        current = next.value;
    }
}

test "areAdjacent - valid cases" {
    try testing.expect(areAdjacent(u8, 0b0000, 0b0001));
    try testing.expect(areAdjacent(u8, 0b0001, 0b0011));
    try testing.expect(areAdjacent(u8, 0b0011, 0b0010));
    try testing.expect(areAdjacent(u8, 0b0010, 0b0110));
}

test "areAdjacent - invalid cases" {
    try testing.expect(!areAdjacent(u8, 0b0000, 0b0011));
    try testing.expect(!areAdjacent(u8, 0b0001, 0b0110));
    try testing.expect(!areAdjacent(u8, 0b1111, 0b0000));
}

test "generateReflected - matches iterative" {
    var iterative = try generateSequence(u8, testing.allocator, 4);
    defer iterative.deinit();
    
    var reflected = try generateReflected(u8, testing.allocator, 4);
    defer reflected.deinit();
    
    try testing.expectEqual(iterative.items.len, reflected.items.len);
    for (iterative.items, reflected.items) |iter, refl| {
        try testing.expectEqual(iter, refl);
    }
}

test "changingBit - correct positions" {
    // i=0: ...0000 -> bit 0 changes
    try testing.expectEqual(@as(u8, 0), changingBit(u8, 0));
    // i=1: ...0001 -> bit 1 changes
    try testing.expectEqual(@as(u8, 1), changingBit(u8, 1));
    // i=2: ...0010 -> bit 0 changes
    try testing.expectEqual(@as(u8, 0), changingBit(u8, 2));
    // i=3: ...0011 -> bit 2 changes
    try testing.expectEqual(@as(u8, 2), changingBit(u8, 3));
}

test "rank and unrank - inverse operations" {
    var i: u8 = 0;
    while (i < 100) : (i += 1) {
        const gray = unrank(u8, i);
        const pos = rank(u8, gray);
        try testing.expectEqual(i, pos);
    }
}

test "Gray code - different integer types" {
    // u16
    try testing.expectEqual(@as(u16, 0b0011), binaryToGray(u16, 0b0010));
    try testing.expectEqual(@as(u16, 0b0010), grayToBinary(u16, 0b0011));
    
    // u32
    try testing.expectEqual(@as(u32, 0b0011), binaryToGray(u32, 0b0010));
    try testing.expectEqual(@as(u32, 0b0010), grayToBinary(u32, 0b0011));
    
    // u64
    try testing.expectEqual(@as(u64, 0b0011), binaryToGray(u64, 0b0010));
    try testing.expectEqual(@as(u64, 0b0010), grayToBinary(u64, 0b0011));
}

test "generateSequence - error on overflow" {
    try testing.expectError(error.OutOfRange, generateSequence(u8, testing.allocator, 9));
}

test "generateSequence - large sequence (5 bits)" {
    var seq = try generateSequence(u8, testing.allocator, 5);
    defer seq.deinit();
    
    try testing.expectEqual(@as(usize, 32), seq.items.len);
    
    // Verify all adjacent pairs differ by one bit
    var i: usize = 0;
    while (i < seq.items.len - 1) : (i += 1) {
        try testing.expect(areAdjacent(u8, seq.items[i], seq.items[i + 1]));
    }
}

test "Gray code - memory safety" {
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var seq = try generateSequence(u8, testing.allocator, 4);
        seq.deinit();
        
        var refl = try generateReflected(u8, testing.allocator, 4);
        refl.deinit();
    }
}
