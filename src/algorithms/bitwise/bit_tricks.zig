//! Bit Manipulation Algorithms
//!
//! This module provides efficient algorithms for common bit manipulation operations.
//! All functions are generic over unsigned integer types.

const std = @import("std");
const testing = std.testing;

/// Count the number of set bits (1s) in an integer using Brian Kernighan's algorithm.
/// Time: O(k) where k is the number of set bits | Space: O(1)
///
/// Example:
/// ```zig
/// const count = popcount(u8, 0b10110100); // Returns 4
/// ```
pub fn popcount(comptime T: type, value: T) u8 {
    comptime {
        if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned) {
            @compileError("popcount requires unsigned integer type");
        }
    }

    var count: u8 = 0;
    var n = value;
    while (n != 0) {
        n &= n - 1; // Clear the lowest set bit
        count += 1;
    }
    return count;
}

/// Count the number of set bits using lookup table (faster for larger types).
/// Time: O(1) for types <= 64 bits | Space: O(1)
pub fn popcountFast(comptime T: type, value: T) u8 {
    comptime {
        if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned) {
            @compileError("popcountFast requires unsigned integer type");
        }
    }

    return @popCount(value);
}

/// Count the number of leading zeros in an integer.
/// Time: O(1) | Space: O(1)
///
/// Example:
/// ```zig
/// const zeros = countLeadingZeros(u8, 0b00001100); // Returns 4
/// ```
pub fn countLeadingZeros(comptime T: type, value: T) u8 {
    comptime {
        if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned) {
            @compileError("countLeadingZeros requires unsigned integer type");
        }
    }

    if (value == 0) return @bitSizeOf(T);
    return @clz(value);
}

/// Count the number of trailing zeros in an integer.
/// Time: O(1) | Space: O(1)
///
/// Example:
/// ```zig
/// const zeros = countTrailingZeros(u8, 0b01101000); // Returns 3
/// ```
pub fn countTrailingZeros(comptime T: type, value: T) u8 {
    comptime {
        if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned) {
            @compileError("countTrailingZeros requires unsigned integer type");
        }
    }

    if (value == 0) return @bitSizeOf(T);
    return @ctz(value);
}

/// Check if a number is a power of two.
/// Time: O(1) | Space: O(1)
///
/// Example:
/// ```zig
/// const is_pow2 = isPowerOfTwo(u32, 16); // Returns true
/// ```
pub fn isPowerOfTwo(comptime T: type, value: T) bool {
    comptime {
        if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned) {
            @compileError("isPowerOfTwo requires unsigned integer type");
        }
    }

    return value != 0 and (value & (value - 1)) == 0;
}

/// Round up to the next power of two.
/// Time: O(1) | Space: O(1)
///
/// Example:
/// ```zig
/// const next = nextPowerOfTwo(u32, 100); // Returns 128
/// ```
pub fn nextPowerOfTwo(comptime T: type, value: T) T {
    comptime {
        if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned) {
            @compileError("nextPowerOfTwo requires unsigned integer type");
        }
    }

    if (value == 0) return 1;
    if (isPowerOfTwo(T, value)) return value;

    const bits = @bitSizeOf(T);
    var v = value - 1;

    // Fill all bits after the most significant bit
    inline for (0..6) |i| { // log2(64) = 6, enough for up to 64-bit integers
        const shift = @as(u8, 1) << @intCast(i);
        if (shift >= bits) break;
        v |= v >> shift;
    }

    return v + 1;
}

/// Compute the parity of an integer (1 if odd number of set bits, 0 if even).
/// Time: O(k) where k is the number of set bits | Space: O(1)
///
/// Example:
/// ```zig
/// const p = parity(u8, 0b10110100); // Returns 0 (4 set bits = even)
/// ```
pub fn parity(comptime T: type, value: T) u1 {
    comptime {
        if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned) {
            @compileError("parity requires unsigned integer type");
        }
    }

    return @intCast(popcount(T, value) & 1);
}

/// Reverse the bits in an integer.
/// Time: O(log n) where n is the bit width | Space: O(1)
///
/// Example:
/// ```zig
/// const rev = reverseBits(u8, 0b10110100); // Returns 0b00101101
/// ```
pub fn reverseBits(comptime T: type, value: T) T {
    comptime {
        if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned) {
            @compileError("reverseBits requires unsigned integer type");
        }
    }

    var result: T = 0;
    var n = value;
    const bits = @bitSizeOf(T);

    for (0..bits) |_| {
        result = (result << 1) | (n & 1);
        n >>= 1;
    }

    return result;
}

/// Convert binary to Gray code.
/// Time: O(1) | Space: O(1)
///
/// Gray code is a binary numeral system where two successive values differ in only one bit.
///
/// Example:
/// ```zig
/// const gray = binaryToGray(u8, 5); // 5 = 0b101, Gray = 0b111
/// ```
pub fn binaryToGray(comptime T: type, value: T) T {
    comptime {
        if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned) {
            @compileError("binaryToGray requires unsigned integer type");
        }
    }

    return value ^ (value >> 1);
}

/// Convert Gray code to binary.
/// Time: O(log n) where n is the bit width | Space: O(1)
///
/// Example:
/// ```zig
/// const binary = grayToBinary(u8, 0b111); // Returns 5 (0b101)
/// ```
pub fn grayToBinary(comptime T: type, gray: T) T {
    comptime {
        if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned) {
            @compileError("grayToBinary requires unsigned integer type");
        }
    }

    var result = gray;
    var shift: u8 = 1;
    const bits = @bitSizeOf(T);

    while (shift < bits) : (shift *= 2) {
        result ^= result >> shift;
    }

    return result;
}

/// Swap two bits at given positions.
/// Time: O(1) | Space: O(1)
///
/// Example:
/// ```zig
/// const swapped = swapBits(u8, 0b10110100, 1, 5); // Swap bits at positions 1 and 5
/// ```
pub fn swapBits(comptime T: type, value: T, i: u8, j: u8) T {
    comptime {
        if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned) {
            @compileError("swapBits requires unsigned integer type");
        }
    }

    const bits = @bitSizeOf(T);
    if (i >= bits or j >= bits) return value;
    if (i == j) return value;

    // Check if bits are different
    if (((value >> @intCast(i)) & 1) != ((value >> @intCast(j)) & 1)) {
        // Toggle both bits
        const mask = (@as(T, 1) << @intCast(i)) | (@as(T, 1) << @intCast(j));
        return value ^ mask;
    }

    return value;
}

/// Extract a bit field from an integer.
/// Time: O(1) | Space: O(1)
///
/// Example:
/// ```zig
/// const field = extractBitField(u8, 0b10110100, 2, 4); // Extract 4 bits starting at position 2: 0b1101
/// ```
pub fn extractBitField(comptime T: type, value: T, position: u8, width: u8) T {
    comptime {
        if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned) {
            @compileError("extractBitField requires unsigned integer type");
        }
    }

    const bits = @bitSizeOf(T);
    if (position >= bits or width == 0) return 0;
    if (position + width > bits) return 0;

    const mask = (@as(T, 1) << @intCast(width)) - 1;
    return (value >> @intCast(position)) & mask;
}

/// Set a bit field in an integer.
/// Time: O(1) | Space: O(1)
///
/// Example:
/// ```zig
/// const result = setBitField(u8, 0b10110100, 2, 4, 0b0101); // Set 4 bits starting at position 2
/// ```
pub fn setBitField(comptime T: type, value: T, position: u8, width: u8, field_value: T) T {
    comptime {
        if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned) {
            @compileError("setBitField requires unsigned integer type");
        }
    }

    const bits = @bitSizeOf(T);
    if (position >= bits or width == 0) return value;
    if (position + width > bits) return value;

    const mask = ((@as(T, 1) << @intCast(width)) - 1) << @intCast(position);
    const cleared = value & ~mask;
    const field = (field_value << @intCast(position)) & mask;
    return cleared | field;
}

/// Find the position of the most significant bit (MSB).
/// Returns the bit position (0-indexed from LSB), or null if value is 0.
/// Time: O(1) | Space: O(1)
///
/// Example:
/// ```zig
/// const msb = findMSB(u8, 0b01101000); // Returns 6
/// ```
pub fn findMSB(comptime T: type, value: T) ?u8 {
    comptime {
        if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned) {
            @compileError("findMSB requires unsigned integer type");
        }
    }

    if (value == 0) return null;
    const bits = @bitSizeOf(T);
    const leading = countLeadingZeros(T, value);
    return @intCast(bits - leading - 1);
}

/// Find the position of the least significant bit (LSB).
/// Returns the bit position (0-indexed from LSB), or null if value is 0.
/// Time: O(1) | Space: O(1)
///
/// Example:
/// ```zig
/// const lsb = findLSB(u8, 0b01101000); // Returns 3
/// ```
pub fn findLSB(comptime T: type, value: T) ?u8 {
    comptime {
        if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned) {
            @compileError("findLSB requires unsigned integer type");
        }
    }

    if (value == 0) return null;
    return countTrailingZeros(T, value);
}

/// Isolate the rightmost set bit (keep only the lowest 1).
/// Time: O(1) | Space: O(1)
///
/// Example:
/// ```zig
/// const isolated = isolateRightmostBit(u8, 0b01101000); // Returns 0b00001000
/// ```
pub fn isolateRightmostBit(comptime T: type, value: T) T {
    comptime {
        if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned) {
            @compileError("isolateRightmostBit requires unsigned integer type");
        }
    }

    return value & (~value +% 1);
}

/// Clear the rightmost set bit.
/// Time: O(1) | Space: O(1)
///
/// Example:
/// ```zig
/// const cleared = clearRightmostBit(u8, 0b01101000); // Returns 0b01100000
/// ```
pub fn clearRightmostBit(comptime T: type, value: T) T {
    comptime {
        if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned) {
            @compileError("clearRightmostBit requires unsigned integer type");
        }
    }

    return value & (value - 1);
}

// ============================================================================
// Tests
// ============================================================================

test "popcount" {
    try testing.expectEqual(@as(u8, 0), popcount(u8, 0b00000000));
    try testing.expectEqual(@as(u8, 1), popcount(u8, 0b00000001));
    try testing.expectEqual(@as(u8, 4), popcount(u8, 0b10110100));
    try testing.expectEqual(@as(u8, 8), popcount(u8, 0b11111111));
    try testing.expectEqual(@as(u8, 16), popcount(u32, 0xFFFF));
    try testing.expectEqual(@as(u8, 32), popcount(u64, 0xFFFFFFFF));
}

test "popcountFast" {
    try testing.expectEqual(@as(u8, 0), popcountFast(u8, 0b00000000));
    try testing.expectEqual(@as(u8, 1), popcountFast(u8, 0b00000001));
    try testing.expectEqual(@as(u8, 4), popcountFast(u8, 0b10110100));
    try testing.expectEqual(@as(u8, 8), popcountFast(u8, 0b11111111));
}

test "countLeadingZeros" {
    try testing.expectEqual(@as(u8, 8), countLeadingZeros(u8, 0));
    try testing.expectEqual(@as(u8, 7), countLeadingZeros(u8, 0b00000001));
    try testing.expectEqual(@as(u8, 4), countLeadingZeros(u8, 0b00001100));
    try testing.expectEqual(@as(u8, 0), countLeadingZeros(u8, 0b10000000));
    try testing.expectEqual(@as(u8, 0), countLeadingZeros(u8, 0xFF));
}

test "countTrailingZeros" {
    try testing.expectEqual(@as(u8, 8), countTrailingZeros(u8, 0));
    try testing.expectEqual(@as(u8, 0), countTrailingZeros(u8, 0b00000001));
    try testing.expectEqual(@as(u8, 3), countTrailingZeros(u8, 0b01101000));
    try testing.expectEqual(@as(u8, 7), countTrailingZeros(u8, 0b10000000));
}

test "isPowerOfTwo" {
    try testing.expect(!isPowerOfTwo(u32, 0));
    try testing.expect(isPowerOfTwo(u32, 1));
    try testing.expect(isPowerOfTwo(u32, 2));
    try testing.expect(!isPowerOfTwo(u32, 3));
    try testing.expect(isPowerOfTwo(u32, 4));
    try testing.expect(!isPowerOfTwo(u32, 5));
    try testing.expect(isPowerOfTwo(u32, 16));
    try testing.expect(!isPowerOfTwo(u32, 100));
    try testing.expect(isPowerOfTwo(u32, 1024));
}

test "nextPowerOfTwo" {
    try testing.expectEqual(@as(u32, 1), nextPowerOfTwo(u32, 0));
    try testing.expectEqual(@as(u32, 1), nextPowerOfTwo(u32, 1));
    try testing.expectEqual(@as(u32, 2), nextPowerOfTwo(u32, 2));
    try testing.expectEqual(@as(u32, 4), nextPowerOfTwo(u32, 3));
    try testing.expectEqual(@as(u32, 16), nextPowerOfTwo(u32, 9));
    try testing.expectEqual(@as(u32, 128), nextPowerOfTwo(u32, 100));
    try testing.expectEqual(@as(u32, 1024), nextPowerOfTwo(u32, 1000));
}

test "parity" {
    try testing.expectEqual(@as(u1, 0), parity(u8, 0b00000000));
    try testing.expectEqual(@as(u1, 1), parity(u8, 0b00000001));
    try testing.expectEqual(@as(u1, 0), parity(u8, 0b10110100)); // 4 bits = even
    try testing.expectEqual(@as(u1, 1), parity(u8, 0b10110101)); // 5 bits = odd
}

test "reverseBits" {
    try testing.expectEqual(@as(u8, 0b00000000), reverseBits(u8, 0b00000000));
    try testing.expectEqual(@as(u8, 0b10000000), reverseBits(u8, 0b00000001));
    try testing.expectEqual(@as(u8, 0b00101101), reverseBits(u8, 0b10110100));
    try testing.expectEqual(@as(u8, 0b11111111), reverseBits(u8, 0b11111111));
}

test "binaryToGray" {
    try testing.expectEqual(@as(u8, 0b000), binaryToGray(u8, 0b000));
    try testing.expectEqual(@as(u8, 0b001), binaryToGray(u8, 0b001));
    try testing.expectEqual(@as(u8, 0b011), binaryToGray(u8, 0b010));
    try testing.expectEqual(@as(u8, 0b010), binaryToGray(u8, 0b011));
    try testing.expectEqual(@as(u8, 0b110), binaryToGray(u8, 0b100));
    try testing.expectEqual(@as(u8, 0b111), binaryToGray(u8, 0b101));
}

test "grayToBinary" {
    try testing.expectEqual(@as(u8, 0b000), grayToBinary(u8, 0b000));
    try testing.expectEqual(@as(u8, 0b001), grayToBinary(u8, 0b001));
    try testing.expectEqual(@as(u8, 0b010), grayToBinary(u8, 0b011));
    try testing.expectEqual(@as(u8, 0b011), grayToBinary(u8, 0b010));
    try testing.expectEqual(@as(u8, 0b100), grayToBinary(u8, 0b110));
    try testing.expectEqual(@as(u8, 0b101), grayToBinary(u8, 0b111));
}

test "Gray code round-trip" {
    // Test that binary -> gray -> binary is identity
    for (0..256) |i| {
        const value: u8 = @intCast(i);
        const gray = binaryToGray(u8, value);
        const back = grayToBinary(u8, gray);
        try testing.expectEqual(value, back);
    }
}

test "swapBits" {
    try testing.expectEqual(@as(u8, 0b10110100), swapBits(u8, 0b10110100, 0, 0)); // No-op
    try testing.expectEqual(@as(u8, 0b10100101), swapBits(u8, 0b10110100, 2, 5));
    try testing.expectEqual(@as(u8, 0b00110101), swapBits(u8, 0b10110100, 0, 7));
}

test "extractBitField" {
    try testing.expectEqual(@as(u8, 0b1101), extractBitField(u8, 0b10110100, 2, 4));
    try testing.expectEqual(@as(u8, 0b101), extractBitField(u8, 0b10110100, 4, 3));
    try testing.expectEqual(@as(u8, 0b0), extractBitField(u8, 0b10110100, 0, 2));
}

test "setBitField" {
    try testing.expectEqual(@as(u8, 0b10010100), setBitField(u8, 0b10110100, 2, 4, 0b0101));
    try testing.expectEqual(@as(u8, 0b01110100), setBitField(u8, 0b10110100, 4, 3, 0b011));
}

test "findMSB" {
    try testing.expectEqual(@as(?u8, null), findMSB(u8, 0));
    try testing.expectEqual(@as(?u8, 0), findMSB(u8, 0b00000001));
    try testing.expectEqual(@as(?u8, 6), findMSB(u8, 0b01101000));
    try testing.expectEqual(@as(?u8, 7), findMSB(u8, 0b10000000));
    try testing.expectEqual(@as(?u8, 15), findMSB(u16, 0xFFFF));
}

test "findLSB" {
    try testing.expectEqual(@as(?u8, null), findLSB(u8, 0));
    try testing.expectEqual(@as(?u8, 0), findLSB(u8, 0b00000001));
    try testing.expectEqual(@as(?u8, 3), findLSB(u8, 0b01101000));
    try testing.expectEqual(@as(?u8, 7), findLSB(u8, 0b10000000));
}

test "isolateRightmostBit" {
    try testing.expectEqual(@as(u8, 0), isolateRightmostBit(u8, 0));
    try testing.expectEqual(@as(u8, 0b00001000), isolateRightmostBit(u8, 0b01101000));
    try testing.expectEqual(@as(u8, 0b00000001), isolateRightmostBit(u8, 0b11111111));
}

test "clearRightmostBit" {
    try testing.expectEqual(@as(u8, 0), clearRightmostBit(u8, 0));
    try testing.expectEqual(@as(u8, 0b01100000), clearRightmostBit(u8, 0b01101000));
    try testing.expectEqual(@as(u8, 0b11111110), clearRightmostBit(u8, 0b11111111));
}

test "bit operations on different integer types" {
    // u16
    try testing.expectEqual(@as(u8, 8), popcount(u16, 0xFF00));
    try testing.expect(isPowerOfTwo(u16, 256));

    // u32
    try testing.expectEqual(@as(u8, 16), popcount(u32, 0xFFFF0000));
    try testing.expect(isPowerOfTwo(u32, 65536));

    // u64
    try testing.expectEqual(@as(u8, 32), popcount(u64, 0xFFFFFFFF00000000));
    try testing.expect(isPowerOfTwo(u64, 1 << 32));
}

test "edge cases: zero values" {
    try testing.expectEqual(@as(u8, 0), popcount(u8, 0));
    try testing.expectEqual(@as(u8, 8), countLeadingZeros(u8, 0));
    try testing.expectEqual(@as(u8, 8), countTrailingZeros(u8, 0));
    try testing.expect(!isPowerOfTwo(u8, 0));
    try testing.expectEqual(@as(u8, 0), reverseBits(u8, 0));
    try testing.expectEqual(@as(?u8, null), findMSB(u8, 0));
    try testing.expectEqual(@as(?u8, null), findLSB(u8, 0));
}

test "edge cases: all bits set" {
    try testing.expectEqual(@as(u8, 8), popcount(u8, 0xFF));
    try testing.expectEqual(@as(u8, 0), countLeadingZeros(u8, 0xFF));
    try testing.expectEqual(@as(u8, 0), countTrailingZeros(u8, 0xFF));
    try testing.expect(!isPowerOfTwo(u8, 0xFF));
    try testing.expectEqual(@as(u8, 0xFF), reverseBits(u8, 0xFF));
}
