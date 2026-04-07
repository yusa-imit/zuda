//! Subset Generation Algorithms
//!
//! This module provides algorithms for generating and iterating over subsets using bitwise operations.
//! These are fundamental for combinatorial problems, backtracking, and dynamic programming on subsets.

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

/// Iterator for all subsets of a set with n elements.
/// Time: O(1) per iteration | Space: O(1)
///
/// Each subset is represented as a bitmask where bit i indicates whether element i is included.
/// Iterates through all 2^n subsets in lexicographic order.
///
/// Example:
/// ```zig
/// var iter = SubsetIterator.init(3);
/// while (iter.next()) |mask| {
///     // mask will be 0b000, 0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111
/// }
/// ```
pub const SubsetIterator = struct {
    n: u6, // Maximum 64 elements
    current: u64,
    total: u64,

    pub fn init(n: u6) SubsetIterator {
        if (n > 63) @panic("SubsetIterator supports at most 63 elements");
        const total = @as(u64, 1) << @intCast(n);
        return .{
            .n = n,
            .current = 0,
            .total = total,
        };
    }

    pub fn next(self: *SubsetIterator) ?u64 {
        if (self.current >= self.total) return null;
        const result = self.current;
        self.current += 1;
        return result;
    }

    pub fn reset(self: *SubsetIterator) void {
        self.current = 0;
    }
};

/// Iterator for subsets of a specific size k.
/// Time: O(1) per iteration | Space: O(1)
///
/// Generates all C(n, k) subsets of size k from a set of n elements.
/// Uses Gosper's hack to efficiently iterate through combinations.
///
/// Example:
/// ```zig
/// var iter = SubsetOfSizeIterator.init(5, 3);
/// while (iter.next()) |mask| {
///     // mask will have exactly 3 bits set
///     // e.g., 0b00111, 0b01011, 0b01101, 0b01110, 0b10011, ...
/// }
/// ```
pub const SubsetOfSizeIterator = struct {
    n: u6,
    k: u6,
    current: ?u64,

    pub fn init(n: u6, k: u6) SubsetOfSizeIterator {
        if (n > 63) @panic("SubsetOfSizeIterator supports at most 63 elements");
        if (k > n) @panic("k must be <= n");

        // Initial subset: k rightmost bits set
        const initial = if (k == 0) null else (@as(u64, 1) << @intCast(k)) - 1;

        return .{
            .n = n,
            .k = k,
            .current = initial,
        };
    }

    pub fn next(self: *SubsetOfSizeIterator) ?u64 {
        const current = self.current orelse return null;

        // Gosper's hack for next combination
        const c = current;
        const a = c & -%c; // Rightmost set bit
        const b = c +% a; // Add to propagate carry
        const d = c ^ b; // XOR to find changed bits
        const e = (d >> 2) / a; // Right-shift to get next combination
        const result = b | e;

        // Check if we've exceeded n bits
        if (result >= (@as(u64, 1) << @intCast(self.n))) {
            self.current = null;
        } else {
            self.current = result;
        }

        return c;
    }

    pub fn reset(self: *SubsetOfSizeIterator) void {
        self.current = if (self.k == 0) null else (@as(u64, 1) << @intCast(self.k)) - 1;
    }
};

/// Count the number of elements in a subset (popcount of bitmask).
/// Time: O(1) | Space: O(1)
///
/// Example:
/// ```zig
/// const size = subsetSize(0b10110100); // Returns 4
/// ```
pub fn subsetSize(mask: u64) u8 {
    return @popCount(mask);
}

/// Check if element at index i is in the subset.
/// Time: O(1) | Space: O(1)
///
/// Example:
/// ```zig
/// const has_element = isInSubset(0b10110100, 2); // Returns true (bit 2 is set)
/// ```
pub fn isInSubset(mask: u64, index: u6) bool {
    if (index >= 64) return false;
    return (mask & (@as(u64, 1) << @intCast(index))) != 0;
}

/// Generate all subsets and collect them into an ArrayList.
/// Time: O(2^n) | Space: O(2^n)
///
/// Returns error.TooManyElements if n > 20 (to prevent excessive memory usage).
///
/// Example:
/// ```zig
/// const subsets = try generateAllSubsets(allocator, 3);
/// defer subsets.deinit();
/// // subsets.items = [0b000, 0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111]
/// ```
pub fn generateAllSubsets(allocator: Allocator, n: u6) !ArrayList(u64) {
    if (n > 20) return error.TooManyElements; // Prevent 2^n > 1M

    const total = @as(usize, 1) << @intCast(n);
    var result = try ArrayList(u64).initCapacity(allocator, total);
    errdefer result.deinit();

    var iter = SubsetIterator.init(n);
    while (iter.next()) |mask| {
        result.appendAssumeCapacity(mask);
    }

    return result;
}

/// Generate all k-sized subsets and collect them into an ArrayList.
/// Time: O(C(n, k)) | Space: O(C(n, k))
///
/// Returns error.TooManyElements if C(n, k) > 1,000,000.
///
/// Example:
/// ```zig
/// const subsets = try generateSubsetsOfSize(allocator, 5, 3);
/// defer subsets.deinit();
/// // subsets.items = all 10 ways to choose 3 elements from 5
/// ```
pub fn generateSubsetsOfSize(allocator: Allocator, n: u6, k: u6) !ArrayList(u64) {
    if (k > n) return error.InvalidParameters;

    // Estimate binomial coefficient to prevent excessive memory
    const estimate = estimateBinomial(n, k);
    if (estimate > 1_000_000) return error.TooManyElements;

    var result = ArrayList(u64).init(allocator);
    errdefer result.deinit();

    var iter = SubsetOfSizeIterator.init(n, k);
    while (iter.next()) |mask| {
        try result.append(mask);
    }

    return result;
}

/// Iterate through all subsets of a given mask (submasks).
/// Time: O(3^k) for k set bits in mask | Space: O(1)
///
/// This iterates through all 2^k subsets where k is the number of set bits in mask.
/// Useful for dynamic programming on subsets.
///
/// Example:
/// ```zig
/// const mask = 0b1011; // Has 3 set bits
/// var submask = mask;
/// while (true) {
///     // Process submask (will visit 0b1011, 0b1010, 0b1001, 0b1000, 0b0011, 0b0010, 0b0001, 0b0000)
///     if (submask == 0) break;
///     submask = (submask - 1) & mask;
/// }
/// ```
pub const SubmaskIterator = struct {
    mask: u64,
    current: ?u64,

    pub fn init(mask: u64) SubmaskIterator {
        return .{
            .mask = mask,
            .current = mask,
        };
    }

    pub fn next(self: *SubmaskIterator) ?u64 {
        const current = self.current orelse return null;

        if (current == 0) {
            self.current = null;
            return 0;
        }

        const result = current;
        self.current = (current -% 1) & self.mask;
        return result;
    }

    pub fn reset(self: *SubmaskIterator) void {
        self.current = self.mask;
    }
};

// Helper function to estimate binomial coefficient
fn estimateBinomial(n: u6, k: u6) usize {
    if (k > n) return 0;
    if (k == 0 or k == n) return 1;

    const k_actual = if (k > n - k) n - k else k;

    var result: usize = 1;
    var i: usize = 0;
    while (i < k_actual) : (i += 1) {
        result = result * (n - i) / (i + 1);
        if (result > 1_000_000) return result; // Early exit
    }

    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "SubsetIterator - basic iteration" {
    var iter = SubsetIterator.init(3);

    try testing.expectEqual(0b000, iter.next().?);
    try testing.expectEqual(0b001, iter.next().?);
    try testing.expectEqual(0b010, iter.next().?);
    try testing.expectEqual(0b011, iter.next().?);
    try testing.expectEqual(0b100, iter.next().?);
    try testing.expectEqual(0b101, iter.next().?);
    try testing.expectEqual(0b110, iter.next().?);
    try testing.expectEqual(0b111, iter.next().?);
    try testing.expect(iter.next() == null);
}

test "SubsetIterator - empty set" {
    var iter = SubsetIterator.init(0);
    try testing.expectEqual(0, iter.next().?);
    try testing.expect(iter.next() == null);
}

test "SubsetIterator - count" {
    var iter = SubsetIterator.init(5);
    var count: usize = 0;
    while (iter.next()) |_| {
        count += 1;
    }
    try testing.expectEqual(32, count); // 2^5
}

test "SubsetIterator - reset" {
    var iter = SubsetIterator.init(2);
    _ = iter.next();
    _ = iter.next();
    iter.reset();
    try testing.expectEqual(0b00, iter.next().?);
}

test "SubsetOfSizeIterator - k=2, n=4" {
    var iter = SubsetOfSizeIterator.init(4, 2);

    var masks = ArrayList(u64).init(testing.allocator);
    defer masks.deinit();

    while (iter.next()) |mask| {
        try masks.append(mask);
        try testing.expectEqual(2, @popCount(mask));
    }

    // C(4, 2) = 6
    try testing.expectEqual(6, masks.items.len);
}

test "SubsetOfSizeIterator - k=0" {
    var iter = SubsetOfSizeIterator.init(5, 0);
    try testing.expect(iter.next() == null);
}

test "SubsetOfSizeIterator - k=n" {
    var iter = SubsetOfSizeIterator.init(3, 3);
    try testing.expectEqual(0b111, iter.next().?);
    try testing.expect(iter.next() == null);
}

test "SubsetOfSizeIterator - reset" {
    var iter = SubsetOfSizeIterator.init(4, 2);
    const first = iter.next().?;
    _ = iter.next();
    iter.reset();
    try testing.expectEqual(first, iter.next().?);
}

test "subsetSize - basic" {
    try testing.expectEqual(0, subsetSize(0b00000000));
    try testing.expectEqual(1, subsetSize(0b00000001));
    try testing.expectEqual(4, subsetSize(0b10110100));
    try testing.expectEqual(8, subsetSize(0b11111111));
}

test "isInSubset - basic" {
    const mask: u64 = 0b10110100;
    try testing.expect(isInSubset(mask, 2));
    try testing.expect(!isInSubset(mask, 3));
    try testing.expect(isInSubset(mask, 4));
    try testing.expect(isInSubset(mask, 5));
    try testing.expect(!isInSubset(mask, 6));
    try testing.expect(isInSubset(mask, 7));
}

test "isInSubset - out of range" {
    try testing.expect(!isInSubset(0xFF, 64));
    try testing.expect(!isInSubset(0xFF, 100));
}

test "generateAllSubsets - basic" {
    const subsets = try generateAllSubsets(testing.allocator, 3);
    defer subsets.deinit();

    try testing.expectEqual(8, subsets.items.len); // 2^3
    try testing.expectEqual(0b000, subsets.items[0]);
    try testing.expectEqual(0b111, subsets.items[7]);
}

test "generateAllSubsets - too many elements" {
    try testing.expectError(error.TooManyElements, generateAllSubsets(testing.allocator, 21));
}

test "generateSubsetsOfSize - basic" {
    const subsets = try generateSubsetsOfSize(testing.allocator, 4, 2);
    defer subsets.deinit();

    try testing.expectEqual(6, subsets.items.len); // C(4, 2)

    for (subsets.items) |mask| {
        try testing.expectEqual(2, @popCount(mask));
    }
}

test "generateSubsetsOfSize - edge cases" {
    const empty = try generateSubsetsOfSize(testing.allocator, 5, 0);
    defer empty.deinit();
    try testing.expectEqual(0, empty.items.len);

    const full = try generateSubsetsOfSize(testing.allocator, 3, 3);
    defer full.deinit();
    try testing.expectEqual(1, full.items.len);
    try testing.expectEqual(0b111, full.items[0]);
}

test "generateSubsetsOfSize - invalid parameters" {
    try testing.expectError(error.InvalidParameters, generateSubsetsOfSize(testing.allocator, 3, 5));
}

test "SubmaskIterator - basic iteration" {
    const mask: u64 = 0b1011; // 3 set bits -> 8 submasks
    var iter = SubmaskIterator.init(mask);

    var masks = ArrayList(u64).init(testing.allocator);
    defer masks.deinit();

    while (iter.next()) |submask| {
        try masks.append(submask);
        // Each submask should be a subset of mask
        try testing.expect((submask & mask) == submask);
    }

    try testing.expectEqual(8, masks.items.len); // 2^3
    try testing.expect(std.mem.indexOfScalar(u64, masks.items, 0b1011) != null);
    try testing.expect(std.mem.indexOfScalar(u64, masks.items, 0b0000) != null);
}

test "SubmaskIterator - single bit" {
    var iter = SubmaskIterator.init(0b1000);

    try testing.expectEqual(0b1000, iter.next().?);
    try testing.expectEqual(0b0000, iter.next().?);
    try testing.expect(iter.next() == null);
}

test "SubmaskIterator - empty mask" {
    var iter = SubmaskIterator.init(0);
    try testing.expectEqual(0, iter.next().?);
    try testing.expect(iter.next() == null);
}

test "SubmaskIterator - reset" {
    var iter = SubmaskIterator.init(0b101);
    const first = iter.next().?;
    _ = iter.next();
    iter.reset();
    try testing.expectEqual(first, iter.next().?);
}

test "estimateBinomial - basic" {
    try testing.expectEqual(1, estimateBinomial(5, 0));
    try testing.expectEqual(5, estimateBinomial(5, 1));
    try testing.expectEqual(10, estimateBinomial(5, 2));
    try testing.expectEqual(10, estimateBinomial(5, 3));
    try testing.expectEqual(5, estimateBinomial(5, 4));
    try testing.expectEqual(1, estimateBinomial(5, 5));
}

test "estimateBinomial - invalid" {
    try testing.expectEqual(0, estimateBinomial(3, 5));
}
