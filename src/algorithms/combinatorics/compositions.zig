//! Integer Compositions
//!
//! A composition of a positive integer n is an ordered sequence of positive integers
//! that sum to n. Unlike partitions (which are unordered), compositions distinguish
//! between different orderings.
//!
//! Examples:
//! - Compositions of 3: [3], [2,1], [1,2], [1,1,1]
//! - Compositions of 4 into 2 parts: [3,1], [2,2], [1,3]
//!
//! Applications:
//! - Combinatorial optimization (resource allocation with ordering)
//! - Sequence analysis (breaking sequences into runs)
//! - Dynamic programming (subset ordering matters)
//! - Probability theory (ordered outcomes)
//! - Cryptography (key space enumeration)
//!
//! References:
//! - Knuth, D. (2005). "The Art of Computer Programming, Vol 4A" Section 7.2.1.4
//! - Stanley, R. (2011). "Enumerative Combinatorics, Vol 1" Chapter 1

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const testing = std.testing;

/// Count the number of compositions of n into exactly k parts.
///
/// Formula: C(n, k) = C(n-1, k-1) where C is binomial coefficient.
/// This follows from the "stars and bars" method: we need to place k-1 dividers
/// among n-1 gaps between n stars, which is equivalent to choosing k-1 positions
/// from n-1 available positions.
///
/// Time: O(min(k, n-k)) via binomial coefficient calculation
/// Space: O(1)
///
/// Example:
/// ```zig
/// const count = try countCompositions(u32, 5, 2); // Returns 4
/// // The 4 compositions are: [4,1], [3,2], [2,3], [1,4]
/// ```
pub fn countCompositions(comptime T: type, n: T, k: T) !T {
    if (@typeInfo(T) != .int) {
        @compileError("countCompositions requires an integer type");
    }

    // Validate inputs
    if (n < 0 or k < 0) {
        return error.NegativeInput;
    }
    if (k == 0) {
        return if (n == 0) 1 else 0;
    }
    if (n == 0) {
        return 0; // No compositions of 0 into k>0 parts
    }
    if (k > n) {
        return 0; // Cannot have more parts than the sum
    }
    if (k == n) {
        return 1; // Only [1,1,...,1]
    }

    // C(n-1, k-1) - binomial coefficient
    return binomial(T, n - 1, k - 1);
}

/// Count total number of compositions of n (into any number of parts).
///
/// Formula: Total compositions of n = 2^(n-1) for n ≥ 1
/// Proof: Each of the n-1 gaps between consecutive units can either have a
/// divider or not, giving 2^(n-1) possibilities.
///
/// Time: O(log n) for exponentiation
/// Space: O(1)
///
/// Example:
/// ```zig
/// const count = try countAllCompositions(u32, 4); // Returns 8
/// // The 8 compositions are: [4], [3,1], [2,2], [1,3], [2,1,1], [1,2,1], [1,1,2], [1,1,1,1]
/// ```
pub fn countAllCompositions(comptime T: type, n: T) !T {
    if (@typeInfo(T) != .int) {
        @compileError("countAllCompositions requires an integer type");
    }

    if (n < 0) {
        return error.NegativeInput;
    }
    if (n == 0) {
        return 1; // Empty composition
    }

    // 2^(n-1) using bit shift
    if (n > @bitSizeOf(T)) {
        return error.Overflow;
    }
    const shift_amount: std.math.Log2Int(T) = @intCast(n - 1);
    return @as(T, 1) << shift_amount;
}

/// Generate all compositions of n into exactly k parts.
///
/// Uses backtracking to generate compositions in lexicographic order.
/// Each composition is represented as a slice of k positive integers.
///
/// Time: O(C(n-1, k-1) * k) where C is binomial coefficient
/// Space: O(C(n-1, k-1) * k) for storing results
///
/// Example:
/// ```zig
/// var compositions = try generateKCompositions(u32, allocator, 5, 2);
/// defer {
///     for (compositions.items) |comp| allocator.free(comp);
///     compositions.deinit();
/// }
/// // Returns: [[4,1], [3,2], [2,3], [1,4]]
/// ```
pub fn generateKCompositions(comptime T: type, allocator: Allocator, n: T, k: T) !ArrayList([]T) {
    if (@typeInfo(T) != .int) {
        @compileError("generateKCompositions requires an integer type");
    }

    if (n < 0 or k < 0) {
        return error.NegativeInput;
    }

    var result = ArrayList([]T){};
    errdefer {
        for (result.items) |comp| allocator.free(comp);
        result.deinit(allocator);
    }

    if (k == 0) {
        if (n == 0) {
            // Empty composition
            const empty = try allocator.alloc(T, 0);
            try result.append(allocator, empty);
        }
        return result;
    }

    if (n == 0 or k > n) {
        return result; // No valid compositions
    }

    // Generate compositions via backtracking
    var current = ArrayList(T){};
    defer current.deinit(allocator);

    try generateKCompositionsHelper(T, allocator, n, k, &current, &result);
    return result;
}

fn generateKCompositionsHelper(
    comptime T: type,
    allocator: Allocator,
    remaining: T,
    parts_left: T,
    current: *ArrayList(T),
    result: *ArrayList([]T),
) !void {
    if (parts_left == 0) {
        if (remaining == 0) {
            // Found a valid composition
            const comp = try allocator.alloc(T, current.items.len);
            @memcpy(comp, current.items);
            try result.append(allocator, comp);
        }
        return;
    }

    // Try each possible value for the current part
    // Range: 1 to (remaining - parts_left + 1)
    // Must leave at least 1 for each remaining part
    const max_value = remaining - parts_left + 1;
    var value: T = 1;
    while (value <= max_value) : (value += 1) {
        try current.append(allocator, value);
        try generateKCompositionsHelper(T, allocator, remaining - value, parts_left - 1, current, result);
        _ = current.pop();
    }
}

/// Generate all compositions of n (into any number of parts).
///
/// Uses recursive generation based on binary representation.
/// Total number of compositions is 2^(n-1).
///
/// Time: O(2^(n-1) * n)
/// Space: O(2^(n-1) * n)
///
/// Example:
/// ```zig
/// var compositions = try generateCompositions(u32, allocator, 4);
/// defer {
///     for (compositions.items) |comp| allocator.free(comp);
///     compositions.deinit();
/// }
/// // Returns: [[4], [3,1], [2,2], [1,3], [2,1,1], [1,2,1], [1,1,2], [1,1,1,1]]
/// ```
pub fn generateCompositions(comptime T: type, allocator: Allocator, n: T) !ArrayList([]T) {
    if (@typeInfo(T) != .int) {
        @compileError("generateCompositions requires an integer type");
    }

    if (n < 0) {
        return error.NegativeInput;
    }

    var result = ArrayList([]T){};
    errdefer {
        for (result.items) |comp| allocator.free(comp);
        result.deinit(allocator);
    }

    if (n == 0) {
        // Empty composition
        const empty = try allocator.alloc(T, 0);
        try result.append(allocator, empty);
        return result;
    }

    // Generate via binary representation
    // Each of the n-1 gaps can have a divider (1) or not (0)
    const num_gaps: u32 = @intCast(n - 1);
    const total_comps: u32 = @intCast(try std.math.powi(u32, 2, num_gaps));

    var mask: u32 = 0;
    while (mask < total_comps) : (mask += 1) {
        var comp = ArrayList(T){};
        defer comp.deinit(allocator);

        var current_part: T = 1;
        var bit_idx: u5 = 0;
        while (bit_idx < num_gaps) : (bit_idx += 1) {
            if ((mask >> bit_idx) & 1 == 1) {
                // Divider present - end current part
                try comp.append(allocator, current_part);
                current_part = 1;
            } else {
                // No divider - extend current part
                current_part += 1;
            }
        }
        // Add the last part
        try comp.append(allocator, current_part);

        // Copy to result
        const comp_slice = try allocator.alloc(T, comp.items.len);
        @memcpy(comp_slice, comp.items);
        try result.append(allocator, comp_slice);
    }

    return result;
}

/// Compute lexicographic rank of a composition.
///
/// Given a composition, compute its position in lexicographically sorted order.
/// Uses the formula based on counting smaller compositions.
///
/// Time: O(k^2) where k is the number of parts
/// Space: O(1)
///
/// Example:
/// ```zig
/// const rank = try lexicographicRank(u32, &[_]u32{2, 1, 1}); // Composition of 4
/// // Returns the rank (0-indexed position in sorted list)
/// ```
pub fn lexicographicRank(comptime T: type, composition: []const T) !T {
    if (@typeInfo(T) != .int) {
        @compileError("lexicographicRank requires an integer type");
    }

    if (composition.len == 0) {
        return 0; // Empty composition has rank 0
    }

    // Compute sum
    var n: T = 0;
    for (composition) |part| {
        if (part <= 0) {
            return error.InvalidComposition;
        }
        n += part;
    }

    const k: T = @intCast(composition.len);
    var rank: T = 0;

    // Count compositions with fewer parts
    if (k > 1) {
        var parts: T = 1;
        while (parts < k) : (parts += 1) {
            const count = try countCompositions(T, n, parts);
            rank += count;
        }
    }

    // Count compositions with same number of parts but lexicographically smaller
    var prefix_sum: T = 0;
    for (composition, 0..) |part, i| {
        // Count compositions where position i has a smaller value
        var smaller_value: T = 1;
        while (smaller_value < part) : (smaller_value += 1) {
            const remaining_sum = n - prefix_sum - smaller_value;
            const remaining_parts: T = k - @as(T, @intCast(i)) - 1;

            if (remaining_parts == 0) {
                if (remaining_sum == 0) {
                    rank += 1;
                }
            } else if (remaining_sum >= remaining_parts) {
                const count = try countCompositions(T, remaining_sum, remaining_parts);
                rank += count;
            }
        }

        prefix_sum += part;
    }

    return rank;
}

/// Compute the composition with given lexicographic rank.
///
/// Given a rank and total sum n, compute the composition at that position.
/// Inverse of lexicographicRank.
///
/// Time: O(n^2)
/// Space: O(n) for allocating result
///
/// Example:
/// ```zig
/// const comp = try lexicographicUnrank(u32, allocator, 4, 3);
/// defer allocator.free(comp);
/// // Returns composition of 4 with rank 3
/// ```
pub fn lexicographicUnrank(comptime T: type, allocator: Allocator, n: T, rank: T) ![]T {
    if (@typeInfo(T) != .int) {
        @compileError("lexicographicUnrank requires an integer type");
    }

    if (n < 0 or rank < 0) {
        return error.NegativeInput;
    }

    if (n == 0) {
        return try allocator.alloc(T, 0);
    }

    var result = ArrayList(T){};
    errdefer result.deinit(allocator);

    var remaining_rank = rank;
    var remaining_sum = n;

    // Determine number of parts by counting compositions
    var k: T = 1;
    while (k <= n) : (k += 1) {
        const count = try countCompositions(T, n, k);
        if (remaining_rank < count) {
            break; // Found the right number of parts
        }
        remaining_rank -= count;
    }

    if (k > n) {
        return error.RankTooLarge;
    }

    // Generate composition with k parts and remaining_rank
    var parts_left: T = k;
    while (parts_left > 0) : (parts_left -= 1) {
        // Find the value for current part
        var value: T = 1;
        const max_value = remaining_sum - parts_left + 1;

        while (value <= max_value) : (value += 1) {
            const after_this_value = remaining_sum - value;
            const parts_after_this: T = parts_left - 1;

            const count = if (parts_after_this == 0)
                if (after_this_value == 0) @as(T, 1) else @as(T, 0)
            else
                try countCompositions(T, after_this_value, parts_after_this);

            if (remaining_rank < count) {
                try result.append(allocator, value);
                remaining_sum -= value;
                break;
            }

            remaining_rank -= count;
        }
    }

    return result.toOwnedSlice(allocator);
}

// Helper: Binomial coefficient
fn binomial(comptime T: type, n: T, k: T) !T {
    if (k > n) return 0;
    if (k == 0 or k == n) return 1;

    // Use symmetry: C(n, k) = C(n, n-k)
    const k_adj = if (k > n - k) n - k else k;

    var result: T = 1;
    var i: T = 0;
    while (i < k_adj) : (i += 1) {
        // result = result * (n - i) / (i + 1)
        result = try std.math.mul(T, result, n - i);
        result = @divTrunc(result, i + 1);
    }

    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "countCompositions: basic cases" {
    try testing.expectEqual(@as(u32, 0), try countCompositions(u32, 0, 0));
    try testing.expectEqual(@as(u32, 1), try countCompositions(u32, 1, 1));
    try testing.expectEqual(@as(u32, 1), try countCompositions(u32, 2, 2));
    try testing.expectEqual(@as(u32, 1), try countCompositions(u32, 5, 5));
}

test "countCompositions: two parts" {
    try testing.expectEqual(@as(u32, 1), try countCompositions(u32, 2, 2)); // [1,1]
    try testing.expectEqual(@as(u32, 2), try countCompositions(u32, 3, 2)); // [2,1], [1,2]
    try testing.expectEqual(@as(u32, 3), try countCompositions(u32, 4, 2)); // [3,1], [2,2], [1,3]
    try testing.expectEqual(@as(u32, 4), try countCompositions(u32, 5, 2)); // [4,1], [3,2], [2,3], [1,4]
}

test "countCompositions: specific values" {
    try testing.expectEqual(@as(u32, 3), try countCompositions(u32, 4, 3)); // C(3,2) = 3
    try testing.expectEqual(@as(u32, 6), try countCompositions(u32, 5, 3)); // C(4,2) = 6
    try testing.expectEqual(@as(u32, 10), try countCompositions(u32, 6, 3)); // C(5,2) = 10
    try testing.expectEqual(@as(u32, 10), try countCompositions(u32, 11, 2)); // C(10,1) = 10
}

test "countCompositions: edge cases" {
    try testing.expectEqual(@as(u32, 0), try countCompositions(u32, 3, 0)); // k=0, n>0
    try testing.expectEqual(@as(u32, 0), try countCompositions(u32, 0, 3)); // n=0, k>0
    try testing.expectEqual(@as(u32, 0), try countCompositions(u32, 3, 5)); // k > n
}

test "countCompositions: negative input error" {
    try testing.expectError(error.NegativeInput, countCompositions(i32, -1, 2));
    try testing.expectError(error.NegativeInput, countCompositions(i32, 5, -1));
}

test "countAllCompositions: basic cases" {
    try testing.expectEqual(@as(u32, 1), try countAllCompositions(u32, 0)); // Empty
    try testing.expectEqual(@as(u32, 1), try countAllCompositions(u32, 1)); // [1]
    try testing.expectEqual(@as(u32, 2), try countAllCompositions(u32, 2)); // [2], [1,1]
    try testing.expectEqual(@as(u32, 4), try countAllCompositions(u32, 3)); // 2^2 = 4
    try testing.expectEqual(@as(u32, 8), try countAllCompositions(u32, 4)); // 2^3 = 8
    try testing.expectEqual(@as(u32, 16), try countAllCompositions(u32, 5)); // 2^4 = 16
}

test "countAllCompositions: larger values" {
    try testing.expectEqual(@as(u32, 512), try countAllCompositions(u32, 10)); // 2^9
    try testing.expectEqual(@as(u32, 1024), try countAllCompositions(u32, 11)); // 2^10
}

test "countAllCompositions: negative input error" {
    try testing.expectError(error.NegativeInput, countAllCompositions(i32, -1));
}

test "generateKCompositions: empty and single element" {
    var comps = try generateKCompositions(u32, testing.allocator, 0, 0);
    defer {
        for (comps.items) |c| testing.allocator.free(c);
        comps.deinit(testing.allocator);
    }
    try testing.expectEqual(@as(usize, 1), comps.items.len);
    try testing.expectEqual(@as(usize, 0), comps.items[0].len);

    var comps2 = try generateKCompositions(u32, testing.allocator, 5, 1);
    defer {
        for (comps2.items) |c| testing.allocator.free(c);
        comps2.deinit(testing.allocator);
    }
    try testing.expectEqual(@as(usize, 1), comps2.items.len);
    try testing.expectEqual(@as(u32, 5), comps2.items[0][0]);
}

test "generateKCompositions: two parts" {
    var comps = try generateKCompositions(u32, testing.allocator, 5, 2);
    defer {
        for (comps.items) |c| testing.allocator.free(c);
        comps.deinit(testing.allocator);
    }

    try testing.expectEqual(@as(usize, 4), comps.items.len);
    // Should be: [4,1], [3,2], [2,3], [1,4]
    try testing.expectEqual(@as(u32, 4), comps.items[0][0]);
    try testing.expectEqual(@as(u32, 1), comps.items[0][1]);
    try testing.expectEqual(@as(u32, 1), comps.items[3][0]);
    try testing.expectEqual(@as(u32, 4), comps.items[3][1]);
}

test "generateKCompositions: three parts of 6" {
    var comps = try generateKCompositions(u32, testing.allocator, 6, 3);
    defer {
        for (comps.items) |c| testing.allocator.free(c);
        comps.deinit(testing.allocator);
    }

    const expected_count = try countCompositions(u32, 6, 3);
    try testing.expectEqual(@as(u32, 10), expected_count);
    try testing.expectEqual(@as(usize, 10), comps.items.len);

    // Verify all sums equal 6
    for (comps.items) |comp| {
        var sum: u32 = 0;
        for (comp) |part| sum += part;
        try testing.expectEqual(@as(u32, 6), sum);
        try testing.expectEqual(@as(usize, 3), comp.len);
    }
}

test "generateKCompositions: invalid inputs" {
    var comps = try generateKCompositions(u32, testing.allocator, 3, 5);
    defer {
        for (comps.items) |c| testing.allocator.free(c);
        comps.deinit(testing.allocator);
    }
    try testing.expectEqual(@as(usize, 0), comps.items.len); // k > n
}

test "generateKCompositions: negative input error" {
    try testing.expectError(error.NegativeInput, generateKCompositions(i32, testing.allocator, -1, 2));
}

test "generateCompositions: small values" {
    var comps1 = try generateCompositions(u32, testing.allocator, 1);
    defer {
        for (comps1.items) |c| testing.allocator.free(c);
        comps1.deinit(testing.allocator);
    }
    try testing.expectEqual(@as(usize, 1), comps1.items.len);

    var comps2 = try generateCompositions(u32, testing.allocator, 2);
    defer {
        for (comps2.items) |c| testing.allocator.free(c);
        comps2.deinit(testing.allocator);
    }
    try testing.expectEqual(@as(usize, 2), comps2.items.len); // [2], [1,1]

    var comps3 = try generateCompositions(u32, testing.allocator, 3);
    defer {
        for (comps3.items) |c| testing.allocator.free(c);
        comps3.deinit(testing.allocator);
    }
    try testing.expectEqual(@as(usize, 4), comps3.items.len); // 2^2
}

test "generateCompositions: n=4" {
    var comps = try generateCompositions(u32, testing.allocator, 4);
    defer {
        for (comps.items) |c| testing.allocator.free(c);
        comps.deinit(testing.allocator);
    }

    try testing.expectEqual(@as(usize, 8), comps.items.len); // 2^3

    // Verify all sums equal 4
    for (comps.items) |comp| {
        var sum: u32 = 0;
        for (comp) |part| {
            try testing.expect(part > 0); // All parts positive
            sum += part;
        }
        try testing.expectEqual(@as(u32, 4), sum);
    }
}

test "generateCompositions: empty composition" {
    var comps = try generateCompositions(u32, testing.allocator, 0);
    defer {
        for (comps.items) |c| testing.allocator.free(c);
        comps.deinit(testing.allocator);
    }
    try testing.expectEqual(@as(usize, 1), comps.items.len);
    try testing.expectEqual(@as(usize, 0), comps.items[0].len);
}

test "generateCompositions: negative input error" {
    try testing.expectError(error.NegativeInput, generateCompositions(i32, testing.allocator, -1));
}

test "lexicographicRank: simple compositions" {
    try testing.expectEqual(@as(u32, 0), try lexicographicRank(u32, &[_]u32{3})); // Smallest of n=3
    try testing.expectEqual(@as(u32, 1), try lexicographicRank(u32, &[_]u32{2, 1}));
    try testing.expectEqual(@as(u32, 2), try lexicographicRank(u32, &[_]u32{1, 2}));
    try testing.expectEqual(@as(u32, 3), try lexicographicRank(u32, &[_]u32{1, 1, 1})); // Largest of n=3
}

test "lexicographicRank: compositions of 4" {
    // All 8 compositions of 4 in lexicographic order
    try testing.expectEqual(@as(u32, 0), try lexicographicRank(u32, &[_]u32{4}));
    try testing.expectEqual(@as(u32, 1), try lexicographicRank(u32, &[_]u32{3, 1}));
    try testing.expectEqual(@as(u32, 2), try lexicographicRank(u32, &[_]u32{2, 2}));
    try testing.expectEqual(@as(u32, 3), try lexicographicRank(u32, &[_]u32{1, 3}));
    try testing.expectEqual(@as(u32, 4), try lexicographicRank(u32, &[_]u32{2, 1, 1}));
    try testing.expectEqual(@as(u32, 5), try lexicographicRank(u32, &[_]u32{1, 2, 1}));
    try testing.expectEqual(@as(u32, 6), try lexicographicRank(u32, &[_]u32{1, 1, 2}));
    try testing.expectEqual(@as(u32, 7), try lexicographicRank(u32, &[_]u32{1, 1, 1, 1}));
}

test "lexicographicRank: empty composition" {
    try testing.expectEqual(@as(u32, 0), try lexicographicRank(u32, &[_]u32{}));
}

test "lexicographicRank: invalid composition error" {
    try testing.expectError(error.InvalidComposition, lexicographicRank(u32, &[_]u32{ 0, 1 }));
    try testing.expectError(error.InvalidComposition, lexicographicRank(i32, &[_]i32{ -1, 2 }));
}

test "lexicographicUnrank: roundtrip with rank" {
    // Test roundtrip for n=4
    const n: u32 = 4;
    const total = try countAllCompositions(u32, n);

    var rank: u32 = 0;
    while (rank < total) : (rank += 1) {
        const comp = try lexicographicUnrank(u32, testing.allocator, n, rank);
        defer testing.allocator.free(comp);

        const computed_rank = try lexicographicRank(u32, comp);
        try testing.expectEqual(rank, computed_rank);

        // Verify sum equals n
        var sum: u32 = 0;
        for (comp) |part| sum += part;
        try testing.expectEqual(n, sum);
    }
}

test "lexicographicUnrank: specific ranks" {
    {
        const comp = try lexicographicUnrank(u32, testing.allocator, 3, 0);
        defer testing.allocator.free(comp);
        try testing.expectEqual(@as(usize, 1), comp.len);
        try testing.expectEqual(@as(u32, 3), comp[0]);
    }

    {
        const comp = try lexicographicUnrank(u32, testing.allocator, 4, 2);
        defer testing.allocator.free(comp);
        try testing.expectEqualSlices(u32, &[_]u32{ 2, 2 }, comp);
    }

    {
        const comp = try lexicographicUnrank(u32, testing.allocator, 4, 7);
        defer testing.allocator.free(comp);
        try testing.expectEqualSlices(u32, &[_]u32{ 1, 1, 1, 1 }, comp);
    }
}

test "lexicographicUnrank: empty composition" {
    const comp = try lexicographicUnrank(u32, testing.allocator, 0, 0);
    defer testing.allocator.free(comp);
    try testing.expectEqual(@as(usize, 0), comp.len);
}

test "lexicographicUnrank: negative input error" {
    try testing.expectError(error.NegativeInput, lexicographicUnrank(i32, testing.allocator, -1, 0));
    try testing.expectError(error.NegativeInput, lexicographicUnrank(i32, testing.allocator, 5, -1));
}

test "lexicographicUnrank: rank too large error" {
    try testing.expectError(error.RankTooLarge, lexicographicUnrank(u32, testing.allocator, 3, 100));
}

test "compositions: memory safety" {
    // Generate and free multiple times to verify no leaks
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var comps = try generateKCompositions(u32, testing.allocator, 5, 3);
        defer {
            for (comps.items) |c| testing.allocator.free(c);
            comps.deinit(testing.allocator);
        }
        try testing.expect(comps.items.len > 0);
    }

    var j: usize = 0;
    while (j < 10) : (j += 1) {
        var comps = try generateCompositions(u32, testing.allocator, 4);
        defer {
            for (comps.items) |c| testing.allocator.free(c);
            comps.deinit(testing.allocator);
        }
        try testing.expectEqual(@as(usize, 8), comps.items.len);
    }
}

test "compositions: type variants" {
    // Test with different integer types
    _ = try countCompositions(u8, 5, 2);
    _ = try countCompositions(u16, 10, 3);
    _ = try countCompositions(u64, 20, 5);
    _ = try countAllCompositions(u8, 7);
    _ = try countAllCompositions(u16, 12);
}
