//! Rod Cutting Problem — Dynamic Programming
//!
//! Given a rod of length n and prices for different lengths,
//! find the maximum revenue obtainable by cutting the rod into pieces.
//!
//! **Problem**: Maximize revenue by choosing optimal cut positions.
//! **Approach**: Bottom-up DP building solutions for lengths 1..n.
//! **Recurrence**: revenue[i] = max(price[j] + revenue[i-j-1]) for all j < i
//!
//! **Time Complexity**: O(n²) — for each length, try all cut positions
//! **Space Complexity**: O(n) — DP table for revenues
//!
//! **Use Cases**:
//! - Manufacturing optimization (cutting raw materials)
//! - Resource allocation (dividing tasks/resources)
//! - Pricing strategy (bundling/unbundling products)
//! - Network bandwidth allocation
//!
//! **Example**:
//! ```zig
//! const prices = [_]i32{ 1, 5, 8, 9, 10, 17, 17, 20 }; // prices[i] = price for length i+1
//! var result = try RodCutting(i32).optimize(allocator, &prices, 8);
//! defer result.deinit();
//! // result.max_revenue = 22 (cut into 2+6 → 5+17=22)
//! // result.cuts contains optimal cut positions
//! ```

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// Rod Cutting optimizer with cut reconstruction
pub fn RodCutting(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Result containing maximum revenue and optimal cut positions
        pub const Result = struct {
            max_revenue: T,
            cuts: []usize, // Cut positions: first cut at cuts[0], second at cuts[1], etc.
            allocator: Allocator,

            pub fn deinit(self: *Result) void {
                self.allocator.free(self.cuts);
            }

            /// Get human-readable cut description
            /// Returns: ArrayList of piece lengths (e.g., [2, 6] for cuts at positions 2 and 6)
            pub fn getPieces(self: *const Result, allocator: Allocator, rod_length: usize) !std.ArrayList(usize) {
                var pieces = try std.ArrayList(usize).initCapacity(allocator, 8);
                errdefer pieces.deinit();

                if (self.cuts.len == 0) {
                    // No cuts — sell whole rod
                    pieces.appendAssumeCapacity(rod_length);
                    return pieces;
                }

                var positions = try std.ArrayList(usize).initCapacity(allocator, self.cuts.len);
                defer positions.deinit();

                // Collect cut positions
                for (self.cuts) |cut| {
                    try positions.append(cut);
                }

                // Sort positions
                std.mem.sort(usize, positions.items, {}, comptime std.sort.asc(usize));

                // Build piece lengths
                var prev: usize = 0;
                for (positions.items) |pos| {
                    if (pos > prev) {
                        try pieces.append(pos - prev);
                        prev = pos;
                    }
                }
                if (prev < rod_length) {
                    try pieces.append(rod_length - prev);
                }

                return pieces;
            }
        };

        /// Solve rod cutting problem (bottom-up DP)
        ///
        /// **Parameters**:
        /// - `allocator`: Memory allocator
        /// - `prices`: Array where prices[i] = price for rod of length i+1
        /// - `rod_length`: Target rod length (must be ≤ prices.len)
        ///
        /// **Returns**: Result with max revenue and optimal cuts
        ///
        /// **Time**: O(n²) where n = rod_length
        /// **Space**: O(n) for DP table and cut tracking
        ///
        /// **Errors**:
        /// - `error.InvalidLength` if rod_length > prices.len or rod_length == 0
        pub fn optimize(allocator: Allocator, prices: []const T, rod_length: usize) !Result {
            if (rod_length == 0 or rod_length > prices.len) return error.InvalidLength;

            // DP table: revenue[i] = max revenue for rod of length i
            var revenue = try allocator.alloc(T, rod_length + 1);
            defer allocator.free(revenue);
            @memset(revenue, 0);

            // Cut tracker: first_cut[i] = optimal first cut position for length i (1-indexed)
            var first_cut = try allocator.alloc(usize, rod_length + 1);
            defer allocator.free(first_cut);
            @memset(first_cut, 0);

            // Bottom-up DP: build solutions for lengths 1..rod_length
            for (1..rod_length + 1) |i| {
                var max_val: T = if (@typeInfo(T) == .int) std.math.minInt(T) else -std.math.inf(T);

                // Try all possible first cuts (1-indexed: cut at position j means piece of length j)
                for (1..i + 1) |j| {
                    const candidate = prices[j - 1] + revenue[i - j];
                    if (candidate > max_val) {
                        max_val = candidate;
                        first_cut[i] = j; // Remember optimal first cut
                    }
                }

                revenue[i] = max_val;
            }

            // Reconstruct cuts
            var cuts = try std.ArrayList(usize).initCapacity(allocator, 8);
            errdefer cuts.deinit();

            var remaining = rod_length;
            var position: usize = 0;
            while (remaining > 0) {
                const cut = first_cut[remaining];
                if (cut == 0) break;

                position += cut;
                if (position < rod_length) {
                    try cuts.append(position);
                }

                remaining -= cut;
            }

            return Result{
                .max_revenue = revenue[rod_length],
                .cuts = try cuts.toOwnedSlice(),
                .allocator = allocator,
            };
        }

        /// Solve for maximum revenue only (no cut reconstruction)
        ///
        /// **Time**: O(n²)
        /// **Space**: O(n)
        pub fn optimizeRevenue(allocator: Allocator, prices: []const T, rod_length: usize) !T {
            if (rod_length == 0 or rod_length > prices.len) return error.InvalidLength;

            var revenue = try allocator.alloc(T, rod_length + 1);
            defer allocator.free(revenue);
            @memset(revenue, 0);

            for (1..rod_length + 1) |i| {
                var max_val: T = if (@typeInfo(T) == .int) std.math.minInt(T) else -std.math.inf(T);
                for (1..i + 1) |j| {
                    const candidate = prices[j - 1] + revenue[i - j];
                    max_val = @max(max_val, candidate);
                }
                revenue[i] = max_val;
            }

            return revenue[rod_length];
        }

        /// Top-down memoized recursive approach
        ///
        /// **Time**: O(n²)
        /// **Space**: O(n) stack + O(n) memo
        pub fn optimizeRecursive(allocator: Allocator, prices: []const T, rod_length: usize) !T {
            if (rod_length == 0 or rod_length > prices.len) return error.InvalidLength;

            var memo = try allocator.alloc(?T, rod_length + 1);
            defer allocator.free(memo);
            @memset(memo, null);
            memo[0] = 0;

            return cutRodMemo(prices, rod_length, memo);
        }

        fn cutRodMemo(prices: []const T, n: usize, memo: []?T) T {
            if (memo[n]) |val| return val;

            var max_val: T = if (@typeInfo(T) == .int) std.math.minInt(T) else -std.math.inf(T);
            for (1..n + 1) |i| {
                max_val = @max(max_val, prices[i - 1] + cutRodMemo(prices, n - i, memo));
            }

            memo[n] = max_val;
            return max_val;
        }
    };
}

// =============== Tests ===============

test "RodCutting: basic 8-length rod" {
    const allocator = testing.allocator;
    const prices = [_]i32{ 1, 5, 8, 9, 10, 17, 17, 20 };

    var result = try RodCutting(i32).optimize(allocator, &prices, 8);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 22), result.max_revenue);

    // Verify pieces (should cut into optimal segments)
    var pieces = try result.getPieces(allocator, 8);
    defer pieces.deinit();

    var total_revenue: i32 = 0;
    for (pieces.items) |length| {
        total_revenue += prices[length - 1];
    }
    try testing.expectEqual(@as(i32, 22), total_revenue);
}

test "RodCutting: no cuts optimal" {
    const allocator = testing.allocator;
    const prices = [_]i32{ 1, 2, 3, 4, 100 }; // Length 5 has best price

    var result = try RodCutting(i32).optimize(allocator, &prices, 5);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 100), result.max_revenue);
    try testing.expectEqual(@as(usize, 0), result.cuts.len);

    var pieces = try result.getPieces(allocator, 5);
    defer pieces.deinit();
    try testing.expectEqual(@as(usize, 1), pieces.items.len);
    try testing.expectEqual(@as(usize, 5), pieces.items[0]);
}

test "RodCutting: length 1" {
    const allocator = testing.allocator;
    const prices = [_]i32{3};

    var result = try RodCutting(i32).optimize(allocator, &prices, 1);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 3), result.max_revenue);
    try testing.expectEqual(@as(usize, 0), result.cuts.len);
}

test "RodCutting: all unit cuts optimal" {
    const allocator = testing.allocator;
    const prices = [_]i32{ 10, 1, 1, 1 }; // Cutting into 1-length pieces is best

    var result = try RodCutting(i32).optimize(allocator, &prices, 4);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 40), result.max_revenue);

    var pieces = try result.getPieces(allocator, 4);
    defer pieces.deinit();
    try testing.expectEqual(@as(usize, 4), pieces.items.len);
}

test "RodCutting: optimizeRevenue (no cuts)" {
    const allocator = testing.allocator;
    const prices = [_]i32{ 1, 5, 8, 9, 10, 17, 17, 20 };

    const revenue = try RodCutting(i32).optimizeRevenue(allocator, &prices, 8);
    try testing.expectEqual(@as(i32, 22), revenue);

    const revenue_small = try RodCutting(i32).optimizeRevenue(allocator, &prices, 4);
    try testing.expectEqual(@as(i32, 10), revenue_small);
}

test "RodCutting: memoized recursive" {
    const allocator = testing.allocator;
    const prices = [_]i32{ 1, 5, 8, 9, 10, 17, 17, 20 };

    const revenue = try RodCutting(i32).optimizeRecursive(allocator, &prices, 8);
    try testing.expectEqual(@as(i32, 22), revenue);
}

test "RodCutting: length 10" {
    const allocator = testing.allocator;
    const prices = [_]i32{ 1, 5, 8, 9, 10, 17, 17, 20, 24, 30 };

    var result = try RodCutting(i32).optimize(allocator, &prices, 10);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 30), result.max_revenue);
}

test "RodCutting: f64 prices" {
    const allocator = testing.allocator;
    const prices = [_]f64{ 1.5, 5.2, 8.1, 9.9 };

    var result = try RodCutting(f64).optimize(allocator, &prices, 4);
    defer result.deinit();

    try testing.expect(result.max_revenue >= 10.0);
}

test "RodCutting: zero length error" {
    const allocator = testing.allocator;
    const prices = [_]i32{ 1, 2, 3 };

    const result = RodCutting(i32).optimize(allocator, &prices, 0);
    try testing.expectError(error.InvalidLength, result);
}

test "RodCutting: length exceeds prices error" {
    const allocator = testing.allocator;
    const prices = [_]i32{ 1, 2, 3 };

    const result = RodCutting(i32).optimize(allocator, &prices, 5);
    try testing.expectError(error.InvalidLength, result);
}

test "RodCutting: large rod (stress)" {
    const allocator = testing.allocator;

    var prices_list = try std.ArrayList(i32).initCapacity(allocator, 100);
    defer prices_list.deinit();

    for (0..100) |i| {
        prices_list.appendAssumeCapacity(@intCast(i + 1));
    }

    const revenue = try RodCutting(i32).optimizeRevenue(allocator, prices_list.items, 50);
    try testing.expect(revenue > 0);
}

test "RodCutting: negative prices" {
    const allocator = testing.allocator;
    const prices = [_]i32{ -5, 10, -3, 15 };

    var result = try RodCutting(i32).optimize(allocator, &prices, 4);
    defer result.deinit();

    // Should choose to cut at positions that maximize (avoid negative prices)
    try testing.expect(result.max_revenue > 0);
}

test "RodCutting: getPieces correctness" {
    const allocator = testing.allocator;
    const prices = [_]i32{ 1, 5, 8, 9, 10, 17, 17, 20 };

    var result = try RodCutting(i32).optimize(allocator, &prices, 8);
    defer result.deinit();

    var pieces = try result.getPieces(allocator, 8);
    defer pieces.deinit();

    // Verify total length
    var total_length: usize = 0;
    for (pieces.items) |length| {
        total_length += length;
    }
    try testing.expectEqual(@as(usize, 8), total_length);
}

test "RodCutting: memory safety" {
    const allocator = testing.allocator;
    const prices = [_]i32{ 1, 5, 8, 9, 10 };

    var result = try RodCutting(i32).optimize(allocator, &prices, 5);
    result.deinit();

    // No leaks expected
}
