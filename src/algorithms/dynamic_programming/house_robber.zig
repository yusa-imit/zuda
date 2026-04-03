const std = @import("std");
const Allocator = std.mem.Allocator;

/// House Robber — Dynamic Programming
///
/// Classic DP problem for planning optimal robbery strategy with non-adjacent constraint.
///
/// **Problem Variants**:
/// 1. House Robber I: Linear street (this file)
/// 2. House Robber II: Circular street (houses form a circle)
/// 3. House Robber III: Binary tree (houses as tree nodes)
///
/// **Algorithm**: Bottom-up DP with optimal substructure
/// - State: rob[i] = maximum money robbing houses 0..i
/// - Recurrence: rob[i] = max(rob[i-1], rob[i-2] + houses[i])
/// - Base cases: rob[0] = houses[0], rob[1] = max(houses[0], houses[1])
///
/// **Variants**:
/// - maxRob(): O(n) time, O(1) space — space-optimized two-variable solution
/// - maxRobTable(): O(n) time, O(n) space — full DP table for reconstruction
/// - maxRobCircular(): O(n) time, O(1) space — houses form a circle (can't rob first and last)
/// - maxRobStrategy(): O(n) time, O(n) space — returns max amount + which houses to rob
///
/// **Time Complexity**: O(n) for all variants
/// **Space Complexity**: O(1) optimized, O(n) for table/reconstruction
///
/// **Use Cases**:
/// - Resource allocation (can't use adjacent resources)
/// - Scheduling (maximize value with non-overlapping constraint)
/// - Game strategy (optimal decision sequence)
/// - Investment planning (mutually exclusive options)
///
/// **References**:
/// - LeetCode #198 (House Robber I — linear)
/// - LeetCode #213 (House Robber II — circular)
/// - LeetCode #337 (House Robber III — binary tree)

/// Maximum amount that can be robbed from houses in a row.
/// Cannot rob two adjacent houses (alarm will trigger).
///
/// **Algorithm**: Space-optimized DP with two variables (O(1) space)
/// - Track rob_prev2 (max up to i-2) and rob_prev1 (max up to i-1)
/// - For each house: rob_curr = max(rob_prev1, rob_prev2 + houses[i])
/// - Roll forward: rob_prev2 = rob_prev1, rob_prev1 = rob_curr
///
/// Time: O(n) | Space: O(1)
///
/// Example:
/// ```
/// houses = [2, 7, 9, 3, 1]
/// rob[0] = 2
/// rob[1] = max(2, 7) = 7
/// rob[2] = max(7, 2+9) = 11
/// rob[3] = max(11, 7+3) = 11
/// rob[4] = max(11, 11+1) = 12
/// → Answer: 12 (rob houses 0, 2, 4)
/// ```
pub fn maxRob(comptime T: type, houses: []const T) T {
    if (houses.len == 0) return 0;
    if (houses.len == 1) return houses[0];
    if (houses.len == 2) return @max(houses[0], houses[1]);

    var rob_prev2: T = houses[0]; // max up to i-2
    var rob_prev1: T = @max(houses[0], houses[1]); // max up to i-1

    for (houses[2..], 2..) |house, i| {
        _ = i; // unused, kept for clarity
        const rob_curr = @max(rob_prev1, rob_prev2 + house);
        rob_prev2 = rob_prev1;
        rob_prev1 = rob_curr;
    }

    return rob_prev1;
}

/// Maximum amount with full DP table (for reconstruction).
/// Returns the DP table where table[i] = max money from houses 0..i.
///
/// Time: O(n) | Space: O(n)
///
/// Example:
/// ```
/// houses = [1, 2, 3, 1]
/// table = [1, 2, 4, 4]
/// → Answer: 4 (rob houses 0, 2)
/// ```
pub fn maxRobTable(comptime T: type, allocator: Allocator, houses: []const T) ![]T {
    if (houses.len == 0) {
        const table = try allocator.alloc(T, 0);
        return table;
    }

    const n = houses.len;
    const table = try allocator.alloc(T, n);
    errdefer allocator.free(table);

    table[0] = houses[0];
    if (n == 1) return table;

    table[1] = @max(houses[0], houses[1]);

    for (houses[2..], 2..) |house, i| {
        table[i] = @max(table[i - 1], table[i - 2] + house);
    }

    return table;
}

/// Maximum amount for circular street (houses form a circle).
/// Cannot rob house 0 and house n-1 simultaneously.
///
/// **Algorithm**: Reduce to two linear subproblems
/// - Case 1: Rob houses 0..n-2 (exclude last house)
/// - Case 2: Rob houses 1..n-1 (exclude first house)
/// - Answer: max(case1, case2)
///
/// Time: O(n) | Space: O(1)
///
/// Example:
/// ```
/// houses = [2, 3, 2] (circular)
/// case1 = maxRob([2, 3]) = 3
/// case2 = maxRob([3, 2]) = 3
/// → Answer: 3 (rob house 1, cannot rob both 0 and 2)
/// ```
pub fn maxRobCircular(comptime T: type, houses: []const T) T {
    if (houses.len == 0) return 0;
    if (houses.len == 1) return houses[0];
    if (houses.len == 2) return @max(houses[0], houses[1]);

    // Case 1: Rob houses 0..n-2 (exclude last)
    const case1 = maxRob(T, houses[0 .. houses.len - 1]);

    // Case 2: Rob houses 1..n-1 (exclude first)
    const case2 = maxRob(T, houses[1..]);

    return @max(case1, case2);
}

/// Result of maxRobStrategy: max amount + which houses to rob
pub fn RobStrategy(comptime T: type) type {
    return struct {
        max_amount: T,
        houses_robbed: []usize, // indices of houses to rob

        pub fn deinit(self: @This(), allocator: Allocator) void {
            allocator.free(self.houses_robbed);
        }
    };
}

/// Maximum amount with full strategy (which houses to rob).
/// Uses backtracking to reconstruct the optimal solution.
///
/// Time: O(n) | Space: O(n)
///
/// Example:
/// ```
/// houses = [2, 7, 9, 3, 1]
/// → max_amount: 12, houses_robbed: [0, 2, 4]
/// ```
pub fn maxRobStrategy(comptime T: type, allocator: Allocator, houses: []const T) !RobStrategy(T) {
    if (houses.len == 0) {
        const robbed = try allocator.alloc(usize, 0);
        return RobStrategy(T){ .max_amount = 0, .houses_robbed = robbed };
    }

    const n = houses.len;
    const table = try maxRobTable(T, allocator, houses);
    defer allocator.free(table);

    // Backtrack to find which houses were robbed
    var robbed_list = try std.ArrayList(usize).initCapacity(allocator, 0);
    errdefer robbed_list.deinit(allocator);

    var i: usize = n;
    while (i > 0) {
        i -= 1;

        if (i == 0) {
            // Always rob house 0 if we reached it
            try robbed_list.append(allocator, 0);
            break;
        } else if (i == 1) {
            // Rob house 0 or 1 (whichever is in the solution)
            if (table[1] == houses[0]) {
                try robbed_list.append(allocator, 0);
            } else {
                try robbed_list.append(allocator, 1);
            }
            break;
        } else {
            // Check if house i was robbed
            if (table[i] != table[i - 1]) {
                // House i was robbed (table[i] = table[i-2] + houses[i])
                try robbed_list.append(allocator, i);
                i -= 1; // Skip i-1 (can't rob adjacent)
            }
            // Otherwise house i was not robbed, continue to i-1
        }
    }

    // Reverse to get houses in ascending order
    std.mem.reverse(usize, robbed_list.items);

    return RobStrategy(T){
        .max_amount = table[n - 1],
        .houses_robbed = try robbed_list.toOwnedSlice(allocator),
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const expect = testing.expect;
const expectEqual = testing.expectEqual;

test "house robber: basic linear street" {
    // Example from LeetCode #198
    const houses = [_]u32{ 1, 2, 3, 1 };
    const result = maxRob(u32, &houses);
    try expectEqual(@as(u32, 4), result); // Rob houses 0 and 2
}

test "house robber: larger street" {
    const houses = [_]u32{ 2, 7, 9, 3, 1 };
    const result = maxRob(u32, &houses);
    try expectEqual(@as(u32, 12), result); // Rob houses 0, 2, 4
}

test "house robber: all same values" {
    const houses = [_]u32{ 5, 5, 5, 5, 5 };
    const result = maxRob(u32, &houses);
    try expectEqual(@as(u32, 15), result); // Rob houses 0, 2, 4
}

test "house robber: increasing values" {
    const houses = [_]u32{ 1, 2, 3, 4, 5 };
    const result = maxRob(u32, &houses);
    try expectEqual(@as(u32, 9), result); // Rob houses 0, 2, 4 → 1+3+5=9
}

test "house robber: edge case empty" {
    const houses = [_]u32{};
    const result = maxRob(u32, &houses);
    try expectEqual(@as(u32, 0), result);
}

test "house robber: edge case one house" {
    const houses = [_]u32{10};
    const result = maxRob(u32, &houses);
    try expectEqual(@as(u32, 10), result);
}

test "house robber: edge case two houses" {
    const houses1 = [_]u32{ 1, 2 };
    try expectEqual(@as(u32, 2), maxRob(u32, &houses1)); // Rob house 1

    const houses2 = [_]u32{ 5, 1 };
    try expectEqual(@as(u32, 5), maxRob(u32, &houses2)); // Rob house 0
}

test "house robber: DP table" {
    const allocator = testing.allocator;
    const houses = [_]u32{ 2, 7, 9, 3, 1 };

    const table = try maxRobTable(u32, allocator, &houses);
    defer allocator.free(table);

    try expectEqual(@as(usize, 5), table.len);
    try expectEqual(@as(u32, 2), table[0]);
    try expectEqual(@as(u32, 7), table[1]);
    try expectEqual(@as(u32, 11), table[2]); // max(7, 2+9)
    try expectEqual(@as(u32, 11), table[3]); // max(11, 7+3)
    try expectEqual(@as(u32, 12), table[4]); // max(11, 11+1)
}

test "house robber: DP table empty" {
    const allocator = testing.allocator;
    const houses = [_]u32{};

    const table = try maxRobTable(u32, allocator, &houses);
    defer allocator.free(table);

    try expectEqual(@as(usize, 0), table.len);
}

test "house robber: DP table single" {
    const allocator = testing.allocator;
    const houses = [_]u32{42};

    const table = try maxRobTable(u32, allocator, &houses);
    defer allocator.free(table);

    try expectEqual(@as(usize, 1), table.len);
    try expectEqual(@as(u32, 42), table[0]);
}

test "house robber: circular street basic" {
    // Cannot rob houses 0 and 2 (they are adjacent in circular layout)
    const houses = [_]u32{ 2, 3, 2 };
    const result = maxRobCircular(u32, &houses);
    try expectEqual(@as(u32, 3), result); // Rob house 1 only
}

test "house robber: circular street larger" {
    const houses = [_]u32{ 1, 2, 3, 1 };
    const result = maxRobCircular(u32, &houses);
    try expectEqual(@as(u32, 4), result); // Rob houses 1 and 3 (not 0 and 2)
}

test "house robber: circular street all same" {
    const houses = [_]u32{ 5, 5, 5, 5 };
    const result = maxRobCircular(u32, &houses);
    try expectEqual(@as(u32, 10), result); // Rob houses 1 and 3 (or 0 and 2)
}

test "house robber: circular edge cases" {
    const houses1 = [_]u32{10};
    try expectEqual(@as(u32, 10), maxRobCircular(u32, &houses1));

    const houses2 = [_]u32{ 5, 1 };
    try expectEqual(@as(u32, 5), maxRobCircular(u32, &houses2));

    const houses3 = [_]u32{};
    try expectEqual(@as(u32, 0), maxRobCircular(u32, &houses3));
}

test "house robber: strategy basic" {
    const allocator = testing.allocator;
    const houses = [_]u32{ 2, 7, 9, 3, 1 };

    const strategy = try maxRobStrategy(u32, allocator, &houses);
    defer strategy.deinit(allocator);

    try expectEqual(@as(u32, 12), strategy.max_amount);
    try expectEqual(@as(usize, 3), strategy.houses_robbed.len);
    try expectEqual(@as(usize, 0), strategy.houses_robbed[0]);
    try expectEqual(@as(usize, 2), strategy.houses_robbed[1]);
    try expectEqual(@as(usize, 4), strategy.houses_robbed[2]);
}

test "house robber: strategy simple" {
    const allocator = testing.allocator;
    const houses = [_]u32{ 1, 2, 3, 1 };

    const strategy = try maxRobStrategy(u32, allocator, &houses);
    defer strategy.deinit(allocator);

    try expectEqual(@as(u32, 4), strategy.max_amount);
    try expectEqual(@as(usize, 2), strategy.houses_robbed.len);
    try expectEqual(@as(usize, 0), strategy.houses_robbed[0]);
    try expectEqual(@as(usize, 2), strategy.houses_robbed[1]);
}

test "house robber: strategy edge cases" {
    const allocator = testing.allocator;

    // Empty
    {
        const houses = [_]u32{};
        const strategy = try maxRobStrategy(u32, allocator, &houses);
        defer strategy.deinit(allocator);
        try expectEqual(@as(u32, 0), strategy.max_amount);
        try expectEqual(@as(usize, 0), strategy.houses_robbed.len);
    }

    // Single
    {
        const houses = [_]u32{42};
        const strategy = try maxRobStrategy(u32, allocator, &houses);
        defer strategy.deinit(allocator);
        try expectEqual(@as(u32, 42), strategy.max_amount);
        try expectEqual(@as(usize, 1), strategy.houses_robbed.len);
        try expectEqual(@as(usize, 0), strategy.houses_robbed[0]);
    }

    // Two houses
    {
        const houses = [_]u32{ 5, 1 };
        const strategy = try maxRobStrategy(u32, allocator, &houses);
        defer strategy.deinit(allocator);
        try expectEqual(@as(u32, 5), strategy.max_amount);
        try expectEqual(@as(usize, 1), strategy.houses_robbed.len);
        try expectEqual(@as(usize, 0), strategy.houses_robbed[0]);
    }
}

test "house robber: f64 type" {
    const houses = [_]f64{ 1.5, 2.0, 3.5, 1.0 };
    const result = maxRob(f64, &houses);
    try expectEqual(@as(f64, 5.0), result); // Rob houses 0 and 2 → 1.5+3.5=5.0
}

test "house robber: large street" {
    const allocator = testing.allocator;
    const houses = try allocator.alloc(u32, 100);
    defer allocator.free(houses);

    // Alternating high and low values
    for (houses, 0..) |*h, i| {
        h.* = if (i % 2 == 0) 10 else 1;
    }

    const result = maxRob(u32, houses);
    try expectEqual(@as(u32, 500), result); // 50 houses with value 10
}

test "house robber: verify non-adjacent constraint" {
    const allocator = testing.allocator;
    const houses = [_]u32{ 5, 1, 3, 1 };

    const strategy = try maxRobStrategy(u32, allocator, &houses);
    defer strategy.deinit(allocator);

    // Verify no adjacent houses in strategy
    for (strategy.houses_robbed[0 .. strategy.houses_robbed.len - 1], 0..) |house, i| {
        const next_house = strategy.houses_robbed[i + 1];
        try expect(next_house >= house + 2); // At least 2 apart
    }
}

test "house robber: memory safety" {
    const allocator = testing.allocator;

    // Test all allocating functions
    {
        const houses = [_]u32{ 1, 2, 3, 4, 5 };
        const table = try maxRobTable(u32, allocator, &houses);
        allocator.free(table);
    }

    {
        const houses = [_]u32{ 2, 7, 9, 3, 1 };
        const strategy = try maxRobStrategy(u32, allocator, &houses);
        strategy.deinit(allocator);
    }

    // testing.allocator will detect leaks
}
