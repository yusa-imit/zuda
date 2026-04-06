const std = @import("std");
const Allocator = std.mem.Allocator;

/// Trapping Rain Water - Dynamic Programming & Two-Pointer Solutions
///
/// This module implements various solutions to compute how much water can be trapped
/// after raining, given an elevation map represented by an array of non-negative integers.
///
/// Problem: Given n non-negative integers representing an elevation map where the width
/// of each bar is 1, compute how much water it can trap after raining.
///
/// Example: [0,1,0,2,1,0,1,3,2,1,2,1]
/// Water trapped: 6 units (2 + 1 + 1 + 2)
///
/// Variants:
/// - trap(): DP with left/right max arrays - O(n) time, O(n) space
/// - trapOptimized(): Two-pointer approach - O(n) time, O(1) space
/// - trapWithDetails(): Returns total water + water at each position
/// - trapStack(): Monotonic stack approach - O(n) time, O(n) space

/// Trapping Rain Water: Dynamic Programming Approach
///
/// Compute water trapped using precomputed left and right maximum heights.
/// At each position, water trapped = min(leftMax, rightMax) - height (if positive).
///
/// Algorithm:
/// 1. Compute leftMax[i] = max height to the left of i (including i)
/// 2. Compute rightMax[i] = max height to the right of i (including i)
/// 3. Water at i = max(0, min(leftMax[i], rightMax[i]) - height[i])
///
/// Time: O(n) where n = length of elevation map
/// Space: O(n) for leftMax and rightMax arrays
///
/// Examples:
/// ```zig
/// const heights = [_]u32{0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1};
/// const water = try trap(u32, allocator, &heights); // Returns 6
/// ```
pub fn trap(comptime T: type, allocator: Allocator, heights: []const T) !T {
    if (heights.len <= 2) return 0;

    const n = heights.len;

    // Allocate left and right max arrays
    const leftMax = try allocator.alloc(T, n);
    defer allocator.free(leftMax);
    const rightMax = try allocator.alloc(T, n);
    defer allocator.free(rightMax);

    // Compute left max
    leftMax[0] = heights[0];
    for (1..n) |i| {
        leftMax[i] = @max(leftMax[i - 1], heights[i]);
    }

    // Compute right max
    rightMax[n - 1] = heights[n - 1];
    var i: usize = n - 1;
    while (i > 0) {
        i -= 1;
        rightMax[i] = @max(rightMax[i + 1], heights[i]);
    }

    // Compute trapped water
    var water: T = 0;
    for (0..n) |j| {
        const minHeight = @min(leftMax[j], rightMax[j]);
        if (minHeight > heights[j]) {
            water += minHeight - heights[j];
        }
    }

    return water;
}

/// Trapping Rain Water: Two-Pointer Optimized Approach
///
/// Space-optimized solution using two pointers and tracking left/right max on-the-fly.
/// No auxiliary arrays needed.
///
/// Algorithm:
/// 1. Use two pointers: left (0) and right (n-1)
/// 2. Track leftMax and rightMax as we move pointers
/// 3. Move pointer with smaller max height (water level determined by smaller side)
/// 4. If current height < max, add water; otherwise update max
///
/// Invariant: Water at position i depends only on min(leftMax[i], rightMax[i])
///
/// Time: O(n) single pass with two pointers
/// Space: O(1) only two variables
///
/// Examples:
/// ```zig
/// const heights = [_]u32{4, 2, 0, 3, 2, 5};
/// const water = trapOptimized(u32, &heights); // Returns 9
/// ```
pub fn trapOptimized(comptime T: type, heights: []const T) T {
    if (heights.len <= 2) return 0;

    var left: usize = 0;
    var right: usize = heights.len - 1;
    var leftMax: T = 0;
    var rightMax: T = 0;
    var water: T = 0;

    while (left < right) {
        if (heights[left] < heights[right]) {
            // Water level determined by left side
            if (heights[left] >= leftMax) {
                leftMax = heights[left];
            } else {
                water += leftMax - heights[left];
            }
            left += 1;
        } else {
            // Water level determined by right side
            if (heights[right] >= rightMax) {
                rightMax = heights[right];
            } else {
                water += rightMax - heights[right];
            }
            right -= 1;
        }
    }

    return water;
}

/// Water Trapped Details
///
/// Result type for trapWithDetails function, containing total water
/// and water trapped at each position.
pub fn WaterDetails(comptime T: type) type {
    return struct {
        total: T,
        water_at: []T, // Caller must free

        pub fn deinit(self: *@This(), allocator: Allocator) void {
            allocator.free(self.water_at);
        }
    };
}

/// Trapping Rain Water: With Position Details
///
/// Returns both total water and water trapped at each position.
/// Useful for visualization and debugging.
///
/// Time: O(n)
/// Space: O(n) for leftMax, rightMax, and water_at arrays
///
/// Examples:
/// ```zig
/// const heights = [_]u32{3, 0, 2, 0, 4};
/// var details = try trapWithDetails(u32, allocator, &heights);
/// defer details.deinit(allocator);
/// // details.total = 7, details.water_at = [0, 3, 1, 3, 0]
/// ```
pub fn trapWithDetails(comptime T: type, allocator: Allocator, heights: []const T) !WaterDetails(T) {
    const n = heights.len;
    const water_at = try allocator.alloc(T, n);
    errdefer allocator.free(water_at);

    if (n <= 2) {
        @memset(water_at, 0);
        return .{ .total = 0, .water_at = water_at };
    }

    // Allocate left and right max arrays
    const leftMax = try allocator.alloc(T, n);
    defer allocator.free(leftMax);
    const rightMax = try allocator.alloc(T, n);
    defer allocator.free(rightMax);

    // Compute left max
    leftMax[0] = heights[0];
    for (1..n) |i| {
        leftMax[i] = @max(leftMax[i - 1], heights[i]);
    }

    // Compute right max
    rightMax[n - 1] = heights[n - 1];
    var i: usize = n - 1;
    while (i > 0) {
        i -= 1;
        rightMax[i] = @max(rightMax[i + 1], heights[i]);
    }

    // Compute trapped water at each position
    var total: T = 0;
    for (0..n) |j| {
        const minHeight = @min(leftMax[j], rightMax[j]);
        if (minHeight > heights[j]) {
            water_at[j] = minHeight - heights[j];
            total += water_at[j];
        } else {
            water_at[j] = 0;
        }
    }

    return .{ .total = total, .water_at = water_at };
}

/// Trapping Rain Water: Monotonic Stack Approach
///
/// Uses a stack to track potential water-trapping bars.
/// When we find a bar taller than stack top, we can trap water.
///
/// Algorithm:
/// 1. Maintain a decreasing monotonic stack of indices
/// 2. When current height > stack top, pop and compute water trapped
/// 3. Water width = current_index - left_boundary - 1
/// 4. Water height = min(heights[left], heights[current]) - heights[bottom]
///
/// Time: O(n) each element pushed/popped once
/// Space: O(n) for stack in worst case (decreasing sequence)
///
/// Examples:
/// ```zig
/// const heights = [_]u32{0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1};
/// const water = try trapStack(u32, allocator, &heights); // Returns 6
/// ```
pub fn trapStack(comptime T: type, allocator: Allocator, heights: []const T) !T {
    if (heights.len <= 2) return 0;

    var stack = try std.ArrayList(usize).initCapacity(allocator, heights.len);
    defer stack.deinit(allocator);

    var water: T = 0;
    var i: usize = 0;

    while (i < heights.len) {
        // While current bar is taller than stack top, we can trap water
        while (stack.items.len > 0 and heights[i] > heights[stack.items[stack.items.len - 1]]) {
            const top = stack.pop() orelse unreachable; // Safe: we checked len > 0
            if (stack.items.len == 0) break; // No left boundary

            const left = stack.items[stack.items.len - 1];
            const width_usize = i - left - 1;
            const width: T = switch (@typeInfo(T)) {
                .int, .comptime_int => @as(T, @intCast(width_usize)),
                .float, .comptime_float => @as(T, @floatFromInt(width_usize)),
                else => @compileError("trapStack only supports int and float types"),
            };
            const height = @min(heights[left], heights[i]) - heights[top];
            water += width * height;
        }

        stack.appendAssumeCapacity(i);
        i += 1;
    }

    return water;
}

/// Maximum Water Container Variant
///
/// Different problem: Given array of heights, find two lines that together with x-axis
/// form a container that holds the most water.
/// Area = min(height[i], height[j]) * (j - i)
///
/// Algorithm: Two pointers from ends, move pointer with smaller height.
///
/// Time: O(n) single pass
/// Space: O(1)
///
/// Examples:
/// ```zig
/// const heights = [_]u32{1, 8, 6, 2, 5, 4, 8, 3, 7};
/// const area = maxArea(u32, &heights); // Returns 49 (indices 1 and 8)
/// ```
pub fn maxArea(comptime T: type, heights: []const T) T {
    if (heights.len < 2) return 0;

    var left: usize = 0;
    var right: usize = heights.len - 1;
    var max_area: T = 0;

    while (left < right) {
        const width_usize = right - left;
        const width: T = switch (@typeInfo(T)) {
            .int, .comptime_int => @as(T, @intCast(width_usize)),
            .float, .comptime_float => @as(T, @floatFromInt(width_usize)),
            else => @compileError("maxArea only supports int and float types"),
        };
        const height = @min(heights[left], heights[right]);
        const area = width * height;
        max_area = @max(max_area, area);

        // Move pointer with smaller height
        if (heights[left] < heights[right]) {
            left += 1;
        } else {
            right -= 1;
        }
    }

    return max_area;
}

// ============================================================================
// Tests
// ============================================================================

test "trapping rain water: basic example" {
    const heights = [_]u32{ 0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1 };
    const water = try trap(u32, std.testing.allocator, &heights);
    try std.testing.expectEqual(@as(u32, 6), water);
}

test "trapping rain water: optimized basic example" {
    const heights = [_]u32{ 0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1 };
    const water = trapOptimized(u32, &heights);
    try std.testing.expectEqual(@as(u32, 6), water);
}

test "trapping rain water: stack approach basic" {
    const heights = [_]u32{ 0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1 };
    const water = try trapStack(u32, std.testing.allocator, &heights);
    try std.testing.expectEqual(@as(u32, 6), water);
}

test "trapping rain water: empty array" {
    const heights = [_]u32{};
    try std.testing.expectEqual(@as(u32, 0), try trap(u32, std.testing.allocator, &heights));
    try std.testing.expectEqual(@as(u32, 0), trapOptimized(u32, &heights));
    try std.testing.expectEqual(@as(u32, 0), try trapStack(u32, std.testing.allocator, &heights));
}

test "trapping rain water: single element" {
    const heights = [_]u32{5};
    try std.testing.expectEqual(@as(u32, 0), try trap(u32, std.testing.allocator, &heights));
    try std.testing.expectEqual(@as(u32, 0), trapOptimized(u32, &heights));
    try std.testing.expectEqual(@as(u32, 0), try trapStack(u32, std.testing.allocator, &heights));
}

test "trapping rain water: two elements" {
    const heights = [_]u32{ 3, 5 };
    try std.testing.expectEqual(@as(u32, 0), try trap(u32, std.testing.allocator, &heights));
    try std.testing.expectEqual(@as(u32, 0), trapOptimized(u32, &heights));
    try std.testing.expectEqual(@as(u32, 0), try trapStack(u32, std.testing.allocator, &heights));
}

test "trapping rain water: flat surface" {
    const heights = [_]u32{ 2, 2, 2, 2, 2 };
    try std.testing.expectEqual(@as(u32, 0), try trap(u32, std.testing.allocator, &heights));
    try std.testing.expectEqual(@as(u32, 0), trapOptimized(u32, &heights));
    try std.testing.expectEqual(@as(u32, 0), try trapStack(u32, std.testing.allocator, &heights));
}

test "trapping rain water: increasing sequence" {
    const heights = [_]u32{ 1, 2, 3, 4, 5 };
    try std.testing.expectEqual(@as(u32, 0), try trap(u32, std.testing.allocator, &heights));
    try std.testing.expectEqual(@as(u32, 0), trapOptimized(u32, &heights));
    try std.testing.expectEqual(@as(u32, 0), try trapStack(u32, std.testing.allocator, &heights));
}

test "trapping rain water: decreasing sequence" {
    const heights = [_]u32{ 5, 4, 3, 2, 1 };
    try std.testing.expectEqual(@as(u32, 0), try trap(u32, std.testing.allocator, &heights));
    try std.testing.expectEqual(@as(u32, 0), trapOptimized(u32, &heights));
    try std.testing.expectEqual(@as(u32, 0), try trapStack(u32, std.testing.allocator, &heights));
}

test "trapping rain water: valley" {
    const heights = [_]u32{ 3, 0, 2, 0, 4 };
    const expected: u32 = 7; // 3 + 1 + 3 = 7
    try std.testing.expectEqual(expected, try trap(u32, std.testing.allocator, &heights));
    try std.testing.expectEqual(expected, trapOptimized(u32, &heights));
    try std.testing.expectEqual(expected, try trapStack(u32, std.testing.allocator, &heights));
}

test "trapping rain water: multiple valleys" {
    const heights = [_]u32{ 4, 2, 0, 3, 2, 5 };
    const expected: u32 = 9; // 2 + 4 + 1 + 2 = 9
    try std.testing.expectEqual(expected, try trap(u32, std.testing.allocator, &heights));
    try std.testing.expectEqual(expected, trapOptimized(u32, &heights));
    try std.testing.expectEqual(expected, try trapStack(u32, std.testing.allocator, &heights));
}

test "trapping rain water: peak in middle" {
    const heights = [_]u32{ 2, 1, 2, 1, 3, 1, 2 };
    // leftMax  = [2, 2, 2, 2, 3, 3, 3]
    // rightMax = [3, 3, 3, 3, 3, 2, 2]
    // water[0] = min(2,3) - 2 = 0
    // water[1] = min(2,3) - 1 = 1
    // water[2] = min(2,3) - 2 = 0
    // water[3] = min(2,3) - 1 = 1
    // water[4] = min(3,3) - 3 = 0
    // water[5] = min(3,2) - 1 = 1
    // water[6] = min(3,2) - 2 = 0
    // Total = 3
    const expected: u32 = 3;
    try std.testing.expectEqual(expected, try trap(u32, std.testing.allocator, &heights));
    try std.testing.expectEqual(expected, trapOptimized(u32, &heights));
    try std.testing.expectEqual(expected, try trapStack(u32, std.testing.allocator, &heights));
}

test "trapping rain water: with details" {
    const heights = [_]u32{ 3, 0, 2, 0, 4 };
    var details = try trapWithDetails(u32, std.testing.allocator, &heights);
    defer details.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 7), details.total);
    try std.testing.expectEqual(@as(usize, 5), details.water_at.len);
    try std.testing.expectEqual(@as(u32, 0), details.water_at[0]); // height 3
    try std.testing.expectEqual(@as(u32, 3), details.water_at[1]); // height 0
    try std.testing.expectEqual(@as(u32, 1), details.water_at[2]); // height 2
    try std.testing.expectEqual(@as(u32, 3), details.water_at[3]); // height 0
    try std.testing.expectEqual(@as(u32, 0), details.water_at[4]); // height 4
}

test "trapping rain water: all zeros" {
    const heights = [_]u32{ 0, 0, 0, 0, 0 };
    try std.testing.expectEqual(@as(u32, 0), try trap(u32, std.testing.allocator, &heights));
    try std.testing.expectEqual(@as(u32, 0), trapOptimized(u32, &heights));
    try std.testing.expectEqual(@as(u32, 0), try trapStack(u32, std.testing.allocator, &heights));
}

test "trapping rain water: symmetric" {
    const heights = [_]u32{ 5, 2, 1, 2, 5 };
    // Actually: 0 + 3 + 4 + 3 + 0 = 10
    // leftMax = [5, 5, 5, 5, 5], rightMax = [5, 5, 5, 5, 5]
    // water[0] = min(5,5) - 5 = 0
    // water[1] = min(5,5) - 2 = 3
    // water[2] = min(5,5) - 1 = 4
    // water[3] = min(5,5) - 2 = 3
    // water[4] = min(5,5) - 5 = 0
    // Total = 10
    try std.testing.expectEqual(@as(u32, 10), try trap(u32, std.testing.allocator, &heights));
    try std.testing.expectEqual(@as(u32, 10), trapOptimized(u32, &heights));
    try std.testing.expectEqual(@as(u32, 10), try trapStack(u32, std.testing.allocator, &heights));
}

test "trapping rain water: large array" {
    var heights: [100]u32 = undefined;
    for (0..100) |i| {
        heights[i] = @as(u32, @intCast((i % 10) + 1));
    }
    const result1 = try trap(u32, std.testing.allocator, &heights);
    const result2 = trapOptimized(u32, &heights);
    const result3 = try trapStack(u32, std.testing.allocator, &heights);
    try std.testing.expectEqual(result1, result2);
    try std.testing.expectEqual(result1, result3);
}

test "trapping rain water: f64 support" {
    const heights = [_]f64{ 0.0, 1.5, 0.0, 2.5, 1.0, 0.0, 1.5, 3.0, 2.0, 1.0, 2.5, 1.0 };
    const result1 = try trap(f64, std.testing.allocator, &heights);
    const result2 = trapOptimized(f64, &heights);
    const result3 = try trapStack(f64, std.testing.allocator, &heights);
    try std.testing.expect(result1 > 0.0);
    try std.testing.expectApproxEqAbs(result1, result2, 0.0001);
    try std.testing.expectApproxEqAbs(result1, result3, 0.0001);
}

test "max area: basic example" {
    const heights = [_]u32{ 1, 8, 6, 2, 5, 4, 8, 3, 7 };
    const area = maxArea(u32, &heights);
    try std.testing.expectEqual(@as(u32, 49), area); // min(8,7) * (8-1) = 7 * 7 = 49
}

test "max area: two elements" {
    const heights = [_]u32{ 1, 2 };
    const area = maxArea(u32, &heights);
    try std.testing.expectEqual(@as(u32, 1), area); // min(1,2) * 1 = 1
}

test "max area: equal heights" {
    const heights = [_]u32{ 5, 5, 5, 5, 5 };
    const area = maxArea(u32, &heights);
    try std.testing.expectEqual(@as(u32, 20), area); // 5 * 4 = 20
}

test "max area: increasing" {
    const heights = [_]u32{ 1, 2, 3, 4, 5 };
    const area = maxArea(u32, &heights);
    try std.testing.expectEqual(@as(u32, 6), area); // min(2,5) * 3 = 2*3 or min(1,5) * 4 = 1*4 = 4, no it's min(2,5)*3 = 6
    // Actually: check all pairs: (1,5): 1*4=4, (2,5): 2*3=6, (3,5): 3*2=6, (4,5): 4*1=4
}

test "max area: f64 support" {
    const heights = [_]f64{ 1.5, 8.0, 6.5, 2.0, 5.0 };
    const area = maxArea(f64, &heights);
    try std.testing.expect(area > 0.0);
}

test "trapping rain water: memory safety" {
    for (0..10) |_| {
        const heights = [_]u32{ 0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1 };
        const result1 = try trap(u32, std.testing.allocator, &heights);
        const result2 = try trapStack(u32, std.testing.allocator, &heights);
        var details = try trapWithDetails(u32, std.testing.allocator, &heights);
        details.deinit(std.testing.allocator);
        try std.testing.expectEqual(@as(u32, 6), result1);
        try std.testing.expectEqual(@as(u32, 6), result2);
    }
}
