const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Maximum product subarray result with location information
pub fn ProductResult(comptime T: type) type {
    return struct {
        product: T,
        start: usize,
        end: usize,

        /// Get the subarray length
        pub fn length(self: @This()) usize {
            return self.end - self.start + 1;
        }
    };
}

/// Find maximum product of contiguous subarray
///
/// Algorithm: Track both maximum and minimum products at each position
/// because a negative number can make the minimum become maximum
///
/// Time: O(n)
/// Space: O(1)
pub fn maxProduct(comptime T: type, nums: []const T) ?T {
    if (nums.len == 0) return null;
    if (nums.len == 1) return nums[0];

    var max_prod = nums[0];
    var min_prod = nums[0];
    var result = nums[0];

    for (nums[1..]) |num| {
        // When multiplied by a negative number, max becomes min and vice versa
        if (num < 0) {
            const temp = max_prod;
            max_prod = min_prod;
            min_prod = temp;
        }

        // Update max and min products
        max_prod = @max(num, max_prod * num);
        min_prod = @min(num, min_prod * num);

        // Update global result
        result = @max(result, max_prod);
    }

    return result;
}

/// Find maximum product with start and end indices
///
/// Time: O(n)
/// Space: O(1)
pub fn maxProductWithIndices(comptime T: type, nums: []const T) ?ProductResult(T) {
    if (nums.len == 0) return null;
    if (nums.len == 1) return ProductResult(T){ .product = nums[0], .start = 0, .end = 0 };

    var max_prod = nums[0];
    var min_prod = nums[0];
    var result = nums[0];
    var result_start: usize = 0;
    var result_end: usize = 0;
    var current_start: usize = 0;

    for (nums[1..], 1..) |num, i| {
        // When multiplied by a negative number, max becomes min
        if (num < 0) {
            const temp = max_prod;
            max_prod = min_prod;
            min_prod = temp;
        }

        // Reset start position if starting fresh
        if (num > max_prod * num) {
            current_start = i;
        }

        // Update max and min products
        max_prod = @max(num, max_prod * num);
        min_prod = @min(num, min_prod * num);

        // Update global result with indices
        if (max_prod > result) {
            result = max_prod;
            result_start = current_start;
            result_end = i;
        }
    }

    return ProductResult(T){
        .product = result,
        .start = result_start,
        .end = result_end,
    };
}

/// Find maximum product using tabulation (for clarity)
///
/// Time: O(n)
/// Space: O(n)
pub fn maxProductTable(comptime T: type, allocator: Allocator, nums: []const T) !?T {
    if (nums.len == 0) return null;
    if (nums.len == 1) return nums[0];

    const n = nums.len;

    // max_dp[i] = maximum product ending at index i
    // min_dp[i] = minimum product ending at index i (for negative numbers)
    var max_dp = try allocator.alloc(T, n);
    defer allocator.free(max_dp);
    var min_dp = try allocator.alloc(T, n);
    defer allocator.free(min_dp);

    max_dp[0] = nums[0];
    min_dp[0] = nums[0];
    var result = nums[0];

    for (1..n) |i| {
        // Three choices: start fresh, extend max, extend min
        max_dp[i] = @max(@max(nums[i], max_dp[i - 1] * nums[i]), min_dp[i - 1] * nums[i]);
        min_dp[i] = @min(@min(nums[i], max_dp[i - 1] * nums[i]), min_dp[i - 1] * nums[i]);
        result = @max(result, max_dp[i]);
    }

    return result;
}

/// Handle zeros by splitting array into subarrays
///
/// Time: O(n)
/// Space: O(1)
pub fn maxProductWithZeros(comptime T: type, nums: []const T) ?T {
    if (nums.len == 0) return null;

    var result: ?T = null;
    var start: usize = 0;

    // Split by zeros and find max in each subarray
    for (nums, 0..) |num, i| {
        if (num == 0) {
            // Process subarray [start..i)
            if (start < i) {
                const sub_result = maxProduct(T, nums[start..i]);
                if (sub_result) |val| {
                    result = if (result) |r| @max(r, val) else val;
                }
            }
            // Zero itself could be the answer
            result = if (result) |r| @max(r, 0) else 0;
            start = i + 1;
        }
    }

    // Process remaining subarray
    if (start < nums.len) {
        const sub_result = maxProduct(T, nums[start..]);
        if (sub_result) |val| {
            result = if (result) |r| @max(r, val) else val;
        }
    }

    return result;
}

/// Count subarrays with product greater than k
///
/// Time: O(n²) - checks all subarrays
/// Space: O(1)
pub fn countProductGreaterThan(comptime T: type, nums: []const T, k: T) usize {
    if (nums.len == 0) return 0;

    var count: usize = 0;

    for (0..nums.len) |i| {
        var product: T = 1;
        for (i..nums.len) |j| {
            product *= nums[j];
            if (product > k) {
                count += 1;
            }
            // Early exit for zero product
            if (product == 0) break;
        }
    }

    return count;
}

/// Find minimum product subarray (for completeness)
///
/// Time: O(n)
/// Space: O(1)
pub fn minProduct(comptime T: type, nums: []const T) ?T {
    if (nums.len == 0) return null;
    if (nums.len == 1) return nums[0];

    var max_prod = nums[0];
    var min_prod = nums[0];
    var result = nums[0];

    for (nums[1..]) |num| {
        if (num < 0) {
            const temp = max_prod;
            max_prod = min_prod;
            min_prod = temp;
        }

        max_prod = @max(num, max_prod * num);
        min_prod = @min(num, min_prod * num);

        result = @min(result, min_prod);
    }

    return result;
}

// ========================== Tests ==========================

test "maxProduct - basic cases" {
    try testing.expectEqual(@as(i32, 6), maxProduct(i32, &.{ 2, 3, -2, 4 }).?);
    try testing.expectEqual(@as(i32, 0), maxProduct(i32, &.{ -2, 0, -1 }).?);
    try testing.expectEqual(@as(i32, 24), maxProduct(i32, &.{ 2, 3, -2, 4, -1 }).?);
    try testing.expectEqual(@as(i32, 2), maxProduct(i32, &.{ 2 }).?);
}

test "maxProduct - all negative" {
    try testing.expectEqual(@as(i32, 12), maxProduct(i32, &.{ -2, -3, -4 }).?);
    try testing.expectEqual(@as(i32, -2), maxProduct(i32, &.{ -2, -3, -1 }).?);
}

test "maxProduct - all positive" {
    try testing.expectEqual(@as(i32, 120), maxProduct(i32, &.{ 2, 3, 4, 5 }).?);
}

test "maxProduct - with zeros" {
    try testing.expectEqual(@as(i32, 6), maxProduct(i32, &.{ 2, 0, 3, 2 }).?);
    try testing.expectEqual(@as(i32, 0), maxProduct(i32, &.{ 0, 2 }).?);
    try testing.expectEqual(@as(i32, 6), maxProduct(i32, &.{ 0, 2, 3 }).?);
}

test "maxProduct - single negative" {
    try testing.expectEqual(@as(i32, 12), maxProduct(i32, &.{ 2, -5, -3, 2 }).?);
}

test "maxProduct - alternating signs" {
    try testing.expectEqual(@as(i32, 8), maxProduct(i32, &.{ -2, 1, -4 }).?);
}

test "maxProduct - empty array" {
    try testing.expectEqual(@as(?i32, null), maxProduct(i32, &.{}));
}

test "maxProductWithIndices - basic" {
    const result = maxProductWithIndices(i32, &.{ 2, 3, -2, 4 }).?;
    try testing.expectEqual(@as(i32, 6), result.product);
    try testing.expectEqual(@as(usize, 0), result.start);
    try testing.expectEqual(@as(usize, 1), result.end);
    try testing.expectEqual(@as(usize, 2), result.length());
}

test "maxProductWithIndices - all negative" {
    const result = maxProductWithIndices(i32, &.{ -2, -3, -4 }).?;
    try testing.expectEqual(@as(i32, 12), result.product);
    try testing.expectEqual(@as(usize, 1), result.start);
    try testing.expectEqual(@as(usize, 2), result.end);
}

test "maxProductWithIndices - single element" {
    const result = maxProductWithIndices(i32, &.{5}).?;
    try testing.expectEqual(@as(i32, 5), result.product);
    try testing.expectEqual(@as(usize, 0), result.start);
    try testing.expectEqual(@as(usize, 0), result.end);
}

test "maxProductTable - basic cases" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(i32, 6), (try maxProductTable(i32, allocator, &.{ 2, 3, -2, 4 })).?);
    try testing.expectEqual(@as(i32, 0), (try maxProductTable(i32, allocator, &.{ -2, 0, -1 })).?);
    try testing.expectEqual(@as(i32, 24), (try maxProductTable(i32, allocator, &.{ 2, 3, -2, 4, -1 })).?);
}

test "maxProductTable - memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        _ = try maxProductTable(i32, allocator, &.{ 2, 3, -2, 4 });
    }
}

test "maxProductWithZeros - splits by zeros" {
    try testing.expectEqual(@as(i32, 6), maxProductWithZeros(i32, &.{ 2, 3, 0, 4 }).?);
    try testing.expectEqual(@as(i32, 6), maxProductWithZeros(i32, &.{ 0, 2, 3, 0, 4 }).?);
    try testing.expectEqual(@as(i32, 0), maxProductWithZeros(i32, &.{ 0, 0, 0 }).?);
}

test "maxProductWithZeros - no zeros" {
    try testing.expectEqual(@as(i32, 24), maxProductWithZeros(i32, &.{ 2, 3, 4 }).?);
}

test "countProductGreaterThan - basic" {
    try testing.expectEqual(@as(usize, 6), countProductGreaterThan(i32, &.{ 1, 2, 3, 4 }, 1));
    try testing.expectEqual(@as(usize, 3), countProductGreaterThan(i32, &.{ 1, 2, 3, 4 }, 3));
}

test "countProductGreaterThan - with negatives" {
    try testing.expectEqual(@as(usize, 4), countProductGreaterThan(i32, &.{ -2, -3, 4 }, 5));
}

test "countProductGreaterThan - empty" {
    try testing.expectEqual(@as(usize, 0), countProductGreaterThan(i32, &.{}, 1));
}

test "minProduct - basic" {
    try testing.expectEqual(@as(i32, -24), minProduct(i32, &.{ 2, 3, -2, 4 }).?);
    try testing.expectEqual(@as(i32, -2), minProduct(i32, &.{ -2, 0, -1 }).?);
}

test "minProduct - all positive" {
    try testing.expectEqual(@as(i32, 2), minProduct(i32, &.{ 2, 3, 4 }).?);
}

test "maxProduct - floating point" {
    try testing.expectEqual(@as(f64, 6.0), maxProduct(f64, &.{ 2.0, 3.0, -2.0, 4.0 }).?);
    try testing.expect(@abs(maxProduct(f64, &.{ 2.5, 1.5 }).? - 3.75) < 1e-10);
}

test "maxProduct - large values" {
    const nums = [_]i64{ 10, 20, -5, 30 };
    try testing.expectEqual(@as(i64, 3000), maxProduct(i64, &nums).?);
}

test "maxProduct - stress test" {
    const allocator = testing.allocator;
    const nums = try allocator.alloc(i32, 1000);
    defer allocator.free(nums);

    // Fill with pattern: 2, -1, 2, -1, ...
    for (nums, 0..) |*num, i| {
        num.* = if (i % 2 == 0) 2 else -1;
    }

    const result = maxProduct(i32, nums);
    try testing.expect(result != null);
    try testing.expect(result.? > 0); // Should find even-length subarray
}
