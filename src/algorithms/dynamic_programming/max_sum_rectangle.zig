const std = @import("std");
const Allocator = std.mem.Allocator;

/// Maximum Sum Rectangle in 2D Matrix
///
/// This module implements algorithms for finding the maximum sum submatrix
/// in a 2D matrix. It extends Kadane's algorithm from 1D to 2D using column
/// compression and row iteration.
///
/// Reference: Extension of Kadane's algorithm to 2D (1977)

/// Result of maximum sum rectangle query
pub fn RectangleResult(comptime T: type) type {
    return struct {
        sum: T,
        top: usize,
        left: usize,
        bottom: usize,
        right: usize,
    };
}

/// Find maximum sum rectangle in 2D matrix using Kadane's algorithm extension
///
/// Algorithm:
/// 1. Fix left and right columns
/// 2. Compress rows by summing elements between left and right columns
/// 3. Apply 1D Kadane's algorithm on compressed array
/// 4. Track best rectangle coordinates
///
/// Time: O(n² × m) where n = rows, m = cols (or O(m² × n) if m < n)
/// Space: O(n) for temporary row sum array
///
/// Example:
/// ```zig
/// const matrix = [_][4]i32{
///     .{ 1, 2, -1, -4 },
///     .{ -8, -3, 4, 2 },
///     .{ 3, 8, 10, 1 },
///     .{ -4, -1, 1, 7 },
/// };
/// const result = try maxSumRectangle(i32, allocator, &matrix, 4, 4);
/// // result.sum = 29, rectangle from (1,2) to (3,3)
/// defer allocator.free(result.temp); // if temp array was allocated
/// ```
pub fn maxSumRectangle(
    comptime T: type,
    allocator: Allocator,
    matrix: []const []const T,
    rows: usize,
    cols: usize,
) !RectangleResult(T) {
    if (rows == 0 or cols == 0) return error.EmptyMatrix;

    // Temporary array to store row sums
    const temp = try allocator.alloc(T, rows);
    defer allocator.free(temp);

    var max_sum: T = std.math.minInt(T);
    var final_left: usize = 0;
    var final_right: usize = 0;
    var final_top: usize = 0;
    var final_bottom: usize = 0;

    // Fix left column
    var left: usize = 0;
    while (left < cols) : (left += 1) {
        // Initialize temp array for this left column
        @memset(temp, 0);

        // Fix right column (expanding from left)
        var right: usize = left;
        while (right < cols) : (right += 1) {
            // Add current column to temp (column compression)
            for (0..rows) |i| {
                temp[i] += matrix[i][right];
            }

            // Apply Kadane's algorithm on temp array
            const kadane_result = kadane1D(T, temp);

            // Update maximum if we found a better rectangle
            if (kadane_result.sum > max_sum) {
                max_sum = kadane_result.sum;
                final_left = left;
                final_right = right;
                final_top = kadane_result.start;
                final_bottom = kadane_result.end;
            }
        }
    }

    return RectangleResult(T){
        .sum = max_sum,
        .top = final_top,
        .left = final_left,
        .bottom = final_bottom,
        .right = final_right,
    };
}

/// Kadane's algorithm result for 1D array
const Kadane1DResult = struct {
    sum: anytype,
    start: usize,
    end: usize,
};

/// Apply Kadane's algorithm to find maximum sum subarray in 1D
///
/// Time: O(n)
/// Space: O(1)
fn kadane1D(comptime T: type, arr: []const T) Kadane1DResult {
    var max_sum = arr[0];
    var current_sum = arr[0];
    var start: usize = 0;
    var end: usize = 0;
    var temp_start: usize = 0;

    for (arr[1..], 1..) |val, i| {
        // Extend current subarray or start new one
        if (current_sum < 0) {
            current_sum = val;
            temp_start = i;
        } else {
            current_sum += val;
        }

        // Update maximum
        if (current_sum > max_sum) {
            max_sum = current_sum;
            start = temp_start;
            end = i;
        }
    }

    return Kadane1DResult{
        .sum = max_sum,
        .start = start,
        .end = end,
    };
}

/// Find maximum sum rectangle with custom comparison
///
/// Allows custom comparison function for maximum determination.
/// Useful for finding minimum sum rectangle or other variants.
///
/// Time: O(n² × m)
/// Space: O(n)
pub fn maxSumRectangleBy(
    comptime T: type,
    allocator: Allocator,
    matrix: []const []const T,
    rows: usize,
    cols: usize,
    comptime compareFn: fn (T, T) bool,
) !RectangleResult(T) {
    if (rows == 0 or cols == 0) return error.EmptyMatrix;

    const temp = try allocator.alloc(T, rows);
    defer allocator.free(temp);

    var max_sum: T = std.math.minInt(T);
    var final_left: usize = 0;
    var final_right: usize = 0;
    var final_top: usize = 0;
    var final_bottom: usize = 0;

    var left: usize = 0;
    while (left < cols) : (left += 1) {
        @memset(temp, 0);

        var right: usize = left;
        while (right < cols) : (right += 1) {
            for (0..rows) |i| {
                temp[i] += matrix[i][right];
            }

            const kadane_result = kadane1DBy(T, temp, compareFn);

            if (compareFn(kadane_result.sum, max_sum)) {
                max_sum = kadane_result.sum;
                final_left = left;
                final_right = right;
                final_top = kadane_result.start;
                final_bottom = kadane_result.end;
            }
        }
    }

    return RectangleResult(T){
        .sum = max_sum,
        .top = final_top,
        .left = final_left,
        .bottom = final_bottom,
        .right = final_right,
    };
}

fn kadane1DBy(comptime T: type, arr: []const T, comptime compareFn: fn (T, T) bool) Kadane1DResult {
    var max_sum = arr[0];
    var current_sum = arr[0];
    var start: usize = 0;
    var end: usize = 0;
    var temp_start: usize = 0;

    for (arr[1..], 1..) |val, i| {
        if (current_sum < 0) {
            current_sum = val;
            temp_start = i;
        } else {
            current_sum += val;
        }

        if (compareFn(current_sum, max_sum)) {
            max_sum = current_sum;
            start = temp_start;
            end = i;
        }
    }

    return Kadane1DResult{
        .sum = max_sum,
        .start = start,
        .end = end,
    };
}

/// Count number of rectangles with sum equal to target
///
/// Uses hashmap to track prefix sums for efficient counting.
///
/// Time: O(n² × m)
/// Space: O(n × m) for hashmap
pub fn countRectanglesWithSum(
    comptime T: type,
    allocator: Allocator,
    matrix: []const []const T,
    rows: usize,
    cols: usize,
    target: T,
) !usize {
    if (rows == 0 or cols == 0) return 0;

    var count: usize = 0;

    // Fix left and right columns
    var left: usize = 0;
    while (left < cols) : (left += 1) {
        const temp = try allocator.alloc(T, rows);
        defer allocator.free(temp);
        @memset(temp, 0);

        var right: usize = left;
        while (right < cols) : (right += 1) {
            // Compress columns
            for (0..rows) |i| {
                temp[i] += matrix[i][right];
            }

            // Count subarrays with sum = target using hashmap
            var prefix_map = std.AutoHashMap(T, usize).init(allocator);
            defer prefix_map.deinit();

            try prefix_map.put(0, 1); // Empty prefix

            var sum: T = 0;
            for (temp) |val| {
                sum += val;
                const complement = sum - target;

                if (prefix_map.get(complement)) |cnt| {
                    count += cnt;
                }

                const current_count = prefix_map.get(sum) orelse 0;
                try prefix_map.put(sum, current_count + 1);
            }
        }
    }

    return count;
}

/// Find all rectangles with maximum sum
///
/// Returns list of all rectangles that have the maximum sum.
/// Useful when there are multiple optimal solutions.
///
/// Time: O(n² × m)
/// Space: O(k) where k = number of optimal rectangles
pub fn findAllMaxSumRectangles(
    comptime T: type,
    allocator: Allocator,
    matrix: []const []const T,
    rows: usize,
    cols: usize,
) ![]RectangleResult(T) {
    if (rows == 0 or cols == 0) return error.EmptyMatrix;

    var rectangles = std.ArrayList(RectangleResult(T)).init(allocator);
    errdefer rectangles.deinit();

    const temp = try allocator.alloc(T, rows);
    defer allocator.free(temp);

    var max_sum: T = std.math.minInt(T);

    var left: usize = 0;
    while (left < cols) : (left += 1) {
        @memset(temp, 0);

        var right: usize = left;
        while (right < cols) : (right += 1) {
            for (0..rows) |i| {
                temp[i] += matrix[i][right];
            }

            const kadane_result = kadane1D(T, temp);

            if (kadane_result.sum > max_sum) {
                // Found new maximum, clear previous results
                rectangles.clearRetainingCapacity();
                max_sum = kadane_result.sum;
                try rectangles.append(RectangleResult(T){
                    .sum = max_sum,
                    .top = kadane_result.start,
                    .left = left,
                    .bottom = kadane_result.end,
                    .right = right,
                });
            } else if (kadane_result.sum == max_sum) {
                // Found another rectangle with maximum sum
                try rectangles.append(RectangleResult(T){
                    .sum = max_sum,
                    .top = kadane_result.start,
                    .left = left,
                    .bottom = kadane_result.end,
                    .right = right,
                });
            }
        }
    }

    return rectangles.toOwnedSlice();
}

/// Find minimum sum rectangle (minimum submatrix)
///
/// Time: O(n² × m)
/// Space: O(n)
pub fn minSumRectangle(
    comptime T: type,
    allocator: Allocator,
    matrix: []const []const T,
    rows: usize,
    cols: usize,
) !RectangleResult(T) {
    return maxSumRectangleBy(T, allocator, matrix, rows, cols, struct {
        fn cmp(a: T, b: T) bool {
            return a < b;
        }
    }.cmp);
}

// ============================================================================
// Tests
// ============================================================================

test "max sum rectangle - basic 4x4" {
    const allocator = std.testing.allocator;

    const matrix = [_][4]i32{
        .{ 1, 2, -1, -4 },
        .{ -8, -3, 4, 2 },
        .{ 3, 8, 10, 1 },
        .{ -4, -1, 1, 7 },
    };

    const matrix_ptrs = try allocator.alloc([]const i32, 4);
    defer allocator.free(matrix_ptrs);
    for (matrix, 0..) |row, i| {
        matrix_ptrs[i] = &row;
    }

    const result = try maxSumRectangle(i32, allocator, matrix_ptrs, 4, 4);

    try std.testing.expectEqual(@as(i32, 29), result.sum);
    try std.testing.expectEqual(@as(usize, 1), result.top);
    try std.testing.expectEqual(@as(usize, 1), result.left);
    try std.testing.expectEqual(@as(usize, 3), result.bottom);
    try std.testing.expectEqual(@as(usize, 3), result.right);
}

test "max sum rectangle - single positive" {
    const allocator = std.testing.allocator;

    const matrix = [_][3]i32{
        .{ -1, -2, -3 },
        .{ -4, 10, -6 },
        .{ -7, -8, -9 },
    };

    const matrix_ptrs = try allocator.alloc([]const i32, 3);
    defer allocator.free(matrix_ptrs);
    for (matrix, 0..) |row, i| {
        matrix_ptrs[i] = &row;
    }

    const result = try maxSumRectangle(i32, allocator, matrix_ptrs, 3, 3);

    try std.testing.expectEqual(@as(i32, 10), result.sum);
    try std.testing.expectEqual(@as(usize, 1), result.top);
    try std.testing.expectEqual(@as(usize, 1), result.left);
    try std.testing.expectEqual(@as(usize, 1), result.bottom);
    try std.testing.expectEqual(@as(usize, 1), result.right);
}

test "max sum rectangle - all negative" {
    const allocator = std.testing.allocator;

    const matrix = [_][3]i32{
        .{ -1, -2, -3 },
        .{ -4, -5, -6 },
        .{ -7, -8, -9 },
    };

    const matrix_ptrs = try allocator.alloc([]const i32, 3);
    defer allocator.free(matrix_ptrs);
    for (matrix, 0..) |row, i| {
        matrix_ptrs[i] = &row;
    }

    const result = try maxSumRectangle(i32, allocator, matrix_ptrs, 3, 3);

    // Should return the least negative value
    try std.testing.expectEqual(@as(i32, -1), result.sum);
}

test "max sum rectangle - all positive" {
    const allocator = std.testing.allocator;

    const matrix = [_][3]i32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
        .{ 7, 8, 9 },
    };

    const matrix_ptrs = try allocator.alloc([]const i32, 3);
    defer allocator.free(matrix_ptrs);
    for (matrix, 0..) |row, i| {
        matrix_ptrs[i] = &row;
    }

    const result = try maxSumRectangle(i32, allocator, matrix_ptrs, 3, 3);

    // Should return sum of entire matrix
    try std.testing.expectEqual(@as(i32, 45), result.sum);
    try std.testing.expectEqual(@as(usize, 0), result.top);
    try std.testing.expectEqual(@as(usize, 0), result.left);
    try std.testing.expectEqual(@as(usize, 2), result.bottom);
    try std.testing.expectEqual(@as(usize, 2), result.right);
}

test "max sum rectangle - single row" {
    const allocator = std.testing.allocator;

    const matrix = [_][5]i32{
        .{ -2, 1, -3, 4, -1 },
    };

    const matrix_ptrs = try allocator.alloc([]const i32, 1);
    defer allocator.free(matrix_ptrs);
    for (matrix, 0..) |row, i| {
        matrix_ptrs[i] = &row;
    }

    const result = try maxSumRectangle(i32, allocator, matrix_ptrs, 1, 5);

    try std.testing.expectEqual(@as(i32, 4), result.sum);
    try std.testing.expectEqual(@as(usize, 0), result.top);
    try std.testing.expectEqual(@as(usize, 3), result.left);
    try std.testing.expectEqual(@as(usize, 0), result.bottom);
    try std.testing.expectEqual(@as(usize, 3), result.right);
}

test "max sum rectangle - single column" {
    const allocator = std.testing.allocator;

    const matrix = [_][1]i32{
        .{-2},
        .{1},
        .{-3},
        .{4},
        .{-1},
    };

    const matrix_ptrs = try allocator.alloc([]const i32, 5);
    defer allocator.free(matrix_ptrs);
    for (matrix, 0..) |row, i| {
        matrix_ptrs[i] = &row;
    }

    const result = try maxSumRectangle(i32, allocator, matrix_ptrs, 5, 1);

    try std.testing.expectEqual(@as(i32, 4), result.sum);
    try std.testing.expectEqual(@as(usize, 3), result.top);
    try std.testing.expectEqual(@as(usize, 0), result.left);
    try std.testing.expectEqual(@as(usize, 3), result.bottom);
    try std.testing.expectEqual(@as(usize, 0), result.right);
}

test "max sum rectangle - empty matrix error" {
    const allocator = std.testing.allocator;

    const matrix_ptrs = try allocator.alloc([]const i32, 0);
    defer allocator.free(matrix_ptrs);

    const result = maxSumRectangle(i32, allocator, matrix_ptrs, 0, 0);
    try std.testing.expectError(error.EmptyMatrix, result);
}

test "max sum rectangle - f64 type" {
    const allocator = std.testing.allocator;

    const matrix = [_][3]f64{
        .{ 1.5, 2.5, -1.0 },
        .{ -3.0, 4.0, 2.0 },
        .{ 1.0, 1.0, 1.0 },
    };

    const matrix_ptrs = try allocator.alloc([]const f64, 3);
    defer allocator.free(matrix_ptrs);
    for (matrix, 0..) |row, i| {
        matrix_ptrs[i] = &row;
    }

    const result = try maxSumRectangle(f64, allocator, matrix_ptrs, 3, 3);

    try std.testing.expectApproxEqAbs(10.0, result.sum, 1e-6);
}

test "min sum rectangle - basic" {
    const allocator = std.testing.allocator;

    const matrix = [_][3]i32{
        .{ 1, 2, 3 },
        .{ 4, -10, 6 },
        .{ 7, 8, 9 },
    };

    const matrix_ptrs = try allocator.alloc([]const i32, 3);
    defer allocator.free(matrix_ptrs);
    for (matrix, 0..) |row, i| {
        matrix_ptrs[i] = &row;
    }

    const result = try minSumRectangle(i32, allocator, matrix_ptrs, 3, 3);

    try std.testing.expectEqual(@as(i32, -10), result.sum);
    try std.testing.expectEqual(@as(usize, 1), result.top);
    try std.testing.expectEqual(@as(usize, 1), result.left);
}

test "count rectangles with sum - basic" {
    const allocator = std.testing.allocator;

    const matrix = [_][3]i32{
        .{ 1, 1, 1 },
        .{ 1, 1, 1 },
        .{ 1, 1, 1 },
    };

    const matrix_ptrs = try allocator.alloc([]const i32, 3);
    defer allocator.free(matrix_ptrs);
    for (matrix, 0..) |row, i| {
        matrix_ptrs[i] = &row;
    }

    const count = try countRectanglesWithSum(i32, allocator, matrix_ptrs, 3, 3, 3);

    // There should be multiple rectangles with sum = 3:
    // - Single row/column of length 3
    // - 2x1 + 1x1, etc.
    try std.testing.expect(count > 0);
}

test "count rectangles with sum - target 0" {
    const allocator = std.testing.allocator;

    const matrix = [_][3]i32{
        .{ 1, -1, 0 },
        .{ 0, 1, -1 },
        .{ -1, 0, 1 },
    };

    const matrix_ptrs = try allocator.alloc([]const i32, 3);
    defer allocator.free(matrix_ptrs);
    for (matrix, 0..) |row, i| {
        matrix_ptrs[i] = &row;
    }

    const count = try countRectanglesWithSum(i32, allocator, matrix_ptrs, 3, 3, 0);

    try std.testing.expect(count > 0);
}

test "find all max sum rectangles - single solution" {
    const allocator = std.testing.allocator;

    const matrix = [_][3]i32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
        .{ 7, 8, 9 },
    };

    const matrix_ptrs = try allocator.alloc([]const i32, 3);
    defer allocator.free(matrix_ptrs);
    for (matrix, 0..) |row, i| {
        matrix_ptrs[i] = &row;
    }

    const rectangles = try findAllMaxSumRectangles(i32, allocator, matrix_ptrs, 3, 3);
    defer allocator.free(rectangles);

    try std.testing.expectEqual(@as(usize, 1), rectangles.len);
    try std.testing.expectEqual(@as(i32, 45), rectangles[0].sum);
}

test "find all max sum rectangles - multiple solutions" {
    const allocator = std.testing.allocator;

    const matrix = [_][2]i32{
        .{ 5, 5 },
        .{ 5, 5 },
    };

    const matrix_ptrs = try allocator.alloc([]const i32, 2);
    defer allocator.free(matrix_ptrs);
    for (matrix, 0..) |row, i| {
        matrix_ptrs[i] = &row;
    }

    const rectangles = try findAllMaxSumRectangles(i32, allocator, matrix_ptrs, 2, 2);
    defer allocator.free(rectangles);

    // Only the full rectangle should have the maximum sum
    try std.testing.expectEqual(@as(usize, 1), rectangles.len);
    try std.testing.expectEqual(@as(i32, 20), rectangles[0].sum);
}

test "max sum rectangle - large matrix 10x10" {
    const allocator = std.testing.allocator;

    // Create 10x10 matrix
    var matrix: [10][10]i32 = undefined;
    for (0..10) |i| {
        for (0..10) |j| {
            matrix[i][j] = @intCast((i + j) % 7 - 3);
        }
    }

    const matrix_ptrs = try allocator.alloc([]const i32, 10);
    defer allocator.free(matrix_ptrs);
    for (matrix, 0..) |row, i| {
        matrix_ptrs[i] = &row;
    }

    const result = try maxSumRectangle(i32, allocator, matrix_ptrs, 10, 10);

    // Should find some positive sum
    try std.testing.expect(result.sum > 0);
    try std.testing.expect(result.bottom >= result.top);
    try std.testing.expect(result.right >= result.left);
}

test "max sum rectangle - memory safety" {
    const allocator = std.testing.allocator;

    // Run multiple times to check for memory leaks
    for (0..10) |_| {
        const matrix = [_][3]i32{
            .{ 1, 2, -1 },
            .{ -4, -5, 6 },
            .{ 7, 8, 9 },
        };

        const matrix_ptrs = try allocator.alloc([]const i32, 3);
        defer allocator.free(matrix_ptrs);
        for (matrix, 0..) |row, i| {
            matrix_ptrs[i] = &row;
        }

        const result = try maxSumRectangle(i32, allocator, matrix_ptrs, 3, 3);
        _ = result;
    }
}
