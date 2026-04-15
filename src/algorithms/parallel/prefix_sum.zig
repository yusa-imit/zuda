const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const Thread = std.Thread;

/// Parallel Prefix Sum (Scan) algorithms
///
/// Prefix sum computes: output[i] = sum(input[0..i+1])
/// Example: [3, 1, 7, 0, 4] → [3, 4, 11, 11, 15]
///
/// Applications:
/// - Stream compaction and filtering
/// - Radix sort (parallel digit distribution)
/// - Tree operations (parallel level traversal)
/// - String matching (parallel failure function)
/// - Quicksort partitioning (parallel partition)
/// - Lexical analysis (parallel token scanning)
///
/// Algorithms:
/// 1. Sequential scan: O(n) time, O(1) space
/// 2. Work-efficient parallel scan (Blelloch): O(n) work, O(log n) depth
/// 3. Parallel inclusive/exclusive scan variants

/// Sequential prefix sum (inclusive scan)
/// output[i] = sum(input[0..i+1])
/// Time: O(n) | Space: O(n)
pub fn inclusiveScan(comptime T: type, input: []const T, allocator: Allocator) ![]T {
    if (input.len == 0) return try allocator.alloc(T, 0);

    const output = try allocator.alloc(T, input.len);
    errdefer allocator.free(output);

    output[0] = input[0];
    for (1..input.len) |i| {
        output[i] = output[i - 1] + input[i];
    }

    return output;
}

/// Sequential prefix sum (exclusive scan)
/// output[i] = sum(input[0..i])
/// output[0] = 0
/// Time: O(n) | Space: O(n)
pub fn exclusiveScan(comptime T: type, input: []const T, allocator: Allocator) ![]T {
    if (input.len == 0) return try allocator.alloc(T, 0);

    const output = try allocator.alloc(T, input.len);
    errdefer allocator.free(output);

    output[0] = 0;
    for (1..input.len) |i| {
        output[i] = output[i - 1] + input[i - 1];
    }

    return output;
}

/// Work-efficient parallel prefix sum (Blelloch algorithm)
/// Uses up-sweep (reduce) and down-sweep phases
/// Work: O(n) | Depth: O(log n) | Space: O(n)
pub fn parallelScan(comptime T: type, input: []const T, allocator: Allocator, num_threads: usize) ![]T {
    if (input.len == 0) return try allocator.alloc(T, 0);
    if (num_threads <= 1 or input.len < 1024) {
        // Fall back to sequential for small inputs or single thread
        return exclusiveScan(T, input, allocator);
    }

    const n = input.len;
    const output = try allocator.alloc(T, n);
    errdefer allocator.free(output);

    // Copy input to output (working array)
    @memcpy(output, input);

    // Up-sweep phase: build reduction tree
    // For each level d from 0 to log(n)-1:
    //   For i with stride 2^(d+1):
    //     output[i*2^(d+1) + 2^(d+1) - 1] += output[i*2^(d+1) + 2^d - 1]
    var d: usize = 0;
    var stride: usize = 1;
    while (stride < n) : ({
        d += 1;
        stride *= 2;
    }) {
        const next_stride = stride * 2;
        var i: usize = 0;
        while (i < n) : (i += next_stride) {
            const left_idx = i + stride - 1;
            const right_idx = i + next_stride - 1;
            if (right_idx < n) {
                output[right_idx] = output[right_idx] + output[left_idx];
            }
        }
    }

    // Set last element to 0 (for exclusive scan)
    output[n - 1] = 0;

    // Down-sweep phase: traverse down tree to build scan
    // For each level d from log(n)-1 down to 0:
    //   For i with stride 2^(d+1):
    //     temp = output[i*2^(d+1) + 2^d - 1]
    //     output[i*2^(d+1) + 2^d - 1] = output[i*2^(d+1) + 2^(d+1) - 1]
    //     output[i*2^(d+1) + 2^(d+1) - 1] += temp
    stride = stride / 2;
    while (stride > 0) : (stride /= 2) {
        const next_stride = stride * 2;
        var i: usize = 0;
        while (i < n) : (i += next_stride) {
            const left_idx = i + stride - 1;
            const right_idx = i + next_stride - 1;
            if (right_idx < n) {
                const temp = output[left_idx];
                output[left_idx] = output[right_idx];
                output[right_idx] = output[right_idx] + temp;
            }
        }
    }

    return output;
}

/// In-place prefix sum (modifies input array)
/// Time: O(n) | Space: O(1)
pub fn scanInPlace(comptime T: type, data: []T) void {
    if (data.len <= 1) return;

    for (1..data.len) |i| {
        data[i] = data[i] + data[i - 1];
    }
}

/// Segmented scan: prefix sum with segment boundaries
/// segments[i] = true indicates start of new segment
/// Time: O(n) | Space: O(n)
pub fn segmentedScan(comptime T: type, input: []const T, segments: []const bool, allocator: Allocator) ![]T {
    if (input.len == 0) return try allocator.alloc(T, 0);
    if (input.len != segments.len) return error.DimensionMismatch;

    const output = try allocator.alloc(T, input.len);
    errdefer allocator.free(output);

    output[0] = input[0];
    for (1..input.len) |i| {
        if (segments[i]) {
            output[i] = input[i]; // Start new segment
        } else {
            output[i] = output[i - 1] + input[i];
        }
    }

    return output;
}

/// Sum reduction using prefix sum
/// Returns total sum of all elements
/// Time: O(n) | Space: O(n)
pub fn reduce(comptime T: type, input: []const T, allocator: Allocator) !T {
    if (input.len == 0) return 0;

    const scan = try inclusiveScan(T, input, allocator);
    defer allocator.free(scan);

    return scan[scan.len - 1];
}

// Tests

test "inclusiveScan basic example" {
    const input = [_]i32{ 3, 1, 7, 0, 4 };
    const expected = [_]i32{ 3, 4, 11, 11, 15 };

    const result = try inclusiveScan(i32, &input, testing.allocator);
    defer testing.allocator.free(result);

    try testing.expectEqualSlices(i32, &expected, result);
}

test "inclusiveScan empty array" {
    const input = [_]i32{};
    const result = try inclusiveScan(i32, &input, testing.allocator);
    defer testing.allocator.free(result);

    try testing.expectEqual(@as(usize, 0), result.len);
}

test "inclusiveScan single element" {
    const input = [_]i32{42};
    const expected = [_]i32{42};

    const result = try inclusiveScan(i32, &input, testing.allocator);
    defer testing.allocator.free(result);

    try testing.expectEqualSlices(i32, &expected, result);
}

test "inclusiveScan all zeros" {
    const input = [_]i32{ 0, 0, 0, 0, 0 };
    const expected = [_]i32{ 0, 0, 0, 0, 0 };

    const result = try inclusiveScan(i32, &input, testing.allocator);
    defer testing.allocator.free(result);

    try testing.expectEqualSlices(i32, &expected, result);
}

test "inclusiveScan all ones" {
    const input = [_]i32{ 1, 1, 1, 1, 1 };
    const expected = [_]i32{ 1, 2, 3, 4, 5 };

    const result = try inclusiveScan(i32, &input, testing.allocator);
    defer testing.allocator.free(result);

    try testing.expectEqualSlices(i32, &expected, result);
}

test "inclusiveScan negative numbers" {
    const input = [_]i32{ -3, 5, -2, 8, -1 };
    const expected = [_]i32{ -3, 2, 0, 8, 7 };

    const result = try inclusiveScan(i32, &input, testing.allocator);
    defer testing.allocator.free(result);

    try testing.expectEqualSlices(i32, &expected, result);
}

test "exclusiveScan basic example" {
    const input = [_]i32{ 3, 1, 7, 0, 4 };
    const expected = [_]i32{ 0, 3, 4, 11, 11 };

    const result = try exclusiveScan(i32, &input, testing.allocator);
    defer testing.allocator.free(result);

    try testing.expectEqualSlices(i32, &expected, result);
}

test "exclusiveScan empty array" {
    const input = [_]i32{};
    const result = try exclusiveScan(i32, &input, testing.allocator);
    defer testing.allocator.free(result);

    try testing.expectEqual(@as(usize, 0), result.len);
}

test "exclusiveScan single element" {
    const input = [_]i32{42};
    const expected = [_]i32{0};

    const result = try exclusiveScan(i32, &input, testing.allocator);
    defer testing.allocator.free(result);

    try testing.expectEqualSlices(i32, &expected, result);
}

test "exclusiveScan first element is zero" {
    const input = [_]i32{ 10, 20, 30 };

    const result = try exclusiveScan(i32, &input, testing.allocator);
    defer testing.allocator.free(result);

    try testing.expectEqual(@as(i32, 0), result[0]);
}

test "parallelScan basic example" {
    const input = [_]i32{ 3, 1, 7, 0, 4, 2, 5, 1 };
    const expected = [_]i32{ 0, 3, 4, 11, 11, 15, 17, 22 };

    const result = try parallelScan(i32, &input, testing.allocator, 2);
    defer testing.allocator.free(result);

    try testing.expectEqualSlices(i32, &expected, result);
}

test "parallelScan power of two length" {
    const input = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    // Exclusive scan: [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120]
    const expected = [_]i32{ 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120 };

    const result = try parallelScan(i32, &input, testing.allocator, 4);
    defer testing.allocator.free(result);

    try testing.expectEqualSlices(i32, &expected, result);
}

test "parallelScan matches sequential scan" {
    const input = [_]i32{ 5, 2, 8, 1, 9, 3, 7, 4 };

    const sequential = try exclusiveScan(i32, &input, testing.allocator);
    defer testing.allocator.free(sequential);

    const parallel = try parallelScan(i32, &input, testing.allocator, 2);
    defer testing.allocator.free(parallel);

    try testing.expectEqualSlices(i32, sequential, parallel);
}

test "parallelScan small input falls back to sequential" {
    const input = [_]i32{ 1, 2, 3 };

    const result = try parallelScan(i32, &input, testing.allocator, 4);
    defer testing.allocator.free(result);

    // Should produce same result as sequential
    const expected = [_]i32{ 0, 1, 3 };
    try testing.expectEqualSlices(i32, &expected, result);
}

test "scanInPlace basic example" {
    var data = [_]i32{ 3, 1, 7, 0, 4 };
    const expected = [_]i32{ 3, 4, 11, 11, 15 };

    scanInPlace(i32, &data);

    try testing.expectEqualSlices(i32, &expected, &data);
}

test "scanInPlace empty array" {
    var data = [_]i32{};
    scanInPlace(i32, &data);
    try testing.expectEqual(@as(usize, 0), data.len);
}

test "scanInPlace single element" {
    var data = [_]i32{42};
    scanInPlace(i32, &data);
    try testing.expectEqual(@as(i32, 42), data[0]);
}

test "segmentedScan basic example" {
    const input = [_]i32{ 3, 1, 7, 0, 4, 2, 5, 1 };
    const segments = [_]bool{ true, false, false, true, false, true, false, false };
    // Segments: [3,1,7] [0,4] [2,5,1]
    const expected = [_]i32{ 3, 4, 11, 0, 4, 2, 7, 8 };

    const result = try segmentedScan(i32, &input, &segments, testing.allocator);
    defer testing.allocator.free(result);

    try testing.expectEqualSlices(i32, &expected, result);
}

test "segmentedScan single segment" {
    const input = [_]i32{ 1, 2, 3, 4, 5 };
    const segments = [_]bool{ true, false, false, false, false };

    const result = try segmentedScan(i32, &input, &segments, testing.allocator);
    defer testing.allocator.free(result);

    // Should behave like regular inclusive scan
    const expected = [_]i32{ 1, 3, 6, 10, 15 };
    try testing.expectEqualSlices(i32, &expected, result);
}

test "segmentedScan all segments" {
    const input = [_]i32{ 5, 3, 8, 2 };
    const segments = [_]bool{ true, true, true, true };

    const result = try segmentedScan(i32, &input, &segments, testing.allocator);
    defer testing.allocator.free(result);

    // Each element is its own segment
    try testing.expectEqualSlices(i32, &input, result);
}

test "segmentedScan dimension mismatch error" {
    const input = [_]i32{ 1, 2, 3 };
    const segments = [_]bool{ true, false };

    const result = segmentedScan(i32, &input, &segments, testing.allocator);
    try testing.expectError(error.DimensionMismatch, result);
}

test "reduce basic example" {
    const input = [_]i32{ 3, 1, 7, 0, 4 };

    const sum = try reduce(i32, &input, testing.allocator);

    try testing.expectEqual(@as(i32, 15), sum);
}

test "reduce empty array" {
    const input = [_]i32{};

    const sum = try reduce(i32, &input, testing.allocator);

    try testing.expectEqual(@as(i32, 0), sum);
}

test "reduce single element" {
    const input = [_]i32{42};

    const sum = try reduce(i32, &input, testing.allocator);

    try testing.expectEqual(@as(i32, 42), sum);
}

test "reduce negative numbers" {
    const input = [_]i32{ -5, 10, -3, 8 };

    const sum = try reduce(i32, &input, testing.allocator);

    try testing.expectEqual(@as(i32, 10), sum);
}

test "prefix sum with u32 type" {
    const input = [_]u32{ 1, 2, 3, 4, 5 };
    const expected = [_]u32{ 1, 3, 6, 10, 15 };

    const result = try inclusiveScan(u32, &input, testing.allocator);
    defer testing.allocator.free(result);

    try testing.expectEqualSlices(u32, &expected, result);
}

test "prefix sum with u64 type" {
    const input = [_]u64{ 100, 200, 300 };
    const expected = [_]u64{ 100, 300, 600 };

    const result = try inclusiveScan(u64, &input, testing.allocator);
    defer testing.allocator.free(result);

    try testing.expectEqualSlices(u64, &expected, result);
}

test "prefix sum large array" {
    const allocator = testing.allocator;

    // Create array [1, 2, 3, ..., 100]
    var input = try allocator.alloc(i32, 100);
    defer allocator.free(input);
    for (0..100) |i| {
        input[i] = @intCast(i + 1);
    }

    const result = try inclusiveScan(i32, input, allocator);
    defer allocator.free(result);

    // Sum of 1..100 = 100*101/2 = 5050
    try testing.expectEqual(@as(i32, 5050), result[99]);
    // Check a few intermediate values
    try testing.expectEqual(@as(i32, 1), result[0]);
    try testing.expectEqual(@as(i32, 3), result[1]); // 1+2
    try testing.expectEqual(@as(i32, 55), result[9]); // sum(1..10)
}

test "memory safety: prefix sum allocator check" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const input = [_]i32{ 1, 2, 3, 4, 5 };

        const result1 = try inclusiveScan(i32, &input, allocator);
        allocator.free(result1);

        const result2 = try exclusiveScan(i32, &input, allocator);
        allocator.free(result2);

        const result3 = try parallelScan(i32, &input, allocator, 2);
        allocator.free(result3);
    }
}
