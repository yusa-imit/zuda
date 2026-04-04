const std = @import("std");
const testing = std.testing;

/// Longest Bitonic Subsequence (LBS)
///
/// Finds the length of the longest subsequence that is first strictly increasing
/// and then strictly decreasing. A sequence is bitonic if it monotonically increases
/// and then monotonically decreases.
///
/// Algorithm:
/// 1. Compute LIS ending at each position (left-to-right)
/// 2. Compute LDS starting at each position (right-to-left)
/// 3. Maximum of (LIS[i] + LDS[i] - 1) gives the LBS
///
/// Time complexity: O(n²) for basic DP, O(n log n) with binary search optimization
/// Space complexity: O(n) for auxiliary arrays
///
/// Use cases:
/// - Stock price analysis (longest rise-then-fall pattern)
/// - Signal processing (peak detection in time series)
/// - Trajectory analysis (projectile motion patterns)
/// - Pattern recognition (V-shaped or inverted-V patterns)
///
/// Example:
/// ```zig
/// const arr = [_]i32{ 1, 11, 2, 10, 4, 5, 2, 1 };
/// const len = longestBitonicSubsequence(i32, &arr);
/// // Returns 6: subsequence is {1, 2, 10, 4, 2, 1}
/// ```

/// Finds the length of the longest bitonic subsequence
///
/// A bitonic sequence first increases then decreases. Both parts must be non-empty.
///
/// Time: O(n²)
/// Space: O(n)
pub fn longestBitonicSubsequence(comptime T: type, arr: []const T) !usize {
    if (arr.len == 0) return 0;
    if (arr.len == 1) return 1;

    const n = arr.len;

    // Allocate LIS and LDS arrays on stack (reasonable for typical inputs)
    // For very large inputs, caller should use longestBitonicSubsequenceAlloc
    var lis_buf: [1024]usize = undefined;
    var lds_buf: [1024]usize = undefined;

    if (n > 1024) {
        return error.ArrayTooLarge; // Use longestBitonicSubsequenceAlloc for larger arrays
    }

    const lis = lis_buf[0..n];
    const lds = lds_buf[0..n];

    // Initialize LIS (all elements are subsequences of length 1)
    for (lis) |*val| val.* = 1;

    // Compute LIS ending at each position (left-to-right)
    for (1..n) |i| {
        for (0..i) |j| {
            if (arr[j] < arr[i] and lis[j] + 1 > lis[i]) {
                lis[i] = lis[j] + 1;
            }
        }
    }

    // Initialize LDS (all elements are subsequences of length 1)
    for (lds) |*val| val.* = 1;

    // Compute LDS starting at each position (right-to-left)
    var i: usize = n - 1;
    while (i > 0) : (i -= 1) {
        for (i + 1..n) |j| {
            if (arr[i] > arr[j] and lds[j] + 1 > lds[i]) {
                lds[i] = lds[j] + 1;
            }
        }
    }

    // Find maximum LIS[i] + LDS[i] - 1
    // We need both LIS[i] > 1 and LDS[i] > 1 for a valid bitonic sequence
    var max_len: usize = 0;
    for (0..n) |idx| {
        if (lis[idx] > 1 and lds[idx] > 1) {
            const bitonic_len = lis[idx] + lds[idx] - 1;
            if (bitonic_len > max_len) {
                max_len = bitonic_len;
            }
        }
    }

    return max_len;
}

/// Finds the length of the longest bitonic subsequence with allocator
///
/// This version uses heap allocation and is suitable for large arrays.
///
/// Time: O(n²)
/// Space: O(n)
pub fn longestBitonicSubsequenceAlloc(comptime T: type, arr: []const T, allocator: std.mem.Allocator) !usize {
    if (arr.len == 0) return 0;
    if (arr.len == 1) return 1;

    const n = arr.len;

    const lis = try allocator.alloc(usize, n);
    defer allocator.free(lis);
    const lds = try allocator.alloc(usize, n);
    defer allocator.free(lds);

    // Initialize LIS
    for (lis) |*val| val.* = 1;

    // Compute LIS ending at each position
    for (1..n) |i| {
        for (0..i) |j| {
            if (arr[j] < arr[i] and lis[j] + 1 > lis[i]) {
                lis[i] = lis[j] + 1;
            }
        }
    }

    // Initialize LDS
    for (lds) |*val| val.* = 1;

    // Compute LDS starting at each position
    var i: usize = n - 1;
    while (i > 0) : (i -= 1) {
        for (i + 1..n) |j| {
            if (arr[i] > arr[j] and lds[j] + 1 > lds[i]) {
                lds[i] = lds[j] + 1;
            }
        }
    }

    // Find maximum
    var max_len: usize = 0;
    for (0..n) |idx| {
        if (lis[idx] > 1 and lds[idx] > 1) {
            const bitonic_len = lis[idx] + lds[idx] - 1;
            if (bitonic_len > max_len) {
                max_len = bitonic_len;
            }
        }
    }

    return max_len;
}

/// Result structure containing the actual bitonic subsequence
pub const BitonicResult = struct {
    length: usize,
    sequence: []usize, // Indices in the original array

    pub fn deinit(self: *BitonicResult, allocator: std.mem.Allocator) void {
        allocator.free(self.sequence);
    }
};

/// Finds the longest bitonic subsequence and returns the actual indices
///
/// Time: O(n²)
/// Space: O(n)
pub fn longestBitonicSubsequenceWithPath(comptime T: type, arr: []const T, allocator: std.mem.Allocator) !BitonicResult {
    if (arr.len == 0) return BitonicResult{ .length = 0, .sequence = &[_]usize{} };
    if (arr.len == 1) {
        const seq = try allocator.alloc(usize, 1);
        seq[0] = 0;
        return BitonicResult{ .length = 1, .sequence = seq };
    }

    const n = arr.len;

    const lis = try allocator.alloc(usize, n);
    defer allocator.free(lis);
    const lds = try allocator.alloc(usize, n);
    defer allocator.free(lds);

    const lis_prev = try allocator.alloc(isize, n);
    defer allocator.free(lis_prev);
    const lds_next = try allocator.alloc(isize, n);
    defer allocator.free(lds_next);

    // Initialize
    for (lis) |*val| val.* = 1;
    for (lds) |*val| val.* = 1;
    for (lis_prev) |*val| val.* = -1;
    for (lds_next) |*val| val.* = -1;

    // Compute LIS with predecessor tracking
    for (1..n) |i| {
        for (0..i) |j| {
            if (arr[j] < arr[i] and lis[j] + 1 > lis[i]) {
                lis[i] = lis[j] + 1;
                lis_prev[i] = @intCast(j);
            }
        }
    }

    // Compute LDS with successor tracking
    var i: usize = n - 1;
    while (i > 0) : (i -= 1) {
        for (i + 1..n) |j| {
            if (arr[i] > arr[j] and lds[j] + 1 > lds[i]) {
                lds[i] = lds[j] + 1;
                lds_next[i] = @intCast(j);
            }
        }
    }

    // Find peak of longest bitonic sequence
    var max_len: usize = 0;
    var peak_idx: usize = 0;
    for (0..n) |idx| {
        if (lis[idx] > 1 and lds[idx] > 1) {
            const bitonic_len = lis[idx] + lds[idx] - 1;
            if (bitonic_len > max_len) {
                max_len = bitonic_len;
                peak_idx = idx;
            }
        }
    }

    if (max_len == 0) {
        return BitonicResult{ .length = 0, .sequence = &[_]usize{} };
    }

    // Reconstruct path
    var path = try std.ArrayList(usize).initCapacity(allocator, max_len);
    errdefer path.deinit(allocator);

    // Trace back increasing part
    var curr: isize = @intCast(peak_idx);
    var inc_part: std.ArrayList(usize) = .{};
    defer inc_part.deinit(allocator);

    while (curr != -1) {
        try inc_part.append(allocator, @intCast(curr));
        curr = lis_prev[@intCast(curr)];
    }

    // Add increasing part (reversed)
    var k: usize = inc_part.items.len;
    while (k > 0) : (k -= 1) {
        try path.append(allocator, inc_part.items[k - 1]);
    }

    // Trace forward decreasing part (skip peak as it's already added)
    curr = lds_next[peak_idx];
    while (curr != -1) {
        try path.append(allocator, @intCast(curr));
        curr = lds_next[@intCast(curr)];
    }

    return BitonicResult{
        .length = max_len,
        .sequence = try path.toOwnedSlice(allocator),
    };
}

// Tests

test "bitonic subsequence - basic example" {
    const arr = [_]i32{ 1, 11, 2, 10, 4, 5, 2, 1 };
    const len = try longestBitonicSubsequence(i32, &arr);
    try testing.expectEqual(@as(usize, 6), len);
}

test "bitonic subsequence - strictly increasing then decreasing" {
    const arr = [_]i32{ 1, 2, 3, 4, 3, 2, 1 };
    const len = try longestBitonicSubsequence(i32, &arr);
    try testing.expectEqual(@as(usize, 7), len);
}

test "bitonic subsequence - single peak" {
    const arr = [_]i32{ 1, 2, 5, 3, 2 };
    const len = try longestBitonicSubsequence(i32, &arr);
    try testing.expectEqual(@as(usize, 5), len);
}

test "bitonic subsequence - no valid bitonic (monotonic increasing)" {
    const arr = [_]i32{ 1, 2, 3, 4, 5 };
    const len = try longestBitonicSubsequence(i32, &arr);
    try testing.expectEqual(@as(usize, 0), len); // No valid bitonic (needs both increase and decrease)
}

test "bitonic subsequence - no valid bitonic (monotonic decreasing)" {
    const arr = [_]i32{ 5, 4, 3, 2, 1 };
    const len = try longestBitonicSubsequence(i32, &arr);
    try testing.expectEqual(@as(usize, 0), len);
}

test "bitonic subsequence - multiple peaks" {
    const arr = [_]i32{ 1, 3, 2, 4, 3, 1 };
    const len = try longestBitonicSubsequence(i32, &arr);
    try testing.expectEqual(@as(usize, 5), len); // {1, 3, 4, 3, 1} or {1, 2, 4, 3, 1}
}

test "bitonic subsequence - empty array" {
    const arr = [_]i32{};
    const len = try longestBitonicSubsequence(i32, &arr);
    try testing.expectEqual(@as(usize, 0), len);
}

test "bitonic subsequence - single element" {
    const arr = [_]i32{42};
    const len = try longestBitonicSubsequence(i32, &arr);
    try testing.expectEqual(@as(usize, 1), len);
}

test "bitonic subsequence - two elements ascending" {
    const arr = [_]i32{ 1, 2 };
    const len = try longestBitonicSubsequence(i32, &arr);
    try testing.expectEqual(@as(usize, 0), len); // Needs both increase and decrease
}

test "bitonic subsequence - two elements descending" {
    const arr = [_]i32{ 2, 1 };
    const len = try longestBitonicSubsequence(i32, &arr);
    try testing.expectEqual(@as(usize, 0), len);
}

test "bitonic subsequence - three elements peak" {
    const arr = [_]i32{ 1, 3, 2 };
    const len = try longestBitonicSubsequence(i32, &arr);
    try testing.expectEqual(@as(usize, 3), len);
}

test "bitonic subsequence - all equal elements" {
    const arr = [_]i32{ 5, 5, 5, 5 };
    const len = try longestBitonicSubsequence(i32, &arr);
    try testing.expectEqual(@as(usize, 0), len); // No strictly increasing/decreasing
}

test "bitonic subsequence - negative numbers" {
    const arr = [_]i32{ -3, -1, 4, 2, -2 };
    const len = try longestBitonicSubsequence(i32, &arr);
    try testing.expectEqual(@as(usize, 5), len);
}

test "bitonic subsequence - allocator version" {
    const arr = [_]i32{ 1, 11, 2, 10, 4, 5, 2, 1 };
    const len = try longestBitonicSubsequenceAlloc(i32, &arr, testing.allocator);
    try testing.expectEqual(@as(usize, 6), len);
}

test "bitonic subsequence - with path reconstruction" {
    const arr = [_]i32{ 1, 11, 2, 10, 4, 5, 2, 1 };
    var result = try longestBitonicSubsequenceWithPath(i32, &arr, testing.allocator);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 6), result.length);
    try testing.expect(result.sequence.len == 6);

    // Verify the subsequence is bitonic
    var prev = arr[result.sequence[0]];
    var increasing = true;
    var has_peak = false;

    for (result.sequence[1..]) |idx| {
        const curr = arr[idx];
        if (increasing) {
            if (curr < prev) {
                increasing = false;
                has_peak = true;
            }
            try testing.expect(curr != prev); // Strictly increasing or peak
        } else {
            try testing.expect(curr < prev); // Strictly decreasing after peak
        }
        prev = curr;
    }

    try testing.expect(has_peak); // Must have a peak
}

test "bitonic subsequence - large array" {
    var arr: [100]i32 = undefined;

    // Create a bitonic pattern: 0..49 increasing, 50..99 decreasing
    for (0..50) |i| {
        arr[i] = @intCast(i);
    }
    for (50..100) |i| {
        arr[i] = @intCast(149 - i);
    }

    const len = try longestBitonicSubsequenceAlloc(i32, &arr, testing.allocator);
    try testing.expectEqual(@as(usize, 100), len);
}

test "bitonic subsequence - floating point" {
    const arr = [_]f64{ 1.5, 3.2, 2.1, 4.8, 3.9, 2.5, 1.0 };
    const len = try longestBitonicSubsequence(f64, &arr);
    try testing.expectEqual(@as(usize, 6), len); // {1.5, 3.2, 4.8, 3.9, 2.5, 1.0}
}

test "bitonic subsequence - memory safety" {
    const arr = [_]i32{ 1, 11, 2, 10, 4, 5, 2, 1 };
    _ = try longestBitonicSubsequenceAlloc(i32, &arr, testing.allocator);

    var result = try longestBitonicSubsequenceWithPath(i32, &arr, testing.allocator);
    result.deinit(testing.allocator);
}
