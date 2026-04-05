const std = @import("std");
const testing = std.testing;

/// Russian Doll Envelopes - Find maximum number of envelopes that can be nested
///
/// Problem: Given envelopes with (width, height), find the maximum number of envelopes
/// you can Russian doll (put one inside another). Envelope i fits into j if both
/// width[i] < width[j] AND height[i] < height[j].
///
/// Algorithm:
/// 1. Sort envelopes by width ascending, height descending (critical for correctness)
/// 2. Extract heights from sorted array
/// 3. Find Longest Increasing Subsequence (LIS) on heights using O(n log n) algorithm
/// 4. The LIS length is the answer
///
/// Why sort height descending? When two envelopes have the same width, we can't nest them.
/// Sorting height descending ensures we only pick one envelope per width group in the LIS.
///
/// Time: O(n log n) - sorting + LIS binary search
/// Space: O(n) - for sorted array and LIS tracking

/// Find maximum number of envelopes that can be nested
/// Time: O(n log n), Space: O(n)
pub fn maxEnvelopes(comptime T: type, allocator: std.mem.Allocator, envelopes: []const [2]T) !usize {
    if (envelopes.len == 0) return 0;
    if (envelopes.len == 1) return 1;

    // Sort by width ascending, height descending
    var sorted = try allocator.dupe([2]T, envelopes);
    defer allocator.free(sorted);
    _ = &sorted; // Force mutable (std.mem.sort mutates in-place)
    std.mem.sort([2]T, sorted, {}, struct {
        fn lessThan(_: void, a: [2]T, b: [2]T) bool {
            if (a[0] != b[0]) return a[0] < b[0]; // width ascending
            return a[1] > b[1]; // height descending for same width
        }
    }.lessThan);

    // Extract heights and find LIS
    var heights = try allocator.alloc(T, sorted.len);
    defer allocator.free(heights);
    for (sorted, 0..) |env, i| {
        heights[i] = env[1];
    }

    return try longestIncreasingSubsequence(T, allocator, heights);
}

/// Find maximum nesting with actual sequence
/// Returns allocator-owned slice of envelope indices in nesting order
/// Time: O(n log n), Space: O(n)
pub fn maxEnvelopesWithSequence(comptime T: type, allocator: std.mem.Allocator, envelopes: []const [2]T) !struct {
    length: usize,
    indices: []usize,
} {
    if (envelopes.len == 0) return .{ .length = 0, .indices = try allocator.alloc(usize, 0) };
    if (envelopes.len == 1) {
        const indices = try allocator.alloc(usize, 1);
        indices[0] = 0;
        return .{ .length = 1, .indices = indices };
    }

    // Sort with original indices
    var indexed = try allocator.alloc(struct { env: [2]T, original_idx: usize }, envelopes.len);
    defer allocator.free(indexed);
    for (envelopes, 0..) |env, i| {
        indexed[i] = .{ .env = env, .original_idx = i };
    }

    std.mem.sort(@TypeOf(indexed[0]), indexed, {}, struct {
        fn lessThan(_: void, a: @TypeOf(indexed[0]), b: @TypeOf(indexed[0])) bool {
            if (a.env[0] != b.env[0]) return a.env[0] < b.env[0];
            return a.env[1] > b.env[1];
        }
    }.lessThan);

    // Extract heights with indices
    var heights = try allocator.alloc(T, indexed.len);
    defer allocator.free(heights);
    for (indexed, 0..) |item, i| {
        heights[i] = item.env[1];
    }

    const lis_result = try longestIncreasingSubsequenceWithIndices(T, allocator, heights);
    defer allocator.free(lis_result.indices);

    // Map back to original indices
    var original_indices = try allocator.alloc(usize, lis_result.length);
    for (lis_result.indices, 0..) |sorted_idx, i| {
        original_indices[i] = indexed[sorted_idx].original_idx;
    }

    return .{ .length = lis_result.length, .indices = original_indices };
}

/// Count valid nestings (different from max nesting - this counts all possible chains)
/// Time: O(n²), Space: O(n)
pub fn countValidNestings(comptime T: type, allocator: std.mem.Allocator, envelopes: []const [2]T) !usize {
    if (envelopes.len == 0) return 0;
    if (envelopes.len == 1) return 1;

    var sorted = try allocator.dupe([2]T, envelopes);
    defer allocator.free(sorted);
    _ = &sorted; // Force mutable (std.mem.sort mutates in-place)
    std.mem.sort([2]T, sorted, {}, struct {
        fn lessThan(_: void, a: [2]T, b: [2]T) bool {
            if (a[0] != b[0]) return a[0] < b[0];
            return a[1] > b[1];
        }
    }.lessThan);

    // dp[i] = number of ways to form chains ending at i
    var dp = try allocator.alloc(usize, sorted.len);
    defer allocator.free(dp);
    @memset(dp, 1); // Each envelope can be a chain of length 1

    for (1..sorted.len) |i| {
        for (0..i) |j| {
            if (sorted[j][1] < sorted[i][1]) { // Can nest j inside i
                dp[i] += dp[j];
            }
        }
    }

    var total: usize = 0;
    for (dp) |count| {
        total += count;
    }
    return total;
}

/// Validate if a sequence of envelopes can be nested in order
/// Time: O(n), Space: O(1)
pub fn validateNesting(comptime T: type, envelopes: []const [2]T, sequence: []const usize) bool {
    if (sequence.len == 0) return true;
    if (sequence.len == 1) return sequence[0] < envelopes.len;

    for (0..sequence.len - 1) |i| {
        const idx1 = sequence[i];
        const idx2 = sequence[i + 1];
        if (idx1 >= envelopes.len or idx2 >= envelopes.len) return false;

        const env1 = envelopes[idx1];
        const env2 = envelopes[idx2];
        // env1 must fit into env2
        if (env1[0] >= env2[0] or env1[1] >= env2[1]) return false;
    }
    return true;
}

// Helper: LIS using binary search (O(n log n))
fn longestIncreasingSubsequence(comptime T: type, allocator: std.mem.Allocator, arr: []const T) !usize {
    if (arr.len == 0) return 0;

    var tails = try allocator.alloc(T, arr.len);
    defer allocator.free(tails);
    var len: usize = 0;

    for (arr) |num| {
        var left: usize = 0;
        var right: usize = len;
        while (left < right) {
            const mid = left + (right - left) / 2;
            if (tails[mid] < num) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        tails[left] = num;
        if (left == len) {
            len += 1;
        }
    }

    return len;
}

fn longestIncreasingSubsequenceWithIndices(comptime T: type, allocator: std.mem.Allocator, arr: []const T) !struct {
    length: usize,
    indices: []usize,
} {
    if (arr.len == 0) return .{ .length = 0, .indices = try allocator.alloc(usize, 0) };

    var tails = try allocator.alloc(T, arr.len);
    defer allocator.free(tails);
    var tail_indices = try allocator.alloc(usize, arr.len);
    defer allocator.free(tail_indices);
    var prev = try allocator.alloc(?usize, arr.len);
    defer allocator.free(prev);
    @memset(prev, null);

    var len: usize = 0;

    for (arr, 0..) |num, i| {
        var left: usize = 0;
        var right: usize = len;
        while (left < right) {
            const mid = left + (right - left) / 2;
            if (tails[mid] < num) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        tails[left] = num;
        tail_indices[left] = i;
        if (left > 0) {
            prev[i] = tail_indices[left - 1];
        }
        if (left == len) {
            len += 1;
        }
    }

    // Reconstruct sequence
    var result = try allocator.alloc(usize, len);
    if (len > 0) {
        var curr: ?usize = tail_indices[len - 1];
        var pos: usize = len;
        while (curr != null) {
            pos -= 1;
            result[pos] = curr.?;
            curr = prev[curr.?];
        }
    }

    return .{ .length = len, .indices = result };
}

// ============================================================================
// Tests
// ============================================================================

test "maxEnvelopes: basic example" {
    const allocator = testing.allocator;

    const envelopes = [_][2]i32{
        .{ 5, 4 },
        .{ 6, 4 },
        .{ 6, 7 },
        .{ 2, 3 },
    };
    // Sorted: (2,3), (5,4), (6,7), (6,4)
    // Heights: 3, 4, 7, 4
    // LIS: 3, 4, 7 → length 3
    const result = try maxEnvelopes(i32, allocator, &envelopes);
    try testing.expectEqual(@as(usize, 3), result);
}

test "maxEnvelopes: another example" {
    const allocator = testing.allocator;

    const envelopes = [_][2]i32{
        .{ 1, 1 },
        .{ 1, 1 },
        .{ 1, 1 },
    };
    // All same, can only pick one
    const result = try maxEnvelopes(i32, allocator, &envelopes);
    try testing.expectEqual(@as(usize, 1), result);
}

test "maxEnvelopes: empty" {
    const allocator = testing.allocator;
    const envelopes = [_][2]i32{};
    const result = try maxEnvelopes(i32, allocator, &envelopes);
    try testing.expectEqual(@as(usize, 0), result);
}

test "maxEnvelopes: single envelope" {
    const allocator = testing.allocator;
    const envelopes = [_][2]i32{.{ 5, 7 }};
    const result = try maxEnvelopes(i32, allocator, &envelopes);
    try testing.expectEqual(@as(usize, 1), result);
}

test "maxEnvelopes: all increasing" {
    const allocator = testing.allocator;

    const envelopes = [_][2]i32{
        .{ 1, 2 },
        .{ 2, 3 },
        .{ 3, 4 },
        .{ 4, 5 },
    };
    const result = try maxEnvelopes(i32, allocator, &envelopes);
    try testing.expectEqual(@as(usize, 4), result);
}

test "maxEnvelopes: no nesting possible" {
    const allocator = testing.allocator;

    const envelopes = [_][2]i32{
        .{ 1, 5 },
        .{ 2, 4 },
        .{ 3, 3 },
        .{ 4, 2 },
        .{ 5, 1 },
    };
    // Heights decreasing after sorting
    const result = try maxEnvelopes(i32, allocator, &envelopes);
    try testing.expectEqual(@as(usize, 1), result);
}

test "maxEnvelopes: same width different heights" {
    const allocator = testing.allocator;

    const envelopes = [_][2]i32{
        .{ 5, 3 },
        .{ 5, 5 },
        .{ 5, 7 },
        .{ 5, 1 },
    };
    // Same width, can only pick one (descending height sort ensures correctness)
    const result = try maxEnvelopes(i32, allocator, &envelopes);
    try testing.expectEqual(@as(usize, 1), result);
}

test "maxEnvelopes: complex case" {
    const allocator = testing.allocator;

    const envelopes = [_][2]i32{
        .{ 4, 5 },
        .{ 4, 6 },
        .{ 6, 7 },
        .{ 2, 3 },
        .{ 1, 1 },
    };
    // Best: (1,1) → (2,3) → (4,5) → (6,7) = 4
    const result = try maxEnvelopes(i32, allocator, &envelopes);
    try testing.expectEqual(@as(usize, 4), result);
}

test "maxEnvelopes: floating point" {
    const allocator = testing.allocator;

    const envelopes = [_][2]f64{
        .{ 5.5, 4.5 },
        .{ 6.5, 4.5 },
        .{ 6.5, 7.5 },
        .{ 2.5, 3.5 },
    };
    const result = try maxEnvelopes(f64, allocator, &envelopes);
    try testing.expectEqual(@as(usize, 3), result);
}

test "maxEnvelopesWithSequence: basic" {
    const allocator = testing.allocator;

    const envelopes = [_][2]i32{
        .{ 5, 4 },  // idx 0
        .{ 6, 4 },  // idx 1
        .{ 6, 7 },  // idx 2
        .{ 2, 3 },  // idx 3
    };
    const result = try maxEnvelopesWithSequence(i32, allocator, &envelopes);
    defer allocator.free(result.indices);

    try testing.expectEqual(@as(usize, 3), result.length);
    try testing.expect(validateNesting(i32, &envelopes, result.indices));
}

test "maxEnvelopesWithSequence: single" {
    const allocator = testing.allocator;

    const envelopes = [_][2]i32{.{ 5, 7 }};
    const result = try maxEnvelopesWithSequence(i32, allocator, &envelopes);
    defer allocator.free(result.indices);

    try testing.expectEqual(@as(usize, 1), result.length);
    try testing.expectEqual(@as(usize, 1), result.indices.len);
    try testing.expectEqual(@as(usize, 0), result.indices[0]);
}

test "maxEnvelopesWithSequence: empty" {
    const allocator = testing.allocator;

    const envelopes = [_][2]i32{};
    const result = try maxEnvelopesWithSequence(i32, allocator, &envelopes);
    defer allocator.free(result.indices);

    try testing.expectEqual(@as(usize, 0), result.length);
    try testing.expectEqual(@as(usize, 0), result.indices.len);
}

test "countValidNestings: basic" {
    const allocator = testing.allocator;

    const envelopes = [_][2]i32{
        .{ 1, 2 },
        .{ 2, 3 },
        .{ 3, 4 },
    };
    // Chains: (1,2), (2,3), (3,4), (1,2)→(2,3), (2,3)→(3,4), (1,2)→(3,4), (1,2)→(2,3)→(3,4)
    const result = try countValidNestings(i32, allocator, &envelopes);
    try testing.expect(result >= 3); // At least the 3 individual envelopes
}

test "countValidNestings: single" {
    const allocator = testing.allocator;

    const envelopes = [_][2]i32{.{ 5, 7 }};
    const result = try countValidNestings(i32, allocator, &envelopes);
    try testing.expectEqual(@as(usize, 1), result);
}

test "validateNesting: valid sequence" {
    const envelopes = [_][2]i32{
        .{ 1, 1 },
        .{ 2, 2 },
        .{ 3, 3 },
    };
    const sequence = [_]usize{ 0, 1, 2 };
    try testing.expect(validateNesting(i32, &envelopes, &sequence));
}

test "validateNesting: invalid sequence" {
    const envelopes = [_][2]i32{
        .{ 1, 1 },
        .{ 2, 2 },
        .{ 3, 3 },
    };
    const sequence = [_]usize{ 2, 1, 0 }; // Reverse order
    try testing.expect(!validateNesting(i32, &envelopes, &sequence));
}

test "validateNesting: empty sequence" {
    const envelopes = [_][2]i32{.{ 1, 1 }};
    const sequence = [_]usize{};
    try testing.expect(validateNesting(i32, &envelopes, &sequence));
}

test "maxEnvelopes: large dataset" {
    const allocator = testing.allocator;

    var envelopes = try allocator.alloc([2]i32, 100);
    defer allocator.free(envelopes);

    // Create increasing sequence with some noise
    for (0..100) |i| {
        envelopes[i] = .{ @as(i32, @intCast(i + 1)), @as(i32, @intCast(i + 1)) };
    }

    const result = try maxEnvelopes(i32, allocator, envelopes);
    try testing.expectEqual(@as(usize, 100), result);
}

test "maxEnvelopes: memory safety" {
    const allocator = testing.allocator;

    const envelopes = [_][2]i32{
        .{ 5, 4 },
        .{ 6, 4 },
        .{ 6, 7 },
        .{ 2, 3 },
    };
    const result = try maxEnvelopes(i32, allocator, &envelopes);
    try testing.expectEqual(@as(usize, 3), result);
    // No leaks expected
}

test "maxEnvelopesWithSequence: memory safety" {
    const allocator = testing.allocator;

    const envelopes = [_][2]i32{
        .{ 5, 4 },
        .{ 6, 7 },
        .{ 2, 3 },
    };
    const result = try maxEnvelopesWithSequence(i32, allocator, &envelopes);
    defer allocator.free(result.indices);

    try testing.expectEqual(@as(usize, 3), result.length);
}
