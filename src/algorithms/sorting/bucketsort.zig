const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// Bucket Sort: Distribution-based sorting for uniformly distributed data.
///
/// Distributes elements into buckets, sorts each bucket, then concatenates results.
/// Excellent for data with known uniform distribution (e.g., normalized values, probabilities).
///
/// Time: O(n + k) average case when data is uniformly distributed, O(n²) worst case
/// Space: O(n + k) where k = number of buckets
///
/// Use cases:
/// - Floating-point data in [0,1) range (probabilities, normalized values)
/// - Uniformly distributed data with known bounds
/// - External sorting with limited memory (bucket = disk file)
/// - Distributed sorting (each machine handles a bucket range)
/// - Preprocessing for other algorithms
///
/// Algorithm:
/// 1. Determine value range [min, max]
/// 2. Create k buckets covering the range
/// 3. Distribute elements into buckets based on value
/// 4. Sort each bucket individually (using insertion sort)
/// 5. Concatenate sorted buckets
///
/// Properties:
/// - Stable: Preserves relative order of equal elements (if bucket sort is stable)
/// - Not in-place: Requires O(n) auxiliary space
/// - Adaptive: Performance depends on data distribution
/// - External: Can be adapted for external sorting
///
/// References:
/// - E.J. Isaac and R.C. Singleton (1956)
/// - Knuth, "The Art of Computer Programming", Vol. 3
pub fn bucketSort(comptime T: type, allocator: Allocator, slice: []T, num_buckets: usize) !void {
    if (slice.len <= 1) return;
    if (num_buckets == 0) return error.InvalidBucketCount;

    // Find min and max values
    var min_val = slice[0];
    var max_val = slice[0];
    for (slice[1..]) |val| {
        if (lessThan(T, val, min_val)) min_val = val;
        if (lessThan(T, max_val, val)) max_val = val;
    }

    // If all values are equal, already sorted
    if (!lessThan(T, min_val, max_val) and !lessThan(T, max_val, min_val)) return;

    // Create buckets
    var buckets = try allocator.alloc(std.ArrayList(T), num_buckets);
    defer {
        for (buckets) |*bucket| bucket.deinit(allocator);
        allocator.free(buckets);
    }

    for (buckets) |*bucket| {
        bucket.* = try std.ArrayList(T).initCapacity(allocator, 0);
    }

    // Distribute elements into buckets
    const range = toFloat(T, max_val) - toFloat(T, min_val);
    for (slice) |val| {
        const normalized = (toFloat(T, val) - toFloat(T, min_val)) / range;
        const bucket_idx = @min(@as(usize, @intFromFloat(normalized * @as(f64, @floatFromInt(num_buckets)))), num_buckets - 1);
        try buckets[bucket_idx].append(allocator, val);
    }

    // Sort each bucket (using insertion sort for small buckets)
    for (buckets) |*bucket| {
        insertionSort(T, bucket.items);
    }

    // Concatenate sorted buckets
    var idx: usize = 0;
    for (buckets) |bucket| {
        for (bucket.items) |val| {
            slice[idx] = val;
            idx += 1;
        }
    }
}

/// Bucket Sort for floating-point data in [0, 1) range.
///
/// Optimized variant for normalized data (probabilities, normalized features).
/// Assumes all values are in [0, 1) range for optimal bucket distribution.
///
/// Time: O(n + k) average case
/// Space: O(n + k)
pub fn bucketSortNormalized(comptime T: type, allocator: Allocator, slice: []T, num_buckets: usize) !void {
    if (slice.len <= 1) return;
    if (num_buckets == 0) return error.InvalidBucketCount;

    comptime {
        if (T != f32 and T != f64) {
            @compileError("bucketSortNormalized requires floating-point type (f32 or f64)");
        }
    }

    // Create buckets
    var buckets = try allocator.alloc(std.ArrayList(T), num_buckets);
    defer {
        for (buckets) |*bucket| bucket.deinit(allocator);
        allocator.free(buckets);
    }

    for (buckets) |*bucket| {
        bucket.* = try std.ArrayList(T).initCapacity(allocator, 0);
    }

    // Distribute elements into buckets (values in [0, 1))
    for (slice) |val| {
        const bucket_idx = @min(@as(usize, @intFromFloat(val * @as(T, @floatFromInt(num_buckets)))), num_buckets - 1);
        try buckets[bucket_idx].append(allocator, val);
    }

    // Sort each bucket
    for (buckets) |*bucket| {
        insertionSort(T, bucket.items);
    }

    // Concatenate sorted buckets
    var idx: usize = 0;
    for (buckets) |bucket| {
        for (bucket.items) |val| {
            slice[idx] = val;
            idx += 1;
        }
    }
}

/// Bucket Sort with custom bucket mapping function.
///
/// Allows flexible bucket assignment via user-provided mapping function.
/// Useful for custom distributions or domain-specific bucket strategies.
///
/// Time: O(n + k) average case
/// Space: O(n + k)
///
/// Parameters:
/// - mapFn: Function mapping value to bucket index [0, num_buckets)
pub fn bucketSortCustom(
    comptime T: type,
    allocator: Allocator,
    slice: []T,
    num_buckets: usize,
    mapFn: fn (T, usize) usize,
) !void {
    if (slice.len <= 1) return;
    if (num_buckets == 0) return error.InvalidBucketCount;

    // Create buckets
    var buckets = try allocator.alloc(std.ArrayList(T), num_buckets);
    defer {
        for (buckets) |*bucket| bucket.deinit(allocator);
        allocator.free(buckets);
    }

    for (buckets) |*bucket| {
        bucket.* = try std.ArrayList(T).initCapacity(allocator, 0);
    }

    // Distribute elements using custom mapping
    for (slice) |val| {
        const bucket_idx = mapFn(val, num_buckets);
        if (bucket_idx >= num_buckets) return error.InvalidBucketIndex;
        try buckets[bucket_idx].append(allocator, val);
    }

    // Sort each bucket
    for (buckets) |*bucket| {
        insertionSort(T, bucket.items);
    }

    // Concatenate sorted buckets
    var idx: usize = 0;
    for (buckets) |bucket| {
        for (bucket.items) |val| {
            slice[idx] = val;
            idx += 1;
        }
    }
}

// Helper: Convert value to float for range calculation
fn toFloat(comptime T: type, val: T) f64 {
    return switch (@typeInfo(T)) {
        .int => @as(f64, @floatFromInt(val)),
        .float => @as(f64, @floatCast(val)),
        else => @compileError("Unsupported type for bucket sort"),
    };
}

// Helper: Less-than comparison for generic types
fn lessThan(comptime T: type, a: T, b: T) bool {
    return switch (@typeInfo(T)) {
        .int, .float => a < b,
        else => @compileError("Unsupported type for comparison"),
    };
}

// Helper: Insertion sort for small bucket sizes
fn insertionSort(comptime T: type, slice: []T) void {
    if (slice.len <= 1) return;

    var i: usize = 1;
    while (i < slice.len) : (i += 1) {
        const key = slice[i];
        var j: usize = i;
        while (j > 0 and lessThan(T, key, slice[j - 1])) : (j -= 1) {
            slice[j] = slice[j - 1];
        }
        slice[j] = key;
    }
}

// ============================================================================
// Tests
// ============================================================================

test "bucket sort: basic integer array" {
    var arr = [_]i32{ 29, 25, 3, 49, 9, 37, 21, 43 };
    try bucketSort(i32, testing.allocator, &arr, 5);
    try testing.expectEqualSlices(i32, &[_]i32{ 3, 9, 21, 25, 29, 37, 43, 49 }, &arr);
}

test "bucket sort: floating point array" {
    var arr = [_]f64{ 0.897, 0.565, 0.656, 0.1234, 0.665, 0.3434 };
    try bucketSort(f64, testing.allocator, &arr, 5);
    try testing.expect(arr[0] < arr[1]);
    try testing.expect(arr[1] < arr[2]);
    try testing.expect(arr[2] < arr[3]);
    try testing.expect(arr[3] < arr[4]);
    try testing.expect(arr[4] < arr[5]);
}

test "bucket sort: already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    try bucketSort(i32, testing.allocator, &arr, 3);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "bucket sort: reverse sorted" {
    var arr = [_]i32{ 5, 4, 3, 2, 1 };
    try bucketSort(i32, testing.allocator, &arr, 3);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "bucket sort: duplicates" {
    var arr = [_]i32{ 5, 2, 8, 2, 9, 1, 5, 5 };
    try bucketSort(i32, testing.allocator, &arr, 4);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 2, 5, 5, 5, 8, 9 }, &arr);
}

test "bucket sort: all equal" {
    var arr = [_]i32{ 7, 7, 7, 7, 7 };
    try bucketSort(i32, testing.allocator, &arr, 3);
    try testing.expectEqualSlices(i32, &[_]i32{ 7, 7, 7, 7, 7 }, &arr);
}

test "bucket sort: single element" {
    var arr = [_]i32{42};
    try bucketSort(i32, testing.allocator, &arr, 3);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &arr);
}

test "bucket sort: two elements" {
    var arr = [_]i32{ 2, 1 };
    try bucketSort(i32, testing.allocator, &arr, 2);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2 }, &arr);
}

test "bucket sort: empty array" {
    var arr = [_]i32{};
    try bucketSort(i32, testing.allocator, &arr, 3);
    try testing.expectEqualSlices(i32, &[_]i32{}, &arr);
}

test "bucket sort: negative numbers" {
    var arr = [_]i32{ -5, 3, -1, 7, -3, 0, 2 };
    try bucketSort(i32, testing.allocator, &arr, 4);
    try testing.expectEqualSlices(i32, &[_]i32{ -5, -3, -1, 0, 2, 3, 7 }, &arr);
}

test "bucket sort: large range" {
    var arr = [_]i32{ 100, 1, 50, 200, 25, 150, 75 };
    try bucketSort(i32, testing.allocator, &arr, 5);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 25, 50, 75, 100, 150, 200 }, &arr);
}

test "bucket sort: invalid bucket count" {
    var arr = [_]i32{ 1, 2, 3 };
    try testing.expectError(error.InvalidBucketCount, bucketSort(i32, testing.allocator, &arr, 0));
}

test "bucket sort: optimal bucket count" {
    // Test with bucket count = array length (typical optimal choice)
    var arr = [_]i32{ 64, 34, 25, 12, 22, 11, 90 };
    try bucketSort(i32, testing.allocator, &arr, arr.len);
    try testing.expectEqualSlices(i32, &[_]i32{ 11, 12, 22, 25, 34, 64, 90 }, &arr);
}

test "bucket sort normalized: floating point in [0, 1)" {
    var arr = [_]f64{ 0.897, 0.565, 0.656, 0.1234, 0.665, 0.3434 };
    try bucketSortNormalized(f64, testing.allocator, &arr, 10);

    // Verify sorted order
    for (arr[0 .. arr.len - 1], 0..) |val, i| {
        try testing.expect(val <= arr[i + 1]);
    }
}

test "bucket sort normalized: f32 support" {
    var arr = [_]f32{ 0.9, 0.5, 0.6, 0.1, 0.7, 0.3 };
    try bucketSortNormalized(f32, testing.allocator, &arr, 10);

    for (arr[0 .. arr.len - 1], 0..) |val, i| {
        try testing.expect(val <= arr[i + 1]);
    }
}

test "bucket sort normalized: boundary values" {
    var arr = [_]f64{ 0.0, 0.999, 0.5, 0.001, 0.75, 0.25 };
    try bucketSortNormalized(f64, testing.allocator, &arr, 10);

    try testing.expect(arr[0] <= arr[1]);
    try testing.expect(arr[1] <= arr[2]);
    try testing.expect(arr[2] <= arr[3]);
    try testing.expect(arr[3] <= arr[4]);
    try testing.expect(arr[4] <= arr[5]);
}

test "bucket sort custom: custom mapping function" {
    // Map function: bucket index based on value ranges [0-5)→0, [5-10)→1, [10-15)→2, [15+)→3
    const mapFn = struct {
        fn map(val: i32, num_buckets: usize) usize {
            _ = num_buckets; // 4 buckets expected
            if (val < 5) return 0;
            if (val < 10) return 1;
            if (val < 15) return 2;
            return 3;
        }
    }.map;

    var arr = [_]i32{ 15, 3, 9, 8, 5, 2, 12, 6 };
    try bucketSortCustom(i32, testing.allocator, &arr, 4, mapFn);

    // Verify sorted
    try testing.expectEqualSlices(i32, &[_]i32{ 2, 3, 5, 6, 8, 9, 12, 15 }, &arr);
}

test "bucket sort custom: invalid bucket index" {
    const badMapFn = struct {
        fn map(_: i32, num_buckets: usize) usize {
            return num_buckets; // Out of bounds
        }
    }.map;

    var arr = [_]i32{ 1, 2, 3 };
    try testing.expectError(error.InvalidBucketIndex, bucketSortCustom(i32, testing.allocator, &arr, 3, badMapFn));
}

test "bucket sort: large array with allocator" {
    const allocator = testing.allocator;
    var arr = try allocator.alloc(i32, 100);
    defer allocator.free(arr);

    // Fill with pseudo-random values
    for (arr, 0..) |*val, i| {
        val.* = @as(i32, @intCast((i * 17 + 13) % 100));
    }

    try bucketSort(i32, allocator, arr, 10);

    // Verify sorted
    for (arr[0 .. arr.len - 1], 0..) |val, i| {
        try testing.expect(val <= arr[i + 1]);
    }
}

test "bucket sort: u8 type" {
    var arr = [_]u8{ 200, 50, 150, 100, 75, 25 };
    try bucketSort(u8, testing.allocator, &arr, 5);
    try testing.expectEqualSlices(u8, &[_]u8{ 25, 50, 75, 100, 150, 200 }, &arr);
}
