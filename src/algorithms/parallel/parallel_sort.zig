//! Sequential baselines for parallel sort, scan, map and filter.
//!
//! No function here spawns threads. Each takes the caller's allocator for scratch space only and
//! releases it before returning; `parallelMap`, `parallelFilter` and `parallelPrefixSum` return a
//! slice the caller owns and frees with the same allocator.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const stack_frames_max = 64;
const chunk_size_min = 100;
const quick_sequential_min = 1000;

/// Merge sort that splits the array into `num_threads` chunks, sorts each chunk and merges.
///
/// The chunk sorts run sequentially: this is the baseline that a task-parallel version must
/// match, and it spawns no threads. Chunks shorter than `chunk_size_min` fall back to a
/// plain sort.
///
/// Precondition: `num_threads > 0`. The sort is stable.
///
/// Time: O(n log n) | Space: O(n) for the merge buffer
///
/// Example:
/// ```zig
/// var arr = [_]i32{ 5, 2, 8, 1, 9, 3, 7, 4, 6 };
/// try parallelMergeSort(i32, allocator, &arr, 4);
/// // arr is now [1, 2, 3, 4, 5, 6, 7, 8, 9]
/// ```
pub fn parallelMergeSort(
    comptime T: type,
    gpa: Allocator,
    arr: []T,
    num_threads: usize,
) Allocator.Error!void {
    assert(num_threads > 0);
    if (arr.len <= 1) return;

    const threads = @min(num_threads, arr.len);
    const chunk_size = @divFloor(arr.len + threads - 1, threads);
    if (chunk_size < chunk_size_min) {
        std.mem.sort(T, arr, {}, comptime std.sort.asc(T));
        return;
    }
    assert(chunk_size < arr.len or threads == 1);

    // Allocate before touching `arr`, so an allocation failure leaves the input unchanged.
    const temp = try gpa.alloc(T, arr.len);
    defer gpa.free(temp);

    var chunk_start: usize = 0;
    while (chunk_start < arr.len) : (chunk_start += chunk_size) {
        const chunk_end = @min(chunk_start + chunk_size, arr.len);
        std.mem.sort(T, arr[chunk_start..chunk_end], {}, comptime std.sort.asc(T));
    }

    var merge_size = chunk_size;
    while (merge_size < arr.len) : (merge_size *= 2) {
        var start: usize = 0;
        while (start < arr.len) : (start += merge_size * 2) {
            const mid = @min(start + merge_size, arr.len);
            const end = @min(start + merge_size * 2, arr.len);
            if (mid < end) merge(T, arr[start..end], temp[start..end], mid - start);
        }
    }
}

/// Merges the sorted runs `arr[0..mid]` and `arr[mid..]` in place, using `temp` as scratch.
/// Takes from the left run on ties, so equal elements keep their order.
fn merge(comptime T: type, arr: []T, temp: []T, mid: usize) void {
    assert(arr.len == temp.len);
    assert(mid <= arr.len);

    var left: usize = 0;
    var right: usize = mid;
    var out: usize = 0;
    while (left < mid and right < arr.len) : (out += 1) {
        if (std.sort.asc(T)({}, arr[right], arr[left])) {
            temp[out] = arr[right];
            right += 1;
        } else {
            temp[out] = arr[left];
            left += 1;
        }
    }
    while (left < mid) : (out += 1) {
        temp[out] = arr[left];
        left += 1;
    }
    while (right < arr.len) : (out += 1) {
        temp[out] = arr[right];
        right += 1;
    }
    assert(out == arr.len);
    @memcpy(arr, temp);
}

/// Quick sort that partitions until `max_depth` levels are consumed, then sorts the remaining
/// ranges with a plain sort (which also bounds the damage of bad pivots on skewed input).
///
/// The partition work runs sequentially and spawns no threads. It uses an explicit bounded
/// stack rather than recursion.
///
/// Time: O(n log n) average | Space: O(1)
///
/// Example:
/// ```zig
/// var arr = [_]i32{ 5, 2, 8, 1, 9 };
/// parallelQuickSort(i32, &arr, 4);
/// // arr is now [1, 2, 5, 8, 9]
/// ```
pub fn parallelQuickSort(comptime T: type, arr: []T, max_depth: usize) void {
    if (arr.len <= 1) return;

    const Frame = struct { start: usize, end: usize, depth: usize };
    // Pushing the larger side and looping on the smaller one keeps the stack at log2(n) frames.
    var stack: [stack_frames_max]Frame = undefined;
    var stack_len: usize = 0;
    stack[0] = .{ .start = 0, .end = arr.len, .depth = max_depth };
    stack_len = 1;

    // Each pass retires one frame and each partition retires at least the pivot, so there are
    // at most `arr.len` partitions and `arr.len + 1` frames.
    for (0..arr.len + 1) |_| {
        if (stack_len == 0) break;
        stack_len -= 1;
        var frame = stack[stack_len];

        while (frame.end - frame.start > 1) {
            if (frame.depth == 0 or frame.end - frame.start < quick_sequential_min) {
                std.mem.sort(T, arr[frame.start..frame.end], {}, comptime std.sort.asc(T));
                break;
            }
            const split = partition(T, arr[frame.start..frame.end]);
            const depth = frame.depth - 1;
            const left: Frame = .{
                .start = frame.start,
                .end = frame.start + split.equal_start,
                .depth = depth,
            };
            const right: Frame = .{
                .start = frame.start + split.equal_end,
                .end = frame.end,
                .depth = depth,
            };
            const left_is_smaller = left.end - left.start < right.end - right.start;
            assert(stack_len < stack_frames_max);
            stack[stack_len] = if (left_is_smaller) right else left;
            stack_len += 1;
            frame = if (left_is_smaller) left else right;
        }
    } else unreachable; // The loop bound above is the proof: no input needs more passes.
    assert(stack_len == 0);
}

/// The three regions `partition` leaves behind, as offsets into the partitioned slice.
const Split = struct {
    /// First index of the run equal to the pivot; everything before it is less.
    equal_start: usize,
    /// One past the run equal to the pivot; everything from here on is greater.
    equal_end: usize,
};

/// Three-way (Dutch national flag) partition around the median-of-three pivot. Keys equal to
/// the pivot form one middle run that is never sorted again, so duplicate-heavy input does not
/// degrade to quadratic time.
fn partition(comptime T: type, arr: []T) Split {
    assert(arr.len >= 2);

    const last = arr.len - 1;
    const middle = @divFloor(last, 2);
    // Order arr[0], arr[middle], arr[last]; the median at `middle` is the pivot value.
    if (std.sort.asc(T)({}, arr[middle], arr[0])) std.mem.swap(T, &arr[middle], &arr[0]);
    if (std.sort.asc(T)({}, arr[last], arr[0])) std.mem.swap(T, &arr[last], &arr[0]);
    if (std.sort.asc(T)({}, arr[last], arr[middle])) std.mem.swap(T, &arr[last], &arr[middle]);
    const pivot = arr[middle];

    var less_end: usize = 0; // arr[0..less_end] < pivot.
    var scan: usize = 0; // arr[less_end..scan] == pivot.
    var greater_start: usize = arr.len; // arr[greater_start..] > pivot.
    while (scan < greater_start) {
        if (std.sort.asc(T)({}, arr[scan], pivot)) {
            std.mem.swap(T, &arr[less_end], &arr[scan]);
            less_end += 1;
            scan += 1;
        } else if (std.sort.asc(T)({}, pivot, arr[scan])) {
            greater_start -= 1;
            std.mem.swap(T, &arr[scan], &arr[greater_start]);
        } else {
            scan += 1;
        }
    }
    // The pivot value occurs at least once, so the middle run is non-empty and both sides shrink.
    assert(less_end < greater_start);
    assert(less_end < arr.len);
    return .{ .equal_start = less_end, .equal_end = greater_start };
}

/// Parallel prefix sum (scan) using work-efficient algorithm
///
/// Computes cumulative sum of array elements in parallel.
/// Uses up-sweep and down-sweep phases for O(n) work.
///
/// Time: O(log n) parallel, O(n) sequential work
/// Space: O(n) for result array
///
/// Example:
/// ```zig
/// const arr = [_]i32{ 1, 2, 3, 4, 5 };
/// const result = try parallelPrefixSum(i32, allocator, &arr);
/// defer allocator.free(result);
/// // result is [1, 3, 6, 10, 15]
/// ```
pub fn parallelPrefixSum(comptime T: type, allocator: Allocator, arr: []const T) ![]T {
    if (arr.len == 0) return try allocator.alloc(T, 0);

    var result = try allocator.alloc(T, arr.len);
    errdefer allocator.free(result);

    @memcpy(result, arr);

    // Sequential implementation (parallel would use threads)
    var sum: T = 0;
    for (result, 0..) |val, i| {
        sum += val;
        result[i] = sum;
    }

    return result;
}

/// Parallel reduction (fold) operation
///
/// Combines array elements using associative binary operation in parallel.
/// Uses tree-based reduction for logarithmic depth.
///
/// Time: O(log n) parallel, O(n) sequential work
/// Space: O(1)
///
/// Example:
/// ```zig
/// const arr = [_]i32{ 1, 2, 3, 4, 5 };
/// const sum = parallelReduce(i32, &arr, 0, add);
/// // sum is 15
/// ```
pub fn parallelReduce(
    comptime T: type,
    arr: []const T,
    init: T,
    comptime op: fn (T, T) T,
) T {
    if (arr.len == 0) return init;

    var result = init;
    for (arr) |val| {
        result = op(result, val);
    }
    return result;
}

fn add(a: i32, b: i32) i32 {
    return a + b;
}

fn mul(a: i32, b: i32) i32 {
    return a * b;
}

/// Parallel map operation
///
/// Applies function to each array element in parallel.
/// Creates new array with transformed values.
///
/// Time: O(1) parallel per element, O(n) sequential work
/// Space: O(n) for result array
///
/// Example:
/// ```zig
/// const arr = [_]i32{ 1, 2, 3, 4, 5 };
/// const result = try parallelMap(i32, i32, allocator, &arr, double);
/// defer allocator.free(result);
/// // result is [2, 4, 6, 8, 10]
/// ```
pub fn parallelMap(
    comptime T: type,
    comptime U: type,
    allocator: Allocator,
    arr: []const T,
    comptime f: fn (T) U,
) ![]U {
    var result = try allocator.alloc(U, arr.len);
    errdefer allocator.free(result);

    for (arr, 0..) |val, i| {
        result[i] = f(val);
    }

    return result;
}

/// Parallel filter operation
///
/// Selects elements satisfying predicate in parallel.
/// Returns new array with filtered values.
///
/// Time: O(n) with parallel partitioning
/// Space: O(n) worst case for result
///
/// Example:
/// ```zig
/// const arr = [_]i32{ 1, 2, 3, 4, 5, 6 };
/// const result = try parallelFilter(i32, allocator, &arr, isEven);
/// defer allocator.free(result);
/// // result is [2, 4, 6]
/// ```
pub fn parallelFilter(
    comptime T: type,
    allocator: Allocator,
    arr: []const T,
    comptime pred: fn (T) bool,
) ![]T {
    var result: std.ArrayList(T) = .empty;
    defer result.deinit(allocator);

    for (arr) |val| {
        if (pred(val)) {
            try result.append(allocator, val);
        }
    }

    return result.toOwnedSlice(allocator);
}

// ============================================================================
// Tests
// ============================================================================

test "parallel merge sort - basic" {
    const allocator = std.testing.allocator;

    var arr = [_]i32{ 5, 2, 8, 1, 9, 3, 7, 4, 6 };
    try parallelMergeSort(i32, allocator, &arr, 2);

    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &arr);
}

test "parallel merge sort - already sorted" {
    const allocator = std.testing.allocator;

    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    try parallelMergeSort(i32, allocator, &arr, 2);

    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "parallel merge sort - reverse sorted" {
    const allocator = std.testing.allocator;

    var arr = [_]i32{ 5, 4, 3, 2, 1 };
    try parallelMergeSort(i32, allocator, &arr, 2);

    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "parallel merge sort - duplicates" {
    const allocator = std.testing.allocator;

    var arr = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    try parallelMergeSort(i32, allocator, &arr, 2);

    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2, 3, 3, 4, 5, 5, 6, 9 }, &arr);
}

test "parallel merge sort - empty" {
    const allocator = std.testing.allocator;

    var arr = [_]i32{};
    try parallelMergeSort(i32, allocator, &arr, 2);

    try std.testing.expectEqualSlices(i32, &[_]i32{}, &arr);
}

test "parallel merge sort - single element" {
    const allocator = std.testing.allocator;

    var arr = [_]i32{42};
    try parallelMergeSort(i32, allocator, &arr, 2);

    try std.testing.expectEqualSlices(i32, &[_]i32{42}, &arr);
}

test "parallel quick sort - basic" {
    var arr = [_]i32{ 5, 2, 8, 1, 9, 3, 7, 4, 6 };
    parallelQuickSort(i32, &arr, 4);

    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &arr);
}

test "parallel quick sort - already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    parallelQuickSort(i32, &arr, 4);

    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "parallel quick sort - duplicates" {
    var arr = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    parallelQuickSort(i32, &arr, 4);

    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2, 3, 3, 4, 5, 5, 6, 9 }, &arr);
}

test "parallel prefix sum - basic" {
    const allocator = std.testing.allocator;

    const arr = [_]i32{ 1, 2, 3, 4, 5 };
    const result = try parallelPrefixSum(i32, allocator, &arr);
    defer allocator.free(result);

    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 3, 6, 10, 15 }, result);
}

test "parallel prefix sum - single element" {
    const allocator = std.testing.allocator;

    const arr = [_]i32{42};
    const result = try parallelPrefixSum(i32, allocator, &arr);
    defer allocator.free(result);

    try std.testing.expectEqualSlices(i32, &[_]i32{42}, result);
}

test "parallel prefix sum - empty" {
    const allocator = std.testing.allocator;

    const arr = [_]i32{};
    const result = try parallelPrefixSum(i32, allocator, &arr);
    defer allocator.free(result);

    try std.testing.expectEqualSlices(i32, &[_]i32{}, result);
}

test "parallel reduce - sum" {
    const arr = [_]i32{ 1, 2, 3, 4, 5 };
    const sum = parallelReduce(i32, &arr, 0, add);

    try std.testing.expectEqual(@as(i32, 15), sum);
}

test "parallel reduce - product" {
    const arr = [_]i32{ 1, 2, 3, 4, 5 };
    const product = parallelReduce(i32, &arr, 1, mul);

    try std.testing.expectEqual(@as(i32, 120), product);
}

test "parallel reduce - empty" {
    const arr = [_]i32{};
    const sum = parallelReduce(i32, &arr, 42, add);

    try std.testing.expectEqual(@as(i32, 42), sum);
}

test "parallel map - double" {
    const allocator = std.testing.allocator;

    const double = struct {
        fn f(x: i32) i32 {
            return x * 2;
        }
    }.f;

    const arr = [_]i32{ 1, 2, 3, 4, 5 };
    const result = try parallelMap(i32, i32, allocator, &arr, double);
    defer allocator.free(result);

    try std.testing.expectEqualSlices(i32, &[_]i32{ 2, 4, 6, 8, 10 }, result);
}

test "parallel map - square" {
    const allocator = std.testing.allocator;

    const square = struct {
        fn f(x: i32) i32 {
            return x * x;
        }
    }.f;

    const arr = [_]i32{ 1, 2, 3, 4, 5 };
    const result = try parallelMap(i32, i32, allocator, &arr, square);
    defer allocator.free(result);

    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 4, 9, 16, 25 }, result);
}

test "parallel filter - even numbers" {
    const allocator = std.testing.allocator;

    const isEven = struct {
        fn f(x: i32) bool {
            return @mod(x, 2) == 0;
        }
    }.f;

    const arr = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const result = try parallelFilter(i32, allocator, &arr, isEven);
    defer allocator.free(result);

    try std.testing.expectEqualSlices(i32, &[_]i32{ 2, 4, 6, 8 }, result);
}

test "parallel filter - positive numbers" {
    const allocator = std.testing.allocator;

    const isPositive = struct {
        fn f(x: i32) bool {
            return x > 0;
        }
    }.f;

    const arr = [_]i32{ -2, -1, 0, 1, 2, 3 };
    const result = try parallelFilter(i32, allocator, &arr, isPositive);
    defer allocator.free(result);

    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3 }, result);
}

test "parallel filter - empty result" {
    const allocator = std.testing.allocator;

    const alwaysFalse = struct {
        fn f(_: i32) bool {
            return false;
        }
    }.f;

    const arr = [_]i32{ 1, 2, 3 };
    const result = try parallelFilter(i32, allocator, &arr, alwaysFalse);
    defer allocator.free(result);

    try std.testing.expectEqualSlices(i32, &[_]i32{}, result);
}

fn fill_random(arr: []i32, seed: u64, value_max: i32) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (arr) |*slot| slot.* = random.intRangeAtMost(i32, -value_max, value_max);
}

fn expect_sorted_like_std(original: []const i32, actual: []const i32) !void {
    const expected = try std.testing.allocator.dupe(i32, original);
    defer std.testing.allocator.free(expected);

    std.mem.sort(i32, expected, {}, std.sort.asc(i32));
    try std.testing.expectEqualSlices(i32, expected, actual);
}

test "parallel merge sort - large inputs across thread counts and ragged chunks" {
    const gpa = std.testing.allocator;
    const sizes = [_]usize{ 200, 1000, 4999, 5000 };
    const thread_counts = [_]usize{ 1, 2, 3, 7, 64 };

    for (sizes, 0..) |size, size_index| {
        for (thread_counts) |threads| {
            const original = try gpa.alloc(i32, size);
            defer gpa.free(original);

            fill_random(original, 0x5eed + size_index, 500);
            const arr = try gpa.dupe(i32, original);
            defer gpa.free(arr);

            try parallelMergeSort(i32, gpa, arr, threads);
            try expect_sorted_like_std(original, arr);
        }
    }
}

test "parallel merge sort - allocation failure leaves the input unchanged" {
    const gpa = std.testing.allocator;
    var arr: [400]i32 = undefined;
    fill_random(&arr, 7, 1000);
    const original = arr;

    var failing = std.testing.FailingAllocator.init(gpa, .{ .fail_index = 0 });
    try std.testing.expectError(
        error.OutOfMemory,
        parallelMergeSort(i32, failing.allocator(), &arr, 2),
    );
    try std.testing.expectEqualSlices(i32, &original, &arr);
}

test "parallel merge sort - boundary sizes and an oversized thread count" {
    const gpa = std.testing.allocator;
    const sizes = [_]usize{ 0, 1, 2, 99, 100, 101, 199, 200, 201 };

    for (sizes) |size| {
        const original = try gpa.alloc(i32, size);
        defer gpa.free(original);

        fill_random(original, 99, 50);
        const arr = try gpa.dupe(i32, original);
        defer gpa.free(arr);

        try parallelMergeSort(i32, gpa, arr, std.math.maxInt(usize));
        try expect_sorted_like_std(original, arr);

        try parallelMergeSort(i32, gpa, arr, 2);
        try expect_sorted_like_std(original, arr);
    }
}

test "parallel quick sort - large inputs across depths and shapes" {
    const gpa = std.testing.allocator;
    const depths = [_]usize{ 0, 1, 3, 64, 1000 };

    for (depths) |depth| {
        for (0..5) |shape| {
            const original = try gpa.alloc(i32, 5000);
            defer gpa.free(original);

            switch (shape) {
                0 => fill_random(original, 11 + depth, 100_000),
                1 => fill_random(original, 12 + depth, 3), // Heavy duplicates.
                2 => for (original, 0..) |*slot, i| {
                    slot.* = @intCast(i);
                },
                3 => for (original, 0..) |*slot, i| {
                    slot.* = @intCast(original.len - i);
                },
                else => @memset(original, 9),
            }
            const arr = try gpa.dupe(i32, original);
            defer gpa.free(arr);

            parallelQuickSort(i32, arr, depth);
            try expect_sorted_like_std(original, arr);
        }
    }
}

test "partition - three regions around the pivot" {
    var arr = [_]i32{ 9, 1, 8, 2, 7, 3, 6, 4, 5, 5, 5 };
    const split = partition(i32, &arr);

    try std.testing.expect(split.equal_start < split.equal_end);
    const pivot = arr[split.equal_start];
    for (arr[0..split.equal_start]) |value| try std.testing.expect(value < pivot);
    for (arr[split.equal_start..split.equal_end]) |value| try std.testing.expectEqual(pivot, value);
    for (arr[split.equal_end..]) |value| try std.testing.expect(value > pivot);

    var two = [_]i32{ 2, 1 };
    const two_split = partition(i32, &two);
    try std.testing.expectEqual(@as(usize, 1), two_split.equal_end - two_split.equal_start);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 2 }, &two);

    var same = [_]i32{ 4, 4, 4, 4 };
    const same_split = partition(i32, &same);
    try std.testing.expectEqual(@as(usize, 0), same_split.equal_start);
    try std.testing.expectEqual(@as(usize, 4), same_split.equal_end);
}

test "parallel quick sort - all-equal keys finish in one pass at any depth" {
    const gpa = std.testing.allocator;
    const arr = try gpa.alloc(i32, 100_000);
    defer gpa.free(arr);

    @memset(arr, 3);
    parallelQuickSort(i32, arr, std.math.maxInt(usize));
    for (arr) |value| try std.testing.expectEqual(@as(i32, 3), value);
}

test "parallel filter and map - every allocation failure is handled" {
    const keep_odd = struct {
        fn f(x: i32) bool {
            return @mod(x, 2) == 1;
        }
    }.f;
    const negate = struct {
        fn f(x: i32) i32 {
            return -x;
        }
    }.f;

    const Runner = struct {
        fn run(gpa: Allocator) !void {
            const arr = [_]i32{ 1, 2, 3, 4, 5, 6, 7 };
            const odd = try parallelFilter(i32, gpa, &arr, keep_odd);
            defer gpa.free(odd);

            const negated = try parallelMap(i32, i32, gpa, &arr, negate);
            defer gpa.free(negated);

            const sums = try parallelPrefixSum(i32, gpa, &arr);
            defer gpa.free(sums);

            try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 3, 5, 7 }, odd);
            try std.testing.expectEqual(@as(i32, -7), negated[6]);
            try std.testing.expectEqual(@as(i32, 28), sums[6]);
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, Runner.run, .{});
}
