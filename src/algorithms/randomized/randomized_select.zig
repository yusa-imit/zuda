const std = @import("std");
const testing = std.testing;

/// Randomized selection algorithm for finding the kth smallest element.
///
/// Time: O(n) expected, O(n²) worst case | Space: O(log n) for recursion
///
/// This is a randomized variant of Quickselect that uses random pivot selection
/// to achieve expected linear time. Unlike the deterministic median-of-medians
/// approach, this is simpler and faster in practice.
///
/// Algorithm:
/// 1. Pick random pivot
/// 2. Partition array around pivot
/// 3. Recursively search in appropriate partition
///
/// Expected time is O(n) due to random pivot selection, which typically gives
/// balanced partitions. The probability of worst-case O(n²) is negligible.
pub fn randomizedSelect(
    comptime T: type,
    array: []T,
    k: usize,
    random: std.Random,
    comptime lessThan: fn (T, T) bool,
) T {
    std.debug.assert(k < array.len);
    return randomizedSelectImpl(T, array, k, random, lessThan);
}

fn randomizedSelectImpl(
    comptime T: type,
    array: []T,
    k: usize,
    random: std.Random,
    comptime lessThan: fn (T, T) bool,
) T {
    if (array.len == 1) return array[0];

    // Random pivot selection
    const pivot_idx = random.intRangeLessThan(usize, 0, array.len);
    const pivot = array[pivot_idx];

    // Partition around pivot
    var i: usize = 0;
    var j: usize = 0;
    var high: usize = array.len;

    while (j < high) {
        if (lessThan(array[j], pivot)) {
            std.mem.swap(T, &array[i], &array[j]);
            i += 1;
            j += 1;
        } else if (lessThan(pivot, array[j])) {
            high -= 1;
            std.mem.swap(T, &array[j], &array[high]);
        } else {
            j += 1;
        }
    }

    // array[0..i] < pivot
    // array[i..j] == pivot
    // array[j..] > pivot

    if (k < i) {
        // kth smallest is in left partition
        return randomizedSelectImpl(T, array[0..i], k, random, lessThan);
    } else if (k < j) {
        // kth smallest is the pivot
        return pivot;
    } else {
        // kth smallest is in right partition
        return randomizedSelectImpl(T, array[j..], k - j, random, lessThan);
    }
}

/// Find the median of an array using randomized selection.
///
/// Time: O(n) expected | Space: O(log n)
///
/// For even-length arrays, returns the lower median.
pub fn median(
    comptime T: type,
    array: []T,
    random: std.Random,
    comptime lessThan: fn (T, T) bool,
) T {
    std.debug.assert(array.len > 0);
    const k = array.len / 2;
    return randomizedSelect(T, array, k, random, lessThan);
}

/// Find the top k elements (k largest) using randomized selection.
///
/// Time: O(n) expected | Space: O(1) in-place + O(log n) recursion
///
/// After this call, the largest k elements are in array[n-k..n], but not sorted.
/// The rest of the array may be modified.
pub fn topK(
    comptime T: type,
    array: []T,
    k: usize,
    random: std.Random,
    comptime lessThan: fn (T, T) bool,
) []T {
    if (k == 0) return array[array.len..];
    if (k >= array.len) return array;

    // Find the (n-k)th smallest element
    // Everything to the right will be >= this element
    _ = randomizedSelect(T, array, array.len - k, random, lessThan);

    return array[array.len - k ..];
}

fn lessThanI32(_: void, a: i32, b: i32) bool {
    return a < b;
}

fn lessThanU32(_: void, a: u32, b: u32) bool {
    return a < b;
}

test "randomized_select: basic kth element" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var array = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    const result = randomizedSelect(i32, &array, 5, random, lessThanI32);

    // When sorted: 1,1,2,3,3,4,5,5,6,9
    // k=5 (6th element) should be 4
    try testing.expectEqual(4, result);
}

test "randomized_select: smallest element (k=0)" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var array = [_]i32{ 5, 2, 8, 1, 9, 3 };
    const result = randomizedSelect(i32, &array, 0, random, lessThanI32);

    try testing.expectEqual(1, result);
}

test "randomized_select: largest element (k=n-1)" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var array = [_]i32{ 5, 2, 8, 1, 9, 3 };
    const result = randomizedSelect(i32, &array, 5, random, lessThanI32);

    try testing.expectEqual(9, result);
}

test "randomized_select: single element" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var array = [_]i32{42};
    const result = randomizedSelect(i32, &array, 0, random, lessThanI32);

    try testing.expectEqual(42, result);
}

test "randomized_select: all duplicates" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var array = [_]i32{ 7, 7, 7, 7, 7 };
    const result = randomizedSelect(i32, &array, 2, random, lessThanI32);

    try testing.expectEqual(7, result);
}

test "randomized_select: with duplicates" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var array = [_]i32{ 1, 3, 2, 3, 4, 3, 5 };
    const result = randomizedSelect(i32, &array, 3, random, lessThanI32);

    // Sorted: 1,2,3,3,3,4,5
    // k=3 (4th element) should be 3
    try testing.expectEqual(3, result);
}

test "randomized_select: large array" {
    const prng = std.Random.DefaultPrng.init(777);
    const random = prng.random();

    const array = try testing.allocator.alloc(u32, 1000);
    defer testing.allocator.free(array);

    // Fill with random values
    for (array) |*val| {
        val.* = random.int(u32);
    }

    const k = 500;
    const result = randomizedSelect(u32, array, k, random, lessThanU32);

    // Verify: count how many elements are < result
    var count_less: usize = 0;
    var count_equal: usize = 0;
    for (array) |val| {
        if (val < result) count_less += 1;
        if (val == result) count_equal += 1;
    }

    // The kth element should have exactly k elements < it (or k in range [count_less, count_less + count_equal])
    try testing.expect(count_less <= k);
    try testing.expect(k < count_less + count_equal);
}

test "randomized_select: median odd length" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var array = [_]i32{ 1, 3, 5, 7, 9 };
    const result = median(i32, &array, random, lessThanI32);

    // Median of 5 elements is the 3rd element when sorted
    try testing.expectEqual(5, result);
}

test "randomized_select: median even length" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var array = [_]i32{ 1, 2, 3, 4, 5, 6 };
    const result = median(i32, &array, random, lessThanI32);

    // Lower median of 6 elements (index 3, 0-indexed) should be 4
    try testing.expectEqual(4, result);
}

test "randomized_select: median unsorted" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var array = [_]i32{ 9, 1, 8, 2, 7, 3, 6 };
    const result = median(i32, &array, random, lessThanI32);

    // Sorted: 1,2,3,6,7,8,9
    // Median (index 3) is 6
    try testing.expectEqual(6, result);
}

test "randomized_select: top k=0" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var array = [_]i32{ 1, 2, 3, 4, 5 };
    const top = topK(i32, &array, 0, random, lessThanI32);

    try testing.expectEqual(0, top.len);
}

test "randomized_select: top k=3" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var array = [_]i32{ 1, 5, 3, 9, 2, 7, 4 };
    const top = topK(i32, &array, 3, random, lessThanI32);

    try testing.expectEqual(3, top.len);

    // Top 3 should be 5, 7, 9 (in some order)
    var sum: i32 = 0;
    for (top) |val| sum += val;
    try testing.expectEqual(21, sum); // 5+7+9=21

    // All top-k elements should be >= 5
    for (top) |val| {
        try testing.expect(val >= 5);
    }
}

test "randomized_select: top k >= n" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var array = [_]i32{ 1, 2, 3, 4, 5 };
    const top = topK(i32, &array, 10, random, lessThanI32);

    try testing.expectEqual(5, top.len);

    var sum: i32 = 0;
    for (top) |val| sum += val;
    try testing.expectEqual(15, sum);
}

test "randomized_select: top k all elements" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var array = [_]i32{ 5, 2, 8, 1, 9 };
    const top = topK(i32, &array, 5, random, lessThanI32);

    try testing.expectEqual(5, top.len);

    var sum: i32 = 0;
    for (top) |val| sum += val;
    try testing.expectEqual(25, sum);
}

test "randomized_select: already sorted ascending" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var array = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const result = randomizedSelect(i32, &array, 5, random, lessThanI32);

    try testing.expectEqual(6, result);
}

test "randomized_select: already sorted descending" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var array = [_]i32{ 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    const result = randomizedSelect(i32, &array, 5, random, lessThanI32);

    try testing.expectEqual(6, result);
}

test "randomized_select: stress test" {
    const prng = std.Random.DefaultPrng.init(999);
    const random = prng.random();

    const array = try testing.allocator.alloc(i32, 500);
    defer testing.allocator.free(array);

    // Fill with random values
    for (array) |*val| {
        val.* = @as(i32, @intCast(random.intRangeAtMost(u32, 0, 1000)));
    }

    // Test multiple k values
    const k_values = [_]usize{ 0, 100, 250, 400, 499 };
    for (k_values) |k| {
        const array_copy = try testing.allocator.alloc(i32, array.len);
        defer testing.allocator.free(array_copy);
        @memcpy(array_copy, array);

        const result = randomizedSelect(i32, array_copy, k, random, lessThanI32);

        // Verify correctness by sorting and comparing
        const sorted = try testing.allocator.alloc(i32, array.len);
        defer testing.allocator.free(sorted);
        @memcpy(sorted, array);

        std.mem.sort(i32, sorted, {}, lessThanI32);
        try testing.expectEqual(sorted[k], result);
    }
}
