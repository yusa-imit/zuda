const std = @import("std");
const testing = std.testing;

/// Fisher-Yates shuffle algorithm for generating random permutations.
///
/// Time: O(n) | Space: O(1)
///
/// The Fisher-Yates (also known as Knuth) shuffle produces an unbiased random
/// permutation of an array. Each of the n! possible permutations has equal probability.
///
/// Algorithm:
/// 1. Iterate from last element to first
/// 2. For each position i, pick random j in [0, i]
/// 3. Swap elements at positions i and j
///
/// Properties:
/// - In-place shuffling (O(1) space)
/// - Unbiased: all permutations equally likely
/// - Linear time complexity
/// - Single pass through array
pub fn shuffle(comptime T: type, array: []T, random: std.Random) void {
    if (array.len <= 1) return;

    var i: usize = array.len - 1;
    while (i > 0) : (i -= 1) {
        const j = random.intRangeLessThan(usize, 0, i + 1);
        std.mem.swap(T, &array[i], &array[j]);
    }
}

/// Partial Fisher-Yates shuffle - generates k random samples from array.
///
/// Time: O(k) | Space: O(1)
///
/// Shuffles only the first k elements of the array, leaving the rest unchanged.
/// This is more efficient than full shuffle when k << n.
///
/// After the call, the first k elements of array contain k randomly selected
/// elements from the original array (without replacement).
pub fn partialShuffle(comptime T: type, array: []T, k: usize, random: std.Random) void {
    if (k == 0 or array.len == 0) return;
    const num_samples = @min(k, array.len);

    var i: usize = 0;
    while (i < num_samples) : (i += 1) {
        const j = random.intRangeLessThan(usize, i, array.len);
        std.mem.swap(T, &array[i], &array[j]);
    }
}

/// Generate a random permutation of integers 0..n-1.
///
/// Time: O(n) | Space: O(n)
///
/// This allocates a new array and fills it with a random permutation.
pub fn randomPermutation(allocator: std.mem.Allocator, n: usize, random: std.Random) ![]usize {
    const perm = try allocator.alloc(usize, n);
    for (perm, 0..) |*p, i| {
        p.* = i;
    }
    shuffle(usize, perm, random);
    return perm;
}

test "fisher_yates: basic shuffle" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var array = [_]i32{ 1, 2, 3, 4, 5 };
    const original = array;

    shuffle(i32, &array, random);

    // Array should be permuted
    var same_count: usize = 0;
    for (array, original) |a, b| {
        if (a == b) same_count += 1;
    }

    // With 5 elements, probability all same is 1/120, so likely different
    try testing.expect(same_count < 5);

    // All original elements should still be present
    var found = [_]bool{false} ** 5;
    for (array) |val| {
        found[@as(usize, @intCast(val - 1))] = true;
    }
    for (found) |f| {
        try testing.expect(f);
    }
}

test "fisher_yates: empty array" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var array = [_]i32{};
    shuffle(i32, &array, random);
    try testing.expectEqual(0, array.len);
}

test "fisher_yates: single element" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var array = [_]i32{42};
    shuffle(i32, &array, random);
    try testing.expectEqual(42, array[0]);
}

test "fisher_yates: distribution test" {
    // Test that each position can receive each element (basic unbiasedness check)
    const prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    const n = 4;
    const trials = 1000;
    var position_counts: [n][n]usize = undefined;

    // Initialize counts to 0
    for (&position_counts) |*row| {
        for (row) |*count| {
            count.* = 0;
        }
    }

    // Run many trials
    var trial: usize = 0;
    while (trial < trials) : (trial += 1) {
        var array = [_]usize{ 0, 1, 2, 3 };
        shuffle(usize, &array, random);

        for (array, 0..) |val, pos| {
            position_counts[pos][val] += 1;
        }
    }

    // Each element should appear in each position roughly trials/n times
    // With 1000 trials and 4 elements, expect ~250 per cell
    // Allow wide margin: at least 100, at most 400
    for (position_counts) |row| {
        for (row) |count| {
            try testing.expect(count >= 100);
            try testing.expect(count <= 400);
        }
    }
}

test "fisher_yates: partial shuffle k=0" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var array = [_]i32{ 1, 2, 3, 4, 5 };
    const original = array;

    partialShuffle(i32, &array, 0, random);

    // Array should be unchanged
    try testing.expectEqualSlices(i32, &original, &array);
}

test "fisher_yates: partial shuffle k=2" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var array = [_]i32{ 1, 2, 3, 4, 5 };

    partialShuffle(i32, &array, 2, random);

    // First 2 elements should be from the original array
    for (array[0..2]) |val| {
        var found = false;
        for ([_]i32{ 1, 2, 3, 4, 5 }) |orig| {
            if (val == orig) {
                found = true;
                break;
            }
        }
        try testing.expect(found);
    }
}

test "fisher_yates: partial shuffle k > n" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var array = [_]i32{ 1, 2, 3 };

    // k=10 > n=3, should shuffle entire array
    partialShuffle(i32, &array, 10, random);

    // All elements should still be present
    var sum: i32 = 0;
    for (array) |val| sum += val;
    try testing.expectEqual(6, sum);
}

test "fisher_yates: random permutation" {
    var prng = std.Random.DefaultPrng.init(99);
    const random = prng.random();

    const perm = try randomPermutation(testing.allocator, 10, random);
    defer testing.allocator.free(perm);

    try testing.expectEqual(10, perm.len);

    // All elements 0..9 should be present exactly once
    var found = [_]bool{false} ** 10;
    for (perm) |val| {
        try testing.expect(val < 10);
        try testing.expect(!found[val]);
        found[val] = true;
    }

    for (found) |f| {
        try testing.expect(f);
    }
}

test "fisher_yates: large array shuffle" {
    const prng = std.Random.DefaultPrng.init(777);
    const random = prng.random();

    const array = try testing.allocator.alloc(usize, 1000);
    defer testing.allocator.free(array);

    for (array, 0..) |*val, i| {
        val.* = i;
    }

    shuffle(usize, array, random);

    // All elements should still be present
    var found = try testing.allocator.alloc(bool, 1000);
    defer testing.allocator.free(found);

    for (found) |*f| f.* = false;
    for (array) |val| {
        try testing.expect(val < 1000);
        try testing.expect(!found[val]);
        found[val] = true;
    }

    for (found) |f| {
        try testing.expect(f);
    }
}

test "fisher_yates: string shuffle" {
    const prng = std.Random.DefaultPrng.init(123);
    const random = prng.random();

    var str = "ABCDEFGH".*;
    shuffle(u8, &str, random);

    // All characters should still be present
    var char_counts = std.mem.zeroes([256]u8);
    for (str) |c| {
        char_counts[c] += 1;
    }

    try testing.expectEqual(1, char_counts['A']);
    try testing.expectEqual(1, char_counts['B']);
    try testing.expectEqual(1, char_counts['C']);
    try testing.expectEqual(1, char_counts['D']);
    try testing.expectEqual(1, char_counts['E']);
    try testing.expectEqual(1, char_counts['F']);
    try testing.expectEqual(1, char_counts['G']);
    try testing.expectEqual(1, char_counts['H']);
}
