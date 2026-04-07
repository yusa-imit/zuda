const std = @import("std");
const testing = std.testing;
const Order = std.math.Order;
const Random = std.Random;

/// Bogo Sort (Stupid Sort, Permutation Sort, Shotgun Sort)
///
/// WARNING: This is an intentionally inefficient algorithm for EDUCATIONAL PURPOSES ONLY.
/// DO NOT use in production! Average case is O((n+1)!) and worst case is unbounded.
///
/// Algorithm: Repeatedly shuffle the array and check if sorted. Demonstrates:
/// - Worst-case complexity analysis
/// - Probabilistic termination
/// - Importance of algorithm selection
/// - Random permutation generation
///
/// Educational value:
/// - Shows what NOT to do
/// - Demonstrates factorial complexity growth
/// - Useful for teaching complexity theory
/// - Highlights importance of deterministic algorithms
///
/// Time complexity:
/// - Best case: O(n) - already sorted, one check
/// - Average case: O((n+1)!) - expected number of permutations
/// - Worst case: Unbounded (may never terminate)
///
/// Space complexity: O(1) - in-place shuffling
///
/// Stability: Not applicable (unstable due to random shuffling)
///
/// References:
/// - Used in complexity theory education
/// - Demonstrates non-polynomial time algorithms
/// - Example of Las Vegas algorithm (always correct when terminates)
pub fn bogoSort(comptime T: type, arr: []T, compareFn: fn (T, T) Order) !void {
    if (arr.len <= 1) return;

    var prng = std.Random.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        std.posix.getrandom(std.mem.asBytes(&seed)) catch @panic("getrandom failed");
        break :blk seed;
    });
    const random = prng.random();

    while (!isSorted(T, arr, compareFn)) {
        shuffle(T, arr, random);
    }
}

/// Bogo Sort for ascending order (convenience wrapper)
///
/// WARNING: Educational purposes only! O((n+1)!) average case.
///
/// Time: O((n+1)!) average, Space: O(1)
pub fn bogoSortAsc(comptime T: type, arr: []T) !void {
    const asc = struct {
        fn compare(a: T, b: T) Order {
            return std.math.order(a, b);
        }
    }.compare;
    try bogoSort(T, arr, asc);
}

/// Bogo Sort for descending order (convenience wrapper)
///
/// WARNING: Educational purposes only! O((n+1)!) average case.
///
/// Time: O((n+1)!) average, Space: O(1)
pub fn bogoSortDesc(comptime T: type, arr: []T) !void {
    const desc = struct {
        fn compare(a: T, b: T) Order {
            return std.math.order(b, a);
        }
    }.compare;
    try bogoSort(T, arr, desc);
}

/// Bogo Sort with Order-based comparison
///
/// WARNING: Educational purposes only! O((n+1)!) average case.
///
/// Time: O((n+1)!) average, Space: O(1)
pub fn bogoSortBy(comptime T: type, arr: []T, order: Order) !void {
    switch (order) {
        .lt => try bogoSortAsc(T, arr),
        .gt => try bogoSortDesc(T, arr),
        .eq => return, // all equal or empty
    }
}

/// Deterministic Bogo Sort (Permutation Sort)
///
/// Generates all permutations deterministically and checks each.
/// Even WORSE than random bogo sort - O(n! × n) guaranteed.
///
/// WARNING: Only suitable for n ≤ 4. DO NOT use with larger arrays!
///
/// Time: O(n! × n), Space: O(n) for recursion stack
pub fn bogoSortDeterministic(comptime T: type, arr: []T, compareFn: fn (T, T) Order) !void {
    if (arr.len <= 1) return;
    if (arr.len > 10) return error.ArrayTooLarge; // Safety limit

    var sorted = false;
    try generatePermutations(T, arr, 0, compareFn, &sorted);
}

fn generatePermutations(comptime T: type, arr: []T, start: usize, compareFn: fn (T, T) Order, sorted: *bool) !void {
    if (sorted.*) return; // Already sorted, stop

    if (start >= arr.len) {
        // Reached a complete permutation, check if sorted
        if (isSorted(T, arr, compareFn)) {
            sorted.* = true;
        }
        return;
    }

    for (start..arr.len) |i| {
        std.mem.swap(T, &arr[start], &arr[i]);
        try generatePermutations(T, arr, start + 1, compareFn, sorted);
        if (!sorted.*) {
            std.mem.swap(T, &arr[start], &arr[i]); // backtrack only if not sorted
        } else {
            return; // Keep the sorted permutation
        }
    }
}

/// Bounded Bogo Sort - stops after max iterations
///
/// Returns error.MaxIterationsExceeded if not sorted within limit.
/// Useful for testing/education without infinite loops.
///
/// Time: O(n × max_iterations), Space: O(1)
pub fn bogoSortBounded(comptime T: type, arr: []T, compareFn: fn (T, T) Order, max_iterations: usize) !void {
    if (arr.len <= 1) return;

    var prng = std.Random.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        std.posix.getrandom(std.mem.asBytes(&seed)) catch @panic("getrandom failed");
        break :blk seed;
    });
    const random = prng.random();

    var iterations: usize = 0;
    while (!isSorted(T, arr, compareFn)) {
        if (iterations >= max_iterations) return error.MaxIterationsExceeded;
        shuffle(T, arr, random);
        iterations += 1;
    }
}

/// Count iterations until sorted (for analysis)
///
/// Returns number of shuffles needed to sort the array.
/// Uses seeded RNG for reproducibility.
///
/// Time: O((n+1)!) average, Space: O(1)
pub fn countBogoSortIterations(comptime T: type, arr: []T, compareFn: fn (T, T) Order, seed: u64) usize {
    if (arr.len <= 1) return 0;

    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    var iterations: usize = 0;
    while (!isSorted(T, arr, compareFn)) {
        shuffle(T, arr, random);
        iterations += 1;
        if (iterations > 1_000_000) break; // safety limit
    }
    return iterations;
}

// Helper: Check if array is sorted
fn isSorted(comptime T: type, arr: []const T, compareFn: fn (T, T) Order) bool {
    if (arr.len <= 1) return true;

    for (0..arr.len - 1) |i| {
        const cmp = compareFn(arr[i], arr[i + 1]);
        if (cmp == .gt) return false;
    }
    return true;
}

// Helper: Fisher-Yates shuffle
fn shuffle(comptime T: type, arr: []T, random: Random) void {
    if (arr.len <= 1) return;

    var i = arr.len;
    while (i > 1) {
        i -= 1;
        const j = random.intRangeLessThan(usize, 0, i + 1);
        std.mem.swap(T, &arr[i], &arr[j]);
    }
}

// ============================================================================
// Tests
// ============================================================================

test "bogo sort - already sorted (best case)" {
    var arr = [_]i32{ 1, 2, 3 };
    try bogoSortAsc(i32, &arr);
    try testing.expectEqual(@as(i32, 1), arr[0]);
    try testing.expectEqual(@as(i32, 2), arr[1]);
    try testing.expectEqual(@as(i32, 3), arr[2]);
}

test "bogo sort - small array ascending" {
    var arr = [_]i32{ 3, 1, 2 };
    const asc = struct {
        fn compare(a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    }.compare;
    try bogoSortBounded(i32, &arr, asc, 10000); // bounded for safety
    try testing.expectEqual(@as(i32, 1), arr[0]);
    try testing.expectEqual(@as(i32, 2), arr[1]);
    try testing.expectEqual(@as(i32, 3), arr[2]);
}

test "bogo sort - small array descending" {
    var arr = [_]i32{ 1, 3, 2 };
    const desc = struct {
        fn compare(a: i32, b: i32) Order {
            return std.math.order(b, a);
        }
    }.compare;
    try bogoSortBounded(i32, &arr, desc, 10000);
    try testing.expectEqual(@as(i32, 3), arr[0]);
    try testing.expectEqual(@as(i32, 2), arr[1]);
    try testing.expectEqual(@as(i32, 1), arr[2]);
}

test "bogo sort - empty array" {
    var arr = [_]i32{};
    try bogoSortAsc(i32, &arr);
    try testing.expectEqual(@as(usize, 0), arr.len);
}

test "bogo sort - single element" {
    var arr = [_]i32{42};
    try bogoSortAsc(i32, &arr);
    try testing.expectEqual(@as(i32, 42), arr[0]);
}

test "bogo sort - two elements" {
    var arr = [_]i32{ 2, 1 };
    const asc = struct {
        fn compare(a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    }.compare;
    try bogoSortBounded(i32, &arr, asc, 1000);
    try testing.expectEqual(@as(i32, 1), arr[0]);
    try testing.expectEqual(@as(i32, 2), arr[1]);
}

test "bogo sort - all equal" {
    var arr = [_]i32{ 5, 5, 5, 5 };
    try bogoSortAsc(i32, &arr);
    for (arr) |val| {
        try testing.expectEqual(@as(i32, 5), val);
    }
}

test "bogo sort - duplicates" {
    var arr = [_]i32{ 3, 1, 2, 1, 3 };
    const asc = struct {
        fn compare(a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    }.compare;
    try bogoSortBounded(i32, &arr, asc, 100000);
    try testing.expectEqual(@as(i32, 1), arr[0]);
    try testing.expectEqual(@as(i32, 1), arr[1]);
    try testing.expectEqual(@as(i32, 2), arr[2]);
    try testing.expectEqual(@as(i32, 3), arr[3]);
    try testing.expectEqual(@as(i32, 3), arr[4]);
}

test "bogo sort - negative numbers" {
    var arr = [_]i32{ -1, -5, 0, -3 };
    const asc = struct {
        fn compare(a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    }.compare;
    try bogoSortBounded(i32, &arr, asc, 100000);
    try testing.expectEqual(@as(i32, -5), arr[0]);
    try testing.expectEqual(@as(i32, -3), arr[1]);
    try testing.expectEqual(@as(i32, -1), arr[2]);
    try testing.expectEqual(@as(i32, 0), arr[3]);
}

test "bogo sort - f64 support" {
    var arr = [_]f64{ 3.14, 1.41, 2.71 };
    const asc = struct {
        fn compare(a: f64, b: f64) Order {
            return std.math.order(a, b);
        }
    }.compare;
    try bogoSortBounded(f64, &arr, asc, 10000);
    try testing.expectApproxEqAbs(@as(f64, 1.41), arr[0], 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 2.71), arr[1], 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 3.14), arr[2], 1e-9);
}

test "bogo sort - custom comparison" {
    const Person = struct {
        age: u32,
        name: []const u8,
    };

    var arr = [_]Person{
        .{ .age = 30, .name = "Alice" },
        .{ .age = 20, .name = "Bob" },
        .{ .age = 25, .name = "Charlie" },
    };

    const compareByAge = struct {
        fn compare(a: Person, b: Person) Order {
            return std.math.order(a.age, b.age);
        }
    }.compare;

    try bogoSortBounded(Person, &arr, compareByAge, 10000);
    try testing.expectEqual(@as(u32, 20), arr[0].age);
    try testing.expectEqual(@as(u32, 25), arr[1].age);
    try testing.expectEqual(@as(u32, 30), arr[2].age);
}

test "bogo sort - Order-based comparison" {
    var arr = [_]i32{ 3, 1, 2 };
    try bogoSortBy(i32, &arr, .lt);
    try testing.expectEqual(@as(i32, 1), arr[0]);
    try testing.expectEqual(@as(i32, 2), arr[1]);
    try testing.expectEqual(@as(i32, 3), arr[2]);
}

test "bogo sort - u8 type" {
    var arr = [_]u8{ 200, 50, 150, 100 };
    const asc = struct {
        fn compare(a: u8, b: u8) Order {
            return std.math.order(a, b);
        }
    }.compare;
    try bogoSortBounded(u8, &arr, asc, 100000);
    try testing.expectEqual(@as(u8, 50), arr[0]);
    try testing.expectEqual(@as(u8, 100), arr[1]);
    try testing.expectEqual(@as(u8, 150), arr[2]);
    try testing.expectEqual(@as(u8, 200), arr[3]);
}

test "bogo sort - reverse sorted input" {
    var arr = [_]i32{ 4, 3, 2, 1 };
    const asc = struct {
        fn compare(a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    }.compare;
    try bogoSortBounded(i32, &arr, asc, 100000);
    try testing.expectEqual(@as(i32, 1), arr[0]);
    try testing.expectEqual(@as(i32, 2), arr[1]);
    try testing.expectEqual(@as(i32, 3), arr[2]);
    try testing.expectEqual(@as(i32, 4), arr[3]);
}

test "bogo sort - deterministic variant small" {
    var arr = [_]i32{ 3, 1, 2 };
    const asc = struct {
        fn compare(a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    }.compare;
    try bogoSortDeterministic(i32, &arr, asc);
    try testing.expectEqual(@as(i32, 1), arr[0]);
    try testing.expectEqual(@as(i32, 2), arr[1]);
    try testing.expectEqual(@as(i32, 3), arr[2]);
}

test "bogo sort - deterministic too large" {
    var arr = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    const asc = struct {
        fn compare(a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    }.compare;
    try testing.expectError(error.ArrayTooLarge, bogoSortDeterministic(i32, &arr, asc));
}

test "bogo sort - bounded max iterations" {
    var arr = [_]i32{ 5, 4, 3, 2, 1 };
    const asc = struct {
        fn compare(a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    }.compare;
    // Very low iteration count should fail
    try testing.expectError(error.MaxIterationsExceeded, bogoSortBounded(i32, &arr, asc, 5));
}

test "bogo sort - iteration counting" {
    var arr = [_]i32{ 2, 1 };
    const asc = struct {
        fn compare(a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    }.compare;
    const iterations = countBogoSortIterations(i32, &arr, asc, 12345);
    // For 2 elements, expected average is 2! = 2, should terminate quickly
    try testing.expect(iterations > 0);
    try testing.expect(iterations < 1000); // Should be very few for 2 elements
}

test "bogo sort - iteration counting already sorted" {
    var arr = [_]i32{ 1, 2, 3 };
    const asc = struct {
        fn compare(a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    }.compare;
    const iterations = countBogoSortIterations(i32, &arr, asc, 12345);
    try testing.expectEqual(@as(usize, 0), iterations); // Already sorted, 0 shuffles
}

test "bogo sort - isSorted helper" {
    const asc = struct {
        fn compare(a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    }.compare;

    const arr1 = [_]i32{ 1, 2, 3, 4 };
    try testing.expect(isSorted(i32, &arr1, asc));

    const arr2 = [_]i32{ 4, 2, 3, 1 };
    try testing.expect(!isSorted(i32, &arr2, asc));

    const arr3 = [_]i32{ 1, 1, 1 };
    try testing.expect(isSorted(i32, &arr3, asc));
}
