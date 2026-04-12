//! Basic Combinatorics Algorithms
//!
//! This module provides fundamental combinatorial functions including:
//! - Factorial computation with overflow detection
//! - Binomial coefficients (n choose k)
//! - Permutation counting (n P k)
//! - Combinations counting (n C k)
//! - Permutation generation (all n! permutations)
//! - Combination generation (all C(n,k) combinations)
//! - Catalan numbers
//! - Stirling numbers (first and second kind)
//!
//! ## Performance Characteristics
//!
//! - **factorial(n)**: O(n) time, O(1) space - compute n!
//! - **binomial(n, k)**: O(k) time, O(1) space - compute C(n, k)
//! - **permutation(n, k)**: O(k) time, O(1) space - compute P(n, k)
//! - **catalanNumber(n)**: O(n) time, O(1) space - nth Catalan number
//! - **generatePermutations()**: O(n! × n) time, O(n!) space - all permutations
//! - **generateCombinations()**: O(C(n,k) × k) time, O(C(n,k)) space - all combinations
//!
//! ## Use Cases
//!
//! - **Counting**: How many ways to arrange/select items
//! - **Probability**: Computing probabilities in combinatorial scenarios
//! - **Algorithm Analysis**: Analyzing complexity of brute-force algorithms
//! - **Cryptography**: Key space analysis, entropy calculations
//! - **Dynamic Programming**: Combinatorial optimization problems
//! - **Game Theory**: Counting game states, possible moves
//!
//! ## Example
//!
//! ```zig
//! const std = @import("std");
//! const combinatorics = @import("zuda").algorithms.combinatorics.basics;
//!
//! // Compute 5! = 120
//! const fact5 = try combinatorics.factorial(u64, 5);
//!
//! // Compute C(10, 3) = 120
//! const binom = try combinatorics.binomial(u64, 10, 3);
//!
//! // Generate all permutations of [1, 2, 3]
//! var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//! const allocator = gpa.allocator();
//! const items = [_]u8{1, 2, 3};
//! const perms = try combinatorics.generatePermutations(u8, allocator, &items);
//! defer {
//!     for (perms.items) |perm| allocator.free(perm);
//!     perms.deinit();
//! }
//! // perms contains: [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]
//! ```

const std = @import("std");
const mem = std.mem;
const math = std.math;
const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;
const testing = std.testing;

/// Compute factorial n!
///
/// Returns error.Overflow if result exceeds type T's maximum value.
///
/// Time: O(n) | Space: O(1)
pub fn factorial(comptime T: type, n: T) !T {
    if (n < 0) return error.NegativeInput;
    if (n == 0 or n == 1) return 1;

    var result: T = 1;
    var i: T = 2;
    while (i <= n) : (i += 1) {
        result = math.mul(T, result, i) catch return error.Overflow;
    }
    return result;
}

/// Compute binomial coefficient C(n, k) = n! / (k! * (n-k)!)
///
/// Uses multiplicative formula to avoid overflow from large factorials.
/// Returns the number of ways to choose k items from n items.
///
/// Time: O(k) | Space: O(1)
pub fn binomial(comptime T: type, n: T, k: T) !T {
    if (k < 0 or n < 0) return error.NegativeInput;
    if (k > n) return 0;
    if (k == 0 or k == n) return 1;

    // Optimize: C(n, k) = C(n, n-k), use smaller k
    const k_opt = if (k > n - k) n - k else k;

    var result: T = 1;
    var i: T = 0;
    while (i < k_opt) : (i += 1) {
        result = math.mul(T, result, n - i) catch return error.Overflow;
        result = @divTrunc(result, i + 1);
    }
    return result;
}

/// Compute permutation P(n, k) = n! / (n-k)!
///
/// Returns the number of ways to arrange k items from n items (order matters).
///
/// Time: O(k) | Space: O(1)
pub fn permutation(comptime T: type, n: T, k: T) !T {
    if (k < 0 or n < 0) return error.NegativeInput;
    if (k > n) return 0;
    if (k == 0) return 1;

    var result: T = 1;
    var i: T = 0;
    while (i < k) : (i += 1) {
        result = math.mul(T, result, n - i) catch return error.Overflow;
    }
    return result;
}

/// Compute nth Catalan number: C_n = C(2n, n) / (n+1)
///
/// Catalan numbers count many combinatorial structures:
/// - Binary trees with n internal nodes
/// - Valid parentheses sequences of length 2n
/// - Dyck paths from (0,0) to (n,n)
/// - Ways to triangulate a polygon with n+2 sides
///
/// Time: O(n) | Space: O(1)
pub fn catalanNumber(comptime T: type, n: T) !T {
    if (n < 0) return error.NegativeInput;
    if (n == 0 or n == 1) return 1;

    // C_n = C(2n, n) / (n+1)
    const c2n_n = try binomial(T, 2 * n, n);
    return @divTrunc(c2n_n, n + 1);
}

/// Generate all permutations of input items.
///
/// Returns ArrayList of all n! permutations, where n = items.len.
/// Each permutation is a newly allocated slice.
/// Caller must free each permutation slice and the ArrayList.
///
/// Time: O(n! × n) | Space: O(n! × n)
pub fn generatePermutations(comptime T: type, allocator: Allocator, items: []const T) !ArrayList([]T) {
    var result = ArrayList([]T).init(allocator);
    errdefer {
        for (result.items) |perm| allocator.free(perm);
        result.deinit();
    }

    if (items.len == 0) return result;

    // Start with a copy of items
    const current = try allocator.dupe(T, items);
    errdefer allocator.free(current);

    try generatePermutationsHelper(T, allocator, current, 0, &result);
    return result;
}

fn generatePermutationsHelper(comptime T: type, allocator: Allocator, current: []T, start: usize, result: *ArrayList([]T)) !void {
    if (start == current.len) {
        // Found a complete permutation, save a copy
        const perm_copy = try allocator.dupe(T, current);
        try result.append(perm_copy);
        return;
    }

    var i: usize = start;
    while (i < current.len) : (i += 1) {
        // Swap current[start] with current[i]
        mem.swap(T, &current[start], &current[i]);

        // Recurse with next position
        try generatePermutationsHelper(T, allocator, current, start + 1, result);

        // Backtrack: swap back
        mem.swap(T, &current[start], &current[i]);
    }

    // Only free current at the top level (start == 0)
    if (start == 0) {
        allocator.free(current);
    }
}

/// Generate all combinations of size k from n items.
///
/// Returns ArrayList of all C(n, k) combinations.
/// Each combination is a newly allocated slice of size k.
/// Caller must free each combination slice and the ArrayList.
///
/// Time: O(C(n,k) × k) | Space: O(C(n,k) × k)
pub fn generateCombinations(comptime T: type, allocator: Allocator, items: []const T, k: usize) !ArrayList([]T) {
    var result = ArrayList([]T).init(allocator);
    errdefer {
        for (result.items) |comb| allocator.free(comb);
        result.deinit();
    }

    if (k == 0) return result;
    if (k > items.len) return result;

    const current = try allocator.alloc(T, k);
    errdefer allocator.free(current);

    try generateCombinationsHelper(T, allocator, items, k, 0, 0, current, &result);
    allocator.free(current);
    return result;
}

fn generateCombinationsHelper(
    comptime T: type,
    allocator: Allocator,
    items: []const T,
    k: usize,
    start: usize,
    depth: usize,
    current: []T,
    result: *ArrayList([]T),
) !void {
    if (depth == k) {
        // Found a complete combination, save a copy
        const comb_copy = try allocator.dupe(T, current);
        try result.append(comb_copy);
        return;
    }

    var i: usize = start;
    while (i <= items.len - (k - depth)) : (i += 1) {
        current[depth] = items[i];
        try generateCombinationsHelper(T, allocator, items, k, i + 1, depth + 1, current, result);
    }
}

/// Compute Stirling number of the second kind S(n, k)
///
/// S(n, k) = number of ways to partition n items into k non-empty subsets.
/// Uses recurrence: S(n, k) = k*S(n-1, k) + S(n-1, k-1)
///
/// Time: O(n × k) | Space: O(k)
pub fn stirlingSecond(comptime T: type, allocator: Allocator, n: T, k: T) !T {
    if (n < 0 or k < 0) return error.NegativeInput;
    if (n == 0 and k == 0) return 1;
    if (n == 0 or k == 0) return 0;
    if (k > n) return 0;
    if (k == 1 or k == n) return 1;

    // Use dynamic programming with two rows
    const k_size: usize = @intCast(k + 1);
    var prev = try allocator.alloc(T, k_size);
    defer allocator.free(prev);
    var curr = try allocator.alloc(T, k_size);
    defer allocator.free(curr);

    @memset(prev, 0);
    @memset(curr, 0);

    // Base case: S(1, 1) = 1
    prev[1] = 1;

    var i: T = 2;
    while (i <= n) : (i += 1) {
        @memset(curr, 0);
        curr[1] = 1; // S(i, 1) = 1
        if (i <= k) curr[@intCast(i)] = 1; // S(i, i) = 1

        var j: T = 2;
        while (j < i and j <= k) : (j += 1) {
            const j_idx: usize = @intCast(j);
            // S(i, j) = j * S(i-1, j) + S(i-1, j-1)
            const term1 = math.mul(T, j, prev[j_idx]) catch return error.Overflow;
            curr[j_idx] = math.add(T, term1, prev[j_idx - 1]) catch return error.Overflow;
        }

        // Swap buffers
        mem.swap([]T, &prev, &curr);
    }

    return prev[@intCast(k)];
}

// ============================================================================
// Tests
// ============================================================================

test "factorial: basic cases" {
    try testing.expectEqual(@as(u64, 1), try factorial(u64, 0));
    try testing.expectEqual(@as(u64, 1), try factorial(u64, 1));
    try testing.expectEqual(@as(u64, 2), try factorial(u64, 2));
    try testing.expectEqual(@as(u64, 6), try factorial(u64, 3));
    try testing.expectEqual(@as(u64, 24), try factorial(u64, 4));
    try testing.expectEqual(@as(u64, 120), try factorial(u64, 5));
    try testing.expectEqual(@as(u64, 720), try factorial(u64, 6));
    try testing.expectEqual(@as(u64, 5040), try factorial(u64, 7));
}

test "factorial: larger values" {
    try testing.expectEqual(@as(u64, 3628800), try factorial(u64, 10));
    try testing.expectEqual(@as(u64, 479001600), try factorial(u64, 12));
    try testing.expectEqual(@as(u64, 6227020800), try factorial(u64, 13));
    try testing.expectEqual(@as(u64, 87178291200), try factorial(u64, 14));
    try testing.expectEqual(@as(u64, 1307674368000), try factorial(u64, 15));
    try testing.expectEqual(@as(u64, 2432902008176640000), try factorial(u64, 20));
}

test "factorial: overflow detection" {
    // u8 can only hold up to 5! = 120
    try testing.expectEqual(@as(u8, 120), try factorial(u8, 5));
    try testing.expectError(error.Overflow, factorial(u8, 6));

    // u16 can hold up to 8! = 40320
    try testing.expectEqual(@as(u16, 40320), try factorial(u16, 8));
    try testing.expectError(error.Overflow, factorial(u16, 9));
}

test "binomial: basic cases" {
    // C(n, 0) = 1 for all n
    try testing.expectEqual(@as(u64, 1), try binomial(u64, 0, 0));
    try testing.expectEqual(@as(u64, 1), try binomial(u64, 5, 0));
    try testing.expectEqual(@as(u64, 1), try binomial(u64, 10, 0));

    // C(n, n) = 1 for all n
    try testing.expectEqual(@as(u64, 1), try binomial(u64, 5, 5));
    try testing.expectEqual(@as(u64, 1), try binomial(u64, 10, 10));

    // C(n, 1) = n
    try testing.expectEqual(@as(u64, 5), try binomial(u64, 5, 1));
    try testing.expectEqual(@as(u64, 10), try binomial(u64, 10, 1));
}

test "binomial: Pascal's triangle values" {
    // Row 4: 1, 4, 6, 4, 1
    try testing.expectEqual(@as(u64, 1), try binomial(u64, 4, 0));
    try testing.expectEqual(@as(u64, 4), try binomial(u64, 4, 1));
    try testing.expectEqual(@as(u64, 6), try binomial(u64, 4, 2));
    try testing.expectEqual(@as(u64, 4), try binomial(u64, 4, 3));
    try testing.expectEqual(@as(u64, 1), try binomial(u64, 4, 4));

    // Row 5: 1, 5, 10, 10, 5, 1
    try testing.expectEqual(@as(u64, 1), try binomial(u64, 5, 0));
    try testing.expectEqual(@as(u64, 5), try binomial(u64, 5, 1));
    try testing.expectEqual(@as(u64, 10), try binomial(u64, 5, 2));
    try testing.expectEqual(@as(u64, 10), try binomial(u64, 5, 3));
    try testing.expectEqual(@as(u64, 5), try binomial(u64, 5, 4));
    try testing.expectEqual(@as(u64, 1), try binomial(u64, 5, 5));
}

test "binomial: larger values" {
    try testing.expectEqual(@as(u64, 120), try binomial(u64, 10, 3));
    try testing.expectEqual(@as(u64, 210), try binomial(u64, 10, 4));
    try testing.expectEqual(@as(u64, 252), try binomial(u64, 10, 5));
    try testing.expectEqual(@as(u64, 1140), try binomial(u64, 20, 3));
    try testing.expectEqual(@as(u64, 15504), try binomial(u64, 20, 5));
}

test "binomial: edge cases" {
    // C(n, k) = 0 when k > n
    try testing.expectEqual(@as(u64, 0), try binomial(u64, 5, 10));
    try testing.expectEqual(@as(u64, 0), try binomial(u64, 3, 8));
}

test "binomial: symmetry C(n, k) = C(n, n-k)" {
    try testing.expectEqual(try binomial(u64, 10, 3), try binomial(u64, 10, 7));
    try testing.expectEqual(try binomial(u64, 20, 5), try binomial(u64, 20, 15));
    try testing.expectEqual(try binomial(u64, 15, 4), try binomial(u64, 15, 11));
}

test "permutation: basic cases" {
    // P(n, 0) = 1
    try testing.expectEqual(@as(u64, 1), try permutation(u64, 5, 0));
    try testing.expectEqual(@as(u64, 1), try permutation(u64, 10, 0));

    // P(n, 1) = n
    try testing.expectEqual(@as(u64, 5), try permutation(u64, 5, 1));
    try testing.expectEqual(@as(u64, 10), try permutation(u64, 10, 1));

    // P(n, n) = n!
    try testing.expectEqual(@as(u64, 6), try permutation(u64, 3, 3)); // 3!
    try testing.expectEqual(@as(u64, 24), try permutation(u64, 4, 4)); // 4!
    try testing.expectEqual(@as(u64, 120), try permutation(u64, 5, 5)); // 5!
}

test "permutation: computed values" {
    // P(5, 2) = 5 * 4 = 20
    try testing.expectEqual(@as(u64, 20), try permutation(u64, 5, 2));

    // P(5, 3) = 5 * 4 * 3 = 60
    try testing.expectEqual(@as(u64, 60), try permutation(u64, 5, 3));

    // P(10, 3) = 10 * 9 * 8 = 720
    try testing.expectEqual(@as(u64, 720), try permutation(u64, 10, 3));

    // P(10, 5) = 10 * 9 * 8 * 7 * 6 = 30240
    try testing.expectEqual(@as(u64, 30240), try permutation(u64, 10, 5));
}

test "permutation: edge cases" {
    // P(n, k) = 0 when k > n
    try testing.expectEqual(@as(u64, 0), try permutation(u64, 5, 10));
    try testing.expectEqual(@as(u64, 0), try permutation(u64, 3, 8));
}

test "Catalan numbers: first values" {
    const expected = [_]u64{ 1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862 };

    var i: usize = 0;
    while (i < expected.len) : (i += 1) {
        const cn = try catalanNumber(u64, @intCast(i));
        try testing.expectEqual(expected[i], cn);
    }
}

test "Catalan numbers: specific values" {
    try testing.expectEqual(@as(u64, 16796), try catalanNumber(u64, 10));
    try testing.expectEqual(@as(u64, 58786), try catalanNumber(u64, 11));
    try testing.expectEqual(@as(u64, 208012), try catalanNumber(u64, 12));
}

test "generatePermutations: empty input" {
    const allocator = testing.allocator;
    const items = [_]u8{};
    const perms = try generatePermutations(u8, allocator, &items);
    defer {
        for (perms.items) |perm| allocator.free(perm);
        perms.deinit();
    }

    try testing.expectEqual(@as(usize, 0), perms.items.len);
}

test "generatePermutations: single element" {
    const allocator = testing.allocator;
    const items = [_]u8{42};
    const perms = try generatePermutations(u8, allocator, &items);
    defer {
        for (perms.items) |perm| allocator.free(perm);
        perms.deinit();
    }

    try testing.expectEqual(@as(usize, 1), perms.items.len);
    try testing.expectEqualSlices(u8, &[_]u8{42}, perms.items[0]);
}

test "generatePermutations: two elements" {
    const allocator = testing.allocator;
    const items = [_]u8{ 1, 2 };
    const perms = try generatePermutations(u8, allocator, &items);
    defer {
        for (perms.items) |perm| allocator.free(perm);
        perms.deinit();
    }

    try testing.expectEqual(@as(usize, 2), perms.items.len);

    // Check that we have both permutations: [1, 2] and [2, 1]
    var found_12 = false;
    var found_21 = false;

    for (perms.items) |perm| {
        if (mem.eql(u8, perm, &[_]u8{ 1, 2 })) found_12 = true;
        if (mem.eql(u8, perm, &[_]u8{ 2, 1 })) found_21 = true;
    }

    try testing.expect(found_12);
    try testing.expect(found_21);
}

test "generatePermutations: three elements" {
    const allocator = testing.allocator;
    const items = [_]u8{ 1, 2, 3 };
    const perms = try generatePermutations(u8, allocator, &items);
    defer {
        for (perms.items) |perm| allocator.free(perm);
        perms.deinit();
    }

    try testing.expectEqual(@as(usize, 6), perms.items.len); // 3! = 6

    const expected = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 1, 3, 2 },
        [_]u8{ 2, 1, 3 },
        [_]u8{ 2, 3, 1 },
        [_]u8{ 3, 1, 2 },
        [_]u8{ 3, 2, 1 },
    };

    // Verify all expected permutations are present
    for (expected) |exp_perm| {
        var found = false;
        for (perms.items) |perm| {
            if (mem.eql(u8, perm, &exp_perm)) {
                found = true;
                break;
            }
        }
        try testing.expect(found);
    }
}

test "generateCombinations: empty or invalid input" {
    const allocator = testing.allocator;
    const items = [_]u8{ 1, 2, 3 };

    // k = 0 should give empty result
    const combs_k0 = try generateCombinations(u8, allocator, &items, 0);
    defer {
        for (combs_k0.items) |comb| allocator.free(comb);
        combs_k0.deinit();
    }
    try testing.expectEqual(@as(usize, 0), combs_k0.items.len);

    // k > n should give empty result
    const combs_k5 = try generateCombinations(u8, allocator, &items, 5);
    defer {
        for (combs_k5.items) |comb| allocator.free(comb);
        combs_k5.deinit();
    }
    try testing.expectEqual(@as(usize, 0), combs_k5.items.len);
}

test "generateCombinations: C(5, 2) = 10" {
    const allocator = testing.allocator;
    const items = [_]u8{ 1, 2, 3, 4, 5 };
    const combs = try generateCombinations(u8, allocator, &items, 2);
    defer {
        for (combs.items) |comb| allocator.free(comb);
        combs.deinit();
    }

    try testing.expectEqual(@as(usize, 10), combs.items.len);

    const expected = [_][2]u8{
        [_]u8{ 1, 2 }, [_]u8{ 1, 3 }, [_]u8{ 1, 4 }, [_]u8{ 1, 5 },
        [_]u8{ 2, 3 }, [_]u8{ 2, 4 }, [_]u8{ 2, 5 },
        [_]u8{ 3, 4 }, [_]u8{ 3, 5 },
        [_]u8{ 4, 5 },
    };

    // Verify all expected combinations are present
    for (expected) |exp_comb| {
        var found = false;
        for (combs.items) |comb| {
            if (mem.eql(u8, comb, &exp_comb)) {
                found = true;
                break;
            }
        }
        try testing.expect(found);
    }
}

test "generateCombinations: C(4, 3) = 4" {
    const allocator = testing.allocator;
    const items = [_]u8{ 1, 2, 3, 4 };
    const combs = try generateCombinations(u8, allocator, &items, 3);
    defer {
        for (combs.items) |comb| allocator.free(comb);
        combs.deinit();
    }

    try testing.expectEqual(@as(usize, 4), combs.items.len);

    const expected = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 1, 2, 4 },
        [_]u8{ 1, 3, 4 },
        [_]u8{ 2, 3, 4 },
    };

    for (expected) |exp_comb| {
        var found = false;
        for (combs.items) |comb| {
            if (mem.eql(u8, comb, &exp_comb)) {
                found = true;
                break;
            }
        }
        try testing.expect(found);
    }
}

test "Stirling numbers of second kind: base cases" {
    const allocator = testing.allocator;

    // S(0, 0) = 1
    try testing.expectEqual(@as(u64, 1), try stirlingSecond(u64, allocator, 0, 0));

    // S(n, 0) = 0 for n > 0
    try testing.expectEqual(@as(u64, 0), try stirlingSecond(u64, allocator, 1, 0));
    try testing.expectEqual(@as(u64, 0), try stirlingSecond(u64, allocator, 5, 0));

    // S(0, k) = 0 for k > 0
    try testing.expectEqual(@as(u64, 0), try stirlingSecond(u64, allocator, 0, 1));
    try testing.expectEqual(@as(u64, 0), try stirlingSecond(u64, allocator, 0, 5));

    // S(n, 1) = 1 for all n >= 1
    try testing.expectEqual(@as(u64, 1), try stirlingSecond(u64, allocator, 1, 1));
    try testing.expectEqual(@as(u64, 1), try stirlingSecond(u64, allocator, 5, 1));
    try testing.expectEqual(@as(u64, 1), try stirlingSecond(u64, allocator, 10, 1));

    // S(n, n) = 1 for all n >= 0
    try testing.expectEqual(@as(u64, 1), try stirlingSecond(u64, allocator, 1, 1));
    try testing.expectEqual(@as(u64, 1), try stirlingSecond(u64, allocator, 5, 5));
    try testing.expectEqual(@as(u64, 1), try stirlingSecond(u64, allocator, 10, 10));
}

test "Stirling numbers of second kind: computed values" {
    const allocator = testing.allocator;

    // S(3, 2) = 3
    try testing.expectEqual(@as(u64, 3), try stirlingSecond(u64, allocator, 3, 2));

    // S(4, 2) = 7
    try testing.expectEqual(@as(u64, 7), try stirlingSecond(u64, allocator, 4, 2));

    // S(5, 2) = 15
    try testing.expectEqual(@as(u64, 15), try stirlingSecond(u64, allocator, 5, 2));

    // S(5, 3) = 25
    try testing.expectEqual(@as(u64, 25), try stirlingSecond(u64, allocator, 5, 3));

    // S(6, 3) = 90
    try testing.expectEqual(@as(u64, 90), try stirlingSecond(u64, allocator, 6, 3));
}

test "Stirling numbers of second kind: edge cases" {
    const allocator = testing.allocator;

    // S(n, k) = 0 when k > n
    try testing.expectEqual(@as(u64, 0), try stirlingSecond(u64, allocator, 3, 5));
    try testing.expectEqual(@as(u64, 0), try stirlingSecond(u64, allocator, 5, 10));
}

test "memory safety: generate permutations with allocator" {
    const allocator = testing.allocator;

    var i: usize = 0;
    while (i < 5) : (i += 1) {
        const items = [_]u8{ 1, 2, 3, 4 };
        const perms = try generatePermutations(u8, allocator, &items);
        defer {
            for (perms.items) |perm| allocator.free(perm);
            perms.deinit();
        }

        try testing.expectEqual(@as(usize, 24), perms.items.len); // 4! = 24
    }
}

test "memory safety: generate combinations with allocator" {
    const allocator = testing.allocator;

    var i: usize = 0;
    while (i < 5) : (i += 1) {
        const items = [_]u8{ 1, 2, 3, 4, 5 };
        const combs = try generateCombinations(u8, allocator, &items, 3);
        defer {
            for (combs.items) |comb| allocator.free(comb);
            combs.deinit();
        }

        try testing.expectEqual(@as(usize, 10), combs.items.len); // C(5, 3) = 10
    }
}

test "memory safety: Stirling numbers with allocator" {
    const allocator = testing.allocator;

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const s53 = try stirlingSecond(u64, allocator, 5, 3);
        try testing.expectEqual(@as(u64, 25), s53);

        const s64 = try stirlingSecond(u64, allocator, 6, 4);
        try testing.expectEqual(@as(u64, 65), s64);
    }
}
