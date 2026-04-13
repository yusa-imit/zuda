//! Stirling Numbers — Combinatorial Coefficients
//!
//! Stirling numbers are fundamental combinatorial sequences with two kinds:
//!
//! **Stirling Numbers of the First Kind (unsigned)**: S(n, k)
//! - Count permutations of n elements with exactly k cycles
//! - Example: S(4, 2) = 11 (permutations of 4 elements with 2 cycles)
//! - Recurrence: S(n, k) = S(n-1, k-1) + (n-1)·S(n-1, k)
//! - Use cases: Permutation cycles, harmonic number coefficients
//!
//! **Stirling Numbers of the Second Kind**: s(n, k)
//! - Count ways to partition n elements into k non-empty subsets
//! - Example: s(4, 2) = 7 (partitions of {1,2,3,4} into 2 subsets)
//! - Recurrence: s(n, k) = s(n-1, k-1) + k·s(n-1, k)
//! - Use cases: Set partitions, surjective functions, Bell numbers (Σ_k s(n,k))
//!
//! **Signed Stirling Numbers of the First Kind**: s(n, k)
//! - Signed version: s(n, k) = (-1)^(n-k) × S(n, k)
//! - Related to falling factorials: x^(n) = Σ_k s(n,k) × x^k
//! - Use cases: Factorial expansion, generating function coefficients
//!
//! ## API Functions
//! - **stirlingFirst(n, k)**: O(n×k) time, O(k) space - unsigned first kind
//! - **stirlingFirstSigned(n, k)**: O(n×k) time, O(k) space - signed first kind
//! - **stirlingSecond(n, k)**: O(n×k) time, O(k) space - second kind
//! - **generateStirlingFirstRow(n)**: O(n²) time, O(n) space - all S(n, k) for k=0..n
//! - **generateStirlingSecondRow(n)**: O(n²) time, O(n) space - all s(n, k) for k=0..n
//! - **generateStirlingFirstTriangle(n)**: O(n³) time, O(n²) space - full triangle S(i,j)
//! - **generateStirlingSecondTriangle(n)**: O(n³) time, O(n²) space - full triangle s(i,j)
//!
//! ## Use Cases
//! - **Permutation cycle structure**: S(n, k) - cryptography, group theory
//! - **Set partitions**: s(n, k) - Bell numbers, combinatorial enumeration
//! - **Factorial expansions**: Signed S(n, k) - symbolic algebra, calculus
//! - **Surjective functions**: s(n, k) counts onto mappings
//!
//! ## Properties
//! - S(0, 0) = s(0, 0) = 1 (empty permutation/partition)
//! - S(n, 0) = s(n, 0) = 0 for n > 0
//! - S(n, n) = s(n, n) = 1 (identity permutation / n singleton sets)
//! - S(n, 1) = (n-1)! (single cycle)
//! - s(n, 1) = 1 (all elements in one set)
//! - s(n, 2) = 2^(n-1) - 1 (partition into 2 non-empty sets)
//! - Bell number B(n) = Σ_k s(n, k) (total partitions)
//!
//! ## References
//! - *Concrete Mathematics* (Graham, Knuth, Patashnik) - Chapter 6
//! - OEIS A008275 (signed first kind), A048993 (unsigned first kind), A008277 (second kind)
//! - Stanley *Enumerative Combinatorics* - Volume 1, Chapter 1

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Compute Stirling number of the first kind (unsigned): S(n, k)
///
/// Counts permutations of n elements with exactly k cycles.
/// Uses dynamic programming with space optimization (only stores previous row).
///
/// Recurrence: S(n, k) = S(n-1, k-1) + (n-1)·S(n-1, k)
/// Base cases:
///   - S(0, 0) = 1
///   - S(n, 0) = 0 for n > 0
///   - S(0, k) = 0 for k > 0
///
/// Time: O(n × k)
/// Space: O(k)
///
/// Example:
/// ```zig
/// const s = stirlingFirst(u64, 4, 2, allocator); // S(4,2) = 11
/// ```
pub fn stirlingFirst(comptime T: type, n: usize, k: usize, allocator: Allocator) !T {
    if (n == 0 and k == 0) return 1;
    if (n == 0 or k == 0) return 0;
    if (k > n) return 0;
    if (k == n) return 1;

    // DP: only store current and previous row
    var prev = try allocator.alloc(T, k + 1);
    defer allocator.free(prev);
    var curr = try allocator.alloc(T, k + 1);
    defer allocator.free(curr);

    // Base case: row 0
    @memset(prev, 0);
    prev[0] = 1;

    // Fill rows 1..n
    for (1..n + 1) |i| {
        @memset(curr, 0);
        for (1..@min(i + 1, k + 1)) |j| {
            // S(i, j) = S(i-1, j-1) + (i-1) × S(i-1, j)
            curr[j] = prev[j - 1];
            if (j <= i - 1 and i > 1) {
                curr[j] += @as(T, @intCast(i - 1)) * prev[j];
            }
        }
        // Swap buffers
        const tmp = prev;
        prev = curr;
        curr = tmp;
    }

    return prev[k];
}

/// Compute Stirling number of the first kind (signed): s(n, k) = (-1)^(n-k) × S(n, k)
///
/// Signed version used in falling factorial expansion:
/// x(x-1)(x-2)...(x-n+1) = Σ_k s(n,k) × x^k
///
/// Time: O(n × k)
/// Space: O(k)
///
/// Example:
/// ```zig
/// const s = try stirlingFirstSigned(i64, 3, 1, allocator); // -2
/// ```
pub fn stirlingFirstSigned(comptime T: type, n: usize, k: usize, allocator: Allocator) !T {
    const unsigned_value = try stirlingFirst(T, n, k, allocator);
    // Sign: (-1)^(n-k)
    const exponent = n - k;
    const sign: T = if (exponent % 2 == 0) 1 else -1;
    return sign * unsigned_value;
}

/// Compute Stirling number of the second kind: s(n, k)
///
/// Counts ways to partition n elements into k non-empty subsets.
/// Uses dynamic programming with space optimization.
///
/// Recurrence: s(n, k) = s(n-1, k-1) + k·s(n-1, k)
/// Base cases:
///   - s(0, 0) = 1
///   - s(n, 0) = 0 for n > 0
///   - s(0, k) = 0 for k > 0
///   - s(n, n) = 1
///
/// Time: O(n × k)
/// Space: O(k)
///
/// Example:
/// ```zig
/// const s = try stirlingSecond(u64, 4, 2, allocator); // 7
/// ```
pub fn stirlingSecond(comptime T: type, n: usize, k: usize, allocator: Allocator) !T {
    if (n == 0 and k == 0) return 1;
    if (n == 0 or k == 0) return 0;
    if (k > n) return 0;
    if (k == n) return 1;

    var prev = try allocator.alloc(T, k + 1);
    defer allocator.free(prev);
    var curr = try allocator.alloc(T, k + 1);
    defer allocator.free(curr);

    @memset(prev, 0);
    prev[0] = 1;

    for (1..n + 1) |i| {
        @memset(curr, 0);
        for (1..@min(i + 1, k + 1)) |j| {
            // s(i, j) = s(i-1, j-1) + j × s(i-1, j)
            curr[j] = prev[j - 1];
            if (j <= i - 1) {
                curr[j] += @as(T, @intCast(j)) * prev[j];
            }
        }
        const tmp = prev;
        prev = curr;
        curr = tmp;
    }

    return prev[k];
}

/// Generate full row of Stirling numbers of the first kind (unsigned): [S(n,0), S(n,1), ..., S(n,n)]
///
/// Returns all S(n, k) for k = 0 to n.
///
/// Time: O(n²)
/// Space: O(n)
///
/// Example:
/// ```zig
/// const row = try generateStirlingFirstRow(u64, 4, allocator);
/// defer allocator.free(row);
/// // row = [0, 6, 11, 6, 1] for n=4
/// ```
pub fn generateStirlingFirstRow(comptime T: type, n: usize, allocator: Allocator) ![]T {
    var result = try allocator.alloc(T, n + 1);
    errdefer allocator.free(result);

    if (n == 0) {
        result[0] = 1;
        return result;
    }

    var prev = try allocator.alloc(T, n + 1);
    defer allocator.free(prev);

    @memset(prev, 0);
    prev[0] = 1;

    for (1..n + 1) |i| {
        @memset(result, 0);
        for (1..i + 1) |j| {
            result[j] = prev[j - 1];
            if (j <= i - 1 and i > 1) {
                result[j] += @as(T, @intCast(i - 1)) * prev[j];
            }
        }
        @memcpy(prev, result);
    }

    return result;
}

/// Generate full row of Stirling numbers of the second kind: [s(n,0), s(n,1), ..., s(n,n)]
///
/// Returns all s(n, k) for k = 0 to n.
///
/// Time: O(n²)
/// Space: O(n)
///
/// Example:
/// ```zig
/// const row = try generateStirlingSecondRow(u64, 4, allocator);
/// defer allocator.free(row);
/// // row = [0, 1, 7, 6, 1] for n=4
/// ```
pub fn generateStirlingSecondRow(comptime T: type, n: usize, allocator: Allocator) ![]T {
    var result = try allocator.alloc(T, n + 1);
    errdefer allocator.free(result);

    if (n == 0) {
        result[0] = 1;
        return result;
    }

    var prev = try allocator.alloc(T, n + 1);
    defer allocator.free(prev);

    @memset(prev, 0);
    prev[0] = 1;

    for (1..n + 1) |i| {
        @memset(result, 0);
        for (1..i + 1) |j| {
            result[j] = prev[j - 1];
            if (j <= i - 1) {
                result[j] += @as(T, @intCast(j)) * prev[j];
            }
        }
        @memcpy(prev, result);
    }

    return result;
}

/// Generate full triangle of Stirling numbers of the first kind (unsigned)
///
/// Returns 2D array where triangle[i][j] = S(i, j) for 0 ≤ i, j ≤ n.
/// Stored as flat array in row-major order.
///
/// Time: O(n³)
/// Space: O(n²)
///
/// Example:
/// ```zig
/// const tri = try generateStirlingFirstTriangle(u64, 4, allocator);
/// defer allocator.free(tri);
/// // Access S(3, 2) = tri[3 * 5 + 2] = 3
/// ```
pub fn generateStirlingFirstTriangle(comptime T: type, n: usize, allocator: Allocator) ![]T {
    const size = (n + 1) * (n + 1);
    var triangle = try allocator.alloc(T, size);
    errdefer allocator.free(triangle);

    @memset(triangle, 0);

    // Row 0: S(0, 0) = 1
    triangle[0] = 1;

    // Fill rows 1..n
    for (1..n + 1) |i| {
        const row_offset = i * (n + 1);
        const prev_row_offset = (i - 1) * (n + 1);

        for (1..i + 1) |j| {
            triangle[row_offset + j] = triangle[prev_row_offset + j - 1];
            if (j <= i - 1 and i > 1) {
                triangle[row_offset + j] += @as(T, @intCast(i - 1)) * triangle[prev_row_offset + j];
            }
        }
    }

    return triangle;
}

/// Generate full triangle of Stirling numbers of the second kind
///
/// Returns 2D array where triangle[i][j] = s(i, j) for 0 ≤ i, j ≤ n.
/// Stored as flat array in row-major order.
///
/// Time: O(n³)
/// Space: O(n²)
///
/// Example:
/// ```zig
/// const tri = try generateStirlingSecondTriangle(u64, 4, allocator);
/// defer allocator.free(tri);
/// // Access s(3, 2) = tri[3 * 5 + 2] = 3
/// ```
pub fn generateStirlingSecondTriangle(comptime T: type, n: usize, allocator: Allocator) ![]T {
    const size = (n + 1) * (n + 1);
    var triangle = try allocator.alloc(T, size);
    errdefer allocator.free(triangle);

    @memset(triangle, 0);

    triangle[0] = 1;

    for (1..n + 1) |i| {
        const row_offset = i * (n + 1);
        const prev_row_offset = (i - 1) * (n + 1);

        for (1..i + 1) |j| {
            triangle[row_offset + j] = triangle[prev_row_offset + j - 1];
            if (j <= i - 1) {
                triangle[row_offset + j] += @as(T, @intCast(j)) * triangle[prev_row_offset + j];
            }
        }
    }

    return triangle;
}

// ============================================================================
// Tests
// ============================================================================

test "stirlingFirst: base cases" {
    const allocator = testing.allocator;

    // S(0, 0) = 1
    try testing.expectEqual(@as(u64, 1), try stirlingFirst(u64, 0, 0, allocator));

    // S(n, 0) = 0 for n > 0
    try testing.expectEqual(@as(u64, 0), try stirlingFirst(u64, 1, 0, allocator));
    try testing.expectEqual(@as(u64, 0), try stirlingFirst(u64, 5, 0, allocator));

    // S(0, k) = 0 for k > 0
    try testing.expectEqual(@as(u64, 0), try stirlingFirst(u64, 0, 1, allocator));
    try testing.expectEqual(@as(u64, 0), try stirlingFirst(u64, 0, 5, allocator));

    // S(n, n) = 1
    try testing.expectEqual(@as(u64, 1), try stirlingFirst(u64, 3, 3, allocator));
    try testing.expectEqual(@as(u64, 1), try stirlingFirst(u64, 7, 7, allocator));

    // S(n, k) = 0 for k > n
    try testing.expectEqual(@as(u64, 0), try stirlingFirst(u64, 3, 5, allocator));
}

test "stirlingFirst: small values (OEIS A048993)" {
    const allocator = testing.allocator;

    // S(3, 1) = (3-1)! = 2
    try testing.expectEqual(@as(u64, 2), try stirlingFirst(u64, 3, 1, allocator));

    // S(3, 2) = 3
    try testing.expectEqual(@as(u64, 3), try stirlingFirst(u64, 3, 2, allocator));

    // S(4, 1) = (4-1)! = 6
    try testing.expectEqual(@as(u64, 6), try stirlingFirst(u64, 4, 1, allocator));

    // S(4, 2) = 11
    try testing.expectEqual(@as(u64, 11), try stirlingFirst(u64, 4, 2, allocator));

    // S(4, 3) = 6
    try testing.expectEqual(@as(u64, 6), try stirlingFirst(u64, 4, 3, allocator));

    // S(5, 2) = 50
    try testing.expectEqual(@as(u64, 50), try stirlingFirst(u64, 5, 2, allocator));

    // S(5, 3) = 35
    try testing.expectEqual(@as(u64, 35), try stirlingFirst(u64, 5, 3, allocator));
}

test "stirlingFirstSigned: basic values (OEIS A008275)" {
    const allocator = testing.allocator;

    // s(0, 0) = 1
    try testing.expectEqual(@as(i64, 1), try stirlingFirstSigned(i64, 0, 0, allocator));

    // s(3, 1) = -2 (sign = (-1)^(3-1) = 1, magnitude = 2)
    try testing.expectEqual(@as(i64, 2), try stirlingFirstSigned(i64, 3, 1, allocator));

    // s(3, 2) = 3 (sign = (-1)^(3-2) = -1, magnitude = 3)
    try testing.expectEqual(@as(i64, -3), try stirlingFirstSigned(i64, 3, 2, allocator));

    // s(4, 2) = 11 (sign = (-1)^(4-2) = 1)
    try testing.expectEqual(@as(i64, 11), try stirlingFirstSigned(i64, 4, 2, allocator));

    // s(4, 3) = -6 (sign = (-1)^(4-3) = -1)
    try testing.expectEqual(@as(i64, -6), try stirlingFirstSigned(i64, 4, 3, allocator));
}

test "stirlingSecond: base cases" {
    const allocator = testing.allocator;

    // s(0, 0) = 1
    try testing.expectEqual(@as(u64, 1), try stirlingSecond(u64, 0, 0, allocator));

    // s(n, 0) = 0 for n > 0
    try testing.expectEqual(@as(u64, 0), try stirlingSecond(u64, 1, 0, allocator));
    try testing.expectEqual(@as(u64, 0), try stirlingSecond(u64, 5, 0, allocator));

    // s(0, k) = 0 for k > 0
    try testing.expectEqual(@as(u64, 0), try stirlingSecond(u64, 0, 1, allocator));
    try testing.expectEqual(@as(u64, 0), try stirlingSecond(u64, 0, 5, allocator));

    // s(n, n) = 1
    try testing.expectEqual(@as(u64, 1), try stirlingSecond(u64, 3, 3, allocator));
    try testing.expectEqual(@as(u64, 1), try stirlingSecond(u64, 7, 7, allocator));

    // s(n, k) = 0 for k > n
    try testing.expectEqual(@as(u64, 0), try stirlingSecond(u64, 3, 5, allocator));
}

test "stirlingSecond: small values (OEIS A008277)" {
    const allocator = testing.allocator;

    // s(n, 1) = 1 (all elements in one set)
    try testing.expectEqual(@as(u64, 1), try stirlingSecond(u64, 3, 1, allocator));
    try testing.expectEqual(@as(u64, 1), try stirlingSecond(u64, 5, 1, allocator));

    // s(3, 2) = 3 (partitions: {{1,2},{3}}, {{1,3},{2}}, {{1},{2,3}})
    try testing.expectEqual(@as(u64, 3), try stirlingSecond(u64, 3, 2, allocator));

    // s(4, 2) = 7
    try testing.expectEqual(@as(u64, 7), try stirlingSecond(u64, 4, 2, allocator));

    // s(4, 3) = 6
    try testing.expectEqual(@as(u64, 6), try stirlingSecond(u64, 4, 3, allocator));

    // s(5, 2) = 15
    try testing.expectEqual(@as(u64, 15), try stirlingSecond(u64, 5, 2, allocator));

    // s(5, 3) = 25
    try testing.expectEqual(@as(u64, 25), try stirlingSecond(u64, 5, 3, allocator));
}

test "stirlingSecond: formula s(n, 2) = 2^(n-1) - 1" {
    const allocator = testing.allocator;

    // s(3, 2) = 2^2 - 1 = 3
    try testing.expectEqual(@as(u64, 3), try stirlingSecond(u64, 3, 2, allocator));

    // s(4, 2) = 2^3 - 1 = 7
    try testing.expectEqual(@as(u64, 7), try stirlingSecond(u64, 4, 2, allocator));

    // s(5, 2) = 2^4 - 1 = 15
    try testing.expectEqual(@as(u64, 15), try stirlingSecond(u64, 5, 2, allocator));

    // s(6, 2) = 2^5 - 1 = 31
    try testing.expectEqual(@as(u64, 31), try stirlingSecond(u64, 6, 2, allocator));
}

test "generateStirlingFirstRow: n=4" {
    const allocator = testing.allocator;

    const row = try generateStirlingFirstRow(u64, 4, allocator);
    defer allocator.free(row);

    // [S(4,0), S(4,1), S(4,2), S(4,3), S(4,4)] = [0, 6, 11, 6, 1]
    try testing.expectEqual(@as(usize, 5), row.len);
    try testing.expectEqual(@as(u64, 0), row[0]);
    try testing.expectEqual(@as(u64, 6), row[1]);
    try testing.expectEqual(@as(u64, 11), row[2]);
    try testing.expectEqual(@as(u64, 6), row[3]);
    try testing.expectEqual(@as(u64, 1), row[4]);
}

test "generateStirlingSecondRow: n=4" {
    const allocator = testing.allocator;

    const row = try generateStirlingSecondRow(u64, 4, allocator);
    defer allocator.free(row);

    // [s(4,0), s(4,1), s(4,2), s(4,3), s(4,4)] = [0, 1, 7, 6, 1]
    try testing.expectEqual(@as(usize, 5), row.len);
    try testing.expectEqual(@as(u64, 0), row[0]);
    try testing.expectEqual(@as(u64, 1), row[1]);
    try testing.expectEqual(@as(u64, 7), row[2]);
    try testing.expectEqual(@as(u64, 6), row[3]);
    try testing.expectEqual(@as(u64, 1), row[4]);
}

test "generateStirlingFirstTriangle: n=4" {
    const allocator = testing.allocator;

    const tri = try generateStirlingFirstTriangle(u64, 4, allocator);
    defer allocator.free(tri);

    const n: usize = 4;
    const stride = n + 1;

    // S(0, 0) = 1
    try testing.expectEqual(@as(u64, 1), tri[0 * stride + 0]);

    // S(3, 2) = 3
    try testing.expectEqual(@as(u64, 3), tri[3 * stride + 2]);

    // S(4, 2) = 11
    try testing.expectEqual(@as(u64, 11), tri[4 * stride + 2]);

    // S(4, 3) = 6
    try testing.expectEqual(@as(u64, 6), tri[4 * stride + 3]);
}

test "generateStirlingSecondTriangle: n=4" {
    const allocator = testing.allocator;

    const tri = try generateStirlingSecondTriangle(u64, 4, allocator);
    defer allocator.free(tri);

    const n: usize = 4;
    const stride = n + 1;

    // s(0, 0) = 1
    try testing.expectEqual(@as(u64, 1), tri[0 * stride + 0]);

    // s(3, 2) = 3
    try testing.expectEqual(@as(u64, 3), tri[3 * stride + 2]);

    // s(4, 2) = 7
    try testing.expectEqual(@as(u64, 7), tri[4 * stride + 2]);

    // s(4, 3) = 6
    try testing.expectEqual(@as(u64, 6), tri[4 * stride + 3]);
}

test "stirlingFirst: type variants (u32, u64, u128)" {
    const allocator = testing.allocator;

    // u32
    try testing.expectEqual(@as(u32, 11), try stirlingFirst(u32, 4, 2, allocator));

    // u64
    try testing.expectEqual(@as(u64, 11), try stirlingFirst(u64, 4, 2, allocator));

    // u128
    try testing.expectEqual(@as(u128, 11), try stirlingFirst(u128, 4, 2, allocator));
}

test "stirlingSecond: type variants (u32, u64, u128)" {
    const allocator = testing.allocator;

    // u32
    try testing.expectEqual(@as(u32, 7), try stirlingSecond(u32, 4, 2, allocator));

    // u64
    try testing.expectEqual(@as(u64, 7), try stirlingSecond(u64, 4, 2, allocator));

    // u128
    try testing.expectEqual(@as(u128, 7), try stirlingSecond(u128, 4, 2, allocator));
}

test "stirlingFirst: larger values" {
    const allocator = testing.allocator;

    // S(6, 3) = 225
    try testing.expectEqual(@as(u64, 225), try stirlingFirst(u64, 6, 3, allocator));

    // S(7, 4) = 735
    try testing.expectEqual(@as(u64, 735), try stirlingFirst(u64, 7, 4, allocator));
}

test "stirlingSecond: larger values" {
    const allocator = testing.allocator;

    // s(6, 3) = 90
    try testing.expectEqual(@as(u64, 90), try stirlingSecond(u64, 6, 3, allocator));

    // s(7, 4) = 350
    try testing.expectEqual(@as(u64, 350), try stirlingSecond(u64, 7, 4, allocator));
}

test "generateStirlingFirstRow: n=0" {
    const allocator = testing.allocator;

    const row = try generateStirlingFirstRow(u64, 0, allocator);
    defer allocator.free(row);

    try testing.expectEqual(@as(usize, 1), row.len);
    try testing.expectEqual(@as(u64, 1), row[0]);
}

test "generateStirlingSecondRow: n=0" {
    const allocator = testing.allocator;

    const row = try generateStirlingSecondRow(u64, 0, allocator);
    defer allocator.free(row);

    try testing.expectEqual(@as(usize, 1), row.len);
    try testing.expectEqual(@as(u64, 1), row[0]);
}

test "stirlingSecond: relationship to Bell numbers" {
    const allocator = testing.allocator;

    // Bell(4) = Σ_k s(4, k) = s(4,0) + s(4,1) + s(4,2) + s(4,3) + s(4,4)
    //         = 0 + 1 + 7 + 6 + 1 = 15

    const row = try generateStirlingSecondRow(u64, 4, allocator);
    defer allocator.free(row);

    var bell_sum: u64 = 0;
    for (row) |val| {
        bell_sum += val;
    }

    try testing.expectEqual(@as(u64, 15), bell_sum);
}

test "stirlingFirst vs stirlingSecond: different values for same (n, k)" {
    const allocator = testing.allocator;

    // S(5, 3) = 35 (first kind)
    // s(5, 3) = 25 (second kind)
    const first = try stirlingFirst(u64, 5, 3, allocator);
    const second = try stirlingSecond(u64, 5, 3, allocator);

    try testing.expectEqual(@as(u64, 35), first);
    try testing.expectEqual(@as(u64, 25), second);
    try testing.expect(first != second);
}

test "memory safety: all functions use testing.allocator" {
    const allocator = testing.allocator;

    // Run multiple operations to detect leaks
    for (0..10) |_| {
        _ = try stirlingFirst(u64, 5, 3, allocator);
        _ = try stirlingSecond(u64, 5, 3, allocator);
        _ = try stirlingFirstSigned(i64, 5, 3, allocator);

        const row1 = try generateStirlingFirstRow(u64, 5, allocator);
        allocator.free(row1);

        const row2 = try generateStirlingSecondRow(u64, 5, allocator);
        allocator.free(row2);

        const tri1 = try generateStirlingFirstTriangle(u64, 3, allocator);
        allocator.free(tri1);

        const tri2 = try generateStirlingSecondTriangle(u64, 3, allocator);
        allocator.free(tri2);
    }
}
