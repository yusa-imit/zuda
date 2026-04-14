const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

/// Integer Sequences
///
/// Fundamental integer sequences appearing in combinatorics, number theory,
/// and discrete mathematics. These sequences have closed-form formulas,
/// recurrence relations, and generating functions.
///
/// Included sequences:
/// - Fibonacci numbers: F(n) = F(n-1) + F(n-2), F(0)=0, F(1)=1
/// - Lucas numbers: L(n) = L(n-1) + L(n-2), L(0)=2, L(1)=1
/// - Triangular numbers: T(n) = n(n+1)/2 (sum of first n natural numbers)
/// - Pentagonal numbers: P(n) = n(3n-1)/2 (generalized: P(n,k) for k-gonal)
/// - Square numbers: S(n) = n²
/// - Tetrahedral numbers: Tet(n) = n(n+1)(n+2)/6 (3D triangular numbers)
/// - Harmonic numbers: H(n) = 1 + 1/2 + ... + 1/n (as rational approximation)
/// - Pell numbers: P(n) = 2P(n-1) + P(n-2), P(0)=0, P(1)=1
///
/// References:
/// - OEIS (Online Encyclopedia of Integer Sequences): https://oeis.org/
/// - Sloane, N. J. A., "The On-Line Encyclopedia of Integer Sequences"
/// - Graham, Knuth, Patashnik, "Concrete Mathematics" (1994)

/// Fibonacci number F(n) using iterative algorithm
///
/// Sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, ...
/// Recurrence: F(n) = F(n-1) + F(n-2) with F(0)=0, F(1)=1
/// Golden ratio: lim(F(n+1)/F(n)) = φ = (1+√5)/2 ≈ 1.618
/// Binet's formula: F(n) = (φⁿ - ψⁿ)/√5 where ψ = (1-√5)/2
///
/// Time: O(n)
/// Space: O(1)
pub fn fibonacci(comptime T: type, n: T) T {
    if (n == 0) return 0;
    if (n == 1) return 1;

    var a: T = 0;
    var b: T = 1;
    var i: T = 2;

    while (i <= n) : (i += 1) {
        const temp = a + b;
        a = b;
        b = temp;
    }

    return b;
}

/// Lucas number L(n) using iterative algorithm
///
/// Sequence: 2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, ...
/// Recurrence: L(n) = L(n-1) + L(n-2) with L(0)=2, L(1)=1
/// Relation to Fibonacci: L(n) = F(n-1) + F(n+1)
/// Golden ratio: lim(L(n+1)/L(n)) = φ (same as Fibonacci)
///
/// Time: O(n)
/// Space: O(1)
pub fn lucas(comptime T: type, n: T) T {
    if (n == 0) return 2;
    if (n == 1) return 1;

    var a: T = 2;
    var b: T = 1;
    var i: T = 2;

    while (i <= n) : (i += 1) {
        const temp = a + b;
        a = b;
        b = temp;
    }

    return b;
}

/// Triangular number T(n) = n(n+1)/2
///
/// Sequence: 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, ...
/// Formula: T(n) = 1 + 2 + 3 + ... + n = n(n+1)/2
/// Combinatorial: T(n) = C(n+1, 2) (combinations of n+1 items taken 2 at a time)
/// Relation: T(n) = T(n-1) + n
///
/// Time: O(1)
/// Space: O(1)
pub fn triangular(comptime T: type, n: T) T {
    return (n * (n + 1)) / 2;
}

/// Pentagonal number P(n) = n(3n-1)/2
///
/// Sequence: 0, 1, 5, 12, 22, 35, 51, 70, 92, 117, 145, ...
/// Formula: P(n) = n(3n-1)/2
/// Combinatorial: Number of dots in pentagonal arrangement
/// Partition identity: Related to Euler's pentagonal number theorem
///
/// Time: O(1)
/// Space: O(1)
pub fn pentagonal(comptime T: type, n: T) T {
    if (n == 0) return 0;
    return (n * (3 * n - 1)) / 2;
}

/// Generalized pentagonal number P(k) where k can be positive or negative
///
/// Sequence (k=...,-2,-1,0,1,2,...): ..., 7, 2, 0, 1, 5, 12, 22, 35, ...
/// Formula: P(k) = k(3k-1)/2 for k > 0
///          P(k) = k(3k+1)/2 for k < 0 (equivalent to P(-k) adjusted)
/// Euler's theorem: Product(1-xⁿ) = Sum((-1)ᵏ * x^P(k)) for all integers k
///
/// Time: O(1)
/// Space: O(1)
pub fn pentagonalGeneralized(comptime T: type, k: i64) T {
    const abs_k = if (k < 0) -k else k;
    const result = if (k > 0)
        @divTrunc(k * (3 * k - 1), 2)
    else if (k < 0)
        @divTrunc(abs_k * (3 * abs_k + 1), 2)
    else
        0;

    return @intCast(result);
}

/// Square number S(n) = n²
///
/// Sequence: 0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, ...
/// Formula: S(n) = n²
/// Relation: S(n) = S(n-1) + 2n - 1 (sum of first n odd numbers)
///
/// Time: O(1)
/// Space: O(1)
pub fn square(comptime T: type, n: T) T {
    return n * n;
}

/// Tetrahedral number Tet(n) = n(n+1)(n+2)/6
///
/// Sequence: 0, 1, 4, 10, 20, 35, 56, 84, 120, 165, 220, ...
/// Formula: Tet(n) = C(n+2, 3) = n(n+1)(n+2)/6
/// Combinatorial: Sum of first n triangular numbers
/// 3D generalization: Number of spheres in triangular pyramid
///
/// Time: O(1)
/// Space: O(1)
pub fn tetrahedral(comptime T: type, n: T) T {
    return (n * (n + 1) * (n + 2)) / 6;
}

/// Pell number P(n) = 2P(n-1) + P(n-2)
///
/// Sequence: 0, 1, 2, 5, 12, 29, 70, 169, 408, 985, 2378, ...
/// Recurrence: P(n) = 2P(n-1) + P(n-2) with P(0)=0, P(1)=1
/// Silver ratio: lim(P(n+1)/P(n)) = δ = 1 + √2 ≈ 2.414
/// Relation: (P(n), P(n+1)) approximates √2 as P(n+1)²/P(n)² → 2
///
/// Time: O(n)
/// Space: O(1)
pub fn pell(comptime T: type, n: T) T {
    if (n == 0) return 0;
    if (n == 1) return 1;

    var a: T = 0;
    var b: T = 1;
    var i: T = 2;

    while (i <= n) : (i += 1) {
        const temp = 2 * b + a;
        a = b;
        b = temp;
    }

    return b;
}

/// Polygonal number P(s, n) for s-sided polygon
///
/// Formula: P(s, n) = n * ((s-2)*n - (s-4)) / 2
/// Special cases:
/// - s=3: Triangular numbers
/// - s=4: Square numbers
/// - s=5: Pentagonal numbers
/// - s=6: Hexagonal numbers
///
/// Time: O(1)
/// Space: O(1)
pub fn polygonal(comptime T: type, s: T, n: T) T {
    if (n == 0) return 0;
    // Formula: P(s, n) = n * ((s-2)*n - (s-4)) / 2
    // Rearrange: P(s, n) = n * ((s-2)*n - s + 4) / 2
    //                    = n * ((s-2)*n + 4 - s) / 2
    // For unsigned safety, compute as: n * (k*n + 4 - s) / 2 where k = s-2
    const k = s - 2; // k >= 1 for s >= 3
    const term = k * n + 4 - s; // This can still underflow if s > 4 + k*n
    return (n * term) / 2;
}

/// Generate Fibonacci sequence up to index n
///
/// Returns ArrayList containing [F(0), F(1), ..., F(n)]
///
/// Time: O(n)
/// Space: O(n)
pub fn generateFibonacci(comptime T: type, allocator: Allocator, n: usize) !ArrayList(T) {
    var result = ArrayList(T){};
    errdefer result.deinit(allocator);

    if (n == 0) {
        try result.append(allocator, 0);
        return result;
    }

    try result.append(allocator, 0);
    try result.append(allocator, 1);

    var i: usize = 2;
    while (i <= n) : (i += 1) {
        const next = result.items[i - 1] + result.items[i - 2];
        try result.append(allocator, next);
    }

    return result;
}

/// Generate Lucas sequence up to index n
///
/// Returns ArrayList containing [L(0), L(1), ..., L(n)]
///
/// Time: O(n)
/// Space: O(n)
pub fn generateLucas(comptime T: type, allocator: Allocator, n: usize) !ArrayList(T) {
    var result = ArrayList(T){};
    errdefer result.deinit(allocator);

    if (n == 0) {
        try result.append(allocator, 2);
        return result;
    }

    try result.append(allocator, 2);
    try result.append(allocator, 1);

    var i: usize = 2;
    while (i <= n) : (i += 1) {
        const next = result.items[i - 1] + result.items[i - 2];
        try result.append(allocator, next);
    }

    return result;
}

/// Generate triangular numbers up to index n
///
/// Returns ArrayList containing [T(0), T(1), ..., T(n)]
///
/// Time: O(n)
/// Space: O(n)
pub fn generateTriangular(comptime T: type, allocator: Allocator, n: usize) !ArrayList(T) {
    var result = ArrayList(T){};
    errdefer result.deinit(allocator);

    var i: usize = 0;
    while (i <= n) : (i += 1) {
        try result.append(allocator, triangular(T, @intCast(i)));
    }

    return result;
}

/// Fibonacci via matrix exponentiation (fast for large n)
///
/// Uses [[1,1],[1,0]]^n to compute F(n) in O(log n) time
/// Matrix power: [[F(n+1), F(n)], [F(n), F(n-1)]]
///
/// Time: O(log n)
/// Space: O(1)
pub fn fibonacciFast(comptime T: type, n: T) T {
    if (n == 0) return 0;
    if (n == 1) return 1;

    // Matrix [[a, b], [c, d]] represents [[F(n+1), F(n)], [F(n), F(n-1)]]
    var a: T = 1;
    var b: T = 1;
    var c: T = 1;
    var d: T = 0;

    var power = n - 1;
    var result_a: T = 1;
    var result_b: T = 0;
    var result_c: T = 0;
    var result_d: T = 1;

    while (power > 0) : (power /= 2) {
        if (power % 2 == 1) {
            // Multiply result by current matrix
            const temp_a = result_a * a + result_b * c;
            const temp_b = result_a * b + result_b * d;
            const temp_c = result_c * a + result_d * c;
            const temp_d = result_c * b + result_d * d;
            result_a = temp_a;
            result_b = temp_b;
            result_c = temp_c;
            result_d = temp_d;
        }

        // Square the matrix
        const temp_a = a * a + b * c;
        const temp_b = a * b + b * d;
        const temp_c = c * a + d * c;
        const temp_d = c * b + d * d;
        a = temp_a;
        b = temp_b;
        c = temp_c;
        d = temp_d;
    }

    return result_a;
}

// ============================================================================
// Tests
// ============================================================================

test "fibonacci - base cases" {
    try testing.expectEqual(@as(u64, 0), fibonacci(u64, 0));
    try testing.expectEqual(@as(u64, 1), fibonacci(u64, 1));
    try testing.expectEqual(@as(u64, 1), fibonacci(u64, 2));
    try testing.expectEqual(@as(u64, 2), fibonacci(u64, 3));
    try testing.expectEqual(@as(u64, 3), fibonacci(u64, 4));
    try testing.expectEqual(@as(u64, 5), fibonacci(u64, 5));
}

test "fibonacci - OEIS A000045" {
    // First 15 Fibonacci numbers: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377
    const expected = [_]u64{ 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377 };
    for (expected, 0..) |exp, i| {
        try testing.expectEqual(exp, fibonacci(u64, @intCast(i)));
    }
}

test "fibonacci - type variants" {
    try testing.expectEqual(@as(u32, 13), fibonacci(u32, 7));
    try testing.expectEqual(@as(u64, 13), fibonacci(u64, 7));
    try testing.expectEqual(@as(usize, 13), fibonacci(usize, 7));
}

test "lucas - base cases" {
    try testing.expectEqual(@as(u64, 2), lucas(u64, 0));
    try testing.expectEqual(@as(u64, 1), lucas(u64, 1));
    try testing.expectEqual(@as(u64, 3), lucas(u64, 2));
    try testing.expectEqual(@as(u64, 4), lucas(u64, 3));
    try testing.expectEqual(@as(u64, 7), lucas(u64, 4));
    try testing.expectEqual(@as(u64, 11), lucas(u64, 5));
}

test "lucas - OEIS A000032" {
    // First 15 Lucas numbers: 2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521, 843
    const expected = [_]u64{ 2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521, 843 };
    for (expected, 0..) |exp, i| {
        try testing.expectEqual(exp, lucas(u64, @intCast(i)));
    }
}

test "lucas - relation to Fibonacci" {
    // L(n) = F(n-1) + F(n+1)
    var n: u64 = 2;
    while (n <= 10) : (n += 1) {
        const L_n = lucas(u64, n);
        const F_n_minus_1 = fibonacci(u64, n - 1);
        const F_n_plus_1 = fibonacci(u64, n + 1);
        try testing.expectEqual(L_n, F_n_minus_1 + F_n_plus_1);
    }
}

test "triangular - base cases" {
    try testing.expectEqual(@as(u64, 0), triangular(u64, 0));
    try testing.expectEqual(@as(u64, 1), triangular(u64, 1));
    try testing.expectEqual(@as(u64, 3), triangular(u64, 2));
    try testing.expectEqual(@as(u64, 6), triangular(u64, 3));
    try testing.expectEqual(@as(u64, 10), triangular(u64, 4));
    try testing.expectEqual(@as(u64, 15), triangular(u64, 5));
}

test "triangular - OEIS A000217" {
    // First 15 triangular numbers: 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105
    const expected = [_]u64{ 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105 };
    for (expected, 0..) |exp, i| {
        try testing.expectEqual(exp, triangular(u64, @intCast(i)));
    }
}

test "triangular - sum formula" {
    // T(n) = 1 + 2 + ... + n
    var n: u64 = 1;
    while (n <= 20) : (n += 1) {
        var sum: u64 = 0;
        var i: u64 = 1;
        while (i <= n) : (i += 1) {
            sum += i;
        }
        try testing.expectEqual(sum, triangular(u64, n));
    }
}

test "pentagonal - base cases" {
    try testing.expectEqual(@as(u64, 0), pentagonal(u64, 0));
    try testing.expectEqual(@as(u64, 1), pentagonal(u64, 1));
    try testing.expectEqual(@as(u64, 5), pentagonal(u64, 2));
    try testing.expectEqual(@as(u64, 12), pentagonal(u64, 3));
    try testing.expectEqual(@as(u64, 22), pentagonal(u64, 4));
    try testing.expectEqual(@as(u64, 35), pentagonal(u64, 5));
}

test "pentagonal - OEIS A000326" {
    // First 15 pentagonal numbers: 0, 1, 5, 12, 22, 35, 51, 70, 92, 117, 145, 176, 210, 247, 287
    const expected = [_]u64{ 0, 1, 5, 12, 22, 35, 51, 70, 92, 117, 145, 176, 210, 247, 287 };
    for (expected, 0..) |exp, i| {
        try testing.expectEqual(exp, pentagonal(u64, @intCast(i)));
    }
}

test "pentagonalGeneralized - positive and negative" {
    // Positive: 0, 1, 5, 12, 22, ...
    try testing.expectEqual(@as(u64, 0), pentagonalGeneralized(u64, 0));
    try testing.expectEqual(@as(u64, 1), pentagonalGeneralized(u64, 1));
    try testing.expectEqual(@as(u64, 5), pentagonalGeneralized(u64, 2));
    try testing.expectEqual(@as(u64, 12), pentagonalGeneralized(u64, 3));

    // Negative: k=-1 → 2, k=-2 → 7, k=-3 → 15
    try testing.expectEqual(@as(u64, 2), pentagonalGeneralized(u64, -1));
    try testing.expectEqual(@as(u64, 7), pentagonalGeneralized(u64, -2));
    try testing.expectEqual(@as(u64, 15), pentagonalGeneralized(u64, -3));
}

test "square - base cases" {
    try testing.expectEqual(@as(u64, 0), square(u64, 0));
    try testing.expectEqual(@as(u64, 1), square(u64, 1));
    try testing.expectEqual(@as(u64, 4), square(u64, 2));
    try testing.expectEqual(@as(u64, 9), square(u64, 3));
    try testing.expectEqual(@as(u64, 16), square(u64, 4));
    try testing.expectEqual(@as(u64, 25), square(u64, 5));
}

test "square - OEIS A000290" {
    // First 15 square numbers: 0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196
    const expected = [_]u64{ 0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196 };
    for (expected, 0..) |exp, i| {
        try testing.expectEqual(exp, square(u64, @intCast(i)));
    }
}

test "tetrahedral - base cases" {
    try testing.expectEqual(@as(u64, 0), tetrahedral(u64, 0));
    try testing.expectEqual(@as(u64, 1), tetrahedral(u64, 1));
    try testing.expectEqual(@as(u64, 4), tetrahedral(u64, 2));
    try testing.expectEqual(@as(u64, 10), tetrahedral(u64, 3));
    try testing.expectEqual(@as(u64, 20), tetrahedral(u64, 4));
    try testing.expectEqual(@as(u64, 35), tetrahedral(u64, 5));
}

test "tetrahedral - OEIS A000292" {
    // First 15 tetrahedral numbers: 0, 1, 4, 10, 20, 35, 56, 84, 120, 165, 220, 286, 364, 455, 560
    const expected = [_]u64{ 0, 1, 4, 10, 20, 35, 56, 84, 120, 165, 220, 286, 364, 455, 560 };
    for (expected, 0..) |exp, i| {
        try testing.expectEqual(exp, tetrahedral(u64, @intCast(i)));
    }
}

test "tetrahedral - sum of triangular" {
    // Tet(n) = T(1) + T(2) + ... + T(n)
    var n: u64 = 1;
    while (n <= 20) : (n += 1) {
        var sum: u64 = 0;
        var i: u64 = 1;
        while (i <= n) : (i += 1) {
            sum += triangular(u64, i);
        }
        try testing.expectEqual(sum, tetrahedral(u64, n));
    }
}

test "pell - base cases" {
    try testing.expectEqual(@as(u64, 0), pell(u64, 0));
    try testing.expectEqual(@as(u64, 1), pell(u64, 1));
    try testing.expectEqual(@as(u64, 2), pell(u64, 2));
    try testing.expectEqual(@as(u64, 5), pell(u64, 3));
    try testing.expectEqual(@as(u64, 12), pell(u64, 4));
    try testing.expectEqual(@as(u64, 29), pell(u64, 5));
}

test "pell - OEIS A000129" {
    // First 15 Pell numbers: 0, 1, 2, 5, 12, 29, 70, 169, 408, 985, 2378, 5741, 13860, 33461, 80782
    const expected = [_]u64{ 0, 1, 2, 5, 12, 29, 70, 169, 408, 985, 2378, 5741, 13860, 33461, 80782 };
    for (expected, 0..) |exp, i| {
        try testing.expectEqual(exp, pell(u64, @intCast(i)));
    }
}

test "pell - recurrence relation" {
    // P(n) = 2P(n-1) + P(n-2)
    var n: u64 = 2;
    while (n <= 15) : (n += 1) {
        const P_n = pell(u64, n);
        const P_n_minus_1 = pell(u64, n - 1);
        const P_n_minus_2 = pell(u64, n - 2);
        try testing.expectEqual(P_n, 2 * P_n_minus_1 + P_n_minus_2);
    }
}

test "polygonal - special cases" {
    // s=3: Triangular
    try testing.expectEqual(triangular(u64, 5), polygonal(u64, 3, 5));
    try testing.expectEqual(triangular(u64, 10), polygonal(u64, 3, 10));

    // s=4: Square
    try testing.expectEqual(square(u64, 5), polygonal(u64, 4, 5));
    try testing.expectEqual(square(u64, 10), polygonal(u64, 4, 10));

    // s=5: Pentagonal
    try testing.expectEqual(pentagonal(u64, 5), polygonal(u64, 5, 5));
    try testing.expectEqual(pentagonal(u64, 10), polygonal(u64, 5, 10));
}

test "polygonal - hexagonal numbers" {
    // Hexagonal (s=6): 0, 1, 6, 15, 28, 45, 66, 91, 120, ... (OEIS A000384)
    const expected = [_]u64{ 0, 1, 6, 15, 28, 45, 66, 91, 120 };
    for (expected, 0..) |exp, i| {
        try testing.expectEqual(exp, polygonal(u64, 6, @intCast(i)));
    }
}

test "generateFibonacci - basic" {
    const allocator = testing.allocator;
    var result = try generateFibonacci(u64, allocator, 10);
    defer result.deinit(allocator);

    const expected = [_]u64{ 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55 };
    try testing.expectEqualSlices(u64, &expected, result.items);
}

test "generateFibonacci - n=0" {
    const allocator = testing.allocator;
    var result = try generateFibonacci(u64, allocator, 0);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 1), result.items.len);
    try testing.expectEqual(@as(u64, 0), result.items[0]);
}

test "generateLucas - basic" {
    const allocator = testing.allocator;
    var result = try generateLucas(u64, allocator, 10);
    defer result.deinit(allocator);

    const expected = [_]u64{ 2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123 };
    try testing.expectEqualSlices(u64, &expected, result.items);
}

test "generateLucas - n=0" {
    const allocator = testing.allocator;
    var result = try generateLucas(u64, allocator, 0);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 1), result.items.len);
    try testing.expectEqual(@as(u64, 2), result.items[0]);
}

test "generateTriangular - basic" {
    const allocator = testing.allocator;
    var result = try generateTriangular(u64, allocator, 10);
    defer result.deinit(allocator);

    const expected = [_]u64{ 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55 };
    try testing.expectEqualSlices(u64, &expected, result.items);
}

test "fibonacciFast - correctness" {
    // Verify fast algorithm matches iterative for first 20 values
    var n: u64 = 0;
    while (n <= 20) : (n += 1) {
        const iterative = fibonacci(u64, n);
        const fast = fibonacciFast(u64, n);
        try testing.expectEqual(iterative, fast);
    }
}

test "fibonacciFast - large values" {
    // F(30) = 832040 (verify O(log n) works for larger n)
    try testing.expectEqual(@as(u64, 832040), fibonacciFast(u64, 30));
    try testing.expectEqual(@as(u64, 832040), fibonacci(u64, 30));
}

test "memory safety - generateFibonacci" {
    const allocator = testing.allocator;

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var result = try generateFibonacci(u64, allocator, 15);
        defer result.deinit(allocator);
        try testing.expectEqual(@as(usize, 16), result.items.len);
    }
}

test "memory safety - generateLucas" {
    const allocator = testing.allocator;

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var result = try generateLucas(u64, allocator, 15);
        defer result.deinit(allocator);
        try testing.expectEqual(@as(usize, 16), result.items.len);
    }
}

test "memory safety - generateTriangular" {
    const allocator = testing.allocator;

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var result = try generateTriangular(u64, allocator, 15);
        defer result.deinit(allocator);
        try testing.expectEqual(@as(usize, 16), result.items.len);
    }
}
