//! Ternary Search Algorithm
//!
//! Ternary search is used to find the maximum or minimum of a unimodal function.
//! It divides the search space into three parts and eliminates one third in each iteration.
//! Commonly used in competitive programming and optimization problems.

const std = @import("std");
const testing = std.testing;

/// Ternary search result for optimization problems
pub const TernaryResult = struct {
    index: usize,
    value: f64,
};

/// Ternary search on a discrete array - finds the index of maximum value in a unimodal array.
/// A unimodal array has values that strictly increase then strictly decrease (or vice versa).
///
/// Time: O(log n) | Space: O(1)
pub fn ternarySearchMax(
    comptime T: type,
    slice: []const T,
) ?usize {
    comptime {
        const type_info = @typeInfo(T);
        if (type_info != .int and type_info != .float) {
            @compileError("ternarySearchMax requires numeric type (int or float)");
        }
    }

    if (slice.len == 0) return null;
    if (slice.len == 1) return 0;

    var left: usize = 0;
    var right: usize = slice.len - 1;

    while (right - left > 2) {
        const mid1 = left + (right - left) / 3;
        const mid2 = right - (right - left) / 3;

        if (slice[mid1] < slice[mid2]) {
            left = mid1;
        } else {
            right = mid2;
        }
    }

    // Check remaining elements
    var max_idx = left;
    var i = left + 1;
    while (i <= right) : (i += 1) {
        if (slice[i] > slice[max_idx]) {
            max_idx = i;
        }
    }

    return max_idx;
}

/// Ternary search on a discrete array - finds the index of minimum value in a unimodal array.
///
/// Time: O(log n) | Space: O(1)
pub fn ternarySearchMin(
    comptime T: type,
    slice: []const T,
) ?usize {
    comptime {
        const type_info = @typeInfo(T);
        if (type_info != .int and type_info != .float) {
            @compileError("ternarySearchMin requires numeric type (int or float)");
        }
    }

    if (slice.len == 0) return null;
    if (slice.len == 1) return 0;

    var left: usize = 0;
    var right: usize = slice.len - 1;

    while (right - left > 2) {
        const mid1 = left + (right - left) / 3;
        const mid2 = right - (right - left) / 3;

        if (slice[mid1] > slice[mid2]) {
            left = mid1;
        } else {
            right = mid2;
        }
    }

    // Check remaining elements
    var min_idx = left;
    var i = left + 1;
    while (i <= right) : (i += 1) {
        if (slice[i] < slice[min_idx]) {
            min_idx = i;
        }
    }

    return min_idx;
}

/// Ternary search on a continuous function - finds the maximum of a unimodal function.
/// The function must be unimodal in the given range [left, right].
///
/// Time: O(log((right-left)/epsilon)) | Space: O(1)
pub fn ternarySearchMaxContinuous(
    comptime F: type,
    func: F,
    left: f64,
    right: f64,
    epsilon: f64,
) TernaryResult {
    var l = left;
    var r = right;

    while (r - l > epsilon) {
        const mid1 = l + (r - l) / 3.0;
        const mid2 = r - (r - l) / 3.0;

        const f1 = func(mid1);
        const f2 = func(mid2);

        if (f1 < f2) {
            l = mid1;
        } else {
            r = mid2;
        }
    }

    const x = (l + r) / 2.0;
    return .{
        .index = 0, // Not applicable for continuous
        .value = func(x),
    };
}

/// Ternary search on a continuous function - finds the minimum of a unimodal function.
///
/// Time: O(log((right-left)/epsilon)) | Space: O(1)
pub fn ternarySearchMinContinuous(
    comptime F: type,
    func: F,
    left: f64,
    right: f64,
    epsilon: f64,
) TernaryResult {
    var l = left;
    var r = right;

    while (r - l > epsilon) {
        const mid1 = l + (r - l) / 3.0;
        const mid2 = r - (r - l) / 3.0;

        const f1 = func(mid1);
        const f2 = func(mid2);

        if (f1 > f2) {
            l = mid1;
        } else {
            r = mid2;
        }
    }

    const x = (l + r) / 2.0;
    return .{
        .index = 0, // Not applicable for continuous
        .value = func(x),
    };
}

// ============================================================================
// Tests
// ============================================================================

test "ternarySearchMax - basic unimodal array" {
    const arr = [_]i32{ 1, 3, 5, 7, 9, 8, 6, 4, 2 };
    const result = ternarySearchMax(i32, &arr);
    try testing.expectEqual(@as(?usize, 4), result);
    try testing.expectEqual(@as(i32, 9), arr[result.?]);
}

test "ternarySearchMax - peak at start" {
    const arr = [_]i32{ 10, 8, 6, 4, 2 };
    const result = ternarySearchMax(i32, &arr);
    try testing.expectEqual(@as(?usize, 0), result);
}

test "ternarySearchMax - peak at end" {
    const arr = [_]i32{ 2, 4, 6, 8, 10 };
    const result = ternarySearchMax(i32, &arr);
    try testing.expectEqual(@as(?usize, 4), result);
}

test "ternarySearchMax - single element" {
    const arr = [_]i32{42};
    const result = ternarySearchMax(i32, &arr);
    try testing.expectEqual(@as(?usize, 0), result);
}

test "ternarySearchMax - two elements" {
    const arr = [_]i32{ 3, 5 };
    const result = ternarySearchMax(i32, &arr);
    try testing.expectEqual(@as(i32, 5), arr[result.?]);
}

test "ternarySearchMax - empty array" {
    const arr = [_]i32{};
    const result = ternarySearchMax(i32, &arr);
    try testing.expectEqual(@as(?usize, null), result);
}

test "ternarySearchMax - floats" {
    const arr = [_]f64{ 1.0, 2.5, 4.0, 5.5, 7.0, 6.0, 4.5, 3.0 };
    const result = ternarySearchMax(f64, &arr);
    try testing.expectEqual(@as(?usize, 4), result);
    try testing.expectEqual(@as(f64, 7.0), arr[result.?]);
}

test "ternarySearchMin - basic unimodal array" {
    const arr = [_]i32{ 9, 7, 5, 3, 1, 2, 4, 6, 8 };
    const result = ternarySearchMin(i32, &arr);
    try testing.expectEqual(@as(?usize, 4), result);
    try testing.expectEqual(@as(i32, 1), arr[result.?]);
}

test "ternarySearchMin - valley at start" {
    const arr = [_]i32{ 1, 3, 5, 7, 9 };
    const result = ternarySearchMin(i32, &arr);
    try testing.expectEqual(@as(?usize, 0), result);
}

test "ternarySearchMin - valley at end" {
    const arr = [_]i32{ 9, 7, 5, 3, 1 };
    const result = ternarySearchMin(i32, &arr);
    try testing.expectEqual(@as(?usize, 4), result);
}

test "ternarySearchMin - single element" {
    const arr = [_]i32{42};
    const result = ternarySearchMin(i32, &arr);
    try testing.expectEqual(@as(?usize, 0), result);
}

test "ternarySearchMin - floats" {
    const arr = [_]f64{ 8.0, 6.0, 4.0, 2.0, 1.0, 3.0, 5.0, 7.0 };
    const result = ternarySearchMin(f64, &arr);
    try testing.expectEqual(@as(?usize, 4), result);
    try testing.expectEqual(@as(f64, 1.0), arr[result.?]);
}

test "ternarySearchMaxContinuous - quadratic function" {
    // f(x) = -(x-5)^2 + 25, maximum at x=5
    const Func = struct {
        fn eval(x: f64) f64 {
            const diff = x - 5.0;
            return -(diff * diff) + 25.0;
        }
    };

    const result = ternarySearchMaxContinuous(
        @TypeOf(Func.eval),
        Func.eval,
        0.0,
        10.0,
        1e-6,
    );

    // Should find maximum near x=5, f(5)=25
    try testing.expect(@abs(result.value - 25.0) < 0.01);
}

test "ternarySearchMinContinuous - quadratic function" {
    // f(x) = (x-3)^2 + 2, minimum at x=3
    const Func = struct {
        fn eval(x: f64) f64 {
            const diff = x - 3.0;
            return (diff * diff) + 2.0;
        }
    };

    const result = ternarySearchMinContinuous(
        @TypeOf(Func.eval),
        Func.eval,
        0.0,
        6.0,
        1e-6,
    );

    // Should find minimum near x=3, f(3)=2
    try testing.expect(@abs(result.value - 2.0) < 0.01);
}

test "ternarySearchMaxContinuous - inverted parabola" {
    // f(x) = -0.5(x-4)^2 + 8, maximum at x=4, f(4)=8
    const Func = struct {
        fn eval(x: f64) f64 {
            const diff = x - 4.0;
            return -0.5 * (diff * diff) + 8.0;
        }
    };

    const result = ternarySearchMaxContinuous(
        @TypeOf(Func.eval),
        Func.eval,
        0.0,
        8.0,
        1e-6,
    );

    // Should find maximum near x=4, f(4)=8
    try testing.expect(@abs(result.value - 8.0) < 0.01);
}

test "ternarySearchMinContinuous - absolute value function" {
    // f(x) = |x - 7|, minimum at x=7
    const Func = struct {
        fn eval(x: f64) f64 {
            return @abs(x - 7.0);
        }
    };

    const result = ternarySearchMinContinuous(
        @TypeOf(Func.eval),
        Func.eval,
        0.0,
        10.0,
        1e-6,
    );

    // Should find minimum near x=7, f(7)=0
    try testing.expect(@abs(result.value) < 0.01);
}

test "ternarySearchMax - large array" {
    const allocator = testing.allocator;
    const n = 10000;

    const arr = try allocator.alloc(i32, n);
    defer allocator.free(arr);

    // Create unimodal array: increases to middle, then decreases
    const peak = n / 2;
    for (arr, 0..) |*val, i| {
        const dist: i32 = @intCast(@abs(@as(i32, @intCast(i)) - @as(i32, @intCast(peak))));
        val.* = 10000 - dist;
    }

    const result = ternarySearchMax(i32, arr);
    try testing.expect(result != null);
    try testing.expect(@abs(@as(i32, @intCast(result.?)) - @as(i32, @intCast(peak))) <= 1);
}

test "ternarySearchMin - large array" {
    const allocator = testing.allocator;
    const n = 10000;

    const arr = try allocator.alloc(i32, n);
    defer allocator.free(arr);

    // Create unimodal array: decreases to middle, then increases
    const valley = n / 2;
    for (arr, 0..) |*val, i| {
        const dist: i32 = @intCast(@abs(@as(i32, @intCast(i)) - @as(i32, @intCast(valley))));
        val.* = dist;
    }

    const result = ternarySearchMin(i32, arr);
    try testing.expect(result != null);
    try testing.expect(@abs(@as(i32, @intCast(result.?)) - @as(i32, @intCast(valley))) <= 1);
}

test "ternarySearchMaxContinuous - sin function" {
    // f(x) = sin(x) in [0, π], maximum at x=π/2
    const Func = struct {
        fn eval(x: f64) f64 {
            return @sin(x);
        }
    };

    const pi = std.math.pi;
    const result = ternarySearchMaxContinuous(
        @TypeOf(Func.eval),
        Func.eval,
        0.0,
        pi,
        1e-6,
    );

    // Should find maximum near sin(π/2) = 1
    try testing.expect(@abs(result.value - 1.0) < 0.01);
}

test "ternarySearchMinContinuous - cos function" {
    // f(x) = cos(x) in [0, π], minimum at x=π
    const Func = struct {
        fn eval(x: f64) f64 {
            return @cos(x);
        }
    };

    const pi = std.math.pi;
    const result = ternarySearchMinContinuous(
        @TypeOf(Func.eval),
        Func.eval,
        0.0,
        pi,
        1e-6,
    );

    // Should find minimum near cos(π) = -1
    try testing.expect(@abs(result.value - (-1.0)) < 0.01);
}
