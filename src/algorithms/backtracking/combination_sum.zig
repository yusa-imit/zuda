const std = @import("std");
const ArrayList = std.ArrayList;

/// Find all unique combinations in candidates where the candidate numbers sum to target.
/// Each number in candidates may be used unlimited times.
///
/// Time: O(N^(T/M)) where N=candidates.len, T=target, M=min(candidates)
/// Space: O(T/M) for recursion depth + result storage
pub fn combinationSum(allocator: std.mem.Allocator, candidates: []const i32, target: i32) !ArrayList([]i32) {
    var results = ArrayList([]i32).init(allocator);
    errdefer {
        for (results.items) |combo| allocator.free(combo);
        results.deinit();
    }

    var current = ArrayList(i32).init(allocator);
    defer current.deinit();

    // Sort for better pruning
    const sorted = try allocator.alloc(i32, candidates.len);
    defer allocator.free(sorted);
    @memcpy(sorted, candidates);
    std.mem.sort(i32, sorted, {}, std.sort.asc(i32));

    try backtrack(allocator, &results, sorted, &current, target, 0);
    return results;
}

fn backtrack(
    allocator: std.mem.Allocator,
    results: *ArrayList([]i32),
    candidates: []const i32,
    current: *ArrayList(i32),
    target: i32,
    start: usize,
) !void {
    if (target == 0) {
        // Found valid combination
        const combo = try allocator.alloc(i32, current.items.len);
        @memcpy(combo, current.items);
        try results.append(combo);
        return;
    }

    if (target < 0) return; // Pruning: exceeded target

    for (start..candidates.len) |i| {
        const candidate = candidates[i];
        if (candidate > target) break; // Pruning: sorted array

        try current.append(candidate);
        try backtrack(allocator, results, candidates, current, target - candidate, i); // i, not i+1, allows reuse
        _ = current.pop();
    }
}

/// Find all unique combinations where each number is used at most once.
///
/// Time: O(2^N * N) - each element can be included or not
/// Space: O(N) for recursion + result storage
pub fn combinationSumUnique(allocator: std.mem.Allocator, candidates: []const i32, target: i32) !ArrayList([]i32) {
    var results = ArrayList([]i32).init(allocator);
    errdefer {
        for (results.items) |combo| allocator.free(combo);
        results.deinit();
    }

    var current = ArrayList(i32).init(allocator);
    defer current.deinit();

    // Sort for pruning and duplicate handling
    const sorted = try allocator.alloc(i32, candidates.len);
    defer allocator.free(sorted);
    @memcpy(sorted, candidates);
    std.mem.sort(i32, sorted, {}, std.sort.asc(i32));

    try backtrackUnique(allocator, &results, sorted, &current, target, 0);
    return results;
}

fn backtrackUnique(
    allocator: std.mem.Allocator,
    results: *ArrayList([]i32),
    candidates: []const i32,
    current: *ArrayList(i32),
    target: i32,
    start: usize,
) !void {
    if (target == 0) {
        const combo = try allocator.alloc(i32, current.items.len);
        @memcpy(combo, current.items);
        try results.append(combo);
        return;
    }

    if (target < 0) return;

    for (start..candidates.len) |i| {
        const candidate = candidates[i];
        if (candidate > target) break;

        // Skip duplicates
        if (i > start and candidates[i] == candidates[i - 1]) continue;

        try current.append(candidate);
        try backtrackUnique(allocator, results, candidates, current, target - candidate, i + 1); // i+1: each used once
        _ = current.pop();
    }
}

test "Combination Sum: basic with reuse" {
    const allocator = std.testing.allocator;
    const candidates = [_]i32{ 2, 3, 6, 7 };
    const target: i32 = 7;

    var combos = try combinationSum(allocator, &candidates, target);
    defer {
        for (combos.items) |combo| allocator.free(combo);
        combos.deinit();
    }

    // [2,2,3] and [7]
    try std.testing.expectEqual(@as(usize, 2), combos.items.len);
}

test "Combination Sum: multiple uses of same number" {
    const allocator = std.testing.allocator;
    const candidates = [_]i32{ 2, 3, 5 };
    const target: i32 = 8;

    var combos = try combinationSum(allocator, &candidates, target);
    defer {
        for (combos.items) |combo| allocator.free(combo);
        combos.deinit();
    }

    // [2,2,2,2], [2,3,3], [3,5]
    try std.testing.expectEqual(@as(usize, 3), combos.items.len);
}

test "Combination Sum: no solution" {
    const allocator = std.testing.allocator;
    const candidates = [_]i32{ 2, 4 };
    const target: i32 = 3;

    var combos = try combinationSum(allocator, &candidates, target);
    defer {
        for (combos.items) |combo| allocator.free(combo);
        combos.deinit();
    }

    try std.testing.expectEqual(@as(usize, 0), combos.items.len);
}

test "Combination Sum: target zero" {
    const allocator = std.testing.allocator;
    const candidates = [_]i32{ 2, 3, 5 };
    const target: i32 = 0;

    var combos = try combinationSum(allocator, &candidates, target);
    defer {
        for (combos.items) |combo| allocator.free(combo);
        combos.deinit();
    }

    // Empty combination sums to 0
    try std.testing.expectEqual(@as(usize, 1), combos.items.len);
    try std.testing.expectEqual(@as(usize, 0), combos.items[0].len);
}

test "Combination Sum Unique: each number used once" {
    const allocator = std.testing.allocator;
    const candidates = [_]i32{ 10, 1, 2, 7, 6, 1, 5 };
    const target: i32 = 8;

    var combos = try combinationSumUnique(allocator, &candidates, target);
    defer {
        for (combos.items) |combo| allocator.free(combo);
        combos.deinit();
    }

    // [1,1,6], [1,2,5], [1,7], [2,6]
    try std.testing.expectEqual(@as(usize, 4), combos.items.len);
}

test "Combination Sum Unique: no solution" {
    const allocator = std.testing.allocator;
    const candidates = [_]i32{ 3, 5, 7 };
    const target: i32 = 2;

    var combos = try combinationSumUnique(allocator, &candidates, target);
    defer {
        for (combos.items) |combo| allocator.free(combo);
        combos.deinit();
    }

    try std.testing.expectEqual(@as(usize, 0), combos.items.len);
}

test "Combination Sum Unique: single element target" {
    const allocator = std.testing.allocator;
    const candidates = [_]i32{ 1, 2, 3 };
    const target: i32 = 3;

    var combos = try combinationSumUnique(allocator, &candidates, target);
    defer {
        for (combos.items) |combo| allocator.free(combo);
        combos.deinit();
    }

    // [1,2] and [3]
    try std.testing.expectEqual(@as(usize, 2), combos.items.len);
}

test "Combination Sum: stress test" {
    const allocator = std.testing.allocator;
    const candidates = [_]i32{ 2, 3, 5 };
    const target: i32 = 10;

    var combos = try combinationSum(allocator, &candidates, target);
    defer {
        for (combos.items) |combo| allocator.free(combo);
        combos.deinit();
    }

    // Multiple combinations possible
    try std.testing.expect(combos.items.len > 0);

    // Verify each combination sums to target
    for (combos.items) |combo| {
        var sum: i32 = 0;
        for (combo) |val| sum += val;
        try std.testing.expectEqual(target, sum);
    }
}
