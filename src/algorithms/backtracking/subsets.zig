const std = @import("std");
const ArrayList = std.ArrayList;

/// Generate all subsets (power set) of an array using backtracking.
///
/// Time: O(N * 2^N) - 2^N subsets, each takes O(N) to copy
/// Space: O(N * 2^N) for storing all subsets
pub fn subsets(comptime T: type, allocator: std.mem.Allocator, items: []const T) !ArrayList([]T) {
    var results = ArrayList([]T).init(allocator);
    errdefer {
        for (results.items) |subset| allocator.free(subset);
        results.deinit();
    }

    var current = ArrayList(T).init(allocator);
    defer current.deinit();

    try backtrack(T, allocator, &results, items, &current, 0);
    return results;
}

fn backtrack(
    comptime T: type,
    allocator: std.mem.Allocator,
    results: *ArrayList([]T),
    items: []const T,
    current: *ArrayList(T),
    start: usize,
) !void {
    // Add current subset
    const subset = try allocator.alloc(T, current.items.len);
    @memcpy(subset, current.items);
    try results.append(subset);

    // Explore adding each remaining element
    for (start..items.len) |i| {
        try current.append(items[i]);
        try backtrack(T, allocator, results, items, current, i + 1);
        _ = current.pop(); // Backtrack
    }
}

/// Generate all subsets of a specific size K.
///
/// Time: O(N * C(N,K)) where C(N,K) is binomial coefficient
/// Space: O(N * C(N,K))
pub fn subsetsOfSize(comptime T: type, allocator: std.mem.Allocator, items: []const T, k: usize) !ArrayList([]T) {
    var results = ArrayList([]T).init(allocator);
    errdefer {
        for (results.items) |subset| allocator.free(subset);
        results.deinit();
    }

    if (k > items.len) return results;

    var current = ArrayList(T).init(allocator);
    defer current.deinit();

    try backtrackSize(T, allocator, &results, items, &current, 0, k);
    return results;
}

fn backtrackSize(
    comptime T: type,
    allocator: std.mem.Allocator,
    results: *ArrayList([]T),
    items: []const T,
    current: *ArrayList(T),
    start: usize,
    k: usize,
) !void {
    if (current.items.len == k) {
        const subset = try allocator.alloc(T, k);
        @memcpy(subset, current.items);
        try results.append(subset);
        return;
    }

    // Need (k - current.len) more elements, ensure enough remain
    const needed = k - current.items.len;
    const remaining = items.len - start;
    if (remaining < needed) return;

    for (start..items.len) |i| {
        try current.append(items[i]);
        try backtrackSize(T, allocator, results, items, current, i + 1, k);
        _ = current.pop();
    }
}

/// Generate all unique subsets (handles duplicates in input).
///
/// Time: O(N * 2^N) worst case, pruning reduces for duplicates
/// Space: O(N * 2^N)
pub fn subsetsUnique(comptime T: type, allocator: std.mem.Allocator, items: []const T) !ArrayList([]T) {
    var results = ArrayList([]T).init(allocator);
    errdefer {
        for (results.items) |subset| allocator.free(subset);
        results.deinit();
    }

    // Sort to group duplicates
    const sorted = try allocator.alloc(T, items.len);
    defer allocator.free(sorted);
    @memcpy(sorted, items);
    std.mem.sort(T, sorted, {}, std.sort.asc(T));

    var current = ArrayList(T).init(allocator);
    defer current.deinit();

    try backtrackUnique(T, allocator, &results, sorted, &current, 0);
    return results;
}

fn backtrackUnique(
    comptime T: type,
    allocator: std.mem.Allocator,
    results: *ArrayList([]T),
    items: []const T,
    current: *ArrayList(T),
    start: usize,
) !void {
    const subset = try allocator.alloc(T, current.items.len);
    @memcpy(subset, current.items);
    try results.append(subset);

    for (start..items.len) |i| {
        // Skip duplicates: if items[i] == items[i-1] and i > start,
        // this would create duplicate subsets
        if (i > start and items[i] == items[i - 1]) continue;

        try current.append(items[i]);
        try backtrackUnique(T, allocator, results, items, current, i + 1);
        _ = current.pop();
    }
}

test "Subsets: basic 3 elements" {
    const allocator = std.testing.allocator;
    const items = [_]i32{ 1, 2, 3 };

    var subs = try subsets(i32, allocator, &items);
    defer {
        for (subs.items) |subset| allocator.free(subset);
        subs.deinit();
    }

    // 2^3 = 8 subsets
    try std.testing.expectEqual(@as(usize, 8), subs.items.len);

    // Verify empty subset exists
    var found_empty = false;
    for (subs.items) |subset| {
        if (subset.len == 0) {
            found_empty = true;
            break;
        }
    }
    try std.testing.expect(found_empty);
}

test "Subsets: empty array" {
    const allocator = std.testing.allocator;
    const items = [_]i32{};

    var subs = try subsets(i32, allocator, &items);
    defer {
        for (subs.items) |subset| allocator.free(subset);
        subs.deinit();
    }

    // Only empty subset
    try std.testing.expectEqual(@as(usize, 1), subs.items.len);
    try std.testing.expectEqual(@as(usize, 0), subs.items[0].len);
}

test "Subsets: single element" {
    const allocator = std.testing.allocator;
    const items = [_]i32{42};

    var subs = try subsets(i32, allocator, &items);
    defer {
        for (subs.items) |subset| allocator.free(subset);
        subs.deinit();
    }

    // [] and [42]
    try std.testing.expectEqual(@as(usize, 2), subs.items.len);
}

test "Subsets: of size K" {
    const allocator = std.testing.allocator;
    const items = [_]i32{ 1, 2, 3, 4 };

    // Choose 2 elements from 4: C(4,2) = 6
    var subs = try subsetsOfSize(i32, allocator, &items, 2);
    defer {
        for (subs.items) |subset| allocator.free(subset);
        subs.deinit();
    }

    try std.testing.expectEqual(@as(usize, 6), subs.items.len);

    // Verify all have size 2
    for (subs.items) |subset| {
        try std.testing.expectEqual(@as(usize, 2), subset.len);
    }
}

test "Subsets: of size 0" {
    const allocator = std.testing.allocator;
    const items = [_]i32{ 1, 2, 3 };

    var subs = try subsetsOfSize(i32, allocator, &items, 0);
    defer {
        for (subs.items) |subset| allocator.free(subset);
        subs.deinit();
    }

    // Only empty subset
    try std.testing.expectEqual(@as(usize, 1), subs.items.len);
    try std.testing.expectEqual(@as(usize, 0), subs.items[0].len);
}

test "Subsets: of size larger than array" {
    const allocator = std.testing.allocator;
    const items = [_]i32{ 1, 2 };

    var subs = try subsetsOfSize(i32, allocator, &items, 5);
    defer {
        for (subs.items) |subset| allocator.free(subset);
        subs.deinit();
    }

    // Impossible, no subsets
    try std.testing.expectEqual(@as(usize, 0), subs.items.len);
}

test "Subsets: unique with duplicates" {
    const allocator = std.testing.allocator;
    const items = [_]i32{ 1, 2, 2 };

    var subs = try subsetsUnique(i32, allocator, &items);
    defer {
        for (subs.items) |subset| allocator.free(subset);
        subs.deinit();
    }

    // Without duplicates: [], [1], [2], [1,2], [2,2], [1,2,2] = 6 unique
    try std.testing.expectEqual(@as(usize, 6), subs.items.len);
}

test "Subsets: unique with all same" {
    const allocator = std.testing.allocator;
    const items = [_]i32{ 3, 3, 3 };

    var subs = try subsetsUnique(i32, allocator, &items);
    defer {
        for (subs.items) |subset| allocator.free(subset);
        subs.deinit();
    }

    // [], [3], [3,3], [3,3,3] = 4 unique
    try std.testing.expectEqual(@as(usize, 4), subs.items.len);
}

test "Subsets: stress test with 5 elements" {
    const allocator = std.testing.allocator;
    const items = [_]i32{ 1, 2, 3, 4, 5 };

    var subs = try subsets(i32, allocator, &items);
    defer {
        for (subs.items) |subset| allocator.free(subset);
        subs.deinit();
    }

    // 2^5 = 32
    try std.testing.expectEqual(@as(usize, 32), subs.items.len);
}
