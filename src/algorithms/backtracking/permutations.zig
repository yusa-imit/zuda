const std = @import("std");
const ArrayList = std.ArrayList;

/// Generate all permutations of an array using backtracking.
///
/// Time: O(N! * N) - N! permutations, each takes O(N) to copy
/// Space: O(N! * N) for storing all permutations
pub fn permute(comptime T: type, allocator: std.mem.Allocator, items: []const T) !ArrayList([]T) {
    var results = ArrayList([]T).init(allocator);
    errdefer {
        for (results.items) |perm| allocator.free(perm);
        results.deinit();
    }

    if (items.len == 0) return results;

    const current = try allocator.alloc(T, items.len);
    defer allocator.free(current);
    @memcpy(current, items);

    const used = try allocator.alloc(bool, items.len);
    defer allocator.free(used);
    @memset(used, false);

    try backtrack(T, allocator, &results, items, current, used, 0);
    return results;
}

fn backtrack(
    comptime T: type,
    allocator: std.mem.Allocator,
    results: *ArrayList([]T),
    items: []const T,
    current: []T,
    used: []bool,
    pos: usize,
) !void {
    if (pos == items.len) {
        // Found a complete permutation
        const perm = try allocator.alloc(T, items.len);
        @memcpy(perm, current);
        try results.append(perm);
        return;
    }

    for (0..items.len) |i| {
        if (!used[i]) {
            current[pos] = items[i];
            used[i] = true;
            try backtrack(T, allocator, results, items, current, used, pos + 1);
            used[i] = false; // Backtrack
        }
    }
}

/// Generate all unique permutations of an array (handles duplicates).
///
/// Time: O(N! * N) worst case, but pruning reduces it for duplicates
/// Space: O(N! * N)
pub fn permuteUnique(comptime T: type, allocator: std.mem.Allocator, items: []const T) !ArrayList([]T) {
    var results = ArrayList([]T).init(allocator);
    errdefer {
        for (results.items) |perm| allocator.free(perm);
        results.deinit();
    }

    if (items.len == 0) return results;

    // Sort to group duplicates
    const sorted = try allocator.alloc(T, items.len);
    defer allocator.free(sorted);
    @memcpy(sorted, items);
    std.mem.sort(T, sorted, {}, std.sort.asc(T));

    const current = try allocator.alloc(T, items.len);
    defer allocator.free(current);

    const used = try allocator.alloc(bool, items.len);
    defer allocator.free(used);
    @memset(used, false);

    try backtrackUnique(T, allocator, &results, sorted, current, used, 0);
    return results;
}

fn backtrackUnique(
    comptime T: type,
    allocator: std.mem.Allocator,
    results: *ArrayList([]T),
    items: []const T,
    current: []T,
    used: []bool,
    pos: usize,
) !void {
    if (pos == items.len) {
        const perm = try allocator.alloc(T, items.len);
        @memcpy(perm, current);
        try results.append(perm);
        return;
    }

    for (0..items.len) |i| {
        // Skip if used
        if (used[i]) continue;

        // Skip duplicates: if items[i] == items[i-1] and items[i-1] not used,
        // then items[i] would generate duplicate permutations
        if (i > 0 and items[i] == items[i - 1] and !used[i - 1]) continue;

        current[pos] = items[i];
        used[i] = true;
        try backtrackUnique(T, allocator, results, items, current, used, pos + 1);
        used[i] = false;
    }
}

test "Permutations: basic 3 elements" {
    const allocator = std.testing.allocator;
    const items = [_]i32{ 1, 2, 3 };

    var perms = try permute(i32, allocator, &items);
    defer {
        for (perms.items) |perm| allocator.free(perm);
        perms.deinit();
    }

    // 3! = 6 permutations
    try std.testing.expectEqual(@as(usize, 6), perms.items.len);

    // Verify one known permutation exists
    var found = false;
    for (perms.items) |perm| {
        if (perm[0] == 1 and perm[1] == 2 and perm[2] == 3) {
            found = true;
            break;
        }
    }
    try std.testing.expect(found);
}

test "Permutations: empty array" {
    const allocator = std.testing.allocator;
    const items = [_]i32{};

    var perms = try permute(i32, allocator, &items);
    defer perms.deinit();

    try std.testing.expectEqual(@as(usize, 0), perms.items.len);
}

test "Permutations: single element" {
    const allocator = std.testing.allocator;
    const items = [_]i32{42};

    var perms = try permute(i32, allocator, &items);
    defer {
        for (perms.items) |perm| allocator.free(perm);
        perms.deinit();
    }

    try std.testing.expectEqual(@as(usize, 1), perms.items.len);
    try std.testing.expectEqual(@as(i32, 42), perms.items[0][0]);
}

test "Permutations: two elements" {
    const allocator = std.testing.allocator;
    const items = [_]i32{ 1, 2 };

    var perms = try permute(i32, allocator, &items);
    defer {
        for (perms.items) |perm| allocator.free(perm);
        perms.deinit();
    }

    // 2! = 2
    try std.testing.expectEqual(@as(usize, 2), perms.items.len);
}

test "Permutations: unique with duplicates" {
    const allocator = std.testing.allocator;
    const items = [_]i32{ 1, 1, 2 };

    var perms = try permuteUnique(i32, allocator, &items);
    defer {
        for (perms.items) |perm| allocator.free(perm);
        perms.deinit();
    }

    // Should be 3!/2! = 3 unique permutations: [1,1,2], [1,2,1], [2,1,1]
    try std.testing.expectEqual(@as(usize, 3), perms.items.len);
}

test "Permutations: unique with all same" {
    const allocator = std.testing.allocator;
    const items = [_]i32{ 5, 5, 5 };

    var perms = try permuteUnique(i32, allocator, &items);
    defer {
        for (perms.items) |perm| allocator.free(perm);
        perms.deinit();
    }

    // Only 1 unique permutation
    try std.testing.expectEqual(@as(usize, 1), perms.items.len);
    try std.testing.expectEqual(@as(i32, 5), perms.items[0][0]);
    try std.testing.expectEqual(@as(i32, 5), perms.items[0][1]);
    try std.testing.expectEqual(@as(i32, 5), perms.items[0][2]);
}

test "Permutations: stress test with 4 elements" {
    const allocator = std.testing.allocator;
    const items = [_]i32{ 1, 2, 3, 4 };

    var perms = try permute(i32, allocator, &items);
    defer {
        for (perms.items) |perm| allocator.free(perm);
        perms.deinit();
    }

    // 4! = 24
    try std.testing.expectEqual(@as(usize, 24), perms.items.len);

    // Verify all permutations are distinct
    for (perms.items, 0..) |perm1, i| {
        for (perms.items[i + 1 ..]) |perm2| {
            const equal = std.mem.eql(i32, perm1, perm2);
            try std.testing.expect(!equal);
        }
    }
}

test "Permutations: five elements gives 120 permutations" {
    const allocator = std.testing.allocator;
    const items = [_]i32{ 1, 2, 3, 4, 5 };

    var perms = try permute(i32, allocator, &items);
    defer {
        for (perms.items) |perm| allocator.free(perm);
        perms.deinit();
    }

    try std.testing.expectEqual(@as(usize, 120), perms.items.len);
    for (perms.items) |perm| {
        try std.testing.expectEqual(@as(usize, 5), perm.len);
    }
}

test "Permutations: reverse permutation present in results" {
    const allocator = std.testing.allocator;
    const items = [_]i32{ 1, 2, 3 };

    var perms = try permute(i32, allocator, &items);
    defer {
        for (perms.items) |perm| allocator.free(perm);
        perms.deinit();
    }

    var found_reverse = false;
    for (perms.items) |perm| {
        if (perm[0] == 3 and perm[1] == 2 and perm[2] == 1) {
            found_reverse = true;
            break;
        }
    }
    try std.testing.expect(found_reverse);
}

test "Permutations: unique with two pairs of duplicates" {
    const allocator = std.testing.allocator;
    const items = [_]i32{ 1, 1, 2, 2 };

    var perms = try permuteUnique(i32, allocator, &items);
    defer {
        for (perms.items) |perm| allocator.free(perm);
        perms.deinit();
    }

    // 4! / (2! * 2!) = 6 unique permutations
    try std.testing.expectEqual(@as(usize, 6), perms.items.len);
}

test "Permutations: unique with no duplicates matches permute count" {
    const allocator = std.testing.allocator;
    const items = [_]i32{ 1, 2, 3 };

    var all_perms = try permute(i32, allocator, &items);
    defer {
        for (all_perms.items) |perm| allocator.free(perm);
        all_perms.deinit();
    }

    var unique_perms = try permuteUnique(i32, allocator, &items);
    defer {
        for (unique_perms.items) |perm| allocator.free(perm);
        unique_perms.deinit();
    }

    try std.testing.expectEqual(all_perms.items.len, unique_perms.items.len);
}

test "Permutations: memory safety loop" {
    const allocator = std.testing.allocator;
    const items = [_]i32{ 10, 20, 30 };
    const dup_items = [_]i32{ 1, 1, 2 };

    for (0..10) |_| {
        var perms = try permute(i32, allocator, &items);
        for (perms.items) |perm| allocator.free(perm);
        perms.deinit();

        var uniqs = try permuteUnique(i32, allocator, &dup_items);
        for (uniqs.items) |perm| allocator.free(perm);
        uniqs.deinit();
    }
}
