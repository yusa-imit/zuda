const std = @import("std");
const testing = std.testing;

/// Activity with start and finish times
pub const Activity = struct {
    start: i32,
    finish: i32,
    id: usize,
};

/// Selects maximum number of non-overlapping activities using greedy approach
///
/// Algorithm: Sort by finish time, greedily select earliest-finishing compatible activity
///
/// Time: O(n log n) — sorting dominates
/// Space: O(n) — result storage
///
/// Example:
/// ```zig
/// const activities = [_]Activity{
///     .{ .start = 1, .finish = 3, .id = 0 },
///     .{ .start = 2, .finish = 5, .id = 1 },
///     .{ .start = 4, .finish = 6, .id = 2 },
/// };
/// var result = try activitySelection(testing.allocator, &activities);
/// defer result.deinit();
/// // result.items contains [0, 2] — activities 0 and 2 are selected
/// ```
pub fn activitySelection(
    allocator: std.mem.Allocator,
    activities: []const Activity,
) !std.ArrayList(usize) {
    if (activities.len == 0) {
        return std.ArrayList(usize).init(allocator);
    }

    // Sort by finish time
    var sorted = try allocator.alloc(Activity, activities.len);
    defer allocator.free(sorted);
    @memcpy(sorted, activities);

    std.mem.sort(Activity, sorted, {}, lessThanByFinish);

    var result = std.ArrayList(usize).init(allocator);
    errdefer result.deinit();

    // Always select first activity (earliest finish)
    try result.append(sorted[0].id);
    var last_finish = sorted[0].finish;

    // Greedily select compatible activities
    for (sorted[1..]) |activity| {
        if (activity.start >= last_finish) {
            try result.append(activity.id);
            last_finish = activity.finish;
        }
    }

    return result;
}

fn lessThanByFinish(_: void, a: Activity, b: Activity) bool {
    return a.finish < b.finish;
}

/// Weighted activity selection — maximizes total weight
///
/// Algorithm: Dynamic programming approach (greedy doesn't work for weighted case)
///
/// Time: O(n log n) — sorting + DP
/// Space: O(n) — DP table
pub fn weightedActivitySelection(
    allocator: std.mem.Allocator,
    activities: []const Activity,
    weights: []const f64,
) !std.ArrayList(usize) {
    if (activities.len == 0) {
        return std.ArrayList(usize).init(allocator);
    }

    std.debug.assert(activities.len == weights.len);

    // Sort by finish time
    const sorted_indices = try allocator.alloc(usize, activities.len);
    defer allocator.free(sorted_indices);
    for (sorted_indices, 0..) |*idx, i| idx.* = i;

    const Context = struct {
        activities: []const Activity,
        pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
            return ctx.activities[a].finish < ctx.activities[b].finish;
        }
    };
    std.mem.sort(usize, sorted_indices, Context{ .activities = activities }, Context.lessThan);

    // DP: dp[i] = max weight using activities 0..i
    var dp = try allocator.alloc(f64, activities.len);
    defer allocator.free(dp);

    var prev = try allocator.alloc(?usize, activities.len);
    defer allocator.free(prev);

    // Base case
    dp[0] = weights[sorted_indices[0]];
    prev[0] = null;

    // Fill DP table
    for (1..activities.len) |i| {
        const current_idx = sorted_indices[i];
        const current_weight = weights[current_idx];

        // Find latest compatible activity
        var latest_compatible: ?usize = null;
        var j: usize = i;
        while (j > 0) {
            j -= 1;
            if (activities[sorted_indices[j]].finish <= activities[current_idx].start) {
                latest_compatible = j;
                break;
            }
        }

        // Include current activity or skip
        const include_weight = if (latest_compatible) |lc| dp[lc] + current_weight else current_weight;
        const exclude_weight = dp[i - 1];

        if (include_weight > exclude_weight) {
            dp[i] = include_weight;
            prev[i] = latest_compatible;
        } else {
            dp[i] = exclude_weight;
            prev[i] = null; // Skip current
        }
    }

    // Reconstruct solution
    var result = std.ArrayList(usize).init(allocator);
    errdefer result.deinit();

    var i: ?usize = activities.len - 1;
    while (i) |idx| {
        if (prev[idx]) |p| {
            try result.append(sorted_indices[idx]);
            i = p;
        } else if (idx == 0 or dp[idx] != dp[idx - 1]) {
            try result.append(sorted_indices[idx]);
            break;
        } else {
            if (idx == 0) break;
            i = idx - 1;
        }
    }

    // Reverse to get chronological order
    std.mem.reverse(usize, result.items);
    return result;
}

// Tests
test "activity selection - basic case" {
    const activities = [_]Activity{
        .{ .start = 1, .finish = 3, .id = 0 },
        .{ .start = 2, .finish = 5, .id = 1 },
        .{ .start = 4, .finish = 6, .id = 2 },
        .{ .start = 6, .finish = 7, .id = 3 },
        .{ .start = 5, .finish = 9, .id = 4 },
        .{ .start = 8, .finish = 9, .id = 5 },
    };

    var result = try activitySelection(testing.allocator, &activities);
    defer result.deinit();

    // Should select activities: 0 (1-3), 2 (4-6), 3 (6-7), 5 (8-9)
    try testing.expectEqual(@as(usize, 4), result.items.len);
    try testing.expect(std.mem.indexOfScalar(usize, result.items, 0) != null);
    try testing.expect(std.mem.indexOfScalar(usize, result.items, 2) != null);
    try testing.expect(std.mem.indexOfScalar(usize, result.items, 3) != null);
    try testing.expect(std.mem.indexOfScalar(usize, result.items, 5) != null);
}

test "activity selection - empty" {
    const activities: []const Activity = &.{};
    var result = try activitySelection(testing.allocator, activities);
    defer result.deinit();
    try testing.expectEqual(@as(usize, 0), result.items.len);
}

test "activity selection - single activity" {
    const activities = [_]Activity{
        .{ .start = 1, .finish = 5, .id = 0 },
    };
    var result = try activitySelection(testing.allocator, activities);
    defer result.deinit();
    try testing.expectEqual(@as(usize, 1), result.items.len);
    try testing.expectEqual(@as(usize, 0), result.items[0]);
}

test "activity selection - all overlapping" {
    const activities = [_]Activity{
        .{ .start = 1, .finish = 10, .id = 0 },
        .{ .start = 2, .finish = 9, .id = 1 },
        .{ .start = 3, .finish = 8, .id = 2 },
    };
    var result = try activitySelection(testing.allocator, activities);
    defer result.deinit();
    // Should select activity 2 (earliest finish)
    try testing.expectEqual(@as(usize, 1), result.items.len);
    try testing.expectEqual(@as(usize, 2), result.items[0]);
}

test "activity selection - no overlaps" {
    const activities = [_]Activity{
        .{ .start = 1, .finish = 2, .id = 0 },
        .{ .start = 3, .finish = 4, .id = 1 },
        .{ .start = 5, .finish = 6, .id = 2 },
    };
    var result = try activitySelection(testing.allocator, activities);
    defer result.deinit();
    try testing.expectEqual(@as(usize, 3), result.items.len);
}

test "weighted activity selection - basic case" {
    const activities = [_]Activity{
        .{ .start = 1, .finish = 3, .id = 0 },
        .{ .start = 2, .finish = 5, .id = 1 },
        .{ .start = 4, .finish = 6, .id = 2 },
    };
    const weights = [_]f64{ 50, 200, 100 };

    var result = try weightedActivitySelection(testing.allocator, &activities, &weights);
    defer result.deinit();

    // Should select activity 1 (weight 200) — best value
    try testing.expect(result.items.len > 0);
    var total_weight: f64 = 0;
    for (result.items) |id| {
        total_weight += weights[id];
    }
    try testing.expect(total_weight >= 200); // At least as good as selecting activity 1
}

test "weighted activity selection - prefer high weight" {
    const activities = [_]Activity{
        .{ .start = 1, .finish = 3, .id = 0 },
        .{ .start = 2, .finish = 10, .id = 1 }, // High weight, long duration
        .{ .start = 4, .finish = 6, .id = 2 },
        .{ .start = 7, .finish = 9, .id = 3 },
    };
    const weights = [_]f64{ 10, 1000, 10, 10 }; // Activity 1 has very high weight

    var result = try weightedActivitySelection(testing.allocator, &activities, &weights);
    defer result.deinit();

    var total_weight: f64 = 0;
    for (result.items) |id| {
        total_weight += weights[id];
    }
    // Should select activity 1 (weight 1000) despite overlaps
    try testing.expect(total_weight >= 1000);
}
