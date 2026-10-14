//! Activity selection: greedy for the unweighted problem, dynamic programming for the weighted
//! one.
//!
//! Activities occupy the half-open interval `[start, finish)`: one that starts exactly when
//! another finishes is compatible with it.
//!
//! Ownership: returned `ArrayList(usize)` values are owned by the caller, who releases them with
//! `deinit(gpa)` using the same allocator. On error nothing is leaked.

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

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
/// Returns the `id` of each selected activity, in order of finish time.
///
/// Example:
/// ```zig
/// const activities = [_]Activity{
///     .{ .start = 1, .finish = 3, .id = 0 },
///     .{ .start = 2, .finish = 5, .id = 1 },
///     .{ .start = 4, .finish = 6, .id = 2 },
/// };
/// var result = try activitySelection(gpa, &activities);
/// defer result.deinit(gpa);
/// // result.items contains [0, 2] — activities 0 and 2 are selected
/// ```
pub fn activitySelection(
    gpa: Allocator,
    activities: []const Activity,
) !ArrayList(usize) {
    if (activities.len == 0) return .empty;

    // Sort by finish time
    const sorted = try gpa.alloc(Activity, activities.len);
    defer gpa.free(sorted);

    @memcpy(sorted, activities);
    std.mem.sort(Activity, sorted, {}, lessThanByFinish);

    var result: ArrayList(usize) = .empty;
    errdefer result.deinit(gpa);

    try result.ensureTotalCapacityPrecise(gpa, activities.len);

    // Always select first activity (earliest finish)
    result.appendAssumeCapacity(sorted[0].id);
    var last_finish = sorted[0].finish;

    // Greedily select compatible activities
    for (sorted[1..]) |activity| {
        if (activity.start >= last_finish) {
            result.appendAssumeCapacity(activity.id);
            last_finish = activity.finish;
        }
    }

    assert(result.items.len >= 1);
    assert(result.items.len <= activities.len);
    return result;
}

fn lessThanByFinish(_: void, a: Activity, b: Activity) bool {
    return a.finish < b.finish;
}

/// Weighted activity selection — maximizes total weight
///
/// Algorithm: Dynamic programming approach (greedy doesn't work for weighted case). Activities
/// are sorted by finish time; for each one a binary search finds how many earlier activities
/// are compatible. An activity with non-positive weight is never selected.
///
/// Time: O(n log n) — sorting + DP with binary search
/// Space: O(n) — DP table
///
/// Returns the index into `activities` (and `weights`) of each selected activity, in order of
/// finish time. `weights.len` must equal `activities.len`.
pub fn weightedActivitySelection(
    gpa: Allocator,
    activities: []const Activity,
    weights: []const f64,
) !ArrayList(usize) {
    assert(activities.len == weights.len);
    if (activities.len == 0) return .empty;

    // Sort indices by finish time.
    const order = try gpa.alloc(usize, activities.len);
    defer gpa.free(order);

    for (order, 0..) |*idx, i| idx.* = i;

    const Context = struct {
        activities: []const Activity,
        pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
            return ctx.activities[a].finish < ctx.activities[b].finish;
        }
    };
    std.mem.sort(usize, order, Context{ .activities = activities }, Context.lessThan);

    // best[k] = max weight using the first k activities in finish order (best[0] = 0).
    // compatible[i] = how many leading activities finish no later than order[i] starts.
    const best = try gpa.alloc(f64, activities.len + 1);
    defer gpa.free(best);

    const compatible = try gpa.alloc(usize, activities.len);
    defer gpa.free(compatible);

    const taken = try gpa.alloc(bool, activities.len);
    defer gpa.free(taken);

    best[0] = 0.0;
    for (order, 0..) |idx, i| {
        compatible[i] = countCompatible(activities, order[0..i], activities[idx].start);
        assert(compatible[i] <= i);
        const include_weight = best[compatible[i]] + weights[idx];
        taken[i] = include_weight > best[i];
        best[i + 1] = if (taken[i]) include_weight else best[i];
    }

    // Reconstruct solution
    var result: ArrayList(usize) = .empty;
    errdefer result.deinit(gpa);

    try result.ensureTotalCapacityPrecise(gpa, activities.len);

    var k = activities.len;
    // Every step lowers `k` by at least one, so `activities.len` steps always suffice.
    for (0..activities.len) |_| {
        if (k == 0) break;
        if (taken[k - 1]) {
            result.appendAssumeCapacity(order[k - 1]);
            k = compatible[k - 1];
        } else {
            k -= 1;
        }
    }
    assert(k == 0);

    // Reverse to get chronological order
    std.mem.reverse(usize, result.items);
    return result;
}

/// Counts the leading activities of `order` (sorted by finish) that finish by `start`.
///
/// Because `order` is sorted by finish time the compatible ones form a prefix, so a binary
/// search finds its length.
fn countCompatible(activities: []const Activity, order: []const usize, start: i32) usize {
    var low: usize = 0;
    var high: usize = order.len;
    // The interval [low, high) halves each pass, so `order.len + 1` passes always suffice.
    for (0..order.len + 1) |_| {
        if (low >= high) break;

        const mid = low + @divFloor(high - low, 2);
        if (activities[order[mid]].finish <= start) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    assert(low == high);
    assert(low <= order.len);
    return low;
}

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
    defer result.deinit(testing.allocator);

    // Should select activities: 0 (1-3), 2 (4-6), 3 (6-7), 5 (8-9)
    try testing.expectEqual(@as(usize, 4), result.items.len);
    try testing.expect(std.mem.findScalar(usize, result.items, 0) != null);
    try testing.expect(std.mem.findScalar(usize, result.items, 2) != null);
    try testing.expect(std.mem.findScalar(usize, result.items, 3) != null);
    try testing.expect(std.mem.findScalar(usize, result.items, 5) != null);
}

test "activity selection - empty" {
    const activities: []const Activity = &.{};
    var result = try activitySelection(testing.allocator, activities);
    defer result.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), result.items.len);
}

test "activity selection - single activity" {
    const activities = [_]Activity{
        .{ .start = 1, .finish = 5, .id = 0 },
    };
    var result = try activitySelection(testing.allocator, &activities);
    defer result.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), result.items.len);
    try testing.expectEqual(@as(usize, 0), result.items[0]);
}

test "activity selection - all overlapping" {
    const activities = [_]Activity{
        .{ .start = 1, .finish = 10, .id = 0 },
        .{ .start = 2, .finish = 9, .id = 1 },
        .{ .start = 3, .finish = 8, .id = 2 },
    };
    var result = try activitySelection(testing.allocator, &activities);
    defer result.deinit(testing.allocator);
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
    var result = try activitySelection(testing.allocator, &activities);
    defer result.deinit(testing.allocator);
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
    defer result.deinit(testing.allocator);

    // Activity 1 (weight 200) beats activities 0 + 2 (50 + 100) and overlaps both.
    try testing.expectEqualSlices(usize, &.{1}, result.items);
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
    defer result.deinit(testing.allocator);

    // Activity 1 (weight 1000) beats the three short ones (10 + 10 + 10) despite the overlaps.
    try testing.expectEqualSlices(usize, &.{1}, result.items);
}

test "activity selection - adjacent activities both selected" {
    // start == finish of previous: activity.start >= last_finish, so adjacent activities are compatible
    const activities = [_]Activity{
        .{ .start = 1, .finish = 5, .id = 0 },
        .{ .start = 5, .finish = 9, .id = 1 },
        .{ .start = 9, .finish = 12, .id = 2 },
    };
    var result = try activitySelection(testing.allocator, &activities);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 3), result.items.len);
}

test "activity selection - unsorted input selects correct count" {
    // Input NOT sorted by finish time — algorithm must sort internally
    const activities = [_]Activity{
        .{ .start = 5, .finish = 9, .id = 0 },
        .{ .start = 1, .finish = 3, .id = 1 },
        .{ .start = 7, .finish = 11, .id = 2 },
        .{ .start = 3, .finish = 6, .id = 3 },
    };
    var result = try activitySelection(testing.allocator, &activities);
    defer result.deinit(testing.allocator);

    // Sorted by finish: id1(1-3), id3(3-6), id0(5-9), id2(7-11)
    // Select id1(1-3), id3(3-6 start>=3 ✓), id2(7-11 start>=6 ✓) = 3
    try testing.expectEqual(@as(usize, 3), result.items.len);
}

test "activity selection - ten non-overlapping activities all selected" {
    var activities: [10]Activity = undefined;
    for (&activities, 0..) |*a, i| {
        a.* = .{
            .start = @intCast(i * 2),
            .finish = @intCast(i * 2 + 1),
            .id = i,
        };
    }
    var result = try activitySelection(testing.allocator, &activities);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 10), result.items.len);
}

test "weighted activity selection - empty gives empty result" {
    const activities: []const Activity = &.{};
    const weights: []const f64 = &.{};
    var result = try weightedActivitySelection(testing.allocator, activities, weights);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 0), result.items.len);
}

test "activity selection - memory safety loop" {
    const activities = [_]Activity{
        .{ .start = 1, .finish = 3, .id = 0 },
        .{ .start = 4, .finish = 6, .id = 1 },
        .{ .start = 5, .finish = 8, .id = 2 },
    };
    for (0..10) |_| {
        var result = try activitySelection(testing.allocator, &activities);
        defer result.deinit(testing.allocator);
        try testing.expect(result.items.len > 0);
    }
}

test "weighted activity selection - combines compatible activities" {
    const activities = [_]Activity{
        .{ .start = 0, .finish = 3, .id = 0 },
        .{ .start = 3, .finish = 6, .id = 1 },
        .{ .start = 2, .finish = 5, .id = 2 },
        .{ .start = 6, .finish = 9, .id = 3 },
    };
    const weights = [_]f64{ 5, 6, 12, 4 };

    var result = try weightedActivitySelection(testing.allocator, &activities, &weights);
    defer result.deinit(testing.allocator);

    // {0,1,3} weighs 15, {2,3} weighs 16: activity 2 overlaps 0 and 1 but is worth more.
    try testing.expectEqualSlices(usize, &.{ 2, 3 }, result.items);
}

test "weighted activity selection - never selects a negative weight" {
    const activities = [_]Activity{
        .{ .start = 0, .finish = 2, .id = 0 },
        .{ .start = 3, .finish = 5, .id = 1 },
    };
    const weights = [_]f64{ -4, 7 };

    var result = try weightedActivitySelection(testing.allocator, &activities, &weights);
    defer result.deinit(testing.allocator);

    try testing.expectEqualSlices(usize, &.{1}, result.items);

    const all_negative = [_]f64{ -4, -7 };
    var none = try weightedActivitySelection(testing.allocator, &activities, &all_negative);
    defer none.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 0), none.items.len);
}

test "activity selection - no leaks when an allocation fails" {
    const Runner = struct {
        fn run(gpa: Allocator) !void {
            const activities = [_]Activity{
                .{ .start = 1, .finish = 3, .id = 0 },
                .{ .start = 2, .finish = 5, .id = 1 },
                .{ .start = 4, .finish = 6, .id = 2 },
            };
            const weights = [_]f64{ 50, 200, 100 };

            var selected = try activitySelection(gpa, &activities);
            defer selected.deinit(gpa);

            var weighted = try weightedActivitySelection(gpa, &activities, &weights);
            defer weighted.deinit(gpa);

            try testing.expectEqualSlices(usize, &.{ 0, 2 }, selected.items);
            try testing.expectEqualSlices(usize, &.{1}, weighted.items);
        }
    };

    try testing.checkAllAllocationFailures(testing.allocator, Runner.run, .{});
}
