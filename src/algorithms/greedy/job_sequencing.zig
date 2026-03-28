const std = @import("std");
const testing = std.testing;

/// Job with deadline and profit
pub const Job = struct {
    id: usize,
    deadline: usize, // Time slot (1-based)
    profit: i64,
};

/// Job sequencing result
pub const JobSequence = struct {
    jobs: []const usize, // Selected job IDs in execution order
    total_profit: i64,
    slots_used: usize,
};

/// Solves job sequencing with deadlines using greedy approach
///
/// Algorithm: Sort by profit (descending), assign each job to latest available slot before deadline
///
/// Time: O(n²) — for each job, find slot
/// Space: O(n) — slot array
///
/// Example:
/// ```zig
/// const jobs = [_]Job{
///     .{ .id = 0, .deadline = 2, .profit = 100 },
///     .{ .id = 1, .deadline = 1, .profit = 50 },
///     .{ .id = 2, .deadline = 2, .profit = 10 },
/// };
/// var result = try jobSequencing(allocator, &jobs);
/// defer allocator.free(result.jobs);
/// // result.jobs = [1, 0], total_profit = 150
/// ```
pub fn jobSequencing(
    allocator: std.mem.Allocator,
    jobs: []const Job,
) !JobSequence {
    if (jobs.len == 0) {
        return .{
            .jobs = &.{},
            .total_profit = 0,
            .slots_used = 0,
        };
    }

    // Find maximum deadline to determine number of slots
    var max_deadline: usize = 0;
    for (jobs) |job| {
        if (job.deadline > max_deadline) {
            max_deadline = job.deadline;
        }
    }

    // Sort jobs by profit (descending)
    const sorted = try allocator.alloc(Job, jobs.len);
    defer allocator.free(sorted);
    @memcpy(sorted, jobs);

    std.mem.sort(Job, sorted, {}, greaterByProfit);

    // Slot array: slots[i] = job ID scheduled at time slot i (0 = empty)
    var slots = try allocator.alloc(?usize, max_deadline);
    defer allocator.free(slots);
    @memset(slots, null);

    var total_profit: i64 = 0;
    var selected_jobs = std.ArrayList(usize).init(allocator);
    defer selected_jobs.deinit();

    // Assign jobs to slots
    for (sorted) |job| {
        // Find latest available slot before deadline
        var slot: usize = job.deadline;
        while (slot > 0) : (slot -= 1) {
            if (slots[slot - 1] == null) {
                slots[slot - 1] = job.id;
                total_profit += job.profit;
                try selected_jobs.append(job.id);
                break;
            }
        }
    }

    // Build result in execution order
    var result_jobs = try allocator.alloc(usize, selected_jobs.items.len);
    var result_idx: usize = 0;
    for (slots) |maybe_job| {
        if (maybe_job) |job_id| {
            result_jobs[result_idx] = job_id;
            result_idx += 1;
        }
    }

    return .{
        .jobs = result_jobs,
        .total_profit = total_profit,
        .slots_used = result_idx,
    };
}

fn greaterByProfit(_: void, a: Job, b: Job) bool {
    return a.profit > b.profit;
}

/// Job sequencing with weighted completion times (minimize weighted sum)
///
/// Algorithm: Sort by weight/processing_time ratio (Smith's rule)
///
/// Time: O(n log n)
/// Space: O(n)
pub fn jobSequencingWeighted(
    allocator: std.mem.Allocator,
    jobs: []const WeightedJob,
) !std.ArrayList(usize) {
    var result = std.ArrayList(usize).init(allocator);
    errdefer result.deinit();

    if (jobs.len == 0) return result;

    // Sort by weight/time ratio (descending) — Smith's rule
    const sorted_indices = try allocator.alloc(usize, jobs.len);
    defer allocator.free(sorted_indices);
    for (sorted_indices, 0..) |*idx, i| idx.* = i;

    const Context = struct {
        jobs: []const WeightedJob,
        pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
            const ratio_a = ctx.jobs[a].ratio();
            const ratio_b = ctx.jobs[b].ratio();
            return ratio_a > ratio_b; // Descending
        }
    };

    std.mem.sort(usize, sorted_indices, Context{ .jobs = jobs }, Context.lessThan);

    for (sorted_indices) |idx| {
        try result.append(idx);
    }

    return result;
}

pub const WeightedJob = struct {
    id: usize,
    processing_time: f64,
    weight: f64, // Importance/priority

    pub fn ratio(self: WeightedJob) f64 {
        return if (self.processing_time > 0) self.weight / self.processing_time else 0;
    }
};

// Tests
test "job sequencing - basic case" {
    const jobs = [_]Job{
        .{ .id = 0, .deadline = 4, .profit = 20 },
        .{ .id = 1, .deadline = 1, .profit = 10 },
        .{ .id = 2, .deadline = 1, .profit = 40 },
        .{ .id = 3, .deadline = 1, .profit = 30 },
    };

    const result = try jobSequencing(testing.allocator, &jobs);
    defer testing.allocator.free(result.jobs);

    // Should select jobs 2 (profit 40) and 0 (profit 20) = 60
    // Job 2 at slot 1 (deadline 1), job 0 at some slot <= 4
    try testing.expectEqual(@as(i64, 60), result.total_profit);
    try testing.expectEqual(@as(usize, 2), result.jobs.len);
}

test "job sequencing - all same deadline" {
    const jobs = [_]Job{
        .{ .id = 0, .deadline = 1, .profit = 100 },
        .{ .id = 1, .deadline = 1, .profit = 50 },
        .{ .id = 2, .deadline = 1, .profit = 25 },
    };

    const result = try jobSequencing(testing.allocator, &jobs);
    defer testing.allocator.free(result.jobs);

    // Only one slot available, select highest profit
    try testing.expectEqual(@as(i64, 100), result.total_profit);
    try testing.expectEqual(@as(usize, 1), result.jobs.len);
    try testing.expectEqual(@as(usize, 0), result.jobs[0]);
}

test "job sequencing - empty" {
    const jobs: []const Job = &.{};
    const result = try jobSequencing(testing.allocator, jobs);
    defer testing.allocator.free(result.jobs);

    try testing.expectEqual(@as(i64, 0), result.total_profit);
    try testing.expectEqual(@as(usize, 0), result.jobs.len);
}

test "job sequencing - all jobs fit" {
    const jobs = [_]Job{
        .{ .id = 0, .deadline = 1, .profit = 10 },
        .{ .id = 1, .deadline = 2, .profit = 20 },
        .{ .id = 2, .deadline = 3, .profit = 30 },
    };

    const result = try jobSequencing(testing.allocator, &jobs);
    defer testing.allocator.free(result.jobs);

    try testing.expectEqual(@as(i64, 60), result.total_profit);
    try testing.expectEqual(@as(usize, 3), result.jobs.len);
}

test "job sequencing - late deadlines" {
    const jobs = [_]Job{
        .{ .id = 0, .deadline = 10, .profit = 100 },
        .{ .id = 1, .deadline = 10, .profit = 50 },
    };

    const result = try jobSequencing(testing.allocator, &jobs);
    defer testing.allocator.free(result.jobs);

    // Both jobs can be scheduled
    try testing.expectEqual(@as(i64, 150), result.total_profit);
    try testing.expectEqual(@as(usize, 2), result.jobs.len);
}

test "weighted job sequencing - Smith's rule" {
    const jobs = [_]WeightedJob{
        .{ .id = 0, .processing_time = 2, .weight = 3 }, // ratio = 1.5
        .{ .id = 1, .processing_time = 4, .weight = 5 }, // ratio = 1.25
        .{ .id = 2, .processing_time = 1, .weight = 4 }, // ratio = 4.0
    };

    var sequence = try jobSequencingWeighted(testing.allocator, &jobs);
    defer sequence.deinit();

    try testing.expectEqual(@as(usize, 3), sequence.items.len);
    // Should order by ratio (descending): job 2, job 0, job 1
    try testing.expectEqual(@as(usize, 2), sequence.items[0]);
    try testing.expectEqual(@as(usize, 0), sequence.items[1]);
    try testing.expectEqual(@as(usize, 1), sequence.items[2]);
}

test "weighted job sequencing - equal ratios" {
    const jobs = [_]WeightedJob{
        .{ .id = 0, .processing_time = 2, .weight = 4 }, // ratio = 2.0
        .{ .id = 1, .processing_time = 1, .weight = 2 }, // ratio = 2.0
    };

    var sequence = try jobSequencingWeighted(testing.allocator, &jobs);
    defer sequence.deinit();

    try testing.expectEqual(@as(usize, 2), sequence.items.len);
}

test "weighted job sequencing - empty" {
    const jobs: []const WeightedJob = &.{};
    var sequence = try jobSequencingWeighted(testing.allocator, jobs);
    defer sequence.deinit();

    try testing.expectEqual(@as(usize, 0), sequence.items.len);
}
