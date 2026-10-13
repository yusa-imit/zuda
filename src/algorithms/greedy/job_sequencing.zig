//! Job sequencing: greedy scheduling of unit-time jobs with deadlines and profits, and Smith's
//! rule for ordering jobs by weighted completion time.
//!
//! Ownership: `JobSequence.jobs` is allocated with the caller's `gpa` and must be freed by the
//! caller with the same allocator (an empty sequence owns nothing, and freeing it is a no-op).
//! The `ArrayList(usize)` from `jobSequencingWeighted` is released with `deinit(gpa)`.

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

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
/// Algorithm: Sort by profit (descending), assign each job to latest available slot before
/// deadline. Jobs with negative profit are never scheduled, and a job whose deadline is zero
/// cannot be scheduled. A deadline beyond the number of jobs is clamped to it, since that many
/// slots are always enough.
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
/// var result = try jobSequencing(gpa, &jobs);
/// defer gpa.free(result.jobs);
/// // result.jobs = [1, 0], total_profit = 150
/// ```
pub fn jobSequencing(
    gpa: Allocator,
    jobs: []const Job,
) !JobSequence {
    if (jobs.len == 0) {
        return .{
            .jobs = &.{},
            .total_profit = 0,
            .slots_used = 0,
        };
    }

    // Sort jobs by profit (descending)
    const sorted = try gpa.alloc(Job, jobs.len);
    defer gpa.free(sorted);

    @memcpy(sorted, jobs);
    std.mem.sort(Job, sorted, {}, greaterByProfit);

    // Slot array: slots[i] = job ID scheduled at time slot i + 1. At most `jobs.len` jobs can
    // run, so a later deadline never needs a later slot.
    const slots = try gpa.alloc(?usize, jobs.len);
    defer gpa.free(slots);

    @memset(slots, null);

    var total_profit: i64 = 0;
    var scheduled_count: usize = 0;

    // Assign jobs to slots
    for (sorted) |job| {
        if (job.profit < 0) break; // Sorted descending: everything after is a loss too.

        // Find latest available slot before deadline
        var slot = @min(job.deadline, jobs.len);
        for (0..slot) |_| {
            if (slots[slot - 1] == null) {
                slots[slot - 1] = job.id;
                total_profit += job.profit;
                scheduled_count += 1;
                break;
            }
            slot -= 1;
        }
    }

    // Build result in execution order
    const result_jobs = try gpa.alloc(usize, scheduled_count);
    var result_idx: usize = 0;
    for (slots) |maybe_job| {
        if (maybe_job) |job_id| {
            result_jobs[result_idx] = job_id;
            result_idx += 1;
        }
    }

    assert(result_idx == scheduled_count);
    assert(result_idx <= jobs.len);
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
/// Algorithm: Sort by weight/processing_time ratio (Smith's rule). Ties keep input order.
///
/// Time: O(n log n)
/// Space: O(n)
///
/// Returns the index into `jobs` of each job, in execution order.
pub fn jobSequencingWeighted(
    gpa: Allocator,
    jobs: []const WeightedJob,
) !ArrayList(usize) {
    var result: ArrayList(usize) = .empty;
    errdefer result.deinit(gpa);

    if (jobs.len == 0) return result;

    // Sort by weight/time ratio (descending) — Smith's rule
    try result.ensureTotalCapacityPrecise(gpa, jobs.len);
    for (0..jobs.len) |i| result.appendAssumeCapacity(i);

    const Context = struct {
        jobs: []const WeightedJob,
        pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
            const ratio_a = ctx.jobs[a].ratio();
            const ratio_b = ctx.jobs[b].ratio();
            return ratio_a > ratio_b; // Descending
        }
    };

    std.mem.sort(usize, result.items, Context{ .jobs = jobs }, Context.lessThan);

    assert(result.items.len == jobs.len);
    return result;
}

pub const WeightedJob = struct {
    id: usize,
    processing_time: f64,
    weight: f64, // Importance/priority

    /// Weight per unit of processing time. A job that takes no time has an infinite ratio when
    /// its weight is positive (it should run first), minus infinity when negative, and zero
    /// when its weight is zero.
    pub fn ratio(self: WeightedJob) f64 {
        if (self.processing_time > 0) return self.weight / self.processing_time;
        if (self.weight > 0) return std.math.inf(f64);
        if (self.weight < 0) return -std.math.inf(f64);
        return 0;
    }
};

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
    defer sequence.deinit(testing.allocator);

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
    defer sequence.deinit(testing.allocator);

    // Equal ratios keep input order.
    try testing.expectEqualSlices(usize, &.{ 0, 1 }, sequence.items);
}

test "weighted job sequencing - empty" {
    const jobs: []const WeightedJob = &.{};
    var sequence = try jobSequencingWeighted(testing.allocator, jobs);
    defer sequence.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 0), sequence.items.len);
}

test "job sequencing - single job selected and profit matches" {
    const jobs = [_]Job{
        .{ .id = 0, .deadline = 1, .profit = 42 },
    };
    const result = try jobSequencing(testing.allocator, &jobs);
    defer testing.allocator.free(result.jobs);

    try testing.expectEqual(@as(i64, 42), result.total_profit);
    try testing.expectEqual(@as(usize, 1), result.jobs.len);
    try testing.expectEqual(@as(usize, 0), result.jobs[0]);
}

test "job sequencing - highest profit wins when slot contested" {
    // Three jobs all need slot 1: only the highest profit (100) should be selected
    const jobs = [_]Job{
        .{ .id = 0, .deadline = 1, .profit = 50 },
        .{ .id = 1, .deadline = 1, .profit = 100 },
        .{ .id = 2, .deadline = 1, .profit = 30 },
    };
    const result = try jobSequencing(testing.allocator, &jobs);
    defer testing.allocator.free(result.jobs);

    try testing.expectEqual(@as(i64, 100), result.total_profit);
    try testing.expectEqual(@as(usize, 1), result.jobs.len);
}

test "weighted job sequencing - single job sequence has one entry" {
    const jobs = [_]WeightedJob{
        .{ .id = 0, .processing_time = 3, .weight = 6 },
    };
    var sequence = try jobSequencingWeighted(testing.allocator, &jobs);
    defer sequence.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), sequence.items.len);
    try testing.expectEqual(@as(usize, 0), sequence.items[0]);
}

test "job sequencing - total profit is sum of selected jobs" {
    const jobs = [_]Job{
        .{ .id = 0, .deadline = 1, .profit = 20 },
        .{ .id = 1, .deadline = 2, .profit = 15 },
        .{ .id = 2, .deadline = 3, .profit = 10 },
        .{ .id = 3, .deadline = 1, .profit = 5 }, // Slot 1 taken by id=0 (higher profit)
    };
    const result = try jobSequencing(testing.allocator, &jobs);
    defer testing.allocator.free(result.jobs);

    // Jobs 0, 1, 2 should be selected (total = 45)
    try testing.expectEqual(@as(i64, 45), result.total_profit);
    try testing.expectEqual(@as(usize, 3), result.jobs.len);
}

test "job sequencing - memory safety loop" {
    const jobs = [_]Job{
        .{ .id = 0, .deadline = 2, .profit = 100 },
        .{ .id = 1, .deadline = 1, .profit = 50 },
        .{ .id = 2, .deadline = 2, .profit = 10 },
    };
    for (0..10) |_| {
        const result = try jobSequencing(testing.allocator, &jobs);
        defer testing.allocator.free(result.jobs);
        try testing.expect(result.total_profit > 0);
    }
}

test "job sequencing - execution order follows the slots" {
    const jobs = [_]Job{
        .{ .id = 7, .deadline = 2, .profit = 100 },
        .{ .id = 8, .deadline = 1, .profit = 50 },
        .{ .id = 9, .deadline = 2, .profit = 10 },
    };
    const result = try jobSequencing(testing.allocator, &jobs);
    defer testing.allocator.free(result.jobs);

    try testing.expectEqualSlices(usize, &.{ 8, 7 }, result.jobs);
    try testing.expectEqual(@as(i64, 150), result.total_profit);
    try testing.expectEqual(@as(usize, 2), result.slots_used);
}

test "job sequencing - huge deadlines need no huge allocation" {
    const jobs = [_]Job{
        .{ .id = 0, .deadline = std.math.maxInt(usize), .profit = 5 },
        .{ .id = 1, .deadline = std.math.maxInt(usize), .profit = 9 },
        .{ .id = 2, .deadline = 0, .profit = 1000 },
    };
    const result = try jobSequencing(testing.allocator, &jobs);
    defer testing.allocator.free(result.jobs);

    // Deadline 0 can never be met; the other two share the three clamped slots.
    try testing.expectEqual(@as(i64, 14), result.total_profit);
    try testing.expectEqualSlices(usize, &.{ 0, 1 }, result.jobs);
}

test "job sequencing - negative profit jobs are skipped" {
    const jobs = [_]Job{
        .{ .id = 0, .deadline = 1, .profit = -30 },
        .{ .id = 1, .deadline = 2, .profit = 10 },
        .{ .id = 2, .deadline = 3, .profit = 0 },
    };
    const result = try jobSequencing(testing.allocator, &jobs);
    defer testing.allocator.free(result.jobs);

    try testing.expectEqual(@as(i64, 10), result.total_profit);
    try testing.expectEqual(@as(usize, 2), result.jobs.len);
    try testing.expect(std.mem.findScalar(usize, result.jobs, 0) == null);
}

test "weighted job sequencing - zero-length jobs go first" {
    const jobs = [_]WeightedJob{
        .{ .id = 0, .processing_time = 1, .weight = 100 },
        .{ .id = 1, .processing_time = 0, .weight = 1 },
        .{ .id = 2, .processing_time = 0, .weight = 0 },
    };
    var sequence = try jobSequencingWeighted(testing.allocator, &jobs);
    defer sequence.deinit(testing.allocator);

    try testing.expectEqualSlices(usize, &.{ 1, 0, 2 }, sequence.items);
}

test "job sequencing - no leaks when an allocation fails" {
    const Runner = struct {
        fn run(gpa: Allocator) !void {
            const jobs = [_]Job{
                .{ .id = 0, .deadline = 2, .profit = 100 },
                .{ .id = 1, .deadline = 1, .profit = 50 },
                .{ .id = 2, .deadline = 2, .profit = 10 },
            };
            const result = try jobSequencing(gpa, &jobs);
            defer gpa.free(result.jobs);

            const weighted = [_]WeightedJob{
                .{ .id = 0, .processing_time = 2, .weight = 3 },
                .{ .id = 1, .processing_time = 1, .weight = 4 },
            };
            var sequence = try jobSequencingWeighted(gpa, &weighted);
            defer sequence.deinit(gpa);

            try testing.expectEqualSlices(usize, &.{ 1, 0 }, result.jobs);
            try testing.expectEqualSlices(usize, &.{ 1, 0 }, sequence.items);
        }
    };

    try testing.checkAllAllocationFailures(testing.allocator, Runner.run, .{});
}
