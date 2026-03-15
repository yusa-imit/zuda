//! Benchmark suite for sorting algorithms
//!
//! Validates PRD performance targets:
//! - TimSort (1M i64, random): ≤ 10% overhead vs std.sort

const std = @import("std");
const zuda = @import("zuda");
const bench = zuda.internal.bench;

const TimSort = zuda.algorithms.sorting.TimSort;

/// Benchmark: std.sort.block for baseline
fn benchStdSort(allocator: std.mem.Allocator) !void {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const count = 1_000_000;
    const data = try allocator.alloc(i64, count);
    defer allocator.free(data);

    for (data) |*item| {
        item.* = random.int(i64);
    }

    std.mem.sort(i64, data, {}, std.sort.asc(i64));
}

/// Benchmark: TimSort with 1M random i64
fn benchTimSort(allocator: std.mem.Allocator) !void {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const count = 1_000_000;
    const data = try allocator.alloc(i64, count);
    defer allocator.free(data);

    for (data) |*item| {
        item.* = random.int(i64);
    }

    try TimSort(i64, void, struct { fn cmp(_: void, a: i64, b: i64) std.math.Order { return std.math.order(a, b); } }.cmp).sort(allocator, data, {});
}

/// Run all sorting benchmarks and output markdown table
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n# Sorting Algorithm Benchmarks\n\n", .{});
    std.debug.print("Validating PRD performance targets:\n", .{});
    std.debug.print("- TimSort: target ≤ 10%% overhead vs std.sort (1M random i64)\n\n", .{});

    var std_sort_ns: u64 = 0;
    var timsort_ns: u64 = 0;

    // std.sort baseline
    {
        std.debug.print("Running std.sort (1M random i64)...\n", .{});

        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchStdSort, .{allocator});
        std_sort_ns = result.mean_ns;

        std.debug.print("  Result: {d} ms (mean over {d} iterations)\n", .{ @divFloor(std_sort_ns, 1_000_000), result.iterations });
    }

    std.debug.print("\n", .{});

    // TimSort
    {
        std.debug.print("Running TimSort (1M random i64)...\n", .{});

        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchTimSort, .{allocator});
        timsort_ns = result.mean_ns;

        std.debug.print("  Result: {d} ms (mean over {d} iterations)\n", .{ @divFloor(timsort_ns, 1_000_000), result.iterations });
    }

    std.debug.print("\n## Comparison\n\n", .{});

    // Calculate relative performance (can be negative if TimSort is faster)
    const diff: i64 = @as(i64, @intCast(timsort_ns)) - @as(i64, @intCast(std_sort_ns));
    const overhead_percent = @divFloor(diff * 100, @as(i64, @intCast(std_sort_ns)));

    if (overhead_percent < 0) {
        std.debug.print("TimSort speedup: {d}% faster than std.sort\n", .{-overhead_percent});
        std.debug.print("✓ PASS: faster than std.sort (exceeds target!)\n", .{});
    } else if (overhead_percent <= 10) {
        std.debug.print("TimSort overhead: {d}%\n", .{overhead_percent});
        std.debug.print("✓ PASS: meets target of ≤ 10%% overhead\n", .{});
    } else {
        std.debug.print("TimSort overhead: {d}%\n", .{overhead_percent});
        std.debug.print("✗ FAIL: exceeds target of ≤ 10%% overhead\n", .{});
    }

    std.debug.print("\n## Summary\n\n", .{});
    std.debug.print("Benchmark suite completed. See results above.\n", .{});
}
