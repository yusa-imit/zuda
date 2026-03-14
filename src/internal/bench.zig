//! Micro-benchmark framework for zuda data structures
//!
//! Provides timer-based benchmarking with warmup iterations,
//! statistical analysis, and markdown output formatting.

const std = @import("std");

/// Benchmark configuration
pub const Config = struct {
    /// Number of warmup iterations before measurement
    warmup_iterations: usize = 5,
    /// Minimum number of iterations to run
    min_iterations: usize = 10,
    /// Maximum number of iterations to run
    max_iterations: usize = 1000,
    /// Target duration in nanoseconds for auto-iteration
    target_duration_ns: u64 = 100_000_000, // 100ms
};

/// Benchmark result statistics
pub const Result = struct {
    /// Total number of iterations performed
    iterations: usize,
    /// Total elapsed time in nanoseconds
    total_ns: u64,
    /// Minimum time per operation in nanoseconds
    min_ns: u64,
    /// Maximum time per operation in nanoseconds
    max_ns: u64,
    /// Mean time per operation in nanoseconds
    mean_ns: u64,
    /// Median time per operation in nanoseconds
    median_ns: u64,
    /// Standard deviation in nanoseconds
    std_dev_ns: u64,

    /// Format the result as a human-readable string
    pub fn format(
        self: Result,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.print(
            "{d} iterations | mean: {d}ns | median: {d}ns | min: {d}ns | max: {d}ns | σ: {d}ns",
            .{ self.iterations, self.mean_ns, self.median_ns, self.min_ns, self.max_ns, self.std_dev_ns },
        );
    }

    /// Format as ops/sec
    pub fn formatOpsPerSec(self: Result, writer: anytype) !void {
        const ops_per_sec = if (self.mean_ns > 0)
            @divFloor(1_000_000_000, self.mean_ns)
        else
            0;
        try writer.print("{d} ops/sec", .{ops_per_sec});
    }

    /// Format as markdown table row
    pub fn formatMarkdownRow(self: Result, writer: anytype, name: []const u8) !void {
        const ops_per_sec = if (self.mean_ns > 0)
            @divFloor(1_000_000_000, self.mean_ns)
        else
            0;

        try writer.print(
            "| {s} | {d} | {d} | {d} | {d} | {d} |\n",
            .{ name, self.mean_ns, self.median_ns, self.min_ns, self.max_ns, ops_per_sec },
        );
    }
};

/// Timer for high-precision benchmarking
const Timer = std.time.Timer;

/// Benchmark context
pub const Benchmark = struct {
    config: Config,
    timer: Timer,
    times: std.ArrayList(u64),
    allocator: std.mem.Allocator,

    /// Initialize a new benchmark with the given configuration
    pub fn init(allocator: std.mem.Allocator, config: Config) !Benchmark {
        return Benchmark{
            .config = config,
            .timer = try Timer.start(),
            .times = std.ArrayList(u64){},
            .allocator = allocator,
        };
    }

    /// Clean up benchmark resources
    pub fn deinit(self: *Benchmark) void {
        self.times.deinit(self.allocator);
    }

    /// Run warmup iterations without measurement
    pub fn warmup(self: *Benchmark, comptime func: anytype, args: anytype) !void {
        var i: usize = 0;
        while (i < self.config.warmup_iterations) : (i += 1) {
            const ReturnType = @typeInfo(@TypeOf(func)).@"fn".return_type.?;
            if (@typeInfo(ReturnType) == .error_union) {
                _ = try @call(.auto, func, args);
            } else {
                _ = @call(.auto, func, args);
            }
        }
    }

    /// Run a single measured iteration
    pub fn runOnce(self: *Benchmark, comptime func: anytype, args: anytype) !void {
        self.timer.reset();
        const ReturnType = @typeInfo(@TypeOf(func)).@"fn".return_type.?;
        if (@typeInfo(ReturnType) == .error_union) {
            _ = try @call(.auto, func, args);
        } else {
            _ = @call(.auto, func, args);
        }
        const elapsed = self.timer.read();
        try self.times.append(self.allocator, elapsed);
    }

    /// Run benchmark with auto-iteration count
    pub fn run(self: *Benchmark, comptime func: anytype, args: anytype) !Result {
        // Warmup
        try self.warmup(func, args);

        // Determine iteration count
        self.timer.reset();
        const ReturnType = @typeInfo(@TypeOf(func)).@"fn".return_type.?;
        if (@typeInfo(ReturnType) == .error_union) {
            _ = try @call(.auto, func, args);
        } else {
            _ = @call(.auto, func, args);
        }
        const single_iteration_ns = self.timer.read();

        const iterations = blk: {
            if (single_iteration_ns == 0) break :blk self.config.max_iterations;

            const calculated = @divFloor(self.config.target_duration_ns, single_iteration_ns);
            if (calculated < self.config.min_iterations) break :blk self.config.min_iterations;
            if (calculated > self.config.max_iterations) break :blk self.config.max_iterations;
            break :blk calculated;
        };

        // Run measured iterations
        var i: usize = 0;
        while (i < iterations) : (i += 1) {
            try self.runOnce(func, args);
        }

        return self.computeResult();
    }

    /// Run benchmark with fixed iteration count
    pub fn runIterations(self: *Benchmark, comptime func: anytype, args: anytype, iterations: usize) !Result {
        // Warmup
        try self.warmup(func, args);

        // Run measured iterations
        var i: usize = 0;
        while (i < iterations) : (i += 1) {
            try self.runOnce(func, args);
        }

        return self.computeResult();
    }

    /// Compute statistics from collected timings
    fn computeResult(self: *Benchmark) Result {
        const times = self.times.items;
        if (times.len == 0) {
            return Result{
                .iterations = 0,
                .total_ns = 0,
                .min_ns = 0,
                .max_ns = 0,
                .mean_ns = 0,
                .median_ns = 0,
                .std_dev_ns = 0,
            };
        }

        // Sort for median calculation (modifies the array)
        std.mem.sort(u64, self.times.items, {}, comptime std.sort.asc(u64));

        var total: u64 = 0;
        var min: u64 = std.math.maxInt(u64);
        var max: u64 = 0;

        for (times) |time| {
            total += time;
            if (time < min) min = time;
            if (time > max) max = time;
        }

        const mean = @divFloor(total, times.len);
        const median = times[@divFloor(times.len, 2)];

        // Calculate standard deviation
        var variance_sum: u128 = 0;
        for (times) |time| {
            const diff: i128 = @as(i128, @intCast(time)) - @as(i128, @intCast(mean));
            variance_sum += @as(u128, @intCast(diff * diff));
        }
        const variance = @divFloor(variance_sum, times.len);
        const std_dev = @as(u64, @intCast(std.math.sqrt(variance)));

        return Result{
            .iterations = times.len,
            .total_ns = total,
            .min_ns = min,
            .max_ns = max,
            .mean_ns = mean,
            .median_ns = median,
            .std_dev_ns = std_dev,
        };
    }
};

/// Simple benchmark runner that prints results
pub fn benchmark(
    allocator: std.mem.Allocator,
    name: []const u8,
    comptime func: anytype,
    args: anytype,
) !void {
    const stdout_file = std.io.getStdOut();
    const stdout = stdout_file.writer();

    var bench = try Benchmark.init(allocator, .{});
    defer bench.deinit();

    const result = try bench.run(func, args);

    try stdout.print("{s}: {}\n", .{ name, result });
}

/// Compare two benchmark results and print comparison
pub fn compare(name_a: []const u8, result_a: Result, name_b: []const u8, result_b: Result, writer: anytype) !void {
    const speedup = if (result_b.mean_ns > 0)
        @as(f64, @floatFromInt(result_b.mean_ns)) / @as(f64, @floatFromInt(result_a.mean_ns))
    else
        0.0;

    try writer.print("\n=== Comparison: {s} vs {s} ===\n", .{ name_a, name_b });
    try writer.print("{s}: mean={d}ns median={d}ns\n", .{ name_a, result_a.mean_ns, result_a.median_ns });
    try writer.print("{s}: mean={d}ns median={d}ns\n", .{ name_b, result_b.mean_ns, result_b.median_ns });
    try writer.print("Speedup: {d:.2}x\n", .{speedup});
}

/// Markdown table formatter for benchmark results
pub const MarkdownTable = struct {
    writer: std.io.AnyWriter,

    pub fn init(writer: std.io.AnyWriter) MarkdownTable {
        return .{ .writer = writer };
    }

    pub fn writeHeader(self: MarkdownTable) !void {
        try self.writer.writeAll("| Benchmark | Mean (ns) | Median (ns) | Min (ns) | Max (ns) | Ops/sec |\n");
        try self.writer.writeAll("|-----------|-----------|-------------|----------|----------|---------|\n");
    }

    pub fn writeRow(self: MarkdownTable, name: []const u8, result: Result) !void {
        try result.formatMarkdownRow(self.writer, name);
    }
};

// Simple benchmark test
fn addNumbers(a: i64, b: i64) i64 {
    return a + b;
}

test "benchmark basic operation" {
    var bench = try Benchmark.init(std.testing.allocator, .{
        .min_iterations = 100,
        .max_iterations = 1000,
    });
    defer bench.deinit();

    const result = try bench.run(addNumbers, .{ @as(i64, 42), @as(i64, 58) });

    try std.testing.expect(result.iterations >= 100);
    try std.testing.expect(result.mean_ns > 0);
}

test "benchmark with fixed iterations" {
    var bench = try Benchmark.init(std.testing.allocator, .{});
    defer bench.deinit();

    const result = try bench.runIterations(addNumbers, .{ @as(i64, 42), @as(i64, 58) }, 50);

    try std.testing.expectEqual(50, result.iterations);
}
