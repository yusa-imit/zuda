//! Benchmark Methodology Example
//!
//! Demonstrates proper benchmarking technique accounting for CPU frequency scaling.
//! Session 526 discovered that cold-run benchmarks can show 40-50% lower performance
//! due to CPU power management ramping up from idle state.
//!
//! Key Insights:
//! - First run (cold): CPU at lower frequency → artificially slow results
//! - Warmup runs (2-5): CPU ramps to full frequency → stable performance
//! - Best practice: Run 2+ iterations, report min/avg of runs 2+
//!
//! Example from session 526:
//! - dot product (1M f64): First run 1.49 GFLOPS → After warmup 2.63 GFLOPS (+76%)
//! - GEMM 256²: First run 2.96 GFLOPS → After warmup 4.54 GFLOPS (+53%)
//!
//! This example shows how to implement warmup-aware benchmarking.

const std = @import("std");
const zuda = @import("zuda");

const BenchResult = struct {
    cold_run_ns: u64,
    warmup_runs_ns: []u64,
    min_warmup_ns: u64,
    avg_warmup_ns: u64,
    speedup: f64,

    fn deinit(self: *BenchResult, allocator: std.mem.Allocator) void {
        allocator.free(self.warmup_runs_ns);
    }
};

/// Run a benchmark with warmup iterations
fn benchmarkWithWarmup(
    comptime name: []const u8,
    comptime func: anytype,
    args: anytype,
    warmup_count: usize,
    allocator: std.mem.Allocator,
) !BenchResult {
    var times = try allocator.alloc(u64, warmup_count);
    errdefer allocator.free(times);

    // Cold run (CPU at lower frequency)
    var timer = try std.time.Timer.start();
    _ = @call(.auto, func, args);
    const cold_ns = timer.read();

    // Warmup runs (CPU ramps to full frequency)
    var min_ns: u64 = std.math.maxInt(u64);
    var sum_ns: u64 = 0;

    for (0..warmup_count) |i| {
        timer.reset();
        _ = @call(.auto, func, args);
        const elapsed = timer.read();
        times[i] = elapsed;

        min_ns = @min(min_ns, elapsed);
        sum_ns += elapsed;
    }

    const avg_ns = sum_ns / warmup_count;
    const speedup = @as(f64, @floatFromInt(cold_ns)) / @as(f64, @floatFromInt(min_ns));

    std.debug.print("{s}:\n", .{name});
    std.debug.print("  Cold run: {d:.2}ms\n", .{@as(f64, @floatFromInt(cold_ns)) / 1_000_000.0});
    std.debug.print("  Min warmup: {d:.2}ms (speedup: {d:.2}×)\n", .{
        @as(f64, @floatFromInt(min_ns)) / 1_000_000.0,
        speedup,
    });
    std.debug.print("  Avg warmup: {d:.2}ms\n\n", .{@as(f64, @floatFromInt(avg_ns)) / 1_000_000.0});

    return BenchResult{
        .cold_run_ns = cold_ns,
        .warmup_runs_ns = times,
        .min_warmup_ns = min_ns,
        .avg_warmup_ns = avg_ns,
        .speedup = speedup,
    };
}

/// Example computation: dot product (simple version without BLAS)
fn dotProduct(x: []const f64, y: []const f64) f64 {
    var sum: f64 = 0.0;
    for (x, y) |xi, yi| {
        sum += xi * yi;
    }
    return sum;
}

/// Example computation: matrix multiplication (simple version)
fn matrixMultiply(A: []const f64, B: []const f64, C: []f64, n: usize) void {
    // C = A * B (all n×n matrices in row-major order)
    @memset(C, 0.0);
    for (0..n) |i| {
        for (0..n) |k| {
            const a_ik = A[i * n + k];
            for (0..n) |j| {
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== CPU Frequency Scaling Benchmark ===\n\n", .{});
    std.debug.print("This example demonstrates the impact of CPU power management\n", .{});
    std.debug.print("on benchmark results. Modern CPUs scale frequency from idle\n", .{});
    std.debug.print("to full performance, causing 40-80% variance in cold runs.\n\n", .{});

    // Example 1: Dot product (1M elements)
    {
        const n = 1_000_000;
        const x = try allocator.alloc(f64, n);
        defer allocator.free(x);
        const y = try allocator.alloc(f64, n);
        defer allocator.free(y);

        // Initialize with non-trivial values
        for (x, 0..) |*val, i| {
            val.* = @as(f64, @floatFromInt(i % 100)) + 1.0;
        }
        for (y, 0..) |*val, i| {
            val.* = @as(f64, @floatFromInt((i * 3) % 100)) + 1.0;
        }

        var result = try benchmarkWithWarmup(
            "Dot Product (1M f64)",
            dotProduct,
            .{ x, y },
            5,
            allocator,
        );
        defer result.deinit(allocator);
    }

    // Example 2: Matrix multiply (256×256, similar to GEMM benchmark)
    {
        const n = 256;
        const size = n * n;

        const A = try allocator.alloc(f64, size);
        defer allocator.free(A);
        const B = try allocator.alloc(f64, size);
        defer allocator.free(B);
        const C = try allocator.alloc(f64, size);
        defer allocator.free(C);

        // Initialize with identity-like pattern
        for (0..n) |i| {
            for (0..n) |j| {
                A[i * n + j] = if (i == j) 1.0 else 0.1;
                B[i * n + j] = @as(f64, @floatFromInt(j + 1));
            }
        }

        var result = try benchmarkWithWarmup(
            "Matrix Multiply (256×256)",
            matrixMultiply,
            .{ A, B, C, n },
            5,
            allocator,
        );
        defer result.deinit(allocator);
    }

    std.debug.print("=== Key Takeaways ===\n\n", .{});
    std.debug.print("1. Always run warmup iterations before measuring performance\n", .{});
    std.debug.print("2. Report min or avg of warmup runs, NOT the cold run\n", .{});
    std.debug.print("3. Typical speedup after warmup: 1.3-2.0× for compute-intensive tasks\n", .{});
    std.debug.print("4. Apple M2/M3: aggressive frequency scaling → higher variance\n", .{});
    std.debug.print("5. Intel/AMD: less aggressive scaling → lower variance\n", .{});
    std.debug.print("\nSee docs/BENCHMARKS.md for zuda's official benchmark results.\n", .{});
}
