//! v2.0 Scientific Computing Benchmarks
//!
//! Comprehensive benchmark suite comparing zuda performance against targets from:
//! - OpenBLAS (BLAS operations)
//! - LAPACK (decompositions)
//! - FFTW (FFT operations)
//! - NumPy (descriptive statistics)
//!
//! Performance targets from docs/milestones.md - v2.0 Performance Targets
//!
//! Status: Comprehensive benchmark suite (5 categories)
//! Performance targets from docs/milestones.md - v2.0 Performance Targets

const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n# zuda v2.0 Scientific Computing Benchmarks\n", .{});
    std.debug.print("═══════════════════════════════════════════════\n\n", .{});

    // Category 1: BLAS Operations
    std.debug.print("## 1. BLAS Operations\n", .{});
    std.debug.print("───────────────────────────────────────────────\n", .{});

    // BLAS: dot product (1M elements)
    {
        const blas = zuda.linalg.blas;
        const n: usize = 1_000_000;
        var x = try zuda.ndarray.NDArray(f64, 1).ones(allocator, &.{n}, .row_major);
        defer x.deinit();
        var y = try zuda.ndarray.NDArray(f64, 1).ones(allocator, &.{n}, .row_major);
        defer y.deinit();

        var timer = try std.time.Timer.start();
        const result = try blas.dot(f64, x, y);
        const elapsed_ns = timer.read();
        const time_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const gflops = (2.0 * @as(f64, @floatFromInt(n))) / @as(f64, @floatFromInt(elapsed_ns));

        std.debug.print("  dot (1M f64):     {d:>8.2} ms  ({d:.2} GFLOPS, result={d:.1})\n", .{ time_ms, gflops, result });
    }

    // BLAS: GEMM 256×256 (Target: 3 GFLOPS)
    {
        const blas = zuda.linalg.blas;
        const n: usize = 256;
        var A = try zuda.ndarray.NDArray(f64, 2).ones(allocator, &.{n, n}, .row_major);
        defer A.deinit();
        var B = try zuda.ndarray.NDArray(f64, 2).ones(allocator, &.{n, n}, .row_major);
        defer B.deinit();
        var C = try zuda.ndarray.NDArray(f64, 2).zeros(allocator, &.{n, n}, .row_major);
        defer C.deinit();

        var timer = try std.time.Timer.start();
        try blas.gemm(f64, 1.0, A, B, 0.0, &C);
        const elapsed_ns = timer.read();
        const time_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const flops = 2.0 * @as(f64, @floatFromInt(n * n * n));
        const gflops = flops / @as(f64, @floatFromInt(elapsed_ns));

        std.debug.print("  GEMM 256×256:     {d:>8.2} ms  ({d:.2} GFLOPS)\n", .{ time_ms, gflops });
    }

    // BLAS: GEMM 1024×1024 (Target: 5 GFLOPS)
    {
        const blas = zuda.linalg.blas;
        const n: usize = 1024;
        var A = try zuda.ndarray.NDArray(f64, 2).ones(allocator, &.{n, n}, .row_major);
        defer A.deinit();
        var B = try zuda.ndarray.NDArray(f64, 2).ones(allocator, &.{n, n}, .row_major);
        defer B.deinit();
        var C = try zuda.ndarray.NDArray(f64, 2).zeros(allocator, &.{n, n}, .row_major);
        defer C.deinit();

        var timer = try std.time.Timer.start();
        try blas.gemm(f64, 1.0, A, B, 0.0, &C);
        const elapsed_ns = timer.read();
        const time_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const flops = 2.0 * @as(f64, @floatFromInt(n * n * n));
        const gflops = flops / @as(f64, @floatFromInt(elapsed_ns));

        std.debug.print("  GEMM 1024×1024:   {d:>8.2} ms  ({d:.2} GFLOPS)\n", .{ time_ms, gflops });
    }

    // Category 2: Linear Algebra Decompositions
    std.debug.print("\n## 2. Linear Algebra Decompositions\n", .{});
    std.debug.print("───────────────────────────────────────────────\n", .{});

    // LU decomposition 512×512 (Target: <200ms)
    {
        const lu = zuda.linalg.lu;
        const n: usize = 512;
        var A = try zuda.ndarray.NDArray(f64, 2).eye(allocator, n, n, 0, .row_major);
        defer A.deinit();
        // Make it non-trivial by adding some values
        for (0..n) |i| {
            for (0..n) |j| {
                const val = @as(f64, @floatFromInt((i + 1) * (j + 1))) / @as(f64, @floatFromInt(n));
                A.set(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) }, val);
            }
        }

        var timer = try std.time.Timer.start();
        var result = try lu.lu(f64, allocator, A);
        defer result.P.deinit();
        defer result.L.deinit();
        defer result.U.deinit();
        const elapsed_ns = timer.read();
        const time_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;

        std.debug.print("  LU 512×512:       {d:>8.2} ms\n", .{time_ms});
    }

    // QR decomposition 256×256 (Target: <500ms)
    {
        const decomp = zuda.linalg.decompositions;
        const m: usize = 256;
        const n: usize = 256;
        var A = try zuda.ndarray.NDArray(f64, 2).zeros(allocator, &.{m, n}, .row_major);
        defer A.deinit();
        for (0..m) |i| {
            for (0..n) |j| {
                const val = @as(f64, @floatFromInt((i + 1) * (j + 1))) / @as(f64, @floatFromInt(n));
                A.set(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) }, val);
            }
        }

        var timer = try std.time.Timer.start();
        var result = try decomp.qr(f64, A, allocator);
        defer result.Q.deinit();
        defer result.R.deinit();
        const elapsed_ns = timer.read();
        const time_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;

        std.debug.print("  QR 256×256:       {d:>8.2} ms\n", .{time_ms});
    }

    // SVD 128×128 (Target: <500ms)
    {
        const decomp = zuda.linalg.decompositions;
        const n: usize = 128;
        var A = try zuda.ndarray.NDArray(f64, 2).zeros(allocator, &.{n, n}, .row_major);
        defer A.deinit();
        for (0..n) |i| {
            for (0..n) |j| {
                const val = @as(f64, @floatFromInt((i + 1) * (j + 1))) / @as(f64, @floatFromInt(n));
                A.set(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) }, val);
            }
        }

        var timer = try std.time.Timer.start();
        var result = try decomp.svd(f64, A, allocator);
        defer result.U.deinit();
        defer result.S.deinit();
        defer result.Vt.deinit();
        const elapsed_ns = timer.read();
        const time_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;

        std.debug.print("  SVD 128×128:      {d:>8.2} ms\n", .{time_ms});
    }

    // Cholesky 512×512 (Target: <200ms)
    {
        const decomp = zuda.linalg.decompositions;
        const n: usize = 512;
        var A = try zuda.ndarray.NDArray(f64, 2).eye(allocator, n, n, 0, .row_major);
        defer A.deinit();
        // Make it positive definite by adding diagonal dominance
        for (0..n) |i| {
            const curr_val = try A.get(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(i)) });
            A.set(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(i)) }, curr_val + @as(f64, @floatFromInt(n)));
        }

        var timer = try std.time.Timer.start();
        var result = try decomp.cholesky(f64, A, allocator);
        defer result.deinit();
        const elapsed_ns = timer.read();
        const time_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;

        std.debug.print("  Cholesky 512×512: {d:>8.2} ms\n", .{time_ms});
    }

    // Category 3: FFT
    std.debug.print("\n## 3. FFT (Signal Processing)\n", .{});
    std.debug.print("───────────────────────────────────────────────\n", .{});

    // FFT 4096 complex (Target: <10μs)
    {
        const fft = zuda.signal.fft;
        const n: usize = 4096;
        var signal = try std.ArrayList(fft.Complex(f64)).initCapacity(allocator, n);
        defer signal.deinit(allocator);
        for (0..n) |i| {
            const t = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n));
            signal.appendAssumeCapacity(fft.Complex(f64).init(@sin(2.0 * std.math.pi * 10.0 * t), 0.0));
        }

        var timer = try std.time.Timer.start();
        const result = try fft.fft(f64, signal.items, allocator);
        defer allocator.free(result);
        const elapsed_ns = timer.read();
        const time_us = @as(f64, @floatFromInt(elapsed_ns)) / 1_000.0;

        std.debug.print("  FFT 4096:         {d:>8.2} μs\n", .{time_us});
    }

    // FFT 1M complex (Target: <30ms)
    {
        const fft = zuda.signal.fft;
        const n: usize = 1_048_576; // 2^20
        var signal = try std.ArrayList(fft.Complex(f64)).initCapacity(allocator, n);
        defer signal.deinit(allocator);
        for (0..n) |i| {
            const t = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n));
            signal.appendAssumeCapacity(fft.Complex(f64).init(@sin(2.0 * std.math.pi * 10.0 * t), 0.0));
        }

        var timer = try std.time.Timer.start();
        const result = try fft.fft(f64, signal.items, allocator);
        defer allocator.free(result);
        const elapsed_ns = timer.read();
        const time_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;

        std.debug.print("  FFT 1M:           {d:>8.2} ms\n", .{time_ms});
    }

    // Category 4: NDArray Operations
    std.debug.print("\n## 4. NDArray Operations\n", .{});
    std.debug.print("───────────────────────────────────────────────\n", .{});

    // Element-wise addition (1M elements, Target: 1 GFLOPS)
    {
        const n: usize = 1_000_000;
        var x = try zuda.ndarray.NDArray(f64, 1).ones(allocator, &.{n}, .row_major);
        defer x.deinit();
        var y = try zuda.ndarray.NDArray(f64, 1).ones(allocator, &.{n}, .row_major);
        defer y.deinit();

        var timer = try std.time.Timer.start();
        var result = try x.add(&y);
        defer result.deinit();
        const elapsed_ns = timer.read();
        const time_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const gflops = @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(elapsed_ns));

        std.debug.print("  add (1M):         {d:>8.2} ms  ({d:.2} GFLOPS)\n", .{ time_ms, gflops });
    }

    // Reduction sum (1M elements)
    {
        const n: usize = 1_000_000;
        var x = try zuda.ndarray.NDArray(f64, 1).ones(allocator, &.{n}, .row_major);
        defer x.deinit();

        var timer = try std.time.Timer.start();
        const result = x.sum();
        const elapsed_ns = timer.read();
        const time_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;

        std.debug.print("  sum (1M):         {d:>8.2} ms  (result={d:.1})\n", .{ time_ms, result });
    }

    // Matrix transpose (1024×1024)
    {
        const n: usize = 1024;
        var A = try zuda.ndarray.NDArray(f64, 2).ones(allocator, &.{n, n}, .row_major);
        defer A.deinit();

        var timer = try std.time.Timer.start();
        _ = A.transpose();
        const elapsed_ns = timer.read();
        const time_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;

        std.debug.print("  transpose 1024²:  {d:>8.2} ms  (view-only)\n", .{time_ms});
    }

    // Category 5: Statistics
    std.debug.print("\n## 5. Statistics\n", .{});
    std.debug.print("───────────────────────────────────────────────\n", .{});

    // Mean, variance, std (1M elements, Target: <1ms each)
    {
        const descriptive = zuda.stats.descriptive;
        const n: usize = 1_000_000;
        var data = try zuda.ndarray.NDArray(f64, 1).zeros(allocator, &.{n}, .row_major);
        defer data.deinit();
        for (0..n) |i| {
            data.set(&.{@as(isize, @intCast(i))}, @as(f64, @floatFromInt(i % 100)));
        }

        var timer = try std.time.Timer.start();
        const mean_val = descriptive.mean(f64, data);
        const elapsed_ns = timer.read();
        const time_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        std.debug.print("  mean (1M):        {d:>8.2} ms  (result={d:.2})\n", .{ time_ms, mean_val });

        timer.reset();
        const var_val = try descriptive.variance(f64, data, 0);
        const elapsed_ns2 = timer.read();
        const time_ms2 = @as(f64, @floatFromInt(elapsed_ns2)) / 1_000_000.0;
        std.debug.print("  variance (1M):    {d:>8.2} ms  (result={d:.2})\n", .{ time_ms2, var_val });

        timer.reset();
        const std_val = try descriptive.stdDev(f64, data, 0);
        const elapsed_ns3 = timer.read();
        const time_ms3 = @as(f64, @floatFromInt(elapsed_ns3)) / 1_000_000.0;
        std.debug.print("  stdDev (1M):      {d:>8.2} ms  (result={d:.2})\n", .{ time_ms3, std_val });
    }

    std.debug.print("\n═══════════════════════════════════════════════\n", .{});
    std.debug.print("Benchmark suite complete!\n", .{});
    std.debug.print("\nPerformance targets (docs/milestones.md):\n", .{});
    std.debug.print("  • BLAS: 5 GFLOPS (GEMM 1024), 3 GFLOPS (256), 2 GFLOPS (dot 1M)\n", .{});
    std.debug.print("  • Linalg: <200ms (LU/Cholesky 512), <500ms (QR 256, SVD 128)\n", .{});
    std.debug.print("  • FFT: <10μs (4096), <30ms (1M)\n", .{});
    std.debug.print("  • NDArray: 1 GFLOPS (element-wise ops)\n", .{});
    std.debug.print("  • Stats: <1ms (mean/var/std 1M)\n", .{});
    std.debug.print("═══════════════════════════════════════════════\n\n", .{});
}
