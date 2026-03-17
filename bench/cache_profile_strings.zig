const std = @import("std");
const zuda = @import("zuda");

/// Cache profiling benchmark for Aho-Corasick string search variants.
/// Measures performance under realistic workload to identify cache behavior.
///
/// Build: zig build bench-cache-profile
/// Run with perf: perf stat -e cache-misses,cache-references,L1-dcache-load-misses ./zig-out/bin/bench-cache-profile

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Aho-Corasick Cache Profile ===\n\n", .{});

    // Generate realistic workload
    const patterns = try generatePatterns(allocator, 1000, 10);
    defer {
        for (patterns) |p| allocator.free(p);
        allocator.free(patterns);
    }

    const text = try generateText(allocator, 1024 * 1024); // 1 MB
    defer allocator.free(text);

    // Benchmark DoubleArrayTrie
    try benchDoubleArrayTrie(allocator, patterns, text);

    std.debug.print("\nDone. Run with:\n", .{});
    std.debug.print("  perf stat -e cache-misses,cache-references,L1-dcache-load-misses,L1-dcache-loads ./zig-out/bin/bench-cache-profile\n", .{});
}

fn benchDoubleArrayTrie(allocator: std.mem.Allocator, patterns: []const []const u8, text: []const u8) !void {
    const DoubleArrayTrie = zuda.containers.strings.DoubleArrayTrie(u8);

    // Build automaton
    var dat = try DoubleArrayTrie.init(allocator, patterns);
    defer dat.deinit();

    // Calculate memory usage: BaseCheck (8 bytes) + is_leaf (1) + fail (4) + output pointer overhead (~24)
    const mem_kb = @divFloor(dat.base_check.len * (@sizeOf(DoubleArrayTrie.BaseCheck) + @sizeOf(bool) + @sizeOf(u32) + 24), 1024);
    std.debug.print("DoubleArrayTrie: {} states, {} KB memory\n", .{
        dat.count(),
        mem_kb,
    });

    // Warmup
    for (0..10) |_| {
        const matches = try dat.findAll(allocator, text);
        allocator.free(matches);
    }

    // Benchmark search phase (hot loop)
    const iterations: usize = 100;
    var timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        const matches = try dat.findAll(allocator, text);
        allocator.free(matches);
    }
    const elapsed_ns = timer.read();

    const bytes_processed = text.len * iterations;
    const throughput_mb_sec = @as(f64, @floatFromInt(bytes_processed)) / (@as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0) / (1024.0 * 1024.0);

    std.debug.print("  Throughput: {d:.1} MB/sec\n", .{throughput_mb_sec});
    std.debug.print("  Total processed: {} MB\n", .{@divFloor(bytes_processed, 1024 * 1024)});
}

fn generatePatterns(allocator: std.mem.Allocator, count: usize, avg_len: usize) ![][]const u8 {
    const patterns = try allocator.alloc([]const u8, count);
    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    for (patterns) |*p| {
        const len = avg_len + random.intRangeAtMost(usize, 0, 5);
        const pattern = try allocator.alloc(u8, len);
        for (pattern) |*c| {
            c.* = 'a' + @as(u8, @intCast(random.intRangeAtMost(usize, 0, 25)));
        }
        p.* = pattern;
    }

    return patterns;
}

fn generateText(allocator: std.mem.Allocator, size: usize) ![]u8 {
    const text = try allocator.alloc(u8, size);
    var prng = std.Random.DefaultPrng.init(67890);
    const random = prng.random();

    for (text) |*c| {
        c.* = 'a' + @as(u8, @intCast(random.intRangeAtMost(usize, 0, 25)));
    }

    return text;
}
