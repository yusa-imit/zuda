//! Benchmark suite for string algorithms
//!
//! Validates PRD performance targets:
//! - Aho-Corasick (1000 patterns, 1MB text): ≥ 500 MB/sec

const std = @import("std");
const zuda = @import("zuda");
const bench = zuda.internal.bench;

const AhoCorasick = zuda.algorithms.string.AhoCorasick;

/// Benchmark: Aho-Corasick with 1000 patterns on 1MB text
fn benchAhoCorasick(allocator: std.mem.Allocator) !void {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // Generate 1000 random patterns (5-15 bytes each)
    const pattern_count = 1000;
    var patterns = std.ArrayList([]const u8).init(allocator);
    defer {
        for (patterns.items) |pattern| {
            allocator.free(pattern);
        }
        patterns.deinit();
    }

    var i: usize = 0;
    while (i < pattern_count) : (i += 1) {
        const len = random.intRangeAtMost(usize, 5, 15);
        const pattern = try allocator.alloc(u8, len);
        for (pattern) |*byte| {
            byte.* = random.intRangeAtMost(u8, 'a', 'z');
        }
        try patterns.append(pattern);
    }

    // Build Aho-Corasick automaton
    var ac = try AhoCorasick.init(allocator, patterns.items);
    defer ac.deinit();

    // Generate 1MB random text
    const text_size = 1024 * 1024;
    const text = try allocator.alloc(u8, text_size);
    defer allocator.free(text);

    for (text) |*byte| {
        byte.* = random.intRangeAtMost(u8, 'a', 'z');
    }

    // Search for patterns in text
    var matches = std.ArrayList(AhoCorasick.Match).init(allocator);
    defer matches.deinit();

    try ac.search(text, &matches);
}

/// Run all string algorithm benchmarks and output markdown table
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n# String Algorithm Benchmarks\n\n", .{});
    std.debug.print("Validating PRD performance targets:\n", .{});
    std.debug.print("- Aho-Corasick: target ≥ 500 MB/sec (1000 patterns, 1MB text)\n\n", .{});

    // Aho-Corasick benchmark
    {
        std.debug.print("Running Aho-Corasick (1000 patterns, 1MB text)...\n", .{});

        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchAhoCorasick, .{allocator});

        const text_size_mb = 1; // 1MB
        const mb_per_sec = @divFloor(1_000_000_000, @divFloor(result.mean_ns, text_size_mb));

        std.debug.print("  Result: {d} MB/sec (mean over {d} iterations)\n", .{ mb_per_sec, result.iterations });

        if (mb_per_sec >= 500) {
            std.debug.print("  ✓ PASS: meets target of ≥ 500 MB/sec\n", .{});
        } else {
            std.debug.print("  ✗ FAIL: below target of ≥ 500 MB/sec\n", .{});
        }
    }

    std.debug.print("\n## Summary\n\n", .{});
    std.debug.print("Benchmark suite completed. See results above.\n", .{});
}
