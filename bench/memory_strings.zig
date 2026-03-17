//! Memory profiling benchmark for string data structures
//!
//! Validates v1.8.0 memory reduction claim:
//! - DoubleArrayTrie: 50-100× reduction vs Generic Aho-Corasick

const std = @import("std");
const zuda = @import("zuda");
const bench = zuda.internal.bench;

const AhoCorasick = zuda.algorithms.string.AhoCorasick;
const AhoCorasickASCII = zuda.algorithms.string.AhoCorasickASCII;
const DoubleArrayTrie = zuda.containers.strings.DoubleArrayTrie;

/// Run memory profiling with MemoryTracker
fn profileAhoCorasickGeneric(allocator: std.mem.Allocator, patterns: []const []const u8) !bench.MemoryStats {
    var tracker = bench.MemoryTracker.init(allocator);
    const tracked_allocator = tracker.allocator();

    // Build automaton with tracking
    var ac = try AhoCorasick(u8).init(tracked_allocator, patterns);
    defer ac.deinit();

    return .{
        .peak_bytes = tracker.peak_bytes,
        .total_allocated = tracker.total_allocated,
        .total_freed = tracker.total_freed,
        .allocation_count = tracker.allocation_count,
        .free_count = tracker.free_count,
        .current_bytes = tracker.current_bytes,
    };
}

/// Run memory profiling for ASCII Aho-Corasick
fn profileAhoCorasickASCII(allocator: std.mem.Allocator, patterns: []const []const u8) !bench.MemoryStats {
    var tracker = bench.MemoryTracker.init(allocator);
    const tracked_allocator = tracker.allocator();

    // Build automaton with tracking
    var ac = try AhoCorasickASCII.init(tracked_allocator, patterns);
    defer ac.deinit();

    return .{
        .peak_bytes = tracker.peak_bytes,
        .total_allocated = tracker.total_allocated,
        .total_freed = tracker.total_freed,
        .allocation_count = tracker.allocation_count,
        .free_count = tracker.free_count,
        .current_bytes = tracker.current_bytes,
    };
}

/// Run memory profiling for DoubleArrayTrie
fn profileDoubleArrayTrie(allocator: std.mem.Allocator, patterns: []const []const u8) !bench.MemoryStats {
    var tracker = bench.MemoryTracker.init(allocator);
    const tracked_allocator = tracker.allocator();

    // Build double-array trie with tracking
    var trie = try DoubleArrayTrie(u8).init(tracked_allocator, patterns);
    defer trie.deinit();

    return .{
        .peak_bytes = tracker.peak_bytes,
        .total_allocated = tracker.total_allocated,
        .total_freed = tracker.total_freed,
        .allocation_count = tracker.allocation_count,
        .free_count = tracker.free_count,
        .current_bytes = tracker.current_bytes,
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n# Memory Profiling: String Data Structures\n\n", .{});
    std.debug.print("Validating v1.8.0 memory reduction claim:\n", .{});
    std.debug.print("- DoubleArrayTrie: 50-100× reduction vs Generic Aho-Corasick\n\n", .{});

    // Generate test patterns (1000 random patterns, 5-15 bytes each)
    std.debug.print("Generating 1000 random patterns (5-15 bytes each)...\n", .{});
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const pattern_count = 1000;
    var patterns: std.ArrayList([]const u8) = .{};
    defer {
        for (patterns.items) |pattern| {
            allocator.free(pattern);
        }
        patterns.deinit(allocator);
    }

    var i: usize = 0;
    while (i < pattern_count) : (i += 1) {
        const len = random.intRangeAtMost(usize, 5, 15);
        const pattern = try allocator.alloc(u8, len);
        for (pattern) |*byte| {
            byte.* = random.intRangeAtMost(u8, 'a', 'z');
        }
        try patterns.append(allocator, pattern);
    }

    std.debug.print("Patterns generated: {d} items\n\n", .{patterns.items.len});

    // Profile Generic Aho-Corasick
    std.debug.print("Profiling AhoCorasick (Generic, HashMap transitions)...\n", .{});
    const generic_stats = try profileAhoCorasickGeneric(allocator, patterns.items);
    std.debug.print("  Peak memory: {d} bytes ({d} KB)\n", .{ generic_stats.peak_bytes, @divFloor(generic_stats.peak_bytes, 1024) });
    std.debug.print("  Allocations: {d} | Frees: {d}\n", .{ generic_stats.allocation_count, generic_stats.free_count });
    std.debug.print("  Total allocated: {d} bytes | Total freed: {d} bytes\n", .{ generic_stats.total_allocated, generic_stats.total_freed });

    // Profile ASCII Aho-Corasick
    std.debug.print("\nProfiling AhoCorasickASCII (dense array transitions)...\n", .{});
    const ascii_stats = try profileAhoCorasickASCII(allocator, patterns.items);
    std.debug.print("  Peak memory: {d} bytes ({d} KB)\n", .{ ascii_stats.peak_bytes, @divFloor(ascii_stats.peak_bytes, 1024) });
    std.debug.print("  Allocations: {d} | Frees: {d}\n", .{ ascii_stats.allocation_count, ascii_stats.free_count });
    std.debug.print("  Total allocated: {d} bytes | Total freed: {d} bytes\n", .{ ascii_stats.total_allocated, ascii_stats.total_freed });

    // Profile DoubleArrayTrie
    std.debug.print("\nProfiling DoubleArrayTrie (double-array with FAIL/OUTPUT)...\n", .{});
    const dat_stats = try profileDoubleArrayTrie(allocator, patterns.items);
    std.debug.print("  Peak memory: {d} bytes ({d} KB)\n", .{ dat_stats.peak_bytes, @divFloor(dat_stats.peak_bytes, 1024) });
    std.debug.print("  Allocations: {d} | Frees: {d}\n", .{ dat_stats.allocation_count, dat_stats.free_count });
    std.debug.print("  Total allocated: {d} bytes | Total freed: {d} bytes\n", .{ dat_stats.total_allocated, dat_stats.total_freed });

    // Calculate memory reduction
    std.debug.print("\n## Memory Comparison\n\n", .{});

    const reduction_vs_generic = @divFloor(generic_stats.peak_bytes, dat_stats.peak_bytes);
    const reduction_pct_generic = @divFloor((generic_stats.peak_bytes - dat_stats.peak_bytes) * 100, generic_stats.peak_bytes);

    const reduction_vs_ascii = @divFloor(ascii_stats.peak_bytes, dat_stats.peak_bytes);
    const reduction_pct_ascii = @divFloor((ascii_stats.peak_bytes - dat_stats.peak_bytes) * 100, ascii_stats.peak_bytes);

    std.debug.print("DoubleArrayTrie vs Generic (HashMap):\n", .{});
    std.debug.print("  Memory reduction: {d}× ({d}% less memory)\n", .{ reduction_vs_generic, reduction_pct_generic });
    std.debug.print("  {d} KB → {d} KB\n", .{ @divFloor(generic_stats.peak_bytes, 1024), @divFloor(dat_stats.peak_bytes, 1024) });

    std.debug.print("\nDoubleArrayTrie vs ASCII (dense array):\n", .{});
    std.debug.print("  Memory reduction: {d}× ({d}% less memory)\n", .{ reduction_vs_ascii, reduction_pct_ascii });
    std.debug.print("  {d} KB → {d} KB\n", .{ @divFloor(ascii_stats.peak_bytes, 1024), @divFloor(dat_stats.peak_bytes, 1024) });

    std.debug.print("\n## Validation\n\n", .{});
    const target_reduction_min = 50;
    const target_reduction_max = 100;

    if (reduction_vs_generic >= target_reduction_min) {
        std.debug.print("✓ PASS: DoubleArrayTrie achieves {d}× reduction vs Generic (target: {d}-{d}×)\n", .{ reduction_vs_generic, target_reduction_min, target_reduction_max });
    } else {
        std.debug.print("✗ FAIL: DoubleArrayTrie achieves {d}× reduction vs Generic (target: {d}-{d}×)\n", .{ reduction_vs_generic, target_reduction_min, target_reduction_max });
    }

    std.debug.print("\n## Summary\n\n", .{});
    std.debug.print("| Implementation | Peak Memory | Reduction vs Generic | Reduction vs ASCII |\n", .{});
    std.debug.print("|----------------|-------------|---------------------|-------------------|\n", .{});
    std.debug.print("| Generic (HashMap) | {d} KB | 1× (baseline) | — |\n", .{@divFloor(generic_stats.peak_bytes, 1024)});
    std.debug.print("| ASCII (dense array) | {d} KB | {d}× | 1× (baseline) |\n", .{ @divFloor(ascii_stats.peak_bytes, 1024), @divFloor(generic_stats.peak_bytes, ascii_stats.peak_bytes) });
    std.debug.print("| DoubleArrayTrie | {d} KB | **{d}×** | **{d}×** |\n", .{ @divFloor(dat_stats.peak_bytes, 1024), reduction_vs_generic, reduction_vs_ascii });
}
