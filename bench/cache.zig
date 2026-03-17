//! Benchmark suite for cache eviction policies
//!
//! Validates PRD performance targets:
//! - LRUCache: O(1) get/put with LRU eviction
//! - LFUCache: O(1) get/put with LFU eviction
//! - ARCCache: Adaptive replacement (L1+L2 split)

const std = @import("std");
const zuda = @import("zuda");
const bench = zuda.internal.bench;

const LRUCache = zuda.containers.cache.LRUCache;
const LFUCache = zuda.containers.cache.LFUCache;
const ARCCache = zuda.containers.cache.ARCCache;

/// Benchmark: LRUCache put operations
fn benchLRUCachePut(allocator: std.mem.Allocator) !void {
    var cache = LRUCache(i64, i64, std.hash_map.AutoContext(i64), null).init(allocator, 1000);
    defer cache.deinit();

    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        _ = try cache.put(@as(i64, @intCast(i)), @as(i64, @intCast(i)));
    }
}

/// Benchmark: LRUCache get operations
fn benchLRUCacheGet(allocator: std.mem.Allocator) !void {
    var cache = LRUCache(i64, i64, std.hash_map.AutoContext(i64), null).init(allocator, 10000);
    defer cache.deinit();

    // Pre-populate cache
    const count = 10_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        _ = try cache.put(@as(i64, @intCast(i)), @as(i64, @intCast(i)));
    }

    // Benchmark get (80% hits, 20% misses)
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const lookup_count = 100_000;
    i = 0;
    while (i < lookup_count) : (i += 1) {
        const key = if (random.int(u8) < 204) // 80%
            @as(i64, @intCast(random.uintLessThan(usize, count)))
        else
            @as(i64, @intCast(count + random.uintLessThan(usize, 1000)));
        _ = cache.get(key);
    }
}

/// Benchmark: LFUCache put operations
fn benchLFUCachePut(allocator: std.mem.Allocator) !void {
    var cache = LFUCache(i64, i64, std.hash_map.AutoContext(i64), null).init(allocator, 1000);
    defer cache.deinit();

    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        _ = try cache.put(@as(i64, @intCast(i)), @as(i64, @intCast(i)));
    }
}

/// Benchmark: LFUCache get operations
fn benchLFUCacheGet(allocator: std.mem.Allocator) !void {
    var cache = LFUCache(i64, i64, std.hash_map.AutoContext(i64), null).init(allocator, 10000);
    defer cache.deinit();

    // Pre-populate cache
    const count = 10_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        _ = try cache.put(@as(i64, @intCast(i)), @as(i64, @intCast(i)));
    }

    // Benchmark get (80% hits, 20% misses)
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const lookup_count = 100_000;
    i = 0;
    while (i < lookup_count) : (i += 1) {
        const key = if (random.int(u8) < 204) // 80%
            @as(i64, @intCast(random.uintLessThan(usize, count)))
        else
            @as(i64, @intCast(count + random.uintLessThan(usize, 1000)));
        _ = try cache.get(key);
    }
}

/// Benchmark: ARCCache put operations
fn benchARCCachePut(allocator: std.mem.Allocator) !void {
    var cache = ARCCache(i64, i64, std.hash_map.AutoContext(i64)).init(allocator, 1000);
    defer cache.deinit();

    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        _ = try cache.put(@as(i64, @intCast(i)), @as(i64, @intCast(i)));
    }
}

/// Benchmark: ARCCache get operations
fn benchARCCacheGet(allocator: std.mem.Allocator) !void {
    var cache = ARCCache(i64, i64, std.hash_map.AutoContext(i64)).init(allocator, 10000);
    defer cache.deinit();

    // Pre-populate cache
    const count = 10_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        _ = try cache.put(@as(i64, @intCast(i)), @as(i64, @intCast(i)));
    }

    // Benchmark get (80% hits, 20% misses)
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const lookup_count = 100_000;
    i = 0;
    while (i < lookup_count) : (i += 1) {
        const key = if (random.int(u8) < 204) // 80%
            @as(i64, @intCast(random.uintLessThan(usize, count)))
        else
            @as(i64, @intCast(count + random.uintLessThan(usize, 1000)));
        _ = cache.get(key);
    }
}

/// Run all cache benchmarks
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n# Cache Benchmarks\n\n", .{});
    std.debug.print("| Cache | Operation | ns/op | Status |\n", .{});
    std.debug.print("|-------|-----------|-------|--------|\n", .{});

    // LRUCache put
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchLRUCachePut, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 100) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| LRUCache | put | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // LRUCache get
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchLRUCacheGet, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 100) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| LRUCache | get | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // LFUCache put
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchLFUCachePut, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 150) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| LFUCache | put | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // LFUCache get
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchLFUCacheGet, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 150) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| LFUCache | get | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // ARCCache put
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchARCCachePut, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 200) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| ARCCache | put | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // ARCCache get
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchARCCacheGet, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 200) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| ARCCache | get | {d} | {s} |\n", .{ ns_per_op, status });
    }

    std.debug.print("\n## Summary\n\n", .{});
    std.debug.print("Cache benchmark suite completed.\n", .{});
    std.debug.print("- LRUCache: O(1) get/put with least-recently-used eviction\n", .{});
    std.debug.print("- LFUCache: O(1) get/put with least-frequently-used eviction\n", .{});
    std.debug.print("- ARCCache: Adaptive replacement with recency/frequency balance\n", .{});
}
