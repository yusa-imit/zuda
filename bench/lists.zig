//! Benchmark suite for list-based containers
//!
//! Validates PRD performance targets for Phase 1 lists:
//! - SkipList: O(log n) insert/search
//! - XorLinkedList: O(1) push/pop
//! - UnrolledLinkedList: Better cache locality than standard linked list
//! - ConcurrentSkipList: Lock-free concurrent access

const std = @import("std");
const zuda = @import("zuda");
const bench = zuda.internal.bench;

const SkipList = zuda.containers.lists.SkipList;
const XorLinkedList = zuda.containers.lists.XorLinkedList;
const UnrolledLinkedList = zuda.containers.lists.UnrolledLinkedList;
const ConcurrentSkipList = zuda.containers.lists.ConcurrentSkipList;

/// Context for integer comparisons
const IntContext = struct {
    pub fn compare(_: IntContext, a: i64, b: i64) std.math.Order {
        return std.math.order(a, b);
    }
};

/// Benchmark: SkipList insert with 100k random keys
fn benchSkipListInsert(allocator: std.mem.Allocator) !void {
    var list = try SkipList(i64, i64, IntContext, IntContext.compare).init(allocator, .{});
    defer list.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const value = random.int(i64);
        _ = try list.insert(value, value);
    }
}

/// Benchmark: SkipList search with 100k keys
fn benchSkipListSearch(allocator: std.mem.Allocator) !void {
    var list = try SkipList(i64, i64, IntContext, IntContext.compare).init(allocator, .{});
    defer list.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // Pre-populate list
    const count = 100_000;
    const keys = try allocator.alloc(i64, count);
    defer allocator.free(keys);

    var i: usize = 0;
    while (i < count) : (i += 1) {
        keys[i] = random.int(i64);
        _ = try list.insert(keys[i], keys[i]);
    }

    // Shuffle for random access
    random.shuffle(i64, @constCast(keys));

    // Benchmark search
    i = 0;
    while (i < count) : (i += 1) {
        _ = list.get(keys[i]);
    }
}

/// Benchmark: XorLinkedList push operations
fn benchXorLinkedListPush(allocator: std.mem.Allocator) !void {
    var list = XorLinkedList(i64).init(allocator);
    defer list.deinit();

    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        try list.pushBack(@as(i64, @intCast(i)));
    }
}

/// Benchmark: XorLinkedList iteration
fn benchXorLinkedListIterate(allocator: std.mem.Allocator) !void {
    var list = XorLinkedList(i64).init(allocator);
    defer list.deinit();

    // Pre-populate list
    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        try list.pushBack(@as(i64, @intCast(i)));
    }

    // Benchmark iteration
    var iter = list.iterator();
    while (iter.next()) |_| {}
}

/// Benchmark: UnrolledLinkedList append
fn benchUnrolledLinkedListAppend(allocator: std.mem.Allocator) !void {
    var list = UnrolledLinkedList(i64, 64).init(allocator);
    defer list.deinit();

    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        try list.append(@as(i64, @intCast(i)));
    }
}

/// Benchmark: UnrolledLinkedList iteration (cache-friendly)
fn benchUnrolledLinkedListIterate(allocator: std.mem.Allocator) !void {
    var list = UnrolledLinkedList(i64, 64).init(allocator);
    defer list.deinit();

    // Pre-populate list
    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        try list.append(@as(i64, @intCast(i)));
    }

    // Benchmark iteration
    var iter = list.iterator();
    while (iter.next()) |_| {}
}

/// Benchmark: ConcurrentSkipList insert
fn benchConcurrentSkipListInsert(allocator: std.mem.Allocator) !void {
    var list = try ConcurrentSkipList(i64, i64, IntContext, IntContext.compare).init(allocator, .{});
    defer list.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const value = random.int(i64);
        _ = try list.insert(value, value);
    }
}

/// Run all list benchmarks
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n# List Benchmarks\n\n", .{});
    std.debug.print("| Container | Operation | ns/op | Status |\n", .{});
    std.debug.print("|-----------|-----------|-------|--------|\n", .{});

    // SkipList insert
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchSkipListInsert, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 500) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| SkipList | insert | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // SkipList search
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchSkipListSearch, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 500) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| SkipList | search | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // XorLinkedList push
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchXorLinkedListPush, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 100) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| XorLinkedList | push | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // XorLinkedList iterate
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchXorLinkedListIterate, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 50) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| XorLinkedList | iterate | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // UnrolledLinkedList insert
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchUnrolledLinkedListAppend, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 100) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| UnrolledLinkedList | append | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // UnrolledLinkedList iterate
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchUnrolledLinkedListIterate, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 30) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| UnrolledLinkedList | iterate | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // ConcurrentSkipList insert
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchConcurrentSkipListInsert, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 1000) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| ConcurrentSkipList | insert | {d} | {s} |\n", .{ ns_per_op, status });
    }

    std.debug.print("\n## Summary\n\n", .{});
    std.debug.print("List benchmark suite completed.\n", .{});
    std.debug.print("- SkipList: O(log n) operations\n", .{});
    std.debug.print("- XorLinkedList: O(1) push, memory-efficient\n", .{});
    std.debug.print("- UnrolledLinkedList: Cache-friendly iteration\n", .{});
    std.debug.print("- ConcurrentSkipList: Lock-free concurrent access\n", .{});
}
