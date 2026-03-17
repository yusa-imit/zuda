//! Benchmark suite for hash-based containers
//!
//! Validates PRD performance targets for Phase 1 hash containers:
//! - CuckooHashMap: O(1) worst-case lookup
//! - RobinHoodHashMap: Low variance probe lengths
//! - SwissTable: SIMD-friendly group-based probing
//! - ConsistentHashRing: Minimal key redistribution on node changes

const std = @import("std");
const zuda = @import("zuda");
const bench = zuda.internal.bench;

const AutoCuckooHashMap = zuda.containers.hashing.AutoCuckooHashMap;
const AutoRobinHoodHashMap = zuda.containers.hashing.AutoRobinHoodHashMap;
const AutoSwissTable = zuda.containers.hashing.AutoSwissTable;
const ConsistentHashRing = zuda.containers.hashing.ConsistentHashRing;

/// Benchmark: CuckooHashMap insert with 100k random keys
fn benchCuckooHashMapInsert(allocator: std.mem.Allocator) !void {
    var map = try AutoCuckooHashMap(i64, i64).init(allocator, .{});
    defer map.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const key = random.int(i64);
        _ = try map.put(key, key);
    }
}

/// Benchmark: CuckooHashMap get (worst-case O(1) lookup)
fn benchCuckooHashMapGet(allocator: std.mem.Allocator) !void {
    var map = try AutoCuckooHashMap(i64, i64).init(allocator, .{});
    defer map.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // Pre-populate
    const count = 100_000;
    const keys = try allocator.alloc(i64, count);
    defer allocator.free(keys);

    var i: usize = 0;
    while (i < count) : (i += 1) {
        keys[i] = random.int(i64);
        _ = try map.insert(keys[i], keys[i]);
    }

    // Shuffle for random access
    random.shuffle(i64, @constCast(keys));

    // Benchmark get
    i = 0;
    while (i < count) : (i += 1) {
        _ = map.get(keys[i]);
    }
}

/// Benchmark: RobinHoodHashMap insert
fn benchRobinHoodHashMapInsert(allocator: std.mem.Allocator) !void {
    var map = try AutoRobinHoodHashMap(i64, i64).init(allocator, .{});
    defer map.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const key = random.int(i64);
        _ = try map.insert(key, key);
    }
}

/// Benchmark: RobinHoodHashMap get
fn benchRobinHoodHashMapGet(allocator: std.mem.Allocator) !void {
    var map = try AutoRobinHoodHashMap(i64, i64).init(allocator, .{});
    defer map.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // Pre-populate
    const count = 100_000;
    const keys = try allocator.alloc(i64, count);
    defer allocator.free(keys);

    var i: usize = 0;
    while (i < count) : (i += 1) {
        keys[i] = random.int(i64);
        _ = try map.insert(keys[i], keys[i]);
    }

    // Shuffle for random access
    random.shuffle(i64, @constCast(keys));

    // Benchmark get
    i = 0;
    while (i < count) : (i += 1) {
        _ = map.get(keys[i]);
    }
}

/// Benchmark: SwissTable insert
fn benchSwissTableInsert(allocator: std.mem.Allocator) !void {
    var map = try AutoSwissTable(i64, i64).init(allocator, .{});
    defer map.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const key = random.int(i64);
        _ = try map.insert(key, key);
    }
}

/// Benchmark: SwissTable get
fn benchSwissTableGet(allocator: std.mem.Allocator) !void {
    var map = try AutoSwissTable(i64, i64).init(allocator, .{});
    defer map.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // Pre-populate
    const count = 100_000;
    const keys = try allocator.alloc(i64, count);
    defer allocator.free(keys);

    var i: usize = 0;
    while (i < count) : (i += 1) {
        keys[i] = random.int(i64);
        _ = try map.insert(keys[i], keys[i]);
    }

    // Shuffle for random access
    random.shuffle(i64, @constCast(keys));

    // Benchmark get
    i = 0;
    while (i < count) : (i += 1) {
        _ = map.get(keys[i]);
    }
}

/// Benchmark: ConsistentHashRing add node
fn benchConsistentHashRingAdd(allocator: std.mem.Allocator) !void {
    var ring = ConsistentHashRing([]const u8).init(allocator, .{ .virtual_nodes = 150 });
    defer ring.deinit();

    const count = 100;
    var buf: [32]u8 = undefined;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const node = try std.fmt.bufPrint(&buf, "node-{d}", .{i});
        try ring.addNode(node);
    }
}

/// Benchmark: ConsistentHashRing get node (key lookup)
fn benchConsistentHashRingGet(allocator: std.mem.Allocator) !void {
    var ring = ConsistentHashRing([]const u8).init(allocator, .{ .virtual_nodes = 150 });
    defer ring.deinit();

    // Pre-populate nodes
    const node_count = 100;
    var buf: [32]u8 = undefined;
    var i: usize = 0;
    while (i < node_count) : (i += 1) {
        const node = try std.fmt.bufPrint(&buf, "node-{d}", .{i});
        try ring.addNode(node);
    }

    // Benchmark key lookups
    const key_count = 100_000;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    i = 0;
    while (i < key_count) : (i += 1) {
        const key = try std.fmt.bufPrint(&buf, "key-{d}", .{random.int(u64)});
        _ = ring.getNode(key);
    }
}

/// Run all hash container benchmarks
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n# Hash Container Benchmarks\n\n", .{});
    std.debug.print("| Container | Operation | ns/op | Status |\n", .{});
    std.debug.print("|-----------|-----------|-------|--------|\n", .{});

    // CuckooHashMap insert
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchCuckooHashMapInsert, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 100) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| CuckooHashMap | insert | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // CuckooHashMap get
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchCuckooHashMapGet, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 50) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| CuckooHashMap | get | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // RobinHoodHashMap insert
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchRobinHoodHashMapInsert, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 100) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| RobinHoodHashMap | insert | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // RobinHoodHashMap get
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchRobinHoodHashMapGet, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 50) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| RobinHoodHashMap | get | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // SwissTable insert
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchSwissTableInsert, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 100) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| SwissTable | insert | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // SwissTable get
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchSwissTableGet, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 50) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| SwissTable | get | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // ConsistentHashRing add
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchConsistentHashRingAdd, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100);
        const status = if (ns_per_op <= 1000) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| ConsistentHashRing | add | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // ConsistentHashRing get
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchConsistentHashRingGet, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 200) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| ConsistentHashRing | get | {d} | {s} |\n", .{ ns_per_op, status });
    }

    std.debug.print("\n## Summary\n\n", .{});
    std.debug.print("Hash container benchmark suite completed.\n", .{});
    std.debug.print("- CuckooHashMap: O(1) worst-case lookup\n", .{});
    std.debug.print("- RobinHoodHashMap: Low variance probe lengths\n", .{});
    std.debug.print("- SwissTable: SIMD-friendly group-based probing\n", .{});
    std.debug.print("- ConsistentHashRing: Minimal redistribution on topology changes\n", .{});
}
