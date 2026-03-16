//! Micro-benchmarks to isolate RedBlackTree performance bottlenecks

const std = @import("std");
const zuda = @import("zuda");
const bench = zuda.internal.bench;

const RedBlackTree = zuda.containers.trees.RedBlackTree;

const IntContext = struct {
    pub fn compare(_: IntContext, a: i64, b: i64) std.math.Order {
        return std.math.order(a, b);
    }
};

/// Measure allocator overhead in isolation
fn benchAllocatorOnly(allocator: std.mem.Allocator) !void {
    const count = 1_000_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const ptr = try allocator.create(i64);
        ptr.* = @intCast(i);
        allocator.destroy(ptr);
    }
}

/// Measure comparison function overhead
fn benchComparisonOnly() !void {
    const context = IntContext{};
    const count = 1_000_000;
    var total: i64 = 0;

    var i: i64 = 0;
    while (i < count) : (i += 1) {
        const order = IntContext.compare(context, i, @divFloor(i, 2));
        total += @intFromEnum(order);
    }

    // Prevent optimization
    if (total == 12345678) unreachable;
}

/// Measure lookup without rebalancing overhead (pre-built tree)
fn benchLookupNoRebalance(allocator: std.mem.Allocator) !void {
    var tree = RedBlackTree(i64, i64, IntContext, IntContext.compare).init(allocator, .{});
    defer tree.deinit();

    // Build balanced tree (sequential inserts create worst-case unbalanced, then rebalanced)
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const count = 100_000; // Smaller for faster benchmark
    var keys: std.ArrayList(i64) = .{};
    defer keys.deinit(allocator);

    var i: usize = 0;
    while (i < count) : (i += 1) {
        try keys.append(allocator, random.int(i64));
    }

    for (keys.items) |key| {
        _ = try tree.insert(key, key);
    }

    // Benchmark lookup
    random.shuffle(i64, keys.items);
    for (keys.items) |key| {
        _ = tree.get(key);
    }
}

/// Measure insert into empty tree (no rebalancing)
fn benchInsertNoRebalance(allocator: std.mem.Allocator) !void {
    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        var tree = RedBlackTree(i64, i64, IntContext, IntContext.compare).init(allocator, .{});
        defer tree.deinit();

        _ = try tree.insert(42, 42); // Single insert, no rebalancing
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n# RedBlackTree Micro-Benchmarks\n\n", .{});
    std.debug.print("Isolating performance bottlenecks:\n\n", .{});

    // Allocator overhead
    {
        std.debug.print("1. Allocator overhead (create + destroy i64)...\n", .{});

        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchAllocatorOnly, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 1_000_000);
        std.debug.print("   Result: {d} ns/op\n\n", .{ns_per_op});
    }

    // Comparison overhead
    {
        std.debug.print("2. Comparison function overhead (IntContext.compare)...\n", .{});

        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchComparisonOnly, .{});
        const ns_per_op = @divFloor(result.mean_ns, 1_000_000);
        std.debug.print("   Result: {d} ns/op\n\n", .{ns_per_op});
    }

    // Lookup on pre-built tree
    {
        std.debug.print("3. Lookup on pre-built tree (100k keys)...\n", .{});

        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchLookupNoRebalance, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        std.debug.print("   Result: {d} ns/op\n\n", .{ns_per_op});
    }

    // Single insert (no rebalancing)
    {
        std.debug.print("4. Single insert into empty tree (no rebalancing)...\n", .{});

        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchInsertNoRebalance, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        std.debug.print("   Result: {d} ns/op\n\n", .{ns_per_op});
    }

    std.debug.print("## Analysis\n\n", .{});
    std.debug.print("- If allocator overhead >> comparison overhead → bottleneck is memory allocation\n", .{});
    std.debug.print("- If lookup ≈ single insert → traversal dominates (cache misses)\n", .{});
    std.debug.print("- If 1M insert >> single insert × 20 → rebalancing overhead is significant\n", .{});
}
