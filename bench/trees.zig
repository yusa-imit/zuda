//! Benchmark suite for tree-based containers
//!
//! Validates PRD performance targets:
//! - RedBlackTree insert: ≤ 200 ns/op (1M random keys)
//! - RedBlackTree lookup: ≤ 150 ns/op (1M random keys)

const std = @import("std");
const zuda = @import("zuda");
const bench = zuda.internal.bench;

const RedBlackTree = zuda.containers.trees.RedBlackTree;

/// Context for integer comparisons
const IntContext = struct {
    pub fn compare(_: IntContext, a: i64, b: i64) std.math.Order {
        return std.math.order(a, b);
    }
};

/// Benchmark: RedBlackTree insert with 1M random keys
fn benchRedBlackTreeInsert(allocator: std.mem.Allocator) !void {
    var tree = RedBlackTree(i64, i64, IntContext, IntContext.compare).init(allocator, .{});
    defer tree.deinit();

    // Generate 1M random keys
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const count = 1_000_000;
    const keys = try allocator.alloc(i64, count);
    defer allocator.free(keys);

    for (keys) |*key| {
        key.* = random.int(i64);
    }

    // Benchmark insertion
    var i: usize = 0;
    while (i < count) : (i += 1) {
        _ = try tree.insert(keys[i], keys[i]);
    }
}

/// Benchmark: RedBlackTree lookup with 1M random keys
fn benchRedBlackTreeLookup(allocator: std.mem.Allocator) !void {
    var tree = RedBlackTree(i64, i64, IntContext, IntContext.compare).init(allocator, .{});
    defer tree.deinit();

    // Generate and insert 1M random keys
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const count = 1_000_000;
    const keys = try allocator.alloc(i64, count);
    defer allocator.free(keys);

    for (keys) |*key| {
        key.* = random.int(i64);
    }

    var i: usize = 0;
    while (i < count) : (i += 1) {
        _ = try tree.insert(keys[i], keys[i]);
    }

    // Benchmark lookup (shuffle keys for random access)
    random.shuffle(i64, @constCast(keys));

    i = 0;
    while (i < count) : (i += 1) {
        _ = tree.get(keys[i]);
    }
}

/// Run all tree benchmarks and output markdown table
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n# Tree Benchmarks\n\n", .{});
    std.debug.print("Validating PRD performance targets:\n", .{});
    std.debug.print("- RedBlackTree insert: target ≤ 200 ns/op (1M random keys)\n", .{});
    std.debug.print("- RedBlackTree lookup: target ≤ 150 ns/op (1M random keys)\n\n", .{});

    // RedBlackTree insert benchmark
    {
        std.debug.print("Running RedBlackTree insert (1M keys)...\n", .{});

        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchRedBlackTreeInsert, .{allocator});

        const ns_per_op = @divFloor(result.mean_ns, 1_000_000);
        std.debug.print("  Result: {d} ns/op (mean over {d} iterations)\n", .{ ns_per_op, result.iterations });

        if (ns_per_op <= 200) {
            std.debug.print("  ✓ PASS: meets target of ≤ 200 ns/op\n", .{});
        } else {
            std.debug.print("  ✗ FAIL: exceeds target of ≤ 200 ns/op\n", .{});
        }
    }

    std.debug.print("\n", .{});

    // RedBlackTree lookup benchmark
    {
        std.debug.print("Running RedBlackTree lookup (1M keys)...\n", .{});

        // Pre-build tree (not timed)
        var tree = RedBlackTree(i64, i64, IntContext, IntContext.compare).init(allocator, .{});
        defer tree.deinit();

        var prng = std.Random.DefaultPrng.init(42);
        const random = prng.random();

        const count = 1_000_000;
        const keys = try allocator.alloc(i64, count);
        defer allocator.free(keys);

        for (keys) |*k| k.* = random.int(i64);

        var i: usize = 0;
        while (i < count) : (i += 1) {
            _ = try tree.insert(keys[i], keys[i]);
        }

        // Shuffle for random access pattern
        random.shuffle(i64, @constCast(keys));

        // Create closure for lookup-only benchmark
        const LookupBench = struct {
            fn run(tree_ptr: *const RedBlackTree(i64, i64, IntContext, IntContext.compare), key_slice: []const i64) void {
                for (key_slice) |key| {
                    _ = tree_ptr.get(key);
                }
            }
        };

        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(LookupBench.run, .{ &tree, keys });

        const ns_per_op = @divFloor(result.mean_ns, count);
        std.debug.print("  Result: {d} ns/op (mean over {d} iterations)\n", .{ ns_per_op, result.iterations });

        if (ns_per_op <= 150) {
            std.debug.print("  ✓ PASS: meets target of ≤ 150 ns/op\n", .{});
        } else {
            std.debug.print("  ✗ FAIL: exceeds target of ≤ 150 ns/op\n", .{});
        }
    }

    std.debug.print("\n## Summary\n\n", .{});
    std.debug.print("Benchmark suite completed. See results above.\n", .{});
}
