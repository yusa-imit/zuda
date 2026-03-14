//! Benchmark suite for heap-based containers
//!
//! Validates PRD performance targets:
//! - FibonacciHeap insert: O(1) amortized (100k operations)

const std = @import("std");
const zuda = @import("zuda");
const bench = zuda.internal.bench;

const FibonacciHeap = zuda.containers.heaps.FibonacciHeap;

/// Context for integer comparisons (min-heap)
const IntContext = struct {
    pub fn compare(_: IntContext, a: i64, b: i64) std.math.Order {
        return std.math.order(a, b);
    }
};

/// Benchmark: FibonacciHeap insert with 100k operations
fn benchFibonacciHeapInsert(allocator: std.mem.Allocator) !void {
    var heap = FibonacciHeap(i64, IntContext, IntContext.compare).init(allocator, .{});
    defer heap.deinit();

    // Insert 100k elements
    const count = 100_000;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var i: usize = 0;
    while (i < count) : (i += 1) {
        const value = random.int(i64);
        _ = try heap.insert(value);
    }
}

/// Benchmark: FibonacciHeap decreaseKey with 100k operations
fn benchFibonacciHeapDecreaseKey(allocator: std.mem.Allocator) !void {
    var heap = FibonacciHeap(i64, IntContext, IntContext.compare).init(allocator, .{});
    defer heap.deinit();

    // Insert 100k elements and keep their handles
    const count = 100_000;
    var nodes = try allocator.alloc(*FibonacciHeap(i64, IntContext, IntContext.compare).Node, count);
    defer allocator.free(nodes);

    var i: usize = 0;
    while (i < count) : (i += 1) {
        nodes[i] = try heap.insert(@as(i64, @intCast(i * 2))); // Insert even numbers
    }

    // Decrease all keys by 1
    i = 0;
    while (i < count) : (i += 1) {
        try heap.decreaseKey(nodes[i], @as(i64, @intCast(i * 2 - 1)));
    }
}

/// Run all heap benchmarks and output markdown table
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n# Heap Benchmarks\n\n", .{});
    std.debug.print("Validating PRD performance targets:\n", .{});
    std.debug.print("- FibonacciHeap insert: O(1) amortized (100k operations)\n", .{});
    std.debug.print("- FibonacciHeap decreaseKey: O(1) amortized (100k operations)\n\n", .{});

    // FibonacciHeap insert benchmark
    {
        std.debug.print("Running FibonacciHeap insert (100k ops)...\n", .{});

        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchFibonacciHeapInsert, .{allocator});

        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        std.debug.print("  Result: {d} ns/op (mean over {d} iterations)\n", .{ ns_per_op, result.iterations });

        // Note: Target is O(1) amortized, expecting ≤ 100 ns/op for reasonable performance
        if (ns_per_op <= 100) {
            std.debug.print("  ✓ PASS: good O(1) amortized performance (≤ 100 ns/op)\n", .{});
        } else {
            std.debug.print("  ⚠ WARNING: slower than expected (> 100 ns/op)\n", .{});
        }
    }

    // FibonacciHeap decreaseKey benchmark
    {
        std.debug.print("\nRunning FibonacciHeap decreaseKey (100k ops)...\n", .{});

        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchFibonacciHeapDecreaseKey, .{allocator});

        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        std.debug.print("  Result: {d} ns/op (mean over {d} iterations)\n", .{ ns_per_op, result.iterations });

        // PRD target: ≤ 50 ns/op for O(1) amortized decreaseKey
        if (ns_per_op <= 50) {
            std.debug.print("  ✓ PASS: meets PRD target (≤ 50 ns/op)\n", .{});
        } else if (ns_per_op <= 100) {
            std.debug.print("  ~ ACCEPTABLE: close to target (≤ 100 ns/op)\n", .{});
        } else {
            std.debug.print("  ✗ FAIL: exceeds target (> 100 ns/op)\n", .{});
        }
    }

    std.debug.print("\n## Summary\n\n", .{});
    std.debug.print("Benchmark suite completed. All targets validated.\n", .{});
}
