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
        try heap.insert(value);
    }
}

/// Run all heap benchmarks and output markdown table
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n# Heap Benchmarks\n\n", .{});
    std.debug.print("Validating PRD performance targets:\n", .{});
    std.debug.print("- FibonacciHeap insert: O(1) amortized (100k operations)\n\n", .{});

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

    std.debug.print("\n## Summary\n\n", .{});
    std.debug.print("Benchmark suite completed. See results above.\n", .{});
    std.debug.print("\n## Note\n\n", .{});
    std.debug.print("FibonacciHeap decrease-key benchmark cannot be run because insert() does not return node handles.\n", .{});
    std.debug.print("This is a known API limitation that needs to be addressed.\n", .{});
}
