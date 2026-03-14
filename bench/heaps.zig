//! Benchmark suite for heap-based containers
//!
//! Validates PRD performance targets:
//! - FibonacciHeap decrease-key: ≤ 50 ns amortized

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

/// Benchmark: FibonacciHeap decrease-key with 100k operations
fn benchFibonacciHeapDecreaseKey(allocator: std.mem.Allocator) !void {
    var heap = FibonacciHeap(i64, IntContext, IntContext.compare).init(allocator, .{});
    defer heap.deinit();

    // Insert 100k elements
    const count = 100_000;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // Track handles for decrease-key
    const handles = try allocator.alloc(*FibonacciHeap(i64, IntContext, IntContext.compare).Node, count);
    defer allocator.free(handles);

    var i: usize = 0;
    while (i < count) : (i += 1) {
        const value = random.int(i64);
        handles[i] = try heap.insert(value);
    }

    // Perform decrease-key on all elements
    i = 0;
    while (i < count) : (i += 1) {
        const current_value = handles[i].value;
        const new_value = current_value - 1000; // Decrease by constant
        try heap.decreaseKey(handles[i], new_value);
    }
}

/// Run all heap benchmarks and output markdown table
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n# Heap Benchmarks\n\n", .{});
    std.debug.print("Validating PRD performance targets:\n", .{});
    std.debug.print("- FibonacciHeap decrease-key: target ≤ 50 ns amortized (100k operations)\n\n", .{});

    // FibonacciHeap decrease-key benchmark
    {
        std.debug.print("Running FibonacciHeap decrease-key (100k ops)...\n", .{});

        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchFibonacciHeapDecreaseKey, .{allocator});

        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        std.debug.print("  Result: {d} ns/op (mean over {d} iterations)\n", .{ ns_per_op, result.iterations });

        if (ns_per_op <= 50) {
            std.debug.print("  ✓ PASS: meets target of ≤ 50 ns/op\n", .{});
        } else {
            std.debug.print("  ✗ FAIL: exceeds target of ≤ 50 ns/op\n", .{});
        }
    }

    std.debug.print("\n## Summary\n\n", .{});
    std.debug.print("Benchmark suite completed. See results above.\n", .{});
}
