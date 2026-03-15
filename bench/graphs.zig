//! Benchmark suite for graph algorithms
//!
//! Validates PRD performance targets:
//! - Dijkstra (1M nodes, 5M edges): ≤ 500 ms

const std = @import("std");
const zuda = @import("zuda");
const bench = zuda.internal.bench;

const AdjacencyList = zuda.containers.graphs.AdjacencyList;
const Dijkstra = zuda.algorithms.graph.Dijkstra;

/// Context for u32 comparisons
const U32Context = struct {
    pub fn compare(_: U32Context, a: u32, b: u32) std.math.Order {
        return std.math.order(a, b);
    }

    pub fn hash(_: U32Context, key: u32) u64 {
        return key;
    }

    pub fn eql(_: U32Context, a: u32, b: u32) bool {
        return a == b;
    }
};

/// Benchmark: Dijkstra on 1M nodes, 5M edges
fn benchDijkstra(allocator: std.mem.Allocator) !void {
    const node_count = 1_000_000;
    const edge_count = 5_000_000;

    // Create graph with adjacency list
    var graph = AdjacencyList(u32, u32, U32Context, U32Context.hash, U32Context.eql).init(allocator, .{}, true);
    defer graph.deinit();

    // Add nodes
    var i: u32 = 0;
    while (i < node_count) : (i += 1) {
        try graph.addVertex(i);
    }

    // Add random weighted edges (avg 5 edges per node)
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var edge_idx: usize = 0;
    while (edge_idx < edge_count) : (edge_idx += 1) {
        const from = random.intRangeAtMost(u32, 0, node_count - 1);
        const to = random.intRangeAtMost(u32, 0, node_count - 1);
        const weight = random.intRangeAtMost(u32, 1, 100);

        // Skip self-loops
        if (from == to) continue;

        try graph.addEdge(from, to, weight);
    }

    // Run Dijkstra from node 0
    var result = try Dijkstra(u32, u32, U32Context).run(
        allocator,
        &graph,
        0,
        .{},
        0, // zero_weight
    );
    defer result.deinit();
}

/// Run all graph benchmarks and output markdown table
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n# Graph Algorithm Benchmarks\n\n", .{});
    std.debug.print("Validating PRD performance targets:\n", .{});
    std.debug.print("- Dijkstra: target ≤ 500 ms (1M nodes, 5M edges)\n\n", .{});

    // Dijkstra benchmark
    {
        std.debug.print("Running Dijkstra (1M nodes, 5M edges)...\n", .{});

        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 1,
            .min_iterations = 3,
            .max_iterations = 5,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchDijkstra, .{allocator});

        const ms = @divFloor(result.mean_ns, 1_000_000);
        std.debug.print("  Result: {d} ms (mean over {d} iterations)\n", .{ ms, result.iterations });

        if (ms <= 500) {
            std.debug.print("  ✓ PASS: meets target of ≤ 500 ms\n", .{});
        } else {
            std.debug.print("  ✗ FAIL: exceeds target of ≤ 500 ms\n", .{});
        }
    }

    std.debug.print("\n## Summary\n\n", .{});
    std.debug.print("Benchmark suite completed. See results above.\n", .{});
}
