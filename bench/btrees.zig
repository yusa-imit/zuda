//! Benchmark suite for B-Tree containers
//!
//! Validates PRD performance targets:
//! - BTree(128) range scan: ≥ 50M keys/sec (sequential)

const std = @import("std");
const zuda = @import("zuda");
const bench = zuda.internal.bench;

const BTree = zuda.containers.trees.BTree;

/// Context for u64 comparisons
const U64Context = struct {
    pub fn compare(_: U64Context, a: u64, b: u64) std.math.Order {
        return std.math.order(a, b);
    }
};

/// Benchmark: BTree(128) range scan on 1M sequential keys
fn benchBTreeRangeScan(allocator: std.mem.Allocator) !void {
    const order = 128;
    var tree = BTree(u64, u64, U64Context, U64Context.compare, order).init(allocator, .{});
    defer tree.deinit();

    // Insert 1M sequential keys
    const count = 1_000_000;
    var i: u64 = 0;
    while (i < count) : (i += 1) {
        try tree.insert(i, i);
    }

    // Range scan from 0 to count-1
    var iter = tree.iterator();
    var scanned: usize = 0;
    while (iter.next()) |_| {
        scanned += 1;
    }

    // Verify we scanned all keys
    if (scanned != count) {
        return error.InvalidScanCount;
    }
}

/// Run all B-Tree benchmarks and output markdown table
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n# B-Tree Benchmarks\n\n", .{});
    std.debug.print("Validating PRD performance targets:\n", .{});
    std.debug.print("- BTree(128) range scan: target ≥ 50M keys/sec (sequential, 1M keys)\n\n", .{});

    // BTree range scan benchmark
    {
        std.debug.print("Running BTree(128) range scan (1M keys)...\n", .{});

        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchBTreeRangeScan, .{allocator});

        const count = 1_000_000;
        const ns_per_key = @divFloor(result.mean_ns, count);
        const keys_per_sec = @divFloor(1_000_000_000, ns_per_key);
        const million_keys_per_sec = @divFloor(keys_per_sec, 1_000_000);

        std.debug.print("  Result: {d} ns/key, {d}M keys/sec (mean over {d} iterations)\n", .{ ns_per_key, million_keys_per_sec, result.iterations });

        if (million_keys_per_sec >= 50) {
            std.debug.print("  ✓ PASS: meets target of ≥ 50M keys/sec\n", .{});
        } else {
            std.debug.print("  ✗ FAIL: below target of ≥ 50M keys/sec\n", .{});
        }
    }

    std.debug.print("\n## Summary\n\n", .{});
    std.debug.print("Benchmark suite completed. See results above.\n", .{});
}
