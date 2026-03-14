//! Benchmark suite for probabilistic data structures
//!
//! Validates PRD performance targets:
//! - BloomFilter lookup: ≥ 100M ops/sec

const std = @import("std");
const zuda = @import("zuda");
const bench = zuda.internal.bench;

const BloomFilter = zuda.containers.probabilistic.BloomFilter;

/// Context for u64 hashing
const U64Context = struct {
    pub fn hash(_: U64Context, key: u64, seed: u64) u64 {
        var h = key +% seed;
        h ^= h >> 33;
        h *%= 0xff51afd7ed558ccd;
        h ^= h >> 33;
        h *%= 0xc4ceb9fe1a85ec53;
        h ^= h >> 33;
        return h;
    }
};

/// Benchmark: BloomFilter lookup with 10M operations
fn benchBloomFilterLookup(allocator: std.mem.Allocator) !void {
    // Create bloom filter with 1M bits, 7 hash functions
    // For 1M elements with k=7: p ≈ 0.01 false positive rate
    const m = 1_000_000 * 10; // 10 bits per element
    const k = 7;
    var filter = try BloomFilter(u64, U64Context, U64Context.hash).init(allocator, m, k, .{});
    defer filter.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // Insert 1M elements
    const insert_count = 1_000_000;
    var i: usize = 0;
    while (i < insert_count) : (i += 1) {
        const value = random.int(u64);
        filter.add(value);
    }

    // Perform 10M lookups (mix of present and absent keys)
    const lookup_count = 10_000_000;
    i = 0;
    while (i < lookup_count) : (i += 1) {
        const value = random.int(u64);
        _ = filter.contains(value);
    }
}

/// Run all probabilistic benchmarks and output markdown table
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n# Probabilistic Data Structure Benchmarks\n\n", .{});
    std.debug.print("Validating PRD performance targets:\n", .{});
    std.debug.print("- BloomFilter lookup: target ≥ 100M ops/sec (10M lookups)\n\n", .{});

    // BloomFilter lookup benchmark
    {
        std.debug.print("Running BloomFilter lookup (10M ops)...\n", .{});

        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchBloomFilterLookup, .{allocator});

        const ns_per_op = @divFloor(result.mean_ns, 10_000_000);
        const ops_per_sec = @divFloor(1_000_000_000, ns_per_op);
        const million_ops_per_sec = @divFloor(ops_per_sec, 1_000_000);

        std.debug.print("  Result: {d} ns/op, {d}M ops/sec (mean over {d} iterations)\n", .{ ns_per_op, million_ops_per_sec, result.iterations });

        if (million_ops_per_sec >= 100) {
            std.debug.print("  ✓ PASS: meets target of ≥ 100M ops/sec\n", .{});
        } else {
            std.debug.print("  ✗ FAIL: below target of ≥ 100M ops/sec\n", .{});
        }
    }

    std.debug.print("\n## Summary\n\n", .{});
    std.debug.print("Benchmark suite completed. See results above.\n", .{});
}
