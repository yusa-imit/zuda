const std = @import("std");
const zuda = @import("zuda");

// HyperLogLog API Demo — Probabilistic cardinality estimation
//
// This example demonstrates HyperLogLog (HLL) usage for counting distinct elements
// with logarithmic space complexity. HLL is used in Redis (PFCOUNT), analytics
// platforms, and any system that needs to count unique items in massive datasets.
//
// Consumer use case: zoltraak (Redis-compatible server) uses HLL for PFCOUNT
// (80 lines custom implementation → zuda.containers.probabilistic.HyperLogLog)

// Context for hashing strings
const StringContext = struct {
    pub fn hash(_: StringContext, key: []const u8) u64 {
        return std.hash.Wyhash.hash(0, key);
    }
};

// Context for hashing integers
const IntContext = struct {
    pub fn hash(_: IntContext, key: u64) u64 {
        return std.hash.Wyhash.hash(0, std.mem.asBytes(&key));
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== HyperLogLog API Demo ===\n\n", .{});

    // Demo 1: Basic cardinality estimation
    try demo1_basic_cardinality(allocator);

    // Demo 2: Stream cardinality with duplicates
    try demo2_stream_cardinality(allocator);

    // Demo 3: Merge operation (distributed counting)
    try demo3_merge_distributed(allocator);

    // Demo 4: Memory efficiency comparison
    try demo4_memory_efficiency(allocator);

    std.debug.print("\n=== API Summary ===\n", .{});
    std.debug.print("HyperLogLog(T, Context, hashFn)\n", .{});
    std.debug.print("  - init(allocator, p, ctx) → !Self\n", .{});
    std.debug.print("  - add(item) → void                    // O(1) time\n", .{});
    std.debug.print("  - count() → u64                       // O(m) time, m = 2^p\n", .{});
    std.debug.print("  - merge(other) → !void                // O(m) time\n", .{});
    std.debug.print("  - clear() → void                      // Reset all registers\n", .{});
    std.debug.print("  - memoryUsage() → usize               // Bytes used\n", .{});
    std.debug.print("\nPrecision p (buckets m = 2^p):\n", .{});
    std.debug.print("  - p=10 (1024 buckets): ~3.2%% error, ~1KB\n", .{});
    std.debug.print("  - p=14 (16384 buckets): ~0.81%% error, ~16KB\n", .{});
    std.debug.print("  - p=16 (65536 buckets): ~0.4%% error, ~64KB\n", .{});
}

// Demo 1: Basic cardinality estimation
// Use case: Count unique visitors to a website
fn demo1_basic_cardinality(allocator: std.mem.Allocator) !void {
    std.debug.print("Demo 1: Basic Cardinality Estimation\n", .{});
    std.debug.print("Use case: Unique visitor counting (website analytics)\n\n", .{});

    const HLL = zuda.containers.probabilistic.HyperLogLog([]const u8, StringContext, StringContext.hash);
    var hll = try HLL.init(allocator, 14, .{}); // p=14 → ~0.81% error
    defer hll.deinit();

    // Simulate visitor IPs
    const visitors = [_][]const u8{
        "192.168.1.1",
        "192.168.1.2",
        "10.0.0.5",
        "172.16.0.10",
        "192.168.1.1", // duplicate
        "10.0.0.6",
        "192.168.1.2", // duplicate
        "172.16.0.11",
    };

    std.debug.print("Adding {} visitor IPs (with duplicates):\n", .{visitors.len});
    for (visitors) |ip| {
        hll.add(ip);
        std.debug.print("  {s}\n", .{ip});
    }

    const estimated = hll.count();
    const actual: u64 = 6; // 6 unique IPs
    const error_pct = @abs(@as(f64, @floatFromInt(estimated)) - @as(f64, @floatFromInt(actual))) / @as(f64, @floatFromInt(actual)) * 100.0;

    std.debug.print("\nEstimated unique visitors: {}\n", .{estimated});
    std.debug.print("Actual unique visitors: {}\n", .{actual});
    std.debug.print("Error: {d:.2}%\n", .{error_pct});
    std.debug.print("Memory used: {} bytes\n\n", .{hll.memoryUsage()});
}

// Demo 2: Stream cardinality with duplicates
// Use case: Count unique search queries in a stream
fn demo2_stream_cardinality(allocator: std.mem.Allocator) !void {
    std.debug.print("Demo 2: Stream Cardinality Estimation\n", .{});
    std.debug.print("Use case: Unique search queries in a high-volume stream\n\n", .{});

    const HLL = zuda.containers.probabilistic.HyperLogLog([]const u8, StringContext, StringContext.hash);
    var hll = try HLL.init(allocator, 14, .{});
    defer hll.deinit();

    // Simulate search query stream (many duplicates)
    const queries = [_][]const u8{
        "zig programming",
        "rust vs zig",
        "zig memory safety",
        "zig programming", // popular query repeated
        "comptime zig",
        "zig vs c",
        "rust vs zig", // popular query repeated
        "zig programming", // popular query repeated
        "zig allocators",
        "zig build system",
        "zig memory safety", // repeated
        "zig vs c", // repeated
    };

    std.debug.print("Processing {} search queries:\n", .{queries.len});
    for (queries, 0..) |query, i| {
        hll.add(query);
        if ((i + 1) % 4 == 0) {
            std.debug.print("After {} queries: estimated unique = {}\n", .{ i + 1, hll.count() });
        }
    }

    const final_estimated = hll.count();
    const actual_unique: u64 = 8; // 8 unique queries

    std.debug.print("\nFinal estimated unique queries: {}\n", .{final_estimated});
    std.debug.print("Actual unique queries: {}\n", .{actual_unique});
    std.debug.print("Total queries processed: {}\n", .{queries.len});
    std.debug.print("Deduplication rate: {d:.1}%\n\n", .{(1.0 - @as(f64, @floatFromInt(actual_unique)) / @as(f64, @floatFromInt(queries.len))) * 100.0});
}

// Demo 3: Merge operation for distributed counting
// Use case: Aggregate unique users across multiple servers
fn demo3_merge_distributed(allocator: std.mem.Allocator) !void {
    std.debug.print("Demo 3: Merge Operation (Distributed Counting)\n", .{});
    std.debug.print("Use case: Aggregate unique users from multiple data centers\n\n", .{});

    const HLL = zuda.containers.probabilistic.HyperLogLog(u64, IntContext, IntContext.hash);

    // Server A (data center 1)
    var hll_a = try HLL.init(allocator, 14, .{});
    defer hll_a.deinit();

    std.debug.print("Server A (user IDs 1-100):\n", .{});
    var i: u64 = 1;
    while (i <= 100) : (i += 1) {
        hll_a.add(i);
    }
    std.debug.print("  Unique users: {}\n", .{hll_a.count()});

    // Server B (data center 2, some overlap with A)
    var hll_b = try HLL.init(allocator, 14, .{});
    defer hll_b.deinit();

    std.debug.print("Server B (user IDs 51-150):\n", .{});
    i = 51;
    while (i <= 150) : (i += 1) {
        hll_b.add(i);
    }
    std.debug.print("  Unique users: {}\n", .{hll_b.count()});

    // Server C (data center 3, some overlap)
    var hll_c = try HLL.init(allocator, 14, .{});
    defer hll_c.deinit();

    std.debug.print("Server C (user IDs 101-200):\n", .{});
    i = 101;
    while (i <= 200) : (i += 1) {
        hll_c.add(i);
    }
    std.debug.print("  Unique users: {}\n", .{hll_c.count()});

    // Merge all servers
    var hll_global = try HLL.init(allocator, 14, .{});
    defer hll_global.deinit();

    try hll_global.merge(&hll_a);
    try hll_global.merge(&hll_b);
    try hll_global.merge(&hll_c);

    const estimated = hll_global.count();
    const actual: u64 = 200; // User IDs 1-200

    std.debug.print("\nGlobal unique users (after merge): {}\n", .{estimated});
    std.debug.print("Actual unique users: {}\n", .{actual});
    std.debug.print("Error: {d:.2}%\n\n", .{@abs(@as(f64, @floatFromInt(estimated)) - @as(f64, @floatFromInt(actual))) / @as(f64, @floatFromInt(actual)) * 100.0});
}

// Demo 4: Memory efficiency comparison
// Use case: Compare HLL vs HashSet for large cardinality
fn demo4_memory_efficiency(allocator: std.mem.Allocator) !void {
    std.debug.print("Demo 4: Memory Efficiency Comparison\n", .{});
    std.debug.print("Use case: HyperLogLog vs HashSet for 100K unique items\n\n", .{});

    const HLL = zuda.containers.probabilistic.HyperLogLog(u64, IntContext, IntContext.hash);

    // HyperLogLog approach
    var hll = try HLL.init(allocator, 14, .{});
    defer hll.deinit();

    std.debug.print("Adding 100,000 unique user IDs...\n", .{});
    var i: u64 = 1;
    while (i <= 100_000) : (i += 1) {
        hll.add(i);
    }

    const hll_estimate = hll.count();
    const hll_memory = hll.memoryUsage();

    // HashSet approach (for comparison)
    var hashset = std.AutoHashMap(u64, void).init(allocator);
    defer hashset.deinit();

    i = 1;
    while (i <= 100_000) : (i += 1) {
        try hashset.put(i, {});
    }

    const hashset_count = hashset.count();
    const hashset_memory = hashset_count * (@sizeOf(u64) + @sizeOf(void) + 8); // Rough estimate: key + value + overhead

    std.debug.print("\nHyperLogLog (p=14):\n", .{});
    std.debug.print("  Estimated count: {}\n", .{hll_estimate});
    std.debug.print("  Memory used: {} bytes ({d:.2} KB)\n", .{ hll_memory, @as(f64, @floatFromInt(hll_memory)) / 1024.0 });

    std.debug.print("\nHashSet (exact):\n", .{});
    std.debug.print("  Exact count: {}\n", .{hashset_count});
    std.debug.print("  Memory used: ~{} bytes ({d:.2} KB)\n", .{ hashset_memory, @as(f64, @floatFromInt(hashset_memory)) / 1024.0 });

    const memory_ratio = @as(f64, @floatFromInt(hashset_memory)) / @as(f64, @floatFromInt(hll_memory));
    std.debug.print("\nHashSet uses {d:.1}x more memory than HyperLogLog\n", .{memory_ratio});
    std.debug.print("HLL error: {d:.2}%\n", .{@abs(@as(f64, @floatFromInt(hll_estimate)) - @as(f64, @floatFromInt(hashset_count))) / @as(f64, @floatFromInt(hashset_count)) * 100.0});
    std.debug.print("\nTrade-off: {d:.1}% error for {d:.0}x memory savings\n\n", .{
        0.81, // p=14 standard error
        memory_ratio,
    });
}
