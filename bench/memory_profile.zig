//! Memory profiling benchmarks for zuda containers
//!
//! Measures memory usage patterns for key data structures
//! to identify excessive allocation overhead.

const std = @import("std");
const zuda = @import("zuda");
const bench = zuda.internal.bench;

const PROFILE_SIZE = 10_000; // Reduced from typical benchmarks to focus on memory patterns

/// Profile RedBlackTree memory usage
fn profileRedBlackTree(allocator: std.mem.Allocator) !void {
    const CompareFn = struct {
        pub fn compare(_: void, a: i64, b: i64) std.math.Order {
            return std.math.order(a, b);
        }
    }.compare;

    var tree = zuda.containers.trees.RedBlackTree(i64, i64, void, CompareFn).init(allocator, {});
    defer tree.deinit();

    var i: i64 = 0;
    while (i < PROFILE_SIZE) : (i += 1) {
        _ = try tree.insert(i, i * 2);
    }
}

/// Profile SkipList memory usage
fn profileSkipList(allocator: std.mem.Allocator) !void {
    const CompareFn = struct {
        pub fn compare(_: void, a: i64, b: i64) std.math.Order {
            return std.math.order(a, b);
        }
    }.compare;

    var list = try zuda.containers.lists.SkipList(i64, i64, void, CompareFn).init(allocator, {});
    defer list.deinit();

    var i: i64 = 0;
    while (i < PROFILE_SIZE) : (i += 1) {
        _ = try list.insert(i, i * 2);
    }
}

/// Profile FibonacciHeap memory usage
fn profileFibonacciHeap(allocator: std.mem.Allocator) !void {
    const CompareFn = struct {
        pub fn compare(_: void, a: i64, b: i64) std.math.Order {
            return std.math.order(a, b);
        }
    }.compare;

    var heap = zuda.containers.heaps.FibonacciHeap(i64, void, CompareFn).init(allocator, {});
    defer heap.deinit();

    var i: i64 = 0;
    while (i < PROFILE_SIZE) : (i += 1) {
        _ = try heap.insert(i);
    }
}

/// Profile BTree memory usage
fn profileBTree(allocator: std.mem.Allocator) !void {
    const I64Context = struct {
        pub fn compare(_: @This(), a: i64, b: i64) std.math.Order {
            return std.math.order(a, b);
        }
    };

    var tree = zuda.containers.trees.BTree(i64, i64, 128, I64Context).init(allocator, .{});
    defer tree.deinit();

    var i: i64 = 0;
    while (i < PROFILE_SIZE) : (i += 1) {
        _ = try tree.insert(i, i * 2);
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Memory Profiling Benchmarks ===\n", .{});
    std.debug.print("Profile size: {d} operations\n\n", .{PROFILE_SIZE});
    std.debug.print("| Container | Mean (ns) | Peak Mem (B) | Current Mem (B) | Allocs | Frees |\n", .{});
    std.debug.print("|-----------|-----------|--------------|-----------------|--------|-------|\n", .{});

    // RedBlackTree
    {
        var tracker = bench.MemoryTracker.init(allocator);
        var b = try bench.Benchmark.initWithMemoryTracking(tracker.allocator(), .{
            .min_iterations = 10,
            .max_iterations = 100,
        }, &tracker);
        defer b.deinit();

        const result = try b.run(profileRedBlackTree, .{tracker.allocator()});
        if (result.memory) |mem| {
            std.debug.print("| RedBlackTree | {d} | {d} | {d} | {d} | {d} |\n", .{ result.mean_ns, mem.peak_bytes, mem.current_bytes, mem.allocation_count, mem.free_count });
        }
    }

    // SkipList
    {
        var tracker = bench.MemoryTracker.init(allocator);
        var b = try bench.Benchmark.initWithMemoryTracking(tracker.allocator(), .{
            .min_iterations = 10,
            .max_iterations = 100,
        }, &tracker);
        defer b.deinit();

        const result = try b.run(profileSkipList, .{tracker.allocator()});
        if (result.memory) |mem| {
            std.debug.print("| SkipList | {d} | {d} | {d} | {d} | {d} |\n", .{ result.mean_ns, mem.peak_bytes, mem.current_bytes, mem.allocation_count, mem.free_count });
        }
    }

    // FibonacciHeap
    {
        var tracker = bench.MemoryTracker.init(allocator);
        var b = try bench.Benchmark.initWithMemoryTracking(tracker.allocator(), .{
            .min_iterations = 10,
            .max_iterations = 100,
        }, &tracker);
        defer b.deinit();

        const result = try b.run(profileFibonacciHeap, .{tracker.allocator()});
        if (result.memory) |mem| {
            std.debug.print("| FibonacciHeap | {d} | {d} | {d} | {d} | {d} |\n", .{ result.mean_ns, mem.peak_bytes, mem.current_bytes, mem.allocation_count, mem.free_count });
        }
    }

    // BTree
    {
        var tracker = bench.MemoryTracker.init(allocator);
        var b = try bench.Benchmark.initWithMemoryTracking(tracker.allocator(), .{
            .min_iterations = 10,
            .max_iterations = 100,
        }, &tracker);
        defer b.deinit();

        const result = try b.run(profileBTree, .{tracker.allocator()});
        if (result.memory) |mem| {
            std.debug.print("| BTree(128) | {d} | {d} | {d} | {d} | {d} |\n", .{ result.mean_ns, mem.peak_bytes, mem.current_bytes, mem.allocation_count, mem.free_count });
        }
    }

    std.debug.print("\n=== Analysis ===\n", .{});
    std.debug.print("Peak Mem (B): Maximum memory allocated at any point during operation\n", .{});
    std.debug.print("Current Mem (B): Memory still allocated after operation (should be 0)\n", .{});
    std.debug.print("\nNon-zero Current Mem indicates a memory leak.\n", .{});
    std.debug.print("High Peak Mem relative to data size indicates excessive allocation overhead.\n\n", .{});
}
