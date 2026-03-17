//! Benchmark suite for queue-based containers
//!
//! Validates PRD performance targets for Phase 1 queues:
//! - Deque: O(1) push/pop from both ends
//! - LockFreeQueue: Lock-free concurrent enqueue/dequeue
//! - LockFreeStack: Lock-free concurrent push/pop
//! - WorkStealingDeque: Chase-Lev algorithm for task schedulers

const std = @import("std");
const zuda = @import("zuda");
const bench = zuda.internal.bench;
const builtin = @import("builtin");

const Deque = zuda.containers.queues.Deque;
const WorkStealingDeque = zuda.containers.queues.WorkStealingDeque;

// LockFreeQueue and LockFreeStack require 128-bit atomic support
// Currently only guaranteed on macOS (x86-64/ARM64)
const has_lockfree_support = switch (builtin.os.tag) {
    .macos => switch (builtin.cpu.arch) {
        .x86_64, .aarch64 => true,
        else => false,
    },
    else => false,
};

const LockFreeQueue = if (has_lockfree_support) zuda.containers.queues.LockFreeQueue else void;
const LockFreeStack = if (has_lockfree_support) zuda.containers.queues.LockFreeStack else void;

/// Benchmark: Deque push_back operations
fn benchDequePushBack(allocator: std.mem.Allocator) !void {
    var deque = Deque(i64).init(allocator);
    defer deque.deinit();

    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        try deque.push_back(@as(i64, @intCast(i)));
    }
}

/// Benchmark: Deque push_front operations
fn benchDequePushFront(allocator: std.mem.Allocator) !void {
    var deque = Deque(i64).init(allocator);
    defer deque.deinit();

    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        try deque.push_front(@as(i64, @intCast(i)));
    }
}

/// Benchmark: Deque pop_back after push_back
fn benchDequePopBack(allocator: std.mem.Allocator) !void {
    var deque = Deque(i64).init(allocator);
    defer deque.deinit();

    // Pre-populate
    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        try deque.push_back(@as(i64, @intCast(i)));
    }

    // Benchmark pop_back
    i = 0;
    while (i < count) : (i += 1) {
        _ = try deque.pop_back();
    }
}

/// Benchmark: LockFreeQueue enqueue operations
fn benchLockFreeQueueEnqueue(allocator: std.mem.Allocator) !void {
    if (comptime !has_lockfree_support) return;
    var queue = try LockFreeQueue(i64).init(allocator);
    defer queue.deinit();

    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        try queue.enqueue(@as(i64, @intCast(i)));
    }
}

/// Benchmark: LockFreeQueue dequeue operations
fn benchLockFreeQueueDequeue(allocator: std.mem.Allocator) !void {
    if (comptime !has_lockfree_support) return;
    var queue = try LockFreeQueue(i64).init(allocator);
    defer queue.deinit();

    // Pre-populate
    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        try queue.enqueue(@as(i64, @intCast(i)));
    }

    // Benchmark dequeue
    i = 0;
    while (i < count) : (i += 1) {
        _ = queue.dequeue();
    }
}

/// Benchmark: LockFreeStack push operations
fn benchLockFreeStackPush(allocator: std.mem.Allocator) !void {
    if (comptime !has_lockfree_support) return;
    var stack = LockFreeStack(i64).init(allocator);
    defer stack.deinit();

    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        try stack.push(@as(i64, @intCast(i)));
    }
}

/// Benchmark: LockFreeStack pop operations
fn benchLockFreeStackPop(allocator: std.mem.Allocator) !void {
    if (comptime !has_lockfree_support) return;
    var stack = LockFreeStack(i64).init(allocator);
    defer stack.deinit();

    // Pre-populate
    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        try stack.push(@as(i64, @intCast(i)));
    }

    // Benchmark pop
    i = 0;
    while (i < count) : (i += 1) {
        _ = stack.pop();
    }
}

/// Benchmark: WorkStealingDeque push operations
fn benchWorkStealingDequePush(allocator: std.mem.Allocator) !void {
    var deque = try WorkStealingDeque(i64).init(allocator);
    defer deque.deinit();

    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        try deque.push(@as(i64, @intCast(i)));
    }
}

/// Benchmark: WorkStealingDeque pop operations
fn benchWorkStealingDequePop(allocator: std.mem.Allocator) !void {
    var deque = try WorkStealingDeque(i64).init(allocator);
    defer deque.deinit();

    // Pre-populate
    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        try deque.push(@as(i64, @intCast(i)));
    }

    // Benchmark pop
    i = 0;
    while (i < count) : (i += 1) {
        _ = deque.pop();
    }
}

/// Benchmark: WorkStealingDeque steal operations
fn benchWorkStealingDequeSteal(allocator: std.mem.Allocator) !void {
    var deque = try WorkStealingDeque(i64).init(allocator);
    defer deque.deinit();

    // Pre-populate
    const count = 100_000;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        try deque.push(@as(i64, @intCast(i)));
    }

    // Benchmark steal
    i = 0;
    while (i < count) : (i += 1) {
        _ = deque.steal();
    }
}

/// Run all queue benchmarks
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n# Queue Benchmarks\n\n", .{});
    std.debug.print("| Container | Operation | ns/op | Status |\n", .{});
    std.debug.print("|-----------|-----------|-------|--------|\n", .{});

    // Deque push_back
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchDequePushBack, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 50) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| Deque | push_back | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // Deque push_front
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchDequePushFront, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 50) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| Deque | push_front | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // Deque pop_back
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchDequePopBack, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 50) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| Deque | pop_back | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // LockFreeQueue enqueue
    if (has_lockfree_support) {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchLockFreeQueueEnqueue, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 100) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| LockFreeQueue | enqueue | {d} | {s} |\n", .{ ns_per_op, status });
    } else {
        std.debug.print("| LockFreeQueue | enqueue | N/A | ⊗ UNSUPPORTED |\n", .{});
    }

    // LockFreeQueue dequeue
    if (has_lockfree_support) {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchLockFreeQueueDequeue, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 100) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| LockFreeQueue | dequeue | {d} | {s} |\n", .{ ns_per_op, status });
    } else {
        std.debug.print("| LockFreeQueue | dequeue | N/A | ⊗ UNSUPPORTED |\n", .{});
    }

    // LockFreeStack push
    if (has_lockfree_support) {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchLockFreeStackPush, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 100) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| LockFreeStack | push | {d} | {s} |\n", .{ ns_per_op, status });
    } else {
        std.debug.print("| LockFreeStack | push | N/A | ⊗ UNSUPPORTED |\n", .{});
    }

    // LockFreeStack pop
    if (has_lockfree_support) {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchLockFreeStackPop, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 100) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| LockFreeStack | pop | {d} | {s} |\n", .{ ns_per_op, status });
    } else {
        std.debug.print("| LockFreeStack | pop | N/A | ⊗ UNSUPPORTED |\n", .{});
    }

    // WorkStealingDeque push
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchWorkStealingDequePush, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 100) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| WorkStealingDeque | push | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // WorkStealingDeque pop
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchWorkStealingDequePop, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 100) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| WorkStealingDeque | pop | {d} | {s} |\n", .{ ns_per_op, status });
    }

    // WorkStealingDeque steal
    {
        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchWorkStealingDequeSteal, .{allocator});
        const ns_per_op = @divFloor(result.mean_ns, 100_000);
        const status = if (ns_per_op <= 100) "✓ PASS" else "⚠ SLOW";
        std.debug.print("| WorkStealingDeque | steal | {d} | {s} |\n", .{ ns_per_op, status });
    }

    std.debug.print("\n## Summary\n\n", .{});
    std.debug.print("Queue benchmark suite completed.\n", .{});
    std.debug.print("- Deque: O(1) amortized push/pop from both ends\n", .{});
    std.debug.print("- LockFreeQueue: Lock-free concurrent FIFO\n", .{});
    std.debug.print("- LockFreeStack: Lock-free concurrent LIFO\n", .{});
    std.debug.print("- WorkStealingDeque: Chase-Lev for work-stealing schedulers\n", .{});
}
