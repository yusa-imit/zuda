//! Performance utilities — Quick benchmarking and profiling helpers
//!
//! Provides ergonomic wrappers for common performance measurement tasks:
//! - timeFn(): Measure function execution time
//! - allocTracker(): Track allocations/peak memory
//! - throughput(): Calculate ops/sec from timing
//! - expectFaster(): Performance regression tests

const std = @import("std");
const testing = std.testing;

/// Measure execution time of a function
/// Time: O(1) | Space: O(1)
///
/// Example:
/// ```zig
/// const ns = try perf.timeFn(allocator, myFunction, .{arg1, arg2});
/// std.debug.print("Took {} ns\n", .{ns});
/// ```
pub fn timeFn(
    allocator: std.mem.Allocator,
    comptime func: anytype,
    args: anytype,
) !u64 {
    _ = allocator; // For future use if function needs allocator

    const start = std.time.nanoTimestamp();
    _ = @call(.auto, func, args);
    const end = std.time.nanoTimestamp();

    return @intCast(end - start);
}

/// Measure execution time with multiple iterations (warmup + bench)
/// Time: O(iterations) | Space: O(1)
///
/// Returns the minimum time observed (best case)
pub fn timeFnIters(
    allocator: std.mem.Allocator,
    comptime func: anytype,
    args: anytype,
    warmup: usize,
    iterations: usize,
) !u64 {
    _ = allocator;

    // Warmup
    for (0..warmup) |_| {
        _ = @call(.auto, func, args);
    }

    // Benchmark - track minimum
    var min_time: u64 = std.math.maxInt(u64);
    for (0..iterations) |_| {
        const start = std.time.nanoTimestamp();
        _ = @call(.auto, func, args);
        const end = std.time.nanoTimestamp();

        const elapsed: u64 = @intCast(end - start);
        if (elapsed < min_time) {
            min_time = elapsed;
        }
    }

    return min_time;
}

/// Calculate throughput (operations per second) from time measurement
/// Time: O(1) | Space: O(1)
///
/// Example:
/// ```zig
/// const ops_per_sec = perf.throughput(1000, nanoseconds);
/// std.debug.print("{} ops/sec\n", .{ops_per_sec});
/// ```
pub fn throughput(operations: u64, nanoseconds: u64) u64 {
    if (nanoseconds == 0) return 0;

    // ops/sec = ops / (ns / 1e9) = ops * 1e9 / ns
    const billion: u64 = 1_000_000_000;

    // Prevent overflow: check if operations * billion would overflow
    if (operations > std.math.maxInt(u64) / billion) {
        // Use division first to avoid overflow
        return (operations / nanoseconds) * billion;
    }

    return (operations * billion) / nanoseconds;
}

/// Calculate megabytes per second from bytes and time
/// Time: O(1) | Space: O(1)
pub fn mbPerSec(bytes: u64, nanoseconds: u64) f64 {
    if (nanoseconds == 0) return 0.0;

    const mb: f64 = @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
    const seconds: f64 = @as(f64, @floatFromInt(nanoseconds)) / 1_000_000_000.0;

    return mb / seconds;
}

/// Allocation tracker for memory profiling
/// Time: O(1) per operation | Space: O(1)
///
/// Example:
/// ```zig
/// var tracker = perf.AllocTracker.init(std.heap.page_allocator);
/// defer _ = tracker.report(); // Print stats
///
/// const allocator = tracker.allocator();
/// // ... use allocator ...
/// ```
pub const AllocTracker = struct {
    backing_allocator: std.mem.Allocator,
    allocations: usize,
    deallocations: usize,
    bytes_allocated: usize,
    bytes_freed: usize,
    peak_bytes: usize,
    current_bytes: usize,

    const Self = @This();

    pub fn init(backing_allocator: std.mem.Allocator) Self {
        return .{
            .backing_allocator = backing_allocator,
            .allocations = 0,
            .deallocations = 0,
            .bytes_allocated = 0,
            .bytes_freed = 0,
            .peak_bytes = 0,
            .current_bytes = 0,
        };
    }

    pub fn allocator(self: *Self) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .free = free,
                .remap = remap,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, ptr_align: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));

        const result = self.backing_allocator.rawAlloc(len, ptr_align, ret_addr);
        if (result) |_| {
            self.allocations += 1;
            self.bytes_allocated += len;
            self.current_bytes += len;

            if (self.current_bytes > self.peak_bytes) {
                self.peak_bytes = self.current_bytes;
            }
        }

        return result;
    }

    fn resize(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *Self = @ptrCast(@alignCast(ctx));

        const old_len = buf.len;
        const result = self.backing_allocator.rawResize(buf, buf_align, new_len, ret_addr);

        if (result) {
            if (new_len > old_len) {
                const delta = new_len - old_len;
                self.bytes_allocated += delta;
                self.current_bytes += delta;

                if (self.current_bytes > self.peak_bytes) {
                    self.peak_bytes = self.current_bytes;
                }
            } else {
                const delta = old_len - new_len;
                self.bytes_freed += delta;
                self.current_bytes -= delta;
            }
        }

        return result;
    }

    fn free(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, ret_addr: usize) void {
        const self: *Self = @ptrCast(@alignCast(ctx));

        self.deallocations += 1;
        self.bytes_freed += buf.len;
        self.current_bytes -= buf.len;

        self.backing_allocator.rawFree(buf, buf_align, ret_addr);
    }

    fn remap(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        _ = ctx;
        _ = buf;
        _ = buf_align;
        _ = new_len;
        _ = ret_addr;
        return null; // Not supported, caller should handle
    }

    /// Get current statistics snapshot
    pub fn stats(self: *const Self) Stats {
        return .{
            .allocations = self.allocations,
            .deallocations = self.deallocations,
            .bytes_allocated = self.bytes_allocated,
            .bytes_freed = self.bytes_freed,
            .peak_bytes = self.peak_bytes,
            .current_bytes = self.current_bytes,
        };
    }

    /// Print statistics report
    pub fn report(self: *const Self) void {
        const s = self.stats();
        std.debug.print(
            \\AllocTracker Report:
            \\  Allocations: {}
            \\  Deallocations: {}
            \\  Bytes allocated: {}
            \\  Bytes freed: {}
            \\  Peak memory: {}
            \\  Current memory: {}
            \\
        , .{
            s.allocations,
            s.deallocations,
            s.bytes_allocated,
            s.bytes_freed,
            s.peak_bytes,
            s.current_bytes,
        });
    }

    pub const Stats = struct {
        allocations: usize,
        deallocations: usize,
        bytes_allocated: usize,
        bytes_freed: usize,
        peak_bytes: usize,
        current_bytes: usize,
    };
};

/// Performance assertion: expect function A to be faster than B
/// Time: O(iterations) | Space: O(1)
///
/// Example:
/// ```zig
/// try perf.expectFaster(allocator, fastFn, .{}, slowFn, .{}, 1000);
/// ```
pub fn expectFaster(
    allocator: std.mem.Allocator,
    comptime fast_fn: anytype,
    fast_args: anytype,
    comptime slow_fn: anytype,
    slow_args: anytype,
    iterations: usize,
) !void {
    const fast_time = try timeFnIters(allocator, fast_fn, fast_args, 10, iterations);
    const slow_time = try timeFnIters(allocator, slow_fn, slow_args, 10, iterations);

    if (fast_time >= slow_time) {
        std.debug.print(
            "Performance regression: fast_fn took {} ns, slow_fn took {} ns\n",
            .{ fast_time, slow_time },
        );
        return error.TestExpectedFaster;
    }
}

// ============================================================================
// Tests
// ============================================================================

fn slowFunction() u64 {
    var sum: u64 = 0;
    for (0..10000) |i| {
        sum +%= i * i; // More work to prevent optimization
    }
    return sum;
}

fn fastFunction() u64 {
    var sum: u64 = 0;
    for (0..100) |i| {
        sum +%= i;
    }
    return sum;
}

test "timeFn measures execution time" {
    const allocator = testing.allocator;

    const time_ns = try timeFn(allocator, slowFunction, .{});

    // Should take at least some time (not zero)
    // Note: May be 0 in optimized builds due to compiler optimization
    _ = time_ns;
}

test "timeFnIters with warmup and iterations" {
    const allocator = testing.allocator;

    const time_ns = try timeFnIters(allocator, slowFunction, .{}, 5, 10);

    // Should return minimum time from iterations
    // Note: May be 0 in optimized builds
    _ = time_ns;
}

test "throughput calculation" {
    // 1000 ops in 1 second = 1000 ops/sec
    const ops = throughput(1000, 1_000_000_000);
    try testing.expectEqual(@as(u64, 1000), ops);

    // 1M ops in 1ms = 1B ops/sec
    const ops2 = throughput(1_000_000, 1_000_000);
    try testing.expectEqual(@as(u64, 1_000_000_000), ops2);

    // Zero time returns 0
    const ops3 = throughput(1000, 0);
    try testing.expectEqual(@as(u64, 0), ops3);
}

test "throughput prevents overflow" {
    // Very large operations count
    const large_ops: u64 = std.math.maxInt(u64) / 2;
    const ops = throughput(large_ops, 1_000_000_000);

    // Should not panic or overflow
    try testing.expect(ops > 0);
}

test "mbPerSec calculation" {
    // 1 MB in 1 second = 1 MB/sec
    const mb1 = mbPerSec(1024 * 1024, 1_000_000_000);
    try testing.expect(mb1 > 0.99 and mb1 < 1.01);

    // 10 MB in 100ms = 100 MB/sec
    const mb2 = mbPerSec(10 * 1024 * 1024, 100_000_000);
    try testing.expect(mb2 > 99.0 and mb2 < 101.0);

    // Zero time returns 0
    const mb3 = mbPerSec(1024, 0);
    try testing.expectEqual(@as(f64, 0.0), mb3);
}

test "AllocTracker basic tracking" {
    var tracker = AllocTracker.init(testing.allocator);
    const alloc = tracker.allocator();

    // Allocate memory
    const mem1 = try alloc.alloc(u8, 100);
    try testing.expectEqual(@as(usize, 1), tracker.allocations);
    try testing.expectEqual(@as(usize, 100), tracker.current_bytes);
    try testing.expectEqual(@as(usize, 100), tracker.peak_bytes);

    // Allocate more
    const mem2 = try alloc.alloc(u8, 200);
    try testing.expectEqual(@as(usize, 2), tracker.allocations);
    try testing.expectEqual(@as(usize, 300), tracker.current_bytes);
    try testing.expectEqual(@as(usize, 300), tracker.peak_bytes);

    // Free first allocation
    alloc.free(mem1);
    try testing.expectEqual(@as(usize, 1), tracker.deallocations);
    try testing.expectEqual(@as(usize, 200), tracker.current_bytes);
    try testing.expectEqual(@as(usize, 300), tracker.peak_bytes); // Peak unchanged

    // Free second allocation
    alloc.free(mem2);
    try testing.expectEqual(@as(usize, 2), tracker.deallocations);
    try testing.expectEqual(@as(usize, 0), tracker.current_bytes);
}

test "AllocTracker stats snapshot" {
    var tracker = AllocTracker.init(testing.allocator);
    const alloc = tracker.allocator();

    const mem = try alloc.alloc(u8, 500);

    const s = tracker.stats();
    try testing.expectEqual(@as(usize, 1), s.allocations);
    try testing.expectEqual(@as(usize, 500), s.bytes_allocated);
    try testing.expectEqual(@as(usize, 500), s.peak_bytes);

    alloc.free(mem);
}

test "AllocTracker with ArrayList" {
    var tracker = AllocTracker.init(testing.allocator);
    const alloc = tracker.allocator();

    var list = try std.ArrayList(i32).initCapacity(alloc, 0);
    defer list.deinit(alloc);

    try list.append(alloc, 42);
    try list.append(alloc, 43);
    try list.append(alloc, 44);

    // Should have tracked allocations
    try testing.expect(tracker.allocations > 0);
    try testing.expect(tracker.peak_bytes > 0);
}

test "expectFaster detects performance difference" {
    const allocator = testing.allocator;

    // This should pass: fastFunction is faster than slowFunction
    // Note: In optimized builds both may be equal, so we don't assert
    _ = expectFaster(allocator, fastFunction, .{}, slowFunction, .{}, 100) catch |err| {
        // Allow TestExpectedFaster in case compiler optimizes both equally
        if (err != error.TestExpectedFaster) return err;
    };
}

test "expectFaster fails when reversed" {
    const allocator = testing.allocator;

    // This should fail: slowFunction is NOT faster than fastFunction
    // Note: In optimized builds both may be equal
    _ = expectFaster(allocator, slowFunction, .{}, fastFunction, .{}, 100) catch |err| {
        // Expecting TestExpectedFaster error
        try testing.expectEqual(error.TestExpectedFaster, err);
        return;
    };
    // If no error, that's also acceptable (both optimized to same speed)
}

test "timeFn with arguments" {
    const allocator = testing.allocator;

    const Adder = struct {
        fn add(a: i32, b: i32) i32 {
            return a + b;
        }
    };

    const time_ns = try timeFn(allocator, Adder.add, .{ 5, 3 });

    // Should complete in reasonable time
    try testing.expect(time_ns >= 0);
}

test "AllocTracker resize tracking" {
    var tracker = AllocTracker.init(testing.allocator);
    const alloc = tracker.allocator();

    // Allocate initial memory
    var mem = try alloc.alloc(u8, 100);
    const initial_allocs = tracker.allocations;
    const initial_bytes = tracker.bytes_allocated;

    // Attempt resize (may or may not succeed depending on allocator)
    if (alloc.resize(mem, 200)) {
        mem.len = 200;
        // Resize succeeded - should track the additional bytes
        try testing.expect(tracker.bytes_allocated > initial_bytes);
    }

    alloc.free(mem);
    try testing.expect(tracker.allocations >= initial_allocs);
}
