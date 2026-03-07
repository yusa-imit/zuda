//! Testing utilities for zuda data structures
//!
//! Provides property-based testing helpers, invariant checkers,
//! and stress test utilities for containers.

const std = @import("std");
const testing = std.testing;

/// Random seed for reproducible tests
pub var prng: std.Random.DefaultPrng = undefined;
pub var random: std.Random = undefined;

/// Initialize the test random number generator with a seed
pub fn initRandom(seed: u64) void {
    prng = std.Random.DefaultPrng.init(seed);
    random = prng.random();
}

/// Initialize with timestamp-based seed
pub fn initRandomDefault() void {
    const seed = @as(u64, @intCast(std.time.milliTimestamp()));
    initRandom(seed);
}

/// Generate a slice of random integers
/// Caller owns the returned memory and must free it
pub fn generateRandomIntegers(comptime T: type, allocator: std.mem.Allocator, count: usize) ![]T {
    const slice = try allocator.alloc(T, count);
    for (slice) |*item| {
        item.* = random.int(T);
    }
    return slice;
}

/// Generate a slice of random integers in a specific range [min, max)
/// Caller owns the returned memory and must free it
pub fn generateRandomIntegersInRange(comptime T: type, allocator: std.mem.Allocator, count: usize, min: T, max: T) ![]T {
    const slice = try allocator.alloc(T, count);
    for (slice) |*item| {
        item.* = min + @as(T, @intCast(@mod(random.int(u64), @as(u64, @intCast(max - min)))));
    }
    return slice;
}

/// Generate a slice of unique random integers in range [0, max)
/// Returns error.Overflow if count > max
/// Caller owns the returned memory and must free it
pub fn generateUniqueRandomIntegers(comptime T: type, allocator: std.mem.Allocator, count: usize, max: T) ![]T {
    if (count > max) return error.Overflow;

    var seen = std.AutoHashMap(T, void).init(allocator);
    defer seen.deinit();

    const slice = try allocator.alloc(T, count);
    var generated: usize = 0;

    while (generated < count) {
        const value = @as(T, @intCast(@mod(random.int(u64), @as(u64, @intCast(max)))));
        const result = try seen.getOrPut(value);
        if (!result.found_existing) {
            slice[generated] = value;
            generated += 1;
        }
    }

    return slice;
}

/// Test operation types for property-based testing
pub const Operation = enum {
    insert,
    remove,
    lookup,
    update,
};

/// Generate a random sequence of operations
/// Caller owns the returned memory and must free it
pub fn generateOperationSequence(allocator: std.mem.Allocator, count: usize) ![]Operation {
    const slice = try allocator.alloc(Operation, count);
    for (slice) |*op| {
        op.* = random.enumValue(Operation);
    }
    return slice;
}

/// Weighted operation sequence generator
pub const OperationWeights = struct {
    insert: u32 = 25,
    remove: u32 = 25,
    lookup: u32 = 25,
    update: u32 = 25,
};

/// Generate a weighted random sequence of operations
/// Caller owns the returned memory and must free it
pub fn generateWeightedOperationSequence(
    allocator: std.mem.Allocator,
    count: usize,
    weights: OperationWeights,
) ![]Operation {
    const total_weight = weights.insert + weights.remove + weights.lookup + weights.update;
    const slice = try allocator.alloc(Operation, count);

    for (slice) |*op| {
        const roll = random.uintLessThan(u32, total_weight);
        if (roll < weights.insert) {
            op.* = .insert;
        } else if (roll < weights.insert + weights.remove) {
            op.* = .remove;
        } else if (roll < weights.insert + weights.remove + weights.lookup) {
            op.* = .lookup;
        } else {
            op.* = .update;
        }
    }

    return slice;
}

/// Stress test configuration
pub const StressTestConfig = struct {
    /// Number of operations to perform
    operation_count: usize = 10000,
    /// Value range for integers [0, max_value)
    max_value: i64 = 10000,
    /// Operation weights
    weights: OperationWeights = .{},
    /// Random seed (0 = use timestamp)
    seed: u64 = 0,
};

/// Memory leak detector context
pub const LeakDetector = struct {
    allocator: std.mem.Allocator,

    pub fn init() LeakDetector {
        return .{ .allocator = testing.allocator };
    }

    pub fn getAllocator(self: *const LeakDetector) std.mem.Allocator {
        return self.allocator;
    }

    /// Verify no leaks occurred during test
    /// This is automatically handled by std.testing.allocator
    pub fn checkLeaks(self: *const LeakDetector) !void {
        _ = self;
        // std.testing.allocator automatically fails if leaks are detected
    }
};

/// Check if a slice is sorted according to a comparison function
pub fn isSorted(
    comptime T: type,
    slice: []const T,
    comptime lessThan: fn (a: T, b: T) bool,
) bool {
    if (slice.len <= 1) return true;

    var i: usize = 1;
    while (i < slice.len) : (i += 1) {
        if (lessThan(slice[i], slice[i - 1])) {
            return false;
        }
    }

    return true;
}

/// Check if a slice contains duplicates
pub fn hasDuplicates(comptime T: type, allocator: std.mem.Allocator, slice: []const T) !bool {
    var seen = std.AutoHashMap(T, void).init(allocator);
    defer seen.deinit();

    for (slice) |item| {
        const result = try seen.getOrPut(item);
        if (result.found_existing) {
            return true;
        }
    }

    return false;
}

/// Verify that two slices contain the same elements (order doesn't matter)
pub fn containsSameElements(
    comptime T: type,
    allocator: std.mem.Allocator,
    expected: []const T,
    actual: []const T,
) !bool {
    if (expected.len != actual.len) return false;

    var expected_counts = std.AutoHashMap(T, usize).init(allocator);
    defer expected_counts.deinit();

    for (expected) |item| {
        const result = try expected_counts.getOrPut(item);
        if (result.found_existing) {
            result.value_ptr.* += 1;
        } else {
            result.value_ptr.* = 1;
        }
    }

    for (actual) |item| {
        const entry = expected_counts.getPtr(item) orelse return false;
        if (entry.* == 0) return false;
        entry.* -= 1;
    }

    return true;
}

test "random integer generation" {
    initRandomDefault();

    const slice = try generateRandomIntegers(i32, testing.allocator, 100);
    defer testing.allocator.free(slice);

    try testing.expectEqual(100, slice.len);
}

test "unique random integer generation" {
    initRandomDefault();

    const slice = try generateUniqueRandomIntegers(i32, testing.allocator, 50, 100);
    defer testing.allocator.free(slice);

    try testing.expectEqual(50, slice.len);
    try testing.expect(!try hasDuplicates(i32, testing.allocator, slice));
}

test "operation sequence generation" {
    initRandomDefault();

    const ops = try generateOperationSequence(testing.allocator, 100);
    defer testing.allocator.free(ops);

    try testing.expectEqual(100, ops.len);
}

test "weighted operation sequence" {
    initRandomDefault();

    const weights = OperationWeights{
        .insert = 50,
        .remove = 20,
        .lookup = 20,
        .update = 10,
    };

    const ops = try generateWeightedOperationSequence(testing.allocator, 1000, weights);
    defer testing.allocator.free(ops);

    try testing.expectEqual(1000, ops.len);
}

test "is sorted check" {
    const sorted = [_]i32{ 1, 2, 3, 4, 5 };
    const unsorted = [_]i32{ 1, 3, 2, 4, 5 };

    try testing.expect(isSorted(i32, &sorted, struct {
        fn lessThan(a: i32, b: i32) bool {
            return a < b;
        }
    }.lessThan));

    try testing.expect(!isSorted(i32, &unsorted, struct {
        fn lessThan(a: i32, b: i32) bool {
            return a < b;
        }
    }.lessThan));
}

test "contains same elements" {
    const a = [_]i32{ 1, 2, 3, 4, 5 };
    const b = [_]i32{ 5, 4, 3, 2, 1 };
    const c = [_]i32{ 1, 2, 3, 4, 6 };

    try testing.expect(try containsSameElements(i32, testing.allocator, &a, &b));
    try testing.expect(!try containsSameElements(i32, testing.allocator, &a, &c));
}
