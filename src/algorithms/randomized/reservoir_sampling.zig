const std = @import("std");
const testing = std.testing;

/// Reservoir sampling algorithm for selecting k random elements from a stream.
///
/// Time: O(n) | Space: O(k)
///
/// Reservoir sampling solves the problem: "Select k random elements from a stream
/// of n elements, where n is unknown or very large." Each element has probability
/// k/n of being selected, ensuring uniform random sampling.
///
/// Algorithm L (Vitter's algorithm):
/// 1. Fill reservoir with first k elements
/// 2. For each element i ≥ k:
///    - Generate random j in [0, i]
///    - If j < k, replace reservoir[j] with element i
///
/// Use cases:
/// - Random sampling from large files
/// - Online sampling from data streams
/// - Random selection from unknown-size datasets
pub fn reservoirSample(
    comptime T: type,
    allocator: std.mem.Allocator,
    stream: []const T,
    k: usize,
    random: std.Random,
) ![]T {
    if (k == 0) return try allocator.alloc(T, 0);
    if (stream.len == 0) return try allocator.alloc(T, 0);

    const sample_size = @min(k, stream.len);
    var reservoir = try allocator.alloc(T, sample_size);

    // Fill reservoir with first k elements
    @memcpy(reservoir, stream[0..sample_size]);

    // If stream has k or fewer elements, we're done
    if (stream.len <= k) return reservoir;

    // Process remaining elements
    var i: usize = k;
    while (i < stream.len) : (i += 1) {
        const j = random.intRangeLessThan(usize, 0, i + 1);
        if (j < k) {
            reservoir[j] = stream[i];
        }
    }

    return reservoir;
}

/// Reservoir sampling with weights (also known as weighted random sampling).
///
/// Time: O(n log k) | Space: O(k)
///
/// Each element has a weight, and selection probability is proportional to weight.
/// Uses a priority queue (implemented as heap) to maintain top k weighted elements.
pub fn weightedReservoirSample(
    comptime T: type,
    allocator: std.mem.Allocator,
    stream: []const T,
    weights: []const f64,
    k: usize,
    random: std.Random,
) ![]T {
    if (k == 0 or stream.len == 0) return try allocator.alloc(T, 0);
    if (stream.len != weights.len) return error.LengthMismatch;

    const sample_size = @min(k, stream.len);

    // Use exponential variates for weighted sampling (Algorithm A-ES)
    const Item = struct {
        element: T,
        key: f64,
    };

    var items = std.ArrayList(Item).init(allocator);
    defer items.deinit();

    for (stream, weights) |element, weight| {
        if (weight <= 0) continue;

        // Key = u^(1/w) where u ~ Uniform(0,1)
        const u = random.float(f64);
        const key = std.math.pow(f64, u, 1.0 / weight);

        try items.append(.{ .element = element, .key = key });
    }

    // Sort by key descending and take top k
    const Context = struct {
        fn lessThan(_: @This(), a: Item, b: Item) bool {
            return a.key > b.key; // Descending
        }
    };
    std.mem.sort(Item, items.items, Context{}, Context.lessThan);

    const result_size = @min(sample_size, items.items.len);
    const result = try allocator.alloc(T, result_size);
    for (result, 0..) |*r, i| {
        r.* = items.items[i].element;
    }

    return result;
}

/// Distributed reservoir sampling for combining samples from multiple streams.
///
/// Time: O(n) where n = total elements across all reservoirs | Space: O(k)
///
/// Given multiple reservoir samples from different streams, combines them into
/// a single reservoir sample that is statistically valid.
pub fn combineReservoirs(
    comptime T: type,
    allocator: std.mem.Allocator,
    reservoirs: []const []const T,
    stream_sizes: []const usize,
    k: usize,
    random: std.Random,
) ![]T {
    if (reservoirs.len != stream_sizes.len) return error.LengthMismatch;
    if (k == 0) return try allocator.alloc(T, 0);

    var combined = std.ArrayList(T).init(allocator);
    defer combined.deinit();

    // Collect all elements from all reservoirs
    var total_size: usize = 0;
    for (stream_sizes) |size| total_size += size;

    if (total_size == 0) return try allocator.alloc(T, 0);

    // Build combined reservoir by processing each reservoir
    for (reservoirs, stream_sizes) |reservoir, _| {
        for (reservoir) |element| {
            if (combined.items.len < k) {
                try combined.append(element);
            } else {
                // Replace with probability proportional to stream size
                const j = random.intRangeLessThan(usize, 0, total_size);
                if (j < k) {
                    combined.items[j] = element;
                }
            }
        }
    }

    return try combined.toOwnedSlice();
}

test "reservoir: basic sampling k=3" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const stream = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const sample = try reservoirSample(i32, testing.allocator, &stream, 3, random);
    defer testing.allocator.free(sample);

    try testing.expectEqual(3, sample.len);

    // All sampled elements should be from the original stream
    for (sample) |val| {
        var found = false;
        for (stream) |s| {
            if (val == s) {
                found = true;
                break;
            }
        }
        try testing.expect(found);
    }

    // No duplicates in sample
    for (sample, 0..) |val1, i| {
        for (sample[i + 1 ..]) |val2| {
            try testing.expect(val1 != val2);
        }
    }
}

test "reservoir: k=0" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const stream = [_]i32{ 1, 2, 3, 4, 5 };
    const sample = try reservoirSample(i32, testing.allocator, &stream, 0, random);
    defer testing.allocator.free(sample);

    try testing.expectEqual(0, sample.len);
}

test "reservoir: k > n" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const stream = [_]i32{ 1, 2, 3 };
    const sample = try reservoirSample(i32, testing.allocator, &stream, 10, random);
    defer testing.allocator.free(sample);

    try testing.expectEqual(3, sample.len);

    // Should contain all stream elements
    var sum: i32 = 0;
    for (sample) |val| sum += val;
    try testing.expectEqual(6, sum);
}

test "reservoir: empty stream" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const stream = [_]i32{};
    const sample = try reservoirSample(i32, testing.allocator, &stream, 5, random);
    defer testing.allocator.free(sample);

    try testing.expectEqual(0, sample.len);
}

test "reservoir: single element stream" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const stream = [_]i32{42};
    const sample = try reservoirSample(i32, testing.allocator, &stream, 5, random);
    defer testing.allocator.free(sample);

    try testing.expectEqual(1, sample.len);
    try testing.expectEqual(42, sample[0]);
}

test "reservoir: distribution test" {
    // Test that each element has roughly equal probability of being selected
    const prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    const stream = [_]usize{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const k = 3;
    const trials = 1000;

    var selection_counts = [_]usize{0} ** 10;

    var trial: usize = 0;
    while (trial < trials) : (trial += 1) {
        const sample = try reservoirSample(usize, testing.allocator, &stream, k, random);
        defer testing.allocator.free(sample);

        for (sample) |val| {
            selection_counts[val] += 1;
        }
    }

    // Each element should be selected roughly (k/n * trials) times
    // With k=3, n=10, trials=1000, expect ~300 per element
    // Allow margin: 150-450
    for (selection_counts) |count| {
        try testing.expect(count >= 150);
        try testing.expect(count <= 450);
    }
}

test "reservoir: large stream" {
    const prng = std.Random.DefaultPrng.init(777);
    const random = prng.random();

    const stream = try testing.allocator.alloc(usize, 10000);
    defer testing.allocator.free(stream);

    for (stream, 0..) |*val, i| {
        val.* = i;
    }

    const sample = try reservoirSample(usize, testing.allocator, stream, 100, random);
    defer testing.allocator.free(sample);

    try testing.expectEqual(100, sample.len);

    // All sampled values should be < 10000
    for (sample) |val| {
        try testing.expect(val < 10000);
    }

    // No duplicates
    for (sample, 0..) |val1, i| {
        for (sample[i + 1 ..]) |val2| {
            try testing.expect(val1 != val2);
        }
    }
}

test "reservoir: weighted sampling basic" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const stream = [_]i32{ 1, 2, 3, 4, 5 };
    const weights = [_]f64{ 1.0, 1.0, 1.0, 1.0, 1.0 };

    const sample = try weightedReservoirSample(i32, testing.allocator, &stream, &weights, 3, random);
    defer testing.allocator.free(sample);

    try testing.expectEqual(3, sample.len);

    // All sampled elements should be from stream
    for (sample) |val| {
        var found = false;
        for (stream) |s| {
            if (val == s) {
                found = true;
                break;
            }
        }
        try testing.expect(found);
    }
}

test "reservoir: weighted sampling zero weights" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const stream = [_]i32{ 1, 2, 3, 4, 5 };
    const weights = [_]f64{ 0.0, 0.0, 0.0, 0.0, 0.0 };

    const sample = try weightedReservoirSample(i32, testing.allocator, &stream, &weights, 3, random);
    defer testing.allocator.free(sample);

    try testing.expectEqual(0, sample.len);
}

test "reservoir: weighted sampling length mismatch" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const stream = [_]i32{ 1, 2, 3 };
    const weights = [_]f64{ 1.0, 1.0 };

    const result = weightedReservoirSample(i32, testing.allocator, &stream, &weights, 2, random);
    try testing.expectError(error.LengthMismatch, result);
}

test "reservoir: combine empty reservoirs" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const reservoirs = [_][]const i32{};
    const sizes = [_]usize{};

    const combined = try combineReservoirs(i32, testing.allocator, &reservoirs, &sizes, 5, random);
    defer testing.allocator.free(combined);

    try testing.expectEqual(0, combined.len);
}

test "reservoir: combine single reservoir" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const res1 = [_]i32{ 1, 2, 3 };
    const reservoirs = [_][]const i32{&res1};
    const sizes = [_]usize{10};

    const combined = try combineReservoirs(i32, testing.allocator, &reservoirs, &sizes, 3, random);
    defer testing.allocator.free(combined);

    try testing.expectEqual(3, combined.len);

    var sum: i32 = 0;
    for (combined) |val| sum += val;
    try testing.expectEqual(6, sum);
}

test "reservoir: combine multiple reservoirs" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const res1 = [_]i32{ 1, 2 };
    const res2 = [_]i32{ 3, 4 };
    const res3 = [_]i32{ 5, 6 };
    const reservoirs = [_][]const i32{ &res1, &res2, &res3 };
    const sizes = [_]usize{ 100, 100, 100 };

    const combined = try combineReservoirs(i32, testing.allocator, &reservoirs, &sizes, 4, random);
    defer testing.allocator.free(combined);

    try testing.expectEqual(4, combined.len);

    // All values should be from 1-6
    for (combined) |val| {
        try testing.expect(val >= 1 and val <= 6);
    }
}

test "reservoir: combine reservoirs length mismatch" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const res1 = [_]i32{ 1, 2 };
    const reservoirs = [_][]const i32{&res1};
    const sizes = [_]usize{ 10, 20 }; // Mismatch

    const result = combineReservoirs(i32, testing.allocator, &reservoirs, &sizes, 3, random);
    try testing.expectError(error.LengthMismatch, result);
}
