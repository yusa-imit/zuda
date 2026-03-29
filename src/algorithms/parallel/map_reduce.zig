const std = @import("std");
const Allocator = std.mem.Allocator;

/// Map-Reduce result containing keys and aggregated values
pub fn MapReduceResult(comptime K: type, comptime V: type) type {
    return struct {
        keys: []K,
        values: []V,
        allocator: Allocator,

        const Self = @This();

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.keys);
            self.allocator.free(self.values);
        }
    };
}

/// Map-Reduce framework for parallel data processing
///
/// Implements the map-reduce pattern: map input to key-value pairs,
/// shuffle and group by key, then reduce values for each key.
///
/// Time: O(n log n) for sorting shuffle phase
/// Space: O(n) for intermediate key-value pairs
///
/// Example:
/// ```zig
/// // Word count example
/// const words = [_][]const u8{ "hello", "world", "hello", "zig" };
/// const result = try mapReduce([]const u8, i32, allocator, &words, mapWord, reduceCount);
/// defer result.deinit();
/// ```
pub fn mapReduce(
    comptime K: type,
    comptime V: type,
    allocator: Allocator,
    input: []const K,
    comptime mapper: fn (K) []const struct { key: K, value: V },
    comptime reducer: fn (K, []const V) V,
) !MapReduceResult(K, V) {
    // Map phase - emit key-value pairs
    var intermediate = std.ArrayList(struct { key: K, value: V }).init(allocator);
    defer intermediate.deinit();

    for (input) |item| {
        const pairs = mapper(item);
        for (pairs) |pair| {
            try intermediate.append(pair);
        }
    }

    if (intermediate.items.len == 0) {
        return MapReduceResult(K, V){
            .keys = try allocator.alloc(K, 0),
            .values = try allocator.alloc(V, 0),
            .allocator = allocator,
        };
    }

    // Shuffle phase - sort by key
    const sortFn = struct {
        fn f(_: void, a: @TypeOf(intermediate.items[0]), b: @TypeOf(intermediate.items[0])) bool {
            return std.sort.asc(K)({}, a.key, b.key);
        }
    }.f;
    std.mem.sort(@TypeOf(intermediate.items[0]), intermediate.items, {}, sortFn);

    // Reduce phase - aggregate values for each key
    var result_keys = std.ArrayList(K).init(allocator);
    defer result_keys.deinit();
    var result_values = std.ArrayList(V).init(allocator);
    defer result_values.deinit();

    var current_key = intermediate.items[0].key;
    var current_values = std.ArrayList(V).init(allocator);
    defer current_values.deinit();

    for (intermediate.items) |pair| {
        if (!std.meta.eql(pair.key, current_key)) {
            // Reduce current group
            const reduced = reducer(current_key, current_values.items);
            try result_keys.append(current_key);
            try result_values.append(reduced);

            // Start new group
            current_key = pair.key;
            current_values.clearRetainingCapacity();
        }
        try current_values.append(pair.value);
    }

    // Reduce last group
    const reduced = reducer(current_key, current_values.items);
    try result_keys.append(current_key);
    try result_values.append(reduced);

    return MapReduceResult(K, V){
        .keys = try result_keys.toOwnedSlice(),
        .values = try result_values.toOwnedSlice(),
        .allocator = allocator,
    };
}

/// Word count mapper - emits (word, 1) for each word
fn wordCountMapper(word: []const u8) []const struct { key: []const u8, value: i32 } {
    const result = [_]struct { key: []const u8, value: i32 }{
        .{ .key = word, .value = 1 },
    };
    return &result;
}

/// Word count reducer - sums counts for each word
fn wordCountReducer(_: []const u8, counts: []const i32) i32 {
    var sum: i32 = 0;
    for (counts) |count| {
        sum += count;
    }
    return sum;
}

/// Parallel group by operation
///
/// Groups elements by key using a key extraction function.
/// Returns map of keys to lists of values.
///
/// Time: O(n log n) for sorting
/// Space: O(n) for grouped results
///
/// Example:
/// ```zig
/// const data = [_]struct{ id: i32, value: i32 }{
///     .{ .id = 1, .value = 10 },
///     .{ .id = 2, .value = 20 },
///     .{ .id = 1, .value = 30 },
/// };
/// const grouped = try groupBy(i32, @TypeOf(data[0]), allocator, &data, getID);
/// defer grouped.deinit();
/// ```
pub fn GroupByResult(comptime K: type, comptime V: type) type {
    return struct {
        keys: []K,
        groups: [][]V,
        allocator: Allocator,

        const Self = @This();

        pub fn deinit(self: *Self) void {
            for (self.groups) |group| {
                self.allocator.free(group);
            }
            self.allocator.free(self.groups);
            self.allocator.free(self.keys);
        }
    };
}

pub fn groupBy(
    comptime K: type,
    comptime V: type,
    allocator: Allocator,
    input: []const V,
    comptime keyFn: fn (V) K,
) !GroupByResult(K, V) {
    if (input.len == 0) {
        return GroupByResult(K, V){
            .keys = try allocator.alloc(K, 0),
            .groups = try allocator.alloc([]V, 0),
            .allocator = allocator,
        };
    }

    // Create key-value pairs
    var pairs = try allocator.alloc(struct { key: K, value: V }, input.len);
    defer allocator.free(pairs);

    for (input, 0..) |item, i| {
        pairs[i] = .{ .key = keyFn(item), .value = item };
    }

    // Sort by key
    const sortFn = struct {
        fn f(_: void, a: @TypeOf(pairs[0]), b: @TypeOf(pairs[0])) bool {
            return std.sort.asc(K)({}, a.key, b.key);
        }
    }.f;
    std.mem.sort(@TypeOf(pairs[0]), pairs, {}, sortFn);

    // Group consecutive equal keys
    var result_keys = std.ArrayList(K).init(allocator);
    defer result_keys.deinit();
    var result_groups = std.ArrayList([]V).init(allocator);
    defer result_groups.deinit();

    var current_key = pairs[0].key;
    var current_group = std.ArrayList(V).init(allocator);
    defer current_group.deinit();

    for (pairs) |pair| {
        if (!std.meta.eql(pair.key, current_key)) {
            try result_keys.append(current_key);
            try result_groups.append(try current_group.toOwnedSlice());

            current_key = pair.key;
            current_group = std.ArrayList(V).init(allocator);
        }
        try current_group.append(pair.value);
    }

    // Add last group
    try result_keys.append(current_key);
    try result_groups.append(try current_group.toOwnedSlice());

    return GroupByResult(K, V){
        .keys = try result_keys.toOwnedSlice(),
        .groups = try result_groups.toOwnedSlice(),
        .allocator = allocator,
    };
}

/// Parallel partition operation
///
/// Splits array into two groups based on predicate.
/// Elements satisfying predicate go to first group, others to second.
///
/// Time: O(n) with parallel partitioning
/// Space: O(n) for result arrays
///
/// Example:
/// ```zig
/// const arr = [_]i32{ 1, 2, 3, 4, 5, 6 };
/// const result = try partition(i32, allocator, &arr, isEven);
/// defer result.true_partition.deinit();
/// defer result.false_partition.deinit();
/// // true_partition: [2, 4, 6]
/// // false_partition: [1, 3, 5]
/// ```
pub fn PartitionResult(comptime T: type) type {
    return struct {
        true_partition: []T,
        false_partition: []T,
        allocator: Allocator,

        const Self = @This();

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.true_partition);
            self.allocator.free(self.false_partition);
        }
    };
}

pub fn partition(
    comptime T: type,
    allocator: Allocator,
    arr: []const T,
    comptime pred: fn (T) bool,
) !PartitionResult(T) {
    var true_list = std.ArrayList(T).init(allocator);
    defer true_list.deinit();
    var false_list = std.ArrayList(T).init(allocator);
    defer false_list.deinit();

    for (arr) |val| {
        if (pred(val)) {
            try true_list.append(val);
        } else {
            try false_list.append(val);
        }
    }

    return PartitionResult(T){
        .true_partition = try true_list.toOwnedSlice(),
        .false_partition = try false_list.toOwnedSlice(),
        .allocator = allocator,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "word count map-reduce" {
    const allocator = std.testing.allocator;

    const words = [_][]const u8{ "hello", "world", "hello", "zig", "world", "hello" };

    var result = try mapReduce([]const u8, i32, allocator, &words, wordCountMapper, wordCountReducer);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 3), result.keys.len);

    // Check counts (order depends on string comparison)
    var hello_found = false;
    var world_found = false;
    var zig_found = false;

    for (result.keys, result.values) |key, count| {
        if (std.mem.eql(u8, key, "hello")) {
            try std.testing.expectEqual(@as(i32, 3), count);
            hello_found = true;
        } else if (std.mem.eql(u8, key, "world")) {
            try std.testing.expectEqual(@as(i32, 2), count);
            world_found = true;
        } else if (std.mem.eql(u8, key, "zig")) {
            try std.testing.expectEqual(@as(i32, 1), count);
            zig_found = true;
        }
    }

    try std.testing.expect(hello_found and world_found and zig_found);
}

test "map-reduce - empty input" {
    const allocator = std.testing.allocator;

    const words = [_][]const u8{};
    var result = try mapReduce([]const u8, i32, allocator, &words, wordCountMapper, wordCountReducer);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 0), result.keys.len);
}

test "group by - integer keys" {
    const allocator = std.testing.allocator;

    const Item = struct { id: i32, value: i32 };
    const data = [_]Item{
        .{ .id = 1, .value = 10 },
        .{ .id = 2, .value = 20 },
        .{ .id = 1, .value = 30 },
        .{ .id = 3, .value = 40 },
        .{ .id = 2, .value = 50 },
    };

    const getID = struct {
        fn f(item: Item) i32 {
            return item.id;
        }
    }.f;

    var grouped = try groupBy(i32, Item, allocator, &data, getID);
    defer grouped.deinit();

    try std.testing.expectEqual(@as(usize, 3), grouped.keys.len);

    // Verify groups
    for (grouped.keys, grouped.groups) |key, group| {
        if (key == 1) {
            try std.testing.expectEqual(@as(usize, 2), group.len);
        } else if (key == 2) {
            try std.testing.expectEqual(@as(usize, 2), group.len);
        } else if (key == 3) {
            try std.testing.expectEqual(@as(usize, 1), group.len);
        }
    }
}

test "group by - empty input" {
    const allocator = std.testing.allocator;

    const Item = struct { id: i32, value: i32 };
    const data = [_]Item{};

    const getID = struct {
        fn f(item: Item) i32 {
            return item.id;
        }
    }.f;

    var grouped = try groupBy(i32, Item, allocator, &data, getID);
    defer grouped.deinit();

    try std.testing.expectEqual(@as(usize, 0), grouped.keys.len);
}

test "partition - even/odd" {
    const allocator = std.testing.allocator;

    const isEven = struct {
        fn f(x: i32) bool {
            return @mod(x, 2) == 0;
        }
    }.f;

    const arr = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var result = try partition(i32, allocator, &arr, isEven);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 4), result.true_partition.len);
    try std.testing.expectEqual(@as(usize, 4), result.false_partition.len);

    // Verify even numbers
    for (result.true_partition) |val| {
        try std.testing.expect(@mod(val, 2) == 0);
    }

    // Verify odd numbers
    for (result.false_partition) |val| {
        try std.testing.expect(@mod(val, 2) == 1);
    }
}

test "partition - all true" {
    const allocator = std.testing.allocator;

    const alwaysTrue = struct {
        fn f(_: i32) bool {
            return true;
        }
    }.f;

    const arr = [_]i32{ 1, 2, 3 };
    var result = try partition(i32, allocator, &arr, alwaysTrue);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 3), result.true_partition.len);
    try std.testing.expectEqual(@as(usize, 0), result.false_partition.len);
}

test "partition - all false" {
    const allocator = std.testing.allocator;

    const alwaysFalse = struct {
        fn f(_: i32) bool {
            return false;
        }
    }.f;

    const arr = [_]i32{ 1, 2, 3 };
    var result = try partition(i32, allocator, &arr, alwaysFalse);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 0), result.true_partition.len);
    try std.testing.expectEqual(@as(usize, 3), result.false_partition.len);
}
