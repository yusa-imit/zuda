//! Sequential map-reduce, group-by and partition over caller-owned slices.
//!
//! Nothing here spawns threads. Every function takes the caller's `gpa`, allocates its scratch and
//! result before emitting or copying (`mapReduce` reserves `pairs_max` pairs up front), and returns
//! results the caller frees with `deinit(gpa)` using the same allocator; result types and the
//! emitter do not store an allocator. A failed allocation leaves nothing behind. Slice keys
//! borrow from the input: a result is valid only while the input it was built from lives.
//!
//! Keys are ordered by a comptime `key_order` function, so slice keys (`[]const u8`) work without
//! `std.meta.eql`. Sorting is stable: values of one key reach the reducer, and a group, in input
//! order. The number of emitted pairs is bounded by the caller's `pairs_max`.
//!
//! Cost sketch for n emitted pairs: one `pairs_max`-pair buffer, one n-value buffer (so the reducer
//! sees a contiguous `[]const V`), an O(n log n) comparison sort, no syscalls.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Order = std.math.Order;
const assert = std.debug.assert;

pub const Error = Allocator.Error || error{PairsLimitExceeded};

/// Key/value pair produced by a mapper.
pub fn PairType(comptime K: type, comptime V: type) type {
    return struct {
        key: K,
        value: V,
    };
}

/// Collects the pairs a mapper emits. Created by `mapReduce`; mappers only call `emit`.
pub fn EmitterType(comptime K: type, comptime V: type) type {
    return struct {
        const Emitter = @This();
        const Pair = PairType(K, V);

        pairs: std.ArrayList(Pair),
        pairs_max: u32,

        /// Appends one pair. Returns `error.PairsLimitExceeded` once `pairs_max` pairs exist.
        /// Never allocates: `mapReduce` reserved `pairs_max` pairs before the first mapper call.
        pub fn emit(emitter: *Emitter, key: K, value: V) Error!void {
            assert(emitter.pairs.capacity >= emitter.pairs_max);
            assert(emitter.pairs.items.len <= emitter.pairs_max);
            if (emitter.pairs.items.len == emitter.pairs_max) return error.PairsLimitExceeded;
            emitter.pairs.appendAssumeCapacity(.{ .key = key, .value = value });
        }
    };
}

/// Result of `mapReduce`: `keys` ascending and unique, `values[i]` reduced from `keys[i]`.
pub fn MapReduceResult(comptime K: type, comptime V: type) type {
    return struct {
        const Result = @This();

        keys: []K,
        values: []V,

        /// Frees the result. `gpa` must be the allocator passed to `mapReduce`.
        pub fn deinit(result: *Result, gpa: Allocator) void {
            assert(result.keys.len == result.values.len);
            gpa.free(result.keys);
            gpa.free(result.values);
            result.* = undefined;
        }
    };
}

pub const MapReduceOptions = struct {
    /// Upper bound on pairs emitted over the whole input.
    pairs_max: u32,
};

/// Maps every input item to key/value pairs, groups the pairs by key, reduces each group.
///
/// `mapper(item, emitter)` calls `emitter.emit(key, value)` zero or more times. `reducer(key,
/// values)` receives all values of one key in emission order and must not retain the slice.
/// `key_order` must be a strict weak ordering; keys that compare `.eq` form one group, and the
/// first emitted key of a group is the one reported.
///
/// Time: O(n log n) in the emitted pair count n | Space: O(n)
pub fn mapReduce(
    comptime In: type,
    comptime K: type,
    comptime V: type,
    gpa: Allocator,
    input: []const In,
    comptime mapper: fn (In, *EmitterType(K, V)) Error!void,
    comptime reducer: fn (K, []const V) V,
    comptime key_order: fn (K, K) Order,
    options: MapReduceOptions,
) Error!MapReduceResult(K, V) {
    const Pair = PairType(K, V);
    var emitter: EmitterType(K, V) = .{
        .pairs = .empty,
        .pairs_max = options.pairs_max,
    };
    defer emitter.pairs.deinit(gpa);

    try emitter.pairs.ensureTotalCapacityPrecise(gpa, options.pairs_max);

    for (input) |item| try mapper(item, &emitter);
    const pairs = emitter.pairs.items;
    assert(pairs.len <= options.pairs_max);

    std.sort.block(Pair, pairs, {}, PairOrderType(K, V, key_order).less_than);

    const values = try gpa.alloc(V, pairs.len);
    defer gpa.free(values);

    for (pairs, 0..) |pair, index| values[index] = pair.value;
    const group_count = count_groups(K, V, pairs, key_order);

    const keys_out = try gpa.alloc(K, group_count);
    errdefer gpa.free(keys_out);

    const values_out = try gpa.alloc(V, group_count);
    errdefer gpa.free(values_out);

    var group_index: usize = 0;
    var run_start: usize = 0;
    while (run_start < pairs.len) : (group_index += 1) {
        const run_end = run_end_of(K, V, pairs, run_start, key_order);
        keys_out[group_index] = pairs[run_start].key;
        values_out[group_index] = reducer(pairs[run_start].key, values[run_start..run_end]);
        run_start = run_end;
    }
    assert(group_index == group_count);

    return .{ .keys = keys_out, .values = values_out };
}

/// Result of `groupBy`: `keys` ascending and unique, `groups[i]` holds the items of `keys[i]`.
pub fn GroupByResult(comptime K: type, comptime V: type) type {
    return struct {
        const Result = @This();

        keys: []K,
        groups: [][]V,
        /// One buffer backs every group; kept so `deinit` never rebuilds it from `groups`.
        flat: []V,

        /// Frees the result. `gpa` must be the allocator passed to `groupBy`.
        pub fn deinit(result: *Result, gpa: Allocator) void {
            assert(result.keys.len == result.groups.len);
            var total: usize = 0;
            for (result.groups) |group| total += group.len;
            assert(total == result.flat.len);
            gpa.free(result.flat);
            gpa.free(result.groups);
            gpa.free(result.keys);
            // Poison the freed result so a use after deinit fails loudly in safe builds.
            result.* = undefined;
        }
    };
}

/// Groups `input` by `key_fn`. Items of one group keep their input order.
///
/// Time: O(n log n) | Space: O(n)
pub fn groupBy(
    comptime K: type,
    comptime V: type,
    gpa: Allocator,
    input: []const V,
    comptime key_fn: fn (V) K,
    comptime key_order: fn (K, K) Order,
) Allocator.Error!GroupByResult(K, V) {
    const Pair = PairType(K, V);
    const pairs = try gpa.alloc(Pair, input.len);
    defer gpa.free(pairs);

    for (input, pairs) |item, *pair| pair.* = .{ .key = key_fn(item), .value = item };
    std.sort.block(Pair, pairs, {}, PairOrderType(K, V, key_order).less_than);

    assert(pairs.len == input.len);
    const group_count = count_groups(K, V, pairs, key_order);

    const keys_out = try gpa.alloc(K, group_count);
    errdefer gpa.free(keys_out);

    const groups_out = try gpa.alloc([]V, group_count);
    errdefer gpa.free(groups_out);

    const flat = try gpa.alloc(V, pairs.len);
    errdefer gpa.free(flat);
    assert(flat.len == input.len);

    var group_index: usize = 0;
    var run_start: usize = 0;
    while (run_start < pairs.len) : (group_index += 1) {
        const run_end = run_end_of(K, V, pairs, run_start, key_order);
        keys_out[group_index] = pairs[run_start].key;
        for (pairs[run_start..run_end], flat[run_start..run_end]) |pair, *slot| slot.* = pair.value;
        groups_out[group_index] = flat[run_start..run_end];
        run_start = run_end;
    }
    assert(group_index == group_count);

    return .{ .keys = keys_out, .groups = groups_out, .flat = flat };
}

/// Result of `partition`: both halves keep input order.
pub fn PartitionResult(comptime T: type) type {
    return struct {
        const Result = @This();

        true_partition: []T,
        false_partition: []T,

        /// Frees the result. `gpa` must be the allocator passed to `partition`.
        pub fn deinit(result: *Result, gpa: Allocator) void {
            gpa.free(result.true_partition);
            gpa.free(result.false_partition);
            result.* = undefined;
        }
    };
}

/// Splits `arr` into the items satisfying `pred` and the rest, each in input order.
/// `pred` must be pure: it runs twice per item (count pass, fill pass).
///
/// Time: O(n) | Space: O(n)
pub fn partition(
    comptime T: type,
    gpa: Allocator,
    arr: []const T,
    comptime pred: fn (T) bool,
) Allocator.Error!PartitionResult(T) {
    var true_count: usize = 0;
    for (arr) |item| {
        if (pred(item)) true_count += 1;
    }

    const true_out = try gpa.alloc(T, true_count);
    errdefer gpa.free(true_out);

    const false_out = try gpa.alloc(T, arr.len - true_count);
    errdefer gpa.free(false_out);

    var true_index: usize = 0;
    var false_index: usize = 0;
    for (arr) |item| {
        if (pred(item)) {
            true_out[true_index] = item;
            true_index += 1;
        } else {
            false_out[false_index] = item;
            false_index += 1;
        }
    }
    assert(true_index == true_count);
    assert(false_index == false_out.len);

    return .{ .true_partition = true_out, .false_partition = false_out };
}

fn PairOrderType(comptime K: type, comptime V: type, comptime key_order: fn (K, K) Order) type {
    return struct {
        fn less_than(_: void, a: PairType(K, V), b: PairType(K, V)) bool {
            return key_order(a.key, b.key) == .lt;
        }
    };
}

/// Index one past the run of pairs whose key equals `pairs[start].key`. Requires sorted pairs.
fn run_end_of(
    comptime K: type,
    comptime V: type,
    pairs: []const PairType(K, V),
    start: usize,
    comptime key_order: fn (K, K) Order,
) usize {
    assert(start < pairs.len);
    var end = start + 1;
    while (end < pairs.len and key_order(pairs[start].key, pairs[end].key) == .eq) end += 1;
    assert(end <= pairs.len);
    // Sorted input: the key after the run is strictly greater (negative space of the run).
    if (end < pairs.len) assert(key_order(pairs[start].key, pairs[end].key) == .lt);
    return end;
}

fn count_groups(
    comptime K: type,
    comptime V: type,
    pairs: []const PairType(K, V),
    comptime key_order: fn (K, K) Order,
) usize {
    var group_count: usize = 0;
    var run_start: usize = 0;
    while (run_start < pairs.len) : (group_count += 1) {
        run_start = run_end_of(K, V, pairs, run_start, key_order);
    }
    assert(group_count <= pairs.len);
    return group_count;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

fn order_i32(a: i32, b: i32) Order {
    return std.math.order(a, b);
}

fn order_text(a: []const u8, b: []const u8) Order {
    return std.mem.order(u8, a, b);
}

fn sum_i32(_: []const u8, counts: []const i32) i32 {
    var sum: i32 = 0;
    for (counts) |count| sum += count;
    return sum;
}

fn word_mapper(line: []const u8, emitter: *EmitterType([]const u8, i32)) Error!void {
    var words = std.mem.tokenizeScalar(u8, line, ' ');
    while (words.next()) |word| try emitter.emit(word, 1);
}

const WordCounts = MapReduceResult([]const u8, i32);

fn word_count(gpa: Allocator, lines: []const []const u8, pairs_max: u32) Error!WordCounts {
    return mapReduce([]const u8, []const u8, i32, gpa, lines, word_mapper, sum_i32, order_text, .{
        .pairs_max = pairs_max,
    });
}

test "mapReduce: word count over slice keys" {
    const lines = [_][]const u8{ "the cat", "the dog the end", "", "cat" };
    var result = try word_count(testing.allocator, &lines, 100);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 4), result.keys.len);
    const expected_keys = [_][]const u8{ "cat", "dog", "end", "the" };
    const expected_counts = [_]i32{ 2, 1, 1, 3 };
    for (expected_keys, expected_counts, result.keys, result.values) |key, count, got_key, got| {
        try testing.expectEqualStrings(key, got_key);
        try testing.expectEqual(count, got);
    }
}

test "mapReduce: empty input and mappers that emit nothing give an empty result" {
    var empty = try word_count(testing.allocator, &.{}, 10);
    defer empty.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), empty.keys.len);
    try testing.expectEqual(@as(usize, 0), empty.values.len);

    const blank = [_][]const u8{ "", "   " };
    var none = try word_count(testing.allocator, &blank, 10);
    defer none.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), none.keys.len);
}

test "mapReduce: one key for every pair reduces to a single group" {
    const lines = [_][]const u8{ "a a a", "a a" };
    var result = try word_count(testing.allocator, &lines, 5);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), result.keys.len);
    try testing.expectEqual(@as(i32, 5), result.values[0]);
}

test "mapReduce: pairs_max is a hard bound (max passes, max+1 fails)" {
    const lines = [_][]const u8{ "a b", "c" };
    var at_max = try word_count(testing.allocator, &lines, 3);
    at_max.deinit(testing.allocator);

    try testing.expectError(error.PairsLimitExceeded, word_count(testing.allocator, &lines, 2));
    try testing.expectError(error.PairsLimitExceeded, word_count(testing.allocator, &lines, 0));
}

fn count_mapper(item: i32, emitter: *EmitterType(i32, i32)) Error!void {
    try emitter.emit(item, 1);
}

fn count_reducer(_: i32, values: []const i32) i32 {
    return @intCast(values.len);
}

fn first_reducer(_: i32, values: []const i32) i32 {
    return values[0];
}

fn mod3_mapper(item: i32, emitter: *EmitterType(i32, i32)) Error!void {
    try emitter.emit(@mod(item, 3), item);
}

test "mapReduce: values reach the reducer in emission order" {
    const input = [_]i32{ 4, 7, 1, 5, 2, 3 };
    const options: MapReduceOptions = .{ .pairs_max = 6 };
    var result = try mapReduce(
        i32,
        i32,
        i32,
        testing.allocator,
        &input,
        mod3_mapper,
        first_reducer,
        order_i32,
        options,
    );
    defer result.deinit(testing.allocator);

    try testing.expectEqualSlices(i32, &.{ 0, 1, 2 }, result.keys);
    try testing.expectEqualSlices(i32, &.{ 3, 4, 5 }, result.values);
}

test "mapReduce: allocation failures leak nothing" {
    const Runner = struct {
        fn run(gpa: Allocator) !void {
            const lines = [_][]const u8{ "x y x", "z y x" };
            var result = try word_count(gpa, &lines, 16);
            defer result.deinit(gpa);
            try testing.expectEqual(@as(usize, 3), result.keys.len);
        }
    };
    try testing.checkAllAllocationFailures(testing.allocator, Runner.run, .{});
}

test "mapReduce: seeded model against a reference count" {
    var prng = std.Random.DefaultPrng.init(0x5eed_0001);
    const random = prng.random();
    for (0..50) |_| {
        var input: [40]i32 = undefined;
        for (&input) |*item| item.* = random.intRangeAtMost(i32, 0, 9);

        const options: MapReduceOptions = .{ .pairs_max = 40 };
        var result = try mapReduce(
            i32,
            i32,
            i32,
            testing.allocator,
            &input,
            count_mapper,
            count_reducer,
            order_i32,
            options,
        );
        defer result.deinit(testing.allocator);

        var reference = [_]i32{0} ** 10;
        for (input) |item| reference[@intCast(item)] += 1;
        var group_index: usize = 0;
        for (reference, 0..) |count, key| {
            if (count == 0) continue;
            try testing.expectEqual(@as(i32, @intCast(key)), result.keys[group_index]);
            try testing.expectEqual(count, result.values[group_index]);
            group_index += 1;
        }
        try testing.expectEqual(group_index, result.keys.len);
    }
}

const Record = struct { id: i32, tag: u8 };

fn record_id(record: Record) i32 {
    return record.id;
}

test "groupBy: groups ascend by key and keep input order inside a group" {
    const data = [_]Record{
        .{ .id = 2, .tag = 'a' },
        .{ .id = 1, .tag = 'b' },
        .{ .id = 2, .tag = 'c' },
        .{ .id = 1, .tag = 'd' },
        .{ .id = 3, .tag = 'e' },
    };
    var result = try groupBy(i32, Record, testing.allocator, &data, record_id, order_i32);
    defer result.deinit(testing.allocator);

    try testing.expectEqualSlices(i32, &.{ 1, 2, 3 }, result.keys);
    try testing.expectEqual(@as(usize, 2), result.groups[0].len);
    try testing.expectEqual(@as(u8, 'b'), result.groups[0][0].tag);
    try testing.expectEqual(@as(u8, 'd'), result.groups[0][1].tag);
    try testing.expectEqual(@as(u8, 'a'), result.groups[1][0].tag);
    try testing.expectEqual(@as(u8, 'c'), result.groups[1][1].tag);
    try testing.expectEqual(@as(usize, 1), result.groups[2].len);
}

test "groupBy: empty input and a single element" {
    var empty = try groupBy(i32, Record, testing.allocator, &.{}, record_id, order_i32);
    defer empty.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), empty.keys.len);
    try testing.expectEqual(@as(usize, 0), empty.groups.len);

    const one = [_]Record{.{ .id = 7, .tag = 'x' }};
    var single = try groupBy(i32, Record, testing.allocator, &one, record_id, order_i32);
    defer single.deinit(testing.allocator);
    try testing.expectEqualSlices(i32, &.{7}, single.keys);
    try testing.expectEqual(@as(usize, 1), single.groups[0].len);
}

test "groupBy: allocation failures leak nothing" {
    const Runner = struct {
        fn run(gpa: Allocator) !void {
            const data = [_]Record{
                .{ .id = 2, .tag = 'a' },
                .{ .id = 1, .tag = 'b' },
                .{ .id = 2, .tag = 'c' },
            };
            var result = try groupBy(i32, Record, gpa, &data, record_id, order_i32);
            defer result.deinit(gpa);
            try testing.expectEqual(@as(usize, 2), result.keys.len);
        }
    };
    try testing.checkAllAllocationFailures(testing.allocator, Runner.run, .{});
}

fn is_even(value: i32) bool {
    return @mod(value, 2) == 0;
}

test "partition: splits by predicate and keeps input order" {
    const arr = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var result = try partition(i32, testing.allocator, &arr, is_even);
    defer result.deinit(testing.allocator);

    try testing.expectEqualSlices(i32, &.{ 2, 4, 6 }, result.true_partition);
    try testing.expectEqualSlices(i32, &.{ 1, 3, 5 }, result.false_partition);
}

test "partition: empty, all true, all false" {
    var empty = try partition(i32, testing.allocator, &.{}, is_even);
    defer empty.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), empty.true_partition.len);
    try testing.expectEqual(@as(usize, 0), empty.false_partition.len);

    var evens = try partition(i32, testing.allocator, &[_]i32{ 2, 4 }, is_even);
    defer evens.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), evens.true_partition.len);
    try testing.expectEqual(@as(usize, 0), evens.false_partition.len);

    var odds = try partition(i32, testing.allocator, &[_]i32{3}, is_even);
    defer odds.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), odds.true_partition.len);
    try testing.expectEqual(@as(usize, 1), odds.false_partition.len);
}

test "partition: allocation failures leak nothing" {
    const Runner = struct {
        fn run(gpa: Allocator) !void {
            var result = try partition(i32, gpa, &[_]i32{ 1, 2, 3, 4 }, is_even);
            defer result.deinit(gpa);
            try testing.expectEqual(@as(usize, 2), result.true_partition.len);
        }
    };
    try testing.checkAllAllocationFailures(testing.allocator, Runner.run, .{});
}

fn ascending_reducer(_: i32, values: []const i32) i32 {
    for (values[1..], values[0 .. values.len - 1]) |next, prev| {
        if (next <= prev) return 0;
    }
    return 1;
}

fn indexed_mapper(item: i32, emitter: *EmitterType(i32, i32)) Error!void {
    try emitter.emit(@mod(item, 4), item);
}

test "mapReduce: stability holds at sizes beyond insertion sort" {
    var input: [600]i32 = undefined;
    for (&input, 0..) |*item, index| item.* = @intCast(index);

    const options: MapReduceOptions = .{ .pairs_max = 600 };
    var result = try mapReduce(
        i32,
        i32,
        i32,
        testing.allocator,
        &input,
        indexed_mapper,
        ascending_reducer,
        order_i32,
        options,
    );
    defer result.deinit(testing.allocator);

    try testing.expectEqualSlices(i32, &.{ 1, 1, 1, 1 }, result.values);
}

fn order_fold_case(a: []const u8, b: []const u8) Order {
    var index: usize = 0;
    while (index < a.len and index < b.len) : (index += 1) {
        const order = std.math.order(std.ascii.toLower(a[index]), std.ascii.toLower(b[index]));
        if (order != .eq) return order;
    }
    return std.math.order(a.len, b.len);
}

test "mapReduce: keys that compare equal form one group named by the first emitted" {
    const lines = [_][]const u8{ "Zed a A b", "B" };
    var result = try mapReduce(
        []const u8,
        []const u8,
        i32,
        testing.allocator,
        &lines,
        word_mapper,
        sum_i32,
        order_fold_case,
        .{ .pairs_max = 8 },
    );
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 3), result.keys.len);
    try testing.expectEqualStrings("a", result.keys[0]);
    try testing.expectEqual(@as(i32, 2), result.values[0]);
    try testing.expectEqualStrings("b", result.keys[1]);
    try testing.expectEqual(@as(i32, 2), result.values[1]);
}

test "groupBy: stability holds at sizes beyond insertion sort" {
    var data: [600]Record = undefined;
    for (&data, 0..) |*record, index| {
        record.* = .{ .id = @intCast(index % 4), .tag = @intCast(index % 251) };
    }
    var result = try groupBy(i32, Record, testing.allocator, &data, record_id, order_i32);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 4), result.groups.len);
    for (result.groups, 0..) |group, key| {
        try testing.expectEqual(@as(usize, 150), group.len);
        for (group, 0..) |record, position| {
            try testing.expectEqual(@as(i32, @intCast(key)), record.id);
            const original = key + position * 4;
            try testing.expectEqual(@as(u8, @intCast(original % 251)), record.tag);
        }
    }
}
