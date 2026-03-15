//! Collect Iterator Utility
//!
//! The Collect utility exhausts an iterator and collects all yielded elements
//! into an ArrayList(T). This is the terminal operation that realizes a lazy
//! iterator chain into a concrete collection.
//!
//! ## Type Parameters
//! - T: The element type collected from the iterator
//! - BaseIter: The concrete iterator type (must have next() -> ?T)
//!
//! ## Time Complexity
//! - O(n) where n is the number of elements in the iterator
//!   (plus the cost of growing ArrayList dynamically)
//!
//! ## Space Complexity
//! - O(n) — allocates ArrayList to store all n elements
//!
//! ## Example
//! ```zig
//! const allocator = std.testing.allocator;
//! var numbers = [_]i32{1, 2, 3};
//! var slice_iter = SliceIterator(i32).init(&numbers);
//! var result = try collect(i32, allocator, &slice_iter);
//! defer result.deinit();
//! // result is ArrayList(i32) with [1, 2, 3]
//! ```

const std = @import("std");
const testing = std.testing;
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;

/// Collect consumes an iterator and returns an ArrayList of all collected elements.
/// The returned ArrayList must be deinitialized by the caller using result.deinit(allocator).
/// Takes a pointer to an iterator to allow mutation of iterator state.
///
/// Time: O(n) where n = number of elements | Space: O(n)
pub fn collect(comptime T: type, allocator: Allocator, iter: anytype) !ArrayList(T) {
    var result: ArrayList(T) = .{};
    errdefer result.deinit(allocator);

    while (iter.next()) |value| {
        try result.append(allocator, value);
    }

    return result;
}

// -- Tests --

/// Simple slice iterator for testing (not part of public API)
fn SliceIterator(comptime T: type) type {
    return struct {
        const SelfIter = @This();

        slice: []const T,
        index: usize = 0,

        fn init(slice: []const T) SelfIter {
            return .{ .slice = slice };
        }

        fn next(self: *SelfIter) ?T {
            if (self.index >= self.slice.len) return null;
            defer self.index += 1;
            return self.slice[self.index];
        }
    };
}

// Map adaptor for testing combined adaptors
fn MapIterator(comptime T: type, comptime U: type, comptime BaseIter: type) type {
    return struct {
        const Self = @This();

        base_iter: BaseIter,
        transform_fn: *const fn (T) U,

        fn init(base_iter: BaseIter, transform_fn: *const fn (T) U) Self {
            return .{
                .base_iter = base_iter,
                .transform_fn = transform_fn,
            };
        }

        fn next(self: *Self) ?U {
            const value = self.base_iter.next() orelse return null;
            return self.transform_fn(value);
        }
    };
}

// Filter adaptor for testing combined adaptors
fn FilterIterator(comptime T: type, comptime BaseIter: type) type {
    return struct {
        const Self = @This();

        base_iter: BaseIter,
        predicate_fn: *const fn (T) bool,

        fn init(base_iter: BaseIter, predicate_fn: *const fn (T) bool) Self {
            return .{
                .base_iter = base_iter,
                .predicate_fn = predicate_fn,
            };
        }

        fn next(self: *Self) ?T {
            while (self.base_iter.next()) |value| {
                if (self.predicate_fn(value)) {
                    return value;
                }
            }
            return null;
        }
    };
}

// Take adaptor for testing combined adaptors
fn TakeIterator(comptime T: type, comptime BaseIter: type) type {
    return struct {
        const Self = @This();

        base_iter: BaseIter,
        remaining: usize,

        fn init(base_iter: BaseIter, count: usize) Self {
            return .{
                .base_iter = base_iter,
                .remaining = count,
            };
        }

        fn next(self: *Self) ?T {
            if (self.remaining == 0) return null;
            defer self.remaining -= 1;
            return self.base_iter.next();
        }
    };
}

// Skip adaptor for testing combined adaptors
fn SkipIterator(comptime T: type, comptime BaseIter: type) type {
    return struct {
        const Self = @This();

        base_iter: BaseIter,
        skipped: bool = false,

        fn init(base_iter: BaseIter, count: usize) Self {
            var self: Self = .{
                .base_iter = base_iter,
                .skipped = false,
            };
            // Skip count elements during initialization
            for (0..count) |_| {
                _ = self.base_iter.next();
            }
            return self;
        }

        fn next(self: *Self) ?T {
            return self.base_iter.next();
        }
    };
}

// Chain adaptor for testing combined adaptors
fn ChainIterator(comptime T: type, comptime BaseIter: type) type {
    return struct {
        const Self = @This();

        first: BaseIter,
        second: BaseIter,
        using_first: bool = true,

        fn init(first: BaseIter, second: BaseIter) Self {
            return .{
                .first = first,
                .second = second,
                .using_first = true,
            };
        }

        fn next(self: *Self) ?T {
            if (self.using_first) {
                if (self.first.next()) |value| {
                    return value;
                }
                self.using_first = false;
            }
            return self.second.next();
        }
    };
}

// Enumerate adaptor for testing Pair collection
fn EnumerateIterator(comptime T: type, comptime BaseIter: type) type {
    return struct {
        const Self = @This();

        pub const Pair = struct {
            index: usize,
            value: T,
        };

        base_iter: BaseIter,
        index: usize = 0,

        fn init(base_iter: BaseIter) Self {
            return .{
                .base_iter = base_iter,
                .index = 0,
            };
        }

        fn next(self: *Self) ?Pair {
            const value = self.base_iter.next() orelse return null;
            defer self.index += 1;
            return .{
                .index = self.index,
                .value = value,
            };
        }
    };
}

// Helper functions for transformations and predicates
fn double(x: i32) i32 {
    return x * 2;
}

fn isEven(x: i32) bool {
    return @rem(x, 2) == 0;
}

fn intToFloat(x: i32) f32 {
    return @floatFromInt(x);
}

test "collect: basic collect from small iterator to ArrayList" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 1, 2, 3 };
    const slice_iter = SliceIterator(i32).init(&numbers);

    var result = try collect(i32, allocator, &slice_iter);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 3), result.items.len);
    try testing.expectEqual(1, result.items[0]);
    try testing.expectEqual(2, result.items[1]);
    try testing.expectEqual(3, result.items[2]);
}

test "collect: empty iterator returns empty ArrayList" {
    const allocator = testing.allocator;
    var numbers = [_]i32{};
    const slice_iter = SliceIterator(i32).init(&numbers);

    var result = try collect(i32, allocator, &slice_iter);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), result.items.len);
}

test "collect: single element iterator" {
    const allocator = testing.allocator;
    var numbers = [_]i32{42};
    const slice_iter = SliceIterator(i32).init(&numbers);

    var result = try collect(i32, allocator, &slice_iter);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 1), result.items.len);
    try testing.expectEqual(42, result.items[0]);
}

test "collect: large dataset (1000 elements)" {
    const allocator = testing.allocator;
    var numbers: [1000]i32 = undefined;
    for (0..1000) |i| {
        numbers[i] = @intCast(i);
    }

    const slice_iter = SliceIterator(i32).init(&numbers);
    var result = try collect(i32, allocator, &slice_iter);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 1000), result.items.len);
    try testing.expectEqual(0, result.items[0]);
    try testing.expectEqual(500, result.items[500]);
    try testing.expectEqual(999, result.items[999]);
}

test "collect: after filter adaptor" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 1, 2, 3, 4, 5, 6 };
    const slice_iter = SliceIterator(i32).init(&numbers);
    const filter = FilterIterator(i32, SliceIterator(i32)).init(slice_iter, isEven);

    var result = try collect(i32, allocator, &filter);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 3), result.items.len);
    try testing.expectEqual(2, result.items[0]);
    try testing.expectEqual(4, result.items[1]);
    try testing.expectEqual(6, result.items[2]);
}

test "collect: after map adaptor" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 1, 2, 3 };
    const slice_iter = SliceIterator(i32).init(&numbers);
    const map = MapIterator(i32, i32, SliceIterator(i32)).init(slice_iter, double);

    var result = try collect(i32, allocator, &map);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 3), result.items.len);
    try testing.expectEqual(2, result.items[0]);
    try testing.expectEqual(4, result.items[1]);
    try testing.expectEqual(6, result.items[2]);
}

test "collect: after chain adaptor" {
    const allocator = testing.allocator;
    var numbers1 = [_]i32{ 1, 2 };
    var numbers2 = [_]i32{ 3, 4 };
    const iter1 = SliceIterator(i32).init(&numbers1);
    const iter2 = SliceIterator(i32).init(&numbers2);
    const chain = ChainIterator(i32, SliceIterator(i32)).init(iter1, iter2);

    var result = try collect(i32, allocator, &chain);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 4), result.items.len);
    try testing.expectEqual(1, result.items[0]);
    try testing.expectEqual(2, result.items[1]);
    try testing.expectEqual(3, result.items[2]);
    try testing.expectEqual(4, result.items[3]);
}

test "collect: after take adaptor" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 1, 2, 3, 4, 5 };
    const slice_iter = SliceIterator(i32).init(&numbers);
    const take = TakeIterator(i32, SliceIterator(i32)).init(slice_iter, 3);

    var result = try collect(i32, allocator, &take);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 3), result.items.len);
    try testing.expectEqual(1, result.items[0]);
    try testing.expectEqual(2, result.items[1]);
    try testing.expectEqual(3, result.items[2]);
}

test "collect: after skip adaptor" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 1, 2, 3, 4, 5 };
    const slice_iter = SliceIterator(i32).init(&numbers);
    const skip = SkipIterator(i32, SliceIterator(i32)).init(slice_iter, 2);

    var result = try collect(i32, allocator, &skip);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 3), result.items.len);
    try testing.expectEqual(3, result.items[0]);
    try testing.expectEqual(4, result.items[1]);
    try testing.expectEqual(5, result.items[2]);
}

test "collect: enumerate pairs (collects struct)" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 10, 20, 30 };
    const slice_iter = SliceIterator(i32).init(&numbers);
    const enum_iter = EnumerateIterator(i32, SliceIterator(i32)).init(slice_iter);

    const PairType = EnumerateIterator(i32, SliceIterator(i32)).Pair;
    var result = try collect(PairType, allocator, &enum_iter);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 3), result.items.len);
    try testing.expectEqual(@as(usize, 0), result.items[0].index);
    try testing.expectEqual(10, result.items[0].value);
    try testing.expectEqual(@as(usize, 1), result.items[1].index);
    try testing.expectEqual(20, result.items[1].value);
    try testing.expectEqual(@as(usize, 2), result.items[2].index);
    try testing.expectEqual(30, result.items[2].value);
}

test "collect: f32 type collection" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 1, 2, 3 };
    const slice_iter = SliceIterator(i32).init(&numbers);
    const map = MapIterator(i32, f32, SliceIterator(i32)).init(slice_iter, intToFloat);

    var result = try collect(f32, allocator, &map);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 3), result.items.len);
    try testing.expectEqual(1.0, result.items[0]);
    try testing.expectEqual(2.0, result.items[1]);
    try testing.expectEqual(3.0, result.items[2]);
}

test "collect: bool type collection" {
    const allocator = testing.allocator;
    var bools = [_]bool{ true, false, true, true };
    const slice_iter = SliceIterator(bool).init(&bools);

    var result = try collect(bool, allocator, &slice_iter);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 4), result.items.len);
    try testing.expectEqual(true, result.items[0]);
    try testing.expectEqual(false, result.items[1]);
    try testing.expectEqual(true, result.items[2]);
    try testing.expectEqual(true, result.items[3]);
}

test "collect: struct type collection" {
    const allocator = testing.allocator;
    const Point = struct { x: i32, y: i32 };

    var points = [_]Point{
        .{ .x = 1, .y = 2 },
        .{ .x = 3, .y = 4 },
        .{ .x = 5, .y = 6 },
    };
    const slice_iter = SliceIterator(Point).init(&points);

    var result = try collect(Point, allocator, &slice_iter);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 3), result.items.len);
    try testing.expectEqual(1, result.items[0].x);
    try testing.expectEqual(2, result.items[0].y);
    try testing.expectEqual(3, result.items[1].x);
    try testing.expectEqual(4, result.items[1].y);
    try testing.expectEqual(5, result.items[2].x);
    try testing.expectEqual(6, result.items[2].y);
}

test "collect: deinit required - no memory leaks with std.testing.allocator" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 1, 2, 3, 4, 5 };
    const slice_iter = SliceIterator(i32).init(&numbers);

    var result = try collect(i32, allocator, &slice_iter);
    // Allocator automatically detects leaks if deinit is not called properly
    result.deinit(allocator);
    // This test passes if no memory leak is detected
}

test "collect: boundary - take zero elements" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 1, 2, 3 };
    const slice_iter = SliceIterator(i32).init(&numbers);
    const take = TakeIterator(i32, SliceIterator(i32)).init(slice_iter, 0);

    var result = try collect(i32, allocator, &take);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), result.items.len);
}

test "collect: boundary - skip all elements" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 1, 2, 3 };
    const slice_iter = SliceIterator(i32).init(&numbers);
    const skip = SkipIterator(i32, SliceIterator(i32)).init(slice_iter, 3);

    var result = try collect(i32, allocator, &skip);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), result.items.len);
}

test "collect: combined adaptors - map then filter then take" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 1, 2, 3, 4, 5 };
    const slice_iter = SliceIterator(i32).init(&numbers);
    const map = MapIterator(i32, i32, SliceIterator(i32)).init(slice_iter, double);
    const filter = FilterIterator(i32, MapIterator(i32, i32, SliceIterator(i32))).init(map, isEven);
    const take = TakeIterator(i32, FilterIterator(i32, MapIterator(i32, i32, SliceIterator(i32)))).init(filter, 2);

    var result = try collect(i32, allocator, &take);
    defer result.deinit(allocator);

    // map: [2, 4, 6, 8, 10]
    // filter (even): [2, 4, 6, 8, 10] (all even)
    // take(2): [2, 4]
    try testing.expectEqual(@as(usize, 2), result.items.len);
    try testing.expectEqual(2, result.items[0]);
    try testing.expectEqual(4, result.items[1]);
}

test "collect: u8 type collection" {
    const allocator = testing.allocator;
    var bytes = [_]u8{ 10, 20, 30, 40, 50 };
    const slice_iter = SliceIterator(u8).init(&bytes);

    var result = try collect(u8, allocator, &slice_iter);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 5), result.items.len);
    try testing.expectEqual(10, result.items[0]);
    try testing.expectEqual(50, result.items[4]);
}

test "collect: ArrayList returned is properly initialized" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 1, 2 };
    const slice_iter = SliceIterator(i32).init(&numbers);

    var result = try collect(i32, allocator, &slice_iter);
    defer result.deinit(allocator);

    // Verify ArrayList properties
    try testing.expect(result.items.len == 2);
    try testing.expect(result.capacity >= 2);
}
