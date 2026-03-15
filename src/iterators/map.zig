//! Map Iterator Adaptor
//!
//! The Map adaptor transforms each element yielded by a base iterator using
//! a provided transform function. It lazily applies the transformation during
//! iteration, making it efficient for large datasets.
//!
//! ## Type Parameters
//! - T: Input element type
//! - U: Output element type (result of transform function)
//!
//! ## Time Complexity
//! - next(): O(1) + O(cost of transform function)
//!
//! ## Space Complexity
//! - O(1) — no allocation; stores only base iterator and function pointer
//!
//! ## Example
//! ```zig
//! var numbers = [_]i32{1, 2, 3};
//! var slice_iter = SliceIterator(i32).init(&numbers);
//! var map = Map(i32, i32).init(slice_iter, &double);
//! while (map.next()) |val| {
//!     // val will be 2, 4, 6, ...
//! }
//! ```

const std = @import("std");
const testing = std.testing;

/// Map adaptor that transforms each element using a transform function.
/// Time: O(1) per element (plus transform cost)
/// Space: O(1) — no additional allocations
///
/// This is a factory function that takes the concrete base iterator type
/// and returns a struct containing that type.
pub fn Map(comptime T: type, comptime U: type, comptime BaseIter: type) type {
    return struct {
        const Self = @This();

        /// Base iterator (concrete type with next() -> ?T method)
        base_iter: BaseIter,
        /// Transform function: T -> U
        transform_fn: *const fn (T) U,

        /// Initialize the Map adaptor with a base iterator and transform function.
        /// Time: O(1) | Space: O(1)
        pub fn init(base_iter: BaseIter, transform_fn: *const fn (T) U) Self {
            return .{
                .base_iter = base_iter,
                .transform_fn = transform_fn,
            };
        }

        /// Get the next transformed element
        /// Calls next() on the base iterator and applies the transform function.
        /// Time: O(1) + transform cost | Space: O(1)
        pub fn next(self: *Self) ?U {
            const value = self.base_iter.next() orelse return null;
            return self.transform_fn(value);
        }
    };
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

/// Transform functions for testing
fn double(x: i32) i32 {
    return x * 2;
}

fn addTen(x: i32) i32 {
    return x + 10;
}

fn intToFloat(x: i32) f32 {
    return @floatFromInt(x);
}

fn floatToInt(x: f32) i32 {
    return @intFromFloat(x);
}

fn negate(x: i32) i32 {
    return -x;
}

fn toBool(x: i32) bool {
    return x > 0;
}

test "map: basic transformation with i32" {
    var numbers = [_]i32{ 1, 2, 3 };
    var iter = SliceIterator(i32).init(&numbers);
    var map = Map(i32, i32, SliceIterator(i32)).init(&iter, double);

    try testing.expectEqual(2, map.next());
    try testing.expectEqual(4, map.next());
    try testing.expectEqual(6, map.next());
    try testing.expectEqual(null, map.next());
}

test "map: type transformation i32 to f32" {
    var numbers = [_]i32{ 1, 2, 3 };
    var iter = SliceIterator(i32).init(&numbers);
    var map = Map(i32, f32, SliceIterator(i32)).init(&iter, intToFloat);

    try testing.expectEqual(1.0, map.next());
    try testing.expectEqual(2.0, map.next());
    try testing.expectEqual(3.0, map.next());
    try testing.expectEqual(null, map.next());
}

test "map: empty iterator returns null immediately" {
    var numbers = [_]i32{};
    var iter = SliceIterator(i32).init(&numbers);
    var map = Map(i32, i32, SliceIterator(i32)).init(&iter, double);

    try testing.expectEqual(null, map.next());
    try testing.expectEqual(null, map.next());
}

test "map: single element" {
    var numbers = [_]i32{42};
    var iter = SliceIterator(i32).init(&numbers);
    var map = Map(i32, i32, SliceIterator(i32)).init(&iter, double);

    try testing.expectEqual(84, map.next());
    try testing.expectEqual(null, map.next());
}

test "map: chained transformations" {
    var numbers = [_]i32{ 1, 2, 3 };
    var iter1 = SliceIterator(i32).init(&numbers);
    var map1 = Map(i32, i32, SliceIterator(i32)).init(&iter1, double);
    var map2 = Map(i32, i32, @TypeOf(map1)).init(&map1, addTen);

    // First: double (1 -> 2, 2 -> 4, 3 -> 6)
    // Second: addTen (2 -> 12, 4 -> 14, 6 -> 16)
    try testing.expectEqual(12, map2.next());
    try testing.expectEqual(14, map2.next());
    try testing.expectEqual(16, map2.next());
    try testing.expectEqual(null, map2.next());
}

test "map: three-level chained transformations" {
    var numbers = [_]i32{ 1, 2 };
    var iter1 = SliceIterator(i32).init(&numbers);
    var map1 = Map(i32, i32, SliceIterator(i32)).init(&iter1, double);
    var map2 = Map(i32, i32, @TypeOf(map1)).init(&map1, addTen);
    var map3 = Map(i32, i32, @TypeOf(map2)).init(&map2, negate);

    // First: double (1 -> 2, 2 -> 4)
    // Second: addTen (2 -> 12, 4 -> 14)
    // Third: negate (12 -> -12, 14 -> -14)
    try testing.expectEqual(-12, map3.next());
    try testing.expectEqual(-14, map3.next());
    try testing.expectEqual(null, map3.next());
}

test "map: transform to bool" {
    var numbers = [_]i32{ -2, -1, 0, 1, 2 };
    var iter = SliceIterator(i32).init(&numbers);
    var map = Map(i32, bool, SliceIterator(i32)).init(&iter, toBool);

    try testing.expectEqual(false, map.next()); // -2 > 0
    try testing.expectEqual(false, map.next()); // -1 > 0
    try testing.expectEqual(false, map.next()); // 0 > 0
    try testing.expectEqual(true, map.next());  // 1 > 0
    try testing.expectEqual(true, map.next());  // 2 > 0
    try testing.expectEqual(null, map.next());
}

test "map: transform negative numbers" {
    var numbers = [_]i32{ -5, -3, 0, 3, 5 };
    var iter = SliceIterator(i32).init(&numbers);
    var map = Map(i32, i32, SliceIterator(i32)).init(&iter, negate);

    try testing.expectEqual(5, map.next());
    try testing.expectEqual(3, map.next());
    try testing.expectEqual(0, map.next());
    try testing.expectEqual(-3, map.next());
    try testing.expectEqual(-5, map.next());
    try testing.expectEqual(null, map.next());
}

test "map: exhaustion requires multiple next calls" {
    var numbers = [_]i32{1};
    var iter = SliceIterator(i32).init(&numbers);
    var map = Map(i32, i32, SliceIterator(i32)).init(&iter, double);

    try testing.expectEqual(2, map.next());
    try testing.expectEqual(null, map.next());
    try testing.expectEqual(null, map.next());
    try testing.expectEqual(null, map.next());
}

test "map: large dataset transformation" {
    var numbers: [1000]i32 = undefined;
    for (0..1000) |i| {
        numbers[i] = @intCast(i);
    }

    var iter = SliceIterator(i32).init(&numbers);
    var map = Map(i32, i32, SliceIterator(i32)).init(&iter, double);

    for (0..1000) |i| {
        const expected: i32 = @intCast(i * 2);
        try testing.expectEqual(expected, map.next());
    }
    try testing.expectEqual(null, map.next());
}

test "map: identity transformation" {
    var numbers = [_]i32{ 1, 2, 3 };
    var iter = SliceIterator(i32).init(&numbers);
    var map = Map(i32, i32, SliceIterator(i32)).init(&iter, identity);

    try testing.expectEqual(1, map.next());
    try testing.expectEqual(2, map.next());
    try testing.expectEqual(3, map.next());
    try testing.expectEqual(null, map.next());
}

fn identity(x: i32) i32 {
    return x;
}

test "map: type transformation f32 to i32" {
    var numbers = [_]f32{ 1.5, 2.7, 3.2 };
    var iter = SliceIterator(f32).init(&numbers);
    var map = Map(f32, i32, SliceIterator(f32)).init(&iter, floatToInt);

    try testing.expectEqual(1, map.next());
    try testing.expectEqual(2, map.next());
    try testing.expectEqual(3, map.next());
    try testing.expectEqual(null, map.next());
}

test "map: pointer preservation through transforms" {
    var numbers = [_]i32{ 10, 20, 30 };
    var iter = SliceIterator(i32).init(&numbers);
    var map = Map(i32, i32, SliceIterator(i32)).init(&iter, double);

    const val1 = map.next();
    const val2 = map.next();
    const val3 = map.next();

    try testing.expectEqual(20, val1);
    try testing.expectEqual(40, val2);
    try testing.expectEqual(60, val3);
}

test "map: state isolation between independent maps" {
    var numbers1 = [_]i32{ 1, 2, 3 };
    var numbers2 = [_]i32{ 10, 20, 30 };

    var iter1 = SliceIterator(i32).init(&numbers1);
    var map1 = Map(i32, i32, SliceIterator(i32)).init(&iter1, double);

    var iter2 = SliceIterator(i32).init(&numbers2);
    var map2 = Map(i32, i32, SliceIterator(i32)).init(&iter2, addTen);

    try testing.expectEqual(2, map1.next());
    try testing.expectEqual(20, map2.next());
    try testing.expectEqual(4, map1.next());
    try testing.expectEqual(30, map2.next());
    try testing.expectEqual(6, map1.next());
    try testing.expectEqual(40, map2.next());
    try testing.expectEqual(null, map1.next());
    try testing.expectEqual(null, map2.next());
}

test "map: with struct field access" {
    const Point = struct { x: i32, y: i32 };
    var points = [_]Point{ .{ .x = 1, .y = 2 }, .{ .x = 3, .y = 4 } };
    var iter = SliceIterator(Point).init(&points);
    var map = Map(Point, i32, SliceIterator(Point)).init(&iter, getX);

    try testing.expectEqual(1, map.next());
    try testing.expectEqual(3, map.next());
    try testing.expectEqual(null, map.next());
}

fn getX(p: struct { x: i32, y: i32 }) i32 {
    return p.x;
}

test "map: transform with complex computation" {
    var numbers = [_]i32{ 2, 3, 4, 5 };
    var iter = SliceIterator(i32).init(&numbers);
    var map = Map(i32, i32, SliceIterator(i32)).init(&iter, square);

    try testing.expectEqual(4, map.next());
    try testing.expectEqual(9, map.next());
    try testing.expectEqual(16, map.next());
    try testing.expectEqual(25, map.next());
    try testing.expectEqual(null, map.next());
}

fn square(x: i32) i32 {
    return x * x;
}

test "map: zero transformation result" {
    var numbers = [_]i32{ 0, 1, 2 };
    var iter = SliceIterator(i32).init(&numbers);
    var map = Map(i32, i32, SliceIterator(i32)).init(&iter, multi_zero);

    try testing.expectEqual(0, map.next());
    try testing.expectEqual(0, map.next());
    try testing.expectEqual(0, map.next());
    try testing.expectEqual(null, map.next());
}

fn multi_zero(x: i32) i32 {
    _ = x;
    return 0;
}

test "map: maximum values handling" {
    const max_i32 = std.math.maxInt(i32);
    var numbers = [_]i32{max_i32};
    var iter = SliceIterator(i32).init(&numbers);
    var map = Map(i32, i32, SliceIterator(i32)).init(&iter, identity);

    try testing.expectEqual(max_i32, map.next());
}

test "map: minimum values handling" {
    const min_i32 = std.math.minInt(i32);
    var numbers = [_]i32{min_i32};
    var iter = SliceIterator(i32).init(&numbers);
    var map = Map(i32, i32, SliceIterator(i32)).init(&iter, identity);

    try testing.expectEqual(min_i32, map.next());
}
