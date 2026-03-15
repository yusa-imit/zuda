//! Filter Iterator Adaptor
//!
//! The Filter adaptor yields only elements from a base iterator that satisfy
//! a provided predicate function. It lazily evaluates the predicate during
//! iteration, yielding only matching elements.
//!
//! ## Type Parameters
//! - T: Element type
//!
//! ## Time Complexity
//! - next(): O(n) worst case where n = number of elements to skip until a match
//!   (amortized O(1) if predicate is simple)
//!
//! ## Space Complexity
//! - O(1) — no allocation; stores only base iterator and function pointer
//!
//! ## Example
//! ```zig
//! var numbers = [_]i32{1, 2, 3, 4, 5};
//! var slice_iter = SliceIterator(i32).init(&numbers);
//! var filter = Filter(i32, SliceIterator(i32)).init(&iter, isEven);
//! while (filter.next()) |val| {
//!     // val will be 2, 4, ...
//! }
//! ```

const std = @import("std");
const testing = std.testing;

/// Filter adaptor that yields only elements matching a predicate.
/// Time: O(n) per element in worst case (amortized O(1) for simple predicates)
/// Space: O(1) — no additional allocations
///
/// This is a factory function that takes the concrete base iterator type
/// and returns a struct containing that type.
pub fn Filter(comptime T: type, comptime BaseIter: type) type {
    return struct {
        const Self = @This();

        /// Base iterator (concrete type with next() -> ?T method)
        base_iter: BaseIter,
        /// Predicate function: T -> bool (returns true if element should be included)
        predicate_fn: *const fn (T) bool,

        /// Initialize the Filter adaptor with a base iterator and predicate function.
        /// Time: O(1) | Space: O(1)
        pub fn init(base_iter: BaseIter, predicate_fn: *const fn (T) bool) Self {
            return .{
                .base_iter = base_iter,
                .predicate_fn = predicate_fn,
            };
        }

        /// Get the next element that satisfies the predicate.
        /// Calls next() on the base iterator, skipping elements where predicate returns false.
        /// Time: O(n) worst case | Space: O(1)
        pub fn next(self: *Self) ?T {
            while (self.base_iter.next()) |value| {
                if (self.predicate_fn(value)) {
                    return value;
                }
            }
            return null;
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

// -- Predicate functions for testing --

fn isEven(x: i32) bool {
    return x % 2 == 0;
}

fn isOdd(x: i32) bool {
    return x % 2 != 0;
}

fn isPositive(x: i32) bool {
    return x > 0;
}

fn isNegative(x: i32) bool {
    return x < 0;
}

fn isNonZero(x: i32) bool {
    return x != 0;
}

fn isGreaterThanTen(x: i32) bool {
    return x > 10;
}

fn isLessThanFive(x: i32) bool {
    return x < 5;
}

fn isTrue(x: bool) bool {
    return x;
}

fn isFalse(x: bool) bool {
    return !x;
}

fn alwaysTrue(x: i32) bool {
    _ = x;
    return true;
}

fn alwaysFalse(x: i32) bool {
    _ = x;
    return false;
}

fn isFloatPositive(x: f32) bool {
    return x > 0.0;
}

fn isFloatGreaterThanTwo(x: f32) bool {
    return x > 2.0;
}

fn stringIsLongerThanThree(s: []const u8) bool {
    return s.len > 3;
}

// -- Unit Tests (15+ cases) --

test "filter: basic filtering keeps evens" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var iter = SliceIterator(i32).init(&numbers);
    var filter = Filter(i32, SliceIterator(i32)).init(&iter, isEven);

    try testing.expectEqual(2, filter.next());
    try testing.expectEqual(4, filter.next());
    try testing.expectEqual(6, filter.next());
    try testing.expectEqual(null, filter.next());
}

test "filter: basic filtering keeps odds" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var iter = SliceIterator(i32).init(&numbers);
    var filter = Filter(i32, SliceIterator(i32)).init(&iter, isOdd);

    try testing.expectEqual(1, filter.next());
    try testing.expectEqual(3, filter.next());
    try testing.expectEqual(5, filter.next());
    try testing.expectEqual(null, filter.next());
}

test "filter: empty iterator returns null" {
    var numbers = [_]i32{};
    var iter = SliceIterator(i32).init(&numbers);
    var filter = Filter(i32, SliceIterator(i32)).init(&iter, isEven);

    try testing.expectEqual(null, filter.next());
    try testing.expectEqual(null, filter.next());
}

test "filter: filter all out (predicate always false)" {
    var numbers = [_]i32{ 1, 3, 5, 7 };
    var iter = SliceIterator(i32).init(&numbers);
    var filter = Filter(i32, SliceIterator(i32)).init(&iter, alwaysFalse);

    try testing.expectEqual(null, filter.next());
    try testing.expectEqual(null, filter.next());
}

test "filter: filter none out (predicate always true)" {
    var numbers = [_]i32{ 1, 2, 3 };
    var iter = SliceIterator(i32).init(&numbers);
    var filter = Filter(i32, SliceIterator(i32)).init(&iter, alwaysTrue);

    try testing.expectEqual(1, filter.next());
    try testing.expectEqual(2, filter.next());
    try testing.expectEqual(3, filter.next());
    try testing.expectEqual(null, filter.next());
}

test "filter: single element passes predicate" {
    var numbers = [_]i32{42};
    var iter = SliceIterator(i32).init(&numbers);
    var filter = Filter(i32, SliceIterator(i32)).init(&iter, isEven);

    try testing.expectEqual(42, filter.next());
    try testing.expectEqual(null, filter.next());
}

test "filter: single element fails predicate" {
    var numbers = [_]i32{41};
    var iter = SliceIterator(i32).init(&numbers);
    var filter = Filter(i32, SliceIterator(i32)).init(&iter, isEven);

    try testing.expectEqual(null, filter.next());
}

test "filter: chained filters (evens then > 10)" {
    var numbers = [_]i32{ 2, 4, 6, 8, 10, 12, 14, 16 };
    var iter1 = SliceIterator(i32).init(&numbers);
    var filter1 = Filter(i32, SliceIterator(i32)).init(&iter1, isEven);
    var filter2 = Filter(i32, @TypeOf(filter1)).init(&filter1, isGreaterThanTen);

    try testing.expectEqual(12, filter2.next());
    try testing.expectEqual(14, filter2.next());
    try testing.expectEqual(16, filter2.next());
    try testing.expectEqual(null, filter2.next());
}

test "filter: chained filters (odds then < 5)" {
    var numbers = [_]i32{ 1, 3, 5, 7, 9, 11 };
    var iter1 = SliceIterator(i32).init(&numbers);
    var filter1 = Filter(i32, SliceIterator(i32)).init(&iter1, isOdd);
    var filter2 = Filter(i32, @TypeOf(filter1)).init(&filter1, isLessThanFive);

    try testing.expectEqual(1, filter2.next());
    try testing.expectEqual(3, filter2.next());
    try testing.expectEqual(null, filter2.next());
}

test "filter: type i32 positive numbers" {
    var numbers = [_]i32{ -5, -2, 0, 3, 8, -1 };
    var iter = SliceIterator(i32).init(&numbers);
    var filter = Filter(i32, SliceIterator(i32)).init(&iter, isPositive);

    try testing.expectEqual(3, filter.next());
    try testing.expectEqual(8, filter.next());
    try testing.expectEqual(null, filter.next());
}

test "filter: type i32 negative numbers" {
    var numbers = [_]i32{ 5, -2, 0, 3, -8, 1 };
    var iter = SliceIterator(i32).init(&numbers);
    var filter = Filter(i32, SliceIterator(i32)).init(&iter, isNegative);

    try testing.expectEqual(-2, filter.next());
    try testing.expectEqual(-8, filter.next());
    try testing.expectEqual(null, filter.next());
}

test "filter: type f32 positive floats" {
    var numbers = [_]f32{ -1.5, 2.3, 0.0, 4.7, -0.5 };
    var iter = SliceIterator(f32).init(&numbers);
    var filter = Filter(f32, SliceIterator(f32)).init(&iter, isFloatPositive);

    try testing.expectEqual(2.3, filter.next());
    try testing.expectEqual(4.7, filter.next());
    try testing.expectEqual(null, filter.next());
}

test "filter: type bool filtering trues" {
    var bools = [_]bool{ false, true, false, true, true };
    var iter = SliceIterator(bool).init(&bools);
    var filter = Filter(bool, SliceIterator(bool)).init(&iter, isTrue);

    try testing.expectEqual(true, filter.next());
    try testing.expectEqual(true, filter.next());
    try testing.expectEqual(true, filter.next());
    try testing.expectEqual(null, filter.next());
}

test "filter: type bool filtering falses" {
    var bools = [_]bool{ true, false, true, false, false };
    var iter = SliceIterator(bool).init(&bools);
    var filter = Filter(bool, SliceIterator(bool)).init(&iter, isFalse);

    try testing.expectEqual(false, filter.next());
    try testing.expectEqual(false, filter.next());
    try testing.expectEqual(false, filter.next());
    try testing.expectEqual(null, filter.next());
}

test "filter: non-zero values" {
    var numbers = [_]i32{ 0, 1, 0, 2, 0, -3, 0 };
    var iter = SliceIterator(i32).init(&numbers);
    var filter = Filter(i32, SliceIterator(i32)).init(&iter, isNonZero);

    try testing.expectEqual(1, filter.next());
    try testing.expectEqual(2, filter.next());
    try testing.expectEqual(-3, filter.next());
    try testing.expectEqual(null, filter.next());
}

test "filter: large dataset (1000 elements)" {
    var numbers: [1000]i32 = undefined;
    for (0..1000) |i| {
        numbers[i] = @intCast(i);
    }

    var iter = SliceIterator(i32).init(&numbers);
    var filter = Filter(i32, SliceIterator(i32)).init(&iter, isEven);

    var count: usize = 0;
    while (filter.next()) |_| {
        count += 1;
    }
    try testing.expectEqual(500, count); // 0, 2, 4, ..., 998
}

test "filter: boundary condition - max i32 value" {
    const max_i32 = std.math.maxInt(i32);
    var numbers = [_]i32{ max_i32, 1, 2 };
    var iter = SliceIterator(i32).init(&numbers);
    var filter = Filter(i32, SliceIterator(i32)).init(&iter, isEven);

    try testing.expectEqual(2, filter.next());
    try testing.expectEqual(null, filter.next());
}

test "filter: boundary condition - min i32 value" {
    const min_i32 = std.math.minInt(i32);
    var numbers = [_]i32{ min_i32, 1, 2 };
    var iter = SliceIterator(i32).init(&numbers);
    var filter = Filter(i32, SliceIterator(i32)).init(&iter, isEven);

    try testing.expectEqual(2, filter.next());
    try testing.expectEqual(null, filter.next());
}

test "filter: zero values specific test" {
    var numbers = [_]i32{ 0, 0, 1, 0, 2 };
    var iter = SliceIterator(i32).init(&numbers);
    var filter = Filter(i32, SliceIterator(i32)).init(&iter, isNonZero);

    try testing.expectEqual(1, filter.next());
    try testing.expectEqual(2, filter.next());
    try testing.expectEqual(null, filter.next());
}

test "filter: iterator exhaustion with multiple next calls" {
    var numbers = [_]i32{ 1, 3, 5 };
    var iter = SliceIterator(i32).init(&numbers);
    var filter = Filter(i32, SliceIterator(i32)).init(&iter, isOdd);

    try testing.expectEqual(1, filter.next());
    try testing.expectEqual(3, filter.next());
    try testing.expectEqual(5, filter.next());
    try testing.expectEqual(null, filter.next());
    try testing.expectEqual(null, filter.next());
    try testing.expectEqual(null, filter.next());
}

test "filter: state isolation between independent filters" {
    var numbers1 = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var numbers2 = [_]i32{ 10, 15, 20, 25, 30 };

    var iter1 = SliceIterator(i32).init(&numbers1);
    var filter1 = Filter(i32, SliceIterator(i32)).init(&iter1, isEven);

    var iter2 = SliceIterator(i32).init(&numbers2);
    var filter2 = Filter(i32, SliceIterator(i32)).init(&iter2, isPositive);

    try testing.expectEqual(2, filter1.next());
    try testing.expectEqual(10, filter2.next());
    try testing.expectEqual(4, filter1.next());
    try testing.expectEqual(15, filter2.next());
    try testing.expectEqual(6, filter1.next());
    try testing.expectEqual(20, filter2.next());
    try testing.expectEqual(null, filter1.next());
    try testing.expectEqual(25, filter2.next());
    try testing.expectEqual(30, filter2.next());
    try testing.expectEqual(null, filter2.next());
}

test "filter: struct type filtering" {
    const Point = struct { x: i32, y: i32 };

    const pointXIsEven = struct {
        fn pred(p: Point) bool {
            return p.x % 2 == 0;
        }
    };

    var points = [_]Point{
        .{ .x = 1, .y = 2 },
        .{ .x = 2, .y = 3 },
        .{ .x = 3, .y = 4 },
        .{ .x = 4, .y = 5 },
    };
    var iter = SliceIterator(Point).init(&points);
    var filter = Filter(Point, SliceIterator(Point)).init(&iter, pointXIsEven.pred);

    const p1 = filter.next();
    try testing.expect(p1 != null);
    try testing.expectEqual(2, p1.?.x);

    const p2 = filter.next();
    try testing.expect(p2 != null);
    try testing.expectEqual(4, p2.?.x);

    try testing.expectEqual(null, filter.next());
}

test "filter: float type with complex predicate" {
    var numbers = [_]f32{ 1.0, 2.5, 3.0, 4.5, 5.0 };
    var iter = SliceIterator(f32).init(&numbers);
    var filter = Filter(f32, SliceIterator(f32)).init(&iter, isFloatGreaterThanTwo);

    try testing.expectEqual(2.5, filter.next());
    try testing.expectEqual(3.0, filter.next());
    try testing.expectEqual(4.5, filter.next());
    try testing.expectEqual(5.0, filter.next());
    try testing.expectEqual(null, filter.next());
}

test "filter: alternating elements (sparse filtering)" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var iter = SliceIterator(i32).init(&numbers);
    var filter = Filter(i32, SliceIterator(i32)).init(&iter, isEven);

    // Should get 2, 4, 6, 8, 10
    var count: usize = 0;
    while (filter.next()) |val| {
        count += 1;
        try testing.expectEqual(0, val % 2);
    }
    try testing.expectEqual(5, count);
}

test "filter: three-level chained filters" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    var iter1 = SliceIterator(i32).init(&numbers);
    var filter1 = Filter(i32, SliceIterator(i32)).init(&iter1, isEven); // 2, 4, 6, 8, 10, 12, 14, 16
    var filter2 = Filter(i32, @TypeOf(filter1)).init(&filter1, isGreaterThanTen); // 12, 14, 16
    var filter3 = Filter(i32, @TypeOf(filter2)).init(&filter2, isLessThanFive); // none (since min is 12)

    try testing.expectEqual(null, filter3.next());
}

test "filter: consecutive false predicates followed by true" {
    var numbers = [_]i32{ 1, 3, 5, 7, 9, 2 };
    var iter = SliceIterator(i32).init(&numbers);
    var filter = Filter(i32, SliceIterator(i32)).init(&iter, isEven);

    // First five are all odd, so predicate is false; last is even
    try testing.expectEqual(2, filter.next());
    try testing.expectEqual(null, filter.next());
}

test "filter: all elements at end pass predicate" {
    var numbers = [_]i32{ 1, 3, 5, 2, 4, 6 };
    var iter = SliceIterator(i32).init(&numbers);
    var filter = Filter(i32, SliceIterator(i32)).init(&iter, isEven);

    // Odds at start, evens at end
    try testing.expectEqual(2, filter.next());
    try testing.expectEqual(4, filter.next());
    try testing.expectEqual(6, filter.next());
    try testing.expectEqual(null, filter.next());
}

test "filter: negative and positive mixed" {
    var numbers = [_]i32{ -10, -5, 0, 5, 10, -3, 8 };
    var iter = SliceIterator(i32).init(&numbers);
    var filter = Filter(i32, SliceIterator(i32)).init(&iter, isEven);

    try testing.expectEqual(-10, filter.next());
    try testing.expectEqual(0, filter.next());
    try testing.expectEqual(10, filter.next());
    try testing.expectEqual(8, filter.next());
    try testing.expectEqual(null, filter.next());
}
