//! TakeWhile Iterator Adaptor
//!
//! The TakeWhile adaptor yields elements from a base iterator while a provided
//! predicate function returns true. Once the predicate returns false for an element,
//! the adaptor stops yielding and returns null for all subsequent calls, even if the
//! base iterator has more elements.
//!
//! ## Type Parameters
//! - T: Element type
//! - BaseIter: The underlying iterator type
//!
//! ## Time Complexity
//! - next(): O(1) amortized — delegates to base iterator and evaluates predicate once per call
//!
//! ## Space Complexity
//! - O(1) — no allocation; stores only base iterator, predicate function pointer, and stopped flag
//!
//! ## Example
//! ```zig
//! var numbers = [_]i32{1, 2, 3, 4, 5, 6, 7};
//! var slice_iter = SliceIterator(i32).init(&numbers);
//! var take_while = TakeWhile(i32, SliceIterator(i32)).init(slice_iter, lessThan5);
//! while (take_while.next()) |val| {
//!     // val will be 1, 2, 3, 4 (stops on 5)
//! }
//! ```

const std = @import("std");
const testing = std.testing;

/// TakeWhile adaptor that yields elements while a predicate is true.
/// Time: O(1) per element
/// Space: O(1) — no additional allocations
///
/// This is a factory function that takes the concrete base iterator type
/// and returns a struct containing that type.
pub fn TakeWhile(comptime T: type, comptime BaseIter: type) type {
    return struct {
        const Self = @This();

        /// Base iterator (concrete type with next() -> ?T method)
        base_iter: BaseIter,
        /// Predicate function: T -> bool (returns true to continue, false to stop)
        predicate_fn: *const fn (T) bool,
        /// Flag tracking whether we've encountered a false predicate
        stopped: bool = false,

        /// Initialize the TakeWhile adaptor with a base iterator and predicate function.
        /// The base_iter can be passed by value (for stack-based iterators) or by reference.
        /// Time: O(1) | Space: O(1)
        pub fn init(base_iter: anytype, predicate_fn: *const fn (T) bool) Self {
            const input_type = @TypeOf(base_iter);
            const actual_iter: BaseIter = if (input_type == BaseIter)
                base_iter
            else if (input_type == *BaseIter)
                base_iter.*
            else
                @compileError("Expected iterator of type " ++ @typeName(BaseIter));

            return .{
                .base_iter = actual_iter,
                .predicate_fn = predicate_fn,
                .stopped = false,
            };
        }

        /// Get the next element while the predicate is true.
        /// Once the predicate returns false for any element, always returns null thereafter.
        /// Time: O(1) | Space: O(1)
        pub fn next(self: *Self) ?T {
            // If we've already seen a false predicate, stop yielding
            if (self.stopped) return null;

            // Try to get the next element from base iterator
            const value = self.base_iter.next() orelse {
                self.stopped = true;
                return null;
            };

            // Check if the predicate is satisfied
            if (self.predicate_fn(value)) {
                return value;
            } else {
                // Predicate failed, mark as stopped and return null
                self.stopped = true;
                return null;
            }
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

fn lessThan5(x: i32) bool {
    return x < 5;
}

fn lessThan10(x: i32) bool {
    return x < 10;
}

fn lessThan100(x: i32) bool {
    return x < 100;
}

fn lessThanZero(x: i32) bool {
    return x < 0;
}

fn isEven(x: i32) bool {
    return @rem(x, 2) == 0;
}

fn isPositive(x: i32) bool {
    return x > 0;
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

fn isFloatLessThanThree(x: f32) bool {
    return x < 3.0;
}

fn stringHasLengthLessThanFive(s: []const u8) bool {
    return s.len < 5;
}

fn isBoolTrue(x: bool) bool {
    return x;
}

// -- Unit Tests (15+ comprehensive cases) --

test "take_while: basic take while < 5 from [1,2,3,4,5,6,7]" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5, 6, 7 };
    var iter = SliceIterator(i32).init(&numbers);
    var take_while = TakeWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(1, take_while.next());
    try testing.expectEqual(2, take_while.next());
    try testing.expectEqual(3, take_while.next());
    try testing.expectEqual(4, take_while.next());
    try testing.expectEqual(null, take_while.next());
}

test "take_while: empty iterator returns null immediately" {
    var numbers = [_]i32{};
    var iter = SliceIterator(i32).init(&numbers);
    var take_while = TakeWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(null, take_while.next());
    try testing.expectEqual(null, take_while.next());
}

test "take_while: all elements match predicate yields all" {
    var numbers = [_]i32{ 1, 2, 3, 4 };
    var iter = SliceIterator(i32).init(&numbers);
    var take_while = TakeWhile(i32, SliceIterator(i32)).init(&iter, lessThan100);

    try testing.expectEqual(1, take_while.next());
    try testing.expectEqual(2, take_while.next());
    try testing.expectEqual(3, take_while.next());
    try testing.expectEqual(4, take_while.next());
    try testing.expectEqual(null, take_while.next());
}

test "take_while: no elements match predicate returns null immediately" {
    var numbers = [_]i32{ 5, 6, 7, 8 };
    var iter = SliceIterator(i32).init(&numbers);
    var take_while = TakeWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(null, take_while.next());
    try testing.expectEqual(null, take_while.next());
}

test "take_while: predicate fails on first element" {
    var numbers = [_]i32{ 10, 1, 2, 3 };
    var iter = SliceIterator(i32).init(&numbers);
    var take_while = TakeWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(null, take_while.next());
    try testing.expectEqual(null, take_while.next());
}

test "take_while: predicate fails in middle stops correctly" {
    var numbers = [_]i32{ 1, 2, 3, 5, 6, 7 };
    var iter = SliceIterator(i32).init(&numbers);
    var take_while = TakeWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(1, take_while.next());
    try testing.expectEqual(2, take_while.next());
    try testing.expectEqual(3, take_while.next());
    try testing.expectEqual(null, take_while.next());
    try testing.expectEqual(null, take_while.next());
}

test "take_while: single element passes predicate" {
    var numbers = [_]i32{3};
    var iter = SliceIterator(i32).init(&numbers);
    var take_while = TakeWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(3, take_while.next());
    try testing.expectEqual(null, take_while.next());
}

test "take_while: single element fails predicate" {
    var numbers = [_]i32{10};
    var iter = SliceIterator(i32).init(&numbers);
    var take_while = TakeWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(null, take_while.next());
}

test "take_while: chaining TakeWhile into Filter pipeline" {
    var numbers = [_]i32{ 2, 4, 6, 8, 10, 12, 14 };
    var iter = SliceIterator(i32).init(&numbers);
    var take_while = TakeWhile(i32, SliceIterator(i32)).init(&iter, lessThan10);

    // take_while yields: 2, 4, 6, 8
    // If we manually filter for even: should all be even
    var count: usize = 0;
    while (take_while.next()) |val| {
        count += 1;
        try testing.expectEqual(0, @rem(val, 2));
    }
    try testing.expectEqual(4, count);
}

test "take_while: stress test 1000 elements take while < 500" {
    var numbers: [1000]i32 = undefined;
    for (0..1000) |i| {
        numbers[i] = @intCast(i);
    }

    var iter = SliceIterator(i32).init(&numbers);
    var take_while = TakeWhile(i32, SliceIterator(i32)).init(&iter, lessThan100);

    var count: usize = 0;
    while (take_while.next()) |_| {
        count += 1;
    }
    try testing.expectEqual(100, count);
}

test "take_while: type f32 with float predicate" {
    var numbers = [_]f32{ 0.5, 1.5, 2.5, 3.5, 4.5 };
    var iter = SliceIterator(f32).init(&numbers);
    var take_while = TakeWhile(f32, SliceIterator(f32)).init(&iter, isFloatLessThanThree);

    try testing.expectEqual(0.5, take_while.next());
    try testing.expectEqual(1.5, take_while.next());
    try testing.expectEqual(2.5, take_while.next());
    try testing.expectEqual(null, take_while.next());
}

test "take_while: type []const u8 with string length predicate" {
    var strings = [_][]const u8{ "hi", "bye", "hello", "world", "ab" };
    var iter = SliceIterator([]const u8).init(&strings);
    var take_while = TakeWhile([]const u8, SliceIterator([]const u8)).init(&iter, stringHasLengthLessThanFive);

    try testing.expectEqualStrings("hi", take_while.next().?);
    try testing.expectEqualStrings("bye", take_while.next().?);
    try testing.expectEqual(null, take_while.next());
}

test "take_while: predicate always true yields all" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5 };
    var iter = SliceIterator(i32).init(&numbers);
    var take_while = TakeWhile(i32, SliceIterator(i32)).init(&iter, alwaysTrue);

    try testing.expectEqual(1, take_while.next());
    try testing.expectEqual(2, take_while.next());
    try testing.expectEqual(3, take_while.next());
    try testing.expectEqual(4, take_while.next());
    try testing.expectEqual(5, take_while.next());
    try testing.expectEqual(null, take_while.next());
}

test "take_while: predicate always false returns null immediately" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5 };
    var iter = SliceIterator(i32).init(&numbers);
    var take_while = TakeWhile(i32, SliceIterator(i32)).init(&iter, alwaysFalse);

    try testing.expectEqual(null, take_while.next());
    try testing.expectEqual(null, take_while.next());
}

test "take_while: even numbers takeWhile from [2,4,6,7,8]" {
    var numbers = [_]i32{ 2, 4, 6, 7, 8 };
    var iter = SliceIterator(i32).init(&numbers);
    var take_while = TakeWhile(i32, SliceIterator(i32)).init(&iter, isEven);

    try testing.expectEqual(2, take_while.next());
    try testing.expectEqual(4, take_while.next());
    try testing.expectEqual(6, take_while.next());
    try testing.expectEqual(null, take_while.next());
}

test "take_while: negative numbers takeWhile(isNegative) from [-3,-2,-1,0,1]" {
    var numbers = [_]i32{ -3, -2, -1, 0, 1 };
    var iter = SliceIterator(i32).init(&numbers);
    var take_while = TakeWhile(i32, SliceIterator(i32)).init(&iter, lessThanZero);

    try testing.expectEqual(-3, take_while.next());
    try testing.expectEqual(-2, take_while.next());
    try testing.expectEqual(-1, take_while.next());
    try testing.expectEqual(null, take_while.next());
}

test "take_while: multiple null calls after exhaustion" {
    var numbers = [_]i32{ 1, 2, 3, 10 };
    var iter = SliceIterator(i32).init(&numbers);
    var take_while = TakeWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(1, take_while.next());
    try testing.expectEqual(2, take_while.next());
    try testing.expectEqual(3, take_while.next());
    try testing.expectEqual(null, take_while.next());
    try testing.expectEqual(null, take_while.next());
    try testing.expectEqual(null, take_while.next());
}

test "take_while: state isolation between independent instances" {
    var numbers1 = [_]i32{ 1, 2, 3, 10, 11 };
    var numbers2 = [_]i32{ 10, 20, 30, 5, 6 };

    var iter1 = SliceIterator(i32).init(&numbers1);
    var take_while1 = TakeWhile(i32, SliceIterator(i32)).init(&iter1, lessThan5);

    var iter2 = SliceIterator(i32).init(&numbers2);
    var take_while2 = TakeWhile(i32, SliceIterator(i32)).init(&iter2, lessThan10);

    try testing.expectEqual(1, take_while1.next());
    try testing.expectEqual(null, take_while2.next());
    try testing.expectEqual(2, take_while1.next());
    try testing.expectEqual(3, take_while1.next());
    try testing.expectEqual(null, take_while1.next());
    try testing.expectEqual(null, take_while2.next());
}

test "take_while: type bool with boolean predicate" {
    var bools = [_]bool{ true, true, false, true };
    var iter = SliceIterator(bool).init(&bools);
    var take_while = TakeWhile(bool, SliceIterator(bool)).init(&iter, isBoolTrue);

    try testing.expectEqual(true, take_while.next());
    try testing.expectEqual(true, take_while.next());
    try testing.expectEqual(null, take_while.next());
}

test "take_while: boundary max i32 value" {
    const max_i32 = std.math.maxInt(i32);
    var numbers = [_]i32{ max_i32, 100, 200 };
    var iter = SliceIterator(i32).init(&numbers);
    // max_i32 is not less than 5, so predicate fails on first
    var take_while = TakeWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(null, take_while.next());
}

test "take_while: boundary min i32 value" {
    const min_i32 = std.math.minInt(i32);
    var numbers = [_]i32{ min_i32, -50, -10, 0, 5 };
    var iter = SliceIterator(i32).init(&numbers);
    // min_i32 is definitely less than 5
    var take_while = TakeWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(min_i32, take_while.next());
    try testing.expectEqual(-50, take_while.next());
    try testing.expectEqual(-10, take_while.next());
    try testing.expectEqual(0, take_while.next());
    try testing.expectEqual(null, take_while.next());
}

test "take_while: chained TakeWhile instances" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var iter = SliceIterator(i32).init(&numbers);
    var take_while1 = TakeWhile(i32, SliceIterator(i32)).init(&iter, lessThan10);
    var take_while2 = TakeWhile(i32, @TypeOf(take_while1)).init(&take_while1, lessThan5);

    // take_while1 limits to < 10: [1,2,3,4,5,6,7,8,9]
    // take_while2 limits to < 5: [1,2,3,4]
    try testing.expectEqual(1, take_while2.next());
    try testing.expectEqual(2, take_while2.next());
    try testing.expectEqual(3, take_while2.next());
    try testing.expectEqual(4, take_while2.next());
    try testing.expectEqual(null, take_while2.next());
}

test "take_while: zero value in sequence [0,1,2,3,10]" {
    var numbers = [_]i32{ 0, 1, 2, 3, 10 };
    var iter = SliceIterator(i32).init(&numbers);
    var take_while = TakeWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(0, take_while.next());
    try testing.expectEqual(1, take_while.next());
    try testing.expectEqual(2, take_while.next());
    try testing.expectEqual(3, take_while.next());
    try testing.expectEqual(null, take_while.next());
}

test "take_while: negative positive boundary [-5,-1,0,1,5]" {
    var numbers = [_]i32{ -5, -1, 0, 1, 5 };
    var iter = SliceIterator(i32).init(&numbers);
    var take_while = TakeWhile(i32, SliceIterator(i32)).init(&iter, isPositive);

    // Only 1 is positive
    try testing.expectEqual(null, take_while.next());
}

test "take_while: positive sequence [1,2,3,4,0,-1]" {
    var numbers = [_]i32{ 1, 2, 3, 4, 0, -1 };
    var iter = SliceIterator(i32).init(&numbers);
    var take_while = TakeWhile(i32, SliceIterator(i32)).init(&iter, isPositive);

    try testing.expectEqual(1, take_while.next());
    try testing.expectEqual(2, take_while.next());
    try testing.expectEqual(3, take_while.next());
    try testing.expectEqual(4, take_while.next());
    try testing.expectEqual(null, take_while.next());
}
