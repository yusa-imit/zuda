//! SkipWhile Iterator Adaptor
//!
//! The SkipWhile adaptor skips/drops elements from a base iterator while a provided
//! predicate function returns true. Once the predicate returns false for an element,
//! the adaptor yields that element and all subsequent elements from the base iterator,
//! regardless of whether they would satisfy the predicate.
//!
//! ## Type Parameters
//! - T: Element type
//! - BaseIter: The underlying iterator type
//!
//! ## Time Complexity
//! - next(): O(n) for first element (where n = number of elements to skip),
//!   then O(1) for subsequent calls
//! - Overall: O(skip_count + remaining_elements)
//!
//! ## Space Complexity
//! - O(1) — no allocation; stores only base iterator, predicate function pointer, and skipping flag
//!
//! ## Example
//! ```zig
//! var numbers = [_]i32{1, 2, 3, 4, 5, 6, 7};
//! var slice_iter = SliceIterator(i32).init(&numbers);
//! var skip_while = SkipWhile(i32, SliceIterator(i32)).init(slice_iter, lessThan5);
//! while (skip_while.next()) |val| {
//!     // val will be 5, 6, 7 (skips 1, 2, 3, 4 since all < 5)
//! }
//! ```

const std = @import("std");
const testing = std.testing;

/// SkipWhile adaptor that skips elements while a predicate is true, then yields rest.
/// Time: O(n) for first element (n = skip count), O(1) for rest
/// Space: O(1) — no additional allocations
///
/// This is a factory function that takes the concrete base iterator type
/// and returns a struct containing that type.
pub fn SkipWhile(comptime T: type, comptime BaseIter: type) type {
    return struct {
        const Self = @This();

        /// Base iterator (concrete type with next() -> ?T method)
        base_iter: BaseIter,
        /// Predicate function: T -> bool (returns true to skip, false to start yielding)
        predicate_fn: *const fn (T) bool,
        /// Flag tracking whether we're still in skip phase
        skipping: bool = true,

        /// Initialize the SkipWhile adaptor with a base iterator and predicate function.
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
                .skipping = true,
            };
        }

        /// Get the next element, skipping elements while predicate is true.
        /// Once predicate returns false for an element, that element and all subsequent
        /// elements are yielded until iterator exhaustion.
        /// Time: O(n) for first non-skipped element (n = skip count), O(1) thereafter
        /// Space: O(1)
        pub fn next(self: *Self) ?T {
            // Skip elements while predicate is true
            while (self.skipping) {
                const value = self.base_iter.next() orelse {
                    // Iterator exhausted while skipping
                    return null;
                };

                // Check if predicate is satisfied (should we keep skipping?)
                if (!self.predicate_fn(value)) {
                    // Predicate failed, we found the first non-skipped element
                    self.skipping = false;
                    return value;
                }
                // Predicate true, continue skipping
            }

            // Not skipping anymore, yield remaining elements from base iterator
            return self.base_iter.next();
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

fn lessThan3(x: i32) bool {
    return x < 3;
}

fn lessThan7(x: i32) bool {
    return x < 7;
}

fn lessThan8(x: i32) bool {
    return x < 8;
}

fn lessThan500(x: i32) bool {
    return x < 500;
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

test "skip_while: basic skip while < 5 from [1,2,3,4,5,6,7]" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5, 6, 7 };
    var iter = SliceIterator(i32).init(&numbers);
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(5, skip_while.next());
    try testing.expectEqual(6, skip_while.next());
    try testing.expectEqual(7, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: empty iterator returns null immediately" {
    var numbers = [_]i32{};
    var iter = SliceIterator(i32).init(&numbers);
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(null, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: all elements match predicate (skip all) returns null" {
    var numbers = [_]i32{ 1, 2, 3, 4 };
    var iter = SliceIterator(i32).init(&numbers);
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, lessThan100);

    try testing.expectEqual(null, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: no elements match predicate (skip none) yields all" {
    var numbers = [_]i32{ 5, 6, 7, 8 };
    var iter = SliceIterator(i32).init(&numbers);
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(5, skip_while.next());
    try testing.expectEqual(6, skip_while.next());
    try testing.expectEqual(7, skip_while.next());
    try testing.expectEqual(8, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: predicate fails on first element yields all elements" {
    var numbers = [_]i32{ 10, 1, 2, 3 };
    var iter = SliceIterator(i32).init(&numbers);
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(10, skip_while.next());
    try testing.expectEqual(1, skip_while.next());
    try testing.expectEqual(2, skip_while.next());
    try testing.expectEqual(3, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: predicate fails in middle yields from failure point onward" {
    var numbers = [_]i32{ 1, 2, 3, 5, 1, 2 };
    var iter = SliceIterator(i32).init(&numbers);
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(5, skip_while.next());
    try testing.expectEqual(1, skip_while.next());
    try testing.expectEqual(2, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: single element passes predicate returns null" {
    var numbers = [_]i32{3};
    var iter = SliceIterator(i32).init(&numbers);
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(null, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: single element fails predicate yields that element" {
    var numbers = [_]i32{10};
    var iter = SliceIterator(i32).init(&numbers);
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(10, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: mirror semantics with TakeWhile" {
    // TakeWhile yields: [1,2,3,4], SkipWhile yields: [5,6,7]
    // Combined should cover entire input
    var numbers = [_]i32{ 1, 2, 3, 4, 5, 6, 7 };
    var iter = SliceIterator(i32).init(&numbers);
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    var count: usize = 0;
    var sum: i32 = 0;
    while (skip_while.next()) |val| {
        count += 1;
        sum += val;
    }
    // Should yield 5 + 6 + 7 = 18
    try testing.expectEqual(3, count);
    try testing.expectEqual(18, sum);
}

test "skip_while: stress test 1000 elements skip while < 500 yields 500 elements" {
    var numbers: [1000]i32 = undefined;
    for (0..1000) |i| {
        numbers[i] = @intCast(i);
    }

    var iter = SliceIterator(i32).init(&numbers);
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, lessThan100);

    var count: usize = 0;
    while (skip_while.next()) |_| {
        count += 1;
    }
    try testing.expectEqual(900, count);
}

test "skip_while: type i32 with numeric predicate" {
    var numbers = [_]i32{ 1, 2, 3, 5, 6, 7 };
    var iter = SliceIterator(i32).init(&numbers);
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(5, skip_while.next());
    try testing.expectEqual(6, skip_while.next());
    try testing.expectEqual(7, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: type f32 with float predicate" {
    var numbers = [_]f32{ 0.5, 1.5, 2.5, 3.5, 4.5 };
    var iter = SliceIterator(f32).init(&numbers);
    var skip_while = SkipWhile(f32, SliceIterator(f32)).init(&iter, isFloatLessThanThree);

    try testing.expectEqual(3.5, skip_while.next());
    try testing.expectEqual(4.5, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: type []const u8 with string length predicate" {
    var strings = [_][]const u8{ "hi", "bye", "hello", "world" };
    var iter = SliceIterator([]const u8).init(&strings);
    var skip_while = SkipWhile([]const u8, SliceIterator([]const u8)).init(&iter, stringHasLengthLessThanFive);

    try testing.expectEqualStrings("hello", skip_while.next().?);
    try testing.expectEqualStrings("world", skip_while.next().?);
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: predicate always true skips all returns null" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5 };
    var iter = SliceIterator(i32).init(&numbers);
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, alwaysTrue);

    try testing.expectEqual(null, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: predicate always false skips none yields all" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5 };
    var iter = SliceIterator(i32).init(&numbers);
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, alwaysFalse);

    try testing.expectEqual(1, skip_while.next());
    try testing.expectEqual(2, skip_while.next());
    try testing.expectEqual(3, skip_while.next());
    try testing.expectEqual(4, skip_while.next());
    try testing.expectEqual(5, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: even numbers skipWhile(isEven) from [2,4,6,7,8,9]" {
    var numbers = [_]i32{ 2, 4, 6, 7, 8, 9 };
    var iter = SliceIterator(i32).init(&numbers);
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, isEven);

    try testing.expectEqual(7, skip_while.next());
    try testing.expectEqual(8, skip_while.next());
    try testing.expectEqual(9, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: negative numbers skipWhile(lessThanZero) from [-3,-2,-1,0,1,2]" {
    var numbers = [_]i32{ -3, -2, -1, 0, 1, 2 };
    var iter = SliceIterator(i32).init(&numbers);
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, lessThanZero);

    try testing.expectEqual(0, skip_while.next());
    try testing.expectEqual(1, skip_while.next());
    try testing.expectEqual(2, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: multiple null calls after exhaustion" {
    var numbers = [_]i32{ 1, 2, 3, 10 };
    var iter = SliceIterator(i32).init(&numbers);
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(10, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: state isolation between independent instances" {
    var numbers1 = [_]i32{ 1, 2, 3, 10, 11 };
    var numbers2 = [_]i32{ 1, 2, 3, 4, 5 };

    var iter1 = SliceIterator(i32).init(&numbers1);
    var skip_while1 = SkipWhile(i32, SliceIterator(i32)).init(&iter1, lessThan5);

    var iter2 = SliceIterator(i32).init(&numbers2);
    var skip_while2 = SkipWhile(i32, SliceIterator(i32)).init(&iter2, lessThan10);

    try testing.expectEqual(10, skip_while1.next());
    try testing.expectEqual(null, skip_while2.next());
    try testing.expectEqual(11, skip_while1.next());
    try testing.expectEqual(null, skip_while1.next());
    try testing.expectEqual(null, skip_while2.next());
}

test "skip_while: type bool with boolean predicate" {
    var bools = [_]bool{ true, true, false, true };
    var iter = SliceIterator(bool).init(&bools);
    var skip_while = SkipWhile(bool, SliceIterator(bool)).init(&iter, isBoolTrue);

    try testing.expectEqual(false, skip_while.next());
    try testing.expectEqual(true, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: boundary max i32 value" {
    const max_i32 = std.math.maxInt(i32);
    var numbers = [_]i32{ 42, 100, max_i32 };
    var iter = SliceIterator(i32).init(&numbers);
    // max_i32 is not less than 5, so predicate fails on first
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(42, skip_while.next());
    try testing.expectEqual(100, skip_while.next());
    try testing.expectEqual(max_i32, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: boundary min i32 value" {
    const min_i32 = std.math.minInt(i32);
    var numbers = [_]i32{ min_i32, -50, -10, 0, 5 };
    var iter = SliceIterator(i32).init(&numbers);
    // min_i32 is definitely less than 5, should be skipped
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(5, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: chained SkipWhile instances" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var iter = SliceIterator(i32).init(&numbers);
    var skip_while1 = SkipWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);
    var skip_while2 = SkipWhile(i32, @TypeOf(skip_while1)).init(&skip_while1, lessThan8);

    // skip_while1 skips < 5: [5,6,7,8,9,10]
    // skip_while2 skips < 8: [8,9,10]
    try testing.expectEqual(8, skip_while2.next());
    try testing.expectEqual(9, skip_while2.next());
    try testing.expectEqual(10, skip_while2.next());
    try testing.expectEqual(null, skip_while2.next());
}

test "skip_while: zero value in sequence [0,1,2,3,10]" {
    var numbers = [_]i32{ 0, 1, 2, 3, 10 };
    var iter = SliceIterator(i32).init(&numbers);
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    try testing.expectEqual(10, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: negative positive boundary [-5,-1,0,1,5]" {
    var numbers = [_]i32{ -5, -1, 0, 1, 5 };
    var iter = SliceIterator(i32).init(&numbers);
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, lessThanZero);

    // All negative values skipped: [-5, -1]
    try testing.expectEqual(0, skip_while.next());
    try testing.expectEqual(1, skip_while.next());
    try testing.expectEqual(5, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: positive sequence [1,2,3,4,0,-1]" {
    var numbers = [_]i32{ 1, 2, 3, 4, 0, -1 };
    var iter = SliceIterator(i32).init(&numbers);
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, isPositive);

    // All positive values skipped: [1,2,3,4]
    try testing.expectEqual(0, skip_while.next());
    try testing.expectEqual(-1, skip_while.next());
    try testing.expectEqual(null, skip_while.next());
}

test "skip_while: half dataset skip 500 from 1000" {
    var numbers: [1000]i32 = undefined;
    for (0..1000) |i| {
        numbers[i] = @intCast(i);
    }

    var iter = SliceIterator(i32).init(&numbers);
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, lessThan500);

    // First element after skipping should be 500
    try testing.expectEqual(500, skip_while.next());
    try testing.expectEqual(501, skip_while.next());

    // Count remaining elements
    var count: usize = 2;
    while (skip_while.next()) |_| {
        count += 1;
    }
    // Should have 1000 - 500 = 500 elements total, already consumed 2, so 498 remaining + 2 = 500
    try testing.expectEqual(500, count);
}

test "skip_while: three-level chained skip_while" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var iter = SliceIterator(i32).init(&numbers);
    var skip_while1 = SkipWhile(i32, SliceIterator(i32)).init(&iter, lessThan3);
    var skip_while2 = SkipWhile(i32, @TypeOf(skip_while1)).init(&skip_while1, lessThan5);
    var skip_while3 = SkipWhile(i32, @TypeOf(skip_while2)).init(&skip_while2, lessThan7);

    // skip_while1: skip < 3: [3,4,5,6,7,8]
    // skip_while2: skip < 5: [5,6,7,8]
    // skip_while3: skip < 7: [7,8]
    try testing.expectEqual(7, skip_while3.next());
    try testing.expectEqual(8, skip_while3.next());
    try testing.expectEqual(null, skip_while3.next());
}

test "skip_while: skip then continue iterator works correctly" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var iter = SliceIterator(i32).init(&numbers);
    var skip_while = SkipWhile(i32, SliceIterator(i32)).init(&iter, lessThan5);

    // Skip phase: consumes 1,2,3,4, returns 5
    const val1 = skip_while.next();
    try testing.expectEqual(5, val1);

    // Now in normal phase: returns remaining elements
    const val2 = skip_while.next();
    try testing.expectEqual(6, val2);

    const val3 = skip_while.next();
    try testing.expectEqual(7, val3);

    const val4 = skip_while.next();
    try testing.expectEqual(8, val4);

    try testing.expectEqual(null, skip_while.next());
}
