//! Chain Iterator Adaptor
//!
//! The Chain adaptor concatenates two iterators of the same element type,
//! yielding all elements from the first iterator, then all elements from
//! the second iterator. It enables sequential composition of heterogeneous
//! iterator types with the same element type.
//!
//! ## Type Parameters
//! - T: Element type (must be the same for both iterators)
//! - FirstIter: First iterator type (must have next() -> ?T)
//! - SecondIter: Second iterator type (must have next() -> ?T)
//!
//! ## Time Complexity
//! - next(): O(1) — single method call on current iterator
//!
//! ## Space Complexity
//! - O(1) — stores only two iterator instances, no additional allocations
//!
//! ## Example
//! ```zig
//! var first = [_]i32{1, 2};
//! var second = [_]i32{3, 4};
//! const first_iter = SliceIterator(i32).init(&first);
//! const second_iter = SliceIterator(i32).init(&second);
//! var chained = Chain(i32, SliceIterator(i32), SliceIterator(i32))
//!     .init(first_iter, second_iter);
//! while (chained.next()) |val| {
//!     // val will be 1, 2, 3, 4
//! }
//! ```

const std = @import("std");
const testing = std.testing;

/// Chain adaptor that concatenates two iterators of the same element type.
/// Time: O(1) per element
/// Space: O(1) — no additional allocations
///
/// This is a factory function that takes two concrete iterator types
/// and returns a struct that chains them together.
pub fn Chain(comptime T: type, comptime FirstIter: type, comptime SecondIter: type) type {
    return struct {
        const Self = @This();

        /// First iterator
        first_iter: FirstIter,
        /// Second iterator
        second_iter: SecondIter,
        /// Flag indicating if we've exhausted the first iterator
        first_exhausted: bool = false,

        /// Initialize the Chain adaptor with two iterators.
        /// Time: O(1) | Space: O(1)
        pub fn init(first_iter: FirstIter, second_iter: SecondIter) Self {
            return .{
                .first_iter = first_iter,
                .second_iter = second_iter,
                .first_exhausted = false,
            };
        }

        /// Get the next element from the chain.
        /// Returns elements from the first iterator until exhausted, then from second.
        /// Time: O(1) | Space: O(1)
        pub fn next(self: *Self) ?T {
            if (!self.first_exhausted) {
                if (self.first_iter.next()) |value| {
                    return value;
                }
                self.first_exhausted = true;
            }
            return self.second_iter.next();
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

test "chain: basic chaining of two non-empty iterators" {
    var first = [_]i32{ 1, 2 };
    var second = [_]i32{ 3, 4 };
    const first_iter = SliceIterator(i32).init(&first);
    const second_iter = SliceIterator(i32).init(&second);
    var chained = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(first_iter, second_iter);

    try testing.expectEqual(1, chained.next());
    try testing.expectEqual(2, chained.next());
    try testing.expectEqual(3, chained.next());
    try testing.expectEqual(4, chained.next());
    try testing.expectEqual(null, chained.next());
}

test "chain: empty first iterator, non-empty second" {
    var first = [_]i32{};
    var second = [_]i32{ 5, 6, 7 };
    const first_iter = SliceIterator(i32).init(&first);
    const second_iter = SliceIterator(i32).init(&second);
    var chained = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(first_iter, second_iter);

    try testing.expectEqual(5, chained.next());
    try testing.expectEqual(6, chained.next());
    try testing.expectEqual(7, chained.next());
    try testing.expectEqual(null, chained.next());
}

test "chain: non-empty first iterator, empty second" {
    var first = [_]i32{ 10, 20, 30 };
    var second = [_]i32{};
    const first_iter = SliceIterator(i32).init(&first);
    const second_iter = SliceIterator(i32).init(&second);
    var chained = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(first_iter, second_iter);

    try testing.expectEqual(10, chained.next());
    try testing.expectEqual(20, chained.next());
    try testing.expectEqual(30, chained.next());
    try testing.expectEqual(null, chained.next());
}

test "chain: both iterators empty" {
    var first = [_]i32{};
    var second = [_]i32{};
    const first_iter = SliceIterator(i32).init(&first);
    const second_iter = SliceIterator(i32).init(&second);
    var chained = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(first_iter, second_iter);

    try testing.expectEqual(null, chained.next());
    try testing.expectEqual(null, chained.next());
}

test "chain: single element in first, single element in second" {
    var first = [_]i32{42};
    var second = [_]i32{99};
    const first_iter = SliceIterator(i32).init(&first);
    const second_iter = SliceIterator(i32).init(&second);
    var chained = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(first_iter, second_iter);

    try testing.expectEqual(42, chained.next());
    try testing.expectEqual(99, chained.next());
    try testing.expectEqual(null, chained.next());
}

test "chain: single element in first, empty second" {
    var first = [_]i32{42};
    var second = [_]i32{};
    const first_iter = SliceIterator(i32).init(&first);
    const second_iter = SliceIterator(i32).init(&second);
    var chained = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(first_iter, second_iter);

    try testing.expectEqual(42, chained.next());
    try testing.expectEqual(null, chained.next());
}

test "chain: empty first, single element in second" {
    var first = [_]i32{};
    var second = [_]i32{42};
    const first_iter = SliceIterator(i32).init(&first);
    const second_iter = SliceIterator(i32).init(&second);
    var chained = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(first_iter, second_iter);

    try testing.expectEqual(42, chained.next());
    try testing.expectEqual(null, chained.next());
}

test "chain: unequal length iterators (longer first)" {
    var first = [_]i32{ 1, 2, 3, 4, 5 };
    var second = [_]i32{ 6 };
    const first_iter = SliceIterator(i32).init(&first);
    const second_iter = SliceIterator(i32).init(&second);
    var chained = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(first_iter, second_iter);

    try testing.expectEqual(1, chained.next());
    try testing.expectEqual(2, chained.next());
    try testing.expectEqual(3, chained.next());
    try testing.expectEqual(4, chained.next());
    try testing.expectEqual(5, chained.next());
    try testing.expectEqual(6, chained.next());
    try testing.expectEqual(null, chained.next());
}

test "chain: unequal length iterators (longer second)" {
    var first = [_]i32{100};
    var second = [_]i32{ 200, 300, 400, 500 };
    const first_iter = SliceIterator(i32).init(&first);
    const second_iter = SliceIterator(i32).init(&second);
    var chained = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(first_iter, second_iter);

    try testing.expectEqual(100, chained.next());
    try testing.expectEqual(200, chained.next());
    try testing.expectEqual(300, chained.next());
    try testing.expectEqual(400, chained.next());
    try testing.expectEqual(500, chained.next());
    try testing.expectEqual(null, chained.next());
}

test "chain: large dataset (1000+ elements total)" {
    var first: [500]i32 = undefined;
    var second: [600]i32 = undefined;

    for (0..500) |i| {
        first[i] = @intCast(i);
    }
    for (0..600) |i| {
        second[i] = @intCast(500 + i);
    }

    const first_iter = SliceIterator(i32).init(&first);
    const second_iter = SliceIterator(i32).init(&second);
    var chained = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(first_iter, second_iter);

    var count: usize = 0;
    var expected: i32 = 0;
    while (chained.next()) |val| {
        try testing.expectEqual(expected, val);
        expected += 1;
        count += 1;
    }
    try testing.expectEqual(1100, count);
}

test "chain: chain of chains (3-level)" {
    var arr1 = [_]i32{1};
    var arr2 = [_]i32{2};
    var arr3 = [_]i32{3};

    const iter1 = SliceIterator(i32).init(&arr1);
    const iter2 = SliceIterator(i32).init(&arr2);
    const chain1 = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(iter1, iter2);

    const iter3 = SliceIterator(i32).init(&arr3);
    var chain2 = Chain(i32, @TypeOf(chain1), SliceIterator(i32)).init(chain1, iter3);

    try testing.expectEqual(1, chain2.next());
    try testing.expectEqual(2, chain2.next());
    try testing.expectEqual(3, chain2.next());
    try testing.expectEqual(null, chain2.next());
}

test "chain: different iterator types with same element type" {
    // Using two SliceIterators with different sources simulates different types
    var arr1 = [_]i32{ 10, 20 };
    var arr2 = [_]i32{ 30, 40, 50 };

    const iter1 = SliceIterator(i32).init(&arr1);
    const iter2 = SliceIterator(i32).init(&arr2);
    var chained = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(iter1, iter2);

    try testing.expectEqual(10, chained.next());
    try testing.expectEqual(20, chained.next());
    try testing.expectEqual(30, chained.next());
    try testing.expectEqual(40, chained.next());
    try testing.expectEqual(50, chained.next());
    try testing.expectEqual(null, chained.next());
}

test "chain: iterator exhaustion behavior" {
    var first = [_]i32{1};
    var second = [_]i32{2};
    const first_iter = SliceIterator(i32).init(&first);
    const second_iter = SliceIterator(i32).init(&second);
    var chained = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(first_iter, second_iter);

    // Exhaust the chain
    _ = chained.next();
    _ = chained.next();

    // Further calls should return null
    try testing.expectEqual(null, chained.next());
    try testing.expectEqual(null, chained.next());
    try testing.expectEqual(null, chained.next());
}

test "chain: state isolation (independent chain instances)" {
    var arr1 = [_]i32{ 1, 2 };
    var arr2 = [_]i32{ 3, 4 };
    var arr3 = [_]i32{ 100, 200 };
    var arr4 = [_]i32{ 300, 400 };

    const iter1_a = SliceIterator(i32).init(&arr1);
    const iter2_a = SliceIterator(i32).init(&arr2);
    var chain1 = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(iter1_a, iter2_a);

    const iter1_b = SliceIterator(i32).init(&arr3);
    const iter2_b = SliceIterator(i32).init(&arr4);
    var chain2 = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(iter1_b, iter2_b);

    try testing.expectEqual(1, chain1.next());
    try testing.expectEqual(100, chain2.next());
    try testing.expectEqual(2, chain1.next());
    try testing.expectEqual(200, chain2.next());
    try testing.expectEqual(3, chain1.next());
    try testing.expectEqual(300, chain2.next());
    try testing.expectEqual(4, chain1.next());
    try testing.expectEqual(400, chain2.next());
    try testing.expectEqual(null, chain1.next());
    try testing.expectEqual(null, chain2.next());
}

test "chain: boundary conditions with zero values" {
    var first = [_]i32{ 0, 0, 0 };
    var second = [_]i32{ 0, 1 };
    const first_iter = SliceIterator(i32).init(&first);
    const second_iter = SliceIterator(i32).init(&second);
    var chained = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(first_iter, second_iter);

    try testing.expectEqual(0, chained.next());
    try testing.expectEqual(0, chained.next());
    try testing.expectEqual(0, chained.next());
    try testing.expectEqual(0, chained.next());
    try testing.expectEqual(1, chained.next());
    try testing.expectEqual(null, chained.next());
}

test "chain: boundary conditions with negative numbers" {
    var first = [_]i32{ -100, -50 };
    var second = [_]i32{ -25, 0, 25 };
    const first_iter = SliceIterator(i32).init(&first);
    const second_iter = SliceIterator(i32).init(&second);
    var chained = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(first_iter, second_iter);

    try testing.expectEqual(-100, chained.next());
    try testing.expectEqual(-50, chained.next());
    try testing.expectEqual(-25, chained.next());
    try testing.expectEqual(0, chained.next());
    try testing.expectEqual(25, chained.next());
    try testing.expectEqual(null, chained.next());
}

test "chain: boundary conditions with max i32 values" {
    const max_i32 = std.math.maxInt(i32);
    var first = [_]i32{ max_i32, max_i32 - 1 };
    var second = [_]i32{ 1, 0 };
    const first_iter = SliceIterator(i32).init(&first);
    const second_iter = SliceIterator(i32).init(&second);
    var chained = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(first_iter, second_iter);

    try testing.expectEqual(max_i32, chained.next());
    try testing.expectEqual(max_i32 - 1, chained.next());
    try testing.expectEqual(1, chained.next());
    try testing.expectEqual(0, chained.next());
    try testing.expectEqual(null, chained.next());
}

test "chain: boundary conditions with min i32 values" {
    const min_i32 = std.math.minInt(i32);
    var first = [_]i32{ min_i32, min_i32 + 1 };
    var second = [_]i32{ -1, 0 };
    const first_iter = SliceIterator(i32).init(&first);
    const second_iter = SliceIterator(i32).init(&second);
    var chained = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(first_iter, second_iter);

    try testing.expectEqual(min_i32, chained.next());
    try testing.expectEqual(min_i32 + 1, chained.next());
    try testing.expectEqual(-1, chained.next());
    try testing.expectEqual(0, chained.next());
    try testing.expectEqual(null, chained.next());
}

test "chain: f32 type - floats" {
    var first = [_]f32{ 1.5, 2.5 };
    var second = [_]f32{ 3.5, 4.5 };
    const first_iter = SliceIterator(f32).init(&first);
    const second_iter = SliceIterator(f32).init(&second);
    var chained = Chain(f32, SliceIterator(f32), SliceIterator(f32)).init(first_iter, second_iter);

    try testing.expectEqual(1.5, chained.next());
    try testing.expectEqual(2.5, chained.next());
    try testing.expectEqual(3.5, chained.next());
    try testing.expectEqual(4.5, chained.next());
    try testing.expectEqual(null, chained.next());
}

test "chain: bool type - booleans" {
    var first = [_]bool{ true, false };
    var second = [_]bool{ true, true, false };
    const first_iter = SliceIterator(bool).init(&first);
    const second_iter = SliceIterator(bool).init(&second);
    var chained = Chain(bool, SliceIterator(bool), SliceIterator(bool)).init(first_iter, second_iter);

    try testing.expectEqual(true, chained.next());
    try testing.expectEqual(false, chained.next());
    try testing.expectEqual(true, chained.next());
    try testing.expectEqual(true, chained.next());
    try testing.expectEqual(false, chained.next());
    try testing.expectEqual(null, chained.next());
}

test "chain: struct type - Point" {
    const Point = struct { x: i32, y: i32 };

    var first = [_]Point{ .{ .x = 1, .y = 2 }, .{ .x = 3, .y = 4 } };
    var second = [_]Point{ .{ .x = 5, .y = 6 } };

    const first_iter = SliceIterator(Point).init(&first);
    const second_iter = SliceIterator(Point).init(&second);
    var chained = Chain(Point, SliceIterator(Point), SliceIterator(Point)).init(first_iter, second_iter);

    const p1 = chained.next();
    try testing.expect(p1 != null);
    try testing.expectEqual(1, p1.?.x);
    try testing.expectEqual(2, p1.?.y);

    const p2 = chained.next();
    try testing.expect(p2 != null);
    try testing.expectEqual(3, p2.?.x);
    try testing.expectEqual(4, p2.?.y);

    const p3 = chained.next();
    try testing.expect(p3 != null);
    try testing.expectEqual(5, p3.?.x);
    try testing.expectEqual(6, p3.?.y);

    try testing.expectEqual(null, chained.next());
}

test "chain: alternating iteration pattern" {
    var first = [_]i32{ 10, 20, 30 };
    var second = [_]i32{ 40, 50 };
    const first_iter = SliceIterator(i32).init(&first);
    const second_iter = SliceIterator(i32).init(&second);
    var chained = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(first_iter, second_iter);

    var count: usize = 0;
    while (chained.next()) |_| {
        count += 1;
    }
    try testing.expectEqual(5, count);
}

test "chain: many elements in first, few in second" {
    var first: [100]i32 = undefined;
    var second = [_]i32{ 500, 600 };

    for (0..100) |i| {
        first[i] = @intCast(i);
    }

    const first_iter = SliceIterator(i32).init(&first);
    const second_iter = SliceIterator(i32).init(&second);
    var chained = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(first_iter, second_iter);

    var count: usize = 0;
    while (chained.next()) |val| {
        if (count < 100) {
            try testing.expectEqual(@as(i32, @intCast(count)), val);
        } else if (count == 100) {
            try testing.expectEqual(500, val);
        } else {
            try testing.expectEqual(600, val);
        }
        count += 1;
    }
    try testing.expectEqual(102, count);
}

test "chain: few elements in first, many in second" {
    var first = [_]i32{ 1, 2 };
    var second: [100]i32 = undefined;

    for (0..100) |i| {
        second[i] = @intCast(10 + i);
    }

    const first_iter = SliceIterator(i32).init(&first);
    const second_iter = SliceIterator(i32).init(&second);
    var chained = Chain(i32, SliceIterator(i32), SliceIterator(i32)).init(first_iter, second_iter);

    try testing.expectEqual(1, chained.next());
    try testing.expectEqual(2, chained.next());

    var count: usize = 2;
    var expected: i32 = 10;
    while (chained.next()) |val| {
        try testing.expectEqual(expected, val);
        expected += 1;
        count += 1;
    }
    try testing.expectEqual(102, count);
}
