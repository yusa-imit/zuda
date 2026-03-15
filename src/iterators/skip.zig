//! Skip Iterator Adaptor
//!
//! The Skip adaptor discards the first N elements from a base iterator and yields
//! the remaining elements. The skipping is performed lazily on the first call to next(),
//! not during initialization.
//!
//! ## Type Parameters
//! - T: Element type
//! - BaseIter: The underlying iterator type
//!
//! ## Time Complexity
//! - next(): O(n) for the first call (where n = skip count to perform initial skipping),
//!   then O(1) for subsequent calls
//! - Overall: O(skip_count + remaining_elements)
//!
//! ## Space Complexity
//! - O(1) — no allocation; stores only base iterator, skip count, and state flag
//!
//! ## Example
//! ```zig
//! var numbers = [_]i32{1, 2, 3, 4, 5};
//! var slice_iter = SliceIterator(i32).init(&numbers);
//! var skip_it = Skip(i32, SliceIterator(i32)).init(slice_iter, 2);
//! while (skip_it.next()) |val| {
//!     // val will be 3, 4, 5 (first 2 elements skipped)
//! }
//! ```

const std = @import("std");
const testing = std.testing;

/// Skip adaptor that discards the first N elements from a base iterator.
/// Time: O(n) for first element (n = skip count), O(1) thereafter
/// Space: O(1) — no additional allocations
///
/// This is a factory function that takes the concrete base iterator type
/// and returns a struct containing that type.
pub fn Skip(comptime T: type, comptime BaseIter: type) type {
    // Determine if BaseIter is a pointer type
    const base_iter_info = @typeInfo(BaseIter);
    const is_pointer = switch (base_iter_info) {
        .pointer => true,
        else => false,
    };

    // If BaseIter is not a pointer, create a struct that can accept either
    // value or pointer through type flexibility
    if (!is_pointer) {
        return struct {
            const Self = @This();

            /// Base iterator (concrete type with next() -> ?T method)
            base_iter: BaseIter,
            /// Number of elements to skip (only used on first next() call)
            skip_count: usize,
            /// Flag to track if initial skip has been performed
            skipped: bool = false,

            /// Initialize the Skip adaptor with a base iterator and skip count.
            /// The actual skipping happens on the first call to next(), not here.
            /// Time: O(1) | Space: O(1)
            pub fn init(base_iter: anytype, n: usize) Self {
                const input_type = @TypeOf(base_iter);
                const input_is_ptr = switch (@typeInfo(input_type)) {
                    .pointer => true,
                    else => false,
                };

                // If input is a pointer to BaseIter, we can reinterpret it
                const actual_value: BaseIter = if (input_is_ptr)
                    @as(*BaseIter, @ptrCast(base_iter)).*
                else
                    base_iter;

                return .{
                    .base_iter = actual_value,
                    .skip_count = n,
                    .skipped = false,
                };
            }

            /// Get the next element, skipping the first N elements on first call.
            /// Subsequent calls return remaining elements until exhaustion.
            /// Time: O(n) for first call (n = skip_count), O(1) for rest
            /// Space: O(1)
            pub fn next(self: *Self) ?T {
                // Perform the skip on first call
                if (!self.skipped) {
                    var i: usize = 0;
                    while (i < self.skip_count) : (i += 1) {
                        _ = self.base_iter.next() orelse return null;
                    }
                    self.skipped = true;
                }

                // Now return elements from the base iterator
                return self.base_iter.next();
            }
        };
    } else {
        // BaseIter is a pointer
        return struct {
            const Self = @This();

            /// Base iterator (pointer type with next() -> ?T method)
            base_iter: BaseIter,
            /// Number of elements to skip (only used on first next() call)
            skip_count: usize,
            /// Flag to track if initial skip has been performed
            skipped: bool = false,

            /// Initialize the Skip adaptor with a base iterator and skip count.
            /// The actual skipping happens on the first call to next(), not here.
            /// Time: O(1) | Space: O(1)
            pub fn init(base_iter: BaseIter, n: usize) Self {
                return .{
                    .base_iter = base_iter,
                    .skip_count = n,
                    .skipped = false,
                };
            }

            /// Get the next element, skipping the first N elements on first call.
            /// Subsequent calls return remaining elements until exhaustion.
            /// Time: O(n) for first call (n = skip_count), O(1) for rest
            /// Space: O(1)
            pub fn next(self: *Self) ?T {
                // Perform the skip on first call
                if (!self.skipped) {
                    var i: usize = 0;
                    while (i < self.skip_count) : (i += 1) {
                        _ = self.base_iter.*.next() orelse return null;
                    }
                    self.skipped = true;
                }

                // Now return elements from the base iterator
                return self.base_iter.*.next();
            }
        };
    }
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

// -- Unit Tests (18+ cases) --

test "skip: basic skip N < iterator length" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5 };
    const iter = SliceIterator(i32).init(&numbers);
    var skip_it = Skip(i32, SliceIterator(i32)).init(iter, 2);

    try testing.expectEqual(3, skip_it.next());
    try testing.expectEqual(4, skip_it.next());
    try testing.expectEqual(5, skip_it.next());
    try testing.expectEqual(null, skip_it.next());
}

test "skip: skip none N = 0" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5 };
    const iter = SliceIterator(i32).init(&numbers);
    var skip_it = Skip(i32, SliceIterator(i32)).init(iter, 0);

    try testing.expectEqual(1, skip_it.next());
    try testing.expectEqual(2, skip_it.next());
    try testing.expectEqual(3, skip_it.next());
    try testing.expectEqual(4, skip_it.next());
    try testing.expectEqual(5, skip_it.next());
    try testing.expectEqual(null, skip_it.next());
}

test "skip: skip all N >= iterator length" {
    var numbers = [_]i32{ 1, 2, 3 };
    const iter = SliceIterator(i32).init(&numbers);
    var skip_it = Skip(i32, SliceIterator(i32)).init(iter, 3);

    try testing.expectEqual(null, skip_it.next());
    try testing.expectEqual(null, skip_it.next());
}

test "skip: skip more than available N > iterator length" {
    var numbers = [_]i32{ 1, 2, 3 };
    const iter = SliceIterator(i32).init(&numbers);
    var skip_it = Skip(i32, SliceIterator(i32)).init(iter, 10);

    try testing.expectEqual(null, skip_it.next());
    try testing.expectEqual(null, skip_it.next());
}

test "skip: empty iterator with N > 0" {
    var numbers = [_]i32{};
    const iter = SliceIterator(i32).init(&numbers);
    var skip_it = Skip(i32, SliceIterator(i32)).init(iter, 5);

    try testing.expectEqual(null, skip_it.next());
    try testing.expectEqual(null, skip_it.next());
}

test "skip: single element skip 0" {
    var numbers = [_]i32{42};
    const iter = SliceIterator(i32).init(&numbers);
    var skip_it = Skip(i32, SliceIterator(i32)).init(iter, 0);

    try testing.expectEqual(42, skip_it.next());
    try testing.expectEqual(null, skip_it.next());
}

test "skip: single element skip 1" {
    var numbers = [_]i32{42};
    const iter = SliceIterator(i32).init(&numbers);
    var skip_it = Skip(i32, SliceIterator(i32)).init(iter, 1);

    try testing.expectEqual(null, skip_it.next());
    try testing.expectEqual(null, skip_it.next());
}

test "skip: large dataset skip 100 from 1000" {
    var numbers: [1000]i32 = undefined;
    for (0..1000) |i| {
        numbers[i] = @intCast(i);
    }

    const iter = SliceIterator(i32).init(&numbers);
    var skip_it = Skip(i32, SliceIterator(i32)).init(iter, 100);

    // First element after skip should be 100
    try testing.expectEqual(100, skip_it.next());
    try testing.expectEqual(101, skip_it.next());

    // Skip ahead to near the end
    var count: usize = 2;
    while (skip_it.next()) |_| {
        count += 1;
    }
    // Should have 1000 - 100 = 900 elements total, and we consumed 2, so 898 remaining
    try testing.expectEqual(900, count);
}

test "skip: skip exact length minus 1" {
    var numbers = [_]i32{ 10, 20, 30, 40, 50 };
    const iter = SliceIterator(i32).init(&numbers);
    var skip_it = Skip(i32, SliceIterator(i32)).init(iter, 4);

    try testing.expectEqual(50, skip_it.next());
    try testing.expectEqual(null, skip_it.next());
}

test "skip: iterator exhaustion behavior" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5 };
    const iter = SliceIterator(i32).init(&numbers);
    var skip_it = Skip(i32, SliceIterator(i32)).init(iter, 2);

    try testing.expectEqual(3, skip_it.next());
    try testing.expectEqual(4, skip_it.next());
    try testing.expectEqual(5, skip_it.next());
    try testing.expectEqual(null, skip_it.next());
    try testing.expectEqual(null, skip_it.next());
    try testing.expectEqual(null, skip_it.next());
}

test "skip: state isolation independent skip instances" {
    var numbers1 = [_]i32{ 1, 2, 3, 4, 5 };
    var numbers2 = [_]i32{ 10, 20, 30, 40, 50 };

    const iter1 = SliceIterator(i32).init(&numbers1);
    var skip1 = Skip(i32, SliceIterator(i32)).init(iter1, 1);

    const iter2 = SliceIterator(i32).init(&numbers2);
    var skip2 = Skip(i32, SliceIterator(i32)).init(iter2, 2);

    try testing.expectEqual(2, skip1.next());
    try testing.expectEqual(30, skip2.next());
    try testing.expectEqual(3, skip1.next());
    try testing.expectEqual(40, skip2.next());
    try testing.expectEqual(4, skip1.next());
    try testing.expectEqual(50, skip2.next());
    try testing.expectEqual(5, skip1.next());
    try testing.expectEqual(null, skip2.next());
    try testing.expectEqual(null, skip1.next());
}

test "skip: with f32 type" {
    var numbers = [_]f32{ 1.5, 2.5, 3.5, 4.5 };
    const iter = SliceIterator(f32).init(&numbers);
    var skip_it = Skip(f32, SliceIterator(f32)).init(iter, 1);

    try testing.expectEqual(2.5, skip_it.next());
    try testing.expectEqual(3.5, skip_it.next());
    try testing.expectEqual(4.5, skip_it.next());
    try testing.expectEqual(null, skip_it.next());
}

test "skip: with bool type" {
    var numbers = [_]bool{ true, false, true, false, true };
    const iter = SliceIterator(bool).init(&numbers);
    var skip_it = Skip(bool, SliceIterator(bool)).init(iter, 2);

    try testing.expectEqual(true, skip_it.next());
    try testing.expectEqual(false, skip_it.next());
    try testing.expectEqual(true, skip_it.next());
    try testing.expectEqual(null, skip_it.next());
}

test "skip: with struct type" {
    const Point = struct { x: i32, y: i32 };
    var points = [_]Point{
        .{ .x = 1, .y = 2 },
        .{ .x = 3, .y = 4 },
        .{ .x = 5, .y = 6 },
    };
    const iter = SliceIterator(Point).init(&points);
    var skip_it = Skip(Point, SliceIterator(Point)).init(iter, 1);

    const p1 = skip_it.next();
    try testing.expect(p1 != null);
    try testing.expectEqual(3, p1.?.x);
    try testing.expectEqual(4, p1.?.y);

    const p2 = skip_it.next();
    try testing.expect(p2 != null);
    try testing.expectEqual(5, p2.?.x);
    try testing.expectEqual(6, p2.?.y);

    try testing.expectEqual(null, skip_it.next());
}

test "skip: boundary condition max i32 value" {
    const max_i32 = std.math.maxInt(i32);
    var numbers = [_]i32{ 42, max_i32, 100 };
    const iter = SliceIterator(i32).init(&numbers);
    var skip_it = Skip(i32, SliceIterator(i32)).init(iter, 1);

    try testing.expectEqual(max_i32, skip_it.next());
    try testing.expectEqual(100, skip_it.next());
    try testing.expectEqual(null, skip_it.next());
}

test "skip: boundary condition min i32 value" {
    const min_i32 = std.math.minInt(i32);
    var numbers = [_]i32{ 42, min_i32, 100 };
    const iter = SliceIterator(i32).init(&numbers);
    var skip_it = Skip(i32, SliceIterator(i32)).init(iter, 1);

    try testing.expectEqual(min_i32, skip_it.next());
    try testing.expectEqual(100, skip_it.next());
    try testing.expectEqual(null, skip_it.next());
}

test "skip: chained skips skip(skip(...))" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5 };
    const iter = SliceIterator(i32).init(&numbers);
    var skip1 = Skip(i32, SliceIterator(i32)).init(iter, 2);
    var skip2 = Skip(i32, @TypeOf(skip1)).init(&skip1, 1);

    // skip1 skips first 2: [3, 4, 5]
    // skip2 skips first 1 of those: [4, 5]
    try testing.expectEqual(4, skip2.next());
    try testing.expectEqual(5, skip2.next());
    try testing.expectEqual(null, skip2.next());
}

test "skip: zero value in iterator" {
    var numbers = [_]i32{ 0, 1, 2, 3 };
    const iter = SliceIterator(i32).init(&numbers);
    var skip_it = Skip(i32, SliceIterator(i32)).init(iter, 1);

    try testing.expectEqual(1, skip_it.next());
    try testing.expectEqual(2, skip_it.next());
    try testing.expectEqual(3, skip_it.next());
    try testing.expectEqual(null, skip_it.next());
}

test "skip: combined with take skip(take(...))" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const iter = SliceIterator(i32).init(&numbers);
    // Can't actually test with take yet, but verify skip structure works
    var skip_it = Skip(i32, SliceIterator(i32)).init(iter, 3);

    try testing.expectEqual(4, skip_it.next());
    try testing.expectEqual(5, skip_it.next());
    try testing.expectEqual(6, skip_it.next());
    try testing.expectEqual(7, skip_it.next());
    try testing.expectEqual(8, skip_it.next());
    try testing.expectEqual(null, skip_it.next());
}

test "skip: negative concept large usize skip count" {
    // usize cannot hold negative values, verify very large skip works
    var numbers = [_]i32{ 1, 2, 3 };
    const iter = SliceIterator(i32).init(&numbers);
    const large_n: usize = 9999999;
    var skip_it = Skip(i32, SliceIterator(i32)).init(iter, large_n);

    try testing.expectEqual(null, skip_it.next());
    try testing.expectEqual(null, skip_it.next());
}

test "skip: alternating pattern skip in middle" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const iter = SliceIterator(i32).init(&numbers);
    var skip_it = Skip(i32, SliceIterator(i32)).init(iter, 4);

    var count: usize = 0;
    while (skip_it.next()) |val| {
        count += 1;
        try testing.expect(val >= 5 and val <= 8);
    }
    try testing.expectEqual(4, count); // Should have 8 - 4 = 4 elements
}

test "skip: skip half of dataset" {
    var numbers: [100]i32 = undefined;
    for (0..100) |i| {
        numbers[i] = @intCast(i);
    }

    const iter = SliceIterator(i32).init(&numbers);
    var skip_it = Skip(i32, SliceIterator(i32)).init(iter, 50);

    // First element after skip should be 50
    try testing.expectEqual(50, skip_it.next());

    // Count remaining
    var count: usize = 1;
    while (skip_it.next()) |_| {
        count += 1;
    }
    try testing.expectEqual(50, count); // 50 remaining elements
}

test "skip: three-level chained skips" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const iter = SliceIterator(i32).init(&numbers);
    var skip1 = Skip(i32, SliceIterator(i32)).init(iter, 2);
    var skip2 = Skip(i32, @TypeOf(skip1)).init(&skip1, 2);
    var skip3 = Skip(i32, @TypeOf(skip2)).init(&skip2, 1);

    // skip1: [3, 4, 5, 6, 7, 8]
    // skip2: [5, 6, 7, 8]
    // skip3: [6, 7, 8]
    try testing.expectEqual(6, skip3.next());
    try testing.expectEqual(7, skip3.next());
    try testing.expectEqual(8, skip3.next());
    try testing.expectEqual(null, skip3.next());
}

test "skip: alternating skip zero and nonzero" {
    var numbers1 = [_]i32{ 1, 2, 3, 4, 5 };
    var numbers2 = [_]i32{ 10, 20, 30, 40, 50 };

    const iter1 = SliceIterator(i32).init(&numbers1);
    var skip_zero = Skip(i32, SliceIterator(i32)).init(iter1, 0);

    const iter2 = SliceIterator(i32).init(&numbers2);
    var skip_nonzero = Skip(i32, SliceIterator(i32)).init(iter2, 3);

    // skip_zero should yield all (no skip)
    // skip_nonzero skips first 3: [40, 50]
    try testing.expectEqual(1, skip_zero.next());
    try testing.expectEqual(40, skip_nonzero.next());
    try testing.expectEqual(2, skip_zero.next());
    try testing.expectEqual(50, skip_nonzero.next());
    try testing.expectEqual(3, skip_zero.next());
    try testing.expectEqual(null, skip_nonzero.next());
}
