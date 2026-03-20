//! FlatMap Iterator Adaptor
//!
//! The FlatMap adaptor applies a transformation function to each element from a base
//! iterator, where the transformation produces another iterator (the "inner iterator").
//! It then flattens these nested iterators by yielding all elements from each inner
//! iterator before moving to the next.
//!
//! ## Type Parameters
//! - T: Input element type from the base iterator
//! - U: Output element type (elements yielded by inner iterators)
//! - BaseIter: The underlying iterator type that yields T
//! - mapFn: Comptime function pointer (T) -> InnerIter (where InnerIter yields ?U)
//!
//! ## Time Complexity
//! - next(): O(1) amortized — constant work per yielded element
//!
//! ## Space Complexity
//! - O(1) — no intermediate allocation; only one inner iterator stored at a time
//!
//! ## Lazy Evaluation
//! - No buffering of results; inner iterators are consumed on-the-fly
//! - Outer iterator is only advanced when the current inner iterator is exhausted
//!
//! ## Example
//! ```zig
//! var outer = [_]i32{1, 2, 3};
//! var outer_iter = SliceIterator(i32).init(&outer);
//!
//! fn toRange(n: i32) SliceIterator(i32) {
//!     // returns an iterator yielding 1..n
//! }
//!
//! var flat = FlatMap(i32, i32, SliceIterator(i32), toRange).init(outer_iter);
//! while (flat.next()) |val| {
//!     // yields: 1, 1, 2, 1, 2, 3
//! }
//! ```

const std = @import("std");
const testing = std.testing;

/// FlatMap adaptor that flattens nested iterators.
/// Time: O(1) amortized per element
/// Space: O(1) — no buffering, only current inner iterator stored
///
/// This is a factory function returning a struct that manages both the outer
/// (base) iterator and the current inner iterator.
///
/// Type Parameters:
/// - T: Input element type from base iterator
/// - U: Output element type (elements yielded by inner iterators)
/// - BaseIter: The base iterator type
/// - InnerIter: The type of iterators produced by mapFn
pub fn FlatMap(
    comptime T: type,
    comptime U: type,
    comptime BaseIter: type,
    comptime InnerIter: type,
) type {
    return struct {
        const Self = @This();

        /// Base iterator that yields T
        base_iter: BaseIter,
        /// Currently active inner iterator (null if exhausted or not yet initialized)
        current_inner: ?InnerIter = null,
        /// Transform function: T -> InnerIter
        mapFn: *const fn (T) InnerIter,

        /// Initialize FlatMap with a base iterator and transform function.
        /// Time: O(1) | Space: O(1)
        pub fn init(base_iter: BaseIter, mapFn: *const fn (T) InnerIter) Self {
            return .{
                .base_iter = base_iter,
                .current_inner = null,
                .mapFn = mapFn,
            };
        }

        /// Get the next element, transparently advancing through inner iterators.
        /// Returns null when all elements from all inner iterators are exhausted.
        /// Time: O(1) amortized | Space: O(1)
        pub fn next(self: *Self) ?U {
            // Main loop: keep trying until we yield an element or run out of outers
            while (true) {
                // If we have a current inner iterator, try to get the next element from it
                if (self.current_inner) |*inner| {
                    if (inner.next()) |elem| {
                        return elem;
                    }
                    // Inner iterator exhausted, loop to fetch next outer
                    self.current_inner = null;
                }

                // No current inner (or it was exhausted), fetch the next outer element
                if (self.base_iter.next()) |outer_elem| {
                    self.current_inner = self.mapFn(outer_elem);
                    // Continue loop to try yielding from this new inner iterator
                } else {
                    // Outer iterator exhausted
                    return null;
                }
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

/// Optional iterator: wraps a slice but can skip some indices
fn OptionalSliceIterator(comptime T: type) type {
    return struct {
        const SelfIter = @This();

        slice: []const ?T,
        index: usize = 0,

        fn init(slice: []const ?T) SelfIter {
            return .{ .slice = slice };
        }

        fn next(self: *SelfIter) ?T {
            while (self.index < self.slice.len) : (self.index += 1) {
                if (self.slice[self.index]) |val| {
                    defer self.index += 1;
                    return val;
                }
            }
            return null;
        }
    };
}

// Transform functions for testing

/// Create an iterator yielding integers from 1 to n (inclusive)
fn range(n: i32) SliceIterator(i32) {
    // For testing, we use pre-allocated ranges
    // This is a simplified approach; a real range would use comptime or allocator
    var result: [5]i32 = undefined;
    var len: usize = 0;
    var i: i32 = 1;
    while (i <= n and len < 5) : (i += 1) {
        result[len] = i;
        len += 1;
    }
    return SliceIterator(i32).init(result[0..len]);
}

/// Replicate element n times
fn replicateSlice(n: i32) SliceIterator(i32) {
    var result: [10]i32 = undefined;
    for (0..@intCast(n)) |i| {
        result[i] = n;
    }
    return SliceIterator(i32).init(result[0..@intCast(n)]);
}

/// Convert an int to a slice of its digits (as f64)
fn digits(n: i32) SliceIterator(f64) {
    var result: [10]f64 = undefined;
    var abs_n: i32 = if (n < 0) -n else n;
    var len: usize = 0;

    if (abs_n == 0) {
        result[0] = 0.0;
        len = 1;
    } else {
        while (abs_n > 0) : (abs_n = @divTrunc(abs_n, 10)) {
            const digit: f64 = @floatFromInt(@mod(abs_n, 10));
            result[len] = digit;
            len += 1;
        }
        // Reverse to get proper order
        var i: usize = 0;
        var j: usize = len - 1;
        while (i < j) {
            std.mem.swap(f64, &result[i], &result[j]);
            i += 1;
            j -= 1;
        }
    }
    return SliceIterator(f64).init(result[0..len]);
}

// -- Tests (15+ comprehensive cases) --

test "flat_map: basic nested arrays to flat sequence" {
    // Outer: [1, 2, 3]
    // Inner for 1: [1]
    // Inner for 2: [1, 2]
    // Inner for 3: [1, 2, 3]
    // Expected: [1, 1, 2, 1, 2, 3]
    var outer = [_]i32{ 1, 2, 3 };
    const outer_iter = SliceIterator(i32).init(&outer);
    var flat = FlatMap(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(outer_iter, range);

    var results: [6]i32 = undefined;
    var count: usize = 0;

    while (flat.next()) |val| {
        results[count] = val;
        count += 1;
    }

    try testing.expectEqual(6, count);
    try testing.expectEqual(1, results[0]);
    try testing.expectEqual(1, results[1]);
    try testing.expectEqual(2, results[2]);
    try testing.expectEqual(1, results[3]);
    try testing.expectEqual(2, results[4]);
    try testing.expectEqual(3, results[5]);
}

test "flat_map: empty outer iterator returns null" {
    var outer = [_]i32{};
    const outer_iter = SliceIterator(i32).init(&outer);
    var flat = FlatMap(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(outer_iter, range);

    try testing.expectEqual(null, flat.next());
    try testing.expectEqual(null, flat.next());
}

test "flat_map: single outer element yields all inner elements" {
    // Outer: [2]
    // Inner for 2: [1, 2]
    // Expected: [1, 2]
    var outer = [_]i32{2};
    const outer_iter = SliceIterator(i32).init(&outer);
    var flat = FlatMap(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(outer_iter, range);

    try testing.expectEqual(1, flat.next());
    try testing.expectEqual(2, flat.next());
    try testing.expectEqual(null, flat.next());
}

test "flat_map: multiple outer elements with correct ordering" {
    // Outer: [1, 1, 2]
    // Inner for 1: [1]
    // Inner for 1: [1]
    // Inner for 2: [1, 2]
    // Expected: [1, 1, 1, 2]
    var outer = [_]i32{ 1, 1, 2 };
    const outer_iter = SliceIterator(i32).init(&outer);
    var flat = FlatMap(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(outer_iter, range);

    try testing.expectEqual(1, flat.next());
    try testing.expectEqual(1, flat.next());
    try testing.expectEqual(1, flat.next());
    try testing.expectEqual(2, flat.next());
    try testing.expectEqual(null, flat.next());
}

test "flat_map: inner iterator type transformation i32 to f64" {
    // Outer: [12]
    // Inner for 12: [1.0, 2.0]
    // Expected: [1.0, 2.0]
    var outer = [_]i32{12};
    const outer_iter = SliceIterator(i32).init(&outer);
    var flat = FlatMap(i32, f64, SliceIterator(i32), SliceIterator(f64)).init(outer_iter, digits);

    try testing.expectEqual(1.0, flat.next());
    try testing.expectEqual(2.0, flat.next());
    try testing.expectEqual(null, flat.next());
}

test "flat_map: early termination when stopping mid-iteration" {
    // Outer: [1, 2, 3]
    // Inner for 1: [1]
    // Inner for 2: [1, 2]
    // We stop before consuming inner for 3
    var outer = [_]i32{ 1, 2, 3 };
    const outer_iter = SliceIterator(i32).init(&outer);
    var flat = FlatMap(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(outer_iter, range);

    try testing.expectEqual(1, flat.next()); // from 1's range
    try testing.expectEqual(1, flat.next()); // from 2's range
    try testing.expectEqual(2, flat.next()); // from 2's range
    // Stop here (don't consume 3's range)
}

test "flat_map: stress test 10 outers with 10 inners each" {
    // Create outer array [1, 2, 3, ..., 10]
    var outer: [10]i32 = undefined;
    for (0..10) |i| {
        outer[i] = @intCast(i + 1);
    }

    const outer_iter = SliceIterator(i32).init(&outer);
    var flat = FlatMap(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(outer_iter, replicateSlice);

    var count: usize = 0;
    while (flat.next()) |_| {
        count += 1;
    }

    // Each i produces i copies: 1 + 2 + 3 + ... + 10 = 55
    try testing.expectEqual(55, count);
}

test "flat_map: different inner sizes [1 elem, 3 elems, 0 elems, 5 elems]" {
    // For this test, we create custom outer that uses fixed inner sizes
    // Outer: [1, 3, 0, 5]
    // Expected: 1 + 3 + 0 + 5 = 9 elements
    var outer = [_]i32{ 1, 3, 0, 5 };
    const outer_iter = SliceIterator(i32).init(&outer);
    var flat = FlatMap(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(outer_iter, replicateSlice);

    var count: usize = 0;
    while (flat.next()) |_| {
        count += 1;
    }

    try testing.expectEqual(9, count);
}

test "flat_map: exhaustion requires multiple null calls" {
    var outer = [_]i32{1};
    const outer_iter = SliceIterator(i32).init(&outer);
    var flat = FlatMap(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(outer_iter, range);

    try testing.expectEqual(1, flat.next());
    try testing.expectEqual(null, flat.next());
    try testing.expectEqual(null, flat.next());
    try testing.expectEqual(null, flat.next());
}

test "flat_map: state isolation between independent flatmaps" {
    var outer1 = [_]i32{ 1, 2 };
    var outer2 = [_]i32{ 2, 1 };

    const iter1 = SliceIterator(i32).init(&outer1);
    var flat1 = FlatMap(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(iter1, range);

    const iter2 = SliceIterator(i32).init(&outer2);
    var flat2 = FlatMap(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(iter2, range);

    // Interleave consumption
    try testing.expectEqual(1, flat1.next()); // 1's range
    try testing.expectEqual(1, flat2.next()); // 2's range
    try testing.expectEqual(1, flat1.next()); // 2's range
    try testing.expectEqual(2, flat2.next()); // 2's range
    try testing.expectEqual(2, flat1.next()); // 2's range
    try testing.expectEqual(1, flat2.next()); // 1's range
    try testing.expectEqual(null, flat1.next());
    try testing.expectEqual(null, flat2.next());
}

test "flat_map: correct ordering across outer boundaries" {
    // Outer: [2, 2]
    // Inner for 2: [1, 2]
    // Inner for 2: [1, 2]
    // Expected: [1, 2, 1, 2]
    var outer = [_]i32{ 2, 2 };
    const outer_iter = SliceIterator(i32).init(&outer);
    var flat = FlatMap(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(outer_iter, range);

    try testing.expectEqual(1, flat.next());
    try testing.expectEqual(2, flat.next());
    try testing.expectEqual(1, flat.next());
    try testing.expectEqual(2, flat.next());
    try testing.expectEqual(null, flat.next());
}

test "flat_map: zero-cost abstraction manual nested loop equivalence" {
    // Manual nested loop
    var outer = [_]i32{ 1, 2, 3 };
    var manual_results: [6]i32 = undefined;
    var manual_count: usize = 0;

    for (outer) |o| {
        var i: i32 = 1;
        while (i <= o) : (i += 1) {
            manual_results[manual_count] = i;
            manual_count += 1;
        }
    }

    // FlatMap equivalent
    const outer_iter = SliceIterator(i32).init(&outer);
    var flat = FlatMap(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(outer_iter, range);
    var flat_results: [6]i32 = undefined;
    var flat_count: usize = 0;

    while (flat.next()) |val| {
        flat_results[flat_count] = val;
        flat_count += 1;
    }

    // Verify they match
    try testing.expectEqual(manual_count, flat_count);
    for (0..flat_count) |i| {
        try testing.expectEqual(manual_results[i], flat_results[i]);
    }
}

test "flat_map: handles single element outer with single element inner" {
    var outer = [_]i32{1};
    const outer_iter = SliceIterator(i32).init(&outer);
    var flat = FlatMap(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(outer_iter, range);

    try testing.expectEqual(1, flat.next());
    try testing.expectEqual(null, flat.next());
}

test "flat_map: multiple outers each with large inner" {
    // Outer: [5, 5]
    // Each 5 produces [1,2,3,4,5]
    // Expected: 10 elements total
    var outer = [_]i32{ 5, 5 };
    const outer_iter = SliceIterator(i32).init(&outer);
    var flat = FlatMap(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(outer_iter, range);

    var count: usize = 0;
    while (flat.next()) |_| {
        count += 1;
    }

    try testing.expectEqual(10, count);
}

test "flat_map: verify element values match expected pattern" {
    // Outer: [1, 2]
    // Expected: [1, 1, 2]
    var outer = [_]i32{ 1, 2 };
    const outer_iter = SliceIterator(i32).init(&outer);
    var flat = FlatMap(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(outer_iter, range);

    const expected = [_]i32{ 1, 1, 2 };
    for (expected) |exp| {
        try testing.expectEqual(exp, flat.next());
    }
    try testing.expectEqual(null, flat.next());
}

test "flat_map: transition between inner iterators maintains state correctly" {
    // Outer: [3, 2, 1]
    // Inner for 3: [1, 2, 3]
    // Inner for 2: [1, 2]
    // Inner for 1: [1]
    // Expected: [1, 2, 3, 1, 2, 1]
    var outer = [_]i32{ 3, 2, 1 };
    const outer_iter = SliceIterator(i32).init(&outer);
    var flat = FlatMap(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(outer_iter, range);

    var results: [6]i32 = undefined;
    for (0..6) |i| {
        results[i] = flat.next() orelse 0;
    }

    try testing.expectEqual(1, results[0]);
    try testing.expectEqual(2, results[1]);
    try testing.expectEqual(3, results[2]);
    try testing.expectEqual(1, results[3]);
    try testing.expectEqual(2, results[4]);
    try testing.expectEqual(1, results[5]);
    try testing.expectEqual(null, flat.next());
}
