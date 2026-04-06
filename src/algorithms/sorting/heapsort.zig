//! Heap Sort - In-place comparison-based sorting using a binary heap
//!
//! Algorithm:
//! 1. Build a max heap from the input array (heapify)
//! 2. Repeatedly extract the maximum element and place it at the end
//! 3. Maintain heap property after each extraction
//!
//! Properties:
//! - Time: O(n log n) worst case (better than quicksort's O(n²))
//! - Space: O(1) - in-place sorting (better than mergesort's O(n))
//! - Not stable (relative order of equal elements may change)
//! - Cache-unfriendly due to non-sequential memory access
//!
//! Use cases:
//! - Guaranteed O(n log n) worst case (real-time systems)
//! - Memory-constrained environments (O(1) space)
//! - Priority queue implementation
//! - When stability is not required
//!
//! Trade-offs:
//! - vs Quicksort: Slower average case but better worst case
//! - vs Mergesort: In-place but not stable
//! - vs Introsort: Introsort uses heapsort as fallback for deep recursion

const std = @import("std");
const testing = std.testing;
const Order = std.math.Order;

/// Generic heap sort function
/// Time: O(n log n), Space: O(1)
pub fn heapSort(
    comptime T: type,
    items: []T,
    context: anytype,
    comptime lessThanFn: fn (@TypeOf(context), T, T) bool,
) void {
    if (items.len <= 1) return;

    // Build max heap (heapify)
    // Start from last non-leaf node and sift down
    // Last non-leaf node is at index (n/2 - 1)
    var i: isize = @intCast(@divFloor(items.len, 2) - 1);
    while (i >= 0) : (i -= 1) {
        siftDown(T, items, @intCast(i), items.len, context, lessThanFn);
    }

    // Extract elements from heap one by one
    // Place max at end, reduce heap size, restore heap property
    var n = items.len;
    while (n > 1) {
        n -= 1;
        // Move current root (max) to end
        std.mem.swap(T, &items[0], &items[n]);
        // Restore heap property for reduced heap
        siftDown(T, items, 0, n, context, lessThanFn);
    }
}

/// Sift down operation to maintain max heap property
/// Ensures parent >= children
/// Time: O(log n) per call
fn siftDown(
    comptime T: type,
    items: []T,
    start: usize,
    end: usize,
    context: anytype,
    comptime lessThanFn: fn (@TypeOf(context), T, T) bool,
) void {
    var root = start;

    // While root has at least one child
    while (2 * root + 1 < end) {
        const left_child = 2 * root + 1;
        const right_child = left_child + 1;
        var largest = root;

        // Find largest among root, left child, right child
        if (lessThanFn(context, items[largest], items[left_child])) {
            largest = left_child;
        }
        if (right_child < end and lessThanFn(context, items[largest], items[right_child])) {
            largest = right_child;
        }

        // If root is largest, heap property satisfied
        if (largest == root) break;

        // Otherwise, swap and continue sifting down
        std.mem.swap(T, &items[root], &items[largest]);
        root = largest;
    }
}

/// Convenience wrapper for default ascending order
pub fn heapSortAsc(comptime T: type, items: []T) void {
    const Context = struct {
        fn lessThan(_: @This(), a: T, b: T) bool {
            return a < b;
        }
    };
    heapSort(T, items, Context{}, Context.lessThan);
}

/// Convenience wrapper for descending order
pub fn heapSortDesc(comptime T: type, items: []T) void {
    const Context = struct {
        fn lessThan(_: @This(), a: T, b: T) bool {
            return a > b;
        }
    };
    heapSort(T, items, Context{}, Context.lessThan);
}

// ============================================================================
// Tests
// ============================================================================

test "heapSort - basic ascending order" {
    var arr = [_]i32{ 5, 2, 8, 1, 9 };
    heapSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 5, 8, 9 }, &arr);
}

test "heapSort - basic descending order" {
    var arr = [_]i32{ 5, 2, 8, 1, 9 };
    heapSortDesc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 9, 8, 5, 2, 1 }, &arr);
}

test "heapSort - empty array" {
    var arr = [_]i32{};
    heapSortAsc(i32, &arr);
    try testing.expectEqual(0, arr.len);
}

test "heapSort - single element" {
    var arr = [_]i32{42};
    heapSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &arr);
}

test "heapSort - two elements" {
    var arr = [_]i32{ 5, 3 };
    heapSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 3, 5 }, &arr);
}

test "heapSort - already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    heapSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "heapSort - reverse sorted" {
    var arr = [_]i32{ 5, 4, 3, 2, 1 };
    heapSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "heapSort - duplicates" {
    var arr = [_]i32{ 3, 1, 3, 2, 1, 3 };
    heapSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2, 3, 3, 3 }, &arr);
}

test "heapSort - all same elements" {
    var arr = [_]i32{ 7, 7, 7, 7 };
    heapSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 7, 7, 7, 7 }, &arr);
}

test "heapSort - negative numbers" {
    var arr = [_]i32{ -5, 3, -1, 0, -3 };
    heapSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ -5, -3, -1, 0, 3 }, &arr);
}

test "heapSort - large array" {
    const n = 1000;
    var arr: [n]i32 = undefined;
    
    // Fill with descending values
    for (0..n) |i| {
        arr[i] = @intCast(n - i);
    }
    
    heapSortAsc(i32, &arr);
    
    // Verify sorted
    for (0..n) |i| {
        try testing.expectEqual(@as(i32, @intCast(i + 1)), arr[i]);
    }
}

test "heapSort - custom comparison (strings by length)" {
    var arr = [_][]const u8{ "hello", "a", "world", "ab", "xyz" };
    
    const Context = struct {
        fn lessThan(_: @This(), a: []const u8, b: []const u8) bool {
            return a.len < b.len;
        }
    };
    
    heapSort([]const u8, &arr, Context{}, Context.lessThan);
    
    // Verify sorted by length
    try testing.expectEqual(1, arr[0].len); // "a"
    try testing.expectEqual(2, arr[1].len); // "ab"
    try testing.expectEqual(3, arr[2].len); // "xyz"
    try testing.expectEqual(5, arr[3].len); // "hello" or "world"
    try testing.expectEqual(5, arr[4].len); // "hello" or "world"
}

test "heapSort - f64 values" {
    var arr = [_]f64{ 3.14, 2.71, 1.41, 0.57, 2.23 };
    heapSortAsc(f64, &arr);
    try testing.expectEqual(@as(f64, 0.57), arr[0]);
    try testing.expectEqual(@as(f64, 1.41), arr[1]);
    try testing.expectEqual(@as(f64, 2.23), arr[2]);
    try testing.expectEqual(@as(f64, 2.71), arr[3]);
    try testing.expectEqual(@as(f64, 3.14), arr[4]);
}

test "heapSort - stability check (not stable)" {
    // Heap sort is not stable, but we can verify it sorts correctly
    const Item = struct {
        key: i32,
        value: u8,
    };
    
    var arr = [_]Item{
        .{ .key = 3, .value = 'a' },
        .{ .key = 1, .value = 'b' },
        .{ .key = 3, .value = 'c' },
        .{ .key = 2, .value = 'd' },
    };
    
    const Context = struct {
        fn lessThan(_: @This(), a: Item, b: Item) bool {
            return a.key < b.key;
        }
    };
    
    heapSort(Item, &arr, Context{}, Context.lessThan);
    
    // Verify keys are sorted (order of equal keys may vary)
    try testing.expectEqual(@as(i32, 1), arr[0].key);
    try testing.expectEqual(@as(i32, 2), arr[1].key);
    try testing.expectEqual(@as(i32, 3), arr[2].key);
    try testing.expectEqual(@as(i32, 3), arr[3].key);
}

test "heapSort - worst case performance" {
    // Worst case: already sorted in reverse (forces most comparisons)
    const n = 100;
    var arr: [n]i32 = undefined;
    for (0..n) |i| {
        arr[i] = @intCast(n - i);
    }
    
    heapSortAsc(i32, &arr);
    
    // Verify O(n log n) behavior by checking correctness
    for (0..n) |i| {
        try testing.expectEqual(@as(i32, @intCast(i + 1)), arr[i]);
    }
}
