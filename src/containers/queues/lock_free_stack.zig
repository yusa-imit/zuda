//! Lock-free stack using Treiber's algorithm.
//!
//! A concurrent stack that supports thread-safe push and pop operations without locks.
//! Uses compare-and-swap (CAS) operations for atomic modifications.
//!
//! ## Algorithm
//!
//! Treiber's stack (1986) is a linked list where the head pointer is updated atomically:
//! - Push: Create new node, atomically swing head pointer to point to it
//! - Pop: Atomically read head, swing head to next node
//!
//! ## Properties
//!
//! - **Lock-free**: At least one thread makes progress in a finite number of steps
//! - **Linearizable**: Operations appear to occur atomically at some point
//! - **ABA-safe**: Uses tagged pointers or version counters to prevent ABA problem
//!
//! ## ABA Problem
//!
//! The ABA problem occurs when:
//! 1. Thread 1 reads head as A
//! 2. Thread 2 pops A, pushes B, pops B, pushes A (same address)
//! 3. Thread 1's CAS succeeds incorrectly (head looks unchanged but state changed)
//!
//! Solution: Use a version counter that increments on every modification.
//!
//! ## Performance
//!
//! - Push: O(1) expected, may retry on contention
//! - Pop: O(1) expected, may retry on contention
//! - Space: O(n) where n is number of elements
//!
//! ## References
//!
//! R. Kent Treiber. "Systems Programming: Coping with Parallelism" (1986)

const std = @import("std");
const Allocator = std.mem.Allocator;
const AtomicOrder = std.builtin.AtomicOrder;

/// Lock-free stack using Treiber's algorithm.
///
/// Thread-safe LIFO structure supporting concurrent push/pop without locks.
///
/// Example:
/// ```zig
/// var stack = LockFreeStack(i32).init(allocator);
/// defer stack.deinit();
///
/// try stack.push(42);
/// try stack.push(17);
///
/// const val = stack.pop(); // Some(17)
/// ```
pub fn LockFreeStack(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Node in the linked stack.
        const Node = struct {
            data: T,
            next: ?*Node,
        };

        /// Tagged pointer to prevent ABA problem.
        ///
        /// Packs pointer and version counter into a single 128-bit value
        /// to enable atomic compare-and-swap.
        const TaggedPtr = struct {
            ptr: ?*Node,
            tag: u64, // Version counter

            fn pack(self: TaggedPtr) u128 {
                const ptr_val: u64 = @intFromPtr(self.ptr);
                return (@as(u128, self.tag) << 64) | ptr_val;
            }

            fn unpack(val: u128) TaggedPtr {
                const ptr_val: u64 = @truncate(val);
                const tag: u64 = @truncate(val >> 64);
                return .{
                    .ptr = @ptrFromInt(ptr_val),
                    .tag = tag,
                };
            }
        };

        allocator: Allocator,
        head: std.atomic.Value(u128), // Atomic tagged pointer

        /// Initialize an empty stack.
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator) Self {
            const initial_tagged = TaggedPtr{ .ptr = null, .tag = 0 };
            return .{
                .allocator = allocator,
                .head = std.atomic.Value(u128).init(initial_tagged.pack()),
            };
        }

        /// Free all nodes in the stack.
        ///
        /// WARNING: Not thread-safe. Ensure no concurrent operations before calling.
        ///
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            var current = TaggedPtr.unpack(self.head.load(.monotonic)).ptr;
            while (current) |node| {
                const next = node.next;
                self.allocator.destroy(node);
                current = next;
            }
        }

        /// Push a value onto the stack.
        ///
        /// Thread-safe. May retry on contention but guaranteed to succeed.
        ///
        /// Time: O(1) expected | Space: O(1)
        pub fn push(self: *Self, value: T) Allocator.Error!void {
            const new_node = try self.allocator.create(Node);
            new_node.data = value;

            while (true) {
                const old_head_packed = self.head.load(.acquire);
                const old_head = TaggedPtr.unpack(old_head_packed);

                new_node.next = old_head.ptr;

                const new_head = TaggedPtr{
                    .ptr = new_node,
                    .tag = old_head.tag +% 1, // Increment version counter
                };

                if (self.head.cmpxchgWeak(
                    old_head_packed,
                    new_head.pack(),
                    .release,
                    .acquire,
                )) |_| {
                    // CAS failed, retry
                    continue;
                } else {
                    // CAS succeeded
                    return;
                }
            }
        }

        /// Pop a value from the stack.
        ///
        /// Thread-safe. Returns null if stack is empty.
        ///
        /// Time: O(1) expected | Space: O(1)
        pub fn pop(self: *Self) ?T {
            while (true) {
                const old_head_packed = self.head.load(.acquire);
                const old_head = TaggedPtr.unpack(old_head_packed);

                const node = old_head.ptr orelse return null;

                const new_head = TaggedPtr{
                    .ptr = node.next,
                    .tag = old_head.tag +% 1, // Increment version counter
                };

                if (self.head.cmpxchgWeak(
                    old_head_packed,
                    new_head.pack(),
                    .release,
                    .acquire,
                )) |_| {
                    // CAS failed, retry
                    continue;
                } else {
                    // CAS succeeded
                    const value = node.data;
                    self.allocator.destroy(node);
                    return value;
                }
            }
        }

        /// Check if the stack is empty.
        ///
        /// Note: Result may be stale immediately after return in concurrent context.
        ///
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            const head_packed = self.head.load(.acquire);
            const head = TaggedPtr.unpack(head_packed);
            return head.ptr == null;
        }

        /// Get the current size of the stack.
        ///
        /// WARNING: O(n) traversal. Result may be stale in concurrent context.
        /// For performance-critical code, avoid this in hot paths.
        ///
        /// Time: O(n) | Space: O(1)
        pub fn count(self: *const Self) usize {
            var current = TaggedPtr.unpack(self.head.load(.acquire)).ptr;
            var n: usize = 0;
            while (current) |node| : (current = node.next) {
                n += 1;
            }
            return n;
        }

        /// Peek at the top value without removing it.
        ///
        /// Returns null if stack is empty.
        /// Note: In concurrent context, the top value may change immediately.
        ///
        /// Time: O(1) | Space: O(1)
        pub fn peek(self: *const Self) ?T {
            const head_packed = self.head.load(.acquire);
            const head = TaggedPtr.unpack(head_packed);
            return if (head.ptr) |node| node.data else null;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "LockFreeStack: init and deinit" {
    var stack = LockFreeStack(i32).init(testing.allocator);
    defer stack.deinit();

    try testing.expect(stack.isEmpty());
    try testing.expectEqual(@as(usize, 0), stack.count());
}

test "LockFreeStack: push and pop single element" {
    var stack = LockFreeStack(i32).init(testing.allocator);
    defer stack.deinit();

    try stack.push(42);
    try testing.expect(!stack.isEmpty());
    try testing.expectEqual(@as(usize, 1), stack.count());

    const val = stack.pop();
    try testing.expectEqual(@as(?i32, 42), val);
    try testing.expect(stack.isEmpty());
}

test "LockFreeStack: LIFO ordering" {
    var stack = LockFreeStack(i32).init(testing.allocator);
    defer stack.deinit();

    try stack.push(1);
    try stack.push(2);
    try stack.push(3);

    try testing.expectEqual(@as(?i32, 3), stack.pop());
    try testing.expectEqual(@as(?i32, 2), stack.pop());
    try testing.expectEqual(@as(?i32, 1), stack.pop());
    try testing.expectEqual(@as(?i32, null), stack.pop());
}

test "LockFreeStack: pop from empty stack" {
    var stack = LockFreeStack(i32).init(testing.allocator);
    defer stack.deinit();

    try testing.expectEqual(@as(?i32, null), stack.pop());
}

test "LockFreeStack: peek" {
    var stack = LockFreeStack(i32).init(testing.allocator);
    defer stack.deinit();

    try testing.expectEqual(@as(?i32, null), stack.peek());

    try stack.push(42);
    try testing.expectEqual(@as(?i32, 42), stack.peek());
    try testing.expectEqual(@as(?i32, 42), stack.peek()); // Peek doesn't remove

    _ = stack.pop();
    try testing.expectEqual(@as(?i32, null), stack.peek());
}

test "LockFreeStack: count" {
    var stack = LockFreeStack(i32).init(testing.allocator);
    defer stack.deinit();

    try testing.expectEqual(@as(usize, 0), stack.count());

    try stack.push(1);
    try testing.expectEqual(@as(usize, 1), stack.count());

    try stack.push(2);
    try testing.expectEqual(@as(usize, 2), stack.count());

    _ = stack.pop();
    try testing.expectEqual(@as(usize, 1), stack.count());

    _ = stack.pop();
    try testing.expectEqual(@as(usize, 0), stack.count());
}

test "LockFreeStack: multiple push and pop" {
    var stack = LockFreeStack(i32).init(testing.allocator);
    defer stack.deinit();

    const n = 100;
    var i: i32 = 0;
    while (i < n) : (i += 1) {
        try stack.push(i);
    }

    try testing.expectEqual(@as(usize, n), stack.count());

    i = n - 1;
    while (i >= 0) : (i -= 1) {
        const val = stack.pop().?;
        try testing.expectEqual(i, val);
    }

    try testing.expect(stack.isEmpty());
}

test "LockFreeStack: stress test with allocator" {
    var stack = LockFreeStack(i32).init(testing.allocator);
    defer stack.deinit();

    // Push 1000 elements
    var i: i32 = 0;
    while (i < 1000) : (i += 1) {
        try stack.push(i);
    }

    // Pop all
    i = 999;
    while (i >= 0) : (i -= 1) {
        const val = stack.pop();
        try testing.expectEqual(@as(?i32, i), val);
    }

    try testing.expect(stack.isEmpty());
}

test "LockFreeStack: interleaved push and pop" {
    var stack = LockFreeStack(i32).init(testing.allocator);
    defer stack.deinit();

    try stack.push(1);
    try stack.push(2);
    try testing.expectEqual(@as(?i32, 2), stack.pop());

    try stack.push(3);
    try stack.push(4);
    try testing.expectEqual(@as(?i32, 4), stack.pop());
    try testing.expectEqual(@as(?i32, 3), stack.pop());
    try testing.expectEqual(@as(?i32, 1), stack.pop());
    try testing.expectEqual(@as(?i32, null), stack.pop());
}

test "LockFreeStack: with complex types" {
    const Point = struct { x: i32, y: i32 };
    var stack = LockFreeStack(Point).init(testing.allocator);
    defer stack.deinit();

    try stack.push(.{ .x = 1, .y = 2 });
    try stack.push(.{ .x = 3, .y = 4 });

    const p2 = stack.pop().?;
    try testing.expectEqual(@as(i32, 3), p2.x);
    try testing.expectEqual(@as(i32, 4), p2.y);

    const p1 = stack.pop().?;
    try testing.expectEqual(@as(i32, 1), p1.x);
    try testing.expectEqual(@as(i32, 2), p1.y);
}

test "LockFreeStack: memory leak detection" {
    var stack = LockFreeStack(i32).init(testing.allocator);
    defer stack.deinit();

    // Push and pop many times to detect leaks
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        try stack.push(@intCast(i));
    }

    while (!stack.isEmpty()) {
        _ = stack.pop();
    }

    // Second round
    i = 0;
    while (i < 100) : (i += 1) {
        try stack.push(@intCast(i));
    }
}

test "LockFreeStack: version counter wraparound safety" {
    var stack = LockFreeStack(i32).init(testing.allocator);
    defer stack.deinit();

    // Simulate high version counter (near wraparound)
    // This tests that wraparound doesn't break correctness
    const initial_tag: u64 = std.math.maxInt(u64) - 10;
    const tagged = LockFreeStack(i32).TaggedPtr{
        .ptr = null,
        .tag = initial_tag,
    };
    stack.head.store(tagged.pack(), .release);

    // Perform operations that will wrap the counter
    var i: usize = 0;
    while (i < 20) : (i += 1) {
        try stack.push(@intCast(i));
    }

    // Verify LIFO order is maintained
    i = 20;
    while (i > 0) {
        i -= 1;
        const val = stack.pop();
        try testing.expectEqual(@as(?i32, @intCast(i)), val);
    }
}
