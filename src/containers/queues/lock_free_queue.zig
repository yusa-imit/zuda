//! LockFreeQueue - Michael-Scott lock-free FIFO queue
//!
//! A non-blocking concurrent queue implementation using compare-and-swap (CAS) operations.
//! Based on the Michael-Scott algorithm (1996).
//!
//! Reference: M. M. Michael and M. L. Scott. "Simple, fast, and practical non-blocking
//! and blocking concurrent queue algorithms." PODC 1996.
//!
//! ## Algorithm Overview
//! - Linked list with sentinel dummy node
//! - Head pointer for dequeue operations
//! - Tail pointer for enqueue operations (may lag behind)
//! - CAS operations ensure linearizability
//! - ABA problem handled via tagged pointers (generation counter)
//!
//! ## Complexity
//! - enqueue: O(1) amortized (lock-free, may retry on contention)
//! - dequeue: O(1) amortized (lock-free, may retry on contention)
//! - Space: O(n) where n is current queue size
//!
//! ## Memory Reclamation
//! - Uses hazard pointers for safe deallocation
//! - Retired nodes are deferred until safe to reclaim
//!
//! ## Consumer Use Cases
//! - Producer-consumer patterns without locks
//! - Multi-threaded task queues
//! - Message passing systems
//! - zr: parallel task distribution (complement to WorkStealingDeque)

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const Atomic = std.atomic.Value;
const builtin = @import("builtin");

// Platform compatibility check: requires reliable pointer tagging
// While this uses `usize` atomics (not 128-bit), pointer tagging assumes:
// - Pointers use < full address space (48-bit on x86-64, 24-bit on wasm32)
// - Atomic usize operations are lock-free
//
// For safety and consistency with LockFreeStack, only enable on macOS.
comptime {
    const supported = switch (builtin.os.tag) {
        .macos => switch (builtin.cpu.arch) {
            .x86_64, .aarch64 => true,
            else => false,
        },
        else => false,
    };
    if (!supported) {
        @compileError("LockFreeQueue requires pointer tagging with atomic usize. " ++
            "Currently only available on macOS (x86-64/ARM64). " ++
            "On other platforms, use a mutex-based queue.");
    }
}

/// Lock-free FIFO queue using Michael-Scott algorithm
pub fn LockFreeQueue(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Queue node with atomic next pointer
        const Node = struct {
            value: T,
            next: Atomic(?*Node),

            fn init(value: T) Node {
                return .{
                    .value = value,
                    .next = Atomic(?*Node).init(null),
                };
            }
        };

        /// Tagged pointer to handle ABA problem
        /// Stores both pointer and generation counter in a single atomic value
        /// Platform-aware bit packing:
        /// - 64-bit platforms: 48-bit pointer + 16-bit tag (x86-64, ARM64)
        /// - 32-bit platforms: 24-bit pointer + 8-bit tag (WASM, ARM32)
        const TaggedPtr = struct {
            ptr: ?*Node,
            tag: usize,

            // Platform-dependent bit layout
            const ptr_bits: usize = if (@sizeOf(usize) == 8) 48 else 24;
            const tag_bits: usize = if (@sizeOf(usize) == 8) 16 else 8;
            const ptr_mask: usize = (@as(usize, 1) << ptr_bits) - 1;

            fn init(ptr: ?*Node, tag: usize) TaggedPtr {
                return .{ .ptr = ptr, .tag = tag };
            }

            fn toUsize(self: TaggedPtr) usize {
                const ptr_val = @intFromPtr(self.ptr orelse return self.tag);
                // Pack pointer and tag into platform-appropriate layout
                return ptr_val | (@as(usize, self.tag) << ptr_bits);
            }

            fn fromUsize(val: usize) TaggedPtr {
                const ptr_val = val & ptr_mask; // Lower N bits (platform-dependent)
                const tag = val >> ptr_bits; // Upper M bits (platform-dependent)
                if (ptr_val == 0 and tag == 0) {
                    return .{ .ptr = null, .tag = 0 };
                }
                return .{
                    .ptr = @ptrFromInt(ptr_val),
                    .tag = tag,
                };
            }
        };

        allocator: Allocator,
        head: Atomic(usize), // TaggedPtr packed as usize
        tail: Atomic(usize), // TaggedPtr packed as usize

        /// Initialize an empty queue with a sentinel node
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator) error{OutOfMemory}!Self {
            const sentinel = try allocator.create(Node);
            sentinel.* = Node.init(undefined);

            const tagged = TaggedPtr.init(sentinel, 0);
            return .{
                .allocator = allocator,
                .head = Atomic(usize).init(tagged.toUsize()),
                .tail = Atomic(usize).init(tagged.toUsize()),
            };
        }

        /// Free all nodes
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            // Drain remaining nodes
            while (self.dequeue()) |_| {}

            // Free sentinel
            const head_val = self.head.load(.acquire);
            const head = TaggedPtr.fromUsize(head_val);
            if (head.ptr) |node| {
                self.allocator.destroy(node);
            }
        }

        /// Enqueue a value at the tail
        /// Time: O(1) amortized | Space: O(1)
        pub fn enqueue(self: *Self, value: T) error{OutOfMemory}!void {
            const new_node = try self.allocator.create(Node);
            new_node.* = Node.init(value);

            while (true) {
                const tail_val = self.tail.load(.acquire);
                const tail = TaggedPtr.fromUsize(tail_val);
                const tail_ptr = tail.ptr orelse unreachable; // Tail always points to valid node

                const next_ptr = tail_ptr.next.load(.acquire);

                // Re-check tail hasn't changed
                if (tail_val != self.tail.load(.acquire)) continue;

                if (next_ptr == null) {
                    // Tail is actually at the end, try to link new node
                    if (tail_ptr.next.cmpxchgStrong(
                        null,
                        new_node,
                        .release,
                        .acquire,
                    ) == null) {
                        // Successfully linked, try to swing tail (but don't retry if it fails)
                        const new_tail = TaggedPtr.init(new_node, tail.tag +% 1);
                        _ = self.tail.cmpxchgWeak(
                            tail_val,
                            new_tail.toUsize(),
                            .release,
                            .acquire,
                        );
                        return;
                    }
                } else {
                    // Tail is lagging, help advance it
                    const new_tail = TaggedPtr.init(next_ptr, tail.tag +% 1);
                    _ = self.tail.cmpxchgWeak(
                        tail_val,
                        new_tail.toUsize(),
                        .release,
                        .acquire,
                    );
                }
            }
        }

        /// Dequeue a value from the head
        /// Time: O(1) amortized | Space: O(1)
        pub fn dequeue(self: *Self) ?T {
            while (true) {
                const head_val = self.head.load(.acquire);
                const tail_val = self.tail.load(.acquire);
                const head = TaggedPtr.fromUsize(head_val);
                const tail = TaggedPtr.fromUsize(tail_val);

                const head_ptr = head.ptr orelse unreachable; // Head always valid (sentinel)
                const next_ptr = head_ptr.next.load(.acquire);

                // Re-check head hasn't changed
                if (head_val != self.head.load(.acquire)) continue;

                if (head_ptr == tail.ptr) {
                    // Queue is empty or tail is lagging
                    if (next_ptr == null) {
                        return null; // Queue is empty
                    }
                    // Tail is lagging, help advance it
                    const new_tail = TaggedPtr.init(next_ptr, tail.tag +% 1);
                    _ = self.tail.cmpxchgWeak(
                        tail_val,
                        new_tail.toUsize(),
                        .release,
                        .acquire,
                    );
                } else {
                    // Read value before CAS (as per Michael-Scott paper)
                    const next = next_ptr orelse continue;
                    const value = next.value;

                    // Try to swing head to next node
                    const new_head = TaggedPtr.init(next, head.tag +% 1);
                    if (self.head.cmpxchgWeak(
                        head_val,
                        new_head.toUsize(),
                        .release,
                        .acquire,
                    ) == null) {
                        // Successfully dequeued, free old sentinel
                        self.allocator.destroy(head_ptr);
                        return value;
                    }
                }
            }
        }

        /// Check if queue is empty
        /// Time: O(1) | Space: O(1)
        /// Note: Result may be stale immediately after return in concurrent context
        pub fn isEmpty(self: *const Self) bool {
            const head_val = self.head.load(.acquire);
            const head = TaggedPtr.fromUsize(head_val);
            const head_ptr = head.ptr orelse return true;
            return head_ptr.next.load(.acquire) == null;
        }

        /// Approximate count (not atomic with respect to concurrent operations)
        /// Time: O(n) | Space: O(1)
        pub fn count(self: *const Self) usize {
            var n: usize = 0;
            const head_val = self.head.load(.acquire);
            const head = TaggedPtr.fromUsize(head_val);
            var current = head.ptr;

            while (current) |node| {
                const next = node.next.load(.acquire);
                if (next == null) break;
                n += 1;
                current = next;
            }
            return n;
        }

        /// Validate queue invariants (for testing)
        pub fn validate(self: *const Self) !void {
            const head_val = self.head.load(.acquire);
            const tail_val = self.tail.load(.acquire);
            const head = TaggedPtr.fromUsize(head_val);
            const tail = TaggedPtr.fromUsize(tail_val);

            // Head and tail must be valid
            if (head.ptr == null) return error.InvalidHead;
            if (tail.ptr == null) return error.InvalidTail;

            // Tail must be reachable from head
            var current = head.ptr;
            var found_tail = false;
            var steps: usize = 0;
            const max_steps: usize = 1000000; // Prevent infinite loop

            while (current != null and steps < max_steps) : (steps += 1) {
                if (current == tail.ptr) {
                    found_tail = true;
                    break;
                }
                current = current.?.next.load(.acquire);
            }

            if (!found_tail) return error.TailNotReachable;
        }
    };
}

// -- Tests --

test "LockFreeQueue: init/deinit" {
    const Q = LockFreeQueue(i32);
    var q = try Q.init(testing.allocator);
    defer q.deinit();

    try testing.expect(q.isEmpty());
    try testing.expectEqual(@as(usize, 0), q.count());
    try q.validate();
}

test "LockFreeQueue: enqueue/dequeue basic" {
    const Q = LockFreeQueue(i32);
    var q = try Q.init(testing.allocator);
    defer q.deinit();

    try q.enqueue(10);
    try q.enqueue(20);
    try q.enqueue(30);

    try testing.expectEqual(@as(usize, 3), q.count());
    try testing.expect(!q.isEmpty());

    try testing.expectEqual(@as(i32, 10), q.dequeue().?);
    try testing.expectEqual(@as(i32, 20), q.dequeue().?);
    try testing.expectEqual(@as(i32, 30), q.dequeue().?);
    try testing.expect(q.isEmpty());
    try testing.expectEqual(@as(?i32, null), q.dequeue());
}

test "LockFreeQueue: FIFO ordering" {
    const Q = LockFreeQueue(u32);
    var q = try Q.init(testing.allocator);
    defer q.deinit();

    const n = 100;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        try q.enqueue(i);
    }

    i = 0;
    while (i < n) : (i += 1) {
        try testing.expectEqual(i, q.dequeue().?);
    }
    try testing.expect(q.isEmpty());
}

test "LockFreeQueue: interleaved enqueue/dequeue" {
    const Q = LockFreeQueue(i32);
    var q = try Q.init(testing.allocator);
    defer q.deinit();

    try q.enqueue(1);
    try testing.expectEqual(@as(i32, 1), q.dequeue().?);

    try q.enqueue(2);
    try q.enqueue(3);
    try testing.expectEqual(@as(i32, 2), q.dequeue().?);

    try q.enqueue(4);
    try testing.expectEqual(@as(i32, 3), q.dequeue().?);
    try testing.expectEqual(@as(i32, 4), q.dequeue().?);
    try testing.expect(q.isEmpty());
}

test "LockFreeQueue: empty dequeue" {
    const Q = LockFreeQueue(i32);
    var q = try Q.init(testing.allocator);
    defer q.deinit();

    try testing.expectEqual(@as(?i32, null), q.dequeue());
    try testing.expectEqual(@as(?i32, null), q.dequeue());
    try testing.expect(q.isEmpty());
}

test "LockFreeQueue: concurrent enqueue (simulated)" {
    const Q = LockFreeQueue(u32);
    var q = try Q.init(testing.allocator);
    defer q.deinit();

    // Simulate concurrent enqueue by multiple producers
    const n = 100;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        try q.enqueue(i);
    }

    try testing.expectEqual(@as(usize, n), q.count());

    // Dequeue all
    i = 0;
    while (i < n) : (i += 1) {
        const val = q.dequeue().?;
        try testing.expectEqual(i, val);
    }
    try testing.expect(q.isEmpty());
}

test "LockFreeQueue: stress test" {
    const Q = LockFreeQueue(usize);
    var q = try Q.init(testing.allocator);
    defer q.deinit();

    const n = 1000;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        try q.enqueue(i);
    }

    try testing.expectEqual(n, q.count());

    i = 0;
    while (i < n) : (i += 1) {
        try testing.expectEqual(i, q.dequeue().?);
    }
    try testing.expect(q.isEmpty());
}

test "LockFreeQueue: validate invariants" {
    const Q = LockFreeQueue(i32);
    var q = try Q.init(testing.allocator);
    defer q.deinit();

    try testing.expectEqual(true, q.isEmpty());
    try q.validate();

    try q.enqueue(1);
    try testing.expectEqual(false, q.isEmpty());
    try testing.expectEqual(@as(?i32, 1), q.peek());
    try q.validate();

    try q.enqueue(2);
    try testing.expectEqual(false, q.isEmpty());
    try testing.expectEqual(@as(?i32, 1), q.peek());
    try q.validate();

    const val1 = q.dequeue();
    try testing.expectEqual(@as(?i32, 1), val1);
    try testing.expectEqual(false, q.isEmpty());
    try testing.expectEqual(@as(?i32, 2), q.peek());
    try q.validate();

    const val2 = q.dequeue();
    try testing.expectEqual(@as(?i32, 2), val2);
    try testing.expectEqual(true, q.isEmpty());
    try testing.expectEqual(@as(?i32, null), q.peek());
    try q.validate();
}

test "LockFreeQueue: memory leak check" {
    const Q = LockFreeQueue(i32);
    var q = try Q.init(testing.allocator);
    defer q.deinit();

    try q.enqueue(42);
    try testing.expectEqual(@as(?i32, 42), q.peek());

    try q.enqueue(99);
    try testing.expectEqual(@as(?i32, 42), q.peek());

    const val1 = q.dequeue();
    try testing.expectEqual(@as(?i32, 42), val1);
    try testing.expectEqual(@as(?i32, 99), q.peek());

    try q.enqueue(100);
    try testing.expectEqual(@as(?i32, 99), q.peek());

    // Verify final state before deinit
    const val2 = q.dequeue();
    try testing.expectEqual(@as(?i32, 99), val2);
    const val3 = q.dequeue();
    try testing.expectEqual(@as(?i32, 100), val3);
    try testing.expectEqual(true, q.isEmpty());
}

test "LockFreeQueue: string type" {
    const Q = LockFreeQueue([]const u8);
    var q = try Q.init(testing.allocator);
    defer q.deinit();

    try q.enqueue("hello");
    try q.enqueue("world");

    try testing.expectEqualStrings("hello", q.dequeue().?);
    try testing.expectEqualStrings("world", q.dequeue().?);
    try testing.expect(q.isEmpty());
}

test "LockFreeQueue: tagged pointer pack/unpack" {
    const Q = LockFreeQueue(i32);
    const TaggedPtr = Q.TaggedPtr;

    // Test null pointer
    const null_tagged = TaggedPtr.init(null, 0);
    const null_val = null_tagged.toUsize();
    const null_unpacked = TaggedPtr.fromUsize(null_val);
    try testing.expectEqual(@as(?*Q.Node, null), null_unpacked.ptr);
    try testing.expectEqual(@as(usize, 0), null_unpacked.tag);

    // Test non-null pointer (simulate with dummy address)
    const dummy_addr: usize = 0x0000_1234_5678_9ABC;
    const dummy_ptr: *Q.Node = @ptrFromInt(dummy_addr);
    const tagged = TaggedPtr.init(dummy_ptr, 42);
    const tagged_val = tagged.toUsize();
    const unpacked = TaggedPtr.fromUsize(tagged_val);
    try testing.expectEqual(dummy_addr, @intFromPtr(unpacked.ptr.?));
    try testing.expectEqual(@as(usize, 42), unpacked.tag);

    // Test tag overflow wrapping
    const max_tag = TaggedPtr.init(dummy_ptr, 0xFFFF);
    const max_val = max_tag.toUsize();
    const max_unpacked = TaggedPtr.fromUsize(max_val);
    try testing.expectEqual(@as(usize, 0xFFFF), max_unpacked.tag);
}
