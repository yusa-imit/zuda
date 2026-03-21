const std = @import("std");

/// Work-stealing deque for efficient task distribution in parallel execution.
///
/// Based on the Chase-Lev algorithm (Dynamic Circular Work-Stealing Deque):
/// https://www.dre.vanderbilt.edu/~schmidt/PDF/work-stealing-dequeue.pdf
///
/// **Key properties**:
/// - Owner thread: push/pop from bottom (LIFO for cache locality)
/// - Stealer threads: steal from top (FIFO for load balancing)
/// - Lock-free for common case (owner pop, stealer steal)
/// - Mutex only for resize operations (rare)
///
/// **API**:
/// - `push(task)`: Owner adds task to bottom (O(1) amortized)
/// - `pop()`: Owner removes task from bottom (O(1), LIFO)
/// - `steal()`: Stealer removes task from top (O(1), FIFO)
/// - `size()`: Approximate size (concurrent snapshot)
///
/// **Consumer**: zr task runner (replaces src/exec/workstealing.zig — 130 LOC)
///
/// Time complexity:
/// - push: O(1) amortized
/// - pop: O(1)
/// - steal: O(1)
/// Space complexity: O(n) where n = number of tasks
pub fn WorkStealingDeque(comptime T: type) type {
    return struct {
        const Self = @This();
        const MIN_CAPACITY = 32;

        items: []T,
        allocator: std.mem.Allocator,
        top: std.atomic.Value(usize), // Stealers pop from top (FIFO)
        bottom: std.atomic.Value(usize), // Owner pushes/pops from bottom (LIFO)
        capacity: usize,
        mutex: std.Thread.Mutex, // Protects resize operations

        /// Initialize a new work-stealing deque.
        /// Time: O(1) | Space: O(MIN_CAPACITY)
        pub fn init(allocator: std.mem.Allocator) !Self {
            const items = try allocator.alloc(T, MIN_CAPACITY);
            return Self{
                .items = items,
                .allocator = allocator,
                .top = std.atomic.Value(usize).init(0),
                .bottom = std.atomic.Value(usize).init(0),
                .capacity = MIN_CAPACITY,
                .mutex = std.Thread.Mutex{},
            };
        }

        /// Deinitialize the deque and free all memory.
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.items);
        }

        /// Push a task to the bottom of the deque (owner thread only).
        /// Time: O(1) amortized (O(n) on resize) | Space: O(1)
        pub fn push(self: *Self, task: T) !void {
            const b = self.bottom.load(.acquire);
            const t = self.top.load(.acquire);
            const current_size = b -% t;

            // Resize if full (only owner can resize)
            if (current_size >= self.capacity - 1) {
                self.mutex.lock();
                defer self.mutex.unlock();
                try self.resize();
            }

            self.items[b % self.capacity] = task;
            // Release ensures the task write is visible before bottom increment
            self.bottom.store(b +% 1, .release);
        }

        /// Pop a task from the bottom of the deque (owner thread only).
        /// Returns null if the deque is empty.
        /// Time: O(1) | Space: O(1)
        pub fn pop(self: *Self) ?T {
            const b = self.bottom.load(.acquire) -% 1;
            self.bottom.store(b, .seq_cst); // Upgraded to seq_cst to act as fence

            const t = self.top.load(.seq_cst); // Upgraded to seq_cst to act as fence

            // Check if deque is non-empty: count = b - t + 1 (accounting for wraparound)
            // When empty: bottom=0, top=0 → b=max_usize, count would wrap to 0
            const count = b -% t +% 1;
            if (count > 0 and t <= b) {
                // Non-empty deque
                const task = self.items[b % self.capacity];

                if (t == b) {
                    // Last item: race with stealers
                    // Try to claim it atomically
                    if (self.top.cmpxchgStrong(t, t +% 1, .seq_cst, .seq_cst)) |_| {
                        // Lost race to a stealer
                        self.bottom.store(b +% 1, .release);
                        return null;
                    }
                    // Won the race
                    self.bottom.store(b +% 1, .release);
                    return task;
                }

                // Multiple items left
                return task;
            } else {
                // Empty deque
                self.bottom.store(b +% 1, .release);
                return null;
            }
        }

        /// Steal a task from the top of the deque (other threads).
        /// Returns null if the deque is empty or if we lose a race with the owner or other stealers.
        /// Time: O(1) | Space: O(1)
        pub fn steal(self: *Self) ?T {
            const t = self.top.load(.seq_cst); // Upgraded to seq_cst to act as fence
            const b = self.bottom.load(.seq_cst); // Upgraded to seq_cst to act as fence

            if (t < b) {
                // Non-empty deque
                const task = self.items[t % self.capacity];

                // Try to claim this task atomically
                if (self.top.cmpxchgStrong(t, t +% 1, .seq_cst, .seq_cst)) |_| {
                    // Lost race to another stealer or the owner
                    return null;
                }

                // Successfully stolen
                return task;
            }

            // Empty deque
            return null;
        }

        /// Returns the approximate size of the deque.
        /// This is a snapshot and may not be accurate due to concurrent modifications.
        /// Time: O(1) | Space: O(1)
        pub fn size(self: *const Self) usize {
            const b = self.bottom.load(.acquire);
            const t = self.top.load(.acquire);
            return b -% t;
        }

        /// Returns true if the deque is empty (approximate).
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.size() == 0;
        }

        /// Resize the deque to double its capacity (owner thread only, must hold mutex).
        /// Time: O(n) where n = current size | Space: O(new_capacity)
        fn resize(self: *Self) !void {
            const new_capacity = self.capacity * 2;
            const new_items = try self.allocator.alloc(T, new_capacity);

            const t = self.top.load(.acquire);
            const b = self.bottom.load(.acquire);

            // Copy existing items to new array
            var i: usize = 0;
            var idx = t;
            while (idx < b) : (idx +%= 1) {
                new_items[i] = self.items[idx % self.capacity];
                i += 1;
            }

            self.allocator.free(self.items);
            self.items = new_items;
            self.capacity = new_capacity;

            // Reset indices to avoid wraparound issues
            self.top.store(0, .release);
            self.bottom.store(i, .release);
        }

        /// Validate internal invariants (for testing).
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            const t = self.top.load(.acquire);
            const b = self.bottom.load(.acquire);

            // Size invariant
            const sz = b -% t;
            if (sz > self.capacity) {
                return error.InvalidSize;
            }

            // Capacity invariant
            if (self.capacity < MIN_CAPACITY) {
                return error.InvalidCapacity;
            }

            // Capacity must be power of 2 times MIN_CAPACITY
            var cap = MIN_CAPACITY;
            while (cap < self.capacity) {
                cap *= 2;
            }
            if (cap != self.capacity) {
                return error.InvalidCapacity;
            }
        }
    };
}

// ===== Tests =====

test "WorkStealingDeque: init and deinit" {
    const testing = std.testing;
    var deque = try WorkStealingDeque(u32).init(testing.allocator);
    defer deque.deinit();

    try testing.expect(deque.isEmpty());
    try testing.expectEqual(@as(usize, 0), deque.size());
    try deque.validate();
}

test "WorkStealingDeque: basic push/pop" {
    const testing = std.testing;
    var deque = try WorkStealingDeque(u32).init(testing.allocator);
    defer deque.deinit();

    try deque.push(1);
    try deque.push(2);
    try deque.push(3);

    try testing.expectEqual(@as(?u32, 3), deque.pop()); // LIFO
    try testing.expectEqual(@as(?u32, 2), deque.pop());
    try testing.expectEqual(@as(?u32, 1), deque.pop());
    try testing.expectEqual(@as(?u32, null), deque.pop());
    try testing.expect(deque.isEmpty());
}

test "WorkStealingDeque: steal" {
    const testing = std.testing;
    var deque = try WorkStealingDeque(u32).init(testing.allocator);
    defer deque.deinit();

    try deque.push(10);
    try deque.push(20);
    try deque.push(30);

    // Steal from the top (FIFO order)
    try testing.expectEqual(@as(?u32, 10), deque.steal());
    try testing.expectEqual(@as(?u32, 20), deque.steal());

    // Pop from bottom (LIFO order)
    try testing.expectEqual(@as(?u32, 30), deque.pop());
    try testing.expectEqual(@as(?u32, null), deque.steal());
}

test "WorkStealingDeque: resize" {
    const testing = std.testing;
    var deque = try WorkStealingDeque(u32).init(testing.allocator);
    defer deque.deinit();

    // Push more than MIN_CAPACITY items to trigger resize
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        try deque.push(i);
    }

    try testing.expectEqual(@as(usize, 100), deque.size());

    // Verify all items are intact after resize
    var expected: u32 = 99;
    while (expected > 0) : (expected -= 1) {
        const item = deque.pop();
        try testing.expectEqual(@as(?u32, expected), item);
    }

    try testing.expectEqual(@as(?u32, 0), deque.pop());
    try testing.expect(deque.isEmpty());
}

test "WorkStealingDeque: concurrent push/steal" {
    const testing = std.testing;
    var deque = try WorkStealingDeque(u32).init(testing.allocator);
    defer deque.deinit();

    // Push items from owner thread
    const num_items = 1000;
    var i: u32 = 0;
    while (i < num_items) : (i += 1) {
        try deque.push(i);
    }

    // Spawn a thief thread to steal items
    const ThiefCtx = struct {
        deque_ptr: *WorkStealingDeque(u32),
        stolen_count: *std.atomic.Value(u32),
    };

    var stolen_count = std.atomic.Value(u32).init(0);
    const thief_ctx = ThiefCtx{
        .deque_ptr = &deque,
        .stolen_count = &stolen_count,
    };

    const thief_fn = struct {
        fn run(ctx: ThiefCtx) void {
            var count: u32 = 0;
            while (ctx.deque_ptr.steal()) |_| {
                count += 1;
            }
            ctx.stolen_count.store(count, .release);
        }
    }.run;

    const thread = try std.Thread.spawn(.{}, thief_fn, .{thief_ctx});
    thread.join();

    // Pop remaining items from owner thread
    var popped_count: u32 = 0;
    while (deque.pop()) |_| {
        popped_count += 1;
    }

    const total = stolen_count.load(.acquire) + popped_count;
    try testing.expectEqual(num_items, total);
}

test "WorkStealingDeque: owner vs stealer ordering" {
    const testing = std.testing;
    var deque = try WorkStealingDeque(u32).init(testing.allocator);
    defer deque.deinit();

    // Push [1, 2, 3, 4, 5]
    try deque.push(1);
    try deque.push(2);
    try deque.push(3);
    try deque.push(4);
    try deque.push(5);

    // Owner pops from bottom (LIFO): gets 5
    try testing.expectEqual(@as(?u32, 5), deque.pop());

    // Stealer steals from top (FIFO): gets 1
    try testing.expectEqual(@as(?u32, 1), deque.steal());

    // Owner pops: gets 4
    try testing.expectEqual(@as(?u32, 4), deque.pop());

    // Stealer steals: gets 2
    try testing.expectEqual(@as(?u32, 2), deque.steal());

    // Remaining: 3
    try testing.expectEqual(@as(?u32, 3), deque.pop());
    try testing.expect(deque.isEmpty());
}

test "WorkStealingDeque: empty edge cases" {
    const testing = std.testing;
    var deque = try WorkStealingDeque(u32).init(testing.allocator);
    defer deque.deinit();

    // Pop from empty
    try testing.expectEqual(@as(?u32, null), deque.pop());

    // Steal from empty
    try testing.expectEqual(@as(?u32, null), deque.steal());

    // Push one, pop one
    try deque.push(42);
    try testing.expectEqual(@as(?u32, 42), deque.pop());
    try testing.expect(deque.isEmpty());

    // Push one, steal one
    try deque.push(99);
    try testing.expectEqual(@as(?u32, 99), deque.steal());
    try testing.expect(deque.isEmpty());
}

test "WorkStealingDeque: last element race" {
    const testing = std.testing;
    var deque = try WorkStealingDeque(u32).init(testing.allocator);
    defer deque.deinit();

    // Push one item
    try deque.push(100);

    // In real concurrent scenario, owner and stealer could race on last element.
    // Here we simulate: either pop or steal succeeds, the other gets null.
    const result1 = deque.pop();
    const result2 = deque.steal();

    // Exactly one should succeed
    const succeeded = @intFromBool(result1 != null) + @intFromBool(result2 != null);
    try testing.expectEqual(@as(u32, 1), succeeded);

    if (result1) |val| {
        try testing.expectEqual(@as(u32, 100), val);
    }
    if (result2) |val| {
        try testing.expectEqual(@as(u32, 100), val);
    }
}

test "WorkStealingDeque: stress test" {
    const testing = std.testing;
    var deque = try WorkStealingDeque(u32).init(testing.allocator);
    defer deque.deinit();

    // Push 10000 items
    var i: u32 = 0;
    while (i < 10000) : (i += 1) {
        try deque.push(i);
    }

    try testing.expectEqual(@as(usize, 10000), deque.size());

    // Pop all
    var count: u32 = 0;
    while (deque.pop()) |_| {
        count += 1;
    }

    try testing.expectEqual(@as(u32, 10000), count);
    try testing.expect(deque.isEmpty());
    try deque.validate();
}

test "WorkStealingDeque: memory leak check" {
    const testing = std.testing;
    var deque = try WorkStealingDeque(u32).init(testing.allocator);
    defer deque.deinit();

    // Push and pop many times to trigger resize
    var round: usize = 0;
    while (round < 10) : (round += 1) {
        var i: u32 = 0;
        while (i < 100) : (i += 1) {
            try deque.push(i);
        }

        // Verify all pushed values can be popped
        var pop_count: u32 = 0;
        var last_val: ?u32 = null;
        while (deque.pop()) |val| {
            pop_count += 1;
            last_val = val;
        }
        try testing.expectEqual(@as(u32, 100), pop_count);
    }

    try testing.expectEqual(true, deque.isEmpty());
    try deque.validate();
}

test "WorkStealingDeque: validate invariants" {
    const testing = std.testing;
    var deque = try WorkStealingDeque(u32).init(testing.allocator);
    defer deque.deinit();

    try testing.expectEqual(true, deque.isEmpty());
    try deque.validate();

    try deque.push(1);
    try testing.expectEqual(false, deque.isEmpty());
    try deque.validate();

    const val = deque.pop();
    try testing.expectEqual(@as(?u32, 1), val);
    try testing.expectEqual(true, deque.isEmpty());
    try deque.validate();

    // Trigger resize and verify count
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        try deque.push(i);
    }
    try testing.expectEqual(false, deque.isEmpty());
    try deque.validate();

    // Pop all and verify they're accessible
    var pop_count: u32 = 0;
    while (deque.pop()) |_| {
        pop_count += 1;
    }
    try testing.expectEqual(@as(u32, 100), pop_count);
    try testing.expectEqual(true, deque.isEmpty());
}

test "WorkStealingDeque: string type" {
    const testing = std.testing;
    var deque = try WorkStealingDeque([]const u8).init(testing.allocator);
    defer deque.deinit();

    try deque.push("hello");
    try deque.push("world");
    try deque.push("foo");

    try testing.expectEqualStrings("foo", deque.pop().?);
    try testing.expectEqualStrings("hello", deque.steal().?);
    try testing.expectEqualStrings("world", deque.pop().?);
}

test "WorkStealingDeque: pop on empty deque returns null (issue #13)" {
    const testing = std.testing;
    var deque = try WorkStealingDeque(u32).init(testing.allocator);
    defer deque.deinit();

    // Empty deque should return null, not garbage
    const result = deque.pop();
    try testing.expectEqual(@as(?u32, null), result);

    // Also test steal on empty deque
    const stolen = deque.steal();
    try testing.expectEqual(@as(?u32, null), stolen);

    // Test pop after push+pop (returns to empty)
    try deque.push(42);
    try testing.expectEqual(@as(?u32, 42), deque.pop());
    try testing.expectEqual(@as(?u32, null), deque.pop());
}
