const std = @import("std");
const testing = std.testing;

/// Deque — double-ended queue with O(1) push/pop at both ends
///
/// Implemented as a circular buffer with dynamic resizing. Supports efficient
/// insertion and removal from both front and back, making it suitable for
/// queue, stack, and sliding window operations.
///
/// Performance characteristics:
/// - push_front/push_back: O(1) amortized
/// - pop_front/pop_back: O(1) amortized
/// - get: O(1)
/// - clear: O(n) if elements need destruction, O(1) otherwise
///
/// Memory layout uses a circular buffer with head/tail pointers that wrap around.
/// When capacity is exceeded, the buffer is reallocated with 2x growth factor.
///
/// Type parameters:
/// - T: element type
pub fn Deque(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const Iterator = struct {
            deque: *const Self,
            index: usize,

            /// Returns next element, or null if exhausted
            /// Time: O(1) | Space: O(1)
            pub fn next(self: *Iterator) ?T {
                if (self.index >= self.deque.length) {
                    return null;
                }

                const value = self.deque.get(self.index) catch unreachable;
                self.index += 1;
                return value;
            }
        };

        allocator: std.mem.Allocator,
        buffer: []T = &[_]T{},
        head: usize = 0, // Index of first element
        tail: usize = 0, // Index of one past last element
        length: usize = 0,

        // -- Lifecycle --

        /// Initialize empty deque
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
            };
        }

        /// Initialize with specified initial capacity
        /// Time: O(n) | Space: O(n)
        pub fn initCapacity(allocator: std.mem.Allocator, initial_capacity: usize) !Self {
            var self = Self.init(allocator);
            if (initial_capacity > 0) {
                self.buffer = try allocator.alloc(T, initial_capacity);
            }
            return self;
        }

        /// Free all allocated memory
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.buffer.len > 0) {
                self.allocator.free(self.buffer);
            }
            self.* = undefined;
        }

        /// Create a deep copy of this deque
        /// Time: O(n) | Space: O(n)
        pub fn clone(self: *const Self) !Self {
            var new_deque = try Self.initCapacity(self.allocator, self.buffer.len);
            errdefer new_deque.deinit();

            var i: usize = 0;
            while (i < self.length) : (i += 1) {
                const value = try self.get(i);
                try new_deque.push_back(value);
            }

            return new_deque;
        }

        // -- Capacity --

        /// Returns the number of elements in the deque
        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            return self.length;
        }

        /// Returns true if the deque is empty
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.length == 0;
        }

        /// Returns the current capacity
        /// Time: O(1) | Space: O(1)
        pub fn capacity(self: *const Self) usize {
            return self.buffer.len;
        }

        /// Ensure the deque has at least the specified capacity
        /// Time: O(n) if reallocation needed | Space: O(n) if reallocation needed
        pub fn ensureCapacity(self: *Self, min_capacity: usize) !void {
            if (self.buffer.len >= min_capacity) {
                return;
            }

            const new_capacity = @max(min_capacity, self.buffer.len * 2);
            const new_buffer = try self.allocator.alloc(T, new_capacity);
            errdefer self.allocator.free(new_buffer);

            // Copy elements in order from old buffer to new buffer
            if (self.length > 0) {
                var i: usize = 0;
                while (i < self.length) : (i += 1) {
                    const old_index = (self.head + i) % self.buffer.len;
                    new_buffer[i] = self.buffer[old_index];
                }
            }

            if (self.buffer.len > 0) {
                self.allocator.free(self.buffer);
            }

            self.buffer = new_buffer;
            self.head = 0;
            self.tail = self.length;
        }

        /// Shrink capacity to fit the current number of elements
        /// Time: O(n) | Space: O(n)
        pub fn shrinkToFit(self: *Self) !void {
            if (self.length == self.buffer.len or self.length == 0) {
                return;
            }

            const new_buffer = try self.allocator.alloc(T, self.length);
            errdefer self.allocator.free(new_buffer);

            // Copy elements in order
            var i: usize = 0;
            while (i < self.length) : (i += 1) {
                const old_index = (self.head + i) % self.buffer.len;
                new_buffer[i] = self.buffer[old_index];
            }

            if (self.buffer.len > 0) {
                self.allocator.free(self.buffer);
            }

            self.buffer = new_buffer;
            self.head = 0;
            self.tail = self.length;
        }

        // -- Modification --

        /// Add element to the back of the deque
        /// Time: O(1) amortized | Space: O(1) amortized
        pub fn push_back(self: *Self, value: T) !void {
            if (self.length == self.buffer.len) {
                const new_capacity = if (self.buffer.len == 0) 8 else self.buffer.len * 2;
                try self.ensureCapacity(new_capacity);
            }

            self.buffer[self.tail] = value;
            self.tail = (self.tail + 1) % self.buffer.len;
            self.length += 1;
        }

        /// Add element to the front of the deque
        /// Time: O(1) amortized | Space: O(1) amortized
        pub fn push_front(self: *Self, value: T) !void {
            if (self.length == self.buffer.len) {
                const new_capacity = if (self.buffer.len == 0) 8 else self.buffer.len * 2;
                try self.ensureCapacity(new_capacity);
            }

            self.head = if (self.head == 0) self.buffer.len - 1 else self.head - 1;
            self.buffer[self.head] = value;
            self.length += 1;
        }

        /// Remove and return element from the back of the deque
        /// Time: O(1) | Space: O(1)
        pub fn pop_back(self: *Self) !T {
            if (self.isEmpty()) return error.DequeEmpty;

            self.tail = if (self.tail == 0) self.buffer.len - 1 else self.tail - 1;
            const value = self.buffer[self.tail];
            self.length -= 1;

            return value;
        }

        /// Remove and return element from the front of the deque
        /// Time: O(1) | Space: O(1)
        pub fn pop_front(self: *Self) !T {
            if (self.isEmpty()) return error.DequeEmpty;

            const value = self.buffer[self.head];
            self.head = (self.head + 1) % self.buffer.len;
            self.length -= 1;

            return value;
        }

        /// Insert element at the specified index
        /// Time: O(min(k, n-k)) where k is index | Space: O(1) amortized
        pub fn insertAt(self: *Self, index: usize, value: T) !void {
            if (index > self.length) return error.IndexOutOfBounds;

            if (index == 0) return self.push_front(value);
            if (index == self.length) return self.push_back(value);

            if (self.length == self.buffer.len) {
                const new_capacity = if (self.buffer.len == 0) 8 else self.buffer.len * 2;
                try self.ensureCapacity(new_capacity);
            }

            // Decide whether to shift elements left or right
            if (index < self.length / 2) {
                // Shift front elements left
                self.head = if (self.head == 0) self.buffer.len - 1 else self.head - 1;

                var i: usize = 0;
                while (i < index) : (i += 1) {
                    const curr_idx = (self.head + i) % self.buffer.len;
                    const next_idx = (self.head + i + 1) % self.buffer.len;
                    self.buffer[curr_idx] = self.buffer[next_idx];
                }

                const insert_idx = (self.head + index) % self.buffer.len;
                self.buffer[insert_idx] = value;
            } else {
                // Shift back elements right
                var i = self.length;
                while (i > index) : (i -= 1) {
                    const curr_idx = (self.head + i) % self.buffer.len;
                    const prev_idx = (self.head + i - 1) % self.buffer.len;
                    self.buffer[curr_idx] = self.buffer[prev_idx];
                }

                const insert_idx = (self.head + index) % self.buffer.len;
                self.buffer[insert_idx] = value;
                self.tail = (self.tail + 1) % self.buffer.len;
            }

            self.length += 1;
        }

        /// Remove and return element at the specified index
        /// Time: O(min(k, n-k)) where k is index | Space: O(1)
        pub fn removeAt(self: *Self, index: usize) !T {
            if (index >= self.length) return error.IndexOutOfBounds;

            if (index == 0) return self.pop_front();
            if (index == self.length - 1) return self.pop_back();

            const remove_idx = (self.head + index) % self.buffer.len;
            const value = self.buffer[remove_idx];

            // Decide whether to shift elements left or right
            if (index < self.length / 2) {
                // Shift front elements right
                var i = index;
                while (i > 0) : (i -= 1) {
                    const curr_idx = (self.head + i) % self.buffer.len;
                    const prev_idx = (self.head + i - 1) % self.buffer.len;
                    self.buffer[curr_idx] = self.buffer[prev_idx];
                }
                self.head = (self.head + 1) % self.buffer.len;
            } else {
                // Shift back elements left
                var i = index;
                while (i < self.length - 1) : (i += 1) {
                    const curr_idx = (self.head + i) % self.buffer.len;
                    const next_idx = (self.head + i + 1) % self.buffer.len;
                    self.buffer[curr_idx] = self.buffer[next_idx];
                }
                self.tail = if (self.tail == 0) self.buffer.len - 1 else self.tail - 1;
            }

            self.length -= 1;
            return value;
        }

        /// Remove all elements from the deque
        /// Time: O(1) | Space: O(1)
        pub fn clear(self: *Self) void {
            self.head = 0;
            self.tail = 0;
            self.length = 0;
        }

        // -- Lookup --

        /// Get element at the specified index
        /// Time: O(1) | Space: O(1)
        pub fn get(self: *const Self, index: usize) !T {
            if (index >= self.length) return error.IndexOutOfBounds;
            const actual_index = (self.head + index) % self.buffer.len;
            return self.buffer[actual_index];
        }

        /// Get pointer to element at the specified index
        /// Time: O(1) | Space: O(1)
        pub fn getPtr(self: *Self, index: usize) !*T {
            if (index >= self.length) return error.IndexOutOfBounds;
            const actual_index = (self.head + index) % self.buffer.len;
            return &self.buffer[actual_index];
        }

        /// Get the first element without removing it
        /// Time: O(1) | Space: O(1)
        pub fn peekFront(self: *const Self) !T {
            if (self.isEmpty()) return error.DequeEmpty;
            return self.buffer[self.head];
        }

        /// Get the last element without removing it
        /// Time: O(1) | Space: O(1)
        pub fn peekBack(self: *const Self) !T {
            if (self.isEmpty()) return error.DequeEmpty;
            const back_index = if (self.tail == 0) self.buffer.len - 1 else self.tail - 1;
            return self.buffer[back_index];
        }

        // -- Iteration --

        /// Returns an iterator over the deque
        /// Time: O(1) | Space: O(1)
        pub fn iterator(self: *const Self) Iterator {
            return .{
                .deque = self,
                .index = 0,
            };
        }

        // -- Bulk Operations --

        /// Create deque from a slice
        /// Time: O(n) | Space: O(n)
        pub fn fromSlice(allocator: std.mem.Allocator, slice: []const T) !Self {
            var deque = try Self.initCapacity(allocator, slice.len);
            errdefer deque.deinit();

            for (slice) |value| {
                try deque.push_back(value);
            }

            return deque;
        }

        /// Convert deque to a newly allocated slice
        /// Time: O(n) | Space: O(n)
        pub fn toSlice(self: *const Self, allocator: std.mem.Allocator) ![]T {
            const slice = try allocator.alloc(T, self.length);
            errdefer allocator.free(slice);

            var i: usize = 0;
            while (i < self.length) : (i += 1) {
                slice[i] = try self.get(i);
            }

            return slice;
        }

        // -- Debug & Validation --

        /// Format the deque for debugging
        pub fn format(
            self: *const Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;

            try writer.writeAll("Deque(");
            try writer.print("len={d}, cap={d}, head={d}, tail={d})", .{
                self.length,
                self.buffer.len,
                self.head,
                self.tail,
            });
        }

        /// Validate internal invariants
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            // Length should not exceed capacity
            if (self.length > self.buffer.len) {
                return error.LengthExceedsCapacity;
            }

            // Head and tail should be within bounds
            if (self.buffer.len > 0) {
                if (self.head >= self.buffer.len) return error.HeadOutOfBounds;
                if (self.tail >= self.buffer.len) return error.TailOutOfBounds;
            }

            // If empty, head and tail should be 0
            if (self.length == 0 and self.buffer.len > 0) {
                if (self.head != 0 or self.tail != 0) {
                    return error.EmptyDequeInvalidPointers;
                }
            }

            // Verify length matches circular buffer state
            const computed_length = if (self.tail >= self.head)
                self.tail - self.head
            else
                self.buffer.len - self.head + self.tail;

            if (self.buffer.len > 0 and self.length != computed_length) {
                return error.LengthMismatch;
            }
        }
    };
}

// -- Tests --

test "Deque: basic push_back and pop_front" {
    var deque = Deque(i32).init(testing.allocator);
    defer deque.deinit();

    try deque.push_back(10);
    try deque.push_back(20);
    try deque.push_back(30);

    try testing.expectEqual(@as(usize, 3), deque.count());
    try testing.expectEqual(@as(i32, 10), try deque.pop_front());
    try testing.expectEqual(@as(i32, 20), try deque.pop_front());
    try testing.expectEqual(@as(i32, 30), try deque.pop_front());
    try testing.expect(deque.isEmpty());

    try deque.validate();
}

test "Deque: push_front and pop_back" {
    var deque = Deque(i32).init(testing.allocator);
    defer deque.deinit();

    try deque.push_front(30);
    try deque.push_front(20);
    try deque.push_front(10);

    try testing.expectEqual(@as(usize, 3), deque.count());
    try testing.expectEqual(@as(i32, 30), try deque.pop_back());
    try testing.expectEqual(@as(i32, 20), try deque.pop_back());
    try testing.expectEqual(@as(i32, 10), try deque.pop_back());

    try deque.validate();
}

test "Deque: mixed push and pop operations" {
    var deque = Deque(i32).init(testing.allocator);
    defer deque.deinit();

    try deque.push_back(1);
    try deque.push_front(0);
    try deque.push_back(2);

    try testing.expectEqual(@as(i32, 0), try deque.pop_front());
    try testing.expectEqual(@as(i32, 2), try deque.pop_back());
    try testing.expectEqual(@as(i32, 1), try deque.pop_front());

    try deque.validate();
}

test "Deque: get and index access" {
    var deque = Deque(i32).init(testing.allocator);
    defer deque.deinit();

    try deque.push_back(10);
    try deque.push_back(20);
    try deque.push_back(30);

    try testing.expectEqual(@as(i32, 10), try deque.get(0));
    try testing.expectEqual(@as(i32, 20), try deque.get(1));
    try testing.expectEqual(@as(i32, 30), try deque.get(2));

    try deque.validate();
}

test "Deque: peekFront and peekBack" {
    var deque = Deque(i32).init(testing.allocator);
    defer deque.deinit();

    try deque.push_back(10);
    try deque.push_back(20);
    try deque.push_back(30);

    try testing.expectEqual(@as(i32, 10), try deque.peekFront());
    try testing.expectEqual(@as(i32, 30), try deque.peekBack());
    try testing.expectEqual(@as(usize, 3), deque.count());

    try deque.validate();
}

test "Deque: insertAt" {
    var deque = Deque(i32).init(testing.allocator);
    defer deque.deinit();

    try deque.push_back(10);
    try deque.push_back(30);
    try deque.insertAt(1, 20);

    try testing.expectEqual(@as(usize, 3), deque.count());
    try testing.expectEqual(@as(i32, 10), try deque.get(0));
    try testing.expectEqual(@as(i32, 20), try deque.get(1));
    try testing.expectEqual(@as(i32, 30), try deque.get(2));

    try deque.validate();
}

test "Deque: removeAt" {
    var deque = Deque(i32).init(testing.allocator);
    defer deque.deinit();

    try deque.push_back(10);
    try deque.push_back(20);
    try deque.push_back(30);
    try deque.push_back(40);

    const removed = try deque.removeAt(1);
    try testing.expectEqual(@as(i32, 20), removed);
    try testing.expectEqual(@as(usize, 3), deque.count());
    try testing.expectEqual(@as(i32, 10), try deque.get(0));
    try testing.expectEqual(@as(i32, 30), try deque.get(1));
    try testing.expectEqual(@as(i32, 40), try deque.get(2));

    try deque.validate();
}

test "Deque: circular buffer wraparound" {
    var deque = Deque(i32).init(testing.allocator);
    defer deque.deinit();

    // Force circular wraparound by alternating push/pop
    try deque.push_back(1);
    try deque.push_back(2);
    try deque.push_back(3);
    _ = try deque.pop_front();
    _ = try deque.pop_front();

    try deque.push_back(4);
    try deque.push_back(5);

    try testing.expectEqual(@as(usize, 3), deque.count());
    try testing.expectEqual(@as(i32, 3), try deque.get(0));
    try testing.expectEqual(@as(i32, 4), try deque.get(1));
    try testing.expectEqual(@as(i32, 5), try deque.get(2));

    try deque.validate();
}

test "Deque: iterator" {
    var deque = Deque(i32).init(testing.allocator);
    defer deque.deinit();

    try deque.push_back(10);
    try deque.push_back(20);
    try deque.push_back(30);

    var expected: i32 = 10;
    var it = deque.iterator();
    while (it.next()) |value| {
        try testing.expectEqual(expected, value);
        expected += 10;
    }

    try testing.expectEqual(@as(i32, 40), expected);
}

test "Deque: clear" {
    var deque = Deque(i32).init(testing.allocator);
    defer deque.deinit();

    try deque.push_back(10);
    try deque.push_back(20);
    try deque.push_back(30);

    deque.clear();

    try testing.expectEqual(@as(usize, 0), deque.count());
    try testing.expect(deque.isEmpty());

    try deque.validate();
}

test "Deque: fromSlice and toSlice" {
    const data = [_]i32{ 1, 2, 3, 4, 5 };
    var deque = try Deque(i32).fromSlice(testing.allocator, &data);
    defer deque.deinit();

    try testing.expectEqual(@as(usize, 5), deque.count());

    const slice = try deque.toSlice(testing.allocator);
    defer testing.allocator.free(slice);

    try testing.expectEqualSlices(i32, &data, slice);

    try deque.validate();
}

test "Deque: clone" {
    var original = Deque(i32).init(testing.allocator);
    defer original.deinit();

    try original.push_back(10);
    try original.push_back(20);
    try original.push_back(30);

    var cloned = try original.clone();
    defer cloned.deinit();

    try testing.expectEqual(original.count(), cloned.count());
    try testing.expectEqual(@as(i32, 10), try cloned.get(0));
    try testing.expectEqual(@as(i32, 20), try cloned.get(1));
    try testing.expectEqual(@as(i32, 30), try cloned.get(2));

    try cloned.validate();
}

test "Deque: capacity management" {
    var deque = Deque(i32).init(testing.allocator);
    defer deque.deinit();

    try testing.expectEqual(@as(usize, 0), deque.capacity());

    try deque.ensureCapacity(10);
    try testing.expect(deque.capacity() >= 10);

    var i: i32 = 0;
    while (i < 5) : (i += 1) {
        try deque.push_back(i);
    }

    try deque.shrinkToFit();
    try testing.expectEqual(@as(usize, 5), deque.capacity());

    try deque.validate();
}

test "Deque: stress test with many operations" {
    var deque = Deque(i32).init(testing.allocator);
    defer deque.deinit();

    // Push many elements
    var i: i32 = 0;
    while (i < 100) : (i += 1) {
        try deque.push_back(i);
    }

    try testing.expectEqual(@as(usize, 100), deque.count());

    // Pop from front
    i = 0;
    while (i < 50) : (i += 1) {
        const value = try deque.pop_front();
        try testing.expectEqual(i, value);
    }

    // Push to front
    i = -1;
    while (i >= -49) : (i -= 1) {
        try deque.push_front(i);
    }

    try testing.expectEqual(@as(usize, 100), deque.count());

    try deque.validate();
}

test "Deque: memory leak check" {
    var deque = Deque(i32).init(testing.allocator);
    defer deque.deinit();

    var i: i32 = 0;
    while (i < 100) : (i += 1) {
        try deque.push_back(i);
    }

    while (!deque.isEmpty()) {
        _ = try deque.pop_front();
    }

    try deque.validate();
}

test "Deque: error cases" {
    var deque = Deque(i32).init(testing.allocator);
    defer deque.deinit();

    // Pop from empty deque
    try testing.expectError(error.DequeEmpty, deque.pop_front());
    try testing.expectError(error.DequeEmpty, deque.pop_back());
    try testing.expectError(error.DequeEmpty, deque.peekFront());
    try testing.expectError(error.DequeEmpty, deque.peekBack());

    try deque.push_back(10);

    // Get out of bounds
    try testing.expectError(error.IndexOutOfBounds, deque.get(5));

    // Remove out of bounds
    try testing.expectError(error.IndexOutOfBounds, deque.removeAt(5));

    // Insert out of bounds
    try testing.expectError(error.IndexOutOfBounds, deque.insertAt(10, 99));

    try deque.validate();
}
