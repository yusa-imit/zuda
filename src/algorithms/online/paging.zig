//! Online Paging Algorithms - Page replacement for memory management
//!
//! Problem: Maintain k pages in cache with limited capacity. When requesting
//! a page not in cache (page fault), evict one page to make room.
//!
//! Competitive Analysis:
//! - LRU: k-competitive (optimal for deterministic)
//! - FIFO: k-competitive
//! - Optimal offline (MIN): Has perfect knowledge of future requests
//!
//! Applications:
//! - OS virtual memory management
//! - Database buffer pool management
//! - Web caching (LRU is common)
//! - Content delivery networks

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Page fault statistics
pub const PageFaultStats = struct {
    total_requests: usize,
    page_faults: usize,
    hits: usize,

    /// Compute hit rate
    /// Time: O(1) | Space: O(1)
    pub fn hitRate(self: PageFaultStats) f64 {
        if (self.total_requests == 0) return 0.0;
        return @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(self.total_requests));
    }

    /// Compute fault rate
    /// Time: O(1) | Space: O(1)
    pub fn faultRate(self: PageFaultStats) f64 {
        if (self.total_requests == 0) return 0.0;
        return @as(f64, @floatFromInt(self.page_faults)) / @as(f64, @floatFromInt(self.total_requests));
    }
};

/// LRU (Least Recently Used) Paging
/// Time: O(1) per request | Space: O(capacity)
pub fn LRU(comptime T: type) type {
    return struct {
        const Self = @This();

        const Node = struct {
            key: T,
            prev: ?*Node,
            next: ?*Node,
        };

        allocator: Allocator,
        capacity: usize,
        cache: std.AutoHashMap(T, *Node), // key -> node
        head: ?*Node, // Most recently used
        tail: ?*Node, // Least recently used
        stats: PageFaultStats,

        /// Initialize LRU cache
        /// Time: O(1) | Space: O(capacity)
        pub fn init(allocator: Allocator, capacity: usize) !Self {
            return Self{
                .allocator = allocator,
                .capacity = capacity,
                .cache = std.AutoHashMap(T, *Node).init(allocator),
                .head = null,
                .tail = null,
                .stats = .{ .total_requests = 0, .page_faults = 0, .hits = 0 },
            };
        }

        /// Clean up resources
        pub fn deinit(self: *Self) void {
            var it = self.cache.valueIterator();
            while (it.next()) |node_ptr| {
                self.allocator.destroy(node_ptr.*);
            }
            self.cache.deinit();
        }

        /// Request a page
        /// Time: O(1) | Space: O(1)
        pub fn request(self: *Self, page: T) !bool {
            self.stats.total_requests += 1;

            if (self.cache.get(page)) |node| {
                // Hit: move to front
                self.stats.hits += 1;
                self.moveToFront(node);
                return false; // No fault
            }

            // Fault: need to load page
            self.stats.page_faults += 1;

            // Evict if at capacity
            if (self.cache.count() >= self.capacity) {
                if (self.tail) |lru_node| {
                    _ = self.cache.remove(lru_node.key);
                    self.removeNode(lru_node);
                    self.allocator.destroy(lru_node);
                }
            }

            // Insert new page
            const node = try self.allocator.create(Node);
            node.* = .{ .key = page, .prev = null, .next = null };
            try self.cache.put(page, node);
            self.addToFront(node);

            return true; // Fault occurred
        }

        /// Get current cache contents (most to least recent)
        /// Time: O(k) where k is cache size | Space: O(k)
        pub fn getCacheContents(self: *Self, allocator: Allocator) ![]T {
            var contents = try std.ArrayList(T).initCapacity(allocator, self.cache.count());
            var current = self.head;
            while (current) |node| {
                contents.appendAssumeCapacity(node.key);
                current = node.next;
            }
            return contents.toOwnedSlice();
        }

        /// Get statistics
        /// Time: O(1) | Space: O(1)
        pub fn getStats(self: Self) PageFaultStats {
            return self.stats;
        }

        // Internal: move node to front (MRU position)
        fn moveToFront(self: *Self, node: *Node) void {
            if (self.head == node) return; // Already at front

            self.removeNode(node);
            self.addToFront(node);
        }

        // Internal: add node to front
        fn addToFront(self: *Self, node: *Node) void {
            node.next = self.head;
            node.prev = null;

            if (self.head) |h| {
                h.prev = node;
            } else {
                self.tail = node;
            }

            self.head = node;
        }

        // Internal: remove node from list
        fn removeNode(self: *Self, node: *Node) void {
            if (node.prev) |p| {
                p.next = node.next;
            } else {
                self.head = node.next;
            }

            if (node.next) |n| {
                n.prev = node.prev;
            } else {
                self.tail = node.prev;
            }
        }
    };
}

/// FIFO (First In First Out) Paging
/// Time: O(1) per request | Space: O(capacity)
pub fn FIFO(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        capacity: usize,
        cache: std.AutoHashMap(T, void),
        queue: std.ArrayList(T), // Insertion order
        stats: PageFaultStats,

        /// Initialize FIFO cache
        /// Time: O(1) | Space: O(capacity)
        pub fn init(allocator: Allocator, capacity: usize) !Self {
            return Self{
                .allocator = allocator,
                .capacity = capacity,
                .cache = std.AutoHashMap(T, void).init(allocator),
                .queue = std.ArrayList(T).init(allocator),
                .stats = .{ .total_requests = 0, .page_faults = 0, .hits = 0 },
            };
        }

        /// Clean up resources
        pub fn deinit(self: *Self) void {
            self.cache.deinit();
            self.queue.deinit();
        }

        /// Request a page
        /// Time: O(1) amortized | Space: O(1)
        pub fn request(self: *Self, page: T) !bool {
            self.stats.total_requests += 1;

            if (self.cache.contains(page)) {
                // Hit
                self.stats.hits += 1;
                return false;
            }

            // Fault
            self.stats.page_faults += 1;

            // Evict oldest if at capacity
            if (self.cache.count() >= self.capacity) {
                const oldest = self.queue.orderedRemove(0);
                _ = self.cache.remove(oldest);
            }

            // Insert new page
            try self.cache.put(page, {});
            try self.queue.append(page);

            return true;
        }

        /// Get current cache contents (oldest to newest)
        /// Time: O(k) where k is cache size | Space: O(k)
        pub fn getCacheContents(self: *Self, allocator: Allocator) ![]T {
            return try allocator.dupe(T, self.queue.items);
        }

        /// Get statistics
        /// Time: O(1) | Space: O(1)
        pub fn getStats(self: Self) PageFaultStats {
            return self.stats;
        }
    };
}

/// Optimal offline paging (MIN algorithm)
/// Requires full knowledge of future requests
/// Time: O(n * k) where n is request count, k is capacity | Space: O(capacity)
pub fn optimalOffline(comptime T: type, allocator: Allocator, requests: []const T, capacity: usize) !usize {
    var cache = std.AutoHashMap(T, void).init(allocator);
    defer cache.deinit();

    var faults: usize = 0;

    for (requests, 0..) |page, i| {
        if (cache.contains(page)) continue; // Hit

        faults += 1;

        // Evict page used farthest in future (or never used again)
        if (cache.count() >= capacity) {
            var evict_page: ?T = null;
            var max_distance: usize = 0;

            var it = cache.keyIterator();
            while (it.next()) |cached_page| {
                // Find next use of this page
                var distance: usize = requests.len; // Never used again
                for (requests[i + 1 ..], 0..) |future_page, d| {
                    if (future_page == cached_page.*) {
                        distance = d;
                        break;
                    }
                }

                if (evict_page == null or distance > max_distance) {
                    evict_page = cached_page.*;
                    max_distance = distance;
                }
            }

            if (evict_page) |evict| {
                _ = cache.remove(evict);
            }
        }

        try cache.put(page, {});
    }

    return faults;
}

/// Compute competitive ratio
/// Time: O(1) | Space: O(1)
pub fn competitiveRatio(online_faults: usize, offline_faults: usize) f64 {
    if (offline_faults == 0) return 1.0;
    return @as(f64, @floatFromInt(online_faults)) / @as(f64, @floatFromInt(offline_faults));
}

// ============================================================================
// Tests
// ============================================================================

test "paging - LRU basic operations" {
    var lru = try LRU(u32).init(testing.allocator, 3);
    defer lru.deinit();

    // First access: fault
    try testing.expect(try lru.request(1));
    try testing.expect(try lru.request(2));
    try testing.expect(try lru.request(3));

    const stats1 = lru.getStats();
    try testing.expectEqual(@as(usize, 3), stats1.page_faults);
    try testing.expectEqual(@as(usize, 0), stats1.hits);

    // Access 1 again: hit
    try testing.expect(!try lru.request(1));

    const stats2 = lru.getStats();
    try testing.expectEqual(@as(usize, 3), stats2.page_faults);
    try testing.expectEqual(@as(usize, 1), stats2.hits);
}

test "paging - LRU eviction" {
    var lru = try LRU(u32).init(testing.allocator, 2);
    defer lru.deinit();

    _ = try lru.request(1);
    _ = try lru.request(2);

    // Access 1 to make it MRU
    _ = try lru.request(1);

    // Add 3: should evict 2 (LRU)
    try testing.expect(try lru.request(3));

    // 2 should fault, 1 and 3 should hit
    try testing.expect(try lru.request(2));
    try testing.expect(!try lru.request(1));
    try testing.expect(!try lru.request(3));
}

test "paging - LRU cache contents" {
    var lru = try LRU(u32).init(testing.allocator, 3);
    defer lru.deinit();

    _ = try lru.request(1);
    _ = try lru.request(2);
    _ = try lru.request(3);
    _ = try lru.request(1); // Move 1 to front

    const contents = try lru.getCacheContents(testing.allocator);
    defer testing.allocator.free(contents);

    try testing.expectEqual(@as(usize, 3), contents.len);
    try testing.expectEqual(@as(u32, 1), contents[0]); // MRU
    try testing.expectEqual(@as(u32, 3), contents[1]);
    try testing.expectEqual(@as(u32, 2), contents[2]); // LRU
}

test "paging - LRU statistics" {
    var lru = try LRU(u32).init(testing.allocator, 2);
    defer lru.deinit();

    _ = try lru.request(1);
    _ = try lru.request(2);
    _ = try lru.request(1);
    _ = try lru.request(3);
    _ = try lru.request(2);

    const stats = lru.getStats();
    try testing.expectEqual(@as(usize, 5), stats.total_requests);
    try testing.expectEqual(@as(usize, 4), stats.page_faults); // 1, 2, 3, 2
    try testing.expectEqual(@as(usize, 1), stats.hits); // 1 (second access)

    try testing.expectApproxEqAbs(0.2, stats.hitRate(), 0.001);
    try testing.expectApproxEqAbs(0.8, stats.faultRate(), 0.001);
}

test "paging - FIFO basic operations" {
    var fifo = try FIFO(u32).init(testing.allocator, 3);
    defer fifo.deinit();

    try testing.expect(try fifo.request(1));
    try testing.expect(try fifo.request(2));
    try testing.expect(try fifo.request(3));

    const stats1 = fifo.getStats();
    try testing.expectEqual(@as(usize, 3), stats1.page_faults);

    try testing.expect(!try fifo.request(1)); // Hit

    const stats2 = fifo.getStats();
    try testing.expectEqual(@as(usize, 1), stats2.hits);
}

test "paging - FIFO eviction" {
    var fifo = try FIFO(u32).init(testing.allocator, 2);
    defer fifo.deinit();

    _ = try fifo.request(1);
    _ = try fifo.request(2);

    // Add 3: should evict 1 (oldest)
    try testing.expect(try fifo.request(3));

    // 1 should fault, 2 and 3 should hit
    try testing.expect(try fifo.request(1));
    try testing.expect(!try fifo.request(2));
    try testing.expect(!try fifo.request(3));
}

test "paging - FIFO cache contents" {
    var fifo = try FIFO(u32).init(testing.allocator, 3);
    defer fifo.deinit();

    _ = try fifo.request(1);
    _ = try fifo.request(2);
    _ = try fifo.request(3);

    const contents = try fifo.getCacheContents(testing.allocator);
    defer testing.allocator.free(contents);

    try testing.expectEqual(@as(usize, 3), contents.len);
    try testing.expectEqual(@as(u32, 1), contents[0]); // Oldest
    try testing.expectEqual(@as(u32, 2), contents[1]);
    try testing.expectEqual(@as(u32, 3), contents[2]); // Newest
}

test "paging - optimal offline" {
    const requests = [_]u32{ 1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5 };
    const faults = try optimalOffline(u32, testing.allocator, &requests, 3);

    // Optimal should minimize faults
    try testing.expect(faults > 0);
    try testing.expect(faults <= requests.len);
}

test "paging - competitive ratio: LRU vs optimal" {
    const requests = [_]u32{ 1, 2, 3, 4, 1, 2, 5, 1, 2, 3 };

    var lru = try LRU(u32).init(testing.allocator, 3);
    defer lru.deinit();

    for (requests) |page| {
        _ = try lru.request(page);
    }

    const lru_faults = lru.getStats().page_faults;
    const opt_faults = try optimalOffline(u32, testing.allocator, &requests, 3);

    const ratio = competitiveRatio(lru_faults, opt_faults);

    // LRU is at most k-competitive (k=3 here)
    try testing.expect(ratio <= 3.0);
    try testing.expect(ratio >= 1.0);
}

test "paging - competitive ratio: FIFO vs optimal" {
    const requests = [_]u32{ 1, 2, 3, 4, 1, 2, 5, 1, 2, 3 };

    var fifo = try FIFO(u32).init(testing.allocator, 3);
    defer fifo.deinit();

    for (requests) |page| {
        _ = try fifo.request(page);
    }

    const fifo_faults = fifo.getStats().page_faults;
    const opt_faults = try optimalOffline(u32, testing.allocator, &requests, 3);

    const ratio = competitiveRatio(fifo_faults, opt_faults);

    // FIFO is at most k-competitive (k=3 here)
    try testing.expect(ratio <= 3.0);
    try testing.expect(ratio >= 1.0);
}

test "paging - LRU vs FIFO comparison" {
    const requests = [_]u32{ 1, 2, 3, 1, 4, 5, 1, 2, 3, 4, 5 };

    var lru = try LRU(u32).init(testing.allocator, 3);
    defer lru.deinit();

    var fifo = try FIFO(u32).init(testing.allocator, 3);
    defer fifo.deinit();

    for (requests) |page| {
        _ = try lru.request(page);
        _ = try fifo.request(page);
    }

    // LRU typically performs better due to recency heuristic
    try testing.expect(lru.getStats().page_faults <= fifo.getStats().page_faults);
}

test "paging - memory safety" {
    var lru = try LRU(u32).init(testing.allocator, 100);
    defer lru.deinit();

    for (0..1000) |i| {
        _ = try lru.request(@intCast(i % 200));
    }

    var fifo = try FIFO(u32).init(testing.allocator, 100);
    defer fifo.deinit();

    for (0..1000) |i| {
        _ = try fifo.request(@intCast(i % 200));
    }
}
