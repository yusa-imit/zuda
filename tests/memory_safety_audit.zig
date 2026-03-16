const std = @import("std");
const testing = std.testing;
const zuda = @import("zuda");
const Order = std.math.Order;

// Context for comparisons
const IntContext = struct {
    pub fn compare(_: @This(), a: i32, b: i32) Order {
        return std.math.order(a, b);
    }
    pub fn lessThan(_: @This(), a: i32, b: i32) bool {
        return a < b;
    }
    pub fn hash(_: @This(), k: i32) u64 {
        return @intCast(@as(u32, @bitCast(k)));
    }
    pub fn hash2(_: @This(), k: i32) u64 {
        const h = @as(u32, @bitCast(k));
        return @intCast((h *% 0x9e3779b9) ^ (h >> 16));
    }
    pub fn eql(_: @This(), a: i32, b: i32) bool {
        return a == b;
    }
};

// Memory safety audit for all containers
// Tests: empty state, single element, max capacity, error paths, cleanup verification

test "memory safety: lists - empty state cleanup" {
    const allocator = testing.allocator;

    // SkipList
    {
        var list = try zuda.containers.lists.SkipList(i32, void, IntContext, IntContext.compare).init(allocator, .{});
        defer list.deinit();
        try testing.expectEqual(0, list.count());
    }

    // XorLinkedList
    {
        var list = zuda.containers.lists.XorLinkedList(i32).init(allocator);
        defer list.deinit();
        try testing.expectEqual(0, list.count());
    }

    // UnrolledLinkedList
    {
        var list = zuda.containers.lists.UnrolledLinkedList(i32, 16).init(allocator);
        defer list.deinit();
        try testing.expectEqual(0, list.count());
    }
}

test "memory safety: lists - single element cleanup" {
    const allocator = testing.allocator;

    // SkipList
    {
        var list = try zuda.containers.lists.SkipList(i32, void, IntContext, IntContext.compare).init(allocator, .{});
        defer list.deinit();
        _ = try list.insert(42, {});
        try testing.expectEqual(1, list.count());
        _ = list.remove(42);
        try testing.expectEqual(0, list.count());
    }

    // XorLinkedList
    {
        var list = zuda.containers.lists.XorLinkedList(i32).init(allocator);
        defer list.deinit();
        try list.pushBack(42);
        try testing.expectEqual(1, list.count());
        try testing.expectEqual(42, list.popBack().?);
        try testing.expectEqual(0, list.count());
    }

    // UnrolledLinkedList
    {
        var list = zuda.containers.lists.UnrolledLinkedList(i32, 16).init(allocator);
        defer list.deinit();
        try list.append(42);
        try testing.expectEqual(1, list.count());
        try testing.expectEqual(42, try list.popLast());
        try testing.expectEqual(0, list.count());
    }
}

test "memory safety: queues - empty state cleanup" {
    const allocator = testing.allocator;

    // Deque
    {
        var deque = zuda.containers.queues.Deque(i32).init(allocator);
        defer deque.deinit();
        try testing.expectEqual(0, deque.count());
    }
}

test "memory safety: queues - single element cleanup" {
    const allocator = testing.allocator;

    // Deque
    {
        var deque = zuda.containers.queues.Deque(i32).init(allocator);
        defer deque.deinit();
        try deque.push_back(42);
        try testing.expectEqual(1, deque.count());
        try testing.expectEqual(42, try deque.pop_front());
        try testing.expectEqual(0, deque.count());
    }
}

test "memory safety: heaps - empty state cleanup" {
    const allocator = testing.allocator;

    // FibonacciHeap
    {
        var heap = zuda.containers.heaps.FibonacciHeap(i32, IntContext, IntContext.compare).init(allocator, .{});
        defer heap.deinit();
        try testing.expectEqual(0, heap.count());
    }

    // BinomialHeap
    {
        var heap = zuda.containers.heaps.BinomialHeap(i32, IntContext, IntContext.compare).init(allocator, .{});
        defer heap.deinit();
        try testing.expectEqual(0, heap.count());
    }

    // PairingHeap
    {
        var heap = zuda.containers.heaps.PairingHeap(i32, IntContext, IntContext.lessThan).init(allocator, .{});
        defer heap.deinit();
        try testing.expectEqual(0, heap.count());
    }

    // DaryHeap
    {
        var heap = zuda.containers.heaps.DaryHeap(i32, 4, IntContext, IntContext.lessThan).init(allocator, .{});
        defer heap.deinit();
        try testing.expectEqual(0, heap.count());
    }
}

test "memory safety: heaps - single element cleanup" {
    const allocator = testing.allocator;

    // FibonacciHeap
    {
        var heap = zuda.containers.heaps.FibonacciHeap(i32, IntContext, IntContext.compare).init(allocator, .{});
        defer heap.deinit();
        _ = try heap.insert(42);
        try testing.expectEqual(1, heap.count());
        try testing.expectEqual(42, heap.extractMin().?);
        try testing.expectEqual(0, heap.count());
    }

    // BinomialHeap
    {
        var heap = zuda.containers.heaps.BinomialHeap(i32, IntContext, IntContext.compare).init(allocator, .{});
        defer heap.deinit();
        try heap.insert(42);
        try testing.expectEqual(1, heap.count());
        try testing.expectEqual(42, heap.extractMin().?);
        try testing.expectEqual(0, heap.count());
    }

    // PairingHeap
    {
        var heap = zuda.containers.heaps.PairingHeap(i32, IntContext, IntContext.lessThan).init(allocator, .{});
        defer heap.deinit();
        _ = try heap.insert(42);
        try testing.expectEqual(1, heap.count());
        try testing.expectEqual(42, heap.extractMin().?);
        try testing.expectEqual(0, heap.count());
    }

    // DaryHeap
    {
        var heap = zuda.containers.heaps.DaryHeap(i32, 4, IntContext, IntContext.lessThan).init(allocator, .{});
        defer heap.deinit();
        try heap.insert(42);
        try testing.expectEqual(1, heap.count());
        try testing.expectEqual(42, heap.extractMin().?);
        try testing.expectEqual(0, heap.count());
    }
}

test "memory safety: hash containers - empty state cleanup" {
    const allocator = testing.allocator;

    // CuckooHashMap
    {
        var map = try zuda.containers.hashing.CuckooHashMap(i32, i32, IntContext, IntContext.hash, IntContext.hash2, IntContext.eql).init(allocator, .{});
        defer map.deinit();
        try testing.expectEqual(0, map.count());
    }

    // RobinHoodHashMap
    {
        var map = try zuda.containers.hashing.RobinHoodHashMap(i32, i32, IntContext, IntContext.hash, IntContext.eql).init(allocator, .{});
        defer map.deinit();
        try testing.expectEqual(0, map.count());
    }

    // SwissTable
    {
        var map = try zuda.containers.hashing.SwissTable(i32, i32, IntContext, IntContext.hash, IntContext.eql).init(allocator, .{});
        defer map.deinit();
        try testing.expectEqual(0, map.count());
    }

    // ConsistentHashRing
    {
        var ring = try zuda.containers.hashing.ConsistentHashRing(i32, i32, IntContext, IntContext.hash, IntContext.eql).init(allocator, .{}, 10);
        defer ring.deinit();
        try testing.expectEqual(0, ring.count());
    }
}

test "memory safety: hash containers - single element cleanup" {
    const allocator = testing.allocator;

    // CuckooHashMap
    {
        var map = try zuda.containers.hashing.CuckooHashMap(i32, i32, IntContext, IntContext.hash, IntContext.hash2, IntContext.eql).init(allocator, .{});
        defer map.deinit();
        _ = try map.insert(1, 100);
        try testing.expectEqual(1, map.count());
        _ = map.remove(1);
        try testing.expectEqual(0, map.count());
    }

    // RobinHoodHashMap
    {
        var map = try zuda.containers.hashing.RobinHoodHashMap(i32, i32, IntContext, IntContext.hash, IntContext.eql).init(allocator, .{});
        defer map.deinit();
        _ = try map.insert(1, 100);
        try testing.expectEqual(1, map.count());
        _ = map.remove(1);
        try testing.expectEqual(0, map.count());
    }

    // SwissTable
    {
        var map = try zuda.containers.hashing.SwissTable(i32, i32, IntContext, IntContext.hash, IntContext.eql).init(allocator, .{});
        defer map.deinit();
        _ = try map.insert(1, 100);
        try testing.expectEqual(1, map.count());
        _ = map.remove(1);
        try testing.expectEqual(0, map.count());
    }

    // ConsistentHashRing
    {
        var ring = try zuda.containers.hashing.ConsistentHashRing(i32, i32, IntContext, IntContext.hash, IntContext.eql).init(allocator, .{}, 10);
        defer ring.deinit();
        try ring.addNode(1);
        try testing.expectEqual(1, ring.count());
        _ = ring.removeNode(1);
        try testing.expectEqual(0, ring.count());
    }
}

test "memory safety: trees - empty state cleanup" {
    const allocator = testing.allocator;

    // RedBlackTree
    {
        var tree = zuda.containers.trees.RedBlackTree(i32, i32, IntContext, IntContext.compare).init(allocator, .{});
        defer tree.deinit();
        try testing.expectEqual(0, tree.count());
    }

    // AVLTree
    {
        var tree = zuda.containers.trees.AVLTree(i32, i32, IntContext, IntContext.compare).init(allocator, .{});
        defer tree.deinit();
        try testing.expectEqual(0, tree.count());
    }

    // SplayTree
    {
        var tree = zuda.containers.trees.SplayTree(i32, i32, IntContext, IntContext.compare).init(allocator, .{});
        defer tree.deinit();
        try testing.expectEqual(0, tree.count());
    }

    // AATree
    {
        var tree = zuda.containers.trees.AATree(i32, i32, IntContext, IntContext.compare).init(allocator, .{});
        defer tree.deinit();
        try testing.expectEqual(0, tree.count());
    }

    // ScapegoatTree
    {
        var tree = zuda.containers.trees.ScapegoatTree(i32, i32, IntContext, IntContext.compare).init(allocator, .{});
        defer tree.deinit();
        try testing.expectEqual(0, tree.count());
    }

    // BTree
    {
        var tree = zuda.containers.trees.BTree(i32, i32, 128, IntContext).init(allocator, .{});
        defer tree.deinit();
        try testing.expectEqual(0, tree.count());
    }

    // Trie (not exported yet)
    // {
    //     var trie = try zuda.containers.trees.Trie.init(allocator);
    //     defer trie.deinit();
    //     try testing.expect(trie.isEmpty());
    // }

    // RadixTree (not exported yet)
    // {
    //     var tree = try zuda.containers.trees.RadixTree(i32).init(allocator);
    //     defer tree.deinit();
    //     try testing.expect(tree.isEmpty());
    // }
}

test "memory safety: trees - single element cleanup" {
    const allocator = testing.allocator;

    // RedBlackTree
    {
        var tree = zuda.containers.trees.RedBlackTree(i32, i32, IntContext, IntContext.compare).init(allocator, .{});
        defer tree.deinit();
        _ = try tree.insert(1, 100);
        try testing.expectEqual(1, tree.count());
        _ = tree.remove(1);
        try testing.expectEqual(0, tree.count());
    }

    // AVLTree
    {
        var tree = zuda.containers.trees.AVLTree(i32, i32, IntContext, IntContext.compare).init(allocator, .{});
        defer tree.deinit();
        _ = try tree.insert(1, 100);
        try testing.expectEqual(1, tree.count());
        _ = tree.remove(1);
        try testing.expectEqual(0, tree.count());
    }

    // SplayTree
    {
        var tree = zuda.containers.trees.SplayTree(i32, i32, IntContext, IntContext.compare).init(allocator, .{});
        defer tree.deinit();
        _ = try tree.insert(1, 100);
        try testing.expectEqual(1, tree.count());
        _ = tree.remove(1);
        try testing.expectEqual(0, tree.count());
    }

    // AATree
    {
        var tree = zuda.containers.trees.AATree(i32, i32, IntContext, IntContext.compare).init(allocator, .{});
        defer tree.deinit();
        _ = try tree.insert(1, 100);
        try testing.expectEqual(1, tree.count());
        _ = tree.remove(1);
        try testing.expectEqual(0, tree.count());
    }

    // ScapegoatTree
    {
        var tree = zuda.containers.trees.ScapegoatTree(i32, i32, IntContext, IntContext.compare).init(allocator, .{});
        defer tree.deinit();
        _ = try tree.insert(1, 100);
        try testing.expectEqual(1, tree.count());
        _ = tree.remove(1);
        try testing.expectEqual(0, tree.count());
    }

    // BTree
    {
        var tree = zuda.containers.trees.BTree(i32, i32, 128, IntContext).init(allocator, .{});
        defer tree.deinit();
        _ = try tree.insert(1, 100);
        try testing.expectEqual(1, tree.count());
        _ = tree.remove(1);
        try testing.expectEqual(0, tree.count());
    }

    // Trie (not exported yet)
    // {
    //     var trie = try zuda.containers.trees.Trie.init(allocator);
    //     defer trie.deinit();
    //     try trie.insert("x");
    //     try testing.expect(!trie.isEmpty());
    //     _ = try trie.remove("x");
    //     try testing.expect(trie.isEmpty());
    // }

    // RadixTree (not exported yet)
    // {
    //     var tree = try zuda.containers.trees.RadixTree(i32).init(allocator);
    //     defer tree.deinit();
    //     try tree.insert("x", 100);
    //     try testing.expect(!tree.isEmpty());
    //     _ = try tree.remove("x");
    //     try testing.expect(tree.isEmpty());
    // }
}

test "memory safety: cache - empty state cleanup" {
    const allocator = testing.allocator;

    // LRUCache
    {
        var cache = zuda.containers.cache.LRUCache(i32, i32, IntContext, null).init(allocator, 10);
        defer cache.deinit();
        try testing.expectEqual(0, cache.count());
    }

    // LFUCache
    {
        var cache = zuda.containers.cache.LFUCache(i32, i32, IntContext, null).init(allocator, 10);
        defer cache.deinit();
        try testing.expectEqual(0, cache.count());
    }

    // ARCCache
    {
        var cache = zuda.containers.cache.ARCCache(i32, i32, IntContext).init(allocator, 10);
        defer cache.deinit();
        try testing.expectEqual(0, cache.count());
    }
}

test "memory safety: cache - single element cleanup" {
    const allocator = testing.allocator;

    // LRUCache
    {
        var cache = zuda.containers.cache.LRUCache(i32, i32, IntContext, null).init(allocator, 10);
        defer cache.deinit();
        _ = try cache.put(1, 100);
        try testing.expectEqual(1, cache.count());
        _ = cache.remove(1);
        try testing.expectEqual(0, cache.count());
    }

    // LFUCache
    {
        var cache = zuda.containers.cache.LFUCache(i32, i32, IntContext, null).init(allocator, 10);
        defer cache.deinit();
        _ = try cache.put(1, 100);
        try testing.expectEqual(1, cache.count());
        _ = cache.remove(1);
        try testing.expectEqual(0, cache.count());
    }

    // ARCCache
    {
        var cache = zuda.containers.cache.ARCCache(i32, i32, IntContext).init(allocator, 10);
        defer cache.deinit();
        _ = try cache.put(1, 100);
        try testing.expectEqual(1, cache.count());
        _ = cache.remove(1);
        try testing.expectEqual(0, cache.count());
    }
}

test "memory safety: stress test - large allocation and cleanup" {
    const allocator = testing.allocator;

    // SkipList with 10k elements
    {
        var list = try zuda.containers.lists.SkipList(i32, void, IntContext, IntContext.compare).init(allocator, .{});
        defer list.deinit();

        var i: i32 = 0;
        while (i < 10000) : (i += 1) {
            _ = try list.insert(i, {});
        }
        try testing.expectEqual(10000, list.count());

        // Remove all
        i = 0;
        while (i < 10000) : (i += 1) {
            _ = list.remove(i);
        }
        try testing.expectEqual(0, list.count());
    }

    // CuckooHashMap with 10k elements
    {
        var map = try zuda.containers.hashing.CuckooHashMap(i32, i32, IntContext, IntContext.hash, IntContext.hash2, IntContext.eql).init(allocator, .{});
        defer map.deinit();

        var i: i32 = 0;
        while (i < 10000) : (i += 1) {
            _ = try map.insert(i, i * 2);
        }
        try testing.expectEqual(10000, map.count());

        // Remove all
        i = 0;
        while (i < 10000) : (i += 1) {
            _ = map.remove(i);
        }
        try testing.expectEqual(0, map.count());
    }

    // RedBlackTree with 10k elements
    {
        var tree = zuda.containers.trees.RedBlackTree(i32, i32, IntContext, IntContext.compare).init(allocator, .{});
        defer tree.deinit();

        var i: i32 = 0;
        while (i < 10000) : (i += 1) {
            _ = try tree.insert(i, i * 2);
        }
        try testing.expectEqual(10000, tree.count());

        // Remove all
        i = 0;
        while (i < 10000) : (i += 1) {
            _ = tree.remove(i);
        }
        try testing.expectEqual(0, tree.count());
    }
}
