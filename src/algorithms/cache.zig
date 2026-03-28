/// Cache Algorithms — Eviction Policies for Bounded Memory
///
/// Provides implementations of common cache replacement algorithms for managing
/// limited-capacity storage with different eviction strategies.
///
/// Categories:
/// - **LRU (Least Recently Used)**: Evicts least recently accessed item
///   - O(1) operations via HashMap + Doubly Linked List
///   - Best for: temporal locality workloads (OS page replacement, buffer pools)
///   - Use case: silica buffer_pool.zig (1237 LOC → zuda migration)
///
/// - **LFU (Least Frequently Used)**: Evicts least frequently accessed item
///   - O(1) operations via HashMap + frequency buckets
///   - Ties broken by LRU within same frequency
///   - Best for: hot/cold data distinction (CDN, query caching)
///   - Use case: zoltraak expiry logic (lazy + active eviction)
///
/// - **FIFO (First In First Out)**: Evicts oldest inserted item
///   - O(1) operations via HashMap + Singly Linked List
///   - Simpler than LRU (no reordering on access)
///   - Best for: scenarios where recency doesn't matter, only insertion order
///   - Use case: ring buffers, message queues
///
/// Algorithm Selection Guidelines:
/// - **Temporal locality** (recent accesses matter): Use LRU
/// - **Access frequency** (hot/cold patterns): Use LFU
/// - **Simplicity** (no reordering overhead): Use FIFO
/// - **Hybrid workloads**: Consider adaptive policies (future: ARC, 2Q)
///
/// Consumer Migrations:
/// - silica buffer_pool.zig (1237 LOC) → LRU (200-300 LOC reduction)
/// - zoltraak expiry (50 LOC) → LRU/LFU (standardization + 91+ tests)

pub const lru = @import("cache/lru.zig");
pub const lfu = @import("cache/lfu.zig");
pub const fifo = @import("cache/fifo.zig");

pub const LRU = lru.LRU;
pub const LFU = lfu.LFU;
pub const FIFO = fifo.FIFO;

test {
    @import("std").testing.refAllDecls(@This());
}
