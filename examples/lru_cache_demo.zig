/// LRU Cache API Demo
///
/// Demonstrates practical usage of LRUCache from zuda's cache module.
/// LRUCache provides O(1) get/put/remove with automatic eviction of
/// least-recently-used entries when capacity is exceeded.
///
/// Use cases:
///   - Web request caching (URL → HTML response)
///   - DNS lookup cache (domain → IP address)
///   - Database buffer pool (page_id → page_data)
///   - Computation memoization (args → result)
///
/// Run: zig build example-lru-cache

const std = @import("std");
const zuda = @import("zuda");

// Example 1: Simple string cache (web request cache simulation)
fn webRequestCacheDemo() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Example 1: Web Request Cache ===\n", .{});

    // Create LRU cache with capacity 3
    // StringContext provides hash/equality for []const u8 keys
    var cache = zuda.containers.cache.LRUCache(
        []const u8, // Key type: URL
        []const u8, // Value type: HTML response
        std.hash_map.StringContext,
        null, // No eviction callback
    ).init(allocator, 3);
    defer cache.deinit();

    // Simulate caching web responses
    std.debug.print("Caching 3 URLs (capacity=3):\n", .{});
    _ = try cache.put("/home", "Homepage HTML");
    _ = try cache.put("/about", "About page HTML");
    _ = try cache.put("/contact", "Contact page HTML");
    std.debug.print("  Cache size: {}\n", .{cache.count()});

    // Access a cached entry (moves it to MRU position)
    if (cache.get("/home")) |html| {
        std.debug.print("  GET /home → {s}\n", .{html});
    }

    // Add 4th entry → triggers eviction of LRU entry (/about)
    std.debug.print("\nAdding 4th entry (exceeds capacity):\n", .{});
    _ = try cache.put("/products", "Products page HTML");
    std.debug.print("  Cache size: {}\n", .{cache.count()});

    // Check which entries remain
    std.debug.print("\nRemaining entries (MRU → LRU order):\n", .{});
    var it = cache.iterator();
    while (it.next()) |entry| {
        std.debug.print("  {s} → {s}\n", .{ entry.key, entry.value });
    }

    // Verify eviction: /about should be gone
    std.debug.print("\nVerify eviction:\n", .{});
    const about_result = cache.get("/about");
    std.debug.print("  GET /about → {s}\n", .{if (about_result) |_| "FOUND" else "EVICTED"});
}

// Example 2: Integer computation cache with eviction callback
var eviction_count: usize = 0;

fn evictionCallback(key: u32, value: u64) void {
    std.debug.print("  [EVICTED] fib({}) = {}\n", .{ key, value });
    eviction_count += 1;
}

fn fibonacciCacheDemo() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Example 2: Fibonacci Memoization with Eviction Callback ===\n", .{});

    // Create cache with eviction callback
    var cache = zuda.containers.cache.LRUCache(
        u32, // Key: n
        u64, // Value: fib(n)
        std.hash_map.AutoContext(u32),
        evictionCallback, // Called when entry is evicted
    ).init(allocator, 5);
    defer cache.deinit();

    std.debug.print("Computing fib(n) with LRU cache (capacity=5):\n", .{});

    // Compute and cache fib(0) to fib(4)
    var n: u32 = 0;
    while (n <= 4) : (n += 1) {
        const result = try computeFibWithCache(&cache, n);
        std.debug.print("  fib({}) = {}\n", .{ n, result });
    }

    std.debug.print("\nCache state: {} entries\n", .{cache.count()});

    // Add fib(5) and fib(6) → triggers evictions
    std.debug.print("\nComputing fib(5) and fib(6) (triggers evictions):\n", .{});
    _ = try computeFibWithCache(&cache, 5);
    _ = try computeFibWithCache(&cache, 6);

    std.debug.print("\nTotal evictions: {}\n", .{eviction_count});
}

fn computeFibWithCache(cache: anytype, n: u32) !u64 {
    // Check cache first
    if (cache.get(n)) |result| {
        return result;
    }

    // Compute and cache
    const result = if (n <= 1) n else blk: {
        const a = try computeFibWithCache(cache, n - 1);
        const b = try computeFibWithCache(cache, n - 2);
        break :blk a + b;
    };

    _ = try cache.put(n, result);
    return result;
}

// Example 3: Custom hash context for struct keys
const User = struct {
    id: u32,
    name: []const u8,

    const HashContext = struct {
        pub fn hash(_: @This(), key: User) u64 {
            return std.hash.Wyhash.hash(0, std.mem.asBytes(&key.id));
        }

        pub fn eql(_: @This(), a: User, b: User) bool {
            return a.id == b.id;
        }
    };
};

fn customContextDemo() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Example 3: Custom Hash Context (User Cache) ===\n", .{});

    // Cache with custom hash context for struct keys
    var cache = zuda.containers.cache.LRUCache(
        User,
        []const u8,
        User.HashContext,
        null,
    ).init(allocator, 3);
    defer cache.deinit();

    std.debug.print("Caching user profiles:\n", .{});
    _ = try cache.put(.{ .id = 1, .name = "Alice" }, "Profile: Software Engineer");
    _ = try cache.put(.{ .id = 2, .name = "Bob" }, "Profile: Data Scientist");
    _ = try cache.put(.{ .id = 3, .name = "Carol" }, "Profile: DevOps Engineer");

    // Lookup by user
    const alice = User{ .id = 1, .name = "Alice" };
    if (cache.get(alice)) |profile| {
        std.debug.print("  User {} → {s}\n", .{ alice.id, profile });
    }

    std.debug.print("\nCache operations:\n", .{});
    std.debug.print("  count(): {}\n", .{cache.count()});
    std.debug.print("  isEmpty(): {}\n", .{cache.isEmpty()});

    // Remove entry
    const removed = cache.remove(alice);
    std.debug.print("  remove(User 1): {?s}\n", .{removed});
    std.debug.print("  count(): {}\n", .{cache.count()});
}

// Example 4: Buffer pool simulation (silica use case)
fn bufferPoolDemo() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Example 4: Buffer Pool Simulation ===\n", .{});
    std.debug.print("(Simulates silica's 1237-line buffer pool with zuda LRUCache)\n\n", .{});

    // page_id → page_data cache
    var buffer_pool = zuda.containers.cache.LRUCache(
        u64, // page_id
        [4096]u8, // page data (4KB pages)
        std.hash_map.AutoContext(u64),
        null,
    ).init(allocator, 4); // Small buffer pool: 4 pages = 16KB
    defer buffer_pool.deinit();

    std.debug.print("Buffer pool capacity: 4 pages (16KB)\n", .{});

    // Load pages into buffer pool
    std.debug.print("\nLoading pages:\n", .{});
    var page_id: u64 = 1;
    while (page_id <= 5) : (page_id += 1) {
        var page_data: [4096]u8 = undefined;
        @memset(&page_data, @intCast(page_id)); // Simulate page content
        _ = try buffer_pool.put(page_id, page_data);
        std.debug.print("  Loaded page {} (buffer size: {})\n", .{ page_id, buffer_pool.count() });
    }

    // Access pattern: 2, 3, 4 (mark as recently used)
    std.debug.print("\nAccess pattern: pages 2, 3, 4\n", .{});
    _ = buffer_pool.get(2);
    _ = buffer_pool.get(3);
    _ = buffer_pool.get(4);

    // Check which page was evicted (should be page 1)
    std.debug.print("\nVerify LRU eviction:\n", .{});
    std.debug.print("  Page 1 in buffer: {}\n", .{buffer_pool.get(1) != null});
    std.debug.print("  Page 5 in buffer: {}\n", .{buffer_pool.get(5) != null});
}

pub fn main() !void {
    try webRequestCacheDemo();
    try fibonacciCacheDemo();
    try customContextDemo();
    try bufferPoolDemo();

    std.debug.print("\n=== API Summary ===\n", .{});
    std.debug.print("LRUCache operations (all O(1)):\n", .{});
    std.debug.print("  - init(allocator, capacity) — Create cache\n", .{});
    std.debug.print("  - deinit() — Free resources\n", .{});
    std.debug.print("  - put(key, value) — Insert/update (evicts LRU if full)\n", .{});
    std.debug.print("  - get(key) — Retrieve value (marks as MRU)\n", .{});
    std.debug.print("  - remove(key) — Delete entry\n", .{});
    std.debug.print("  - count() — Number of entries\n", .{});
    std.debug.print("  - isEmpty() — Check if empty\n", .{});
    std.debug.print("  - iterator() — Iterate MRU → LRU order\n", .{});
    std.debug.print("\nFor full API details: src/containers/cache/lru_cache.zig\n", .{});
}
