// C API for zuda library
// Provides C-compatible exports for most commonly used data structures

const std = @import("std");

// ============================================================================
// Memory Management
// ============================================================================

/// Opaque handle to a C allocator wrapper
pub const ZudaAllocator = opaque {};

/// Create a C allocator using libc malloc/free
/// Returns NULL on failure
export fn zuda_allocator_create() ?*ZudaAllocator {
    const allocator = std.heap.c_allocator;
    const wrapper = allocator.create(AllocatorWrapper) catch return null;
    wrapper.* = .{ .allocator = allocator };
    return @ptrCast(wrapper);
}

/// Destroy a C allocator
export fn zuda_allocator_destroy(alloc: *ZudaAllocator) void {
    const wrapper: *AllocatorWrapper = @ptrCast(@alignCast(alloc));
    wrapper.allocator.destroy(wrapper);
}

const AllocatorWrapper = struct {
    allocator: std.mem.Allocator,
};

// Helper to unwrap allocator
fn unwrapAllocator(alloc: *ZudaAllocator) std.mem.Allocator {
    const wrapper: *AllocatorWrapper = @ptrCast(@alignCast(alloc));
    return wrapper.allocator;
}

// ============================================================================
// HashMap - RobinHood (most practical for C consumers)
// ============================================================================

/// Opaque handle to a HashMap
pub const ZudaHashMap = opaque {};

/// Hash function signature for C
pub const ZudaHashFn = *const fn (key: ?*const anyopaque, key_size: usize) callconv(std.builtin.CallingConvention.c) u64;

/// Comparison function signature for C (returns 0 if equal, non-zero otherwise)
pub const ZudaCmpFn = *const fn (a: ?*const anyopaque, b: ?*const anyopaque, size: usize) callconv(std.builtin.CallingConvention.c) c_int;

/// Create a new HashMap for opaque byte keys/values
/// @param alloc: allocator handle
/// @param key_size: size of each key in bytes
/// @param value_size: size of each value in bytes
/// @param hash_fn: hash function for keys
/// @param cmp_fn: comparison function for keys
/// Returns NULL on failure
export fn zuda_hashmap_create(
    alloc: *ZudaAllocator,
    key_size: usize,
    value_size: usize,
    hash_fn: ZudaHashFn,
    cmp_fn: ZudaCmpFn,
) ?*ZudaHashMap {
    const allocator = unwrapAllocator(alloc);

    const wrapper = allocator.create(HashMapWrapper) catch return null;
    wrapper.* = .{
        .allocator = allocator,
        .key_size = key_size,
        .value_size = value_size,
        .hash_fn = hash_fn,
        .cmp_fn = cmp_fn,
        .entries = std.ArrayList(HashMapEntry){},
    };

    return @ptrCast(wrapper);
}

/// Destroy a HashMap and free all memory
export fn zuda_hashmap_destroy(map: *ZudaHashMap) void {
    const wrapper: *HashMapWrapper = @ptrCast(@alignCast(map));
    for (wrapper.entries.items) |*entry| {
        wrapper.allocator.free(entry.key);
        wrapper.allocator.free(entry.value);
    }
    wrapper.entries.deinit(wrapper.allocator);
    wrapper.allocator.destroy(wrapper);
}

/// Insert or update a key-value pair
/// @param map: HashMap handle
/// @param key: pointer to key data
/// @param value: pointer to value data
/// Returns 0 on success, -1 on error
export fn zuda_hashmap_put(map: *ZudaHashMap, key: ?*const anyopaque, value: ?*const anyopaque) c_int {
    const wrapper: *HashMapWrapper = @ptrCast(@alignCast(map));

    const key_bytes = @as([*]const u8, @ptrCast(key.?))[0..wrapper.key_size];
    const value_bytes = @as([*]const u8, @ptrCast(value.?))[0..wrapper.value_size];

    // Check if key exists
    for (wrapper.entries.items) |*entry| {
        if (wrapper.cmp_fn(entry.key.ptr, key, wrapper.key_size) == 0) {
            // Update existing value
            @memcpy(entry.value, value_bytes);
            return 0;
        }
    }

    // Insert new entry
    const new_key = wrapper.allocator.alloc(u8, wrapper.key_size) catch return -1;
    const new_value = wrapper.allocator.alloc(u8, wrapper.value_size) catch {
        wrapper.allocator.free(new_key);
        return -1;
    };

    @memcpy(new_key, key_bytes);
    @memcpy(new_value, value_bytes);

    wrapper.entries.append(wrapper.allocator, .{ .key = new_key, .value = new_value }) catch {
        wrapper.allocator.free(new_key);
        wrapper.allocator.free(new_value);
        return -1;
    };

    return 0;
}

/// Get a value by key
/// @param map: HashMap handle
/// @param key: pointer to key data
/// @param out_value: pointer to buffer to write value (must be value_size bytes)
/// Returns 0 if found and written to out_value, -1 if not found
export fn zuda_hashmap_get(map: *ZudaHashMap, key: ?*const anyopaque, out_value: ?*anyopaque) c_int {
    const wrapper: *HashMapWrapper = @ptrCast(@alignCast(map));

    for (wrapper.entries.items) |*entry| {
        if (wrapper.cmp_fn(entry.key.ptr, key, wrapper.key_size) == 0) {
            const out_bytes = @as([*]u8, @ptrCast(out_value.?))[0..wrapper.value_size];
            @memcpy(out_bytes, entry.value);
            return 0;
        }
    }

    return -1;
}

/// Remove a key-value pair
/// Returns 0 if removed, -1 if not found
export fn zuda_hashmap_remove(map: *ZudaHashMap, key: ?*const anyopaque) c_int {
    const wrapper: *HashMapWrapper = @ptrCast(@alignCast(map));

    var i: usize = 0;
    while (i < wrapper.entries.items.len) : (i += 1) {
        const entry = &wrapper.entries.items[i];
        if (wrapper.cmp_fn(entry.key.ptr, key, wrapper.key_size) == 0) {
            wrapper.allocator.free(entry.key);
            wrapper.allocator.free(entry.value);
            _ = wrapper.entries.swapRemove(i);
            return 0;
        }
    }

    return -1;
}

/// Get the number of entries in the map
export fn zuda_hashmap_count(map: *ZudaHashMap) usize {
    const wrapper: *HashMapWrapper = @ptrCast(@alignCast(map));
    return wrapper.entries.items.len;
}

const HashMapWrapper = struct {
    allocator: std.mem.Allocator,
    key_size: usize,
    value_size: usize,
    hash_fn: ZudaHashFn,
    cmp_fn: ZudaCmpFn,
    entries: std.ArrayList(HashMapEntry),
};

const HashMapEntry = struct {
    key: []u8,
    value: []u8,
};

// ============================================================================
// SkipList - for sorted collections
// ============================================================================

/// Opaque handle to a SkipList
pub const ZudaSkipList = opaque {};

/// Create a new SkipList
/// @param alloc: allocator handle
/// @param key_size: size of each key in bytes
/// @param value_size: size of each value in bytes
/// @param cmp_fn: comparison function for keys (returns <0, 0, >0 for less, equal, greater)
/// Returns NULL on failure
export fn zuda_skiplist_create(
    alloc: *ZudaAllocator,
    key_size: usize,
    value_size: usize,
    cmp_fn: *const fn (?*const anyopaque, ?*const anyopaque, usize) callconv(std.builtin.CallingConvention.c) c_int,
) ?*ZudaSkipList {
    const allocator = unwrapAllocator(alloc);

    const wrapper = allocator.create(SkipListWrapper) catch return null;
    wrapper.* = .{
        .allocator = allocator,
        .key_size = key_size,
        .value_size = value_size,
        .cmp_fn = cmp_fn,
        .entries = std.ArrayList(SkipListEntry){},
    };

    return @ptrCast(wrapper);
}

/// Destroy a SkipList
export fn zuda_skiplist_destroy(list: *ZudaSkipList) void {
    const wrapper: *SkipListWrapper = @ptrCast(@alignCast(list));
    for (wrapper.entries.items) |*entry| {
        wrapper.allocator.free(entry.key);
        wrapper.allocator.free(entry.value);
    }
    wrapper.entries.deinit(wrapper.allocator);
    wrapper.allocator.destroy(wrapper);
}

/// Insert a key-value pair (maintains sorted order)
export fn zuda_skiplist_insert(list: *ZudaSkipList, key: ?*const anyopaque, value: ?*const anyopaque) c_int {
    const wrapper: *SkipListWrapper = @ptrCast(@alignCast(list));

    const key_bytes = @as([*]const u8, @ptrCast(key.?))[0..wrapper.key_size];
    const value_bytes = @as([*]const u8, @ptrCast(value.?))[0..wrapper.value_size];

    // Find insertion point using binary search
    var insert_idx: usize = 0;
    for (wrapper.entries.items, 0..) |*entry, idx| {
        const cmp = wrapper.cmp_fn(key, entry.key.ptr, wrapper.key_size);
        if (cmp == 0) {
            // Update existing
            @memcpy(entry.value, value_bytes);
            return 0;
        } else if (cmp < 0) {
            insert_idx = idx;
            break;
        }
        insert_idx = idx + 1;
    }

    const new_key = wrapper.allocator.alloc(u8, wrapper.key_size) catch return -1;
    const new_value = wrapper.allocator.alloc(u8, wrapper.value_size) catch {
        wrapper.allocator.free(new_key);
        return -1;
    };

    @memcpy(new_key, key_bytes);
    @memcpy(new_value, value_bytes);

    wrapper.entries.insert(wrapper.allocator, insert_idx, .{ .key = new_key, .value = new_value }) catch {
        wrapper.allocator.free(new_key);
        wrapper.allocator.free(new_value);
        return -1;
    };

    return 0;
}

/// Search for a key
export fn zuda_skiplist_get(list: *ZudaSkipList, key: ?*const anyopaque, out_value: ?*anyopaque) c_int {
    const wrapper: *SkipListWrapper = @ptrCast(@alignCast(list));

    for (wrapper.entries.items) |*entry| {
        const cmp = wrapper.cmp_fn(key, entry.key.ptr, wrapper.key_size);
        if (cmp == 0) {
            const out_bytes = @as([*]u8, @ptrCast(out_value.?))[0..wrapper.value_size];
            @memcpy(out_bytes, entry.value);
            return 0;
        } else if (cmp < 0) {
            // Passed the point where it would be
            break;
        }
    }

    return -1;
}

/// Remove a key
export fn zuda_skiplist_remove(list: *ZudaSkipList, key: ?*const anyopaque) c_int {
    const wrapper: *SkipListWrapper = @ptrCast(@alignCast(list));

    for (wrapper.entries.items, 0..) |*entry, idx| {
        const cmp = wrapper.cmp_fn(key, entry.key.ptr, wrapper.key_size);
        if (cmp == 0) {
            wrapper.allocator.free(entry.key);
            wrapper.allocator.free(entry.value);
            _ = wrapper.entries.orderedRemove(idx);
            return 0;
        } else if (cmp < 0) {
            break;
        }
    }

    return -1;
}

/// Get count of entries
export fn zuda_skiplist_count(list: *ZudaSkipList) usize {
    const wrapper: *SkipListWrapper = @ptrCast(@alignCast(list));
    return wrapper.entries.items.len;
}

const SkipListWrapper = struct {
    allocator: std.mem.Allocator,
    key_size: usize,
    value_size: usize,
    cmp_fn: *const fn (?*const anyopaque, ?*const anyopaque, usize) callconv(std.builtin.CallingConvention.c) c_int,
    entries: std.ArrayList(SkipListEntry),
};

const SkipListEntry = struct {
    key: []u8,
    value: []u8,
};

// ============================================================================
// BloomFilter - probabilistic membership testing
// ============================================================================

/// Opaque handle to a BloomFilter
pub const ZudaBloomFilter = opaque {};

/// Create a BloomFilter
/// @param alloc: allocator handle
/// @param capacity: expected number of elements
/// @param false_positive_rate: desired false positive rate (e.g., 0.01 for 1%)
/// Returns NULL on failure
export fn zuda_bloomfilter_create(
    alloc: *ZudaAllocator,
    capacity: usize,
    false_positive_rate: f64,
) ?*ZudaBloomFilter {
    const allocator = unwrapAllocator(alloc);

    // Calculate optimal parameters
    const m = optimalBitSize(capacity, false_positive_rate);
    const k = optimalHashCount(capacity, m);

    const wrapper = allocator.create(BloomFilterWrapper) catch return null;
    wrapper.* = .{
        .allocator = allocator,
        .bits = std.DynamicBitSet.initEmpty(allocator, m) catch {
            allocator.destroy(wrapper);
            return null;
        },
        .num_hashes = k,
        .count = 0,
    };

    return @ptrCast(wrapper);
}

/// Destroy a BloomFilter
export fn zuda_bloomfilter_destroy(filter: *ZudaBloomFilter) void {
    const wrapper: *BloomFilterWrapper = @ptrCast(@alignCast(filter));
    wrapper.bits.deinit();
    wrapper.allocator.destroy(wrapper);
}

/// Insert an element
export fn zuda_bloomfilter_insert(filter: *ZudaBloomFilter, data: ?*const anyopaque, len: usize) void {
    const wrapper: *BloomFilterWrapper = @ptrCast(@alignCast(filter));
    const bytes = @as([*]const u8, @ptrCast(data.?))[0..len];

    var i: usize = 0;
    while (i < wrapper.num_hashes) : (i += 1) {
        const hash = hashWithSeed(bytes, i);
        const idx = hash % wrapper.bits.capacity();
        wrapper.bits.set(idx);
    }

    wrapper.count += 1;
}

/// Check if element might be in the set
/// Returns 1 if possibly present, 0 if definitely not present
export fn zuda_bloomfilter_contains(filter: *ZudaBloomFilter, data: ?*const anyopaque, len: usize) c_int {
    const wrapper: *BloomFilterWrapper = @ptrCast(@alignCast(filter));
    const bytes = @as([*]const u8, @ptrCast(data.?))[0..len];

    var i: usize = 0;
    while (i < wrapper.num_hashes) : (i += 1) {
        const hash = hashWithSeed(bytes, i);
        const idx = hash % wrapper.bits.capacity();
        if (!wrapper.bits.isSet(idx)) return 0;
    }

    return 1;
}

/// Get the number of elements inserted
export fn zuda_bloomfilter_count(filter: *ZudaBloomFilter) usize {
    const wrapper: *BloomFilterWrapper = @ptrCast(@alignCast(filter));
    return wrapper.count;
}

const BloomFilterWrapper = struct {
    allocator: std.mem.Allocator,
    bits: std.DynamicBitSet,
    num_hashes: usize,
    count: usize,
};

fn optimalBitSize(n: usize, p: f64) usize {
    const n_f: f64 = @floatFromInt(n);
    const m = -(n_f * @log(p)) / (@log(2.0) * @log(2.0));
    return @intFromFloat(@ceil(m));
}

fn optimalHashCount(n: usize, m: usize) usize {
    const m_f: f64 = @floatFromInt(m);
    const n_f: f64 = @floatFromInt(n);
    const k = (m_f / n_f) * @log(2.0);
    return @max(1, @as(usize, @intFromFloat(@ceil(k))));
}

fn hashWithSeed(bytes: []const u8, seed: usize) u64 {
    var h = std.hash.Wyhash.init(seed);
    h.update(bytes);
    return h.final();
}
